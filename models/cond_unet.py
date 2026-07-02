"""Conditional denoising U-Net (epsilon_theta).

Implements the denoiser of Eq. (3)/(5):

    epsilon_theta(x_t, t, E_mask, E_text)

The U-Net is conditioned on anatomical priors through the Dense Hint Input
(DHI) module, whose dense embedding ``E_mask`` is concatenated with the noisy
latent ``z_t`` at the input and whose multi-scale hints are injected at every
encoder/decoder stage. Semantic guidance is injected by the Knowledge-Guided
Refinement Module (KGRM), which refines the feature stream at every stage using
the multi-granular text embedding.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dhi import DenseHintInput
from .kgrm import KnowledgeGuidedRefinementModule
from .unet_blocks import DownBlock, UpBlock, MidBlock, timestep_embedding


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        latent_channels: int = 4,
        model_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 4),
        num_heads: int = 8,
        head_dim: int = 64,
        mask_in_channels: int = 1,
        text_dim: int = 768,
        hint_channels: int = 32,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        time_dim = model_channels * 4

        self.dhi = DenseHintInput(
            in_channels=mask_in_channels,
            base_channels=32,
            latent_channels=latent_channels,
            num_resolutions=len(channel_mults),
            hint_channels=hint_channels,
        )

        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_conv = nn.Conv2d(latent_channels * 2, model_channels, kernel_size=3, padding=1)

        enc_chs = [model_channels * m for m in channel_mults]
        self.encoders = nn.ModuleList()
        self.encoder_kgrms = nn.ModuleList()
        self.hint_adapters = nn.ModuleList()
        in_ch = model_channels
        for i, out_ch in enumerate(enc_chs):
            self.encoders.append(DownBlock(in_ch, out_ch, time_dim, use_attn=True))
            self.encoder_kgrms.append(KnowledgeGuidedRefinementModule(out_ch, text_dim, num_heads, head_dim))
            self.hint_adapters.append(self._zero_conv(hint_channels, out_ch))
            in_ch = out_ch

        self.mid = MidBlock(enc_chs[-1], time_dim)
        self.mid_kgrm = KnowledgeGuidedRefinementModule(enc_chs[-1], text_dim, num_heads, head_dim)

        self.decoders = nn.ModuleList()
        self.decoder_kgrms = nn.ModuleList()
        rev = list(reversed(enc_chs))
        in_ch = enc_chs[-1]
        for i, out_ch in enumerate(rev):
            self.decoders.append(UpBlock(in_ch, out_ch, time_dim, use_attn=True))
            self.decoder_kgrms.append(KnowledgeGuidedRefinementModule(out_ch, text_dim, num_heads, head_dim))
            in_ch = out_ch

        self.out = nn.Sequential(
            nn.GroupNorm(min(8, enc_chs[0]), enc_chs[0]),
            nn.SiLU(),
            nn.Conv2d(enc_chs[0], latent_channels, kernel_size=3, padding=1),
        )

    @staticmethod
    def _zero_conv(in_ch: int, out_ch: int) -> nn.Module:
        m = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        nn.init.zeros_(m.weight)
        nn.init.zeros_(m.bias)
        return m

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor,
        text_tokens: torch.Tensor,
        text_global: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        e_mask = self.dhi(mask)
        hints: List[torch.Tensor] = self.dhi.multi_scale(mask)

        x = self.input_conv(torch.cat([z_t, e_mask], dim=1))

        t_emb = self.time_embed(timestep_embedding(t, self.time_embed[0].in_features))

        skips: List[torch.Tensor] = []
        for i, (enc, kgrm, adapt) in enumerate(zip(self.encoders, self.encoder_kgrms, self.hint_adapters)):
            hint = hints[i] if i < len(hints) else None
            if hint is not None:
                hint = F.interpolate(hint, size=x.shape[2:], mode="nearest")
                x = x + adapt(hint)
            x, skip = enc(x, t_emb)
            x = kgrm(x, text_tokens, text_global)
            skips.append(skip)

        x = self.mid(x, t_emb)
        x = self.mid_kgrm(x, text_tokens, text_global)

        for dec, kgrm in zip(self.decoders, self.decoder_kgrms):
            skip = skips.pop()
            x = dec(x, skip, t_emb)
            x = kgrm(x, text_tokens, text_global)

        return self.out(x)
