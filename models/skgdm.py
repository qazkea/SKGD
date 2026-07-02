"""Structural Knowledge Guided Diffusion Model (SKGDM).

The full framework (Eq. 3, 5, 8):

    z = E(x)                                   (VAE encode to latent)
    z_t = sqrt(alpha_bar_t) z + sqrt(1 - alpha_bar_t) eps   (forward diffusion)
    h_t = Phi_UNet(z_t, t | E_mask; theta)     (anatomy-aware conditioning)
    eps_pred = eps_theta(z_t, t, E_mask, F_out)             (denoiser)
    L_total = E || eps - eps_pred ||^2          (training objective)

Frozen: VAE image encoder, MedCLIP text encoder.
Trainable: DHI encoder, KGRM, conditional U-Net, and the text projection W_proj.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cond_unet import ConditionalUNet
from .scheduler import NoiseScheduler
from .text_encoder import TextEncoderWrapper
from .vae import VAEWrapper


@dataclass
class SKGDMConfig:
    latent_channels: int = 4
    model_channels: int = 64
    channel_mults: Tuple[int, ...] = (1, 2, 4, 4)
    num_heads: int = 8
    head_dim: int = 64
    mask_in_channels: int = 1
    text_dim: int = 768
    hint_channels: int = 32
    image_size: int = 256
    vae_scale_factor: float = 0.18215


class SKGDM(nn.Module):
    def __init__(
        self,
        vae: VAEWrapper,
        text_encoder: TextEncoderWrapper,
        unet: Optional[ConditionalUNet] = None,
        scheduler: Optional[NoiseScheduler] = None,
        config: Optional[SKGDMConfig] = None,
    ):
        super().__init__()
        self.config = config or SKGDMConfig(text_dim=text_encoder.proj_dim)
        self.vae = vae
        self.text_encoder = text_encoder
        self.unet = unet or ConditionalUNet(
            latent_channels=self.config.latent_channels,
            model_channels=self.config.model_channels,
            channel_mults=self.config.channel_mults,
            num_heads=self.config.num_heads,
            head_dim=self.config.head_dim,
            mask_in_channels=self.config.mask_in_channels,
            text_dim=self.config.text_dim,
            hint_channels=self.config.hint_channels,
        )
        self.scheduler = scheduler or NoiseScheduler()

    def trainable_parameters(self):
        params = list(self.unet.parameters())
        params += [p for p in self.text_encoder.parameters() if p.requires_grad]
        return params

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.text_encoder(input_ids, attention_mask)

    def compute_loss(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            z0 = self.vae.encode(image)

        noise = torch.randn_like(z0)
        t = self.scheduler.training_timesteps(z0.shape[0], z0.device)
        z_t = self.scheduler.add_noise(z0, noise, t)

        text_tokens, text_global = self.encode_text(input_ids, attention_mask)
        noise_pred = self.unet(z_t, t, mask, text_tokens, text_global)

        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(
        self,
        mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_tokens, text_global = self.encode_text(input_ids, attention_mask)

        latent_h = mask.shape[-1] // 8
        latent_w = mask.shape[-1] // 8
        z = torch.randn(
            (mask.shape[0], self.config.latent_channels, latent_h, latent_w),
            device=mask.device,
        )

        for t in self.scheduler.inference_timesteps:
            t_batch = torch.full((z.shape[0],), int(t), device=z.device, dtype=torch.long)
            noise_pred = self.unet(z, t_batch, mask, text_tokens, text_global)
            z = self.scheduler.ddim_step(noise_pred, t_batch, z)

        return self.vae.decode(z)
