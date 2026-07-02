"""Dense Hint Input (DHI) module.

Implements Eq. (4) of the paper:

    E_mask = f_phi(M; theta_mask) in R^{H' x W' x C_emb}

A lightweight convolutional encoder with residual connections and multi-scale
feature extraction, used to embed anatomical priors (segmentation masks) into
the latent space so they can condition the denoising U-Net.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, groups=groups)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, groups=groups)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.silu(h + self.skip(x))


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DenseHintInput(nn.Module):
    """Dense Hint Input encoder.

    Maps a segmentation mask ``M`` to a dense anatomical embedding ``E_mask`` at
    the latent resolution, and additionally exposes a hierarchy of multi-scale
    feature maps for injection at every stage of the conditional U-Net.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        latent_channels: int = 4,
        num_resolutions: int = 3,
        hint_channels: int = 32,
    ):
        super().__init__()
        self.num_resolutions = num_resolutions
        self.hint_channels = hint_channels

        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.res_blocks = nn.ModuleList()
        self.hint_projs = nn.ModuleList()
        ch = base_channels
        for i in range(num_resolutions):
            self.res_blocks.append(ResidualConvBlock(ch, ch))
            self.hint_projs.append(nn.Conv2d(ch, hint_channels, kernel_size=1))
            self.down_blocks.append(Downsample(ch))
            ch = min(ch * 2, 128)

        self.res_blocks.append(ResidualConvBlock(ch, ch))
        self.hint_projs.append(nn.Conv2d(ch, hint_channels, kernel_size=1))

        self.out_conv = nn.Conv2d(ch, latent_channels, kernel_size=1)

        self._out_channels = ch
        self.latent_channels = latent_channels

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """Return the dense anatomical embedding ``E_mask`` (B, C_emb, H', W')."""
        h = self.init_conv(mask)
        for res, down in zip(self.res_blocks, self.down_blocks):
            h = res(h)
            h = down(h)
        h = self.res_blocks[-1](h)
        return self.out_conv(h)

    def multi_scale(self, mask: torch.Tensor) -> List[torch.Tensor]:
        """Return per-resolution feature maps (each ``hint_channels`` wide) for
        hierarchical injection at every stage of the conditional U-Net."""
        feats: List[torch.Tensor] = []
        h = self.init_conv(mask)
        for res, proj, down in zip(self.res_blocks, self.hint_projs, self.down_blocks):
            h = res(h)
            feats.append(proj(h))
            h = down(h)
        h = self.res_blocks[-1](h)
        feats.append(self.hint_projs[-1](h))
        return feats
