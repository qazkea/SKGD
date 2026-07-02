"""Building blocks for the conditional denoising U-Net."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8):
        super().__init__()
        g = min(groups, in_ch)
        self.norm1 = nn.GroupNorm(g, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        g2 = min(groups, out_ch)
        self.norm2 = nn.GroupNorm(g2, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8, head_dim: int = 64):
        super().__init__()
        inner = num_heads * head_dim
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.q = nn.Conv2d(channels, inner, 1)
        self.k = nn.Conv2d(channels, inner, 1)
        self.v = nn.Conv2d(channels, inner, 1)
        self.out = nn.Conv2d(inner, channels, 1)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        r = x
        x = self.norm(x)
        q = self.q(x).reshape(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
        k = self.k(x).reshape(b, self.num_heads, self.head_dim, h * w)
        v = self.v(x).reshape(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0, 1, 3, 2).reshape(b, -1, h, w)
        return r + self.out(out)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, use_attn: bool = False):
        super().__init__()
        self.res1 = ResBlock(in_ch, out_ch, time_dim)
        self.res2 = ResBlock(out_ch, out_ch, time_dim)
        self.attn = SelfAttention(out_ch) if use_attn else nn.Identity()
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb):
        x = self.res1(x, t_emb)
        x = self.attn(self.res2(x, t_emb))
        return self.down(x), x


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, use_attn: bool = False):
        super().__init__()
        self.res1 = ResBlock(in_ch + out_ch, out_ch, time_dim)
        self.res2 = ResBlock(out_ch, out_ch, time_dim)
        self.attn = SelfAttention(out_ch) if use_attn else nn.Identity()

    def forward(self, x, skip, t_emb):
        x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        return self.attn(self.res2(self.res1(x, t_emb), t_emb))


class MidBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.res1 = ResBlock(channels, channels, time_dim)
        self.attn = SelfAttention(channels)
        self.res2 = ResBlock(channels, channels, time_dim)

    def forward(self, x, t_emb):
        return self.res2(self.attn(self.res1(x, t_emb)), t_emb)
