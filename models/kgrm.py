"""Knowledge-Guided Refinement Module (KGRM).

Implements Eq. (7) of the paper:

    F_out = Attn_local(V_local, T_local) (+) Attn_global(V_global, T_global)

The module dynamically decomposes an intermediate U-Net feature map ``F_in``
into a local visual representation (via a shallow convolutional block, keeping
spatial detail) and a global visual representation (via Global Average
Pooling, giving a holistic vector). The text embedding ``E_text`` is likewise
decomposed into token-level (local) and sentence-level (global) features.
Granular-specific cross-attention is performed along two parallel paths and the
results are fused by element-wise addition followed by layer normalization.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """Standard cross-attention: ``query`` attends to ``context``."""

    def __init__(self, query_dim: int, context_dim: int, num_heads: int = 8, head_dim: int = 64):
        super().__init__()
        inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
        self.norm = nn.LayerNorm(query_dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if query.dim() == 4:
            b, c, h, w = query.shape
            query = query.flatten(2).transpose(1, 2)
            spatial = (h, w)
        else:
            spatial = None

        q = self.to_q(query)
        k = self.to_k(context)
        v = self.to_v(context)

        q = q.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
        k = k.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
        v = v.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).flatten(2)
        out = self.to_out(out)
        out = self.norm(query + out)

        if spatial is not None:
            b, n, _ = out.shape
            out = out.transpose(1, 2).unflatten(2, spatial)
        return out


class KnowledgeGuidedRefinementModule(nn.Module):
    """KGRM: multi-granularity alignment between visual and textual features."""

    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
    ):
        super().__init__()
        self.local_conv = nn.Sequential(
            nn.Conv2d(visual_dim, visual_dim, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, visual_dim), visual_dim),
            nn.SiLU(),
            nn.Conv2d(visual_dim, visual_dim, kernel_size=3, padding=1),
        )
        self.global_proj = nn.Linear(visual_dim, visual_dim)

        self.local_attn = CrossAttention(visual_dim, text_dim, num_heads, head_dim)
        self.global_attn = CrossAttention(visual_dim, text_dim, num_heads, head_dim)

        self.fuse_norm = nn.GroupNorm(min(8, visual_dim), visual_dim)

    def forward(
        self,
        f_in: torch.Tensor,
        text_tokens: torch.Tensor,
        text_global: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Refine ``F_in`` using text guidance.

        Args:
            f_in: intermediate U-Net feature map (B, C, H, W).
            text_tokens: per-token text features (B, L, D) used as local context.
            text_global: sentence-level text feature (B, 1, D) used as global
                context. If ``None``, the mean-pooled tokens are used.

        Returns:
            Refined feature map ``F_out`` (B, C, H, W).
        """
        v_local = self.local_conv(f_in)

        pooled = f_in.mean(dim=(2, 3))
        v_global = self.global_proj(pooled).unsqueeze(1)

        if text_global is None:
            text_global = text_tokens.mean(dim=1, keepdim=True)

        local_out = self.local_attn(v_local, text_tokens)
        global_out = self.global_attn(v_global, text_global)
        global_out = global_out.transpose(1, 2).unsqueeze(-1).expand_as(local_out)

        f_out = local_out + global_out
        return self.fuse_norm(f_out)
