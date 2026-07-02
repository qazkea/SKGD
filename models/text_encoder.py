"""Text encoder wrapper (MedCLIP, frozen) with a learnable projection.

Implements Eq. (6) of the paper:

    E_text = psi(T) W_proj  in R^d

The paper adopts MedCLIP [10], a vision-language model pretrained on large-scale
medical image-text pairs, as the text encoder ``psi``. Its Transformer text
encoder captures both high-level disease categories and fine-grained medical
terminology. A learnable linear projection ``W_proj`` maps the encoder output to
a dense semantic embedding ``E_text``. The encoder is frozen; ``W_proj`` is
trainable.

For runnability, the wrapper falls back to a standard CLIP text encoder
(``transformers.CLIPTextModel``) when the ``medclip`` package is unavailable.
The interface (per-token features + a sentence-level feature) is identical.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class TextEncoderWrapper(nn.Module):
    def __init__(
        self,
        text_dim: int = 768,
        proj_dim: Optional[int] = None,
        max_length: int = 77,
        text_encoder: Optional[nn.Module] = None,
        tokenizer=None,
    ):
        super().__init__()
        proj_dim = proj_dim or text_dim
        self.text_dim = text_dim
        self.proj_dim = proj_dim
        self.max_length = max_length

        self.text_encoder = text_encoder
        if self.text_encoder is not None:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.proj = nn.Linear(text_dim, proj_dim)
        self.tokenizer = tokenizer

    @classmethod
    def from_clip(cls, model_name: str = "openai/clip-vit-base-patch32", proj_dim: Optional[int] = None):
        from transformers import CLIPTextModel, CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained(model_name)
        text_encoder = CLIPTextModel.from_pretrained(model_name)
        text_dim = text_encoder.config.hidden_size
        return cls(text_dim=text_dim, proj_dim=proj_dim, text_encoder=text_encoder, tokenizer=tokenizer)

    @classmethod
    def from_medclip(cls, proj_dim: Optional[int] = None):
        try:
            from medclip import MedCLIPModel  # type: ignore
        except ImportError as e:
            raise ImportError(
                "The `medclip` package is required for the faithful MedCLIP text "
                "encoder. Install it or use TextEncoderWrapper.from_clip()."
            ) from e

        medclip = MedCLIPModel()
        text_dim = medclip.text_encoder.config.hidden_size
        return cls(text_dim=text_dim, proj_dim=proj_dim, text_encoder=medclip.text_encoder)

    def tokenize(self, texts):
        if self.tokenizer is None:
            raise RuntimeError("No tokenizer attached.")
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.text_encoder is None:
            raise RuntimeError("Text encoder is not initialised.")

        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        tokens = out.last_hidden_state

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = tokens.mean(dim=1)
        global_feat = pooled.unsqueeze(1)

        tokens = self.proj(tokens)
        global_feat = self.proj(global_feat)
        return tokens, global_feat
