"""Variational Autoencoder wrapper (frozen).

Following the Latent Diffusion Model paradigm, a pre-trained VAE image encoder
``E`` compresses the chest X-ray ``x`` into a latent representation ``z = E(x)``
and a decoder ``D`` reconstructs the image from the latent. The VAE is kept
frozen during end-to-end training of the SKGDM framework.
"""

from typing import Optional

import torch
import torch.nn as nn


class VAEWrapper(nn.Module):
    def __init__(self, vae: Optional[nn.Module] = None, scale_factor: float = 0.18215):
        super().__init__()
        self.vae = vae
        self.scale_factor = scale_factor
        if self.vae is not None:
            for p in self.vae.parameters():
                p.requires_grad = False

    @classmethod
    def from_pretrained(cls, pretrained_path: Optional[str] = None, subfolder: str = "vae", **kwargs):
        from diffusers import AutoencoderKL

        if pretrained_path is None or pretrained_path == "":
            vae = AutoencoderKL(**kwargs)
        else:
            vae = AutoencoderKL.from_pretrained(pretrained_path, subfolder=subfolder)
        return cls(vae)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            raise RuntimeError("VAE is not initialised.")
        posterior = self.vae.encode(x).latent_dist
        z = posterior.sample()
        return z * self.scale_factor

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            raise RuntimeError("VAE is not initialised.")
        return self.vae.decode(z / self.scale_factor).sample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)
