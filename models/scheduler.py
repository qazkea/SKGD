"""Noise scheduler wrappers (DDPM / DDIM).

The forward diffusion process follows Eq. (1):

    q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) x_{t-1}, beta_t I)

Training uses a DDPM scheduler with ``T = 1000`` steps; inference uses a
lightweight 74-step DDIM sampler to balance sampling speed and quality, as
described in the implementation details of the paper.
"""

from typing import Optional

import torch


class NoiseScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 74,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
    ):
        from diffusers import DDPMScheduler, DDIMScheduler

        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps

        self.train_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            clip_sample=False,
        )
        self.infer_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            clip_sample=False,
        )
        self.infer_scheduler.set_timesteps(num_inference_steps)

    @property
    def alphas_cumprod(self) -> torch.Tensor:
        return self.train_scheduler.alphas_cumprod

    def add_noise(
        self,
        original: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        return self.train_scheduler.add_noise(original, noise, timesteps)

    def training_timesteps(self, batch_size: int, device) -> torch.Tensor:
        return torch.randint(0, self.num_train_timesteps, (batch_size,), device=device)

    def ddim_step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        return self.infer_scheduler.step(model_output, timestep, sample).prev_sample

    @property
    def inference_timesteps(self):
        return self.infer_scheduler.timesteps
