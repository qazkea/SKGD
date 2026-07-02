"""Central configuration for SKGDM.

Dataset / pretrained paths are intentionally left empty; fill them in with your
local MIMIC-CXR / CXRS / CXLSeg paths and VAE / text-encoder checkpoints before
training. Hyper-parameters follow the implementation details reported in the
paper (Sec. IV-C).
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DataConfig:
    mimic_cxr_dir: str = ""
    cxrs_dir: str = ""
    cxlseg_dir: str = ""
    segmentation_mask_dir: str = ""
    reports_csv: str = ""
    image_size: int = 256
    mask_size: int = 256
    max_length: int = 77
    num_workers: int = 8


@dataclass
class ModelConfig:
    latent_channels: int = 4
    model_channels: int = 64
    channel_mults: Tuple[int, ...] = (1, 2, 4, 4)
    num_heads: int = 8
    head_dim: int = 64
    mask_in_channels: int = 1
    text_dim: int = 768
    hint_channels: int = 32
    vae_pretrained: str = ""
    text_encoder_name: str = "openai/clip-vit-base-patch32"
    use_medclip: bool = True
    vae_scale_factor: float = 0.18215


@dataclass
class TrainConfig:
    output_dir: str = "./outputs"
    learning_rate: float = 1e-5
    batch_size: int = 64
    num_train_timesteps: int = 1000
    num_inference_steps: int = 74
    epochs: int = 100
    save_every: int = 5
    mixed_precision: str = "fp16"
    seed: int = 42


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
