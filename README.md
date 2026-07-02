# SKGD: Structural Knowledge Guided Diffusion Model for Chest X-Ray Image Generation

Official implementation of **"Structural Knowledge Guided Diffusion Model for
Chest X-Ray Image Generation"** (SKGDM).

SKGDM is a unified diffusion-based framework that explicitly integrates
domain-specific **semantic knowledge** (from radiology reports via MedCLIP) and
explicit **anatomical priors** (from segmentation masks) into the diffusion
denoising process. This ensures generated chest X-rays are both semantically
aligned with medical descriptions and structurally faithful to anatomical
constraints.

## Method overview

The framework builds on Latent Diffusion Models (LDM). A frozen VAE encodes the
image `x` into a latent `z = E(x)`; Gaussian noise is added over `T = 1000`
steps; a conditional U-Net `epsilon_theta` predicts the noise conditioned on an
anatomical embedding `E_mask` and refined semantic guidance `F_out`.

### Core modules

| Module | Description | Paper Eq. |
|--------|-------------|-----------|
| **DHI** (Dense Hint Input) | Lightweight conv encoder with residual connections and multi-scale feature extraction; embeds the segmentation mask `M` into a dense anatomical embedding `E_mask` and per-stage hints. | Eq. (4) |
| **KGRM** (Knowledge-Guided Refinement Module) | Decomposes U-Net features into local (shallow conv) and global (GAP) visual streams, aligns each with the corresponding granular text features via dual cross-attention, and fuses them with element-wise addition + LayerNorm. | Eq. (7) |
| **Conditional U-Net** | Denoiser `epsilon_theta(z_t, t, E_mask, F_out)` injecting DHI hints at every encoder/decoder stage and KGRM refinement after each block. | Eq. (3), (5) |
| **Text encoder** | Frozen MedCLIP text encoder `psi` with a learnable projection `W_proj` producing `E_text = psi(T) W_proj`. | Eq. (6) |

### Training objective

```
L_total = E_{z0, eps, t, C} || eps - eps_theta(z_t, t, E_mask, F_out) ||^2   (Eq. 8)
```

Frozen: VAE image encoder, MedCLIP text encoder. Trainable: DHI, KGRM,
conditional U-Net, and the text projection `W_proj`.

## Repository structure

```
SKGD/
├── config.py              # configuration (dataset/pretrained paths left empty)
├── data.py                # chest X-ray dataset skeleton (image, mask, report)
├── train.py               # end-to-end training (Eq. 8)
├── sample.py              # 74-step DDIM inference
├── requirements.txt
└── models/
    ├── dhi.py             # Dense Hint Input (Eq. 4)
    ├── kgrm.py            # Knowledge-Guided Refinement Module (Eq. 7)
    ├── unet_blocks.py     # ResBlock / SelfAttention / Down/Mid/Up blocks
    ├── cond_unet.py       # conditional U-Net with DHI + KGRM injection (Eq. 3, 5)
    ├── vae.py             # frozen VAE wrapper (LDM)
    ├── text_encoder.py    # MedCLIP text encoder + W_proj (Eq. 6)
    ├── scheduler.py       # DDPM (T=1000) / DDIM (74-step) schedulers (Eq. 1)
    └── skgdm.py           # full SKGDM pipeline + loss (Eq. 8) + sampling
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

Fill in the dataset paths in `config.py` (or pass them as CLI arguments), then:

```bash
python train.py --manifest path/to/manifest.csv \
                --image-root path/to/images \
                --mask-root path/to/masks \
                --output-dir ./outputs
```

The manifest CSV is expected to have columns `image`, `mask`, `report`.

### Sampling

```bash
python sample.py --ckpt ./outputs/skgdm_epoch100.pt \
                 --mask path/to/lung_mask.png \
                 --report "No significant lesions in both lungs." \
                 --output ./outputs/sample.png
```

## Implementation details (from the paper, Sec. IV-C)

- Image resolution: 256x256 (generation), 448x448 (segmentation)
- Learning rate: 1e-5 (generation), 1e-4 (segmentation)
- Batch size: 64 (generation), 8 (segmentation)
- Diffusion steps: T = 1000 (DDPM), 74-step DDIM for inference
- Frozen VAE + MedCLIP text encoder; trainable DHI, KGRM, U-Net, W_proj

## Datasets

Experiments use MIMIC-CXR, CXRS, and CXLSeg. Dataset and pretrained checkpoint
paths are intentionally left empty in `config.py`; fill them in before running.
