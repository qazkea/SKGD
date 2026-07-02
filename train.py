"""Training entry point for SKGDM.

Optimises the end-to-end objective of Eq. (8):

    L_total = E_{z0, eps, t, C} || eps - eps_theta(z_t, t, E_mask, F_out) ||^2

Frozen: VAE image encoder, MedCLIP text encoder.
Trainable: DHI encoder, KGRM, conditional U-Net, text projection W_proj.
"""

import os
import argparse

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from config import Config
from data import CXRDataset
from models import SKGDM, ConditionalUNet, VAEWrapper, TextEncoderWrapper, NoiseScheduler
from models.skgdm import SKGDMConfig


def build_model(cfg: Config, device: torch.device) -> SKGDM:
    vae = VAEWrapper.from_pretrained(cfg.model.vae_pretrained)
    text_encoder = TextEncoderWrapper.from_medclip() if cfg.model.use_medclip else TextEncoderWrapper.from_clip(cfg.model.text_encoder_name)
    unet = ConditionalUNet(
        latent_channels=cfg.model.latent_channels,
        model_channels=cfg.model.model_channels,
        channel_mults=cfg.model.channel_mults,
        num_heads=cfg.model.num_heads,
        head_dim=cfg.model.head_dim,
        mask_in_channels=cfg.model.mask_in_channels,
        text_dim=cfg.model.text_dim,
        hint_channels=cfg.model.hint_channels,
    )
    scheduler = NoiseScheduler(
        num_train_timesteps=cfg.train.num_train_timesteps,
        num_inference_steps=cfg.train.num_inference_steps,
    )
    model = SKGDM(vae=vae, text_encoder=text_encoder, unet=unet, scheduler=scheduler,
                  config=SKGDMConfig(text_dim=text_encoder.proj_dim))
    model.to(device)
    model.vae.eval()
    model.text_encoder.text_encoder.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="")
    parser.add_argument("--image-root", default="")
    parser.add_argument("--mask-root", default="")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = Config()
    cfg.data.mimic_cxr_dir = args.image_root or cfg.data.mimic_cxr_dir
    cfg.data.segmentation_mask_dir = args.mask_root or cfg.data.segmentation_mask_dir

    model = build_model(cfg, device)
    tokenizer = model.text_encoder.tokenizer

    dataset = CXRDataset(
        manifest_csv=args.manifest or cfg.data.reports_csv,
        image_root=cfg.data.mimic_cxr_dir,
        mask_root=cfg.data.segmentation_mask_dir,
        tokenizer=tokenizer,
        image_size=cfg.data.image_size,
        mask_size=cfg.data.mask_size,
        max_length=cfg.data.max_length,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=cfg.data.num_workers, pin_memory=True)

    optimizer = AdamW(model.trainable_parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, (image, mask, input_ids, attention_mask) in enumerate(loader):
            image = image.to(device)
            mask = mask.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss = model.compute_loss(image, mask, input_ids, attention_mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.5f}")

        avg = running / max(len(loader), 1)
        print(f"== epoch {epoch} avg loss {avg:.5f} ==")

        if epoch % cfg.train.save_every == 0 or epoch == args.epochs:
            ckpt = os.path.join(args.output_dir, f"skgdm_epoch{epoch}.pt")
            torch.save({"unet": model.unet.state_dict(),
                        "text_proj": model.text_encoder.proj.state_dict()}, ckpt)
            print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
