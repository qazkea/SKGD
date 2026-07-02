"""Sampling entry point for SKGDM (74-step DDIM inference).

Loads a trained checkpoint, runs anatomy- and knowledge-guided DDIM sampling,
decodes the latent with the frozen VAE, and saves the generated chest X-ray.
"""

import os
import argparse

import torch
from PIL import Image

from config import Config
from models import SKGDM, ConditionalUNet, VAEWrapper, TextEncoderWrapper, NoiseScheduler
from models.skgdm import SKGDMConfig


def load_model(cfg: Config, ckpt_path: str, device: torch.device) -> SKGDM:
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
    state = torch.load(ckpt_path, map_location=device)
    model.unet.load_state_dict(state["unet"])
    if "text_proj" in state:
        model.text_encoder.proj.load_state_dict(state["text_proj"])
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--mask", required=True, help="path to a binary lung segmentation mask")
    parser.add_argument("--report", default="No significant lesions in both lungs.")
    parser.add_argument("--output", default="./outputs/sample.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    model = load_model(cfg, args.ckpt, device)

    mask_img = Image.open(args.mask).convert("L").resize((cfg.data.mask_size, cfg.data.mask_size), Image.NEAREST)
    mask = torch.frombuffer(mask_img.tobytes(), dtype=torch.uint8).float().view(1, 1, cfg.data.mask_size, cfg.data.mask_size) / 255.0
    mask = mask.to(device)

    enc = model.text_encoder.tokenize([args.report])
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        image = model.sample(mask, input_ids, attention_mask)

    image = (image.clamp(-1, 1) + 1.0) / 2.0
    image = (image[0, 0].cpu().numpy() * 255).astype("uint8")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    Image.fromarray(image).save(args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
