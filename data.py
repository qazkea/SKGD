"""Dataset utilities.

A skeleton Chest X-ray dataset that yields ``(image, mask, input_ids,
attention_mask)`` tuples. Fill in the manifest CSV path (columns: ``image``,
``mask``, ``report``) and root directories in ``config.DataConfig`` to point at
your local MIMIC-CXR / CXRS / CXLSeg data; the dataset paths are intentionally
left empty.
"""

import os
from typing import Optional

import torch
from torch.utils.data import Dataset

from PIL import Image


def _to_tensor(img: Image.Image, size: int) -> torch.Tensor:
    img = img.convert("L").resize((size, size), Image.BILINEAR)
    t = torch.frombuffer(img.tobytes(), dtype=torch.uint8).float().view(size, size) / 127.5 - 1.0
    return t.unsqueeze(0)


def _mask_to_tensor(mask: Image.Image, size: int) -> torch.Tensor:
    mask = mask.convert("L").resize((size, size), Image.NEAREST)
    t = torch.frombuffer(mask.tobytes(), dtype=torch.uint8).float().view(size, size) / 255.0
    return t.unsqueeze(0)


class CXRDataset(Dataset):
    def __init__(self, manifest_csv: str, image_root: str = "", mask_root: str = "", tokenizer=None,
                 image_size: int = 256, mask_size: int = 256, max_length: int = 77):
        if not manifest_csv:
            raise ValueError("manifest_csv is empty. Fill in the dataset path in config.DataConfig.")
        import csv

        self.samples = []
        with open(manifest_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.samples.append(row)
        self.image_root = image_root
        self.mask_root = mask_root
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.mask_size = mask_size
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        row = self.samples[idx]
        img = Image.open(os.path.join(self.image_root, row["image"]))
        mask = Image.open(os.path.join(self.mask_root, row["mask"]))
        image = _to_tensor(img, self.image_size)
        mask_t = _mask_to_tensor(mask, self.mask_size)

        report = row.get("report", "")
        if self.tokenizer is None:
            input_ids = torch.zeros(self.max_length, dtype=torch.long)
            attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        else:
            enc = self.tokenizer(report, padding="max_length", truncation=True,
                                  max_length=self.max_length, return_tensors="pt")
            input_ids = enc["input_ids"][0]
            attention_mask = enc["attention_mask"][0]

        return image, mask_t, input_ids, attention_mask
