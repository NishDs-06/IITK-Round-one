"""Loading of the clean image set, in exactly the order used by the notebook."""

from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image

from oracle import CLASS_NAMES

IMG_PER_CLASS = 100


def image_names(img_per_class: int = IMG_PER_CLASS):
    """`airplane_0 ... truck_99` in the notebook's order."""
    return [f"{name}_{i}" for name in CLASS_NAMES for i in range(img_per_class)]


def load_clean(data_dir: str, device: str = "cpu", img_per_class: int = IMG_PER_CLASS):
    """Return (x01, y_true, names).

    x01 : (N,3,32,32) float32 in [0,1] -- exact 8-bit pixel values, k/255.
    y    : (N,) int64 ground-truth labels.
    """
    names = image_names(img_per_class)
    labels = [CLASS_NAMES.index(n.rsplit("_", 1)[0]) for n in names]
    arrs = []
    for n in names:
        p = os.path.join(data_dir, "clean_images", f"{n}.png")
        arrs.append(np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8))
    x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).contiguous()
    x01 = x.to(torch.float32).div_(255.0).to(device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    return x01, y, names


def save_adv_images(x01: torch.Tensor, names, out_dir: str) -> None:
    """Write adversarial images as `<name>_adv.png` (lossless 8-bit RGB)."""
    os.makedirs(out_dir, exist_ok=True)
    arr = (x01.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu()
    arr = arr.permute(0, 2, 3, 1).numpy()
    for a, n in zip(arr, names):
        Image.fromarray(a, mode="RGB").save(os.path.join(out_dir, f"{n}_adv.png"))
