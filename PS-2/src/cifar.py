"""Public CIFAR-10 train/test split, read from the HuggingFace parquet copy.

Only used for (a) characterising the victim's clean accuracy and (b) training
local surrogate networks.  It is public data: no information about the victim
model is involved.
"""
from __future__ import annotations

import io
import os

import numpy as np
import torch
from PIL import Image

ROOT = "data/cifar10_hf"


def _decode(parquet_path):
    import pyarrow.parquet as pq
    tbl = pq.read_table(parquet_path)
    cols = tbl.column_names
    img_col = "img" if "img" in cols else ("image" if "image" in cols else cols[0])
    lab_col = "label" if "label" in cols else cols[-1]
    imgs = tbl.column(img_col).to_pylist()
    labels = np.asarray(tbl.column(lab_col).to_pylist(), dtype=np.int64)
    out = np.empty((len(imgs), 32, 32, 3), dtype=np.uint8)
    for i, rec in enumerate(imgs):
        raw = rec["bytes"] if isinstance(rec, dict) else rec
        out[i] = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
    return out, labels


def load(split="train", root=ROOT, device="cpu", cache=True):
    """Return (x_uint8 (N,3,32,32) on `device`, y int64)."""
    cache_path = os.path.join(root, f"{split}.pt")
    if cache and os.path.exists(cache_path):
        d = torch.load(cache_path, map_location="cpu", weights_only=True)
    else:
        arr, lab = _decode(os.path.join(root, f"{split}.parquet"))
        d = {"x": torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous(),
             "y": torch.from_numpy(lab)}
        if cache:
            torch.save(d, cache_path)
    return d["x"].to(device), d["y"].to(device)


if __name__ == "__main__":
    for s in ("train", "test"):
        x, y = load(s)
        print(s, tuple(x.shape), x.dtype, "labels", int(y.min()), int(y.max()),
              "counts", torch.bincount(y).tolist())
