"""Black-box characterisation of the victim: clean accuracy on the public
CIFAR-10 test split and label stability under common corruptions.

A low clean accuracy together with high noise tolerance is the signature of an
adversarially trained (robust) model, which sets the floor on how small an
imperceptible perturbation can be.  Everything here uses labels only.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle import load_oracle
from data import load_clean


def main(dev="cuda"):
    import cifar
    orc = load_oracle("data/p2_data/cnn_model.pt", device=dev, batch_size=2048)

    xu, y = cifar.load("test", device=dev)
    x = xu.float().div(255.0)
    pred = orc.predict(x)
    acc = (pred == y).float().mean().item()
    print(f"CIFAR-10 test accuracy (10000 imgs) : {100*acc:.2f} %")
    xtr, ytr = cifar.load("train", device=dev)
    atr = (orc.predict(xtr.float().div(255.0)) == ytr).float().mean().item()
    print(f"CIFAR-10 TRAIN accuracy (50000 imgs): {100*atr:.2f} %")
    print("  (standard CIFAR CNN ~ 93-96 % | L-inf adversarially trained ~ 80-88 %)")

    xc, yc, _ = load_clean("data/p2_data", device=dev)

    print("\nlabel stability under corruptions (fraction of the 1000 provided "
          "images whose label changes):")
    for name, fn in [
        ("gauss sigma=2/255 ", lambda v: v + torch.randn_like(v) * (2 / 255)),
        ("gauss sigma=8/255 ", lambda v: v + torch.randn_like(v) * (8 / 255)),
        ("gauss sigma=16/255", lambda v: v + torch.randn_like(v) * (16 / 255)),
        ("gauss sigma=32/255", lambda v: v + torch.randn_like(v) * (32 / 255)),
        ("uniform 8/255     ", lambda v: v + (torch.rand_like(v) * 2 - 1) * (8 / 255)),
        ("uniform 16/255    ", lambda v: v + (torch.rand_like(v) * 2 - 1) * (16 / 255)),
        ("blur 3x3          ", lambda v: F.avg_pool2d(F.pad(v, (1,) * 4, mode="reflect"), 3, 1)),
        ("quantise to 32 lvl", lambda v: (v * 31).round() / 31),
    ]:
        flip = (orc.predict(fn(xc).clamp(0, 1)) != yc).float().mean().item()
        print(f"  {name}: {100*flip:5.1f} %")


if __name__ == "__main__":
    main()
