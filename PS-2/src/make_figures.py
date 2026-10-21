"""Figures for the report: clean vs adversarial vs amplified perturbation,
and the distribution of per-image distortion."""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_clean, image_names
from oracle import CLASS_NAMES, load_oracle
import metrics as M


def load_adv(adv_dir, names, device="cpu"):
    arrs = [np.asarray(Image.open(os.path.join(adv_dir, f"{n}_adv.png")).convert("RGB"),
                       dtype=np.uint8) for n in names]
    # keep the non-contiguous permuted layout that a PIL-based loader gives,
    # so what we measure is what an evaluator sees
    x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).float().div_(255.0)
    return x.to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/p2_data")
    p.add_argument("--adv", default="adv_images")
    p.add_argument("--out", default="report")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    x, y, names = load_clean(a.data, device=a.device)
    xa = load_adv(a.adv, names, device=a.device)
    orc = load_oracle(os.path.join(a.data, "cnn_model.pt"), device=a.device,
                      batch_size=2048)
    pred_c = orc.predict(x)
    pred_a = orc.predict(xa)
    ok = pred_a != y
    r = M.report(x, xa, ok)
    print(M.format_report(r))

    # ---------------- qualitative panel -----------------------------------
    rng = np.random.default_rng(0)
    pick = [int(c * 100 + rng.integers(100)) for c in range(10)]
    fig, axes = plt.subplots(3, 10, figsize=(17, 5.6))
    for j, i in enumerate(pick):
        c = x[i].cpu().permute(1, 2, 0).numpy()
        adv = xa[i].cpu().permute(1, 2, 0).numpy()
        d = (adv - c)
        amp = np.clip(0.5 + d * 8.0, 0, 1)      # 8x amplified, mid-grey = 0
        for row, (img, ttl) in enumerate([
                (c, f"{CLASS_NAMES[pred_c[i]]}"),
                (adv, f"{CLASS_NAMES[pred_a[i]]}"),
                (amp, f"L2={((d*255)**2).sum()**0.5:.0f} Li={np.abs(d*255).max():.0f}")]):
            ax = axes[row, j]
            ax.imshow(img)
            ax.set_title(ttl, fontsize=8)
            # keep the frame off but the axis alive, so the row label shows
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
    for row, lab in enumerate(["clean\n(prediction)", "adversarial\n(prediction)",
                               "difference\n($\\times$8, offset)"]):
        axes[row, 0].set_ylabel(lab, fontsize=9)
    fig.suptitle("Clean vs. adversarial images and the amplified perturbation", y=0.99)
    plt.tight_layout()
    f1 = os.path.join(a.out, "examples.png")
    plt.savefig(f1, dpi=160)
    print("wrote", f1)

    # ---------------- distortion distributions -----------------------------
    d = ((xa - x) * 255).flatten(1)
    l2v = d.norm(dim=1).cpu().numpy()
    liv = d.abs().max(dim=1).values.cpu().numpy()
    mse = (d ** 2).mean(dim=1)
    psnr = (10 * torch.log10(255.0 ** 2 / mse.clamp(min=1e-12))).cpu().numpy()
    ssimv = M.ssim(x, xa).cpu().numpy()

    fig, ax = plt.subplots(1, 4, figsize=(16, 3.4))
    for k, (v, name) in enumerate([(l2v, r"$\ell_2$ (0-255 scale)"),
                                   (liv, r"$\ell_\infty$ (0-255 scale)"),
                                   (psnr, "PSNR (dB)"),
                                   (ssimv, "SSIM")]):
        ax[k].hist(v, bins=50, color="#3b6ea5", edgecolor="white", linewidth=0.3)
        ax[k].axvline(np.median(v), color="#c0392b", lw=1.4,
                      label=f"median {np.median(v):.3g}")
        ax[k].set_xlabel(name)
        ax[k].set_ylabel("images")
        ax[k].legend(fontsize=8)
    plt.tight_layout()
    f2 = os.path.join(a.out, "distortion.png")
    plt.savefig(f2, dpi=160)
    print("wrote", f2)


if __name__ == "__main__":
    main()
