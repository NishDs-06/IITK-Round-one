"""
Re-run only the final verification stage on already-written adversarial images.

The attack stages are unchanged; this re-applies the margin requirement of
`verify.py` to the files in `adv_images/`, rewrites the ones that needed a
larger perturbation, and then checks the result by reading the files back from
disk again.

    python src/reharden.py --adv adv_images
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import metrics as M
import verify as V
from data import load_clean, save_adv_images
from make_figures import load_adv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/p2_data")
    p.add_argument("--adv", default="adv_images")
    p.add_argument("--metrics", default="report/metrics_final.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--shrink", type=float, default=0.97)
    p.add_argument("--step", type=float, default=1.05)
    a = p.parse_args()
    dev = a.device
    model_path = os.path.join(a.data, "cnn_model.pt")

    x, y, names = load_clean(a.data, device=dev)
    xa = load_adv(a.adv, names, device=dev)
    n = x.shape[0]
    backends = V.make_backends(model_path, dev)
    print(f"[reharden] {n} images, {len(backends)} backends x "
          f"{len(V.input_views(x[:1], dev))} input representations")

    before = V.consensus_adversarial(backends, xa, y)
    margin = V.robustly_adversarial(backends, x, y, xa, a.shrink)
    print(f"[reharden] as written: {int(before.sum())}/{n} adversarial, "
          f"{int(margin.sum())}/{n} with a {100*(1-a.shrink):.0f} % radial margin")

    x_new, ok = V.harden(backends, x, y, xa, step=a.step, shrink=a.shrink)
    d = M.report(x, x_new, ok)
    print(f"[reharden] after: {int(ok.sum())}/{n} robust   "
          f"L2 median {d['l2_median']:.2f} mean {d['l2_mean']:.2f} "
          f"max {d['l2_max']:.2f}")

    save_adv_images(x_new, names, a.adv)
    reread = load_adv(a.adv, names, device=dev)
    assert torch.equal((reread * 255).round().to(torch.uint8),
                       (x_new * 255).round().to(torch.uint8)), \
        "saved PNGs differ from the verified tensors"
    ok_disk = V.consensus_adversarial(backends, reread, y)
    ok_margin = V.robustly_adversarial(backends, x, y, reread, a.shrink)
    print(f"[reharden] re-read from disk: {int(ok_disk.sum())}/{n} adversarial, "
          f"{int(ok_margin.sum())}/{n} with margin")

    r = M.report(x, reread, ok_disk)
    if os.path.exists(a.metrics):
        old = json.load(open(a.metrics))
        for k in ("queries_total", "queries_per_image", "queries_per_stage",
                  "wall_clock_s", "config"):
            if k in old:
                r[k] = old[k]
    r["verification"] = {
        "backends": len(backends),
        "input_representations": len(V.input_views(x[:1], dev)),
        "radial_margin": 1 - a.shrink,
        "verification_queries": sum(b.n_queries for b in backends),
    }
    with open(a.metrics, "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("\n" + M.format_report(r))


if __name__ == "__main__":
    main()
