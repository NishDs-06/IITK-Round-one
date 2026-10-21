"""Diagnostics: HSJA convergence rate and health of the label-only normal
estimate (how balanced the +-1 answers are at the probe radius)."""
import os
import sys
import math
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attack as A
from data import load_clean
from oracle import load_oracle


def run(n_img=20, iters=60, b0=100, bmax=400, gamma=1.0, bsearch=20, dev="cuda",
        max_batch=2048):
    torch.manual_seed(0)
    orc = load_oracle("data/p2_data/cnn_model.pt", device=dev, batch_size=max_batch)
    x, y, _ = load_clean("data/p2_data", device=dev)
    idx = torch.arange(0, 1000, 1000 // n_img, device=dev)[:n_img]
    x, y = x[idx].contiguous(), y[idx].contiguous()
    orc.reset_counter()

    seeds = A.init_seeds(x, y, 4)
    ct = A.BestTracker(x)
    x_adv, d0 = A.best_initial_boundary_point(orc, x, y, seeds, bsearch, ct)
    print(f"init boundary L2(0-255) median {(d0*255).median():.1f}   q={orc.n_queries}")

    d = x[0].numel()
    theta = gamma / d ** 1.5
    t0 = time.time()
    for t in range(1, iters + 1):
        dist = A.l2(x_adv - x)
        delta = (math.sqrt(d) * theta * dist).clamp(min=1e-7)
        B = int(min(b0 * math.sqrt(t), bmax))

        # --- instrumented gradient estimate: report balance of +-1 answers ---
        u = torch.randn(n_img, B, *x.shape[1:], device=dev)
        u = u / (u.flatten(2).norm(dim=2).view(n_img, B, 1, 1, 1) + 1e-12)
        dl = delta.view(-1, 1, 1, 1, 1)
        pert = (x_adv.unsqueeze(1) + dl * u).clamp(0, 1)
        u = (pert - x_adv.unsqueeze(1)) / dl
        adv = orc.is_adversarial(pert.reshape(n_img * B, *x.shape[1:]),
                                 y.repeat_interleave(B)).view(n_img, B)
        f = adv.float() * 2 - 1
        fbar = f.mean(1, keepdim=True)
        w = torch.where(fbar.abs() >= 1 - 1e-6, fbar.sign(), f - fbar)
        grad = (w.view(n_img, B, 1, 1, 1) * u).mean(1)
        grad = grad / (A.l2(grad).view(-1, 1, 1, 1) + 1e-12)

        moved, ok = A.geometric_step(orc, x_adv, grad, y, dist / math.sqrt(t), tracker=ct)
        x_adv, _ = A.line_search_boundary(orc, x, moved, y, bsearch, ct)

        if t % 5 == 0 or t <= 3:
            dd = A.l2(x_adv - x) * 255
            li = A.linf(x_adv - x) * 255
            print(f"it {t:3d} B={B:4d} |fbar| {fbar.abs().mean():.3f} "
                  f"advfrac {adv.float().mean():.3f} step_ok {ok.float().mean():.2f} "
                  f"| L2 med {dd.median():7.1f} mean {dd.mean():7.1f} "
                  f"Linf med {li.median():5.1f} | q={orc.n_queries:,} "
                  f"({time.time()-t0:.0f}s)")
    return ct


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--b0", type=int, default=100)
    p.add_argument("--bmax", type=int, default=400)
    p.add_argument("--gamma", type=float, default=1.0)
    a = p.parse_args()
    run(n_img=a.n, iters=a.iters, b0=a.b0, bmax=a.bmax, gamma=a.gamma)
