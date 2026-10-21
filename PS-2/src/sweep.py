"""Hyper-parameter sweep: distortion vs. query budget for different
subspace schedules / sample counts, all on the same images and seed."""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attack as A
import metrics as M
from data import load_clean
from oracle import load_oracle


def parse_sched(s):
    """'12:8,22:16,30:0' -> [(12,8),(22,16),(30,0)]"""
    if not s:
        return None
    return [tuple(int(v) for v in part.split(":")) for part in s.split(",")]


def one_run(orc, x, y, name, iters, b0, bmax, sched, bsearch=20, seeds=4,
            greedy_rounds=3, verbose=True, max_batch=2048, walk=0,
            walk_k=0, walk_before=0, geo=0, geo_k=0, alt=0, nets=None,
            grid=0, rescue=0):
    orc.reset_counter()
    torch.manual_seed(0)
    sd = A.init_seeds(x, y, seeds)

    qt = A.BestTracker(x)
    s0 = orc.quantize_pixels(sd[:, 0])
    qt.update(s0, orc.is_adversarial(s0, y))

    ct = A.BestTracker(x)
    x_adv, _ = A.best_initial_boundary_point(orc, x, y, sd, bsearch, ct)
    print(f"    [init] L2med {(A.l2(x_adv-x)*255).median():7.1f} "
          f"q/img {orc.n_queries/x.shape[0]:.0f}")

    if nets:
        import transfer as T
        pts, rad = T.transfer_init(orc, nets, x, y, tracker=ct)
        better = torch.isfinite(rad) & (rad < A.l2(x_adv - x))
        x_adv[better] = pts[better]
        print(f"    [xfer] L2med {(A.l2(x_adv-x)*255).median():7.1f} "
              f"improved {int(better.sum())}/{x.shape[0]} "
              f"q/img {orc.n_queries/x.shape[0]:.0f}")

    t0 = time.time()

    def log(t, b, k, dist, q):
        if verbose and (t % 5 == 0 or t == 1):
            print(f"    it {t:3d} B={b:4d} k={k:2d} L2med {(dist*255).median():7.1f} "
                  f"q/img {q/x.shape[0]:7.0f}  ({time.time()-t0:.0f}s)")

    if walk_before:
        _, x_adv = A.boundary_walk(orc, x, y, x_adv, steps=walk_before,
                                   tracker=ct, subspace_k=walk_k)
        print(f"    [pre-walk] L2med {(A.l2(x_adv-x)*255).median():7.1f} "
              f"q/img {orc.n_queries/x.shape[0]:.0f}")

    if iters:
        A.hsja(orc, x, y, x_adv, iters=iters, b0=b0, b_max=bmax,
               bsearch_steps=bsearch, tracker=ct, max_batch=max_batch,
               subspace_schedule=sched, log=log)
    q_hsja = orc.n_queries

    if rescue:
        pass
    if geo:
        _, g_adv = A.geometric_walk(orc, x, y, ct.best.clone(), steps=geo,
                                    tracker=ct, subspace_k=geo_k)
        print(f"    [geo ] L2med {(ct.dist*255).median():7.1f} "
              f"q/img {orc.n_queries/x.shape[0]:.0f}")

    if walk:
        _, x_adv = A.boundary_walk(orc, x, y, ct.best.clone(), steps=walk,
                                   tracker=ct, subspace_k=walk_k)
        print(f"    [walk] L2med {(ct.dist*255).median():7.1f} "
              f"q/img {orc.n_queries/x.shape[0]:.0f}")

    for _ in range(alt):
        _, g = A.geometric_walk(orc, x, y, ct.best.clone(), steps=300, tracker=ct)
        _, w = A.boundary_walk(orc, x, y, ct.best.clone(), steps=300, tracker=ct)
        print(f"    [alt ] L2med {(ct.dist*255).median():7.1f} "
              f"q/img {orc.n_queries/x.shape[0]:.0f}")

    A.quantize_finalize(orc, x, y, ct.best, qt)
    print(f"    [quant] L2med {(qt.dist*255).median():7.1f} "
          f"mean {(qt.dist*255).mean():7.1f} q/img {orc.n_queries/x.shape[0]:.0f}")
    if grid:
        A.refine_on_grid(orc, x, y, qt, geo_steps=grid, walk_steps=grid, rounds=1)
    A.polish_uint8(orc, x, y, qt, greedy_rounds=greedy_rounds)
    q_total = orc.n_queries

    final = orc.quantize_pixels(qt.best)
    ok = orc.is_adversarial(final, y)
    r = M.report(x, final, ok, n_queries=q_total)
    print(f"  [{name}] ASR {r['success_rate']:.1f}%  L2 med {r['l2_median']:.1f} "
          f"mean {r['l2_mean']:.1f} | Linf med {r['linf_median']:.0f} | "
          f"PSNR {r['psnr_mean']:.2f} | SSIM {r['ssim_mean']:.4f} | "
          f"q/img {q_total/x.shape[0]:.0f} (hsja {q_hsja/x.shape[0]:.0f})")
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--b0", type=int, default=100)
    p.add_argument("--bmax", type=int, default=400)
    p.add_argument("--sched", action="append", default=[],
                   help="e.g. --sched '' (full space)  --sched '30:8'")
    p.add_argument("--device", default="cuda")
    p.add_argument("--walk", type=int, default=0)
    p.add_argument("--walk-k", type=int, default=0)
    p.add_argument("--walk-before", type=int, default=0)
    p.add_argument("--geo", type=int, default=0)
    p.add_argument("--geo-k", type=int, default=0)
    p.add_argument("--alt", type=int, default=0)
    p.add_argument("--transfer", action="store_true")
    p.add_argument("--grid", type=int, default=0)
    p.add_argument("--surrogate-glob", default="data/surrogates/*.pt")
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()

    orc = load_oracle("data/p2_data/cnn_model.pt", device=a.device, batch_size=2048)
    x, y, _ = load_clean("data/p2_data", device=a.device)
    idx = torch.arange(0, 1000, max(1, 1000 // a.n), device=a.device)[:a.n]
    x, y = x[idx].contiguous(), y[idx].contiguous()
    print(f"sweep on {x.shape[0]} images, iters={a.iters}, b0={a.b0}, bmax={a.bmax}")

    nets = None
    if a.transfer:
        import glob
        import surrogate as S
        paths = sorted(glob.glob(a.surrogate_glob))
        nets = S.load_surrogates(paths, device=a.device)
        print(f"loaded {len(nets)} surrogates: {[p.split('/')[-1] for p in paths]}")

    for s in (a.sched or [""]):
        print(f"\n--- schedule {s!r} ---")
        one_run(orc, x, y, s or "full", a.iters, a.b0, a.bmax, parse_sched(s),
                seeds=a.seeds, walk=a.walk, walk_k=a.walk_k,
                walk_before=a.walk_before, verbose=not a.quiet,
                geo=a.geo, geo_k=a.geo_k, alt=a.alt, nets=nets, grid=a.grid)


if __name__ == "__main__":
    main()
