"""
Driver for the hard-label black-box attack on the provided CNN.

    python src/run_attack.py --data data/p2_data --out adv_images

Stages (all of them see only predicted hard labels):
  0  free adversarial seeds (clean images of other classes)
  1  optional transfer stage: directions from locally trained substitute
     networks, radius chosen by the victim's labels
  2  HopSkipJump refinement of the boundary point
  3  boundary random walk (cheap per unit of progress near convergence)
  4  8-bit quantisation that the perturbation must survive
  5  integer-grid polishing (L-inf clip / prune / greedy shrink)
  6  multi-backend verification of exactly the PNGs that get written

Use `--no-transfer` for the pure query-only variant.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import attack as A
import metrics as M
import verify as V
from data import load_clean, save_adv_images
from oracle import load_oracle


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/p2_data")
    p.add_argument("--out", default="adv_images")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--subset", type=int, default=0,
                   help="images per class to attack (0 = all 100)")
    # initialisation / transfer
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--no-transfer", action="store_true")
    p.add_argument("--surrogate-glob", default="data/surrogates/*.pt")
    p.add_argument("--surrogate-epochs", type=int, default=20)
    p.add_argument("--prior-lam", type=float, default=0.0,
                   help=">0 biases HSJA probes toward the surrogate gradient")
    # HopSkipJump
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--b0", type=int, default=100)
    p.add_argument("--bmax", type=int, default=400)
    p.add_argument("--bsearch", type=int, default=20)
    # boundary refinement
    p.add_argument("--refine-rounds", type=int, default=1)
    p.add_argument("--geo", type=int, default=0)
    p.add_argument("--walk", type=int, default=1500)
    p.add_argument("--walk-k", type=int, default=0)
    p.add_argument("--rescue-rounds", type=int, default=0)
    p.add_argument("--rescue-frac", type=float, default=0.25)
    p.add_argument("--rescue-geo", type=int, default=600)
    p.add_argument("--rescue-walk", type=int, default=600)
    # finalisation
    p.add_argument("--grid-rounds", type=int, default=1)
    p.add_argument("--grid-geo", type=int, default=600)
    p.add_argument("--grid-walk", type=int, default=600)
    p.add_argument("--greedy-rounds", type=int, default=3)
    p.add_argument("--max-batch", type=int, default=2048)
    p.add_argument("--no-verify-cpu", action="store_true")
    p.add_argument("--verify-shrink", type=float, default=0.97,
                   help="an image is accepted only if the perturbation "
                        "shrunk to this fraction is still adversarial")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="run")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    dev = args.device
    t_start = time.time()
    model_path = os.path.join(args.data, "cnn_model.pt")

    oracle = load_oracle(model_path, device=dev, batch_size=args.max_batch)
    ipc = args.subset if args.subset > 0 else 100
    x01, y, names = load_clean(args.data, device=dev, img_per_class=ipc)
    n = x01.shape[0]
    print(f"[data] {n} images on {dev}")

    pred = oracle.predict(x01)
    acc = (pred == y).float().mean().item()
    print(f"[check] clean accuracy {100*acc:.2f} %")
    assert acc == 1.0, "clean images are expected to be classified correctly"
    rev = torch.arange(n - 1, -1, -1, device=dev)
    assert torch.equal(oracle.predict(x01[rev]).flip(0), pred), \
        "oracle is not batch-independent"
    print("[check] oracle is deterministic and batch-independent")
    oracle.reset_counter()
    q_marks = {}

    # ---------------------------------------------------------- stage 0: seeds
    seeds = A.init_seeds(x01, y, n_seeds=args.seeds)
    qt = A.BestTracker(x01)                       # quantised: decides output
    s0 = oracle.quantize_pixels(seeds[:, 0])
    qt.update(s0, oracle.is_adversarial(s0, y))
    assert torch.isfinite(qt.dist).all(), "seeding failed"

    ct = A.BestTracker(x01)                       # continuous: refinement
    x_adv, d0 = A.best_initial_boundary_point(oracle, x01, y, seeds,
                                              args.bsearch, ct)
    print(f"[stage 0] nearest-image seeds -> boundary  "
          f"L2 median {(d0*255).median():.1f}   q/img {oracle.n_queries/n:.0f}")
    q_marks["seed_init"] = oracle.n_queries

    # ------------------------------------------------------- stage 1: transfer
    sampler = None
    if not args.no_transfer:
        import surrogate as S
        import transfer as T
        # Charge the one-off cost of labelling the public CIFAR-10 split with
        # the victim's own hard labels to the query budget, even when the
        # labels come from cache: 60,000 queries shared by 1,000 images.
        _, _, n_label_q = S.victim_labels(oracle=oracle, device=dev)
        nets, paths = S.train_or_load(oracle=oracle, device=dev,
                                      epochs=args.surrogate_epochs)
        print(f"[stage 1] {len(nets)} substitute networks "
              f"{[os.path.basename(p) for p in paths]}; "
              f"labelling cost {n_label_q:,} queries "
              f"({n_label_q/n:.0f} per attacked image)")
        pts, r = T.transfer_init(oracle, nets, x01, y, tracker=ct)
        hit = torch.isfinite(r)
        better = hit & (r < A.l2(x_adv - x01))
        x_adv[better] = pts[better]
        print(f"[stage 1] transfer improved {int(better.sum())}/{n} starts; "
              f"L2 median now {(A.l2(x_adv-x01)*255).median():.1f}   "
              f"q/img {oracle.n_queries/n:.0f}")
        if args.prior_lam > 0:
            sampler = T.prior_sampler(nets, lam=args.prior_lam)
    q_marks["transfer"] = oracle.n_queries

    # ----------------------------------------------------------- stage 2: HSJA
    def log(t, b, k, dist, q):
        if t % 5 == 0 or t == 1:
            print(f"    [hsja] it {t:3d}  B={b:4d}  L2 median "
                  f"{(dist*255).median():7.2f}   q/img {q/n:.0f}")

    if args.iters:
        A.hsja(oracle, x01, y, x_adv, iters=args.iters, b0=args.b0,
               b_max=args.bmax, bsearch_steps=args.bsearch, tracker=ct,
               max_batch=args.max_batch, sampler=sampler, log=log)
    print(f"[stage 2] HopSkipJump  L2 median {(ct.dist*255).median():.2f}   "
          f"q/img {oracle.n_queries/n:.0f}")
    q_marks["hsja"] = oracle.n_queries

    # ------------------------------------------------------ stage 3: refine
    for r in range(args.refine_rounds):
        if args.geo:
            A.geometric_walk(oracle, x01, y, ct.best.clone(), steps=args.geo,
                             tracker=ct, log_every=max(1, args.geo // 2))
        if args.walk:
            A.boundary_walk(oracle, x01, y, ct.best.clone(), steps=args.walk,
                            tracker=ct, subspace_k=args.walk_k,
                            log_every=max(1, args.walk // 2))
        print(f"[stage 3.{r}] boundary refine  "
              f"L2 median {(ct.dist*255).median():.2f}   "
              f"q/img {oracle.n_queries/n:.0f}")
    if args.rescue_rounds:
        A.rescue_worst(oracle, x01, y, ct, frac=args.rescue_frac,
                       geo_steps=args.rescue_geo, walk_steps=args.rescue_walk,
                       rounds=args.rescue_rounds)
        print(f"[stage 3r] rescue tail  L2 median {(ct.dist*255).median():.2f}  "
              f"mean {(ct.dist*255).mean():.2f}  max {(ct.dist*255).max():.2f}  "
              f"q/img {oracle.n_queries/n:.0f}")
    q_marks["refine"] = oracle.n_queries

    # ------------------------------------------------- stage 4/5: 8-bit output
    A.quantize_finalize(oracle, x01, y, ct.best, qt)
    print(f"[stage 4] 8-bit quantisation  L2 median {(qt.dist*255).median():.2f}   "
          f"q/img {oracle.n_queries/n:.0f}")
    if args.grid_rounds:
        A.refine_on_grid(oracle, x01, y, qt, geo_steps=args.grid_geo,
                         walk_steps=args.grid_walk, rounds=args.grid_rounds)
        print(f"[stage 4b] grid-space refine  "
              f"L2 median {(qt.dist*255).median():.2f}   "
              f"q/img {oracle.n_queries/n:.0f}")
    A.polish_uint8(oracle, x01, y, qt, greedy_rounds=args.greedy_rounds)
    print(f"[stage 5] integer-grid polish  L2 median {(qt.dist*255).median():.2f}   "
          f"q/img {oracle.n_queries/n:.0f}")
    q_marks["finalize"] = oracle.n_queries
    total_queries = oracle.n_queries

    # ------------------------------------------------- stage 6: verification
    x_final = oracle.quantize_pixels(qt.best)
    backends = V.make_backends(model_path, dev, args.max_batch,
                               cpu=not args.no_verify_cpu)
    x_final, verified = V.harden(backends, x01, y, x_final,
                                 shrink=args.verify_shrink)
    if not bool(verified.all()):      # last resort: the free clean-image seed
        bad = (~verified).nonzero(as_tuple=True)[0]
        x_final[bad] = oracle.quantize_pixels(seeds[bad, 0])
        verified = V.consensus_adversarial(backends, x_final, y)
    vq = sum(b.n_queries for b in backends)
    print(f"[stage 6] robustly adversarial on {len(backends)} backends x "
          f"{len(V.input_views(x01[:1], dev))} input representations, with a "
          f"{100*(1-args.verify_shrink):.0f}% radial margin: "
          f"{int(verified.sum())}/{n}   "
          f"({vq:,} verification queries, not charged to the attack budget)")

    # -------------------------------------------------------------- report
    r = M.report(x01, x_final, verified, n_queries=total_queries)
    r["queries_per_stage"] = {k: v for k, v in q_marks.items()}
    r["wall_clock_s"] = round(time.time() - t_start, 1)
    r["config"] = vars(args)
    print("\n" + "=" * 64)
    print(M.format_report(r))
    print("=" * 64)

    if not args.no_save:
        save_adv_images(x_final, names, args.out)
        print(f"[save] {n} images -> {args.out}/")
        # Re-read the files that were just written and re-check them, so the
        # headline number belongs to the artefacts actually being submitted.
        from make_figures import load_adv
        reread = load_adv(args.out, names, device=dev)
        assert torch.equal((reread * 255).round().to(torch.uint8),
                           (x_final * 255).round().to(torch.uint8)), \
            "saved PNGs differ from the verified tensors"
        ok_disk = V.consensus_adversarial(backends, reread, y)
        print(f"[save] re-read from disk: {int(ok_disk.sum())}/{n} still "
              f"adversarial on all {len(backends)} backends")
        r["success_rate_from_disk"] = ok_disk.float().mean().item() * 100
        os.makedirs("report", exist_ok=True)
        with open(f"report/metrics_{args.tag}.json", "w") as f:
            json.dump(r, f, indent=2, default=str)
    return r


if __name__ == "__main__":
    main()
