"""
Transfer stage: turn surrogate gradients into candidate *directions*, then let
the victim's hard labels decide how far along each one we actually have to go.

The surrogates never tell us whether an image is adversarial -- only the victim
does, through `Oracle.is_adversarial`.  The surrogate contributes a direction;
the oracle contributes the (minimal) radius.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attack as A
from attack import _b, l2


# ------------------------------------------------------------ radius search
def radius_search(oracle, x01, y, direction, grid, bsearch=10, tracker=None):
    """Smallest radius r on `grid` (then bisected) with `x + r*dir` adversarial.

    `direction` is unit-L2.  Returns (point, radius); radius is +inf where no
    grid value worked.
    """
    n = x01.shape[0]
    dev = x01.device
    hi = torch.full((n,), float("inf"), device=dev)
    for r in grid:
        todo = torch.isinf(hi).nonzero(as_tuple=True)[0]
        if todo.numel() == 0:
            break
        cand = (x01[todo] + r * direction[todo]).clamp(0, 1)
        ok = oracle.is_adversarial(cand, y[todo])
        if tracker is not None:
            tracker.update(cand, ok, todo)
        hi[todo[ok]] = r

    found = ~torch.isinf(hi)
    lo = torch.zeros(n, device=dev)
    # the previous grid point is a known-failing lower bound
    g = torch.tensor(list(grid), device=dev)
    for i in range(1, len(grid)):
        lo = torch.where(hi == g[i], g[i - 1], lo)

    for _ in range(bsearch):
        idx = found.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            break
        mid = 0.5 * (lo[idx] + hi[idx])
        cand = (x01[idx] + _b(mid) * direction[idx]).clamp(0, 1)
        ok = oracle.is_adversarial(cand, y[idx])
        if tracker is not None:
            tracker.update(cand, ok, idx)
        hi[idx] = torch.where(ok, mid, hi[idx])
        lo[idx] = torch.where(ok, lo[idx], mid)

    pts = (x01 + _b(torch.nan_to_num(hi, posinf=0.0)) * direction).clamp(0, 1)
    return pts, hi


# ------------------------------------------------------- candidate directions
def candidate_directions(nets, x01, y, chunk=250):
    """A small, diverse set of unit-L2 directions from the surrogate ensemble."""
    import surrogate as S
    specs = [("ens-ddn", None, "ddn", 0.0, 120),
             ("ens-l2", None, "l2", 1.0, 40),
             ("ens-linf", None, "linf", 16 / 255, 40),
             ("ens-grad", None, "grad", 0.0, 0)]
    for i in range(len(nets)):
        specs.append((f"net{i}-ddn", i, "ddn", 0.0, 80))

    out = []
    for name, which, kind, eps, steps in specs:
        sub = nets if which is None else [nets[which]]
        parts = []
        for s in range(0, x01.shape[0], chunk):
            xb, yb = x01[s:s + chunk], y[s:s + chunk]
            if kind == "grad":
                parts.append(S.ensemble_loss_grad(sub, xb, yb))
            elif kind == "ddn":
                parts.append(S.ensemble_ddn(sub, xb, yb, steps=steps)[0])
            else:
                parts.append(S.ensemble_pgd_direction(sub, xb, yb, eps=eps,
                                                      steps=steps, norm=kind))
        out.append((name, torch.cat(parts)))
    return out


def transfer_init(oracle, nets, x01, y, tracker=None, grid=None, bsearch=12,
                  log=print):
    """Line-search every surrogate direction against the victim and keep the
    closest adversarial point found, per image."""
    if grid is None:
        # unit-L2 radii in [0,1] pixel space; 1.0 here == 255 on the 0-255 scale
        grid = (0.05, 0.1, 0.18, 0.3, 0.5, 0.85, 1.4, 2.4, 4.0, 6.5)
    if tracker is None:
        tracker = A.BestTracker(x01)

    best = None
    best_d = torch.full((x01.shape[0],), float("inf"), device=x01.device)
    for name, d in candidate_directions(nets, x01, y):
        pts, r = radius_search(oracle, x01, y, d, grid, bsearch, tracker)
        ok = torch.isfinite(r)
        if log:
            rr = r[ok] * 255
            log(f"    [{name:10s}] hit {100*ok.float().mean():5.1f}%  "
                f"radius median {rr.median() if ok.any() else float('nan'):7.1f} "
                f"(0-255)   q={oracle.n_queries:,}")
        upd = ok & (r < best_d)
        if best is None:
            best = pts.clone()
            best_d = torch.where(upd, r, best_d)
            best[~upd] = x01[~upd]
        else:
            best[upd] = pts[upd]
            best_d[upd] = r[upd]
    return best, best_d


# ------------------------------------------- prior-biased probe distribution
def prior_sampler(nets, lam=0.5):
    """Probe directions biased towards the surrogate-ensemble gradient.

    u = normalise( sqrt(lam) * g + sqrt(1-lam) * random ), the mixture used by
    prior-guided random gradient-free methods (P-RGF, Cheng et al. 2019).
    """
    import surrogate as S

    def sample(m, n_samples, shape, device, x_adv, y, k=0, generator=None):
        u = A.sample_directions(m, n_samples, shape, device, k=k,
                                generator=generator)
        if lam <= 0:
            return u
        g = S.ensemble_loss_grad(nets, x_adv, y).unsqueeze(1)
        u = (lam ** 0.5) * g + ((1 - lam) ** 0.5) * u
        return u / (u.flatten(2).norm(dim=2).view(m, n_samples, 1, 1, 1) + 1e-12)

    return sample
