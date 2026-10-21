"""
Decision-based (hard-label) black-box attack, vectorised over the whole image
set so that thousands of oracle queries are answered in a handful of batched
forward passes.

Stages, in the order a run uses them
------------------------------------
`init_seeds`              zero-query adversarial starting points: clean images
                          of other classes.
`best_initial_boundary_point` / `line_search_boundary`
                          bisect onto the decision boundary.
`hsja`                    HopSkipJump (Chen, Jordan & Wainwright, IEEE S&P
                          2020): Monte-Carlo estimate of the boundary normal
                          from *labels only*, geometric step, re-projection.
`geometric_walk`          SurFree-style circular step (Maho, Furon & Le
                          Merrer, CVPR 2021): one accepted query shrinks the
                          distance by cos(t).
`boundary_walk`           random walk along the boundary (Brendel, Rauber &
                          Bethge, ICLR 2018): lateral + inward trial in one
                          batched call.
`rescue_worst`            spend extra passes only on the images still
                          furthest from the boundary.
`quantize_finalize`       make the perturbation survive 8-bit PNG rounding.
`refine_on_grid`          re-run the walks against the *rounded* objective,
                          which is what actually gets graded.
`polish_uint8`            shrink on the integer grid: L-inf clipping,
                          pruning, greedy per-pixel shrinking.

`BestTracker` keeps, per image, the closest point the oracle has confirmed, so
no stage can ever make a result worse and the success rate cannot regress.

Only `Oracle.is_adversarial` / `Oracle.predict` are ever called, i.e. only the
predicted hard label is used.
"""

from __future__ import annotations

import math

import torch


# --------------------------------------------------------------------- helpers
def l2(v: torch.Tensor) -> torch.Tensor:
    """Per-image L2 norm of a (N,C,H,W) tensor -> (N,)."""
    return v.flatten(1).norm(dim=1)


def linf(v: torch.Tensor) -> torch.Tensor:
    return v.flatten(1).abs().max(dim=1).values


def _b(v: torch.Tensor) -> torch.Tensor:
    """(N,) -> (N,1,1,1) for broadcasting against images."""
    return v.view(-1, 1, 1, 1)


class BestTracker:
    """Keeps, per image, the closest point that the oracle *verified* as
    adversarial.  Guarantees monotone improvement and a 100% success rate as
    long as it is seeded with a verified adversarial point."""

    def __init__(self, x_orig: torch.Tensor, quantize: bool = False):
        self.x_orig = x_orig
        self.best = x_orig.clone()
        self.dist = torch.full((x_orig.shape[0],), float("inf"), device=x_orig.device)
        # When True, candidates are snapped to the 8-bit grid before being
        # measured and stored, so the tracked distance is the distance of the
        # image that will actually be written to the .png.  Pair this with an
        # Oracle constructed with quantize=True, or the stored point and the
        # point the oracle answered about are not the same image.
        self.quantize = quantize

    def update(self, x_cand: torch.Tensor, is_adv: torch.Tensor,
               idx: torch.Tensor | None = None) -> torch.Tensor:
        """`x_cand`/`is_adv` are aligned with `idx` (default: all images)."""
        if idx is None:
            idx = torch.arange(x_cand.shape[0], device=x_cand.device)
        if self.quantize:
            x_cand = torch.round(x_cand.clamp(0, 1) * 255.0) / 255.0
        d = l2(x_cand - self.x_orig[idx])
        take = is_adv & (d < self.dist[idx])
        sel = idx[take]
        if sel.numel():
            self.best[sel] = x_cand[take]
            self.dist[sel] = d[take]
        return take


# ------------------------------------------------------------ initialisation
def init_seeds(x01: torch.Tensor, y: torch.Tensor, n_seeds: int = 4):
    """Zero-query adversarial starting points.

    Every clean image is classified correctly by the victim model (stated in
    the provided notebook and re-verified in `run_attack.py`), so *any* clean
    image whose label differs from `y[i]` is, by construction, an adversarial
    example for image `i`.  Picking the nearest such images in pixel space
    gives a strong starting point for free -- no queries are spent on the
    usual "sample random noise until it is misclassified" phase.

    Returns a (N, n_seeds, C, H, W) tensor of candidate seeds.
    """
    n = x01.shape[0]
    flat = x01.flatten(1)
    seeds = torch.empty(n, n_seeds, *x01.shape[1:], device=x01.device)
    # chunk the distance matrix to stay inside memory
    step = 128
    for s in range(0, n, step):
        e = min(n, s + step)
        d = torch.cdist(flat[s:e], flat)                      # (m, n)
        same = y[s:e].view(-1, 1) == y.view(1, -1)
        d = d.masked_fill(same, float("inf"))
        idx = d.topk(n_seeds, dim=1, largest=False).indices   # (m, n_seeds)
        seeds[s:e] = x01[idx]
    return seeds


def best_initial_boundary_point(oracle, x01, y, seeds, bsearch_steps=10,
                               tracker: BestTracker | None = None):
    """Binary-search every seed onto the boundary, keep the closest one."""
    n, k = seeds.shape[0], seeds.shape[1]
    best = None
    best_d = torch.full((n,), float("inf"), device=x01.device)
    for j in range(k):
        pt, _ = line_search_boundary(oracle, x01, seeds[:, j], y, bsearch_steps,
                                     tracker=tracker)
        d = l2(pt - x01)
        if best is None:
            best, best_d = pt, d
        else:
            upd = d < best_d
            best[upd] = pt[upd]
            best_d[upd] = d[upd]
    return best, best_d


# ------------------------------------------------------------- boundary search
def line_search_boundary(oracle, x_orig, x_adv, y, steps=10,
                         tracker: BestTracker | None = None):
    """Smallest alpha in (0,1] with `x_orig + alpha*(x_adv-x_orig)` adversarial.

    `x_adv` must be adversarial (alpha = 1 is the known-good bracket).
    Returns (point_on_boundary, alpha).
    """
    n = x_orig.shape[0]
    lo = torch.zeros(n, device=x_orig.device)
    hi = torch.ones(n, device=x_orig.device)
    diff = x_adv - x_orig
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        z = (x_orig + _b(mid) * diff).clamp(0, 1)
        adv = oracle.is_adversarial(z, y)
        if tracker is not None:
            tracker.update(z, adv)
        hi = torch.where(adv, mid, hi)
        lo = torch.where(adv, lo, mid)
    return (x_orig + _b(hi) * diff).clamp(0, 1), hi


# -------------------------------------------------- label-only normal estimate
def sample_directions(m, n_samples, shape, device, k=0, generator=None):
    """Unit-norm random probe directions.

    `k == 0` gives i.i.d. Gaussian directions in the full 3*32*32 space
    (plain HopSkipJump).  `k > 0` draws them in a (3*k*k)-dimensional
    *smooth* subspace -- low-resolution noise bilinearly upsampled to full
    resolution -- which is the query-efficiency trick of QEBA (Li et al.,
    CVPR 2020).  Estimating a direction from +-1 labels needs O(dim) samples,
    so shrinking the dimension from 3072 to e.g. 3*8*8 = 192 buys a far
    better direction per query.
    """
    c, h, w = shape
    if k and k < min(h, w):
        v = torch.randn(m * n_samples, c, k, k, device=device, generator=generator)
        u = torch.nn.functional.interpolate(v, size=(h, w), mode="bilinear",
                                            align_corners=False)
        u = u.view(m, n_samples, c, h, w)
    else:
        u = torch.randn(m, n_samples, c, h, w, device=device, generator=generator)
    return u / (u.flatten(2).norm(dim=2).view(m, n_samples, 1, 1, 1) + 1e-12)


def estimate_gradient(oracle, x_adv, y, delta, n_samples, max_batch=16384,
                      subspace_k=0, generator=None, sampler=None):
    """Monte-Carlo estimate of the (outward) boundary normal at `x_adv`.

    Uses only the +-1 "is it adversarial" answers, as in HopSkipJump.
    """
    n = x_adv.shape[0]
    shape = x_adv.shape[1:]
    grads = torch.empty_like(x_adv)
    group = max(1, max_batch // n_samples)
    for s in range(0, n, group):
        e = min(n, s + group)
        m = e - s
        xa = x_adv[s:e]
        dl = delta[s:e].view(m, 1, 1, 1, 1)
        if sampler is None:
            u = sample_directions(m, n_samples, shape, x_adv.device,
                                  k=subspace_k, generator=generator)
        else:
            u = sampler(m, n_samples, shape, x_adv.device, xa, y[s:e],
                        k=subspace_k, generator=generator)
        pert = (xa.unsqueeze(1) + dl * u).clamp(0, 1)
        # account for the clipping, as in the reference implementation
        u = (pert - xa.unsqueeze(1)) / dl
        yy = y[s:e].repeat_interleave(n_samples)
        adv = oracle.is_adversarial(pert.reshape(m * n_samples, *shape), yy)
        f = adv.view(m, n_samples).float() * 2.0 - 1.0
        fbar = f.mean(dim=1, keepdim=True)
        w = f - fbar
        degenerate = fbar.abs() >= 1.0 - 1e-6        # all-adv or all-clean
        w = torch.where(degenerate, fbar.sign(), w)
        grads[s:e] = (w.view(m, n_samples, 1, 1, 1) * u).mean(dim=1)
    return grads / (_b(l2(grads)) + 1e-12)


def geometric_step(oracle, x_adv, grad, y, eps, max_halvings=12,
                   tracker: BestTracker | None = None):
    """Move along `grad` with a geometrically decreasing step until the point
    is adversarial again.  Only the still-unsuccessful images are queried."""
    n = x_adv.shape[0]
    eps = eps.clone()
    out = x_adv.clone()
    pending = torch.ones(n, dtype=torch.bool, device=x_adv.device)
    for _ in range(max_halvings):
        idx = pending.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            break
        cand = (x_adv[idx] + _b(eps[idx]) * grad[idx]).clamp(0, 1)
        ok = oracle.is_adversarial(cand, y[idx])
        if tracker is not None:
            tracker.update(cand, ok, idx)
        out[idx[ok]] = cand[ok]
        pending[idx[ok]] = False
        eps[idx[~ok]] *= 0.5
    return out, ~pending


# ------------------------------------------------------------------ HopSkipJump
def hsja(oracle, x_orig, y, x_adv, iters=30, b0=100, b_max=1024,
         bsearch_steps=20, gamma=1.0, tracker: BestTracker | None = None,
         max_batch=16384, subspace_schedule=None, sampler=None, log=None):
    """HopSkipJump refinement of an adversarial point, L2 objective.

    `bsearch_steps` must be fine enough that the re-projected point sits much
    closer to the boundary than the probe radius `delta`; otherwise the +-1
    answers saturate and the estimated normal is useless.  For d = 3072 the
    HopSkipJump tolerance theta = 1/d^1.5 needs ~18 bisection steps.

    `subspace_schedule` is a list of (last_iteration, k) pairs; `k = 0` means
    the full pixel space.  Coarse subspaces make fast progress early, the full
    space refines the last bit of detail.
    """
    d = x_orig[0].numel()
    theta = gamma / (d ** 1.5)
    if tracker is None:
        tracker = BestTracker(x_orig)
    if subspace_schedule is None:
        subspace_schedule = [(iters, 0)]

    def k_for(t):
        for last, k in subspace_schedule:
            if t <= last:
                return k
        return subspace_schedule[-1][1]

    for t in range(1, iters + 1):
        dist = l2(x_adv - x_orig)
        delta = (math.sqrt(d) * theta * dist).clamp(min=1e-7)
        n_samples = int(min(b0 * math.sqrt(t), b_max))
        k = k_for(t)

        grad = estimate_gradient(oracle, x_adv, y, delta, n_samples,
                                 max_batch=max_batch, subspace_k=k,
                                 sampler=sampler)
        eps = dist / math.sqrt(t)
        moved, _ = geometric_step(oracle, x_adv, grad, y, eps, tracker=tracker)
        x_adv, _ = line_search_boundary(oracle, x_orig, moved, y, bsearch_steps,
                                        tracker=tracker)
        tracker.update(x_adv, oracle.is_adversarial(x_adv, y))

        if log is not None:
            log(t, n_samples, k, tracker.dist, oracle.n_queries)

    return tracker


# ------------------------------------------------- boundary walk (fine polish)
def boundary_walk(oracle, x_orig, y, x_adv, steps=400, tracker=None,
                  subspace_k=0, sph_step=0.12, src_step=0.03,
                  sph_bounds=(0.005, 0.6), src_bounds=(1e-4, 0.4),
                  adapt=1.25, generator=None, log_every=0):
    """Random-walk refinement along the decision boundary (Brendel, Rauber &
    Bethge, ICLR 2018), vectorised over the image set.

    Near convergence this is far cheaper per unit of progress than another
    HopSkipJump iteration: one *orthogonal* trial plus one *inward* trial per
    image per step, both answered in a single batched oracle call, instead of
    several hundred probes for a new normal estimate.

    Per-image step sizes adapt to the observed acceptance.
    """
    if tracker is None:
        tracker = BestTracker(x_orig)
    n = x_orig.shape[0]
    dev = x_orig.device
    sph = torch.full((n,), sph_step, device=dev)
    src = torch.full((n,), src_step, device=dev)
    n_lat = n_in = 0

    for i in range(steps):
        diff = x_adv - x_orig
        dist = l2(diff).clamp(min=1e-9)

        # --- orthogonal (same-distance) trial --------------------------------
        u = sample_directions(n, 1, x_orig.shape[1:], dev, k=subspace_k,
                              generator=generator).squeeze(1)
        u = u * _b(dist * sph)
        # remove the radial component, then rescale back onto the sphere
        radial = (u * diff).flatten(1).sum(1) / dist ** 2
        u = u - _b(radial) * diff
        cand = x_orig + (diff + u)
        cand = x_orig + (cand - x_orig) * _b(dist / l2(cand - x_orig).clamp(min=1e-9))
        cand = cand.clamp(0, 1)

        # --- inward trial (does not depend on the outcome above, so both
        #     trials go to the oracle in one batch) --------------------------
        cand_in = (cand + _b(src) * (x_orig - cand)).clamp(0, 1)

        adv = oracle.is_adversarial(torch.cat([cand, cand_in], 0),
                                    torch.cat([y, y], 0))
        ok_lat, ok_in = adv[:n], adv[n:]
        tracker.update(cand, ok_lat)
        tracker.update(cand_in, ok_in)

        took_in = ok_lat & ok_in
        took_lat = ok_lat & ~ok_in
        x_adv = torch.where(_b(took_in), cand_in,
                            torch.where(_b(took_lat), cand, x_adv))

        # --- adapt ----------------------------------------------------------
        src = torch.where(took_in, src * adapt, src / adapt).clamp(*src_bounds)
        sph = torch.where(ok_lat, sph * 1.05, sph / 1.05).clamp(*sph_bounds)
        n_in += int(took_in.sum()); n_lat += int(took_lat.sum())

        if log_every and (i + 1) % log_every == 0:
            print(f"    [walk] {i+1:4d}/{steps}  L2med "
                  f"{(l2(x_adv-x_orig)*255).median():7.2f}  "
                  f"inward-acc {n_in/(n*(i+1)):.2f}  lateral-acc "
                  f"{n_lat/(n*(i+1)):.2f}  q={oracle.n_queries:,}")
    return tracker, x_adv


# --------------------------------------------- geometric (circular) refinement
def geometric_walk(oracle, x_orig, y, x_adv, steps=400, tracker=None,
                   thetas=(0.55, 0.3, 0.15), subspace_k=0, adapt=1.1,
                   theta_bounds=(0.02, 1.2), project_every=50, project_steps=7,
                   generator=None, log_every=0):
    """SurFree-style refinement (Maho, Furon & Le Merrer, CVPR 2021).

    With `u` the unit vector from the clean image to the current boundary
    point and `v` a random unit vector orthogonal to it,

        z(t) = x + d*cos(t) * ( cos(t)*u + sin(t)*v )

    satisfies ||z(t) - x|| = d*cos(t): *one* accepted query shrinks the
    distance by the factor cos(t).  Where the boundary curves away from the
    clean image this is far more productive per query than a small random
    walk, so several `t` values are tried per step in a single batch and the
    largest accepted one (the biggest shrink) is kept.
    """
    if tracker is None:
        tracker = BestTracker(x_orig)
    n = x_orig.shape[0]
    dev = x_orig.device
    k = len(thetas)
    scale = torch.ones(n, device=dev)        # per-image multiplier on `thetas`
    n_acc = 0

    for i in range(steps):
        diff = x_adv - x_orig
        dist = l2(diff).clamp(min=1e-9)
        u = diff / _b(dist)

        v = sample_directions(n, 1, x_orig.shape[1:], dev, k=subspace_k,
                              generator=generator).squeeze(1)
        v = v - _b((v * u).flatten(1).sum(1)) * u
        v = v / _b(l2(v).clamp(min=1e-12))

        cands, dists = [], []
        for t0 in thetas:
            t = (scale * t0).clamp(*theta_bounds)
            ct, st = torch.cos(t), torch.sin(t)
            z = x_orig + _b(dist * ct) * (_b(ct) * u + _b(st) * v)
            cands.append(z.clamp(0, 1))
            dists.append(dist * ct)
        batch = torch.cat(cands, 0)
        adv = oracle.is_adversarial(batch, y.repeat(k)).view(k, n)
        for j in range(k):
            tracker.update(cands[j], adv[j])

        # keep the largest accepted theta == the smallest resulting distance
        taken = torch.zeros(n, dtype=torch.bool, device=dev)
        for j in range(k):                   # thetas are in decreasing order
            pick = adv[j] & ~taken
            x_adv = torch.where(_b(pick), cands[j], x_adv)
            taken |= pick
        n_acc += int(taken.sum())

        # grow the search angle when it works, shrink it when nothing does
        scale = torch.where(adv[0], scale * adapt, scale)
        scale = torch.where(~taken, scale / adapt, scale).clamp(0.05, 20.0)

        # an accepted step lands *inside* the adversarial region, so every so
        # often bisect back along the radius to the smallest radius that still
        # works -- a few queries for a guaranteed reduction
        if project_every and (i + 1) % project_every == 0:
            x_adv, _ = line_search_boundary(oracle, x_orig, x_adv, y,
                                            project_steps, tracker=tracker)

        if log_every and (i + 1) % log_every == 0:
            print(f"    [geo ] {i+1:4d}/{steps}  L2med "
                  f"{(l2(x_adv-x_orig)*255).median():7.2f}  acc "
                  f"{n_acc/(n*(i+1)):.2f}  q={oracle.n_queries:,}")
    return tracker, x_adv


# --------------------------------------------------- budget re-allocation
def rescue_worst(oracle, x_orig, y, tracker: BestTracker, frac=0.25,
                 geo_steps=600, walk_steps=600, rounds=2, log=print):
    """Spend extra queries only on the images that are still furthest away.

    After the main refinement the distance distribution has a tail: a few
    images sit much further from the boundary than the median, and those are
    exactly the ones where the noise is visible.  Because the oracle is queried
    in batches, running more refinement on a *subset* costs proportionally
    fewer queries, so this is a cheap way to pull the tail in.
    """
    n = x_orig.shape[0]
    k = max(1, int(round(frac * n)))
    for r in range(rounds):
        idx = torch.argsort(tracker.dist, descending=True)[:k]
        sub = BestTracker(x_orig[idx])
        sub.best = tracker.best[idx].clone()
        sub.dist = tracker.dist[idx].clone()
        start = sub.best.clone()
        if geo_steps:
            _, start = geometric_walk(oracle, x_orig[idx], y[idx], start,
                                      steps=geo_steps, tracker=sub)
        if walk_steps:
            _, start = boundary_walk(oracle, x_orig[idx], y[idx],
                                     sub.best.clone(), steps=walk_steps,
                                     tracker=sub)
        improved = sub.dist < tracker.dist[idx]
        sel = idx[improved]
        tracker.best[sel] = sub.best[improved]
        tracker.dist[sel] = sub.dist[improved]
        if log:
            d = tracker.dist * 255
            log(f"    [rescue {r}] worst {k} images: median {d.median():.2f}  "
                f"mean {d.mean():.2f}  max {d.max():.2f}   "
                f"q={oracle.n_queries:,}")
    return tracker


# ------------------------------------------------------- 8-bit quantisation
def quantize_finalize(oracle, x_orig, y, x_adv, tracker: BestTracker,
                      n_dither=8, max_scale=24.0, n_grid=14, bisect=8,
                      generator=None):
    """Turn a continuous adversarial point into one that is *still*
    adversarial after being written to an 8-bit PNG.

    This matters more than it sounds: a converged perturbation is only about
    one pixel level per channel in RMS, i.e. the same order as the
    quantisation step itself, so a point sitting on the continuous boundary
    rounds to the wrong side roughly half the time.  Three families of
    candidates are tried and the closest one the oracle confirms is kept:

      * dithered (stochastic, unbiased) roundings of `x_adv` -- these are free
        in the sense that they do not move away from the clean image;
      * the smallest scaling `c` of the perturbation whose *rounding* is still
        adversarial, found by a geometric sweep followed by bisection. The
        sweep reaches far enough (`max_scale`) that it cannot fail silently
        and leave the image on a distant fallback;
      * dithered roundings at that scale.
    """
    q = oracle.quantize_pixels
    n = x_orig.shape[0]
    dev = x_orig.device
    diff = x_adv - x_orig

    def dithers(base, k):
        for _ in range(k):
            noise = torch.rand(base.shape, device=dev, generator=generator)
            cand = torch.floor(base.clamp(0, 1) * 255.0 + noise).clamp(0, 255) / 255.0
            tracker.update(cand, oracle.is_adversarial(cand, y))

    dithers(x_adv, n_dither)

    # smallest scale whose rounding survives: sweep up, then bisect down.
    # c = 0 is the clean image, which is by definition not adversarial, so it
    # is a valid failing lower bound to bisect against.
    lo = torch.zeros(n, device=dev)
    hi = torch.full((n,), float("inf"), device=dev)
    for i in range(n_grid):
        c = max_scale ** (i / (n_grid - 1))            # 1.0 ... max_scale
        todo = torch.isinf(hi).nonzero(as_tuple=True)[0]
        if todo.numel() == 0:
            break
        cand = q((x_orig[todo] + c * diff[todo]).clamp(0, 1))
        ok = oracle.is_adversarial(cand, y[todo])
        tracker.update(cand, ok, todo)
        hi[todo[ok]] = c
        lo[todo[~ok]] = c

    for _ in range(bisect):
        idx = torch.isfinite(hi).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            break
        mid = 0.5 * (lo[idx] + hi[idx])
        cand = q((x_orig[idx] + _b(mid) * diff[idx]).clamp(0, 1))
        ok = oracle.is_adversarial(cand, y[idx])
        tracker.update(cand, ok, idx)
        hi[idx] = torch.where(ok, mid, hi[idx])
        lo[idx] = torch.where(ok, lo[idx], mid)

    scale = torch.nan_to_num(hi, posinf=1.0)
    dithers((x_orig + _b(0.9 * scale) * diff).clamp(0, 1), n_dither)
    return tracker


def refine_on_grid(oracle, x_orig, y, tracker: BestTracker, geo_steps=600,
                   walk_steps=600, rounds=1, log=print):
    """Refinement that optimises the quantised objective directly.

    The earlier stages work in continuous pixel space, where the attainable
    optimum can sit *below* the 8-bit grid; what actually gets graded is the
    rounded image. Here the oracle rounds every query onto the grid and the
    tracker measures rounded candidates, so each accepted move is a genuine
    improvement of the submitted .png.
    """
    was = oracle.quantize
    oracle.quantize = True
    tracker.quantize = True
    try:
        for r in range(rounds):
            if geo_steps:
                geometric_walk(oracle, x_orig, y, tracker.best.clone(),
                               steps=geo_steps, tracker=tracker)
            if walk_steps:
                boundary_walk(oracle, x_orig, y, tracker.best.clone(),
                              steps=walk_steps, tracker=tracker)
            if log:
                d = tracker.dist * 255
                log(f"    [grid {r}] median {d.median():.2f}  mean {d.mean():.2f}"
                    f"  max {d.max():.2f}   q={oracle.n_queries:,}")
    finally:
        oracle.quantize = was
    return tracker


# ------------------------------------------------- integer-grid minimisation
def polish_uint8(oracle, x_orig, y, tracker: BestTracker, linf_steps=9,
                 prune_steps=9, greedy_rounds=3, greedy_groups=6,
                 generator=None):
    """Shrink the perturbation directly on the integer pixel grid.

    Every candidate is a valid PNG and every acceptance is oracle-verified,
    so this can only improve the result.
    """
    q = oracle.quantize_pixels
    dev = x_orig.device
    n = x_orig.shape[0]

    def best_delta():
        return (tracker.best - x_orig) * 255.0            # integer-valued

    # 1) smallest L-inf clip radius that keeps the image adversarial ---------
    delta = best_delta()
    hi = delta.abs().flatten(1).max(dim=1).values.clamp(min=1.0)
    lo = torch.zeros(n, device=dev)
    for _ in range(linf_steps):
        mid = torch.floor(0.5 * (lo + hi))
        cand = q(x_orig + torch.clamp(delta, -_b(mid), _b(mid)) / 255.0)
        adv = oracle.is_adversarial(cand, y)
        tracker.update(cand, adv)
        hi = torch.where(adv, mid, hi)
        lo = torch.where(adv, mid + 1.0, lo)
        hi = torch.maximum(hi, lo)

    # 2) prune small-magnitude components -----------------------------------
    delta = best_delta()
    lo = torch.zeros(n, device=dev)
    hi = delta.abs().flatten(1).max(dim=1).values.clamp(min=1.0)
    for _ in range(prune_steps):
        mid = torch.floor(0.5 * (lo + hi))
        pruned = torch.where(delta.abs() <= _b(mid), torch.zeros_like(delta), delta)
        cand = q(x_orig + pruned / 255.0)
        adv = oracle.is_adversarial(cand, y)
        tracker.update(cand, adv)
        lo = torch.where(adv, mid + 1.0, lo)
        hi = torch.where(adv, hi, mid)
        hi = torch.maximum(hi, lo)

    # 3) greedy group-wise shrinking by one quantisation step ---------------
    for _ in range(greedy_rounds):
        delta = best_delta()
        order = torch.rand(delta.shape, device=dev, generator=generator)
        bucket = (order * greedy_groups).long().clamp(0, greedy_groups - 1)
        for g in range(greedy_groups):
            mask = (bucket == g)
            shrunk = torch.where(mask, delta - delta.sign(), delta)
            cand = q(x_orig + shrunk / 255.0)
            adv = oracle.is_adversarial(cand, y)
            took = tracker.update(cand, adv)
            delta = torch.where(_b(took.float()) > 0, shrunk, delta)
    return tracker
