"""Pipeline driver that is inlined into the submission notebook.

It uses only the names defined by the inlined attack / substitute-model code
(l2, _b, BestTracker, init_seeds, hsja, geometric_walk, ..., transfer_init)
plus `torch`, so the whole thing runs as a single notebook cell.
"""
import torch


def png_roundtrip(x01):
    """The tensor an evaluator gets after loading our .png files.

    `transforms.ToTensor()` (and any PIL-based loader) produces an HWC uint8
    array that is permuted to CHW, i.e. a *non-contiguous* view.  Feeding a
    differently laid-out tensor to cuDNN can select a different kernel and
    therefore return the other label for a point sitting on the decision
    boundary.  Verifying this exact representation -- not just the contiguous
    tensor we computed -- is what makes the success rate measured here the
    same one measured after the files are read back.
    """
    a = (x01.clamp(0, 1) * 255.0).round().to(torch.uint8)
    nhwc = a.permute(0, 2, 3, 1).contiguous()
    return nhwc.permute(0, 3, 1, 2).float().div(255.0)


def run_attack(oracle, x01, y, cfg):
    """Return (adversarial images on the 8-bit grid, verified mask, query log).

    `x01` are the clean images in [0,1] pixel space, `y` their true labels.
    """
    n = x01.shape[0]
    dev = x01.device
    torch.manual_seed(cfg.get("seed", 0))
    q_log = {}

    def show(tag, dist, extra=""):
        d = dist * 255
        print(f"{tag:26s} L2 median {d.median():8.2f}  mean {d.mean():8.2f}"
              f"  max {d.max():8.2f}   q/img {oracle.n_queries/n:7.0f} {extra}")

    # ---- stage 0: free adversarial seeds (0 queries) -------------------
    # The victim classifies all 1000 provided images correctly, so a clean
    # image carrying a different label is *already* adversarial for this one.
    # The nearest such images are strong starting points that cost nothing.
    seeds = init_seeds(x01, y, n_seeds=cfg["seeds"])

    # The quantised tracker holds the images that will be saved.  Seeding it
    # with a clean 8-bit image of another class puts a 100% success-rate floor
    # under the run: later stages can only replace it with something closer
    # that the oracle has itself confirmed.
    qt = BestTracker(x01)
    s0 = oracle.quantize_pixels(seeds[:, 0])
    qt.update(s0, oracle.is_adversarial(s0, y))

    # ---- stage 1: bisect onto the decision boundary --------------------
    ct = BestTracker(x01)                        # continuous-space refinement
    x_adv, _ = best_initial_boundary_point(oracle, x01, y, seeds,
                                           cfg["bsearch"], ct)
    show("[0-1] seeds -> boundary", ct.dist)
    q_log["init"] = oracle.n_queries

    # ---- stage 1b: substitute-model directions -------------------------
    if cfg.get("transfer"):
        print("[1b]  training substitute networks on victim-queried labels ...")
        nets, _ = train_or_load(oracle=oracle, epochs=cfg["surrogate_epochs"],
                                device=dev)
        pts, rad = transfer_init(oracle, nets, x01, y, tracker=ct)
        better = torch.isfinite(rad) & (rad < l2(x_adv - x01))
        x_adv[better] = pts[better]
        show("[1b]  transfer directions", ct.dist,
             f"({int(better.sum())}/{n} improved)")
    q_log["transfer"] = oracle.n_queries

    # ---- stage 2: HopSkipJump (optional) -------------------------------
    def _log(t, b, k, dist, q):
        if t % 5 == 0 or t == 1:
            print(f"      [hsja] it {t:3d} B={b:4d}  L2 median "
                  f"{(dist*255).median():8.2f}   q/img {q/n:7.0f}")

    if cfg["iters"]:
        hsja(oracle, x01, y, x_adv, iters=cfg["iters"], b0=cfg["b0"],
             b_max=cfg["bmax"], bsearch_steps=cfg["bsearch"], tracker=ct,
             max_batch=cfg["max_batch"], log=_log)
        show("[2]   HopSkipJump", ct.dist)
    q_log["hsja"] = oracle.n_queries

    # ---- stage 3: boundary refinement ----------------------------------
    for r in range(cfg["refine_rounds"]):
        if cfg["geo_steps"]:
            geometric_walk(oracle, x01, y, ct.best.clone(),
                           steps=cfg["geo_steps"], tracker=ct)
        if cfg["walk_steps"]:
            boundary_walk(oracle, x01, y, ct.best.clone(),
                          steps=cfg["walk_steps"], tracker=ct)
        show(f"[3.{r}] boundary refine", ct.dist)
    if cfg.get("rescue_rounds"):
        rescue_worst(oracle, x01, y, ct, frac=cfg["rescue_frac"],
                     geo_steps=cfg["rescue_geo"], walk_steps=cfg["rescue_walk"],
                     rounds=cfg["rescue_rounds"], log=None)
        show("[3r]  rescue the tail", ct.dist)
    q_log["refine"] = oracle.n_queries

    # ---- stage 4: survive 8-bit quantisation ---------------------------
    quantize_finalize(oracle, x01, y, ct.best, qt)
    show("[4]   8-bit quantisation", qt.dist)

    # ---- stage 4b: refine the quantised objective directly -------------
    if cfg.get("grid_rounds"):
        refine_on_grid(oracle, x01, y, qt, geo_steps=cfg["grid_geo"],
                       walk_steps=cfg["grid_walk"], rounds=cfg["grid_rounds"],
                       log=None)
        show("[4b]  grid-space refine", qt.dist)

    # ---- stage 5: integer-grid polish ----------------------------------
    polish_uint8(oracle, x01, y, qt, greedy_rounds=cfg["greedy_rounds"])
    show("[5]   integer-grid polish", qt.dist)
    q_log["finalize"] = oracle.n_queries

    # ---- stage 6: verification that survives a change of machine -------
    # Points on the boundary are separated from the correct label by a
    # numerically tiny margin, so batch size, tensor layout and even the last
    # bit of the division by 255 can flip them back.  Accept only images that
    # keep the wrong label with room to spare: see `robustly_adversarial`.
    x_final = oracle.quantize_pixels(qt.best)

    def _loader(path, device="cpu", batch_size=2048):
        m = oracle.model if str(device) == str(dev) else None
        if m is None:
            m = torch.jit.load(path, map_location=device)
            m.eval()
        return Oracle(m, MEAN, STD, device=device, batch_size=batch_size)

    backends = make_backends(cfg["model_path"], dev, cfg["max_batch"],
                             loader=_loader)
    x_final, verified = harden(backends, x01, y, x_final,
                               shrink=cfg.get("verify_shrink", 0.97))
    if not bool(verified.all()):     # last resort: the free clean-image seed
        bad = (~verified).nonzero(as_tuple=True)[0]
        x_final[bad] = oracle.quantize_pixels(seeds[bad, 0])
        verified = consensus_adversarial(backends, x_final, y)
    print(f"[6]   robustly adversarial on {len(backends)} evaluators x "
          f"{len(input_views(x01[:1], dev))} input representations: "
          f"{int(verified.sum())}/{n}")
    return x_final, verified, q_log
