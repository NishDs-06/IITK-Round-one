"""Post-run packaging: re-measure the saved PNGs, make the figures, build the
report PDF and splice the headline numbers into README.md.

    python src/finish.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import metrics as M
from data import load_clean
from make_figures import load_adv
from oracle import load_oracle


def measure(data_dir, adv_dir, device, metrics_json):
    """Re-measure from disk: this is the number for the submitted files."""
    x, y, names = load_clean(data_dir, device=device)
    xa = load_adv(adv_dir, names, device=device)
    orc = load_oracle(os.path.join(data_dir, "cnn_model.pt"), device=device,
                      batch_size=1024)
    ok = orc.is_adversarial(xa, y)
    r = M.report(x, xa, ok)
    if os.path.exists(metrics_json):           # keep the query accounting
        old = json.load(open(metrics_json))
        for k in ("queries_total", "queries_per_image", "queries_per_stage",
                  "wall_clock_s", "config"):
            if k in old:
                r[k] = old[k]
    return r


NB_KEYS = [
    ("seeds", "seeds"), ("bsearch", "bsearch"),
    ("surrogate_epochs", "surrogate_epochs"),
    ("iters", "iters"), ("b0", "b0"), ("bmax", "bmax"),
    ("refine_rounds", "refine_rounds"),
    ("geo_steps", "geo"), ("walk_steps", "walk"),
    ("rescue_rounds", "rescue_rounds"), ("rescue_frac", "rescue_frac"),
    ("rescue_geo", "rescue_geo"), ("rescue_walk", "rescue_walk"),
    ("grid_rounds", "grid_rounds"), ("grid_geo", "grid_geo"),
    ("grid_walk", "grid_walk"), ("greedy_rounds", "greedy_rounds"),
    ("max_batch", "max_batch"), ("seed", "seed"),
    ("verify_shrink", "verify_shrink"),
]
FLOAT_KEYS = {"rescue_frac", "verify_shrink"}


def sync_notebook_config(run_config, path="report/final_config.json"):
    """Write the notebook CONFIG from the parameters the run actually used.

    The notebook has to carry the *finalised* parameters, so they are derived
    from the driver's own arguments rather than maintained by hand.
    """
    cfg = {"transfer": not run_config.get("no_transfer", False)}
    for nb_key, arg_key in NB_KEYS:
        if arg_key in run_config:
            v = run_config[arg_key]
            cfg[nb_key] = float(v) if nb_key in FLOAT_KEYS else int(v)
    cfg["model_path"] = "./p2_data/cnn_model.pt"
    # keep a stable, readable key order
    order = ["seeds", "bsearch", "transfer", "surrogate_epochs", "iters",
             "b0", "bmax", "refine_rounds", "geo_steps", "walk_steps",
             "rescue_rounds", "rescue_frac", "rescue_geo", "rescue_walk",
             "grid_rounds", "grid_geo", "grid_walk", "greedy_rounds",
             "max_batch", "verify_shrink", "seed", "model_path"]
    cfg = {k: cfg[k] for k in order if k in cfg}
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)
    return path


README_TMPL = """## Results

Measured by reading the 1000 submitted `.png` files back from disk and querying
the provided model:

| | |
| --- | --- |
| **Attack success rate** | **{asr:.1f} %** ({nok}/{n} images misclassified) |
| L2 per image (0-255 scale) | mean **{l2m:.1f}**, median {l2med:.1f}, max {l2max:.1f} |
| L-inf per image (0-255 scale) | mean **{lim:.1f}**, median {limed:.0f}, max {limax:.0f} |
| RMSE (0-255 scale) | mean **{rmse:.2f}** pixel levels |
| PSNR | mean **{psnr:.2f} dB**, median {psnrmed:.2f} dB |
| SSIM | mean **{ssim:.5f}**, min {ssimmin:.4f} |
| Changed pixel values | {l0:.0f} of 3072 ({l0f:.0f} %) |
| Model inferences | {qt:,} total, **{qpi:,.0f} per image** |

A mean RMSE of {rmse:.2f} pixel levels means the average pixel value moves by
about one quantisation step — the smallest change an 8-bit image can hold —
and {l0keep:.0f} % of all pixel values are left bit-identical. The noise is
invisible at 1:1; the difference images in `report/` are amplified 8x to show
anything at all.

![examples](report/examples.png)
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/p2_data")
    p.add_argument("--adv", default="adv_images")
    p.add_argument("--metrics", default="report/metrics_final.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip-figures", action="store_true")
    a = p.parse_args()

    r = measure(a.data, a.adv, a.device, a.metrics)
    print(M.format_report(r))
    with open(a.metrics, "w") as f:
        json.dump(r, f, indent=2, default=str)

    py = sys.executable
    if isinstance(r.get("config"), dict):
        cfg_path = sync_notebook_config(r["config"])
        subprocess.run([py, "src/build_notebook.py", "--config", cfg_path,
                        "--out", "p2_code.ipynb"], check=True)
    if not a.skip_figures:
        subprocess.run([py, "src/make_figures.py", "--data", a.data,
                        "--adv", a.adv, "--out", "report"], check=True)
    subprocess.run([py, "src/make_report.py", "--metrics", a.metrics,
                    "--out", "report/report.pdf"], check=True)

    block = README_TMPL.format(
        asr=r["success_rate"], nok=r["n_images"] - r["n_failed"],
        n=r["n_images"], l2m=r["l2_mean"], l2med=r["l2_median"],
        l2max=r["l2_max"], lim=r["linf_mean"], limed=r["linf_median"],
        limax=r["linf_max"], rmse=r["rmse_mean"], psnr=r["psnr_mean"],
        psnrmed=r["psnr_median"], ssim=r["ssim_mean"], ssimmin=r["ssim_min"],
        l0=r["l0_mean"], l0f=100 * r["l0_frac_mean"],
        l0keep=100 * (1 - r["l0_frac_mean"]),
        qt=r.get("queries_total", 0), qpi=r.get("queries_per_image", 0))
    readme = open("README.md").read()
    marker = "<!--RESULTS-->"
    if marker in readme:
        head, _, tail = readme.partition(marker)
        # Drop a block inserted by an earlier run, then keep everything from
        # the next H2 on.  Without the first step, re-running would stack a
        # second "## Results" section on top of the old one.
        nxt = tail.find("\n## ")
        if nxt >= 0:
            if tail[nxt:].lstrip("\n").startswith("## Results"):
                after = tail.find("\n## ", nxt + 1)
                tail = tail[after:] if after >= 0 else ""
            else:
                tail = tail[nxt:]
        readme = head + marker + "\n\n" + block + tail
        open("README.md", "w").write(readme)
        print("README.md results section updated")


if __name__ == "__main__":
    main()
