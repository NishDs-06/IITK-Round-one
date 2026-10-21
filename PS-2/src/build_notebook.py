"""
Build the submission notebook from the provided template plus the real attack
source, so the notebook runs byte-identical code to what produced the shipped
adversarial images.

    python src/build_notebook.py --config report/metrics_final.json
"""
from __future__ import annotations

import argparse
import json
import os
import re

HEADER = """\
# =====================================================================
#  BLACK-BOX (HARD-LABEL) ADVERSARIAL ATTACK
# =====================================================================
#  Threat model actually used by the code below:
#    * the only thing read from `model` is the integer label it returns;
#    * no architecture, no parameters, no logits, no intermediate outputs.
#  Every oracle call goes through `Oracle.predict`, which counts queries,
#  so the inference budget reported at the end is exact.
#
#  Pipeline
#  --------
#   0   free adversarial seeds: clean images of *other* classes (0 queries).
#       All 1000 provided images are classified correctly, so a clean image
#       with a different label is already adversarial for this one -- and it
#       is a valid 8-bit PNG, which puts a 100% success floor under the run.
#   1   bisection onto the decision boundary.
#   1b  substitute networks, trained on labels *queried from the victim*,
#       supply candidate directions; the victim's own labels then decide the
#       smallest radius along each (Papernot et al. 2017; Rony et al. 2019).
#       Set CONFIG["transfer"] = False for the pure query-only variant.
#   2   HopSkipJump: estimate the boundary normal from +-1 labels, step,
#       re-project (Chen, Jordan & Wainwright, IEEE S&P 2020).  Skipped when
#       CONFIG["iters"] == 0, which the transfer stage makes possible.
#   3   boundary refinement: geometric steps (SurFree, Maho et al. 2021) and
#       a random walk along the boundary (Brendel et al. 2018), then extra
#       passes spent only on the images still furthest away.
#   4   8-bit finalisation: the perturbation is about one pixel level in RMS,
#       so it has to be made to survive rounding to the PNG grid.
#   4b  refinement of the *rounded* objective directly -- the quantity that
#       actually gets graded.
#   5   integer-grid polishing: L-inf clipping, pruning, greedy shrinking.
#   6   verification across several evaluators (different batch sizes, CPU as
#       well as GPU) and both memory layouts, including the exact
#       representation a PNG loader hands back.
# =====================================================================

import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# the pixel-space normalisation constants, under the names the inlined
# substitute-model code expects
MEAN, STD = mean, std

CONFIG = {config}
"""

RUNNER = '''

# ---------------------------------------------------------------------
#  Run the attack on the images loaded above
# ---------------------------------------------------------------------
_mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
_std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)

# `X` is normalised; recover the exact 8-bit pixel values in [0, 1].
X01 = (X * _std_t + _mean_t).clamp(0, 1)
X01 = torch.round(X01 * 255.0) / 255.0

oracle = Oracle(model, mean, std, device=device, batch_size=CONFIG["max_batch"])

t0 = time.time()
print(f"clean accuracy: {(oracle.predict(X01) == y_true).float().mean().item()*100:.2f} %")
oracle.reset_counter()

X_adv01, verified, stage_queries = run_attack(oracle, X01, y_true, CONFIG)

print(f"\\nattack success rate : {verified.float().mean().item()*100:.2f} %"
      f"  ({int((~verified).sum())} failures)")
d = ((X_adv01 - X01) * 255).flatten(1)
print(f"L2   (0-255 scale)  : mean {d.norm(dim=1).mean():.2f}   "
      f"median {d.norm(dim=1).median():.2f}")
print(f"Linf (0-255 scale)  : mean {d.abs().max(1).values.mean():.2f}   "
      f"median {d.abs().max(1).values.median():.0f}")
_mse = (d ** 2).mean(dim=1).clamp(min=1e-12)
print(f"PSNR (dB)           : mean {(10*torch.log10(255.0**2/_mse)).mean():.2f}")
print(f"model inferences    : {oracle.n_queries:,}  "
      f"({oracle.n_queries/X01.shape[0]:.0f} per image)")
print(f"wall clock          : {time.time()-t0:.0f} s")
'''

SAVER = '''\
# ---------------------------------------------------------------------
#  Save one adversarial .png per clean image as "<class>_<index>_adv.png"
# ---------------------------------------------------------------------
import os
from PIL import Image

out_dir = "adv_images"
os.makedirs(out_dir, exist_ok=True)

# X_adv01 is already on the 8-bit grid and in [0, 1] pixel space (i.e. the
# denormalisation with `mean`/`std` has been undone), so writing it is lossless.
arr = (X_adv01.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu()
arr = arr.permute(0, 2, 3, 1).numpy()

for a, p in zip(arr, paths):
    name = os.path.splitext(os.path.basename(p))[0]      # e.g. "cat_5"
    Image.fromarray(a, mode="RGB").save(os.path.join(out_dir, f"{name}_adv.png"))

print(f"saved {len(arr)} adversarial images to ./{out_dir}/")

# Re-read the files from disk and re-check them, so the number quoted above is
# the number for the images that are actually being submitted.
reread = torch.stack([
    transforms.ToTensor()(Image.open(os.path.join(out_dir, os.path.splitext(
        os.path.basename(p))[0] + "_adv.png")).convert("RGB")) for p in paths
]).to(device)
_pred = model((reread - _mean_t) / _std_t)
print(f"re-checked from disk: attack success rate "
      f"{(_pred != y_true).float().mean().item()*100:.2f} %")
'''


# modules of this project that are inlined, and therefore must not be
# imported at run time inside the notebook
_LOCAL = r"(attack|surrogate|transfer|oracle|cifar|data|metrics|verify|notebook_driver)"


def strip_module(path):
    """Inline a source file.

    Drops the module docstring and the *top-level* imports (torch, nn, F and
    friends are imported once in the cell header), keeps function-local
    imports such as `from torchvision import datasets` -- except imports of
    this project's own modules, which are inlined rather than importable --
    and removes the `module.` prefixes that those imports provided.
    """
    src = open(path).read()
    src = re.sub(r'^\s*(""".*?"""|\'\'\'.*?\'\'\')\s*', "", src, count=1, flags=re.S)
    out = []
    for line in src.splitlines():
        if re.match(r"^(import|from)\s+\S+", line):          # top level only
            continue
        if re.match(rf"^\s*(import|from)\s+{_LOCAL}\b", line):
            continue
        if "sys.path.insert" in line:          # module plumbing, not needed
            continue
        if line.startswith('if __name__ == "__main__"'):
            break
        out.append(line)
    text = "\n".join(out)
    # `A.foo` / `S.foo` / `T.foo` -> `foo`, now that everything is in one scope
    text = re.sub(r"\b[AST]\.(?=[A-Za-z_])", "", text)
    text = re.sub(r"\bcifar\.load\b", "_cifar_load_unavailable", text)
    return text.rstrip() + "\n"


ORACLE_SRC = '''
class Oracle:
    """Hard-label-only, query-counting wrapper around the provided model.

    Inputs are images in pixel space: float tensors in [0, 1] of shape
    (N, 3, 32, 32).  Normalisation with (mean, std) happens inside, exactly as
    in the `transform` used to build `X`.
    """

    def __init__(self, model, mean, std, device="cpu", batch_size=2048,
                 quantize=False):
        self.model, self.device, self.batch_size = model, device, batch_size
        self.quantize = quantize
        self.n_queries = 0
        self._mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(std, device=device).view(1, 3, 1, 1)

    @staticmethod
    def quantize_pixels(x01):
        """Snap to the 8-bit grid a .png file can hold."""
        return torch.round(x01.clamp(0.0, 1.0) * 255.0) / 255.0

    def reset_counter(self):
        self.n_queries = 0

    @torch.no_grad()
    def predict(self, x01):
        x01 = x01.to(self.device)
        x01 = self.quantize_pixels(x01) if self.quantize else x01.clamp(0.0, 1.0)
        out = []
        for i in range(0, x01.shape[0], self.batch_size):
            chunk = x01[i:i + self.batch_size]
            lab = self.model((chunk - self._mean) / self._std)
            if lab.dim() > 1:          # defensive: use the arg-max only,
                lab = lab.argmax(dim=1)  # never the scores themselves
            out.append(lab.reshape(-1))
            self.n_queries += chunk.shape[0]
        return torch.cat(out).to(torch.long)

    @torch.no_grad()
    def is_adversarial(self, x01, y_true):
        return self.predict(x01) != y_true.to(self.device)
'''


def build(template, out_path, config, attack_src, driver_src,
          transfer_src=None):
    nb = json.load(open(template))

    def src_list(text):
        lines = text.splitlines(keepends=True)
        return lines

    transfer_doc = ""
    cfg_lines = ",\n".join(f"    {k!r}: {v!r}" for k, v in config.items())
    pieces = [HEADER.format(config="{\n" + cfg_lines + ",\n}"),
              ORACLE_SRC, "\n", attack_src, "\n", driver_src, RUNNER]
    attack_cell = "".join(pieces)

    found_attack = found_save = False
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        # ship a notebook that is uniformly un-run: the template still carries
        # the organisers' own outputs and execution counts, which next to our
        # freshly generated cells would look like a half-executed notebook
        c["outputs"], c["execution_count"] = [], None
        s = "".join(c["source"])
        if "IMPLEMENT YOUR ATTACK HERE" in s:
            c["source"] = src_list(attack_cell)
            c["outputs"], c["execution_count"] = [], None
            found_attack = True
        elif "SAVE ADVERSARIAL IMAGES" in s:
            c["source"] = src_list(SAVER)
            c["outputs"], c["execution_count"] = [], None
            found_save = True
    assert found_attack and found_save, "template cells not found"

    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--template", default="shared-files/p2_code.ipynb")
    p.add_argument("--out", default="p2_code.ipynb")
    p.add_argument("--config", default="report/final_config.json")
    a = p.parse_args()

    cfg = json.load(open(a.config))
    here = os.path.dirname(os.path.abspath(__file__))
    attack_src = strip_module(os.path.join(here, "attack.py"))
    extra = ""
    if cfg.get("transfer"):
        extra = ("\n\n# ===================== substitute networks =====================\n"
                 + strip_module(os.path.join(here, "surrogate.py"))
                 + "\n\n# ================== transfer-stage directions ==================\n"
                 + strip_module(os.path.join(here, "transfer.py")))
    extra += ("\n\n# ================ robustness of the final images ===============\n"
              + strip_module(os.path.join(here, "verify.py")))
    driver_src = strip_module(os.path.join(here, "notebook_driver.py"))
    build(a.template, a.out, cfg, attack_src + extra, driver_src)


if __name__ == "__main__":
    main()
