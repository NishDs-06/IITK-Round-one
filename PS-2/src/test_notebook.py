"""Smoke-test the generated notebook: run its code cells in order, in a
temporary working directory, with a deliberately tiny attack configuration.

This checks the cell that will actually be submitted -- including the save
cell -- without waiting for a full-budget run.

    python src/test_notebook.py --nb p2_code.ipynb
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile


TINY = """
CONFIG.update(iters=1, b0=16, bmax=16, bsearch=5, walk_steps=8,
              geo_steps=8, refine_rounds=1, greedy_rounds=1, seeds=2,
              rescue_rounds=1, rescue_geo=6, rescue_walk=6,
              grid_rounds=1, grid_geo=6, grid_walk=6, max_batch=1024)
print("[test] using tiny CONFIG:", CONFIG)
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nb", default="p2_code.ipynb")
    p.add_argument("--zip", default="shared-files/p2_data.zip")
    p.add_argument("--keep", action="store_true")
    p.add_argument("--link-data", default="data",
                   help="symlink this directory into the temp cwd so cached "
                        "CIFAR-10 / substitute checkpoints are reused")
    a = p.parse_args()

    nb = json.load(open(os.path.abspath(a.nb)))
    zip_abs = os.path.abspath(a.zip)
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]

    data_abs = os.path.abspath(a.link_data) if a.link_data else None
    work = tempfile.mkdtemp(prefix="nbtest_")
    if data_abs and os.path.isdir(data_abs):
        os.symlink(data_abs, os.path.join(work, "data"))
    print(f"[test] working directory {work}")
    cwd = os.getcwd()
    try:
        os.chdir(work)
        g = {"__name__": "__main__"}
        for i, src in enumerate(cells):
            # point the notebook at the real archive, and skip the sample plot
            src = re.sub(r'^zip_path = .*$', f'zip_path = {zip_abs!r}', src,
                         flags=re.M)
            src = src.replace("plt.show()", "pass  # plt.show()")
            if "CONFIG = {" in src:                    # shrink the budget
                src = src.replace("\noracle = Oracle(", TINY + "\noracle = Oracle(")
            print(f"\n===== cell {i} =====")
            exec(compile(src, f"<cell {i}>", "exec"), g)
        n = len(os.listdir(os.path.join(work, "adv_images")))
        print(f"\n[test] adv_images written: {n}")
        assert n == 1000, f"expected 1000 files, got {n}"
        names = sorted(os.listdir(os.path.join(work, "adv_images")))
        print(f"[test] first/last: {names[0]} ... {names[-1]}")
        from PIL import Image
        im = Image.open(os.path.join(work, "adv_images", names[0]))
        print(f"[test] sample image: {im.size} {im.mode} {im.format}")
        assert im.size == (32, 32) and im.mode == "RGB" and im.format == "PNG"
        print("[test] PASS")
    finally:
        os.chdir(cwd)
        if not a.keep:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
