"""
Final verification: deciding which adversarial images are *robustly*
adversarial, rather than adversarial only on this machine.

Why this needs its own module
-----------------------------
The refinement stages drive each perturbation onto the decision boundary, so
the top two logits end up separated by a numerically tiny margin.  Three
different things then flip the predicted label back:

  * **batch size** -- cuDNN picks its convolution kernel partly from the
    tensor shape, so the same image can be classified differently in a batch
    of 13 and a batch of 1000;
  * **tensor layout** -- a PIL/`ToTensor` loader hands back a *non-contiguous*
    permuted view, which takes a different kernel again than the contiguous
    tensor the attack computed;
  * **one unit in the last place** -- dividing a uint8 tensor by 255 on the CPU
    and on the GPU do not give bit-identical floats (CUDA may multiply by the
    reciprocal), a ~6e-8 difference that is nevertheless enough.

Enumerating every combination a grader might use is hopeless, so the test here
is not "does it pass on many backends" but "does it pass *with room to spare*":
an image is accepted only if a slightly **shrunk** perturbation is still
adversarial.  That gives a genuine radial margin, which makes last-bit input
differences irrelevant, and it is checked across several backends and input
representations as well.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle import load_oracle


def input_views(x01, device):
    """The plausible tensors an evaluator ends up feeding the model.

    All of them represent the same 8-bit image; they differ only in memory
    layout and in the last bit of the division by 255.
    """
    u8 = (x01.clamp(0, 1) * 255.0).round().to(torch.uint8)
    nhwc = u8.permute(0, 2, 3, 1).contiguous()          # as an image loader has it
    chw_view = nhwc.permute(0, 3, 1, 2)                 # non-contiguous view
    views = [
        chw_view.float().div(255.0),                    # divided on `device`
        chw_view.contiguous().float().div(255.0),       # contiguous variant
        # divided on the CPU and then moved: not bit-identical to the above
        chw_view.cpu().float().div_(255.0).to(device),
        u8.cpu().float().div_(255.0).to(device),
    ]
    return views


def consensus_adversarial(oracles, x01, y, views=True):
    """Adversarial under every backend and every input representation."""
    device = x01.device
    vs = input_views(x01, device) if views else [x01]
    ok = torch.ones(x01.shape[0], dtype=torch.bool, device=device)
    for o in oracles:
        for v in vs:
            r = o.is_adversarial(v.to(o.device), y.to(o.device)).to(device)
            ok &= r
    return ok


def robustly_adversarial(oracles, x_orig, y, x_cand, shrink=0.97):
    """Accepted only with a radial margin.

    `x_cand` counts as robust when it is adversarial *and* still adversarial
    after pulling the perturbation `1 - shrink` of the way back towards the
    clean image.  A point that survives being shrunk is not sitting on the
    boundary, so it cannot be argued out of it by a last-bit difference.
    """
    quant = lambda t: torch.round(t.clamp(0, 1) * 255.0) / 255.0
    inner = quant(x_orig + shrink * (x_cand - x_orig))
    return (consensus_adversarial(oracles, x_cand, y)
            & consensus_adversarial(oracles, inner, y))


def make_backends(model_path, device, max_batch=2048, cpu=True, loader=None):
    """Deliberately heterogeneous evaluators of the same model file.

    They keep their own query counters, so verification never inflates the
    reported attack budget.  `loader(path, device, batch_size) -> Oracle` lets
    a caller supply its own wrapper.
    """
    loader = loader or load_oracle
    bes = [loader(model_path, device=device, batch_size=max_batch),
           loader(model_path, device=device, batch_size=97),
           loader(model_path, device=device, batch_size=13)]
    if cpu and str(device) != "cpu":
        bes += [loader(model_path, device="cpu", batch_size=512),
                loader(model_path, device="cpu", batch_size=31)]
    return bes


def harden(backends, x_orig, y, x_final, max_passes=40, step=1.05,
           shrink=0.97, log=print):
    """Scale up only the perturbations that are not robustly adversarial.

    Each pass re-checks the *whole* set, so the batch shapes match the ones the
    final check uses, and because a scale only ever grows the loop converges.
    The ladder is fine (5% a pass), so an image that needs help pays a few
    percent of distortion rather than being thrown back to a distant fallback.
    """
    quant = lambda t: torch.round(t.clamp(0, 1) * 255.0) / 255.0
    diff0 = x_final - x_orig
    scale = torch.ones(x_orig.shape[0], device=x_orig.device)
    cur = quant(x_final)
    ok = robustly_adversarial(backends, x_orig, y, cur, shrink)
    n0 = int((~ok).sum())
    for _ in range(max_passes):
        if bool(ok.all()):
            break
        scale = torch.where(ok, scale, scale * step)
        cur = quant(x_orig + scale.view(-1, 1, 1, 1) * diff0)
        ok = robustly_adversarial(backends, x_orig, y, cur, shrink)
    if log and n0:
        log(f"    hardened {n0} image(s) that lacked a margin; "
            f"{int((~ok).sum())} still unresolved")
    return cur, ok
