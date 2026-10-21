"""Imperceptibility metrics, computed on the 8-bit images that are shipped."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _gauss_window(sigma: float = 1.5, size: int = 11, device="cpu"):
    c = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :]).view(1, 1, size, size)


def ssim(a: torch.Tensor, b: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Mean SSIM per image for (N,C,H,W) tensors in [0, data_range].

    11x11 Gaussian window, sigma 1.5, per channel, with the border cropped
    (i.e. a `valid` convolution) rather than padded -- the convention used by
    scikit-image and the original paper. On 32x32 inputs padding would let
    border artefacts dominate.
    """
    c = a.shape[1]
    win = _gauss_window(device=a.device).repeat(c, 1, 1, 1)
    mu_a = F.conv2d(a, win, groups=c)
    mu_b = F.conv2d(b, win, groups=c)
    saa = F.conv2d(a * a, win, groups=c) - mu_a ** 2
    sbb = F.conv2d(b * b, win, groups=c) - mu_b ** 2
    sab = F.conv2d(a * b, win, groups=c) - mu_a * mu_b
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / \
        ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    return s.flatten(1).mean(dim=1)


def report(x_clean: torch.Tensor, x_adv: torch.Tensor, success: torch.Tensor,
           n_queries: int | None = None) -> dict:
    """All distances are measured on the 0..255 integer pixel scale."""
    a = (x_clean.clamp(0, 1) * 255).round()
    b = (x_adv.clamp(0, 1) * 255).round()
    d = (b - a)
    flat = d.flatten(1)
    n, dim = flat.shape

    l2 = flat.norm(dim=1)
    li = flat.abs().max(dim=1).values
    l0 = (flat != 0).float().sum(dim=1)
    mse = (flat ** 2).mean(dim=1)
    psnr = 10.0 * torch.log10(255.0 ** 2 / mse.clamp(min=1e-12))
    ss = ssim(a / 255.0, b / 255.0)

    out = {
        "n_images": n,
        "success_rate": success.float().mean().item() * 100.0,
        "n_failed": int((~success).sum().item()),
        "l2_mean": l2.mean().item(),
        "l2_median": l2.median().item(),
        "l2_max": l2.max().item(),
        "l2_norm01_mean": (l2 / 255.0).mean().item(),
        "linf_mean": li.mean().item(),
        "linf_median": li.median().item(),
        "linf_max": li.max().item(),
        "l0_mean": l0.mean().item(),
        "l0_frac_mean": (l0 / dim).mean().item(),
        "rmse_mean": mse.sqrt().mean().item(),
        "psnr_mean": psnr.mean().item(),
        "psnr_median": psnr.median().item(),
        "ssim_mean": ss.mean().item(),
        "ssim_min": ss.min().item(),
    }
    if n_queries is not None:
        out["queries_total"] = n_queries
        out["queries_per_image"] = n_queries / max(n, 1)
    return out


def format_report(r: dict) -> str:
    lines = [
        f"images                : {r['n_images']}",
        f"attack success rate   : {r['success_rate']:.2f} %   (failed: {r['n_failed']})",
        f"L2   (0-255 scale)    : mean {r['l2_mean']:.2f}   median {r['l2_median']:.2f}   max {r['l2_max']:.2f}",
        f"L2   ([0,1] scale)    : mean {r['l2_norm01_mean']:.4f}",
        f"Linf (0-255 scale)    : mean {r['linf_mean']:.2f}   median {r['linf_median']:.1f}   max {r['linf_max']:.0f}",
        f"L0   changed pixels   : mean {r['l0_mean']:.0f} / 3072  ({100*r['l0_frac_mean']:.1f} %)",
        f"RMSE (0-255 scale)    : mean {r['rmse_mean']:.3f}",
        f"PSNR  (dB)            : mean {r['psnr_mean']:.2f}   median {r['psnr_median']:.2f}",
        f"SSIM                  : mean {r['ssim_mean']:.5f}   min {r['ssim_min']:.5f}",
    ]
    if "queries_total" in r:
        lines.append(f"model inferences      : {r['queries_total']:,}  "
                     f"({r['queries_per_image']:.1f} per image)")
    return "\n".join(lines)
