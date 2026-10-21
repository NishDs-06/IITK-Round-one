"""
Black-box oracle wrapper around the provided TorchScript CNN.

The attack is only ever allowed to see the *hard label* returned by the model.
This module is the single point of contact with the model: everything else in
the codebase talks to `Oracle` and therefore physically cannot observe
architecture, parameters, logits or intermediate activations.

Every forward pass through the victim model is counted, so the reported query
budget is exact.
"""

from __future__ import annotations

import torch

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


class Oracle:
    """Hard-label-only, query-counting wrapper.

    Inputs are always images in *pixel space*, i.e. float tensors in [0, 1]
    with shape (N, 3, H, W).  Normalisation with (MEAN, STD) happens inside,
    mirroring the `transform` used in the provided notebook.
    """

    def __init__(self, model, device="cpu", batch_size=4096, quantize=False):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        # If True, every image is snapped to the 8-bit PNG grid before being
        # shown to the model, so the oracle answers questions about images that
        # can actually be stored as .png files.
        self.quantize = quantize
        self.n_queries = 0
        self._mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def quantize_pixels(x01: torch.Tensor) -> torch.Tensor:
        """Snap [0,1] images onto the 8-bit grid that a .png file can hold."""
        return torch.round(x01.clamp(0.0, 1.0) * 255.0) / 255.0

    def reset_counter(self) -> None:
        self.n_queries = 0

    # ------------------------------------------------------------- the oracle
    @torch.no_grad()
    def predict(self, x01: torch.Tensor) -> torch.Tensor:
        """Return the predicted hard labels for a batch of [0,1] images."""
        x01 = x01.to(self.device, non_blocking=True)
        if self.quantize:
            x01 = self.quantize_pixels(x01)
        else:
            x01 = x01.clamp(0.0, 1.0)

        out = []
        for i in range(0, x01.shape[0], self.batch_size):
            chunk = x01[i:i + self.batch_size]
            z = (chunk - self._mean) / self._std
            y = self.model(z)
            # The provided model already returns labels. If a future model were
            # to return a score vector we take the arg-max and discard the
            # scores immediately, so no confidence information is ever used.
            if y.dim() > 1:
                y = y.argmax(dim=1)
            out.append(y.reshape(-1))
            self.n_queries += chunk.shape[0]
        return torch.cat(out).to(torch.long)

    @torch.no_grad()
    def is_adversarial(self, x01: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Boolean mask: True where the model's label differs from `y_true`."""
        return self.predict(x01) != y_true.to(self.device)


def load_oracle(model_path: str, device: str = "cpu", **kwargs) -> Oracle:
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return Oracle(model, device=device, **kwargs)
