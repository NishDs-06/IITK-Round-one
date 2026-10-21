# IITK Round One

Submissions for the two problem statements of round one. Each lives in its own
directory with its own README, code and deliverables.

| | problem | status |
| --- | --- | --- |
| [`PS-1/`](PS-1/) | Problem statement 1 | — |
| [`PS-2/`](PS-2/) | **Black-box adversarial attack on a CNN** | complete |

## PS-2 — Black-box adversarial attack on a CNN

A decision-based (**hard-label**) attack on the provided CIFAR-10 CNN: the
model may be queried at arbitrary inputs and only its predicted label read —
no architecture, parameters, logits or intermediate outputs. One adversarial
`.png` is produced for each of the 1000 clean images.

| | |
| --- | --- |
| **Attack success rate** | **100.00 %** (1000/1000) |
| **PSNR** | mean **49.00 dB** |
| **SSIM** | mean **0.99762** |
| L2 per image (0-255 scale) | mean 59.0, median 58.3 |
| L-inf per image (0-255 scale) | mean 6.9, median 7 |
| Model inferences | **16,900 per image** |

Measured by reading the submitted `.png` files back from disk and querying the
provided model. See [`PS-2/README.md`](PS-2/README.md) for the method and
[`PS-2/report/report.pdf`](PS-2/report/report.pdf) for the written report.

![PS-2 examples](PS-2/report/examples.png)
