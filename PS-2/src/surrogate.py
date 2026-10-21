"""
Optional transfer component: train surrogate CIFAR-10 classifiers locally and
use their gradients as a *prior* for the direction search.

This stays inside the black-box threat model: the surrogates are trained on the
public CIFAR-10 training split, never on the victim's parameters, architecture,
gradients or scores.  The victim is still only ever asked for hard labels, and
every perturbation derived from a surrogate is accepted only after the victim
has confirmed the label flip.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle import MEAN, STD


# --------------------------------------------------------------- architectures
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                       nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        out = out + (x if self.short is None else self.short(x))
        return F.relu(out)


class ResNet(nn.Module):
    """ResNet-18/-10 style CIFAR network with a configurable width."""

    def __init__(self, blocks=(2, 2, 2, 2), width=64, n_cls=10):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv2d(3, w, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(w), nn.ReLU(inplace=True))
        layers, cin = [], w
        for i, nb in enumerate(blocks):
            cout = w * 2 ** i
            for j in range(nb):
                layers.append(BasicBlock(cin, cout, stride=(2 if (j == 0 and i > 0) else 1)))
                cin = cout
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(cin, n_cls)

    def forward(self, x):
        h = self.body(self.stem(x))
        return self.head(F.adaptive_avg_pool2d(h, 1).flatten(1))


class VGGish(nn.Module):
    def __init__(self, n_cls=10, width=64):
        super().__init__()
        cfg = [width, width, "M", 2 * width, 2 * width, "M",
               4 * width, 4 * width, "M", 8 * width, 8 * width, "M"]
        layers, cin = [], 3
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(2))
            else:
                layers += [nn.Conv2d(cin, v, 3, padding=1, bias=False),
                           nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                cin = v
        self.features = nn.Sequential(*layers)
        # four 2x2 pools take 32x32 down to 2x2
        self.head = nn.Linear(cin * 4, n_cls)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


def build(name: str) -> nn.Module:
    return {
        "resnet18": lambda: ResNet((2, 2, 2, 2), 64),
        "resnet10": lambda: ResNet((1, 1, 1, 1), 64),
        "resnet18w32": lambda: ResNet((2, 2, 2, 2), 32),
        "vgg": lambda: VGGish(),
        "vggw32": lambda: VGGish(width=32),
    }[name]()


class Normalized(nn.Module):
    """Wraps a surrogate so it takes [0,1] pixel inputs, like the Oracle."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("m", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("s", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x01):
        return self.net((x01 - self.m) / self.s)


# -------------------------------------------------------------------- training
def cifar_tensors(root="data/cifar10_hf", train=True, device="cuda"):
    """Public CIFAR-10 split as (uint8 (N,3,32,32) on `device`, labels).

    Uses a local copy if one is present, otherwise downloads the standard
    torchvision copy -- so the same code runs in a fresh notebook.
    """
    split = "train" if train else "test"
    if os.path.exists(os.path.join(root, f"{split}.parquet")) or \
       os.path.exists(os.path.join(root, f"{split}.pt")):
        try:
            import cifar
            return cifar.load(split, root=root, device=device)
        except ImportError:
            pass        # inlined into a notebook: fall through to torchvision
    import numpy as np
    from torchvision import datasets
    ds = datasets.CIFAR10("data/cifar10", train=train, download=True)
    x = torch.from_numpy(np.asarray(ds.data)).permute(0, 3, 1, 2).contiguous()
    y = torch.tensor(ds.targets, dtype=torch.long)
    return x.to(device), y.to(device)


VICTIM_LABEL_CACHE = "data/cifar10_hf/victim_labels.pt"


def victim_labels(oracle=None, model_path="data/p2_data/cnn_model.pt",
                  root="data/cifar10_hf", device="cuda", batch_size=1024,
                  cache=VICTIM_LABEL_CACHE):
    """Label the public CIFAR-10 splits with the *victim's* hard labels.

    This is the substitute-model construction of Papernot et al. (2017): the
    surrogate learns the victim's own decision function from nothing but
    queried labels, so the transfer stage uses no information the black box
    does not hand out.  Cost is a one-off 60,000 queries shared by all 1,000
    attacked images (60 per image), and it is charged to `oracle`'s counter
    when an oracle is passed in.

    Returns (train_labels, test_labels, n_queries_spent).
    """
    if cache and os.path.exists(cache):
        d = torch.load(cache, map_location="cpu", weights_only=True)
        nq = int(d["n_queries"])
        if oracle is not None:      # keep the budget honest on a cache hit
            oracle.n_queries += nq
        return d["train"].to(device), d["test"].to(device), nq

    own = oracle is None
    if own:
        from oracle import load_oracle
        oracle = load_oracle(model_path, device=device, batch_size=batch_size)
    q0 = oracle.n_queries
    out = {}
    for split in ("train", "test"):
        xu, _ = cifar_tensors(root, split == "train", device)
        labs = []
        for i in range(0, xu.shape[0], 5000):
            labs.append(oracle.predict(xu[i:i + 5000].float().div(255.0)).cpu())
        out[split] = torch.cat(labs)
    nq = oracle.n_queries - q0
    out["n_queries"] = torch.tensor(nq)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        torch.save(out, cache)
    return out["train"].to(device), out["test"].to(device), nq


SURROGATE_SPEC = (("resnet18", 0), ("vgg", 0), ("resnet10", 1), ("vggw32", 2))


def train_or_load(oracle=None, out_dir="data/surrogates", epochs=20, bs=512,
                  lr=0.3, device="cuda", spec=SURROGATE_SPEC, log=print):
    """Return a list of substitute networks, training any that are missing.

    Checkpoints are reused across runs, so the expensive part happens once.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, seed in spec:
        path = os.path.join(out_dir, f"{name}_eps0.0000_s{seed}.pt")
        if not os.path.exists(path):
            log(f"  training substitute {name} (seed {seed}) ...")
            train_one(name, epochs=epochs, bs=bs, lr=lr, device=device,
                      seed=seed, out_dir=out_dir, log=log, oracle=oracle)
        paths.append(path)
    return load_surrogates(paths, device=device), paths


def augment(xb_uint8):
    """Random crop (pad 4) + horizontal flip, done on the GPU."""
    n = xb_uint8.shape[0]
    x = xb_uint8.float().div_(255.0)
    x = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i = torch.randint(0, 9, (2,))
    x = x[:, :, i[0]:i[0] + 32, i[1]:i[1] + 32]
    flip = torch.rand(n, device=x.device) < 0.5
    x[flip] = torch.flip(x[flip], dims=[3])
    return x


def train_one(name, epochs=30, bs=512, lr=0.2, label_smooth=0.0,
              adv_eps=0.0, device="cuda", seed=0, root="data/cifar10_hf",
              out_dir="data/surrogates", log=print, use_victim_labels=True,
              oracle=None):
    """Train one surrogate.  `adv_eps > 0` enables fast FGSM-RS adversarial
    training (Wong et al., ICLR 2020), giving a *robust* surrogate whose
    gradients transfer to robust victims."""
    torch.manual_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    xtr, ytr = cifar_tensors(root, True, device)
    xte, yte = cifar_tensors(root, False, device)
    if use_victim_labels:
        vtr, vte, nq = victim_labels(oracle=oracle, root=root, device=device)
        agree = (vtr == ytr).float().mean().item()
        log(f"  [{name}] training on VICTIM labels "
            f"({100*agree:.2f}% agree with ground truth, {nq} queries)")
        ytr, yte = vtr, vte
    net = Normalized(build(name)).to(device).to(memory_format=torch.channels_last)

    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    steps = epochs * (xtr.shape[0] // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.25)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smooth)
    scaler = torch.amp.GradScaler(device)
    step = 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(xtr.shape[0], device=device)
        for i in range(0, xtr.shape[0] - bs + 1, bs):
            idx = perm[i:i + bs]
            xb = augment(xtr[idx]).to(memory_format=torch.channels_last)
            yb = ytr[idx]
            if adv_eps > 0:
                d = (torch.rand_like(xb) * 2 - 1) * adv_eps
                d = (xb + d).clamp(0, 1) - xb
                d.requires_grad_(True)
                with torch.amp.autocast(device):
                    loss = crit(net(xb + d), yb)
                g, = torch.autograd.grad(scaler.scale(loss), d)
                d = (d + 1.25 * adv_eps * g.sign()).clamp(-adv_eps, adv_eps).detach()
                xb = (xb + d).clamp(0, 1)
            with torch.amp.autocast(device):
                loss = crit(net(xb), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if step < steps - 1:
                sched.step()
            step += 1
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            log(f"  [{name}] epoch {ep+1}/{epochs}  loss {loss.item():.3f}  "
                f"test acc {evaluate(net, xte, yte)*100:.2f}%")
    path = os.path.join(out_dir, f"{name}_eps{adv_eps:.4f}_s{seed}.pt")
    torch.save({"name": name, "state": net.state_dict(), "adv_eps": adv_eps}, path)
    log(f"  [{name}] saved -> {path}")
    return net, path


@torch.no_grad()
def evaluate(net, x_uint8, y, bs=512):
    net.eval()
    correct = 0
    for i in range(0, x_uint8.shape[0], bs):
        xb = x_uint8[i:i + bs].float().div(255.0)
        correct += (net(xb).argmax(1) == y[i:i + bs]).sum().item()
    return correct / x_uint8.shape[0]


def load_surrogates(paths, device="cuda"):
    nets = []
    for p in paths:
        ck = torch.load(p, map_location=device, weights_only=False)
        net = Normalized(build(ck["name"])).to(device)
        net.load_state_dict(ck["state"])
        net.eval()
        for q in net.parameters():
            q.requires_grad_(False)
        nets.append(net)
    return nets


# --------------------------------------------------- ensemble gradient / attack
def ensemble_loss_grad(nets, x01, y, targeted_logit_margin=True):
    """Gradient of a margin loss of the surrogate *ensemble* w.r.t. the image.

    Returns a unit-norm direction that, white-box on the surrogates, increases
    the chance of a label flip.  The victim never sees any of this.
    """
    x = x01.clone().requires_grad_(True)
    total = 0.0
    for net in nets:
        logits = net(x)
        if targeted_logit_margin:
            true_logit = logits.gather(1, y.view(-1, 1)).squeeze(1)
            other = logits.scatter(1, y.view(-1, 1), float("-inf")).max(1).values
            total = total + (other - true_logit).mean()
        else:
            total = total - F.cross_entropy(logits, y)
    g, = torch.autograd.grad(total, x)
    return g / (g.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12)


def ensemble_pgd_direction(nets, x01, y, eps=16 / 255, steps=30, norm="l2",
                           momentum=1.0):
    """MI-FGSM / PGD on the surrogate ensemble; returns the perturbation
    direction (unit L2 norm) to be line-searched against the victim."""
    x = x01.clone()
    delta = torch.zeros_like(x)
    g_acc = torch.zeros_like(x)
    alpha = 2.5 * eps / steps
    for _ in range(steps):
        xd = (x + delta).clamp(0, 1)
        g = ensemble_loss_grad(nets, xd, y)
        g_acc = momentum * g_acc + g / (g.flatten(1).abs().mean(1).view(-1, 1, 1, 1) + 1e-12)
        if norm == "linf":
            delta = (delta + alpha * g_acc.sign()).clamp(-eps, eps)
        else:
            step = alpha * g_acc / (g_acc.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12)
            delta = delta + step
            dn = delta.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
            delta = delta * torch.clamp(eps / (dn + 1e-12), max=1.0)
        delta = (x + delta).clamp(0, 1) - x
    d = delta
    return d / (d.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12)


def ensemble_ddn(nets, x01, y, steps=100, gamma=0.05, init_norm=1.0,
                 quorum=None):
    """Minimum-L2 attack on the surrogate ensemble (Rony et al., CVPR 2019).

    Decoupled direction and norm: follow the loss gradient for the direction
    and shrink/grow the norm depending on whether the current iterate already
    fools the ensemble.  The *direction* of the result is our best white-box
    guess at where the victim's boundary is closest; the victim itself decides
    the radius afterwards.
    """
    n = x01.shape[0]
    quorum = quorum if quorum is not None else max(1, (len(nets) + 1) // 2)
    delta = torch.zeros_like(x01)
    eps = torch.full((n,), init_norm, device=x01.device)
    best = torch.zeros_like(x01)
    best_n = torch.full((n,), float("inf"), device=x01.device)

    for k in range(steps):
        lr = 1.0 * (0.01 ** (k / max(1, steps - 1)))      # cosine-ish decay
        xd = (x01 + delta).clamp(0, 1).requires_grad_(True)
        loss, votes = 0.0, 0
        for net in nets:
            lg = net(xd)
            loss = loss + F.cross_entropy(lg, y, reduction="sum")
            votes = votes + (lg.argmax(1) != y).long()
        g, = torch.autograd.grad(loss, xd)
        g = g / (g.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12)

        is_adv = votes >= quorum
        dn = delta.flatten(1).norm(dim=1)
        better = is_adv & (dn < best_n)
        best[better] = delta[better]
        best_n[better] = dn[better]

        delta = delta + lr * g
        eps = torch.where(is_adv, eps * (1 - gamma), eps * (1 + gamma))
        dn = delta.flatten(1).norm(dim=1).clamp(min=1e-12)
        delta = delta * (eps / dn).view(-1, 1, 1, 1)
        delta = ((x01 + delta).clamp(0, 1) - x01).detach()

    # images the ensemble never flipped: fall back to the last iterate
    miss = torch.isinf(best_n)
    best[miss] = delta[miss]
    d = best
    return d / (d.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12), best_n


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="resnet18,vgg,resnet10")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--adv-eps", type=float, default=0.0)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ground-truth", action="store_true")
    a = p.parse_args()
    import time
    for i, m in enumerate(a.models.split(",")):
        t = time.time()
        train_one(m, epochs=a.epochs, adv_eps=a.adv_eps, seed=i, device=a.device,
                  bs=a.bs, lr=a.lr, use_victim_labels=not a.ground_truth)
        print(f"  [{m}] trained in {time.time()-t:.0f}s")
