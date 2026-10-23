"""
Why the composite loss of Eq. (9) trains more weakly than plain cross-entropy.

The report claims two mechanisms:

  1. the focal factor (1 - q)^gamma switches the cost-sensitive term off as soon
     as the student becomes confident, which on a near-saturated benchmark is
     almost immediately;
  2. without Hinton's T^2 correction the divergence term's gradient is scaled by
     1/T, so at T = 4 it carries a quarter of its usual weight.

Both are properties of the specification, not of any particular run, so they can
be measured directly: sweep the student's confidence on a correct prediction and
plot the gradient each term delivers to the logits.  Writes
figures/fig_loss_diagnosis.png and results/loss_diagnosis.json.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from config import (ALPHA, BETA, FIGURES, FOCAL_ALPHA, FOCAL_GAMMA, RESULTS,
                    TEMPERATURE)


def logits_for_confidence(p: float, n_classes: int = 10) -> torch.Tensor:
    """Logits whose softmax puts probability `p` on class 0, rest spread evenly."""
    rest = (1.0 - p) / (n_classes - 1)
    probs = torch.full((1, n_classes), rest)
    probs[0, 0] = p
    return torch.log(probs)


def grad_norm(loss: torch.Tensor, z: torch.Tensor) -> float:
    g, = torch.autograd.grad(loss, z, retain_graph=True)
    return float(g.norm())


def sweep(n_classes: int = 10, teacher_conf: float = 0.95):
    conf = np.linspace(0.15, 0.999, 60)
    out = {k: [] for k in ("ce", "cs", "kl", "kl_t2")}
    zt = logits_for_confidence(teacher_conf, n_classes)
    y = torch.zeros(1, dtype=torch.long)

    for p in conf:
        z = logits_for_confidence(float(p), n_classes).clone().requires_grad_(True)

        # plain cross-entropy -- the no-distillation control
        out["ce"].append(grad_norm(F.cross_entropy(z, y), z))

        # Eq. (7) exactly as specified: constant alpha_i = 0.25, gamma = 2
        q = F.softmax(z, 1)
        focus = (1.0 - q).clamp_min(0).pow(FOCAL_GAMMA)
        onehot = F.one_hot(y, n_classes).float()
        cs = -(FOCAL_ALPHA * focus * onehot * torch.log_softmax(z, 1)).sum(1).mean()
        out["cs"].append(grad_norm(cs, z))

        # Eq. (6) as written -- no T^2 factor -- and the Hinton-corrected variant
        T = TEMPERATURE
        pt = F.softmax(zt / T, 1)
        kl = (pt * (torch.log(pt.clamp_min(1e-8)) - F.log_softmax(z / T, 1))).sum(1).mean()
        out["kl"].append(grad_norm(kl, z))
        out["kl_t2"].append(grad_norm(kl * T ** 2, z))

    return conf, {k: np.array(v) for k, v in out.items()}


def main():
    conf, g = sweep()
    # the weights Eq. (9) actually applies
    weighted_cs = BETA * g["cs"]
    weighted_kl = ALPHA * g["kl"]
    weighted_kl_t2 = ALPHA * g["kl_t2"]
    total_paper = weighted_cs + weighted_kl

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))

    ax = axes[0]
    ax.plot(conf, g["ce"], lw=2, color="#1F3864", label="cross-entropy (no KD)")
    ax.plot(conf, weighted_cs, lw=2, color="#C0504D",
            label=r"$\beta \cdot L_{CS}$  (Eq. 7, $\alpha_i=0.25$)")
    ax.plot(conf, weighted_kl, lw=2, color="#ED7D31",
            label=r"$\alpha \cdot L_{KL}$  (Eq. 6, no $T^2$)")
    ax.plot(conf, total_paper, lw=2.4, ls="--", color="black",
            label="Eq. (9) total (paper)")
    ax.set_xlabel("student confidence on the correct class")
    ax.set_ylabel(r"$\|\partial L / \partial z\|$")
    ax.set_yscale("log")
    ax.set_title("Gradient delivered to the logits")
    ax.legend(fontsize=7.5)

    ax = axes[1]
    ratio = total_paper / np.maximum(g["ce"], 1e-12)
    ratio_fixed = (weighted_cs + weighted_kl_t2) / np.maximum(g["ce"], 1e-12)
    ax.plot(conf, ratio, lw=2.2, color="#C0504D", label="Eq. (9) as written")
    ax.plot(conf, ratio_fixed, lw=2.2, color="#70AD47",
            label=r"with Hinton's $T^2$ restored")
    ax.axhline(1.0, color="grey", lw=1, ls=":")
    ax.set_xlabel("student confidence on the correct class")
    ax.set_ylabel("gradient relative to cross-entropy")
    ax.set_yscale("log")
    ax.set_title("The composite loss is weaker than plain CE\n"
                 "over the whole confident region")
    ax.legend(fontsize=8)

    fig.suptitle("Why Eq. (9) under-trains a near-saturated student",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_loss_diagnosis.png")
    plt.close(fig)

    at = {}
    for p in (0.5, 0.9, 0.99):
        i = int(np.argmin(np.abs(conf - p)))
        at[f"conf_{p}"] = {
            "ce": float(g["ce"][i]),
            "beta_cs": float(weighted_cs[i]),
            "alpha_kl": float(weighted_kl[i]),
            "total_paper": float(total_paper[i]),
            "ratio_to_ce": float(ratio[i]),
            "ratio_to_ce_with_T2": float(ratio_fixed[i]),
        }
    (RESULTS / "loss_diagnosis.json").write_text(
        json.dumps({"temperature": TEMPERATURE, "alpha": ALPHA, "beta": BETA,
                    "focal_alpha": FOCAL_ALPHA, "focal_gamma": FOCAL_GAMMA,
                    "at": at}, indent=2), encoding="utf-8")

    for k, v in at.items():
        print(f"{k}: CE={v['ce']:.4f}  beta*CS={v['beta_cs']:.4f}  "
              f"alpha*KL={v['alpha_kl']:.4f}  total/CE={v['ratio_to_ce']:.3f}  "
              f"(with T^2: {v['ratio_to_ce_with_T2']:.3f})")


if __name__ == "__main__":
    main()
