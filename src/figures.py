"""
Turns results/runs.jsonl + results/complexity.json into the figures and the
LaTeX table fragments used by the phase-2 report.

Figures are written to figures/, LaTeX fragments to report/tables/.
Every panel that has a counterpart in the paper is drawn next to the paper's
own number so the two can be read together.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from config import (BEST_COMBINATION, DATASETS, FIGURES, RESULTS,
                    WEIGHT_COMBINATIONS)

TABLES = RESULTS.parent / "report" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 9, "axes.grid": True, "grid.alpha": .25,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
})

C_OURS = "#1F3864"
C_PAPER = "#C0504D"
C_ALT = ["#4472C4", "#70AD47", "#ED7D31", "#7030A0", "#C00000", "#2E75B6"]
METRICS = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
MET_LABEL = {"accuracy": "Accuracy", "macro_precision": "Precision",
             "macro_recall": "Recall", "macro_f1": "F1-Score"}

# ---- the paper's own numbers, for side-by-side plotting -------------------- #
PAPER_TABLE_VI = {
    "ToN_IoT": {
        "1DCNN": (97.85, 93.52, 91.32, 92.41), "SNN": (97.50, 92.99, 90.24, 91.59),
        "RNN": (98.10, 94.20, 91.51, 92.83), "TCN": (98.35, 94.98, 91.92, 93.43),
        "1DCNN-BiLSTM": (99.54, 95.23, 94.50, 94.86),
        "LMT-SDNN": (99.69, 96.50, 95.31, 95.90)},
    "BoT_IoT": {
        "1DCNN": (97.90, 96.65, 95.72, 96.18), "SNN": (97.65, 96.36, 95.20, 95.78),
        "RNN": (98.10, 96.92, 95.83, 96.37), "TCN": (98.40, 97.37, 96.21, 96.78),
        "1DCNN-BiLSTM": (99.35, 98.20, 97.56, 97.88),
        "LMT-SDNN": (99.86, 98.93, 98.45, 98.63)},
}
PAPER_TEACHERS = {          # read off Fig. 4
    "ToN_IoT": {"Teacher1-ResidualInception": (98.9, 95.9, 95.6, 95.8),
                "Teacher2-1DCNN-BiLSTM": (98.6, 95.8, 95.3, 95.6)},
    "BoT_IoT": {"Teacher1-ResidualInception": (99.2, 97.9, 97.0, 97.4),
                "Teacher2-1DCNN-BiLSTM": (99.1, 97.7, 96.9, 97.2)},
}


# --------------------------------------------------------------------------- #
def load_runs() -> list[dict]:
    f = RESULTS / "runs.jsonl"
    if not f.exists():
        return []
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]


def pick(runs, **kw):
    out = []
    for r in runs:
        c = r["config"]
        if all(c.get(k) == v for k, v in kw.items()):
            out.append(r)
    return out


def one(runs, **kw):
    r = pick(runs, **kw)
    return r[-1] if r else None


def vals(run) -> tuple:
    m = run["metrics"]
    return tuple(m[k] for k in METRICS)


def _bar_pair(ax, labels, ours, paper, title, ylabel="%", ylim=None):
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, ours, w, label="This reproduction", color=C_OURS)
    if paper is not None:
        ax.bar(x + w / 2, paper, w, label="Reported in the paper", color=C_PAPER,
               hatch="//", alpha=.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)


# --------------------------------------------------------------------------- #
#  Fig A -- Table VI reproduction
# --------------------------------------------------------------------------- #
def fig_main(runs):
    order = ["1DCNN", "SNN", "RNN", "TCN", "1DCNN-BiLSTM", "LMT-SDNN"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for r, ds in enumerate(DATASETS):
        for c, met in enumerate(METRICS):
            ax = axes[r, c]
            ours, paper, labels = [], [], []
            for m in order:
                run = one(runs, stage="main", dataset=ds, model=m)
                if run is None:
                    continue
                labels.append("LMT-SDNN\n(ours)" if m == "LMT-SDNN" else m)
                ours.append(run["metrics"][met])
                paper.append(PAPER_TABLE_VI[ds][m][METRICS.index(met)])
            if not ours:
                continue
            lo = min(min(ours), min(paper)) - 2
            _bar_pair(ax, labels, ours, paper, f"{ds} - {MET_LABEL[met]}",
                      ylim=(max(0, lo), 100.4))
    handles = [Patch(facecolor=C_OURS, label="This reproduction"),
               Patch(facecolor=C_PAPER, hatch="//", label="Reported in the paper")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Table VI reproduction: distilled students, macro-averaged\n"
                 "(y-axes are cropped a couple of points below the smallest bar "
                 "so the differences are legible)", fontweight="bold", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGURES / "fig_table6_main.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Fig B -- distillation gain (paper Fig. 7)
# --------------------------------------------------------------------------- #
def fig_distill_gain(runs):
    order = ["1DCNN", "SNN", "RNN", "TCN", "1DCNN-BiLSTM", "LMT-SDNN"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, ds in zip(axes, DATASETS):
        labels, kd, nokd = [], [], []
        for m in order:
            a = one(runs, stage="main", dataset=ds, model=m)
            b = one(runs, stage="nodistill", dataset=ds, model=m)
            if a is None or b is None:
                continue
            labels.append(m)
            kd.append(a["metrics"]["macro_f1"])
            nokd.append(b["metrics"]["macro_f1"])
        if not labels:
            continue
        x = np.arange(len(labels))
        ax.bar(x - .2, nokd, .4, label="without distillation", color="#9DC3E6")
        ax.bar(x + .2, kd, .4, label="with multi-teacher KD", color=C_OURS)
        for i, (a, b) in enumerate(zip(nokd, kd)):
            ax.annotate(f"{b-a:+.2f}", (i, max(a, b) + .25), ha="center",
                        fontsize=7, color=("#1B7F3B" if b >= a else "#B22222"))
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
        lo, hi = min(nokd + kd), max(nokd + kd)
        pad = max(0.35, (hi - lo) * 0.35)
        ax.set_ylim(lo - pad, hi + pad * 2.2)      # headroom for labels + legend
        ax.set_ylabel("macro F1 (%)"); ax.set_title(ds)
        ax.legend(fontsize=7, loc="upper left", ncol=2)
    fig.suptitle("Effect of multi-teacher distillation (paper Fig. 7)", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGURES / "fig_distillation_gain.png"); plt.close(fig)


# --------------------------------------------------------------------------- #
#  Fig C -- teacher ablation (paper Fig. 4)
# --------------------------------------------------------------------------- #
def fig_teacher_ablation(runs):
    tags = [("teacher1-only", "Teacher 1 only\n(Residual Inception)"),
            ("teacher2-only", "Teacher 2 only\n(1D-CNN+BiLSTM)"),
            ("both", "Both teachers\n(LMT-SDNN)")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, ds in zip(axes, DATASETS):
        x = np.arange(len(METRICS)); w = 0.26
        for i, (tag, lab) in enumerate(tags):
            run = one(runs, stage="tablation", dataset=ds, tag=tag)
            if run is None:
                continue
            ax.bar(x + (i - 1) * w, vals(run), w, label=lab, color=C_ALT[i])
        ax.set_xticks(x); ax.set_xticklabels([MET_LABEL[m] for m in METRICS])
        ax.set_title(ds); ax.set_ylabel("%")
        ax.set_ylim(80, 101)          # honest axis, unlike the paper's 70 % floor
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Single-teacher vs multi-teacher distillation (paper Fig. 4)",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGURES / "fig_teacher_ablation.png"); plt.close(fig)


# --------------------------------------------------------------------------- #
#  Fig D -- weight combinations (paper Fig. 2 radar)
# --------------------------------------------------------------------------- #
def fig_weights(runs):
    """
    The paper shows this sweep as a radar plot on a 0-100 axis (its Fig. 2), on
    which every combination is indistinguishable.  We draw the same data on an
    axis scaled to the actual spread, and print that spread, so the reader can
    see how small the effect being argued about really is.
    """
    combos = list(WEIGHT_COMBINATIONS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, ds in zip(axes, DATASETS):
        labs, f1, rec = [], [], []
        for cb in combos:
            run = one(runs, stage="weights", dataset=ds, tag=cb)
            if run is None:
                continue
            labs.append(f"{cb}\n{WEIGHT_COMBINATIONS[cb]}")
            f1.append(run["metrics"]["macro_f1"])
            rec.append(run["metrics"]["macro_recall"])
        if not labs:
            continue
        x = np.arange(len(labs))
        best_i = int(np.argmax(f1))
        colors = [C_OURS] * len(labs)
        paper_i = combos.index(BEST_COMBINATION) if BEST_COMBINATION in combos else -1
        ax.plot(x, f1, "-", color="#BDD7EE", lw=1.5, zorder=1)
        for i in range(len(labs)):
            ax.scatter(x[i], f1[i], s=90, zorder=3,
                       color=("#70AD47" if i == best_i else colors[i]),
                       marker=("D" if i == paper_i else "o"),
                       edgecolor="black", linewidth=.6)
            ax.annotate(f"{f1[i]:.2f}", (x[i], f1[i]), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=7.5)
        span = max(f1) - min(f1)
        ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=7)
        ax.set_ylabel("macro F1 (%)")
        ax.set_ylim(min(f1) - span * 0.8 - .05, max(f1) + span * 0.9 + .05)
        ax.set_title(f"{ds}   (total spread = {span:.2f} pp)")
    handles = [plt.Line2D([], [], marker="D", ls="", color=C_OURS, markeredgecolor="k",
                          label="Comb-4: the paper's choice"),
               plt.Line2D([], [], marker="o", ls="", color="#70AD47", markeredgecolor="k",
                          label="best in this reproduction")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(r"Loss-weight combinations $(\alpha,\beta,\gamma)$ of Table III"
                 "\n(axis scaled to the real spread; the paper's radar plot runs 0-100)",
                 fontweight="bold", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURES / "fig_weight_combos.png"); plt.close(fig)


# --------------------------------------------------------------------------- #
#  Fig E -- loss ablation
# --------------------------------------------------------------------------- #
def fig_loss_ablation(runs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, ds in zip(axes, DATASETS):
        rows = pick(runs, stage="lossablation", dataset=ds)
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: r["metrics"]["macro_f1"])
        labs = [r["config"]["tag"] for r in rows]
        f1 = [r["metrics"]["macro_f1"] for r in rows]
        rec = [r["metrics"]["macro_recall"] for r in rows]
        y = np.arange(len(labs))
        ax.barh(y - .2, f1, .4, label="macro F1", color=C_OURS)
        ax.barh(y + .2, rec, .4, label="macro Recall", color="#9DC3E6")
        for i, v in enumerate(f1):
            ax.annotate(f"{v:.2f}", (v + .15, i - .2), va="center", fontsize=7)
        ax.set_yticks(y); ax.set_yticklabels(labs)
        ax.set_xlim(min(f1 + rec) - 3, 101); ax.set_title(ds); ax.set_xlabel("%")
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Contribution of each term of Eq. (9)", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGURES / "fig_loss_ablation.png"); plt.close(fig)


# --------------------------------------------------------------------------- #
#  Fig F -- temperature sensitivity (a hyper-parameter the paper never states)
# --------------------------------------------------------------------------- #
def fig_temperature(runs):
    fig, ax = plt.subplots(figsize=(7, 4))
    drawn = False
    for j, ds in enumerate(DATASETS):
        rows = sorted(pick(runs, stage="temperature", dataset=ds),
                      key=lambda r: r["config"]["temperature"])
        if not rows:
            continue
        drawn = True
        t = [r["config"]["temperature"] for r in rows]
        for k, met in enumerate(["macro_f1", "macro_recall"]):
            ax.plot(t, [r["metrics"][met] for r in rows], "o-",
                    color=C_ALT[j], ls="-" if k == 0 else "--", ms=4,
                    label=f"{ds} - {MET_LABEL[met]}")
    if drawn:
        ax.set_xlabel("distillation temperature T"); ax.set_ylabel("%")
        ax.set_title("Sensitivity to a hyper-parameter the paper never reports")
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGURES / "fig_temperature.png"); plt.close(fig)


# --------------------------------------------------------------------------- #
#  Fig G -- the alpha_i question in Eq. (7)
# --------------------------------------------------------------------------- #
def fig_alpha(runs):
    modes = [("constant", r"$\alpha_i=0.25$  (as written)"),
             ("inverse_freq", r"$\alpha_i \propto 1/n_i$"),
             ("effective_number", r"$\alpha_i$ effective-number")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, ds in zip(axes, DATASETS):
        x = np.arange(2); w = .26
        for i, (mode, lab) in enumerate(modes):
            run = one(runs, stage="csalpha", dataset=ds, tag=mode)
            if run is None:
                continue
            m = run["metrics"]
            ax.bar(x + (i - 1) * w, [m["macro_f1"], m["macro_recall"]], w,
                   label=lab, color=C_ALT[i])
        ax.set_xticks(x); ax.set_xticklabels(["macro F1", "macro Recall"])
        ax.set_title(ds); ax.set_ylabel("%"); ax.legend(fontsize=7, loc="lower right")
        ax.set_ylim(80, 101)
    fig.suptitle(r"Eq. (7): a constant $\alpha_i$ cannot re-weight classes",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGURES / "fig_alpha_mode.png"); plt.close(fig)


# --------------------------------------------------------------------------- #
#  Fig H -- stability across seeds (paper Fig. 3)
# --------------------------------------------------------------------------- #
def fig_stability(runs):
    order = ["1DCNN", "SNN", "RNN", "TCN", "1DCNN-BiLSTM", "LMT-SDNN"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, ds in zip(axes, DATASETS):
        data, labs = [], []
        for m in order:
            rows = pick(runs, stage="seeds", dataset=ds, model=m)
            if len(rows) < 2:
                continue
            data.append([r["metrics"]["macro_f1"] for r in rows])
            labs.append(m)
        if not data:
            continue
        bp = ax.boxplot(data, labels=labs, patch_artist=True, widths=.55)
        for i, b in enumerate(bp["boxes"]):
            b.set_facecolor(C_OURS if labs[i] == "LMT-SDNN" else "#BDD7EE")
            b.set_alpha(.9)
        for med in bp["medians"]:
            med.set_color("black")
        ax.set_xticklabels(labs, rotation=20, ha="right")
        ax.set_ylabel("macro F1 (%)"); ax.set_title(ds)
    fig.suptitle("Run-to-run stability over independent seeds (paper Fig. 3)",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGURES / "fig_stability.png"); plt.close(fig)


# --------------------------------------------------------------------------- #
#  Fig I -- confusion matrices (paper Fig. 6)
# --------------------------------------------------------------------------- #
def fig_confusion(runs):
    for ds, spec in DATASETS.items():
        run = one(runs, stage="main", dataset=ds, model="LMT-SDNN")
        if run is None:
            continue
        cm = np.array(run["metrics"]["confusion_matrix"], dtype=float)
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
        n = len(spec.classes)
        fig, ax = plt.subplots(figsize=(1.05 * n + 3, 0.95 * n + 2.6))
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(spec.classes, rotation=45, ha="right")
        ax.set_yticklabels(spec.classes)
        for i in range(n):
            for j in range(n):
                if norm[i, j] >= 0.05:
                    ax.text(j, i, f"{norm[i,j]:.1f}", ha="center", va="center",
                            fontsize=7,
                            color="white" if norm[i, j] > 55 else "black")
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"LMT-SDNN confusion matrix - {ds}  (row-normalised, %)")
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=.8, label="% of true class")
        fig.tight_layout(); fig.savefig(FIGURES / f"fig_confusion_{ds}.png")
        plt.close(fig)


# --------------------------------------------------------------------------- #
#  Fig J -- what the headline number is actually measuring
# --------------------------------------------------------------------------- #
def fig_validity(runs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    settings = [("main", None, "Paper protocol\n(random split)"),
                ("leakage", "host IPs as features", "+ host IP\nas a feature"),
                ("quota", "proportional quotas", "Proportional (imbalanced)\nclass quotas"),
                ("blocksplit", "contamination-free split", "Contamination-free\nsplit")]
    for ax, ds in zip(axes, DATASETS):
        labs, f1, acc = [], [], []
        for stage, tag, lab in settings:
            run = (one(runs, stage=stage, dataset=ds, model="LMT-SDNN")
                   if tag is None else one(runs, stage=stage, dataset=ds, tag=tag))
            if run is None:
                continue
            labs.append(lab)
            f1.append(run["metrics"]["macro_f1"])
            acc.append(run["metrics"]["accuracy"])
        if not labs:
            continue
        x = np.arange(len(labs))
        ax.bar(x - .2, acc, .4, label="Accuracy", color="#9DC3E6")
        ax.bar(x + .2, f1, .4, label="macro F1", color=C_OURS)
        base = f1[0]
        for i, (a, v) in enumerate(zip(acc, f1)):
            ax.annotate(f"{v:.2f}", (i + .2, v + .18), ha="center", fontsize=7)
            ax.annotate(f"{a:.2f}", (i - .2, a + .18), ha="center", fontsize=7,
                        color="#333333")
            if i:
                d = v - base
                ax.annotate(f"{d:+.2f}", (i, min(a, v) - 1.15), ha="center",
                            fontsize=7.5, fontweight="bold",
                            color=("#1B7F3B" if d >= 0 else "#B22222"))
        ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8)
        lo = min(acc + f1)
        # Axis is truncated because every value sits in a narrow high band; the
        # range is stated on the axis label so the reader is not misled.
        ax.set_ylim(lo - 3.0, 101.2)
        ax.set_ylabel(f"% (axis starts at {lo-3.0:.0f})")
        ax.set_title(ds)
        ax.legend(fontsize=7, loc="lower left", ncol=2)
    fig.suptitle("How much of the headline score survives a stricter protocol?",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGURES / "fig_validity.png"); plt.close(fig)


# --------------------------------------------------------------------------- #
#  Fig K -- the proposal
# --------------------------------------------------------------------------- #
def fig_proposal(runs):
    variants = [("main", None, "LMT-SDNN\n(reproduced)"),
                ("proposal", "+adaptive teachers", "+ entropy-gated\nteachers"),
                ("proposal", "+class-aware alpha", "+ class-aware\n" + r"$\alpha_i$"),
                ("proposal", "LMT-SDNN++", "LMT-SDNN++\n(both)")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, ds in zip(axes, DATASETS):
        labs, f1, rec = [], [], []
        for stage, tag, lab in variants:
            run = (one(runs, stage=stage, dataset=ds, model="LMT-SDNN")
                   if tag is None else one(runs, stage=stage, dataset=ds, tag=tag))
            if run is None:
                continue
            labs.append(lab); f1.append(run["metrics"]["macro_f1"])
            rec.append(run["metrics"]["macro_recall"])
        if not labs:
            continue
        x = np.arange(len(labs))
        ax.bar(x - .2, f1, .4, label="macro F1", color=C_OURS)
        ax.bar(x + .2, rec, .4, label="macro Recall", color="#70AD47")
        base = f1[0]
        for i, v in enumerate(f1):
            ax.annotate(f"{v-base:+.2f}" if i else "baseline", (i - .2, v + .2),
                        ha="center", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8)
        ax.set_ylim(min(f1 + rec) - 2, 101); ax.set_ylabel("%"); ax.set_title(ds)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Proposed extension: entropy-gated teachers + class-aware cost",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGURES / "fig_proposal.png"); plt.close(fig)


# --------------------------------------------------------------------------- #
#  Fig L -- compression
# --------------------------------------------------------------------------- #
def fig_compression():
    f = RESULTS / "complexity.json"
    if not f.exists():
        return
    data = json.loads(f.read_text(encoding="utf-8"))["models"]
    order = ["Teacher(Total)", "1DCNN", "SNN", "RNN", "TCN", "1DCNN-BiLSTM", "LMT-SDNN"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, ds in zip(axes, DATASETS):
        rows = data.get(ds, {})
        labs = [m for m in order if m in rows]
        p = [rows[m]["params"] for m in labs]
        ax.bar(labs, p, color=[C_PAPER if m == "Teacher(Total)" else
                               (C_OURS if m == "LMT-SDNN" else "#9DC3E6") for m in labs])
        ax.set_yscale("log"); ax.set_ylabel("parameters (log scale)")
        for i, v in enumerate(p):
            ax.annotate(f"{v:,}", (i, v * 1.15), ha="center", fontsize=6.5, rotation=90)
        ax.set_xticklabels(labs, rotation=20, ha="right")
        ax.set_title(ds); ax.set_ylim(top=max(p) * 6)
    fig.suptitle("Model size across the distillation family (Table VII)",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGURES / "fig_compression.png"); plt.close(fig)


# --------------------------------------------------------------------------- #
#  LaTeX tables
# --------------------------------------------------------------------------- #
def _w(name: str, body: str):
    (TABLES / name).write_text(body, encoding="utf-8")


def _esc(s: str) -> str:
    """Escape the LaTeX-special characters that occur in our identifiers."""
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"),
                 ("&", r"\&"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def tex_table6(runs):
    order = ["1DCNN", "SNN", "RNN", "TCN", "1DCNN-BiLSTM", "LMT-SDNN"]
    L = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
         r"& & \multicolumn{4}{c}{\lr{this reproduction}} & "
         r"\multicolumn{2}{c}{\lr{paper}}\\",
         r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}",
         r"\lr{Dataset} & \lr{Model} & \lr{Acc} & \lr{Prec} & \lr{Rec} & "
         r"\lr{F1} & \lr{Acc} & \lr{F1}\\", r"\midrule"]
    for ds in DATASETS:
        first = True
        for m in order:
            run = one(runs, stage="main", dataset=ds, model=m)
            if run is None:
                continue
            a, p, r, f = vals(run)
            pa, _, _, pf = PAPER_TABLE_VI[ds][m]
            nm = r"\textbf{\lr{LMT-SDNN}}" if m == "LMT-SDNN" else rf"\lr{{{_esc(m)}}}"
            head = rf"\lr{{{_esc(ds)}}}" if first else ""
            first = False
            L.append(rf"{head} & {nm} & \num{{{a:.2f}}} & \num{{{p:.2f}}} & "
                     rf"\num{{{r:.2f}}} & \num{{{f:.2f}}} & \num{{{pa:.2f}}} & "
                     rf"\num{{{pf:.2f}}}\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    _w("table6.tex", "\n".join(L))


def tex_table7():
    f = RESULTS / "complexity.json"
    if not f.exists():
        return
    d = json.loads(f.read_text(encoding="utf-8"))
    from complexity import PAPER_TABLE_VII
    name_map = {"Teacher(Total)": "Teacher(Total)", "1DCNN": "KD-1DCNN",
                "SNN": "KD-SNN", "RNN": "KD-RNN", "TCN": "KD-TCN",
                "1DCNN-BiLSTM": "KD-1DCNN-BiLSTM", "LMT-SDNN": "LMT-SDNN(ours)"}
    L = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r"\lr{Dataset} & \lr{Model} & \lr{params (ours)} & \lr{params (paper)} & "
         r"\lr{size KB (ours)} & \lr{$\downarrow$ vs teacher}\\", r"\midrule"]
    for ds in DATASETS:
        first = True
        for k, pk in name_map.items():
            rows = d["models"].get(ds, {})
            if k not in rows:
                continue
            r = rows[k]
            pv = PAPER_TABLE_VII[pk]
            pp = pv[0] if ds == "ToN_IoT" else pv[3]
            drop = r.get("params_drop_pct")
            nm = r"\textbf{\lr{LMT-SDNN}}" if k == "LMT-SDNN" else rf"\lr{{{_esc(pk)}}}"
            head = rf"\lr{{{_esc(ds)}}}" if first else ""
            first = False
            # `\reftag` is defined in the preamble; a bare "---" would ask the
            # Persian body font for an em-dash ligature it does not carry
            L.append(rf"{head} & {nm} & \num{{{r['params']:,}}} & \num{{{pp:,}}} & "
                     rf"\num{{{r['size_kb']:.2f}}} & "
                     + (rf"\num{{{drop:.2f}\%}}\\" if drop is not None else r"\reftag\\"))
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    _w("table7.tex", "\n".join(L))


def tex_audit():
    from complexity import audit_paper_table_vii
    L = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r"\lr{Model} & \lr{Dataset} & \lr{params} & \lr{size KB} & "
         r"\lr{bytes/param} & \lr{FLOPs/param}\\", r"\midrule"]
    for r in audit_paper_table_vii():
        bad = not r["consistent_fp32"]
        bpp = rf"\down{{\num{{{r['bytes_per_param_implied']:.2f}}}}}" if bad \
            else rf"\num{{{r['bytes_per_param_implied']:.2f}}}"
        L.append(rf"\lr{{{_esc(r['model'])}}} & \lr{{{_esc(r['dataset'])}}} & "
                 rf"\num{{{r['reported_params']:,}}} & "
                 rf"\num{{{r['reported_size_kb']:.2f}}} & {bpp} & "
                 rf"\num{{{r['flops_per_param']:.2f}}}\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    _w("table_audit.tex", "\n".join(L))


def tex_dataset():
    import json as _j
    L = [r"\begin{tabular}{llrrr}", r"\toprule",
         r"\lr{Dataset} & \lr{Class} & \lr{train} & \lr{test} & \lr{total}\\",
         r"\midrule"]
    from config import DATA_PROC, SEED
    for ds in DATASETS:
        meta_f = DATA_PROC / f"{ds}_seed{SEED}_record.json"
        if not meta_f.exists():
            continue
        meta = _j.loads(meta_f.read_text(encoding="utf-8"))
        first = True
        for c in meta["classes"]:
            tr = meta["train_class_counts"][c]
            te = meta["test_class_counts"][c]
            head = rf"\lr{{{_esc(ds)}}}" if first else ""
            first = False
            L.append(rf"{head} & \lr{{{c}}} & \num{{{tr:,}}} & \num{{{te:,}}} & "
                     rf"\num{{{tr+te:,}}}\\")
        L.append(r"\midrule")
        L.append(rf"& \textbf{{\lr{{total}}}} & \textbf{{\num{{{meta['n_train']:,}}}}} & "
                 rf"\textbf{{\num{{{meta['n_test']:,}}}}} & "
                 rf"\textbf{{\num{{{meta['n_train']+meta['n_test']:,}}}}}\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    _w("table_dataset.tex", "\n".join(L))


def tex_numbers(runs):
    """
    Emit \\newcommand definitions for every number quoted in the prose, so the
    report text can never drift away from the experiments that produced it.
    """
    out: list[str] = [r"% auto-generated by figures.py -- do not edit"]

    def cmd(name, value, fmt="{:.2f}"):
        # Wrap in \lr{}: a bare "97.46" dropped into RTL prose gets reordered by
        # the bidi algorithm and comes out as 46/97.
        v = fmt.format(value) if isinstance(value, (int, float)) else value
        out.append(rf"\newcommand{{\{name}}}{{\lr{{{v}}}}}")

    short = {"ToN_IoT": "Ton", "BoT_IoT": "Bot"}
    met_short = {"accuracy": "Acc", "macro_precision": "Prec",
                 "macro_recall": "Rec", "macro_f1": "FOne"}

    for ds in DATASETS:
        s = short[ds]
        run = one(runs, stage="main", dataset=ds, model="LMT-SDNN")
        if run:
            for met, ms in met_short.items():
                cmd(f"our{s}{ms}", run["metrics"][met])
                cmd(f"paper{s}{ms}", PAPER_TABLE_VI[ds]["LMT-SDNN"][METRICS.index(met)])
                cmd(f"delta{s}{ms}",
                    run["metrics"][met] - PAPER_TABLE_VI[ds]["LMT-SDNN"][METRICS.index(met)],
                    "{:+.2f}")
        for i, tname in enumerate(["Teacher1-ResidualInception", "Teacher2-1DCNN-BiLSTM"]):
            t = one(runs, stage="teachers", dataset=ds, model=tname)
            if t:
                cmd(f"teach{'One' if i==0 else 'Two'}{s}Acc", t["metrics"]["accuracy"])
                cmd(f"teach{'One' if i==0 else 'Two'}{s}FOne", t["metrics"]["macro_f1"])
        nod = one(runs, stage="nodistill", dataset=ds, model="LMT-SDNN")
        if nod and run:
            cmd(f"nokd{s}FOne", nod["metrics"]["macro_f1"])
            cmd(f"kdgain{s}", run["metrics"]["macro_f1"] - nod["metrics"]["macro_f1"], "{:+.2f}")
        for tag, nm in (("teacher1-only", "TOne"), ("teacher2-only", "TTwo"), ("both", "TBoth")):
            t = one(runs, stage="tablation", dataset=ds, tag=tag)
            if t:
                cmd(f"abl{nm}{s}FOne", t["metrics"]["macro_f1"])
        for tag, nm in (("constant", "Const"), ("inverse_freq", "Inv"),
                        ("effective_number", "Eff")):
            t = one(runs, stage="csalpha", dataset=ds, tag=tag)
            if t:
                cmd(f"alpha{nm}{s}FOne", t["metrics"]["macro_f1"])
                cmd(f"alpha{nm}{s}Rec", t["metrics"]["macro_recall"])
        lk = one(runs, stage="leakage", dataset=ds, tag="host IPs as features")
        if lk:
            cmd(f"leak{s}Acc", lk["metrics"]["accuracy"])
            cmd(f"leak{s}FOne", lk["metrics"]["macro_f1"])
        q = one(runs, stage="quota", dataset=ds, tag="proportional quotas")
        qn = one(runs, stage="quota", dataset=ds, tag="proportional, no KD")
        if q:
            cmd(f"quota{s}Acc", q["metrics"]["accuracy"])
            cmd(f"quota{s}FOne", q["metrics"]["macro_f1"])
            cmd(f"quota{s}Rec", q["metrics"]["macro_recall"])
        if q and qn:
            cmd(f"quota{s}NoKdFOne", qn["metrics"]["macro_f1"])
            cmd(f"quota{s}KdGain",
                q["metrics"]["macro_f1"] - qn["metrics"]["macro_f1"], "{:+.2f}")
        bs = one(runs, stage="blocksplit", dataset=ds, tag="contamination-free split")
        if bs:
            cmd(f"block{s}Acc", bs["metrics"]["accuracy"])
            cmd(f"block{s}FOne", bs["metrics"]["macro_f1"])
            if run:
                cmd(f"blockDrop{s}",
                    run["metrics"]["macro_f1"] - bs["metrics"]["macro_f1"], "{:+.2f}")
        pp = one(runs, stage="proposal", dataset=ds, tag="LMT-SDNN++")
        if pp and run:
            cmd(f"prop{s}FOne", pp["metrics"]["macro_f1"])
            cmd(f"prop{s}Rec", pp["metrics"]["macro_recall"])
            cmd(f"propGain{s}", pp["metrics"]["macro_f1"] - run["metrics"]["macro_f1"], "{:+.2f}")
        rows = pick(runs, stage="seeds", dataset=ds, model="LMT-SDNN")
        if len(rows) >= 2:
            v = np.array([r["metrics"]["macro_f1"] for r in rows])
            cmd(f"seed{s}Mean", v.mean())
            cmd(f"seed{s}Std", v.std(ddof=1))
        best_t, best_v = None, -1
        for r in pick(runs, stage="temperature", dataset=ds):
            if r["metrics"]["macro_f1"] > best_v:
                best_v, best_t = r["metrics"]["macro_f1"], r["config"]["temperature"]
        if best_t is not None:
            cmd(f"bestT{s}", best_t, "{:g}")
            cmd(f"bestT{s}FOne", best_v)
            allv = [r["metrics"]["macro_f1"] for r in pick(runs, stage="temperature", dataset=ds)]
            cmd(f"spanT{s}", max(allv) - min(allv))
        best_c, best_cv = None, -1
        for r in pick(runs, stage="weights", dataset=ds):
            if r["metrics"]["macro_f1"] > best_cv:
                best_cv, best_c = r["metrics"]["macro_f1"], r["config"]["tag"]
        if best_c:
            cmd(f"bestComb{s}", best_c.replace("Comb-", "Comb-"), "{}")
            cmd(f"bestComb{s}FOne", best_cv)

    cx = RESULTS / "complexity.json"
    if cx.exists():
        d = json.loads(cx.read_text(encoding="utf-8"))["models"]
        for ds in DATASETS:
            s = short[ds]
            r = d.get(ds, {})
            if "LMT-SDNN" in r and "Teacher(Total)" in r:
                cmd(f"params{s}", f"{r['LMT-SDNN']['params']:,}", "{}")
                cmd(f"sizeKB{s}", r["LMT-SDNN"]["size_kb"])
                cmd(f"teachParams{s}", f"{r['Teacher(Total)']['params']:,}", "{}")
                cmd(f"dropParams{s}", r["LMT-SDNN"]["params_drop_pct"])
                cmd(f"latStudent{s}", r["LMT-SDNN"]["latency_ms_single"], "{:.3f}")
                cmd(f"latTeacher{s}", r["Teacher(Total)"].get(
                    "latency_ms_single", 0.0), "{:.3f}")
                cmd(f"speedup{s}",
                    r["Teacher(Total)"].get("latency_ms_single", 0.0)
                    / max(r["LMT-SDNN"]["latency_ms_single"], 1e-9), "{:.1f}")
    # Any macro the report quotes but this run could not compute (because the
    # corresponding stage has not been executed) gets a visible placeholder, so
    # the document still builds and the gap is obvious in the PDF.
    import re
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", "\n".join(out)))
    prefixes = ("our", "paper", "delta", "teach", "nokd", "kdgain", "abl",
                "alpha", "leak", "block", "prop", "seed", "bestT", "spanT",
                "bestComb", "params", "sizeKB", "dropParams", "lat", "speedup")
    referenced: set[str] = set()
    for src in list(TABLES.parent.glob("sections/*.tex")) + \
            list(TABLES.parent.glob("*.tex")):
        try:
            body = src.read_text(encoding="utf-8")
        except OSError:
            continue
        for name in re.findall(r"\\([A-Za-z]+)", body):
            # every generated macro names its dataset, which keeps this from
            # colliding with maths commands such as \alpha or \propto
            if name.startswith(prefixes) and ("Ton" in name or "Bot" in name):
                referenced.add(name)
    missing = sorted(referenced - defined)
    if missing:
        out.append(r"% placeholders for values this run did not produce")
        for name in missing:
            out.append(rf"\providecommand{{\{name}}}{{\textbf{{?}}}}")
        print(f"     (numbers.tex: {len(missing)} placeholder(s): "
              f"{', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''})")
    _w("numbers.tex", "\n".join(out) + "\n")


def tex_latency():
    """
    The three 'lightweightness' yardsticks side by side.  Generated rather than
    hand-written, because the latency column is re-measured on every run and a
    frozen copy in the prose would drift away from results/complexity.json.
    """
    f = RESULTS / "complexity.json"
    if not f.exists():
        return
    rows = json.loads(f.read_text(encoding="utf-8"))["models"].get("ToN_IoT", {})
    order = ["Teacher(Total)", "TCN", "LMT-SDNN", "1DCNN-BiLSTM", "RNN",
             "1DCNN", "SNN"]
    base = rows.get("LMT-SDNN", {}).get("latency_ms_single")
    L = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r"مدل & پارامتر & \lr{FLOPs} (جدول \lr{V}) & تأخیر (\lr{ms}) & "
         r"نسبت به دانش‌آموز\\", r"\midrule"]
    for m in order:
        if m not in rows:
            continue
        r = rows[m]
        lat = r.get("latency_ms_single", float("nan"))
        nm = r"\textbf{\lr{LMT-SDNN}}" if m == "LMT-SDNN" else rf"\lr{{{m}}}"
        if m == "LMT-SDNN":
            rel = r"\textbf{مبنای مقایسه}"
        elif base and lat > base:
            rel = rf"\num{{{lat/base:.2f}$\times$}} کندتر"
        elif base:
            rel = rf"\up{{\num{{{base/lat:.2f}$\times$}} تندتر}}"
        else:
            rel = "---"
        cells = (rf"\num{{{r['params']:,}}} & \num{{{r['paper_flops']:,}}} & "
                 rf"\num{{{lat:.3f}}} & {rel}")
        if m == "LMT-SDNN":
            cells = (rf"\textbf{{\num{{{r['params']:,}}}}} & "
                     rf"\textbf{{\num{{{r['paper_flops']:,}}}}} & "
                     rf"\textbf{{\num{{{lat:.3f}}}}} & {rel}")
        L.append(rf"{nm} & {cells}\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    _w("table_latency.tex", "\n".join(L))

    # numbers the surrounding prose quotes
    out = [r"% auto-generated by figures.py -- do not edit"]
    t, s = rows.get("Teacher(Total)"), rows.get("LMT-SDNN")
    if t and s:
        out.append(rf"\newcommand{{\latSpeedup}}{{\lr{{{t['latency_ms_single']/s['latency_ms_single']:.2f}}}}}")
        out.append(rf"\newcommand{{\paramRatio}}{{\lr{{{t['params']/s['params']:.0f}}}}}")
        out.append(rf"\newcommand{{\flopRatio}}{{\lr{{{t['paper_flops']/s['paper_flops']:.0f}}}}}")
    c1 = rows.get("1DCNN")
    if c1 and s:
        out.append(rf"\newcommand{{\cnnFlopRatio}}{{\lr{{{c1['paper_flops']/s['paper_flops']:.1f}}}}}")
        out.append(rf"\newcommand{{\cnnSpeedup}}{{\lr{{{s['latency_ms_single']/c1['latency_ms_single']:.2f}}}}}")
    _w("numbers_latency.tex", "\n".join(out) + "\n")


def tex_stability(runs):
    """Mean +- std of macro F1 over independent seeds, ranked."""
    L = [r"\begin{tabular}{llrrr}", r"\toprule",
         r"\lr{Dataset} & \lr{Model} & \lr{runs} & \lr{macro F1 (mean)} & "
         r"\lr{std}\\", r"\midrule"]
    for ds in DATASETS:
        rows = []
        for m in ["1DCNN", "SNN", "RNN", "TCN", "1DCNN-BiLSTM", "LMT-SDNN"]:
            rs = pick(runs, stage="seeds", dataset=ds, model=m)
            if not rs:
                continue
            v = np.array([r["metrics"]["macro_f1"] for r in rs])
            rows.append((m, len(v), v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0))
        rows.sort(key=lambda r: -r[2])
        first = True
        for m, n, mu, sd in rows:
            nm = r"\textbf{\lr{LMT-SDNN}}" if m == "LMT-SDNN" else rf"\lr{{{m}}}"
            head = rf"\lr{{{ds.replace('_', chr(92) + '_')}}}" if first else ""
            first = False
            L.append(rf"{head} & {nm} & \num{{{n}}} & \num{{{mu:.2f}}} & "
                     rf"\lr{{$\pm$~{sd:.2f}}}\\")
        L.append(r"\midrule")
    if len(L) > 5:
        L[-1] = r"\bottomrule"
    else:
        L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    _w("table_stability.tex", "\n".join(L))


def tex_summary(runs):
    """One compact 'did it reproduce' table for the abstract of the report."""
    L = [r"\begin{tabular}{llrrr}", r"\toprule",
         r"\lr{Dataset} & \lr{Metric} & \lr{paper} & \lr{reproduced} & "
         r"\lr{$\Delta$}\\", r"\midrule"]
    for ds in DATASETS:
        run = one(runs, stage="main", dataset=ds, model="LMT-SDNN")
        if run is None:
            continue
        first = True
        for i, met in enumerate(METRICS):
            p = PAPER_TABLE_VI[ds]["LMT-SDNN"][i]
            o = run["metrics"][met]
            d = o - p
            col = r"\up" if d >= -0.5 else r"\down"
            head = rf"\lr{{{_esc(ds)}}}" if first else ""
            first = False
            L.append(rf"{head} & \lr{{{MET_LABEL[met]}}} & \num{{{p:.2f}}} & "
                     rf"\num{{{o:.2f}}} & {col}{{\num{{{d:+.2f}}}}}\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    _w("table_summary.tex", "\n".join(L))


# --------------------------------------------------------------------------- #
def main():
    runs = load_runs()
    print(f"loaded {len(runs)} runs")
    for fn in (fig_main, fig_distill_gain, fig_teacher_ablation, fig_weights,
               fig_loss_ablation, fig_temperature, fig_alpha, fig_stability,
               fig_confusion, fig_validity, fig_proposal):
        try:
            fn(runs)
            print(f"  ok  {fn.__name__}")
        except Exception as e:
            print(f"  --  {fn.__name__}: {type(e).__name__}: {e}")
    for fn in (fig_compression,):
        try:
            fn(); print(f"  ok  {fn.__name__}")
        except Exception as e:
            print(f"  --  {fn.__name__}: {e}")
    for fn, a in ((tex_table6, runs), (tex_summary, runs), (tex_numbers, runs),
                  (tex_stability, runs)):
        try:
            fn(a); print(f"  ok  {fn.__name__}")
        except Exception as e:
            print(f"  --  {fn.__name__}: {e}")
    for fn in (tex_table7, tex_audit, tex_dataset, tex_latency):
        try:
            fn(); print(f"  ok  {fn.__name__}")
        except Exception as e:
            print(f"  --  {fn.__name__}: {e}")


if __name__ == "__main__":
    main()
