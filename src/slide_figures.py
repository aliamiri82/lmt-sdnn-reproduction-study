"""
Slide-sized figures.

The report figures are drawn for a full text column and carry two datasets,
axis labels and a legend; projected at slide size their type becomes
unreadable.  These are redrawn for a 16:9 frame: one message per figure, type
large enough to read from the back of a room, and no element that does not
carry information.

Output: figures/slides/*.png
"""
from __future__ import annotations

import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import FIGURES, RESULTS

OUT = FIGURES / "slides"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 15, "axes.titlesize": 17, "axes.titleweight": "bold",
    "axes.labelsize": 15, "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 14, "legend.frameon": False,
    "axes.grid": True, "grid.alpha": .25,
    "axes.spines.top": False, "axes.spines.right": False,
})


# Labels stay in English.  Matplotlib does no Arabic shaping and no bidi
# reordering, so Persian handed to it comes out with the letters isolated and
# reversed; and every other figure in this project is already labelled in
# English, so the deck stays internally consistent.  The Persian commentary
# lives in the slide text around each figure.

OURS = "#1F3864"
OTHER = "#9DC3E6"
BAD = "#B22222"
GOOD = "#1B7F3B"
WARN = "#B8860B"


def load_runs():
    f = RESULTS / "runs.jsonl"
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]


def pick(runs, **kw):
    return [r for r in runs if all(r["config"].get(k) == v for k, v in kw.items())]


def one(runs, **kw):
    r = pick(runs, **kw)
    return r[-1] if r else None


ORDER = ["TCN", "1DCNN", "1DCNN-BiLSTM", "SNN", "LMT-SDNN", "RNN"]


# --------------------------------------------------------------------------- #
def fig_stability(runs):
    """The headline negative result: seed noise swamps the architecture gap."""
    data, labels = [], []
    for m in ORDER:
        rows = pick(runs, stage="seeds", dataset="ToN_IoT", model=m)
        if len(rows) < 2:
            continue
        data.append([r["metrics"]["macro_f1"] for r in rows])
        labels.append(m)

    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=.55,
                    medianprops=dict(color="black", lw=2))
    for i, b in enumerate(bp["boxes"]):
        b.set_facecolor(OURS if labels[i] == "LMT-SDNN" else OTHER)
    for i, vals in enumerate(data):
        ax.scatter([i + 1] * len(vals), vals, color="black", s=18, zorder=3, alpha=.7)

    ax.set_ylabel("macro F1 (%)")
    ax.set_title("ToN_IoT — three independent seeds per architecture")
    lo = min(min(d) for d in data) - .15
    hi = max(max(d) for d in data) + .25
    ax.set_ylim(lo, hi)
    # band showing one typical standard deviation
    idx = labels.index("LMT-SDNN")
    mu = np.mean(data[idx])
    sd = np.std(data[idx], ddof=1)
    ax.axhspan(mu - sd, mu + sd, color=OURS, alpha=.10, zorder=0)
    # keep the caption clear of the boxes: park it in the empty upper-left
    ax.annotate("shaded band = ±1 std of LMT-SDNN\nit covers four of the six models",
                xy=(0.02, 0.97), xycoords="axes fraction", fontsize=12.5,
                color=OURS, ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec=OURS, alpha=.85, lw=1))
    fig.tight_layout()
    fig.savefig(OUT / "sl_stability.png")
    plt.close(fig)


def fig_kd_gain(runs):
    """Distillation adds nothing here."""
    labels, kd, nokd = [], [], []
    for m in ORDER:
        a = one(runs, stage="main", dataset="ToN_IoT", model=m)
        b = one(runs, stage="nodistill", dataset="ToN_IoT", model=m)
        if not a or not b:
            continue
        labels.append(m)
        kd.append(a["metrics"]["macro_f1"])
        nokd.append(b["metrics"]["macro_f1"])

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    ax.bar(x - .21, nokd, .42, label="without distillation", color=OTHER)
    ax.bar(x + .21, kd, .42, label="with multi-teacher KD", color=OURS)
    for i, (a, b) in enumerate(zip(nokd, kd)):
        d = b - a
        ax.annotate(f"{d:+.2f}", (i, max(a, b) + .12), ha="center",
                    fontsize=13, fontweight="bold",
                    color=GOOD if d > 0 else BAD)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylim(min(nokd + kd) - .6, max(nokd + kd) + .55)
    ax.set_ylabel("macro F1 (%)")
    ax.set_title("ToN_IoT — every difference is within seed noise")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "sl_kd_gain.png")
    plt.close(fig)


def fig_flops_latency():
    """FLOPs and measured latency disagree — the low-power headline."""
    d = json.loads((RESULTS / "complexity.json").read_text(encoding="utf-8"))
    rows = d["models"]["ToN_IoT"]
    names = ["Teacher(Total)", "TCN", "LMT-SDNN", "1DCNN", "SNN"]
    names = [n for n in names if n in rows]
    flops = np.array([rows[n]["paper_flops"] for n in names], dtype=float)
    lat = np.array([rows[n]["latency_ms_single"] for n in names], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    colors = [OURS if n == "LMT-SDNN" else (BAD if n == "Teacher(Total)" else OTHER)
              for n in names]

    axes[0].barh(names[::-1], flops[::-1], color=colors[::-1])
    axes[0].set_xscale("log")
    axes[0].set_xlabel("FLOPs (log)")
    axes[0].set_title("what the paper measures", fontsize=15)
    axes[0].invert_yaxis()

    axes[1].barh(names[::-1], lat[::-1], color=colors[::-1])
    axes[1].set_xlabel("measured latency (ms / flow)")
    axes[1].set_title("what actually matters on an edge CPU", fontsize=15)
    axes[1].invert_yaxis()
    for i, v in enumerate(lat[::-1]):
        axes[1].annotate(f"{v:.2f}", (v, i), xytext=(4, 0),
                         textcoords="offset points", va="center", fontsize=12)

    fig.suptitle("LMT-SDNN has 54x fewer FLOPs than 1DCNN — and is 5.6x slower",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "sl_flops_latency.png")
    plt.close(fig)


def fig_validity(runs):
    """How much of the headline number survives a stricter protocol."""
    settings = [("main", None, "paper protocol\n(random split)"),
                ("leakage", "host IPs as features", "+ host IP\nas a feature"),
                ("blocksplit", "contamination-free split", "contamination-free\nsplit")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for ax, ds in zip(axes, ["ToN_IoT", "BoT_IoT"]):
        labs, f1 = [], []
        for stage, tag, lab in settings:
            r = (one(runs, stage=stage, dataset=ds, model="LMT-SDNN") if tag is None
                 else one(runs, stage=stage, dataset=ds, tag=tag))
            if not r:
                continue
            labs.append(lab)
            f1.append(r["metrics"]["macro_f1"])
        if not labs:
            continue
        cols = [OURS, WARN, BAD][:len(labs)]
        ax.bar(range(len(labs)), f1, .55, color=cols)
        for i, v in enumerate(f1):
            ax.annotate(f"{v:.1f}", (i, v + .6), ha="center",
                        fontsize=14, fontweight="bold")
        if len(f1) >= 3:
            ax.annotate(f"{f1[-1]-f1[0]:+.1f}", (len(f1) - 1, f1[-1] - 6),
                        ha="center", fontsize=15, fontweight="bold", color=BAD)
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels(labs, fontsize=12)
        ax.set_ylim(88, 103)
        ax.set_ylabel("macro F1 (%)")
        ax.set_title(ds)
    fig.tight_layout()
    fig.savefig(OUT / "sl_validity.png")
    plt.close(fig)


def fig_proposal(runs):
    # short labels: two-line captions collide at this panel width
    variants = [("main", None, "baseline"),
                ("proposal", "+adaptive teachers", "+teachers"),
                ("proposal", "+class-aware alpha", r"+$\alpha_i$"),
                ("proposal", "LMT-SDNN++", "both")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for ax, ds in zip(axes, ["ToN_IoT", "BoT_IoT"]):
        labs, rec = [], []
        for stage, tag, lab in variants:
            r = (one(runs, stage=stage, dataset=ds, model="LMT-SDNN") if tag is None
                 else one(runs, stage=stage, dataset=ds, tag=tag))
            if not r:
                continue
            labs.append(lab)
            rec.append(r["metrics"]["macro_recall"])
        if not labs:
            continue
        cols = [OTHER] * len(labs)
        cols[-1] = OURS
        ax.bar(range(len(labs)), rec, .55, color=cols)
        base = rec[0]
        for i, v in enumerate(rec):
            ax.annotate("baseline" if i == 0 else f"{v-base:+.2f}",
                        (i, v + .05), ha="center", fontsize=13,
                        fontweight="bold" if i else "normal")
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels(labs, fontsize=13)
        ax.set_ylim(min(rec) - .35, max(rec) + .3)
        ax.set_ylabel("macro Recall (%)")
        ax.set_title(ds)
    fig.suptitle("both components act only during training - zero inference cost",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "sl_proposal.png")
    plt.close(fig)


def fig_compression():
    """One glance: the student really is two orders of magnitude smaller."""
    d = json.loads((RESULTS / "complexity.json").read_text(encoding="utf-8"))
    rows = d["models"]["ToN_IoT"]
    names = ["Teacher(Total)", "1DCNN-BiLSTM", "RNN", "TCN", "1DCNN", "SNN", "LMT-SDNN"]
    names = [n for n in names if n in rows]
    vals = [rows[n]["params"] for n in names]
    cols = [BAD if n == "Teacher(Total)" else (OURS if n == "LMT-SDNN" else OTHER)
            for n in names]

    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    ax.barh(names[::-1], vals[::-1], color=cols[::-1])
    ax.set_xscale("log")
    ax.set_xlabel("parameters (log scale)")
    ax.invert_yaxis()
    for i, v in enumerate(vals[::-1]):
        ax.annotate(f"{v:,}", (v, i), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=12)
    ax.set_title("173x smaller than the two teachers combined", fontsize=16)
    ax.set_xlim(right=max(vals) * 4)
    fig.tight_layout()
    fig.savefig(OUT / "sl_compression.png")
    plt.close(fig)


def main():
    runs = load_runs()
    for fn, a in ((fig_stability, runs), (fig_kd_gain, runs),
                  (fig_validity, runs), (fig_proposal, runs)):
        try:
            fn(a)
            print("  ok ", fn.__name__)
        except Exception as e:
            print(f"  -- {fn.__name__}: {type(e).__name__}: {e}")
    for fn in (fig_flops_latency, fig_compression):
        try:
            fn()
            print("  ok ", fn.__name__)
        except Exception as e:
            print(f"  -- {fn.__name__}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
