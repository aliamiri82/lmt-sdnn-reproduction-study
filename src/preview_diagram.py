"""
Render the diagram description to PNG so the layout can be checked before the
.vsdx / .drawio files are handed over.  Neither Visio nor draw.io is available
here, so this is the only way to see what was actually built.
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon

import make_diagram as D

OUT = pathlib.Path(__file__).parent.parent / "diagram"


def anchor(a, b):
    """Edge endpoints on the facing sides of two boxes."""
    ax, ay, aw, ah = D.N[a][:4]
    bx, by, bw, bh = D.N[b][:4]
    acx, acy = ax + aw / 2, ay + ah / 2
    bcx, bcy = bx + bw / 2, by + bh / 2
    if abs(bcx - acx) >= abs(bcy - acy):
        p0 = (ax + aw, acy) if bcx > acx else (ax, acy)
        p1 = (bx, bcy) if bcx > acx else (bx + bw, bcy)
    else:
        p0 = (acx, ay + ah) if bcy > acy else (acx, ay)
        p1 = (bcx, by) if bcy > acy else (bcx, by + bh)
    return p0, p1


def main():
    fig, ax = plt.subplots(figsize=(D.CANVAS_W / 100, D.CANVAS_H / 100), dpi=110)
    ax.set_xlim(0, D.CANVAS_W)
    ax.set_ylim(D.CANVAS_H, 0)          # top-left origin, like the spec
    ax.axis("off")

    for nid, (x, y, w, h, label, kind, fill, fs, bold) in D.N.items():
        fc = fill if kind != "text" else "none"
        ec = D.LINE if kind != "text" else "none"
        if kind == "cloud":
            ax.add_patch(Ellipse((x + w / 2, y + h / 2), w, h,
                                 facecolor=fc, edgecolor=ec, lw=1))
        elif kind == "trap":
            ax.add_patch(Polygon([(x + w * .12, y), (x + w * .88, y),
                                  (x + w, y + h), (x, y + h)],
                                 closed=True, facecolor=fc, edgecolor=ec, lw=1))
        elif kind == "cyl":
            ax.add_patch(FancyBboxPatch((x, y + h * .12), w, h * .76,
                                        boxstyle="round,pad=0,rounding_size=8",
                                        facecolor=fc, edgecolor=ec, lw=1))
            ax.add_patch(Ellipse((x + w / 2, y + h * .12), w, h * .24,
                                 facecolor=fc, edgecolor=ec, lw=1))
        elif kind in ("round", "panel", "group"):
            r = 10 if kind != "panel" else 6
            ax.add_patch(FancyBboxPatch((x, y), w, h,
                                        boxstyle=f"round,pad=0,rounding_size={r}",
                                        facecolor=fc, edgecolor=ec, lw=1))
        elif kind != "text":
            ax.add_patch(FancyBboxPatch((x, y), w, h,
                                        boxstyle="square,pad=0",
                                        facecolor=fc, edgecolor=ec, lw=1))
        if label:
            dark = fill in (D.BLUE_D,)
            va = "top" if kind == "group" else "center"
            yy = y + 10 if kind == "group" else y + h / 2
            ax.text(x + w / 2, yy, label, ha="center", va=va,
                    fontsize=fs * 0.82, color="white" if dark else "black",
                    fontweight="bold" if bold else "normal", zorder=5)

    for a, b, label, dashed, pts in D.E:
        p0, p1 = anchor(a, b)
        if pts:
            # polyline: plain segments first, arrowhead only on the final one
            chain = [p0] + list(pts) + [p1]
            for s, e in zip(chain[:-2], chain[1:-1]):
                ax.plot([s[0], e[0]], [s[1], e[1]], color=D.LINE, lw=1, zorder=4)
            p0 = chain[-2]
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>",
                                     mutation_scale=11, lw=1,
                                     color=D.LINE, zorder=4,
                                     linestyle="--" if dashed else "-",
                                     shrinkA=0, shrinkB=2))

    fig.tight_layout(pad=0.2)
    p = OUT / "preview.png"
    fig.savefig(p, dpi=110, bbox_inches="tight", facecolor="white")
    print(f"  {p}")


if __name__ == "__main__":
    main()
