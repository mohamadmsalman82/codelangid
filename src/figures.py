"""
CodeLangID -- all report figures.

Palette follows the validated reference instance (light surface):
  categorical slot 1 = #2a78d6 (blue), slot 2 = #008300 (green)
  sequential = single blue hue, light->dark (confusion heatmaps)
Direct labels accompany every line (secondary encoding: the blue/green pair
sits in the tritan CVD floor band, so identity is never colour-alone).

Usage:  python3 src/figures.py
Output: figures/*.pdf
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

BLUE, GREEN = "#2a78d6", "#008300"
INK, SECOND, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
       "#2a78d6", "#256abf", "#184f95", "#0d366b"]
BLUES = LinearSegmentedColormap.from_list("seq_blue", SEQ)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def box(ax, x, y, w, h, text, fc=SURFACE, ec=AXIS, fs=6.5, tc=INK, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.008,rounding_size=0.02",
                                fc=fc, ec=ec, lw=0.9, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, zorder=3, linespacing=1.35,
            fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=8, lw=0.9, color=MUTED, zorder=1))


# ------------------------------------------------------------ Fig 1: pipeline
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(6.6, 2.05))
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.axis("off"); ax.grid(False)

    box(ax, 0.005, 0.30, 0.135, 0.40,
        "INPUT\nraw snippet\n\n$\\mathtt{def\\ add(a,b):}$\n$\\mathtt{\\ \\ return\\ a+b}$",
        fs=6, tc=SECOND)
    arrow(ax, 0.142, 0.50, 0.163, 0.50)

    box(ax, 0.165, 0.34, 0.115, 0.32,
        "Char encode\n256 chars\npad / truncate\nvocab $V{=}102$", fs=6)
    arrow(ax, 0.282, 0.50, 0.303, 0.50)

    box(ax, 0.305, 0.34, 0.105, 0.32, "Embedding\n$256\\times32$\nlearned", fs=6)
    arrow(ax, 0.412, 0.50, 0.437, 0.50)

    # three parallel conv branches
    for i, (k, y) in enumerate([(3, 0.70), (5, 0.40), (7, 0.10)]):
        box(ax, 0.44, y, 0.115, 0.22, f"Conv1D $k{{=}}{k}$\n128 filters + ReLU",
            fs=5.8, ec=BLUE)
        arrow(ax, 0.437, 0.50, 0.44, y + 0.11)
        arrow(ax, 0.557, y + 0.11, 0.60, 0.50)
    ax.text(0.4975, 0.955, "parallel branches", ha="center", fontsize=5.5,
            color=BLUE, style="italic")

    box(ax, 0.60, 0.34, 0.105, 0.32, "Global\nmax-pool\n+ concat\n(384)", fs=6)
    arrow(ax, 0.707, 0.50, 0.728, 0.50)

    box(ax, 0.73, 0.34, 0.10, 0.32, "Dropout 0.5\nFC $384{\\rightarrow}10$\nsoftmax", fs=6)
    arrow(ax, 0.832, 0.50, 0.853, 0.50)

    box(ax, 0.855, 0.30, 0.14, 0.40,
        "OUTPUT\n$P(\\mathrm{lang}\\mid\\mathrm{snippet})$\n\nPython  0.96\nRuby    0.02",
        fs=6, ec=GREEN, tc=SECOND)

    ax.text(0.5, -0.06, "68,938 trainable parameters  ·  trained from scratch",
            ha="center", fontsize=6, color=MUTED, style="italic")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_pipeline.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote fig_pipeline.pdf")


# ------------------------------------------------------- Fig 2: learning curves
def fig_curves():
    h = json.loads((RES / "cnn_raw.json").read_text())["history"]
    ep = np.arange(1, len(h["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.2))

    for ax, (a, b, name, lo) in zip(axes, [
            ("train_loss", "val_loss", "Cross-entropy loss", None),
            ("train_acc", "val_acc", "Accuracy", None)]):
        ax.plot(ep, h[a], color=BLUE, lw=2, label="train")
        ax.plot(ep, h[b], color=GREEN, lw=2, label="validation")
        ax.set_xlabel("epoch"); ax.set_ylabel(name)
        ax.set_xlim(1, len(ep))
        # direct labels (secondary encoding, not colour alone)
        ax.annotate("train", (ep[-1], h[a][-1]), textcoords="offset points",
                    xytext=(-2, -9 if "loss" in a else 6), fontsize=6.5,
                    color=BLUE, ha="right", fontweight="bold")
        ax.annotate("validation", (ep[-1], h[b][-1]), textcoords="offset points",
                    xytext=(-2, 6 if "loss" in a else -9), fontsize=6.5,
                    color=GREEN, ha="right", fontweight="bold")

    best = int(np.argmax(h["val_acc"]))
    axes[1].axvline(best + 1, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=0)
    axes[1].annotate(f"best epoch {best+1}\nval acc {h['val_acc'][best]:.3f}",
                     (best + 1, axes[1].get_ylim()[0]),
                     textcoords="offset points", xytext=(-5, 5),
                     fontsize=6, color=SECOND, va="bottom", ha="right")
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_curves.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote fig_curves.pdf")


# ---------------------------------------------------- Fig 3: confusion matrices
def fig_confusion():
    d = json.loads((RES / "cnn_raw.json").read_text())
    labels = d["labels"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    for ax, split, title in zip(axes, ["test", "heldout"], [
            f"(a) Rosetta test  ·  acc {d['test']['accuracy']*100:.1f}%",
            f"(b) GitHub held-out  ·  acc {d['heldout']['accuracy']*100:.1f}%"]):
        cm = np.array(d[split]["confusion"], dtype=float)
        cm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cm, cmap=BLUES, vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=5.5)
        ax.set_yticks(range(len(labels)), labels, fontsize=5.5)
        ax.set_title(title, fontsize=7, color=INK, pad=6)
        ax.set_xlabel("predicted", fontsize=6.5)
        ax.grid(False)
        if split == "test":
            ax.set_ylabel("true", fontsize=6.5)
        for i in range(len(labels)):
            for j in range(len(labels)):
                v = cm[i, j]
                if v >= 0.01:
                    ax.text(j, i, f"{v*100:.0f}", ha="center", va="center",
                            fontsize=4.6, color="white" if v > 0.5 else SECOND)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.text(0.5, -0.02, "cell = % of true-class snippets predicted as column "
             "(row-normalised)", ha="center", fontsize=6, color=MUTED)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_confusion.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote fig_confusion.pdf")


if __name__ == "__main__":
    fig_pipeline()
    fig_curves()
    fig_confusion()
