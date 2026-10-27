"""
experiments/paper_figures.py
Figures for the paper, from the published runs. Writes PDFs to --out.

  fig_position.pdf     share of hallucinated objects by caption-position decile
  fig_paired.pdf       paired AUROC differences within position bins (cross-fitted)
  fig_layers.pdf       calibration AUROC of max-patch cosine for every layer pair
  fig_label_eff.pdf    AUROC within position bins against the number of labelled objects
                       (needs results/paper/label_efficiency_*.json)

Usage:
    python experiments/paper_figures.py --out <paper>/figures
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from caption_bench.evaluate import load_run

RUNS = [  # (label, run folder, label-efficiency key)
    ("LLaVA-1.5-7B", "results/caption_runs/llava15_7b_4bit_n1000_s42", "llava15_7b"),
    ("LLaVA-1.5-13B", "results/caption_runs/llava15_13b_n1000_s42", "llava15_13b"),
    ("Qwen2.5-VL-7B", "results/caption_runs/qwen25vl_7b_n1000_s42", "qwen25vl_7b"),
]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]      # categorical slots 1-3, validated for CVD
MARKERS = ["o", "s", "^"]
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e4e4e1"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})


def recessive_grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_position(out):
    """Share of hallucinated first mentions per decile of the object's token index."""
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for (name, run, _), color, marker in zip(RUNS, COLORS, MARKERS):
        t = load_run(run, hidden=False)
        edges = np.quantile(t.positions, np.linspace(0, 1, 11)[1:-1])
        bins = np.digitize(t.positions, edges)
        share = [100 * t.labels[bins == k].mean() for k in range(10)]
        x = np.arange(1, 11)
        ax.plot(x, share, color=color, marker=marker, markersize=4, linewidth=1.5,
                markeredgecolor="white", markeredgewidth=0.6)
        ax.annotate(name, (10, share[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=7, color=INK)
    ax.set_xlim(0.6, 10.4)
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0, None)
    ax.set_xlabel("Position of the object in the caption (decile of token index)")
    ax.set_ylabel("Hallucinated objects (%)")
    recessive_grid(ax)
    fig.subplots_adjust(right=0.74)
    fig.savefig(os.path.join(out, "fig_position.pdf"), bbox_inches="tight")
    plt.close(fig)


PAIRS = [  # (detector, baseline, row label)
    ("Best latent stat (at)", "NLL", "Latent − NLL"),
    ("Best latent stat (at)", "Internal Confidence (calib layer)", "Latent − Internal Confidence"),
    ("Fusion: NLL + best latent", "NLL", "(NLL + latent) − NLL"),
    ("Best latent stat (at)", "GLSim (settings chosen on calibration)", "Latent − GLSim"),
    ("Best latent stat (at)", "Linear probe on hidden states", "Latent − linear probe"),
]


def fig_paired(out):
    """Paired bootstrap differences within position bins, one row per comparison."""
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    offsets = [-0.22, 0.0, 0.22]
    for (name, run, _), color, marker, dy in zip(RUNS, COLORS, MARKERS, offsets):
        cv = json.load(open(os.path.join(run, "eval", "eval.json")))["crossfit"]
        comps = {(c["detector"], c["baseline"]): c["within_position"] for c in cv["comparisons"]}
        for row, (a, b, _) in enumerate(PAIRS):
            w = comps[(a, b)]
            y = row + dy
            ax.plot([100 * w["ci"][0], 100 * w["ci"][1]], [y, y], color=color, linewidth=1.2)
            ax.plot(100 * w["diff"], y, marker=marker, color=color, markersize=4.5,
                    markeredgecolor="white", markeredgewidth=0.6, linestyle="none",
                    label=name if row == 0 else None)
    ax.axvline(0, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_yticks(range(len(PAIRS)))
    ax.set_yticklabels([p[2] for p in PAIRS])
    ax.invert_yaxis()
    ax.set_xlabel("Difference in AUROC within position bins (points, 95% CI)")
    recessive_grid(ax, "x")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3,
              handletextpad=0.3, columnspacing=0.9)
    fig.savefig(os.path.join(out, "fig_paired.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_layers(out):
    """Calibration AUROC of max-patch cosine for each (object-token layer, image-token layer)."""
    maps = [np.array(json.load(open(os.path.join(run, "eval", "eval.json")))["calib_layer_map_max_cosine"])
            for _, run, _ in RUNS]
    vmin, vmax = 0.60, max(m.max() for m in maps)   # below 0.60 is clipped to the lightest step
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.3), constrained_layout=True)
    for ax, m, (name, _, _) in zip(axes, maps, RUNS):
        im = ax.imshow(m, origin="lower", cmap=BLUES, vmin=vmin, vmax=vmax, aspect="equal")
        lt, li = np.unravel_index(np.argmax(m), m.shape)
        ax.plot(li, lt, marker="o", markersize=5, markerfacecolor="none",
                markeredgecolor="#eb6834", markeredgewidth=1.2)
        ax.set_title(name, color=INK)
        ax.set_xlabel("Image-token layer")
        n = m.shape[0] - 1
        ticks = list(range(0, n + 1, 8 if n > 32 else 7 if n == 28 else 8))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
    axes[0].set_ylabel("Object-token layer")
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01)
    cb.set_label("Calibration AUROC")
    cb.outline.set_linewidth(0.4)
    fig.savefig(os.path.join(out, "fig_layers.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_label_eff(out):
    """Latent statistic and linear probe trained on n labelled objects (cross-fitted)."""
    files = [f"results/paper/label_efficiency_{key}.json" for _, _, key in RUNS]
    if not all(os.path.exists(f) for f in files):
        print("skipping fig_label_eff: run experiments/label_efficiency.py first")
        return
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.1), sharey=True, constrained_layout=True)
    for ax, f, (name, _, _), color in zip(axes, files, RUNS, COLORS):
        res = json.load(open(f))["sizes"]
        n = np.array([v["n_train"] for v in res.values()])
        for det, style, label in (("latent", "-", "Latent statistic"), ("probe", "--", "Linear probe")):
            mean = np.array([100 * v[det]["auroc_pos"][0] for v in res.values()])
            sd = np.array([100 * v[det]["auroc_pos"][1] for v in res.values()])
            c = color if det == "latent" else INK
            ax.plot(n, mean, style, color=c, marker="o", markersize=3, linewidth=1.4, label=label)
            ax.fill_between(n, mean - sd, mean + sd, color=c, alpha=0.12, linewidth=0)
        ax.set_xscale("log")
        ax.set_title(name, color=INK)
        ax.set_xlabel("Labelled objects used for fitting")
        recessive_grid(ax)
        ax.legend(frameon=False, loc="lower right")
    axes[0].set_ylabel("AUROC within position bins")
    fig.savefig(os.path.join(out, "fig_label_eff.pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--only", nargs="*", help="figure names, e.g. position layers")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    figs = {"position": fig_position, "paired": fig_paired, "layers": fig_layers,
            "label_eff": fig_label_eff}
    for name, fn in figs.items():
        if not args.only or name in args.only:
            fn(args.out)
            print("wrote", name)


if __name__ == "__main__":
    main()
