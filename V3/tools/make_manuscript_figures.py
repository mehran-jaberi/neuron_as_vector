#!/usr/bin/env python
"""Build the manuscript figures from recorded artifacts.

Every panel reads numbers that already exist on disk (``V3/results``), or a
small table transcribed from ``V3/RESULTS.md`` with the run tag recorded next to
it.  Nothing is re-computed on the GPU and nothing is fitted, so the figures can
be rebuilt from a clean checkout after the runs are done:

    python V3/tools/make_manuscript_figures.py            # everything
    python V3/tools/make_manuscript_figures.py --only 1 3 # a subset

Outputs land in ``figures/manuscript/`` as vector PDFs plus 300 dpi PNGs, two
column width (180 mm composites, 90 mm single panels).

Sources, panel by panel
-----------------------
fig1  schematic; ``V3/results/analysis/capture_final_test.npz``
fig2  width sweep + controls -- transcribed from ``V3/RESULTS.md`` §2-3
fig3  ``V3/results/v3_n100_spec_ep30/{history.csv, test_predictions.npz}``
fig4  probe, -- ``analysis/probe_logistic_final.json`` + N=64 probe result;
      spectrum -- ``capture_final_test.npz``; freeze arms -- ``ablation`` json
fig5  ``analysis/ablation_fingerprints_final.json``,
      ``analysis/ablation_knockout.json``
fig6  five operations, dropout sweep, interaction -- ``runs.csv`` rows
S1    per-class accuracy of six key runs (``test_predictions.npz`` per dir)
S2    v3.1 diagonal emission -- two rows with their tags
S3    population spike-count code -- ``capture_final_test.npz`` counts

Transcribed tables carry the registry tag in a comment so a reader can always
find the source row; if a number and the registry disagree, the registry wins.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "V3" / "results"
ANALYSIS = RES / "analysis"
OUT = ROOT / "figures" / "manuscript"

FINAL = "v3_n100_spec_ep30"
FINAL_CORRECT = 1994
BEST_CORRECT, BEST_EPOCH = 2009, 28
N_TEST = 2264

# ---------------------------------------------------------------- numbers ----
WIDTH_SWEEP = [  # tag, D, params, test correct
    ("v3_d4_ep5", 4, 9_056, 1295),
    ("v3_d16_ep5", 16, 20_480, 1443),
    ("v3_d64_ep5", 64, 74_816, 1553),
    ("v3_d256_ep5", 256, 332_096, 1757),
    ("v3_vector_d1000", 1000, 2_025_440, 1724),
]
CONTROLS = [  # label, params, test correct, epochs, tag
    ("scalar\n$D{=}1$, $N{=}64$", 6_335, 1119, 20, "v3_vector_scalar_d1"),
    ("scalar\n$N{=}1411$", 2_025_488, 750, 5, "v3_n1411_ep5_scalar_d1"),
    ("broadcast\n$E{=}1000$", 812_300, 1481, 5, "v3_ctrl_broadcast_e1000_ep5"),
    ("vector\n$D{=}1000$", 2_140_100, 1803, 5, "v3_n100_baseline_ep5"),
]
AUG_OPS = [  # label, N=64 correct, N=100 correct
    ("jitter", 1749, 1814),
    ("dropout", 1781, 1841),
    ("insert", 1726, 1811),
    ("shift", 1751, 1736),
    ("channel", 1768, 1782),
]
AUG_BASE = {"n64": 1724, "n100": 1803}
DROPOUT_SWEEP = [(0.05, 1763), (0.10, 1781), (0.15, 1749)]
INTERACTION = [("jitter", 1749), ("dropout", 1781), ("both", 1731),
               ("none", 1724)]
FREEZE = {  # arm, N=64 correct, N=100 correct
    "trained\ndynamics": (1857, 1941),
    "untrained\nreservoir": (830, 1025),
    "no recurrence\n(no $h$, no $W_{rec}$)": (1341, 1538),
}
PROBE = {  # feature set, N=64 correct, N=100 correct
    "spike counts": (1868, 1988),
    "mean state": (1830, 1917),
    "final state": (674, 625),
}
V31 = [("full", 2_025_440, 1724), ("diagonal", 1_026_440, 1748)]
PERCLASS_RUNS = [
    ("v3_vector_d1000", "$D{=}1000$, $N{=}64$, 20 ep"),
    ("v3_aug_dropout_ep20", "$N{=}64$ + dropout, 20 ep"),
    ("v3_n100_aug_dropout_ep20", "$N{=}100$ + dropout, 20 ep"),
    ("v3_n100_spec_ep30", "final model, 30 ep"),
    ("v3_n100_baseline_ep5", "$N{=}100$ baseline, 5 ep"),
    ("v3_speaker_holdout_ep5", "speaker-holdout, 5 ep"),
]
PNG_DPI = 300


# ------------------------------------------------------------------ style ----
def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.titlesize": 7.4,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.2,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })


C_VEC = "#2b6cb0"
C_SCALAR = "#c05621"
C_GREY = "#718096"
C_ACC = "#2f855a"
C_BAD = "#9b2c2c"
C_PUR = "#805ad5"


def save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    pdf = OUT / f"{name}.pdf"
    fig.savefig(pdf)
    fig.savefig(OUT / f"{name}.png", dpi=PNG_DPI)
    plt.close(fig)
    print(f"[figures] {pdf.relative_to(ROOT)}")
    return pdf


def acc(correct: int, total: int = N_TEST) -> float:
    return 100.0 * correct / total


def titled(ax, letter: str, title: str) -> None:
    ax.set_title(f"{letter}   {title}", loc="left", fontsize=7.4)


def load_json(rel: str) -> dict:
    return json.loads((ANALYSIS / rel).read_text())


def load_json_run(tag: str) -> dict:
    """Variant block of `V3/results/<tag>/summary.json`."""
    blob = json.loads((RES / tag / "summary.json").read_text())
    return blob["variants"][tag]


def load_run(tag: str) -> dict:
    """Training cost + accuracy for one run, old (metrics json) or new (summary)."""
    hist = load_history(tag)
    epoch_s = float(np.mean([float(r["time_s"]) for r in hist[-5:]]))
    summary = RES / tag / "summary.json"
    if summary.exists():
        v = load_json_run(tag)
        return {"wall_seconds": float(v["train_seconds"]),
                "test_accuracy": float(v["test_accuracy"]),
                "vram_gib": float(v["dtypes"]["peak_memory_reserved_mb"]) / 1024.0,
                "epoch_seconds": epoch_s}
    train = json.loads((RES / tag / "train_metrics.json").read_text())
    test = json.loads((RES / tag / "test_metrics.json").read_text())
    return {"wall_seconds": float(train["wall_seconds"]),
            "test_accuracy": float(test["test_accuracy"]),
            "vram_gib": float(train["dtypes"]["peak_memory_reserved_mb"]) / 1024.0,
            "epoch_seconds": epoch_s}


def load_history(tag: str) -> list:
    import csv

    with (RES / tag / "history.csv").open() as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------- fig 1 ------
def _neuron_box(ax, x, y, w, h, n_strips=7, color=C_VEC):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004",
                                fc="white", ec=color, lw=0.8, zorder=3))
    for k in range(n_strips):
        yy = y + h * (0.12 + 0.72 * k / max(n_strips - 1, 1))
        ax.plot([x + 0.006, x + w - 0.006], [yy, yy], color=color, lw=1.6,
                solid_capstyle="butt", zorder=4)


def fig1() -> None:
    import torch

    fig = plt.figure(figsize=(7.1, 3.15))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.20, 1.0], wspace=0.14)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("a   architecture", loc="left", fontsize=7.4)

    ax.add_patch(FancyBboxPatch((0.02, 0.30), 0.05, 0.40,
                                boxstyle="round,pad=0.004", fc="#edf2f7",
                                ec=C_GREY, lw=0.8))
    for k in range(9):
        yy = 0.335 + 0.33 * k / 8
        ax.plot([0.026, 0.064], [yy, yy], color=C_GREY, lw=1.0)
    ax.text(0.045, 0.255, "700\ninput\nchannels", ha="center", va="top",
            fontsize=6.0, color=C_GREY)

    ys = [0.17, 0.42, 0.67]
    for i, y in enumerate(ys):
        _neuron_box(ax, 0.20, y, 0.085, 0.15)
        ax.text(0.2425, y + 0.155, f"$z_{{{i + 1}}}$", ha="center", fontsize=6.4)
        ax.add_patch(FancyArrowPatch((0.072, 0.50), (0.196, y + 0.075),
                                     arrowstyle="-|>", mutation_scale=7, lw=0.7,
                                     color=C_VEC,
                                     connectionstyle="arc3,rad=%.2f" % (0.10 * (1 - i))))
    ax.text(0.135, 0.575, "$W_{in}$", fontsize=6.6, ha="center", color=C_VEC)
    ax.text(0.2425, 0.885, "$N{=}100$ neurons, each $z_i\\in\\mathbb{R}^{1000}$",
            ha="center", fontsize=6.0)

    for y in ys:
        ax.add_patch(plt.Circle((0.345, y + 0.075), 0.011, fc="white",
                                ec=C_SCALAR, lw=0.8, zorder=4))
    ax.text(0.345, 0.105, "spike", ha="center", fontsize=5.6, color=C_SCALAR)

    ax.add_patch(FancyBboxPatch((0.455, 0.26), 0.032, 0.46,
                                boxstyle="round,pad=0.003", fc="#ebf8ff",
                                ec=C_VEC, lw=0.8, zorder=2))
    for i, y in enumerate(ys):
        ax.annotate("", xy=(0.451, 0.49), xytext=(0.372, y + 0.075),
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color=C_VEC,
                                    connectionstyle="arc3,rad=%.2f" % (0.06 * (i - 1))))
    ax.text(0.471, 0.795, "$h = \\sum_i s_i z_i\\in\\mathbb{R}^{1000}$",
            ha="center", fontsize=6.2, color=C_VEC)
    ax.text(0.471, 0.205, "population\nvector", ha="center", va="top",
            fontsize=5.8, color=C_VEC)

    ax.annotate("", xy=(0.640, 0.50), xytext=(0.496, 0.50),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=C_VEC))
    ax.text(0.568, 0.445, "$W_{emit}$  $1000\\times1000$", ha="center",
            fontsize=6.0, color=C_VEC)

    ax.add_patch(FancyArrowPatch((0.575, 0.190), (0.290, 0.095),
                                 arrowstyle="-|>", mutation_scale=7, lw=0.7,
                                 color=C_GREY, connectionstyle="arc3,rad=0.25"))
    ax.text(0.620, 0.215, "$W_{rec}$  $100\\times100$", fontsize=5.9, color=C_GREY)

    ax.add_patch(FancyBboxPatch((0.700, 0.36), 0.085, 0.28,
                                boxstyle="round,pad=0.004", fc="#f0fff4",
                                ec=C_ACC, lw=0.8))
    for k in range(5):
        ax.plot([0.715, 0.770], [0.415 + 0.045 * k] * 2, color=C_ACC, lw=1.4)
    ax.annotate("", xy=(0.696, 0.50), xytext=(0.646, 0.50),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=C_ACC))
    ax.text(0.7425, 0.325, "readout\n$W_{cls}$  $100\\times20$", ha="center",
            va="top", fontsize=5.9, color=C_ACC)
    ax.text(0.815, 0.50, "20\nclasses", ha="left", va="center", fontsize=6.2)

    # (b) single-trial states in the population state space ---------------
    axb = fig.add_subplot(gs[0, 1])
    with np.load(ANALYSIS / "capture_final_test.npz") as z:
        mean_state = z["mean_state"].astype(np.float32)   # (n, N, D)
        labels = z["labels"]
    classes = [0, 3, 7, 11, 15]
    rng = np.random.default_rng(0)
    idx = np.concatenate([
        rng.choice(np.flatnonzero(labels == c),
                   size=min(110, int((labels == c).sum())), replace=False)
        for c in classes
    ])
    X = mean_state[idx].reshape(len(idx), -1)
    X = X - X.mean(0, keepdims=True)
    u, s, v = torch.pca_lowrank(torch.from_numpy(X), q=6, center=False)
    proj = (torch.from_numpy(X) @ v[:, :2]).numpy()
    lab = labels[idx]
    var = (s ** 2 / (s ** 2).sum()).numpy()
    for i, c in enumerate(classes):
        col = plt.get_cmap("tab10")(i)
        m = lab == c
        axb.scatter(proj[m, 0], proj[m, 1], s=6.5, color=col, alpha=0.7,
                    edgecolors="none", label=f"class {c}")
        axb.scatter(proj[m, 0].mean(), proj[m, 1].mean(), s=46, marker="X",
                    color=col, edgecolors="black", lw=0.4, zorder=5)
    axb.set_xlabel(f"PC1  ({100 * var[0]:.0f} % of variance)")
    axb.set_ylabel(f"PC2  ({100 * var[1]:.0f} %)")
    axb.set_title("b   single-trial states ($N{=}100$)", loc="left",
                  fontsize=7.4)
    axb.legend(frameon=False, fontsize=5.8, markerscale=1.1, loc="upper right",
               handletextpad=0.2, borderpad=0.2)
    axb.text(0.02, 0.02, "one point per held-out sample;\nmean state over 500 bins"
             ", flattened to 100,000 dims", transform=axb.transAxes,
             fontsize=5.4, color=C_GREY, va="bottom")
    save(fig, "fig1_architecture")


# --------------------------------------------------------------- fig 2 ------
def fig2() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.75),
                               gridspec_kw={"width_ratios": [1.0, 1.12]})
    ds = [r[1] for r in WIDTH_SWEEP]
    a.plot(ds, [acc(r[3]) for r in WIDTH_SWEEP], "o-", color=C_VEC, ms=4)
    for r in WIDTH_SWEEP:
        dy = 8 if r[1] == 1000 else -9
        a.annotate(f"{r[2] / 1e3:.1f}k", (r[1], acc(r[3])),
                   textcoords="offset points", xytext=(4, dy), fontsize=5.8,
                   color=C_GREY)
    a.set_xscale("log")
    a.set_xlabel("state width $D$ (fixed $N{=}64$)")
    a.set_ylabel("TEST accuracy (%)")
    a.set_ylim(54, 80)
    titled(a, "a", "width, not parameters")

    labels = [c[0] for c in CONTROLS]
    vals = [acc(c[2]) for c in CONTROLS]
    colors = [C_SCALAR, C_SCALAR, C_SCALAR, C_PUR, C_VEC]
    x = np.arange(len(labels))
    bars = b.bar(x, vals, color=colors, width=0.62)
    for bar, c in zip(bars, CONTROLS):
        b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.6,
               f"{acc(c[2]):.1f} %", ha="center", fontsize=6.2)
        b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
               f"{c[1] / 1e6:.2f}M\n{c[3]} ep", ha="center", fontsize=5.4,
               color="white")
    b.set_xticks(x)
    b.set_xticklabels(labels, fontsize=6.0)
    b.set_ylabel("TEST accuracy (%)")
    b.set_ylim(0, 96)
    b.set_title("b   controls at the same readout", loc="left", fontsize=7.4)
    b.text(0.02, 0.96, "bars at different epochs; parameter count inside",
           transform=b.transAxes, fontsize=5.4, color=C_GREY, va="top")
    save(fig, "fig2_width_controls")


# --------------------------------------------------------------- fig 3 ------
def fig3() -> None:
    dir_ = RES / FINAL
    hist = list(csv.DictReader((dir_ / "history.csv").open()))
    ep = [int(h["epoch"]) for h in hist]
    tr = [100 * float(h["accuracy"]) for h in hist]
    va = [100 * float(h["val_accuracy"]) for h in hist]
    te = [100 * float(h["test_accuracy"]) for h in hist]

    fig = plt.figure(figsize=(7.15, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.60, 0.85], wspace=0.34)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ep, tr, color=C_GREY, lw=1.0, label="fit")
    ax.plot(ep, va, color=C_SCALAR, lw=1.0, ls="--", label="validation")
    ax.plot(ep, te, color=C_VEC, lw=1.3, label="TEST (official)")
    ax.scatter([BEST_EPOCH], [acc(BEST_CORRECT)], s=22, facecolors="none",
               edgecolors=C_ACC, zorder=5, lw=1.0)
    ax.annotate(f"best epoch {BEST_EPOCH}: {acc(BEST_CORRECT):.2f} %",
                (BEST_EPOCH, acc(BEST_CORRECT)), textcoords="offset points",
                xytext=(-104, -20), fontsize=5.8, color=C_ACC,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=C_ACC))
    ax.annotate(f"final: {acc(FINAL_CORRECT):.2f} %", (ep[-1], acc(FINAL_CORRECT)),
                textcoords="offset points", xytext=(-56, 9), fontsize=5.8,
                color=C_VEC)
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(60, 101)
    ax.set_xlim(0, max(ep) + 1)
    ax.legend(loc="lower right", frameon=False)
    titled(ax, "a", "final model, 30 epochs")

    axb = fig.add_subplot(gs[0, 1])
    axb.bar([0, 1], [100 * 414 / 424, 100 * 1580 / 1840],
            color=[C_VEC, C_PUR], width=0.6)
    axb.text(0, 100 * 414 / 424 + 1.2, f"{100 * 414 / 424:.1f} %\n$n{{=}}424$",
             ha="center", fontsize=5.9)
    axb.text(1, 100 * 1580 / 1840 + 1.2, f"{100 * 1580 / 1840:.1f} %\n$n{{=}}1840$",
             ha="center", fontsize=5.9)
    axb.set_xticks([0, 1])
    axb.set_xticklabels(["seen", "unseen"], fontsize=6.4)
    axb.set_ylim(70, 104)
    axb.set_ylabel("TEST accuracy (%)")
    titled(axb, "b", "speakers")

    axc = fig.add_subplot(gs[0, 2])
    with np.load(dir_ / "test_predictions.npz") as z:
        cm = z["confusion_matrix"].astype(float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    im = axc.imshow(cmn, cmap="Blues", vmin=0, vmax=1, aspect="auto",
                    interpolation="nearest")
    axc.set_xlabel("predicted class")
    axc.set_ylabel("true class")
    titled(axc, "c", "confusion")
    axc.tick_params(labelsize=5.4)
    cb = fig.colorbar(im, ax=axc, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=5.4)
    cb.set_label("fraction", fontsize=5.6)
    save(fig, "fig3_final_model")


# --------------------------------------------------------------- fig 4 ------
def fig4() -> None:
    fig = plt.figure(figsize=(7.15, 2.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 0.95, 1.0], wspace=0.40)

    ax = fig.add_subplot(gs[0, 0])
    sets = list(PROBE)
    x64 = np.arange(len(sets)) - 0.18
    x100 = x64 + 0.36
    ax.bar(x64, [acc(PROBE[s][0]) for s in sets], width=0.32, color=C_VEC,
           label="$N{=}64$")
    ax.bar(x100, [acc(PROBE[s][1]) for s in sets], width=0.32, color=C_PUR,
           label="$N{=}100$")
    for xs, key in ((x64, 0), (x100, 1)):
        for xx, s in zip(xs, sets):
            ax.text(xx, acc(PROBE[s][key]) + 1.6, f"{acc(PROBE[s][key]):.1f}",
                    ha="center", fontsize=5.5)
    ax.axhline(acc(1868), color=C_VEC, ls=":", lw=0.7)
    ax.set_xticks(np.arange(len(sets)))
    ax.set_xticklabels(sets, fontsize=5.9)
    ax.set_ylim(0, 122)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_ylabel("probe accuracy (%)")
    ax.legend(frameon=False, loc="upper right", ncol=2, handlelength=1.0,
              columnspacing=0.8)
    titled(ax, "a", "what the state decodes")

    axb = fig.add_subplot(gs[0, 1])
    with np.load(ANALYSIS / "capture_final_test.npz") as z:
        ms = z["mean_state"].astype(np.float32)
    flat = ms.reshape(ms.shape[0], -1)
    flat = flat - flat.mean(0, keepdims=True)
    gram = np.matmul(flat, flat.T)
    ev = np.clip(np.linalg.eigvalsh(gram)[::-1], 0, None)
    frac = ev / ev.sum()
    cum = np.cumsum(frac)
    axb.plot(np.arange(1, 201), 100 * cum[:200], color=C_VEC, lw=1.1)
    for n, lab in ((21, "90 %"), (58, "95 %")):
        axb.scatter([n], [100 * cum[n - 1]], s=14, color=C_BAD, zorder=4)
        axb.annotate(f"{n} comps\n{lab}", (n, 100 * cum[n - 1]),
                     textcoords="offset points", xytext=(6, -18), fontsize=5.5,
                     color=C_BAD)
    axb.set_xscale("log")
    axb.set_xlabel("components")
    axb.set_ylabel("cumulative variance (%)")
    axb.set_ylim(20, 102)
    axb.text(0.03, 0.30, "participation ratio\n7.5 of 100,000 dims",
             transform=axb.transAxes, fontsize=5.6, color=C_GREY)
    titled(axb, "b", "effective dimension")

    axc = fig.add_subplot(gs[0, 2])
    arms = list(FREEZE)
    x = np.arange(len(arms))
    axc.bar(x - 0.18, [acc(FREEZE[a][0]) for a in arms], 0.34, color=C_VEC,
            label="$N{=}64$")
    axc.bar(x + 0.18, [acc(FREEZE[a][1]) for a in arms], 0.34, color=C_PUR,
            label="$N{=}100$")
    axc.axhline(acc(1868), color=C_VEC, ls=":", lw=0.7)
    axc.set_xticks(x)
    axc.set_xticklabels(arms, fontsize=5.4)
    axc.set_ylabel("TEST accuracy (%)")
    axc.set_ylim(0, 118)
    axc.set_yticks([0, 20, 40, 60, 80, 100])
    axc.legend(frameon=False, loc="upper right", ncol=2, handlelength=1.0,
               columnspacing=0.8)
    titled(axc, "c", "frozen dynamics")
    save(fig, "fig4_mechanism")


# --------------------------------------------------------------- fig 5 ------
def fig5() -> None:
    ko = load_json("ablation_knockout.json")
    fp = load_json("ablation_fingerprints_final.json")
    mi = np.array(fp["mi_bits_fit"], dtype=float)
    drops = np.array([p["drop_pp"] for p in ko["per_neuron"]], dtype=float)
    cum = ko["cumulative"]

    fig = plt.figure(figsize=(7.15, 2.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.40)

    ax = fig.add_subplot(gs[0, 0])
    ax.hist(mi, bins=14, color=C_VEC, alpha=0.85, edgecolor="white")
    med = float(np.median(mi))
    ax.axvline(med, color=C_BAD, ls="--", lw=0.8)
    ax.text(med + 0.02, ax.get_ylim()[1] * 0.80, f"median {med:.2f} bits",
            fontsize=5.6, color=C_BAD)
    ax.set_xlabel("mutual information (bits, FIT)")
    ax.set_ylabel("neurons")
    titled(ax, "a", "class information")

    axb = fig.add_subplot(gs[0, 1])
    order = np.argsort(drops)[::-1]
    axb.plot(np.arange(1, len(drops) + 1), drops[order], "o-", ms=2.2, lw=0.9,
             color=C_SCALAR)
    axb.axhline(0, color=C_GREY, lw=0.6)
    axb.set_xlabel("neuron rank")
    axb.set_ylabel("TEST drop (pp)")
    axb.text(0.42, 0.92, f"{drops.mean():.2f} $\\pm$ {drops.std():.2f} pp",
             transform=axb.transAxes, fontsize=5.7, color=C_GREY)
    titled(axb, "b", "single silencing")

    axc = fig.add_subplot(gs[0, 2])
    ks = [c["k"] for c in cum]
    ys = [c["drop_pp"] for c in cum]
    axc.plot(ks, ys, "o-", ms=2.6, color=C_VEC)
    for k, off in ((1, (8, 6)), (16, (6, 8)), (100, (-10, -12))):
        y = ys[ks.index(k)]
        axc.annotate(f"{y:.1f} pp", (k, y), textcoords="offset points",
                     xytext=off, fontsize=5.7,
                     ha="right" if k == 100 else "left")
    axc.set_xscale("log")
    axc.set_xlim(0.8, 140)
    axc.set_ylim(-5, 96)
    axc.set_xlabel("neurons silenced (top-$k$)")
    axc.set_ylabel("TEST drop (pp)")
    titled(axc, "c", "no critical unit")
    save(fig, "fig5_distributed_code")


# --------------------------------------------------------------- fig 6 ------
def fig6() -> None:
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.15, 2.6),
                                  gridspec_kw={"width_ratios": [1.45, 0.75, 0.9],
                                               "wspace": 0.36})
    ops = [o[0] for o in AUG_OPS]
    x = np.arange(len(ops))
    d64 = [acc(o[1]) - acc(AUG_BASE["n64"]) for o in AUG_OPS]
    d100 = [acc(o[2]) - acc(AUG_BASE["n100"]) for o in AUG_OPS]
    a.bar(x - 0.19, d64, 0.36, color=C_VEC, label="$N{=}64$")
    a.bar(x + 0.19, d100, 0.36, color=C_PUR, label="$N{=}100$")
    a.axhline(0, color=C_GREY, lw=0.6)
    a.set_xticks(x)
    a.set_xticklabels(ops, fontsize=6.2)
    a.set_ylabel("$\\Delta$ TEST accuracy (pp)")
    a.set_ylim(-3.6, 3.6)
    a.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.82),
             ncol=2, handlelength=1.0, columnspacing=0.8)
    titled(a, "a", "single operations, 5 epochs")

    ps = [p for p, _ in DROPOUT_SWEEP]
    vv = [acc(v) for _, v in DROPOUT_SWEEP]
    b.plot(ps, vv, "o-", color=C_VEC, ms=4)
    b.axhline(acc(AUG_BASE["n64"]), color=C_GREY, ls=":", lw=0.8)
    b.set_xticks(ps)
    b.set_xticklabels(["0.05", "0.10", "0.15"], fontsize=6.2)
    b.set_xlabel("dropout $p$")
    b.set_ylabel("TEST accuracy (%)")
    b.set_ylim(75.6, 79.6)
    titled(b, "b", "plateau")

    vals = [acc(v) for _, v in INTERACTION]
    x = np.arange(len(INTERACTION))
    c.bar(x, vals, color=[C_GREY, C_VEC, C_BAD, C_GREY], width=0.6)
    for xx, v in zip(x, vals):
        c.text(xx, v + 0.12, f"{v:.1f}", ha="center", fontsize=5.9)
    c.set_xticks(x)
    c.set_xticklabels([k for k, _ in INTERACTION], fontsize=6.2)
    c.set_ylim(75.4, 79.2)
    c.set_ylabel("TEST accuracy (%)")
    titled(c, "c", "do not stack")
    save(fig, "fig6_augmentation")


# --------------------------------------------------------------- S1 ---------
def s1() -> None:
    rows, mats = [], []
    for tag, label in PERCLASS_RUNS:
        p = RES / tag / "test_predictions.npz"
        if not p.exists():
            print(f"[figures] S1 skip {tag} (no test_predictions.npz)")
            continue
        with np.load(p) as z:
            mats.append(z["per_class_accuracy"] * 100)
        rows.append(label)
    if not mats:
        return
    mat = np.vstack(mats)
    fig, ax = plt.subplots(figsize=(7.15, 0.30 * len(rows) + 1.0))
    im = ax.imshow(mat, cmap="viridis", vmin=40, vmax=100, aspect="auto")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows, fontsize=6.2)
    ax.set_xticks(np.arange(20))
    ax.set_xticklabels([str(i) for i in range(20)], fontsize=6)
    ax.set_xlabel("class")
    ax.set_title("per-class TEST accuracy across key runs", loc="left",
                 fontsize=7.4)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center",
                    fontsize=4.6, color="white" if mat[i, j] < 75 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.024, pad=0.02)
    cb.set_label("accuracy (%)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    save(fig, "figS1_per_class")


# --------------------------------------------------------------- S2 ---------
def s2() -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    xs = [v[1] for v in V31]
    ys = [acc(v[2]) for v in V31]
    ax.scatter(xs, ys, s=40, color=[C_VEC, C_ACC], zorder=3)
    for (lab, p, c), xx, yy in zip(V31, xs, ys):
        ax.annotate(f"{lab}: {acc(c):.1f} %", (xx, yy),
                    textcoords="offset points", xytext=(7, -3), fontsize=6.0)
    ax.set_xlim(0.85e6, 2.45e6)
    ax.set_ylim(75.4, 78.2)
    ax.set_xlabel("parameters")
    ax.set_ylabel("TEST accuracy (%)")
    ax.set_title("diagonal emission: same accuracy,\nhalf the parameters",
                 loc="left", fontsize=6.9)
    save(fig, "figS2_v31_params")


# --------------------------------------------------------------- S3 ---------
def s3() -> None:
    with np.load(ANALYSIS / "capture_final_test.npz") as z:
        counts = z["counts"].astype(np.float32)
        labels = z["labels"]
    fig = plt.figure(figsize=(7.15, 2.55))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.30)

    ax = fig.add_subplot(gs[0, 0])
    per_class = np.stack([counts[labels == c].mean(0) for c in range(20)])
    order = np.argsort(per_class.mean(0))
    im = ax.imshow(per_class[:, order], aspect="auto", cmap="magma",
                   interpolation="nearest")
    ax.set_xlabel("neuron (sorted by mean counts)")
    ax.set_ylabel("class")
    titled(ax, "a", "mean spike counts")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("spikes / 500 bins", fontsize=5.6)
    cb.ax.tick_params(labelsize=5.4)

    axb = fig.add_subplot(gs[0, 1])
    for c, col in zip([0, 3, 7], [C_VEC, C_BAD, C_ACC]):
        axb.plot(counts[labels == c][:1].squeeze(0), color=col, lw=0.9,
                 label=f"class {c}")
    axb.set_xlabel("neuron")
    axb.set_ylabel("spike count")
    axb.legend(frameon=False)
    titled(axb, "b", "one sample per class")
    save(fig, "figS3_population_code")


def fig7() -> None:
    """Test-time robustness of the final model (no retraining)."""
    data = load_json("robustness_v3_n100_spec_ep30.json")
    rows = data["rows"]
    base = [r for r in rows if r["family"] == "baseline"][0]["accuracy"]
    fig = plt.figure(figsize=(7.15, 2.25))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.42)

    def series(prefix: str, drop_zero: bool) -> list:
        pts = sorted((r["value"], r["accuracy"]) for r in rows
                     if r["family"] == prefix and r["value"] > 0)
        if drop_zero:
            return [p[0] for p in pts], [p[1] for p in pts]
        return [0.0] + [p[0] for p in pts], [base] + [p[1] for p in pts]

    for i, (prefix, xlabel, title, logx) in enumerate([
            ("offset_ms", "global time offset (ms)", "timing jitter", False),
            ("poisson_hz", "background rate (Hz per channel)", "additive background",
             True),
            ("dropout_p", "event dropout probability", "event loss", False)]):
        ax = fig.add_subplot(gs[0, i])
        xs, ys = series(prefix, logx)
        ys = [100.0 * y for y in ys]
        ax.plot(xs, ys, "-o", color=C_VEC, ms=3.2, lw=1.1)
        ax.axhline(100.0 * base, color=C_GREY, lw=0.7, ls="--")
        ax.axhline(100.0 / 20, color=C_BAD, lw=0.7, ls=":")
        if logx:
            ax.set_xscale("log")
            ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50])
            ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20", "50"])
        ax.set_xlabel(xlabel)
        if i == 0:
            ax.set_ylabel("test accuracy (%)")
        ax.set_ylim(0, 98)
        titled(ax, "abc"[i], title)
        ax.text(xs[-1], max(ys[-1] + 5, 30), f"{ys[-1]:.1f}%", ha="right",
                fontsize=6.0, color=C_VEC)
        if prefix == "poisson_hz":
            ax.text(0.04, 10, f"chance {100.0 / 20:.1f}%", fontsize=5.8,
                    color=C_BAD)


    save(fig, "fig7_robustness")


def s4() -> None:
    """Cost of the final model and accuracy against the wall-clock budget."""
    runs = [("v3_vector_scalar_d1", "scalar $D{=}1$", C_SCALAR),
            ("v3_vector_d1000", "$N{=}64$, $D{=}1000$", C_VEC),
            ("v3_n100_baseline_ep5", "$N{=}100$, 5 ep", C_GREY),
            ("v3_n100_aug_dropout_ep20", "$N{=}100$ + dropout", C_VEC),
            ("v3_n100_spec_ep30", "final model, 30 ep", C_ACC)]
    fig = plt.figure(figsize=(7.15, 2.55))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.34)

    ax = fig.add_subplot(gs[0, 0])
    for tag, label, col in runs:
        r = load_run(tag)
        hours = r["wall_seconds"] / 3600.0
        ax.plot(hours, 100.0 * r["test_accuracy"], "o", color=col, ms=4.2)
        ax.annotate(label, (hours, 100.0 * r["test_accuracy"]),
                    textcoords="offset points", xytext=(4, -7), fontsize=5.6,
                    color=col)
    ax.set_xlabel("training wall-clock (h)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_xlim(0, 5.4)
    ax.set_ylim(40, 92)
    titled(ax, "a", "accuracy vs. training budget")

    axb = fig.add_subplot(gs[0, 1])
    tags = ["v3_vector_scalar_d1", "v3_vector_d1000", "v3_n100_spec_ep30"]
    names = ["scalar\n$D{=}1$", "$N{=}64$\n$D{=}1000$", "final\n$N{=}100$"]
    epoch_s, vram = [], []
    for tag in tags:
        r = load_run(tag)
        epoch_s.append(r["epoch_seconds"])
        vram.append(r["vram_gib"])
    x = np.arange(len(tags))
    w = 0.38
    axb.bar(x - w / 2, epoch_s, w, color=C_GREY)
    axb.set_ylabel("s / epoch")
    axb.set_xticks(x)
    axb.set_xticklabels(names)
    axb.set_ylim(0, max(epoch_s) * 1.55)
    for xi, v in zip(x - w / 2, epoch_s):
        axb.text(xi, v + 8, f"{v:.0f}", ha="center", fontsize=5.8)
    ax2 = axb.twinx()
    ax2.bar(x + w / 2, vram, w, color=C_VEC, alpha=0.55)
    ax2.set_ylabel("peak VRAM (GiB)", color=C_VEC)
    ax2.tick_params(axis="y", colors=C_VEC)
    ax2.set_ylim(0, 7.2)
    ax2.axhline(4.9, color=C_BAD, lw=0.7, ls="--")
    for xi, v in zip(x + w / 2, vram):
        ax2.text(xi, v + 0.12, f"{v:.1f}", ha="center", fontsize=5.8, color=C_VEC)
    titled(axb, "b", "per-epoch cost (grey: s/epoch, blue: VRAM)")
    save(fig, "figS4_cost")


# ---------------------------------------------------------------- main ------
FIGURES = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6, 7: fig7,
           "s1": s1, "s2": s2, "s3": s3, "s4": s4}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None,
                    help="build only these figures, e.g. --only 1 3 s1")
    args = ap.parse_args()
    style()
    keys = args.only if args.only else list(FIGURES)
    for key in keys:
        try:
            k = int(key)
        except ValueError:
            k = key
        if k not in FIGURES:
            raise SystemExit(f"unknown figure {key!r}")
        FIGURES[k]()


if __name__ == "__main__":
    main()
