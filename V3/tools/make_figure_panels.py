#!/usr/bin/env python
"""Build every manuscript panel as its own standalone figure.

``make_manuscript_figures.py`` composes multi-panel figures (a, b, c ...).  A
journal submission or a Word document often needs the panels one at a time, so
each panel here is a fresh ``plt.figure`` with a single axes, its own title,
its own axis labels, its own legend and the plot's own colorbar: a cropped
panel carries everything it needs.  There are no panel letters and no shared
axis labels.

The numbers are *not* re-derived.  Every panel reads the same artifact as its
counterpart in ``make_manuscript_figures.py`` -- the capture npz, ``history.csv``
files, ``test_predictions.npz`` and the JSONs under ``V3/results/analysis/`` --
or the same transcribed table (``WIDTH_SWEEP``, ``CONTROLS``, ... with the
registry tag next to it).  Style is shared with that script: 7.5 pt base font,
Type-42 PDF fonts, PDF plus 300 dpi PNG.

Outputs land in ``figures/manuscript/panels/``:

    python V3/tools/make_figure_panels.py                  # every panel
    python V3/tools/make_figure_panels.py --only fig03 fig07
    python V3/tools/make_figure_panels.py --only fig1      # unique prefixes work

CPU only: nothing here touches the GPU (the one PCA in fig02 runs on the CPU).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "V3" / "results"
ANALYSIS = RES / "analysis"
OUT = ROOT / "figures" / "manuscript" / "panels"

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
    ("scalar\n$N{=}1000$", 1_024_703, 987, 30, "v3_emerg_n1000_d1_ep30_scalar_d1"),
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
    print(f"[panels] {pdf.relative_to(ROOT)}")
    return pdf


def acc(correct: int, total: int = N_TEST) -> float:
    return 100.0 * correct / total


def title(ax, text: str) -> None:
    """Plain descriptive panel title (never a letter label)."""
    ax.set_title(text, loc="left", fontsize=7.4)


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
    with (RES / tag / "history.csv").open() as fh:
        return list(csv.DictReader(fh))


_CAPTURE: dict | None = None


def capture() -> dict:
    """Load the 900 MB capture npz once; every panel reuses the same arrays."""
    global _CAPTURE
    if _CAPTURE is None:
        with np.load(ANALYSIS / "capture_final_test.npz") as z:
            _CAPTURE = {"mean_state": z["mean_state"],   # (n, N, D) float16
                        "counts": z["counts"].astype(np.float32),
                        "labels": z["labels"]}
    return _CAPTURE


def _neuron_box(ax, x, y, w, h, n_strips=7, color=C_VEC):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004",
                                fc="white", ec=color, lw=0.8, zorder=3))
    for k in range(n_strips):
        yy = y + h * (0.12 + 0.72 * k / max(n_strips - 1, 1))
        ax.plot([x + 0.006, x + w - 0.006], [yy, yy], color=color, lw=1.6,
                solid_capstyle="butt", zorder=4)


# ------------------------------------------------------------- fig01 --------
def fig01_schematic() -> None:
    """Architecture schematic (was fig1 panel a).

    Re-laid out on the wider single-panel canvas so that no label sits on a
    box, an arrow or another label; the drawing is the same one.
    """
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.845)
    ax.axis("off")
    title(ax, "Population-vector architecture")


    # 700 input channels --------------------------------------------------
    ax.add_patch(FancyBboxPatch((0.020, 0.300), 0.055, 0.400,
                                boxstyle="round,pad=0.004", fc="#edf2f7",
                                ec=C_GREY, lw=0.8))
    for k in range(9):
        yy = 0.335 + 0.330 * k / 8
        ax.plot([0.026, 0.069], [yy, yy], color=C_GREY, lw=1.0)
    ax.text(0.0475, 0.265, "700\ninput\nchannels", ha="center", va="top",
            fontsize=6.0, color=C_GREY)

    # N vector-valued neurons ---------------------------------------------
    ys = [0.10, 0.34, 0.58]
    for i, y in enumerate(ys):
        _neuron_box(ax, 0.155, y, 0.085, 0.15)
        ax.text(0.1975, y + 0.160, f"$z_{{{i + 1}}}$", ha="center", fontsize=6.4)
        ax.add_patch(FancyArrowPatch((0.078, 0.50), (0.151, y + 0.075),
                                     arrowstyle="-|>", mutation_scale=7, lw=0.7,
                                     color=C_VEC,
                                     connectionstyle="arc3,rad=%.2f" % (0.10 * (1 - i))))
    ax.text(0.118, 0.540, "$W_{in}$", fontsize=6.6, ha="center", va="bottom",
            color=C_VEC)
    ax.text(0.020, 0.792, "$N{=}100$ neurons, each $z_i\\in\\mathbb{R}^{1000}$",
            ha="left", va="bottom", fontsize=6.0)

    # spikes ---------------------------------------------------------------
    for y in ys:
        ax.plot([0.325], [y + 0.075], marker="o", ms=4.4, mfc="white",
                mec=C_SCALAR, mew=0.8, zorder=4, clip_on=False)
    ax.text(0.325, 0.700, "spike", ha="center", va="bottom", fontsize=5.6,
            color=C_SCALAR)

    # population vector ----------------------------------------------------
    ax.add_patch(FancyBboxPatch((0.420, 0.245), 0.034, 0.440,
                                boxstyle="round,pad=0.003", fc="#ebf8ff",
                                ec=C_VEC, lw=0.8, zorder=2))
    for i, y in enumerate(ys):
        ax.annotate("", xy=(0.413, 0.485), xytext=(0.337, y + 0.075),
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color=C_VEC,
                                    connectionstyle="arc3,rad=%.2f" % (0.06 * (i - 1))))
    ax.text(0.450, 0.752, "$h = \\sum_i s_i z_i\\in\\mathbb{R}^{1000}$",
            ha="center", va="bottom", fontsize=6.2, color=C_VEC)
    ax.text(0.450, 0.702, "population vector", ha="center", va="bottom",
            fontsize=5.8, color=C_VEC)

    # emission -------------------------------------------------------------
    ax.annotate("", xy=(0.648, 0.465), xytext=(0.462, 0.465),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=C_VEC))
    ax.text(0.555, 0.415, "$W_{emit}$  $1000\\times1000$", ha="center",
            va="top", fontsize=6.0, color=C_VEC)

    # recurrent feedback ---------------------------------------------------
    ax.add_patch(FancyArrowPatch((0.625, 0.170), (0.240, 0.075),
                                 arrowstyle="-|>", mutation_scale=7, lw=0.7,
                                 color=C_GREY, connectionstyle="arc3,rad=0.15"))
    ax.text(0.470, 0.128, "$W_{rec}$  $100\\times100$", ha="center",
            va="bottom", fontsize=5.9, color=C_GREY)

    # readout --------------------------------------------------------------
    ax.add_patch(FancyBboxPatch((0.665, 0.330), 0.085, 0.300,
                                boxstyle="round,pad=0.004", fc="#f0fff4",
                                ec=C_ACC, lw=0.8))
    for k in range(5):
        ax.plot([0.680, 0.735], [0.380 + 0.050 * k] * 2, color=C_ACC, lw=1.4)
    ax.annotate("", xy=(0.662, 0.480), xytext=(0.653, 0.480),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=C_ACC,
                                mutation_scale=8))
    ax.text(0.7075, 0.300, "readout\n$W_{cls}$  $100\\times20$", ha="center",
            va="top", fontsize=5.9, color=C_ACC)
    ax.text(0.762, 0.480, "20\nclasses", ha="left", va="center", fontsize=6.2)

    fig.tight_layout()
    save(fig, "fig01_schematic")


# ------------------------------------------------------------- fig02 --------
def fig02_single_trial_states() -> None:
    """PCA of single-trial population states (was fig1 panel b)."""
    # CPU only: this job must never initialise CUDA.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import torch

    torch.manual_seed(0)          # the source leaves it to the default state (0)
    cap = capture()
    mean_state = cap["mean_state"].astype(np.float32)   # (n, N, D)
    labels = cap["labels"]
    classes = [0, 3, 7, 11, 15]
    rng = np.random.default_rng(0)
    idx = np.concatenate([
        rng.choice(np.flatnonzero(labels == c),
                   size=min(110, int((labels == c).sum())), replace=False)
        for c in classes
    ])
    X = mean_state[idx].reshape(len(idx), -1)
    del mean_state
    X = X - X.mean(0, keepdims=True)
    u, s, v = torch.pca_lowrank(torch.from_numpy(X), q=6, center=False)
    proj = (torch.from_numpy(X) @ v[:, :2]).numpy()
    del X
    lab = labels[idx]
    var = (s ** 2 / (s ** 2).sum()).numpy()

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for i, c in enumerate(classes):
        col = plt.get_cmap("tab10")(i)
        m = lab == c
        ax.scatter(proj[m, 0], proj[m, 1], s=6.5, color=col, alpha=0.7,
                   edgecolors="none", label=f"class {c}")
        ax.scatter(proj[m, 0].mean(), proj[m, 1].mean(), s=46, marker="X",
                   color=col, edgecolors="black", lw=0.4, zorder=5)
    ax.set_xlabel(f"PC1  ({100 * var[0]:.0f} % of variance)")
    ax.set_ylabel(f"PC2  ({100 * var[1]:.0f} %)")
    title(ax, "Population states ($N{=}100$)")
    ax.legend(frameon=False, fontsize=5.6, ncol=2, loc="lower left",
              markerscale=1.8, handletextpad=0.3, columnspacing=0.8,
              borderpad=0.2, labelspacing=0.3)
    ax.text(0.01, 0.985, "single held-out trials; 500-bin mean state, "
            "100,000 dims", transform=ax.transAxes, ha="left", va="top",
            fontsize=5.2, color=C_GREY)
    ax.set_ylim(-30.5, 33.0)
    fig.tight_layout()
    save(fig, "fig02_single_trial_states")


# ------------------------------------------------------------- fig03 --------
def fig03_width_sweep() -> None:
    """TEST accuracy against state width (was fig2 panel a)."""
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ds = [r[1] for r in WIDTH_SWEEP]
    ax.plot(ds, [acc(r[3]) for r in WIDTH_SWEEP], "o-", color=C_VEC, ms=4)
    for r in WIDTH_SWEEP:
        last = r[1] == 1000
        ax.annotate(f"{r[2] / 1e3:.1f}k", (r[1], acc(r[3])),
                    textcoords="offset points",
                    xytext=(-5, 7) if last else (5, -10),
                    ha="right" if last else "left", fontsize=5.8,
                    color=C_GREY)
    ax.set_xscale("log")
    ax.set_xlim(2.6, 2100)
    ax.set_xlabel("state width $D$ (fixed $N{=}64$)")
    ax.set_ylabel("TEST accuracy (%)")
    ax.set_ylim(54, 80)
    title(ax, "TEST accuracy vs. state width")
    fig.tight_layout()
    save(fig, "fig03_width_sweep")


# ------------------------------------------------------------- fig04 --------
def fig04_controls() -> None:
    """Control arms at the same readout (was fig2 panel b)."""
    fig, ax = plt.subplots(figsize=(3.8, 2.9))
    labels = [c[0] for c in CONTROLS]
    vals = [acc(c[2]) for c in CONTROLS]
    colors = [C_SCALAR, C_SCALAR, C_SCALAR, C_PUR, C_VEC]
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, width=0.62)
    for bar, c in zip(bars, CONTROLS):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.6,
                f"{acc(c[2]):.1f} %", ha="center", fontsize=6.2)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                f"{c[1] / 1e6:.2f}M\n{c[3]} ep", ha="center", fontsize=5.4,
                color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.0)
    ax.set_ylabel("TEST accuracy (%)")
    ax.set_ylim(0, 96)
    title(ax, "Controls at the same readout")
    ax.text(0.02, 0.96, "bars at different epochs; parameter count inside",
            transform=ax.transAxes, fontsize=5.4, color=C_GREY, va="top")
    fig.tight_layout()
    save(fig, "fig04_controls")


# ------------------------------------------------------------- fig05 --------
def fig05_training_curves() -> None:
    """Fit/validation/TEST curves of the final model (was fig3 panel a)."""
    hist = load_history(FINAL)
    ep = [int(h["epoch"]) for h in hist]
    tr = [100 * float(h["accuracy"]) for h in hist]
    va = [100 * float(h["val_accuracy"]) for h in hist]
    te = [100 * float(h["test_accuracy"]) for h in hist]

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.plot(ep, tr, color=C_GREY, lw=1.0, label="fit")
    ax.plot(ep, va, color=C_SCALAR, lw=1.0, ls="--", label="validation")
    ax.plot(ep, te, color=C_VEC, lw=1.3, label="TEST (official)")
    ax.scatter([BEST_EPOCH], [acc(BEST_CORRECT)], s=22, facecolors="none",
               edgecolors=C_ACC, zorder=5, lw=1.0)
    ax.annotate(f"best ep {BEST_EPOCH}: {acc(BEST_CORRECT):.2f} %\n"
                f"final ep {ep[-1]}: {acc(FINAL_CORRECT):.2f} %",
                (BEST_EPOCH, acc(BEST_CORRECT)), textcoords="offset points",
                xytext=(-2, 8), ha="right", va="bottom", fontsize=5.8,
                color=C_VEC,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=C_ACC))
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(60, 101)
    ax.set_xlim(0, max(ep) + 1)
    ax.legend(loc="lower right", frameon=False)
    title(ax, "Final model, 30 epochs")
    fig.tight_layout()
    save(fig, "fig05_training_curves")


# ------------------------------------------------------------- fig06 --------
def fig06_seen_unseen() -> None:
    """Seen versus unseen speakers (was fig3 panel b)."""
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.bar([0, 1], [100 * 414 / 424, 100 * 1580 / 1840],
           color=[C_VEC, C_PUR], width=0.6)
    ax.text(0, 100 * 414 / 424 + 1.2, f"{100 * 414 / 424:.1f} %\n$n{{=}}424$",
            ha="center", fontsize=5.9)
    ax.text(1, 100 * 1580 / 1840 + 1.2, f"{100 * 1580 / 1840:.1f} %\n$n{{=}}1840$",
            ha="center", fontsize=5.9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["seen", "unseen"], fontsize=6.4)
    ax.set_ylim(70, 104)
    ax.set_ylabel("TEST accuracy (%)")
    ax.set_xlabel("speaker split")
    title(ax, "Seen vs. unseen speakers")
    fig.tight_layout()
    save(fig, "fig06_seen_unseen")


# ------------------------------------------------------------- fig07 --------
def fig07_confusion() -> None:
    """Confusion matrix of the final model (was fig3 panel c)."""
    with np.load(RES / FINAL / "test_predictions.npz") as z:
        cm = z["confusion_matrix"].astype(float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(3.2, 2.9))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1, aspect="auto",
                   interpolation="nearest")
    ax.set_xticks([0, 5, 10, 15])
    ax.set_yticks([0, 5, 10, 15])
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    title(ax, "Confusion matrix, final model")
    ax.tick_params(labelsize=5.4)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=5.4)
    cb.set_label("fraction", fontsize=5.6)
    fig.tight_layout()
    save(fig, "fig07_confusion")


# ------------------------------------------------------------- fig08 --------
def fig08_probe() -> None:
    """Linear probe on the population state (was fig4 panel a)."""
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
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
    ax.set_xlabel("feature set")
    ax.legend(frameon=False, loc="upper right", ncol=2, handlelength=1.0,
              columnspacing=0.8)
    title(ax, "What the state decodes")
    fig.tight_layout()
    save(fig, "fig08_probe")


# ------------------------------------------------------------- fig09 --------
def fig09_effective_dim() -> None:
    """Cumulative spectrum of the state (was fig4 panel b)."""
    cap = capture()
    ms = cap["mean_state"].astype(np.float32)
    flat = ms.reshape(ms.shape[0], -1)
    del ms
    flat = flat - flat.mean(0, keepdims=True)
    gram = np.matmul(flat, flat.T)
    del flat
    ev = np.clip(np.linalg.eigvalsh(gram)[::-1], 0, None)
    del gram
    frac = ev / ev.sum()
    cum = np.cumsum(frac)

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.plot(np.arange(1, 201), 100 * cum[:200], color=C_VEC, lw=1.1)
    for n, lab in ((21, "90 %"), (58, "95 %")):
        ax.scatter([n], [100 * cum[n - 1]], s=14, color=C_BAD, zorder=4)
        ax.annotate(f"{n} comps\n{lab}", (n, 100 * cum[n - 1]),
                    textcoords="offset points", xytext=(6, -18), fontsize=5.5,
                    color=C_BAD)
    ax.set_xscale("log")
    ax.set_xlabel("components")
    ax.set_ylabel("cumulative variance (%)")
    ax.set_ylim(20, 102)
    ax.text(0.03, 0.30, "participation ratio\n7.5 of 100,000 dims",
            transform=ax.transAxes, fontsize=5.6, color=C_GREY)
    title(ax, "Effective dimension of the state")
    fig.tight_layout()
    save(fig, "fig09_effective_dim")


# ------------------------------------------------------------- fig10 --------
def fig10_frozen_dynamics() -> None:
    """Trained versus frozen dynamics (was fig4 panel c)."""
    fig, ax = plt.subplots(figsize=(3.8, 2.9))
    arms = list(FREEZE)
    x = np.arange(len(arms))
    ax.bar(x - 0.18, [acc(FREEZE[a][0]) for a in arms], 0.34, color=C_VEC,
           label="$N{=}64$")
    ax.bar(x + 0.18, [acc(FREEZE[a][1]) for a in arms], 0.34, color=C_PUR,
           label="$N{=}100$")
    ax.axhline(acc(1868), color=C_VEC, ls=":", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(arms, fontsize=5.4)
    ax.set_ylabel("TEST accuracy (%)")
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.legend(frameon=False, loc="upper right", ncol=2, handlelength=1.0,
              columnspacing=0.8)
    title(ax, "Trained vs. frozen dynamics")
    fig.tight_layout()
    save(fig, "fig10_frozen_dynamics")


# ------------------------------------------------------------- fig11 --------
def fig11_mi_distribution() -> None:
    """Per-neuron mutual information (was fig5 panel a)."""
    fp = load_json("ablation_fingerprints_final.json")
    mi = np.array(fp["mi_bits_fit"], dtype=float)

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.hist(mi, bins=14, color=C_VEC, alpha=0.85, edgecolor="white")
    med = float(np.median(mi))
    ax.axvline(med, color=C_BAD, ls="--", lw=0.8)
    ax.set_ylim(0, 17)
    ax.text(med + 0.02, 16.6, f"median {med:.2f} bits", va="top",
            fontsize=5.6, color=C_BAD)
    ax.set_xlabel("mutual information (bits, FIT)")
    ax.set_ylabel("neurons")
    title(ax, "Class information per neuron")
    fig.tight_layout()
    save(fig, "fig11_mi_distribution")


# ------------------------------------------------------------- fig12 --------
def fig12_knockout_drop() -> None:
    """Sorted single-neuron silencing effect (was fig5 panel b)."""
    ko = load_json("ablation_knockout.json")
    drops = np.array([p["drop_pp"] for p in ko["per_neuron"]], dtype=float)
    order = np.argsort(drops)[::-1]

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.plot(np.arange(1, len(drops) + 1), drops[order], "o-", ms=2.2, lw=0.9,
            color=C_SCALAR)
    ax.axhline(0, color=C_GREY, lw=0.6)
    ax.set_xlabel("neuron rank")
    ax.set_ylabel("TEST drop (pp)")
    ax.text(0.42, 0.92, f"{drops.mean():.2f} $\\pm$ {drops.std():.2f} pp",
            transform=ax.transAxes, fontsize=5.7, color=C_GREY)
    title(ax, "Single-neuron silencing")
    fig.tight_layout()
    save(fig, "fig12_knockout_drop")


# ------------------------------------------------------------- fig13 --------
def fig13_cumulative_topk() -> None:
    """Cumulative top-k silencing (was fig5 panel c)."""
    ko = load_json("ablation_knockout.json")
    cum = ko["cumulative"]
    ks = [c["k"] for c in cum]
    ys = [c["drop_pp"] for c in cum]

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.plot(ks, ys, "o-", ms=2.6, color=C_VEC)
    for k, off, ha in ((1, (2, 12), "left"), (16, (-6, 6), "right"),
                       (100, (-10, 4), "right")):
        y = ys[ks.index(k)]
        ax.annotate(f"{y:.1f} pp", (k, y), textcoords="offset points",
                    xytext=off, fontsize=5.7, ha=ha)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 140)
    ax.set_ylim(-5, 96)
    ax.set_xlabel("neurons silenced (top-$k$)")
    ax.set_ylabel("TEST drop (pp)")
    title(ax, "Cumulative top-$k$ silencing")
    fig.tight_layout()
    save(fig, "fig13_cumulative_topk")


# ------------------------------------------------------------- fig14 --------
def fig14_aug_single_ops() -> None:
    """Single augmentation operations (was fig6 panel a)."""
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ops = [o[0] for o in AUG_OPS]
    x = np.arange(len(ops))
    d64 = [acc(o[1]) - acc(AUG_BASE["n64"]) for o in AUG_OPS]
    d100 = [acc(o[2]) - acc(AUG_BASE["n100"]) for o in AUG_OPS]
    ax.bar(x - 0.19, d64, 0.36, color=C_VEC, label="$N{=}64$")
    ax.bar(x + 0.19, d100, 0.36, color=C_PUR, label="$N{=}100$")
    ax.axhline(0, color=C_GREY, lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(ops, fontsize=6.2)
    ax.set_ylabel("$\\Delta$ TEST accuracy (pp)")
    ax.set_ylim(-3.6, 3.6)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.82),
              ncol=2, handlelength=1.0, columnspacing=0.8)
    title(ax, "Single operations, 5 epochs")
    fig.tight_layout()
    save(fig, "fig14_aug_single_ops")


# ------------------------------------------------------------- fig15 --------
def fig15_aug_dropout_plateau() -> None:
    """Dropout-probability sweep (was fig6 panel b)."""
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ps = [p for p, _ in DROPOUT_SWEEP]
    vv = [acc(v) for _, v in DROPOUT_SWEEP]
    ax.plot(ps, vv, "o-", color=C_VEC, ms=4, label="dropout")
    ax.axhline(acc(AUG_BASE["n64"]), color=C_GREY, ls=":", lw=0.8,
               label="no-augmentation baseline")
    ax.set_xticks(ps)
    ax.set_xticklabels(["0.05", "0.10", "0.15"], fontsize=6.2)
    ax.set_xlabel("dropout $p$")
    ax.set_ylabel("TEST accuracy (%)")
    ax.set_ylim(75.6, 79.6)
    ax.legend(frameon=False, loc="lower right", fontsize=5.8)
    title(ax, "Dropout plateau ($N{=}64$, 5 epochs)")
    fig.tight_layout()
    save(fig, "fig15_aug_dropout_plateau")


# ------------------------------------------------------------- fig16 --------
def fig16_aug_interaction() -> None:
    """Single operations versus their combination (was fig6 panel c)."""
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    vals = [acc(v) for _, v in INTERACTION]
    x = np.arange(len(INTERACTION))
    ax.bar(x, vals, color=[C_GREY, C_VEC, C_BAD, C_GREY], width=0.6)
    for xx, v in zip(x, vals):
        ax.text(xx, v + 0.12, f"{v:.1f}", ha="center", fontsize=5.9)
    ax.set_xticks(x)
    ax.set_xticklabels([k for k, _ in INTERACTION], fontsize=6.2)
    ax.set_ylim(75.4, 79.2)
    ax.set_ylabel("TEST accuracy (%)")
    ax.set_xlabel("operation")
    title(ax, "Single vs. combined augmentation")
    fig.tight_layout()
    save(fig, "fig16_aug_interaction")


# ------------------------------------------------------------- fig17-19 -----
def _robustness(prefix: str, xlabel: str, name: str, fig_title: str,
                logx: bool) -> None:
    data = load_json("robustness_v3_n100_spec_ep30.json")
    rows = data["rows"]
    base = [r for r in rows if r["family"] == "baseline"][0]["accuracy"]

    pts = sorted((r["value"], r["accuracy"]) for r in rows
                 if r["family"] == prefix and r["value"] > 0)
    if logx:
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    else:
        xs = [0.0] + [p[0] for p in pts]
        ys = [base] + [p[1] for p in pts]
    ys = [100.0 * y for y in ys]

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.plot(xs, ys, "-o", color=C_VEC, ms=3.2, lw=1.1)
    ax.axhline(100.0 * base, color=C_GREY, lw=0.7, ls="--")
    ax.axhline(100.0 / 20, color=C_BAD, lw=0.7, ls=":")
    if logx:
        ax.set_xscale("log")
        ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50])
        ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20", "50"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(0, 98)
    title(ax, fig_title)
    ax.text(xs[-1], max(ys[-1] + 5, 30), f"{ys[-1]:.1f}%", ha="right",
            fontsize=6.0, color=C_VEC)
    if prefix == "poisson_hz":
        ax.text(0.02, 0.055, f"chance {100.0 / 20:.1f}%", fontsize=5.8,
                color=C_BAD, transform=ax.transAxes, ha="left", va="bottom")
    fig.tight_layout()
    save(fig, name)


def fig17_robustness_offset() -> None:
    """Accuracy under a global time offset (was fig7 panel a)."""
    _robustness("offset_ms", "global time offset (ms)",
                "fig17_robustness_offset", "TEST accuracy vs. timing jitter",
                False)


def fig18_robustness_background() -> None:
    """Accuracy under additive background (was fig7 panel b, log x)."""
    _robustness("poisson_hz", "background rate (Hz per channel)",
                "fig18_robustness_background",
                "TEST accuracy vs. additive background", True)


def fig19_robustness_dropout() -> None:
    """Accuracy under event dropout (was fig7 panel c)."""
    _robustness("dropout_p", "event dropout probability",
                "fig19_robustness_dropout",
                "TEST accuracy vs. event dropout", False)


# ------------------------------------------------------------- fig20 --------
def fig20_per_class_matrix() -> None:
    """Per-class accuracy of the key runs (was fig S1)."""
    rows, mats = [], []
    for tag, label in PERCLASS_RUNS:
        p = RES / tag / "test_predictions.npz"
        if not p.exists():
            print(f"[panels] fig20 skip {tag} (no test_predictions.npz)")
            continue
        with np.load(p) as z:
            mats.append(z["per_class_accuracy"] * 100)
        rows.append(label)
    if not mats:
        return
    mat = np.vstack(mats)

    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    im = ax.imshow(mat, cmap="viridis", vmin=40, vmax=100, aspect="auto")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows, fontsize=6.2)
    ax.set_xticks(np.arange(20))
    ax.set_xticklabels([str(i) for i in range(20)], fontsize=6)
    ax.set_xlabel("class")
    title(ax, "Per-class TEST accuracy")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center",
                    fontsize=4.6, color="white" if mat[i, j] < 75 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.024, pad=0.02)
    cb.set_label("accuracy (%)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    fig.tight_layout()
    save(fig, "fig20_per_class_matrix")


# ------------------------------------------------------------- fig21 --------
def fig21_v31_params() -> None:
    """Diagonal emission versus full matrix (was fig S2)."""
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
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
    title(ax, "Diagonal emission: half the parameters")
    fig.tight_layout()
    save(fig, "fig21_v31_params")


# ------------------------------------------------------------- fig22 --------
def fig22_mean_counts() -> None:
    """Mean spike counts per class and neuron (was fig S3 panel a)."""
    cap = capture()
    counts, labels = cap["counts"], cap["labels"]
    per_class = np.stack([counts[labels == c].mean(0) for c in range(20)])
    order = np.argsort(per_class.mean(0))

    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    im = ax.imshow(per_class[:, order], aspect="auto", cmap="magma",
                   interpolation="nearest")
    ax.set_xticks([0, 25, 50, 75])
    ax.set_yticks([0, 4, 8, 12, 16])
    ax.set_xlabel("neuron (sorted by mean counts)")
    ax.set_ylabel("class")
    title(ax, "Mean spike counts by class")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("spikes / 500 bins", fontsize=5.6)
    cb.ax.tick_params(labelsize=5.4)
    fig.tight_layout()
    save(fig, "fig22_mean_counts")


# ------------------------------------------------------------- fig23 --------
def fig23_sample_counts() -> None:
    """One sample per class (was fig S3 panel b)."""
    cap = capture()
    counts, labels = cap["counts"], cap["labels"]

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for c, col in zip([0, 3, 7], [C_VEC, C_BAD, C_ACC]):
        ax.plot(counts[labels == c][:1].squeeze(0), color=col, lw=0.9,
                label=f"class {c}")
    ax.set_xlabel("neuron")
    ax.set_ylabel("spike count")
    ax.legend(frameon=False)
    title(ax, "One sample per class")
    fig.tight_layout()
    save(fig, "fig23_sample_counts")


# ------------------------------------------------------------- fig24 --------
def fig24_cost_accuracy() -> None:
    """Accuracy against the wall-clock budget (was fig S4 panel a)."""
    runs = [("v3_vector_scalar_d1", "scalar $D{=}1$", C_SCALAR),
            ("v3_vector_d1000", "$N{=}64$, $D{=}1000$", C_VEC),
            ("v3_n100_baseline_ep5", "$N{=}100$, 5 ep", C_GREY),
            ("v3_n100_aug_dropout_ep20", "$N{=}100$ + dropout", C_VEC),
            ("v3_n100_spec_ep30", "final model, 30 ep", C_ACC)]

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for tag, label, col in runs:
        r = load_run(tag)
        hours = r["wall_seconds"] / 3600.0
        ax.plot(hours, 100.0 * r["test_accuracy"], "o", color=col, ms=4.2)
        last = tag == "v3_n100_spec_ep30"
        ax.annotate(label, (hours, 100.0 * r["test_accuracy"]),
                    textcoords="offset points",
                    xytext=(-5, 9) if last else (4, -7),
                    ha="right" if last else "left", fontsize=5.6, color=col)
    ax.set_xlabel("training wall-clock (h)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_xlim(0, 5.5)
    ax.set_ylim(40, 92)
    title(ax, "Accuracy vs. training budget")
    fig.tight_layout()
    save(fig, "fig24_cost_accuracy")


# ------------------------------------------------------------- fig25 --------
def fig25_cost_per_epoch() -> None:
    """Per-epoch time and peak VRAM (was fig S4 panel b)."""
    tags = ["v3_vector_scalar_d1", "v3_vector_d1000", "v3_n100_spec_ep30"]
    names = ["scalar\n$D{=}1$", "$N{=}64$\n$D{=}1000$", "final\n$N{=}100$"]
    epoch_s, vram = [], []
    for tag in tags:
        r = load_run(tag)
        epoch_s.append(r["epoch_seconds"])
        vram.append(r["vram_gib"])

    fig, ax = plt.subplots(figsize=(3.8, 2.9))
    x = np.arange(len(tags))
    w = 0.38
    ax.bar(x - w / 2, epoch_s, w, color=C_GREY)
    ax.set_ylabel("s / epoch")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, max(epoch_s) * 1.45)
    for xi, v in zip(x - w / 2, epoch_s):
        ax.text(xi, v / 2, f"{v:.0f}", ha="center", va="center",
                fontsize=5.8, color="white")
    ax2 = ax.twinx()
    ax2.bar(x + w / 2, vram, w, color=C_VEC, alpha=0.55)
    ax2.set_ylabel("peak VRAM (GiB)", color=C_VEC)
    ax2.tick_params(axis="y", colors=C_VEC)
    ax2.set_ylim(0, 7.2)
    budget = ax2.axhline(4.9, color=C_BAD, lw=0.7, ls="--",
                         label="4.9 GiB budget")
    for xi, v in zip(x + w / 2, vram):
        ax2.text(xi, v + 0.30, f"{v:.1f}", ha="center", fontsize=5.8,
                 color=C_VEC)
    ax.legend(handles=[Patch(facecolor=C_GREY, label="s / epoch"),
                       Patch(facecolor=C_VEC, alpha=0.55,
                             label="peak VRAM (GiB)"),
                       budget],
              frameon=False, loc="upper left", fontsize=5.8,
              handlelength=1.1, borderpad=0.2)
    title(ax, "Per-epoch cost and peak VRAM")
    fig.tight_layout()
    save(fig, "fig25_cost_per_epoch")


# ---------------------------------------------------------------- main ------
PANELS = [
    ("fig01_schematic", fig01_schematic),
    ("fig02_single_trial_states", fig02_single_trial_states),
    ("fig03_width_sweep", fig03_width_sweep),
    ("fig04_controls", fig04_controls),
    ("fig05_training_curves", fig05_training_curves),
    ("fig06_seen_unseen", fig06_seen_unseen),
    ("fig07_confusion", fig07_confusion),
    ("fig08_probe", fig08_probe),
    ("fig09_effective_dim", fig09_effective_dim),
    ("fig10_frozen_dynamics", fig10_frozen_dynamics),
    ("fig11_mi_distribution", fig11_mi_distribution),
    ("fig12_knockout_drop", fig12_knockout_drop),
    ("fig13_cumulative_topk", fig13_cumulative_topk),
    ("fig14_aug_single_ops", fig14_aug_single_ops),
    ("fig15_aug_dropout_plateau", fig15_aug_dropout_plateau),
    ("fig16_aug_interaction", fig16_aug_interaction),
    ("fig17_robustness_offset", fig17_robustness_offset),
    ("fig18_robustness_background", fig18_robustness_background),
    ("fig19_robustness_dropout", fig19_robustness_dropout),
    ("fig20_per_class_matrix", fig20_per_class_matrix),
    ("fig21_v31_params", fig21_v31_params),
    ("fig22_mean_counts", fig22_mean_counts),
    ("fig23_sample_counts", fig23_sample_counts),
    ("fig24_cost_accuracy", fig24_cost_accuracy),
    ("fig25_cost_per_epoch", fig25_cost_per_epoch),
]


def resolve(wanted: list, names: list) -> list:
    """Accept full panel names or unambiguous prefixes."""
    chosen = []
    for w in wanted:
        if w in names:
            chosen.append(w)
            continue
        hits = [n for n in names if n.startswith(w)]
        if len(hits) == 1:
            chosen.append(hits[0])
        elif not hits:
            raise SystemExit(f"unknown panel {w!r}\nknown: {', '.join(names)}")
        else:
            raise SystemExit(f"ambiguous panel {w!r}: {', '.join(hits)}")
    return list(dict.fromkeys(chosen))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None,
                    help="build only these panels, e.g. --only fig03 fig07 "
                         "(unique prefixes work)")
    args = ap.parse_args()
    style()
    names = [n for n, _ in PANELS]
    funcs = dict(PANELS)
    chosen = names if not args.only else resolve(args.only, names)
    for name in chosen:
        funcs[name]()


if __name__ == "__main__":
    main()
