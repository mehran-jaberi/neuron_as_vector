"""Figures for the V3 experiment.

Deliberately limited to the training / classification process:

* training curves (loss, accuracy, firing rate, epoch time)
* activity visualisation for representative SHD samples (input raster, spike
  raster, population rate over time)
* confusion matrix and test-set accuracy
* the vector-vs-scalar comparison bar chart

No PCA / UMAP / manifold / state-geometry analysis: that is not the question V3
asks.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C_VEC = "#1b6ca8"
C_SCA = "#b3541e"


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------- #
def plot_training(history: list[dict], path: str | Path, title: str = "V3 training") -> Path:
    ep = [h["epoch"] for h in history]
    has_val = "val_accuracy" in history[0]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    ax = axes[0, 0]
    ax.plot(ep, [h["loss"] for h in history], "o-", ms=3, color=C_VEC, label="train")
    if has_val:
        ax.plot(ep, [h["val_loss"] for h in history], "s-", ms=3, color=C_SCA, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (CE + rate reg)")
    ax.set_title("loss")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(ep, [h["accuracy"] for h in history], "o-", ms=3, color=C_VEC, label="train")
    if has_val:
        ax.plot(ep, [h["val_accuracy"] for h in history], "s-", ms=3, color=C_SCA, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("accuracy")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(ep, [h["spikes_per_sample"] for h in history], "o-", ms=3, color="#444")
    ax.set_xlabel("epoch")
    ax.set_ylabel("spikes per sample")
    ax.set_title("population activity")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(ep, [h["time_s"] for h in history], "o-", ms=3, color="#666")
    ax.set_xlabel("epoch")
    ax.set_ylabel("seconds")
    ax.set_title("epoch wall time")
    ax.grid(alpha=0.3)

    fig.suptitle(title)
    return _save(fig, path)


# ---------------------------------------------------------------------- #
def plot_sample_activity(
    x_sample: np.ndarray,
    spikes_sample: np.ndarray,
    path: str | Path,
    bin_ms: float,
    title: str = "representative SHD sample",
) -> Path:
    """``x_sample`` ``(T, C)`` binned input, ``spikes_sample`` ``(T, N)`` spikes."""
    T, C = x_sample.shape
    t = np.arange(T) * bin_ms / 1000.0
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)

    ax = axes[0]
    ti, ci = np.nonzero(x_sample)
    ax.scatter(t[ti], ci, s=1.0, color="#2b5d8a", marker=".")
    ax.set_ylabel("input channel")
    ax.set_title(f"input events ({int(x_sample.sum())} active cells)")

    ax = axes[1]
    ti2, ni2 = np.nonzero(spikes_sample)
    ax.scatter(t[ti2], ni2, s=2.0, color="#a33", marker=".")
    ax.set_ylabel("neuron")
    ax.set_title(f"population spike raster ({int(spikes_sample.sum())} spikes)")

    ax = axes[2]
    rate = spikes_sample.reshape(T, -1).mean(axis=1) / (bin_ms / 1000.0)
    ax.plot(t, rate, color="#1b6ca8")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("mean rate [Hz]")
    ax.set_title("population firing rate over time")

    fig.suptitle(title)
    return _save(fig, path)


# ---------------------------------------------------------------------- #
def plot_confusion(cm: np.ndarray, path: str | Path, accuracy: float, title: str = "test") -> Path:
    k = cm.shape[0]
    off = cm.sum() - np.trace(cm)
    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(1, cm.max()))
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels([str(i) for i in range(k)], fontsize=7)
    ax.set_yticklabels([str(i) for i in range(k)], fontsize=7)
    ax.set_title(
        f"{title} confusion matrix - accuracy {accuracy * 100:.2f}%  ({off}/{cm.sum()} errors)"
    )
    for i in range(k):
        for j in range(k):
            v = int(cm[i, j])
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=6,
                        color="white" if v > 0.55 * cm.max() else "black")
    fig.colorbar(im, ax=ax, shrink=0.85)
    return _save(fig, path)


# ---------------------------------------------------------------------- #
def plot_per_class(per_class: np.ndarray, path: str | Path, title: str = "test") -> Path:
    k = per_class.size
    fig, ax = plt.subplots(figsize=(9, 3.4))
    cols = ["#1b6ca8" if i < 10 else "#4a9c5d" for i in range(k)]
    ax.bar(range(k), per_class * 100, color=cols)
    ax.set_xticks(range(k))
    ax.set_xticklabels([str(i) for i in range(k)], fontsize=8)
    ax.set_ylabel("accuracy [%]")
    ax.set_xlabel("class (0-9 English digits, 10-19 German digits)")
    ax.set_ylim(0, 100)
    ax.axhline(100 * per_class.mean(), color="#a33", ls="--", lw=1,
               label=f"mean {100 * per_class.mean():.1f}%")
    ax.set_title(f"{title} per-class accuracy")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)


# ---------------------------------------------------------------------- #
def plot_comparison(rows: list[dict], path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    names = [r["label"] for r in rows]
    acc = [100.0 * r["test_accuracy"] for r in rows]
    cols = [C_VEC if "D=" in n and "D=1" not in n else C_SCA for n in names]
    bars = ax.bar(names, acc, color=cols)
    for b, r in zip(bars, rows):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6,
                f"{100 * r['test_accuracy']:.2f}%\n({r['test_correct']}/{r['test_total']})",
                ha="center", fontsize=8)
    ax.set_ylabel("official SHD test accuracy [%]")
    ax.set_ylim(0, max(acc) * 1.30 + 1)
    ax.set_title("vector-valued neurons vs scalar baseline")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)


def plot_training_comparison(histories: dict[str, list[dict]], path: str | Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for name, hist in histories.items():
        ep = [h["epoch"] for h in hist]
        col = C_SCA if "D=1" in name else C_VEC
        axes[0].plot(ep, [h["loss"] for h in hist], label=name, color=col)
        axes[1].plot(ep, [h["accuracy"] for h in hist], label=name, color=col)
        if "val_accuracy" in hist[0]:
            axes[1].plot(ep, [h["val_accuracy"] for h in hist], ls="--", color=col, alpha=0.7)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("train loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy (solid train, dashed val)")
    axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("vector (D=1000) vs scalar (D=1) training")
    return _save(fig, path)


__all__ = [
    "plot_training",
    "plot_sample_activity",
    "plot_confusion",
    "plot_per_class",
    "plot_comparison",
    "plot_training_comparison",
]
