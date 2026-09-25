"""Publication-style figures for the neuron-as-vector study.

Each ``plot_*`` function draws exactly one of the project's figures and saves it
in every configured format (PNG + PDF by default). All functions are pure with
respect to their inputs: they take already-computed numbers (from the analysis
scripts) and never run the model, download data, or touch the network. This keeps
figure generation cheap, deterministic and decoupled from training.

The figures, and the question each answers, are:

======  ===================================================================
Fig 1   Does the network actually learn? (training/validation curves)
Fig 2   Where do the hidden neurons live in representation space? (PCA)
Fig 3   Is representation distance related to functional distance? (primary)
Fig 4   Are representation *neighbours* functionally similar? (kNN)
Fig 5   Which representation content matters, and do controls stay null?
Fig 6   What does the functional fingerprint look like? (class x neuron map)
======  ===================================================================

Matplotlib is forced to the non-interactive ``Agg`` backend so figures can be
produced on a headless machine / in CI.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")  # noqa: E402  (must precede pyplot import)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

# A small, readable, colour-blind-safe palette reused across figures.
PALETTE = {
    "blue": "#3b6fb0",
    "orange": "#e08a2c",
    "green": "#4c9a52",
    "red": "#c0504d",
    "purple": "#8064a2",
    "grey": "#8c8c8c",
    "dark": "#2b2b2b",
}

DEFAULT_DPI = 200
DEFAULT_FORMATS = ("png", "pdf")


# --------------------------------------------------------------------------
# Saving helpers
# --------------------------------------------------------------------------
def _finish(fig: Figure, path: str | Path, *, formats: Sequence[str] = DEFAULT_FORMATS, dpi: int = DEFAULT_DPI) -> list[Path]:
    """Save ``fig`` next to ``path`` in every requested format and close it."""
    base = Path(path)
    base.parent.mkdir(parents=True, exist_ok=True)
    # Strip an existing suffix so a caller may pass e.g. "figure.png" safely.
    stem = base.with_suffix("")
    written: list[Path] = []
    for fmt in formats:
        out = stem.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        written.append(out)
    plt.close(fig)
    return written


def _clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3)


# --------------------------------------------------------------------------
# Figure 1 - training curves
# --------------------------------------------------------------------------
def plot_training_curves(
    history: Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    title: str = "Training dynamics",
    formats: Sequence[str] = DEFAULT_FORMATS,
    dpi: int = DEFAULT_DPI,
) -> list[Path]:
    """Loss and accuracy against epoch, train vs validation, plus firing rate.

    ``history`` is the list of per-epoch records produced by
    :func:`src.training.train_model` (or the ``history.jsonl``/CSV saved by the
    training script).
    """
    if not history:
        raise ValueError("history is empty; nothing to plot")
    epochs = [int(r.get("epoch", i)) for i, r in enumerate(history)]
    tr_loss = [float(r.get("train_loss", np.nan)) for r in history]
    va_loss = [float(r.get("val_loss", np.nan)) for r in history]
    tr_acc = [float(r.get("train_accuracy", np.nan)) for r in history]
    va_acc = [float(r.get("val_accuracy", np.nan)) for r in history]
    rate = [float(r.get("val_hidden_rate_hz", np.nan)) for r in history]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    ax = axes[0]
    ax.plot(epochs, tr_loss, color=PALETTE["blue"], label="train")
    ax.plot(epochs, va_loss, color=PALETTE["orange"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy")
    ax.set_title("Loss")
    ax.legend(frameon=False)
    _clean_axes(ax)

    ax = axes[1]
    ax.plot(epochs, tr_acc, color=PALETTE["blue"], label="train")
    ax.plot(epochs, va_acc, color=PALETTE["orange"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy")
    _clean_axes(ax)

    ax = axes[2]
    ax.plot(epochs, rate, color=PALETTE["green"])
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean hidden rate (Hz)")
    ax.set_title("Hidden firing rate")
    _clean_axes(ax)

    fig.suptitle(title)
    fig.tight_layout()
    return _finish(fig, path, formats=formats, dpi=dpi)


# --------------------------------------------------------------------------
# Figure 2 - representation-space geometry (PCA)
# --------------------------------------------------------------------------
def plot_representation_pca(
    coords: np.ndarray,
    path: str | Path,
    *,
    values: np.ndarray | None = None,
    value_label: str = "firing rate (Hz)",
    explained: Sequence[float] | None = None,
    labels: np.ndarray | None = None,
    title: str = "Hidden neurons in representation space",
    formats: Sequence[str] = DEFAULT_FORMATS,
    dpi: int = DEFAULT_DPI,
) -> list[Path]:
    """2-D PCA scatter of the neuron representation space.

    ``coords`` is ``(n_neurons, 2)`` (from :meth:`RepresentationSpace.pca`).
    ``values`` optionally colour the points by a scalar per neuron (e.g. firing
    rate, or a "preferred class" index via ``labels``).
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("coords must be (n_neurons, >=2)")

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    if labels is not None:
        labels = np.asarray(labels)
        for cls in np.unique(labels):
            sel = labels == cls
            ax.scatter(coords[sel, 0], coords[sel, 1], s=18, alpha=0.85, label=f"class {int(cls)}")
        ax.legend(frameon=False, fontsize=7, ncols=2)
    elif values is not None:
        values = np.asarray(values, dtype=np.float64)
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, s=20, cmap="viridis")
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(value_label)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=18, color=PALETTE["blue"], alpha=0.85)

    if explained is not None and len(explained) >= 2:
        ax.set_xlabel(f"PC1 ({100 * explained[0]:.0f}% var)")
        ax.set_ylabel(f"PC2 ({100 * explained[1]:.0f}% var)")
    else:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    ax.set_title(title)
    _clean_axes(ax)
    fig.tight_layout()
    return _finish(fig, path, formats=formats, dpi=dpi)


# --------------------------------------------------------------------------
# Figure 3 - representation distance vs functional distance (primary result)
# --------------------------------------------------------------------------
def plot_distance_correlation(
    dx: np.ndarray,
    dy: np.ndarray,
    path: str | Path,
    *,
    mantel: Mapping[str, Any] | None = None,
    curve: Mapping[str, Any] | None = None,
    null: np.ndarray | None = None,
    title: str = "Representation distance vs functional distance",
    formats: Sequence[str] = DEFAULT_FORMATS,
    dpi: int = DEFAULT_DPI,
) -> list[Path]:
    """Scatter of pair distances with a binned trend and optional Mantel null.

    Left panel: representation-space distance (x) against functional-fingerprint
    distance (y) for every neuron pair, with a quantile-binned mean +/- SEM.
    Right panel: the permutation null of the primary statistic with the observed
    value marked, when ``null`` is supplied.
    """
    dx = np.asarray(dx, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)

    has_null = null is not None and np.asarray(null).size > 0
    if has_null:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
        ax, ax_null = axes
    else:
        fig, ax = plt.subplots(figsize=(5.6, 4.6))
        ax_null = None

    ax.scatter(dx, dy, s=4, alpha=0.15, color=PALETTE["grey"], rasterized=True)
    if curve is not None and np.asarray(curve.get("bin_center", [])).size:
        c = np.asarray(curve["bin_center"], dtype=np.float64)
        m = np.asarray(curve["bin_mean"], dtype=np.float64)
        s = np.asarray(curve["bin_sem"], dtype=np.float64)
        ax.errorbar(c, m, yerr=s, color=PALETTE["red"], marker="o", ms=4, lw=1.4, capsize=2, label="binned mean")
        ax.legend(frameon=False)
    ax.set_xlabel("representation distance")
    ax.set_ylabel("functional distance")
    ax.set_title("Per-pair distances")
    _clean_axes(ax)

    if ax_null is not None:
        null = np.asarray(null, dtype=np.float64)
        ax_null.hist(null, bins=40, color=PALETTE["blue"], alpha=0.7)
        if mantel is not None and mantel.get("statistic") is not None:
            ax_null.axvline(
                float(mantel["statistic"]), color=PALETTE["red"], lw=2,
                label=f"observed r={float(mantel['statistic']):.2f}",
            )
            ax_null.legend(frameon=False)
        if mantel is not None and mantel.get("p_value") is not None:
            ax_null.set_title(f"Mantel null (p={float(mantel['p_value']):.3g})")
        else:
            ax_null.set_title("Mantel null")
        ax_null.set_xlabel("null statistic")
        ax_null.set_ylabel("count")
        _clean_axes(ax_null)

    fig.suptitle(title)
    fig.tight_layout()
    return _finish(fig, path, formats=formats, dpi=dpi)


# --------------------------------------------------------------------------
# Figure 4 - kNN functional similarity
# --------------------------------------------------------------------------
def plot_knn_effect(
    knn_rows: Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    title: str = "Functional similarity of representation neighbours",
    formats: Sequence[str] = DEFAULT_FORMATS,
    dpi: int = DEFAULT_DPI,
) -> list[Path]:
    """Effect size (z) and significance of the k-nearest-neighbour analysis.

    A *negative* z means neighbours are functionally more similar than chance
    (the hypothesis of interest), so the sign convention is stated on the axis.
    """
    rows = [r for r in knn_rows if r.get("k") is not None]
    if not rows:
        raise ValueError("no kNN rows to plot")
    rows = sorted(rows, key=lambda r: int(r["k"]))
    ks = [int(r["k"]) for r in rows]
    z = [float(r.get("effect_size_z", np.nan)) for r in rows]
    p = [float(r.get("p_value", np.nan)) for r in rows]

    fig, ax = plt.subplots(figsize=(5.6, 4.3))
    bars = ax.bar([str(k) for k in ks], z, color=[PALETTE["green"] if val < 0 else PALETTE["grey"] for val in z])
    ax.axhline(0.0, color=PALETTE["dark"], lw=0.8)
    ax.set_xlabel("k (number of neighbours)")
    ax.set_ylabel("effect size z  (negative = neighbours more similar)")
    for bar, pv in zip(bars, p):
        if np.isfinite(pv):
            star = "*" if pv < 0.05 else ""
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{star}\np={pv:.2g}",
                    ha="center", va="bottom" if bar.get_height() >= 0 else "top", fontsize=7)
    ax.set_title(title)
    _clean_axes(ax)
    fig.tight_layout()
    return _finish(fig, path, formats=formats, dpi=dpi)


# --------------------------------------------------------------------------
# Figure 5 - ablation / controls summary
# --------------------------------------------------------------------------
def plot_ablation_summary(
    rows: Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    r_key: str = "primary_metric_mantel_spearman_r",
    p_key: str = "primary_metric_p_value",
    name_key: str = "representation",
    title: str = "Representation content vs geometry-function correlation",
    formats: Sequence[str] = DEFAULT_FORMATS,
    dpi: int = DEFAULT_DPI,
) -> list[Path]:
    """Horizontal bars of the primary statistic for each representation variant.

    Control variants (``is_control`` truthy) and the deliberately circular
    ``fingerprint`` variant are drawn in grey / hatched so they cannot be mistaken
    for evidence.
    """
    rows = [r for r in rows if r.get(name_key) is not None]
    if not rows:
        raise ValueError("no ablation rows to plot")
    rows = sorted(rows, key=lambda r: (r.get(r_key) is None, r.get(r_key) if r.get(r_key) is not None else 0.0))

    names = [str(r[name_key]) for r in rows]
    vals = [float(r[r_key]) if r.get(r_key) is not None else np.nan for r in rows]
    ps = [float(r.get(p_key, np.nan)) for r in rows]

    colors = []
    for r in rows:
        if r.get("circular__do_not_report_as_evidence"):
            colors.append(PALETTE["purple"])
        elif r.get("is_control"):
            colors.append(PALETTE["grey"])
        else:
            colors.append(PALETTE["blue"])

    fig, ax = plt.subplots(figsize=(7.4, max(3.0, 0.42 * len(rows) + 1.6)))
    y = np.arange(len(rows))
    ax.barh(y, vals, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0.0, color=PALETTE["dark"], lw=0.8)
    for yi, (v, pv) in enumerate(zip(vals, ps)):
        if np.isfinite(v):
            star = "*" if np.isfinite(pv) and pv < 0.05 else ""
            ax.text(v, yi, f" {v:.2f}{star}", va="center", fontsize=7)
    ax.set_xlabel("primary Mantel Spearman r  (* p<0.05)")
    ax.set_title(title)
    _clean_axes(ax)
    fig.tight_layout()
    return _finish(fig, path, formats=formats, dpi=dpi)


# --------------------------------------------------------------------------
# Figure 6 - functional fingerprint heat map
# --------------------------------------------------------------------------
def plot_fingerprint_heatmap(
    rate_matrix: np.ndarray,
    path: str | Path,
    *,
    class_names: Sequence[str] | None = None,
    neuron_order: np.ndarray | None = None,
    reliability: Mapping[str, Any] | None = None,
    title: str = "Functional fingerprint (class-conditioned rate)",
    formats: Sequence[str] = DEFAULT_FORMATS,
    dpi: int = DEFAULT_DPI,
) -> list[Path]:
    """Class x neuron mean firing-rate map, neurons optionally reordered.

    Reordering neurons (e.g. by preferred class then rate) makes the block
    structure of the fingerprint visible, which is what the geometry analysis
    tries to recover from the *representation* alone.
    """
    R = np.asarray(rate_matrix, dtype=np.float64)
    if R.ndim != 2:
        raise ValueError("rate_matrix must be (n_classes, n_hidden)")
    if neuron_order is not None:
        R = R[:, np.asarray(neuron_order, dtype=np.int64)]

    fig, ax = plt.subplots(figsize=(9, 3.6))
    im = ax.imshow(R, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_xlabel("hidden neuron (reordered)")
    ax.set_ylabel("class")
    if class_names is not None:
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_yticklabels([str(c) for c in class_names], fontsize=7)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("mean rate (Hz)")
    if reliability is not None and reliability.get("matrix_reliability_spearman") is not None:
        ax.set_title(f"{title}\nnoise ceiling (matrix reliability) r={float(reliability['matrix_reliability_spearman']):.2f}")
    else:
        ax.set_title(title)
    fig.tight_layout()
    return _finish(fig, path, formats=formats, dpi=dpi)


# --------------------------------------------------------------------------
# Convenience: make every figure from a results bundle
# --------------------------------------------------------------------------
def make_all_figures(
    bundle: Mapping[str, Any],
    out_dir: str | Path,
    *,
    formats: Sequence[str] = DEFAULT_FORMATS,
    dpi: int = DEFAULT_DPI,
) -> list[Path]:
    """Render every figure that the supplied result bundle has the data for.

    ``bundle`` is the structure saved by ``scripts/run_geometry_analysis.py``
    (plus the training history from ``results/<tag>_history.json``). Missing
    sections are skipped rather than raising, so figures can be produced
    incrementally.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _try(fn, *args, name: str, **kwargs):
        with contextlib.suppress(Exception):
            written.extend(fn(*args, path=out_dir / name, formats=formats, dpi=dpi, **kwargs))

    history = bundle.get("history")
    if history:
        _try(plot_training_curves, history, name="figure1_training_curves")

    pca = bundle.get("pca")
    if pca is not None:
        coords = np.asarray(pca["coords"])
        _try(
            plot_representation_pca,
            coords,
            name="figure2_representation_pca",
            values=np.asarray(pca["values"]) if pca.get("values") is not None else None,
            value_label=str(pca.get("value_label", "value")),
            explained=pca.get("explained"),
        )

    primary = bundle.get("primary")
    if primary is not None:
        _try(
            plot_distance_correlation,
            np.asarray(primary["dx"]),
            np.asarray(primary["dy"]),
            name="figure3_distance_correlation",
            mantel=primary.get("mantel"),
            curve=primary.get("curve"),
            null=np.asarray(primary["null"]) if primary.get("null") is not None else None,
        )

    knn_rows = bundle.get("knn_rows")
    if knn_rows:
        _try(plot_knn_effect, knn_rows, name="figure4_knn_effect")

    ablation_rows = bundle.get("ablation_rows")
    if ablation_rows:
        _try(plot_ablation_summary, ablation_rows, name="figure5_ablation_summary")

    fingerprint = bundle.get("fingerprint")
    if fingerprint is not None:
        _try(
            plot_fingerprint_heatmap,
            np.asarray(fingerprint["rate_matrix"]),
            name="figure6_fingerprint_heatmap",
            class_names=fingerprint.get("class_names"),
            neuron_order=np.asarray(fingerprint["order"]) if fingerprint.get("order") is not None else None,
            reliability=fingerprint.get("reliability"),
        )

    return written
