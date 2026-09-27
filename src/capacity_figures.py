"""The three figures of the neuron-vector capacity evaluation.

Descriptive only: the inferential content lives in the Mantel statistics, their permutation
nulls, the rate controls, the cross-validated prediction and the seed variability computed in
:mod:`src.vector_capacity`; the figures merely display those numbers.

1. primary functional similarity vs total dimension (structured-only vs structured+residual)
2. primary vs class-rate functional correspondence across conditions (with controls)
3. neuron-neuron distance relationship for one representative condition
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

DEFAULT_FORMATS = ("png", "pdf")
DEFAULT_DPI = 200

_KIND_COLORS = {
    "structured": "#1f77b4",
    "structured_plus_residual": "#d62728",
    "control": "#7f7f7f",
}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _finish(fig, path_stem: Path, *, formats: Sequence[str], dpi: int) -> list[Path]:
    out: list[Path] = []
    for fmt in formats:
        p = path_stem.with_suffix(f".{fmt}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        out.append(p)
    return out


def _clean(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _finite(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _conclusion_note(fig, text: str = "descriptive only - no ranking is implied") -> None:
    fig.text(0.5, -0.02, text, ha="center", va="top", fontsize=8, color="#555555")


# --------------------------------------------------------------------------
# Figure 1
# --------------------------------------------------------------------------
def figure1_capacity_vs_dimension(
    payload: Mapping[str, Any],
    out_dir: str | Path,
    *,
    metric: str = "primary_metric_value",
    formats: Sequence[str] = DEFAULT_FORMATS,
    dpi: int = DEFAULT_DPI,
) -> list[Path]:
    """Primary functional similarity vs total dimension, structured vs structured+residual."""
    plt = _plt()
    rows = [row for row in payload.get("rows", []) if row.get("representation_kind") != "control"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    structured = [r for r in rows if r.get("representation_kind") == "structured"]
    residual = [r for r in rows if r.get("representation_kind") == "structured_plus_residual"]

    if structured:
        xs = [int(r["total_d"]) for r in structured]
        ys = [_finite(r.get(metric)) for r in structured]
        order = np.argsort(xs)
        ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], "o-", color=_KIND_COLORS["structured"],
                label="structured only", lw=1.5, ms=6)
    if residual:
        families: dict[tuple[int, int], list[float]] = {}
        dims: dict[tuple[int, int], int] = {}
        for row in residual:
            key = (int(row["structured_d"]), int(row["residual_d"]))
            families.setdefault(key, []).append(_finite(row.get(metric)))
            dims[key] = int(row["total_d"])
        keys = sorted(families, key=lambda k: dims[k])
        xs = [dims[k] for k in keys]
        means = [float(np.nanmean(families[k])) for k in keys]
        stds = [float(np.nanstd(families[k])) for k in keys]
        ax.errorbar(xs, means, yerr=stds, fmt="s--", color=_KIND_COLORS["structured_plus_residual"],
                    label="structured + residual (mean +/- SD over seeds)", lw=1.5, ms=6, capsize=3)
        for k, x in zip(keys, xs):
            ax.plot([x] * len(families[k]), families[k], ".", color=_KIND_COLORS["structured_plus_residual"], ms=4)

    controls = payload.get("controls", [])
    for row in controls:
        if row.get("representation") == "control_rate_only":
            value = _finite(row.get(metric))
            if np.isfinite(value):
                ax.axhline(value, color=_KIND_COLORS["control"], ls=":", lw=1.2,
                           label="rate-only control")

    ax.set_xlabel("total representation dimension d")
    ax.set_ylabel("Mantel Spearman r (primary target)")
    ax.set_title("Primary functional similarity vs representation capacity")
    ax.legend(fontsize=8, frameon=False)
    _clean(ax)
    _conclusion_note(fig)
    return _finish(fig, out_dir / "figure1_capacity_vs_dimension", formats=formats, dpi=dpi)


# --------------------------------------------------------------------------
# Figure 2
# --------------------------------------------------------------------------
def figure2_primary_vs_class_rate(
    payload: Mapping[str, Any],
    out_dir: str | Path,
    *,
    formats: Sequence[str] = DEFAULT_FORMATS,
    dpi: int = DEFAULT_DPI,
) -> list[Path]:
    """Primary (individual-stimulus) vs secondary (class-rate) correspondence, per condition."""
    plt = _plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    for row in payload.get("rows", []):
        kind = row.get("representation_kind")
        x = _finite(row.get("class_rate_metric_value"))
        y = _finite(row.get("primary_metric_value"))
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        marker = "s" if kind == "structured_plus_residual" else "o"
        ax.scatter([x], [y], marker=marker, s=46, color=_KIND_COLORS.get(kind, "#333333"),
                   alpha=0.85, label=kind if kind else None)
        ax.annotate(str(row.get("representation")), (x, y), fontsize=6, xytext=(3, 3),
                    textcoords="offset points", color="#444444")
    for row in payload.get("controls", []):
        x = _finite(row.get("class_rate_metric_value"))
        y = _finite(row.get("primary_metric_value"))
        if np.isfinite(x) and np.isfinite(y):
            ax.scatter([x], [y], marker="x", s=42, color=_KIND_COLORS["control"], alpha=0.9)
            ax.annotate(str(row.get("representation")), (x, y), fontsize=6, xytext=(3, 3),
                        textcoords="offset points", color="#444444")

    handles, labels = ax.get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    ax.legend(unique.values(), unique.keys(), fontsize=8, frameon=False)
    ax.set_xlabel("Mantel Spearman r (class-rate target)")
    ax.set_ylabel("Mantel Spearman r (individual-stimulus target)")
    ax.set_title("Primary vs secondary functional correspondence")
    _clean(ax)
    _conclusion_note(fig)
    return _finish(fig, out_dir / "figure2_primary_vs_class_rate", formats=formats, dpi=dpi)


# --------------------------------------------------------------------------
# Figure 3
# --------------------------------------------------------------------------
def figure3_distance_relationship(
    payload: Mapping[str, Any],
    out_dir: str | Path,
    *,
    condition: str = "structured_48",
    formats: Sequence[str] = DEFAULT_FORMATS,
    dpi: int = DEFAULT_DPI,
) -> list[Path]:
    """Representation distance vs functional distance for one representative condition."""
    plt = _plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    curves = payload.get("curves", {})
    if condition not in curves:
        raise KeyError(f"no distance curve stored for condition {condition!r}; available: {list(curves)}")
    curve = curves[condition]

    centers = np.asarray(curve["bin_center"], dtype=float)
    means = np.asarray(curve["bin_mean"], dtype=float)
    sems = np.asarray(curve["bin_sem"], dtype=float)
    counts = np.asarray(curve["bin_count"], dtype=float)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax.errorbar(centers, means, yerr=sems, fmt="o-", color="#1f77b4", ms=4, lw=1.4, capsize=2)
    ax.set_xlabel("representation-space distance (standardised vectors, Euclidean)")
    ax.set_ylabel("mean functional distance\n(individual-stimulus profiles)")
    ax.set_title(f"Condition {condition}: distance relationship (quantile bins)")
    _clean(ax)

    ax2.scatter(np.asarray(curve["x"]), np.asarray(curve["y"]), s=2, alpha=0.15, color="#333333")
    ax2.set_xlabel("representation distance")
    ax2.set_ylabel("functional distance")
    ax2.set_title("pair cloud (subsampled)")
    _clean(ax2)
    ax2.text(0.98, 0.02, f"pairs per bin: {int(counts.min())}-{int(counts.max())}", ha="right",
             va="bottom", transform=ax2.transAxes, fontsize=7, color="#666666")
    fig.suptitle("Neuron-neuron distance relationship (representative condition, descriptive)", fontsize=10)
    return _finish(fig, out_dir / "figure3_distance_relationship", formats=formats, dpi=dpi)


def write_all_figures(
    payload: Mapping[str, Any],
    out_dir: str | Path,
    *,
    curve_condition: str = "structured_48",
    formats: Sequence[str] = DEFAULT_FORMATS,
    dpi: int = DEFAULT_DPI,
) -> dict[str, list[str]]:
    """Render the three figures and return their paths by figure name."""
    written: dict[str, list[str]] = {}
    written["figure1_capacity_vs_dimension"] = [
        str(p) for p in figure1_capacity_vs_dimension(payload, out_dir, formats=formats, dpi=dpi)
    ]
    written["figure2_primary_vs_class_rate"] = [
        str(p) for p in figure2_primary_vs_class_rate(payload, out_dir, formats=formats, dpi=dpi)
    ]
    try:
        written["figure3_distance_relationship"] = [
            str(p) for p in figure3_distance_relationship(
                payload, out_dir, condition=curve_condition, formats=formats, dpi=dpi
            )
        ]
    except KeyError:
        written["figure3_distance_relationship"] = []
    return written


__all__ = [
    "DEFAULT_FORMATS",
    "DEFAULT_DPI",
    "figure1_capacity_vs_dimension",
    "figure2_primary_vs_class_rate",
    "figure3_distance_relationship",
    "write_all_figures",
]