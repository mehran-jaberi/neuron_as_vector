"""Figures of the neuron-vector capacity evaluation (descriptive only).

The inferential content lives in the Mantel statistics, their permutation nulls, the rate
controls, the cross-validated prediction and the seed variability computed in
:mod:`src.vector_capacity` and :mod:`src.rate_robustness`; the figures merely display those
numbers.

First capacity study:

1. primary functional similarity vs total dimension (structured-only vs structured+residual)
2. primary vs class-rate functional correspondence across conditions (with controls)
3. neuron-neuron distance relationship for one representative condition

Rate-confound / robustness study:

4. rate decomposition of the target for one representative representation
5. representation comparison across the raw / neuron-centered / neuron z-scored targets
6. activity diagnostic (activity-only vs structural vs structural+activity)

No figure implies a ranking; annotations state the measured values.
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

#: Reporting order of the response-target variants (rate-robustness study).
TARGET_VARIANT_ORDER: tuple[str, ...] = (
    "raw",
    "neuron_centered",
    "neuron_zscored",
    "mean_rate",
    "column_standardised_then_neuron_centered",
    "row_l2_normalised",
)

_TARGET_ROLE_COLORS = {
    "main": "#1f77b4",
    "ordering_sensitivity": "#ff7f0e",
    "existing_control": "#7f7f7f",
}

#: Reporting order of the focused representation set (rate-robustness study).
REPRESENTATION_ORDER: tuple[str, ...] = (
    "structured_48",
    "structured_64",
    "structured_100",
    "full_48+16",
    "full_48+52",
    "activity_only",
    "activity_source",
    "structural_48_plus_activity",
)

_ACTIVITY_COLOR = "#2ca02c"



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


def _conclusion_note(fig, text: str = "descriptive only - no ranking is implied", *, y: float = -0.02) -> None:
    fig.text(0.5, y, text, ha="center", va="top", fontsize=8, color="#555555")


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
    """Render the first study's figures and return their paths by figure name."""
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


# --------------------------------------------------------------------------
# Rate-confound / robustness figures
# --------------------------------------------------------------------------
def _robustness_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        row for row in payload.get("rows", [])
        if row.get("representation_role") != "representation_control"
    ]


def _rate_role_color(role: Any) -> str:
    return _TARGET_ROLE_COLORS.get(str(role), "#333333")


def _variant_label(name: str) -> str:
    return {
        "raw": "raw (headline)",
        "neuron_centered": "neuron-centered",
        "neuron_zscored": "neuron z-scored",
        "mean_rate": "mean-rate target",
        "column_standardised_then_neuron_centered": "col-std then centered",
        "row_l2_normalised": "row-L2 (existing control)",
    }.get(name, name)


def rate_figure1_target_decomposition(
    payload: Mapping[str, Any],
    out_dir: str | Path,
    *,
    condition: str = "structured_48",
    formats: Sequence[str] = DEFAULT_FORMATS,
    dpi: int = DEFAULT_DPI,
) -> list[Path]:
    """Metric for one representation against every response-target variant (with CI)."""
    plt = _plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = {
        str(row["target_variant"]): row
        for row in payload.get("rows", [])
        if row.get("representation") == condition
    }
    order = [name for name in TARGET_VARIANT_ORDER if name in rows]
    if not order:
        raise KeyError(f"no target-variant rows stored for representation {condition!r}")

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ys = np.arange(len(order))[::-1]
    for y, name in zip(ys, order):
        row = rows[name]
        value = _finite(row.get("value"))
        low, high = _finite(row.get("bootstrap_low")), _finite(row.get("bootstrap_high"))
        color = _rate_role_color(row.get("target_role"))
        if np.isfinite(value) and np.isfinite(low) and np.isfinite(high):
            ax.errorbar([value], [y], xerr=[[value - low], [high - value]], fmt="o",
                        color=color, ms=6, capsize=3)
        else:
            ax.plot([value], [y], "o", color=color, ms=6)
        ax.annotate(f"{value:+.3f}" if np.isfinite(value) else "n/a", (value, y),
                    xytext=(6, 0), textcoords="offset points", fontsize=8, color="#444444")
    ax.axvline(0.0, color="#999999", lw=1.0, ls=":")
    ax.set_yticks(ys)
    ax.set_yticklabels([_variant_label(name) for name in order], fontsize=8)
    ax.set_xlabel("Mantel Spearman r vs representation geometry")
    ax.set_title(f"Target decomposition: {condition}", fontsize=10)
    _clean(ax)
    _conclusion_note(fig, "descriptive only - 95% neuron-bootstrap CI where computed")
    return _finish(fig, out_dir / "rate_figure1_target_decomposition", formats=formats, dpi=dpi)


def rate_figure2_representation_comparison(
    payload: Mapping[str, Any],
    out_dir: str | Path,
    *,
    variants: Sequence[str] = ("raw", "neuron_centered", "neuron_zscored"),
    formats: Sequence[str] = DEFAULT_FORMATS,
    dpi: int = DEFAULT_DPI,
) -> list[Path]:
    """Representations vs the raw / neuron-centered / neuron z-scored targets.

    Residual families are shown as the mean over seeds with the individual seeds as small
    markers, so seed variability is visible rather than averaged away.
    """
    plt = _plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _robustness_rows(payload)

    # group rows into (family, target) -> list of values, keeping the focus set
    families: dict[str, dict[str, list[float]]] = {}
    labels: dict[str, str] = {}
    for row in rows:
        representation = str(row.get("representation"))
        if representation.startswith("full_48+16_seed"):
            family = "full_48+16"
        elif representation.startswith("full_48+52_seed"):
            family = "full_48+52"
        elif representation.startswith("activity_source_"):
            family = "activity_source"
        else:
            family = representation
        labels.setdefault(family, family)
        target = str(row.get("target_variant"))
        value = _finite(row.get("value"))
        if not np.isfinite(value):
            continue
        families.setdefault(family, {}).setdefault(target, []).append(value)

    order = [name for name in REPRESENTATION_ORDER if name in families]
    order += [name for name in sorted(families) if name not in order]
    if not order:
        raise KeyError("no representation rows stored in the payload")

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    width = 0.8 / max(len(variants), 1)
    xs = np.arange(len(order))
    for index, variant in enumerate(variants):
        offsets = xs + (index - (len(variants) - 1) / 2.0) * width
        means, seeds = [], []
        for family in order:
            values = families.get(family, {}).get(variant, [])
            means.append(float(np.mean(values)) if values else float("nan"))
            seeds.append(values)
        color = _rate_role_color("main") if variant == "raw" else (
            "#ff7f0e" if variant == "neuron_centered" else "#2ca02c"
        )
        ax.bar(offsets, means, width=width, color=color, alpha=0.75,
               label=_variant_label(variant))
        for x, values in zip(offsets, seeds):
            if len(values) > 1:
                ax.plot([x] * len(values), values, "k.", ms=4, alpha=0.8)
    ax.axhline(0.0, color="#999999", lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(order, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Mantel Spearman r")
    ax.set_title("Representation comparison across target variants", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    _clean(ax)
    _conclusion_note(fig, "residual families: bar = mean over seeds, dots = individual seeds", y=-0.34)
    return _finish(fig, out_dir / "rate_figure2_representation_comparison", formats=formats, dpi=dpi)


def rate_figure3_activity_diagnostic(
    payload: Mapping[str, Any],
    out_dir: str | Path,
    *,
    representations: Sequence[str] = ("activity_only", "structured_48", "structural_48_plus_activity"),
    variants: Sequence[str] = ("raw", "neuron_centered", "neuron_zscored", "mean_rate"),
    formats: Sequence[str] = DEFAULT_FORMATS,
    dpi: int = DEFAULT_DPI,
) -> list[Path]:
    """Activity-containing representations vs the structural baseline, per target variant."""
    plt = _plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _robustness_rows(payload)

    values: dict[tuple[str, str], float] = {}
    for row in rows:
        representation = str(row.get("representation"))
        if representation not in representations:
            continue
        value = _finite(row.get("value"))
        if np.isfinite(value):
            values[(representation, str(row.get("target_variant")))] = value
    if not values:
        raise KeyError("no activity / structural rows stored in the payload")

    present = [name for name in representations if any(key[0] == name for key in values)]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    width = 0.8 / max(len(present), 1)
    xs = np.arange(len(variants))
    for index, representation in enumerate(present):
        offsets = xs + (index - (len(present) - 1) / 2.0) * width
        heights = [values.get((representation, variant), float("nan")) for variant in variants]
        color = _KIND_COLORS["structured"] if representation == "structured_48" else _ACTIVITY_COLOR
        ax.bar(offsets, heights, width=width, color=color, alpha=0.8, label=representation)
    ax.axhline(0.0, color="#999999", lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([_variant_label(variant) for variant in variants], fontsize=8)
    ax.set_ylabel("Mantel Spearman r")
    ax.set_title("Activity diagnostic: does the activity block explain the correspondence?", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    _clean(ax)
    _conclusion_note(fig)
    return _finish(fig, out_dir / "rate_figure3_activity_diagnostic", formats=formats, dpi=dpi)


def write_rate_robustness_figures(
    payload: Mapping[str, Any],
    out_dir: str | Path,
    *,
    condition: str = "structured_48",
    formats: Sequence[str] = DEFAULT_FORMATS,
    dpi: int = DEFAULT_DPI,
) -> dict[str, list[str]]:
    """Render the three rate-robustness figures and return their paths by name."""
    written: dict[str, list[str]] = {}
    written["rate_figure1_target_decomposition"] = [
        str(p) for p in rate_figure1_target_decomposition(
            payload, out_dir, condition=condition, formats=formats, dpi=dpi
        )
    ]
    written["rate_figure2_representation_comparison"] = [
        str(p) for p in rate_figure2_representation_comparison(payload, out_dir, formats=formats, dpi=dpi)
    ]
    written["rate_figure3_activity_diagnostic"] = [
        str(p) for p in rate_figure3_activity_diagnostic(payload, out_dir, formats=formats, dpi=dpi)
    ]
    return written


# --------------------------------------------------------------------------
# Source-extension figures (functional-response / temporal evaluation)
# --------------------------------------------------------------------------
_TARGET_DECOMPOSITION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("raw", "raw_r"),
    ("neuron-centered", "centered_r"),
    ("neuron z-scored", "zscored_r"),
    ("mean-rate target", "mean_rate_r"),
)

_ABLATION_COLORS = {
    "structured": "#1f77b4",
    "functional_response": "#d62728",
    "temporal": "#9467bd",
    "functional_response+temporal": "#2ca02c",
    "functional_response+temporal|block": "#8c564b",
    "structured_48": "#1f77b4",
    "structured_48_plus_temporal": "#9467bd",
}
_ABLATION_LABELS = {
    "structured": "A: structural",
    "functional_response": "B: + functional response",
    "temporal": "C: + temporal",
    "functional_response+temporal": "D: + both",
    "functional_response+temporal|block": "D: + both (block masking)",
}


def _source_extension_table(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    table = payload.get("condition_table")
    if not table:
        raise KeyError("the payload has no 'condition_table' (source-extension results)")
    return list(table)


def _entries(payload: Mapping[str, Any], representation: str) -> list[Mapping[str, Any]]:
    return [
        entry for entry in _source_extension_table(payload)
        if str(entry.get("representation")) == representation
    ]


def _ablation_groups(
    payload: Mapping[str, Any],
    metric: str,
) -> list[tuple[str, str, list[float]]]:
    """Ordered ``(group_key, display_label, values)`` for the source-ablation comparison."""
    ablation = payload.get("ablation_table") or []
    groups: list[tuple[str, str, list[float]]] = []
    for row in _source_extension_table(payload):
        if str(row.get("condition_role")) != "deterministic":
            continue
        if str(row.get("representation")) not in ("structured_48", "structured_48_plus_temporal"):
            continue
        value = _finite(row.get(metric))
        if np.isfinite(value):
            groups.append((str(row["representation"]), None, [value]))  # type: ignore[arg-type]

    ordered_keys = [
        "structured",
        "functional_response",
        "temporal",
        "functional_response+temporal",
        "functional_response+temporal|block",
    ]
    for key in ordered_keys:
        values = []
        for row in ablation:
            source = str(row.get("source_key"))
            mask = str(row.get("mask_mode"))
            tag = source if mask != "block" else f"{source}|block"
            if tag != key:
                continue
            value = _finite(row.get(metric))
            if np.isfinite(value):
                values.append(value)
        if values:
            groups.append((key, _ABLATION_LABELS.get(key, key), values))
    return groups


def source_extension_figure1_target_decomposition(
    payload: Mapping[str, Any],
    out_dir: str | Path,
    *,
    representation: str = "structured_48",
    previous_decomposition: Mapping[str, Any] | None = None,
    formats: Sequence[str] = DEFAULT_FORMATS,
    dpi: int = DEFAULT_DPI,
) -> list[Path]:
    """Target decomposition for one representation, with the previous study overlaid."""
    plt = _plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = _entries(payload, representation)
    if not entries:
        raise KeyError(f"no condition_table rows for representation {representation!r}")

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    xs = np.arange(len(_TARGET_DECOMPOSITION_COLUMNS))
    values, lows, highs = [], [], []
    for _label, column in _TARGET_DECOMPOSITION_COLUMNS:
        collected = [v for entry in entries if np.isfinite(v := _finite(entry.get(column)))]
        values.append(float(np.mean(collected)) if collected else float("nan"))
        low_column, high_column = column.replace("_r", "_bootstrap_low"), column.replace("_r", "_bootstrap_high")
        lo = [v for entry in entries if np.isfinite(v := _finite(entry.get(low_column)))]
        hi = [v for entry in entries if np.isfinite(v := _finite(entry.get(high_column)))]
        lows.append(float(np.mean(lo)) if lo else float("nan"))
        highs.append(float(np.mean(hi)) if hi else float("nan"))

    ax.bar(xs, values, width=0.5, color="#1f77b4", alpha=0.8, label=f"this stage ({representation})")
    for x, value, low, high in zip(xs, values, lows, highs):
        if np.isfinite(low) and np.isfinite(high):
            ax.errorbar([x], [value], yerr=[[value - low], [high - value]], fmt="none",
                        ecolor="#333333", capsize=3, lw=1.2)
    if previous_decomposition:
        previous = [
            _finite(previous_decomposition.get(column)) for _label, column in _TARGET_DECOMPOSITION_COLUMNS
        ]
        ax.plot(xs, previous, "o", mfc="none", mec="#7f7f7f", ms=8, mew=1.4,
                label="previous robustness study (structured_48)")
    ax.axhline(0.0, color="#999999", lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([label for label, _column in _TARGET_DECOMPOSITION_COLUMNS], fontsize=8)
    ax.set_ylabel("Mantel Spearman r")
    ax.set_title(f"Target decomposition: {representation}", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    _clean(ax)
    _conclusion_note(fig, "bars = mean over checkpoints (seeds where applicable); whiskers = bootstrap CI")
    return _finish(fig, out_dir / "source_extension_figure1_target_decomposition", formats=formats, dpi=dpi)


def source_extension_figure2_source_ablation(
    payload: Mapping[str, Any],
    out_dir: str | Path,
    *,
    metric: str = "zscored_r",
    formats: Sequence[str] = DEFAULT_FORMATS,
    dpi: int = DEFAULT_DPI,
) -> list[Path]:
    """The key figure: the source ablation against the neuron z-scored target."""
    plt = _plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = _ablation_groups(payload, metric)
    if not groups:
        raise KeyError("no ablation rows stored in the payload")

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    xs = np.arange(len(groups))
    for x, (key, label, values) in zip(xs, groups):
        color = _ABLATION_COLORS.get(key, "#333333")
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=0))
        ax.bar([x], [mean], width=0.6, color=color, alpha=0.8)
        if len(values) > 1:
            ax.errorbar([x], [mean], yerr=[std], fmt="none", ecolor="#333333", capsize=3, lw=1.2)
            ax.plot([x] * len(values), values, "k.", ms=4, alpha=0.8)
        ax.annotate(f"{mean:+.3f}", (x, mean), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8, color="#444444")
    ax.axhline(0.0, color="#999999", lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([label or key for key, label, _values in groups], rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("Mantel Spearman r (neuron z-scored target)")
    ax.set_title("Source ablation: rate- and amplitude-independent correspondence", fontsize=10)
    _clean(ax)
    _conclusion_note(fig, "bars = mean over checkpoints and residual seeds; dots = individual replicates", y=-0.30)
    return _finish(fig, out_dir / "source_extension_figure2_source_ablation", formats=formats, dpi=dpi)


def source_extension_figure3_raw_vs_shape(
    payload: Mapping[str, Any],
    out_dir: str | Path,
    *,
    formats: Sequence[str] = DEFAULT_FORMATS,
    dpi: int = DEFAULT_DPI,
) -> list[Path]:
    """Raw-target vs neuron z-scored-target correspondence for the ablation conditions."""
    plt = _plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ablation = payload.get("ablation_table") or []
    if not ablation:
        raise KeyError("no ablation rows stored in the payload")

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    plotted: set[str] = set()
    shape_groups = {key: values for key, _label, values in _ablation_groups(payload, "zscored_r")}
    for row in ablation:
        raw = _finite(row.get("raw_r"))
        shape = _finite(row.get("zscored_r"))
        if not (np.isfinite(raw) and np.isfinite(shape)):
            continue
        source = str(row.get("source_key"))
        mask = str(row.get("mask_mode"))
        key = source if mask != "block" else f"{source}|block"
        ax.scatter([raw], [shape], s=34, color=_ABLATION_COLORS.get(key, "#333333"), alpha=0.85,
                   label=_ABLATION_LABELS.get(key, key) if key not in plotted else None)
        plotted.add(key)
    for (key, label, values) in _ablation_groups(payload, "raw_r"):
        shape_values = shape_groups.get(key)
        if shape_values is None or not np.isfinite(np.mean(values)):
            continue
        ax.scatter([float(np.mean(values))], [float(np.mean(shape_values))],
                   marker="D", s=42, edgecolor="#111111", facecolor="none", lw=1.0, zorder=5)
    ax.axhline(0.0, color="#999999", lw=1.0)
    ax.axvline(0.0, color="#999999", lw=1.0)
    ax.set_xlabel("raw-target Mantel r (level + amplitude retained)")
    ax.set_ylabel("neuron z-scored target Mantel r (shape only)")
    ax.set_title("Raw vs stimulus-specific shape correspondence", fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    ax.legend(unique.values(), unique.keys(), fontsize=8, frameon=False, loc="best")
    _clean(ax)
    _conclusion_note(fig, "points = individual (checkpoint, residual seed) replicates; diamonds = group means")
    return _finish(fig, out_dir / "source_extension_figure3_raw_vs_shape", formats=formats, dpi=dpi)


def write_source_extension_figures(
    payload: Mapping[str, Any],
    out_dir: str | Path,
    *,
    representation: str = "structured_48",
    previous_decomposition: Mapping[str, Any] | None = None,
    formats: Sequence[str] = DEFAULT_FORMATS,
    dpi: int = DEFAULT_DPI,
) -> dict[str, list[str]]:
    """Render the three source-extension figures and return their paths by name."""
    written: dict[str, list[str]] = {}
    written["source_extension_figure1_target_decomposition"] = [
        str(p) for p in source_extension_figure1_target_decomposition(
            payload, out_dir, representation=representation,
            previous_decomposition=previous_decomposition, formats=formats, dpi=dpi,
        )
    ]
    written["source_extension_figure2_source_ablation"] = [
        str(p) for p in source_extension_figure2_source_ablation(
            payload, out_dir, formats=formats, dpi=dpi
        )
    ]
    written["source_extension_figure3_raw_vs_shape"] = [
        str(p) for p in source_extension_figure3_raw_vs_shape(
            payload, out_dir, formats=formats, dpi=dpi
        )
    ]
    return written


__all__ = [
    "DEFAULT_FORMATS",
    "DEFAULT_DPI",
    "TARGET_VARIANT_ORDER",
    "REPRESENTATION_ORDER",
    "figure1_capacity_vs_dimension",
    "figure2_primary_vs_class_rate",
    "figure3_distance_relationship",
    "write_all_figures",
    "rate_figure1_target_decomposition",
    "rate_figure2_representation_comparison",
    "rate_figure3_activity_diagnostic",
    "write_rate_robustness_figures",
    "source_extension_figure1_target_decomposition",
    "source_extension_figure2_source_ablation",
    "source_extension_figure3_raw_vs_shape",
    "write_source_extension_figures",
]