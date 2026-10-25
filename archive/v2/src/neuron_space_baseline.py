"""Neuron-space baseline experiment: canonical representations, PRIMARY ANALYSIS
and the full control table.

Scientific question
-------------------
Can a **label-free structural representation of an individual hidden neuron**
predict its **function**, and does any such relationship survive the trivial
explanation "it is just firing-rate magnitude"?

This module is the single, canonical implementation of the experiment. It keeps
the separation that the whole study depends on:

* **representation** - label-free. Built from the network parameters
  (``intrinsic`` / ``input_conn`` / ``recurrent_in`` / ``recurrent_out``) and,
  optionally, from label-free firing statistics measured on the FIT split.
* **functional fingerprint** - uses class labels by construction, is measured on
  the held-out **PROBE** split, and is *never* an input to the representation.
* **official TEST** - never touched here.

PRIMARY ANALYSIS (not "pre-registered")
---------------------------------------
The primary statistic is the Spearman correlation between the condensed
(neuron-pair) distance matrix of the representation and the condensed distance
matrix of the primary fingerprint, with a **Mantel permutation test** whose null
relabels **neurons** (never pairs, so neuron pairs are not treated as independent
observations). Reported alongside: the permutation p-value, its resolution floor
``1/(n_perm+1)``, the null mean/SD, the effect-size z and a neuron-bootstrap CI.

The rate question
-----------------
The result is *not* defended with a partial Mantel. It is defended (or falsified)
by combining:

1. the raw geometry/function association,
2. the association against a **rate-normalized** fingerprint (overall firing-rate
   magnitude removed), and a rate-only baseline representation,
3. a **rate-matched** stratified Mantel (only neuron pairs with similar firing
   rates are compared),
4. cross-validated prediction,
5. shuffled and random controls,
6. a rewired-recurrent control,
7. replication across seeds.

The partial Mantel is retained only as a secondary, exploratory statistic and is
explicitly labelled as *not* proof of rate independence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .controls import VariantSpec, build_variant_space
from .functional_fingerprint import FingerprintSpace
from .geometry_analysis import geometry_function_analysis
from .neurons import FeatureBlock, NeuronRepresentationSet
from .prediction import (
    DEFAULT_ALPHAS,
    cross_validated_knn,
    cross_validated_ridge,
    fold_assignment,
    make_shared_folds,
)
from .representations import EmptyFeatureSelectionError, RepresentationSpace

# --------------------------------------------------------------------------
# The canonical representation/control set (order = reporting order)
# --------------------------------------------------------------------------
#: The canonical names, in the order they appear in the results table. The first
#: five are single-source baselines; ``structural`` is the PRIMARY representation;
#: the rest are controls.
CANONICAL_REPRESENTATION_NAMES: tuple[str, ...] = (
    "rate_only",
    "activity_only",
    "input_conn_only",
    "recurrent_only",
    "intrinsic_only",
    "structural",
    "structural_plus_activity",
    "random",
    "neuron_shuffle",
)

#: Representations whose cross-validated prediction is compared on identical folds.
CV_COMPARISON_NAMES: tuple[str, ...] = (
    "rate_only",
    "intrinsic_only",
    "input_conn_only",
    "recurrent_only",
    "structural",
    "structural_plus_activity",
    "random",
    "activity_only",
    "neuron_shuffle",
)

#: The PRIMARY representation under study.
PRIMARY_REPRESENTATION = "structural"

#: Human-readable label of the primary functional target (for the table).
FUNCTIONAL_TARGET_LABELS: dict[str, str] = {
    "tuning": "class_rate_20d_probe",
    "tuning_rate_normalized": "class_rate_norm_20d_probe",
    "temporal": "temporal_probe",
}


def nsb_variants() -> list[VariantSpec]:
    """The canonical representation/control set, exactly once each.

    ``structural`` is the PRIMARY representation. ``random`` and
    ``neuron_shuffle`` are negative controls; ``rate_only`` / ``activity_only``
    are the trivial baselines the structural representation must beat.
    """
    intrinsic = FeatureBlock.INTRINSIC.value
    input_conn = FeatureBlock.INPUT_CONN.value
    rec_in = FeatureBlock.RECURRENT_IN.value
    rec_out = FeatureBlock.RECURRENT_OUT.value
    activity = FeatureBlock.ACTIVITY.value
    structural = FeatureBlock.structural()

    return [
        VariantSpec(
            name="rate_only", kind="rate_only", is_control=True,
            description=(
                "TRIVIAL BASELINE: a single scalar mean firing rate per neuron. "
                "The multi-feature representation must beat this to be interesting."
            ),
        ),
        VariantSpec(
            name="activity_only", kind="blocks", blocks=[activity], use_activity=True,
            description="label-free firing statistics only (no connectivity, no parameters)",
        ),
        VariantSpec(
            name="input_conn_only", kind="blocks", blocks=[input_conn],
            description="input-connectivity statistics only",
        ),
        VariantSpec(
            name="recurrent_only", kind="blocks", blocks=[rec_in, rec_out],
            description="recurrent connectivity only (incoming + outgoing statistics)",
        ),
        VariantSpec(
            name="intrinsic_only", kind="blocks", blocks=[intrinsic],
            description=(
                "per-neuron learned parameters only. In this architecture that is the "
                "generic learned bias (a learned excitability offset, NOT a biophysical "
                "parameter); the block is empty for an untrained network and is skipped."
            ),
        ),
        VariantSpec(
            name="structural", kind="blocks", blocks=structural,
            description=(
                "PRIMARY REPRESENTATION: all structural blocks, label-free, no data, "
                "no labels (intrinsic + input connectivity + recurrent in/out)."
            ),
        ),
        VariantSpec(
            name="structural_plus_activity", kind="blocks", blocks=structural + [activity],
            use_activity=True,
            description="primary structural representation plus label-free activity statistics",
        ),
        VariantSpec(
            name="random", kind="random", is_control=True,
            description="i.i.d. Gaussian features, dimensionality matched to the structural representation",
        ),
        VariantSpec(
            name="neuron_shuffle", kind="shuffled", is_control=True,
            description=(
                "the structural representation with neuron rows permuted (destroys the "
                "representation<->function pairing; should give no association)"
            ),
        ),
    ]


# --------------------------------------------------------------------------
# One representation
# --------------------------------------------------------------------------
def _nan(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f


def evaluate_representation(
    space: RepresentationSpace,
    *,
    tuning: FingerprintSpace,
    tuning_normalized: FingerprintSpace,
    rate_absdiff: np.ndarray | None,
    n_perm: int,
    k_values: Sequence[int],
    seed: int,
    bootstrap: int = 0,
    n_strata: int = 5,
    prediction: Mapping[str, Any] | None = None,
    cv: Any | None = None,
) -> dict[str, Any]:
    """Raw + rate-normalized + rate-matched + predictive metrics for one representation.

    Every metric is measured against the *same* canonical fingerprint objects and
    (when ``cv`` is supplied) on the *same* neuron folds.
    """
    raw = geometry_function_analysis(
        space.X, tuning.X,
        n_perm=n_perm, k_values=k_values, seed=seed,
        rate_absdiff_condensed=rate_absdiff, nuisance_condensed=rate_absdiff,
        rate_matched_strata=n_strata, primary_bootstrap=bootstrap,
        primary_return_null=False,
    )
    norm = geometry_function_analysis(
        space.X, tuning_normalized.X,
        n_perm=n_perm, k_values=[], seed=seed, include_curves=False,
    )
    p_raw = raw["primary_mantel_spearman"]
    p_norm = norm["primary_mantel_spearman"]
    rm = raw.get("rate_matched_mantel", {})
    partial = raw.get("partial_mantel_controlling_for_nuisance", {})

    row: dict[str, Any] = {
        "n_features": int(len(space.feature_names)),
        "n_informative_features": int(space.standardizer.n_informative),
        "feature_names": list(space.feature_names),
        "blocks": list(space.blocks),
        # raw geometry/function association vs the primary (raw) fingerprint
        "raw_r": _nan(p_raw.get("statistic")),
        "raw_p": _nan(p_raw.get("p_value")),
        "raw_p_floor": _nan(p_raw.get("p_value_floor")),
        "raw_at_floor": bool(p_raw.get("at_resolution_floor")),
        "raw_null_mean": _nan(p_raw.get("null_mean")),
        "raw_null_std": _nan(p_raw.get("null_std")),
        "raw_effect_size_z": _nan(p_raw.get("effect_size_z")),
        "raw_ci_low": _nan((p_raw.get("bootstrap_ci") or {}).get("low")),
        "raw_ci_high": _nan((p_raw.get("bootstrap_ci") or {}).get("high")),
        # association vs the RATE-NORMALIZED fingerprint (primary rate control)
        "rate_normalized_r": _nan(p_norm.get("statistic")),
        "rate_normalized_p": _nan(p_norm.get("p_value")),
        "rate_normalized_at_floor": bool(p_norm.get("at_resolution_floor")),
        "rate_normalized_effect_size_z": _nan(p_norm.get("effect_size_z")),
        # rate-matched stratified Mantel (proper rate-matched control)
        "rate_matched_r": _nan(rm.get("statistic")),
        "rate_matched_p": _nan(rm.get("p_value")),
        "rate_matched_effect_size_z": _nan(rm.get("effect_size_z")),
        # partial Mantel: SECONDARY / EXPLORATORY ONLY
        "partial_mantel_r_secondary_exploratory": _nan(partial.get("statistic")),
        "partial_mantel_p_secondary_exploratory": _nan(partial.get("p_value")),
        "knn": raw.get("knn", {}).get("table", []),
    }

    rn, rn_p = row["rate_normalized_r"], row["rate_normalized_p"]
    rmp, rmm = row["rate_matched_p"], row["rate_matched_r"]
    row["survives_rate_normalization"] = bool(np.isfinite(rn) and rn > 0 and np.isfinite(rn_p) and rn_p < 0.05)
    row["survives_rate_matching"] = bool(np.isfinite(rmm) and rmm > 0 and np.isfinite(rmp) and rmp < 0.05)
    if row["survives_rate_normalization"] and row["survives_rate_matching"]:
        row["rate_control_verdict"] = "survives both rate controls"
    elif row["survives_rate_normalization"]:
        row["rate_control_verdict"] = "survives rate normalization only"
    elif row["survives_rate_matching"]:
        row["rate_control_verdict"] = "survives rate matching only"
    else:
        row["rate_control_verdict"] = "not distinguishable from a firing-rate effect"

    pred_cfg = dict(prediction or {})
    if pred_cfg.get("enabled", True):
        n_splits = int(pred_cfg.get("n_splits", 5))
        alphas = tuple(pred_cfg.get("alphas", DEFAULT_ALPHAS))
        ridge = cross_validated_ridge(
            space.X, tuning.X, n_splits=n_splits, alphas=alphas, seed=seed, cv=cv
        )["metrics"]
        knn = cross_validated_knn(
            space.X, tuning.X, n_splits=n_splits, k=int(pred_cfg.get("knn_k", 5)), seed=seed, cv=cv
        )["metrics"]
        row.update({
            "cv_r2_mean": _nan(ridge.get("r2_mean")),
            "cv_r2_median": _nan(ridge.get("r2_median")),
            "cv_r2_overall": _nan(ridge.get("r2_overall")),
            "cv_pearson_r_mean": _nan(ridge.get("pearson_r_mean")),
            "cv_pearson_r_median": _nan(ridge.get("pearson_r_median")),
            "cv_rmse_mean": _nan(ridge.get("rmse_mean")),
            "cv_nrmse_mean": _nan(ridge.get("nrmse_mean")),
            "cv_mae_mean": _nan(ridge.get("mae_mean")),
            "cv_fingerprint_distance_spearman": _nan(ridge.get("predicted_vs_true_fingerprint_distance_spearman")),
            "cv_knn_r2_mean": _nan(knn.get("r2_mean")),
            "cv_knn_pearson_r_mean": _nan(knn.get("pearson_r_mean")),
            "cv_knn_nrmse_mean": _nan(knn.get("nrmse_mean")),
            "cv_n_splits": int(ridge.get("n_splits", n_splits)),
        })
    return row


# --------------------------------------------------------------------------
# Cross-validated prediction across representations (IDENTICAL folds)
# --------------------------------------------------------------------------
def run_cv_comparison(
    representation_matrices: Mapping[str, np.ndarray],
    Y: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 0,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    knn_k: int = 5,
    folds: Any | None = None,
) -> dict[str, Any]:
    """Ridge (and kNN) CV prediction of ``Y`` from every representation, same folds.

    One ``KFold`` splitter is built once and reused for every representation, so
    differences between rows cannot come from a different neuron partition. The
    fold assignment is returned so it can be audited/verified.
    """
    Y = np.asarray(Y, dtype=np.float64)
    n = Y.shape[0]
    folds = folds if folds is not None else make_shared_folds(n, n_splits, seed)
    assignment = fold_assignment(n, n_splits, seed, cv=folds)
    out: dict[str, Any] = {
        "n_neurons": int(n),
        "n_targets": int(Y.shape[1]),
        "n_splits": int(getattr(folds, "n_splits", n_splits)),
        "same_folds_for_all": True,
        "fold_assignment": assignment.tolist(),
        "representations": {},
    }
    for name, X in representation_matrices.items():
        X = np.asarray(X, dtype=np.float64)
        if X.shape[0] != n:
            raise ValueError(f"Representation {name!r} has {X.shape[0]} neurons, expected {n}")
        ridge = cross_validated_ridge(X, Y, n_splits=n_splits, alphas=alphas, seed=seed, cv=folds)
        knn = cross_validated_knn(X, Y, n_splits=n_splits, k=knn_k, seed=seed, cv=folds)
        rm = ridge["metrics"]
        km = knn["metrics"]
        out["representations"][name] = {
            "n_features": int(X.shape[1]),
            "ridge": {k: v for k, v in rm.items() if k != "per_target"},
            "ridge_per_target": rm.get("per_target", []),
            "knn": {k: v for k, v in km.items() if k != "per_target"},
            "cv_r2_mean": _nan(rm.get("r2_mean")),
            "cv_pearson_r_mean": _nan(rm.get("pearson_r_mean")),
            "cv_nrmse_mean": _nan(rm.get("nrmse_mean")),
            "cv_rmse_mean": _nan(rm.get("rmse_mean")),
            "cv_knn_r2_mean": _nan(km.get("r2_mean")),
            "cv_knn_pearson_r_mean": _nan(km.get("pearson_r_mean")),
        }
    return out


# --------------------------------------------------------------------------
# One network condition (after / before / rewired)
# --------------------------------------------------------------------------
@dataclass
class NSBCondition:
    """One network condition with its label-free representation and its fingerprints."""

    name: str
    structural: NeuronRepresentationSet
    activity: NeuronRepresentationSet
    fingerprints: Mapping[str, FingerprintSpace]
    metadata: dict[str, Any] = field(default_factory=dict)


def _rate_absdiff_condensed(activity: NeuronRepresentationSet) -> np.ndarray:
    X, names = activity.to_matrix(blocks=[FeatureBlock.ACTIVITY.value])
    idx = [i for i, n in enumerate(names) if n.endswith(".rate_hz")]
    if not idx:
        idx = [i for i, n in enumerate(names) if n.endswith(".log_rate_hz")]
    rates = X[:, idx[0]] if idx else np.zeros(X.shape[0])
    r = rates.reshape(-1, 1)
    return np.abs(r - r.T)[np.triu_indices(r.size, 1)]


def run_condition(
    condition: NSBCondition,
    *,
    tuning_key: str,
    tuning_normalized_key: str,
    n_perm: int,
    k_values: Sequence[int],
    seed: int,
    bootstrap: int = 0,
    n_strata: int = 5,
    control_repeats: int = 5,
    prediction: Mapping[str, Any] | None = None,
    variants: Sequence[VariantSpec] | None = None,
) -> dict[str, Any]:
    """Evaluate every canonical representation/control for one network condition.

    Returns ``{"rows": [...], "spaces": {...}, "fingerprint_rate_matrix": ...,
    "rate_absdiff": ...}``. Rows are JSON-safe dicts (one per representation;
    multi-repeat controls emit ``<name>#<rep>`` rows plus a mean/SD summary).
    """
    variants = list(variants or nsb_variants())
    if tuning_key not in condition.fingerprints:
        raise KeyError(f"condition {condition.name!r} has no fingerprint {tuning_key!r}")
    if tuning_normalized_key not in condition.fingerprints:
        raise KeyError(f"condition {condition.name!r} has no fingerprint {tuning_normalized_key!r}")
    tuning = condition.fingerprints[tuning_key]
    tuning_norm = condition.fingerprints[tuning_normalized_key]
    rate_absdiff = _rate_absdiff_condensed(condition.activity)

    # A single structural space provides the dimensionality for the random control
    # and the matrix that the neuron-shuffle control permutes.
    structural_space, _ = build_variant_space(
        VariantSpec(name="_structural_reference", kind="blocks", blocks=FeatureBlock.structural()),
        condition.structural, condition.activity,
    )
    n_neurons = structural_space.n_neurons
    folds = make_shared_folds(n_neurons, int((prediction or {}).get("n_splits", 5)), seed)

    rows: list[dict[str, Any]] = []
    spaces: dict[str, RepresentationSpace] = {}
    names_seen: set[str] = set()

    for spec in variants:
        repeats = int(control_repeats) if spec.kind in ("random", "shuffled") else 1
        if repeats < 1:
            repeats = 1
        rep_rows: list[dict[str, Any]] = []
        for rep_idx in range(repeats):
            try:
                space, _circular = build_variant_space(
                    spec, condition.structural, condition.activity, condition.fingerprints,
                    random_seed=seed + 1000 * rep_idx, shuffle_seed=seed + 1000 * rep_idx,
                    full_space=structural_space,
                )
            except EmptyFeatureSelectionError as exc:
                rows.append({
                    "condition": condition.name, "representation": spec.name,
                    "is_control": bool(spec.is_control), "skipped": True,
                    "skip_reason": str(exc), "n_features": 0,
                    "description": spec.description,
                })
                names_seen.add(spec.name)
                break
            row = evaluate_representation(
                space, tuning=tuning, tuning_normalized=tuning_norm, rate_absdiff=rate_absdiff,
                n_perm=n_perm, k_values=k_values, seed=seed, bootstrap=bootstrap,
                n_strata=n_strata, prediction=prediction, cv=folds,
            )
            row.update({"condition": condition.name, "representation": spec.name,
                        "repeat": rep_idx, "is_control": bool(spec.is_control),
                        "description": spec.description, "skipped": False})
            rep_rows.append(row)
            if rep_idx == 0:
                spaces[spec.name] = space

        if not rep_rows:
            continue
        names_seen.add(spec.name)
        if len(rep_rows) == 1:
            rows.append(rep_rows[0])
        else:
            # Aggregate repeated controls: keep the mean row and record the spread.
            mean_row = dict(rep_rows[0])
            for key in ("raw_r", "rate_normalized_r", "rate_matched_r",
                        "cv_r2_mean", "cv_pearson_r_mean", "cv_nrmse_mean"):
                vals = np.array([_nan(r.get(key)) for r in rep_rows], dtype=np.float64)
                mean_row[key] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
                mean_row[f"{key}_std"] = float(np.nanstd(vals)) if np.isfinite(vals).any() else float("nan")
            mean_row["n_repeats"] = len(rep_rows)
            mean_row["repeat_values"] = {
                key: [float(_nan(r.get(key))) for r in rep_rows]
                for key in ("raw_r", "rate_normalized_r", "cv_r2_mean")
            }
            mean_row["representation"] = spec.name
            rows.append(mean_row)

    # Ordered to match CANONICAL_REPRESENTATION_NAMES where possible.
    order = {n: i for i, n in enumerate(CANONICAL_REPRESENTATION_NAMES)}
    rows.sort(key=lambda r: (order.get(r["representation"], 999),))

    return {
        "rows": rows,
        "spaces": spaces,
        "rate_absdiff": rate_absdiff,
        "tuning_condensed": tuning.condensed(),
        "tuning_normalized_condensed": tuning_norm.condensed(),
        "representation_names": sorted(names_seen),
    }


# --------------------------------------------------------------------------
# Before vs after learning
# --------------------------------------------------------------------------
def before_after_nsb(
    after_rows: Sequence[Mapping[str, Any]],
    before_rows: Sequence[Mapping[str, Any]],
    *,
    representation: str = PRIMARY_REPRESENTATION,
) -> list[dict[str, Any]]:
    """Per-representation change in raw / rate-normalized association and prediction.

    Both conditions use the same architecture, the same initialisation seed and the
    same PROBE samples; only the weights differ (untrained vs trained).
    """
    after_by = {r["representation"]: r for r in after_rows if not r.get("skipped")}
    before_by = {r["representation"]: r for r in before_rows if not r.get("skipped")}
    out: list[dict[str, Any]] = []
    for name in CANONICAL_REPRESENTATION_NAMES:
        ra = after_by.get(name)
        rb = before_by.get(name)
        if ra is None and rb is None:
            continue
        out.append({
            "variant": name,
            "representation": name,
            "raw_r_before": (rb or {}).get("raw_r"),
            "raw_r_after": (ra or {}).get("raw_r"),
            "delta_raw_r": _diff(ra, rb, "raw_r"),
            "rate_normalized_r_before": (rb or {}).get("rate_normalized_r"),
            "rate_normalized_r_after": (ra or {}).get("rate_normalized_r"),
            "delta_rate_normalized_r": _diff(ra, rb, "rate_normalized_r"),
            "rate_matched_r_before": (rb or {}).get("rate_matched_r"),
            "rate_matched_r_after": (ra or {}).get("rate_matched_r"),
            "delta_rate_matched_r": _diff(ra, rb, "rate_matched_r"),
            "prediction_r2_before": (rb or {}).get("cv_r2_mean"),
            "prediction_r2_after": (ra or {}).get("cv_r2_mean"),
            "delta_prediction_r2": _diff(ra, rb, "cv_r2_mean"),
            "cv_nrmse_before": (rb or {}).get("cv_nrmse_mean"),
            "cv_nrmse_after": (ra or {}).get("cv_nrmse_mean"),
            "note": "" if (ra is not None and rb is not None) else "missing in one condition (e.g. empty block before learning)",
        })
    return out


def _diff(a: Mapping[str, Any] | None, b: Mapping[str, Any] | None, key: str) -> float:
    if a is None or b is None:
        return float("nan")
    va, vb = _nan(a.get(key)), _nan(b.get(key))
    if not (np.isfinite(va) and np.isfinite(vb)):
        return float("nan")
    return va - vb


# --------------------------------------------------------------------------
# Multi-seed aggregation
# --------------------------------------------------------------------------
def _ci95_t(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray([_nan(v) for v in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan"), float("nan")
    from scipy.stats import t as _t

    mean = float(arr.mean())
    sem = float(arr.std(ddof=1) / np.sqrt(arr.size))
    if sem <= 0:
        return mean, mean
    half = float(_t.ppf(0.975, df=arr.size - 1) * sem)
    return mean - half, mean + half


def aggregate_primary_over_seeds(
    per_seed: Mapping[int, Mapping[str, Any]],
    *,
    representation: str = PRIMARY_REPRESENTATION,
    metrics: Sequence[str] = ("raw_r", "rate_normalized_r", "rate_matched_r",
                              "cv_r2_mean", "cv_pearson_r_mean", "cv_nrmse_mean"),
) -> dict[str, Any]:
    """Per-seed values, mean, SD and 95% CI of the primary metrics across seeds."""
    out: dict[str, Any] = {"representation": representation, "seeds": sorted(per_seed), "metrics": {}}
    for metric in metrics:
        vals = []
        for s in sorted(per_seed):
            row = _row_for(per_seed[s], representation)
            vals.append(_nan(row.get(metric)) if row else float("nan"))
        finite = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
        lo, hi = _ci95_t(vals)
        out["metrics"][metric] = {
            "per_seed": [float(v) for v in vals],
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "sd": float(finite.std(ddof=1)) if finite.size > 1 else float("nan"),
            "ci95_low": lo,
            "ci95_high": hi,
            "n_seeds": int(finite.size),
        }
    return out


def _row_for(seed_result: Mapping[str, Any], representation: str) -> dict[str, Any] | None:
    for row in seed_result.get("after_learning", {}).get("rows", []):
        if row.get("representation") == representation and not row.get("skipped"):
            return row
    return None


# --------------------------------------------------------------------------
# The ONE canonical results table
# --------------------------------------------------------------------------
CANONICAL_TABLE_COLUMNS: tuple[str, ...] = (
    "representation",
    "functional_target",
    "distance_metric",
    "Mantel_r",
    "permutation_p",
    "rate_normalized_r",
    "CV_R2",
    "CV_correlation",
    "CV_error",
    "seed",
    "checkpoint",
    "probe_split",
    "representation_dimensionality",
    # --- supporting columns (never a substitute for the required ones above) ---
    "condition",
    "is_control",
    "permutation_p_floor",
    "permutation_at_floor",
    "null_mean",
    "null_std",
    "effect_size_z",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "rate_matched_r",
    "rate_matched_p",
    "partial_mantel_r_secondary_exploratory",
    "CV_nRMSE",
    "CV_RMSE",
    "CV_kNN_R2",
    "n_features",
    "n_informative_features",
    "n_repeats",
    "rate_control_verdict",
    "skipped",
    "skip_reason",
    "description",
)


def build_canonical_table(
    per_seed: Mapping[int, Mapping[str, Any]],
    *,
    checkpoints: Mapping[int, str],
    probe_split: str,
    functional_target_key: str = "tuning",
    distance_metric: str = "spearman",
    conditions: Sequence[str] = ("after_learning",),
    functional_target_label: str | None = None,
) -> list[dict[str, Any]]:
    """Assemble the single canonical results table (one row per representation/seed).

    The first thirteen columns are the required reporting schema; everything after
    them is supporting detail. Controls are included, not filtered out.
    """
    target_label = functional_target_label or FUNCTIONAL_TARGET_LABELS.get(
        functional_target_key, functional_target_key
    )
    rows: list[dict[str, Any]] = []
    for seed in sorted(per_seed):
        for condition in conditions:
            block = per_seed[seed].get(condition, {})
            for r in block.get("rows", []):
                rows.append({
                    "representation": r.get("representation"),
                    "functional_target": target_label,
                    "distance_metric": distance_metric,
                    "Mantel_r": r.get("raw_r"),
                    "permutation_p": r.get("raw_p"),
                    "rate_normalized_r": r.get("rate_normalized_r"),
                    "CV_R2": r.get("cv_r2_mean"),
                    "CV_correlation": r.get("cv_pearson_r_mean"),
                    "CV_error": r.get("cv_nrmse_mean"),
                    "seed": int(seed),
                    "checkpoint": checkpoints.get(seed),
                    "probe_split": probe_split,
                    "representation_dimensionality": r.get("n_features"),
                    "condition": r.get("condition", condition),
                    "is_control": r.get("is_control"),
                    "permutation_p_floor": r.get("raw_p_floor"),
                    "permutation_at_floor": r.get("raw_at_floor"),
                    "null_mean": r.get("raw_null_mean"),
                    "null_std": r.get("raw_null_std"),
                    "effect_size_z": r.get("raw_effect_size_z"),
                    "bootstrap_ci_low": r.get("raw_ci_low"),
                    "bootstrap_ci_high": r.get("raw_ci_high"),
                    "rate_matched_r": r.get("rate_matched_r"),
                    "rate_matched_p": r.get("rate_matched_p"),
                    "partial_mantel_r_secondary_exploratory": r.get("partial_mantel_r_secondary_exploratory"),
                    "CV_nRMSE": r.get("cv_nrmse_mean"),
                    "CV_RMSE": r.get("cv_rmse_mean"),
                    "CV_kNN_R2": r.get("cv_knn_r2_mean"),
                    "n_features": r.get("n_features"),
                    "n_informative_features": r.get("n_informative_features"),
                    "n_repeats": r.get("n_repeats"),
                    "rate_control_verdict": r.get("rate_control_verdict"),
                    "skipped": bool(r.get("skipped", False)),
                    "skip_reason": r.get("skip_reason"),
                    "description": r.get("description"),
                })
    return rows


def asked_questions_summary(
    per_seed: Mapping[int, Mapping[str, Any]],
    *,
    primary: str = PRIMARY_REPRESENTATION,
    rate_only: str = "rate_only",
    random_name: str = "random",
    shuffle_name: str = "neuron_shuffle",
) -> dict[str, Any]:
    """Compact answers to the eight questions the report must address.

    Pure bookkeeping over the (already computed) after-learning rows for seed 0 -
    it never selects a favourable result, it reports the whole picture.
    """
    seed0 = per_seed[min(per_seed)]
    rows = {r["representation"]: r for r in seed0["after_learning"]["rows"] if not r.get("skipped")}
    p = rows.get(primary, {})
    r0 = rows.get(rate_only, {})
    rnd = rows.get(random_name, {})
    shf = rows.get(shuffle_name, {})
    before_after = seed0.get("before_after", [])
    ba_struct = next((b for b in before_after if b.get("variant") == primary), {})
    rewired = seed0.get("rewired", {})
    return {
        "q1_repr_correlates_with_function": {
            "raw_r": p.get("raw_r"), "p": p.get("raw_p"),
            "z": p.get("raw_effect_size_z"),
            "ci": [p.get("raw_ci_low"), p.get("raw_ci_high")],
        },
        "q2_is_it_just_firing_rate": {
            "structural_raw_r": p.get("raw_r"),
            "rate_only_raw_r": r0.get("raw_r"),
            "structural_beats_rate_only": bool(_nan(p.get("raw_r")) > _nan(r0.get("raw_r"))),
            "structural_cv_r2": p.get("cv_r2_mean"),
            "rate_only_cv_r2": r0.get("cv_r2_mean"),
            "structural_cv_beats_rate_only": bool(_nan(p.get("cv_r2_mean")) > _nan(r0.get("cv_r2_mean"))),
        },
        "q3_rate_normalization_preserves_relationship": {
            "structural_rate_normalized_r": p.get("rate_normalized_r"),
            "structural_rate_normalized_p": p.get("rate_normalized_p"),
            "structural_rate_matched_r": p.get("rate_matched_r"),
            "structural_rate_matched_p": p.get("rate_matched_p"),
            "verdict": p.get("rate_control_verdict"),
        },
        "q4_predicts_fingerprints": {
            "cv_r2_mean": p.get("cv_r2_mean"), "cv_r": p.get("cv_pearson_r_mean"),
            "cv_nrmse": p.get("cv_nrmse_mean"),
            "random_cv_r2_mean": rnd.get("cv_r2_mean"),
            "beats_random": bool(_nan(p.get("cv_r2_mean")) > _nan(rnd.get("cv_r2_mean"))),
        },
        "q5_survives_neuron_permutation": {
            "shuffle_raw_r": shf.get("raw_r"), "shuffle_p": shf.get("raw_p"),
            "structural_raw_r": p.get("raw_r"),
            "destroys_effect": bool(abs(_nan(shf.get("raw_r"))) < 0.05),
            "representation_invariance_verified": seed0.get("permutation_invariance", {}).get("passed"),
        },
        "q6_survives_rewiring": {
            mode: {
                "structural_raw_r": next(
                    (r.get("raw_r") for r in block.get("rows", [])
                     if r.get("representation") == primary and not r.get("skipped")), None),
                "structural_rate_normalized_r": next(
                    (r.get("rate_normalized_r") for r in block.get("rows", [])
                     if r.get("representation") == primary and not r.get("skipped")), None),
            }
            for mode, block in rewired.items()
        } if rewired else {},
        "q7_learning_strengthens_organization": {
            "raw_r_before": ba_struct.get("raw_r_before"),
            "raw_r_after": ba_struct.get("raw_r_after"),
            "delta_raw_r": ba_struct.get("delta_raw_r"),
            "rate_normalized_r_before": ba_struct.get("rate_normalized_r_before"),
            "rate_normalized_r_after": ba_struct.get("rate_normalized_r_after"),
            "delta_rate_normalized_r": ba_struct.get("delta_rate_normalized_r"),
            "cv_r2_before": ba_struct.get("prediction_r2_before"),
            "cv_r2_after": ba_struct.get("prediction_r2_after"),
        },
    }
