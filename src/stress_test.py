"""Stress-test the neuron-space result against trivial explanations.

The scientific question is whether the representation-function relationship is
real or could be produced by something trivial. This module evaluates a fixed set
of representations/controls **through the identical pipeline** and, for *every*
one, asks the mandatory question:

    "Could this be explained simply by firing rate?"

by reporting, alongside the raw association:

* the association against a **rate-normalized** functional fingerprint, and
* a **rate-matched** stratified Mantel (only neuron pairs with similar firing
  rates are compared),

plus a cross-validated **predictive** metric (ridge regression, representation ->
fingerprint) and the permutation significance with its resolution floor.

The required representations/controls:

1.  firing-rate-only representation
2.  activity-only representation
3.  input-connectivity-only representation
4.  recurrent-connectivity-only representation
5.  intrinsic/dynamical-only representation
6.  full structural representation (the primary conservative representation)
7.  structural + activity
8.  random representation
9.  hidden-neuron permutation / shuffle control
10. rewired recurrent-network control (a *condition*, since the network changes)

plus the **before-learning** and **after-learning** conditions on the same
architecture and the same analysis-probe data.

No result is hidden: every row is reported (including controls and skipped
variants), and the compact table contains the five required columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .controls import VariantSpec, build_variant_space
from .functional_fingerprint import FingerprintSpace
from .geometry_analysis import geometry_function_analysis, mantel_test
from .neurons import FeatureBlock, NeuronRepresentationSet
from .prediction import cross_validated_ridge
from .representations import EmptyFeatureSelectionError, RepresentationSpace


# --------------------------------------------------------------------------
# The required representations / controls
# --------------------------------------------------------------------------
def stress_variants() -> list[VariantSpec]:
    """The fixed set of representations/controls evaluated for every condition."""
    intrinsic = FeatureBlock.INTRINSIC.value
    input_conn = FeatureBlock.INPUT_CONN.value
    rec_in = FeatureBlock.RECURRENT_IN.value
    rec_out = FeatureBlock.RECURRENT_OUT.value
    activity = FeatureBlock.ACTIVITY.value
    structural = FeatureBlock.structural()

    return [
        VariantSpec(
            name="firing_rate_only", kind="rate_only",
            description="1. firing-rate-only representation (one scalar per neuron)",
            is_control=True,
        ),
        VariantSpec(
            name="activity_only", kind="blocks", blocks=[activity], use_activity=True,
            description="2. activity-only representation (label-free firing statistics)",
        ),
        VariantSpec(
            name="input_conn_only", kind="blocks", blocks=[input_conn],
            description="3. input-connectivity-only representation",
        ),
        VariantSpec(
            name="recurrent_only", kind="blocks", blocks=[rec_in, rec_out],
            description="4. recurrent-connectivity-only representation (incoming + outgoing)",
        ),
        VariantSpec(
            name="intrinsic_dynamical_only", kind="blocks", blocks=[intrinsic],
            description=(
                "5. intrinsic/dynamical-only representation. In the default 'bias' "
                "architecture this is the generic learned bias only (no genuine "
                "dynamical parameters exist); skipped when the block is empty."
            ),
        ),
        VariantSpec(
            name="structural_full", kind="blocks", blocks=structural,
            description="6. full structural representation (PRIMARY conservative, label-free)",
        ),
        VariantSpec(
            name="structural_plus_activity", kind="blocks", blocks=structural + [activity],
            use_activity=True,
            description="7. structural + activity representation",
        ),
        VariantSpec(
            name="random_representation", kind="random",
            description="8. random representation (i.i.d. Gaussian, matched dimensionality)",
            is_control=True,
        ),
        VariantSpec(
            name="shuffled_neurons", kind="shuffled",
            description="9. hidden-neuron permutation/shuffle control",
            is_control=True,
        ),
    ]


@dataclass
class StressCondition:
    """One network condition (after learning, before learning, or rewired)."""

    name: str
    structural: NeuronRepresentationSet
    activity: NeuronRepresentationSet
    fingerprints: Mapping[str, FingerprintSpace]
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Per-variant evaluation
# --------------------------------------------------------------------------
def _rate_absdiff_condensed(activity: NeuronRepresentationSet) -> np.ndarray:
    X, names = activity.to_matrix(blocks=[FeatureBlock.ACTIVITY.value])
    idx = [i for i, n in enumerate(names) if n.endswith(".rate_hz")]
    if not idx:
        idx = [i for i, n in enumerate(names) if n.endswith(".log_rate_hz")]
    rates = X[:, idx[0]] if idx else np.zeros(X.shape[0])
    r = rates.reshape(-1, 1)
    return np.abs(r - r.T)[np.triu_indices(r.size, 1)]


def evaluate_variant(
    space: RepresentationSpace,
    tuning: FingerprintSpace,
    tuning_normalized: FingerprintSpace,
    rate_absdiff: np.ndarray,
    *,
    n_perm: int,
    k_values: Sequence[int],
    seed: int,
    prediction_cfg: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Raw + rate-normalized + rate-matched + predictive metrics for one variant."""
    raw = geometry_function_analysis(
        space.X, tuning.X,
        n_perm=n_perm, k_values=k_values, seed=seed,
        rate_absdiff_condensed=rate_absdiff, nuisance_condensed=rate_absdiff,
        primary_return_null=True,
    )
    norm = geometry_function_analysis(
        space.X, tuning_normalized.X,
        n_perm=n_perm, k_values=[], seed=seed, include_curves=False,
        primary_return_null=True,
    )
    p_raw = raw["primary_mantel_spearman"]
    p_norm = norm["primary_mantel_spearman"]
    rm = raw.get("rate_matched_mantel", {})
    partial = raw.get("partial_mantel_controlling_for_nuisance", {})

    row: dict[str, Any] = {
        "n_features": len(space.feature_names),
        "n_informative_features": int(space.standardizer.n_informative),
        "feature_names": list(space.feature_names),
        "blocks": list(space.blocks),
        # raw geometry-function association (vs the primary tuning fingerprint)
        "raw_r": p_raw.get("statistic"),
        "raw_p": p_raw.get("p_value"),
        "raw_p_floor": p_raw.get("p_value_floor"),
        "raw_at_floor": p_raw.get("at_resolution_floor"),
        "raw_null_mean": p_raw.get("null_mean"),
        "raw_null_std": p_raw.get("null_std"),
        "raw_effect_size_z": p_raw.get("effect_size_z"),
        "raw_bootstrap_ci": p_raw.get("bootstrap_ci"),
        # rate-normalized fingerprint association (PRIMARY rate control)
        "rate_normalized_r": p_norm.get("statistic"),
        "rate_normalized_p": p_norm.get("p_value"),
        "rate_normalized_at_floor": p_norm.get("at_resolution_floor"),
        # rate-matched stratified Mantel (proper rate-matched control)
        "rate_matched_r": rm.get("statistic"),
        "rate_matched_p": rm.get("p_value"),
        "rate_matched_effect_size_z": rm.get("effect_size_z"),
        # partial Mantel: secondary / exploratory ONLY
        "partial_mantel_r_secondary_exploratory": partial.get("statistic"),
        "partial_mantel_p_secondary_exploratory": partial.get("p_value"),
        # supporting kNN
        "knn": raw.get("knn", {}).get("table", []),
    }

    # mandatory question: could this be explained simply by firing rate?
    rn = row["rate_normalized_r"]
    rmp = row["rate_matched_p"]
    rn_p = row["rate_normalized_p"]
    row["survives_rate_normalization"] = bool(
        rn is not None and np.isfinite(rn) and rn > 0 and rn_p is not None and rn_p < 0.05
    )
    row["survives_rate_matching"] = bool(
        row["rate_matched_r"] is not None and np.isfinite(row["rate_matched_r"])
        and row["rate_matched_r"] > 0 and rmp is not None and rmp < 0.05
    )
    row["rate_control_verdict"] = (
        "survives both rate controls"
        if (row["survives_rate_normalization"] and row["survives_rate_matching"])
        else (
            "survives rate normalization only"
            if row["survives_rate_normalization"]
            else (
                "survives rate matching only"
                if row["survives_rate_matching"]
                else "not distinguishable from a firing-rate effect"
            )
        )
    )

    pred_cfg = dict(prediction_cfg or {})
    if pred_cfg.get("enabled", True):
        pred = cross_validated_ridge(
            space.X, tuning.X,
            n_splits=int(pred_cfg.get("n_splits", 5)),
            alphas=tuple(pred_cfg.get("alphas", (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0))),
            seed=seed,
        )["metrics"]
        row["prediction_r2_mean"] = pred.get("r2_mean")
        row["prediction_pearson_r_mean"] = pred.get("pearson_r_mean")
        row["prediction_nrmse_mean"] = pred.get("nrmse_mean")
        row["prediction_r2_overall"] = pred.get("r2_overall")
    return {"row": row, "raw_analysis": raw, "norm_analysis": norm}


# --------------------------------------------------------------------------
# Condition runner
# --------------------------------------------------------------------------
def run_condition(
    condition: StressCondition,
    *,
    variants: Sequence[VariantSpec] | None = None,
    tuning_key: str = "tuning",
    tuning_normalized_key: str = "tuning_rate_normalized",
    n_perm: int = 1000,
    k_values: Sequence[int] = (3, 5, 10, 20),
    seed: int = 0,
    weighting: str = "equal",
    normalize_rows: bool = False,
    prediction_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate every representation/control for one network condition."""
    variants = list(variants or stress_variants())
    if tuning_key not in condition.fingerprints:
        raise KeyError(f"condition {condition.name!r} has no fingerprint {tuning_key!r}")
    if tuning_normalized_key not in condition.fingerprints:
        raise KeyError(f"condition {condition.name!r} has no fingerprint {tuning_normalized_key!r}")
    tuning = condition.fingerprints[tuning_key]
    tuning_norm = condition.fingerprints[tuning_normalized_key]
    rate_absdiff = _rate_absdiff_condensed(condition.activity)

    # reference "full" space used to match the random/shuffle controls' dimensionality
    full_blocks = list(FeatureBlock.structural()) + [FeatureBlock.ACTIVITY.value]
    full_space, _ = build_variant_space(
        VariantSpec(name="_full_reference", kind="blocks", blocks=full_blocks, use_activity=True),
        condition.structural, condition.activity,
        weighting=weighting, normalize_rows=normalize_rows,
    )

    rows: list[dict[str, Any]] = []
    analyses: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {
        "dy_tuning": tuning.condensed(),
        "dy_tuning_rate_normalized": tuning_norm.condensed(),
        "rate_absdiff": rate_absdiff,
    }
    for spec in variants:
        try:
            space, circular = build_variant_space(
                spec, condition.structural, condition.activity,
                condition.fingerprints,
                weighting=weighting, normalize_rows=normalize_rows,
                random_seed=seed, shuffle_seed=seed, full_space=full_space,
            )
        except EmptyFeatureSelectionError as exc:
            rows.append({
                "condition": condition.name,
                "variant": spec.name,
                "kind": spec.kind,
                "is_control": spec.is_control,
                "description": spec.description,
                "skipped": True,
                "skip_reason": str(exc),
                "n_features": 0,
            })
            continue

        out = evaluate_variant(
            space, tuning, tuning_norm, rate_absdiff,
            n_perm=n_perm, k_values=k_values, seed=seed, prediction_cfg=prediction_cfg,
        )
        row = out["row"]
        row.update({
            "condition": condition.name,
            "variant": spec.name,
            "kind": spec.kind,
            "is_control": spec.is_control,
            "circular__do_not_report_as_evidence": bool(circular),
            "description": spec.description,
            "skipped": False,
        })
        rows.append(row)
        null = out["raw_analysis"].get("_primary_null")
        if null is not None:
            arrays[f"null_{condition.name}__{spec.name}"] = np.asarray(null, dtype=np.float64)
        analyses[f"{condition.name}::{spec.name}"] = {
            k: v for k, v in out["raw_analysis"].items() if not str(k).startswith("_")
        }
    return {"rows": rows, "analyses": analyses, "arrays": arrays}


# --------------------------------------------------------------------------
# Full stress test
# --------------------------------------------------------------------------
def run_stress_test(
    conditions: Sequence[StressCondition],
    *,
    variants: Sequence[VariantSpec] | None = None,
    tuning_key: str = "tuning",
    tuning_normalized_key: str = "tuning_rate_normalized",
    n_perm: int = 1000,
    k_values: Sequence[int] = (3, 5, 10, 20),
    seed: int = 0,
    weighting: str = "equal",
    normalize_rows: bool = False,
    prediction_cfg: Mapping[str, Any] | None = None,
    before_condition: str = "before_learning",
    after_condition: str = "after_learning",
    reference_variant: str = "structural_full",
) -> dict[str, Any]:
    """Run every condition and assemble the single table + before/after summary."""
    all_rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    analyses: dict[str, Any] = {}
    for condition in conditions:
        res = run_condition(
            condition, variants=variants,
            tuning_key=tuning_key, tuning_normalized_key=tuning_normalized_key,
            n_perm=n_perm, k_values=k_values, seed=seed,
            weighting=weighting, normalize_rows=normalize_rows,
            prediction_cfg=prediction_cfg,
        )
        all_rows.extend(res["rows"])
        arrays.update(res["arrays"])
        analyses.update(res["analyses"])

    table = build_stress_table(all_rows, reference_variant=reference_variant)
    before_after = before_after_summary(all_rows, before=before_condition, after=after_condition)
    return {
        "rows": all_rows,
        "table": table,
        "before_after": before_after,
        "analyses": analyses,
        "arrays": arrays,
        "meta": {
            "tuning_key": tuning_key,
            "tuning_normalized_key": tuning_normalized_key,
            "n_perm": int(n_perm),
            "p_value_floor": 1.0 / (1.0 + n_perm),
            "seed": int(seed),
            "conditions": [c.name for c in conditions],
            "n_variants": len(list(variants or stress_variants())),
            "reference_variant": reference_variant,
            "no_cherry_picking": (
                "Every representation/control is reported, including negative and "
                "skipped results; the primary result is the structural representation, "
                "not the best row of the table."
            ),
            "rate_control_policy": (
                "Independence from firing rate is argued from the rate-normalized "
                "fingerprint and the rate-matched stratified Mantel; the partial Mantel "
                "is secondary/exploratory only."
            ),
        },
    }


# --------------------------------------------------------------------------
# The single table
# --------------------------------------------------------------------------
def build_stress_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    reference_variant: str = "structural_full",
) -> list[dict[str, Any]]:
    """Compact table: representation/control, raw, rate-normalized, predictive, significance."""
    table: list[dict[str, Any]] = []
    for r in rows:
        table.append({
            "representation_control": r.get("variant"),
            "condition": r.get("condition"),
            "n_features": r.get("n_features"),
            "raw_geometry_function_r": r.get("raw_r"),
            "rate_normalized_association_r": r.get("rate_normalized_r"),
            "rate_matched_association_r": r.get("rate_matched_r"),
            "predictive_metric": r.get("prediction_r2_mean"),
            "predictive_metric_name": "ridge_cv_r2_mean",
            "predictive_pearson_r": r.get("prediction_pearson_r_mean"),
            "permutation_significance_p": r.get("raw_p"),
            "permutation_p_value_floor": r.get("raw_p_floor"),
            "permutation_at_resolution_floor": r.get("raw_at_floor"),
            "is_control": r.get("is_control"),
            "is_reference": r.get("variant") == reference_variant,
            "rate_control_verdict": r.get("rate_control_verdict"),
            "skipped": r.get("skipped", False),
            "skip_reason": r.get("skip_reason"),
            "description": r.get("description"),
        })
    return table


def before_after_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    before: str = "before_learning",
    after: str = "after_learning",
) -> list[dict[str, Any]]:
    """Per-variant change in raw / rate-normalized association and prediction."""
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {
        (r["condition"], r["variant"]): r for r in rows if not r.get("skipped", False)
    }
    out: list[dict[str, Any]] = []
    variants = sorted({r["variant"] for r in rows if r.get("condition") == after})
    for variant in variants:
        rb = by_key.get((before, variant))
        ra = by_key.get((after, variant))
        if rb is None or ra is None:
            out.append({
                "variant": variant,
                "raw_r_before": (rb or {}).get("raw_r"),
                "raw_r_after": (ra or {}).get("raw_r"),
                "delta_raw_r": None,
                "rate_normalized_r_before": (rb or {}).get("rate_normalized_r"),
                "rate_normalized_r_after": (ra or {}).get("rate_normalized_r"),
                "delta_rate_normalized_r": None,
                "prediction_r2_before": (rb or {}).get("prediction_r2_mean"),
                "prediction_r2_after": (ra or {}).get("prediction_r2_mean"),
                "delta_prediction_r2": None,
                "note": "missing in one condition (e.g. empty block before learning)",
            })
            continue
        out.append({
            "variant": variant,
            "raw_r_before": rb.get("raw_r"),
            "raw_r_after": ra.get("raw_r"),
            "delta_raw_r": _nan_sub(ra.get("raw_r"), rb.get("raw_r")),
            "rate_normalized_r_before": rb.get("rate_normalized_r"),
            "rate_normalized_r_after": ra.get("rate_normalized_r"),
            "delta_rate_normalized_r": _nan_sub(ra.get("rate_normalized_r"), rb.get("rate_normalized_r")),
            "rate_matched_r_before": rb.get("rate_matched_r"),
            "rate_matched_r_after": ra.get("rate_matched_r"),
            "delta_rate_matched_r": _nan_sub(ra.get("rate_matched_r"), rb.get("rate_matched_r")),
            "prediction_r2_before": rb.get("prediction_r2_mean"),
            "prediction_r2_after": ra.get("prediction_r2_mean"),
            "delta_prediction_r2": _nan_sub(ra.get("prediction_r2_mean"), rb.get("prediction_r2_mean")),
            "note": "",
        })
    return out


def _nan_sub(a: Any, b: Any) -> float:
    try:
        a = float(a)
        b = float(b)
    except (TypeError, ValueError):
        return float("nan")
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return a - b


# --------------------------------------------------------------------------
# Leakage guard: the fingerprint split must not be the model-selection split
# --------------------------------------------------------------------------
def fingerprint_split_leakage_guard(
    checkpoint_extra: Mapping[str, Any] | None,
    split_info: Mapping[str, Any] | None,
    eval_split: str,
) -> dict[str, Any]:
    """Verify the fingerprint split is independent of model selection.

    The fingerprint is an evaluation target and must not be measured on the
    speakers used to *select* the checkpoint. This compares the model-selection
    speakers recorded in the checkpoint's provenance with the speakers of the split
    the fingerprint is measured on.

    Handles both the corrected 3-way split (model selection on ``dev``) and the
    legacy 2-way split (model selection on ``held_out_speakers``).
    """
    checkpoint_extra = dict(checkpoint_extra or {})
    split_info = dict(split_info or {})
    si = dict(checkpoint_extra.get("split_info", {}) or {})

    sel_speakers: list[int] | None = None
    sel_split = None
    if si.get("strategy") == "speaker_aware_train_dev_probe":
        sel_split = "dev"
        sel_speakers = list((si.get("splits", {}) or {}).get("dev", {}).get("speakers", []) or [])
    elif "held_out_speakers" in si:
        sel_split = "val (2-way split)"
        sel_speakers = list(si.get("held_out_speakers", []) or [])

    key = "dev" if eval_split in ("dev", "val") else eval_split
    fp_speakers = list((split_info.get("splits", {}) or {}).get(key, {}).get("speakers", []) or [])
    overlap = sorted(set(sel_speakers or []) & set(fp_speakers))

    passed: bool | None
    if sel_speakers is None or not fp_speakers:
        passed = None
        note = "could not determine the model-selection split from the recorded provenance"
    elif overlap:
        passed = False
        note = (
            f"LEAKAGE: the fingerprint split ('{eval_split}', speakers {fp_speakers}) overlaps "
            f"the model-selection {sel_split} speakers {sel_speakers}. The fingerprint is not "
            "independent of model selection - use the corrected 3-way-split checkpoint."
        )
    else:
        passed = True
        note = (
            f"OK: fingerprint split ('{eval_split}', speakers {fp_speakers}) is disjoint from the "
            f"model-selection {sel_split} speakers {sel_speakers}."
        )
    return {
        "fingerprint_split": eval_split,
        "fingerprint_split_speakers": fp_speakers,
        "model_selection_split": sel_split,
        "model_selection_speakers": sel_speakers,
        "overlap_speakers": overlap,
        "passed": passed,
        "note": note,
    }