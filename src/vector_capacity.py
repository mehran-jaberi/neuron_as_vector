"""Scientific capacity evaluation of the neuron-vector representation.

This module is a **thin layer**: it connects the composed neuron vectors
(``src/neuron_vector.py``) to the repository's existing scientific machinery and adds no
new statistics of its own.

```
FIT  (label-free)  ->  NeuronRecordBank -> StructuredVectorEncoder / learned residual -> (n, d)
PROBE (held out)   ->  functional targets (individual-stimulus response, class-rate, temporal)
                         |
                         v
   existing geometry (Mantel + rate-normalized + rate-matched + kNN)
   existing cross-validated prediction (RidgeCV / kNN on shared folds)
   existing controls (rate-only, random, neuron-shuffle)
```

Split discipline (hard):
* **FIT** is used only to construct the representation (record bank, structured encoder,
  residual training and its normalisation statistics, and the label-free firing-rate
  reference used by the rate controls).
* **PROBE** is used only to build functional targets and to compute evaluation metrics.
* The official **TEST** split is never read by this module or its script.

No representation is ever fitted or normalised on PROBE: representation matrices are treated
as frozen inputs to :func:`evaluate_condition`.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .evaluation import ActivityAccumulatorResult
from .functional_fingerprint import (
    FINGERPRINT_PRESETS,
    FingerprintConfig,
    FingerprintSpace,
    class_conditioned_fingerprint,
    fingerprint_definition,
    stimulus_response_config,
    stimulus_response_fingerprint,
)
from .geometry_analysis import geometry_function_analysis, primary_metric_row
from .neuron_record import NeuronRecordBank
from .neuron_vector import NeuronVectorError, build_neuron_vectors
from .prediction import (
    DEFAULT_ALPHAS,
    cross_validated_knn,
    cross_validated_ridge,
    make_shared_folds,
)
from .representations import RepresentationSpace, random_baseline_space, shuffle_control_space
from .residual import ResidualError, ResidualResult
from .structured_vector import StructuredVectorEncoder
from .v2_config import V2Config

#: Representation kinds.
KIND_STRUCTURED = "structured"
KIND_STRUCTURED_RESIDUAL = "structured_plus_residual"
KIND_CONTROL = "control"

#: Target names.
TARGET_PRIMARY = "stimulus_response"
TARGET_PRIMARY_RATE_NORMALIZED = "stimulus_response_rate_normalized"
TARGET_CLASS_RATE = "class_rate_20d"
TARGET_TEMPORAL = "temporal"

#: Canonical preprocessing (the repository's ``RepresentationSpace`` convention).
REPRESENTATION_WEIGHTING = "uniform"
REPRESENTATION_NORMALIZE_ROWS = False
REPRESENTATION_PREPROCESSING = (
    "RepresentationSpace(weighting='uniform', normalize_rows=False): column z-scoring, "
    "Euclidean distances; identical preprocessing for every condition"
)

#: Machine-readable result-table columns (order preserved).
RESULT_COLUMNS: tuple[str, ...] = (
    "representation",
    "representation_kind",
    "total_d",
    "structured_d",
    "residual_d",
    "residual_seed",
    "n_neurons",
    "probe_n",
    "primary_target",
    "primary_metric",
    "primary_metric_value",
    "primary_p_value",
    "primary_p_value_floor",
    "primary_effect_size_z",
    "primary_ci_low",
    "primary_ci_high",
    "primary_rate_normalized_r",
    "primary_rate_matched_r",
    "primary_knn_best_k",
    "primary_knn_effect_size_z",
    "class_rate_metric",
    "class_rate_metric_value",
    "temporal_metric",
    "temporal_metric_value",
    "prediction_target",
    "prediction_model",
    "prediction_metric",
    "prediction_metric_value",
    "prediction_r2_class_rate",
    "prediction_r2_temporal",
    "control_rate_only_primary_r",
    "checkpoint",
    "tag",
)

DEFAULT_STRUCTURED_DIMS: tuple[int, ...] = (32, 48, 64, 100, 128)
DEFAULT_RESIDUAL_CONDITIONS: tuple[tuple[int, int], ...] = ((48, 16), (48, 52))
DEFAULT_RESIDUAL_SEEDS: tuple[int, ...] = (0, 1, 2)
CONTROL_MATCHED_DIM = 100


class VectorCapacityError(ValueError):
    """Raised for an invalid capacity-evaluation request."""


# --------------------------------------------------------------------------
# Conditions
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RepresentationCondition:
    """One representation condition of the study."""

    kind: str
    total_d: int
    structured_d: int
    residual_d: int
    residual_seed: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in (KIND_STRUCTURED, KIND_STRUCTURED_RESIDUAL):
            raise VectorCapacityError(f"unknown representation kind {self.kind!r}")
        if self.total_d != self.structured_d + self.residual_d:
            raise VectorCapacityError(
                f"condition violates total_d = structured_d + residual_d: {self.to_dict()}"
            )
        if self.kind == KIND_STRUCTURED and self.residual_d != 0:
            raise VectorCapacityError("structured-only conditions must have residual_d = 0")
        if self.kind == KIND_STRUCTURED_RESIDUAL:
            if self.residual_d <= 0:
                raise VectorCapacityError("residual conditions need residual_d > 0")
            if self.residual_seed is None:
                raise VectorCapacityError("residual conditions must record their training seed")

    @property
    def label(self) -> str:
        """Stable, human-readable condition label used in tables and figures."""
        if self.kind == KIND_STRUCTURED:
            return f"structured_{self.total_d}"
        return f"full_{self.structured_d}+{self.residual_d}_seed{self.residual_seed}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "kind": self.kind,
            "total_d": int(self.total_d),
            "structured_d": int(self.structured_d),
            "residual_d": int(self.residual_d),
            "residual_seed": None if self.residual_seed is None else int(self.residual_seed),
        }


def default_conditions(
    structured_dims: Iterable[int] = DEFAULT_STRUCTURED_DIMS,
    residual_conditions: Iterable[tuple[int, int]] = DEFAULT_RESIDUAL_CONDITIONS,
    residual_seeds: Iterable[int] = DEFAULT_RESIDUAL_SEEDS,
) -> tuple[RepresentationCondition, ...]:
    """The study's conditions: structured-only dims plus structured+residual at matched totals."""
    conditions: list[RepresentationCondition] = [
        RepresentationCondition(KIND_STRUCTURED, int(d), int(d), 0) for d in structured_dims
    ]
    for structured_d, residual_d in residual_conditions:
        for seed in residual_seeds:
            conditions.append(
                RepresentationCondition(
                    KIND_STRUCTURED_RESIDUAL,
                    int(structured_d) + int(residual_d),
                    int(structured_d),
                    int(residual_d),
                    int(seed),
                )
            )
    return tuple(conditions)


# --------------------------------------------------------------------------
# Functional targets (PROBE only)
# --------------------------------------------------------------------------
@dataclass
class EvaluationTargets:
    """The functional targets used for evaluation (all built from PROBE)."""

    primary: FingerprintSpace
    primary_rate_normalized: FingerprintSpace
    class_rate: FingerprintSpace
    temporal: FingerprintSpace
    metadata: dict[str, Any] = field(default_factory=dict)

    def get(self, name: str) -> FingerprintSpace:
        mapping = {
            TARGET_PRIMARY: self.primary,
            TARGET_PRIMARY_RATE_NORMALIZED: self.primary_rate_normalized,
            TARGET_CLASS_RATE: self.class_rate,
            TARGET_TEMPORAL: self.temporal,
        }
        if name not in mapping:
            raise VectorCapacityError(f"unknown target {name!r}")
        return mapping[name]

    @property
    def n_neurons(self) -> int:
        return self.primary.n_neurons

    @property
    def n_stimuli(self) -> int:
        return int(self.primary.X.shape[1])

    def summary(self) -> dict[str, Any]:
        return {
            "n_neurons": self.n_neurons,
            "stimulus_response_dimension": int(self.primary.X.shape[1]),
            "class_rate_dimension": int(self.class_rate.X.shape[1]),
            "temporal_dimension": int(self.temporal.X.shape[1]),
            "primary_definition": self.metadata.get("primary_definition"),
            "definitions": {
                name: fingerprint_definition(space.config, dimension=int(space.X.shape[1]))
                for name, space in (
                    (TARGET_PRIMARY, self.primary),
                    (TARGET_PRIMARY_RATE_NORMALIZED, self.primary_rate_normalized),
                    (TARGET_CLASS_RATE, self.class_rate),
                    (TARGET_TEMPORAL, self.temporal),
                )
            },
            "metadata": dict(self.metadata),
        }


def build_evaluation_targets(
    probe_result: ActivityAccumulatorResult,
    *,
    probe_split_label: str = "probe",
    n_psth_bins: int = 10,
    min_spikes_for_latency: float = 1.0,
) -> EvaluationTargets:
    """Build all functional targets from a **single** labelled PROBE activity pass.

    * primary: per-neuron response to each individual stimulus (``(n_neurons, n_probe)``)
    * primary rate control: the same target with row-normalised (magnitude-removed) profiles
    * secondary: the existing 20-D class-conditioned rate fingerprint (``tuning`` preset)
    * exploratory: the existing temporal preset (PSTH + temporal centre/dispersion + latency)
    """
    if not isinstance(probe_result, ActivityAccumulatorResult):
        raise VectorCapacityError(
            f"PROBE targets need an ActivityAccumulatorResult, got {type(probe_result).__name__}"
        )
    split = str(probe_split_label).lower()
    if "train" in split or "fit" in split:
        raise VectorCapacityError(
            f"evaluation targets must come from a held-out PROBE split, got {probe_split_label!r}"
        )

    X_primary, names_primary = stimulus_response_fingerprint(
        probe_result.counts, bin_ms=probe_result.bin_ms, n_bins=probe_result.n_bins
    )
    primary = FingerprintSpace(
        X_raw=X_primary,
        feature_names=names_primary,
        config=stimulus_response_config(eval_split=probe_split_label),
        meta={
            "target": TARGET_PRIMARY,
            "split": probe_split_label,
            "n_stimuli": int(X_primary.shape[1]),
            "response_definition": "mean firing rate (Hz) per individual stimulus = count / duration",
            "averaging": "none across stimuli; each evaluated sample stays a separate column",
        },
    )
    primary_normalized = FingerprintSpace(
        X_raw=X_primary,
        feature_names=names_primary,
        config=stimulus_response_config(
            eval_split=probe_split_label, normalize_rows=True, standardize="none"
        ),
        meta={
            "target": TARGET_PRIMARY_RATE_NORMALIZED,
            "split": probe_split_label,
            "note": "response patterns only: per-neuron profiles L2-normalised (magnitude removed)",
        },
    )

    presets = {
        TARGET_CLASS_RATE: list(FINGERPRINT_PRESETS["tuning"]),
        TARGET_TEMPORAL: list(FINGERPRINT_PRESETS["temporal"]),
    }
    built: dict[str, FingerprintSpace] = {}
    for name, feature_sets in presets.items():
        X, feature_names = class_conditioned_fingerprint(
            probe_result.class_psth,
            probe_result.class_counts,
            probe_result.class_n,
            probe_result.class_first_spike_sum,
            probe_result.class_first_spike_count,
            n_bins=probe_result.n_bins,
            bin_ms=probe_result.bin_ms,
            feature_sets=feature_sets,
            n_psth_bins=n_psth_bins,
            min_spikes_for_latency=min_spikes_for_latency,
        )
        config = FingerprintConfig(
            feature_sets=feature_sets,
            standardize="column",
            normalize_rows=False,
            metric="euclidean",
            n_psth_bins=n_psth_bins,
            min_spikes_for_latency=min_spikes_for_latency,
            eval_split=probe_split_label,
        )
        built[name] = FingerprintSpace(
            X_raw=X,
            feature_names=feature_names,
            config=config,
            meta={"target": name, "split": probe_split_label, "preset_feature_sets": feature_sets},
        )

    metadata = {
        "probe_split": probe_split_label,
        "probe_n_samples": int(probe_result.n_samples),
        "n_bins": int(probe_result.n_bins),
        "bin_ms": float(probe_result.bin_ms),
        "primary_definition": (
            "individual-stimulus response profile: counts[s, h] / (n_bins * bin_ms / 1000) Hz, "
            "one column per PROBE utterance (no averaging across stimuli)"
        ),
        "primary_uses_labels": False,
        "class_rate_target_uses_labels": True,
        "temporal_target_uses_labels": True,
        "labels_used_only_in_targets": True,
    }
    return EvaluationTargets(
        primary=primary,
        primary_rate_normalized=primary_normalized,
        class_rate=built[TARGET_CLASS_RATE],
        temporal=built[TARGET_TEMPORAL],
        metadata=metadata,
    )


def fit_rate_reference(fit_result: ActivityAccumulatorResult) -> np.ndarray:
    """Label-free FIT firing rates (Hz) per neuron - used by the rate controls."""
    counts = np.asarray(fit_result.counts, dtype=np.float64)
    duration_s = max(float(fit_result.duration_s), 1e-9)
    return counts.sum(axis=0) / (max(int(fit_result.n_samples), 1) * duration_s)


def rate_absdiff_condensed(rates: np.ndarray) -> np.ndarray:
    """Condensed ``|rate_i - rate_j|`` (identical definition to ``abs_rate_difference_condensed``)."""
    r = np.asarray(rates, dtype=np.float64).reshape(-1, 1)
    return np.abs(r - r.T)[np.triu_indices(r.size, 1)]


# --------------------------------------------------------------------------
# Evaluation settings and one condition's evaluation
# --------------------------------------------------------------------------
@dataclass
class EvaluationSettings:
    """All evaluation knobs, recorded verbatim in the output metadata."""

    n_perm: int = 2000
    bootstrap: int = 500
    k_values: tuple[int, ...] = (3, 5, 10, 20)
    seed: int = 0
    n_splits: int = 5
    knn_k: int = 5
    alphas: tuple[float, ...] = tuple(DEFAULT_ALPHAS)
    random_control_seed: int = 12345
    shuffle_control_seed: int = 777
    rate_matched_strata: int = 5
    include_curves: bool = False
    checkpoint: str = ""
    tag: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_perm": int(self.n_perm),
            "bootstrap": int(self.bootstrap),
            "k_values": [int(k) for k in self.k_values],
            "seed": int(self.seed),
            "n_splits": int(self.n_splits),
            "knn_k": int(self.knn_k),
            "alphas": [float(a) for a in self.alphas],
            "random_control_seed": int(self.random_control_seed),
            "shuffle_control_seed": int(self.shuffle_control_seed),
            "rate_matched_strata": int(self.rate_matched_strata),
            "include_curves": bool(self.include_curves),
            "permutation_procedure": "neuron relabelling (Mantel), one-sided greater",
            "bootstrap_procedure": "neuron resampling with replacement (existing implementation)",
            "prediction_cv": "KFold over neurons, shared folds across conditions/targets",
            "checkpoint": self.checkpoint,
            "tag": self.tag,
        }


def representation_space(
    X: np.ndarray,
    feature_names: Sequence[str],
) -> RepresentationSpace:
    """Wrap a frozen representation matrix in the canonical preprocessing space."""
    matrix = np.asarray(X, dtype=np.float64)
    if matrix.ndim != 2:
        raise VectorCapacityError(f"representation matrix must be 2-D, got {matrix.shape}")
    if matrix.shape[1] != len(feature_names):
        raise VectorCapacityError(
            f"representation has {matrix.shape[1]} columns but {len(feature_names)} feature names"
        )
    return RepresentationSpace(
        X_raw=matrix,
        feature_names=list(feature_names),
        weighting=REPRESENTATION_WEIGHTING,
        normalize_rows=REPRESENTATION_NORMALIZE_ROWS,
    )


def _geometry(
    space_matrix: np.ndarray,
    target: FingerprintSpace,
    *,
    settings: EvaluationSettings,
    rates_absdiff: np.ndarray | None,
    k_values: Sequence[int],
    include_curves: bool,
) -> dict[str, Any]:
    return geometry_function_analysis(
        space_matrix,
        target.X,
        n_perm=settings.n_perm,
        k_values=k_values,
        seed=settings.seed,
        rate_absdiff_condensed=rates_absdiff,
        nuisance_condensed=rates_absdiff,
        rate_matched_strata=settings.rate_matched_strata,
        primary_bootstrap=settings.bootstrap,
        include_curves=include_curves,
    )


def evaluate_condition(
    X: np.ndarray,
    feature_names: Sequence[str],
    targets: EvaluationTargets,
    *,
    settings: EvaluationSettings,
    rates_absdiff: np.ndarray | None = None,
    collect_curve: bool = False,
    description: Mapping[str, Any] | None = None,
    cv: Any | None = None,
) -> dict[str, Any]:
    """Evaluate one frozen representation matrix against every target (existing machinery)."""
    space = representation_space(X, feature_names)
    if rates_absdiff is not None and np.asarray(rates_absdiff).size != space.condensed().size:
        raise VectorCapacityError(
            "rate-difference control size does not match the representation's pair count"
        )

    primary = _geometry(
        space.X, targets.primary, settings=settings, rates_absdiff=rates_absdiff,
        k_values=settings.k_values, include_curves=collect_curve,
    )
    primary_normalized = _geometry(
        space.X, targets.primary_rate_normalized, settings=settings, rates_absdiff=None,
        k_values=(), include_curves=False,
    )
    class_rate = _geometry(
        space.X, targets.class_rate, settings=settings, rates_absdiff=rates_absdiff,
        k_values=(), include_curves=False,
    )
    temporal = _geometry(
        space.X, targets.temporal, settings=settings, rates_absdiff=None,
        k_values=(), include_curves=False,
    )

    n = space.n_neurons
    cv = cv if cv is not None else make_shared_folds(n, settings.n_splits, settings.seed)
    ridge_primary = cross_validated_ridge(
        space.X, targets.primary.X, n_splits=settings.n_splits, alphas=settings.alphas,
        seed=settings.seed, cv=cv,
    )["metrics"]
    knn_primary = cross_validated_knn(
        space.X, targets.primary.X, n_splits=settings.n_splits, k=settings.knn_k,
        seed=settings.seed, cv=cv,
    )["metrics"]
    ridge_class = cross_validated_ridge(
        space.X, targets.class_rate.X, n_splits=settings.n_splits, alphas=settings.alphas,
        seed=settings.seed, cv=cv,
    )["metrics"]
    ridge_temporal = cross_validated_ridge(
        space.X, targets.temporal.X, n_splits=settings.n_splits, alphas=settings.alphas,
        seed=settings.seed, cv=cv,
    )["metrics"]

    headline = primary_metric_row("", primary)
    rm = primary.get("rate_matched_mantel", {})

    row: dict[str, Any] = {
        "n_neurons": int(n),
        "probe_n": int(targets.n_stimuli),
        "primary_target": TARGET_PRIMARY,
        "primary_metric": "mantel_spearman_r",
        "primary_metric_value": headline.get("primary_metric_mantel_spearman_r"),
        "primary_p_value": headline.get("primary_metric_p_value"),
        "primary_p_value_floor": headline.get("primary_metric_p_value_floor"),
        "primary_effect_size_z": headline.get("primary_metric_effect_size_z"),
        "primary_ci_low": headline.get("primary_metric_ci_low"),
        "primary_ci_high": headline.get("primary_metric_ci_high"),
        "primary_rate_normalized_r": primary_normalized.get("primary_mantel_spearman", {}).get("statistic"),
        "primary_rate_matched_r": rm.get("statistic"),
        "primary_rate_matched_p_value": rm.get("p_value"),
        "primary_rate_matched_method": rm.get("method"),
        "primary_pearson_r": headline.get("pearson_r"),
        "primary_knn_best_k": headline.get("best_knn_k"),
        "primary_knn_effect_size_z": headline.get("best_knn_effect_size_z"),
        "primary_knn_p_value": headline.get("best_knn_p_value"),
        "class_rate_metric": "mantel_spearman_r",
        "class_rate_metric_value": class_rate.get("primary_mantel_spearman", {}).get("statistic"),
        "class_rate_p_value": class_rate.get("primary_mantel_spearman", {}).get("p_value"),
        "temporal_metric": "mantel_spearman_r",
        "temporal_metric_value": temporal.get("primary_mantel_spearman", {}).get("statistic"),
        "prediction_target": TARGET_PRIMARY,
        "prediction_model": "ridge_cv",
        "prediction_metric": "cv_r2_mean",
        "prediction_metric_value": ridge_primary.get("r2_mean"),
        "prediction_pearson_r": ridge_primary.get("pearson_r_mean"),
        "prediction_r2_class_rate": ridge_class.get("r2_mean"),
        "prediction_r2_temporal": ridge_temporal.get("r2_mean"),
        "prediction_knn_pearson_r": knn_primary.get("pearson_r_mean"),
        "control_rate_only_primary_r": None,  # filled only for the rate-only control row
        "n_features": int(space.X.shape[1]),
    }
    if description:
        row.update(dict(description))
    if collect_curve and "_distance_curve" in primary:
        row["_distance_curve"] = primary["_distance_curve"]
        row["_primary_null"] = primary.get("_primary_null")
    return row


def condition_description(condition: RepresentationCondition, settings: EvaluationSettings) -> dict[str, Any]:
    return {
        "representation": condition.label,
        "representation_kind": condition.kind,
        "total_d": int(condition.total_d),
        "structured_d": int(condition.structured_d),
        "residual_d": int(condition.residual_d),
        "residual_seed": None if condition.residual_seed is None else int(condition.residual_seed),
        "checkpoint": settings.checkpoint,
        "tag": settings.tag,
    }


# --------------------------------------------------------------------------
# Representation construction (FIT only)
# --------------------------------------------------------------------------
def select_residual(
    residuals: Mapping[Any, Any] | None,
    condition: RepresentationCondition,
) -> ResidualResult:
    """Resolve the frozen residual artifact for a condition, rejecting mismatches.

    Two mapping shapes are accepted so the caller cannot silently use the wrong
    artifact:

    * ``{residual_d: {seed: ResidualResult}}`` - the capacity-study convention, one
      trained artifact per ``(residual dimension, seed)`` pair. This is what
      :mod:`scripts.evaluate_vector_capacity` produces.
    * ``{seed: ResidualResult}`` - single-family convenience; only unambiguous when
      the condition's ``residual_d`` matches the artifact, which is always verified.

    The resolved artifact's ``residual_dim`` must equal the condition's ``residual_d``;
    the trained residual never enters as extra structured/projection coordinates.
    """
    seed = condition.residual_seed
    nested = residuals.get(condition.residual_d) if isinstance(residuals, Mapping) else None
    if isinstance(nested, Mapping):
        residual = nested.get(seed)
    elif isinstance(residuals, Mapping):
        residual = residuals.get(seed)
    else:
        residual = None
    if not isinstance(residual, ResidualResult):
        raise VectorCapacityError(
            f"condition {condition.label} needs a trained residual artifact for "
            f"residual_d={condition.residual_d}, seed={seed} (mapping keyed by residual "
            "dimension then seed, or by seed for a single-family study)"
        )
    if int(residual.residual_dim) != int(condition.residual_d):
        raise VectorCapacityError(
            f"condition {condition.label} expects a residual with residual_dim="
            f"{condition.residual_d} but the supplied artifact has {residual.residual_dim}"
        )
    return residual


def condition_matrix(
    condition: RepresentationCondition,
    bank: NeuronRecordBank,
    *,
    residuals: Mapping[Any, Any] | None = None,
    enabled_blocks: Sequence[str] | None = None,
    chunk_size: int | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Materialise the frozen representation matrix for one condition (FIT-derived only)."""
    if condition.kind == KIND_STRUCTURED:
        vectors = StructuredVectorEncoder(
            bank,
            structured_d=condition.structured_d,
            enabled_blocks=enabled_blocks,
            chunk_size=chunk_size,
        ).encode()
        return np.asarray(vectors.X, dtype=np.float64), tuple(vectors.feature_names)
    residual = select_residual(residuals, condition)
    try:
        vectors = build_neuron_vectors(
            bank,
            structured_d=condition.structured_d,
            residual=residual,
            enabled_blocks=enabled_blocks,
            chunk_size=chunk_size,
        )
    except NeuronVectorError as exc:
        raise VectorCapacityError(str(exc)) from exc
    return np.asarray(vectors.X, dtype=np.float64), tuple(vectors.feature_names)


# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------
def control_rows(
    *,
    fit_rates: np.ndarray,
    targets: EvaluationTargets,
    settings: EvaluationSettings,
    rates_absdiff: np.ndarray | None,
    matched_dim: int = CONTROL_MATCHED_DIM,
    shuffle_reference: tuple[np.ndarray, Sequence[str]] | None = None,
    cv: Any | None = None,
) -> list[dict[str, Any]]:
    """The established controls, evaluated with the identical PROBE procedure.

    * ``control_rate_only`` - the label-free FIT firing rate as a 1-D representation
    * ``control_random_{d}`` - i.i.d. Gaussian representation matched in dimension
    * ``control_neuron_shuffle_{d}`` - the struct&shuffle control (neuron-row permutation)
    """
    rates = np.asarray(fit_rates, dtype=np.float64).reshape(-1, 1)
    rows: list[dict[str, Any]] = []

    rate_space = representation_space(rates, ["activity.rate_hz"])
    row = evaluate_condition(
        rate_space.X, ["activity.rate_hz"], targets, settings=settings, rates_absdiff=rates_absdiff,
        cv=cv,
    )
    row.update({
        "representation": "control_rate_only",
        "representation_kind": KIND_CONTROL,
        "total_d": 1,
        "structured_d": None,
        "residual_d": None,
        "residual_seed": None,
        "control_rate_only_primary_r": row.get("primary_metric_value"),
        "checkpoint": settings.checkpoint,
        "tag": settings.tag,
    })
    rows.append(row)

    random_space = random_baseline_space(
        rates.shape[0], int(matched_dim), seed=settings.random_control_seed, weighting="uniform",
        meta={"control": "random", "matched_dim": int(matched_dim)},
    )
    row = evaluate_condition(
        random_space.X, random_space.feature_names, targets, settings=settings,
        rates_absdiff=rates_absdiff, cv=cv,
    )
    row.update({
        "representation": f"control_random_{int(matched_dim)}",
        "representation_kind": KIND_CONTROL,
        "total_d": int(matched_dim),
        "structured_d": None,
        "residual_d": None,
        "residual_seed": None,
        "control_rate_only_primary_r": None,
        "checkpoint": settings.checkpoint,
        "tag": settings.tag,
    })
    rows.append(row)

    if shuffle_reference is not None:
        X_ref, names_ref = shuffle_reference
        reference_space = representation_space(X_ref, names_ref)
        shuffled = shuffle_control_space(reference_space, seed=settings.shuffle_control_seed)
        row = evaluate_condition(
            shuffled.X, shuffled.feature_names, targets, settings=settings,
            rates_absdiff=rates_absdiff, cv=cv,
        )
        row.update({
            "representation": f"control_neuron_shuffle_{int(X_ref.shape[1])}",
            "representation_kind": KIND_CONTROL,
            "total_d": int(X_ref.shape[1]),
            "structured_d": None,
            "residual_d": None,
            "residual_seed": None,
            "control_rate_only_primary_r": None,
            "checkpoint": settings.checkpoint,
            "tag": settings.tag,
        })
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Study orchestration
# --------------------------------------------------------------------------
def run_capacity_study(
    bank: NeuronRecordBank,
    targets: EvaluationTargets,
    conditions: Sequence[RepresentationCondition],
    *,
    settings: EvaluationSettings,
    fit_rates: np.ndarray,
    residuals: Mapping[int, ResidualResult] | None = None,
    enabled_blocks: Sequence[str] | None = None,
    chunk_size: int | None = None,
    controls: bool = True,
    curve_for: str | None = "structured_48",
) -> dict[str, Any]:
    """Evaluate every condition (and the controls) with the identical PROBE procedure."""
    rates_absdiff = rate_absdiff_condensed(fit_rates)
    cv = make_shared_folds(targets.n_neurons, settings.n_splits, settings.seed)

    rows: list[dict[str, Any]] = []
    curves: dict[str, Any] = {}
    matrices: dict[str, tuple[np.ndarray, tuple[str, ...]]] = {}
    for condition in conditions:
        X, names = condition_matrix(
            condition, bank, residuals=residuals, enabled_blocks=enabled_blocks, chunk_size=chunk_size
        )
        if X.shape[1] != condition.total_d:
            raise VectorCapacityError(
                f"condition {condition.label} produced {X.shape[1]} features, expected {condition.total_d}"
            )
        matrices[condition.label] = (X, names)
        collect_curve = curve_for is not None and condition.label == curve_for
        row = evaluate_condition(
            X, names, targets, settings=settings, rates_absdiff=rates_absdiff,
            collect_curve=collect_curve, description=condition_description(condition, settings), cv=cv,
        )
        if "_distance_curve" in row:
            curves[condition.label] = row.pop("_distance_curve")
        rows.append(row)

    control_row_list: list[dict[str, Any]] = []
    if controls:
        shuffle_reference = matrices.get(f"structured_{CONTROL_MATCHED_DIM}")
        control_row_list = control_rows(
            fit_rates=fit_rates,
            targets=targets,
            settings=settings,
            rates_absdiff=rates_absdiff,
            shuffle_reference=shuffle_reference,
            cv=cv,
        )

    return {
        "rows": rows,
        "controls": control_row_list,
        "curves": curves,
        "settings": settings.to_dict(),
        "targets": targets.summary(),
        "preprocessing": REPRESENTATION_PREPROCESSING,
    }


# --------------------------------------------------------------------------
# Aggregation / serialisation
# --------------------------------------------------------------------------
def summarise_across_seeds(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Mean/std/per-seed summary for the residual conditions (never collapsing the seeds)."""
    metrics = (
        "primary_metric_value",
        "primary_rate_normalized_r",
        "primary_rate_matched_r",
        "class_rate_metric_value",
        "temporal_metric_value",
        "prediction_metric_value",
    )
    groups: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("representation_kind") != KIND_STRUCTURED_RESIDUAL:
            continue
        key = (int(row["structured_d"]), int(row["residual_d"]))
        groups.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for (structured_d, residual_d), group in sorted(groups.items()):
        entry: dict[str, Any] = {
            "representation_family": f"full_{structured_d}+{residual_d}",
            "representation_kind": KIND_STRUCTURED_RESIDUAL,
            "structured_d": structured_d,
            "residual_d": residual_d,
            "total_d": structured_d + residual_d,
            "n_seeds": len(group),
            "residual_seeds": [row.get("residual_seed") for row in group],
        }
        for metric in metrics:
            values = np.array(
                [float(row[metric]) for row in group if row.get(metric) is not None and np.isfinite(float(row[metric]))],
                dtype=np.float64,
            )
            entry[f"{metric}_mean"] = float(values.mean()) if values.size else None
            entry[f"{metric}_std"] = float(values.std(ddof=0)) if values.size else None
            entry[f"{metric}_values"] = [float(v) for v in values]
        summary.append(entry)

    structured_rows = [row for row in rows if row.get("representation_kind") == KIND_STRUCTURED]
    for row in structured_rows:
        summary.append({
            "representation_family": row["representation"],
            "representation_kind": KIND_STRUCTURED,
            "structured_d": row["structured_d"],
            "residual_d": 0,
            "total_d": row["total_d"],
            "n_seeds": 1,
            "residual_seeds": [],
            **{
                f"{metric}_mean": row.get(metric)
                for metric in metrics
            },
            **{
                f"{metric}_std": 0.0
                for metric in metrics
            },
            **{
                f"{metric}_values": [row.get(metric)]
                for metric in metrics
            },
        })
    return summary


def public_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Drop private (``_``-prefixed) values so a row is JSON/CSV friendly."""
    return {str(k): v for k, v in row.items() if not str(k).startswith("_")}


def write_results(result: Mapping[str, Any], *, csv_path: str | Path, json_path: str | Path) -> dict[str, Path]:
    """Write the machine-readable result table (CSV) and the full payload (JSON)."""
    rows = [public_row(r) for r in list(result.get("rows", [])) + list(result.get("controls", []))]
    csv_path, json_path = Path(csv_path), Path(json_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    for name in RESULT_COLUMNS:
        fieldnames.append(name)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fieldnames})

    payload = {
        "rows": rows,
        "summary_across_seeds": summarise_across_seeds(rows),
        "settings": result.get("settings"),
        "targets": result.get("targets"),
        "preprocessing": result.get("preprocessing"),
        "metadata": result.get("metadata"),
        "result_columns": list(RESULT_COLUMNS),
    }
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
    return {"csv": csv_path, "json": json_path}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


__all__ = [
    "KIND_STRUCTURED",
    "KIND_STRUCTURED_RESIDUAL",
    "KIND_CONTROL",
    "TARGET_PRIMARY",
    "TARGET_PRIMARY_RATE_NORMALIZED",
    "TARGET_CLASS_RATE",
    "TARGET_TEMPORAL",
    "RESULT_COLUMNS",
    "DEFAULT_STRUCTURED_DIMS",
    "DEFAULT_RESIDUAL_CONDITIONS",
    "DEFAULT_RESIDUAL_SEEDS",
    "CONTROL_MATCHED_DIM",
    "REPRESENTATION_PREPROCESSING",
    "VectorCapacityError",
    "RepresentationCondition",
    "EvaluationTargets",
    "EvaluationSettings",
    "default_conditions",
    "build_evaluation_targets",
    "fit_rate_reference",
    "rate_absdiff_condensed",
    "representation_space",
    "evaluate_condition",
    "select_residual",
    "condition_matrix",
    "condition_description",
    "control_rows",
    "run_capacity_study",
    "summarise_across_seeds",
    "public_row",
    "write_results",
]