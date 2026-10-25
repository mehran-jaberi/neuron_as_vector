"""Rate-confound / robustness analysis of the neuron-vector capacity evaluation.

Why this module exists
----------------------
The first capacity study measured a *positive* Mantel correspondence between the frozen
neuron-vector geometry and the individual-stimulus PROBE response target
(``r ≈ +0.11 … +0.13``) — but the 1-D **FIT rate-only** control reached ``r = +0.663``.
Before any new biological feature block is added, the evaluation must be able to separate
*global firing-rate tendency* from *stimulus-specific response structure*.

The headline target is ``R[h, s] / (pdist over standardised columns)`` where ``R`` is the
per-neuron firing rate for individual PROBE utterances. It standardises **across neurons
within each stimulus column**, which removes the *column* mean, not each neuron's own mean
over stimuli. The consequence is testable and is what this module quantifies: the per-neuron
mean of the column-standardised headline target still correlates ~0.999 with the raw
per-neuron rate on the canonical checkpoint.

What it does
------------
It decomposes the response target into explicitly ordered pipelines of pure transforms
(:data:`ResponseTargetVariant`) —

.. code-block:: text

    raw            : R                     -> column_standardise
    neuron_centered: R -> neuron_center     -> column_standardise
    neuron_zscored : R -> neuron_zscore     -> column_standardise
    mean_rate      : R -> mean_over_stimuli -> column_standardise
    cs_then_center : R -> column_standardise -> neuron_center   (ordering sensitivity)
    row_l2         : R -> row_l2_normalise                      (existing 1st-study control)

— and evaluates a focused set of representations against every variant with the **same**
machinery as the first study (:func:`src.vector_capacity.geometry_for_target`, i.e.
:func:`src.geometry_analysis.geometry_function_analysis`: identical Mantel procedure,
permutation count, seed and bootstrap convention), plus a compact set of existing controls.

Semantics of the pre-existing controls (verified in code, not inferred from names)
----------------------------------------------------------------------------------
* ``control_rate_only`` (representation side, first study) — the representation matrix is the
  **FIT** mean rate per neuron (Hz), wrapped in :class:`~src.representations.RepresentationSpace`,
  which column z-scores it; its Euclidean condensed distance is therefore
  ``|rate_i - rate_j| / std_FIT``. The statistic is the Spearman correlation between
  *that* distance vector and the target's distance vector. It is **not** "correlate rates
  with responses"; it asks how much FIT-rate geometry mimics the target geometry.
* ``primary_rate_normalized`` (target side, first study) — the target is the raw response with
  each neuron's profile L2-normalised (``FingerprintSpace(normalize_rows=True,
  standardize="none")``); magnitude (L2 norm) is removed, not the mean. It is reproduced here
  exactly as :data:`TARGET_ROW_L2` (pipeline ``[row_l2_normalise]``).
* ``rate_matched`` (pair-stratified control) — pairs are binned into 5 quantile strata of
  ``|rate_i - rate_j|`` (FIT rates) and the Mantel Spearman is computed *within* each stratum,
  combined by sample-size-weighted Fisher z with the strata held fixed under the permutation
  null. Removes the between-rate component of the comparison.
* ``partial_mantel`` (exploratory, secondary) — residualises the ranks of both distance
  vectors on the ranks of ``|rate_i - rate_j|`` before correlating. It is *not* proof of
  rate-independence and is reported only as a secondary number.
* ``mean_rate`` (this module, target side) — a 1-D *functional target* ``mean_s R[h, s]``.
  This is a different object from ``control_rate_only``: there the rate is the representation
  and the target is the response profile; here the rate *is* the target and the
  representation geometry is compared against it.

Discipline
----------
FIT builds the representation (bank, residual training); PROBE builds every target variant;
TEST is never read. All target transforms are evaluation-only transformations of the PROBE
response matrix and never enter representation construction. Every transform operates on the
2-D ``(n_neurons, n_stimuli)`` matrix; no time-resolved tensor is created.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .evaluation import ActivityAccumulatorResult
from .functional_fingerprint import (
    COLUMN_STANDARDISE_STEP,
    MEAN_OVER_STIMULI_STEP,
    NEURON_CENTER_STEP,
    NEURON_ZSCORE_STEP,
    ROW_L2_NORMALISE_STEP,
    STIMULUS_RESPONSE_FEATURE_SET,
    ZERO_VARIANCE_STD_EPS,
    FingerprintConfig,
    FingerprintSpace,
    apply_response_pipeline,
    stimulus_response_matrix,
)
from .geometry_analysis import primary_metric_row
from .neuron_record import NeuronRecordBank
from .prediction import cross_validated_ridge, make_shared_folds
from .representations import random_baseline_space, shuffle_control_space
from .residual import ResidualResult
from .structured_vector import StructuredVectorEncoder
from .vector_capacity import (
    KIND_CONTROL,
    KIND_STRUCTURED,
    KIND_STRUCTURED_RESIDUAL,
    REPRESENTATION_PREPROCESSING,
    EvaluationSettings,
    RepresentationCondition,
    VectorCapacityError,
    assert_held_out_split,
    condition_matrix,
    geometry_for_target,
    rate_absdiff_condensed,
    representation_space,
)
from .v2_config import DEFAULT_ENABLED_BLOCKS

#: Provenance schema of a rate-robustness payload.
SCHEMA = "neuron_vector_rate_robustness/v1"

#: Target variant names (stable identifiers used in every table/figure).
TARGET_RAW = "raw"
TARGET_NEURON_CENTERED = "neuron_centered"
TARGET_NEURON_ZSCORED = "neuron_zscored"
TARGET_MEAN_RATE = "mean_rate"
TARGET_CS_THEN_CENTERED = "column_standardised_then_neuron_centered"
TARGET_ROW_L2 = "row_l2_normalised"

#: Roles of a target variant.
ROLE_MAIN = "main"
ROLE_ORDERING_SENSITIVITY = "ordering_sensitivity"
ROLE_EXISTING_CONTROL = "existing_control"

#: Roles of a representation condition.
REP_ROLE_REPRESENTATION = "representation"
REP_ROLE_ACTIVITY = "activity_diagnostic"
REP_ROLE_CONTROL = "representation_control"

#: Dimension of the representation used for the shuffle control (matches the first study).
CONTROL_MATCHED_DIM = 100

#: The focused representation set (the first study's full sweep is deliberately not repeated).
DEFAULT_STRUCTURED_DIMS: tuple[int, ...] = (48, 64, 100)
DEFAULT_RESIDUAL_CONDITIONS: tuple[tuple[int, int], ...] = ((48, 16), (48, 52))
DEFAULT_RESIDUAL_SEEDS: tuple[int, ...] = (0, 1, 2)
ACTIVITY_BLOCK = "activity"


class RateRobustnessError(ValueError):
    """Raised for an invalid rate-robustness request or an incompatible checkpoint."""


# --------------------------------------------------------------------------
# Target variants
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ResponseTargetVariant:
    """One explicitly ordered response-target pipeline.

    ``steps`` is applied left-to-right to the raw ``(n_neurons, n_stimuli)`` response
    matrix. ``centering``/``scaling`` are machine-readable summaries of which nuisance
    components the pipeline removes, and ``role`` separates the main decomposition from the
    ordering-sensitivity and existing-control rows.
    """

    name: str
    steps: tuple[str, ...]
    centering: str
    scaling: str
    role: str
    description: str
    bootstrap: bool = True
    predict: bool = False
    rate_matched: bool = False
    k_values: tuple[int, ...] = ()

    @property
    def pipeline(self) -> str:
        return " -> ".join(self.steps) if self.steps else "(identity)"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "steps": list(self.steps),
            "pipeline": self.pipeline,
            "centering": self.centering,
            "scaling": self.scaling,
            "role": self.role,
            "description": self.description,
            "bootstrap": bool(self.bootstrap),
            "predict": bool(self.predict),
            "rate_matched": bool(self.rate_matched),
            "k_values": [int(k) for k in self.k_values],
        }


#: The target decomposition of this study, in reporting order.
TARGET_VARIANTS: tuple[ResponseTargetVariant, ...] = (
    ResponseTargetVariant(
        name=TARGET_RAW,
        steps=(COLUMN_STANDARDISE_STEP,),
        centering="none",
        scaling="column_std",
        role=ROLE_MAIN,
        description=(
            "the existing headline first-study target: R (Hz) with column z-scoring across "
            "neurons; keeps each neuron's overall level"
        ),
        bootstrap=True,
        predict=True,
        rate_matched=True,
        k_values=(3, 5),
    ),
    ResponseTargetVariant(
        name=TARGET_NEURON_CENTERED,
        steps=(NEURON_CENTER_STEP, COLUMN_STANDARDISE_STEP),
        centering="neuron_mean",
        scaling="column_std",
        role=ROLE_MAIN,
        description=(
            "per-neuron baseline rate removed (R - mean_s R), then the canonical column "
            "standardisation; keeps stimulus-specific deviations and their amplitude"
        ),
        bootstrap=True,
        predict=True,
    ),
    ResponseTargetVariant(
        name=TARGET_NEURON_ZSCORED,
        steps=(NEURON_ZSCORE_STEP, COLUMN_STANDARDISE_STEP),
        centering="neuron_mean",
        scaling="neuron_std_then_column_std",
        role=ROLE_MAIN,
        description=(
            "per-neuron baseline and per-neuron amplitude removed ((R - mean_s R)/std_s R), "
            "then column standardisation; only the relative stimulus-response profile remains"
        ),
        bootstrap=True,
        predict=True,
    ),
    ResponseTargetVariant(
        name=TARGET_MEAN_RATE,
        steps=(MEAN_OVER_STIMULI_STEP, COLUMN_STANDARDISE_STEP),
        centering="none",
        scaling="column_std",
        role=ROLE_MAIN,
        description=(
            "1-D functional target: the per-neuron mean response over stimuli (Hz), z-scored "
            "across neurons; this is the target-space rate control, NOT the rate-only "
            "representation control"
        ),
        bootstrap=True,
        predict=True,
    ),
    ResponseTargetVariant(
        name=TARGET_CS_THEN_CENTERED,
        steps=(COLUMN_STANDARDISE_STEP, NEURON_CENTER_STEP),
        centering="neuron_mean_after_column_standardisation",
        scaling="column_std",
        role=ROLE_ORDERING_SENSITIVITY,
        description=(
            "ordering sensitivity: column standardisation FIRST, then per-neuron centering. "
            "Included because the two orders do not commute"
        ),
        bootstrap=False,
        predict=False,
    ),
    ResponseTargetVariant(
        name=TARGET_ROW_L2,
        steps=(ROW_L2_NORMALISE_STEP,),
        centering="none",
        scaling="row_l2",
        role=ROLE_EXISTING_CONTROL,
        description=(
            "the existing first-study target-side rate control: per-neuron L2-normalised raw "
            "profiles (magnitude removed). Reproduces primary_rate_normalized exactly"
        ),
        bootstrap=False,
        predict=False,
    ),
)

TARGET_VARIANTS_BY_NAME: dict[str, ResponseTargetVariant] = {v.name: v for v in TARGET_VARIANTS}


def main_target_variants(variants: Sequence[ResponseTargetVariant] = TARGET_VARIANTS) -> tuple[ResponseTargetVariant, ...]:
    """The four main decomposition targets: raw, neuron-centered, neuron-z-scored, mean-rate."""
    return tuple(v for v in variants if v.role == ROLE_MAIN)


# --------------------------------------------------------------------------
# Target construction (PROBE only)
# --------------------------------------------------------------------------
@dataclass
class ResponseTargets:
    """Every response-target variant of the study, all built from one PROBE activity pass."""

    spaces: dict[str, FingerprintSpace]
    variants: tuple[ResponseTargetVariant, ...]
    raw_response: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.variants[i].name for i in range(len(self.variants)))

    @property
    def n_neurons(self) -> int:
        return int(self.raw_response.shape[0])

    @property
    def n_stimuli(self) -> int:
        return int(self.raw_response.shape[1])

    @property
    def mean_rate_hz(self) -> np.ndarray:
        return self.raw_response.mean(axis=1)

    def variant(self, name: str) -> ResponseTargetVariant:
        for variant in self.variants:
            if variant.name == name:
                return variant
        raise RateRobustnessError(f"unknown target variant {name!r}; known: {list(self.names)}")

    def get(self, name: str) -> FingerprintSpace:
        if name not in self.spaces:
            raise RateRobustnessError(f"unknown target variant {name!r}; known: {list(self.names)}")
        return self.spaces[name]

    def summary(self) -> dict[str, Any]:
        return {
            "n_neurons": self.n_neurons,
            "n_stimuli": self.n_stimuli,
            "variants": [v.to_dict() for v in self.variants],
            "variant_dimensions": {name: int(space.X.shape[1]) for name, space in self.spaces.items()},
            "metadata": dict(self.metadata),
        }


def _target_feature_names(variant: ResponseTargetVariant, dimension: int) -> list[str]:
    if variant.name == TARGET_MEAN_RATE or dimension == 1:
        return [f"{variant.name}.mean_hz"]
    return [f"{variant.name}.s{k:04d}" for k in range(dimension)]


def build_response_targets(
    probe_result: ActivityAccumulatorResult,
    *,
    probe_split_label: str = "probe",
    variants: Sequence[ResponseTargetVariant] = TARGET_VARIANTS,
    eps: float = ZERO_VARIANCE_STD_EPS,
) -> ResponseTargets:
    """Build every response-target variant from a single labelled PROBE activity pass.

    The raw response matrix is computed once by the shared
    :func:`src.functional_fingerprint.stimulus_response_matrix` and transformed in memory.
    Each variant is wrapped in a :class:`FingerprintSpace` whose own standardisation is
    ``"none"`` — the pipeline steps have already applied every transform, so the operation
    order is exactly the one recorded in the variant, never re-applied implicitly.
    """
    if not isinstance(probe_result, ActivityAccumulatorResult):
        raise RateRobustnessError(
            f"response targets need an ActivityAccumulatorResult, got {type(probe_result).__name__}"
        )
    split = assert_held_out_split(probe_split_label)

    raw = stimulus_response_matrix(
        probe_result.counts, bin_ms=probe_result.bin_ms, n_bins=probe_result.n_bins
    )
    spaces: dict[str, FingerprintSpace] = {}
    per_variant: dict[str, dict[str, Any]] = {}
    for variant in variants:
        X, info = apply_response_pipeline(raw, variant.steps, eps=eps)
        spaces[variant.name] = FingerprintSpace(
            X_raw=X,
            feature_names=_target_feature_names(variant, X.shape[1]),
            config=FingerprintConfig(
                feature_sets=[STIMULUS_RESPONSE_FEATURE_SET],
                standardize="none",
                normalize_rows=False,
                metric="euclidean",
                eval_split=probe_split_label,
            ),
            meta={
                "target_variant": variant.name,
                "target_role": variant.role,
                "pipeline": variant.pipeline,
                "pipeline_steps": list(variant.steps),
                "centering": variant.centering,
                "scaling": variant.scaling,
                "split": probe_split_label,
                "uses_labels": False,
                "pipeline_info": info,
                "note": (
                    "evaluation-only transform of the PROBE response matrix; the pipeline "
                    "order is explicit and the operations are not commutative"
                ),
            },
        )
        per_variant[variant.name] = info

    metadata = {
        "probe_split": split,
        "probe_n_samples": int(probe_result.n_samples),
        "n_bins": int(probe_result.n_bins),
        "bin_ms": float(probe_result.bin_ms),
        "duration_s": float(probe_result.duration_s),
        "raw_response_definition": (
            "R[h, s] = counts[s, h] / (n_bins * bin_ms / 1000) Hz (stimulus_response_matrix); "
            "one column per PROBE utterance, no averaging across stimuli"
        ),
        "standardisation_order": (
            "each variant applies its own steps to the RAW response matrix, then wraps the "
            "result in FingerprintSpace(standardize='none', normalize_rows=False): the "
            "column standardisation is therefore ALWAYS the last step unless the variant "
            "explicitly puts it first (column_standardised_then_neuron_centered)"
        ),
        "operations_do_not_commute": True,
        "zero_variance_rule": (
            "a per-neuron profile with std_s R[h, s] < 1e-8 is kept (never dropped) and its "
            "whole z-scored row is set to 0.0"
        ),
        "raw_rate_summary": {
            "mean_hz_min": float(raw.mean(axis=1).min()),
            "mean_hz_median": float(np.median(raw.mean(axis=1))),
            "mean_hz_max": float(raw.mean(axis=1).max()),
            "std_hz_min": float(raw.std(axis=1).min()),
            "std_hz_median": float(np.median(raw.std(axis=1))),
            "std_hz_max": float(raw.std(axis=1).max()),
            "zero_variance_neurons": int((raw.std(axis=1) < float(eps)).sum()),
            "silent_neurons": int((raw.mean(axis=1) == 0.0).sum()),
            "all_zero_stimulus_columns": int((raw.sum(axis=0) == 0.0).sum()),
        },
        "per_variant_pipeline_info": per_variant,
        "uses_labels": False,
        "labels_used_only_in_targets": True,
    }
    return ResponseTargets(
        spaces=spaces, variants=tuple(variants), raw_response=raw, metadata=metadata
    )


# --------------------------------------------------------------------------
# Representation conditions (focused set)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RobustnessCondition:
    """One focused representation condition (may override the enabled record blocks)."""

    label: str
    structured_d: int
    residual_d: int = 0
    residual_seed: int | None = None
    blocks: tuple[str, ...] | None = None
    kind: str = KIND_STRUCTURED
    role: str = REP_ROLE_REPRESENTATION

    def __post_init__(self) -> None:
        if self.structured_d < 1:
            raise RateRobustnessError(f"{self.label}: structured_d must be >= 1")
        if self.residual_d < 0:
            raise RateRobustnessError(f"{self.label}: residual_d must be >= 0")
        if self.residual_d > 0:
            if self.kind != KIND_STRUCTURED_RESIDUAL:
                raise RateRobustnessError(f"{self.label}: residual conditions must use the residual kind")
            if self.residual_seed is None:
                raise RateRobustnessError(f"{self.label}: residual conditions must record a seed")
        elif self.kind != KIND_STRUCTURED:
            raise RateRobustnessError(f"{self.label}: non-residual conditions must use the structured kind")

    @property
    def total_d(self) -> int:
        return int(self.structured_d) + int(self.residual_d)

    @property
    def block_set(self) -> tuple[str, ...]:
        return tuple(self.blocks) if self.blocks is not None else tuple(DEFAULT_ENABLED_BLOCKS)

    def representation_condition(self) -> RepresentationCondition:
        """The equivalent :class:`RepresentationCondition` consumed by ``condition_matrix``."""
        return RepresentationCondition(
            self.kind,
            total_d=self.total_d,
            structured_d=int(self.structured_d),
            residual_d=int(self.residual_d),
            residual_seed=None if self.residual_seed is None else int(self.residual_seed),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "representation": self.label,
            "representation_kind": self.kind,
            "representation_role": self.role,
            "block_set": list(self.block_set),
            "total_d": self.total_d,
            "structured_d": int(self.structured_d),
            "residual_d": int(self.residual_d),
            "residual_seed": None if self.residual_seed is None else int(self.residual_seed),
        }


def activity_block_dimensions(bank: NeuronRecordBank) -> dict[str, int]:
    """Natural dimensions of the existing ``activity`` block (derived, never hard-coded).

    * ``activity_level0`` - the block's own deterministic summary features (the historical
      label-free activity representation);
    * ``activity_source`` - the full deterministic source the encoder derives from the block
      (level 0 + level 1 detail), still label-free FIT activity;
    * ``structural_plus_activity_level0`` - the canonical code prefix that contains exactly
      the 48 structural coordinates followed by the activity block's summary features.
    """
    activity = StructuredVectorEncoder(bank, structured_d=1, enabled_blocks=(ACTIVITY_BLOCK,))
    combined = StructuredVectorEncoder(
        bank, structured_d=1, enabled_blocks=(*DEFAULT_ENABLED_BLOCKS, ACTIVITY_BLOCK)
    )
    return {
        "activity_level0": int(activity.plan.level0_dimension),
        "activity_source": int(activity.plan.source_dimension),
        "structural_plus_activity_level0": int(combined.plan.level0_dimension),
    }


def focused_conditions(
    bank: NeuronRecordBank,
    *,
    structured_dims: Iterable[int] = DEFAULT_STRUCTURED_DIMS,
    residual_conditions: Iterable[tuple[int, int]] = DEFAULT_RESIDUAL_CONDITIONS,
    residual_seeds: Iterable[int] = DEFAULT_RESIDUAL_SEEDS,
    include_activity: bool = True,
) -> tuple[RobustnessCondition, ...]:
    """The focused representation set: structured, matched residual families, activity."""
    conditions: list[RobustnessCondition] = [
        RobustnessCondition(label=f"structured_{int(d)}", structured_d=int(d))
        for d in structured_dims
    ]
    for structured_d, residual_d in residual_conditions:
        for seed in residual_seeds:
            conditions.append(
                RobustnessCondition(
                    label=f"full_{int(structured_d)}+{int(residual_d)}_seed{int(seed)}",
                    structured_d=int(structured_d),
                    residual_d=int(residual_d),
                    residual_seed=int(seed),
                    kind=KIND_STRUCTURED_RESIDUAL,
                )
            )
    if include_activity:
        dims = activity_block_dimensions(bank)
        conditions.append(
            RobustnessCondition(
                label="activity_only",
                structured_d=dims["activity_level0"],
                blocks=(ACTIVITY_BLOCK,),
                role=REP_ROLE_ACTIVITY,
            )
        )
        conditions.append(
            RobustnessCondition(
                label=f"activity_source_{dims['activity_source']}",
                structured_d=dims["activity_source"],
                blocks=(ACTIVITY_BLOCK,),
                role=REP_ROLE_ACTIVITY,
            )
        )
        conditions.append(
            RobustnessCondition(
                label="structural_48_plus_activity",
                structured_d=dims["structural_plus_activity_level0"],
                blocks=(*DEFAULT_ENABLED_BLOCKS, ACTIVITY_BLOCK),
                role=REP_ROLE_ACTIVITY,
            )
        )
    return tuple(conditions)


def robustness_matrix(
    condition: RobustnessCondition,
    bank: NeuronRecordBank,
    *,
    residuals: Mapping[Any, Any] | None = None,
    chunk_size: int | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Materialise one focused condition via the existing composition path (FIT-derived)."""
    X, names = condition_matrix(
        condition.representation_condition(),
        bank,
        residuals=residuals,
        enabled_blocks=condition.blocks,
        chunk_size=chunk_size,
    )
    if X.shape[1] != condition.total_d:
        raise RateRobustnessError(
            f"condition {condition.label} produced {X.shape[1]} features, expected {condition.total_d}"
        )
    return X, names


# --------------------------------------------------------------------------
# Evaluation of one (representations, target variant) pair
# --------------------------------------------------------------------------
def evaluate_target(
    X: np.ndarray,
    feature_names: Sequence[str],
    target_space: FingerprintSpace,
    variant: ResponseTargetVariant,
    *,
    settings: EvaluationSettings,
    probe_n: int | None = None,
    rates_absdiff: np.ndarray | None = None,
    cv: Any | None = None,
    description: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Geometry (and optional prediction) for one frozen representation and one target.

    Uses :func:`src.vector_capacity.geometry_for_target`, i.e. the same canonical
    composition the first study used (identical Mantel procedure, permutations, seed and
    bootstrap convention). ``rates_absdiff`` (from the FIT rates) is only used for the
    variants that explicitly request the rate-matched stratified control.
    """
    space = representation_space(X, feature_names)
    if rates_absdiff is not None and np.asarray(rates_absdiff).size != space.condensed().size:
        raise RateRobustnessError("rate-difference control size does not match the pair count")

    local_settings = settings if variant.bootstrap else dataclasses.replace(settings, bootstrap=0)
    analysis = geometry_for_target(
        space.X,
        target_space,
        settings=local_settings,
        rates_absdiff=(rates_absdiff if variant.rate_matched else None),
        k_values=tuple(variant.k_values),
        include_curves=False,
    )
    headline = primary_metric_row("", analysis)

    row: dict[str, Any] = {
        "n_neurons": int(space.n_neurons),
        "probe_n": None if probe_n is None else int(probe_n),
        "target_variant": variant.name,
        "target_role": variant.role,
        "target_pipeline": variant.pipeline,
        "target_centering": variant.centering,
        "target_scaling": variant.scaling,
        "target_dimension": int(target_space.X.shape[1]),
        "metric": "mantel_spearman_r",
        "value": headline.get("primary_metric_mantel_spearman_r"),
        "permutation_p": headline.get("primary_metric_p_value"),
        "permutation_p_floor": headline.get("primary_metric_p_value_floor"),
        "effect_size_z": headline.get("primary_metric_effect_size_z"),
        "bootstrap_low": headline.get("primary_metric_ci_low"),
        "bootstrap_high": headline.get("primary_metric_ci_high"),
        "pearson_r": headline.get("pearson_r"),
        "knn_best_k": headline.get("best_knn_k"),
        "knn_effect_size_z": headline.get("best_knn_effect_size_z"),
    }
    if variant.rate_matched:
        row["rate_matched_r"] = analysis.get("rate_matched_mantel", {}).get("statistic")
        row["rate_matched_p"] = analysis.get("rate_matched_mantel", {}).get("p_value")
        row["partial_mantel_r_controlling_rate"] = headline.get(
            "partial_mantel_r_controlling_firing_rate"
        )
        row["partial_mantel_status"] = headline.get("partial_mantel_status")
    if variant.predict:
        folds = cv if cv is not None else make_shared_folds(
            space.n_neurons, settings.n_splits, settings.seed
        )
        ridge = cross_validated_ridge(
            space.X, target_space.X, n_splits=settings.n_splits, alphas=settings.alphas,
            seed=settings.seed, cv=folds,
        )["metrics"]
        row["prediction_metric"] = "cv_r2_mean"
        row["prediction_value"] = ridge.get("r2_mean")
        row["prediction_pearson_r"] = ridge.get("pearson_r_mean")
    if description:
        row.update(dict(description))
    return row


def run_rate_robustness(
    bank: NeuronRecordBank,
    targets: ResponseTargets,
    conditions: Sequence[RobustnessCondition],
    *,
    settings: EvaluationSettings,
    fit_rates: np.ndarray,
    residuals: Mapping[Any, Any] | None = None,
    variants: Sequence[ResponseTargetVariant] = TARGET_VARIANTS,
    chunk_size: int | None = None,
    checkpoint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate every focused condition against every target variant."""
    rates_absdiff = rate_absdiff_condensed(fit_rates)
    cv = make_shared_folds(targets.n_neurons, settings.n_splits, settings.seed)
    provenance = dict(checkpoint or {})

    rows: list[dict[str, Any]] = []
    for condition in conditions:
        X, names = robustness_matrix(condition, bank, residuals=residuals, chunk_size=chunk_size)
        for variant in variants:
            row = evaluate_target(
                X,
                names,
                targets.get(variant.name),
                variant,
                settings=settings,
                probe_n=targets.n_stimuli,
                rates_absdiff=rates_absdiff,
                cv=cv,
                description={
                    **condition.to_dict(),
                    **provenance,
                    "evaluation_seed": int(settings.seed),
                    "n_perm": int(settings.n_perm),
                    "bootstrap_n": int(settings.bootstrap) if variant.bootstrap else 0,
                },
            )
            rows.append(row)

    return {
        "rows": rows,
        "settings": settings.to_dict(),
        "targets": targets.summary(),
        "conditions": [c.to_dict() for c in conditions],
        "preprocessing": REPRESENTATION_PREPROCESSING,
        "checkpoint": provenance,
    }


def representation_control_rows(
    *,
    fit_rates: np.ndarray,
    targets: ResponseTargets,
    settings: EvaluationSettings,
    variants: Sequence[ResponseTargetVariant] | None = None,
    rates_absdiff: np.ndarray | None = None,
    matched_dim: int = CONTROL_MATCHED_DIM,
    shuffle_reference: tuple[np.ndarray, Sequence[str]] | None = None,
    checkpoint: Mapping[str, Any] | None = None,
    cv: Any | None = None,
) -> list[dict[str, Any]]:
    """The existing representation-side controls, evaluated against the main targets.

    * ``control_rate_only`` - the label-free **FIT** rate as a 1-D representation
      (distance ``|rate_i - rate_j| / std``; a representation-side control).
    * ``control_random_{d}`` - i.i.d. Gaussian representation matched in dimension.
    * ``control_neuron_shuffle_{d}`` - the representation with neuron rows permuted.

    Only the main target variants are evaluated, so the control null is directly comparable
    with the decomposition table.
    """
    chosen = tuple(variants) if variants is not None else main_target_variants()
    if rates_absdiff is None:
        rates_absdiff = rate_absdiff_condensed(fit_rates)
    rates = np.asarray(fit_rates, dtype=np.float64).reshape(-1, 1)
    provenance = dict(checkpoint or {})
    common = {
        "residual_d": None,
        "residual_seed": None,
        "representation_role": REP_ROLE_CONTROL,
        **provenance,
        "evaluation_seed": int(settings.seed),
        "n_perm": int(settings.n_perm),
    }

    rows: list[dict[str, Any]] = []

    def _run(label: str, X: np.ndarray, names: Sequence[str], total_d: int) -> None:
        for variant in chosen:
            row = evaluate_target(
                X, names, targets.get(variant.name), variant, settings=settings,
                probe_n=targets.n_stimuli, rates_absdiff=rates_absdiff, cv=cv,
                description={
                    "representation": label,
                    "representation_kind": KIND_CONTROL,
                    "block_set": ["control"],
                    "total_d": int(total_d),
                    "structured_d": None,
                    "bootstrap_n": int(settings.bootstrap) if variant.bootstrap else 0,
                    **common,
                },
            )
            row["rate_only_primary_r"] = (
                row.get("value") if label == "control_rate_only" and variant.name == TARGET_RAW else None
            )
            rows.append(row)

    rate_space = representation_space(rates, ["activity.rate_hz"])
    _run("control_rate_only", rate_space.X, ["activity.rate_hz"], 1)

    random_space = random_baseline_space(
        rates.shape[0], int(matched_dim), seed=settings.random_control_seed, weighting="uniform",
        meta={"control": "random", "matched_dim": int(matched_dim)},
    )
    _run(f"control_random_{int(matched_dim)}", random_space.X, random_space.feature_names, int(matched_dim))

    if shuffle_reference is not None:
        X_ref, names_ref = shuffle_reference
        reference_space = representation_space(X_ref, names_ref)
        shuffled = shuffle_control_space(reference_space, seed=settings.shuffle_control_seed)
        _run(
            f"control_neuron_shuffle_{int(X_ref.shape[1])}",
            shuffled.X,
            shuffled.feature_names,
            int(X_ref.shape[1]),
        )
    return rows


# --------------------------------------------------------------------------
# Checkpoint comparability
# --------------------------------------------------------------------------
#: Architecture fields that must agree for two checkpoints to be comparable.
ARCHITECTURE_FIELDS: tuple[str, ...] = (
    "n_input", "n_hidden", "n_output", "n_bins", "bin_ms",
    "tau_mem_ms", "tau_syn_ms", "threshold", "readout_mode", "neuron_param_mode",
    "recurrent_density", "signed_input_weights", "signed_recurrent_weights",
)


@dataclass
class CheckpointCompatibility:
    """Outcome of the checkpoint comparability check."""

    compatible: bool
    reasons: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "compatible": bool(self.compatible),
            "reasons": list(self.reasons),
            "details": dict(self.details),
        }


def representation_schema(bank: NeuronRecordBank) -> dict[str, Any]:
    """The deterministic encoder schema a bank induces (feature names and dimensions)."""
    encoder = StructuredVectorEncoder(bank, structured_d=1)
    return {
        "present_blocks": list(encoder.plan.present_blocks),
        "level0_dimension": int(encoder.plan.level0_dimension),
        "level1_dimension": int(encoder.plan.level1_dimension),
        "source_dimension": int(encoder.plan.source_dimension),
        "level0_layout": [f"{b}.{f}" for b, f in encoder.plan.level0_layout],
        "level1_layout": [f"{b}.{f}" for b, f in encoder.plan.level1_layout],
    }


def check_checkpoint_compatibility(
    reference: tuple[str, Any, NeuronRecordBank],
    candidate: tuple[str, Any, NeuronRecordBank],
) -> CheckpointCompatibility:
    """Verify a candidate checkpoint is scientifically comparable to the reference.

    Requires (a) identical architecture on :data:`ARCHITECTURE_FIELDS` and (b) an identical
    representation schema (same present blocks, same feature names and dimensions).
    Incomparable checkpoints are rejected rather than aggregated.
    """
    reference_name, reference_model, reference_bank = reference
    candidate_name, candidate_model, candidate_bank = candidate
    reasons: list[str] = []
    details: dict[str, Any] = {
        "reference": reference_name,
        "candidate": candidate_name,
        "architecture": {},
        "schema": {},
    }

    for field_name in ARCHITECTURE_FIELDS:
        a = getattr(reference_model.cfg, field_name, None)
        b = getattr(candidate_model.cfg, field_name, None)
        if a != b:
            reasons.append(f"architecture field {field_name!r}: {a!r} != {b!r}")
        details["architecture"][field_name] = {"reference": a, "candidate": b}

    schema_a = representation_schema(reference_bank)
    schema_b = representation_schema(candidate_bank)
    details["schema"] = {"reference": schema_a, "candidate": schema_b}
    for key in ("present_blocks", "level0_layout", "level1_layout", "level0_dimension", "source_dimension"):
        if schema_a[key] != schema_b[key]:
            reasons.append(f"representation schema {key!r} differs between checkpoints")
    if reference_bank.n_neurons != candidate_bank.n_neurons:
        reasons.append(
            f"n_neurons differs: {reference_bank.n_neurons} != {candidate_bank.n_neurons}"
        )

    return CheckpointCompatibility(compatible=not reasons, reasons=reasons, details=details)


# --------------------------------------------------------------------------
# Aggregation / serialisation
# --------------------------------------------------------------------------
#: Canonical machine-readable columns (order preserved); the prompt's fields come first.
RESULT_COLUMNS: tuple[str, ...] = (
    "checkpoint",
    "checkpoint_sha256_16",
    "representation",
    "representation_kind",
    "representation_role",
    "block_set",
    "total_d",
    "structured_d",
    "residual_d",
    "residual_seed",
    "target_variant",
    "target_role",
    "target_pipeline",
    "target_centering",
    "target_scaling",
    "target_dimension",
    "metric",
    "value",
    "bootstrap_low",
    "bootstrap_high",
    "permutation_p",
    "effect_size_z",
    "rate_matched_r",
    "rate_only_primary_r",
    "prediction_metric",
    "prediction_value",
    "n_neurons",
    "probe_n",
    "evaluation_seed",
    "n_perm",
    "bootstrap_n",
)

_MAIN_METRICS = ("value", "rate_matched_r", "prediction_value")


def _metric_values(rows: Sequence[Mapping[str, Any]], metric: str) -> np.ndarray:
    values = []
    for row in rows:
        raw = row.get(metric)
        if raw is None:
            continue
        try:
            number = float(raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            values.append(number)
    return np.asarray(values, dtype=np.float64)


def summarise_seed_variability(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Mean / SD / per-seed values for the residual conditions, **within each checkpoint**.

    Seeds are never collapsed away, and seed variability is kept separate from checkpoint
    variability: grouping is by ``(checkpoint, structured_d, residual_d, target_variant)``.
    """
    groups: dict[tuple[Any, int, int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("residual_d") in (None, 0):
            continue
        key = (
            row.get("checkpoint"),
            int(row["structured_d"]),
            int(row["residual_d"]),
            str(row["target_variant"]),
        )
        groups.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for (checkpoint, structured_d, residual_d, target_variant), group in sorted(
        groups.items(), key=lambda item: (str(item[0][0]), item[0][1], item[0][2], item[0][3])
    ):
        entry: dict[str, Any] = {
            "checkpoint": checkpoint,
            "structured_d": structured_d,
            "residual_d": residual_d,
            "total_d": structured_d + residual_d,
            "target_variant": target_variant,
            "n_seeds": len(group),
            "residual_seeds": sorted(
                int(r["residual_seed"]) for r in group if r.get("residual_seed") is not None
            ),
        }
        for metric in _MAIN_METRICS:
            values = _metric_values(group, metric)
            entry[f"{metric}_mean"] = float(values.mean()) if values.size else None
            entry[f"{metric}_std"] = float(values.std(ddof=0)) if values.size else None
            entry[f"{metric}_values"] = [float(v) for v in values]
        summary.append(entry)
    return summary


def summarise_checkpoint_variability(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Mean / SD / per-checkpoint values for every condition, across checkpoints.

    Rows are first averaged **within** each checkpoint (so residual families contribute their
    seed mean), then summarised across checkpoints. Control rows are excluded.
    """
    per_checkpoint: dict[tuple[str, str], dict[Any, list[float]]] = {}
    for row in rows:
        if row.get("representation_role") == REP_ROLE_CONTROL:
            continue
        value = row.get("value")
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(number):
            continue
        key = (str(row["representation"]), str(row["target_variant"]))
        per_checkpoint.setdefault(key, {}).setdefault(row.get("checkpoint"), []).append(number)

    summary: list[dict[str, Any]] = []
    for (representation, target_variant), by_checkpoint in sorted(per_checkpoint.items()):
        per_mean = {checkpoint: float(np.mean(values)) for checkpoint, values in by_checkpoint.items()}
        values = np.asarray(list(per_mean.values()), dtype=np.float64)
        summary.append({
            "representation": representation,
            "target_variant": target_variant,
            "n_checkpoints": len(per_mean),
            "checkpoints": sorted(str(c) for c in per_mean),
            "value_mean": float(values.mean()),
            "value_std": float(values.std(ddof=0)),
            "value_values": [float(v) for v in values],
            "per_checkpoint_mean": {str(k): float(v) for k, v in per_mean.items()},
        })
    return summary


def rate_decomposition_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compact diagnostic table: one row per representation with the main target metrics.

    Columns follow the brief: ``raw``, ``neuron_centered``, ``neuron_zscored`` and
    ``mean_rate`` Mantel r, plus the descriptive differences ``Δ_centering`` and
    ``Δ_scaling`` and the secondary ``row_l2`` control when present. With several
    checkpoints (and residual seeds) each cell is the mean over those replicates, and
    ``n_rows`` records how many rows were averaged.
    """
    grouped: dict[tuple[str, str], list[float]] = {}
    first: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if row.get("representation_role") == REP_ROLE_CONTROL:
            continue
        representation = str(row["representation"])
        target = str(row["target_variant"])
        first.setdefault(representation, row)
        value = row.get("value")
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            grouped.setdefault((representation, target), []).append(number)

    def cell(representation: str, target: str) -> tuple[float | None, int]:
        values = grouped.get((representation, target), [])
        return (float(np.mean(values)) if values else None, len(values))

    table: list[dict[str, Any]] = []
    for representation, sample in sorted(first.items()):
        raw, n_raw = cell(representation, TARGET_RAW)
        centered, _ = cell(representation, TARGET_NEURON_CENTERED)
        zscored, _ = cell(representation, TARGET_NEURON_ZSCORED)
        mean_rate, _ = cell(representation, TARGET_MEAN_RATE)
        row_l2, _ = cell(representation, TARGET_ROW_L2)
        entry: dict[str, Any] = {
            "representation": representation,
            "total_d": sample.get("total_d"),
            "structured_d": sample.get("structured_d"),
            "residual_d": sample.get("residual_d"),
            "raw_r": raw,
            "centered_r": centered,
            "zscored_r": zscored,
            "mean_rate_r": mean_rate,
            "row_l2_r": row_l2,
            "n_rows": n_raw,
        }
        entry["delta_centering"] = (
            float(centered) - float(raw) if centered is not None and raw is not None else None
        )
        entry["delta_scaling"] = (
            float(zscored) - float(centered) if zscored is not None and centered is not None else None
        )
        table.append(entry)
    return table


def public_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Drop private values so a row is JSON/CSV friendly."""
    return {str(k): v for k, v in row.items() if not str(k).startswith("_")}


def write_results(
    result: Mapping[str, Any],
    *,
    csv_path: str | Path,
    json_path: str | Path,
) -> dict[str, Path]:
    """Write the rate-robustness result table (CSV) and the full payload (JSON).

    Written to dedicated filenames so the first capacity study's ``results.csv`` /
    ``results.json`` / ``metadata.json`` are never touched.
    """
    rows = [public_row(r) for r in result.get("rows", [])]
    csv_path, json_path = Path(csv_path), Path(json_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = list(RESULT_COLUMNS)
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
        "schema": SCHEMA,
        "rows": rows,
        "rate_decomposition_table": rate_decomposition_table(rows),
        "summary_seed_variability": summarise_seed_variability(rows),
        "summary_checkpoint_variability": summarise_checkpoint_variability(rows),
        "settings": result.get("settings"),
        "targets": result.get("targets"),
        "conditions": result.get("conditions"),
        "preprocessing": result.get("preprocessing"),
        "checkpoints": result.get("checkpoints"),
        "compatibility": result.get("compatibility"),
        "metadata": result.get("metadata"),
        "result_columns": list(RESULT_COLUMNS),
    }
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
    return {"csv": csv_path, "json": json_path}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


__all__ = [
    "SCHEMA",
    "TARGET_RAW",
    "TARGET_NEURON_CENTERED",
    "TARGET_NEURON_ZSCORED",
    "TARGET_MEAN_RATE",
    "TARGET_CS_THEN_CENTERED",
    "TARGET_ROW_L2",
    "ROLE_MAIN",
    "ROLE_ORDERING_SENSITIVITY",
    "ROLE_EXISTING_CONTROL",
    "REP_ROLE_REPRESENTATION",
    "REP_ROLE_ACTIVITY",
    "REP_ROLE_CONTROL",
    "CONTROL_MATCHED_DIM",
    "DEFAULT_STRUCTURED_DIMS",
    "DEFAULT_RESIDUAL_CONDITIONS",
    "DEFAULT_RESIDUAL_SEEDS",
    "ARCHITECTURE_FIELDS",
    "RESULT_COLUMNS",
    "RateRobustnessError",
    "ResponseTargetVariant",
    "TARGET_VARIANTS",
    "TARGET_VARIANTS_BY_NAME",
    "main_target_variants",
    "ResponseTargets",
    "build_response_targets",
    "RobustnessCondition",
    "activity_block_dimensions",
    "focused_conditions",
    "robustness_matrix",
    "evaluate_target",
    "run_rate_robustness",
    "representation_control_rows",
    "CheckpointCompatibility",
    "representation_schema",
    "check_checkpoint_compatibility",
    "summarise_seed_variability",
    "summarise_checkpoint_variability",
    "rate_decomposition_table",
    "public_row",
    "write_results",
]
