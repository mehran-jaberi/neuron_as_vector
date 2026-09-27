"""Source-extension evaluation: do the new label-free sources change the relationship?

The previous robustness study established that the connectivity/intrinsic representation has a
modest correspondence with individual-stimulus PROBE responses and that, once each neuron's
mean level and amplitude are removed (the neuron z-scored target), that correspondence is
indistinguishable from the null — while activity-derived representations carry substantially
more (partly rate-aligned) information.

This module evaluates the two new label-free representation sources added in the previous
stage (:mod:`src.functional_response`, the coarse temporal block) with the **identical** target
pipelines, geometry machinery, controls and statistical conventions of the robustness study, so
the results are directly comparable and joinable:

.. code-block:: text

    FIT (label-free)  ->  NeuronRecordBank  ->  deterministic structured / +temporal
                                          ->  learned residual (source ablation A/B/C/D)
    PROBE (held out)  ->  response targets (raw | neuron_centered | neuron_zscored | mean_rate)
                                          +  the existing class-rate / temporal targets
                                          ->  the existing Mantel / kNN-free / prediction metrics

The central question is whether the new sources move the **neuron z-scored** (rate- and
amplitude-independent) target, not merely the raw one. Everything here is descriptive: no
ranking, no "best", and no causal language.

Source ablation (the core of this stage), at a fixed residual architecture/training protocol:

======  ==========================================  ==================================
key     residual source view                        isolates
======  ==========================================  ==================================
A       level-0 + level-1 + raw-connectivity views  the previous stage's source
B       A + the functional-response projection      the functional-response contribution
C       A + the coarse temporal block               the temporal contribution
D       A + both new sources                        the combined contribution
======  ==========================================  ==================================

Discipline: FIT builds every representation (bank, residual sources, residual training);
PROBE builds the targets and the metrics; the residual configuration is pre-specified and never
tuned on PROBE; TEST is never read.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .neuron_record import TEMPORAL_BLOCK, NeuronRecordBank
from .neuron_vector import NeuronVectorError, build_neuron_vectors
from .prediction import make_shared_folds
from .rate_robustness import (
    DEFAULT_RESIDUAL_SEEDS,
    RESULT_COLUMNS as BASE_RESULT_COLUMNS,
    ResponseTargetVariant,
    ResponseTargets,
    TARGET_MEAN_RATE,
    TARGET_NEURON_CENTERED,
    TARGET_NEURON_ZSCORED,
    TARGET_RAW,
    build_response_targets,
    evaluate_target,
    main_target_variants,
)
from .functional_fingerprint import FingerprintSpace
from .residual import ResidualError, ResidualSourceConfig, ResidualResult
from .structured_vector import StructuredVectorError, StructuredVectorEncoder
from .vector_capacity import (
    KIND_STRUCTURED,
    KIND_STRUCTURED_RESIDUAL,
    REPRESENTATION_PREPROCESSING,
    TARGET_CLASS_RATE,
    TARGET_TEMPORAL,
    EvaluationSettings,
    EvaluationTargets,
    VectorCapacityError,
    build_evaluation_targets,
    rate_absdiff_condensed,
)
from .v2_config import DEFAULT_ENABLED_BLOCKS

#: Provenance schema of a source-extension payload.
SCHEMA = "neuron_vector_source_extension/v1"

#: Residual source-view keys (the A/B/C/D ablation, plus the structural-only case).
SOURCE_STRUCTURAL = "structural"
SOURCE_FUNCTIONAL = "functional_response"
SOURCE_TEMPORAL = "temporal"
SOURCE_FUNCTIONAL_TEMPORAL = "functional_response+temporal"
SOURCE_KEYS: tuple[str, ...] = (
    SOURCE_STRUCTURAL,
    SOURCE_FUNCTIONAL,
    SOURCE_TEMPORAL,
    SOURCE_FUNCTIONAL_TEMPORAL,
)
SOURCE_DESCRIPTIONS: dict[str, str] = {
    SOURCE_STRUCTURAL: "level-0 + level-1 deterministic summaries + fixed raw-connectivity views",
    SOURCE_FUNCTIONAL: "structural + the fixed projection of the label-free FIT individual-stimulus response profile",
    SOURCE_TEMPORAL: "structural + the coarse label-free FIT temporal block",
    SOURCE_FUNCTIONAL_TEMPORAL: "structural + the functional-response projection + the coarse temporal block",
}
#: Short tags used in condition labels.
SOURCE_TAGS: dict[str, str] = {
    SOURCE_STRUCTURAL: "structural",
    SOURCE_FUNCTIONAL: "functional",
    SOURCE_TEMPORAL: "temporal",
    SOURCE_FUNCTIONAL_TEMPORAL: "functional_temporal",
}

#: Masking strategies (config ``vector.residual.mask_mode``).
MASK_COORDINATE = "coordinate"
MASK_BLOCK = "block"

#: Condition roles.
ROLE_DETERMINISTIC = "deterministic"
ROLE_SOURCE_ABLATION = "source_ablation"
ROLE_MASK_DIAGNOSTIC = "mask_diagnostic"
ROLE_SECONDARY_TARGET = "secondary_target"

#: Dimension used for the source ablation and the mask diagnostic (the brief).
ABLATION_RESIDUAL_DIM = 16

#: The ablation keys evaluated at :data:`ABLATION_RESIDUAL_DIM`.
ABLATION_SOURCE_KEYS: tuple[str, ...] = (
    SOURCE_STRUCTURAL,
    SOURCE_FUNCTIONAL,
    SOURCE_TEMPORAL,
    SOURCE_FUNCTIONAL_TEMPORAL,
)

#: Source keys evaluated at the larger residual dimension as well (the brief's §4).
LARGE_DIM_SOURCE_KEYS: tuple[str, ...] = (
    SOURCE_STRUCTURAL,
    SOURCE_FUNCTIONAL,
    SOURCE_FUNCTIONAL_TEMPORAL,
)

DEFAULT_STRUCTURED_D = 48
DEFAULT_RESIDUAL_DIMS: tuple[int, ...] = (16, 52)
DEFAULT_ACTIVITY_BLOCK = "activity"

#: Machine-readable columns of the long result table (the base table plus the source fields).
RESULT_COLUMNS: tuple[str, ...] = BASE_RESULT_COLUMNS + (
    "source_key",
    "source_description",
    "source_functional_response",
    "source_temporal",
    "source_dimension",
    "mask_mode",
    "condition_role",
    "artifact_key",
)

#: Metrics summarised across seeds/checkpoints.
SUMMARY_METRICS: tuple[str, ...] = (
    "raw_r",
    "centered_r",
    "zscored_r",
    "mean_rate_r",
    "class_rate_r",
    "temporal_r",
)


class SourceExtensionError(ValueError):
    """Raised for an invalid source-extension request."""


# --------------------------------------------------------------------------
# Conditions
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SourceCondition:
    """One representation condition of the source-extension study.

    ``structured_d``/``residual_d`` follow the repository convention
    ``total_d = structured_d + residual_d``; ``source_key``/``mask_mode`` describe how the
    residual's input view was built (the ablation), and ``artifact_key`` is the stable key
    under which the frozen residual artifact is looked up.
    """

    label: str
    role: str
    structured_d: int
    residual_d: int = 0
    residual_seed: int | None = None
    source_key: str = SOURCE_STRUCTURAL
    blocks: tuple[str, ...] | None = None
    mask_mode: str = MASK_COORDINATE
    kind: str = KIND_STRUCTURED
    notes: str = ""

    def __post_init__(self) -> None:
        if not str(self.label):
            raise SourceExtensionError("a condition needs a non-empty label")
        if self.role not in (ROLE_DETERMINISTIC, ROLE_SOURCE_ABLATION, ROLE_MASK_DIAGNOSTIC,
                             ROLE_SECONDARY_TARGET):
            raise SourceExtensionError(f"unknown condition role {self.role!r}")
        if self.source_key not in SOURCE_KEYS:
            raise SourceExtensionError(
                f"unknown source_key {self.source_key!r}; known: {list(SOURCE_KEYS)}"
            )
        if self.mask_mode not in (MASK_COORDINATE, MASK_BLOCK):
            raise SourceExtensionError(f"unknown mask_mode {self.mask_mode!r}")
        if int(self.structured_d) < 1:
            raise SourceExtensionError(f"{self.label}: structured_d must be >= 1")
        if int(self.residual_d) < 0:
            raise SourceExtensionError(f"{self.label}: residual_d must be >= 0")
        if self.residual_d > 0:
            if self.kind != KIND_STRUCTURED_RESIDUAL:
                raise SourceExtensionError(
                    f"{self.label}: conditions with a residual must use the residual kind"
                )
            if self.residual_seed is None:
                raise SourceExtensionError(
                    f"{self.label}: residual conditions must record their training seed"
                )
            if self.source_key == SOURCE_TEMPORAL and self.mask_mode != MASK_COORDINATE:
                raise SourceExtensionError(f"{self.label}: the mask diagnostic uses the combined source")
        elif self.kind != KIND_STRUCTURED:
            raise SourceExtensionError(
                f"{self.label}: conditions without a residual must use the structured kind"
            )

    @property
    def total_d(self) -> int:
        return int(self.structured_d) + int(self.residual_d)

    @property
    def block_set(self) -> tuple[str, ...]:
        return tuple(self.blocks) if self.blocks is not None else tuple(DEFAULT_ENABLED_BLOCKS)

    @property
    def source_flags(self) -> tuple[bool, bool]:
        """``(source_functional_response, source_temporal)`` implied by the source key."""
        return (
            self.source_key in (SOURCE_FUNCTIONAL, SOURCE_FUNCTIONAL_TEMPORAL),
            self.source_key in (SOURCE_TEMPORAL, SOURCE_FUNCTIONAL_TEMPORAL),
        )

    @property
    def artifact_key(self) -> str:
        """Stable key of the frozen residual artifact this condition consumes.

        Well defined for deterministic conditions too (they consume no artifact), so the key can
        be recorded in every result row without special cases.
        """
        seed = "none" if self.residual_seed is None else f"seed{int(self.residual_seed)}"
        return "|".join((self.source_key, self.mask_mode, f"dres{int(self.residual_d)}", seed))

    def to_dict(self) -> dict[str, Any]:
        functional, temporal = self.source_flags
        return {
            "representation": self.label,
            "condition_role": self.role,
            "representation_kind": self.kind,
            "block_set": list(self.block_set),
            "total_d": self.total_d,
            "structured_d": int(self.structured_d),
            "residual_d": int(self.residual_d),
            "residual_seed": None if self.residual_seed is None else int(self.residual_seed),
            "source_key": self.source_key,
            "source_description": SOURCE_DESCRIPTIONS[self.source_key],
            "source_functional_response": bool(functional),
            "source_temporal": bool(temporal),
            "mask_mode": self.mask_mode,
            "artifact_key": self.artifact_key,
            "notes": self.notes,
        }


def source_config_for(
    source_key: str,
    *,
    enabled_blocks: Sequence[str] | None = None,
    functional_source_dim: int | None = None,
    functional_projection_seed: int | None = None,
    functional_normalization: str | None = None,
    functional_chunk_size: int | None = None,
) -> ResidualSourceConfig:
    """The (deterministic) residual input view of one ablation key.

    Only the *source* changes between the ablation arms; the residual architecture, the
    training protocol and the masking strategy stay fixed.
    """
    if source_key not in SOURCE_KEYS:
        raise SourceExtensionError(f"unknown source_key {source_key!r}; known: {list(SOURCE_KEYS)}")
    functional, temporal = source_key in (SOURCE_FUNCTIONAL, SOURCE_FUNCTIONAL_TEMPORAL), (
        source_key in (SOURCE_TEMPORAL, SOURCE_FUNCTIONAL_TEMPORAL)
    )
    mapping: dict[str, Any] = {
        "enabled_blocks": list(enabled_blocks) if enabled_blocks is not None else None,
        "include_functional_response": bool(functional),
        "include_temporal": bool(temporal),
    }
    if functional_source_dim is not None:
        mapping["functional_source_dim"] = int(functional_source_dim)
    if functional_projection_seed is not None:
        mapping["functional_projection_seed"] = int(functional_projection_seed)
    if functional_normalization is not None:
        mapping["functional_normalization"] = str(functional_normalization)
    if functional_chunk_size is not None:
        mapping["functional_chunk_size"] = int(functional_chunk_size)
    try:
        return ResidualSourceConfig.from_mapping(mapping)
    except ResidualError as exc:
        raise SourceExtensionError(str(exc)) from exc


def _activity_dimensions(bank: NeuronRecordBank) -> dict[str, int]:
    """Natural dimensions of the existing ``activity`` block (derived, never assumed)."""
    single = StructuredVectorEncoder(bank, structured_d=1, enabled_blocks=(DEFAULT_ACTIVITY_BLOCK,))
    combined = StructuredVectorEncoder(
        bank, structured_d=1, enabled_blocks=(*DEFAULT_ENABLED_BLOCKS, DEFAULT_ACTIVITY_BLOCK)
    )
    return {
        "activity_level0": int(single.plan.level0_dimension),
        "structural_plus_activity_level0": int(combined.plan.level0_dimension),
    }


def structured_plus_temporal_dimension(bank: NeuronRecordBank) -> int:
    """Level-0 dimension of ``structural blocks + temporal`` (derived from the bank's plan)."""
    encoder = StructuredVectorEncoder(
        bank, structured_d=1, enabled_blocks=(*DEFAULT_ENABLED_BLOCKS, TEMPORAL_BLOCK)
    )
    present = encoder.plan.present_blocks
    if TEMPORAL_BLOCK not in present:
        raise SourceExtensionError(
            "the bank has no temporal block: build it with a temporal request "
            "(vector.enabled_blocks containing 'temporal', vector.residual.source_temporal=true, "
            "or an explicit temporal_resolution=) before evaluating the temporal source"
        )
    return int(encoder.plan.level0_dimension)


def focused_source_conditions(
    bank: NeuronRecordBank,
    *,
    structured_d: int = DEFAULT_STRUCTURED_D,
    residual_dims: Iterable[int] = DEFAULT_RESIDUAL_DIMS,
    residual_seeds: Iterable[int] = DEFAULT_RESIDUAL_SEEDS,
    include_activity: bool = True,
    include_temporal_deterministic: bool = True,
    include_mask_diagnostic: bool = True,
) -> tuple[SourceCondition, ...]:
    """The focused condition set: the two deterministic extensions + the residual ablation."""
    structural_blocks = tuple(DEFAULT_ENABLED_BLOCKS)
    conditions: list[SourceCondition] = [
        SourceCondition(
            label=f"structured_{int(structured_d)}",
            role=ROLE_DETERMINISTIC,
            structured_d=int(structured_d),
            notes="the previous stages' baseline (default block selection)",
        )
    ]
    if include_temporal_deterministic:
        temporal_d = structured_plus_temporal_dimension(bank)
        conditions.append(
            SourceCondition(
                label=f"structured_{int(structured_d)}_plus_temporal",
                role=ROLE_DETERMINISTIC,
                structured_d=temporal_d,
                blocks=(*structural_blocks, TEMPORAL_BLOCK),
                notes="the structural summary coordinates followed by the coarse temporal bins",
            )
        )
    if include_activity:
        dims = _activity_dimensions(bank)
        conditions.append(
            SourceCondition(
                label="activity_only",
                role=ROLE_DETERMINISTIC,
                structured_d=dims["activity_level0"],
                blocks=(DEFAULT_ACTIVITY_BLOCK,),
                notes="the existing label-free activity summary block (previous study baseline)",
            )
        )
        conditions.append(
            SourceCondition(
                label="structural_48_plus_activity",
                role=ROLE_DETERMINISTIC,
                structured_d=dims["structural_plus_activity_level0"],
                blocks=(*structural_blocks, DEFAULT_ACTIVITY_BLOCK),
                notes="the 48 structural coordinates followed by the activity summary features",
            )
        )

    seeds = tuple(int(s) for s in residual_seeds)
    dims_sorted = tuple(int(d) for d in residual_dims)
    for residual_d in dims_sorted:
        keys = ABLATION_SOURCE_KEYS if residual_d == ABLATION_RESIDUAL_DIM else LARGE_DIM_SOURCE_KEYS
        for key in keys:
            for seed in seeds:
                suffix = "" if key == SOURCE_STRUCTURAL else f"_plus_{SOURCE_TAGS[key]}"
                conditions.append(
                    SourceCondition(
                        label=f"full_{int(structured_d)}+{residual_d}_seed{seed}{suffix}",
                        role=ROLE_SOURCE_ABLATION,
                        structured_d=int(structured_d),
                        residual_d=residual_d,
                        residual_seed=seed,
                        source_key=key,
                        kind=KIND_STRUCTURED_RESIDUAL,
                        notes=SOURCE_DESCRIPTIONS[key],
                    )
                )
        if include_mask_diagnostic and residual_d == ABLATION_RESIDUAL_DIM:
            for seed in seeds:
                conditions.append(
                    SourceCondition(
                        label=(
                            f"full_{int(structured_d)}+{residual_d}_seed{seed}"
                            f"_plus_{SOURCE_TAGS[SOURCE_FUNCTIONAL_TEMPORAL]}_blockmask"
                        ),
                        role=ROLE_MASK_DIAGNOSTIC,
                        structured_d=int(structured_d),
                        residual_d=residual_d,
                        residual_seed=seed,
                        source_key=SOURCE_FUNCTIONAL_TEMPORAL,
                        mask_mode=MASK_BLOCK,
                        kind=KIND_STRUCTURED_RESIDUAL,
                        notes="the combined source with source-block masking (architectural diagnostic)",
                    )
                )
    return tuple(conditions)


def condition_matrix(
    condition: SourceCondition,
    bank: NeuronRecordBank,
    *,
    residual: ResidualResult | None = None,
    chunk_size: int | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Materialise one condition's frozen representation matrix ``(n_neurons, total_d)``.

    Uses the repository's composition path (:func:`src.neuron_vector.build_neuron_vectors`), so
    the matrix is exactly ``[z_structured, z_residual]`` with the configured decomposition.
    """
    if condition.residual_d > 0 and residual is None:
        raise SourceExtensionError(f"condition {condition.label} needs its frozen residual artifact")
    if condition.residual_d == 0 and residual is not None:
        raise SourceExtensionError(f"condition {condition.label} takes no residual artifact")
    try:
        vectors = build_neuron_vectors(
            bank,
            structured_d=int(condition.structured_d),
            residual=residual,
            enabled_blocks=condition.block_set,
            chunk_size=chunk_size,
        )
    except (NeuronVectorError, StructuredVectorError) as exc:
        raise SourceExtensionError(str(exc)) from exc
    X = np.asarray(vectors.X, dtype=np.float64)
    if X.shape[1] != condition.total_d:
        raise SourceExtensionError(
            f"condition {condition.label} produced {X.shape[1]} coordinates, expected total_d="
            f"{condition.total_d} (= structured_d {condition.structured_d} + residual_d "
            f"{condition.residual_d})"
        )
    return X, tuple(vectors.feature_names)


# --------------------------------------------------------------------------
# Target variants of this stage
# --------------------------------------------------------------------------
def stage_target_variants() -> tuple[ResponseTargetVariant, ...]:
    """The four main response-target variants, with this stage's (bounded) evaluation cost.

    Identical names, pipelines, ordering and bootstrap convention as the robustness study; kNN
    is switched off (it was characterised there) and ridge prediction is kept for the two
    primary outcomes only (the raw and the neuron z-scored target).
    """
    variants: list[ResponseTargetVariant] = []
    for variant in main_target_variants():
        variants.append(
            dataclasses.replace(
                variant,
                k_values=(),
                predict=variant.name in (TARGET_RAW, TARGET_NEURON_ZSCORED),
            )
        )
    return tuple(variants)


# --------------------------------------------------------------------------
# Targets: the robustness response variants + the existing secondary targets
# --------------------------------------------------------------------------
@dataclass
class StageTargets:
    """The union of this stage's target families, all built from one labelled PROBE pass.

    * ``response`` - the robustness study's response-target pipelines
      (``raw`` / ``neuron_centered`` / ``neuron_zscored`` / ``mean_rate``, unchanged);
    * ``functional`` - the first study's target object, used here for its ``class_rate_20d``
      and ``temporal`` fingerprints (unchanged definitions, secondary context).

    ``get`` dispatches by name so a condition can be evaluated against every target through
    one interface; nothing is recomputed or re-standardised here.
    """

    response: ResponseTargets
    functional: EvaluationTargets

    def get(self, name: str) -> FingerprintSpace:
        if name in self.response.spaces:
            return self.response.get(name)
        return self.functional.get(name)

    @property
    def n_neurons(self) -> int:
        return self.response.n_neurons

    @property
    def n_stimuli(self) -> int:
        return self.response.n_stimuli

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.response.names) + (TARGET_CLASS_RATE, TARGET_TEMPORAL)

    def summary(self) -> dict[str, Any]:
        return {
            "response_targets": self.response.summary(),
            "secondary_targets": {
                TARGET_CLASS_RATE: self.functional.class_rate.meta | {
                    "definition": self.functional.summary()["definitions"][TARGET_CLASS_RATE]
                },
                TARGET_TEMPORAL: self.functional.temporal.meta | {
                    "definition": self.functional.summary()["definitions"][TARGET_TEMPORAL]
                },
            },
            "n_neurons": self.n_neurons,
            "n_stimuli": self.n_stimuli,
            "probe_split": self.response.metadata.get("probe_split"),
        }


def build_stage_targets(
    probe_result: Any,
    *,
    probe_split_label: str = "probe",
    n_psth_bins: int = 10,
    min_spikes_for_latency: float = 1.0,
) -> StageTargets:
    """Build every target of this stage from a single labelled PROBE activity pass.

    Reuses the two existing builders unchanged (no target definition is modified here).
    """
    return StageTargets(
        response=build_response_targets(probe_result, probe_split_label=probe_split_label),
        functional=build_evaluation_targets(
            probe_result,
            probe_split_label=probe_split_label,
            n_psth_bins=int(n_psth_bins),
            min_spikes_for_latency=float(min_spikes_for_latency),
        ),
    )


def secondary_target_variants() -> tuple[ResponseTargetVariant, ...]:
    """The existing class-rate and temporal targets, as secondary context (unchanged definitions)."""
    return (
        ResponseTargetVariant(
            name=TARGET_CLASS_RATE,
            steps=(),
            centering="class_conditioned",
            scaling="column_std",
            role=ROLE_SECONDARY_TARGET,
            description="the existing 20-D class-conditioned rate fingerprint (secondary)",
            bootstrap=False,
            predict=False,
            rate_matched=False,
            k_values=(),
        ),
        ResponseTargetVariant(
            name=TARGET_TEMPORAL,
            steps=(),
            centering="class_conditioned",
            scaling="column_std",
            role=ROLE_SECONDARY_TARGET,
            description="the existing temporal fingerprint preset (secondary, exploratory)",
            bootstrap=False,
            predict=False,
            rate_matched=False,
            k_values=(),
        ),
    )


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run_source_extension(
    bank: NeuronRecordBank,
    targets: StageTargets,
    conditions: Sequence[SourceCondition],
    *,
    settings: EvaluationSettings,
    fit_rates: np.ndarray,
    residuals: Mapping[str, ResidualResult] | None = None,
    variants: Sequence[ResponseTargetVariant] | None = None,
    secondary: Sequence[ResponseTargetVariant] | None = None,
    checkpoint: Mapping[str, Any] | None = None,
    chunk_size: int | None = None,
) -> dict[str, Any]:
    """Evaluate every condition against the main and the secondary targets."""
    chosen = tuple(variants) if variants is not None else stage_target_variants()
    chosen_secondary = tuple(secondary) if secondary is not None else secondary_target_variants()
    rates_absdiff = rate_absdiff_condensed(fit_rates)
    cv = make_shared_folds(targets.n_neurons, settings.n_splits, settings.seed)
    provenance = dict(checkpoint or {})
    residuals = dict(residuals or {})

    rows: list[dict[str, Any]] = []
    matrices: dict[str, tuple[int, int]] = {}
    for condition in conditions:
        residual = None
        if condition.residual_d > 0:
            residual = residuals.get(condition.artifact_key)
            if not isinstance(residual, ResidualResult):
                raise SourceExtensionError(
                    f"condition {condition.label} needs the frozen residual artifact "
                    f"{condition.artifact_key!r}; pass it in `residuals`"
                )
            if int(residual.residual_dim) != int(condition.residual_d):
                raise SourceExtensionError(
                    f"condition {condition.label} expects residual_dim={condition.residual_d} but the "
                    f"artifact has {residual.residual_dim}"
                )
        try:
            X, names = condition_matrix(condition, bank, residual=residual, chunk_size=chunk_size)
        except (VectorCapacityError, ValueError) as exc:
            raise SourceExtensionError(str(exc)) from exc
        matrices[condition.label] = (int(X.shape[0]), int(X.shape[1]))

        description = {
            **condition.to_dict(),
            **provenance,
            "evaluation_seed": int(settings.seed),
            "n_perm": int(settings.n_perm),
        }
        for variant in chosen + chosen_secondary:
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
                    **description,
                    "bootstrap_n": int(settings.bootstrap) if variant.bootstrap else 0,
                    "source_dimension": None,
                },
            )
            rows.append(row)

    return {
        "rows": rows,
        "settings": settings.to_dict(),
        "targets": targets.summary(),
        "conditions": [c.to_dict() for c in conditions],
        "matrices": matrices,
        "preprocessing": REPRESENTATION_PREPROCESSING,
        "checkpoint": provenance,
        "variants": [v.name for v in chosen],
        "secondary_variants": [v.name for v in chosen_secondary],
    }


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def _value(rows: Sequence[Mapping[str, Any]], target: str, metric: str = "value") -> float | None:
    for row in rows:
        if str(row.get("target_variant")) == target:
            raw = row.get(metric)
            if raw is None:
                continue
            try:
                number = float(raw)
            except (TypeError, ValueError):
                continue
            if np.isfinite(number):
                return number
    return None


def condition_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One compact row per (condition, checkpoint, residual seed) - the §14 table.

    Every target variant of the study appears as its own column, with the bootstrap interval
    of the raw and neuron z-scored targets and the permutation p-value of the raw target.
    Rows are never merged across checkpoints or seeds.
    """
    grouped: dict[tuple[str, Any, Any], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["representation"]), row.get("checkpoint"), row.get("residual_seed"))
        grouped.setdefault(key, []).append(row)

    table: list[dict[str, Any]] = []
    for (representation, checkpoint, residual_seed), group in sorted(
        grouped.items(), key=lambda item: (str(item[0][1]), str(item[0][0]), str(item[0][2]))
    ):
        sample = group[0]
        entry: dict[str, Any] = {
            "representation": representation,
            "checkpoint": checkpoint,
            "residual_seed": residual_seed,
            "condition_role": sample.get("condition_role"),
            "source_key": sample.get("source_key"),
            "source_functional_response": sample.get("source_functional_response"),
            "source_temporal": sample.get("source_temporal"),
            "mask_mode": sample.get("mask_mode"),
            "total_d": sample.get("total_d"),
            "structured_d": sample.get("structured_d"),
            "residual_d": sample.get("residual_d"),
            "artifact_key": sample.get("artifact_key"),
            "raw_r": _value(group, TARGET_RAW),
            "centered_r": _value(group, TARGET_NEURON_CENTERED),
            "zscored_r": _value(group, TARGET_NEURON_ZSCORED),
            "mean_rate_r": _value(group, TARGET_MEAN_RATE),
            "class_rate_r": _value(group, TARGET_CLASS_RATE),
            "temporal_r": _value(group, TARGET_TEMPORAL),
            "raw_bootstrap_low": _value(group, TARGET_RAW, "bootstrap_low"),
            "raw_bootstrap_high": _value(group, TARGET_RAW, "bootstrap_high"),
            "raw_permutation_p": _value(group, TARGET_RAW, "permutation_p"),
            "zscored_bootstrap_low": _value(group, TARGET_NEURON_ZSCORED, "bootstrap_low"),
            "zscored_bootstrap_high": _value(group, TARGET_NEURON_ZSCORED, "bootstrap_high"),
            "zscored_permutation_p": _value(group, TARGET_NEURON_ZSCORED, "permutation_p"),
            "raw_prediction_r2": _value(group, TARGET_RAW, "prediction_value"),
            "zscored_prediction_r2": _value(group, TARGET_NEURON_ZSCORED, "prediction_value"),
        }
        entry["delta_centering"] = (
            None if entry["centered_r"] is None or entry["raw_r"] is None
            else float(entry["centered_r"]) - float(entry["raw_r"])
        )
        entry["delta_scaling"] = (
            None if entry["zscored_r"] is None or entry["centered_r"] is None
            else float(entry["zscored_r"]) - float(entry["centered_r"])
        )
        table.append(entry)
    return table


def ablation_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The A/B/C/D source ablation at :data:`ABLATION_RESIDUAL_DIM`, per checkpoint and seed.

    Each row carries the measured values and the descriptive difference against arm A
    (``delta_vs_structural``) for both the raw and the neuron z-scored target. No ranking.
    """
    relevant = [
        row for row in rows
        if row.get("condition_role") in (ROLE_SOURCE_ABLATION, ROLE_MASK_DIAGNOSTIC)
        and int(row.get("residual_d") or 0) == ABLATION_RESIDUAL_DIM
    ]
    table = condition_table(relevant)
    baseline: dict[tuple[Any, Any], dict[str, Any]] = {}
    for entry in table:
        if entry["source_key"] == SOURCE_STRUCTURAL and entry["mask_mode"] == MASK_COORDINATE:
            baseline[(entry["checkpoint"], entry["residual_seed"])] = entry
    for entry in table:
        base = baseline.get((entry["checkpoint"], entry["residual_seed"]), {})
        entry["delta_vs_structural_raw"] = (
            None if entry["raw_r"] is None or base.get("raw_r") is None
            else float(entry["raw_r"]) - float(base["raw_r"])
        )
        entry["delta_vs_structural_zscored"] = (
            None if entry["zscored_r"] is None or base.get("zscored_r") is None
            else float(entry["zscored_r"]) - float(base["zscored_r"])
        )
    return table


def summarise_seed_variability(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Mean / SD / per-seed values, grouped by (checkpoint, source, mask, residual_d)."""
    groups: dict[tuple[Any, str, str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        if not row.get("residual_d"):
            continue
        key = (
            row.get("checkpoint"),
            str(row.get("source_key")),
            str(row.get("mask_mode")),
            int(row["residual_d"]),
        )
        groups.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for (checkpoint, source_key, mask_mode, residual_d), group in sorted(
        groups.items(), key=lambda item: (str(item[0][0]), item[0][3], item[0][1], item[0][2])
    ):
        entry: dict[str, Any] = {
            "checkpoint": checkpoint,
            "source_key": source_key,
            "mask_mode": mask_mode,
            "residual_d": residual_d,
            "structured_d": int(group[0].get("structured_d") or 0),
            "total_d": int(group[0].get("total_d") or 0),
            "n_seeds": len({row.get("residual_seed") for row in group}),
            "residual_seeds": sorted(
                int(r["residual_seed"]) for r in group if r.get("residual_seed") is not None
            ),
        }
        compact = condition_table(group)
        for metric in SUMMARY_METRICS:
            values = [
                float(table_row[metric]) for table_row in compact  # type: ignore[index]
                if table_row.get(metric) is not None
            ]
            entry[f"{metric}_mean"] = float(np.mean(values)) if values else None
            entry[f"{metric}_std"] = float(np.std(values, ddof=0)) if values else None
            entry[f"{metric}_values"] = values
        summary.append(entry)
    return summary


def summarise_checkpoint_variability(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Mean / SD / per-checkpoint values for the deterministic conditions (all target variants)."""
    grouped: dict[str, dict[Any, dict[str, Any]]] = {}
    for row in rows:
        if row.get("condition_role") != ROLE_DETERMINISTIC:
            continue
        representation = str(row["representation"])
        target = str(row["target_variant"])
        value = row.get("value")
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(number):
            continue
        grouped.setdefault(representation, {}).setdefault(target, {}).setdefault(
            row.get("checkpoint"), []
        ).append(number)

    summary: list[dict[str, Any]] = []
    for representation, by_target in sorted(grouped.items()):
        for target, by_checkpoint in sorted(by_target.items()):
            per_checkpoint = {
                checkpoint: float(np.mean(values))
                for checkpoint, values in by_checkpoint.items()
            }
            values = np.asarray(list(per_checkpoint.values()), dtype=np.float64)
            summary.append({
                "representation": representation,
                "target_variant": target,
                "n_checkpoints": int(len(per_checkpoint)),
                "checkpoints": sorted(str(c) for c in per_checkpoint),
                "value_mean": float(values.mean()),
                "value_std": float(values.std(ddof=0)),
                "value_values": [float(v) for v in values],
                "per_checkpoint_mean": {str(k): float(v) for k, v in per_checkpoint.items()},
            })
    return summary


def public_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Drop private values so a row is JSON/CSV friendly."""
    return {str(k): v for k, v in row.items() if not str(k).startswith("_")}


def build_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    """The full JSON payload of a source-extension run (tables + summaries + provenance).

    Exposed separately from :func:`write_results` so the figures are rendered from exactly the
    structure that is stored on disk.
    """
    rows = [public_row(r) for r in result.get("rows", [])]
    return {
        "schema": SCHEMA,
        "rows": rows,
        "condition_table": condition_table(rows),
        "ablation_table": ablation_table(rows),
        "summary_seed_variability": summarise_seed_variability(rows),
        "summary_checkpoint_variability": summarise_checkpoint_variability(rows),
        "settings": result.get("settings"),
        "targets": result.get("targets"),
        "conditions": result.get("conditions"),
        "matrices": result.get("matrices"),
        "preprocessing": result.get("preprocessing"),
        "checkpoints": result.get("checkpoints"),
        "compatibility": result.get("compatibility"),
        "source_verification": result.get("source_verification"),
        "metadata": result.get("metadata"),
        "result_columns": list(RESULT_COLUMNS),
    }


def write_results(
    result: Mapping[str, Any],
    *,
    csv_path: str | Path,
    json_path: str | Path,
) -> dict[str, Path]:
    """Write the long result table (CSV) and the full payload (JSON).

    Dedicated filenames only: the previous stages' ``results.csv``/``results.json``/
    ``metadata.json`` and ``rate_robustness_*`` files are never touched.
    """
    payload = build_payload(result)
    rows = payload["rows"]
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
    "SOURCE_STRUCTURAL",
    "SOURCE_FUNCTIONAL",
    "SOURCE_TEMPORAL",
    "SOURCE_FUNCTIONAL_TEMPORAL",
    "SOURCE_KEYS",
    "SOURCE_DESCRIPTIONS",
    "SOURCE_TAGS",
    "MASK_COORDINATE",
    "MASK_BLOCK",
    "ROLE_DETERMINISTIC",
    "ROLE_SOURCE_ABLATION",
    "ROLE_MASK_DIAGNOSTIC",
    "ROLE_SECONDARY_TARGET",
    "ABLATION_RESIDUAL_DIM",
    "ABLATION_SOURCE_KEYS",
    "LARGE_DIM_SOURCE_KEYS",
    "DEFAULT_STRUCTURED_D",
    "DEFAULT_RESIDUAL_DIMS",
    "RESULT_COLUMNS",
    "SUMMARY_METRICS",
    "SourceExtensionError",
    "SourceCondition",
    "StageTargets",
    "build_stage_targets",
    "source_config_for",
    "structured_plus_temporal_dimension",
    "focused_source_conditions",
    "condition_matrix",
    "stage_target_variants",
    "secondary_target_variants",
    "run_source_extension",
    "build_payload",
    "condition_table",
    "ablation_table",
    "summarise_seed_variability",
    "summarise_checkpoint_variability",
    "public_row",
    "write_results",
]
