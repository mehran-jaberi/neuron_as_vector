"""Stable V2 pipeline API: the single high-level entry point of the neuron-vector work.

This module is a **thin orchestration layer** over the modules that already implement the
architecture. It adds no representation logic, no metric and no fitting of its own:

```
                 control panel (scripts/v2_control_panel.py)
                              |
                              v
                     V2Run  (this module)
                              |
      +-----------------------+------------------------+
      |                       |                        |
V2Config                 build phase               evaluate phase
(src/v2_config.py)       (FIT only)                (PROBE only)
      |                       |                        |
      |            NeuronRecordBank                    |
      |            -> StructuredVectorEncoder          |
      |            -> LearnedResidual (optional)       |
      |            -> NeuronVector (n, d)              |
      |            -> NeuronVectorArtifact (frozen)    |
      |                                                |
      +-------------------------------->  Mantel / prediction / controls
                                          (existing machinery, unchanged)
```

Design rules (deliberate)

* **One configuration system.** Every knob comes from :class:`~src.v2_config.V2Config`;
  a *preset* is nothing but a documented bag of configuration leaves, and a ``--override``
  is the repository's existing ``dotted.key=value`` convention.
* **No hidden scientific choices.** Nothing is selected from PROBE: there is no automatic
  dimension, normalisation, seed or mask-mode selection anywhere in this module.
* **Phases are explicit.** "Build the representation" (FIT) and "evaluate the
  representation" (PROBE) are separate operations; building never evaluates.
* **Run identity is deterministic.** ``run_id`` is a hash of the resolved
  representation-relevant configuration, the component schema versions and the checkpoint
  identity - never of the wall clock.
* **Caches are verified, never adapted.** A cached residual must match the checkpoint, the
  source schema (and hash), the residual dimension, the seed *and* the training protocol
  before it is reused; a cached vector artifact must match its run id, dimensions, names
  and content hash.

Safety: FIT is used to build representations and to train the residual; PROBE is used only
to build evaluation targets and metrics; the official TEST split is never opened
(:data:`DATA_POLICY`).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .data import (
    load_shd_recordings,
    make_synthetic_shd,
    make_train_dev_probe_split,
    subset_by_class,
)
from .evaluation import ActivityAccumulatorResult, collect_activity
from .model import RecurrentLIFSNN, architecture_mismatches
from .neuron_record import RECORD_SCHEMA, TEMPORAL_BLOCK, NeuronRecordBank, build_neuron_record_bank
from .neuron_vector import (
    ARTIFACT_SCHEMA,
    SCHEMA as NEURON_VECTOR_SCHEMA,
    NeuronVectorArtifact,
    NeuronVectorError,
    NeuronVectors,
    neuron_vectors_from_config,
)
from .prediction import make_shared_folds
from .rate_robustness import (
    build_response_targets,
    evaluate_target,
    main_target_variants,
)
from .residual import (
    RESIDUAL_SCHEMA,
    SOURCE_SCHEMA,
    ResidualError,
    ResidualResult,
    ResidualSourceConfig,
    ResidualTrainingConfig,
    build_residual_source,
    residual_training_mismatches,
    train_residual,
)
from .source_extension import secondary_target_variants
from .structured_vector import SCHEMA as STRUCTURED_VECTOR_SCHEMA
from .utils import Config, get_device, load_config, parse_scalar, save_json
from .v2_config import (
    DEFAULT_ENABLED_BLOCKS,
    V2Config,
    V2ConfigError,
)
from .vector_capacity import (
    EvaluationSettings,
    build_evaluation_targets,
    fit_rate_reference,
    rate_absdiff_condensed,
)

#: Provenance schema of the pipeline's own metadata (resolved config, results).
SCHEMA = "v2_pipeline/v1"

#: Schema of the deterministic run identifier.
RUN_ID_SCHEMA = "v2_run_id/v1"

#: The one dataset split policy this project allows (documented and enforced).
DATA_POLICY = {
    "fit": "representation construction (record bank, residual training) only",
    "dev": "never used by the control panel (model selection belongs to training)",
    "probe": "evaluation targets and metrics only",
    "test": "never accessed",
    "official_test_loaded": False,
}

#: Repository root (``<root>/src/v2_pipeline.py`` -> ``<root>``).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG_PATH = "configs/neuron_space_baseline.yaml"
DEFAULT_OUT_DIR = "results/neuron_vector_capacity"
DEFAULT_CHECKPOINT = "checkpoints/sweep_l2_0.pt"

#: Evaluation modes the control panel actually supports (nothing else is offered).
SUPPORTED_EVALUATION_MODES: tuple[str, ...] = ("response",)
DEFAULT_EVALUATION_MODE = "response"
DEFAULT_EVALUATION_TARGET = "all"

#: Version identifiers of every component a frozen artifact depends on. A change here
#: invalidates cached artifacts (it is part of the run identity).
COMPONENT_SCHEMAS: dict[str, str] = {
    "record_bank": RECORD_SCHEMA,
    "structured_vector": STRUCTURED_VECTOR_SCHEMA,
    "residual_source": SOURCE_SCHEMA,
    "learned_residual": RESIDUAL_SCHEMA,
    "neuron_vector": NEURON_VECTOR_SCHEMA,
    "neuron_vector_artifact": ARTIFACT_SCHEMA,
}


class PipelineError(ValueError):
    """Raised for an invalid control-panel request or an incompatible cached artifact."""


# --------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------
#: Documented, fully-specified representation presets. Each entry is a bag of ordinary
#: configuration leaves (``vector.*``) applied *before* the CLI overrides, so a preset is
#: never a hidden execution path. Every preset pins the **split** (``structured_d`` +
#: ``learned_residual_d``), the block selection and the residual source switches; the total
#: dimension ``d`` is always derived from that split, so overriding one component (e.g.
#: ``--override vector.structured_d=64``) stays consistent instead of tripping the
#: ``d = structured_d + learned_residual_d`` invariant.
PRESETS: dict[str, dict[str, Any]] = {
    "historical_48": {
        "vector.structured_d": 48,
        "vector.learned_residual_d": 0,
        "vector.enabled_blocks": list(DEFAULT_ENABLED_BLOCKS),
        "vector.residual.enabled": False,
        "vector.residual.source_functional_response": False,
        "vector.residual.source_temporal": False,
        "vector.residual.mask_mode": "coordinate",
    },
    "structured_64": {
        "vector.structured_d": 64,
        "vector.learned_residual_d": 0,
        "vector.enabled_blocks": list(DEFAULT_ENABLED_BLOCKS),
        "vector.residual.enabled": False,
        "vector.residual.source_functional_response": False,
        "vector.residual.source_temporal": False,
        "vector.residual.mask_mode": "coordinate",
    },
    "structured_100": {
        "vector.structured_d": 100,
        "vector.learned_residual_d": 0,
        "vector.enabled_blocks": list(DEFAULT_ENABLED_BLOCKS),
        "vector.residual.enabled": False,
        "vector.residual.source_functional_response": False,
        "vector.residual.source_temporal": False,
        "vector.residual.mask_mode": "coordinate",
    },
    "temporal_structured": {
        "vector.structured_d": 58,
        "vector.learned_residual_d": 0,
        "vector.enabled_blocks": [*DEFAULT_ENABLED_BLOCKS, TEMPORAL_BLOCK],
        "vector.temporal_resolution": 10,
        "vector.residual.enabled": False,
        "vector.residual.source_functional_response": False,
        "vector.residual.source_temporal": False,
        "vector.residual.mask_mode": "coordinate",
    },
    "functional_64": {
        "vector.structured_d": 48,
        "vector.learned_residual_d": 16,
        "vector.enabled_blocks": list(DEFAULT_ENABLED_BLOCKS),
        "vector.residual.enabled": True,
        "vector.residual.source_functional_response": True,
        "vector.residual.source_temporal": False,
        "vector.residual.mask_mode": "coordinate",
    },
    "functional_100": {
        "vector.structured_d": 48,
        "vector.learned_residual_d": 52,
        "vector.enabled_blocks": list(DEFAULT_ENABLED_BLOCKS),
        "vector.residual.enabled": True,
        "vector.residual.source_functional_response": True,
        "vector.residual.source_temporal": False,
        "vector.residual.mask_mode": "coordinate",
    },
}

#: Exact decomposition of every preset, written down so the docs cannot drift.
PRESET_DECOMPOSITIONS: dict[str, str] = {
    "historical_48": "48 = 48 structured + 0 residual (default structural blocks; regression anchor)",
    "structured_64": "64 = 64 structured + 0 residual (level-0 48 + first 16 level-1 coordinates)",
    "structured_100": "100 = 100 structured + 0 residual (level-0 48 + 52 level-1 coordinates)",
    "temporal_structured": (
        "58 = 58 structured + 0 residual (default blocks + 10 coarse temporal bins)"
    ),
    "functional_64": (
        "64 = 48 structured + 16 residual (residual consumes structural + functional-response)"
    ),
    "functional_100": (
        "100 = 48 structured + 52 residual (residual consumes structural + functional-response)"
    ),
}

#: What each preset is for (one line, descriptive only).
PRESET_DESCRIPTIONS: dict[str, str] = {
    "historical_48": "the historical 48-D structural representation (bit-identical anchor)",
    "structured_64": "a wider purely deterministic structural vector",
    "structured_100": "the widest purely deterministic structural vector before projection",
    "temporal_structured": "the deterministic structural vector plus coarse temporal bins",
    "functional_64": "the source-extension arm B decomposition at residual_d = 16",
    "functional_100": "the source-extension arm B decomposition at residual_d = 52",
}


def available_presets() -> tuple[str, ...]:
    """Names of the presets, in documentation order."""
    return tuple(PRESETS)


def preset_summary_rows() -> list[tuple[str, str]]:
    """``(name, "decomposition — description")`` rows for console/documentation output."""
    return [
        (name, f"{PRESET_DECOMPOSITIONS[name]} - {PRESET_DESCRIPTIONS[name]}")
        for name in available_presets()
    ]


def apply_preset(cfg: Config, preset: str) -> Config:
    """Apply a preset's configuration leaves in place (before CLI overrides).

    A preset pins the *split* (``structured_d`` + ``learned_residual_d``), so any total
    dimension the configuration file pinned is removed first: ``d`` is then derived from the
    split, which keeps a later ``--override vector.structured_d=...`` consistent instead of
    contradicting a stale ``vector.d``.
    """
    if preset not in PRESETS:
        raise PipelineError(
            f"unknown preset {preset!r}; available: {list(available_presets())}"
        )
    vector_section = cfg.get("vector")
    if isinstance(vector_section, Mapping) and "d" in vector_section:
        del vector_section["d"]
    for key, value in PRESETS[preset].items():
        cfg.set_path(key, list(value) if isinstance(value, list) else value)
    return cfg


def apply_overrides(cfg: Config, overrides: Sequence[str]) -> Config:
    """Apply the repository's ``dotted.key=value`` overrides (same parsing as ``load_config``)."""
    for override in overrides or ():
        if "=" not in override:
            raise PipelineError(f"override {override!r} must have the form key.subkey=value")
        key, _, value = override.partition("=")
        cfg.set_path(key.strip(), parse_scalar(value))
    return cfg


def _dotted_paths(mapping: Mapping[str, Any], prefix: str = "") -> set[str]:
    """Every dotted path (leaves and intermediate nodes) of a nested mapping."""
    paths: set[str] = set()
    for key, value in mapping.items():
        path = f"{prefix}{key}"
        paths.add(path)
        if isinstance(value, Mapping):
            paths |= _dotted_paths(value, prefix=f"{path}.")
    return paths


def _v2_schema_paths() -> set[str]:
    """Dotted paths of every V2 configuration field (so a new section leaf is not a typo)."""
    from dataclasses import fields as dataclass_fields

    from .v2_config import (
        ExperimentConfig,
        MemoryConfig,
        NetworkConfig,
        PrecisionConfig,
        VectorConfig,
        VectorResidualConfig,
    )

    paths: set[str] = set()
    for prefix, cls in (
        ("vector", VectorConfig),
        ("memory", MemoryConfig),
        ("precision", PrecisionConfig),
        ("network", NetworkConfig),
        ("experiment", ExperimentConfig),
    ):
        for field_info in dataclass_fields(cls):
            paths.add(f"{prefix}.{field_info.name}")
    for field_info in dataclass_fields(VectorResidualConfig):
        paths.add(f"vector.residual.{field_info.name}")
    return paths


def _unknown_override_warnings(before: Mapping[str, Any], overrides: Sequence[str]) -> list[str]:
    """Warn about overrides that match no known key (a typo would otherwise be silent).

    The check is deliberately a *warning*: the override mechanism itself is unchanged
    (``Config.set_path`` accepts any dotted path, as everywhere else in the repository), but
    the control panel reports an override it cannot place instead of pretending it applied.
    """
    warnings: list[str] = []
    if not overrides:
        return warnings
    known = _dotted_paths(before) | _v2_schema_paths()
    for override in overrides:
        key = str(override).partition("=")[0].strip()
        if key in known or any(path.startswith(f"{key}.") for path in known):
            continue
        parent = key.rsplit(".", 1)[0] if "." in key else ""
        if parent and not any(p == parent or p.startswith(f"{parent}.") for p in known):
            warnings.append(
                f"override {key!r} does not match any known configuration key "
                f"(section {parent!r} is not part of this configuration); it was applied to the "
                "raw Config but has no effect on the V2 configuration"
            )
        else:
            warnings.append(
                f"override {key!r} is not a known key of section {parent!r}; it was applied to "
                "the raw Config but is not read by the V2 configuration"
            )
    return warnings


# --------------------------------------------------------------------------
# Run identity
# --------------------------------------------------------------------------
def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def representation_identity(v2: V2Config) -> dict[str, Any]:
    """The representation-relevant part of the configuration (everything else is ignored).

    Only what can change the frozen matrix (or how it must be interpreted) enters the run
    identity: the vector block (dimensions, blocks, temporal resolution, functional-source
    settings, the whole residual protocol), the vector/activity dtypes and the component
    schema versions. Memory/chunk sizes and evaluation settings deliberately do not.
    """
    return {
        "schema": RUN_ID_SCHEMA,
        "components": dict(COMPONENT_SCHEMAS),
        "vector": v2.vector.to_dict(),
        "dtype": {
            "vector_dtype": v2.precision.vector_dtype,
            "activity_dtype": v2.precision.activity_dtype,
        },
    }


def run_identity(v2: V2Config, *, checkpoint_id: str) -> dict[str, Any]:
    """The exact payload the run id hashes (recorded in the resolved configuration too).

    Only *values* enter the identity: the preset **label** is deliberately excluded, so two
    different ways of stating the same resolved configuration (e.g. a preset and the
    equivalent dotted overrides) share one run id - and therefore one cached artifact.
    """
    payload = representation_identity(v2)
    payload["checkpoint"] = str(checkpoint_id)
    return payload


def compute_run_id(
    v2: V2Config,
    *,
    checkpoint_id: str,
    preset: str | None = None,  # accepted for API symmetry; the label is not part of the id
) -> str:
    """Deterministic identity of a frozen representation (``sha256[:16]``).

    Same resolved configuration + same checkpoint -> same id; a representation-relevant change
    to either -> a different id. No wall-clock time enters the hash (a timestamp is recorded
    separately), and the preset name does not either (only the values it resolves to).
    """
    payload = run_identity(v2, checkpoint_id=checkpoint_id)
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def _file_fingerprint(path: Path) -> dict[str, Any]:
    """Content fingerprint of a file (identical convention to the stage scripts)."""
    digest = sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path),
        "size": int(path.stat().st_size),
        "sha256": digest,
        "sha256_16": digest[:16],
    }


def _cache_key(payload: Mapping[str, Any]) -> str:
    """Cache key convention shared with the stage scripts (16 hex chars)."""
    return sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# Resolved configuration
# --------------------------------------------------------------------------
@dataclass
class ResolvedConfig:
    """Everything the pipeline resolved before touching any data.

    ``cfg`` is the raw :class:`~src.utils.Config` (YAML + preset + CLI overrides) so the
    exact configuration can be re-loaded or edited; ``v2`` is its validated V2 view.
    """

    schema: str
    run_id: str
    cfg: Config
    v2: V2Config
    config_path: str
    preset: str | None
    cli_overrides: tuple[str, ...]
    checkpoint_path: Path | None
    checkpoint_sha256_16: str | None
    checkpoint_exists: bool
    out_dir: Path
    evaluation_mode: str
    evaluation_target: str
    n_perm: int
    bootstrap: int
    n_splits: int
    warnings: tuple[str, ...]
    notes: tuple[str, ...] = ()
    created_utc: str = ""

    # -- derived views -------------------------------------------------------
    @property
    def checkpoint_id(self) -> str:
        return self.checkpoint_sha256_16 or f"missing:{self.checkpoint_path}"

    @property
    def d(self) -> int:
        return int(self.v2.vector.d)

    @property
    def structured_d(self) -> int:
        return int(self.v2.vector.structured_d)

    @property
    def residual_d(self) -> int:
        return int(self.v2.vector.learned_residual_d)

    @property
    def residual_enabled(self) -> bool:
        return bool(self.v2.vector.residual.enabled)

    @property
    def temporal_block_requested(self) -> bool:
        """True when the bank must carry the coarse temporal block (same rule as the builder)."""
        return (
            TEMPORAL_BLOCK in self.v2.vector.enabled_blocks
            or bool(self.v2.vector.residual.source_temporal)
        )

    @property
    def residual_protocol(self) -> dict[str, Any]:
        return self.v2.vector.residual.to_dict()

    # -- paths ---------------------------------------------------------------
    @property
    def cache_dir(self) -> Path:
        return self.out_dir / "cache"

    @property
    def residual_dir(self) -> Path:
        return self.out_dir / "residuals"

    @property
    def vector_dir(self) -> Path:
        return self.out_dir / "vectors"

    @property
    def evaluation_dir(self) -> Path:
        return self.out_dir / "evaluations"

    @property
    def vector_artifact_path(self) -> Path:
        return self.vector_dir / f"{self.run_id}.npz"

    @property
    def resolved_config_path(self) -> Path:
        return self.out_dir / "resolved_configs" / f"{self.run_id}.json"

    # -- reporting -----------------------------------------------------------
    def dimension_lines(self) -> list[str]:
        return [
            f"d = {self.d}",
            f"    structured = {self.structured_d}",
            f"    residual   = {self.residual_d}"
            + (" (learned, FIT-only, self-supervised)" if self.residual_d else " (disabled)"),
        ]

    def summary_lines(self) -> list[str]:
        v = self.v2
        r = v.vector.residual
        lines = [
            "V2 resolved configuration",
            "=========================",
            f"run_id            : {self.run_id}   (sha256[:16] of the representation identity)",
            f"preset            : {self.preset or '(none)'}",
            f"config            : {self.config_path}",
            f"cli overrides     : {', '.join(self.cli_overrides) if self.cli_overrides else '(none)'}",
            f"checkpoint        : {self.checkpoint_path} "
            f"(sha256[:16]={self.checkpoint_sha256_16}, exists={self.checkpoint_exists})",
            "",
            "Model",
            "-----",
            f"  n_hidden        : {v.network.n_hidden}",
            f"  model_type      : {v.network.model_type}",
            f"  n_layers        : {v.network.n_layers} "
            f"({'implemented' if v.network.multi_layer_implemented else 'NOT IMPLEMENTED'})",
            f"  n_input / n_bins: {v.snn.n_input} / {v.snn.n_bins} ({v.snn.bin_ms} ms bins)",
            "",
            "Data policy",
            "-----------",
            "  FIT   -> representation construction + residual training",
            "  DEV   -> not used by the control panel",
            "  PROBE -> evaluation targets and metrics only",
            "  TEST  -> never accessed",
            "",
            "Representation",
            "--------------",
            *(f"  {line}" for line in self.dimension_lines()),
            f"  enabled blocks  : {', '.join(v.vector.enabled_blocks)}",
            f"  present blocks  : {', '.join(self.present_block_names())}",
            f"  temporal block  : {'requested' if self.temporal_block_requested else 'absent'}"
            f" (resolution {v.vector.temporal_resolution})",
            f"  network_context : not implemented",
            "",
            "Sources",
            "-------",
            f"  activity (structured encoder block) : "
            f"{'enabled' if 'activity' in v.vector.enabled_blocks else 'disabled'}",
            f"  functional_response (residual view) : "
            f"{'enabled' if r.source_functional_response else 'disabled'}"
            f" (dim {v.vector.functional_source_dim}, "
            f"projection seed {v.vector.functional_projection_seed}, "
            f"normalization {v.vector.functional_source_normalization})",
            f"  temporal (residual view)            : "
            f"{'enabled' if r.source_temporal else 'disabled'}",
            "",
            "Learned residual",
            "----------------",
            f"  enabled         : {r.enabled} (residual_d={self.residual_d})",
            f"  protocol        : {r.epochs} epochs, batch {r.batch_size}, lr {r.learning_rate}, "
            f"hidden {r.hidden_dim}",
            f"  masking         : mode={r.mask_mode}, fraction={r.mask_fraction}, "
            f"mask_seed={r.mask_seed}, min_visible={r.minimum_visible_features}",
            f"  split           : val_fraction={r.val_fraction}, split_seed={r.split_seed}, "
            f"seed={r.seed}",
            f"  standardization : {r.standardization}",
            "",
            "Compute / memory",
            "----------------",
            f"  device          : {v.memory.device}   storage: {v.memory.storage}",
            f"  batch sizes     : train {v.memory.train_batch_size}, eval {v.memory.eval_batch_size}, "
            f"record {v.memory.record_batch_size}",
            f"  chunk sizes     : activity {v.memory.activity_chunk_size}, "
            f"representation {v.memory.representation_chunk_size}",
            f"  dtype           : vector {v.precision.vector_dtype}, activity "
            f"{v.precision.activity_dtype}, model {v.precision.model_dtype}",
            f"  mixed_precision : {v.memory.mixed_precision}   "
            f"max_input_tokens: {v.memory.max_input_tokens}",
            "",
            "Evaluation",
            "----------",
            f"  mode            : {self.evaluation_mode}",
            f"  target          : {self.evaluation_target}",
            f"  statistics      : n_perm={self.n_perm}, bootstrap={self.bootstrap}, "
            f"n_splits={self.n_splits}, seed={v.experiment.seed if v.experiment.seed is not None else 0}",
            "",
            "Artifacts of this run",
            "---------------------",
            f"  vector artifact : {self.vector_artifact_path}",
            f"  resolved config : {self.resolved_config_path}",
            f"  residual cache  : {self.residual_dir}",
            f"  evaluation      : {self.evaluation_dir}",
        ]
        if self.warnings:
            lines += ["", "Warnings", "--------", *(f"  - {w}" for w in self.warnings)]
        if self.notes:
            lines += ["", "Notes", "-----", *(f"  - {n}" for n in self.notes)]
        return lines

    def present_block_names(self) -> list[str]:
        """Blocks the bank will carry (the implemented ones the configuration selects)."""
        from .v2_config import IMPLEMENTED_RECORD_BLOCKS

        present = [b for b in self.v2.vector.enabled_blocks if b in IMPLEMENTED_RECORD_BLOCKS]
        if self.temporal_block_requested and TEMPORAL_BLOCK not in present:
            present.append(TEMPORAL_BLOCK)
        return present

    def to_dict(self) -> dict[str, Any]:
        v = self.v2
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "created_utc": self.created_utc,
            "preset": self.preset,
            "preset_description": PRESET_DESCRIPTIONS.get(self.preset or "", None),
            "preset_decomposition": PRESET_DECOMPOSITIONS.get(self.preset or "", None),
            "config_path": self.config_path,
            "cli_overrides": list(self.cli_overrides),
            "checkpoint": {
                "path": str(self.checkpoint_path) if self.checkpoint_path else None,
                "sha256_16": self.checkpoint_sha256_16,
                "exists": self.checkpoint_exists,
                "id": self.checkpoint_id,
            },
            "representation": {
                "d": self.d,
                "structured_d": self.structured_d,
                "residual_d": self.residual_d,
                "enabled_blocks": list(v.vector.enabled_blocks),
                "present_blocks": self.present_block_names(),
                "temporal_resolution": int(v.vector.temporal_resolution),
                "temporal_block_requested": bool(self.temporal_block_requested),
                "coordinate_order": "structured coordinates first, residual coordinates last",
            },
            "sources": {
                "activity": "activity" in v.vector.enabled_blocks,
                "functional_response": bool(v.vector.residual.source_functional_response),
                "functional_source_dim": int(v.vector.functional_source_dim),
                "functional_projection_seed": int(v.vector.functional_projection_seed),
                "functional_source_normalization": v.vector.functional_source_normalization,
                "temporal": bool(v.vector.residual.source_temporal),
            },
            "residual": self.residual_protocol,
            "v2_config": v.to_dict(),
            "memory": v.memory.to_dict(),
            "precision": v.precision.to_dict(),
            "evaluation": {
                "mode": self.evaluation_mode,
                "target": self.evaluation_target,
                "n_perm": int(self.n_perm),
                "bootstrap": int(self.bootstrap),
                "n_splits": int(self.n_splits),
                "seed": v.experiment.seed if v.experiment.seed is not None else 0,
            },
            "data_policy": dict(DATA_POLICY),
            "component_schemas": dict(COMPONENT_SCHEMAS),
            "run_identity": run_identity(v, checkpoint_id=self.checkpoint_id),
            "run_id_derivation": (
                "sha256[:16] of the canonical JSON of run_identity (resolved values + checkpoint; "
                "the preset label and the wall clock are excluded)"
            ),
            "warnings": list(self.warnings),
            "notes": list(self.notes),
        }


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_config(
    *,
    config_path: str | os.PathLike[str] = DEFAULT_CONFIG_PATH,
    preset: str | None = None,
    overrides: Sequence[str] = (),
    checkpoint: str | os.PathLike[str] | None = None,
    out_dir: str | os.PathLike[str] = DEFAULT_OUT_DIR,
    evaluation_mode: str = DEFAULT_EVALUATION_MODE,
    evaluation_target: str = DEFAULT_EVALUATION_TARGET,
    n_perm: int = 2000,
    bootstrap: int = 500,
    n_splits: int = 5,
    device: str | None = None,
    strict: bool = False,
) -> ResolvedConfig:
    """Resolve + validate the configuration without loading any data.

    This is the function behind ``--dry-run``: it reads the YAML, applies the preset and
    the CLI overrides, resolves the *existing* V2 validation, enforces the pipeline's own
    hard rules (implemented blocks only, single-layer SNN, fp32 model precision) and
    fingerprints the checkpoint (the file bytes, not the model). It never opens a dataset, a
    bank, a residual or the SNN, and it trains nothing. The architecture of the selected
    checkpoint is verified when the representation is built - and, for immediate feedback, by
    :func:`src.v2_panel.validate_state` - not here.
    """
    if evaluation_mode not in SUPPORTED_EVALUATION_MODES:
        raise PipelineError(
            f"evaluation_mode {evaluation_mode!r} is not implemented; "
            f"supported: {list(SUPPORTED_EVALUATION_MODES)}"
        )
    if evaluation_target not in EVALUATION_TARGETS:
        raise PipelineError(
            f"unknown evaluation_target {evaluation_target!r}; use 'all' or one of "
            f"{sorted(k for k in EVALUATION_TARGETS if k != 'all')}"
        )
    cfg = load_config(config_path)
    cfg_before_overrides = cfg.to_dict()
    if preset is not None:
        apply_preset(cfg, preset)
    apply_overrides(cfg, overrides)
    if device is not None:
        cfg.set_path("run.device", str(device))

    try:
        v2 = V2Config.from_config(cfg, strict=strict, warn=False)
    except V2ConfigError as exc:
        raise PipelineError(str(exc)) from exc
    warnings = list(v2.warnings)
    warnings.extend(_unknown_override_warnings(cfg_before_overrides, overrides))

    # -- pipeline hard rules (things this pipeline cannot build) --------------
    unimplemented_blocks = v2.vector.unimplemented_blocks
    if unimplemented_blocks:
        raise PipelineError(
            "the pipeline cannot build the requested record block(s) "
            f"{unimplemented_blocks}; network_context is not implemented. "
            f"available blocks: {sorted(set(v2.vector.enabled_blocks) - set(unimplemented_blocks))}"
        )
    if not v2.network.multi_layer_implemented:
        raise PipelineError(
            f"model.n_layers={v2.network.n_layers} is not implemented (single hidden layer only)"
        )
    if v2.precision.model_dtype != "float32":
        raise PipelineError(
            f"precision.model_dtype={v2.precision.model_dtype!r} is not implemented "
            "(the SNN is trained and evaluated in float32)"
        )
    if v2.memory.mixed_precision:
        raise PipelineError("memory.mixed_precision=true is not implemented")
    if v2.memory.storage != "cpu":
        warnings.append(
            f"memory.storage={v2.memory.storage!r} is not implemented by the current builders; "
            "the record bank and the vectors are produced CPU-side"
        )

    resolved_checkpoint = _resolve_path(
        checkpoint
        if checkpoint is not None
        else cfg.get_path("select_recipe.reuse_seed0_checkpoint", DEFAULT_CHECKPOINT)
    )
    exists = resolved_checkpoint is not None and Path(resolved_checkpoint).exists()
    fingerprint = _file_fingerprint(Path(resolved_checkpoint)) if exists else None
    checkpoint_id = fingerprint["sha256_16"] if fingerprint else f"missing:{resolved_checkpoint}"
    run_id = compute_run_id(v2, checkpoint_id=checkpoint_id, preset=preset)

    notes: list[str] = []
    if not exists:
        notes.append(
            f"checkpoint {resolved_checkpoint} does not exist yet; the run id uses the "
            "'missing:' identifier and a build would fail until it is present"
        )
    if v2.vector.residual.enabled and not v2.vector.residual.source_functional_response and not (
        v2.vector.residual.source_temporal
    ):
        notes.append(
            "the learned residual consumes the structural source only "
            "(no functional-response / temporal source enabled)"
        )

    return ResolvedConfig(
        schema=SCHEMA,
        run_id=run_id,
        cfg=cfg,
        v2=v2,
        config_path=str(config_path),
        preset=preset,
        cli_overrides=tuple(str(o) for o in overrides),
        checkpoint_path=Path(resolved_checkpoint) if resolved_checkpoint else None,
        checkpoint_sha256_16=fingerprint["sha256_16"] if fingerprint else None,
        checkpoint_exists=bool(exists),
        out_dir=_resolve_path(out_dir),
        evaluation_mode=evaluation_mode,
        evaluation_target=evaluation_target,
        n_perm=int(n_perm),
        bootstrap=int(bootstrap),
        n_splits=int(n_splits),
        warnings=tuple(warnings),
        notes=tuple(notes),
        created_utc=_now_utc(),
    )


def _resolve_path(path: str | os.PathLike[str] | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:  # pragma: no cover - outside the repo
        return str(path)


# --------------------------------------------------------------------------
# Dry run
# --------------------------------------------------------------------------
def dry_run_report(resolved: ResolvedConfig) -> dict[str, Any]:
    """The plan of what a build/evaluate would do, without touching any data."""
    r = resolved
    residual_action = (
        f"train (200-epoch default protocol) or reuse the cached artifact for "
        f"d_res={r.residual_d}, seed={r.v2.vector.residual.seed}"
        if r.residual_enabled
        else "skipped (residual_d = 0: no model is instantiated)"
    )
    data_source = "synthetic" if bool(r.cfg.get_path("run.synthetic", False)) else (
        str(Path(str(r.cfg.get_path("paths.data_dir", "data"))) / "shd_train.h5")
    )
    steps = [
        f"1. load the frozen checkpoint {r.checkpoint_path} (never modified)",
        f"2. stream label-free FIT activity ({data_source}) through the existing accumulator, "
        f"batch {r.v2.memory.record_batch_size}, cached under {r.cache_dir}",
        f"3. build the NeuronRecordBank (blocks: {', '.join(r.present_block_names())})",
        f"4. residual: {residual_action}",
        f"5. compose the neuron vectors z = [z_structured, z_residual] -> (n_neurons, {r.d})",
        f"6. freeze the artifact at {r.vector_artifact_path}",
    ]
    evaluation_steps = [
        f"1. load the frozen vector artifact for run_id={r.run_id} (verified)",
        f"2. collect label-free FIT activity (rate reference) and PROBE activity (targets)",
        f"3. evaluate target '{r.evaluation_target}' in mode '{r.evaluation_mode}' with "
        f"n_perm={r.n_perm}, bootstrap={r.bootstrap}, n_splits={r.n_splits}",
        f"4. write {r.evaluation_dir}/<evaluation key>.json",
    ]
    return {
        "schema": SCHEMA,
        "run_id": r.run_id,
        "phases": {"build": steps, "evaluate": evaluation_steps},
        "checkpoint": {
            "path": str(r.checkpoint_path),
            "exists": r.checkpoint_exists,
            "sha256_16": r.checkpoint_sha256_16,
        },
        "data": {
            "source": data_source,
            "strategy": r.cfg.get_path("data.prefer_speaker_aware", True),
            "split_seed": r.cfg.get_path("data.split_seed", None),
            "dev_fraction": r.cfg.get_path("data.dev_fraction", 0.1),
            "probe_fraction": r.cfg.get_path("data.probe_fraction", 0.1),
            "n_fit": r.cfg.get_path("experiment.n_fit", None),
            "n_probe": r.cfg.get_path("experiment.n_probe", None),
            "policy": dict(DATA_POLICY),
        },
        "dimension": {
            "d": r.d,
            "structured_d": r.structured_d,
            "residual_d": r.residual_d,
            "expression": f"{r.d} = {r.structured_d} structured + {r.residual_d} residual",
        },
        "blocks": r.present_block_names(),
        "residual": (r.residual_protocol if r.residual_enabled else None),
        "resolved_config_path": str(r.resolved_config_path),
        "vector_artifact_path": str(r.vector_artifact_path),
        "notes": list(r.notes),
        "warnings": list(r.warnings),
    }


# --------------------------------------------------------------------------
# Data (FIT / PROBE only; TEST is never opened)
# --------------------------------------------------------------------------
def build_fit_probe_recordings(resolved: ResolvedConfig, *, max_per_class: int | None = None) -> dict[str, Any]:
    """FIT and PROBE recordings using the canonical split parameters (never TEST).

    Mirrors the stage scripts' split resolution exactly: ``data.split_seed`` (falling back
    to ``seed``), ``data.dev_fraction``/``data.probe_fraction``,
    ``data.prefer_speaker_aware``; it reads only the SHD **training** file (or the synthetic
    generator), so the official test file cannot be opened from here.
    """
    cfg = resolved.cfg
    seed = int(cfg.get_path("seed", 0))
    split_seed_raw = cfg.get_path("data.split_seed", None)
    split_seed = seed if split_seed_raw is None else int(split_seed_raw)
    dev_fraction = float(cfg.get_path("data.dev_fraction", 0.1))
    probe_fraction = float(cfg.get_path("data.probe_fraction", 0.1))
    prefer_speaker = bool(cfg.get_path("data.prefer_speaker_aware", True))
    debug = bool(cfg.get_path("run.debug", False))
    synthetic = bool(cfg.get_path("run.synthetic", False))

    if synthetic:
        n_channels = int(cfg.get_path("run.synthetic_n_channels", 40))
        n_classes = int(cfg.get_path("run.synthetic_n_classes", 5))
        cfg.set_path("model.n_input", n_channels)
        cfg.set_path("model.n_output", n_classes)
        cfg.set_path("train.n_classes", n_classes)
        rec = make_synthetic_shd(
            n_samples=int(cfg.get_path("run.synthetic_n_samples", 600)),
            n_classes=n_classes,
            n_channels=n_channels,
            n_bins=int(cfg.get_path("model.n_bins", 500)),
            bin_ms=float(cfg.get_path("model.bin_ms", 2.0)),
            seed=seed,
        )
        source_name = "synthetic_train"
    else:
        train_h5 = PROJECT_ROOT / str(cfg.get_path("paths.data_dir", "data")) / "shd_train.h5"
        if not train_h5.exists():
            raise PipelineError(
                f"{train_h5} not found; download SHD first (see README) or use "
                "--override run.synthetic=true"
            )
        rec = load_shd_recordings(train_h5, layout=str(cfg.get_path("data.layout", "auto")))
        source_name = "shd_train"

    fit, dev, probe, split_info = make_train_dev_probe_split(
        rec,
        dev_fraction=dev_fraction,
        probe_fraction=probe_fraction,
        seed=split_seed,
        prefer_speaker_aware=prefer_speaker,
    )
    if debug:
        fit = subset_by_class(fit, max_per_class=40, seed=split_seed)
        if len(probe) > 0:
            probe = subset_by_class(probe, max_per_class=20, seed=split_seed)
    limit = max_per_class if max_per_class is not None else cfg.get_path("data.max_per_class", None)
    if limit:
        fit = subset_by_class(fit, max_per_class=int(limit), seed=split_seed)
    if len(probe) == 0:
        raise PipelineError(
            "the resolved split has no PROBE recordings (no speaker metadata?); "
            "the evaluation needs a held-out PROBE split"
        )
    return {
        "fit": fit,
        "dev": dev,
        "probe": probe,
        "split_info": dict(split_info),
        "source": source_name,
        "split_seed": split_seed,
        "test_loaded": False,
    }


def _load_or_collect_activity(
    *,
    role: str,
    model: RecurrentLIFSNN,
    recordings: Any,
    device: Any,
    cache_dir: Path,
    cache_common: Mapping[str, Any],
    use_cache: bool,
    with_labels: bool,
    batch_size: int,
    verbose: bool = False,
) -> tuple[ActivityAccumulatorResult, bool]:
    """Return ``(accumulator, from_cache)`` for one role, using the shared cache convention."""
    if role not in ("fit", "probe"):
        raise PipelineError(f"activity role must be 'fit' or 'probe', got {role!r}")
    meta = {**dict(cache_common), "role": role, "n": len(recordings), "with_labels": bool(with_labels)}
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key({"role": role, **meta})
    npz_path = cache_dir / f"{role}_activity_{key}.npz"
    meta_path = cache_dir / f"{role}_activity_{key}.json"
    if use_cache and npz_path.exists() and meta_path.exists():
        stored = json.loads(meta_path.read_text(encoding="utf-8"))
        if stored == meta:
            if verbose:
                print(f"[cache] {role} activity <- {npz_path.name}")
            return ActivityAccumulatorResult.load(str(npz_path)), True
    indices = np.arange(len(recordings), dtype=np.int64)
    t0 = time.time()
    result = collect_activity(
        model,
        recordings,
        indices,
        device=device,
        batch_size=int(batch_size),
        n_classes=int(model.cfg.n_output),
        with_labels=bool(with_labels),
        collect_voltage=False,
    )
    if verbose:
        print(f"[activity] {role}: n={result.n_samples} in {time.time() - t0:.1f}s")
    if use_cache:
        result.save(str(npz_path))
        meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return result, False


# --------------------------------------------------------------------------
# Build phase
# --------------------------------------------------------------------------
@dataclass
class BuildResult:
    """Outcome of the representation build (FIT only - no PROBE, no metrics)."""

    run_id: str
    artifact_path: Path
    n_neurons: int
    d: int
    structured_d: int
    residual_d: int
    feature_names: tuple[str, ...]
    bank_blocks: tuple[str, ...]
    checkpoint_sha256_16: str
    fit_activity_cached: bool
    residual_trained_now: bool | None
    residual_cache_path: Path | None
    residual_protocol_mismatches: dict[str, tuple[Any, Any]]
    seconds: float
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def dimension_expression(self) -> str:
        return f"{self.d} = {self.structured_d} structured + {self.residual_d} residual"

    def summary_lines(self) -> list[str]:
        lines = [
            f"[build] run_id={self.run_id}",
            f"[build] representation shape: ({self.n_neurons}, {self.d})  [{self.dimension_expression}]",
            f"[build] record bank blocks available: {', '.join(self.bank_blocks)}",
            f"[build] artifact: {self.artifact_path}",
            f"[build] fit activity cache: {'hit' if self.fit_activity_cached else 'collected'}",
        ]
        if self.residual_d:
            lines.append(
                f"[build] residual: d_res={self.residual_d} "
                f"{'trained now' if self.residual_trained_now else 'reused from cache'} "
                f"({self.residual_cache_path})"
            )
        else:
            lines.append("[build] residual: none (learned_residual_d = 0)")
        lines.append(f"[build] elapsed: {self.seconds:.1f}s")
        return lines


def _load_or_train_residual(
    *,
    bank: NeuronRecordBank,
    source_config: ResidualSourceConfig,
    training_config: ResidualTrainingConfig,
    cache_path: Path,
    use_cache: bool,
    verbose: bool = False,
) -> tuple[ResidualResult, bool, dict[str, tuple[Any, Any]]]:
    """Train or reuse a residual artifact, verifying schema *and* training protocol.

    The schema checks come from :meth:`ResidualResult.load`; the protocol check is
    :func:`residual_training_mismatches`, because the loader deliberately does not compare
    hyper-parameters. A cached artifact whose protocol differs is rejected (and retrained)
    rather than silently reused, and the mismatching fields are reported in the run output.
    """
    mismatches: dict[str, tuple[Any, Any]] = {}
    if use_cache and cache_path.exists():
        source = build_residual_source(bank, source_config)
        try:
            residual = ResidualResult.load(
                cache_path,
                expected_feature_names=source.feature_names,
                expected_residual_dim=training_config.residual_dim,
                map_location="cpu",
            )
        except (ResidualError, RuntimeError, KeyError) as exc:
            if verbose:
                print(f"[cache] residual cache rejected ({exc}); retraining")
            residual = None
        else:
            mismatches = residual_training_mismatches(residual, training_config)
            if mismatches:
                if verbose:
                    print(
                        "[cache] residual cache protocol mismatch "
                        f"{ {k: v for k, v in mismatches.items()} }; retraining"
                    )
                residual = None
            elif verbose:
                print(
                    f"[cache] residual d_res={training_config.residual_dim} "
                    f"seed={training_config.seed} <- {cache_path.name} (protocol verified)"
                )
            if residual is not None:
                return residual, False, mismatches
    source = build_residual_source(bank, source_config)
    t0 = time.time()
    residual = train_residual(bank, config=training_config, source=source)
    if verbose:
        print(
            f"[residual] d_res={training_config.residual_dim} seed={training_config.seed} "
            f"trained in {time.time() - t0:.1f}s (best val {residual.best_val_loss:.4f})"
        )
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        residual.save(cache_path)
    return residual, True, mismatches


def _cache_common(
    resolved: ResolvedConfig,
    model: RecurrentLIFSNN,
    recordings_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """The shared cache-key inputs (identical convention to the stage scripts)."""
    checkpoint_path = Path(resolved.checkpoint_path)
    return {
        "checkpoint": resolved.checkpoint_sha256_16,
        "checkpoint_path": _relative_or_absolute(checkpoint_path),
        "n_bins": int(model.cfg.n_bins),
        "bin_ms": float(model.cfg.bin_ms),
        "split_seed": int(recordings_bundle["split_seed"]),
        "split_strategy": recordings_bundle["split_info"].get("strategy"),
        "source": recordings_bundle["source"],
    }


def residual_cache_path(
    resolved: ResolvedConfig,
    *,
    cache_common: Mapping[str, Any],
    source_schema_hash: str,
    residual_d: int,
    seed: int,
    mask_mode: str,
) -> Path:
    """Deterministic cache path of a residual artifact (shared with the stage scripts)."""
    base = _cache_key(
        {
            **dict(cache_common),
            "schema_hash": str(source_schema_hash),
            "blocks": list(resolved.v2.vector.enabled_blocks),
        }
    )
    suffix = "_maskblock" if str(mask_mode) == "block" else ""
    return resolved.residual_dir / f"{base}_dres{int(residual_d)}_seed{int(seed)}{suffix}.pt"


def _load_checkpoint(resolved: ResolvedConfig) -> RecurrentLIFSNN:
    if resolved.checkpoint_path is None or not Path(resolved.checkpoint_path).exists():
        raise PipelineError(
            f"checkpoint not found: {resolved.checkpoint_path}; pass --checkpoint or "
            "set select_recipe.reuse_seed0_checkpoint"
        )
    model, _extra = RecurrentLIFSNN.load(str(resolved.checkpoint_path), map_location="cpu")
    mismatch = architecture_mismatches(resolved.cfg.get_path("model", {}) or {}, model)
    if mismatch:
        detail = ", ".join(
            f"model.{k}: config={a!r} vs checkpoint={b!r}" for k, (a, b) in mismatch.items()
        )
        raise PipelineError(f"config/checkpoint architecture mismatch: {detail}")
    return model


def build_representation(
    resolved: ResolvedConfig,
    *,
    use_cache: bool = True,
    verbose: bool = False,
) -> BuildResult:
    """Build the frozen ``(n_neurons, d)`` representation from FIT only."""
    started = time.time()
    v2 = resolved.v2
    bundle = build_fit_probe_recordings(resolved)
    model = _load_checkpoint(resolved)
    device = None if v2.memory.device in ("auto", None) else (
        "cpu" if v2.memory.device == "cpu" else get_device(prefer_cuda=True)
    )
    common = _cache_common(resolved, model, bundle)
    fit_activity, fit_cached = _load_or_collect_activity(
        role="fit",
        model=model,
        recordings=bundle["fit"],
        device=device,
        cache_dir=resolved.cache_dir,
        cache_common=common,
        use_cache=use_cache,
        with_labels=False,
        batch_size=int(v2.memory.record_batch_size),
        verbose=verbose,
    )
    bank = build_neuron_record_bank(model, activity=fit_activity, config=v2)

    residual: ResidualResult | None = None
    residual_trained_now: bool | None = None
    residual_path: Path | None = None
    mismatches: dict[str, tuple[Any, Any]] = {}
    if resolved.residual_d > 0:
        source_config = ResidualSourceConfig.from_v2_config(v2)
        source = build_residual_source(bank, source_config)
        training = ResidualTrainingConfig.from_v2_config(v2)
        residual_path = residual_cache_path(
            resolved,
            cache_common=common,
            source_schema_hash=source.schema_hash,
            residual_d=resolved.residual_d,
            seed=int(training.seed),
            mask_mode=str(training.mask_mode),
        )
        residual, residual_trained_now, mismatches = _load_or_train_residual(
            bank=bank,
            source_config=source_config,
            training_config=training,
            cache_path=residual_path,
            use_cache=use_cache,
            verbose=verbose,
        )

    try:
        vectors: NeuronVectors = neuron_vectors_from_config(
            v2, bank, residual=residual, chunk_size=int(v2.memory.representation_chunk_size)
        )
    except (NeuronVectorError, ResidualError) as exc:
        raise PipelineError(str(exc)) from exc

    provenance = {
        "schema": SCHEMA,
        "phase": "build",
        "created_utc": _now_utc(),
        "run_id": resolved.run_id,
        "preset": resolved.preset,
        "cli_overrides": list(resolved.cli_overrides),
        "config_path": resolved.config_path,
        "checkpoint": {
            "path": str(resolved.checkpoint_path),
            "sha256_16": resolved.checkpoint_sha256_16,
        },
        "data": {
            "source": bundle["source"],
            "split_strategy": bundle["split_info"].get("strategy"),
            "split_seed": int(bundle["split_seed"]),
            "n_fit": int(len(bundle["fit"])),
            "fit_activity_cached": bool(fit_cached),
            "policy": dict(DATA_POLICY),
            "test_loaded": False,
        },
        "dimensions": {
            "d": resolved.d,
            "structured_d": resolved.structured_d,
            "residual_d": resolved.residual_d,
            "expression": f"{resolved.d} = {resolved.structured_d} structured + {resolved.residual_d} residual",
        },
        "bank": dict(bank.provenance),
        "residual_protocol": resolved.residual_protocol if resolved.residual_d else None,
        "residual_cache": str(residual_path) if residual_path is not None else None,
        "residual_cache_protocol_mismatches": {
            str(k): [v[0], v[1]] for k, v in mismatches.items()
        },
        "component_schemas": dict(COMPONENT_SCHEMAS),
        "composition": dict(vectors.provenance),
    }
    artifact = NeuronVectorArtifact.from_vectors(vectors, run_id=resolved.run_id, provenance=provenance)
    artifact.save(resolved.vector_artifact_path)

    result = BuildResult(
        run_id=resolved.run_id,
        artifact_path=resolved.vector_artifact_path,
        n_neurons=artifact.n_neurons,
        d=artifact.d,
        structured_d=artifact.structured_d,
        residual_d=artifact.residual_d,
        feature_names=artifact.feature_names,
        bank_blocks=tuple(bank.block_names),
        checkpoint_sha256_16=str(resolved.checkpoint_sha256_16),
        fit_activity_cached=bool(fit_cached),
        residual_trained_now=residual_trained_now,
        residual_cache_path=residual_path,
        residual_protocol_mismatches=mismatches,
        seconds=time.time() - started,
        provenance=provenance,
    )
    return result


def load_vector_artifact(resolved: ResolvedConfig, *, verify_hash: bool = True) -> NeuronVectorArtifact:
    """Load this run's frozen artifact, verifying identity, dimensions and content hash."""
    return NeuronVectorArtifact.load(
        resolved.vector_artifact_path,
        expected_run_id=resolved.run_id,
        expected_d=resolved.d,
        expected_structured_d=resolved.structured_d,
        expected_residual_d=resolved.residual_d,
        verify_hash=verify_hash,
    )


# --------------------------------------------------------------------------
# Evaluate phase
# --------------------------------------------------------------------------
@dataclass
class EvaluationResult:
    """Outcome of the PROBE evaluation of a frozen representation."""

    run_id: str
    artifact_path: Path
    evaluation_path: Path
    mode: str
    target: str
    rows: list[dict[str, Any]]
    settings: dict[str, Any]
    targets: dict[str, Any]
    seconds: float
    warnings: tuple[str, ...] = ()

    def summary_lines(self) -> list[str]:
        lines = [
            f"[evaluate] run_id={self.run_id} mode={self.mode} target={self.target}",
            f"[evaluate] artifact: {self.artifact_path}",
            f"[evaluate] statistics: n_perm={self.settings['n_perm']}, "
            f"bootstrap={self.settings['bootstrap']}, n_splits={self.settings['n_splits']}, "
            f"seed={self.settings['seed']}",
            f"[evaluate] results: {self.evaluation_path}",
        ]
        for row in self.rows:
            value = row.get("value")
            low = row.get("bootstrap_low")
            high = row.get("bootstrap_high")
            interval = ""
            if low is not None and high is not None and np.isfinite(float(low)) and np.isfinite(float(high)):
                interval = f" CI[{float(low):+.3f}, {float(high):+.3f}]"
            lines.append(
                f"  target {row['target_variant']:<18s} mantel_spearman_r = "
                f"{value if value is None else f'{float(value):+.4f}'}{interval}"
                f" (p={row.get('permutation_p')})"
            )
        lines.append(f"[evaluate] elapsed: {self.seconds:.1f}s")
        return lines


def _known_evaluation_targets() -> dict[str, str]:
    """``{target name: family}`` for every target the evaluation layer really supports."""
    names: dict[str, str] = {"all": "all main response variants"}
    for variant in main_target_variants():
        names[str(variant.name)] = "main response target"
    for variant in secondary_target_variants():
        names[str(variant.name)] = "secondary (class-conditioned) target"
    return names


EVALUATION_TARGETS: dict[str, str] = _known_evaluation_targets()


def _select_variants(target: str):
    """Resolve the evaluation target name to concrete target variants (no PROBE access)."""
    if target not in EVALUATION_TARGETS:
        raise PipelineError(
            f"unknown evaluation target {target!r}; use 'all' or one of "
            f"{sorted(k for k in EVALUATION_TARGETS if k != 'all')}"
        )
    mains = {v.name: v for v in main_target_variants()}
    secondary = {v.name: v for v in secondary_target_variants()}
    if target == "all":
        return tuple(mains.values()), ()
    if target in mains:
        return (mains[target],), ()
    return (), (secondary[target],)


def evaluate_representation(
    resolved: ResolvedConfig,
    *,
    artifact: NeuronVectorArtifact | None = None,
    use_cache: bool = True,
    verbose: bool = False,
) -> EvaluationResult:
    """Evaluate a frozen representation against PROBE targets (existing metrics only).

    The representation is *loaded*, never rebuilt or adapted: the artifact's run id,
    dimensions and content hash are verified first. Only PROBE is used for the targets and
    only FIT for the rate reference; TEST is never opened.
    """
    started = time.time()
    v2 = resolved.v2
    artifact = artifact or load_vector_artifact(resolved)
    main_variants, secondary_variants = _select_variants(resolved.evaluation_target)

    bundle = build_fit_probe_recordings(resolved)
    model = _load_checkpoint(resolved)
    device = None if v2.memory.device in ("auto", None) else (
        "cpu" if v2.memory.device == "cpu" else get_device(prefer_cuda=True)
    )
    common = _cache_common(resolved, model, bundle)
    batch = int(v2.memory.record_batch_size)
    fit_activity, _ = _load_or_collect_activity(
        role="fit",
        model=model,
        recordings=bundle["fit"],
        device=device,
        cache_dir=resolved.cache_dir,
        cache_common=common,
        use_cache=use_cache,
        with_labels=False,
        batch_size=batch,
        verbose=verbose,
    )
    probe_activity, probe_cached = _load_or_collect_activity(
        role="probe",
        model=model,
        recordings=bundle["probe"],
        device=device,
        cache_dir=resolved.cache_dir,
        cache_common=common,
        use_cache=use_cache,
        with_labels=True,
        batch_size=batch,
        verbose=verbose,
    )

    response = build_response_targets(probe_activity, probe_split_label="probe")
    settings = EvaluationSettings(
        n_perm=int(resolved.n_perm),
        bootstrap=int(resolved.bootstrap),
        k_values=(),
        seed=int(v2.experiment.seed if v2.experiment.seed is not None else 0),
        n_splits=int(resolved.n_splits),
        checkpoint=resolved.run_id,
        tag=f"{resolved.preset or 'config'}_v2panel",
    )
    rates_absdiff = rate_absdiff_condensed(fit_rate_reference(fit_activity))
    cv = make_shared_folds(artifact.n_neurons, settings.n_splits, settings.seed)
    functional = None
    if secondary_variants:
        functional = build_evaluation_targets(
            probe_activity,
            probe_split_label="probe",
            n_psth_bins=int(resolved.cfg.get_path("fingerprint.n_psth_bins", 10)),
            min_spikes_for_latency=float(resolved.cfg.get_path("fingerprint.min_spikes_for_latency", 1.0)),
        )

    def _space(name: str):
        if name in response.spaces:
            return response.get(name)
        if functional is None:  # pragma: no cover - guarded by _select_variants
            raise PipelineError(f"target {name!r} needs the secondary target builder")
        return functional.get(name)

    description = {
        "representation": f"run:{resolved.run_id}",
        "representation_kind": "structured_plus_residual" if resolved.residual_d else "structured",
        "representation_role": "control_panel",
        "run_id": resolved.run_id,
        "checkpoint": str(resolved.checkpoint_path),
        "checkpoint_sha256_16": resolved.checkpoint_sha256_16,
        "preset": resolved.preset,
        "evaluation_mode": resolved.evaluation_mode,
        "total_d": resolved.d,
        "structured_d": resolved.structured_d,
        "residual_d": resolved.residual_d,
        "enabled_blocks": list(v2.vector.enabled_blocks),
        "source_functional_response": bool(v2.vector.residual.source_functional_response),
        "source_temporal": bool(v2.vector.residual.source_temporal),
        "mask_mode": v2.vector.residual.mask_mode,
        "probe_split": response.metadata.get("probe_split"),
        "fit_activity_cached": True,
        "probe_activity_cached": bool(probe_cached),
    }
    rows: list[dict[str, Any]] = []
    for variant in (*main_variants, *secondary_variants):
        rows.append(
            evaluate_target(
                artifact.X,
                artifact.feature_names,
                _space(variant.name),
                variant,
                settings=settings,
                probe_n=response.n_stimuli,
                rates_absdiff=rates_absdiff,
                cv=cv,
                description={
                    **description,
                    "n_perm": int(settings.n_perm),
                    "bootstrap_n": int(settings.bootstrap),
                },
            )
        )

    evaluation_payload = {
        "schema": SCHEMA,
        "phase": "evaluate",
        "created_utc": _now_utc(),
        "run_id": resolved.run_id,
        "mode": resolved.evaluation_mode,
        "target": resolved.evaluation_target,
        "artifact": {
            "path": str(resolved.vector_artifact_path),
            "summary": artifact.summary(),
        },
        "settings": settings.to_dict(),
        "targets": {
            "response": response.summary(),
            "secondary": (functional.summary() if functional is not None else None),
        },
        "rows": rows,
        "data_policy": dict(DATA_POLICY),
        "notes": [
            "The representation was loaded from a frozen artifact and never rebuilt or adapted.",
            "No parameter of the representation was selected or tuned from PROBE.",
        ],
    }
    key = _cache_key(
        {
            "run_id": resolved.run_id,
            "mode": resolved.evaluation_mode,
            "target": resolved.evaluation_target,
            "settings": settings.to_dict(),
        }
    )
    resolved.evaluation_dir.mkdir(parents=True, exist_ok=True)
    evaluation_path = resolved.evaluation_dir / f"evaluation_{key}.json"
    save_json(evaluation_payload, evaluation_path)

    result = EvaluationResult(
        run_id=resolved.run_id,
        artifact_path=resolved.vector_artifact_path,
        evaluation_path=evaluation_path,
        mode=resolved.evaluation_mode,
        target=resolved.evaluation_target,
        rows=rows,
        settings=settings.to_dict(),
        targets=evaluation_payload["targets"],
        seconds=time.time() - started,
        warnings=tuple(resolved.warnings),
    )
    return result


# --------------------------------------------------------------------------
# The stable high-level API
# --------------------------------------------------------------------------
@dataclass
class V2Run:
    """One resolved V2 run: build the representation, then (separately) evaluate it.

    ```python
    from src.v2_pipeline import V2Run

    run = V2Run.from_config(preset="functional_64")   # resolve only, no data
    run.resolved.summary_lines()                      # inspect
    build = run.build_representation()                # FIT only
    evaluation = run.evaluate()                       # PROBE only
    ```
    """

    resolved: ResolvedConfig

    @classmethod
    def from_config(cls, **kwargs: Any) -> "V2Run":
        """Resolve the configuration (see :func:`resolve_config`) without loading data."""
        return cls(resolved=resolve_config(**kwargs))

    @classmethod
    def from_resolved(cls, resolved: ResolvedConfig) -> "V2Run":
        return cls(resolved=resolved)

    # -- convenience ---------------------------------------------------------
    @property
    def run_id(self) -> str:
        return self.resolved.run_id

    def dry_run(self) -> dict[str, Any]:
        return dry_run_report(self.resolved)

    def build_representation(self, *, use_cache: bool = True, verbose: bool = False) -> BuildResult:
        result = build_representation(self.resolved, use_cache=use_cache, verbose=verbose)
        save_json(self.resolved.to_dict(), self.resolved.resolved_config_path)
        return result

    def evaluate(
        self,
        *,
        artifact: NeuronVectorArtifact | None = None,
        use_cache: bool = True,
        verbose: bool = False,
    ) -> EvaluationResult:
        return evaluate_representation(self.resolved, artifact=artifact, use_cache=use_cache, verbose=verbose)


# --------------------------------------------------------------------------
# Test-suite status (for the parent summary; no science, no TEST)
# --------------------------------------------------------------------------
def _pytest_collected_count(*, timeout: int = 1800) -> int | None:
    """Collected test count from a ``pytest --collect-only -q`` subprocess (or ``None``)."""
    import re

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - defensive
        return None
    output = f"{proc.stdout}\n{proc.stderr}"
    counts = [int(m) for m in re.findall(r"^tests/.*:\s*(\d+)\s*$", output, flags=re.MULTILINE)]
    if counts:
        return int(sum(counts))
    explicit = re.search(r"(\d+)\s+tests?\s+collected", output)
    return int(explicit.group(1)) if explicit else None


def test_suite_status(*, run_tests: bool = False, timeout: int = 3600) -> dict[str, Any]:
    """Collected test count, and optionally the full-suite result (subprocess pytest).

    ``run_tests=False`` collects only (fast); ``run_tests=True`` runs the full suite and
    records the outcome so later invocations can report it without re-running. The
    repository's pytest options are deliberately quiet, so a missing summary line is not a
    failure signal: the process exit code is authoritative and the *count* is then taken from
    a collection pass (``result_source`` records which path was used).
    """
    import re

    record_path = PROJECT_ROOT / "results" / "neuron_vector_capacity" / "last_test_run.json"
    status: dict[str, Any] = {
        "collected": None,
        "passed": None,
        "failed": None,
        "exit_code": None,
        "result_source": None,
        "mode": "run" if run_tests else "collect-only",
        "command": None,
        "recorded_utc": None,
        "timed_out": False,
    }
    args = ["--collect-only", "-q"] if not run_tests else ["-q"]
    command = [sys.executable, "-m", "pytest", *args]
    status["command"] = " ".join(["uv", "run", "pytest", *args])
    try:
        proc = subprocess.run(
            command, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=timeout
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - defensive
        status["error"] = str(exc)
        return status
    output = f"{proc.stdout}\n{proc.stderr}"
    counts = [int(m) for m in re.findall(r"^tests/.*:\s*(\d+)\s*$", output, flags=re.MULTILINE)]
    if counts:
        status["collected"] = int(sum(counts))
    explicit = re.search(r"(\d+)\s+tests?\s+collected", output)
    if explicit:
        status["collected"] = int(explicit.group(1))
    if run_tests:
        passed = re.search(r"(\d+)\s+passed", output)
        failed = re.search(r"(\d+)\s+failed", output)
        errored = re.search(r"(\d+)\s+error", output)
        status["passed"] = int(passed.group(1)) if passed else None
        failures = 0
        if failed:
            failures += int(failed.group(1))
        if errored:
            failures += int(errored.group(1))
        status["failed"] = failures if (failed or errored) else None
        status["exit_code"] = int(proc.returncode)
        status["recorded_utc"] = _now_utc()
        if status["passed"] is not None:
            status["result_source"] = "summary_line"
            if status["failed"] is None:
                status["failed"] = 0
        else:
            # quiet output: the exit code decides, the count comes from a collection pass
            if status["collected"] is None:
                status["collected"] = _pytest_collected_count(timeout=timeout)
            status["result_source"] = "exit_code"
            if proc.returncode == 0 and status["collected"]:
                status["passed"] = int(status["collected"])
                status["failed"] = 0
        try:  # record it so a later invocation can report the result without re-running
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        except OSError:  # pragma: no cover - defensive
            pass
    else:
        if record_path.exists():
            try:
                status["last_full_run"] = json.loads(record_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):  # pragma: no cover - defensive
                pass
    return status


# --------------------------------------------------------------------------
# Parent summary
# --------------------------------------------------------------------------
ACTIVE_DOCS: tuple[str, ...] = ("README.md", "VECTOR_V2_AUDIT.md", "V2_STATUS.md")

CORE_MODULES: tuple[str, ...] = (
    "src/v2_config.py (single configuration system)",
    "src/neuron_record.py (NeuronRecordBank)",
    "src/structured_vector.py (StructuredVectorEncoder)",
    "src/functional_response.py (label-free functional-response source)",
    "src/residual.py (learned residual)",
    "src/neuron_vector.py (composition + frozen artifact)",
    "src/v2_pipeline.py (stable API + presets + run identity)",
    "src/v2_panel.py (notebook-facing control-surface state layer)",
    "src/vector_capacity.py / rate_robustness.py / source_extension.py (evaluation layers)",
    "notebooks/V2_Control_Panel.ipynb + scripts/v2_control_panel.py (the two interfaces)",
)

SCIENTIFIC_STATE = (
    "structural representations (48-D, label-free) have a modest correspondence to "
    "individual-stimulus PROBE responses; much of their raw correspondence tracks response "
    "level/amplitude and is strongly attenuated by neuron z-scoring; the label-free activity "
    "representation carries substantially more correspondence (largely rate-aligned); adding "
    "the label-free functional-response source to the learned residual produces a small but "
    "reproducible increase on the neuron z-scored target (delta ~ +0.03 at residual_d=16, "
    "+0.06 at residual_d=52; 9/9 checkpoint x seed replicates), while the temporal source "
    "alone does not move that target and the combined source behaves like the functional "
    "source alone; the block-mask diagnostic is not evidence for block masking as a better "
    "protocol; checkpoint variability exceeds residual-seed variability; no final 'best' "
    "dimension or normalization has been established."
)


def parent_summary(
    resolved: ResolvedConfig | None = None,
    *,
    test_status: Mapping[str, Any] | None = None,
    include_tests: bool = True,
    run_tests: bool = False,
) -> str:
    """The mandatory machine-readable state summary (no data, no training, no TEST)."""
    r = resolved or resolve_config()
    tests = dict(test_status) if test_status is not None else (
        test_suite_status(run_tests=run_tests) if include_tests else {}
    )
    collected = tests.get("collected")
    if tests.get("passed") is not None:
        source = "summary line" if tests.get("result_source") == "summary_line" else "exit code"
        tests_line = (
            f"{collected if collected is not None else '?'} collected; "
            f"{tests['passed']} passed, {tests.get('failed', 0)} failed "
            f"(full suite, exit {tests.get('exit_code')}, from {source})"
        )
    elif tests.get("exit_code") is not None:
        tests_line = (
            f"{collected if collected is not None else '?'} collected; full suite FAILED "
            f"(exit {tests.get('exit_code')}); failure details were not parsed - run "
            "'uv run pytest -q' for the reason"
        )
    elif isinstance(tests.get("last_full_run"), Mapping) and tests["last_full_run"].get("passed"):
        last = tests["last_full_run"]
        tests_line = (
            f"{collected if collected is not None else '?'} collected; last recorded full run: "
            f"{last['passed']} passed, {last.get('failed', 0)} failed "
            f"({last.get('recorded_utc')}; re-run with --run-tests to refresh)"
        )
    elif collected is not None:
        tests_line = (
            f"{collected} collected; full-suite result not run in this invocation "
            "(use --run-tests)"
        )
    else:
        tests_line = "unknown (pytest could not be invoked in this environment)"

    presets = "\n".join(
        f"  {name}: {PRESET_DECOMPOSITIONS[name]}" for name in available_presets()
    )
    blocks = ", ".join(DEFAULT_ENABLED_BLOCKS)
    lines = [
        "=== V2 PARENT SUMMARY ===",
        "",
        "STATUS:",
        "Representation architecture frozen; one backend (src/v2_pipeline.py) with two interfaces - "
        "the Jupyter control panel (notebooks/V2_Control_Panel.ipynb, primary interactive) and the "
        "CLI (scripts/v2_control_panel.py, headless/reproducible); no new representation source, "
        "no network_context, no multi-layer support, TEST never used.",
        "",
        "TESTS:",
        tests_line,
        "",
        "ACTIVE DOCS:",
        "  " + ", ".join(ACTIVE_DOCS),
        "",
        "CORE MODULES:",
        *(f"  {m}" for m in CORE_MODULES),
        "",
        "CURRENT ARCHITECTURE:",
        "  frozen SNN checkpoint + label-free FIT -> NeuronRecordBank -> "
        "{structural source, activity block, functional-response source, temporal block} -> "
        "StructuredVectorEncoder (+ optional learned residual) -> NeuronVector (n_neurons, d)",
        "",
        "DIMENSION SEMANTICS:",
        "  d = structured_d + residual_d   (final vector z = [z_structured, z_residual])",
        f"  48 = 48 structured + 0 residual | 64 = 48 structured + 16 residual | "
        f"100 = 48 structured + 52 residual",
        "  (level-0/level-1 source dimensions and the functional/temporal source dimensions are "
        "source-space sizes, not final vector dimensions)",
        "",
        "DEFAULT:",
        f"  d = {r.d} = {r.structured_d} structured + {r.residual_d} residual; "
        f"enabled_blocks = {blocks}; residual disabled; no functional-response/temporal source",
        "",
        "AVAILABLE PRESETS:",
        presets,
        "",
        "SOURCES:",
        "  structural (intrinsic, input_conn, recurrent_in, recurrent_out): implemented, default",
        "  activity (12 label-free FIT summaries + optional per-sample counts): implemented, opt-in",
        "  functional_response (fixed 64-D projection of the FIT per-stimulus response profile): "
        "implemented, opt-in via vector.residual.source_functional_response (requires the residual)",
        "  temporal (coarse label-free FIT PSTH bins, vector.temporal_resolution): implemented, opt-in",
        "  network_context: NOT IMPLEMENTED (declared only; selecting it is rejected)",
        "",
        "RESIDUAL:",
        "  implemented (src/residual.py), FIT-only, self-supervised masked reconstruction, "
        "SNN frozen; default protocol: 200 epochs, batch 64, Adam lr 1e-3, hidden 64, "
        "mask_fraction 0.25 (coordinate masking, mask_seed 0), val_fraction 0.2 (split_seed 0), "
        "train_standardise; persisted as learned_residual/v1 with schema/feature-hash/dimension "
        "checks, and the control panel additionally verifies the training protocol before reuse",
        "",
        "CONTROL PANEL:",
        "  interactive (primary): notebooks/V2_Control_Panel.ipynb (state layer src/v2_panel.py)",
        "    edit the controls in place -> validate -> build (FIT) -> inspect -> evaluate (PROBE, optional)",
        "  headless (reproducible): uv run python scripts/v2_control_panel.py ...",
        "  uv run python scripts/v2_control_panel.py --list-presets",
        "  uv run python scripts/v2_control_panel.py --preset historical_48 --dry-run",
        "  uv run python scripts/v2_control_panel.py --preset functional_64            # build (FIT only)",
        "  uv run python scripts/v2_control_panel.py --preset functional_64 --evaluate  # PROBE only",
        "  overrides: --override vector.structured_d=64 --override vector.learned_residual_d=0",
        "",
        "CACHE:",
        f"  run_id = sha256[:16] of the resolved representation values "
        f"({{component schemas, vector config incl. the residual protocol, vector/activity dtype, "
        f"checkpoint id}}; the preset label and the wall clock are excluded) "
        f"(current default run_id = {r.run_id})",
        "  artifact types: fit/probe activity cache (npz+json, metadata-verified), residual "
        "artifact (learned_residual/v1, schema+feature-hash+dimension+protocol verified), "
        "frozen neuron-vector artifact (neuron_vector_artifact/v1, run-id+dimension+name+"
        "content-hash verified), evaluation artifact (json, keyed by run id + settings)",
        "",
        "SAFETY:",
        "  FIT -> representation construction + residual training; PROBE -> targets/metrics only; "
        "TEST -> never accessed (the pipeline and the control panel have no code path that opens "
        "the official test recordings)",
        "",
        "NETWORK_CONTEXT:",
        "not implemented",
        "",
        "MULTI_LAYER:",
        "not implemented",
        "",
        "SCIENTIFIC STATE:",
        f"  {SCIENTIFIC_STATE}",
        "",
        "NEXT STEP:",
        "  Use the control panel to build/evaluate presets and to state configurations for the "
        "next scientific study (e.g. a dimension or normalization comparison) - the architecture "
        "itself is frozen; network_context and multi-layer support remain unimplemented.",
        "=== END V2 PARENT SUMMARY ===",
    ]
    return "\n".join(lines)


__all__ = [
    "SCHEMA",
    "RUN_ID_SCHEMA",
    "DATA_POLICY",
    "PROJECT_ROOT",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_OUT_DIR",
    "DEFAULT_CHECKPOINT",
    "SUPPORTED_EVALUATION_MODES",
    "EVALUATION_TARGETS",
    "COMPONENT_SCHEMAS",
    "PRESETS",
    "PRESET_DECOMPOSITIONS",
    "PRESET_DESCRIPTIONS",
    "ACTIVE_DOCS",
    "CORE_MODULES",
    "SCIENTIFIC_STATE",
    "PipelineError",
    "ResolvedConfig",
    "BuildResult",
    "EvaluationResult",
    "V2Run",
    "available_presets",
    "preset_summary_rows",
    "apply_preset",
    "apply_overrides",
    "representation_identity",
    "run_identity",
    "compute_run_id",
    "resolve_config",
    "dry_run_report",
    "build_fit_probe_recordings",
    "residual_cache_path",
    "build_representation",
    "load_vector_artifact",
    "evaluate_representation",
    "test_suite_status",
    "parent_summary",
]
