"""Label-free, self-supervised learned residual over a :class:`NeuronRecordBank`.

Position in the V2 architecture
-------------------------------
::

    frozen SNN checkpoint + label-free FIT
                 |
                 v
         NeuronRecordBank                        (source information, label-free)
                 |
        +--------+---------+
        |                  |
        v                  v
    StructuredVectorEncoder  learned residual    (this module: trained, FIT-only)
        |                  |
        v                  v
     z_structured        z_residual
        |                  |
        +--------+---------+
                 v
        NeuronVector composition                 (src/neuron_vector.py)

The residual is the only *learned* component, but it never touches the SNN: it is trained on
feature vectors exported from a frozen record bank. It never backpropagates into the
network, never re-reads the checkpoint, and never modifies model parameters.

Scientific boundary (enforced, not merely documented)
-----------------------------------------------------
* **Labels are structurally impossible to pass.** Every public entry point
  (:func:`build_residual_source`, :class:`ResidualTrainer`, :func:`train_residual`,
  :class:`ResidualResult`) takes only a record bank / source matrix plus configuration -
  there is no labels, PROBE, TEST, fingerprint or decoding-target parameter.
* **FIT only.** The source view is built exclusively from the bank, and the bank is built
  from label-free FIT statistics; the source builder rejects a bank whose activity
  provenance names a ``probe``/``test`` split. The residual's internal train/validation
  split is a split over *neurons* (examples), so no new data is touched.
* **Self-supervised objective.** Masked reconstruction: a random deterministic subset of
  source coordinates is withheld (zeroed in standardised space) and the loss is computed
  **on the withheld coordinates only**. There is no identity shortcut and no label-derived
  target.
* **Normalisation.** Optional standardisation statistics are computed from the residual
  *training* neuron subset only (FIT-derived), never from PROBE/TEST/labels; the chosen mode
  and its provenance are recorded.
* **No forbidden tensors.** Everything is ``(n_neurons, F)`` features and
  ``(batch, F)`` / ``(batch, d_residual)`` minibatches. No ``(neurons, samples, time)``,
  ``(neurons, samples, d)`` or ``(neurons, neurons, d)`` tensor is ever created.
* **Source view.** The deterministic view has five optional parts: level-0 summaries,
  level-1 detail, fixed raw-connectivity projections, the label-free FIT
  individual-stimulus response projection (:mod:`src.functional_response`) and the coarse
  label-free FIT temporal block. The last two are **opt-in**; when enabled they still read
  only the bank's stored FIT structures (``activity.samples`` and the pooled FIT PSTH) -
  never labels, PROBE, TEST or the SNN. Masking is coordinate-level by default and can be
  switched to source-group level (``structural`` / ``functional_response`` / ``temporal``).

Reproducibility
---------------
All stochastic sources are seeded from the configuration: the global Python/NumPy/Torch
seeds (:func:`src.utils.set_seed`), the neuron train/validation split seed, the per-epoch
mask seed, and the per-epoch shuffle seed. On a fixed device and dtype, the same bank +
configuration + seed produce the same vectors.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from .neuron_record import (
    TEMPORAL_BLOCK,
    NeuronRecordBank,
    assert_label_free_fit_activity,
)
from .functional_response import (
    ACTIVITY_BLOCK,
    DEFAULT_CHUNK_SIZE as DEFAULT_FUNCTIONAL_CHUNK_SIZE,
    FunctionalResponseConfig,
    FunctionalResponseError,
    build_functional_response_source,
)
from .structured_vector import (
    DEFAULT_PROJECTION_SEED,
    StructuredVectorEncoder,
    StructuredVectorError,
)
from .utils import get_device, set_seed
from .v2_config import RESIDUAL_STANDARDIZATIONS, SOURCE_MASK_MODES, V2Config

#: Provenance schema of the residual source view.
SOURCE_SCHEMA = "residual_source/v1"

#: Provenance schema of a persisted residual artifact.
RESIDUAL_SCHEMA = "learned_residual/v1"

#: Offset added to ``mask_seed`` for the fixed evaluation masks.
EVAL_MASK_OFFSET = 10_000_000

#: Default width of the fixed raw-connectivity view per weighted block.
DEFAULT_RAW_VIEW_DIM = 32

#: Default dimension of the label-free functional-response projection in the source view.
DEFAULT_FUNCTIONAL_RESPONSE_VIEW_DIM = 64

#: Source groups used by the residual's per-source diagnostics and by block masking.
SOURCE_GROUP_STRUCTURAL = "structural"
SOURCE_GROUP_FUNCTIONAL = "functional_response"
SOURCE_GROUP_TEMPORAL = "temporal"
SOURCE_GROUPS: tuple[str, ...] = (
    SOURCE_GROUP_STRUCTURAL,
    SOURCE_GROUP_FUNCTIONAL,
    SOURCE_GROUP_TEMPORAL,
)

#: Feature-name prefixes that identify the functional-response and temporal groups.
FUNCTIONAL_FEATURE_PREFIX = "functional_response."
TEMPORAL_FEATURE_PREFIX = f"{TEMPORAL_BLOCK}."

_SUPPORTED_DTYPES = ("float32", "float16", "bfloat16")
_SUPPORTED_DEVICES = ("cpu", "cuda", "auto")
#: Kept as an alias of the canonical list defined by the configuration layer.
_SUPPORTED_NORMALIZATION = RESIDUAL_STANDARDIZATIONS
_SUPPORTED_ACTIVATIONS = ("gelu", "relu", "tanh")


class ResidualError(ValueError):
    """Raised for an invalid residual source/configuration or an incompatible artifact."""


# --------------------------------------------------------------------------
# Source view
# --------------------------------------------------------------------------
@dataclass
class ResidualSourceConfig:
    """Deterministic definition of the residual's input view of a record bank.

    The five parts of the view, in order:

    1. ``include_level0`` - the bank's deterministic summary features,
    2. ``include_level1`` - the encoder's deterministic within-block detail,
    3. ``include_raw_weight_view`` - a fixed seeded projection of the raw connectivity,
    4. ``include_functional_response`` - a fixed seeded projection of the label-free FIT
       individual-stimulus response profile (:mod:`src.functional_response`),
    5. ``include_temporal`` - the coarse label-free FIT temporal block.

    Parts 4 and 5 are **opt-in** (default ``False``): with the defaults the resulting
    feature names - and therefore the schema hash - are identical to the previous stage, so
    existing residuals remain loadable and the historical path is unchanged.
    """

    enabled_blocks: Sequence[str] | str | None = None
    include_level0: bool = True
    include_level1: bool = True
    include_raw_weight_view: bool = True
    raw_view_dim: int = DEFAULT_RAW_VIEW_DIM
    projection_seed: int = DEFAULT_PROJECTION_SEED
    # -- new label-free sources (opt-in) -------------------------------------
    include_functional_response: bool = False
    functional_source_dim: int = DEFAULT_FUNCTIONAL_RESPONSE_VIEW_DIM
    functional_projection_seed: int = 0
    functional_normalization: str = "raw"
    functional_chunk_size: int = DEFAULT_FUNCTIONAL_CHUNK_SIZE
    include_temporal: bool = False

    def __post_init__(self) -> None:
        if not (
            self.include_level0
            or self.include_level1
            or self.include_raw_weight_view
            or self.include_functional_response
            or self.include_temporal
        ):
            raise ResidualError(
                "the residual source needs at least one of include_level0, include_level1, "
                "include_raw_weight_view, include_functional_response or include_temporal"
            )
        self.raw_view_dim = _positive_int(self.raw_view_dim, "raw_view_dim")
        self.projection_seed = _non_negative_int(self.projection_seed, "projection_seed")
        self.include_functional_response = bool(self.include_functional_response)
        self.include_temporal = bool(self.include_temporal)
        # reuses the functional-response configuration's validation (dimension, seed,
        # normalization mode, chunk size) rather than duplicating its rules; its errors are
        # surfaced as ResidualError so this module keeps a single error type
        try:
            functional = FunctionalResponseConfig(
                source_dim=self.functional_source_dim,
                projection_seed=self.functional_projection_seed,
                normalization=self.functional_normalization,
                chunk_size=self.functional_chunk_size,
            )
        except FunctionalResponseError as exc:
            raise ResidualError(str(exc)) from exc
        self.functional_source_dim = int(functional.source_dim)
        self.functional_projection_seed = int(functional.projection_seed)
        self.functional_normalization = functional.normalization
        self.functional_chunk_size = int(functional.chunk_size)

    def to_dict(self) -> dict[str, Any]:
        blocks = self.enabled_blocks
        if blocks is not None and not isinstance(blocks, str):
            blocks = [str(b) for b in blocks]
        return {
            "enabled_blocks": blocks,
            "include_level0": bool(self.include_level0),
            "include_level1": bool(self.include_level1),
            "include_raw_weight_view": bool(self.include_raw_weight_view),
            "raw_view_dim": int(self.raw_view_dim),
            "projection_seed": int(self.projection_seed),
            "include_functional_response": bool(self.include_functional_response),
            "functional_source_dim": int(self.functional_source_dim),
            "functional_projection_seed": int(self.functional_projection_seed),
            "functional_normalization": self.functional_normalization,
            "functional_chunk_size": int(self.functional_chunk_size),
            "include_temporal": bool(self.include_temporal),
        }

    def to_functional_config(self) -> FunctionalResponseConfig:
        """The functional-response configuration implied by this source config."""
        return FunctionalResponseConfig(
            source_dim=self.functional_source_dim,
            projection_seed=self.functional_projection_seed,
            normalization=self.functional_normalization,
            chunk_size=self.functional_chunk_size,
        )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "ResidualSourceConfig":
        valid = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in dict(mapping or {}).items() if k in valid})

    @classmethod
    def from_v2_config(
        cls, config: V2Config, **overrides: Any
    ) -> "ResidualSourceConfig":
        """Build from the V2 configuration (``vector.*`` and ``vector.residual.*``).

        With ``vector.residual.source_functional_response`` / ``source_temporal`` left at
        their default ``false``, this reproduces exactly the previous stage's source view
        (same blocks, same raw-view width and seed).
        """
        mapping: dict[str, Any] = {
            "enabled_blocks": list(config.vector.enabled_blocks),
            "include_functional_response": bool(config.vector.residual.source_functional_response),
            "functional_source_dim": int(config.vector.functional_source_dim),
            "functional_projection_seed": int(config.vector.functional_projection_seed),
            "functional_normalization": config.vector.functional_source_normalization,
            "functional_chunk_size": int(config.memory.representation_chunk_size),
            "include_temporal": bool(config.vector.residual.source_temporal),
        }
        mapping.update(overrides)
        return cls.from_mapping(mapping)


@dataclass
class ResidualSource:
    """A deterministic, 2-D, CPU-side feature view of a record bank."""

    X: np.ndarray
    feature_names: tuple[str, ...]
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float64)
        if self.X.ndim != 2:
            raise ResidualError(
                f"residual source must be a 2-D (n_neurons, n_features) matrix, got {self.X.shape}"
            )
        self.feature_names = tuple(str(n) for n in self.feature_names)
        if self.X.shape[1] != len(self.feature_names):
            raise ResidualError(
                f"source matrix has {self.X.shape[1]} columns but {len(self.feature_names)} feature names"
            )
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ResidualError("residual source feature names must be unique")
        if not np.isfinite(self.X).all():
            bad = int((~np.isfinite(self.X)).sum())
            raise ResidualError(f"residual source contains {bad} non-finite entries")
        self.provenance = dict(self.provenance)
        self.provenance.setdefault("schema", SOURCE_SCHEMA)
        self.provenance.setdefault("uses_labels", False)
        self.provenance.setdefault("n_neurons", int(self.X.shape[0]))
        self.provenance.setdefault("n_features", int(self.X.shape[1]))

    @property
    def n_neurons(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def schema_hash(self) -> str:
        return feature_schema_hash(self.feature_names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_neurons": self.n_neurons,
            "n_features": self.n_features,
            "feature_names": list(self.feature_names),
            "schema_hash": self.schema_hash,
            "provenance": self.provenance,
        }


def feature_schema_hash(feature_names: Sequence[str], *, schema: str = SOURCE_SCHEMA) -> str:
    """Stable hash of a feature *schema* (names and order), used to reject mismatches."""
    payload = json.dumps({"schema": schema, "names": [str(n) for n in feature_names]}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _raw_weight_view(weights: np.ndarray, width: int, *, seed: int, block_index: int) -> np.ndarray:
    """Fixed, data-independent projection of a raw connectivity matrix (``(n, k) -> (n, width)``)."""
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 2:
        raise ResidualError(f"raw weight view expects a 2-D matrix, got {w.shape}")
    out = min(int(width), int(w.shape[1]))
    g = np.empty((w.shape[1], out), dtype=np.float64)
    scale = 1.0 / np.sqrt(float(w.shape[1]))
    for column in range(out):
        rng = np.random.default_rng([int(seed), int(block_index), int(column)])
        g[:, column] = rng.standard_normal(w.shape[1]) * scale
    return w @ g


def source_group_of_feature(name: str) -> str:
    """Group of one source coordinate (used for diagnostics and block masking)."""
    text = str(name)
    if text.startswith(FUNCTIONAL_FEATURE_PREFIX):
        return SOURCE_GROUP_FUNCTIONAL
    if text.startswith(TEMPORAL_FEATURE_PREFIX):
        return SOURCE_GROUP_TEMPORAL
    return SOURCE_GROUP_STRUCTURAL


def source_groups_from_names(feature_names: Sequence[str]) -> dict[str, list[int]]:
    """``{group: [coordinate indices]}`` for a source schema (empty groups omitted)."""
    groups: dict[str, list[int]] = {group: [] for group in SOURCE_GROUPS}
    for index, name in enumerate(feature_names):
        groups[source_group_of_feature(name)].append(int(index))
    return {group: indices for group, indices in groups.items() if indices}


def _append_unique(
    parts: list[np.ndarray],
    names: list[str],
    X_new: np.ndarray,
    names_new: Sequence[str],
) -> dict[str, Any]:
    """Append only coordinates that are not already present; report the de-duplication.

    A block that the structured encoder already selects (e.g. ``temporal`` in
    ``vector.enabled_blocks``) must not appear twice in the source view: adding the same
    coordinate twice would silently double-weight it and break name uniqueness.
    """
    existing = set(names)
    keep: list[int] = []
    duplicated: list[str] = []
    for index, name in enumerate(names_new):
        if name in existing:
            duplicated.append(str(name))
        else:
            existing.add(name)
            keep.append(int(index))
    if keep:
        parts.append(np.asarray(X_new, dtype=np.float64)[:, keep])
        names.extend(str(names_new[i]) for i in keep)
    return {
        "appended_coordinates": len(keep),
        "duplicate_coordinates_skipped": len(duplicated),
        "duplicate_examples": duplicated[:5],
        "reason": (
            "coordinates already present in an earlier source part (e.g. the structured "
            "level-0 prefix already selected this block); never counted twice"
            if duplicated else ""
        ),
    }


def build_residual_source(
    bank: NeuronRecordBank,
    config: ResidualSourceConfig | None = None,
) -> ResidualSource:
    """Build the deterministic source view used to train and evaluate the residual.

    The view contains, in this order:

    1. the bank's **level-0** deterministic summary features (when ``include_level0``),
    2. the encoder's **level-1** deterministic within-block detail (when ``include_level1``),
    3. a **fixed seeded projection of the raw connectivity** of each weighted block
       (when ``include_raw_weight_view``) - source information the structured summaries do
       not expose,
    4. a **fixed seeded projection of the label-free FIT individual-stimulus response
       profile** (when ``include_functional_response``),
    5. the **coarse label-free FIT temporal block** (when ``include_temporal``).

    Only the record bank is read: no labels, no PROBE/TEST data, no model forward pass.
    Every part is identifiable in the provenance, and the coordinate groups
    (:func:`source_group_of_feature`) drive the per-source training diagnostics and the
    optional block masking.
    """
    if not isinstance(bank, NeuronRecordBank):
        raise ResidualError(f"residual source requires a NeuronRecordBank, got {type(bank).__name__}")
    if bank.uses_labels is not False:  # pragma: no cover - the property is always False
        raise ResidualError("the record bank must be label-free (uses_labels=False)")
    config = config or ResidualSourceConfig()
    _assert_bank_activity_is_label_free_fit(bank)

    try:
        probe = StructuredVectorEncoder(bank, structured_d=1, enabled_blocks=config.enabled_blocks)
        blocks = list(probe.plan.present_blocks)
        level0_dim = probe.level0_dimension
        source_dim = probe.source_dimension
    except StructuredVectorError as exc:  # pragma: no cover - defensive translation
        raise ResidualError(str(exc)) from exc

    parts: list[np.ndarray] = []
    names: list[str] = []
    if config.include_level0 and config.include_level1:
        full = StructuredVectorEncoder(
            bank, structured_d=source_dim, enabled_blocks=config.enabled_blocks
        ).encode()
        parts.append(full.X)
        names.extend(full.feature_names)
    elif config.include_level0:
        level0 = StructuredVectorEncoder(
            bank, structured_d=level0_dim, enabled_blocks=config.enabled_blocks
        ).encode()
        parts.append(level0.X)
        names.extend(level0.feature_names)
    elif config.include_level1:
        full = StructuredVectorEncoder(
            bank, structured_d=source_dim, enabled_blocks=config.enabled_blocks
        ).encode()
        parts.append(full.X[:, level0_dim:])
        names.extend(full.feature_names[level0_dim:])

    raw_view_dims: dict[str, int] = {}
    if config.include_raw_weight_view:
        for index, block_name in enumerate(blocks):
            block = bank.get_block(block_name)
            if block.weights is None:
                continue
            view = _raw_weight_view(
                block.weights, config.raw_view_dim, seed=config.projection_seed, block_index=index
            )
            raw_view_dims[block_name] = int(view.shape[1])
            parts.append(view)
            names.extend(f"{block_name}.raw_view_{j:02d}" for j in range(view.shape[1]))

    functional_info: dict[str, Any] = {
        "enabled": False,
        "reason": "include_functional_response=False",
    }
    if config.include_functional_response:
        try:
            functional = build_functional_response_source(bank, config.to_functional_config())
        except FunctionalResponseError as exc:
            raise ResidualError(str(exc)) from exc
        dedup = _append_unique(parts, names, functional.X, functional.feature_names)
        functional_info = {"enabled": True, **functional.provenance, **dedup}

    temporal_info: dict[str, Any] = {
        "enabled": False,
        "reason": "include_temporal=False",
    }
    if config.include_temporal:
        try:
            temporal_block = bank.get_block(TEMPORAL_BLOCK)
        except Exception as exc:
            raise ResidualError(
                "include_temporal=True requires a bank built with the coarse temporal block "
                "(label-free FIT activity plus vector.enabled_blocks containing 'temporal' or "
                "vector.residual.source_temporal=true, and vector.temporal_resolution set); "
                f"this bank has none ({exc})"
            ) from exc
        temporal_X, temporal_names = temporal_block.to_matrix()
        dedup = _append_unique(parts, names, temporal_X, temporal_names)
        temporal_prov = bank.provenance.get("temporal", {})
        temporal_info = {
            "enabled": True,
            "source_type": "coarse_label_free_fit_temporal",
            "source_split": temporal_prov.get("split"),
            "uses_labels": False,
            "input_shape": [int(temporal_X.shape[0]), int(temporal_X.shape[1])],
            "available_dimension": int(temporal_X.shape[1]),
            "output_dimension": int(dedup["appended_coordinates"]),
            "feature_names": list(temporal_names),
            "normalization": "mean label-free FIT firing rate per coarse bin (Hz), no further scaling",
            "temporal_resolution": temporal_prov.get("resolution"),
            "binning": temporal_block.metadata.get("binning"),
            "bins": list(temporal_prov.get("bins", [])),
            "class_conditioned": False,
            "time_resolved_tensor_stored": False,
            **dedup,
        }

    if not parts:  # pragma: no cover - guarded by ResidualSourceConfig validation
        raise ResidualError("the residual source view is empty")

    X = np.concatenate(parts, axis=1)
    groups = source_groups_from_names(names)
    source_blocks = [
        {
            "name": "level0",
            "kind": "deterministic_summaries",
            "dimension": int(level0_dim if config.include_level0 else 0),
        },
        {
            "name": "level1",
            "kind": "deterministic_detail",
            "dimension": int((source_dim - level0_dim) if config.include_level1 else 0),
        },
        {
            "name": "raw_connectivity_view",
            "kind": "fixed_projection_of_raw_weights",
            "dimension": int(sum(raw_view_dims.values())),
            "per_block": dict(raw_view_dims),
        },
        {
            "name": "functional_response",
            "kind": "label_free_fit_individual_stimulus_response",
            "dimension": int(functional_info.get("output_dimension", 0)),
            "available_dimension": int(functional_info.get("output_dimension", 0) or 0),
            "enabled": bool(functional_info.get("enabled")),
        },
        {
            "name": "temporal",
            "kind": "label_free_fit_coarse_temporal",
            "dimension": int(temporal_info.get("output_dimension", 0)),
            "available_dimension": int(temporal_info.get("available_dimension", 0) or 0),
            "enabled": bool(temporal_info.get("enabled")),
            "note": (
                "the residual source always carries the full union of the five parts; a part "
                "whose coordinates already entered an earlier part contributes nothing new "
                "here (see duplicate_coordinates_skipped), and source_groups reports the "
                "final per-group coordinate counts"
            ),
        },
    ]
    provenance = {
        "schema": SOURCE_SCHEMA,
        "uses_labels": False,
        "fit_only": True,
        "block_names": blocks,
        "level0_dimension": int(level0_dim if config.include_level0 else 0),
        "level1_dimension": int((source_dim - level0_dim) if config.include_level1 else 0),
        "raw_view_dimensions": raw_view_dims,
        "raw_view_projection_seed": int(config.projection_seed) if raw_view_dims else None,
        "raw_view_type": "fixed_gaussian_1_over_sqrt_width" if raw_view_dims else None,
        "functional_response": functional_info,
        "temporal": temporal_info,
        "source_blocks": source_blocks,
        "source_groups": groups,
        "source_config": config.to_dict(),
        "bank": {
            "schema": bank.provenance.get("schema"),
            "n_neurons": bank.n_neurons,
            "uses_labels": False,
            "activity_split": bank.provenance.get("activity", {}).get("split"),
        },
        "note": (
            "Deterministic view of the label-free record bank: level-0 summaries, level-1 "
            "deterministic detail, a fixed seeded projection of the raw connectivity and "
            "(when enabled) the label-free FIT individual-stimulus response projection and "
            "coarse temporal block. No labels, no PROBE, no TEST, no functional fingerprint "
            "and no model forward pass."
        ),
    }
    return ResidualSource(X=X, feature_names=tuple(names), provenance=provenance)


def _assert_bank_activity_is_label_free_fit(bank: NeuronRecordBank) -> None:
    """Reject a bank whose activity provenance names a PROBE/TEST split."""
    try:
        assert_label_free_fit_activity(bank)
    except Exception as exc:
        raise ResidualError(str(exc)) from exc


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class ResidualTrainingConfig:
    """Training configuration of the learned residual (every value is explicit and logged)."""

    residual_dim: int = 52
    hidden_dim: int = 64
    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float | None = 1.0
    dropout: float = 0.0
    activation: str = "gelu"
    mask_fraction: float = 0.25
    mask_seed: int = 0
    minimum_visible_features: int = 8
    visible_loss_weight: float = 0.0
    mask_mode: str = "coordinate"
    normalization: str = "train_standardise"
    val_fraction: float = 0.2
    split_seed: int = 0
    seed: int = 0
    restore_best: bool = True
    device: str = "cpu"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        self.residual_dim = _positive_int(self.residual_dim, "residual_dim")
        self.hidden_dim = _positive_int(self.hidden_dim, "hidden_dim")
        self.epochs = _positive_int(self.epochs, "epochs")
        self.batch_size = _positive_int(self.batch_size, "batch_size")
        self.lr = _positive_float(self.lr, "lr")
        self.weight_decay = _non_negative_float(self.weight_decay, "weight_decay")
        if self.grad_clip is not None:
            self.grad_clip = _positive_float(self.grad_clip, "grad_clip")
        self.dropout = _non_negative_float(self.dropout, "dropout")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ResidualError(f"dropout must be in [0, 1), got {self.dropout}")
        self.dropout = float(self.dropout)
        self.activation = str(self.activation).lower()
        if self.activation not in _SUPPORTED_ACTIVATIONS:
            raise ResidualError(
                f"activation must be one of {list(_SUPPORTED_ACTIVATIONS)}, got {self.activation!r}"
            )
        self.mask_fraction = _fraction(self.mask_fraction, "mask_fraction")
        self.mask_seed = _non_negative_int(self.mask_seed, "mask_seed")
        self.minimum_visible_features = _positive_int(
            self.minimum_visible_features, "minimum_visible_features"
        )
        self.visible_loss_weight = _non_negative_float(self.visible_loss_weight, "visible_loss_weight")
        self.mask_mode = str(self.mask_mode).strip().lower()
        if self.mask_mode not in SOURCE_MASK_MODES:
            raise ResidualError(
                f"mask_mode must be one of {list(SOURCE_MASK_MODES)}, got {self.mask_mode!r}"
            )
        self.normalization = str(self.normalization).lower()
        if self.normalization not in _SUPPORTED_NORMALIZATION:
            raise ResidualError(
                f"normalization must be one of {list(_SUPPORTED_NORMALIZATION)}, got {self.normalization!r}"
            )
        self.val_fraction = float(self.val_fraction)
        if not 0.0 <= self.val_fraction < 0.5:
            raise ResidualError(f"val_fraction must be in [0, 0.5), got {self.val_fraction}")
        self.split_seed = _non_negative_int(self.split_seed, "split_seed")
        self.seed = _non_negative_int(self.seed, "seed")
        self.restore_best = bool(self.restore_best)
        self.device = str(self.device).lower()
        if self.device not in _SUPPORTED_DEVICES:
            raise ResidualError(f"device must be one of {list(_SUPPORTED_DEVICES)}, got {self.device!r}")
        self.dtype = str(self.dtype).lower()
        if self.dtype not in _SUPPORTED_DTYPES:
            raise ResidualError(f"dtype must be one of {list(_SUPPORTED_DTYPES)}, got {self.dtype!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "ResidualTrainingConfig":
        valid = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        int_fields = {"residual_dim", "hidden_dim", "epochs", "batch_size", "mask_seed",
                      "minimum_visible_features", "split_seed", "seed"}
        kwargs: dict[str, Any] = {}
        for key, value in dict(mapping or {}).items():
            if key not in valid:
                continue
            if key in int_fields and isinstance(value, str):
                value = int(float(value))
            kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def from_v2_config(
        cls,
        config: V2Config,
        *,
        residual_dim: int | None = None,
        **overrides: Any,
    ) -> "ResidualTrainingConfig":
        """Build from the V2 configuration.

        The dimension comes from ``vector.learned_residual_d``, the dtype from
        ``precision.vector_dtype`` and the whole training protocol from
        ``vector.residual`` (mask mode, seed, split/mask seeds, hidden width, epochs,
        batch size, learning rate, mask fraction, minimum visible features,
        validation fraction and standardization). Explicit ``overrides`` still win, so
        the stage scripts keep the protocol they were written with.
        """
        dimension = int(config.vector.learned_residual_d) if residual_dim is None else int(residual_dim)
        if dimension <= 0:
            raise ResidualError(
                "residual_dim must be > 0 to train a residual; the configuration requests "
                f"learned_residual_d={config.vector.learned_residual_d} "
                f"(residual.enabled={config.vector.residual.enabled}). Set "
                "vector.learned_residual_d > 0 and vector.residual.enabled=true, or skip the residual."
            )
        r = config.vector.residual
        mapping: dict[str, Any] = {
            "residual_dim": dimension,
            "dtype": config.precision.vector_dtype,
            "mask_mode": r.mask_mode,
            "seed": int(r.seed),
            "split_seed": int(r.split_seed),
            "mask_seed": int(r.mask_seed),
            "hidden_dim": int(r.hidden_dim),
            "epochs": int(r.epochs),
            "batch_size": int(r.batch_size),
            "lr": float(r.learning_rate),
            "mask_fraction": float(r.mask_fraction),
            "minimum_visible_features": int(r.minimum_visible_features),
            "val_fraction": float(r.val_fraction),
            "normalization": r.standardization,
        }
        mapping.update(overrides)
        return cls.from_mapping(mapping)


#: Training-protocol fields a cached artifact must match before it may be reused.
PROTOCOL_FIELDS: tuple[str, ...] = (
    "residual_dim",
    "hidden_dim",
    "epochs",
    "batch_size",
    "lr",
    "mask_fraction",
    "mask_seed",
    "minimum_visible_features",
    "val_fraction",
    "split_seed",
    "seed",
    "normalization",
    "mask_mode",
)


def residual_training_mismatches(
    residual: "ResidualResult",
    expected: ResidualTrainingConfig,
) -> dict[str, tuple[Any, Any]]:
    """Protocol fields where a loaded artifact differs from the requested training config.

    :meth:`ResidualResult.load` verifies the *schema* (feature names, dimensions, hash
    integrity) but deliberately not the training hyper-parameters, so a caller that reuses
    cached artifacts keyed only by source schema must check them explicitly: silently
    reusing an artifact trained under a different protocol would change the numbers without
    any visible failure. Returns ``{field: (artifact_value, expected_value)}``; an empty
    dict means the artifact is protocol-compatible.
    """
    if not isinstance(residual, ResidualResult):
        raise ResidualError(
            f"protocol comparison needs a ResidualResult, got {type(residual).__name__}"
        )
    if not isinstance(expected, ResidualTrainingConfig):
        raise ResidualError(
            f"protocol comparison needs a ResidualTrainingConfig, got {type(expected).__name__}"
        )
    actual = residual.config
    found: dict[str, tuple[Any, Any]] = {}
    for name in PROTOCOL_FIELDS:
        a = getattr(actual, name, None)
        b = getattr(expected, name, None)
        if isinstance(a, float) or isinstance(b, float):
            if a is None or b is None or not np.isclose(float(a), float(b), rtol=0.0, atol=1e-12):
                found[name] = (a, b)
        elif a != b:
            found[name] = (a, b)
    return found


# --------------------------------------------------------------------------
# Network (small MLPs - deliberately modest, documented)
# --------------------------------------------------------------------------
def _activation(name: str) -> nn.Module:
    return {"gelu": nn.GELU(), "relu": nn.ReLU(), "tanh": nn.Tanh()}[name]


def _mlp(dims: Sequence[int], *, activation: str, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index in range(len(dims) - 1):
        layers.append(nn.Linear(int(dims[index]), int(dims[index + 1])))
        if index < len(dims) - 2:
            layers.append(_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class ResidualEncoder(nn.Module):
    """Small encoder MLP: ``input_dim -> hidden_dim -> residual_dim`` (linear output)."""

    def __init__(
        self,
        input_dim: int,
        residual_dim: int,
        hidden_dim: int = 64,
        *,
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = _positive_int(input_dim, "input_dim")
        self.residual_dim = _positive_int(residual_dim, "residual_dim")
        self.hidden_dim = _positive_int(hidden_dim, "hidden_dim")
        self.activation = str(activation).lower()
        if self.activation not in _SUPPORTED_ACTIVATIONS:
            raise ResidualError(f"unsupported activation {activation!r}")
        self.dropout = float(dropout)
        self.net = _mlp((self.input_dim, self.hidden_dim, self.residual_dim),
                        activation=self.activation, dropout=self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def network_config(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "residual_dim": self.residual_dim,
            "hidden_dim": self.hidden_dim,
            "activation": self.activation,
            "dropout": self.dropout,
        }

    @classmethod
    def from_network_config(cls, config: Mapping[str, Any]) -> "ResidualEncoder":
        return cls(
            int(config["input_dim"]),
            int(config["residual_dim"]),
            int(config["hidden_dim"]),
            activation=str(config.get("activation", "gelu")),
            dropout=float(config.get("dropout", 0.0)),
        )


class ReconstructionDecoder(nn.Module):
    """Small decoder MLP: ``residual_dim -> hidden_dim -> input_dim``."""

    def __init__(
        self,
        input_dim: int,
        residual_dim: int,
        hidden_dim: int = 64,
        *,
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = _positive_int(input_dim, "input_dim")
        self.residual_dim = _positive_int(residual_dim, "residual_dim")
        self.hidden_dim = _positive_int(hidden_dim, "hidden_dim")
        self.activation = str(activation).lower()
        self.dropout = float(dropout)
        self.net = _mlp((self.residual_dim, self.hidden_dim, self.input_dim),
                        activation=self.activation, dropout=self.dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def network_config(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "residual_dim": self.residual_dim,
            "hidden_dim": self.hidden_dim,
            "activation": self.activation,
            "dropout": self.dropout,
        }

    @classmethod
    def from_network_config(cls, config: Mapping[str, Any]) -> "ReconstructionDecoder":
        return cls(
            int(config["input_dim"]),
            int(config["residual_dim"]),
            int(config["hidden_dim"]),
            activation=str(config.get("activation", "gelu")),
            dropout=float(config.get("dropout", 0.0)),
        )


# --------------------------------------------------------------------------
# Masking
# --------------------------------------------------------------------------
def make_mask(
    n_rows: int,
    n_features: int,
    *,
    mask_fraction: float,
    minimum_visible_features: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Boolean ``(n_rows, n_features)`` mask; ``True`` marks a withheld coordinate.

    Random but deterministic given ``rng``; never derived from labels. Each row withholds
    ``round(mask_fraction * n_features)`` coordinates (at least one) while keeping at least
    ``minimum_visible_features`` coordinates visible.
    """
    if n_rows < 1 or n_features < 1:
        raise ResidualError(f"mask shape must be positive, got ({n_rows}, {n_features})")
    fraction = _fraction(mask_fraction, "mask_fraction")
    min_visible = _positive_int(minimum_visible_features, "minimum_visible_features")
    if min_visible >= n_features:
        raise ResidualError(f"minimum_visible_features={min_visible} must be < n_features={n_features}")
    n_mask = int(round(fraction * n_features))
    n_mask = max(1, min(n_mask, n_features - min_visible))
    mask = np.zeros((n_rows, n_features), dtype=bool)
    for row in range(n_rows):
        mask[row, rng.choice(n_features, size=n_mask, replace=False)] = True
    return mask


def make_block_mask(
    n_rows: int,
    n_features: int,
    *,
    groups: Mapping[str, Sequence[int]],
    mask_fraction: float,
    minimum_visible_features: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Boolean ``(n_rows, n_features)`` mask that withholds **one whole source group** per row.

    Source-aware masking: for each example one group (``structural`` /
    ``functional_response`` / ``temporal``) is chosen at random and all of its coordinates
    are withheld, so the model must reconstruct a source from the others. Groups are tried
    in random order and the first one that keeps at least ``minimum_visible_features``
    coordinates visible is used; if no group satisfies that (e.g. a single very large
    group), the row falls back to the deterministic coordinate-level mask - so the
    ``minimum_visible_features`` guarantee always holds. Deterministic given ``rng`` and
    never derived from labels, PROBE or TEST.
    """
    if n_rows < 1 or n_features < 1:
        raise ResidualError(f"mask shape must be positive, got ({n_rows}, {n_features})")
    min_visible = _positive_int(minimum_visible_features, "minimum_visible_features")
    if min_visible >= n_features:
        raise ResidualError(f"minimum_visible_features={min_visible} must be < n_features={n_features}")
    names = sorted(str(name) for name in groups)
    indices = {str(name): [int(i) for i in groups[name]] for name in names}
    sizes = {name: len(values) for name, values in indices.items()}
    if not names:
        raise ResidualError("block masking needs at least one source group")

    mask = np.zeros((n_rows, n_features), dtype=bool)
    fallback_rows: list[int] = []
    for row in range(n_rows):
        chosen: str | None = None
        for position in rng.permutation(len(names)):
            candidate = names[int(position)]
            if n_features - sizes[candidate] >= min_visible:
                chosen = candidate
                break
        if chosen is None:
            fallback_rows.append(row)
        else:
            mask[row, indices[chosen]] = True
    if fallback_rows:
        fallback = make_mask(
            len(fallback_rows),
            n_features,
            mask_fraction=mask_fraction,
            minimum_visible_features=min_visible,
            rng=rng,
        )
        mask[np.asarray(fallback_rows, dtype=np.int64)] = fallback
    return mask


def make_source_mask(
    n_rows: int,
    n_features: int,
    *,
    mode: str,
    mask_fraction: float,
    minimum_visible_features: int,
    rng: np.random.Generator,
    groups: Mapping[str, Sequence[int]] | None = None,
) -> np.ndarray:
    """Dispatch to the configured masking strategy (``coordinate`` or ``block``)."""
    strategy = str(mode).strip().lower()
    if strategy == "coordinate":
        return make_mask(
            n_rows,
            n_features,
            mask_fraction=mask_fraction,
            minimum_visible_features=minimum_visible_features,
            rng=rng,
        )
    if strategy == "block":
        if not groups:
            raise ResidualError("mask_mode='block' requires the source coordinate groups")
        return make_block_mask(
            n_rows,
            n_features,
            groups=groups,
            mask_fraction=mask_fraction,
            minimum_visible_features=minimum_visible_features,
            rng=rng,
        )
    raise ResidualError(f"mask_mode must be one of {list(SOURCE_MASK_MODES)}, got {mode!r}")


def group_masked_mse(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    groups: Mapping[str, Sequence[int]],
) -> dict[str, float]:
    """Per-source-group MSE on the **withheld** coordinates (training diagnostics only).

    ``nan`` for a group that had no withheld coordinate in this batch, so the caller can
    distinguish "not masked here" from "perfectly reconstructed". These numbers are
    diagnostics, not scientific results.
    """
    diff_sq = (reconstruction - target) ** 2
    out: dict[str, float] = {}
    for name, columns in groups.items():
        index = torch.as_tensor([int(c) for c in columns], dtype=torch.long, device=mask.device)
        sub_mask = mask.index_select(1, index)
        if not bool(sub_mask.any()):
            out[str(name)] = float("nan")
            continue
        denom = sub_mask.sum().clamp(min=1)
        out[str(name)] = float(
            ((diff_sq.index_select(1, index) * sub_mask).sum() / denom).item()
        )
    return out


def apply_mask(x: torch.Tensor, mask: Any) -> torch.Tensor:
    """Copy of ``x`` with withheld coordinates zeroed (standardised space)."""
    mask_t = mask if torch.is_tensor(mask) else torch.as_tensor(mask, dtype=torch.bool, device=x.device)
    return x.masked_fill(mask_t, 0.0)


def masked_reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mask: Any,
    *,
    visible_weight: float = 0.0,
) -> torch.Tensor:
    """MSE on the **withheld** coordinates (plus an optional weighted visible term).

    ``mask`` is ``True`` where the input coordinate was withheld, so the loss targets the
    true (standardised) values at exactly those coordinates.
    """
    mask_t = mask if torch.is_tensor(mask) else torch.as_tensor(mask, dtype=torch.bool, device=reconstruction.device)
    diff_sq = (reconstruction - target) ** 2
    withheld = mask_t.sum().clamp(min=1)
    loss = (diff_sq * mask_t).sum() / withheld
    if visible_weight > 0:
        visible = (~mask_t).sum().clamp(min=1)
        loss = loss + float(visible_weight) * (diff_sq * (~mask_t)).sum() / visible
    return loss


# --------------------------------------------------------------------------
# Result / artifact
# --------------------------------------------------------------------------
@dataclass
class ResidualResult:
    """Trained residual: encoder + decoder, normalisation, schema, history and provenance.

    This object is both the training output and the deployable artifact
    (:meth:`save` / :meth:`load`); :meth:`encode` produces ``z_residual``.
    """

    encoder: ResidualEncoder
    decoder: ReconstructionDecoder
    config: ResidualTrainingConfig
    feature_names: tuple[str, ...]
    source_schema: dict[str, Any]
    normalization: dict[str, Any]
    history: list[dict[str, Any]] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("nan")
    provenance: dict[str, Any] = field(default_factory=dict)
    source_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.feature_names = tuple(str(n) for n in self.feature_names)
        if not self.feature_names:
            raise ResidualError("a residual must record at least one source feature name")
        self.normalization = dict(self.normalization)
        self.provenance = dict(self.provenance)
        self.provenance.setdefault("schema", RESIDUAL_SCHEMA)
        self.provenance.setdefault("uses_labels", False)
        self.provenance.setdefault("fit_only", True)

    # -- dimensions / metadata ----------------------------------------------
    @property
    def residual_dim(self) -> int:
        return int(self.config.residual_dim)

    @property
    def input_dim(self) -> int:
        return len(self.feature_names)

    @property
    def schema_hash(self) -> str:
        return feature_schema_hash(self.feature_names)

    @property
    def uses_labels(self) -> bool:
        """Always ``False``: the residual is trained on label-free FIT-derived features."""
        return False

    def verify_source(self, source: ResidualSource) -> None:
        """Refuse to encode a source that does not match the trained feature schema."""
        if tuple(source.feature_names) != self.feature_names or source.schema_hash != self.schema_hash:
            raise ResidualError(
                "incompatible residual source: the feature schema does not match the trained model "
                f"({source.n_features} features vs {self.input_dim}; schema hash "
                f"{source.schema_hash[:12]} vs {self.schema_hash[:12]}). Re-train the residual or use the "
                "matching record bank - incompatible models are never adapted silently."
            )

    # -- normalisation / encoding -------------------------------------------
    def _standardise(self, X: np.ndarray) -> np.ndarray:
        if self.normalization.get("mode", "none") == "none":
            return np.asarray(X, dtype=np.float64)
        mean = np.asarray(self.normalization["mean"], dtype=np.float64)
        std = np.asarray(self.normalization["std"], dtype=np.float64)
        return (np.asarray(X, dtype=np.float64) - mean) / std

    def encode(self, bank_or_source: NeuronRecordBank | ResidualSource) -> np.ndarray:
        """Residual vectors ``(n_neurons, residual_dim)`` for a bank or a prebuilt source view."""
        if isinstance(bank_or_source, ResidualSource):
            source = bank_or_source
        elif isinstance(bank_or_source, NeuronRecordBank):
            source = build_residual_source(
                bank_or_source, ResidualSourceConfig.from_mapping(self.source_config)
            )
        else:
            raise ResidualError(
                f"encode expects a NeuronRecordBank or ResidualSource, got {type(bank_or_source).__name__}"
            )
        self.verify_source(source)
        device = _resolve_device(self.config.device)
        dtype, _ = _resolve_dtype(self.config.dtype, device)
        x = torch.as_tensor(self._standardise(source.X), dtype=dtype, device=device)
        self.encoder.eval()
        with torch.no_grad():
            z = self.encoder(x).detach().cpu().numpy().astype(np.float64)
        if z.shape != (source.n_neurons, self.residual_dim):  # pragma: no cover - defensive
            raise ResidualError(f"internal error: encoder returned shape {z.shape}")
        return z

    # -- persistence ---------------------------------------------------------
    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": RESIDUAL_SCHEMA,
            "residual_dim": self.residual_dim,
            "input_dim": self.input_dim,
            "hidden_dim": int(self.config.hidden_dim),
            "feature_names": list(self.feature_names),
            "feature_schema_hash": self.schema_hash,
            "source_schema": dict(self.source_schema),
            "source_config": dict(self.source_config),
            "normalization": self.normalization,
            "config": self.config.to_dict(),
            "network": self.encoder.network_config(),
            "encoder_state_dict": {k: v.detach().cpu() for k, v in self.encoder.state_dict().items()},
            "decoder_state_dict": {k: v.detach().cpu() for k, v in self.decoder.state_dict().items()},
            "history": list(self.history),
            "best_epoch": int(self.best_epoch),
            "best_val_loss": float(self.best_val_loss),
            "provenance": dict(self.provenance),
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_payload(), p)
        return p

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_feature_names: Sequence[str] | None = None,
        expected_residual_dim: int | None = None,
        map_location: Any = "cpu",
    ) -> "ResidualResult":
        """Load a persisted residual, refusing incompatible schemas or dimensions.

        The artifact is a local, trusted training product (like the repository's checkpoints),
        so it is loaded with ``weights_only=False``.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"residual artifact not found: {p}")
        payload = torch.load(p, map_location=map_location, weights_only=False)
        if not isinstance(payload, Mapping):
            raise ResidualError(f"{p} does not contain a residual payload")
        schema = payload.get("schema")
        if schema != RESIDUAL_SCHEMA:
            raise ResidualError(f"unsupported residual schema {schema!r}; expected {RESIDUAL_SCHEMA!r}")
        names = tuple(str(n) for n in payload.get("feature_names", ()))
        if not names:
            raise ResidualError(f"{p} has no feature names; refusing to load")
        if feature_schema_hash(names) != payload.get("feature_schema_hash"):
            raise ResidualError(f"{p} failed its feature-schema integrity check (hash mismatch)")
        if int(payload.get("input_dim", -1)) != len(names):
            raise ResidualError(f"{p} is inconsistent: input_dim != len(feature_names)")
        if expected_feature_names is not None and tuple(expected_feature_names) != names:
            raise ResidualError(
                "incompatible residual source: the requested feature names do not match the artifact "
                f"({len(tuple(expected_feature_names))} vs {len(names)} features)"
            )
        residual_dim = int(payload.get("residual_dim", -1))
        if residual_dim < 1:
            raise ResidualError(f"{p} has an invalid residual_dim={residual_dim}")
        if expected_residual_dim is not None and int(expected_residual_dim) != residual_dim:
            raise ResidualError(
                f"incompatible residual dimension: artifact has {residual_dim}, "
                f"requested {int(expected_residual_dim)}"
            )

        config = ResidualTrainingConfig.from_mapping(payload.get("config", {}))
        config.residual_dim = residual_dim
        config.hidden_dim = int(payload.get("hidden_dim", config.hidden_dim))
        encoder = ResidualEncoder.from_network_config(
            payload.get("network", {"input_dim": len(names), "residual_dim": residual_dim,
                                    "hidden_dim": config.hidden_dim})
        )
        decoder = ReconstructionDecoder.from_network_config(
            payload.get("network", {"input_dim": len(names), "residual_dim": residual_dim,
                                    "hidden_dim": config.hidden_dim})
        )
        encoder.load_state_dict(payload["encoder_state_dict"])
        decoder.load_state_dict(payload["decoder_state_dict"])
        encoder.eval()
        decoder.eval()
        return cls(
            encoder=encoder,
            decoder=decoder,
            config=config,
            feature_names=names,
            source_schema=dict(payload.get("source_schema", {})),
            normalization=dict(payload.get("normalization", {"mode": "none"})),
            history=list(payload.get("history", [])),
            best_epoch=int(payload.get("best_epoch", -1)),
            best_val_loss=float(payload.get("best_val_loss", float("nan"))),
            provenance=dict(payload.get("provenance", {})),
            source_config=dict(payload.get("source_config", {})),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "schema": RESIDUAL_SCHEMA,
            "uses_labels": False,
            "residual_dim": self.residual_dim,
            "input_dim": self.input_dim,
            "hidden_dim": int(self.config.hidden_dim),
            "source_schema_hash": self.schema_hash,
            "normalization": self.normalization.get("mode"),
            "mask": {
                "mask_fraction": self.config.mask_fraction,
                "mask_seed": self.config.mask_seed,
                "minimum_visible_features": self.config.minimum_visible_features,
            },
            "epochs": len(self.history),
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "provenance": self.provenance,
        }


# --------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------
@dataclass
class ResidualTrainer:
    """Self-supervised masked-reconstruction trainer (FIT-only, label-free)."""

    config: ResidualTrainingConfig

    def fit(
        self,
        source: ResidualSource,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> ResidualResult:
        """Train the residual on a residual source view and return the artifact."""
        config = self.config
        if not isinstance(source, ResidualSource):
            raise ResidualError(f"fit expects a ResidualSource, got {type(source).__name__}")
        n_rows, n_features = source.n_neurons, source.n_features
        groups = source_groups_from_names(source.feature_names)
        if config.minimum_visible_features >= n_features:
            raise ResidualError(
                f"minimum_visible_features={config.minimum_visible_features} must be < source features "
                f"{n_features}"
            )
        if n_rows < 2:
            raise ResidualError(f"the residual needs at least 2 neuron examples, got {n_rows}")

        set_seed(int(config.seed))  # python / numpy / torch (+ cudnn determinism)
        device = _resolve_device(config.device)
        dtype, dtype_warnings = _resolve_dtype(config.dtype, device)

        train_idx, val_idx = self._split(n_rows)
        normalization = self._fit_normalization(source.X, train_idx)
        X_std = self._apply_normalization(source.X, normalization)
        x_train = torch.as_tensor(X_std[train_idx], dtype=dtype, device=device)
        x_val = (
            torch.as_tensor(X_std[val_idx], dtype=dtype, device=device) if val_idx.size else None
        )

        encoder = ResidualEncoder(
            n_features, config.residual_dim, config.hidden_dim,
            activation=config.activation, dropout=config.dropout,
        ).to(device=device, dtype=dtype)
        decoder = ReconstructionDecoder(
            n_features, config.residual_dim, config.hidden_dim,
            activation=config.activation, dropout=config.dropout,
        ).to(device=device, dtype=dtype)
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=config.lr, weight_decay=config.weight_decay,
        )

        eval_mask_train = torch.as_tensor(
            make_source_mask(
                train_idx.size, n_features,
                mode=config.mask_mode,
                mask_fraction=config.mask_fraction,
                minimum_visible_features=config.minimum_visible_features,
                rng=np.random.default_rng([config.mask_seed, EVAL_MASK_OFFSET]),
                groups=groups,
            ),
            dtype=torch.bool, device=device,
        )
        eval_mask_val = (
            torch.as_tensor(
                make_source_mask(
                    val_idx.size, n_features,
                    mode=config.mask_mode,
                    mask_fraction=config.mask_fraction,
                    minimum_visible_features=config.minimum_visible_features,
                    rng=np.random.default_rng([config.mask_seed, EVAL_MASK_OFFSET + 1]),
                    groups=groups,
                ),
                dtype=torch.bool, device=device,
            )
            if val_idx.size
            else None
        )

        history: list[dict[str, Any]] = []
        best_state: dict[str, dict[str, torch.Tensor]] | None = None
        best_epoch, best_val = -1, float("inf")
        for epoch in range(config.epochs):
            encoder.train(True)
            decoder.train(True)
            mask_all = make_source_mask(
                train_idx.size, n_features,
                mode=config.mask_mode,
                mask_fraction=config.mask_fraction,
                minimum_visible_features=config.minimum_visible_features,
                rng=np.random.default_rng([config.mask_seed, epoch]),
                groups=groups,
            )
            order = np.random.default_rng([config.seed, epoch, 12345]).permutation(train_idx.size)
            epoch_losses: list[float] = []
            for start in range(0, order.size, config.batch_size):
                rows = order[start:start + config.batch_size]
                mask = torch.as_tensor(mask_all[rows], dtype=torch.bool, device=device)
                target = x_train[rows]
                optimizer.zero_grad(set_to_none=True)
                reconstruction = decoder(encoder(apply_mask(target, mask)))
                loss = masked_reconstruction_loss(
                    reconstruction, target, mask, visible_weight=config.visible_loss_weight
                )
                loss.backward()
                if config.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        list(encoder.parameters()) + list(decoder.parameters()), config.grad_clip
                    )
                optimizer.step()
                epoch_losses.append(float(loss.detach().item()))

            train_loss, train_group_losses = self._evaluate(
                encoder, decoder, x_train, eval_mask_train, groups
            )
            if x_val is not None and eval_mask_val is not None:
                val_loss, val_group_losses = self._evaluate(
                    encoder, decoder, x_val, eval_mask_val, groups
                )
            else:
                val_loss, val_group_losses = float("nan"), {}
            history.append({
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_masked_mse": float(train_loss),
                "val_masked_mse": float(val_loss),
                "optimization_loss_mean": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
                "mask_fraction_realised": float(mask_all.mean()),
                "mask_mode": config.mask_mode,
                **{f"train_masked_mse_{name}": value for name, value in train_group_losses.items()},
                **{f"val_masked_mse_{name}": value for name, value in val_group_losses.items()},
            })

            score = val_loss if np.isfinite(val_loss) else train_loss
            if score <= best_val:
                best_val, best_epoch = float(score), int(epoch)
                best_state = {
                    "encoder": {k: v.detach().clone() for k, v in encoder.state_dict().items()},
                    "decoder": {k: v.detach().clone() for k, v in decoder.state_dict().items()},
                }

        if config.restore_best and best_state is not None:
            encoder.load_state_dict(best_state["encoder"])
            decoder.load_state_dict(best_state["decoder"])
        encoder.eval()
        decoder.eval()

        provenance = {
            "schema": RESIDUAL_SCHEMA,
            "uses_labels": False,
            "fit_only": True,
            "objective": "masked_reconstruction_mse_on_withheld_coordinates",
            "mask_policy": {
                "type": "random_per_example_deterministic",
                "mode": config.mask_mode,
                "mask_fraction": config.mask_fraction,
                "mask_seed": config.mask_seed,
                "minimum_visible_features": config.minimum_visible_features,
                "eval_mask_seed_offset": EVAL_MASK_OFFSET,
                "masked_coordinates_are_zeroed_in_standardised_input": True,
                "source_groups": {name: len(values) for name, values in groups.items()},
                "block_mask_note": (
                    "mask_mode='block': one whole source group (structural / "
                    "functional_response / temporal) is withheld per example, with a "
                    "coordinate-level fallback whenever the visibility guarantee would be "
                    "violated"
                    if config.mask_mode == "block"
                    else "mask_mode='coordinate': individual coordinates are withheld"
                ),
            },
            "source_groups": {name: [int(i) for i in values] for name, values in groups.items()},
            "source_diagnostics": {
                "per_source_history_fields": (
                    [f"train_masked_mse_{name}" for name in groups]
                    + [f"val_masked_mse_{name}" for name in groups]
                ),
                "note": (
                    "per-source masked reconstruction MSE (withheld coordinates only); training "
                    "diagnostics, not scientific results"
                ),
            },
            "seeds": {
                "global_seed": config.seed,
                "split_seed": config.split_seed,
                "mask_seed": config.mask_seed,
                "shuffle_seed_derivation": "[seed, epoch, 12345]",
                "mask_seed_derivation": "[mask_seed, epoch] (evaluation: [mask_seed, 10_000_000])",
            },
            "split": {
                "kind": "neuron_examples_within_fit",
                "n_train": int(train_idx.size),
                "n_val": int(val_idx.size),
                "val_fraction": float(config.val_fraction),
                "split_seed": config.split_seed,
                "train_indices": [int(i) for i in train_idx],
                "val_indices": [int(i) for i in val_idx],
                "probe_or_test_used": False,
            },
            "normalization": normalization["mode"],
            "normalization_provenance": normalization["provenance"],
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "dtype_warnings": dtype_warnings,
            "source_schema": source.to_dict(),
            "network": {
                "encoder": f"{n_features} -> {config.hidden_dim} -> {config.residual_dim}",
                "decoder": f"{config.residual_dim} -> {config.hidden_dim} -> {n_features}",
                "activation": config.activation,
                "dropout": config.dropout,
            },
            "snn_checkpoint_modified": False,
            "note": (
                "Trained with masked reconstruction on label-free FIT-derived source features only. No "
                "labels, no PROBE, no TEST, no functional fingerprint and no decoding target; the SNN "
                "checkpoint was never loaded or modified by this trainer."
            ),
        }
        if context:
            provenance["context"] = dict(context)

        return ResidualResult(
            encoder=encoder,
            decoder=decoder,
            config=config,
            feature_names=source.feature_names,
            source_schema=source.to_dict(),
            normalization=normalization,
            history=history,
            best_epoch=best_epoch,
            best_val_loss=best_val,
            provenance=provenance,
            source_config=source.provenance.get("source_config", {}),
        )

    # -- internals -----------------------------------------------------------
    def _split(self, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng([int(self.config.split_seed), 17])
        perm = rng.permutation(n_rows)
        n_val = int(round(self.config.val_fraction * n_rows))
        n_val = max(0, min(n_val, n_rows - 1))
        val_idx = np.sort(perm[:n_val]) if n_val else np.empty(0, dtype=np.int64)
        train_idx = np.sort(perm[n_val:])
        return train_idx.astype(np.int64), val_idx.astype(np.int64)

    def _fit_normalization(self, X: np.ndarray, train_idx: np.ndarray) -> dict[str, Any]:
        if self.config.normalization == "none":
            return {"mode": "none", "mean": None, "std": None,
                    "provenance": "none (no data-derived scaling)"}
        stats = X[train_idx]
        mean = stats.mean(axis=0)
        std = np.where(stats.std(axis=0) < 1e-8, 1.0, stats.std(axis=0))
        return {
            "mode": "train_standardise",
            "mean": [float(v) for v in mean],
            "std": [float(v) for v in std],
            "provenance": (
                "FIT-derived: mean/std computed on the residual training neuron split only "
                "(no PROBE, no TEST, no labels)"
            ),
        }

    @staticmethod
    def _apply_normalization(X: np.ndarray, normalization: Mapping[str, Any]) -> np.ndarray:
        if normalization.get("mode") == "none":
            return np.asarray(X, dtype=np.float64)
        mean = np.asarray(normalization["mean"], dtype=np.float64)
        std = np.asarray(normalization["std"], dtype=np.float64)
        return (np.asarray(X, dtype=np.float64) - mean) / std

    @staticmethod
    def _evaluate(
        encoder: ResidualEncoder,
        decoder: ReconstructionDecoder,
        x: torch.Tensor,
        mask: torch.Tensor,
        groups: Mapping[str, Sequence[int]] | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Masked reconstruction loss plus the per-source-group breakdown (diagnostics)."""
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            reconstruction = decoder(encoder(apply_mask(x, mask)))
            loss = masked_reconstruction_loss(reconstruction, x, mask)
            per_group = group_masked_mse(reconstruction, x, mask, groups) if groups else {}
        return float(loss.detach().cpu().item()), per_group


# --------------------------------------------------------------------------
# Convenience API (no labels / PROBE / TEST parameters, by construction)
# --------------------------------------------------------------------------
def train_residual(
    bank: NeuronRecordBank | None = None,
    *,
    residual_dim: int | None = None,
    config: ResidualTrainingConfig | None = None,
    source_config: ResidualSourceConfig | None = None,
    source: ResidualSource | None = None,
    context: Mapping[str, Any] | None = None,
    **config_overrides: Any,
) -> ResidualResult:
    """Train a label-free residual and return the artifact.

    ``bank`` (or a prebuilt ``source``) is the only data input: there is no labels, PROBE,
    TEST or fingerprint parameter anywhere in this API.
    """
    if source is None:
        if bank is None:
            raise ResidualError("provide either `bank` or a prebuilt `source`")
        source = build_residual_source(bank, source_config)
    elif not isinstance(source, ResidualSource):
        raise ResidualError(f"source must be a ResidualSource, got {type(source).__name__}")
    if config is None:
        if residual_dim is None:
            raise ResidualError("either `config` or `residual_dim` must be given to train a residual")
        config = ResidualTrainingConfig.from_mapping({"residual_dim": int(residual_dim), **config_overrides})
    elif residual_dim is not None or config_overrides:
        raise ResidualError(
            "pass either an explicit `config` or `residual_dim`/keyword overrides, not both "
            "(residual_dim would otherwise be silently ignored)"
        )
    return ResidualTrainer(config).fit(source, context=context)


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------
def _resolve_device(mode: str) -> torch.device:
    mode = str(mode).lower()
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        device = get_device(prefer_cuda=True)
        if device.type != "cuda":
            raise ResidualError("device='cuda' was requested but CUDA is not available")
        return device
    return get_device(prefer_cuda=True)


def _resolve_dtype(name: str, device: torch.device) -> tuple[torch.dtype, list[str]]:
    """Torch dtype for a device, with a documented CPU fallback to float32."""
    mapping = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if name not in mapping:
        raise ResidualError(f"dtype must be one of {list(mapping)}, got {name!r}")
    if device.type == "cpu" and name != "float32":
        return torch.float32, [
            f"precision '{name}' is not used on CPU; falling back to float32 for the residual "
            "(CPU fallback must remain usable)"
        ]
    return mapping[name], []


def _positive_int(value: Any, name: str) -> int:
    number = _as_int(value, name)
    if number < 1:
        raise ResidualError(f"{name} must be >= 1, got {number}")
    return number


def _non_negative_int(value: Any, name: str) -> int:
    number = _as_int(value, name)
    if number < 0:
        raise ResidualError(f"{name} must be >= 0, got {number}")
    return number


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ResidualError(f"{name} must be an integer, got {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ResidualError(f"{name} must be an integer, got {value!r}") from exc


def _positive_float(value: Any, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise ResidualError(f"{name} must be a positive finite number, got {value!r}")
    return number


def _non_negative_float(value: Any, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0:
        raise ResidualError(f"{name} must be a non-negative finite number, got {value!r}")
    return number


def _fraction(value: Any, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or not 0.0 < number < 1.0:
        raise ResidualError(f"{name} must be in (0, 1), got {value!r}")
    return number


__all__ = [
    "SOURCE_SCHEMA",
    "RESIDUAL_SCHEMA",
    "EVAL_MASK_OFFSET",
    "DEFAULT_RAW_VIEW_DIM",
    "DEFAULT_FUNCTIONAL_RESPONSE_VIEW_DIM",
    "SOURCE_GROUP_STRUCTURAL",
    "SOURCE_GROUP_FUNCTIONAL",
    "SOURCE_GROUP_TEMPORAL",
    "SOURCE_GROUPS",
    "ResidualError",
    "ResidualSourceConfig",
    "ResidualSource",
    "source_group_of_feature",
    "source_groups_from_names",
    "ResidualTrainingConfig",
    "PROTOCOL_FIELDS",
    "residual_training_mismatches",
    "ResidualEncoder",
    "ReconstructionDecoder",
    "ResidualTrainer",
    "ResidualResult",
    "build_residual_source",
    "train_residual",
    "make_mask",
    "make_block_mask",
    "make_source_mask",
    "group_masked_mse",
    "apply_mask",
    "masked_reconstruction_loss",
    "feature_schema_hash",
]