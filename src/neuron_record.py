"""Unified, label-free, columnar ``NeuronRecord`` representation (V2 architecture layer).

This module replaces the *architecture* of the neuron representation while keeping
the existing scientific definitions untouched. It is the layer between "a trained
checkpoint + label-free FIT data" and "a deterministic structured vector":

    checkpoint + FIT data
              |
              v
       NeuronRecordBank            <- this module (columnar, label-free, dimension-free)
              |
              v
    deterministic structured encoder   <- later stage
              |
              v  (+ learned residual)    <- later stage

Design contract
---------------
* **Columnar / array based.** Every quantity is stored as a NumPy array whose
  leading axis is the neuron index. There is no per-neuron Python object in the
  primary representation; the object-based :class:`src.neurons.NeuronRepresentation`
  classes remain available only as a *compatibility layer*
  (:meth:`NeuronRecordBank.to_representation_set`).
* **Label-free by construction.** The bank can only be built from model parameters
  and label-free FIT statistics. ``uses_labels`` is a read-only property that is
  always ``False``, a supplied activity accumulator is rejected if it contains any
  class-conditioned statistics or labels, and no builder accepts labels at all.
* **Dimension-free.** The bank stores *source information* (raw connectivity plus the
  deterministic summary statistics the current representation uses). It makes no
  assumption about the final vector dimension ``d``; the 48-D compatibility encoder
  below is just one consumer.
* **Memory bounded.** No array may have more than two dimensions. In particular the
  forbidden shapes ``(n_neurons, n_samples, n_time)``, ``(n_samples, n_time,
  n_neurons)``, ``(n_neurons, n_samples, d)``, ``(n_neurons, n_hidden, d)`` and
  ``(n_neurons, n_time, d)`` are rejected by :func:`validate_record_array`. Activity
  is collected with the existing *streaming* :class:`src.evaluation.ActivityAccumulator`,
  so raw ``(samples, time, neurons)`` traces are never materialised.

Weight orientation (must not be confused)
-----------------------------------------
The repository convention (``src.model``) is ``w_rec[i, j]`` = weight **from** ``j``
**to** ``i`` (``forward`` computes ``s_prev @ w_rec.T``). Therefore, for neuron ``j``:

* **incoming** recurrent weights (onto ``j``) are the **row** ``w_rec[j, :]``;
* **outgoing** recurrent weights (from ``j``) are the **column** ``w_rec[:, j]``;
* input weights onto ``j`` are the **column** ``w_in[:, j]``.

Each block exposes this explicitly through ``orientation`` and through the array
layout (``weights[i]`` is neuron ``i``'s vector), and
``recurrent_out.weights == recurrent_in.weights.T`` by construction.

Compatibility
-------------
:meth:`NeuronRecordBank.to_structured_matrix` and
:func:`structural_matrix_from_record_bank` reproduce the *current* deterministic
48-D structural representation exactly (same feature formulas, reused from
:mod:`src.neurons`, same block order and the same alphabetical within-block order),
and :meth:`NeuronRecordBank.to_representation_set` returns the existing
:class:`src.neurons.NeuronRepresentationSet` so every downstream module works
unchanged.

Not implemented in this stage (and never faked here): the learned residual,
variable-dimensional encoders, the ``temporal`` and ``network_context`` blocks.
Those blocks are declared by :data:`src.v2_config.V2_RECORD_BLOCKS` but the bank
reports them as unimplemented rather than inventing placeholder features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .evaluation import ActivityAccumulatorResult, collect_activity
from .neurons import (
    FeatureBlock,
    NeuronRepresentation,
    NeuronRepresentationSet,
    activity_features_from_psth,
    add_first_spike_features,
    input_connectivity_features,
    intrinsic_feature_provenance,
    intrinsic_features_from_model,
    recurrent_incoming_features,
    recurrent_outgoing_features,
    recurrent_relationship_features,
)
from .utils import get_device, sanitize_features
from .v2_config import (
    IMPLEMENTED_RECORD_BLOCKS,
    V2_RECORD_BLOCKS,
    V2Config,
)

#: Schema identifier written into every bank's provenance.
RECORD_SCHEMA = "neuron_record_bank/v1"

#: Positional/dtype information that the pristine 48-D compatibility encoder uses.
STRUCTURAL_BLOCKS: tuple[str, ...] = tuple(FeatureBlock.structural())

#: Human-readable, machine-readable description of where each block's numbers come from.
BLOCK_SOURCES: dict[str, str] = {
    "intrinsic": "model parameters: per-neuron learned quantities (only those that vary)",
    "input_conn": "model parameter w_in (column per neuron) + label-free summary statistics",
    "recurrent_in": "model parameter w_rec row per neuron (incoming) + label-free statistics",
    "recurrent_out": "model parameter w_rec column per neuron (outgoing) + label-free statistics",
    "activity": "label-free FIT spike statistics collected with the streaming accumulator",
    "temporal": "not implemented in this stage",
    "network_context": "not implemented in this stage",
}


class NeuronRecordError(ValueError):
    """Raised for an invalid record-bank construction or a forbidden tensor shape."""


# --------------------------------------------------------------------------
# Memory guard
# --------------------------------------------------------------------------
def validate_record_array(
    array: Any,
    name: str,
    n_neurons: int,
    *,
    allow_sample_major: bool = False,
) -> np.ndarray:
    """Reject record-bank arrays whose shape violates the V2 memory contract.

    Allowed: 1-D ``(n_neurons,)`` and 2-D ``(n_neurons, k)``. A 2-D
    ``(n_samples, n_neurons)`` array is allowed only when ``allow_sample_major``
    is set (used for the optional per-sample activity counts).

    Anything with three or more dimensions - the bulk of the forbidden
    ``(neurons, samples, time)`` / ``(neurons, samples, d)`` family - raises
    :class:`NeuronRecordError`.
    """
    arr = np.asarray(array)
    if arr.ndim == 0:
        raise NeuronRecordError(
            f"{name}: scalar arrays are not allowed in a record block; expected a neuron axis"
        )
    if arr.ndim > 2:
        raise NeuronRecordError(
            f"{name}: forbidden shape {arr.shape}. Record-bank arrays must be 1-D "
            f"(n_neurons,) or 2-D (n_neurons, k). Tensors such as "
            f"(n_neurons, n_samples, n_time) or (n_neurons, n_samples, d) are explicitly "
            f"forbidden by the V2 memory contract; reduce them with the streaming accumulator "
            f"or chunked construction instead."
        )
    if arr.ndim == 1:
        if arr.shape[0] != n_neurons:
            raise NeuronRecordError(
                f"{name}: expected {n_neurons} entries (one per neuron), got shape {arr.shape}"
            )
        return arr
    # 2-D
    if arr.shape[0] == n_neurons:
        if arr.shape[1] < 1:
            raise NeuronRecordError(f"{name}: the per-neuron feature/weight axis is empty ({arr.shape})")
        return arr
    if allow_sample_major and arr.shape[1] == n_neurons:
        return arr
    raise NeuronRecordError(
        f"{name}: unexpected shape {arr.shape} for {n_neurons} neurons. Feature/weight matrices "
        f"must be neuron-major (n_neurons, k); a sample-major (n_samples, n_neurons) matrix is "
        f"allowed only for the optional per-sample activity counts."
    )


# --------------------------------------------------------------------------
# One columnar block
# --------------------------------------------------------------------------
@dataclass
class RecordBlock:
    """A columnar block of the record: scalar features and/or raw weight vectors.

    Parameters
    ----------
    name:
        Block name (one of :data:`src.v2_config.V2_RECORD_BLOCKS`).
    n_neurons:
        Number of neurons the block describes; every array's neuron axis must match.
    features:
        ``{feature_name: (n_neurons,) array}`` of deterministic scalar summaries.
    weights:
        Optional ``(n_neurons, k)`` matrix of raw vectors (e.g. one connectivity row
        per neuron), with ``weights[i]`` describing neuron ``i``.
    flags:
        Optional ``{flag_name: (n_neurons,) bool|float array}`` bookkeeping columns that
        are *not* representation features (e.g. ``silent_neuron``).
    samples:
        Optional ``(n_samples, n_neurons)`` per-sample scalar matrix (activity counts).
        This is the only sample-major array the contract allows.
    orientation:
        Explicit human-readable statement of what ``weights[i]`` means.
    """

    name: str
    n_neurons: int
    features: Mapping[str, np.ndarray] = field(default_factory=dict)
    weights: np.ndarray | None = None
    weight_names: tuple[str, ...] = ()
    flags: Mapping[str, np.ndarray] = field(default_factory=dict)
    samples: np.ndarray | None = None
    orientation: str = ""
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = str(self.name)
        if not self.name:
            raise NeuronRecordError("record block name must be a non-empty string")
        self.n_neurons = int(self.n_neurons)
        if self.n_neurons < 1:
            raise NeuronRecordError(f"block {self.name!r}: n_neurons must be >= 1")

        prepared: dict[str, np.ndarray] = {}
        for fname, values in dict(self.features).items():
            raw = np.asarray(values, dtype=np.float64)
            # validate the *original* shape first: flattening before validating would
            # silently accept a forbidden (n_neurons, n_samples, n_time) tensor.
            validate_record_array(raw, f"{self.name}.{fname}", self.n_neurons)
            if raw.ndim != 1:
                raise NeuronRecordError(
                    f"{self.name}.{fname}: a scalar feature must be one value per neuron (1-D), "
                    f"got shape {raw.shape}"
                )
            arr = np.array(raw, copy=True)
            arr.setflags(write=False)
            prepared[str(fname)] = arr
        # deterministic feature order (alphabetical, as the existing to_matrix does)
        self.features = MappingProxyType(dict(sorted(prepared.items())))

        prepared_flags: dict[str, np.ndarray] = {}
        for flag, values in dict(self.flags).items():
            raw = np.asarray(values)
            validate_record_array(raw, f"{self.name}.flag[{flag}]", self.n_neurons)
            if raw.ndim != 1:
                raise NeuronRecordError(
                    f"{self.name}.flag[{flag}]: a flag must be one value per neuron (1-D), "
                    f"got shape {raw.shape}"
                )
            arr = np.array(raw if raw.dtype == bool else raw.astype(np.float64), copy=True)
            arr.setflags(write=False)
            prepared_flags[str(flag)] = arr
        self.flags = MappingProxyType(dict(sorted(prepared_flags.items())))

        if self.weights is not None:
            raw_weights = np.asarray(self.weights, dtype=np.float64)
            validate_record_array(raw_weights, f"{self.name}.weights", self.n_neurons)
            if raw_weights.ndim != 2:
                raise NeuronRecordError(
                    f"{self.name}.weights must be a 2-D (n_neurons, k) matrix, "
                    f"got shape {raw_weights.shape}"
                )
            weights = np.array(raw_weights, copy=True)
            weights.setflags(write=False)
            self.weights = weights
            if self.weight_names and len(self.weight_names) != weights.shape[1]:
                raise NeuronRecordError(
                    f"block {self.name!r}: weight_names has {len(self.weight_names)} entries but "
                    f"weights has {weights.shape[1]} columns"
                )

        if self.samples is not None:
            raw_samples = np.asarray(self.samples, dtype=np.float32)
            validate_record_array(
                raw_samples, f"{self.name}.samples", self.n_neurons, allow_sample_major=True
            )
            if raw_samples.ndim != 2 or raw_samples.shape[1] != self.n_neurons:
                raise NeuronRecordError(
                    f"{self.name}.samples must be (n_samples, n_neurons) = (n_samples, "
                    f"{self.n_neurons}); got shape {raw_samples.shape}. The (n_neurons, n_samples) "
                    f"orientation is forbidden."
                )
            samples = np.array(raw_samples, copy=True)
            samples.setflags(write=False)
            self.samples = samples

        if not self.features and self.weights is None and self.samples is None:
            raise NeuronRecordError(
                f"block {self.name!r} is empty: it must carry features, weights or samples"
            )
        if self.source:
            self.metadata = {**self.metadata, "source": self.source}

    # -- introspection -------------------------------------------------------
    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.features)

    @property
    def flag_names(self) -> tuple[str, ...]:
        return tuple(self.flags)

    @property
    def n_features(self) -> int:
        return len(self.features)

    @property
    def n_weights(self) -> int:
        return 0 if self.weights is None else int(self.weights.shape[1])

    @property
    def n_sample_rows(self) -> int:
        return 0 if self.samples is None else int(self.samples.shape[0])

    def to_matrix(self) -> tuple[np.ndarray, list[str]]:
        """``(X, names)`` for the scalar features, ordered by :attr:`feature_names`."""
        X = np.empty((self.n_neurons, self.n_features), dtype=np.float64)
        for col, fname in enumerate(self.feature_names):
            X[:, col] = self.features[fname]
        return X, [f"{self.name}.{n}" for n in self.feature_names]

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_neurons": self.n_neurons,
            "n_features": self.n_features,
            "feature_names": list(self.feature_names),
            "n_weights": self.n_weights,
            "weight_shape": None if self.weights is None else list(self.weights.shape),
            "n_flags": len(self.flags),
            "flag_names": list(self.flag_names),
            "samples_shape": None if self.samples is None else list(self.samples.shape),
            "orientation": self.orientation,
            "source": self.source,
        }


# --------------------------------------------------------------------------
# The bank
# --------------------------------------------------------------------------
@dataclass
class NeuronRecordBank:
    """Columnar, label-free, dimension-free representation of every hidden neuron.

    Parameters
    ----------
    n_neurons:
        Number of hidden neurons (``model.cfg.n_hidden``).
    blocks:
        ``{block_name: RecordBlock}``. Populated blocks depend on the inputs: the
        structural blocks always exist, ``activity`` exists only when label-free FIT
        statistics were supplied/collected.
    declared_blocks:
        The full V2 schema (including blocks that are not implemented yet).
    provenance:
        Machine-readable description of every source (see :meth:`summary`).
    """

    n_neurons: int
    blocks: dict[str, RecordBlock]
    provenance: dict[str, Any] = field(default_factory=dict)
    declared_blocks: tuple[str, ...] = V2_RECORD_BLOCKS

    def __post_init__(self) -> None:
        self.n_neurons = int(self.n_neurons)
        if self.n_neurons < 1:
            raise NeuronRecordError(f"n_neurons must be >= 1, got {self.n_neurons}")
        self.declared_blocks = tuple(str(b) for b in self.declared_blocks)
        self.blocks = {str(name): block for name, block in dict(self.blocks).items()}
        if not self.blocks:
            raise NeuronRecordError("a NeuronRecordBank must contain at least one block")
        # canonical block order: declared order first, then any extras (defensive)
        ordered = {name: self.blocks[name] for name in self.declared_blocks if name in self.blocks}
        for name in self.blocks:
            ordered.setdefault(name, self.blocks[name])
        self.blocks = ordered

        for name, block in self.blocks.items():
            if not isinstance(block, RecordBlock):
                raise NeuronRecordError(f"block {name!r} must be a RecordBlock, got {type(block)!r}")
            if block.n_neurons != self.n_neurons:
                raise NeuronRecordError(
                    f"block {name!r} covers {block.n_neurons} neurons but the bank has {self.n_neurons}"
                )
        self.provenance = dict(self.provenance)
        self.provenance.setdefault("schema", RECORD_SCHEMA)
        self.provenance["uses_labels"] = False
        self.provenance.setdefault("n_neurons", self.n_neurons)
        self.provenance.setdefault(
            "blocks",
            {name: {"source": BLOCK_SOURCES.get(name, ""), "uses_labels": False} for name in self.blocks},
        )
        self.assert_no_forbidden_tensors()

    # -- hard invariants -----------------------------------------------------
    @property
    def uses_labels(self) -> bool:
        """Always ``False``: the bank is label-free by construction (read-only)."""
        return False

    def assert_no_forbidden_tensors(self) -> bool:
        """Re-check that no stored array violates the memory contract; return ``True``."""
        for name, block in self.blocks.items():
            for fname, arr in block.features.items():
                validate_record_array(arr, f"{name}.{fname}", self.n_neurons)
            for flag, arr in block.flags.items():
                validate_record_array(arr, f"{name}.flag[{flag}]", self.n_neurons)
            if block.weights is not None:
                validate_record_array(block.weights, f"{name}.weights", self.n_neurons)
            if block.samples is not None:
                validate_record_array(
                    block.samples, f"{name}.samples", self.n_neurons, allow_sample_major=True
                )
        return True

    # -- schema / introspection ---------------------------------------------
    @property
    def block_names(self) -> tuple[str, ...]:
        """Names of the blocks actually present in this bank."""
        return tuple(self.blocks)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Qualified ``"block.feature"`` names, in deterministic block+feature order."""
        names: list[str] = []
        for bname, block in self.blocks.items():
            names.extend(f"{bname}.{fname}" for fname in block.feature_names)
        return tuple(names)

    @property
    def unimplemented_blocks(self) -> tuple[str, ...]:
        """Declared blocks that this stage does not implement."""
        return tuple(b for b in self.declared_blocks if b not in IMPLEMENTED_RECORD_BLOCKS)

    def block_status(self) -> dict[str, dict[str, Any]]:
        """Implementation/presence status of every declared block (machine-readable)."""
        status: dict[str, dict[str, Any]] = {}
        for name in self.declared_blocks:
            implemented = name in IMPLEMENTED_RECORD_BLOCKS
            present = name in self.blocks
            if present:
                reason = ""
            elif implemented:
                reason = "implemented but not present (no source data supplied for this bank)"
            else:
                reason = "not implemented in this stage"
            status[name] = {
                "implemented": bool(implemented),
                "present": bool(present),
                "uses_labels": False,
                "source": BLOCK_SOURCES.get(name, ""),
                "reason": reason,
            }
        return status

    def get_block(self, name: str) -> RecordBlock:
        """Return a block, with an explicit error for unknown/unimplemented blocks."""
        key = str(name)
        if key in self.blocks:
            return self.blocks[key]
        if key in self.declared_blocks:
            status = self.block_status()[key]
            raise NeuronRecordError(
                f"block {key!r} is declared in the V2 schema but not available in this bank "
                f"({status['reason']})"
            )
        raise NeuronRecordError(
            f"unknown block {key!r}; valid block names are {list(self.declared_blocks)}"
        )

    def _resolve_blocks(self, blocks: str | Iterable[str] | None) -> list[str]:
        if blocks is None:
            return list(self.block_names)
        if isinstance(blocks, str):
            raw = [part.strip() for part in blocks.replace(",", " ").split() if part.strip()]
        else:
            raw = [str(b) for b in blocks]
        unknown = sorted({b for b in raw if b not in self.declared_blocks})
        if unknown:
            raise NeuronRecordError(
                f"unknown block(s) {unknown}; valid block names are {list(self.declared_blocks)}"
            )
        selected = [b for b in self.declared_blocks if b in raw]
        unimplemented = [b for b in selected if b in self.unimplemented_blocks]
        if unimplemented:
            raise NeuronRecordError(
                f"block(s) {unimplemented} are declared but not implemented in this stage"
            )
        return selected

    # -- matrix construction (chunk friendly) -------------------------------
    def to_structured_matrix(
        self,
        blocks: str | Iterable[str] | None = None,
        chunk_size: int | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Assemble the scalar feature matrix ``(n_neurons, n_features)``.

        ``blocks=None`` uses every present block in canonical order; a block that is
        implemented but absent (e.g. ``intrinsic`` for an untrained model, or
        ``activity`` when no FIT data was supplied) simply contributes no columns,
        matching the existing :meth:`NeuronRepresentationSet.to_matrix` behaviour.

        ``chunk_size`` builds the matrix in neuron chunks, so peak temporary memory is
        ``chunk_size x n_features`` instead of ``n_neurons x n_features``.
        """
        selected = self._resolve_blocks(blocks)
        columns: list[tuple[str, str, str]] = []
        for bname in selected:
            block = self.blocks.get(bname)
            if block is None:
                continue
            for fname in block.feature_names:
                columns.append((f"{bname}.{fname}", bname, fname))

        X = np.empty((self.n_neurons, len(columns)), dtype=np.float64)
        names = [c[0] for c in columns]
        if not columns:
            return X, names
        size = self.n_neurons if chunk_size is None else int(chunk_size)
        if size < 1:
            raise NeuronRecordError(f"chunk_size must be >= 1, got {size}")
        for start in range(0, self.n_neurons, size):
            stop = min(start + size, self.n_neurons)
            for col, (_qualified, bname, fname) in enumerate(columns):
                X[start:stop, col] = self.blocks[bname].features[fname][start:stop]
        return X, names

    # -- compatibility with the existing representation objects -------------
    def to_representation_set(
        self,
        blocks: str | Iterable[str] | None = None,
    ) -> NeuronRepresentationSet:
        """Compatibility adapter to the existing object-based representation.

        The returned :class:`~src.neurons.NeuronRepresentationSet` is numerically
        identical to the one produced by the existing builders (for the structural
        blocks: :func:`~src.neurons.extract_structural_representations`; for the
        activity block: :func:`~src.neurons.build_activity_representations`), so every
        downstream module (spaces, geometry, prediction, controls) works unchanged.
        """
        selected = self._resolve_blocks(blocks)
        reps: list[NeuronRepresentation] = []
        for j in range(self.n_neurons):
            features: dict[str, dict[str, float]] = {}
            metadata: dict[str, Any] = {}
            for bname in selected:
                block = self.blocks.get(bname)
                if block is None:
                    continue
                features[bname] = {fname: float(block.features[fname][j]) for fname in block.feature_names}
                if bname == FeatureBlock.ACTIVITY.value:
                    for flag, arr in block.flags.items():
                        metadata[flag] = bool(arr[j]) if arr.dtype == bool else float(arr[j])
                else:
                    metadata.setdefault("source", "structure")
            reps.append(NeuronRepresentation(neuron_id=j, features=features, metadata=metadata))
        return NeuronRepresentationSet(
            reps,
            meta={
                "kind": "neuron_record",
                "uses_labels": False,
                "n_hidden": self.n_neurons,
                "blocks_present": list(self.block_names),
                "record_schema": RECORD_SCHEMA,
                "note": (
                    "Compatibility view of a NeuronRecordBank; values are taken verbatim from the "
                    "columnar blocks, so downstream results are unchanged."
                ),
            },
        )

    # -- reporting -----------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        return {
            "schema": RECORD_SCHEMA,
            "n_neurons": self.n_neurons,
            "uses_labels": False,
            "block_names": list(self.block_names),
            "feature_names": list(self.feature_names),
            "declared_blocks": list(self.declared_blocks),
            "unimplemented_blocks": list(self.unimplemented_blocks),
            "blocks": {name: block.summary() for name, block in self.blocks.items()},
            "block_status": self.block_status(),
        }

    # -- construction --------------------------------------------------------
    @classmethod
    def from_model(cls, model: Any, **kwargs: Any) -> "NeuronRecordBank":
        """Build directly from an in-memory :class:`~src.model.RecurrentLIFSNN`."""
        return build_neuron_record_bank(model, **kwargs)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Any,
        *,
        map_location: Any = "cpu",
        **kwargs: Any,
    ) -> "NeuronRecordBank":
        """Build from a saved checkpoint (loaded with the repository's loader)."""
        from .model import RecurrentLIFSNN

        model, _extra = RecurrentLIFSNN.load(str(checkpoint), map_location=map_location)
        return build_neuron_record_bank(model, **kwargs)


# --------------------------------------------------------------------------
# Block builders (reuse the existing scientific definitions verbatim)
# --------------------------------------------------------------------------
def _intrinsic_block(model: Any, n_neurons: int) -> RecordBlock | None:
    """Per-neuron learned parameters, using the existing "only if it varies" rule."""
    intrinsic = intrinsic_features_from_model(model)
    if not intrinsic:
        return None  # untrained / constant parameters -> no intrinsic column (existing rule)
    features = {name: np.asarray(values, dtype=np.float64) for name, values in intrinsic.items()}
    return RecordBlock(
        name=FeatureBlock.INTRINSIC.value,
        n_neurons=n_neurons,
        features=features,
        source=BLOCK_SOURCES[FeatureBlock.INTRINSIC.value],
        metadata={"feature_kind": "generic_learned_or_dynamical_per_neuron_parameter"},
    )


def _input_conn_block(model: Any, n_neurons: int, *, include_tonotopic: bool) -> RecordBlock:
    """Input-weight vectors (column per neuron) + the existing summary statistics."""
    import torch

    with torch.no_grad():
        w_in = model.w_in.detach().cpu().numpy().astype(np.float64)  # (n_input, n_hidden)
    if w_in.shape[1] != n_neurons:
        raise NeuronRecordError(
            f"w_in has {w_in.shape[1]} columns but the bank has {n_neurons} neurons"
        )
    columns: dict[str, list[float]] = {}
    key_order: list[str] | None = None
    for j in range(n_neurons):
        feats = input_connectivity_features(w_in[:, j], include_tonotopic=include_tonotopic)
        if key_order is None:
            key_order = list(feats)
            columns = {k: [] for k in key_order}
        elif list(feats) != key_order:
            raise NeuronRecordError("input_connectivity_features returned inconsistent feature names")
        for k, v in feats.items():
            columns[k].append(float(v))
    features = {k: np.asarray(v, dtype=np.float64) for k, v in columns.items()}
    weights = np.ascontiguousarray(w_in.T)  # row j = w_in[:, j] (weights onto hidden neuron j)
    return RecordBlock(
        name=FeatureBlock.INPUT_CONN.value,
        n_neurons=n_neurons,
        features=features,
        weights=weights,
        weight_names=tuple(f"channel_{c}" for c in range(w_in.shape[0])),
        orientation="weights[j, c] = w_in[c, j] (input weight from channel c onto neuron j); row j = neuron j",
        source=BLOCK_SOURCES[FeatureBlock.INPUT_CONN.value],
        metadata={"include_tonotopic_features": bool(include_tonotopic)},
    )


def _recurrent_blocks(model: Any, n_neurons: int) -> tuple[RecordBlock, RecordBlock]:
    """Incoming (row) and outgoing (column) recurrent connectivity, kept distinct."""
    import torch

    with torch.no_grad():
        w_rec = model.w_rec.detach().cpu().numpy().astype(np.float64)  # w_rec[i, j] = from j to i
    if w_rec.shape != (n_neurons, n_neurons):
        raise NeuronRecordError(
            f"w_rec has shape {w_rec.shape} but the bank has {n_neurons} neurons"
        )

    incoming_cols: dict[str, list[float]] = {}
    outgoing_cols: dict[str, list[float]] = {}
    in_order: list[str] | None = None
    out_order: list[str] | None = None
    for j in range(n_neurons):
        inc = recurrent_incoming_features(w_rec[j, :])  # row j = weights onto j
        inc.update(recurrent_relationship_features(w_rec, j))
        out = recurrent_outgoing_features(w_rec[:, j])  # column j = weights from j
        if in_order is None:
            in_order, out_order = list(inc), list(out)
            incoming_cols = {k: [] for k in in_order}
            outgoing_cols = {k: [] for k in out_order}
        elif list(inc) != in_order or list(out) != out_order:
            raise NeuronRecordError("recurrent feature extraction returned inconsistent feature names")
        for k, v in inc.items():
            incoming_cols[k].append(float(v))
        for k, v in out.items():
            outgoing_cols[k].append(float(v))

    incoming = RecordBlock(
        name=FeatureBlock.RECURRENT_IN.value,
        n_neurons=n_neurons,
        features={k: np.asarray(v, dtype=np.float64) for k, v in incoming_cols.items()},
        weights=np.ascontiguousarray(w_rec),  # row j = w_rec[j, :] (weights onto neuron j)
        weight_names=tuple(f"from_neuron_{i}" for i in range(n_neurons)),
        orientation=(
            "weights[j, i] = w_rec[j, i] (weight from neuron i onto neuron j). "
            "Incoming = ROW w_rec[j, :]; forward computes s_prev @ w_rec.T, so neuron j "
            "receives sum_i s_prev[i] * w_rec[j, i]."
        ),
        source=BLOCK_SOURCES[FeatureBlock.RECURRENT_IN.value],
        metadata={"relationship_features_included": True},
    )
    outgoing = RecordBlock(
        name=FeatureBlock.RECURRENT_OUT.value,
        n_neurons=n_neurons,
        features={k: np.asarray(v, dtype=np.float64) for k, v in outgoing_cols.items()},
        weights=np.ascontiguousarray(w_rec.T),  # row j = w_rec[:, j] (weights from neuron j)
        weight_names=tuple(f"to_neuron_{i}" for i in range(n_neurons)),
        orientation=(
            "weights[j, i] = w_rec[i, j] (weight from neuron j onto neuron i). "
            "Outgoing = COLUMN w_rec[:, j]; equals recurrent_in.weights.T."
        ),
        source=BLOCK_SOURCES[FeatureBlock.RECURRENT_OUT.value],
        metadata={"relationship_features_included": False},
    )
    return incoming, outgoing


def _assert_label_free_accumulator(result: ActivityAccumulatorResult) -> None:
    """Reject an activity accumulator that carries labels or class-conditioned data."""
    problems: list[str] = []
    if result.labels is not None:
        problems.append("labels")
    if np.any(result.class_psth):
        problems.append("class_psth")
    if np.any(result.class_counts):
        problems.append("class_counts")
    if np.any(result.class_n):
        problems.append("class_n")
    if np.any(result.class_first_spike_sum) or np.any(result.class_first_spike_count):
        problems.append("class_first_spike_*")
    if problems:
        raise NeuronRecordError(
            "the supplied activity accumulator contains label-derived quantities "
            f"({', '.join(problems)}); a NeuronRecordBank must be label-free. Re-collect it with "
            "collect_activity(..., with_labels=False)."
        )


def _activity_block(
    result: ActivityAccumulatorResult,
    *,
    store_sample_counts: bool,
    split_name: str,
    orientation_note: str,
) -> RecordBlock:
    """Label-free FIT activity block, reusing the existing feature definitions."""
    features, flags = activity_features_from_psth(
        result.psth, result.counts, bin_ms=result.bin_ms, n_samples=result.n_samples
    )
    features = add_first_spike_features(
        features,
        result.first_spike_sum,
        result.first_spike_count,
        duration_ms=result.duration_ms,
    )
    # sanitize column-wise: elementwise identical to the per-element sanitize used by
    # build_activity_representations, because sanitize_features is elementwise.
    features = {name: sanitize_features(np.asarray(values, dtype=np.float64)) for name, values in features.items()}
    flag_arrays = {
        "silent_neuron": np.asarray(flags["silent_neuron"], dtype=bool),
        "total_spikes": np.asarray(flags["total_spikes"], dtype=np.float64),
    }
    samples = result.counts.astype(np.float32) if store_sample_counts else None
    return RecordBlock(
        name=FeatureBlock.ACTIVITY.value,
        n_neurons=int(result.n_hidden),
        features=features,
        flags=flag_arrays,
        samples=samples,
        orientation=orientation_note,
        source=BLOCK_SOURCES[FeatureBlock.ACTIVITY.value],
        metadata={
            "n_samples": int(result.n_samples),
            "n_bins": int(result.n_bins),
            "bin_ms": float(result.bin_ms),
            "split": split_name,
            "sample_counts_stored": bool(store_sample_counts),
            "sample_counts_dtype": "float32",
            "voltage_statistics_used": False,
        },
    )


# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------
def build_neuron_record_bank(
    model: Any,
    *,
    fit_rec: Any | None = None,
    fit_idx: Sequence[int] | np.ndarray | None = None,
    activity: ActivityAccumulatorResult | None = None,
    config: V2Config | None = None,
    include_tonotopic: bool = False,
    store_sample_counts: bool = True,
    with_activity: bool = True,
    device: Any | None = None,
    batch_size: int | None = None,
) -> NeuronRecordBank:
    """Build the canonical label-free :class:`NeuronRecordBank` for a model.

    Which inputs are used (nothing else is ever read):

    * ``model`` - parameters (``w_in``, ``w_rec``, per-neuron learned parameters) only.
    * either ``activity`` (an *already* label-free :class:`ActivityAccumulatorResult`,
      e.g. from ``collect_activity(..., with_labels=False)``) **or** ``fit_rec`` +
      ``fit_idx``, in which case the activity statistics are collected here with the
      existing streaming accumulator. Passing both is an error.
    * ``config`` (optional :class:`~src.v2_config.V2Config`) supplies the memory policy:
      ``memory.record_batch_size`` (streaming batch), ``memory.representation_chunk_size``
      (recorded in provenance; used by :meth:`NeuronRecordBank.to_structured_matrix`
      only when a chunk size is requested), and ``memory.storage`` (must be CPU-side:
      ``cpu`` or ``memmap`` - GPU storage is not implemented in this stage). The
      dense-input token budget is checked before the activity pass.

    The builder takes **no labels argument**: it cannot read class labels, class
    firing rates or any class-conditioned quantity, and it never touches PROBE or
    TEST data (only the FIT recordings/indices passed in are simulated).

    Returns a CPU-side, read-only :class:`NeuronRecordBank`.
    """
    if activity is not None and fit_rec is not None:
        raise NeuronRecordError("pass either `activity` or `fit_rec`, not both")
    if config is not None:
        if config.memory.storage == "gpu":
            raise NeuronRecordError(
                "GPU storage for the record bank is not implemented in this stage; "
                "use memory.storage='cpu'"
            )

    import torch

    with torch.no_grad():
        n_neurons = int(model.cfg.n_hidden)
        n_input = int(model.cfg.n_input)
        n_bins = int(model.cfg.n_bins)

    resolved_activity: ActivityAccumulatorResult | None = None
    split_name = "none"
    used_batch_size: int | None = None
    if with_activity:
        if activity is not None:
            resolved_activity = activity
            split_name = "supplied_accumulator"
        elif fit_rec is not None:
            idx = np.arange(len(fit_rec), dtype=np.int64) if fit_idx is None else np.asarray(fit_idx, dtype=np.int64)
            used_batch_size = int(batch_size) if batch_size is not None else (
                int(config.memory.record_batch_size) if config is not None else 32
            )
            if config is not None:
                config.memory.check_input_budget(batch_size=used_batch_size, n_bins=n_bins, n_input=n_input)
            resolved_activity = collect_activity(
                model,
                fit_rec,
                idx,
                device=device if device is not None else get_device(prefer_cuda=True),
                batch_size=used_batch_size,
                n_classes=int(model.cfg.n_output),
                with_labels=False,
                collect_voltage=False,
            )
            split_name = str(getattr(fit_rec, "name", "fit"))
        if resolved_activity is not None:
            _assert_label_free_accumulator(resolved_activity)

    # -- blocks (all reusing the existing scientific definitions) -------------
    blocks: dict[str, RecordBlock] = {}
    intrinsic = _intrinsic_block(model, n_neurons)
    if intrinsic is not None:
        blocks[FeatureBlock.INTRINSIC.value] = intrinsic
    blocks[FeatureBlock.INPUT_CONN.value] = _input_conn_block(
        model, n_neurons, include_tonotopic=include_tonotopic
    )
    rec_in, rec_out = _recurrent_blocks(model, n_neurons)
    blocks[FeatureBlock.RECURRENT_IN.value] = rec_in
    blocks[FeatureBlock.RECURRENT_OUT.value] = rec_out
    if resolved_activity is not None:
        blocks[FeatureBlock.ACTIVITY.value] = _activity_block(
            resolved_activity,
            store_sample_counts=store_sample_counts,
            split_name=split_name,
            orientation_note=(
                "feature[j] summarises the label-free FIT spiking of neuron j; "
                "samples[s, j] is neuron j's spike count on FIT sample s (no class information)"
            ),
        )

    provenance = _build_provenance(
        model,
        n_neurons=n_neurons,
        n_input=n_input,
        n_bins=n_bins,
        blocks=blocks,
        split_name=split_name,
        used_batch_size=used_batch_size,
        include_tonotopic=include_tonotopic,
        store_sample_counts=store_sample_counts,
        config=config,
    )
    return NeuronRecordBank(n_neurons=n_neurons, blocks=blocks, provenance=provenance)


def _build_provenance(
    model: Any,
    *,
    n_neurons: int,
    n_input: int,
    n_bins: int,
    blocks: Mapping[str, RecordBlock],
    split_name: str,
    used_batch_size: int | None,
    include_tonotopic: bool,
    store_sample_counts: bool,
    config: V2Config | None,
) -> dict[str, Any]:
    """Machine-readable provenance: what produced each block and what is label-free."""
    intrinsic_info = intrinsic_feature_provenance(model)
    provenance: dict[str, Any] = {
        "schema": RECORD_SCHEMA,
        "created_by": "src.neuron_record.build_neuron_record_bank",
        "uses_labels": False,
        "n_neurons": n_neurons,
        "model": {
            "n_input": n_input,
            "n_hidden": n_neurons,
            "n_output": int(model.cfg.n_output),
            "n_bins": n_bins,
            "bin_ms": float(model.cfg.bin_ms),
            "neuron_param_mode": str(model.cfg.neuron_param_mode),
            "readout_mode": str(model.cfg.readout_mode),
            "intrinsic_feature_provenance": intrinsic_info,
        },
        "weight_orientation": {
            "w_rec[i, j]": "weight from neuron j to neuron i (forward: s_prev @ w_rec.T)",
            "recurrent_in.weights[i]": "row w_rec[i, :] (weights onto neuron i)",
            "recurrent_out.weights[i]": "column w_rec[:, i] (weights from neuron i)",
            "input_conn.weights[i]": "column w_in[:, i] (weights from input channels onto neuron i)",
        },
        "blocks": {
            name: {
                "source": BLOCK_SOURCES.get(name, ""),
                "uses_labels": False,
                "n_features": block.n_features,
                "feature_names": list(block.feature_names),
                "n_weights": block.n_weights,
                "orientation": block.orientation,
            }
            for name, block in blocks.items()
        },
        "activity": {
            "present": "activity" in blocks,
            "split": split_name,
            "batch_size": used_batch_size,
            "sample_counts_stored": bool(store_sample_counts),
            "collect_voltage": False,
            "label_free": True,
        },
        "include_tonotopic_features": bool(include_tonotopic),
        "declared_blocks": list(V2_RECORD_BLOCKS),
        "implemented_blocks": [b for b in V2_RECORD_BLOCKS if b in IMPLEMENTED_RECORD_BLOCKS],
        "unimplemented_blocks": [b for b in V2_RECORD_BLOCKS if b not in IMPLEMENTED_RECORD_BLOCKS],
        "leakage_contract": (
            "This bank is built exclusively from model parameters and label-free FIT "
            "statistics. Class labels, class-conditioned PSTHs/rates, functional "
            "fingerprints and the official TEST set are never read here."
        ),
    }
    if config is not None:
        provenance["v2_config"] = {
            "vector": {
                "d": config.vector.d,
                "structured_d": config.vector.structured_d,
                "learned_residual_d": config.vector.learned_residual_d,
                "residual_enabled": bool(config.vector.residual.enabled),
                "enabled_blocks": list(config.vector.enabled_blocks),
                "temporal_resolution": config.vector.temporal_resolution,
                "context_depth": config.vector.context_depth,
            },
            "memory": config.memory.to_dict(),
            "precision": config.precision.to_dict(),
            "warnings": list(config.warnings),
        }
    return provenance


def structural_matrix_from_record_bank(
    bank: NeuronRecordBank,
    *,
    chunk_size: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Compatibility encoder: the current deterministic 48-D structural representation.

    Equivalent to ``extract_structural_representations(model).to_matrix()`` (same
    formulas, same block order, same within-block alphabetical order), but read from
    the columnar bank. The name is explicit about the *only* dimension implemented in
    this stage; variable-dimensional encoders are a later stage.
    """
    return bank.to_structured_matrix(blocks=STRUCTURAL_BLOCKS, chunk_size=chunk_size)


__all__ = [
    "RECORD_SCHEMA",
    "STRUCTURAL_BLOCKS",
    "BLOCK_SOURCES",
    "NeuronRecordError",
    "RecordBlock",
    "NeuronRecordBank",
    "build_neuron_record_bank",
    "structural_matrix_from_record_bank",
    "validate_record_array",
]
