"""Full neuron-vector composition: ``z_i = [z_structured_i, z_residual_i]``.

This module does **only** composition:

* it asks :class:`src.structured_vector.StructuredVectorEncoder` for the deterministic
  structured part,
* optionally asks a trained :class:`src.residual.ResidualResult` for the learned residual
  part,
* concatenates them along the feature axis and records where every coordinate came from.

It performs no training, no fitting, no evaluation and no normalization of its own; it does
not modify the geometry/prediction/control code, and it never touches PROBE or TEST data.

Invariant (checked, not assumed)::

    full_dimension == structured_d + learned_residual_d

With ``learned_residual_d = 0`` (``vector.residual.enabled = false``) the API reduces to the
existing deterministic structured vector exactly and instantiates no residual model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .neuron_record import NeuronRecordBank
from .residual import ResidualError, ResidualResult, build_residual_source, ResidualSourceConfig
from .structured_vector import (
    DEFAULT_PROJECTION_SEED,
    StructuredVectorEncoder,
    StructuredVectorError,
)
from .v2_config import V2Config

#: Provenance schema of a composed neuron-vector set.
SCHEMA = "neuron_vector/v1"

#: Schema of a *persisted* neuron-vector artifact (the frozen ``(n, d)`` matrix on disk).
ARTIFACT_SCHEMA = "neuron_vector_artifact/v1"


class NeuronVectorError(ValueError):
    """Raised for an invalid composition request."""


@dataclass
class NeuronVectors:
    """Composed vectors ``(n_neurons, d)`` with full coordinate provenance."""

    X: np.ndarray
    structured_names: tuple[str, ...]
    residual_names: tuple[str, ...]
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float64)
        if self.X.ndim != 2:
            raise NeuronVectorError(f"neuron vectors must be 2-D, got shape {self.X.shape}")
        self.structured_names = tuple(str(n) for n in self.structured_names)
        self.residual_names = tuple(str(n) for n in self.residual_names)
        if self.X.shape[1] != len(self.structured_names) + len(self.residual_names):
            raise NeuronVectorError(
                f"composition mismatch: X has {self.X.shape[1]} columns but "
                f"{len(self.structured_names)} structured + {len(self.residual_names)} residual names"
            )
        self.provenance = dict(self.provenance)
        self.provenance.setdefault("schema", SCHEMA)
        self.provenance.setdefault("uses_labels", False)
        self.provenance.setdefault("full_dimension", int(self.X.shape[1]))

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Final coordinate order: structured coordinates first, residual coordinates last."""
        return self.structured_names + self.residual_names

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.X.shape[0]), int(self.X.shape[1]))

    @property
    def n_neurons(self) -> int:
        return int(self.X.shape[0])

    @property
    def structured_d(self) -> int:
        return len(self.structured_names)

    @property
    def residual_d(self) -> int:
        return len(self.residual_names)

    @property
    def full_dimension(self) -> int:
        return int(self.X.shape[1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "structured_d": self.structured_d,
            "residual_d": self.residual_d,
            "feature_names": list(self.feature_names),
            "structured_names": list(self.structured_names),
            "residual_names": list(self.residual_names),
            "provenance": self.provenance,
        }


def build_neuron_vectors(
    bank: NeuronRecordBank,
    *,
    structured_d: int,
    residual: ResidualResult | None = None,
    enabled_blocks: Sequence[str] | str | None = None,
    projection_seed: int = DEFAULT_PROJECTION_SEED,
    chunk_size: int | None = None,
) -> NeuronVectors:
    """Compose ``[z_structured, z_residual]`` from a record bank.

    ``residual=None`` produces the structured vector only (``residual_d = 0``); no model is
    instantiated in that case and the result is bit-identical to the structured encoder
    output. When a trained :class:`~src.residual.ResidualResult` is passed, its feature
    schema is verified against this bank before use.
    """
    if not isinstance(bank, NeuronRecordBank):
        raise NeuronVectorError(f"composition requires a NeuronRecordBank, got {type(bank).__name__}")
    try:
        structured = StructuredVectorEncoder(
            bank,
            structured_d=int(structured_d),
            enabled_blocks=enabled_blocks,
            projection_seed=projection_seed,
            chunk_size=chunk_size,
        ).encode()
    except StructuredVectorError as exc:
        raise NeuronVectorError(str(exc)) from exc

    structured_names = tuple(structured.feature_names)
    residual_names: tuple[str, ...] = ()
    z_residual: np.ndarray | None = None
    residual_provenance: dict[str, Any] = {
        "used": False,
        "reason": "no residual artifact supplied (learned_residual_d = 0)",
        "feature_names": [],
        "residual_dim": 0,
        "uses_labels": False,
    }
    if residual is not None:
        if not isinstance(residual, ResidualResult):
            raise NeuronVectorError(
                f"residual must be a ResidualResult, got {type(residual).__name__}"
            )
        try:
            z_residual = residual.encode(bank)
        except ResidualError as exc:
            raise NeuronVectorError(str(exc)) from exc
        if residual.residual_dim != z_residual.shape[1]:  # pragma: no cover - defensive
            raise NeuronVectorError("internal error: residual artifact dimension mismatch")
        residual_names = tuple(f"residual[{j}]" for j in range(residual.residual_dim))
        residual_provenance = {
            "used": True,
            "source": "src.residual (self-supervised masked reconstruction, FIT only)",
            "schema": residual.provenance.get("schema"),
            "residual_dim": residual.residual_dim,
            "input_dim": residual.input_dim,
            "source_schema_hash": residual.schema_hash,
            "feature_names": list(residual_names),
            "normalization": residual.normalization.get("mode"),
            "mask": {
                "mask_fraction": residual.config.mask_fraction,
                "mask_seed": residual.config.mask_seed,
                "minimum_visible_features": residual.config.minimum_visible_features,
            },
            "seeds": residual.provenance.get("seeds"),
            "best_epoch": residual.best_epoch,
            "best_val_loss": residual.best_val_loss,
            "uses_labels": False,
            "residual_provenance": dict(residual.provenance),
        }

    X = structured.X if z_residual is None else np.concatenate([structured.X, z_residual], axis=1)
    full_dimension = int(X.shape[1])
    expected = int(structured_d) + (0 if residual is None else residual.residual_dim)
    if full_dimension != expected:
        raise NeuronVectorError(
            f"composition invariant violated: full_dimension={full_dimension} != "
            f"structured_d={structured_d} + learned_residual_d={0 if residual is None else residual.residual_dim}"
        )

    provenance = {
        "schema": SCHEMA,
        "uses_labels": False,
        "full_dimension": full_dimension,
        "structured_d": int(structured_d),
        "residual_d": 0 if residual is None else int(residual.residual_dim),
        "coordinate_order": (
            "structured coordinates first (level 0, then level 1, then deterministic projection), "
            "residual coordinates last (residual[0], residual[1], ...)"
        ),
        "structured": {
            "source": "src.structured_vector.StructuredVectorEncoder",
            "schema": structured.provenance.get("schema"),
            "level0_dimension": structured.provenance.get("level0_dimension"),
            "level1_dimension": structured.provenance.get("level1_dimension"),
            "source_dimension": structured.provenance.get("source_dimension"),
            "requested_blocks": structured.provenance.get("requested_blocks"),
            "present_blocks": structured.provenance.get("present_blocks"),
            "projection_used": structured.provenance.get("projection_used"),
            "projection_seed": structured.provenance.get("projection_seed"),
            "projection": structured.provenance.get("projection"),
            "feature_names": list(structured_names),
            "uses_labels": False,
        },
        "residual": residual_provenance,
        "bank": {
            "schema": bank.provenance.get("schema"),
            "n_neurons": bank.n_neurons,
            "uses_labels": False,
            "activity_split": bank.provenance.get("activity", {}).get("split"),
        },
    }
    return NeuronVectors(
        X=X,
        structured_names=structured_names,
        residual_names=residual_names,
        provenance=provenance,
    )


def neuron_vectors_from_config(
    config: V2Config,
    bank: NeuronRecordBank,
    *,
    residual: ResidualResult | None = None,
    chunk_size: int | None = None,
) -> NeuronVectors:
    """Compose using the V2 configuration's ``structured_d`` and ``learned_residual_d``.

    * ``learned_residual_d = 0`` -> structured vector only; passing a residual is an error
      (the configuration explicitly asks for no residual).
    * ``learned_residual_d > 0`` -> a trained residual artifact is required and its dimension
      must match exactly; nothing is trained or adapted here.
    """
    structured_d = int(config.vector.structured_d)
    residual_d = int(config.vector.learned_residual_d)
    if residual_d == 0:
        if residual is not None:
            raise NeuronVectorError(
                "the configuration requests no residual (vector.learned_residual_d=0, "
                "vector.residual.enabled=false) but a residual artifact was supplied"
            )
    else:
        if residual is None:
            raise NeuronVectorError(
                f"the configuration requests learned_residual_d={residual_d} "
                f"(vector.residual.enabled={config.vector.residual.enabled}) but no trained residual "
                "artifact was supplied; train one with src.residual.train_residual or set "
                "vector.learned_residual_d=0"
            )
        if int(residual.residual_dim) != residual_d:
            raise NeuronVectorError(
                f"the trained residual has residual_dim={residual.residual_dim} but the configuration "
                f"requests learned_residual_d={residual_d}"
            )

    if chunk_size is None:
        chunk_size = int(config.memory.representation_chunk_size)
    vectors = build_neuron_vectors(
        bank,
        structured_d=structured_d,
        residual=residual,
        enabled_blocks=config.vector.enabled_blocks,
        chunk_size=chunk_size,
    )
    vectors.provenance["configured"] = {
        "d": config.vector.d,
        "structured_d": structured_d,
        "learned_residual_d": residual_d,
        "residual_enabled": bool(config.vector.residual.enabled),
        "enabled_blocks": list(config.vector.enabled_blocks),
        "representation_chunk_size": int(config.memory.representation_chunk_size),
        "invariant_holds": int(config.vector.d) == structured_d + residual_d,
    }
    return vectors


def default_residual_source_config(residual: ResidualResult) -> ResidualSourceConfig:
    """The source-view configuration a trained residual expects (for reuse/debugging)."""
    return ResidualSourceConfig.from_mapping(residual.source_config)


# --------------------------------------------------------------------------
# Persisted artifact (the frozen representation on disk)
# --------------------------------------------------------------------------
def _matrix_digest(X: np.ndarray) -> str:
    """SHA-256 of the matrix bytes in a canonical (C-contiguous float64) layout."""
    canonical = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


@dataclass
class NeuronVectorArtifact:
    """A frozen ``(n_neurons, d)`` representation plus everything needed to verify it.

    This is the on-disk form of a built representation. It is **frozen**: it carries the
    resolved dimensions, the coordinate names, the run identity and the provenance of the
    build, so a later evaluation (or another process) can reuse exactly the same matrix and
    can *prove* it is the same one (schema + dimension + name + run-id + content hash).
    """

    X: np.ndarray
    feature_names: tuple[str, ...]
    structured_d: int
    residual_d: int
    run_id: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float64)
        if self.X.ndim != 2:
            raise NeuronVectorError(f"artifact matrix must be 2-D, got shape {self.X.shape}")
        self.feature_names = tuple(str(n) for n in self.feature_names)
        self.structured_d = int(self.structured_d)
        self.residual_d = int(self.residual_d)
        self.run_id = str(self.run_id)
        self.provenance = dict(self.provenance)
        if self.X.shape[1] != len(self.feature_names):
            raise NeuronVectorError(
                f"artifact has {self.X.shape[1]} columns but {len(self.feature_names)} coordinate names"
            )
        if self.X.shape[1] != self.structured_d + self.residual_d:
            raise NeuronVectorError(
                "artifact violates d = structured_d + residual_d: "
                f"{self.X.shape[1]} != {self.structured_d} + {self.residual_d}"
            )
        self.provenance.setdefault("schema", ARTIFACT_SCHEMA)
        self.provenance.setdefault("uses_labels", False)
        self.provenance.setdefault("full_dimension", int(self.X.shape[1]))

    @property
    def n_neurons(self) -> int:
        return int(self.X.shape[0])

    @property
    def d(self) -> int:
        return int(self.X.shape[1])

    @classmethod
    def from_vectors(
        cls,
        vectors: NeuronVectors,
        *,
        run_id: str = "",
        provenance: Mapping[str, Any] | None = None,
    ) -> "NeuronVectorArtifact":
        """Freeze composed vectors (the values are copied, not referenced)."""
        if not isinstance(vectors, NeuronVectors):
            raise NeuronVectorError(f"expected NeuronVectors, got {type(vectors).__name__}")
        prov = dict(vectors.provenance)
        if provenance:
            prov.update(dict(provenance))
        return cls(
            X=np.array(vectors.X, dtype=np.float64, copy=True),
            feature_names=vectors.feature_names,
            structured_d=vectors.structured_d,
            residual_d=vectors.residual_d,
            run_id=run_id,
            provenance=prov,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "schema": ARTIFACT_SCHEMA,
            "run_id": self.run_id,
            "shape": [self.n_neurons, self.d],
            "structured_d": self.structured_d,
            "residual_d": self.residual_d,
            "coordinate_order": "structured coordinates first, residual coordinates last",
            "matrix_sha256": _matrix_digest(self.X),
            "uses_labels": False,
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            X=np.asarray(self.X, dtype=np.float64),
            feature_names=np.asarray(self.feature_names, dtype=str),
            structured_d=np.asarray([self.structured_d], dtype=np.int64),
            residual_d=np.asarray([self.residual_d], dtype=np.int64),
            run_id=np.asarray([self.run_id], dtype=str),
            matrix_sha256=np.asarray([_matrix_digest(self.X)], dtype=str),
            provenance=np.asarray([json.dumps(self.provenance, sort_keys=True, default=str)], dtype=str),
            schema=np.asarray([ARTIFACT_SCHEMA], dtype=str),
        )
        return p

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_run_id: str | None = None,
        expected_feature_names: Sequence[str] | None = None,
        expected_d: int | None = None,
        expected_structured_d: int | None = None,
        expected_residual_d: int | None = None,
        verify_hash: bool = True,
    ) -> "NeuronVectorArtifact":
        """Load a frozen artifact, refusing anything that does not match what was asked for.

        Every check is explicit: schema, dimensions (``d``, ``structured_d``,
        ``residual_d``), coordinate names, run identity and (by default) the content
        hash. A mismatch raises :class:`NeuronVectorError`; nothing is adapted silently.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"neuron-vector artifact not found: {p}")
        with np.load(str(p), allow_pickle=False) as data:
            schema = str(data["schema"][0])
            if schema != ARTIFACT_SCHEMA:
                raise NeuronVectorError(
                    f"unsupported neuron-vector artifact schema {schema!r}; expected {ARTIFACT_SCHEMA!r}"
                )
            X = np.asarray(data["X"], dtype=np.float64)
            names = tuple(str(n) for n in data["feature_names"])
            structured_d = int(data["structured_d"][0])
            residual_d = int(data["residual_d"][0])
            run_id = str(data["run_id"][0])
            stored_hash = str(data["matrix_sha256"][0])
            provenance = json.loads(str(data["provenance"][0]))
        artifact = cls(
            X=X,
            feature_names=names,
            structured_d=structured_d,
            residual_d=residual_d,
            run_id=run_id,
            provenance=provenance,
        )
        if verify_hash and _matrix_digest(artifact.X) != stored_hash:
            raise NeuronVectorError(f"{p} failed its content-hash integrity check")
        if expected_run_id is not None and str(expected_run_id) != artifact.run_id:
            raise NeuronVectorError(
                f"artifact run_id={artifact.run_id!r} does not match the requested "
                f"run_id={expected_run_id!r}"
            )
        if expected_feature_names is not None and tuple(expected_feature_names) != artifact.feature_names:
            raise NeuronVectorError(
                "incompatible artifact: the requested coordinate names do not match the "
                f"artifact ({len(tuple(expected_feature_names))} vs {len(artifact.feature_names)} "
                "coordinates)"
            )
        for name, expected in (
            ("d", expected_d),
            ("structured_d", expected_structured_d),
            ("residual_d", expected_residual_d),
        ):
            if expected is None:
                continue
            actual = {
                "d": artifact.d,
                "structured_d": artifact.structured_d,
                "residual_d": artifact.residual_d,
            }[name]
            if int(expected) != int(actual):
                raise NeuronVectorError(
                    f"incompatible artifact: {name}={actual} but {int(expected)} was requested"
                )
        return artifact


def residual_source_for(bank: NeuronRecordBank, residual: ResidualResult) -> Any:
    """Build the exact source view a trained residual expects from ``bank`` (verifies nothing)."""
    return build_residual_source(bank, default_residual_source_config(residual))


__all__ = [
    "SCHEMA",
    "ARTIFACT_SCHEMA",
    "NeuronVectorError",
    "NeuronVectors",
    "NeuronVectorArtifact",
    "build_neuron_vectors",
    "neuron_vectors_from_config",
    "default_residual_source_config",
    "residual_source_for",
]