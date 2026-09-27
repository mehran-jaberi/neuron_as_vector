"""Label-free individual-stimulus functional-response source (FIT only).

Why this module exists
----------------------
The record bank already stores the per-sample spike counts of every hidden neuron on the
label-free **FIT** split::

    activity.samples  ->  (n_fit_samples, n_neurons)   float32

That matrix is small (canonical: 5922 x 256 = 6.1 MB) but the learned residual only ever
saw compressed summaries of it (level-0/level-1 statistics plus fixed projections of the
raw connectivity). This module turns the *stimulus-level response profile* into a
first-class, compact, deterministic source view so the residual can learn from it::

    samples (n_fit_samples, n_neurons)
        |
        v
    optional per-neuron normalisation over the sample axis        (raw | neuron_centered | neuron_zscored)
        |
        v
    fixed, seeded, data-independent Gaussian projection of the sample axis
        |
        v
    (n_neurons, functional_source_dim)     default functional_source_dim = 64

Design constraints (enforced, not assumed)
------------------------------------------
* **Label-free.** Only ``activity.samples`` (FIT spike counts) is read. No labels, no class
  PSTHs/rates, no speaker identity, no PROBE, no TEST, no model forward pass. The bank's
  activity split is validated to be FIT-only
  (:func:`src.neuron_record.assert_label_free_fit_activity`).
* **Deterministic and fixed.** The projection is not fitted to anything: column ``j`` is
  drawn from ``default_rng([projection_seed, j]).standard_normal(n_samples) / sqrt(n_samples)``,
  so it is prefix-stable (``dim=32`` columns are the first 32 columns of ``dim=64``) and
  independent of the data. The projection seed is a dedicated parameter
  (``functional_projection_seed``), never shared with the structured projection seed, the
  residual mask seed or the training seed.
* **Memory.** Neither ``(n_neurons, n_fit_samples)`` nor any time-resolved tensor is ever
  materialised: the statistics are accumulated and the projection is applied in
  **sample-axis chunks** (``(chunk, n_neurons)`` at a time, transposed per chunk only), and
  the only persistent arrays are the stored sample-major counts (owned by the bank), the
  ``(n_samples, source_dim)`` projection matrix and the ``(n_neurons, source_dim)`` result.
* **Separate from the PROBE target.** The normalisation helpers here are the *source-side*
  transform of the FIT response profile, deliberately independent of
  :mod:`src.functional_fingerprint` (which holds the PROBE evaluation targets): fusing the
  two would be a leakage hazard and would couple the source view to evaluation choices.

Normalisation order (explicit)
------------------------------
The per-neuron transform is applied to the **raw counts, over the sample axis**, *before*
the projection; the projection never sees labels and the transform is never re-applied
afterwards::

    raw            : nothing removed (preserves magnitude/rate information)
    neuron_centered: v - mean_s(v)                (removes the neuron's baseline rate)
    neuron_zscored : (v - mean_s(v)) / std_s(v)   (removes baseline and amplitude)

Zero-variance rule: a neuron whose per-sample response has ``std_s < 1e-8`` is **kept**
(never dropped, so the neuron set and ordering are unchanged); its centred profile is
exactly zero and therefore its projected coordinates are exactly zero.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .neuron_record import (
    NeuronRecordBank,
    assert_label_free_fit_activity,
)
from .v2_config import (
    DEFAULT_FUNCTIONAL_PROJECTION_SEED,
    DEFAULT_FUNCTIONAL_SOURCE_DIM,
    FUNCTIONAL_SOURCE_NORMALIZATIONS,
    MAX_FUNCTIONAL_SOURCE_DIM,
    V2Config,
)

#: Provenance schema of a functional-response source view.
FUNCTIONAL_RESPONSE_SCHEMA = "functional_response_source/v1"

#: Block the per-sample counts are read from (the record bank's activity block).
ACTIVITY_BLOCK = "activity"

#: Default number of samples per projection chunk.
DEFAULT_CHUNK_SIZE = 256

#: Per-neuron profile std below which the profile is treated as zero-variance.
ZERO_VARIANCE_STD_EPS = 1e-8

#: Sanity guard on the projection matrix size (``n_samples x source_dim`` float64).
MAX_PROJECTION_BYTES = 2**31


class FunctionalResponseError(ValueError):
    """Raised for an invalid functional-response source request."""


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class FunctionalResponseConfig:
    """Deterministic definition of the compact functional-response source view."""

    source_dim: int = DEFAULT_FUNCTIONAL_SOURCE_DIM
    projection_seed: int = DEFAULT_FUNCTIONAL_PROJECTION_SEED
    normalization: str = "raw"
    chunk_size: int = DEFAULT_CHUNK_SIZE

    def __post_init__(self) -> None:
        self.source_dim = _positive_int(self.source_dim, "source_dim", MAX_FUNCTIONAL_SOURCE_DIM)
        self.projection_seed = _non_negative_int(self.projection_seed, "projection_seed")
        self.normalization = str(self.normalization).strip().lower()
        if self.normalization not in FUNCTIONAL_SOURCE_NORMALIZATIONS:
            raise FunctionalResponseError(
                f"normalization must be one of {list(FUNCTIONAL_SOURCE_NORMALIZATIONS)}, "
                f"got {self.normalization!r}"
            )
        self.chunk_size = _positive_int(self.chunk_size, "chunk_size")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_dim": int(self.source_dim),
            "projection_seed": int(self.projection_seed),
            "normalization": self.normalization,
            "chunk_size": int(self.chunk_size),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "FunctionalResponseConfig":
        valid = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in dict(mapping or {}).items() if k in valid})

    @classmethod
    def from_v2_config(
        cls, config: V2Config, **overrides: Any
    ) -> "FunctionalResponseConfig":
        """Read ``vector.functional_source_*`` from the V2 configuration."""
        mapping: dict[str, Any] = {
            "source_dim": config.vector.functional_source_dim,
            "projection_seed": config.vector.functional_projection_seed,
            "normalization": config.vector.functional_source_normalization,
        }
        mapping.update(overrides)
        return cls.from_mapping(mapping)


# --------------------------------------------------------------------------
# Source container
# --------------------------------------------------------------------------
@dataclass
class FunctionalResponseSource:
    """``(n_neurons, source_dim)`` label-free functional-response view of a FIT split."""

    X: np.ndarray
    feature_names: tuple[str, ...]
    n_samples: int
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float64)
        if self.X.ndim != 2:
            raise FunctionalResponseError(
                f"the functional-response source must be 2-D (n_neurons, source_dim), got {self.X.shape}"
            )
        self.feature_names = tuple(str(n) for n in self.feature_names)
        if self.X.shape[1] != len(self.feature_names):
            raise FunctionalResponseError(
                f"source matrix has {self.X.shape[1]} columns but {len(self.feature_names)} feature names"
            )
        if not np.isfinite(self.X).all():
            raise FunctionalResponseError("the functional-response source contains non-finite entries")
        self.n_samples = int(self.n_samples)
        if self.n_samples < 1:
            raise FunctionalResponseError(f"n_samples must be >= 1, got {self.n_samples}")
        self.provenance = dict(self.provenance)
        self.provenance.setdefault("schema", FUNCTIONAL_RESPONSE_SCHEMA)
        self.provenance.setdefault("source_type", "individual_stimulus_response")
        self.provenance.setdefault("uses_labels", False)
        self.provenance.setdefault("source_split", "fit")
        self.provenance.setdefault("n_neurons", int(self.X.shape[0]))
        self.provenance.setdefault("output_dimension", int(self.X.shape[1]))

    @property
    def n_neurons(self) -> int:
        return int(self.X.shape[0])

    @property
    def source_dim(self) -> int:
        return int(self.X.shape[1])

    @property
    def sample_order_hash(self) -> str:
        return str(self.provenance.get("sample_order", {}).get("sample_order_hash", ""))

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_neurons": self.n_neurons,
            "source_dim": self.source_dim,
            "n_samples": self.n_samples,
            "feature_names": list(self.feature_names),
            "sample_order_hash": self.sample_order_hash,
            "provenance": self.provenance,
        }


# --------------------------------------------------------------------------
# Reads and deterministic helpers
# --------------------------------------------------------------------------
def response_counts_from_bank(
    bank: NeuronRecordBank, *, block: str = ACTIVITY_BLOCK
) -> tuple[np.ndarray, str]:
    """Return ``(samples, split)``: the stored ``(n_fit_samples, n_neurons)`` FIT counts.

    The counts are the first-class per-stimulus response data; they are **never**
    transposed in full here. Rejects a bank with no stored per-sample counts and a bank
    whose activity split is not FIT.
    """
    if not isinstance(bank, NeuronRecordBank):
        raise FunctionalResponseError(
            f"the functional-response source requires a NeuronRecordBank, got {type(bank).__name__}"
        )
    split = assert_label_free_fit_activity(bank)
    try:
        activity = bank.get_block(block)
    except Exception as exc:  # NeuronRecordError: implemented but absent in this bank
        raise FunctionalResponseError(
            f"the functional-response source needs the {block!r} block with stored per-sample "
            f"counts; this bank has none ({exc})"
        ) from exc
    if activity.samples is None:
        raise FunctionalResponseError(
            f"the {block!r} block was built with store_sample_counts=False, so the per-stimulus "
            "response profile is unavailable; rebuild the bank with store_sample_counts=True"
        )
    samples = np.asarray(activity.samples)
    if samples.ndim != 2 or samples.shape[1] != bank.n_neurons:
        raise FunctionalResponseError(
            f"stored samples must be (n_samples, n_neurons) = (n_samples, {bank.n_neurons}), "
            f"got {samples.shape}"
        )
    return samples, split


def sample_order_hash(samples: np.ndarray) -> str:
    """Stable hash of the stored sample order and values (``sha256`` of shape|dtype|bytes).

    The per-stimulus source depends on the order of the FIT utterances, so the exact order
    used is recorded with the source. The counts are stored sample-major by the record bank,
    so this hash is invariant to nothing else: it changes if the split, the ordering or any
    count changes. No labels or speaker identity participate.
    """
    arr = np.asarray(samples)
    if arr.ndim != 2:
        raise FunctionalResponseError(f"sample_order_hash expects a 2-D matrix, got {arr.shape}")
    contiguous = np.ascontiguousarray(arr)
    digest = hashlib.sha256()
    digest.update(f"{arr.shape[0]}x{arr.shape[1]}:{arr.dtype.str}:".encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def response_statistics(
    samples: np.ndarray,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-neuron ``(mean, std, zero_variance_mask)`` over the sample axis, chunked.

    Accumulated in float64 over sample chunks so the ``(n_neurons, n_samples)`` transpose is
    never materialised. ``std`` uses the population convention (``ddof=0``); the variance is
    clamped at 0 against floating-point cancellation.
    """
    arr = np.asarray(samples, dtype=np.float64)
    n_samples, n_neurons = arr.shape
    size = _positive_int(chunk_size, "chunk_size")
    total = np.zeros(n_neurons, dtype=np.float64)
    total_sq = np.zeros(n_neurons, dtype=np.float64)
    for start in range(0, n_samples, size):
        block = arr[start:start + size]
        total += block.sum(axis=0)
        total_sq += np.square(block).sum(axis=0)
    mean = total / float(n_samples)
    variance = np.maximum(total_sq / float(n_samples) - np.square(mean), 0.0)
    std = np.sqrt(variance)
    zero_variance = std < ZERO_VARIANCE_STD_EPS
    return mean, std, zero_variance


def projection_matrix(
    n_samples: int,
    source_dim: int,
    *,
    seed: int = DEFAULT_FUNCTIONAL_PROJECTION_SEED,
) -> np.ndarray:
    """Fixed, data-independent Gaussian projection of the sample axis: ``(n_samples, dim)``.

    Column ``j`` is ``default_rng([seed, j]).standard_normal(n_samples) / sqrt(n_samples)``,
    so the matrix is deterministic, prefix-stable in ``source_dim`` and never fitted to the
    data (it does not depend on the counts at all).
    """
    n_samples = _positive_int(n_samples, "n_samples")
    source_dim = _positive_int(source_dim, "source_dim", MAX_FUNCTIONAL_SOURCE_DIM)
    seed = _non_negative_int(seed, "projection_seed")
    n_bytes = n_samples * source_dim * 8
    if n_bytes > MAX_PROJECTION_BYTES:
        raise FunctionalResponseError(
            f"the projection matrix would need {n_bytes / 2**20:.0f} MiB "
            f"({n_samples} x {source_dim} float64); lower functional_source_dim or chunk the "
            "sample axis further"
        )
    scale = 1.0 / np.sqrt(float(n_samples))
    matrix = np.empty((n_samples, source_dim), dtype=np.float64)
    for column in range(source_dim):
        rng = np.random.default_rng([int(seed), int(column)])
        matrix[:, column] = rng.standard_normal(n_samples) * scale
    return matrix


def project_response_profile(
    samples: np.ndarray,
    config: FunctionalResponseConfig | None = None,
) -> np.ndarray:
    """Project a ``(n_samples, n_neurons)`` FIT response profile to ``(n_neurons, dim)``.

    The per-neuron normalisation (when requested) is applied in the same chunked pass as the
    projection, so peak memory is ``chunk x n_neurons`` plus the projection matrix; the
    ``(n_neurons, n_samples)`` transpose is never materialised in full.
    """
    config = config or FunctionalResponseConfig()
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 2:
        raise FunctionalResponseError(
            f"the response profile must be (n_samples, n_neurons), got {arr.shape}"
        )
    n_samples, n_neurons = arr.shape
    if n_samples < 1 or n_neurons < 1:
        raise FunctionalResponseError(f"empty response profile: {arr.shape}")

    mean, std, zero_variance = response_statistics(arr, chunk_size=config.chunk_size)
    if config.normalization == "raw":
        centre = np.zeros(n_neurons, dtype=np.float64)
        scale = np.ones(n_neurons, dtype=np.float64)
    elif config.normalization == "neuron_centered":
        centre = mean
        scale = np.ones(n_neurons, dtype=np.float64)
    else:  # neuron_zscored
        centre = mean
        scale = np.where(zero_variance, 1.0, std)

    projection = projection_matrix(n_samples, config.source_dim, seed=config.projection_seed)
    out = np.zeros((n_neurons, config.source_dim), dtype=np.float64)
    size = config.chunk_size
    for start in range(0, n_samples, size):
        stop = min(start + size, n_samples)
        block = (arr[start:stop] - centre) / scale  # (chunk, n_neurons)
        # transpose only this chunk: (n_neurons, chunk) @ (chunk, dim) -> (n_neurons, dim)
        out += block.T @ projection[start:stop]
    # a zero-variance neuron has an exactly constant profile: its centred deviations are 0
    if zero_variance.any() and config.normalization != "raw":
        out[zero_variance] = 0.0
    out = np.ascontiguousarray(out)
    if not np.isfinite(out).all():
        raise FunctionalResponseError("the projected source contains non-finite entries")
    return out


def build_functional_response_source(
    bank: NeuronRecordBank,
    config: FunctionalResponseConfig | None = None,
    *,
    block: str = ACTIVITY_BLOCK,
) -> FunctionalResponseSource:
    """Build the compact label-free functional-response source view of a record bank."""
    config = config or FunctionalResponseConfig()
    samples, split = response_counts_from_bank(bank, block=block)
    mean, std, zero_variance = response_statistics(samples, chunk_size=config.chunk_size)
    X = project_response_profile(samples, config)
    feature_names = tuple(f"functional_response.proj_{j:02d}" for j in range(X.shape[1]))
    provenance = {
        "schema": FUNCTIONAL_RESPONSE_SCHEMA,
        "source_type": "individual_stimulus_response",
        "source_split": split,
        "uses_labels": False,
        "fit_only": True,
        "definition": (
            "per-neuron response profile over individual FIT utterances "
            "(activity.samples[s, i] spike count of neuron i on FIT utterance s), then an "
            "explicit per-neuron transform over the sample axis, then a fixed seeded "
            "projection of the sample axis"
        ),
        "input_shape": [int(samples.shape[0]), int(samples.shape[1])],
        "n_samples": int(samples.shape[0]),
        "n_neurons": int(samples.shape[1]),
        "output_dimension": int(X.shape[1]),
        "feature_names": list(feature_names),
        "normalization": {
            "mode": config.normalization,
            "axis": "sample axis, per neuron (mean_s / std_s)",
            "order": "normalize the response profile first, then project (never re-applied)",
            "mean_over_samples": [float(v) for v in mean],
            "std_over_samples": [float(v) for v in std],
            "zero_variance_neurons": int(zero_variance.sum()),
            "zero_variance_rule": (
                "std_s < 1e-8: the neuron is kept and its centred/standardised profile is "
                "exactly zero, so its projected coordinates are exactly zero"
            ),
            "units_before_normalization": "spike counts per FIT utterance",
        },
        "projection": {
            "type": "fixed_gaussian_1_over_sqrt_n_samples",
            "seed": int(config.projection_seed),
            "n_samples": int(samples.shape[0]),
            "output_dimension": int(X.shape[1]),
            "generator": (
                "column j = numpy.random.default_rng([projection_seed, j])"
                ".standard_normal(n_samples) / sqrt(n_samples)"
            ),
            "prefix_stable": True,
            "data_dependent": False,
            "learned": False,
            "fitted": False,
            "matrix_bytes": int(samples.shape[0] * X.shape[1] * 8),
        },
        "sample_order": {
            "stored_order_preserved": True,
            "sample_order_hash": sample_order_hash(samples),
            "hash_method": "sha256(shape|dtype|sample-major bytes of activity.samples)",
            "derived_from": (
                "record-bank activity.samples (FIT split; order fixed by the deterministic "
                "split seed, never by labels or speaker identity)"
            ),
            "labels_used_for_ordering": False,
            "speaker_identity_used": False,
        },
        "chunk_size": int(config.chunk_size),
        "source_config": config.to_dict(),
        "bank": {
            "schema": bank.provenance.get("schema"),
            "n_neurons": bank.n_neurons,
            "uses_labels": False,
            "activity_split": bank.provenance.get("activity", {}).get("split"),
        },
        "note": (
            "Compact deterministic source view of the label-free FIT individual-stimulus "
            "responses. No labels, no class PSTH/rate, no speaker identity, no PROBE, no TEST, "
            "no model forward pass; the projection is fixed and data-independent."
        ),
    }
    return FunctionalResponseSource(
        X=X, feature_names=feature_names, n_samples=int(samples.shape[0]), provenance=provenance
    )


# --------------------------------------------------------------------------
# Small validated converters (mirrors the repository's config coercion style)
# --------------------------------------------------------------------------
def _positive_int(value: Any, name: str, maximum: int | None = None) -> int:
    number = _as_int(value, name)
    if number < 1:
        raise FunctionalResponseError(f"{name} must be >= 1, got {number}")
    if maximum is not None and number > maximum:
        raise FunctionalResponseError(f"{name}={number} exceeds the maximum {maximum}")
    return number


def _non_negative_int(value: Any, name: str) -> int:
    number = _as_int(value, name)
    if number < 0:
        raise FunctionalResponseError(f"{name} must be >= 0, got {number}")
    return number


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise FunctionalResponseError(f"{name} must be an integer, got {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise FunctionalResponseError(f"{name} must be an integer, got {value!r}") from exc


__all__ = [
    "FUNCTIONAL_RESPONSE_SCHEMA",
    "ACTIVITY_BLOCK",
    "DEFAULT_CHUNK_SIZE",
    "ZERO_VARIANCE_STD_EPS",
    "MAX_PROJECTION_BYTES",
    "FunctionalResponseError",
    "FunctionalResponseConfig",
    "FunctionalResponseSource",
    "response_counts_from_bank",
    "sample_order_hash",
    "response_statistics",
    "projection_matrix",
    "project_response_profile",
    "build_functional_response_source",
]
