"""Deterministic variable-dimensional structured encoder over a ``NeuronRecordBank``.

Architecture position
---------------------
::

    model/checkpoint + label-free FIT
                |
                v
        NeuronRecordBank          <- source information (src/neuron_record.py)
                |
                v
     StructuredVectorEncoder      <- THIS MODULE: deterministic representation
                |
                v
        z_structured in R^d_struct
                |
                +--- later --->  learned residual (NOT implemented)

The encoder is **entirely deterministic and has no fitted state**: the same record
bank, blocks and dimension always produce exactly the same matrix. Nothing here is
trained, nothing is fitted to data, and no class label, PROBE or TEST information is
ever read (the record bank is label-free by construction and remains the only input).

Dimension policy
----------------
The coordinate order is canonical and fixed:

1. **Level 0** - the record bank's existing deterministic summary features, in
   canonical block order and alphabetical within each block (exactly the ordering of
   :meth:`NeuronRecordBank.to_structured_matrix`). For the default block selection on
   a trained model this is the historical 48-D structural representation.
2. **Level 1** - deterministic within-block detail computed from the raw vectors that
   the record bank already stores (signed/absolute quantiles, an absolute-value mass
   histogram, top-k signed weights, threshold shares - see
   :data:`WEIGHT_DETAIL_NAMES`), plus per-sample count distribution summaries for the
   activity block when per-sample counts are stored (:data:`ACTIVITY_DETAIL_NAMES`).
   Every coordinate has an explicit name and definition; nothing is fabricated for
   blocks that are not implemented.
3. **Level 2** - a **fixed, seeded, data-independent** Gaussian projection used only
   to fill coordinates beyond the interpretable source dimension
   (:data:`DEFAULT_PROJECTION_SEED` and the ``projection_seed`` argument). It is a
   deterministic transformation, not a learned representation, and it operates on the
   ``(n_neurons, source_features)`` matrix - never on activity traces.

Selection is a **prefix** of that canonical order: ``structured_d <= level0`` takes the
first Level-0 coordinates; ``level0 < structured_d <= source`` appends Level-1 detail;
``structured_d > source`` additionally appends projected coordinates. The prefix
property is tested: ``encode(d1)[:, :d2] == encode(d2)`` for ``d2 <= d1``.

Interpretability caveat: Level-0/Level-1 coordinates are individually interpretable;
projected coordinates are mixtures of the raw (unstandardised) source features and are
therefore *not* individually interpretable. Their provenance records the seed and the
source/output dimensions. Downstream standardisation (e.g. by
:class:`src.representations.RepresentationSpace`) is a separate, later choice.

Memory
------
Every array is CPU-side and 2-D at most: ``(n_neurons, source_features)``,
``(n_neurons, structured_d)`` and the ``(source_features, projection_dim)`` projection
matrix. Chunked construction (``chunk_size``, e.g.
``memory.representation_chunk_size``) bounds the temporaries: source coordinates are
bit-identical between chunked and unchunked construction, and projected coordinates agree
to floating-point tolerance (different matrix shapes can make BLAS reorder summations).
The forbidden tensors ``(n_neurons, n_samples, d)``, ``(n_neurons, n_time, d)``,
``(n_neurons, n_neurons, d)`` and ``(n_neurons, n_samples, n_time)`` are never created.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .neuron_record import NeuronRecordBank
from .v2_config import DEFAULT_ENABLED_BLOCKS, IMPLEMENTED_RECORD_BLOCKS, V2_RECORD_BLOCKS, V2Config

#: Provenance schema identifier.
SCHEMA = "structured_vector/v1"

#: Default seed of the fixed projection (the only stochastic element, and it is data-independent).
DEFAULT_PROJECTION_SEED = 0

#: Upper bound on ``structured_d`` accepted by this stage (sanity guard for the projection matrix).
MAX_STRUCTURED_D = 16384

#: Signed / absolute quantile levels used by the Level-1 weight detail.
SIGNED_QUANTILE_LEVELS: tuple[float, ...] = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)

#: Number of equal-width bins for the Level-1 absolute-weight mass histogram.
ABS_HISTOGRAM_BINS = 8

#: Number of largest-|w| signed weights kept per neuron by the Level-1 detail.
TOP_SIGNED_COUNT = 3

#: Quantile levels of the per-sample count distribution used by the Level-1 activity detail.
ACTIVITY_COUNT_QUANTILES: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)

#: Minimum width of a weight vector for the Level-1 detail (top-k needs enough entries).
MIN_WEIGHT_WIDTH = max(TOP_SIGNED_COUNT, 1)


class StructuredVectorError(ValueError):
    """Raised for an invalid encoder request or an unsupported/unknown block."""


# --------------------------------------------------------------------------
# Level-1 detail definitions (names are part of the API: they appear in provenance)
# --------------------------------------------------------------------------
def _qname(prefix: str, q: float) -> str:
    return f"{prefix}{int(round(q * 100)):02d}"


#: Ordered Level-1 detail names for any block that stores weight vectors.
WEIGHT_DETAIL_NAMES: tuple[str, ...] = (
    tuple(_qname("q", q) for q in SIGNED_QUANTILE_LEVELS)
    + tuple(_qname("abs_q", q) for q in SIGNED_QUANTILE_LEVELS)
    + tuple(f"abs_hist_{b:02d}" for b in range(ABS_HISTOGRAM_BINS))
    + tuple(f"top_abs_{k}" for k in range(1, TOP_SIGNED_COUNT + 1))
    + ("share_gt_mean_abs", "share_gt_2mean_abs")
)

#: Ordered Level-1 detail names for the activity block (requires stored per-sample counts).
ACTIVITY_DETAIL_NAMES: tuple[str, ...] = (
    tuple(_qname("count_q", q) for q in ACTIVITY_COUNT_QUANTILES)
    + ("frac_zero_samples", "max_count", "mean_nonzero_count")
)

#: Human-readable definitions, recorded in provenance so no coordinate is anonymous.
DETAIL_DEFINITIONS: dict[str, str] = {
    **{name: f"signed quantile {q:g} of the stored weight vector" for name, q in
       zip((_qname("q", q) for q in SIGNED_QUANTILE_LEVELS), SIGNED_QUANTILE_LEVELS)},
    **{name: f"quantile {q:g} of |weight| for the stored weight vector" for name, q in
       zip((_qname("abs_q", q) for q in SIGNED_QUANTILE_LEVELS), SIGNED_QUANTILE_LEVELS)},
    **{f"abs_hist_{b:02d}": (
        f"fraction of the total |weight| mass in equal-width bin {b} of "
        f"{ABS_HISTOGRAM_BINS} bins spanning [0, max|weight|]"
    ) for b in range(ABS_HISTOGRAM_BINS)},
    **{f"top_abs_{k}": f"signed value of the k-th largest |weight| (k={k})" for k in range(1, TOP_SIGNED_COUNT + 1)},
    "share_gt_mean_abs": "fraction of weights with |w| strictly above mean(|w|)",
    "share_gt_2mean_abs": "fraction of weights with |w| strictly above 2*mean(|w|)",
    **{name: f"quantile {q:g} of the neuron's per-sample spike counts (FIT)"
       for name, q in zip((_qname("count_q", q) for q in ACTIVITY_COUNT_QUANTILES), ACTIVITY_COUNT_QUANTILES)},
    "frac_zero_samples": "fraction of FIT samples in which the neuron emitted no spike",
    "max_count": "largest per-sample spike count on FIT",
    "mean_nonzero_count": "mean per-sample spike count over the FIT samples with at least one spike",
}


def _weight_detail_chunk(weights: np.ndarray) -> np.ndarray:
    """Ordered Level-1 detail columns for a chunk of weight vectors (``(c, k) -> (c, 31)``)."""
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 2:
        raise StructuredVectorError(f"weight detail expects a 2-D (neurons, width) chunk, got {w.shape}")
    if w.shape[1] < MIN_WEIGHT_WIDTH:
        raise StructuredVectorError(
            f"weight detail needs at least {MIN_WEIGHT_WIDTH} weights per neuron, got width {w.shape[1]}"
        )
    abs_w = np.abs(w)
    columns: list[np.ndarray] = [np.quantile(w, q, axis=1) for q in SIGNED_QUANTILE_LEVELS]
    columns += [np.quantile(abs_w, q, axis=1) for q in SIGNED_QUANTILE_LEVELS]

    # absolute-value mass histogram (equal-width bins over [0, max|w|] per neuron)
    mx = abs_w.max(axis=1, keepdims=True)
    total = abs_w.sum(axis=1, keepdims=True)
    safe_mx = np.where(mx > 0.0, mx, 1.0)
    position = abs_w / safe_mx  # in [0, 1]; all-zero rows give 0
    mass = np.zeros((w.shape[0], ABS_HISTOGRAM_BINS), dtype=np.float64)
    for b in range(ABS_HISTOGRAM_BINS):
        lo = b / ABS_HISTOGRAM_BINS
        hi = (b + 1) / ABS_HISTOGRAM_BINS
        selected = (position >= lo) & (position < hi)
        if b == ABS_HISTOGRAM_BINS - 1:  # include position == 1 in the last bin
            selected = selected | (position >= hi)
        mass[:, b] = (abs_w * selected).sum(axis=1)
    fraction = mass / np.where(total > 0.0, total, 1.0)
    columns += [fraction[:, b] for b in range(ABS_HISTOGRAM_BINS)]

    # top-k signed weights (stable sort -> deterministic under ties)
    order = np.argsort(-abs_w, axis=1, kind="stable")[:, :TOP_SIGNED_COUNT]
    top = np.take_along_axis(w, order, axis=1)
    columns += [top[:, k] for k in range(TOP_SIGNED_COUNT)]

    mean_abs = abs_w.mean(axis=1, keepdims=True)
    columns.append((abs_w > mean_abs).mean(axis=1))
    columns.append((abs_w > 2.0 * mean_abs).mean(axis=1))
    return np.column_stack(columns)


def _activity_detail_chunk(samples: np.ndarray) -> np.ndarray:
    """Ordered Level-1 detail columns for per-sample counts (``(S, c) -> (c, 8)``)."""
    s = np.asarray(samples, dtype=np.float64)
    if s.ndim != 2:
        raise StructuredVectorError(f"activity detail expects a 2-D (samples, neurons) chunk, got {s.shape}")
    columns: list[np.ndarray] = [np.quantile(s, q, axis=0) for q in ACTIVITY_COUNT_QUANTILES]
    columns.append((s <= 0).mean(axis=0))
    columns.append(s.max(axis=0))
    nonzero = (s > 0).sum(axis=0)
    sums = s.sum(axis=0)
    columns.append(np.divide(sums, nonzero, out=np.zeros_like(sums), where=nonzero > 0))
    return np.column_stack(columns)


def _detail_names_for_block(block: Any) -> tuple[str, ...]:
    """Level-1 detail names a block can contribute (empty if it has no raw vectors)."""
    names: list[str] = []
    if block.weights is not None:
        names.extend(WEIGHT_DETAIL_NAMES)
    if block.name == "activity" and block.samples is not None:
        names.extend(ACTIVITY_DETAIL_NAMES)
    return tuple(names)


# --------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Coordinate:
    """Machine-readable provenance of one output coordinate."""

    index: int
    name: str
    kind: str  # "level0" | "level1" | "projection"
    block: str
    source: str
    definition: str
    uses_labels: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EncoderPlan:
    """Deterministic coordinate plan derived from a record bank (no data fitting)."""

    structured_d: int
    requested_blocks: tuple[str, ...]
    present_blocks: tuple[str, ...]
    level0_layout: tuple[tuple[str, str], ...]
    level1_layout: tuple[tuple[str, str], ...]
    projection: dict[str, Any] | None
    coordinates: tuple[Coordinate, ...]
    detail_definitions: dict[str, tuple[str, ...]]

    @property
    def level0_dimension(self) -> int:
        return len(self.level0_layout)

    @property
    def level1_dimension(self) -> int:
        return len(self.level1_layout)

    @property
    def source_dimension(self) -> int:
        return self.level0_dimension + self.level1_dimension

    @property
    def output_dimension(self) -> int:
        return self.structured_d

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.coordinates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "structured_d": self.structured_d,
            "requested_blocks": list(self.requested_blocks),
            "present_blocks": list(self.present_blocks),
            "level0_dimension": self.level0_dimension,
            "level1_dimension": self.level1_dimension,
            "source_dimension": self.source_dimension,
            "projection": dict(self.projection) if self.projection else {"used": False},
            "coordinates": [c.to_dict() for c in self.coordinates],
            "detail_definitions": {b: list(names) for b, names in self.detail_definitions.items()},
        }


# --------------------------------------------------------------------------
# Output container
# --------------------------------------------------------------------------
@dataclass
class StructuredVectors:
    """Deterministic structured vectors plus their per-coordinate provenance."""

    X: np.ndarray
    plan: EncoderPlan
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float64)
        if self.X.ndim != 2:
            raise StructuredVectorError(f"structured matrix must be 2-D, got shape {self.X.shape}")
        if self.X.shape[1] != self.plan.structured_d:
            raise StructuredVectorError(
                f"structured matrix has {self.X.shape[1]} columns but structured_d={self.plan.structured_d}"
            )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.plan.feature_names

    @property
    def coordinates(self) -> tuple[Coordinate, ...]:
        return self.plan.coordinates

    @property
    def metadata(self) -> list[dict[str, Any]]:
        """Per-coordinate machine-readable metadata (list of dicts)."""
        return [c.to_dict() for c in self.plan.coordinates]

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.X.shape[0]), int(self.X.shape[1]))

    @property
    def n_neurons(self) -> int:
        return int(self.X.shape[0])

    @property
    def structured_d(self) -> int:
        return int(self.X.shape[1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "feature_names": list(self.feature_names),
            "metadata": self.metadata,
            "provenance": self.provenance,
        }


# --------------------------------------------------------------------------
# Encoder
# --------------------------------------------------------------------------
@dataclass
class StructuredVectorEncoder:
    """Deterministic encoder: ``NeuronRecordBank -> (n_neurons, structured_d)``.

    Parameters
    ----------
    bank:
        The label-free source of information (:class:`src.neuron_record.NeuronRecordBank`).
    structured_d:
        Requested output dimension. Any value in ``[1, MAX_STRUCTURED_D]`` is supported.
    enabled_blocks:
        Which record blocks may contribute. ``None`` means the canonical default block
        set (:data:`src.v2_config.DEFAULT_ENABLED_BLOCKS`), which reproduces the historical
        48-D structural representation. Blocks that are declared but not implemented
        (``temporal``, ``network_context``) raise, and implemented-but-absent blocks
        (e.g. ``intrinsic`` for an untrained model) simply contribute nothing.
    projection_seed:
        Seed of the fixed Level-2 projection (used only when ``structured_d`` exceeds the
        interpretable source dimension).
    chunk_size:
        Optional number of neurons per chunk for :meth:`encode`; results are identical to
        the unchunked construction.
    warnings, configured:
        Optional provenance carried from :meth:`from_config`.
    """

    bank: NeuronRecordBank
    structured_d: int
    enabled_blocks: Sequence[str] | str | None = None
    projection_seed: int = DEFAULT_PROJECTION_SEED
    chunk_size: int | None = None
    warnings: list[str] = field(default_factory=list)
    configured: Mapping[str, Any] | None = None
    plan: EncoderPlan = field(init=False)
    _projection: np.ndarray | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.bank, NeuronRecordBank):
            raise StructuredVectorError(
                f"encoder input must be a NeuronRecordBank, got {type(self.bank).__name__}"
            )
        self.structured_d = _as_positive_int(self.structured_d, "structured_d", MAX_STRUCTURED_D)
        self.projection_seed = _as_positive_int(self.projection_seed, "projection_seed", 2**31 - 1, allow_zero=True)
        if self.chunk_size is not None:
            self.chunk_size = _as_positive_int(self.chunk_size, "chunk_size", 2**31 - 1)
        self.warnings = list(self.warnings)
        self.configured = dict(self.configured) if self.configured else None
        self.plan = self._build_plan()

    # -- plan construction ---------------------------------------------------
    def _resolve_blocks(self) -> tuple[list[str], list[str]]:
        if self.enabled_blocks is None:
            requested = list(DEFAULT_ENABLED_BLOCKS)
        elif isinstance(self.enabled_blocks, str):
            requested = [part.strip() for part in self.enabled_blocks.replace(",", " ").split() if part.strip()]
        else:
            requested = [str(b) for b in self.enabled_blocks]
        if not requested:
            raise StructuredVectorError("enabled_blocks must select at least one block")

        unknown = sorted({b for b in requested if b not in self.bank.declared_blocks})
        if unknown:
            raise StructuredVectorError(
                f"unknown block(s) {unknown}; valid block names are {list(self.bank.declared_blocks)}"
            )
        unimplemented = sorted({b for b in requested if b in self.bank.unimplemented_blocks})
        if unimplemented:
            raise StructuredVectorError(
                f"block(s) {unimplemented} are declared in the V2 schema but not implemented in this "
                f"stage; no placeholder features are generated"
            )
        # canonical order, de-duplicated
        requested = [b for b in self.bank.declared_blocks if b in set(requested)]
        present = [b for b in requested if b in self.bank.blocks]
        return requested, present

    def _build_plan(self) -> EncoderPlan:
        requested, present = self._resolve_blocks()

        level0_layout: list[tuple[str, str]] = []
        for bname in present:
            block = self.bank.get_block(bname)
            level0_layout.extend((bname, fname) for fname in block.feature_names)

        level1_layout: list[tuple[str, str]] = []
        detail_definitions: dict[str, tuple[str, ...]] = {}
        for bname in present:
            block = self.bank.get_block(bname)
            names = _detail_names_for_block(block)
            if names:
                detail_definitions[bname] = names
                level1_layout.extend((bname, name) for name in names)

        level0_dim = len(level0_layout)
        source_dim = level0_dim + len(level1_layout)
        if source_dim == 0:
            raise StructuredVectorError(
                f"the enabled block(s) {requested} provide no features in this record bank; "
                f"nothing can be encoded"
            )

        projection_dim = max(0, self.structured_d - source_dim)
        projection: dict[str, Any] | None = None
        if projection_dim > 0:
            projection = {
                "type": "fixed_gaussian",
                "seed": int(self.projection_seed),
                "source_dimension": int(source_dim),
                "output_dimension": int(projection_dim),
                "scaling": "1/sqrt(source_dimension)",
                "generator": (
                    "column j = numpy.random.default_rng([seed, j]).standard_normal(source_dimension)"
                    " / sqrt(source_dimension)  (per-column seeding -> prefix-stable)"
                ),
                "interpretable": False,
                "learned": False,
            }

        coordinates = self._coordinates(level0_layout, level1_layout, projection)
        if len(coordinates) != self.structured_d:  # pragma: no cover - defensive
            raise StructuredVectorError(
                f"internal plan error: produced {len(coordinates)} coordinates for structured_d="
                f"{self.structured_d}"
            )
        return EncoderPlan(
            structured_d=int(self.structured_d),
            requested_blocks=tuple(requested),
            present_blocks=tuple(present),
            level0_layout=tuple(level0_layout),
            level1_layout=tuple(level1_layout),
            projection=projection,
            coordinates=tuple(coordinates),
            detail_definitions=detail_definitions,
        )

    def _coordinates(
        self,
        level0_layout: Sequence[tuple[str, str]],
        level1_layout: Sequence[tuple[str, str]],
        projection: Mapping[str, Any] | None,
    ) -> list[Coordinate]:
        coords: list[Coordinate] = []
        for block, feature in level0_layout:
            if len(coords) >= self.structured_d:
                break
            name = f"{block}.{feature}"
            coords.append(
                Coordinate(
                    index=len(coords),
                    name=name,
                    kind="level0",
                    block=block,
                    source=name,
                    definition=f"existing deterministic summary feature of block {block!r}",
                )
            )
        for block, detail in level1_layout:
            if len(coords) >= self.structured_d:
                break
            name = f"{block}.{detail}"
            coords.append(
                Coordinate(
                    index=len(coords),
                    name=name,
                    kind="level1",
                    block=block,
                    source=f"{block}.weights" if detail in WEIGHT_DETAIL_NAMES else f"{block}.samples",
                    definition=DETAIL_DEFINITIONS.get(detail, detail),
                )
            )
        if projection is not None:
            for j in range(int(projection["output_dimension"])):
                coords.append(
                    Coordinate(
                        index=len(coords),
                        name=f"projection[{j}]",
                        kind="projection",
                        block="projection",
                        source=(
                            f"level0+level1 ({projection['source_dimension']} source features) "
                            f"-> fixed gaussian projection (seed={projection['seed']})"
                        ),
                        definition=(
                            "fixed seeded data-independent Gaussian projection of the source feature "
                            "matrix; a mixture, not an individually interpretable quantity"
                        ),
                    )
                )
        return coords

    # -- introspection -------------------------------------------------------
    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.plan.feature_names

    @property
    def feature_metadata(self) -> list[dict[str, Any]]:
        return [c.to_dict() for c in self.plan.coordinates]

    @property
    def coordinates(self) -> tuple[Coordinate, ...]:
        return self.plan.coordinates

    @property
    def output_dimension(self) -> int:
        return self.plan.output_dimension

    @property
    def source_dimension(self) -> int:
        return self.plan.source_dimension

    @property
    def level0_dimension(self) -> int:
        return self.plan.level0_dimension

    @property
    def level1_dimension(self) -> int:
        return self.plan.level1_dimension

    @property
    def seed(self) -> int:
        return int(self.projection_seed)

    @property
    def uses_labels(self) -> bool:
        """Always ``False``: the encoder consumes a label-free record bank only."""
        return False

    @property
    def provenance(self) -> dict[str, Any]:
        spec = self.plan.projection
        return {
            "schema": SCHEMA,
            "uses_labels": False,
            "deterministic": True,
            "fitted_state": False,
            "learned": False,
            "residual_implemented": False,
            "structured_d": self.plan.structured_d,
            "output_dimension": self.plan.output_dimension,
            "level0_dimension": self.plan.level0_dimension,
            "level1_dimension": self.plan.level1_dimension,
            "source_dimension": self.plan.source_dimension,
            "requested_blocks": list(self.plan.requested_blocks),
            "present_blocks": list(self.plan.present_blocks),
            "absent_blocks": [b for b in self.plan.requested_blocks if b not in set(self.plan.present_blocks)],
            "block_detail_dimensions": {b: len(n) for b, n in self.plan.detail_definitions.items()},
            "level1_definitions": {b: list(n) for b, n in self.plan.detail_definitions.items()},
            "selection_policy": (
                "prefix of the canonical coordinate order: level 0 (block order, alphabetical within "
                "block) then level 1 (block order, fixed detail order) then fixed projection"
            ),
            "projection": dict(spec) if spec else {"used": False, "seed": int(self.projection_seed)},
            "projection_used": bool(spec),
            "projection_seed": int(self.projection_seed),
            "chunk_size": self.chunk_size,
            "bank": {
                "schema": self.bank.provenance.get("schema"),
                "n_neurons": self.bank.n_neurons,
                "uses_labels": False,
            },
            "configured": dict(self.configured) if self.configured else None,
            "warnings": list(self.warnings),
        }

    # -- construction --------------------------------------------------------
    def projection_matrix(self) -> np.ndarray | None:
        """The fixed projection matrix ``(source_dimension, projection_dim)``, or ``None``."""
        spec = self.plan.projection
        if spec is None:
            return None
        if self._projection is None:
            source = int(spec["source_dimension"])
            out = int(spec["output_dimension"])
            # Each column is generated from default_rng([seed, column_index]) so that the
            # projection is *prefix-stable* in the output dimension: extending structured_d
            # never changes the projected coordinates that were already present.
            g = np.empty((source, out), dtype=np.float64)
            scale = 1.0 / np.sqrt(float(source))
            for column in range(out):
                rng = np.random.default_rng([int(spec["seed"]), int(column)])
                g[:, column] = rng.standard_normal(source) * scale
            if g.ndim != 2 or g.shape != (source, out):  # pragma: no cover - defensive
                raise StructuredVectorError("internal projection error: unexpected matrix shape")
            g.setflags(write=False)
            self._projection = g
        return self._projection

    def _level0_chunk(self, start: int, stop: int) -> np.ndarray:
        if not self.plan.level0_layout:
            return np.empty((stop - start, 0), dtype=np.float64)
        columns = [
            self.bank.get_block(block).features[feature][start:stop]
            for block, feature in self.plan.level0_layout
        ]
        return np.column_stack(columns)

    def _level1_chunk(self, start: int, stop: int) -> np.ndarray:
        if not self.plan.level1_layout:
            return np.empty((stop - start, 0), dtype=np.float64)
        per_block: dict[tuple[str, str], np.ndarray] = {}
        for bname in self.plan.detail_definitions:
            block = self.bank.get_block(bname)
            if block.weights is not None:
                detail = _weight_detail_chunk(block.weights[start:stop])
                for i, name in enumerate(WEIGHT_DETAIL_NAMES):
                    per_block[(bname, name)] = detail[:, i]
            if block.name == "activity" and block.samples is not None:
                detail = _activity_detail_chunk(block.samples[:, start:stop])
                for i, name in enumerate(ACTIVITY_DETAIL_NAMES):
                    per_block[(bname, name)] = detail[:, i]
        return np.column_stack([per_block[key] for key in self.plan.level1_layout])

    def _encode_chunk(self, start: int, stop: int) -> np.ndarray:
        d = self.plan.structured_d
        level0 = self._level0_chunk(start, stop)
        if d <= level0.shape[1]:
            return level0[:, :d]
        parts = [level0]
        remaining = d - level0.shape[1]
        if remaining > 0 and self.plan.level1_dimension > 0:
            take = min(remaining, self.plan.level1_dimension)
            parts.append(self._level1_chunk(start, stop)[:, :take])
            remaining -= take
        if remaining > 0:
            source = np.concatenate(parts, axis=1)
            projection = self.projection_matrix()
            if projection is None:  # pragma: no cover - defensive
                raise StructuredVectorError("internal plan error: projection coordinates without a matrix")
            parts.append(source @ projection[:, :remaining])
        return np.concatenate(parts, axis=1)

    def encode(self) -> StructuredVectors:
        """Materialise the deterministic structured vectors (chunked over neurons)."""
        n = self.bank.n_neurons
        d = self.plan.structured_d
        X = np.empty((n, d), dtype=np.float64)
        size = n if self.chunk_size is None else int(self.chunk_size)
        for start in range(0, n, size):
            stop = min(start + size, n)
            X[start:stop] = self._encode_chunk(start, stop)
        return StructuredVectors(X=X, plan=self.plan, provenance=self.provenance)

    # -- config integration --------------------------------------------------
    @classmethod
    def from_config(
        cls,
        config: V2Config,
        bank: NeuronRecordBank,
        *,
        on_residual_request: str = "error",
        projection_seed: int = DEFAULT_PROJECTION_SEED,
        chunk_size: int | None = None,
    ) -> "StructuredVectorEncoder":
        """Build an encoder from the existing V2 configuration.

        ``vector.structured_d`` is the output dimension. ``vector.enabled_blocks`` selects
        the contributing blocks. ``memory.representation_chunk_size`` is used unless
        ``chunk_size`` is given explicitly.

        The learned residual is **not implemented**: if the configuration requests one
        (``vector.residual.enabled`` or ``vector.learned_residual_d > 0``),
        ``on_residual_request="error"`` (default) raises, while ``"warn"`` returns an
        encoder for the structured part only and records the request in
        ``warnings``/``provenance`` - the residual is never silently pretended to exist.
        """
        if on_residual_request not in ("error", "warn"):
            raise StructuredVectorError(
                f"on_residual_request must be 'error' or 'warn', got {on_residual_request!r}"
            )
        residual_enabled = bool(config.vector.residual.enabled)
        residual_dim = int(config.vector.learned_residual_d)
        warnings: list[str] = []
        if residual_enabled or residual_dim > 0:
            message = (
                "the learned residual is configured (vector.residual.enabled="
                f"{residual_enabled}, vector.learned_residual_d={residual_dim}) but is not implemented "
                "in this stage; the encoder produces only the deterministic structured part "
                f"({int(config.vector.structured_d)} dimensions)"
            )
            if on_residual_request == "error":
                raise StructuredVectorError(message)
            warnings.append(message)

        resolved_chunk = (
            int(chunk_size) if chunk_size is not None else int(config.memory.representation_chunk_size)
        )
        configured = {
            "vector": {
                "d": config.vector.d,
                "structured_d": config.vector.structured_d,
                "learned_residual_d": config.vector.learned_residual_d,
                "residual_enabled": residual_enabled,
                "enabled_blocks": list(config.vector.enabled_blocks),
                "temporal_resolution": config.vector.temporal_resolution,
                "context_depth": config.vector.context_depth,
            },
            "memory": {
                "representation_chunk_size": config.memory.representation_chunk_size,
                "record_batch_size": config.memory.record_batch_size,
                "storage": config.memory.storage,
            },
            "residual_requested": bool(residual_enabled or residual_dim > 0),
            "residual_implemented": False,
        }
        return cls(
            bank=bank,
            structured_d=int(config.vector.structured_d),
            enabled_blocks=tuple(config.vector.enabled_blocks),
            projection_seed=projection_seed,
            chunk_size=resolved_chunk,
            warnings=warnings,
            configured=configured,
        )


# --------------------------------------------------------------------------
# Convenience API
# --------------------------------------------------------------------------
def encode_structured_vectors(
    bank: NeuronRecordBank,
    *,
    structured_d: int,
    enabled_blocks: Sequence[str] | str | None = None,
    projection_seed: int = DEFAULT_PROJECTION_SEED,
    chunk_size: int | None = None,
) -> StructuredVectors:
    """Encode ``bank`` into deterministic structured vectors of dimension ``structured_d``.

    Convenience wrapper around :class:`StructuredVectorEncoder`; returns the matrix, the
    coordinate names and the per-coordinate provenance.
    """
    encoder = StructuredVectorEncoder(
        bank=bank,
        structured_d=structured_d,
        enabled_blocks=enabled_blocks,
        projection_seed=projection_seed,
        chunk_size=chunk_size,
    )
    return encoder.encode()


def _as_positive_int(value: Any, name: str, maximum: int, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool):
        raise StructuredVectorError(f"{name} must be an integer, got {value!r}")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise StructuredVectorError(f"{name} must be an integer, got {value!r}") from exc
    minimum = 0 if allow_zero else 1
    if number < minimum or number > maximum:
        raise StructuredVectorError(f"{name} must be in [{minimum}, {maximum}], got {number}")
    return number


__all__ = [
    "SCHEMA",
    "DEFAULT_PROJECTION_SEED",
    "MAX_STRUCTURED_D",
    "SIGNED_QUANTILE_LEVELS",
    "ABS_HISTOGRAM_BINS",
    "TOP_SIGNED_COUNT",
    "ACTIVITY_COUNT_QUANTILES",
    "WEIGHT_DETAIL_NAMES",
    "ACTIVITY_DETAIL_NAMES",
    "DETAIL_DEFINITIONS",
    "StructuredVectorError",
    "Coordinate",
    "EncoderPlan",
    "StructuredVectors",
    "StructuredVectorEncoder",
    "encode_structured_vectors",
]