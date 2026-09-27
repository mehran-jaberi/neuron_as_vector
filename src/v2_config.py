"""V2 configuration infrastructure: vector / memory / precision / network / experiment / simulation.

Scope (important)
-----------------
This module is **configuration only**. It adds the typed control surface that the
V2 "unified NeuronRecord" work will need and validates it. It deliberately does
**not** implement the ``NeuronRecord``, the structured vector encoder, the learned
residual, the new functional target, or any analysis, and it does not change any
scientific behaviour.

Design (follows the repository's existing conventions)
------------------------------------------------------
* The configuration objects are dataclasses with ``from_mapping`` / ``from_config``
  classmethods, exactly like :class:`src.model.SNNConfig` and
  :class:`src.training.TrainConfig`.
* They are built on top of the existing :class:`src.utils.Config` container, so a
  single YAML file plus the existing ``--override dotted.key=value`` arguments
  configure them (see :func:`src.utils.load_config`).
* ``V2Config.from_config`` resolves the *existing* model/train blocks into
  :class:`SNNConfig` / :class:`TrainConfig` as well, so one object exposes both the
  current experiment and the new V2 controls.

Backward compatibility
----------------------
An existing config (``configs/baseline.yaml``, ``baseline_repaired.yaml``,
``neuron_space_baseline.yaml``, ``analysis.yaml``) has no V2 sections. Loading it
still yields usable V2 defaults that describe today's behaviour:

* ``vector`` defaults to the current 48-D structural representation
  (``d = structured_d = 48``), with ``learned_residual_d = 0`` and
  ``residual.enabled = false``;
* ``memory.train_batch_size`` / ``memory.eval_batch_size`` mirror the existing
  ``train.batch_size`` / ``train.eval_batch_size`` unless overridden;
* every precision field is ``float32``.

The V2 dimension invariant
---------------------------
``vector.d = vector.structured_d + vector.learned_residual_d``

Exactly one of the three may be derived automatically from the other two; missing
values fall back to the documented defaults (``structured_d = 48``,
``learned_residual_d = 0``). Contradictory combinations raise :class:`V2ConfigError`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping, Sequence

import numpy as np

from .model import SNNConfig
from .training import TrainConfig
from .utils import Config


# --------------------------------------------------------------------------
# Errors and registries
# --------------------------------------------------------------------------
class V2ConfigError(ValueError):
    """Raised for an invalid or contradictory V2 configuration.

    Subclasses :class:`ValueError` so callers that already treat configuration
    problems as value errors (as the existing code does) keep working.
    """


#: The current structural representation dimension, used as the default vector
#: budget so that a config without a ``vector`` section describes today's setup.
DEFAULT_STRUCTURED_D = 48

#: Default dense-input token budget (``batch x n_bins x n_input``). The current
#: measured maximum is B=256, T=700, C=700 = 125 440 000 tokens (~478 MiB fp32),
#: so this cap leaves headroom for the existing baseline while preventing an
#: accidental order-of-magnitude larger ``(B, T, C)`` allocation. ``0`` disables.
DEFAULT_MAX_INPUT_TOKENS = 150_000_000

#: Largest accepted value of ``model.n_layers`` (sanity bound; only 1 is implemented).
MAX_N_LAYERS = 64

#: All block names the future unified record will support (configuration interface
#: only - see :data:`IMPLEMENTED_RECORD_BLOCKS`).
V2_RECORD_BLOCKS: tuple[str, ...] = (
    "intrinsic",
    "input_conn",
    "recurrent_in",
    "recurrent_out",
    "activity",
    "temporal",
    "network_context",
)

#: Blocks that the current code can actually build. The others are accepted by the
#: configuration (so the interface is usable for planning) but are reported as
#: not implemented and rejected when ``strict=True``.
IMPLEMENTED_RECORD_BLOCKS = frozenset(
    {"intrinsic", "input_conn", "recurrent_in", "recurrent_out", "activity", "temporal"}
)

#: Default block selection = the current primary (structural) representation.
DEFAULT_ENABLED_BLOCKS: tuple[str, ...] = (
    "intrinsic",
    "input_conn",
    "recurrent_in",
    "recurrent_out",
)

SUPPORTED_MODEL_TYPES: tuple[str, ...] = ("recurrent_lif",)
SUPPORTED_DEVICES: tuple[str, ...] = ("auto", "cpu", "cuda")
SUPPORTED_STORAGE: tuple[str, ...] = ("cpu", "gpu", "memmap")
SUPPORTED_DATASETS: tuple[str, ...] = ("shd", "synthetic")
SUPPORTED_SPLIT_STRATEGIES: tuple[str, ...] = ("speaker_aware", "stratified")

#: Canonical dtype names and the accepted aliases (normalised to the canonical form).
DTYPE_ALIASES: dict[str, str] = {
    "float32": "float32",
    "fp32": "float32",
    "float16": "float16",
    "fp16": "float16",
    "half": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
}
SUPPORTED_DTYPES: tuple[str, ...] = ("float32", "float16", "bfloat16")

_BYTES_PER_DTYPE: dict[str, int] = {"float32": 4, "float16": 2, "bfloat16": 2}

#: Default dimension of the compact label-free functional-response source view.
DEFAULT_FUNCTIONAL_SOURCE_DIM = 64

#: Default seed of the fixed functional-response projection (separate from every other seed).
DEFAULT_FUNCTIONAL_PROJECTION_SEED = 0

#: Upper sanity bound for ``vector.functional_source_dim`` (the projection matrix is
#: ``n_fit_samples x functional_source_dim``).
MAX_FUNCTIONAL_SOURCE_DIM = 4096

#: Normalisation modes of the individual-stimulus functional-response source view.
FUNCTIONAL_SOURCE_NORMALIZATIONS: tuple[str, ...] = ("raw", "neuron_centered", "neuron_zscored")

#: Standardisation modes of the learned residual's input view (the canonical list; the
#: residual implementation imports it from here so the rule lives in exactly one place).
RESIDUAL_STANDARDIZATIONS: tuple[str, ...] = ("train_standardise", "none")

#: Masking strategies of the learned residual's reconstruction objective.
SOURCE_MASK_MODES: tuple[str, ...] = ("coordinate", "block")


# --------------------------------------------------------------------------
# Coercion helpers (robust against YAML/CLI string values)
# --------------------------------------------------------------------------
def _as_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise V2ConfigError(f"{field_name} must be an integer, got {value!r}")
    if isinstance(value, float):
        if not float(value).is_integer():
            raise V2ConfigError(f"{field_name} must be an integer, got {value!r}")
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise V2ConfigError(f"{field_name} must be an integer, got {value!r}") from exc


def _as_optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _as_int(value, field_name)


def _as_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "1", "yes", "on"):
            return True
        if low in ("false", "0", "no", "off"):
            return False
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    raise V2ConfigError(f"{field_name} must be a boolean, got {value!r}")


def _as_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise V2ConfigError(f"{field_name} must be a non-empty string, got {value!r}")
    return value.strip()


def _normalize_dtype(value: Any, field_name: str) -> str:
    text = _as_str(value, field_name).lower()
    canonical = DTYPE_ALIASES.get(text)
    if canonical is None:
        raise V2ConfigError(
            f"{field_name}: unsupported dtype {value!r}; supported values are "
            f"{list(SUPPORTED_DTYPES)} (aliases: fp32/fp16/bf16)"
        )
    return canonical


def _normalize_blocks(value: Any, field_name: str = "vector.enabled_blocks") -> list[str]:
    """Accept a list or a comma/space separated string; dedupe in canonical order."""
    if value is None:
        return list(DEFAULT_ENABLED_BLOCKS)
    if isinstance(value, str):
        raw = [part for part in value.replace(",", " ").split() if part]
    elif isinstance(value, Sequence):
        raw = [str(item).strip() for item in value if str(item).strip()]
    else:
        raise V2ConfigError(f"{field_name} must be a list of block names, got {value!r}")
    unknown = sorted({name for name in raw if name not in V2_RECORD_BLOCKS})
    if unknown:
        raise V2ConfigError(
            f"{field_name}: unknown block(s) {unknown}; valid names are {list(V2_RECORD_BLOCKS)}"
        )
    # de-duplicate, keep the canonical registry order for reproducibility
    return [name for name in V2_RECORD_BLOCKS if name in raw]


def _filter_mapping(cls: type, mapping: Mapping[str, Any] | None) -> dict[str, Any]:
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in dict(mapping or {}).items() if k in valid}


# --------------------------------------------------------------------------
# Vector configuration
# --------------------------------------------------------------------------
@dataclass
class VectorResidualConfig:
    """Configuration of the learned residual and of the sources it may consume.

    ``enabled``/``learned_residual_d > 0`` cross-check exactly as before. The three new
    source flags are **opt-in**: with their defaults the residual source view is
    bit-identical to the previous stage (same feature names, same schema hash), so the
    historical configuration path is unchanged.

    ``source_functional_response``
        Include the compact fixed projection of the label-free FIT individual-stimulus
        response profile (``(n_fit_samples, n_neurons)`` counts).
    ``source_temporal``
        Include the coarse label-free FIT temporal block (mean firing rate per coarse
        bin). Independent of ``vector.enabled_blocks``: the source can consume the block
        even when the structured encoder does not select it.
    ``mask_mode``
        ``coordinate`` (default, the previous behaviour) withholds individual source
        coordinates; ``block`` withholds one whole source group per example
        (structural / functional_response / temporal) so a source must be reconstructed
        from the others.

    The remaining fields are the **training protocol** of the learned residual. Their
    defaults are exactly the protocol the capacity / rate-robustness / source-extension
    studies used (200 epochs, batch 64, Adam lr 1e-3, mask fraction 0.25, a 0.2 neuron
    validation split inside FIT, coordinate masking, train-split standardisation), so an
    existing config reproduces those artifacts unchanged. They are exposed here so that a
    control panel can *state* and reproduce the protocol from the one configuration system
    instead of hard-coding it in a script.
    """

    enabled: bool = False
    source_functional_response: bool = False
    source_temporal: bool = False
    mask_mode: str = "coordinate"
    # -- training protocol (defaults = the established protocol) --------------
    seed: int = 0
    split_seed: int = 0
    mask_seed: int = 0
    hidden_dim: int = 64
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3
    mask_fraction: float = 0.25
    minimum_visible_features: int = 8
    val_fraction: float = 0.2
    standardization: str = "train_standardise"

    def __post_init__(self) -> None:
        self.enabled = _as_bool(self.enabled, "vector.residual.enabled")
        self.source_functional_response = _as_bool(
            self.source_functional_response, "vector.residual.source_functional_response"
        )
        self.source_temporal = _as_bool(
            self.source_temporal, "vector.residual.source_temporal"
        )
        self.mask_mode = _as_str(self.mask_mode, "vector.residual.mask_mode").lower()
        if self.mask_mode not in SOURCE_MASK_MODES:
            raise V2ConfigError(
                f"vector.residual.mask_mode must be one of {list(SOURCE_MASK_MODES)}, "
                f"got {self.mask_mode!r}"
            )
        for name in ("seed", "split_seed", "mask_seed"):
            value = _as_int(getattr(self, name), f"vector.residual.{name}")
            if value < 0:
                raise V2ConfigError(f"vector.residual.{name} must be >= 0, got {value}")
            setattr(self, name, value)
        for name in ("hidden_dim", "epochs", "batch_size", "minimum_visible_features"):
            value = _as_int(getattr(self, name), f"vector.residual.{name}")
            if value < 1:
                raise V2ConfigError(f"vector.residual.{name} must be >= 1, got {value}")
            setattr(self, name, value)
        try:
            learning_rate = float(self.learning_rate)
        except (TypeError, ValueError) as exc:
            raise V2ConfigError(
                f"vector.residual.learning_rate must be a number, got {self.learning_rate!r}"
            ) from exc
        if not learning_rate > 0:
            raise V2ConfigError(
                f"vector.residual.learning_rate must be > 0, got {learning_rate}"
            )
        self.learning_rate = learning_rate
        for name in ("mask_fraction", "val_fraction"):
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError) as exc:
                raise V2ConfigError(f"vector.residual.{name} must be a number") from exc
            setattr(self, name, value)
        if not 0.0 <= self.mask_fraction < 1.0:
            raise V2ConfigError(
                f"vector.residual.mask_fraction must be in [0, 1), got {self.mask_fraction}"
            )
        if not 0.0 <= self.val_fraction < 0.5:
            raise V2ConfigError(
                f"vector.residual.val_fraction must be in [0, 0.5), got {self.val_fraction}"
            )
        self.standardization = _as_str(
            self.standardization, "vector.residual.standardization"
        ).lower()
        if self.standardization not in RESIDUAL_STANDARDIZATIONS:
            raise V2ConfigError(
                "vector.residual.standardization must be one of "
                f"{list(RESIDUAL_STANDARDIZATIONS)}, got {self.standardization!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "VectorResidualConfig":
        raw = dict(mapping or {})
        # `normalization` is accepted as a synonym so an existing residual-style key is
        # never silently dropped; `standardization` is the documented name.
        if "normalization" in raw and "standardization" not in raw:
            raw["standardization"] = raw["normalization"]
        return cls(**_filter_mapping(cls, raw))


@dataclass
class VectorConfig:
    """Vector dimension and composition policy (``n = neurons``, ``d = dimensions``).

    ``d``, ``structured_d`` and ``learned_residual_d`` obey the invariant
    ``d = structured_d + learned_residual_d``. At most one of the three may be
    omitted (it is derived from the other two); if more are omitted they fall back
    to the documented defaults, after which the invariant is enforced.
    """

    d: int | None = None
    structured_d: int | None = None
    learned_residual_d: int | None = None
    enabled_blocks: list[str] = field(default_factory=lambda: list(DEFAULT_ENABLED_BLOCKS))
    temporal_resolution: int = 10
    context_depth: int = 0
    #: Compact label-free FIT individual-stimulus response source (residual input view).
    functional_source_dim: int = DEFAULT_FUNCTIONAL_SOURCE_DIM
    functional_projection_seed: int = DEFAULT_FUNCTIONAL_PROJECTION_SEED
    functional_source_normalization: str = "raw"
    residual: VectorResidualConfig = field(default_factory=VectorResidualConfig)

    def __post_init__(self) -> None:
        if isinstance(self.residual, Mapping):
            self.residual = VectorResidualConfig.from_mapping(self.residual)
        elif not isinstance(self.residual, VectorResidualConfig):
            raise V2ConfigError(
                f"vector.residual must be a mapping or VectorResidualConfig, got {type(self.residual)!r}"
            )
        self.enabled_blocks = _normalize_blocks(self.enabled_blocks)
        if not self.enabled_blocks:
            raise V2ConfigError("vector.enabled_blocks must select at least one block")

        self.d = _as_optional_int(self.d, "vector.d")
        self.structured_d = _as_optional_int(self.structured_d, "vector.structured_d")
        self.learned_residual_d = _as_optional_int(
            self.learned_residual_d, "vector.learned_residual_d"
        )
        self._resolve_dimensions()

        self.temporal_resolution = _as_int(self.temporal_resolution, "vector.temporal_resolution")
        if self.temporal_resolution < 1:
            raise V2ConfigError(
                f"vector.temporal_resolution must be >= 1, got {self.temporal_resolution}"
            )
        self.context_depth = _as_int(self.context_depth, "vector.context_depth")
        if self.context_depth < 0:
            raise V2ConfigError(f"vector.context_depth must be >= 0, got {self.context_depth}")

        self.functional_source_dim = _as_int(
            self.functional_source_dim, "vector.functional_source_dim"
        )
        if self.functional_source_dim < 1:
            raise V2ConfigError(
                f"vector.functional_source_dim must be >= 1, got {self.functional_source_dim}"
            )
        if self.functional_source_dim > MAX_FUNCTIONAL_SOURCE_DIM:
            raise V2ConfigError(
                f"vector.functional_source_dim={self.functional_source_dim} exceeds the sanity "
                f"bound {MAX_FUNCTIONAL_SOURCE_DIM}"
            )
        self.functional_projection_seed = _as_int(
            self.functional_projection_seed, "vector.functional_projection_seed"
        )
        if self.functional_projection_seed < 0:
            raise V2ConfigError(
                "vector.functional_projection_seed must be >= 0, got "
                f"{self.functional_projection_seed}"
            )
        self.functional_source_normalization = _as_str(
            self.functional_source_normalization, "vector.functional_source_normalization"
        ).lower()
        if self.functional_source_normalization not in FUNCTIONAL_SOURCE_NORMALIZATIONS:
            raise V2ConfigError(
                "vector.functional_source_normalization must be one of "
                f"{list(FUNCTIONAL_SOURCE_NORMALIZATIONS)}, got "
                f"{self.functional_source_normalization!r}"
            )
        self._validate_residual_sources()

    def _validate_residual_sources(self) -> None:
        """A source can only be requested when the learned residual consumes it."""
        for name, requested in (
            ("source_functional_response", self.residual.source_functional_response),
            ("source_temporal", self.residual.source_temporal),
        ):
            if requested and not self.residual.enabled:
                raise V2ConfigError(
                    f"vector.residual.{name}=true requires vector.residual.enabled=true"
                )
            if requested and self.learned_residual_d <= 0:
                raise V2ConfigError(
                    f"vector.residual.{name}=true requires vector.learned_residual_d > 0 "
                    "(nothing would consume the source)"
                )

    # -- dimension resolution ------------------------------------------------
    def _resolve_dimensions(self) -> None:
        d, s, r = self.d, self.structured_d, self.learned_residual_d
        if d is None and s is None and r is None:
            # No V2 vector section: reproduce the current 48-D structural default.
            s, r = DEFAULT_STRUCTURED_D, 0
            d = s + r
        elif d is not None and s is None and r is None:
            if self.residual.enabled:
                raise V2ConfigError(
                    "vector.d was given without vector.structured_d / vector.learned_residual_d "
                    "while vector.residual.enabled=true; specify the split explicitly "
                    "(learned_residual_d = d - structured_d)."
                )
            # Deterministic-only vector of dimension d.
            s, r = d, 0
        else:
            if s is None:
                s = (d - r) if (d is not None and r is not None) else DEFAULT_STRUCTURED_D
            if r is None:
                r = (d - s) if (d is not None and s is not None) else 0
            if d is None:
                d = s + r

        for name, value in (
            ("vector.d", d),
            ("vector.structured_d", s),
            ("vector.learned_residual_d", r),
        ):
            if value is not None and value < 0:
                raise V2ConfigError(f"{name} must be >= 0, got {value}")
        if not d:
            raise V2ConfigError(f"vector.d must be >= 1, got {d}")
        if d != s + r:
            raise V2ConfigError(
                "inconsistent vector dimension: "
                f"d({d}) != structured_d({s}) + learned_residual_d({r})"
            )
        if r > 0 and not self.residual.enabled:
            raise V2ConfigError(
                f"vector.learned_residual_d={r} > 0 requires vector.residual.enabled=true"
            )
        if self.residual.enabled and r == 0:
            raise V2ConfigError(
                "vector.residual.enabled=true requires vector.learned_residual_d > 0"
            )
        self.d, self.structured_d, self.learned_residual_d = int(d), int(s), int(r)

    # -- introspection -------------------------------------------------------
    @property
    def unimplemented_blocks(self) -> list[str]:
        """Enabled blocks the current code cannot build yet."""
        return [b for b in self.enabled_blocks if b not in IMPLEMENTED_RECORD_BLOCKS]

    @property
    def residual_implemented(self) -> bool:
        """True: the learned residual is implemented (:mod:`src.residual`)."""
        return True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "VectorConfig":
        raw = dict(mapping or {})
        residual = VectorResidualConfig.from_mapping(raw.get("residual") or {})
        kwargs = {k: v for k, v in _filter_mapping(cls, raw).items() if k != "residual"}
        return cls(residual=residual, **kwargs)

    @classmethod
    def from_config(cls, cfg: Config, prefix: str = "vector") -> "VectorConfig":
        return cls.from_mapping(cfg.get_path(prefix, {}) or {})


# --------------------------------------------------------------------------
# Memory configuration
# --------------------------------------------------------------------------
@dataclass
class MemoryConfig:
    """Stage-specific memory controls (defaults follow the repository's audit).

    The batch sizes are deliberately separate per stage: the audit measured
    ~610 MiB for a training step at B=128, ~490 MiB for evaluation at B=256 and
    ~1.58 GiB for a *recorded* activity pass at B=256 (which temporarily retains two
    ``(B, T, H)`` trace sets). A recorded batch of 32-64 removes that duplication
    without changing any result.
    """

    train_batch_size: int = 128
    eval_batch_size: int = 256
    record_batch_size: int = 32
    activity_chunk_size: int = 32
    representation_chunk_size: int = 256
    device: str = "auto"
    storage: str = "cpu"
    mixed_precision: bool = False
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS

    def __post_init__(self) -> None:
        for name in (
            "train_batch_size",
            "eval_batch_size",
            "record_batch_size",
            "activity_chunk_size",
            "representation_chunk_size",
        ):
            value = _as_int(getattr(self, name), f"memory.{name}")
            if value < 1:
                raise V2ConfigError(f"memory.{name} must be >= 1, got {value}")
            setattr(self, name, value)

        self.device = _as_str(self.device, "memory.device").lower()
        if self.device not in SUPPORTED_DEVICES:
            raise V2ConfigError(
                f"memory.device must be one of {list(SUPPORTED_DEVICES)}, got {self.device!r}"
            )
        self.storage = _as_str(self.storage, "memory.storage").lower()
        if self.storage not in SUPPORTED_STORAGE:
            raise V2ConfigError(
                f"memory.storage must be one of {list(SUPPORTED_STORAGE)}, got {self.storage!r}"
            )
        self.mixed_precision = _as_bool(self.mixed_precision, "memory.mixed_precision")
        self.max_input_tokens = _as_int(self.max_input_tokens, "memory.max_input_tokens")
        if self.max_input_tokens < 0:
            raise V2ConfigError(
                f"memory.max_input_tokens must be >= 0 (0 disables the guard), got {self.max_input_tokens}"
            )

    # -- token budget --------------------------------------------------------
    @property
    def max_input_tokens_enabled(self) -> bool:
        return self.max_input_tokens > 0

    def estimated_input_tokens(self, *, batch_size: int, n_bins: int, n_input: int) -> int:
        return estimate_input_tokens(batch_size, n_bins, n_input)

    def check_input_budget(
        self,
        *,
        batch_size: int,
        n_bins: int,
        n_input: int,
        dtype_name: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Estimate (and, if the guard is on, enforce) a dense ``(B, T, C)`` input."""
        return check_input_token_budget(
            batch_size,
            n_bins,
            n_input,
            max_tokens=self.max_input_tokens if max_tokens is None else max_tokens,
            dtype_name=dtype_name or "float32",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "MemoryConfig":
        return cls(**_filter_mapping(cls, mapping))

    @classmethod
    def from_config(cls, cfg: Config, *, train: TrainConfig | None = None) -> "MemoryConfig":
        """Build from ``memory``; fall back to the existing ``train``/``run`` blocks.

        Explicit ``memory.*`` values always win. Batch sizes are only inherited
        from ``train.batch_size`` / ``train.eval_batch_size`` when the memory
        section does not set them, and the device from ``run.device`` when
        ``memory.device`` is absent - so an existing config keeps its behaviour
        without gaining a new section.
        """
        raw = dict(cfg.get_path("memory", {}) or {})
        train = train if train is not None else TrainConfig.from_config(cfg)
        if "train_batch_size" not in raw:
            raw["train_batch_size"] = train.batch_size
        if "eval_batch_size" not in raw:
            raw["eval_batch_size"] = train.eval_batch_size
        if "device" not in raw:
            run_device = cfg.get_path("run.device", None)
            if run_device is not None:
                raw["device"] = run_device
        return cls.from_mapping(raw)


# --------------------------------------------------------------------------
# Precision configuration
# --------------------------------------------------------------------------
@dataclass
class PrecisionConfig:
    """Numerical/storage precision, kept *separate* from representation capacity.

    ``vector_dtype`` and ``activity_dtype`` are storage/accumulation precisions for
    the (future) neuron vectors and activity statistics; ``model_dtype`` is the SNN
    compute precision. Only ``float32`` training is implemented today, so
    ``model_dtype != float32`` is accepted by the configuration but requires a
    CUDA device and is reported as not-yet-implemented behaviour.
    """

    vector_dtype: str = "float32"
    activity_dtype: str = "float32"
    model_dtype: str = "float32"

    def __post_init__(self) -> None:
        self.vector_dtype = _normalize_dtype(self.vector_dtype, "precision.vector_dtype")
        self.activity_dtype = _normalize_dtype(self.activity_dtype, "precision.activity_dtype")
        self.model_dtype = _normalize_dtype(self.model_dtype, "precision.model_dtype")

    def _dtype_name(self, which: str) -> str:
        table = {
            "vector": self.vector_dtype,
            "activity": self.activity_dtype,
            "model": self.model_dtype,
        }
        if which not in table:
            raise V2ConfigError(
                f"precision target must be one of {sorted(table)}, got {which!r}"
            )
        return table[which]

    def to_numpy_dtype(self, which: str = "vector") -> np.dtype:
        """NumPy dtype for a target. Raises for ``bfloat16`` (no NumPy equivalent)."""
        name = self._dtype_name(which)
        if name == "bfloat16":
            raise V2ConfigError(
                f"precision.{which}_dtype='bfloat16' has no NumPy dtype; "
                "NumPy-backed storage cannot represent it"
            )
        return np.dtype(name)

    def to_torch_dtype(self, which: str = "vector") -> Any:
        """Torch dtype for a target (torch is imported lazily)."""
        import torch

        return getattr(torch, self._dtype_name(which))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "PrecisionConfig":
        return cls(**_filter_mapping(cls, mapping))

    @classmethod
    def from_config(cls, cfg: Config, prefix: str = "precision") -> "PrecisionConfig":
        return cls.from_mapping(cfg.get_path(prefix, {}) or {})


# --------------------------------------------------------------------------
# Network configuration (view over the existing `model` block)
# --------------------------------------------------------------------------
@dataclass
class NetworkConfig:
    """Network-level controls, including the future multi-layer fields.

    ``n_hidden`` and the existing ``SNNConfig`` remain the source of truth for the
    current single-layer model. ``model_type`` / ``n_layers`` /
    ``neurons_per_layer`` are the *configuration interface* for later stages; the
    SNN implementation is untouched by this module.
    """

    model_type: str = "recurrent_lif"
    n_hidden: int = 256
    n_layers: int = 1
    neurons_per_layer: list[int] | None = None

    def __post_init__(self) -> None:
        self.model_type = _as_str(self.model_type, "model.model_type").lower()
        if self.model_type not in SUPPORTED_MODEL_TYPES:
            raise V2ConfigError(
                f"model.model_type must be one of {list(SUPPORTED_MODEL_TYPES)}, "
                f"got {self.model_type!r}"
            )
        self.n_hidden = _as_int(self.n_hidden, "model.n_hidden")
        if self.n_hidden < 1:
            raise V2ConfigError(f"model.n_hidden must be >= 1, got {self.n_hidden}")
        self.n_layers = _as_int(self.n_layers, "model.n_layers")
        if self.n_layers < 1:
            raise V2ConfigError(f"model.n_layers must be >= 1, got {self.n_layers}")
        if self.n_layers > MAX_N_LAYERS:
            raise V2ConfigError(
                f"model.n_layers must be <= {MAX_N_LAYERS}, got {self.n_layers}"
            )

        npl = self.neurons_per_layer
        if npl is None:
            self.neurons_per_layer = [self.n_hidden] * self.n_layers
        else:
            if isinstance(npl, (int, float)):
                npl = [npl]
            if not isinstance(npl, Sequence) or isinstance(npl, str):
                raise V2ConfigError(
                    f"model.neurons_per_layer must be a list of integers, got {npl!r}"
                )
            sizes = [_as_int(v, "model.neurons_per_layer") for v in npl]
            if len(sizes) != self.n_layers:
                raise V2ConfigError(
                    f"model.neurons_per_layer has {len(sizes)} entries but "
                    f"model.n_layers={self.n_layers}"
                )
            if any(size < 1 for size in sizes):
                raise V2ConfigError(
                    f"model.neurons_per_layer entries must all be >= 1, got {sizes}"
                )
            if self.n_layers == 1 and sizes[0] != self.n_hidden:
                raise V2ConfigError(
                    f"model.neurons_per_layer={sizes} disagrees with model.n_hidden={self.n_hidden}"
                )
            self.neurons_per_layer = sizes

    @property
    def multi_layer_implemented(self) -> bool:
        """False in this stage: only a single hidden layer is implemented."""
        return self.n_layers == 1

    def require_implemented(self) -> None:
        """Raise if the requested architecture is not implemented yet."""
        if not self.multi_layer_implemented:
            raise V2ConfigError(
                f"model.n_layers={self.n_layers} was requested but multi-layer SNN support "
                "is not implemented in this stage (only n_layers=1)"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_config(cls, cfg: Config, *, snn: SNNConfig | None = None) -> "NetworkConfig":
        raw = dict(cfg.get_path("model", {}) or {})
        snn = snn if snn is not None else SNNConfig.from_config(cfg)
        return cls(
            model_type=raw.get("model_type", "recurrent_lif"),
            n_hidden=snn.n_hidden,
            n_layers=raw.get("n_layers", 1),
            neurons_per_layer=raw.get("neurons_per_layer", None),
        )


# --------------------------------------------------------------------------
# Experiment configuration (FIT / DEV / PROBE / TEST stays as implemented)
# --------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    """Typed view of the existing experiment controls.

    Nothing here changes the split implementation:
    FIT stays the label-free representation source, DEV the model-selection split,
    PROBE the held-out functional analysis set, and the official TEST set is never
    touched by the neuron-space analysis. ``n_fit`` / ``n_probe`` are *declared
    expected sizes*; when set they are asserted against the realised split via
    :meth:`validate_against_split` and are never used to re-split the data.
    """

    dataset: str = "shd"
    split: str = "speaker_aware"
    n_fit: int | None = None
    n_probe: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        self.dataset = _as_str(self.dataset, "experiment.dataset").lower()
        if self.dataset not in SUPPORTED_DATASETS:
            raise V2ConfigError(
                f"experiment.dataset must be one of {list(SUPPORTED_DATASETS)}, got {self.dataset!r}"
            )
        self.split = _as_str(self.split, "experiment.split").lower()
        if self.split not in SUPPORTED_SPLIT_STRATEGIES:
            raise V2ConfigError(
                f"experiment.split must be one of {list(SUPPORTED_SPLIT_STRATEGIES)}, "
                f"got {self.split!r}"
            )
        for name in ("n_fit", "n_probe"):
            value = _as_optional_int(getattr(self, name), f"experiment.{name}")
            if value is not None and value < 1:
                raise V2ConfigError(f"experiment.{name} must be >= 1 when set, got {value}")
            setattr(self, name, value)
        self.seed = _as_int(self.seed, "experiment.seed")
        if self.seed < 0:
            raise V2ConfigError(f"experiment.seed must be >= 0, got {self.seed}")

    def validate_against_split(self, split_info: Mapping[str, Any]) -> dict[str, Any]:
        """Check declared expected sizes against a realised split's info dict.

        ``split_info`` is the dict returned by the existing split helpers. Only the
        checks whose source key is present are performed; nothing is modified.
        """
        info = dict(split_info or {})
        checks: dict[str, Any] = {}
        if self.n_fit is not None:
            realised = info.get("n_train")
            if realised is None:
                raise V2ConfigError(
                    "experiment.n_fit was declared but split_info has no 'n_train' entry"
                )
            if int(realised) != self.n_fit:
                raise V2ConfigError(
                    f"experiment.n_fit={self.n_fit} but the realised FIT split has {realised} samples"
                )
            checks["n_fit"] = int(realised)
        if self.n_probe is not None:
            realised = info.get("n_probe")
            if realised is None:
                raise V2ConfigError(
                    "experiment.n_probe was declared but split_info has no 'n_probe' entry"
                )
            if int(realised) != self.n_probe:
                raise V2ConfigError(
                    f"experiment.n_probe={self.n_probe} but the realised PROBE split has {realised} samples"
                )
            checks["n_probe"] = int(realised)
        return checks

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_config(cls, cfg: Config, prefix: str = "experiment") -> "ExperimentConfig":
        raw = dict(cfg.get_path(prefix, {}) or {})
        dataset = raw.get("dataset", None)
        if dataset is None:
            dataset = "synthetic" if bool(cfg.get_path("run.synthetic", False)) else "shd"
        split = raw.get("split", None)
        if split is None:
            split = "speaker_aware" if bool(cfg.get_path("data.prefer_speaker_aware", True)) else "stratified"
        seed = raw.get("seed", None)
        if seed is None:
            seed = cfg.get_path("seed", 0)
        return cls(
            dataset=dataset,
            split=split,
            n_fit=raw.get("n_fit", None),
            n_probe=raw.get("n_probe", None),
            seed=seed,
        )


# --------------------------------------------------------------------------
# Simulation configuration (view over the existing `model` block)
# --------------------------------------------------------------------------
@dataclass
class SimulationConfig:
    """Typed view of the temporal discretisation: ``n_bins``, ``bin_ms`` (= dt).

    Uses the repository's existing names (``n_bins`` / ``bin_ms``); ``dt_ms`` and
    ``duration_ms`` are derived properties. The 700-bin / 2 ms baseline is unchanged
    because the values come straight from the model block.
    """

    n_bins: int = 500
    bin_ms: float = 2.0

    def __post_init__(self) -> None:
        self.n_bins = _as_int(self.n_bins, "model.n_bins")
        if self.n_bins < 1:
            raise V2ConfigError(f"model.n_bins must be >= 1, got {self.n_bins}")
        try:
            self.bin_ms = float(self.bin_ms)
        except (TypeError, ValueError) as exc:
            raise V2ConfigError(f"model.bin_ms must be a number, got {self.bin_ms!r}") from exc
        if self.bin_ms <= 0:
            raise V2ConfigError(f"model.bin_ms must be > 0, got {self.bin_ms}")

    @property
    def dt_ms(self) -> float:
        """Simulation timestep in ms (the repository calls it ``bin_ms``)."""
        return self.bin_ms

    @property
    def duration_ms(self) -> float:
        return self.n_bins * self.bin_ms

    @property
    def duration_s(self) -> float:
        return self.duration_ms / 1000.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_snn_config(cls, snn: SNNConfig) -> "SimulationConfig":
        return cls(n_bins=snn.n_bins, bin_ms=snn.bin_ms)

    @classmethod
    def from_config(cls, cfg: Config, *, snn: SNNConfig | None = None) -> "SimulationConfig":
        snn = snn if snn is not None else SNNConfig.from_config(cfg)
        return cls.from_snn_config(snn)


# --------------------------------------------------------------------------
# Memory budget helpers
# --------------------------------------------------------------------------
def estimate_input_tokens(batch_size: int, n_bins: int, n_input: int) -> int:
    """Dense input size in elements: ``B * T * C``."""
    batch_size = _as_int(batch_size, "batch_size")
    n_bins = _as_int(n_bins, "n_bins")
    n_input = _as_int(n_input, "n_input")
    for name, value in (("batch_size", batch_size), ("n_bins", n_bins), ("n_input", n_input)):
        if value < 1:
            raise V2ConfigError(f"{name} must be >= 1, got {value}")
    return batch_size * n_bins * n_input


def dtype_itemsize(dtype_name: str) -> int:
    """Bytes per element for a supported dtype name."""
    canonical = DTYPE_ALIASES.get(str(dtype_name).strip().lower())
    if canonical is None:
        raise V2ConfigError(
            f"unsupported dtype {dtype_name!r}; supported values are {list(SUPPORTED_DTYPES)}"
        )
    return _BYTES_PER_DTYPE[canonical]


def check_input_token_budget(
    batch_size: int,
    n_bins: int,
    n_input: int,
    *,
    max_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
    dtype_name: str = "float32",
) -> dict[str, Any]:
    """Guard a dense ``(B, T, C)`` allocation before it happens.

    Returns a report with the estimated token count and byte size; raises
    :class:`V2ConfigError` when ``max_tokens > 0`` and the estimate exceeds it
    (``max_tokens = 0`` disables the guard). This is the token-budget guard
    recommended by the repository audit; it is a standalone helper and is **not**
    wired into any existing computation.
    """
    tokens = estimate_input_tokens(batch_size, n_bins, n_input)
    max_tokens = _as_int(max_tokens, "memory.max_input_tokens")
    if max_tokens < 0:
        raise V2ConfigError(
            f"memory.max_input_tokens must be >= 0 (0 disables the guard), got {max_tokens}"
        )
    itemsize = dtype_itemsize(dtype_name or "float32")
    if max_tokens > 0 and tokens > max_tokens:
        raise V2ConfigError(
            "dense input tensor would be too large: "
            f"batch_size({batch_size}) * n_bins({n_bins}) * n_input({n_input}) = {tokens:,} tokens "
            f"({tokens * itemsize / 2**20:.1f} MiB at {dtype_name}) exceeds "
            f"memory.max_input_tokens={max_tokens:,}. Lower the batch size/chunk size or raise the limit."
        )
    return {
        "batch_size": int(batch_size),
        "n_bins": int(n_bins),
        "n_input": int(n_input),
        "tokens": int(tokens),
        "dtype": DTYPE_ALIASES.get(str(dtype_name).strip().lower(), str(dtype_name)),
        "bytes": int(tokens * itemsize),
        "max_tokens": int(max_tokens),
        "within_budget": bool(max_tokens == 0 or tokens <= max_tokens),
    }


# --------------------------------------------------------------------------
# Top-level V2 configuration
# --------------------------------------------------------------------------
@dataclass
class V2Config:
    """The resolved V2 configuration: new sections + the existing model/train configs."""

    vector: VectorConfig
    memory: MemoryConfig
    precision: PrecisionConfig
    network: NetworkConfig
    experiment: ExperimentConfig
    simulation: SimulationConfig
    snn: SNNConfig
    train: TrainConfig
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.warnings = list(validate_v2_config(self, strict=False))

    # -- construction --------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        cfg: Config,
        *,
        strict: bool = False,
        warn: bool = True,
    ) -> "V2Config":
        """Resolve a :class:`Config` into a validated :class:`V2Config`.

        ``strict=True`` turns "requested but not implemented in this stage"
        warnings (multi-layer SNN, not-yet-implemented record blocks,
        mixed-precision training) into :class:`V2ConfigError`; ``warn=True`` prints them
        to stdout. The learned residual and the temporal block *are* implemented, so
        selecting them is not a warning.
        """
        snn = SNNConfig.from_config(cfg)
        train = TrainConfig.from_config(cfg)
        v2 = cls(
            vector=VectorConfig.from_config(cfg),
            memory=MemoryConfig.from_config(cfg, train=train),
            precision=PrecisionConfig.from_config(cfg),
            network=NetworkConfig.from_config(cfg, snn=snn),
            experiment=ExperimentConfig.from_config(cfg),
            simulation=SimulationConfig.from_snn_config(snn),
            snn=snn,
            train=train,
        )
        if strict:
            v2.require_implemented()
        if warn:
            for message in v2.warnings:
                print(f"[v2-config] warning: {message}")
        return v2

    # -- validation ----------------------------------------------------------
    def validate(self, *, strict: bool = False) -> list[str]:
        """Re-run cross-section validation and return the (current) warnings."""
        self.warnings = list(validate_v2_config(self, strict=strict))
        return list(self.warnings)

    def require_implemented(self) -> None:
        """Raise if anything is configured that this stage does not implement yet."""
        validate_v2_config(self, strict=True)

    # -- serialisation / reporting ------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary_rows(self) -> list[tuple[str, Any]]:
        """Flat ``(name, value)`` rows for console reporting and provenance."""
        v, m, p, n, e, s = (
            self.vector,
            self.memory,
            self.precision,
            self.network,
            self.experiment,
            self.simulation,
        )
        return [
            ("model.type", n.model_type),
            ("model.n_hidden (n)", n.n_hidden),
            ("model.n_layers", n.n_layers),
            ("vector.d", v.d),
            ("vector.structured_d", v.structured_d),
            ("vector.learned_residual_d", v.learned_residual_d),
            ("vector.residual.enabled", v.residual.enabled),
            ("vector.enabled_blocks", ", ".join(v.enabled_blocks)),
            ("vector.temporal_resolution", v.temporal_resolution),
            ("vector.context_depth", v.context_depth),
            ("vector.functional_source_dim", v.functional_source_dim),
            ("vector.functional_projection_seed", v.functional_projection_seed),
            ("vector.functional_source_normalization", v.functional_source_normalization),
            ("vector.residual.source_functional_response", v.residual.source_functional_response),
            ("vector.residual.source_temporal", v.residual.source_temporal),
            ("vector.residual.mask_mode", v.residual.mask_mode),
            ("vector.residual.seed", v.residual.seed),
            ("vector.residual.split_seed", v.residual.split_seed),
            ("vector.residual.mask_seed", v.residual.mask_seed),
            ("vector.residual.hidden_dim", v.residual.hidden_dim),
            ("vector.residual.epochs", v.residual.epochs),
            ("vector.residual.batch_size", v.residual.batch_size),
            ("vector.residual.learning_rate", v.residual.learning_rate),
            ("vector.residual.mask_fraction", v.residual.mask_fraction),
            ("vector.residual.minimum_visible_features", v.residual.minimum_visible_features),
            ("vector.residual.val_fraction", v.residual.val_fraction),
            ("vector.residual.standardization", v.residual.standardization),
            ("memory.train_batch_size", m.train_batch_size),
            ("memory.eval_batch_size", m.eval_batch_size),
            ("memory.record_batch_size", m.record_batch_size),
            ("memory.activity_chunk_size", m.activity_chunk_size),
            ("memory.representation_chunk_size", m.representation_chunk_size),
            ("memory.device", m.device),
            ("memory.storage", m.storage),
            ("memory.mixed_precision", m.mixed_precision),
            ("memory.max_input_tokens", m.max_input_tokens),
            ("precision.vector_dtype", p.vector_dtype),
            ("precision.activity_dtype", p.activity_dtype),
            ("precision.model_dtype", p.model_dtype),
            ("experiment.dataset", e.dataset),
            ("experiment.split", e.split),
            ("experiment.seed", e.seed),
            ("simulation.n_bins", s.n_bins),
            ("simulation.bin_ms (dt)", s.bin_ms),
            ("simulation.duration_ms", s.duration_ms),
            ("train.batch_size (existing)", self.train.batch_size),
            ("train.eval_batch_size (existing)", self.train.eval_batch_size),
            ("model.n_input (existing)", self.snn.n_input),
        ]


def validate_v2_config(v2: V2Config, *, strict: bool = False) -> list[str]:
    """Cross-section validation shared by the dataclass and ``from_config``.

    Hard errors (impossible dtype/device/storage combinations, contradictory
    precision, temporal resolution finer than the simulation) always raise.
    "Requested but not implemented in this stage" conditions are returned as
    warnings, or raised when ``strict=True``.
    """
    warnings: list[str] = []
    mem, prec, vec, net, sim = v2.memory, v2.precision, v2.vector, v2.network, v2.simulation

    # -- impossible combinations (always hard errors) ------------------------
    if mem.storage == "gpu" and mem.device == "cpu":
        raise V2ConfigError(
            "memory.storage='gpu' requires a CUDA device, but memory.device='cpu'"
        )
    if prec.model_dtype != "float32" and mem.device == "cpu":
        raise V2ConfigError(
            f"precision.model_dtype='{prec.model_dtype}' requires memory.device != 'cpu' "
            "(only CUDA supports it)"
        )
    if mem.mixed_precision and mem.device == "cpu":
        raise V2ConfigError(
            "memory.mixed_precision=true requires memory.device != 'cpu' (only CUDA supports it)"
        )
    if mem.mixed_precision and prec.model_dtype != "float32":
        raise V2ConfigError(
            "memory.mixed_precision=true expects fp32 master weights; "
            f"precision.model_dtype is '{prec.model_dtype}'"
        )
    for field_name in ("vector_dtype", "activity_dtype"):
        if getattr(prec, field_name) == "bfloat16" and mem.storage in ("cpu", "memmap"):
            raise V2ConfigError(
                f"precision.{field_name}='bfloat16' requires memory.storage='gpu': "
                f"NumPy-backed '{mem.storage}' storage has no bfloat16 dtype"
            )
    if vec.temporal_resolution > sim.n_bins:
        raise V2ConfigError(
            f"vector.temporal_resolution={vec.temporal_resolution} cannot exceed "
            f"simulation.n_bins={sim.n_bins}"
        )

    # -- not implemented in this stage (warnings, or errors when strict) -----
    unimplemented: list[str] = []
    if not net.multi_layer_implemented:
        unimplemented.append(f"model.n_layers={net.n_layers} (multi-layer SNN)")
    if vec.unimplemented_blocks:
        unimplemented.append(
            f"vector.enabled_blocks includes not-yet-implemented block(s) {vec.unimplemented_blocks}"
        )
    if prec.model_dtype != "float32":
        unimplemented.append(f"precision.model_dtype='{prec.model_dtype}' (mixed-precision training)")
    if unimplemented:
        message = "requested but not implemented in this configuration stage: " + "; ".join(unimplemented)
        if strict:
            raise V2ConfigError(message + " (use strict=False or remove the request)")
        warnings.append(message)

    # -- advisory warnings ---------------------------------------------------
    if mem.max_input_tokens > 0:
        train_tokens = mem.train_batch_size * sim.n_bins * v2.snn.n_input
        if train_tokens > mem.max_input_tokens:
            warnings.append(
                f"memory.train_batch_size({mem.train_batch_size}) * n_bins({sim.n_bins}) * "
                f"n_input({v2.snn.n_input}) = {train_tokens:,} tokens exceeds "
                f"memory.max_input_tokens={mem.max_input_tokens:,}"
            )
    if prec.model_dtype in ("float16", "bfloat16") and not mem.mixed_precision:
        warnings.append(
            f"precision.model_dtype='{prec.model_dtype}' without memory.mixed_precision=true "
            "is unusual; enable mixed precision or use float32"
        )
    return warnings
