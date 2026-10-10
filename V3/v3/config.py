"""V3 configuration: the single place where every experiment parameter lives.

V3 tests the *original* "C" version of the neuron-as-vector idea: a neuron's
internal state is genuinely ``D``-dimensional and the ``D`` dimensions
participate directly in its own recurrent dynamics, in its interaction with the
other neurons, and in spike generation.  There is no scalar neuron anywhere in
the vector model and no post-hoc embedding of anything.

The whole experiment is described by :class:`V3Config`.  Nothing in the model,
the data loader, the training loop or the notebook hard-codes ``N`` or ``D``.

Primary configuration (see ``configs/v3_default.yaml``)::

    N = 64     D = 1000     CUDA     float16
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

V3_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = V3_ROOT.parent


@dataclass
class StateRegularizationConfig:
    """Controlled imprecision applied to the neuron internal state ``z(t)``.

    This is **not** about the numerical datatype of the model.  ``V3Config.dtype``
    / ``amp`` decide how CUDA stores and multiplies numbers (fp16 state, fp32
    masters, autocast/GradScaler); this object decides whether the *internal
    vector state itself* is deliberately perturbed or quantized.

    Three separate knobs, kept explicit:

    1. training arithmetic dtype  -> ``V3Config.dtype`` / ``amp``
    2. precision of the state      -> ``quantize_bits`` here
    3. stochastic perturbation     -> ``noise_std`` here

    Modes
    -----
    ``none``               state untouched (baseline; default).
    ``noise``              ``z <- z + N(0, std^2)``.
    ``quantization``       ``z <- Q(z)`` symmetric straight-through quantizer.
    ``noise_quantization`` quantize first, then add noise.

    The ``apply_during_*`` flags choose the phase.  Default is **training only**:
    the perturbation acts as a training-time regularizer and inference (validation
    / the official test set) stays at full precision unless explicitly requested.
    """

    mode: str = "none"
    noise_std: float = 0.0        # 0.001 .. 0.05 for the screening sweep
    quantize_bits: int = 8        # 16 / 12 / 8 / 6 / 4 for the screening sweep
    quantize_clip: float = 1.0    # max magnitude of z (tanh bound; safety guard)
    apply_during_training: bool = True
    apply_during_validation: bool = False
    apply_during_test: bool = False

    MODES = ("none", "noise", "quantization", "noise_quantization")

    @property
    def uses_noise(self) -> bool:
        return self.mode in ("noise", "noise_quantization")

    @property
    def uses_quantization(self) -> bool:
        return self.mode in ("quantization", "noise_quantization")

    @property
    def noise_enabled(self) -> bool:
        """Noise is genuinely active (mode selects it and the std is positive)."""
        return self.uses_noise and float(self.noise_std) > 0.0

    @property
    def quantization_enabled(self) -> bool:
        return self.uses_quantization

    def phase_active(self, phase: str) -> bool:
        """Whether the perturbation is applied in ``phase`` (train | val | test)."""
        if self.mode == "none":
            return False
        if phase == "train":
            return bool(self.apply_during_training)
        if phase == "val":
            return bool(self.apply_during_validation)
        if phase == "test":
            return bool(self.apply_during_test)
        raise ValueError(f"unknown phase {phase!r} (expected train | val | test)")

    def validate(self) -> None:
        if self.mode not in self.MODES:
            raise ValueError(f"state_regularization.mode must be one of {self.MODES}, got {self.mode!r}")
        if float(self.noise_std) < 0.0:
            raise ValueError("state_regularization.noise_std must be >= 0")
        if not 2 <= int(self.quantize_bits) <= 16:
            raise ValueError("state_regularization.quantize_bits must be in [2, 16]")
        if float(self.quantize_clip) <= 0.0:
            raise ValueError("state_regularization.quantize_clip must be > 0")

    @classmethod
    def from_dict(cls, data: dict, strict: bool = False) -> "StateRegularizationConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown and strict:
            raise ValueError(f"unknown state_regularization keys: {sorted(unknown)}")
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class TimingConfig:
    """Temporal discretization of the SHD experiment.

    Two independent quantities define the timing; the number of dynamical steps
    follows from them::

        num_time_steps = sequence_duration_ms / time_bin_ms     (must be an integer)

    ``time_bin_ms`` is simultaneously the SHD bin width **and** the model's
    integration step ``dt`` (one bin == one dynamical step), so the leak factor
    ``alpha = dt/tau`` changes with it: 2 ms -> ``alpha = 2/tau``,
    4 ms -> ``alpha = 4/tau``.

    Defaults are the **2 ms** discretization
    (1000 ms / 2 ms -> 500 steps, ``alpha = 0.1`` at ``tau = 20 ms``).  The
    historical 4 ms reference (250 steps, ``alpha = 0.2``) is still available by
    setting ``time_bin_ms: 4``.
    """

    sequence_duration_ms: float = 1000.0
    time_bin_ms: float = 2.0

    @property
    def num_time_steps(self) -> int:
        return int(round(float(self.sequence_duration_ms) / float(self.time_bin_ms)))

    @property
    def simulation_dt_ms(self) -> float:
        return float(self.time_bin_ms)

    @property
    def duration_s(self) -> float:
        return float(self.sequence_duration_ms) / 1000.0

    def validate(self) -> None:
        if float(self.time_bin_ms) <= 0:
            raise ValueError("timing.time_bin_ms must be > 0")
        if float(self.sequence_duration_ms) <= 0:
            raise ValueError("timing.sequence_duration_ms must be > 0")
        steps = float(self.sequence_duration_ms) / float(self.time_bin_ms)
        if abs(steps - round(steps)) > 1e-9:
            raise ValueError(
                "timing.sequence_duration_ms / time_bin_ms must be an integer number of "
                f"steps, got {self.sequence_duration_ms}/{self.time_bin_ms} = {steps}"
            )
        if round(steps) < 1:
            raise ValueError("timing must yield at least one time step")

    @classmethod
    def from_dict(cls, data: dict, strict: bool = False) -> "TimingConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown and strict:
            raise ValueError(f"unknown timing keys: {sorted(unknown)}")
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class V3Config:
    """All V3 experiment parameters."""

    # ---- run identity ----------------------------------------------------
    tag: str = "v3"
    seed: int = 0

    # ---- data (SHD) ------------------------------------------------------
    train_h5: str = "data/shd_train.h5"
    test_h5: str = "data/shd_test.h5"
    n_inputs: int = 700
    n_classes: int = 20
    # temporal discretization: the step count T is *derived*, never stored
    timing: TimingConfig = field(default_factory=TimingConfig)
    binary_input: bool = True    # >=1 event in a bin -> 1 (standard SHD encoding)
    val_fraction: float = 0.10   # stratified validation split, taken from the TRAIN file
    split_seed: int = 0
    cache_events: bool = True    # cache binned event codes in RAM (faster epochs)
    max_train_samples: int = 0   # 0 = use the whole training split (subset only for quick runs)
    shuffle_train: bool = True   # shuffle FIT batches between epochs
    shuffle_val: bool = False    # validation order is deterministic

    # ---- model -----------------------------------------------------------
    n_neurons: int = 64          # N
    state_dim: int = 1000        # D
    mix_rank: int = 64           # rank of the shared intra-neuron mixing map
    tau_ms: float = 20.0         # state leak time constant [ms]
    input_gain: float = 5.0      # scale of the shared input projection init
    surrogate_beta: float = 4.0  # fast-sigmoid surrogate slope
    surrogate_gamma: float = 1.0 # fast-sigmoid surrogate gain
    readout: str = "sum"         # "sum" (spike count) | "mean"
    rate_reg: float = 1.0e-4     # weight of the firing-rate regulariser (0 disables)
    rate_target_hz: float = 10.0
    emit_identity_init: bool = True  # init W_emit ~ I (+ small noise)

    # ---- training --------------------------------------------------------
    batch_size: int = 32
    epochs: int = 5             # primary control of how much training is done
    max_train_batches: int | None = None  # None = all batches; int = cap per epoch
    learning_rate: float = 2.0e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    optimizer: str = "adam"
    lr_schedule: str = "cosine"  # "cosine" | "none"
    warmup_frac: float = 0.03
    label_smoothing: float = 0.0

    # ---- precision / device ---------------------------------------------
    device: str = "cuda"
    dtype: str = "float16"       # dtype of the neural state / major model tensors
    amp: bool = True             # autocast + GradScaler for fp32 master params
    grad_checkpoint_chunks: int = 10  # split T into this many checkpointed chunks
    select_best_val: bool = True      # keep the best-validation-accuracy epoch as secondary

    # ---- controlled state imprecision (disabled by default) --------------
    # This is the *only* new scientific manipulation in the imprecision stage: it
    # deliberately perturbs/quantizes the neuron's internal vector state z(t).
    # It is independent of `dtype`/`amp` (the training arithmetic).  Default
    # `mode="none"` reproduces the baseline exactly.
    state_regularization: StateRegularizationConfig = field(
        default_factory=StateRegularizationConfig
    )

    # ---- io --------------------------------------------------------------
    out_dir: str = "results"        # resolved relative to V3/ when not absolute
    ckpt_dir: str = "checkpoints"
    figure_dir: str = "figures"
    log_every: int = 0           # >0 prints every k batches (in addition to tqdm)

    # ------------------------------------------------------------------ #
    # derived quantities
    # ------------------------------------------------------------------ #
    @property
    def bin_ms(self) -> float:
        """Width of one temporal bin [ms] (== ``timing.time_bin_ms``)."""
        return float(self.timing.time_bin_ms)

    @property
    def n_bins(self) -> int:
        """Number of temporal steps T, derived from the timing block."""
        return int(self.timing.num_time_steps)

    @property
    def time_bin_ms(self) -> float:
        return self.bin_ms

    @property
    def sequence_duration_ms(self) -> float:
        return float(self.timing.sequence_duration_ms)

    @property
    def num_time_steps(self) -> int:
        return self.n_bins

    @property
    def dt_ms(self) -> float:
        """Integration step.  One bin == one dynamical step."""
        return float(self.timing.simulation_dt_ms)

    @property
    def simulation_dt_ms(self) -> float:
        return self.dt_ms

    @property
    def alpha(self) -> float:
        """Discrete leak factor ``dt / tau`` of the state update."""
        return min(1.0, float(self.dt_ms) / float(self.tau_ms))

    @property
    def duration_s(self) -> float:
        return self.timing.duration_s

    @property
    def window_s(self) -> float:
        return self.timing.duration_s

    @property
    def torch_dtype(self):
        import torch

        return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
            str(self.dtype).lower()
        ]

    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        if self.n_neurons < 1 or self.state_dim < 1:
            raise ValueError("n_neurons and state_dim must be >= 1")
        self.timing.validate()
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError("val_fraction must be in [0, 1)")
        if self.readout not in ("sum", "mean"):
            raise ValueError("readout must be 'sum' or 'mean'")
        if self.optimizer not in ("adam", "adamw", "sgd"):
            raise ValueError("optimizer must be adam | adamw | sgd")
        if self.lr_schedule not in ("cosine", "none"):
            raise ValueError("lr_schedule must be cosine | none")
        if self.mix_rank < 1:
            raise ValueError("mix_rank must be >= 1")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha = dt/tau must be in (0, 1]")
        if self.grad_checkpoint_chunks < 1:
            raise ValueError("grad_checkpoint_chunks must be >= 1")
        if self.max_train_batches is not None and int(self.max_train_batches) < 1:
            raise ValueError("max_train_batches must be null (all batches) or >= 1")
        self.state_regularization.validate()

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        d = asdict(self)
        d["derived"] = {
            "sequence_duration_ms": self.sequence_duration_ms,
            "time_bin_ms": self.time_bin_ms,
            "num_time_steps": self.num_time_steps,
            "simulation_dt_ms": self.simulation_dt_ms,
            "dt_ms": self.dt_ms,
            "alpha": self.alpha,
            "duration_s": self.duration_s,
            "window_s": self.window_s,
        }
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def resolved(self) -> "V3Config":
        """Validated copy of this configuration."""
        self.validate()
        return dataclasses.replace(self)

    def path(self, value: str) -> Path:
        """Resolve a data path relative to the repository root."""
        p = Path(value)
        return p if p.is_absolute() else (REPO_ROOT / p)

    def v3_path(self, value: str) -> Path:
        """Resolve an output path relative to the ``V3/`` folder."""
        p = Path(value)
        return p if p.is_absolute() else (V3_ROOT / p)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, data: dict, strict: bool = False) -> "V3Config":
        data = dict(data)
        # backward compatibility: derive a timing block from the legacy flat
        # `bin_ms` / `n_bins` keys if no explicit `timing` block is present
        if "timing" not in data and ("bin_ms" in data or "n_bins" in data):
            bin_ms = float(data.pop("bin_ms", 4.0))
            n_bins = int(data.pop("n_bins", 250))
            data["timing"] = {
                "time_bin_ms": bin_ms,
                "sequence_duration_ms": bin_ms * n_bins,
            }
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known - {"derived"}
        if unknown and strict:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        kwargs = {}
        for k, v in data.items():
            if k not in known:
                continue
            if k == "state_regularization" and isinstance(v, dict):
                v = StateRegularizationConfig.from_dict(v, strict=strict)
            elif k == "timing" and isinstance(v, dict):
                v = TimingConfig.from_dict(v, strict=strict)
            kwargs[k] = v
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path, strict: bool = True) -> "V3Config":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data, strict=strict)

    def with_overrides(self, **overrides) -> "V3Config":
        return dataclasses.replace(self, **overrides)


def describe(cfg: V3Config) -> str:
    """Short human-readable summary used in logs and notebook output."""
    reg = cfg.state_regularization
    return (
        f"V3Config(tag={cfg.tag!r}, N={cfg.n_neurons}, D={cfg.state_dim}, "
        f"T={cfg.num_time_steps}x{cfg.time_bin_ms:g}ms={cfg.sequence_duration_ms:g}ms, "
        f"batch={cfg.batch_size}, epochs={cfg.epochs}, lr={cfg.learning_rate:g}, "
        f"device={cfg.device}, dtype={cfg.dtype}, state_reg={reg.mode})"
    )


def parameter_groups(cfg: V3Config) -> dict:
    """Analytic parameter budget of the primary model (used for reporting)."""
    N, D, C, R = cfg.n_neurons, cfg.state_dim, cfg.n_inputs, min(cfg.mix_rank, cfg.state_dim)
    groups = {
        "w_in": C * D,
        "w_down": D * R,
        "w_up": R * D,
        "w_emit": D * D,
        "w_rec": N * N,
        "gate": N * D,
        "bias": N * D,
        "w_out": N * D,
        "theta": N,
        "w_cls": N * cfg.n_classes,
    }
    groups["total"] = sum(groups.values())
    return groups


__all__ = [
    "V3Config",
    "TimingConfig",
    "StateRegularizationConfig",
    "V3_ROOT",
    "REPO_ROOT",
    "describe",
    "parameter_groups",
    "math",
]
