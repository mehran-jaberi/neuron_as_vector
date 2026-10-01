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
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

V3_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = V3_ROOT.parent


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
    bin_ms: float = 4.0          # width of one temporal bin [ms]
    n_bins: int = 250            # number of temporal steps T (250 * 4 ms = 1000 ms)
    binary_input: bool = True    # >=1 event in a bin -> 1 (standard SHD encoding)
    val_fraction: float = 0.10   # stratified validation split, taken from the TRAIN file
    split_seed: int = 0
    cache_events: bool = True    # cache binned event codes in RAM (faster epochs)
    max_train_samples: int = 0   # 0 = use the whole training split (subset only for quick runs)

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
    epochs: int = 20             # primary control of how much training is done
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

    # ---- io --------------------------------------------------------------
    out_dir: str = "results"        # resolved relative to V3/ when not absolute
    ckpt_dir: str = "checkpoints"
    figure_dir: str = "figures"
    log_every: int = 0           # >0 prints every k batches (in addition to tqdm)

    # ------------------------------------------------------------------ #
    # derived quantities
    # ------------------------------------------------------------------ #
    @property
    def dt_ms(self) -> float:
        """Integration step.  One bin == one dynamical step."""
        return float(self.bin_ms)

    @property
    def alpha(self) -> float:
        """Discrete leak factor ``dt / tau`` of the state update."""
        return min(1.0, float(self.dt_ms) / float(self.tau_ms))

    @property
    def duration_s(self) -> float:
        return self.n_bins * self.bin_ms / 1000.0

    @property
    def window_s(self) -> float:
        return self.n_bins * self.bin_ms / 1000.0

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
        if self.n_bins < 1 or self.bin_ms <= 0:
            raise ValueError("n_bins and bin_ms must be positive")
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

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        d = asdict(self)
        d["derived"] = {
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
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known - {"derived"}
        if unknown and strict:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        kwargs = {k: v for k, v in data.items() if k in known}
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
    return (
        f"V3Config(tag={cfg.tag!r}, N={cfg.n_neurons}, D={cfg.state_dim}, "
        f"T={cfg.n_bins}x{cfg.bin_ms:g}ms={cfg.duration_s:g}s, "
        f"batch={cfg.batch_size}, epochs={cfg.epochs}, lr={cfg.learning_rate:g}, "
        f"device={cfg.device}, dtype={cfg.dtype})"
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


__all__ = ["V3Config", "V3_ROOT", "REPO_ROOT", "describe", "parameter_groups", "math"]
