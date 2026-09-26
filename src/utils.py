"""Shared utilities: configuration, seeding, device selection, paths, standardisation.

Everything in this module is deliberately dependency-light (stdlib + numpy) so
that it can be imported by every other module and by the tests.
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, get_type_hints

import numpy as np
import yaml

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# PROJECT_ROOT is derived from this file's location, never from the user's home
# directory or the current working directory. All other paths are built from it.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
CONFIG_DIR: Path = PROJECT_ROOT / "configs"
CHECKPOINT_DIR: Path = PROJECT_ROOT / "checkpoints"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Create ``path`` (and parents) if needed and return it as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
class Config(dict):
    """A small dict subclass with attribute-style access and dotted lookup.

    ``cfg.hidden_size`` and ``cfg["hidden_size"]`` are equivalent, and
    ``cfg.get_path("train.lr")`` walks nested dictionaries. Missing keys raise
    unless ``default`` is supplied.
    """

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - trivial
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, Mapping) and not isinstance(value, Config):
            return Config(value)
        if isinstance(value, list):
            return [Config._wrap(v) for v in value]
        return value

    def __getitem__(self, key: str) -> Any:
        return self._wrap(dict.__getitem__(self, key))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in self.items():
            if isinstance(value, Mapping):
                out[key] = Config(value).to_dict()
            elif isinstance(value, list):
                out[key] = [Config(v).to_dict() if isinstance(v, Mapping) else v for v in value]
            else:
                out[key] = value
        return out

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """Walk a dotted path such as ``"train.lr"``; return ``default`` if absent."""
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return self._wrap(node)

    def require(self, dotted: str) -> Any:
        sentinel = object()
        value = self.get_path(dotted, sentinel)
        if value is sentinel:
            raise KeyError(f"Missing required configuration key: {dotted!r}")
        return value

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node: Any = self
        for part in parts[:-1]:
            # Fetch the *actual* nested mapping (by reference) instead of wrapping
            # it in a fresh ``Config``: ``Config(existing)`` copies the top-level
            # keys, so mutating the copy would silently discard the assignment.
            nxt = node.get(part) if isinstance(node, Mapping) else None
            if not isinstance(nxt, Mapping):
                nxt = Config()
                node[part] = nxt
            node = nxt
        node[parts[-1]] = value

    def save(self, path: str | os.PathLike[str]) -> Path:
        p = Path(path)
        ensure_dir(p.parent)
        with p.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)
        return p


def parse_scalar(text: str) -> Any:
    """Parse a CLI override value into a Python scalar.

    YAML 1.1 only recognises floats written with a decimal point, so a value like
    ``1e-3`` is returned as the *string* ``'1e-3'``. That silently produced a
    string where a float was expected (e.g. ``train.l2_spikes``) and crashed
    training with ``'>' not supported between instances of 'str' and 'int'``.
    Here we additionally coerce numeric-looking strings with a strict, full-match
    regex so that legitimate non-numeric strings (``reset=subtract``) are left
    untouched.
    """
    value = yaml.safe_load(text)
    if not isinstance(value, str):
        return value
    s = value.strip()
    if s == "":
        return value
    if re.fullmatch(r"[+-]?\d+", s):
        return int(s)
    if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?", s):
        try:
            return float(s)
        except ValueError:  # pragma: no cover - regex already guarantees a float
            return value
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none", "~"):
        return None
    return value


def _coerce_value(value: Any, annotation: Any) -> Any:
    """Best-effort coercion of a config value to a dataclass field's type."""
    if value is None or annotation is None:
        return value
    try:
        if annotation is bool:
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        if annotation is int:
            return int(float(value)) if isinstance(value, str) else int(value)
        if annotation is float:
            return float(value)
        if annotation is str:
            return str(value)
    except (TypeError, ValueError):
        return value
    return value


def coerce_dataclass_kwargs(cls: type, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Coerce ``kwargs`` to the annotated types of dataclass ``cls``.

    Guards against values arriving as strings (YAML 1.1 quirks, CLI overrides)
    where a number is expected.
    """
    try:
        hints = get_type_hints(cls)
    except Exception:  # pragma: no cover - defensive
        return dict(kwargs)
    return {k: _coerce_value(v, hints.get(k)) for k, v in dict(kwargs).items()}


def load_config(path: str | os.PathLike[str], overrides: Sequence[str] | None = None) -> Config:
    """Load a YAML config file and optionally apply ``key.subkey=value`` overrides.

    Override values are parsed as YAML scalars (with an extra numeric fallback, see
    :func:`parse_scalar`) so that ``lr=1e-3`` becomes a float, ``hidden_size=512``
    an int, ``use_cuda=true`` a bool, and ``blocks=[input_conn,activity]`` a list.
    """
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"Config file {path} must contain a YAML mapping at the top level.")
    cfg = Config(raw)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override {override!r} must have the form key.subkey=value")
        key, _, value = override.partition("=")
        cfg.set_path(key.strip(), parse_scalar(value))
    return cfg


def merge_configs(base: Mapping[str, Any], override: Mapping[str, Any]) -> Config:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    out = Config({k: v for k, v in base.items()})
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = merge_configs(out[key], value)
        else:
            out[key] = value
    return out


# --------------------------------------------------------------------------
# Seeding / device
# --------------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG we depend on.

    ``deterministic=True`` additionally asks cuDNN for deterministic kernels.
    Full bitwise reproducibility on GPU is not guaranteed by PyTorch; we
    document this limitation in the README.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - torch is a hard dependency in practice
        pass


def get_device(prefer_cuda: bool = True) -> "torch.device":  # noqa: F821
    """Return a CUDA device when available and requested, otherwise CPU."""
    import torch

    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_device(device: "torch.device | None" = None) -> dict[str, Any]:  # noqa: F821
    """Report torch/CUDA versions and GPU name for logging and provenance files."""
    import torch

    info: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device": str(device) if device is not None else None,
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        info["gpu_memory_gb"] = round(props.total_memory / 1024**3, 2)
    return info


# --------------------------------------------------------------------------
# JSON helpers
# --------------------------------------------------------------------------
def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def save_json(obj: Any, path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    return p


def load_json(path: str | os.PathLike[str]) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# Standardisation
# --------------------------------------------------------------------------
@dataclass
class Standardizer:
    """Column-wise standardisation that survives zero-variance columns.

    Constant columns (e.g. an intrinsic block when every neuron shares the same
    threshold) are centred to a constant zero and marked as non-informative
    instead of producing NaNs. This is important because the whole point of the
    project is that some feature blocks may carry no information at all.
    """

    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    constant_mask: np.ndarray | None = None

    def fit(self, X: np.ndarray, eps: float = 1e-8) -> "Standardizer":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected a 2-D array, got shape {X.shape}")
        self.mean = X.mean(axis=0)
        std = X.std(axis=0, ddof=0)
        self.constant_mask = std < eps
        self.scale = np.where(self.constant_mask, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("Standardizer must be fitted before calling transform().")
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean) / self.scale

    def fit_transform(self, X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        return self.fit(X, eps=eps).transform(X)

    @property
    def n_informative(self) -> int:
        if self.constant_mask is None:
            return 0
        return int((~self.constant_mask).sum())


def sanitize_features(arr: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Replace non-finite values and return a float64 copy.

    Feature extraction is defensive: any NaN/Inf (e.g. from a division by zero
    when a neuron never spikes) is replaced by ``fill`` so that downstream
    distance computations are always well defined.
    """
    out = np.asarray(arr, dtype=np.float64).copy()
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = fill
    return out


def class_balanced_indices(labels: np.ndarray, max_per_class: int, seed: int = 0) -> np.ndarray:
    """Deterministically subsample at most ``max_per_class`` items per class."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    chosen: list[np.ndarray] = []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        if idx.size > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        chosen.append(np.sort(idx))
    return np.concatenate(chosen) if chosen else np.empty(0, dtype=int)


def format_table(rows: Iterable[Sequence[Any]], headers: Sequence[str]) -> str:
    """Render a simple fixed-width text table (used for console summaries)."""
    rows = [[str(c) for c in r] for r in rows]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*headers), fmt.format(*["-" * w for w in widths])]
    lines += [fmt.format(*r) for r in rows]
    return "\n".join(lines)
