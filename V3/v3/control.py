"""Programmatic control panel: temporary configuration overrides for one run.

The "control panel" is the notebook cell that lets the operator change a few
knobs (epochs, timing, ``N``, ``D``, seed, state regularization, augmentation)
for one session without editing ``configs/v3_default.yaml``.  This module is the
thin layer behind it and the programmatic equivalent of the CLI's
``--override``::

    # one-off experiment, same registry row as a CLI run
    from v3.control import run_configured
    run_configured({"epochs": 20, "timing.time_bin_ms": 2.0})

    # exactly the CLI defaults (bit-identical to today's default run)
    run_configured()

There is **no second configuration system here**: the override grammar
(``key=value`` items, dotted keys, type coercion) lives in :mod:`v3.config`
(``apply_override_items`` / ``apply_override_mapping``) and the training loop
lives in :mod:`v3.experiment` / :mod:`v3.train`, exactly as for
``V3/run_experiment.py``.  This module only concatenates those pieces:

    YAML defaults -> shared override grammar -> (optional) CLI quick config
                  -> v3.experiment.train_variant -> v3.registry.register_run

The YAML file is opened read-only; the *merged* configuration is what reaches
``train_variant`` and therefore what ``register_run`` writes into the run's
``config.yaml``.
"""

from __future__ import annotations

import dataclasses
import difflib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config import (
    V3_ROOT,
    V3Config,
    apply_override_items,
    apply_override_mapping,
)
from .experiment import resolve_device, test_variant
from .registry import RunRegistry

#: Same file the CLI loads by default (``run_experiment.DEFAULT_CONFIG``).
DEFAULT_CONFIG = V3_ROOT / "configs" / "v3_default.yaml"

#: Variant names accepted by :func:`run_configured` (``None`` == the CLI default).
VARIANTS = ("vector", "scalar", "both")


# ---------------------------------------------------------------------- #
# configuration layer (no training, safe to call from tests)
# ---------------------------------------------------------------------- #
def _leaf_paths(cfg: Any, prefix: str = "") -> list[str]:
    """Every dotted path that addresses a scalar leaf of ``cfg``.

    Reflection over the dataclass fields (rather than a hard-coded list), so a
    new config field -- e.g. the ``augmentation`` block -- is addressable
    through the same mapping without touching this module.
    """
    paths: list[str] = []
    for f in dataclasses.fields(cfg):
        value = getattr(cfg, f.name)
        path = f"{prefix}{f.name}"
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            paths.extend(_leaf_paths(value, f"{path}."))
        else:
            paths.append(path)
    return paths


def valid_override_paths(cfg: V3Config | None = None) -> list[str]:
    """Every override key accepted by :func:`apply_overrides` for ``cfg``."""
    return _leaf_paths(cfg if cfg is not None else V3Config())


def _override_keys(overrides: Mapping | Sequence[str]) -> list[str]:
    if isinstance(overrides, Mapping):
        return [str(k) for k in overrides]
    return [str(item).split("=", 1)[0].strip() for item in overrides]


def _with_suggestion(message: str, keys: list[str], cfg: V3Config) -> str:
    """Append the closest valid field to an unknown-key error message."""
    lowered = message.lower()
    if "closest" in lowered or "did you mean" in lowered:
        return message
    valid = valid_override_paths(cfg)
    valid_set = set(valid)
    candidates = [k for k in keys if k not in valid_set]
    # prefer the key the message actually names, if any
    named = [k for k in candidates if k in message]
    probe = named[0] if named else (candidates[0] if len(candidates) == 1 else None)
    if probe is None:
        return message
    close = difflib.get_close_matches(probe, valid, n=1, cutoff=0.6)
    if close:
        return f"{message}; closest valid field: {close[0]!r}"
    return f"{message}; no close match among the configuration fields"


def apply_overrides(cfg: V3Config, overrides: Mapping | Sequence[str] | None) -> V3Config:
    """Apply temporary overrides on top of ``cfg`` with the CLI's grammar.

    ``overrides`` is either a mapping of dotted keys to values
    (``{"timing.time_bin_ms": 2.0}``) or a list of ``"key=value"`` strings
    (``["epochs=20"]``); ``None``/empty returns ``cfg`` unchanged.  Values are
    coerced to the field's declared type by the shared grammar, so
    ``{"epochs": "20"}`` yields ``int(20)``.

    Unknown keys raise :class:`ValueError` naming the key and the closest valid
    field.
    """
    if overrides is None:
        return cfg
    if isinstance(overrides, Mapping):
        mapping = dict(overrides)
        if not mapping:
            return cfg
        try:
            return apply_override_mapping(cfg, mapping)
        except (ValueError, KeyError, SystemExit) as exc:  # CLI raises SystemExit
            raise ValueError(_with_suggestion(str(exc), _override_keys(mapping), cfg)) from None
    if isinstance(overrides, Sequence) and not isinstance(overrides, (str, bytes)):
        items = [str(item) for item in overrides]
        if not items:
            return cfg
        try:
            return apply_override_items(cfg, items)
        except (ValueError, KeyError, SystemExit) as exc:
            raise ValueError(_with_suggestion(str(exc), _override_keys(items), cfg)) from None
    raise TypeError(
        "overrides must be a mapping of dotted keys or a list of 'key=value' strings, "
        f"got {type(overrides).__name__}"
    )


def build_config(
    overrides: Mapping | Sequence[str] | None = None,
    quick: bool = False,
    config_file: str | Path | None = None,
) -> V3Config:
    """``configs/v3_default.yaml`` + temporary overrides (+ the CLI quick profile).

    The YAML file is only read.  ``quick=True`` applies the *same* quick
    overrides the CLI applies (``run_experiment.QUICK_OVERRIDES``), so a quick
    merged config equals ``V3/run_experiment.py --quick`` for the same keys.
    The returned configuration is validated (``V3Config.resolved``).
    """
    path = Path(config_file) if config_file is not None else DEFAULT_CONFIG
    cfg = V3Config.from_yaml(path)
    cfg = apply_overrides(cfg, overrides)
    if quick:
        # imported lazily: run_experiment is a top-level script next to the package
        from run_experiment import QUICK_OVERRIDES

        cfg = cfg.with_overrides(**QUICK_OVERRIDES)
    return cfg.resolved()


# ---------------------------------------------------------------------- #
# run layer (delegates to the CLI's own functions; no training loop here)
# ---------------------------------------------------------------------- #
def _variant_names(variant: str | None) -> list[str]:
    if variant is None:
        return ["vector"]  # the CLI default (--variant vector)
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS} or None, got {variant!r}")
    return ["vector", "scalar"] if variant == "both" else [variant]


def run_configured(
    overrides: Mapping | Sequence[str] | None = None,
    quick: bool = False,
    variant: str | None = None,
    register: bool = True,
    *,
    config_file: str | Path | None = None,
    device: str | None = None,
    show_progress: bool = True,
    test: bool = True,
) -> dict:
    """Train, test and (optionally) register one configured run.

    Same code path as ``V3/run_experiment.py``: the config is loaded the same
    way, the training/evaluation calls are ``run_experiment.run_variant`` and
    ``v3.experiment.test_variant``, and registration goes through
    ``run_experiment.record_run`` -> ``v3.registry.register_run`` (same row
    schema, same timestamped ``V3/results/<run_id>/`` directory, same tags).
    ``overrides=None`` is therefore the default CLI run.

    Parameters
    ----------
    overrides : mapping of dotted keys or list of ``"key=value"`` strings
    quick : apply the CLI's quick profile (small ``N``/``D``/subset)
    variant : ``"vector"`` (default), ``"scalar"`` or ``"both"``
    register : write the registry row + run directory (default ``True``)
    config_file : alternative YAML file (default ``configs/v3_default.yaml``)
    device : override the device string (as the CLI's ``--device``)
    test : evaluate the official test set (default ``True``)

    Returns
    -------
    dict with ``config`` (the merged config), ``quick``, ``device``,
    ``variants`` (tag -> info dict), ``test_metrics`` (tag -> metrics or None),
    ``run_dirs`` (tag -> path or None) and ``registry`` (``RunRegistry`` or None).
    """
    # imported lazily: run_experiment is a top-level script next to the package
    from run_experiment import apply_variant, record_run, run_variant

    cfg = build_config(overrides, quick=quick, config_file=config_file)
    if device is not None:
        cfg = cfg.with_overrides(device=str(device))
    dev = resolve_device(cfg)

    registry = RunRegistry(cfg.v3_path(cfg.out_dir)) if register else None
    result: dict = {
        "config": cfg,
        "quick": bool(quick),
        "device": dev,
        "variants": {},
        "test_metrics": {},
        "run_dirs": {},
        "registry": registry,
    }
    for name in _variant_names(variant):
        vcfg = apply_variant(cfg, name)
        info = run_variant(vcfg, dev, show_progress=show_progress)
        info["quick"] = bool(quick)
        variant_obj = info.pop("result")
        tm = test_variant(variant_obj, which="final", show_progress=show_progress) if test else None
        run_dir = record_run(vcfg, variant_obj, info, tm, registry) if register else None
        result["variants"][vcfg.tag] = info
        result["test_metrics"][vcfg.tag] = tm
        result["run_dirs"][vcfg.tag] = run_dir
        del variant_obj
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - CPU-only installs / no CUDA
            pass
    return result


__all__ = [
    "DEFAULT_CONFIG",
    "VARIANTS",
    "apply_overrides",
    "build_config",
    "run_configured",
    "valid_override_paths",
]
