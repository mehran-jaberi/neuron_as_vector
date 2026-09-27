"""Shared helpers for the ``scripts/`` entry points.

Kept deliberately small: argument parsing, config loading, device resolution,
dataset construction (real SHD or synthetic) and model (de)serialisation. Every
script adds the project root to ``sys.path`` so that ``import src...`` works
regardless of the current working directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import-time side effect
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (  # noqa: E402
    SHDRecordings,
    load_shd,
    make_synthetic_shd,
    make_train_dev_probe_split,
    make_validation_split,
    subset_by_class,
)
from src.model import RecurrentLIFSNN, architecture_mismatches, build_model  # noqa: E402
from src.utils import (  # noqa: E402
    Config,
    describe_device,
    ensure_dir,
    get_device,
    load_config,
    save_json,
    set_seed,
)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(description: str, default_config: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=default_config, help="path to a YAML config file")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted config override, e.g. --override train.epochs=5 (repeatable)",
    )
    parser.add_argument("--tag", default=None, help="run tag used in output filenames")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--synthetic", action="store_true", help="use the built-in synthetic dataset")
    parser.add_argument("--debug", action="store_true", help="tiny dataset mode for smoke tests")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def load_run_config(args: argparse.Namespace) -> Config:
    cfg = load_config(args.config, overrides=args.override)
    if args.tag is not None:
        cfg.set_path("run.tag", args.tag)
    if args.device is not None:
        cfg.set_path("run.device", args.device)
    if args.seed is not None:
        cfg.set_path("seed", int(args.seed))
    if args.synthetic:
        cfg.set_path("run.synthetic", True)
    if args.debug:
        cfg.set_path("run.debug", True)
    return cfg


# --------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------
def resolve_device(cfg: Config) -> Any:
    mode = str(cfg.get_path("run.device", "auto"))
    if mode == "cpu":
        import torch

        return torch.device("cpu")
    if mode == "cuda":
        device = get_device(prefer_cuda=True)
        if device.type != "cuda":
            print("[warn] run.device=cuda requested but CUDA is unavailable; falling back to CPU")
        return device
    return get_device(prefer_cuda=True)


def announce(cfg: Config, device: Any) -> None:
    info = describe_device(device)
    print(
        "[env] torch={torch} cuda_available={cuda_available} cuda={cuda_version} device={device}".format(**info)
    )
    if info.get("gpu_name"):
        print(f"[env] gpu={info['gpu_name']} ({info.get('gpu_memory_gb')} GB)")
    print(f"[env] seed={cfg.get_path('seed', 0)} tag={cfg.get_path('run.tag', 'run')}")


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
def result_path(cfg: Config, filename: str) -> Path:
    d = ensure_dir(PROJECT_ROOT / str(cfg.get_path("paths.results_dir", "results")))
    return d / filename


def checkpoint_path(cfg: Config, tag: str | None = None) -> Path:
    d = ensure_dir(PROJECT_ROOT / str(cfg.get_path("paths.checkpoint_dir", "checkpoints")))
    tag = tag or str(cfg.get_path("run.tag", "run"))
    return d / f"{tag}.pt"


def figure_dir(cfg: Config) -> Path:
    return ensure_dir(PROJECT_ROOT / str(cfg.get_path("paths.figures_dir", "figures")))


def display_path(path: Path | str) -> str:
    """Repo-relative path when possible, absolute otherwise (external output directories)."""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
def build_recordings(cfg: Config) -> dict[str, Any]:
    """Return ``{"train", "dev", "probe", "val", "test", "split_info", "name"}``.

    Four-way split semantics (see ``src.data.make_train_dev_probe_split``):

    * ``train`` - fitting (gradient updates).
    * ``dev``   - speaker-aware development split for model selection, carved from
      the official SHD training file.
    * ``probe`` - speaker-aware split carved from the official training file, locked
      after the fact and reported only as a final internal sanity check.
    * ``test``  - the official SHD test file, fully locked (never used for training
      or model selection).

    ``"val"`` is kept as a backward-compatible alias for ``"dev"`` so the existing
    representation/analysis scripts keep working.

    * Synthetic mode carves train/dev/probe/test out of one synthetic dataset (no
      scientific meaning; it only exercises the code paths).
    * Real mode uses the official SHD train/test files and splits the official
      training data into train/dev/probe with a speaker-aware split.
    """
    seed = int(cfg.get_path("seed", 0))
    # The data split may be decoupled from the model/optimisation seed so that a
    # multi-seed robustness study varies only initialisation / optimisation while
    # holding the train/dev/probe split (and hence the fingerprint target) fixed.
    # ``data.split_seed: null`` (the default) means "follow ``seed``".
    split_seed_raw = cfg.get_path("data.split_seed", None)
    split_seed = seed if split_seed_raw is None else int(split_seed_raw)
    data_dir = PROJECT_ROOT / str(cfg.get_path("paths.data_dir", "data"))
    dev_fraction = float(cfg.get_path("data.dev_fraction", 0.1))
    probe_fraction = float(cfg.get_path("data.probe_fraction", 0.1))
    prefer_speaker = bool(cfg.get_path("data.prefer_speaker_aware", True))
    debug = bool(cfg.get_path("run.debug", False))
    synthetic = bool(cfg.get_path("run.synthetic", False))

    if synthetic:
        n_channels = int(cfg.get_path("run.synthetic_n_channels", 40))
        n_classes = int(cfg.get_path("run.synthetic_n_classes", 5))
        # Auto-align the model/readout dimensions with the synthetic dataset so a
        # smoke test "just works". The input dimensionality MUST match the number
        # of synthetic channels and the readout MUST match the class count.
        cfg.set_path("model.n_input", n_channels)
        cfg.set_path("model.n_output", n_classes)
        cfg.set_path("train.n_classes", n_classes)
        rec = make_synthetic_shd(
            n_samples=int(cfg.get_path("run.synthetic_n_samples", 600)),
            n_classes=n_classes,
            n_channels=n_channels,
            n_bins=int(cfg.get_path("model.n_bins", 500)),
            bin_ms=float(cfg.get_path("model.bin_ms", 2.0)),
            seed=seed,
        )
        train_full, dev, probe, split_info = make_train_dev_probe_split(
            rec, dev_fraction=dev_fraction, probe_fraction=probe_fraction,
            seed=split_seed, prefer_speaker_aware=prefer_speaker,
        )
        # Carve a synthetic "test" set out of the training remainder (no meaning).
        n = len(train_full)
        rng_idx = list(range(n))
        import numpy as np

        rng = np.random.default_rng(split_seed + 1)
        rng.shuffle(rng_idx)
        n_test = max(1, int(0.2 * n))
        test_idx = rng_idx[:n_test]
        train_idx = rng_idx[n_test:]
        test = train_full.subset(test_idx, name="synthetic_test")
        train = train_full.subset(train_idx, name="synthetic_train")
        split_info = dict(split_info)
        split_info["synthetic_test_n"] = len(test)
        return {
            "train": train,
            "dev": dev,
            "probe": probe,
            "val": dev,
            "test": test,
            "split_info": split_info,
            "name": "synthetic",
        }

    data = load_shd(data_dir, download=bool(cfg.get_path("data.download", True)),
                    layout=str(cfg.get_path("data.layout", "auto")))
    train, dev, probe, split_info = make_train_dev_probe_split(
        data["train"], dev_fraction=dev_fraction, probe_fraction=probe_fraction,
        seed=split_seed, prefer_speaker_aware=prefer_speaker,
    )
    if debug:
        train = subset_by_class(train, max_per_class=40, seed=split_seed)
        dev = subset_by_class(dev, max_per_class=20, seed=split_seed)
        if len(probe) > 0:
            probe = subset_by_class(probe, max_per_class=20, seed=split_seed)
    max_per_class = cfg.get_path("data.max_per_class", None)
    if max_per_class:
        train = subset_by_class(train, max_per_class=int(max_per_class), seed=split_seed)
    return {
        "train": train,
        "dev": dev,
        "probe": probe,
        "val": dev,  # backward-compatible alias for the analysis scripts
        "test": data["test"],
        "split_info": split_info,
        "name": "shd",
    }


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_model_from_config(cfg: Config, *, seed: int, device: Any) -> RecurrentLIFSNN:
    model = build_model(cfg, seed=seed, device=None)
    return model.to(device)


# A config/checkpoint architecture mismatch is silent and scientifically invalid
# (e.g. the untrained "before learning" model would see a different time window
# than the trained one), so load_checkpoint turns it into a hard error.
def load_checkpoint(cfg: Config, device: Any, tag: str | None = None) -> tuple[RecurrentLIFSNN, dict[str, Any]]:
    path = checkpoint_path(cfg, tag)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Train one first with "
            f"`uv run python scripts/train.py --config configs/baseline.yaml`."
        )
    model, extra = RecurrentLIFSNN.load(str(path), map_location="cpu")
    mismatch = architecture_mismatches(cfg.get_path("model", {}) or {}, model)
    if mismatch:
        detail = ", ".join(f"model.{k}: config={a!r} vs checkpoint={b!r}" for k, (a, b) in mismatch.items())
        overrides = " ".join(f"--override model.{k}={b}" for k, (_, b) in mismatch.items())
        raise ValueError(
            f"The analysis config's model block does not match checkpoint '{path.name}' "
            f"({detail}). The architecture must agree, otherwise downstream comparisons "
            f"(e.g. the before/after-learning model) are silently invalid. Fix with: {overrides}"
        )
    return model.to(device), extra


def apply_seed(cfg: Config, device: Any) -> None:
    set_seed(int(cfg.get_path("seed", 0)))
