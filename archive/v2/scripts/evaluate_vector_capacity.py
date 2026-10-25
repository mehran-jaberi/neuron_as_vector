"""Scientific capacity evaluation of the neuron-vector representation (PROBE).

End-to-end, reproducible entry point for the V2 capacity study. It reuses the repository's
existing scientific machinery and adds no new statistics:

```
frozen SNN checkpoint + label-free FIT  ->  NeuronRecordBank  ->  structured / +residual vectors
PROBE (held out)                        ->  functional targets  ->  existing geometry / prediction / controls
```

Stages: load the canonical configuration and the **frozen** checkpoint (never re-trained here) ->
build the label-free FIT record bank (one streamed activity pass, cached) -> train/cache the
learned residual(s) on FIT only -> collect PROBE activity **once** (cached) -> build the
functional targets -> freeze the representations -> evaluate every condition and control ->
write ``results.csv``/``results.json``/``metadata.json`` + three figures.

Split discipline: FIT builds the representation; PROBE builds the targets and the metrics; the
official TEST file is **never opened** by this script.

Examples
--------
    uv run python scripts/evaluate_vector_capacity.py
    uv run python scripts/evaluate_vector_capacity.py --structured-dims 48,64 --residual-conditions 48:16
    uv run python scripts/evaluate_vector_capacity.py --quick --override data.max_per_class=40
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from _common import PROJECT_ROOT, display_path, load_config  # noqa: F401  (adds the project root to sys.path)

from src.capacity_figures import write_all_figures  # noqa: E402
from src.data import (  # noqa: E402
    load_shd_recordings,
    make_synthetic_shd,
    make_train_dev_probe_split,
    subset_by_class,
)
from src.evaluation import ActivityAccumulatorResult, collect_activity  # noqa: E402
from src.model import RecurrentLIFSNN, architecture_mismatches  # noqa: E402
from src.neuron_record import build_neuron_record_bank  # noqa: E402
from src.residual import (  # noqa: E402
    ResidualError,
    ResidualResult,
    ResidualSourceConfig,
    ResidualTrainingConfig,
    build_residual_source,
    train_residual,
)
from src.utils import get_device, save_json  # noqa: E402
from src.v2_config import V2Config, V2ConfigError  # noqa: E402
from src.vector_capacity import (  # noqa: E402
    DEFAULT_RESIDUAL_CONDITIONS,
    DEFAULT_RESIDUAL_SEEDS,
    DEFAULT_STRUCTURED_DIMS,
    EvaluationSettings,
    VectorCapacityError,
    build_evaluation_targets,
    default_conditions,
    fit_rate_reference,
    run_capacity_study,
    write_results,
)

DEFAULT_CHECKPOINT = "checkpoints/sweep_l2_0.pt"
DEFAULT_OUT_DIR = "results/neuron_vector_capacity"
DEFAULT_FIGURES_DIR = "figures/neuron_vector_capacity"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _parse_args(argv: Sequence[str] | None):
    parser = argparse.ArgumentParser(
        description="Scientific capacity evaluation of the V2 neuron-vector representation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="configs/neuron_space_baseline.yaml")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--checkpoint", default=None,
                        help=f"frozen SNN checkpoint (default: config select_recipe, else {DEFAULT_CHECKPOINT})")
    parser.add_argument("--structured-dims", default=",".join(str(d) for d in DEFAULT_STRUCTURED_DIMS))
    parser.add_argument("--residual-conditions", default=",".join(f"{s}:{r}" for s, r in DEFAULT_RESIDUAL_CONDITIONS))
    parser.add_argument("--residual-seeds", default=",".join(str(s) for s in DEFAULT_RESIDUAL_SEEDS))
    parser.add_argument("--n-perm", type=int, default=2000)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--k-values", default="3,5,10,20")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--tag", default=None)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--figures-dir", default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--no-cache", action="store_true", help="recompute activity and residuals")
    parser.add_argument("--quick", action="store_true",
                        help="fast smoke settings (few permutations, one residual seed)")
    parser.add_argument("--no-controls", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _parse_int_list(text: str) -> list[int]:
    return [int(part) for part in str(text).replace(" ", "").split(",") if part]


def _parse_pairs(text: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for part in str(text).replace(" ", "").split(","):
        if not part:
            continue
        left, _, right = part.partition(":")
        pairs.append((int(left), int(right)))
    return pairs


# --------------------------------------------------------------------------
# Data: FIT / PROBE (the official TEST file is never opened here)
# --------------------------------------------------------------------------
def build_fit_probe_recordings(cfg: Any) -> dict[str, Any]:
    """FIT and PROBE recordings using the canonical split parameters, without TEST.

    Mirrors ``scripts/_common.build_recordings`` for the two splits this study uses, but
    reads only the SHD **training** file, so the official test file is never opened.
    """
    seed = int(cfg.get_path("seed", 0))
    split_seed_raw = cfg.get_path("data.split_seed", None)
    split_seed = seed if split_seed_raw is None else int(split_seed_raw)
    dev_fraction = float(cfg.get_path("data.dev_fraction", 0.1))
    probe_fraction = float(cfg.get_path("data.probe_fraction", 0.1))
    prefer_speaker = bool(cfg.get_path("data.prefer_speaker_aware", True))
    debug = bool(cfg.get_path("run.debug", False))
    synthetic = bool(cfg.get_path("run.synthetic", False))

    if synthetic:
        n_channels = int(cfg.get_path("run.synthetic_n_channels", 40))
        n_classes = int(cfg.get_path("run.synthetic_n_classes", 5))
        # mirror scripts/_common.build_recordings: align the model block with the synthetic data
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
        source_name = "synthetic_train"
    else:
        train_h5 = PROJECT_ROOT / str(cfg.get_path("paths.data_dir", "data")) / "shd_train.h5"
        if not train_h5.exists():
            raise FileNotFoundError(
                f"{train_h5} not found; download SHD first (see README) or pass --override run.synthetic=true"
            )
        rec = load_shd_recordings(train_h5, layout=str(cfg.get_path("data.layout", "auto")))
        source_name = "shd_train"

    fit, dev, probe, split_info = make_train_dev_probe_split(
        rec, dev_fraction=dev_fraction, probe_fraction=probe_fraction, seed=split_seed,
        prefer_speaker_aware=prefer_speaker,
    )
    if debug:
        fit = subset_by_class(fit, max_per_class=40, seed=split_seed)
        if len(probe) > 0:
            probe = subset_by_class(probe, max_per_class=20, seed=split_seed)
    max_per_class = cfg.get_path("data.max_per_class", None)
    if max_per_class:
        fit = subset_by_class(fit, max_per_class=int(max_per_class), seed=split_seed)
    return {
        "fit": fit,
        "probe": probe,
        "split_info": dict(split_info),
        "source": source_name,
        "split_seed": split_seed,
        "test_loaded": False,
    }


# --------------------------------------------------------------------------
# Caching (activity + residual artifacts)
# --------------------------------------------------------------------------
def _file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = sha256(path.read_bytes()).hexdigest()[:16]
    return {"path": str(path), "size": int(stat.st_size), "mtime": int(stat.st_mtime), "sha256_16": digest}


def _cache_key(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return sha256(text.encode("utf-8")).hexdigest()[:16]


def _load_or_collect_activity(
    *,
    role: str,
    model: RecurrentLIFSNN,
    recordings: Any,
    device: Any,
    cache_dir: Path,
    cache_meta: Mapping[str, Any],
    use_cache: bool,
    verbose: bool,
    with_labels: bool,
    batch_size: int,
    collect_voltage: bool = False,
) -> tuple[ActivityAccumulatorResult, bool]:
    """Return ``(accumulator, from_cache)`` for one dataset role, using a verified cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key({"role": role, **dict(cache_meta)})
    npz_path = cache_dir / f"{role}_activity_{key}.npz"
    meta_path = cache_dir / f"{role}_activity_{key}.json"
    if use_cache and npz_path.exists() and meta_path.exists():
        stored = json.loads(meta_path.read_text(encoding="utf-8"))
        if stored == dict(cache_meta):
            if verbose:
                print(f"[cache] {role} activity <- {npz_path.name}")
            return ActivityAccumulatorResult.load(str(npz_path)), True
        if verbose:
            print(f"[cache] {role} activity cache metadata mismatch; recomputing")
    indices = np.arange(len(recordings), dtype=np.int64)
    t0 = time.time()
    result = collect_activity(
        model, recordings, indices, device=device, batch_size=int(batch_size),
        n_classes=int(model.cfg.n_output), with_labels=bool(with_labels),
        collect_voltage=bool(collect_voltage),
    )
    if verbose:
        print(f"[activity] {role}: n={result.n_samples} in {time.time() - t0:.1f}s")
    if use_cache:
        result.save(str(npz_path))
        meta_path.write_text(json.dumps(dict(cache_meta), indent=2, default=str), encoding="utf-8")
    return result, False


def _load_or_train_residual(
    *,
    bank: Any,
    source_config: ResidualSourceConfig,
    training_config: ResidualTrainingConfig,
    cache_path: Path,
    expected_names: Sequence[str],
    use_cache: bool,
    verbose: bool,
) -> tuple[ResidualResult, bool]:
    """Return ``(residual, trained_now)``; a cached artifact is verified against the schema."""
    if use_cache and cache_path.exists():
        try:
            residual = ResidualResult.load(
                cache_path, expected_feature_names=expected_names,
                expected_residual_dim=training_config.residual_dim, map_location="cpu",
            )
            if verbose:
                print(f"[cache] residual d_res={training_config.residual_dim} seed={training_config.seed} <- {cache_path.name}")
            return residual, False
        except (ResidualError, RuntimeError) as exc:
            if verbose:
                print(f"[cache] residual cache rejected ({exc}); retraining")
    source = build_residual_source(bank, source_config)
    t0 = time.time()
    residual = train_residual(bank, config=training_config, source=source)
    if verbose:
        print(
            f"[residual] d_res={training_config.residual_dim} seed={training_config.seed} "
            f"trained in {time.time() - t0:.1f}s (best val {residual.best_val_loss:.4f})"
        )
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        residual.save(cache_path)
    return residual, True


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.time()

    cfg = load_config(args.config, overrides=args.override)
    # data first: the synthetic path aligns the model block with the synthetic dataset
    recs = build_fit_probe_recordings(cfg)
    try:
        v2 = V2Config.from_config(cfg, warn=True)
    except V2ConfigError as exc:
        print(f"[v2-config] error: {exc}", file=sys.stderr)
        return 2

    checkpoint = args.checkpoint or str(
        cfg.get_path("select_recipe.reuse_seed0_checkpoint", DEFAULT_CHECKPOINT)
    )
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    if not checkpoint_path.exists():
        print(f"[error] checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 2
    tag = args.tag or f"{checkpoint_path.stem}_v2cap"

    if args.quick:
        args.n_perm = min(int(args.n_perm), 50)
        args.bootstrap = 0
        args.structured_dims = "48,64"
        args.residual_seeds = "0"
    structured_dims = _parse_int_list(args.structured_dims)
    residual_conditions = _parse_pairs(args.residual_conditions)
    residual_seeds = _parse_int_list(args.residual_seeds)
    k_values = tuple(int(k) for k in str(args.k_values).replace(" ", "").split(",") if k)

    # -- frozen model --------------------------------------------------------
    model, checkpoint_extra = RecurrentLIFSNN.load(str(checkpoint_path), map_location="cpu")
    mismatch = architecture_mismatches(cfg.get_path("model", {}) or {}, model)
    if mismatch:
        detail = ", ".join(f"model.{k}: config={a!r} vs checkpoint={b!r}" for k, (a, b) in mismatch.items())
        print(f"[error] config/checkpoint architecture mismatch: {detail}", file=sys.stderr)
        return 2
    device = get_device(prefer_cuda=True) if (args.device in (None, "auto")) else (
        get_device(prefer_cuda=True) if args.device == "cuda" else "cpu"
    )

    # -- data (FIT + PROBE only; TEST never opened) --------------------------
    fit_rec, probe_rec = recs["fit"], recs["probe"]
    print(
        f"[data] FIT n={len(fit_rec)} PROBE n={len(probe_rec)} | split={recs['split_info'].get('strategy')} "
        f"| seed={recs['split_seed']} | test_loaded={recs['test_loaded']}"
    )

    out_dir = PROJECT_ROOT / str(args.out_dir)
    figures_dir = PROJECT_ROOT / str(args.figures_dir)
    cache_dir = out_dir / "cache"
    residual_dir = out_dir / "residuals"

    fingerprint = _file_fingerprint(checkpoint_path)
    cache_common = {
        "checkpoint": fingerprint["sha256_16"],
        "checkpoint_path": str(checkpoint_path.relative_to(PROJECT_ROOT)) if checkpoint_path.is_relative_to(PROJECT_ROOT) else str(checkpoint_path),
        "n_bins": int(model.cfg.n_bins),
        "bin_ms": float(model.cfg.bin_ms),
        "split_seed": int(recs["split_seed"]),
        "split_strategy": recs["split_info"].get("strategy"),
        "source": recs["source"],
    }

    # -- FIT activity + bank ------------------------------------------------
    activity_batch = int(v2.memory.record_batch_size)
    fit_activity, fit_from_cache = _load_or_collect_activity(
        role="fit", model=model, recordings=fit_rec, device=device, cache_dir=cache_dir,
        cache_meta={**cache_common, "role": "fit", "n": len(fit_rec), "with_labels": False},
        use_cache=not args.no_cache, verbose=True, with_labels=False, batch_size=activity_batch,
    )
    bank = build_neuron_record_bank(model, activity=fit_activity, config=v2)
    print(f"[bank] {bank.n_neurons} neurons, blocks={list(bank.block_names)} (cache={fit_from_cache})")

    # -- learned residuals (FIT only, one artifact per residual dimension x seed)
    source_config = ResidualSourceConfig.from_v2_config(v2)
    source = build_residual_source(bank, source_config)
    print(f"[residual] source dimension F={source.n_features} (schema {source.schema_hash[:12]})")

    residuals: dict[int, dict[int, ResidualResult]] = {}
    residual_info: list[dict[str, Any]] = []
    residual_cache_key = _cache_key({
        **cache_common, "schema_hash": source.schema_hash, "blocks": list(v2.vector.enabled_blocks),
    })
    for _structured_d, residual_d in residual_conditions:
        for seed in residual_seeds:
            training = ResidualTrainingConfig.from_v2_config(
                v2, residual_dim=int(residual_d), seed=int(seed), mask_seed=0,
                hidden_dim=64, epochs=200, batch_size=64, lr=1e-3, mask_fraction=0.25,
                normalization="train_standardise", val_fraction=0.2, split_seed=0, device="cpu",
            )
            path = residual_dir / f"{residual_cache_key}_dres{residual_d}_seed{seed}.pt"
            residual, trained_now = _load_or_train_residual(
                bank=bank, source_config=source_config, training_config=training, cache_path=path,
                expected_names=source.feature_names, use_cache=not args.no_cache, verbose=True,
            )
            residuals.setdefault(int(residual_d), {})[int(seed)] = residual
            residual_info.append({
                "residual_d": int(residual_d), "seed": int(seed), "trained_now": bool(trained_now),
                "input_dim": residual.input_dim, "schema_hash": residual.schema_hash,
                "source_schema_hash": residual.source_schema.get("schema_hash"),
                "normalization": residual.normalization.get("mode"),
                "mask_fraction": residual.config.mask_fraction, "mask_seed": residual.config.mask_seed,
                "epochs": len(residual.history), "best_epoch": residual.best_epoch,
                "best_val_loss": residual.best_val_loss,
                "trained_on_probe": False, "snn_modified": False,
                "dtype": residual.provenance.get("dtype"), "device": residual.provenance.get("device"),
            })

    # -- PROBE targets (one labelled pass, cached) ---------------------------
    probe_activity, probe_from_cache = _load_or_collect_activity(
        role="probe", model=model, recordings=probe_rec, device=device, cache_dir=cache_dir,
        cache_meta={**cache_common, "role": "probe", "n": len(probe_rec), "with_labels": True},
        use_cache=not args.no_cache, verbose=True, with_labels=True, batch_size=activity_batch,
    )
    targets = build_evaluation_targets(
        probe_activity, probe_split_label="probe",
        n_psth_bins=int(cfg.get_path("fingerprint.n_psth_bins", 10)),
        min_spikes_for_latency=float(cfg.get_path("fingerprint.min_spikes_for_latency", 1.0)),
    )
    print(
        f"[targets] primary={targets.primary.X.shape} class_rate={targets.class_rate.X.shape} "
        f"temporal={targets.temporal.X.shape} (cache={probe_from_cache})"
    )

    # -- evaluate ------------------------------------------------------------
    conditions = default_conditions(structured_dims, residual_conditions, residual_seeds)
    settings = EvaluationSettings(
        n_perm=int(args.n_perm), bootstrap=int(args.bootstrap), k_values=k_values,
        seed=int(cfg.get_path("seed", 0)), n_splits=int(args.n_splits),
        checkpoint=str(checkpoint_path.name), tag=tag,
    )
    fit_rates = fit_rate_reference(fit_activity)
    print(f"[evaluate] {len(conditions)} conditions | n_perm={settings.n_perm} | bootstrap={settings.bootstrap}")
    result = run_capacity_study(
        bank, targets, conditions, settings=settings, fit_rates=fit_rates,
        residuals=residuals, enabled_blocks=tuple(v2.vector.enabled_blocks),
        chunk_size=int(v2.memory.representation_chunk_size),
        controls=not args.no_controls, curve_for="structured_48",
    )

    metadata = {
        "schema": "neuron_vector_capacity/v1",
        "tag": tag,
        "config": str(args.config),
        "checkpoint": {
            **fingerprint,
            "extra": {k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v))
                      for k, v in dict(checkpoint_extra).items()},
            "architecture": {
                "n_input": int(model.cfg.n_input), "n_hidden": int(model.cfg.n_hidden),
                "n_output": int(model.cfg.n_output), "n_bins": int(model.cfg.n_bins),
                "bin_ms": float(model.cfg.bin_ms), "readout_mode": str(model.cfg.readout_mode),
                "neuron_param_mode": str(model.cfg.neuron_param_mode),
                "threshold": float(model.cfg.threshold), "tau_mem_ms": float(model.cfg.tau_mem_ms),
                "tau_syn_ms": float(model.cfg.tau_syn_ms),
            },
        },
        "data": {
            "source": recs["source"],
            "fit_n": int(len(fit_rec)),
            "probe_n": int(len(probe_rec)),
            "split_seed": int(recs["split_seed"]),
            "split_info": recs["split_info"],
            "official_test_loaded": False,
            "fit_role": "representation construction + residual training only",
            "probe_role": "functional targets and evaluation only",
        },
        "activity": {
            "fit_cached": bool(fit_from_cache), "probe_cached": bool(probe_from_cache),
            "batch_size": activity_batch, "collect_voltage": False,
            "cache_key_common": cache_common,
        },
        "residuals": residual_info,
        "conditions": [condition.to_dict() for condition in conditions],
        "targets": targets.summary(),
        "evaluation": settings.to_dict(),
        "preprocessing": result["preprocessing"],
        "controls": [row.get("representation") for row in result["controls"]],
        "curve_condition": "structured_48" if result["curves"] else None,
        "result_columns": [
            "representation", "total_d", "structured_d", "residual_d", "residual_seed",
            "primary_metric_value (Mantel Spearman r, individual-stimulus target)",
            "primary_rate_normalized_r", "primary_rate_matched_r",
            "class_rate_metric_value", "temporal_metric_value", "prediction_metric_value",
        ],
        "notes": [
            "FIT builds the representation; PROBE builds targets and metrics; TEST is never opened.",
            "No representation (including the residual) is fitted or normalised on PROBE.",
            "Matched total dimensions: structured_64 vs full_48+16, structured_100 vs full_48+52.",
            "Residual conditions report every seed separately; no ranking or 'best dimension' is implied.",
            "The learned component is FIT-only masked reconstruction; no PROBE-guided selection was used.",
        ],
        "runtime_seconds": None,
    }
    result["metadata"] = metadata

    written = write_results(
        result,
        csv_path=out_dir / "results.csv",
        json_path=out_dir / "results.json",
    )
    metadata["outputs"] = {k: display_path(v) for k, v in written.items()}
    if not args.no_figures:
        written_figs = write_all_figures(result, figures_dir, curve_condition="structured_48")
        metadata["figures"] = written_figs

    metadata["runtime_seconds"] = round(time.time() - started, 1)
    save_json(metadata, out_dir / "metadata.json")

    print(f"[done] {len(result['rows'])} condition rows, {len(result['controls'])} control rows")
    for row in result["rows"]:
        print(
            f"  {row['representation']:<28s} d={row['total_d']:<4d} "
            f"primary_r={_fmt(row['primary_metric_value'])} "
            f"rate_norm_r={_fmt(row['primary_rate_normalized_r'])} "
            f"class_rate_r={_fmt(row['class_rate_metric_value'])} "
            f"temporal_r={_fmt(row['temporal_metric_value'])} "
            f"cv_r2={_fmt(row['prediction_metric_value'])}"
        )
    for row in result["controls"]:
        print(
            f"  {row['representation']:<28s} d={row['total_d']:<4d} "
            f"primary_r={_fmt(row['primary_metric_value'])}"
        )
    print(f"[results] {written['csv']}")
    print(f"[results] {written['json']}")
    print(f"[metadata] {out_dir / 'metadata.json'}")
    if not args.no_figures:
        print(f"[figures] {(figures_dir).relative_to(PROJECT_ROOT) if figures_dir.is_relative_to(PROJECT_ROOT) else figures_dir}")
    return 0


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(number) else f"{number:+.3f}"


if __name__ == "__main__":
    raise SystemExit(main())