"""Headless runner for the V3 experiment.

Same code path as ``V3_SHD_experiment.ipynb`` (both call ``v3.experiment``), so
the CLI and the notebook produce identical numbers.

Examples
--------
::

    # primary experiment: N=64 genuine D=1000 vector neurons + scalar baseline
    .venv\\Scripts\\python.exe V3\\run_experiment.py --variant both

    # quick end-to-end smoke of the same code path (minutes)
    .venv\\Scripts\\python.exe V3\\run_experiment.py --variant both --quick

    # vector only, custom overrides
    .venv\\Scripts\\python.exe V3\\run_experiment.py --variant vector \\
        --override epochs=30 --override learning_rate=5e-4
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

V3_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(V3_ROOT))

from v3 import plots  # noqa: E402
from v3.config import StateRegularizationConfig, TimingConfig, V3Config, describe  # noqa: E402
from v3.experiment import resolve_device, test_variant, train_variant, variant_dirs  # noqa: E402
from v3.registry import RunRegistry, register_run  # noqa: E402
from v3.train import save_json  # noqa: E402

DEFAULT_CONFIG = V3_ROOT / "configs" / "v3_default.yaml"

# nested dataclass blocks addressable with dotted --override keys
NESTED_BLOCKS = {
    "timing": TimingConfig,
    "state_regularization": StateRegularizationConfig,
}

QUICK_OVERRIDES = dict(
    n_neurons=16,
    state_dim=64,
    mix_rank=8,
    timing=TimingConfig(sequence_duration_ms=240.0, time_bin_ms=2.0),
    batch_size=16,
    epochs=2,
    max_train_samples=256,
    val_fraction=0.15,
)


def _coerce(anno: str, raw: str):
    if "bool" in anno:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if "int" in anno and "float" not in anno:
        return int(raw)
    if "float" in anno:
        return float(raw)
    return raw


def apply_overrides(cfg: V3Config, items: list[str]) -> V3Config:
    """Apply ``key=value`` overrides, including one level of nesting.

    ``timing.time_bin_ms=2``, ``state_regularization.mode=noise`` and
    ``state_regularization.noise_std=0.01`` update the nested block in place
    (other nested fields keep their configured values).
    """
    vfields = {f.name: f for f in dataclasses.fields(V3Config)}
    flat: dict = {}
    nested: dict[str, dict] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--override must be key=value, got {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if "." in key:
            parent, child = key.split(".", 1)
            if parent not in NESTED_BLOCKS:
                raise SystemExit(f"unknown override key {key!r}")
            nfields = {f.name: f for f in dataclasses.fields(NESTED_BLOCKS[parent])}
            if child not in nfields:
                raise SystemExit(f"unknown override key {key!r}")
            nested.setdefault(parent, {})[child] = _coerce(str(nfields[child].type), raw)
            continue
        if key not in vfields:
            raise SystemExit(f"unknown configuration key {key!r}")
        flat[key] = _coerce(str(vfields[key].type), raw)
    for parent, kwargs in nested.items():
        cfg = cfg.with_overrides(**{parent: dataclasses.replace(getattr(cfg, parent), **kwargs)})
    if flat:
        cfg = cfg.with_overrides(**flat)
    return cfg


def record_run(vcfg: V3Config, variant, info: dict, test_metrics: dict | None, registry: RunRegistry) -> Path:
    """Append one registry row + per-run artifacts for a completed variant.

    Delegates to the shared ``v3.registry.register_run`` -- the same entry point
    the notebook uses, so CLI and notebook rows share one schema.
    """
    extra = {k: v for k, v in info.items() if k not in ("history", "result")}
    return register_run(
        registry, vcfg,
        n_parameters=info["n_parameters"],
        fit_metrics={"accuracy": info.get("fit_accuracy")},
        val_metrics={"accuracy": info.get("val_accuracy")},
        test_metrics=test_metrics,
        train_seconds=info.get("train_seconds", float("nan")),
        duration_seconds=float(info.get("train_seconds", 0.0))
        + float(info.get("init_and_data_seconds", 0.0)),
        checkpoint=info.get("checkpoint", ""),
        best_val_accuracy=info.get("best_val_accuracy"),
        best_epoch=info.get("best_epoch"),
        notes=f"quick={info.get('quick', False)}",
        extra_metrics=extra,
        history=info.get("history"),
    )



def apply_variant(cfg: V3Config, variant: str) -> V3Config:
    if variant == "scalar":
        base_tag = cfg.tag.replace("_d1000", "") if cfg.tag.endswith("_d1000") else cfg.tag
        return cfg.with_overrides(state_dim=1, mix_rank=1, tag=f"{base_tag}_scalar_d1")
    return cfg


def run_variant(cfg: V3Config, device, show_progress=True) -> dict:
    t0 = time.perf_counter()
    v = train_variant(cfg, device=device, show_progress=show_progress)
    out = {
        "tag": cfg.tag,
        "label": v.label,
        "n_neurons": cfg.n_neurons,
        "state_dim": cfg.state_dim,
        "n_parameters": v.model.n_parameters(),
        "parameter_groups": v.model.param_groups(),
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "n_bins": cfg.n_bins,
        "bin_ms": cfg.bin_ms,
        "device": str(v.device),
        "dtype": cfg.dtype,
        "train_seconds": v.result.total_seconds,
        "init_and_data_seconds": v.wall_seconds - v.result.total_seconds,
        "fit_accuracy": v.fit_metrics["accuracy"],
        "val_accuracy": v.val_metrics.get("accuracy", float("nan")),
        "best_val_accuracy": v.result.best_val_accuracy,
        "best_epoch": v.result.best_epoch,
        "rate_mean_hz": v.fit_metrics["rate_mean_hz"],
        "rate_max_hz": v.fit_metrics["rate_max_hz"],
        "spikes_per_sample": v.fit_metrics["spikes_per_sample"],
        "dtypes": v.result.dtypes,
        "history": v.result.history,
        "checkpoint": str(v.checkpoint_path),
        "result": v,
    }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="V3 vector-neuron SHD experiment")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--variant", choices=["vector", "scalar", "both"], default="vector")
    ap.add_argument("--quick", action="store_true", help="tiny end-to-end smoke of the same code path")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-test", action="store_true", help="train only; do not open the official test set")
    ap.add_argument("--test-best-val", action="store_true",
                    help="also evaluate the best-validation checkpoint on the test set")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args(argv)

    cfg = V3Config.from_yaml(args.config)
    cfg = apply_overrides(cfg, args.override)
    if args.quick:
        cfg = cfg.with_overrides(**QUICK_OVERRIDES)
    if args.device:
        cfg = cfg.with_overrides(device=args.device)
    device = resolve_device(cfg)
    print(f"[V3] config: {describe(cfg)}")
    print(f"[V3] device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    show = not args.no_progress

    variants = ["vector", "scalar"] if args.variant == "both" else [args.variant]
    summary: dict = {"variants": {}, "quick": bool(args.quick), "config_file": args.config}
    rows: list[dict] = []
    histories: dict[str, list[dict]] = {}
    base_dirs = variant_dirs(cfg)
    registry = RunRegistry(cfg.v3_path(cfg.out_dir))

    for name in variants:
        vcfg = apply_variant(cfg, name)
        print("\n" + "=" * 78 + f"\n[V3] variant: {name}  ->  {describe(vcfg)}\n" + "=" * 78)
        info = run_variant(vcfg, device, show_progress=show)
        info["quick"] = bool(args.quick)
        variant_obj = info.pop("result")
        tm = None
        if not args.no_test:
            tm = test_variant(variant_obj, which="final", show_progress=show)
            info.update(
                test_correct=int(tm["test_correct"]),
                test_total=int(tm["test_total"]),
                test_accuracy=float(tm["test_accuracy"]),
                test_rate_mean_hz=float(tm["rate_mean_hz"]),
                test_max_memory_allocated_mb=float(tm["max_memory_allocated_mb"]),
            )
            if args.test_best_val and variant_obj.result.state_dict is not None:
                bm = test_variant(variant_obj, which="best_val", save=False, show_progress=show)
                info.update(
                    test_accuracy_best_val=float(bm["test_accuracy"]),
                    test_correct_best_val=int(bm["test_correct"]),
                )
        run_dir = record_run(vcfg, variant_obj, info, tm, registry)
        info["registry_run_dir"] = str(run_dir)
        print(f"[V3] run recorded in {run_dir}")
        summary["variants"][vcfg.tag] = info
        rows.append(info)
        histories[info["label"]] = info["history"]
        del variant_obj
        torch.cuda.empty_cache()

    if len(rows) == 2 and not args.no_test:
        cmp_path = base_dirs["figures"] / "comparison_test_accuracy.png"
        plots.plot_comparison(rows, cmp_path)
        plots.plot_training_comparison(histories, base_dirs["figures"] / "comparison_training.png")
        summary["comparison"] = {
            "figure_test_accuracy": str(cmp_path),
            "delta_test_accuracy_vector_minus_scalar": float(
                rows[0]["test_accuracy"] - rows[1]["test_accuracy"]
            ),
        }
    out_path = base_dirs["results"] / ("summary_quick.json" if args.quick else "summary.json")
    save_json(summary, out_path)
    print(f"\n[V3] summary written to {out_path}")
    for r in rows:
        if "test_accuracy" in r:
            print(f"  {r['tag']:>24s}  test {r['test_correct']:5d}/{r['test_total']:5d} = "
                  f"{100 * r['test_accuracy']:6.2f}%   (N={r['n_neurons']}, D={r['state_dim']}, "
                  f"{r['n_parameters']:,} params)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
