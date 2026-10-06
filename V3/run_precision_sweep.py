"""Lightweight 5-epoch precision-screening runner for the V3 internal state.

Screens **state quantization** of the neuron's internal vector state ``z(t)``:
the differentiable, symmetric uniform quantizer with a straight-through
estimator that already lives in :mod:`v3.state_regularization`.  This is the
*precision of the state itself*, not the CUDA arithmetic dtype and not hardware
FP8/FP4 arithmetic.

Variants (reference architecture ``N=64``, ``D=1000``, ``1000 ms / 2 ms -> T=500``)::

    baseline   mode=none            (unmodified reference behaviour)
    8bit       mode=quantization, quantize_bits=8
    4bit       mode=quantization, quantize_bits=4

Everything else stays identical to the reference configuration.  The runner
**reuses** the existing config system, model construction, training loop,
evaluation and registry -- it never duplicates the training loop and never
creates a second result database.  Each completed variant produces exactly one
``v3.registry.register_run`` row (one configuration -> one registry row -> one
timestamped artifact directory).

The screening stage is deliberately short (``epochs=5``, forced).  A 20-epoch
run is only done later, by hand, for a configuration that looks promising.

Examples
--------
::

    # the two requested screening variants (8-bit and 4-bit state quantization)
    .venv\\Scripts\\python.exe V3\\run_precision_sweep.py

    # include the unmodified baseline as a control
    .venv\\Scripts\\python.exe V3\\run_precision_sweep.py --variants baseline,8bit,4bit

    # optional full-size forward sanity check (D=1000, T=500, no training)
    .venv\\Scripts\\python.exe V3\\run_precision_sweep.py --smoke
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import torch

V3_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(V3_ROOT))

from run_experiment import apply_overrides  # noqa: E402  (reuse the shared override parser)
from v3.config import V3Config, describe  # noqa: E402
from v3.experiment import effective_config, resolve_device, test_variant, train_variant  # noqa: E402
from v3.model import VectorNeuronPopulation  # noqa: E402
from v3.registry import RunRegistry, register_run  # noqa: E402

DEFAULT_CONFIG = V3_ROOT / "configs" / "v3_default.yaml"

# Screening is fixed at 5 epochs; the global YAML default is not changed.
SCREEN_EPOCHS = 5

# variant name -> quantize_bits (None == baseline, mode "none")
PRECISION_VARIANTS: dict[str, int | None] = {
    "baseline": None,
    "8bit": 8,
    "4bit": 4,
}

# distinct tags so a screening run can never overwrite the 20-epoch reference
# checkpoint (checkpoints are saved as ``<tag>.pt``)
TAG_SUFFIX = {
    "baseline": "_prec_baseline",
    "8bit": "_prec_q8",
    "4bit": "_prec_q4",
}


def _root_tag(tag: str) -> str:
    for suffix in TAG_SUFFIX.values():
        if tag.endswith(suffix):
            return tag[: -len(suffix)]
    return tag


def build_precision_config(
    cfg: V3Config, variant: str, *, epochs: int = SCREEN_EPOCHS
) -> V3Config:
    """Return the screening configuration for ``variant`` (pure, no training).

    Only the run identity (``tag``), the training budget (``epochs``) and the
    ``state_regularization`` block are touched; every other field is inherited
    from ``cfg`` unchanged.
    """
    if variant not in PRECISION_VARIANTS:
        raise SystemExit(
            f"unknown variant {variant!r}; choose from {sorted(PRECISION_VARIANTS)}"
        )
    reg = cfg.state_regularization
    if PRECISION_VARIANTS[variant] is None:
        # baseline: the state is untouched (bit-identical to the reference path)
        reg = dataclasses.replace(reg, mode="none")
    else:
        bits = int(PRECISION_VARIANTS[variant])
        reg = dataclasses.replace(
            reg,
            mode="quantization",
            quantize_bits=bits,
            apply_during_training=True,
            apply_during_validation=False,
            apply_during_test=False,
        )
    tag = f"{_root_tag(cfg.tag)}{TAG_SUFFIX[variant]}"
    return cfg.with_overrides(epochs=int(epochs), state_regularization=reg, tag=tag).resolved()


def register_variant(
    registry: RunRegistry,
    cfg: V3Config,
    *,
    n_parameters: int,
    fit_metrics: dict | None = None,
    val_metrics: dict | None = None,
    test_metrics: dict | None = None,
    train_seconds: float = float("nan"),
    duration_seconds: float = float("nan"),
    checkpoint: str = "",
    best_val_accuracy: float | None = None,
    best_epoch: int | None = None,
    variant: str = "",
) -> Path:
    """Register exactly one completed variant through the shared entry point.

    Thin wrapper over :func:`v3.registry.register_run`; it exists so the sweep
    (and its tests) have a single, explicit place that records a variant.
    """
    return register_run(
        registry,
        cfg,
        n_parameters=n_parameters,
        fit_metrics=fit_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        train_seconds=train_seconds,
        duration_seconds=duration_seconds,
        checkpoint=checkpoint,
        best_val_accuracy=best_val_accuracy,
        best_epoch=best_epoch,
        notes=f"precision_sweep variant={variant} epochs={cfg.epochs}",
    )


def run_precision_sweep(
    cfg: V3Config,
    device: torch.device,
    variants: list[str],
    *,
    epochs: int = SCREEN_EPOCHS,
    test: bool = True,
    show_progress: bool = True,
) -> list[dict]:
    """Train + evaluate + register one row per requested variant.

    Reuses ``v3.experiment.train_variant`` / ``test_variant`` (the exact code
    path the notebook and the main CLI use) and ``v3.registry.register_run``.
    """
    registry = RunRegistry(cfg.v3_path(cfg.out_dir))
    rows: list[dict] = []
    for name in variants:
        vcfg = build_precision_config(cfg, name, epochs=epochs)
        print("\n" + "=" * 78 + f"\n[precision] variant: {name}  ->  {describe(vcfg)}\n" + "=" * 78)
        t0 = time.perf_counter()
        v = train_variant(vcfg, device=device, show_progress=show_progress)
        test_metrics = test_variant(v, which="final", show_progress=show_progress) if test else None
        run_dir = register_variant(
            registry,
            v.cfg,
            n_parameters=v.model.n_parameters(),
            fit_metrics={"accuracy": v.fit_metrics.get("accuracy")},
            val_metrics={"accuracy": v.val_metrics.get("accuracy", float("nan"))},
            test_metrics=test_metrics,
            train_seconds=v.result.total_seconds,
            duration_seconds=v.wall_seconds,
            checkpoint=str(v.checkpoint_path),
            best_val_accuracy=v.result.best_val_accuracy,
            best_epoch=v.result.best_epoch,
            variant=name,
        )
        row = {
            "variant": name,
            "tag": v.cfg.tag,
            "epochs": v.cfg.epochs,
            "quantize_bits": v.cfg.state_regularization.quantize_bits,
            "mode": v.cfg.state_regularization.mode,
            "n_parameters": v.model.n_parameters(),
            "fit_accuracy": v.fit_metrics.get("accuracy"),
            "val_accuracy": v.val_metrics.get("accuracy", float("nan")),
            "test_accuracy": (test_metrics or {}).get("test_accuracy", float("nan")),
            "test_correct": (test_metrics or {}).get("test_correct", 0),
            "test_total": (test_metrics or {}).get("test_total", 0),
            "wall_seconds": time.perf_counter() - t0,
            "registry_run_dir": str(run_dir),
        }
        print(f"[precision] {name} recorded in {run_dir}")
        rows.append(row)
        del v
        torch.cuda.empty_cache()
    return rows


def smoke_forward(cfg: V3Config, variant: str, device: torch.device | None = None) -> dict:
    """One forward pass through the real model (no training).

    Used by ``--smoke`` to confirm the reference shape ``D=1000, T=500``
    completes for a given quantization setting without NaNs/Infs or shape
    errors.  The quantizer path is exercised by selecting the training phase.
    """
    device = device or resolve_device(cfg)
    vcfg = effective_config(build_precision_config(cfg, variant), device)
    model = VectorNeuronPopulation(vcfg).to(device)
    model.eval()
    model.set_phase("train")  # so the quantizer is active during the check
    x = torch.randn(2, vcfg.n_bins, vcfg.n_inputs, device=device, dtype=torch.float32)
    with torch.no_grad():
        logits, spikes = model(x)
    finite = bool(torch.isfinite(logits).all().item() and torch.isfinite(spikes).all().item())
    return {
        "variant": variant,
        "mode": vcfg.state_regularization.mode,
        "quantize_bits": vcfg.state_regularization.quantize_bits,
        "D": vcfg.state_dim,
        "T": vcfg.n_bins,
        "logits_shape": tuple(logits.shape),
        "spikes_shape": tuple(spikes.shape),
        "finite": finite,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="V3 5-epoch state-precision screening sweep")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument(
        "--variants",
        default="8bit,4bit",
        help=f"comma-separated subset of {sorted(PRECISION_VARIANTS)} (default 8bit,4bit)",
    )
    ap.add_argument("--epochs", type=int, default=SCREEN_EPOCHS,
                    help=f"screening budget, forced to {SCREEN_EPOCHS} by default")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-test", action="store_true",
                    help="train only; do not open the official test set")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="full-size forward sanity check only (no training)")
    args = ap.parse_args(argv)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in PRECISION_VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; choose from {sorted(PRECISION_VARIANTS)}")

    cfg = V3Config.from_yaml(args.config)
    cfg = apply_overrides(cfg, args.override)
    if args.device:
        cfg = cfg.with_overrides(device=args.device)
    device = resolve_device(cfg)
    print(f"[precision] base config: {describe(cfg)}")
    print(f"[precision] device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"[precision] screening epochs: {args.epochs}")

    if args.smoke:
        for name in variants:
            print(f"[precision] smoke {name}: {smoke_forward(cfg, name, device)}")
        return 0

    rows = run_precision_sweep(
        cfg, device, variants,
        epochs=args.epochs, test=not args.no_test, show_progress=not args.no_progress,
    )
    print("\n[precision] summary")
    for r in rows:
        acc = r["test_accuracy"]
        acc_s = f"{100 * acc:6.2f}%" if acc == acc else "  n/a  "
        print(f"  {r['variant']:>8s}  mode={r['mode']:<12s} bits={r['quantize_bits']:<2d} "
              f"epochs={r['epochs']}  test {r['test_correct']:5d}/{r['test_total']:5d} = {acc_s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
