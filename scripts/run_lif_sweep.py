"""Controlled sweep runner for a trustworthy recurrent LIF baseline on SHD.

Scientific rules enforced here:

* **Selection uses DEV only.** PROBE and the official TEST set are never used to
  choose a configuration.
* The official TEST set is evaluated **only** for the single selected configuration
  (``--evaluate-test``), never inside the sweep.
* Circuit-health diagnostics are recorded for every run (rate percentiles,
  high/low-rate fractions, membrane-potential statistics, per-class recall), so a
  configuration is never chosen on accuracy alone.

Outputs:
* ``results/lif_baseline_sweep.csv`` - one row per run;
* ``results/lif_sweep/<label>.json`` - full history + metrics per run.

Example::

    uv run python scripts/run_lif_sweep.py --config configs/baseline_repaired.yaml \
        --grid l2_spikes --epochs 10 --early-stopping 0
"""

from __future__ import annotations

import csv
import time
from typing import Any

import numpy as np
from tqdm import tqdm

from _common import (  # noqa: E402
    build_recordings,
    checkpoint_path,
    resolve_device,
    result_path,
)
from src.evaluation import circuit_health, evaluate_model  # noqa: E402
from src.model import build_model  # noqa: E402
from src.training import TrainConfig, train_model  # noqa: E402
from src.utils import ensure_dir, load_config, save_json  # noqa: E402


# --------------------------------------------------------------------------
# Grids
# --------------------------------------------------------------------------
SPIKE_VALUES = (0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3)


def grid_l2_spikes() -> list[tuple[str, dict[str, Any]]]:
    return [
        (f"l2_{v:g}", {"train.l2_spikes": v, "train.target_rate_hz": 10.0})
        for v in SPIKE_VALUES
    ]


def grid_target_strength() -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for target in (5.0, 20.0, 50.0):
        for lam in (1e-5, 1e-4, 1e-3):
            rows.append((f"t{target:g}_l{lam:g}", {"train.target_rate_hz": target, "train.l2_spikes": lam}))
    return rows


def grid_readout() -> list[tuple[str, dict[str, Any]]]:
    return [(f"readout_{m}", {"model.readout_mode": m}) for m in ("mean", "last", "sum")]


def grid_window() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("bin2_n700_1400ms", {"model.bin_ms": 2.0, "model.n_bins": 700}),
        ("bin2_n500_1000ms", {"model.bin_ms": 2.0, "model.n_bins": 500}),
        ("bin4_n300_1200ms", {"model.bin_ms": 4.0, "model.n_bins": 300}),
        ("bin4_n250_1000ms", {"model.bin_ms": 4.0, "model.n_bins": 250}),
    ]


def grid_weight_decay() -> list[tuple[str, dict[str, Any]]]:
    return [
        (f"wd_{v:g}", {"train.optimizer": "adamw", "train.weight_decay": v})
        for v in (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
    ]


GRIDS = {
    "l2_spikes": grid_l2_spikes,
    "target_strength": grid_target_strength,
    "readout": grid_readout,
    "window": grid_window,
    "weight_decay": grid_weight_decay,
}


# --------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------
def run_one(
    label: str,
    grid_overrides: dict[str, Any],
    *,
    config_path: str,
    cli_overrides: list[str],
    recs: dict,
    device: Any,
    epochs: int,
    patience: int,
    seed: int,
    evaluate_test: bool,
    verbose: bool,
    base_cfg: Any,
    fit_subset: int = 2000,
) -> dict[str, Any]:
    overrides = [f"{k}={v}" for k, v in grid_overrides.items()] + list(cli_overrides)
    cfg = load_config(config_path, overrides=overrides)
    n_classes = int(cfg.get_path("train.n_classes", 20))
    batch_size = int(cfg.get_path("train.eval_batch_size", 256))

    model = build_model(cfg, seed=seed, device=device)
    sim = model.cfg
    tcfg = TrainConfig.from_mapping(cfg.get_path("train", {}))
    tcfg.epochs = int(epochs)
    tcfg.early_stopping_patience = int(patience)
    tcfg.select_by = "dev_accuracy"

    train_rec, dev_rec = recs["train"], recs["dev"]
    train_idx = np.arange(len(train_rec))
    dev_idx = np.arange(len(dev_rec))
    # a bounded FIT subset keeps the per-run diagnostics cheap; DEV is always full
    fit_idx = train_idx[: int(max(256, min(fit_subset, train_idx.size)))]

    t0 = time.time()
    result = train_model(
        model, train_rec, train_idx, dev_rec, dev_idx, tcfg,
        device=device, seed=seed, verbose=verbose,
    )
    wall = time.time() - t0

    # persist the best-DEV checkpoint (train_model already restored the best weights)
    model.save(
        str(checkpoint_path(base_cfg, f"sweep_{label}")),
        extra={"sweep_label": label, "grid_overrides": dict(grid_overrides),
               "model_config": sim.to_dict(), "train_config": tcfg.to_dict()},
    )

    fit_m = evaluate_model(model, train_rec, fit_idx, device=device, batch_size=batch_size,
                           n_classes=n_classes, compute_confusion=True)
    dev_m = evaluate_model(model, dev_rec, dev_idx, device=device, batch_size=batch_size,
                           n_classes=n_classes, compute_confusion=True)
    health = circuit_health(model, train_rec, fit_idx, device=device, batch_size=batch_size,
                            n_classes=n_classes)
    rates = np.asarray(health["rate_hz_per_neuron"], dtype=np.float64)
    dev_recall = np.asarray(dev_m.get("per_class_accuracy", []), dtype=np.float64)

    duration_s = sim.duration_ms / 1000.0
    row: dict[str, Any] = {
        "label": label,
        "epochs_requested": int(epochs),
        "epochs_run": len(result.history),
        "best_epoch": int(result.best_epoch),
        "wall_time_s": round(wall, 1),
        "seed": int(seed),
        "n_hidden": int(sim.n_hidden),
        "n_bins": int(sim.n_bins),
        "bin_ms": float(sim.bin_ms),
        "window_ms": float(sim.duration_ms),
        "readout_mode": sim.readout_mode,
        "l2_spikes": float(tcfg.l2_spikes),
        "target_rate_hz": float(tcfg.target_rate_hz),
        "target_spikes_per_example": float(tcfg.target_rate_hz * duration_s),
        # accuracy
        "fit_accuracy": fit_m["accuracy"],
        "dev_accuracy": dev_m["accuracy"],
        "fit_loss": fit_m["loss"],
        "dev_loss": dev_m["loss"],
        # circuit health (hidden layer, measured on FIT)
        "rate_mean_hz": float(rates.mean()) if rates.size else float("nan"),
        "rate_median_hz": float(np.median(rates)) if rates.size else float("nan"),
        "rate_p90_hz": float(np.percentile(rates, 90)) if rates.size else float("nan"),
        "rate_max_hz": float(rates.max()) if rates.size else float("nan"),
        "frac_gt_100hz": float((rates > 100).mean()) if rates.size else float("nan"),
        "frac_gt_200hz": float((rates > 200).mean()) if rates.size else float("nan"),
        "frac_lt_1hz": float((rates < 1).mean()) if rates.size else float("nan"),
        "frac_lt_5hz": float((rates < 5).mean()) if rates.size else float("nan"),
        "silent_fraction": float(health["silent_neuron_fraction"]),
        "v_mean": float(health["v_global_mean"]),
        "v_std": float(health["v_global_std"]),
        "v_min": float(health["v_global_min"]),
        "v_max": float(health["v_global_max"]),
        # class behaviour (DEV)
        "dev_median_class_recall": float(np.nanmedian(dev_recall)) if dev_recall.size else float("nan"),
        "dev_min_class_recall": float(np.nanmin(dev_recall)) if dev_recall.size else float("nan"),
        "dev_n_classes_recall_lt_0p1": int(np.nansum(dev_recall < 0.1)),
        "dev_class_recall": [None if not np.isfinite(v) else round(float(v), 4) for v in dev_recall],
        # training dynamics
        "final_grad_norm_mean": float(result.history[-1]["grad_norm_mean"]) if result.history else float("nan"),
        "train_loss_first": float(result.history[0]["train_loss"]) if result.history else float("nan"),
        "train_loss_last": float(result.history[-1]["train_loss"]) if result.history else float("nan"),
        "grad_scale": float(model.cfg.input_weight_scale),
        "rec_scale": float(model.cfg.recurrent_weight_scale),
        "optimizer": str(tcfg.optimizer),
        "weight_decay": float(tcfg.weight_decay),
        "checkpoint": f"sweep_{label}.pt",
        "evaluate_test": bool(evaluate_test),
        "test_accuracy": None,
    }
    if evaluate_test:
        test_rec = recs["test"]
        test_m = evaluate_model(model, test_rec, np.arange(len(test_rec)), device=device,
                                batch_size=batch_size, n_classes=n_classes, compute_confusion=True)
        row["test_accuracy"] = test_m["accuracy"]
        row["test_class_recall"] = test_m.get("per_class_accuracy")

    # persist the full history for this run
    ensure_dir(result_path(base_cfg, "lif_sweep"))
    hist_path = result_path(base_cfg, f"lif_sweep/{label}.json")
    save_json(
        {
            "label": label,
            "grid_overrides": {k: (str(v) if not isinstance(v, (int, float, bool)) else v)
                               for k, v in grid_overrides.items()},
            "train_config": tcfg.to_dict(),
            "model_config": sim.to_dict(),
            "history": result.history,
            "dev_per_class_recall": dev_recall.tolist(),
            "dev_prediction_distribution": None,
            "row": row,
        },
        hist_path,
    )
    return row


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def _parse_args(argv: list[str] | None) -> "argparse.Namespace":
    import argparse

    p = argparse.ArgumentParser(description="Controlled recurrent-LIF baseline sweep (DEV-selected).")
    p.add_argument("--config", default="configs/baseline_repaired.yaml")
    p.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--grid", default="l2_spikes", choices=sorted(GRIDS))
    p.add_argument("--only", default="", help="comma-separated subset of grid labels to run")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--early-stopping", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--out", default="lif_baseline_sweep.csv")
    p.add_argument("--fit-subset", type=int, default=2000,
                   help="FIT samples used for the per-run accuracy/health diagnostics")
    p.add_argument("--evaluate-test", action="store_true",
                   help="evaluate the official TEST set (use ONLY for the selected config)")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import argparse  # noqa: F401  (kept for the type annotation above)

    args = _parse_args(argv)
    base_cfg = load_config(args.config, overrides=args.override)
    device = resolve_device(base_cfg)
    grid = GRIDS[args.grid]()
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        grid = [(lbl, ov) for lbl, ov in grid if lbl in wanted]
        if not grid:
            raise SystemExit(f"--only {args.only!r} matched no labels in grid {args.grid!r}")

    recs = build_recordings(base_cfg)
    print(
        f"[sweep] grid={args.grid} runs={len(grid)} epochs={args.epochs} "
        f"patience={args.early_stopping} device={device} | DEV-selected; "
        f"official TEST {'EVALUATED' if args.evaluate_test else 'WITHHELD'}"
    )

    rows: list[dict[str, Any]] = []
    out_path = result_path(base_cfg, args.out)
    with tqdm(total=len(grid), desc=args.grid) as bar:
        for label, overrides in grid:
            bar.set_postfix_str(label)
            try:
                row = run_one(
                    label, overrides,
                    config_path=args.config, cli_overrides=list(args.override),
                    recs=recs, device=device, epochs=args.epochs, patience=args.early_stopping,
                    seed=args.seed, evaluate_test=args.evaluate_test, verbose=not args.quiet,
                    base_cfg=base_cfg, fit_subset=args.fit_subset,
                )
            except Exception as exc:  # keep the sweep going and record the failure
                row = {"label": label, "error": f"{type(exc).__name__}: {exc}", **overrides}
                print(f"[sweep][error] {label}: {exc}")
            rows.append(row)
            _write_csv(rows, out_path)
            _log_row(row)
            bar.update(1)
    print(f"[save] {out_path}")
    return 0


def _write_csv(rows: list[dict[str, Any]], path) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in row.items()})


def _log_row(row: dict[str, Any]) -> None:
    if "error" in row:
        return
    print(
        f"    {row['label']:<16s} fit={_f(row['fit_accuracy'])} dev={_f(row['dev_accuracy'])} "
        f"rate={_f(row['rate_mean_hz'], 1)}Hz >200Hz={_f(row['frac_gt_200hz'], 3)} "
        f"medrecall={_f(row['dev_median_class_recall'], 3)} "
        f"nclass<0.1={row['dev_n_classes_recall_lt_0p1']} V[{_f(row['v_min'], 1)},{_f(row['v_max'], 1)}]"
    )


def _f(v: Any, n: int = 3) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(f) else f"{f:.{n}f}"


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))