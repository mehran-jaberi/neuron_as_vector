"""Train the recurrent LIF SNN on SHD (or the built-in synthetic dataset).

This is stage 4 of the pipeline (see the README). It produces a checkpoint and a
training history; the official SHD *test* set is evaluated and reported here for
completeness, but is never used to make any training or model-selection decision.

Examples
--------
Smoke test without any download (synthetic data, tiny network)::

    uv run python scripts/train.py --config configs/baseline.yaml --synthetic \
        --override model.n_hidden=64 --override model.n_bins=100 \
        --override model.n_input=40 --override model.n_output=5 \
        --override train.n_classes=5 --override run.synthetic_n_samples=200 \
        --override run.synthetic_n_channels=40 --override train.epochs=2

.. note::
   ``--override`` is a repeatable flag: pass one ``--override KEY=VALUE`` per
   leaf you want to change (space-separated extra keys are *not* accepted).
   In synthetic mode you must keep ``model.n_input``/``model.n_output`` and
   ``train.n_classes`` consistent with ``run.synthetic_n_channels`` /
   ``run.synthetic_n_classes``.

Real training run::

    uv run python scripts/train.py --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from _common import (  # noqa: E402  (scripts dir is on sys.path when run)
    PROJECT_ROOT,
    announce,
    apply_seed,
    build_model_from_config,
    build_recordings,
    checkpoint_path,
    load_run_config,
    parse_args,
    resolve_device,
    result_path,
)
from src.evaluation import evaluate_model
from src.model import count_parameters
from src.training import TrainConfig, activity_report, train_model
from src.utils import describe_device, ensure_dir, save_json


def _write_history_csv(history: list[dict], path: Path) -> None:
    if not history:
        return
    ensure_dir(path.parent)
    fields = list(history[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    args = parse_args("Train the recurrent LIF SNN (SHD or synthetic).", "configs/baseline.yaml")
    cfg = load_run_config(args)
    device = resolve_device(cfg)
    announce(cfg, device)
    apply_seed(cfg, device)

    tag = str(cfg.get_path("run.tag", "run"))
    recs = build_recordings(cfg)
    train_rec, val_rec, test_rec = recs["train"], recs["val"], recs["test"]
    train_idx = np.arange(len(train_rec))
    val_idx = np.arange(len(val_rec))
    test_idx = np.arange(len(test_rec))

    model = build_model_from_config(cfg, seed=int(cfg.get_path("seed", 0)), device=device)
    n_params = count_parameters(model)
    print(f"[model] hidden={model.cfg.n_hidden} params={n_params}")

    tcfg = TrainConfig.from_config(cfg)

    t0 = time.time()
    result = train_model(
        model,
        train_rec,
        train_idx,
        val_rec,
        val_idx,
        tcfg,
        device=device,
        seed=int(cfg.get_path("seed", 0)),
        verbose=bool(cfg.get_path("run.verbose", True)),
    )
    wall = time.time() - t0

    # Circuit-health diagnostics: a silent hidden layer makes the whole study vacuous.
    health_train = activity_report(model, train_rec, train_idx, device=device)

    metrics: dict = {}
    metrics["train"] = evaluate_model(
        model, train_rec, train_idx, device=device, batch_size=tcfg.eval_batch_size, n_classes=tcfg.n_classes
    )
    metrics["val"] = evaluate_model(
        model, val_rec, val_idx, device=device, batch_size=tcfg.eval_batch_size, n_classes=tcfg.n_classes
    )
    metrics["test"] = evaluate_model(
        model, test_rec, test_idx, device=device, batch_size=tcfg.eval_batch_size, n_classes=tcfg.n_classes
    )

    print(
        f"[eval] train acc {metrics['train']['accuracy']:.4f} | "
        f"val acc {metrics['val']['accuracy']:.4f} | "
        f"test acc {metrics['test']['accuracy']:.4f}"
    )

    extra = {
        "history": result.history,
        "best_epoch": result.best_epoch,
        "best_score": result.best_score,
        "select_by": result.select_by,
        "split_info": recs["split_info"],
        "dataset": recs["name"],
        "seed": int(cfg.get_path("seed", 0)),
        "device": describe_device(device),
        "n_parameters": n_params,
        "hidden_health": health_train,
    }
    ckpt = checkpoint_path(cfg, tag)
    model.save(str(ckpt), extra=extra)
    print(f"[save] checkpoint -> {ckpt}")

    bundle = {
        "tag": tag,
        "dataset": recs["name"],
        "seed": int(cfg.get_path("seed", 0)),
        "wall_time_s": wall,
        "best_epoch": result.best_epoch,
        "best_score": result.best_score,
        "select_by": result.select_by,
        "n_train": result.n_train,
        "n_val": result.n_val,
        "split_info": recs["split_info"],
        "device": describe_device(device),
        "n_parameters": n_params,
        "hidden_health_train": health_train,
        "metrics": metrics,
        "config": cfg.to_dict(),
    }
    save_json(bundle, result_path(cfg, f"{tag}_train_summary.json"))
    save_json(result.history, result_path(cfg, f"{tag}_history.json"))
    _write_history_csv(result.history, result_path(cfg, f"{tag}_history.csv"))
    print(f"[save] results  -> {result_path(cfg, f'{tag}_train_summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
