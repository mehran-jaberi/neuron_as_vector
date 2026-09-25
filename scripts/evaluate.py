"""Evaluate a trained checkpoint on the train/val/test splits.

Stage 4b of the pipeline. Reports accuracy, loss, per-class accuracy and the
confusion matrix. The official test split is only ever *read* here, never used to
inform any modelling choice; the summary is written with that warning attached.

Example
-------
    uv run python scripts/evaluate.py --config configs/analysis.yaml --tag baseline
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from _common import (  # noqa: E402
    announce,
    apply_seed,
    build_recordings,
    checkpoint_path,
    load_checkpoint,
    load_run_config,
    parse_args,
    resolve_device,
    result_path,
)
from src.evaluation import evaluate_model
from src.utils import ensure_dir, save_json


def _write_confusion(confusion: list[list[int]], classes: list[int], path: Path) -> None:
    ensure_dir(path.parent)
    array = np.asarray(confusion, dtype=np.int64)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["true\\pred", *[str(c) for c in classes]])
        for i, row in enumerate(array):
            writer.writerow([str(classes[i]), *[int(v) for v in row]])


def main(argv: list[str] | None = None) -> int:
    args = parse_args("Evaluate a trained checkpoint.", "configs/analysis.yaml")
    cfg = load_run_config(args)
    device = resolve_device(cfg)
    announce(cfg, device)
    apply_seed(cfg, device)

    tag = str(cfg.get_path("run.tag", "baseline"))
    model, extra = load_checkpoint(cfg, device, tag)
    n_classes = int(cfg.get_path("train.n_classes", model.cfg.n_output))

    recs = build_recordings(cfg)
    splits = {"train": recs["train"], "val": recs["val"], "test": recs["test"]}

    summary: dict = {"tag": tag, "dataset": recs["name"], "n_classes": n_classes, "splits": {}}
    classes = list(range(n_classes))
    for name, rec in splits.items():
        idx = np.arange(len(rec))
        metrics = evaluate_model(
            model, rec, idx, device=device, batch_size=int(cfg.get_path("train.eval_batch_size", 256)),
            n_classes=n_classes, compute_confusion=True,
        )
        summary["splits"][name] = metrics
        print(
            f"[eval] {name:5s} n={metrics['n_samples']:5d} "
            f"acc={metrics['accuracy']:.4f} loss={metrics['loss']:.4f} "
            f"silent={metrics['hidden_silent_fraction']:.3f}"
        )
        confusion = metrics.get("confusion_matrix")
        if confusion is not None:
            _write_confusion(confusion, classes, result_path(cfg, f"{tag}_confusion_{name}.csv"))

    summary["warning"] = (
        "The 'test' split corresponds to the official SHD test set. It is reported for "
        "transparency only and was never used for any training or model-selection decision."
    )
    out = result_path(cfg, f"{tag}_evaluation.json")
    save_json(summary, out)
    print(f"[save] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
