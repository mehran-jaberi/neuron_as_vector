"""Persistent experiment registry for V3.

Every completed run appends exactly one row to ``V3/results/runs.csv`` and gets a
timestamped directory under ``V3/results/`` holding the exact configuration and
metrics that produced it::

    V3/results/
        runs.csv
        2026-10-02_00-15-30/
            config.yaml
            metrics.json
            confusion_matrix.csv
            summary.txt

The registry is deliberately small: it is a thin, testable wrapper around a CSV
append plus a per-run directory, not a database.  Model checkpoints are *not*
copied here (the existing checkpoints stay where the training code already saves
them and the path is recorded in the row).
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

RUN_FIELDS = [
    "timestamp",
    "run_id",
    "tag",
    "N",
    "D",
    "parameters",
    "epochs",
    "seed",
    "batch_size",
    "learning_rate",
    "optimizer",
    "scheduler",
    "training_precision",
    "device",
    "n_bins",
    "bin_ms",
    "state_regularization_mode",
    "state_noise_enabled",
    "state_noise_std",
    "state_quantization_enabled",
    "state_quantization_bits",
    "state_reg_apply_validation",
    "state_reg_apply_test",
    "train_accuracy",
    "val_accuracy",
    "best_val_accuracy",
    "best_epoch",
    "test_accuracy",
    "test_correct",
    "test_total",
    "english_accuracy",
    "german_accuracy",
    "train_seconds",
    "duration_seconds",
    "checkpoint",
    "notes",
]


def format_run_id(when: datetime) -> str:
    """``2026-10-02_00-15-30`` from the *local* system time."""
    return when.strftime("%Y-%m-%d_%H-%M-%S")


def _json_default(o):
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class RunRegistry:
    """Append-only CSV registry + per-run artifact directory."""

    def __init__(self, results_dir: str | Path):
        self.results_dir = Path(results_dir)
        self.runs_csv = self.results_dir / "runs.csv"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    def new_run_id(self, when: datetime | None = None) -> str:
        """Unique, timestamp-based run id; disambiguated if it already exists."""
        base = format_run_id(when or datetime.now())
        run_id = base
        k = 1
        while (self.results_dir / run_id).exists():
            k += 1
            run_id = f"{base}_{k:02d}"
        return run_id

    # ------------------------------------------------------------------ #
    def record(
        self,
        row: dict,
        *,
        config_yaml: str | None = None,
        metrics: dict | None = None,
        confusion_matrix=None,
        per_class_accuracy=None,
        summary: str | None = None,
        timestamp: datetime | None = None,
    ) -> Path:
        """Append one run record and write its artifact directory.

        Returns the run directory path.  Writes ``runs.csv`` (append), and
        ``config.yaml``, ``metrics.json``, ``confusion_matrix.csv`` and
        ``summary.txt`` when the corresponding data is supplied.
        """
        when = timestamp or datetime.now()
        run_id = self.new_run_id(when)
        run_dir = self.results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        full = {k: row.get(k, "") for k in RUN_FIELDS}
        full["timestamp"] = when.isoformat(timespec="seconds")
        full["run_id"] = run_id
        # keep any extra caller-supplied columns after the canonical ones
        for k, v in row.items():
            if k not in full:
                full[k] = v

        self._append_csv(full)

        if config_yaml is not None:
            (run_dir / "config.yaml").write_text(config_yaml, encoding="utf-8")
        if metrics is not None:
            (run_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2, default=_json_default), encoding="utf-8"
            )
        if confusion_matrix is not None:
            self._write_confusion_csv(confusion_matrix, run_dir / "confusion_matrix.csv")
        (run_dir / "summary.txt").write_text(
            summary if summary is not None else self.format_summary(full), encoding="utf-8"
        )
        return run_dir

    # ------------------------------------------------------------------ #
    def _append_csv(self, row: dict) -> None:
        exists = self.runs_csv.exists()
        fieldnames = list(RUN_FIELDS) + [k for k in row if k not in RUN_FIELDS]
        with open(self.runs_csv, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _write_confusion_csv(cm, path: Path) -> None:
        cm = [[int(v) for v in row] for row in cm]
        n = len(cm)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["true\\pred"] + [str(j) for j in range(n)])
            for i, r in enumerate(cm):
                writer.writerow([str(i)] + r)

    # ------------------------------------------------------------------ #
    @staticmethod
    def format_summary(row: dict) -> str:
        lines = [f"run_id: {row.get('run_id', '')}", f"timestamp: {row.get('timestamp', '')}"]
        for k in RUN_FIELDS:
            if k in ("run_id", "timestamp"):
                continue
            lines.append(f"{k}: {row.get(k, '')}")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------ #
    def read_runs(self) -> list[dict]:
        if not self.runs_csv.exists():
            return []
        with open(self.runs_csv, "r", newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))


def split_accuracy(confusion_matrix, english_classes: int = 10) -> tuple[float, float]:
    """(english, german) accuracy from a confusion matrix, classes 0..9 / 10..19.

    English digits are labels 0-9 and German digits are labels 10-19 in SHD.
    Returns NaNs when a group has no samples.
    """
    import numpy as np

    cm = np.asarray(confusion_matrix, dtype=float)
    n = cm.shape[0]
    k = min(int(english_classes), n)
    out = []
    for lo, hi in ((0, k), (k, n)):
        block = cm[lo:hi, lo:hi]
        total = block.sum()
        out.append(float(block.diagonal().sum() / total) if total > 0 else float("nan"))
    return out[0], out[1]


def top_confusion_pairs(confusion_matrix, k: int = 10) -> list[dict]:
    """The ``k`` largest off-diagonal confusions: (true, pred, count)."""
    import numpy as np

    cm = np.asarray(confusion_matrix, dtype=float)
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append({"true": int(i), "pred": int(j), "count": int(cm[i, j])})
    pairs.sort(key=lambda p: p["count"], reverse=True)
    return pairs[: int(k)]


def run_row_from_variant(
    *,
    cfg,
    n_parameters: int,
    fit_metrics: dict | None = None,
    val_metrics: dict | None = None,
    test_metrics: dict | None = None,
    train_seconds: float = float("nan"),
    duration_seconds: float = float("nan"),
    checkpoint: str = "",
    notes: str = "",
) -> dict:
    """Build a registry row from a completed :class:`v3.experiment.Variant` run."""
    reg = cfg.state_regularization
    english = german = float("nan")
    row = {k: "" for k in RUN_FIELDS}
    row.update(
        {
            "tag": cfg.tag,
            "N": int(cfg.n_neurons),
            "D": int(cfg.state_dim),
            "parameters": int(n_parameters),
            "epochs": int(cfg.epochs),
            "seed": int(cfg.seed),
            "batch_size": int(cfg.batch_size),
            "learning_rate": float(cfg.learning_rate),
            "optimizer": cfg.optimizer,
            "scheduler": cfg.lr_schedule,
            "training_precision": str(cfg.dtype),
            "device": str(cfg.device),
            "n_bins": int(cfg.n_bins),
            "bin_ms": float(cfg.bin_ms),
            "state_regularization_mode": reg.mode,
            "state_noise_enabled": bool(reg.noise_enabled),
            "state_noise_std": float(reg.noise_std),
            "state_quantization_enabled": bool(reg.quantization_enabled),
            "state_quantization_bits": int(reg.quantize_bits),
            "state_reg_apply_validation": bool(reg.apply_during_validation),
            "state_reg_apply_test": bool(reg.apply_during_test),
            "train_accuracy": float((fit_metrics or {}).get("accuracy", float("nan"))),
            "val_accuracy": float((val_metrics or {}).get("accuracy", float("nan"))),
            "train_seconds": float(train_seconds),
            "duration_seconds": float(duration_seconds),
            "checkpoint": str(checkpoint),
            "notes": notes,
        }
    )
    if test_metrics:
        cm = test_metrics.get("confusion_matrix")
        if cm is not None:
            english, german = split_accuracy(cm)
        row.update(
            test_accuracy=float(test_metrics.get("test_accuracy", float("nan"))),
            test_correct=int(test_metrics.get("test_correct", 0)),
            test_total=int(test_metrics.get("test_total", 0)),
            english_accuracy=english,
            german_accuracy=german,
        )
    return row


__all__ = [
    "RunRegistry",
    "RUN_FIELDS",
    "format_run_id",
    "split_accuracy",
    "top_confusion_pairs",
    "run_row_from_variant",
]
