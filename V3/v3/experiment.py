"""Orchestration of one V3 variant (vector ``D=1000`` or scalar ``D=1``).

Both the notebook and the headless CLI call exactly these two functions, so the
numbers reported in the notebook and from the command line come from the same
code path::

    v = train_variant(cfg)      # FIT split only: never sees the official test set
    r = test_variant(v)         # official SHD test set, frozen parameters

The FIT/VAL split is carved out of the official *training* file; the official
test file is only ever opened inside :func:`test_variant`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import plots
from .config import V3Config, V3_ROOT
from .data import SHDEventStore, Split, make_split, official_test_split
from .model import VectorNeuronPopulation
from .train import FitResult, evaluate, fit, save_json, set_seed


def resolve_device(cfg: V3Config) -> torch.device:
    want = str(cfg.device).lower()
    if want == "auto":
        want = "cuda" if torch.cuda.is_available() else "cpu"
    if want.startswith("cuda") and not torch.cuda.is_available():
        print("[V3] CUDA requested but unavailable -> falling back to CPU/float32")
        want = "cpu"
    return torch.device(want)


def effective_config(cfg: V3Config, device: torch.device) -> V3Config:
    """CPU cannot run the fp16 path usefully; fall back to float32 (debug mode)."""
    if device.type == "cpu" and cfg.dtype != "float32":
        print("[V3] CPU device -> forcing dtype=float32 (fp16 CPU is a debug-only path)")
        cfg = cfg.with_overrides(dtype="float32")
    return cfg.resolved()


# ---------------------------------------------------------------------- #
@dataclass
class Variant:
    """Everything one trained variant produced (in memory + on disk)."""

    cfg: V3Config
    device: torch.device
    store: SHDEventStore
    fit_split: Split
    val_split: Split
    model: VectorNeuronPopulation
    result: FitResult
    fit_metrics: dict
    val_metrics: dict
    checkpoint_path: Path
    dirs: dict = field(default_factory=dict)
    figures: dict = field(default_factory=dict)
    wall_seconds: float = 0.0

    @property
    def label(self) -> str:
        return f"D={self.cfg.state_dim} (N={self.cfg.n_neurons})"

    def history_rows(self) -> list[dict]:
        return self.result.history


def variant_dirs(cfg: V3Config) -> dict:
    out = cfg.v3_path(cfg.out_dir)
    ck = cfg.v3_path(cfg.ckpt_dir)
    fg = cfg.v3_path(cfg.figure_dir)
    run_dir = out / cfg.tag
    return {
        "results": run_dir,
        "checkpoints": ck,
        "figures": fg,
        "run_figures": fg / cfg.tag,
    }


# ---------------------------------------------------------------------- #
def train_variant(cfg: V3Config, device: torch.device | None = None, show_progress: bool = True) -> Variant:
    """Load SHD, build the population, train it, save checkpoint + history + curves."""
    t_start = time.perf_counter()
    device = device or resolve_device(cfg)
    cfg = effective_config(cfg, device)
    dirs = variant_dirs(cfg)
    for p in (dirs["results"], dirs["checkpoints"], dirs["run_figures"]):
        p.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)
    store = SHDEventStore(cfg.path(cfg.train_h5), cfg)
    fit_split, val_split = make_split(store, cfg)
    if cfg.max_train_samples and cfg.max_train_samples < len(fit_split):
        sub = fit_split.indices[: cfg.max_train_samples]
        fit_split = Split("fit_subset", sub, store.labels[sub], store.speakers[sub])
        print(f"[V3] QUICK MODE: training on the first {len(fit_split)} FIT samples only")
    print(f"[V3] {cfg.tag}: FIT {len(fit_split)} / VAL {len(val_split)} samples "
          f"(official test set not touched)")

    model = VectorNeuronPopulation(cfg).to(device)
    # initialise the spike thresholds so every neuron starts at ~rate_target_hz
    probe = store.batch(fit_split.indices[: min(cfg.batch_size, len(fit_split))])
    model.calibrate_threshold(
        torch.from_numpy(probe).to(device, non_blocking=True), target_rate_hz=cfg.rate_target_hz
    )
    print(f"[V3] threshold calibrated; parameters = {model.n_parameters():,}")

    result = fit(model, store, fit_split, val_split, cfg, device, show_progress=show_progress)

    fit_metrics = evaluate(model, store, fit_split, cfg, device)
    val_metrics = evaluate(model, store, val_split, cfg, device) if len(val_split) else {}
    val_metrics.pop("logits", None)
    val_metrics.pop("labels", None)

    ckpt_path = dirs["checkpoints"] / f"{cfg.tag}.pt"
    torch.save(
        {
            "schema": "v3_checkpoint/v1",
            "config": cfg.to_dict(),
            "state_dict": model.state_dict(),
            "best_val_state_dict": result.state_dict,
            "best_epoch": result.best_epoch,
            "best_val_accuracy": result.best_val_accuracy,
            "fit_metrics": fit_metrics,
            "val_metrics": val_metrics,
        },
        ckpt_path,
    )
    if result.state_dict is not None:
        torch.save(
            {"schema": "v3_checkpoint_bestval/v1", "config": cfg.to_dict(),
             "state_dict": result.state_dict, "epoch": result.best_epoch},
            dirs["checkpoints"] / f"{cfg.tag}_bestval.pt",
        )
    csv_path = dirs["results"] / "history.csv"
    _write_history_csv(result.history, csv_path)
    save_json({"config": cfg.to_dict(), "parameters": model.param_groups(),
               "n_parameters": model.n_parameters()},
              dirs["results"] / "config.json")
    save_json(
        {
            "fit": fit_metrics,
            "val": val_metrics,
            "n_parameters": model.n_parameters(),
            "parameter_groups": model.param_groups(),
            "wall_seconds": result.total_seconds,
            "dtypes": result.dtypes,
        },
        dirs["results"] / "train_metrics.json",
    )
    tfig = plots.plot_training(
        result.history, dirs["run_figures"] / "training_curves.png",
        title=f"V3 {cfg.tag}: N={cfg.n_neurons}, D={cfg.state_dim}, T={cfg.n_bins}",
    )
    afigs = save_activity_figures(store, fit_split, model, device, dirs["run_figures"],
                                 cfg, n=1, prefix="activity_fit_sample")
    wall = time.perf_counter() - t_start
    print(
        f"[V3] {cfg.tag} trained: {cfg.epochs} epochs in {result.total_seconds:.1f}s  "
        f"(FIT acc {fit_metrics['accuracy']:.4f}, VAL acc {val_metrics.get('accuracy', float('nan')):.4f})"
    )
    return Variant(
        cfg=cfg, device=device, store=store, fit_split=fit_split, val_split=val_split,
        model=model, result=result, fit_metrics=fit_metrics, val_metrics=val_metrics,
        checkpoint_path=ckpt_path, dirs=dirs,
        figures={"training": tfig, "activity_fit": afigs}, wall_seconds=wall,
    )


def _write_history_csv(history: list[dict], path: Path) -> None:
    keys = sorted({k for h in history for k in h})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(keys) + "\n")
        for h in history:
            fh.write(",".join(str(h.get(k, "")) for k in keys) + "\n")


# ---------------------------------------------------------------------- #
@torch.no_grad()
def test_variant(
    v: Variant,
    which: str = "final",
    save: bool = True,
    show_progress: bool = True,
    activity_samples: int = 3,
) -> dict:
    """Evaluate the frozen model on **every** official SHD test sample."""
    from tqdm.auto import tqdm

    from .data import BatchIterator

    cfg, device = v.cfg, v.device
    model = v.model
    if which == "best_val":
        if v.result.state_dict is None:
            raise ValueError("no best-validation state was kept for this run")
        model.load_state_dict(v.result.state_dict)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    store = SHDEventStore(cfg.path(cfg.test_h5), cfg)
    split = official_test_split(store)
    print(f"[V3] {cfg.tag}: official TEST {len(split)} samples (speakers "
          f"{split.summary()['speakers']})")

    bar = tqdm(total=len(BatchIterator(store, split, cfg.batch_size, shuffle=False)),
               desc="test", unit="batch", leave=False, disable=not show_progress, mininterval=2.0)
    metrics = evaluate(model, store, split, cfg, device, collect_predictions=True,
                       progress=bar, phase="test")
    bar.close()

    logits = metrics.pop("logits")
    labels = metrics.pop("labels")
    indices = metrics.pop("indices")
    preds = logits.argmax(axis=1)
    correct = int((preds == labels).sum())
    total = int(labels.size)
    acc = correct / total
    n_classes = int(cfg.n_classes)

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)))
    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)

    metrics.update(
        {
            "test_correct": correct,
            "test_total": total,
            "test_accuracy": acc,
            "test_errors": total - correct,
            "which": which,
            "checkpoint": str(v.checkpoint_path),
            "confusion_matrix": cm,
            "per_class_accuracy": per_class,
            "predictions": preds,
            "labels": labels,
            "logits": logits,
            "indices": indices,
        }
    )
    print(f"[V3] {cfg.tag} ({which}) TEST: correct {correct} / total {total} = {acc * 100:.2f}%")

    if save:
        run_fig = v.dirs["run_figures"]
        cfig = plots.plot_confusion(cm, run_fig / "test_confusion_matrix.png", acc,
                                    title=f"V3 {cfg.tag} ({which})")
        pfig = plots.plot_per_class(per_class, run_fig / "test_per_class_accuracy.png",
                                    title=f"V3 {cfg.tag} ({which})")
        v.figures["confusion"] = cfig
        v.figures["per_class"] = pfig
        np.savez_compressed(
            v.dirs["results"] / "test_predictions.npz",
            indices=indices, labels=labels, predictions=preds, logits=logits,
            confusion_matrix=cm, per_class_accuracy=per_class,
        )
        save_json(
            {k: val for k, val in metrics.items()
             if k not in ("confusion_matrix", "per_class_accuracy", "predictions", "labels", "logits")},
            v.dirs["results"] / "test_metrics.json",
        )
        save_json({"confusion_matrix": cm.tolist(), "per_class_accuracy": per_class.tolist()},
                  v.dirs["results"] / "test_confusion.json")
        v.figures["activity"] = save_activity_figures(
            store, split, model, device, v.dirs["run_figures"], cfg,
            n=activity_samples, prefix="activity_test_sample",
        )
    store.close()
    return metrics


def save_activity_figures(store, split, model, device, out_dir: Path, cfg,
                          n: int = 3, prefix: str = "activity") -> list[Path]:
    """Input raster + spike raster + population rate for representative samples.

    ``store`` and ``split`` must belong together (the test store is never used to
    index a fit split).
    """
    idx = split.indices[:n]
    x = store.batch(idx)
    with torch.no_grad():
        _, spikes = model(torch.from_numpy(x).to(device, non_blocking=True))
    sp = spikes.float().cpu().numpy()
    labels = store.labels_of(idx)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(len(idx)):
        paths.append(
            plots.plot_sample_activity(
                x[i], sp[i], out_dir / f"{prefix}{i}.png", bin_ms=cfg.bin_ms,
                title=f"V3 {cfg.tag}: {split.name} sample (true class {int(labels[i])}, "
                      f"{int(sp[i].sum())} population spikes, "
                      f"{int(x[i].sum())} input cells)",
            )
        )
    return paths


__all__ = [
    "Variant",
    "train_variant",
    "test_variant",
    "save_activity_figures",
    "resolve_device",
    "effective_config",
    "variant_dirs",
]
