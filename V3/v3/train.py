"""Training / evaluation loops for the V3 vector-neuron population.

Conventional choices on purpose:

* optimizer   : Adam (AdamW/SGD also wired) on the fp32 master parameters
* loss        : cross-entropy (+ optional firing-rate regulariser), fp32
* precision   : fp16 for the state and every large activation, fp32 for the
                loss, the logits, the surrogate-gradient backward and the
                optimizer state; ``torch.amp.GradScaler`` for fp16 underflow
* BPTT        : full backprop-through-time with chunked gradient checkpointing

The test set is only ever touched by :func:`evaluate` when the caller passes the
official test split, and never inside :func:`fit`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .config import V3Config
from .data import BatchIterator, SHDEventStore, Split
from .model import VectorNeuronPopulation


# ---------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_optimizer(model: torch.nn.Module, cfg: V3Config) -> torch.optim.Optimizer:
    kwargs = dict(lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    return torch.optim.SGD(model.parameters(), momentum=0.9, **kwargs)


def make_scheduler(optimizer, cfg: V3Config, steps_per_epoch: int):
    if cfg.lr_schedule == "none":
        return None
    total = max(1, steps_per_epoch * int(cfg.epochs))
    warm = int(max(1, round(float(cfg.warmup_frac) * total)))

    def lr_lambda(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(1, total - warm)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, prog)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------- #
def _to_device(x: np.ndarray, cfg: V3Config, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(x)
    if cfg.dtype != "float32":
        # the binned SHD input is 0/1, exactly representable in fp16: the host
        # cast is lossless and halves the host->device transfer
        t = t.to(cfg.torch_dtype)
    return t.to(device, non_blocking=True)


@torch.no_grad()
def _rate_stats(spikes: torch.Tensor, cfg: V3Config) -> dict:
    """Per-neuron firing-rate statistics for a whole split.

    ``spikes`` is ``(n_samples, T, N)``.  The rate is the mean per-step spike
    probability divided by the bin duration; the mean must therefore be taken over
    **both** the sample and the time axis (summing over the split and dividing only
    by ``T`` would inflate the rate by the number of samples).
    """
    s = spikes.float()
    n_steps = s.shape[1]
    counts = s.sum(dim=(0, 1))
    per_step = counts / max(1, s.shape[0]) / max(1, n_steps)
    rate = per_step / (cfg.bin_ms / 1000.0)
    return {
        "rate_mean_hz": float(rate.mean()),
        "rate_median_hz": float(rate.median()),
        "rate_max_hz": float(rate.max()),
        "rate_p90_hz": float(rate.quantile(0.9)),
        "silent_neurons": int((counts == 0).sum()),
        "spikes_per_sample": float(s.sum(dim=(1, 2)).mean()),
    }


# ---------------------------------------------------------------------- #
def train_one_epoch(
    model: VectorNeuronPopulation,
    store: SHDEventStore,
    split: Split,
    cfg: V3Config,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
    epoch: int,
    progress=None,
) -> dict:
    """One genuine pass over the training split.  ``progress`` is a tqdm bar.

    The FIT/VAL split is fixed for the whole run; only the *order* of the FIT
    batches changes between epochs (``cfg.shuffle_train``), seeded by
    ``(cfg.seed, epoch)`` so a given seed reproduces the same order.
    """
    model.train()
    model.set_phase("train")
    it = BatchIterator(store, split, cfg.batch_size, shuffle=cfg.shuffle_train,
                       seed=cfg.seed, epoch=epoch)
    total_loss = total_ce = 0.0
    correct = total = 0
    spikes_seen = 0
    t0 = time.perf_counter()
    for x_np, y_np in it:
        x = _to_device(x_np, cfg, device)
        y = torch.from_numpy(y_np).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, spikes = model(x)
        loss, ce = model.loss(logits, spikes, y)
        scaler.scale(loss).backward()
        if cfg.grad_clip and cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        bs = y.shape[0]
        total_loss += float(loss.detach()) * bs
        total_ce += float(ce.detach()) * bs
        correct += int((logits.detach().argmax(-1) == y).sum())
        total += bs
        spikes_seen += int(spikes.detach().float().sum())
        if progress is not None:
            progress.update(1)
            progress.set_postfix(
                loss=f"{total_loss / max(total, 1):.3f}",
                acc=f"{correct / max(total, 1):.3f}",
            )
        if cfg.log_every and (progress is None or progress.n % cfg.log_every == 0):
            print(
                f"    batch {total // bs:4d}/{len(it)}  loss {total_loss / total:.4f} "
                f"acc {correct / total:.4f}",
                flush=True,
            )
    dt = time.perf_counter() - t0
    return {
        "epoch": int(epoch),
        "loss": total_loss / max(total, 1),
        "ce": total_ce / max(total, 1),
        "accuracy": correct / max(total, 1),
        "n_samples": int(total),
        "spikes_per_sample": spikes_seen / max(total, 1),
        "time_s": dt,
        "device": str(device),
    }


@torch.no_grad()
def evaluate(
    model: VectorNeuronPopulation,
    store: SHDEventStore,
    split: Split,
    cfg: V3Config,
    device: torch.device,
    collect_predictions: bool = False,
    progress=None,
    phase: str = "val",
) -> dict:
    """Loss / accuracy / rate statistics over a whole split.  No gradients.

    ``phase`` ("val" | "test") selects whether controlled state imprecision is
    active during evaluation.  It defaults to "val", so the official test set must
    be evaluated with ``phase="test"`` explicitly -- and even then noise /
    quantization are off unless the experiment turns them on for test.
    """
    model.eval()
    model.set_phase(phase)
    it = BatchIterator(store, split, cfg.batch_size, shuffle=cfg.shuffle_val, epoch=0)
    total_loss = total = correct = 0
    spk_all = []
    logits_all, labels_all = [], []
    t0 = time.perf_counter()
    for x_np, y_np in it:
        x = _to_device(x_np, cfg, device)
        y = torch.from_numpy(y_np).to(device)
        logits, spikes = model(x)
        loss, _ = model.loss(logits, spikes, y)
        bs = y.shape[0]
        total_loss += float(loss) * bs
        correct += int((logits.argmax(-1) == y).sum())
        total += bs
        spk_all.append(spikes.float().cpu())
        if collect_predictions:
            logits_all.append(logits.float().cpu().numpy())
            labels_all.append(y_np)
        if progress is not None:
            progress.update(1)
    dt = time.perf_counter() - t0
    spikes = torch.cat(spk_all, dim=0) if spk_all else torch.zeros(0, 1, model.n_neurons)
    out = {
        "split": split.name,
        "n_samples": int(total),
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "correct": int(correct),
        "time_s": dt,
        "max_memory_allocated_mb": (
            float(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0
        ),
    }
    out.update(_rate_stats(spikes, cfg))
    if collect_predictions:
        out["logits"] = np.concatenate(logits_all, axis=0)
        out["labels"] = np.concatenate(labels_all, axis=0)
        out["indices"] = np.asarray(split.indices)
    return out


# ---------------------------------------------------------------------- #
@dataclass
class FitResult:
    model: VectorNeuronPopulation
    history: list[dict] = field(default_factory=list)
    best_val_accuracy: float = 0.0
    best_epoch: int = -1
    state_dict: dict | None = None
    init_seconds: float = 0.0
    total_seconds: float = 0.0
    n_parameters: int = 0
    parameter_groups: dict = field(default_factory=dict)
    device: str = ""
    dtypes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d.pop("model", None)
        d.pop("state_dict", None)
        return d


def count_parameters(model) -> dict:
    return {name: int(p.numel()) for name, p in model.named_parameters()}


def fit(
    model: VectorNeuronPopulation,
    store: SHDEventStore,
    fit_split: Split,
    val_split: Split | None,
    cfg: V3Config,
    device: torch.device,
    show_progress: bool = True,
) -> FitResult:
    """Train the population.  The official test split is never passed in here."""
    from tqdm.auto import tqdm

    set_seed(cfg.seed)
    model.to(device)
    optimizer = make_optimizer(model, cfg)
    steps_per_epoch = len(BatchIterator(store, fit_split, cfg.batch_size, shuffle=cfg.shuffle_train))
    scheduler = make_scheduler(optimizer, cfg, steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.amp) and device.type == "cuda")

    res = FitResult(
        model=model,
        n_parameters=model.n_parameters(),
        parameter_groups=model.param_groups(),
        device=str(device),
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    total_t0 = time.perf_counter()
    for epoch in range(1, int(cfg.epochs) + 1):
        bar = tqdm(
            total=steps_per_epoch,
            desc=f"epoch {epoch:3d}/{cfg.epochs}",
            unit="batch",
            leave=False,
            disable=not show_progress,
            mininterval=2.0,
        )
        tr = train_one_epoch(
            model, store, fit_split, cfg, optimizer, scheduler, scaler, device, epoch, progress=bar
        )
        bar.close()
        rec = {
            **tr,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "grad_scale": float(scaler.get_scale()) if scaler.is_enabled() else 1.0,
        }
        if val_split is not None and len(val_split) > 0:
            vb = tqdm(
                total=len(BatchIterator(store, val_split, cfg.batch_size, shuffle=False)),
                desc=f"  val {epoch:3d}",
                unit="batch",
                leave=False,
                disable=not show_progress,
                mininterval=2.0,
            )
            va = evaluate(model, store, val_split, cfg, device, progress=vb)
            vb.close()
            rec["val_loss"] = va["loss"]
            rec["val_accuracy"] = va["accuracy"]
            rec["val_rate_mean_hz"] = va["rate_mean_hz"]
            rec["train_rate_mean_hz"] = tr["spikes_per_sample"] / cfg.duration_s / cfg.n_neurons
        if device.type == "cuda":
            rec["peak_memory_mb"] = float(torch.cuda.max_memory_allocated() / 2**20)
        res.history.append(rec)
        print(
            f"epoch {epoch:3d}/{cfg.epochs}  loss {rec['loss']:.4f}  acc {rec['accuracy']:.4f}"
            + (f"  val_acc {rec.get('val_accuracy', float('nan')):.4f}" if "val_accuracy" in rec else "")
            + f"  lr {rec['lr']:.2e}  {rec['time_s']:.1f}s",
            flush=True,
        )
        if cfg.select_best_val and "val_accuracy" in rec:
            if rec["val_accuracy"] > res.best_val_accuracy:
                res.best_val_accuracy = float(rec["val_accuracy"])
                res.best_epoch = int(epoch)
                res.state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
    res.total_seconds = time.perf_counter() - total_t0
    if device.type == "cuda":
        res.dtypes = {
            "state_dtype": str(cfg.torch_dtype).replace("torch.", ""),
            "parameter_dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
            "peak_memory_allocated_mb": float(torch.cuda.max_memory_allocated() / 2**20),
            "peak_memory_reserved_mb": float(torch.cuda.max_memory_reserved() / 2**20),
        }
    return res


# ---------------------------------------------------------------------- #
def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    return str(o)


__all__ = [
    "fit",
    "train_one_epoch",
    "evaluate",
    "make_optimizer",
    "make_scheduler",
    "set_seed",
    "save_json",
    "FitResult",
    "count_parameters",
]
