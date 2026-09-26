"""Training loop for the recurrent SNN.

The training code is intentionally conventional: Adam + cross-entropy on the
accumulated readout, optional L2 penalties and an optional homeostatic
firing-rate penalty that keeps the hidden population analysable. The point of the
project is what happens *after* training, so reliability and observability matter
more than squeezing out the last percent of accuracy.

Every epoch we log training loss/accuracy, validation loss/accuracy **and** the
hidden-circuit diagnostics (mean firing rate, fraction of silent neurons). The
best-validation checkpoint is selected on validation data only; the official test
set is never touched here.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .data import SHDRecordings, iterate_batches
from .evaluation import evaluate_model
from .model import RecurrentLIFSNN
from .utils import get_device


@dataclass
class TrainConfig:
    """Optimisation hyper-parameters (all configurable from YAML/CLI)."""

    lr: float = 1e-3
    optimizer: str = "adam"  # "adam" | "adamw" | "sgd"
    weight_decay: float = 0.0
    epochs: int = 20
    batch_size: int = 128
    l2_in: float = 0.0
    l2_rec: float = 0.0
    l2_out: float = 0.0
    l2_spikes: float = 0.0  # homeostatic penalty on |hidden rate - target|
    target_rate_hz: float = 5.0
    grad_clip: float = 1.0
    scheduler: str = "cosine"  # "none" | "cosine" | "step"
    min_lr_factor: float = 0.01
    warmup_epochs: int = 0
    early_stopping_patience: int = 0  # 0 disables early stopping
    select_by: str = "dev_accuracy"  # "dev_accuracy" | "dev_loss"
    eval_batch_size: int = 256
    n_classes: int = 20

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "TrainConfig":
        valid = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        mapping = {k: v for k, v in dict(mapping or {}).items() if k in valid}
        return cls(**mapping)

    @classmethod
    def from_config(cls, cfg: Any, prefix: str = "train") -> "TrainConfig":
        node = cfg.get_path(prefix, {}) or {}
        return cls.from_mapping(dict(node))


@dataclass
class TrainResult:
    """Everything the training stage produces, ready for serialisation."""

    history: list[dict[str, Any]]
    best_epoch: int
    best_score: float
    best_state_dict: dict[str, Any]
    select_by: str
    wall_time_s: float
    n_train: int
    n_dev: int
    config: dict[str, Any] = field(default_factory=dict)

    def history_records(self) -> list[dict[str, Any]]:
        return self.history


def _make_optimizer(model: RecurrentLIFSNN, tcfg: TrainConfig) -> Any:
    import torch

    groups: list[dict[str, Any]] = [{"params": [model.w_in, model.w_rec, model.w_out, model.b_out]}]
    extras = [p for n, p in model.named_parameters() if n in ("b_hid", "log_tau_offset")]
    if extras:
        groups[0]["params"] = groups[0]["params"] + extras
    if tcfg.optimizer == "adam":
        return torch.optim.Adam(groups, lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    if tcfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    if tcfg.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=tcfg.lr, momentum=0.9, weight_decay=tcfg.weight_decay)
    raise ValueError(f"Unknown optimizer {tcfg.optimizer!r}")


def _make_scheduler(optimizer: Any, tcfg: TrainConfig) -> Any:
    import torch

    if tcfg.scheduler == "none":
        return None
    if tcfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(tcfg.epochs, 1), eta_min=tcfg.lr * tcfg.min_lr_factor
        )
    if tcfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(tcfg.epochs // 4, 1), gamma=0.5)
    raise ValueError(f"Unknown scheduler {tcfg.scheduler!r}")


def compute_loss(
    model: RecurrentLIFSNN,
    out: Mapping[str, Any],
    y: Any,
    tcfg: TrainConfig,
) -> Any:
    """Cross-entropy plus the configured regularisation terms."""
    import torch.nn.functional as F

    loss = F.cross_entropy(out["logits"], y)
    if tcfg.l2_in > 0:
        loss = loss + tcfg.l2_in * (model.w_in**2).mean()
    if tcfg.l2_rec > 0:
        loss = loss + tcfg.l2_rec * (model.w_rec**2).mean()
    if tcfg.l2_out > 0:
        loss = loss + tcfg.l2_out * (model.w_out**2).mean()
    if tcfg.l2_spikes > 0:
        duration_s = max(model.cfg.duration_ms / 1000.0, 1e-9)
        rate = out["spike_count"].mean(dim=0) / duration_s
        loss = loss + tcfg.l2_spikes * ((rate - tcfg.target_rate_hz) ** 2).mean()
    return loss


def train_model(
    model: RecurrentLIFSNN,
    train_rec: SHDRecordings,
    train_idx: np.ndarray,
    dev_rec: SHDRecordings,
    dev_idx: np.ndarray,
    tcfg: TrainConfig,
    *,
    device: Any = None,
    seed: int = 0,
    verbose: bool = True,
    on_epoch: Callable[[dict[str, Any]], None] | None = None,
) -> TrainResult:
    """Train ``model`` in place and return history + the best-dev weights.

    ``dev_rec``/``dev_idx`` are the speaker-aware development split used for model
    selection (checkpoint / early stopping). The ``probe`` split and the official
    test set are *not* used here; they are reported separately by ``train.py``.

    Parameters
    ----------
    select_by:
        ``"dev_accuracy"`` (maximise) or ``"dev_loss"`` (minimise). The official
        test set is not used for this selection; ``scripts/evaluate.py`` reports
        test metrics only for the already-frozen checkpoint.
    """
    import torch

    device = device or get_device()
    model = model.to(device)
    sim = model.cfg
    optimizer = _make_optimizer(model, tcfg)
    scheduler = _make_scheduler(optimizer, tcfg)
    rng = np.random.default_rng(seed)

    history: list[dict[str, Any]] = []
    best_score = -np.inf if tcfg.select_by == "dev_accuracy" else np.inf
    best_epoch = -1
    best_state: dict[str, Any] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    epochs_without_improvement = 0
    t_start = time.time()

    for epoch in range(tcfg.epochs):
        model.train()
        t_epoch_start = time.time()
        epoch_seed = int(rng.integers(0, 2**31 - 1))
        losses: list[float] = []
        n_correct = 0
        n_seen = 0
        spike_rate = np.zeros(sim.n_hidden, dtype=np.float64)
        grad_norms: list[float] = []
        n_batches = 0
        for batch in iterate_batches(
            train_rec,
            train_idx,
            batch_size=tcfg.batch_size,
            n_bins=sim.n_bins,
            bin_ms=sim.bin_ms,
            shuffle=True,
            seed=epoch_seed,
            device=None,
        ):
            x = batch["x"].to(device=device, dtype=torch.float32)
            y = batch["y"].to(device=device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = compute_loss(model, out, y, tcfg)
            loss.backward()
            if tcfg.grad_clip and tcfg.grad_clip > 0:
                # clip_grad_norm_ returns the total (pre-clip) gradient norm.
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
                grad_norms.append(float(total_norm))
            else:
                with torch.no_grad():
                    total_norm = torch.sqrt(
                        torch.stack(
                            [p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None]
                        ).sum()
                    )
                grad_norms.append(float(total_norm))
            optimizer.step()

            losses.append(float(loss.item()))
            pred = out["logits"].detach().argmax(dim=1)
            n_correct += int((pred == y).sum().item())
            n_seen += int(y.numel())
            spike_rate += out["spike_count"].detach().mean(dim=0).double().cpu().numpy()
            n_batches += 1

        if scheduler is not None:
            scheduler.step()

        train_loss = float(np.mean(losses)) if losses else float("nan")
        train_acc = float(n_correct / n_seen) if n_seen else float("nan")
        rates_hz = spike_rate / max(n_batches, 1) / max(sim.duration_ms / 1000.0, 1e-9)
        grad_norm_mean = float(np.mean(grad_norms)) if grad_norms else float("nan")
        grad_norm_max = float(np.max(grad_norms)) if grad_norms else float("nan")

        dev_metrics = evaluate_model(
            model, dev_rec, dev_idx, device=device, batch_size=tcfg.eval_batch_size,
            n_classes=tcfg.n_classes, compute_confusion=False,
        )
        record = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "dev_loss": dev_metrics["loss"],
            "dev_accuracy": dev_metrics["accuracy"],
            "train_hidden_rate_hz": float(rates_hz.mean()),
            "dev_hidden_rate_hz": dev_metrics["hidden_mean_rate_hz"],
            "dev_hidden_silent_fraction": dev_metrics["hidden_silent_fraction"],
            "grad_norm_mean": grad_norm_mean,
            "grad_norm_max": grad_norm_max,
            "epoch_time_s": float(time.time() - t_epoch_start),
        }
        history.append(record)

        score = record["dev_accuracy"] if tcfg.select_by == "dev_accuracy" else -record["dev_loss"]
        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose:
            print(
                f"[train] epoch {epoch:3d} | train loss {train_loss:.4f} acc {train_acc:.4f} "
                f"| dev loss {record['dev_loss']:.4f} acc {record['dev_accuracy']:.4f} "
                f"| rate {record['train_hidden_rate_hz']:.2f} Hz "
                f"| silent {record['dev_hidden_silent_fraction']:.2f} "
                f"| grad {grad_norm_mean:.3f}"
                + ("  *" if improved else "")
            )
        if on_epoch is not None:
            on_epoch(record)

        if tcfg.early_stopping_patience and epochs_without_improvement >= tcfg.early_stopping_patience:
            if verbose:
                print(f"[train] early stopping after {epochs_without_improvement} epochs without improvement")
            break

    # Restore the best-dev weights so downstream analysis sees the selected model.
    model.load_state_dict(best_state)
    return TrainResult(
        history=history,
        best_epoch=best_epoch,
        best_score=float(best_score),
        best_state_dict=best_state,
        select_by=tcfg.select_by,
        wall_time_s=float(time.time() - t_start),
        n_train=int(np.asarray(train_idx).size),
        n_dev=int(np.asarray(dev_idx).size),
        config=tcfg.to_dict(),
    )


def activity_report(model: RecurrentLIFSNN, rec: SHDRecordings, indices: np.ndarray, *, device: Any = None) -> dict[str, Any]:
    """Quick circuit-health check: hidden firing rates, silent fraction, entropy.

    Run this before any expensive analysis. A network whose hidden units are all
    silent produces a degenerate representation and the geometry analysis would be
    vacuous; the diagnostics here are logged with every metric file so this cannot
    pass unnoticed.
    """
    from .evaluation import collect_activity

    device = device or get_device()
    res = collect_activity(model, rec, indices, device=device, with_labels=False)
    duration_s = max(res.duration_s, 1e-9)
    rates = res.counts.sum(axis=0) / (res.n_samples * duration_s)
    return {
        "mean_rate_hz": float(rates.mean()),
        "median_rate_hz": float(np.median(rates)),
        "max_rate_hz": float(rates.max()),
        "min_rate_hz": float(rates.min()),
        "silent_neuron_fraction": float((res.counts.sum(axis=0) <= 0).mean()),
        "low_rate_neuron_fraction_below_0p5hz": float((rates < 0.5).mean()),
        "mean_spike_count_per_sample": float(res.counts.sum(axis=1).mean()),
        "n_samples": int(res.n_samples),
    }
