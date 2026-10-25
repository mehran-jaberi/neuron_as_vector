"""Model evaluation and hidden-activity collection.

Two responsibilities:

1. :func:`evaluate_model` - accuracy / loss / confusion, plus the *state* of the
   hidden circuit (mean firing rate, fraction of silent neurons). Those circuit
   diagnostics matter scientifically: a network that reaches high accuracy with
   90 % dead neurons would make the neuron-representation analysis meaningless.
2. :class:`ActivityAccumulator` - streams the hidden layer over a dataset and
   accumulates exactly the sufficient statistics needed downstream, without ever
   materialising a ``(N, T, H)`` tensor (which would be several GB):

   * per-sample spike counts ``(N, H)`` -> firing-rate variance, Fano factor,
     fraction of silent samples;
   * pooled PSTH ``(H, T)`` -> label-free temporal activity features;
   * class-conditioned PSTH ``(C, H, T)``, class counts and class-wise first-spike
     statistics -> the functional fingerprint.

   Keeping the label-free and label-conditioned accumulators in the same object
   but in separate fields makes it obvious which output may feed the
   representation (``psth``, ``counts``) and which must stay an evaluation target
   (``class_*``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .data import SHDRecordings, iterate_batches
from .model import RecurrentLIFSNN, SNNConfig
from .utils import get_device


# --------------------------------------------------------------------------
# Accumulator
# --------------------------------------------------------------------------
@dataclass
class ActivityAccumulatorResult:
    """Sufficient statistics collected by :class:`ActivityAccumulator`."""

    counts: np.ndarray  # (N, H) spike counts per sample
    psth: np.ndarray  # (H, T) label-free pooled spike-time histogram
    class_psth: np.ndarray  # (C, H, T) class-conditioned PSTH
    class_counts: np.ndarray  # (C, H)
    class_n: np.ndarray  # (C,)
    first_spike_sum: np.ndarray  # (H,) sum of per-sample first-spike times (ms)
    first_spike_count: np.ndarray  # (H,)
    class_first_spike_sum: np.ndarray  # (C, H)
    class_first_spike_count: np.ndarray  # (C, H)
    v_mean: np.ndarray  # (H,) mean membrane potential
    n_samples: int
    n_bins: int
    n_hidden: int
    bin_ms: float
    # Global membrane-potential statistics over all recorded (sample, time, neuron)
    # entries. Defaulted so older saved payloads (without them) still load.
    v_global_mean: float = 0.0
    v_global_std: float = 0.0
    v_global_min: float = 0.0
    v_global_max: float = 0.0
    labels: np.ndarray | None = None  # (N,) kept only for bookkeeping/audit
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.n_bins * self.bin_ms / 1000.0

    @property
    def duration_ms(self) -> float:
        return self.n_bins * self.bin_ms

    def save(self, path: str) -> None:
        path = str(path)
        payload = {
            "counts": self.counts.astype(np.float32),
            "psth": self.psth.astype(np.float32),
            "class_psth": self.class_psth.astype(np.float32),
            "class_counts": self.class_counts.astype(np.float32),
            "class_n": self.class_n.astype(np.int64),
            "first_spike_sum": self.first_spike_sum.astype(np.float32),
            "first_spike_count": self.first_spike_count.astype(np.float32),
            "class_first_spike_sum": self.class_first_spike_sum.astype(np.float32),
            "class_first_spike_count": self.class_first_spike_count.astype(np.float32),
            "v_mean": self.v_mean.astype(np.float32),
            "n_samples": np.array([self.n_samples]),
            "n_bins": np.array([self.n_bins]),
            "n_hidden": np.array([self.n_hidden]),
            "bin_ms": np.array([self.bin_ms]),
            "v_global_mean": np.array([self.v_global_mean], dtype=np.float64),
            "v_global_std": np.array([self.v_global_std], dtype=np.float64),
            "v_global_min": np.array([self.v_global_min], dtype=np.float64),
            "v_global_max": np.array([self.v_global_max], dtype=np.float64),
            "labels": self.labels if self.labels is not None else np.array([], dtype=np.int64),
        }
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str) -> "ActivityAccumulatorResult":
        with np.load(str(path), allow_pickle=False) as data:
            labels = data["labels"]

            def _scalar(key: str, default: float) -> float:
                return float(data[key][0]) if key in data.files else default

            return cls(
                counts=data["counts"].astype(np.float64),
                psth=data["psth"].astype(np.float64),
                class_psth=data["class_psth"].astype(np.float64),
                class_counts=data["class_counts"].astype(np.float64),
                class_n=data["class_n"].astype(np.float64),
                first_spike_sum=data["first_spike_sum"].astype(np.float64),
                first_spike_count=data["first_spike_count"].astype(np.float64),
                class_first_spike_sum=data["class_first_spike_sum"].astype(np.float64),
                class_first_spike_count=data["class_first_spike_count"].astype(np.float64),
                v_mean=data["v_mean"].astype(np.float64),
                n_samples=int(data["n_samples"][0]),
                n_bins=int(data["n_bins"][0]),
                n_hidden=int(data["n_hidden"][0]),
                bin_ms=float(data["bin_ms"][0]),
                v_global_mean=_scalar("v_global_mean", 0.0),
                v_global_std=_scalar("v_global_std", 0.0),
                v_global_min=_scalar("v_global_min", 0.0),
                v_global_max=_scalar("v_global_max", 0.0),
                labels=None if labels.size == 0 else labels.astype(np.int64),
            )


class ActivityAccumulator:
    """Streaming accumulator of hidden-layer statistics.

    Parameters
    ----------
    n_hidden, n_bins, n_classes, bin_ms:
        Shapes / timing of the simulation.
    collect_voltage:
        Also track the mean hidden membrane potential per neuron (a cheap
        excitability descriptor).
    """

    def __init__(
        self,
        n_hidden: int,
        n_bins: int,
        n_classes: int,
        bin_ms: float,
        *,
        collect_voltage: bool = True,
    ) -> None:
        self.n_hidden = int(n_hidden)
        self.n_bins = int(n_bins)
        self.n_classes = int(n_classes)
        self.bin_ms = float(bin_ms)
        self.collect_voltage = bool(collect_voltage)

        self._counts: list[np.ndarray] = []
        self._labels: list[np.ndarray] = []
        self.psth = np.zeros((self.n_hidden, self.n_bins), dtype=np.float64)
        self.class_psth = np.zeros((self.n_classes, self.n_hidden, self.n_bins), dtype=np.float64)
        self.class_counts = np.zeros((self.n_classes, self.n_hidden), dtype=np.float64)
        self.class_n = np.zeros(self.n_classes, dtype=np.float64)
        self.first_spike_sum = np.zeros(self.n_hidden, dtype=np.float64)
        self.first_spike_count = np.zeros(self.n_hidden, dtype=np.float64)
        self.class_first_spike_sum = np.zeros((self.n_classes, self.n_hidden), dtype=np.float64)
        self.class_first_spike_count = np.zeros((self.n_classes, self.n_hidden), dtype=np.float64)
        self._v_sum = np.zeros(self.n_hidden, dtype=np.float64)
        self._v_n = 0
        # Global (pooled over sample/time/neuron) membrane-potential accumulators.
        self._v_global_sum = 0.0
        self._v_global_sq = 0.0
        self._v_global_min = np.inf
        self._v_global_max = -np.inf
        self._v_global_count = 0
        self.n_samples = 0
        self._t_ms = np.arange(self.n_bins, dtype=np.float64) * self.bin_ms

    # ------------------------------------------------------------------
    def update(
        self,
        hidden_spikes: np.ndarray | Any,
        labels: np.ndarray | Any | None = None,
        hidden_v: np.ndarray | Any | None = None,
    ) -> None:
        """Add one batch of hidden spikes ``(B, T, H)`` (and optionally voltages)."""
        spikes = np.asarray(hidden_spikes, dtype=np.float64)
        if spikes.ndim != 3:
            raise ValueError(f"Expected hidden spikes of shape (B, T, H), got {spikes.shape}")
        B, T, H = spikes.shape
        if H != self.n_hidden:
            raise ValueError(f"Accumulator configured for {self.n_hidden} neurons, got {H}")
        if T != self.n_bins:
            raise ValueError(f"Accumulator configured for {self.n_bins} time bins, got {T}")

        counts = spikes.sum(axis=1)  # (B, H)
        self._counts.append(counts)
        self.psth += spikes.sum(axis=0).T  # (H, T)

        # First spike per sample (ms); non-spiking samples contribute nothing.
        has_spike = counts > 0
        first_idx = np.argmax(spikes > 0, axis=1)  # 0 where no spike, ignored via mask
        first_ms = first_idx.astype(np.float64) * self.bin_ms
        self.first_spike_sum += np.where(has_spike, first_ms, 0.0).sum(axis=0)
        self.first_spike_count += has_spike.sum(axis=0)

        if labels is not None:
            y = np.asarray(labels).astype(np.int64).reshape(-1)
            if y.size != B:
                raise ValueError("labels must have one entry per sample in the batch")
            self._labels.append(y)
            for cls in np.unique(y):
                sel = y == cls
                if cls < 0 or cls >= self.n_classes:
                    continue
                block = spikes[sel]
                self.class_psth[cls] += block.sum(axis=0).T  # (H, T)
                self.class_counts[cls] += counts[sel].sum(axis=0)
                self.class_n[cls] += float(sel.sum())
                hs = has_spike[sel]
                fi = first_ms[sel]
                self.class_first_spike_sum[cls] += np.where(hs, fi, 0.0).sum(axis=0)
                self.class_first_spike_count[cls] += hs.sum(axis=0)

        if hidden_v is not None and self.collect_voltage:
            v = np.asarray(hidden_v, dtype=np.float64)
            self._v_sum += v.mean(axis=(0, 1))
            self._v_n += 1
            # Pooled global membrane-potential statistics (running, memory-bounded).
            self._v_global_sum += float(v.sum())
            self._v_global_sq += float((v * v).sum())
            if v.size:
                self._v_global_min = min(self._v_global_min, float(v.min()))
                self._v_global_max = max(self._v_global_max, float(v.max()))
                self._v_global_count += int(v.size)

        self.n_samples += B

    # ------------------------------------------------------------------
    def finalize(self) -> ActivityAccumulatorResult:
        if not self._counts:
            raise RuntimeError("No data was accumulated; call update() at least once.")
        counts = np.concatenate(self._counts, axis=0)
        labels = np.concatenate(self._labels, axis=0) if self._labels else None
        v_mean = self._v_sum / self._v_n if self._v_n > 0 else np.zeros(self.n_hidden)
        if self._v_global_count > 0:
            vg_mean = self._v_global_sum / self._v_global_count
            vg_var = max(self._v_global_sq / self._v_global_count - vg_mean * vg_mean, 0.0)
            v_global_mean = float(vg_mean)
            v_global_std = float(np.sqrt(vg_var))
            v_global_min = float(self._v_global_min)
            v_global_max = float(self._v_global_max)
        else:
            v_global_mean = v_global_std = v_global_min = v_global_max = 0.0
        return ActivityAccumulatorResult(
            counts=counts,
            psth=self.psth.copy(),
            class_psth=self.class_psth.copy(),
            class_counts=self.class_counts.copy(),
            class_n=self.class_n.copy(),
            first_spike_sum=self.first_spike_sum.copy(),
            first_spike_count=self.first_spike_count.copy(),
            class_first_spike_sum=self.class_first_spike_sum.copy(),
            class_first_spike_count=self.class_first_spike_count.copy(),
            v_mean=v_mean,
            n_samples=int(self.n_samples),
            n_bins=int(self.n_bins),
            n_hidden=int(self.n_hidden),
            bin_ms=float(self.bin_ms),
            v_global_mean=v_global_mean,
            v_global_std=v_global_std,
            v_global_min=v_global_min,
            v_global_max=v_global_max,
            labels=labels,
            meta={"collect_voltage": self.collect_voltage},
        )


# --------------------------------------------------------------------------
# Running the model over data
# --------------------------------------------------------------------------
def _to_device(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    import torch

    return {
        "x": batch["x"].to(device=device, dtype=torch.float32),
        "y": batch["y"].to(device=device),
        "idx": batch["idx"],
    }


def evaluate_model(
    model: RecurrentLIFSNN,
    rec: SHDRecordings,
    indices: Sequence[int] | np.ndarray,
    *,
    device: Any = None,
    batch_size: int = 256,
    n_classes: int = 20,
    compute_confusion: bool = True,
) -> dict[str, Any]:
    """Accuracy / loss / circuit diagnostics for a model on a set of samples."""
    import torch
    import torch.nn.functional as F

    device = device or get_device()
    model = model.to(device)
    model.eval()
    sim = model.cfg
    total = 0
    correct = 0
    loss_sum = 0.0
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64) if compute_confusion else None
    spike_rate_sum = np.zeros(model.cfg.n_hidden, dtype=np.float64)
    n_batches = 0

    with torch.no_grad():
        for batch in iterate_batches(
            rec, indices, batch_size=batch_size, n_bins=sim.n_bins, bin_ms=sim.bin_ms, device=None
        ):
            b = _to_device(batch, device)
            out = model(b["x"])
            loss = F.cross_entropy(out["logits"], b["y"])
            pred = out["logits"].argmax(dim=1)
            total += b["y"].numel()
            correct += int((pred == b["y"]).sum().item())
            loss_sum += float(loss.item()) * b["y"].numel()
            spike_rate_sum += out["spike_count"].mean(dim=0).double().cpu().numpy()
            n_batches += 1
            if confusion is not None:
                y_cpu = b["y"].cpu().numpy()
                p_cpu = pred.cpu().numpy()
                np.add.at(confusion, (y_cpu, p_cpu), 1)

    mean_spike_count = spike_rate_sum / max(n_batches, 1)
    duration_s = sim.duration_ms / 1000.0
    rates = mean_spike_count / max(duration_s, 1e-9)
    metrics: dict[str, Any] = {
        "n_samples": int(total),
        "accuracy": float(correct / total) if total else float("nan"),
        "loss": float(loss_sum / total) if total else float("nan"),
        "hidden_mean_rate_hz": float(rates.mean()) if rates.size else float("nan"),
        "hidden_max_rate_hz": float(rates.max()) if rates.size else float("nan"),
        "hidden_silent_fraction": float((rates <= 0).mean()) if rates.size else float("nan"),
        "hidden_rate_hz": rates.tolist(),
    }
    if confusion is not None:
        metrics["confusion_matrix"] = confusion.tolist()
        per_class = []
        for c in range(n_classes):
            denom = confusion[c].sum()
            per_class.append(float(confusion[c, c] / denom) if denom > 0 else float("nan"))
        metrics["per_class_accuracy"] = per_class
    return metrics


def collect_activity(
    model: RecurrentLIFSNN,
    rec: SHDRecordings,
    indices: Sequence[int] | np.ndarray,
    *,
    device: Any = None,
    batch_size: int = 256,
    n_classes: int = 20,
    with_labels: bool = True,
    collect_voltage: bool = True,
) -> ActivityAccumulatorResult:
    """Run the model over ``indices`` and accumulate hidden-layer statistics.

    ``with_labels=False`` produces a strictly label-free accumulator, which is the
    form used when building the activity part of the neuron representation. When
    ``with_labels=True`` the class-conditioned statistics used for the functional
    fingerprint are filled as well.

    Runs under ``torch.no_grad()`` and records the full hidden trace only for the
    current batch, so peak memory is one batch rather than the whole dataset.
    """
    import torch

    device = device or get_device()
    model = model.to(device)
    model.eval()
    sim = model.cfg
    acc = ActivityAccumulator(
        n_hidden=sim.n_hidden, n_bins=sim.n_bins, n_classes=n_classes, bin_ms=sim.bin_ms,
        collect_voltage=collect_voltage,
    )
    with torch.no_grad():
        for batch in iterate_batches(
            rec, indices, batch_size=batch_size, n_bins=sim.n_bins, bin_ms=sim.bin_ms, device=None
        ):
            b = _to_device(batch, device)
            out = model(b["x"], record=True)
            acc.update(
                out["hidden_spikes"].cpu().numpy(),
                labels=b["y"].cpu().numpy() if with_labels else None,
                hidden_v=out["hidden_v"].cpu().numpy() if collect_voltage else None,
            )
    return acc.finalize()


def circuit_health(
    model: RecurrentLIFSNN,
    rec: SHDRecordings,
    indices: Sequence[int] | np.ndarray,
    *,
    device: Any = None,
    batch_size: int = 256,
    n_classes: int = 20,
) -> dict[str, Any]:
    """Comprehensive circuit-health diagnostics for the hidden layer.

    Reports, for the hidden population on ``indices``:
      * hidden firing-rate distribution (per-neuron rates + summary percentiles),
      * silent-neuron fraction,
      * spike counts (total, and mean per sample),
      * membrane-potential statistics (per-neuron mean V and pooled global
        mean/std/min/max over all recorded (sample, time, neuron) entries).

    This is label-free (no class information is used).
    """
    res = collect_activity(
        model, rec, indices, device=device, batch_size=batch_size,
        n_classes=n_classes, with_labels=False, collect_voltage=True,
    )
    duration_s = max(res.duration_s, 1e-9)
    # Per-neuron firing rate (Hz): total spikes / (n_samples * duration).
    rates = res.counts.sum(axis=0) / (res.n_samples * duration_s)
    rates = np.asarray(rates, dtype=np.float64)
    total_spikes = res.counts.sum(axis=0)
    q = [0.0, 10.0, 25.0, 50.0, 75.0, 90.0, 100.0]
    pctl = np.percentile(rates, q) if rates.size else np.full(len(q), np.nan)
    return {
        "n_samples": int(res.n_samples),
        "n_hidden": int(res.n_hidden),
        "duration_s": float(res.duration_s),
        # firing-rate distribution
        "rate_hz_mean": float(rates.mean()) if rates.size else float("nan"),
        "rate_hz_std": float(rates.std()) if rates.size else float("nan"),
        "rate_hz_percentiles": {f"p{int(p)}": float(v) for p, v in zip(q, pctl)},
        "rate_hz_per_neuron": rates.tolist(),
        # silent neurons
        "silent_neuron_fraction": float((total_spikes <= 0).mean()) if rates.size else float("nan"),
        "n_silent_neurons": int((total_spikes <= 0).sum()) if rates.size else 0,
        "low_rate_fraction_lt_0p5hz": float((rates < 0.5).mean()) if rates.size else float("nan"),
        # spike counts
        "total_spikes": int(total_spikes.sum()) if rates.size else 0,
        "mean_total_spikes_per_sample": float(res.counts.sum(axis=1).mean()) if res.counts.size else float("nan"),
        "mean_spike_count_per_neuron_per_sample": float(res.counts.mean()) if res.counts.size else float("nan"),
        # membrane potential statistics
        "v_mean_per_neuron": np.asarray(res.v_mean, dtype=np.float64).tolist(),
        "v_global_mean": float(res.v_global_mean),
        "v_global_std": float(res.v_global_std),
        "v_global_min": float(res.v_global_min),
        "v_global_max": float(res.v_global_max),
    }


def split_half_indices(
    labels: np.ndarray, *, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Split sample indices into two class-stratified halves.

    Used to estimate the noise ceiling of the functional fingerprint: the two
    halves are independent measurements of the same quantity, so their agreement
    bounds how well any representation could predict the fingerprint.
    """
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    a: list[np.ndarray] = []
    b: list[np.ndarray] = []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        idx = rng.permutation(idx)
        half = idx.size // 2
        a.append(np.sort(idx[:half]))
        b.append(np.sort(idx[half:]))
    return (
        np.concatenate(a) if a else np.empty(0, dtype=np.int64),
        np.concatenate(b) if b else np.empty(0, dtype=np.int64),
    )
