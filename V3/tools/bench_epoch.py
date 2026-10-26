"""Epoch-level throughput benchmark for the V3 training loop.

Measures the *current* code path (no registry row, no checkpoints, no figures):
one or more genuine ``train_one_epoch`` passes over the FIT split, with the
device-wide GPU utilisation sampled from ``nvidia-smi`` in a background thread,
and with the time spent in ``SHDEventStore.batch`` (host-side binning /
cache reads) accumulated separately from the rest of the step.

The reference protocol used in ``V3/PERF.md`` is::

    .venv/Scripts/python.exe V3/tools/bench_epoch.py --epochs 2

which is ``N=64``, ``D=1000``, 2 ms bins (T=500), batch 128, FIT split only
(7340 samples, 58 batches at batch 128).

``--data-only`` skips the model and only times the host-side batch preparation
(cold = the first epoch, before the RAM event cache is warm; warm = later
epochs), which is the component the TODO blames for the GPU idling.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

V3_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = V3_ROOT.parent
for p in (str(V3_ROOT), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_experiment import DEFAULT_CONFIG, apply_overrides  # noqa: E402
from v3.config import V3Config  # noqa: E402
from v3.data import SHDEventStore, make_split  # noqa: E402

# the protocol from the task brief / V3/TODO.md §Performance
PROTOCOL = dict(
    n_neurons=64,
    state_dim=1000,
    batch_size=128,
    epochs=1,
    tag="perf_bench",
)


# ---------------------------------------------------------------------- #
class GpuSampler(threading.Thread):
    """Sample device-wide GPU utilisation (and used memory) from nvidia-smi."""

    def __init__(self, interval_s: float = 0.25):
        super().__init__(daemon=True)
        self.interval_s = float(interval_s)
        self.samples: list[tuple[float, float]] = []
        self._stop_flag = threading.Event()

    def run(self) -> None:  # pragma: no cover - timing helper
        while not self._stop_flag.is_set():
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                line = out.stdout.strip().splitlines()[0]
                util, mem = [float(v) for v in line.split(",")]
                self.samples.append((util, mem))
            except Exception:
                pass
            self._stop_flag.wait(self.interval_s)

    def stop(self) -> dict:
        self._stop_flag.set()
        self.join(timeout=10)
        if not self.samples:
            return {"util_mean": float("nan"), "util_max": float("nan"), "used_mb_max": float("nan")}
        utils = [s[0] for s in self.samples]
        mems = [s[1] for s in self.samples]
        return {
            "util_mean": float(np.mean(utils)),
            "util_max": float(np.max(utils)),
            "used_mb_max": float(np.max(mems)),
            "n_samples": len(utils),
        }


class TimedStore:
    """Proxy that accumulates wall time spent inside ``store.batch``."""

    def __init__(self, store: SHDEventStore):
        self._store = store
        self.batch_seconds = 0.0
        self.n_batches = 0

    def __getattr__(self, name):
        return getattr(self._store, name)

    def batch(self, indices):
        t0 = time.perf_counter()
        out = self._store.batch(indices)
        self.batch_seconds += time.perf_counter() - t0
        self.n_batches += 1
        return out

    def reset(self) -> None:
        self.batch_seconds = 0.0
        self.n_batches = 0


# ---------------------------------------------------------------------- #
def build(cfg: V3Config):
    import torch

    from v3.experiment import resolve_device
    from v3.model import VectorNeuronPopulation
    from v3.train import set_seed

    device = resolve_device(cfg)
    set_seed(cfg.seed)
    store = SHDEventStore(cfg.path(cfg.train_h5), cfg)
    fit_split, val_split = make_split(store, cfg)
    if cfg.max_train_samples and cfg.max_train_samples < len(fit_split):
        from v3.data import Split

        sub = fit_split.indices[: cfg.max_train_samples]
        fit_split = Split("fit_subset", sub, store.labels[sub], store.speakers[sub])
    model = VectorNeuronPopulation(cfg).to(device)
    probe = store.batch(fit_split.indices[: min(cfg.batch_size, len(fit_split))])
    model.calibrate_threshold(
        torch.from_numpy(probe).to(device, non_blocking=True), target_rate_hz=cfg.rate_target_hz
    )
    return device, store, fit_split, val_split, model


def data_only(store: SHDEventStore, split, cfg: V3Config, epochs: int) -> list[dict]:
    rows = []
    for e in range(1, epochs + 1):
        rng = np.random.default_rng([cfg.seed, e])
        idx = split.indices
        idx = idx[rng.permutation(idx.size)] if cfg.shuffle_train else idx
        t0 = time.perf_counter()
        n_batches = 0
        for start in range(0, idx.size, cfg.batch_size):
            store.batch(idx[start : start + cfg.batch_size])
            n_batches += 1
        dt = time.perf_counter() - t0
        rows.append(
            {
                "epoch": e,
                "seconds": dt,
                "batches": n_batches,
                "ms_per_batch": 1000.0 * dt / max(1, n_batches),
                "samples_per_s": idx.size / dt,
            }
        )
        print(
            f"[bench] data-only epoch {e}: {dt:.2f}s  {1000.0 * dt / max(1, n_batches):.1f} ms/batch "
            f"({idx.size / dt:.0f} samples/s)",
            flush=True,
        )
    return rows


def full_epochs(store: SHDEventStore, split, cfg: V3Config, model, device, epochs: int) -> list[dict]:
    import torch

    from v3.train import make_optimizer, make_scheduler, train_one_epoch

    ts = TimedStore(store)
    optimizer = make_optimizer(model, cfg)
    steps = len(range(0, split.indices.size, cfg.batch_size))
    scheduler = make_scheduler(optimizer, cfg, steps)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.amp) and device.type == "cuda")
    rows = []
    for e in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        ts.reset()
        sampler = GpuSampler()
        sampler.start()
        t0 = time.perf_counter()
        rec = train_one_epoch(
            model, ts, split, cfg, optimizer, scheduler, scaler, device, e, progress=None
        )
        dt = time.perf_counter() - t0
        gpu = sampler.stop()
        peak_alloc = peak_res = 0.0
        if device.type == "cuda":
            peak_alloc = float(torch.cuda.max_memory_allocated() / 2**20)
            peak_res = float(torch.cuda.max_memory_reserved() / 2**20)
        row = {
            "epoch": e,
            "seconds": dt,
            "samples": int(rec["n_samples"]),
            "samples_per_s": rec["n_samples"] / dt,
            "data_seconds": ts.batch_seconds,
            "data_fraction": ts.batch_seconds / dt,
            "data_ms_per_batch": 1000.0 * ts.batch_seconds / max(1, ts.n_batches),
            "peak_allocated_mb": peak_alloc,
            "peak_reserved_mb": peak_res,
            "accuracy": rec["accuracy"],
            "loss": rec["loss"],
            **gpu,
        }
        rows.append(row)
        print(
            f"[bench] epoch {e}: {dt:.2f}s  {row['samples_per_s']:.1f} samples/s  "
            f"data {row['data_seconds']:.2f}s ({100.0 * row['data_fraction']:.0f}%)  "
            f"util mean {row['util_mean']:.0f}% max {row['util_max']:.0f}%  "
            f"peak {peak_alloc:.0f}/{peak_res:.0f} MiB",
            flush=True,
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="V3 epoch throughput benchmark")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--data-only", action="store_true")
    ap.add_argument("--no-protocol", action="store_true", help="do not apply the N=64/D=1000/batch=128 preset")
    ap.add_argument("--label", default="run")
    ap.add_argument("--json", default=None, help="write the measurements to this JSON file")
    args = ap.parse_args(argv)

    cfg = V3Config.from_yaml(args.config)
    if not args.no_protocol:
        cfg = cfg.with_overrides(**PROTOCOL)
    cfg = apply_overrides(cfg, args.override)
    device, store, fit_split, val_split, model = build(cfg)
    print(
        f"[bench] {args.label}: N={cfg.n_neurons} D={cfg.state_dim} T={cfg.n_bins} "
        f"batch={cfg.batch_size} FIT={len(fit_split)} device={device} "
        f"binned_cache={getattr(cfg, 'binned_cache', False)}",
        flush=True,
    )
    t0 = time.perf_counter()
    if args.data_only:
        rows = data_only(store, fit_split, cfg, args.epochs)
    else:
        rows = full_epochs(store, fit_split, cfg, model, device, args.epochs)
    payload = {
        "label": args.label,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": dict(
            n_neurons=cfg.n_neurons,
            state_dim=cfg.state_dim,
            n_bins=cfg.n_bins,
            bin_ms=cfg.bin_ms,
            batch_size=cfg.batch_size,
            n_fit=len(fit_split),
            binned_cache=bool(getattr(cfg, "binned_cache", False)),
            data_only=bool(args.data_only),
            device=str(device),
        ),
        "epochs": rows,
        "wall_seconds": time.perf_counter() - t0,
    }
    if str(device).startswith("cuda"):
        import torch

        payload["gpu"] = torch.cuda.get_device_name(0)
        payload["torch"] = torch.__version__
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[bench] wrote {args.json}", flush=True)
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
