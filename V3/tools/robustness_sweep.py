#!/usr/bin/env python
"""Test-time robustness sweep for a frozen checkpoint.

Perturbs the *input* at inference time and measures how accuracy decays.  This
is deliberately separate from the training-time augmentation workstream: the
augmentation code lives in ``v3/event_augmentation.py`` and only ever touches
training batches, while here nothing is trained and the checkpoint stays frozen.

Three families, all applied to the binned input tensor ``(B, T, C)``:

* ``offset``    shift every channel by k bins (k = 1, 2, 5, 10 bins = 2..20 ms)
* ``poisson``   add independent Poisson background events at 5/10/20/50 Hz
* ``dropout``   delete a fraction p of the input events (p = 0.05..0.30)

A condition is only meaningful next to a baseline run through *this* script, so
the baseline (no perturbation) is always recomputed and stored.

    python V3/tools/robustness_sweep.py --checkpoint V3/checkpoints/archive_20plus/v3_n100_spec_ep30_ep30.pt
    python V3/tools/robustness_sweep.py --checkpoint V3/checkpoints/v3_vector_scalar_d1.pt

Writes ``V3/results/analysis/robustness_<tag>.json``.  No registry row: this is
an evaluation of an existing recorded run, not a new run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "V3"))

from v3.data import BatchIterator, SHDEventStore, official_test_split  # noqa: E402
from v3.model import VectorNeuronPopulation  # noqa: E402

from run_analysis import _to_device, load_checkpoint  # noqa: E402

OUT = ROOT / "V3" / "results" / "analysis"

OFFSETS_MS = [2, 4, 10, 20]
POISSON_HZ = [0.5, 1.0, 2.0, 5, 10, 20, 50]
DROPOUT_P = [0.05, 0.10, 0.20, 0.30]


def _shift(x: torch.Tensor, bins: int) -> torch.Tensor:
    if bins <= 0:
        return x
    out = torch.zeros_like(x)
    out[:, bins:] = x[:, :-bins]
    return out


def _poisson(x: torch.Tensor, rate_hz: float, bin_ms: float) -> torch.Tensor:
    """Add independent background spikes at ``rate_hz`` per channel."""
    p = min(rate_hz * bin_ms / 1000.0, 1.0)
    add = (torch.rand_like(x) < p).to(x.dtype)
    return torch.clamp(x + add, max=1.0)


def _dropout(x: torch.Tensor, p: float) -> torch.Tensor:
    keep = (torch.rand_like(x) >= p).to(x.dtype)
    return x * keep


@torch.no_grad()
def evaluate(model, cfg, device, store, split, batch_size, transform) -> tuple[int, int]:
    """One pass over the official TEST split under ``transform``."""
    correct = total = 0
    it = BatchIterator(store, split, batch_size, shuffle=False)
    for x_np, y_np in it:
        x = _to_device(x_np, cfg, device)
        if transform is not None:
            x = transform(x)
        logits = model(x)[0]
        pred = logits.float().argmax(1)
        y = torch.from_numpy(np.asarray(y_np)).to(pred.device)
        correct += int((pred == y).sum())
        total += int(y.numel())
    return correct, total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, cfg, device, _ = load_checkpoint(args.checkpoint, args.device)
    batch_size = int(args.batch_size or cfg.batch_size)
    store = SHDEventStore(cfg.path(cfg.test_h5), cfg)
    split = official_test_split(store)
    model.eval()
    model.set_phase("test")

    print(f"[robustness] {Path(args.checkpoint).name} | tag={cfg.tag} "
          f"N={model.n_neurons} D={model.state_dim} device={device} "
          f"n={len(split)}")

    bin_ms = float(cfg.bin_ms)
    conditions: list[tuple[str, str, float, object]] = [("baseline", "none", 0.0, None)]
    for ms in OFFSETS_MS:
        bins = int(round(ms / bin_ms))
        conditions.append(("offset_ms", "offset", float(ms),
                           (lambda b: (lambda x: _shift(x, b)))(bins)))
    for hz in POISSON_HZ:
        conditions.append(("poisson_hz", "poisson", float(hz),
                           (lambda r: (lambda x: _poisson(x, r, bin_ms)))(hz)))
    for p in DROPOUT_P:
        conditions.append(("dropout_p", "dropout", float(p),
                           (lambda q: (lambda x: _dropout(x, q)))(p)))

    rows = []
    for family, kind, value, transform in conditions:
        tic = time.perf_counter()
        correct, total = evaluate(model, cfg, device, store, split, batch_size,
                                  transform)
        rows.append({
            "family": family,
            "kind": kind,
            "value": value,
            "correct": correct,
            "total": total,
            "accuracy": correct / max(total, 1),
            "seconds": round(time.perf_counter() - tic, 2),
        })
        print(f"[robustness] {family:11s} {value:6.2f} -> {correct}/{total} "
              f"= {100 * correct / total:.2f}%  ({rows[-1]['seconds']:.1f}s)")

    payload = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_tag": cfg.tag,
        "n_neurons": int(model.n_neurons),
        "state_dim": int(model.state_dim),
        "bin_ms": bin_ms,
        "n_test": int(len(split)),
        "device": str(device),
        "seed": args.seed,
        "protocol": "input perturbation after binning; model frozen; no training",
        "rows": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"robustness_{cfg.tag}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"[robustness] wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
