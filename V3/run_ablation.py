"""Mechanism ablations for the frozen ``D=1000`` reference model.

Part A -- freeze the dynamics, fit only the readout (``w_cls``)::

    A1  trained dynamics (the checkpoint), cached exact spike counts
    A2  untrained dynamics (same config, fresh seed = 1), threshold calibrated
        on FIT exactly as training does -> reservoir control
    A3  trained weights with the state dynamics removed at the state-dict level
        (``w_rec`` / ``w_emit`` / ``w_down`` / ``w_up`` zeroed: input drive only)

Part B -- per-neuron fingerprints::

    B1  mutual information between each neuron's spike count and the class,
        plus its mean rate (CPU, from the cached capture)
    B2  single-neuron knockout sweep on official TEST (``theta_i = -1e9`` makes
        the neuron emit no spikes; it therefore also stops contributing to the
        population signal ``h`` and the recurrent drive ``c``)

Every readout is a single linear softmax fitted on FIT (Adam, zero init, no
weight decay, <= 20 epochs, early stop on VAL, best-VAL epoch restored).  Only
``w_cls`` is fitted; the dynamics runs under ``torch.no_grad()``.

Run from the repository root::

    .venv\\Scripts\\python.exe V3\\run_ablation.py freeze
    .venv\\Scripts\\python.exe V3\\run_ablation.py fingerprints
    .venv\\Scripts\\python.exe V3\\run_ablation.py knockout
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from v3.config import V3Config
from v3.data import BatchIterator, SHDEventStore, make_split, official_test_split
from v3.experiment import effective_config, resolve_device
from v3.model import VectorNeuronPopulation
from v3.train import set_seed

V3_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = V3_DIR / "results" / "analysis"
VECTOR_CHECKPOINT = V3_DIR / "checkpoints" / "v3_vector_d1000.pt"

#: the frozen reference readout reproduces this on official TEST (confirmed)
REFERENCE_CORRECT, REFERENCE_TOTAL = 1868, 2264

#: readout-only fitting protocol (fixed a priori; same optimiser/lr/batch as the
#: validated probe in ``run_analysis.py``, but no L2 because the brief asks for
#: plain cross-entropy, and at most 20 epochs with early stopping on VAL).
READOUT_EPOCHS = 20
READOUT_LR = 0.01
READOUT_BATCH = 256
READOUT_SEED = 0
READOUT_PATIENCE = 5

#: A2 uses a different ``cfg.seed`` so the init generator (``seed + 12345``)
#: differs from the trained model's (``12345``).  Everything else is identical.
A2_SEED = 1

#: ``theta`` is *added* to the potential, so silencing a neuron means driving it
#: strongly negative.  The brief says ``theta_i += 1e9``, which would pin the
#: neuron above threshold (always firing) instead.  Verified with
#: :func:`silence_neuron` and a runtime spike-count check.
SILENCE_OFFSET = -1e9

CAPTURE_SPLITS = ("fit", "val", "test")
CAPTURE_PHASE = {"fit": "val", "val": "val", "test": "test"}
MI_BINS = 8
#: KO threshold in percentage points (one accuracy point = 22.64 samples here).
DROP_THRESHOLD_PP = 1.0


# ---------------------------------------------------------------------- #
# generic helpers
# ---------------------------------------------------------------------- #
def _to_device(x: np.ndarray, cfg: V3Config, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(x)
    if cfg.dtype != "float32":
        t = t.to(cfg.torch_dtype)  # 0/1 input: lossless
    return t.to(device, non_blocking=True)


def load_payload(path: str | Path = VECTOR_CHECKPOINT) -> tuple[V3Config, dict]:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if "config" not in payload or "state_dict" not in payload:
        raise ValueError(f"{path}: not a v3 checkpoint (missing config/state_dict)")
    return V3Config.from_dict(payload["config"]), payload


def build_model(cfg: V3Config, device: torch.device, state_dict: dict | None = None) -> VectorNeuronPopulation:
    model = VectorNeuronPopulation(cfg).to(device)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def rates_hz_from_counts(counts: np.ndarray, cfg: V3Config) -> np.ndarray:
    """Per-neuron mean rate [Hz] from spike counts (``T * bin_ms`` = 1 s)."""
    duration_s = cfg.n_bins * cfg.bin_ms / 1000.0
    return np.asarray(counts, dtype=np.float64).mean(axis=0) / duration_s


def open_splits(cfg: V3Config):
    """Open the two h5 files and rebuild the documented FIT/VAL/TEST splits."""
    store_train = SHDEventStore(cfg.path(cfg.train_h5), cfg)
    store_test = SHDEventStore(cfg.path(cfg.test_h5), cfg)
    fit_split, val_split = make_split(store_train, cfg)
    test_split = official_test_split(store_test)
    return {
        "fit": (store_train, fit_split),
        "val": (store_train, val_split),
        "test": (store_test, test_split),
    }


# ---------------------------------------------------------------------- #
# part A -- frozen-dynamics capture + readout-only fitting
# ---------------------------------------------------------------------- #
@torch.no_grad()
def capture_counts(
    model: VectorNeuronPopulation,
    cfg: V3Config,
    device: torch.device,
    store: SHDEventStore,
    split,
    phase: str,
    batch_size: int = 128,
) -> dict:
    """Run one split and keep only the spike counts (the readout's feature)."""
    model.eval()
    model.set_phase(phase)
    n, N, K = len(split), model.n_neurons, int(cfg.n_classes)
    counts = np.zeros((n, N), dtype=np.float32)
    logits = np.zeros((n, K), dtype=np.float32)
    labels = np.asarray(split.labels, dtype=np.int64)
    indices = np.asarray(split.indices, dtype=np.int64)
    tic = time.perf_counter()
    pos = 0
    for x_np, y_np in BatchIterator(store, split, batch_size, shuffle=False):
        x = _to_device(x_np, cfg, device)
        logits_b, spikes_b = model(x)
        b = int(y_np.size)
        counts[pos : pos + b] = spikes_b.float().sum(1).cpu().numpy()
        logits[pos : pos + b] = logits_b.float().cpu().numpy()
        pos += b
    seconds = time.perf_counter() - tic
    preds = logits.argmax(1)
    correct = int((preds == labels).sum())
    return {
        "counts": counts,
        "labels": labels,
        "indices": indices,
        "correct": correct,
        "total": int(labels.size),
        "accuracy": correct / max(int(labels.size), 1),
        "seconds": seconds,
        "rates_hz": rates_hz_from_counts(counts, cfg),
    }


@torch.no_grad()
def sample_counts(model: VectorNeuronPopulation, cfg: V3Config, device: torch.device,
                  x_np: np.ndarray) -> np.ndarray:
    """Spike counts on one batch (used to verify that a knockout is silent)."""
    model.eval()
    _, spikes = model(_to_device(x_np, cfg, device))
    return spikes.float().sum(1).cpu().numpy()


def fit_readout(
    counts_fit: np.ndarray,
    labels_fit: np.ndarray,
    counts_val: np.ndarray,
    labels_val: np.ndarray,
    counts_test: np.ndarray,
    labels_test: np.ndarray,
    *,
    epochs: int = READOUT_EPOCHS,
    lr: float = READOUT_LR,
    batch_size: int = READOUT_BATCH,
    seed: int = READOUT_SEED,
    patience: int = READOUT_PATIENCE,
) -> dict:
    """Fit only the linear readout ``w_cls`` with plain cross-entropy.

    ``logits = spike_counts @ w``.  Adam on a zero-initialised ``(N, K)`` weight,
    at most ``epochs`` epochs, early stop on VAL accuracy (ties broken by VAL
    loss), and the best-VAL weights are restored before TEST is scored.
    """
    xf = torch.as_tensor(np.asarray(counts_fit, dtype=np.float32))
    yf = torch.as_tensor(np.asarray(labels_fit, dtype=np.int64))
    xv = torch.as_tensor(np.asarray(counts_val, dtype=np.float32))
    yv = torch.as_tensor(np.asarray(labels_val, dtype=np.int64))
    xt = torch.as_tensor(np.asarray(counts_test, dtype=np.float32))
    yt = torch.as_tensor(np.asarray(labels_test, dtype=np.int64))
    if not (xf.shape[1] == xv.shape[1] == xt.shape[1]):
        raise ValueError("feature dimensions differ between splits")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    N, K = xf.shape[1], int(max(int(yf.max()), int(yv.max()), int(yt.max()))) + 1
    w = torch.zeros(N, K, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([w], lr=float(lr))
    n = xf.shape[0]

    history: list[dict] = []
    best_w: torch.Tensor | None = None
    best = {"val_accuracy": -1.0, "val_loss": float("inf"), "epoch": -1}

    def _val() -> tuple[float, float]:
        with torch.no_grad():
            logits = xv @ w
            acc = float((logits.argmax(1) == yv).float().mean())
            loss = float(F.cross_entropy(logits, yv))
        return acc, loss

    def _test() -> tuple[int, float]:
        with torch.no_grad():
            logits = xt @ w
            correct = int((logits.argmax(1) == yt).sum())
        return correct, correct / max(int(yt.numel()), 1)

    epochs_run = 0
    for epoch in range(int(epochs)):
        epochs_run += 1
        tic = time.perf_counter()
        perm = torch.as_tensor(rng.permutation(n), dtype=torch.int64)
        for start in range(0, n, int(batch_size)):
            idx = perm[start : start + int(batch_size)]
            loss = F.cross_entropy(xf[idx] @ w, yf[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        epoch_s = time.perf_counter() - tic
        val_acc, val_loss = _val()
        improved = (val_acc > best["val_accuracy"] + 1e-12) or (
            abs(val_acc - best["val_accuracy"]) <= 1e-12 and val_loss < best["val_loss"] - 1e-12
        )
        if improved:
            best = {"val_accuracy": val_acc, "val_loss": val_loss, "epoch": epoch}
            best_w = w.detach().clone()
        history.append(
            {
                "epoch": epoch,
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "best": bool(improved),
                "seconds": epoch_s,
            }
        )
        if not improved and (epoch - best["epoch"]) >= int(patience):
            break

    if best_w is not None:
        with torch.no_grad():
            w.copy_(best_w)
    correct, accuracy = _test()
    return {
        "w": w.detach().numpy(),
        "n_features": int(N),
        "n_classes": int(K),
        "history": history,
        "best_epoch": int(best["epoch"]),
        "best_val_accuracy": float(best["val_accuracy"]),
        "epochs_run": int(epochs_run),
        "epoch_seconds_mean": float(np.mean([h["seconds"] for h in history])),
        "epoch_seconds_total": float(np.sum([h["seconds"] for h in history])),
        "test_correct": int(correct),
        "test_total": int(yt.numel()),
        "test_accuracy": float(accuracy),
    }


def readout_accuracy(counts: np.ndarray, labels: np.ndarray, w: np.ndarray) -> dict:
    """Score a fixed linear readout ``counts @ w`` (counts-only path)."""
    logits = np.asarray(counts, dtype=np.float64) @ np.asarray(w, dtype=np.float64)
    preds = logits.argmax(1)
    labels = np.asarray(labels)
    correct = int((preds == labels).sum())
    total = int(labels.size)
    return {"correct": correct, "total": total, "accuracy": correct / max(total, 1)}


def gap_pp(correct: int, total: int) -> float:
    """Accuracy gap to the 1868/2264 reference, in percentage points."""
    return 100.0 * (correct / max(total, 1) - REFERENCE_CORRECT / REFERENCE_TOTAL)


def _arm_summary(result: dict, capture_seconds: dict | None, rates: dict | None) -> dict:
    out = {
        "test_correct": int(result["test_correct"]),
        "test_total": int(result["test_total"]),
        "test_accuracy": float(result["test_accuracy"]),
        "gap_pp": float(gap_pp(result["test_correct"], result["test_total"])),
        "readout_best_epoch": int(result["best_epoch"]),
        "readout_best_val_accuracy": float(result["best_val_accuracy"]),
        "readout_epochs_run": int(result["epochs_run"]),
        "readout_epoch_seconds_mean": float(result["epoch_seconds_mean"]),
        "readout_seconds_total": float(result["epoch_seconds_total"]),
        "readout_val_curve": [
            (int(h["epoch"]), float(h["val_accuracy"])) for h in result["history"]
        ],
    }
    if capture_seconds is not None:
        out["capture_seconds"] = {
            k: float(v) if isinstance(v, (int, float)) else v for k, v in capture_seconds.items()
        }
    if rates is not None:
        out["rates_hz"] = rates
    return out


def _merge_arms(path: Path, payload: dict) -> dict:
    """Merge newly computed arms into an existing JSON so partial runs compose."""
    if not path.exists():
        return payload
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return payload
    merged = dict(existing)
    merged["protocol"] = {**existing.get("protocol", {}), **payload.get("protocol", {})}
    merged["arms"] = {**existing.get("arms", {}), **payload.get("arms", {})}
    return merged


def run_freeze(args: argparse.Namespace) -> dict:
    """A1/A2/A3: fit only the readout on frozen dynamics."""
    arms = [a.strip().lower() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ("a1", "a2", "a3")]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; choose from a1,a2,a3")

    path = Path(args.checkpoint)
    cfg, payload = load_payload(path)
    cap_tag = str(getattr(args, "capture_tag", "") or "")
    cap_prefix = f"capture_{cap_tag}" if cap_tag else "capture"
    suffix = f"_{cap_tag}" if cap_tag else ""
    reference_correct = int(getattr(args, "reference_correct", REFERENCE_CORRECT))
    out = {
        "protocol": {
            "checkpoint": str(path),
            "reference": {"correct": reference_correct, "total": REFERENCE_TOTAL,
                          "accuracy": reference_correct / REFERENCE_TOTAL},
            "readout": {
                "loss": "cross_entropy",
                "optimizer": "adam",
                "lr": READOUT_LR,
                "batch_size": READOUT_BATCH,
                "init": "zeros",
                "max_epochs": READOUT_EPOCHS,
                "early_stop": f"VAL accuracy, patience {READOUT_PATIENCE}, best-VAL weights",
                "l2": 0.0,
            },
            "a2_seed": A2_SEED,
            "a3_zeroed": ["w_rec", "w_emit", "w_down", "w_up"],
            "silence_offset": SILENCE_OFFSET,
            "note": (
                "A1 reads the cached exact counts of the frozen checkpoint "
                "(capture_*.npz); the checkpoint's own w_cls is scored on the same "
                "counts as an alignment gate."
            ),
        },
        "arms": {},
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- A1: trained dynamics, fresh readout ---------------------------
    if "a1" in arms:
        cached = {}
        for split in CAPTURE_SPLITS:
            with np.load(ANALYSIS_DIR / f"{cap_prefix}_{split}.npz") as d:
                cached[split] = (d["counts"].astype(np.float32), d["labels"].astype(np.int64))
        gate_w = payload["state_dict"]["w_cls"].float().numpy()
        gate = readout_accuracy(cached["test"][0], cached["test"][1], gate_w)
        print(
            f"[A1] cached-counts alignment gate: checkpoint w_cls "
            f"{gate['correct']}/{gate['total']} = {gate['accuracy'] * 100:.2f}%"
        )
        if gate["correct"] != reference_correct:
            raise RuntimeError(
                "cached capture does not reproduce the reference readout "
                f"({gate['correct']} != {reference_correct}); re-run `run_analysis.py capture`"
            )
        tic = time.perf_counter()
        model_result = fit_readout(
            cached["fit"][0], cached["fit"][1],
            cached["val"][0], cached["val"][1],
            cached["test"][0], cached["test"][1],
        )
        wall = time.perf_counter() - tic
        np.savez_compressed(ANALYSIS_DIR / f"ablation_readout_a1{suffix}.npz", w=model_result["w"])
        out["arms"]["A1"] = _arm_summary(
            model_result,
            capture_seconds={"fit": 0.0, "val": 0.0, "test": 0.0, "source": "cached capture"},
            rates={"test_median_hz": float(np.median(rates_hz_from_counts(cached["test"][0], cfg)))},
        )
        out["arms"]["A1"]["gate"] = gate
        out["arms"]["A1"]["wall_seconds"] = wall
        out["arms"]["A1"]["device"] = "cpu (cached counts)"
        print(
            f"[A1] trained dynamics + fresh readout: {model_result['test_correct']}/"
            f"{model_result['test_total']} = {model_result['test_accuracy'] * 100:.2f}% "
            f"(gap {gap_pp(model_result['test_correct'], model_result['test_total']):+.2f} pp, "
            f"best epoch {model_result['best_epoch']}, {wall:.1f} s total)"
        )

    # ---- A2: untrained dynamics, calibrated threshold ------------------
    if "a2" in arms:
        cfg2 = replace(cfg, seed=A2_SEED, tag=f"{cfg.tag}_untrained_seed{A2_SEED}")
        device2 = resolve_device(cfg2)
        if str(args.device) not in ("", "auto", "None"):
            device2 = torch.device(args.device)
            if device2.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
        cfg2 = effective_config(cfg2, device2)
        set_seed(A2_SEED)
        model = build_model(cfg2, device2)
        splits = open_splits(cfg2)
        probe = splits["fit"][0].batch(
            splits["fit"][1].indices[: min(cfg2.batch_size, len(splits["fit"][1]))]
        )
        rate = model.calibrate_threshold(
            torch.from_numpy(probe).to(device2), target_rate_hz=cfg2.rate_target_hz
        )
        model.eval()
        print(
            f"[A2] untrained dynamics (cfg.seed={cfg2.seed}, init generator "
            f"{cfg2.seed + 12345}); threshold calibrated on "
            f"{min(cfg2.batch_size, len(splits['fit'][1]))} FIT samples (target {rate:g} Hz)"
        )
        cap, secs = {}, {}
        for split in CAPTURE_SPLITS:
            store, sp = splits[split]
            cap[split] = capture_counts(model, cfg2, device2, store, sp, CAPTURE_PHASE[split],
                                        batch_size=int(args.batch_size))
            secs[split] = cap[split]["seconds"]
            np.savez_compressed(
                ANALYSIS_DIR / f"ablation_counts_a2{suffix}_{split}.npz",
                counts=cap[split]["counts"], labels=cap[split]["labels"],
                indices=cap[split]["indices"],
            )
            print(f"    {split}: {cap[split]['correct']}/{cap[split]['total']} with the "
                  f"(zero) readout, {secs[split]:.1f} s")
        result = fit_readout(
            cap["fit"]["counts"], cap["fit"]["labels"],
            cap["val"]["counts"], cap["val"]["labels"],
            cap["test"]["counts"], cap["test"]["labels"],
        )
        np.savez_compressed(ANALYSIS_DIR / f"ablation_readout_a2{suffix}.npz", w=result["w"])
        rates = {
            "fit_median_hz": float(np.median(cap["fit"]["rates_hz"])),
            "fit_min_hz": float(np.min(cap["fit"]["rates_hz"])),
            "fit_max_hz": float(np.max(cap["fit"]["rates_hz"])),
            "test_median_hz": float(np.median(cap["test"]["rates_hz"])),
        }
        out["arms"]["A2"] = _arm_summary(result, capture_seconds=secs, rates=rates)
        out["arms"]["A2"]["device"] = str(device2)
        out["arms"]["A2"]["cfg_seed"] = int(cfg2.seed)
        out["arms"]["A2"]["max_memory_allocated_mb"] = (
            float(torch.cuda.max_memory_allocated() / 2**20) if device2.type == "cuda" else 0.0
        )
        print(
            f"[A2] untrained dynamics + fresh readout: {result['test_correct']}/"
            f"{result['test_total']} = {result['test_accuracy'] * 100:.2f}% "
            f"(gap {gap_pp(result['test_correct'], result['test_total']):+.2f} pp)"
        )

    # ---- A3: trained weights, state dynamics zeroed --------------------
    if "a3" in arms:
        device3 = resolve_device(cfg)
        if str(args.device) not in ("", "auto", "None"):
            device3 = torch.device(args.device)
            if device3.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
        cfg3 = effective_config(cfg, device3)
        model = build_model(cfg3, device3, payload["state_dict"])
        with torch.no_grad():
            for name in ("w_rec", "w_emit", "w_down", "w_up"):
                getattr(model, name).zero_()
        splits = open_splits(cfg3)
        cap, secs = {}, {}
        for split in CAPTURE_SPLITS:
            store, sp = splits[split]
            cap[split] = capture_counts(model, cfg3, device3, store, sp, CAPTURE_PHASE[split],
                                        batch_size=int(args.batch_size))
            secs[split] = cap[split]["seconds"]
            np.savez_compressed(
                ANALYSIS_DIR / f"ablation_counts_a3{suffix}_{split}.npz",
                counts=cap[split]["counts"], labels=cap[split]["labels"],
                indices=cap[split]["indices"],
            )
            print(f"    {split}: {cap[split]['correct']}/{cap[split]['total']} with the "
                  f"(untrained-on-this-dynamics) w_cls, {secs[split]:.1f} s")
        result = fit_readout(
            cap["fit"]["counts"], cap["fit"]["labels"],
            cap["val"]["counts"], cap["val"]["labels"],
            cap["test"]["counts"], cap["test"]["labels"],
        )
        np.savez_compressed(ANALYSIS_DIR / f"ablation_readout_a3{suffix}.npz", w=result["w"])
        rates = {
            "fit_median_hz": float(np.median(cap["fit"]["rates_hz"])),
            "fit_min_hz": float(np.min(cap["fit"]["rates_hz"])),
            "fit_max_hz": float(np.max(cap["fit"]["rates_hz"])),
            "test_median_hz": float(np.median(cap["test"]["rates_hz"])),
        }
        out["arms"]["A3"] = _arm_summary(result, capture_seconds=secs, rates=rates)
        out["arms"]["A3"]["device"] = str(device3)
        out["arms"]["A3"]["max_memory_allocated_mb"] = (
            float(torch.cuda.max_memory_allocated() / 2**20) if device3.type == "cuda" else 0.0
        )
        print(
            f"[A3] input-drive-only dynamics + fresh readout: {result['test_correct']}/"
            f"{result['test_total']} = {result['test_accuracy'] * 100:.2f}% "
            f"(gap {gap_pp(result['test_correct'], result['test_total']):+.2f} pp)"
        )

    merged = _merge_arms(ANALYSIS_DIR / f"ablation_freeze{suffix}.json", out)
    with open(ANALYSIS_DIR / f"ablation_freeze{suffix}.json", "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2)
    print(f"\nwrote {ANALYSIS_DIR / 'ablation_freeze.json'}")
    return out


# ---------------------------------------------------------------------- #
# part B1 -- per-neuron class information
# ---------------------------------------------------------------------- #
def mi_bits(counts: np.ndarray, labels: np.ndarray, n_bins: int = MI_BINS) -> float:
    """Plug-in mutual information [bits] between a count vector and the class.

    The count axis is discretised into at most ``n_bins`` quantile bins (edges
    computed on the passed sample; duplicate edges dropped).  Plug-in MI is
    biased upward by ~``(K-1)(B-1)/(2 N ln 2)``; with N=7340, K=20, B=8 that is
    ~0.013 bits, so differences above ~0.02 bits are not bias artefacts.
    """
    x = np.asarray(counts, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.int64).ravel()
    if x.size != y.size:
        raise ValueError("counts and labels must have the same length")
    if x.size == 0:
        raise ValueError("empty input")
    classes, yy = np.unique(y, return_inverse=True)
    K = classes.size
    if np.all(x == x[0]):
        return 0.0
    if n_bins > 1:
        edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]))
        b = np.digitize(x, edges)
    else:
        b = np.zeros_like(x, dtype=np.int64)
    B = int(b.max()) + 1
    joint = np.zeros((B, K), dtype=np.float64)
    np.add.at(joint, (b.astype(np.int64), yy), 1.0)
    joint /= x.size
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    outer = px @ py
    mask = joint > 0
    return float(max(np.sum(joint[mask] * np.log2(joint[mask] / outer[mask])), 0.0))


def mi_per_neuron(counts: np.ndarray, labels: np.ndarray, n_bins: int = MI_BINS) -> np.ndarray:
    counts = np.asarray(counts)
    return np.asarray([mi_bits(counts[:, i], labels, n_bins) for i in range(counts.shape[1])],
                      dtype=np.float64)


def entropy_bits(labels: np.ndarray) -> float:
    p = np.bincount(np.asarray(labels, dtype=np.int64)).astype(np.float64)
    p = p[p > 0] / p.sum()
    return float(-np.sum(p * np.log2(p)))


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank)."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho via average ranks; 0.0 if either input is constant."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError("a and b must have the same length")
    if a.size < 2:
        return 0.0
    ra, rb = _rankdata(a), _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = math.sqrt(float(ra @ ra) * float(rb @ rb))
    return float(ra @ rb / denom) if denom > 0 else 0.0


def fingerprint_summary(
    mi: np.ndarray, rates_hz: np.ndarray, mi_test: np.ndarray | None = None, k: int = 5
) -> dict:
    mi = np.asarray(mi, dtype=np.float64)
    rates_hz = np.asarray(rates_hz, dtype=np.float64)
    order = np.argsort(-mi, kind="mergesort")
    total = float(mi.sum())
    shares = mi / total if total > 0 else np.zeros_like(mi)
    effective = float(1.0 / np.sum(shares**2)) if total > 0 else 0.0

    def _row(i: int) -> dict:
        row = {"neuron": int(i), "mi_bits": float(mi[i]), "rate_hz": float(rates_hz[i])}
        if mi_test is not None:
            row["mi_test_bits"] = float(mi_test[i])
        return row

    summary = {
        "n_neurons": int(mi.size),
        "mi_total_bits": total,
        "mi_mean_bits": float(mi.mean()),
        "mi_std_bits": float(mi.std()),
        "mi_min_bits": float(mi.min()),
        "mi_median_bits": float(np.median(mi)),
        "mi_max_bits": float(mi.max()),
        "mi_p25_bits": float(np.percentile(mi, 25)),
        "mi_p75_bits": float(np.percentile(mi, 75)),
        "mi_cv": float(mi.std() / mi.mean()) if mi.mean() > 0 else float("nan"),
        "top5_share": float(shares[order[:k]].sum()) if total > 0 else 0.0,
        "effective_informative_neurons": effective,
        "n_below_0.05_bits": int((mi < 0.05).sum()),
        "rate_hz_median": float(np.median(rates_hz)),
        "rate_hz_min": float(rates_hz.min()),
        "rate_hz_max": float(rates_hz.max()),
        "top": [_row(int(i)) for i in order[:k]],
        "bottom": [_row(int(i)) for i in order[-k:][::-1]],
        "rank_order": order.tolist(),
    }
    if mi_test is not None:
        summary["spearman_mi_fit_vs_test"] = spearman(mi, mi_test)
    return summary


def run_fingerprints(args: argparse.Namespace) -> dict:
    """B1: per-neuron mutual information and mean rate (CPU)."""
    cap_tag = str(getattr(args, "capture_tag", "") or "")
    cap_prefix = f"capture_{cap_tag}" if cap_tag else "capture"
    suffix = f"_{cap_tag}" if cap_tag else ""
    cfg, _ = load_payload(getattr(args, "checkpoint", VECTOR_CHECKPOINT))
    with np.load(ANALYSIS_DIR / f"{cap_prefix}_fit.npz") as d:
        counts_fit = d["counts"].astype(np.float32)
        labels_fit = d["labels"].astype(np.int64)
    with np.load(ANALYSIS_DIR / f"{cap_prefix}_test.npz") as d:
        counts_test = d["counts"].astype(np.float32)
        labels_test = d["labels"].astype(np.int64)

    tic = time.perf_counter()
    mi_fit = mi_per_neuron(counts_fit, labels_fit, n_bins=int(args.mi_bins))
    mi_test = mi_per_neuron(counts_test, labels_test, n_bins=int(args.mi_bins))
    rate_fit = rates_hz_from_counts(counts_fit, cfg)
    rate_test = rates_hz_from_counts(counts_test, cfg)
    seconds = time.perf_counter() - tic

    # split-half reliability of the FIT ranking (even vs odd FIT samples)
    ev = np.arange(0, counts_fit.shape[0], 2)
    od = np.arange(1, counts_fit.shape[0], 2)
    mi_a = mi_per_neuron(counts_fit[ev], labels_fit[ev], n_bins=int(args.mi_bins))
    mi_b = mi_per_neuron(counts_fit[od], labels_fit[od], n_bins=int(args.mi_bins))

    out = {
        "protocol": {
            "rank_split": "fit (n=%d)" % counts_fit.shape[0],
            "mi": "plug-in mutual information [bits], count quantile bins",
            "mi_bins": int(args.mi_bins),
            "mi_bias_note": "plug-in MI biased up by ~(K-1)(B-1)/(2 N ln2) ~ 0.013 bits",
            "class_entropy_bits": entropy_bits(labels_fit),
            "test_split": "official TEST, descriptive only (not used for the ranking)",
            "rate": "mean spike count per second = mean count / (T * bin_ms)",
            "seconds": seconds,
        },
        "mi_fit_vs_test_spearman": spearman(mi_fit, mi_test),
        "split_half_spearman": spearman(mi_a, mi_b),
        "fit": fingerprint_summary(mi_fit, rate_fit, mi_test=mi_test),
        "test": fingerprint_summary(mi_test, rate_test),
        "mi_bits_fit": mi_fit.tolist(),
        "mi_bits_test": mi_test.tolist(),
        "rate_hz_fit": rate_fit.tolist(),
        "rate_hz_test": rate_test.tolist(),
    }
    np.savez_compressed(
        ANALYSIS_DIR / f"ablation_fingerprints{suffix}.npz",
        mi_bits_fit=mi_fit, mi_bits_test=mi_test,
        rate_hz_fit=rate_fit, rate_hz_test=rate_test,
    )
    with open(ANALYSIS_DIR / f"ablation_fingerprints{suffix}.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    s = out["fit"]
    print(
        f"[B1] FIT n={counts_fit.shape[0]}: MI {s['mi_min_bits']:.3f}..."
        f"{s['mi_max_bits']:.3f} bits (median {s['mi_median_bits']:.3f}), "
        f"top-{args.mi_bins} share {s['top5_share']:.2f}, effective "
        f"{s['effective_informative_neurons']:.1f}/{s['n_neurons']} neurons"
    )
    for row in s["top"]:
        print(f"    top    neuron {row['neuron']:2d}: MI {row['mi_bits']:.3f} bits, "
              f"rate {row['rate_hz']:.1f} Hz, MI(test) {row['mi_test_bits']:.3f}")
    for row in s["bottom"]:
        print(f"    bottom neuron {row['neuron']:2d}: MI {row['mi_bits']:.3f} bits, "
              f"rate {row['rate_hz']:.1f} Hz, MI(test) {row['mi_test_bits']:.3f}")
    print(f"    split-half rho {out['split_half_spearman']:.3f}; "
          f"FIT-vs-TEST rho {out['mi_fit_vs_test_spearman']:.3f}")
    print(f"wrote {ANALYSIS_DIR / f'ablation_fingerprints{suffix}.json'} ({seconds:.1f} s CPU)")
    return out


# ---------------------------------------------------------------------- #
# part B2 -- knockout sweep
# ---------------------------------------------------------------------- #
def silence_neuron(theta: torch.Tensor, index: int, offset: float = SILENCE_OFFSET) -> torch.Tensor:
    """Return a copy of ``theta`` with neuron ``index`` driven below threshold.

    ``theta`` is *added* to the potential (``o_i = <z_i, w_out_i> + theta_i``),
    so a large **negative** offset is what silences.  The input tensor is not
    modified.
    """
    out = theta.detach().clone()
    out[index] = out[index] + float(offset)
    return out


def deletion_summary(drops_pp: np.ndarray, threshold_pp: float = DROP_THRESHOLD_PP) -> dict:
    """Summarise a per-neuron accuracy-drop vector (percentage points)."""
    d = np.asarray(drops_pp, dtype=np.float64).ravel()
    order = np.argsort(-d, kind="mergesort")
    return {
        "max_drop_pp": float(d.max()),
        "min_drop_pp": float(d.min()),
        "mean_drop_pp": float(d.mean()),
        "std_drop_pp": float(d.std()),
        "n_above_threshold": int((d > threshold_pp).sum()),
        "n_above_zero": int((d > 0).sum()),
        "n_improved": int((d < 0).sum()),
        "threshold_pp": float(threshold_pp),
        "sorted_drops_pp": d[order].tolist(),
        "sorted_neuron_order": order.tolist(),
        "most_important_neuron": int(order[0]),
    }


@torch.no_grad()
def test_correct(
    model: VectorNeuronPopulation, cfg: V3Config, device: torch.device,
    store: SHDEventStore, split, batch_size: int = 128,
) -> int:
    model.eval()
    model.set_phase("test")
    correct = 0
    for x_np, y_np in BatchIterator(store, split, batch_size, shuffle=False):
        logits, _ = model(_to_device(x_np, cfg, device))
        correct += int((logits.argmax(1).cpu().numpy() == y_np).sum())
    return correct


def run_knockout(args: argparse.Namespace) -> dict:
    """B2: silence each neuron in turn on official TEST."""
    path = Path(args.checkpoint)
    cfg, payload = load_payload(path)
    cap_tag = str(getattr(args, "capture_tag", "") or "")
    suffix = f"_{cap_tag}" if cap_tag else ""
    device = resolve_device(cfg)
    if str(args.device) not in ("", "auto", "None"):
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
    cfg = effective_config(cfg, device)
    model = build_model(cfg, device, payload["state_dict"])
    splits = open_splits(cfg)
    store, split = splits["test"]
    batch_size = int(args.batch_size)
    n_neurons = int(model.n_neurons)
    theta_orig = model.theta.detach().clone()

    tic = time.perf_counter()
    baseline = test_correct(model, cfg, device, store, split, batch_size)
    baseline_s = time.perf_counter() - tic
    print(f"[B2] intact model TEST: {baseline}/{len(split)} = "
          f"{100 * baseline / len(split):.2f}% ({baseline_s:.1f} s)")

    # runtime gate: the silenced neuron must emit zero spikes, and the other
    # neurons must see a changed input (so the lesion actually propagates).
    x0, _ = next(iter(BatchIterator(store, split, batch_size, shuffle=False)))
    base_counts = sample_counts(model, cfg, device, x0)
    model.theta.data.copy_(silence_neuron(theta_orig, 0))
    sil_counts = sample_counts(model, cfg, device, x0)
    model.theta.data.copy_(theta_orig)
    if int(sil_counts[:, 0].sum()) != 0:
        raise RuntimeError("knockout gate failed: silenced neuron still spikes")
    if np.array_equal(base_counts, sil_counts):
        raise RuntimeError("knockout gate failed: lesion does not propagate")
    print(f"[B2] knockout gate: neuron 0 emits {int(sil_counts[:, 0].sum())} spikes, "
          f"{int((base_counts != sil_counts).any(0).sum())} neurons' counts changed")

    per_eval: list[dict] = []
    total_s = 0.0
    for i in range(n_neurons):
        tic = time.perf_counter()
        model.theta.data.copy_(silence_neuron(theta_orig, i))
        correct = test_correct(model, cfg, device, store, split, batch_size)
        model.theta.data.copy_(theta_orig)
        dt = time.perf_counter() - tic
        total_s += dt
        per_eval.append(
            {
                "neuron": i,
                "correct": int(correct),
                "accuracy": correct / len(split),
                "drop_pp": 100.0 * (baseline - correct) / len(split),
                "seconds": dt,
            }
        )
        if i == 0:
            eta = dt * (n_neurons - 1)
            print(f"[B2] first knockout {dt:.1f} s -> ETA {eta / 60:.1f} min for the sweep")

    drops = np.asarray([r["drop_pp"] for r in per_eval], dtype=np.float64)
    summary = deletion_summary(drops)

    # cumulative deletion of the top-k neurons (k chosen on TEST -> disclosed)
    cumulative: list[dict] = []
    seen = set()
    for k in (1, 2, 4, 8, 16, 32, n_neurons):
        k = min(int(k), n_neurons)
        if k in seen:
            continue
        seen.add(k)
        keep = summary["sorted_neuron_order"][:k]
        if k == 1:
            correct = per_eval[keep[0]]["correct"]
            dt = per_eval[keep[0]]["seconds"]
            total_s += 0.0
        else:
            tic = time.perf_counter()
            theta = theta_orig.clone()
            for idx in keep:
                theta[idx] = theta[idx] + SILENCE_OFFSET
            model.theta.data.copy_(theta)
            correct = test_correct(model, cfg, device, store, split, batch_size)
            model.theta.data.copy_(theta_orig)
            dt = time.perf_counter() - tic
            total_s += dt
        cumulative.append(
            {
                "k": k,
                "select": "top-k by individual TEST drop (test-selected, disclosed)",
                "correct": int(correct),
                "accuracy": correct / len(split),
                "drop_pp": 100.0 * (baseline - correct) / len(split),
                "seconds": dt,
            }
        )
        print(f"[B2] cumulative top-{k:2d}: {correct}/{len(split)} = "
              f"{100 * correct / len(split):.2f}% (drop {cumulative[-1]['drop_pp']:.2f} pp)")

    # correlation with B1 (FIT-ranked MI)
    fp_path = ANALYSIS_DIR / f"ablation_fingerprints{suffix}.npz"
    correlation = {"available": False}
    if fp_path.exists():
        with np.load(fp_path) as d:
            mi_fit = d["mi_bits_fit"]
            rate_fit = d["rate_hz_fit"]
        correlation = {
            "available": True,
            "mi_source": "FIT mutual information (B1)",
            "spearman_mi_vs_drop": spearman(mi_fit, drops),
            "spearman_rate_vs_drop": spearman(rate_fit, drops),
            "note": "positive rho = higher class information predicts a bigger knockout drop",
        }
        print(f"[B2] Spearman(MI_FIT, drop) = {correlation['spearman_mi_vs_drop']:+.3f}; "
              f"Spearman(rate_FIT, drop) = {correlation['spearman_rate_vs_drop']:+.3f}")

    out = {
        "protocol": {
            "checkpoint": str(path),
            "split": "official TEST",
            "n": len(split),
            "baseline_correct": int(baseline),
            "baseline_accuracy": baseline / len(split),
            "silence_offset": SILENCE_OFFSET,
            "silence_note": (
                "theta is added to the potential, so silencing is theta_i = -1e9 "
                "(the brief's '+1e9' would pin the neuron above threshold); a silenced "
                "neuron also stops contributing to the population signal and the "
                "recurrent drive, because its spike is 0."
            ),
            "cumulative_selection": "top-k by individual TEST drop; test-selected, disclosed",
            "batch_size": batch_size,
            "device": str(device),
            "baseline_seconds": baseline_s,
        },
        "per_neuron": per_eval,
        "summary": summary,
        "cumulative": cumulative,
        "correlation": correlation,
        "seconds_per_eval_mean": float(np.mean([r["seconds"] for r in per_eval])),
        "gpu_seconds_total": total_s,
        "max_memory_allocated_mb": (
            float(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0
        ),
    }
    with open(ANALYSIS_DIR / "ablation_knockout.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print(
        f"[B2] drop curve: max {summary['max_drop_pp']:+.2f} pp (neuron "
        f"{summary['most_important_neuron']}), min {summary['min_drop_pp']:+.2f} pp, "
        f"{summary['n_above_threshold']}/{n_neurons} neurons individually > "
        f"{summary['threshold_pp']:.0f} pp, {summary['n_improved']} deletions improved TEST"
    )
    print(f"[B2] GPU {total_s / 60:.1f} min, {out['seconds_per_eval_mean']:.1f} s/eval, "
          f"peak {out['max_memory_allocated_mb']:.0f} MB")
    print(f"wrote {ANALYSIS_DIR / 'ablation_knockout.json'}")
    return out


# ---------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threads", type=int, default=8,
                   help="torch CPU threads (default 8: under the 60%% CPU cap)")
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("freeze", help="A1/A2/A3: fit only the readout on frozen dynamics")
    f.add_argument("--checkpoint", default=str(VECTOR_CHECKPOINT))
    f.add_argument("--capture-tag", default="",
                   help="use capture_<tag>_*.npz and write ablation_*_<tag> files")
    f.add_argument("--reference-correct", type=int, default=REFERENCE_CORRECT,
                   help="expected checkpoint readout on TEST (alignment gate)")
    f.add_argument("--arms", default="a1,a2,a3")
    f.add_argument("--device", default="auto")
    f.add_argument("--batch-size", type=int, default=128)
    f.set_defaults(func=run_freeze)

    fp = sub.add_parser("fingerprints", help="B1: per-neuron class information + rate")
    fp.add_argument("--mi-bins", type=int, default=MI_BINS)
    fp.add_argument("--capture-tag", default="",
                    help="use capture_<tag>_*.npz and write ablation_fingerprints_<tag>.json")
    fp.add_argument("--checkpoint", default=str(VECTOR_CHECKPOINT))
    fp.set_defaults(func=run_fingerprints)

    k = sub.add_parser("knockout", help="B2: silence each neuron in turn on TEST")
    k.add_argument("--checkpoint", default=str(VECTOR_CHECKPOINT))
    k.add_argument("--capture-tag", default="",
                   help="read ablation_fingerprints_<tag>.npz for the MI correlation")
    k.add_argument("--device", default="auto")
    k.add_argument("--batch-size", type=int, default=128)
    k.set_defaults(func=run_knockout)
    return p


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(int(args.threads))
    return args.func(args)


if __name__ == "__main__":
    main()
