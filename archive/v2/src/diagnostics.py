"""Numerical-health diagnostics for the LIF network and the analysis splits.

This module exists to answer, with measurements rather than assumptions:

* Are the LIF equations numerically healthy (membrane potential and synaptic
  current on the same scale as the threshold)?
* Is the firing-rate computation correct (spikes per timestep *and* Hz)?
* Is class collapse caused by preprocessing, imbalance, the loss, the dynamics or
  under-training?
* Which speakers are in FIT / DEV / PROBE / TEST, and what do the
  seen-speaker / held-out-speaker / official-test evaluation regimes mean?

Nothing here changes a model or a split; it only measures.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

PERCENTILES = (0, 1, 10, 25, 50, 75, 90, 99, 100)


def summarize(values: np.ndarray, *, name: str | None = None) -> dict[str, Any]:
    """Percentile summary of a 1-D array (finite values only)."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    q = np.percentile(v, PERCENTILES)
    out: dict[str, Any] = {"n": int(v.size), "mean": float(v.mean()), "std": float(v.std())}
    out.update({f"p{int(p)}": float(val) for p, val in zip(PERCENTILES, q)})
    if name:
        out["name"] = name
    return out


def histogram(values: np.ndarray, *, lo: float, hi: float, bins: int = 60) -> dict[str, Any]:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    v = v[np.isfinite(v)]
    counts, edges = np.histogram(v, bins=bins, range=(lo, hi))
    below = int((v < lo).sum())
    above = int((v > hi).sum())
    return {
        "edges": edges.tolist(),
        "counts": counts.tolist(),
        "n_below_range": below,
        "n_above_range": above,
    }


def measure_rates(spikes_per_step_per_neuron: np.ndarray, *, bin_ms: float) -> dict[str, Any]:
    """Report firing rate in **spikes/timestep** and in Hz, with percentiles.

    ``Hz = (spikes per timestep) / (bin_ms / 1000)``; for ``bin_ms = 2`` this is
    ``spikes_per_step * 500``. Reporting both makes it impossible for a unit error
    to hide behind a plausible-looking Hz number.
    """
    s = np.asarray(spikes_per_step_per_neuron, dtype=np.float64).reshape(-1)
    hz = s / (float(bin_ms) / 1000.0)
    return {
        "bin_ms": float(bin_ms),
        "spikes_per_timestep": summarize(s),
        "hz": summarize(hz),
        "mean_hz": float(hz.mean()) if hz.size else float("nan"),
        "median_hz": float(np.median(hz)) if hz.size else float("nan"),
        "p10_hz": float(np.percentile(hz, 10)) if hz.size else float("nan"),
        "p90_hz": float(np.percentile(hz, 90)) if hz.size else float("nan"),
        "max_hz": float(hz.max()) if hz.size else float("nan"),
        "silent_fraction_rate_eq_0": float((hz <= 0).mean()) if hz.size else float("nan"),
        "fraction_below_1hz": float((hz < 1.0).mean()) if hz.size else float("nan"),
        "fraction_below_5hz": float((hz < 5.0).mean()) if hz.size else float("nan"),
        "fraction_above_100hz": float((hz > 100.0).mean()) if hz.size else float("nan"),
        "fraction_above_200hz": float((hz > 200.0).mean()) if hz.size else float("nan"),
    }


# --------------------------------------------------------------------------
# Dynamics measurement
# --------------------------------------------------------------------------
def _subsample_tensor(t: Any, k: int, rng: np.random.Generator) -> np.ndarray:
    import torch

    flat = t.detach().reshape(-1).float().cpu().numpy()
    if flat.size > k:
        idx = rng.choice(flat.size, size=k, replace=False)
        flat = flat[idx]
    return flat


def measure_dynamics(
    model: Any,
    rec: Any,
    indices: Sequence[int] | np.ndarray,
    *,
    device: Any = None,
    batch_size: int = 32,
    n_samples: int = 64,
    pool_per_tensor: int = 200_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Measure membrane potential, synaptic current and its decomposition.

    Returns distributions (summary + histogram) for the membrane potential, the
    synaptic current ``i_syn``, the input current ``x @ W_in + b_hid`` and the
    recurrent current ``s_prev @ W_rec^T``, plus per-neuron means and the
    hidden spike probability per timestep.
    """
    import torch

    from .data import iterate_batches
    from .utils import get_device

    device = device or get_device()
    model = model.to(device)
    model.eval()
    sim = model.cfg
    rng = np.random.default_rng(seed)
    idx = np.asarray(indices, dtype=np.int64)[: int(n_samples)]

    w_in = model.w_in.detach().to(device)
    w_rec = model.w_rec.detach().to(device)
    b_hid = model.b_hid.detach().to(device) if getattr(model, "b_hid", None) is not None else None
    thr = float(model.effective_threshold().mean().item())

    pools: dict[str, list[np.ndarray]] = {k: [] for k in ("v", "i_syn", "i_in", "i_rec", "x")}
    per_neuron_v: list[np.ndarray] = []
    per_neuron_i: list[np.ndarray] = []
    spike_prob: list[np.ndarray] = []
    spikes_per_step_neuron: list[np.ndarray] = []
    t_axis_samples: list[np.ndarray] = []
    n_used = 0

    with torch.no_grad():
        for batch in iterate_batches(
            rec, idx, batch_size=batch_size, n_bins=sim.n_bins, bin_ms=sim.bin_ms, device=None
        ):
            x = batch["x"].to(device=device, dtype=torch.float32)
            out = model(x, record=True)
            v = out["hidden_v"]
            i_syn = out["hidden_i_syn"]
            s = out["hidden_spikes"]
            i_in = torch.einsum("btc,ch->bth", x, w_in)
            if b_hid is not None:
                i_in = i_in + b_hid
            s_prev = torch.zeros_like(s)
            if s.shape[1] > 1:
                s_prev[:, 1:, :] = s[:, :-1, :]
            i_rec = torch.einsum("btj,ij->bti", s_prev, w_rec)

            for key, tensor in (("v", v), ("i_syn", i_syn), ("i_in", i_in), ("i_rec", i_rec), ("x", x)):
                pools[key].append(_subsample_tensor(tensor, pool_per_tensor // 4, rng))
            per_neuron_v.append(v.mean(dim=(0, 1)).float().cpu().numpy())
            per_neuron_i.append(i_syn.mean(dim=(0, 1)).float().cpu().numpy())
            spike_prob.append(s.mean(dim=(0, 1)).float().cpu().numpy())
            spikes_per_step_neuron.append(s.mean(dim=0).mean(dim=0).float().cpu().numpy())
            # mean input drive as a function of time (pooled over batch/neurons)
            t_axis_samples.append(i_syn.mean(dim=(0, 2)).float().cpu().numpy())
            n_used += int(x.shape[0])

    pooled = {k: np.concatenate(v) if v else np.zeros(0) for k, v in pools.items()}
    pn_v = np.mean(np.stack(per_neuron_v, axis=0), axis=0) if per_neuron_v else np.zeros(0)
    pn_i = np.mean(np.stack(per_neuron_i, axis=0), axis=0) if per_neuron_i else np.zeros(0)
    sp_prob = np.mean(np.stack(spike_prob, axis=0), axis=0) if spike_prob else np.zeros(0)
    sp_step = np.mean(np.stack(spikes_per_step_neuron, axis=0), axis=0) if spikes_per_step_neuron else np.zeros(0)
    t_curve = np.mean(np.stack(t_axis_samples, axis=0), axis=0) if t_axis_samples else np.zeros(0)

    result: dict[str, Any] = {
        "n_samples_measured": int(n_used),
        "threshold": thr,
        "n_bins": int(sim.n_bins),
        "bin_ms": float(sim.bin_ms),
        "duration_ms": float(sim.duration_ms),
        "membrane_potential": {
            "summary": summarize(pooled["v"]),
            "histogram": histogram(pooled["v"], lo=-60.0, hi=60.0, bins=60),
            "per_neuron_mean": summarize(pn_v),
            "fraction_above_threshold": float((pooled["v"] > thr).mean()) if pooled["v"].size else float("nan"),
            "ratio_absmax_to_threshold": float(np.abs(pooled["v"]).max() / max(thr, 1e-9)) if pooled["v"].size else float("nan"),
        },
        "synaptic_current_i_syn": {
            "summary": summarize(pooled["i_syn"]),
            "histogram": histogram(pooled["i_syn"], lo=-60.0, hi=60.0, bins=60),
            "per_neuron_mean": summarize(pn_i),
        },
        "input_current": {
            "summary": summarize(pooled["i_in"]),
            "histogram": histogram(pooled["i_in"], lo=-20.0, hi=20.0, bins=60),
        },
        "recurrent_current": {
            "summary": summarize(pooled["i_rec"]),
            "histogram": histogram(pooled["i_rec"], lo=-20.0, hi=20.0, bins=60),
        },
        "input_events_per_bin": {
            "summary": summarize(pooled["x"]),
            "histogram": histogram(pooled["x"], lo=0.0, hi=20.0, bins=20),
            "fraction_multi_event_bins": float((pooled["x"] > 1.0).mean()) if pooled["x"].size else float("nan"),
        },
        "hidden_spike_probability_per_timestep": summarize(sp_prob),
        "rates": measure_rates(sp_step, bin_ms=float(sim.bin_ms)),
        "mean_i_syn_time_curve": t_curve.tolist(),
        "weights": {
            "w_in": {**summarize(w_in.detach().cpu().numpy().reshape(-1)), "shape": list(w_in.shape)},
            "w_rec": {**summarize(w_rec.detach().cpu().numpy().reshape(-1)), "shape": list(w_rec.shape)},
            "w_out": {**summarize(model.w_out.detach().cpu().numpy().reshape(-1)), "shape": list(model.w_out.shape)},
            "b_hid": summarize(b_hid.detach().cpu().numpy()) if b_hid is not None else None,
        },
        # subsampled raw pools for saving to NPZ (not part of the JSON summary)
        "_pools": {k: v.astype(np.float32) for k, v in pooled.items()},
        "_per_neuron": {
            "v_mean": pn_v.astype(np.float32),
            "i_syn_mean": pn_i.astype(np.float32),
            "spike_probability": sp_prob.astype(np.float32),
            "spikes_per_timestep": sp_step.astype(np.float32),
        },
    }
    return result


# --------------------------------------------------------------------------
# Class behaviour
# --------------------------------------------------------------------------
def class_input_stats(rec: Any, indices: Sequence[int] | np.ndarray, *, n_classes: int) -> dict[str, Any]:
    """Per-class sample counts, utterance span (ms) and event counts."""
    indices = np.asarray(indices, dtype=np.int64)
    labels = rec.labels_array[indices]
    spans = np.zeros(indices.size)
    events = np.zeros(indices.size)
    for k, i in enumerate(indices):
        lo, hi = int(rec.offsets[i]), int(rec.offsets[i + 1])
        events[k] = hi - lo
        if hi > lo:
            spans[k] = float(rec.times_ms[hi - 1] - rec.times_ms[lo])
    out: dict[str, Any] = {}
    for c in range(n_classes):
        sel = labels == c
        out[str(c)] = {
            "n_samples": int(sel.sum()),
            "mean_span_ms": float(spans[sel].mean()) if sel.any() else float("nan"),
            "mean_events": float(events[sel].mean()) if sel.any() else float("nan"),
            "mean_events_per_s": float((events[sel] / np.maximum(spans[sel] / 1000.0, 1e-9)).mean()) if sel.any() else float("nan"),
        }
    return out


def measure_class_behaviour(
    model: Any,
    rec: Any,
    indices: Sequence[int] | np.ndarray,
    *,
    device: Any = None,
    batch_size: int = 256,
    n_classes: int = 20,
) -> dict[str, Any]:
    """Per-class logits, loss, recall and the prediction distribution."""
    import torch
    import torch.nn.functional as F

    from .data import iterate_batches
    from .utils import get_device

    device = device or get_device()
    model = model.to(device)
    model.eval()
    sim = model.cfg
    indices = np.asarray(indices, dtype=np.int64)

    true_counts = np.zeros(n_classes, dtype=np.int64)
    pred_counts = np.zeros(n_classes, dtype=np.int64)
    correct = np.zeros(n_classes, dtype=np.int64)
    loss_sum = np.zeros(n_classes, dtype=np.float64)
    logit_sum_true = np.zeros(n_classes, dtype=np.float64)
    logit_sum_pred = np.zeros(n_classes, dtype=np.float64)
    logit_sum_all = np.zeros(n_classes, dtype=np.float64)
    max_logit_sum = 0.0
    margin_sum = 0.0
    n = 0

    with torch.no_grad():
        for batch in iterate_batches(
            rec, indices, batch_size=batch_size, n_bins=sim.n_bins, bin_ms=sim.bin_ms, device=None
        ):
            b = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            out = model(b["x"])
            logits = out["logits"]
            y = b["y"]
            per_sample = F.cross_entropy(logits, y, reduction="none").cpu().numpy()
            pred = logits.argmax(dim=1)
            lg = logits.cpu().numpy()
            y_np = y.cpu().numpy()
            p_np = pred.cpu().numpy()
            for c in range(n_classes):
                sel = y_np == c
                true_counts[c] += int(sel.sum())
                if sel.any():
                    loss_sum[c] += float(per_sample[sel].sum())
                    logit_sum_true[c] += float(lg[sel, c].sum())
            for c in range(n_classes):
                pc = p_np == c
                pred_counts[c] += int(pc.sum())
                if pc.any():
                    logit_sum_pred[c] += float(lg[pc, c].sum())
            logit_sum_all += lg.sum(axis=0)
            correct += np.bincount(p_np[y_np == p_np], minlength=n_classes)[:n_classes]
            max_logit_sum += float(lg.max(axis=1).sum())
            top2 = np.sort(lg, axis=1)[:, -2:]
            margin_sum += float((top2[:, 1] - top2[:, 0]).sum())
            n += int(y_np.size)

    per_class = {}
    for c in range(n_classes):
        per_class[str(c)] = {
            "n_true": int(true_counts[c]),
            "n_predicted": int(pred_counts[c]),
            "recall": float(correct[c] / true_counts[c]) if true_counts[c] else float("nan"),
            "mean_loss": float(loss_sum[c] / true_counts[c]) if true_counts[c] else float("nan"),
            "mean_logit_true_class": float(logit_sum_true[c] / true_counts[c]) if true_counts[c] else float("nan"),
            "mean_logit_when_predicted": float(logit_sum_pred[c] / pred_counts[c]) if pred_counts[c] else float("nan"),
        }
    return {
        "n_samples": int(n),
        "accuracy": float(correct.sum() / n) if n else float("nan"),
        "prediction_distribution": pred_counts.tolist(),
        "prediction_fraction": (pred_counts / n).tolist() if n else [],
        "mean_logit_per_class": (logit_sum_all / n).tolist() if n else [],
        "mean_max_logit": float(max_logit_sum / n) if n else float("nan"),
        "mean_top1_minus_top2_margin": float(margin_sum / n) if n else float("nan"),
        "per_class": per_class,
    }


# --------------------------------------------------------------------------
# Split semantics
# --------------------------------------------------------------------------
def build_speaker_table(
    train_rec: Any,
    test_rec: Any,
    split_info: Mapping[str, Any],
) -> dict[str, Any]:
    """Table of which speakers occur in FIT / DEV / PROBE / TEST (and how often).

    ``fit``/``dev``/``probe`` come from the recorded split metadata; ``test`` is the
    official SHD test file, which is never modified.
    """
    splits = dict(split_info.get("splits", {}) or {})
    fit_speakers = [int(s) for s in splits.get("train", {}).get("speakers", [])]
    dev_speakers = [int(s) for s in splits.get("dev", {}).get("speakers", [])]
    probe_speakers = [int(s) for s in splits.get("probe", {}).get("speakers", [])]
    train_file_speakers = sorted(int(s) for s in np.unique(np.asarray(train_rec.speakers)).tolist())

    test_spk = np.asarray(test_rec.speakers)
    test_unique, test_counts = np.unique(test_spk, return_counts=True)
    test_map = {int(s): int(c) for s, c in zip(test_unique, test_counts)}

    def _role(speaker: int) -> list[str]:
        roles = []
        if speaker in fit_speakers:
            roles.append("fit")
        if speaker in dev_speakers:
            roles.append("dev")
        if speaker in probe_speakers:
            roles.append("probe")
        if speaker not in train_file_speakers:
            roles.append("novel_not_in_train_file")
        return roles

    rows = []
    for s in sorted(set(train_file_speakers) | set(test_map)):
        rows.append({
            "speaker": int(s),
            "in_train_file": bool(s in train_file_speakers),
            "roles": _role(int(s)),
            "n_in_train_file": int((np.asarray(train_rec.speakers) == s).sum())
            if s in train_file_speakers else 0,
            "n_in_test": int(test_map.get(int(s), 0)),
        })
    return {
        "fit_speakers": fit_speakers,
        "dev_speakers": dev_speakers,
        "probe_speakers": probe_speakers,
        "train_file_speakers": train_file_speakers,
        "test_speakers_present": sorted(int(s) for s in test_unique),
        "rows": rows,
        "note": (
            "Official test set is a MIXTURE: speakers 4,5 are not in the training file "
            "at all; speakers 2,6,8 are held out of the FIT subset; the remaining test "
            "samples come from FIT speakers. It is never modified."
        ),
    }


def evaluation_regimes(test_rec: Any, fit_speakers: Sequence[int]) -> dict[str, Any]:
    """Masks defining the three evaluation regimes A/B/C on the official test set."""
    speakers = np.asarray(test_rec.speakers)
    fit = np.asarray(list(fit_speakers), dtype=np.int64)
    seen = np.isin(speakers, fit)
    held_out = ~seen
    return {
        "A_seen_speaker": {
            "description": "official test samples whose speaker was used in FIT",
            "mask": seen,
            "n_samples": int(seen.sum()),
        },
        "B_held_out_speaker": {
            "description": "official test samples whose speaker was held out of FIT",
            "mask": held_out,
            "n_samples": int(held_out.sum()),
        },
        "C_official_test": {
            "description": "the complete official SHD test file (locked)",
            "mask": np.ones(speakers.size, dtype=bool),
            "n_samples": int(speakers.size),
        },
    }