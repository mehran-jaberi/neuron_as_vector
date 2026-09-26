"""Numerical-health diagnostics: LIF dynamics, firing rates, class collapse, splits.

Read-only with respect to the data and the split: it loads a checkpoint, measures
the network's internal quantities, and writes a report. Never selects a model.

Outputs (``results/``):

* ``<tag>_diagnostics.json`` - all measurements (dynamics, rates, class behaviour,
  speaker table, evaluation regimes);
* ``<tag>_diagnostics.npz``  - subsampled raw distributions and per-neuron stats.

Example::

    uv run python scripts/diagnose.py --config configs/analysis.yaml
"""

from __future__ import annotations

import numpy as np

from _common import (  # noqa: E402
    announce,
    apply_seed,
    build_recordings,
    load_checkpoint,
    load_run_config,
    parse_args,
    resolve_device,
    result_path,
)
from src.diagnostics import (  # noqa: E402
    build_speaker_table,
    class_input_stats,
    evaluation_regimes,
    measure_class_behaviour,
    measure_dynamics,
)
from src.evaluation import evaluate_model  # noqa: E402
from src.utils import save_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = parse_args("Numerical-health diagnostics for the LIF network.", "configs/analysis.yaml")
    cfg = load_run_config(args)
    device = resolve_device(cfg)
    announce(cfg, device)
    apply_seed(cfg, device)

    tag = str(cfg.get_path("run.tag", "baseline_v2"))
    batch_size = int(cfg.get_path("train.eval_batch_size", 256))
    n_classes = int(cfg.get_path("train.n_classes", 20))
    n_dyn_samples = int(cfg.get_path("diagnostics.n_samples", 64))

    model, extra = load_checkpoint(cfg, device, tag)
    recs = build_recordings(cfg)

    report: dict = {"tag": tag, "dataset": recs["name"], "seed": int(cfg.get_path("seed", 0))}

    # ---- splits: speakers and roles ------------------------------------
    report["split_info"] = recs["split_info"]
    speaker_table = build_speaker_table(recs["train"], recs["test"], recs["split_info"])
    report["speaker_table"] = speaker_table
    fit_speakers = speaker_table["fit_speakers"]

    # ---- accuracy per split + regimes A/B/C ----------------------------
    per_split: dict = {}
    for name in ("train", "dev", "probe", "test"):
        rec = recs[name]
        if len(rec) == 0:
            continue
        m = evaluate_model(model, rec, np.arange(len(rec)), device=device, batch_size=batch_size,
                           n_classes=n_classes, compute_confusion=True)
        per_split[name] = {
            "n_samples": m["n_samples"], "accuracy": m["accuracy"], "loss": m["loss"],
            "hidden_mean_rate_hz": m["hidden_mean_rate_hz"],
            "hidden_silent_fraction": m["hidden_silent_fraction"],
            "per_class_accuracy": m.get("per_class_accuracy"),
            "confusion_matrix": m.get("confusion_matrix"),
        }
        print(f"[split] {name:5s} n={m['n_samples']:5d} acc={m['accuracy']:.4f} "
              f"hidden_rate={m['hidden_mean_rate_hz']:.1f} Hz")
    report["per_split"] = per_split

    regimes = evaluation_regimes(recs["test"], fit_speakers)
    regime_acc: dict = {}
    for key, spec in regimes.items():
        mask = spec["mask"]
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        m = evaluate_model(model, recs["test"], idx, device=device, batch_size=batch_size,
                           n_classes=n_classes, compute_confusion=True)
        regime_acc[key] = {
            "description": spec["description"], "n_samples": m["n_samples"],
            "accuracy": m["accuracy"], "loss": m["loss"],
            "per_class_accuracy": m.get("per_class_accuracy"),
        }
        print(f"[regime] {key:22s} n={m['n_samples']:5d} acc={m['accuracy']:.4f}")
    report["evaluation_regimes"] = regime_acc

    # ---- dynamics -------------------------------------------------------
    fit_idx = np.arange(len(recs["train"]))
    dyn = measure_dynamics(model, recs["train"], fit_idx, device=device,
                           batch_size=min(32, batch_size), n_samples=n_dyn_samples, seed=int(cfg.get_path("seed", 0)))
    arrays = {f"pool_{k}": v for k, v in dyn.pop("_pools").items()}
    pn = dyn.pop("_per_neuron")
    arrays.update({f"per_neuron_{k}": v for k, v in pn.items()})
    report["dynamics"] = dyn

    v = dyn["membrane_potential"]["summary"]
    i = dyn["synaptic_current_i_syn"]["summary"]
    ii = dyn["input_current"]["summary"]
    ir = dyn["recurrent_current"]["summary"]
    print(f"[dynamics] V      mean={v['mean']:.3f} p1={v['p1']:.2f} p50={v['p50']:.3f} p99={v['p99']:.2f} "
          f"max={v['p100']:.2f} |V|/thr={dyn['membrane_potential']['ratio_absmax_to_threshold']:.1f}")
    print(f"[dynamics] i_syn  mean={i['mean']:.3f} p1={i['p1']:.2f} p99={i['p99']:.2f}")
    print(f"[dynamics] i_in   mean={ii['mean']:.3f} p1={ii['p1']:.2f} p99={ii['p99']:.2f}")
    print(f"[dynamics] i_rec  mean={ir['mean']:.3f} p1={ir['p1']:.2f} p99={ir['p99']:.2f}")
    rt = dyn["rates"]
    print(f"[rates] spikes/step mean={rt['spikes_per_timestep']['mean']:.4f} "
          f"p90={rt['spikes_per_timestep']['p90']:.4f} max={rt['spikes_per_timestep']['p100']:.4f}")
    print(f"[rates] Hz mean={rt['mean_hz']:.1f} median={rt['median_hz']:.1f} max={rt['max_hz']:.1f} "
          f"<1Hz={rt['fraction_below_1hz']:.3f} >200Hz={rt['fraction_above_200hz']:.3f}")

    # ---- class behaviour ------------------------------------------------
    report["class_input_stats"] = {
        "fit": class_input_stats(recs["train"], fit_idx, n_classes=n_classes),
        "test": class_input_stats(recs["test"], np.arange(len(recs["test"])), n_classes=n_classes),
    }
    report["class_behaviour"] = {}
    for name in ("train", "probe", "test"):
        rec = recs[name]
        if len(rec) == 0:
            continue
        cb = measure_class_behaviour(model, rec, np.arange(len(rec)), device=device,
                                     batch_size=batch_size, n_classes=n_classes)
        report["class_behaviour"][name] = cb
        worst = sorted(((v["recall"], int(c)) for c, v in cb["per_class"].items() if np.isfinite(v["recall"])))[:5]
        print(f"[class] {name:5s} acc={cb['accuracy']:.4f} margin={cb['mean_top1_minus_top2_margin']:.3f} "
              f"worst recall: {[(c, round(r, 3)) for r, c in worst]}")

    report["provenance"] = {
        "model_extra_keys": sorted(extra.keys()) if isinstance(extra, dict) else [],
        "n_hidden": int(model.cfg.n_hidden),
        "n_bins": int(model.cfg.n_bins),
        "bin_ms": float(model.cfg.bin_ms),
        "threshold": float(model.cfg.threshold),
        "tau_mem_ms": float(model.cfg.tau_mem_ms),
        "tau_syn_ms": float(model.cfg.tau_syn_ms),
        "input_weight_scale": float(model.cfg.input_weight_scale),
        "recurrent_weight_scale": float(model.cfg.recurrent_weight_scale),
        "neuron_param_mode": str(model.cfg.neuron_param_mode),
    }

    save_json(report, result_path(cfg, f"{tag}_diagnostics.json"))
    np.savez_compressed(str(result_path(cfg, f"{tag}_diagnostics.npz")), **arrays)
    print(f"[save] -> {result_path(cfg, f'{tag}_diagnostics.json')}")
    print(f"[save] -> {result_path(cfg, f'{tag}_diagnostics.npz')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())