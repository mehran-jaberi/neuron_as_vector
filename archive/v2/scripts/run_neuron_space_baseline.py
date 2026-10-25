"""The neuron-space baseline experiment (PRIMARY ANALYSIS) end to end.

For every replication seed this script:

1. loads the selected baseline checkpoint;
2. builds the **label-free** structural + activity representation on FIT;
3. measures the **canonical** class-conditioned fingerprints on the held-out PROBE
   split (raw 20-d rate profile, rate-normalized profile, temporal - secondary);
4. runs the PRIMARY ANALYSIS (Mantel Spearman + neuron-relabelling permutation
   test + bootstrap CI) for the structural representation and for every control
   (rate-only, activity-only, input-only, recurrent-only, intrinsic-only,
   structural, structural+activity, random, neuron-shuffle);
5. runs cross-validated prediction on **identical folds** for every representation;
6. repeats the whole analysis for the before-learning network and for the
   rewired-recurrent controls, preserving the recurrent weight distribution;
7. verifies structural and activity **permutation invariance** (mandatory);
8. audits the fingerprint's split-half reliability;
9. writes per-seed artifacts, then aggregates the primary metrics across seeds.

The official TEST set is never read. Nothing here selects a configuration.

Outputs (``results/neuron_space_baseline/``):

* ``seed_<s>/<...>.json|.npz`` - per-seed representations, fingerprints, distance
  matrices, geometry statistics, predictive statistics, control and condition rows;
* ``canonical_results_table.csv|.json`` - the ONE canonical table;
* ``summary.json`` - all seeds + the aggregated primary metrics;
* ``multiseed_summary.json`` - mean/SD/per-seed/95% CI;
* ``figure_bundle.json|.npz`` - everything the figures need.

Example::

    uv run python scripts/run_neuron_space_baseline.py --config configs/neuron_space_baseline.yaml
"""

from __future__ import annotations

import csv
import sys
from typing import Any

import numpy as np

from _common import (  # noqa: E402
    PROJECT_ROOT,
    build_recordings,
    ensure_dir,
    load_config,
    resolve_device,
)
from _pipeline import build_fingerprints, build_representation_bundle  # noqa: E402
from src.controls import reliability_suite  # noqa: E402
from src.evaluation import circuit_health, split_half_indices  # noqa: E402
from src.functional_fingerprint import FINGERPRINT_PRESETS, fingerprint_definition  # noqa: E402
from src.model import build_model  # noqa: E402
from src.neuron_space_baseline import (  # noqa: E402
    CANONICAL_TABLE_COLUMNS,
    NSBCondition,
    PRIMARY_REPRESENTATION,
    aggregate_primary_over_seeds,
    asked_questions_summary,
    before_after_nsb,
    build_canonical_table,
    nsb_variants,
    run_condition,
    run_cv_comparison,
)
from src.permutation import (  # noqa: E402
    check_activity_permutation_invariance,
    check_structural_permutation_invariance,
)
from src.rewiring import REWIRE_MODES, rewire_recurrent, rewiring_report  # noqa: E402
from src.utils import save_json  # noqa: E402

OUTPUT_DIR = "results/neuron_space_baseline"

# The canonical fingerprint presets measured on PROBE (one labelled pass).
PRESETS: dict[str, list[str]] = {
    "tuning": list(FINGERPRINT_PRESETS["tuning"]),
    "tuning_rate_normalized": list(FINGERPRINT_PRESETS["tuning_rate_normalized"]),
    "temporal": list(FINGERPRINT_PRESETS["temporal"]),
}


def _parse_args(argv: list[str] | None):
    import argparse

    p = argparse.ArgumentParser(description="Neuron-space baseline experiment (PRIMARY ANALYSIS).")
    p.add_argument("--config", default="configs/neuron_space_baseline.yaml")
    p.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--seeds", default="", help="comma-separated seeds (default: config multiseed.seeds)")
    p.add_argument("--n-perm", type=int, default=None, help="override analysis.n_perm")
    p.add_argument("--n-perm-conditions", type=int, default=None)
    p.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    p.add_argument("--smoke", action="store_true", help="tiny/fast settings for a code-path smoke test")
    p.add_argument(
        "--artifacts-only", action="store_true",
        help="only (re)write the per-seed representation/fingerprint/distance-matrix "
             "artifacts; does not touch the analysis results",
    )
    return p.parse_args(argv)


# --------------------------------------------------------------------------
# Condition construction
# --------------------------------------------------------------------------
def _fingerprints(model, eval_rec, eval_idx, *, device, n_classes, batch_size,
                  fp_standardize, fp_normalize_rows, fp_metric, n_psth_bins, min_spikes_for_latency):
    return build_fingerprints(
        model, eval_rec, eval_idx, device=device, n_classes=n_classes, batch_size=batch_size,
        presets=PRESETS, standardize=fp_standardize, normalize_rows=fp_normalize_rows,
        metric=fp_metric, n_psth_bins=n_psth_bins, min_spikes_for_latency=min_spikes_for_latency,
    )


def _build_condition(name, model, *, device, ref_rec, ref_idx, eval_rec, eval_idx, n_classes,
                     batch_size, weighting, normalize_rows, block_weights, include_tonotopic,
                     fp_standardize, fp_normalize_rows, fp_metric, n_psth_bins,
                     min_spikes_for_latency, metadata=None) -> NSBCondition:
    bundle = build_representation_bundle(
        model, ref_rec, ref_idx, device=device, n_classes=n_classes, batch_size=batch_size,
        weighting=weighting, normalize_rows=normalize_rows, block_weights=block_weights,
        primary_blocks=["intrinsic", "input_conn", "recurrent_in", "recurrent_out"],
        include_tonotopic_features=include_tonotopic,
    )
    fp = _fingerprints(
        model, eval_rec, eval_idx, device=device, n_classes=n_classes, batch_size=batch_size,
        fp_standardize=fp_standardize, fp_normalize_rows=fp_normalize_rows, fp_metric=fp_metric,
        n_psth_bins=n_psth_bins, min_spikes_for_latency=min_spikes_for_latency,
    )
    cond = NSBCondition(
        name=name, structural=bundle["structural"], activity=bundle["activity"],
        fingerprints=fp["fingerprints"], metadata=dict(metadata or {}),
    )
    return cond, bundle, fp


# --------------------------------------------------------------------------
# One seed
# --------------------------------------------------------------------------
def run_seed(
    seed: int,
    checkpoint: str,
    *,
    cfg,
    device,
    recs,
    out_dir,
    n_perm: int,
    n_perm_conditions: int,
    bootstrap: int,
    k_values,
    n_strata: int,
    control_repeats: int,
    prediction_cfg,
    include_before: bool,
    rewired_modes,
) -> dict[str, Any]:
    n_classes = int(cfg.get_path("train.n_classes", 20))
    batch_size = int(cfg.get_path("train.eval_batch_size", 256))
    weighting = str(cfg.get_path("representations.weighting", "equal"))
    normalize_rows = bool(cfg.get_path("representations.normalize_rows", False))
    block_weights = cfg.get_path("representations.block_weights", None)
    block_weights = {str(k): float(v) for k, v in block_weights.items()} if block_weights else None
    include_tonotopic = bool(cfg.get_path("representations.include_tonotopic_features", False))
    activity_split = str(cfg.get_path("representations.activity_split", "train"))
    eval_split = str(cfg.get_path("fingerprint.eval_split", "probe"))

    fp_standardize = str(cfg.get_path("fingerprint.standardize", "column"))
    fp_normalize_rows = bool(cfg.get_path("fingerprint.normalize_rows", False))
    fp_metric = str(cfg.get_path("fingerprint.metric", "euclidean"))
    n_psth_bins = int(cfg.get_path("fingerprint.n_psth_bins", 10))
    min_spikes_for_latency = float(cfg.get_path("fingerprint.min_spikes_for_latency", 1.0))

    ref_rec, eval_rec = recs[activity_split], recs[eval_split]
    ref_idx = np.arange(len(ref_rec))
    eval_idx = np.arange(len(eval_rec))

    path = PROJECT_ROOT / checkpoint
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    from src.model import RecurrentLIFSNN, architecture_mismatches

    model, extra = RecurrentLIFSNN.load(str(path), map_location="cpu")
    mismatch = architecture_mismatches(cfg.get_path("model", {}) or {}, model)
    if mismatch:
        raise ValueError(f"Config/checkpoint architecture mismatch for {path.name}: {mismatch}")
    model = model.to(device)

    common = dict(
        device=device, ref_rec=ref_rec, ref_idx=ref_idx, eval_rec=eval_rec, eval_idx=eval_idx,
        n_classes=n_classes, batch_size=batch_size, weighting=weighting, normalize_rows=normalize_rows,
        block_weights=block_weights, include_tonotopic=include_tonotopic,
        fp_standardize=fp_standardize, fp_normalize_rows=fp_normalize_rows, fp_metric=fp_metric,
        n_psth_bins=n_psth_bins, min_spikes_for_latency=min_spikes_for_latency,
    )
    print(
        f"[seed {seed}] checkpoint={checkpoint} | activity on '{activity_split}' (n={len(ref_rec)}) "
        f"| fingerprint on '{eval_split}' (n={len(eval_rec)})"
    )

    after_cond, after_bundle, after_fp = _build_condition(
        "after_learning", model, metadata={"role": "trained checkpoint"}, **common
    )

    def _cond_result(cond, n_perm_used, with_bootstrap):
        return run_condition(
            cond, tuning_key="tuning", tuning_normalized_key="tuning_rate_normalized",
            n_perm=n_perm_used, k_values=k_values, seed=seed,
            bootstrap=(bootstrap if with_bootstrap else 0), n_strata=n_strata,
            control_repeats=control_repeats, prediction=prediction_cfg, variants=nsb_variants(),
        )

    after = _cond_result(after_cond, n_perm, True)

    # ---- before learning (same architecture, same init seed, same data) ------
    before = None
    before_cond = None
    if include_before:
        untrained = build_model(model.cfg, seed=seed, device=device)
        before_cond, _before_bundle, _before_fp = _build_condition(
            "before_learning", untrained, metadata={"role": "untrained, same arch/init seed"}, **common
        )
        before = _cond_result(before_cond, n_perm_conditions, False)

    # ---- rewired recurrent controls -----------------------------------------
    rewired: dict[str, Any] = {}
    flattened: dict[str, Any] = {}
    for mode in rewired_modes:
        if mode not in REWIRE_MODES:
            raise ValueError(f"Unknown rewired mode {mode!r}; choose from {list(REWIRE_MODES)}")
        rew = rewire_recurrent(model, mode=mode, seed=seed)
        report = rewiring_report(model, rew, mode=mode)
        cond, _b, _f = _build_condition(
            f"rewired_{mode}", rew, metadata={"role": f"rewired recurrent ({mode})", **report}, **common
        )
        res = _cond_result(cond, n_perm_conditions, False)
        rewired[mode] = {"rows": res["rows"], "report": report}
        flattened[f"rewired_{mode}"] = {"rows": res["rows"], "report": report}
        print(
            f"[seed {seed}] rewired_{mode}: multiset={report['weight_multiset_preserved']} "
            f"row_multi={report['row_multisets_preserved']} col_multi={report['column_multisets_preserved']} "
            f"changed={report['fraction_positions_changed']:.2f}"
        )

    # ---- before vs after learning -------------------------------------------
    before_after = before_after_nsb(
        after["rows"], before["rows"] if before else [], representation=PRIMARY_REPRESENTATION
    )

    # ---- permutation invariance (mandatory) ---------------------------------
    perm_inv = check_structural_permutation_invariance(
        model, seed=seed, include_tonotopic=include_tonotopic
    )
    print(
        f"[seed {seed}] structural permutation invariance: passed={perm_inv['passed']} "
        f"sensitive={perm_inv['sensitive_features'][:5]}"
    )
    act_perm_inv = None
    try:
        n_probe = min(512, len(ref_rec))
        act_perm_inv = check_activity_permutation_invariance(
            model, ref_rec, np.arange(n_probe), seed=seed, device=device,
            n_classes=n_classes, batch_size=batch_size,
        )
        print(f"[seed {seed}] activity permutation invariance: passed={act_perm_inv['passed']}")
    except Exception as exc:  # pragma: no cover - diagnostic only
        act_perm_inv = {"passed": None, "error": f"{type(exc).__name__}: {exc}"}
        print(f"[seed {seed}] activity permutation invariance could not run: {exc}")

    # ---- fingerprint reliability (descriptive noise ceiling) ----------------
    labels = eval_rec.labels_array
    a_idx, b_idx = split_half_indices(labels, seed=seed)
    halves = {}
    for half_name, idx in (("a", a_idx), ("b", b_idx)):
        halves[half_name] = build_fingerprints(
            model, eval_rec, idx, device=device, n_classes=n_classes, batch_size=batch_size,
            presets={"tuning": PRESETS["tuning"]}, standardize=fp_standardize,
            normalize_rows=fp_normalize_rows, metric=fp_metric, n_psth_bins=n_psth_bins,
            min_spikes_for_latency=min_spikes_for_latency,
        )["fingerprints"]["tuning"]
    reliability = reliability_suite(
        after_cond.fingerprints["tuning"], halves["a"], halves["b"],
        labels=labels, idx_a=a_idx, idx_b=b_idx, split_name=eval_split,
    )
    print(
        f"[seed {seed}] fingerprint reliability: matrix_r={reliability['matrix_reliability_spearman']:.3f} "
        f"full_sb={reliability['matrix_reliability_full_spearman_brown']:.3f} "
        f"attenuation={reliability['attenuation_factor_sqrt_ceiling']:.3f}"
    )

    # ---- cross-validated prediction on IDENTICAL folds ----------------------
    rep_matrices = {name: sp.X for name, sp in after["spaces"].items()}
    predictive = run_cv_comparison(
        rep_matrices, after_cond.fingerprints["tuning"].X,
        n_splits=int(prediction_cfg.get("n_splits", 5)), seed=seed,
        alphas=tuple(prediction_cfg.get("alphas", (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0))),
        knn_k=int(prediction_cfg.get("knn_k", 5)),
    )

    # ---- model health -------------------------------------------------------
    health = circuit_health(model, ref_rec, ref_idx[: min(2000, len(ref_rec))],
                            device=device, batch_size=batch_size, n_classes=n_classes)
    rates = np.asarray(health["rate_hz_per_neuron"], dtype=np.float64)
    v_mean_arr = np.asarray(health["v_mean_per_neuron"], dtype=np.float64)

    # ---- persist per-seed artifacts ----------------------------------------
    seed_dir = ensure_dir(out_dir / f"seed_{seed}")
    structural_space = after["spaces"][PRIMARY_REPRESENTATION]
    tuning_cond = after["tuning_condensed"]
    tuning_norm_cond = after["tuning_normalized_condensed"]

    arrays = {
        "structural_X": structural_space.X,
        "structural_dx": structural_space.condensed(),
        "tuning_dy": tuning_cond,
        "tuning_rate_normalized_dy": tuning_norm_cond,
        "neuron_rate_hz": rates,
        "class_rate_matrix": after_fp["rate_matrix"],
        "rate_absdiff": after["rate_absdiff"],
        "structural_feature_names": np.array(structural_space.feature_names, dtype=object),
    }
    np.savez_compressed(str(seed_dir / f"seed_{seed}_arrays.npz"), **arrays)

    seed_summary = {
        "seed": int(seed),
        "checkpoint": checkpoint,
        "activity_split": activity_split,
        "fingerprint_eval_split": eval_split,
        "primary_representation": PRIMARY_REPRESENTATION,
        "primary_fingerprint": fingerprint_definition(
            after_cond.fingerprints["tuning"].config,
            dimension=len(after_cond.fingerprints["tuning"].feature_names),
        ),
        "representation_summary": structural_space.summary(),
        "representation_diagnostics": after_bundle["diagnostics"],
        "fingerprints": {n: {"dim": len(s.feature_names), "config": s.config.to_dict()}
                         for n, s in after_cond.fingerprints.items()},
        "after_learning": {"rows": after["rows"], "meta": {
            "n_perm": n_perm, "bootstrap": bootstrap, "k_values": list(k_values),
            "rate_matched_strata": n_strata, "control_repeats": control_repeats,
        }},
        "before_learning": {"rows": before["rows"]} if before else None,
        "rewired": rewired,
        **flattened,  # also expose rewired_<mode> keys for the canonical table
        "before_after": before_after,
        "permutation_invariance": perm_inv,
        "activity_permutation_invariance": act_perm_inv,
        "fingerprint_reliability": reliability,
        "predictive": predictive,
        "model_health": {
            "hidden_mean_rate_hz": float(rates.mean()) if rates.size else float("nan"),
            "hidden_median_rate_hz": float(np.median(rates)) if rates.size else float("nan"),
            "hidden_max_rate_hz": float(rates.max()) if rates.size else float("nan"),
            "frac_gt_200hz": float((rates > 200).mean()) if rates.size else float("nan"),
            "silent_fraction": float(health["silent_neuron_fraction"]),
            "rate_hz_percentiles": health["rate_hz_percentiles"],
            "v_mean_p1": float(np.percentile(v_mean_arr, 1)) if v_mean_arr.size else float("nan"),
            "v_mean_p50": float(np.percentile(v_mean_arr, 50)) if v_mean_arr.size else float("nan"),
            "v_mean_p99": float(np.percentile(v_mean_arr, 99)) if v_mean_arr.size else float("nan"),
            "v_global_mean": float(health["v_global_mean"]),
            "v_global_std": float(health["v_global_std"]),
            "v_global_min": float(health["v_global_min"]),
            "v_global_max": float(health["v_global_max"]),
        },
        "model_extra_keys": sorted(extra.keys()) if isinstance(extra, dict) else [],
    }
    save_json(seed_summary, seed_dir / f"seed_{seed}_summary.json")
    print(f"[seed {seed}] artifacts -> {seed_dir}")
    return seed_summary


# --------------------------------------------------------------------------
# Artifact export (representation / fingerprint / distance-matrix objects)
# --------------------------------------------------------------------------
def export_seed_artifacts(
    seed: int, checkpoint: str, *, cfg, device, recs, out_dir,
) -> dict[str, str]:
    """(Re)write the explicit per-seed objects the report references.

    These are the human/`numpy`-inspectable representations, fingerprint matrices
    and full (square) distance matrices. Kept separate from the analysis so they can
    be regenerated without recomputing any statistic.
    """
    from src.model import RecurrentLIFSNN, architecture_mismatches
    from src.utils import save_json

    n_classes = int(cfg.get_path("train.n_classes", 20))
    batch_size = int(cfg.get_path("train.eval_batch_size", 256))
    weighting = str(cfg.get_path("representations.weighting", "equal"))
    normalize_rows = bool(cfg.get_path("representations.normalize_rows", False))
    block_weights = cfg.get_path("representations.block_weights", None)
    block_weights = {str(k): float(v) for k, v in block_weights.items()} if block_weights else None
    include_tonotopic = bool(cfg.get_path("representations.include_tonotopic_features", False))
    activity_split = str(cfg.get_path("representations.activity_split", "train"))
    eval_split = str(cfg.get_path("fingerprint.eval_split", "probe"))

    path = PROJECT_ROOT / checkpoint
    model, _extra = RecurrentLIFSNN.load(str(path), map_location="cpu")
    mismatch = architecture_mismatches(cfg.get_path("model", {}) or {}, model)
    if mismatch:
        raise ValueError(f"Config/checkpoint architecture mismatch for {path.name}: {mismatch}")
    model = model.to(device)

    ref_rec, eval_rec = recs[activity_split], recs[eval_split]
    cond, bundle, fp = _build_condition(
        "after_learning", model, metadata={"role": "trained checkpoint"},
        device=device, ref_rec=ref_rec, ref_idx=np.arange(len(ref_rec)),
        eval_rec=eval_rec, eval_idx=np.arange(len(eval_rec)), n_classes=n_classes,
        batch_size=batch_size, weighting=weighting, normalize_rows=normalize_rows,
        block_weights=block_weights, include_tonotopic=include_tonotopic,
        fp_standardize=str(cfg.get_path("fingerprint.standardize", "column")),
        fp_normalize_rows=bool(cfg.get_path("fingerprint.normalize_rows", False)),
        fp_metric=str(cfg.get_path("fingerprint.metric", "euclidean")),
        n_psth_bins=int(cfg.get_path("fingerprint.n_psth_bins", 10)),
        min_spikes_for_latency=float(cfg.get_path("fingerprint.min_spikes_for_latency", 1.0)),
    )

    seed_dir = ensure_dir(out_dir / f"seed_{seed}")
    written: dict[str, str] = {}
    # 1. per-neuron representations (structural + label-free activity)
    rep_path = seed_dir / f"seed_{seed}_representations.json"
    bundle["structural"].save_json(str(rep_path))
    bundle["activity"].save_json(str(seed_dir / f"seed_{seed}_activity_representations.json"))
    written["representations"] = str(rep_path)
    # 2. fingerprint matrices (raw + standardized) per key
    from scipy.spatial.distance import squareform

    fp_payload: dict[str, np.ndarray] = {
        "class_rate_matrix": fp["rate_matrix"],
        "class_n": np.asarray(fp["result"].class_n, dtype=np.float64),
    }
    dist_payload: dict[str, np.ndarray] = {}
    structural_space = bundle["spaces"]["primary"]
    dist_payload["structural_D"] = structural_space.distances()
    dist_payload["structural_feature_names"] = np.array(structural_space.feature_names, dtype=object)
    for key, space in cond.fingerprints.items():
        fp_payload[f"{key}_X_raw"] = space.X_raw
        fp_payload[f"{key}_X"] = space.X
        fp_payload[f"{key}_feature_names"] = np.array(space.feature_names, dtype=object)
        d = space.condensed()
        dist_payload[f"{key}_D"] = squareform(d) if d.size else np.zeros((space.n_neurons, space.n_neurons))
    np.savez_compressed(str(seed_dir / f"seed_{seed}_fingerprints.npz"), **fp_payload)
    np.savez_compressed(str(seed_dir / f"seed_{seed}_distance_matrices.npz"), **dist_payload)
    written["fingerprints"] = str(seed_dir / f"seed_{seed}_fingerprints.npz")
    written["distance_matrices"] = str(seed_dir / f"seed_{seed}_distance_matrices.npz")
    save_json(
        {
            "seed": int(seed), "checkpoint": checkpoint,
            "activity_split": activity_split, "fingerprint_eval_split": eval_split,
            "representation": structural_space.summary(),
            "fingerprint_definitions": {
                k: {"config": s.config.to_dict(), "feature_names": list(s.feature_names)}
                for k, s in cond.fingerprints.items()
            },
            "fingerprints_npz": f"seed_{seed}_fingerprints.npz",
            "distance_matrices_npz": f"seed_{seed}_distance_matrices.npz",
            "note": "structural_D / <key>_D are full square Euclidean distance matrices.",
        },
        seed_dir / f"seed_{seed}_artifacts_index.json",
    )
    print(f"[artifacts] seed {seed} -> {seed_dir}")
    return written


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def _write_csv(rows: list[dict], path) -> None:
    if not rows:
        return
    fields = list(CANONICAL_TABLE_COLUMNS) + [k for k in rows[0] if k not in CANONICAL_TABLE_COLUMNS]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _figure_bundle(per_seed, *, primary_seed: int) -> dict[str, Any]:
    seed_res = per_seed[primary_seed]
    seed_dir = PROJECT_ROOT / OUTPUT_DIR / f"seed_{primary_seed}"
    with np.load(str(seed_dir / f"seed_{primary_seed}_arrays.npz"), allow_pickle=True) as data:
        arrays = {k: data[k] for k in data.files}
    after_rows = seed_res["after_learning"]["rows"]
    structural = next((r for r in after_rows if r.get("representation") == PRIMARY_REPRESENTATION), {})
    return {
        "primary_seed": int(primary_seed),
        "structural_X": arrays["structural_X"],
        "structural_dx": arrays["structural_dx"],
        "tuning_dy": arrays["tuning_dy"],
        "tuning_rate_normalized_dy": arrays["tuning_rate_normalized_dy"],
        "neuron_rate_hz": arrays["neuron_rate_hz"],
        "class_rate_matrix": arrays["class_rate_matrix"],
        "structural_feature_names": [str(n) for n in arrays["structural_feature_names"]],
        "after_rows": after_rows,
        "before_after": seed_res["before_after"],
        "rewired": {m: {"rows": v["rows"], "report": v["report"]} for m, v in seed_res["rewired"].items()},
        "structural_knn": structural.get("knn", []),
        "cv_representations": {k: {kk: vv for kk, vv in v.items() if kk not in ("ridge", "knn", "ridge_per_target")}
                               for k, v in seed_res["predictive"]["representations"].items()},
        "reliability": seed_res["fingerprint_reliability"],
        "multiseed_raw_r": {
            str(s): next((r.get("raw_r") for r in per_seed[s]["after_learning"]["rows"]
                          if r.get("representation") == PRIMARY_REPRESENTATION), None)
            for s in per_seed
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = load_config(args.config, overrides=args.override)
    if args.device:
        cfg.set_path("run.device", args.device)
    device = resolve_device(cfg)

    an = dict(cfg.get_path("analysis", {}) or {})
    n_perm = int(args.n_perm if args.n_perm is not None else an.get("n_perm", 10000))
    n_perm_conditions = int(
        args.n_perm_conditions if args.n_perm_conditions is not None
        else an.get("n_perm_conditions", 2000)
    )
    bootstrap = int(an.get("bootstrap", 2000))
    k_values = [int(k) for k in an.get("k_values", [3, 5, 10, 20])]
    n_strata = int(an.get("rate_matched_strata", 5))
    control_repeats = int(an.get("control_repeats", 5))
    prediction_cfg = dict(an.get("prediction", {"enabled": True}))

    cond_cfg = dict(cfg.get_path("conditions", {}) or {})
    include_before = bool(cond_cfg.get("include_before_learning", True))
    rewired_modes = [str(m) for m in cond_cfg.get("rewired_modes", ["global", "rowwise", "columnwise"])]

    seeds = (
        [int(s) for s in args.seeds.split(",") if s.strip()]
        if args.seeds
        else [int(s) for s in cfg.get_path("multiseed.seeds", [0, 1, 2])]
    )

    if args.smoke:
        n_perm = min(n_perm, 200)
        n_perm_conditions = min(n_perm_conditions, 100)
        bootstrap = min(bootstrap, 100)
        control_repeats = 1
        rewired_modes = rewired_modes[:1]

    out_dir = ensure_dir(PROJECT_ROOT / OUTPUT_DIR)
    print(
        f"[nsb] seeds={seeds} n_perm={n_perm} n_perm_conditions={n_perm_conditions} "
        f"bootstrap={bootstrap} device={device} | official TEST: NEVER READ"
    )

    # checkpoint manifest
    manifest_path = out_dir / "checkpoints.json"
    reuse_seed0 = (cfg.get_path("select_recipe.reuse_seed0_checkpoint", None)
                   if cfg.get_path("select_recipe", None) else None)
    checkpoints: dict[int, str] = {}
    manifest = None
    if manifest_path.exists():
        from src.utils import load_json

        manifest = load_json(manifest_path)
    for s in seeds:
        if manifest and str(s) in manifest.get("checkpoints", {}):
            checkpoints[s] = manifest["checkpoints"][str(s)]
        elif s == 0 and reuse_seed0:
            checkpoints[s] = str(reuse_seed0).replace("\\", "/")
        else:
            checkpoints[s] = f"checkpoints/nsb_seed{s}.pt"
    missing = [s for s, p in checkpoints.items() if not (PROJECT_ROOT / p).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing checkpoints for seeds {missing}: {[checkpoints[s] for s in missing]}. "
            f"Run `uv run python scripts/train_nsb_baseline.py --config {args.config}` first."
        )

    recs = build_recordings(cfg)

    if args.artifacts_only:
        for seed in seeds:
            export_seed_artifacts(seed, checkpoints[seed], cfg=cfg, device=device, recs=recs, out_dir=out_dir)
        print(f"[nsb] artifact export complete -> {out_dir}")
        return 0

    per_seed: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        per_seed[seed] = run_seed(
            seed, checkpoints[seed], cfg=cfg, device=device, recs=recs, out_dir=out_dir,
            n_perm=n_perm, n_perm_conditions=n_perm_conditions, bootstrap=bootstrap,
            k_values=k_values, n_strata=n_strata, control_repeats=control_repeats,
            prediction_cfg=prediction_cfg, include_before=include_before,
            rewired_modes=rewired_modes,
        )

    # ---- canonical table + aggregates --------------------------------------
    condition_names = ["after_learning"]
    if include_before:
        condition_names.append("before_learning")
    condition_names += [f"rewired_{m}" for m in rewired_modes]
    primary_fp_dim = int(per_seed[seeds[0]]["fingerprints"]["tuning"]["dim"])
    canonical = build_canonical_table(
        per_seed, checkpoints={s: checkpoints[s] for s in seeds},
        probe_split=str(cfg.get_path("fingerprint.eval_split", "probe")),
        functional_target_key="tuning", distance_metric="spearman",
        conditions=condition_names,
        functional_target_label=f"class_rate_{primary_fp_dim}d_probe",
    )
    _write_csv(canonical, out_dir / "canonical_results_table.csv")
    save_json(canonical, out_dir / "canonical_results_table.json")

    agg = aggregate_primary_over_seeds(per_seed, representation=PRIMARY_REPRESENTATION)
    save_json(agg, out_dir / "multiseed_summary.json")

    questions = asked_questions_summary(per_seed)
    summary = {
        "experiment": "neuron-space baseline (PRIMARY ANALYSIS)",
        "primary_representation": PRIMARY_REPRESENTATION,
        "primary_fingerprint_key": "tuning",
        "probe_split": str(cfg.get_path("fingerprint.eval_split", "probe")),
        "activity_split": str(cfg.get_path("representations.activity_split", "train")),
        "official_test_used": False,
        "notation": {
            "PRIMARY_ANALYSIS": (
                "Spearman Mantel correlation between the label-free structural neuron "
                "representation and the class-conditioned PROBE fingerprint, with a "
                "neuron-relabelling permutation test. Not called pre-registered."
            ),
            "pairs_not_independent": (
                "The permutation relabels neurons, not pairs; neuron pairs are never "
                "treated as independent observations."
            ),
            "rate_control_policy": (
                "Rate independence is argued from the rate-normalized fingerprint and the "
                "rate-matched stratified Mantel; the partial Mantel is secondary/exploratory."
            ),
            "no_cherry_picking": (
                "Every representation and control is reported, including negative and "
                "skipped results. The primary result is the structural representation."
            ),
        },
        "checkpoints": checkpoints,
        "seeds": seeds,
        "settings": {
            "n_perm": n_perm, "n_perm_conditions": n_perm_conditions, "bootstrap": bootstrap,
            "k_values": k_values, "rate_matched_strata": n_strata, "control_repeats": control_repeats,
            "rewired_modes": rewired_modes, "include_before_learning": include_before,
            "prediction": prediction_cfg,
        },
        "multiseed_primary": agg,
        "questions": questions,
        "per_seed": {str(s): per_seed[s] for s in seeds},
    }
    save_json(summary, out_dir / "summary.json")

    # ---- figure bundle ------------------------------------------------------
    primary_seed = 0 if 0 in per_seed else min(per_seed)
    bundle = _figure_bundle(per_seed, primary_seed=primary_seed)
    save_json({k: v for k, v in bundle.items() if k not in (
        "structural_X", "structural_dx", "tuning_dy", "tuning_rate_normalized_dy",
        "neuron_rate_hz", "class_rate_matrix")}, out_dir / "figure_bundle.json")
    np.savez_compressed(
        str(out_dir / "figure_bundle.npz"),
        structural_X=bundle["structural_X"], structural_dx=bundle["structural_dx"],
        tuning_dy=bundle["tuning_dy"], tuning_rate_normalized_dy=bundle["tuning_rate_normalized_dy"],
        neuron_rate_hz=bundle["neuron_rate_hz"], class_rate_matrix=bundle["class_rate_matrix"],
        structural_feature_names=np.array(bundle["structural_feature_names"], dtype=object),
    )

    # ---- figures ------------------------------------------------------------
    try:
        from src.nsb_figures import render_all

        written = render_all(out_dir, formats=[str(f) for f in cfg.get_path("figures.formats", ["png", "pdf"])],
                             dpi=int(cfg.get_path("figures.dpi", 200)))
        print(f"[nsb] figures written: {len(written)} files -> {out_dir / 'figures'}")
    except Exception as exc:  # pragma: no cover
        print(f"[nsb][warn] figure rendering failed: {type(exc).__name__}: {exc}")

    # ---- console headline ---------------------------------------------------
    p = next(r for r in per_seed[primary_seed]["after_learning"]["rows"]
             if r.get("representation") == PRIMARY_REPRESENTATION)
    r0 = next((r for r in per_seed[primary_seed]["after_learning"]["rows"]
               if r.get("representation") == "rate_only"), {})
    print("\n[PRIMARY ANALYSIS]")
    print(
        f"  structural (seed {primary_seed}): raw r={p['raw_r']:.3f} p={p['raw_p']:.4g} "
        f"z={p['raw_effect_size_z']:.2f} CI=[{p['raw_ci_low']:.3f},{p['raw_ci_high']:.3f}]"
    )
    print(
        f"  structural rate-normalized r={p['rate_normalized_r']:.3f} | "
        f"rate-only raw r={r0.get('raw_r'):.3f} | CV R2 structural={p['cv_r2_mean']:.3f} "
        f"rate-only={r0.get('cv_r2_mean'):.3f}"
    )
    m = agg["metrics"]
    if np.isfinite(m["raw_r"]["mean"]):
        print(
            f"\n[multiseed] structural raw r = {m['raw_r']['mean']:.3f} "
            f"± {m['raw_r']['sd']:.3f} (per-seed {m['raw_r']['per_seed']})"
        )
    print(f"[save] outputs -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))