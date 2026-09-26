"""Stress-test: is the neuron-space result real, or a trivial property of the network?

Runs the fixed set of representations/controls (see ``src/stress_test.py``) through
the identical pipeline for the **after-learning**, **before-learning** and
**rewired recurrent-network** conditions, on the same architecture and the same
held-out analysis-probe data. For every row it reports the raw geometry-function
association, the rate-normalized association, the rate-matched association, a
cross-validated predictive metric and the permutation significance with its floor.

Outputs (``results/``):

* ``<tag>_stress_test.json`` - the single machine-readable table + all details +
  the before/after comparison + the fingerprint reliability audit;
* ``<tag>_stress_test.csv``  - the compact table;
* ``<tag>_stress_test.npz``  - permutation null distributions + distance vectors.
"""

from __future__ import annotations

import csv

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
from _pipeline import build_fingerprints, build_representation_bundle  # noqa: E402
from src.controls import reliability_suite  # noqa: E402
from src.evaluation import split_half_indices  # noqa: E402
from src.functional_fingerprint import (  # noqa: E402
    FINGERPRINT_PRESETS,
    PRIMARY_FINGERPRINT_PRESET,
    fingerprint_definition,
)
from src.model import build_model  # noqa: E402
from src.rewiring import REWIRE_MODES, rewire_recurrent, rewiring_report  # noqa: E402
from src.stress_test import (  # noqa: E402
    StressCondition,
    fingerprint_split_leakage_guard,
    run_stress_test,
    stress_variants,
)
from src.utils import save_json  # noqa: E402


def _write_csv(rows: list[dict], path) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _build_condition(
    name: str,
    model,
    *,
    cfg,
    device,
    ref_rec,
    ref_idx,
    eval_rec,
    eval_idx,
    n_classes: int,
    batch_size: int,
    weighting: str,
    normalize_rows: bool,
    block_weights,
    include_tonotopic: bool,
    fp_standardize: str,
    fp_normalize_rows: bool,
    fp_metric: str,
    n_psth_bins: int,
    min_spikes_for_latency: float,
    metadata: dict | None = None,
) -> StressCondition:
    bundle = build_representation_bundle(
        model, ref_rec, ref_idx,
        device=device, n_classes=n_classes, batch_size=batch_size,
        weighting=weighting, normalize_rows=normalize_rows, block_weights=block_weights,
        primary_blocks=["intrinsic", "input_conn", "recurrent_in", "recurrent_out"],
        include_tonotopic_features=include_tonotopic,
    )
    fp = build_fingerprints(
        model, eval_rec, eval_idx,
        device=device, n_classes=n_classes, batch_size=batch_size,
        presets={
            PRIMARY_FINGERPRINT_PRESET: list(FINGERPRINT_PRESETS[PRIMARY_FINGERPRINT_PRESET]),
            "tuning_rate_normalized": list(FINGERPRINT_PRESETS["tuning_rate_normalized"]),
        },
        standardize=fp_standardize, normalize_rows=fp_normalize_rows, metric=fp_metric,
        n_psth_bins=n_psth_bins, min_spikes_for_latency=min_spikes_for_latency,
    )
    return StressCondition(
        name=name,
        structural=bundle["structural"],
        activity=bundle["activity"],
        fingerprints=fp["fingerprints"],
        metadata=dict(metadata or {}),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(
        "Stress-test the neuron-space result against trivial explanations.",
        "configs/analysis.yaml",
    )
    cfg = load_run_config(args)
    device = resolve_device(cfg)
    announce(cfg, device)
    apply_seed(cfg, device)

    tag = str(cfg.get_path("run.tag", "baseline"))
    seed = int(cfg.get_path("seed", 0))
    n_classes = int(cfg.get_path("train.n_classes", 20))
    batch_size = int(cfg.get_path("train.eval_batch_size", 256))

    weighting = str(cfg.get_path("representations.weighting", "equal"))
    normalize_rows = bool(cfg.get_path("representations.normalize_rows", False))
    block_weights = cfg.get_path("representations.block_weights", None)
    block_weights = {str(k): float(v) for k, v in block_weights.items()} if block_weights else None
    include_tonotopic = bool(cfg.get_path("representations.include_tonotopic_features", False))

    fp_standardize = str(cfg.get_path("fingerprint.standardize", "column"))
    fp_normalize_rows = bool(cfg.get_path("fingerprint.normalize_rows", False))
    fp_metric = str(cfg.get_path("fingerprint.metric", "euclidean"))
    n_psth_bins = int(cfg.get_path("function_analysis.n_psth_bins", 10))
    min_spikes_for_latency = float(cfg.get_path("fingerprint.min_spikes_for_latency", 1.0))
    eval_split = str(cfg.get_path("fingerprint.eval_split", "probe"))
    activity_split = str(cfg.get_path("representations.activity_split", "train"))

    st = cfg.get_path("stress_test", {}) or {}
    st = dict(st)
    n_perm = int(st.get("n_perm", cfg.get_path("function_analysis.n_perm", 1000)))
    k_values = [int(k) for k in st.get("k_values", [3, 5, 10, 20])]
    rewired_modes = [str(m) for m in st.get("rewired_modes", list(REWIRE_MODES))]
    include_before = bool(st.get("include_before_learning", True))
    prediction_cfg = dict(st.get("prediction", {"enabled": True}))
    run_reliability_audit = bool(st.get("reliability_audit", True))

    # ---- model + data ---------------------------------------------------
    model, extra = load_checkpoint(cfg, device, tag)
    recs = build_recordings(cfg)
    if eval_split not in recs or activity_split not in recs:
        raise ValueError(f"Need splits; have {sorted(recs)}")
    if eval_split == "test":
        print(
            "[warn] the fingerprint is being measured on the OFFICIAL TEST set - allowed only "
            "as an evaluation target, never as a model input."
        )
    ref_rec, ref_idx = recs[activity_split], np.arange(len(recs[activity_split]))
    eval_rec, eval_idx = recs[eval_split], np.arange(len(recs[eval_split]))
    print(f"[data] activity on '{activity_split}' | fingerprint on '{eval_split}' n={len(eval_rec)}")

    # Leakage guard: the fingerprint must not be measured on the model-selection split.
    leakage_guard = fingerprint_split_leakage_guard(extra, recs.get("split_info"), eval_split)
    print(f"[leakage] {leakage_guard['note']}")
    if leakage_guard["passed"] is False:
        print(
            "[leakage][WARN] the reported association may be inflated by model-selection "
            "leakage; prefer the corrected 3-way-split checkpoint (e.g. tag=baseline_v2)."
        )

    common = dict(
        cfg=cfg, device=device, ref_rec=ref_rec, ref_idx=ref_idx, eval_rec=eval_rec, eval_idx=eval_idx,
        n_classes=n_classes, batch_size=batch_size, weighting=weighting, normalize_rows=normalize_rows,
        block_weights=block_weights, include_tonotopic=include_tonotopic,
        fp_standardize=fp_standardize, fp_normalize_rows=fp_normalize_rows, fp_metric=fp_metric,
        n_psth_bins=n_psth_bins, min_spikes_for_latency=min_spikes_for_latency,
    )

    conditions: list[StressCondition] = []

    # 1. after learning (the trained checkpoint)
    after = _build_condition("after_learning", model, metadata={"role": "trained checkpoint"}, **common)
    conditions.append(after)
    print(f"[condition] after_learning: trained checkpoint '{tag}'")

    # 2. before learning (SAME architecture, SAME initialisation seed, SAME probe)
    if include_before:
        untrained = build_model(model.cfg, seed=seed, device=device)
        before = _build_condition(
            "before_learning", untrained,
            metadata={"role": "untrained model, same architecture and init seed"}, **common,
        )
        conditions.append(before)
        print("[condition] before_learning: untrained, same architecture/init seed")

    # 3. rewired recurrent-network controls (from the trained checkpoint)
    rewire_reports: dict[str, dict] = {}
    for mode in rewired_modes:
        if mode not in REWIRE_MODES:
            raise ValueError(f"Unknown rewired mode {mode!r}; choose from {list(REWIRE_MODES)}")
        rewired = rewire_recurrent(model, mode=mode, seed=seed)
        report = rewiring_report(model, rewired, mode=mode)
        rewire_reports[mode] = report
        cond = _build_condition(
            f"rewired_{mode}", rewired, metadata={"role": f"rewired recurrent ({mode})", **report}, **common
        )
        conditions.append(cond)
        print(
            f"[condition] rewired_{mode}: multiset={report['weight_multiset_preserved']} "
            f"row_multi={report['row_multisets_preserved']} col_multi={report['column_multisets_preserved']} "
            f"changed={report['fraction_positions_changed']:.2f}"
        )

    # ---- run the stress test -------------------------------------------
    result = run_stress_test(
        conditions,
        variants=stress_variants(),
        tuning_key=PRIMARY_FINGERPRINT_PRESET,
        tuning_normalized_key="tuning_rate_normalized",
        n_perm=n_perm, k_values=k_values, seed=seed,
        weighting=weighting, normalize_rows=normalize_rows,
        prediction_cfg=prediction_cfg,
        before_condition="before_learning",
        after_condition="after_learning",
        reference_variant="structural_full",
    )
    summary = {
        "tag": tag,
        "dataset": recs["name"],
        "seed": seed,
        "primary_preset": PRIMARY_FINGERPRINT_PRESET,
        "primary_fingerprint": fingerprint_definition(
            after.fingerprints[PRIMARY_FINGERPRINT_PRESET].config,
            dimension=len(after.fingerprints[PRIMARY_FINGERPRINT_PRESET].feature_names),
        ),
        "fingerprint_eval_split": eval_split,
        "activity_split": activity_split,
        "meta": result["meta"],
        "table": result["table"],
        "rows": result["rows"],
        "before_after": result["before_after"],
        "rewiring_reports": rewire_reports,
        "leakage_guard": leakage_guard,
        "split_info": recs["split_info"],
        "model_extra_keys": sorted(extra.keys()) if isinstance(extra, dict) else [],
    }

    # ---- fingerprint reliability audit (after-learning, primary fingerprint) ----
    if run_reliability_audit:
        a_idx, b_idx = split_half_indices(eval_rec.labels_array, seed=seed)
        common_fp = dict(
            device=device, n_classes=n_classes, batch_size=batch_size,
            presets={"tuning": list(FINGERPRINT_PRESETS["tuning"])},
            standardize=fp_standardize, normalize_rows=fp_normalize_rows, metric=fp_metric,
            n_psth_bins=n_psth_bins, min_spikes_for_latency=min_spikes_for_latency,
        )
        fp_a = build_fingerprints(model, eval_rec, a_idx, **common_fp)["fingerprints"]["tuning"]
        fp_b = build_fingerprints(model, eval_rec, b_idx, **common_fp)["fingerprints"]["tuning"]
        rel = reliability_suite(
            after.fingerprints["tuning"], fp_a, fp_b,
            labels=eval_rec.labels_array, idx_a=a_idx, idx_b=b_idx, split_name=eval_split,
        )
        summary["fingerprint_reliability_audit"] = rel
        print(
            f"[reliability] half_r={rel['matrix_reliability_spearman']:.3f} "
            f"full_sb={rel['matrix_reliability_full_spearman_brown']:.3f} "
            f"attenuation={rel['attenuation_factor_sqrt_ceiling']:.3f} "
            f"audit_passed={rel.get('audit', {}).get('passed')}"
        )

    # ---- persist --------------------------------------------------------
    json_path = result_path(cfg, f"{tag}_stress_test.json")
    save_json(summary, json_path)
    csv_path = result_path(cfg, f"{tag}_stress_test.csv")
    _write_csv(result["table"], csv_path)
    npz_path = result_path(cfg, f"{tag}_stress_test.npz")
    np.savez_compressed(str(npz_path), **{k: np.asarray(v) for k, v in result["arrays"].items()})
    print(f"[save] summary -> {json_path}")
    print(f"[save] table   -> {csv_path}")
    print(f"[save] arrays  -> {npz_path}")

    # ---- console table --------------------------------------------------
    print("\n[stress table] condition | representation/control | raw r | rate-norm r | rate-matched r | pred R2 | p")
    for row in result["table"]:
        if row["skipped"]:
            print(f"  {row['condition']:20s} {row['representation_control']:26s} SKIPPED")
            continue
        print(
            f"  {row['condition']:20s} {row['representation_control']:26s} "
            f"{_fmt(row['raw_geometry_function_r'])} {_fmt(row['rate_normalized_association_r'])} "
            f"{_fmt(row['rate_matched_association_r'])} {_fmt(row['predictive_metric'])} {_fmt(row['permutation_significance_p'])}"
        )
    return 0


def _fmt(v: object) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "  n/a"
    if not np.isfinite(f):
        return "  n/a"
    return f"{f:+.3f}"


if __name__ == "__main__":
    raise SystemExit(main())