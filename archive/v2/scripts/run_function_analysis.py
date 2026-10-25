"""Primary geometry/function analysis (representation vs. independent fingerprint).

This is the PRIMARY ANALYSIS entry point (see README §7 and
``src/function_analysis.py``). It:

1. builds the **label-free** neuron representation from a trained checkpoint;
2. measures **independent, class-conditioned fingerprints** on the held-out
   analysis-**probe** split - a class-tuning fingerprint (A), a rate-normalized
   tuning fingerprint (B) and a temporal fingerprint (C);
3. runs the primary Mantel analysis (Spearman distance-distance correlation with
   a neuron-relabelling permutation test, bootstrap CI and an honest p-value
   resolution floor);
4. runs the required controls (rate-only, rate-normalized fingerprint,
   rate-matched stratified Mantel, random representation, shuffled neurons,
   kNN for k = 3/5/10/20) and a secondary exploratory partial Mantel;
5. runs a cross-validated prediction of each fingerprint from the representation.

Outputs (``results/``):

* ``<tag>_function_analysis.json`` - one machine-readable summary with all primary
  and control results;
* ``<tag>_function_analysis.npz`` - the permutation null distributions and the
  condensed distance vectors.

Example (smoke test, no download)::

    uv run python scripts/run_function_analysis.py --config configs/analysis.yaml \
        --synthetic --override model.n_hidden=64 --override model.n_bins=100 \
        --override model.n_input=40 --override model.n_output=5 \
        --override train.n_classes=5 --override run.synthetic_n_samples=200 \
        --override run.synthetic_n_channels=40 --override function_analysis.n_perm=200
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
from _pipeline import build_fingerprints, build_representation_bundle  # noqa: E402
from src.evaluation import split_half_indices  # noqa: E402
from src.functional_fingerprint import (  # noqa: E402
    EXPLORATORY_FINGERPRINT_PRESET,
    FINGERPRINT_PRESETS,
    PRIMARY_FINGERPRINT_PRESET,
    fingerprint_definition,
    split_half_reliability,
)
from src.function_analysis import run_function_analysis  # noqa: E402
from src.utils import save_json  # noqa: E402


DEFAULT_PRESETS: dict[str, list[str]] = {
    "tuning": list(FINGERPRINT_PRESETS["tuning"]),
    "tuning_rate_normalized": list(FINGERPRINT_PRESETS["tuning_rate_normalized"]),
    "temporal": list(FINGERPRINT_PRESETS["temporal"]),
    "tuning_plus_temporal": list(FINGERPRINT_PRESETS["tuning_plus_temporal"]),
}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(
        "Primary geometry/function analysis: representation vs. independent fingerprint.",
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
    primary_blocks = [str(b) for b in cfg.get_path(
        "representations.primary_blocks",
        ["intrinsic", "input_conn", "recurrent_in", "recurrent_out"],
    )]
    activity_split = str(cfg.get_path("representations.activity_split", "train"))

    fa = cfg.get_path("function_analysis", {}) or {}
    fa = dict(fa)
    presets = {str(k): list(v) if isinstance(v, list) else v for k, v in (fa.get("presets") or DEFAULT_PRESETS).items()}
    primary_fp = str(fa.get("primary_fingerprint", PRIMARY_FINGERPRINT_PRESET))
    rate_norm_fp = fa.get("rate_normalized_fingerprint", "tuning_rate_normalized")
    rate_norm_fp = str(rate_norm_fp) if rate_norm_fp else None
    secondary_fps = [str(v) for v in fa.get("secondary_fingerprints", [EXPLORATORY_FINGERPRINT_PRESET, "tuning_plus_temporal"])]
    n_perm = int(fa.get("n_perm", cfg.get_path("geometry.n_perm", 10000)))
    bootstrap = int(fa.get("bootstrap", 2000))
    k_values = [int(k) for k in fa.get("k_values", cfg.get_path("geometry.k_values", [3, 5, 10, 20]))]
    n_strata = int(fa.get("rate_matched_strata", 5))
    random_repeats = int(fa.get("random_repeats", 5))
    n_psth_bins = int(fa.get("n_psth_bins", 10))
    prediction_cfg = dict(fa.get("prediction", {"enabled": True}))

    # Fingerprint configuration (feature sets come from the presets above).
    fp_standardize = str(cfg.get_path("fingerprint.standardize", "column"))
    fp_normalize_rows = bool(cfg.get_path("fingerprint.normalize_rows", False))
    fp_metric = str(cfg.get_path("fingerprint.metric", "euclidean"))
    min_spikes_for_latency = float(cfg.get_path("fingerprint.min_spikes_for_latency", 1.0))
    eval_split = str(cfg.get_path("fingerprint.eval_split", "probe"))

    # ---- model + data ---------------------------------------------------
    model, extra = load_checkpoint(cfg, device, tag)
    recs = build_recordings(cfg)
    if activity_split not in recs:
        raise ValueError(f"representations.activity_split={activity_split!r} not in {sorted(recs)}")
    if eval_split not in recs:
        raise ValueError(
            f"fingerprint.eval_split={eval_split!r} not in {sorted(recs)}. "
            "Use 'probe' (recommended), 'dev'/'val' or 'test'."
        )
    if eval_split == "test":
        print(
            "[warn] the fingerprint is being measured on the OFFICIAL TEST set. This is "
            "allowed only because the fingerprint is an evaluation target, never a model "
            "input; the test set must still never inform any modelling decision."
        )
    if eval_split in ("dev", "val"):
        print(
            "[warn] fingerprint.eval_split is the model-selection split. Prefer the held-out "
            "'probe' split so the fingerprint is independent of model selection."
        )

    ref_rec = recs[activity_split]
    ref_idx = np.arange(len(ref_rec))
    eval_rec = recs[eval_split]
    eval_idx = np.arange(len(eval_rec))

    print(f"[data] representation on '{activity_split}' | fingerprint on '{eval_split}' n={len(eval_rec)}")

    # ---- representation (label-free) ------------------------------------
    bundle = build_representation_bundle(
        model, ref_rec, ref_idx,
        device=device, n_classes=n_classes, batch_size=batch_size,
        weighting=weighting, normalize_rows=normalize_rows, block_weights=block_weights,
        include_activity_in_primary="activity" in primary_blocks,
        primary_blocks=primary_blocks,
        include_tonotopic_features=include_tonotopic,
    )
    rep_space = bundle["spaces"]["primary"]
    diag = bundle["diagnostics"]
    print(
        f"[representation] {rep_space.n_neurons} neurons x {len(rep_space.feature_names)} features "
        f"(blocks={rep_space.blocks}); mean_rate={diag['mean_rate_hz']:.1f} Hz"
    )

    # ---- fingerprints (independent target, uses labels) -----------------
    fp_bundle = build_fingerprints(
        model, eval_rec, eval_idx,
        device=device, n_classes=n_classes, batch_size=batch_size,
        presets=presets,
        standardize=fp_standardize, normalize_rows=fp_normalize_rows, metric=fp_metric,
        n_psth_bins=n_psth_bins, min_spikes_for_latency=min_spikes_for_latency,
    )
    fingerprints = fp_bundle["fingerprints"]
    for name, space in fingerprints.items():
        print(f"[fingerprint] {name:24s} dim={len(space.feature_names):3d}")
    if primary_fp not in fingerprints:
        raise KeyError(f"primary_fingerprint {primary_fp!r} not built; presets={sorted(fingerprints)}")

    # ---- reliability (noise ceiling) of the primary fingerprint ---------
    labels = eval_rec.labels_array
    a_idx, b_idx = split_half_indices(labels, seed=seed)
    fp_a = build_fingerprints(
        model, eval_rec, a_idx, device=device, n_classes=n_classes, batch_size=batch_size,
        presets={primary_fp: presets[primary_fp]},
        standardize=fp_standardize, normalize_rows=fp_normalize_rows, metric=fp_metric,
        n_psth_bins=n_psth_bins, min_spikes_for_latency=min_spikes_for_latency,
    )["fingerprints"][primary_fp]
    fp_b = build_fingerprints(
        model, eval_rec, b_idx, device=device, n_classes=n_classes, batch_size=batch_size,
        presets={primary_fp: presets[primary_fp]},
        standardize=fp_standardize, normalize_rows=fp_normalize_rows, metric=fp_metric,
        n_psth_bins=n_psth_bins, min_spikes_for_latency=min_spikes_for_latency,
    )["fingerprints"][primary_fp]
    reliability = split_half_reliability(fp_a, fp_b)
    print(
        f"[reliability] primary fingerprint matrix_r={reliability['matrix_reliability_spearman']:.3f} "
        f"ceiling_attenuation={reliability['attenuation_factor_sqrt_ceiling']:.3f}"
    )

    # ---- primary analysis + controls + prediction -----------------------
    result = run_function_analysis(
        rep_space,
        fingerprints,
        primary_fingerprint=primary_fp,
        rate_normalized_fingerprint=rate_norm_fp,
        secondary_fingerprints=secondary_fps,
        activity_reps=bundle["activity"],
        n_perm=n_perm,
        k_values=k_values,
        seed=seed,
        n_curve_bins=int(cfg.get_path("geometry.n_curve_bins", 20)),
        random_repeats=random_repeats,
        n_strata=n_strata,
        bootstrap=bootstrap,
        prediction=prediction_cfg,
    )
    summary = result["summary"]
    arrays = result["arrays"]

    summary["tag"] = tag
    summary["dataset"] = recs["name"]
    summary["seed"] = seed
    summary["fingerprint_eval_split"] = eval_split
    summary["activity_split"] = activity_split
    summary["fingerprint_reliability_primary"] = reliability
    summary["representation_diagnostics"] = diag
    summary["representation_summary"] = rep_space.summary()
    summary["fingerprints"] = {
        name: {"dim": len(space.feature_names), "feature_names": list(space.feature_names),
               "config": space.config.to_dict(), "meta": space.meta}
        for name, space in fingerprints.items()
    }
    summary["primary_preset"] = primary_fp
    summary["primary_fingerprint"] = fingerprint_definition(
        fingerprints[primary_fp].config, dimension=len(fingerprints[primary_fp].feature_names)
    )
    summary["split_info"] = recs["split_info"]
    summary["model_extra_keys"] = sorted(extra.keys()) if isinstance(extra, dict) else []

    # ---- persist --------------------------------------------------------
    json_path = result_path(cfg, f"{tag}_function_analysis.json")
    save_json(summary, json_path)
    npz_path = result_path(cfg, f"{tag}_function_analysis.npz")
    np.savez_compressed(str(npz_path), **{k: np.asarray(v) for k, v in arrays.items()})
    print(f"[save] summary -> {json_path}")
    print(f"[save] arrays  -> {npz_path}")

    # ---- console headline ----------------------------------------------
    h = summary["primary_headline"]
    print(
        f"[primary] {primary_fp}: Mantel r={h['primary_mantel_spearman_r']:.3f} "
        f"p={h['p_value']:.4g} (floor {h['p_value_floor']:.1e}, at_floor={h['at_resolution_floor']}) "
        f"z={h['effect_size_z']:.2f} CI=[{h['bootstrap_ci']['low']:.3f},{h['bootstrap_ci']['high']:.3f}]"
    )
    if h.get("rate_matched_mantel_r") is not None:
        print(
            f"[rate-matched] r={h['rate_matched_mantel_r']:.3f} "
            f"p={h['rate_matched_mantel_p_value']:.4g} z={h['rate_matched_mantel_effect_size_z']:.2f}"
        )
    for key in ("rate_normalized_fingerprint_headline", "shuffled_neurons_headline", "rate_only_headline"):
        block = summary["controls"].get(key)
        if block:
            print(
                f"[control] {key.replace('_headline',''):26s} r={block['primary_mantel_spearman_r']:.3f} "
                f"p={block['p_value']:.4g}"
            )
    rc = summary["controls"]["random_representation"]
    print(f"[control] random_representation      r={rc['primary_r_mean']:.3f} ± {rc['primary_r_std']:.3f}")
    pred = summary.get("prediction", {})
    if primary_fp in pred:
        m = pred[primary_fp]["ridge"]
        print(
            f"[prediction] ridge rep->{primary_fp}: r={m['pearson_r_mean']:.3f} "
            f"R2={m['r2_mean']:.3f} nRMSE={m['nrmse_mean']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
