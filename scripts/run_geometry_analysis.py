"""Geometry-function analysis: does neuron-space geometry predict function?

Stage 6-7 of the pipeline. This is the scientific core of the project. It takes a
trained checkpoint and asks the *pre-registered* question:

    Do hidden neurons that are close in the structured representation space also
    tend to have similar functional fingerprints (held-out, class-conditioned
    responses)?

Pipeline
--------
1. Build the label-free neuron representation (structural + optional activity)
   and the independent functional fingerprint on a held-out split.
2. Run the primary analysis (Spearman Mantel test on the two distance matrices)
   on the *primary* representation, collecting the condensed distance vectors,
   the binned distance-distance curve and the kNN table for figures.
3. Run the full ablation / control suite (``src.controls.default_variants``):
   random null, rate-only trivial baseline, shuffled control, per-block and
   leave-one-block-out variants, and the (circular, never-reported) fingerprint
   variant, all through the identical geometry pipeline. A partial Mantel test
   controls for firing-rate distance.
4. Compare an untrained and a trained network in the identical architecture
   (before/after learning).

Outputs (all under ``results/``)
--------------------------------
``<tag>_ablation.csv`` / ``.json``   headline table, one row per variant
``<tag>_geometry_analyses.json``     full analysis per variant
``<tag>_before_after.json``          untrained vs trained comparison
``<tag>_figure_data.npz``            arrays used by make_figures.py
``<tag>_figure_bundle.json``         structured data used by make_figures.py

Examples
--------
Smoke test (synthetic)::

    uv run python scripts/run_geometry_analysis.py --config configs/analysis.yaml \
        --synthetic --override model.n_hidden=64 --override model.n_bins=100 \
        --override model.n_input=40 --override model.n_output=5 \
        --override train.n_classes=5 --override run.synthetic_n_samples=200 \
        --override run.synthetic_n_channels=40 --override geometry.n_perm=200
"""

from __future__ import annotations

import csv
from pathlib import Path

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
from _pipeline import build_fingerprint, build_representation_bundle  # noqa: E402
from src.controls import (  # noqa: E402
    before_after_table,
    default_variants,
    rate_nuisance_condensed,
    reliability_suite,
    run_variant_suite,
)
from src.evaluation import split_half_indices  # noqa: E402
from src.functional_fingerprint import FingerprintConfig  # noqa: E402
from src.geometry_analysis import geometry_function_analysis  # noqa: E402
from src.model import build_model, count_parameters  # noqa: E402
from src.utils import ensure_dir, save_json  # noqa: E402


# --------------------------------------------------------------------------
# Small I/O helpers
# --------------------------------------------------------------------------
def _write_rows_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    # Union of keys, preserving first-seen order.
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fingerprint_order(rate_matrix: np.ndarray) -> np.ndarray:
    """Order neurons by preferred class, then by mean rate (for the heat map)."""
    R = np.asarray(rate_matrix, dtype=np.float64)  # (C, H)
    if R.size == 0:
        return np.zeros(0, dtype=np.int64)
    preferred = R.argmax(axis=0)
    mean_rate = R.mean(axis=0)
    return np.lexsort((mean_rate, preferred))


def _pca_block(space, values: np.ndarray | None) -> dict:
    pca = space.pca(n_components=2)
    return {
        "coords": np.asarray(pca["scores"]),
        "explained": [float(v) for v in pca["explained_variance_ratio"]],
        "values": None if values is None else np.asarray(values, dtype=np.float64),
        "value_label": "firing rate (Hz)",
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = parse_args("Geometry-function analysis of hidden neurons.", "configs/analysis.yaml")
    cfg = load_run_config(args)
    device = resolve_device(cfg)
    announce(cfg, device)
    apply_seed(cfg, device)

    tag = str(cfg.get_path("run.tag", "baseline"))
    seed = int(cfg.get_path("seed", 0))
    n_classes = int(cfg.get_path("train.n_classes", 20))
    batch_size = int(cfg.get_path("train.eval_batch_size", 256))

    n_perm = int(cfg.get_path("geometry.n_perm", 1000))
    k_values = [int(k) for k in cfg.get_path("geometry.k_values", [3, 5, 10, 20])]
    n_curve_bins = int(cfg.get_path("geometry.n_curve_bins", 20))
    random_repeats = int(cfg.get_path("geometry.random_repeats", 5))

    weighting = str(cfg.get_path("representations.weighting", "equal"))
    normalize_rows = bool(cfg.get_path("representations.normalize_rows", False))
    block_weights = cfg.get_path("representations.block_weights", None)
    block_weights = {str(k): float(v) for k, v in block_weights.items()} if block_weights else None
    include_tonotopic_features = bool(cfg.get_path("representations.include_tonotopic_features", False))
    primary_blocks = [str(b) for b in cfg.get_path(
        "representations.primary_blocks",
        ["intrinsic", "input_conn", "recurrent_in", "recurrent_out"],
    )]
    activity_split = str(cfg.get_path("representations.activity_split", "train"))
    eval_split = str(cfg.get_path("fingerprint.eval_split", "val"))

    model, extra = load_checkpoint(cfg, device, tag)
    recs = build_recordings(cfg)
    ref_rec = recs[activity_split]
    ref_idx = np.arange(len(ref_rec))
    eval_rec = recs[eval_split]
    eval_idx = np.arange(len(eval_rec))
    fp_config = FingerprintConfig.from_mapping(cfg.get_path("fingerprint", {}) or {})

    # ---- representation + fingerprint for the trained model --------------
    bundle = build_representation_bundle(
        model, ref_rec, ref_idx,
        device=device, n_classes=n_classes, batch_size=batch_size,
        weighting=weighting, normalize_rows=normalize_rows,
        block_weights=block_weights,
        include_activity_in_primary="activity" in primary_blocks,
        primary_blocks=primary_blocks,
        include_tonotopic_features=include_tonotopic_features,
    )
    structural = bundle["structural"]
    activity = bundle["activity"]
    primary_space = bundle["spaces"]["primary"]
    diag = bundle["diagnostics"]

    fp = build_fingerprint(
        model, eval_rec, eval_idx,
        device=device, n_classes=n_classes, batch_size=batch_size, fp_config=fp_config,
    )
    fingerprint = fp["space"]
    print(
        f"[data] representation={primary_space.n_neurons} neurons / {len(primary_space.feature_names)} features "
        f"| fingerprint on '{eval_split}' n={fingerprint.meta['n_samples']}"
    )

    nuisance = rate_nuisance_condensed(activity)

    # ---- primary analysis on the primary representation ------------------
    primary_analysis = geometry_function_analysis(
        primary_space.X, fingerprint.X,
        n_perm=n_perm, k_values=k_values, seed=seed,
        n_curve_bins=n_curve_bins, nuisance_condensed=nuisance,
    )
    p = primary_analysis["primary_mantel_spearman"]
    print(
        f"[primary] Mantel Spearman r={p['statistic']:.3f} p={p['p_value']:.4g} "
        f"(null mean {p['null_mean']:.3f}, z={p['effect_size_z']:.2f})"
    )

    # ---- full ablation / control suite -----------------------------------
    suite = run_variant_suite(
        default_variants(), structural, fingerprint,
        activity_reps=activity,
        n_perm=n_perm, k_values=k_values, seed=seed,
        weighting=weighting, normalize_rows=normalize_rows, block_weights=block_weights,
        nuisance_condensed=nuisance, n_random_repeats=random_repeats,
    )
    rows = suite["rows"]
    _write_rows_csv(rows, result_path(cfg, f"{tag}_ablation.csv"))
    save_json(
        {"n_perm": n_perm, "rows": rows, "analyses": suite["analyses"],
         "primary_blocks": primary_blocks, "weighting": weighting},
        result_path(cfg, f"{tag}_ablation.json"),
    )
    print(f"[save] ablation table -> {result_path(cfg, f'{tag}_ablation.csv')}")

    # ---- before vs after learning ----------------------------------------
    before_after = None
    if bool(cfg.get_path("controls.include_before_after", True)):
        untrained = build_model(model.cfg, seed=seed, device=device)
        nt = count_parameters(untrained)
        print(f"[before/after] untrained model params={nt['total']}")

        bundle_before = build_representation_bundle(
            untrained, ref_rec, ref_idx,
            device=device, n_classes=n_classes, batch_size=batch_size,
            weighting=weighting, normalize_rows=normalize_rows,
            include_activity_in_primary="activity" in primary_blocks,
            primary_blocks=primary_blocks,
        )
        fp_before = build_fingerprint(
            untrained, eval_rec, eval_idx,
            device=device, n_classes=n_classes, batch_size=batch_size, fp_config=fp_config,
        )
        nuisance_before = rate_nuisance_condensed(bundle_before["activity"])
        before_after = before_after_table(
            bundle_before["spaces"], bundle["spaces"],
            fp_before["space"], fingerprint,
            n_perm=n_perm, k_values=k_values, seed=seed,
            nuisance_before=nuisance_before, nuisance_after=nuisance,
        )
        _write_rows_csv(before_after["rows"], result_path(cfg, f"{tag}_before_after.csv"))
        save_json(before_after, result_path(cfg, f"{tag}_before_after.json"))
        for row in before_after["rows"]:
            if row.get("variant") == "structural_full":
                print(
                    f"[before/after] structural_full: r_before={row['r_before']:.3f} "
                    f"r_after={row['r_after']:.3f} delta={row['delta_r_after_minus_before']:+.3f}"
                )

    # ---- fingerprint reliability -----------------------------------------
    a_idx, b_idx = split_half_indices(eval_rec.labels_array, seed=seed)
    fp_a = build_fingerprint(model, eval_rec, a_idx, device=device, n_classes=n_classes,
                             batch_size=batch_size, fp_config=fp_config)
    fp_b = build_fingerprint(model, eval_rec, b_idx, device=device, n_classes=n_classes,
                             batch_size=batch_size, fp_config=fp_config)
    reliability = reliability_suite(fingerprint, fp_a["space"], fp_b["space"])
    print(f"[reliability] matrix_r={reliability['matrix_reliability_spearman']:.3f}")

    # ---- figure data ------------------------------------------------------
    rate_mean = fp["rate_matrix"].mean(axis=0)
    order = _fingerprint_order(fp["rate_matrix"])
    np.savez_compressed(
        result_path(cfg, f"{tag}_figure_data.npz"),
        dx=primary_analysis["_condensed"]["representation"],
        dy=primary_analysis["_condensed"]["functional"],
        curve_center=primary_analysis["_distance_curve"]["bin_center"],
        curve_mean=primary_analysis["_distance_curve"]["bin_mean"],
        curve_sem=primary_analysis["_distance_curve"]["bin_sem"],
        pca_coords=_pca_block(primary_space, rate_mean)["coords"],
        rate_matrix=fp["rate_matrix"],
        neuron_order=order,
        class_n=np.asarray(fp["class_n"]),
    )
    history_path = result_path(cfg, f"{tag}_history.json")
    history = None
    if history_path.exists():
        from src.utils import load_json

        history = load_json(history_path)

    figure_bundle = {
        "tag": tag,
        "history": history,
        "pca": {
            "coords": _pca_block(primary_space, rate_mean)["coords"].tolist(),
            "explained": _pca_block(primary_space, rate_mean)["explained"],
            "values": rate_mean.tolist(),
            "value_label": "firing rate (Hz)",
        },
        "primary": {
            "dx": primary_analysis["_condensed"]["representation"].tolist(),
            "dy": primary_analysis["_condensed"]["functional"].tolist(),
            "mantel": primary_analysis["primary_mantel_spearman"],
            "curve": {
                "bin_center": primary_analysis["_distance_curve"]["bin_center"].tolist(),
                "bin_mean": primary_analysis["_distance_curve"]["bin_mean"].tolist(),
                "bin_sem": primary_analysis["_distance_curve"]["bin_sem"].tolist(),
            },
        },
        "knn_rows": primary_analysis["knn"]["table"],
        "ablation_rows": rows,
        "fingerprint": {
            "rate_matrix": fp["rate_matrix"].tolist(),
            "class_names": [str(c) for c in range(n_classes)],
            "order": order.tolist(),
            "reliability": reliability,
        },
    }
    save_json(figure_bundle, result_path(cfg, f"{tag}_figure_bundle.json"))

    summary = {
        "tag": tag,
        "seed": seed,
        "dataset": recs["name"],
        "n_perm": n_perm,
        "primary_blocks": primary_blocks,
        "primary_mantel_spearman": p,
        "representation_diagnostics": diag,
        "fingerprint_reliability": reliability,
        "activity_split": activity_split,
        "fingerprint_eval_split": eval_split,
        "n_variants": len(rows),
        "split_info": recs["split_info"],
    }
    save_json(summary, result_path(cfg, f"{tag}_geometry_summary.json"))
    print(f"[save] geometry summary -> {result_path(cfg, f'{tag}_geometry_summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
