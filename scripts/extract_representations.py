"""Extract structured neuron representations and the functional fingerprint.

Stage 5 of the pipeline (see README). From a trained checkpoint this script
builds, for the *hidden neurons*, two strictly separated objects:

* the **neuron representation** (the object under study, label-free): intrinsic
  parameters, input-connectivity statistics, recurrent in/out statistics, and
  (optionally) label-free activity statistics measured on a reference split;
* the **functional fingerprint** (the independent evaluation target, uses
  labels): class-conditioned held-out responses.

The leakage boundary is enforced structurally: the representation is built from
the model parameters and a *label-free* accumulator, while the fingerprint is
built from a separate, labelled accumulator on a held-out split. Both are written
to ``results/`` together with a fingerprint-reliability (noise-ceiling) estimate
so that a reader can see how strong a geometry-function correlation could ever be.

Examples
--------
Smoke test (synthetic data, no download)::

    uv run python scripts/extract_representations.py --config configs/analysis.yaml \
        --synthetic --override model.n_hidden=64 --override model.n_bins=100 \
        --override model.n_input=40 --override model.n_output=5 \
        --override train.n_classes=5 --override run.synthetic_n_samples=200 \
        --override run.synthetic_n_channels=40

Real run::

    uv run python scripts/extract_representations.py --config configs/analysis.yaml
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
from _pipeline import build_fingerprint, build_representation_bundle  # noqa: E402
from src.controls import reliability_suite  # noqa: E402
from src.evaluation import split_half_indices  # noqa: E402
from src.functional_fingerprint import FingerprintConfig  # noqa: E402
from src.permutation import check_structural_permutation_invariance  # noqa: E402
from src.utils import save_json  # noqa: E402


def _save_space(space, path) -> None:
    space.save(str(path))


def _save_fingerprint(fp: dict, path_npz, path_json) -> None:
    space = fp["space"]
    np.savez_compressed(
        str(path_npz),
        X_raw=space.X_raw,
        X=space.X,
        feature_names=np.array(space.feature_names, dtype=object),
        rate_matrix=fp["rate_matrix"],
        class_n=fp["class_n"],
        mean=space.standardizer.mean,
        scale=space.standardizer.scale,
        constant_mask=space.standardizer.constant_mask,
    )
    save_json(
        {
            "config": space.config.to_dict(),
            "meta": space.meta,
            "feature_names": space.feature_names,
            "class_n": np.asarray(fp["class_n"]).tolist(),
        },
        path_json,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(
        "Extract structured neuron representations and the functional fingerprint.",
        "configs/analysis.yaml",
    )
    cfg = load_run_config(args)
    device = resolve_device(cfg)
    announce(cfg, device)
    apply_seed(cfg, device)

    tag = str(cfg.get_path("run.tag", "baseline"))
    n_classes = int(cfg.get_path("train.n_classes", 20))
    batch_size = int(cfg.get_path("train.eval_batch_size", 256))

    model, extra = load_checkpoint(cfg, device, tag)
    recs = build_recordings(cfg)

    activity_split = str(cfg.get_path("representations.activity_split", "train"))
    eval_split = str(cfg.get_path("fingerprint.eval_split", "val"))
    if activity_split not in recs:
        raise ValueError(f"representations.activity_split={activity_split!r} not in {sorted(recs)}")
    if eval_split not in recs:
        raise ValueError(f"fingerprint.eval_split={eval_split!r} not in {sorted(recs)}")
    if eval_split == "test":
        print(
            "[warn] the fingerprint is being measured on the OFFICIAL TEST set. This is "
            "allowed only because the fingerprint is an evaluation target, never a model "
            "input; the test set must still never inform any modelling decision."
        )

    ref_rec = recs[activity_split]
    ref_idx = np.arange(len(ref_rec))

    weighting = str(cfg.get_path("representations.weighting", "equal"))
    normalize_rows = bool(cfg.get_path("representations.normalize_rows", False))
    block_weights = cfg.get_path("representations.block_weights", None)
    block_weights = {str(k): float(v) for k, v in block_weights.items()} if block_weights else None
    include_tonotopic_features = bool(cfg.get_path("representations.include_tonotopic_features", False))
    primary_blocks = cfg.get_path(
        "representations.primary_blocks",
        ["intrinsic", "input_conn", "recurrent_in", "recurrent_out"],
    )
    primary_blocks = [str(b) for b in primary_blocks]
    include_activity_in_primary = "activity" in primary_blocks

    bundle = build_representation_bundle(
        model,
        ref_rec,
        ref_idx,
        device=device,
        n_classes=n_classes,
        batch_size=batch_size,
        weighting=weighting,
        normalize_rows=normalize_rows,
        block_weights=block_weights,
        include_activity_in_primary=include_activity_in_primary,
        primary_blocks=primary_blocks,
        include_tonotopic_features=include_tonotopic_features,
    )
    diag = bundle["diagnostics"]
    print(
        f"[representation] neurons={diag['n_hidden']} mean_rate={diag['mean_rate_hz']:.2f} Hz "
        f"silent_frac={diag['silent_neuron_fraction']:.3f}"
    )
    if diag["mean_rate_hz"] <= 0 or diag["silent_neuron_fraction"] >= 0.99:
        print(
            "[warn] the hidden layer is (almost) silent. The entire study is vacuous if "
            "neurons do not spike; retrain with a homeostatic penalty or lower threshold."
        )

    # ---- persist representation objects ---------------------------------
    bundle["structural"].save_json(str(result_path(cfg, f"{tag}_neuron_representations.json")))
    bundle["activity"].save_json(str(result_path(cfg, f"{tag}_neuron_activity_representations.json")))
    for name, space in bundle["spaces"].items():
        _save_space(space, result_path(cfg, f"{tag}_space_{name}.npz"))
    print(f"[save] spaces -> {result_path(cfg, f'{tag}_space_primary.npz')}")

    # ---- functional fingerprint (uses labels, held-out) ------------------
    fp_config = FingerprintConfig.from_mapping(cfg.get_path("fingerprint", {}) or {})
    eval_rec = recs[eval_split]
    eval_idx = np.arange(len(eval_rec))
    fp = build_fingerprint(
        model,
        eval_rec,
        eval_idx,
        device=device,
        n_classes=n_classes,
        batch_size=batch_size,
        fp_config=fp_config,
        min_spikes_for_latency=float(cfg.get_path("fingerprint.min_spikes_for_latency", 1.0)),
    )
    _save_fingerprint(
        fp,
        result_path(cfg, f"{tag}_fingerprint.npz"),
        result_path(cfg, f"{tag}_fingerprint.json"),
    )
    print(f"[fingerprint] split={eval_split} n={fp['space'].meta['n_samples']} features={len(fp['space'].feature_names)}")

    # ---- fingerprint reliability / noise ceiling -------------------------
    labels = eval_rec.labels_array
    a_idx, b_idx = split_half_indices(labels, seed=int(cfg.get_path("seed", 0)))
    fp_a = build_fingerprint(model, eval_rec, a_idx, device=device, n_classes=n_classes,
                             batch_size=batch_size, fp_config=fp_config)
    fp_b = build_fingerprint(model, eval_rec, b_idx, device=device, n_classes=n_classes,
                             batch_size=batch_size, fp_config=fp_config)
    reliability = reliability_suite(fp["space"], fp_a["space"], fp_b["space"])
    print(
        f"[reliability] matrix_r={reliability['matrix_reliability_spearman']:.3f} "
        f"attenuation=sqrt(ceiling)={reliability['attenuation_factor_sqrt_ceiling']:.3f}"
    )

    # ---- permutation-invariance of the structural representation ---------
    permutation_report = check_structural_permutation_invariance(
        model,
        seed=int(cfg.get_path("seed", 0)),
        include_tonotopic=include_tonotopic_features,
    )
    sensitive = permutation_report["sensitive_features"]
    if permutation_report["passed"]:
        print(
            f"[permutation] structural representation is permutation-INVARIANT "
            f"({permutation_report['n_features']} features)"
        )
    else:
        print(
            f"[permutation][WARN] permutation-SENSITIVE features detected: {sensitive}. "
            "These depend on the arbitrary neuron ordering and must not be used."
        )
    save_json(permutation_report, result_path(cfg, f"{tag}_permutation_invariance.json"))

    summary = {
        "tag": tag,
        "dataset": recs["name"],
        "seed": int(cfg.get_path("seed", 0)),
        "activity_split": activity_split,
        "fingerprint_eval_split": eval_split,
        "n_classes": n_classes,
        "weighting": weighting,
        "block_weights": block_weights,
        "include_tonotopic_features": include_tonotopic_features,
        "primary_blocks": primary_blocks,
        "representation_diagnostics": diag,
        "representation_feature_kinds": {
            name: space.summary()["feature_kinds"] for name, space in bundle["spaces"].items()
        },
        "representation_spaces": {name: space.summary() for name, space in bundle["spaces"].items()},
        "permutation_invariance": permutation_report,
        "fingerprint_meta": fp["space"].meta,
        "fingerprint_reliability": reliability,
        "split_info": recs["split_info"],
        "model_extra_keys": sorted(extra.keys()) if isinstance(extra, dict) else [],
    }
    save_json(summary, result_path(cfg, f"{tag}_representations_summary.json"))
    print(f"[save] summary -> {result_path(cfg, f'{tag}_representations_summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
