"""Rate-confound / robustness evaluation of the neuron-vector representation (PROBE).

Dedicated, reproducible entry point for the second stage of the capacity programme. It asks
how much of the first study's positive Mantel correspondence between neuron-vector geometry
and individual-stimulus PROBE responses is explained by *global firing rate* rather than
*stimulus-specific response structure*.

```
FIT   (label-free)  ->  NeuronRecordBank -> structured / +residual / activity representations
PROBE (held out)    ->  response matrix R[h, s] -> explicit target pipelines (raw, neuron
                        centered, neuron z-scored, mean rate, ordering sensitivity, row-L2)
                        -> the SAME geometry machinery as the first study + controls
```

It reuses the first study's caching infrastructure (imported from
``scripts/evaluate_vector_capacity.py``) so FIT/PROBE activity and residual artifacts are
shared, and writes to **new** files so the first study's results are never overwritten.

Examples
--------
    uv run python scripts/evaluate_vector_rate_robustness.py
    uv run python scripts/evaluate_vector_rate_robustness.py --checkpoints checkpoints/sweep_l2_0.pt
    uv run python scripts/evaluate_vector_rate_robustness.py --quick --no-figures
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from _common import PROJECT_ROOT, display_path, load_config  # noqa: F401  (adds the project root to sys.path)

from src.capacity_figures import write_rate_robustness_figures  # noqa: E402
from src.model import RecurrentLIFSNN, architecture_mismatches  # noqa: E402
from src.neuron_record import build_neuron_record_bank  # noqa: E402
from src.rate_robustness import (  # noqa: E402
    DEFAULT_RESIDUAL_CONDITIONS,
    DEFAULT_RESIDUAL_SEEDS,
    DEFAULT_STRUCTURED_DIMS,
    SCHEMA,
    RobustnessCondition,
    build_response_targets,
    check_checkpoint_compatibility,
    focused_conditions,
    representation_control_rows,
    robustness_matrix,
    run_rate_robustness,
    write_results,
)
from src.residual import (  # noqa: E402
    ResidualSourceConfig,
    ResidualTrainingConfig,
    build_residual_source,
)
from src.utils import save_json  # noqa: E402
from src.v2_config import V2Config, V2ConfigError  # noqa: E402
from src.vector_capacity import EvaluationSettings, fit_rate_reference  # noqa: E402

# Reuse the first study's caching + data helpers verbatim (no duplicated cache logic).
from evaluate_vector_capacity import (  # noqa: E402
    _cache_key,
    _file_fingerprint,
    _load_or_collect_activity,
    _load_or_train_residual,
    build_fit_probe_recordings,
)

DEFAULT_CHECKPOINTS = "checkpoints/sweep_l2_0.pt,checkpoints/nsb_seed1.pt,checkpoints/nsb_seed2.pt"
DEFAULT_OUT_DIR = "results/neuron_vector_capacity"
DEFAULT_FIGURES_DIR = "figures/neuron_vector_capacity"
RESULTS_CSV = "rate_robustness_results.csv"
RESULTS_JSON = "rate_robustness_results.json"
METADATA_JSON = "rate_robustness_metadata.json"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _parse_args(argv: Sequence[str] | None):
    parser = argparse.ArgumentParser(
        description="Rate-confound / robustness evaluation of the V2 neuron-vector representation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="configs/neuron_space_baseline.yaml")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--checkpoints", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--structured-dims", default=",".join(str(d) for d in DEFAULT_STRUCTURED_DIMS))
    parser.add_argument("--residual-conditions", default=",".join(
        f"{s}:{r}" for s, r in DEFAULT_RESIDUAL_CONDITIONS))
    parser.add_argument("--residual-seeds", default=",".join(str(s) for s in DEFAULT_RESIDUAL_SEEDS))
    parser.add_argument("--n-perm", type=int, default=2000)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--tag", default="v2rate")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--figures-dir", default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-controls", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="fast smoke settings (few permutations, first checkpoint only)")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _parse_int_list(text: str) -> list[int]:
    return [int(part) for part in str(text).replace(" ", "").split(",") if part]


def _parse_pairs(text: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for part in str(text).replace(" ", "").split(","):
        if not part:
            continue
        left, _, right = part.partition(":")
        pairs.append((int(left), int(right)))
    return pairs


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _architecture_snapshot(model: RecurrentLIFSNN) -> dict[str, Any]:
    cfg = model.cfg
    return {
        key: getattr(cfg, key)
        for key in (
            "n_input", "n_hidden", "n_output", "n_bins", "bin_ms", "tau_mem_ms", "tau_syn_ms",
            "threshold", "readout_mode", "neuron_param_mode", "recurrent_density",
            "signed_input_weights", "signed_recurrent_weights",
        )
    }


def _preserved_first_study_artifacts(out_dir: Path) -> dict[str, Any]:
    """Hash the first study's files so the metadata records that they were left intact."""
    preserved: dict[str, Any] = {}
    for name in ("results.csv", "results.json", "metadata.json"):
        path = out_dir / name
        if path.exists():
            preserved[name] = _file_fingerprint(path)
    return preserved


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.time()

    cfg = load_config(args.config, overrides=args.override)
    recs = build_fit_probe_recordings(cfg)  # opens only the SHD training file (never TEST)
    try:
        v2 = V2Config.from_config(cfg, warn=True)
    except V2ConfigError as exc:
        print(f"[v2-config] error: {exc}", file=sys.stderr)
        return 2

    checkpoint_names = [p.strip() for p in str(args.checkpoints).split(",") if p.strip()]
    if args.quick:
        args.n_perm = min(int(args.n_perm), 50)
        args.bootstrap = 0
        checkpoint_names = checkpoint_names[:1]
        args.structured_dims = "48"
        args.residual_seeds = "0"
    if not checkpoint_names:
        print("[error] no checkpoints given", file=sys.stderr)
        return 2

    structured_dims = _parse_int_list(args.structured_dims)
    residual_conditions = _parse_pairs(args.residual_conditions)
    residual_seeds = _parse_int_list(args.residual_seeds)

    out_dir = PROJECT_ROOT / str(args.out_dir)
    figures_dir = PROJECT_ROOT / str(args.figures_dir)
    cache_dir = out_dir / "cache"
    residual_dir = out_dir / "residuals"
    preserved = _preserved_first_study_artifacts(out_dir)

    fit_rec, probe_rec = recs["fit"], recs["probe"]
    print(
        f"[data] FIT n={len(fit_rec)} PROBE n={len(probe_rec)} | split={recs['split_info'].get('strategy')} "
        f"| seed={recs['split_seed']} | test_loaded={recs['test_loaded']}"
    )

    all_rows: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    compatibility_records: list[dict[str, Any]] = []
    reference: tuple[str, RecurrentLIFSNN, Any] | None = None
    targets_summary: dict[str, Any] | None = None
    conditions_summary: list[dict[str, Any]] = []
    settings_summary: dict[str, Any] | None = None
    preprocessing_summary: str | None = None

    for checkpoint_name in checkpoint_names:
        checkpoint_path = _resolve(checkpoint_name)
        label = checkpoint_path.stem
        if not checkpoint_path.exists():
            print(f"[error] checkpoint not found: {checkpoint_path}", file=sys.stderr)
            return 2

        model, checkpoint_extra = RecurrentLIFSNN.load(str(checkpoint_path), map_location="cpu")
        mismatch = architecture_mismatches(cfg.get_path("model", {}) or {}, model)
        if mismatch:
            detail = ", ".join(f"model.{k}: config={a!r} vs checkpoint={b!r}" for k, (a, b) in mismatch.items())
            print(f"[error] config/checkpoint architecture mismatch ({label}): {detail}", file=sys.stderr)
            return 2

        fingerprint = _file_fingerprint(checkpoint_path)
        cache_common = {
            "checkpoint": fingerprint["sha256_16"],
            "checkpoint_path": (
                str(checkpoint_path.relative_to(PROJECT_ROOT))
                if checkpoint_path.is_relative_to(PROJECT_ROOT) else str(checkpoint_path)
            ),
            "n_bins": int(model.cfg.n_bins),
            "bin_ms": float(model.cfg.bin_ms),
            "split_seed": int(recs["split_seed"]),
            "split_strategy": recs["split_info"].get("strategy"),
            "source": recs["source"],
        }
        provenance = {
            "checkpoint": label,
            "checkpoint_sha256_16": fingerprint["sha256_16"],
            "checkpoint_path": cache_common["checkpoint_path"],
            "split_seed": int(recs["split_seed"]),
            "architecture": _architecture_snapshot(model),
        }

        activity_batch = int(v2.memory.record_batch_size)
        fit_activity, fit_cached = _load_or_collect_activity(
            role="fit", model=model, recordings=fit_rec, device=_device(args.device), cache_dir=cache_dir,
            cache_meta={**cache_common, "role": "fit", "n": len(fit_rec), "with_labels": False},
            use_cache=not args.no_cache, verbose=True, with_labels=False,
            batch_size=activity_batch,
        )
        bank = build_neuron_record_bank(model, activity=fit_activity, config=v2)
        print(f"[{label}] bank {bank.n_neurons} neurons, blocks={list(bank.block_names)} (cache={fit_cached})")

        compatibility: dict[str, Any] | None = None
        if reference is None:
            reference = (label, model, bank)
        else:
            outcome = check_checkpoint_compatibility(reference, (label, model, bank))
            compatibility = outcome.to_dict()
            compatibility_records.append(compatibility)
            if not outcome.compatible:
                print(
                    f"[{label}] SKIPPED: not comparable to {reference[0]} -> {outcome.reasons}",
                    file=sys.stderr,
                )
                continue
            print(f"[{label}] comparable to {reference[0]} (architecture + representation schema)")

        source_config = ResidualSourceConfig.from_v2_config(v2)
        source = build_residual_source(bank, source_config)
        residual_cache_key = _cache_key({
            **cache_common, "schema_hash": source.schema_hash,
            "blocks": list(v2.vector.enabled_blocks),
        })
        residuals: dict[int, dict[int, Any]] = {}
        for _structured_d, residual_d in residual_conditions:
            for seed in residual_seeds:
                training = ResidualTrainingConfig.from_v2_config(
                    v2, residual_dim=int(residual_d), seed=int(seed), mask_seed=0,
                    hidden_dim=64, epochs=200, batch_size=64, lr=1e-3, mask_fraction=0.25,
                    normalization="train_standardise", val_fraction=0.2, split_seed=0,
                    device="cpu",
                )
                path = residual_dir / f"{residual_cache_key}_dres{residual_d}_seed{seed}.pt"
                residual, trained_now = _load_or_train_residual(
                    bank=bank, source_config=source_config, training_config=training,
                    cache_path=path, expected_names=source.feature_names,
                    use_cache=not args.no_cache, verbose=True,
                )
                residuals.setdefault(int(residual_d), {})[int(seed)] = residual
                print(
                    f"[{label}] residual d_res={residual_d} seed={seed} "
                    f"{'trained' if trained_now else 'cached'}"
                )

        probe_activity, probe_cached = _load_or_collect_activity(
            role="probe", model=model, recordings=probe_rec, device=_device(args.device),
            cache_dir=cache_dir,
            cache_meta={**cache_common, "role": "probe", "n": len(probe_rec), "with_labels": True},
            use_cache=not args.no_cache, verbose=True, with_labels=True,
            batch_size=activity_batch,
        )
        targets = build_response_targets(probe_activity, probe_split_label="probe")
        print(
            f"[{label}] targets " + ", ".join(
                f"{name}{space.X.shape}" for name, space in targets.spaces.items()
            )
        )

        conditions = focused_conditions(
            bank, structured_dims=structured_dims, residual_conditions=residual_conditions,
            residual_seeds=residual_seeds,
        )
        settings = EvaluationSettings(
            n_perm=int(args.n_perm), bootstrap=int(args.bootstrap), k_values=(3, 5, 10, 20),
            seed=int(cfg.get_path("seed", 0)), n_splits=int(args.n_splits),
            checkpoint=label, tag=str(args.tag),
        )
        fit_rates = fit_rate_reference(fit_activity)
        print(
            f"[{label}] evaluating {len(conditions)} conditions x {len(targets.variants)} target "
            f"variants | n_perm={settings.n_perm} bootstrap={settings.bootstrap}"
        )
        result = run_rate_robustness(
            bank, targets, conditions, settings=settings, fit_rates=fit_rates,
            residuals=residuals, chunk_size=int(v2.memory.representation_chunk_size),
            checkpoint=provenance,
        )

        if not args.no_controls:
            shuffle_condition = RobustnessCondition(label="structured_100", structured_d=100)
            X_shuffle, names_shuffle = robustness_matrix(shuffle_condition, bank)
            controls = representation_control_rows(
                fit_rates=fit_rates, targets=targets, settings=settings,
                shuffle_reference=(X_shuffle, names_shuffle), checkpoint=provenance,
            )
            result["rows"].extend(controls)
            print(f"[{label}] representation controls: {len(controls)} rows")

        all_rows.extend(result["rows"])
        if targets_summary is None:
            targets_summary = result["targets"]
            conditions_summary = result["conditions"]
            settings_summary = result["settings"]
            preprocessing_summary = result["preprocessing"]
        checkpoint_records.append({
            **provenance,
            "extra": {
                k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v))
                for k, v in dict(checkpoint_extra).items()
            },
            "fit_cached": bool(fit_cached),
            "probe_cached": bool(probe_cached),
            "compatibility": compatibility,
        })

    if not all_rows:
        print("[error] no rows produced", file=sys.stderr)
        return 2

    combined: dict[str, Any] = {
        "rows": all_rows,
        "settings": settings_summary,
        "targets": targets_summary,
        "conditions": conditions_summary,
        "preprocessing": preprocessing_summary,
        "checkpoints": checkpoint_records,
        "compatibility": compatibility_records,
    }
    written = write_results(
        combined,
        csv_path=out_dir / RESULTS_CSV,
        json_path=out_dir / RESULTS_JSON,
    )

    metadata: dict[str, Any] = {
        "schema": SCHEMA,
        "tag": str(args.tag),
        "config": str(args.config),
        "checkpoints": checkpoint_records,
        "compatibility": compatibility_records,
        "data": {
            "source": recs["source"],
            "fit_n": int(len(fit_rec)),
            "probe_n": int(len(probe_rec)),
            "split_seed": int(recs["split_seed"]),
            "split_info": recs["split_info"],
            "official_test_loaded": False,
            "fit_role": "representation construction + residual training only",
            "probe_role": "response targets and evaluation only",
        },
        "targets": targets_summary,
        "conditions": conditions_summary,
        "evaluation": settings_summary,
        "controls": ["control_rate_only", "control_random_100", "control_neuron_shuffle_100"],
        "preserved_first_study_artifacts": preserved,
        "outputs": {k: display_path(v) for k, v in written.items()},
        "runtime_seconds": None,
        "notes": [
            "Second stage of the capacity programme: rate/confound decomposition. No representation "
            "architecture was added and no temporal/network-context block was introduced.",
            "FIT builds the representation (bank, residual training); PROBE builds every response target; "
            "TEST is never opened.",
            "The response-target transforms (neuron centering, neuron z-scoring, mean rate, row L2) are "
            "evaluation-only transformations of the PROBE response matrix and never enter representation "
            "construction; their order is explicit and recorded per variant.",
            "control_rate_only (representation side) and mean_rate (target side) are different analyses "
            "and are reported separately.",
            "The first study's results.csv / results.json / metadata.json are not modified; the hashes "
            "recorded above were computed from those files at run time.",
        ],
    }
    if not args.no_figures:
        metadata["figures"] = write_rate_robustness_figures(combined, figures_dir)
    metadata["runtime_seconds"] = round(time.time() - started, 1)
    save_json(metadata, out_dir / METADATA_JSON)

    print(f"[done] {len(all_rows)} rows across {len(checkpoint_records)} checkpoint(s)")
    print(f"[results] {written['csv']}")
    print(f"[results] {written['json']}")
    print(f"[metadata] {out_dir / METADATA_JSON}")
    return 0


def _device(mode: str | None):
    if mode == "cpu":
        return "cpu"
    return None if mode in (None, "auto") else mode


if __name__ == "__main__":
    raise SystemExit(main())
