"""Evaluate the new label-free source information (functional response + temporal).

Second scientific stage of the capacity programme's representation work: it asks whether the
functional-response projection and/or the coarse temporal block change the correspondence
between the neuron representation and stimulus-specific PROBE response structure — in
particular the **neuron z-scored** target, which removes each neuron's mean level and amplitude.

```
FIT   (label-free)  ->  NeuronRecordBank (with the coarse temporal block)
                    ->  deterministic structured / + temporal / activity conditions
                    ->  learned residual with the source ablation A/B/C/D (+ block masking)
PROBE (held out)    ->  the robustness study's response targets (raw/centered/z-scored/mean-rate)
                    +  the existing class-rate / temporal targets
                    ->  the same Mantel / prediction machinery and conventions
```

Everything reuses the previous stages' machinery unchanged
(:mod:`src.source_extension`, :mod:`src.rate_robustness`, :meth:`src.vector_capacity` geometry),
writes to **new** files only, and never opens the official TEST split.

Examples
--------
    uv run python scripts/evaluate_vector_source_extension.py
    uv run python scripts/evaluate_vector_source_extension.py --quick --no-figures
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from _common import PROJECT_ROOT, display_path, load_config  # noqa: F401  (adds the project root to sys.path)

from src.capacity_figures import write_source_extension_figures  # noqa: E402
from src.functional_response import FunctionalResponseConfig, build_functional_response_source  # noqa: E402
from src.model import RecurrentLIFSNN, architecture_mismatches  # noqa: E402
from src.neuron_record import TEMPORAL_BLOCK, build_neuron_record_bank  # noqa: E402
from src.rate_robustness import check_checkpoint_compatibility  # noqa: E402
from src.residual import ResidualResult, ResidualTrainingConfig, build_residual_source  # noqa: E402
from src.source_extension import (  # noqa: E402
    ABLATION_RESIDUAL_DIM,
    DEFAULT_RESIDUAL_DIMS,
    MASK_BLOCK,
    SCHEMA,
    SOURCE_FUNCTIONAL_TEMPORAL,
    SOURCE_KEYS,
    SourceCondition,
    SourceExtensionError,
    build_stage_targets,
    build_payload,
    focused_source_conditions,
    run_source_extension,
    source_config_for,
    secondary_target_variants,
    stage_target_variants,
    write_results,
)
from src.utils import save_json  # noqa: E402
from src.v2_config import V2Config, V2ConfigError  # noqa: E402
from src.vector_capacity import EvaluationSettings, fit_rate_reference  # noqa: E402

# Reuse the previous stages' caching + data helpers verbatim (no duplicated cache logic).
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
RESULTS_CSV = "source_extension_results.csv"
RESULTS_JSON = "source_extension_results.json"
METADATA_JSON = "source_extension_metadata.json"

#: Files from the previous stages that this script must never overwrite.
PRESERVED_FILES: tuple[str, ...] = (
    "results.csv",
    "results.json",
    "metadata.json",
    "rate_robustness_results.csv",
    "rate_robustness_results.json",
    "rate_robustness_metadata.json",
)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _parse_args(argv: Sequence[str] | None):
    parser = argparse.ArgumentParser(
        description="Source-extension evaluation (functional-response and temporal sources).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="configs/neuron_space_baseline.yaml")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--checkpoints", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--structured-d", type=int, default=48)
    parser.add_argument("--residual-dims", default=",".join(str(d) for d in DEFAULT_RESIDUAL_DIMS))
    parser.add_argument("--residual-seeds", default="0,1,2")
    parser.add_argument("--n-perm", type=int, default=2000)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--tag", default="v2srcext")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--figures-dir", default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-activity", action="store_true", help="skip the activity conditions")
    parser.add_argument("--no-mask-diagnostic", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="fast smoke settings (few permutations, first checkpoint, one seed)")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _parse_int_list(text: str) -> list[int]:
    return [int(part) for part in str(text).replace(" ", "").split(",") if part]


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _device(mode: str | None):
    if mode == "cpu":
        return "cpu"
    return None if mode in (None, "auto") else mode


def _artifact_path(
    residual_dir: Path, base_key: str, *, residual_d: int, seed: int, mask_mode: str
) -> Path:
    """Artifact path: the source schema + dimension + seed, plus the masking variant.

    ``coordinate`` (the historical default) keeps the previous stages' filename, so the
    structural-only artifacts cached there are reused; ``block`` is a distinct artifact of the
    same source schema (different training protocol), hence the explicit suffix.
    """
    suffix = "" if mask_mode != MASK_BLOCK else "_maskblock"
    return residual_dir / f"{base_key}_dres{int(residual_d)}_seed{int(seed)}{suffix}.pt"


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


def _preserved_artifacts(out_dir: Path) -> dict[str, Any]:
    """Hash the previous stages' files so the metadata records that they were left intact."""
    preserved: dict[str, Any] = {}
    for name in PRESERVED_FILES:
        path = out_dir / name
        if path.exists():
            preserved[name] = _file_fingerprint(path)
    return preserved


def _verify_sources(bank: Any, v2: V2Config, *, functional_source_dim: int | None) -> dict[str, Any]:
    """Record the verified definitions of the two new sources (no scientific interpretation)."""
    temporal = dict(bank.provenance.get("temporal", {}))
    temporal_present = bool(temporal.get("present"))
    functional = build_functional_response_source(
        bank,
        FunctionalResponseConfig(
            source_dim=int(functional_source_dim or v2.vector.functional_source_dim),
            projection_seed=int(v2.vector.functional_projection_seed),
            normalization=str(v2.vector.functional_source_normalization),
        ),
    )
    return {
        "temporal": {
            "present": temporal_present,
            "mode": temporal.get("mode"),
            "resolution": temporal.get("resolution"),
            "split": temporal.get("split"),
            "n_samples": temporal.get("n_samples"),
            "units": "mean label-free FIT firing rate (Hz) per coarse interval per neuron",
            "binning": "linspace(0, n_bins, resolution + 1).round(), >= 1 sim bin per coarse bin",
            "bins": list(temporal.get("bins", [])),
            "class_conditioned": bool(temporal.get("class_conditioned")),
            "uses_labels": False,
        },
        "functional_response": {
            "source_type": functional.provenance.get("source_type"),
            "source_split": functional.provenance.get("source_split"),
            "input_shape": functional.provenance.get("input_shape"),
            "output_dimension": functional.provenance.get("output_dimension"),
            "projection": dict(functional.provenance.get("projection", {})),
            "normalization": dict(functional.provenance.get("normalization", {})),
            "sample_order": dict(functional.provenance.get("sample_order", {})),
            "uses_labels": False,
            "display_name": functional.feature_names[0] if functional.feature_names else None,
        },
    }


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
    residual_dims = _parse_int_list(args.residual_dims)
    residual_seeds = _parse_int_list(args.residual_seeds)
    if args.quick:
        args.n_perm = min(int(args.n_perm), 50)
        args.bootstrap = 0
        checkpoint_names = checkpoint_names[:1]
        residual_dims = [ABLATION_RESIDUAL_DIM]
        residual_seeds = residual_seeds[:1]
    if not checkpoint_names:
        print("[error] no checkpoints given", file=sys.stderr)
        return 2

    out_dir = PROJECT_ROOT / str(args.out_dir)
    figures_dir = PROJECT_ROOT / str(args.figures_dir)
    cache_dir = out_dir / "cache"
    residual_dir = out_dir / "residuals"
    preserved = _preserved_artifacts(out_dir)

    fit_rec, probe_rec = recs["fit"], recs["probe"]
    print(
        f"[data] FIT n={len(fit_rec)} PROBE n={len(probe_rec)} | split={recs['split_info'].get('strategy')} "
        f"| seed={recs['split_seed']} | test_loaded={recs['test_loaded']}"
    )
    temporal_resolution = int(v2.vector.temporal_resolution)
    print(
        f"[v2] structured_d={v2.vector.structured_d} learned_residual_d={v2.vector.learned_residual_d} "
        f"temporal_resolution={temporal_resolution} functional_source_dim={v2.vector.functional_source_dim} "
        f"({v2.vector.functional_source_normalization}, seed {v2.vector.functional_projection_seed}, "
        f"mask_mode={v2.vector.residual.mask_mode})"
    )

    all_rows: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    compatibility_records: list[dict[str, Any]] = []
    source_verification: dict[str, Any] = {}
    residual_records: dict[str, Any] = {}
    reference: tuple[str, RecurrentLIFSNN, Any] | None = None
    targets_summary: dict[str, Any] | None = None
    conditions_summary: list[dict[str, Any]] = []
    settings_summary: dict[str, Any] | None = None
    matrices_summary: dict[str, Any] = {}
    variants_summary: dict[str, Any] = {}

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
            use_cache=not args.no_cache, verbose=True, with_labels=False, batch_size=activity_batch,
        )
        # the bank carries the coarse temporal block so the temporal conditions are buildable
        bank = build_neuron_record_bank(
            model, activity=fit_activity, config=v2, temporal_resolution=temporal_resolution
        )
        print(
            f"[{label}] bank {bank.n_neurons} neurons, blocks={list(bank.block_names)} "
            f"(fit cache={fit_cached}, temporal present={TEMPORAL_BLOCK in bank.block_names})"
        )

        compatibility: dict[str, Any] | None = None
        if reference is None:
            reference = (label, model, bank)
        else:
            outcome = check_checkpoint_compatibility(reference, (label, model, bank))
            compatibility = outcome.to_dict()
            compatibility_records.append(compatibility)
            if not outcome.compatible:
                print(f"[{label}] SKIPPED: not comparable to {reference[0]} -> {outcome.reasons}",
                      file=sys.stderr)
                continue
            print(f"[{label}] comparable to {reference[0]} (architecture + representation schema)")

        conditions = focused_source_conditions(
            bank,
            structured_d=int(args.structured_d),
            residual_dims=residual_dims,
            residual_seeds=residual_seeds,
            include_activity=not args.no_activity,
            include_mask_diagnostic=not args.no_mask_diagnostic,
        )
        residuals, records = _prepare_residuals(
            bank, conditions, v2=v2, cache_common=cache_common,
            residual_dir=residual_dir, use_cache=not args.no_cache,
        )
        residual_records[label] = records
        if records:
            trained = sum(1 for r in records.values() if r["trained_now"])
            print(
                f"[{label}] residual artifacts: {len(records)} "
                f"({trained} trained now, {len(records) - trained} from cache), "
                f"source dims={sorted({int(r['source_n_features']) for r in records.values()})}"
            )

        probe_activity, probe_cached = _load_or_collect_activity(
            role="probe", model=model, recordings=probe_rec, device=_device(args.device),
            cache_dir=cache_dir,
            cache_meta={**cache_common, "role": "probe", "n": len(probe_rec), "with_labels": True},
            use_cache=not args.no_cache, verbose=True, with_labels=True, batch_size=activity_batch,
        )
        targets = build_stage_targets(
            probe_activity, probe_split_label="probe",
            n_psth_bins=int(cfg.get_path("fingerprint.n_psth_bins", 10)),
            min_spikes_for_latency=float(cfg.get_path("fingerprint.min_spikes_for_latency", 1.0)),
        )
        print(
            f"[{label}] targets: response={len(targets.response.spaces)} variants "
            f"({targets.n_stimuli} PROBE utterances) + class_rate/temporal (cache={probe_cached})"
        )

        settings = EvaluationSettings(
            n_perm=int(args.n_perm), bootstrap=int(args.bootstrap), k_values=(3, 5, 10, 20),
            seed=int(cfg.get_path("seed", 0)), n_splits=int(args.n_splits),
            checkpoint=label, tag=str(args.tag),
        )
        result = run_source_extension(
            bank, targets, conditions, settings=settings,
            fit_rates=fit_rate_reference(fit_activity),
            residuals=residuals,
            checkpoint=provenance,
            chunk_size=int(v2.memory.representation_chunk_size),
        )
        print(
            f"[{label}] evaluated {len(conditions)} conditions x "
            f"{len(stage_target_variants()) + len(secondary_target_variants())} targets "
            f"| n_perm={settings.n_perm} bootstrap={settings.bootstrap} "
            f"| {time.time() - started:.0f}s elapsed"
        )

        all_rows.extend(result["rows"])
        if targets_summary is None:
            targets_summary = result["targets"]
            conditions_summary = result["conditions"]
            settings_summary = result["settings"]
            variants_summary = {
                "main": list(result["variants"]),
                "secondary": list(result["secondary_variants"]),
            }
        matrices_summary[label] = result["matrices"]
        source_verification.setdefault(label, _verify_sources(
            bank, v2, functional_source_dim=v2.vector.functional_source_dim
        ))
        checkpoint_records.append({
            **provenance,
            "extra": {
                k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v))
                for k, v in dict(checkpoint_extra).items()
            },
            "fit_cached": bool(fit_cached),
            "probe_cached": bool(probe_cached),
            "temporal_block_present": bool(TEMPORAL_BLOCK in bank.block_names),
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
        "checkpoints": checkpoint_records,
        "compatibility": compatibility_records,
        "source_verification": source_verification,
        "matrices": matrices_summary,
        "metadata": None,
    }

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
            "fit_role": "representation construction + residual source + residual training only",
            "probe_role": "response targets and evaluation only",
        },
        "targets": targets_summary,
        "conditions": conditions_summary,
        "evaluation": settings_summary,
        "target_variants": variants_summary,
        "source_verification": source_verification,
        "matrices": matrices_summary,
        "residual_source_artifacts": residual_records,
        "residual_source_configurations": {
            key: source_config_for(
                key,
                enabled_blocks=v2.vector.enabled_blocks,
                functional_source_dim=int(v2.vector.functional_source_dim),
                functional_projection_seed=int(v2.vector.functional_projection_seed),
                functional_normalization=str(v2.vector.functional_source_normalization),
            ).to_dict()
            for key in SOURCE_KEYS
        },
        "preserved_previous_artifacts": preserved,
        "outputs": None,
        "figures": None,
        "runtime_seconds": None,
        "notes": [
            "Source-extension evaluation: the functional-response projection and the coarse temporal "
            "block are evaluated with the previous stages' targets, geometry machinery and conventions.",
            "FIT builds every representation and trains every residual; PROBE builds the targets and the "
            "metrics; the official TEST split is never opened.",
            "The residual configuration (sources, dimension, seed, mask mode, architecture, epochs) is "
            "pre-specified and identical across conditions; nothing was tuned on the PROBE metric.",
            "Only the residual's input view differs across the A/B/C/D ablation arms.",
            "No ranking, no 'best' and no causal claim is implied; the measured values and their "
            "uncertainty are reported.",
        ],
    }
    combined["metadata"] = metadata

    payload = build_payload(combined)
    written = write_results(
        combined,
        csv_path=out_dir / RESULTS_CSV,
        json_path=out_dir / RESULTS_JSON,
    )
    metadata["outputs"] = {k: display_path(v) for k, v in written.items()}
    if not args.no_figures:
        previous = _load_previous_decomposition(out_dir)
        metadata["figures"] = write_source_extension_figures(
            payload, figures_dir, previous_decomposition=previous
        )
    metadata["runtime_seconds"] = round(time.time() - started, 1)
    save_json(metadata, out_dir / METADATA_JSON)

    print(f"[done] {len(all_rows)} rows across {len(checkpoint_records)} checkpoint(s)")
    for name, digest in preserved.items():
        print(f"[preserved] {name} sha256[:16]={digest['sha256_16']}")
    print(f"[results] {written['csv']}")
    print(f"[results] {written['json']}")
    print(f"[metadata] {out_dir / METADATA_JSON}")
    return 0


def _prepare_residuals(
    bank: Any,
    conditions: Sequence[SourceCondition],
    *,
    v2: V2Config,
    cache_common: dict[str, Any],
    residual_dir: Path,
    use_cache: bool,
) -> tuple[dict[str, ResidualResult], dict[str, Any]]:
    """Build the deterministic source view and train/load the frozen residual per condition.

    The residual architecture, epochs, mask fraction, seed handling and FIT/validation split are
    identical for every condition (and identical to the previous stages); only the residual's
    **input view** (the A/B/C/D ablation) and the mask mode differ. Artifacts are cached per
    ``(checkpoint, source schema, residual dim, residual seed, mask mode)``.
    """
    residuals: dict[str, ResidualResult] = {}
    records: dict[str, Any] = {}
    for condition in conditions:
        if condition.residual_d == 0 or condition.artifact_key in residuals:
            continue
        source_config = source_config_for(
            condition.source_key,
            enabled_blocks=v2.vector.enabled_blocks,
            functional_source_dim=int(v2.vector.functional_source_dim),
            functional_projection_seed=int(v2.vector.functional_projection_seed),
            functional_normalization=str(v2.vector.functional_source_normalization),
            functional_chunk_size=int(v2.memory.representation_chunk_size),
        )
        source = build_residual_source(bank, source_config)
        base_key = _cache_key({
            **cache_common,
            "schema_hash": source.schema_hash,
            "blocks": list(v2.vector.enabled_blocks),
        })
        path = _artifact_path(
            residual_dir, base_key,
            residual_d=condition.residual_d, seed=int(condition.residual_seed),
            mask_mode=condition.mask_mode,
        )
        training = ResidualTrainingConfig.from_v2_config(
            v2,
            residual_dim=int(condition.residual_d), seed=int(condition.residual_seed),
            mask_seed=0, hidden_dim=64, epochs=200, batch_size=64, lr=1e-3, mask_fraction=0.25,
            normalization="train_standardise", val_fraction=0.2, split_seed=0, device="cpu",
            mask_mode=condition.mask_mode,
        )
        residual, trained_now = _load_or_train_residual(
            bank=bank, source_config=source_config, training_config=training, cache_path=path,
            expected_names=source.feature_names, use_cache=use_cache, verbose=True,
        )
        residuals[condition.artifact_key] = residual
        functional_prov = source.provenance.get("functional_response", {})
        records[condition.artifact_key] = {
            **condition.to_dict(),
            "source_n_features": int(source.n_features),
            "source_schema_hash": source.schema_hash,
            "source_groups": {
                name: len(values)
                for name, values in source.provenance.get("source_groups", {}).items()
            },
            "source_blocks": [
                {"name": b.get("name"), "dimension": b.get("dimension"), "enabled": b.get("enabled")}
                for b in source.provenance.get("source_blocks", [])
            ],
            "functional_sample_order_hash": (
                functional_prov.get("sample_order", {}).get("sample_order_hash")
                if functional_prov.get("enabled") else None
            ),
            "artifact_path": display_path(path),
            "trained_now": bool(trained_now),
            "best_epoch": int(getattr(residual, "best_epoch", -1)),
            "best_val_loss": float(getattr(residual, "best_val_loss", float("nan"))),
            "history_last": dict(residual.history[-1]) if residual.history else {},
        }
    return residuals, records


def _load_previous_decomposition(out_dir: Path) -> dict[str, Any] | None:
    """The previous robustness study's decomposition table, as optional figure context."""
    import json

    path = out_dir / "rate_robustness_results.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # pragma: no cover - defensive
        return None
    table = payload.get("rate_decomposition_table")
    if not table:
        return None
    for entry in table:
        if entry.get("representation") == "structured_48":
            return entry
    return None


if __name__ == "__main__":
    raise SystemExit(main())
