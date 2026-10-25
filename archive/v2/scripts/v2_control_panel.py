"""V2 control panel: the single canonical entry point of the neuron-vector pipeline.

A thin CLI over :mod:`src.v2_pipeline` (which is itself a thin layer over the existing
:class:`~src.v2_config.V2Config` system). It answers one question per run:

* *what would this configuration do?*      -> ``--dry-run`` (loads nothing, trains nothing)
* *build the representation*                -> ``--build`` (FIT only; default phase)
* *evaluate the frozen representation*      -> ``--evaluate`` (PROBE only; never implicit)
* *what is the state of the project?*       -> ``--summary-for-parent`` / ``--list-presets``

Safety: building uses FIT only, evaluation uses PROBE only, and the official TEST split is
never opened. Representation construction and evaluation are **separate** operations: this
program never evaluates as a side effect of building.

Examples
--------
    uv run python scripts/v2_control_panel.py --list-presets
    uv run python scripts/v2_control_panel.py --preset historical_48 --dry-run
    uv run python scripts/v2_control_panel.py --preset historical_48
    uv run python scripts/v2_control_panel.py --preset structured_64 --build
    uv run python scripts/v2_control_panel.py --preset functional_64 --build
    uv run python scripts/v2_control_panel.py --preset functional_64 --evaluate --n-perm 200
    uv run python scripts/v2_control_panel.py --override vector.structured_d=64 \\
        --override vector.learned_residual_d=0 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from _common import PROJECT_ROOT, display_path  # noqa: F401  (adds the project root to sys.path)

from src.utils import save_json  # noqa: E402
from src.v2_pipeline import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG_PATH,
    DEFAULT_EVALUATION_TARGET,
    DEFAULT_OUT_DIR,
    DATA_POLICY,
    PipelineError,
    PRESET_DECOMPOSITIONS,
    SUPPORTED_EVALUATION_MODES,
    V2Run,
    available_presets,
    dry_run_report,
    parent_summary,
    preset_summary_rows,
    resolve_config,
)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="v2_control_panel.py",
        description=(
            "Canonical V2 neuron-vector control panel: resolve, validate, build (FIT) and "
            "evaluate (PROBE) the frozen representation. TEST is never accessed."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "presets:\n"
            + "\n".join(f"  {name:20s} {PRESET_DECOMPOSITIONS[name]}" for name in available_presets())
        ),
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="YAML configuration file")
    parser.add_argument("--preset", default=None, help="representation preset (see --list-presets)")
    parser.add_argument(
        "--override", action="append", default=[], metavar="KEY=VALUE",
        help="dotted configuration override (repeatable; applied after the preset)",
    )
    parser.add_argument("--checkpoint", default=None, help=f"checkpoint path (default: {DEFAULT_CHECKPOINT})")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="artifact directory")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--build", action="store_true", help="build the representation (FIT only)")
    parser.add_argument("--evaluate", action="store_true", help="evaluate the frozen representation (PROBE only)")
    parser.add_argument("--dry-run", action="store_true", help="resolve + validate + print the plan; load nothing")
    parser.add_argument("--no-cache", action="store_true", help="recompute activity and retrain the residual")
    parser.add_argument("--json", action="store_true", help="print the resolved configuration as JSON")
    parser.add_argument("--write-resolved-config", action="store_true",
                        help="write resolved_config.json for the resolved run (build phase writes it anyway)")
    parser.add_argument("--list-presets", action="store_true", help="list the presets and exit")
    parser.add_argument("--summary-for-parent", action="store_true",
                        help="print the machine-readable project state summary and exit")
    parser.add_argument("--run-tests", action="store_true",
                        help="with --summary-for-parent: run the full test suite for the reported result")
    parser.add_argument("--no-tests", action="store_true",
                        help="with --summary-for-parent: do not even collect the test count")
    parser.add_argument("--evaluation-mode", default=SUPPORTED_EVALUATION_MODES[0],
                        choices=list(SUPPORTED_EVALUATION_MODES))
    parser.add_argument("--evaluation-target", default=DEFAULT_EVALUATION_TARGET,
                        help="'all' (the four response variants) or one target name")
    parser.add_argument("--n-perm", type=int, default=2000)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--strict", action="store_true", help="reject any 'not implemented' request")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _print_data_policy() -> None:
    print("data policy: FIT -> build/train | DEV -> unused | PROBE -> evaluate | TEST -> never accessed")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.list_presets:
        print("V2 representation presets (each is a bag of ordinary configuration leaves):")
        for name, text in preset_summary_rows():
            print(f"  {name:20s} {text}")
        print("\ndefault (no preset): the historical 48-D configuration "
              f"({PRESET_DECOMPOSITIONS['historical_48']})")
        return 0

    try:
        resolved = resolve_config(
            config_path=args.config,
            preset=args.preset,
            overrides=args.override,
            checkpoint=args.checkpoint,
            out_dir=args.out_dir,
            evaluation_mode=args.evaluation_mode,
            evaluation_target=args.evaluation_target,
            n_perm=args.n_perm,
            bootstrap=args.bootstrap,
            n_splits=args.n_splits,
            device=args.device,
            strict=args.strict,
        )
    except PipelineError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    if args.summary_for_parent:
        print(parent_summary(
            resolved,
            include_tests=not args.no_tests,
            run_tests=bool(args.run_tests and not args.no_tests),
        ))
        return 0

    payload: dict[str, Any] = {
        "schema": resolved.schema,
        "run_id": resolved.run_id,
        "phase": (
            "build+evaluate" if (args.build and args.evaluate) else
            "build" if args.build else
            "evaluate" if args.evaluate else
            "inspect"
        ),
        "resolved_config": resolved.to_dict(),
        "data_policy": dict(DATA_POLICY),
    }

    if not args.json:
        for line in resolved.summary_lines():
            print(line)
        print()
        _print_data_policy()

    if args.write_resolved_config and not (args.build or args.evaluate):
        path = save_json(resolved.to_dict(), resolved.resolved_config_path)
        if not args.json:
            print(f"[config] resolved configuration written to {display_path(path)}")

    if args.dry_run or not (args.build or args.evaluate):
        plan = dry_run_report(resolved)
        payload["dry_run"] = plan
        if not args.json:
            print()
            print("dry run: no data is loaded, nothing is trained, PROBE is not evaluated")
            print(f"  run_id  : {plan['run_id']}")
            print(f"  dimension: {plan['dimension']['expression']}")
            print(f"  encoder blocks: {', '.join(plan['blocks'])}")
            print(f"  checkpoint: {plan['checkpoint']['path']} (exists={plan['checkpoint']['exists']})")
            print("  build steps:")
            for step in plan["phases"]["build"]:
                print(f"    {step}")
            print("  evaluate steps:")
            for step in plan["phases"]["evaluate"]:
                print(f"    {step}")
            for note in plan["notes"]:
                print(f"  note: {note}")
            for warning in plan["warnings"]:
                print(f"  warning: {warning}")
            if not (args.build or args.evaluate):
                print()
                print("no phase selected: pass --build (FIT only) and/or --evaluate (PROBE only); "
                      "--dry-run inspected the configuration only")
        if args.json:
            print(json.dumps(payload, indent=2, default=str))
        return 0

    run = V2Run.from_resolved(resolved)

    def _record(build_result=None, evaluation_result=None) -> None:
        if build_result is not None:
            payload["build"] = {
                "artifact_path": display_path(build_result.artifact_path),
                "shape": [build_result.n_neurons, build_result.d],
                "dimension_expression": build_result.dimension_expression,
                "bank_blocks": list(build_result.bank_blocks),
                "feature_names": list(build_result.feature_names),
                "fit_activity_cached": build_result.fit_activity_cached,
                "residual_trained_now": build_result.residual_trained_now,
                "residual_cache_path": (
                    display_path(build_result.residual_cache_path)
                    if build_result.residual_cache_path is not None else None
                ),
                "residual_protocol_mismatches": {
                    k: list(v) for k, v in build_result.residual_protocol_mismatches.items()
                },
                "seconds": round(build_result.seconds, 1),
            }
        if evaluation_result is not None:
            payload["evaluation"] = {
                "mode": evaluation_result.mode,
                "target": evaluation_result.target,
                "path": display_path(evaluation_result.evaluation_path),
                "settings": evaluation_result.settings,
                "rows": [
                    {
                        "target_variant": row.get("target_variant"),
                        "value": row.get("value"),
                        "bootstrap_low": row.get("bootstrap_low"),
                        "bootstrap_high": row.get("bootstrap_high"),
                        "permutation_p": row.get("permutation_p"),
                        "prediction_value": row.get("prediction_value"),
                    }
                    for row in evaluation_result.rows
                ],
                "seconds": round(evaluation_result.seconds, 1),
            }

    build_result = None
    evaluation_result = None
    try:
        if args.build:
            build_result = run.build_representation(
                use_cache=not args.no_cache, verbose=args.verbose
            )
            _record(build_result=build_result)
            if not args.json:
                print()
                for line in build_result.summary_lines():
                    print(line)
            # the resolved configuration is written next to the artifact (same run id)
            save_json(resolved.to_dict(), resolved.resolved_config_path)
            if not args.json:
                print(f"[config] resolved configuration: {display_path(resolved.resolved_config_path)}")
        if args.evaluate:
            evaluation_result = run.evaluate(
                use_cache=not args.no_cache, verbose=args.verbose
            )
            _record(evaluation_result=evaluation_result)
            if not args.json:
                print()
                for line in evaluation_result.summary_lines():
                    print(line)
    except PipelineError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
