"""Print the resolved V2 configuration (configuration demonstration only).

This script does **not** train a model, build a ``NeuronRecord``, or run any
experiment. It loads a YAML config plus ``--override dotted.key=value`` arguments
through the repository's existing :func:`src.utils.load_config`, resolves and
validates the V2 configuration object (:class:`src.v2_config.V2Config`) and prints
it. Use it to confirm that ``n_hidden``, ``d``, ``structured_d``,
``learned_residual_d``, the batch/chunk sizes, dtype and device can all be changed
without editing any source code.

.. note::
   For *execution* the canonical V2 entry point is ``scripts/v2_control_panel.py``
   (:mod:`src.v2_pipeline`): it resolves the same configuration, supports presets and
   ``--dry-run``, and additionally builds/evaluates the representation. This script
   remains the minimal configuration-only demonstrator.

Examples::

    uv run python scripts/show_v2_config.py
    uv run python scripts/show_v2_config.py --config configs/neuron_space_baseline.yaml
    uv run python scripts/show_v2_config.py \
        --override vector.d=64 --override vector.structured_d=48 \
        --override vector.learned_residual_d=16 --override vector.residual.enabled=true \
        --override memory.record_batch_size=16 --override memory.representation_chunk_size=64 \
        --override precision.vector_dtype=fp16 --override memory.device=cpu

Exit codes: 0 = resolved successfully, 2 = invalid/contradictory configuration.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from _common import PROJECT_ROOT  # noqa: F401  (puts the project root on sys.path)

from src.utils import format_table, load_config
from src.v2_config import V2Config, V2ConfigError


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/v2_example.yaml", help="path to a YAML config file")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted config override, e.g. --override vector.d=100 (repeatable)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="turn 'requested but not implemented in this stage' warnings into errors",
    )
    parser.add_argument("--no-warn", dest="warn", action="store_false", help="do not print warnings")
    parser.add_argument("--json", action="store_true", help="print the resolved configuration as JSON")
    parser.set_defaults(warn=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        cfg = load_config(args.config, overrides=args.override)
        # `warn=False`: this script prints the resolved warnings itself, once, below.
        v2 = V2Config.from_config(cfg, strict=args.strict, warn=False)
    except V2ConfigError as exc:
        print(f"[v2-config] error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(v2.to_dict(), indent=2, default=str))
        return 0

    print(format_table(v2.summary_rows(), ["v2 parameter", "value"]))
    if args.warn:
        print()
        if v2.warnings:
            for message in v2.warnings:
                print(f"[v2-config] warning: {message}")
        else:
            print("[v2-config] no warnings")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via `uv run`
    raise SystemExit(main())
