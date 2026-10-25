"""Multi-seed robustness study: run the whole pipeline at several seeds.

Stage 8 (robustness). This driver answers a question that a single training run
cannot: *how stable is the reported geometry-function relationship across
initialisations?* It runs the full three-stage pipeline -- train ->
extract representations -> geometry analysis -- once per seed and then
aggregates the PRIMARY ANALYSIS metric (the Spearman Mantel correlation
between the label-free *structural* representation space and the independent
functional fingerprint) into a mean +/- std summary.

Design notes
------------
* The train/validation split is deliberately **held fixed** across seeds via
  ``data.split_seed`` (see ``configs/analysis.yaml``). Only the initialisation
  and optimisation order vary, so the spread reflects model variability rather
  than a moving evaluation target.
* Resume-friendly: every stage is skipped when its output file already exists
  (``multiseed.skip_existing``). Seed 0 reuses the existing ``baseline`` tag, so
  a previously completed baseline run needs nothing recomputed; further seeds
  use tag ``<base>_s<seed>``.
* Each stage runs as an independent child process (``sys.executable``) calling
  the canonical scripts, so this driver never duplicates stage logic.

Outputs (under ``results/``)
----------------------------
``multiseed_summary.json``   per-seed rows + aggregated mean/std statistics
``multiseed_summary.csv``    the same table, for spreadsheets / SI appendices
``multiseed_forest.png/pdf`` forest plot of the primary metric across seeds

Examples
--------
    uv run python scripts/run_multiseed.py --config configs/analysis.yaml
    uv run python scripts/run_multiseed.py --config configs/analysis.yaml \
        --override multiseed.seeds=[0,1,2,3,4]
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

from _common import (  # noqa: E402
    PROJECT_ROOT,
    figure_dir,
    load_run_config,
    parse_args,
)
from src.utils import ensure_dir, load_json, save_json  # noqa: E402


# --------------------------------------------------------------------------
# Stage execution
# --------------------------------------------------------------------------
def _stage_tag(base_tag: str, seed: int) -> str:
    """Seed 0 keeps the plain base tag (reusing the completed baseline run)."""
    return base_tag if seed == 0 else f"{base_tag}_s{seed}"


def _run(command: list[str], log_path: Path) -> None:
    print(f"[cmd] {' '.join(command)}")
    ensure_dir(log_path.parent)
    with open(log_path, "w", encoding="utf-8") as fh:
        proc = subprocess.run(command, cwd=str(PROJECT_ROOT), stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(
            f"stage failed (exit {proc.returncode}); see {log_path}:\n  {' '.join(command)}"
        )


def _run_seed(
    seed: int,
    *,
    base_tag: str,
    train_config: str,
    analysis_config: str,
    split_seed: int,
    python: str,
    skip_existing: bool,
) -> None:
    tag = _stage_tag(base_tag, seed)
    print(f"\n==== seed {seed} (tag={tag}) ====")

    stages = [
        (
            "train",
            "scripts/train.py",
            train_config,
            f"{tag}_train_summary.json",
        ),
        (
            "extract",
            "scripts/extract_representations.py",
            analysis_config,
            f"{tag}_representations_summary.json",
        ),
        (
            "geometry",
            "scripts/run_geometry_analysis.py",
            analysis_config,
            f"{tag}_geometry_summary.json",
        ),
    ]
    for name, script, config, sentinel in stages:
        sentinel_path = _results_dir() / sentinel
        if skip_existing and sentinel_path.exists():
            print(f"[skip] {name} already done -> {sentinel_path.name}")
            continue
        command = [
            python,
            script,
            "--config",
            config,
            "--tag",
            tag,
            "--seed",
            str(seed),
            "--override",
            f"data.split_seed={split_seed}",
        ]
        _run(command, _results_dir() / f"_{tag}_{name}.log")


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def _read_structural_row(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("variant") == "structural_full" or row.get("representation") == "structural_full":
                return row
    return {}


def _float(value, default=np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _collect_seed(seed: int, base_tag: str) -> dict:
    tag = _stage_tag(base_tag, seed)
    row: dict = {"seed": seed, "tag": tag}

    geom = load_json(_results_dir() / f"{tag}_geometry_summary.json")
    primary = geom.get("primary_mantel_spearman", {})
    row["mantel_spearman_r"] = _float(primary.get("statistic"))
    row["mantel_p_value"] = _float(primary.get("p_value"))
    row["mantel_effect_size_z"] = _float(primary.get("effect_size_z"))
    row["representation_mean_rate_hz"] = _float(
        geom.get("representation_diagnostics", {}).get("mean_rate_hz")
    )
    row["fingerprint_reliability_r"] = _float(
        geom.get("fingerprint_reliability", {}).get("matrix_reliability_spearman")
    )

    struct_row = _read_structural_row(_results_dir() / f"{tag}_ablation.csv")
    row["partial_mantel_r_controlling_rate"] = _float(
        struct_row.get("partial_mantel_r_controlling_firing_rate")
    )

    train = load_json(_results_dir() / f"{tag}_train_summary.json")
    metrics = train.get("metrics", {})
    row["best_val_accuracy"] = _float(train.get("best_score"))
    row["test_accuracy"] = _float(metrics.get("test", {}).get("accuracy"))
    row["hidden_mean_rate_hz_train"] = _float(
        train.get("hidden_health_train", {}).get("mean_rate_hz")
    )
    return row


def _aggregate(rows: list[dict]) -> dict:
    def stats(key: str) -> dict:
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            return {"n": 0}
        return {
            "n": int(vals.size),
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            "min": float(vals.min()),
            "max": float(vals.max()),
            "values": [float(v) for v in vals],
        }

    return {
        "n_seeds": len(rows),
        "mantel_spearman_r": stats("mantel_spearman_r"),
        "partial_mantel_r_controlling_rate": stats("partial_mantel_r_controlling_rate"),
        "fingerprint_reliability_r": stats("fingerprint_reliability_r"),
        "test_accuracy": stats("test_accuracy"),
        "best_val_accuracy": stats("best_val_accuracy"),
        "representation_mean_rate_hz": stats("representation_mean_rate_hz"),
    }


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _forest_plot(rows: list[dict], agg: dict, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = [int(r["seed"]) for r in rows]
    vals = np.array([r["mantel_spearman_r"] for r in rows], dtype=np.float64)
    if np.all(np.isnan(vals)):
        return []
    mean = agg["mantel_spearman_r"]["mean"]
    std = agg["mantel_spearman_r"]["std"]

    fig, ax = plt.subplots(figsize=(6.2, 0.6 * len(rows) + 2.0))
    ys = np.arange(len(rows))[::-1]
    ax.errorbar(vals, ys, xerr=0.0, fmt="o", color="#1f77b4", ms=8, capsize=0, label="per-seed r")
    ax.axvline(mean, color="#d62728", ls="--", lw=1.5, label=f"mean r={mean:.3f}")
    if std > 0:
        ax.axvspan(mean - std, mean + std, color="#d62728", alpha=0.12, label=f"+/-1 std ({std:.3f})")
    ax.axvline(0.0, color="0.6", lw=1.0)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"seed {s}" for s in seeds])
    ax.set_xlabel("structural-representation Mantel Spearman r (geometry vs function)")
    ax.set_title("Multi-seed robustness of the primary geometry-function result")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    written: list[Path] = []
    for fmt in formats:
        path = out_dir / f"multiseed_forest.{fmt}"
        fig.savefig(path, dpi=dpi)
        written.append(path)
    plt.close(fig)
    return written


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _results_dir() -> Path:
    return ensure_dir(PROJECT_ROOT / "results")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = parse_args(
        "Run the full pipeline across multiple seeds and aggregate the primary metric.",
        "configs/analysis.yaml",
    )
    cfg = load_run_config(args)

    base_tag = str(cfg.get_path("run.tag", "baseline"))
    seeds = [int(s) for s in cfg.get_path("multiseed.seeds", [0, 1, 2])]
    train_config = str(cfg.get_path("multiseed.train_config", "configs/baseline.yaml"))
    analysis_config = str(cfg.get_path("multiseed.analysis_config", "configs/analysis.yaml"))
    skip_existing = bool(cfg.get_path("multiseed.skip_existing", True))
    forest = bool(cfg.get_path("multiseed.forest_plot", True))
    formats = [str(f) for f in cfg.get_path("figures.formats", ["png", "pdf"])]
    dpi = int(cfg.get_path("figures.dpi", 200))

    split_seed_raw = cfg.get_path("data.split_seed", 0)
    split_seed = 0 if split_seed_raw is None else int(split_seed_raw)

    python = sys.executable
    print(f"[multiseed] seeds={seeds} base_tag={base_tag} split_seed={split_seed}")

    for seed in seeds:
        _run_seed(
            seed,
            base_tag=base_tag,
            train_config=train_config,
            analysis_config=analysis_config,
            split_seed=split_seed,
            python=python,
            skip_existing=skip_existing,
        )

    rows = [_collect_seed(seed, base_tag) for seed in seeds]
    agg = _aggregate(rows)

    _write_csv(rows, _results_dir() / "multiseed_summary.csv")
    save_json(
        {
            "base_tag": base_tag,
            "seeds": seeds,
            "split_seed": split_seed,
            "per_seed": rows,
            "aggregate": agg,
        },
        _results_dir() / "multiseed_summary.json",
    )

    r = agg["mantel_spearman_r"]
    pr = agg["partial_mantel_r_controlling_rate"]
    ta = agg["test_accuracy"]
    print("\n[multiseed] primary Mantel r: "
          f"mean={r.get('mean', float('nan')):.3f} std={r.get('std', float('nan')):.3f} "
          f"(n={r.get('n', 0)})")
    print(f"[multiseed] partial r (ctrl rate): mean={pr.get('mean', float('nan')):.3f} "
          f"std={pr.get('std', float('nan')):.3f}")
    print(f"[multiseed] test accuracy: mean={ta.get('mean', float('nan')):.3f} "
          f"std={ta.get('std', float('nan')):.3f}")
    print(f"[save] {_results_dir() / 'multiseed_summary.json'}")

    if forest:
        written = _forest_plot(rows, agg, figure_dir(cfg), formats, dpi)
        for path in written:
            print(f"[figure] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
