"""Train the selected baseline recipe at the multi-seed replication seeds.

The neuron-space experiment must be replicated across seeds (0, 1, 2) with the
**identical** recipe, architecture and data split, so that seed-to-seed variation
reflects only initialisation / optimisation - not a different experiment.

The recipe is the one that was selected on DEV in the baseline-repair stage:

    readout_mode = sum ; l2_spikes = 0 ; Adam lr 1e-3 ; cosine schedule
    40-epoch budget with DEV early stopping (patience 8) ; 700 x 2 ms ; 256 hidden

Seed 0 reuses the already-selected checkpoint (``checkpoints/sweep_l2_0.pt``) so
the experiment runs on exactly the checkpoint the previous stage certified; the
remaining seeds are trained here. **No selection happens in this script**: DEV
early stopping is the only criterion, and PROBE / TEST are never inspected.

Outputs
-------
* ``checkpoints/nsb_seed<seed>.pt`` for every newly trained seed;
* ``results/neuron_space_baseline/checkpoints.json`` - seed -> checkpoint path.

Example::

    uv run python scripts/train_nsb_baseline.py --config configs/neuron_space_baseline.yaml
"""

from __future__ import annotations

import sys

import numpy as np

from _common import (  # noqa: E402
    PROJECT_ROOT,
    checkpoint_path,
    ensure_dir,
    load_config,
    resolve_device,
)
from src.evaluation import circuit_health, evaluate_model  # noqa: E402
from src.model import build_model  # noqa: E402
from src.training import TrainConfig, train_model  # noqa: E402
from src.utils import save_json  # noqa: E402

OUTPUT_DIR = "results/neuron_space_baseline"


def _parse_args(argv: list[str] | None):
    import argparse

    p = argparse.ArgumentParser(description="Train the selected LIF baseline at the replication seeds.")
    p.add_argument("--config", default="configs/neuron_space_baseline.yaml")
    p.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--seeds", default="", help="comma-separated seeds (default: config multiseed.seeds)")
    p.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    p.add_argument("--force", action="store_true", help="retrain even if a checkpoint already exists")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = load_config(args.config, overrides=args.override)
    if args.device:
        cfg.set_path("run.device", args.device)
    device = resolve_device(cfg)

    recipe = dict(cfg.get_path("select_recipe", {}) or {})
    base_config = str(recipe.get("base_config", "configs/baseline_repaired.yaml"))
    overrides = dict(recipe.get("overrides", {}) or {})
    reuse_seed0 = recipe.get("reuse_seed0_checkpoint", None)
    seeds = (
        [int(s) for s in args.seeds.split(",") if s.strip()]
        if args.seeds
        else [int(s) for s in cfg.get_path("multiseed.seeds", [0, 1, 2])]
    )
    out_dir = ensure_dir(PROJECT_ROOT / OUTPUT_DIR)
    print(f"[nsb-train] base={base_config} overrides={overrides} seeds={seeds} device={device}")

    manifest: dict[str, str] = {}
    for seed in seeds:
        # Seed 0: reuse the certified checkpoint.
        if seed == 0 and reuse_seed0 is not None:
            reuse_path = PROJECT_ROOT / str(reuse_seed0)
            if reuse_path.exists() and not args.force:
                manifest[str(seed)] = str(reuse_seed0).replace("\\", "/")
                print(f"[nsb-train] seed 0 -> reusing certified checkpoint {reuse_seed0}")
                continue

        target = checkpoint_path(cfg, f"nsb_seed{seed}")
        if target.exists() and not args.force:
            manifest[str(seed)] = f"checkpoints/{target.name}"
            print(f"[nsb-train] seed {seed} -> existing {target.name} (skip; use --force to retrain)")
            continue

        ov = [f"{k}={v}" for k, v in overrides.items()] + [f"seed={seed}"]
        run_cfg = load_config(base_config, overrides=ov)
        n_classes = int(run_cfg.get_path("train.n_classes", 20))
        batch_size = int(run_cfg.get_path("train.eval_batch_size", 256))

        from _common import build_recordings

        recs = build_recordings(run_cfg)
        train_rec, dev_rec = recs["train"], recs["dev"]
        train_idx = np.arange(len(train_rec))
        dev_idx = np.arange(len(dev_rec))

        model = build_model(run_cfg, seed=seed, device=device)
        tcfg = TrainConfig.from_mapping(run_cfg.get_path("train", {}))
        print(
            f"[nsb-train] seed {seed}: epochs_budget={tcfg.epochs} patience="
            f"{tcfg.early_stopping_patience} readout={model.cfg.readout_mode} l2_spikes={tcfg.l2_spikes}"
        )
        result = train_model(
            model, train_rec, train_idx, dev_rec, dev_idx, tcfg,
            device=device, seed=seed, verbose=not args.quiet,
        )

        dev_m = evaluate_model(model, dev_rec, dev_idx, device=device, batch_size=batch_size,
                               n_classes=n_classes, compute_confusion=True)
        health = circuit_health(model, train_rec, train_idx[:2000], device=device,
                                batch_size=batch_size, n_classes=n_classes)
        rates = np.asarray(health["rate_hz_per_neuron"], dtype=np.float64)

        model.save(str(target), extra={
            "nsb_seed": int(seed),
            "nsb_recipe_overrides": {k: str(v) for k, v in overrides.items()},
            "model_config": model.cfg.to_dict(),
            "train_config": tcfg.to_dict(),
            "split_info": recs["split_info"],
            "epochs_run": len(result.history),
            "best_epoch": int(result.best_epoch),
            "dev_accuracy": float(dev_m["accuracy"]),
            "hidden_mean_rate_hz": float(rates.mean()) if rates.size else float("nan"),
            "hidden_max_rate_hz": float(rates.max()) if rates.size else float("nan"),
        })
        manifest[str(seed)] = f"checkpoints/{target.name}"
        print(
            f"[nsb-train] seed {seed} saved -> {target.name} | epochs_run={len(result.history)} "
            f"best_epoch={result.best_epoch} dev={dev_m['accuracy']:.4f} "
            f"rate_mean={rates.mean():.2f} Hz"
        )

    save_json(
        {
            "seeds": seeds,
            "checkpoints": manifest,
            "base_config": base_config,
            "recipe_overrides": {k: str(v) for k, v in overrides.items()},
            "note": (
                "Seed 0 reuses the DEV-selected certified checkpoint. All seeds use the "
                "identical recipe, architecture and fixed data split; no PROBE/TEST "
                "information is used for selection or early stopping."
            ),
        },
        out_dir / "checkpoints.json",
    )
    print(f"[nsb-train] manifest -> {out_dir / 'checkpoints.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))