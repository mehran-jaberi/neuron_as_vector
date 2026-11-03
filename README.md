# neuron_as_vector — software guide

A Vector-Valued Spiking Neuron (VSN) on the Spiking Heidelberg Digits (SHD)
dataset: each neuron's internal state is a $D$-dimensional vector that is itself
part of the dynamics, instead of the scalar membrane potential of a LIF neuron.

This file is the **software / repository guide** — what lives where, how to run
things, how results are recorded. The science (model equations, measured
numbers, what is and is not established) is in [`SCIENCE.md`](SCIENCE.md).
Rules for agents are in [`AGENTS.md`](AGENTS.md), the chronological ledger in
[`INVESTIGATION.md`](INVESTIGATION.md), the debt list in
[`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).

## Quick start

```bash
# tests: 400+, ~5 s on CPU
.venv/Scripts/python.exe -m pytest V3/tests

# 2-minute end-to-end sanity check of the real code path
.venv/Scripts/python.exe V3/run_smoke_test.py

# final-model training (~4.5 h on the 6 GiB laptop GPU), see SCIENCE.md §5
.venv/Scripts/python.exe V3/run_experiment.py --variant vector \
  --override n_neurons=100 --override epochs=30 \
  --override augmentation.enabled=true \
  --override augmentation.event_dropout.enabled=true \
  --override augmentation.event_blend.enabled=true \
  --override adaptation.enabled=true --override state_dynamics.enabled=true \
  --override state_dynamics.trainable_alpha=true --override state_dynamics.learnable_z0=true \
  --override input_delays.enabled=true --override input_delay_line.enabled=true \
  --override input_delay_line.copies=2 --override adlif.enabled=true \
  --override two_stage.enabled=true --override surrogate=slayer \
  --override timing.silence_tail_ms=100
```

## Layout

```
AGENTS.md              hard rules for every agent (the PI gates every step)
INVESTIGATION.md       dated ledger: every result, decision and confound
SCIENCE.md             the write-up: equations, numbers, controls, caveats
KNOWN_ISSUES.md        flimsy results and what would close each one
V3/
  run_experiment.py    headless CLI: train, test, register, plot
  run_smoke_test.py    2-minute end-to-end check of the real code path
  run_analysis.py      capture / probe / geometry on a frozen checkpoint
  run_ablation.py      freeze / fingerprints / knockout ablations
  run_precision_sweep.py  state-precision sweep (8-bit / 4-bit)
  v3/model.py          the neuron: state update, emission, spiking, readout
  v3/train.py          training + evaluation loops
  v3/experiment.py     orchestration shared by the CLI and the notebook
  v3/data.py           SHD event store, binning, splits, augmentation hook
  v3/event_augmentation.py  raw-event augmentations (before binning, train only)
  v3/config.py         all config dataclasses, YAML/CLI override resolution
  v3/control.py        programmatic panel: build_config / run_configured
  v3/registry.py       append-only run registry (results/runs.csv)
  v3/plots.py          figures
  tools/build_notebook.py   assembles V3_SHD_experiment.ipynb from cells
  tools/run_notebook.py     executes the notebook end to end
  configs/v3_default.yaml   recorded defaults
  tests/               ~400 tests; every feature is pinned by tests
  results/runs.csv     one row per run (git-ignored)
archive/v2/            the V2 LIF project, read-only
archive/brian/         removed early prototype (recoverable from git history)
data/                  SHD h5 files (git-ignored)
```

## How a run works

1. `run_experiment.py` parses `--variant` and repeated `--override key=value`
   pairs (`a.b=value` and `a.b.c=value` reach nested config blocks), merges them
   over `configs/v3_default.yaml`, validates, and runs.
2. `v3/experiment.py` builds the event store and the class-stratified FIT/VAL
   split, constructs the model, calibrates per-neuron thresholds to the target
   rate, trains with `v3/train.py`, scores the official TEST split once at the
   end, writes `V3/results/<tag>/` (config, per-epoch `history.csv`, metrics,
   per-sample predictions, confusion) and appends one row to
   `V3/results/runs.csv` via `v3/registry.register_run`.
3. `history.csv` has one row per epoch: fit accuracy/loss, val accuracy/loss,
   rate, learning rate, time, and — with `eval_test_each_epoch=true` — TEST
   accuracy per epoch. That flag is how the final model's learning curve is
   recorded.
4. Runs of 20+ epochs also copy the checkpoint to
   `V3/checkpoints/archive_20plus/<tag>_ep<epochs>.pt`.

## Feature blocks (all default-off, individually switchable)

Every mechanism added after the baseline lives in its own nested config block
and is off unless enabled, so the baseline path stays bit-identical:

| block | flag | what it adds |
|---|---|---|
| adaptation | `adaptation.enabled` | per-neuron spike-frequency adaptation current |
| state dynamics | `state_dynamics.enabled` | trainable per-neuron $\alpha$, learnable $z_0$ |
| input delays | `input_delays.enabled` | learnable per-channel conduction delays (0–20 ms) |
| delay line | `input_delay_line.enabled` | $K$ shifted input copies (concat or sum) |
| adLIF | `adlif.enabled` | fast/slow adaptation currents subtracted from the threshold |
| two-stage | `two_stage.enabled` | second rank-64 mixer gated by learnable $\lambda$ |
| attention | `attention.enabled` | learned softmax over time bins for the readout |
| surrogate | `surrogate=fast_sigmoid\|slayer` | SLAYER $1/(1+\alpha\lvert u\rvert)^2$ gradient |
| silence tail | `timing.silence_tail_ms` | zero bins appended after the stimulus (≤10 %) |
| augmentation | `augmentation.enabled` | raw-event augmentations before binning, training only |

`V3/v3/config.py` is the single source of truth for names and defaults.
`V3/v3/control.py` exposes the same thing to Python (`build_config`,
`apply_overrides`, `run_configured`) for the notebook panel.

## Mechanism analysis

```bash
# capture state features from a frozen checkpoint, then probe/geometry
.venv/Scripts/python.exe V3/run_analysis.py capture --checkpoint <ckpt> --tag final
.venv/Scripts/python.exe V3/run_analysis.py probe    --tag final
.venv/Scripts/python.exe V3/run_analysis.py geometry --tag final

# freeze dynamics and fit only the readout; per-neuron fingerprints; knockout
.venv/Scripts/python.exe V3/run_ablation.py freeze   --checkpoint <ckpt>
.venv/Scripts/python.exe V3/run_ablation.py fingerprints
.venv/Scripts/python.exe V3/run_ablation.py knockout
```

## Data

`data/shd_train.h5` (8156 samples, speakers 0–3 and 6–11) and
`data/shd_test.h5` (2264 samples, speakers 0–11; speakers 4–5 never appear in
training). FIT/VAL is a class-stratified random 10 % split of the training file
and therefore shares speakers with FIT — monitoring only. The official TEST
file is the only speaker-disjoint measurement. See `SCIENCE.md` §6.

## Git and history

`main` is the only development branch. Between 2026-10 and 2026-11 every
mechanism was developed on its own branch (`feat/m1-adaptation` …
`feat/m13-tail`), tested there, and merged with one commit per feature
(`git log --oneline --merges main`). Because every block is default-off, the
history reads as a sequence of independent, reversible features. The V2 project
and the early Brian prototype are archived (`archive/v2/`,
`git log brian-implementation`).

## Tests

```bash
.venv/Scripts/python.exe -m pytest V3/tests
.venv/Scripts/python.exe -m pytest V3/tests/test_feature_combinations.py -q
```

Tests may be added, never weakened. Any change to `V3/v3/model.py` or
`V3/v3/train.py` must keep `run_smoke_test.py` green.
