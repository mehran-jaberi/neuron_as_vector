# neuron_as_vector

Do **vector-valued** spiking neurons — a neuron whose internal state is a
$D$-dimensional vector that is itself part of the dynamics — compute more than
scalar neurons, and how much more on SHD?

Active code: **`V3/`**. Everything earlier is archived (see [Archives](#archives)).

## Current numbers

| model | config | epochs | test (official, 2264) | tag |
|---|---|---|---|---|
| vector neuron | `N=64`, `D=1000`, 2 ms | 20 | **1868 = 82.51 %** | confirmed |
| scalar neuron | `N=64`, `D=1`, same training | 20 | 1119 = 49.43 % | confirmed |
| V2 LIF reference | `N=256`, T=1400 ms | 20 | 59.98 % | doc-claim |

Numbers are seed 0 only (no error bar yet), and the vector-vs-scalar gap is
**confounded** — see `INVESTIGATION.md` for what is and is not established.

## Layout

```
AGENTS.md            rules for every agent working in this repo (read first)
INVESTIGATION.md     living ledger: verified numbers, verdicts, confounds, decisions
V3/                  the active experiment (code, tests, notebook, results)
  v3/model.py        the vector-neuron population — the science
  v3/train.py        training / evaluation loops
  v3/experiment.py   orchestration shared by the CLI and the notebook
  run_experiment.py  headless CLI
  results/runs.csv   the run registry: one row per run, never edited by hand
archive/             V2 and earlier work, kept for reference, not maintained
data/                SHD (`shd_train.h5`, `shd_test.h5`), git-ignored
```

## Run it

```bash
# full experiment (vector + scalar), ~3.5 h on the 3060
.venv/Scripts/python.exe V3/run_experiment.py --variant both

# quick end-to-end check of the same code path (minutes)
.venv/Scripts/python.exe V3/run_experiment.py --variant both --quick

# tests
.venv/Scripts/python.exe -m pytest V3/tests
```

See `V3/README.md` for the model, `V3/RESULTS.md` for every measured number,
`V3/TODO.md` for deliberately unimplemented work.

## Archives

- `archive/v2/` — the V2 LIF project: `src/`, `scripts/`, `configs/`, `tests/`,
  `notebooks/`, `README_V2.md`, `V2_STATUS.md`, `VECTOR_V2_AUDIT.md`, and its
  `artifacts/` (checkpoints, results, figures). Its baseline checkpoint
  `archive/v2/artifacts/checkpoints/sweep_l2_0.pt` is the 59.98 % reference
  quoted in `V3/RESULTS.md`.
- `archive/brian/` — the early Brian prototype (fixed high-dimensional
  dynamical system with a trained readout) was removed from the tree in
  2026-10; it is recoverable from git history on branch `brian-implementation`.
