# INVESTIGATION — living ledger

Shared memory for human + agents. Append-only: add dated entries at the bottom of a
section, correct a wrong line in place only if it was never used to justify a decision.
Numbers are tagged `confirmed` (20 epochs) / `screening` (≤ 5 epochs) / `doc-claim`
(present in a doc, not re-verified here) / `PI-reported` (stated by the PI, no artifact).

Question under study: **is a genuinely vector-valued neuron (state `z_i(t) ∈ R^D`, every
coordinate part of the dynamics) a better unit of computation than a scalar spiking neuron,
and by how much on SHD?**

## Current stage

Repo consolidation (archive V2, V3 becomes the repo). Then: a fair control design for the
vector-vs-scalar claim. Imprecision stage is closed as a negative result.

## Verified numbers (V3, `N=64`, `D=1000`, 2 ms / `T=500`, seed 0)

| config | epochs | tag | val | test | source |
|---|---|---|---|---|---|
| reference, no regularization | 20 | confirmed | 96.32 % | **1868/2264 = 82.51 %** | `runs.csv` 2026-10-02_14-31-15 |
| scalar baseline `D=1` | 20 | confirmed | 53.68 % | 1119/2264 = 49.43 % | `V3/RESULTS.md` §3 |
| reference rerun | 20 | confirmed | 96.32 % | 1868/2264 = 82.51 % | `runs.csv` 2026-10-03_22-31-13 (bit-identical → seed 0 is deterministic) |
| previous 4 ms reference | 20 | doc-claim | — | 1807/2264 = 79.81 % | `V3/README.md` §12 |

Parameter count: vector 2,025,440 vs scalar 6,335 (**×320**).

## Screening results (5 epochs, seed 0 — never quoted as final)

| config | val | test | Δ vs same-epoch baseline (76.15 %) |
|---|---|---|---|
| no regularization | 93.01 % | 1724/2264 = 76.15 % | — |
| noise 0.001 | 92.89 % | 1755/2264 = 77.52 % | +1.37 |
| noise 0.005 | 92.89 % | 1729/2264 = 76.37 % | +0.22 |
| noise 0.01 | 92.77 % | 1773/2264 = 78.31 % | +2.16 |
| noise 0.02 | 93.63 % | 1784/2264 = 78.80 % | **+2.65** |
| noise 0.05 | 87.87 % | 1546/2264 = 68.29 % | −7.86 |
| quantization 8-bit | 90.69 % | 1694/2264 = 74.82 % | −1.33 |
| quantization 4-bit | 16.67 % | 368/2264 = 16.25 % | −59.9 |

## Verdicts

- **State noise does not improve generalization at 20 epochs.** The best 5-epoch setting
  (0.02, +2.65 pp) was re-run at 20 epochs: 1847/2264 = 81.58 % vs the 82.51 % reference
  (−0.93 pp). The 5-epoch gain was the regularizer slowing early overfitting, not a better
  final model. *Closed.*
- **8-bit state quantization costs ~1.3 pp at 5 epochs; 4-bit collapses to 16.25 %.**
  A collapse to near the 5 % chance level is more consistent with a broken quantizer range
  at low bit-width (STE clamp / scale floor) than with 4-bit being intrinsically
  unusable. *Unverified — instrument question, not a scientific one. Low priority.*
- **D sweep is settled:** `D=2000` gives no accuracy gain, `D=500` is worse
  (`PI-reported`; epoch count and run rows not yet located). Do not redo this sweep.

## Open confounds in the headline claim

1. **`D=1` changes three things at once**: state width, the population-vector broadcast
   (`h(t)` requires `W_emit`, which collapses to 1×1 at `D=1`, so the recurrent drive
   `c_i(t)·h(t)` loses all vector content), and the parameter count (×320). "Vector beats
   scalar" is therefore not yet attributable to dimensionality.
   See `V3/v3/model.py` (`w_emit`, `pop_scale`, `_segment`).
2. **The scalar baseline is untuned** (49.43 % vs ~71 % quoted for a plain recurrent SHD
   baseline), so +33.08 pp partly measures a weak control.
3. **One seed.** No error bar anywhere. Deferred by PI to the end of the project.
4. **`VAL` is not speaker-independent** (stratified split of the train file; test speakers
   4–5 never occur in training), so `VAL` is optimistic as a model-selection signal.

## Decisions (dated)

- **2026-10-25** — Rules adopted (`AGENTS.md`): PI gates every step and every git action;
  all results kept, no pre-registered keep/discard rule; 5-epoch screen → 20-epoch confirm;
  no multi-seed until the end; resource caps 80 % VRAM / 60 % CPU / 80 % RAM.
- **2026-10-25** — Registry history accepted as-is: rows predating the registry and runs
  the notebook failed to register are **not** backfilled or re-run.
- **2026-10-25** — Red flag "registry gaps" dropped by PI. Red flag "no decision rule"
  dropped by PI (exploratory mode; keep everything).
- **2026-10-25** — Commit `cbfe239` ("noise experiment (0.02 noise=>+2%)") records a
  5-epoch screening result; the 20-epoch run did not reproduce it. History is not
  rewritten; this ledger is the correction.

## Known gaps (accepted, do not silently fix)

- Three 4 ms registry rows have an empty `time_bin_ms` (written before the timing columns
  existed).
- `V3_SHD_experiment.ipynb` does not register its run through the registry (PI: known).
- `checkpoints/sweep_l2_0.pt` is the V2 baseline cited by `V3/RESULTS.md`; it must survive
  the archive move (path reference updated).

## Next gate

PI decides the archive layout; then the first red-flag batch is briefed.
