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
- ~~`V3_SHD_experiment.ipynb` does not register its run (PI: known)~~ — **corrected
  2026-10-25**: it did register, but only from the last cell, after every plot, reload and
  comparison cell; the committed notebook shows the run row written and then a kernel crash.
  The real defect was *loss on crash*. Fixed in `V3/tools/build_notebook.py` (new `8b. Register
  run` cell immediately after training); notebook regenerated; 4 static tests added
  (`V3/tests/test_notebook_registration.py`, 78 tests green).
- `checkpoints/sweep_l2_0.pt` is the V2 baseline cited by `V3/RESULTS.md`; it must survive
  the archive move (path reference updated).

## Next gate

Red-flag fixes (S1a, S3b, S4a, S6a) and the notebook-registration fix are applied and
uncommitted. PI decides: commit (and push from VS Code) → then the next direction (abstract
draft vs anything else). No new experiment is running.

---

## 2026-10-25 — red-flag decisions (PI)

Post-audit red-flag round (`CLAIMS_AUDIT.md`); docs-only follow-up. No training
run and no commit in this round.

- **S1a — adopted.** The `D=1` scalar is an **untuned, size-matched control
  (ablation)**, not a competitive baseline; the docs relabel it and state that no
  literature-baseline improvement is claimed from 49.43 %. Support (`screening`,
  5 epochs, seed 0, `V3/results/runs.csv`): scalar flat at 35.20 % (N=64, 6,335
  params), 31.71 % (N=256, 72,383), 33.13 % (N=1411, 2,025,488); vector 57.20 %
  at `D=4` (9,056 params) and 76.15 % at `D=1000` (2,025,440).
- **S3b — adopted.** `VAL` (96.32 %) is annotated as speaker-overlapping
  (class-stratified random 10 % split of the training file,
  `V3/v3/data.py:169-198`), monitoring only, **not a generalisation estimate**.
- **S4a — adopted.** Protocol disclosure added to the docs: screening and model
  selection used official TEST accuracy at 5 epochs (n = 2264, ±0.84 pp binomial
  SE), by PI decision, because `VAL` saturates (96–99 %); the 20-epoch headline
  is the final-epoch model, not a test-selected checkpoint.
- **S6a — adopted.** Headline numbers use final-epoch weights
  (`V3/v3/experiment.py:192` default `which="final"`;
  `V3/v3/train.py:330-334` clones the best-`VAL` state into `result.state_dict`
  without restoring it). `checkpoints/<tag>_bestval.pt` is diagnostic only — it
  is selected on the speaker-overlapping `VAL` and must not be used to reproduce
  headline numbers.
- **Multi-seed — rejected for now.** No multi-seed runs in this round; it would be
  the last step of the project and would likely run on another machine.
- **4 ms 79.81 % (1807/2264) `doc-claim` — left as accepted docs.** No action; the
  number stays a `doc-claim` and was not re-verified.
- **Claims audit (`CLAIMS_AUDIT.md`) — flagged, docs accepted as-is.** No further
  doc changes beyond S1a/S3b/S4a/S6a in this round.

## 2026-10-26 — mechanism bucket and throughput bucket (measured)

Everything below is measured on the frozen 20-epoch checkpoint
`V3/checkpoints/v3_vector_d1000.pt` (official TEST n=2264, model readout
1868 = 82.51 %) or benchmarked against the current training loop. Details and
caveats live in `V3/ANALYSIS.md` and `V3/PERF.md`.

### What the vector state is for

| measurement | result |
|---|---|
| linear probe on 64 spike counts (fit FIT, TEST) | 1868 = 82.51 % (equals the readout) |
| linear probe on the time-mean state (64,000 dims) | 1830 = 80.83 %, Δ = −1.68 pp, 95 % CI [−3.00, −0.31] |
| linear probe on the final-timestep state | 674 = 29.77 % |
| population participation ratio (mean state) | 6.64 of 64,000 dims (23/70/460 comps at 90/95/99 %) |
| per-neuron participation ratio | median 4.21 of 1,000 (P10–P90 3.05–6.20) |
| scalar control (D=1) participation ratio | 1.71 of 64 |

No evidence the raw state decodes better than the spike counts: the width is
not extra readout bandwidth. All decodable signal sits in the temporal mean,
not the final state — consistent with a count readout.

### Is it the dynamics or the readout? (only `w_cls` fit, dynamics frozen)

| arm | dynamics | TEST |
|---|---|---|
| A1 | trained checkpoint | 1857 = 82.02 % (−0.49 pp) |
| A2 | untrained net, recalibrated threshold | 830 = 36.66 % (−45.85 pp) |
| A3 | trained weights, recurrence zeroed (leaky integrator only) | 1341 = 59.23 % (−23.28 pp) |

The learned vector dynamics are worth **+45.9 pp** over a random reservoir with
the same readout capacity; the recurrence is worth +23.3 pp; the readout adds
almost nothing irreducible. The vector width is load-bearing through the
dynamics, not at the classifier.

### Are individual neurons necessary? (knockout by silencing, θ − 1e9)

Mean TEST drop 1.02 ± 1.07 pp, worst neuron 4.46 pp, one neuron (14) *improves*
accuracy by 1.24 pp when silenced; 28/64 neurons cost > 1 pp individually;
cumulative k=1/2/4/8/16/32/64 → 4.46/7.38/13.07/22.22/34.63/43.99/77.96 pp.
Per-neuron class mutual information is flat (0.084–0.742 bits, 57/64 effective)
and does not predict causal necessity (Spearman +0.20). Distributed and
redundant; no critical unit.

### Throughput (1 epoch, FIT 7340, N=64 D=1000 T=500 @ 2 ms, batch 128)

- TODO premise **measured false**: host `np.bincount` is 8.44 s = 4 % of a warm
  epoch (GPU util 94.6 %), not ~30 %.
- Opt-in pre-binned cache (`--override binned_cache=true`, default off):
  196.72 → 190.52 s (−3.2 %), bit-identical tensors. 2.66 GiB under
  `V3/cache/` (git-ignored).
- `--resume` restores model/optimizer/scheduler/epoch/RNG and writes a new
  `<tag>_resumed` row; continuation is bit-identical on `lr_schedule=none`.
- CI entry point added (`.github/workflows/tests.yml`), not yet executed on
  GitHub.
- `torch.compile` not usable on this Windows build (`TritonMissing`); kernel
  untouched.

Tests 135 green (was 124). No commit yet in this round.
