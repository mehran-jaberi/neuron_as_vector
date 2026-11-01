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
| aug: jitter 0.5 ms | 92.52 % | 1749/2264 = 77.25 % | +1.10 |
| aug: event dropout 0.1 | 92.65 % | 1781/2264 = 78.66 % | **+2.51** |
| aug: insertion 50 Hz | 92.52 % | 1726/2264 = 76.24 % | +0.09 |
| aug: global shift 100 ms | 92.89 % | 1751/2264 = 77.34 % | +1.19 |
| aug: channel dropout 0.1 | 92.40 % | 1768/2264 = 78.09 % | +1.94 |

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

## 2026-10-26 — raw-event augmentation (input-noise workstream)

Landed (uncommitted, PI decides the commit): `V3/v3/event_augmentation.py`
(`augment(times, units, cfg, rng)`, raw events, **before** the 2-ms binning; fixed
order channel dropout → event dropout → insertion → global shift → jitter →
`0 <= t < window_s` guard), an `augmentation` config block (all ops off by default,
phase flags `train/val/test = True/False/False`), `SHDEventStore.sample_events`
with cache bypass and per-epoch RNG, FIT-store wiring plus a hard assertion that
VAL/TEST batches are byte-identical with augmentation on/off, and a notebook
control panel (`V3/v3/control.py`, cell 0b) that applies the same
`--override` grammar temporarily. 185 tests green; `run_smoke_test.py` passes.

Screening at N=64 (table above): **event dropout and channel dropout move the
5-epoch number (+2.51 and +1.94 pp), jitter/shift marginally (+1.10/+1.19),
insertion not at all (+0.09).** Two caveats recorded now:

- These are **unpaired** against the 5-epoch baseline: the baseline run predates
  `test_predictions.npz`, so only the augmentation arms have per-sample
  predictions. Binomial SE is ±0.87 pp, so only B is near 2σ.
- **VAL is flat (92.4–92.9 % vs 93.0 % baseline) while TEST moves.** Consistent
  with augmentation attacking speaker-specific overfitting rather than generic
  fit (81 % of TEST is the two unseen speakers), not with a better fit.
- Cost: ~216 s/epoch vs 196.7 s unaugmented (+10 %), and the event cache is
  bypassed by design so every epoch is a fresh realization.

N=100 batch (baseline + the same five arms, 5 epochs, seed 0) finished — all six
rows are in the registry, and the paired picture (per-sample predictions, 10 000
resamples) is:

| arm (N=100) | correct/2264 | % | Δ vs N=100 baseline | 95 % CI | VERDICT |
|---|---|---|---|---|---|
| baseline | 1803 | 79.64 | — | — | — |
| event dropout 0.1 | 1841 | 81.32 | **+1.68** | [ +0.18, +3.14] | only arm whose CI excludes 0 |
| jitter 0.5 ms | 1814 | 80.12 | +0.49 | [ −0.88, +1.90] | noise |
| insertion 50 Hz | 1811 | 79.99 | +0.35 | [ −1.02, +1.68] | noise |
| channel dropout 0.1 | 1782 | 78.71 | −0.93 | [ −2.52, +0.71] | sign flip vs N=64 (+1.94) |
| global shift 100 ms | 1736 | 76.68 | −2.96 | [ −4.51, −1.41] | hurts |

Reading: at N=64 five arms looked better than baseline; at N=100 the baseline
itself is +3.49 pp stronger (79.64 vs 76.15), and only **event dropout** survives
both screens. Channel dropout and global shift flip sign, jitter/insertion stay
inside noise — the same failure mode as state noise 0.02 (5-epoch gain, 20-epoch
loss). This is not yet evidence: both surviving numbers are 5-epoch `screening`
and dropout's 20-epoch outcome is unmeasured. No commit yet in this round.

### Dropout probability sweep, N=64, 5 epochs (PI-directed)

| p | correct/2264 | % | train acc | val acc |
|---|---|---|---|---|
| 0.05 | 1763 | 77.87 | 95.57 | 93.63 |
| 0.10 | 1781 | 78.67 | 95.41 | 92.65 |
| 0.15 | 1749 | 77.25 | 93.65 | 90.93 |

Paired (same test samples, 10 000 resamples): 0.10−0.05 = +0.80 pp CI
[−0.75, +2.34]; 0.10−0.15 = +1.41 pp CI [−0.09, +2.92]; 0.05−0.15 = +0.62 pp
CI [−0.93, +2.16]. **The three points are statistically indistinguishable** —
a plateau 0.05–0.15 rather than a resolved optimum; 0.15 visibly underfits at
5 epochs (train 93.65). No commit yet in this round.

### 20-epoch confirmation, p=0.10, N=64 (PI-directed)

| run | test | Δ vs reference | train acc | val acc |
|---|---|---|---|---|
| reference `v3_vector_d1000` (20 ep, no aug) | 1868/2264 = 82.51 % | — | — | 96.32 % |
| `v3_aug_dropout_ep20` (20 ep, p=0.10) | **1885/2264 = 83.26 %** | **+0.75 pp** | 99.88 % | 96.69 % |
| same run, best-VAL checkpoint | 1863/2264 = 82.29 % | −0.22 pp | — | — |

Paired bootstrap (10 000 resamples, per-sample predictions of both runs):
**+0.75 pp, 95 % CI [−0.75, +2.25], one-sided p ≈ 17 %.** Speaker split (1840
unseen / 424 seen): unseen 80.38 % vs 79.29 % reference (+1.09 pp), seen 95.75 %
vs 96.46 % (−0.71 pp).

Verdict: **the 5-epoch +2.51 pp shrank to +0.75 pp at 20 epochs and is not
distinguishable from the reference.** Same failure mode as state noise 0.02, but
milder — dropout does not *hurt*, it just does not demonstrably help. Note the
model still memorises FIT (99.88 % train accuracy at epoch 20), so 10 % event
dropout does not act as a meaningful capacity constraint at this budget.
Best-VAL checkpoint is worse than final (82.29 vs 83.26), consistent with VAL
being optimistic for these weights. No commit yet in this round.

### Stacking test: jitter 0.5 ms + event dropout 0.10, N=64, 5 epochs (PI-directed)

Measured `v3_aug_jitter_dropout_ep5`: **1731/2264 = 76.46 %** (train 94.78,
val 93.14) — i.e. essentially the 5-epoch baseline (1724) and **below both
single ops**. Paired bootstrap: combo − dropout(0.10) = **−2.21 pp, 95 % CI
[−3.80, −0.66]**; combo − jitter(0.5) = −0.80 pp CI [−2.34, +0.80].

Verdict: **the two weak ops do not stack.** Two individually positive 5-epoch
deltas combine to zero; stronger evidence that the 5-epoch deltas were
run-to-run fit noise rather than a mechanism. No commit yet in this round.
