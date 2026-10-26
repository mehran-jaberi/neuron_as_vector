# ANALYSIS — what the vector state carries

Model: `V3/checkpoints/v3_vector_d1000.pt`, 20 epochs, seed 0, final-epoch weights — the
reference model of `INVESTIGATION.md` (TEST 1868/2264 = 82.51 %, **confirmed**).
Control: `V3/checkpoints/v3_vector_scalar_d1.pt` (same 20-epoch budget, same seed;
TEST 1119/2264 = 49.43 %, **confirmed**). All numbers below were reproduced
from the frozen checkpoints by `V3/run_analysis.py` (no retraining, no config changes).

## Methods

1. `model.forward(x, return_states=True)` under `torch.no_grad()`, checkpoint dtype, batch 128,
   `phase="val"` for fit/val and `phase="test"` for the official test file.
2. Captured per sample: spike counts (the model's own readout feature), the final state
   (after the last update) and the mean state (fp32 time-mean of the `(N,D)` state read at
   each of the 500 timesteps); states stored as fp16.
3. **Primary probe — validated.** Multinomial logistic regression (linear softmax) in torch.
   Features are standardised per dimension with FIT mean/std only. Adam (lr 0.01, batch 256,
   30 epochs, zero init, fixed seed 0); L2 penalty `λ·ΣW²`, with `λ` selected on VAL
   (816 samples) from a decades-wide grid `{1e-8, …, 1}`; zero init and one seeded shuffle
   make a fit deterministic. Fitted on FIT (7340); TEST (2264) is only ever scored. The
   reported table uses one fit per feature set; the raw-vs-standardised comparison and the
   λ/TEST sensitivity sweep below are auxiliary diagnostics, disclosed, and were **not**
   used to select any headline probe.
4. **Estimator gate.** The counts probe must land within 1 pp of the model's own counts-only
   readout (1868/2264 = 82.51 %), i.e. 1845–1890 correct. It scores **1868/2264**.
5. **Uncertainty.** Paired bootstrap (1000 resamples of the 2264 TEST samples, identical
   resample indices for every probe) on `accuracy(mean state) − accuracy(counts)`.
6. Legacy ridge probe (one-hot ridge, 5-fold accuracy CV of `λ` inside the fitting split) is
   kept only as the estimator-limited robustness note below; its counts row is 4.5–5.8 pp
   below the validated probe and below the trained readout.
7. Geometry: PCA of the mean state over the 2264 TEST samples; participation ratio
   `PR=(Σλ)²/Σλ²`; components needed for 90/95/99 % variance; same per neuron (its
   1000-dim subspace across samples). Scalar control captured and analysed identically.

## Result 1 — validated linear probe (TEST n = 2264)

| probe feature (dim)          | VAL accuracy (selected λ) | TEST correct / 2264 | TEST accuracy | in-sample FIT |
|------------------------------|---------------------------|---------------------|---------------|---------------|
| model readout, counts (64)   | —                         | 1868                | 82.51 % *confirmed* | —        |
| spike counts (64)            | 96.94 % (1e-4)            | 1868                | 82.51 %       | 100.00 %      |
| mean state (64 000)          | 96.08 % (1e-6)            | 1830                | 80.83 %       | 99.86 %       |
| final state (64 000)         | 40.81 % (1e-8)            | 674                 | 29.77 %       | 48.79 %       |
| counts + mean state (64 064) | 95.96 % (1e-8)            | 1819                | 80.34 %       | 99.70 %       |

- **Gate passes exactly**: the validated counts probe reproduces the model's counts-only
  readout (1868/2264), so labels, split alignment and the estimator are not in question on
  the 64-dim feature.
- **Delta (mean state − counts)**: −1.68 pp (−38 samples), 95 % paired-bootstrap CI
  **[−3.00, −0.31] pp; the CI excludes 0** (fraction of resamples with delta ≤ 0 = 0.995).
- **Standardisation matters, and not uniformly.** On TEST the mean state scores 1849
  (81.67 %) raw vs 1830 (80.83 %) standardised, while the final state scores 629 (27.78 %)
  raw vs 674 (29.77 %) standardised. Standardisation also changes the VAL-optimal λ
  (mean state: 1e-7 raw vs 1e-6 standardised). It is not a free gain.
- **λ sensitivity, disclosed.** A post-hoc TEST sweep of the mean-state probe (standardised,
  30 epochs; not used for selection) gives 1744 (77.03 %) at 1e-5, 1830 (80.83 %) at 1e-6,
  1795 (79.28 %) at 1e-4 and 1808 (79.86 %) at 1e-3. All are below the counts probe's 1868,
  so the *sign* of the delta is stable, but its magnitude is not; the CI above is conditional
  on the VAL-selected λ and does not include λ-selection variance. `final_state` and
  `counts+mean_state` select λ=1e-8 at the grid edge, with VAL curves that are flat to
  within ~1 pp (`counts+mean_state`) and ~4 pp (`final_state`) over 1e-8…1e-5, so their
  exact numbers are the least stable of the four.
- **Reading.** Under the validated estimator the mean state decodes at **80.83 %**, close to
  but statistically below the **82.51 %** count readout; the final state carries little class
  information (29.77 %); and adding counts to the mean state does not improve on either
  (80.34 %, with VAL preferring the mean state alone). The earlier ridge estimate of 83.30 %
  for the mean state is **not confirmed**. There is no evidence that the vector state decodes
  better than the counts: at the optimistic end of the delta CI it is still 0.3 pp below.
  Caveat: VAL is not speaker-disjoint from FIT (`INVESTIGATION.md:64`), so VAL-based λ
  selection is optimistic for speaker-correlated state features; the counts probe is anchored
  by the gate, the state numbers inherit that caveat.

## Robustness note — legacy ridge probe (estimator-limited, do not headline)

Ridge regression on one-hot targets, features standardised on the fitting split, ridge
strength chosen by 5-fold stratified **accuracy** CV inside the fitting split; TEST only
scored. Preserved for provenance (`V3/results/analysis/probe_results.json`); **every row is
estimator-limited**.

| probe feature (dim)          | fitted on VAL (816) | fitted on FIT (7340) | in-sample FIT |
|------------------------------|---------------------|----------------------|---------------|
| spike counts (64)            | 1736 (76.68 %)      | 1766 (78.00 %)       | 95.91 %       |
| final state (64 000)         | 713 (31.49 %)       | 761 (33.61 %)        | 84.59 %       |
| mean state (64 000)          | 1808 (79.86 %)      | 1886 (83.30 %)       | 100.00 %      |
| counts + mean state (64 064) | 1808 (79.86 %)      | 1891 (83.52 %)       | 100.00 %      |

- The ridge count probe is **4.5–5.8 pp below** the validated counts probe and below the
  trained counts readout: a probe that cannot recover a 64-dim linear readout is
  estimator-limited, and every state-vs-counts gap it reports inherits that.
- Its 83.30 % mean-state estimate is not reproduced by the validated probe (80.83 %). Treat
  the ridge state rows as an **unconfirmed upper estimate**, not as evidence that the state
  beats the counts.
- GCV-selected λ instead of CV-selected λ lowers the state rows by 2.8–5.5 points (e.g.
  mean state FIT-fitted 80.48 % vs 83.30 %) and the counts rows by ~1 point; both are
  recorded in `probe_results.json`.
- The old "best" entry (`counts+mean state`, FIT-fitted, 1891/2264) was selected **post hoc
  on TEST** and is therefore optimistic; per-class numbers are in the JSON.

## Result 2 — state geometry (TEST, mean state)

| scope                                   | ambient dim | PR (eff. dims) | 90 % | 95 % | 99 % |
|-----------------------------------------|-------------|----------------|------|------|------|
| vector, population (all neurons)        | 64 000      | 6.64           | 23   | 70   | 460  |
| vector, per neuron (median of 64)       | 1 000       | 4.21           | 9    | 15   | 222  |
| scalar D=1, population                  | 64          | 1.71           | 3    | 4    | 8    |
| scalar D=1, per neuron                  | 1           | 1.00           | 1    | 1    | 1    |

- Vector per-neuron PR: mean 4.51, median 4.21, P10–P90 3.05–6.20, min 2.64, max 6.86.
- Top-1 / top-10 population variance fractions: 31.9 % / 82.8 %.
- Per-neuron diagonal variance PR: median 719 of 1000, and **zero** coordinates have zero
  variance — the variance is spread over all 1000 coordinates while the covariance is
  low-rank. Participation ratio therefore answers "how many directions", not "how many
  coordinates are non-constant".
- The scalar population is itself ~1.7-dimensional across samples (75.1 % of its variance
  in one component, 3 components for 90 %), so between-neuron spread inflates the
  population dimension; the vector state adds ~4 effective dimensions *per neuron* over
  the scalar's 1.
- Because one neuron's 1000-dim state varies, across samples, along only ~4 effective
  directions, the vector state should be read as a handful of dynamically accessible
  directions per neuron, not as 1000 (or 64) independent channels.

## What this does not show

1. **Not causal.** A linear probe proves the class is decodable from the state, not that
   the model's readout (`counts @ w_cls`) uses those directions. The state may carry
   information the readout ignores, or vice versa; no ablation of state dimensions was run.
2. **FIT-fitted probes are in-sample for the model.** They reach ~100 % in-sample accuracy
   and are trained on the same examples the model trained on; TEST is held out from the
   probe, but the features themselves are in-sample, so the TEST numbers may still be
   optimistic about a probe trained on a fresh model.
3. **Linearity and selection variance.** A linear probe cannot rule out nonlinear
   decodability. The state probe's λ sits on a flat VAL plateau and its TEST accuracy moves
   by several points across near-tied λ (see Result 1); VAL is speaker-overlapping, so the
   state-vs-counts delta is not resolved below ~±2 pp by this protocol alone.
4. **Geometry ≠ usage.** PCA is computed on the TEST mean state of one checkpoint/seed, and
   PCA directions need not align with readout directions or with dynamics actually used in
   the forward loop.
5. **No scalar probe.** Only the scalar geometry was captured/measured here; the D=1 vs
   D=1000 comparison is geometric, not a probe-accuracy comparison.
6. **fp16 capture.** States were stored as fp16 before probing/PCA (counts are exact);
   this is far below the probe's own error bars but is not bit-exact.
7. The weak classes (17, 19) remain weak under the validated probe too (counts: 37.2 %,
   50.0 %; mean state: 43.0 %, 31.2 %) and were not investigated.

## Reproduce

```
python V3/run_analysis.py capture                 # fit, val, test  (~2.5 min GPU)
python V3/run_analysis.py capture --checkpoint V3/checkpoints/v3_vector_scalar_d1.pt \
    --tag scalar_d1 --splits test                 # (~20 s GPU)
python V3/run_analysis.py probe --device auto --bootstrap 1000   # validated table + CI
python V3/run_analysis.py probe --estimator ridge                # legacy ridge table
python V3/run_analysis.py geometry                # vector (default tag), then --tag scalar_d1
```

Outputs: `V3/results/analysis/capture_{split}.npz`, `capture_scalar_d1_test.npz`,
`probe_logistic.json`, `probe_logistic_predictions.npz`, `probe_results.json`,
`geometry.json`, `geometry_scalar_d1.json` (git-ignored).

## Result 2 (mechanism) — is it the dynamics or the readout?

*Numbering note: these two mechanism sections are the "Result 2 / Result 3" of the
mechanism brief; this file already numbers the state-geometry section Result 2, hence the
`(mechanism)` suffixes.*

Protocol tag: **confirmed** (checkpoint `V3/checkpoints/v3_vector_d1000.pt`, 20 epochs,
seed 0; official TEST n = 2264). Every arm freezes the dynamics and fits **only** the
readout `w_cls` (64 x 20) on the per-sample spike counts (500 bins): zero init, Adam
lr 0.01, batch 256, plain cross-entropy, no L2, at most 20 epochs, early stop on VAL
accuracy with patience 5, best-VAL weights restored before TEST is scored. Code:
`V3/run_ablation.py freeze` (constants `V3/run_ablation.py:57-76`).

| arm | dynamics (frozen) | w_cls used | TEST | acc | gap vs 82.51 % |
|-----|-------------------|------------|------|-----|----------------|
| gate | trained checkpoint | its own (trained) | 1868/2264 | 82.51 % | 0.00 pp |
| A1 | trained checkpoint | fresh, on cached exact counts | 1857/2264 | 82.02 % | −0.49 pp |
| A2 | untrained, `cfg.seed=1` (init generator 12346), threshold recalibrated on the same 128 FIT samples as training | fresh | 830/2264 | 36.66 % | −45.85 pp |
| A3 | trained, `w_rec = w_emit = w_down = w_up = 0` (input drive only) | fresh | 1341/2264 | 59.23 % | −23.28 pp |

- A1: the readout is **re-derivable**. A fresh linear softmax on the frozen counts lands
  0.49 pp below the checkpoint's own readout (best VAL epoch 7, 10 epochs, 1.2 s CPU);
  the 82.51 % therefore does not depend on the trained `w_cls`'s particular values
  (well inside the brief's 2 pp co-adaptation threshold).
- A2: the reservoir control scores 36.66 %. An untrained network of the same size with a
  trained-population-style threshold calibration (10 Hz, first 128 FIT samples) already
  makes the classes linearly decodable far above the 5 % chance level, but 45.85 pp below
  the trained model. Its zero-init readout predicts 365/7340 FIT (4.97 %) and 103/2264
  TEST (4.55 %), confirming the network really is untrained.
- A3: removing the state dynamics from the trained weights leaves 59.23 %. Keeping the
  trained input drive (w_in / gate / bias) but zeroing the mixing (`w_down w_up`) and the
  spike->population (`w_emit`) and spike->recurrent (`w_rec`) terms costs 23.28 pp; the
  checkpoint's trained `w_cls` on those counts drops to 174/2264 (7.69 %), i.e. it is not
  transferable to the altered dynamics.
- Reading the three arms together: the readout costs ~0.5 pp, untrained dynamics cost
  45.9 pp, zeroed state dynamics cost 23.3 pp. The learned recurrent/population coupling —
  not the readout — carries the margin over a trained input-drive path, and the trained
  weights as a whole carry another 23 pp over a random reservoir.
- Cost: A2 and A3 captures 74.9 + 8.5 + 22.8 s and 75.3 + 8.4 + 22.8 s (fit/val/test,
  batch 128, fp16 GPU; 461 MB peak torch allocation); A1 is CPU-only (cached counts).

What this does **not** show:

1. One checkpoint and **one** A2 seed (`cfg.seed=1`); there is no seed distribution, so
   the 45.85 pp and 23.28 pp gaps are single draws, not confidence intervals.
2. A2 changes the weights *and* the dynamics: it is a reservoir control, not a
   dynamics-only ablation. A3 keeps trained input weights but still runs a leaky
   integrator `z <- z + alpha(tanh(drive) - z)`, so 59.23 % includes a learned temporal
   filter of the input drive — it is not a static feed-forward probe of the input.
3. A3 is a state-dict-level (off-manifold) intervention on the trained model: it deletes
   four trained matrices at once and never runs a forward pass the training distribution
   produced weights for.
4. Early stopping uses VAL, which is speaker-overlapping with FIT, and A1's features are
   FIT counts, i.e. in-sample for the model (same caveat as Result 1). The protocol
   assumes the VAL-optimal readout epoch transfers to TEST.
5. Linear readout only; a nonlinear readout could score A2/A3 differently. No per-class
   breakdown for any arm.
6. Nothing here says *which* dynamical feature the readout uses; A1/A3 only bound how
   much of TEST accuracy survives removal.

## Result 3 (mechanism) — are the neurons individually necessary?

Protocol tag: **confirmed** for the B2 baseline (1868/2264, 20 epochs, seed 0);
B1 and the B2 ranking are **descriptive on this checkpoint** (single seed). B1
(`V3/run_ablation.py fingerprints`, CPU) uses FIT (n = 7340) spike counts: plug-in MI
[bits] between an 8-quantile-binned count and the class (K = 20, class entropy
4.322 bits; plug-in bias ~0.013 bits), plus mean rate. B2 (`V3/run_ablation.py knockout`,
GPU) silences one neuron at a time, `theta_i <- theta_i - 1e9`, and re-evaluates the full
official TEST (n = 2264). Because `theta` is **added** to the potential
(`V3/v3/model.py:233`), silencing is a large negative offset; the brief's `+1e9` would pin
the neuron above threshold (verified: 0 spikes when silenced, every-bin firing with
`+1e9` — `V3/tests/test_ablation.py`). Each knockout starts from a pristine `theta`, so
lesions do not accumulate. The ranking is TEST-accuracy selected (**disclosed**).

B1 — per-neuron class information (FIT):

| statistic | value |
|-----------|-------|
| MI min / median / max | 0.084 / 0.446 / 0.742 bits |
| MI mean ± std (CV) | 0.455 ± 0.159 (0.349) |
| MI IQR | 0.336 – 0.603 bits |
| total MI / top-5 share | 29.14 bits / 12.2 % |
| effective informative neurons `1 / sum(p_i^2)` | 57.0 / 64 |
| neurons below 0.05 bits | 0 |
| rate min / median / max | 11.1 / 17.3 / 29.9 Hz |
| split-half Spearman (even vs odd FIT) | 0.987 |
| FIT-vs-TEST MI Spearman (descriptive) | 0.859 |

| rank | top 5 (FIT MI) | MI bits | rate Hz | bottom 5 | MI bits | rate Hz |
|------|----------------|---------|---------|----------|---------|---------|
| 1 | neuron 38 | 0.742 | 13.6 | neuron 14 | 0.084 | 17.5 |
| 2 | neuron 6 | 0.721 | 22.0 | neuron 35 | 0.180 | 14.1 |
| 3 | neuron 34 | 0.711 | 17.4 | neuron 56 | 0.204 | 15.9 |
| 4 | neuron 21 | 0.702 | 13.0 | neuron 22 | 0.217 | 13.3 |
| 5 | neuron 51 | 0.687 | 29.9 | neuron 5 | 0.248 | 12.7 |

The distribution is **flat, not sparse**: every neuron carries class information (none
below 0.05 bits), the top 5 hold only 12 % of the total, and the effective informative
count is 57 of 64. In absolute terms each neuron is weak (0.08–0.74 bits against a
4.32-bit class entropy), so the 82.51 % is a population code, not a few tuned detectors.

B2 — single-neuron knockout, official TEST (baseline re-measured 1868/2264 = 82.51 %):

| k silenced (sorted by individual drop) | TEST | acc | drop |
|---|---|---|---|
| 1 | 1767/2264 | 78.05 % | 4.46 pp |
| 2 | 1701/2264 | 75.13 % | 7.38 pp |
| 4 | 1572/2264 | 69.43 % | 13.07 pp |
| 8 | 1365/2264 | 60.29 % | 22.22 pp |
| 16 | 1084/2264 | 47.88 % | 34.63 pp |
| 32 | 872/2264 | 38.52 % | 43.99 pp |
| 64 | 103/2264 | 4.55 % | 77.96 pp |

- Single-neuron drops: max 4.46 pp (neuron 33, 1767/2264), min −1.24 pp (neuron 14,
  1896/2264 — deleting it **improves** TEST), mean 1.02 ± 1.07 pp.
- 54/64 deletions hurt, 8 improve, 2 change nothing; 28/64 neurons individually cost
  more than 1 pp (one neuron = 0.044 pp, so these are 23+ sample differences).
- Top drops: neurons 33 (4.46), 39 (3.53), 35 (3.14), 32 (2.92), 21 (2.69 pp).
  Improvements: neurons 14 (−1.24), 51 (−0.49), 49 (−0.27), 41 (−0.22), 37 (−0.18 pp).
- The B1 ranking does **not** predict knockout importance: Spearman(MI_FIT, drop) =
  +0.200, Spearman(rate_FIT, drop) = +0.175. Only one neuron (21) is in both top-5 sets;
  B1's #5 (neuron 51) is *helped* by deletion, and B1's bottom-1 (neuron 14) gains the
  most from deletion. Class information and causal necessity are different quantities.
- Necessity is distributed: no single neuron is worth more than 4.5 pp, yet the top 8
  cost 22 pp and the top 16 cost 35 pp. Function is redundant across the population, not
  concentrated in a sparse "grandmother" set.
- Cost: 18.5 s per knockout evaluation, 21.6 min total (plus a 23.0 s baseline),
  355 MB peak torch allocation.

What this does **not** show:

1. B2's ranking and cumulative curve are **TEST-selected**, so the sorted curve is
   optimistic for held-out selection; the 8 "improvements" (up to 1.24 pp) are
   same-sample TEST descriptions, not validated gains.
2. A knockout is a dynamical lesion, not removal of a private code: a silenced neuron
   also stops contributing its spikes to the population signal `h` and the recurrent
   drive `c`, so downstream activity changes; the drop is not "the class information in
   this neuron was necessary".
3. `theta_i = -1e9` is far outside the trained regime; the intervention is large and
   off-manifold, so no small-perturbation or linear-response reading is available.
4. MI is a binned plug-in estimate (bias ~0.013 bits) of *any* relation between counts
   and class, not of linear decodability by `counts @ w_cls`; the readout's fixed weights
   and population redundancy can make a high-MI neuron dispensable.
5. One checkpoint/seed, no retraining or compensation after the lesion, no
   retrained-readout leave-one-out, no per-class drop analysis.
6. B1 ranks on FIT while B2 ranks on TEST; a strict ranking-to-causality test would need
   the ranking selected on held-out data (not run).

## Reproduce (mechanism ablations)

```
python V3/run_ablation.py freeze --arms a1,a2,a3   # A1 CPU (cached counts); A2/A3 ~3.6 min GPU
python V3/run_ablation.py fingerprints             # B1, CPU <1 s
python V3/run_ablation.py knockout                 # B2, 21.6 min GPU, 355 MB peak
python -m pytest V3/tests -q                       # 124 passed
python V3/run_smoke_test.py                        # all smoke checks passed
```

Outputs (git-ignored, `V3/results/analysis/`): `ablation_freeze.json`,
`ablation_fingerprints.json`, `ablation_fingerprints.npz`, `ablation_knockout.json`,
`ablation_readout_a{1,2,3}.npz`, `ablation_counts_a{2,3}_{fit,val,test}.npz`.
Total GPU time for both parts: ~25.5 min (A2+A3 ~3.6 min, B2 21.6 min); peak torch
allocation 461 MB, device total stayed at 1.54 GiB of 6 GiB.

