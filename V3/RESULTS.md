# V3 results

Measured numbers only. Every value below comes from the executed
`V3_SHD_experiment.ipynb` (20 code cells, zero errors; executed top-to-bottom
with a real Jupyter kernel by `V3/tools/run_notebook.py`, `V3_QUICK` unset, the
YAML default `epochs: 20`) and from the artifacts that run wrote under
`V3/results/`, `V3/checkpoints/` and `V3/figures/`. The same run is recorded in
the registry as `V3/results/runs.csv` row `2026-10-02_14-31-15` and under
`V3/results/2026-10-02_14-31-15/`.

Those three directories are ignored by the repository's root `.gitignore` (the
repository's convention), so the numbers are reproduced in full here and in the
notebook itself.

The **default temporal discretization is now 2 ms** (500 steps, $\alpha = 0.1$).
The previous 4 ms reference (250 steps, $\alpha = 0.2$) is retained for comparison
(see the note at the end of §1).

**Configuration** (`configs/v3_default.yaml`, `seed = 0`, **one seed — no error bar**):

| | vector | scalar baseline |
|---|---|---|
| neurons `N` | 64 | 64 |
| state dimension `D` | **1000** | **1** |
| trainable parameters | 2,025,440 | 6,335 |
| epochs / batch | 20 / 128 | 20 / 128 |
| optimizer / schedule | Adam, lr 1e-3, cosine, 3% warmup, clip 1.0 | same |
| loss | CE + 1e-4·(rate − 10 Hz)² | same |
| data | SHD, FIT 7340 / VAL 816 (stratified split of the train file) | same |
| temporal resolution | **500 × 2 ms = 1000 ms**, binary input (dt = 2 ms, α = dt/τ = 0.1) | same |
| surrogate | fast sigmoid γu/(1+β\|u\|), β=4, γ=1 | same |
| device / dtype | CUDA (RTX 3060 Laptop, 6 GiB) / fp16 state, fp32 masters + loss | same |

## 1. The headline

```
PRIMARY RESULT — official SHD test set, 2264 samples, frozen parameters

VECTOR  D=1000   correct:  1868
VECTOR  D=1000   total:    2264
VECTOR  D=1000   accuracy: 82.51%

SCALAR  D=1     correct:  1119
SCALAR  D=1     total:    2264
SCALAR  D=1     accuracy: 49.43%

vector − scalar = +33.08 percentage points
```

Reference points: a plain recurrent SHD baseline is usually quoted around 71%;
the repository's own V2 LIF model (N=256, T=1400 ms, 2 ms bins, 20 epochs)
measured **59.98%** on the same test file (`checkpoints/sweep_l2_0.pt`,
`results/lif_baseline_selection.json`).  The previous V3 reference — the same
$N=64$, $D=1000$ population at the **4 ms** discretization ($T=250$, α = 0.2),
20 epochs — measured **79.81%** (1807/2264). The 2 ms run is **+2.70 percentage
points** above that, i.e. halving the time step and the leak `alpha` improved the
benchmark rather than merely changing its resolution.

## 2. Training

Vector (`D=1000`, 2,025,440 parameters, 20 epochs, 2 ms):

| epoch | train loss | train CE | train acc | val loss | val acc | rate | spikes/sample | epoch time |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.9840 | 1.9180 | 37.26% | 1.2104 | 61.89% | 24.55 Hz | 1571 | 220.9 s |
| 2 | 0.9246 | 0.8527 | 70.53% | 0.7634 | 75.61% | 27.60 Hz | 1766 | 204.0 s |
| 3 | 0.6011 | 0.5402 | 81.06% | 0.5042 | 84.44% | 27.34 Hz | 1750 | 204.6 s |
| 4 | 0.4743 | 0.4249 | 85.48% | 0.4043 | 88.11% | 25.68 Hz | 1643 | 204.2 s |
| 5 | 0.3424 | 0.2998 | 90.18% | 0.4536 | 86.52% | 25.26 Hz | 1617 | 204.3 s |
| 6 | 0.3052 | 0.2663 | 91.13% | 0.4700 | 85.05% | 23.91 Hz | 1530 | 204.0 s |
| 7 | 0.2308 | 0.1948 | 93.56% | 0.3842 | 87.50% | 23.54 Hz | 1507 | 204.2 s |
| 8 | 0.1876 | 0.1538 | 94.84% | 0.2222 | 94.36% | 23.28 Hz | 1490 | 204.6 s |
| 9 | 0.1575 | 0.1293 | 95.59% | 0.2409 | 92.40% | 22.66 Hz | 1450 | 205.8 s |
| 10 | 0.1295 | 0.1020 | 96.54% | 0.2330 | 93.38% | 22.18 Hz | 1420 | 203.7 s |
| 11 | 0.0972 | 0.0747 | 97.74% | 0.1522 | 95.71% | 21.56 Hz | 1380 | 204.2 s |
| 12 | 0.0704 | 0.0516 | 98.60% | 0.1589 | 95.22% | 21.00 Hz | 1344 | 204.0 s |
| 13 | 0.0499 | 0.0330 | 99.21% | 0.1358 | 96.20% | 20.70 Hz | 1325 | 203.7 s |
| 14 | 0.0381 | 0.0240 | 99.51% | 0.1221 | 96.08% | 19.84 Hz | 1270 | 203.9 s |
| 15 | 0.0282 | 0.0161 | 99.70% | 0.1136 | 96.08% | 19.17 Hz | 1227 | 203.7 s |
| **16** | 0.0208 | 0.0106 | 99.90% | **0.1021** | **97.67%** | 18.66 Hz | 1195 | 207.0 s |
| 17 | 0.0173 | 0.0085 | 99.93% | 0.1200 | 96.45% | 18.07 Hz | 1157 | 201.7 s |
| 18 | 0.0152 | 0.0067 | 99.99% | 0.1162 | 96.57% | 17.98 Hz | 1151 | 203.9 s |
| 19 | 0.0137 | 0.0057 | 100.00% | 0.1135 | 96.69% | 17.77 Hz | 1137 | 204.0 s |
| 20 | 0.0135 | 0.0057 | 100.00% | 0.1129 | 96.32% | 17.71 Hz | 1133 | 204.4 s |

total 4243.5 s (≈ 204 s/epoch; the first epoch is slower because it also fills
the in-RAM event cache); FIT accuracy over the whole training split **100.00%**
(loss 0.0134); best-validation epoch = **16** (val 0.9767). The population closes
the FIT/VAL gap by epoch 8 and then mildly overfits (train 100% vs val 96.3% at
epoch 20); the final (not best-val) weights are what §3 evaluates.

Scalar (`D=1`, 6,335 parameters, 20 epochs, 2 ms): FIT accuracy **58.01%**,
VAL **53.68%**, 1380.6 s total (≈ 69 s/epoch).

## 3. Test

The official SHD test set (2264 samples, speakers `{0..11}` — including speakers 4
and 5, which never occur in the training file). Parameters frozen, gradients
disabled, every sample evaluated, no early stopping and no selection on test.

| | vector `D=1000` | scalar `D=1` |
|---|---|---|
| correct | **1868** | **1119** |
| total | 2264 | 2264 |
| accuracy | **82.51%** | **49.43%** |
| errors | 396 | 1145 |
| loss (CE + rate reg) | 0.7728 | not recorded |
| mean / median / p90 / max rate | 18.22 / 17.71 / 25.10 / 37.02 Hz | not recorded |
| silent neurons (of 64) | 0 | not recorded |
| spikes per sample | 1166 | not recorded |
| per-class accuracy min / median / max | 38.8% / 88.6% / 100.0% | not recorded |
| classes with recall < 10% | **0** | not recorded |
| English digits 0–9 / German 10–19 | 90.31% / 75.08% | not recorded |
| peak CUDA memory | 2284 MiB allocated / 2328 MiB reserved | not recorded |
| test pass wall time | 24.1 s | not recorded |

The vector population arrives at a healthy, non-saturated regime (18.2 Hz mean,
37.0 Hz maximum, no silent neurons) and its errors are concentrated in a few
German-digit confusions. The largest off-diagonal confusions (true → predicted,
count) are: `13 → 5` (49), `19 → 9` (32), `17 → 3` (26), `2 → 0` (21),
`17 → 7` (19), `12 → 13` (17); the weakest class is 17 (recall 38.8%) and the
strongest are 4 and 18 (100%).

Predictions, logits, the confusion matrix and per-class accuracy are saved under
`V3/results/<tag>/` and in the run directory `V3/results/2026-10-02_14-31-15/`;
the figures are in `V3/figures/<tag>/`.

## 4. Activity figures

Written for the vector model: training curves, test confusion matrix, per-class
accuracy bar chart, and input-raster / spike-raster / population-rate panels for
three test samples and one FIT sample.

## 5. Precision actually observed

| tensor / operation | dtype |
|---|---|
| state `z`, spikes, `a`, all large activations in the step | float16 — verified in notebook §5: `z` is `torch.float16` before and after the update |
| weights as used in the matmuls | float16 (cast from the fp32 masters in every segment) |
| master weights held by the optimizer | float32 — verified: `['torch.float32']` |
| logits / loss | float32 — verified |
| optimizer state | float32 (Adam) |
| pre-spike potential and threshold comparison | float32 inside the surrogate `autograd.Function` |

## 6. Resource use

* Peak CUDA memory over a **full training epoch** (probe at batch 128 with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`): **2284 MiB allocated,
  2328 MiB reserved** at the 2 ms discretization (`N=64, D=1000, T=500`).
* Throughput at 2 ms: **4243.5 s for 20 vector epochs** (≈ 204 s/epoch, ≈ 3.5 s
  per training batch of 128, ≈ 36 samples/s) and **24.1 s for the full 2264-sample
  test pass**. Halving the time step doubles `T` (250 → 500), which is why the
  per-epoch time and memory are roughly double the 4 ms reference
  (`N=64, D=1000, T=250, batch=128`: 1108 MiB allocated / 1148 MiB reserved, ≈ 2.2 s
  per batch). The Python-level per-step loop, not the FLOPs, is the bottleneck —
  see `TODO.md`.
* The D=1 scalar baseline is much faster per batch (1380.6 s for 20 epochs,
  ≈ 69 s/epoch, ≈ 3× faster per epoch): its per-neuron state is one number, so the
  rank-R mixing and the `D×D` emission disappear.

## 7. What failed or had to be changed

All of these were found and fixed during development; they are recorded because
each is a real trap for this architecture.

0. **Un-checkpointed diagnostic → 150× slowdown (the most expensive mistake).**
   The notebook's structural sanity check originally ran a full 250-step
   `model._segment` call with **gradients enabled and no checkpointing**, and kept
   the result in the notebook namespace. At `D=1000` that single call allocates
   **8431 MiB** (measured; reserved 8494 MiB) on a 6 GiB device, so NVIDIA's WDDM
   driver silently paged GPU memory to host RAM and every subsequent training step
   ran at **333 s/batch instead of 2.2 s/batch** — an ETA of 5 h 16 m for the first
   epoch. The sanity checks now run under `torch.no_grad()`, free their tensors and
   print the live footprint (20 MiB allocated afterwards).
   The allocator setting is *not* the cause: a plain script trains at 2.2 s/batch
   both with and without `expandable_segments`.
1. **Inverted threshold sign.** `theta` enters the potential additively, so
   `s = 1[raw + theta > 0]` fires *more* when `theta` is positive. The first
   calibration set `theta` to the positive (1−p) quantile and produced a **223 Hz**
   population (89% of steps spiking). Correct initialisation is
   `theta_i = −quantile(raw_i, 1−p)`.
2. **Non-converging threshold iteration.** Because the measured potentials already
   contain the current offset, the fixed-point update must *accumulate*
   (`theta -= quantile(pot)`); using `theta = −quantile(pot)` oscillates with
   period 2. Three accumulating iterations give 10.53 Hz against a 10 Hz target
   (min 9.03, max 12.93 Hz).
3. **Unscaled population signal → saturation.** With `h(t) = Σ_j s_j(t) z_j(t)` the
   drive grows like √N and `tanh` saturates: measured `|z| = 0.807` mean with a
   0.999 maximum. Dividing the population signal by √N fixed it (`|z| ≈ 0.1`).
4. **Random readout init.** With a spike-count readout the logit scale grows with
   the firing rate, so `W_cls ~ N(0, 1/√N)` gave an initial cross-entropy of
   **28.4** (against `ln 20 = 3.0`). Zero-initialising `W_cls` gives exactly 3.00.
5. **Matrix orientation.** Every parameter is used as `x @ param`, and `w_rec` is
   indexed `[pre, post]`. A transposed `w_up` raised a shape error immediately
   (caught by the smoke test).
6. **Broadcasting.** `gate * a[:, k]` needs `a[:, k].unsqueeze(1)` to broadcast over
   the batch axis.
7. **CUDA allocator fragmentation.** The 250-step BPTT graph allocates and frees
   many differently-sized tensors. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   (set in `v3/__init__.py`, before the first CUDA allocation) keeps the reserved
   pool at the true peak instead of letting it drift upwards. It is a safety
   margin, not a speed fix — see finding 0.
8. **Test-time activity figure indexed the wrong store.** A helper built the
   "FIT sample" raster by indexing the *test* store with FIT indices. Replaced by
   `save_activity_figures(store, split, ...)`, always called with a matched
   store/split pair.
9. **Harness, not science:** the headless notebook executor originally treated a
   60 s silence from the kernel as fatal. `ipykernel` buffers a long cell's
   stdout/stderr until the cell finishes, so a 20-epoch training cell looks silent
   for minutes. The executor now tolerates empty polls and prints a heartbeat.
10. **Transient `CUDA driver error: device not ready`** on one launch (immediately
    after a previous run's teardown). Relaunching the identical run succeeded; no
    code change was needed.

## 8. Reproducing these numbers

```powershell
.venv\Scripts\python.exe V3\tools\build_notebook.py       # assemble the .ipynb
.venv\Scripts\python.exe V3\tools\run_notebook.py         # execute it top to bottom
# identical results through the CLI (same code path):
.venv\Scripts\python.exe V3\run_experiment.py --variant both
```
