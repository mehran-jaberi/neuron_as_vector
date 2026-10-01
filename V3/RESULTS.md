# V3 results

Measured numbers only. Every value below comes from the executed
`V3_SHD_experiment.ipynb` (19/19 code cells, zero errors, 712 s wall; executed
top-to-bottom with a real Jupyter kernel by `V3/tools/run_notebook.py`,
`V3_QUICK` unset, `V3_OVERRIDES='{"epochs": 3}'`) and from the artifacts that run
wrote under `V3/results/`, `V3/checkpoints/` and `V3/figures/`.

Those three directories are ignored by the repository's root `.gitignore` (the
repository's convention), so the numbers are reproduced in full here and in the
notebook itself.

The **committed execution uses a 3-epoch schedule** so the turnaround stays short.
The YAML default is `epochs: 20`; that full run is deferred and is one command
away (see §8).

**Configuration** (`configs/v3_default.yaml`, `seed = 0`, **one seed — no error bar**):

| | vector | scalar baseline |
|---|---|---|
| neurons `N` | 64 | 64 |
| state dimension `D` | **1000** | **1** |
| trainable parameters | 2,025,440 | 6,335 |
| epochs / batch | 3 / 128 | 3 / 128 |
| optimizer / schedule | Adam, lr 1e-3, cosine, 3% warmup, clip 1.0 | same |
| loss | CE + 1e-4·(rate − 10 Hz)² | same |
| data | SHD, FIT 7340 / VAL 816 (stratified split of the train file) | same |
| temporal resolution | 250 × 4 ms = 1000 ms, binary input | same |
| surrogate | fast sigmoid γu/(1+β\|u\|), β=4, γ=1 | same |
| device / dtype | CUDA (RTX 3060 Laptop, 6 GiB) / fp16 state, fp32 masters + loss | same |

## 1. The headline

```
PRIMARY RESULT — official SHD test set, 2264 samples, frozen parameters

VECTOR  D=1000   correct:  1625
VECTOR  D=1000   total:    2264
VECTOR  D=1000   accuracy: 71.78%

SCALAR  D=1      correct:   601
SCALAR  D=1      total:    2264
SCALAR  D=1      accuracy: 26.55%

vector − scalar = +45.23 percentage points
```

Reference points (not measured in this run): a plain recurrent SHD baseline is
usually quoted around 71%; the repository's own V2 LIF model (N=256, T=1400 ms,
2 ms bins, 20 epochs) measured **59.98%** on the same test file
(`checkpoints/sweep_l2_0.pt`, `results/lif_baseline_selection.json`).

## 2. Training

Vector (`D=1000`, 2,025,440 parameters):

| epoch | train loss | train CE | train acc | val loss | val acc | firing rate | spikes/sample | epoch time |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.7030 | 1.6665 | 45.33% | 0.9983 | 67.16% | 19.21 Hz | 1229 | 160.7 s |
| 2 | 0.7629 | 0.7164 | 77.26% | 0.6061 | 83.82% | 23.65 Hz | 1514 | 136.6 s |
| 3 | 0.4701 | 0.4268 | 88.01% | 0.4903 | 86.89% | 24.72 Hz | 1582 | 131.5 s |

total 445.5 s; FIT accuracy over the whole training split 90.35%; best-validation
epoch = 3 (val 0.8689). The first epoch is slower because it also fills the in-RAM
event cache.

Scalar (`D=1`, 6,335 parameters): train acc 12.90% → 20.42% → 23.66%, val acc
18.63% → 24.88% → 24.63%, 8.73 → 11.01 Hz, epochs 61.3 / 40.8 / 41.6 s, total
151.0 s; FIT accuracy 24.29%.

## 3. Test

The official SHD test set (2264 samples, speakers `{0..11}` — including speakers 4
and 5, which never occur in the training file). Parameters frozen, gradients
disabled, every sample evaluated, no early stopping and no selection on test.

| | vector `D=1000` | scalar `D=1` |
|---|---|---|
| correct | **1625** | 601 |
| total | 2264 | 2264 |
| accuracy | **71.78%** | 26.55% |
| errors | 639 | 1663 |
| loss (CE + rate reg) | 0.893 | 2.303 |
| mean / median / p90 / max rate | 24.09 / 21.55 / 38.66 / 95.45 Hz | 10.72 / 5.21 / 24.98 / 66.23 Hz |
| silent neurons (of 64) | 0 | 10 |
| spikes per sample | 1542 | 686 |
| per-class accuracy min / median / max | 37.0% / 72.0% / 100.0% | 0.0% / 25.4% / 68.8% |
| classes with recall < 10% | **0** | 5 |
| English digits 0–9 / German 10–19 | 74.91% / 68.42% | 23.18% / 29.57% |
| peak CUDA memory | 1283 MiB allocated / 1330 MiB reserved | 144 MiB |
| test pass wall time | 18.8 s | 10.3 s |

Both models heal their firing rates during training (calibration starts at 10.5 Hz;
the vector population settles at ~24 Hz, well below saturation, with no silent
neurons and a 95 Hz maximum).

Predictions, logits, the confusion matrix and per-class accuracy are saved under
`V3/results/<tag>/`; the figures are in `V3/figures/<tag>/`.

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
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`): **1108 MiB allocated,
  1148 MiB reserved**; `nvidia-smi` reports **~1.7 GiB** in use during the
  notebook run (the rest is the desktop's own GPU usage on this WDDM machine).
* Throughput: **2.2 s per training batch** measured in isolation at
  `N=64, D=1000, T=250, batch=128` (≈ 58 samples/s), i.e. ≈ 2.1 min per epoch of
  7340 training samples. Batch size barely matters (63 → 78 samples/s from
  batch 32 → 256), so the Python-level 250-step loop, not the FLOPs, is the
  bottleneck — see `TODO.md`. The notebook run additionally pays host-side event
  binning, giving ≈ 3.5 s per batch in the first epoch.
* The D=1 scalar baseline is roughly twice as fast per batch: its per-neuron state
  is one number, so the rank-R mixing and the `D×D` emission disappear.

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
