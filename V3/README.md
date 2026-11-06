# V3 — a population of genuine $D$-dimensional vector-valued neurons

> **Question.** How much of the official SHD test set can a population of
> $D = 1000$ **vector-valued** spiking neurons classify correctly?

> **Answer (executed, 20 epochs, 2 ms, single seed).** **82.51%** — 1868 of the
> 2264 official SHD test samples. The identical population with scalar neurons
> ($D=1$, same 64 neurons, same training) reaches **49.43%**, but that is an
> **untuned, size-matched control (ablation)**, not a competitive baseline. See
> [§8](#8-results) and `RESULTS.md`.
>
> **Current experimental reference.** $N=64$, $D=1000$, 2,025,440 parameters,
> **2 ms** discretization ($T=500$, $\alpha=0.1$), 20 epochs: **82.51%**
> (1868/2264), recorded in the executed `V3_SHD_experiment.ipynb` and in the
> registry run `2026-10-02_14-31-15`. Imprecision screening is compared against
> this 82.51% reference; see [§11](#11-controlled-state-imprecision--the-run-registry).

This folder is an independent experiment. It is **not** an extension of V2 and
does not import, modify or reproduce any of the V2 architecture (the old root
`src/`, `scripts/`, `configs/` now live under `archive/v2/` and are untouched). It
is also not a rerun of the early Brian prototype (`brian implementation/`, removed
from the tree in 2026-10 — a *fixed* high-dimensional dynamical system with a
trained readout). Here the vector dynamics themselves are trained end-to-end.

> **Mechanism, measured on the frozen 20-epoch checkpoint** (`ANALYSIS.md`): a
> linear probe on the raw state does not beat the spike counts (80.83% vs
> 82.51%, Δ = −1.68 pp, 95% CI [−3.00, −0.31]), so the width is not extra readout
> bandwidth. It is a learned, low-dimensional dynamical substrate: participation
> ratio 6.64 of 64,000 dimensions population-wide and 4.21 per neuron of 1,000
> (scalar control 1.71), refitting only the readout on the trained dynamics still
> gives 82.02%, random dynamics 36.66%, recurrence removed 59.23%. Neurons are
> distributed and redundant: silencing any single neuron costs 1.02 ± 1.07 pp on
> average, worst 4.46 pp.
>
> **Throughput** (`PERF.md`, 1 epoch, N=64, D=1000, T=500, 2 ms): 196.72 s warm at
> 94.6% GPU utilisation. `--override binned_cache=true` gives −3.2% with
> bit-identical tensors; `--resume <ckpt>` continues a run and writes a
> `<tag>_resumed` row; `torch.compile` is unavailable on this Windows build
> (`TritonMissing`).

---

## 1. The hypothesis (the "C" version of neuron-as-vector)

A neuron's internal computational state **is** a high-dimensional vector. It is
not a scalar unit that receives a vector embedding afterwards, and it is not a
vector of statistics extracted from a simulated scalar neuron.

For neuron $i$ of a population of $N$ neurons:

$$\mathbf z_i(t) \in \mathbb R^{D}, \qquad D = 1000, \qquad Z(t) \in \mathbb R^{N \times D}.$$

Every row of $Z$ is one neuron, and *every one of its $D$ coordinates is part of
the neuron's dynamical state*: it is projected from the input, it is mixed with
the other coordinates on every step, it is emitted to the rest of the population
when the neuron spikes, and it generates the spike through a learned linear
readout of itself.

**What is explicitly *not* done**

* no `scalar neuron -> simulate -> features -> PCA/embedding` pipeline,
* no `nn.Linear` or any other layer that turns a scalar neuron into a vector,
* no 1000 conventional hidden neurons relabelled as one vector neuron
  (a "neuron" here is one row of the state tensor, not 1000 rows),
* no fixed dynamics with a trained readout (the Brian experiment's limitation).

---

## 2. The model

**One neuron** owns the state $\mathbf z_i(t)\in\mathbb R^D$, a spike output
$s_i(t)\in\{0,1\}$, and a trainable input gate $\mathbf g_i$, bias
$\mathbf b_i$, spike-readout vector $\mathbf w^{out}_i$ and threshold offset
$\theta_i$. The $D$ coordinates of one neuron are coupled by the shared
rank-$R$ operator $W_{down}W_{up}$ ($R = 64 \ll D$).

Notation: $T$ dynamical steps, $\Delta t$ = one bin, $\tau$ = leak time constant,
$\alpha = \Delta t/\tau$, $\mathbf x(t)\in\mathbb R^{700}$ the binned SHD input,
$W_{in}\in\mathbb R^{700\times D}$ a projection **shared** by all neurons,
$W_{emit}\in\mathbb R^{D\times D}$ and $W_{rec}\in\mathbb R^{N\times N}$ shared.

$$
\mathbf a(t) = \mathbf x(t)\,W_{in} \quad\text{(one matmul over the whole sequence)}
$$

then for every $t$:

$$
\begin{aligned}
o_i(t) &= \langle \mathbf z_i(t),\, \mathbf w^{out}_i\rangle + \theta_i
   &&\text{scalar pre-spike potential}\\[2pt]
s_i(t) &= \mathbf 1\!\left[o_i(t) > 0\right]
   &&\text{Heaviside spike, fast-sigmoid surrogate gradient}\\[6pt]
\mathbf h(t) &= \frac{1}{\sqrt N}\Big(\sum_j s_j(t)\,\mathbf z_j(t)\Big) W_{emit}
   &&\text{population signal } \in \mathbb R^{D}\\[2pt]
c_i(t) &= \sum_j W_{rec}[j,i]\, s_j(t)
   &&\text{scalar recurrent drive}\\[6pt]
\mathbf d_i(t) &= \mathbf g_i \odot \mathbf a(t) \;+\; c_i(t)\,\mathbf h(t) \;+\; \mathbf b_i
   &&\text{vector drive } \in \mathbb R^{D}\\[2pt]
\mathbf m_i(t) &= \big(\mathbf z_i(t)\,W_{down}\big) W_{up}
   &&\text{intra-neuron mixing, rank } R\\[6pt]
\mathbf z_i(t{+}1) &= (1-\alpha)\,\mathbf z_i(t) \;+\; \alpha\,
   \tanh\!\big(\mathbf m_i(t) + \mathbf d_i(t)\big)
   &&\text{leaky integration of the vector state}
\end{aligned}
$$

**Readout / classifier** (spike counts, 20 SHD classes):

$$
\text{logits} \;=\; \Big(\sum_t \mathbf s(t)\Big) W_{cls}
\;\in\; \mathbb R^{20},
\qquad
\hat y = \arg\max \text{logits}.
$$

### Design choices, with reasons

| choice | reason |
|---|---|
| $\tanh$ on the vector argument | hard-bounds the state ($\lvert z\rvert \le 1$), which is what makes pure `fp16` safe. It is applied to $\mathbf m_i+\mathbf d_i$, so it does **not** decouple the $D$ coordinates; the coupling comes from $W_{down}W_{up}$, $W_{in}$ and $\mathbf h$. Any pointwise nonlinearity would do — it is not part of the hypothesis. |
| state leak $\alpha = \Delta t/\tau$, $\tau = 20$ ms | standard leaky-integrator dynamics; $\Delta t$ = one 2 ms bin, so each bin is one dynamical step |
| $1/\sqrt N$ on the population signal | mean-field scaling: a sum of $N$ unit contributions has magnitude $\sim\sqrt N$; without it the drive (and the firing rate) grows with $N$ |
| shared $W_{in}$, shared rank-$R$ mixing, shared $W_{emit}$ | the naive per-neuron $D\times D$ recurrent matrix would cost $N D^2 \approx 6.4\times10^7$ parameters *per neuron*. Sharing keeps the **state** genuinely $D$-dimensional while making the parameters affordable. |
| `sum` readout, not `mean` | a mean readout shrinks the logit scale by $T$; the cross-entropy is then cheapest to reduce by raising firing rates, which saturates the population. Spike counts keep the logit scale in a sane range. |
| zero-initialised $W_{cls}$ | with a spike-count readout the logit scale grows with the firing rate, so a random readout init starts far from the loss basin. Zero init gives exactly $-\ln 20 \approx 3.0$ initial cross-entropy. |
| per-neuron threshold initialisation, then trained | `theta` enters additively, so a neuron fires when `raw_i(t) > -theta_i`; `theta_i` is initialised to the **negated** empirical $(1-p)$ quantile of the neuron's raw pre-spike potential ($p = \text{rate}\cdot\Delta t$) so every neuron starts at the target firing rate. It stays trainable; why it matters is documented in `RESULTS.md`. |
| firing-rate regulariser ($\lambda = 10^{-4}$, target 10 Hz) | guards against the rate/saturation feedback loop (higher rate $\to$ larger population signal $\to$ larger drive $\to$ higher rate) |

### What $N$ and $D$ mean

* $N$ = the number of **neurons** (rows of the state tensor). Configurable.
* $D$ = the number of coordinates of **one** neuron's internal state (width of
  each row). $D = 1000$ for the primary experiment.

Both are ordinary keys in `configs/v3_default.yaml`; nothing in the model, data
loader or notebook hard-codes them. Setting `state_dim: 1` gives the scalar
control population *through the identical code path*.

### Emission parameterisation: `emission.mode`

`emission.mode: full` (default) keeps the historical shared $D\times D$ matrix
$W_{emit}$ (1,000,000 parameters at $D = 1000$). `emission.mode: diagonal`
replaces it by a per-coordinate trainable gain $\mathbf g \in \mathbb R^D$
(1,000 parameters, initialised to exactly $\mathbf 1$ — the diagonal analogue of
the $I + 0.02\cdot\text{noise}$ init):

$$
\mathbf h(t) = \frac{1}{\sqrt N}\Big(\sum_j s_j(t)\,\mathbf z_j(t)\Big) \odot \mathbf g
$$

The spike still carries a $D$-vector and $\mathbf h(t) \in \mathbb R^D$; the gate,
recurrence and state dynamics are untouched. This is a **parameterisation/constraint
study, not a new hypothesis**: it removes 999,000 parameters (2,025,440 → 1,026,440 at
$N = 64$, $D = 1000$). Measured at 5 epochs, $N=64$: 1748 = 77.21 % against the
1724 = 76.15 % baseline, i.e. inside the ±0.87 pp noise band — the dense mix is
not load-bearing (`RESULTS.md` §4.1). Switch it with
`--override emission.mode=diagonal`. The two modes have different parameter sets, so
their checkpoints are not interchangeable under the default strict `load_state_dict`.

### Scalar input current: `input_current.enabled`

Design A replaces the shared $700\times1000$ projection and the per-neuron gate
with **one scalar current per neuron**, $a_{i,t}=\langle\mathbf x(t),\mathbf g_i\rangle$,
$\mathbf g_i\in\mathbb R^{700}$ (70,000 parameters), added to the spike
potential $o_i(t)$:

$$
o_i(t) = \langle \mathbf z_i(t), \mathbf w^{out}_i\rangle + \theta_i + a_{i,t}
$$

The vector state then never sees the input; its only drive is its own bias and
the recurrent population signal. Two requirements:

- `input_current.bias_init_std` must be **> 0** (e.g. `0.5`). With `bias = 0`
  and $z_0 = 0$ every forcing term is proportional to $\mathbf z$, so the state
  has a globally stable fixed point at zero and the model degenerates into a
  threshold-only classifier ($|\mathbf z| = 0$ for all 500 steps, chance
  accuracy).
- It is **mutually exclusive with `input_delay_line`** (the delay line widens
  the input from $C$ to $KC$; the current needs the original $C$). Validation
  rejects the pair.

Measured (`RESULTS.md` §9.1): A′ at $N=100$ with diagonal emission, 20 epochs,
raw input — **1937 = 85.56 %** with 411,100 parameters.

### The other mechanism switches

`adaptation` (M1), `state_dynamics` (`trainable_alpha`, `learnable_z0`, M2),
`input_delays` (M4), `adlif` (M8), `two_stage` (M12), `attention` (M11b),
`timing.silence_tail_ms` (M13), `surrogate` (`slayer`), `optimizer`,
`augmentation.event_blend` (M7) and the five raw-event operations all live in
the same config; `--help-overrides` prints every key with its default.

**Dead weight.** `adlif.enabled` combined with `two_stage.enabled` is inert:
the adLIF currents subtract from `pre`, and the two-stage update rebuilds the
state from `mix` and the inner `tanh` without reading `pre`, so
`adlif_gain_fast`/`_slow` get `grad = None` (200 parameters that never train in
the final $N=100$ model). See `TODO.md` §Dead weight.

---

## 3. SHD setup

* Official event-based files: `data/shd_train.h5` (8156 samples, speakers
  `{0,1,2,3,6,7,8,9,10,11}`) and `data/shd_test.h5` (2264 samples, speakers
  `{0..11}` — speakers 4 and 5 never occur in the training file).
* Events are binned onto `T x C = 500 x 700` with **2 ms bins** (a 1000 ms
  window, one dynamical step per bin) and **binarised** (≥1 event in a bin → 1,
  the standard SHD encoding). The event/temporal structure is preserved; nothing
  is collapsed into a static feature vector.
* The window covers **96.6%** of training and **99.0%** of test utterance spans;
  events after 1000 ms are dropped (measured clip fraction: 0.013% of events on
  the training file for the sampled subset used in the notebook).
* Measured input statistics: 7846 events/sample, i.e. ≈2.2% of the `(500, 700)`
  grid cells active at the 2 ms default (≈4.3% of the `(250, 700)` cells at 4 ms).
* `FIT` / `VAL` are a class-stratified random 90/10 split of the **training**
  file. The official test file is opened only in the test phase and never used
  for any decision.

---

## 4. Training

* genuine end-to-end training of every parameter above, by
  **backprop-through-time** with **surrogate gradients**
* surrogate: the derivative of the fast-sigmoid primitive
  $\gamma u/(1+\beta|u|)$, i.e. $\gamma/(1+\beta|u|)^2$ ($\beta = 4$,
  $\gamma = 1$). Implemented as an explicit `autograd.Function` so the gradient
  is exactly that expression; the notebook checks it against autograd.
* loss: cross-entropy (+ firing-rate regulariser), computed in **fp32**
* optimizer: **Adam** (`lr = 1e-3`, no weight decay), cosine schedule with 3%
  warmup, gradient-norm clipping at 1.0
* batches of 128, `tqdm` progress per batch, per-epoch train loss / train
  accuracy / validation accuracy / firing rate / wall time recorded
* gradient checkpointing over 10 time-chunks keeps VRAM inside the 6 GiB budget

## 5. Test

Training is stopped, all parameters are frozen
(`requires_grad_(False)`), gradients are disabled and **every** one of the 2264
official test samples is evaluated. The reported numbers are exact counts
(`correct`, `total`, `accuracy`) plus the full confusion matrix and the test
predictions saved to `V3/results/<tag>/test_predictions.npz`.

## 6. CUDA / FP16 behaviour (what is actually fp16)

| tensor / operation | dtype |
|---|---|
| neural state `z`, spikes, `a`, and every large activation in the step | **float16** |
| all model weights *as used in the matmuls* (cast from fp32 masters each step) | **float16** |
| master weights held by the optimizer | float32 |
| optimizer state (Adam moments) | float32 |
| pre-spike potentials, threshold comparison, surrogate-gradient backward | float32 |
| logits, cross-entropy, rate regulariser, loss | float32 |
| gradient scaling | `torch.amp.GradScaler` (fp16 activation gradients are rescaled) |

The model does **not** claim that everything is fp16. The state and every large
tensor are genuinely fp16 (the notebook prints their dtypes, and
`run_smoke_test.py` asserts them); the numerically delicate parts stay fp32.
This is only possible because `tanh` bounds the state, so no fp16 overflow
occurs — unlike the fp16 problems seen in the Brian 2 experiment.

### GPU memory (the one thing that really bites)

The per-step backprop-through-time graph is the memory driver. Training uses
chunked gradient checkpointing (`grad_checkpoint_chunks`), which stores the state
only at chunk boundaries. Measured peak over a full epoch at the **default 2 ms**
discretization, `N=64, D=1000, T=500, batch=128`: **2284 MiB allocated /
2328 MiB reserved**, ≈ 3.5 s per batch (the full 20-epoch training run took
4243.5 s). Halving the time step doubles `T` and roughly doubles both time and
memory per epoch.

Two traps, both measured on the 6 GiB RTX 3060 Laptop:

* An **un-checkpointed** 250-step forward *with gradients enabled* allocates
  **8.4 GiB** at `D=1000` — more than the device has. NVIDIA's WDDM driver then
  silently pages GPU memory to host RAM and every subsequent training step runs
  roughly **150× slower** (measured: 333 s/batch instead of 2.2 s/batch, an ETA
  of 5 h 16 m for the first epoch). Every diagnostic in the notebook therefore
  runs under `torch.no_grad()` and frees its tensors, and the notebook prints the
  live CUDA footprint right after those checks so the condition is visible.
  If you add your own diagnostics, keep this in mind.
* `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set before the first CUDA
  allocation (in `v3/__init__.py`, `setdefault` so your own setting wins). It is
  not what makes training fast, but it keeps the allocator's reserved pool at the
  true peak instead of letting fragmentation drift it upwards.

If CUDA is unavailable the code falls back to `device=cpu, dtype=float32` for
debugging only. The project is not designed around CPU execution.

---

## 7. How to run

Everything needs the repository virtual environment.

```powershell
# --- the notebook is the primary deliverable and runs top to bottom ---
jupyter lab            # open V3/V3_SHD_experiment.ipynb  (ipykernel is installed)
#   or execute it headlessly with a real kernel:
.venv\Scripts\python.exe V3\tools\run_notebook.py

# --- headless CLI, exactly the same code path ---
.venv\Scripts\python.exe V3\run_experiment.py --variant both
.venv\Scripts\python.exe V3\run_experiment.py --variant both --quick     # minutes
.venv\Scripts\python.exe V3\run_experiment.py --variant vector --override epochs=30

# --- mechanics smoke test (surrogate maths, dtypes, coupling, fp16, VRAM) ---
.venv\Scripts\python.exe V3\run_smoke_test.py
```

Set `V3_QUICK=1` before launching the notebook kernel to run a minutes-long
smoke of the identical code path. `V3_OVERRIDES` takes a JSON object of
configuration overrides and lets you shorten or vary a run without editing the
notebook:

```powershell
$env:V3_OVERRIDES='{"epochs": 3}'; .venv\Scripts\python.exe V3\tools\run_notebook.py
```

Artifacts written per run (all under `V3/`):

```
results/<tag>/config.json              resolved configuration + parameter budget
results/<tag>/history.csv|.json        per-epoch loss / accuracy / rate / time
results/<tag>/train_metrics.json       FIT + VAL metrics, dtypes, wall time
results/<tag>/test_metrics.json        exact test counts, accuracy, rate stats
results/<tag>/test_confusion.json      confusion matrix + per-class accuracy
results/<tag>/test_predictions.npz     indices, labels, predictions, logits
checkpoints/<tag>.pt                   final model + config + metrics
checkpoints/<tag>_bestval.pt           best-validation-epoch model (secondary)
figures/<tag>/                         training curves, confusion, per-class, activity
results/summary.json                   vector-vs-scalar summary (CLI --variant both)
```

## 8. Results

**Executed** (`V3_SHD_experiment.ipynb`, top to bottom, 20 code cells, no errors).
The default configuration is `epochs: 20` at the **2 ms** discretization
(`T=500`, `alpha=0.1`). Both variants use the identical code path, data, readout,
optimizer and epoch count. Full per-epoch numbers are in `RESULTS.md`; the run is
also in the registry (`V3/results/runs.csv`, run `2026-10-02_14-31-15`).

| | vector `D=1000` | scalar control `D=1` (untuned) |
|---|---|---|
| trainable parameters | 2,025,440 | 6,335 |
| FIT accuracy (final epoch) | 100.00% | 58.01% |
| VAL accuracy (final epoch)¹ | 96.32% | 53.68% |
| **TEST correct / total** | **1868 / 2264** | **1119 / 2264** |
| **TEST accuracy** | **82.51%** | **49.43%** |
| test errors | 396 | 1145 |
| per-class accuracy min / median / max | 38.8% / 88.6% / 100.0% | not recorded |
| classes below 10% recall | **0** | not recorded |
| English (0–9) / German (10–19) | 90.31% / 75.08% | not recorded |
| mean / median / max firing rate | 18.22 / 17.71 / 37.02 Hz | not recorded |
| silent neurons (of 64) | 0 | not recorded |
| peak CUDA memory (train) | 2284 MiB allocated / 2328 MiB reserved | not recorded |
| test pass wall time | 24.1 s for all 2264 samples | not recorded |
| training wall time (20 epochs) | 4243.5 s (≈ 204 s/epoch) | 1380.6 s (≈ 69 s/epoch) |

¹ `VAL` is a class-stratified **random** 10 % split of the training file
(`V3/v3/data.py:169-198`), so it shares speakers with `FIT`: it is
speaker-overlapping, used for monitoring only, and is **not a generalisation
estimate**. The `TEST` row is the comparable number.

**The vector population beats this control by +33.08 percentage points**
(82.51% vs 49.43%) with the same 64 neurons, the same input, the same readout and
the same training procedure. The scalar row is an **untuned, size-matched
control (ablation)**, not a competitive baseline: no improvement over a
literature baseline is claimed from this gap, and 49.43% says nothing about what
a tuned scalar model could reach. At 20 epochs the control has learned much more
than at 3 epochs (49.43% vs 26.55%), so the gap is **not** merely "the scalar
model had not trained yet" — but it is one seed, one schedule, and three
simultaneous changes (`D`, the population broadcast, the parameter count; see
`INVESTIGATION.md`). The repository's own V2 LIF model (N=256, T=1400 ms,
20 epochs) measures 59.98% (`doc-claim`). A plain recurrent SHD baseline is
usually quoted around 71%; this repository has not reproduced one, so that figure
is context only and no claim rests on it.

The honest capacity comparison is the 5-epoch screen below (`screening`, seed 0,
2 ms, identical training for every row, never a headline). The scalar family is
flat from 6,335 to 2,025,488 parameters while the vector family rises steeply
with `D`:

| family | `N` | `D` | parameters | 5-epoch TEST | tag |
|---|---|---|---|---|---|
| scalar | 64 | 1 | 6,335 | 797/2264 = 35.20% | screening |
| scalar | 256 | 1 | 72,383 | 718/2264 = 31.71% | screening |
| scalar | 1411 | 1 | 2,025,488 | 750/2264 = 33.13% | screening |
| vector | 64 | 4 | 9,056 | 1295/2264 = 57.20% | screening |
| vector | 64 | 16 | 20,480 | 1443/2264 = 63.74% | screening |
| vector | 64 | 64 | 74,816 | 1553/2264 = 68.60% | screening |
| vector | 64 | 256 | 332,096 | 1757/2264 = 77.61% | screening |
| vector | 64 | 1000 | 2,025,440 | 1724/2264 = 76.15% | screening |

**Protocol disclosure.** Screening and model selection used the official
**TEST accuracy at 5 epochs** (n = 2264, binomial SE ±0.84 pp), by PI decision:
`VAL` is speaker-overlapping and saturated near the end of training (96–99 %), so
it could not discriminate between variants. No test sample was trained on and no
test data builds features; the disclosure is that the 5-epoch selector saw test
accuracy, so 5-epoch test numbers carry a selection effect and are tagged
`screening`, never headline.

**Which weights the headline uses.** The 82.51% comes from the **final-epoch**
weights (`v3/experiment.py:192` defaults to `which="final"`;
`v3/train.py:330-334` only clones the best-`VAL` state into `result.state_dict`,
it never restores it). `checkpoints/<tag>_bestval.pt` is selected on the
speaker-overlapping `VAL` and is **diagnostic only** — it must not be used to
reproduce the headline numbers.

Training progressed as (vector; the full 20-epoch table is in `RESULTS.md`):

| epoch | train loss | train acc | val loss | val acc | rate | spikes/sample | epoch time |
|---|---|---|---|---|---|---|---|
| 1 | 1.9840 | 37.26% | 1.2104 | 61.89% | 24.55 Hz | 1571 | 220.9 s |
| 2 | 0.9246 | 70.53% | 0.7634 | 75.61% | 27.60 Hz | 1766 | 204.0 s |
| 5 | 0.3424 | 90.18% | 0.4536 | 86.52% | 25.26 Hz | 1617 | 204.3 s |
| 10 | 0.1295 | 96.54% | 0.2330 | 93.38% | 22.18 Hz | 1420 | 203.7 s |
| **16** | 0.0208 | 99.90% | **0.1021** | **97.67%** | 18.66 Hz | 1195 | 207.0 s |
| 20 | 0.0135 | 100.00% | 0.1129 | 96.32% | 17.71 Hz | 1133 | 204.4 s |

best-validation epoch = **16** (val 0.9767). Figures written to `V3/figures/<tag>/`:
training curves, test confusion matrix, per-class accuracy, and input/spike
rasters for representative test and FIT samples.

## 9. Layout

```
V3/
  README.md                     this file
  TODO.md                       deliberately unimplemented future work
  RESULTS.md                    the measured numbers, nothing else
  V3_SHD_experiment.ipynb       the primary deliverable, executed top to bottom
  configs/v3_default.yaml       every experimental parameter, in one place
  v3/config.py                  the configuration dataclass
  v3/data.py                    SHD loading, event binning at the configured bin width, FIT/VAL/TEST splits
  v3/model.py                   the vector-neuron population (the science)
  v3/train.py                   training / evaluation loops
  v3/experiment.py              orchestration used by BOTH the notebook and the CLI
  v3/plots.py                   training and test figures
  v3/state_regularization.py    controlled state noise + STE quantization
  v3/registry.py                persistent run registry (runs.csv + per-run dirs)
  tests/conftest.py             puts V3/ on sys.path for the isolated test module
  tests/test_state_regularization.py  focused noise/quantization/phase tests
  tests/test_registry.py        focused registry tests
  tests/test_timing.py          timing derivation + SHD event binning tests
  tests/test_shuffle_and_state.py  shuffle semantics + state-reset tests
  run_experiment.py             headless CLI
  run_smoke_test.py             mechanics / precision / VRAM smoke test
  tools/build_notebook.py       assembles the .ipynb from its cell list
  tools/run_notebook.py         executes the .ipynb with a real kernel
  results/ checkpoints/ figures/  outputs (results/runs.csv = the registry)
```

## 10. Known limitations

1. **Single seed.** No error bar on any number. See `TODO.md`.
2. **`VAL` is not speaker-independent** (it is a stratified random split of the
   training file, `V3/v3/data.py:169-198`, so it shares speakers with `FIT`). The
   published SHD train/test gap is dominated by speaker shift — the official test
   set contains speakers 4 and 5, which are absent from the training file — so
   `VAL` is optimistic and is used only for monitoring and best-checkpoint
   selection. The **test** number is the comparable one.
3. **Truncated tail.** A 1000 ms window drops events beyond 1000 ms for ~3.4% of
   training and ~1.0% of test utterances.
4. **No data augmentation.** Published competitive SHD numbers typically use
   event dropout / time-shift augmentation; none is used here.
5. **Modest state magnitude.** Because binned SHD input is sparse, the state sits
   at order $10^{-1}$ rather than near the $\tanh$ bound, so the neurons operate
   in the gently-nonlinear regime. This is measured in the notebook, not assumed.
6. **Not a leaderboard entry.** This is a first working implementation of the
   vector-neuron population; throughput optimisation, architecture search and
   longer schedules are all in `TODO.md`.
7. `V3/run_smoke_test.py` is a standalone script, not a `pytest` module; it lives
   outside the repository's `tests/` suite on purpose (V3 is isolated). The
   focused unit/regression tests live in `V3/tests/` (also isolated).

## 11. Controlled state imprecision + the run registry

This stage asks one question only:

> Does deliberately reducing the precision of each neuron's internal vector state
> improve SHD generalization?

It changes **nothing** about the architecture, the data, the optimizer, the
scheduler or the readout. The only new manipulation is a perturbation applied to
the neuron state $\mathbf z(t)$ at the point where the next timestep reads it.

### Three distinct notions of "precision" (kept explicit)

| # | what | where |
|---|---|---|
| 1 | training arithmetic dtype (`fp16` state, autocast, `GradScaler`) | `dtype` / `amp` |
| 2 | precision of the state representation | `state_regularization.quantize_bits` |
| 3 | stochastic perturbation of the state | `state_regularization.noise_std` |

Ordinary fp16/mixed-precision training is **not** the experiment. The experiment
is (2) and/or (3), applied deliberately and configurably.

### Modes

Configured in one block, disabled by default:

```yaml
state_regularization:
  mode: none          # none | noise | quantization | noise_quantization
  noise_std: 0.01
  quantize_bits: 8
  quantize_clip: 1.0
  apply_during_training: true
  apply_during_validation: false
  apply_during_test: false
```

* **noise** — $\mathbf z \leftarrow \mathbf z + \epsilon$, $\epsilon\sim N(0,\sigma^2)$.
* **quantization** — a symmetric, dynamic-range uniform quantizer with a
  straight-through estimator: `scale = clamp(max|z|, eps, clip)`,
  `q = round(clamp(z/scale * (2^(b-1)-1), -levels, levels))`,
  `z_q = q/levels*scale`, and the backward pass is the identity
  (`z + (z_q - z).detach()`). The forward pass is genuinely quantized, training
  stays end-to-end. `scale` is detached; the range is bounded by `clip` (the
  `tanh` invariant $|z|\le 1$) and floored at `eps` so a degenerate range cannot
  amplify float noise.
* **noise_quantization** — quantize, then add noise.

A final `clamp(-1, 1)` preserves the $\tanh$ bound the fp16 path relies on.

### Phase behaviour

The perturbation is **training-only by default** (a regularizer). Validation and
the official test evaluation run at full precision unless the experiment opts in
via `apply_during_validation` / `apply_during_test`. This separates
*training-time regularization* from *deliberately degraded inference*; the two are
never conflated. Because `mode: none` short-circuits before any tensor op, the
default configuration is the exact baseline.

### Run registry

Every completed run appends exactly one row to `V3/results/runs.csv` and writes a
timestamped directory:

```
V3/results/
  runs.csv
  2026-10-02_00-15-30/
    config.yaml            # the exact configuration used
    metrics.json           # fit / val / test metrics (no big arrays)
    confusion_matrix.csv
    summary.txt
```

The row records timestamp, run id, `N`, `D`, parameters, epochs, seed, batch size,
learning rate, optimizer, scheduler, training precision, noise/quantization
settings (mode, std, bits, enabled), train/val/test accuracy, correct/total,
English (0-9) and German (10-19) accuracy, durations, checkpoint path and notes,
plus the temporal settings (`sequence_duration_ms`, `time_bin_ms`,
`num_time_steps`, `simulation_dt_ms`) and the shuffle flags.
Run ids come from the **local system time** (`%Y-%m-%d_%H-%M-%S`, disambiguated
with a `_NN` suffix on collision). The registry is append-only and never
overwrites; if the column set grows, the existing file is rewritten with the
union of the old and new columns so earlier rows survive. `V3_SHD_experiment.ipynb`
registers its completed run through the **same** `v3.registry.register_run` entry
point as the CLI (no second format). Test confusion matrices are diagnostics,
never a training signal.

### Screening workflow

The reference (2 ms, 20 epochs) took ~72 min of vector training; **screening** is
run at 5 epochs (~17 min) and the promising settings are only then re-run at 20
epochs. The YAML default is `epochs: 20`, so pass `--override epochs=5` to screen.
The runner accepts dotted overrides, so no YAML edits are needed between runs:

```powershell
# baseline screening (mode none, 5 epochs, 2 ms default)
.venv\Scripts\python.exe V3\run_experiment.py --variant vector --override epochs=5

# noise sweep
.venv\Scripts\python.exe V3\run_experiment.py --variant vector --override epochs=5 `
  --override state_regularization.mode=noise --override state_regularization.noise_std=0.01

# quantization sweep
.venv\Scripts\python.exe V3\run_experiment.py --variant vector --override epochs=5 `
  --override state_regularization.mode=quantization --override state_regularization.quantize_bits=8
```

To see every key the override grammar accepts (with its current value and type),
ask the runner instead of the documentation — the list is generated from the
config dataclasses, so it cannot go stale:

```powershell
.venv\Scripts\python.exe V3\run_experiment.py --help-overrides
```

Suggested screening values: `noise_std` ∈ {0.001, 0.005, 0.01, 0.02, 0.05};
`quantize_bits` ∈ {16, 12, 8, 6, 4}. Nothing launches a grid automatically — each
command runs exactly one configuration and records exactly one row. The measured
5-epoch 2 ms baseline is 76.15% (1724/2264) for context.

### Optional model features on `main` (all default off)

Three features were developed on their own branches, screened, and merged into
`main` after the screen. Each is an independent nested config block, disabled by
default, and **the disabled path is bit-identical to the baseline** (verified by
`V3/tests/test_feature_combinations.py` and by a cross-checkout logits hash
against the pre-merge commit). They can be enabled individually, in any pair, or
all at once:

| block | feature | extra parameters (N=64, D=1000) |
|---|---|---|
| `adaptation` | spike-frequency adaptation: `pot -= w_ad * b`, `b = rho*b + s` | 64 (`w_ad`) |
| `state_dynamics` | per-neuron leak `alpha_i = sigmoid(raw_i)` and learned `z0` | 64 + 64,000 |
| `input_delays` | per-input-channel fractional delays, `0.5–10 ms` init | 700 |

```powershell
# all three at once
.venv\Scripts\python.exe V3\run_experiment.py --variant vector --override epochs=5 `
  --override adaptation.enabled=true `
  --override state_dynamics.enabled=true `
  --override state_dynamics.trainable_alpha=true `
  --override state_dynamics.learnable_z0=true `
  --override input_delays.enabled=true
```

Screen status of each block (5 epochs, official TEST, N=64/D=1000, baseline
1724 = 76.15%): `adaptation` 1853 = 81.85% at 5 epochs but **1861 = 82.20%** at
20 epochs (−0.31 vs reference); `state_dynamics` 1740 = 76.86% at 5 epochs and
**1929 = 85.20%** at 20 epochs (+2.69, the gain comes from the learned `z0`, not
the leak — the learned `alpha` stayed at 0.098); `input_delays` 1769 = 78.14% at
5 epochs and **1837 = 81.14%** at 20 epochs (−1.37). Numbers carry the
`confirmed` (20 epochs) / `screening` (5 epochs) tag of rule 6.

### Parameter efficiency: diagonal emission (branch `v3.1`, engineering only)

`emission.mode=diagonal` replaces the shared `D x D` emission with one learned
gain per coordinate: `h = ((s*z).sum(1) * pop_scale) * g`, `g in R^D`. It keeps
the emitted object in `R^D` — the spike still carries a vector — and only
removes the mixing, so this is a software/parameter improvement of the same
model, not a new model.

| mode | parameters (N=64, D=1000) | 5-epoch TEST |
|---|---|---|
| `full` (default) | 2,025,440 | 1724 = 76.15% |
| `diagonal` | **1,026,440** | **1748 = 77.21%** |

Half the parameters were that one matrix, and the 5-epoch number is inside the
±0.87 pp noise band (`screening`, single seed). The default stays `full`
because every headline run used it. Verification: `full` mode is bit-identical
to the pre-change code (all non-emission tensors, logits maxdiff 0.0), and with
a genuinely diagonal emission matrix the diagonal path reproduces the full path
exactly (maxdiff 0.0 in fp32). Merged into `main` as `7848247` (branch `v3.1`).

## 12. Current experimental reference

The **headline** is the final merged model (`RESULTS.md` §9):

```
N = 100   D = 1000   epochs = 30   params = 3,069,201   batch = 96
timing: 1000 ms / 2 ms -> T = 500 + 50 tail bins, alpha = 0.1
```

```
VECTOR  all blocks  3,069,201 params   TEST 1994/2264 = 88.07%   (epoch 28: 2009 = 88.74%)
```

The historical reference that the screening runs were compared against is the
plain 20-epoch vector model:

```
N = 64    D = 1000    epochs = 20    params = 2,025,440
VECTOR  D=1000  N=64    2,025,440 params   test  1868/2264 =  82.51%
```

The size-matched scalar control ($D=1$, $N=64$, 6,335 parameters) reaches
49.43% at 20 epochs and the capacity-matched scalar ($N=1411$) 33.13% at 5
epochs, which is the comparison that matters for the width claim.

The lean scalar-current variant (design A′, `RESULTS.md` §9.1) is
`--override input_current.enabled=true --override input_current.bias_init_std=0.5
--override emission.mode=diagonal`: 411,100 parameters, 1937 = 85.56 % at
N=100, 20 epochs, no augmentation.

## 13. Temporal resolution and data order

### Timing

The temporal discretization is one explicit block; the number of dynamical steps
is **derived**, never stored:

```yaml
timing:
  sequence_duration_ms: 1000
  time_bin_ms: 2          # T = 1000 / 2 = 500   (DEFAULT)
```

`time_bin_ms` is both the SHD bin width **and** the model integration step `dt`
(one bin == one dynamical step), so the leak `alpha = dt/tau` follows:

| `time_bin_ms` | steps `T` | window | `alpha` (tau=20 ms) | status |
|---|---|---|---|---|
| 2 ms | 500 | 1000 ms | 0.1 | **default discretization** |
| 4 ms | 250 | 1000 ms | 0.2 | historical reference (`time_bin_ms: 4`) |

Events are binned **directly from the original timestamps** at the requested
`time_bin_ms` (never by splitting the 4 ms bins in half). The same 1000 ms window
and the same truncation policy are used at both resolutions: events with
`t < 1000 ms` are kept (`t = 0` goes to bin 0), the bin index is clamped to the
last bin, and events at/after the cutoff are dropped. A non-integer
`sequence_duration_ms / time_bin_ms` is rejected at configuration time. Switching
the default 2 ms to 4 ms changes `T`, `alpha` and the input tensor shape
(`[B, 500, 700]` → `[B, 250, 700]`) with no special-casing anywhere.

### Data order (verified by `V3/tests/test_shuffle_and_state.py`)

* **FIT/VAL split** — one fixed stratified random 90/10 split of the *training*
  file (`split_seed`); the same examples are in FIT/VAL for every epoch.
* **FIT batches** — shuffled between epochs (`shuffle_train: true`), seeded by
  `(seed, epoch)` so a given seed reproduces the order. The full FIT set is seen
  exactly once per epoch (no `drop_last`); shuffling changes the order, not the
  set of training examples.
* **VAL batches** — `shuffle_val: false`, so validation order and accuracy are
  deterministic and comparable across epochs. Validation is *not* a fresh random
  sample each epoch.
* **State reset** — every batch starts from a fresh zero state ($Z(0)=0$); no
  neuron state or autograd graph carries over between batches.
