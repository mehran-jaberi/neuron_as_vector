# V3 — a population of genuine $D$-dimensional vector-valued neurons

> **Question.** How much of the official SHD test set can a population of
> $D = 1000$ **vector-valued** spiking neurons classify correctly?

> **Answer (executed, 20 epochs, 2 ms, single seed).** **82.51%** — 1868 of the
> 2264 official SHD test samples. The identical population with scalar neurons
> ($D=1$, same 64 neurons, same training) reaches **49.43%**. See [§8](#8-results)
> and `RESULTS.md`.
>
> **Current experimental reference.** $N=64$, $D=1000$, 2,025,440 parameters,
> **2 ms** discretization ($T=500$, $\alpha=0.1$), 20 epochs: **82.51%**
> (1868/2264), recorded in the executed `V3_SHD_experiment.ipynb` and in the
> registry run `2026-10-02_14-31-15`. The previous 4 ms reference ($T=250$,
> $\alpha=0.2$) was 79.81% (1807/2264). Imprecision screening is compared against
> this 82.51% reference; see [§11](#11-controlled-state-imprecision--the-run-registry).

This folder is an independent experiment. It is **not** an extension of V2 and
does not import, modify or reproduce any of the V2 architecture
(`src/`, `scripts/`, `configs/` of the repository root are untouched). It is also
not a rerun of `brian implementation/`, which was a *fixed* high-dimensional
dynamical system with a trained readout. Here the vector dynamics themselves are
trained end-to-end.

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
baseline population *through the identical code path*.

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
4243.5 s). The earlier 4 ms reference (`T=250`) measured 1108 MiB allocated /
1148 MiB reserved and ≈ 2.2 s per batch; halving the time step doubles `T` and
roughly doubles both time and memory per epoch.

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

| | vector `D=1000` | scalar baseline `D=1` |
|---|---|---|
| trainable parameters | 2,025,440 | 6,335 |
| FIT accuracy (final epoch) | 100.00% | 58.01% |
| VAL accuracy (final epoch) | 96.32% | 53.68% |
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

**The vector population beats the scalar population by +33.08 percentage points**
(82.51% vs 49.43%) with the same 64 neurons, the same input, the same readout and
the same training procedure. At 20 epochs the scalar baseline has learned much
more than at 3 epochs (49.43% vs 26.55%), so the vector advantage is **not**
merely "the scalar model had not trained yet"; the vector population reaches
82.51%, above the repository's own V2 LIF baseline (59.98%, N=256, 20 epochs) and
well above the ~71% usually quoted for a plain recurrent SHD baseline.

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
2. **`VAL` is not speaker-independent** (it is a stratified split of the
   training file, so it shares speakers with `FIT`). The published SHD
   train/test gap is dominated by speaker shift — the official test set contains
   speakers 4 and 5, which are absent from the training file — so `VAL` is
   optimistic and is used only for monitoring and best-checkpoint selection. The
   **test** number is the comparable one.
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

Suggested screening values: `noise_std` ∈ {0.001, 0.005, 0.01, 0.02, 0.05};
`quantize_bits` ∈ {16, 12, 8, 6, 4}. Nothing launches a grid automatically — each
command runs exactly one configuration and records exactly one row. The measured
5-epoch 2 ms baseline is 76.15% (1724/2264) for context.

## 12. Current experimental reference

```
N = 64    D = 1000    epochs = 20    params = 2,025,440
timing: 1000 ms / 2 ms  ->  T = 500, alpha = 0.1
```

```
VECTOR  D=1000  N=64    2,025,440 params   test  1868/2264 =  82.51%
```

Verified from the executed `V3_SHD_experiment.ipynb` (the stored test evaluation
cells) and the registry run `2026-10-02_14-31-15`. This is the number every
imprecision screening run is compared against. The previous 4 ms reference
(250 steps, alpha 0.2, same 20 epochs) measured **79.81%** (1807/2264); older
measured points (all single-seed): $D=1000$ at 3 epochs (4 ms) = 71.78%; scalar
$D=1$ at 3 epochs = 26.55% and at 20 epochs (2 ms) = 49.43%; $N=128$, $D=1000$
(4 ms) = 79.24%; $N=64$, $D=2000$ (4 ms) = 77.96%. Doubling $N$ (64→128) or $D$
(1000→2000) did not improve on the reference, whereas halving the time step
(4 ms → 2 ms) did (+2.70 points).

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
