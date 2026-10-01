# V3 — a population of genuine $D$-dimensional vector-valued neurons

> **Question.** How much of the official SHD test set can a population of
> $D = 1000$ **vector-valued** spiking neurons classify correctly?

> **Answer (executed, 3 epochs, single seed).** **71.78%** — 1625 of the 2264
official SHD test samples. The identical population with scalar neurons
($D=1$, same 64 neurons, same training) reaches **26.55%**. See [§8](#8-results)
and `RESULTS.md`. The YAML's default 20-epoch schedule is the deferred next run.

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
| state leak $\alpha = \Delta t/\tau$, $\tau = 20$ ms | standard leaky-integrator dynamics; $\Delta t$ = one 4 ms bin, so each bin is one dynamical step |
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
* Events are binned onto `T x C = 250 x 700` with **4 ms bins** (a 1000 ms
  window, one dynamical step per bin) and **binarised** (≥1 event in a bin → 1,
  the standard SHD encoding). The event/temporal structure is preserved; nothing
  is collapsed into a static feature vector.
* The window covers **96.6%** of training and **99.0%** of test utterance spans;
  events after 1000 ms are dropped (measured clip fraction: 0.013% of events on
  the training file for the sampled subset used in the notebook).
* Measured input statistics: 7846 events/sample, 4.3% of the `(T, C)` grid cells
  active.
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

The 250-step backprop-through-time graph is the memory driver. Training uses
chunked gradient checkpointing (`grad_checkpoint_chunks`), which stores the state
only at chunk boundaries — measured peak over a full epoch at
`N=64, D=1000, T=250, batch=128`: **1108 MiB allocated / 1148 MiB reserved**, with
`nvidia-smi` reporting ~1.7 GiB in use and ~2.2 s per batch.

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

**Executed** (`V3_SHD_experiment.ipynb`, top to bottom, 19/19 code cells, no errors,
712 s wall). The committed execution used a short **3-epoch** schedule
(`V3_OVERRIDES='{"epochs": 3}'`); the YAML default is `epochs: 20` and that
full run is the deferred next step (see below). Both variants use the identical
code path, data, readout, optimizer and epoch count.

| | vector `D=1000` | scalar baseline `D=1` |
|---|---|---|
| trainable parameters | 2,025,440 | 6,335 |
| FIT accuracy (final epoch) | 90.35% | 24.29% |
| VAL accuracy (final epoch) | 86.89% | 24.63% |
| **TEST correct / total** | **1625 / 2264** | **601 / 2264** |
| **TEST accuracy** | **71.78%** | **26.55%** |
| test errors | 639 | 1663 |
| per-class accuracy min / median / max | 37.0% / 72.0% / 100.0% | 0.0% / 25.4% / 68.8% |
| classes below 10% recall | **0** | 5 |
| English (0–9) / German (10–19) | 74.91% / 68.42% | 23.18% / 29.57% |
| mean / median / max firing rate | 24.09 / 21.55 / 95.45 Hz | 10.72 / 5.21 / 66.23 Hz |
| silent neurons (of 64) | 0 | 10 |
| peak CUDA memory (train) | 1283 MiB allocated / 1330 MiB reserved | 144 MiB |
| test pass wall time | 18.8 s for all 2264 samples | 10.3 s |
| training wall time (3 epochs) | 445.5 s (161/137/132 s per epoch) | 151.0 s |

**The vector population beats the scalar population by +45.23 percentage points**
(71.78% vs 26.55%) with the same 64 neurons, the same input, the same readout and
the same training procedure. The scalar baseline is still at chance-plus
(FIT 24.3%, five classes with <10% recall, 10 of its 64 neurons silent), i.e. it has
barely started to learn after 3 epochs, whereas the vector population is already
at 71.78% test accuracy — above the repository's own V2 LIF baseline
(59.98%, N=256, 20 epochs) and close to the ~71% usually quoted for a plain
recurrent SHD baseline. The full 20-epoch run is expected to go further.

Training progressed as (vector):

| epoch | train loss | train acc | val loss | val acc | rate | spikes/sample | epoch time |
|---|---|---|---|---|---|---|---|
| 1 | 1.7030 | 45.33% | 0.9983 | 67.16% | 19.21 Hz | 1229 | 160.7 s |
| 2 | 0.7629 | 77.26% | 0.6061 | 83.82% | 23.65 Hz | 1514 | 136.6 s |
| 3 | 0.4701 | 88.01% | 0.4903 | 86.89% | 24.72 Hz | 1582 | 131.5 s |

Figures written to `V3/figures/<tag>/`: training curves, test confusion matrix,
per-class accuracy, and input/spike rasters for representative test and FIT samples.

### Deferred: the full 20-epoch run

The committed notebook execution uses 3 epochs to keep the turnaround short. To
produce the full result, run the identical code path with the YAML's 20 epochs:

```powershell
# notebook, 20 epochs, ~65 min (43 min vector + 22 min scalar)
.venv\Scripts\python.exe V3\tools\run_notebook.py
# or the equivalent CLI invocation (same functions, same seeds)
.venv\Scripts\python.exe V3\run_experiment.py --variant both
```

Expected cost: ≈ 2.2 min per vector epoch and ≈ 0.7 min per scalar epoch of GPU
work, plus ~30% host-side event binning (`TODO.md`).

## 9. Layout

```
V3/
  README.md                     this file
  TODO.md                       deliberately unimplemented future work
  RESULTS.md                    the measured numbers, nothing else
  V3_SHD_experiment.ipynb       the primary deliverable, executed top to bottom
  configs/v3_default.yaml       every experimental parameter, in one place
  v3/config.py                  the configuration dataclass
  v3/data.py                    SHD loading, 4 ms event binning, FIT/VAL/TEST splits
  v3/model.py                   the vector-neuron population (the science)
  v3/train.py                   training / evaluation loops
  v3/experiment.py              orchestration used by BOTH the notebook and the CLI
  v3/plots.py                   training and test figures
  run_experiment.py             headless CLI
  run_smoke_test.py             mechanics / precision / VRAM smoke test
  tools/build_notebook.py       assembles the .ipynb from its cell list
  tools/run_notebook.py         executes the .ipynb with a real kernel
  results/ checkpoints/ figures/  outputs
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
   outside the repository's `tests/` suite on purpose (V3 is isolated).
