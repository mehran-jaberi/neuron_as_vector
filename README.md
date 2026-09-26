# Neuron as Vector

**Do structured representations of individual hidden neurons in a trained recurrent
spiking neural network (SNN) capture their functional organisation?**

This repository is a self-contained, reproducible proof-of-concept study. Its
central object is *not* the classifier's accuracy. Its central object is the
**representation of each individual hidden neuron** and the question of whether
the **geometry** of those representations reflects the neurons' **function**.

The intended target is a [SNUFA](https://snufa.net/) 2026 submission.

> **Result in one line** (see [§7](#7-results) for the full analysis): a label-free
> representation built only from the network's parameters predicts individual
> neurons' functional fingerprints with a Mantel correlation of
> **$r = 0.298 \pm 0.051$** across three seeds ($p = 0.000999$ in every seed),
> and the effect **survives a partial Mantel test controlling for firing rate**
> ($r = 0.242 \pm 0.061$). Chance and shuffle controls are null.

---

## 1. The scientific question

> Can individual neurons in a trained recurrent spiking neural network be
> represented in a structured multidimensional space such that geometric
> relationships between neurons reflect their functional organisation?

We make this precise and falsifiable in the following sense. For every hidden
neuron $i$ we build two *independent* descriptions:

- a **neuron representation** $\phi_i \in \mathbb{R}^d$ — **label-free**,
  derived from the network parameters (and optionally label-free firing
  statistics); and
- a **functional fingerprint** $\psi_i \in \mathbb{R}^m$ — **label-using**,
  derived from *class-conditioned, held-out* responses.

We then ask whether the two induced geometries agree:

$$
\text{Spearman}\!\left(\ \mathrm{vec}\,D^{\phi},\ \ \mathrm{vec}\,D^{\psi}\ \right) \;>\; 0 ,
$$

where $D^{\phi}$ and $D^{\psi}$ are the matrices of pairwise distances among
neurons in representation space and fingerprint space respectively. A positive
Mantel correlation means *neurons that are close in representation space tend to
be functionally similar*. This is the **pre-registered primary metric**; every
alternative answer is treated as a *control* and reported alongside it.

### The leakage boundary

The representation must never see labels or the evaluation set; the fingerprint
must. This boundary is enforced **structurally in code** (see `scripts/_pipeline.py`):
the representation is built from the model parameters plus a *label-free*
activity accumulator on a reference split, while the fingerprint is built from a
*separate, labelled* accumulator on a held-out split. The official SHD **test**
set is never used for model selection or representation building.

---

## 2. Dataset — SHD (Spiking Heidelberg Digits)

The **Spiking Heidelberg Digits** dataset (SHD) is used as the benchmark task:

- 20 output classes (spoken digits 0–9, two speakers each);
- 700 input channels (cochlear model output);
- event-based (asynchronous spike) streams with per-sample **speaker** metadata;
- official split: **8156** training / **2264** test samples; the two test
  speakers are **test-only** (never seen in training);
- license: **CC BY 4.0**.

The official train/test split is used as given, so the reported test accuracy is
on genuinely held-out speakers. Model selection uses a validation split carved
out of the official **training** data only:

- **speaker-aware** split (whole speakers held out) by default, which avoids the
  optimistic bias that per-sample random splits create here; and
- a **stratified random** split as a documented fallback when speaker metadata
  is unavailable.

The data is stored as HDF5 (`.h5`) with `spikes`, `labels` and `extra`
(speaker) groups; it is downloaded automatically on first use, or read from a
user-specified local directory.

**Citation** (please cite if you use this work):

> Cramer, B., Stradmann, Y., Schemmel, J., & Zenke, F.
> *The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking
> Neural Networks.* IEEE Transactions on Neural Networks and Learning Systems.
> DOI: [10.1109/TNNLS.2020.3044364](https://doi.org/10.1109/TNNLS.2020.3044364)

---

## 3. Network architecture

A discrete-time **current-based leaky integrate-and-fire (LIF)** recurrent SNN,
implemented in PyTorch and trained with **surrogate-gradient backpropagation
through time (BPTT)**. It is deliberately a *small, analyzable* network, **not**
a state-of-the-art SHD classifier.

```
700 input channels
        │  W_in
        ▼
256 recurrent LIF hidden neurons  ──recurrent──▶  (W_rec, hidden→hidden)
        │  W_out
        ▼
20 readout neurons  (linear leaky integration → class logits)
```

| Element            | Default            | Configurable |
| ------------------ | ------------------ | ------------ |
| Input channels     | 700                | `model.n_input` |
| Hidden neurons     | **256** (128/256/512) | `model.n_hidden` |
| Output neurons     | 20                 | `model.n_output` |
| Time bins          | 500 @ 2 ms          | `model.n_bins`, `model.bin_ms` (e.g. 1000 @ 1 ms) |
| Membrane time const | 20 ms              | `model.tau_mem_ms` |
| Synaptic time const | 5 ms               | `model.tau_syn_ms` |
| Threshold          | 1.0                | `model.threshold` |
| Reset              | subtract (soft)     | `model.reset` (`subtract`/`zero`) |
| Surrogate gradient | SuperSpike fast-sigmoid | `model.surrogate_beta`, `model.surrogate_gamma` |
| Learnable neuron params | bias           | `model.neuron_param_mode` (`none`/`bias`/`bias_tau`) |

The model exposes, for every hidden neuron: **input weights** $W_{in}$, **recurrent
weights** $W_{rec}$ (in/out), **output weights** $W_{out}$, **membrane
potentials** $V(t)$, **spikes** $S(t)$, and the **output activity**. These are the
raw materials for the neuron representations.

Weight initialisation: input weights scaled by `input_weight_scale` (default 10),
recurrent weights by `recurrent_weight_scale` (default 2); recurrent connectivity
is dense by default (`recurrent_density = 1.0`), with optional self-connection
removal.

---

## 4. Neuron representation (label-free — the object under study)

For each hidden neuron we assemble a feature vector from **structured blocks**.
The default *primary* representation uses the four structural blocks; the
label-free `activity` block is added when `activity` is listed in
`representations.primary_blocks`.

| Block | Name (`FeatureBlock`) | Content |
| ----- | --------------------- | ------- |
| **Intrinsic** | `intrinsic` | per-neuron parameters the architecture actually learns per neuron |
| **Input conn.** | `input_conn` | channel-permutation-invariant statistics of $W_{in}[\cdot, i]$ |
| **Recurrent in** | `recurrent_in` | statistics of the incoming weights *onto* $i$ (**row** $W_{rec}[i, :]$) + in/out relationship features |
| **Recurrent out** | `recurrent_out` | statistics of the outgoing weights *from* $i$ (**column** $W_{rec}[:, i]$) |
| **Activity** | `activity` | **label-free** PSTH + first-spike statistics on a reference split |

Each block is z-scored and concatenated; block-level **weighting** (`equal` per
block, `uniform` per feature, or `custom` with explicit `block_weights`) and
optional row normalisation are configurable
(`configs/analysis.yaml → representations`). The result is exposed as a
`RepresentationSpace` (`src/representations.py`) with pairwise distances and full
feature provenance. A detailed, feature-by-feature audit is in
[`AUDIT_REPRESENTATION.md`](AUDIT_REPRESENTATION.md).

### 4.1 Generic learned bias vs. genuine dynamical parameters

The `intrinsic` block is *only* populated by parameters that genuinely vary across
neurons. Two kinds are kept strictly apart:

- **`learned_bias`** — a generic learned per-neuron additive input (an excitability
  offset). It is a learned parameter but it is **not** a biophysical parameter and
  is never described as one (`GENERIC_LEARNED_FEATURE_NAMES`).
- **`tau_mem_ms`** — a genuine per-neuron *dynamical* parameter, emitted only when
  the model learns it per neuron (`neuron_param_mode="bias_tau"`;
  `DYNAMICAL_FEATURE_NAMES`).

Shared constants (threshold, reset, the base membrane time constant when it is not
learned per neuron) are **not** emitted, because a constant carries no per-neuron
information. In the default `bias` architecture there are therefore **no genuine
per-neuron dynamical parameters**: the `intrinsic` block contains exactly one
informative feature, `learned_bias`. An untrained network (all-zero bias) has an
empty `intrinsic` block, which is reported honestly. Ablation variants
`excitability_only` (learned bias) and `dynamical_only` (genuine dynamics) make the
distinction measurable.

### 4.2 Input-channel ordering (tonotopic features removed)

The input-connectivity features are **all invariant to a permutation of the 700
input channels**. The former "cochlear-weighted centre of mass" and "spread"
features are ordering-dependent: they are meaningful only if channel index
increases monotonically with characteristic frequency. That ordering cannot be
established from the local documentation or from the SHD HDF5 files (which carry
no channel metadata — verified), so **they are not part of the primary
representation**. They remain available behind the explicit opt-in
`representations.include_tonotopic_features: true` and are listed in
`TONOTOPIC_FEATURE_NAMES`.

### 4.3 Permutation invariance

The representation is explicitly tested for invariance to the arbitrary hidden-neuron
ordering (`src/permutation.py`): the hidden units are randomly relabelled, **all**
corresponding parameters (`W_in` columns, `W_rec` rows *and* columns, `W_out` rows,
per-neuron parameters) are permuted consistently, the permuted network is checked to
be functionally identical, and the representations are recomputed. Any feature that
changes is reported as permutation-sensitive. A deliberately index-based control
representation demonstrates that the detector fires. The structural representation
**passes** the test. The check also runs automatically during
`scripts/extract_representations.py` and is written to
`results/<tag>_permutation_invariance.json`.

### 4.4 Dimensionality (trained baseline, 256 neurons)

- Primary (structural) representation: **48** = `intrinsic` 1 (`learned_bias`) +
  `input_conn` 14 + `recurrent_in` 19 + `recurrent_out` 14.
- Activity block: **12**.
- `structural + activity`: **60**.

See `AUDIT_REPRESENTATION.md` §7 for the exact feature list.

---

## 5. Functional fingerprint (uses labels — the evaluation target)

For each hidden neuron we build a **class-conditioned response fingerprint** from
the neuron's activity on a **held-out** split (`val` by default). It measures what
the neuron *does*, as a function of the stimulus class:

| Feature set    | Content |
| -------------- | ------- |
| `class_rate`   | mean firing rate per class (20 values) |
| `class_count`  | per-class spike-count statistics |
| `class_latency`| per-class first-spike latency (when a minimum spike count is met) |

Features are column-standardised, and a `FingerprintSpace`
(`src/functional_fingerprint.py`) exposes pairwise distances under a configurable
metric. A **split-half reliability (noise ceiling)** estimate is computed from
`src.controls.reliability_suite` so a reader can see the *maximum* geometry–function
correlation that fingerprint noise would ever permit.

---

## 6. Hypothesis

**Primary (pre-registered).** The Spearman **Mantel correlation** between the
representation-space distance matrix and the functional-fingerprint distance matrix
is **positive and significant** under a permutation test.

**What the controls must rule out.** A positive result only means something if it
survives controls that could produce a spurious correlation (see §8).

---

## 7. Results

This section reports the outcome of the **baseline experiment** (256 recurrent
LIF hidden neurons, 20 training epochs on the official SHD training split,
speaker-aware validation split, seed 0) plus a **three-seed robustness study**.
All numbers are produced by the scripts in §14–§15 and are stored under
`results/` and `figures/`; nothing here is hand-entered.

> **Representation revision (2026-09-26).** The numbers in §7.2–§7.6 were produced
> with the *pre-audit* representation (v1). Following the feature audit
> ([`AUDIT_REPRESENTATION.md`](AUDIT_REPRESENTATION.md)) the representation was
> corrected: the tonotopic centre-of-mass features were removed from the primary
> representation, the recurrent in/out orientation was fixed, the learned bias was
> renamed `learned_bias` and separated from genuine dynamical parameters, and
> duplicate/mislabelled activity features were dropped. The primary
> representation is now 48-dimensional (§4.4). The quantitative values below are
> therefore a record of v1 and are expected to change on re-running the pipeline;
> they are retained for provenance, not as a claim about the current code.

> **Headline.** Hidden neurons occupy a structured, *label-free* space built only
> from the network's own parameters (`structural` representation:
> intrinsic + input-connectivity + recurrent in/out statistics — **no data, no
> labels**). Distances in that space significantly predict the neurons'
> independent functional fingerprints (class-conditioned held-out responses).
> The effect holds across seeds and survives a partial Mantel test that controls
> for firing rate.

### 7.1 The trained network (context)

The classifier is a means to an end, not the object of study, but the
representation analysis is only meaningful if the hidden layer is alive.

| Quantity | Baseline (seed 0) |
| -------- | ----------------- |
| Train / validation / test accuracy | 0.636 / 0.529 / 0.614 |
| Hidden mean firing rate | 173.3 Hz |
| Hidden rate range | 36.9 – 356.4 Hz |
| Silent neurons | **0.00** |
| Learnable parameters | 250 132 |

No silent neurons and no degenerate rates, so the hidden layer carries usable
signal for the representation study.

### 7.2 Primary result — geometry vs. function

The pre-registered statistic (§6) is the Spearman Mantel correlation between the
structural-representation distance matrix and the functional-fingerprint distance
matrix.

| Statistic | Value |
| --------- | ----- |
| Mantel Spearman *r* | **0.349** |
| Permutation *p* (10 000 perms) | **0.000999** |
| Effect size *z* | **10.75** |
| Null mean / std | −0.001 / 0.033 |
| Neuron pairs compared | 32 640 (256 neurons) |

**Interpretation.** *r* = 0.349 is far outside the permutation null
(*z* ≈ 10.7). Neurons that are close in the parameter-derived representation
space genuinely tend to have similar functional responses. This directly answers
the project's central question in the affirmative.

### 7.3 How strong could it be? (noise ceiling)

The fingerprint is measured on a finite held-out set, so it is itself noisy. A
split-half reliability estimate bounds any achievable correlation:

| Statistic | Value |
| --------- | ----- |
| Fingerprint matrix reliability (*r*) | **0.979** |
| Attenuation factor (√ceiling) | 0.989 |

Because the target is ~98 % reliable, the observed *r* = 0.349 is **not** an
artifact of a noisy fingerprint — the ceiling is high and the correlation is
genuinely below it.

### 7.4 Is it just firing rate?

A trivial baseline is "a single firing rate per neuron". We report both a
rate-only control and a **partial Mantel** that partials rate-distance out of the
structural representation.

| Variant | Mantel *r* | *z* | Partial *r* (ctrl. rate) |
| ------- | ---------- | --- | ------------------------ |
| `rate_only` | 0.419 | 18.2 | 0.005 |
| **`structural_full` (primary)** | **0.349** | **10.7** | **0.312** |
| `connectivity_only` | 0.306 | 9.6 | 0.265 |
| `intrinsic_only` | 0.330 | 11.7 | 0.346 |
| `input_conn_only` | 0.291 | 9.4 | 0.320 |
| `recurrent_only` | 0.213 | 7.1 | 0.135 |
| `activity_only` | 0.615 | 21.2 | 0.500 |
| `structural + activity` | 0.477 | 14.6 | 0.392 |

Two things stand out:

1. **Rate alone is cheap but shallow.** `rate_only` has a high raw *r* = 0.419,
   but once rate-distance is partialled out it collapses to **0.005** — as
   expected, since it *is* rate.
2. **Structure is not rate.** The structural representation keeps
   **partial *r* = 0.312** after controlling for rate (*p* = 0.000999). So the
   geometry carries functional information **beyond** a simple firing-rate code.
   Every structural sub-block contributes something on its own, and the effect
   is strongest when structure and activity are combined (0.477).

### 7.5 Negative controls behave as required

| Control | Mantel *r* | *p* | Expected |
| ------- | ---------- | --- | -------- |
| `random_null` (5 repeats) | ≈ 0.00 (z ≈ 0) | n.s. | null ✓ |
| `shuffled_control` (neuron rows permuted) | 0.019 | 0.26 | null ✓ |
| `fingerprint_descriptive` (circular) | 1.000 | — | upper bound only, flagged |

The chance and shuffle controls are indistinguishable from zero, so the primary
effect is **not** an artifact of the pipeline or of dimensionality.

### 7.6 Multi-seed robustness

A single run cannot separate signal from initialization luck, so the full
pipeline (train → extract → geometry) was rerun at seeds {0, 1, 2} with the
train/validation split **held fixed** (`data.split_seed = 0`).

| Seed | Structural Mantel *r* | *z* | *p* | Partial *r* (ctrl. rate) | Test acc. |
| ---- | --------------------- | --- | --- | ------------------------ | --------- |
| 0 | 0.349 | 10.75 | 0.000999 | 0.312 | 0.614 |
| 1 | 0.248 | 7.82 | 0.000999 | 0.195 | 0.624 |
| 2 | 0.296 | 9.60 | 0.000999 | 0.221 | 0.669 |
| **mean ± std** | **0.298 ± 0.051** | — | all significant | **0.242 ± 0.061** | **0.636 ± 0.030** |

**Interpretation.** The effect is **positive and significant in every seed**
(*p* = 0.000999 each) and never collapses, while the chance-level controls stay
at zero. The rate-controlled effect also remains significant across all seeds.
Variability is modest (*r* spread ≈ 0.05), matching the fingerprint reliability
spread (0.960 – 0.979).

### 7.7 Summary of the answer

| Question | Answer |
| -------- | ------ |
| Do neurons live in a structured space? | Yes — a label-free, parameter-derived representation exists and is non-degenerate. |
| Does its geometry reflect functional organization? | Yes — Mantel *r* = 0.298 ± 0.051, *p* = 0.000999 at every seed. |
| Is it just firing rate? | No — partial Mantel *r* = 0.242 ± 0.061, significant at every seed. |
| Are the controls clean? | Yes — random/shuffle null ≈ 0; circular control flagged and excluded. |
| Reproducible? | Yes — same qualitative result across seeds; `uv run pytest` green. |

**Caveats.** This is a proof of concept on SHD (speech digits, 20 classes). The
effect is moderate (*r* ≈ 0.3), not a claim of a perfect geometry → function map;
the fingerprint noise ceiling (§7.3) bounds headroom. Only 3 seeds were run by
default (`multiseed.seeds`), and only the structural + activity blocks with a
single architecture (256 hidden neurons) were studied so far. See §18 for the
deferred extensions and open questions.

---

## 8. Controls

All controls run through the *identical* geometry pipeline
(`src/controls.py → default_variants`), so they are directly comparable:

| Variant | Purpose |
| ------- | ------- |
| `random_null`        | shuffled neuron labels — null distribution |
| `rate_only`          | trivial baseline: does a single firing rate already explain everything? |
| `activity_only`      | label-free activity geature block alone |
| `connectivity_only`  | structural (weight) blocks alone |
| `structural_activity`| structural + activity |
| per-block & leave-one-block-out | which representation content matters |
| `fingerprint` (circular) | sanity check — **never reported as a result** |
| **partial Mantel** controlling for firing-rate distance | is the effect more than rate? |
| **before/after learning** | untrained vs trained network in the same architecture |

---

## 9. Installation (using `uv`)

This project uses [`uv`](https://docs.astral.sh/uv/) exclusively (no Conda).
Python is pinned to **3.11** via `.python-version`.

### 9.1 Install `uv`

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# macOS / Linux / WSL
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 9.2 Create the virtual environment

```powershell
uv venv --python 3.11
```

This creates `.venv` with a managed CPython 3.11 interpreter.

> **Windows/Anaconda note.** If `uv` accidentally picks a Conda interpreter,
> force the managed build: `$env:UV_PYTHON_PREFERENCE = 'only-managed'` before
> creating the venv.

### 9.3 Install dependencies

CUDA-enabled PyTorch is installed from the dedicated PyTorch index declared in
`pyproject.toml` (`[[tool.uv.index]]` → `https://download.pytorch.org/whl/cu126`,
routed only for `torch`), so a **CUDA** build is selected rather than the CPU-only
wheel.

```powershell
uv sync --all-groups
```

> The `torch` CUDA wheel is large (~2.5 GB). If the download is interrupted,
> `uv sync` can be re-run; see §17 for a resumable manual fallback.

> **Assumption (RTX 3060, driver CUDA 13.4).** The default index is `cu126`. If a
> driver cannot run cu126 wheels, change the URL to `cu121` in `pyproject.toml`,
> then `uv lock && uv sync`.

### 9.4 Verify PyTorch and CUDA

```powershell
uv run python --version
uv run python -c "import torch; print('torch', torch.__version__)"
uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Expected for an RTX 3060 machine: `CUDA available: True` and the GPU name printed.

### 9.5 Minimal import test

```powershell
uv run python -c "import numpy, scipy, pandas, matplotlib, h5py, sklearn, yaml; import src.model, src.data, src.neurons, src.training, src.evaluation, src.representations, src.functional_fingerprint, src.geometry_analysis, src.controls; print('imports OK')"
```

---

## 10. Project layout

```
neuron_as_vector/
├── configs/
│   ├── baseline.yaml        # training configuration
│   └── analysis.yaml        # representation / fingerprint / geometry / figures
├── scripts/                 # thin CLI entry points (uv run python scripts/…)
│   ├── _common.py           # shared CLI, config, device, dataset, model helpers
│   ├── _pipeline.py         # representation + fingerprint construction (leakage boundary)
│   ├── train.py             # stage 4  — train a model
│   ├── evaluate.py          # stage 4b — accuracy / confusion
│   ├── extract_representations.py  # stage 5
│   ├── run_geometry_analysis.py    # stages 6–7
│   └── make_figures.py             # stage 8
├── src/                     # the analysis code-base (namespace package)
│   ├── data.py              # SHD loading, binning, splits, synthetic data
│   ├── model.py             # recurrent LIF SNN
│   ├── neurons.py           # per-neuron feature blocks
│   ├── representations.py   # RepresentationSpace (the object under study)
│   ├── permutation.py       # hidden-neuron permutation-invariance test
│   ├── functional_fingerprint.py
│   ├── geometry_analysis.py # Mantel / kNN analysis
│   ├── controls.py          # null controls, ablations, reliability
│   ├── training.py          # surrogate BPTT loop
│   ├── evaluation.py        # metrics / activity collection
│   ├── visualization.py     # Figures 1–6
│   └── utils.py             # config, seeding, device, I/O
├── tests/                   # fast synthetic tests (no downloads)
├── pyproject.toml
└── README.md
```

---

## 11. Smoke tests (no download required)

Everything below runs on a **synthetic** dataset, so you can validate the full
numerical pipeline before touching SHD. Run the test suite first:

```powershell
uv run pytest
```

Then a tiny end-to-end training smoke test:

```powershell
uv run python scripts/train.py --config configs/baseline.yaml --synthetic `
    --override model.n_hidden=64 --override model.n_bins=100 `
    --override run.synthetic_n_samples=200 --override train.epochs=2
```

> In synthetic mode the model input/readout dimensions are **auto-aligned** to the
> synthetic channel/class counts, so you only need to set the network size you
> want. `--override` is a *repeatable* flag: pass one `--override KEY=VALUE` per
> leaf (space-separated extra keys are not accepted).

---

## 12. Downloading SHD

By default the first real run downloads SHD into `data/`:

```powershell
uv run python scripts/train.py --config configs/baseline.yaml
```

To use a local copy instead, set `paths.data_dir` to a folder containing the SHD
HDF5 files (or set `data.download: false`).

> **The `zenkelab.org` host is often very slow** (tens of kB/s), so the in-process
> download can stall. If it does, fetch the archives with a resumable downloader
> and extract them into `data/` yourself:
>
> ```powershell
> foreach ($f in 'shd_train.h5.zip','shd_test.h5.zip') {
>     curl.exe -L -C - --retry 10 --retry-delay 5 --retry-all-errors `
>         -o "data\$f" "https://zenkelab.org/datasets/$f"
> }
> uv run python -c "import zipfile;[zipfile.ZipFile(f'data/{z}').extractall('data') for z in ['shd_train.h5.zip','shd_test.h5.zip']]"
> ```
>
> The loader auto-detects the SHD HDF5 layout and picks up the speaker metadata,
> so the speaker-aware validation split (§2) works out of the box.

**Debug mode** (small subset, fast):

```powershell
uv run python scripts/train.py --config configs/baseline.yaml --debug
```

---

## 13. Training

Real baseline run (256 hidden neurons, 20 epochs):

```powershell
uv run python scripts/train.py --config configs/baseline.yaml
```

Outputs land in `checkpoints/` (`baseline.pt`) and `results/`
(`baseline_train_summary.json`, `baseline_history.{json,csv}`). A **circuit-health
diagnostic** is printed (mean hidden firing rate, silent-neuron fraction) because a
silent hidden layer would make the whole study vacuous.

Evaluate a checkpoint:

```powershell
uv run python scripts/evaluate.py --config configs/analysis.yaml --tag baseline
```

---

## 14. Analysis — from neurons to geometry

**Stage 5 — extract representations and fingerprints:**

```powershell
uv run python scripts/extract_representations.py --config configs/analysis.yaml
```

Writes the neuron representation space, the functional fingerprint, and a
fingerprint reliability (noise-ceiling) estimate under `results/`.

**Stages 6–7 — geometry–function analysis and controls:**

```powershell
uv run python scripts/run_geometry_analysis.py --config configs/analysis.yaml
```

Writes the headline ablation table (`baseline_ablation.csv`/`.json`), the full
per-variant analyses (`baseline_geometry_analyses.json`), the before/after
comparison (`baseline_before_after.json`) and the figure data
(`baseline_figure_data.npz`, `baseline_figure_bundle.json`).

---

## 15. Reproducing the figures

**Stage 8 — render Figures 1–6** from the saved bundle (no model, no data needed):

```powershell
uv run python scripts/make_figures.py --config configs/analysis.yaml
```

Figures are written to `figures/` in the configured formats.

| Figure | File | Question it answers |
| ------ | ---- | ------------------- |
| 1 | `figure1_training_curves` | did the network actually learn? |
| 2 | `figure2_representation_pca` | where do hidden neurons live? |
| 3 | `figure3_distance_correlation` | **geometry vs function (primary)** |
| 4 | `figure4_knn_effect` | are neighbours functionally similar? |
| 5 | `figure5_ablation_summary` | which representation content matters? |
| 6 | `figure6_fingerprint_heatmap` | what does the fingerprint look like? |

---

## 16. Multi-seed robustness study

A single training run cannot separate a real effect from initialisation luck.
The multi-seed driver reruns the *whole* pipeline (train → extract → geometry)
at several seeds and aggregates the pre-registered primary metric:

```powershell
uv run python scripts/run_multiseed.py --config configs/analysis.yaml
```

- Seeds are configured by `multiseed.seeds` (default `[0, 1, 2]`). Seed 0
  **reuses the existing `baseline` tag**; further seeds use `<base>_s<seed>`.
- The train/validation split is held **fixed** across seeds via
  `data.split_seed`, so the spread reflects model/optimisation variability
  rather than a moving evaluation target.
- Stages are **resume-friendly**: any stage whose output already exists is
  skipped (`multiseed.skip_existing`).
- Outputs: `results/multiseed_summary.json` / `.csv` (per-seed rows + mean ± std
  for the primary Mantel r, the partial Mantel r controlling firing rate,
  fingerprint reliability and test accuracy) and
  `figures/multiseed_forest.png`/`.pdf`.

---

## 17. Resumable PyTorch install (Windows fallback)

If `uv sync` keeps aborting on the large CUDA wheel, download it with a resumable
downloader and install the local file:

```powershell
$u = 'https://download.pytorch.org/whl/cu126/torch-2.14.0%2Bcu126-cp311-cp311-win_amd64.whl'
$out = "$PWD\.wheels\$(Split-Path $u -Leaf)"
New-Item -ItemType Directory -Force -Path (Split-Path $out) | Out-Null
for ($i=1; $i -le 200; $i++) {
    if ((Test-Path $out) -and (Get-Item $out).Length -ge 2602751531) { break }
    curl.exe -L -C - --retry 5 --retry-delay 5 --retry-all-errors -o $out $u
}
uv pip install $out
uv sync --all-groups
```

---

## 18. Reproducibility notes & assumptions

- **Staging.** The project is built and validated in strict stages:
  (1) environment → (2) synthetic smoke test → (3) tiny SHD test → (4) baseline
  training → (5) representation extraction → (6) geometry → (7) controls →
  (8) multi-seed final run. No long training happens before the representation
  and analysis code are verified. Stage (8) is driven by `scripts/run_multiseed.py`
  (§16).
- **Seeding.** All randomness is seeded (`seed` in the configs); `set_seed`
  enables deterministic CuDNN where possible. The train/validation split uses a
  separate `data.split_seed` (default: follow `seed`) so a multi-seed study can
  hold the split fixed while varying initialisation/optimisation.
- **Paths.** All paths use `pathlib` and are rooted at the project directory; no
  hard-coded home directory and no fragile relative paths.
- **Documented assumptions.**
  1. The spec's `neuron_representation.py` is implemented as
     `src/representations.py`.
  2. **No channel ordering is assumed.** The input-connectivity features are all
     invariant to a permutation of the 700 channels. The tonotopic centre-of-mass
     and spread features are excluded from the primary representation because no
     local documentation or data establishes the channel ordering; they are
     opt-in only (`representations.include_tonotopic_features`, §4.2).
  3. Validation uses a **speaker-aware** split by default, with a stratified
     random fallback (§2).
  4. Default config values live in `configs/baseline.yaml` and
     `configs/analysis.yaml`; every scientifically important choice is
     overridable via `--override dotted.key=value`.
  5. CUDA PyTorch is installed from the `cu126` index for the RTX 3060
     (driver CUDA 13.4); switch to `cu121` if required (§9.3).
  6. **SHD file layout.** The official HDF5 files store spikes as a *group*
     `spikes/times` + `spikes/units` (ragged object arrays, one entry per sample)
     with speaker ids under `extra/speaker`. `src.data.parse_shd_h5` auto-detects
     this canonical layout (as well as flat and per-sample-group variants); the
     detected layout is recorded in each recording's metadata and in the saved
     JSON summaries.
- **No silent CPU fallback.** The code selects CUDA when available and *prints a
  warning* if CUDA is requested but unavailable; it never silently installs a
  CPU-only build to "make it work".
- **Learned neuron embeddings** (autoencoder / metric learning / contrastive /
  graph embedding) are a *deferred* extension and are only attempted once the
  first experiment runs reliably end-to-end.

---

## License

MIT (see `pyproject.toml`). SHD is distributed under CC BY 4.0; please cite
Cramer et al. (2020) as above.
