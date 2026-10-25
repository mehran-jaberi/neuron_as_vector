# VECTOR_V2_AUDIT.md

**Scope: audit only.** This document inspects the existing repository at commit `d43d9a4`
("neuron as vector baseline") and does not implement, tune, or change any scientific result.
Nothing in the codebase, configs, checkpoints, results, or figures was modified to produce it.

Audit date: 2026-09-27.
Environment: Python 3.11 (`uv`), torch 2.14.0+cu126, RTX 3060 Laptop (6 GB), Windows.

---

## 0. Executive summary

1. **What exists today.** A reproducible pipeline: recurrent LIF SNN (256 hidden, 700×2 ms)
   → trained on SHD with a 3-way speaker-disjoint FIT/DEV/PROBE split (official TEST untouched)
   → a **48-dimensional label-free structural representation** per hidden neuron
   (+12-d label-free activity representation) → functional fingerprints measured on PROBE
   → Mantel/kNN geometry analysis, CV prediction, permutation/shuffle/rewiring controls,
   3-seed replication. 127 tests pass. This is coherent, well-instrumented, and reusable.

2. **There is no configurable vector dimension `d` and no learned representation anywhere in
   this repository.** The only representation builder is deterministic (`NeuronRepresentation` →
   `NeuronRepresentationSet` → `RepresentationSpace`). No `d`, no latent encoder, no autoencoder,
   no residual, and no `nn.Linear` at all (grep-verified; the readout is a plain matrix multiply
   in `forward`).
   Consequently, the reported "≈2.5 GB VRAM around d=100" **cannot be attributed to code in this
   repo**; it must have come from a prototype that is not committed here. Section 3 reconstructs
   the most plausible mechanism and gives the measured memory model of everything that *is* here.

3. **Where VRAM actually goes today** (measured, Section 3): the dense binned input tensor
   `(B, T, n_input)` and, when `record=True`, the three recorded traces `(B, T, n_hidden)`
   materialised inside `RecurrentLIFSNN.forward`. Everything downstream — representations,
   fingerprints, distances, prediction, controls — is NumPy/CPU and negligible in VRAM.
   Measured peaks: training step at B=128 → **610 MiB**, at B=256 → **1.20 GiB**;
   recorded activity pass (`collect_activity`) at B=256 → **1.58 GiB peak / 1.67 GiB reserved**.

4. **The new `NeuronRecord` design is a natural, small extension of what exists**, but it needs a
   columnar (array-based) record bank, a dimension-budgeted deterministic structured encoder, a
   strictly label-free learned residual trained on FIT, and a memory contract. Section 6 gives the
   recommended architecture; Section 7 the memory strategy. No scientific conclusions are
   touched.

---

## 1. Current architecture

### 1.1 Model (`src/model.py`)

`RecurrentLIFSNN` — single hidden layer, current-based synapses, implicit-Euler LIF,
soft reset, linear leaky-integrator readout:

```
I_t = β·I_{t-1} + W_in x_t + W_rec s_{t-1} + b_hid     β = exp(-Δt/τ_syn) = 0.6703
V_t = α·V_{t-1} + (1-α)·I_t                            α = exp(-Δt/τ_mem) = 0.9048
s_t = Θ(V_t - θ)   (Heaviside forward / fast-sigmoid surrogate backward)
V_t ← V_t - θ·s_t  (soft reset; reset gradient detached)
O_t = ρ·O_{t-1} + W_out s_t + b_out
logits = mean | sum | last over {O_t}
```

| Quantity | Symbol | Shape | Notes |
|---|---|---|---|
| input weights | `w_in` | (700, 256) | column `j` = neuron `j` |
| recurrent weights | `w_rec` | (256, 256) | `w_rec[i, j]` = from `j` to `i` |
| readout weights | `w_out` | (256, 20) | |
| per-neuron bias | `b_hid` | (256,) | `neuron_param_mode="bias"` → emitted as `learned_bias` |
| self mask buffer | `self_mask` | (256, 256) | structural, relabelled by permutation code |

Selected baseline (`checkpoints/sweep_l2_0.pt`, seed 0; replicated by `nsb_seed1/2.pt`):
`readout_mode="sum"`, `l2_spikes=0`, healthy dynamics (~5 Hz), TEST acc 0.5998.
Checkpoint handling: `save()` stores `{"model_config", "state_dict", "extra"}`;
`load()` uses `torch.load(..., weights_only=False)`; `scripts/_common.load_checkpoint` and
`src/model.architecture_mismatches` hard-error on `n_input/n_hidden/n_output/n_bins/bin_ms`
disagreement between config and checkpoint. Multi-seed reuse is explicit via
`select_recipe.reuse_seed0_checkpoint`.

### 1.2 Data (`src/data.py`)

* Canonical SHD HDF5 (`shd_ragged` group layout) parsed into a sparse `SHDRecordings`
  (flat `times_ms` float32, `units` int16, `offsets` int64). Raw events only; no dense caching.
* Dense binning **per batch** via `batch_events_to_bins` → `(B, T, C)` float32 counts;
  allocated directly on the requested device when `device=` is passed.
* Splits (`make_train_dev_probe_split`): whole-speaker, seeded by `data.split_seed`
  (fixed at 0 for all model seeds). Realised: FIT 5922 / DEV 998 (spk 2) / PROBE 1236 (spk 6, 8)
  / official TEST 2264. DEV = model selection, PROBE = locked analysis target, TEST never read
  in the neuron-space stage.
* `iterate_batches` yields `{"x": (B,T,C), "y": (B,), "idx": (B,)}`; in every analysis path it is
  called with `device=None` (bins built on CPU, then copied to GPU by the caller).

### 1.3 Training (`src/training.py`)

Adam/adamw/sgd + cosine/step schedule, optional L1/L2 terms and the homeostatic
`l2_spikes` penalty; DEV-only model selection (accuracy or loss), early stopping;
history logs loss/acc/firing-rate/silent-fraction/grad-norm. **Training is the memory-critical
stage** because of BPTT through 700 timesteps (Section 3.1).

### 1.4 Configs

| File | Role | Status |
|---|---|---|
| `configs/baseline.yaml` | historical training config | superseded |
| `configs/baseline_repaired.yaml` | homeostasis-enabled training config | reproducible recipe |
| `configs/neuron_space_baseline.yaml` | **canonical current experiment** (+`select_recipe`) | primary |
| `configs/analysis.yaml` | legacy single-fingerprint analysis path | superseded by NSB |

All scripts support `--override dotted.key=value` (coerced; see `src/utils.parse_scalar`) and
`--synthetic`. Model and train blocks are dataclasses (`SNNConfig`, `TrainConfig`) with
`from_config`/`from_mapping` validation; analysis sections are read ad-hoc via `cfg.get_path`.

### 1.5 Tests

14 test modules / 127 tests, all passing. Coverage includes model dynamics and surrogate
gradients, readout mode equivalence (`sum == T·mean`), data layouts/splits, accumulator
consistency, representation orientation (row=in, column=out), channel-permutation invariance,
hidden-neuron permutation invariance, rewiring multiset preservation, geometry/Mantel
correctness (planted-structure tests), fingerprint reliability/leakage guards, prediction
fold-sharing, and config parsing. **These tests are the main safety net for V2.**

---

## 2. Current representation pipeline

### 2.1 The neuron object today (`src/neurons.py`)

`NeuronRepresentation` is a **dict-of-dicts of Python floats** per neuron:
`features[block][name] = float`, blocks:

| Block | Emitted features | Current count (bias-mode baseline) |
|---|---|---|
| `intrinsic` | `learned_bias` (only if it varies across neurons) | 1 |
| `input_conn` | 9 signed vector stats + 5 concentration stats | 14 |
| `recurrent_in` | 14 (row `w_rec[j,:]`) + 5 incoming/outgoing relationship stats | 19 |
| `recurrent_out` | 14 (column `w_rec[:,j]`) | 14 |
| **structural total** | | **48** |
| `activity` (label-free, FIT) | rate, std, silent frac, Fano, spike-time CV, temporal centre/dispersion/entropy, peak rate, active-bin frac, log-rate, first-spike latency | 12 |

Provenance discipline that must be preserved in V2:
* only *varying* per-neuron parameters are emitted; shared constants (θ=1, reset, base τ) are not;
* `learned_bias` is classified `generic_learned`, never "biophysical"; per-neuron τ (only in
  `bias_tau` mode) is `dynamical`; tonotopic channel features are opt-in only;
* `recurrent_in` = **row** `w_rec[j,:]`; `recurrent_out` = **column** `w_rec[:,j]`.

`NeuronRepresentationSet.to_matrix()` assembles `(n_hidden, n_features)` with a Python loop over
neurons calling `to_vector()` per neuron — O(n·f) Python overhead, an obvious V2 bottleneck for
large `n`, but currently trivial.

### 2.2 Metric space (`src/representations.py`)

`RepresentationSpace` = raw matrix + names + per-column `Standardizer` + block weighting
(`equal` 1/√k per feature, `uniform`, or `custom` block weights) + optional row L2 normalisation.
Distances are SciPy `pdist` → condensed vectors; `distances()` returns `(n, n)`.
Constant columns are detected and reported, never NaN-poisoned. **This layer is
dimension-agnostic and is directly reusable for V2 vectors.**

### 2.3 Activity collection (`src/evaluation.py`)

`collect_activity(model, rec, idx, record=True)` streams batches and
`ActivityAccumulator` keeps only sufficient statistics:

| Statistic | Shape | dtype | Purpose |
|---|---|---|---|
| `counts` | (N, H) | float64 | rates, Fano, silent frac |
| `psth` | (H, T) | float64 | label-free temporal features |
| `class_psth` | (C, H, T) | float64 | fingerprint (labels!) |
| `class_counts`, `class_n` | (C,H), (C,) | float64 | class rates |
| first-spike sums/counts | (H,), (C,H) | float64 | latency |

The design principle — *never materialise an `(N, T, H)` dataset tensor* — is already correct and
must be carried into V2. However `update()` currently converts each batch's `(B,T,H)` trace to
**float64 NumPy**, i.e. a 2× CPU copy per batch, and `class_psth` is float64 (28.7 MB at
C=20,H=256,T=700 — fine now, not at larger n/T/C).

### 2.4 Functional fingerprint (`src/functional_fingerprint.py`)

The independent evaluation target. Presets: `tuning` (20-d class-rate, **primary**),
`tuning_rate_normalized` (primary rate control), `tuning_with_latency` (secondary),
`temporal` (PSTH + centre + dispersion + latency, exploratory).
`FingerprintSpace` standardises columns and exposes the same condensed-distance API as the
representation. Split-half reliability is computed with a shared standardiser, matching metrics,
and a Spearman-Brown full-length ceiling. `uses_labels=True` and `leakage_warning` travel with the
object. **V2's "individual-stimulus neuronal responses on PROBE" primary target is an extension of
this module** (a new `stimulus_response` feature family), not a rewrite.

### 2.5 Geometry, prediction, controls

* `src/geometry_analysis.py`: Mantel Spearman on condensed distances, neuron-relabelling
  permutation null (vectorised via `_pair_index_matrix`), p-value resolution floor, neuron
  bootstrap CI, kNN effect sizes, rate-normalised/rate-matched/partial Mantel (partial explicitly
  secondary/exploratory), binned distance–distance curves. **All operate on `(n, f)` matrices —
  fully reusable for any `d`.**
* `src/prediction.py`: `RidgeCV` and kNN regression of fingerprints from representations,
  KFold **across neurons**, shared folds across representations, out-of-fold metrics + a
  distance-level Spearman. Dimension-agnostic; reusable.
* `src/controls.py` + `src/neuron_space_baseline.py`: canonical 9-representation table
  (rate-only, activity-only, input-only, recurrent-only, intrinsic-only, **structural**,
  structural+activity, random, neuron-shuffle), empty-selection skipping, IV-fold comparison.
* `src/rewiring.py`, `src/permutation.py`, `src/stress_test.py`: rewired-recurrent controls
  (multiset-preserving per row/column/global), hidden-neuron permutation invariance checks
  (structural features exactly invariant; activity invariant up to float32 threshold flips),
  and the 127-test safety net.

### 2.6 Canonical run sequence (`scripts/run_neuron_space_baseline.py`)

Per seed: load checkpoint (arch-guard) → `build_representation_bundle` (structural + label-free
activity on FIT) → `build_fingerprints` (3 presets on PROBE, one labelled pass) → `run_condition`
(every variant × Mantel/rate-controls/CV-prediction on shared folds) → before-learning model →
3 rewired controls → permutation-invariance checks → split-half reliability → per-seed artifacts
→ cross-seed aggregation + canonical table + figures. Official TEST is never read.

---

## 3. Current VRAM bottlenecks

### 3.1 Measured (RTX 3060 Laptop 6 GB, torch 2.14.0+cu126; `H=256, T=700, C=700`)

A throwaway probe (temp file, not committed) measured torch allocator peaks on the real code
paths. `MB` = MiB.

| Stage | B | allocated | **peak** | reserved |
|---|---|---|---|---|
| dense input `(B,700,700)` fp32 | 64 / 128 / 256 | — | 121 / 249 / 488 | — |
| `forward(record=False)` (same x alive) | 64 / 128 / 256 | +0.4…2 MB | **130 / 251 / 490** | 142 / 264 / 506 |
| `forward(record=True)` | 64 / 128 / 256 | — | **265 / 521 / 1032** | 274 / 528 / 1048 |
| **train step** (fwd+bwd+Adam, no record) | 64 / 128 / 256 | 141 / 261 / 500 | **315 / 610 / 1199** | 320 / 616 / 1206 |
| **`collect_activity(with_labels=True)`** | 128 / 256 | 18.4 | **801 / 1582** | **1120 / 1706** |

### 3.2 Tensor-by-tensor trace

**Model inference, `record=False`** — persistent: `x (B,T,C)`; per timestep:
`recurrent_input (B,H)`, `x_t @ w_in (B,H)`, `i_syn (B,H)`, `v (B,H)`, `s (B,H)`,
`o (B,O)`, plus accumulators `spike_count (B,H)`, `o_sum (B,O)`.
Peak ≈ `x` + a few `(B,H)` transients → at B=256 the **input tensor alone is 488 MiB (99 %)**
of the 490 MiB peak.

**Model inference, `record=True`** — additionally allocates, once, `rec_spikes/rec_v/rec_i`
`(B,T,H)` fp32 and `rec_o (B,T,O)`; the loop writes into them.
Extra = `3·4·B·T·H` = **525 MiB at B=256** + 14 MiB output trace → matches 1032−490.
This is the accidental `samples × time × neurons` class of tensor, and it is **deliberate here**
(voltage/current traces are needed for diagnostics/features) but must be budgeted.

**Training** — BPTT keeps saved activations per timestep (`s_prev`, `u=V−θ`, `|u|`,
surrogate numerator/denominator, reset operand, `x[:,t,:]` view, `s` for `w_out`, …).
Graph ≈ 700 MiB at B=256 (≈1 MiB/step ≈ 4 saved `(B,H)` tensors of 256 KiB); plus `x` 488 MiB
→ 1199 MiB. Training at the configured `batch_size=128` peaks at **610 MiB**; there is headroom
on the 6 GB card unless batch is raised above ~512.

**`collect_activity`** — measured 1582 MiB peak at B=256, which decomposes as
`x` (488) + new traces (539) + **the previous batch's traces still alive** (~543): the loop
rebinds `out = model(...)` while the old `out` binding is only released *after* the new forward
completes, and `hidden_v`, `hidden_i_syn`, `hidden_spikes` are then `.cpu().numpy()`-copied.
So this pass pays for two recorded trace sets at once. Lowering the recorded batch to 32–64
(or deleting/freeing traces per batch) removes ~1 GiB at B=256. There is no `empty_cache()`
or peak logging anywhere in the codebase.

**Everything downstream — CPU only.** `counts (N,H)` fp64 = 12 MB; `psth (H,T)` fp64 = 1.4 MB;
`class_psth (C,H,T)` fp64 = 28.7 MB; Mantel null `(n_perm,)` fp64 = 80 kB for 10 000 perms;
condensed distances `(n(n−1)/2)` = 0.26 MB per representation; `_pair_index_matrix (n,n)` int64
= 0.5 MB; prediction is sklearn on `(N, ≤60)` and `(N, 20)`. **The analysis layer has no VRAM
problem at all and scales with `n²` only through condensed distances, which stays small for
`n` up to a few thousand (n=4096 → 8.4 M pairs → 67 MB fp64).**

### 3.3 Where `~2.5 GB at d≈100` comes from — audit finding

* Grep-verified: no `d`, no learned encoder, no residual, no `nn.Linear` outside the LIF
  readout; the largest representation is 48-D (`structural`) / 60-D (`full`). The **current repo
  cannot reproduce a `d`-dependent 2.5 GB observation**; the measured high-water marks are
  610 MiB (training, configured batch 128) and 1.58 GiB (`collect_activity`, batch 256).
  2.5 GB is consistent with *either* a recorded pass at batch ≈ 400, *or* several recorded passes
  in one process with the allocator's reserved pool (reserved grew to 1.67 GiB in a single pass),
  *or* — most likely — a non-committed prototype.
* The prototype arithmetic that does produce ~2.5 GB at `d = 100` is a **float64 tensor with axes
  `neuron × sample × d`**: `256 × 5922 × 100 × 8 B = 1.21 GB` for the FIT split; two such tensors
  alive (e.g. a structured matrix and a residual matrix, or a value plus a standardised copy)
  = 2.4 GB, and >2.5 GB with any batch traces present. In fp32 the same tensor is 606 MB, so
  "≈2.5 GB around d=100" strongly suggests **two fp32 or one fp64 `(n_neurons, n_samples, d)`
  intermediate kept on the GPU**. Nothing in this repo builds such a tensor; V2 must forbid it
  structurally (Section 7).
* Secondary suspect: `(n_samples × T × d)` or `(n_neurons × T × d)` "feature" tensors if the
  temporal block of the record were naively stored per neuron **per sample**. Section 6/7 remove
  the need for both.

### 3.4 Forbidden-shape checklist for V2 (hard rules)

Any tensor whose shape multiplies two of `{n_neurons, n_samples, T, d}` is a bug unless it is
one of the three budgeted transients:

| Shape | Status |
|---|---|
| `(B, T, C)` input, `B ≤ 64` for recorded passes | budgeted, dominant |
| `(B, T, H)` recorded traces, `B ≤ 32–64`, freed per batch | budgeted |
| `(n_neurons, n_features)` representation/record matrices | fine (CPU) |
| `(n_neurons, n_samples)`, `(n_neurons, T)`, `(n_neurons, n_samples, d)`, `(n, n, d)`, `(n, T, d)` | **forbidden** (must be reduced streaming) |
| per-sample traces `(S, T, H)` materialised | forbidden (use accumulator) |

---

## 4. Reusable components

| Module / asset | Verdict | Notes for V2 |
|---|---|---|
| `src/model.py` `RecurrentLIFSNN` | **reuse, extend** | single-layer today; dynamics/readout/surrogate unchanged. Add `n_layers` later; keep arch-mismatch guard |
| `src/data.py` | **reuse** | event parsing + streaming batched binning + speaker-disjoint splits are correct; pass `device=` only with small B |
| `src/evaluation.ActivityAccumulator` | **reuse, optimise** | right sufficient-statistics pattern; switch float64→float32, add per-layer + label-free temporal stats, stream into the record |
| `src/functional_fingerprint.py` | **reuse, extend** | presets/reliability/leakage metadata; add `stimulus_response` family for the new primary target |
| `src/geometry_analysis.py` | **reuse as-is** | completely dimension-agnostic `(n, f)` in / distances out; Mantel null/p-floor/bootstrap/rate controls all keep their meaning |
| `src/prediction.py` | **reuse as-is** | CV across neurons, shared folds |
| `src/representations.py` | **reuse, adapt** | standardiser/weighting/`pdist` reused; add a vector-space adapter for `z_i` |
| `src/controls.py`, `src/rewiring.py`, `src/permutation.py` | **reuse** | variant suite, rewired multiset controls, invariance checks apply verbatim to any new `z`; extend invariance to the learned residual |
| `src/neuron_space_baseline.py`, `src/nsb_figures.py`, `scripts/run_neuron_space_baseline.py` | **reuse pattern** | canonical table / per-seed artifacts / aggregation are the template for V2 runs |
| `src/utils.py` `Config`, `Standardizer`, `sanitize_features`, `set_seed` | **reuse** | |
| `scripts/_common.py` (paths, overrides, checkpoint guard, splits) | **reuse** | add `memory`/`vector` config plumbing |
| `tests/` (127) | **reuse, extend** | add dimension/permutation/leakage/memory-budget contract tests |
| Checkpoints (`sweep_l2_0.pt`, `nsb_seed1/2.pt`) | **reuse** | same circuits; V2 analysis must be run against these unchanged models |

---

## 5. Code that should be replaced or refactored

| # | Target | Why | Action |
|---|---|---|---|
| 1 | `NeuronRepresentation` micro-object (dict of dict of Python float) | not fit for one unified record; slow per-neuron assembly; no temporal/context blocks | replace with array-based `NeuronRecord` / columnar `NeuronRecordBank` (§6); keep the *feature functions* |
| 2 | `NeuronRepresentationSet.to_matrix` Python loop | O(n) Python per neuron; blocking for large n | vectorised block assembly from arrays |
| 3 | `ActivityAccumulator.update` float64 copies of `(B,T,H)` | 2× CPU cost/memory per batch | accumulate float32; `class_psth` float32 |
| 4 | `collect_activity` batch/retention behaviour | previous batch's recorded traces stay alive into the next forward (measured +543 MiB at B=256); hard-coded default B=256 | per-batch explicit free, `torch.inference_mode()`, configurable small batch for recorded passes, optional `record=False` path using `spike_count` only |
| 5 | `diagnostics.measure_dynamics` | materialises v/i_syn/i_in/i_rec/x `(B,T,H)` then flattens to CPU to subsample | compute summaries/histograms streaming |
| 6 | `SNNConfig` / forward loop | single hidden layer hard-wired | add `n_layers`; layers as stacked parameters; per-layer record/accumulator; keep default = 1 |
| 7 | Checkpoint format | `torch.load(weights_only=False)` pickling risk; arch keys do not include `n_layers` | add `n_layers` to `ARCHITECTURE_KEYS`; prefer `weights_only=True` with a plain-dict config |
| 8 | Config schema | no place for `d`, composition, precision, or memory bounds | add top-level `vector:` and `memory:` blocks (defaults preserve current behaviour) |
| 9 | Legacy paths (`analysis.yaml`, `extract_representations.py`, `run_geometry_analysis.py`) | duplicate/older metric definitions than the canonical NSB path | keep only as compatibility shims; V2 grows from the NSB path |
| 10 | Memory instrumentation | no allocator logging/limits anywhere | add per-stage peak logging + a token budget assert (cheap, prevents repeats of the d-scaling incident) |

Explicitly preserved: LIF equations, readout modes, splits, fingerprint definitions,
PRIMARY metric, control suite, and all existing results (this audit changes no result).

---

## 6. Recommended architecture for the unified `NeuronRecord`

### 6.1 Concept

Per hidden neuron `i`, one unified label-free record (stored **columnar** across neurons for
vectorisation):

```
NeuronRecord_i = {
  intrinsic        : learned per-neuron params actually present (bias; per-neuron τ if mode allows)
  input_conn       : w_in[:, i]  (full 700-d column) + summary statistics (current 14-d block)
  rec_in           : w_rec[i, :] (full row, optional top-k sparse view) + stats + in/out relation
  rec_out          : w_rec[:, i] (full column, optional top-k) + stats
  activity         : label-free FIT responses: per-sample counts (S,), rate/Fano/silent, PSTH (T,)
  temporal         : label-free FIT dynamics: coarse PSTH, temporal centre/dispersion/entropy,
                     spike-time CV, first-spike latency, optional autocorrelation timescale
  network_context  : label-free graph context: normalised in/out strength ranks, degree,
                     participation/reciprocity, layer index (when n_layers > 1)
  provenance       : {uses_labels: False, splits: {activity: fit}, checkpoint, model_seed,
                      n_hidden, n_input, n_bins, bin_ms, arch_hash}
}
```

Design rules:
* A record may hold **high-dimensional raw vectors** (700-d input column, 256-d rows/cols) —
  they are `O(n · n_input)` and negligible; what must never exist is `(n, S, T)` raw activity.
* Raw activity is reduced **streaming** into the activity/temporal blocks (reuse the accumulator);
  per-sample counts `(S, n)` are allowed (S=5922, n=256 → 6 MB fp32).
* `provenance.uses_labels = False` is asserted at construction; the fingerprint module remains the
  only label-consuming component.

### 6.2 From record to vector `z_i ∈ R^d` (configurable)

```
z_i = [ S_i (deterministic structured, d_struct)  ⊕  R_i (learned residual, d_res = d − d_struct) ]
```

**Deterministic hierarchical structured encoder `S_i`** (no data fitting, no labels; identical for
the same record and architecture):
1. *Level 0 — block summaries*: the existing 48-d structural set (+ label-free activity/temporal/
   context summaries as configured). Reproduces today's representation exactly at `d = |level 0|`.
2. *Level 1 — within-block detail*: for each raw block, a **fixed-order** deterministic expansion
   (e.g. sorted-|·| quantiles, top-k signed weights, histogram-of-|weights| bins, coarse PSTH bins).
   Budget is assigned per block by a documented rule.
3. *Level 2 — fixed projection*: a **seeded, data-independent** random orthogonal/Gaussian
   projection of the assembled raw record to fill the remaining structured budget. Deterministic
   given the seed; no fitting on any data.
Each output coordinate carries a `feature_name` and a block/source provenance entry, so
`RepresentationSpace`, block weighting, and ablation analysis continue to work unchanged.

**Learned residual `R_i`** (only for the residual budget):
* Small MLP (`record embedding → hidden ≤ 256 → d_res`), trained **only** on FIT data with a
  **label-free** objective. Recommended candidates, in order of conservatism:
  (a) reconstruct masked parts of the record from the rest (self-supervised),
  (b) predict held-out FIT per-sample activity statistics from structure,
  (c) contrastive similarity between neurons over FIT response profiles *without class labels*.
* Hard contract, enforced in code: inputs are record tensors + FIT label-free statistics only.
  Raise if a labels array, a `FingerprintSpace`, or a split named `probe`/`test` is passed.
  Extend the existing `uses_labels` / `fingerprint_split_leakage_guard` pattern to the trainer.
* Determinism and reproducibility: seed recorded; residual checkpoint saved and referenced in the
  same `provenance` block; the residual is re-fitted per model seed (never across seeds).

**Dimension policy** (`vector.d` in config):
`d ≤ |level 0|` → deterministic prefix/selection of level 0 (documented ordering);
`|level 0| < d ≤ |level 1|` → level 0 + level 1 budget;
`d > |level 1|` → level 0 + level 1 + fixed projection, and the residual (if enabled) takes
`d_res = d − d_struct` capped by config; `residual.enabled=false` gives a fully deterministic
`d`-vector at any size. `d` never affects any GPU tensor shape (see §7).

### 6.3 Evaluation side (unchanged interfaces)

`z` is consumed as an `(n, d)` matrix, so the existing evaluation stack applies directly:

* Primary (new): **individual-stimulus responses on held-out PROBE** → extend
  `FINGERPRINT_FEATURE_SETS` with a `stimulus_response` family producing a per-neuron response
  matrix over PROBE stimuli (reliability-audited; standardisation as today). The current
  `tuning` (class-rate) preset remains available and comparable to prior results.
* Secondary: `temporal` preset (already implemented).
* Exploratory: `tuning`/`class_rate` class-conditioned tuning.
* Controls/graph metrics/prediction: unchanged (`rate_only`, `activity_only`, random,
  neuron-shuffle, rewired, before-learning, shared-fold CV).

### 6.4 Proposed module layout (no code written yet)

```
src/neuron_record.py     NeuronRecord / NeuronRecordBank (columnar), build/validate/serialise,
                         provenance + label-free guards, permutation-invariance of records
src/structured_vector.py deterministic hierarchical encoder (block budgets, level 0/1/2),
                         feature provenance per output coordinate
src/residual.py          label-free residual MLP + trainer (FIT only) + leakage guards
src/neuron_vector.py     NeuronVectorConfig(d, composition, precision), build z, adapter to
                         RepresentationSpace / distance APIs
src/memory_guard.py      token-budget assert, per-stage peak-memory logging, batch policy helpers
scripts/build_neuron_vectors.py  new canonical entry point (mirrors run_neuron_space_baseline)
tests/test_neuron_record.py, test_structured_vector.py, test_residual.py,
      test_neuron_vector_contract.py   (dimension, invariance, leakage, memory budget)
```

Multi-layer readiness: add `layer` to the record and accumulate per layer; `permute_hidden_neurons`
and the accumulator gain a layer axis; `w_rec` becomes `(L, H, H)`. This is a later refactor —
V2 should *not* implement it now, only avoid hard-coding single-layer assumptions in the new
record code (e.g., do not assume `w_rec` is 2-D in the record builder; keep a `layer` field).

---

## 7. Recommended memory strategy

### 7.1 Budget formula (fp32, bytes)

```
per recorded batch:   4·B·T·C   (dense input)  +  4·B·T·(k·H + n_output)   (k traces)
per BPTT step (train):~4·B·H·(≈4 saved tensors); graph ≈ 700 steps · 4·B·H·4·4 bytes
never:                4·N·S·d, 4·N·T·d, 4·N·N·d, S·T·H materialised
```

Concrete policy for this project (`T=700, C=700, H=256`):

| Pass | Recommended B | Predicted peak (formula) | Measured today |
|---|---|---|---|
| training (BPTT) | 128 (current default) | ~0.6 GiB | 610 MiB ✓ |
| evaluation `record=False` | 256 | 0.5 GiB | 490 MiB ✓ |
| recorded activity/fingerprint | **32–64** (new default) | 0.13–0.26 GiB | 1.58 GiB at B=256 |
| record-bank assembly, distances, CV | CPU | — | — |
| residual MLP training | neurons as batch (≤ few thousand) | < 50 MiB | — |

At `n` and `d` up to 10× current values these numbers barely move: **no GPU term scales with n or
d**; only the CPU-side `(n, d)` matrices and `n²` condensed distances scale (n=4096, d=1024 →
33 MB fp64; distances 67 MB), both safe.

### 7.2 Implementation rules

1. **Streaming only.** Activity/temporal blocks are reduced per batch into accumulators
   (counts, PSTH, sums). Delete `out` and `.cpu()` copies immediately; never let two recorded
   batches coexist (fixes the measured +543 MiB duplication).
2. **dtype discipline.** float32 for all accumulation and for `(B,T,·)` traces; float64 only for
   small `(n,f)`/`(n,d)` matrices and distance statistics (numerical stability of Mantel/z-scores).
   `class_psth` float32 (halves current 28.7 MB).
3. **`torch.inference_mode()`** for collection; `record=True` only when voltage/current is truly
   needed by a configured block; otherwise accumulate from `spike_count` and per-step spikes.
4. **Batch policy by stage**, in config: `memory.train_batch_size`, `memory.eval_batch_size`,
   `memory.record_batch_size` (default 32–64), `memory.record_voltage` (bool).
5. **Token budget guard.** Before allocating the dense input, assert
   `B·T·C ≤ memory.max_input_tokens` (e.g. 32 M ≈ 128 MiB fp32) and raise with a clear message.
   Log `torch.cuda.max_memory_allocated()` per stage into the run artifacts.
6. **Deterministic structured encoder is data-free**, so it can be built on CPU from the
   checkpoint alone; the residual trainer is the only learned part and is `n × d`-sized.
7. **Checkpointing the residual** reuses `RecurrentLIFSNN.save` conventions
   (`config + state_dict + extra`), and every `z` artifact records the residual checkpoint hash.
8. **Regression tests** (add to the 127): `z` shape `(n, d)` for a sweep of `d`;
   exact invariance of `S` and (up to tolerance) of `z` under hidden-neuron permutation;
   leakage guard raising on labels/`probe`/`test`; a CUDA memory-budget test with a tiny config
   asserting `max_memory_allocated` below a cap; determinism across repeated builds.

---

## 8. Scalability assessment against the roadmap

| Requirement | Today | Needed |
|---|---|---|
| arbitrary `n_hidden` | **works** (`SNNConfig`, generic loops; configs just fix 256) | sweep tests; vectorised record/feature assembly (drop the per-neuron Python loop) |
| arbitrary `d` | **absent** | `NeuronVectorConfig` + structured encoder + optional residual (§6) |
| representation composition | **partly** (block select/include/exclude/weighting) | expose as first-class config for the `d` budget; add new blocks (temporal, context) |
| numerical precision | float32 traces / float64 stats (implicit) | explicit `precision` config; float32 accumulation |
| model architecture | single hidden layer, linear readout | `n_layers` parameterisation (stack `w_rec`, per-layer traces/records) — later stage |
| multiple hidden layers | **absent** | interfaces above must carry `layer` from day one |
| large sweeps on faster hardware | scripts are sequential, no memory logging | stage-level budgets + peak-memory artifacts; keep analysis CPU-side so GPU sweeps are not VRAM-bound |

---

## 9. What this audit deliberately did not do

* No representation was implemented; no `d` parameter was added; no learned residual was trained.
* No config, checkpoint, result, figure, or test was modified.
* No scientific conclusion was re-derived or revised; the existing 48-D results stand as recorded
  in `archive/documentation/NEURON_SPACE_BASELINE_REPORT.md` (structural Mantel r = 0.174 ± 0.049; weaker than
  rate-only r = 0.388 and activity-only CV R² = 0.711 — i.e. the current representation is
  reproducible but weak, which is exactly the starting point for the unified-record work).

**Audit complete. Stopping here as instructed. The next stage (design of `NeuronRecord` and the
`d`-budgeted vector) should wait for explicit approval.**