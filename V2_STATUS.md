# V2_STATUS.md

**Single current-status document for the implemented V2 "neuron as vector" architecture.**

Last updated: 2026-09-27. Authoritative sources for anything not covered here: the code
itself and the tests.

---

## 0. Documentation map (where everything lives)

| document | role |
|---|---|
| `README.md` | project overview + the canonical V1 / neuron-space-baseline scientific experiment and its recorded results |
| `VECTOR_V2_AUDIT.md` | the design audit behind the V2 architecture (measured memory model, forbidden tensors, design rules) |
| **`V2_STATUS.md`** (this file) | the implemented V2 architecture: configuration, record bank, structured encoder, learned residual, composition, tests, limitations |
| `archive/README.md` | what the archive is and the rules for using it |
| `archive/ARCHIVE_MANIFEST.md` | one row per archived artifact (runs and documentation), with reasons and dates |
| `archive/documentation/` | historical audits, baseline reports and per-stage reports (preserved verbatim) |
| `archive/intermediate_runs/` | smoke/debug checkpoints and run outputs (not active) |

Active documentation is deliberately only three files: `README.md`, `VECTOR_V2_AUDIT.md`,
`V2_STATUS.md`. Per-stage implementation details live in `archive/documentation/` and are
historical, not current status.

---

## 1. Architecture at a glance

```
frozen SNN checkpoint  +  label-free FIT recordings
                 |
                 v
        NeuronRecordBank                 <- source information (columnar, label-free)
                 |
    +------------+--------------+-----------------------------+
    |                           |                             |
    v                           v                             v
structural blocks      temporal block (coarse        activity block
(intrinsic, input,     FIT PSTH bins; only when       (12 summaries + the
 rec in/out)           requested)                    (n_fit_samples, n)
                                                     stored counts)
    |                           |                             |
    |                           |                             v
    |                           |              FunctionalResponseSource
    |                           |              (fixed seeded projection of
    |                           |               the per-stimulus profile)
    |                           |                             |
    +-------------+-------------+-----------------------------+
                  v
      +----------+-----------+--------------------+
      |                      |                    |
      v                      v                    v
StructuredVectorEncoder   LearnedResidual (self-supervised, FIT only; may consume
      |                   the level-0/1 summaries, the raw-connectivity views, the
      |                   functional-response projection and the temporal block)
      v                      v
 z_structured (d_struct)   z_residual (d_residual)
      |                      |
      +----------+-----------+
                 v
      NeuronVector composition  ->  z = [z_structured, z_residual]   (d = d_struct + d_residual)
                 |
                 v
     (unchanged) geometry / prediction / controls consume an (n, d) matrix
```

Layer contracts that must not be mixed up:

| layer | consumes | produces | learned? | labels? |
|---|---|---|---|---|
| `NeuronRecordBank` | frozen model parameters + label-free FIT statistics | columnar source records | no | **no** |
| `FunctionalResponseSource` (`src/functional_response.py`) | a bank's stored `(n_fit_samples, n)` FIT counts | deterministic `(n, functional_source_dim)` | no (fixed projection) | **no** |
| `StructuredVectorEncoder` | a `NeuronRecordBank` | deterministic `(n, d_struct)` | no (no fitted state) | **no** |
| learned residual | a `NeuronRecordBank` (via a deterministic source view) | learned `(n, d_residual)` | yes (self-supervised) | **no** |
| `NeuronVector` composition | structured vectors + residual vectors | `(n, d)` | — | **no** |
| functional fingerprint / scientific evaluation | PROBE + class labels | evaluation metrics | — | yes (the only component allowed to) |
| capacity evaluation (`src/vector_capacity.py`) | a frozen `(n, d)` matrix + PROBE targets | Mantel/kNN/prediction/control metrics | — | yes (targets only, never the representation) |
| rate-confound / robustness (`src/rate_robustness.py`) | a frozen `(n, d)` matrix + PROBE **response-target variants** | the same Mantel/kNN/prediction/control metrics | — | yes (targets only) |

---

## 2. Configuration (`src/v2_config.py`)

Single configuration system (dataclasses on top of `src.utils.Config`); YAML, the existing
`--override dotted.key=value` arguments and Python construction all share one validation
path. Entry point:

```python
from src.v2_config import V2Config
v2 = V2Config.from_config(load_config("configs/neuron_space_baseline.yaml"))  # strict=False default
v2 = V2Config.from_config(cfg, strict=True, warn=False)                       # reject unimplemented requests
```

| section | fields (defaults) |
|---|---|
| `vector` | `d`, `structured_d`, `learned_residual_d` (defaults 48/48/0), `enabled_blocks` (the four structural blocks), `temporal_resolution` (10), `context_depth` (0), `functional_source_dim` (64), `functional_projection_seed` (0), `functional_source_normalization` (`raw`\|`neuron_centered`\|`neuron_zscored`), `residual.enabled` (false), `residual.source_functional_response` (false), `residual.source_temporal` (false), `residual.mask_mode` (`coordinate`\|`block`) |
| `memory` | `train_batch_size` (inherits `train.batch_size`), `eval_batch_size` (inherits `train.eval_batch_size`), `record_batch_size` (32), `activity_chunk_size` (32), `representation_chunk_size` (256), `device` (inherits `run.device`), `storage` (`cpu`\|`gpu`\|`memmap`), `mixed_precision` (false), `max_input_tokens` (150 000 000; `0` disables) |
| `precision` | `vector_dtype`, `activity_dtype`, `model_dtype` (`float32` \| `float16` \| `bfloat16`; aliases `fp32`/`fp16`/`bf16`) |
| `model` (existing + future keys) | `n_hidden`, `n_bins`, `bin_ms`, … plus the interface-only `model_type`, `n_layers`, `neurons_per_layer` |
| `experiment` (optional) | `dataset`, `split`, `n_fit`, `n_probe`, `seed` — all `null` means "derive from the existing blocks" |

**Invariant:** `d = structured_d + learned_residual_d` (enforced; at most one of the three
may be derived; `learned_residual_d = 0` is valid). Hard errors are raised for negative/zero
dimensions, unknown blocks, invalid batch/chunk sizes, unknown dtypes/devices/storage modes,
impossible precision/device combinations (e.g. half-precision model on CPU, `bfloat16` with
NumPy-backed CPU storage, GPU storage on a CPU device) and for dense-input tensors exceeding
`memory.max_input_tokens`. Requests for features that are still not implemented (multi-layer
SNNs, the `network_context` block, mixed-precision training) produce warnings, or errors
under `strict=True`; V2 pipeline code must call `V2Config.require_implemented()`.
The `temporal` block is now genuinely implemented, so selecting it is not a warning, and
`residual.source_functional_response`/`source_temporal` are hard-error cross-checked against
`residual.enabled` + `learned_residual_d > 0` (a source nobody consumes is a contradiction).
`functional_source_dim` is bounded (1 … 4096) and `temporal_resolution ≤ model.n_bins` is
enforced. `StructuredVectorEncoder.from_config` still refuses a residual request outright
(that encoder builds only the deterministic structured part) - build the full vector with
`neuron_vectors_from_config`.

Helpers: `estimate_input_tokens(B,T,C)`, `dtype_itemsize`, `check_input_token_budget(...)`,
`MemoryConfig.check_input_budget(...)`. Loading a config without V2 sections reproduces the
current experiment exactly (e.g. `n_hidden=256`, `n_bins=700`, `readout_mode=sum`,
`l2_spikes=0.001`).

---

## 3. `NeuronRecordBank` (`src/neuron_record.py`)

Columnar, read-only, CPU-side source information: one array per quantity with the neuron
index as the leading axis. Builder (no labels parameter anywhere):

```python
bank = build_neuron_record_bank(model, fit_rec=fit_rec, config=v2, device="cpu")   # streams FIT activity
bank = NeuronRecordBank.from_checkpoint("checkpoints/sweep_l2_0.pt", fit_rec=fit_rec)
bank = build_neuron_record_bank(model, with_activity=False)                        # structural only
```

| block | features | raw vectors | notes |
|---|---|---|---|
| `intrinsic` | per-neuron **varying** learned parameters only (`learned_bias`; `+tau_mem_ms` in `bias_tau` mode) | — | absent for an untrained model (existing rule) |
| `input_conn` | 14 summary statistics (+2 opt-in tonotopic) | `weights (n, n_input)`, row `i` = `w_in[:, i]` | |
| `recurrent_in` | 14 statistics + 5 incoming/outgoing relationship statistics | `weights (n, n_hidden)`, row `i` = `w_rec[i, :]` = **incoming** | repo convention: `w_rec[i, j]` = from `j` to `i` (`forward` uses `s_prev @ w_rec.T`) |
| `recurrent_out` | 14 statistics | `weights (n, n_hidden)`, row `i` = `w_rec[:, i]` = **outgoing** (= `recurrent_in.weights.T`) | |
| `activity` | 12 label-free FIT features + `silent_neuron` / `total_spikes` flags | optional `samples (n_samples, n)` float32 per-sample spike counts | streamed with the existing `ActivityAccumulator` |
| `temporal` | coarse label-free FIT PSTH bins (`bin_00` …, one mean firing rate per bin) | — | implemented 2026-09-27; built only when requested (see below) |
| `network_context` | **declared but not implemented** | — | `get_block` raises; no placeholder features |

**Temporal block** (`temporal`, implemented): derived from the label-free **pooled FIT PSTH**
(`ActivityAccumulatorResult.psth`, `(n_hidden, n_bins)`) by summing each coarse interval and
dividing by `n_samples x interval_duration_s` — i.e. *the mean FIT firing rate (Hz) of the
neuron in that interval*. Binning is the repository's existing coarse-PSTH convention,
`numpy.linspace(0, n_bins, temporal_resolution + 1).round()` with at least one simulation bin
per coarse bin (canonical: 700 bins x 2 ms at resolution 10 → ten 140 ms bins covering
0–1400 ms). Coordinates are named `temporal.bin_00` … and each bin's `[start_ms, stop_ms)`,
`start_bin`/`stop_bin` and definition are recorded in the bank provenance. No class labels,
no class PSTH, no speaker identity and no time-resolved tensor are involved. The block is
built **only when requested** — `vector.enabled_blocks` contains `temporal`, or
`vector.residual.source_temporal=true`, or the builder is called with an explicit
`temporal_resolution=` — so the default path is byte-for-byte unchanged (`temporal.present is
false` with an explicit reason in the provenance). Implemented-but-absent blocks (a temporal
request with no FIT activity) contribute nothing, exactly like `intrinsic` on an untrained
model.

API: `n_neurons`, `block_names`, `feature_names`, `uses_labels` (always `False`, read-only),
`get_block`, `block_status()`, `to_structured_matrix(blocks, chunk_size)`,
`to_representation_set()` (compatibility view of the existing object API), `provenance`,
`assert_no_forbidden_tensors()`.

Guarantees: label-free by construction (a supplied accumulator carrying labels or
class-conditioned arrays is rejected; scrambled FIT labels cannot change the bank); all
arrays are 1-D `(n,)` or 2-D `(n, k)` (plus the sample-major counts) — 3-D tensors such as
`(n_neurons, n_samples, n_time)` are rejected before any flattening; measured on the
canonical model: structural bank 2.46 MB in 0.11 s; a full FIT activity pass (5922 samples)
runs in ~62 s at batch 32 with a 205 MB CUDA peak; adding the temporal block costs ~0.14 s
and `(n, temporal_resolution)` float64 (~20 KB canonical). The stored `(n_fit_samples, n)`
float32 counts (6.1 MB canonical) are the data behind both the functional-response source
and the temporal block; they are never transposed as a whole.

---

## 4. `StructuredVectorEncoder` (`src/structured_vector.py`)

Deterministic, dimension-free-of-the-bank encoder producing `z_structured ∈ R^d_struct`.
It has **no fitted state** and consumes only a `NeuronRecordBank`.

```python
from src.structured_vector import StructuredVectorEncoder, encode_structured_vectors
vectors = encode_structured_vectors(bank, structured_d=100)                    # (n, 100)
encoder = StructuredVectorEncoder(bank, structured_d=100, enabled_blocks=None,
                                  projection_seed=0, chunk_size=None)
encoder = StructuredVectorEncoder.from_config(v2, bank)                        # residual request -> error
```

**Coordinate order** (a prefix of it is returned for smaller `d`):

1. **Level 0** — the bank's existing deterministic summary features, in canonical block
   order and alphabetical within a block (for the default four structural blocks on a
   trained model this is the historical **48-D** representation, reproduced exactly).
2. **Level 1** — deterministic within-block detail: per weighted block 31 coordinates
   (9 signed quantiles, 9 absolute quantiles, 8-bin absolute-mass histogram, 3 top signed
   weights, 2 threshold shares); for the activity block, 8 per-sample count summaries when
   counts are stored.
3. **Level 2** — a **fixed, seeded, data-independent** Gaussian projection
   (`1/sqrt(source_dim)`) used only beyond the interpretable source dimension; column `j`
   is drawn from `default_rng([seed, j])`, so the projection is prefix-stable. It is not
   learned, not fitted, and its coordinates are documented as non-interpretable mixtures.

Dimension policy: `d ≤ level0` → prefix of Level 0; `level0 < d ≤ source` → Level 0 + prefix
of Level 1; `d > source` → everything + projected coordinates. On the canonical 256-neuron
model: `level0 = 48`, `level1 = 93`, `source = 141` (with `activity` enabled: 60 / 101 / 161;
with `temporal` enabled at resolution 10 instead: 58 / 151). Every implemented block is
selectable, so `enabled_blocks = [intrinsic, input_conn, recurrent_in, recurrent_out,
temporal]` yields exactly the 48 structural summary coordinates followed by the 10 temporal
bins at `structured_d = 58`; the **default** selection stays the four structural blocks, so
the historical 48-D output is unchanged whether or not the bank carries a temporal block.

Provenance per coordinate (`kind`, `block`, `source`, `definition`, `uses_labels=False`) and
per encoder (`level0/1/source/output` dimensions, block selection, projection spec + seed,
selection policy, chunk size, bank provenance, configured V2 snapshot, warnings).

Memory: CPU only, arrays at most 2-D; `chunk_size` bounds temporaries. Source coordinates
are bit-identical between chunked and unchunked construction; projected coordinates agree to
floating-point tolerance. Measured: d=48 ≈0 ms, d=141 ≈67 ms / 0.28 MB, d=1000 ≈100 ms /
1.95 MB (+0.92 MB projection matrix).

Compatibility: at `structured_d = 48` with the default blocks the output equals the existing
48-D representation **exactly** (max abs diff 0.0) on the canonical architecture and on the
trained checkpoints `sweep_l2_0.pt`, `nsb_seed1.pt`, `nsb_seed2.pt`; exact at 47 for an
untrained model and 49 for a `bias_tau` model. Neuron counts 32/64/128/256 are exercised.

**Learned residual:** implemented by `src/residual.py` with full-vector composition in
`src/neuron_vector.py` — see §5 and §6.

---

## 5. Learned residual (`src/residual.py`)

The **only** learned component. It is trained on feature vectors exported from a frozen
record bank; it never loads the checkpoint, never backpropagates into the SNN and never
modifies model parameters (tested: the state dict is bit-identical before/after).

```python
from src.residual import (
    ResidualSourceConfig, ResidualTrainingConfig, ResidualTrainer, ResidualResult,
    build_residual_source, train_residual, make_mask, apply_mask, masked_reconstruction_loss,
)

source  = build_residual_source(bank)                                  # deterministic source view
result  = train_residual(bank, residual_dim=52)                        # trains, returns artifact
result  = train_residual(bank, config=ResidualTrainingConfig.from_v2_config(v2))
z_res   = result.encode(bank)                                          # (n_neurons, residual_dim)
result.save("checkpoints/residual.pt")
loaded  = ResidualResult.load("checkpoints/residual.pt", expected_feature_names=source.feature_names,
                              expected_residual_dim=52)
```

**Input feature schema** (`build_residual_source`), in order:

1. Level-0 deterministic summaries from the record bank,
2. Level-1 deterministic within-block detail (the same definitions the structured encoder
   uses),
3. a **fixed seeded projection of the raw connectivity** of each weighted block
   (`raw_view_dim`, default 32 columns per block) — source information the 31 summaries do
   not expose, which is what gives the residual additional capacity rather than duplicating
   `z_structured`.

Shape `(n_neurons, F)` on CPU; canonical 256-neuron model: `F = 48 + 93 + 96 = 237`
(level 0 + level 1 + raw views). No sample or time axis exists anywhere in the pipeline.

**Objective** (self-supervised, label-free): masked reconstruction

```
standardise(x)  ->  zero out a deterministic random subset of coordinates
                ->  encoder -> z_residual -> decoder -> reconstruction
loss = MSE(reconstruction, target) on the WITHHELD coordinates only
```

(`visible_loss_weight` optionally adds a visible term; default 0.0, so the loss is computed
purely on withheld coordinates — no identity shortcut.)

**Masking policy** (`ResidualTrainingConfig`): `mask_fraction` (0.25), `mask_seed` (0), deterministic
`minimum_visible_features` (8); masks are drawn with `default_rng([mask_seed, epoch])` per
example, the evaluation masks with `[mask_seed, 10^7]` and `[mask_seed, 10^7 + 1]`; masked
coordinates are zeroed in standardised space. No labels are involved in mask generation.

**Architecture** (deliberately small): encoder `input_dim -> hidden_dim (64) -> residual_dim`,
GELU, optional dropout (0.0); decoder `residual_dim -> hidden_dim -> input_dim`. `residual_dim`
comes from `vector.learned_residual_d` via `ResidualTrainingConfig.from_v2_config`.

**Training split:** a deterministic split over *neuron examples* within FIT
(`val_fraction` 0.2, `split_seed` 0) with best-validation state restored; indices are recorded
in provenance. PROBE/TEST are never read; the source builder rejects a bank whose activity
provenance names a `probe`/`test` split.

**Normalisation:** `train_standardise` (default) fits mean/std on the residual *training*
neuron subset only — explicitly recorded as FIT-derived; `none` disables it. Fixed,
data-independent scaling is used for the raw-view projection (`1/sqrt(width)`, seeded per
column).

**Persistence** (`learned_residual/v1` payload): schema, `residual_dim`, `input_dim`,
`hidden_dim`, `feature_names` + `feature_schema_hash`, source schema/config, normalisation
statistics, full training config, network spec, encoder/decoder state dicts, history, seeds,
provenance. `ResidualResult.load` refuses a mismatched schema, feature names, `input_dim`,
`residual_dim` or a failed hash integrity check — incompatible artifacts are never adapted
silently.

**Reproducibility:** global Python/NumPy/Torch seed (`set_seed`), `split_seed`, `mask_seed`,
per-epoch shuffle seed. Same bank + config + seed reproduces the vectors exactly (tested).

**Precision:** `precision.vector_dtype` (`float32`/`float16`/`bfloat16`, aliases accepted); on
CPU a non-fp32 request falls back to `float32` with a recorded warning, so CPU use stays
usable. No FP8, no change to SNN training precision.

**Diagnostics recorded (no scientific evaluation):** per-epoch `train_loss`, `val_loss`,
`train_masked_mse`, `val_masked_mse`, `optimization_loss_mean`, realised mask fraction, plus
`best_epoch`/`best_val_loss`.

#### Label-free functional-response source (`src/functional_response.py`)

The bank already stores the per-stimulus response data (`activity.samples`: `(n_fit_samples,
 n_neurons)` float32 spike counts, 6.1 MB canonical). This module turns it into a compact,
deterministic, label-free source view:

```
samples (n_fit_samples, n_neurons)
  -> explicit per-neuron transform over the sample axis  (raw | neuron_centered | neuron_zscored)
  -> fixed seeded Gaussian projection of the sample axis
  -> (n_neurons, functional_source_dim)                  default 64
```

| property | value |
|---|---|
| definition | `activity.samples[s, i]` = spike count of neuron `i` on FIT utterance `s` |
| split | FIT only (a bank whose activity provenance names probe/test is rejected) |
| normalisation | `raw` (default; keeps the magnitude/rate information), `neuron_centered` (`v - mean_s v`), `neuron_zscored` (`(v - mean_s v)/std_s v`); applied to the counts **before** the projection and never re-applied |
| zero-variance rule | `std_s < 1e-8` ⇒ the neuron is **kept** and its profile is exactly zero, so its projected coordinates are exactly zero (`zero_variance_neurons` is recorded) |
| projection | fixed, data-independent, never fitted or learned: column `j = default_rng([functional_projection_seed, j]).standard_normal(n_samples)/sqrt(n_samples)`; prefix-stable in the dimension |
| seed | dedicated `functional_projection_seed` (default 0) - never shared with the structured projection seed, the residual mask seed or the training seed |
| ordering | the stored FIT order is preserved and hashed: `sample_order_hash = sha256(shape\|dtype\|sample-major bytes)`; no label or speaker identity participates |
| memory | CPU only; statistics accumulated and the projection applied in sample chunks (`chunk x n_neurons`); the `(n_neurons, n_fit_samples)` transpose and any 3-D tensor are never materialised; the projection matrix is `(n_samples, functional_source_dim)` float64 (~3.0 MB canonical) |

```python
from src.functional_response import FunctionalResponseConfig, build_functional_response_source
source = build_functional_response_source(bank, FunctionalResponseConfig(source_dim=64))
source.X                     # (n_neurons, 64)
source.sample_order_hash
source.provenance            # source type/split, uses_labels=False, normalisation stats + order,
                             # zero-variance rule, projection spec + seed, sample-order hash
```

#### Temporal source in the residual

`include_temporal=True` appends the bank's `temporal` block (all coarse bins, canonical
`bin_00…` order, mean FIT firing rate in Hz, no further scaling). It is independent of the
structured encoder's `enabled_blocks`: the residual can consume it even when the structured
vector does not select it. Requesting it from a bank without the block is an explicit error
(never a silent skip), and coordinates already contributed by an earlier part of the view are
skipped with a recorded de-duplication count.

#### Residual source view (expanded)

`build_residual_source(bank, config)` assembles up to five identifiable parts, in order:

| part | flag (default) | dimension |
|---|---|---|
| level-0 deterministic summaries | `include_level0` (**true**) | `level0_dim` |
| level-1 deterministic within-block detail | `include_level1` (**true**) | `source_dim - level0_dim` |
| fixed raw-connectivity projections | `include_raw_weight_view` (**true**) | `min(raw_view_dim, width)` per weighted block (32) |
| functional-response projection | `include_functional_response` (**false**) | `functional_source_dim` (64) |
| coarse temporal block | `include_temporal` (**false**) | `temporal_resolution` (10) |

Canonical budget: 48 + 93 + 96 = **237** by default (unchanged), 301 with the functional
source, 247 with the temporal block, 311 with both. Because the two new flags default to
`false`, the feature names - and therefore the schema hash - are identical to the previous
stage, so existing residual artifacts stay loadable and the historical numbers remain
reproducible. `ResidualSourceConfig.from_v2_config(v2)` reads `vector.enabled_blocks`,
`vector.residual.source_functional_response`/`source_temporal`, `vector.functional_source_dim`,
`functional_projection_seed`, `functional_source_normalization`; the two evaluation scripts use
it, so the new sources are reachable from a config alone. Every part is identified in
`provenance.functional_response`, `provenance.temporal`, `provenance.source_blocks` and
`provenance.source_config`, with the final per-group coordinate counts in
`provenance.source_groups`.

#### Masking: coordinate vs source-block

`mask_mode` (`vector.residual.mask_mode`, default `coordinate`) selects the strategy of the
same masked-reconstruction objective:

* `coordinate` - unchanged: `mask_fraction` (0.25) of the coordinates per example, at least
  `minimum_visible_features` (8) visible;
* `block` - one whole **source group** per example (`structural` / `functional_response` /
  `temporal`, classified by feature-name prefix) is withheld, so the model must reconstruct a
  source from the others. Groups are tried in random order and the first that keeps enough
  coordinates visible is used; a row falls back to coordinate-level masking when no group
  satisfies the visibility guarantee. Masks remain deterministic (per-epoch
  `default_rng([mask_seed, epoch])`, evaluation `[mask_seed, 10^7(+1)]`) and label-free.

#### Per-source training diagnostics

In addition to the existing fields, the history now records the withheld-coordinate MSE of
each present source group (`train_masked_mse_structural`,
`train_masked_mse_functional_response`, `train_masked_mse_temporal` and the `val_` variants),
the realised `mask_mode` and `mask_fraction_realised`. These are **training diagnostics only**;
no scientific interpretation is attached to them. `provenance.source_diagnostics` lists the
fields and `provenance.source_groups` the coordinate ranges.

**Persistence:** the new fields flow through the existing `learned_residual/v1` payload
(`config.mask_mode`, `source_config`, and `source_schema.provenance` with the per-source blocks
and groups). `ResidualResult.load` still refuses a mismatched feature schema, so a residual
trained on the expanded source cannot be silently applied to the old source view. No schema
bump was needed: an older payload without `mask_mode` loads with the default `coordinate`.

---

## 6. Full neuron vector (`src/neuron_vector.py`)

Composition only — no training, no fitting, no evaluation:

```python
from src.neuron_vector import build_neuron_vectors, neuron_vectors_from_config

vectors = build_neuron_vectors(bank, structured_d=48)                       # 48-D, no model
vectors = build_neuron_vectors(bank, structured_d=48, residual=result)      # 48 + d_residual
vectors = neuron_vectors_from_config(v2, bank, residual=result)             # from the V2 config
vectors.X                 # (n_neurons, d)
vectors.feature_names     # structured names ... then residual[0], residual[1], ...
vectors.provenance        # structured source, residual source, coordinate order, total dimension
```

Invariant (checked): `full_dimension == structured_d + learned_residual_d`. Coordinate order
is always *structured first, residual last*. Provenance exposes the structured source (block
selection, level dimensions, projection spec/seed), the residual source (artifact schema, source
schema hash, mask policy, seeds, best epoch and the full training provenance), the bank
provenance and, when built from the V2 config, the configured dimensions.

Backward compatibility: with `learned_residual_d = 0` the API returns the deterministic
structured vector **exactly** (bit-identical to `StructuredVectorEncoder`, which is itself
exactly the historical 48-D representation) and instantiates no residual model. Verified at
48 + 0 -> 48, 48 + 16 -> 64 and 48 + 52 -> 100 with consistent coordinate ordering.

---

## 7. Label-leakage contract (applies to every layer above)

* The only component allowed to consume class labels is the functional-fingerprint /
  evaluation machinery (`src/functional_fingerprint.py`), and only on the held-out PROBE
  split.
* `NeuronRecordBank`, `StructuredVectorEncoder`, the learned residual and the composition
  expose **no labels parameter**; all their provenance records `uses_labels: False`.
* The functional-response source and the temporal block are built from the bank's stored FIT
  counts and pooled FIT PSTH only: no class labels, no class PSTH/rate, no speaker identity,
  no PROBE, no TEST, no model forward pass. The FIT split guard
  (`assert_label_free_fit_activity`) is shared by both consumers.
* FIT is the only data source for representation information. PROBE and the official TEST
  split are never read by representation construction.
* Normalisation statistics, where used, are derived only from the FIT-derived training
  portion of the residual examples (documented per layer in §5).

---

## 8. Memory rules (global)

Allowed: `(n_neurons, k)` feature/weight matrices, `(n_neurons, d)` vectors, the
`(n_samples, n_neurons)` activity counts, and compact minibatches of feature vectors.

New arrays of the functional-response / temporal stage (all CPU, all 2-D): the stored counts
`(n_fit_samples, n_neurons)` float32 (6.1 MB canonical, owned by the bank), the
`(n_neurons, functional_source_dim)` projected source (~131 KB), the
`(n_samples, functional_source_dim)` float64 projection matrix (~3.0 MB), a
`(chunk, n_neurons)` sample-chunk temporary (chunk default 256 → ~0.5 MB), and the
`(n_neurons, temporal_resolution)` temporal block (~20 KB). GPU transfers are unchanged and
scale only with `n_neurons x source_features` (a source minibatch), never with the FIT sample
count: the whole projection and temporal summarisation happen in NumPy on the CPU.

**Forbidden** and rejected by guards: `(n_neurons, n_samples, n_time)`,
`(n_samples, n_time, n_neurons)`, `(n_neurons, n_samples, d)`, `(n_neurons, n_hidden, d)`,
`(n_neurons, n_time, d)`, `(n_neurons, n_neurons, d)`. Nothing on the GPU may scale as
`n_neurons · n_samples · d`. Stage-specific batch controls: `memory.train_batch_size`,
`eval_batch_size`, `record_batch_size`, `activity_chunk_size`,
`representation_chunk_size`, and the `max_input_tokens` guard for dense `(B, T, C)` inputs.

---

## 9. Tests

```
uv run pytest -q    ->  448 passed
```

| file | tests | covers |
|---|---|---|
| `tests/test_v2_config.py` | 59 | configuration defaults, backward compatibility of the four existing configs, dimension invariant, dtype/device/storage validation, token budget, CLI overrides |
| `tests/test_neuron_record.py` | 45 | columnar structure, orientation (incoming row / outgoing column), label-free guarantees, exact 48-D compatibility, memory guards, provenance |
| `tests/test_structured_vector.py` | 51 | exact 48-D baseline, prefix dimension policy, Level-1 detail, projection determinism/prefix-stability, block selection, neuron counts, chunking, label-freedom, V2-config integration |
| `tests/test_residual.py` | 37 | source view and variants, no-labels API, probe-split rejection, relabelled-FIT invariance, masking determinism/targets, architecture and config validation, training diagnostics, SNN-frozen check, reproducibility, persistence with mismatch rejection, CPU precision fallback, memory guards |
| `tests/test_neuron_vector.py` | 16 | 48/64/100 composition, coordinate order, provenance, `residual_d=0` exactness, config integration, schema/dimension mismatch rejection, determinism, chunking |
| `tests/test_vector_capacity.py` | 33 | PROBE-only targets and split discipline, individual-stimulus target shape/definition/label-independence, frozen representations, the dimension invariant and matched-total decompositions, residual artifact selection, SNN isolation, reproducibility, memory shapes, result table + figures |
| `tests/test_rate_robustness.py` | 32 | raw-target equivalence with the first study, exact neuron centering/z-scoring, deterministic zero-variance rule, explicit (non-commuting) pipeline order, mean-rate and row-L2 targets, activity dimensions, checkpoint comparability, PROBE-only/label discipline, representation invariance, 1-D prediction, reproducibility, memory |
| `tests/test_functional_response.py` | 48 | functional-response shape/definition, deterministic + prefix-stable + seed-dependent projection, sample-order hash, all normalisation modes, zero-variance rule, label-free provenance, chunked == unchunked; temporal bin edges/values/resolution/names/provenance; the 48-D default staying exact and temporal appearing only when requested; residual source expansion, de-duplication, schema persistence + mismatch rejection, both mask modes, per-source diagnostics, SNN-frozen check; `48+0/48+16/48+52` and a temporal structured + residual configuration |
| pre-existing suite | 127 | model, data, training, evaluation, geometry, prediction, fingerprint, controls, rewiring, permutation, reliability, NSB |

---

## 10. Known limitations

* Single hidden layer only (`model.n_layers > 1` is configuration-only and rejected in
  strict mode); the record/encoder layers carry a `layer`-agnostic design but no multi-layer
  implementation exists.
* `temporal` and `network_context` record blocks: `temporal` is now **implemented** (coarse
  pooled-FIT-PSTH bins, selectable and available to the residual); `network_context` remains
  declared but not implemented (no placeholder features).
* The structured encoder's projection coordinates are not individually interpretable.
* The learned residual is a small MLP trained with masked reconstruction on FIT-derived
  features. It has now been evaluated once on PROBE (§11): its presence does not separate the
  measured primary/class-rate/temporal/CV metrics from the structured-only conditions at
  matched total dimension, and the residual itself shows seed-to-seed spread of the same order
  as those differences. No claim of improvement is made or supported.
* The residual's `raw_view` coordinates are a fixed random projection of the raw
  connectivity, chosen for additional capacity; they are not individually interpretable.
* The functional-response source coordinates are likewise a fixed random projection: they are
  not individually interpretable, and the source view preserves the *statistics* of the
  per-stimulus profile (through 64 projections), not the per-stimulus responses themselves.
* The temporal block is a **pooled (population-averaged over FIT utterances) coarse PSTH**,
  not a per-stimulus temporal profile: it carries the neuron's mean time course, so
  stimulus-specific temporal structure is not represented by this block.
* The functional-response projection depends on the order of the FIT utterances; the exact
  order is pinned by the deterministic split and recorded as a hash, but a different split or
  sample order would produce a different (still label-free) source view.
* Enabling the new sources changes the residual input schema (and therefore its schema hash);
  no residual artifact trained on the old schema can be reused with the new one by design.
* The new sources have **not** been scientifically evaluated: no PROBE analysis, no capacity
  sweep and no comparison against the previous configuration was performed in this stage.
* Residual training is reproducible on a fixed device/dtype; CUDA is only bitwise
  reproducible to the extent documented for the repository's other models.
* GPU/memmap storage for records and the record-bank serialisation are not implemented
  (records are rebuilt from the checkpoint/FIT each time; the residual *is* persistable).
* The repository's scientific results (README, archived reports) refer to the V1 pipeline
  and its 48-D representation; they were not recomputed in the V2 work.

---

## 11. Scientific capacity evaluation (PROBE)

The **first stage allowed to touch PROBE**. It measures how the information in the neuron
representation changes with representation capacity, and what the learned residual adds
beyond the deterministic structured component. It is an **evaluation study**: it reports
measured values and uncertainty and deliberately produces **no ranking, no "best"
dimension and no "improves"/"optimal" claim**. The SNN was not re-trained; the residual
was not tuned on PROBE; the official TEST split was never opened.

### Entry point and artifacts

```bash
uv run python scripts/evaluate_vector_capacity.py                 # full run, ~10 min
uv run python scripts/evaluate_vector_capacity.py --quick         # smoke settings
```

| artifact | location |
|---|---|
| machine-readable table | `results/neuron_vector_capacity/results.csv` (14 rows), `results.json` (rows + per-seed summary + targets/preprocessing) |
| provenance | `results/neuron_vector_capacity/metadata.json` |
| activity + residual caches (checkpoint/split/seed-keyed) | `results/neuron_vector_capacity/cache/`, `.../residuals/` |
| figures | `figures/neuron_vector_capacity/figure{1,2,3}_*.{png,pdf}` |

### Conditions (11) and the prespecified comparisons

| condition | total_d | structured_d | residual_d | seeds |
|---|---|---|---|---|
| `structured_32`, `structured_48`, `structured_64`, `structured_100`, `structured_128` | 32/48/64/100/128 | = total_d | 0 | — |
| `full_48+16_seed{0,1,2}` | 64 | 48 | 16 | 0, 1, 2 |
| `full_48+52_seed{0,1,2}` | 100 | 48 | 52 | 0, 1, 2 |

Matched-total comparisons: `structured_64` vs `full_48+16` and `structured_100` vs
`full_48+52`; `structured_48` is the common baseline. The two residual families have the
same total dimension as their structured partners but a **different decomposition** — they
are never treated as the same construction. No extra residual variants were trained and
nothing was selected on PROBE.

### Data, checkpoint, split

* checkpoint `checkpoints/sweep_l2_0.pt` (`sha256` first 16 hex `ca6db04913814746`),
  256 hidden neurons, 700 × 2 ms bins, `readout_mode=sum`, `neuron_param_mode=bias`; frozen
  for every condition.
* FIT **5922** samples (speakers 9, 3, 7, 11, 0, 10, 1) — representation construction and
  residual training only; the residual's normalisation statistics come from FIT only.
* PROBE **1236** samples (speakers 6, 8) — functional targets and metrics only.
* split seed 0, strategy `speaker_aware_train_dev_probe`; the official TEST file is never
  opened (`metadata.data.official_test_loaded = false`). One FIT activity pass and one
  labelled PROBE pass are collected once and reused by every condition (verified caches).

### Primary functional target (individual-stimulus responses)

`R[h, s] = counts[s, h] / (n_bins · bin_ms / 1000)` Hz — the repository's existing firing-rate
definition (`count / duration`) applied **per individual stimulus** instead of per class:
`(n_neurons, n_probe_stimuli) = (256, 1236)`. No averaging across stimuli; each PROBE
utterance is one column (`fp_stimulus.s####`). One "individual stimulus" is one SHD utterance
(SHD has no repeated-stimulus identifier — limitation below). The target is label-free; it is
built by the new `stimulus_response_fingerprint` in `src/functional_fingerprint.py`, which is
now a registered feature set (`stimulus_response`) but explicitly **not** class-conditioned.
A rate-magnitude-removed variant (row-L2-normalised profiles) serves as the primary rate control.

### Secondary and exploratory targets (unchanged definitions)

* `class_rate_20d`: the existing PRIMARY preset (`tuning`, 20 class-conditioned mean rates).
* `temporal` (260-d): the existing exploratory preset — coarse class PSTH (20 classes ×
  10 bins) + temporal centre + dispersion + (censored) first-spike latency.

### Metrics (all from the existing stack)

Mantel Spearman r on condensed Euclidean distances (representation `RepresentationSpace.X`
under the canonical preprocessing: column z-scoring, `weighting="uniform"`, no row
normalisation; target `FingerprintSpace.X`), one-sided neuron-relabelling permutation null
with **2000 permutations**, **500**-resample neuron bootstrap 95% CI, rate-matched stratified
Mantel, kNN effect sizes (k ∈ {3, 5, 10, 20}), and out-of-fold ridge / kNN prediction
(`KFold` over neurons, 5 folds, folds **shared** across conditions and targets, `RidgeCV`
alphas selected inside the training fold only). Permutation/bootstrap seed 0.

### Measured results (observation, not interpretation)

| representation | total_d | primary r | primary 95% CI | p | z | rate-matched r | rate-norm. r | class-rate r | temporal r | CV R² (ridge) |
|---|---|---|---|---|---|---|---|---|---|---|
| structured_32 | 32 | +0.122 | [0.032, 0.227] | 0.0040 | 2.87 | +0.150 | +0.118 | +0.142 | +0.325 | +0.116 |
| structured_48 | 48 | +0.113 | [0.030, 0.216] | 0.0055 | 2.68 | +0.117 | +0.129 | +0.139 | +0.319 | +0.117 |
| structured_64 | 64 | +0.127 | [0.047, 0.230] | 0.0010 | 3.04 | +0.136 | +0.155 | +0.155 | +0.336 | +0.116 |
| structured_100 | 100 | +0.119 | [0.034, 0.224] | 0.0035 | 2.83 | +0.132 | +0.133 | +0.146 | +0.327 | +0.112 |
| structured_128 | 128 | +0.121 | [0.039, 0.229] | 0.0035 | 2.87 | +0.127 | +0.125 | +0.148 | +0.322 | +0.117 |
| full_48+16 (seeds 0/1/2) | 64 | +0.114 / +0.123 / +0.113 | all CIs exclude 0 | 0.0025–0.0065 | 2.67–2.90 | +0.116–0.121 | +0.131–0.136 | +0.140–0.151 | +0.316–0.329 | +0.099–0.108 |
| full_48+52 (seeds 0/1/2) | 100 | +0.129 / +0.119 / +0.128 | all CIs exclude 0 | 0.0010–0.0040 | 2.76–3.01 | +0.121–0.136 | +0.131–0.134 | +0.148–0.160 | +0.322–0.337 | +0.112–0.128 |
| control_rate_only | 1 | +0.663 | [0.587, 0.727] | 0.0005 | 19.42 | +0.279 | −0.119 | +0.721 | +0.423 | +0.426 |
| control_random_100 | 100 | −0.054 | [−0.112, 0.026] | 0.936 | −1.50 | −0.039 | +0.035 | −0.060 | −0.002 | −0.015 |
| control_neuron_shuffle_100 | 100 | +0.041 | [−0.020, 0.124] | 0.159 | 1.02 | +0.049 | −0.023 | +0.048 | +0.020 | −0.024 |

p-values are floored at 1/(2000+1) = 5.0 × 10⁻⁴. Residual seed means (SD over 3 seeds) on
the primary metric: **0.1169 (0.0042)** for 48+16 and **0.1253 (0.0045)** for 48+52. kNN
best-k = 3 for every real
representation (z = 3.3–4.2) and k = 5/20 for the random/shuffle controls (z = −0.28 / 1.42).

**Patterns in the measured values** — stated as observations only:

* the primary correspondence is positive for every structured and residual condition
  (r ≈ +0.11 … +0.13, all bootstrap CIs excluding 0);
* it is **flat** across d = 32 → 128 (no monotone trend) and the residual-seed spread is
  comparable to the differences between total dimensions;
* on matched totals, `full_48+16` does **not** exceed `structured_64` on the primary metric,
  and `full_48+52` is within the seed spread of `structured_100`; the same holds for the
  class-rate, temporal and CV-R² columns;
* the 1-dimensional **FIT rate-only control reaches r = +0.663**, far above every 32–128-D
  representation on the same target; after removing response magnitude (rate-normalised
  target) its r drops to −0.119, and the rate-matched Mantel gives +0.279;
* random and neuron-shuffle controls are small and not distinguishable from their nulls.

These are descriptive measurements at one checkpoint and one split seed; they are not a
statement about a "best" dimension, and no post-hoc trend is fitted.

### Documentation deviations and harness corrections (smallest fixes)

* `condition_matrix` selected the residual artifact by **seed only**, which cannot hold two
  residual dimensions per seed; it now resolves `{residual_d: {seed: artifact}}` (or a flat
  seed map) and **verifies** `artifact.residual_dim == condition.residual_d`
  (`select_residual`). Regression-tested.
* `stimulus_response` was registered in `FINGERPRINT_FEATURE_SETS` (complete configuration /
  provenance surface) and `class_conditioned_fingerprint` now raises a clear error if it is
  ever routed through class-conditioned statistics. Regression-tested.
* the kNN headline in `evaluate_condition` is now taken from the existing
  `primary_metric_row` instead of a second, differently-ordered selection rule.
* No existing scientific definition, metric, split or preprocessing convention was changed.

### Limitations (of this evaluation)

* "Individual stimulus" = one SHD utterance; SHD provides no repeated-stimulus identifier, so
  repeated recordings across speakers cannot be grouped. The primary target therefore
  preserves utterance-level variance by definition.
* The primary target has 1236 columns (allowed `(n_neurons, n_stimuli)` target, ~2.5 MB);
  it is standardised column-wise, so columns with near-zero variance contribute little.
* p-values come from a neuron-relabelling Mantel null with a resolution floor; the study
  reports effect sizes, CIs and seed spread and runs no family-wise significance campaign
  across the 11 conditions.
* The bootstrap CI uses the repository's naive neuron bootstrap (duplicated neurons are a
  documented limitation of that implementation).
* Residuals were trained on CPU in float32 (recorded); residual training reproducibility on
  CUDA is only as documented for the other models.
* One checkpoint (`sweep_l2_0.pt`) and one split seed (0) for the *capacity sweep* itself; the
  rate-robustness stage below repeats its focused set on three comparable checkpoints.

---

### Rate-confound / robustness analysis (second stage of the same programme)

**Why it was added.** The first study measured a positive primary correspondence
(r ≈ +0.11 … +0.13) *and* a 1-D FIT **rate-only** control at r = +0.663 against the same
target. Before any new biological block (temporal / network context) is justified, the
evaluation has to separate *global firing-rate tendency* from *stimulus-specific response
structure*. This stage is that methodological checkpoint; it adds **no** representation
architecture and no new biological features.

**Verified semantics of the pre-existing controls** (read from the code, not inferred from the names):

| name | side | actual definition |
|---|---|---|
| `control_rate_only` | representation | the label-free **FIT** mean rate per neuron as a 1-D representation; `RepresentationSpace` column-z-scores it, so its condensed distance is `|rate_i − rate_j| / std_FIT`. The statistic is the Spearman correlation of *that* distance vector with the target's distance vector — not "rates vs responses" |
| `primary_rate_normalized` (target side) | target | the raw response with each neuron's profile L2-normalised (`FingerprintSpace(standardize="none", normalize_rows=True)`); removes the L2 magnitude, not the mean. Reproduced bit-identically here as the `row_l2_normalised` variant |
| `rate_matched` | pair-stratified | 5 quantile strata of `|rate_i − rate_j|` (FIT rates); within-stratum Mantel Spearman combined by sample-size-weighted Fisher z, strata held fixed under the relabelling null |
| `partial_mantel` | exploratory | rank-residualises both distance vectors on rank(`|rate_i − rate_j|`); **secondary only**, not a proof of rate-independence |
| `mean_rate` (new, target side) | target | the PROBE per-neuron mean response `mean_s R[h,s]` as a 1-D target. A different object from `control_rate_only`: there the rate is the *representation*, here the rate *is* the target |

**The headline target's own standardisation was verified:** `FingerprintSpace(standardize="column")`
z-scores **across neurons within each stimulus column** (zero-variance columns → constant 0,
flagged). That removes the *column* mean, not each neuron's mean over stimuli — on `sweep_l2_0`
the per-neuron mean of the column-standardised target correlates **0.999** with the raw
per-neuron rate. This is precisely the confound quantified below.

**Target decomposition** (PROBE-only, evaluation-only). `R[h,s] = counts[s,h] / (n_bins·bin_ms/1000)`
Hz, one column per PROBE utterance (the shared `stimulus_response_matrix`). Each variant is an
**ordered pipeline** applied to `R`, then wrapped in `FingerprintSpace(standardize="none")`, so
the transform order is exactly the one named and never re-applied implicitly:

| variant | pipeline (left → right) | nuisance removed |
|---|---|---|
| `raw` | `column_standardise` | none at the neuron level — **bit-identical to the first study's primary target** |
| `neuron_centered` | `neuron_center → column_standardise` | per-neuron baseline `mean_s R` |
| `neuron_zscored` | `neuron_zscore → column_standardise` | per-neuron baseline **and** amplitude `std_s R` |
| `mean_rate` | `mean_over_stimuli → column_standardise` | everything except the per-neuron mean |
| `column_standardised_then_neuron_centered` | `column_standardise → neuron_center` | ordering sensitivity (the two orders do not commute) |
| `row_l2_normalised` | `row_l2_normalise` | per-neuron L2 magnitude (the first study's existing control) |

Zero-variance rule: `std_s R < 1e-8` ⇒ the neuron is **kept** and its whole z-scored row is set
to `0.0` (deterministic; 0 such neurons on this data). Ordering is explicit, not assumed: after
`…→ neuron_center` row means are exactly 0, after `column_standardise → neuron_center` they are
0 too but the two orders still differ in general (here they agree closely, e.g. structured_48
+0.165 vs +0.167).

**Representations, checkpoints, discipline.** 12 focused conditions: `structured_48/64/100`,
`full_48+16_seed0/1/2`, `full_48+52_seed0/1/2`, plus the activity diagnostics
`activity_only` (12-d = the block's own summary features), `activity_source_20` (20-d = the
block's full deterministic source) and `structural_48_plus_activity` (60-d = the 48 structural
prefix + the 12 activity features; dimensions derived from the encoder plan, not hard-coded).
Residuals are the first study's cached artifacts for `sweep_l2_0` and the same protocol
(FIT-only, 200 epochs, seeds 0/1/2) for the other checkpoints; no new residual families were
trained. Three checkpoints — `sweep_l2_0.pt`, `nsb_seed1.pt`, `nsb_seed2.pt` — were verified
**comparable** before aggregation (identical architecture fields and identical representation
schema: same blocks, feature names and level-0/source dimensions). FIT/PROBE/TEST discipline is
unchanged (TEST never opened); every target transform is a PROBE-only, evaluation-only transform
of the 2-D `(neurons, stimuli)` matrix.

```bash
uv run python scripts/evaluate_vector_rate_robustness.py          # full, ~26 min
uv run python scripts/evaluate_vector_rate_robustness.py --quick  # smoke settings
```

| artifact | location |
|---|---|
| results (new files; the first study's are untouched) | `results/neuron_vector_capacity/rate_robustness_results.csv` (252 rows) / `rate_robustness_results.json` |
| provenance | `results/neuron_vector_capacity/rate_robustness_metadata.json` (includes the first study's file hashes, recorded to document that they were not modified) |
| figures | `figures/neuron_vector_capacity/rate_figure{1,2,3}_*.{png,pdf}` |

**Measured results** (Mantel Spearman r; mean over the 3 checkpoints; residual families also
averaged over seeds; permutation p ≤ 0.0005 unless noted; 95% neuron-bootstrap CI)

| representation | blocks | d | raw | neuron-centered | neuron z-scored | mean-rate | row-L2 |
|---|---|---|---|---|---|---|---|
| `structured_48` | structural | 48 | **+0.146** | +0.167 | +0.030 | +0.095 | +0.081 |
| `structured_64` | structural | 64 | +0.163 | +0.190 | +0.044 | +0.100 | +0.094 |
| `structured_100` | structural | 100 | +0.151 | +0.180 | +0.037 | +0.087 | +0.091 |
| `full_48+16` (3 seeds) | +residual | 64 | +0.149 … +0.154 | +0.171 … +0.175 | +0.031 … +0.036 | +0.095 … +0.099 | +0.084 … +0.085 |
| `full_48+52` (3 seeds) | +residual | 100 | +0.149 … +0.156 | +0.170 … +0.181 | +0.034 … +0.035 | +0.088 … +0.099 | +0.084 … +0.089 |
| `activity_only` | activity | 12 | +0.405 | +0.276 | +0.111 | +0.489 | +0.316 |
| `activity_source_20` | activity | 20 | +0.522 | +0.359 | +0.079 | +0.619 | +0.211 |
| `structural_48_plus_activity` | structural+activity | 60 | +0.266 | +0.227 | +0.053 | +0.263 | +0.189 |
| `control_rate_only` | FIT rate | 1 | +0.680 | +0.493 | +0.024 | +0.773 | — |
| `control_random_100` | random | 100 | +0.006 | +0.007 | +0.009 | +0.006 | — |
| `control_neuron_shuffle_100` | shuffle | 100 | +0.030 | +0.025 | +0.005 | +0.030 | — |

Uncertainty for the structural baseline (`structured_48`, n = 3 checkpoints): raw +0.1465
(sd 0.039) CI [0.030, 0.302]; centered +0.1672 (sd 0.044) CI [0.035, 0.336] (p ≤ 0.0010);
z-scored +0.0295 (sd 0.021) CI [−0.011, 0.115] (**p ≤ 0.384 — not distinguishable from 0**);
mean-rate target +0.0946 (sd 0.031) CI [+0.003, 0.218]. `activity_source_20`: raw +0.522
(sd 0.007) CI [0.419, 0.628]; centered +0.359 CI [0.234, 0.495]; z-scored +0.079 CI
[−0.020, 0.208] (p ≤ 0.037); mean-rate +0.619 CI [0.538, 0.705]. Descriptive differences:
Δ_centering = +0.021 (structural_48) and Δ_scaling = −0.138; for activity_only Δ_scaling =
−0.165 and for activity_source_20 −0.280.

**Direct observations** (values only):

* neuron-centering does **not** reduce the structural/residual correspondence
  (Δ_centering = +0.021 … +0.029); it *reduces* the activity-block correspondence
  (−0.129 at 12-d, −0.163 at 20-d);
* neuron-z-scoring collapses the structural/residual correspondence
  (Δ_scaling = −0.136 … −0.147, leaving +0.030 … +0.044, with CIs spanning 0 for
  structured_48/100 and p up to 0.38), and lowers the activity ones to +0.111 / +0.079 / +0.053;
* the mean-rate target alone reaches only +0.087 … +0.100 for the structural/residual
  conditions but +0.489 / +0.619 for the activity conditions;
* adding the activity block to the structural prefix raises the raw correspondence from
  +0.146 (48-d) to +0.266 (60-d) and the CV R² from +0.117 to +0.544;
* the residual families are within seed spread of their structured partners on every target
  variant (seed SD within a checkpoint 0.001–0.007);
* checkpoint variability is larger than seed variability: `structured_48` raw is +0.113
  (sweep_l2_0, exactly reproducing the first study), +0.200 (nsb_seed1), +0.126 (nsb_seed2),
  sd ≈ 0.039; z-scored is +0.004 / +0.029 / +0.055;
* the largest numbers in the study belong to the 1-D FIT rate representation (+0.680 raw,
  +0.773 against the mean-rate target); random and neuron-shuffle controls stay at ≈ 0.005–0.030.

**Interpretation** (clearly separated from the observations):

* For the connectivity/intrinsic representation, the correspondence with individual-stimulus
  responses is consistent with per-neuron response *level/amplitude* structure rather than
  relative stimulus-response *shape*: once baseline and amplitude are removed, the estimate is
  indistinguishable from the null.
* The label-free activity block reproduces substantially more of the PROBE functional
  organisation than connectivity, and a large part of that is rate-aligned; a small
  stimulus-specific component survives but is not well resolved by 3 checkpoints.
* The raw target's geometry is largely a firing-rate geometry, which is consistent with the
  1-D FIT-rate representation reaching +0.680 while the 48-D structural vector reaches +0.146.
* No representation, target or normalisation is "best": all variants are reported, and no
  post-hoc trend or ranking is fitted.

**Limitations of this stage.** One "stimulus" is one SHD utterance (no repeated-stimulus id);
three checkpoints and one split seed only — checkpoint variability is comparable to the effects
being discussed, so small differences must not be over-read; the bootstrap is the repository's
naive neuron bootstrap; the numbers depend on the target pipeline, so every variant (and the
alternative operation order) is reported rather than selecting the most favourable one; the
activity features are FIT-derived label-free statistics and their stronger correspondence is
measured here, not causally attributed.

---

## 12. Next planned stage

1. **Evaluate** the new label-free sources on PROBE with the existing evaluation layer
   (`scripts/evaluate_vector_capacity.py` / `evaluate_vector_rate_robustness.py`, unchanged):
   `structured_48 + temporal`, `full + residual(functional_response)`, and the rate-decomposition
   targets, so the temporal/functional additions are measured the same way as everything else.
2. Only after that: the `network_context` block, and then multi-layer support.

The functional-response source and the coarse temporal block are implemented and tested but
**not yet evaluated**; no scientific claim is made about them here, and the SNN, the split and
the official TEST set are untouched.