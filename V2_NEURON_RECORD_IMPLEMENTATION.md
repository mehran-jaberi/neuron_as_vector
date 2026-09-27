# V2_NEURON_RECORD_IMPLEMENTATION.md

**Stage: the unified label-free `NeuronRecord` layer (infrastructure/refactoring only).**
No learned residual, no variable-dimensional encoder, no new functional target, no
experiment, no notebook, no change to the model, training, data splits, fingerprints,
metrics, controls, checkpoints or any existing scientific result.

Date: 2026-09-27. Environment: Python 3.11 (`uv`), torch 2.14.0+cu126, RTX 3060 Laptop.

---

## 1. Files changed

| File | Status | Purpose |
|---|---|---|
| `src/neuron_record.py` | **added** | `RecordBlock`, `NeuronRecordBank`, `build_neuron_record_bank`, compatibility encoder, memory guard |
| `tests/test_neuron_record.py` | **added** | 45 tests (structure, orientation, leakage, compatibility, memory) |
| `V2_NEURON_RECORD_IMPLEMENTATION.md` | **added** | this document |
| *(existing files)* | **unchanged** | `src/model.py`, `src/training.py`, `src/data.py`, `src/neurons.py`, `src/evaluation.py`, `src/v2_config.py`, all scripts, configs, results |

`git status` shows only the two new source/test files; no existing file was modified.

---

## 2. New API

```python
from src.neuron_record import (
    NeuronRecordBank, RecordBlock, NeuronRecordError,
    build_neuron_record_bank, structural_matrix_from_record_bank, validate_record_array,
)

# from an in-memory model ...
bank = build_neuron_record_bank(model, fit_rec=fit_rec, config=v2_config)
# ... or directly from a checkpoint
bank = NeuronRecordBank.from_checkpoint("checkpoints/sweep_l2_0.pt", fit_rec=fit_rec, config=v2_config)
bank = NeuronRecordBank.from_model(model, with_activity=False)

bank.n_neurons            # 256  (indexed by neuron)
bank.feature_names        # ("intrinsic.learned_bias", "input_conn.entropy", ...) deterministic
bank.block_names          # ("intrinsic", "input_conn", "recurrent_in", "recurrent_out", "activity")
bank.uses_labels          # False  (read-only property; hard invariant)
bank.get_block("input_conn")          # RecordBlock (columnar arrays)
bank.block_status()                   # implemented / present / reason, per declared block
bank.to_structured_matrix(blocks=..., chunk_size=...)   # (X, names), chunk-friendly
bank.to_representation_set(blocks=...)                  # compatibility objects
bank.provenance                       # machine-readable source metadata
bank.assert_no_forbidden_tensors()    # memory-contract re-check
structural_matrix_from_record_bank(bank)                # the current 48-D compatibility encoder
```

The record builder has **no labels parameter**: inputs are the model, either a
label-free `ActivityAccumulatorResult` or (`fit_rec`, `fit_idx`), and an optional
`V2Config`. It never accepts or reads PROBE/TEST data.

---

## 3. Architecture

```
RecordBlock                      NeuronRecordBank
  name                             n_neurons
  n_neurons                        blocks: {name -> RecordBlock}   (canonical order)
  features: {name -> (n,)}         declared_blocks: full V2 schema
  weights:  (n, k) | None          provenance: machine-readable
  flags:    {name -> (n,)}         uses_labels -> always False
  samples:  (n_samples, n) | None
  orientation: str                 → to_structured_matrix()  (n, f)
  source: str                      → to_representation_set() (existing objects)
  metadata: dict
```

* **Columnar, indexed by neuron.** Every quantity is a NumPy array with the neuron
  index as its leading axis. No per-neuron Python object is used as the primary
  representation; the existing `NeuronRepresentation` classes appear only inside the
  explicit compatibility adapter.
* **Immutable/read-only.** Feature, flag, weight and sample arrays are copied and
  flagged read-only; the feature/flag mappings are `MappingProxyType`, so a constructed
  bank cannot drift.
* **Dimension-free.** The bank stores *source information* and the deterministic
  summary statistics the current representation uses. It contains no assumption about
  the final vector dimension `d`; the 48-D matrix is one consumer
  (`structural_matrix_from_record_bank`), not part of the schema.
* **Deterministic.** Block order is the declared V2 order; feature order is
  alphabetical within a block (matching `NeuronRepresentationSet.to_matrix`); two
  identical builds produce bit-identical matrices (tested).

---

## 4. Implemented blocks (exact contents, shapes, orientations)

Measured on the canonical 256-neuron baseline (`checkpoints/sweep_l2_0.pt`):
structural build 0.11 s, **2.46 MB** total.

### `intrinsic` - 1 feature, no weights

| item | value |
|---|---|
| features | `intrinsic.learned_bias` (256,) |
| source | `intrinsic_features_from_model(model)` - **only parameters that actually vary** |
| rule | an untrained / constant parameter emits **no** intrinsic block (existing rule preserved) |
| bias_tau mode | emits `intrinsic.learned_bias` **and** `intrinsic.tau_mem_ms` (49-D structural) |

### `input_conn` - 14 features, weights (256, 700)

| item | value |
|---|---|
| features | `entropy, l1, l2, max_abs, mean, n_significant_frac, neg_frac, participation_ratio_frac, pos_frac, pos_neg_balance, rms, std, top1pct_share, top5pct_share` |
| `weights` | **(n_neurons, n_input)**, `weights[j] = w_in[:, j]` (1.37 MB) |
| optional | `include_tonotopic=True` adds `channel_com`, `channel_spread` (16 features; off by default, as in the existing representation) |
| orientation | `weights[j, c] = w_in[c, j]` - input weight from channel `c` onto neuron `j` |

### `recurrent_in` - 19 features, weights (256, 256)

| item | value |
|---|---|
| features | the 14 vector/concentration statistics of the neuron's incoming vector **plus** `in_out_asymmetry, in_out_correlation, in_out_cosine, reciprocal_strength, self_connection` |
| `weights` | **(n_neurons, n_hidden)**, `weights[i] = w_rec[i, :]` - the **row** = weights **onto** neuron `i` (0.50 MB) |
| orientation | `weights[i, j] = w_rec[i, j]`; forward computes `s_prev @ w_rec.T`, so neuron `i` receives `sum_j s_prev[j] * w_rec[i, j]` |

### `recurrent_out` - 14 features, weights (256, 256)

| item | value |
|---|---|
| features | the 14 vector/concentration statistics of the neuron's outgoing vector |
| `weights` | **(n_neurons, n_hidden)**, `weights[i] = w_rec[:, i]` - the **column** = weights **from** neuron `i` (0.50 MB) |
| relation | `recurrent_out.weights == recurrent_in.weights.T` (by construction, tested) |

### `activity` - 12 features, optional (5922, 256) counts, 2 flags

| item | value |
|---|---|
| features | `active_bin_fraction, fano_factor, first_spike_latency_ms, log_rate_hz, peak_rate_hz, rate_hz, rate_std_hz, silent_fraction, spike_time_cv, temporal_center_ms, temporal_dispersion_ms, temporal_entropy` |
| flags | `silent_neuron` (bool), `total_spikes` (float) - bookkeeping, not representation features |
| `samples` | **(n_samples, n_neurons) float32** per-sample spike counts (5.78 MB for FIT 5922 x 256); `store_sample_counts=False` disables |
| source | `collect_activity(..., with_labels=False)` + `activity_features_from_psth` + `add_first_spike_features` (existing definitions, reused verbatim) |
| voltage | not stored (`collect_voltage=False`); the 12 existing features do not use membrane-voltage statistics |
| provenance | split name, `n_samples`, `batch_size`, `label_free=True` |

### Declared but not implemented

`temporal` and `network_context` are part of the V2 schema
(`src.v2_config.V2_RECORD_BLOCKS`) but are **absent** from the bank:
`get_block("temporal")` / `to_structured_matrix(blocks=["temporal"])` raise
`NeuronRecordError` explaining that the block is not implemented, and
`block_status()` reports `implemented=False, present=False, reason="not implemented in
this stage"`. **No placeholder or fake features are generated.**

---

## 5. Label-leakage guarantees

| Guarantee | Enforcement | Test |
|---|---|---|
| `uses_labels = False` always | read-only property; assignment raises `AttributeError` | `test_bank_is_label_free_by_construction` |
| no labels input path | `build_neuron_record_bank` has no label/probe/test parameter | `test_builder_accepts_no_labels_argument` |
| labels cannot influence output | relabelled FIT recordings produce a bit-identical bank | `test_scrambled_labels_cannot_change_the_record` |
| class-conditioned data rejected | a supplied accumulator with non-zero `class_psth`/`class_counts`/`class_n`/`class_first_spike_*` or attached `labels` raises | `test_labelled_accumulator_is_rejected` |
| provenance states it | `provenance["uses_labels"] is False`, per-block `uses_labels: False`, `leakage_contract` text | `test_provenance_is_machine_readable`, `test_activity_block_declares_label_free_provenance` |

The architectural separation is unchanged and explicit:

```
FIT, label-free  ->  NeuronRecordBank            (this module; uses_labels = False)
FIT, label-free  ->  later learned residual      (not implemented)
PROBE + labels   ->  functional fingerprint      (src/functional_fingerprint.py only)
official TEST    ->  never read by this module
```

---

## 6. Memory strategy

**Allowed shapes** (CPU, all built): `(n_neurons,)`, `(n_neurons, k)` for
features/weights, `(n_samples, n_neurons)` for activity counts.
**Forbidden and rejected** by `validate_record_array`: any array with ≥ 3 dimensions,
i.e. `(n_neurons, n_samples, n_time)`, `(n_samples, n_time, n_neurons)`,
`(n_neurons, n_samples, d)`, `(n_neurons, n_hidden, d)`, `(n_neurons, n_time, d)` -
plus the wrong orientation for counts `(n_neurons, n_samples)`. The validator runs on
the **original** array shape (before any flattening), so a 3-D tensor cannot slip
through, and `bank.assert_no_forbidden_tensors()` re-checks every stored array.

**Streaming collection.** Activity is accumulated by the existing
`ActivityAccumulator` in sample batches; raw `(samples, time, neurons)` traces exist
only for the current batch inside `collect_activity`. No full FIT trace is retained.
`memory.record_batch_size` (default 32) sets the streaming batch, the V2 dense-input
token budget is checked before the pass, and `memory.storage` must be CPU-side
(`cpu`/`memmap`); `gpu` storage raises.

**Measured (canonical model, real FIT split, batch 32).**

| quantity | value |
|---|---|
| structural bank build | 0.11 s, 2.46 MB (1.37 weights + 1.00 recurrent + 0.09 features) |
| FIT activity pass (5922 samples, T=700) | 62 s, `counts (5922, 256)` fp32 = 5.78 MB |
| **CUDA peak during the FIT pass** | **205 MB** (reserved 290 MB) |
| same pass at batch 256 (audit) | 1582 MB peak / 1706 MB reserved |
| full bank on disk-free CPU storage | ~8.3 MB |

Chunking: `bank.to_structured_matrix(chunk_size=k)` fills the output in neuron chunks
so temporaries are `k x n_features`; the result is bit-identical to the unchunked call
(tested with `k = 1, 7, n`).

---

## 7. Compatibility with the existing representation

The compatibility encoder reuses the **same functions** that define the current
representation (`input_connectivity_features`, `recurrent_incoming_features`,
`recurrent_relationship_features`, `recurrent_outgoing_features`,
`intrinsic_features_from_model`, `activity_features_from_psth`,
`add_first_spike_features`) and the same ordering rules (V2 block order, alphabetical
within block), so the numbers are not re-derived.

| test | result |
|---|---|
| canonical 256-neuron architecture, whole 48-D matrix | names identical, **max abs diff 0.0** |
| real trained checkpoints `sweep_l2_0.pt`, `nsb_seed1.pt`, `nsb_seed2.pt` | names identical, **max abs diff 0.0** (both directions) |
| untrained model (no intrinsic block) | 47-D match, exact |
| `bias_tau` model (dynamic + generic intrinsic) | 49-D match, exact |
| `include_tonotopic=True` | 50-D match, exact |
| per-block selection (`input_conn`, recurrent pair, string form) | exact match |
| `to_representation_set().to_matrix()` vs existing object matrix | exact |
| activity block vs existing activity pipeline (same accumulator) | 12-D match, `<= 1e-12` |
| bank built from `fit_rec` vs from a pre-collected accumulator | exact |

Downstream modules (`src/representations.py`, `src/geometry_analysis.py`,
`src/prediction.py`, `src/controls.py`, fingerprint code) were **not touched** and can
consume either `bank.to_structured_matrix()` or `bank.to_representation_set()`.

---

## 8. Intentionally not implemented (never faked)

```
learned residual                = NOT implemented (configuration interface only)
variable-dimensional encoder    = NOT implemented (only the current 48-D encoder exists)
temporal block                  = NOT implemented
network_context block           = NOT implemented
GPU / memmap storage            = NOT implemented (storage config is validated, CPU used)
bank persistence (save/load)    = NOT implemented (documented as future work)
multi-layer records             = NOT implemented (single hidden layer only)
```

The record schema itself is dimension-free, so future encoders (`structured_d = 20/48/64/100`)
can read the same bank without changing it - but **no such encoder exists yet**.

---

## 9. Tests run and results

```
uv run pytest tests/test_neuron_record.py -q   ->  45 passed   (0 skipped: all three
                                                   canonical checkpoints were present)
uv run pytest -q                               -> 231 passed   (186 pre-existing + 45 new)
```

New tests (`tests/test_neuron_record.py`): columnar structure and block shapes;
deterministic, qualified, alphabetically ordered feature names; read-only arrays;
input/recurrent orientation with a unique-value toy matrix; forward-pass grounding of
the incoming-row semantics; label-free guarantees and scrambled-label invariance;
labelled-accumulator rejection; canonical/untrained/dynamical/tonotopic/block-selection
compatibility; activity compatibility and flags; chunking; forbidden-shape guards;
sample-major orientation guard; GPU-storage and token-budget rejections;
declared-but-unimplemented blocks; provenance/JSON; determinism; `from_checkpoint`
vs `from_model`.

---

## 10. Unavoidable deviations from the specification

1. **Recurrent orientation (spec sections 4C/4D) - repository convention wins.**
   The specification states `recurrent_in(i) = w_rec[:, i]` and
   `recurrent_out(i) = w_rec[i, :]`, which assumes the convention
   `w_rec[i, j] = weight from i to j`. This repository uses the **opposite**
   convention (`src/model.py`: `w_rec[i, j]` = weight **from j to i**; `forward`
   computes `s_prev @ w_rec.T`), so the incoming weights of neuron `i` are the **row**
   `w_rec[i, :]` and the outgoing weights are the **column** `w_rec[:, i]`. The
   existing 48-D features are defined that way, and acceptance criterion 8 (exact
   numerical reproduction of the current representation) takes precedence over the
   spec's literal transpose. The distinction is therefore preserved, documented and
   tested explicitly: `recurrent_in.weights[i] == w_rec[i, :]`,
   `recurrent_out.weights[i] == w_rec[:, i]`, `out.weights == in.weights.T`, plus a
   forward-pass test showing neuron `i` receives `sum_j s_prev[j] * w_rec[i, j]`.
   Adopting the spec's literal orientation would have silently changed all recurrent
   features relative to the current results.
2. **Activity sample counts are stored by default** (5.78 MB fp32 for FIT), because the
   spec explicitly allows `(n_samples, n_neurons)` counts and they are needed by future
   label-free encoders. `store_sample_counts=False` disables them.
3. **Voltage statistics are not stored.** The existing 12 activity features do not use
   membrane-voltage statistics, so the pass runs with `collect_voltage=False`; adding
   voltage columns would be new science, not a refactor.
4. **`memory.record_batch_size` drives the streaming batch** (the audit's recorded-pass
   knob). `activity_chunk_size` is not consumed yet (the accumulator is already chunked
   by sample batches); both are recorded in provenance.
5. **No serialization.** Saving/loading a bank was not in this stage's scope; the
   in-memory columnar layout is the deliverable.
6. **`NeuronRecordBank` construction does not accept `uses_labels`** - passing it is a
   `TypeError`, and the attribute is a read-only property returning `False`.

---

## 11. Example

```python
from src.model import RecurrentLIFSNN
from src.data import load_shd, make_train_dev_probe_split
from src.utils import load_config
from src.v2_config import V2Config
from src.neuron_record import build_neuron_record_bank, structural_matrix_from_record_bank

cfg = V2Config.from_config(load_config("configs/neuron_space_baseline.yaml"))
model, _ = RecurrentLIFSNN.load("checkpoints/sweep_l2_0.pt", map_location="cpu")
fit_rec, _, _, _ = make_train_dev_probe_split(load_shd("data", download=False)["train"],
                                              dev_fraction=0.1, probe_fraction=0.1, seed=0)

bank = build_neuron_record_bank(model, fit_rec=fit_rec, config=cfg)   # label-free, streaming
X_struct, names = structural_matrix_from_record_bank(bank)            # (256, 48) == existing
X_full, full_names = bank.to_structured_matrix()                      # (256, 60) with activity
reps = bank.to_representation_set()                                   # existing object API
```

**Stop here.** The learned residual, variable-dimensional encoders, temporal and
network-context blocks remain unimplemented, as required for this stage.