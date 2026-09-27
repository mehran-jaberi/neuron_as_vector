# V2_STRUCTURED_VECTOR_IMPLEMENTATION.md

**Stage: repository cleanup/archival + the deterministic variable-dimensional structured
encoder.** The learned residual, trainable encoders, new functional targets, capacity
sweeps and notebook work remain **out of scope and unimplemented**.

Date: 2026-09-27. Environment: Python 3.11 (`uv`), torch 2.14.0+cu126, numpy (repo lock).

The three layers are now clearly separated:

```
NeuronRecordBank         = source information      (src/neuron_record.py)  - label-free, columnar
StructuredVectorEncoder  = deterministic vector    (src/structured_vector.py) - THIS STAGE
learned residual         = future stage            (NOT implemented anywhere)
```

---

## 1. Repository cleanup performed (Part A)

### 1.1 Method

`git status`, `git ls-files` (72 tracked files), the directory tree, and a full
`git grep` reference sweep across `README.md`, every `*.md`, `configs/`, `scripts/`,
`src/` and `tests/` were used to decide, per artifact, whether it is still referenced.
A file was moved **only** if it was (a) untracked/generated, (b) referenced by nothing in
the tracked tree, and (c) unmistakably an intermediate/smoke/debug run output.
Nothing was deleted; contents are preserved exactly; no git history was rewritten.

### 1.2 Result: 51 artifacts archived, 0 tracked files moved

| family | count | destination |
|---|---|---|
| smoke/debug checkpoints (`readme_check.pt`, `shd_tiny.pt`, `synthetic_smoke.pt`, `debug_v2.pt`) | 4 | `archive/intermediate_runs/checkpoints/` |
| smoke/verification run outputs (`readme_check_*`, `shd_tiny_*`, `synthetic_smoke_*`, `debug_v2_*`, `nsb_smoke_*`, `nsb_run.log`) | 47 | `archive/intermediate_runs/results/` |

Archive layout and full per-file manifest: **`archive/README.md`** and
**`archive/ARCHIVE_MANIFEST.md`** (original path, archive path, reason, tracked status,
reference-check status, date). The archived binaries stay out of git via
`archive/intermediate_runs/.gitignore`, mirroring the existing policy for `results/` and
`checkpoints/`; the manifest itself is tracked.

### 1.3 Deliberately kept in place

Every tracked config, script and document is **actively referenced** and was therefore not
archived - in particular:

* `configs/baseline.yaml`, `baseline_repaired.yaml`, `analysis.yaml` are loaded by
  `tests/test_v2_config.py`; `neuron_space_baseline.yaml` is the canonical config and
  references `baseline_repaired.yaml`; `v2_example.yaml` is the demo default.
* all `scripts/*.py` are referenced by README run instructions and/or tests
  (`show_v2_config` is imported by `tests/test_v2_config.py`).
* the audits/reports (`AUDIT.md`, `AUDIT_DYNAMICS.md`, `AUDIT_REPRESENTATION.md`,
  `LIF_BASELINE_REPORT.md`, `NEURON_SPACE_BASELINE_REPORT.md`, `VECTOR_V2_AUDIT.md`, the
  V2 stage reports) are the design record and are linked from README/`src/neurons.py`.
* `results/runcheck_*` and `checkpoints/runcheck.pt` are cited in `AUDIT_DYNAMICS.md`;
  the `sweep_*.pt` checkpoints are referenced by the `lif_baseline_sweep*.csv` checkpoint
  columns and substantiate `LIF_BASELINE_REPORT.md`; `sweep_l2_0.pt`, `nsb_seed1.pt` and
  `nsb_seed2.pt` are exercised by tests.
* `results/neuron_space_baseline/**` (canonical results), `figures/**`, `data/**`.

Ambiguous artifacts were left untouched, as instructed. Verification: `git status` shows
only the new archive documentation and the two new source/test files; the full suite passes
(282 tests) with no broken import, config or path reference.

---

## 2. New module and API (Part B)

`src/structured_vector.py` - additions only; no existing file was modified.

```python
from src.structured_vector import (
    StructuredVectorEncoder, StructuredVectorError, StructuredVectors,
    EncoderPlan, Coordinate, encode_structured_vectors,
    WEIGHT_DETAIL_NAMES, ACTIVITY_DETAIL_NAMES, DETAIL_DEFINITIONS,
    DEFAULT_PROJECTION_SEED, MAX_STRUCTURED_D,
)

# explicit
vectors = encode_structured_vectors(record_bank, structured_d=100)          # -> StructuredVectors
# or bound to the bank + configuration
encoder = StructuredVectorEncoder(record_bank, structured_d=100, enabled_blocks=None,
                                  projection_seed=0, chunk_size=None)
vectors = encoder.encode()

vectors.X                 # (n_neurons, structured_d) float64, CPU
vectors.feature_names     # tuple of coordinate names, length d
vectors.metadata          # per-coordinate dicts (index, name, kind, block, source, definition)
vectors.provenance        # machine-readable encoder provenance
encoder.output_dimension  # d
encoder.source_dimension  # level0 + level1
encoder.feature_names / feature_metadata / seed / uses_labels / provenance / plan
encoder.projection_matrix()   # (source_dimension, projection_dim) or None

# from the existing V2 configuration (no second config system)
encoder = StructuredVectorEncoder.from_config(v2_config, record_bank)       # residual request -> error
encoder = StructuredVectorEncoder.from_config(v2_config, record_bank, on_residual_request="warn")
```

The encoder **only** consumes a `NeuronRecordBank`; it has no fitted state, no `fit`/`train`
method, no optimizer, no loss, and no labels parameter.

---

## 3. Encoder architecture and dimension policy

### 3.1 Canonical coordinate order (the whole design in one list)

```
level 0: for each enabled+present block, in canonical block order,
         its existing summary features alphabetically           <- source of truth: the record bank
level 1: for each enabled+present block, in canonical block order,
         its deterministic within-block detail in fixed order   <- computed here, named + documented
level 2: fixed seeded data-independent Gaussian projection      <- only if structured_d > source_dimension
```

`structured_d` is a **prefix** of that order:

| request | result |
|---|---|
| `structured_d <= level0_dimension` | first `structured_d` Level-0 coordinates (deterministic selection, no transform) |
| `level0 < structured_d <= source_dimension` | all Level 0 + the first `structured_d - level0` Level-1 coordinates |
| `structured_d > source_dimension` | all Level 0 + all Level 1 + `structured_d - source` projected coordinates |

The prefix property is tested across dimension pairs (and across the projection boundary):
extending `d` never changes an existing coordinate.

### 3.2 Level 0 - the existing deterministic summaries

Level 0 is read verbatim from the record bank (`block.features[...]`), in the **same order
as** `NeuronRecordBank.to_structured_matrix`: canonical block order
(`intrinsic, input_conn, recurrent_in, recurrent_out, activity, temporal, network_context`)
and alphabetical within a block. Names are the established qualified form
(`intrinsic.learned_bias`, `input_conn.entropy`, ...). No formula is duplicated and no
feature is redefined, renormalised or reordered. A block that is implemented but absent in
the bank (e.g. `intrinsic` for an untrained model, `activity` when no FIT data was supplied)
contributes nothing - the encoder inspects block availability instead of inventing columns.
`temporal` and `network_context` raise (see §7).

### 3.3 Level 1 - deterministic within-block detail

**Weight blocks** (`input_conn`, `recurrent_in`, `recurrent_out` - any block storing
`weights`), 31 coordinates per block, in this fixed order (names shown without the block
prefix; the full name is e.g. `input_conn.abs_q90`):

| group | names | definition |
|---|---|---|
| signed quantiles | `q01 q05 q10 q25 q50 q75 q90 q95 q99` | quantiles of the stored weight vector (levels 0.01 ... 0.99) |
| absolute quantiles | `abs_q01 ... abs_q99` | quantiles of the absolute weights |
| absolute mass histogram | `abs_hist_00 ... abs_hist_07` | fraction of the total `|w|` mass in 8 equal-width bins spanning `[0, max|w|]` |
| top-k signed | `top_abs_1 top_abs_2 top_abs_3` | signed value of the k-th largest `|w|` (stable sort, deterministic under ties) |
| threshold shares | `share_gt_mean_abs share_gt_2mean_abs` | fraction of weights with `|w|` above `mean|w|` / `2·mean|w|` |

**Activity block** (only when per-sample counts are stored, i.e. `store_sample_counts=True`),
8 coordinates: `count_q05 count_q25 count_q50 count_q75 count_q95 frac_zero_samples
max_count mean_nonzero_count`, all computed from the stored label-free FIT counts.

All Level-1 quantities are pure deterministic functions of vectors the record bank already
stores, are finite for degenerate inputs (all-zero rows guarded), and carry a written
definition in `DETAIL_DEFINITIONS` (exposed in the plan/provenance, so no coordinate is an
anonymous number). On the canonical 256-neuron model: **level 0 = 48, level 1 = 93
(31 × 3 weighted blocks), source dimension = 141** (with the activity block enabled:
level 0 = 60, level 1 = 101, source dimension = 161).

### 3.4 Level 2 - fixed seeded projection

* Type: fixed Gaussian, scaling `1/sqrt(source_dimension)`; **not learned**, no fitting,
  no PCA/ICA/NMF/autoencoder/UMAP/t-SNE (explicitly out of scope per spec B10).
* Seed: `projection_seed`, default `DEFAULT_PROJECTION_SEED = 0` (the only stochastic
  element and it is data-independent). `from_config` uses the same default.
* Generation: **column `j` is drawn from `numpy.random.default_rng([seed, j])`**, so the
  projection is *prefix-stable*: increasing `structured_d` never changes existing projected
  coordinates.
* Input: the `(n_neurons, source_dimension)` matrix (Level 0 + Level 1), never activity
  traces or per-sample tensors.
* Provenance: `type`, `seed`, `source_dimension`, `output_dimension`, `scaling`,
  `generator`, `interpretable: False`, `learned: False`.
* Caveat (documented): the projection is applied to the raw, unstandardised source matrix,
  so projected coordinates are mixtures dominated by large-scale features and are **not**
  individually interpretable. Downstream standardisation (e.g.
  `RepresentationSpace`) is a separate, later choice.

### 3.5 Enabled blocks

`enabled_blocks` accepts a list or a comma/space-separated string; names are validated
against the record bank's declared blocks, de-duplicated and re-ordered canonically.
`None` means the canonical default block set
(`src.v2_config.DEFAULT_ENABLED_BLOCKS` = `intrinsic, input_conn, recurrent_in,
recurrent_out`), which is the historical 48-D anchor. Enabling `activity` (when the bank has
it) adds its 12 Level-0 features - and, at larger `d`, its 8 Level-1 count summaries. Unknown
names and declared-but-unimplemented blocks (`temporal`, `network_context`) raise
`StructuredVectorError`; no placeholder features are generated.

### 3.6 V2 configuration integration

`StructuredVectorEncoder.from_config(config, bank)` uses the existing `V2Config`:

| config field | effect |
|---|---|
| `vector.structured_d` | output dimension `d_struct` of this encoder |
| `vector.enabled_blocks` | block selection |
| `memory.representation_chunk_size` | default `chunk_size` (overridable per call) |
| `vector.residual.enabled` / `vector.learned_residual_d` | **not implemented**: default `on_residual_request="error"` raises; `"warn"` returns the structured part only and records the request in `warnings` + `provenance` (`residual_implemented: False`). The residual is never silently pretended to exist, and a requested non-zero residual dimension is never filled with fabricated coordinates |
| `vector.temporal_resolution` / `vector.context_depth` | recorded in provenance; they do not create features (temporal/context blocks are unimplemented) |
| `vector.d` | recorded; the composition `z = [z_structured, z_residual]` belongs to a future stage |

---

## 4. Provenance

Per coordinate (`vectors.metadata[i]` / `encoder.feature_metadata[i]`):
`index`, `name`, `kind` (`level0` | `level1` | `projection`), `block`, `source`,
`definition`, `uses_labels` (always `False`).

Encoder provenance (`vectors.provenance`): `schema`, `uses_labels: False`,
`deterministic: True`, `fitted_state: False`, `learned: False`,
`residual_implemented: False`, `structured_d`, `output_dimension`,
`level0_dimension`, `level1_dimension`, `source_dimension`, `requested_blocks`,
`present_blocks`, `absent_blocks`, `block_detail_dimensions`, `level1_definitions`,
`selection_policy`, `projection` (or `{"used": False}`), `projection_used`,
`projection_seed`, `chunk_size`, `bank` (schema, `n_neurons`, `uses_labels: False`),
`configured` (the V2 vector/memory snapshot when built via `from_config`), `warnings`.
Everything is JSON-serialisable (tested).

---

## 5. Memory behaviour

Measured on the canonical 256-neuron model (`sweep_l2_0.pt`, structural bank 2.46 MB):

| `structured_d` | time | `X` size | projection matrix |
|---|---|---|---|
| 48 | ~0 ms (bank slicing only) | 0.09 MB | none |
| 100 | ~94 ms | 0.20 MB | none |
| 141 (= source) | ~67 ms | 0.28 MB | none |
| 160 | ~80 ms | 0.31 MB | 0.020 MB (141 × 19) |
| 1000 | ~100 ms | 1.95 MB | 0.92 MB (141 × 859) |
| 1000, `chunk_size=16` | ~120 ms | 1.95 MB | same (peak temporaries bounded by the chunk) |

* CPU-side only; no GPU tensors, no `torch` dependency in this module.
* Largest arrays: `(n_neurons, source_features)`, `(n_neurons, structured_d)` and
  `(source_features, projection_dim)` - all 2-D.
* `chunk_size` bounds the per-chunk temporaries to `chunk × source_features`.
  Source (Level 0/1) coordinates are **bit-identical** between chunked and unchunked
  construction; projected coordinates agree within floating-point tolerance
  (different matrix shapes can make BLAS reorder summations). Both facts are tested.
* The forbidden tensors `(n_neurons, n_samples, d)`, `(n_neurons, n_time, d)`,
  `(n_neurons, n_neurons, d)`, `(n_neurons, n_samples, n_time)` are never created -
  the encoder has no sample or time axis at all.

---

## 6. Exact 48-D compatibility

`structured_d=48` with the default blocks is the historical anchor and is reproduced
**exactly**:

| target | result |
|---|---|
| canonical 256-neuron architecture (from `configs/neuron_space_baseline.yaml`) | names identical, **max abs diff 0.0**, shape (256, 48) |
| real checkpoints `sweep_l2_0.pt`, `nsb_seed1.pt`, `nsb_seed2.pt` | names identical, **max abs diff 0.0** |
| `d = level0_dimension` (no transform path) | bit-identical to `bank.to_structured_matrix(...)` |
| untrained model (no intrinsic block) | exact at `d = 47` (= its level-0 dimension) |
| `bias_tau` model (dynamical + generic intrinsic) | exact at `d = 49` |

No feature is redefined, no normalisation is introduced, and no ordering changes: Level 0 is
read from the same arrays the compatibility path uses.

---

## 7. Explicitly unimplemented (never faked)

```
learned residual             = NOT implemented (config request -> error, or warn+structured-only)
trainable/learned encoder    = NOT implemented (no MLP, autoencoder, contrastive objective)
variable-d fitting (PCA etc) = NOT implemented (the encoder has no fitted state)
temporal block               = NOT implemented (request -> StructuredVectorError)
network_context block        = NOT implemented (request -> StructuredVectorError)
new functional targets       = NOT implemented
PROBE/TEST usage             = none (only a label-free NeuronRecordBank is consumed)
capacity sweep / experiments = NOT performed in this stage
notebook                     = not part of this stage
multi-layer support          = NOT implemented
```

---

## 8. Supported neuron counts

The encoder contains no assumption about `n_hidden`; it is exercised at **32, 64, 128 and
256** hidden neurons (toy `SNNConfig` models with varying per-neuron bias) with output shape
always `(n_neurons, structured_d)`. The canonical 256-neuron architecture is additionally
checked against the historical representation. The SNN itself was not modified.

---

## 9. Tests run and results

```
uv run pytest tests/test_structured_vector.py -q   ->  51 passed
uv run pytest tests/test_neuron_record.py -q       ->  45 passed   (unchanged)
uv run pytest tests/test_v2_config.py -q           ->  59 passed   (unchanged)
uv run pytest -q                                   -> 282 passed   (231 pre-existing + 51 new)
```

New tests cover: exact 48-D compatibility (canonical architecture + three trained
checkpoints + untrained + `bias_tau`); Level-0/Level-1 dimensions, names and ordering;
lower dimensions (1, 2, 8, 16, 32, 47) as deterministic prefixes; `d = level0` returning the
source matrix unchanged; higher dimensions (49-141) adding documented Level-1 detail; the
projection appearing only beyond the source dimension, its determinism (same seed →
identical, different seed → different projected columns, source columns unchanged) and its
prefix stability; block selection (including `activity`); neuron counts 32/64/128/256;
chunking equality; 2-D-only memory assertions and invalid-input rejection; label-free
guarantees (no label/PROBE/TEST parameter, relabelled FIT data cannot change the output);
absence of fitted state and repeatability; complete, JSON-serialisable provenance; V2-config
integration including the residual error/warn semantics and unimplemented-block rejection.

---

## 10. Deviations from the specification

1. **Projection prefix stability (improvement over the literal spec).** The spec asks for a
   fixed seeded projection; seeding a single `default_rng(seed)` would make the projected
   columns depend on the requested output width (NumPy fills a `(source, k)` draw in
   row-major order). The implementation therefore seeds **each column** with
   `default_rng([seed, column])`, so the whole coordinate system - including projected
   coordinates - obeys the prefix property.
2. **Chunked vs unchunked equality is exact for source coordinates and tolerance-based for
   projected coordinates.** Bit-identical results across different BLAS shapes cannot be
   guaranteed; the contract is documented and tested as
   `rtol=1e-12, atol=1e-12` (measured max difference 0.0 on this machine).
3. **`structured_d` for an untrained model.** Because the record bank follows the existing
   rule "emit an intrinsic feature only if it varies across neurons", an untrained model has
   a 47-D Level 0. The historical anchor for such a model is therefore `d = 47` (tested);
   `d = 48` would legitimately pull the first Level-1 detail coordinate.
4. **Default `enabled_blocks = None` means the four structural blocks** (the historical
   anchor), not "all present blocks". `activity` participates only when explicitly enabled,
   matching `vector.enabled_blocks` in the configuration and keeping the 48-D default exact.
5. **No capacity evaluation.** Per the spec, this stage performs no experiment, no geometry,
   no prediction and no sweep - only machinery plus its compatibility guarantee.

**Stop here.** `z_structured` exists; `z_residual` does not, and no learned component,
variable-dimensional fitting or new scientific evaluation has been introduced.