# V2_CONFIG_IMPLEMENTATION.md

**Stage: configuration/control infrastructure only.** No `NeuronRecord`, structured
encoder, learned residual, functional target, notebook, experiment, SNN change or
scientific result is part of this step. The existing pipeline and results are
untouched; this document records what was added and how it was verified.

Date: 2026-09-27. Environment: Python 3.11 (`uv`), torch 2.14.0+cu126.

---

## 1. Files changed

| File | Status | Purpose |
|---|---|---|
| `src/v2_config.py` | **added** | V2 configuration dataclasses (`VectorConfig`, `MemoryConfig`, `PrecisionConfig`, `NetworkConfig`, `ExperimentConfig`, `SimulationConfig`, `V2Config`), validation, dimension resolution, token-budget helpers |
| `configs/v2_example.yaml` | **added** | Tiny demonstration config (not read by any pipeline; `run.synthetic: true`) |
| `scripts/show_v2_config.py` | **added** | Resolve + print/validate a config (no model, no data, no experiment) |
| `tests/test_v2_config.py` | **added** | 59 focused tests for the configuration infrastructure |
| *(any existing file)* | **unchanged** | `src/model.py`, `src/training.py`, `src/data.py`, `src/neurons.py`, all scripts, all existing configs, checkpoints, results, figures |

The module follows the repository's existing configuration conventions: dataclasses
with `to_dict` / `from_mapping` / `from_config` (like `SNNConfig`, `TrainConfig`,
`FingerprintConfig`) built on the existing `src.utils.Config` container, so the same
YAML files, the same `--override dotted.key=value` mechanism and direct Python
construction all work. No second configuration framework was introduced.

---

## 2. New configuration fields

### `vector:` (new section)

| Field | Type | Meaning |
|---|---|---|
| `vector.d` | int \| null | total vector dimension per neuron |
| `vector.structured_d` | int \| null | deterministic structured part |
| `vector.learned_residual_d` | int \| null | learned residual part |
| `vector.enabled_blocks` | list[str] or string | record blocks to include: `intrinsic`, `input_conn`, `recurrent_in`, `recurrent_out`, `activity`, `temporal`, `network_context` |
| `vector.temporal_resolution` | int | coarse time bins for the future temporal block |
| `vector.context_depth` | int | graph-context hops for the future `network_context` block |
| `vector.residual.enabled` | bool | learned-residual switch (not implemented in this stage) |

### `memory:` (new section)

| Field | Type | Meaning |
|---|---|---|
| `memory.train_batch_size` | int | training batch (BPTT) |
| `memory.eval_batch_size` | int | evaluation batch (no recorded traces) |
| `memory.record_batch_size` | int | recorded-activity/trace batch (audit: 32-64) |
| `memory.activity_chunk_size` | int | chunking for activity collection |
| `memory.representation_chunk_size` | int | chunking over neurons when building vectors |
| `memory.device` | `"auto" \| "cpu" \| "cuda"` | compute device |
| `memory.storage` | `"cpu" \| "gpu" \| "memmap"` | where large neuron representations live (configuration only) |
| `memory.mixed_precision` | bool | AMP switch (not implemented in this stage) |
| `memory.max_input_tokens` | int | dense-input guard `B*T*C`; `0` disables |

### `precision:` (new section)

| Field | Type | Meaning |
|---|---|---|
| `precision.vector_dtype` | `float32 \| float16 \| bfloat16` (aliases `fp32/fp16/bf16/half`) | stored neuron-vector precision |
| `precision.activity_dtype` | same | activity/trace accumulation precision |
| `precision.model_dtype` | same | SNN compute precision (only `float32` implemented) |

### `model:` (existing block, future optional keys; nothing else changes)

| Field | Type | Meaning |
|---|---|---|
| `model.model_type` | str | network family; only `recurrent_lif` is supported |
| `model.n_layers` | int | future multi-layer count (only 1 implemented) |
| `model.neurons_per_layer` | list[int] \| null | future per-layer widths; must agree with `n_hidden` for one layer |

`model.n_hidden` keeps its name (no silent rename to `n`).

### `experiment:` (new section; all-null derives from the existing blocks)

| Field | Type | Meaning |
|---|---|---|
| `experiment.dataset` | `"shd" \| "synthetic"` \| null | null → derived from `run.synthetic` |
| `experiment.split` | `"speaker_aware" \| "stratified"` \| null | null → derived from `data.prefer_speaker_aware` |
| `experiment.n_fit` | int \| null | **declared expected** FIT size (asserted, never used to re-split) |
| `experiment.n_probe` | int \| null | **declared expected** PROBE size (asserted, never used to re-split) |
| `experiment.seed` | int \| null | null → the top-level `seed` |

### Simulation (no new section)

`n_bins` / `bin_ms` remain the canonical keys in the existing `model` block; they
are exposed as a typed `SimulationConfig` view with derived `dt_ms` (= `bin_ms`),
`duration_ms` and `duration_s`. This avoids a second source of truth: the 700-bin /
2 ms baseline is unchanged.

### Python API

```python
from src.utils import Config, load_config
from src.v2_config import V2Config

v2 = V2Config.from_config(load_config("configs/neuron_space_baseline.yaml"))   # strict=False default
v2 = V2Config.from_config(cfg, strict=True, warn=False)                        # reject unimplemented requests
v2.vector.d, v2.memory.record_batch_size, v2.precision.vector_dtype            # resolved values
v2.snn, v2.train                                                               # the existing dataclasses
v2.memory.check_input_budget(batch_size=256, n_bins=700, n_input=700)          # token-budget guard
v2.to_dict(); v2.summary_rows(); v2.warnings
```

---

## 3. Defaults

Loading a config **without** any V2 section yields:

| Setting | Default | Rationale |
|---|---|---|
| `vector.d` / `structured_d` / `learned_residual_d` | 48 / 48 / 0 | equals the current structural representation |
| `vector.residual.enabled` | false | no residual exists yet |
| `vector.enabled_blocks` | `[intrinsic, input_conn, recurrent_in, recurrent_out]` | the current primary representation |
| `vector.temporal_resolution` / `context_depth` | 10 / 0 | coarse PSTH convention / no graph expansion |
| `memory.train_batch_size` | `train.batch_size` (128 in current configs) | preserves the existing training regime |
| `memory.eval_batch_size` | `train.eval_batch_size` (256) | preserves the existing evaluation regime |
| `memory.record_batch_size` | 32 | audit recommendation (32-64); removes trace duplication |
| `memory.activity_chunk_size` | 32 | same audit rationale |
| `memory.representation_chunk_size` | 256 | CPU-side chunking over neurons |
| `memory.device` | `run.device` else `"auto"` | no second device setting |
| `memory.storage` | `"cpu"` | audit: large representations stay CPU-side |
| `memory.mixed_precision` | false | training precision is not changed |
| `memory.max_input_tokens` | 150 000 000 | above the current maximum `256*700*700 = 125.44M`; `0` disables |
| `precision.*` | `float32` | repository uses fp32 compute/fp64 stats |
| `model.model_type` / `n_layers` | `recurrent_lif` / 1 | current architecture |
| `experiment.*` | derived from `run.synthetic`, `data.prefer_speaker_aware`, top-level `seed` | no new sources of truth |

**Dimension resolution.** Exactly one of `d`, `structured_d`, `learned_residual_d`
may be omitted (derived from the other two). If more are omitted they fall back to
`structured_d = 48`, `learned_residual_d = 0` and `d` is derived; `d` alone with the
residual enabled is rejected as ambiguous; `d` alone with the residual disabled
means `structured_d = d`. The invariant `d = structured_d + learned_residual_d` is
then enforced. Examples: `structured_d=64 → d=64,r=0`; `d=100,structured_d=48,
residual.enabled=true → r=52`; `d=100 → s=100,r=0`.

---

## 4. Validation rules

All violations raise `V2ConfigError` (a `ValueError` subclass), consistent with the
existing `ValueError`-style configuration errors.

**Always hard errors**

| Rule | Example |
|---|---|
| dimensions non-negative, `d >= 1` | `vector.d=0`, `vector.structured_d=-5` |
| invariant `d == structured_d + learned_residual_d` | `d=100, s=48, r=40` |
| residual flag/capacity agreement | `r=16` with `residual.enabled=false`; `enabled=true` with `r=0` |
| `enabled_blocks` non-empty, known names, no duplicates | `["intrinsic","typo"]`, `[]` |
| `temporal_resolution >= 1` and `<= model.n_bins`; `context_depth >= 0` | `temporal_resolution=0` |
| batch/chunk sizes `>= 1` | `record_batch_size=0` |
| device/storage enums | `device="tpu"`, `storage="tape"` |
| dtype enums/aliases | `vector_dtype="float64"`, `model_dtype="int8"` |
| model type enum, `n_layers` in `[1, 64]`, per-layer width consistency | `n_layers=0`, `n_layers=2, neurons_per_layer=[128]` |
| experiment enums, `n_fit/n_probe >= 1` when set, `seed >= 0` | `dataset="mnist"` |
| impossible dtype/device/storage combinations | half-precision model on CPU; AMP on CPU; AMP with non-fp32 master weights; `storage="gpu"` with `device="cpu"`; `bfloat16` vector/activity with NumPy-backed `cpu`/`memmap` storage |
| dense-input token budget | `B*T*C > memory.max_input_tokens` via `check_input_token_budget` |

**Warnings (raised instead when `strict=True`) — "not implemented in this stage"**

* `model.n_layers != 1` (multi-layer SNN),
* `enabled_blocks` containing `temporal` or `network_context`,
* `vector.residual.enabled=true` (learned residual),
* `precision.model_dtype != float32` (mixed-precision training).

**Advisory warnings**

* `train_batch_size * n_bins * n_input` exceeds `max_input_tokens`;
* half-precision `model_dtype` without `memory.mixed_precision=true`.

**Token-budget helper (audit recommendation)**

```python
estimate_input_tokens(B, T, C)                      # B*T*C
dtype_itemsize("fp16")                              # 2
check_input_token_budget(B, T, C, max_tokens=..., dtype_name=...)
MemoryConfig.check_input_budget(batch_size=..., n_bins=..., n_input=...)
```

The helpers are standalone: they are **not** wired into training/evaluation, so no
existing computation changed. `max_tokens=0` disables the guard.

---

## 5. Backward-compatibility behaviour

* No existing file was modified, so existing scripts keep their current behaviour
  bit-for-bit.
* All four existing configs (`baseline.yaml`, `baseline_repaired.yaml`,
  `neuron_space_baseline.yaml`, `analysis.yaml`) load into `V2Config` with
  `warnings == []` and reproduce their current model/training values (e.g.
  `n_hidden=256`, `n_bins=700`, `readout_mode=sum` for the canonical recipe,
  `l2_spikes=0.001` for the repaired baseline).
* `V2Config.from_config` does not mutate the input `Config` (tested).
* A config without a `memory` section inherits `train.batch_size` /
  `train.eval_batch_size` / `run.device`, so batch and device settings cannot
  silently drift.
* A config without a `vector` section describes exactly today's 48-D structural
  representation with the residual disabled.
* The canonical neuron-space experiment remains constructible: `v2.snn` and
  `v2.train` are the same `SNNConfig` / `TrainConfig` objects the current scripts
  build, and splits, fingerprints, metrics, controls, checkpoints and results are
  untouched.

---

## 6. Tests added

`tests/test_v2_config.py` — 59 tests:

| Requirement | Tests |
|---|---|
| default configuration loads | `test_default_configuration_loads`, `test_from_config_does_not_mutate_the_input_config` |
| existing baseline configs still load | `test_existing_baseline_config_still_loads` (4 configs, parametrised), `test_existing_recipe_values_survive_the_v2_layer` |
| `vector.d` / `structured_d` / `learned_residual_d` overrides | `test_vector_d_can_be_overridden`, `test_vector_structured_d_can_be_overridden`, `test_learned_residual_d_can_be_zero` |
| consistent / inconsistent / invalid dimensions | `test_consistent_dimensions_pass`, `test_inconsistent_dimension_configurations_fail` (4 cases), `test_invalid_dimensions_fail` (4 cases) |
| blocks interface | `test_enabled_blocks_normalization_and_validation`, `test_future_blocks_are_accepted_but_reported_unimplemented`, `test_invalid_temporal_resolution_fails`, `test_negative_context_depth_fails_but_zero_is_valid`, `test_temporal_resolution_cannot_exceed_n_bins` |
| dtype/device/storage validation | `test_invalid_dtype_device_combinations_fail` (5 cases), `test_unsupported_dtype_string_fails`, `test_invalid_device_and_storage_strings_fail`, `test_dtype_aliases_normalize_to_canonical_names`, `test_precision_helper_dtypes` |
| memory overrides / validation | `test_memory_batch_and_chunk_sizes_can_be_overridden`, `test_invalid_batch_and_chunk_sizes_fail` (10 cases), `test_memory_defaults_mirror_train_block_and_explicit_values_win`, `test_memory_device_inherits_run_device` |
| `max_input_tokens` | `test_max_input_tokens_validation`, `test_estimate_input_tokens_and_dtype_itemsize`, `test_input_token_budget_guard`, `test_memory_config_check_input_budget_uses_configured_limit` |
| command-line dotted overrides | `test_dotted_overrides_propagate`, `test_dotted_override_can_raise_the_token_budget`, `test_show_v2_config_script_demonstrates_overrides` |
| network interface | `test_network_view_reads_the_model_block`, `test_invalid_layer_counts_and_model_types_fail`, `test_multi_layer_is_accepted_but_reported_unimplemented`, `test_strict_mode_rejects_unimplemented_requests` |
| experiment / simulation | `test_experiment_view_derives_existing_controls`, `test_experiment_expected_sizes_are_checked_not_applied`, `test_simulation_view_matches_model_block` |
| reporting | `test_to_dict_is_json_serialisable`, `test_summary_rows_cover_the_demo_knobs` |

No tests were added for functionality that does not exist.

## 7. Tests passed

```
uv run pytest tests/test_v2_config.py -q   ->  59 passed
uv run pytest -q                           -> 186 passed (127 pre-existing + 59 new), 0 failed
```

---

## 8. Design decisions and limitations

1. **Single configuration mechanism.** All new sections are dataclasses with
   `from_mapping`/`from_config` on top of `src.utils.Config`. YAML, `--override`
   and Python construction share one validation path; overrides with scientific
   notation and list values are handled by the existing `parse_scalar`.
2. **`d`, `structured_d`, `learned_residual_d` are first-class and cross-checked.**
   The invariant is enforced at construction time, so a contradictory vector
   configuration cannot exist as an object. `learned_residual_d = 0` is valid and
   is the default.
3. **Interface versus implementation is explicit.** `temporal`, `network_context`,
   `n_layers > 1` and the learned residual can be written in a config (so the
   interface is usable for planning) but are reported in `V2Config.warnings`;
   `strict=True` (and `scripts/show_v2_config.py --strict`) turns them into errors.
   Nothing is silently ignored. V2 pipeline code must call
   `V2Config.require_implemented()` before consuming these settings.
4. **No duplicated simulation/experiment sources of truth.** `n_bins`/`bin_ms`
   remain the canonical model keys (exposed as a typed view with derived
   `dt_ms`/`duration_ms`); `dataset`/`split`/`seed` are derived from the existing
   `run.synthetic` / `data.prefer_speaker_aware` / top-level `seed` unless the new
   optional `experiment:` section overrides them.
5. **`experiment.n_fit` / `n_probe` never re-split data.** They are declared
   expectations; `validate_against_split(split_info)` checks them against the
   realised `n_train` / `n_probe` and raises on mismatch. FIT stays the label-free
   source, DEV the model-selection split, PROBE the held-out analysis set and the
   official TEST set is never touched by the neuron-space analysis.
6. **Capacity and precision stay separate.** `vector.d` is capacity;
   `precision.*` is numerical/storage precision. FP8 was not introduced, SNN
   training precision was not changed, and quantisation was not added.
7. **Realistic precision support.** Only `float32`, `float16`, `bfloat16` are
   accepted. `bfloat16` requires `memory.storage="gpu"` because the `cpu`/`memmap`
   backends are NumPy-based and NumPy has no bfloat16. A non-fp32 `model_dtype`
   requires a non-CPU device. `float64` is intentionally not accepted.
8. **`memory.storage` is configuration only.** The storage system itself (cpu/gpu/
   memmap) is not implemented; the modes exist so the future pipeline and its
   validation can be described now. Default is CPU-side storage as recommended by
   the audit.
9. **Token-budget default.** `max_input_tokens = 150 000 000` was chosen to sit
   above the current measured maximum (`256*700*700 = 125.44M` tokens ≈ 478 MiB
   fp32) so existing runs are never blocked, while a 2-4× accidental blow-up is
   caught. `0` disables the guard; recorded passes are separately bounded by
   `record_batch_size` (default 32 → 15.68M tokens).
10. **Limitations.** (a) `model.n_layers > 1` is not implemented and warns/raises;
    the new fields must be relaxed when multi-layer support lands.
    (b) `model.model_type` currently accepts only `recurrent_lif`.
    (c) `neurons_per_layer` is only meaningfully checked for one layer.
    (d) The token guard is a helper, not yet applied inside training/evaluation -
    intentionally, to avoid changing existing behaviour in this stage.
    (e) Warnings are visible on stdout (`[v2-config] warning: ...`) and available
    programmatically via `V2Config.warnings`.

---

## Demonstration (no source edits required)

```powershell
uv run python scripts/show_v2_config.py                     # configs/v2_example.yaml
uv run python scripts/show_v2_config.py --json              # machine-readable resolved config
uv run python scripts/show_v2_config.py --strict            # rejects unimplemented requests

uv run python scripts/show_v2_config.py `
    --override model.n_hidden=128 `
    --override vector.d=64 --override vector.structured_d=48 `
    --override vector.learned_residual_d=16 --override vector.residual.enabled=true `
    --override vector.enabled_blocks="intrinsic,input_conn" `
    --override memory.record_batch_size=16 --override memory.representation_chunk_size=64 `
    --override precision.vector_dtype=fp16 --override memory.device=cpu
```

The observed run with default `configs/v2_example.yaml` resolves
`n_hidden=256, d=48, structured_d=48, learned_residual_d=0, record_batch_size=32,
representation_chunk_size=256, vector_dtype=float32, device=auto` with no warnings;
the override run above changes each of those values and prints a single
"learned residual not implemented" warning; a contradictory override
(`d=100, structured_d=48, learned_residual_d=40`) exits with code 2 and an
actionable error message. This proves the configuration values propagate into the
resolved object and validation layer without editing source code.

**Stop here.** No `NeuronRecord` work has been started; waiting for the next prompt.