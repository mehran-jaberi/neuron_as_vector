# V2 Control Panel — editable-parameter reference

`V2_Control_Panel.ipynb` is the **primary interactive interface** to the frozen V2 neuron-vector
pipeline. It configures one representation, builds it (FIT only), inspects the frozen artifact, and
optionally evaluates it on PROBE. Every value you edit is translated by `src/v2_panel.py` into the
same calls the headless CLI uses (`scripts/v2_control_panel.py`), so the notebook, the CLI and the
library produce identical configurations, run ids and artifacts.

This file documents what each editable variable means. It is a companion to the notebook, not a
replacement: the notebook cells are the source of truth and print the resolved values.

> **Safety.** Representation construction is **FIT-only**. Evaluation uses **PROBE** targets and
> metrics only. The official **TEST** recordings are never accessed by this panel (or by the CLI).

---

## How to use the notebook

1. Open `V2_Control_Panel.ipynb` and run the cells top → bottom.
2. Each control cell prints the current values and defines `# <- EDITABLE` variables.
3. Edit those variables, then re-run the cell.
4. Re-run **section 11 (Validate + preview)** — it is the notebook's dry run: it loads no data,
   trains nothing and evaluates nothing. It is equivalent to
   `uv run python scripts/v2_control_panel.py --preset <preset> --dry-run`.
5. Then run **section 13 (Build)**, **section 14 (Inspect)** and, if you want, **section 15
   (Evaluate)**.

Headless / automated runs use the environment variables the notebook documents:

| Variable | Overrides |
|---|---|
| `V2_PANEL_CONFIG` | `CONFIG_PATH` |
| `V2_PANEL_CHECKPOINT` | `CHECKPOINT` |
| `V2_PANEL_OUT_DIR` | `OUT_DIR` |

### Conventions used below

* **Panel-only** — a notebook/session setting that is *not* a YAML key; it is passed to the resolver
  as a dedicated argument (config file, preset, checkpoint, output dir, evaluation settings).
* **Config-backed** — maps to a dotted configuration key. When the value differs from the preset
  baseline it becomes a `--override key=value` entry; the notebook shows the exact override list in
  section 16.
* **Derived / display-only** — `d`, `n_layers`, `network_context`. These are computed and cannot be
  set. `network_context` and multi-layer SNNs are **not implemented**.
* `d = structured_d + learned_residual_d` always holds. Changing a representation-affecting control
  changes the **run id** and therefore the artifact; evaluation settings, `out_dir` and the preset
  label do not.
* Defaults shown are the ones resolved from `configs/neuron_space_baseline.yaml` with the
  `historical_48` preset. A different config or preset changes them — the notebook always prints the
  live values.

---

## 1. Panel settings — section 1

| Variable | Panel control | Type | Default | Meaning |
|---|---|---|---|---|
| `CONFIG_PATH` | `config_path` | path | `configs/neuron_space_baseline.yaml` | YAML configuration file to start from (relative paths are anchored to the repository root). |
| `OUT_DIR` | `out_dir` | path | `results/neuron_vector_capacity` | Artifact directory: `vectors/`, `residuals/`, `cache/`, `resolved_configs/`, `evaluations/`. |
| `CHECKPOINT` | `checkpoint` | path | `checkpoints/sweep_l2_0.pt` | Frozen SNN checkpoint used to produce the record bank. `model.*` must match it exactly. |

## 2. Preset — section 2

| Variable | Panel control | Type | Default | Meaning |
|---|---|---|---|---|
| `PRESET` | `preset` | choice (panel-only) | `historical_48` | A named configuration from the backend. `None` = the current config default. A preset only sets values; it is **not** a hidden execution path. |

Available presets (also printed by the cell and `--list-presets`):

| Preset | Decomposition | Description |
|---|---|---|
| `historical_48` | `48 = 48 + 0` | The historical 48-D structural representation (bit-identical regression anchor). |
| `structured_64` | `64 = 64 + 0` | Wider purely deterministic structural vector (level-0 48 + 16 level-1 coordinates). |
| `structured_100` | `100 = 100 + 0` | Widest deterministic structural vector before projection (level-0 48 + 52 level-1). |
| `temporal_structured` | `58 = 58 + 0` | Default structural blocks **+** 10 coarse temporal bins. |
| `functional_64` | `64 = 48 + 16` | Residual consumes structural **+** functional-response source (arm B at `residual_d=16`). |
| `functional_100` | `100 = 48 + 52` | Same source at `residual_d=52`. |

## 3. Representation dimensions — section 3

| Variable | Panel control | Config key | Type / allowed | Default | Meaning |
|---|---|---|---|---|---|
| `STRUCTURED_D` | `structured_d` | `vector.structured_d` | int ≥ 1 (encoder ceiling 16384) | `48` | Number of deterministic structured coordinates. Because `d` is derived, you may also set the total with `panel.set_controls(state, structured_d=TOTAL_D, learned_residual_d=0)`. |
| `LEARNED_RESIDUAL_D` | `learned_residual_d` | `vector.learned_residual_d` | int ≥ 0 | `0` | Number of learned-residual output coordinates. `0` disables the residual (no model is instantiated). Must be `> 0` for any residual **source** to have an effect. |

The cell prints `d = structured_d + residual_d` and flags any combination the backend would reject.

## 4. Structural blocks — section 4

| Variable | Panel control | Config key | Type / allowed | Default | Meaning |
|---|---|---|---|---|---|
| `ENABLED_BLOCKS` | `enabled_blocks` | `vector.enabled_blocks` | list (or comma-separated string) of: `intrinsic`, `input_conn`, `recurrent_in`, `recurrent_out`, `activity`, `temporal` | `[intrinsic, input_conn, recurrent_in, recurrent_out]` | Which record blocks the **structured encoder** selects. At least one is required; unknown names are rejected. `network_context` is declared but **not implemented** and is not selectable. |

## 5. Functional-response source — section 5

Label-free source view of the FIT per-stimulus response profile (a fixed, seeded projection). It is an
input of the **learned residual**, so it requires `residual_enabled` with `learned_residual_d > 0`.

| Variable | Panel control | Config key | Type / allowed | Default | Meaning |
|---|---|---|---|---|---|
| `FUNCTIONAL_RESPONSE` | `source_functional_response` | `vector.residual.source_functional_response` | bool | `False` | Let the residual consume the functional-response projection. |
| `FUNCTIONAL_SOURCE_DIM` | `functional_source_dim` | `vector.functional_source_dim` | int, 1…4096 | `64` | Projection dimension of the response profile. |
| `FUNCTIONAL_PROJECTION_SEED` | `functional_projection_seed` | `vector.functional_projection_seed` | int ≥ 0 | `0` | Seed of the fixed projection (never shared with other seeds). |
| `FUNCTIONAL_NORMALIZATION` | `functional_source_normalization` | `vector.functional_source_normalization` | `raw` \| `neuron_centered` \| `neuron_zscored` | `raw` | Per-neuron transform applied before the projection. |

## 6. Temporal source — section 6

The coarse label-free FIT temporal block. It can be selected by the **structured encoder** (section 4)
and/or consumed by the **learned residual** here; the two switches are independent.

| Variable | Panel control | Config key | Type / allowed | Default | Meaning |
|---|---|---|---|---|---|
| `TEMPORAL_SOURCE` | `source_temporal` | `vector.residual.source_temporal` | bool | `False` | Let the residual consume the coarse temporal block. |
| `TEMPORAL_RESOLUTION` | `temporal_resolution` | `vector.temporal_resolution` | int ≥ 1 | `10` | Number of coarse FIT PSTH bins in the temporal block. |

## 7. Residual protocol — section 7

The learned residual is the only trained component (FIT-only, self-supervised). The cell prints the
**drift** whenever a value differs from the established protocol, so a modified protocol can never go
unnoticed.

| Variable | Panel control | Config key | Type / allowed | Default | Meaning |
|---|---|---|---|---|---|
| `RESIDUAL_ENABLED` | `residual_enabled` | `vector.residual.enabled` | bool | `False` | Train a learned residual (requires `learned_residual_d > 0`). |
| `RESIDUAL_SEED` | `residual_seed` | `vector.residual.seed` | int ≥ 0 | `0` | Training seed of the residual. |
| `RESIDUAL_SPLIT_SEED` | `residual_split_seed` | `vector.residual.split_seed` | int ≥ 0 | `0` | Seed of the neuron-level train/validation split inside FIT. |
| `RESIDUAL_MASK_SEED` | `residual_mask_seed` | `vector.residual.mask_seed` | int ≥ 0 | `0` | Seed of the coordinate-mask generation. |
| `RESIDUAL_HIDDEN_DIM` | `residual_hidden_dim` | `vector.residual.hidden_dim` | int ≥ 1 | `64` | Hidden width of the residual MLPs. |
| `RESIDUAL_EPOCHS` | `residual_epochs` | `vector.residual.epochs` | int ≥ 1 | `200` | Training epochs (established protocol: 200). |
| `RESIDUAL_BATCH_SIZE` | `residual_batch_size` | `vector.residual.batch_size` | int ≥ 1 | `64` | Training batch size (established protocol: 64). |
| `RESIDUAL_LEARNING_RATE` | `residual_learning_rate` | `vector.residual.learning_rate` | float ≥ 0 | `0.001` | Adam learning rate (established protocol: `1e-3`). |
| `RESIDUAL_MASK_FRACTION` | `residual_mask_fraction` | `vector.residual.mask_fraction` | float, 0…0.999 | `0.25` | Fraction of coordinates withheld per example (established: 0.25). |
| `RESIDUAL_MIN_VISIBLE` | `residual_minimum_visible_features` | `vector.residual.minimum_visible_features` | int ≥ 1 | `8` | Minimum coordinates kept visible per example. |
| `RESIDUAL_VAL_FRACTION` | `residual_val_fraction` | `vector.residual.val_fraction` | float, 0…0.499 | `0.2` | Validation fraction of the FIT neurons (established: 0.2). |
| `RESIDUAL_STANDARDIZATION` | `residual_standardization` | `vector.residual.standardization` | `train_standardise` \| `none` | `train_standardise` | Standardization of the residual input view. |
| `RESIDUAL_MASK_MODE` | `residual_mask_mode` | `vector.residual.mask_mode` | `coordinate` \| `block` | `coordinate` | Coordinate masking (default) or whole-source block masking (diagnostic only). |

## 8. Compute / memory / precision — section 8

| Variable | Panel control | Config key | Type / allowed | Default | Meaning |
|---|---|---|---|---|---|
| `DEVICE` | `device` | `memory.device` | `auto` \| `cpu` \| `cuda` | `auto` | Device for activity collection and residual training. |
| `STORAGE` | `storage` | `memory.storage` | `cpu` \| `gpu` \| `memmap` | `cpu` | Record-bank storage. **Only `cpu` is implemented**; other values are reported as warnings. |
| `TRAIN_BATCH_SIZE` | `train_batch_size` | `memory.train_batch_size` | int ≥ 1 | `128` | Training batch size. |
| `EVAL_BATCH_SIZE` | `eval_batch_size` | `memory.eval_batch_size` | int ≥ 1 | `256` | Evaluation batch size. |
| `RECORD_BATCH_SIZE` | `record_batch_size` | `memory.record_batch_size` | int ≥ 1 | `32` | Streaming batch of the recorded FIT activity pass. |
| `ACTIVITY_CHUNK_SIZE` | `activity_chunk_size` | `memory.activity_chunk_size` | int ≥ 1 | `32` | Activity accumulation chunk. |
| `REPRESENTATION_CHUNK_SIZE` | `representation_chunk_size` | `memory.representation_chunk_size` | int ≥ 1 | `256` | Representation construction chunk. |
| `VECTOR_DTYPE` | `vector_dtype` | `precision.vector_dtype` | `float32` \| `float16` \| `bfloat16` | `float32` | Storage dtype of the neuron vectors. |
| `ACTIVITY_DTYPE` | `activity_dtype` | `precision.activity_dtype` | `float32` \| `float16` \| `bfloat16` | `float32` | Storage dtype of the activity statistics. |
| `MODEL_DTYPE` | `model_dtype` | `precision.model_dtype` | `float32` \| `float16` \| `bfloat16` | `float32` | SNN compute dtype. **Only `float32` is implemented**; anything else is rejected. |
| `MIXED_PRECISION` | `mixed_precision` | `memory.mixed_precision` | bool | `False` | Mixed-precision training. **Not implemented**; `True` is rejected. |
| `MAX_INPUT_TOKENS` | `max_input_tokens` | `memory.max_input_tokens` | int ≥ 0 | `150000000` | Dense-input token budget guard (`0` disables it). |

## 9. Checkpoint — section 9

| Variable | Panel control | Type | Default | Meaning |
|---|---|---|---|---|
| `CHECKPOINT` | `checkpoint` | path (panel-only) | `checkpoints/sweep_l2_0.pt` | Same setting as in section 1. The cell lists the available checkpoints, prints the selected one's architecture and sha256, and shows whether the current config is compatible. |

`model.n_hidden` and `model.model_type` must match the checkpoint exactly; the backend rejects a
mismatch. These two model controls are **not** surfaced as separate editable variables in the
notebook — they come from the config and are verified automatically. (They can be changed with
`panel.set_controls(state, n_hidden=..., model_type=...)`, but only to values the checkpoint
supports.)

## 10. Evaluation settings — section 10

Evaluation is a **separate operation**: it runs on PROBE against the frozen representation and never
changes the configuration that produced it. Listed choices come from the existing evaluation stack.

| Variable | Panel control | Type / allowed | Default | Meaning |
|---|---|---|---|---|
| `EVALUATION_TARGET` | `evaluation_target` | choice (panel-only): `all` \| `class_rate_20d` \| `mean_rate` \| `neuron_centered` \| `neuron_zscored` \| `raw` \| `temporal` | `all` | Which response target(s) to evaluate. `all` = the four response variants. |
| `EVALUATION_MODE` | `evaluation_mode` | choice (panel-only): `response` | `response` | Evaluation mode. Only the response-target mode is implemented. |
| `N_PERM` | `n_perm` | int ≥ 0 (panel-only) | `2000` | Number of Mantel permutations for the null distribution. |
| `BOOTSTRAP` | `bootstrap` | int ≥ 0 (panel-only) | `500` | Bootstrap resamples for the confidence interval (`0` disables it). |
| `N_SPLITS` | `n_splits` | int ≥ 1 (panel-only) | `5` | Cross-validation folds over neurons for the prediction controls. |

## 11. Build cache — section 13

| Variable | Panel control | Type | Default | Meaning |
|---|---|---|---|---|
| `USE_CACHE` | *(notebook-local flag)* | bool | `True` | Reuse verified cached artifacts instead of recomputing. Set to `False` to force a rebuild from scratch. |

The build cell reuses an artifact only when it validates fully (checkpoint, source schema + hash,
dimensions, seeds, and the whole residual training protocol); otherwise it retrains/rebuilds.

---

## Panels controls not surfaced as variables

These exist in the state layer and can be set with `panel.set_controls(state, ...)`, but the notebook
does not define a dedicated variable for them:

| Panel control | Config key | Type | Default | Meaning |
|---|---|---|---|---|
| `strict` | — (panel-only) | bool | `False` | Reject every "not implemented" request instead of warning. |
| `n_hidden` | `model.n_hidden` | int ≥ 1 | config value (`256`) | Hidden layer width — must match the checkpoint. |
| `model_type` | `model.model_type` | `recurrent_lif` | `recurrent_lif` | Only a single-layer recurrent LIF is implemented. |
| `seed` | `seed` | int ≥ 0 | `0` | Run seed: split fallback **+** evaluation permutation/CV seed. |

Derived / display-only (never set): `d`, `n_layers`, `network_context`.

---

## What affects the run id

The run id is `sha256[:16]` of the resolved **representation** values:

* component schema versions,
* the vector configuration (dimensions, enabled blocks, sources, and the **whole residual protocol**),
* the vector/activity dtypes,
* the checkpoint identifier.

It deliberately **excludes** the preset label, `out_dir`, `CONFIG_PATH`, the wall clock, and all
evaluation settings (`evaluation_target`, `evaluation_mode`, `n_perm`, `bootstrap`, `n_splits`, `seed`).
Two configurations therefore can never share an artifact by accident, and a compatible artifact is
identified purely by its content-defining values.

---

## Related files

| File | Role |
|---|---|
| `notebooks/V2_Control_Panel.ipynb` | This interactive control panel (primary interface). |
| `src/v2_panel.py` | Notebook-facing state layer (controls, defaults, validation, previews). |
| `src/v2_pipeline.py` | Stable backend API, presets, run identity, cache/artifact handling. |
| `scripts/v2_control_panel.py` | Headless CLI equivalent. |
| `src/v2_config.py` | Single configuration system (the YAML schema these controls map to). |
| `configs/v2_example.yaml` | Example configuration with the V2 keys. |
| `tests/test_v2_panel.py` | Tests for this state layer and the notebook itself (incl. a kernel-free cell execution test). |
