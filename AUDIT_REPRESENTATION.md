# AUDIT — Neuron representation (feature-by-feature)

Date: 2026-09-26. Scope: the **neuron representation only** (everything that turns
a hidden neuron into a label-free feature vector). Companion to `AUDIT.md`, which
covered the repository as a whole.

This audit answers three questions demanded by the brief:

1. Is every feature in the representation actually justified?
2. Is the existing cochlear "centre of mass" feature justified by the SHD channel
   ordering?
3. Are generic learned bias terms kept distinct from genuine neuron-specific
   dynamical parameters?

Legend: ✅ kept · 🔧 renamed/corrected · ❌ removed · ☑️ opt-in only · ➖ not emitted.

---

## 0. Summary of changes

| Area | Verdict |
| ---- | ------- |
| Cochlear "centre of mass" / spread | ❌ removed from the default/primary representation, ☑️ available behind `include_tonotopic_features: true` |
| Recurrent **in**/**out** orientation | 🔧 fixed to follow the model convention (row = incoming, column = outgoing) |
| Per-neuron learned **bias** | 🔧 renamed `bias` → `learned_bias`, explicitly classified as *generic learned*, **not** biophysical |
| Genuine per-neuron **dynamical** parameters | ✅ only emitted when they actually vary (`bias_tau` mode); shared constants ➖ |
| Exact duplicate activity features | ❌ dropped (`mean_spike_count`, `first_spike_latency_norm`, `burstiness`) |
| Mislabelled "ISI" features | 🔧 `isi_cv` → `spike_time_cv` (it is the CV of pooled spike *times*) |
| Standardisation + block weighting | ✅ z-score per feature; `equal` / `uniform` / **new** `custom` per-block weights |
| Permutation invariance | ✅ explicit test (`src/permutation.py`), all structural features pass |

**Final primary (structural) representation: 48 dimensions.**
`structural + activity`: **60 dimensions**. See §7 for the exact feature list.

---

## 1. Why the tonotopic centre-of-mass was removed

The old `input_conn.channel_com` / `channel_spread` features summarise
`W_in[:, j]` by its weighted mean/variance over the **channel index**, implicitly
assuming channel index increases monotonically with cochlear characteristic
frequency (a tonotopic ordering).

Evidence available **locally**:

- The README stated the ordering as an *assumption* ("we assume the natural
  cochlear/channel ordering"), not a verified fact.
- The SHD HDF5 files contain **no channel metadata at all** — verified by walking
  the full HDF5 tree of both `shd_train.h5` and `shd_test.h5`: the only groups are
  `labels`, `spikes/{times,units}` and `extra/{speaker,keys,meta_info}`. There is
  no frequency, position, or channel-order attribute anywhere.
- No local documentation file describes a channel ordering.

Because the ordering cannot be established confidently from local documentation or
code, the two ordering-dependent features are **excluded from the primary
representation** rather than resting on an unsupported assumption. They remain
available (and clearly flagged) behind the explicit opt-in
`representations.include_tonotopic_features: true`, and are listed in
`src.neurons.TONOTOPIC_FEATURE_NAMES`.

Every remaining `input_conn` feature is a symmetric function of the multiset of
input weights and is therefore **invariant to any permutation of the channels** —
a stronger and assumption-free property (test:
`test_primary_structural_features_are_channel_permutation_invariant`).

---

## 2. `intrinsic` block

Old contents: `tau_mem_ms`, `threshold`, `reset`, `bias`.

| Old feature | Verdict | Reason |
| ----------- | ------- | ------ |
| `tau_mem_ms` | ✅ / ➖ | genuine per-neuron *dynamical* parameter, **but only emitted when it actually varies** (i.e. `neuron_param_mode="bias_tau"` and non-degenerate learned offsets). In the default `bias` mode it is the shared constant 20 ms ➖ |
| `threshold` | ➖ | a shared config constant (1.0); identical for every neuron, no per-neuron information |
| `reset` | ➖ | a shared config constant derived from `cfg.reset`; not a learned parameter at all |
| `bias` | 🔧 → `learned_bias` | a **generic learned additive input** (excitability offset). Legitimately *learned* and per-neuron, but **not** a biophysical parameter |

**Key audit finding.** In the default `bias` architecture there are **no genuine
per-neuron dynamical parameters**: the threshold and reset are shared constants
and the base membrane time constant is not learned per neuron. The old
"`intrinsic`" block therefore contained exactly one informative feature — the
learned bias — and the previously reported `intrinsic_only` Mantel correlation was
in fact driven entirely by that generic learned term, not by biophysics.

**How the distinction is enforced in code.**

- Feature name `learned_bias` (never `bias`, never described as biophysical).
- `src.neurons.GENERIC_LEARNED_FEATURE_NAMES = {"intrinsic.learned_bias"}` and
  `DYNAMICAL_FEATURE_NAMES = {"intrinsic.tau_mem_ms"}`.
- `classify_feature(name)` returns `"generic_learned"` / `"dynamical"` and the
  classification is written into every space summary.
- Ablation variants `excitability_only` (learned bias only) and `dynamical_only`
  (genuine dynamics only; skipped when absent) make the difference measurable.

At initialisation (untrained network) the bias and the tau offset are exactly zero,
so the intrinsic block is **empty** and reported as such
(`intrinsic_block_present: false`). This is honest: an untrained network has no
neuron-specific intrinsic parameters to speak of.

---

## 3. `input_conn` block

Original: 9 signed statistics + 5 concentration/sparsity summaries + 2 tonotopic.

| Feature | Verdict | Reason |
| ------- | ------- | ------ |
| `mean`, `std`, `l1`, `l2`, `rms`, `max_abs` | ✅ | standard magnitude/dispersion descriptors of the input weight vector; channel-permutation invariant |
| `pos_frac`, `neg_frac`, `pos_neg_balance` | ✅ | sign composition; meaningful for signed input weights |
| `entropy`, `participation_ratio_frac`, `top5pct_share`, `top1pct_share`, `n_significant_frac` | ✅ | concentration / sparsity descriptors; `n_significant_frac` uses a documented mean+std threshold |
| `channel_com`, `channel_spread` | ❌ default / ☑️ opt-in | require an unverified channel ordering (§1) |

14 default features; 16 when tonotopic opt-in is enabled.

---

## 4. `recurrent_in` / `recurrent_out` blocks

**Orientation fix.** In `src/model.py`, `w_rec[i, j]` is the weight **from `j` to
`i`** (`forward` computes `s_prev @ w_rec.t()`). Hence:

- **row** `w_rec[j, :]` = weights **onto** neuron `j` → `recurrent_in`
- **column** `w_rec[:, j]` = weights **from** neuron `j` → `recurrent_out`

The old code had these swapped (column → `recurrent_in`, row → `recurrent_out`).
Because both blocks applied the same symmetric statistics, the pooled feature set
was unchanged and no reported distance moved — but the labels and the science
narrative were wrong. They are now correct, and `recurrent_in`/`recurrent_out`
genuinely separate incoming from outgoing structure.

| Feature | Verdict | Reason |
| ------- | ------- | ------ |
| standard stats + concentration stats on the incoming row | ✅ | magnitude/sparsity of the weights onto the neuron |
| standard stats + concentration stats on the outgoing column | ✅ | magnitude/sparsity of the weights from the neuron |
| `self_connection` = `w_rec[j, j]` | ✅ | direct autapse; constant only if self-connections are structurally disabled (then flagged non-informative) |
| `in_out_correlation` | ✅ | reciprocity: does the neuron talk to the partners it listens to? (Pearson corr of incoming vs outgoing vectors) |
| `in_out_cosine` | ✅ | scale-free reciprocity |
| `in_out_asymmetry` | ✅ | signed relative difference of incoming vs outgoing L2 norms (sign now correct) |
| `reciprocal_strength` | ✅ | mean product of |incoming|·|outgoing| over shared partners |

`recurrent_in`: 14 + 5 = **19** features; `recurrent_out`: **14** features.

---

## 5. `activity` block (label-free)

Original: 15 features stored + `first_spike_latency_ms` + `first_spike_latency_norm`.

| Old feature | Verdict | Reason |
| ----------- | ------- | ------ |
| `rate_hz` | ✅ | mean firing rate |
| `log_rate_hz` | ✅ | log-rate parameterisation; retained because the `rate_only` control uses it (note: monotone in `rate_hz`, so the two are correlated) |
| `rate_std_hz` | ✅ | across-sample rate variability |
| `silent_fraction` | ✅ | fraction of samples with no spike |
| `fano_factor` | ✅ | count variance / mean |
| `mean_spike_count` | ❌ | **exact duplicate** of `rate_hz` (count = rate × fixed duration); identical after z-scoring |
| `isi_mean_ms` | ❌ | not an ISI; it is `bin_ms / active_bin_fraction`, a deterministic function of `active_bin_fraction` |
| `isi_cv` | 🔧 → `spike_time_cv` | computed from the pooled spike-**time** distribution ⇒ it is the CV of spike times, not of interspike intervals |
| `burstiness` | ❌ | deterministic monotone transform of the (renamed) `spike_time_cv`; redundant |
| `temporal_center_ms` | ✅ | mean spike time of the pooled PSTH |
| `temporal_dispersion_ms` | ✅ | std of spike times |
| `temporal_entropy` | ✅ | normalised entropy of the PSTH |
| `peak_rate_hz` | ✅ | peak bin rate |
| `active_bin_fraction` | ✅ | temporal coverage |
| `first_spike_latency_ms` | ✅ | mean per-sample first-spike latency (censored at stimulus duration) |
| `first_spike_latency_norm` | ❌ | **exact duplicate** of `first_spike_latency_ms` (fixed divisor) |

Final `activity` block: **12** features (11 + `first_spike_latency_ms`).

---

## 6. Standardisation, weighting and permutation invariance

- **Standardisation.** Per-feature z-scoring; zero-variance columns are mapped to a
  constant 0 and reported as non-informative (never NaN).
- **Weighting.** `weighting: equal` (each block contributes equally),
  `uniform` (each feature equally) or **`custom`** with explicit
  `block_weights` (a block contributes its weight to the squared distance).
- **Permutation invariance** (`src/permutation.py`). The network's hidden neurons
  are randomly relabelled and *every* hidden-indexed parameter is permuted
  consistently (`w_in` columns, `w_rec` rows and columns, `w_out` rows, per-neuron
  parameters). The permuted network is checked to be functionally identical, then
  the representations are recomputed and compared neuron-by-neuron. Any feature
  whose value changes is reported as permutation-sensitive. A
  `neuron_index_representation` control demonstrates that the detector fires on a
  deliberately index-based representation.

The structural representation **passes** the test (48/48 features invariant),
including when the opt-in tonotopic features are enabled (permuting hidden neurons
does not permute input channels).

---

## 7. Final representation — exact contents and dimensionality

Primary (`representations.primary_blocks`, default) = `intrinsic + input_conn +
recurrent_in + recurrent_out`, measured on the trained baseline checkpoint
(`checkpoints/baseline.pt`, 256 hidden neurons, `neuron_param_mode: bias`).

**Primary / structural dimensionality = 48.**

| Block | n | Features |
| ----- | - | -------- |
| `intrinsic` | 1 | `learned_bias` *(generic learned — not biophysical)* |
| `input_conn` | 14 | `mean`, `std`, `l1`, `l2`, `rms`, `max_abs`, `pos_frac`, `neg_frac`, `pos_neg_balance`, `entropy`, `participation_ratio_frac`, `top5pct_share`, `top1pct_share`, `n_significant_frac` |
| `recurrent_in` | 19 | the 14 as above **plus** `self_connection`, `in_out_correlation`, `in_out_cosine`, `in_out_asymmetry`, `reciprocal_strength` |
| `recurrent_out` | 14 | the 14 as above |

**Optional activity block = 12:** `rate_hz`, `log_rate_hz`, `rate_std_hz`,
`silent_fraction`, `fano_factor`, `spike_time_cv`, `temporal_center_ms`,
`temporal_dispersion_ms`, `temporal_entropy`, `peak_rate_hz`,
`active_bin_fraction`, `first_spike_latency_ms`.

**`structural + activity` (full) = 60.**

Feature kinds: exactly one `generic_learned` feature (`intrinsic.learned_bias`);
zero `dynamical` features in the default architecture; zero `tonotopic` features
by default.

---

## 8. Ablation variants (all through the identical geometry pipeline)

Required by the brief and present in `src.controls.default_variants`:

| Variant | Blocks | n (baseline) |
| ------- | ------ | ------------ |
| `intrinsic_only` | intrinsic | 1 (= the learned bias) |
| `excitability_only` | intrinsic, filtered to `learned_bias` | 1 |
| `dynamical_only` | intrinsic, genuine dynamical params | **skipped** (none in `bias` mode) |
| `input_conn_only` | input_conn | 14 |
| `recurrent_only` | recurrent_in + recurrent_out | 33 |
| `activity_only` | activity | 12 |
| `structural_full` | intrinsic + input_conn + recurrent_in + recurrent_out | 48 |
| `structural_plus_activity` | structural + activity | 60 |

Plus the existing controls (`random_null`, `rate_only`, `shuffled_control`,
leave-one-block-out, circular `fingerprint_descriptive`) and the partial Mantel
controlling for firing rate.

Empty-selection variants (e.g. `dynamical_only` for a `bias`-mode model) are
recorded with `skipped: true` and a reason, rather than silently dropped or
crashing.

---

## 9. Companion — functional fingerprint and primary analysis

The representation is only meaningful against an **independent** measurement of
what a neuron does. That target is built in `src/functional_fingerprint.py` from
class-conditioned, held-out responses and is **never** an input to the
representation.

### 9.1 Fingerprint families

| Family | Feature set(s) | Notes |
| ------ | -------------- | ----- |
| A. class tuning | `class_rate` | 20-dim class response profile (the primary target) |
| B. rate-normalized tuning | `class_rate_norm` | class profile ÷ per-neuron class-mean rate, so overall magnitude cannot dominate similarity |
| C. temporal | `class_psth`, `class_temporal_center`, `class_temporal_dispersion`, `class_latency` | coarse PSTH + temporal centre/dispersion + first-spike timing |

**Fixed from the earlier audit.** The default fingerprint feature set no longer
includes `class_count`, which audit item C3 showed to be an exact duplicate of
`class_rate` after column z-scoring. `min_spikes_for_latency` is now actually
applied (audit item B3), and the fingerprint is measured on the held-out
**analysis-probe** split rather than the model-selection split (audit item C2).

### 9.2 Primary analysis and controls

`src/function_analysis.py` (run by `scripts/run_function_analysis.py`) implements:

* the primary Spearman **Mantel** test on condensed distance vectors, with a
  neuron-relabelling permutation null (pairs are **not** treated as independent),
  a bootstrapped CI, the full null distribution, and the explicit
  `p_value_floor = 1/(n_perm+1)` with an `at_resolution_floor` flag;
* controls: `rate_only`, `rate_normalized_fingerprint` (**primary rate control**),
  `rate_matched` stratified Mantel (**proper rate-matched control**),
  `random_representation`, `shuffled_neurons`, and kNN for `k = 3, 5, 10, 20`;
* cross-validated **prediction** (`src/prediction.py`): ridge and k-NN regression
  from representation to fingerprint, `KFold` across neurons, with standardisation
  and `alpha` selection fit inside each training fold, reporting Pearson *r*,
  *R²*, RMSE/nRMSE and MAE.

> **Partial Mantel is secondary and exploratory.** It is explicitly labelled
> `secondary_exploratory__not_a_proof_of_rate_independence`. Independence from
> firing rate is argued from the rate-normalized fingerprint and the rate-matched
> stratified Mantel, **not** from the partial Mantel.

The single machine-readable summary is
`results/<tag>_function_analysis.json` (with null distributions and distance
vectors in the companion `.npz`).
