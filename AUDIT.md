# AUDIT — "Neuron as Vector" repository

Date: 2026-09-26. Scope: audit only — no code changes made.
Method: full read of `README.md`, every module in `src/`, every script in
`scripts/`, both configs, all tests, and cross-checking of every README number
against the files in `results/`, `checkpoints/`, `figures/`; plus direct
inspection of the raw SHD HDF5 files and numerical re-verification of the
binning path. Test suite re-run: **28/28 passed** (pytest 9.1.1, ~1 s).

Legend: ✅ verified correct · ⚠️ caveat · ❌ bug or wrong claim.

---

## A. What is correct

### A1. Project structure and tooling
- ✅ File layout matches README §10 exactly (`src/`, `scripts/`, `tests/`,
  `configs/`); all 6 figures + forest plot exist under `figures/`.
- ✅ `uv` setup is sound: `uv.lock`, `.python-version` (3.11) present;
  `pyproject.toml` pins torch to the `cu126` index via an explicit
  `[[tool.uv.index]]` + `[tool.uv.sources]`; `package = false` is deliberate;
  pytest configured via `[tool.pytest.ini_options]` with `pythonpath=["."]`.
- ✅ Installed environment matches provenance files
  (`torch 2.14.0+cu126`, RTX 3060 Laptop, recorded in
  `results/baseline_train_summary.json`).

### A2. SHD loading (`src/data.py`)
- ✅ `parse_shd_h5` auto-detects the canonical ragged layout
  (`spikes/times` + `spikes/units` + `extra/speaker`); verified against the real
  files: 8156 train / 2264 test samples, 20 classes, speaker ids under
  `extra/speaker`.
- ✅ SHD time units handled correctly: raw times are **seconds**, converted to
  ms via `times * 1000.0` (verified: raw max 1.3691 s → 1369.1 ms).
- ✅ Per-sample sorting, defensive sample-boundary recovery, and explicit errors
  when boundaries cannot be recovered.
- ✅ Units 0–699 (max unit index 699 verified), `int16` storage.

### A3. Splits and leakage boundary
- ✅ Official test file is **never** used for training, model selection,
  representation building, or fingerprint construction (checked every script).
- ✅ Train/val split is speaker-aware, disjoint and complete
  (6920 + 1236 = 8156; held-out speakers {6, 8}; no class repair was triggered,
  `classes_repaired: {}`).
- ✅ `data.split_seed` correctly decouples the split from the model seed; the
  multi-seed runs genuinely share one split (identical `split_info` in all three
  `*_geometry_summary.json` files).
- ✅ Representation is label-free: `collect_activity(..., with_labels=False)`
  never passes labels to the accumulator; structural features use parameters
  only; `uses_labels`/`uses_data` metadata flags travel with every object.
- ✅ Neuron IDs never become features (`NeuronRepresentation.to_vector` uses
  only feature blocks; alignment between representation and fingerprint is by
  row index = neuron index in both).

### A4. Binning (`src/data.py::events_to_bins`, `batch_events_to_bins`)
- ✅ `floor(t / bin_ms)` scatter with out-of-window events dropped (documented).
- ✅ The vectorised batched path is **bit-identical** to the per-sample
  reference: re-verified numerically on synthetic data and on the first 25 real
  SHD samples.

### A5. Model (`src/model.py`)
- ✅ Weight orientations are internally consistent:
  `w_in` (C, H) with `x @ w_in`; `w_rec[i, j]` = weight **j → i** used as
  `s_prev @ w_rec.t()`; `w_out` (H, O) with `s @ w_out`.
- ✅ Dynamics match the docstring: current-based implicit-Euler LIF with
  `α = exp(−Δt/τ_mem)`, `β = exp(−Δt/τ_syn)`; subtractive and hard reset both
  implemented; readout = time-averaged leaky integrator.
- ✅ Surrogate gradient: forward is exactly Heaviside, backward is the
  fast-sigmoid `γ/(1+β|u|)²` (unit-tested, incl. derivative = γ at u = 0).
- ✅ τ-clamp in `alpha_tensor` prevents α ≥ 1; seeded `reset_parameters`
  generator makes init reproducible.

### A6. Training (`src/training.py`, `scripts/train.py`)
- ✅ CE loss on averaged readout logits; grad clip; cosine schedule; per-epoch
  seeded shuffling; best-validation checkpoint restored before saving; test set
  evaluated for reporting only.
- ✅ Circuit-health diagnostics (rate, silent fraction) logged every epoch and
  saved.

### A7. Checkpoint handling
- ✅ `RecurrentLIFSNN.save/load` round-trips config + state; the analysis
  scripts rebuild the architecture from the checkpoint's own config.

### A8. Activity extraction and representations
- ✅ Streaming `ActivityAccumulator` keeps memory bounded; label-free and
  class-conditioned statistics are separate fields.
- ✅ First-spike latency censoring is explicit and finite (no NaNs);
  `sanitize_features` replaces non-finite values; `Standardizer` maps constant
  columns to 0 and reports them as non-informative.
- ✅ Deterministic feature ordering (sorted names), block-equal weighting,
  z-scoring.

### A9. Geometry/statistics core
- ✅ Mantel relabelling null `D[p][:, p]` implemented correctly with a `(1+x)/(1+n)`
  one-sided p-value; kNN null and partial Mantel (rank-residualised) present.
- ✅ Controls are run through the identical pipeline; the circular
  `fingerprint_descriptive` variant is flagged `circular__do_not_report_as_evidence`
  and behaves as expected (r = 1.000).
- ✅ Before/after comparison rebuilds the *exact* untrained initialisation
  (same seed), so it is a fair same-architecture comparison.

### A10. Reported numbers reproduce from `results/`
- ✅ README §7.1: accuracies 0.636 / 0.529 / 0.614, mean rate 173.3 Hz,
  range 36.9–356.4 Hz, 0 silent neurons, 250 132 parameters — all match
  `baseline_train_summary.json`.
- ✅ README §7.2–§7.5: Mantel r = 0.349, z = 10.75, null −0.001/0.033,
  32 640 pairs; reliability 0.979/0.989; full ablation table
  (`baseline_ablation.csv`) and shuffle/random controls all match.
- ✅ README §7.6 multi-seed table matches `multiseed_summary.json`
  (r = 0.298 ± 0.051; partial 0.242 ± 0.061; test acc 0.636 ± 0.030).
- ✅ Tests: 28/28 pass, fully synthetic (no downloads).

---

## B. Bugs or likely bugs

### B1. ❌ Permutation count mismatch / p-values are the resolution floor (IMPORTANT)
- README §7.2 claims **"Permutation p (10 000 perms) = 0.000999"**, but
  `configs/analysis.yaml` sets `geometry.n_perm: 1000`, and every result file
  (`results/baseline_geometry_summary.json`, all `*_ablation.csv`,
  `results/multiseed_summary.json`) records `n_perm: 1000` and
  `p_value = 0.000999000999…` — which is **exactly 1/(1000+1)**, the minimum
  attainable p-value in `src/geometry_analysis.py::mantel_test`
  (`p = (1 + #{null ≥ obs}) / (1 + n_perm)`).
- So: (i) the README's "10 000 perms" is factually wrong, and (ii) every
  "significant" p-value in the paper is merely the resolution limit of the
  permutation count, not a measured probability (the z-scores of 8–11 imply the
  true p is far smaller). This also applies to all partial-Mantel, kNN and
  per-seed p-values.

### B2. ❌ Recurrent "in"/"out" blocks are swapped relative to the model convention
- `src/model.py` defines `w_rec[i, j]` = weight **from j to i**
  (verified against `forward`: `recurrent_input = s_prev @ w_rec.t()`), so
  **row j = incoming** weights onto neuron j and **column j = outgoing**.
- `src/neurons.py::extract_structural_representations` computes
  `recurrent_in` from `w_rec[:, j]` (a **column**, i.e. outgoing) and
  `recurrent_out` from `w_rec[j, :]` (a **row**, i.e. incoming). The docstrings
  of `recurrent_incoming_features`/`recurrent_outgoing_features` and README §4
  repeat the same wrong claim ("in-degree statistics of columns of W_rec").
- Impact: the two blocks compute identical symmetric statistics, so the swap is
  an isometry (feature permutation) and **does not change any reported distance
  or Mantel value** (the only asymmetric feature, `in_out_asymmetry`, is
  sign-flipped globally, also an isometry). But the labels/interpretation are
  wrong, and any future analysis separating in from out would be mislabelled.

### B3. ❌ `min_spikes_for_latency` is a dead parameter
- Declared in `configs/analysis.yaml`, `FingerprintConfig` and plumbed through
  `scripts/_pipeline.py::build_fingerprint`, but never applied: latency
  censoring in `src/functional_fingerprint.py::class_conditioned_fingerprint`
  always uses `class_first_spike_count > 0` regardless of the configured
  minimum.

### B4. ❌ False docstring claim about the SHD time window (silent truncation)
- `src/data.py` docstring: "SHD events end well before the default 1000 ms
  window" — **false**. Verified on the real files: max event time is
  **1369 ms** (train) and **1170 ms** (test). With the default
  `n_bins=500 × bin_ms=2.0` = 1000 ms window, later events are silently
  dropped. Measured impact is small (≈0.02 % of train events, ≈3.4 % of train
  samples affected; 0.00 %/0.97 % for test), but the claim is wrong and the
  truncation is undocumented to the user.

### B5. ⚠️ Dead config knobs in the geometry stage
- `geometry.representation_metric` / `geometry.fingerprint_metric` in
  `configs/analysis.yaml` are never passed to
  `src/geometry_analysis.py::geometry_function_analysis` by
  `scripts/run_geometry_analysis.py` / `src/controls.py::run_variant_suite`
  (the Euclidean defaults happen to match the config, so results are unchanged).

### B6. ⚠️ Minor dead/misleading code
- `src/training.py::train_model` records `epoch_time_s: None` (never timed).
- `src/data.py::N_TRAIN_SAMPLES` / `N_TEST_SAMPLES` defined but never used.
- `src/functional_fingerprint.py::attenuation_correct` defined but never called
  anywhere.
- `src/model.py::RecurrentLIFSNN.forward` contains `if T != cfg.n_bins: pass` —
  a silent no-op that can mask a binning/config mismatch (the accumulator would
  raise later, but the model itself does not).
- `src/model.py::RecurrentLIFSNN.load` uses `torch.load(..., weights_only=False)`
  (acceptable for own checkpoints; note for supply-chain hygiene).

### B7. ⚠️ Reported firing rates are batch-mean-of-means
- `src/evaluation.py::evaluate_model` and `src/training.py::train_model`
  accumulate `spike_count.mean(dim=0)` per batch and average over batches, so
  the last (smaller) batch is weighted equally — the reported hidden rates are
  very slightly biased (accuracies are computed correctly per sample).

---

## C. Scientific / statistical problems

### C1. ❌ All significant p-values saturate the permutation floor (IMPORTANT)
Every effect (primary, partial, per-block, kNN, all seeds) reports exactly
p = 1/1001. The study cannot distinguish p = 1e-4 from p = 1e-10, and README
§7.6's "p = 0.000999 in every seed" is an artefact of `n_perm = 1000`, not
evidence of identical significance across seeds. Report as `p < 1/(n_perm+1)`
or increase `n_perm` (permutations here are cheap: pure NumPy on 32 640-dim
vectors).

### C2. ❌ The evaluation target is measured on the model-selection split
The fingerprint is built on **val** (`fingerprint.eval_split: val`), but val is
also what selected the checkpoint (best-val-epoch, `select_by: val_accuracy` in
`src/training.py::train_model`). The fingerprint is therefore independent of the
*representation* but **not** independent of *model selection* — a mild
adaptive-overfitting loop that the README's "held-out" language does not
acknowledge. Cleanest fix: measure the fingerprint on the official test set
(only as an evaluation target, which the code already supports with a warning)
or on a third split.

### C3. ❌ `class_count` fingerprint features duplicate `class_rate`
In `src/functional_fingerprint.py::class_conditioned_fingerprint`,
`counts = class_counts / n_per_class` and `rate = counts / duration_s` are
exact positive scalar multiples per class; after `standardize: column`
(per-column z-score across neurons) `z(count_c) ≡ z(rate_c)` **exactly**. So
40 of the 60 fingerprint dimensions are redundant copies, and Euclidean
fingerprint distance double-weights rate information relative to latency.
README §5 presents the three feature sets as distinct content. This also
inflates the split-half "noise ceiling".

### C4. ❌ README overstates the test set's speaker independence
README §2: "the two test speakers are **test-only**" and "the reported test
accuracy is on genuinely held-out speakers". Verified against
`data/shd_test.h5`: the test set is dominated by the two novel speakers
(4: 900, 5: 940 samples) **but also contains 424/2264 ≈ 19 % of samples from
the 10 speakers present in the training file** (23–52 samples each). Test
accuracy 0.614 is therefore a mixture; the claim is an overstatement.

### C5. ❌ ISI-based activity features are mislabelled
`src/neurons.py::activity_features_from_psth` computes `isi_mean_ms`, `isi_cv`
and `burstiness` from the **pooled per-neuron PSTH**, i.e. from the temporal
distribution of spike times across the whole recording window — not from
interspike intervals. `isi_cv` is actually the CV of spike *times* (temporal
dispersion / mean spike time), a different quantity from the ISI CV (a neuron
firing tonically all trial has large "isi_cv" here but a small true ISI CV).
The README calls this an "approximation"; it is in fact a different measure.
These features sit in the activity block, which drives some of the strongest
reported correlations (`activity_only` r = 0.615).

### C6. ⚠️ No multiple-comparison correction
18 variants × 3 seeds of permutation tests are reported with no correction and
no note that all p-values are at the floor (C1). The pre-registration of one
primary metric mitigates this, but the ablation table is implicitly read as a
set of significant findings.

### C7. ⚠️ Partial-Mantel null keeps the nuisance fixed
`src/geometry_analysis.py::partial_mantel_test` permutes only `dy` while the
nuisance `dz` (firing-rate distances) stays aligned with `dx`. Standard partial
Mantel implementations permute Y and Z jointly. The chosen null ("fingerprint
labelling is arbitrary, rate structure fixed") is defensible but should be
stated; also the nuisance rate is measured on train while the fingerprint is on
val, so the control variable is not exactly the rate that generated the target.

### C8. ⚠️ Fixed 1 s window vs variable utterance durations
Rates/counts/latencies are normalised by the fixed 1000 ms window, while real
SHD utterances span ≈0.3–1.37 s (verified). Class-conditional duration
differences therefore leak into both the label-free activity block and the
label-using fingerprint; neither is duration-normalised.

### C9. ⚠️ Very high hidden firing rates
Mean hidden rate ≈ 173 Hz per neuron (≈44 000 spikes/sample across 256
neurons) with no homeostatic penalty (`l2_spikes = 0`). This is correct as
implemented and honestly reported, but it is far from biological plausibility
and from typical SHD SNN operating points; worth a caveat for a SNUFA audience.

### C10. ⚠️ Minor README factual error on the 20 classes
README §2: "20 output classes (spoken digits 0–9, two speakers each)". The
actual class keys in the data (`extra/keys`) are the digits 0–9 spoken in **two
languages** (English `zero…nine` + German `null…neun`), not two speakers.

### C11. ⚠️ Val accuracy below test accuracy
val 0.529 < test 0.614 (all seeds). Unusual ordering; consistent with C4
(~19 % of test samples come from speakers seen in training) and/or val speakers
{6, 8} being harder. Not a bug, but should be discussed rather than left
unexplained.

---

## D. Reproducibility problems

### D1. ⚠️ Key artefacts are not version-controlled
`.gitignore` excludes `results/`, `checkpoints/`, `figures/` and `data/`, so
none of the numbers cited by the README are in the repository. Reproducing them
requires downloading SHD and re-training 3 seeds (~10 min/seed on an RTX 3060).
For a paper submission, commit at least the small JSON/CSV summaries or deposit
them.

### D2. ❌ README/config/results disagree on the permutation count (see B1)
A reader who reruns `run_geometry_analysis.py` with the committed config gets
`n_perm=1000`, while README §7.2 documents 10 000.

### D3. ⚠️ `run_multiseed.py` silently reuses stale outputs
`multiseed.skip_existing: true` skips any stage whose sentinel file exists,
with no config/content hash check. Changing the config and re-running will
silently mix old and new artefacts. Output files carry no fingerprint of the
config that produced them.

### D4. ⚠️ GPU runs are not bitwise reproducible
`src/utils.py::set_seed` sets cuDNN deterministic flags but not
`torch.use_deterministic_algorithms(True)`; the README documents this. The
checkpointed baseline numbers therefore cannot be reproduced bit-exactly on
CUDA (qualitative results should be stable).

### D5. ⚠️ No integrity verification of downloaded data
`src/data.py::download_shd` downloads and extracts SHD without any
checksum/size verification; a truncated or corrupted download would be parsed
without complaint as long as the HDF5 structure is intact.

### D6. ⚠️ `epoch_time_s` and other dead fields (B6) reduce log usefulness
Minor, but the history files carry permanently-null columns.

---

## E. Recommended fixes (ordered by importance)

1. **Fix the permutation-count story (B1, C1, D2).** Either raise
   `geometry.n_perm` (e.g. 10 000–100 000 — cheap for this problem size) and
   regenerate the geometry/multiseed results, or report all p-values as
   `p < 1/(n_perm+1)`. Correct README §7.2 ("10 000 perms") to match whatever
   is actually run. Files: `configs/analysis.yaml`,
   `src/geometry_analysis.py::mantel_test`/`partial_mantel_test`,
   `README.md`, rerun of `scripts/run_geometry_analysis.py` +
   `scripts/run_multiseed.py`.
2. **Make the fingerprint independent of model selection (C2).** Measure it on
   the official test set (as evaluation target only — the code path already
   exists and warns) or on a third carved-out split; update README §1/§5
   language accordingly. Files: `configs/analysis.yaml`,
   `scripts/extract_representations.py`, `scripts/run_geometry_analysis.py`.
3. **Fix the fingerprint redundancy (C3).** Drop `class_count` (it is a
   deterministic copy of `class_rate` after column z-scoring) or make it
   genuinely different (e.g. variance/count-distribution stats); re-run the
   ablation and re-verify the headline numbers. Files:
   `src/functional_fingerprint.py::class_conditioned_fingerprint`, README §5.
4. **Fix the recurrent in/out labelling (B2).** Swap column/row usage in
   `src/neurons.py::extract_structural_representations` (or re-document the
   convention end-to-end) and correct README §4. No numeric change expected,
   but the science narrative must match the dynamics.
5. **Correct README claims about the test set and classes (C4, C10).** State
   that ~19 % of official test samples come from speakers also present in
   training; optionally report test accuracy restricted to the two novel
   speakers; fix the "two speakers each" class description. Files:
   `README.md` §2/§7.
6. **Fix the time-window story (B4).** Extend the window to cover real SHD
   utterances (e.g. `n_bins=700` @ 2 ms = 1400 ms, or per-sample durations) or
   document the measured 0.02 % truncation; correct the false docstring in
   `src/data.py`.
7. **Implement or remove `min_spikes_for_latency` (B3)** in
   `src/functional_fingerprint.py::class_conditioned_fingerprint`.
8. **Rename/recompute the ISI features (C5)** in
   `src/neurons.py::activity_features_from_psth` (either compute true ISIs from
   spike times or rename to `temporal_*`), and update README §4.
9. **Wire the dead config knobs or delete them (B5, B6):** pass
   `geometry.representation_metric`/`fingerprint_metric` through
   `run_variant_suite`, raise on `T != cfg.n_bins` in
   `src/model.py::RecurrentLIFSNN.forward`, fill or drop `epoch_time_s`, remove
   unused constants / `attenuation_correct`.
10. **Reproducibility hardening (D1, D3, D5):** commit small result summaries
    (or deposit artefacts), record a config hash in every output and make
    `skip_existing` hash-aware, add checksum verification to `download_shd`.
11. **Statistics hygiene (C6, C7):** add a multiple-comparisons note to the
    ablation table, state the partial-Mantel null convention, and consider
    duration-normalised rates (C8) as a robustness check.

---

## Appendix: verification notes

- Batched binning (`batch_events_to_bins`) re-verified bit-identical to the
  per-sample reference on synthetic and real SHD samples.
- Raw SHD facts measured directly: train 8156 samples / 63 993 588 events /
  max time 1.3691 s / speakers {0,1,2,3,6,7,8,9,10,11}; test 2264 samples /
  18 631 462 events / max time 1.1699 s / speakers 4–5 dominant (900/940) plus
  23–52 samples from each training speaker.
- Val split = speakers {6, 8} (447 + 789 = 1236 samples), no class repair.
- All README table entries in §7.1–§7.6 were matched cell-by-cell against
  `results/baseline_train_summary.json`, `results/baseline_geometry_summary.json`,
  `results/baseline_ablation.csv`, `results/baseline_before_after.csv` and
  `results/multiseed_summary.json` — all agree, with the exceptions listed in
  B1/C4/C10.
