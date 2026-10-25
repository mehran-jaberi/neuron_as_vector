# CLAIMS_AUDIT — V3 docs vs stored artifacts

Read-only audit, 2026-10-25. Claims audited in `V3/README.md` (537 lines) and
`V3/RESULTS.md` (223 lines), with the ledger cross-checks the brief requested.
Artifacts read: `V3/results/runs.csv` (14 rows), `V3/results/<run_id>/metrics.json`,
`config.yaml`, `confusion_matrix.csv`, `V3/results/v3_vector_d1000/summary.json` +
`notebook_summary.json` + `test_metrics.json`, `archive/v2/artifacts/results/lif_baseline_selection.json`,
`V3/V3_SHD_experiment.ipynb` (cell metadata and outputs only), and metadata/spike times of
`data/shd_train.h5` / `data/shd_test.h5` (git-ignored data, read on CPU only).
Nothing was trained, no GPU work was run, no existing file was modified.

Verdicts are exactly the five tokens requested. `CONFIRMED-20EP` is also used for
dataset facts and resolved-config values that the 20-epoch run depends on (their
evidence cites the `.h5` data or `config.yaml` instead of a run row); "artifact-only"
marks claims with a stored artifact but no `runs.csv` row. `DOC-CLAIM` means no registry row and no artifact.

| claim | file:line | evidence | verdict |
|---|---|---|---|
| 1868/2264 official test samples = 82.51% (vector D=1000, 20 ep, 2 ms) | V3/RESULTS.md:39-41 | runs.csv 2026-10-02_14-31-15: epochs 20, test_correct 1868, test_total 2264, test_accuracy 0.825088; metrics.json test.accuracy 0.8250883 | CONFIRMED-20EP |
| 1119/2264 = 49.43% scalar (D=1, 20 ep, 2 ms) | V3/RESULTS.md:43-45 | artifact-only: notebook_summary.json scalar 1119/2264 = 0.494258; V3/results/v3_vector_scalar_d1/test_metrics.json correct 1119, n_samples 2264, accuracy 0.4942580; no scalar row in runs.csv | CONFIRMED-20EP |
| vector minus scalar = +33.08 percentage points | V3/RESULTS.md:47 | 82.5088 - 49.4258 = 33.083 | CONFIRMED-20EP |
| "a plain recurrent SHD baseline is usually quoted around 71%" | V3/RESULTS.md:50 | no citation in docs and no repository artifact | UNVERIFIABLE |
| V2 LIF 59.98% (N=256, T=1400 ms, 2 ms bins, 20 epochs) on the same test file | V3/RESULTS.md:51-54 | archive/v2/artifacts/results/lif_baseline_selection.json: test_accuracy_evaluated_once 0.5998; config n_hidden 256, bin_ms 2.0, n_bins 700 (=1400 ms), epochs_run 20, best_epoch 11, seed 0; checkpoint present | CONFIRMED-20EP |
| 4 ms reference (T=250, alpha=0.2), 20 epochs, 79.81% = 1807/2264 | V3/RESULTS.md:54-56 | no 4 ms 20-epoch row, summary, checkpoint or log; 1807/79.81 occur only in docs and the registry unit-test fixture V3/tests/test_registry.py:36 | DOC-CLAIM |
| 2 ms run is +2.70 pp above the 4 ms reference | V3/RESULTS.md:56-57 | arithmetic 82.51 - 79.81 = 2.70, but inherits the unverified 79.81 | DOC-CLAIM |
| 20-epoch per-epoch trajectory table (loss, CE, train acc, val loss/acc, rate, spikes, epoch time) | V3/RESULTS.md:64-85 | all 20 rows equal history_vector in notebook_summary.json (ep1 1.9840/37.26%/61.89%/1570.9/220.9; ep16 0.0208/99.90%/97.67%/1194.5/207.0; ep20 0.0135/100.00%/96.32%/1133.3/204.4) | CONFIRMED-20EP |
| total 4243.5 s for 20 vector epochs | V3/RESULTS.md:87 | notebook_summary.json train_seconds 4243.508985 | CONFIRMED-20EP |
| approx 204 s/epoch, paired with the 4243.5 s total | V3/RESULTS.md:87; V3/README.md:284 | 4243.5/20 = 212.2 s; the epoch-time column sums to 4100.6 s (205.0 s/epoch); 142.9 s unaccounted | CONTRADICTED |
| FIT accuracy 100.00%, loss 0.0134 (whole training split) | V3/RESULTS.md:88-89 | metrics.json vector.fit accuracy 1.0, loss 0.0133623 | CONFIRMED-20EP |
| best-validation epoch 16, val 0.9767 | V3/RESULTS.md:89 | history ep16 val_accuracy 0.9767; runs.csv best_val_accuracy 0.976716, best_epoch 16 | CONFIRMED-20EP |
| scalar FIT 58.01%, VAL 53.68%, 1380.6 s (approx 69 s/epoch) | V3/RESULTS.md:93-94 | artifact-only: notebook_summary.json scalar fit 0.580109, val 0.536765, train 1380.6401 s; 1380.6401/20 = 69.03 | CONFIRMED-20EP |
| test errors 396 (vector) / 1145 (scalar) | V3/RESULTS.md:107 | metrics test.errors 396; scalar 2264 - 1119 = 1145 | CONFIRMED-20EP |
| test loss 0.7728 (CE + rate reg) | V3/RESULTS.md:108 | metrics test.loss 0.7728320 | CONFIRMED-20EP |
| rate mean/median/p90/max 18.22/17.71/25.10/37.02 Hz | V3/RESULTS.md:109 | metrics test rate_mean_hz 18.2174, rate_median_hz 17.7129, rate_p90_hz 25.1020, rate_max_hz 37.0150 | CONFIRMED-20EP |
| silent neurons 0 of 64 | V3/RESULTS.md:110 | metrics test.silent_neurons 0 | CONFIRMED-20EP |
| spikes per sample 1166 | V3/RESULTS.md:111 | metrics test.spikes_per_sample 1165.91 | CONFIRMED-20EP |
| per-class recall min/median/max 38.8/88.6/100.0; classes with recall below 10%: 0 | V3/RESULTS.md:112-113 | confusion_matrix.csv (21x21, total 2264): min 38.84 (class 17), median 88.56, max 1.0 (classes 4 and 18) | CONFIRMED-20EP |
| English digits 90.31% / German digits 75.08% | V3/RESULTS.md:114; V3/README.md:279 | confusion_matrix.csv per-class recalls: classes 0-9 mean 0.9031, classes 10-19 mean 0.7508. NOTE runs.csv for the same run stores english_accuracy 0.9504 / german_accuracy 0.9291 (semantics undocumented) | CONFIRMED-20EP |
| peak CUDA memory 2284 MiB allocated / 2328 MiB reserved | V3/RESULTS.md:115,149-150; V3/README.md:282 | notebook_summary.json dtypes peak 2284.230 / 2328.0 MiB; the CLI 20-epoch run reports 2260.878 / 2308.0 MiB (different run, 0.5-1% lower) | CONFIRMED-20EP |
| test pass wall time 24.1 s for 2264 samples | V3/RESULTS.md:116 | metrics test.time_s 24.0646 | CONFIRMED-20EP |
| top confusions 13->5 49, 19->9 32, 17->3 26, 2->0 21, 17->7 19, 12->13 17; weakest class 17 (38.8%), strongest 4 and 18 (100%) | V3/RESULTS.md:121-123 | confusion_matrix.csv matches every count and every recall | CONFIRMED-20EP |
| notebook "executed top to bottom, 20 code cells, zero errors" | V3/RESULTS.md:4 | ipynb has 20 code cells, but the final cell has execution_count null and an error output "The Kernel crashed while executing code in the current cell"; its stdout shows the run was registered before the crash | CONTRADICTED |
| state and large activations fp16; master weights fp32; logits fp32 | V3/RESULTS.md:137-144; V3/README.md:181-184 | summary.json dtypes state_dtype float16, parameter_dtype float32; per-epoch peak memory recorded in history | CONFIRMED-20EP |
| approx 3.5 s per training batch of 128; approx 36 samples/s | V3/RESULTS.md:151-152; V3/README.md:193 | 4100.6 s / (20 x 58 batches) = 3.53 s/batch; 7340 samples / 204 s = 35.98 samples/s | CONFIRMED-20EP |
| 4 ms reference measured 1108 MiB allocated / 1148 MiB reserved and approx 2.2 s/batch; per-epoch time and memory "roughly double" at 2 ms | V3/RESULTS.md:153-155; V3/README.md:194-196 | no 4 ms 20-epoch artifact; nearest 4 ms artifact (5-epoch row 2026-10-02_11-24-16) reports 1267.9/1328.0 MiB and 102.4 s/epoch (approx 1.77 s/batch): 15% more memory, about 20% less time per batch than the doc | DOC-CLAIM |
| scalar baseline approx 3x faster per epoch | V3/RESULTS.md:158-160 | 1380.6/20 = 69.03 s vs 4243.5/20 = 212.18 s = 3.07x | CONFIRMED-20EP |
| un-checkpointed diagnostic: 8431 MiB allocated (8494 reserved), 333 s/batch, 150x slower, ETA 5 h 16 m | V3/RESULTS.md:167-174; V3/README.md:200-204 | no stored artifact; internally 333/2.2 = 151x | UNVERIFIABLE |
| rate-calibration diagnosis: 223 Hz population, 89% of steps spiking | V3/RESULTS.md:180-181 | no stored artifact | UNVERIFIABLE |
| 10.53 Hz against a 10 Hz target (min 9.03, max 12.93) | V3/RESULTS.md:186-187 | no stored artifact | UNVERIFIABLE |
| pipe z pipe = 0.807 mean, max 0.999 | V3/RESULTS.md:189 | no stored artifact | UNVERIFIABLE |
| test CE 28.4 vs ln 20 = 3.0 | V3/RESULTS.md:193 | no stored artifact (ln 20 = 2.996, arithmetic correct) | UNVERIFIABLE |
| "identical results through the CLI (same code path)" | V3/RESULTS.md:221 | two 20-epoch rows: 14-31-15 (notebook) and 22-31-13 (CLI), both 1868/2264; histories identical to 1e-12 on every field except peak_memory_mb (2276-2284 vs 2253-2261) and wall time (4243.5 vs 3992.9 s) | CONFIRMED-20EP |
| abstract: 82.51% = 1868 of 2264, 20 epochs, 2 ms, single seed | V3/README.md:6-13 | same artifact as the first row; runs.csv seed 0, epochs 20, time_bin_ms 2.0 | CONFIRMED-20EP |
| abstract: scalar reaches 49.43% | V3/README.md:7-8 | same artifact as the 1119/2264 row | CONFIRMED-20EP |
| 2,025,440 trainable parameters (vector) | V3/README.md:11,271,477,482; V3/RESULTS.md:25 | runs.csv parameters 2025440 on every vector row; notebook_summary parameter_groups sum exactly 2,025,440 | CONFIRMED-20EP |
| 6,335 trainable parameters (scalar) | V3/README.md:271; V3/RESULTS.md:25 | V3/results/v3_vector_scalar_d1/config.json n_parameters 6335; groups 700+1+1+1+4096+64+64+64+64+1280 = 6,335 | CONFIRMED-20EP |
| 4 ms reference 79.81% (1807/2264), "same 20 epochs" | V3/README.md:15,488 | same as the RESULTS.md 4 ms reference row (no artifact) | DOC-CLAIM |
| naive per-neuron DxD would cost N D^2 = 6.4e7 parameters "per neuron" | V3/README.md:108 | N*D^2 = 64e6 = 6.4e7 is the total, not per neuron; per neuron is D^2 = 1.0e6 | CONTRADICTED |
| data files: 8156 train samples, 2264 test samples | V3/README.md:128-129 | data/shd_train.h5 labels shape (8156,), data/shd_test.h5 (2264,) | CONFIRMED-20EP |
| train speakers {0,1,2,3,6,7,8,9,10,11}; test speakers {0..11}; 4 and 5 never in train | V3/README.md:128-130 | h5 extra/speaker uniques match exactly (train lacks 4 and 5) | CONFIRMED-20EP |
| 1000 ms window covers 96.6% of training and 99.0% of test utterance spans | V3/README.md:135 | computed: 3.37% of train samples (275/8156) and 0.97% of test samples (22/2264) end after 1000 ms, so 96.63% / 99.03% are fully covered | CONFIRMED-20EP |
| measured clip fraction 0.013% of events ("sampled subset used in the notebook") | V3/README.md:136-137 | full files: 0.0171% of train events (10,941) and 0.0040% of test events (742) are at or after 1000 ms; the sampled subset is not identified | UNVERIFIABLE |
| 7846 events/sample | V3/README.md:138 | computed mean 7846.2 (train); test 8229.4 | CONFIRMED-20EP |
| approx 2.2% of the (500,700) grid cells active | V3/README.md:138-139 | mean unique active cells 7837.4 / 350,000 = 2.239% | CONFIRMED-20EP |
| approx 4.3% of the (250,700) cells active at 4 ms | V3/README.md:139 | mean unique active cells 7680.9 / 175,000 = 4.389% (rounds to 4.4) | CONTRADICTED |
| FIT/VAL is a class-stratified 90/10 split of the training file | V3/README.md:140 | config.yaml val_fraction 0.1; 0.9 x 8156 = 7340 FIT / 816 VAL; final VAL 786/816 = 96.32% consistent | CONFIRMED-20EP |
| README section 8 results table (params, FIT/VAL, test counts, errors, per-class, English/German, memory, 24.1 s, 4243.5 s, 1380.6 s) | V3/README.md:269-284 | duplicates RESULTS sections 1-3; every row matched the artifacts listed above | CONFIRMED-20EP |
| reference took approx 72 min; 5-epoch screening approx 17 min | V3/README.md:451-452 | 4243.5 s = 70.7 min; 5 x 204 s = 17.0 min | CONFIRMED-20EP |
| measured 5-epoch 2 ms baseline is 76.15% (1724/2264) | V3/README.md:472 | runs.csv 2026-10-02_12-40-49: epochs 5, 1724/2264 = 0.761484 | SCREENING-5EP |
| D=1000 at 3 epochs (4 ms) = 71.78% | V3/README.md:489 | no 3-epoch row in runs.csv and no artifact; appears only in docs and git history | DOC-CLAIM |
| scalar D=1 at 3 epochs = 26.55% | V3/README.md:289,490 | no 3-epoch row and no artifact | DOC-CLAIM |
| N=128, D=1000 (4 ms) = 79.24% | V3/README.md:490-491 | no row and no artifact; no epoch count given | DOC-CLAIM |
| N=64, D=2000 (4 ms) = 77.96% | V3/README.md:491 | no row and no artifact; no epoch count given | DOC-CLAIM |
| doubling N or D did not improve on the reference | V3/README.md:491-492 | depends on the two unlabelled 4 ms points, compared with the 2 ms 82.51% reference rather than the 4 ms 79.81% reference | DOC-CLAIM |
| halving the time step (4 ms -> 2 ms) did (+2.70 points) | V3/README.md:492-493 | depends on the unverified 79.81% | DOC-CLAIM |
| approx 3.4% of train and approx 1.0% of test tail dropped (limitations) | V3/README.md:340-348 | computed 3.37% / 0.97% of samples have spikes after 1000 ms | CONFIRMED-20EP |
| "Executed ... top to bottom, 20 code cells, no errors" | V3/README.md:263 | same notebook artifact as the RESULTS.md zero-errors claim | CONTRADICTED |
| training configuration: 20 epochs, 2 ms/500 steps, alpha=0.1, Adam lr 1e-3 without weight decay, cosine + 3% warmup, grad clip 1.0, batch 128, 10 checkpoint chunks, beta=4, gamma=1, FIT 7340 / VAL 816, seed 0 | V3/RESULTS.md:21-32; V3/README.md:151-159 | config.yaml of run 14-31-15: epochs 20, batch_size 128, alpha 0.1, time_bin_ms 2.0, sequence_duration_ms 1000, surrogate_beta 4.0, surrogate_gamma 1.0, learning_rate 0.001, optimizer adam, lr_schedule cosine, warmup_frac 0.03, grad_clip 1.0, grad_checkpoint_chunks 10, val_fraction 0.1; runs.csv seed 0, n_bins 500 | CONFIRMED-20EP |
| 4 ms reference configuration: T=250, alpha=0.2 | V3/README.md:497-499 | config.yaml of 4 ms row 2026-10-02_11-24-16: alpha 0.2, n_bins 250, time_bin_ms 4.0 | CONFIRMED-20EP |
| 5-epoch screening, mode none: 76.15% (1724/2264) | INVESTIGATION.md:34 | runs.csv 2026-10-02_12-40-49 matches exactly | SCREENING-5EP |
| 5-epoch screening, noise std 0.001: 77.52% (1755/2264) | INVESTIGATION.md:35 | runs.csv 2026-10-02_20-44-28: 1755/2264 = 0.775177 | SCREENING-5EP |
| 5-epoch screening, noise std 0.005: 76.37% (1729/2264) | INVESTIGATION.md:36 | runs.csv 2026-10-02_21-10-07: 1729/2264 = 0.763693 | SCREENING-5EP |
| 5-epoch screening, noise std 0.01: 78.31% (1773/2264) | INVESTIGATION.md:37 | runs.csv 2026-10-02_21-36-05: 1773/2264 = 0.783127 | SCREENING-5EP |
| 5-epoch screening, noise std 0.02: 78.80% (1784/2264) | INVESTIGATION.md:38 | runs.csv 2026-10-02_22-01-45: 1784/2264 = 0.787986 | SCREENING-5EP |
| 5-epoch screening, noise std 0.05: 68.29% (1546/2264) | INVESTIGATION.md:38 | runs.csv 2026-10-02_22-27-32: 1546/2264 = 0.682862 | SCREENING-5EP |
| 5-epoch screening, 8-bit quantization: 74.82% (1694/2264) | INVESTIGATION.md:39 | runs.csv 2026-10-05_23-28-58: 1694/2264 = 0.748233 | SCREENING-5EP |
| 5-epoch screening, 4-bit quantization: 16.25% (368/2264) | INVESTIGATION.md:39 | runs.csv 2026-10-06_18-09-21: 368/2264 = 0.162544 | SCREENING-5EP |
| 4 ms 5-epoch screening: 72.26 / 74.47 / 73.94% (1636 / 1686 / 1674) | V3/TODO.md:37-39 | runs.csv 11-24-16 (1636 = 72.2615), 11-39-57 (1686 = 74.470), 11-57-40 (1674 = 73.9399) | SCREENING-5EP |
| 20-epoch noise std 0.02: 1847/2264 = 81.58%, VAL 97.18% | INVESTIGATION.md:44 | runs.csv 2026-10-03_21-19-42: epochs 20, 1847/2264 = 0.815813, val_accuracy 0.971814 | CONFIRMED-20EP |
| the 20-epoch rerun did not reproduce the 5-epoch noise gain (minus 0.93 pp) | INVESTIGATION.md:45-46 | 81.581 - 82.509 = -0.93 | CONFIRMED-20EP |
| the rerun is "bit-identical" to the first 20-epoch run (seed 0 deterministic) | INVESTIGATION.md:23 | rows 14-31-15 and 22-31-13: identical test counts and confusion matrix, histories identical to 1e-12 except peak_memory_mb (2276-2284 vs 2253-2261); weights are not retained, so weight-level bit-identity is not testable | CONFIRMED-20EP |
| commit cbfe239 ("0.02 noise => +2%") records a 5-epoch screening result | INVESTIGATION.md:76-78 | 5-epoch 78.80 vs 76.15 = +2.65 pp; the 20-epoch run was -0.93 pp | SCREENING-5EP |
| the notebook does not register its run through the registry (accepted gap) | INVESTIGATION.md:84 | ipynb cell 34 stdout: "registered run : .../V3/results/2026-10-02_14-31-15"; runs.csv row notes "notebook"; V3/README.md:444-445 states it registers via v3.registry.register_run | CONTRADICTED |
| vector 2,025,440 vs scalar 6,335 ("x320") | INVESTIGATION.md:26 | 2,025,440 / 6,335 = 319.72; 6,335 x 320 = 2,027,200, not 2,025,440 | CONTRADICTED |

Verdict counts (76 claims): CONFIRMED-20EP 41, SCREENING-5EP 11, DOC-CLAIM 10,
CONTRADICTED 7, UNVERIFIABLE 7.

## Problems found (ranked by severity)

1. **High — the 4 ms 20-epoch reference is unbacked (DOC-CLAIM).**
   `V3/RESULTS.md:54-56` and `V3/README.md:15,488` quote **79.81% (1807/2264)**
   at 20 epochs, but no 4 ms 20-epoch row exists in `V3/results/runs.csv`, no
   summary/checkpoint/log reproduces 1807, and the only non-doc occurrence of
   1807/79.81 is the unit-test fixture `V3/tests/test_registry.py:36`. The number
   is in the README abstract and generates the "+2.70 pp" claim
   (`V3/RESULTS.md:56-57`, `V3/README.md:492-493`).

2. **High — README §12 mixes epochs and provenance in the "current experimental
   reference".** `V3/README.md:489-491` lists 71.78% (3 epochs, 4 ms) and 26.55%
   (3 epochs, scalar) — labelled but screening-scale and with no registry row —
   plus 79.24% (N=128) and 77.96% (N=64, D=2000), which carry **no epoch count**
   and no row at all. The conclusion "doubling N or D did not improve on the
   reference" (`V3/README.md:491-492`) then compares those 4 ms runs with the
   2 ms 82.51% headline instead of the 4 ms 79.81% reference. Against AGENTS.md
   rule 6 ("every number carries epochs, seed, correct/total and a tag"), these
   are untagged numbers inside the reference section.

3. **Medium — the x320 parameter ratio is wrong.** `INVESTIGATION.md:26` says
   "vector 2,025,440 vs scalar 6,335 (x320)". Actual: 2,025,440 / 6,335 =
   **319.72**, and 6,335 x 320 = 2,027,200, not 2,025,440. The V3 docs never
   state x320, so the docs are unaffected; the ledger line should say "~320x" or
   "0.31% of the vector count".

4. **Medium — "zero errors" is contradicted by the committed notebook.**
   `V3/RESULTS.md:4` and `V3/README.md:263` say the notebook executed top to
   bottom with zero errors. In `V3/V3_SHD_experiment.ipynb` the last code cell
   has `execution_count: null` and an error output: "The Kernel crashed while
   executing code in the current cell or a previous cell."
   The run itself is corroborated (the cell's stdout shows "registered run :
   .../2026-10-02_14-31-15" before the crash and the registry row exists), so
   this is a provenance sentence that is false as written, not a bad number.

5. **Medium — the 4 ms resource reference conflicts with the nearest 4 ms
   artifact.** `V3/RESULTS.md:153-155` states 1108/1148 MiB allocated/reserved
   and ~2.2 s/batch at 4 ms, but the only stored 4 ms artifact
   (`V3/results/2026-10-02_11-24-16`, 5 epochs) reports **1267.9/1328.0 MiB**
   and 511.98 s / 5 epochs = 102.4 s/epoch (~1.77 s/batch): 15% more memory and
   ~20% less time per batch than the doc. The 20-epoch 4 ms run predates the
   registry, so the doc's exact values cannot be checked. Note the "roughly
   double" statement holds against the 5-epoch artifact (204/102.4 = 1.99x), not
   against 2.2 s/batch (1.6x).

6. **Medium — `runs.csv` English/German columns contradict the docs and the
   confusion matrix.** For run `2026-10-02_14-31-15`, runs.csv stores
   `english_accuracy 0.95039` and `german_accuracy 0.92909`, while the docs
   (`V3/RESULTS.md:114`, `V3/README.md:279`) and the 21x21 test confusion matrix
   give **90.31% / 75.08%**. The column semantics are undocumented and, if they
   are meant to be digit-group recalls, the columns are wrong.

7. **Medium — arithmetic wording error "per neuron".** `V3/README.md:108` says a
   naive per-neuron DxD matrix "would cost N D^2 ~ 6.4e7 parameters *per
   neuron*". N*D^2 = 64 x 10^6 is the **total**; per neuron it is D^2 = 1.0e6.
   The sentence overstates by 64x (the affordability point is unaffected).

8. **Medium — the ledger's "notebook does not register its run" gap is stale.**
   `INVESTIGATION.md:84` lists this under "accepted, do not silently fix".
   The notebook's own last-cell stdout says "registered run :
   .../2026-10-02_14-31-15", `runs.csv` contains that row with `notes: notebook`,
   and `V3/README.md:444-445` states the notebook registers through
   `v3.registry.register_run`. Either the gap means something narrower (e.g. only
   quick runs) or it is wrong; as written it contradicts the artifacts.

9. **Low — "~204 s/epoch" does not match the 4243.5 s total.**
   `V3/RESULTS.md:87` and `V3/README.md:284` pair the total with 204 s/epoch.
   4243.5/20 = 212.2 s; the epoch-time column sums to 4100.6 s (205.0 s/epoch)
   and 142.9 s is unaccounted for by the table. The "first epoch fills the event
   cache" note explains the 220.9 s first epoch, not the gap.

10. **Low — two dataset statistics are slightly off or unverifiable.**
    `V3/README.md:139` says ~4.3% of the (250,700) cells are active at 4 ms;
    counting unique active cells gives **4.389%** (rounds to 4.4). The clip
    fraction 0.013% (`V3/README.md:136-137`) cannot be checked because the
    "sampled subset" is unnamed; on the full train file it is **0.0171%** of
    events (0.0040% on test).

11. **Info — no V3 doc claims a commit message describes a 20-epoch result.**
    The only relevant commit, `cbfe239` ("noise experiment (0.02 noise=>+2%)"),
    describes a 5-epoch screening gain: 78.80% vs 76.15% = **+2.65 pp**. The
    20-epoch rerun gave 81.58% vs 82.51% = **-0.93 pp**, and
    `INVESTIGATION.md:76-78` already records that correction. The commit
    message's "+2%" is a screening-scale claim, not a 20-epoch one.

12. **Info — resource numbers are run-specific.** The notebook run and the CLI
    rerun have identical metrics but peak memory 2284 vs 2261 MiB and wall time
    4243.5 vs 3992.9 s; the docs quote only the notebook values
    (`V3/RESULTS.md:115,149-150`).

No screening (5-epoch or shorter) number is used as a headline in
`V3/README.md` or `V3/RESULTS.md`; the only screening number in the results
narrative is the explicitly labelled 5-epoch 76.15% context line
(`V3/README.md:472`). The unlabelled-epoch numbers are the two README §12
points in problem 2 (79.24 and 77.96; 79.81 at `V3/README.md:15` also lacks the
epoch count inside that sentence, although §12 line 488 labels it 20 epochs).

**Do the headline numbers survive?** Yes. 82.51% (1868/2264) and 49.43%
(1119/2264) are backed by the registry row and stored artifacts, the full
20-epoch trajectory and every test diagnostic reproduce exactly, so the
headline survives; the 4 ms reference (79.81/1807), the 3-epoch and N/D-sweep
"measured" points, and the "zero errors" sentence do not survive, and the four
numeric/labelling slips (204 s/epoch, 6.4e7 "per neuron", 4.3%, x320) need
correction without changing the headline.
