# Repository archive manifest

**Operation type: archival only.** No file was deleted. Every archived artifact was *moved*
and its contents are preserved exactly. No git history was rewritten (no commit was
amended, squashed, reset or rebased).

* Date of archival: 2026-09-27
* Repository revision at archival time: `b29aa18` ("audit_v2_3")
* Archive root: `archive/`
* Method: for every candidate, references were searched with `git grep` across the tracked
  tree (`README.md`, `*.md`, `configs/`, `scripts/`, `src/`, `tests/`) and with a scan of the
  generated-artifact directories. A candidate was moved **only** if it was
  (a) untracked/generated, (b) referenced by nothing in the tracked tree, and (c) an
  unmistakable intermediate/smoke/debug run output. Ambiguous artifacts were left in place
  (see "Deliberately not archived").

## Archived artifacts (51)

All rows: `tracked = no` (these files were gitignored generated outputs, as `results/` and
`checkpoints/` are in `.gitignore`), `references checked = yes`.

| original path | archive path | reason | tracked | refs checked | date |
|---|---|---|---|---|---|
| `checkpoints/debug_v2.pt` | `archive/intermediate_runs/checkpoints/debug_v2.pt` | intermediate run output (tiny/smoke/debug checkpoint); superseded, no tracked reference | no | yes | 2026-09-27 |
| `checkpoints/readme_check.pt` | `archive/intermediate_runs/checkpoints/readme_check.pt` | intermediate run output (tiny/smoke/debug checkpoint); superseded, no tracked reference | no | yes | 2026-09-27 |
| `checkpoints/shd_tiny.pt` | `archive/intermediate_runs/checkpoints/shd_tiny.pt` | intermediate run output (tiny/smoke/debug checkpoint); superseded, no tracked reference | no | yes | 2026-09-27 |
| `checkpoints/synthetic_smoke.pt` | `archive/intermediate_runs/checkpoints/synthetic_smoke.pt` | intermediate run output (tiny/smoke/debug checkpoint); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/debug_v2_history.csv` | `archive/intermediate_runs/results/debug_v2_history.csv` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/debug_v2_history.json` | `archive/intermediate_runs/results/debug_v2_history.json` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/debug_v2_train_summary.json` | `archive/intermediate_runs/results/debug_v2_train_summary.json` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/nsb_run.log` | `archive/intermediate_runs/results/nsb_run.log` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/nsb_smoke_history.csv` | `archive/intermediate_runs/results/nsb_smoke_history.csv` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/nsb_smoke_history.json` | `archive/intermediate_runs/results/nsb_smoke_history.json` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/readme_check_history.csv` | `archive/intermediate_runs/results/readme_check_history.csv` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/readme_check_history.json` | `archive/intermediate_runs/results/readme_check_history.json` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/readme_check_train_summary.json` | `archive/intermediate_runs/results/readme_check_train_summary.json` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/shd_tiny_ablation.csv` | `archive/intermediate_runs/results/shd_tiny_ablation.csv` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/shd_tiny_ablation.json` | `archive/intermediate_runs/results/shd_tiny_ablation.json` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/shd_tiny_before_after.csv` | `archive/intermediate_runs/results/shd_tiny_before_after.csv` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/shd_tiny_before_after.json` | `archive/intermediate_runs/results/shd_tiny_before_after.json` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/shd_tiny_figure_bundle.json` | `archive/intermediate_runs/results/shd_tiny_figure_bundle.json` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/shd_tiny_figure_data.npz` | `archive/intermediate_runs/results/shd_tiny_figure_data.npz` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/shd_tiny_fingerprint.json` | `archive/intermediate_runs/results/shd_tiny_fingerprint.json` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/shd_tiny_fingerprint.npz` | `archive/intermediate_runs/results/shd_tiny_fingerprint.npz` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/shd_tiny_geometry_summary.json` | `archive/intermediate_runs/results/shd_tiny_geometry_summary.json` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/shd_tiny_history.csv` | `archive/intermediate_runs/results/shd_tiny_history.csv` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/shd_tiny_history.json` | `archive/intermediate_runs/results/shd_tiny_history.json` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/shd_tiny_neuron_activity_representations.json` | `archive/intermediate_runs/results/shd_tiny_neuron_activity_representations.json` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/shd_tiny_neuron_representations.json` | `archive/intermediate_runs/results/shd_tiny_neuron_representations.json` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/shd_tiny_representations_summary.json` | `archive/intermediate_runs/results/shd_tiny_representations_summary.json` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/shd_tiny_space_activity.npz` | `archive/intermediate_runs/results/shd_tiny_space_activity.npz` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/shd_tiny_space_full.npz` | `archive/intermediate_runs/results/shd_tiny_space_full.npz` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/shd_tiny_space_primary.npz` | `archive/intermediate_runs/results/shd_tiny_space_primary.npz` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/shd_tiny_space_structural.npz` | `archive/intermediate_runs/results/shd_tiny_space_structural.npz` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/shd_tiny_train_summary.json` | `archive/intermediate_runs/results/shd_tiny_train_summary.json` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/synthetic_smoke_ablation.csv` | `archive/intermediate_runs/results/synthetic_smoke_ablation.csv` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/synthetic_smoke_ablation.json` | `archive/intermediate_runs/results/synthetic_smoke_ablation.json` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/synthetic_smoke_before_after.csv` | `archive/intermediate_runs/results/synthetic_smoke_before_after.csv` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/synthetic_smoke_before_after.json` | `archive/intermediate_runs/results/synthetic_smoke_before_after.json` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/synthetic_smoke_figure_bundle.json` | `archive/intermediate_runs/results/synthetic_smoke_figure_bundle.json` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/synthetic_smoke_figure_data.npz` | `archive/intermediate_runs/results/synthetic_smoke_figure_data.npz` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/synthetic_smoke_fingerprint.json` | `archive/intermediate_runs/results/synthetic_smoke_fingerprint.json` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/synthetic_smoke_fingerprint.npz` | `archive/intermediate_runs/results/synthetic_smoke_fingerprint.npz` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/synthetic_smoke_geometry_summary.json` | `archive/intermediate_runs/results/synthetic_smoke_geometry_summary.json` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/synthetic_smoke_history.csv` | `archive/intermediate_runs/results/synthetic_smoke_history.csv` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/synthetic_smoke_history.json` | `archive/intermediate_runs/results/synthetic_smoke_history.json` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/synthetic_smoke_neuron_activity_representations.json` | `archive/intermediate_runs/results/synthetic_smoke_neuron_activity_representations.json` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/synthetic_smoke_neuron_representations.json` | `archive/intermediate_runs/results/synthetic_smoke_neuron_representations.json` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/synthetic_smoke_representations_summary.json` | `archive/intermediate_runs/results/synthetic_smoke_representations_summary.json` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/synthetic_smoke_space_activity.npz` | `archive/intermediate_runs/results/synthetic_smoke_space_activity.npz` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/synthetic_smoke_space_full.npz` | `archive/intermediate_runs/results/synthetic_smoke_space_full.npz` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/synthetic_smoke_space_primary.npz` | `archive/intermediate_runs/results/synthetic_smoke_space_primary.npz` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/synthetic_smoke_space_structural.npz` | `archive/intermediate_runs/results/synthetic_smoke_space_structural.npz` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |
| `results/synthetic_smoke_train_summary.json` | `archive/intermediate_runs/results/synthetic_smoke_train_summary.json` | intermediate run output (smoke/debug/verification run); superseded, no tracked reference | no | yes | 2026-09-27 |

## Deliberately NOT archived (ambiguous or active - kept in place)

Each of the following was inspected and reference-checked, and was **kept** because it is
actively required or is evidence for a statement in the current documentation:

| artifact family | why kept |
|---|---|
| `configs/baseline.yaml`, `configs/baseline_repaired.yaml`, `configs/analysis.yaml`, `configs/neuron_space_baseline.yaml` | `tests/test_v2_config.py` loads all four; `neuron_space_baseline.yaml` is the canonical V2/NSB config and references `baseline_repaired.yaml`; `analysis.yaml` is referenced by README and by `multiseed.analysis_config` |
| `configs/v2_example.yaml` | default `--config` of `scripts/show_v2_config.py`; used by `tests/test_v2_config.py` |
| all `scripts/*.py` | referenced by README (run instructions) and/or by `tests/test_v2_config.py` (`show_v2_config`); `_common.py`/`_pipeline.py` are imported by the other scripts |
| `AUDIT.md`, `AUDIT_DYNAMICS.md`, `AUDIT_REPRESENTATION.md`, `LIF_BASELINE_REPORT.md`, `NEURON_SPACE_BASELINE_REPORT.md` | kept here in the first pass because they were linked from README/`src/neurons.py`; in the second (documentation) pass their current content was summarised into `V2_STATUS.md` and they were archived to `archive/documentation/` with all active links repointed - see the section above |
| `V2_CONFIG_IMPLEMENTATION.md`, `V2_NEURON_RECORD_IMPLEMENTATION.md`, `V2_STRUCTURED_VECTOR_IMPLEMENTATION.md` | kept in the first pass as current stage reports; archived to `archive/documentation/` once `V2_STATUS.md` became the single status document |
| `VECTOR_V2_AUDIT.md` | **still active** - the permanent design audit behind the V2 architecture (kept at the repository root) |
| `results/runcheck_*`, `checkpoints/runcheck.pt` | cited as the failing-run evidence in `AUDIT_DYNAMICS.md` |
| `checkpoints/sweep_*.pt` (incl. `sweep_l2_0.pt`) | `sweep_l2_0.pt` is the selected baseline used by tests/reports; the sweep checkpoints are referenced by the `checkpoint` column of `results/lif_baseline_sweep*.csv` and substantiate `LIF_BASELINE_REPORT.md` |
| `checkpoints/baseline.pt`, `baseline_v2.pt`, `baseline_repaired_v1.pt`, `baseline_s1.pt`, `baseline_s2.pt`, `nsb_seed1.pt`, `nsb_seed2.pt` | cited in the audits/reports; `nsb_seed1/2.pt` + `sweep_l2_0.pt` are exercised by `tests/test_neuron_record.py` |
| `results/baseline_*`, `results/baseline_s1_*`, `results/baseline_s2_*`, `results/multiseed_summary.*`, `results/lif_baseline_*` | substantive (superseded but cited) experiment results; inputs of the referenced summaries |
| `results/neuron_space_baseline/**` | the canonical current results (`NEURON_SPACE_BASELINE_REPORT.md`) |
| `figures/**` | referenced by README and the multiseed report section |
| `data/**` | the active SHD dataset (no re-download needed) |

No tracked file was archived, so no `git mv` was required.

## Archived documentation (8 files, 2026-09-27, second consolidation pass)

The V2 documentation consolidation deliberately reduced the active Markdown set to
three files (`README.md`, `VECTOR_V2_AUDIT.md`, `V2_STATUS.md`). The historical
implementation reports below were moved with `git mv` (history preserved) into
`archive/documentation/`. **Their contents were not edited during the move**, so a moved
document may contain links to its original root-level location or to another archived
document. Everything the current architecture needs from them was summarised into
`V2_STATUS.md` first.

| original path | archive path | category | reason | tracked | refs checked | date |
|---|---|---|---|---|---|---|
| `AUDIT.md` | `archive/documentation/AUDIT.md` | historical audit | whole-repository audit of the V1 pipeline; superseded by `VECTOR_V2_AUDIT.md` + `V2_STATUS.md` | yes (`git mv`) | yes | 2026-09-27 |
| `AUDIT_DYNAMICS.md` | `archive/documentation/AUDIT_DYNAMICS.md` | historical audit | LIF dynamics/saturation audit for the V1 baseline; the operative facts are in `V2_STATUS.md` | yes (`git mv`) | yes | 2026-09-27 |
| `AUDIT_REPRESENTATION.md` | `archive/documentation/AUDIT_REPRESENTATION.md` | historical audit | feature-level audit that defined the 48-D V1 features; current dimensions are in `V2_STATUS.md` | yes (`git mv`) | yes | 2026-09-27 |
| `LIF_BASELINE_REPORT.md` | `archive/documentation/LIF_BASELINE_REPORT.md` | historical report | baseline repair/sweep report for the V1 recipe | yes (`git mv`) | yes | 2026-09-27 |
| `NEURON_SPACE_BASELINE_REPORT.md` | `archive/documentation/NEURON_SPACE_BASELINE_REPORT.md` | historical report | stage-7 scientific results; superseded as *status* by `V2_STATUS.md` (the results themselves are unchanged) | yes (`git mv`) | yes | 2026-09-27 |
| `V2_CONFIG_IMPLEMENTATION.md` | `archive/documentation/V2_CONFIG_IMPLEMENTATION.md` | stage report | V2 configuration stage detail; summarised in `V2_STATUS.md` | yes (`git mv`) | yes | 2026-09-27 |
| `V2_NEURON_RECORD_IMPLEMENTATION.md` | `archive/documentation/V2_NEURON_RECORD_IMPLEMENTATION.md` | stage report | NeuronRecord stage detail; summarised in `V2_STATUS.md` | yes (`git mv`) | yes | 2026-09-27 |
| `V2_STRUCTURED_VECTOR_IMPLEMENTATION.md` | `archive/documentation/V2_STRUCTURED_VECTOR_IMPLEMENTATION.md` | stage report | structured-encoder stage detail; summarised in `V2_STATUS.md` | yes (`git mv`) | yes | 2026-09-27 |

Reference updates performed in the same pass (no broken active links remain):

| file | change |
|---|---|
| `README.md` | added the documentation map (pointing to `V2_STATUS.md`, `VECTOR_V2_AUDIT.md`, `archive/documentation/`, `archive/ARCHIVE_MANIFEST.md`); 6 links/mentions of `AUDIT_REPRESENTATION.md`, `AUDIT_DYNAMICS.md`, `LIF_BASELINE_REPORT.md` repointed to `archive/documentation/...` |
| `VECTOR_V2_AUDIT.md` | reference to `NEURON_SPACE_BASELINE_REPORT.md` repointed to `archive/documentation/...` |
| `configs/baseline_repaired.yaml` | comment reference to `AUDIT_DYNAMICS.md` repointed to `archive/documentation/...` |
| `src/neurons.py` | docstring reference to `AUDIT_REPRESENTATION.md` repointed to `archive/documentation/...` |
| `archive/README.md` | layout updated with `documentation/`; active documentation set stated; note that archived documents keep their original internal links |

## Verification after archiving

* `git status --short` reports only the new `archive/` documentation files (the moved
  binaries are ignored via `archive/intermediate_runs/.gitignore`, matching the existing
  policy for `results/` and `checkpoints/`).
* `uv run pytest -q` passes (231 tests at the time of the run-artifact pass; the full suite
  was later 282 after the structured-encoder stage and 335 after the residual + composition
  stage) - no import, path or config reference was broken.
* `git grep` for every archived family name (`readme_check`, `shd_tiny`,
  `synthetic_smoke`, `debug_v2`, `nsb_smoke`, `nsb_run.log`) returns no matches in the
  tracked tree.
* Documentation pass: `git mv` recorded all eight moves as renames (history preserved);
  `git grep` for each moved document name finds only the updated
  `archive/documentation/...` references; the active Markdown set is exactly
  `README.md`, `VECTOR_V2_AUDIT.md`, `V2_STATUS.md` at the repository root; the full test
  suite still passes and no source, test, config or script file was modified except the
  four reference/documentation edits listed above.
