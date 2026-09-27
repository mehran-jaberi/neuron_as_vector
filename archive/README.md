# archive/

**This directory is intentionally retained history. It is not the active implementation.**

Nothing here is imported, executed, configured or required by the current repository:

* the active code lives in `src/` and `scripts/`,
* the active configurations live in `configs/`,
* the active tests live in `tests/`,
* the canonical experiment and its results live in `scripts/run_neuron_space_baseline.py`
  and `results/neuron_space_baseline/`,
* the active documentation is exactly three files at the repository root:
  `README.md` (project + canonical V1/NSB experiment), `VECTOR_V2_AUDIT.md` (the design
  audit behind the V2 architecture) and `V2_STATUS.md` (**the single current-status
  document** for the implemented V2 architecture).

## Contents

```
archive/
    README.md                 <- this file
    ARCHIVE_MANIFEST.md       <- one row per archived artifact (runs and documentation)
    documentation/            <- historical audits, baseline reports and stage reports
    intermediate_runs/
        checkpoints/          <- tiny/smoke/debug training checkpoints
        results/              <- smoke/verification run outputs
```

### documentation/

Historical documents moved from the repository root in the V2 documentation consolidation
(`AUDIT.md`, `AUDIT_DYNAMICS.md`, `AUDIT_REPRESENTATION.md`, `LIF_BASELINE_REPORT.md`,
`NEURON_SPACE_BASELINE_REPORT.md`, `V2_CONFIG_IMPLEMENTATION.md`,
`V2_NEURON_RECORD_IMPLEMENTATION.md`, `V2_STRUCTURED_VECTOR_IMPLEMENTATION.md`). They are
preserved **exactly as written** - their content was not edited during the move, so a
document inside `documentation/` may contain links that refer to its original root-level
location or to another archived document. Read them as historical records; for the current
state use `V2_STATUS.md`.

Everything the current architecture needs from these documents is summarised in
`V2_STATUS.md` (documentation map, architecture, dimensions, memory rules, label-leakage
contract, tests, limitations).

### intermediate_runs/

Generated artifacts (smoke/debug checkpoints and run outputs) moved in the earlier
cleanup. They were **moved, never deleted**, with contents preserved byte-for-byte, and are
ignored by git for the same reason the original `checkpoints/` and `results/` directories
are (large generated binaries); see `intermediate_runs/.gitignore`.

## Rules

* Nothing in this archive is deleted; `archive/ARCHIVE_MANIFEST.md` records every move
  (original path, archive path, category, reason, reference check, date).
* Do not copy archived files back into the active tree, and do not point active code,
  configurations, tests or documentation at archived paths other than by explicit
  historical reference such as `archive/documentation/<name>.md`.
