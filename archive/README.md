# archive/

**This directory is intentionally retained history. It is not the active implementation.**

Nothing here is imported, executed, configured or required by the current repository:

* the active code lives in `src/` and `scripts/`,
* the active configurations live in `configs/`,
* the active tests live in `tests/`,
* the canonical experiment and its results live in `scripts/run_neuron_space_baseline.py`
  and `results/neuron_space_baseline/`,
* the current documentation is the repository `README.md` plus the stage reports at the
  repository root (`VECTOR_V2_AUDIT.md`, `V2_CONFIG_IMPLEMENTATION.md`,
  `V2_NEURON_RECORD_IMPLEMENTATION.md`, `V2_STRUCTURED_VECTOR_IMPLEMENTATION.md`) and the
  established audits (`AUDIT.md`, `AUDIT_DYNAMICS.md`, `AUDIT_REPRESENTATION.md`,
  `LIF_BASELINE_REPORT.md`, `NEURON_SPACE_BASELINE_REPORT.md`).

## Contents

```
archive/
    README.md                 <- this file
    ARCHIVE_MANIFEST.md       <- one row per archived artifact
    intermediate_runs/
        checkpoints/          <- tiny/smoke/debug training checkpoints
        results/              <- smoke/verification run outputs
```

The archived artifacts under `intermediate_runs/` were **moved, never deleted**, with
their contents preserved byte-for-byte. They are ignored by git for the same reason the
original `checkpoints/` and `results/` directories are (large generated binaries); see
`intermediate_runs/.gitignore`.

If you need an archived artifact for historical inspection, read it from here - do not copy
it back into the active tree and do not point active code, configs or tests at it.
