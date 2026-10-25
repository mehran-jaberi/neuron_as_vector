# AGENTS.md — working rules for every agent in this repo

Read this file and `INVESTIGATION.md` before doing anything. They are the shared memory.
Append findings; do not rewrite history.

## Roles

- **PI (human).** Sets direction, approves every step, approves and directs all commits and
  git changes. Nothing below overrides the PI.
- **Lead.** Owns the conversation, writes task briefs, adjudicates results, keeps
  `INVESTIGATION.md` current, keeps the context window slim.
- **Sub-agent.** Executes exactly one brief. Measures, reports, stops. Never decides scope,
  never picks between scientific options, never commits.

## Hard rules

1. **No step without an explicit go.** Work begins only on a PI message that clears it.
2. **Git is PI-directed.** No commit, branch, merge, rebase, tag, push, or deletion unless
   the PI's message names that operation. Never force-push. Never rewrite published history.
3. **Resource caps** (RTX 3060 Laptop, 6 GiB VRAM, shared with the PI's own work):
   - VRAM **< 80 %** of 6 GiB — i.e. **< ~4.9 GiB total** including processes that are not
     ours. Check `nvidia-smi` before launching; the budget is what is left, not 4.9 GiB.
   - CPU **< 60 %** (≈ 9 of 16 logical processors).
   - RAM **< 80 %** of ~32 GiB (≈ 25 GiB).
   - One training job at a time. Never build an un-checkpointed BPTT graph
     (measured 8.4 GiB at `D=1000`; see `V3/RESULTS.md` §7.0).
4. **Context budget is a hard limit.** Never read a large file whole — windowed reads
   (≤ 200 lines), `grep` for anchors. Reports are tables of numbers + `file:line`
   references, never code dumps. Default report ≤ 300 words.
5. **Keep every result.** All runs are recorded in `V3/results/runs.csv` via
   `v3.registry.register_run`. Never delete a run, never re-run one until it looks good.
   Negative results are results.
6. **Every number carries its provenance**: epochs, seed, `correct/total`, and a tag —
   `confirmed` (20 epochs) / `screening` (≤ 5 epochs) / `doc-claim` (written in a doc,
   not re-verified). A screening number never appears in a headline.
7. **Screening workflow.** For every model-update variant: one 5-epoch run, all variants
   recorded; the selected variant then gets a 20-epoch run; only the 20-epoch number is
   quoted as a result. No multi-seed work until the PI says so.
8. **Test set discipline.** The official test file is never trained on and never used to
   build features. Any selection made using test accuracy must be disclosed.
9. **Changes are gated by tests**: `python -m pytest V3/tests` green, plus
   `V3/run_smoke_test.py` for anything touching model/training code. Tests may be added,
   never weakened.
10. **No new dependencies** and no change to recorded config defaults without PI approval.
11. **File ownership.** A task's write scope is stated in its brief. Do not edit files
    outside it; do not touch `V3/v3/model.py` and `V3/v3/train.py` in the same task as
    another agent.
12. **No silent scope creep.** If the brief is ambiguous, or a measurement contradicts the
    brief, stop and report. Do not "helpfully" fix something else.

## Repo conventions

- Science code lives in `V3/`. V2 and earlier work is archived under `archive/` — readable,
  not maintained, not imported.
- Data: `data/shd_train.h5`, `data/shd_test.h5` (git-ignored).
- One configuration → one registry row → one `V3/results/<run_id>/` directory.
- Docs: `V3/README.md` (what/how), `V3/RESULTS.md` (numbers only), `V3/TODO.md` (not
  implemented), `INVESTIGATION.md` (living ledger).
