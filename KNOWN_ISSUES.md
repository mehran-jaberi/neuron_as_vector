# KNOWN_ISSUES.md — flimsy or questionable results to fix later

Living list, opened by PI direction (2026-11-01). Each entry states exactly what is weak,
what the result currently claims, and what would close it. Do not silently delete entries;
strike them through with a date and the closing evidence.

| # | issue | current claim | why it is flimsy | what closes it |
|---|---|---|---|---|
| 1 | **N=100 20-epoch result has no N=100 baseline** (2026-11-01) | `v3_n100_aug_dropout_ep20` = 1995/2264 = **88.12 %** (N=100, D=1000, event dropout 0.10, 20 ep) | there is no N=100, 20-epoch, no-augmentation run, so "more neurons" and "dropout" cannot be separated; the number is one lever-pulled configuration | one N=100 20-epoch baseline run (~115 min). PI deferred it (m00887: "no"); required before any attribution sentence |
| 2 | **scalar comparator is untuned** | vector 82.51 % vs scalar D=1 49.43 % (and capacity-matched scalars 31–35 %) | the scalar control was never tuned (published recurrent LIF reaches ~71 % on SHD), so the gap overstates the architectural advantage | a tuned scalar baseline (lr sweep at the same parameter count), or explicit "untuned control" wording everywhere |
| 3 | **single seed** | every headline number is seed 0 | no error bar; a ±1 pp difference cannot be resolved | multi-seed runs (PI: last thing, possibly another machine) |
| 4 | **VAL split is speaker-overlapping** | VAL 96.3–97.1 % on 20-epoch runs | the split is a class-stratified random 10 %, so VAL shares speakers with FIT and is optimistic | speaker-held-out split + rerun, or "do not quote VAL as a result" |
| 5 | **4 ms 79.81 % reference** | doc-only claim in archived V2 notes | no registry row, checkpoint or log survives | **closed 2026-11-03**: the claim was deleted from `README.md`/`RESULTS.md` (PI order); the 4 ms setting survives only as a documented config option |
| 6 | **selection on the official TEST set** | variants chosen by TEST accuracy at 5 epochs | winner's-curse inflation of the quoted number (n=2264, ±0.84 pp SE) | disclose in every table (done); a proper held-out selection split would close it |
| 7 | **training schedule experiments are meaningless while FIT saturates** (2026-11-02) | 20-epoch runs reach 99.9–100 % train accuracy (baseline, M1, M2) | longer schedules cannot buy anything when the training set is already memorised; the 20-epoch budget is *not* the binding constraint | only after the final architecture underfits at 20 epochs: run longer schedules then, and treat schedule as a tuning knob, not a lever |

| 8 | **final-model block set chosen on 5-epoch TEST** (2026-11-03) | `v3_n100_spec_ep30` = 1994/2264 = **88.07 %** (best epoch 2009 = 88.74 %) | the nine enabled blocks were the arms that looked best at 5 epochs (plus the two that were merged for infrastructure reasons), so the 88 % is a winner's-curse number; M8 and M12 scored exactly +0.13 and 0.00 pp at 5 epochs and are still on | an ablation of the final set at 30 epochs (drop-one / add-one), or a selection split never used for choosing |

## Notes

- Items 2–6 were already disclosed in `V3/README.md` / `INVESTIGATION.md`; they stay listed
  here so nothing is lost between workstreams.
- `INVESTIGATION.md` remains the ledger of record for numbers; this file is the debt list.
