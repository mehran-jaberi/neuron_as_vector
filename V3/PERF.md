# V3 performance work — measured 2026-10-26

Machine: Windows 11, RTX 3060 Laptop (6 GiB, 1026 MiB already held by another
process), 16 logical CPUs, Python 3.11.14, torch 2.14.0+cu126, numpy 2.4.6.

Protocol for every row: SHD **FIT** split only (7340 samples), N=64, D=1000,
2 ms bins (T=500), batch 128, fp16 state + AMP, grad checkpointing (10 chunks),
1 epoch per timed row, official test set untouched.
Tool: `V3/tools/bench_epoch.py` (GPU utilisation sampled from `nvidia-smi` every
0.25 s; `data` = accumulated `store.batch()` wall time inside the epoch).

## Results

| # | path | epoch | s/epoch | samples/s | data s (share) | GPU util mean/max | peak alloc/reserved MiB |
|---|------|------:|--------:|----------:|---------------:|------------------:|------------------------:|
| 1 | on-the-fly `np.bincount` | 1 (cold pages) | 212.14 | 34.6 | 24.57 (12 %) | 86.6 % / 99 % | 2253 / 2306 |
| 2 | on-the-fly `np.bincount` | 2 | **196.72** | 37.3 | 8.44 (4 %) | 94.6 % / 99 % | 2253 / 2306 |
| 3 | pre-binned cache (uint8) | 1 | 191.03 | 38.4 | 3.30 (2 %) | 95.8 % / 99 % | 2253 / 2306 |
| 4 | pre-binned cache (uint8) | 2 | **190.52** | 38.5 | 2.62 (1 %) | 96.5 % / 99 % | 2253 / 2306 |

- Warm-vs-warm gain of the cache: **6.20 s/epoch = 3.2 %** (196.72 → 190.52 s).
- First-epoch gain (also removes the cold h5py read): 21.11 s = 10.0 %.
- Cache build is one-off: 34.5 s / 8156 train samples → 2.66 GiB `dense.uint8`
  under `V3/cache/binned/<key>/` (git-ignored, rebuilt by
  `V3/tools/build_binned_cache.py`); test split ≈ 0.74 GiB.
- Bit-identity: covered by `V3/tests/test_binned_cache.py` (`torch.equal` against
  the on-the-fly tensors for fit/val/test iteration order and random batches).

**The TODO premise does not hold here.** The GPU is not idling to ~20 % while
host binning runs: mean utilisation is 86.6 % (cold pages) / 94.6 % warm (max 99 %) with the
on-the-fly path, and host binning is 4 % of a warm epoch (8.4 s of 196.7 s), not ~30 %.
The cache still wins, but by 3 %, because the dense `(B,T,C)` rebuild
(`bincount` zero-fills ~358 MB int64, then a 179 MB float32 copy) is mostly
overlapped with, or small compared to, the 3.4 s of GPU work per batch.
`binned_cache` therefore stays **opt-in (default False)**; enable with
`--override binned_cache=true`.

## Resume from checkpoint

`V3/run_experiment.py --resume <checkpoint>` restores model, optimizer,
scheduler, GradScaler, epoch counter and RNG, trains up to
`--override epochs=N`, and writes a **new** registry row whose tag is
`<original>_resumed` (the original row and run directory are untouched; the tag
is not suffixed twice on chained resumes). Checkpoints are now
`schema: v3_checkpoint/v2` and carry that state; v1 checkpoints still load
(model + epoch only, with a printed warning that the optimizer restarts).
Quick config on CPU (`N=16, D=64`) 1 epoch → `--resume --override epochs=2`:
row 1 `perf_resume_e2e` (1 epoch), row 2 `perf_resume_e2e_resumed` (2 epochs,
`notes: resumed_from=…`). Scheduler caveat: the restored LR schedule is
re-evaluated against the **new** total epoch count, so a resumed run is not
step-for-step identical to an uninterrupted N-epoch run unless both used the
same total; with `lr_schedule=none` the continuation is bit-identical and
`V3/tests/test_resume.py` asserts exactly that.

## CI

`.github/workflows/tests.yml`: `uv sync --locked --group dev` then
`uv run --no-sync pytest V3/tests -q` on `ubuntu-latest`. `pip install -e .` is
not supported (`[tool.uv] package = false`, no build backend). `V3/tests` is
**dataset-free**: no test opens `data/shd_*.h5`; synthetic HDF5 files are written
per test. Note the lock routes torch to the cu126 index, so CI downloads a CUDA
wheel even though it runs on CPU.

## torch.compile — gate closed

`V3/tools/compile_probe.py` on one `_segment` call (B=128, 10 timesteps,
fp16 state): eager **44.8 ms/call**; both `default` and `fullgraph=True` fail
after ~3–6 s with

```
torch._inductor.exc.TritonMissing: Cannot find a working triton installation.
```

(`torch/_inductor/scheduler.py:9312`; also `flop_counter.py:113: triton not
found`). There is no working Triton on this Windows build, so Inductor cannot
generate kernels; no further work on the fused step.

## Not attempted

- DataLoader worker prefetch: measured ceiling is the residual host time —
  2.6 s/epoch (1.3 %) with the cache, 8.4 s (4.3 %) without — while the GPU is
  already at 96.5 % mean utilisation. Windows `spawn` would also duplicate the
  store/code cache per worker. Not worth the risk; no flag added.
- uint8 → fp16 conversion in the cache loader (would save one 179 MB copy/batch;
  kept float32 so the bit-identity test can use `torch.equal`).
- Pinning the dense cache in shared memory / `madvise`, and building the cache
  in parallel (build is a single-process 34.5 s one-off).
- Any change to the model kernel, batch size, or AMP numerics.
