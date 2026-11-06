# V3 TODO

## Current stage: controlled state imprecision (+ persistent run registry)

Implemented (see `README.md` §11 for the full description):

- [x] `state_regularization` block: `mode` ∈ {none, noise, quantization,
      noise_quantization}, `noise_std`, `quantize_bits`, `quantize_clip`, and
      per-phase `apply_during_training/validation/test` flags. Disabled by
      default; `mode: none` is bit-identical to the baseline.
- [x] noise applied to the **internal state** `z(t)` where the next timestep
      reads it (not to the classifier input / labels / input data)
- [x] differentiable STE state quantizer (dynamic range, bounded by the `tanh`
      invariant, fp16-safe)
- [x] training-only by default; validation and official test stay at full
      precision unless explicitly enabled
- [x] persistent registry: one row per completed run in `V3/results/runs.csv`
      plus a timestamped `V3/results/<run_id>/` with `config.yaml`,
      `metrics.json`, `confusion_matrix.csv`, `summary.txt`
- [x] the notebook registers completed runs through the **same**
      `v3.registry.register_run` entry point as the CLI (no second format)
- [x] timing block (`sequence_duration_ms`, `time_bin_ms`) with the step count
      **derived** and validated as an integer: 2 ms -> 500 steps (**default**),
      4 ms -> 250 steps (historical reference); `time_bin_ms` is also the model `dt`
- [x] verified data order: fixed stratified FIT/VAL split, FIT shuffled per epoch
      (`shuffle_train`), deterministic VAL (`shuffle_val`), fresh state per batch
- [x] focused tests in `V3/tests/` (noise, quantization, phase gating, timing,
      SHD binning, shuffle/state reset, registry)

### Screening status

Measured runs (all single seed, `N=64`, `D=1000`, 2,025,440 params; full rows in
`V3/results/runs.csv`):

| when | discretization | epochs | state regularization | test | correct/total |
|---|---|---|---|---|---|
| 2026-10-02 11:24 | 4 ms (T=250) | 5 | none | 72.26% | 1636/2264 |
| 2026-10-02 11:39 | 4 ms (T=250) | 5 | noise std 0.01 | 74.47% | 1686/2264 |
| 2026-10-02 11:57 | 4 ms (T=250) | 5 | quantization 8-bit | 73.94% | 1674/2264 |
| 2026-10-02 12:40 | **2 ms** (T=500) | 5 | none | 76.15% | 1724/2264 |
| **2026-10-02 14:31** | **2 ms** (T=500) | **20** | none | **82.51%** | **1868/2264** |

- [x] baseline regression run (`mode: none`) — same params/forward logic, no regression
- [x] first 5-epoch screening: noise `std 0.01` and 8-bit quantization (both at 4 ms)
- [x] 2 ms resolution run (`time_bin_ms=2` → 500 steps, `alpha = 0.1`) —
      5-epoch 76.15%, 20-epoch **82.51%**
- [x] noise sweep at the current 2 ms default: `std` ∈ {0.001, 0.005, 0.01, 0.02, 0.05}
      0.02 worked best in 5 epoch runs. Did not translate to any improvements in 20 epochs.
- [x] quantization sweep at 2 ms: `bits` ∈ {16, 12, 8, 6, 4}
      loss of accuracy when lowering Q
- [x] combined `noise_quantization` for the most promising single settings
      redundent after the last experiment
- [ ] Higher percision/alternate data format runs: bf16, fp32
- [x] 20-epoch reruns of the promising imprecision regime(s), compared against the
      **82.51%** reference

Reference (2 ms, 20 epochs, `mode: none`):

```
VECTOR  D=1000  N=64    2,025,440 params   test  1868/2264 = 82.51%
```

The YAML default is now `epochs: 20` at 2 ms; 5-epoch screening runs are ~17 min
each, the full 20-epoch run took ~72 min of vector training (4243.5 s) plus
1380.6 s for the scalar baseline.

### Future imprecision work (not implemented)

- [x] per-neuron / per-coordinate quantization (currently one global scale)
      redundant after the last experiments
- [ ] learned or scheduled `noise_std` / `bits` (annealing)
- [x] other quantization schemes (logarithmic, stochastic rounding, QAT fake-quant)
      redundant after the last experiments
- [ ] adversarial / Worst-case state perturbation as a robustness probe
- [ ] imprecision applied to the input drive or the emitted population signal

Everything below is **not implemented**. Nothing in this file is wired into the
active interface; if a feature is implemented it works, if it is not it lives here.

## Model

- [x] alternative vector dynamics (e.g. explicit vector membrane + adaptation
      channels instead of a single leaky tanh state) — **tested 2026-11-02**
      (M1: adaptation channel added to the threshold, leaky tanh kept, branch
      `feat/m1-adaptation`): 5-epoch TEST **81.85 % vs 76.15 % baseline
      (+5.70 pp)**, the only feature with a screening gain that survived the
      re-screen. Not yet confirmed at 20 epochs.
- [ ] alternative neuron-interaction mechanisms
      - full per-neuron `D x D` recurrent operator (currently shared + rank-R)
      - attention/gating over the population signal instead of `c_i(t) * h(t)`
      - delay lines / axonal conduction delays between neurons
- [x] per-neuron emission matrices (currently one shared `W_emit`) — **tested
      2026-11-02, dead** (M5 rank-8 correction, branch `feat/m5-emission`):
      5-epoch 75.80 % (−0.35 pp) for +520 k params
- [x] trainable `tau` (leak) per neuron or per coordinate — **tested 2026-11-02,
      no effect** (M2, `feat/m2-tau-z0`): the learned `alpha` stayed at 0.098
      (init 0.1, per-neuron range 0.078–0.110) even after 20 epochs
- [x] learnable initial state `z_i(0)` instead of zeros — **tested 2026-11-02,
      adopted** (M2): 20-epoch TEST **85.20 % vs 82.51 % (+2.69 pp)**; the gain
      comes from `z0` (mean |z0| 0.054), not from `alpha`
- [ ] >1 spiking mechanism (currently Heaviside + fast-sigmoid surrogate) —
      arctan tested (M6), diverged at `surrogate_alpha=1.0` in fp16 (see
      `KNOWN_ISSUES.md`); boxcar / exponential / SLAYER still untested
- [ ] state normalisation / homeostasis other than the firing-rate regulariser
- [ ] multi-layer populations (V3 nests only one vector-neuron layer)

## Training

- [ ] alternative surrogate gradients (boxcar, exponential, arctan, SLAYER-`alpha`)
      — arctan tested 2026-11-02: diverged (197 Hz firing, AMP gradient-scale
      floor, loss explosion) at `surrogate_alpha=1.0`; needs alpha < 1 / clipping
      before it can be judged. The others are untested.
- [ ] alternative optimisers (AdamW with decoupled decay, SGD+momentum, LAMB)
      — AdamW (`weight_decay=0.01`) tested 2026-11-02, dead: 75.93 % (−0.22 pp)
- [ ] per-group learning rates (state dynamics vs readout)
      — readout LR ×3 tested 2026-11-02, dead: 75.97 % (−0.18 pp)
- [ ] longer training schedules / early stopping on validation loss
- [x] data augmentation (time-shift, channel dropout, event jitter/sub-sampling),
      which is how the published SHD numbers close the train/test speaker gap
      — implemented 2026-11-01 (raw events, before binning) and screened
      2026-11-01/02; dropout 0.10 is the only op that survived to 20 epochs
      (+0.75 pp, not distinguishable) and the ops do not stack
- [ ] truncated BPTT with overlap, or forward-mode/hebbian alternatives
- [ ] `torch.compile` / fused per-step kernel (the Python timestep loop is the
      current throughput bottleneck: ~75 samples/s at N=64, D=1000, T=250)

## Robustness (input augmentation — the next workstream)

Specified by the PI (2026-10-26) from an external design note. The hard rule:
augmentation acts on **raw events before binning**, never on the already-binned
binary tensor, and never on VAL or official TEST.

```text
raw SHD event stream → augmentation → 2-ms binning → 500 × 700 → V3
```

**Implemented 2026-11-01** (`V3/v3/event_augmentation.py`, `V3/v3/control.py`,
38-cell notebook); all five ops measured, results in `INVESTIGATION.md`.

- [x] event-augmentation module (one reusable `augment(events, config, rng)` API,
      e.g. `V3/v3/event_augmentation.py`) — no per-augmentation loaders, no
      duplicated binning, existing binning consumes the augmented events
- [x] **A. timestamp jitter**: `t_i' = t_i + ε_i`, `ε_i ~ N(0, σ_t²)`,
      `time_jitter_std_ms` configurable (start at 0.5), events leaving the
      1000 ms window follow the existing windowing policy, fresh realisation per
      training sample
- [x] **B. event dropout**: drop each raw event with
      `event_dropout_probability`, labels preserved, source data never mutated
- [x] **C. event insertion**: add artificial events with valid SHD channels
      inside the 1000 ms window, parameterised by an unambiguous insertion
      amount/probability
- [x] **D. global time shift**: `t_i' = t_i + Δ`, symmetric `Δ` bounded by
      `global_time_shift_max_ms`, sequence duration unchanged
- [x] **E. channel dropout**: choose channels, remove all their events for that
      sample, then bin (channel identities, not the post-binning tensor)
- [x] training-only phase gating: FIT on, VAL off, TEST off — must not leak when
      the same dataset objects are reused for evaluation
- [x] config block + CLI/notebook wiring, disabled by default (`mode: none`
      equivalent must stay bit-identical to the unaugmented baseline)
- [x] tests: synthetic-event unit tests per augmenter, off-by-default
      bit-identical test, phase-gating test
- [x] control panel: temporary overrides equivalent to `--override` (epochs,
      timing, `N`, `D`, state regularization, augmentation, seed), constructing a
      transient config layer on top of the YAML default — not a second config system

## Experiments

- [ ] optimizer sweep:
    - Adam (current)
    - AdamW
    - SGD + momentum
    - LAMB
- [ ] more `N` values (N=16/32/128/256) to separate "vector state" from "more units"
      — N=100 tested 2026-11-01: 5-epoch 79.64 %, 20-epoch + dropout 0.10
      **88.12 %**; N=16/32/128/256 untested
- [x] more `D` values (D=1, 4, 16, 64, 256, 1000, 4096) - the size of the effect
      — 5-epoch sweep 2026-10-25: D=4 57.20 %, 16 63.74 %, 64 68.60 %,
      256 77.61 %, 1000 76.15 % (D=4096 untested; D=2000/500 measured earlier
      by the PI, no gain / worse)
- [ ] multiple seeds per configuration (the single-seed result has no error bar)
- [x] `tau_ms` / `dt_ms` / `bin_ms` sweep (currently T=500 x 2 ms fixed)
- [ ] speaker-held-out validation (the current VAL split is stratified, not
      speaker-independent - it cannot measure speaker generalisation)
- [ ] learning-curve / dataset-size scaling
- [ ] leaderboard optimisation (this is a first working implementation, not a
      competitive SHD entry)

### Final experiments (PI, 2026-11-02 — run these once the architecture is frozen)

Longer schedules are deferred: every 20-epoch run reaches 99.9–100 % FIT
accuracy, so the schedule is not the binding constraint (`KNOWN_ISSUES.md` #7).
Revisit only if the frozen architecture underfits at 20 epochs.

- [ ] longer training schedules / early stopping on validation loss (deferred)
- [ ] `bf16` vs `fp32` vs `fp16` **accuracy** comparison at the frozen
      architecture (accuracy, not throughput)
- [ ] multi-seed error bars on the final architecture

## Performance

Measured 2026-10-26 (see `PERF.md`): one FIT epoch (N=64, D=1000, T=500, 2 ms,
batch 128) is **196.72 s** warm, GPU utilisation **94.6 %**, host binning
**8.44 s = 4 %** — the "~30 % of the epoch" premise below was false.

- [x] fused step kernels (branch `perf/loop-kernels`, merged 2026-11-06): the
      `pot`/`h` matmuls go through `mm`/`bmm`, the drive/drive-add/tanh/leak are
      fused in place. Measured **4.04 -> 3.18 s/step (1.27x)** at N=100, D=1000,
      B=96; spikes and logits identical, the state agrees to ~1 ulp. The four
      `torch.equal` parity tests were relaxed to `assert_close(1e-5)` for the
      state only (spikes/logits still exact) — fp32 summation order, not maths.
- [ ] **Padé tanh (candidate C): parked as a dead end.** Implemented on branch
      `feat/pade-tanh` (`184de5d`, rebased `904bc30`) with `activation.kind`,
      tests and docs. A single training step aborts inside the checkpointed
      recompute with `RuntimeError: CUDA driver error: device not ready`; it
      reproduces with fp32 state, with one chunk and with the upcast removed,
      runs under `CUDA_LAUNCH_BLOCKING=1` and in an isolated probe, and the tanh
      path trains normally on the same GPU. Not merged; the speed argument is
      already answered by the fused loop above.

- [x] host-side binning overlap / pre-binned cache — landed opt-in
      (`--override binned_cache=true`, default off): 190.52 s (−3.2 %), tensors
      bit-identical, 2.66 GiB under `V3/cache/` (git-ignored). `DataLoader`
      workers measured unnecessary at 96.5 % GPU util.
- [ ] CUDA kernel optimisation / custom fused step (biggest lever: the Python
      500-step loop, not the FLOPs, is the bottleneck). `torch.compile` is not
      usable on this Windows build (`TritonMissing`); a hand-written fused step
      is the remaining option.
- [ ] memory optimisation (gradient checkpointing is already used; chunks=10)
      and memory instrumentation: log `max_memory_reserved` per epoch, and guard
      against ever building an un-checkpointed 500-step graph (~8.4 GiB at D=1000)
- [ ] larger `N`, larger `D`, longer `T`
- [ ] `bf16` comparison against `fp16`
- [ ] multi-GPU / gradient accumulation to raise the effective batch size

## Analysis (deliberately out of scope for V3's primary question)

Done 2026-10-26 on the frozen 20-epoch checkpoint — see `ANALYSIS.md`.

- [x] per-neuron functional fingerprints from the vector state (knockout, 64
      TEST passes: mean 1.02 ± 1.07 pp, no critical unit, MI does not predict
      causal necessity)
- [x] decoder probe on the raw `(N, D)` state (logistic, fit on FIT, TEST n=2264:
      counts 82.51 %, mean state 80.83 %, Δ = −1.68 pp, 95 % CI [−3.00, −0.31])
- [x] state-geometry / PCA / manifold diagnostics (population PR 6.64 of 64,000;
      per-neuron median 4.21 of 1,000; scalar control 1.71)
- [ ] spike-train statistics beyond mean/median/max rate
- [x] ablation: freeze the state dynamics and train only the readout
      (82.02 % vs 82.51 %; untrained dynamics 36.66 %; recurrence zeroed 59.23 %)

## Reproducibility / infrastructure

- [x] resume training from a checkpoint (`--resume <ckpt>`; model, optimizer,
      scheduler, scaler, epoch, RNG restored; new `<tag>_resumed` registry row;
      bit-identical continuation proven for `lr_schedule=none`)
- [ ] deterministic GPU kernels (`torch.use_deterministic_algorithms`)
- [ ] automated regression test for the SHD loader inside `tests/`
      (`V3/run_smoke_test.py` is a standalone script, not a pytest module)
- [x] CI entry point that runs the quick configuration end-to-end
      (`.github/workflows/tests.yml`: `uv sync --locked --group dev` + pytest on
      CPU; not yet executed on GitHub — the repo has not been pushed)

## Feature-branch ledger (2026-11-02)

Six feature branches were cut from `43741d9`, screened at 5 epochs on the
official TEST set (n=2264, SE ±0.87 pp), and re-screened after their bugs were
fixed. Verdicts are on accuracy alone; "dead" means the branch is not merged and
is kept only as history.

| branch | feature | 5-ep first | 5-ep re-screen | 20-ep | verdict |
|---|---|---|---|---|---|
| `feat/m1-adaptation` | adaptive threshold (SFA), gain 0.1 | 75.49 % (gain 0) | **81.85 %** | 82.20 % (−0.31) | **merged** — kept as a default-off block; the 5-epoch gain did not survive 20 epochs |
| `feat/m2-tau-z0` | per-neuron alpha + learnable z0 | 76.86 % | — (20 ep instead) | **85.20 % (+2.69)** | **merged** — the only surviving gain; it comes from `z0`, the learned alpha stayed at 0.098 |
| `feat/m3-reset` | multiplicative spike reset | 75.53 % (beta 0.5) | 75.44 % (beta 0.02) | not run | **dead** — fixed but no gain |
| `feat/m4-delays` | learnable input delays 0–20 ms | 76.15 % (zero-grad bug) | **78.14 %** (fixed) | 81.14 % (−1.37) | **merged** — kept as a default-off block; same 5→20 regression |
| `feat/m5-emission` | per-neuron rank-8 emission | 75.80 % | — | not run | **dead** — −0.35 pp for +520 k params |
| `feat/m6-optimizer` | arctan surrogate / AdamW / readout LR | 20.67 / 75.93 / 75.97 % | — | not run | **dead** — arctan diverged at alpha=1.0 (stability bug, not the surrogate); AdamW and readout-LR are neutral-to-worse |

Baseline row for every comparison: `v3_vector_d1000`, 5 epochs, 1724/2264 =
76.15 %, or 20 epochs, 1868/2264 = 82.51 %.

**Merge (2026-11-02, PI-directed).** `M1`, `M2` and `M4` are merged into `main`
(`9201b18`, `e1fcbfc`, `8074cd2`) as three independent, default-off config
blocks (`adaptation`, `state_dynamics`, `input_delays`). Conflicts were in
`V3/v3/config.py` (three nested-block dataclasses, validation, `from_dict`,
`NESTED_BLOCKS`, `__all__`) and in `V3/v3/model.py` (parameter registration and
the `_segment` signature/return arity: adaptation adds the `b` trace to the
segment tuple). Verified after the merge:
`V3/tests` 257 passed; the disabled path is **bit-identical** to the pre-merge
commit (same parameter count 11876 on the tiny test config and the same logits
hash, cross-checked against a checkout of `e01aaba`); all three features at once
train (forward + backward finite, every new parameter group non-zero gradient);
new `V3/tests/test_feature_combinations.py` pins down "each alone, any pair, all
three" so a later merge cannot silently break the combination.

## Dead weight (do not "fix" silently)

- `adlif.enabled` + `two_stage.enabled`: the adLIF currents subtract from
  `pre`, but the two-stage update rebuilds the state from `mix` and the inner
  `tanh`, never reading `pre`. `adlif_gain_fast` / `adlif_gain_slow` therefore
  get `grad = None` — 200 parameters that never train in the final N=100 model.
  Making adLIF real changes the model and needs its own 20-epoch confirmation.
- `input_delay_line` cannot be combined with `input_current` (the line widens
  the input from C to KC; the scalar current needs C). Validation rejects the
  pair; it is not silently ignored.
- `feat/pade-tanh` stays unmerged (CUDA device-not-ready in the checkpointed
  recompute); the fused loop already provides the speed-up it was meant to buy.
