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
- [ ] noise sweep at the current 2 ms default: `std` ∈ {0.001, 0.005, 0.01, 0.02, 0.05}
- [ ] quantization sweep at 2 ms: `bits` ∈ {16, 12, 8, 6, 4}
- [ ] combined `noise_quantization` for the most promising single settings
- [ ] 20-epoch reruns of the promising imprecision regime(s), compared against the
      **82.51%** reference

Reference (2 ms, 20 epochs, `mode: none`):

```
VECTOR  D=1000  N=64    2,025,440 params   test  1868/2264 = 82.51%
```

The YAML default is now `epochs: 20` at 2 ms; 5-epoch screening runs are ~17 min
each, the full 20-epoch run took ~72 min of vector training (4243.5 s) plus
1380.6 s for the scalar baseline.

### Future imprecision work (not implemented)

- [ ] per-neuron / per-coordinate quantization (currently one global scale)
- [ ] learned or scheduled `noise_std` / `bits` (annealing)
- [ ] other quantization schemes (logarithmic, stochastic rounding, QAT fake-quant)
- [ ] adversarial / Worst-case state perturbation as a robustness probe
- [ ] imprecision applied to the input drive or the emitted population signal

Everything below is **not implemented**. Nothing in this file is wired into the
active interface; if a feature is implemented it works, if it is not it lives here.

## Model

- [ ] alternative vector dynamics (e.g. explicit vector membrane + adaptation
      channels instead of a single leaky tanh state)
- [ ] alternative neuron-interaction mechanisms
      - full per-neuron `D x D` recurrent operator (currently shared + rank-R)
      - attention/gating over the population signal instead of `c_i(t) * h(t)`
      - delay lines / axonal conduction delays between neurons
- [ ] per-neuron emission matrices (currently one shared `W_emit`)
- [ ] trainable `tau` (leak) per neuron or per coordinate
- [ ] learnable initial state `z_i(0)` instead of zeros
- [ ] >1 spiking mechanism (currently Heaviside + fast-sigmoid surrogate)
- [ ] state normalisation / homeostasis other than the firing-rate regulariser
- [ ] multi-layer populations (V3 nests only one vector-neuron layer)

## Training

- [ ] alternative surrogate gradients (boxcar, exponential, arctan, SLAYER-`alpha`)
- [ ] alternative optimisers (AdamW with decoupled decay, SGD+momentum, LAMB)
- [ ] per-group learning rates (state dynamics vs readout)
- [ ] longer training schedules / early stopping on validation loss
- [ ] data augmentation (time-shift, channel dropout, event jitter/sub-sampling),
      which is how the published SHD numbers close the train/test speaker gap
- [ ] truncated BPTT with overlap, or forward-mode/hebbian alternatives
- [ ] `torch.compile` / fused per-step kernel (the Python timestep loop is the
      current throughput bottleneck: ~75 samples/s at N=64, D=1000, T=250)

## Experiments

- [ ] more `N` values (N=16/32/128/256) to separate "vector state" from "more units"
- [ ] more `D` values (D=1, 4, 16, 64, 256, 1000, 4096) - the size of the effect
- [ ] multiple seeds per configuration (the single-seed result has no error bar)
- [ ] stronger baselines: tuned LIF/ALIF SHD baselines, and the V2 LIF model
      (`sweep_l2_0`, test 0.5998) as an external reference point
- [ ] `tau_ms` / `dt_ms` / `bin_ms` sweep (currently T=500 x 2 ms fixed)
- [ ] speaker-held-out validation (the current VAL split is stratified, not
      speaker-independent - it cannot measure speaker generalisation)
- [ ] learning-curve / dataset-size scaling
- [ ] leaderboard optimisation (this is a first working implementation, not a
      competitive SHD entry)

## Performance

- [ ] overlap host-side event binning with GPU compute (a `DataLoader` with
      worker processes, or a one-off pre-binned cache). Measured: the GPU drops
      to ~20% utilisation once every few batches while the host rebuilds the
      dense `(B, T, C)` tensor with `np.bincount`, which costs ~30% of the epoch
      (3.1 min/epoch observed end-to-end vs 2.1 min/epoch of pure GPU work)
- [ ] CUDA kernel optimisation / custom fused step (biggest lever: the Python
      250-step loop, not the FLOPs, is the bottleneck — ~2.2 s/batch and
      58 samples/s at N=64, D=1000, T=250, batch 128)
- [ ] memory optimisation (gradient checkpointing is already used; chunks=10)
      and memory instrumentation: log `max_memory_reserved` per epoch, and guard
      against ever building an un-checkpointed 250-step graph (~8.4 GiB at D=1000)
- [ ] larger `N`, larger `D`, longer `T`
- [ ] `bf16` comparison against `fp16`
- [ ] multi-GPU / gradient accumulation to raise the effective batch size

## Analysis (deliberately out of scope for V3's primary question)

- [ ] per-neuron functional fingerprints from the vector state
- [ ] decoder probe on the raw `(N, D)` state (linear probe per class)
- [ ] state-geometry / PCA / manifold diagnostics
- [ ] spike-train statistics beyond mean/median/max rate
- [ ] ablation: freeze the state dynamics and train only the readout (this is
      the Brian-experiment limitation V3 exists to avoid, so it is worth
      measuring as a contrast, not as the headline)

## Reproducibility / infrastructure

- [ ] resume training from a checkpoint
- [ ] deterministic GPU kernels (`torch.use_deterministic_algorithms`)
- [ ] automated regression test for the SHD loader inside `tests/`
      (`V3/run_smoke_test.py` is a standalone script, not a pytest module)
- [ ] CI entry point that runs the quick configuration end-to-end
