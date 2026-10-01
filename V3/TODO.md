# V3 TODO

## Immediate next step (deferred by request)

- [x] get a measured full-dimensional result — done at **3 epochs**: vector
      `D=1000` = **71.78%** (1625/2264), scalar `D=1` = **26.55%** (601/2264) on the
      official SHD test set (see `RESULTS.md`)
- [ ] **run the full 20-epoch schedule** (the value already in
      `configs/v3_default.yaml`):
      `.venv\Scripts\python.exe V3\tools\run_notebook.py` — ≈ 43 min for the
      vector model + ≈ 22 min for the scalar baseline, plus evaluation. No code
      change is needed; the notebook picks `epochs: 20` up from the YAML.

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
- [ ] `tau_ms` / `dt_ms` / `bin_ms` sweep (currently T=250 x 4 ms fixed)
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
