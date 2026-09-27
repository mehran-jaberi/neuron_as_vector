# LIF Baseline Report — finding a trustworthy recurrent LIF baseline on SHD

Date: 2026-09-26. Scope: establish a **scientifically defensible recurrent LIF
baseline** for SHD (healthy dynamics + competent classification + stable training +
no obvious class collapse). **Not** in scope: adaptive LIF, graph embeddings,
dimensionality sweeps, FP8, Optuna, notebooks, or any neuron-space conclusion.

> **Selection discipline.** Every configuration is selected on **DEV only**. PROBE
> and the official **TEST** set are never used to select anything; TEST is evaluated
> once for the single selected configuration (§8). No configuration is chosen for
> its neuron-geometry correlation.

Artifacts: `results/lif_baseline_sweep*.csv`, `results/lif_sweep/<label>.json`,
`scripts/run_lif_sweep.py`, `scripts/diagnose.py`.

---

## 1. Equations as implemented (`src/model.py`)

Let `Δt = bin_ms`, `T = n_bins`. For hidden neuron `h`:

**Input (event counts per bin).**
`x_t[c] = #{events on channel c with floor(t_ms/Δt) = t}` (`np.add.at` / `index_add_`).

**Synaptic current** (current-based; zero-order-hold exact decay):
```
I_t = β·I_{t-1} + (W_inᵀ x_t)_h + (W_rec s_{t-1})_h + b_hid ,   β = exp(−Δt/τ_syn)
```
`W_rec[i,j]` = weight **from j to i**; the recurrent term uses `s_{t-1}` **only**
(counted once).

**Membrane potential:**
```
V_t = α·V_{t-1} + (1−α)·I_t ,                                   α = exp(−Δt/τ_mem)
```

**Threshold / spike (surrogate):** `s_t = Θ(V_t − θ)`, θ = 1.0; forward is exactly
the Heaviside, the backward pass uses the SuperSpike fast-sigmoid `γ/(1+β|u|)²`.

**Reset:** `V_t ← V_t − θ·s_t` (subtract) or `V_t·(1−s_t)` (zero); the spike
indicator is detached from autograd.

**Readout (configurable aggregation `model.readout_mode`):**
```
O_t = ρ·O_{t-1} + W_outᵀ s_t + b_out
logits = (1/T)·Σ_t O_t      (mode "mean", historical default)
logits = O_T                (mode "last", final state only)
logits = Σ_t O_t            (mode "sum", accumulated)
```
With `ρ = 0` the readout is linear, so **"sum" = T × "mean" exactly** (a
reparameterisation, tested in `tests/test_readout.py`). "last" is genuinely
different information.

Derived constants for the baseline window (Δt = 2 ms, τ_mem = 20 ms, τ_syn = 5 ms):
`α = 0.9048`, `β = 0.6703`, synaptic DC gain `1/(1−β) = 3.03`.

## 2. Regulariser audit (`l2_spikes`, `target_rate_hz`)

Implemented in `src/training.py::compute_loss` as

```
loss = CE(logits, y) + l2_spikes · mean_h ( (rate_h − target_rate_hz)² )
rate_h = ( mean_over_batch( spike_count_per_sample[h] ) ) / duration_s ,  duration_s = T·Δt/1000
```

* `rate_h` **is a firing rate in Hz** (spikes per sample divided by the simulated
  duration in seconds), averaged over the batch. `target_rate_hz` is therefore
  genuinely in Hz — but it is a **per-neuron batch-mean rate**, not a total spike
  count.
* The penalty has units **Hz²** (squared rate), *not* spikes. Its natural
  coefficient scale is `1/Hz²`. Concretely, at a 160 Hz rate and a 5 Hz target the
  term is `(155)² = 2.4×10⁴`; with `l2_spikes = 1e-3` that is **≈24**, i.e. ~20×
  the cross-entropy (≈1.3) — the regulariser dominates and suppresses activity.
  This is why `1e-3` produced severe underfitting (see §5).
* Equivalent, reporting-only conversion used throughout the sweep CSV:
  `target_spikes_per_example = target_rate_hz × duration_s`
  (for 10 Hz and a 1.4 s window, 14 spikes/sample/neuron).
* We do **not** rename the target: it represents Hz, correctly.

## 3. Data processing and window / padding

* Events → dense counts per `(t, channel)` by `floor(t_ms/Δt)`; events outside the
  window are dropped. Multi-event bins are ≈1×10⁻⁵ of bins (negligible).
* Measured SHD utterance **span** (train file, n = 8156): mean **716 ms**, median
  705 ms, p90 899 ms, p99 1093 ms, max 1369 ms.
* Therefore the historical window **700 × 2 ms = 1400 ms is ~49 % padding**
  on average. Candidate windows:

| window | coverage of utterances | mean padding |
| ------ | ---------------------- | ------------ |
| 700 × 2 ms = 1400 ms (historical) | 100.0 % | **48.9 %** |
| 500 × 2 ms = 1000 ms | 96.7 % | 28.4 % |
| 300 × 4 ms = 1200 ms | 99.9 % | 40.4 % |
| 250 × 4 ms = 1000 ms | 96.7 % | 28.4 % |

* Splits (speaker-aware, `split_seed = 0`, official test untouched):
  FIT = speakers 9,3,7,11,0,10,1 (n = 5922); DEV = speaker 2 (998);
  PROBE = speakers 6,8 (1236); TEST = official SHD (2264, a mixture — see
  `AUDIT_DYNAMICS.md` §6).

## 4. Sweep parameters

Runner: `scripts/run_lif_sweep.py` (tqdm; writes `results/lif_baseline_sweep*.csv`
and per-run history). Selection: **DEV accuracy + circuit health only**.

Grids: `l2_spikes` (0 … 1e-3), `target_strength` (target × strength),
`readout` (mean / last / sum), `window` (the four windows above). Architecture is
fixed (256 hidden, dense recurrent, signed weights, `neuron_param_mode=bias`).

---

## 5. Results — spike-regularisation sweep

`results/lif_baseline_sweep_l2.csv` (8 epochs, architecture and window fixed,
**readout = sum**, DEV-selected; PROBE/TEST withheld):

| `l2_spikes` | FIT acc | DEV acc | rate mean (Hz) | >200 Hz | median class recall | classes < 0.1 |
| ----------- | ------- | ------- | -------------- | ------- | ------------------- | ------------- |
| 0        | 0.877 | 0.315 | 5.1 | 0.000 | 0.191 | 8 |
| 1e-6     | 0.874 | 0.324 | 5.1 | 0.000 | 0.208 | 9 |
| 3e-6     | 0.851 | 0.321 | 5.1 | 0.000 | 0.162 | 8 |
| 1e-5     | 0.851 | 0.322 | 5.1 | 0.000 | 0.172 | 8 |
| 3e-5     | 0.877 | 0.319 | 5.1 | 0.000 | 0.181 | 9 |
| 1e-4     | 0.878 | 0.325 | 5.1 | 0.000 | 0.220 | 9 |
| 3e-4     | 0.876 | 0.328 | 5.0 | 0.000 | 0.191 | 9 |
| 1e-3     | 0.879 | 0.327 | 5.0 | 0.000 | 0.190 | 9 |

**Finding (important): the spike penalty is inert once the readout is fixed.**
Rates are ~5 Hz at *every* λ, including λ = 0, and DEV is statistically flat
(0.315–0.328). The reason is exactly the audit of §2: the penalty is
`λ·mean_h((rate_h − 10 Hz)²)`, and with the healthy rate already *below* the target,
the term is small and its gradient is weak compared with the cross-entropy. The
pathological 160 Hz firing of the historical model was **not** caused by a missing
spike penalty (§6) — it was caused by the readout scale. The earlier choice
`l2_spikes = 1e-3` only appeared to be the fix because it was fighting a symptom.

**Training-duration evidence (same runs, per-epoch):** FIT rises monotonically
0.15 → 0.43 → 0.58 → 0.69 → 0.76 → 0.80 → 0.83 → 0.86 and is **still rising
strongly at epoch 8**; DEV rises 0.15 → 0.22 → 0.25 → 0.28 → 0.30 → 0.31 → 0.31 →
0.32 and is still rising at the last epoch (`best_epoch = 7` for 6/8 runs). The model
is therefore **undertrained**, not over-regularised. A 40-epoch confirmation with DEV
early stopping is reported in §8.

## 5.1 Regularisation beyond the spike penalty — weight decay

`results/lif_baseline_sweep_wd.csv` (AdamW, 12 epochs, readout = sum, DEV-selected):

| `weight_decay` | FIT | DEV | rate mean | median class recall | classes < 0.1 |
| -------------- | --- | --- | --------- | ------------------- | ------------- |
| 0    | 0.925 | 0.340 | 5.2 Hz | 0.192 | 9 |
| 1e-4 | 0.927 | 0.329 | 5.2 Hz | 0.172 | 9 |
| 1e-3 | 0.924 | 0.329 | 5.2 Hz | 0.181 | 9 |
| 1e-2 | 0.923 | 0.334 | 5.2 Hz | 0.167 | 9 |

**Finding: weight decay is also inert.** FIT stays ≈0.925 and DEV ≈0.33 at every
value including 1e-2. The FIT/DEV gap is therefore **not** classical
weight-magnitude overfitting — it is **speaker-distribution shift**: the model fits
the seven FIT speakers (FIT 0.92) and does not transfer to the unseen DEV speaker
(DEV 0.33). Neither a rate penalty nor weight decay can close a distribution-shift
gap, which is consistent with published SHD results where the large gains
(71.4 % → 83.2 %) come from **augmentation / noise** rather than regularisation — a
lever that is deliberately *outside* the plain recurrent LIF baseline.

## 6. Results — readout aggregation

`results/lif_baseline_sweep_smoke.csv`, 1 epoch each (exploratory check; a
matched-epoch confirmation is in §8):

| readout | FIT | DEV | rate mean | classes < 0.1 |
| ------- | --- | --- | --------- | ------------- |
| `mean` (time-averaged, historical) | 0.096 | 0.051 | 11.7 Hz | 19 |
| `last` (final state) | 0.053 | 0.049 | 10.0 Hz | 19 |
| **`sum` (accumulated)** | **0.323** | **0.144** | 4.9 Hz | 12 |

With `ρ = 0` the readout is linear, so `sum = T × mean` **exactly**
(`tests/test_readout.py`); the two differ only by the effective optimisation scale of
the readout. `last` discards almost all temporal information and is at chance at this
stage.

**Mechanism.** With the time-averaged readout, `logits ≈ (1/T)·Σ s·W_out`, so the
logit scale is ~T (700×) smaller than the natural accumulated readout. Cross-entropy
with tiny logits has large gradients, and the cheapest way to enlarge them is to
raise the hidden firing rate — which is exactly the 160 Hz saturation of the previous
baseline. The accumulated readout removes that incentive, so the hidden layer settles
at ~5 Hz **with no spike penalty at all**. A secondary consequence: because zero-padding
contributes no spikes, `sum` is also immune to the ~49 % padding that dilutes `mean`.

> This is a **readout-scale / optimisation defect**, not a missing-regulariser
> problem, and it is the most consequential finding of this stage.

## 7. Results — window / time resolution

`results/lif_baseline_sweep_window.csv` (readout = sum, 12 epochs, DEV-selected):

| window | bins × ms | FIT | DEV | rate mean | median class recall | classes < 0.1 | wall time |
| ------ | --------- | --- | --- | --------- | ------------------- | ------------- | --------- |
| 1400 ms (historical) | 700 × 2 | 0.925 | **0.340** | 5.2 Hz | 0.192 | 9 | 419 s |
| 1000 ms | 500 × 2 | 0.932 | 0.313 | 6.7 Hz | 0.121 | 9 | 349 s |
| 1200 ms | 300 × 4 | 0.931 | 0.332 | 6.3 Hz | 0.183 | 9 | 283 s |
| 1000 ms | 250 × 4 | 0.877 | 0.321 | 6.5 Hz | 0.181 | 8 | 242 s |

**Finding: the window is a weak accuracy lever.** DEV varies by only 0.03 across a
1400 → 1000 ms change in coverage and a 2 ms → 4 ms change in resolution, and the
historical 1400 ms window is (marginally) the best. This is expected once the
**accumulated** readout is used: zero-padding contributes no spikes, so it does not
dilute the readout the way it does for the time-averaged readout. The practical value
of a shorter window is **compute** (4 ms × 300 bins is ≈1.5× faster than 700 × 2 ms
at essentially the same DEV), not accuracy. We therefore keep the 1400 ms window for
the selected baseline (maximum event coverage) and document 4 ms × 300 as the
efficiency option.

## 8. DEV-based selection and one locked TEST evaluation

### 8.1 Selection (DEV + circuit health only)

Candidates considered (PROBE/TEST never inspected):

| candidate | DEV acc | rate mean | >200 Hz | median class recall | classes < 0.1 |
| --------- | ------- | --------- | ------- | ------------------- | ------------- |
| `mean` readout, λ=0 (historical) | saturated 160 Hz | 0.281 | — | — | — |
| `sum`, λ ∈ {0…1e-3}, 8 ep | 0.315–0.328 | 5.1 Hz | 0.000 | 0.16–0.22 | 8–9 |
| `sum`, wd ∈ {0…1e-2}, 12 ep | 0.329–0.340 | 5.2 Hz | 0.000 | 0.17–0.19 | 9 |
| `sum`, windows 1000–1400 ms, 12 ep | 0.313–0.340 | 5.2–6.7 Hz | 0.000 | 0.12–0.19 | 8–9 |
| **`sum`, λ=0, 1400 ms, 40-ep schedule, DEV early stop (20 ep, best 11)** | **0.374** | **5.0 Hz** | **0.000** | **0.302** | **7** |

**Chosen baseline:** accumulated readout (`model.readout_mode = sum`), **no spike
penalty** (`l2_spikes = 0`), Adam, lr 1e-3, cosine over a 40-epoch budget with DEV
early stopping (patience 8, stopped at epoch 20), 700 bins × 2 ms (1400 ms), 256
hidden. It has the best DEV accuracy and the best circuit health; it was **not**
chosen for any neuron-geometry property.

### 8.2 One locked TEST evaluation (`checkpoints/sweep_l2_0.pt`)

Split accuracies (official TEST read once, after selection):

| split | n | accuracy | hidden mean rate |
| ----- | - | -------- | ---------------- |
| FIT | 5922 | 0.9215 | 5.0 Hz |
| DEV (speaker 2) | 998 | 0.3737 | 4.1 Hz |
| PROBE (speakers 6,8) | 1236 | 0.6853 | 4.8 Hz |
| **TEST (official, locked)** | 2264 | **0.5998** | 5.1 Hz |

Speaker regimes on the official TEST set:

| regime | definition | n | accuracy |
| ------ | ---------- | - | -------- |
| A seen-speaker | test speaker ∈ FIT | 308 | **0.8312** |
| B held-out-speaker | test speaker ∉ FIT | 1956 | **0.5634** |
| C official test | all | 2264 | **0.5998** |

Per-class recall on TEST: **median 0.558, minimum 0.214, and no class below 0.1**
(worst: neun 0.214, five 0.336, three 0.353, vier 0.357, one 0.361). Mean
top1−top2 logit margin 4.58.

Circuit health (measured on FIT): hidden rate mean **5.2 Hz**, median 3.9 Hz, max
46.5 Hz, **0 % > 200 Hz**, 11.7 % < 1 Hz; V p1/p50/p99 = **−4.95 / −0.03 / +0.88**
(θ = 1), max |V| 12.5; `i_rec` p1/p99 = −0.91 / +0.73 (recurrence no longer
dominates); pooled V mean −0.256, std 0.98.

### 8.3 Comparison with the previous baselines

| baseline | TEST acc. | min class recall | classes < 0.1 | rate mean | >200 Hz | V p1 / p99 | logit margin |
| -------- | --------- | ---------------- | ------------- | --------- | ------- | ---------- | ------------ |
| `baseline_v2` (saturated) | 0.5875 | 0.000 | 1 | 166.4 Hz | 0.281 | −14.15 / +15.19 | 0.53 |
| `baseline_repaired_v1` (mean + homeostasis) | 0.4271 | 0.008 | 3 | 21.2 Hz | 0.000 | −20.22 / +0.98 | 0.26 |
| **`sweep_l2_0` (selected)** | **0.5998** | **0.214** | **0** | **5.2 Hz** | **0.000** | **−4.95 / +0.88** | **4.58** |

The selected baseline improves **accuracy** (+1.2 pts over the saturated model and
+17 pts over the homeostasis-only repair), **removes class collapse** on TEST
(minimum recall 0.214, none below 0.1), **enlarges the class margins ~9×**, and puts
the circuit in a physiological regime (5 Hz, 0 % hyperactive) with tight subthreshold
voltages (p99 ≈ θ; the large negative excursions of the repaired model, p1 = −20, are
gone because the readout fix removes the need for large recurrent drive).

## 9. Final quality gate and remaining limitations

| gate | status |
| ---- | ------ |
| A. equations verified | ✅ `AUDIT_DYNAMICS.md` §1–2; numerics re-checked here (§1) |
| B. preprocessing verified | ✅ event-count binning, window coverage and splits documented (§3) |
| C. FIT/DEV indicate learning | ✅ FIT 0.15→0.92 monotone; DEV 0.15→0.37 |
| D. class collapse substantially reduced | ✅ TEST min recall 0.214, **0** classes < 0.1 (was 1 at 0.000) |
| E. defensible activity regime | ✅ 5.2 Hz mean, 0 % > 200 Hz, V p99 ≈ θ |
| F. performance vs ordinary recurrent SHD baselines | ⚠️ **partial** — TEST 0.5998 vs the published **71.4 %** plain recurrent SNN |
| G. no test/probe leakage | ✅ DEV-only selection; leakage guard passes; TEST read once |

**Why the gap to the 71.4 % plain recurrent baseline (documented, not silently
accepted):**

1. **No augmentation / noise.** The published SHD table separates 71.4 % (plain
   recurrent) from **83.2 % (recurrent + augmentation/noise)**. Augmentation is
   deliberately outside the "plain recurrent LIF" baseline requested here, so the
   achievable ceiling is the 71.4 % row and the largest single missing lever is
   absent by design.
2. **Training budget.** The 40-epoch schedule early-stopped at epoch 20 (best DEV at
   epoch 11) because *DEV alone* plateaued; FIT was still improving. Typical SHD SNNs
   train for 100+ epochs with tuned schedules, which is outside this stage's scope.
3. **Model class.** One recurrent layer of 256 LIF units with a linear spike-count
   readout and shared time constants is the *plain* baseline; the higher published
   numbers (82.7 % heterogeneous τ, 84.4 % adaptation, >90 % adaptive/delay) use
   mechanisms this stage is explicitly not allowed to add.
4. **DEV is a single, anomalously hard speaker.** DEV accuracy (0.374) is far below
   PROBE (0.685) and TEST (0.600) — the same model. Speaker 2 is much harder than
   speakers 6/8/4/5, and DEV has 7 classes with < 0.1 recall while TEST has **none**.
   DEV-only selection is mandated by the brief, but a single-speaker DEV is a noisy
   criterion and the early stop it triggers may be premature for other speakers.
5. **No hyper-parameter optimisation** (lr, batch size, hidden size) was performed,
   by instruction.

**Other limitations.** One seed only (no multi-seed spread for this baseline). The
voltage tails still reach |V| ≈ 12 (and ≈ 22 on the dev split) on a minority of
timesteps because the synaptic DC gain is `1/(1−β) ≈ 3`; p1/p99 are now tight, so
this is cosmetic rather than pathological, but a future principled fix would rescale
`input_weight_scale`/`recurrent_weight_scale` or `τ_syn` (documented in
`AUDIT_DYNAMICS.md` §10). Class collapse is not fully explained: it is speaker- and
language-specific (English digits 2/3/4/7 collapse on DEV but not on TEST), and
investigating it further is a separate task.

> **No neuron-space conclusions are updated here.** The representation/geometry
> analysis has not been re-run on this baseline; that is a later stage.

## 10. Reproducing

```powershell
# regularization sweep (DEV-selected; PROBE/TEST withheld)
uv run python scripts/run_lif_sweep.py --config configs/baseline_repaired.yaml `
    --grid l2_spikes --epochs 8 --override model.readout_mode=sum --out lif_baseline_sweep_l2.csv
# weight decay, window, readout grids
uv run python scripts/run_lif_sweep.py --config configs/baseline_repaired.yaml --grid weight_decay --only wd_0,wd_0.0001,wd_0.001,wd_0.01 --epochs 12 --override model.readout_mode=sum --out lif_baseline_sweep_wd.csv
uv run python scripts/run_lif_sweep.py --config configs/baseline_repaired.yaml --grid window --epochs 12 --override model.readout_mode=sum --out lif_baseline_sweep_window.csv
# selected baseline (saves checkpoints/sweep_l2_0.pt)
uv run python scripts/run_lif_sweep.py --config configs/baseline_repaired.yaml --grid l2_spikes --only l2_0 --epochs 40 --early-stopping 8 --override train.epochs=40 --override model.readout_mode=sum --out lif_baseline_sweep_selected.csv
# the single locked TEST evaluation
uv run python scripts/evaluate.py --config configs/analysis.yaml --override run.tag=sweep_l2_0
uv run python scripts/diagnose.py --config configs/analysis.yaml --override run.tag=sweep_l2_0
```

All rows are merged into `results/lif_baseline_sweep.csv`.