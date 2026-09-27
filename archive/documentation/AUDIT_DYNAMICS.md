# AUDIT — LIF dynamics, firing rates, class collapse, splits, metric consistency

Date: 2026-09-26. Scope: the **numerical health of the LIF implementation** and the
**analysis definitions** that depend on it. Companion to `AUDIT.md` (whole repo),
`AUDIT_REPRESENTATION.md` (neuron representation).

Method: read the actual source (`src/model.py`, `src/data.py`, `src/evaluation.py`,
`src/training.py`), then **measure** the network with a new diagnostics module
(`src/diagnostics.py`, `scripts/diagnose.py`) on real SHD data. Nothing here is
assumed; every number is reproducible from
`results/<tag>_diagnostics.{json,npz}`.

Legend: ✅ correct · ⚠️ caveat · ❌ bug (fixed) · 🔍 finding.

---

## 0. Verdict

| Question | Verdict |
| -------- | ------- |
| Are the LIF equations correct? | ✅ mathematically correct and stable |
| Are units / dt / time constants correct? | ✅ verified |
| Is the recurrent current counted once? | ✅ yes (`s_{t-1}` only) |
| Is the firing-rate computation correct? | ✅ correct units; ⚠️ minor batch-weighting bias |
| Is binning of repeated events correct? | ✅ counts are correct; repeats ≈ 1e-5 of bins |
| Why is V ≈ [-39, +49] with θ = 1? | 🔍 **operating-point/homeostasis problem, not a bug**: cross-entropy with no homeostatic term drives a common-mode depolarisation; θ is correct |
| Why is the rate ≈ 160 Hz? | 🔍 same cause (saturation), verified as spikes/timestep first |
| Is class collapse an imbalance/preprocessing bug? | ❌ no — classes are balanced and class 1 is not atypical; it is a consequence of the saturated hidden layer |
| Are the splits well defined? | ⚠️ yes, but the *test* set is a mixture → three explicit evaluation regimes added |
| Is the "primary" metric consistent? | ❌ no — fixed: one canonical primary fingerprint now enforced |
| Config override bug | ❌ `--override ...=1e-3` was parsed as a string and crashed training — fixed |

The implementation is **not broken**; the network was run in an **unhealthy
operating point**. The repair is a parameterisation change (enable the *existing*
homeostatic penalty), **not** a new architecture and **not** threshold hacking.

---

## 1. The equations as implemented

Read from `src/model.py::RecurrentLIFSNN.forward` and `src/data.py::events_to_bins`.

**Input.** For sample with events `(t_c, c)`, the dense input at bin `t`, channel
`c` is the **number of events** falling in that bin:

$$x_t[c] = \#\{\text{events on channel } c \text{ with } \lfloor t_{\text{ms}}/\Delta t \rfloor = t\}$$

(`np.add.at` / `index_add_` — i.e. **counts**, not a binary indicator).

**Synaptic current (current-based synapse, zero-order-hold exact decay):**

$$I_t = \beta I_{t-1} + \big(W_{\text{in}}^\top x_t\big)_h + \big(W_{\text{rec}}\,s_{t-1}\big)_h + b_{\text{hid}}, \qquad \beta = e^{-\Delta t/\tau_{\text{syn}}}$$

with `w_rec[i, j]` = weight **from `j` to `i`** (verified against `s_prev @ w_rec.t()`),
so the recurrent term is the weights **onto** neuron `h` and is counted **once**
(previous step's spikes only).

**Membrane potential (exact homogeneous decay, current held over Δt):**

$$V_t = \alpha V_{t-1} + (1-\alpha)\,I_t, \qquad \alpha = e^{-\Delta t/\tau_{\text{mem}}}$$

**Threshold / spike (surrogate):**

$$s_t = \Theta(V_t - \vartheta), \qquad \vartheta = 1.0$$

forward value is exactly the Heaviside; the backward pass uses the SuperSpike
fast-sigmoid derivative `γ/(1+β|u|)²` with `u = V_t − ϑ`.

**Reset** (spike indicator detached from autograd — no second gradient path):

$$V_t \leftarrow V_t - \vartheta\,s_t \ (\text{subtract}) \quad\text{or}\quad V_t(1-s_t)\ (\text{zero})$$

**Readout** (linear leaky integrator, `ρ = readout_leak`):

$$O_t = \rho O_{t-1} + W_{\text{out}}^\top s_t + b_{\text{out}}, \qquad \text{logits} = \frac{1}{T}\sum_{t=1}^{T} O_t$$

**Numbers used by the failing run** (`baseline_v2`/`runcheck`): `Δt = 2 ms`,
`τ_mem = 20 ms` → α = 0.9048; `τ_syn = 5 ms` → β = 0.6703, so the synaptic filter has
a **DC gain** `1/(1−β) = 3.03`.

---

## 2. What is correct (and what is merely a caveat)

| Item | Status | Notes |
| ---- | ------ | ----- |
| Synaptic accumulation | ✅ | correct first-order filter; `β` applied to `I_{t-1}` |
| Membrane update | ✅ | stable for α < 1; α is clamped via `tau.clamp(min=bin_ms*1.01)` |
| Recurrent current counted once | ✅ | uses `s_{t-1}` only |
| Reset | ✅ | indicator detached; forward behaviour exact |
| Readout | ✅ | time-averaged; `ρ=0` ⇒ logits are linear in mean rates |
| Time constants / dt | ✅ | ms throughout; `α,β` from `Δt/τ` |
| Binning of repeated events | ✅ | counts; measured fraction of multi-event bins = **1e-5** (negligible) |
| Firing-rate units | ✅ | `Hz = spikes\_per\_timestep / (bin\_ms/1000)`; verified numerically |
| Docstring wording | ⚠️ | docstring says α=exp(−Δt/τ) is "implicit Euler"; it is the exact ZOH discretisation (implicit Euler would give τ/(τ+Δt)=0.909). Stable either way; wording only |
| Rate aggregation | ⚠️ | `evaluate_model` and the train loop average **per-batch means** (last batch weighted equally). `circuit_health` is sample-weighted. Small bias |

---

## 3. The huge membrane potential — root cause

Measured on real SHD data (`scripts/diagnose.py`, first 64 FIT samples):

| Quantity | **Untrained** (init) | **Trained** (`baseline_v2`) |
| -------- | -------------------- | --------------------------- |
| V mean / std | −0.131 / **0.718** | −0.500 / **5.042** |
| V p1 / p99 | −3.43 / 0.90 | −14.15 / 15.19 |
| V min / max | — / 2.75 | ≈ −38.7 / **+49.4** (pooled all data) |
| `\|V\|max / θ` | ~2.8 | **41.9** |
| i_syn mean | 0.033 | **2.950** |
| i_in mean | 0.011 | 0.267 |
| i_rec mean | −0.003 | 0.714 |
| spikes/timestep | 0.016 | **0.333** |
| mean rate | **8.0 Hz** | **166.4 Hz** |

**The untrained network is healthy** (σ_V ≈ 0.72·θ, 8 Hz). The pathology appears
**during training**. So this is *not* an initialisation-scaling bug.

**Decomposition of the current (trained):**

* `i_in` mean = (events/bin ≈ 11.8) × mean(`w_in`) + `b_hid`
  = 11.8 × **0.0234** + 0.008 = **0.284** (measured 0.267). The *mean* of `w_in`
  became **positive**.
* `i_rec` mean = spike_prob (0.333) × `n_hidden` (256) × mean(`w_rec`)
  = 85 × **0.0076** = **0.646** (measured 0.714). The *mean* of `w_rec` became
  **positive**.
* Synaptic DC gain: `i_syn ≈ (i_in + i_rec)/(1−β) = 0.98/0.330 = 2.97` (measured
  **2.95**). ✅ The filter amplification is exactly as designed.

**Interpretation.** The weight *variances* barely changed from initialisation
(`w_in` std 0.218 → 0.232; `w_rec` std 0.125 → 0.136), but their **means drifted
positive**. A positive mean over ~12 input events per bin and ~85 active recurrent
weights produces a **common-mode depolarising current** which, multiplied by the
synaptic DC gain of ~3, drives `i_syn ≈ 3θ` and saturates the layer.

**Why does training do this?** With plain cross-entropy, a stronger readout signal
helps: `logits = mean_t(s) @ W_out`, so raising the hidden firing rate increases the
logit scale and lowers the loss. With `l2_in = l2_rec = 0` (no weight decay) and
`l2_spikes = 0` (**no homeostasis**), nothing penalises the common-mode drift, so the
optimiser walks into saturation. This is a **legitimate consequence of the chosen
objective**, not of the equations.

> ✅ Conclusion: the equations are correct. The failure is a **missing homeostatic
> term / unregularised operating point**. The fix does **not** touch θ.

---

## 4. The very high firing rates

Rates are reported **first as spikes/timestep**, then converted:

$$f = \frac{\text{spikes}}{\text{timestep}} \Big/ \frac{\Delta t}{1000} = \frac{0.3328}{0.002} = 166.4\ \text{Hz}\ ✅$$

Measured per-neuron (`baseline_v2`, FIT):

| | spikes/timestep | Hz |
| --- | --- | --- |
| mean | 0.333 | 166.4 |
| median | — | 163.8 |
| p90 | 0.489 | ~245 |
| max | 0.709 | 354.3 |
| fraction < 1 Hz | — | 0.008 |
| fraction > 100 Hz | — | large |
| fraction > 200 Hz | — | **0.281** |
| `silent_fraction` (rate ≤ 0) | — | 0.000 |

The unit conversion is correct; the number is genuinely huge. Note that
`silent_fraction` counts only **exactly zero** rates, so "0 silent neurons" hid
**2 neurons below 0.5 Hz** — circuit health must not rely on that single criterion.

---

## 5. Class collapse — root cause

Measured (`results/baseline_v2_diagnostics.json`):

* **Class counts are balanced** (~288/class in FIT, 60/class in PROBE, ~110/class in
  TEST) ⇒ ❌ **not** class imbalance.
* **Input energy/duration is not the cause**: class 1 has span 664 ms and 9 146
  events (13 773 ev/s) — mid-range, not atypical. The quietest class (6: 5 306
  events) is not in the worst-recall set ⇒ ❌ not a duration/energy confound.
* **Prediction distribution**: class 1 is essentially **never predicted**
  (PROBE 0/62, TEST 10/108, FIT 28/288), while classes 19/10 are over-predicted
  (TEST ratios 1.77× / 1.98×). Logit margins are small (mean top1−top2 = 0.53,
  mean max logit 2.66).
* **Worst recalls** (TEST): class 1 = 0.000, class 0 = 0.107, class 17 = 0.190,
  class 15 = 0.277, class 7 = 0.358.

**Interpretation.** The hidden layer is saturated (all neurons ~same high rate), so
the rate profile is compressed and dominated by a common component; a linear readout
on such a representation separates classes poorly, and cross-entropy without class
weighting settles on avoiding the most confusable class. The collapse is a
**downstream symptom of the operating point** (§3), not a preprocessing, imbalance
or loss bug. Therefore the fix is the same: keep the hidden layer in a healthy
firing regime — not blind class weighting.

---

## 6. Split semantics (FIT / DEV / PROBE / TEST)

Explicit roles (unchanged data, official test never modified):

| Split | Role | Selection? |
| ----- | ---- | ---------- |
| **FIT** (`train`) | gradient updates | yes (fitting) |
| **DEV** | architecture / hyper-parameter selection | **yes** (checkpoint) |
| **PROBE** | functional-neuron analysis, after the model is fixed | **no** — never used to choose the model |
| **TEST** | final locked benchmark / generalisation | **no** |

Speaker table (`speaker_aware_train_dev_probe`, seed 0):

| Split | Speakers | n |
| ----- | -------- | - |
| FIT | 9, 3, 7, 11, 0, 10, 1 | 5922 |
| DEV | 2 | 998 |
| PROBE | 6, 8 | 1236 |
| TEST (official) | 0–11, incl. **4, 5 not in the train file at all** | 2264 |

The official TEST set is a **mixture**:

| Evaluation regime | Definition | n | accuracy |
| ----------------- | ---------- | - | -------- |
| **A. seen-speaker** | TEST samples whose speaker ∈ FIT | 308 | 0.562 |
| **B. held-out-speaker** | TEST samples whose speaker ∉ FIT (DEV + PROBE + novel 4,5) | 1956 | 0.592 |
| **C. official test** | all TEST samples (locked) | 2264 | 0.588 |

Because B is dominated by the *relatively easy* novel speakers 4,5, TEST accuracy
(0.588) is **not** comparable to PROBE (0.440). The three regimes are now reported
separately; the claim "test accuracy = held-out speakers" is too broad and removed.

---

## 7. One primary functional target (metric consistency)

Previously `run_geometry_analysis` used a 40-d *rate+latency* fingerprint (r ≈ 0.345)
while `run_function_analysis`/`run_stress_test` used a 20-d *rate-only* fingerprint
(r ≈ 0.303) — two different "primary" numbers. This is fixed:

* **PRIMARY** = `tuning` — **20-d class-conditioned firing-rate profile**
  (`class_rate`), `standardize=column`, `normalize_rows=false`,
  `metric=euclidean`, **PROBE** split, all neurons.
* **SECONDARY** = `tuning_with_latency` — rate + first-spike latency.
* **EXPLORATORY** = `temporal` — PSTH + temporal centre/dispersion + latency.

Defined once in `src/functional_fingerprint.py`
(`PRIMARY_FINGERPRINT_PRESET`, `PRIMARY_FINGERPRINT_SETTINGS`,
`primary_fingerprint_config()`), selected by `fingerprint.primary_preset` in the
config, used by **all three** scripts, and **written into every result artifact**
(`primary_fingerprint` block with feature sets, dimension, standardisation,
normalisation, distance metric and split).

## 8. Rate control

The primary rate control remains the **rate-normalized class-tuning
fingerprint** (`class_rate_norm`: the class profile divided by the neuron's mean
rate), with the **rate-matched stratified Mantel** as the second proper control.
The partial Mantel is retained only as a secondary/exploratory statistic.

---

## 9. Fix and the repaired baseline (`baseline_repaired_v1`)

**Fix applied** (`configs/baseline_repaired.yaml`; no new architecture, θ unchanged):

* `train.l2_spikes = 1e-3` — enable the **existing** homeostatic firing-rate penalty
  `l2_spikes · mean((rate_hz − target)²)`.
* `train.target_rate_hz = 10.0`.

Calibration (short 2-epoch runs, target 10 Hz):

| `l2_spikes` | final rate | V std | range |
| ----------- | ---------- | ----- | ----- |
| 0 (old) | ~160 Hz | 5.04 | ≈ ±49 |
| 1e-4 | 30 Hz | 1.83 | [−20.9, 11.3] |
| **1e-3** | **~13 Hz** | **0.91** | [−18.5, 9.6] |
| 1e-2 | 10 Hz | 0.33 | [−7.6, 4.5] |

**Repaired run (20 epochs, seed 0, same split):**

| Split | accuracy | hidden rate |
| ----- | -------- | ----------- |
| FIT | 0.4401 | 19.6 Hz |
| DEV | 0.2415 | 13.9 Hz |
| PROBE | 0.3074 | 17.2 Hz |
| TEST | 0.4271 | 20.2 Hz |
| TEST A seen-speaker | 0.4351 | |
| TEST B held-out-speaker | 0.4259 | |

### 9.1 Are the dynamics healthy? — **YES**

| Quantity | Saturated (`baseline_v2`) | Repaired | Verdict |
| -------- | ------------------------- | -------- | ------- |
| spikes/timestep | 0.333 | **0.042** | ✅ |
| mean / median / max rate | 166 / 164 / 354 Hz | **21.2 / 20.7 / 40.8 Hz** | ✅ |
| fraction > 200 Hz | 0.281 | **0.000** | ✅ |
| fraction < 1 Hz | 0.008 | 0.000 | ✅ |
| `i_syn` mean | +2.95 | −1.17 | ✅ |
| `w_in` mean / `w_rec` mean | +0.0234 / +0.0076 | −0.0225 / −0.0230 | ✅ common mode removed |
| V p50 / p99 | +0.165 / **+15.19** | −0.510 / **+0.98 (≈ θ)** | ✅ bulk subthreshold |
| V p1 (negative tail) | −14.15 | −20.22 | ⚠️ heavy tails remain |

The operating point is now **sparse and subthreshold** (p99(V) ≈ θ, no neuron
above 200 Hz) instead of saturated. **Caveat:** the voltage tails are still large
(|V| up to ~40 on a minority of timesteps) because the synaptic DC gain
`1/(1−β) ≈ 3` makes the current scale ~3θ; fully removing that would require
rescaling the weights or τ_syn — deliberately **not** done here so as not to
combine fixes.

### 9.2 Is class collapse resolved? — **NO (it became broader)**

| | `baseline_v2` | repaired |
| - | ------------- | -------- |
| TEST accuracy | 0.5875 | 0.4271 |
| median class recall | 0.617 | **0.368** |
| classes with recall < 0.1 | 1 | **3** (12: 0.008, 6: 0.028, 10: 0.033) |
| mean top1−top2 logit margin | 0.531 | **0.255** |

Saturation *was* a computational resource: clamping the hidden layer gave ~8×
larger mean rates and hence larger logits. With homeostasis the hidden code is more
stereotyped (margins halve) and the linear readout underfits (FIT 0.44, still rising
at epoch 20). So the dynamics are healthy but the classifier is **weaker and still
collapses on ~3 classes**. This is a training/optimisation limitation (readout
signal ∝ mean rate; per-neuron rate clamping flattens the code), not a dynamics bug.
**Per the brief, no class weighting was added** and no further tuning was attempted.

### 9.3 Is the metric inconsistency resolved? — **YES**

All three scripts now use the single canonical **PRIMARY** fingerprint (20-d
class-conditioned firing-rate profile, `class_rate`; column-standardised, Euclidean,
PROBE split, all neurons), defined once in `src/functional_fingerprint.py` and
recorded in every artifact (`primary_fingerprint` block).

### 9.4 New primary geometry/function result

On the repaired model (PROBE, n = 1236, `n_perm = 2000`, floor 5.0e-4, fingerprint
reliability half-r = 0.986 → ceiling 0.997):

| Statistic | Saturated (`baseline_v2`) | **Repaired** |
| --------- | ------------------------- | ------------ |
| **PRIMARY** Mantel r (20-d class-tuning) | 0.303 | **0.472** (p < 5e-4, z = 14.46, CI [0.394, 0.563]) |
| rate-normalized fingerprint r | 0.186 | **0.399** |
| rate-matched stratified r | 0.225 | **0.279** |
| shuffled-neuron control | 0.048 (p = 0.056) | −0.038 (p = 0.89, null ✅) |
| random representation | 0.008 | −0.022 (null ✅) |
| rate-only representation | 0.457 | **0.492** |
| ridge-CV prediction R² (rep → tuning) | 0.363 | **0.408** |

**Interpretation.** With healthy dynamics the representation↔function relationship
is **stronger and much less rate-dominated** (rate-normalized r 0.186 → 0.399;
rate-matched 0.225 → 0.279), and the negative controls are clean. However, the
**rate-only** representation (a single scalar per neuron) still matches the full
structural representation (0.492 vs 0.472), so the structural blocks are **not yet
shown to beat firing rate** as a functional predictor — the same caveat as before,
now in a healthy regime.