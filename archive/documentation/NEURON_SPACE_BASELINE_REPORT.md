# Neuron-Space Baseline Report

**Question.** Can an individual hidden neuron be represented in a structured
label-free vector space such that distance between neurons predicts similarity in
their function?

**Stage.** The scientific experiment, run on the repaired, DEV-selected recurrent
LIF baseline. This is the **PRIMARY ANALYSIS**. It is *not* called pre-registered.

**Bottom line (one paragraph).** The label-free structural neuron representation
*is* significantly and reproducibly associated with neuron function
(Mantel Spearman $r = 0.174 \pm 0.049$ across seeds 0/1/2, $p \le 5\times10^{-4}$ in
every seed), and the association is not destroyed by removing firing-rate
magnitude. **But it does not outperform the trivial baselines.** A *single scalar*
firing rate per neuron reproduces a stronger association
($r = 0.388 \pm 0.012$) and a *better* out-of-sample prediction
(CV $R^2 = 0.312$ vs $0.179$), and a 12-dimensional label-free **activity**
representation dominates everything (CV $R^2 = 0.711$). The stronger version of the
hypothesis — that a *structural* representation predicts function *beyond* trivial
firing-rate magnitude — is therefore **not supported** for this baseline.

---

## 1. Setup

### 1.1 Baseline (unchanged, as instructed)

The experiment uses the configuration selected on DEV in the previous stage. No
training configuration was modified, and no adaptive neurons, dimensionality
sweeps, FP8, Optuna or notebook were introduced.

| item | value | source |
|---|---|---|
| checkpoint (seed 0) | `checkpoints/sweep_l2_0.pt` | certified DEV-selected baseline |
| checkpoint (seed 1) | `checkpoints/nsb_seed1.pt` | identical recipe, seed 1 |
| checkpoint (seed 2) | `checkpoints/nsb_seed2.pt` | identical recipe, seed 2 |
| architecture | 700 input → 256 recurrent LIF → 20 readout | `src/model.py` |
| window | 700 bins × 2 ms = 1400 ms |  |
| readout | accumulated / `sum` |  |
| spike regularisation | none (`l2_spikes = 0`) |  |
| optimiser | Adam, lr 1e-3, cosine schedule |  |
| budget | 40 epochs, DEV early stopping (patience 8) |  |
| data split | fixed (`data.split_seed = 0`) — identical for all seeds |  |

Seeds were re-trained with the *identical* recipe so that seed-to-seed variation
reflects only initialisation/optimisation. Circuit health is comparable across
seeds and healthy in all three:

| seed | epochs run | DEV acc | hidden mean rate | max rate | frac > 200 Hz | silent |
|---|---|---|---|---|---|---|
| 0 | 20 | 0.3737 | 4.98 Hz | 46.7 Hz | 0.000 | 0.000 |
| 1 | 22 | 0.2946 | 4.63 Hz | 26.6 Hz | 0.000 | 0.000 |
| 2 | 30 | 0.2996 | 5.51 Hz | 45.4 Hz | 0.000 | 0.000 |

### 1.2 Splits and the leakage boundary

* **FIT** (`train`, 5922 samples) — where the label-free activity statistics of the
  representation are measured. Never used for the fingerprint.
* **DEV** (speaker 2) — model selection only. Not used by this analysis.
* **PROBE** (speakers 6 & 8, 1236 samples) — where *all* fingerprints are measured.
  Disjoint from FIT and from the model-selection speaker; never used for model
  selection and never used to build the representation.
* **official TEST** (2264 samples) — **never read in this stage.** Not once.

### 1.3 The two objects, and the barrier between them

**Representation (label-free; the object under study).** Per-neuron structural
feature blocks, exactly as already implemented:

| block | content | dim |
|---|---|---|
| `intrinsic` | per-neuron *learned* bias (a generic learned excitability offset — **not** a biophysical parameter) | 1 |
| `input_conn` | permutation-invariant statistics of the input weight vector | 14 |
| `recurrent_in` | statistics of the incoming recurrent row $W_{rec}[j,:]$ (+ relationship features) | 19 |
| `recurrent_out` | statistics of the outgoing recurrent column $W_{rec}[:,j]$ | 14 |
| **`structural` (PRIMARY)** | all of the above | **48** |
| `activity` (label-free) | firing statistics measured on FIT (rate, Fano, temporal spread, …) | 12 |

The primary representation contains **no class labels, no class-conditioned
responses, and no functional fingerprint**, and is built without ever touching
PROBE or TEST. Blocks are z-scored per feature and weighted equally per block
(`weighting: equal`), so no block wins merely by having more features.

**Functional fingerprint (independent measurement; uses labels).** The one
canonical primary target is a **20-dimensional class-conditioned firing-rate
profile measured on PROBE** (`class_rate`, preset `tuning`). Two secondary targets
(`class_rate_norm`, `temporal`) exist but are never substituted for the primary.

**The barrier.** The fingerprint is measured on a different split, is not an input
of the representation, and is never used to select anything.

### 1.4 PRIMARY ANALYSIS statistic

For the 256 hidden neurons we form
$D_{\text{repr}}(i,j)$ (Euclidean distance in the standardised representation
space) and $D_{\text{func}}(i,j)$ (Euclidean distance between class-rate profiles),
and report the **Spearman correlation of the two upper-triangle distance vectors**:

$$
r_{\text{Mantel}} \;=\; \operatorname{Spearman}\!\Big(\{D_{\text{repr}}(i,j)\}_{i<j},\; \{D_{\text{func}}(i,j)\}_{i<j}\Big).
$$

**Permutation framework (audited).** The null relabels **neurons**, not pairs:

$$
r^{(k)} = \operatorname{Spearman}\!\Big(\{D_{\text{repr}}(i,j)\},\; \{D_{\text{func}}(\pi_k(i),\pi_k(j))\}\Big),
\qquad \pi_k \sim \text{Uniform}(S_{256}),
$$

implemented as a single indexing pass (`dy[pair_index[p[i], p[j]]]`), which is
numerically identical to the direct computation. Audited points:

* neuron **pairs are not treated as independent observations** — the permutation
  acts on neurons, and the dependence structure between pairs is preserved;
* the reported $p$-value is one-sided for $r>0$ and is bounded below by the
  resolution floor $1/(N_{\text{perm}}+1)$; when the observation saturates the
  floor this is reported explicitly rather than as an exact value;
* the bootstrap CI resamples **neurons** (with replacement), so pair dependence is
  preserved; duplicated neurons produce zero-distance pairs, so the interval is
  reported as approximate;
* ranks are computed once and reused under permutation (permutation preserves
  ties), so the null is exact for Spearman.

Settings: $N_{\text{perm}} = 10\,000$ (floor $10^{-4}$) for the canonical table and
$2\,000$ for the before-learning / rewired conditions; bootstrap 2000; $n = 256$
neurons, $32\,640$ pairs.

---

## 2. PRIMARY ANALYSIS result

Structural representation (48-d, label-free) vs the 20-d PROBE class-rate profile:

| seed | Mantel $r$ | $p$ | effect-size $z$ | bootstrap 95% CI | null mean | null SD |
|---|---|---|---|---|---|---|
| 0 | **+0.1335** | $5.0\times10^{-4}$ | +3.34 | [+0.060, +0.237] | +0.0002 | 0.0399 |
| 1 | **+0.2280** | $1.0\times10^{-4}$ (floor) | +5.58 | [+0.146, +0.328] | −0.0003 | 0.0409 |
| 2 | **+0.1604** | $2.0\times10^{-4}$ | +4.04 | [+0.080, +0.263] | +0.0002 | 0.0396 |

**Multi-seed:** $r = 0.174 \pm 0.049$, 95% CI (t, df=2) $[0.053, 0.295]$.

The local (kNN) analysis agrees: for $k \in \{3,5,10,20\}$ the representation-space
neighbours of a neuron are significantly **more** functionally similar than under
neuron relabelling ($z = +4.2$ to $+4.6$, $p = 1\times10^{-4}$ at every $k$).

**Fingerprint reliability (descriptive only).** Split-half distance correlation of
the fingerprint on two disjoint class-stratified PROBE halves is
$0.994$–$0.997$ (Spearman–Brown full-length ceiling $0.997$–$0.998$), with the
audit passing (independent splits, same neurons, same classes, same statistic,
same metric, no leakage, no overlap). **Consequence:** the fingerprint is
essentially noise-free, so the modest $r$ is *not* limited by measurement noise.
Reliability is reported as a property of the target, never as evidence for the
hypothesis.

---

## 3. Is it just firing rate?

This is the central question, and it is answered from four independent angles, not
from a partial Mantel.

Multi-seed means ± SD over seeds 0/1/2 (all rows are in
`canonical_results_table.csv`):

| representation | dim | raw $r$ | rate-normalized $r$ | rate-matched $r$ | CV $R^2$ | CV corr |
|---|---|---|---|---|---|---|
| `rate_only` (trivial baseline) | 1 | **0.388 ± 0.012** | 0.199 ± 0.067 | −0.241 ± 0.015 | **0.312 ± 0.032** | 0.554 |
| `activity_only` (label-free) | 12 | **0.455 ± 0.020** | **0.326 ± 0.065** | 0.173 ± 0.025 | **0.711 ± 0.052** | 0.844 |
| `input_conn_only` | 14 | 0.157 ± 0.031 | 0.207 ± 0.038 | 0.209 ± 0.024 | 0.116 ± 0.037 | 0.246 |
| `recurrent_only` | 33 | 0.107 ± 0.034 | 0.062 ± 0.072 | 0.080 ± 0.023 | 0.041 ± 0.025 | 0.200 |
| `intrinsic_only` | 1 | 0.137 ± 0.039 | 0.127 ± 0.047 | 0.150 ± 0.039 | 0.025 ± 0.010 | 0.095 |
| **`structural` (PRIMARY)** | **48** | 0.174 ± 0.049 | 0.174 ± 0.069 | 0.180 ± 0.037 | 0.179 ± 0.027 | 0.399 |
| `structural + activity` | 60 | 0.314 ± 0.040 | 0.270 ± 0.056 | 0.206 ± 0.011 | 0.741 ± 0.030 | 0.862 |
| `random` (control) | 48 | −0.002 ± 0.014 | 0.001 ± 0.015 | −0.011 ± 0.020 | −0.023 ± 0.010 | −0.091 |
| `neuron_shuffle` (control) | 48 | 0.011 ± 0.018 | 0.003 ± 0.009 | 0.000 ± 0.021 | −0.027 ± 0.004 | −0.081 |

(`rate-matched $r$` is a stratified Mantel computed only within strata of similar
firing-rate difference; `rate_only` is degenerate inside a stratum — its distance
is nearly constant there — so its negative value reflects noise, not anti-signal.)

**Verdict.**

1. The structural association (`0.174`) is **weaker than the single-scalar
   firing-rate baseline** (`0.388`) on the raw target.
2. On the rate-normalized target the two are statistically indistinguishable
   (`0.174` vs `0.199`), so structural does **not** add incremental information
   beyond a scalar firing rate.
3. The label-free **activity** block — 12 unsupervised firing statistics — is by
   far the strongest representation (CV $R^2 = 0.711$ vs $0.179$), so the
   informative content is in what the neurons *do*, not in their static structure.
4. Controls behave exactly as they must: `random` and `neuron_shuffle` are
   $\approx 0$ ($|r| < 0.03$) on every axis and their CV $R^2$ is negative.

---

## 4. Does rate normalization preserve the relationship?

Yes for the structural representation, and this is the one non-trivial positive
finding — but it is small.

| target | structural $r$ | $p$ |
|---|---|---|
| raw class-rate profile | 0.174 ± 0.049 | ≤ 5×10⁻⁴ (each seed) |
| **rate-normalized profile** (per-neuron magnitude divided out) | 0.174 ± 0.069 | 1×10⁻⁴ (seed 0) |
| rate-matched stratified Mantel | 0.180 ± 0.037 | 3×10⁻⁴ (seed 0) |

For the structural representation the rate-normalized association is *not lower
than* the raw one (0.174 vs 0.174), whereas for `rate_only` it drops sharply
(0.388 → 0.199). So the structural representation carries a component of
*response-profile shape* information that is not per-neuron firing-rate magnitude —
the shape signal is simply masked in the raw comparison by the much larger
magnitude signal. The rate-matched stratified control, which compares only neuron
pairs with similar firing rates, agrees ($0.180 \pm 0.037$, all seeds $p < 0.001$).

**But** this surviving shape signal is no larger than what a single scalar firing
rate already carries after the same normalization (`rate_only` 0.199), and it is
far smaller than the activity signal (0.326). The partial Mantel (which controls
for firing-rate distance with a linear nuisance model) is reported in the artifacts
but is **secondary and exploratory only** — it is explicitly not used as proof of
rate independence.

---

## 5. Does the representation predict functional fingerprints?

Cross-validated ridge regression across **neurons** (KFold, 5 splits), with **one
shared fold assignment for every representation**, standardisation and ridge
`alpha` fitted inside each training fold only:

| representation | CV $R^2$ | CV Pearson $r$ | CV nRMSE |
|---|---|---|---|
| `rate_only` | 0.312 ± 0.032 | 0.554 ± 0.027 | 0.828 |
| `activity_only` | **0.711 ± 0.052** | 0.844 ± 0.027 | 0.530 |
| `input_conn_only` | 0.116 ± 0.037 | 0.246 ± 0.095 | 0.933 |
| `recurrent_only` | 0.041 ± 0.025 | 0.200 ± 0.052 | 0.979 |
| `intrinsic_only` | 0.025 ± 0.010 | 0.095 ± 0.041 | 0.987 |
| **`structural`** | 0.179 ± 0.027 | 0.399 ± 0.034 | 0.900 |
| `structural + activity` | 0.741 ± 0.030 | 0.861 ± 0.016 | 0.502 |
| `random` | −0.023 ± 0.010 | −0.091 ± 0.048 | 1.011 |
| `neuron_shuffle` | −0.027 ± 0.004 | −0.081 ± 0.011 | 1.013 |

So the structural representation *does* predict the fingerprint well above chance
($R^2 = 0.179$ vs random $-0.023$), but **a one-dimensional firing rate predicts it
almost twice as well** ($R^2 = 0.312$), and the label-free activity block predicts
it four times as well ($R^2 = 0.711$).

---

## 6. Does it survive neuron permutation?

Two distinct tests, both mandatory.

**(a) Permutation invariance of the representation (the hidden-neuron relabeling
test).** The trained network is relabelled with a random permutation of the hidden
units, applied consistently to the input columns, both recurrent axes, the readout
rows and the per-neuron parameters (and, after the fix below, the structural
`self_mask` buffer). Recomputing the representation, the same *functional* neuron
must receive the same representation:

| seed | structural invariance | sensitive features |
|---|---|---|
| 0 | **passed** | none |
| 1 | **passed** | none |
| 2 | **passed** | none |

Tolerance for the structural blocks is `atol = 1e-8, rtol = 1e-8` (near
floating-point exact). **The primary representation is exactly invariant to neuron
relabelling in all three seeds**, so it is genuinely a property of the neuron and
not of its array index.

**(b) Activity block invariance — seed-2 discrepancy, diagnosed, not removed.**
The label-free `activity` block is not part of the primary representation
(`primary_blocks` excludes it), but it is a component of
`structural + activity`, so it was tested too. It passed for seeds 0 and 1 and was
flagged for seed 2 (6 of 12 features, max relative difference 0.1 %–4.0 %). The
cause was determined rather than assumed:

* the differences correspond to **416 differing spike entries out of
  91 750 400** ($4.5\times10^{-4}\,\%$) over 512 FIT samples, with a maximum
  per-neuron spike-count difference of **exactly 1 spike**;
* the original and permuted model are functionally identical *up to floating-point
  reassociation*: permuting the summation order changes the membrane potential by
  $\sim10^{-7}$ relative, and because the spike is a hard threshold a handful of
  neurons sitting within that distance of threshold flip;
* all six flagged features are **continuous rate statistics**
  (`rate_hz`, `rate_std_hz`, `log_rate_hz`, `peak_rate_hz`, `fano_factor`,
  `active_bin_fraction`) — none is an index-dependent feature, and no feature that
  could encode the neuron's position was flagged.

This is a numerical-precision artifact of a threshold-crossing spiking network, not
a dependence on neuron identity. It is reported here rather than silenced, and the
statistic on which the study depends (the primary structural representation) is
exactly invariant.

**(c) The shuffle control.** Permuting the representation rows against a fixed
fingerprint destroys the association (`neuron_shuffle` $r = 0.011$, $p = 0.44$;
CV $R^2 = -0.027$), confirming the association is carried by the
representation↔neuron pairing and not by the row ordering.

---

## 7. Does it survive rewiring?

The recurrent matrix was rewired while preserving its weight distribution as far as
possible. Seed 0:

| mode | preserved | destroyed | positions changed | structural raw $r$ | structural rate-norm $r$ |
|---|---|---|---|---|---|
| original | — | — | — | 0.133 | 0.252 |
| `global` | exact weight multiset | all per-neuron marginals **and** all relations | 100 % | 0.098 | 0.200 |
| `rowwise` | multiset **and** every incoming row multiset (`recurrent_in` unchanged) | outgoing marginals and all relations | 100 % | 0.078 | 0.286 |
| `columnwise` | multiset **and** every outgoing column multiset (`recurrent_out` unchanged) | incoming marginals and all relations | 100 % | 0.038 | 0.243 |

Interpretation (honest, and mixed):

* the raw association is **reduced but not eliminated** by destroying the specific
  learned wiring;
* the rate-normalized association is essentially **unchanged** and even rises in the
  `rowwise` case — but note that under rewiring the *fingerprint is recomputed for a
  different network*, so these are associations between two different networks'
  structure and function, not a "loss of signal" decomposition;
* `rowwise` preserves the `recurrent_in` block exactly while the function changes,
  so a surviving `recurrent_in`-driven association cannot be attributed to the
  specific learned relations.

Conclusion: the association is **not** a pure artifact of relational wiring, but the
control also does not show that the learned wiring is necessary — it is consistent
with the association being carried largely by distributional (marginal) weight
statistics plus firing-rate magnitude.

---

## 8. Does learning create or strengthen organization?

Same architecture, same initialisation seed, same PROBE samples; only the weights
differ (untrained initialisation vs trained checkpoint). Seed 0:

| representation | raw $r$ before → after | rate-normalized $r$ before → after | CV $R^2$ before → after |
|---|---|---|---|
| `structural` | **0.019 → 0.133** (+0.115) | **0.078 → 0.252** (+0.175) | 0.259 → 0.162 (**−0.097**) |
| `rate_only` | 0.381 → 0.397 (+0.017) | 0.539 → 0.178 (−0.361) | 0.503 → 0.348 (−0.155) |
| `activity_only` | 0.409 → 0.468 (+0.059) | 0.482 → 0.353 (−0.130) | 0.956 → 0.746 (−0.210) |

**Two-sided answer.**

* **Yes at the geometric level:** learning increases the structural
  representation's distance–distance association with function substantially
  ($\times 7$ on the raw target; $+0.175$ on the rate-normalized target), and this
  is the clearest "learning organises neuron-space" signal in the study.
* **No at the predictive level:** cross-validated $R^2$ from the static
  representation **decreases** after learning for *every* representation, not just
  the structural one. The untrained network's responses are apparently more
  predictable from its (random) weight statistics than the trained network's
  responses are from its learned weight statistics — consistent with learning
  weaving function into dynamics that the static weight summaries do not capture.
  This is reported as an observation; it is not over-interpreted.

---

## 9. Does the result replicate over 3 seeds?

Yes, in direction and significance, with the same qualitative ordering.

| seed | structural raw $r$ | $p$ | $z$ | rate-norm $r$ | rate-matched $r$ | CV $R^2$ |
|---|---|---|---|---|---|---|
| 0 | 0.133 | 5.0×10⁻⁴ | 3.34 | 0.252 (p=1e-4) | 0.142 (p=3e-4) | 0.162 |
| 1 | 0.228 | 1.0×10⁻⁴ (floor) | 5.58 | 0.123 | 0.217 | 0.210 |
| 2 | 0.160 | 2.0×10⁻⁴ | 4.04 | 0.148 | 0.182 | 0.164 |
| **mean ± SD** | **0.174 ± 0.049** | | | **0.174 ± 0.069** | **0.180 ± 0.037** | **0.179 ± 0.027** |

All three seeds are significant at the $10^{-3}$ level or better (seed 1 saturates the
$1/(N_{\text{perm}}+1)=10^{-4}$ resolution floor, reported as $p < 10^{-4}$); the ordering
`rate_only > structural` holds in every seed; `random` and `neuron_shuffle` are
$\approx 0$ in every seed. The data split is identical across seeds, so this is
replication of initialisation/optimisation only. The official TEST set was not
evaluated in this stage.

---

## 10. Answers to the eight questions

| # | Question | Answer |
|---|---|---|
| 1 | Does the representation correlate with function? | **Yes, weakly but significantly and reproducibly.** $r = 0.174 \pm 0.049$, $p \le 5\times10^{-4}$ in all seeds; kNN neighbours are significantly more similar ($z \approx +4.4$). |
| 2 | Is the effect mainly firing rate? | **Yes.** A 1-d firing rate gives $r = 0.388$ and CV $R^2 = 0.312$, both **better** than the 48-d structural representation ($0.174$, $0.179$). 12-d label-free activity dominates ($R^2 = 0.711$). |
| 3 | Does rate-normalization preserve the relationship? | **Yes, for structural** ($0.174$ rate-normalized vs $0.174$ raw; rate-matched $0.180$) — so a *small* shape signal exists — **but it is no stronger than the rate-only baseline's own rate-normalized association** ($0.199$). |
| 4 | Does the representation predict functional fingerprints? | **Yes, above chance** (CV $R^2 = 0.179$ vs random $-0.023$) — **but worse than a single scalar** ($0.312$) and far worse than activity ($0.711$). |
| 5 | Does it survive neuron permutation? | **Yes.** The structural representation is *exactly* invariant to hidden-neuron relabelling (all seeds, zero sensitive features); the shuffle control destroys the effect ($r = 0.011$, $p = 0.44$). The seed-2 *activity*-block flag is a diagnosed floating-point threshold-flip artifact (1 spike per neuron out of 91.75 M entries), not index dependence. |
| 6 | Does it survive rewiring? | **Partially.** Raw $r$ drops (0.133 → 0.038–0.098) but is not eliminated; rate-normalized $r$ is essentially unchanged (0.200–0.286). Not necessary that the specific learned wiring is used. |
| 7 | Does learning strengthen the organization? | **At the geometric level yes** (raw 0.019 → 0.133; rate-normalized 0.078 → 0.252); **at the predictive level no** (CV $R^2$ 0.259 → 0.162, and it drops for every representation). |
| 8 | Does the result replicate over 3 seeds? | **Yes** — consistent sign, $p \le 5\times10^{-4}$, CV $R^2 = 0.179 \pm 0.027$, same qualitative ordering of representations. |

---

## 11. Critical interpretation

* We do **not** conclude "neuron-space works" merely because $r > 0$. The
  comparison that matters is against the trivial baselines, and the structural
  representation **loses** that comparison on both the geometric and the predictive
  axis.
* The one genuinely non-trivial positive is that the structural association is
  **not** carried by firing-rate magnitude alone (it survives rate normalization and
  rate matching), i.e. static structure does encode some response-*shape*
  information. That signal is small and, crucially, **not larger than what one
  scalar firing rate already provides**.
* The strongest signal in the study is the **label-free activity** representation,
  not the static structure. Any future claim of "neuron space" should be made about
  what neurons *do* (activity), not about how they are *wired*, unless a structural
  representation can be shown to add information beyond activity — which is not
  the case here (`structural + activity` $R^2 = 0.741$ vs `activity_only`
  $0.711$; the increment is within one SD and comes with 48 extra dimensions).
* The fingerprint's near-perfect reliability (0.995–0.998) rules out measurement
  noise as the reason for the modest correlations. The ceiling is not the problem;
  the representation is.

**Honest summary of what is and is not established.** Established: (i) a small,
reproducible, significant label-free structural↔function association; (ii) it is
not purely rate magnitude; (iii) it is not a row-ordering or random artifact; (iv)
learning strengthens it geometrically. Not established: that the structural
representation is a *useful* or *superior* description of neuron function. On this
baseline it is dominated by firing rate and by activity, and the central stronger
hypothesis is falsified.

---

## 12. Limitations

1. **Single architecture.** One recurrent LIF baseline (256 hidden, current-based
   synapses, `bias` intrinsic mode, no adaptation, no delays). The `intrinsic` block
   is a generic learned bias, not biophysics; genuine per-neuron dynamical
   parameters are absent, so the "intrinsic" axis is weakly tested.
2. **DEV is one unusually hard speaker.** DEV accuracy (0.37/0.29/0.30) is far below
   PROBE (0.69) for the same models, so DEV-only selection/early stopping is noisy
   and the three seeds are not equally fit. This affects *which* network is analysed
   more than the direction of the conclusion.
3. **Representation metric choices.** Per-feature z-scoring and equal per-block
   weighting are defensible but not unique; a different weighting would move the
   numbers. The control set (`random`, `neuron_shuffle`) and the ordering of
   representations are, however, stable across seeds.
4. **The bootstrap CI** duplicates neurons (a documented limitation of naive neuron
   resampling) and is reported as approximate.
5. **The rewired comparison** necessarily compares two different networks'
   structure *and* function, so "survival" is a statement about the persistence of
   an association across networks rather than a clean ablation of one component.
6. **PROBE is one held-out speaker pair (6, 8)**; the fingerprint is a
   class-conditioned rate profile, so the functional target is itself
   speaker-specific.
7. **Interpretation of the rate-matched control for `rate_only`** is degenerate
   inside strata and must not be read as evidence against firing rate.

---

## 13. Artifacts

Everything is under `results/neuron_space_baseline/`.

| file | content |
|---|---|
| `canonical_results_table.csv` / `.json` | **the ONE canonical table** — one row per (representation, condition, seed) with `representation`, `functional_target`, `distance_metric`, `Mantel_r`, `permutation_p`, `rate_normalized_r`, `CV_R2`, `CV_correlation`, `CV_error`, `seed`, `checkpoint`, `probe_split`, `representation_dimensionality` (plus supporting columns) |
| `summary.json` | everything: per-seed results, the eight-question answers, settings, provenance |
| `multiseed_summary.json` | per-seed / mean / SD / 95% CI of the primary metrics |
| `checkpoints.json` | seed → checkpoint provenance |
| `figure_bundle.json` / `.npz` | all data needed to re-render the figures |
| `seed_<s>/seed_<s>_summary.json` | per-seed representations, fingerprints, distance matrices, geometry + predictive + control statistics, invariance reports, reliability audit |
| `seed_<s>/seed_<s>_arrays.npz` | structural representation matrix, the condensed distance matrices $D_{\text{repr}}$, $D_{\text{func}}$ (raw and rate-normalized), per-neuron rates, the class-rate matrix |
| `seed_<s>/seed_<s>_representations.json` | **per-neuron** structural feature records (every block × neuron) |
| `seed_<s>/seed_<s>_activity_representations.json` | **per-neuron** label-free activity feature records |
| `seed_<s>/seed_<s>_fingerprints.npz` | every fingerprint's raw and standardised matrix + feature names, the class-rate matrix and per-class counts |
| `seed_<s>/seed_<s>_distance_matrices.npz` | **full square** Euclidean distance matrices: `structural_D`, `tuning_D`, `tuning_rate_normalized_D`, `temporal_D` |
| `seed_<s>/seed_<s>_artifacts_index.json` | index/definition of the objects above |
| `figures/figure1…9.*` (png + pdf) | 1 network/representation schematic · 2 PCA of the representation · 3 distance vs function · 4 raw vs rate-normalized · 5 control comparison · 6 kNN similarity · 7 before vs after · 8 rewiring · 9 CV prediction |

PCA (Figure 2) is descriptive only and is never used as evidence. The official
TEST set was **not** read at any point in this stage.

**Related code.** `src/neuron_space_baseline.py` (canonical representations,
PRIMARY ANALYSIS, controls, CV comparison, aggregation, canonical table),
`scripts/run_neuron_space_baseline.py`, `scripts/train_nsb_baseline.py`,
`src/nsb_figures.py`, `configs/neuron_space_baseline.yaml`,
`tests/test_neuron_space_baseline.py`. The analysis reuses the existing
`src/geometry_analysis.py`, `src/functional_fingerprint.py`, `src/representations.py`,
`src/controls.py`, `src/rewiring.py` and `src/prediction.py`.

**Fixes made to the existing analysis code during the audit** (only genuine
bugs/inconsistencies): hidden-neuron permutation now relabels the structural
`self_mask` buffer; the CV predictors accept an explicit shared splitter so all
representations are scored on identical folds; `regression_metrics` now reports
`rmse_mean`/`rmse_median` (previously the reported prediction error was `NaN`); the
kNN effect-size sign convention is documented correctly (positive = neighbours more
similar) and the figure axis was corrected; the "pre-registered" wording was
replaced by **PRIMARY ANALYSIS** in the analysis code paths.

**Tests:** 127 passing, including 9 new tests for the canonical representation set,
the `self_mask` permutation, identical-fold CV, the canonical table schema,
multi-seed aggregation and before/after deltas.