# Science — Vector-Valued Spiking Neurons on SHD

The question: **do vector-valued spiking neurons — a neuron whose internal state
is a $D$-dimensional vector that is itself part of the dynamics — compute more
than scalar neurons, and how much more on the Spiking Heidelberg Digits (SHD)
dataset?**

Everything below was measured in this repository on the official SHD test file
(2264 samples, speakers 4–5 held out of training). `INVESTIGATION.md` is the
ledger of record (every run, dated); this file is the write-up.

## 1. The neuron

Baseline VSN (all equations per neuron $i$, state vector
$z_i(t)\in\mathbb{R}^D$):

$$
h_t = \frac{1}{\sqrt{N}}\sum_{i} s_{i,t}\, z_{i,t}\, W_{\text{emit}},
\qquad
c_{i,t} = \sum_{j} s_{j,t}\, W_{\text{rec}}[j,i]
$$

$$
m_{i,t} = \big(z_{i,t} W_{\text{down}}\big) W_{\text{up}}
\qquad
\text{drive}_{i,t} = a_t + c_{i,t}\, h_t + b_i
$$

$$
z_{i,t+1} = z_{i,t} + \alpha\Big(\tanh\big(m_{i,t} + \text{drive}_{i,t}\big) - z_{i,t}\Big),
\qquad \alpha = \frac{\Delta t}{\tau}
$$

$$
\text{pot}_{i,t} = \langle z_{i,t},\, w^{\text{out}}_i\rangle + \theta_i,
\qquad s_{i,t} = \Theta(\text{pot}_{i,t}) \ \text{(surrogate gradient)}
$$

$$
\hat{y} = \Big(\sum_t s_t\Big) W_{\text{cls}} .
$$

The shared input basis is $a_t = x_t W_{\text{in}}$ with $x_t\in\{0,1\}^{700}$
the 2 ms binned events; $W_{\text{emit}}\in\mathbb{R}^{D\times D}$, the rank-64
mixing $W_{\text{down}}W_{\text{up}}$, recurrent $W_{\text{rec}}\in\mathbb{R}^{N\times N}$,
and a zero-initialised linear readout on spike counts. $N$ neurons, $D=1000$,
$\Delta t = 2$ ms, $T=500$ steps, fp16 state with fp32 masters.

**Why it is a different neuron.** The state update is vector-valued and the
recurrence acts in state space, so the *transmission width equals the state
width*: at $D=1$ the population signal $h_t$ collapses to one number (this is
exactly what the scalar control measures, §4).

## 2. Final model (2026-11-03)

The final model switches on every merged mechanism at once, at $N=100$:

| block | setting | what it adds |
|---|---|---|
| adaptation | `adaptation.enabled` | $b_{i,t+1}=\rho b_{i,t}+s_{i,t}$, threshold $-\;w^{\text{ad}}_i b_{i,t}$ |
| state dynamics | `state_dynamics.enabled` | per-neuron trainable $\alpha_i=\sigma(\text{raw}_i)$, learnable $z_0$ |
| input delays | `input_delays.enabled` | per-channel delays $d_c = 20\,\text{ms}\cdot\sigma(\text{raw}_c)$, frozen at bin resolution |
| delay line | `input_delay_line` copies 2 | two shifted copies of the input, concatenated |
| adLIF | `adlif.enabled` | fast ($\rho_f=0.9$) and slow ($\rho_s=0.99$) currents subtracted pre-spike |
| two-stage | `two_stage.enabled` | inner mixer $z\leftarrow z+\alpha(\tanh(m+\lambda\tanh(m'+u))-z)$, $\lambda\in[0.1,2]$ |
| surrogate | `surrogate=slayer`, $\alpha_s=2$ | SLAYER $1/(1+\alpha_s|u|)^2$ instead of fast sigmoid |
| silence tail | `timing.silence_tail_ms=100` | 50 zero bins after the 1000 ms stimulus (5 % of $T$) |
| augmentation | dropout 0.10 + same-class blend | raw events before binning, training only |

$N=100$, $D=1000$, 30 epochs, batch 96, Adam $10^{-3}$ cosine, 3,069,201
parameters.

## 3. Headline numbers

All TEST numbers: official SHD test file, 2264 samples, frozen weights, seed 0.

| model | epochs | fit | val* | TEST | tag |
|---|---|---|---|---|---|
| baseline VSN, $N=64$, $D=1000$ | 20 | 99.9 % | 96.32 % | **1868 = 82.51 %** | `v3_vector_d1000` |
| + learnable $z_0$ & $\alpha$ (M2) | 20 | 100 % | 96.45 % | **1929 = 85.20 %** | `v3_m2_tau_z0_ep20` |
| + event dropout 0.10 only | 20 | 100 % | 97.06 % | **1995 = 88.12 %** | `v3_n100_aug_dropout_ep20` |
| **final model, $N=100$, all blocks** | 30 | 99.78 % | 97.67 % | **1994 = 88.07 %** (best epoch 28: **2009 = 88.74 %**) | `v3_n100_spec_ep30` |

\* VAL is a class-stratified random 10 % of the training file; it shares
speakers with FIT and is **not** a generalisation estimate (§6).

The final-model run is almost stationary over the last epochs (88.74 / 87.94 /
88.07 % on epochs 28–30), so the number is not a knife-edge selection.

## 4. Controls

| control | config | TEST | what it rules out |
|---|---|---|---|
| scalar neuron, size-matched | $D=1$, $N=64$, 6,335 params | 1119 = 49.43 % | state width matters (untuned control, see below) |
| scalar neuron, capacity-matched | $D=1$, $N=1411$, 2,025,488 params | 750 = 33.13 % (5 ep) | capacity alone cannot substitute for width |
| scalar $D=1$, $N=100$ | 5 epochs | *running* | — |
| scalar $D=1$ + broadcast emission width $E$ | `--variant broadcast` | *running* | separates *state* width from *transmission* width |
| V2 LIF reference | $N=256$, T=1400 ms | 1798/2998 = 59.98 % (doc-claim) | not our baseline; context only |

At matched parameter count the gap is +36.9 pp (73 k params: scalar 31.71 %
vs vector $D=64$ 68.60 %) and +43.0 pp (2 M params: scalar 33.13 % vs vector
$D=1000$ 76.15 %), both at 5 epochs — `screening`, baseline 76.15 %.

The scalar controls are **untuned, size-matched ablations**, not competitive
baselines (a tuned recurrent LIF reaches ~71 % on SHD); the repository does not
claim a literature-baseline improvement from the scalar gap.

## 5. What the width is doing (mechanism)

Measured on the frozen 20-epoch baseline checkpoint ($N=64$, $D=1000$):

- **Readout information.** A logistic probe on spike counts reaches 1868 =
  82.51 % (identical to the model's own readout); on the mean state
  (64,000 dims) it reaches 1830 = 80.83 %; on the final state only 674 =
  29.77 %. Δ(mean − counts) = −1.68 pp, 95 % CI [−3.00, −0.31]: the raw state
  does **not** add decodable information beyond the counts; the usable signal
  is the temporal mean, not the endpoint.
- **Effective dimension.** Participation ratio of the population mean state is
  6.64 of 64,000 dimensions (23 components to 90 %, 460 to 99 %). Per neuron,
  median 4.21 of 1,000 (P10–P90 3.05–6.20); scalar control 1.71 of 64.
- **Freeze-dynamics ablations** (only $w_{\text{cls}}$ refit): trained
  checkpoint 82.02 %; untrained network of the same architecture 36.66 % (−45.9
  pp); trained weights with recurrence and emission removed 59.23 % (−23.3 pp).
  The learned dynamics, not the readout capacity, carry the accuracy.
- **Knockout map.** Silencing one neuron costs 1.02 ± 1.07 pp on average
  (max 4.46 pp, one neuron *helps* by 1.24 pp); cumulative top-$k$ silencing
  gives 4.46 / 7.38 / 13.07 / 22.22 / 34.63 / 43.99 / 77.96 pp for
  $k=1,2,4,8,16,32,64$. Necessity is distributed and redundant; mutual
  information per neuron (0.084–0.742 bits, 57/64 effective, none silent) is
  uncorrelated with causal necessity (Spearman +0.20).

These four analyses are being re-run on the final $N=100$ checkpoint.

## 6. Splits, statistics, selection

- FIT 7340 / VAL 816 is a **class-stratified random 10 % split of the training
  file**, so VAL shares speakers with FIT and is optimistic. It is used for
  monitoring and checkpoint selection only.
- The official TEST file (2264 samples) is speaker-disjoint: speakers 4–5 never
  occur in training, so 1840 of 2264 test samples are from unseen speakers.
  Accuracy on those is ~79–80 % vs ~96 % on seen speakers.
- Binomial standard error at n = 2264 is ±0.84 pp; a 5-epoch difference must
  clear ~2 SE to be taken seriously.
- Model selection was done on **5-epoch TEST accuracy** (disclosed), so the
  quoted 88 % carries an unquantified winner's-curse penalty. Single seed.

## 7. What is not claimed

Not claimed, and why (`KNOWN_ISSUES.md`): no multi-seed error bar; no N=100
all-off baseline (a 30-epoch run at 88 % cannot be split between width, dropout
and the other blocks); scalar comparator untuned; VAL is speaker-overlapping;
block set chosen on TEST at 5 epochs. A tuned scalar baseline and a
broadcast-fixed scalar control are in progress.

## 8. Reproducing

```powershell
.venv\Scripts\python.exe -m pytest V3/tests
.venv\Scripts\python.exe V3\run_smoke_test.py
# final model (see README.md for the full override list)
.venv\Scripts\python.exe V3\run_experiment.py --variant vector --override n_neurons=100 --override epochs=30 ...
# mechanism
.venv\Scripts\python.exe V3\run_analysis.py capture --checkpoint <ckpt> --tag final
```

Result directories are git-ignored by convention; every number above is
reproduced in `V3/RESULTS.md` and in the dated ledger `INVESTIGATION.md`.
