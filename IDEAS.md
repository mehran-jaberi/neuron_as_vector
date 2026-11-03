# IDEAS — raw thinking space

Scratchpad for ideas, questions and half-formed arguments that come up in discussion.
This file is **not** a plan, **not** a TODO and **not** a decision log.

Status rules:

- Nothing here is committed to. An entry existing does **not** mean we will do it.
- Nothing here is evidence. Numbers quoted here are either re-derived from code/config
  (say so) or `PI-reported` / `unverified`.
- Promotion path: raw thread here → discussed on paper → either a dated decision in
  `INVESTIGATION.md`, a task in `V3/TODO.md`, or deleted. Never promote silently.
- Append-only, dated, one numbered thread per topic so we can point at them
  ("thread 2") in conversation.

---

## Thread 1 — 2026-10-27 — Where do 2,025,440 parameters come from?

**Question (PI):** "we have 64 neurons with 1000 element vector each, that should be
64000" — why 2 M?

**Resolution.** `64 × 1000 = 64,000` is the **state** `Z(t) ∈ R^{64×1000}`
(`V3/v3/model.py:14`): an activation, no `nn.Parameter`. Parameter budget is a
different object.

Derived from defaults `C=700, N=64, D=1000, R=64, K=20`
(`V3/v3/config.py:381,397-399`), analytic form `V3/v3/config.py:742-758`, cross-checked
against `model.n_parameters()` in `V3/run_smoke_test.py:226`:

| group | shape | count | scales as |
|---|---|---|---|
| `w_emit` | 1000×1000 | 1,000,000 | D², N-free |
| `w_in` | 700×1000 | 700,000 | C·D, N-free |
| `w_down`+`w_up` | 1000×64 + 64×1000 | 128,000 | 2·D·R |
| `gate` | N×D | 64,000 | N·D |
| `bias` | N×D | 64,000 | N·D |
| `w_out` | N×D | 64,000 | N·D |
| `w_rec` | N×N | 4,096 | N² |
| `w_cls` | N×K | 1,280 | N·K |
| `theta` | N | 64 | N |
| **total** | | **2,025,440** | |

Consequences recorded:

- Only three tensors are actually `N×D`-shaped (`gate`, `bias`, `w_out`). Per-neuron
  own cost = 3,001 params (1000 gain, 1000 bias, 1000 readout, 1 threshold) + 64
  recurrent. Total "state-shaped + neuron-indexed" = 196,160 = **9.7 %**.
- The other **90.3 %** is shared substrate independent of neuron count: `w_emit`
  (1 M) and `w_in` (700 k). 64→128 neurons adds only ~196 k; `D` 1000→2000 adds ~1.7 M.
- Scalar `D=1` control is 6,335, of which 4,096 is its own `w_rec` (65 %), and its
  `w_in`+`w_emit` are 701 in total.
- Therefore the ×320 headline gap is essentially a **D² vs D** parameter comparison
  as much as a dimensionality one → this is open confound #1, `INVESTIGATION.md:61-65`.

**Still open.** Whether any of this changes the *claim* we are allowed to make, versus
just how we phrase the control. PI to think on paper.

---

## Thread 2 — 2026-10-27 — Does `w_emit` need to be full rank?

**Idea (from thread 1).** Of the 2 M, 1 M is the single shared `D×D` emission matrix
that maps the emitted population vector back into `R^D` for `h(t)`
(`V3/v3/model.py:29`). It is initialised as `I + 0.02·noise`
(`V3/v3/model.py:137-140`, `emit_identity_init` default `True`,
`V3/v3/config.py:407`), i.e. near-identity, and is freely trainable afterwards.

Options sketched: diagonal emission (per-coordinate gain), rank-`R` emission,
identity-frozen emission, or structured (banded) emission. All keep `h(t) ∈ R^D`,
so the "spikes carry vectors, not scalars" hypothesis is untouched; they only remove
parameters and constrain *how* coordinates mix in the broadcast.

**PI status (2026-10-27):** "yes we can make it diagonal technically" — feasibility
acknowledged. **2026-11-03:** implemented as an engineering variant on branch
`v3.1` (`emission.mode=full|diagonal`, default `full`, commit `fd02e07`);
verified bit-identical in `full` mode and exactly equivalent to a diagonal
emission matrix in fp32; measured 5 epochs, `N=64`, `D=1000`: **diagonal
1748 = 77.21 %** vs baseline **1724 = 76.15 %** at **1,026,440 vs 2,025,440
parameters** (`screening`, single seed). Framed as a parameter-efficiency
improvement of the same model, not a new model; the dense mode stays the
headline because the headline runs used it.

**Open questions.** Is the `D×D` mixing part of the hypothesis or an implementation
convenience? If it is not part of the hypothesis, is it defensible to keep it in the
headline model? What is the matched-parameter control then — do we cut `w_emit`, or
grow the scalar's parameters to match?

---

## Thread 3 — 2026-10-27 — Are we comparing the right scalar control?

Not yet discussed; noted because thread 1 and the `D=1` confound keep pointing here.
The `D=1` control collapses three things at once (state width, population-vector
broadcast, parameter count). A scalar neuron with a *comparable* parameter budget
(`w_in` 700→700·D? `w_emit` 1×D?) would separate "vector state" from "much bigger
model". `PI-reported` that `D=1` is an untuned ablation, not a baseline
(`INVESTIGATION.md:111-116`), so this is about what the control must hold fixed.

**Still open.** Definition of a fair control. No proposal yet.

---

## Thread 4 — 2026-10-29 — Neuron-by-neuron simulation with vector-valued synapses

**Idea (PI).** Simulate the population neuron-by-neuron. 64 neurons, all-to-all. Each
directed pair `(i→j)` owns a 10-dim synaptic code, so connectivity is a tensor
`CON ∈ R^{64×64×10}` (asymmetric). At each step neuron `j` receives the concatenation of
the 10-vectors from the 63 others that spiked → a 630-dim input vector; soma does a dot
product (or elementwise product) of that with a 630-dim internal vector `u_j` to drive a
LIF voltage. On spike, neuron `i` emits a 630-dim output vector (`out_i`), of which the
`j`-th 10-chunk is delivered to `j`; voltage resets. `CON` is "the aggregate of these
synaptic weights". Open question raised by PI: how do the 700 input channels and 20 output
classes fit in.

**Clean mathematical reading.** `CON[i,j,:]` is the reshape of the emitter's chunk, so the
minimal form is two tensors, `out ∈ R^{64×630}` and `u ∈ R^{64×630}` = **80,640 params**
(independent of the diagonal, 640 of the 40,960 pairs). The effective scalar weight is
`w_ij = <CON[i,j,:], u_j[chunk_i]>`. This is the current model's `h(t)` scheme
(`V3/v3/model.py:29`) with the *shared* `W_emit` replaced by a *per-neuron* emitter and
the *scalar* `w_rec` replaced by a *bilinear per-pair* form.

**Three variants hide in the sketch, with different scientific content:**

- **V1 — static code, dot-product soma.** Every `w_ij` is a free scalar (64×64 of them),
  represented by 20 numbers per pair. Model class is *identical to a scalar LIF network
  with delays*; the code is a 20× redundant factorization, i.e. a reparameterization, not
  extra capacity. Gauge freedom: `out_chunk → c·`, `u_chunk → /c` leaves `w_ij` fixed, so
  the loss has flat directions and weight decay chooses the scale. Testable diagnostic:
  the `‖out_chunk‖ / ‖u_chunk‖` ratio over training.
- **V2 — state-dependent code, dot-product soma.** `out_i = f(z_i(t))`. Now `w_ij(t)`
  varies with the source's state. But if the soma state `z_i` is a *scalar* voltage, then
  630 numbers produced from one scalar are activity-dependent *gains* only — the network
  becomes LIF with short-term-plasticity-like per-pair gains, and the 10-dim code is a
  low-rank basis for those gains.
- **V3 — elementwise soma.** The input vector must be *kept*, so the state becomes
  `V_j ∈ R^{630}`. This is the only variant where the state is genuinely vector-valued,
  and its coordinates are indexed by synaptic channel rather than by a free internal
  dimension — "state lives in synaptic space". Needs an explicit learned reduction
  630→1 for the spike threshold, otherwise the vector claim is smuggled in.

**Cost (estimates, not measured).** V1/V2: 80,640 params, ~40 k MACs per sample-step vs
~9.2 M in the current model (`w_down`+`w_up` 8.2 M + `w_emit` 1 M) → order 200× cheaper
per step, which would make the 5-epoch screening loop minutes instead of 204 s/epoch.
V3: state 64×630 = 40,320 per sample, 1.6× *less* than the current 64×1000, but still
needs the checkpointing discipline (`V3/RESULTS.md` §7.0, 8.4 GiB un-checkpointed).

**Input, 700 channels (three options).** (a) *Uniform*: treat the 700 channels as extra
source rows that "fire" with `x_c(t)`; the tensor extends to `764×64×10` and the input
enters through exactly the same machinery — most faithful to the idea. Full cost
700×64×10 = 448,000; with a shared 10-dim code per channel plus a per-neuron 10-dim
gain it collapses to 700×10 + 64×10 = 7,640. (b) Current model's shared basis
`x(t)·W_in` with per-neuron gate. (c) Per-neuron 700-dim input weight (44,800).

**Output, 20 classes (two options).** (a) Spike-count readout `Σ_t s_i(t) W_cls`,
64×20 = 1,280 (what exists). (b) *Uniform*: the 20 classes become 20 non-spiking
integrator "targets" that receive through the same 630-dim code (20×630 = 12,600) and
their logit is the terminal voltage — keeps the invariant "all communication is a vector
code", costs almost nothing.

**Why this might matter more than it looks.** The vector lives in a place that costs
10×-per-pair instead of 1000×-per-neuron: 80 k params vs 2.03 M. That makes a
matched-parameter fight against a scalar LIF *possible* for the first time (thread 3),
instead of the current 6,335-vs-2,025,440 mismatch.

**Still open (blocking the design).** Is `out_i` a constant, or a function of the
neuron's own state? Is the soma state scalar (dot product) or vector (elementwise)? The
answers pick V1/V2/V3, and everything else — control, claim, cost — follows from that.
Also undecided: delays per pair (4,096, non-differentiable) or per source (64); what
initializes `out`/`u` so that `w_ij` does not explode under a 10-dim inner product.

**Status (PI, 2026-10-29):** raw idea under discussion. No design agreed, no code, no run.

### Thread 4 — decisions taken (PI, 2026-10-29)

- Delays: **per pair**, 4,096 of them, **non-differentiable** (fixed).
- `u` (internal vector) is a **static learned constant**, not a dynamical variable: trained
  and then left alone.
- The interaction that determines *what and how* neurons communicate is the pairing
  `CON[i,j,:]` with `CON[j,i,:]` — i.e. the effective coupling is built from the
  elementwise product of the two directions of the same pair.
- Soma: **elementwise** ("to keep the soma feature rich").

**Two structural consequences (derived, worth knowing before coding).**

1. **Half of `CON` is pure gauge.** If only the elementwise product
   `c_ij = CON[i,j,:] ⊙ CON[j,i,:]` ever reaches the dynamics, then scaling
   `CON[i,j,:] → s·` and `CON[j,i,:] → (1/s)·` changes nothing. `40,960` ordered entries
   map onto `2,016` unordered pairs × 10 = `20,160` real numbers, so **20,160 of the
   40,960 connectivity parameters are unidentifiable**, and the diagonal block
   (`64×10 = 640`) is dead weight. Effective model: 64 neurons whose synapses are
   described by 10 numbers each — richer than `w_rec`'s 4,096 scalars, not 10× richer.
   Clean reparameterisation: one 10-vector per *unordered* pair (20,160), no gauge.
2. **Elementwise + a linear readout collapses back to a scalar neuron, exactly.** With
   `V_j(t+1) = (1-α)V_j(t) + α·u_j ⊙ I_j(t)` and any fixed readout `Σ_k p_j[k]V_j[k]`,
   the 630 coordinates are just 630 leaky copies of one scalar, and the potential equals
   a leaky integral of `Σ_i s_i(t-d_ij)·Σ_k (p_j u_j)[k]·CON[i,j,k]·CON[j,i,k]` — i.e.
   plain scalar LIF with a generated `w_ij`. Only `p_j ⊙ u_j` is identifiable, so having
   both `u_j` and a separate readout `p_j` wastes another 40,320. To make the 630 dims
   observable, a coordinate-wise nonlinearity must sit *inside* the state, as the current
   model does (`V3/v3/model.py:39`, `tanh` applied to the vector drive before the leaky
   integration): `V ← (1-α)V + α·tanh(u_j ⊙ I_j)`.

So with the current choices the *neuron* stays scalar and the vector lives in the
**synapse**. That is a real hypothesis, just a different one from thread 1's.

---

## Thread 5 — 2026-10-29 — Silence tail after the input ends

**Requirement (PI).** Run extra empty time after the input so that processing can
complete; the tail may not exceed **+10 %** of the sequence.

**Notes.** At 2 ms bins, `α = 0.1` → `τ = dt/α = 20 ms` (`V3/v3/model.py:36`), so
10 % = 50 steps = 100 ms = **5τ** → membrane residual `e^-5 ≈ 0.7 %`. The cap matches the
"5 time constants" rule. But membrane settling ≠ network settling: recurrent loop gain,
not `τ`, sets the tail length, so the rate trace must be checked, not assumed.

**Caveats.** (i) The tail must be present in **training**, not appended at test time — a
post-hoc tail shifts the spike-count logit scale and would be measured as a drop for a
trivial reason. (ii) Spike-count readout sums over all `T`, so the tail adds drift spikes
(18.22 Hz mean rate × 0.1 s ≈ 1.8 spikes/neuron). (iii) Both variants need the same tail
or the comparison is contaminated. (iv) `T: 500 → 550` is +10 % wall time and a recorded
config change, not a free edit.

**Cheap ways to test it before paying for a run.** Read out over a late window only (no
extra compute); or freeze features and re-fit the linear head with/without a tail, which
answers "does settling help at all" in minutes.

**Still open.** Whether the tail is inside the state or only at the readout; acceptance
criterion for "processing completed" (rate decayed to baseline? logit stable?).
