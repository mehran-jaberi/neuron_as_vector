"""V3 vector-valued neuron population.

Concept
-------
A neuron is *not* a scalar unit that is later embedded into a vector.  Neuron
``i`` owns a genuinely ``D``-dimensional dynamical state

.. math::

    \\mathbf z_i(t) \\in \\mathbb R^{D}, \\qquad D = 1000,

and the ``D`` dimensions participate directly in (a) the neuron's own recurrent
dynamics, (b) its interaction with the other neurons, and (c) the generation of
its spike output.  The full population state is ``Z(t) in R^{N x D}``.

Update rule (explicit Euler / leaky integration of the vector state)
--------------------------------------------------------------------
With ``x(t) in R^C`` the binned SHD input, ``a(t) = x(t) W_in in R^D`` the shared
input projection and ``s(t) in {0,1}^N`` the population spike vector:

1. spike generation (read the state, then update it)::

       o_i(t) = <z_i(t), w_out_i> + theta_i          (scalar potential)
       s_i(t) = H(o_i(t))                            (Heaviside, surrogate grad)

2. population interaction -- the *vector state* of the neurons is what is
   communicated; neuron ``j``'s spike emits its own state ``z_j(t)``::

       h(t)   = ( (1/sqrt(N)) sum_j s_j(t) z_j(t) ) W_emit^T   in R^D  (population signal)
       c_i(t) = sum_j W_rec[i,j] s_j(t)                        in R    (scalar recurrent drive)

3. per-neuron vector drive::

       d_i(t) = g_i * a(t) + c_i(t) * h(t) + b_i      in R^D

4. state update (``alpha = dt/tau``)::

       m_i(t) = ( z_i(t) W_down ) W_up^T              in R^D    (shared rank-R mixing)
       z_i(t+1) = (1 - alpha) z_i(t) + alpha * tanh( m_i(t) + d_i(t) )

Optional trainable state dynamics (``V3Config.state_dynamics``, off by default)
replace the scalar ``alpha`` with a per-neuron ``alpha_i = sigmoid(raw_i)``
(``trainable_alpha``) and/or replace ``z(0) = 0`` with a learned ``(N, D)``
initial state (``learnable_z0``).  With both flags off, the update above is
executed unchanged, so the baseline stays bit-identical.
Optional second nonlinear mixing stage (``V3Config.two_stage``, off by default)
------------------------------------------------------------------------------
With ``two_stage.enabled`` each step gains an independent rank-``R2`` mixer and
a second ``tanh`` *before* the state update::

       m2_i(t) = ( z_i(t) W_down2 ) W_up2^T           in R^D    (inner mixer)
       inner   = tanh( m2_i(t) + d_i(t) )
       z_i(t+1) = (1 - alpha) z_i(t) + alpha * tanh( m_i(t) + lam * inner )

``lam`` is not a free scalar.  The optimised quantity is a raw scalar
``lam_raw`` and the gain is
``lam = lam_min + (lam_max - lam_min) * sigmoid(lam_raw)``, warm-started at
``lam = lam_init`` (1.0 by default).  With ``0 < lam_min < lam_init < lam_max``
the gain always lies inside ``(lam_min, lam_max)``: finite and strictly positive
for every finite ``lam_raw``, so training can scale the inner stage but can
never switch its drive off.  This adds depth *inside* one
timestep without cutting the state's connection to its own history: ``z_i(t)``
is still the only recurrent variable, and the outer mixing ``m_i(t)`` and the
drive ``d_i(t)`` are unchanged.  With ``enabled = False`` no parameter is
created and the baseline update above runs unchanged (bit-identical).

Readout (spike counts, 20 SHD classes)::

       logits = ( sum_t s(t) ) W_cls                  in R^20

Learning-matrix emission and R2 readout (``V3Config.learning_matrix``)
-----------------------------------------------------------------------
With the block enabled the model has neither an emission map nor a count
classifier.  The pooled population state is not projected into a learned latent
space -- it *is* the row of the learning matrix, and the classifier reads that
matrix over time::

       L_t          = pooled_t = (1/sqrt(N)) sum_j s_j(t) z_j(t)   in R^D
       logits[b, k] = sum_t a[t, k] < L_t[b, :], v[:, k] >        in R^K

``v`` is ``(D, K)``, drawn ``N(0, 1/sqrt(D))``; ``a`` is ``(T, K)``, initialised
uniformly at ``1/T`` over the timesteps actually executed.  ``W_emit`` and
``W_cls`` are not registered in this mode; the state update, recurrence,
threshold and adaptation are untouched, and the returned tuple is unchanged:
the spike counts are still formed, they simply do not reach the logits.  With
``enabled=False`` (the default) both are registered as before, no extra RNG draw
is consumed and the forward pass is bit-identical.

Learnable input delays
----------------------
With ``cfg.input_delays.enabled`` every input channel ``c`` owns one delay
``d_c`` [ms], stored as an unconstrained parameter ``r_c`` and mapped to
``d_c = max_ms * sigmoid(r_c)``.  That keeps ``d_c`` strictly inside
``(0, max_ms)`` *by construction* -- no clamp sits on the forward path, so the
parameter is never pinned to a boundary where ``clamp``'s derivative is zero
(the reason the first implementation, initialised at exactly 0 ms, never
trained).

The delay is applied as a **fractional** shift, i.e. linear interpolation
between neighbouring bins.  With ``d_c / dt = k_c + f_c`` (``k_c`` integer,
``f_c in [0, 1)``)::

    out[t, c] = (1 - f_c) * x[t - k_c, c] + f_c * x[t - k_c - 1, c]

with zero fill where an index is negative.  ``floor`` carries no gradient, so
``d(out)/d(f_c) = x[t-k_c-1] - x[t-k_c]``: the delay receives the gradient of an
actual time shift rather than the gradient of a constant DC offset (the second
bug of the first implementation).  No ``round`` and no straight-through
estimator is involved.  The shift is applied to the whole sequence before it is
split into the checkpointed segments, so a delayed sample never goes missing at
a segment boundary.
Optional two-timescale adaptation (``adlif.enabled``)
-----------------------------------------------------
With the feature on, every neuron also carries two scalar spike-triggered
currents (one fast, one slow) that subtract from the pre-activation *inside* the
``tanh``::

       u_fast(t) = rho_fast u_fast(t-1) + s(t-1),     rho_fast = 0.9
       u_slow(t) = rho_slow u_slow(t-1) + s(t-1),     rho_slow = 0.99
       pre_i(t)  = m_i(t) + d_i(t) - g_fast,i u_fast,i(t) - g_slow,i u_slow,i(t)

``g_fast`` / ``g_slow`` are one learnable scalar per neuron, initialised at 0:
the mechanism is exactly inert at initialisation but the currents still
accumulate from the spikes, so the gains are not stuck on a zero gradient (see
the M1 history in ``KNOWN_ISSUES.md`` / ``INVESTIGATION.md``).  Both currents
are per-batch state and are zeroed at the start of every forward pass, exactly
like ``z``; the update is threaded through the gradient-checkpointed chunks.
With ``enabled = False`` no current, no parameter and no new code path exists.

Why ``tanh``
------------
The state must stay bounded for fp16 to be numerically safe, and ``tanh`` gives a
hard bound ``|z| <= 1`` with a bounded, well-conditioned gradient.  It is a
consistency device, not part of the hypothesis: the hypothesis is that the state
is ``D``-dimensional and dynamical, which holds for any pointwise nonlinearity.
The nonlinearity is applied to the *vector* ``m_i(t) + d_i(t)``, so it does not
decouple the ``D`` dimensions -- the coupling comes from ``W_down/W_up``, from
the shared input projection ``W_in`` and from the population signal ``h(t)``.

Why the parameterisation is cheap
---------------------------------
The naive ``D x D`` recurrent matrix per neuron would cost ``N*D*D ~ 6.4e7``
parameters per neuron.  Instead:

* ``W_in`` is shared by all neurons (each neuron gates it with its own ``g_i``),
* the intra-neuron mixing ``W_down @ W_up`` has rank ``R`` and is shared,
* the population signal uses one shared ``D x D`` emission ``W_emit``.

The *state* stays genuinely ``D``-dimensional and the mixing genuinely couples
the ``D`` coordinates; only the parameters are shared.
"""

from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .config import V3Config
from .input_delay_line import apply_input_delay_line
from .config import SURROGATE_KINDS, V3Config
from .state_regularization import regularize_state


# ---------------------------------------------------------------------- #
def _autocast_disabled(device_type: str):
    """Context manager that forces full-precision surrogate-gradient math.

    Autocast cannot be switched off on arbitrary devices with this API, so the
    device types we run on get an explicit ``enabled=False`` region and anything
    else (e.g. ``mps``) gets a no-op.  The surrogate gradient is a *backward*
    computation: under ``torch.autocast`` its elementwise ops could otherwise be
    silently downcast, which is exactly the fp16-saturation failure mode that
    made the arctan surrogate diverge (see ``INVESTIGATION.md``).
    """
    if device_type in ("cpu", "cuda"):
        return torch.autocast(device_type=device_type, enabled=False)
    return contextlib.nullcontext()


def surrogate_derivative(
    u: torch.Tensor,
    beta: float = 4.0,
    gamma: float = 1.0,
    *,
    kind: str = "fast_sigmoid",
    alpha: float = 2.0,
) -> torch.Tensor:
    """Derivative of the spike surrogate with respect to the potential ``u``.

    ``u`` is the pre-activation distance from threshold; a spike is emitted for
    ``u > 0``.  Two shapes are supported:

    ``fast_sigmoid`` (default, baseline)
        derivative of ``gamma*u / (1 + beta*|u|)``, i.e.
        ``gamma / (1 + beta*|u|)^2``.  Bit-identical to the historical
        implementation (same operands, same order).
    ``slayer`` (M11a)
        ``1 / (1 + alpha*|u|)^2``, the SLAYER fast-sigmoid derivative.
        ``alpha`` defaults to ``2.0`` (M11a robustness fix): the derivative is
        then at most ``(beta/alpha)^2 = 4`` times the baseline fast-sigmoid
        derivative for every ``|u|`` (asymptotically 4x, 3.77x at ``|u|=8``),
        whereas the old ``alpha=0.5`` width reached 63x in the far field and
        overflowed the fp16 BPTT (pinned by the M11a tests).

    Precision contract
    ------------------
    Reduced-precision inputs (fp16/bf16, the model's state dtype) are promoted
    to **fp32** and the whole derivative is evaluated inside an autocast-disabled
    region, so the surrogate gradient is never computed in half precision and
    never subject to autocast.  fp64 inputs keep fp64 arithmetic, which is what
    the high-precision gradient checks in ``run_smoke_test.py`` assert; fp32
    inputs are computed in fp32.
    """
    kind = str(kind)
    if kind not in SURROGATE_KINDS:
        raise ValueError(f"unknown surrogate {kind!r} (expected one of {SURROGATE_KINDS})")
    if kind == "slayer":
        alpha = float(alpha)
        if not alpha > 0.0:
            raise ValueError("SLAYER surrogate_alpha must be > 0")
    u_det = u.detach()
    u_w = u_det if u_det.dtype in (torch.float32, torch.float64) else u_det.float()
    with _autocast_disabled(u.device.type):
        if kind == "slayer":
            return 1.0 / (1.0 + alpha * u_w.abs()).square()
        return float(gamma) / (1.0 + float(beta) * u_w.abs()).square()


# ---------------------------------------------------------------------- #
class SurrogateSpike(torch.autograd.Function):
    """Heaviside spike with a fast-sigmoid or SLAYER surrogate gradient.

    Explicit autograd Function (rather than a hand-written straight-through
    expression) so that the gradient is exactly the derivative of the intended
    surrogate primitive:

    ``fast_sigmoid`` (default)
        ``d/du [ gamma * u / (1 + beta|u|) ] = gamma / (1 + beta|u|)^2``
    ``slayer`` (M11a)
        ``d/du [ u / (1 + alpha|u|) ] = 1 / (1 + alpha|u|)^2``

    The forward is the exact Heaviside ``1[u > 0]`` in both cases.  Gradients
    are computed by :func:`surrogate_derivative` in fp32 (fp64 for fp64 inputs)
    with autocast explicitly disabled.
    """

    @staticmethod
    def forward(
        ctx,
        u: torch.Tensor,
        beta: float,
        gamma: float,
        kind: str = "fast_sigmoid",
        alpha: float = 2.0,
    ) -> torch.Tensor:
        u_w = u if u.dtype in (torch.float32, torch.float64) else u.float()
        ctx.save_for_backward(u_w)
        ctx.beta = float(beta)
        ctx.gamma = float(gamma)
        ctx.kind = str(kind)
        ctx.alpha = float(alpha)
        return (u_w > 0).to(u.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (u_w,) = ctx.saved_tensors
        deriv = surrogate_derivative(
            u_w, ctx.beta, ctx.gamma, kind=ctx.kind, alpha=ctx.alpha
        )
        return (grad_out.to(deriv.dtype) * deriv).to(grad_out.dtype), None, None, None, None


def surrogate_spike(
    u: torch.Tensor,
    beta: float = 4.0,
    gamma: float = 1.0,
    *,
    kind: str = "fast_sigmoid",
    alpha: float = 2.0,
) -> torch.Tensor:
    """Heaviside spike; ``kind`` selects the surrogate gradient (see M11a)."""
    return SurrogateSpike.apply(u, beta, gamma, kind, alpha)


def _lam_raw_init(lam_init: float, lam_min: float, lam_max: float) -> float:
    """Raw logit whose sigmoid maps back to the warm-start ``lam_init``.

    ``lam_init`` must lie strictly inside ``(lam_min, lam_max)``, which
    ``TwoStageConfig.validate`` enforces; the ratio is still clamped away from
    the open interval's endpoints so an extreme (but finite) setting cannot
    produce an infinite raw parameter.
    """
    p = (float(lam_init) - float(lam_min)) / (float(lam_max) - float(lam_min))
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


# ---------------------------------------------------------------------- #
class VectorNeuronPopulation(nn.Module):
    """``N`` neurons, each with a trainable ``D``-dimensional dynamical state."""

    def __init__(self, cfg: V3Config):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        N = int(cfg.n_neurons)
        D = int(cfg.state_dim)
        C = int(cfg.n_inputs)
        R = int(min(cfg.mix_rank, D))
        K = int(cfg.n_classes)
        self.n_neurons, self.state_dim, self.n_inputs, self.mix_rank, self.n_classes = N, D, C, R, K
        self.state_dtype = cfg.torch_dtype
        # Learning-matrix emission (R2): when on, ``h_t`` is the pooled state
        # itself and the classifier reads the matrix over time, so neither the
        # emission map nor the count classifier is registered (see
        # LearningMatrixConfig).
        lm = cfg.learning_matrix
        self.learning_matrix_enabled = bool(lm.enabled)

        g = torch.Generator().manual_seed(int(cfg.seed) + 12345)

        def rnd(*shape, scale=1.0):
            return (torch.randn(*shape, generator=g) * scale).to(torch.float32)

        # width of the tensor the shared input projection actually sees: K*C
        # when the delay line stacks shifted copies (concat), C otherwise.  The
        # /sqrt(C_in) init rule keeps the *total* input drive at initialisation
        # on today's scale instead of growing by sqrt(K) when K copies are
        # stacked; with the delay line disabled C_in == C, so both the shape and
        # the RNG draw are unchanged (bit-identical init).
        C_in = int(cfg.input_delay_line.input_channels(C))
        self.input_channels = C_in
        self.w_in = nn.Parameter(rnd(C_in, D, scale=float(cfg.input_gain) / math.sqrt(C_in)))  # shared input basis
        self.w_down = nn.Parameter(rnd(D, R, scale=1.0 / math.sqrt(D)))    # shared rank-R mixing
        self.w_up = nn.Parameter(rnd(R, D, scale=0.5 / math.sqrt(R)))      # (init mixing gain ~0.5)
        # The emission tensor is *drawn* in both modes -- one call, one generator
        # draw -- so an enabled build shares every init draw with the baseline
        # build; only the registration is skipped, because R2 has no emission
        # map (h_t := pooled_t).
        if cfg.emit_identity_init:
            emit_init = torch.eye(D, dtype=torch.float32) + rnd(D, D, scale=0.02)
        else:
            emit_init = rnd(D, D, scale=1.0 / math.sqrt(D))
        self.w_emit: nn.Parameter | None = (
            None if lm.enabled else nn.Parameter(emit_init)
        )
        self.w_rec = nn.Parameter(rnd(N, N, scale=1.0 / math.sqrt(N)))
        self.gate = nn.Parameter(torch.ones(N, D, dtype=torch.float32) + rnd(N, D, scale=0.05))
        self.bias = nn.Parameter(torch.zeros(N, D, dtype=torch.float32))
        self.w_out = nn.Parameter(rnd(N, D, scale=1.0 / math.sqrt(D)))
        self.theta = nn.Parameter(torch.zeros(N, dtype=torch.float32))
        # zero-initialised classifier: the initial logits are exactly 0, i.e. the
        # initial cross-entropy is ln(20).  With a spike-count readout the logit
        # scale grows with the firing rate, so a random readout init would start
        # far from the loss basin.  Absent in R2, where the matrix readout has
        # its own (zero-free) initialisation, see below.
        self.w_cls: nn.Parameter | None = (
            None if lm.enabled else nn.Parameter(torch.zeros(N, K, dtype=torch.float32))
        )

        # Spike-frequency adaptation (off by default): one scalar adaptive gain
        # per neuron, read against a decaying spike-count trace.  When the
        # feature is disabled nothing is registered, so the baseline parameter
        # set and the baseline computation are unchanged.
        ad = cfg.adaptation
        if not ad.enabled:
            self.w_ad = None
        else:
            gain = torch.full((N,), float(ad.gain_init), dtype=torch.float32)
            if ad.learnable_gain:
                self.w_ad = nn.Parameter(gain)
            else:
                # fixed gain: a buffer, so the optimiser never sees it
                self.register_buffer("w_ad", gain)

        # ---- optional trainable state dynamics (default: fixed alpha, z(0)=0)
        # The parameters exist only when the feature is in force, so the disabled
        # model keeps exactly the baseline parameter set (same optimiser state,
        # state_dict and registry parameter budget).
        sd = cfg.state_dynamics
        self.raw_alpha: nn.Parameter | None = None   # (N,) leak logits
        self.z0: nn.Parameter | None = None          # (N, D) initial state
        if sd.alpha_trainable:
            # sigmoid(raw) == alpha_init at initialisation
            self.raw_alpha = nn.Parameter(
                torch.full((N,), float(sd.alpha_init_logit()), dtype=torch.float32)
            )
        if sd.z0_learnable:
            self.z0 = nn.Parameter(torch.zeros(N, D, dtype=torch.float32))

        # learnable per-input-channel conduction delays [ms].  The *stored*
        # parameter is the unconstrained pre-sigmoid value ``raw``; the effective
        # delay is ``max_ms * sigmoid(raw)``, so it is strictly inside
        # ``(0, max_ms)`` and never sits on a boundary where the gradient dies.
        # ``None`` when the feature is disabled: the disabled model then has
        # exactly the baseline parameters and the input path is untouched
        # (bit-identical).
        self.input_delay_raw: nn.Parameter | None = None
        if cfg.input_delays.enabled:
            self.input_delay_raw = nn.Parameter(
                self._delay_ms_to_raw(
                    self._initial_input_delays_ms(C), float(cfg.input_delays.max_ms)
                ),
                requires_grad=bool(cfg.input_delays.learnable),
            )

        # two-timescale adaptation currents (adlif.enabled): one learnable scalar
        # gain per neuron per timescale.  They exist only when the feature is on,
        # so the disabled path keeps the baseline parameter list and state_dict
        # exactly.  `gain_init` is the initial *gain* value, not a raw
        # reparameterisation: 0 (the default) makes the subtracted current
        # exactly 0 at t = 0 while the currents themselves still accumulate.
        adlif = cfg.adlif
        self.adlif_enabled = bool(adlif.enabled)
        self.adlif_use_fast = self.adlif_enabled and bool(adlif.use_fast)
        self.adlif_use_slow = self.adlif_enabled and bool(adlif.use_slow)
        self.register_parameter("adlif_gain_fast", None)
        self.register_parameter("adlif_gain_slow", None)
        if self.adlif_use_fast:
            self.adlif_gain_fast = nn.Parameter(
                torch.full((N,), float(adlif.gain_init), dtype=torch.float32)
            )
        if self.adlif_use_slow:
            self.adlif_gain_slow = nn.Parameter(
                torch.full((N,), float(adlif.gain_init), dtype=torch.float32)
            )
        # ---- learnt temporal attention over the readout (M11b) -----------
        # One logit per time bin, zero-initialised so the softmax starts uniform
        # and the initial forward reuses the plain spike-count readout (the
        # readout rescales the uniform weights back to the count scale).  The
        # parameter is created *only* when the block is enabled: the disabled
        # model has the exact same parameter set and forward as before.
        if bool(cfg.attention.enabled):
            self.attn_logits = nn.Parameter(
                torch.zeros(int(cfg.total_steps), dtype=torch.float32)
            )
        else:
            self.attn_logits = None

        # ---- optional second nonlinear mixing stage (M12, off by default) ----
        # Created only when enabled, so the default model keeps exactly the
        # baseline parameter set (state_dict order, optimiser state and the
        # registry parameter budget).  ``w_down2`` uses the *same* scheme and
        # scale as ``w_down``, and ``w_up2`` the same as ``w_up``: a zero or
        # shrunken init would leave the inner stage -- and ``lam`` -- inert at
        # step 0, the failure mode this feature has to avoid.  The gain itself
        # is a sigmoid of ``lam_raw`` (never a free scalar), so it cannot be
        # trained to 0 or below.
        ts = cfg.two_stage
        self.w_down2: nn.Parameter | None = None   # (D, R2) inner down-projection
        self.w_up2: nn.Parameter | None = None     # (R2, D) inner up-projection
        self.lam_raw: nn.Parameter | None = None   # scalar logit of the inner-stage gain
        if ts.active:
            R2 = int(min(int(ts.rank), D))
            self.w_down2 = nn.Parameter(rnd(D, R2, scale=1.0 / math.sqrt(D)))
            self.w_up2 = nn.Parameter(rnd(R2, D, scale=0.5 / math.sqrt(R2)))
            self.lam_raw = nn.Parameter(
                torch.tensor(
                    _lam_raw_init(ts.lam_init, ts.lam_min, ts.lam_max),
                    dtype=torch.float32,
                )
            )

        # ---- learning-matrix emission (R2, off by default) -------------------
        # No emission map and no count classifier: ``h_t := pooled_t`` is the
        # learning-matrix row L_t, and the classifier reads the matrix over time,
        # ``logits[b, k] = sum_t a[t, k] * (L_t[b] @ v[:, k])``.  Both matrices
        # are created only when the block is enabled, so the disabled model keeps
        # exactly the baseline parameter set, state_dict and init stream.  They
        # are drawn *last*, after every baseline draw, so an enabled build also
        # leaves all shared parameters at their baseline values.
        self.v_matrix: nn.Parameter | None = None  # (D, K) learning-matrix basis
        self.a_time: nn.Parameter | None = None    # (T, K) per-bin, per-class weights
        if lm.enabled:
            self.v_matrix = nn.Parameter(rnd(D, K, scale=1.0 / math.sqrt(D)))
            T_steps = int(cfg.total_steps)
            self.a_time = nn.Parameter(
                torch.full((T_steps, K), 1.0 / T_steps, dtype=torch.float32)
            )

        # expose the analytic flag so the class itself never hard-codes D > 1
        self.is_scalar_baseline = D == 1
        # mean-field scaling: a sum of N unit contributions has magnitude ~sqrt(N),
        # so the population signal is divided by sqrt(N) to stay O(1) and to keep the
        # drive (and hence the firing rate) from growing with the number of neurons
        self.pop_scale = 1.0 / math.sqrt(N)
        # phase that selects whether controlled state imprecision is active.
        # None -> infer from training/eval mode.  Callers set it explicitly:
        #   "train" during optimisation, "val" for validation, "test" for the
        #   official test set.  Only training is perturbed by default.
        self.state_reg_phase: str | None = None

    # ------------------------------------------------------------------ #
    def param_groups(self) -> dict:
        return {name: int(p.numel()) for name, p in self.named_parameters()}

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def set_phase(self, phase: str) -> None:
        """Select the phase ("train" | "val" | "test") for state imprecision.

        Entering the ``"train"`` phase is the per-epoch boundary of the training
        loop (``v3.train.train_one_epoch`` calls this once, before the first
        batch of the epoch), so the input-delay parameter is projected into its
        safe pre-sigmoid range here -- see :meth:`clamp_input_delays_`.  The
        effective delay itself is already bounded by construction.
        """
        if phase not in ("train", "val", "test"):
            raise ValueError(f"unknown phase {phase!r}")
        self.state_reg_phase = phase
        if phase == "train":
            self.clamp_input_delays_()

    def _reg_phase(self) -> str:
        if self.state_reg_phase is not None:
            return self.state_reg_phase
        return "train" if self.training else "val"

    def alpha_per_neuron(self) -> torch.Tensor | None:
        """Per-neuron leak ``alpha_i = sigmoid(raw_i)`` in fp32, or ``None``.

        ``None`` whenever ``state_dynamics.trainable_alpha`` is not in force, so
        the state update falls back to the scalar ``dt/tau`` leak of the
        baseline.
        """
        if self.raw_alpha is None:
            return None
        return torch.sigmoid(self.raw_alpha.float())

    def init_state(self, batch: int, device: torch.device) -> torch.Tensor:
        """Initial state ``z(0)``: zeros, or the learned ``(N, D)`` state repeated.

        With ``learnable_z0`` the returned ``(B, N, D)`` tensor is a read-only
        view of the ``z0`` parameter broadcast over the batch, so the gradient of
        every sample flows back into ``z0``.
        """
        if self.z0 is None:
            return torch.zeros(
                batch, self.n_neurons, self.state_dim, device=device, dtype=self.state_dtype
            )
        return (
            self.z0.to(device=device, dtype=self.state_dtype)
            .unsqueeze(0)
            .expand(batch, self.n_neurons, self.state_dim)
        )

    def init_adaptation_state(self, batch: int, device: torch.device) -> torch.Tensor:
        """Initial (zero) adaptation trace ``b`` of shape ``(B, N)``."""
        return torch.zeros(
            batch, self.n_neurons, device=device, dtype=self.state_dtype
        )

    # ------------------------------------------------------------------ #
    # learnable per-input-channel conduction delays
    # ------------------------------------------------------------------ #
    def _initial_input_delays_ms(self, n_channels: int) -> torch.Tensor:
        """Deterministic interior init of the per-channel delays [ms].

        The delays start uniformly inside ``[init_ms, max_ms / 2]`` -- with the
        defaults ``[0.5, 10]`` ms -- i.e. strictly inside ``(0, max_ms)``.  An
        exact boundary (the old 0 ms init) is precisely where ``clamp`` has zero
        derivative and the parameter never trained, so it must not be used.

        A dedicated generator keeps this draw out of the weight-initialisation
        stream: enabling the feature does not perturb any other parameter, so the
        A/B with ``enabled: false`` still differs only by the delay path.
        """
        delays = self.cfg.input_delays
        max_ms = float(delays.max_ms)
        lo = float(delays.init_ms)
        hi = max(lo, 0.5 * max_ms)
        gen = torch.Generator().manual_seed(int(self.cfg.seed) + 54321)
        u = torch.rand(n_channels, generator=gen)
        return (lo + (hi - lo) * u).to(torch.float32)

    @staticmethod
    def _delay_ms_to_raw(delays_ms: torch.Tensor, max_ms: float) -> torch.Tensor:
        """Inverse of ``max_ms * sigmoid(raw)``; keeps ``raw`` finite and interior.

        The ``1e-6`` guard only protects the inverse against a caller passing
        exactly 0 / ``max_ms``; the forward mapping itself is a plain sigmoid,
        with no clamp on the differentiable path.
        """
        p = (delays_ms / float(max_ms)).clamp(1e-6, 1.0 - 1e-6)
        return torch.log(p / (1.0 - p)).to(torch.float32)

    def effective_input_delays_ms(self) -> torch.Tensor:
        """Per-channel delays in ms: ``max_ms * sigmoid(raw)`` (never at a bound)."""
        if self.input_delay_raw is None:
            raise RuntimeError("input_delays.enabled is False for this model")
        return float(self.cfg.input_delays.max_ms) * torch.sigmoid(self.input_delay_raw)

    def input_delay_bins(self) -> torch.Tensor:
        """Per-channel delays in *fractional* bins: ``delay_ms / bin_ms``."""
        return self.effective_input_delays_ms() / float(self.cfg.bin_ms)

    @torch.no_grad()
    def set_input_delays_ms(self, delays_ms) -> None:
        """Write fixed delays given in ms (scalar broadcasts; used by tests)."""
        if self.input_delay_raw is None:
            raise RuntimeError("input_delays.enabled is False for this model")
        d = torch.as_tensor(delays_ms, dtype=torch.float32, device=self.input_delay_raw.device)
        d = torch.broadcast_to(d, self.input_delay_raw.shape)
        self.input_delay_raw.copy_(
            self._delay_ms_to_raw(d, float(self.cfg.input_delays.max_ms))
        )

    @torch.no_grad()
    def clamp_input_delays_(self, limit: float = 10.0) -> int:
        """Project the pre-sigmoid parameter into ``[-limit, limit]``; return #changed.

        The effective delay is bounded by construction (``max_ms * sigmoid``), so
        this is a *numerical* guard, not the thing that enforces ``[0, max_ms]``:
        it stops ``raw`` from drifting so far that fp32 ``sigmoid`` saturates to
        exactly 0 or 1, which would freeze the delay at a bound (same failure
        mode as the original zero init).  ``limit=10`` leaves ``sigmoid`` strictly
        interior (``~4.5e-5`` slope).

        Called once per training epoch from :meth:`set_phase` (the epoch boundary
        of ``v3.train.train_one_epoch``) and once after ``fit`` in
        ``v3.experiment.train_variant``.
        """
        if self.input_delay_raw is None:
            return 0
        limit = float(limit)
        before = self.input_delay_raw.detach().clone()
        self.input_delay_raw.clamp_(-limit, limit)
        return int((self.input_delay_raw.detach() != before).sum())

    def _shift_by_bins(self, x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
        """Integer per-channel shift with zero fill: ``out[t, c] = x[t - k_c, c]``.

        ``bins`` is a long tensor of non-negative bin counts.  A channel with
        ``k_c >= T`` is left silent (all zeros); every other channel reads
        zeros for the first ``k_c`` steps of the window.
        """
        B, T, C = x.shape
        out = torch.zeros_like(x)
        if T == 0:
            return out
        for kk in torch.unique(bins).tolist():
            if kk < 0 or kk >= T:
                continue
            mask = bins == kk
            if kk == 0:
                out[:, :, mask] = x[:, :, mask]
            else:
                out[:, kk:, mask] = x[:, : T - kk, mask]
        return out

    def _fractional_shift(self, x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
        """Linear-interpolation (fractional) delay, differentiable in ``bins``.

        For ``d_c = k_c + f_c`` bins (``k`` integer, ``f in [0, 1)``)::

            out[t, c] = (1 - f_c) * x[t - k_c, c] + f_c * x[t - k_c - 1, c]

        with zero fill where an index is negative.  ``floor`` has zero gradient,
        so ``d(out)/d(f_c) = x[t-k_c-1] - x[t-k_c]`` and a delay change moves the
        waveform in time instead of adding a constant.  No ``round`` and no
        straight-through estimator on this path.
        """
        if x.shape[1] == 0:
            return torch.zeros_like(x)
        k = torch.floor(bins.detach()).to(torch.long)
        f = bins - k.to(bins.dtype)                      # (C,) differentiable
        x0 = self._shift_by_bins(x, k)
        x1 = self._shift_by_bins(x, k + 1)
        w0 = (1.0 - f).to(x.dtype).reshape(1, 1, -1)
        w1 = f.to(x.dtype).reshape(1, 1, -1)
        return x0 * w0 + x1 * w1

    def delayed_input(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the per-channel delays to ``x: (B, T, C)``.

        The *whole* sequence is shifted before it is split into the model's time
        segments, so the history a delayed channel needs is never lost at a
        segment boundary: a sample pulled from an earlier segment was already
        written into the shifted tensor.

        Disabled -> ``x`` is returned unchanged (same object).  A delay of
        exactly 0 bins would also be bit-identical to ``x``, but the sigmoid
        parameterisation keeps the delay strictly positive.
        """
        if self.input_delay_raw is None:
            return x
        return self._fractional_shift(x, self.input_delay_bins())

    def init_adaptation(self, batch: int, device: torch.device):
        """Zero adaptation currents for a fresh batch: ``(u_fast, u_slow)``.

        Mirrors :meth:`init_state`: the currents are per-batch recurrent state and
        must never leak from one batch into the next.  An inactive timescale is
        ``None``; with the feature off both are ``None``.
        """
        if not self.adlif_enabled:
            return None, None

        def zeros() -> torch.Tensor:
            return torch.zeros(
                batch, self.n_neurons, device=device, dtype=self.state_dtype
            )

        return (
            zeros() if self.adlif_use_fast else None,
            zeros() if self.adlif_use_slow else None,
        )

    # ------------------------------------------------------------------ #
    def attention_weights(self, num_bins: int | None = None) -> torch.Tensor:
        """Learned probability distribution over the ``T`` time bins (M11b).

        Returns ``softmax(attn_logits[:T] / temperature)`` as an fp32 ``(T,)``
        tensor that sums to 1.  Raises if attention is disabled; this keeps the
        disabled path parameter-free and makes misuse loud.
        """
        if self.attn_logits is None:
            raise RuntimeError("temporal attention is disabled (cfg.attention.enabled=False)")
        T = int(self.cfg.total_steps if num_bins is None else num_bins)
        if T < 1 or T > self.attn_logits.numel():
            raise ValueError(
                f"attention has {self.attn_logits.numel()} bin logits, got T={T}"
            )
        temperature = float(self.cfg.attention.temperature)
        if not temperature > 0.0:  # cfg.validate() checks this too
            raise ValueError("attention.temperature must be > 0")
        # fp32 math regardless of the state dtype: the distribution is a readout
        # weight and T can be 500, so the softmax is never done in half precision
        with _autocast_disabled(self.attn_logits.device.type):
            return torch.softmax(self.attn_logits[:T].float() / temperature, dim=0)

    def readout_counts(self, spikes: torch.Tensor) -> torch.Tensor:
        """Collapse ``(B, T, N)`` spikes to the ``(B, N)`` classifier input.

        Off (default): exact spike-count sum, or its mean when ``cfg.readout ==
        "mean"`` -- identical operations and order to the historical readout.

        On: ``sum_t (T * a_t) * c_t`` with the learned bin distribution ``a``.
        The softmax normalises over bins, so ``a`` sums to 1; left like that,
        ``counts_att`` would be ``1/T`` of the plain count scale (T=500), which
        made the classifier gradient 500x too small and left the zero-initialised
        readout stuck.  The weights are therefore multiplied by ``T`` *before*
        the reduction, so their mean is 1 and uniform attention reproduces the
        plain count sum exactly (``T * (1/T) == 1`` in fp32 for the bin counts
        used here).  ``cfg.readout`` is ignored when attention is on.
        """
        T = int(spikes.shape[1])
        counts = spikes.float().sum(1)                     # (B, N) exact integers
        if self.attn_logits is None:
            if self.cfg.readout == "mean":
                counts = counts / T
            return counts
        a = self.attention_weights(T)                      # (T,) fp32, sums to 1
        scale = a * T                                      # (T,), mean == 1
        return (spikes.float() * scale.view(1, T, 1)).sum(1)   # (B, N)

    # ------------------------------------------------------------------ #
    def _weights(self):
        """Cast every weight tensor to the state dtype (fp16 by default).

        Master weights stay fp32 so the optimizer keeps full precision; the
        computation itself is genuinely fp16.  ``w_ad`` is ``None`` when
        spike-frequency adaptation is disabled (nothing registered then);
        ``w_emit`` / ``w_cls`` are ``None`` in the learning-matrix mode, where
        neither the emission map nor the count classifier exists.
        """
        dt = self.state_dtype
        w_ad = None if self.w_ad is None else self.w_ad.to(dt)
        # absent (None) in the learning-matrix mode: nothing was registered
        w_emit = None if self.w_emit is None else self.w_emit.to(dt)
        w_cls = None if self.w_cls is None else self.w_cls.to(dt)
        return (
            self.w_in.to(dt),
            self.w_down.to(dt),
            self.w_up.to(dt),
            w_emit,
            self.w_rec.to(dt),
            self.gate.to(dt),
            self.bias.to(dt),
            self.w_out.to(dt),
            self.theta.to(dt),
            w_cls,
            w_ad,
        )

    # ------------------------------------------------------------------ #
    def effective_lam(self) -> torch.Tensor:
        """Effective inner-stage gain in fp32: ``lam_min + span * sigmoid(raw)``.

        Bounded by construction to ``(lam_min, lam_max)`` (up to fp32 saturation
        exactly at the endpoints for ``|lam_raw|`` in the hundreds), hence
        finite and strictly positive: no gradient can switch the stage off.
        Only defined when ``two_stage.enabled`` is true.
        """
        if self.lam_raw is None:
            raise RuntimeError("two_stage is disabled: effective_lam() is undefined")
        ts = self.cfg.two_stage
        lo, hi = float(ts.lam_min), float(ts.lam_max)
        return lo + (hi - lo) * torch.sigmoid(self.lam_raw.float())

    def two_stage_weights(self):
        """Casted ``(w_down2, w_up2, lam)`` of the second stage, or ``None``.

        ``lam`` is the *effective* gain
        (:meth:`effective_lam`, cast to the state dtype); ``lam_raw`` itself is
        never returned, so the update line sees a single bounded scalar.
        ``None`` whenever ``two_stage.enabled`` is false, in which case the
        timestep update runs the baseline single-mixer expression untouched.
        Kept as a method (not part of :meth:`_weights`) so the baseline weight
        tuple and its unpacking are unchanged.
        """
        if self.w_down2 is None or self.w_up2 is None or self.lam_raw is None:
            return None
        dt = self.state_dtype
        return self.w_down2.to(dt), self.w_up2.to(dt), self.effective_lam().to(dt)

    # ------------------------------------------------------------------ #
    def _segment_any(
        self,
        z: torch.Tensor,
        a_chunk: torch.Tensor,
        b: torch.Tensor | None = None,
        u_fast: torch.Tensor | None = None,
        u_slow: torch.Tensor | None = None,
        state_sum: torch.Tensor | None = None,
        lm_a: torch.Tensor | None = None,
    ):
        """Run one chunk of timesteps with every optional mechanism.

        Returns ``(z_next, spikes, potentials, b_next, u_fast_next, u_slow_next)``
        -- always six values, so ``torch.utils.checkpoint`` sees a fixed arity and
        only tensors as positional arguments (inactive states arrive as
        zero-length tensors and are never read).  ``b`` is the spike-frequency
        adaptation trace ``b <- rho*b + s`` of M1; ``u_fast`` / ``u_slow`` are the
        two adLIF currents of M8.  The M1 adaptation current is subtracted from the
        spike-generation potential; the adLIF currents are subtracted from the
        pre-activation inside the tanh.  With both features off the computation is
        bit-identical to the baseline.

        ``lm_a`` is the ``(chunk, K)`` slice of the learning-matrix time weights
        ``a``.  When it is given (R2 mode) a seventh value, this chunk's fp32
        ``(B, K)`` matrix-readout partial, is appended to the tuple; the readout
        rows are folded as the chunk runs, so ``(B, T, D)`` is never formed.  The
        baseline call omits the argument and keeps the six-value result.
        """
        w_in, w_down, w_up, w_emit, w_rec, gate, bias, w_out, theta, _, w_ad = self._weights()
        dt = self.state_dtype
        alpha = float(self.cfg.alpha)
        # per-neuron leak when trainable; computed once per chunk and broadcast
        # over D.  None (default) keeps the scalar baseline update untouched.
        alpha_i = self.alpha_per_neuron()
        if alpha_i is not None:
            alpha_i = alpha_i.to(dt).view(1, self.n_neurons, 1)
        beta, gamma = float(self.cfg.surrogate_beta), float(self.cfg.surrogate_gamma)
        adapt = bool(self.cfg.adaptation.enabled)
        rho = float(self.cfg.adaptation.rho)
        if adapt and (b is None or b.numel() == 0):
            b = self.init_adaptation_state(int(z.shape[0]), z.device)
        adlif = bool(self.adlif_enabled)
        rho_fast = float(self.cfg.adlif.rho_fast)
        rho_slow = float(self.cfg.adlif.rho_slow)
        g_fast = self.adlif_gain_fast.to(dt) if (adlif and self.adlif_use_fast) else None
        g_slow = self.adlif_gain_slow.to(dt) if (adlif and self.adlif_use_slow) else None
        use_fast = u_fast is not None and u_fast.numel() > 0 and g_fast is not None
        use_slow = u_slow is not None and u_slow.numel() > 0 and g_slow is not None
        surrogate_kind = str(self.cfg.surrogate)
        surrogate_alpha = float(self.cfg.surrogate_alpha)
        two_stage = self.two_stage_weights()   # None on the baseline path
        if two_stage is not None:
            w_down2, w_up2, lam = two_stage

        spikes, pots = [], []
        # R2: fp32 (B, K) partial of sum_t a[t] (L_t @ v), accumulated in this
        # chunk; None on the baseline path (then ``w_emit`` is not None).
        lm_part = None
        if lm_a is not None:
            lm_part = torch.zeros(
                z.shape[0], self.n_classes, device=z.device, dtype=torch.float32
            )
        for k in range(a_chunk.shape[1]):
            if state_sum is not None:
                state_sum += z                                        # fp32 running sum
            pot = (z * w_out).sum(-1) + theta                          # (B, N) fp16
            if adapt:
                pot = pot - w_ad * b                                   # adaptation current
            pots.append(pot)
            s = surrogate_spike(
                pot, beta, gamma, kind=surrogate_kind, alpha=surrogate_alpha
            )                                                          # (B, N) fp16
            spikes.append(s)
            if adapt:
                b = rho * b + s                                        # decaying spike trace
            h = (s.unsqueeze(-1) * z).sum(1) * self.pop_scale          # (B, D) pooled
            if w_emit is not None:
                h = h @ w_emit                                         # (B, D) emitted
            if lm_part is not None:
                # R2: L_t := h_t is the pooled state itself (no emission map), so
                # this bin contributes ``a[t] (x) v`` to the running logits.  fp32
                # like the count classifier; (B, T, D) is never materialised.
                lm_part = lm_part + (h.float() @ self.v_matrix) * lm_a[k].float()
            c = s @ w_rec                                              # (B, N)
            mix = (z @ w_down) @ w_up                                   # (B, N, D)
            drive = gate * a_chunk[:, k].unsqueeze(1) + c.unsqueeze(-1) * h.unsqueeze(1) + bias
            pre = mix + drive
            if use_fast:
                pre = pre - (g_fast.view(1, -1, 1) * u_fast.unsqueeze(-1)).to(dt)
            if use_slow:
                pre = pre - (g_slow.view(1, -1, 1) * u_slow.unsqueeze(-1)).to(dt)
            if two_stage is None:
                new_z = torch.tanh(pre)
            else:
                # second, independent rank-R2 mixer + tanh, at the same state;
                # `drive` is shared and `lam` (bounded sigmoid gain, init 1.0)
                # gates the inner stage.  The adLIF currents act on `pre` only,
                # so the inner stage sees the bare mix + drive, as on its branch.
                mix2 = (z @ w_down2) @ w_up2
                inner = torch.tanh(mix2 + drive)
                new_z = torch.tanh(mix + lam * inner)
            if alpha_i is None:
                z = z + alpha * (new_z - z)
            else:
                z = z + alpha_i * (new_z - z)
            z = z.to(dt)
            if adlif:
                if use_fast:
                    u_fast = rho_fast * u_fast + s                     # advance with this spike
                if use_slow:
                    u_slow = rho_slow * u_slow + s
            # controlled imprecision is applied to the *actual* state that the
            # next timestep's spike generation / population signal / mixing read,
            # so its effect genuinely propagates through the dynamics.
            if self.cfg.state_regularization.mode != "none":
                z = regularize_state(z, self.cfg.state_regularization, self._reg_phase())
        out = (
            z,
            torch.stack(spikes, dim=1),
            torch.stack(pots, dim=1),
            b,
            u_fast,
            u_slow,
        )
        if lm_part is not None:
            out = out + (lm_part,)     # R2: this chunk's fp32 (B, K) readout partial
        return out

    def _segment(self, z: torch.Tensor, a_chunk: torch.Tensor,
                 b: torch.Tensor | None = None,
                 state_sum: torch.Tensor | None = None):
        """Back-compatible 3- or 4-tuple view of :meth:`_segment_any` (M1 only).

        Returns ``(z, spikes, pots)`` when adaptation is off and
        ``(z, spikes, pots, b)`` when it is on, exactly as before the merge.
        """
        z, s, p, b, _, _ = self._segment_any(z, a_chunk, b, None, None, state_sum)
        return (z, s, p, b) if bool(self.cfg.adaptation.enabled) else (z, s, p)

    def _segment_adlif(self, z, a_chunk, u_fast, u_slow, state_sum=None):
        """Back-compatible 5-tuple view of :meth:`_segment_any` (adLIF only)."""
        z, s, p, _, u_fast, u_slow = self._segment_any(
            z, a_chunk, None, u_fast, u_slow, state_sum
        )
        return z, u_fast, u_slow, s, p

    def forward(
        self,
        x: torch.Tensor,
        return_potentials: bool = False,
        return_states: bool = False,
    ):
        """``x``: ``(B, T, C)`` float tensor.

        With ``cfg.input_delays.enabled`` the input is first delayed per channel
        by its learned fractional delay (``delayed_input``, linear interpolation
        between neighbouring bins, zero fill at the start of the window), so the
        per-step gate ``a(t)`` is computed from the delayed sequence.

        Returns ``(logits, spikes[, pots][, z_final, z_mean])``; the optional
        state summary is appended after the optional potentials.  ``z_final`` is
        the ``(B, N, D)`` state after the last update, ``z_mean`` the ``(B, N, D)``
        time-mean of the state read at each timestep (accumulated in fp32,
        returned in the state dtype).  ``return_states=True`` is an analysis
        path: it is not supported together with gradient checkpointing.

        ``cfg.adaptation.enabled`` changes no return value: the ``(B, N)``
        adaptation trace is internal state, zero at the first timestep and
        carried across the time segments exactly like ``z``.

        ``cfg.learning_matrix.enabled`` likewise changes no return value.  The
        logits then come from the learning-matrix readout (the per-chunk
        ``(B, K)`` partials are summed here) instead of ``counts @ W_cls``;
        ``spikes`` -- and every rate/statistic derived from them -- are
        unchanged.
        """
        B, T, C = x.shape
        if C != self.n_inputs:
            raise ValueError(f"expected {self.n_inputs} input channels, got {C}")
        max_steps = int(self.cfg.total_steps)   # stimulus bins + optional silence tail
        if T > max_steps:
            raise ValueError(f"expected at most {max_steps} timesteps, got {T}")
        # learnable per-input-channel conduction delays, applied to the whole
        # sequence *before* it is split into segments (see ``delayed_input``)
        x = self.delayed_input(x)
        dt = self.state_dtype
        w_in = self.w_in.to(dt)

        # Input delay line (M10): K shifted copies of the already-binned input,
        # applied once per batch *before* the checkpointed temporal loop.  The
        # helper returns `x` itself when the block is disabled or copies == 1,
        # so the baseline path below is unchanged and bit-identical.
        x = apply_input_delay_line(
            x, self.cfg.input_delay_line, bin_ms=self.cfg.bin_ms, dtype=dt
        )

        # shared input projection: one big matmul over the whole sequence
        a = (x.reshape(B * T, -1).to(dt) @ w_in).reshape(B, T, self.state_dim)

        z = self.init_state(B, x.device)
        adapt = bool(self.cfg.adaptation.enabled)
        # adaptation trace (B, N): reset at the start of every sequence, then
        # carried across the time segments exactly like the state z
        b = self.init_adaptation_state(B, x.device) if adapt else None
        chunks = int(self.cfg.grad_checkpoint_chunks)
        step = max(1, math.ceil(T / chunks))
        use_ckpt = self.training and checkpoint_supported(z)
        if return_states and use_ckpt:
            raise ValueError(
                "return_states=True is not supported together with gradient "
                "checkpointing; call forward under torch.no_grad()/eval()"
            )
        # fp32 accumulator: a fp16 running sum would lose ~1e-1 relative accuracy
        # after T=500 additions.  Never stores per-timestep states.
        state_sum = (
            torch.zeros(B, self.n_neurons, self.state_dim, device=x.device, dtype=torch.float32)
            if return_states
            else None
        )
        spikes, pots = [], []
        # Learning-matrix readout (R2): fp32 (B, K) accumulator of the per-chunk
        # partials.  The (B, T, D) matrix is never materialised -- each segment
        # folds its own readout rows against ``v_matrix`` / ``a_time`` as it runs.
        lm_logits = (
            torch.zeros(B, self.n_classes, device=x.device, dtype=torch.float32)
            if self.learning_matrix_enabled
            else None
        )
        if self.adlif_enabled:
            # per-batch reset of the adaptation currents, like `z` above
            u_fast, u_slow = self.init_adaptation(B, x.device)
        else:
            u_fast = u_slow = None
        empty = torch.zeros(0, device=x.device, dtype=dt)  # inactive-state placeholder
        for t0 in range(0, T, step):
            # rebuild the argument triple every chunk so the states returned by
            # the previous chunk (adaptation trace / adLIF currents) are carried
            # forward exactly like ``z``
            b_arg = b if b is not None else empty
            uf_arg = u_fast if u_fast is not None else empty
            us_arg = u_slow if u_slow is not None else empty
            a_chunk = a[:, t0 : t0 + step]
            # R2: the time weights of *this* chunk (None on the baseline path, so
            # the baseline call keeps its six positional arguments)
            lm_chunk = self.a_time[t0 : t0 + step] if lm_logits is not None else None
            if use_ckpt:
                # fixed arity, tensors only -- see VectorNeuronPopulation._segment_any
                if lm_chunk is None:
                    seg = checkpoint(
                        self._segment_any, z, a_chunk, b_arg, uf_arg, us_arg,
                        use_reentrant=False,
                    )
                else:
                    seg = checkpoint(
                        self._segment_any, z, a_chunk, b_arg, uf_arg, us_arg,
                        lm_a=lm_chunk, use_reentrant=False,
                    )
            elif lm_chunk is None:
                seg = self._segment_any(z, a_chunk, b_arg, uf_arg, us_arg, state_sum)
            else:
                seg = self._segment_any(
                    z, a_chunk, b_arg, uf_arg, us_arg, state_sum, lm_chunk
                )
            z, s, p, b, u_fast, u_slow = seg[:6]
            if lm_logits is not None:
                lm_logits = lm_logits + seg[6]
            spikes.append(s)
            if return_potentials:
                pots.append(p)
        spikes = torch.cat(spikes, dim=1)                              # (B, T, N)

        counts = self.readout_counts(spikes)                           # (B, N) fp32
        if lm_logits is not None:
            # R2: the classifier reads the accumulated (B, K) matrix projection;
            # the spike counts are still formed, exactly as before, but do not
            # reach the logits -- there is no count classifier in this mode.
            logits = lm_logits
        else:
            logits = counts @ self.w_cls.float()                       # (B, K) fp32
        out = [logits, spikes]
        if return_potentials:
            out.append(torch.cat(pots, dim=1))
        if return_states:
            out.append(z)                                              # final state (B,N,D)
            out.append((state_sum / T).to(dt))                         # time-mean state (B,N,D)
        return tuple(out)

    # ------------------------------------------------------------------ #
    def rate_hz(self, spikes: torch.Tensor) -> torch.Tensor:
        """Per-neuron mean firing rate [Hz] over batch and time."""
        return spikes.float().mean(dim=(0, 1)) / (self.cfg.bin_ms / 1000.0)

    def loss(self, logits: torch.Tensor, spikes: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Cross-entropy (+ optional firing-rate regulariser).  Always fp32."""
        ce = torch.nn.functional.cross_entropy(
            logits.float(), targets, label_smoothing=float(self.cfg.label_smoothing)
        )
        if self.cfg.rate_reg > 0:
            rate = self.rate_hz(spikes)
            pen = (rate - float(self.cfg.rate_target_hz)).square().mean()
            return ce + float(self.cfg.rate_reg) * pen, ce
        return ce, ce

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def calibrate_threshold(self, x: torch.Tensor, target_rate_hz: float | None = None) -> float:
        """Initialise ``theta`` so that every neuron starts near ``target_rate_hz``.

        ``theta`` enters as an *additive* offset (``o_i = <z_i, w_out_i> + theta_i``),
        so a neuron spikes when ``raw_i(t) > -theta_i``: the calibrated offset is the
        negated (1 - p) quantile of the raw potential, with
        ``p = target_rate_hz * dt`` the target per-step firing probability.
        The first eighth of the sequence is skipped so the state-onset transient
        does not drag the quantile down.  ``theta`` stays trainable.
        """
        target = float(self.cfg.rate_target_hz if target_rate_hz is None else target_rate_hz)
        p = min(0.49, max(1e-3, target * (self.cfg.bin_ms / 1000.0)))
        was_training = self.training
        self.eval()
        self.theta.data.zero_()
        for _ in range(3):  # fixed-point iteration: the rate feeds back on the drive
            _, _, pots = self.forward(x, return_potentials=True)
            t0 = int(pots.shape[1] // 8)  # skip the state-onset transient
            flat = pots[:, t0:].reshape(-1, self.n_neurons).float()
            # pots already include the current offset, so subtract its quantile
            self.theta.data -= torch.quantile(flat, 1.0 - p, dim=0).to(self.theta.dtype)
        if was_training:
            self.train()
        return target


def checkpoint_supported(z: torch.Tensor) -> bool:
    """Gradient checkpointing is used on accelerators and on autograd-ready tensors."""
    return bool(z.is_cuda or z.requires_grad)


__all__ = [
    "VectorNeuronPopulation",
    "SurrogateSpike",
    "surrogate_spike",
    "surrogate_derivative",
]
