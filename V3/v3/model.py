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

Readout (spike counts, 20 SHD classes)::

       logits = ( sum_t s(t) ) W_cls                  in R^20

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

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .config import V3Config
from .state_regularization import regularize_state


# ---------------------------------------------------------------------- #
class SurrogateSpike(torch.autograd.Function):
    """Heaviside spike with a fast-sigmoid surrogate gradient.

    Explicit autograd Function (rather than a hand-written straight-through
    expression) so that the gradient is exactly the derivative of the intended
    surrogate primitive ``gamma * u / (1 + beta * |u|)``::

        d/du [ gamma * u / (1 + beta|u|) ] = gamma / (1 + beta|u|)^2

    which equals ``gamma`` at ``u = 0``.  ``tests``-style gradient checks in
    ``run_smoke_test.py`` verify this numerically.
    """

    @staticmethod
    def forward(ctx, u: torch.Tensor, beta: float, gamma: float) -> torch.Tensor:
        u_w = u if u.dtype in (torch.float32, torch.float64) else u.float()
        ctx.save_for_backward(u_w)
        ctx.beta = float(beta)
        ctx.gamma = float(gamma)
        return (u_w > 0).to(u.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (u_w,) = ctx.saved_tensors
        deriv = ctx.gamma / (1.0 + ctx.beta * u_w.abs()).square()
        return (grad_out.to(deriv.dtype) * deriv).to(grad_out.dtype), None, None


def surrogate_spike(u: torch.Tensor, beta: float = 4.0, gamma: float = 1.0) -> torch.Tensor:
    return SurrogateSpike.apply(u, beta, gamma)


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

        g = torch.Generator().manual_seed(int(cfg.seed) + 12345)

        def rnd(*shape, scale=1.0):
            return (torch.randn(*shape, generator=g) * scale).to(torch.float32)

        self.w_in = nn.Parameter(rnd(C, D, scale=float(cfg.input_gain) / math.sqrt(C)))  # shared input basis
        self.w_down = nn.Parameter(rnd(D, R, scale=1.0 / math.sqrt(D)))    # shared rank-R mixing
        self.w_up = nn.Parameter(rnd(R, D, scale=0.5 / math.sqrt(R)))      # (init mixing gain ~0.5)
        if cfg.emit_identity_init:
            self.w_emit = nn.Parameter(torch.eye(D, dtype=torch.float32) + rnd(D, D, scale=0.02))
        else:
            self.w_emit = nn.Parameter(rnd(D, D, scale=1.0 / math.sqrt(D)))
        self.w_rec = nn.Parameter(rnd(N, N, scale=1.0 / math.sqrt(N)))
        self.gate = nn.Parameter(torch.ones(N, D, dtype=torch.float32) + rnd(N, D, scale=0.05))
        self.bias = nn.Parameter(torch.zeros(N, D, dtype=torch.float32))
        self.w_out = nn.Parameter(rnd(N, D, scale=1.0 / math.sqrt(D)))
        self.theta = nn.Parameter(torch.zeros(N, dtype=torch.float32))
        # zero-initialised classifier: the initial logits are exactly 0, i.e. the
        # initial cross-entropy is ln(20).  With a spike-count readout the logit
        # scale grows with the firing rate, so a random readout init would start
        # far from the loss basin.
        self.w_cls = nn.Parameter(torch.zeros(N, K, dtype=torch.float32))

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
    def _weights(self):
        """Cast every weight tensor to the state dtype (fp16 by default).

        Master weights stay fp32 so the optimizer keeps full precision; the
        computation itself is genuinely fp16.  ``w_ad`` is ``None`` when
        spike-frequency adaptation is disabled (nothing registered then).
        """
        dt = self.state_dtype
        w_ad = None if self.w_ad is None else self.w_ad.to(dt)
        return (
            self.w_in.to(dt),
            self.w_down.to(dt),
            self.w_up.to(dt),
            self.w_emit.to(dt),
            self.w_rec.to(dt),
            self.gate.to(dt),
            self.bias.to(dt),
            self.w_out.to(dt),
            self.theta.to(dt),
            self.w_cls.to(dt),
            w_ad,
        )

    # ------------------------------------------------------------------ #
    def _segment_any(
        self,
        z: torch.Tensor,
        a_chunk: torch.Tensor,
        b: torch.Tensor | None = None,
        u_fast: torch.Tensor | None = None,
        u_slow: torch.Tensor | None = None,
        state_sum: torch.Tensor | None = None,
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

        spikes, pots = [], []
        for k in range(a_chunk.shape[1]):
            if state_sum is not None:
                state_sum += z                                        # fp32 running sum
            pot = (z * w_out).sum(-1) + theta                          # (B, N) fp16
            if adapt:
                pot = pot - w_ad * b                                   # adaptation current
            pots.append(pot)
            s = surrogate_spike(pot, beta, gamma)                      # (B, N) fp16
            spikes.append(s)
            if adapt:
                b = rho * b + s                                        # decaying spike trace
            h = ((s.unsqueeze(-1) * z).sum(1) * self.pop_scale) @ w_emit    # (B, D)
            c = s @ w_rec                                              # (B, N)
            mix = (z @ w_down) @ w_up                                   # (B, N, D)
            drive = gate * a_chunk[:, k].unsqueeze(1) + c.unsqueeze(-1) * h.unsqueeze(1) + bias
            pre = mix + drive
            if use_fast:
                pre = pre - (g_fast.view(1, -1, 1) * u_fast.unsqueeze(-1)).to(dt)
            if use_slow:
                pre = pre - (g_slow.view(1, -1, 1) * u_slow.unsqueeze(-1)).to(dt)
            if alpha_i is None:
                z = z + alpha * (torch.tanh(pre) - z)
            else:
                z = z + alpha_i * (torch.tanh(pre) - z)
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
        return (
            z,
            torch.stack(spikes, dim=1),
            torch.stack(pots, dim=1),
            b,
            u_fast,
            u_slow,
        )

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
        """
        B, T, C = x.shape
        if C != self.n_inputs:
            raise ValueError(f"expected {self.n_inputs} input channels, got {C}")
        if T > self.cfg.n_bins:
            raise ValueError(f"expected at most {self.cfg.n_bins} timesteps, got {T}")
        # learnable per-input-channel conduction delays, applied to the whole
        # sequence *before* it is split into segments (see ``delayed_input``)
        x = self.delayed_input(x)
        dt = self.state_dtype
        w_in = self.w_in.to(dt)

        # shared input projection: one big matmul over the whole sequence
        a = (x.reshape(B * T, C).to(dt) @ w_in).reshape(B, T, self.state_dim)

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
            if use_ckpt:
                # fixed arity, tensors only -- see VectorNeuronPopulation._segment_any
                z, s, p, b, u_fast, u_slow = checkpoint(
                    self._segment_any, z, a_chunk, b_arg, uf_arg, us_arg,
                    use_reentrant=False,
                )
            else:
                z, s, p, b, u_fast, u_slow = self._segment_any(
                    z, a_chunk, b_arg, uf_arg, us_arg, state_sum
                )
            spikes.append(s)
            if return_potentials:
                pots.append(p)
        spikes = torch.cat(spikes, dim=1)                              # (B, T, N)

        counts = spikes.float().sum(1)                                 # (B, N) exact integers
        if self.cfg.readout == "mean":
            counts = counts / spikes.shape[1]
        logits = counts @ self.w_cls.float()                           # (B, K) fp32
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
]
