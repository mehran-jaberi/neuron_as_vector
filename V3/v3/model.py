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

Readout (spike counts, 20 SHD classes)::

       logits = ( sum_t s(t) ) W_cls                  in R^20

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
        """Select the phase ("train" | "val" | "test") for state imprecision."""
        if phase not in ("train", "val", "test"):
            raise ValueError(f"unknown phase {phase!r}")
        self.state_reg_phase = phase

    def _reg_phase(self) -> str:
        if self.state_reg_phase is not None:
            return self.state_reg_phase
        return "train" if self.training else "val"

    def init_state(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(
            batch, self.n_neurons, self.state_dim, device=device, dtype=self.state_dtype
        )

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
        computation itself is genuinely fp16.
        """
        dt = self.state_dtype
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
        )

    # ------------------------------------------------------------------ #
    def _segment(
        self,
        z: torch.Tensor,
        a_chunk: torch.Tensor,
        state_sum: torch.Tensor | None = None,
    ):
        """Run one chunk of timesteps.  Returns ``(z_next, spikes, potentials)``.

        Kept as a bound method so ``torch.utils.checkpoint`` can recompute it.

        ``state_sum`` is an optional ``(B, N, D)`` fp32 accumulator; when given,
        the state that is *read* at every timestep is added to it in place.  It is
        used by :meth:`forward` (``return_states=True``) to obtain the time-mean
        state without storing per-timestep states, and must not require grad.
        """
        w_in, w_down, w_up, w_emit, w_rec, gate, bias, w_out, theta, _ = self._weights()
        dt = self.state_dtype
        alpha = float(self.cfg.alpha)
        beta, gamma = float(self.cfg.surrogate_beta), float(self.cfg.surrogate_gamma)

        spikes, pots = [], []
        for k in range(a_chunk.shape[1]):
            if state_sum is not None:
                state_sum += z                                        # fp32 running sum
            pot = (z * w_out).sum(-1) + theta                          # (B, N) fp16
            pots.append(pot)
            s = surrogate_spike(pot, beta, gamma)                      # (B, N) fp16
            spikes.append(s)
            h = ((s.unsqueeze(-1) * z).sum(1) * self.pop_scale) @ w_emit    # (B, D)
            c = s @ w_rec                                              # (B, N)
            mix = (z @ w_down) @ w_up                                   # (B, N, D)
            drive = gate * a_chunk[:, k].unsqueeze(1) + c.unsqueeze(-1) * h.unsqueeze(1) + bias
            z = z + alpha * (torch.tanh(mix + drive) - z)
            z = z.to(dt)
            # controlled imprecision is applied to the *actual* state that the
            # next timestep's spike generation / population signal / mixing read,
            # so its effect genuinely propagates through the dynamics.
            if self.cfg.state_regularization.mode != "none":
                z = regularize_state(z, self.cfg.state_regularization, self._reg_phase())
        return z, torch.stack(spikes, dim=1), torch.stack(pots, dim=1)

    # ------------------------------------------------------------------ #
    def _segment_adlif(
        self,
        z: torch.Tensor,
        a_chunk: torch.Tensor,
        u_fast: torch.Tensor | None,
        u_slow: torch.Tensor | None,
        state_sum: torch.Tensor | None = None,
    ):
        """Run one chunk with the two-timescale adaptation currents (feature ON).

        This is the adaptation-enabled twin of :meth:`_segment`; the original
        method is deliberately left byte-identical so the disabled path cannot
        drift and stays bit-identical to the baseline.  **Keep the two timestep
        loops in sync when the base dynamics change.**

        ``u_fast`` / ``u_slow`` are the ``(B, N)`` currents at the *start* of the
        chunk (already advanced to time ``t0``), or ``None`` for an inactive
        timescale.  Returns ``(z_next, u_fast, u_slow, spikes, potentials)``,
        where the returned currents are the ones the next chunk starts from.

        Ordering: the current entering step ``k`` is
        ``u(k) = rho * u(k-1) + s(k-1)``.  It is advanced with *this* step's
        spike right after the state update, so ``pre`` at step ``k`` reads the
        current driven by ``s(k-1)`` (and step 0 reads ``u = 0``).
        """
        w_in, w_down, w_up, w_emit, w_rec, gate, bias, w_out, theta, _ = self._weights()
        dt = self.state_dtype
        alpha = float(self.cfg.alpha)
        beta, gamma = float(self.cfg.surrogate_beta), float(self.cfg.surrogate_gamma)
        rho_fast = float(self.cfg.adlif.rho_fast)
        rho_slow = float(self.cfg.adlif.rho_slow)
        # cast the per-neuron gains to the state dtype once per chunk
        g_fast = self.adlif_gain_fast.to(dt) if u_fast is not None else None
        g_slow = self.adlif_gain_slow.to(dt) if u_slow is not None else None

        spikes, pots = [], []
        for k in range(a_chunk.shape[1]):
            if state_sum is not None:
                state_sum += z                                        # fp32 running sum
            pot = (z * w_out).sum(-1) + theta                          # (B, N) fp16
            pots.append(pot)
            s = surrogate_spike(pot, beta, gamma)                      # (B, N) fp16
            spikes.append(s)
            h = ((s.unsqueeze(-1) * z).sum(1) * self.pop_scale) @ w_emit    # (B, D)
            c = s @ w_rec                                              # (B, N)
            mix = (z @ w_down) @ w_up                                   # (B, N, D)
            drive = gate * a_chunk[:, k].unsqueeze(1) + c.unsqueeze(-1) * h.unsqueeze(1) + bias
            # adaptation currents: subtract `gain * u` inside the tanh, broadcast
            # over the D coordinates.  Casting the term back to the state dtype
            # keeps `pre` in the dtype the baseline uses; with gain_init = 0 the
            # subtraction is exactly 0.0, so the enabled-at-init path is inert.
            pre = mix + drive
            if u_fast is not None:
                pre = pre - (g_fast.view(1, -1, 1) * u_fast.unsqueeze(-1)).to(dt)
            if u_slow is not None:
                pre = pre - (g_slow.view(1, -1, 1) * u_slow.unsqueeze(-1)).to(dt)
            z = z + alpha * (torch.tanh(pre) - z)
            z = z.to(dt)
            # advance the order-1 currents with this step's spike
            if u_fast is not None:
                u_fast = rho_fast * u_fast + s
            if u_slow is not None:
                u_slow = rho_slow * u_slow + s
            # controlled imprecision is applied to the *actual* state that the
            # next timestep's spike generation / population signal / mixing read,
            # so its effect genuinely propagates through the dynamics.
            if self.cfg.state_regularization.mode != "none":
                z = regularize_state(z, self.cfg.state_regularization, self._reg_phase())
        return (
            z,
            u_fast,
            u_slow,
            torch.stack(spikes, dim=1),
            torch.stack(pots, dim=1),
        )

    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,
        return_potentials: bool = False,
        return_states: bool = False,
    ):
        """``x``: ``(B, T, C)`` float tensor.

        Returns ``(logits, spikes[, pots][, z_final, z_mean])``; the optional
        state summary is appended after the optional potentials.  ``z_final`` is
        the ``(B, N, D)`` state after the last update, ``z_mean`` the ``(B, N, D)``
        time-mean of the state read at each timestep (accumulated in fp32,
        returned in the state dtype).  ``return_states=True`` is an analysis
        path: it is not supported together with gradient checkpointing.
        """
        B, T, C = x.shape
        if C != self.n_inputs:
            raise ValueError(f"expected {self.n_inputs} input channels, got {C}")
        if T > self.cfg.n_bins:
            raise ValueError(f"expected at most {self.cfg.n_bins} timesteps, got {T}")
        dt = self.state_dtype
        w_in = self.w_in.to(dt)

        # shared input projection: one big matmul over the whole sequence
        a = (x.reshape(B * T, C).to(dt) @ w_in).reshape(B, T, self.state_dim)

        z = self.init_state(B, x.device)
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
        for t0 in range(0, T, step):
            a_chunk = a[:, t0 : t0 + step]
            if self.adlif_enabled:
                if use_ckpt:
                    z, u_fast, u_slow, s, p = checkpoint(
                        self._segment_adlif, z, a_chunk, u_fast, u_slow, use_reentrant=False
                    )
                else:
                    z, u_fast, u_slow, s, p = self._segment_adlif(
                        z, a_chunk, u_fast, u_slow, state_sum
                    )
            elif use_ckpt:
                z, s, p = checkpoint(self._segment, z, a_chunk, use_reentrant=False)
            else:
                z, s, p = self._segment(z, a_chunk, state_sum)
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


__all__ = ["VectorNeuronPopulation", "SurrogateSpike", "surrogate_spike"]
