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
    def _segment(
        self,
        z: torch.Tensor,
        a_chunk: torch.Tensor,
        b: torch.Tensor | None = None,
        state_sum: torch.Tensor | None = None,
    ):
        """Run one chunk of timesteps.  Returns ``(z_next, spikes, potentials)``.

        Kept as a bound method so ``torch.utils.checkpoint`` can recompute it;
        the positional arguments are all tensors so the checkpointed call needs
        no non-tensor arguments.

        ``b`` is the optional ``(B, N)`` spike-frequency adaptation trace.  With
        ``cfg.adaptation.enabled`` the spike-generation potential becomes
        ``pot = (z * w_out).sum(-1) + theta - w_ad * b`` and the trace is updated
        after the spike as ``b = rho * b + s``.  ``b`` is carried across segments
        exactly like the state ``z`` (passing ``b=None`` starts from a zero
        trace), so the returned tuple then has a fourth element ``b_next``; with
        the feature off the three-tuple and the computation are exactly the
        baseline.

        ``state_sum`` is an optional ``(B, N, D)`` fp32 accumulator; when given,
        the state that is *read* at every timestep is added to it in place.  It is
        used by :meth:`forward` (``return_states=True``) to obtain the time-mean
        state without storing per-timestep states, and must not require grad.
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
        if adapt and b is None:
            b = self.init_adaptation_state(int(z.shape[0]), z.device)

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
            if alpha_i is None:
                z = z + alpha * (torch.tanh(mix + drive) - z)
            else:
                z = z + alpha_i * (torch.tanh(mix + drive) - z)
            z = z.to(dt)
            # controlled imprecision is applied to the *actual* state that the
            # next timestep's spike generation / population signal / mixing read,
            # so its effect genuinely propagates through the dynamics.
            if self.cfg.state_regularization.mode != "none":
                z = regularize_state(z, self.cfg.state_regularization, self._reg_phase())
        out = (z, torch.stack(spikes, dim=1), torch.stack(pots, dim=1))
        if adapt:
            return out + (b,)
        return out

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

        ``cfg.adaptation.enabled`` changes no return value: the ``(B, N)``
        adaptation trace is internal state, zero at the first timestep and
        carried across the time segments exactly like ``z``.
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
        for t0 in range(0, T, step):
            a_chunk = a[:, t0 : t0 + step]
            if use_ckpt:
                if adapt:
                    # all positional args are tensors: z, a_chunk, b
                    z, s, p, b = checkpoint(
                        self._segment, z, a_chunk, b, use_reentrant=False
                    )
                else:
                    z, s, p = checkpoint(self._segment, z, a_chunk, use_reentrant=False)
            else:
                if adapt:
                    z, s, p, b = self._segment(z, a_chunk, b, state_sum)
                else:
                    z, s, p = self._segment(z, a_chunk, state_sum=state_sum)
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
