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
    alpha: float = 0.5,
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
        alpha: float = 0.5,
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
    alpha: float = 0.5,
) -> torch.Tensor:
    """Heaviside spike; ``kind`` selects the surrogate gradient (see M11a)."""
    return SurrogateSpike.apply(u, beta, gamma, kind, alpha)


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

        # ---- learnt temporal attention over the readout (M11b) -----------
        # One logit per time bin, zero-initialised so the softmax starts uniform
        # and the initial forward is the mean-count readout.  The parameter is
        # created *only* when the block is enabled: the disabled model has the
        # exact same parameter set and forward as before.
        if bool(cfg.attention.enabled):
            self.attn_logits = nn.Parameter(torch.zeros(int(cfg.n_bins), dtype=torch.float32))
        else:
            self.attn_logits = None

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

    # ------------------------------------------------------------------ #
    def attention_weights(self, num_bins: int | None = None) -> torch.Tensor:
        """Learned probability distribution over the ``T`` time bins (M11b).

        Returns ``softmax(attn_logits[:T] / temperature)`` as an fp32 ``(T,)``
        tensor that sums to 1.  Raises if attention is disabled; this keeps the
        disabled path parameter-free and makes misuse loud.
        """
        if self.attn_logits is None:
            raise RuntimeError("temporal attention is disabled (cfg.attention.enabled=False)")
        T = int(self.cfg.n_bins if num_bins is None else num_bins)
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
        On: ``sum_t a_t * c_t`` with the learned bin distribution ``a``.  In
        that case ``cfg.readout`` is ignored: the softmax already normalises
        over bins, so dividing by ``T`` again would shrink the logit scale.
        """
        T = int(spikes.shape[1])
        counts = spikes.float().sum(1)                     # (B, N) exact integers
        if self.attn_logits is None:
            if self.cfg.readout == "mean":
                counts = counts / T
            return counts
        a = self.attention_weights(T)                      # (T,) fp32, sums to 1
        return (spikes.float() * a.view(1, T, 1)).sum(1)   # (B, N)

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
        surrogate_kind = str(self.cfg.surrogate)
        surrogate_alpha = float(self.cfg.surrogate_alpha)

        spikes, pots = [], []
        for k in range(a_chunk.shape[1]):
            if state_sum is not None:
                state_sum += z                                        # fp32 running sum
            pot = (z * w_out).sum(-1) + theta                          # (B, N) fp16
            pots.append(pot)
            s = surrogate_spike(
                pot, beta, gamma, kind=surrogate_kind, alpha=surrogate_alpha
            )                                                          # (B, N) fp16
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
        for t0 in range(0, T, step):
            a_chunk = a[:, t0 : t0 + step]
            if use_ckpt:
                z, s, p = checkpoint(self._segment, z, a_chunk, use_reentrant=False)
            else:
                z, s, p = self._segment(z, a_chunk, state_sum)
            spikes.append(s)
            if return_potentials:
                pots.append(p)
        spikes = torch.cat(spikes, dim=1)                              # (B, T, N)

        counts = self.readout_counts(spikes)                           # (B, N) fp32
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
    "surrogate_derivative",
]
