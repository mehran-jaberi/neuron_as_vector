"""Parity and gradient tests for the fused timestep kernels of ``_segment_any``.

The optimisation under test replaces three chains of small elementwise kernels
with matmuls and in-place updates **without changing the equations**:

* ``pot = (z * w_out).sum(-1) + theta`` becomes one ``torch.mm`` + ``diagonal``
  (+ an in-place adaptation ``addcmul_``), because only the per-neuron diagonal
  of ``z @ w_out.T`` is needed;
* ``h = ((s.unsqueeze(-1) * z).sum(1) * pop_scale) @ w_emit`` becomes
  ``bmm(s.unsqueeze(1), z)`` + ``@ w_emit``, so the ``(B, N, D)`` product is
  never materialised;
* ``drive`` / ``pre`` are accumulated in place and the leak update is one
  ``torch.lerp`` (``addcmul`` for the per-neuron leak).

Every parity test therefore re-implements the *old* equations independently
(one timestep at a time, like ``test_two_stage.py::_baseline_reference``) and
checks the fast path against them in fp32 (``atol=1e-5``), plus the structural
promises that must not regress: same return tuple, chunk-boundary equivalence,
autograd through the in-place updates, and the optional feature paths
(adaptation, adLIF, two-stage, per-neuron leak, diagonal emission).

CPU-only and tiny (N=4, D=8, C=6, K=3, R=2, B=3, T=4): no dataset, no GPU.
"""

from __future__ import annotations

import pytest
import torch

from v3.config import (
    AdaptationConfig,
    AdlifConfig,
    StateDynamicsConfig,
    TimingConfig,
    TwoStageConfig,
    V3Config,
)
from v3.model import VectorNeuronPopulation, surrogate_spike

N, D, C, K, R = 4, 8, 6, 3, 2          # neurons, state dim, inputs, classes, base rank
B, T = 3, 4                             # batch, timesteps
ATOL = 1e-5                             # fp32 parity budget of the fast path


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def _cfg(*, chunks: int = 2, dtype: str = "float32", timesteps: int = T,
         **overrides) -> V3Config:
    cfg = V3Config(
        n_neurons=N,
        state_dim=D,
        mix_rank=R,
        n_inputs=C,
        n_classes=K,
        timing=TimingConfig(sequence_duration_ms=2.0 * timesteps, time_bin_ms=2.0),
        batch_size=B,
        device="cpu",
        dtype=dtype,
        amp=False,
        grad_checkpoint_chunks=chunks,
    )
    return cfg.with_overrides(**overrides)


def _model(*, chunks: int = 2, dtype: str = "float32",
           **overrides) -> VectorNeuronPopulation:
    model = VectorNeuronPopulation(_cfg(chunks=chunks, dtype=dtype, **overrides))
    with torch.no_grad():
        # Deterministic, well separated thresholds.  With the default theta = 0
        # every potential of the first steps sits within ~1e-3 of zero, so all
        # spikes would be 0 and a parity test on spikes would be vacuous.
        model.theta.copy_(torch.linspace(-0.4, 0.4, N))
    return model


def _binary_input(batch: int = B, timesteps: int = T, channels: int = C) -> torch.Tensor:
    torch.manual_seed(11)
    return torch.randint(0, 2, (batch, timesteps, channels)).float()


def _reference(model: VectorNeuronPopulation, x: torch.Tensor, *,
               two_stage: bool = False, adapt: bool = False, adlif: bool = False):
    """The pre-optimisation equations, re-implemented one timestep at a time."""
    dt = model.state_dtype
    w_in = model.w_in.to(dt)
    w_rec = model.w_rec.to(dt)
    gate = model.gate.to(dt)
    bias = model.bias.to(dt)
    w_out = model.w_out.to(dt)
    theta = model.theta.to(dt)
    w_ad = None if model.w_ad is None else model.w_ad.to(dt)
    # mixer switch (feat/mixer-switch): ``mix = 0`` when the pair is absent
    mixer_block = getattr(model.cfg, "mixer", None)
    mix_on = mixer_block is None or bool(mixer_block.enabled)
    w_down = None if model.w_down is None else model.w_down.to(dt)
    w_up = None if model.w_up is None else model.w_up.to(dt)
    # emission: full matrix, per-coordinate gain (diagonal, v3.1) or none (R2)
    w_emit = None if model.w_emit is None else model.w_emit.to(dt)
    w_emit_diag = getattr(model, "w_emit_diag", None)
    alpha = float(model.cfg.alpha)
    alpha_i = None
    if model.raw_alpha is not None:
        alpha_i = torch.sigmoid(model.raw_alpha.float()).to(dt).view(1, N, 1)
    beta, gamma = float(model.cfg.surrogate_beta), float(model.cfg.surrogate_gamma)

    lam = None
    if two_stage:
        ts = model.cfg.two_stage
        lo, hi = float(ts.lam_min), float(ts.lam_max)
        lam = (lo + (hi - lo) * torch.sigmoid(model.lam_raw.float())).to(dt)
        w_down2, w_up2 = model.w_down2.to(dt), model.w_up2.to(dt)

    if adlif:
        g_fast = model.adlif_gain_fast.to(dt)
        g_slow = model.adlif_gain_slow.to(dt)
        rho_fast = float(model.cfg.adlif.rho_fast)
        rho_slow = float(model.cfg.adlif.rho_slow)
        u_fast = torch.zeros(B, N, dtype=dt)
        u_slow = torch.zeros(B, N, dtype=dt)

    rho = float(model.cfg.adaptation.rho)
    b = torch.zeros(B, N, dtype=dt) if adapt else None

    batch, steps, _ = x.shape
    a = (x.reshape(batch * steps, -1).to(dt) @ w_in).reshape(batch, steps, D)
    z = model.init_state(batch, x.device)
    spikes, pots = [], []
    for k in range(steps):
        pot = (z * w_out).sum(-1) + theta
        if adapt:
            pot = pot - w_ad * b
        pots.append(pot)
        s = surrogate_spike(
            pot, beta, gamma,
            kind=str(model.cfg.surrogate), alpha=float(model.cfg.surrogate_alpha),
        )
        spikes.append(s)
        if adapt:
            b = rho * b + s
        pooled = (s.unsqueeze(-1) * z).sum(1) * model.pop_scale
        if w_emit_diag is not None:
            h = pooled * w_emit_diag.to(dt)
        elif w_emit is None:
            h = pooled
        else:
            h = pooled @ w_emit
        c = s @ w_rec
        mix = (z @ w_down) @ w_up if mix_on else 0.0
        drive = gate * a[:, k].unsqueeze(1) + c.unsqueeze(-1) * h.unsqueeze(1) + bias
        pre = mix + drive
        if adlif:
            pre = pre - (g_fast.view(1, -1, 1) * u_fast.unsqueeze(-1)).to(dt)
            pre = pre - (g_slow.view(1, -1, 1) * u_slow.unsqueeze(-1)).to(dt)
        if two_stage:
            mix2 = (z @ w_down2) @ w_up2
            inner = torch.tanh(mix2 + drive)
            new_z = torch.tanh(mix + lam * inner)
        else:
            new_z = torch.tanh(pre)
        if alpha_i is None:
            z = z + alpha * (new_z - z)
        else:
            z = z + alpha_i * (new_z - z)
        z = z.to(dt)
        if adlif:
            u_fast = rho_fast * u_fast + s
            u_slow = rho_slow * u_slow + s
    spikes = torch.stack(spikes, dim=1)
    pots = torch.stack(pots, dim=1)
    counts = spikes.float().sum(1)
    if model.cfg.readout == "mean":
        counts = counts / spikes.shape[1]
    logits = counts @ model.w_cls.float()
    return logits, spikes, pots, z


def _parity(model: VectorNeuronPopulation, x: torch.Tensor, *, atol: float = ATOL,
            **flags):
    """Full forward of the fast path against the independent reference."""
    model.eval()
    with torch.no_grad():
        logits, spikes, pots, z_final, _ = model(
            x, return_potentials=True, return_states=True
        )
        ref_logits, ref_spikes, ref_pots, ref_z = _reference(model, x, **flags)
    assert torch.allclose(pots, ref_pots, atol=atol), "potentials diverged"
    assert torch.allclose(z_final, ref_z, atol=atol), "final state diverged"
    assert torch.allclose(logits, ref_logits, atol=atol), "logits diverged"
    assert torch.equal(spikes, ref_spikes), "spike train diverged"
    return logits, spikes, pots, z_final


# ---------------------------------------------------------------------- #
# 1. parity with the pre-optimisation equations
# ---------------------------------------------------------------------- #
def test_fp32_forward_matches_reference():
    torch.manual_seed(0)
    model = _model()
    _, spikes, _, _ = _parity(model, _binary_input())
    # non-degenerate: thresholds are crossed in both directions
    assert 0 < int(spikes.sum()) < spikes.numel()


def test_fp32_adaptation_matches_reference():
    torch.manual_seed(0)
    model = _model(adaptation=AdaptationConfig(enabled=True, gain_init=0.2))
    _parity(model, _binary_input(), adapt=True)


def test_fp32_adlif_matches_reference():
    torch.manual_seed(0)
    model = _model(adlif=AdlifConfig(enabled=True, gain_init=0.1))
    _parity(model, _binary_input(), adlif=True)


def test_fp32_two_stage_matches_reference():
    torch.manual_seed(0)
    model = _model(two_stage=TwoStageConfig(enabled=True, rank=R))
    _parity(model, _binary_input(), two_stage=True)


def test_fp32_per_neuron_leak_matches_reference():
    torch.manual_seed(0)
    model = _model(state_dynamics=StateDynamicsConfig(enabled=True, trainable_alpha=True))
    _parity(model, _binary_input())


def test_mixer_off_matches_reference():
    if not hasattr(V3Config(), "mixer"):
        pytest.skip("cfg.mixer is a feat/mixer-switch feature, not present on this branch")
    try:                                        # pragma: no cover - branch dependent
        from v3.config import MixerConfig
    except ImportError:                         # pragma: no cover
        pytest.skip("MixerConfig not available")
    torch.manual_seed(0)
    model = _model(mixer=MixerConfig(enabled=False))
    _parity(model, _binary_input())


def test_diagonal_emission_matches_reference():
    try:                                        # pragma: no cover - branch dependent
        from v3.config import EmissionConfig
    except ImportError:                         # pragma: no cover
        pytest.skip("emission.mode is a v3.1 feature, not present on this branch")
    torch.manual_seed(0)
    model = _model(emission=EmissionConfig(mode="diagonal"))
    _parity(model, _binary_input())


# ---------------------------------------------------------------------- #
# 2. chunk boundaries and autograd
# ---------------------------------------------------------------------- #
def test_chunked_path_is_bit_identical_to_single_chunk():
    torch.manual_seed(0)
    single = _model(chunks=1).eval()
    torch.manual_seed(0)
    chunked = _model(chunks=2).eval()
    x = _binary_input()
    with torch.no_grad():
        one = single(x, return_potentials=True, return_states=True)
        two = chunked(x, return_potentials=True, return_states=True)
    for lhs, rhs in zip(one, two):
        assert torch.equal(lhs, rhs)


@pytest.mark.parametrize("chunks", [1, 2])
def test_backward_parameter_grads_are_finite(chunks):
    torch.manual_seed(0)
    model = _model(chunks=chunks).train()
    logits, _ = model(_binary_input())
    logits.float().sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has a non-finite gradient"


def test_chunked_backward_under_gradient_checkpointing():
    """Learned ``z0`` makes ``checkpoint_supported`` true on CPU, so this runs
    the actual ``torch.utils.checkpoint`` branch (non-reentrant recompute)."""
    from v3.model import checkpoint_supported

    torch.manual_seed(0)
    model = _model(
        chunks=2,
        state_dynamics=StateDynamicsConfig(enabled=True, learnable_z0=True),
    ).train()
    x = _binary_input()
    assert checkpoint_supported(model.init_state(B, x.device))
    logits, _ = model(x)
    logits.float().sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has a non-finite gradient"


# ---------------------------------------------------------------------- #
# 3. the production dtype is preserved end to end
# ---------------------------------------------------------------------- #
def test_fp16_state_dtype_is_preserved():
    torch.manual_seed(0)
    model = _model(dtype="float16").eval()
    with torch.no_grad():
        try:
            logits, spikes, pots, z_final, z_mean = model(
                _binary_input(), return_potentials=True, return_states=True
            )
        except RuntimeError as exc:              # pragma: no cover - torch build dependent
            pytest.skip(f"fp16 CPU path unavailable in this torch build: {exc}")
    assert pots.dtype == torch.float16
    assert spikes.dtype == torch.float16
    assert z_final.dtype == torch.float16 and z_mean.dtype == torch.float16
    assert torch.isfinite(pots.float()).all()
    assert torch.isfinite(logits).all()
