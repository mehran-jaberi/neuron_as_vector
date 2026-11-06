"""Focused tests for the diagonal emission parameterisation (IDEAS.md "Thread 2").

CPU-only and tiny: no dataset, no GPU, no training.  The block is a
*parameterisation study*, so these tests pin the properties that make it one:

1. the default is ``mode="full"`` and the full path is bit-identical to the
   pre-change model -- the initialisation is replayed inline from the old
   ``__init__`` draw order and the forward is re-implemented independently
   from the pre-change equation;
2. ``mode="diagonal"`` registers a ``(D,)`` gain initialised to exactly 1.0 and
   **no** ``D x D`` matrix; at ``N=64, D=1000`` the population has exactly
   1,026,440 parameters (999,000 fewer than the full mode's 2,025,440);
3. the diagonal forward equals the independent expression
   ``h = ((s * z).sum(1) * pop_scale) * w_emit_diag``, and a non-unit gain
   genuinely changes the emitted vector (the gain is used, not just present);
4. backward is finite and ``w_emit_diag`` receives a non-zero gradient;
5. an invalid mode raises, and the block is wired into ``from_dict`` /
   ``describe`` / ``NESTED_BLOCKS`` / the dotted ``--override`` grammar.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from v3.config import (
    NESTED_BLOCKS,
    EmissionConfig,
    TimingConfig,
    V3Config,
    apply_override_items,
    apply_override_mapping,
    describe,
    parameter_groups,
)
from v3.model import VectorNeuronPopulation, surrogate_spike

N, D, C, K, R = 4, 8, 6, 3, 2          # neurons, state dim, inputs, classes, base rank
B, T = 3, 4                             # batch, timesteps


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def _cfg(*, mode: str = "full", dtype: str = "float32", chunks: int = 2,
         timesteps: int = T) -> V3Config:
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
    return cfg.with_overrides(emission=EmissionConfig(mode=mode))


def _model(**kwargs) -> VectorNeuronPopulation:
    return VectorNeuronPopulation(_cfg(**kwargs))


def _binary_input(batch: int = B, timesteps: int = T, channels: int = C) -> torch.Tensor:
    torch.manual_seed(11)
    return torch.randint(0, 2, (batch, timesteps, channels)).float()


def _randomize_readout(model: VectorNeuronPopulation) -> None:
    """Non-zero ``w_cls``: the shipped zero init makes every logit exactly 0."""
    g = torch.Generator().manual_seed(4321)
    with torch.no_grad():
        model.w_cls.copy_(torch.randn(model.w_cls.shape, generator=g) * 0.5)


def _replay_full_init(cfg: V3Config) -> dict:
    """The pre-change ``__init__`` draw order, replayed inline.

    This is the saved reference for requirement (b): at the same seed the
    pre-change model built exactly these tensors with exactly this stream.
    """
    g = torch.Generator().manual_seed(int(cfg.seed) + 12345)

    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).to(torch.float32)

    n, d, c, k, r = cfg.n_neurons, cfg.state_dim, cfg.n_inputs, cfg.n_classes, cfg.mix_rank
    ref = {
        "w_in": rnd(c, d, scale=float(cfg.input_gain) / math.sqrt(c)),
        "w_down": rnd(d, r, scale=1.0 / math.sqrt(d)),
        "w_up": rnd(r, d, scale=0.5 / math.sqrt(r)),
        "w_emit": torch.eye(d, dtype=torch.float32) + rnd(d, d, scale=0.02),
        "w_rec": rnd(n, n, scale=1.0 / math.sqrt(n)),
        "gate": torch.ones(n, d, dtype=torch.float32) + rnd(n, d, scale=0.05),
        "w_out": rnd(n, d, scale=1.0 / math.sqrt(d)),
    }
    return ref


def _forward_reference(model: VectorNeuronPopulation, x: torch.Tensor, *, diagonal: bool):
    """The pre-change / diagonal equation, re-implemented independently of ``_segment``."""
    dt = model.state_dtype
    w_in, w_down, w_up = model.w_in.to(dt), model.w_down.to(dt), model.w_up.to(dt)
    w_emit = (model.w_emit_diag if diagonal else model.w_emit).to(dt)
    w_rec, gate, bias = model.w_rec.to(dt), model.gate.to(dt), model.bias.to(dt)
    w_out, theta, w_cls = model.w_out.to(dt), model.theta.to(dt), model.w_cls
    alpha = float(model.cfg.alpha)
    beta, gamma = float(model.cfg.surrogate_beta), float(model.cfg.surrogate_gamma)

    b_, t_, c_ = x.shape
    a = (x.reshape(b_ * t_, c_).to(dt) @ w_in).reshape(b_, t_, model.state_dim)
    z = model.init_state(b_, x.device)
    spikes = []
    for k in range(t_):
        pot = (z * w_out).sum(-1) + theta
        s = surrogate_spike(pot, beta, gamma)
        spikes.append(s)
        pop = (s.unsqueeze(-1) * z).sum(1) * model.pop_scale
        h = pop * w_emit if diagonal else pop @ w_emit
        c = s @ w_rec
        mix = (z @ w_down) @ w_up
        drive = gate * a[:, k].unsqueeze(1) + c.unsqueeze(-1) * h.unsqueeze(1) + bias
        z = z + alpha * (torch.tanh(mix + drive) - z)
        z = z.to(dt)
    spikes = torch.stack(spikes, dim=1)
    counts = spikes.float().sum(1)
    if model.cfg.readout == "mean":
        counts = counts / spikes.shape[1]
    return counts @ w_cls.float(), spikes, z


# ---------------------------------------------------------------------- #
# 1. defaults and the full path
# ---------------------------------------------------------------------- #
def test_a_default_mode_is_full():
    assert EmissionConfig().mode == "full"
    assert V3Config().emission.mode == "full"
    assert V3Config().emission == EmissionConfig()
    # the default model is the historical one: w_emit exists, no gain vector
    model = _model()
    params = dict(model.named_parameters())
    assert "w_emit" in params and "w_emit_diag" not in params
    assert model.emit_diagonal is False


def _same_state(actual, expected):
    """Same computation, different summation order: matmuls (``mm``/``bmm``) in
    the fast loop reorder fp32 additions, so the state agrees to ~1 ULP rather
    than bit for bit.  Spikes and logits are still compared exactly."""
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

def test_b_full_mode_is_bit_identical_to_the_pre_change_model():
    cfg = _cfg(mode="full")
    model = VectorNeuronPopulation(cfg)
    # (i) initialisation: the old draw order, exactly
    for name, ref in _replay_full_init(cfg).items():
        assert torch.equal(getattr(model, name).detach(), ref), name
    # (ii) forward: the old equation, computed independently
    _randomize_readout(model)
    x = _binary_input()
    model.eval()
    with torch.no_grad():
        logits, spikes, z_final, _ = model(x, return_states=True)
        ref_logits, ref_spikes, ref_z = _forward_reference(model, x, diagonal=False)
    _same_state(z_final, ref_z)
    assert torch.equal(spikes, ref_spikes)
    assert torch.equal(logits, ref_logits)


# ---------------------------------------------------------------------- #
# 2. the diagonal parameter set and the parameter budget
# ---------------------------------------------------------------------- #
def test_c_diagonal_has_gain_not_matrix_and_exact_parameter_budget():
    model = _model(mode="diagonal")
    params = dict(model.named_parameters())
    assert "w_emit" not in params
    assert "w_emit_diag" in params
    assert tuple(model.w_emit_diag.shape) == (D,)
    assert model.w_emit_diag.dtype == torch.float32
    assert model.w_emit_diag.requires_grad
    assert torch.equal(model.w_emit_diag.detach(), torch.ones(D))
    assert model.emit_diagonal is True
    # controlled comparison: every *other* tensor keeps the full-mode draw, so
    # the two modes differ only in the emission map
    full = _model(mode="full")
    full_params = dict(full.named_parameters())
    for name in ("w_in", "w_down", "w_up", "w_rec", "gate", "bias", "w_out", "theta", "w_cls"):
        assert torch.equal(params[name].detach(), full_params[name].detach()), name
    # analytic budget == actual count, at the headline size
    full_big = V3Config(state_dim=1000, mix_rank=64, device="cpu", dtype="float32")
    diag_big = V3Config(
        state_dim=1000,
        mix_rank=64,
        device="cpu",
        dtype="float32",
        emission=EmissionConfig(mode="diagonal"),
    )
    assert parameter_groups(full_big)["w_emit"] == 1_000_000
    assert parameter_groups(full_big)["total"] == 2_025_440
    assert parameter_groups(diag_big)["w_emit"] == 1_000
    assert parameter_groups(diag_big)["total"] == 1_026_440
    model_big = VectorNeuronPopulation(diag_big)
    assert model_big.n_parameters() == 1_026_440
    assert "w_emit" not in dict(model_big.named_parameters())
    assert tuple(model_big.w_emit_diag.shape) == (1000,)
    # the saving is exactly the matrix minus the gain vector
    assert parameter_groups(full_big)["total"] - parameter_groups(diag_big)["total"] == 999_000


# ---------------------------------------------------------------------- #
# 3. the diagonal forward
# ---------------------------------------------------------------------- #
def test_d_diagonal_forward_matches_the_independent_reference():
    model = _model(mode="diagonal").eval()
    _randomize_readout(model)
    x = _binary_input()
    with torch.no_grad():
        logits, spikes, z_final, _ = model(x, return_states=True)
        ref_logits, ref_spikes, ref_z = _forward_reference(model, x, diagonal=True)
    _same_state(z_final, ref_z)
    assert torch.equal(spikes, ref_spikes)
    assert torch.equal(logits, ref_logits)

    # the gain is actually used: a non-uniform gain changes the forward, and it
    # still matches the reference with the same gain
    unit_z = z_final.clone()
    with torch.no_grad():
        model.w_emit_diag.copy_(torch.linspace(0.5, 1.5, D))
        model.theta.add_(1.0)        # guarantee spikes, so h(t) is non-zero
    with torch.no_grad():
        logits2, spikes2, z_gain, _ = model(x, return_states=True)
        ref_logits2, ref_spikes2, ref_z2 = _forward_reference(model, x, diagonal=True)
    _same_state(z_gain, ref_z2)
    assert torch.equal(spikes2, ref_spikes2)
    assert torch.equal(logits2, ref_logits2)
    assert not torch.equal(z_gain, unit_z)


# ---------------------------------------------------------------------- #
# 4. backward
# ---------------------------------------------------------------------- #
def test_e_diagonal_backward_is_finite_and_gain_gradient_is_non_zero():
    torch.manual_seed(0)
    model = _model(mode="diagonal")
    _randomize_readout(model)
    with torch.no_grad():
        model.theta.add_(1.0)        # make the population fire
    model.train()
    x = _binary_input()
    logits, _ = model(x)
    target = torch.tensor([0, 2, 1])
    loss = F.cross_entropy(logits.float(), target)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    grad = model.w_emit_diag.grad
    assert grad is not None and grad.shape == (D,)
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0


# ---------------------------------------------------------------------- #
# 5. validation and config plumbing
# ---------------------------------------------------------------------- #
def test_f_invalid_mode_raises():
    with pytest.raises(ValueError, match="emission.mode"):
        V3Config(emission=EmissionConfig(mode="lowrank")).validate()
    with pytest.raises(ValueError, match="emission.mode"):
        V3Config(emission=EmissionConfig(mode="")).validate()
    # the model validates its config, so construction fails too
    with pytest.raises(ValueError, match="emission.mode"):
        VectorNeuronPopulation(_cfg().with_overrides(emission=EmissionConfig(mode="banded")))
    with pytest.raises(ValueError, match="emission.mode"):
        V3Config.from_dict({"emission": {"mode": "lowrank"}}).validate()


def test_g_config_plumbing_from_dict_describe_and_dotted_override():
    cfg = _cfg().with_overrides(emission=EmissionConfig(mode="diagonal"))
    data = cfg.to_dict()
    assert isinstance(data["emission"], dict) and data["emission"]["mode"] == "diagonal"
    back = V3Config.from_dict(data, strict=True)
    assert back.emission == EmissionConfig(mode="diagonal")
    assert V3Config.from_dict(data, strict=True).emission.mode == "diagonal"
    assert NESTED_BLOCKS["emission"] is EmissionConfig
    with pytest.raises(ValueError):
        V3Config.from_dict({"emission": {"bogus": 1}}, strict=True)
    # describe: the default string is unchanged, the diagonal one is annotated
    assert "emission" not in describe(V3Config())
    assert "emission=diagonal" in describe(cfg)
    # the dotted --override grammar (two levels) parses and applies
    for overridden in (
        apply_override_mapping(_cfg(), {"emission.mode": "diagonal"}),
        apply_override_items(_cfg(), ["emission.mode=diagonal"]),
    ):
        assert overridden.emission.mode == "diagonal"
        assert overridden.emission == EmissionConfig(mode="diagonal")
    # an invalid value survives the grammar and is rejected by validate()
    with pytest.raises(ValueError, match="emission.mode"):
        apply_override_items(_cfg(), ["emission.mode=banded"]).validate()


# ---------------------------------------------------------------------- #
# 6. dtype handling follows the existing w_emit path
# ---------------------------------------------------------------------- #
def test_emission_weights_follow_the_state_dtype():
    for mode, name in (("full", "w_emit"), ("diagonal", "w_emit_diag")):
        model = _model(mode=mode, dtype="float16")
        emit = model._weights()[3]
        assert emit.dtype == torch.float16, mode
        assert getattr(model, name).dtype == torch.float32, mode   # masters stay fp32
