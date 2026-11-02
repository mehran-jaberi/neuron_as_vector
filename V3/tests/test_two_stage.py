"""Focused tests for the optional second nonlinear mixing stage (M12).

CPU-only and tiny: no dataset, no GPU and no training.  These pin the feature's
four hard requirements:

1. the disabled path is bit-identical to today's single-mixer equation and adds
   no parameters,
2. ``w_down2`` / ``w_up2`` are drawn exactly like ``w_down`` / ``w_up`` (same
   generator, same scale) and not zeros or tiny values,
3. ``w_down2``, ``w_up2`` and ``lam_raw`` all take a non-zero *first* gradient
   on a small synthetic input, and one SGD step moves ``lam_raw`` (and hence the
   effective gain),
4. the enabled stage works under ``torch.utils.checkpoint`` (10 chunks): the
   checkpointed and direct forwards agree in value and in gradient,
5. the gain is parameterised as
   ``lam = lam_min + (lam_max - lam_min) * sigmoid(lam_raw)`` with
   ``0 < lam_min < lam_init < lam_max``: the initial value is ``lam_init``, the
   effective gain stays inside ``(lam_min, lam_max)`` for extreme raw values,
   and ``lam`` can never be trained to 0 or below (the old free scalar could,
   which silently switched the drive -- and the spikes -- off).
"""

from __future__ import annotations

import math
from unittest import mock

import pytest
import torch

import v3.model as model_module
from v3.config import (
    NESTED_BLOCKS,
    TimingConfig,
    TwoStageConfig,
    V3Config,
    apply_override_items,
    parameter_groups,
)
from v3.model import VectorNeuronPopulation, surrogate_spike

N, D, C, K, R = 4, 8, 6, 3, 2          # neurons, state dim, inputs, classes, base rank
B, T = 3, 4                             # batch, timesteps


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def _cfg(*, enabled: bool = False, rank: int = 64, lam_min: float = 0.1,
         lam_max: float = 2.0, lam_init: float = 1.0,
         dtype: str = "float32", chunks: int = 2, timesteps: int = T) -> V3Config:
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
    return cfg.with_overrides(
        two_stage=TwoStageConfig(
            enabled=enabled,
            rank=rank,
            lam_min=lam_min,
            lam_max=lam_max,
            lam_init=lam_init,
        )
    )


def _model(**kwargs) -> VectorNeuronPopulation:
    return VectorNeuronPopulation(_cfg(**kwargs))


def _binary_input(batch: int = B, timesteps: int = T, channels: int = C) -> torch.Tensor:
    torch.manual_seed(11)
    return torch.randint(0, 2, (batch, timesteps, channels)).float()


def _baseline_reference(model: VectorNeuronPopulation, x: torch.Tensor):
    """Today's single-mixer equation, re-implemented independently of ``_segment``."""
    dt = model.state_dtype
    w_in, w_down, w_up = model.w_in.to(dt), model.w_down.to(dt), model.w_up.to(dt)
    w_emit, w_rec = model.w_emit.to(dt), model.w_rec.to(dt)
    gate, bias = model.gate.to(dt), model.bias.to(dt)
    w_out, theta = model.w_out.to(dt), model.theta.to(dt)
    alpha = float(model.cfg.alpha)
    beta, gamma = float(model.cfg.surrogate_beta), float(model.cfg.surrogate_gamma)

    B_, T_, C_ = x.shape
    a = (x.reshape(B_ * T_, C_).to(dt) @ w_in).reshape(B_, T_, model.state_dim)
    z = model.init_state(B_, x.device)
    spikes = []
    for k in range(T_):
        pot = (z * w_out).sum(-1) + theta
        s = surrogate_spike(pot, beta, gamma)
        spikes.append(s)
        h = ((s.unsqueeze(-1) * z).sum(1) * model.pop_scale) @ w_emit
        c = s @ w_rec
        mix = (z @ w_down) @ w_up
        drive = gate * a[:, k].unsqueeze(1) + c.unsqueeze(-1) * h.unsqueeze(1) + bias
        z = z + alpha * (torch.tanh(mix + drive) - z)
        z = z.to(dt)
    spikes = torch.stack(spikes, dim=1)
    counts = spikes.float().sum(1)
    if model.cfg.readout == "mean":
        counts = counts / spikes.shape[1]
    logits = counts @ model.w_cls.float()
    return logits, spikes, z


# ---------------------------------------------------------------------- #
# 1. defaults and the disabled path
# ---------------------------------------------------------------------- #
def test_defaults_are_off():
    block = TwoStageConfig()
    assert block.enabled is False
    assert block.rank == 64
    assert block.lam_min == 0.1
    assert block.lam_max == 2.0
    assert block.lam_init == 1.0
    assert block.active is False
    # the default model must not enable the feature
    assert V3Config().two_stage.enabled is False
    assert V3Config().two_stage == block


def test_disabled_forward_matches_baseline_equation_bit_identically():
    torch.manual_seed(0)
    x = _binary_input()
    model = _model(enabled=False).eval()
    with torch.no_grad():
        logits, spikes, z_final, _ = model(x, return_states=True)
        ref_logits, ref_spikes, ref_z = _baseline_reference(model, x)
    # bit-identical: the disabled branch is the untouched expression
    assert torch.equal(z_final, ref_z)
    assert torch.equal(spikes, ref_spikes)
    assert torch.equal(logits, ref_logits)


def test_disabled_adds_no_parameters():
    base = _model(enabled=False)
    keys = list(dict(base.named_parameters()).keys())
    assert "w_down2" not in keys and "w_up2" not in keys and "lam_raw" not in keys
    assert base.w_down2 is None and base.w_up2 is None and base.lam_raw is None
    assert base.two_stage_weights() is None
    # the effective gain only exists together with the stage it belongs to
    with pytest.raises(RuntimeError, match="disabled"):
        base.effective_lam()
    # the analytic budget of the baseline config is exactly the parameter count,
    # so enabling the feature is the only thing that can add parameters
    assert base.n_parameters() == parameter_groups(_cfg(enabled=False))["total"]
    assert base.n_parameters() == parameter_groups(_cfg(enabled=True))["total"] - (
        2 * D * min(64, D) + 1
    )


# ---------------------------------------------------------------------- #
# 2. config plumbing
# ---------------------------------------------------------------------- #
def test_dotted_override_parses_two_stage():
    cfg = apply_override_items(
        _cfg(),
        [
            "two_stage.enabled=true",
            "two_stage.rank=8",
            "two_stage.lam_min=0.2",
            "two_stage.lam_max=1.5",
            "two_stage.lam_init=0.5",
        ],
    )
    assert cfg.two_stage.enabled is True
    assert cfg.two_stage.rank == 8
    assert cfg.two_stage.lam_min == pytest.approx(0.2)
    assert cfg.two_stage.lam_max == pytest.approx(1.5)
    assert cfg.two_stage.lam_init == pytest.approx(0.5)
    assert cfg.two_stage.active is True


def test_from_dict_strict_and_roundtrip():
    cfg = _cfg(enabled=True, rank=3, lam_init=0.25)
    data = cfg.to_dict()
    assert isinstance(data["two_stage"], dict)
    back = V3Config.from_dict(data, strict=True)
    assert back.two_stage == TwoStageConfig(enabled=True, rank=3, lam_init=0.25)
    assert "two_stage" in NESTED_BLOCKS
    assert NESTED_BLOCKS["two_stage"] is TwoStageConfig
    with pytest.raises(ValueError):
        V3Config.from_dict({"two_stage": {"bogus": 1}}, strict=True)


def test_validation_rejects_bad_two_stage():
    with pytest.raises(ValueError):
        V3Config(two_stage=TwoStageConfig(rank=0)).validate()
    # valid values stay valid, including a warm start close to an endpoint
    V3Config(two_stage=TwoStageConfig(enabled=True, rank=1)).validate()
    V3Config(
        two_stage=TwoStageConfig(
            enabled=True, rank=1, lam_min=0.15, lam_max=0.16, lam_init=0.155
        )
    ).validate()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lam_min": 0.0},
        {"lam_min": -0.5},
        {"lam_min": 1.0},                  # == lam_init
        {"lam_min": 1.5},                  # > lam_init
        {"lam_max": 1.0},                  # == lam_init
        {"lam_max": 0.5},                  # < lam_init
        {"lam_min": float("nan")},
        {"lam_init": float("nan")},
        {"lam_init": float("inf")},
        {"lam_max": float("-inf")},
    ],
)
def test_validation_rejects_bad_lam_bounds(kwargs):
    block = TwoStageConfig(enabled=True, **kwargs)
    with pytest.raises(ValueError):
        V3Config(two_stage=block).validate()


# ---------------------------------------------------------------------- #
# 3. parameter budget
# ---------------------------------------------------------------------- #
def test_parameter_groups_records_new_parameters():
    base = parameter_groups(_cfg(enabled=False))
    on = parameter_groups(_cfg(enabled=True, rank=64))  # R2 = min(64, D) = 8
    r2 = min(64, D)
    assert "w_down2" not in base and "w_up2" not in base and "lam_raw" not in base
    assert on["w_down2"] == D * r2
    assert on["w_up2"] == r2 * D
    assert on["lam_raw"] == 1
    assert on["total"] - base["total"] == 2 * D * r2 + 1


def test_parameter_budget_at_d1000_r64():
    big = V3Config(state_dim=1000, mix_rank=64)
    base = parameter_groups(big)
    on = parameter_groups(big.with_overrides(two_stage=TwoStageConfig(enabled=True)))
    assert base["total"] == 2_025_440
    assert on["w_down2"] == 64_000
    assert on["w_up2"] == 64_000
    assert on["lam_raw"] == 1
    assert on["total"] - base["total"] == 128_001
    assert on["total"] == 2_153_441


# ---------------------------------------------------------------------- #
# 4. initialisation: exactly the same draw as w_down / w_up
# ---------------------------------------------------------------------- #
def test_w_down2_and_w_up2_are_drawn_exactly_like_w_down():
    cfg = _cfg(enabled=True, rank=64)
    model = VectorNeuronPopulation(cfg)
    assert model.w_down2.shape == (D, D) and model.w_up2.shape == (D, D)  # R2 capped at D

    # replay the generator stream in __init__ order; the replay is validated by
    # reproducing the *existing* parameters first, so this test is exact and not
    # a statistical check.
    g = torch.Generator().manual_seed(int(cfg.seed) + 12345)

    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).to(torch.float32)

    n_inputs = cfg.n_inputs
    w_in = rnd(C, D, scale=5.0 / math.sqrt(C))
    w_down = rnd(D, R, scale=1.0 / math.sqrt(D))
    w_up = rnd(R, D, scale=0.5 / math.sqrt(R))
    w_emit = torch.eye(D, dtype=torch.float32) + rnd(D, D, scale=0.02)
    w_rec = rnd(N, N, scale=1.0 / math.sqrt(N))
    gate = torch.ones(N, D, dtype=torch.float32) + rnd(N, D, scale=0.05)
    w_out = rnd(N, D, scale=1.0 / math.sqrt(D))
    r2 = min(int(cfg.two_stage.rank), D)
    exp_down2 = rnd(D, r2, scale=1.0 / math.sqrt(D))
    exp_up2 = rnd(r2, D, scale=0.5 / math.sqrt(r2))

    assert n_inputs == C  # replay hard-codes the same widths as _cfg
    assert torch.equal(model.w_in.detach(), w_in)
    assert torch.equal(model.w_down.detach(), w_down)
    assert torch.equal(model.w_up.detach(), w_up)
    assert torch.equal(model.w_emit.detach(), w_emit)
    assert torch.equal(model.w_rec.detach(), w_rec)
    assert torch.equal(model.gate.detach(), gate)
    assert torch.equal(model.w_out.detach(), w_out)
    # ... and the new stage continues the same stream with the same scheme/scale
    assert torch.equal(model.w_down2.detach(), exp_down2)
    assert torch.equal(model.w_up2.detach(), exp_up2)
    assert model.w_down2.dtype == torch.float32 and model.w_up2.dtype == torch.float32
    assert model.lam_raw.shape == () and model.lam_raw.dtype == torch.float32
    assert float(model.effective_lam().detach()) == pytest.approx(1.0)
    assert (
        model.lam_raw.requires_grad
        and model.w_down2.requires_grad
        and model.w_up2.requires_grad
    )


def test_rank_is_capped_by_state_dim_and_lam_raw_is_scalar():
    for rank in (1, 3, D, 64):
        model = _model(enabled=True, rank=rank)
        r2 = min(rank, D)
        assert model.w_down2.shape == (D, r2)
        assert model.w_up2.shape == (r2, D)
        assert model.lam_raw.shape == ()
        assert model.effective_lam().shape == ()
        assert float(model.effective_lam().detach()) == pytest.approx(1.0)


# ---------------------------------------------------------------------- #
# 4b. bounded gain: lam = lam_min + (lam_max - lam_min) * sigmoid(lam_raw)
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("raw", [-100.0, -5.0, 0.0, 5.0, 100.0])
def test_effective_lam_stays_within_bounds(raw):
    model = _model(enabled=True).eval()
    lo = float(model.cfg.two_stage.lam_min)
    hi = float(model.cfg.two_stage.lam_max)
    with torch.no_grad():
        model.lam_raw.fill_(raw)
    lam = model.effective_lam().detach()
    assert lam.dtype == torch.float32 and lam.shape == ()
    assert bool(torch.isfinite(lam))
    # never below the floor and never above the ceiling, whatever the optimiser
    # does to the raw parameter
    assert lo <= float(lam) <= hi
    # fp32 sigmoid saturates to exactly 0/1 at |raw| = 100, so the strict
    # interior is only representable for moderate raw values
    if abs(raw) < 100.0:
        assert lo < float(lam) < hi


@pytest.mark.parametrize("lam_init", [0.15, 0.5, 1.0, 1.9])
def test_effective_lam_at_init_equals_lam_init(lam_init):
    model = _model(enabled=True, lam_init=lam_init)
    lam = float(model.effective_lam().detach())
    assert lam == pytest.approx(lam_init, rel=0.0, abs=1e-6)
    assert model.cfg.two_stage.lam_min < lam < model.cfg.two_stage.lam_max


def test_effective_lam_is_monotone_in_lam_raw():
    model = _model(enabled=True)
    values = []
    with torch.no_grad():
        for raw in (-10.0, -5.0, 0.0, 5.0, 10.0):
            model.lam_raw.fill_(raw)
            values.append(float(model.effective_lam()))
    assert values == sorted(values)
    assert len(set(values)) == len(values)


def test_effective_lam_is_fp32_on_an_fp16_model():
    model = _model(enabled=True, dtype="float16")
    assert model.effective_lam().dtype == torch.float32
    assert model.two_stage_weights()[2].dtype == torch.float16


# ---------------------------------------------------------------------- #
# 5. first gradient and lam_raw moving
# ---------------------------------------------------------------------- #
def test_first_backward_gives_nonzero_grads_and_moves_lam_raw():
    torch.manual_seed(3)
    x = torch.randint(0, 2, (B, T, C)).float()
    model = _model(enabled=True)
    model.train()
    model.zero_grad(set_to_none=True)
    _, spikes = model(x)
    # a small, self-contained scalar objective: spike count
    spikes.float().sum().backward()

    for name in ("w_down2", "w_up2"):
        grad = getattr(model, name).grad
        assert grad is not None, name
        assert torch.isfinite(grad).all(), name
        assert int(torch.count_nonzero(grad)) > 0, name
        assert int(torch.count_nonzero(grad)) >= grad.numel() // 2, name
        assert float(grad.abs().max()) > 1e-6, name
    assert model.lam_raw.grad is not None
    assert torch.isfinite(model.lam_raw.grad).all()
    assert float(model.lam_raw.grad.abs().sum()) > 0.0  # not trapped at lam = 1.0

    # one plain SGD step must move lam_raw, and the effective gain with it
    before = float(model.lam_raw.detach())
    lam_before = float(model.effective_lam().detach())
    with torch.no_grad():
        model.lam_raw -= 0.1 * model.lam_raw.grad
    after = float(model.lam_raw.detach())
    lam_after = float(model.effective_lam().detach())
    assert after != before
    assert lam_after != lam_before
    assert abs(lam_after - 1.0) > 1e-4
    assert model.cfg.two_stage.lam_min < lam_after < model.cfg.two_stage.lam_max


def test_gradient_flows_to_lam_raw_and_sgd_step_moves_it():
    torch.manual_seed(7)
    x = torch.randint(0, 2, (B, T, C)).float()
    model = _model(enabled=True)
    model.train()
    before = float(model.lam_raw.detach())
    lam_before = float(model.effective_lam().detach())
    _, spikes = model(x)
    spikes.float().sum().backward()
    grad = model.lam_raw.grad
    assert grad is not None and grad.shape == ()
    assert bool(torch.isfinite(grad))
    assert float(grad.abs()) > 0.0
    opt = torch.optim.SGD([model.lam_raw], lr=0.5)
    opt.step()
    assert float(model.lam_raw.detach()) != before
    lam_after = float(model.effective_lam().detach())
    assert lam_after != lam_before
    assert model.cfg.two_stage.lam_min < lam_after < model.cfg.two_stage.lam_max


def test_lam_changes_the_forward():
    torch.manual_seed(0)
    x = _binary_input()
    model = _model(enabled=True).eval()
    with torch.no_grad():
        _, _, z1, _ = model(x, return_states=True)
        model.lam_raw.add_(2.0)   # effective gain rises, still bounded
        _, _, z2, _ = model(x, return_states=True)
    assert not torch.equal(z1, z2)


def test_one_step_closed_form():
    """At T=1 the first state is exactly alpha * tanh(lam * tanh(gate * a))."""
    torch.manual_seed(5)
    x = torch.randint(0, 2, (B, 1, C)).float()
    model = _model(enabled=True).eval()
    with torch.no_grad():
        _, _, z1, _ = model(x, return_states=True)
        a = x[:, 0] @ model.w_in
        expected = float(model.cfg.alpha) * torch.tanh(
            float(model.effective_lam().detach())
            * torch.tanh(model.gate[None] * a.unsqueeze(1))
        )
    assert z1.shape == (B, N, D)
    assert torch.allclose(z1, expected, atol=1e-7, rtol=0.0)


def test_two_stage_weights_are_cast_to_state_dtype():
    model = _model(enabled=True, dtype="float16")
    weights = model.two_stage_weights()
    assert weights is not None
    w_down2, w_up2, lam = weights
    assert w_down2.dtype == torch.float16 and w_up2.dtype == torch.float16
    assert lam.dtype == torch.float16
    # the effective gain stays fp32 (sigmoid of the master raw parameter)
    assert float(model.effective_lam().detach()) == pytest.approx(
        float(lam.detach().float()), abs=1e-3
    )
    # master weights stay fp32
    assert model.w_down2.dtype == torch.float32 and model.lam_raw.dtype == torch.float32


def test_fp16_forward_and_backward_are_finite():
    torch.manual_seed(0)
    x = _binary_input()
    model = _model(enabled=True, dtype="float16")
    model.train()
    logits, spikes = model(x)
    assert torch.isfinite(logits).all()
    spikes.float().sum().backward()
    for name in ("w_down2", "w_up2", "lam_raw"):
        grad = getattr(model, name).grad
        assert grad is not None and torch.isfinite(grad).all(), name
        assert float(grad.float().abs().sum()) > 0.0, name


# ---------------------------------------------------------------------- #
# 6. gradient checkpointing (10 chunks)
# ---------------------------------------------------------------------- #
def _run_direct_or_checkpointed(model, x, use_checkpoint: bool):
    real_checkpoint = model_module.checkpoint
    calls = {"n": 0}

    def counting_checkpoint(*args, **kwargs):
        calls["n"] += 1
        return real_checkpoint(*args, **kwargs)

    with mock.patch.object(model_module, "checkpoint", counting_checkpoint), mock.patch.object(
        model_module, "checkpoint_supported", lambda z: use_checkpoint
    ):
        model.zero_grad(set_to_none=True)
        logits, spikes = model(x)
        loss = logits.pow(2).sum() + 1e-3 * spikes.float().sum()
        loss.backward()
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    return logits.detach().clone(), spikes.detach().clone(), grads, calls["n"]


@pytest.mark.parametrize("enabled", [False, True])
def test_checkpointed_matches_direct_forward_and_gradient(enabled):
    torch.manual_seed(0)
    x = torch.randint(0, 2, (B, 10, C)).float()          # T = 10 ...
    model = _model(enabled=enabled, chunks=10, timesteps=10)  # ... 10 chunks of 1 step
    model.train()

    d_logits, d_spikes, d_grads, d_calls = _run_direct_or_checkpointed(model, x, False)
    assert d_calls == 0
    c_logits, c_spikes, c_grads, c_calls = _run_direct_or_checkpointed(model, x, True)
    assert c_calls == 10  # one checkpoint call per chunk

    assert torch.equal(d_logits, c_logits)
    assert torch.equal(d_spikes, c_spikes)
    assert set(d_grads) == set(c_grads)
    for name in d_grads:
        assert torch.allclose(d_grads[name], c_grads[name], atol=1e-6, rtol=1e-5), name
    if enabled:
        for name in ("w_down2", "w_up2", "lam_raw"):
            assert int(torch.count_nonzero(c_grads[name])) > 0, name
