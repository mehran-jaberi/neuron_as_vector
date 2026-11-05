"""Focused tests for the two-timescale adaptation currents (M8, adLIF-style).

CPU-only and tiny: configuration surface, inert zero-gain initialisation, the
timestep dynamics on a hand-built input whose expected pre-activation is
computable in closed form, per-neuron gain gradients, the per-batch state reset,
and the gradient-checkpointed chunk path.

Baseline guarantee: with ``adlif.enabled = False`` no parameter is created and
the update rule is untouched, so a default model is the bit-identical reference
for the zero-gain enabled path.
"""

from __future__ import annotations

import pytest
import torch

import v3.model as model_module
from v3.config import (
    AdlifConfig,
    NESTED_BLOCKS,
    TimingConfig,
    V3Config,
    parameter_groups,
)
from v3.model import VectorNeuronPopulation

N, D, C, K, R = 6, 8, 3, 4, 4
T_MAX = 10  # n_bins of the tiny timing block; tests use <= T_MAX steps


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def small_config(adlif: AdlifConfig | None = None) -> V3Config:
    cfg = V3Config(
        n_neurons=N,
        state_dim=D,
        mix_rank=R,
        n_inputs=C,
        n_classes=K,
        dtype="float32",
        device="cpu",
        grad_checkpoint_chunks=2,
        timing=TimingConfig(sequence_duration_ms=float(T_MAX * 2), time_bin_ms=2.0),
        adlif=AdlifConfig() if adlif is None else adlif,
    )
    cfg.validate()
    return cfg


def make_model(enabled: bool = False, **adlif_overrides) -> VectorNeuronPopulation:
    """Tiny model; identical weights for every call because init uses a seeded generator."""
    cfg = small_config(AdlifConfig(enabled=enabled, **adlif_overrides))
    return VectorNeuronPopulation(cfg)


def zero_weights(model: VectorNeuronPopulation) -> None:
    with torch.no_grad():
        for name in ("w_in", "w_down", "w_up", "w_emit", "w_rec", "gate", "bias", "w_out", "theta", "w_cls"):
            getattr(model, name).zero_()


def set_gains(model: VectorNeuronPopulation, fast: float, slow: float) -> None:
    with torch.no_grad():
        if model.adlif_gain_fast is not None:
            model.adlif_gain_fast.fill_(fast)
        if model.adlif_gain_slow is not None:
            model.adlif_gain_slow.fill_(slow)


def spike_loss(model: VectorNeuronPopulation, x: torch.Tensor) -> torch.Tensor:
    """A loss that reaches every spike: the default zero w_cls cannot."""
    with torch.no_grad():
        model.w_cls.normal_(0.0, 0.1)
        model.theta.fill_(1.0)  # positive from t = 0, so every neuron fires
    logits, spikes = model(x)[:2]
    targets = torch.arange(x.shape[0]) % K
    return torch.nn.functional.cross_entropy(logits.float(), targets) + 1e-3 * spikes.float().sum()


def conditioned_model(enabled: bool = True, **adlif_overrides) -> VectorNeuronPopulation:
    """Zero mixing/drive so the tanh sits at 0 and the gain grad is well conditioned.

    ``theta = 1`` makes every neuron spike from ``t = 0`` (so ``u(t)`` grows from
    step 1), while ``mix = drive = 0`` keeps ``pre = -g u`` in the linear regime of
    the tanh, where ``d tanh(0) = 1``.
    """
    model = make_model(enabled=enabled, **adlif_overrides)
    zero_weights(model)
    with torch.no_grad():
        model.theta.fill_(1.0)
        model.w_out.normal_(0.0, 0.1)
        model.w_cls.normal_(0.0, 0.1)
    return model


# ---------------------------------------------------------------------- #
# configuration surface
# ---------------------------------------------------------------------- #
def test_default_block_is_off_with_the_documented_defaults():
    adlif = AdlifConfig()
    assert adlif.enabled is False
    assert adlif.use_fast is True and adlif.use_slow is True
    assert adlif.rho_fast == 0.9 and adlif.rho_slow == 0.99
    assert adlif.gain_init == 0.0
    adlif.validate()

    cfg = V3Config()
    assert cfg.adlif.enabled is False
    assert cfg.to_dict()["adlif"]["enabled"] is False
    assert NESTED_BLOCKS["adlif"] is AdlifConfig


def test_validation_rejects_unusable_settings():
    with pytest.raises(ValueError, match="at least|use_fast"):
        AdlifConfig(enabled=True, use_fast=False, use_slow=False).validate()
    for rho in (1.0, -0.05, float("nan")):
        with pytest.raises(ValueError, match="rho_fast"):
            AdlifConfig(rho_fast=rho).validate()
        with pytest.raises(ValueError, match="rho_slow"):
            AdlifConfig(rho_slow=rho).validate()
    with pytest.raises(ValueError, match="gain_init"):
        AdlifConfig(gain_init=float("inf")).validate()


def test_nested_block_roundtrips_through_from_dict():
    cfg = V3Config.from_dict(
        {"adlif": {"enabled": True, "gain_init": 0.25, "use_slow": False}}
    )
    assert isinstance(cfg.adlif, AdlifConfig)
    assert cfg.adlif.enabled is True
    assert cfg.adlif.gain_init == 0.25
    assert cfg.adlif.use_slow is False
    assert cfg.adlif.use_fast is True

    with pytest.raises(ValueError, match="unknown adlif keys"):
        AdlifConfig.from_dict({"nope": 1}, strict=True)


# ---------------------------------------------------------------------- #
# parameter budget
# ---------------------------------------------------------------------- #
def test_off_path_adds_no_parameter():
    model = make_model()
    assert model.adlif_enabled is False
    assert model.adlif_use_fast is False
    assert model.adlif_use_slow is False
    assert model.adlif_gain_fast is None
    assert model.adlif_gain_slow is None
    assert model.init_adaptation(3, torch.device("cpu")) == (None, None)
    assert not [n for n, _ in model.named_parameters() if "adlif" in n]
    assert not [k for k in model.state_dict() if "adlif" in k]
    groups = parameter_groups(model.cfg)
    assert "adlif_gain_fast" not in groups
    assert groups["total"] == model.n_parameters()


def test_enabled_parameters_exist_only_for_active_timescales():
    both = make_model(enabled=True)
    params = dict(both.named_parameters())
    assert params["adlif_gain_fast"].shape == (N,)
    assert params["adlif_gain_slow"].shape == (N,)
    assert torch.equal(params["adlif_gain_fast"], torch.zeros(N))
    groups = parameter_groups(both.cfg)
    assert groups["adlif_gain_fast"] == N
    assert groups["adlif_gain_slow"] == N
    assert groups["total"] == both.n_parameters()

    fast_only = make_model(enabled=True, use_slow=False)
    names = dict(fast_only.named_parameters())
    assert "adlif_gain_fast" in names and "adlif_gain_slow" not in names
    assert fast_only.adlif_gain_slow is None
    assert parameter_groups(fast_only.cfg)["total"] == fast_only.n_parameters()

    slow_only = make_model(enabled=True, use_fast=False)
    names = dict(slow_only.named_parameters())
    assert "adlif_gain_fast" not in names and "adlif_gain_slow" in names
    assert slow_only.adlif_gain_fast is None
    assert parameter_groups(slow_only.cfg)["total"] == slow_only.n_parameters()


# ---------------------------------------------------------------------- #
# zero-gain initialisation is exactly inert
# ---------------------------------------------------------------------- #
def test_zero_gain_enabled_path_is_bit_identical_to_the_baseline():
    x = torch.randn(2, T_MAX, C)
    baseline = make_model()
    enabled = make_model(enabled=True)
    baseline.eval()
    enabled.eval()
    with torch.no_grad():
        base_logits, base_spikes = baseline(x)
        new_logits, new_spikes = enabled(x)
    assert torch.equal(base_logits, new_logits)
    assert torch.equal(base_spikes, new_spikes)


def test_segment_twin_is_bit_identical_at_zero_gain_but_currents_move():
    model = make_model(enabled=True)
    baseline = make_model()
    model.eval()
    baseline.eval()
    z = torch.randn(2, N, D)
    a = torch.randn(2, 4, D)  # already projected: (B, steps, D)
    zeros = torch.zeros(2, N)
    with torch.no_grad():
        z0, s0, p0 = baseline._segment(z, a)
        z1, uf, us, s1, p1 = model._segment_adlif(z, a, zeros, zeros)
    assert torch.equal(z0, z1)
    assert torch.equal(s0, s1)
    assert torch.equal(p0, p1)
    # the currents accumulate from the spikes even though the gains are exactly 0
    assert torch.count_nonzero(uf) > 0
    assert torch.count_nonzero(us) > 0


# ---------------------------------------------------------------------- #
# hand-computed two-timescale dynamics
# ---------------------------------------------------------------------- #
def test_two_timescale_dynamics_match_the_hand_computed_pre_activation():
    """Zero the mixing/drive so ``pre_k = -g_f u_f(k) - g_s u_s(k)`` exactly."""
    model = make_model(enabled=True)
    zero_weights(model)
    set_gains(model, 0.5, 0.3)
    with torch.no_grad():
        model.theta.fill_(3.0)  # pot = 3 > 0 at every step -> s(t) = 1

    B, T = 2, 8
    rho_f, rho_s = model.cfg.adlif.rho_fast, model.cfg.adlif.rho_slow
    x = torch.zeros(B, T, C)
    a_chunk = torch.zeros(B, T, D)  # the projected drive is zero too
    z0 = torch.zeros(B, N, D)

    model.eval()
    with torch.no_grad():
        # returned u(T) = sum_{j=0}^{T-1} rho^j * 1 = (1 - rho^T) / (1 - rho)
        z1, uf, us, spikes, _ = model._segment_adlif(
            z0, a_chunk, torch.zeros(B, N), torch.zeros(B, N)
        )
        uf_expected = (1.0 - rho_f**T) / (1.0 - rho_f)
        us_expected = (1.0 - rho_s**T) / (1.0 - rho_s)
        assert torch.allclose(uf, torch.full_like(uf, uf_expected), atol=1e-6)
        assert torch.allclose(us, torch.full_like(us, us_expected), atol=1e-6)
        assert torch.equal(spikes, torch.ones_like(spikes))

        # and the state trajectory from the closed-form pre-activation:
        #   z_{k+1} = z_k + alpha (tanh(-g_f u_f(k) - g_s u_s(k)) - z_k)
        alpha = float(model.cfg.alpha)
        z_expected = torch.zeros(B, N, D)
        for k in range(T):
            uf_k = (1.0 - rho_f**k) / (1.0 - rho_f)
            us_k = (1.0 - rho_s**k) / (1.0 - rho_s)
            pre_k = -0.5 * uf_k - 0.3 * us_k
            z_expected = z_expected + alpha * (torch.tanh(torch.tensor(pre_k)) - z_expected)
        assert torch.allclose(z1, z_expected, atol=1e-6, rtol=0.0)

        # forward() threads the same currents through the checkpointed chunks and
        # must land on the same final state (chunks = 2, so the second chunk starts
        # from the current advanced by the first five steps).
        _, out_spikes, z_final, _ = model(x, return_states=True)
        assert torch.equal(out_spikes, torch.ones_like(out_spikes))
        assert torch.allclose(z_final, z_expected, atol=1e-6, rtol=0.0)


def test_reference_loop_with_random_weights_and_nonzero_gains():
    """Independent re-implementation of the documented equations."""
    model = make_model(enabled=True)
    set_gains(model, 0.2, 0.35)
    model.eval()
    torch.manual_seed(7)
    x = torch.randn(2, T_MAX, C)
    B, T = x.shape[0], x.shape[1]
    cfg = model.cfg
    w_in, w_down, w_up, w_emit, w_rec, gate, bias, w_out, theta, _, _ = model._weights()
    alpha = float(cfg.alpha)
    beta, gamma = float(cfg.surrogate_beta), float(cfg.surrogate_gamma)

    z = torch.zeros(B, N, D)
    uf = torch.zeros(B, N)
    us = torch.zeros(B, N)
    ref_spikes = []
    with torch.no_grad():
        a = x @ w_in
        g_f, g_s = model.adlif_gain_fast, model.adlif_gain_slow
        for k in range(T):
            pot = (z * w_out).sum(-1) + theta
            s = model_module.surrogate_spike(pot, beta, gamma)
            ref_spikes.append(s)
            h = ((s.unsqueeze(-1) * z).sum(1) * model.pop_scale) @ w_emit
            c = s @ w_rec
            drive = gate * a[:, k].unsqueeze(1) + c.unsqueeze(-1) * h.unsqueeze(1) + bias
            pre = (z @ w_down) @ w_up + drive
            pre = pre - (g_f * uf).unsqueeze(-1) - (g_s * us).unsqueeze(-1)
            z = z + alpha * (torch.tanh(pre) - z)
            uf = cfg.adlif.rho_fast * uf + s
            us = cfg.adlif.rho_slow * us + s
        _, spikes_out, z_final, _ = model(x, return_states=True)
    assert torch.equal(spikes_out, torch.stack(ref_spikes, dim=1))
    assert torch.allclose(z_final, z, atol=1e-6, rtol=0.0)


# ---------------------------------------------------------------------- #
# gradients
# ---------------------------------------------------------------------- #
def test_gains_receive_nonzero_gradient_at_zero_init():
    model = conditioned_model(enabled=True)
    model.eval()
    x = torch.randn(4, T_MAX, C)
    loss = spike_loss(model, x)
    loss.backward()
    for name in ("adlif_gain_fast", "adlif_gain_slow"):
        gain = getattr(model, name)
        assert gain.grad is not None
        assert torch.isfinite(gain.grad).all()
        assert torch.count_nonzero(gain.grad) > 0, name


def test_two_steps_are_gradient_dead_so_the_test_uses_more_steps():
    """Documents the M1 trap: u(0) = 0 and pre(T-1) is never read back."""
    model = conditioned_model(enabled=True, gain_init=0.3)
    model.eval()
    loss = spike_loss(model, torch.randn(2, 2, C))
    loss.backward()
    for name in ("adlif_gain_fast", "adlif_gain_slow"):
        gain = getattr(model, name)
        assert gain.grad is None or torch.count_nonzero(gain.grad) == 0


def test_gain_gradient_flows_through_single_timescale_variants():
    for kwargs in ({"use_slow": False}, {"use_fast": False}):
        model = conditioned_model(enabled=True, **kwargs)
        model.eval()
        loss = spike_loss(model, torch.randn(2, T_MAX, C))
        loss.backward()
        for name in ("adlif_gain_fast", "adlif_gain_slow"):
            gain = getattr(model, name)
            if gain is None:
                continue
            assert torch.count_nonzero(gain.grad) > 0, (kwargs, name)


# ---------------------------------------------------------------------- #
# per-batch state reset
# ---------------------------------------------------------------------- #
def test_state_is_reset_between_batches():
    model = make_model(enabled=True)
    model.eval()
    torch.manual_seed(3)
    x1 = torch.randn(2, T_MAX, C)
    x2 = torch.randn(3, T_MAX, C)
    with torch.no_grad():
        logits_a, spikes_a = model(x1)
        model(x2)  # must not leak its currents into the next batch
        logits_b, spikes_b = model(x1)
    assert torch.equal(logits_a, logits_b)
    assert torch.equal(spikes_a, spikes_b)


def test_init_adaptation_is_zero_and_per_batch():
    model = make_model(enabled=True)
    uf, us = model.init_adaptation(5, torch.device("cpu"))
    assert uf.shape == (5, N) and us.shape == (5, N)
    assert uf.dtype == model.state_dtype and us.dtype == model.state_dtype
    assert torch.count_nonzero(uf) == 0 and torch.count_nonzero(us) == 0
    assert uf.data_ptr() != us.data_ptr()

    fast_only = make_model(enabled=True, use_slow=False)
    uf, us = fast_only.init_adaptation(2, torch.device("cpu"))
    assert uf is not None and us is None

    slow_only = make_model(enabled=True, use_fast=False)
    uf, us = slow_only.init_adaptation(2, torch.device("cpu"))
    assert uf is None and us is not None

    off = make_model()
    assert off.init_adaptation(2, torch.device("cpu")) == (None, None)


# ---------------------------------------------------------------------- #
# gradient checkpointing
# ---------------------------------------------------------------------- #
def test_checkpointed_chunks_match_the_direct_path(monkeypatch):
    model = conditioned_model(enabled=True)
    model.train()
    torch.manual_seed(11)
    x = torch.randn(2, T_MAX, C)

    real_checkpoint = model_module.checkpoint
    calls = {"n": 0}

    def counting_checkpoint(*args, **kwargs):
        calls["n"] += 1
        return real_checkpoint(*args, **kwargs)

    def run(use_checkpointing: bool):
        monkeypatch.setattr(model_module, "checkpoint", counting_checkpoint)
        monkeypatch.setattr(model_module, "checkpoint_supported", lambda z: use_checkpointing)
        model.zero_grad(set_to_none=True)
        logits, spikes = model(x)[:2]
        loss = logits.pow(2).sum() + 1e-3 * spikes.float().sum()
        loss.backward()
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
        return logits.detach(), spikes.detach(), grads

    direct_out, direct_spikes, direct_grads = run(False)
    assert calls["n"] == 0
    ckpt_out, ckpt_spikes, ckpt_grads = run(True)
    assert calls["n"] == 2  # T = 10 over 2 chunks

    assert torch.equal(direct_out, ckpt_out)
    assert torch.equal(direct_spikes, ckpt_spikes)
    assert set(direct_grads) == set(ckpt_grads)
    for name in direct_grads:
        assert torch.allclose(direct_grads[name], ckpt_grads[name], atol=1e-6, rtol=1e-5), name
    # the gains are still trained through the checkpointed path
    assert torch.count_nonzero(ckpt_grads["adlif_gain_fast"]) > 0
    assert torch.count_nonzero(ckpt_grads["adlif_gain_slow"]) > 0


def test_checkpointed_single_timescale_chunk_path_runs(monkeypatch):
    model = conditioned_model(enabled=True, use_fast=False)
    model.train()
    monkeypatch.setattr(model_module, "checkpoint_supported", lambda z: True)
    logits, spikes = model(torch.randn(2, T_MAX, C))[:2]
    loss = logits.pow(2).sum() + 1e-3 * spikes.float().sum()
    loss.backward()
    assert torch.count_nonzero(model.adlif_gain_slow.grad) > 0


# ---------------------------------------------------------------------- #
# wiring: the off path never enters the new code
# ---------------------------------------------------------------------- #
def test_off_path_never_runs_the_adlif_currents():
    """The adLIF currents exist only when the feature is on.

    After the M1/M8 loop merge both configurations share :meth:`_segment_any`;
    what must differ is the *state* handed to it -- zero-length placeholders
    when adLIF is off, live ``(B, N)`` currents when it is on.
    """
    model = make_model()
    model.eval()
    seen = {"n": 0}
    original = model._segment_any

    def counting(z, a, b=None, u_fast=None, u_slow=None, state_sum=None, cur=None, **kwargs):
        assert u_fast is not None and u_fast.numel() == 0
        assert u_slow is not None and u_slow.numel() == 0
        seen["n"] += 1
        return original(z, a, b, u_fast, u_slow, state_sum, cur, **kwargs)

    model._segment_any = counting
    with torch.no_grad():
        model(torch.randn(2, 4, C))
    assert seen["n"] > 0

    enabled = make_model(enabled=True)
    enabled.eval()
    live = {"fast": 0, "slow": 0}
    base = enabled._segment_any

    def watching(z, a, b=None, u_fast=None, u_slow=None, state_sum=None, cur=None):
        live["fast"] += int(u_fast is not None and u_fast.numel() > 0)
        live["slow"] += int(u_slow is not None and u_slow.numel() > 0)
        return base(z, a, b, u_fast, u_slow, state_sum, cur)

    enabled._segment_any = watching
    with torch.no_grad():
        enabled(torch.randn(2, 4, C))
    assert live["fast"] > 0 and live["slow"] > 0
