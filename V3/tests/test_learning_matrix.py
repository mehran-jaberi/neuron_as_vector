"""Focused tests for the learning-matrix emission and its R2 readout.

CPU-only and tiny: no dataset, no GPU and no training.  These pin the feature's
hard requirements:

1. the block is off by default, and a disabled build is bit-identical -- same
   parameters, same ``state_dict``, same logits and spikes -- to a build whose
   config never mentions it,
2. enabling it registers ``v_matrix`` ``(D, K)`` and ``a_time`` ``(T, K)``, and
   neither the emission map ``w_emit`` nor the count classifier ``w_cls``, while
   every other parameter keeps its baseline init draw,
3. the budget arithmetic: ``D*K + T*K`` replace ``D*D + N*K``,
4. the learning-matrix row really is the pooled state (``L_t = h_t = pooled_t``):
   the logits equal a hand-computed ``sum_t a[t, k] <L_t[b, :], v[:, k]>`` and the
   state trajectory is unchanged,
5. gradients reach both new parameters,
6. the per-chunk partials sum to the single-pass logits, including through
   ``torch.utils.checkpoint``,
7. the dotted override ``learning_matrix.enabled=true`` parses.
"""

from __future__ import annotations

from unittest import mock

import pytest
import torch

import v3.model as model_module
from v3.config import (
    NESTED_BLOCKS,
    LearningMatrixConfig,
    TimingConfig,
    V3Config,
    apply_override_items,
    describe,
    parameter_groups,
)
from v3.model import VectorNeuronPopulation, surrogate_spike

N, D, C, K, R = 4, 8, 6, 3, 2      # neurons, state dim, inputs, classes, base rank
B, T = 3, 4                         # batch, timesteps


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def _cfg(*, enabled: bool = False, timesteps: int = T, chunks: int = 2) -> V3Config:
    cfg = V3Config(
        n_neurons=N,
        state_dim=D,
        mix_rank=R,
        n_inputs=C,
        n_classes=K,
        timing=TimingConfig(sequence_duration_ms=2.0 * timesteps, time_bin_ms=2.0),
        batch_size=B,
        device="cpu",
        dtype="float32",
        amp=False,
        grad_checkpoint_chunks=chunks,
    )
    if not enabled:
        return cfg
    return cfg.with_overrides(learning_matrix=LearningMatrixConfig(enabled=True))


def _model(**kwargs) -> VectorNeuronPopulation:
    return VectorNeuronPopulation(_cfg(**kwargs))


def _binary_input(batch: int = B, timesteps: int = T, channels: int = C) -> torch.Tensor:
    torch.manual_seed(11)
    return torch.randint(0, 2, (batch, timesteps, channels)).float()


def _r2_reference(model: VectorNeuronPopulation, x: torch.Tensor):
    """The documented R2 equations, re-implemented independently of ``_segment``.

    With no emission map the pooled state *is* the emitted row, ``L_t = h_t``, so
    the learning matrix is simply the sequence of population signals and the
    classifier reads it over time.  Returns ``(logits, spikes, z_final, L)``.
    """
    dt = model.state_dtype
    w_in, w_down, w_up = model.w_in.to(dt), model.w_down.to(dt), model.w_up.to(dt)
    w_rec, gate, bias = model.w_rec.to(dt), model.gate.to(dt), model.bias.to(dt)
    w_out, theta = model.w_out.to(dt), model.theta.to(dt)
    alpha = float(model.cfg.alpha)
    beta, gamma = float(model.cfg.surrogate_beta), float(model.cfg.surrogate_gamma)
    kind, s_alpha = str(model.cfg.surrogate), float(model.cfg.surrogate_alpha)

    B_, T_, C_ = x.shape
    a = (x.reshape(B_ * T_, C_).to(dt) @ w_in).reshape(B_, T_, model.state_dim)
    z = model.init_state(B_, x.device)
    spikes, rows = [], []
    for k in range(T_):
        pot = (z * w_out).sum(-1) + theta
        s = surrogate_spike(pot, beta, gamma, kind=kind, alpha=s_alpha)
        spikes.append(s)
        h = (s.unsqueeze(-1) * z).sum(1) * model.pop_scale       # L_t: no emission map
        rows.append(h.float())
        c = s @ w_rec
        mix = (z @ w_down) @ w_up
        drive = gate * a[:, k].unsqueeze(1) + c.unsqueeze(-1) * h.unsqueeze(1) + bias
        z = z + alpha * (torch.tanh(mix + drive) - z)
        z = z.to(dt)
    L = torch.stack(rows, dim=1)                                 # (B, T, D) learning matrix
    a_time = model.a_time[:T_].float()                           # (T, K)
    logits = torch.einsum("tk,btd,dk->bk", a_time, L, model.v_matrix.float())
    return logits, torch.stack(spikes, dim=1), z, L


# ---------------------------------------------------------------------- #
# 1. defaults, config plumbing and the disabled path
# ---------------------------------------------------------------------- #
def test_defaults_are_off():
    block = LearningMatrixConfig()
    assert block.enabled is False
    assert block.readout == "r2"
    assert V3Config().learning_matrix.enabled is False
    assert V3Config().learning_matrix == block
    assert "learning_matrix" in NESTED_BLOCKS
    assert NESTED_BLOCKS["learning_matrix"] is LearningMatrixConfig


def test_readout_is_validated():
    V3Config(learning_matrix=LearningMatrixConfig(enabled=True)).validate()
    with pytest.raises(ValueError):
        V3Config(learning_matrix=LearningMatrixConfig(readout="counts")).validate()


def test_from_dict_strict_and_roundtrip():
    cfg = _cfg(enabled=True)
    data = cfg.to_dict()
    assert isinstance(data["learning_matrix"], dict)
    back = V3Config.from_dict(data, strict=True)
    assert back.learning_matrix == LearningMatrixConfig(enabled=True)
    with pytest.raises(ValueError):
        V3Config.from_dict({"learning_matrix": {"bogus": 1}}, strict=True)
    assert "learning_matrix:r2" in describe(cfg)
    assert "learning_matrix" not in describe(_cfg(enabled=False))


def test_disabled_is_bit_identical_to_a_build_without_the_block():
    torch.manual_seed(0)
    x = _binary_input()
    plain = _model().eval()                     # config never mentions the block
    explicit = VectorNeuronPopulation(
        _cfg().with_overrides(learning_matrix=LearningMatrixConfig(enabled=False))
    ).eval()
    assert plain.learning_matrix_enabled is False
    assert explicit.learning_matrix_enabled is False
    assert list(plain.state_dict()) == list(explicit.state_dict())
    for key, tensor in plain.state_dict().items():
        assert torch.equal(tensor, explicit.state_dict()[key]), key
    # a random count classifier, so the readout check below is not the trivial
    # all-zero one the zero initialisation would give
    with torch.no_grad():
        plain.w_cls.normal_(0.0, 0.1)
        explicit.w_cls.copy_(plain.w_cls)
        p_logits, p_spikes = plain(x)[:2]
        e_logits, e_spikes = explicit(x)[:2]
    assert torch.equal(p_spikes, e_spikes)
    assert torch.equal(p_logits, e_logits)
    # the baseline readout is untouched: counts through the count classifier
    counts = p_spikes.float().sum(1)
    assert torch.equal(p_logits, counts @ plain.w_cls.float())
    assert p_logits.abs().sum() > 0
    assert plain.w_emit is not None and plain.w_cls is not None


# ---------------------------------------------------------------------- #
# 2. the enabled parameter set
# ---------------------------------------------------------------------- #
def test_enabled_registers_v_and_a_and_no_emission_or_classifier():
    model = _model(enabled=True)
    params = dict(model.named_parameters())
    assert params["v_matrix"].shape == (D, K)
    assert params["a_time"].shape == (T, K)
    assert model.w_emit is None and model.w_cls is None
    keys = set(model.state_dict())
    assert {"v_matrix", "a_time"} <= keys
    assert not {"w_emit", "w_cls"} & keys
    groups = model.param_groups()
    assert {"v_matrix", "a_time"} <= set(groups)
    assert not {"w_emit", "w_cls"} & set(groups)
    # uniform time weights, non-zero random basis
    assert torch.equal(model.a_time, torch.full((T, K), 1.0 / T))
    assert model.v_matrix.abs().sum() > 0


def test_enabled_shares_every_baseline_init_draw():
    base = _model(enabled=False)
    lm = _model(enabled=True)
    base_sd, lm_sd = base.state_dict(), lm.state_dict()
    shared = [key for key in base_sd if key in lm_sd]
    # every baseline parameter survives, at its baseline value: enabling the block
    # adds draws at the end of the stream and consumes none in between
    assert {"w_in", "w_down", "w_up", "w_rec", "gate", "bias", "w_out", "theta"} <= set(shared)
    for key in shared:
        assert torch.equal(base_sd[key], lm_sd[key]), key


# ---------------------------------------------------------------------- #
# 3. parameter budget
# ---------------------------------------------------------------------- #
def test_parameter_budget_arithmetic():
    base = _model(enabled=False)
    lm = _model(enabled=True)
    assert base.n_parameters() == parameter_groups(_cfg(enabled=False))["total"]
    assert lm.n_parameters() == parameter_groups(_cfg(enabled=True))["total"]

    on = parameter_groups(_cfg(enabled=True))
    assert on["v_matrix"] == D * K
    assert on["a_time"] == T * K
    assert "w_emit" not in on and "w_cls" not in on

    # the SHD budget (N=64, D=1000, K=20, T=500): D*K + T*K replace D*D + N*K
    big = V3Config(state_dim=1000, mix_rank=64)
    b = parameter_groups(big)
    r2 = parameter_groups(big.with_overrides(learning_matrix=LearningMatrixConfig(enabled=True)))
    assert b["total"] == 2_025_440
    assert r2["v_matrix"] == 1000 * 20 and r2["a_time"] == 500 * 20
    assert r2["total"] == b["total"] - 1000 * 1000 - 64 * 20 + 1000 * 20 + 500 * 20
    assert r2["total"] == 1_054_160


# ---------------------------------------------------------------------- #
# 4. the learning-matrix row is the pooled state
# ---------------------------------------------------------------------- #
def test_logits_are_the_hand_computed_matrix_readout():
    torch.manual_seed(0)
    x = _binary_input()
    model = _model(enabled=True).eval()
    with torch.no_grad():
        model.theta.fill_(1.0)          # a non-trivial spike / state trajectory
        logits, spikes, z_final, _ = model(x, return_states=True)
        ref_logits, ref_spikes, ref_z, matrix = _r2_reference(model, x)
    assert logits.shape == (B, K)
    assert matrix.abs().sum() > 0
    # the state update is untouched: the same trajectory, up to fp32 summation
    # order (the fast loop uses matmuls), spikes and readout stay exact
    torch.testing.assert_close(z_final, ref_z, rtol=1e-5, atol=1e-6)
    assert torch.equal(spikes, ref_spikes)
    # the readout is exactly the hand-computed sum over the learning matrix,
    # i.e. h_t is the pooled state and no emission map sits in between
    torch.testing.assert_close(logits, ref_logits, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------- #
# 5. gradients
# ---------------------------------------------------------------------- #
def test_gradient_reaches_v_and_a():
    torch.manual_seed(0)
    x = _binary_input()
    model = _model(enabled=True)
    with torch.no_grad():
        model.theta.fill_(1.0)          # positive from t = 0, so every neuron fires
    logits, spikes = model(x)[:2]
    assert logits.abs().sum() > 0
    loss = logits.pow(2).sum() + 1e-3 * spikes.float().sum()
    loss.backward()
    assert model.v_matrix.grad is not None and model.v_matrix.grad.abs().sum() > 0
    assert model.a_time.grad is not None and model.a_time.grad.abs().sum() > 0


# ---------------------------------------------------------------------- #
# 6. chunked accumulation (the guard against materialising (B, T, D))
# ---------------------------------------------------------------------- #
def test_chunked_forward_sums_to_the_single_pass_logits():
    torch.manual_seed(0)
    timesteps = 6
    x = _binary_input(batch=B, timesteps=timesteps)
    single = VectorNeuronPopulation(_cfg(enabled=True, timesteps=timesteps, chunks=1)).eval()
    chunked = VectorNeuronPopulation(_cfg(enabled=True, timesteps=timesteps, chunks=3)).eval()
    # same seed, so the two builds share every parameter
    for key, tensor in single.state_dict().items():
        assert torch.equal(tensor, chunked.state_dict()[key]), key
    with torch.no_grad():
        single_logits, single_spikes = single(x)[:2]
        direct_logits, direct_spikes = chunked(x)[:2]
    assert torch.equal(direct_spikes, single_spikes)
    torch.testing.assert_close(direct_logits, single_logits, rtol=1e-5, atol=1e-6)

    # and the checkpointed path returns the very same accumulated readout
    with mock.patch.object(model_module, "checkpoint_supported", lambda z: True), \
            mock.patch.object(model_module, "checkpoint",
                              wraps=model_module.checkpoint) as spy:
        chunked.train()
        ck_logits, ck_spikes = chunked(x)[:2]
    assert spy.call_count == 3          # T=6, chunks=3 -> step=2 -> three segments
    assert torch.equal(ck_spikes.detach(), single_spikes)
    torch.testing.assert_close(ck_logits.detach(), single_logits, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------- #
# 7. CLI / dotted override grammar
# ---------------------------------------------------------------------- #
def test_dotted_override_parses_learning_matrix():
    cfg = apply_override_items(
        _cfg(),
        ["learning_matrix.enabled=true", "learning_matrix.readout=r2"],
    )
    assert cfg.learning_matrix.enabled is True
    assert cfg.learning_matrix.readout == "r2"
    assert cfg.learning_matrix == LearningMatrixConfig(enabled=True)
    off = apply_override_items(_cfg(enabled=True), ["learning_matrix.enabled=false"])
    assert off.learning_matrix.enabled is False
    # the override grammar only parses values; the readout is checked by validate()
    bad = apply_override_items(_cfg(), ["learning_matrix.readout=r1"])
    with pytest.raises(ValueError):
        bad.validate()
