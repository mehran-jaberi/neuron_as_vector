"""M11a SLAYER surrogate gradient + M11b learned temporal-attention readout.

CPU-only, deterministic.  The bit-identity tests are the contract: the default
configuration must reproduce the historical fast-sigmoid gradient and the
spike-count readout exactly (``torch.equal``), while the new paths are opt-in.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from v3.config import AttentionConfig, TimingConfig, V3Config
from v3.data import Split
from v3.model import (
    VectorNeuronPopulation,
    surrogate_derivative,
    surrogate_spike,
)
from v3.train import clip_gradients_, fit, set_seed


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
class _LegacySurrogate(torch.autograd.Function):
    """The pre-M11a surrogate, verbatim; only the bit-identity tests use it."""

    @staticmethod
    def forward(ctx, u, beta, gamma):
        u_w = u if u.dtype in (torch.float32, torch.float64) else u.float()
        ctx.save_for_backward(u_w)
        ctx.beta = float(beta)
        ctx.gamma = float(gamma)
        return (u_w > 0).to(u.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (u_w,) = ctx.saved_tensors
        deriv = ctx.gamma / (1.0 + ctx.beta * u_w.abs()).square()
        return (grad_out.to(deriv.dtype) * deriv).to(grad_out.dtype), None, None


def _cfg(**kw) -> V3Config:
    base = dict(
        seed=5, n_neurons=4, state_dim=6, mix_rank=3, n_inputs=5, n_classes=3,
        timing=TimingConfig(8.0, 2.0),          # T = 4
        device="cpu", dtype="float32", amp=False,
        batch_size=6, epochs=1, val_fraction=0.0, shuffle_train=True,
        lr_schedule="none", max_train_samples=0, grad_checkpoint_chunks=2,
        rate_reg=0.0,                             # loss == plain cross-entropy
        tag="m11",
    )
    base.update(kw)
    return V3Config(**base)


def _model(cfg: V3Config) -> VectorNeuronPopulation:
    set_seed(cfg.seed)
    return VectorNeuronPopulation(cfg)


def _dense_x(batch: int = 2, n_bins: int = 4, n_inputs: int = 5) -> torch.Tensor:
    g = torch.Generator().manual_seed(4)
    return (torch.rand(batch, n_bins, n_inputs, generator=g) < 0.4).to(torch.float32)


class _ToyStore:
    """Same minimal protocol as ``test_resume._ToyStore``: no HDF5."""

    def __init__(self, n: int = 24, n_classes: int = 3, n_bins: int = 4, n_inputs: int = 5):
        rng = np.random.default_rng(0)
        self.x = (rng.random((n, n_bins, n_inputs)) < 0.25).astype(np.float32)
        self.labels = rng.integers(0, n_classes, n).astype(np.int64)
        self.speakers = np.zeros(n, dtype=np.int64)

    def batch(self, indices):
        return self.x[np.asarray(indices, dtype=np.int64)]

    def labels_of(self, indices):
        return self.labels[np.asarray(indices, dtype=np.int64)]


# ---------------------------------------------------------------------- #
# M11a: the default surrogate is bit-identical to the historical one
# ---------------------------------------------------------------------- #
def test_shipped_yaml_defaults_are_the_baseline_path():
    cfg = V3Config.from_yaml(
        Path(__file__).resolve().parents[1] / "configs" / "v3_default.yaml"
    )
    assert cfg.surrogate == "fast_sigmoid"
    assert cfg.surrogate_alpha == 0.5
    assert cfg.readout == "sum"
    assert cfg.attention.enabled is False


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16])
def test_default_surrogate_is_bit_identical_to_legacy(dtype):
    values = torch.linspace(-3.0, 3.0, 25, dtype=dtype)
    u_new = values.clone().requires_grad_(True)
    u_old = values.clone().requires_grad_(True)

    out_new = surrogate_spike(u_new)                    # all defaults
    out_old = _LegacySurrogate.apply(u_old, 4.0, 1.0)
    assert torch.equal(out_new, out_old)               # Heaviside forward

    g_new = torch.autograd.grad(out_new.sum(), u_new)[0]
    g_old = torch.autograd.grad(out_old.sum(), u_old)[0]
    assert g_new.dtype == g_old.dtype == dtype
    assert torch.equal(g_new, g_old), "default fast-sigmoid gradient changed"


def test_fast_sigmoid_derivative_uses_the_historical_operand_order():
    u = torch.linspace(-3.0, 3.0, 25, dtype=torch.float32)
    expected = 1.0 / (1.0 + 4.0 * u.abs()).square()
    assert torch.equal(surrogate_derivative(u, 4.0, 1.0), expected)
    assert torch.equal(surrogate_derivative(u, 4.0, 2.5), 2.5 / (1.0 + 4.0 * u.abs()).square())


def test_default_model_forward_and_loss_are_the_plain_count_readout():
    m = _model(_cfg())
    m.eval()
    with torch.no_grad():
        m.w_cls.copy_(torch.randn(4, 3, generator=torch.Generator().manual_seed(2)))
    x, y = _dense_x(), torch.tensor([0, 2])
    with torch.no_grad():
        logits, spikes = m(x)
    counts = spikes.float().sum(1)
    assert torch.equal(m.readout_counts(spikes), counts)
    assert torch.equal(logits, counts @ m.w_cls.float())
    with torch.no_grad():
        loss, ce = m.loss(logits, spikes, y)
    assert torch.equal(loss, torch.nn.functional.cross_entropy(logits.float(), y))
    assert torch.equal(loss, ce)


def test_mean_readout_is_still_the_count_sum_over_t():
    m = _model(_cfg(readout="mean"))
    m.eval()
    with torch.no_grad():
        _logits, spikes = m(_dense_x())
    assert torch.equal(m.readout_counts(spikes), spikes.float().sum(1) / spikes.shape[1])


# ---------------------------------------------------------------------- #
# M11a: SLAYER shape
# ---------------------------------------------------------------------- #
def test_slayer_derivative_at_zero_and_at_alpha():
    u = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=torch.float64)
    deriv = surrogate_derivative(u, kind="slayer", alpha=0.5)
    assert deriv.dtype == torch.float64
    assert float(deriv[2]) == 1.0                        # u = 0
    assert float(deriv[3]) == pytest.approx(1.0 / 1.25 ** 2)   # u = +alpha = 0.5
    assert float(deriv[1]) == pytest.approx(1.0 / 1.25 ** 2)   # u = -alpha
    assert float(deriv[1]) == float(deriv[3])            # even function
    expected = 1.0 / (1.0 + 0.5 * u.abs()).square()
    assert torch.equal(deriv, expected)
    # a different width changes the decay, not the value at threshold
    assert float(surrogate_derivative(torch.zeros(1), kind="slayer", alpha=2.0)) == 1.0
    wide = surrogate_derivative(torch.tensor([2.0]), kind="slayer", alpha=2.0)
    assert float(wide) == pytest.approx(1.0 / 25.0)


def test_slayer_forward_is_the_same_heaviside():
    u = torch.linspace(-2.0, 2.0, 9, dtype=torch.float32)
    assert torch.equal(surrogate_spike(u, kind="slayer"), (u > 0).to(torch.float32))


def test_slayer_is_wired_through_the_model_config():
    """`surrogate="slayer"`: same forward, different backward, finite gradients."""
    m = _model(_cfg(surrogate="slayer", surrogate_alpha=0.5))
    m.eval()
    with torch.no_grad():
        m.w_cls.normal_(0.0, 0.1, generator=torch.Generator().manual_seed(3))
    x, y = _dense_x(), torch.tensor([0, 2])
    logits, spikes = m(x)
    assert torch.equal(spikes, (spikes > 0).to(spikes.dtype))   # still Heaviside
    loss, _ce = m.loss(logits, spikes, y)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.w_out.grad is not None and float(m.w_out.grad.abs().sum()) > 0.0
    assert m.theta.grad is not None and torch.isfinite(m.theta.grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_surrogate_gradient_is_fp32_and_ignores_autocast(dtype):
    u32 = torch.linspace(-3.0, 3.0, 17, dtype=torch.float32)
    u_low = u32.to(dtype)
    u_wide = u_low.float()                               # what the model will see
    reference = 1.0 / (1.0 + 0.5 * u_wide.abs()).square()

    assert surrogate_derivative(u_low, kind="slayer").dtype == torch.float32
    assert torch.equal(surrogate_derivative(u_low, kind="slayer"), reference)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert torch.is_autocast_enabled("cpu")
        inside = surrogate_derivative(u_low, kind="slayer")
    assert inside.dtype == torch.float32
    assert torch.equal(inside, reference), "autocast changed the surrogate gradient"

    # the autograd path returns the gradient in the input dtype, but computes it
    # from fp32 (same values as the fp32 reference)
    u_grad = u_low.clone().requires_grad_(True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = surrogate_spike(u_grad, kind="slayer")
        grad = torch.autograd.grad(out.sum(), u_grad)[0]
    assert grad.dtype == dtype
    # autograd can only hand back the input dtype, but the value must be the fp32
    # derivative rounded *once* at the end -- not arithmetic done in bf16/fp16.
    assert torch.equal(grad, reference.to(dtype))


def test_surrogate_derivative_keeps_fp64():
    """run_smoke_test.py checks fp64 gradients to 1e-9: fp64 must stay fp64."""
    u = torch.linspace(-4.0, 4.0, 41, dtype=torch.float64)
    deriv = surrogate_derivative(u)
    assert deriv.dtype == torch.float64
    assert torch.equal(deriv, 1.0 / (1.0 + 4.0 * u.abs()).square())


def test_bad_surrogate_arguments_raise():
    with pytest.raises(ValueError, match="unknown surrogate"):
        surrogate_derivative(torch.zeros(3), kind="arctan")
    with pytest.raises(ValueError, match="SLAYER surrogate_alpha"):
        surrogate_derivative(torch.zeros(3), kind="slayer", alpha=0.0)


def test_config_validates_surrogate_and_grad_clip():
    _cfg(surrogate="slayer", surrogate_alpha=0.5).validate()
    with pytest.raises(ValueError, match="surrogate must be one of"):
        _cfg(surrogate="arctan").validate()
    with pytest.raises(ValueError, match="surrogate_alpha"):
        _cfg(surrogate="slayer", surrogate_alpha=-1.0).validate()
    assert _cfg(grad_clip=0.0).validate() is None        # 0 == disabled
    with pytest.raises(ValueError, match="grad_clip"):
        _cfg(grad_clip=-1.0).validate()
    with pytest.raises(ValueError, match="temperature"):
        _cfg(attention=AttentionConfig(temperature=0.0)).validate()


# ---------------------------------------------------------------------- #
# M11a: grad_clip wiring
# ---------------------------------------------------------------------- #
def test_grad_clip_helper_noop_at_zero_and_unscales_when_active():
    linear = torch.nn.Linear(3, 2)
    opt = torch.optim.SGD(linear.parameters(), lr=0.1)
    for p in linear.parameters():
        p.grad = torch.full_like(p, 10.0)
    before = [p.grad.detach().clone() for p in linear.parameters()]

    unscaled: list[object] = []

    class _Scaler:
        def unscale_(self, optimizer):
            unscaled.append(optimizer)

    scaler = _Scaler()
    assert clip_gradients_(linear, opt, scaler, 0.0) == 0.0
    assert unscaled == []                                # nothing touched when off
    for p, g in zip(linear.parameters(), before):
        assert torch.equal(p.grad, g)

    original_norm = math.sqrt(sum(float(p.grad.pow(2).sum()) for p in linear.parameters()))
    returned = clip_gradients_(linear, opt, scaler, 0.5)
    assert unscaled == [opt]                             # unscaled before clipping
    assert returned == pytest.approx(original_norm, rel=1e-6)   # pre-clip norm
    clipped = math.sqrt(sum(float(p.grad.pow(2).sum()) for p in linear.parameters()))
    assert clipped == pytest.approx(0.5, rel=1e-6)

    # a non-positive value (config rejects it) is still a tolerant no-op
    assert clip_gradients_(linear, opt, scaler, -1.0) == 0.0


def test_fit_runs_one_epoch_with_grad_clip_disabled_and_attention_on():
    store = _ToyStore()
    cfg = _cfg(grad_clip=0.0, attention=AttentionConfig(enabled=True))
    model = _model(cfg)
    idx = np.arange(store.labels.size)
    split = Split("fit", idx, store.labels[idx], store.speakers[idx])
    result = fit(model, store, split, None, cfg, torch.device("cpu"), show_progress=False)
    assert len(result.history) == 1
    assert math.isfinite(result.history[0]["loss"])
    assert model.attn_logits is not None and model.attn_logits.grad is not None
    assert torch.isfinite(model.attn_logits.grad).all()


# ---------------------------------------------------------------------- #
# M11b: disabled path
# ---------------------------------------------------------------------- #
def test_attention_off_is_parameter_free_and_bit_identical():
    m = _model(_cfg())
    m.eval()
    assert m.attn_logits is None
    assert "attn_logits" not in dict(m.named_parameters())
    assert not any("attn" in k for k in m.state_dict())
    with pytest.raises(RuntimeError, match="attention is disabled"):
        m.attention_weights()

    with torch.no_grad():
        m.w_cls.copy_(torch.randn(4, 3, generator=torch.Generator().manual_seed(7)))
        logits, spikes = m(_dense_x())
    counts = spikes.float().sum(1)
    assert torch.equal(m.readout_counts(spikes), counts)
    assert torch.equal(logits, counts @ m.w_cls.float())

    on = _model(_cfg(attention=AttentionConfig(enabled=True)))
    assert on.n_parameters() == m.n_parameters() + m.cfg.n_bins   # only the T logits


# ---------------------------------------------------------------------- #
# M11b: weights and readout
# ---------------------------------------------------------------------- #
def test_attention_weights_form_a_distribution():
    m = _model(_cfg(attention=AttentionConfig(enabled=True, temperature=1.0)))
    m.eval()
    with torch.no_grad():
        m.attn_logits.copy_(torch.tensor([-1.0, 0.0, 1.0, 2.0]))
    a = m.attention_weights()
    assert a.shape == (4,) and a.dtype == torch.float32
    assert float(a.detach().sum()) == pytest.approx(1.0, abs=1e-6)
    assert bool((a > 0).all())
    assert torch.equal(a, torch.softmax(torch.tensor([-1.0, 0.0, 1.0, 2.0]), dim=0))
    # a prefix of bins is a valid distribution too (used with a shorter sequence)
    assert torch.equal(m.attention_weights(2), torch.softmax(torch.tensor([-1.0, 0.0]), dim=0))

    larger_temperature = m.attention_weights(4).detach()
    m.cfg.attention.temperature = 4.0
    flatter = m.attention_weights(4)
    assert float(flatter.detach().max()) < float(larger_temperature.max())
    assert float(flatter.detach().sum()) == pytest.approx(1.0, abs=1e-6)
    assert torch.equal(flatter, torch.softmax(torch.tensor([-1.0, 0.0, 1.0, 2.0]) / 4.0, dim=0))

    with pytest.raises(ValueError, match="bin logits"):
        m.attention_weights(5)


def test_attention_starts_uniform_and_matches_mean_counts():
    m = _model(_cfg(attention=AttentionConfig(enabled=True)))
    m.eval()
    with torch.no_grad():
        _logits, spikes = m(_dense_x())
    assert torch.equal(m.attention_weights(), torch.full((4,), 0.25))
    assert torch.equal(m.readout_counts(spikes), spikes.float().mean(1))


def test_hand_computed_two_bin_attention_readout():
    cfg = _cfg(
        n_neurons=4, n_classes=3,
        timing=TimingConfig(4.0, 2.0),          # T = 2 bins
        attention=AttentionConfig(enabled=True, temperature=1.0),
    )
    m = _model(cfg)
    m.eval()
    w = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6],
                      [-0.1, -0.2, -0.3], [0.7, -0.4, 0.25]])
    logits_in = torch.tensor([-0.5, 0.5])
    with torch.no_grad():
        m.w_cls.copy_(w)
        m.attn_logits.copy_(logits_in)
    with torch.no_grad():
        logits, spikes = m(_dense_x(n_bins=2))
    assert spikes.shape == (2, 2, 4)

    a = torch.softmax(logits_in, dim=0)
    counts = spikes.float()[:, 0, :] * a[0] + spikes.float()[:, 1, :] * a[1]
    assert torch.equal(logits, counts @ w)

    # independent double-precision hand computation for sample 0
    a0 = math.exp(-0.5) / (math.exp(-0.5) + math.exp(0.5))
    a1 = 1.0 - a0
    s = spikes.tolist()[0]
    w_list = w.tolist()
    cnt = [a0 * s[0][n] + a1 * s[1][n] for n in range(4)]
    expected = [sum(cnt[n] * w_list[n][k] for n in range(4)) for k in range(3)]
    assert logits.tolist()[0] == pytest.approx(expected, abs=1e-5)

    # a genuinely non-uniform distribution changes the logits vs the mean readout
    with torch.no_grad():
        m.attn_logits.zero_()
        uniform_logits, _ = m(_dense_x(n_bins=2))
    assert not torch.equal(logits, uniform_logits)


def test_attention_gradients_reach_the_attention_parameters():
    m = _model(_cfg(attention=AttentionConfig(enabled=True)))
    m.eval()
    x, y = _dense_x(), torch.tensor([0, 2])
    logits, spikes = m(x)
    loss, _ce = m.loss(logits, spikes, y)
    loss.backward()
    # w_cls is zero-init, so dL/dcounts == grad_logits @ w_cls.T == 0 exactly:
    # the attention logits cannot move before the classifier does.
    assert m.attn_logits.grad is not None
    assert torch.equal(m.attn_logits.grad, torch.zeros(4))

    m.zero_grad(set_to_none=True)
    with torch.no_grad():
        m.w_cls.normal_(0.0, 0.1, generator=torch.Generator().manual_seed(11))
    logits, spikes = m(x)
    loss, _ce = m.loss(logits, spikes, y)
    loss.backward()
    assert m.attn_logits.grad is not None
    assert float(m.attn_logits.grad.abs().sum()) > 0.0
    assert m.w_cls.grad is not None
    assert float(m.w_cls.grad.abs().sum()) > 0.0
