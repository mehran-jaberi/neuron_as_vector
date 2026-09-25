"""Tests for the recurrent LIF SNN: dynamics, forward pass and surrogate gradients."""

from __future__ import annotations

import math

import numpy as np
import torch

from src.model import SNNConfig, build_model, lif_voltage_update, surrogate_spike


def test_forward_logits_and_spike_count_shapes(tiny_model, tiny_input, tiny_snn_config):
    x = torch.as_tensor(tiny_input, dtype=torch.float32)
    out = tiny_model(x)
    B = x.shape[0]
    assert out["logits"].shape == (B, tiny_snn_config.n_output)
    assert out["spike_count"].shape == (B, tiny_snn_config.n_hidden)
    assert torch.isfinite(out["logits"]).all()


def test_recorded_traces_shapes(tiny_model, tiny_input, tiny_snn_config):
    x = torch.as_tensor(tiny_input, dtype=torch.float32)
    out = tiny_model(x, record=True)
    B, T = x.shape[0], x.shape[1]
    assert out["hidden_spikes"].shape == (B, T, tiny_snn_config.n_hidden)
    assert out["hidden_v"].shape == (B, T, tiny_snn_config.n_hidden)
    assert out["output_activity_trace"].shape == (B, T, tiny_snn_config.n_output)
    # recorded spikes are a proper subset of {0, 1}
    s = out["hidden_spikes"]
    assert torch.all((s == 0) | (s == 1))


def test_surrogate_gradient_flows_to_weights(tiny_snn_config):
    # Fresh model so we do not pollute the session fixture's gradients.
    model = build_model(tiny_snn_config, seed=0)
    x = torch.randn(4, tiny_snn_config.n_bins, tiny_snn_config.n_input)
    out = model(x)
    loss = out["logits"].pow(2).mean()
    loss.backward()
    assert model.w_in.grad is not None
    assert torch.isfinite(model.w_in.grad).all()
    assert model.w_in.grad.abs().sum().item() > 0.0


def test_surrogate_spike_forward_is_heaviside():
    u = torch.tensor([-1.0, -0.1, 0.0, 0.1, 1.0], dtype=torch.float64)
    s = surrogate_spike(u, beta=5.0, gamma=0.3)
    assert torch.equal(s, (u > 0).to(u.dtype))


def test_surrogate_spike_gradient_is_bounded_and_centred():
    u = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    surrogate_spike(u, beta=5.0, gamma=0.3).backward()
    # Fast-sigmoid derivative at u=0 is exactly gamma.
    assert np.isclose(u.grad.item(), 0.3, atol=1e-9)


def test_lif_voltage_update_matches_formula():
    v = torch.tensor([0.0], dtype=torch.float64)
    i = torch.tensor([1.0], dtype=torch.float64)
    alpha = math.exp(-2.0 / 20.0)
    out = lif_voltage_update(v, i, alpha)
    assert np.isclose(out.item(), (1.0 - alpha), atol=1e-9)


def test_model_is_deterministic_under_seed():
    cfg = SNNConfig(n_input=10, n_hidden=8, n_output=3, n_bins=12)
    a = build_model(cfg, seed=1)
    b = build_model(cfg, seed=1)
    assert torch.allclose(a.w_in, b.w_in)
    assert torch.allclose(a.w_rec, b.w_rec)
    assert torch.allclose(a.w_out, b.w_out)


def test_derived_time_constants():
    cfg = SNNConfig(bin_ms=2.0, tau_mem_ms=20.0, tau_syn_ms=5.0)
    assert np.isclose(cfg.alpha, math.exp(-2.0 / 20.0))
    assert np.isclose(cfg.beta, math.exp(-2.0 / 5.0))
    assert np.isclose(cfg.duration_ms, cfg.n_bins * cfg.bin_ms)
