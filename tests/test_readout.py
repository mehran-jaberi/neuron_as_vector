"""Tests for the configurable readout aggregation (mean / last / sum)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.model import SNNConfig, build_model


def _tiny_cfg(**kw) -> SNNConfig:
    base = dict(n_input=20, n_hidden=16, n_output=4, n_bins=30, bin_ms=2.0, neuron_param_mode="bias")
    base.update(kw)
    return SNNConfig(**base)


def _x(cfg: SNNConfig, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(5, cfg.n_bins, cfg.n_input, generator=g)


def test_default_readout_mode_is_mean():
    assert SNNConfig().readout_mode == "mean"


def test_invalid_readout_mode_is_rejected():
    with pytest.raises(ValueError):
        build_model(_tiny_cfg(readout_mode="attention"))


def test_sum_readout_equals_T_times_mean_readout():
    # with readout_leak = 0 the readout is linear, so the accumulated readout is
    # exactly T x the time-averaged readout (a reparameterisation, not new capacity)
    cfg_mean = _tiny_cfg(readout_mode="mean")
    cfg_sum = _tiny_cfg(readout_mode="sum")
    m_mean = build_model(cfg_mean, seed=1)
    m_sum = build_model(cfg_sum, seed=1)
    m_sum.load_state_dict(m_mean.state_dict())
    x = _x(cfg_mean)
    with torch.no_grad():
        a = m_mean(x)["logits"]
        b = m_sum(x)["logits"]
    assert torch.allclose(b, cfg_mean.n_bins * a, atol=1e-4)


def test_last_readout_uses_only_the_final_state():
    cfg = _tiny_cfg(readout_mode="last")
    model = build_model(cfg, seed=2)
    x = _x(cfg)
    with torch.no_grad():
        out = model(x, record=True)
    logits = out["logits"]
    # final readout state O_T = sum_t s_t W_out + b_out (rho=0); reconstruct it
    o_final = out["output_activity_trace"][:, -1, :]
    assert torch.allclose(logits, o_final, atol=1e-4)


def test_mean_and_last_differ_when_activity_is_not_stationary():
    cfg_mean = _tiny_cfg(readout_mode="mean")
    cfg_last = _tiny_cfg(readout_mode="last")
    m_mean = build_model(cfg_mean, seed=3)
    m_last = build_model(cfg_last, seed=3)
    m_last.load_state_dict(m_mean.state_dict())
    x = _x(cfg_mean)
    with torch.no_grad():
        a = m_mean(x)["logits"].numpy()
        b = m_last(x)["logits"].numpy()
    assert not np.allclose(a, b, atol=1e-6)