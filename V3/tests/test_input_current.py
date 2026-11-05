"""Tests for the scalar input current (design A) on top of diagonal emission."""

from __future__ import annotations

import math

import pytest
import torch

from v3.config import V3Config, apply_override_items, parameter_groups
from v3.model import VectorNeuronPopulation


def _cfg(*, current: bool, diagonal: bool = False) -> V3Config:
    cfg = V3Config()
    cfg.n_neurons = 8
    cfg.state_dim = 16
    cfg.n_inputs = 700
    cfg.n_classes = 20
    cfg.timing.sequence_duration_ms = 40.0  # 20 bins at 2 ms
    cfg.grad_checkpoint_chunks = 2
    cfg.input_current.enabled = current
    if diagonal:
        cfg.emission.mode = "diagonal"
    return cfg


def _model(cfg: V3Config) -> VectorNeuronPopulation:
    torch.manual_seed(0)
    return VectorNeuronPopulation(cfg)


def _x(batch: int = 4, steps: int = 20) -> torch.Tensor:
    return (torch.rand(batch, steps, 700) < 0.02).float()


def test_defaults_are_off():
    cfg = V3Config()
    assert cfg.input_current.enabled is False
    assert cfg.input_current.init_std > 0.0


def test_init_std_must_be_positive():
    cfg = V3Config()
    cfg.input_current.enabled = True
    cfg.input_current.init_std = 0.0
    with pytest.raises(ValueError, match="init_std"):
        cfg.validate()
    cfg.input_current.init_std = float("nan")
    with pytest.raises(ValueError, match="init_std"):
        cfg.validate()


def test_mutually_exclusive_with_delay_line():
    cfg = V3Config()
    cfg.input_current.enabled = True
    cfg.input_delay_line.enabled = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        cfg.validate()


def test_flag_off_registers_no_current_parameter():
    m = _model(_cfg(current=False))
    assert m.w_in is not None
    assert m.gate is not None
    assert m.w_in_current is None
    names = dict(m.named_parameters())
    assert "w_in_current" not in names


def test_enabled_replaces_projection_and_gate():
    cfg = _cfg(current=True)
    m = _model(cfg)
    assert m.w_in is None
    assert m.gate is None
    assert m.w_in_current is not None
    assert tuple(m.w_in_current.shape) == (8, 700)
    # the parameter set shrinks: C*D + N*D -> N*C
    budget = parameter_groups(cfg)
    assert budget["w_in"] == 0
    assert budget["w_in_current"] == 8 * 700
    assert "gate" not in budget
    assert budget["total"] == sum(p.numel() for p in m.parameters())


def test_forward_runs_and_produces_spikes():
    m = _model(_cfg(current=True))
    out = m(_x())
    logits, spikes = out[0], out[1]
    assert tuple(logits.shape) == (4, 20)
    assert tuple(spikes.shape) == (4, 20, 8)
    assert float(spikes.sum()) > 0.0


def test_gradient_reaches_the_current():
    m = _model(_cfg(current=True))
    with torch.no_grad():
        m.w_cls.normal_(0.0, 0.1)
    m(_x())[0].sum().backward()
    grad = m.w_in_current.grad
    assert grad is not None
    assert float(grad.abs().sum()) > 0.0
    # the shared projection must not exist, so nothing else can receive input
    assert m.w_in is None


def test_current_changes_firing_when_threshold_is_offset():
    m = _model(_cfg(current=True))
    with torch.no_grad():
        m.theta.fill_(1.5)
    x = _x(batch=8)
    with torch.no_grad():
        base = float(m(x)[1].sum())
        m.w_in_current.mul_(3.0)
        scaled = float(m(x)[1].sum())
    assert base != scaled


def test_current_flows_through_the_forward_chunks():
    """Gradient checkpointing must carry the current into every chunk."""
    cfg = _cfg(current=True)
    cfg.grad_checkpoint_chunks = 4
    m = _model(cfg)
    with torch.no_grad():
        m.w_cls.normal_(0.0, 0.1)
    m(_x())[0].sum().backward()
    assert m.w_in_current.grad is not None
    assert float(m.w_in_current.grad.abs().sum()) > 0.0


def test_diagonal_emission_and_current_compose():
    cfg = _cfg(current=True, diagonal=True)
    m = _model(cfg)
    assert m.emit_diagonal is True
    assert m.w_in_current is not None
    out = m(_x())
    assert tuple(out[0].shape) == (4, 20)


def test_override_via_string_form():
    cfg = V3Config()
    cfg = apply_override_items(
        cfg, ["input_current.enabled=true", "input_current.init_std=0.2"]
    )
    assert cfg.input_current.enabled is True
    assert cfg.input_current.init_std == pytest.approx(0.2)


def test_bias_init_std_defaults_to_zero_and_keeps_bias_at_zero():
    cfg = V3Config()
    cfg.n_neurons, cfg.state_dim = 4, 8
    cfg.input_current.enabled = True
    model = _model(cfg)
    assert model.input_current is True
    assert torch.count_nonzero(model.bias) == 0
    assert torch.count_nonzero(model.z0 if False else model.bias) == 0


def test_bias_init_std_breaks_the_zero_state_fixed_point():
    cfg = V3Config()
    cfg.n_neurons, cfg.state_dim = 4, 8
    cfg.input_current.enabled = True
    cfg.input_current.bias_init_std = 0.5
    model = _model(cfg)
    assert torch.count_nonzero(model.bias) > 0
    assert float(model.bias.std()) > 0.1
    model.eval()
    with torch.no_grad():
        model.calibrate_threshold(_x(batch=2, steps=20))
        _, _, z, _ = model(_x(batch=2, steps=20), return_states=True)
    assert float(z.abs().mean()) > 1e-3


def test_bias_init_std_must_be_non_negative():
    cfg = V3Config()
    cfg.input_current.enabled = True
    cfg.input_current.bias_init_std = -1.0
    with pytest.raises(ValueError, match="bias_init_std"):
        cfg.validate()
    cfg.input_current.bias_init_std = float("inf")
    with pytest.raises(ValueError, match="bias_init_std"):
        cfg.validate()
