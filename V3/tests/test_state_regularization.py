"""Focused tests for controlled state imprecision (noise + quantization).

CPU-only and tiny: these exercise the primitives and the plumbing into the model,
not any real training run.
"""

from __future__ import annotations

import math

import pytest
import torch

from v3.config import StateRegularizationConfig, TimingConfig, V3Config
from v3.model import VectorNeuronPopulation
from v3.state_regularization import add_state_noise, quantize_state, regularize_state


# ---------------------------------------------------------------------- #
# noise
# ---------------------------------------------------------------------- #
def test_noise_disabled_is_identity():
    z = torch.randn(8, 4, dtype=torch.float32)
    out = add_state_noise(z, 0.0)
    assert out is z  # unchanged, same object
    assert torch.equal(out, z)


def test_noise_enabled_perturbs_state():
    torch.manual_seed(0)
    z = torch.zeros(256, dtype=torch.float32)
    out = add_state_noise(z, 0.01)
    assert not torch.equal(out, z)
    assert torch.isfinite(out).all()


def test_noise_std_is_respected():
    torch.manual_seed(1234)
    z = torch.zeros(2_000_000, dtype=torch.float32)
    out = add_state_noise(z, 0.01)
    measured = float(out.std())
    assert abs(measured - 0.01) < 3e-4, measured


def test_noise_preserves_dtype():
    z = torch.zeros(32, dtype=torch.float16)
    out = add_state_noise(z, 0.02)
    assert out.dtype == torch.float16


# ---------------------------------------------------------------------- #
# quantization
# ---------------------------------------------------------------------- #
def test_quantization_disabled_is_identity():
    z = torch.randn(64, dtype=torch.float32)
    assert quantize_state(z, 16) is z
    reg = StateRegularizationConfig(mode="none")
    assert regularize_state(z, reg, "train") is z


def test_quantization_reduces_representable_values():
    torch.manual_seed(0)
    z = torch.rand(100_000, dtype=torch.float32) * 2 - 1  # dense continuous range
    q = quantize_state(z, 4)
    assert torch.isfinite(q).all()
    assert torch.unique(z).numel() > 1000
    assert torch.unique(q).numel() <= 16  # 2**bits


def test_quantization_is_bounded_and_finite():
    torch.manual_seed(1)
    z = torch.randn(1000, dtype=torch.float32) * 3.0  # far outside [-1, 1]
    q = quantize_state(z, 8, clip=1.0)
    assert torch.isfinite(q).all()
    assert float(q.abs().max()) <= 1.0 + 1e-6


def test_quantization_keeps_gradient_path_alive():
    z = torch.randn(512, dtype=torch.float32, requires_grad=True)
    q = quantize_state(z, 4)
    q.sum().backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert float(z.grad.abs().sum()) > 0.0


@pytest.mark.parametrize("factory", [
    lambda: torch.zeros(128),
    lambda: torch.full((128,), 5.0),
    lambda: torch.full((128,), -5.0),
    lambda: torch.full((128,), 1e-12),
    lambda: torch.full((128,), 1e12),
])
def test_quantization_degenerate_ranges_do_not_nan(factory):
    z = factory().to(torch.float32)
    q = quantize_state(z, 7)
    assert torch.isfinite(q).all()
    assert float(q.abs().max()) <= 1.0 + 1e-6


# ---------------------------------------------------------------------- #
# phase gating + model plumbing
# ---------------------------------------------------------------------- #
def test_default_config_is_baseline():
    reg = StateRegularizationConfig()
    assert reg.mode == "none"
    assert not reg.phase_active("train")
    assert not reg.phase_active("val")
    assert not reg.phase_active("test")


def test_training_only_by_default():
    reg = StateRegularizationConfig(mode="noise", noise_std=0.01)
    assert reg.phase_active("train")
    assert not reg.phase_active("val")
    assert not reg.phase_active("test")


def _tiny_cfg(**reg_kwargs) -> V3Config:
    cfg = V3Config(
        n_neurons=4,
        state_dim=16,
        mix_rank=4,
        n_inputs=10,
        n_classes=3,
        timing=TimingConfig(sequence_duration_ms=16.0, time_bin_ms=2.0),  # T = 8
        batch_size=4,
        device="cpu",
        dtype="float32",
        amp=False,
        grad_checkpoint_chunks=2,
    )
    if reg_kwargs:
        cfg = cfg.with_overrides(state_regularization=StateRegularizationConfig(**reg_kwargs))
    return cfg


def _forward(cfg: V3Config, x: torch.Tensor, phase: str):
    model = VectorNeuronPopulation(cfg)
    model.eval()
    model.set_phase(phase)
    with torch.no_grad():
        logits, spikes = model(x)
    return logits, spikes


def test_disabled_mode_forward_is_deterministic():
    torch.manual_seed(0)
    x = torch.randint(0, 2, (4, 8, 10)).float()
    cfg = _tiny_cfg()  # mode=none
    _, spikes1 = _forward(cfg, x, "train")
    _, spikes2 = _forward(cfg, x, "train")
    assert torch.equal(spikes1, spikes2)


def test_enabled_noise_changes_training_forward_but_not_test():
    torch.manual_seed(0)
    x = torch.randint(0, 2, (4, 8, 10)).float()
    _, base = _forward(_tiny_cfg(), x, "train")
    cfg_on = _tiny_cfg(mode="noise", noise_std=0.05)
    # training phase: perturbed
    torch.manual_seed(0)
    _, train_on = _forward(cfg_on, x, "train")
    assert not torch.equal(base, train_on)
    # official test phase: must stay unperturbed
    _, test_on = _forward(cfg_on, x, "test")
    assert torch.equal(base, test_on)


def test_enabled_quantization_changes_training_forward():
    torch.manual_seed(0)
    x = torch.randint(0, 2, (4, 8, 10)).float()
    _, base = _forward(_tiny_cfg(), x, "train")
    _, quant = _forward(_tiny_cfg(mode="quantization", quantize_bits=4), x, "train")
    assert not torch.equal(base, quant)


def test_phase_validation_of_config_rejects_bad_values():
    with pytest.raises(ValueError):
        V3Config(state_regularization=StateRegularizationConfig(mode="bogus")).validate()
    with pytest.raises(ValueError):
        V3Config(state_regularization=StateRegularizationConfig(noise_std=-0.1)).validate()
    with pytest.raises(ValueError):
        V3Config(state_regularization=StateRegularizationConfig(quantize_bits=1)).validate()


def test_yaml_roundtrip_of_nested_state_regularization():
    cfg = _tiny_cfg(mode="noise_quantization", noise_std=0.02, quantize_bits=6)
    data = cfg.to_dict()
    assert isinstance(data["state_regularization"], dict)
    back = V3Config.from_dict(data, strict=True)
    assert back.state_regularization.mode == "noise_quantization"
    assert back.state_regularization.noise_std == pytest.approx(0.02)
    assert back.state_regularization.quantize_bits == 6


def test_math_import_still_available():
    # guard against accidental removal of the module-level `math` re-export
    from v3.config import math as m

    assert m.sqrt(4) == 2.0
