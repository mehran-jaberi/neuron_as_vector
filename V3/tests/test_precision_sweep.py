"""Focused tests for the V3 precision-screening runner.

CPU-only and tiny: these check the *plumbing* (config construction, the
quantizer being reached by the model, the 5-epoch screen budget, and one
registry row per variant).  No real training run happens here.
"""

from __future__ import annotations

import pytest
import torch

import v3.model as model_module
from v3.config import StateRegularizationConfig, TimingConfig, V3Config
from v3.registry import RunRegistry
from run_precision_sweep import (
    DEFAULT_CONFIG,
    PRECISION_VARIANTS,
    SCREEN_EPOCHS,
    build_precision_config,
    register_variant,
    smoke_forward,
)

VARIANT_NAMES = ["baseline", "8bit", "4bit"]


def _tiny_base() -> V3Config:
    """A small but structurally identical config (fast CPU forward)."""
    return V3Config(
        n_neurons=4,
        state_dim=8,
        mix_rank=4,
        n_inputs=4,
        timing=TimingConfig(sequence_duration_ms=8.0, time_bin_ms=2.0),
        batch_size=2,
        epochs=20,
        device="cpu",
        dtype="float32",
        amp=False,
        grad_checkpoint_chunks=1,
        state_regularization=StateRegularizationConfig(),
    )


# ---------------------------------------------------------------------- #
# variant table + config construction
# ---------------------------------------------------------------------- #
def test_variant_table_has_baseline_and_the_two_requested_bits():
    assert set(PRECISION_VARIANTS) == {"baseline", "8bit", "4bit"}
    assert PRECISION_VARIANTS["baseline"] is None
    assert PRECISION_VARIANTS["8bit"] == 8
    assert PRECISION_VARIANTS["4bit"] == 4


def test_screen_budget_is_five_epochs():
    assert SCREEN_EPOCHS == 5


@pytest.mark.parametrize("variant,bits", [("8bit", 8), ("4bit", 4)])
def test_quantization_variant_sets_mode_bits_and_phases(variant, bits):
    cfg = build_precision_config(_tiny_base(), variant)
    reg = cfg.state_regularization
    assert reg.mode == "quantization"
    assert reg.quantize_bits == bits
    assert reg.quantization_enabled is True
    assert reg.uses_quantization is True
    assert reg.apply_during_training is True
    assert reg.apply_during_validation is False
    assert reg.apply_during_test is False


def test_baseline_variant_stays_mode_none():
    cfg = build_precision_config(_tiny_base(), "baseline")
    reg = cfg.state_regularization
    assert reg.mode == "none"
    assert reg.quantization_enabled is False
    assert reg.uses_quantization is False
    assert reg.phase_active("train") is False


def test_five_epoch_override_is_forced_regardless_of_base():
    base = _tiny_base()
    assert base.epochs == 20
    for name in VARIANT_NAMES:
        assert build_precision_config(base, name).epochs == 5
    # explicit budget still wins
    assert build_precision_config(base, "8bit", epochs=3).epochs == 3


def test_reference_architecture_is_preserved():
    base = V3Config.from_yaml(DEFAULT_CONFIG).with_overrides(epochs=20)
    for name in VARIANT_NAMES:
        cfg = build_precision_config(base, name)
        assert cfg.n_neurons == 64
        assert cfg.state_dim == 1000
        assert cfg.sequence_duration_ms == 1000.0
        assert cfg.time_bin_ms == 2.0
        assert cfg.simulation_dt_ms == 2.0
        assert cfg.num_time_steps == 500
        # unchanged training/optimizer/seed/dtype
        assert cfg.batch_size == base.batch_size
        assert cfg.learning_rate == base.learning_rate
        assert cfg.optimizer == base.optimizer
        assert cfg.lr_schedule == base.lr_schedule
        assert cfg.seed == base.seed
        assert cfg.dtype == base.dtype


def test_variant_tags_are_distinct_and_do_not_collide_with_reference():
    base = V3Config.from_yaml(DEFAULT_CONFIG)
    tags = {build_precision_config(base, n).tag for n in VARIANT_NAMES}
    assert len(tags) == 3
    assert base.tag not in tags  # never overwrite the 20-epoch reference checkpoint


def test_unknown_variant_is_rejected():
    with pytest.raises(SystemExit):
        build_precision_config(_tiny_base(), "16bit")


# ---------------------------------------------------------------------- #
# the configuration actually reaches the state quantizer in the model
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("variant,bits", [("8bit", 8), ("4bit", 4)])
def test_quantization_configuration_reaches_the_state_quantizer(monkeypatch, variant, bits):
    cfg = build_precision_config(_tiny_base(), variant)
    calls: list[tuple[int, str, str]] = []
    real = model_module.regularize_state

    def spy(z, reg, phase):
        calls.append((int(reg.quantize_bits), reg.mode, phase))
        return real(z, reg, phase)

    monkeypatch.setattr(model_module, "regularize_state", spy)
    model = model_module.VectorNeuronPopulation(cfg)
    model.eval()
    model.set_phase("train")  # quantizer active in the training phase
    x = torch.randn(2, cfg.n_bins, cfg.n_inputs)
    with torch.no_grad():
        logits, spikes = model(x)
    assert torch.isfinite(logits).all() and torch.isfinite(spikes).all()
    assert calls, "regularize_state was never called"
    assert any(b == bits and mode == "quantization" and phase == "train" for b, mode, phase in calls)


def test_baseline_configuration_never_reaches_the_quantizer(monkeypatch):
    cfg = build_precision_config(_tiny_base(), "baseline")
    calls: list[tuple[int, str, str]] = []
    real = model_module.regularize_state

    def spy(z, reg, phase):
        calls.append((int(reg.quantize_bits), reg.mode, phase))
        return real(z, reg, phase)

    monkeypatch.setattr(model_module, "regularize_state", spy)
    model = model_module.VectorNeuronPopulation(cfg)
    model.eval()
    model.set_phase("train")
    x = torch.randn(2, cfg.n_bins, cfg.n_inputs)
    with torch.no_grad():
        model(x)
    assert calls == []


# ---------------------------------------------------------------------- #
# smoke forward (tiny dims; the full D=1000/T=500 check is the CLI --smoke)
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("variant", ["8bit", "4bit"])
def test_smoke_forward_is_finite(variant):
    out = smoke_forward(_tiny_base(), variant, torch.device("cpu"))
    assert out["finite"] is True
    assert out["mode"] == "quantization"
    assert out["quantize_bits"] == PRECISION_VARIANTS[variant]
    assert out["logits_shape"][0] == 2


# ---------------------------------------------------------------------- #
# registry: one completed variant -> exactly one row
# ---------------------------------------------------------------------- #
def test_each_variant_produces_one_registry_entry(tmp_path):
    registry = RunRegistry(tmp_path)
    for name in VARIANT_NAMES:
        register_variant(
            registry,
            build_precision_config(_tiny_base(), name),
            n_parameters=123,
            variant=name,
        )
    runs = registry.read_runs()
    assert len(runs) == 3
    assert len([p for p in tmp_path.glob("*/") if p.is_dir()]) == 3


def test_registry_preserves_quantization_bits(tmp_path):
    registry = RunRegistry(tmp_path)
    for name in VARIANT_NAMES:
        register_variant(
            registry,
            build_precision_config(_tiny_base(), name),
            n_parameters=123,
            variant=name,
        )
    by_variant = {}
    for row in registry.read_runs():
        by_variant[row["tag"]] = row
    for name in VARIANT_NAMES:
        tag = build_precision_config(_tiny_base(), name).tag
        row = by_variant[tag]
        assert row["epochs"] == "5"
        if name == "baseline":
            assert row["state_regularization_mode"] == "none"
            assert row["state_quantization_enabled"] == "False"
        else:
            assert row["state_regularization_mode"] == "quantization"
            assert row["state_quantization_enabled"] == "True"
            assert row["state_quantization_bits"] == str(PRECISION_VARIANTS[name])


def test_registry_keeps_timing_and_phase_metadata(tmp_path):
    registry = RunRegistry(tmp_path)
    register_variant(
        registry,
        build_precision_config(_tiny_base(), "4bit"),
        n_parameters=123,
        variant="4bit",
    )
    row = registry.read_runs()[0]
    assert row["sequence_duration_ms"] == "8.0"
    assert row["time_bin_ms"] == "2.0"
    assert row["num_time_steps"] == "4"
    assert row["state_reg_apply_validation"] == "False"
    assert row["state_reg_apply_test"] == "False"
