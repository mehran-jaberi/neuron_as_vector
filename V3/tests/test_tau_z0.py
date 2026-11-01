"""Focused tests for the trainable state dynamics: per-neuron leak and ``z(0)``.

CPU-only and tiny: these exercise the parameterisation and the plumbing into the
model, not any real training run.  The default (both flags off) must stay
bit-identical to the fixed-alpha / zero-state baseline.
"""

from __future__ import annotations

import math

import pytest
import torch

from v3.config import StateDynamicsConfig, TimingConfig, V3Config, apply_override_items
from v3.model import VectorNeuronPopulation

N, D, C, K, B, T = 4, 8, 6, 3, 2, 4


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def _cfg(**sd_kwargs) -> V3Config:
    cfg = V3Config(
        n_neurons=N,
        state_dim=D,
        mix_rank=2,
        n_inputs=C,
        n_classes=K,
        timing=TimingConfig(sequence_duration_ms=2.0 * T, time_bin_ms=2.0),  # T = 4
        batch_size=B,
        device="cpu",
        dtype="float32",
        amp=False,
        grad_checkpoint_chunks=2,
    )
    if sd_kwargs:
        cfg = cfg.with_overrides(state_dynamics=StateDynamicsConfig(**sd_kwargs))
    return cfg


def _x(steps: int = T) -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randint(0, 2, (B, steps, C)).float()


def _model(cfg: V3Config) -> VectorNeuronPopulation:
    # eval() keeps the autograd graph (only the training-time checkpoint path
    # and the state-regularisation phase depend on the mode)
    return VectorNeuronPopulation(cfg).eval()


# ---------------------------------------------------------------------- #
# defaults and config wiring
# ---------------------------------------------------------------------- #
def test_defaults_are_off():
    sd = StateDynamicsConfig()
    assert sd.enabled is False
    assert sd.trainable_alpha is False
    assert sd.learnable_z0 is False
    assert sd.alpha_init == pytest.approx(0.1)
    assert sd.active is False
    assert V3Config().state_dynamics == sd


def test_master_switch_gates_the_sub_flags():
    staged = StateDynamicsConfig(trainable_alpha=True, learnable_z0=True)
    assert staged.active is False
    assert staged.alpha_trainable is False
    assert staged.z0_learnable is False
    on = StateDynamicsConfig(enabled=True, trainable_alpha=True, learnable_z0=True)
    assert on.active is True
    assert on.alpha_trainable is True
    assert on.z0_learnable is True


def test_from_dict_parses_the_nested_block():
    cfg = V3Config.from_dict(
        {"state_dynamics": {"enabled": True, "trainable_alpha": True, "alpha_init": 0.25}}
    )
    assert isinstance(cfg.state_dynamics, StateDynamicsConfig)
    assert cfg.state_dynamics.enabled is True
    assert cfg.state_dynamics.trainable_alpha is True
    assert cfg.state_dynamics.learnable_z0 is False
    assert cfg.state_dynamics.alpha_init == pytest.approx(0.25)
    with pytest.raises(ValueError):
        V3Config.from_dict({"state_dynamics": {"alph_init": 0.2}}, strict=True)
    # unknown keys stay ignored when not strict (existing convention)
    ignored = V3Config.from_dict({"state_dynamics": {"alph_init": 0.2}})
    assert ignored.state_dynamics.alpha_init == pytest.approx(0.1)


def test_yaml_file_wiring(tmp_path):
    path = tmp_path / "tau_z0.yaml"
    path.write_text(
        "state_dynamics:\n"
        "  enabled: true\n"
        "  learnable_z0: true\n"
        "  alpha_init: 0.2\n",
        encoding="utf-8",
    )
    cfg = V3Config.from_yaml(path, strict=True)
    assert cfg.state_dynamics.enabled is True
    assert cfg.state_dynamics.learnable_z0 is True
    assert cfg.state_dynamics.trainable_alpha is False
    assert cfg.state_dynamics.alpha_init == pytest.approx(0.2)


def test_dotted_overrides_reach_the_block():
    cfg = apply_override_items(
        _cfg(),
        [
            "state_dynamics.enabled=true",
            "state_dynamics.trainable_alpha=true",
            "state_dynamics.alpha_init=0.2",
        ],
    )
    assert cfg.state_dynamics.enabled is True
    assert cfg.state_dynamics.trainable_alpha is True
    assert cfg.state_dynamics.alpha_init == pytest.approx(0.2)
    assert cfg.state_dynamics.learnable_z0 is False
    # other blocks keep their values
    assert cfg.state_regularization.mode == _cfg().state_regularization.mode
    with pytest.raises(ValueError):
        apply_override_items(_cfg(), ["state_dynamics.enable=true"])


def test_with_overrides_and_dict_roundtrip():
    sd = StateDynamicsConfig(enabled=True, learnable_z0=True)
    cfg = _cfg().with_overrides(state_dynamics=sd)
    assert cfg.state_dynamics is sd
    data = cfg.to_dict()
    assert isinstance(data["state_dynamics"], dict)
    back = V3Config.from_dict(data, strict=True)
    assert back.state_dynamics == sd


def test_validate_rejects_unusable_alpha_init_when_alpha_is_trainable():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            V3Config(
                state_dynamics=StateDynamicsConfig(
                    enabled=True, trainable_alpha=True, alpha_init=bad
                )
            ).validate()
    # an inert flag must not make a config unusable ...
    V3Config(state_dynamics=StateDynamicsConfig(alpha_init=0.0)).validate()
    # ... and a valid init is accepted
    V3Config(
        state_dynamics=StateDynamicsConfig(
            enabled=True, trainable_alpha=True, alpha_init=0.3
        )
    ).validate()
    with pytest.raises(ValueError):
        V3Config(state_dynamics=StateDynamicsConfig(alpha_init=math.inf)).validate()


# ---------------------------------------------------------------------- #
# flag-off equivalence
# ---------------------------------------------------------------------- #
def test_flags_off_is_bit_identical_and_adds_no_parameters():
    x = _x()
    base = _model(_cfg())
    # sub-flags staged but the master switch is off -> inert
    staged = _model(_cfg(trainable_alpha=True, learnable_z0=True, alpha_init=0.3))
    assert base.raw_alpha is None and base.z0 is None
    assert staged.raw_alpha is None and staged.z0 is None
    assert base.n_parameters() == staged.n_parameters()
    with torch.no_grad():
        logits_base, spikes_base = base(x)
        logits_staged, spikes_staged = staged(x)
    assert torch.equal(logits_base, logits_staged)
    assert torch.equal(spikes_base, spikes_staged)


def test_only_the_requested_parameter_is_created():
    alpha_only = _model(_cfg(enabled=True, trainable_alpha=True))
    assert alpha_only.raw_alpha is not None and alpha_only.z0 is None
    z0_only = _model(_cfg(enabled=True, learnable_z0=True))
    assert z0_only.raw_alpha is None and z0_only.z0 is not None


# ---------------------------------------------------------------------- #
# alpha parameterisation
# ---------------------------------------------------------------------- #
def test_alpha_parameterisation_matches_alpha_init():
    model = _model(_cfg(enabled=True, trainable_alpha=True, alpha_init=0.1))
    assert model.raw_alpha is not None
    assert model.raw_alpha.shape == (N,)
    alpha = torch.sigmoid(model.raw_alpha.detach())
    assert torch.allclose(alpha, torch.full((N,), 0.1), atol=1e-6)
    per_neuron = model.alpha_per_neuron()
    assert per_neuron is not None
    assert per_neuron.shape == (N,)
    assert torch.allclose(per_neuron, alpha)

    other = _model(_cfg(enabled=True, trainable_alpha=True, alpha_init=0.35))
    assert torch.allclose(
        torch.sigmoid(other.raw_alpha.detach()), torch.full((N,), 0.35), atol=1e-6
    )


def test_alpha_is_used_per_neuron_in_the_state_update():
    model = _model(_cfg(enabled=True, trainable_alpha=True, alpha_init=0.25))
    assert model.raw_alpha is not None
    with torch.no_grad():
        model.raw_alpha[0] += 1.0  # make one neuron's leak different
    alpha = model.alpha_per_neuron().detach()
    assert not torch.allclose(alpha[0], alpha[1])

    x = _x(steps=1)  # one step: z(1) = alpha_i * tanh(gate_i * a)
    with torch.no_grad():
        _, _, z_final, _ = model(x, return_states=True)
    a = x[:, 0].to(model.state_dtype) @ model.w_in.to(model.state_dtype)  # (B, D)
    gate = model.gate.to(model.state_dtype)                              # (N, D)
    expected = alpha[None, :, None] * torch.tanh(gate[None] * a.unsqueeze(1))
    assert torch.allclose(z_final, expected, atol=1e-5)


# ---------------------------------------------------------------------- #
# z0 parameterisation
# ---------------------------------------------------------------------- #
def test_z0_shape_and_zero_initialisation():
    model = _model(_cfg(enabled=True, learnable_z0=True))
    assert model.z0 is not None
    assert model.z0.shape == (N, D)
    assert torch.equal(model.z0.detach(), torch.zeros(N, D))
    z = model.init_state(B, torch.device("cpu"))
    assert z.shape == (B, N, D)
    assert torch.equal(z, torch.zeros(B, N, D))


def test_z0_is_used_at_the_first_timestep():
    model = _model(_cfg(enabled=True, learnable_z0=True))
    assert model.z0 is not None
    with torch.no_grad():
        model.z0.fill_(0.5)
    x = _x()
    with torch.no_grad():
        _, _, pots = model(x, return_potentials=True)
    expected_t0 = (
        model.z0.detach().to(model.state_dtype) * model.w_out.to(model.state_dtype)
    ).sum(-1) + model.theta.to(model.state_dtype)
    assert torch.allclose(
        pots[:, 0], expected_t0.unsqueeze(0).expand(B, N), atol=1e-6
    )
    # the same weights with z(0) = 0 read a different potential at t = 0
    base = _model(_cfg())
    with torch.no_grad():
        _, _, pots_base = base(x, return_potentials=True)
    assert not torch.allclose(pots[:, 0], pots_base[:, 0])


# ---------------------------------------------------------------------- #
# gradients
# ---------------------------------------------------------------------- #
def test_gradients_reach_alpha_and_z0_when_enabled():
    model = _model(_cfg(enabled=True, trainable_alpha=True, learnable_z0=True))
    x = _x()
    _, spikes = model(x)
    spikes.float().sum().backward()
    for name, param in (("raw_alpha", model.raw_alpha), ("z0", model.z0)):
        assert param is not None
        assert param.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(param.grad).all()
        assert float(param.grad.abs().sum()) > 0.0, f"zero gradient for {name}"
