"""Focused tests for the spike-frequency adaptation channel.

CPU-only and tiny: these exercise the config plumbing, the flag-off baseline
equivalence, the ``b <- rho*b + s`` trace recurrence, the ``w_ad`` gradient and
the carry of ``b`` through the gradient-checkpointed segment path.  No dataset
and no training run is needed.
"""

from __future__ import annotations

import pytest
import torch
import yaml

from v3.config import (
    AdaptationConfig,
    TimingConfig,
    V3Config,
    apply_override_items,
    apply_override_mapping,
    parameter_groups,
)
from v3.model import VectorNeuronPopulation

B, T, C = 4, 8, 10


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def _cfg(adaptation: AdaptationConfig | None = None) -> V3Config:
    """Tiny CPU config: T = 8 timesteps, and the adaptation block as requested."""
    cfg = V3Config(
        n_neurons=4,
        state_dim=16,
        mix_rank=4,
        n_inputs=C,
        n_classes=3,
        timing=TimingConfig(sequence_duration_ms=16.0, time_bin_ms=2.0),  # T = 8
        batch_size=B,
        device="cpu",
        dtype="float32",
        amp=False,
        grad_checkpoint_chunks=2,
    )
    if adaptation is not None:
        cfg = cfg.with_overrides(adaptation=adaptation)
    return cfg


def _model(cfg: V3Config) -> VectorNeuronPopulation:
    return VectorNeuronPopulation(cfg)


def _raw_inputs(seed: int = 0) -> torch.Tensor:
    """Binary spike input of the shape forward() expects: ``(B, T, C)``."""
    torch.manual_seed(seed)
    return torch.randint(0, 2, (B, T, C)).float()


def _projected_inputs(cfg: V3Config, seed: int = 0) -> torch.Tensor:
    """Already-projected input ``(B, T, D)``, i.e. what ``_segment`` consumes."""
    torch.manual_seed(seed)
    return torch.randn(B, T, cfg.state_dim)


def _tonic(model: VectorNeuronPopulation) -> None:
    """Force a spike train that is independent of the (random) weights.

    ``w_out = 0`` makes the raw drive exactly zero, so ``pot = theta - w_ad*b``;
    ``theta = 1`` makes every neuron spike at every timestep when ``w_ad = 0``.
    """
    with torch.no_grad():
        model.theta.fill_(1.0)
        model.w_out.zero_()


# ---------------------------------------------------------------------- #
# config: defaults, validation, override grammar, roundtrips
# ---------------------------------------------------------------------- #
def test_adaptation_defaults_are_off():
    ad = AdaptationConfig()
    assert ad.enabled is False
    assert ad.rho == pytest.approx(0.95)
    assert ad.gain_init == pytest.approx(0.1)
    assert ad.learnable_gain is True
    # the model config carries the block with exactly those defaults
    assert V3Config().adaptation == ad
    assert V3Config().adaptation.enabled is False


def test_adaptation_validation_rejects_unstable_decay():
    # rho must lie in [0, 1): rho = 1 integrates the spike trace without bound
    AdaptationConfig(enabled=True, rho=0.0).validate()
    AdaptationConfig(enabled=True, rho=0.999).validate()
    for bad in (
        AdaptationConfig(enabled=True, rho=-0.01),
        AdaptationConfig(enabled=True, rho=1.0),
        AdaptationConfig(enabled=True, rho=float("nan")),
        AdaptationConfig(gain_init=float("inf")),
    ):
        with pytest.raises(ValueError):
            bad.validate()
    # the sub-validator is wired into V3Config.validate()
    with pytest.raises(ValueError):
        V3Config(adaptation=AdaptationConfig(rho=1.5)).validate()


def test_dotted_override_grammar_addresses_the_adaptation_block():
    base = V3Config()
    out = apply_override_items(
        base,
        [
            "adaptation.enabled=true",
            "adaptation.rho=0.9",
            "adaptation.gain_init=0.25",
            "adaptation.learnable_gain=false",
        ],
    )
    assert out.adaptation.enabled is True
    assert out.adaptation.rho == pytest.approx(0.9)
    assert isinstance(out.adaptation.rho, float)  # CLI strings are coerced
    assert out.adaptation.gain_init == pytest.approx(0.25)
    assert out.adaptation.learnable_gain is False
    assert base.adaptation.enabled is False  # the input config is not mutated


def test_nested_override_mapping_merges_into_the_adaptation_block():
    out = apply_override_mapping(
        V3Config(), {"adaptation": {"enabled": True, "rho": 0.5}}
    )
    assert out.adaptation.enabled is True
    assert out.adaptation.rho == pytest.approx(0.5)
    assert out.adaptation.gain_init == pytest.approx(0.1)  # unmentioned -> default
    assert out.adaptation.learnable_gain is True
    with pytest.raises(ValueError):
        apply_override_items(V3Config(), ["adaptation.nope=1"])
    with pytest.raises(ValueError):
        apply_override_mapping(V3Config(), {"adaptation": {"nope": 1}})


def test_adaptation_survives_dict_and_yaml_roundtrips(tmp_path):
    cfg = _cfg(AdaptationConfig(enabled=True, rho=0.8, gain_init=0.3, learnable_gain=False))
    data = cfg.to_dict()
    assert isinstance(data["adaptation"], dict)
    assert data["adaptation"]["enabled"] is True
    back = V3Config.from_dict(data, strict=True)
    assert back.adaptation == cfg.adaptation
    assert back == cfg

    path = tmp_path / "v3_adaptation.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    loaded = V3Config.from_yaml(path, strict=True)
    assert loaded.adaptation == cfg.adaptation


def test_parameter_budget_tracks_the_adaptation_gain():
    base = _cfg()
    learnable = _cfg(AdaptationConfig(enabled=True))
    fixed = _cfg(AdaptationConfig(enabled=True, learnable_gain=False))
    assert parameter_groups(base)["total"] == _model(base).n_parameters()
    assert parameter_groups(learnable)["total"] == _model(learnable).n_parameters()
    assert parameter_groups(fixed)["total"] == _model(fixed).n_parameters()
    assert "w_ad" not in parameter_groups(base)
    assert parameter_groups(learnable)["w_ad"] == learnable.n_neurons
    assert "w_ad" not in parameter_groups(fixed)  # a buffer is not a parameter


# ---------------------------------------------------------------------- #
# parameter registration
# ---------------------------------------------------------------------- #
def test_flag_off_registers_nothing_and_keeps_the_three_tuple():
    model = _model(_cfg())
    assert model.w_ad is None
    assert "w_ad" not in model.state_dict()
    assert all(name != "w_ad" for name, _ in model.named_parameters())
    z0 = model.init_state(B, torch.device("cpu"))
    out = model._segment(z0, _projected_inputs(model.cfg))
    assert len(out) == 3  # every external caller unpacks exactly three values


def test_learnable_gain_starts_at_gain_init():
    cfg = _cfg(AdaptationConfig(enabled=True, gain_init=0.37))
    model = _model(cfg)
    assert model.w_ad.shape == (cfg.n_neurons,)
    assert model.w_ad.requires_grad is True
    assert "w_ad" in dict(model.named_parameters())
    assert torch.allclose(model.w_ad, torch.full((cfg.n_neurons,), 0.37))


def test_fixed_gain_is_a_buffer_without_gradient():
    cfg = _cfg(AdaptationConfig(enabled=True, gain_init=0.5, learnable_gain=False))
    model = _model(cfg)
    assert model.w_ad.requires_grad is False
    assert "w_ad" in model.state_dict()
    assert "w_ad" not in dict(model.named_parameters())
    assert torch.allclose(model.w_ad, torch.full((cfg.n_neurons,), 0.5))


# ---------------------------------------------------------------------- #
# flag-off equivalence with the baseline forward
# ---------------------------------------------------------------------- #
def test_zero_gain_enabled_forward_is_bit_identical_to_the_baseline():
    x = _raw_inputs()
    base = _model(_cfg())  # the pre-feature code path
    on = _model(_cfg(AdaptationConfig(enabled=True, gain_init=0.0)))
    base.eval()
    on.eval()
    with torch.no_grad():
        logits_b, spikes_b, pots_b = base(x, return_potentials=True)
        logits_o, spikes_o, pots_o = on(x, return_potentials=True)
    # gain_init = 0 makes the adaptation current vanish, so the whole forward
    # pass must stay bit-identical to the disabled baseline
    assert torch.equal(logits_b, logits_o)
    assert torch.equal(spikes_b, spikes_o)
    assert torch.equal(pots_b, pots_o)


def test_forward_chunking_does_not_change_the_adapted_output():
    cfg = _cfg(AdaptationConfig(enabled=True, gain_init=2.0, rho=0.9))
    model = _model(cfg)
    _tonic(model)  # theta = 1, w_out = 0 -> the adaptation current dominates
    model.eval()
    x = _raw_inputs()
    with torch.no_grad():
        logits2, spikes2 = model(x)

    one_chunk = _model(cfg.with_overrides(grad_checkpoint_chunks=1))
    _tonic(one_chunk)
    one_chunk.eval()
    with torch.no_grad():
        logits1, spikes1 = one_chunk(x)
    # b is carried across the two segments exactly as it would be inside one
    assert torch.equal(spikes2, spikes1)
    assert torch.equal(logits2, logits1)


# ---------------------------------------------------------------------- #
# the adaptation trace itself
# ---------------------------------------------------------------------- #
def test_nonzero_gain_suppresses_the_spike_train():
    base = _model(_cfg())
    on = _model(_cfg(AdaptationConfig(enabled=True, gain_init=2.0, rho=0.9)))
    _tonic(base)
    _tonic(on)
    base.eval()
    on.eval()
    x = _raw_inputs()
    with torch.no_grad():
        _, spikes_base = base(x)
        _, spikes_on = on(x)
    # baseline: pot = theta = 1 at every step -> tonic spiking
    assert torch.equal(spikes_base, torch.ones_like(spikes_base))
    # adapted: pot = 1 - w_ad * b, so only the first timestep spikes
    assert float(spikes_on[:, 0].mean()) == pytest.approx(1.0)
    assert torch.equal(spikes_on[:, 1:], torch.zeros_like(spikes_on[:, 1:]))


def test_b_is_the_decayed_spike_count():
    cfg = _cfg(AdaptationConfig(enabled=True, gain_init=0.0, rho=0.9))
    model = _model(cfg)
    _tonic(model)  # w_ad = 0 -> the trace does not perturb the spike train
    model.eval()
    a = _projected_inputs(cfg)
    z0 = model.init_state(B, torch.device("cpu"))
    z, s, pots, b = model._segment(z0, a)
    assert b.shape == (B, cfg.n_neurons)
    assert b.dtype == model.state_dtype
    assert torch.equal(s, torch.ones_like(s))  # tonic spiking

    expected = torch.zeros_like(b)
    for t in range(s.shape[1]):
        expected = 0.9 * expected + s[:, t]
    assert torch.allclose(b, expected, atol=1e-6)
    # closed form: b_T = sum_j rho**(T-1-j) * s_j
    decayed = sum(
        0.9 ** (s.shape[1] - 1 - j) * s[:, j] for j in range(s.shape[1])
    )
    assert torch.allclose(b, decayed, atol=1e-5)


def test_b_carries_across_segments_like_the_state():
    cfg = _cfg(AdaptationConfig(enabled=True, gain_init=0.0, rho=0.8))
    model = _model(cfg)
    _tonic(model)
    model.eval()
    a = _projected_inputs(cfg)
    z0 = model.init_state(B, torch.device("cpu"))

    _, s_full, _, b_full = model._segment(z0, a)
    split = T // 2
    z1, _, _, b1 = model._segment(z0, a[:, :split])
    z2, s2, _, b2 = model._segment(z1, a[:, split:], b1)
    assert torch.equal(s2, s_full[:, split:])
    assert torch.allclose(b2, b_full, atol=1e-6)
    assert not torch.allclose(b1, b_full)  # the second segment did add to b


# ---------------------------------------------------------------------- #
# gradients
# ---------------------------------------------------------------------- #
def test_w_ad_receives_a_gradient_when_enabled():
    cfg = _cfg(AdaptationConfig(enabled=True, gain_init=0.0))
    model = _model(cfg)
    _tonic(model)  # spikes -> b > 0 -> the adaptation current is live
    model.train()
    x = _raw_inputs()
    logits, spikes, pots = model(x, return_potentials=True)
    pots.sum().backward()
    assert model.w_ad.grad is not None
    assert torch.isfinite(model.w_ad.grad).all()
    # d pot_t / d w_ad = -b_t, and b_t > 0 after the first timestep
    assert float(model.w_ad.grad.abs().sum()) > 0.0


def test_forward_carries_b_through_the_checkpointed_segments(monkeypatch):
    # on CPU forward() only checkpoints when the state requires grad, so make
    # the initial state a leaf that does: this forces the checkpoint path
    cfg = _cfg(AdaptationConfig(enabled=True, gain_init=0.6, rho=0.9))
    model = _model(cfg)
    _tonic(model)
    real_init = model.init_state

    def _graded_init(batch, device):
        return real_init(batch, device).requires_grad_(True)

    monkeypatch.setattr(model, "init_state", _graded_init)
    model.train()
    x = _raw_inputs()
    logits, spikes, pots = model(x, return_potentials=True)
    assert torch.isfinite(logits).all()
    pots.sum().backward()
    assert model.w_ad.grad is not None
    assert torch.isfinite(model.w_ad.grad).all()
    assert float(model.w_ad.grad.abs().sum()) > 0.0

    # the checkpointed segments must reproduce the eager spike train
    model.eval()
    with torch.no_grad():
        _, spikes_eager, _ = model(x, return_potentials=True)
    assert torch.equal(spikes.detach(), spikes_eager)
