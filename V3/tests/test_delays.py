"""Focused tests for the learnable per-input-channel conduction delays.

CPU-only, tiny synthetic sequences; no dataset and no training run.
"""

from __future__ import annotations

import pytest
import torch

from v3.config import (
    InputDelayConfig,
    TimingConfig,
    V3Config,
    V3_ROOT,
    apply_override_items,
    parameter_groups,
)
from v3.model import VectorNeuronPopulation


# ---------------------------------------------------------------------- #
# configuration
# ---------------------------------------------------------------------- #
def test_defaults_are_off():
    delay = InputDelayConfig()
    assert delay.enabled is False
    assert delay.max_ms == 20.0
    assert delay.learnable is True
    assert V3Config().input_delays == delay


def test_block_from_dict_and_validation():
    block = InputDelayConfig.from_dict({"enabled": True, "max_ms": 8.0})
    assert block.enabled is True
    assert block.max_ms == 8.0 and block.learnable is True
    with pytest.raises(ValueError):
        InputDelayConfig.from_dict({"maxx_ms": 1.0}, strict=True)
    with pytest.raises(ValueError):
        InputDelayConfig(max_ms=-1.0).validate()
    InputDelayConfig(max_ms=0.0).validate()  # 0 is a valid cap


def test_default_yaml_has_the_block_and_roundtrips():
    cfg = V3Config.from_yaml(V3_ROOT / "configs" / "v3_default.yaml")
    assert cfg.n_inputs == 700 and cfg.bin_ms == 2.0
    assert cfg.input_delays.enabled is False
    assert cfg.input_delays.max_ms == 20.0
    assert cfg.input_delays.learnable is True
    back = V3Config.from_dict(cfg.to_dict(), strict=True)
    assert back.input_delays == cfg.input_delays


def test_dotted_overrides_reach_the_block():
    base = V3Config()
    on = base.with_overrides(**{"input_delays.enabled": True})
    assert on.input_delays.enabled is True
    assert base.input_delays.enabled is False  # the input is never mutated

    items = apply_override_items(
        base,
        [
            "input_delays.enabled=true",
            "input_delays.max_ms=7",
            "input_delays.learnable=false",
        ],
    )
    assert items.input_delays.enabled is True
    assert items.input_delays.max_ms == 7.0
    assert items.input_delays.learnable is False

    replaced = base.with_overrides(input_delays=InputDelayConfig(True, 5.0, False))
    assert replaced.input_delays.enabled is True
    assert replaced.input_delays.max_ms == 5.0
    assert replaced.input_delays.learnable is False


# ---------------------------------------------------------------------- #
# model helpers
# ---------------------------------------------------------------------- #
def _cfg(**kwargs) -> V3Config:
    params = dict(
        n_neurons=2,
        state_dim=4,
        mix_rank=4,
        n_inputs=8,
        n_classes=3,
        tau_ms=20.0,
        timing=TimingConfig(sequence_duration_ms=1000.0, time_bin_ms=2.0),
        device="cpu",
        dtype="float32",
        amp=False,
    )
    params.update(kwargs)
    return V3Config(**params)


def _delay_cfg(**delay_kwargs) -> V3Config:
    """Enabled-delay config; ``delay_kwargs`` go to the ``InputDelayConfig``."""
    return _cfg(input_delays=InputDelayConfig(enabled=True, **delay_kwargs))


def _set_delays(model: VectorNeuronPopulation, delays_ms) -> None:
    """Write the delay parameters (a scalar broadcasts to every channel)."""
    with torch.no_grad():
        model.input_delay_ms.copy_(torch.as_tensor(delays_ms, dtype=torch.float32))


def _manual_shift(x: torch.Tensor, bins) -> torch.Tensor:
    """``out[b, t, c] = x[b, t - bins[c], c]`` with zero fill at the start."""
    out = torch.zeros_like(x)
    T = x.shape[1]
    for c, k in enumerate(bins):
        if k < T:
            out[:, k:, c] = x[:, : T - k, c]
    return out


# ---------------------------------------------------------------------- #
# model behaviour
# ---------------------------------------------------------------------- #
def test_disabled_model_is_the_untouched_baseline():
    cfg = _cfg()
    model = VectorNeuronPopulation(cfg)
    assert model.input_delay_ms is None
    x = torch.rand(2, 5, cfg.n_inputs)
    assert model.delayed_input(x) is x
    with pytest.raises(RuntimeError):
        model.effective_input_delays_ms()
    assert "input_delays" not in parameter_groups(cfg)
    assert parameter_groups(cfg)["total"] == model.n_parameters()


def test_enabled_at_init_equals_the_baseline():
    base = _cfg()
    on_cfg = base.with_overrides(input_delays=InputDelayConfig(enabled=True))
    on = VectorNeuronPopulation(on_cfg).eval()
    off = VectorNeuronPopulation(base).eval()  # same seed -> identical weights
    assert on.input_delay_ms is not None
    assert on.input_delay_ms.numel() == base.n_inputs
    assert torch.equal(on.effective_input_delays_ms(), torch.zeros(base.n_inputs))
    assert parameter_groups(on_cfg)["total"] == on.n_parameters()
    assert parameter_groups(on_cfg)["total"] == parameter_groups(base)["total"] + base.n_inputs

    x = torch.rand(2, 6, base.n_inputs)
    assert torch.equal(on.delayed_input(x), x)  # zero delay, zero fill
    with torch.no_grad():
        got = on(x, return_states=True)
        ref = off(x, return_states=True)
    assert len(got) == len(ref) == 4
    for a, b in zip(got, ref):
        assert torch.equal(a, b)


def test_shift_by_k_bins_matches_a_manual_shift():
    cfg = _delay_cfg()
    model = VectorNeuronPopulation(cfg).eval()
    T = 6
    x = (
        torch.arange(T).view(1, T, 1) * 10.0
        + torch.arange(cfg.n_inputs).view(1, 1, -1)
    ).to(torch.float32)
    # dt = 2 ms, so 0/2/4/10/12 ms are 0/1/2/5/6 bins and 6 bins empty the whole
    # window.  Fractional delays: 3 ms -> round(1.5) = 2 bins, 1 ms -> 0 bins.
    delays = [0.0, 2.0, 4.0, 10.0, 12.0, 3.0, 1.0, 0.0]
    bins = [0, 1, 2, 5, 6, 2, 0, 0]
    _set_delays(model, delays)
    assert model.input_delay_bins().tolist() == [float(k) for k in bins]
    with torch.no_grad():
        out = model.delayed_input(x)
    assert torch.equal(out, _manual_shift(x, bins))
    assert torch.equal(out[:, :, 4], torch.zeros(1, T))  # 6 bins silences it


def test_delay_bins_follow_the_run_dt():
    cfg = _cfg(
        input_delays=InputDelayConfig(enabled=True),
        timing=TimingConfig(sequence_duration_ms=1000.0, time_bin_ms=4.0),
    )
    assert cfg.bin_ms == 4.0 and cfg.n_bins == 250
    model = VectorNeuronPopulation(cfg)
    _set_delays(model, 10.0)  # 10 ms / 4 ms = 2.5 bins
    assert model.input_delay_bins().tolist() == [2.0] * cfg.n_inputs


def test_far_delays_are_clamped_to_the_range():
    cfg = _delay_cfg(max_ms=20.0)
    model = VectorNeuronPopulation(cfg)
    delays = [-5.0, 0.0, 100.0] + [0.0] * (cfg.n_inputs - 3)
    _set_delays(model, delays)
    eff = model.effective_input_delays_ms()
    assert float(eff[0]) == 0.0
    assert float(eff[1]) == 0.0
    assert float(eff[2]) == pytest.approx(20.0)
    assert float(model.input_delay_bins()[2]) == pytest.approx(10.0)
    with torch.no_grad():
        model.input_delay_ms.fill_(0.0)
        model.input_delay_ms[0] = 42.0
        model.input_delay_ms[1] = -1.0
    assert model.clamp_input_delays_() == 2
    assert float(model.input_delay_ms[0]) == pytest.approx(20.0)
    assert float(model.input_delay_ms[1]) == 0.0
    assert model.clamp_input_delays_() == 0


def test_the_delay_parameter_gets_a_gradient_through_the_ste():
    cfg = _delay_cfg()
    model = VectorNeuronPopulation(cfg).eval()
    _set_delays(model, 5.0)  # 2.5 bins: the rounding is active
    x = torch.rand(1, 8, cfg.n_inputs)
    out = model.delayed_input(x)
    out.sum().backward()
    grad = model.input_delay_ms.grad
    assert grad is not None and grad.shape == (cfg.n_inputs,)
    # d(sum(out)) / d(delay_ms) = B * T / dt; the zero-valued straight-through
    # term is the only differentiable path through the integer shift.
    expected = out.shape[0] * out.shape[1] / cfg.bin_ms
    assert torch.allclose(grad, torch.full_like(grad, expected))


def test_frozen_delays_are_not_trainable():
    cfg = _delay_cfg(learnable=False)
    model = VectorNeuronPopulation(cfg)
    assert model.input_delay_ms is not None
    assert model.input_delay_ms.requires_grad is False
    out = model.delayed_input(torch.rand(1, 3, cfg.n_inputs))
    assert out.requires_grad is False


def test_the_shift_spans_the_time_segments():
    # 8 steps in 4 checkpoint-sized segments of 2: a 3-bin delay pulls samples
    # from a previous segment, so the shifted sequence must be built once for
    # the whole window instead of segment by segment.
    base = _cfg(n_inputs=3, grad_checkpoint_chunks=4)
    on_cfg = base.with_overrides(input_delays=InputDelayConfig(enabled=True))
    on = VectorNeuronPopulation(on_cfg).eval()
    off = VectorNeuronPopulation(base).eval()
    T, step = 8, 2
    x = torch.rand(2, T, 3)
    _set_delays(on, [6.0, 0.0, 2.0])  # 3, 0, 1 bins at dt = 2 ms
    assert on.input_delay_bins().tolist() == [3.0, 0.0, 1.0]
    with torch.no_grad():
        got = on(x, return_states=True)
        ref = off(_manual_shift(x, [3, 0, 1]), return_states=True)
    for a, b in zip(got, ref):
        assert torch.equal(a, b)

    # a segment-local shift zero-fills at every segment start instead, which
    # drops the history a delay longer than one segment needs
    naive_in = torch.zeros_like(x)
    for t0 in range(0, T, step):
        naive_in[:, t0 : t0 + step] = _manual_shift(x[:, t0 : t0 + step], [3, 0, 1])
    with torch.no_grad():
        naive = off(naive_in, return_states=True)
    assert not torch.allclose(got[-1], naive[-1])
