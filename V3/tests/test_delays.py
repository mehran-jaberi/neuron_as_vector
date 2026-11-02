"""Focused tests for the learnable per-input-channel conduction delays.

CPU-only, tiny synthetic sequences; no dataset and no training run.
"""

from __future__ import annotations

import math

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
    model.set_input_delays_ms(delays_ms)


def _reference_frac_shift(x: torch.Tensor, delays_ms, bin_ms: float) -> torch.Tensor:
    """``out[t, c] = (1-f) x[t-k, c] + f x[t-k-1, c]`` with zero fill at the start."""
    B, T, C = x.shape
    out = torch.zeros_like(x)
    for c in range(C):
        d = float(delays_ms[c]) / float(bin_ms)
        k = int(math.floor(d))
        f = d - k
        for b in range(B):
            for t in range(T):
                v = 0.0
                if 0 <= t - k < T:
                    v += (1.0 - f) * float(x[b, t - k, c])
                if 0 <= t - k - 1 < T:
                    v += f * float(x[b, t - k - 1, c])
                out[b, t, c] = v
    return out


# ---------------------------------------------------------------------- #
# model behaviour
# ---------------------------------------------------------------------- #
def test_disabled_model_is_the_untouched_baseline():
    cfg = _cfg()
    model = VectorNeuronPopulation(cfg)
    assert model.input_delay_raw is None
    x = torch.rand(2, 5, cfg.n_inputs)
    assert model.delayed_input(x) is x
    with pytest.raises(RuntimeError):
        model.effective_input_delays_ms()
    assert "input_delays" not in parameter_groups(cfg)
    assert parameter_groups(cfg)["total"] == model.n_parameters()


def test_enabling_delays_adds_one_parameter_and_keeps_the_weights():
    base = _cfg()
    on_cfg = base.with_overrides(input_delays=InputDelayConfig(enabled=True))
    on = VectorNeuronPopulation(on_cfg).eval()
    off = VectorNeuronPopulation(base).eval()  # same seed -> identical weights
    assert on.input_delay_raw is not None
    assert on.input_delay_raw.numel() == base.n_inputs
    assert "input_delay_raw" not in off.state_dict()
    for key, value in off.state_dict().items():
        assert torch.equal(on.state_dict()[key], value), key
    assert parameter_groups(on_cfg)["total"] == on.n_parameters()
    assert parameter_groups(on_cfg)["total"] == parameter_groups(base)["total"] + base.n_inputs

    # the delays start strictly inside (0, max_ms) and differ across channels
    delays = on.effective_input_delays_ms()
    assert float(delays.min()) > 0.0
    assert float(delays.max()) < base.input_delays.max_ms
    assert delays.unique().numel() > 1

    x = torch.rand(2, 6, base.n_inputs)
    with torch.no_grad():
        got = on(x, return_states=True)
        ref = off(x, return_states=True)
    assert len(got) == len(ref) == 4
    assert not torch.equal(got[-1], ref[-1])  # a non-zero delay changes the run


def test_fractional_shift_matches_the_reference_interpolation():
    cfg = _delay_cfg()
    model = VectorNeuronPopulation(cfg).eval()
    T = 6
    x = (
        torch.arange(T).view(1, T, 1) * 10.0
        + torch.arange(cfg.n_inputs).view(1, 1, -1)
    ).to(torch.float32)
    # dt = 2 ms: 1/3/5/9/13 ms are 0.5/1.5/2.5/4.5/6.5 bins and 6.5 bins empty
    # the whole window.  The delay is fractional, so the value is interpolated
    # between the two neighbouring bins instead of being rounded.
    delays = [1.0, 3.0, 5.0, 9.0, 13.0, 7.5, 2.5, 0.5]
    _set_delays(model, delays)
    actual = model.effective_input_delays_ms().detach()
    assert torch.allclose(actual, torch.tensor(delays), atol=1e-5)
    assert torch.allclose(
        model.input_delay_bins().detach(), torch.tensor(delays) / cfg.bin_ms, atol=1e-5
    )
    with torch.no_grad():
        out = model.delayed_input(x)
    assert torch.allclose(
        out, _reference_frac_shift(x, actual.tolist(), cfg.bin_ms), atol=1e-3
    )
    assert torch.equal(out[:, :, 4], torch.zeros(1, T))  # 6.5 bins >= T silences it


def test_delay_bins_follow_the_run_dt():
    cfg = _cfg(
        input_delays=InputDelayConfig(enabled=True),
        timing=TimingConfig(sequence_duration_ms=1000.0, time_bin_ms=4.0),
    )
    assert cfg.bin_ms == 4.0 and cfg.n_bins == 250
    model = VectorNeuronPopulation(cfg)
    _set_delays(model, 10.0)  # 10 ms / 4 ms = 2.5 bins, kept fractional
    assert model.input_delay_bins().tolist() == pytest.approx([2.5] * cfg.n_inputs)


def test_far_delays_stay_in_range_and_the_safety_clamp_projects_raw():
    cfg = _delay_cfg(max_ms=20.0)
    model = VectorNeuronPopulation(cfg)
    _set_delays(model, [-5.0, 0.0, 100.0] + [5.0] * (cfg.n_inputs - 3))
    eff = model.effective_input_delays_ms()
    assert float(eff[0]) < 1e-3          # 20 * sigmoid(-13.8) ~ 2e-5
    assert float(eff[1]) < 1e-3
    assert 19.9 < float(eff[2]) < 20.0
    assert bool((eff >= 0.0).all()) and bool((eff <= 20.0).all())
    # the raw parameter drifted outside the numerical guard's window
    assert model.clamp_input_delays_() == 3
    assert float(model.input_delay_raw.abs().max()) <= 10.0
    assert 0.0 < float(model.effective_input_delays_ms().min())
    assert model.clamp_input_delays_() == 0


def test_delay_gradient_follows_the_shift_not_a_dc_bias():
    """A ramp input: the gradient must be the time-shift one, with a negative sign.

    For ``out[t] = (1-f) x[t-k] + f x[t-k-1]`` and ``x[t] = t`` the derivative
    w.r.t. the delay is ``-(T - k - 1)`` bins (``x[0] = 0`` kills the ``t = k``
    term), i.e. ``-(T - k - 1) / bin_ms`` in ms -- negative.  The old
    straight-through term produced ``+B * T / bin_ms``, a DC bias.
    """
    cfg = _delay_cfg()
    model = VectorNeuronPopulation(cfg).eval()
    T = 8
    x = torch.arange(T, dtype=torch.float32).view(1, T, 1).repeat(1, 1, cfg.n_inputs)
    _set_delays(model, 5.0)                       # 2.5 bins -> k = 2, f = 0.5
    assert model.input_delay_bins().tolist() == pytest.approx([2.5] * cfg.n_inputs)
    out = model.delayed_input(x)
    out.sum().backward()
    grad = model.input_delay_raw.grad
    assert grad is not None and bool((grad < 0).all())
    s = torch.sigmoid(model.input_delay_raw.detach())
    expected = -(T - 3) / cfg.bin_ms * cfg.input_delays.max_ms * s * (1.0 - s)
    assert torch.allclose(grad, expected, rtol=1e-4, atol=1e-6)


def test_frozen_delays_are_not_trainable():
    cfg = _delay_cfg(learnable=False)
    model = VectorNeuronPopulation(cfg)
    assert model.input_delay_raw is not None
    assert model.input_delay_raw.requires_grad is False
    out = model.delayed_input(torch.rand(1, 3, cfg.n_inputs))
    assert out.requires_grad is False


def test_the_shift_spans_the_time_segments():
    # 8 steps in 4 checkpoint-sized segments of 2: a 3.25-bin delay pulls
    # samples from a previous segment, so the shifted sequence must be built
    # once for the whole window instead of segment by segment.
    base = _cfg(n_inputs=3, grad_checkpoint_chunks=4)
    on_cfg = base.with_overrides(input_delays=InputDelayConfig(enabled=True))
    on = VectorNeuronPopulation(on_cfg).eval()
    off = VectorNeuronPopulation(base).eval()
    T, step = 8, 2
    x = torch.rand(2, T, 3)
    delays = [6.5, 0.5, 2.5]  # 3.25, 0.25, 1.25 bins at dt = 2 ms
    _set_delays(on, delays)
    assert on.input_delay_bins().tolist() == pytest.approx([3.25, 0.25, 1.25], abs=1e-5)
    with torch.no_grad():
        got = on(x, return_states=True)
        ref = off(_reference_frac_shift(x, delays, base.bin_ms), return_states=True)
    for a, b in zip(got, ref):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-3)

    # a segment-local shift zero-fills at every segment start instead, which
    # drops the history a delay longer than one segment needs
    naive_in = torch.zeros_like(x)
    for t0 in range(0, T, step):
        naive_in[:, t0 : t0 + step] = _reference_frac_shift(
            x[:, t0 : t0 + step], delays, base.bin_ms
        )
    with torch.no_grad():
        naive = off(naive_in, return_states=True)
    assert not torch.allclose(got[-1], naive[-1])
