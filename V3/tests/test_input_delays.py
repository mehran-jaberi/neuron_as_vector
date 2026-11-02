"""Tests for the *differentiable* per-input-channel conduction delays.

These pin the contract of the fractional (linear-interpolation) delay that
replaced the integer straight-through shift:

* flag off -> no extra parameter, same ``state_dict`` keys, input path untouched;
* learnable delays start strictly inside ``(0, max_ms)`` and differ per channel;
* ``out[t] = (1-f) x[t-k] + f x[t-k-1]`` exactly, with zero fill at the window
  start and a silent channel when ``k >= T``;
* the delay parameter receives a non-zero gradient at initialisation, the
  interpolation is differentiable in the bin count (finite-difference check),
  and one Adam step actually moves the delays;
* the sigmoid parameterisation never leaves ``[0, max_ms]``;
* ``learnable=False`` keeps a frozen, writable delay map.

CPU-only, tiny synthetic sequences; no dataset, no training run.
"""

from __future__ import annotations

import math

import pytest
import torch

from v3.config import (
    InputDelayConfig,
    TimingConfig,
    V3Config,
    parameter_groups,
)
from v3.model import VectorNeuronPopulation


# ---------------------------------------------------------------------- #
# helpers
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
    return _cfg(input_delays=InputDelayConfig(enabled=True, **delay_kwargs))


def _reference_frac_shift(x: torch.Tensor, delays_ms, bin_ms: float) -> torch.Tensor:
    """Explicit per-channel reference: ``(1-f) x[t-k] + f x[t-k-1]``, zero fill."""
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


def _rand(*shape, seed: int = 0) -> torch.Tensor:
    return torch.rand(*shape, generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------------- #
# flag off: bit-identical baseline
# ---------------------------------------------------------------------- #
BASELINE_PARAM_KEYS = (
    "w_in",
    "w_down",
    "w_up",
    "w_emit",
    "w_rec",
    "gate",
    "bias",
    "w_out",
    "theta",
    "w_cls",
)


def test_flag_off_adds_no_parameter_and_keeps_the_state_dict_keys():
    """Baseline ``main`` has exactly these 10 parameters, in this order."""
    cfg = _cfg()
    model = VectorNeuronPopulation(cfg)
    assert model.input_delay_raw is None
    assert tuple(model.state_dict().keys()) == BASELINE_PARAM_KEYS
    assert "input_delays" not in parameter_groups(cfg)
    assert model.n_parameters() == parameter_groups(cfg)["total"]
    with pytest.raises(RuntimeError):
        model.effective_input_delays_ms()


def test_flag_off_input_path_is_the_identity():
    cfg = _cfg()
    model = VectorNeuronPopulation(cfg).eval()
    x = _rand(2, 5, cfg.n_inputs)
    assert model.delayed_input(x) is x


def test_flag_off_forward_never_enters_the_delay_path():
    cfg = _cfg()
    model = VectorNeuronPopulation(cfg).eval()
    x = _rand(2, 6, cfg.n_inputs, seed=1)
    with torch.no_grad():
        ref = model(x)[0]

    def _boom(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("delay path used while input_delays is disabled")

    model._fractional_shift = _boom
    with torch.no_grad():
        got = model(x)[0]
    assert torch.equal(got, ref)


def test_enabling_delays_does_not_perturb_the_other_parameters():
    off = VectorNeuronPopulation(_cfg())
    on = VectorNeuronPopulation(_delay_cfg())
    off_sd, on_sd = off.state_dict(), on.state_dict()
    assert set(on_sd) == set(off_sd) | {"input_delay_raw"}
    for key, value in off_sd.items():
        assert torch.equal(on_sd[key], value), key


# ---------------------------------------------------------------------- #
# initialisation
# ---------------------------------------------------------------------- #
def test_learnable_init_is_interior_and_spread_over_channels():
    cfg = _delay_cfg()
    model = VectorNeuronPopulation(cfg)
    delays = model.effective_input_delays_ms()
    assert delays.shape == (cfg.n_inputs,)
    assert bool(torch.isfinite(delays).all())
    # strictly interior: the old 0.0 init was exactly where clamp's gradient died
    assert float(delays.min()) > 0.0
    assert float(delays.max()) < cfg.input_delays.max_ms
    # drawn from [init_ms, max_ms / 2]
    assert float(delays.min()) >= cfg.input_delays.init_ms - 1e-6
    assert float(delays.max()) <= 0.5 * cfg.input_delays.max_ms + 1e-6
    # per-channel spread, not one shared constant
    assert delays.unique().numel() > 1
    # and the pre-sigmoid parameter is not sitting at the sigmoid's 0.5 point
    assert bool((model.input_delay_raw.detach().abs() > 1e-3).any())


def test_init_is_reproducible_and_follows_init_ms():
    cfg = _delay_cfg(init_ms=2.0, max_ms=8.0)
    a = VectorNeuronPopulation(cfg).effective_input_delays_ms().detach()
    b = VectorNeuronPopulation(cfg).effective_input_delays_ms().detach()
    assert torch.equal(a, b)                       # seeded, no global RNG draw
    assert float(a.min()) >= 2.0 - 1e-6
    assert float(a.max()) <= 4.0 + 1e-6            # max_ms / 2
    other_cfg = _cfg(
        seed=12345,
        input_delays=InputDelayConfig(enabled=True, init_ms=2.0, max_ms=8.0),
    )
    other = VectorNeuronPopulation(other_cfg).effective_input_delays_ms().detach()
    assert not torch.equal(a, other)


def test_init_ms_must_stay_strictly_interior_when_learnable():
    with pytest.raises(ValueError):
        InputDelayConfig(enabled=True, learnable=True, init_ms=0.0).validate()
    with pytest.raises(ValueError):
        InputDelayConfig(enabled=True, learnable=True, max_ms=5.0, init_ms=5.0).validate()
    with pytest.raises(ValueError):
        InputDelayConfig(enabled=True, learnable=True, max_ms=0.0).validate()
    # the forward map divides by max_ms, so a zero cap is only legal while off
    with pytest.raises(ValueError):
        InputDelayConfig(enabled=True, learnable=False, max_ms=0.0).validate()
    # frozen / disabled maps are allowed to sit on a bound
    InputDelayConfig(enabled=True, learnable=False, init_ms=0.0).validate()
    InputDelayConfig(max_ms=0.0).validate()
    InputDelayConfig().validate()


# ---------------------------------------------------------------------- #
# the fractional shift itself
# ---------------------------------------------------------------------- #
def test_fractional_shift_equals_hand_computed_linear_interpolation():
    cfg = _delay_cfg()
    model = VectorNeuronPopulation(cfg).eval()
    T = 7
    x = _rand(1, T, cfg.n_inputs, seed=2)
    # bins 0.5 / 1.5 / 3.75 / 2.5 -- deliberately never a whole number of bins,
    # so the floor() branch cannot flip on a last-bit rounding difference.
    requested = [1.0, 3.0, 7.5] + [5.0] * (cfg.n_inputs - 3)
    model.set_input_delays_ms(requested)
    actual = model.effective_input_delays_ms().detach()
    assert torch.allclose(actual, torch.tensor(requested), atol=1e-5)
    assert torch.allclose(
        model.input_delay_bins().detach(),
        torch.tensor(requested) / cfg.bin_ms,
        atol=1e-5,
    )
    with torch.no_grad():
        got = model.delayed_input(x)
    want = _reference_frac_shift(x, actual.tolist(), cfg.bin_ms)
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-5)


def test_fractional_shift_zero_fills_the_window_start():
    cfg = _delay_cfg()
    model = VectorNeuronPopulation(cfg).eval()
    T = 6
    x = _rand(1, T, cfg.n_inputs, seed=3)
    # 3.0 ms -> 1.5 bins; 13.0 ms -> 6.5 bins >= T -> the channel is silent
    model.set_input_delays_ms([3.0, 13.0] + [1.0] * (cfg.n_inputs - 2))
    with torch.no_grad():
        out = model.delayed_input(x)
    # 1.5 bins: t = 0 reads x[-1] and x[-2] only -> exact zero
    assert torch.equal(out[:, 0, 0], torch.zeros(1))
    # t = 1 reads x[0] with weight (1 - 0.5) and x[-1] (zero)
    assert torch.allclose(out[:, 1, 0], 0.5 * x[:, 0, 0], atol=1e-5)
    # t = 2 = 0.5 x[1] + 0.5 x[0]
    assert torch.allclose(out[:, 2, 0], 0.5 * (x[:, 1, 0] + x[:, 0, 0]), atol=1e-5)
    # k >= T: whole trace silent
    assert torch.equal(out[:, :, 1], torch.zeros(1, T))


def test_fractional_shift_is_applied_before_segmentation():
    """A delay longer than one checkpoint segment still pulls real history."""
    base = _cfg(n_inputs=3, grad_checkpoint_chunks=4)
    on = VectorNeuronPopulation(
        base.with_overrides(input_delays=InputDelayConfig(enabled=True))
    ).eval()
    off = VectorNeuronPopulation(base).eval()
    T, step = 8, 2
    x = _rand(2, T, 3, seed=4)
    delays = [6.5, 0.5, 2.5]                        # 3.25 / 0.25 / 1.25 bins
    on.set_input_delays_ms(delays)
    with torch.no_grad():
        got = on(x, return_states=True)
        ref = off(_reference_frac_shift(x, delays, base.bin_ms), return_states=True)
    for a, b in zip(got, ref):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-3)

    # a segment-local interpolation drops the history a 3.25-bin delay needs
    naive_in = torch.zeros_like(x)
    for t0 in range(0, T, step):
        naive_in[:, t0 : t0 + step] = _reference_frac_shift(
            x[:, t0 : t0 + step], delays, base.bin_ms
        )
    with torch.no_grad():
        naive = off(naive_in, return_states=True)
    assert not torch.allclose(got[-1], naive[-1])


def test_fp16_delayed_input_stays_finite_and_differentiable():
    cfg = _cfg(dtype="float16", input_delays=InputDelayConfig(enabled=True))
    model = VectorNeuronPopulation(cfg)
    x = (_rand(2, 6, cfg.n_inputs, seed=5) > 0.7).to(torch.float16)
    out = model.delayed_input(x)
    assert out.dtype == torch.float16
    assert bool(torch.isfinite(out).all())
    assert float(out.max()) <= 1.0 + 1e-3
    out.float().sum().backward()
    grad = model.input_delay_raw.grad
    assert grad is not None and grad.dtype == torch.float32
    assert bool(torch.isfinite(grad).all())


# ---------------------------------------------------------------------- #
# gradients (the bug that made the first implementation inert)
# ---------------------------------------------------------------------- #
def test_delay_parameter_gets_a_gradient_at_init():
    cfg = _delay_cfg()
    model = VectorNeuronPopulation(cfg)
    x = _rand(1, 8, cfg.n_inputs, seed=6)
    weight = torch.randn(x.shape, generator=torch.Generator().manual_seed(7))
    out = model.delayed_input(x)
    (out * weight).sum().backward()
    grad = model.input_delay_raw.grad
    assert grad is not None and grad.shape == (cfg.n_inputs,)
    assert bool(torch.isfinite(grad).all())
    assert float(grad.abs().max()) > 0.0
    # nearly every channel must receive gradient, not one lucky channel
    assert int((grad.abs() > 0).sum()) >= cfg.n_inputs // 2


def test_interpolation_matches_finite_differences_in_the_bins():
    cfg = _delay_cfg()
    model = VectorNeuronPopulation(cfg).eval()
    x = _rand(2, 9, 3, seed=8)
    weight = torch.randn(x.shape, generator=torch.Generator().manual_seed(9))
    bins = torch.tensor([0.4, 1.3, 2.7], requires_grad=True)
    out = model._fractional_shift(x, bins)
    (out * weight).sum().backward()
    grad = bins.grad
    assert grad is not None and bool((grad.abs() > 0).all())

    eps = 1e-3          # far from an integer boundary, so the branch is stable
    for c in range(3):
        plus = bins.detach().clone()
        minus = bins.detach().clone()
        plus[c] += eps
        minus[c] -= eps
        with torch.no_grad():
            lp = float((model._fractional_shift(x, plus) * weight).sum())
            lm = float((model._fractional_shift(x, minus) * weight).sum())
        fd = (lp - lm) / (2.0 * eps)
        assert fd == pytest.approx(float(grad[c]), rel=2e-3, abs=1e-4)


def test_one_adam_step_moves_the_delays():
    cfg = _delay_cfg()
    model = VectorNeuronPopulation(cfg)
    max_ms = cfg.input_delays.max_ms
    before = model.effective_input_delays_ms().detach().clone()
    opt = torch.optim.Adam([model.input_delay_raw], lr=0.05)
    x = _rand(2, 10, cfg.n_inputs, seed=10)
    weight = torch.randn(x.shape, generator=torch.Generator().manual_seed(11))
    opt.zero_grad()
    (model.delayed_input(x) * weight).sum().backward()
    opt.step()
    after = model.effective_input_delays_ms().detach()
    moved = (after - before).abs()
    assert float(moved.max()) > 0.0
    assert int((moved > 0).sum()) >= cfg.n_inputs // 2
    assert bool((after > 0.0).all()) and bool((after < max_ms).all())


# ---------------------------------------------------------------------- #
# range, fixed delays
# ---------------------------------------------------------------------- #
def test_max_ms_bound_is_respected_without_a_forward_clamp():
    cfg = _delay_cfg(max_ms=20.0)
    model = VectorNeuronPopulation(cfg)
    for extreme in (1e4, -1e4):
        with torch.no_grad():
            model.input_delay_raw.fill_(extreme)
        delays = model.effective_input_delays_ms()
        assert bool(torch.isfinite(delays).all())
        assert float(delays.min()) >= 0.0
        assert float(delays.max()) <= cfg.input_delays.max_ms


def test_clamp_input_delays_projects_the_raw_parameter_and_is_idempotent():
    cfg = _delay_cfg()
    model = VectorNeuronPopulation(cfg)
    with torch.no_grad():
        model.input_delay_raw.fill_(1e4)
    assert model.clamp_input_delays_() == cfg.n_inputs
    assert float(model.input_delay_raw.abs().max()) <= 10.0
    # still strictly inside the bounds, so sigmoid keeps a non-zero slope
    assert 0.0 < float(model.effective_input_delays_ms().min())
    assert float(model.effective_input_delays_ms().max()) < cfg.input_delays.max_ms
    assert model.clamp_input_delays_() == 0

    # entering the train phase (the per-epoch boundary) clamps as well
    with torch.no_grad():
        model.input_delay_raw.fill_(-1e4)
    model.set_phase("train")
    assert float(model.input_delay_raw.min()) >= -10.0
    assert 0.0 < float(model.effective_input_delays_ms().min())
    assert model.clamp_input_delays_() == 0


def test_clamp_input_delays_is_a_noop_when_disabled():
    model = VectorNeuronPopulation(_cfg())
    assert model.clamp_input_delays_() == 0


def test_frozen_delays_are_fixed_but_writable():
    cfg = _delay_cfg(learnable=False)
    model = VectorNeuronPopulation(cfg)
    assert model.input_delay_raw.requires_grad is False
    model.set_input_delays_ms([5.0, 1.5] + [4.0] * (cfg.n_inputs - 2))
    x = _rand(1, 6, cfg.n_inputs, seed=12)
    out = model.delayed_input(x)
    assert out.requires_grad is False
    want = _reference_frac_shift(
        x, model.effective_input_delays_ms().detach().tolist(), cfg.bin_ms
    )
    assert torch.allclose(out, want, atol=1e-5, rtol=1e-5)


def test_input_delays_enabled_models_report_one_extra_parameter_group():
    base = _cfg()
    on_cfg = base.with_overrides(input_delays=InputDelayConfig(enabled=True))
    on = VectorNeuronPopulation(on_cfg)
    groups = parameter_groups(on_cfg)
    assert groups["input_delays"] == base.n_inputs
    assert groups["total"] == on.n_parameters()
    assert groups["total"] == parameter_groups(base)["total"] + base.n_inputs
