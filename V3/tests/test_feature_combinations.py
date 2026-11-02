"""Combination tests for the three feature branches merged into ``main``.

``main`` now carries three independent, default-off model features from three
branches:

* M1 spike-frequency adaptation (``cfg.adaptation``),
* M2 trainable per-neuron leak and learned initial state (``cfg.state_dynamics``),
* M4 learnable per-input-channel delays (``cfg.input_delays``).

The merge is only useful if the three can be switched on **individually and all
at the same time**, so this module pins that contract down:

1. all three off   -> the parameter set is exactly the baseline one;
2. each one alone  -> the expected extra parameters appear, forward is finite;
3. any pair        -> the expected parameter counts add up, forward is finite;
4. all three at once -> forward + backward are finite and every new parameter
   group actually receives gradient.

CPU only, tiny configuration, a handful of timesteps: this is a plumbing test,
not a training run.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from v3.config import (
    AdaptationConfig,
    InputDelayConfig,
    StateDynamicsConfig,
    TimingConfig,
    V3Config,
)
from v3.model import VectorNeuronPopulation

N, D, C, K, B, T = 4, 8, 6, 3, 2, 4

FEATURES = ("adaptation", "state_dynamics", "input_delays")


def _base() -> V3Config:
    return V3Config(
        n_neurons=N,
        state_dim=D,
        mix_rank=2,
        n_inputs=C,
        n_classes=K,
        timing=TimingConfig(sequence_duration_ms=2.0 * T, time_bin_ms=2.0),
        batch_size=B,
        device="cpu",
        dtype="float32",
        amp=False,
        grad_checkpoint_chunks=2,
    )


def _cfg(*active: str) -> V3Config:
    """Small config with exactly the features in ``active`` switched on."""
    overrides = {}
    if "adaptation" in active:
        overrides["adaptation"] = AdaptationConfig(enabled=True)
    if "state_dynamics" in active:
        overrides["state_dynamics"] = StateDynamicsConfig(
            enabled=True, trainable_alpha=True, learnable_z0=True
        )
    if "input_delays" in active:
        overrides["input_delays"] = InputDelayConfig(enabled=True)
    return _base().with_overrides(**overrides)


def _x() -> torch.Tensor:
    torch.manual_seed(11)
    return torch.randint(0, 2, (B, T, C)).float()


def _extra_parameters(*active: str) -> int:
    """Parameters the given features are expected to add."""
    extra = 0
    if "adaptation" in active:
        extra += N
    if "state_dynamics" in active:
        extra += N + N * D
    if "input_delays" in active:
        extra += C
    return extra


# ---------------------------------------------------------------------- #
# 1. nothing on: the baseline parameter set is untouched
# ---------------------------------------------------------------------- #
def test_no_features_is_the_baseline_parameter_set():
    assert VectorNeuronPopulation(_cfg()).n_parameters() == VectorNeuronPopulation(_base()).n_parameters()


def test_off_forward_matches_the_baseline_bit_for_bit():
    torch.manual_seed(3)
    off = VectorNeuronPopulation(_cfg()).eval()
    x = _x()
    with torch.no_grad():
        logits_off, spikes_off = off(x)
    baseline = VectorNeuronPopulation(_base()).eval()
    torch.manual_seed(3)
    baseline = VectorNeuronPopulation(_base()).eval()
    with torch.no_grad():
        logits_base, spikes_base = baseline(x)
    assert torch.equal(logits_off, logits_base)
    assert torch.equal(spikes_off, spikes_base)


# ---------------------------------------------------------------------- #
# 2. each feature alone, 3. any pair
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "active",
    list(itertools.chain.from_iterable(
        itertools.combinations(FEATURES, k) for k in (1, 2)
    )),
)
def test_feature_subsets_add_exactly_their_parameters_and_run(active: tuple[str, ...]):
    cfg = _cfg(*active)
    model = VectorNeuronPopulation(cfg).eval()
    expected = VectorNeuronPopulation(_base()).n_parameters() + _extra_parameters(*active)
    assert model.n_parameters() == expected
    with torch.no_grad():
        logits, spikes = model(_x())
    assert torch.isfinite(logits).all()
    assert torch.isfinite(spikes).all()


# ---------------------------------------------------------------------- #
# 4. all three at once: forward and backward work, every group gets gradient
# ---------------------------------------------------------------------- #
def test_all_three_features_train_together():
    cfg = _cfg(*FEATURES)
    model = VectorNeuronPopulation(cfg).train()
    expected = VectorNeuronPopulation(_base()).n_parameters() + _extra_parameters(*FEATURES)
    assert model.n_parameters() == expected

    logits, spikes = model(_x())
    loss = logits.pow(2).mean() + 0.1 * spikes.pow(2).mean()
    loss.backward()

    assert torch.isfinite(loss).all()
    for name in ("w_ad", "raw_alpha", "z0", "input_delay_raw"):
        param = dict(model.named_parameters())[name]
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all()
        assert float(param.grad.abs().sum()) > 0.0, f"{name} gradient is zero"


def test_all_three_features_keep_eval_forward_finite():
    model = VectorNeuronPopulation(_cfg(*FEATURES)).eval()
    with torch.no_grad():
        logits, spikes = model(_x())
    assert torch.isfinite(logits).all()
    assert torch.isfinite(spikes).all()
    # delays stay strictly interior (never on a dead clamp boundary)
    delays = model.effective_input_delays_ms()
    assert float(delays.min()) > 0.0
    assert float(delays.max()) < float(model.cfg.input_delays.max_ms)


def test_each_feature_can_be_switched_off_again_by_overrides():
    """A run may start from a fully enabled config and disable one feature."""
    cfg = _cfg(*FEATURES).with_overrides(
        **{"adaptation.enabled": False, "state_dynamics.enabled": False}
    )
    model = VectorNeuronPopulation(cfg)
    expected = VectorNeuronPopulation(_base()).n_parameters() + _extra_parameters("input_delays")
    assert model.n_parameters() == expected
