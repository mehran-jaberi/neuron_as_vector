"""M10 input delay line: K time-shifted copies of the already-binned input.

CPU-only and tiny: these exercise the shift/sum/concat primitives, the config
validation, the parameter budget and the plumbing into the model forward -- not
any training run.  The contract the model relies on is:

* ``enabled=False`` and ``copies=1`` are the *same object* as the input and
  leave the model's parameters and forward bit-identical to the baseline;
* ``sum`` mode is the exact shifted sum with zeros at the boundary;
* ``concat`` mode stacks the shifted copies along the channel axis and adds
  exactly ``(K-1)*C*D`` parameters to ``w_in``;
* the delay line is applied once, before the temporal loop, and feeds ``w_in``.
"""

from __future__ import annotations

import pytest
import torch

from v3.config import (
    InputDelayLineConfig,
    TimingConfig,
    V3Config,
    apply_override_items,
    apply_override_mapping,
    parameter_groups,
)
from v3.input_delay_line import apply_input_delay_line, delay_line
from v3.model import VectorNeuronPopulation


# ---------------------------------------------------------------------- #
def _tiny_cfg(**kw) -> V3Config:
    """T = 10 bins of 4 ms; C = 6 channels, D = 8, tiny fp32 model on CPU."""
    kw.setdefault("device", "cpu")
    kw.setdefault("dtype", "float32")
    kw.setdefault("amp", False)
    return V3Config(
        n_neurons=4,
        state_dim=8,
        n_inputs=6,
        mix_rank=3,
        timing=TimingConfig(sequence_duration_ms=40.0, time_bin_ms=4.0),
        **kw,
    )


def _block(**kw) -> InputDelayLineConfig:
    kw.setdefault("enabled", True)
    return InputDelayLineConfig(**kw)


# ---------------------------------------------------------------------- #
# primitives
# ---------------------------------------------------------------------- #
X3 = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])   # (B=1, T=3, C=2)


def test_single_copy_is_the_input_object_in_both_modes():
    for mode in ("concat", "sum"):
        out = delay_line(X3, copies=1, stride_bins=3, mode=mode)
        assert out is X3
        block = _block(copies=1, stride_ms=12.0, mode=mode)
        assert apply_input_delay_line(X3, block, bin_ms=4.0) is X3


def test_disabled_block_returns_the_input_object():
    block = InputDelayLineConfig(enabled=False, copies=4, stride_ms=10.0, mode="concat")
    assert apply_input_delay_line(X3, block, bin_ms=4.0) is X3


def test_sum_mode_matches_hand_built_shifted_sum():
    out = delay_line(X3, copies=3, stride_bins=1, mode="sum")
    expected = torch.tensor([[[1.0, 2.0], [4.0, 6.0], [9.0, 12.0]]])
    assert out.shape == X3.shape
    assert torch.equal(out, expected)


def test_sum_mode_stride_two_and_boundary_zeros():
    out = delay_line(X3, copies=3, stride_bins=2, mode="sum")
    # t=0: x[0]; t=1: x[1]; t=2: x[2] + x[0] (t=2-4 < 0 contributes nothing)
    expected = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [6.0, 8.0]]])
    assert torch.equal(out, expected)
    # every delay longer than the sequence: the sum is just the input
    long = delay_line(X3, copies=4, stride_bins=5, mode="sum")
    assert torch.equal(long, X3)


def test_concat_layout_is_shifted_channel_blocks():
    out = delay_line(X3, copies=3, stride_bins=1, mode="concat", dtype=torch.float32)
    assert out.shape == (1, 3, 6)
    expected = torch.tensor(
        [
            [
                [1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                [3.0, 4.0, 1.0, 2.0, 0.0, 0.0],
                [5.0, 6.0, 3.0, 4.0, 1.0, 2.0],
            ]
        ]
    )
    assert torch.equal(out, expected)


def test_concat_dtype_override_and_boundary():
    out = delay_line(X3, copies=2, stride_bins=2, mode="concat", dtype=torch.float16)
    assert out.dtype == torch.float16
    expected = torch.tensor(
        [[[1.0, 2.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0], [5.0, 6.0, 1.0, 2.0]]],
        dtype=torch.float16,
    )
    assert torch.equal(out, expected)


def test_input_is_never_mutated():
    x = X3.clone()
    for mode in ("concat", "sum"):
        delay_line(x, copies=3, stride_bins=1, mode=mode)
        assert torch.equal(x, X3)


# ---------------------------------------------------------------------- #
# parameters and forward identity
# ---------------------------------------------------------------------- #
def test_disabled_block_adds_no_parameters():
    base = VectorNeuronPopulation(_tiny_cfg())
    # deliberately "wild" inert settings: disabled must ignore every knob
    off = VectorNeuronPopulation(
        _tiny_cfg(
            input_delay_line=InputDelayLineConfig(
                enabled=False, copies=9, stride_ms=10.0, mode="sum"
            )
        )
    )
    assert tuple(off.w_in.shape) == (6, 8)
    assert off.n_parameters() == base.n_parameters()
    for (n0, p0), (n1, p1) in zip(base.named_parameters(), off.named_parameters()):
        assert n0 == n1
        assert torch.equal(p0, p1), n0


@pytest.mark.parametrize("mode", ["concat", "sum"])
def test_copies_one_is_bit_identical_to_the_current_forward(mode):
    base = VectorNeuronPopulation(_tiny_cfg())
    one = VectorNeuronPopulation(
        _tiny_cfg(input_delay_line=_block(copies=1, stride_ms=8.0, mode=mode))
    )
    assert one.n_parameters() == base.n_parameters()
    assert tuple(one.w_in.shape) == tuple(base.w_in.shape)
    for (n0, p0), (n1, p1) in zip(base.named_parameters(), one.named_parameters()):
        assert n0 == n1
        assert torch.equal(p0, p1), n0

    g = torch.Generator().manual_seed(7)
    x = (torch.rand(2, 10, 6, generator=g) < 0.2).float()
    base.eval()
    one.eval()
    with torch.no_grad():
        logits0, spikes0 = base(x)
        logits1, spikes1 = one(x)
    assert torch.equal(logits0, logits1)
    assert torch.equal(spikes0, spikes1)


def test_concat_parameter_count_delta_and_analytics():
    C, D = 6, 8
    base = VectorNeuronPopulation(_tiny_cfg())
    cfg = _tiny_cfg(input_delay_line=_block(copies=3, stride_ms=8.0, mode="concat"))
    model = VectorNeuronPopulation(cfg)
    assert tuple(model.w_in.shape) == (3 * C, D)
    assert model.n_parameters() - base.n_parameters() == (3 - 1) * C * D
    groups = parameter_groups(cfg)
    assert groups["w_in"] == 3 * C * D
    assert groups["total"] == model.n_parameters()


def test_sum_mode_leaves_the_parameter_count_unchanged():
    base = VectorNeuronPopulation(_tiny_cfg())
    cfg = _tiny_cfg(input_delay_line=_block(copies=4, stride_ms=8.0, mode="sum"))
    model = VectorNeuronPopulation(cfg)
    assert tuple(model.w_in.shape) == (6, 8)
    assert model.n_parameters() == base.n_parameters()
    assert parameter_groups(cfg) == parameter_groups(_tiny_cfg())


@pytest.mark.parametrize("mode", ["concat", "sum"])
def test_enabled_forward_runs_backward(mode):
    cfg = _tiny_cfg(input_delay_line=_block(copies=2, stride_ms=4.0, mode=mode))
    model = VectorNeuronPopulation(cfg)
    g = torch.Generator().manual_seed(11)
    x = (torch.rand(2, 10, 6, generator=g) < 0.3).float()

    logits, spikes = model(x)
    assert logits.shape == (2, cfg.n_classes)
    assert spikes.shape == (2, 10, 4)
    logits.square().mean().backward()
    assert model.w_in.grad is not None
    assert torch.isfinite(model.w_in.grad).all()
    assert tuple(model.w_in.grad.shape) == tuple(model.w_in.shape)


@pytest.mark.parametrize("mode", ["concat", "sum"])
def test_forward_projection_consumes_the_delayed_input(mode):
    """The temporal loop must receive ``delay_line(x) @ w_in``, not ``x @ w_in``."""
    cfg = _tiny_cfg(input_delay_line=_block(copies=2, stride_ms=8.0, mode=mode))
    model = VectorNeuronPopulation(cfg)
    model.eval()
    g = torch.Generator().manual_seed(5)
    x = (torch.rand(2, 10, 6, generator=g) < 0.2).float()

    chunks = []
    original = model._segment_any

    def spy(z, a_chunk, b=None, u_fast=None, u_slow=None, state_sum=None, cur=None, **kwargs):
        chunks.append(a_chunk)
        return original(z, a_chunk, b, u_fast, u_slow, state_sum, cur, **kwargs)

    model._segment_any = spy
    with torch.no_grad():
        model(x)
    a_model = torch.cat(chunks, dim=1)

    delayed = apply_input_delay_line(
        x, cfg.input_delay_line, bin_ms=cfg.bin_ms, dtype=model.state_dtype
    )
    dt = model.state_dtype
    expected = (delayed.reshape(2 * 10, -1).to(dt) @ model.w_in.to(dt)).reshape(
        2, 10, model.state_dim
    )
    assert torch.equal(a_model, expected)


# ---------------------------------------------------------------------- #
# validation
# ---------------------------------------------------------------------- #
def test_validation_rejects_bad_values():
    with pytest.raises(ValueError):
        InputDelayLineConfig(copies=0).validate()
    with pytest.raises(ValueError):
        InputDelayLineConfig(mode="bogus").validate()
    with pytest.raises(ValueError):
        InputDelayLineConfig(stride_ms=-4.0).validate()
    # concat is capped at 4 copies ...
    with pytest.raises(ValueError):
        _block(copies=5, stride_ms=4.0, mode="concat").validate(
            bin_ms=4.0, sequence_duration_ms=40.0
        )
    # ... sum may use more (no extra parameters)
    _block(copies=8, stride_ms=4.0, mode="sum").validate(
        bin_ms=4.0, sequence_duration_ms=40.0
    )
    # stride must be a whole number of bins ...
    with pytest.raises(ValueError):
        _block(copies=2, stride_ms=10.0).validate(bin_ms=4.0, sequence_duration_ms=40.0)
    # ... must fit inside the sequence ...
    with pytest.raises(ValueError):
        _block(copies=2, stride_ms=44.0).validate(bin_ms=4.0, sequence_duration_ms=40.0)
    # ... and a multi-copy delay line needs a positive stride
    with pytest.raises(ValueError):
        _block(copies=2, stride_ms=0.0).validate(bin_ms=4.0, sequence_duration_ms=40.0)
    # copies=1 is always fine: no shift is applied
    _block(copies=1, stride_ms=0.0).validate(bin_ms=4.0, sequence_duration_ms=40.0)


def test_disabled_block_ignores_timing_dependent_checks():
    # 10 ms is not a whole number of 4 ms bins and 9 copies exceed the concat
    # cap, but the block is off, so validation must stay out of the way.
    InputDelayLineConfig(enabled=False, copies=9, stride_ms=10.0, mode="concat").validate(
        bin_ms=4.0, sequence_duration_ms=40.0
    )


def test_stride_bins_conversion():
    assert InputDelayLineConfig(stride_ms=10.0).stride_bins(2.0) == 5
    assert InputDelayLineConfig(stride_ms=8.0).stride_bins(4.0) == 2
    assert InputDelayLineConfig(stride_ms=0.0).stride_bins(2.0) == 0
    with pytest.raises(ValueError):
        InputDelayLineConfig(stride_ms=10.0).stride_bins(4.0)


def test_model_config_validation_uses_the_timing_block():
    bad = _tiny_cfg(input_delay_line=_block(copies=2, stride_ms=10.0, mode="concat"))
    with pytest.raises(ValueError):
        bad.validate()


# ---------------------------------------------------------------------- #
# config round-trip and overrides
# ---------------------------------------------------------------------- #
def test_default_block_is_off_and_roundtrips():
    default = V3Config()
    dl = default.input_delay_line
    assert dl.enabled is False
    assert dl.copies == 4
    assert float(dl.stride_ms) == 10.0
    assert dl.mode == "concat"
    assert default.effective_input_channels == default.n_inputs
    assert parameter_groups(default)["w_in"] == default.n_inputs * default.state_dim
    default.validate()  # nothing about the inert block may break a default run

    data = default.to_dict()
    assert data["input_delay_line"] == {
        "enabled": False,
        "copies": 4,
        "stride_ms": 10.0,
        "mode": "concat",
    }
    back = V3Config.from_dict(data, strict=True)
    assert back.input_delay_line == dl


def test_enabled_block_roundtrips_through_from_dict_to_dict():
    block = _block(copies=2, stride_ms=12.0, mode="sum")
    cfg = _tiny_cfg(input_delay_line=block)
    data = cfg.to_dict()
    assert data["input_delay_line"]["copies"] == 2
    back = V3Config.from_dict(data, strict=True)
    assert back.input_delay_line == block
    assert back.effective_input_channels == cfg.n_inputs


def test_override_grammar_reaches_the_block():
    cfg = apply_override_items(
        _tiny_cfg(),
        [
            "input_delay_line.enabled=true",
            "input_delay_line.copies=2",
            "input_delay_line.stride_ms=8",
            "input_delay_line.mode=sum",
        ],
    )
    dl = cfg.input_delay_line
    assert dl.enabled is True
    assert dl.copies == 2
    assert float(dl.stride_ms) == 8.0
    assert dl.mode == "sum"
    cfg.validate()
    # a nested mapping is merged into the existing block, keeping untouched fields
    merged = apply_override_mapping(cfg, {"input_delay_line": {"copies": 3, "mode": "concat"}})
    assert merged.input_delay_line.copies == 3
    assert merged.input_delay_line.mode == "concat"
    assert float(merged.input_delay_line.stride_ms) == 8.0
    assert merged.input_delay_line.enabled is True
