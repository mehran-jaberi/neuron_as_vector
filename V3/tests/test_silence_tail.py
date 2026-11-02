"""Silence tail (M13): all-zero input appended after the stimulus.

CPU-only, synthetic HDF5; no real dataset, no training run.  The tail must be
absent by default (bit-identical baseline), present in *every* phase, exactly
zero, and stored out of the pre-binned cache.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from v3.binned_cache import ENV_ROOT, BinnedCache
from v3.config import TimingConfig, V3Config
from v3.data import SHDEventStore, make_split
from v3.model import VectorNeuronPopulation

CELLS = 8


# ---------------------------------------------------------------------- #
# synthetic SHD data and configs
# ---------------------------------------------------------------------- #
def _write_shd(path, times, units, labels=None):
    """Canonical SHD layout from explicit per-sample event lists."""
    n = len(times)
    labels = np.zeros(n, dtype=np.uint16) if labels is None else np.asarray(labels, np.uint16)
    with h5py.File(path, "w") as fh:
        td = fh.create_dataset("spikes/times", (n,), dtype=h5py.vlen_dtype(np.float64))
        ud = fh.create_dataset("spikes/units", (n,), dtype=h5py.vlen_dtype(np.int64))
        for i, (ts, us) in enumerate(zip(times, units, strict=True)):
            td[i] = np.asarray(ts, dtype=np.float64)
            ud[i] = np.asarray(us, dtype=np.int64)
        fh.create_dataset("labels", data=labels)
        fh.create_dataset("extra/speaker", data=np.zeros(n, dtype=np.uint16))


def _cfg(tail_ms: float = 0.0, bin_ms: float = 10.0, **kw) -> V3Config:
    # 10 ms bins -> 100 stimulus steps: tiny, but still exercises (B, T, C).
    kw.setdefault("cache_events", False)
    kw.setdefault("binned_cache", False)
    return V3Config(
        n_inputs=CELLS,
        n_classes=4,
        timing=TimingConfig(
            sequence_duration_ms=1000.0, time_bin_ms=bin_ms, silence_tail_ms=tail_ms
        ),
        val_fraction=0.5,
        device="cpu",
        dtype="float32",
        amp=False,
        **kw,
    )


@pytest.fixture()
def shd_file(tmp_path):
    path = tmp_path / "train.h5"
    _write_shd(
        path,
        times=[
            [0.10, 0.50, 0.999], [0.20],
            [0.75, 0.76], [0.01],
            [0.30, 0.40], [0.60],
            [0.90], [0.05, 0.95],
        ],
        units=[
            [2, 5, 7], [1],
            [3, 3], [0],
            [4, 4], [6],
            [7], [2, 0],
        ],
        labels=[0, 0, 1, 1, 2, 2, 3, 3],
    )
    return path


@pytest.fixture()
def cache_root(tmp_path, monkeypatch):
    root = tmp_path / "binned"
    monkeypatch.setenv(ENV_ROOT, str(root))
    return root


# ---------------------------------------------------------------------- #
# default 0.0 keeps the baseline bit-identical
# ---------------------------------------------------------------------- #
def test_default_zero_tail_keeps_tensors_and_model_output_bit_identical(shd_file):
    cfg = _cfg()
    assert cfg.tail_steps == 0 and cfg.total_steps == cfg.n_bins == 100
    store = SHDEventStore(shd_file, cfg)
    idx = np.arange(4)
    x = store.batch(idx)
    assert x.shape == (4, cfg.n_bins, CELLS)

    # unchanged reference: the dense 0/1 tensor implied by the binned codes
    ref = np.zeros_like(x)
    for k, i in enumerate(idx):
        codes = store.sample_codes(int(i))
        ref[k].reshape(-1)[codes] = 1.0
    assert np.array_equal(x, ref)

    # a tail config leaves the stimulus window and the model math untouched
    cfg_tail = _cfg(tail_ms=100.0)
    store_tail = SHDEventStore(shd_file, cfg_tail)
    x_tail = store_tail.batch(idx)
    assert x_tail.shape == (4, 110, CELLS)
    assert np.array_equal(x_tail[:, : cfg.n_bins], x)
    assert not x_tail[:, cfg.n_bins :].any()

    m0 = VectorNeuronPopulation(cfg)
    m_tail = VectorNeuronPopulation(cfg_tail)
    assert m0.n_parameters() == m_tail.n_parameters()
    with torch.no_grad():
        logits0, spikes0 = m0(torch.from_numpy(x))
        logits1, spikes1 = m_tail(torch.from_numpy(x))
    assert torch.equal(logits0, logits1)
    assert torch.equal(spikes0, spikes1)
    store.close()
    store_tail.close()


# ---------------------------------------------------------------------- #
# arithmetic and the 10 % cap
# ---------------------------------------------------------------------- #
def test_tail_and_total_step_arithmetic():
    t = TimingConfig(1000.0, 2.0, 100.0)
    assert (t.tail_steps, t.total_steps, t.num_time_steps) == (50, 550, 500)
    t.validate()
    t4 = TimingConfig(1000.0, 4.0, 100.0)
    assert (t4.tail_steps, t4.total_steps, t4.num_time_steps) == (25, 275, 250)
    t4.validate()
    default = TimingConfig()
    assert (default.tail_steps, default.total_steps) == (0, 500)
    default.validate()


@pytest.mark.parametrize("tail_ms", [0.0, 2.0, 50.0, 100.0])
def test_tail_at_or_below_the_cap_is_accepted(tail_ms):
    t = TimingConfig(1000.0, 2.0, tail_ms)
    t.validate()
    assert t.tail_steps == round(tail_ms / 2.0)
    V3Config(timing=t).validate()


@pytest.mark.parametrize("tail_ms", [-2.0, -0.001, 1.0, 3.0, 102.0, 120.0, 200.0])
def test_invalid_tail_is_rejected(tail_ms):
    # negative, sub-bin, and over the +10 % cap (all must raise)
    with pytest.raises(ValueError):
        TimingConfig(1000.0, 2.0, tail_ms).validate()
    with pytest.raises(ValueError):
        V3Config(timing=TimingConfig(1000.0, 2.0, tail_ms)).validate()


@pytest.mark.parametrize("tail_ms", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_tail_is_rejected(tail_ms):
    with pytest.raises(ValueError):
        TimingConfig(1000.0, 2.0, tail_ms).validate()


# ---------------------------------------------------------------------- #
# the appended bins are exactly zero and hold no events
# ---------------------------------------------------------------------- #
def test_tail_bins_are_exactly_zero_and_hold_no_events(shd_file):
    cfg = _cfg(tail_ms=100.0)
    store = SHDEventStore(shd_file, cfg)
    assert store.tail_steps == 10
    x = store.batch(np.arange(4))
    assert x.shape == (4, cfg.total_steps, CELLS) == (4, 110, 8)
    tail = x[:, cfg.n_bins :]
    assert tail.dtype == np.float32
    assert tail.shape == (4, 10, CELLS)
    assert not tail.any()

    # every kept event is inside the stimulus window (none can land in the tail)
    for i in range(4):
        codes = store.sample_codes(i)
        assert codes.size > 0
        assert int((codes // CELLS).max()) < cfg.n_bins
    # and the stimulus region really carries the events (the tail is not masking them)
    assert x[:, : cfg.n_bins].sum() > 0
    assert x[0, 10, 2] == 1.0 and x[0, 50, 5] == 1.0
    store.close()


# ---------------------------------------------------------------------- #
# every phase and the pre-binned cache
# ---------------------------------------------------------------------- #
def test_batch_has_total_steps_in_every_phase_and_with_the_cache(shd_file, cache_root):
    cfg_off = _cfg(tail_ms=100.0)
    cfg_on = _cfg(tail_ms=100.0, binned_cache=True)
    store_off = SHDEventStore(shd_file, cfg_off)
    # Pre-build the cache from a finished store, as the existing cache tests do:
    # building inside __init__ touches augmentation gates that are set after it.
    builder = SHDEventStore(shd_file, cfg_off)
    BinnedCache.build(builder, verbose=False)
    builder.close()
    store_on = SHDEventStore(shd_file, cfg_on)
    assert store_on.binned is not None
    # the cache itself stores the stimulus window only
    assert store_on.binned.shape[1] == cfg_on.n_bins

    fit, val = make_split(store_off, cfg_off)
    assert len(fit) > 0 and len(val) > 0
    for phase, idx in (
        ("train", fit.indices),
        ("val", val.indices),
        ("test", np.arange(4, dtype=np.int64)),
    ):
        store_off.set_phase(phase)
        store_on.set_phase(phase)
        x_off = store_off.batch(idx)
        x_on = store_on.batch(idx)
        assert x_off.shape == x_on.shape == (len(idx), cfg_off.total_steps, CELLS)
        assert np.array_equal(x_off, x_on)
        assert not x_on[:, cfg_off.n_bins :].any()
    store_off.close()
    store_on.close()


# ---------------------------------------------------------------------- #
# the model runs on the extended window and counts the late spikes
# ---------------------------------------------------------------------- #
def _late_activity_model(cfg: V3Config) -> VectorNeuronPopulation:
    """2 neurons that keep spiking after the last stimulus bin."""
    model = VectorNeuronPopulation(cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.w_in[:, 0] = 1.0
        model.gate[:, 0] = 1.0
        model.w_out[:, 0] = 1.0
        model.theta.fill_(-0.5)
    model.eval()
    return model


def test_model_forward_runs_on_the_extended_window_and_counts_late_spikes():
    cfg = _cfg(tail_ms=100.0, bin_ms=2.0, n_neurons=2, state_dim=4, mix_rank=4)
    assert cfg.n_bins == 500 and cfg.total_steps == 550
    model = _late_activity_model(cfg)
    x = torch.zeros(1, cfg.total_steps, CELLS)
    x[0, cfg.n_bins - 20 :, 0] = 1.0   # the stimulus ends 20 bins before the tail

    with torch.no_grad():
        logits_t, spikes_t = model(x)
        logits_u, spikes_u = model(x[:, : cfg.n_bins])
    assert spikes_t.shape == (1, cfg.total_steps, cfg.n_neurons)
    assert spikes_u.shape == (1, cfg.n_bins, cfg.n_neurons)
    assert spikes_t[:, cfg.n_bins :].sum() > 0     # late activity falls in the tail
    assert spikes_t.sum() > spikes_u.sum()         # counts are summed over the tail
    assert logits_t.shape == logits_u.shape == (1, cfg.n_classes)
    with pytest.raises(ValueError):
        model(torch.zeros(1, cfg.total_steps + 1, CELLS))
