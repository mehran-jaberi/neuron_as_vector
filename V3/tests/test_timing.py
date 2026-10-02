"""Focused tests for the timing configuration and SHD event binning.

CPU-only, tiny synthetic HDF5 files; no real dataset and no training run.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from v3.config import TimingConfig, V3Config
from v3.data import SHDEventStore
from v3.model import VectorNeuronPopulation


# ---------------------------------------------------------------------- #
# timing derivation
# ---------------------------------------------------------------------- #
def test_default_timing_is_the_2ms_reference():
    t = TimingConfig()
    assert t.time_bin_ms == 2.0
    assert t.sequence_duration_ms == 1000.0
    assert t.num_time_steps == 500
    assert t.simulation_dt_ms == 2.0
    t.validate()


def test_4ms_timing_gives_250_steps():
    t = TimingConfig(sequence_duration_ms=1000.0, time_bin_ms=4.0)
    assert t.num_time_steps == 250
    assert t.simulation_dt_ms == 4.0
    t.validate()


def test_2ms_timing_gives_500_steps():
    t = TimingConfig(sequence_duration_ms=1000.0, time_bin_ms=2.0)
    assert t.num_time_steps == 500
    assert t.simulation_dt_ms == 2.0
    t.validate()


def test_config_exposes_derived_timing_and_alpha():
    cfg4 = V3Config(tau_ms=20.0, timing=TimingConfig(1000.0, 4.0))
    assert cfg4.n_bins == 250 and cfg4.time_bin_ms == 4.0
    assert cfg4.dt_ms == 4.0 and cfg4.simulation_dt_ms == 4.0
    assert cfg4.alpha == pytest.approx(4.0 / 20.0)
    cfg2 = V3Config(tau_ms=20.0, timing=TimingConfig(1000.0, 2.0))
    assert cfg2.n_bins == 500 and cfg2.dt_ms == 2.0
    assert cfg2.alpha == pytest.approx(2.0 / 20.0)


@pytest.mark.parametrize("duration,bin_ms", [(1000.0, 3.0), (1000.0, 7.0), (999.0, 4.0)])
def test_non_integer_step_count_is_rejected(duration, bin_ms):
    with pytest.raises(ValueError):
        TimingConfig(sequence_duration_ms=duration, time_bin_ms=bin_ms).validate()
    with pytest.raises(ValueError):
        V3Config(timing=TimingConfig(sequence_duration_ms=duration, time_bin_ms=bin_ms)).validate()


@pytest.mark.parametrize("duration,bin_ms", [(0.0, 4.0), (1000.0, 0.0), (-1.0, 4.0), (1000.0, -2.0)])
def test_non_positive_timing_is_rejected(duration, bin_ms):
    with pytest.raises(ValueError):
        TimingConfig(sequence_duration_ms=duration, time_bin_ms=bin_ms).validate()


def test_legacy_flat_keys_are_migrated_to_a_timing_block():
    cfg = V3Config.from_dict({"bin_ms": 2.0, "n_bins": 500})
    assert cfg.time_bin_ms == 2.0 and cfg.num_time_steps == 500
    assert cfg.sequence_duration_ms == 1000.0
    cfg4 = V3Config.from_dict({"bin_ms": 4.0, "n_bins": 250})
    assert cfg4.num_time_steps == 250


def test_yaml_roundtrip_of_the_timing_block():
    cfg = V3Config(timing=TimingConfig(sequence_duration_ms=1000.0, time_bin_ms=2.0))
    data = cfg.to_dict()
    assert isinstance(data["timing"], dict)
    back = V3Config.from_dict(data, strict=True)
    assert back.num_time_steps == 500 and back.time_bin_ms == 2.0


def test_default_yaml_loads():
    from v3.config import V3_ROOT

    cfg = V3Config.from_yaml(V3_ROOT / "configs" / "v3_default.yaml")
    assert cfg.num_time_steps == 500 and cfg.time_bin_ms == 2.0
    assert cfg.shuffle_train is True and cfg.shuffle_val is False


# ---------------------------------------------------------------------- #
# model dynamical timestep
# ---------------------------------------------------------------------- #
def _leak_after_one_step(time_bin_ms: float) -> float:
    """With all weights zeroed, z1 = (1 - alpha) * z0, so z1 measures alpha."""
    cfg = V3Config(
        n_neurons=2,
        state_dim=4,
        mix_rank=4,
        n_inputs=8,
        n_classes=3,
        tau_ms=20.0,
        timing=TimingConfig(sequence_duration_ms=1000.0, time_bin_ms=time_bin_ms),
        device="cpu",
        dtype="float32",
        amp=False,
    )
    model = VectorNeuronPopulation(cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        z0 = torch.ones(1, cfg.n_neurons, cfg.state_dim)
        a = torch.zeros(1, 1, cfg.state_dim)
        z1, _, _ = model._segment(z0, a)
    return float(z1.mean())


def test_model_uses_the_configured_simulation_timestep():
    # 4 ms -> alpha = 0.2 -> z1 = 0.8 ; 2 ms -> alpha = 0.1 -> z1 = 0.9
    assert _leak_after_one_step(4.0) == pytest.approx(0.8, abs=1e-6)
    assert _leak_after_one_step(2.0) == pytest.approx(0.9, abs=1e-6)


# ---------------------------------------------------------------------- #
# SHD event binning (synthetic HDF5 with the canonical layout)
# ---------------------------------------------------------------------- #
def _write_tiny_shd(path, times, units, labels, speakers):
    n = len(times)
    with h5py.File(path, "w") as fh:
        td = fh.create_dataset("spikes/times", (n,), dtype=h5py.vlen_dtype(np.float64))
        ud = fh.create_dataset("spikes/units", (n,), dtype=h5py.vlen_dtype(np.int64))
        for i in range(n):
            td[i] = np.asarray(times[i], dtype=np.float64)
            ud[i] = np.asarray(units[i], dtype=np.int64)
        fh.create_dataset("labels", data=np.asarray(labels, dtype=np.uint16))
        fh.create_dataset("extra/speaker", data=np.asarray(speakers, dtype=np.uint16))


def _cfg(bin_ms, cells=8):
    return V3Config(
        n_inputs=cells,
        timing=TimingConfig(sequence_duration_ms=1000.0, time_bin_ms=bin_ms),
        cache_events=False,
    )


def test_binning_4ms_places_events_in_the_right_bin(tmp_path):
    # t=0 -> bin 0; 0.001/0.002 -> bin 0; 0.0045 -> bin 1; 0.999 -> bin 249;
    # t=1.0 exactly and t=1.2 beyond the window are dropped.
    p = tmp_path / "shd4.h5"
    _write_tiny_shd(p, [[0.0, 0.001, 0.002, 0.0045, 0.999, 1.0, 1.2]],
                    [[1, 1, 2, 3, 4, 5, 6]], [0], [0])
    store = SHDEventStore(p, _cfg(4.0))
    codes = store.sample_codes(0)
    # bin*C + channel, C=8: 0*8+1=1, 0*8+2=2, 1*8+3=11, 249*8+4=1996
    assert sorted(codes.tolist()) == [1, 2, 11, 1996]
    x = store.batch(np.array([0]))
    assert x.shape == (1, 250, 8)
    assert x[0, 0, 1] == 1 and x[0, 0, 2] == 1 and x[0, 1, 3] == 1 and x[0, 249, 4] == 1
    assert x.sum() == 4  # the t=1.0 and t=1.2 events were dropped
    store.close()


def test_binning_2ms_gives_500_steps_and_correct_bins(tmp_path):
    # 0.001 -> bin 0; 0.002 and 0.003 -> bin 1; 0.004 -> bin 2; 0.999 -> bin 499;
    # t=1.0 exactly is dropped.  The bins are computed from raw timestamps, not
    # by splitting 4 ms bins.
    p = tmp_path / "shd2.h5"
    _write_tiny_shd(p, [[0.001, 0.002, 0.003, 0.004, 0.999, 1.0]],
                    [[1, 1, 2, 3, 4, 5]], [3], [7])
    store = SHDEventStore(p, _cfg(2.0))
    codes = store.sample_codes(0)
    # 0*8+1=1, 1*8+1=9, 1*8+2=10, 2*8+3=19, 499*8+4=3996
    assert sorted(codes.tolist()) == [1, 9, 10, 19, 3996]
    x = store.batch(np.array([0]))
    assert x.shape == (1, 500, 8)
    assert x[0, 0, 1] == 1 and x[0, 1, 1] == 1 and x[0, 1, 2] == 1
    assert x[0, 2, 3] == 1 and x[0, 499, 4] == 1
    assert x.sum() == 5
    store.close()


def test_window_policy_is_unchanged_between_2ms_and_4ms(tmp_path):
    """Same utterance window: both keep exactly the events with t < 1000 ms."""
    times = [0.0, 0.25, 0.5, 0.75, 0.999, 1.0, 1.001]
    units = [0, 1, 2, 3, 4, 5, 6]
    p4, p2 = tmp_path / "a.h5", tmp_path / "b.h5"
    _write_tiny_shd(p4, [times], [units], [0], [0])
    _write_tiny_shd(p2, [times], [units], [0], [0])
    s4, s2 = SHDEventStore(p4, _cfg(4.0)), SHDEventStore(p2, _cfg(2.0))
    assert s4.sample_codes(0).size == 5  # 0.0 .. 0.999 kept, 1.0/1.001 dropped
    assert s2.sample_codes(0).size == 5
    s4.close()
    s2.close()
