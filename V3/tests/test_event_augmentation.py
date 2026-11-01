"""Raw-event augmentation: semantics, gating, caches and the override grammar.

CPU-only, synthetic events; no real dataset and no training run.  The tests pin
the contract of the brief: augmentation acts on raw ``(time, channel)`` events
before binning, never on VAL/TEST, never caches a perturbed realisation, drops
negative times, and is deterministic for a fixed seed/epoch.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from v3.config import (
    ChannelDropoutConfig,
    EventAugmentationConfig,
    EventDropoutConfig,
    EventInsertionConfig,
    GlobalTimeShiftConfig,
    TimeJitterConfig,
    TimingConfig,
    V3Config,
    apply_override_items,
    apply_override_mapping,
)
from v3.data import BatchIterator, SHDEventStore, Split, make_split
from v3.event_augmentation import augment
from v3.experiment import assert_eval_batches_unchanged


# ---------------------------------------------------------------------- #
# fixtures / helpers
# ---------------------------------------------------------------------- #
def _all_on() -> EventAugmentationConfig:
    return EventAugmentationConfig(
        enabled=True,
        time_jitter=TimeJitterConfig(enabled=True, std_ms=2.0),
        event_dropout=EventDropoutConfig(enabled=True, probability=0.2),
        event_insertion=EventInsertionConfig(enabled=True, rate_hz=100.0),
        global_time_shift=GlobalTimeShiftConfig(enabled=True, max_ms=50.0),
        channel_dropout=ChannelDropoutConfig(enabled=True, probability=0.1),
    )


def _cfg(cells: int = 8, augmentation: EventAugmentationConfig | None = None, **kw) -> V3Config:
    # 10 ms bins over a 1000 ms window -> 100 steps; small and fast
    kw.setdefault("timing", TimingConfig(sequence_duration_ms=1000.0, time_bin_ms=10.0))
    kw.setdefault("device", "cpu")
    kw.setdefault("dtype", "float32")
    kw.setdefault("amp", False)
    kw.setdefault("cache_events", True)
    return V3Config(
        n_inputs=cells,
        n_classes=2,
        augmentation=augmentation or EventAugmentationConfig(),
        **kw,
    )


def _write_shd(path, samples, n_classes: int = 2):
    with h5py.File(path, "w") as fh:
        td = fh.create_dataset("spikes/times", (len(samples),), dtype=h5py.vlen_dtype(np.float64))
        ud = fh.create_dataset("spikes/units", (len(samples),), dtype=h5py.vlen_dtype(np.int64))
        for i, (ts, us) in enumerate(samples):
            td[i] = np.asarray(ts, dtype=np.float64)
            ud[i] = np.asarray(us, dtype=np.int64)
        fh.create_dataset("labels", data=np.arange(len(samples), dtype=np.uint16) % n_classes)
        fh.create_dataset("extra/speaker", data=np.zeros(len(samples), dtype=np.uint16))
    return path


def _toy_store(tmp_path, augmentation=None, cells: int = 8, n_samples: int = 3,
               events_per_sample: int = 40, **kw) -> tuple[SHDEventStore, V3Config]:
    rng = np.random.default_rng(7)
    samples = []
    for _ in range(n_samples):
        ts = np.sort(rng.uniform(0.0, 0.999, size=events_per_sample))
        us = rng.integers(0, cells, size=events_per_sample)
        samples.append((ts, us))
    path = _write_shd(tmp_path / "shd.h5", samples)
    cfg = _cfg(cells=cells, augmentation=augmentation, **kw)
    return SHDEventStore(path, cfg), cfg


def _split(store: SHDEventStore, name: str, indices) -> Split:
    idx = np.asarray(indices, dtype=np.int64)
    return Split(name, idx, store.labels[idx], store.speakers[idx])


# ---------------------------------------------------------------------- #
# augment(): per-augmenter semantics
# ---------------------------------------------------------------------- #
def test_disabled_augmentation_returns_an_unchanged_copy():
    ts = np.array([0.1, 0.2, 0.99])
    us = np.array([1, 2, 3], dtype=np.int64)
    out_t, out_u = augment(
        ts, us, EventAugmentationConfig(), np.random.default_rng(0),
        window_s=1.0, n_inputs=4,
    )
    assert np.array_equal(out_t, ts) and np.array_equal(out_u, us)
    assert out_t is not ts and out_u is not us
    # master switch alone (no sub-augmenter) is also a no-op
    out_t, out_u = augment(
        ts, us, EventAugmentationConfig(enabled=True), np.random.default_rng(0),
        window_s=1.0, n_inputs=4,
    )
    assert np.array_equal(out_t, ts) and np.array_equal(out_u, us)


def test_inputs_are_never_mutated():
    rng = np.random.default_rng(3)
    ts = rng.uniform(0.0, 0.9, size=2000)
    us = rng.integers(0, 20, size=2000).astype(np.int64)
    ts_ref, us_ref = ts.copy(), us.copy()
    augment(ts, us, _all_on(), np.random.default_rng(0), window_s=1.0, n_inputs=20)
    assert np.array_equal(ts, ts_ref) and np.array_equal(us, us_ref)


def test_time_jitter_statistics():
    n = 20000
    rng = np.random.default_rng(0)
    ts = rng.uniform(0.2, 0.8, size=n)  # far enough from the window edges to stay
    us = rng.integers(0, 10, size=n).astype(np.int64)
    cfg = EventAugmentationConfig(
        enabled=True, time_jitter=TimeJitterConfig(enabled=True, std_ms=2.0)
    )
    out_t, out_u = augment(ts, us, cfg, np.random.default_rng(1), window_s=1.0, n_inputs=10)
    assert out_t.size == n  # nothing filtered
    delta = out_t - ts
    assert abs(float(delta.mean())) < 1e-4                       # symmetric
    assert float(delta.std()) == pytest.approx(0.002, rel=0.05)  # 2 ms


def test_event_dropout_fraction_and_subsequence():
    n = 20000
    rng = np.random.default_rng(0)
    ts = rng.uniform(0.0, 0.99, size=n)
    us = rng.integers(0, 10, size=n).astype(np.int64)
    cfg = EventAugmentationConfig(
        enabled=True, event_dropout=EventDropoutConfig(enabled=True, probability=0.3)
    )
    out_t, out_u = augment(ts, us, cfg, np.random.default_rng(2), window_s=1.0, n_inputs=10)
    assert 1.0 - out_t.size / n == pytest.approx(0.3, abs=0.01)
    assert np.all(np.isin(out_t, ts))          # no newly created timestamps
    assert np.all(np.isin(out_u, us))          # no newly created channels
    # probability 1 empties the sample; 0 keeps it
    out_t, _ = augment(
        ts, us,
        EventAugmentationConfig(enabled=True,
                                event_dropout=EventDropoutConfig(enabled=True, probability=1.0)),
        np.random.default_rng(2), window_s=1.0, n_inputs=10,
    )
    assert out_t.size == 0


def test_insertion_rate_ranges_and_zero_rate():
    cfg = EventAugmentationConfig(
        enabled=True, event_insertion=EventInsertionConfig(enabled=True, rate_hz=500.0)
    )
    empty_t = np.zeros(0, dtype=np.float64)
    empty_u = np.zeros(0, dtype=np.int64)
    out_t, out_u = augment(empty_t, empty_u, cfg, np.random.default_rng(4),
                           window_s=1.0, n_inputs=10)
    assert 400 <= out_t.size <= 600                       # Poisson(500)
    assert 0.0 <= out_t.min() and out_t.max() < 1.0
    assert out_u.min() >= 0 and out_u.max() < 10
    zero = EventAugmentationConfig(
        enabled=True, event_insertion=EventInsertionConfig(enabled=True, rate_hz=0.0)
    )
    out_t, _ = augment(empty_t, empty_u, zero, np.random.default_rng(4),
                       window_s=1.0, n_inputs=10)
    assert out_t.size == 0


def test_channel_dropout_never_drops_every_channel():
    cells = 6
    rng = np.random.default_rng(0)
    ts = np.tile(np.linspace(0.01, 0.9, 50), cells)
    us = np.repeat(np.arange(cells), 50).astype(np.int64)
    cfg = EventAugmentationConfig(
        enabled=True, channel_dropout=ChannelDropoutConfig(enabled=True, probability=0.5)
    )
    for seed in range(50):
        out_t, out_u = augment(ts, us, cfg, np.random.default_rng(seed),
                               window_s=1.0, n_inputs=cells)
        assert out_t.size > 0                       # some channel always survives
        assert out_u.min() >= 0 and out_u.max() < cells
    # probability 1 leaves exactly one channel, never none
    certain = EventAugmentationConfig(
        enabled=True, channel_dropout=ChannelDropoutConfig(enabled=True, probability=1.0)
    )
    out_t, out_u = augment(ts, us, certain, np.random.default_rng(0),
                           window_s=1.0, n_inputs=cells)
    assert out_t.size > 0
    assert np.unique(out_u).size == 1


def test_global_shift_is_one_shift_per_sample_and_bounded():
    ts = np.array([0.4, 0.5, 0.6])
    us = np.array([1, 2, 3], dtype=np.int64)
    cfg = EventAugmentationConfig(
        enabled=True, global_time_shift=GlobalTimeShiftConfig(enabled=True, max_ms=100.0)
    )
    out_t, _ = augment(ts, us, cfg, np.random.default_rng(5), window_s=2.0, n_inputs=4)
    delta = out_t - ts
    assert np.allclose(delta, delta[0])
    assert -0.1 <= float(delta[0]) <= 0.1
    # symmetric over many draws
    deltas = [
        float(augment(ts, us, cfg, np.random.default_rng(s), window_s=2.0, n_inputs=4)[0][0] - ts[0])
        for s in range(500)
    ]
    assert abs(float(np.mean(deltas))) < 0.01


def test_negative_and_out_of_window_times_are_filtered():
    ts = np.array([-0.02, 0.5, 0.999, 1.0, -0.001])
    us = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    # max_ms=0 makes the shift a no-op; the window filter must still run
    cfg = EventAugmentationConfig(
        enabled=True, global_time_shift=GlobalTimeShiftConfig(enabled=True, max_ms=0.0)
    )
    out_t, out_u = augment(ts, us, cfg, np.random.default_rng(6), window_s=1.0, n_inputs=5)
    assert np.array_equal(out_t, np.array([0.5, 0.999]))
    assert np.array_equal(out_u, np.array([1, 2]))
    assert np.all(out_t >= 0.0) and np.all(out_t < 1.0)


# ---------------------------------------------------------------------- #
# binning guard: negative times can never become negative codes
# ---------------------------------------------------------------------- #
def test_bin_events_drops_negative_times(tmp_path):
    store = _toy_binning_store(tmp_path)
    codes = store._bin_events(
        np.array([-0.15, -0.05, 0.05]),
        np.array([1, 1, 1], dtype=np.int64),
    )
    # 10 ms bins: t=0.05 is bin 5, t<0 must be dropped (old code: bin -1/-0.5)
    assert codes.tolist() == [5 * store.n_inputs + 1]
    assert np.all(codes >= 0)


def test_store_drops_negative_raw_times_even_without_augmentation(tmp_path):
    # t=-0.15 would bin to index -1 with the old code and corrupt the bincount
    samples = [(np.array([-0.15, -0.05, 0.2, 0.99, 1.0]),
                np.array([1, 2, 3, 4, 5], dtype=np.int64))]
    path = _write_shd(tmp_path / "neg.h5", samples)
    store = SHDEventStore(
        path,
        _cfg(cells=8, timing=TimingConfig(sequence_duration_ms=1000.0, time_bin_ms=100.0)),
    )
    x = store.batch(np.array([0]))
    assert x.shape == (1, 10, 8)
    assert x.sum() == 2                                  # only 0.2 and 0.99 survive
    assert x[0, 2, 3] == 1 and x[0, 9, 4] == 1
    assert x[0, 0].sum() == 0 and x[0].sum() >= 0


def _toy_binning_store(tmp_path) -> SHDEventStore:
    path = _write_shd(
        tmp_path / "bin.h5", [(np.array([0.05]), np.array([1], dtype=np.int64))]
    )
    return SHDEventStore(path, _cfg())


# ---------------------------------------------------------------------- #
# config surface
# ---------------------------------------------------------------------- #
def test_defaults_are_all_off_and_serialise():
    cfg = V3Config()
    assert cfg.augmentation.active() is False
    data = cfg.to_dict()
    aug = data["augmentation"]
    for block in ("time_jitter", "event_dropout", "event_insertion",
                  "global_time_shift", "channel_dropout"):
        assert aug[block]["enabled"] is False
    assert aug["apply_during_training"] is True
    assert aug["apply_during_validation"] is False
    assert aug["apply_during_test"] is False
    back = V3Config.from_dict(data, strict=True)
    assert back.augmentation == cfg.augmentation
    import yaml

    yaml.safe_dump(data)  # serialisable exactly as registry.register_run writes it


def test_phase_active_flags():
    cfg = EventAugmentationConfig(
        enabled=True, time_jitter=TimeJitterConfig(enabled=True)
    )
    assert cfg.phase_active("train") is True
    assert cfg.phase_active("val") is False
    assert cfg.phase_active("test") is False
    with pytest.raises(ValueError):
        cfg.phase_active("bogus")
    assert EventAugmentationConfig().active() is False


@pytest.mark.parametrize(
    "block",
    [
        TimeJitterConfig(enabled=True, std_ms=-1.0),
        EventDropoutConfig(enabled=True, probability=-0.1),
        EventDropoutConfig(enabled=True, probability=1.1),
        EventInsertionConfig(enabled=True, rate_hz=-1.0),
        GlobalTimeShiftConfig(enabled=True, max_ms=-1.0),
        ChannelDropoutConfig(enabled=True, probability=2.0),
    ],
)
def test_validate_rejects_out_of_range_values(block):
    cfg = EventAugmentationConfig(enabled=True)
    name = type(block).__name__
    field = {
        "TimeJitterConfig": "time_jitter",
        "EventDropoutConfig": "event_dropout",
        "EventInsertionConfig": "event_insertion",
        "GlobalTimeShiftConfig": "global_time_shift",
        "ChannelDropoutConfig": "channel_dropout",
    }[name]
    setattr(cfg, field, block)
    with pytest.raises(ValueError):
        cfg.validate()
    with pytest.raises(ValueError):
        V3Config(augmentation=cfg).validate()


# ---------------------------------------------------------------------- #
# override grammar: two levels of dotted nesting
# ---------------------------------------------------------------------- #
def test_override_items_support_two_levels():
    cfg = V3Config()
    out = apply_override_items(
        cfg,
        [
            "augmentation.enabled=true",
            "augmentation.time_jitter.enabled=true",
            "augmentation.time_jitter.std_ms=0.5",
            "augmentation.event_dropout.probability=0.25",
            "timing.time_bin_ms=4",
            "epochs=3",
        ],
    )
    assert out.augmentation.enabled is True
    assert out.augmentation.time_jitter.enabled is True
    assert out.augmentation.time_jitter.std_ms == pytest.approx(0.5)
    assert out.augmentation.event_dropout.probability == pytest.approx(0.25)
    assert out.timing.time_bin_ms == 4.0 and out.epochs == 3
    assert cfg.augmentation.enabled is False  # input untouched


def test_override_mapping_accepts_dotted_and_nested_forms():
    cfg = V3Config()
    out = apply_override_mapping(
        cfg,
        {
            "augmentation.enabled": True,
            "augmentation": {"channel_dropout": {"enabled": True, "probability": 0.25}},
            "state_regularization.mode": "noise",
        },
    )
    assert out.augmentation.enabled is True
    assert out.augmentation.channel_dropout.enabled is True
    assert out.augmentation.channel_dropout.probability == pytest.approx(0.25)
    assert out.augmentation.time_jitter.std_ms == pytest.approx(0.5)  # preserved
    assert out.state_regularization.mode == "noise"


@pytest.mark.parametrize(
    "item",
    [
        "augmentation.time_jitter.nope=1",
        "augmentation.nope.x=1",
        "augmentation.time_jitter.std_ms.deep=1",
        "augmentation=1",
        "not-an-assignment",
    ],
)
def test_override_rejects_unknown_or_malformed_keys(item):
    with pytest.raises(ValueError):
        apply_override_items(V3Config(), [item])


def test_override_mapping_rejects_unknown_nested_key():
    with pytest.raises(ValueError):
        apply_override_mapping(V3Config(), {"augmentation": {"time_jitter": {"nope": 1}}})


# ---------------------------------------------------------------------- #
# store gating, caches, determinism
# ---------------------------------------------------------------------- #
def test_augmentation_bypasses_ram_and_dense_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("V3_BINNED_CACHE", str(tmp_path / "cache"))  # avoid real cache
    store, _ = _toy_store(tmp_path, _all_on(), cache_events=True, binned_cache=True)
    assert store._cache is None
    assert store.binned is None
    stats = store.stats(sample=2)
    assert stats["cache_events_active"] is False
    assert stats["binned_cache_active"] is False
    assert stats["augmentation"]["active"] is True
    assert any("bypass" in w for w in stats["warnings"])


def test_augmentation_off_keeps_the_caches(tmp_path):
    store, _ = _toy_store(tmp_path)
    assert store._cache is not None
    assert store.binned is None  # binned_cache is off by default
    assert store.stats(sample=2)["warnings"] == []


def test_phase_gating_and_never_val_test(tmp_path):
    on_store, cfg = _toy_store(tmp_path, _all_on())
    off_store = SHDEventStore(on_store.path, _cfg())
    on_store.set_augmentation_indices(np.arange(on_store.n_samples))
    idx = np.array([0, 1])

    on_store.set_phase("val")
    assert np.array_equal(on_store.batch(idx), off_store.batch(idx))
    on_store.set_phase("train")
    assert not np.array_equal(on_store.batch(idx), off_store.batch(idx))
    # even if the config explicitly allows test-time augmentation, never TEST
    on_store.set_phase("test")
    assert np.array_equal(on_store.batch(idx), off_store.batch(idx))
    # hard off-switch wins over the training phase
    on_store.set_phase("train")
    on_store.force_augmentation_off(True)
    assert np.array_equal(on_store.batch(idx), off_store.batch(idx))
    with pytest.raises(ValueError):
        on_store.set_phase("bogus")


def test_index_mask_keeps_fit_samples_only(tmp_path):
    on_store, _ = _toy_store(tmp_path, _all_on())
    off_store = SHDEventStore(on_store.path, _cfg())
    on_store.set_phase("train")
    on_store.set_augmentation_indices([0])
    assert not np.array_equal(on_store.batch(np.array([0])), off_store.batch(np.array([0])))
    assert np.array_equal(on_store.batch(np.array([1])), off_store.batch(np.array([1])))


def test_determinism_for_fixed_seed_and_epoch(tmp_path):
    store, _ = _toy_store(tmp_path, _all_on())
    store.set_phase("train")
    store.set_augmentation_indices(np.arange(store.n_samples))
    idx = np.array([0, 1, 2])
    store.set_epoch(3)
    a = store.batch(idx)
    store.set_epoch(3)
    b = store.batch(idx)
    store.set_epoch(4)
    c = store.batch(idx)
    assert np.array_equal(a, b)          # same seed * 1000 + epoch -> identical
    assert not np.array_equal(a, c)      # fresh realisation next epoch


def test_batch_iterator_reseeds_per_epoch(tmp_path):
    store, _ = _toy_store(tmp_path, _all_on())
    store.set_phase("train")
    store.set_augmentation_indices(np.arange(store.n_samples))
    split = _split(store, "fit", [0, 1, 2])
    first = [x for x, _ in BatchIterator(store, split, 2, shuffle=False, seed=0, epoch=5)]
    again = [x for x, _ in BatchIterator(store, split, 2, shuffle=False, seed=0, epoch=5)]
    other = [x for x, _ in BatchIterator(store, split, 2, shuffle=False, seed=0, epoch=6)]
    for a, b in zip(first, again):
        assert np.array_equal(a, b)
    assert any(not np.array_equal(a, b) for a, b in zip(first, other))


def test_sample_events_returns_raw_seconds(tmp_path):
    ts = np.array([0.01, 0.02, 0.03])
    us = np.array([1, 2, 3], dtype=np.int64)
    path = _write_shd(tmp_path / "raw.h5", [(ts, us)])
    store = SHDEventStore(path, _cfg())
    got_ts, got_us = store.sample_events(0)
    assert got_ts.dtype == np.float64 and got_us.dtype == np.int64
    assert np.array_equal(got_ts, ts) and np.array_equal(got_us, us)


def test_off_path_is_bit_identical(tmp_path):
    store_a, _ = _toy_store(tmp_path)
    store_b = SHDEventStore(store_a.path, _cfg())
    store_a.set_augmentation_indices(np.arange(store_a.n_samples))
    store_a.set_phase("train")
    idx = np.arange(store_a.n_samples)
    assert np.array_equal(store_a.batch(idx), store_b.batch(idx))


# ---------------------------------------------------------------------- #
# experiment guard
# ---------------------------------------------------------------------- #
def test_eval_batches_unchanged_guard_passes_and_fails(tmp_path):
    store, cfg = _toy_store(tmp_path, _all_on())
    store.set_augmentation_indices([0])
    val = _split(store, "val", [1])
    assert_eval_batches_unchanged(store, val, cfg, phase="train")  # must not raise

    leaky = SHDEventStore(store.path, cfg)
    leaky.set_augmentation_indices([0, 1])  # deliberately includes the eval sample
    leaky.set_phase("train")
    with pytest.raises(AssertionError):
        assert_eval_batches_unchanged(leaky, val, cfg, phase="train")
