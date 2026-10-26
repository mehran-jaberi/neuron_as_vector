"""Pre-binned input cache: bit-identity with the on-the-fly path.

CPU-only, synthetic HDF5; no dataset and no training run.

The contract in ``V3/v3/binned_cache.py`` is that a cached batch is
**bit-identical** to the batch the current on-the-fly ``np.bincount`` path
produces for the same indices in the same order, for every split, with and
without duplicates inside a bin, and for both the binary (0/1) and the event
count encoding.  These tests assert exactly that with ``torch.equal``.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
import torch

from v3.binned_cache import ENV_ROOT, BinnedCache, cache_key, default_root
from v3.config import TimingConfig, V3Config
from v3.data import BatchIterator, SHDEventStore, make_split, official_test_split


# ---------------------------------------------------------------------- #
def _write_shd(path, n_samples: int, cells: int, seed: int = 0, max_events: int = 12):
    """Canonical SHD layout with random events; duplicated (bin, unit) pairs included."""
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as fh:
        td = fh.create_dataset("spikes/times", (n_samples,), dtype=h5py.vlen_dtype(np.float64))
        ud = fh.create_dataset("spikes/units", (n_samples,), dtype=h5py.vlen_dtype(np.int64))
        for i in range(n_samples):
            k = int(rng.integers(1, max_events))
            # duplicate the first two events to force repeated (bin, unit) pairs
            times = rng.uniform(0.0, 1.0, size=k)
            units = rng.integers(0, cells, size=k)
            if k >= 2:
                times[1], units[1] = times[0], units[0]
            if k >= 3:
                times[2], units[2] = times[0], units[0]
            td[i] = times
            ud[i] = units
        fh.create_dataset("labels", data=np.asarray(rng.integers(0, 4, size=n_samples), dtype=np.uint16))
        fh.create_dataset("extra/speaker", data=np.zeros(n_samples, dtype=np.uint16))


def _cfg(cells: int = 6, bin_ms: float = 10.0, **kw) -> V3Config:
    # 10 ms bins -> 100 steps: tiny, still exercises the (B, T, C) layout
    return V3Config(
        n_inputs=cells,
        timing=TimingConfig(sequence_duration_ms=1000.0, time_bin_ms=bin_ms),
        cache_events=False,          # exercise the cold on-the-fly path every call
        binned_cache=False,
        **kw,
    )


@pytest.fixture()
def cache_root(tmp_path, monkeypatch):
    root = tmp_path / "binned"
    monkeypatch.setenv(ENV_ROOT, str(root))
    return root


@pytest.fixture()
def shd_files(tmp_path):
    train = tmp_path / "train.h5"
    test = tmp_path / "test.h5"
    _write_shd(train, n_samples=37, cells=6, seed=1)
    _write_shd(test, n_samples=11, cells=6, seed=2)
    return train, test


def _assert_iterators_identical(cfg_off, cfg_on, train, test, batch_size=8):
    torch.manual_seed(0)
    for name, path, shuffle in (("fit", train, True), ("val", train, False), ("test", test, False)):
        off = SHDEventStore(path, cfg_off)
        on = SHDEventStore(path, cfg_on)
        assert on.binned is not None and off.binned is None
        if name == "test":
            split = official_test_split(off)
        else:
            fit, val = make_split(off, cfg_off)
            split = fit if name == "fit" else val
        it_off = BatchIterator(off, split, batch_size, shuffle=shuffle, seed=0, epoch=3)
        it_on = BatchIterator(on, split, batch_size, shuffle=shuffle, seed=0, epoch=3)
        n_compared = 0
        for (x_off, y_off), (x_on, y_on) in zip(it_off, it_on, strict=True):
            assert x_on.shape == x_off.shape == (len(y_off), cfg_off.n_bins, cfg_off.n_inputs)
            # the exact assertion the brief requires: same values, same order
            assert torch.equal(torch.from_numpy(x_on), torch.from_numpy(x_off))
            assert np.array_equal(y_on, y_off)
            n_compared += len(y_off)
        assert n_compared == split.indices.size
        off.close()
        on.close()


# ---------------------------------------------------------------------- #
def test_cached_batches_are_bit_identical_for_every_split(shd_files, cache_root):
    train, test = shd_files
    cfg_off = _cfg()
    # build the cache through the authoritative on-the-fly store (a small batch
    # size forces several build batches)
    builder = SHDEventStore(train, cfg_off)
    cache = BinnedCache.build(builder, verbose=False)
    assert cache.dtype == np.uint8
    assert cache.nbytes == 37 * 100 * 6
    builder.close()
    test_builder = SHDEventStore(test, cfg_off)
    BinnedCache.build(test_builder, verbose=False)
    test_builder.close()

    cfg_on = cfg_off.with_overrides(binned_cache=True)
    _assert_iterators_identical(cfg_off, cfg_on, train, test)


def test_random_single_batches_are_bit_identical(shd_files, cache_root):
    train, _ = shd_files
    cfg_off = _cfg()
    builder = SHDEventStore(train, cfg_off)
    cache = BinnedCache.build(builder, verbose=False)
    builder.close()

    rng = np.random.default_rng(0)
    cfg_on = cfg_off.with_overrides(binned_cache=True)
    off = SHDEventStore(train, cfg_off)
    on = SHDEventStore(train, cfg_on)
    for _ in range(8):
        b = int(rng.integers(1, 12))
        idx = rng.choice(37, size=b, replace=False)
        x_off = off.batch(idx)
        x_on = on.batch(idx)
        assert x_on.dtype == x_off.dtype == np.float32
        assert torch.equal(torch.from_numpy(x_on), torch.from_numpy(x_off))
    # out-of-order / repeated indices must match too
    idx = np.array([5, 5, 0, 36, 2], dtype=np.int64)
    assert torch.equal(
        torch.from_numpy(on.batch(idx)), torch.from_numpy(off.batch(idx))
    )
    assert torch.equal(torch.from_numpy(on.batch(idx)).sum(dim=1), torch.from_numpy(off.batch(idx)).sum(dim=1))
    off.close()
    on.close()
    assert cache.meta["schema"] == "v3_binned_cache/v1"


def test_counts_encoding_round_trips(tmp_path, cache_root):
    """binary_input=False stores counts in uint16; duplicates become 2/3, not 1."""
    p = tmp_path / "counts.h5"
    _write_shd(p, n_samples=5, cells=4, seed=7)
    cfg_off = _cfg(cells=4, bin_ms=10.0, binary_input=False)
    builder = SHDEventStore(p, cfg_off)
    x_ref = builder.batch(np.array([0, 1, 2]))
    assert x_ref.max() >= 2.0          # the duplicated events in a bin were counted
    cache = BinnedCache.build(builder, verbose=False)
    assert cache.dtype == np.uint16
    builder.close()
    cfg_on = cfg_off.with_overrides(binned_cache=True)
    store = SHDEventStore(p, cfg_on)
    x_on = store.batch(np.array([0, 1, 2]))
    assert torch.equal(torch.from_numpy(x_on), torch.from_numpy(x_ref))
    store.close()


# ---------------------------------------------------------------------- #
def test_default_path_is_unchanged(shd_files):
    train, _ = shd_files
    store = SHDEventStore(train, _cfg())
    assert store.binned is None                       # opt-in only
    x = store.batch(np.arange(4))
    assert x.dtype == np.float32 and set(np.unique(x)) <= {0.0, 1.0}
    store.close()


def test_key_does_not_mix_layouts(shd_files):
    train, _ = shd_files
    a = cache_key(train, _cfg(bin_ms=10.0))
    b = cache_key(train, _cfg(bin_ms=5.0))
    c = cache_key(train, _cfg(bin_ms=10.0, binary_input=False))
    d = cache_key(train, _cfg(bin_ms=10.0, val_fraction=0.2))   # irrelevant to binning
    assert len({a, b, c}) == 3
    assert a == d


def test_incomplete_cache_is_not_opened(shd_files, cache_root):
    train, _ = shd_files
    cfg = _cfg()
    key = cache_key(train, cfg)
    d = cache_root / key
    d.mkdir(parents=True)
    (d / "dense.uint8").write_bytes(b"\x00" * 16)         # truncated, no meta.json
    assert BinnedCache.open(train, cfg) is None
    # a complete build then ignores the garbage
    builder = SHDEventStore(train, cfg)
    cache = BinnedCache.build(builder, verbose=False)
    assert cache.nbytes == 37 * 100 * 6
    builder.close()
    assert BinnedCache.open(train, cfg).nbytes == cache.nbytes


def test_meta_records_provenance(shd_files, cache_root):
    train, _ = shd_files
    cfg = _cfg()
    builder = SHDEventStore(train, cfg)
    BinnedCache.build(builder, verbose=False)
    builder.close()
    directory = default_root() / cache_key(train, cfg)
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    for k in ("n_samples", "n_bins", "n_inputs", "binary_input", "dtype", "file", "key"):
        assert k in meta
    assert meta["n_bins"] == 100 and meta["n_inputs"] == 6 and meta["dtype"] == "uint8"
    assert (directory / meta["dense_file"]).exists()
