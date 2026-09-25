"""Tests for the data layer: synthetic generation, splits and batch iteration."""

from __future__ import annotations

import numpy as np

from src.data import iterate_batches, make_synthetic_shd, make_validation_split, subset_by_class


def test_synthetic_dataset_shape_and_metadata(synthetic_rec):
    rec = synthetic_rec
    assert len(rec) == 120
    assert rec.n_channels == 20
    assert rec.labels_array.shape == (120,)
    assert rec.speakers is not None
    assert np.all(rec.offsets[:-1] <= rec.offsets[1:])  # monotone offsets


def test_subset_preserves_order_and_labels(synthetic_rec):
    idx = np.array([5, 3, 9, 1])
    sub = synthetic_rec.subset(idx, name="sub")
    assert len(sub) == 4
    assert sub.name == "sub"
    assert np.array_equal(sub.labels_array, synthetic_rec.labels_array[idx])


def test_validation_split_is_disjoint_and_complete(synthetic_rec):
    train, val, info = make_validation_split(synthetic_rec, val_fraction=0.25, seed=0)
    assert len(train) + len(val) == len(synthetic_rec)
    assert len(val) > 0
    assert "n_speakers_total" in info


def test_iterate_batches_shapes_and_coverage(synthetic_rec):
    idx = np.arange(len(synthetic_rec))
    seen = 0
    for batch in iterate_batches(synthetic_rec, idx, batch_size=16, n_bins=30, bin_ms=2.0):
        assert batch["x"].shape[0] == batch["y"].shape[0]
        assert batch["x"].shape[1] == 30
        assert batch["x"].shape[2] == 20
        assert batch["x"].dtype.is_floating_point
        seen += batch["x"].shape[0]
    assert seen == len(synthetic_rec)


def test_subset_by_class_balances(synthetic_rec):
    sub = subset_by_class(synthetic_rec, max_per_class=5, seed=0)
    _, counts = np.unique(sub.labels_array, return_counts=True)
    assert counts.max() <= 5


def test_synthetic_is_deterministic():
    a = make_synthetic_shd(n_samples=20, n_classes=3, n_channels=8, n_bins=10, seed=7)
    b = make_synthetic_shd(n_samples=20, n_classes=3, n_channels=8, n_bins=10, seed=7)
    assert np.array_equal(a.times_ms, b.times_ms)
    assert np.array_equal(a.units, b.units)


def test_parse_shd_canonical_ragged_layout(tmp_path):
    """Regression: the official SHD HDF5 stores spikes as a *group* of ragged
    ``times``/``units`` object arrays plus speakers under ``extra/speaker``. The
    parser must auto-detect this canonical layout (not the flat/per-sample-group
    ones) and return per-sample offsets, labels and speakers."""
    import h5py

    from src.data import parse_shd_h5

    rng = np.random.default_rng(0)
    n = 5
    counts = [10, 7, 4, 9, 3]
    # deliberately UNSORTED per sample so we can check the parser sorts them
    times = [rng.random(c) for c in counts]
    units = [rng.integers(0, 700, size=c).astype(np.int64) for c in counts]
    labels = np.array([0, 1, 2, 3, 4], dtype=np.uint16)
    speakers = np.array([0, 1, 2, 3, 6], dtype=np.uint16)

    path = tmp_path / "shd_small.h5"
    with h5py.File(path, "w") as f:
        g = f.create_group("spikes")
        dt = h5py.special_dtype(vlen=np.float64)
        du = h5py.special_dtype(vlen=np.int64)
        g.create_dataset("times", (n,), dtype=dt)
        g.create_dataset("units", (n,), dtype=du)
        for i in range(n):
            g["times"][i] = times[i]
            g["units"][i] = units[i]
        f.create_dataset("labels", data=labels)
        f.create_group("extra").create_dataset("speaker", data=speakers)

    parsed = parse_shd_h5(path, layout="auto", verbose=False)

    assert parsed["layout_used"] == "shd_ragged"
    assert parsed["labels"].tolist() == labels.astype(np.int64).tolist()
    assert np.array_equal(parsed["speakers"], speakers.astype(np.int64))
    assert parsed["offsets"].tolist() == [0, 10, 17, 21, 30, 33]
    assert parsed["units"].dtype == np.int16
    # times are converted to ms and sorted *within* each sample
    for i in range(n):
        lo, hi = int(parsed["offsets"][i]), int(parsed["offsets"][i + 1])
        seg = parsed["times_ms"][lo:hi]
        assert np.all(np.diff(seg) >= 0)
    assert parsed["times_ms"].max() <= 1000.0
