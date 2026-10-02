"""Shuffle semantics, fixed FIT/VAL split, and recurrent-state reset.

CPU-only, synthetic; no dataset and no training run.
"""

from __future__ import annotations

import numpy as np
import torch

from v3.config import TimingConfig, V3Config
from v3.data import BatchIterator, Split, make_split
from v3.model import VectorNeuronPopulation


class _StubStore:
    """Minimal stand-in for SHDEventStore (splits + batching only)."""

    def __init__(self, labels: np.ndarray, speakers: np.ndarray):
        self.labels = labels
        self.speakers = speakers

    def batch(self, indices):
        return np.zeros((len(indices), 3, 2), dtype=np.float32)

    def labels_of(self, indices):
        return self.labels[np.asarray(indices, dtype=np.int64)]


def _toy_store(n_per_class: int = 40, n_classes: int = 4) -> _StubStore:
    labels = np.repeat(np.arange(n_classes), n_per_class)
    speakers = np.zeros_like(labels)
    return _StubStore(labels, speakers)


# ---------------------------------------------------------------------- #
# FIT / VAL split
# ---------------------------------------------------------------------- #
def test_fit_val_split_is_fixed_across_epochs():
    store = _toy_store()
    cfg = V3Config(val_fraction=0.10, split_seed=0)
    fit1, val1 = make_split(store, cfg)
    fit2, val2 = make_split(store, cfg)
    assert np.array_equal(fit1.indices, fit2.indices)   # same examples every epoch
    assert np.array_equal(val1.indices, val2.indices)
    assert len(set(fit1.indices) & set(val1.indices)) == 0
    assert fit1.indices.size + val1.indices.size == store.labels.size


def test_fit_val_split_is_stratified_and_seed_controlled():
    store = _toy_store(n_per_class=100, n_classes=4)
    fit, val = make_split(store, V3Config(val_fraction=0.10, split_seed=0))
    for c in range(4):
        n_val_c = int((val.labels == c).sum())
        assert n_val_c == 10  # 10% of each class
        assert int((fit.labels == c).sum()) == 90
    # a different split_seed gives a different partition
    fit_b, _ = make_split(store, V3Config(val_fraction=0.10, split_seed=123))
    assert not np.array_equal(fit.indices, fit_b.indices)


# ---------------------------------------------------------------------- #
# batching / shuffling
# ---------------------------------------------------------------------- #
class _IndexStore:
    """Store whose ``labels_of`` returns the indices, so batch order is observable."""

    def __init__(self, n: int):
        self.labels = np.zeros(n, dtype=np.int64)
        self.speakers = np.zeros(n, dtype=np.int64)

    def batch(self, indices):
        return np.zeros((len(indices), 3, 2), dtype=np.float32)

    def labels_of(self, indices):
        return np.asarray(indices, dtype=np.int64)


def _iteration_order(split, shuffle, seed=0, epoch=0, batch_size=7):
    store = _IndexStore(int(split.indices.size))
    it = BatchIterator(store, split, batch_size=batch_size, shuffle=shuffle,
                       seed=seed, epoch=epoch)
    order = []
    for _, y in it:
        order.extend(y.tolist())
    return np.asarray(order)


def _index_split(n: int) -> Split:
    idx = np.arange(n)
    return Split("fit", idx, np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64))


def test_shuffle_true_covers_each_sample_once_and_changes_order():
    split = _index_split(42)
    e0 = _iteration_order(split, shuffle=True, seed=1, epoch=0)
    e0_again = _iteration_order(split, shuffle=True, seed=1, epoch=0)
    e1 = _iteration_order(split, shuffle=True, seed=1, epoch=1)
    assert sorted(e0.tolist()) == sorted(split.indices.tolist())  # same set, once each
    assert e0.size == split.indices.size
    assert np.array_equal(e0, e0_again)                            # reproducible
    assert not np.array_equal(e0, e1)                              # epoch changes order


def test_shuffle_false_is_exactly_the_split_order():
    split = _index_split(42)
    order = _iteration_order(split, shuffle=False)
    assert np.array_equal(order, split.indices)


def test_iterator_yields_every_sample_once_when_shuffled():
    split = _index_split(42)
    order = _iteration_order(split, shuffle=True, seed=3, epoch=2, batch_size=7)
    assert sorted(order.tolist()) == sorted(split.indices.tolist())


# ---------------------------------------------------------------------- #
# recurrent state reset
# ---------------------------------------------------------------------- #
def _tiny_model(seed: int = 0) -> VectorNeuronPopulation:
    cfg = V3Config(
        seed=seed, n_neurons=4, state_dim=6, mix_rank=3, n_inputs=5, n_classes=3,
        timing=TimingConfig(sequence_duration_ms=12.0, time_bin_ms=3.0),  # T = 4
        device="cpu", dtype="float32", amp=False,
    )
    return VectorNeuronPopulation(cfg)


def test_state_is_reset_between_forwards():
    torch.manual_seed(0)
    model = _tiny_model()
    model.eval()
    x1 = (torch.rand(2, 4, 5) < 0.3).float()
    x2 = (torch.rand(2, 4, 5) < 0.3).float()
    with torch.no_grad():
        a = model(x1)[1]
        b = model(x2)[1]
        a_again = model(x1)[1]
        # running x2 after x1 must not change x2's result (no carry-over),
        c = model(x2)[1]
    assert torch.equal(a, a_again)  # fresh zero state each forward
    assert torch.equal(b, c)
    assert model.init_state(2, torch.device("cpu")).abs().sum() == 0
