"""Tests for the neuron-representation layer (the object under study)."""

from __future__ import annotations

import numpy as np

from src.neurons import (
    FeatureBlock,
    extract_structural_representations,
    intrinsic_features_from_model,
)
from src.representations import (
    RepresentationSpace,
    build_space_from_representations,
    random_baseline_space,
    shuffle_control_space,
)
from src.utils import Standardizer


def test_structural_representation_covers_every_neuron(tiny_model, tiny_snn_config):
    reps = extract_structural_representations(tiny_model)
    assert len(reps) == tiny_snn_config.n_hidden
    assert reps.meta["uses_labels"] is False
    assert reps.meta["uses_data"] is False
    # every neuron exposes the four structural blocks
    for rep in reps:
        assert set(rep.available_blocks()) == set(FeatureBlock.structural())


def test_to_matrix_is_deterministic_and_finite(tiny_model):
    reps = extract_structural_representations(tiny_model)
    X1, names1 = reps.to_matrix(blocks=FeatureBlock.structural())
    X2, names2 = reps.to_matrix(blocks=FeatureBlock.structural())
    assert names1 == names2
    assert np.array_equal(X1, X2)
    assert np.isfinite(X1).all()


def test_intrinsic_features_are_per_neuron(tiny_model, tiny_snn_config):
    feats = intrinsic_features_from_model(tiny_model)
    for key, values in feats.items():
        assert np.asarray(values).shape == (tiny_snn_config.n_hidden,), key


def test_representation_space_geometry():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((12, 5))
    names = [f"a.f{i}" for i in range(5)]
    space = RepresentationSpace(X_raw=X, feature_names=names, weighting="uniform")
    D = space.distances()
    assert D.shape == (12, 12)
    assert np.allclose(np.diag(D), 0.0)
    assert np.allclose(D, D.T)
    condensed = space.condensed()
    assert condensed.size == 12 * 11 // 2


def test_equal_weighting_downsizes_large_blocks():
    rng = np.random.default_rng(1)
    # two blocks with very different numbers of features
    names = ["big.f0", "big.f1", "big.f2", "small.g0"]
    X = rng.standard_normal((10, 4))
    equal = RepresentationSpace(X_raw=X, feature_names=names, weighting="equal")
    counts = equal.block_feature_counts()
    assert counts["big"] == 3 and counts["small"] == 1
    # equal weighting scales each feature by 1/sqrt(block size)
    assert np.isclose(equal.block_scales[0], 1.0 / np.sqrt(3))


def test_standardizer_handles_constant_columns():
    X = np.column_stack([np.ones(6), np.arange(6.0)])
    st = Standardizer().fit(X)
    assert st.n_informative == 1
    Z = st.transform(X)
    assert np.isfinite(Z).all()
    assert np.allclose(Z[:, 0], 0.0)


def test_random_and_shuffled_controls_preserve_dimensionality():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((15, 6))
    names = [f"a.f{i}" for i in range(6)]
    space = RepresentationSpace(X_raw=X, feature_names=names)
    rand = random_baseline_space(15, 6, seed=0)
    assert rand.n_neurons == 15 and rand.X.shape[1] == 6
    shuffled = shuffle_control_space(space, seed=0)
    # same set of rows, different order
    assert shuffled.n_neurons == space.n_neurons
    a = np.sort(space.X_raw.sum(axis=1))
    b = np.sort(shuffled.X_raw.sum(axis=1))
    assert np.allclose(a, b)
