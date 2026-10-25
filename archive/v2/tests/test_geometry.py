"""Tests for the geometry-function analysis (Mantel, kNN, composite analysis)."""

from __future__ import annotations

import numpy as np

from src.geometry_analysis import (
    geometry_function_analysis,
    knn_analysis,
    mantel_test,
    partial_mantel_test,
)
from src.representations import RepresentationSpace


def _space_from(X: np.ndarray) -> RepresentationSpace:
    names = [f"a.f{i}" for i in range(X.shape[1])]
    return RepresentationSpace(X_raw=X, feature_names=names, weighting="uniform")


def test_mantel_detects_a_planted_matching_geometry():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 6))
    Y = X + 0.05 * rng.standard_normal(X.shape)  # nearly identical geometry
    dx = _space_from(X).condensed()
    dy = _space_from(Y).condensed()
    res = mantel_test(dx, dy, n_perm=200, seed=0)
    assert res.statistic > 0.8
    assert res.p_value < 0.05


def test_mantel_is_null_for_independent_geometries():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((40, 6))
    Y = rng.standard_normal((40, 6))
    dx = _space_from(X).condensed()
    dy = _space_from(Y).condensed()
    res = mantel_test(dx, dy, n_perm=300, seed=0)
    assert abs(res.statistic) < 0.25


def test_partial_mantel_controls_for_nuisance():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((30, 5))
    # Y shares geometry with X *and* with the nuisance Z
    Z = rng.standard_normal((30, 5))
    Y = 0.5 * X + 0.5 * Z
    dx = _space_from(X).condensed()
    dy = _space_from(Y).condensed()
    dz = _space_from(Z).condensed()
    res = partial_mantel_test(dx, dy, dz, n_perm=200, seed=0)
    assert np.isfinite(res.statistic)


def test_knn_analysis_runs_and_scores_k_values():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((25, 4))
    D = _space_from(X).distances()
    res = knn_analysis(X, D, k_values=[3, 5], n_perm=100, seed=0)
    assert res.get("k_values") == [3, 5]


def test_geometry_function_analysis_composite_keys():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((30, 5))
    Y = X + 0.1 * rng.standard_normal(X.shape)
    out = geometry_function_analysis(X, Y, n_perm=100, k_values=[3, 5], seed=0)
    assert "primary_mantel_spearman" in out
    assert out["primary_mantel_spearman"]["statistic"] > 0.5
    assert out["n_neurons"] == 30
    assert out["n_pairs"] == 30 * 29 // 2


def test_distance_matrix_round_trips_to_condensed():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((10, 3))
    space = _space_from(X)
    from src.geometry_analysis import condensed_to_matrix

    condensed = space.condensed()
    full = condensed_to_matrix(condensed, space.n_neurons)
    assert np.allclose(full, space.distances())
