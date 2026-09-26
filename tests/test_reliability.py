"""Tests for the fingerprint reliability / noise-ceiling audit.

These verify the claims the ceiling calculation depends on: independent halves,
class balance, the right (held-out) split, a consistent distance metric, the
correct reliability formula, and no accidental leakage.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation import split_half_indices
from src.functional_fingerprint import (
    FingerprintConfig,
    FingerprintSpace,
    audit_split_halves,
    reliability_audit,
    split_half_reliability,
)


def _space(X: np.ndarray, *, metric: str = "euclidean", n_classes: int | None = None) -> FingerprintSpace:
    names = [f"fp_rate.class{c}" for c in range(X.shape[1])]
    meta = {"uses_labels": True}
    if n_classes is not None:
        meta["n_classes"] = n_classes
    return FingerprintSpace(X_raw=X, feature_names=names, config=FingerprintConfig(metric=metric), meta=meta)


# --------------------------------------------------------------------------
# Split construction audit
# --------------------------------------------------------------------------
def test_split_half_indices_are_disjoint_and_complete():
    labels = np.repeat(np.arange(5), 7)  # 35 samples, 5 classes
    a, b = split_half_indices(labels, seed=0)
    audit = audit_split_halves(labels, a, b, n_classes=5, split_name="probe")
    assert audit["halves_disjoint"] is True
    assert audit["n_overlap"] == 0
    assert audit["halves_cover_all_samples"] is True
    assert audit["n_uncovered"] == 0
    # stratified: every class present in both halves, proportions close
    assert audit["all_classes_present_in_both_halves"] is True
    assert audit["class_stratified"] is True


def test_audit_detects_overlap_and_non_coverage():
    labels = np.repeat(np.arange(4), 5)  # 20 samples
    bad_a = np.arange(0, 12)
    bad_b = np.arange(8, 20)  # overlaps 8..11 and covers all
    audit = audit_split_halves(labels, bad_a, bad_b, n_classes=4, split_name="probe")
    assert audit["halves_disjoint"] is False
    assert audit["n_overlap"] == 4
    short_b = np.arange(12, 16)  # leaves 16..19 uncovered
    audit2 = audit_split_halves(labels, bad_a, short_b, n_classes=4)
    assert audit2["halves_cover_all_samples"] is False
    assert audit2["n_uncovered"] == 4


# --------------------------------------------------------------------------
# Reliability calculation
# --------------------------------------------------------------------------
def test_identical_halves_have_unit_reliability():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 6))
    rel = split_half_reliability(_space(X), _space(X))
    assert rel["matrix_reliability_spearman"] == pytest.approx(1.0)
    assert rel["matrix_reliability_full_spearman_brown"] == pytest.approx(1.0)
    assert rel["attenuation_factor_sqrt_ceiling"] == pytest.approx(1.0)
    assert rel["attenuation_factor_half_length"] == pytest.approx(1.0)


def test_spearman_brown_formula_is_applied():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((25, 5))
    Y = X + 0.5 * rng.standard_normal((25, 5))
    rel = split_half_reliability(_space(X), _space(Y))
    r = rel["matrix_reliability_spearman"]
    if np.isfinite(r):
        assert rel["matrix_reliability_full_spearman_brown"] == pytest.approx(2 * r / (1 + r))
        # the full-length ceiling is never below the half-length one when r > 0
        if r > 0:
            assert rel["attenuation_factor_sqrt_ceiling"] >= rel["attenuation_factor_half_length"] - 1e-12


def test_reliability_requires_a_consistent_metric():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((15, 4))
    with pytest.raises(ValueError):
        split_half_reliability(_space(X, metric="euclidean"), _space(X, metric="correlation"))
    # an explicit metric overrides and is flagged as a warning
    rel = split_half_reliability(
        _space(X, metric="euclidean"), _space(X, metric="correlation"), metric="euclidean"
    )
    assert rel["metrics_match"] is False
    assert rel["metric"] == "euclidean"
    assert rel["warnings"]


def test_reliability_requires_same_number_of_neurons():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError):
        split_half_reliability(_space(rng.standard_normal((10, 3))), _space(rng.standard_normal((8, 3))))


def test_common_standardizer_is_used_and_reported():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((18, 5)) * np.array([1.0, 10.0, 100.0, 0.1, 5.0])
    Y = X + 0.2 * rng.standard_normal((18, 5)) * np.array([1.0, 10.0, 100.0, 0.1, 5.0])
    from src.utils import Standardizer

    st = Standardizer().fit(np.vstack([X, Y]))
    rel = split_half_reliability(_space(X), _space(Y), common_standardizer=st)
    assert rel["common_standardizer_used"] is True
    assert np.isfinite(rel["matrix_reliability_spearman"])
    rel_plain = split_half_reliability(_space(X), _space(Y))
    assert rel_plain["common_standardizer_used"] is False


# --------------------------------------------------------------------------
# Full audit + leakage
# --------------------------------------------------------------------------
def test_reliability_audit_passes_for_a_proper_held_out_split():
    rng = np.random.default_rng(5)
    labels = np.repeat(np.arange(4), 10)  # 40 samples
    a, b = split_half_indices(labels, seed=0)
    base = rng.standard_normal((12, 4))
    half_a = _space(base + 0.1 * rng.standard_normal((12, 4)), n_classes=4)
    half_b = _space(base + 0.1 * rng.standard_normal((12, 4)), n_classes=4)
    audit = reliability_audit(
        labels=labels, idx_a=a, idx_b=b, half_a=half_a, half_b=half_b, full=half_a,
        split_name="probe", n_classes=4,
    )
    assert audit["passed"] is True
    assert audit["checks"]["halves_disjoint"] is True
    assert audit["checks"]["same_distance_metric"] is True
    assert audit["checks"]["split_is_held_out"] is True
    assert audit["checks"]["fingerprint_uses_labels_by_design"] is True


def test_reliability_audit_flags_training_split_and_overlap():
    rng = np.random.default_rng(6)
    labels = np.repeat(np.arange(4), 10)
    half_a = _space(rng.standard_normal((12, 4)), n_classes=4)
    half_b = _space(rng.standard_normal((12, 4)), n_classes=4)
    # using the training split and overlapping halves must fail the audit
    audit = reliability_audit(
        labels=labels, idx_a=np.arange(20), idx_b=np.arange(10, 30),
        half_a=half_a, half_b=half_b, full=half_a, split_name="train", n_classes=4,
    )
    assert audit["passed"] is False
    assert audit["checks"]["split_is_not_train"] is False
    assert audit["checks"]["halves_disjoint"] is False