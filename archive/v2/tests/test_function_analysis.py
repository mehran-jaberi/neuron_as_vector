"""Tests for the functional fingerprint, primary analysis, controls and prediction.

These tests exercise the *scientific separation* (label-free representation vs.
independent labelled fingerprint) and the statistical machinery (null
distribution, p-value floor, rate-matched control, cross-validated prediction),
all on tiny synthetic data with no downloads.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.evaluation import collect_activity
from src.functional_fingerprint import (
    FINGERPRINT_PRESETS,
    FingerprintConfig,
    FingerprintSpace,
    class_conditioned_fingerprint,
    resolve_feature_sets,
    split_half_reliability,
)
from src.function_analysis import abs_rate_difference_condensed, run_function_analysis
from src.geometry_analysis import (
    bootstrap_mantel_ci,
    geometry_function_analysis,
    mantel_test,
    primary_metric_row,
    rate_matched_mantel,
)
from src.neurons import (
    FeatureBlock,
    activity_features_from_psth,
    build_activity_representations,
    extract_structural_representations,
)
from src.prediction import cross_validated_knn, cross_validated_ridge, regression_metrics
from src.representations import build_space_from_representations


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _class_stats(n_classes=4, n_hidden=6, n_bins=30, seed=0):
    rng = np.random.default_rng(seed)
    class_psth = rng.random((n_classes, n_hidden, n_bins)) * 5.0
    class_counts = class_psth.sum(axis=2)
    class_n = np.full(n_classes, 25.0)
    first_sum = rng.random((n_classes, n_hidden)) * 20.0
    first_count = (rng.random((n_classes, n_hidden)) > 0.3) * rng.integers(1, 5, (n_classes, n_hidden))
    return class_psth, class_counts, class_n, first_sum, first_count.astype(float)


def _fingerprint(name, *, n_classes=4, n_hidden=6, seed=0, **kwargs):
    stats = _class_stats(n_classes, n_hidden, seed=seed)
    X, names = class_conditioned_fingerprint(
        *stats, n_bins=stats[0].shape[2], bin_ms=2.0,
        feature_sets=FINGERPRINT_PRESETS[name], **kwargs,
    )
    return FingerprintSpace(X_raw=X, feature_names=names), X


# --------------------------------------------------------------------------
# A/B/C: fingerprint families
# --------------------------------------------------------------------------
def test_class_tuning_fingerprint_is_C_dimensional():
    space, X = _fingerprint("tuning", n_classes=4)
    assert X.shape == (6, 4)
    assert space.feature_names == [f"fp_rate.class{c}" for c in range(4)]


def test_rate_normalized_fingerprint_removes_overall_magnitude():
    # Two neurons with the *same relative* profile but very different magnitudes
    # must be identical after rate normalization.
    class_psth = np.zeros((3, 2, 10))
    rel = np.array([1.0, 2.0, 3.0])[:, None]  # relative rates
    class_psth[:, 0, :] = rel * 10.0
    class_psth[:, 1, :] = rel * 1.0
    counts = class_psth.sum(axis=2)
    norm = np.full(3, 5.0)
    first_sum = np.zeros((3, 2))
    first_count = np.zeros((3, 2))
    X, names = class_conditioned_fingerprint(
        class_psth, counts, norm, first_sum, first_count,
        n_bins=10, bin_ms=2.0, feature_sets=["class_rate_norm"],
    )
    assert names[0].startswith("fp_rate_norm")
    # rows are proportional to the same relative profile -> identical
    assert np.allclose(X[0], X[1], atol=1e-9)
    assert np.isclose(X[0].mean(), 1.0)  # normalized to mean 1 over classes


def test_temporal_fingerprint_contains_psth_center_dispersion_latency():
    space, X = _fingerprint("temporal", n_classes=4, n_psth_bins=5)
    joined = " ".join(space.feature_names)
    assert "fp_psth." in joined
    assert "fp_tcenter." in joined
    assert "fp_tdispersion." in joined
    assert "fp_latency." in joined
    # 4 classes * (5 psth bins + 1 center + 1 dispersion + 1 latency) = 32
    assert X.shape == (6, 32)


def test_resolve_feature_sets_expands_presets_and_rejects_unknown():
    assert resolve_feature_sets("tuning") == ["class_rate"]
    assert "class_rate_norm" in resolve_feature_sets("tuning_rate_normalized")
    with pytest.raises(ValueError):
        resolve_feature_sets(["not_a_feature_set"])


def test_min_spikes_for_latency_censors_low_count_classes():
    stats = _class_stats(n_classes=2, n_hidden=1, n_bins=10)
    class_psth, counts, norm, first_sum, first_count = stats
    first_sum = np.zeros_like(first_sum)
    first_count = np.array([[1.0], [1.0]])  # exactly at threshold=1 -> observed
    lat_1, _ = class_conditioned_fingerprint(
        class_psth, counts, norm, first_sum, first_count,
        n_bins=10, bin_ms=2.0, feature_sets=["class_latency"], min_spikes_for_latency=1.0,
    )
    lat_2, _ = class_conditioned_fingerprint(
        class_psth, counts, norm, first_sum, first_count,
        n_bins=10, bin_ms=2.0, feature_sets=["class_latency"], min_spikes_for_latency=2.0,
    )
    assert lat_1[0, 0] == 0.0  # trusted -> observed latency
    assert lat_2[0, 0] == pytest.approx(20.0)  # censored at duration = 10 * 2 ms


def test_fingerprint_config_resolves_presets_and_defaults_to_probe():
    cfg = FingerprintConfig.from_mapping({"feature_sets": "tuning", "eval_split": "probe"})
    assert cfg.feature_sets == ["class_rate"]
    assert cfg.eval_split == "probe"
    assert cfg.n_psth_bins == 10


# --------------------------------------------------------------------------
# Primary statistic: null distribution, p-value floor, CI
# --------------------------------------------------------------------------
def test_mantel_returns_null_distribution_and_p_value_floor():
    rng = np.random.default_rng(0)
    n = 30
    X = rng.standard_normal((n, 5))
    Y = X + 0.3 * rng.standard_normal((n, 5))
    from scipy.spatial.distance import pdist

    dx, dy = pdist(X), pdist(Y)
    res = mantel_test(dx, dy, n_perm=120, seed=0, return_null=True, bootstrap=80)
    assert res.null is not None and res.null.size == 120
    assert res.p_value_floor == pytest.approx(1.0 / 121)
    assert res.statistic > 0.5
    assert res.bootstrap_ci_low <= res.statistic <= res.bootstrap_ci_high
    d = res.to_dict()
    assert d["null_distribution_included"] is True
    assert "q0.95" in d["null_quantiles"]
    assert "p_value_interpretation" in d


def test_mantel_flags_resolution_floor_saturation():
    rng = np.random.default_rng(1)
    n = 25
    X = rng.standard_normal((n, 4))
    Y = X.copy()  # identical geometry -> every permutation is worse
    from scipy.spatial.distance import pdist

    dx, dy = pdist(X), pdist(Y)
    res = mantel_test(dx, dy, n_perm=50, seed=0)
    assert res.statistic == pytest.approx(1.0)
    assert res.p_value == pytest.approx(1.0 / 51)
    assert res.at_resolution_floor is True


def test_bootstrap_ci_is_symmetric_around_a_perfect_correlation():
    rng = np.random.default_rng(2)
    n = 20
    X = rng.standard_normal((n, 3))
    from scipy.spatial.distance import pdist

    dx = pdist(X)
    low, high, samples = bootstrap_mantel_ci(dx, dx, n_boot=60, seed=0)
    assert samples.size == 60
    assert low == pytest.approx(1.0, abs=1e-6) and high == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------
def test_rate_matched_mantel_reports_stratified_method():
    rng = np.random.default_rng(3)
    n = 40
    X = rng.standard_normal((n, 5))
    Y = X + 0.4 * rng.standard_normal((n, 5))
    from scipy.spatial.distance import pdist

    dx, dy = pdist(X), pdist(Y)
    rate = rng.random(n) * 100.0
    rate_abs = np.abs(rate[:, None] - rate[None, :])[np.triu_indices(n, 1)]
    res = rate_matched_mantel(dx, dy, rate_abs, n_strata=4, n_perm=100, seed=0)
    assert res.method == "rate_matched_stratified"
    assert np.isfinite(res.statistic)
    assert res.p_value >= res.p_value_floor


def test_partial_mantel_is_labelled_secondary_exploratory():
    rng = np.random.default_rng(4)
    n = 30
    X = rng.standard_normal((n, 4))
    Y = X + 0.2 * rng.standard_normal((n, 4))
    Z = rng.standard_normal(n)
    from scipy.spatial.distance import pdist

    dz = np.abs(Z[:, None] - Z[None, :])[np.triu_indices(n, 1)]
    out = geometry_function_analysis(
        X, Y, n_perm=80, k_values=[3], seed=0,
        nuisance_condensed=dz, rate_absdiff_condensed=dz,
    )
    partial = out["partial_mantel_controlling_for_nuisance"]
    assert partial["status"] == "secondary_exploratory"
    assert "NOT proof" in partial["warning"]
    assert "rate_matched_mantel" in out
    row = primary_metric_row("x", out)
    assert row["partial_mantel_status"].startswith("secondary_exploratory")
    assert row["rate_matched_mantel_r"] is not None


# --------------------------------------------------------------------------
# Cross-validated prediction
# --------------------------------------------------------------------------
def test_ridge_cv_recovers_a_planted_linear_map():
    rng = np.random.default_rng(5)
    n, d, m = 80, 10, 4
    X = rng.standard_normal((n, d))
    W = rng.standard_normal((d, m))
    Y = X @ W + 0.05 * rng.standard_normal((n, m))
    res = cross_validated_ridge(X, Y, n_splits=5, seed=0)
    metrics = res["metrics"]
    assert metrics["pearson_r_mean"] > 0.8
    assert metrics["r2_mean"] > 0.6
    assert metrics["nrmse_mean"] < 0.5
    assert metrics["n_targets"] == m


def test_ridge_cv_does_not_predict_noise_better_than_chance():
    rng = np.random.default_rng(6)
    n, d, m = 60, 8, 3
    X = rng.standard_normal((n, d))
    Y = rng.standard_normal((n, m))
    res = cross_validated_ridge(X, Y, n_splits=5, seed=0)
    # out-of-sample R² on independent noise cannot be strongly positive
    assert res["metrics"]["r2_mean"] < 0.5


def test_knn_cv_prediction_runs():
    rng = np.random.default_rng(7)
    n, d = 40, 6
    X = rng.standard_normal((n, d))
    Y = X[:, :2] + 0.1 * rng.standard_normal((n, 2))
    res = cross_validated_knn(X, Y, n_splits=4, k=3, seed=0)
    assert res["metrics"]["model"] == "knn"
    assert np.isfinite(res["metrics"]["pearson_r_mean"])


def test_regression_metrics_perfect_prediction():
    rng = np.random.default_rng(8)
    Y = rng.standard_normal((30, 3))
    m = regression_metrics(Y, Y)
    assert m["r2_overall"] == pytest.approx(1.0)
    assert m["nrmse_mean"] == pytest.approx(0.0)
    assert m["predicted_vs_true_fingerprint_distance_spearman"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# End-to-end orchestration on tiny synthetic data
# --------------------------------------------------------------------------
def _tiny_bundle(tiny_model, synthetic_rec, tiny_snn_config, seed=0):
    idx = np.arange(len(synthetic_rec))
    # label-free representation (structural + activity)
    structural = extract_structural_representations(tiny_model)
    acc = collect_activity(
        tiny_model, synthetic_rec, idx, device=torch.device("cpu"),
        n_classes=tiny_snn_config.n_output, with_labels=False, collect_voltage=False,
    )
    feats, flags = activity_features_from_psth(
        acc.psth, acc.counts, bin_ms=acc.bin_ms, n_samples=acc.n_samples
    )
    from src.neurons import add_first_spike_features

    feats = add_first_spike_features(feats, acc.first_spike_sum, acc.first_spike_count,
                                     duration_ms=acc.duration_ms)
    activity = build_activity_representations(feats, flags)
    rep_space = build_space_from_representations(
        structural, FeatureBlock.structural(), weighting="equal"
    )
    # labelled fingerprints
    labelled = collect_activity(
        tiny_model, synthetic_rec, idx, device=torch.device("cpu"),
        n_classes=tiny_snn_config.n_output, with_labels=True, collect_voltage=False,
    )
    fingerprints = {}
    for name in ("tuning", "tuning_rate_normalized"):
        X, names = class_conditioned_fingerprint(
            labelled.class_psth, labelled.class_counts, labelled.class_n,
            labelled.class_first_spike_sum, labelled.class_first_spike_count,
            n_bins=labelled.n_bins, bin_ms=labelled.bin_ms,
            feature_sets=FINGERPRINT_PRESETS[name],
        )
        fingerprints[name] = FingerprintSpace(X_raw=X, feature_names=names)
    return rep_space, activity, fingerprints, labelled


def test_run_function_analysis_end_to_end(tiny_model, synthetic_rec, tiny_snn_config):
    rep_space, activity, fingerprints, _ = _tiny_bundle(
        tiny_model, synthetic_rec, tiny_snn_config
    )
    result = run_function_analysis(
        rep_space, fingerprints,
        primary_fingerprint="tuning",
        rate_normalized_fingerprint="tuning_rate_normalized",
        secondary_fingerprints=[],
        activity_reps=activity,
        n_perm=60, k_values=[3, 5], seed=0, random_repeats=2, n_strata=3, bootstrap=40,
        prediction={"enabled": False},
    )
    summary = result["summary"]
    # separation of concerns is recorded
    assert summary["meta"]["pairs_not_independent"] is True
    assert "label-free" in summary["meta"]["leakage_boundary"]
    # primary block
    p = summary["primary"]["primary_mantel_spearman"]
    assert {"p_value", "p_value_floor", "null_quantiles", "at_resolution_floor"} <= set(p)
    # all required controls are present
    for key in ("rate_only", "rate_normalized_fingerprint", "random_representation", "shuffled_neurons"):
        assert key in summary["controls"], key
    # rate_matched control is part of the primary analysis
    assert "rate_matched_mantel" in summary["primary"]
    # kNN table covers the requested k values
    ks = [row["k"] for row in summary["primary_headline"]["knn"]]
    assert ks == [3, 5]
    # arrays: null distribution + distance vectors
    assert "null_primary" in result["arrays"]
    assert result["arrays"]["null_primary"].size == 60


def test_abs_rate_difference_condensed_is_nonnegative(tiny_model, synthetic_rec, tiny_snn_config):
    _, activity, _, _ = _tiny_bundle(tiny_model, synthetic_rec, tiny_snn_config)
    d = abs_rate_difference_condensed(activity)
    n = len(activity)
    assert d.size == n * (n - 1) // 2
    assert np.all(d >= -1e-12)


def test_reliability_of_fingerprint_split_halves(tiny_model, synthetic_rec, tiny_snn_config):
    _, _, _, labelled = _tiny_bundle(tiny_model, synthetic_rec, tiny_snn_config)
    from src.evaluation import split_half_indices

    a, b = split_half_indices(synthetic_rec.labels_array, seed=0)
    spaces = []
    for idx in (a, b):
        res = collect_activity(
            tiny_model, synthetic_rec, idx, device=torch.device("cpu"),
            n_classes=tiny_snn_config.n_output, with_labels=True, collect_voltage=False,
        )
        X, names = class_conditioned_fingerprint(
            res.class_psth, res.class_counts, res.class_n,
            res.class_first_spike_sum, res.class_first_spike_count,
            n_bins=res.n_bins, bin_ms=res.bin_ms, feature_sets=["class_rate"],
        )
        spaces.append(FingerprintSpace(X_raw=X, feature_names=names))
    rel = split_half_reliability(spaces[0], spaces[1])
    assert -1.0 <= rel["matrix_reliability_spearman"] <= 1.0
    assert "attenuation_factor_sqrt_ceiling" in rel
