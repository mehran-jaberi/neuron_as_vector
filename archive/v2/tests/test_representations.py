"""Tests for the neuron-representation layer (the object under study)."""

from __future__ import annotations

import numpy as np
import pytest

from src.neurons import (
    DYNAMICAL_FEATURE_NAMES,
    GENERIC_LEARNED_FEATURE_NAMES,
    TONOTOPIC_FEATURE_NAMES,
    FeatureBlock,
    add_first_spike_features,
    activity_features_from_psth,
    classify_feature,
    extract_structural_representations,
    input_connectivity_features,
    intrinsic_features_from_model,
    recurrent_incoming_features,
    recurrent_outgoing_features,
)
from src.representations import (
    EmptyFeatureSelectionError,
    RepresentationSpace,
    build_space_from_representations,
    random_baseline_space,
    shuffle_control_space,
)
from src.utils import Standardizer


# --------------------------------------------------------------------------
# Structural representation: blocks and coverage
# --------------------------------------------------------------------------
def test_structural_representation_covers_required_blocks(trained_like_model, tiny_snn_config):
    reps = extract_structural_representations(trained_like_model)
    assert len(reps) == tiny_snn_config.n_hidden
    assert reps.meta["uses_labels"] is False
    assert reps.meta["uses_data"] is False
    # a trained-like model exposes all four structural blocks
    assert set(reps[0].available_blocks()) == set(FeatureBlock.structural())


def test_untrained_model_has_no_informative_intrinsic_features(tiny_model):
    # An untrained bias-mode model has an all-zero bias -> no per-neuron information.
    feats = intrinsic_features_from_model(tiny_model)
    assert feats == {}
    reps = extract_structural_representations(tiny_model)
    assert FeatureBlock.INTRINSIC.value not in reps[0].available_blocks()
    assert reps.meta["intrinsic_block_present"] is False


def test_to_matrix_is_deterministic_and_finite(trained_like_model):
    reps = extract_structural_representations(trained_like_model)
    X1, names1 = reps.to_matrix(blocks=FeatureBlock.structural())
    X2, names2 = reps.to_matrix(blocks=FeatureBlock.structural())
    assert names1 == names2
    assert np.array_equal(X1, X2)
    assert np.isfinite(X1).all()


def test_intrinsic_features_are_per_neuron(trained_like_model, tiny_snn_config):
    feats = intrinsic_features_from_model(trained_like_model)
    assert feats  # non-empty because the (simulated) bias varies
    for key, values in feats.items():
        assert np.asarray(values).shape == (tiny_snn_config.n_hidden,), key


# --------------------------------------------------------------------------
# Generic learned bias vs genuine dynamical parameters
# --------------------------------------------------------------------------
def test_learned_bias_is_classified_as_generic_not_dynamical(trained_like_model):
    feats = intrinsic_features_from_model(trained_like_model)
    assert set(feats) == {"learned_bias"}
    rep = extract_structural_representations(trained_like_model)[0]
    assert set(rep.generic_learned_features()) == {"intrinsic.learned_bias"}
    assert rep.dynamical_features() == {}
    assert classify_feature("intrinsic.learned_bias") == "generic_learned"
    assert "intrinsic.learned_bias" in GENERIC_LEARNED_FEATURE_NAMES
    assert "intrinsic.learned_bias" not in DYNAMICAL_FEATURE_NAMES


def test_bias_tau_model_emits_genuine_dynamical_parameter(dynamical_model):
    feats = intrinsic_features_from_model(dynamical_model)
    assert "tau_mem_ms" in feats
    rep = extract_structural_representations(dynamical_model)[0]
    assert "intrinsic.tau_mem_ms" in rep.dynamical_features()
    assert classify_feature("intrinsic.tau_mem_ms") == "dynamical"


# --------------------------------------------------------------------------
# Tonotopic / channel-ordering features
# --------------------------------------------------------------------------
def test_tonotopic_features_excluded_by_default_and_opt_in_only():
    rng = np.random.default_rng(0)
    w = rng.standard_normal(700)
    default = input_connectivity_features(w)
    assert "channel_com" not in default and "channel_spread" not in default
    opted = input_connectivity_features(w, include_tonotopic=True)
    assert "channel_com" in opted and "channel_spread" in opted
    assert TONOTOPIC_FEATURE_NAMES == {"input_conn.channel_com", "input_conn.channel_spread"}


def test_primary_structural_features_are_channel_permutation_invariant():
    # Permuting the input channels must leave every default feature unchanged.
    rng = np.random.default_rng(1)
    w = rng.standard_normal(64)
    perm = rng.permutation(w.size)
    base = input_connectivity_features(w)
    shuffled = input_connectivity_features(w[perm])
    for key in base:
        assert np.isclose(base[key], shuffled[key]), key


def test_tonotopic_features_are_channel_permutation_sensitive():
    rng = np.random.default_rng(2)
    w = np.abs(rng.standard_normal(64)) + 0.01
    perm = rng.permutation(w.size)
    base = input_connectivity_features(w, include_tonotopic=True)
    shuffled = input_connectivity_features(w[perm], include_tonotopic=True)
    assert not np.isclose(base["channel_com"], shuffled["channel_com"])


# --------------------------------------------------------------------------
# Weight orientation of the recurrent blocks
# --------------------------------------------------------------------------
def test_recurrent_in_uses_row_and_out_uses_column(trained_like_model):
    w_rec = trained_like_model.w_rec.detach().cpu().numpy()
    reps = extract_structural_representations(trained_like_model)
    j = 3
    expected_in = recurrent_incoming_features(w_rec[j, :])  # row = onto j
    expected_out = recurrent_outgoing_features(w_rec[:, j])  # column = from j
    assert np.isclose(reps[j].features["recurrent_in"]["l2"], expected_in["l2"])
    assert np.isclose(reps[j].features["recurrent_out"]["l2"], expected_out["l2"])


# --------------------------------------------------------------------------
# Activity block: clean naming, no exact duplicates
# --------------------------------------------------------------------------
def test_activity_features_are_cleanly_named():
    rng = np.random.default_rng(0)
    psth = rng.random((5, 20))
    counts = rng.integers(0, 5, size=(30, 5)).astype(float)
    feats, _flags = activity_features_from_psth(psth, counts, bin_ms=2.0, n_samples=30)
    assert "spike_time_cv" in feats
    for removed in ("isi_cv", "burstiness", "isi_mean_ms", "mean_spike_count"):
        assert removed not in feats
    feats = add_first_spike_features(feats, np.zeros(5), np.ones(5), duration_ms=40.0)
    assert "first_spike_latency_ms" in feats
    assert "first_spike_latency_norm" not in feats


# --------------------------------------------------------------------------
# No labels / ids / fingerprint leak into the representation
# --------------------------------------------------------------------------
def test_structural_features_avoid_ids_labels_and_fingerprint(trained_like_model):
    reps = extract_structural_representations(trained_like_model)
    names = reps.feature_names()
    assert names
    for bad in ("neuron_id", "neuron_index", "fingerprint", "class"):
        assert all(bad not in n for n in names), bad
    assert reps.meta["uses_labels"] is False
    assert reps.meta["uses_data"] is False


# --------------------------------------------------------------------------
# RepresentationSpace: geometry, weighting, filtering
# --------------------------------------------------------------------------
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
    names = ["big.f0", "big.f1", "big.f2", "small.g0"]
    X = rng.standard_normal((10, 4))
    equal = RepresentationSpace(X_raw=X, feature_names=names, weighting="equal")
    counts = equal.block_feature_counts()
    assert counts["big"] == 3 and counts["small"] == 1
    assert np.isclose(equal.block_scales[0], 1.0 / np.sqrt(3))


def test_custom_block_weighting_sets_squared_contribution():
    rng = np.random.default_rng(0)
    names = ["a.f0", "a.f1", "a.f2", "b.g0"]
    X = rng.standard_normal((40, 4))
    space = RepresentationSpace(
        X_raw=X, feature_names=names, weighting="custom", block_weights={"a": 2.0, "b": 0.5}
    )
    Z = space.X
    assert np.isclose((Z[:, :3] ** 2).sum(axis=1).mean(), 2.0, rtol=1e-6)
    assert np.isclose((Z[:, 3] ** 2).mean(), 0.5, rtol=1e-6)


def test_custom_weighting_requires_weights():
    X = np.random.default_rng(0).standard_normal((6, 2))
    with pytest.raises(ValueError):
        RepresentationSpace(X_raw=X, feature_names=["a.f0", "b.f1"], weighting="custom")


def test_feature_level_filter_isolates_learned_bias(trained_like_model):
    reps = extract_structural_representations(trained_like_model)
    space = build_space_from_representations(
        reps, [FeatureBlock.INTRINSIC.value], include_features=["learned_bias"]
    )
    assert space.feature_names == ["intrinsic.learned_bias"]
    with pytest.raises(EmptyFeatureSelectionError):
        build_space_from_representations(
            reps, [FeatureBlock.INTRINSIC.value], include_features=["tau_mem_ms"]
        )


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
    assert shuffled.n_neurons == space.n_neurons
    a = np.sort(space.X_raw.sum(axis=1))
    b = np.sort(shuffled.X_raw.sum(axis=1))
    assert np.allclose(a, b)


# --------------------------------------------------------------------------
# Ablation set completeness
# --------------------------------------------------------------------------
def test_required_ablations_are_present():
    from src.controls import default_variants

    names = {v.name for v in default_variants()}
    required = {
        "intrinsic_only",
        "input_conn_only",
        "recurrent_only",
        "activity_only",
        "structural_full",
        "structural_plus_activity",
    }
    assert required.issubset(names)
    # and the learned-bias / dynamical distinction is explicit in the ablation set
    assert "excitability_only" in names
    assert "dynamical_only" in names
