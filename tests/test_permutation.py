"""Permutation-invariance tests for the neuron representation.

The scientific requirement is explicit: if the hidden-neuron *ordering* is
permuted together with every corresponding parameter, the representation attached
to the same functional neuron must be unchanged. Anything that changes is
permutation-sensitive and invalid as a neuron representation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.neurons import (
    activity_features_from_psth,
    build_activity_representations,
    extract_structural_representations,
)
from src.permutation import (
    assert_permutation_invariant,
    check_structural_permutation_invariance,
    compare_representation_sets,
    compare_space_invariance,
    functional_equivalence_error,
    neuron_index_representation,
    permute_hidden_neurons,
    random_permutation,
)
from src.representations import build_space_from_representations


# --------------------------------------------------------------------------
# Step 3: the permuted network must be functionally identical
# --------------------------------------------------------------------------
def test_permuted_model_is_functionally_identical(tiny_model, tiny_input):
    n = tiny_model.cfg.n_hidden
    perm = random_permutation(n, seed=7)
    permuted = permute_hidden_neurons(tiny_model, perm)
    x = torch.as_tensor(np.asarray(tiny_input), dtype=tiny_model.w_in.dtype)

    with torch.no_grad():
        a = tiny_model(x, record=True)
        b = permuted(x, record=True)

    # readout is unchanged
    assert torch.allclose(a["logits"], b["logits"], atol=1e-5)
    # hidden spikes are relabelled: new neuron k behaves like old neuron perm[k]
    perm_t = torch.as_tensor(perm, dtype=torch.long)
    expected = a["hidden_spikes"].index_select(2, perm_t)
    # allow rare threshold flips from floating-point reassociation
    assert (expected - b["hidden_spikes"]).abs().mean().item() < 1e-3

    err = functional_equivalence_error(tiny_model, permuted, x)
    assert err["logits_max_abs_diff"] < 1e-5


def test_permute_requires_a_true_permutation(tiny_model):
    with pytest.raises(ValueError):
        permute_hidden_neurons(tiny_model, np.zeros(tiny_model.cfg.n_hidden, dtype=int))


# --------------------------------------------------------------------------
# Steps 1-5: structural representation invariance
# --------------------------------------------------------------------------
def test_structural_representation_is_permutation_invariant(trained_like_model):
    report = check_structural_permutation_invariance(trained_like_model, seed=3)
    assert report["passed"], report["sensitive_features"]
    assert report["sensitive_features"] == []
    assert report["sensitive_blocks"] == []
    assert report["n_features"] > 0
    assert_permutation_invariant(report)


def test_tonotopic_structural_features_stay_invariant_under_hidden_permutation(trained_like_model):
    # Permuting hidden neurons does NOT permute input channels, so even the
    # (opt-in) tonotopic features are invariant here - they are ordering-dependent
    # in the *channel* sense, not in the hidden-neuron sense.
    report = check_structural_permutation_invariance(
        trained_like_model, seed=4, include_tonotopic=True
    )
    assert report["passed"], report["sensitive_features"]
    assert any("channel_com" in f for f in report["invariant_features"])


def test_representation_geometry_is_permutation_invariant(trained_like_model):
    n = trained_like_model.cfg.n_hidden
    perm = random_permutation(n, seed=5)
    ref_reps = extract_structural_representations(trained_like_model)
    perm_reps = extract_structural_representations(permute_hidden_neurons(trained_like_model, perm))
    ref_space = build_space_from_representations(ref_reps, ["intrinsic", "input_conn", "recurrent_in", "recurrent_out"])
    perm_space = build_space_from_representations(perm_reps, ["intrinsic", "input_conn", "recurrent_in", "recurrent_out"])
    result = compare_space_invariance(ref_space, perm_space, perm)
    assert result["passed"], result


# --------------------------------------------------------------------------
# The detector must actually catch a permutation-sensitive representation
# --------------------------------------------------------------------------
def test_detector_flags_neuron_index_representation(tiny_model):
    n = tiny_model.cfg.n_hidden
    perm = random_permutation(n, seed=6)
    reference = neuron_index_representation(tiny_model)
    permuted = neuron_index_representation(permute_hidden_neurons(tiny_model, perm))
    report = compare_representation_sets(reference, permuted, perm)
    assert report["passed"] is False
    assert "intrinsic.neuron_index" in report["sensitive_features"]
    with pytest.raises(AssertionError):
        assert_permutation_invariant(report)
    # the control can be explicitly tolerated
    assert_permutation_invariant(report, allow=["intrinsic.neuron_index"])


# --------------------------------------------------------------------------
# The activity block is a per-neuron function (deterministic equivariance test)
# --------------------------------------------------------------------------
def test_activity_feature_extraction_is_permutation_equivariant():
    rng = np.random.default_rng(0)
    n_neurons, n_bins = 6, 25
    psth = rng.random((n_neurons, n_bins))
    counts = rng.integers(0, 6, size=(40, n_neurons)).astype(float)
    feats, flags = activity_features_from_psth(psth, counts, bin_ms=2.0, n_samples=40)

    perm = rng.permutation(n_neurons)
    feats_perm, flags_perm = activity_features_from_psth(
        psth[perm], counts[:, perm], bin_ms=2.0, n_samples=40
    )
    for name, values in feats.items():
        assert np.allclose(feats_perm[name], np.asarray(values)[perm]), name
    assert np.array_equal(flags_perm["silent_neuron"], np.asarray(flags["silent_neuron"])[perm])

    reps = build_activity_representations(feats, flags)
    reps_perm = build_activity_representations(feats_perm, flags_perm)
    report = compare_representation_sets(reps, reps_perm, perm)
    assert report["passed"], report["sensitive_features"]


def test_activity_representation_is_invariant_on_permuted_model(tiny_model, synthetic_rec):
    from src.permutation import check_activity_permutation_invariance

    n = tiny_model.cfg.n_hidden
    perm = random_permutation(n, seed=8)
    report = check_activity_permutation_invariance(
        tiny_model,
        synthetic_rec,
        np.arange(len(synthetic_rec)),
        perm,
        device=torch.device("cpu"),
        n_classes=4,
        batch_size=64,
        atol=1e-3,
        rtol=1e-2,
    )
    assert report["passed"], report["sensitive_features"]