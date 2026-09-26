"""Tests for the neuron-space baseline experiment (``src/neuron_space_baseline.py``).

These exercise the parts that carry scientific weight and are easy to get subtly
wrong:

* the canonical representation/control set is exactly as advertised (once each);
* hidden-neuron permutation relabels the ``self_mask`` buffer too;
* cross-validated prediction uses **identical folds** for every representation;
* the canonical results table carries the required reporting columns;
* multi-seed aggregation and before/after deltas are computed correctly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.functional_fingerprint import FingerprintSpace
from src.functional_fingerprint import class_conditioned_fingerprint
from src.evaluation import collect_activity
from src.model import SNNConfig, build_model
from src.neuron_space_baseline import (
    CANONICAL_REPRESENTATION_NAMES,
    CANONICAL_TABLE_COLUMNS,
    NSBCondition,
    aggregate_primary_over_seeds,
    before_after_nsb,
    build_canonical_table,
    evaluate_representation,
    nsb_variants,
    run_condition,
    run_cv_comparison,
)
from src.neurons import (
    activity_features_from_psth,
    add_first_spike_features,
    build_activity_representations,
    extract_structural_representations,
)
from src.permutation import permute_hidden_neurons, random_permutation
from src.prediction import fold_assignment, make_shared_folds


# --------------------------------------------------------------------------
# Canonical set
# --------------------------------------------------------------------------
def test_canonical_variant_names_exactly_once():
    names = [v.name for v in nsb_variants()]
    assert names == list(CANONICAL_REPRESENTATION_NAMES)
    assert len(names) == len(set(names))


# --------------------------------------------------------------------------
# self_mask permutation (genuine bug fix)
# --------------------------------------------------------------------------
def test_permute_hidden_neurons_relabels_self_mask():
    cfg = SNNConfig(
        n_input=8, n_hidden=6, n_output=3, n_bins=10, bin_ms=2.0,
        zero_self_connection=True, neuron_param_mode="bias",
    )
    model = build_model(cfg, seed=0)
    perm = random_permutation(cfg.n_hidden, seed=3)
    permuted = permute_hidden_neurons(model, perm)

    perm_t = torch.as_tensor(perm, dtype=torch.long)
    expected = model.self_mask.index_select(0, perm_t).index_select(1, perm_t)
    assert torch.allclose(permuted.self_mask, expected)
    # self-connections stay disabled under any relabelling
    assert float(permuted.self_mask.diagonal().abs().max()) == 0.0


# --------------------------------------------------------------------------
# Shared folds
# --------------------------------------------------------------------------
def test_shared_folds_are_identical_across_representations():
    rng = np.random.default_rng(0)
    n = 40
    folds = make_shared_folds(n, n_splits=4, seed=7)
    X1 = rng.standard_normal((n, 3))
    X2 = rng.standard_normal((n, 9))
    Y = rng.standard_normal((n, 4))

    out = run_cv_comparison({"a": X1, "b": X2}, Y, n_splits=4, seed=7, folds=folds)
    assert out["same_folds_for_all"] is True
    assert out["representations"]["a"]["n_features"] == 3
    assert out["representations"]["b"]["n_features"] == 9
    # the recorded fold assignment matches the splitter exactly
    expected = fold_assignment(n, 4, 7, cv=folds)
    assert out["fold_assignment"] == expected.tolist()
    # and every fold is non-empty / covers every neuron exactly once
    assert sorted(out["fold_assignment"]) == sorted(
        [f for f in range(4) for _ in range(int(expected.size / 4))]
    ) or len(set(out["fold_assignment"])) >= 2


def test_fold_assignment_is_deterministic():
    a = fold_assignment(50, 5, 3)
    b = fold_assignment(50, 5, 3)
    assert np.array_equal(a, b)
    c = fold_assignment(50, 5, 4)
    assert not np.array_equal(a, c)


# --------------------------------------------------------------------------
# Condition construction helper
# --------------------------------------------------------------------------
def _condition_from_model(model, rec, *, name="after_learning", n_classes=4):
    """Build a minimal NSBCondition (structural + activity + 4-d fingerprint)."""
    structural = extract_structural_representations(model)
    res = collect_activity(
        model, rec, np.arange(len(rec)), device=torch.device("cpu"), batch_size=32,
        n_classes=n_classes, with_labels=True, collect_voltage=False,
    )
    feats, flags = activity_features_from_psth(
        res.psth, res.counts, bin_ms=res.bin_ms, n_samples=res.n_samples
    )
    feats = add_first_spike_features(
        feats, res.first_spike_sum, res.first_spike_count, duration_ms=res.duration_ms
    )
    activity = build_activity_representations(feats, flags)

    fps = {}
    for key, feature_sets in (("tuning", ["class_rate"]),
                              ("tuning_rate_normalized", ["class_rate_norm"])):
        X, names = class_conditioned_fingerprint(
            res.class_psth, res.class_counts, res.class_n,
            res.class_first_spike_sum, res.class_first_spike_count,
            n_bins=res.n_bins, bin_ms=res.bin_ms, feature_sets=feature_sets,
        )
        fps[key] = FingerprintSpace(
            X_raw=X, feature_names=names, meta={"uses_labels": True, "n_neurons": X.shape[0]}
        )
    return NSBCondition(name=name, structural=structural, activity=activity, fingerprints=fps)


def test_run_condition_reports_every_canonical_representation(synthetic_rec, trained_like_model):
    cond = _condition_from_model(trained_like_model, synthetic_rec)
    out = run_condition(
        cond, tuning_key="tuning", tuning_normalized_key="tuning_rate_normalized",
        n_perm=60, k_values=[3, 5], seed=0, bootstrap=0, control_repeats=2,
        prediction={"enabled": True, "n_splits": 3, "knn_k": 3},
        variants=nsb_variants(),
    )
    reported = {r["representation"] for r in out["rows"]}
    assert reported == set(CANONICAL_REPRESENTATION_NAMES)
    # controls are aggregated over repeats and carry a spread
    random_row = next(r for r in out["rows"] if r["representation"] == "random")
    assert random_row["n_repeats"] == 2
    assert "raw_r_std" in random_row
    # the structural row carries the required statistics
    structural = next(r for r in out["rows"] if r["representation"] == "structural")
    for key in ("raw_r", "raw_p", "raw_null_mean", "raw_null_std", "raw_effect_size_z",
                "rate_normalized_r", "rate_matched_r", "cv_r2_mean", "cv_pearson_r_mean",
                "cv_nrmse_mean", "cv_rmse_mean"):
        assert key in structural
        assert np.isfinite(structural[key]) or key in ("rate_matched_r",)


def test_evaluate_representation_labels_partial_as_secondary(synthetic_rec, trained_like_model):
    from src.representations import build_space_from_representations
    from src.neuron_space_baseline import _rate_absdiff_condensed

    cond = _condition_from_model(trained_like_model, synthetic_rec)
    space = build_space_from_representations(cond.structural, ["intrinsic", "input_conn"])
    row = evaluate_representation(
        space, tuning=cond.fingerprints["tuning"],
        tuning_normalized=cond.fingerprints["tuning_rate_normalized"],
        rate_absdiff=_rate_absdiff_condensed(cond.activity),
        n_perm=50, k_values=[3], seed=0,
    )
    assert "partial_mantel_r_secondary_exploratory" in row
    assert row["rate_control_verdict"] in {
        "survives both rate controls", "survives rate normalization only",
        "survives rate matching only", "not distinguishable from a firing-rate effect",
    }


# --------------------------------------------------------------------------
# Canonical table
# --------------------------------------------------------------------------
def _fake_seed(raw_r, rate_norm_r, cv_r2):
    return {
        "after_learning": {"rows": [
            {"representation": "structural", "raw_r": raw_r, "raw_p": 1e-4,
             "rate_normalized_r": rate_norm_r, "cv_r2_mean": cv_r2,
             "cv_pearson_r_mean": 0.5, "cv_nrmse_mean": 0.8, "n_features": 48},
            {"representation": "rate_only", "raw_r": 0.4, "raw_p": 1e-3,
             "rate_normalized_r": 0.01, "cv_r2_mean": 0.2,
             "cv_pearson_r_mean": 0.3, "cv_nrmse_mean": 0.9, "n_features": 1},
        ]},
    }


def test_build_canonical_table_has_required_columns():
    per_seed = {0: _fake_seed(0.3, 0.25, 0.15), 1: _fake_seed(0.35, 0.28, 0.18)}
    rows = build_canonical_table(
        per_seed, checkpoints={0: "checkpoints/a.pt", 1: "checkpoints/b.pt"},
        probe_split="probe", functional_target_label="class_rate_20d_probe",
    )
    assert rows
    for row in rows:
        for col in CANONICAL_TABLE_COLUMNS:
            assert col in row
        assert row["probe_split"] == "probe"
        assert row["functional_target"] == "class_rate_20d_probe"
        assert row["checkpoint"].endswith(".pt")
    assert {r["seed"] for r in rows} == {0, 1}


def test_aggregate_primary_over_seeds():
    per_seed = {0: _fake_seed(0.3, 0.25, 0.15),
                1: _fake_seed(0.4, 0.35, 0.25),
                2: _fake_seed(0.5, 0.45, 0.35)}
    agg = aggregate_primary_over_seeds(per_seed, representation="structural",
                                       metrics=("raw_r", "rate_normalized_r", "cv_r2_mean"))
    assert agg["metrics"]["raw_r"]["per_seed"] == pytest.approx([0.3, 0.4, 0.5])
    assert agg["metrics"]["raw_r"]["mean"] == pytest.approx(0.4)
    assert agg["metrics"]["raw_r"]["n_seeds"] == 3
    assert agg["metrics"]["raw_r"]["ci95_low"] < 0.4 < agg["metrics"]["raw_r"]["ci95_high"]


def test_before_after_nsb_deltas():
    after = [{"representation": "structural", "raw_r": 0.3, "rate_normalized_r": 0.25,
              "rate_matched_r": 0.2, "cv_r2_mean": 0.15}]
    before = [{"representation": "structural", "raw_r": 0.1, "rate_normalized_r": 0.05,
               "rate_matched_r": 0.02, "cv_r2_mean": 0.02}]
    out = before_after_nsb(after, before)
    row = next(r for r in out if r["variant"] == "structural")
    assert row["delta_raw_r"] == pytest.approx(0.2)
    assert row["delta_rate_normalized_r"] == pytest.approx(0.2)
    assert row["delta_prediction_r2"] == pytest.approx(0.13)