"""Tests for the stress-test orchestrator (controls, before/after, rewiring)."""

from __future__ import annotations

import numpy as np
import torch

from src.evaluation import collect_activity
from src.functional_fingerprint import FINGERPRINT_PRESETS, FingerprintSpace, class_conditioned_fingerprint
from src.neurons import (
    add_first_spike_features,
    activity_features_from_psth,
    build_activity_representations,
    extract_structural_representations,
)
from src.rewiring import rewire_recurrent
from src.stress_test import (
    StressCondition,
    before_after_summary,
    build_stress_table,
    fingerprint_split_leakage_guard,
    run_stress_test,
    stress_variants,
)

REQUIRED_VARIANTS = {
    "firing_rate_only",
    "activity_only",
    "input_conn_only",
    "recurrent_only",
    "intrinsic_dynamical_only",
    "structural_full",
    "structural_plus_activity",
    "random_representation",
    "shuffled_neurons",
}


def _build_condition(name, model, rec, cfg) -> StressCondition:
    idx = np.arange(len(rec))
    structural = extract_structural_representations(model)
    acc = collect_activity(
        model, rec, idx, device=torch.device("cpu"),
        n_classes=cfg.n_output, with_labels=False, collect_voltage=False,
    )
    feats, flags = activity_features_from_psth(
        acc.psth, acc.counts, bin_ms=acc.bin_ms, n_samples=acc.n_samples
    )
    feats = add_first_spike_features(
        feats, acc.first_spike_sum, acc.first_spike_count, duration_ms=acc.duration_ms
    )
    activity = build_activity_representations(feats, flags)

    labelled = collect_activity(
        model, rec, idx, device=torch.device("cpu"),
        n_classes=cfg.n_output, with_labels=True, collect_voltage=False,
    )
    fps = {}
    for key in ("tuning", "tuning_rate_normalized"):
        X, names = class_conditioned_fingerprint(
            labelled.class_psth, labelled.class_counts, labelled.class_n,
            labelled.class_first_spike_sum, labelled.class_first_spike_count,
            n_bins=labelled.n_bins, bin_ms=labelled.bin_ms,
            feature_sets=FINGERPRINT_PRESETS[key],
        )
        fps[key] = FingerprintSpace(X_raw=X, feature_names=names)
    return StressCondition(
        name=name, structural=structural, activity=activity, fingerprints=fps, metadata={}
    )


def _tiny_result(tiny_model, trained_like_model, synthetic_rec, tiny_snn_config):
    conditions = [
        _build_condition("after_learning", trained_like_model, synthetic_rec, tiny_snn_config),
        _build_condition("before_learning", tiny_model, synthetic_rec, tiny_snn_config),
        _build_condition(
            "rewired_global",
            rewire_recurrent(trained_like_model, mode="global", seed=0),
            synthetic_rec, tiny_snn_config,
        ),
    ]
    return run_stress_test(
        conditions,
        n_perm=40,
        k_values=[3, 5],
        seed=0,
        prediction_cfg={"enabled": True, "n_splits": 3, "alphas": (0.1, 1.0)},
        before_condition="before_learning",
        after_condition="after_learning",
    )


# --------------------------------------------------------------------------
# Variant set
# --------------------------------------------------------------------------
def test_stress_variants_cover_the_required_ten():
    names = {v.name for v in stress_variants()}
    assert REQUIRED_VARIANTS.issubset(names)
    # controls are explicitly flagged
    controls = {v.name for v in stress_variants() if v.is_control}
    assert {"firing_rate_only", "random_representation", "shuffled_neurons"}.issubset(controls)


# --------------------------------------------------------------------------
# Table
# --------------------------------------------------------------------------
def test_build_stress_table_has_the_five_required_columns():
    row = {
        "condition": "after_learning", "variant": "structural_full", "n_features": 48,
        "raw_r": 0.3, "raw_p": 1e-3, "raw_p_floor": 1e-3, "raw_at_floor": True,
        "rate_normalized_r": 0.2, "rate_matched_r": 0.15, "prediction_r2_mean": 0.1,
        "prediction_pearson_r_mean": 0.35, "is_control": False, "description": "x",
        "rate_control_verdict": "survives both rate controls",
    }
    table = build_stress_table([row])
    assert len(table) == 1
    for key in (
        "representation_control",
        "raw_geometry_function_r",
        "rate_normalized_association_r",
        "predictive_metric",
        "permutation_significance_p",
    ):
        assert key in table[0], key


def test_run_stress_test_reports_every_variant_for_every_condition(
    tiny_model, trained_like_model, synthetic_rec, tiny_snn_config
):
    result = _tiny_result(tiny_model, trained_like_model, synthetic_rec, tiny_snn_config)
    n_variants = len(stress_variants())
    conditions = result["meta"]["conditions"]
    assert len(result["table"]) == n_variants * len(conditions)
    # no result is hidden: every variant appears for every condition
    seen = {(r["condition"], r["representation_control"]) for r in result["table"]}
    for cond in conditions:
        for v in REQUIRED_VARIANTS:
            assert (cond, v) in seen


def test_run_stress_test_row_has_rate_controls_and_prediction(
    tiny_model, trained_like_model, synthetic_rec, tiny_snn_config
):
    result = _tiny_result(tiny_model, trained_like_model, synthetic_rec, tiny_snn_config)
    rows = [r for r in result["rows"] if not r.get("skipped")]
    assert rows
    for r in rows:
        # the mandatory rate-control question is always answered
        assert "rate_control_verdict" in r
        assert "rate_normalized_r" in r
        assert "rate_matched_r" in r
        # a predictive metric is reported
        assert "prediction_r2_mean" in r
        # permutation floor is recorded
        assert r["raw_p_floor"] == 1.0 / 41
    # partial Mantel is only ever labelled secondary/exploratory
    assert all("only" in result["meta"]["rate_control_policy"] for _ in [0])


def test_before_learning_has_skipped_empty_intrinsic(
    tiny_model, trained_like_model, synthetic_rec, tiny_snn_config
):
    result = _tiny_result(tiny_model, trained_like_model, synthetic_rec, tiny_snn_config)
    before = [r for r in result["table"]
              if r["condition"] == "before_learning" and r["representation_control"] == "intrinsic_dynamical_only"]
    assert before and before[0]["skipped"] is True
    # the after-learning condition (perturbed bias) has an informative intrinsic block
    after = [r for r in result["table"]
             if r["condition"] == "after_learning" and r["representation_control"] == "intrinsic_dynamical_only"]
    assert after and after[0]["skipped"] is False


def test_before_after_summary_reports_deltas(
    tiny_model, trained_like_model, synthetic_rec, tiny_snn_config
):
    result = _tiny_result(tiny_model, trained_like_model, synthetic_rec, tiny_snn_config)
    ba = result["before_after"]
    assert ba
    keys = {row["variant"] for row in ba}
    assert "structural_full" in keys
    structural = [row for row in ba if row["variant"] == "structural_full"][0]
    for key in ("raw_r_before", "raw_r_after", "delta_raw_r",
                "rate_normalized_r_before", "rate_normalized_r_after",
                "prediction_r2_before", "prediction_r2_after"):
        assert key in structural


def test_before_after_summary_handles_missing_variant():
    rows = [
        {"condition": "after_learning", "variant": "structural_full", "raw_r": 0.3,
         "rate_normalized_r": 0.2, "prediction_r2_mean": 0.1, "skipped": False},
    ]
    out = before_after_summary(rows, before="before_learning", after="after_learning")
    assert out[0]["variant"] == "structural_full"
    assert out[0]["delta_raw_r"] is None
    assert "missing" in out[0]["note"]


def test_rewired_condition_runs(tiny_model, trained_like_model, synthetic_rec, tiny_snn_config):
    result = _tiny_result(tiny_model, trained_like_model, synthetic_rec, tiny_snn_config)
    assert "rewired_global" in result["meta"]["conditions"]
    rew = [r for r in result["rows"] if r["condition"] == "rewired_global" and not r.get("skipped")]
    assert rew
    assert all(np.isfinite(r["prediction_r2_mean"] or 0.0) for r in rew)


# --------------------------------------------------------------------------
# Leakage guard: fingerprint must not be measured on the model-selection split
# --------------------------------------------------------------------------
def test_leakage_guard_passes_for_disjoint_three_way_split():
    extra = {
        "split_info": {
            "strategy": "speaker_aware_train_dev_probe",
            "splits": {"dev": {"speakers": [2]}},
        }
    }
    split_info = {"splits": {"probe": {"speakers": [6, 8]}, "dev": {"speakers": [2]}}}
    guard = fingerprint_split_leakage_guard(extra, split_info, "probe")
    assert guard["passed"] is True
    assert guard["overlap_speakers"] == []
    assert guard["model_selection_split"] == "dev"


def test_leakage_guard_flags_legacy_two_way_split():
    # legacy checkpoint selected on held_out_speakers [6, 8]; the probe is [6, 8]
    extra = {"split_info": {"strategy": "speaker_aware", "held_out_speakers": [6, 8]}}
    split_info = {"splits": {"probe": {"speakers": [6, 8]}}}
    guard = fingerprint_split_leakage_guard(extra, split_info, "probe")
    assert guard["passed"] is False
    assert guard["overlap_speakers"] == [6, 8]
    assert "LEAKAGE" in guard["note"]


def test_leakage_guard_is_unevaluable_without_provenance():
    guard = fingerprint_split_leakage_guard({}, {"splits": {"probe": {"speakers": [6, 8]}}}, "probe")
    assert guard["passed"] is None
    assert "could not determine" in guard["note"]