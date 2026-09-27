"""Tests for the scientific capacity evaluation of the neuron-vector representation.

Coverage (the scientific-stage contract, not the architecture):

* **Split discipline** - PROBE is evaluation-only, representation construction stays
  FIT-only, the official TEST split is never opened.
* **Target construction** - the individual-stimulus response target
  ``(n_neurons, n_probe_stimuli)``, its exact definition, no averaging across stimuli,
  no class labels, repeated examples kept separate.
* **Representation invariance** - the vectors are frozen before evaluation, and neither
  targets nor labels can change them.
* **Dimension convention** - every planned condition satisfies
  ``total_d = structured_d + residual_d``; matched total dimensions stay distinct
  decompositions; the residual contributes exactly its own coordinates.
* **Residual isolation** - training/evaluating the residual never modifies the SNN.
* **Reproducibility** - identical settings reproduce identical tables byte for byte.
* **Memory** - only documented shapes appear (no ``(neurons, samples, d)`` tensors).
* **End-to-end** - conditions, controls, result table and figures are produced.

Everything is built from the repo's tiny synthetic fixtures; no dataset or checkpoint is
needed.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
import torch

from src.capacity_figures import write_all_figures
from src.data import make_train_dev_probe_split
from src.evaluation import collect_activity
from src.functional_fingerprint import (
    FINGERPRINT_FEATURE_SETS,
    STIMULUS_RESPONSE_FEATURE_SET,
    class_conditioned_fingerprint,
    resolve_feature_sets,
    stimulus_response_fingerprint,
)
from src.model import build_model
from src.neuron_record import build_neuron_record_bank
from src.residual import (
    ResidualError,
    ResidualTrainingConfig,
    build_residual_source,
    train_residual,
)
from src.structured_vector import StructuredVectorEncoder
from src.vector_capacity import (
    CONTROL_MATCHED_DIM,
    DEFAULT_RESIDUAL_CONDITIONS,
    DEFAULT_RESIDUAL_SEEDS,
    DEFAULT_STRUCTURED_DIMS,
    KIND_CONTROL,
    KIND_STRUCTURED,
    KIND_STRUCTURED_RESIDUAL,
    RESULT_COLUMNS,
    TARGET_PRIMARY,
    EvaluationSettings,
    RepresentationCondition,
    VectorCapacityError,
    build_evaluation_targets,
    condition_matrix,
    default_conditions,
    evaluate_condition,
    fit_rate_reference,
    run_capacity_study,
    select_residual,
    summarise_across_seeds,
    write_results,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: The dimension set the study specifies (32/48/64/100/128 + 48+16 / 48+52).
PLANNED_STRUCTURED_DIMS = (32, 48, 64, 100, 128)
PLANNED_RESIDUAL_CONDITIONS = ((48, 16), (48, 52))


# --------------------------------------------------------------------------
# Fixtures (tiny: 16 hidden neurons, ~60 FIT / ~30 PROBE utterances)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def capacity_model(tiny_snn_config):
    """A tiny model with varying per-neuron bias (as after training)."""
    model = build_model(tiny_snn_config, seed=3)
    with torch.no_grad():
        model.b_hid.copy_(torch.linspace(-0.5, 0.5, tiny_snn_config.n_hidden))
    return model


@pytest.fixture(scope="module")
def capacity_split(synthetic_rec):
    fit, dev, probe, info = make_train_dev_probe_split(
        synthetic_rec, dev_fraction=0.25, probe_fraction=0.25, seed=0, prefer_speaker_aware=True
    )
    assert len(probe) > 0, "the synthetic dataset must yield a non-empty speaker-disjoint PROBE"
    return fit, dev, probe, info


@pytest.fixture(scope="module")
def fit_activity(capacity_model, capacity_split):
    """The label-free FIT activity pass (collected once, as the script does)."""
    fit_rec = capacity_split[0]
    return collect_activity(
        capacity_model, fit_rec, np.arange(len(fit_rec), dtype=np.int64),
        device="cpu", batch_size=32, n_classes=4, with_labels=False, collect_voltage=False,
    )


@pytest.fixture(scope="module")
def fit_bank(capacity_model, fit_activity):
    """The FIT-only (label-free) record bank the whole study is built from."""
    return build_neuron_record_bank(capacity_model, activity=fit_activity)


@pytest.fixture(scope="module")
def probe_activity(capacity_model, capacity_split):
    probe_rec = capacity_split[2]
    return collect_activity(
        capacity_model, probe_rec, np.arange(len(probe_rec), dtype=np.int64),
        device="cpu", batch_size=32, n_classes=4, with_labels=True, collect_voltage=False,
    )


@pytest.fixture(scope="module")
def targets(probe_activity):
    return build_evaluation_targets(probe_activity, probe_split_label="probe", n_psth_bins=4)


@pytest.fixture(scope="module")
def residuals_by_dim(fit_bank):
    """``{residual_d: {seed: ResidualResult}}`` - exactly the shape the script produces."""
    out: dict[int, dict[int, object]] = {}
    for residual_d in (16, 52):
        config = ResidualTrainingConfig.from_mapping(
            {
                "residual_dim": residual_d, "hidden_dim": 8, "epochs": 5,
                "batch_size": 8, "seed": 0, "mask_seed": 0,
            }
        )
        out[residual_d] = {0: train_residual(fit_bank, config=config)}
    return out


@pytest.fixture(scope="module")
def settings():
    return EvaluationSettings(
        n_perm=10, bootstrap=0, k_values=(3,), seed=0, n_splits=2, knn_k=2,
        checkpoint="tiny", tag="test",
    )


@pytest.fixture(scope="module")
def study(fit_bank, fit_activity, targets, residuals_by_dim, settings):
    conditions = default_conditions(
        PLANNED_STRUCTURED_DIMS, PLANNED_RESIDUAL_CONDITIONS, residual_seeds=(0,)
    )
    return run_capacity_study(
        fit_bank,
        targets,
        conditions,
        settings=settings,
        fit_rates=fit_rate_reference(fit_activity),
        residuals=residuals_by_dim,
        controls=True,
        curve_for="structured_48",
    )


# --------------------------------------------------------------------------
# Split discipline
# --------------------------------------------------------------------------
@pytest.mark.parametrize("label", ["fit", "train", "FIT", "Train"])
def test_evaluation_targets_reject_non_probe_splits(probe_activity, label):
    with pytest.raises(VectorCapacityError, match="held-out PROBE"):
        build_evaluation_targets(probe_activity, probe_split_label=label)


def test_probe_bank_is_rejected_by_the_residual_source(capacity_model, capacity_split):
    """A bank whose activity was measured on PROBE cannot enter representation training."""
    probe_rec = capacity_split[2]
    probe_bank = build_neuron_record_bank(capacity_model, fit_rec=probe_rec, device="cpu", batch_size=32)
    assert probe_bank.provenance["activity"]["split"] == "probe"
    with pytest.raises(ResidualError, match="FIT only"):
        build_residual_source(probe_bank)


def test_targets_are_built_from_probe_only(probe_activity, targets, capacity_split):
    probe_rec = capacity_split[2]
    assert targets.metadata["probe_split"] == "probe"
    assert targets.metadata["probe_n_samples"] == len(probe_rec)
    assert targets.primary.meta["split"] == "probe"
    assert targets.n_stimuli == probe_activity.n_samples


def test_script_never_reads_the_official_test_split():
    """Static + metadata discipline: the entry point opens only the training file."""
    source = (PROJECT_ROOT / "scripts" / "evaluate_vector_capacity.py").read_text(encoding="utf-8")
    assert "shd_test" not in source
    assert "official_test_loaded" in source
    assert 'test_loaded": False' in source


def test_representation_apis_have_no_label_or_target_inputs():
    for func in (condition_matrix, select_residual, train_residual, build_residual_source):
        names = set(inspect.signature(func).parameters)
        assert not (names & {"labels", "y", "target", "targets", "probe", "test"}), names
    names = set(inspect.signature(stimulus_response_fingerprint).parameters)
    assert "labels" not in names and "targets" not in names


# --------------------------------------------------------------------------
# Target construction
# --------------------------------------------------------------------------
def test_stimulus_response_target_shape_and_definition(probe_activity, targets):
    counts = np.asarray(probe_activity.counts, dtype=np.float64)
    duration_s = probe_activity.n_bins * probe_activity.bin_ms / 1000.0
    assert targets.primary.X_raw.shape == (probe_activity.n_hidden, probe_activity.n_samples)
    assert targets.n_stimuli == probe_activity.n_samples
    assert np.array_equal(targets.primary.X_raw, counts.T / duration_s)
    assert len(targets.primary.feature_names) == probe_activity.n_samples
    assert targets.primary.feature_names[0] == "fp_stimulus.s0000"


def test_primary_target_does_not_average_across_stimuli(targets):
    assert targets.primary.meta["averaging"].startswith("none across stimuli")
    assert targets.metadata["primary_uses_labels"] is False
    assert targets.metadata["primary_definition"].startswith("individual-stimulus response profile")


def test_primary_target_is_independent_of_class_labels(probe_activity, targets):
    """Shuffling the PROBE labels cannot change the primary (label-free) target."""
    rng = np.random.default_rng(0)
    shuffled = dataclasses.replace(
        probe_activity, labels=rng.permutation(np.asarray(probe_activity.labels))
    )
    other = build_evaluation_targets(shuffled, probe_split_label="probe", n_psth_bins=4)
    assert np.array_equal(targets.primary.X, other.primary.X)
    assert np.array_equal(targets.primary_rate_normalized.X, other.primary_rate_normalized.X)


def test_repeated_stimulus_examples_stay_separate_columns():
    """Two identical utterances remain two distinct stimulus columns (no class collapse)."""
    counts = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 0.0]])
    X, names = stimulus_response_fingerprint(counts, bin_ms=2.0, n_bins=30)
    assert X.shape == (2, 3)
    assert len(set(names)) == 3
    assert np.allclose(X[:, 0], X[:, 1])
    assert not np.allclose(X[:, 0], X[:, 2])


def test_stimulus_response_feature_set_is_registered_and_never_class_conditioned():
    """Regression: the primary target is a first-class feature set but not class-based."""
    assert STIMULUS_RESPONSE_FEATURE_SET in FINGERPRINT_FEATURE_SETS
    assert resolve_feature_sets(STIMULUS_RESPONSE_FEATURE_SET) == [STIMULUS_RESPONSE_FEATURE_SET]
    class_psth = np.zeros((4, 2, 30))
    with pytest.raises(ValueError, match="not a class-conditioned feature set"):
        class_conditioned_fingerprint(
            class_psth,
            np.zeros((4, 2)),
            np.array([1.0, 1.0, 1.0, 1.0]),
            np.zeros((4, 2)),
            np.zeros((4, 2)),
            n_bins=30,
            bin_ms=2.0,
            feature_sets=[STIMULUS_RESPONSE_FEATURE_SET],
        )


# --------------------------------------------------------------------------
# Representation invariance / freezing
# --------------------------------------------------------------------------
def test_representation_is_frozen_before_evaluation(fit_bank, targets, residuals_by_dim, settings):
    condition = RepresentationCondition(KIND_STRUCTURED, 48, 48, 0)
    X, names = condition_matrix(condition, fit_bank)
    X_before = X.copy()
    residual_before = residuals_by_dim[16][0].encode(fit_bank)

    first = evaluate_condition(X, names, targets, settings=settings)
    second = evaluate_condition(X, names, targets, settings=settings)

    assert np.array_equal(X, X_before), "evaluation must not modify the representation matrix"
    assert np.array_equal(residuals_by_dim[16][0].encode(fit_bank), residual_before)
    for key in ("primary_metric_value", "class_rate_metric_value", "temporal_metric_value"):
        assert first[key] == second[key]


def test_different_probe_targets_cannot_change_the_vectors(fit_bank, probe_activity):
    rng = np.random.default_rng(1)
    other_targets = build_evaluation_targets(
        dataclasses.replace(probe_activity, labels=rng.permutation(np.asarray(probe_activity.labels))),
        probe_split_label="probe", n_psth_bins=4,
    )
    condition = RepresentationCondition(KIND_STRUCTURED, 64, 64, 0)
    X_plain, _ = condition_matrix(condition, fit_bank)
    assert other_targets.n_stimuli == probe_activity.n_samples
    X_again, _ = condition_matrix(condition, fit_bank)
    assert np.array_equal(X_plain, X_again)


# --------------------------------------------------------------------------
# Dimension convention
# --------------------------------------------------------------------------
def test_planned_dimension_sets_match_the_study_specification():
    for d in PLANNED_STRUCTURED_DIMS:
        assert d in DEFAULT_STRUCTURED_DIMS
    for pair in PLANNED_RESIDUAL_CONDITIONS:
        assert pair in DEFAULT_RESIDUAL_CONDITIONS
    assert set(DEFAULT_RESIDUAL_SEEDS) >= {0, 1, 2}


def test_conditions_construct_with_the_declared_decomposition(fit_bank, residuals_by_dim):
    conditions = default_conditions(
        PLANNED_STRUCTURED_DIMS, PLANNED_RESIDUAL_CONDITIONS, residual_seeds=(0,)
    )
    for condition in conditions:
        X, names = condition_matrix(
            condition, fit_bank, residuals=residuals_by_dim
        )
        assert X.shape == (fit_bank.n_neurons, condition.total_d)
        assert condition.total_d == condition.structured_d + condition.residual_d
        assert len(names) == condition.total_d


def test_matched_total_dimensions_are_distinct_decompositions(fit_bank, residuals_by_dim):
    """structured_64 vs structured_48+residual_16: same total, different construction."""
    structured_64 = RepresentationCondition(KIND_STRUCTURED, 64, 64, 0)
    full_48_16 = RepresentationCondition(KIND_STRUCTURED_RESIDUAL, 64, 48, 16, 0)
    structured_100 = RepresentationCondition(KIND_STRUCTURED, 100, 100, 0)
    full_48_52 = RepresentationCondition(KIND_STRUCTURED_RESIDUAL, 100, 48, 52, 0)

    X64, _ = condition_matrix(structured_64, fit_bank)
    X48_16, names = condition_matrix(full_48_16, fit_bank, residuals=residuals_by_dim)
    X100, _ = condition_matrix(structured_100, fit_bank)
    X48_52, _ = condition_matrix(full_48_52, fit_bank, residuals=residuals_by_dim)

    assert X64.shape == X48_16.shape == (fit_bank.n_neurons, 64)
    assert X100.shape == X48_52.shape == (fit_bank.n_neurons, 100)
    assert not np.allclose(X64, X48_16)
    assert not np.allclose(X100, X48_52)

    # ... and the residual condition is exactly [structured prefix, residual coordinates]
    structured_48 = StructuredVectorEncoder(fit_bank, structured_d=48).encode()
    residual_16 = residuals_by_dim[16][0].encode(fit_bank)
    residual_52 = residuals_by_dim[52][0].encode(fit_bank)
    assert np.array_equal(X48_16[:, :48], structured_48.X)
    assert np.array_equal(X48_16[:, 48:], residual_16)
    assert np.array_equal(X48_52[:, :48], structured_48.X)
    assert np.array_equal(X48_52[:, 48:], residual_52)
    assert names[:48] == tuple(structured_48.feature_names)
    assert names[48:] == tuple(f"residual[{j}]" for j in range(16))


def test_inconsistent_condition_decompositions_are_rejected():
    with pytest.raises(VectorCapacityError, match="total_d"):
        RepresentationCondition(KIND_STRUCTURED, 64, 48, 8)
    with pytest.raises(VectorCapacityError, match="structured-only"):
        RepresentationCondition(KIND_STRUCTURED, 64, 48, 16)
    with pytest.raises(VectorCapacityError, match="residual_d > 0"):
        RepresentationCondition(KIND_STRUCTURED_RESIDUAL, 64, 64, 0)
    with pytest.raises(VectorCapacityError, match="seed"):
        RepresentationCondition(KIND_STRUCTURED_RESIDUAL, 64, 48, 16)


def test_condition_labels_distinguish_the_decompositions():
    assert RepresentationCondition(KIND_STRUCTURED, 64, 64, 0).label == "structured_64"
    assert RepresentationCondition(KIND_STRUCTURED_RESIDUAL, 64, 48, 16, 2).label == "full_48+16_seed2"


# --------------------------------------------------------------------------
# Residual selection / isolation
# --------------------------------------------------------------------------
def test_select_residual_accepts_both_mapping_shapes_and_rejects_mismatches(residuals_by_dim):
    condition_16 = RepresentationCondition(KIND_STRUCTURED_RESIDUAL, 64, 48, 16, 0)
    condition_52 = RepresentationCondition(KIND_STRUCTURED_RESIDUAL, 100, 48, 52, 0)
    flat = {0: residuals_by_dim[16][0]}

    assert select_residual(residuals_by_dim, condition_16) is residuals_by_dim[16][0]
    assert select_residual(residuals_by_dim, condition_52) is residuals_by_dim[52][0]
    assert select_residual(flat, condition_16) is residuals_by_dim[16][0]
    with pytest.raises(VectorCapacityError, match="residual_dim=52"):
        select_residual(flat, condition_52)
    with pytest.raises(VectorCapacityError, match="needs a trained residual"):
        select_residual(residuals_by_dim, RepresentationCondition(
            KIND_STRUCTURED_RESIDUAL, 64, 48, 16, 7
        ))
    with pytest.raises(VectorCapacityError, match="needs a trained residual"):
        select_residual(None, condition_16)


def test_residual_training_does_not_modify_the_model(capacity_model, fit_bank):
    before = {k: v.detach().clone() for k, v in capacity_model.state_dict().items()}
    config = ResidualTrainingConfig.from_mapping(
        {"residual_dim": 8, "hidden_dim": 8, "epochs": 3, "batch_size": 8, "seed": 0}
    )
    residual = train_residual(fit_bank, config=config)
    after = capacity_model.state_dict()
    assert set(before) == set(after)
    for key, value in before.items():
        assert torch.equal(value, after[key]), f"the SNN parameter {key} changed"
    assert residual.provenance["uses_labels"] is False
    assert residual.config.seed == 0


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def test_same_settings_reproduce_the_same_metrics(fit_bank, fit_activity, targets, residuals_by_dim, settings):
    conditions = default_conditions((48,), ((48, 16),), residual_seeds=(0,))
    first = run_capacity_study(
        fit_bank, targets, conditions, settings=settings,
        fit_rates=fit_rate_reference(fit_activity), residuals=residuals_by_dim,
        controls=False, curve_for=None,
    )
    second = run_capacity_study(
        fit_bank, targets, conditions, settings=settings,
        fit_rates=fit_rate_reference(fit_activity), residuals=residuals_by_dim,
        controls=False, curve_for=None,
    )
    assert [row["primary_metric_value"] for row in first["rows"]] == [
        row["primary_metric_value"] for row in second["rows"]
    ]
    assert [row["prediction_metric_value"] for row in first["rows"]] == [
        row["prediction_metric_value"] for row in second["rows"]
    ]


def test_write_results_is_deterministic(tmp_path):
    rows = [{"representation": "structured_48", "total_d": 48, "primary_metric_value": 0.5}]
    payload = {"rows": rows, "controls": [], "settings": {}, "targets": {}, "preprocessing": "x"}
    first = write_results(payload, csv_path=tmp_path / "a.csv", json_path=tmp_path / "a.json")
    second = write_results(payload, csv_path=tmp_path / "b.csv", json_path=tmp_path / "b.json")
    assert first["csv"].read_bytes() == second["csv"].read_bytes()
    assert first["json"].read_bytes() == second["json"].read_bytes()


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------
def _max_ndim(value) -> int:
    if isinstance(value, np.ndarray):
        return value.ndim
    if isinstance(value, dict):
        return max((_max_ndim(v) for v in value.values()), default=0)
    if isinstance(value, (list, tuple)):
        return max((_max_ndim(v) for v in value), default=0)
    return 0


def test_no_forbidden_tensor_shapes_in_the_evaluation_path(fit_bank, targets, study):
    fit_bank.assert_no_forbidden_tensors()
    X_rep, _rep_names = fit_bank.to_representation_set().to_matrix()
    assert X_rep.ndim == 2
    assert _max_ndim(targets.primary.X_raw) == 2
    assert _max_ndim(targets.primary_rate_normalized.X_raw) == 2
    assert _max_ndim(targets.class_rate.X_raw) == 2
    assert _max_ndim(targets.temporal.X_raw) == 2
    for row in study["rows"] + study["controls"]:
        assert _max_ndim(row) <= 2, f"condition {row.get('representation')} stored a >2-D array"
    # the only neuron x stimulus matrix is the functional target itself
    assert targets.primary.X_raw.shape == (fit_bank.n_neurons, targets.n_stimuli)


def test_condition_matrices_are_2d(fit_bank, residuals_by_dim):
    for condition in default_conditions((48, 100), ((48, 16),), residual_seeds=(0,)):
        X, names = condition_matrix(condition, fit_bank, residuals=residuals_by_dim)
        assert X.ndim == 2 and X.shape[0] == fit_bank.n_neurons and X.shape[1] == len(names)


# --------------------------------------------------------------------------
# End-to-end study, controls, result table
# --------------------------------------------------------------------------
def test_study_produces_one_row_per_condition_with_the_documented_columns(study, fit_bank, targets):
    expected = len(PLANNED_STRUCTURED_DIMS) + len(PLANNED_RESIDUAL_CONDITIONS)
    assert len(study["rows"]) == expected
    labels = {row["representation"] for row in study["rows"]}
    assert "structured_48" in labels and "full_48+16_seed0" in labels and "full_48+52_seed0" in labels
    for row in study["rows"]:
        assert row["n_neurons"] == fit_bank.n_neurons
        assert row["probe_n"] == targets.n_stimuli
        assert row["n_features"] == row["total_d"]
        assert row["total_d"] == row["structured_d"] + row["residual_d"]
        assert row["primary_target"] == TARGET_PRIMARY
        assert row["primary_metric"] == "mantel_spearman_r"
        for key in ("primary_metric_value", "class_rate_metric_value", "temporal_metric_value",
                    "prediction_metric_value", "primary_rate_normalized_r",
                    "primary_rate_matched_r", "primary_knn_best_k"):
            assert key in row
        for column in RESULT_COLUMNS:
            assert column in row


def test_controls_are_evaluated_with_the_identical_procedure(study, targets):
    assert len(study["controls"]) == 3
    kinds = {row["representation"]: row for row in study["controls"]}
    assert "control_rate_only" in kinds
    assert f"control_random_{CONTROL_MATCHED_DIM}" in kinds
    assert f"control_neuron_shuffle_{CONTROL_MATCHED_DIM}" in kinds
    for row in study["controls"]:
        assert row["representation_kind"] == KIND_CONTROL
        assert row["primary_target"] == TARGET_PRIMARY
        assert row["probe_n"] == targets.n_stimuli
    assert kinds["control_rate_only"]["total_d"] == 1
    assert kinds["control_rate_only"]["control_rate_only_primary_r"] == (
        kinds["control_rate_only"]["primary_metric_value"]
    )


def test_class_rate_and_temporal_targets_are_the_existing_definitions(targets):
    assert targets.class_rate.X_raw.shape == (targets.n_neurons, 4)  # 4 synthetic classes
    assert targets.temporal.X_raw.shape[1] > targets.class_rate.X_raw.shape[1]
    assert targets.class_rate.config.feature_sets == ["class_rate"]
    assert "class_psth" in targets.temporal.config.feature_sets
    assert "class_latency" in targets.temporal.config.feature_sets
    assert targets.class_rate.config.eval_split == "probe"


def test_seed_summary_keeps_the_individual_values(study):
    summary = summarise_across_seeds(study["rows"])
    families = {entry["representation_family"]: entry for entry in summary}
    assert "full_48+16" in families and "full_48+52" in families
    entry = families["full_48+16"]
    assert entry["n_seeds"] == 1
    assert entry["primary_metric_value_values"] == [next(
        row["primary_metric_value"] for row in study["rows"]
        if row["representation"] == "full_48+16_seed0"
    )]
    assert entry["primary_metric_value_std"] == 0.0
    # structured-only conditions are recorded once, with no seed variability
    assert families["structured_32"]["residual_seeds"] == []
    assert families["structured_32"]["primary_metric_value_values"] == [
        next(row["primary_metric_value"] for row in study["rows"]
             if row["representation"] == "structured_32")
    ]


def test_result_table_round_trips_to_csv_and_json(study, tmp_path):
    payload = {**study, "metadata": {"schema": "neuron_vector_capacity/v1"}}
    written = write_results(payload, csv_path=tmp_path / "results.csv", json_path=tmp_path / "results.json")
    header = written["csv"].read_text(encoding="utf-8").splitlines()[0].split(",")
    for column in ("representation", "total_d", "structured_d", "residual_d", "residual_seed",
                   "n_neurons", "probe_n", "primary_metric_value"):
        assert column in header
    stored = json.loads(written["json"].read_text(encoding="utf-8"))
    assert len(stored["rows"]) == len(study["rows"]) + len(study["controls"])
    assert stored["metadata"]["schema"] == "neuron_vector_capacity/v1"
    assert "summary_across_seeds" in stored


def test_figures_render_into_the_dedicated_directory(study, tmp_path):
    written = write_all_figures(study, tmp_path, curve_condition="structured_48")
    assert set(written) == {
        "figure1_capacity_vs_dimension", "figure2_primary_vs_class_rate",
        "figure3_distance_relationship",
    }
    for name, paths in written.items():
        assert paths, f"{name} was not rendered"
        for path in paths:
            assert Path(path).exists()


def test_sha256_helper_detects_changed_checkpoint_bytes(tmp_path):
    """Provenance guard: the metadata fingerprint must change when the file changes."""
    path = tmp_path / "model.pt"
    path.write_bytes(b"a" * 32)
    first = sha256(path.read_bytes()).hexdigest()[:16]
    path.write_bytes(b"a" * 31 + b"b")
    assert sha256(path.read_bytes()).hexdigest()[:16] != first