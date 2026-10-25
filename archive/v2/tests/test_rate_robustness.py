"""Tests for the rate-confound / robustness analysis (``src/rate_robustness.py``).

Coverage:

* **target transformations** - the raw target is bit-identical to the first study's primary
  target; neuron centering / z-scoring are exact; the zero-variance rule is deterministic;
  the pipeline order is explicit and the two orders demonstrably differ;
* **rate target** - the mean-rate target is exact (and its geometry is ``|Δrate| / std``);
  the existing row-L2 control is reproduced exactly;
* **label discipline** - every target is PROBE-only and no target enters representation or
  residual construction;
* **representation invariance** - changing the target pipeline cannot change the vectors;
* **checkpoint handling** - only compatible checkpoints are accepted;
* **reproducibility** - identical inputs/seeds reproduce identical targets and metrics;
* **memory** - only 2-D ``(neurons, stimuli)`` shapes appear.

Everything runs on the tiny synthetic fixtures (no dataset or trained checkpoint needed).
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.capacity_figures import write_rate_robustness_figures
from src.evaluation import collect_activity
from src.functional_fingerprint import (
    COLUMN_STANDARDISE_STEP,
    MEAN_OVER_STIMULI_STEP,
    NEURON_CENTER_STEP,
    NEURON_ZSCORE_STEP,
    ROW_L2_NORMALISE_STEP,
    FingerprintConfig,
    FingerprintSpace,
    apply_response_pipeline,
    mean_over_stimuli,
    neuron_center,
    neuron_zscore,
    stimulus_response_config,
    stimulus_response_matrix,
)
from src.data import make_train_dev_probe_split
from src.model import SNNConfig, build_model
from src.neuron_record import build_neuron_record_bank
from src.prediction import cross_validated_knn, cross_validated_ridge, make_shared_folds
from src.rate_robustness import (
    REP_ROLE_CONTROL,
    RESULT_COLUMNS,
    ROLE_EXISTING_CONTROL,
    ROLE_MAIN,
    ROLE_ORDERING_SENSITIVITY,
    TARGET_CS_THEN_CENTERED,
    TARGET_MEAN_RATE,
    TARGET_NEURON_CENTERED,
    TARGET_NEURON_ZSCORED,
    TARGET_RAW,
    TARGET_ROW_L2,
    TARGET_VARIANTS,
    ResponseTargetVariant,
    RobustnessCondition,
    activity_block_dimensions,
    build_response_targets,
    check_checkpoint_compatibility,
    focused_conditions,
    rate_decomposition_table,
    representation_control_rows,
    robustness_matrix,
    run_rate_robustness,
    summarise_checkpoint_variability,
    summarise_seed_variability,
    write_results,
)
from src.residual import ResidualTrainingConfig, build_residual_source, train_residual
from src.structured_vector import StructuredVectorEncoder
from src.vector_capacity import (
    EvaluationSettings,
    assert_held_out_split,
    build_evaluation_targets,
    fit_rate_reference,
    representation_space,
)
from src.v2_config import DEFAULT_ENABLED_BLOCKS


# --------------------------------------------------------------------------
# Fixtures (16 hidden neurons, ~60 FIT / ~30 PROBE utterances)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rate_model(tiny_snn_config):
    model = build_model(tiny_snn_config, seed=3)
    with torch.no_grad():
        model.b_hid.copy_(torch.linspace(-0.5, 0.5, tiny_snn_config.n_hidden))
    return model


@pytest.fixture(scope="module")
def rate_split(synthetic_rec):
    fit, dev, probe, info = make_train_dev_probe_split(
        synthetic_rec, dev_fraction=0.25, probe_fraction=0.25, seed=0, prefer_speaker_aware=True
    )
    assert len(probe) > 0
    return fit, dev, probe, info


@pytest.fixture(scope="module")
def fit_activity(rate_model, rate_split):
    fit_rec = rate_split[0]
    return collect_activity(
        rate_model, fit_rec, np.arange(len(fit_rec), dtype=np.int64), device="cpu",
        batch_size=32, n_classes=4, with_labels=False, collect_voltage=False,
    )


@pytest.fixture(scope="module")
def rate_bank(rate_model, fit_activity):
    return build_neuron_record_bank(rate_model, activity=fit_activity)


@pytest.fixture(scope="module")
def probe_activity(rate_model, rate_split):
    probe_rec = rate_split[2]
    return collect_activity(
        rate_model, probe_rec, np.arange(len(probe_rec), dtype=np.int64), device="cpu",
        batch_size=32, n_classes=4, with_labels=True, collect_voltage=False,
    )


@pytest.fixture(scope="module")
def response_targets(probe_activity):
    return build_response_targets(probe_activity, probe_split_label="probe")


@pytest.fixture(scope="module")
def rate_residuals(rate_bank):
    out: dict[int, dict[int, object]] = {}
    for residual_d in (16, 52):
        config = ResidualTrainingConfig.from_mapping(
            {"residual_dim": residual_d, "hidden_dim": 8, "epochs": 3, "batch_size": 8, "seed": 0}
        )
        out[residual_d] = {0: train_residual(rate_bank, config=config)}
    return out


@pytest.fixture(scope="module")
def rate_settings():
    return EvaluationSettings(
        n_perm=10, bootstrap=0, k_values=(3,), seed=0, n_splits=2, knn_k=2,
        checkpoint="tiny", tag="test",
    )


@pytest.fixture(scope="module")
def rate_study(rate_bank, fit_activity, response_targets, rate_residuals, rate_settings):
    conditions = focused_conditions(rate_bank, structured_dims=(48,), residual_seeds=(0,))
    result = run_rate_robustness(
        rate_bank, response_targets, conditions, settings=rate_settings,
        fit_rates=fit_rate_reference(fit_activity), residuals=rate_residuals,
        checkpoint={"checkpoint": "tiny", "checkpoint_sha256_16": "test"},
    )
    X_ref, names_ref = robustness_matrix(
        RobustnessCondition(label="structured_100", structured_d=100), rate_bank
    )
    result["rows"].extend(
        representation_control_rows(
            fit_rates=fit_rate_reference(fit_activity), targets=response_targets,
            settings=rate_settings, shuffle_reference=(X_ref, names_ref),
            checkpoint={"checkpoint": "tiny", "checkpoint_sha256_16": "test"},
        )
    )
    return result


def _max_ndim(value) -> int:
    if isinstance(value, np.ndarray):
        return value.ndim
    if isinstance(value, dict):
        return max((_max_ndim(v) for v in value.values()), default=0)
    if isinstance(value, (list, tuple)):
        return max((_max_ndim(v) for v in value), default=0)
    return 0


# --------------------------------------------------------------------------
# Target transformations: raw
# --------------------------------------------------------------------------
def test_raw_target_is_exactly_the_first_study_primary_target(probe_activity, response_targets):
    """The ``raw`` variant reproduces the first study's primary target bit-for-bit.

    The variant's ``X_raw`` is the *output of its named pipeline* (here the column
    standardisation), and its ``FingerprintSpace`` adds no further transform - so
    ``raw.X == raw.X_raw == primary.X``.
    """
    existing = build_evaluation_targets(probe_activity, probe_split_label="probe", n_psth_bins=4)
    raw = response_targets.get(TARGET_RAW)
    assert raw.meta["pipeline"] == COLUMN_STANDARDISE_STEP
    assert raw.X.shape == existing.primary.X.shape
    assert np.array_equal(raw.X, existing.primary.X)
    assert np.array_equal(raw.X_raw, existing.primary.X)
    assert not np.array_equal(raw.X_raw, existing.primary.X_raw)  # the raw Hz matrix is input, not output


def test_raw_pipeline_equals_column_standardisation(probe_activity):
    R = stimulus_response_matrix(probe_activity.counts, bin_ms=probe_activity.bin_ms,
                                 n_bins=probe_activity.n_bins)
    X, info = apply_response_pipeline(R, (COLUMN_STANDARDISE_STEP,))
    reference = FingerprintSpace(
        X_raw=R, feature_names=[f"s{i}" for i in range(R.shape[1])],
        config=stimulus_response_config(eval_split="probe"),
    )
    assert np.array_equal(X, reference.X)
    assert info["pipeline"] == COLUMN_STANDARDISE_STEP


def test_pipeline_rejects_unknown_steps():
    with pytest.raises(ValueError, match="unknown response-pipeline step"):
        apply_response_pipeline(np.ones((3, 4)), ("not_a_step",))


# --------------------------------------------------------------------------
# Target transformations: neuron centering / z-scoring
# --------------------------------------------------------------------------
def test_neuron_center_is_exact(probe_activity):
    R = stimulus_response_matrix(probe_activity.counts, bin_ms=probe_activity.bin_ms,
                                 n_bins=probe_activity.n_bins)
    centered = neuron_center(R)
    assert np.allclose(centered, R - R.mean(axis=1, keepdims=True))
    assert np.allclose(centered.mean(axis=1), 0.0, atol=1e-12)
    assert not np.allclose(centered, R)


def test_neuron_zscore_is_exact_and_keeps_the_neuron_set(probe_activity):
    R = stimulus_response_matrix(probe_activity.counts, bin_ms=probe_activity.bin_ms,
                                 n_bins=probe_activity.n_bins)
    Z, zero_variance = neuron_zscore(R)
    assert Z.shape == R.shape
    assert zero_variance.shape == (R.shape[0],)
    nonzero = ~zero_variance
    expected = (R[nonzero] - R[nonzero].mean(axis=1, keepdims=True)) / R[nonzero].std(axis=1, keepdims=True)
    assert np.allclose(Z[nonzero], expected)
    assert np.allclose(Z[nonzero].mean(axis=1), 0.0, atol=1e-12)
    assert np.allclose(Z[nonzero].std(axis=1), 1.0, atol=1e-12)


def test_zero_variance_rule_is_deterministic_and_keeps_the_row():
    R = np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0], [0.0, 0.0, 0.0]])
    first, mask_first = neuron_zscore(R)
    second, mask_second = neuron_zscore(R)
    assert np.array_equal(mask_first, mask_second)
    assert list(mask_first) == [False, True, True]
    assert np.array_equal(first, second)
    assert np.all(first[1] == 0.0) and np.all(first[2] == 0.0)
    assert first.shape == R.shape  # no neuron dropped


def test_pipeline_order_is_explicit_and_the_orders_differ(probe_activity):
    R = stimulus_response_matrix(probe_activity.counts, bin_ms=probe_activity.bin_ms,
                                 n_bins=probe_activity.n_bins)
    centered, info_a = apply_response_pipeline(R, (NEURON_CENTER_STEP, COLUMN_STANDARDISE_STEP))
    cs_first, info_b = apply_response_pipeline(R, (COLUMN_STANDARDISE_STEP, NEURON_CENTER_STEP))
    assert info_a["pipeline"] == "neuron_center -> column_standardise"
    assert info_b["pipeline"] == "column_standardise -> neuron_center"
    # the last step decides the row means: centering last => exactly zero row means
    assert np.allclose(cs_first.mean(axis=1), 0.0, atol=1e-12)
    assert not np.allclose(centered.mean(axis=1), 0.0, atol=1e-9)
    assert not np.allclose(centered, cs_first)


def test_target_variants_declare_their_pipeline_and_role():
    by_name = {v.name: v for v in TARGET_VARIANTS}
    assert by_name[TARGET_RAW].role == ROLE_MAIN
    assert by_name[TARGET_NEURON_CENTERED].centering == "neuron_mean"
    assert by_name[TARGET_NEURON_ZSCORED].scaling == "neuron_std_then_column_std"
    assert by_name[TARGET_RAW].steps[0] == COLUMN_STANDARDISE_STEP
    assert by_name[TARGET_CS_THEN_CENTERED].role == ROLE_ORDERING_SENSITIVITY
    assert by_name[TARGET_ROW_L2].role == ROLE_EXISTING_CONTROL
    for variant in TARGET_VARIANTS:
        assert variant.pipeline == " -> ".join(variant.steps)


# --------------------------------------------------------------------------
# Rate target and the existing controls
# --------------------------------------------------------------------------
def test_mean_rate_target_is_exact(probe_activity, response_targets):
    R = stimulus_response_matrix(probe_activity.counts, bin_ms=probe_activity.bin_ms,
                                 n_bins=probe_activity.n_bins)
    space = response_targets.get(TARGET_MEAN_RATE)
    assert space.X_raw.shape == (R.shape[0], 1)
    assert np.allclose(mean_over_stimuli(R), R.mean(axis=1, keepdims=True))
    z = (R.mean(axis=1) - R.mean(axis=1).mean()) / R.mean(axis=1).std()
    assert np.allclose(space.X[:, 0], z)
    assert np.allclose(response_targets.mean_rate_hz, R.mean(axis=1))


def test_mean_rate_target_geometry_is_the_absolute_rate_difference(probe_activity, response_targets):
    from scipy.spatial.distance import pdist

    rates = response_targets.mean_rate_hz
    z = (rates - rates.mean()) / rates.std()
    expected = pdist(z.reshape(-1, 1))
    assert np.allclose(response_targets.get(TARGET_MEAN_RATE).condensed(), expected)


def test_row_l2_target_reproduces_the_existing_rate_control(response_targets):
    """The new ``row_l2_normalised`` variant is the first study's ``primary_rate_normalized``."""
    space = response_targets.get(TARGET_ROW_L2)
    R = response_targets.raw_response
    reference = FingerprintSpace(
        X_raw=R, feature_names=[f"s{i}" for i in range(R.shape[1])],
        config=FingerprintConfig(standardize="none", normalize_rows=True, eval_split="probe"),
    )
    assert np.array_equal(space.X, reference.X)
    norms = np.linalg.norm(space.X, axis=1)
    assert np.allclose(norms, 1.0)


def test_rate_only_representation_control_is_the_standardised_fit_rate(fit_activity):
    """The representation-side rate control is a 1-D z-scored FIT-rate representation."""
    from scipy.spatial.distance import pdist

    rates = fit_rate_reference(fit_activity)
    space = representation_space(rates.reshape(-1, 1), ["activity.rate_hz"])
    z = (rates - rates.mean()) / rates.std()
    assert np.allclose(space.condensed(), pdist(z.reshape(-1, 1)))


def test_target_features_and_metadata_document_the_pipeline(response_targets):
    meta = response_targets.metadata
    assert meta["operations_do_not_commute"] is True
    assert "std_s R" in meta["zero_variance_rule"]
    assert meta["probe_split"] == "probe"
    assert meta["uses_labels"] is False
    for variant in response_targets.variants:
        assert response_targets.get(variant.name).meta["pipeline"] == variant.pipeline
    assert response_targets.n_stimuli == response_targets.raw_response.shape[1]


# --------------------------------------------------------------------------
# Label discipline / probe-only
# --------------------------------------------------------------------------
@pytest.mark.parametrize("label", ["fit", "train", "FIT"])
def test_response_targets_reject_non_probe_splits(probe_activity, label):
    with pytest.raises(ValueError, match="held-out PROBE"):
        build_response_targets(probe_activity, probe_split_label=label)
    with pytest.raises(ValueError, match="held-out PROBE"):
        assert_held_out_split(label)


def test_target_transforms_take_no_labels_and_no_representation():
    for func in (neuron_center, mean_over_stimuli):
        names = set(inspect.signature(func).parameters)
        assert not (names & {"labels", "y", "targets", "bank", "X"})
    from src.rate_robustness import evaluate_target, run_rate_robustness  # noqa: F401
    for func in (evaluate_target, run_rate_robustness):
        names = set(inspect.signature(func).parameters)
        assert "labels" not in names
    # the residual source/training APIs remain label-free and target-free
    for func in (build_residual_source, train_residual):
        assert "labels" not in set(inspect.signature(func).parameters)


def test_building_targets_does_not_change_the_representation(rate_bank, probe_activity, rate_residuals):
    condition = RobustnessCondition(
        label="full_48+16_seed0", structured_d=48, residual_d=16, residual_seed=0,
        kind="structured_plus_residual",
    )
    X_before, names_before = robustness_matrix(condition, rate_bank, residuals=rate_residuals)
    residual_before = rate_residuals[16][0].encode(rate_bank)
    build_response_targets(probe_activity, probe_split_label="probe")
    X_after, names_after = robustness_matrix(condition, rate_bank, residuals=rate_residuals)
    assert names_after == names_before
    assert np.array_equal(X_before, X_after)
    assert np.array_equal(rate_residuals[16][0].encode(rate_bank), residual_before)

    # the structured-only path is likewise unaffected by any target construction
    structured = RobustnessCondition(label="structured_48", structured_d=48)
    a, _ = robustness_matrix(structured, rate_bank)
    build_response_targets(probe_activity, probe_split_label="probe")
    b, _ = robustness_matrix(structured, rate_bank)
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------
# Dimensions / conditions
# --------------------------------------------------------------------------
def test_activity_dimensions_are_derived_from_the_encoder_plan(rate_bank):
    dims = activity_block_dimensions(rate_bank)
    single = StructuredVectorEncoder(rate_bank, structured_d=1, enabled_blocks=("activity",))
    combined = StructuredVectorEncoder(
        rate_bank, structured_d=1, enabled_blocks=(*DEFAULT_ENABLED_BLOCKS, "activity")
    )
    assert dims["activity_level0"] == single.plan.level0_dimension
    assert dims["activity_source"] == single.plan.source_dimension
    assert dims["structural_plus_activity_level0"] == combined.plan.level0_dimension
    assert dims["activity_level0"] < dims["activity_source"]


def test_focused_conditions_construct_with_the_declared_dimensions(rate_bank, rate_residuals):
    conditions = focused_conditions(rate_bank, structured_dims=(48, 64), residual_seeds=(0,))
    labels = {c.label for c in conditions}
    for expected in ("structured_48", "structured_64", "full_48+16_seed0",
                     "full_48+52_seed0", "activity_only", "structural_48_plus_activity"):
        assert expected in labels
    for condition in conditions:
        X, names = robustness_matrix(condition, rate_bank, residuals=rate_residuals)
        assert X.shape == (rate_bank.n_neurons, condition.total_d)
        assert len(names) == condition.total_d


def test_missing_residual_artifact_is_rejected(rate_bank):
    condition = RobustnessCondition(
        label="full_48+16_seed9", structured_d=48, residual_d=16, residual_seed=9,
        kind="structured_plus_residual",
    )
    with pytest.raises(ValueError, match="needs a trained residual"):
        robustness_matrix(condition, rate_bank, residuals={16: {0: None}})


def test_structural_plus_activity_contains_the_48_structural_prefix(rate_bank):
    dims = activity_block_dimensions(rate_bank)
    structural = StructuredVectorEncoder(rate_bank, structured_d=48).encode()
    combined_X, _ = robustness_matrix(
        [c for c in focused_conditions(rate_bank) if c.label == "structural_48_plus_activity"][0],
        rate_bank,
    )
    activity_X, _ = robustness_matrix(
        [c for c in focused_conditions(rate_bank) if c.label == "activity_only"][0], rate_bank
    )
    assert np.array_equal(combined_X[:, :48], structural.X)
    assert np.array_equal(combined_X[:, 48:], activity_X)
    assert combined_X.shape[1] == 48 + dims["activity_level0"]


# --------------------------------------------------------------------------
# Checkpoint handling
# --------------------------------------------------------------------------
def test_checkpoint_compatibility_accepts_identical_and_rejects_different(rate_bank, rate_model):
    same = check_checkpoint_compatibility(("a", rate_model, rate_bank), ("b", rate_model, rate_bank))
    assert same.compatible and same.reasons == []

    other_cfg = SNNConfig(n_input=20, n_hidden=8, n_output=4, n_bins=30, bin_ms=2.0)
    other_model = build_model(other_cfg, seed=1)
    with torch.no_grad():
        other_model.b_hid.copy_(torch.linspace(-0.5, 0.5, 8))
    other_bank = build_neuron_record_bank(other_model, with_activity=False)
    different = check_checkpoint_compatibility(
        ("a", rate_model, rate_bank), ("other", other_model, other_bank)
    )
    assert not different.compatible
    assert any("n_hidden" in reason for reason in different.reasons)
    assert any("n_neurons" in reason for reason in different.reasons)


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def test_targets_and_metrics_reproduce_exactly(probe_activity, rate_bank, fit_activity,
                                                rate_residuals, rate_settings):
    first = build_response_targets(probe_activity, probe_split_label="probe")
    second = build_response_targets(probe_activity, probe_split_label="probe")
    for variant in TARGET_VARIANTS:
        assert np.array_equal(first.get(variant.name).X, second.get(variant.name).X)

    conditions = focused_conditions(rate_bank, structured_dims=(48,), residual_seeds=(0,))
    a = run_rate_robustness(
        rate_bank, first, conditions, settings=rate_settings,
        fit_rates=fit_rate_reference(fit_activity), residuals=rate_residuals,
    )
    b = run_rate_robustness(
        rate_bank, second, conditions, settings=rate_settings,
        fit_rates=fit_rate_reference(fit_activity), residuals=rate_residuals,
    )
    assert [r["value"] for r in a["rows"]] == [r["value"] for r in b["rows"]]


def test_prediction_accepts_a_single_column_target(rate_bank, response_targets, rate_settings):
    """Regression: the mean-rate target is 1-D and must be consumable by the CV predictors."""
    X, _ = robustness_matrix(
        [c for c in focused_conditions(rate_bank) if c.label == "structured_48"][0], rate_bank
    )
    Y = response_targets.get(TARGET_MEAN_RATE).X
    assert Y.shape[1] == 1
    ridge = cross_validated_ridge(
        X, Y, n_splits=rate_settings.n_splits, seed=0,
        cv=make_shared_folds(X.shape[0], rate_settings.n_splits, 0),
    )
    assert "r2_mean" in ridge["metrics"]
    knn = cross_validated_knn(X, Y, n_splits=rate_settings.n_splits, k=2, seed=0)
    assert "r2_mean" in knn["metrics"]
    with pytest.raises(ValueError, match="2-D"):
        cross_validated_ridge(X, Y[:, 0], n_splits=2, seed=0)


# --------------------------------------------------------------------------
# Study-level checks
# --------------------------------------------------------------------------
def test_study_rows_cover_conditions_and_variants(rate_study, rate_bank, response_targets):
    rows = rate_study["rows"]
    representations = {row["representation"] for row in rows}
    assert {"structured_48", "full_48+16_seed0", "activity_only",
            "control_rate_only", "control_random_100"} <= representations
    for row in rows:
        assert row["n_neurons"] == rate_bank.n_neurons
        assert row["probe_n"] == response_targets.n_stimuli
        if row["representation_role"] != REP_ROLE_CONTROL:
            assert row["total_d"] == row["structured_d"] + (row["residual_d"] or 0)
        assert row["metric"] == "mantel_spearman_r"
        assert row["target_pipeline"] == " -> ".join(
            next(v for v in TARGET_VARIANTS if v.name == row["target_variant"]).steps
        )
    control = next(r for r in rows if r["representation"] == "control_rate_only")
    assert control["representation_role"] == REP_ROLE_CONTROL
    assert control["total_d"] == 1
    assert control["rate_only_primary_r"] == next(
        r["value"] for r in rows
        if r["representation"] == "control_rate_only" and r["target_variant"] == TARGET_RAW
    )


def test_decomposition_table_computes_the_descriptive_differences(rate_study):
    table = rate_decomposition_table(rate_study["rows"])
    assert table
    for entry in table:
        if entry["delta_centering"] is not None:
            assert entry["delta_centering"] == pytest.approx(entry["centered_r"] - entry["raw_r"])
        if entry["delta_scaling"] is not None:
            assert entry["delta_scaling"] == pytest.approx(entry["zscored_r"] - entry["centered_r"])
    families = {entry["representation"] for entry in table}
    assert "full_48+16_seed0" in families
    assert "activity_source_20" in families


def test_seed_and_checkpoint_summaries_keep_individual_values(rate_study):
    seeds = summarise_seed_variability(rate_study["rows"])
    assert seeds
    entry = seeds[0]
    assert entry["n_seeds"] >= 1
    assert len(entry["value_values"]) == entry["n_seeds"]
    assert entry["value_std"] == pytest.approx(
        np.std(entry["value_values"], ddof=0) if entry["n_seeds"] > 1 else 0.0
    )
    checkpoints = summarise_checkpoint_variability(rate_study["rows"])
    assert any(row["representation"] == "structured_48" for row in checkpoints)
    assert all(row["n_checkpoints"] == 1 for row in checkpoints)


def test_results_write_to_dedicated_files_with_the_documented_columns(rate_study, tmp_path):
    csv_path = tmp_path / "rate_robustness_results.csv"
    json_path = tmp_path / "rate_robustness_results.json"
    written = write_results(rate_study, csv_path=csv_path, json_path=json_path)
    header = written["csv"].read_text(encoding="utf-8").splitlines()[0].split(",")
    for column in ("target_variant", "target_centering", "target_scaling", "checkpoint",
                   "representation", "total_d", "structured_d", "residual_d", "residual_seed",
                   "metric", "value", "bootstrap_low", "bootstrap_high", "permutation_p"):
        assert column in header
        assert column in RESULT_COLUMNS
    payload = json.loads(written["json"].read_text(encoding="utf-8"))
    assert payload["schema"] == "neuron_vector_rate_robustness/v1"
    assert "rate_decomposition_table" in payload
    # the first study's filenames are never produced here
    assert not (tmp_path / "results.csv").exists()
    assert not (tmp_path / "metadata.json").exists()


def test_rate_robustness_figures_render(rate_study, tmp_path):
    written = write_rate_robustness_figures(rate_study, tmp_path)
    assert set(written) == {
        "rate_figure1_target_decomposition",
        "rate_figure2_representation_comparison",
        "rate_figure3_activity_diagnostic",
    }
    for name, paths in written.items():
        assert paths, f"{name} was not rendered"
        for path in paths:
            assert Path(path).exists()


def test_only_two_dimensional_targets_are_materialised(rate_bank, response_targets, rate_study):
    rate_bank.assert_no_forbidden_tensors()
    assert response_targets.raw_response.ndim == 2
    for name, space in response_targets.spaces.items():
        assert space.X_raw.ndim == 2, name
        assert _max_ndim(space.X_raw) == 2
    for row in rate_study["rows"]:
        assert _max_ndim(row) <= 2


def test_variant_pipeline_strings_are_explicit():
    variant = ResponseTargetVariant(
        name="custom", steps=(NEURON_CENTER_STEP, COLUMN_STANDARDISE_STEP),
        centering="neuron_mean", scaling="column_std", role=ROLE_MAIN, description="custom",
    )
    assert variant.pipeline == f"{NEURON_CENTER_STEP} -> {COLUMN_STANDARDISE_STEP}"
    # an unknown step is only rejected when the pipeline is applied
    with pytest.raises(ValueError, match="unknown response-pipeline step"):
        apply_response_pipeline(np.ones((2, 3)), ("nope",))