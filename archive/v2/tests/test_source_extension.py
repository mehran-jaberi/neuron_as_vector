"""Tests for the source-extension evaluation (functional-response + temporal sources).

Coverage:

* **source ablation** - the functional-response and temporal sources can be enabled
  independently and together, the four A/B/C/D arms exist at the ablation dimension, and the
  mask diagnostic is added exactly once;
* **configuration** - source flags propagate from the condition to the deterministic source
  view, and ``total_d = structured_d + residual_d`` is never ambiguous;
* **evaluation** - every condition constructs the declared dimension, the target variants are
  the previous study's definitions, and the previous stages' files are never written;
* **leakage** - all sources are FIT-only, PROBE labels only enter the evaluation targets, and
  the official TEST split is never opened;
* **reproducibility** - the same configuration reproduces the same evaluation.

Everything runs on the tiny synthetic fixtures; no dataset or trained checkpoint is needed.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data import make_train_dev_probe_split
from src.evaluation import ActivityAccumulatorResult, collect_activity
from src.model import build_model
from src.neuron_record import TEMPORAL_BLOCK, build_neuron_record_bank
from src.rate_robustness import build_response_targets, main_target_variants
from src.residual import (
    SOURCE_GROUP_FUNCTIONAL,
    SOURCE_GROUP_STRUCTURAL,
    SOURCE_GROUP_TEMPORAL,
    ResidualError,
    ResidualTrainingConfig,
    build_residual_source,
    source_groups_from_names,
    train_residual,
)
from src.source_extension import (
    ABLATION_RESIDUAL_DIM,
    ABLATION_SOURCE_KEYS,
    MASK_BLOCK,
    MASK_COORDINATE,
    RESULT_COLUMNS,
    ROLE_DETERMINISTIC,
    ROLE_MASK_DIAGNOSTIC,
    ROLE_SECONDARY_TARGET,
    ROLE_SOURCE_ABLATION,
    SCHEMA,
    SOURCE_FUNCTIONAL,
    SOURCE_FUNCTIONAL_TEMPORAL,
    SOURCE_STRUCTURAL,
    SOURCE_TEMPORAL,
    SourceCondition,
    SourceExtensionError,
    ablation_table,
    build_payload,
    build_stage_targets,
    condition_matrix,
    condition_table,
    focused_source_conditions,
    run_source_extension,
    secondary_target_variants,
    source_config_for,
    stage_target_variants,
    structured_plus_temporal_dimension,
    summarise_checkpoint_variability,
    summarise_seed_variability,
    write_results,
)
from src.capacity_figures import write_source_extension_figures
from src.vector_capacity import EvaluationSettings, fit_rate_reference
from src.v2_config import DEFAULT_ENABLED_BLOCKS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRUCTURAL_BLOCKS = tuple(DEFAULT_ENABLED_BLOCKS)


# --------------------------------------------------------------------------
# Fixtures (16 hidden neurons, ~60 FIT / ~30 PROBE utterances)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def se_model(tiny_snn_config):
    model = build_model(tiny_snn_config, seed=5)
    with torch.no_grad():
        model.b_hid.copy_(torch.linspace(-0.5, 0.5, tiny_snn_config.n_hidden))
    return model


@pytest.fixture(scope="module")
def se_split(synthetic_rec):
    fit, dev, probe, info = make_train_dev_probe_split(
        synthetic_rec, dev_fraction=0.25, probe_fraction=0.25, seed=0, prefer_speaker_aware=True
    )
    assert len(probe) > 0
    return fit, dev, probe, info


@pytest.fixture(scope="module")
def se_fit_activity(se_model, se_split):
    fit = se_split[0]
    return collect_activity(
        se_model, fit, np.arange(len(fit), dtype=np.int64), device="cpu",
        batch_size=32, n_classes=4, with_labels=False, collect_voltage=False,
    )


@pytest.fixture(scope="module")
def se_probe_activity(se_model, se_split):
    probe = se_split[2]
    return collect_activity(
        se_model, probe, np.arange(len(probe), dtype=np.int64), device="cpu",
        batch_size=32, n_classes=4, with_labels=True, collect_voltage=False,
    )


@pytest.fixture(scope="module")
def se_bank(se_model, se_fit_activity):
    """A bank that carries the coarse temporal block (needed by the temporal conditions)."""
    return build_neuron_record_bank(se_model, activity=se_fit_activity, temporal_resolution=10)


@pytest.fixture(scope="module")
def se_targets(se_probe_activity):
    return build_stage_targets(se_probe_activity, probe_split_label="probe")


@pytest.fixture(scope="module")
def se_settings():
    return EvaluationSettings(
        n_perm=10, bootstrap=0, k_values=(), seed=0, n_splits=2, checkpoint="tiny", tag="test",
    )


@pytest.fixture(scope="module")
def se_conditions(se_bank):
    return focused_source_conditions(se_bank, structured_d=48, residual_dims=(16,), residual_seeds=(0,))


def _train(se_bank, source_key: str, *, residual_d: int = 16, seed: int = 0, mask_mode: str = MASK_COORDINATE,
           epochs: int = 3):
    source_config = source_config_for(
        source_key, enabled_blocks=STRUCTURAL_BLOCKS, functional_source_dim=32
    )
    source = build_residual_source(se_bank, source_config)
    residual = train_residual(
        se_bank,
        config=ResidualTrainingConfig.from_mapping(
            {
                "residual_dim": residual_d, "hidden_dim": 8, "epochs": epochs, "batch_size": 8,
                "seed": seed, "mask_seed": 0, "mask_mode": mask_mode,
            }
        ),
        source=source,
    )
    return source, residual


@pytest.fixture(scope="module")
def se_residuals(se_bank):
    """The frozen artifacts of the four ablation arms plus the block-mask variant (d=16, seed 0)."""
    out = {}
    for key in ABLATION_SOURCE_KEYS:
        source_config = source_config_for(
            key, enabled_blocks=STRUCTURAL_BLOCKS, functional_source_dim=32
        )
        source = build_residual_source(se_bank, source_config)
        residual = train_residual(
            se_bank,
            config=ResidualTrainingConfig.from_mapping(
                {"residual_dim": 16, "hidden_dim": 8, "epochs": 3, "batch_size": 8, "seed": 0}
            ),
            source=source,
        )
        out[f"{key}|{MASK_COORDINATE}|dres16|seed0"] = residual
    out[f"{SOURCE_FUNCTIONAL_TEMPORAL}|{MASK_BLOCK}|dres16|seed0"] = _train(
        se_bank, SOURCE_FUNCTIONAL_TEMPORAL, mask_mode=MASK_BLOCK
    )[1]
    return out


@pytest.fixture(scope="module")
def se_result(se_bank, se_fit_activity, se_targets, se_conditions, se_residuals, se_settings):
    return run_source_extension(
        se_bank, se_targets, se_conditions, settings=se_settings,
        fit_rates=fit_rate_reference(se_fit_activity), residuals=se_residuals,
        checkpoint={"checkpoint": "tiny"},
    )


# --------------------------------------------------------------------------
# Source ablation
# --------------------------------------------------------------------------
def test_source_flags_per_ablation_arm(se_bank):
    expected = {
        SOURCE_STRUCTURAL: (False, False),
        SOURCE_FUNCTIONAL: (True, False),
        SOURCE_TEMPORAL: (False, True),
        SOURCE_FUNCTIONAL_TEMPORAL: (True, True),
    }
    base = build_residual_source(
        se_bank, source_config_for(SOURCE_STRUCTURAL, enabled_blocks=STRUCTURAL_BLOCKS)
    )
    functional = build_residual_source(
        se_bank,
        source_config_for(SOURCE_FUNCTIONAL, enabled_blocks=STRUCTURAL_BLOCKS, functional_source_dim=32),
    )
    temporal = build_residual_source(
        se_bank, source_config_for(SOURCE_TEMPORAL, enabled_blocks=STRUCTURAL_BLOCKS)
    )
    both = build_residual_source(
        se_bank,
        source_config_for(
            SOURCE_FUNCTIONAL_TEMPORAL, enabled_blocks=STRUCTURAL_BLOCKS, functional_source_dim=32
        ),
    )
    assert functional.n_features == base.n_features + 32
    assert temporal.n_features == base.n_features + 10
    assert both.n_features == base.n_features + 42
    assert both.n_features == functional.n_features + temporal.n_features - base.n_features

    for key, (wants_functional, wants_temporal) in expected.items():
        condition = SourceCondition(
            label=f"c_{key}", role=ROLE_SOURCE_ABLATION, structured_d=48,
            residual_d=16, residual_seed=0, source_key=key,
            kind="structured_plus_residual",
        )
        assert condition.source_flags == (wants_functional, wants_temporal)


def test_functional_source_can_be_enabled_independently(se_bank):
    source = build_residual_source(
        se_bank,
        source_config_for(SOURCE_FUNCTIONAL, enabled_blocks=STRUCTURAL_BLOCKS, functional_source_dim=32),
    )
    assert source.provenance["functional_response"]["enabled"] is True
    assert source.provenance["temporal"]["enabled"] is False
    groups = source_groups_from_names(source.feature_names)
    assert SOURCE_GROUP_FUNCTIONAL in groups and SOURCE_GROUP_TEMPORAL not in groups


def test_temporal_source_can_be_enabled_independently(se_bank):
    source = build_residual_source(
        se_bank, source_config_for(SOURCE_TEMPORAL, enabled_blocks=STRUCTURAL_BLOCKS)
    )
    assert source.provenance["temporal"]["enabled"] is True
    assert source.provenance["functional_response"]["enabled"] is False
    groups = source_groups_from_names(source.feature_names)
    assert SOURCE_GROUP_TEMPORAL in groups and SOURCE_GROUP_FUNCTIONAL not in groups


def test_both_sources_can_be_enabled_simultaneously(se_bank):
    source = build_residual_source(
        se_bank,
        source_config_for(
            SOURCE_FUNCTIONAL_TEMPORAL, enabled_blocks=STRUCTURAL_BLOCKS, functional_source_dim=32
        ),
    )
    groups = source_groups_from_names(source.feature_names)
    assert {SOURCE_GROUP_STRUCTURAL, SOURCE_GROUP_FUNCTIONAL, SOURCE_GROUP_TEMPORAL} <= set(groups)
    assert source.provenance["temporal"]["enabled"] is True
    assert source.provenance["functional_response"]["enabled"] is True


def test_temporal_source_requires_the_block(se_model, se_fit_activity):
    plain = build_neuron_record_bank(se_model, activity=se_fit_activity)  # no temporal block
    with pytest.raises(ResidualError, match="temporal"):
        build_residual_source(plain, source_config_for(SOURCE_TEMPORAL, enabled_blocks=STRUCTURAL_BLOCKS))


def test_ablation_conditions_are_complete_and_keyed_uniquely(se_bank):
    conditions = focused_source_conditions(se_bank, structured_d=48, residual_dims=(16,), residual_seeds=(0, 1))
    ablation = [c for c in conditions if c.role == ROLE_SOURCE_ABLATION]
    keys = {c.source_key for c in ablation}
    assert keys == set(ABLATION_SOURCE_KEYS)
    assert len(ablation) == len(ABLATION_SOURCE_KEYS) * 2  # two seeds
    diagnostics = [c for c in conditions if c.role == ROLE_MASK_DIAGNOSTIC]
    assert len(diagnostics) == 2 and all(c.mask_mode == MASK_BLOCK for c in diagnostics)
    assert all(c.source_key == SOURCE_FUNCTIONAL_TEMPORAL for c in diagnostics)

    keys_seen = [c.artifact_key for c in conditions if c.residual_d > 0]
    assert len(keys_seen) == len(set(keys_seen)), "artifact keys must be unique per condition"
    coordinate = next(c for c in ablation if c.source_key == SOURCE_FUNCTIONAL_TEMPORAL)
    assert coordinate.artifact_key != diagnostics[0].artifact_key  # mask mode is part of the key

    large = focused_source_conditions(se_bank, structured_d=48, residual_dims=(52,), residual_seeds=(0,))
    large_keys = {c.source_key for c in large if c.role == ROLE_SOURCE_ABLATION}
    assert large_keys == {SOURCE_STRUCTURAL, SOURCE_FUNCTIONAL, SOURCE_FUNCTIONAL_TEMPORAL}


def test_condition_validation():
    with pytest.raises(SourceExtensionError, match="role"):
        SourceCondition(label="x", role="nope", structured_d=48)
    with pytest.raises(SourceExtensionError, match="source_key"):
        SourceCondition(label="x", role=ROLE_DETERMINISTIC, structured_d=48, source_key="nope")
    with pytest.raises(SourceExtensionError, match="mask_mode"):
        SourceCondition(label="x", role=ROLE_DETERMINISTIC, structured_d=48, mask_mode="nope")
    with pytest.raises(SourceExtensionError, match="seed"):
        SourceCondition(
            label="x", role=ROLE_SOURCE_ABLATION, structured_d=48, residual_d=16,
            residual_seed=None, kind="structured_plus_residual",
        )
    with pytest.raises(SourceExtensionError, match="residual"):
        SourceCondition(label="x", role=ROLE_DETERMINISTIC, structured_d=48, kind="structured_plus_residual")


# --------------------------------------------------------------------------
# Configuration / dimensions
# --------------------------------------------------------------------------
def test_no_ambiguity_between_structured_residual_and_total(se_bank, se_residuals):
    conditions = focused_source_conditions(se_bank, structured_d=48, residual_dims=(16,), residual_seeds=(0,))
    for condition in conditions:
        assert condition.total_d == condition.structured_d + condition.residual_d
        residual = se_residuals.get(condition.artifact_key) if condition.residual_d else None
        X, names = condition_matrix(condition, se_bank, residual=residual)
        assert X.shape[1] == condition.total_d
        assert len(names) == condition.total_d
    # a condition whose declared decomposition disagrees with the artifact it is given is rejected
    mismatched = SourceCondition(
        label="full_48+52_seed0", role=ROLE_SOURCE_ABLATION, structured_d=48,
        residual_d=52, residual_seed=0, kind="structured_plus_residual",
    )
    wrong_artifact = se_residuals["structural|coordinate|dres16|seed0"]  # residual_dim == 16
    with pytest.raises(SourceExtensionError, match="total_d"):
        condition_matrix(mismatched, se_bank, residual=wrong_artifact)


def test_condition_matrix_requires_the_matching_artifact(se_bank):
    condition = SourceCondition(
        label="full_48+16_seed0_plus_functional", role=ROLE_SOURCE_ABLATION, structured_d=48,
        residual_d=16, residual_seed=0, source_key=SOURCE_FUNCTIONAL, kind="structured_plus_residual",
    )
    with pytest.raises(SourceExtensionError, match="frozen residual"):
        condition_matrix(condition, se_bank, residual=None)


def test_temporal_structured_dimension_is_derived_from_the_plan(se_bank):
    assert structured_plus_temporal_dimension(se_bank) == 48 + 10
    conditions = focused_source_conditions(se_bank, structured_d=48, residual_dims=(16,), residual_seeds=(0,))
    temporal = next(c for c in conditions if c.label == "structured_48_plus_temporal")
    assert temporal.total_d == structured_plus_temporal_dimension(se_bank)
    assert temporal.blocks == (*STRUCTURAL_BLOCKS, TEMPORAL_BLOCK)
    X, names = condition_matrix(temporal, se_bank)
    assert X.shape[1] == temporal.total_d
    assert names[-1] == "temporal.bin_09"


def test_source_config_for_rejects_unknown_keys():
    with pytest.raises(SourceExtensionError, match="source_key"):
        source_config_for("nope")


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def test_every_condition_constructs_its_declared_dimension(se_result, se_conditions):
    matrices = se_result["matrices"]
    assert len(matrices) == len(se_conditions)
    for condition in se_conditions:
        rows, columns = matrices[condition.label]
        assert columns == condition.total_d


def test_target_variants_match_the_previous_study(se_targets, se_probe_activity):
    variants = stage_target_variants()
    reference = {v.name: v for v in main_target_variants()}
    assert [v.name for v in variants] == list(reference)
    for variant in variants:
        base = reference[variant.name]
        assert variant.pipeline == base.pipeline
        assert variant.steps == base.steps
        assert variant.bootstrap == base.bootstrap
        assert variant.rate_matched == base.rate_matched

    fresh = build_response_targets(se_probe_activity, probe_split_label="probe")
    for name in reference:
        assert np.array_equal(se_targets.response.get(name).X, fresh.get(name).X)


def test_secondary_targets_are_the_existing_definitions(se_targets):
    secondary = secondary_target_variants()
    assert [v.name for v in secondary] == ["class_rate_20d", "temporal"]
    assert all(v.role == ROLE_SECONDARY_TARGET for v in secondary)
    assert all(not v.bootstrap and not v.predict for v in secondary)
    assert se_targets.get("class_rate_20d").config.feature_sets == ["class_rate"]
    assert "class_psth" in se_targets.get("temporal").config.feature_sets
    assert se_targets.get("class_rate_20d").X_raw.shape == (se_targets.n_neurons, 4)


def test_result_rows_cover_conditions_and_targets(se_result, se_conditions, se_targets):
    rows = se_result["rows"]
    expected_targets = {v.name for v in stage_target_variants()} | {"class_rate_20d", "temporal"}
    per_condition: dict[str, set[str]] = {}
    for row in rows:
        per_condition.setdefault(str(row["representation"]), set()).add(str(row["target_variant"]))
        assert row["probe_n"] == se_targets.n_stimuli
        assert row["total_d"] == row["structured_d"] + (row["residual_d"] or 0)
        assert row["metric"] == "mantel_spearman_r"
        assert row["source_key"] in (SOURCE_STRUCTURAL, SOURCE_FUNCTIONAL, SOURCE_TEMPORAL,
                                     SOURCE_FUNCTIONAL_TEMPORAL)
    assert len(per_condition) == len(se_conditions)
    assert all(targets == expected_targets for targets in per_condition.values())


def test_condition_table_has_every_target_and_uncertainty_column(se_result):
    table = condition_table(se_result["rows"])
    assert table
    for entry in table:
        for column in ("raw_r", "centered_r", "zscored_r", "mean_rate_r", "class_rate_r", "temporal_r",
                       "raw_bootstrap_low", "raw_bootstrap_high", "zscored_bootstrap_low",
                       "zscored_bootstrap_high", "zscored_permutation_p",
                       "source_functional_response", "source_temporal", "mask_mode", "total_d"):
            assert column in entry, column
        if entry["delta_centering"] is not None:
            assert entry["delta_centering"] == pytest.approx(entry["centered_r"] - entry["raw_r"])


def test_ablation_table_reports_differences_against_the_structural_arm(se_result):
    table = ablation_table(se_result["rows"])
    assert table
    baseline = {
        (entry["checkpoint"], entry["residual_seed"]): entry
        for entry in table
        if entry["source_key"] == SOURCE_STRUCTURAL and entry["mask_mode"] == MASK_COORDINATE
    }
    assert baseline, "the structural arm must be present in the ablation table"
    for entry in table:
        base = baseline.get((entry["checkpoint"], entry["residual_seed"]))
        assert base is not None
        for metric, column in (("raw_r", "delta_vs_structural_raw"),
                               ("zscored_r", "delta_vs_structural_zscored")):
            if entry[metric] is None or base[metric] is None:
                assert entry[column] is None
            else:
                assert entry[column] == pytest.approx(entry[metric] - base[metric])


def test_summaries_keep_individual_values(se_result):
    seeds = summarise_seed_variability(se_result["rows"])
    assert seeds
    for entry in seeds:
        assert entry["n_seeds"] >= 1
        assert len(entry["raw_r_values"]) == entry["n_seeds"]
    checkpoints = summarise_checkpoint_variability(se_result["rows"])
    assert any(row["representation"] == "structured_48" for row in checkpoints)


def test_write_results_uses_dedicated_files_only(se_result, tmp_path):
    payload = build_payload(se_result)
    written = write_results(se_result, csv_path=tmp_path / "source_extension_results.csv",
                            json_path=tmp_path / "source_extension_results.json")
    assert sorted(p.name for p in Path(tmp_path).iterdir()) == [
        "source_extension_results.csv", "source_extension_results.json"
    ]
    header = written["csv"].read_text(encoding="utf-8").splitlines()[0].split(",")
    for column in ("representation", "total_d", "structured_d", "residual_d", "source_key",
                   "source_functional_response", "source_temporal", "mask_mode", "value",
                   "bootstrap_low", "permutation_p"):
        assert column in header
        assert column in RESULT_COLUMNS
    stored = json.loads(written["json"].read_text(encoding="utf-8"))
    assert stored["schema"] == SCHEMA
    for key in ("condition_table", "ablation_table", "summary_seed_variability",
                "summary_checkpoint_variability", "result_columns"):
        assert key in stored
    assert payload["schema"] == stored["schema"]


def test_figures_render(se_result, tmp_path):
    payload = build_payload(se_result)
    written = write_source_extension_figures(payload, tmp_path)
    assert set(written) == {
        "source_extension_figure1_target_decomposition",
        "source_extension_figure2_source_ablation",
        "source_extension_figure3_raw_vs_shape",
    }
    for name, paths in written.items():
        assert paths, f"{name} was not rendered"
        for path in paths:
            assert Path(path).exists()


# --------------------------------------------------------------------------
# Leakage
# --------------------------------------------------------------------------
def test_all_sources_are_fit_only(se_bank):
    for key in ABLATION_SOURCE_KEYS:
        source = build_residual_source(
            se_bank, source_config_for(key, enabled_blocks=STRUCTURAL_BLOCKS, functional_source_dim=32)
        )
        assert source.provenance["uses_labels"] is False
        assert source.provenance["fit_only"] is True
        assert "probe" not in str(source.provenance["bank"]["activity_split"]).lower()
        functional = source.provenance["functional_response"]
        if functional.get("enabled"):
            assert functional["uses_labels"] is False
            assert "probe" not in str(functional.get("source_split", "")).lower()
        temporal = source.provenance["temporal"]
        if temporal.get("enabled"):
            assert temporal["uses_labels"] is False
            assert bool(temporal.get("class_conditioned")) is False


def test_probe_split_banks_are_rejected(se_model, se_split):
    probe_bank = build_neuron_record_bank(
        se_model, fit_rec=se_split[2], device="cpu", batch_size=32, temporal_resolution=10
    )
    with pytest.raises(ResidualError, match="FIT only"):
        build_residual_source(
            probe_bank, source_config_for(SOURCE_FUNCTIONAL, enabled_blocks=STRUCTURAL_BLOCKS)
        )


def test_targets_reject_non_probe_splits(se_probe_activity):
    with pytest.raises(ValueError, match="held-out PROBE"):
        build_stage_targets(se_probe_activity, probe_split_label="fit")
    with pytest.raises(ValueError, match="held-out PROBE"):
        build_stage_targets(se_probe_activity, probe_split_label="train")


def test_script_never_opens_the_official_test_split():
    source = (PROJECT_ROOT / "scripts" / "evaluate_vector_source_extension.py").read_text(encoding="utf-8")
    assert "shd_test" not in source
    assert '"official_test_loaded": False' in source


def test_evaluation_apis_have_no_labels_or_test_inputs():
    for func in (run_source_extension, condition_matrix, build_stage_targets):
        names = set(inspect.signature(func).parameters)
        assert not (names & {"labels", "y", "test", "probe_labels"}), names
    for func in (build_residual_source, train_residual):
        assert "labels" not in set(inspect.signature(func).parameters)


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def test_same_configuration_reproduces_the_evaluation(se_bank, se_fit_activity, se_targets,
                                                      se_conditions, se_residuals, se_settings):
    subset = [c for c in se_conditions if c.residual_d == 0][:1]
    first = run_source_extension(
        se_bank, se_targets, subset, settings=se_settings,
        fit_rates=fit_rate_reference(se_fit_activity), residuals=se_residuals,
    )
    second = run_source_extension(
        se_bank, se_targets, subset, settings=se_settings,
        fit_rates=fit_rate_reference(se_fit_activity), residuals=se_residuals,
    )
    assert [row["value"] for row in first["rows"]] == [row["value"] for row in second["rows"]]
    assert first["rows"][0]["representation"] == second["rows"][0]["representation"]


def test_condition_matrices_are_independent_of_the_targets(se_bank, se_probe_activity, se_residuals):
    condition = next(
        c for c in focused_source_conditions(se_bank, structured_d=48, residual_dims=(16,), residual_seeds=(0,))
        if c.role == ROLE_SOURCE_ABLATION and c.source_key == SOURCE_FUNCTIONAL
    )
    residual = se_residuals[condition.artifact_key]
    before, _ = condition_matrix(condition, se_bank, residual=residual)
    build_stage_targets(se_probe_activity, probe_split_label="probe")
    after, _ = condition_matrix(condition, se_bank, residual=residual)
    assert np.array_equal(before, after)


def test_residual_seed_is_part_of_the_condition_identity(se_bank):
    conditions = focused_source_conditions(se_bank, structured_d=48, residual_dims=(16,), residual_seeds=(0, 1, 2))
    seeds = sorted(
        c.residual_seed for c in conditions
        if c.role == ROLE_SOURCE_ABLATION and c.source_key == SOURCE_STRUCTURAL
    )
    assert seeds == [0, 1, 2]
    keys = [c.artifact_key for c in conditions if c.residual_d]
    assert len(keys) == len(set(keys))