"""Focused tests for the persistent V3 run registry."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from v3.config import StateRegularizationConfig, TimingConfig, V3Config
from v3.registry import RUN_FIELDS, RunRegistry, register_run, run_row_from_variant, split_accuracy


def _row(**over):
    row = {k: "" for k in RUN_FIELDS}
    row.update(
        tag="v3_test",
        N=64,
        D=1000,
        parameters=2025440,
        epochs=5,
        seed=0,
        batch_size=128,
        learning_rate=1e-3,
        optimizer="adam",
        scheduler="cosine",
        training_precision="float16",
        state_regularization_mode="none",
        state_noise_enabled=False,
        state_noise_std=0.0,
        state_quantization_enabled=False,
        state_quantization_bits=8,
        train_accuracy=0.9,
        val_accuracy=0.8,
        test_accuracy=0.7981,
        test_correct=1807,
        test_total=2264,
    )
    row.update(over)
    return row


def test_one_completed_run_creates_exactly_one_record(tmp_path):
    reg = RunRegistry(tmp_path)
    reg.record(_row())
    runs = reg.read_runs()
    assert len(runs) == 1
    assert list(tmp_path.glob("*/")) != []


def test_timestamp_present_and_well_formed(tmp_path):
    reg = RunRegistry(tmp_path)
    run_dir = reg.record(_row())
    runs = reg.read_runs()
    assert runs[0]["run_id"] == run_dir.name
    assert runs[0]["timestamp"]  # non-empty
    # matches the local-time format 2026-10-02_00-15-30
    datetime.strptime(run_dir.name, "%Y-%m-%d_%H-%M-%S")


def test_run_directory_contains_artifacts(tmp_path):
    reg = RunRegistry(tmp_path)
    cm = np.eye(20, dtype=int) * 3
    run_dir = reg.record(
        _row(),
        config_yaml="N: 64\n",
        metrics={"test_accuracy": 0.7981},
        confusion_matrix=cm,
        summary="hello",
    )
    assert (run_dir / "config.yaml").read_text() == "N: 64\n"
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "confusion_matrix.csv").exists()
    assert (run_dir / "summary.txt").read_text() == "hello"


def test_records_are_appended_not_overwritten(tmp_path):
    reg = RunRegistry(tmp_path)
    reg.record(_row(notes="first"))
    reg.record(_row(notes="second"))
    runs = reg.read_runs()
    assert len(runs) == 2
    assert runs[0]["notes"] == "first"
    assert runs[1]["notes"] == "second"


def test_config_values_stored_correctly(tmp_path):
    reg = RunRegistry(tmp_path)
    reg.record(_row(state_regularization_mode="noise", state_noise_enabled=True, state_noise_std=0.02))
    runs = reg.read_runs()
    assert runs[0]["state_regularization_mode"] == "noise"
    assert runs[0]["state_noise_std"] == "0.02"
    assert runs[0]["state_noise_enabled"] == "True"


def test_same_timestamp_gets_unique_run_ids(tmp_path):
    reg = RunRegistry(tmp_path)
    when = datetime(2026, 10, 2, 0, 15, 30)
    d1 = reg.record(_row(), timestamp=when)
    d2 = reg.record(_row(), timestamp=when)
    assert d1 != d2
    assert d1.name == "2026-10-02_00-15-30"
    assert d2.name == "2026-10-02_00-15-30_02"


def test_split_accuracy_english_german():
    cm = np.zeros((20, 20), dtype=int)
    for i in range(10):          # english: 8/10 correct per class
        cm[i, i] = 8
        cm[i, (i + 1) % 10] = 2
    for i in range(10, 20):      # german: 5/10 correct per class
        cm[i, i] = 5
        cm[i, 10 + (i + 1) % 10] = 5
    english, german = split_accuracy(cm)
    assert english == pytest.approx(0.8)
    assert german == pytest.approx(0.5)


def test_row_from_variant_extracts_config_and_accuracies():
    cfg = V3Config(
        tag="v3_x",
        n_neurons=64,
        state_dim=1000,
        epochs=5,
        state_regularization=StateRegularizationConfig(mode="noise", noise_std=0.01),
    )
    cm = np.eye(20, dtype=int) * 2
    test_metrics = {
        "test_accuracy": 0.75,
        "test_correct": 15,
        "test_total": 20,
        "confusion_matrix": cm,
    }
    row = run_row_from_variant(
        cfg=cfg,
        n_parameters=2025440,
        fit_metrics={"accuracy": 0.9},
        val_metrics={"accuracy": 0.85},
        test_metrics=test_metrics,
        train_seconds=100.0,
        duration_seconds=110.0,
        checkpoint="V3/checkpoints/v3_x.pt",
    )
    assert row["N"] == 64 and row["D"] == 1000
    assert row["state_regularization_mode"] == "noise"
    assert row["state_noise_enabled"] is True
    assert row["test_accuracy"] == pytest.approx(0.75)
    assert row["english_accuracy"] == pytest.approx(1.0)
    assert row["german_accuracy"] == pytest.approx(1.0)
    assert set(RUN_FIELDS).issubset(set(row))


def test_timing_and_shuffle_fields_are_recorded():
    cfg = V3Config(
        tag="v3_2ms", n_neurons=64, state_dim=1000,
        timing=TimingConfig(sequence_duration_ms=1000.0, time_bin_ms=2.0),
    )
    row = run_row_from_variant(cfg=cfg, n_parameters=1)
    assert row["sequence_duration_ms"] == 1000.0
    assert row["time_bin_ms"] == 2.0
    assert row["num_time_steps"] == 500
    assert row["simulation_dt_ms"] == 2.0
    assert row["shuffle_train"] is True
    assert row["shuffle_val"] is False


def test_default_row_records_the_4ms_reference():
    row = run_row_from_variant(cfg=V3Config(), n_parameters=1)
    assert row["sequence_duration_ms"] == 1000.0
    assert row["time_bin_ms"] == 4.0
    assert row["num_time_steps"] == 250
    assert row["simulation_dt_ms"] == 4.0


def test_header_migration_preserves_old_rows(tmp_path):
    """An existing runs.csv with an older header must not be corrupted."""
    (tmp_path / "runs.csv").write_text("run_id,notes\nold_row,legacy\n", encoding="utf-8")
    reg = RunRegistry(tmp_path)
    reg.record(_row(notes="new"))
    runs = reg.read_runs()
    assert len(runs) == 2
    assert runs[0]["run_id"] == "old_row" and runs[0]["notes"] == "legacy"
    assert runs[1]["notes"] == "new"
    assert "num_time_steps" in runs[1] and "sequence_duration_ms" in runs[1]


def test_register_run_is_the_shared_entry_point(tmp_path):
    """The notebook and the CLI both call this; it writes one row + artifacts."""
    cm = np.eye(20, dtype=int) * 2
    test_metrics = {"test_accuracy": 0.8, "test_correct": 16, "test_total": 20,
                    "confusion_matrix": cm}
    cfg = V3Config(tag="v3_reg", epochs=5)
    reg = RunRegistry(tmp_path)
    run_dir = register_run(
        reg, cfg,
        n_parameters=2025440,
        fit_metrics={"accuracy": 0.9},
        val_metrics={"accuracy": 0.85},
        test_metrics=test_metrics,
        train_seconds=100.0,
        duration_seconds=110.0,
        checkpoint="x.pt",
        best_val_accuracy=0.85,
        best_epoch=3,
        notes="notebook",
        extra_metrics={"vector": {"fit": {"accuracy": 0.9}}},
        history=[{"epoch": 1, "loss": 1.0}],
    )
    runs = reg.read_runs()
    assert len(runs) == 1
    assert runs[0]["test_correct"] == "16"
    assert runs[0]["num_time_steps"] == "250"
    assert runs[0]["notes"] == "notebook"
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "confusion_matrix.csv").exists()
    assert (run_dir / "summary.txt").exists()
