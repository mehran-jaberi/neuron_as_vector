"""Tests for the numerical-health diagnostics (pure measurement helpers)."""

from __future__ import annotations

import numpy as np

from src.diagnostics import (
    build_speaker_table,
    class_input_stats,
    evaluation_regimes,
    histogram,
    measure_rates,
    summarize,
)


def test_summarize_reports_percentiles_and_ignores_non_finite():
    v = np.array([1.0, 2.0, 3.0, 4.0, np.nan, np.inf])
    s = summarize(v)
    assert s["n"] == 4
    assert s["mean"] == 2.5
    assert s["p0"] == 1.0 and s["p100"] == 4.0
    assert "p50" in s


def test_measure_rates_converts_spikes_per_step_to_hz():
    # 0.25 spikes per 2 ms timestep = 0.25 / 0.002 = 125 Hz
    rates = measure_rates(np.array([0.0, 0.25, 0.5]), bin_ms=2.0)
    assert rates["mean_hz"] == (0.0 + 0.25 + 0.5) / 3 / 0.002
    assert rates["max_hz"] == 0.5 / 0.002
    assert rates["spikes_per_timestep"]["mean"] == (0.0 + 0.25 + 0.5) / 3
    # fraction bookkeeping
    r = measure_rates(np.array([0.0, 0.0005, 0.5, 0.9]), bin_ms=2.0)
    assert r["fraction_below_1hz"] == 0.5  # 0 and 0.0005 (0.25 Hz)
    assert r["fraction_above_200hz"] == 0.5  # 0.5 and 0.9 -> 250 and 450 Hz


def test_histogram_bounds_and_overflow_counts():
    v = np.array([-100.0, 0.0, 0.5, 1.0, 100.0])
    h = histogram(v, lo=0.0, hi=1.0, bins=2)
    assert h["n_below_range"] == 1 and h["n_above_range"] == 1
    assert sum(h["counts"]) == 3


def test_class_input_stats_on_synthetic(synthetic_rec):
    idx = np.arange(len(synthetic_rec))
    stats = class_input_stats(synthetic_rec, idx, n_classes=4)
    assert set(stats) == {"0", "1", "2", "3"}
    total = sum(s["n_samples"] for s in stats.values())
    assert total == len(synthetic_rec)
    for s in stats.values():
        assert s["mean_events"] > 0
        assert np.isfinite(s["mean_span_ms"])


def test_build_speaker_table_and_regimes():
    class _Rec:
        def __init__(self, speakers):
            self.speakers = np.asarray(speakers)

    train = _Rec([0, 0, 1, 1, 2, 2, 3, 3])
    test = _Rec([0, 1, 2, 3, 4, 4, 5, 5, 5])
    split_info = {
        "splits": {
            "train": {"speakers": [0, 1]},
            "dev": {"speakers": [2]},
            "probe": {"speakers": [3]},
        }
    }
    table = build_speaker_table(train, test, split_info)
    assert table["fit_speakers"] == [0, 1]
    assert table["dev_speakers"] == [2]
    assert table["probe_speakers"] == [3]
    roles = {row["speaker"]: row["roles"] for row in table["rows"]}
    assert "fit" in roles[0]
    assert "dev" in roles[2]
    assert "probe" in roles[3]
    assert "novel_not_in_train_file" in roles[4]
    # test speakers 4,5 are not in the training file
    assert 4 in table["test_speakers_present"] and 5 in table["test_speakers_present"]

    regimes = evaluation_regimes(test, table["fit_speakers"])
    assert regimes["A_seen_speaker"]["n_samples"] == 2   # speakers 0,1
    assert regimes["B_held_out_speaker"]["n_samples"] == 7
    assert regimes["C_official_test"]["n_samples"] == 9
    assert (regimes["A_seen_speaker"]["mask"] & regimes["B_held_out_speaker"]["mask"]).sum() == 0