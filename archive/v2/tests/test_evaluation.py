"""Tests for evaluation diagnostics and training-loop health logging."""

from __future__ import annotations

import numpy as np
import torch

from src.data import iterate_batches
from src.evaluation import ActivityAccumulator, circuit_health
from src.model import build_model
from src.training import TrainConfig, train_model


def test_accumulator_tracks_membrane_stats(tiny_snn_config):
    acc = ActivityAccumulator(
        n_hidden=tiny_snn_config.n_hidden,
        n_bins=tiny_snn_config.n_bins,
        n_classes=tiny_snn_config.n_output,
        bin_ms=tiny_snn_config.bin_ms,
        collect_voltage=True,
    )
    B, T, H = 4, tiny_snn_config.n_bins, tiny_snn_config.n_hidden
    spikes = np.zeros((B, T, H))
    v = np.random.default_rng(0).random((B, T, H))
    acc.update(spikes, labels=None, hidden_v=v)
    res = acc.finalize()
    assert np.isclose(res.v_global_mean, float(v.mean()))
    assert np.isclose(res.v_global_max, float(v.max()))
    assert np.isclose(res.v_global_min, float(v.min()))
    assert res.v_global_std > 0.0


def test_circuit_health_reports_all_diagnostics(tiny_model, synthetic_rec, tiny_snn_config):
    idx = np.arange(min(16, len(synthetic_rec)))
    health = circuit_health(
        tiny_model, synthetic_rec, idx, batch_size=8,
        n_classes=tiny_snn_config.n_output, device=torch.device("cpu"),
    )
    for key in (
        "rate_hz_mean",
        "rate_hz_percentiles",
        "silent_neuron_fraction",
        "total_spikes",
        "mean_total_spikes_per_sample",
        "v_global_mean",
        "v_global_std",
        "v_global_min",
        "v_global_max",
        "rate_hz_per_neuron",
    ):
        assert key in health, key
    assert len(health["rate_hz_per_neuron"]) == tiny_snn_config.n_hidden
    p = health["rate_hz_percentiles"]
    assert p["p0"] <= p["p50"] <= p["p100"]
    assert health["v_global_max"] >= health["v_global_min"]


def test_training_history_logs_grad_norm_and_dev_metrics(tiny_snn_config, synthetic_rec):
    from src.data import make_train_dev_probe_split

    train, dev, probe, _ = make_train_dev_probe_split(
        synthetic_rec, dev_fraction=0.25, probe_fraction=0.25, seed=0
    )
    model = build_model(tiny_snn_config, seed=0)
    tcfg = TrainConfig(epochs=1, batch_size=16, eval_batch_size=16,
                       n_classes=tiny_snn_config.n_output, scheduler="none")
    result = train_model(
        model, train, np.arange(len(train)), dev, np.arange(len(dev)),
        tcfg, device=torch.device("cpu"), seed=0, verbose=False,
    )
    assert len(result.history) == 1
    rec = result.history[0]
    for key in ("train_accuracy", "dev_accuracy", "dev_loss", "grad_norm_mean",
                "grad_norm_max", "epoch_time_s"):
        assert key in rec, key
    assert np.isfinite(rec["grad_norm_mean"])
    assert rec["epoch_time_s"] is not None and rec["epoch_time_s"] >= 0.0
    assert result.n_dev == len(dev)
