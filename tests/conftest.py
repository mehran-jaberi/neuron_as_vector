"""Shared pytest fixtures.

Everything the tests need is generated *synthetically* on the fly: no dataset is
downloaded, no checkpoint from a real run is required, and the networks are tiny.
This keeps the suite fast enough to run on every change while still exercising the
full numerical pipeline (binning -> LIF forward -> surrogate backward ->
representation extraction -> fingerprint -> geometry -> controls).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data import SHDRecordings, make_synthetic_shd
from src.model import SNNConfig, build_model


@pytest.fixture(scope="session")
def synthetic_rec() -> SHDRecordings:
    """A small, learnable spiking dataset with speaker metadata."""
    return make_synthetic_shd(
        n_samples=120,
        n_classes=4,
        n_channels=20,
        n_bins=30,
        bin_ms=2.0,
        n_speakers=4,
        seed=0,
        spikes_per_sample=80,
    )


@pytest.fixture(scope="session")
def tiny_snn_config() -> SNNConfig:
    """A deliberately small architecture matched to the synthetic dataset."""
    return SNNConfig(
        n_input=20,
        n_hidden=16,
        n_output=4,
        n_bins=30,
        bin_ms=2.0,
        neuron_param_mode="bias",
    )


@pytest.fixture(scope="session")
def tiny_model(tiny_snn_config: SNNConfig):
    return build_model(tiny_snn_config, seed=0)


@pytest.fixture(scope="session")
def tiny_input(tiny_snn_config: SNNConfig) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.random((6, tiny_snn_config.n_bins, tiny_snn_config.n_input)).astype(np.float32)
