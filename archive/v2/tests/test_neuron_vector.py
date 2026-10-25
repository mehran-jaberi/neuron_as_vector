"""Tests for the full neuron-vector composition (``src/neuron_vector.py``).

Coverage: the ``[z_structured, z_residual]`` composition invariant, dimension examples
(48, 64, 100), coordinate ordering and provenance, backward compatibility at
``learned_residual_d = 0``, schema/dimension mismatch rejection, determinism and memory.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.model import SNNConfig, build_model
from src.neuron_record import build_neuron_record_bank
from src.neuron_vector import (
    SCHEMA,
    NeuronVectorError,
    NeuronVectors,
    build_neuron_vectors,
    neuron_vectors_from_config,
)
from src.neurons import extract_structural_representations
from src.residual import ResidualResult, ResidualTrainingConfig, train_residual
from src.structured_vector import StructuredVectorEncoder
from src.utils import Config
from src.v2_config import V2Config

STRUCTURED_BLOCKS = ("intrinsic", "input_conn", "recurrent_in", "recurrent_out")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _tiny_model(seed: int = 1):
    cfg = SNNConfig(n_input=20, n_hidden=16, n_output=4, n_bins=30, bin_ms=2.0)
    model = build_model(cfg, seed=seed)
    with torch.no_grad():
        model.b_hid.copy_(torch.linspace(-0.5, 0.5, cfg.n_hidden))
    return model


def _bank(seed: int = 1):
    return build_neuron_record_bank(_tiny_model(seed), with_activity=False)


def _tiny_tau_model():
    """A model whose intrinsic block carries an extra dynamical feature (tau_mem_ms)."""
    cfg = SNNConfig(n_input=20, n_hidden=16, n_output=4, n_bins=30, bin_ms=2.0,
                    neuron_param_mode="bias_tau")
    model = build_model(cfg, seed=2)
    with torch.no_grad():
        model.b_hid.copy_(torch.linspace(-0.5, 0.5, cfg.n_hidden))
        model.log_tau_offset.copy_(torch.linspace(-0.3, 0.3, cfg.n_hidden))
    return model


def _residual(bank, dimension: int, *, seed: int = 0, source_config=None) -> ResidualResult:
    config = ResidualTrainingConfig.from_mapping(
        {"residual_dim": dimension, "hidden_dim": 16, "epochs": 10, "batch_size": 8, "seed": seed}
    )
    return train_residual(bank, config=config, source_config=source_config)


def _config(**vector) -> V2Config:
    return V2Config.from_config(Config({"vector": vector}), warn=False)


# --------------------------------------------------------------------------
# Backward compatibility: residual_d = 0
# --------------------------------------------------------------------------
def test_48_plus_0_reduces_to_the_structured_vector_exactly():
    bank = _bank()
    structured = StructuredVectorEncoder(bank, structured_d=48).encode()
    vectors = build_neuron_vectors(bank, structured_d=48)
    assert vectors.shape == (bank.n_neurons, 48)
    assert vectors.residual_d == 0 and vectors.residual_names == ()
    assert vectors.structured_names == structured.feature_names
    assert np.array_equal(vectors.X, structured.X)  # bit-identical, no extra model
    assert vectors.provenance["residual"]["used"] is False
    assert "learned_residual_d = 0" in vectors.provenance["residual"]["reason"]


def test_48_plus_0_matches_the_historical_48d_representation():
    model = _tiny_model()
    bank = build_neuron_record_bank(model, with_activity=False)
    X_old, names_old = extract_structural_representations(model).to_matrix()
    vectors = build_neuron_vectors(bank, structured_d=48)
    assert list(vectors.feature_names) == list(names_old)
    assert np.array_equal(vectors.X, X_old)


def test_from_config_zero_residual_needs_no_model_and_rejects_one():
    bank = _bank()
    config = _config(d=48, structured_d=48, learned_residual_d=0, residual={"enabled": False})
    vectors = neuron_vectors_from_config(config, bank)
    assert vectors.shape == (bank.n_neurons, 48)
    assert vectors.provenance["configured"]["invariant_holds"] is True
    assert vectors.provenance["configured"]["residual_enabled"] is False

    with pytest.raises(NeuronVectorError, match="no residual"):
        neuron_vectors_from_config(config, bank, residual=_residual(bank, 8))


# --------------------------------------------------------------------------
# Dimension examples (48, 64, 100)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("residual_dim,expected", [(0, 48), (16, 64), (52, 100)])
def test_dimension_examples(residual_dim, expected):
    bank = _bank()
    residual = None if residual_dim == 0 else _residual(bank, residual_dim)
    vectors = build_neuron_vectors(bank, structured_d=48, residual=residual)
    assert vectors.shape == (bank.n_neurons, expected)
    assert vectors.full_dimension == expected == vectors.structured_d + vectors.residual_d
    assert vectors.structured_d == 48
    assert vectors.residual_d == residual_dim


def test_coordinate_order_and_names():
    bank = _bank()
    residual = _residual(bank, 16)
    vectors = build_neuron_vectors(bank, structured_d=48, residual=residual)
    assert vectors.feature_names == vectors.structured_names + vectors.residual_names
    assert vectors.residual_names == tuple(f"residual[{j}]" for j in range(16))
    assert vectors.feature_names[0] == "intrinsic.learned_bias"
    assert vectors.feature_names[-1] == "residual[15]"
    # values are exactly the two parts
    structured = StructuredVectorEncoder(bank, structured_d=48).encode()
    assert np.array_equal(vectors.X[:, :48], structured.X)
    assert np.array_equal(vectors.X[:, 48:], residual.encode(bank))


def test_from_config_with_matching_residual():
    bank = _bank()
    residual = _residual(bank, 16)
    config = _config(d=64, structured_d=48, learned_residual_d=16, residual={"enabled": True})
    vectors = neuron_vectors_from_config(config, bank, residual=residual)
    assert vectors.shape == (bank.n_neurons, 64)
    configured = vectors.provenance["configured"]
    assert configured["d"] == 64 and configured["learned_residual_d"] == 16
    assert configured["invariant_holds"] is True
    assert configured["representation_chunk_size"] == config.memory.representation_chunk_size


def test_from_config_requires_and_validates_the_residual():
    bank = _bank()
    config = _config(d=64, structured_d=48, learned_residual_d=16, residual={"enabled": True})
    with pytest.raises(NeuronVectorError, match="no trained residual"):
        neuron_vectors_from_config(config, bank)

    wrong_dimension = _residual(bank, 8)
    with pytest.raises(NeuronVectorError, match="learned_residual_d=16"):
        neuron_vectors_from_config(config, bank, residual=wrong_dimension)


# --------------------------------------------------------------------------
# Schema and input validation
# --------------------------------------------------------------------------
def test_schema_mismatch_between_residual_and_bank_is_rejected(tiny_model, synthetic_rec):
    from src.residual import ResidualSourceConfig

    bank_plain = build_neuron_record_bank(tiny_model, with_activity=False)

    # (a) a residual trained on a source view that includes the activity block ...
    bank_with_activity = build_neuron_record_bank(
        tiny_model, fit_rec=synthetic_rec, device="cpu", batch_size=32
    )
    residual_with_activity = _residual(
        bank_with_activity, 8,
        source_config=ResidualSourceConfig(
            enabled_blocks=["intrinsic", "input_conn", "recurrent_in", "recurrent_out", "activity"]
        ),
    )
    with pytest.raises(NeuronVectorError, match="incompatible residual source"):
        build_neuron_vectors(bank_plain, structured_d=48, residual=residual_with_activity)

    # (b) ... or on a bias_tau model whose intrinsic block has an extra dynamical feature
    bank_tau = build_neuron_record_bank(_tiny_tau_model(), with_activity=False)
    residual_tau = _residual(bank_tau, 8)
    with pytest.raises(NeuronVectorError, match="incompatible residual source"):
        build_neuron_vectors(bank_plain, structured_d=48, residual=residual_tau)


def test_composition_rejects_wrong_input_types():
    bank = _bank()
    with pytest.raises(NeuronVectorError, match="NeuronRecordBank"):
        build_neuron_vectors(np.zeros((4, 4)), structured_d=8)  # type: ignore[arg-type]
    with pytest.raises(NeuronVectorError, match="ResidualResult"):
        build_neuron_vectors(bank, structured_d=48, residual="not-a-residual")  # type: ignore[arg-type]


def test_from_config_rejects_an_inconsistent_vector_configuration():
    # the V2 config layer already rejects this, so build the mapping check through it
    with pytest.raises(ValueError):
        _config(d=64, structured_d=48, learned_residual_d=8)


# --------------------------------------------------------------------------
# Provenance / memory / determinism
# --------------------------------------------------------------------------
def test_provenance_is_complete_and_serialisable():
    bank = _bank()
    residual = _residual(bank, 16)
    vectors = build_neuron_vectors(bank, structured_d=48, residual=residual)
    provenance = vectors.provenance
    assert provenance["schema"] == SCHEMA
    assert provenance["uses_labels"] is False
    assert provenance["full_dimension"] == 64
    assert provenance["structured_d"] == 48 and provenance["residual_d"] == 16
    assert "structured coordinates first" in provenance["coordinate_order"]
    assert provenance["structured"]["source"].endswith("StructuredVectorEncoder")
    assert provenance["residual"]["used"] is True
    assert provenance["residual"]["uses_labels"] is False
    assert provenance["residual"]["residual_dim"] == 16
    assert "residual_provenance" in provenance["residual"]
    assert provenance["bank"]["uses_labels"] is False
    payload = vectors.to_dict()
    assert payload["structured_names"] == list(vectors.structured_names)
    assert payload["residual_names"] == list(vectors.residual_names)
    assert isinstance(json.dumps(payload), str)


def test_memory_and_shape_guards():
    bank = _bank()
    residual = _residual(bank, 16)
    vectors = build_neuron_vectors(bank, structured_d=48, residual=residual)
    assert vectors.X.ndim == 2
    assert vectors.X.shape == (bank.n_neurons, 64)
    assert vectors.n_neurons == bank.n_neurons
    # a source/vector pair must never be (neurons, samples, d)
    assert vectors.X.shape[0] == bank.n_neurons and vectors.X.shape[1] == 64
    assert np.asarray(residual.encode(bank)).ndim == 2


def test_composition_is_deterministic_and_chunking_agnostic():
    bank = _bank()
    residual = _residual(bank, 16)
    a = build_neuron_vectors(bank, structured_d=48, residual=residual)
    b = build_neuron_vectors(bank, structured_d=48, residual=residual)
    assert np.array_equal(a.X, b.X)
    assert a.feature_names == b.feature_names
    chunked = build_neuron_vectors(bank, structured_d=48, residual=residual, chunk_size=3)
    assert np.array_equal(chunked.X, a.X)


def test_neuron_vectors_are_immutable_in_shape_and_names():
    with pytest.raises(NeuronVectorError, match="composition mismatch"):
        NeuronVectors(X=np.zeros((4, 3)), structured_names=("a",), residual_names=("b",))
    with pytest.raises(NeuronVectorError, match="2-D"):
        NeuronVectors(X=np.zeros((4, 3, 2)), structured_names=("a",), residual_names=("b", "c"))