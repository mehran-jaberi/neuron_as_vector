"""Tests for the label-free functional-response and temporal sources.

Coverage:

* **functional-response source** - exact shape, deterministic/prefix-stable/seed-dependent
  projection, sample-order hash, the three normalisation modes, the documented
  zero-variance rule and label-free provenance;
* **temporal block** - exact bin boundaries, deterministic values on a toy activity tensor,
  resolution-driven dimension, feature names and label-free provenance;
* **structured encoder** - the historical 48-D default stays exact, and the temporal block
  appears only when it is requested (and present);
* **residual** - input dimension adapts to the new sources, the schema is persisted and
  mismatches are rejected, masks stay deterministic, per-source diagnostics are recorded,
  and the SNN is never modified;
* **full vector** - ``48+0 / 48+16 / 48+52`` and a temporal structured configuration.

Everything runs on the tiny synthetic fixtures; no dataset, checkpoint or PROBE data is used.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.evaluation import ActivityAccumulatorResult, collect_activity
from src.functional_response import (
    FUNCTIONAL_RESPONSE_SCHEMA,
    FunctionalResponseConfig,
    FunctionalResponseError,
    FunctionalResponseSource,
    build_functional_response_source,
    project_response_profile,
    projection_matrix,
    response_counts_from_bank,
    response_statistics,
    sample_order_hash,
)
from src.data import make_train_dev_probe_split
from src.neuron_record import (
    TEMPORAL_BLOCK,
    build_neuron_record_bank,
    assert_label_free_fit_activity,
    temporal_bin_edges,
)
from src.neuron_vector import build_neuron_vectors, neuron_vectors_from_config
from src.residual import (
    SOURCE_GROUP_FUNCTIONAL,
    SOURCE_GROUP_STRUCTURAL,
    SOURCE_GROUP_TEMPORAL,
    ResidualError,
    ResidualSourceConfig,
    ResidualTrainingConfig,
    build_residual_source,
    make_block_mask,
    make_source_mask,
    source_group_of_feature,
    source_groups_from_names,
    train_residual,
)
from src.structured_vector import StructuredVectorEncoder
from src.utils import Config, load_config
from src.v2_config import (
    DEFAULT_FUNCTIONAL_SOURCE_DIM,
    FUNCTIONAL_SOURCE_NORMALIZATIONS,
    V2Config,
    V2ConfigError,
    VectorConfig,
    VectorResidualConfig,
)

STRUCTURAL_BLOCKS = ("intrinsic", "input_conn", "recurrent_in", "recurrent_out")


# --------------------------------------------------------------------------
# Fixtures (16 hidden neurons, ~60 FIT utterances, toy PSTH available)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fr_model(tiny_snn_config):
    from src.model import build_model

    model = build_model(tiny_snn_config, seed=3)
    with torch.no_grad():
        model.b_hid.copy_(torch.linspace(-0.5, 0.5, tiny_snn_config.n_hidden))
    return model


@pytest.fixture(scope="module")
def fr_activity(fr_model, synthetic_rec):
    fit, dev, probe, info = make_train_dev_probe_split(
        synthetic_rec, dev_fraction=0.25, probe_fraction=0.25, seed=0, prefer_speaker_aware=True
    )
    return collect_activity(
        fr_model, fit, np.arange(len(fit), dtype=np.int64), device="cpu",
        batch_size=32, n_classes=4, with_labels=False, collect_voltage=False,
    )


@pytest.fixture(scope="module")
def fr_bank(fr_model, fr_activity):
    """Bank without the temporal block (the historical default)."""
    return build_neuron_record_bank(fr_model, activity=fr_activity)


@pytest.fixture(scope="module")
def temporal_bank(fr_model, fr_activity):
    return build_neuron_record_bank(fr_model, activity=fr_activity, temporal_resolution=10)


def _toy_activity(n_neurons: int = 16, n_bins: int = 30, n_samples: int = 4, bin_ms: float = 2.0):
    """An :class:`ActivityAccumulatorResult` with a hand-computable PSTH (label-free)."""
    psth = np.zeros((n_neurons, n_bins), dtype=np.float64)
    psth[0, :] = 1.0                 # neuron 0: 1 count in every sim bin (summed over samples)
    psth[1, : n_bins // 2] = 1.0     # neuron 1: only in the first half
    psth[2, :] = 4.0                 # neuron 2: 4 counts in every sim bin
    counts = psth.sum(axis=1, keepdims=True)  # per-sample matrix is not used by the block
    counts = np.repeat(counts, n_samples, axis=1).T  # (n_samples, n_neurons)
    zeros_c = np.zeros((4, n_neurons), dtype=np.float64)
    zeros_h = np.zeros(n_neurons, dtype=np.float64)
    return ActivityAccumulatorResult(
        counts=counts,
        psth=psth,
        class_psth=np.zeros((4, n_neurons, n_bins)),
        class_counts=np.zeros((4, n_neurons)),
        class_n=np.zeros(4),
        first_spike_sum=zeros_h.copy(),
        first_spike_count=zeros_h.copy(),
        class_first_spike_sum=zeros_c.copy(),
        class_first_spike_count=zeros_c.copy(),
        v_mean=zeros_h.copy(),
        n_samples=n_samples,
        n_bins=n_bins,
        n_hidden=n_neurons,
        bin_ms=bin_ms,
        labels=None,
    )


# --------------------------------------------------------------------------
# Functional-response source
# --------------------------------------------------------------------------
def test_functional_response_shape_and_projection(fr_bank):
    source = build_functional_response_source(fr_bank, FunctionalResponseConfig(source_dim=64))
    assert isinstance(source, FunctionalResponseSource)
    assert source.X.shape == (fr_bank.n_neurons, 64)
    assert len(source.feature_names) == 64
    assert source.feature_names[0] == "functional_response.proj_00"
    assert source.n_samples == fr_bank.get_block("activity").samples.shape[0]
    assert np.isfinite(source.X).all()


def test_default_projection_matches_the_manual_definition(fr_bank):
    """``raw`` projection == ``samples.T @ G`` with the documented per-column generator."""
    config = FunctionalResponseConfig(source_dim=8)
    samples, _ = response_counts_from_bank(fr_bank)
    expected = samples.astype(np.float64).T @ projection_matrix(
        samples.shape[0], 8, seed=config.projection_seed
    )
    got = project_response_profile(samples, config)
    assert np.allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_projection_is_prefix_stable_and_seed_reproducible(fr_bank):
    small = build_functional_response_source(fr_bank, FunctionalResponseConfig(source_dim=32))
    large = build_functional_response_source(fr_bank, FunctionalResponseConfig(source_dim=64))
    assert np.array_equal(large.X[:, :32], small.X)
    again = build_functional_response_source(fr_bank, FunctionalResponseConfig(source_dim=64))
    assert np.array_equal(again.X, large.X)


def test_different_projection_seed_changes_the_projection(fr_bank):
    a = build_functional_response_source(fr_bank, FunctionalResponseConfig(source_dim=32, projection_seed=0))
    b = build_functional_response_source(fr_bank, FunctionalResponseConfig(source_dim=32, projection_seed=1))
    assert not np.allclose(a.X, b.X)
    assert a.provenance["projection"]["seed"] == 0 and b.provenance["projection"]["seed"] == 1


def test_functional_source_dimension_changes_output_dimension(fr_bank):
    for dim in (8, 32, 64):
        source = build_functional_response_source(fr_bank, FunctionalResponseConfig(source_dim=dim))
        assert source.X.shape == (fr_bank.n_neurons, dim)
        assert source.provenance["output_dimension"] == dim


@pytest.mark.parametrize("mode", FUNCTIONAL_SOURCE_NORMALIZATIONS)
def test_normalization_modes_are_documented_and_applied_before_projection(fr_bank, mode):
    config = FunctionalResponseConfig(source_dim=16, normalization=mode)
    source = build_functional_response_source(fr_bank, config)
    samples, _ = response_counts_from_bank(fr_bank)
    mean, std, zero = response_statistics(samples)
    if mode == "raw":
        centre, scale = np.zeros_like(mean), np.ones_like(std)
    elif mode == "neuron_centered":
        centre, scale = mean, np.ones_like(std)
    else:
        centre, scale = mean, np.where(zero, 1.0, std)
    expected = ((samples - centre) / scale).T @ projection_matrix(samples.shape[0], 16, seed=0)
    if mode != "raw" and zero.any():
        expected[zero] = 0.0
    assert np.allclose(source.X, expected, rtol=1e-10, atol=1e-10)
    prov = source.provenance["normalization"]
    assert prov["mode"] == mode
    assert prov["order"].startswith("normalize the response profile first")
    assert len(prov["mean_over_samples"]) == fr_bank.n_neurons


def test_neuron_centering_removes_the_baseline_and_zscoring_removes_amplitude():
    samples = np.array([[0.0, 1.0], [2.0, 2.0], [10.0, 10.0], [1.0, 10.0]])
    raw = project_response_profile(samples, FunctionalResponseConfig(source_dim=4, normalization="raw"))
    centred = project_response_profile(
        samples, FunctionalResponseConfig(source_dim=4, normalization="neuron_centered")
    )
    zscored = project_response_profile(
        samples, FunctionalResponseConfig(source_dim=4, normalization="neuron_zscored")
    )
    # both transformations change the projected view of the first neuron
    assert not np.allclose(centred[0], raw[0])
    assert not np.allclose(zscored[0], raw[0])
    # z-scoring is invariant to a positive rescaling of a neuron's profile
    scaled = samples * np.array([3.0, 0.5])
    zscored_scaled = project_response_profile(
        scaled, FunctionalResponseConfig(source_dim=4, normalization="neuron_zscored")
    )
    assert np.allclose(zscored_scaled[0], zscored[0], rtol=1e-10)
    assert np.allclose(zscored_scaled[1], zscored[1], rtol=1e-10)
    # ... while the raw view is not
    raw_scaled = project_response_profile(
        scaled, FunctionalResponseConfig(source_dim=4, normalization="raw")
    )
    assert not np.allclose(raw_scaled[0], raw[0])


def test_zero_variance_neuron_rule_is_deterministic_and_keeps_the_row():
    samples = np.array([[1.0, 5.0], [1.0, 5.0], [1.0, 5.0]])  # (n_samples, n_neurons): both constant
    config = FunctionalResponseConfig(source_dim=8, normalization="neuron_zscored")
    first = project_response_profile(samples, config)
    second = project_response_profile(samples, config)
    assert np.array_equal(first, second)
    assert np.all(first == 0.0)
    assert first.shape == (2, 8)  # no neuron dropped
    _, _, zero = response_statistics(samples)
    assert zero.all()


def test_sample_order_hash_is_stable_and_order_sensitive(fr_bank):
    samples, _ = response_counts_from_bank(fr_bank)
    digest = sample_order_hash(samples)
    assert digest == sample_order_hash(samples.copy())
    assert len(digest) == 64
    permuted = samples[np.arange(samples.shape[0])[::-1]]
    assert sample_order_hash(permuted) != digest
    source = build_functional_response_source(fr_bank, FunctionalResponseConfig(source_dim=8))
    assert source.sample_order_hash == digest
    order = source.provenance["sample_order"]
    assert order["stored_order_preserved"] is True
    assert order["labels_used_for_ordering"] is False
    assert order["speaker_identity_used"] is False


def test_functional_response_provenance_is_label_free_and_complete(fr_bank):
    source = build_functional_response_source(fr_bank, FunctionalResponseConfig(source_dim=32))
    prov = source.provenance
    assert prov["schema"] == FUNCTIONAL_RESPONSE_SCHEMA
    assert prov["source_type"] == "individual_stimulus_response"
    assert prov["source_split"] != "" and "probe" not in prov["source_split"]
    assert prov["uses_labels"] is False and prov["fit_only"] is True
    assert prov["input_shape"][1] == fr_bank.n_neurons
    assert prov["projection"]["data_dependent"] is False
    assert prov["projection"]["learned"] is False and prov["projection"]["fitted"] is False
    assert prov["projection"]["prefix_stable"] is True
    assert isinstance(json.dumps(prov), str)  # machine-readable
    assert source.to_dict()["sample_order_hash"] == source.sample_order_hash


def test_functional_response_rejects_missing_counts_and_bad_config(tiny_model):
    plain = build_neuron_record_bank(tiny_model, with_activity=False)
    with pytest.raises(FunctionalResponseError, match="activity"):
        build_functional_response_source(plain)
    with pytest.raises(FunctionalResponseError, match="source_dim"):
        FunctionalResponseConfig(source_dim=0)
    with pytest.raises(FunctionalResponseError, match="normalization"):
        FunctionalResponseConfig(normalization="standardised")
    with pytest.raises(FunctionalResponseError, match="projection_seed"):
        FunctionalResponseConfig(projection_seed=-1)
    with pytest.raises(FunctionalResponseError, match="NeuronRecordBank"):
        build_functional_response_source(np.zeros((3, 3)))  # type: ignore[arg-type]


def test_functional_response_api_has_no_labels_parameters():
    for func in (build_functional_response_source, project_response_profile, response_counts_from_bank):
        names = set(inspect.signature(func).parameters)
        assert not (names & {"labels", "y", "targets", "probe", "test"}), names
    assert "labels" not in set(inspect.signature(sample_order_hash).parameters)


def test_functional_response_never_builds_a_forbidden_tensor(fr_bank):
    """The full ``(n_neurons, n_samples)`` transpose is never materialised (chunked only)."""
    source = build_functional_response_source(
        fr_bank, FunctionalResponseConfig(source_dim=16, chunk_size=7)
    )
    assert source.X.ndim == 2
    samples, _ = response_counts_from_bank(fr_bank)
    assert samples.shape[0] > 7  # chunking was actually exercised
    big = build_functional_response_source(fr_bank, FunctionalResponseConfig(source_dim=16))
    assert np.allclose(source.X, big.X, rtol=1e-12, atol=1e-12)  # chunked == unchunked


# --------------------------------------------------------------------------
# Temporal block
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "n_bins,resolution,expected",
    [
        (30, 10, [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, 18), (18, 21), (21, 24), (24, 27), (27, 30)]),
        (30, 5, [(0, 6), (6, 12), (12, 18), (18, 24), (24, 30)]),
        (700, 10, [(0, 70), (70, 140), (140, 210), (210, 280), (280, 350), (350, 420), (420, 490),
                   (490, 560), (560, 630), (630, 700)]),
        (7, 3, [(0, 2), (2, 5), (5, 7)]),
    ],
)
def test_temporal_bin_edges_are_exact(n_bins, resolution, expected):
    assert temporal_bin_edges(n_bins, resolution) == expected


def test_temporal_bin_edges_validation():
    with pytest.raises(ValueError):
        temporal_bin_edges(10, 11)
    with pytest.raises(ValueError):
        temporal_bin_edges(10, 0)
    with pytest.raises(ValueError):
        temporal_bin_edges(0, 1)


def test_temporal_block_values_on_a_toy_activity_tensor(fr_model):
    activity = _toy_activity()
    bank = build_neuron_record_bank(fr_model, activity=activity, temporal_resolution=10)
    block = bank.get_block(TEMPORAL_BLOCK)
    X, names = block.to_matrix()
    assert X.shape == (16, 10)
    assert names == [f"temporal.bin_{k:02d}" for k in range(10)]
    # bin 0 covers sim bins [0, 3) -> 6 ms at bin_ms = 2.0; rate = counts / (n_samples * seconds)
    duration_s = 3 * 2.0 / 1000.0
    assert np.allclose(X[0], 3.0 / (4 * duration_s))          # neuron 0: 1 count per sim bin
    assert np.allclose(X[2], 12.0 / (4 * duration_s))         # neuron 2: 4 counts per sim bin
    assert np.allclose(X[1, :5], 3.0 / (4 * duration_s))      # neuron 1 active in the first half
    assert np.allclose(X[1, 5:], 0.0)
    assert np.allclose(X[3:], 0.0)                            # all other neurons silent
    assert block.feature_names == tuple(f"bin_{k:02d}" for k in range(10))


def test_temporal_resolution_changes_output_dimension(fr_model):
    activity = _toy_activity()
    five = build_neuron_record_bank(fr_model, activity=activity, temporal_resolution=5)
    ten = build_neuron_record_bank(fr_model, activity=activity, temporal_resolution=10)
    X5, names5 = five.get_block(TEMPORAL_BLOCK).to_matrix()
    X10, _ = ten.get_block(TEMPORAL_BLOCK).to_matrix()
    assert X5.shape == (16, 5) and X10.shape == (16, 10)
    assert names5 == [f"temporal.bin_{k:02d}" for k in range(5)]
    # equal-width bins: the coarser rate is the mean of the two finer rates
    assert np.allclose(X5[:, 0], X10[:, :2].mean(axis=1))
    assert np.allclose(X5[:, 4], X10[:, 8:].mean(axis=1))


def test_temporal_block_is_absent_unless_requested(fr_model, fr_activity):
    default = build_neuron_record_bank(fr_model, activity=fr_activity)
    assert TEMPORAL_BLOCK not in default.block_names
    assert default.provenance["temporal"]["requested"] is False
    assert default.provenance["temporal"]["present"] is False
    requested = build_neuron_record_bank(fr_model, activity=fr_activity, temporal_resolution=10)
    assert TEMPORAL_BLOCK in requested.block_names
    assert requested.provenance["temporal"]["present"] is True


def test_config_can_request_the_temporal_block(fr_model, fr_activity):
    cfg = V2Config.from_config(
        Config({"vector": {"enabled_blocks": ["intrinsic", "temporal"], "temporal_resolution": 6}}),
        warn=False,
    )
    bank = build_neuron_record_bank(fr_model, activity=fr_activity, config=cfg)
    assert TEMPORAL_BLOCK in bank.block_names
    X, names = bank.get_block(TEMPORAL_BLOCK).to_matrix()
    assert X.shape[1] == 6 and len(names) == 6
    assert bank.provenance["temporal"]["resolution"] == 6

    # a configuration that only asks the residual for temporal also builds the block
    cfg2 = V2Config.from_config(
        Config({
            "vector": {
                "d": 64, "structured_d": 48, "learned_residual_d": 16,
                "residual": {"enabled": True, "source_temporal": True},
            }
        }),
        warn=False,
    )
    bank2 = build_neuron_record_bank(fr_model, activity=fr_activity, config=cfg2)
    assert TEMPORAL_BLOCK in bank2.block_names


def test_temporal_provenance_is_label_free_and_descriptive(fr_model):
    activity = _toy_activity()
    bank = build_neuron_record_bank(fr_model, activity=activity, temporal_resolution=5)
    prov = bank.provenance["temporal"]
    assert prov["present"] is True and prov["label_free"] is True
    assert prov["class_conditioned"] is False and prov["uses_labels"] is False
    assert prov["mode"] == "fit_population_psth_coarse_bins"
    assert prov["n_samples"] == 4 and prov["resolution"] == 5
    assert len(prov["bins"]) == 5
    first = prov["bins"][0]
    assert first["name"] == "bin_00" and (first["start_bin"], first["stop_bin"]) == (0, 6)
    assert first["start_ms"] == 0.0 and first["stop_ms"] == 12.0
    assert "no class conditioning" in bank.get_block(TEMPORAL_BLOCK).orientation
    assert isinstance(json.dumps(prov), str)


def test_bank_activity_split_guard(tiny_model, synthetic_rec):
    fit, dev, probe, info = make_train_dev_probe_split(
        synthetic_rec, dev_fraction=0.25, probe_fraction=0.25, seed=0, prefer_speaker_aware=True
    )
    probe_bank = build_neuron_record_bank(tiny_model, fit_rec=probe, device="cpu", batch_size=32)
    with pytest.raises(Exception, match="FIT only"):
        assert_label_free_fit_activity(probe_bank)


# --------------------------------------------------------------------------
# Structured encoder integration
# --------------------------------------------------------------------------
def test_default_48d_path_is_exactly_unchanged(fr_bank, temporal_bank):
    """The temporal block must not leak into the default configuration."""
    default = StructuredVectorEncoder(fr_bank, structured_d=48).encode()
    with_temporal_available = StructuredVectorEncoder(temporal_bank, structured_d=48).encode()
    assert np.array_equal(default.X, with_temporal_available.X)
    assert default.feature_names == with_temporal_available.feature_names
    assert default.plan.level0_dimension == 48


def test_temporal_coordinates_appear_only_when_requested(temporal_bank):
    plan = StructuredVectorEncoder(temporal_bank, structured_d=1).plan
    assert "temporal" not in plan.requested_blocks
    assert "temporal" not in plan.present_blocks

    encoder = StructuredVectorEncoder(
        temporal_bank,
        structured_d=1,
        enabled_blocks=[*STRUCTURAL_BLOCKS, "temporal"],
    )
    assert list(encoder.plan.present_blocks)[-1] == "temporal"
    # 48 structural summary coordinates followed by the 10 temporal bins
    assert encoder.plan.level0_dimension == 48 + 10
    assert encoder.plan.level0_layout[48:58] == tuple(
        ("temporal", f"bin_{k:02d}") for k in range(10)
    )
    # and they are real output coordinates once the requested dimension covers them
    full = StructuredVectorEncoder(
        temporal_bank,
        structured_d=encoder.plan.level0_dimension,
        enabled_blocks=[*STRUCTURAL_BLOCKS, "temporal"],
    )
    assert full.feature_names[48:58] == tuple(f"temporal.bin_{k:02d}" for k in range(10))


def test_temporal_structured_dimension_follows_the_configuration(fr_model, fr_activity):
    for resolution in (4, 10):
        cfg = V2Config.from_config(
            Config({"vector": {"enabled_blocks": [*STRUCTURAL_BLOCKS, "temporal"],
                               "temporal_resolution": resolution}}),
            warn=False,
        )
        bank = build_neuron_record_bank(fr_model, activity=fr_activity, config=cfg)
        enabled = list(cfg.vector.enabled_blocks)
        plan = StructuredVectorEncoder(bank, structured_d=1, enabled_blocks=enabled).plan
        level0 = plan.level0_dimension
        # the structural summary dimension is whatever the bank provides; derive, not assume
        structural_only = build_neuron_record_bank(fr_model, activity=fr_activity)
        baseline = StructuredVectorEncoder(structural_only, structured_d=1).plan.level0_dimension
        assert level0 == baseline + resolution
        encoder = StructuredVectorEncoder.from_config(cfg, bank)
        assert encoder.plan.structured_d == cfg.vector.structured_d
        assert "temporal" in encoder.plan.present_blocks


def test_structural_48_plus_temporal_vector(fr_model, fr_activity):
    cfg = V2Config.from_config(
        Config({"vector": {"enabled_blocks": [*STRUCTURAL_BLOCKS, "temporal"], "temporal_resolution": 10}}),
        warn=False,
    )
    bank = build_neuron_record_bank(fr_model, activity=fr_activity, config=cfg)
    enabled = [*STRUCTURAL_BLOCKS, "temporal"]
    structured_d = StructuredVectorEncoder(bank, structured_d=1, enabled_blocks=enabled).plan.level0_dimension
    vectors = build_neuron_vectors(bank, structured_d=structured_d, enabled_blocks=enabled)
    assert vectors.shape == (bank.n_neurons, structured_d)
    assert vectors.feature_names[-1] == "temporal.bin_09"
    assert vectors.structured_d == structured_d and vectors.residual_d == 0


# --------------------------------------------------------------------------
# Residual source expansion
# --------------------------------------------------------------------------
def test_residual_source_input_dimension_adapts_to_the_new_sources(fr_bank, temporal_bank):
    base = build_residual_source(fr_bank)
    with_functional = build_residual_source(
        fr_bank, ResidualSourceConfig(include_functional_response=True, functional_source_dim=32)
    )
    with_temporal = build_residual_source(temporal_bank, ResidualSourceConfig(include_temporal=True))
    with_both = build_residual_source(
        temporal_bank,
        ResidualSourceConfig(include_functional_response=True, functional_source_dim=32,
                             include_temporal=True),
    )
    assert with_functional.n_features == base.n_features + 32
    assert with_temporal.n_features == base.n_features + 10
    assert with_both.n_features == base.n_features + 42
    assert with_functional.feature_names[: base.n_features] == base.feature_names
    assert with_temporal.feature_names[: base.n_features] == base.feature_names


def test_default_residual_source_schema_is_unchanged(fr_bank):
    """The historical source view (and therefore its schema hash) must not move."""
    source = build_residual_source(fr_bank)
    names = list(source.feature_names)
    expected_raw = sum(source.provenance["raw_view_dimensions"].values())
    assert sum("raw_view_" in n for n in names) == expected_raw
    assert not any(n.startswith("functional_response.") for n in names)
    assert not any(n.startswith("temporal.") for n in names)
    assert source.provenance["functional_response"]["enabled"] is False
    assert source.provenance["temporal"]["enabled"] is False
    assert set(source_groups_from_names(names)) == {SOURCE_GROUP_STRUCTURAL}


def test_source_groups_classify_coordinates():
    names = [
        "intrinsic.learned_bias", "input_conn.raw_view_00",
        "functional_response.proj_00", "functional_response.proj_01", "temporal.bin_00",
    ]
    groups = source_groups_from_names(names)
    assert groups[SOURCE_GROUP_STRUCTURAL] == [0, 1]
    assert groups[SOURCE_GROUP_FUNCTIONAL] == [2, 3]
    assert groups[SOURCE_GROUP_TEMPORAL] == [4]
    assert source_group_of_feature("temporal.bin_03") == SOURCE_GROUP_TEMPORAL
    assert source_group_of_feature("functional_response.proj_09") == SOURCE_GROUP_FUNCTIONAL
    assert source_group_of_feature("recurrent_in.raw_view_01") == SOURCE_GROUP_STRUCTURAL


def test_source_does_not_duplicate_temporal_coordinates(fr_model, fr_activity):
    """A block selected by the encoder must not be appended twice to the residual source."""
    cfg = V2Config.from_config(
        Config({"vector": {"enabled_blocks": [*STRUCTURAL_BLOCKS, "temporal"], "temporal_resolution": 10}}),
        warn=False,
    )
    bank = build_neuron_record_bank(fr_model, activity=fr_activity, config=cfg)
    # the structured part already selects `temporal` here, so the dedicated temporal append
    # must not add the same coordinates a second time
    source = build_residual_source(
        bank,
        ResidualSourceConfig(
            enabled_blocks=[*STRUCTURAL_BLOCKS, "temporal"], include_temporal=True
        ),
    )
    names = list(source.feature_names)
    assert len(names) == len(set(names))
    assert sum(n.startswith("temporal.") for n in names) == 10
    assert source.provenance["temporal"]["duplicate_coordinates_skipped"] == 10
    assert source.provenance["temporal"]["output_dimension"] == 0


def test_residual_source_config_validation():
    with pytest.raises(ResidualError, match="at least one"):
        ResidualSourceConfig(
            include_level0=False, include_level1=False, include_raw_weight_view=False,
            include_functional_response=False, include_temporal=False,
        )
    with pytest.raises(ResidualError, match="source_dim"):
        ResidualSourceConfig(include_functional_response=True, functional_source_dim=0)
    with pytest.raises(ResidualError, match="normalization"):
        ResidualSourceConfig(include_functional_response=True, functional_normalization="nope")
    config = ResidualSourceConfig.from_mapping({"include_temporal": True, "unknown_key": 1})
    assert config.include_temporal is True


def test_residual_source_config_is_reachable_from_the_v2_config():
    v2 = V2Config.from_config(
        Config({
            "vector": {
                "d": 64, "structured_d": 48, "learned_residual_d": 16,
                "functional_source_dim": 48, "functional_projection_seed": 7,
                "functional_source_normalization": "neuron_centered",
                "residual": {
                    "enabled": True, "source_functional_response": True,
                    "source_temporal": True, "mask_mode": "block",
                },
            }
        }),
        warn=False,
    )
    source_config = ResidualSourceConfig.from_v2_config(v2)
    assert source_config.include_functional_response is True
    assert source_config.include_temporal is True
    assert source_config.functional_source_dim == 48
    assert source_config.functional_projection_seed == 7
    assert source_config.functional_normalization == "neuron_centered"
    assert source_config.enabled_blocks == ["intrinsic", "input_conn", "recurrent_in", "recurrent_out"]
    training = ResidualTrainingConfig.from_v2_config(v2, residual_dim=16)
    assert training.mask_mode == "block"


# --------------------------------------------------------------------------
# Residual training with the expanded source
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def expanded_source(fr_model, fr_activity):
    cfg = V2Config.from_config(
        Config({"vector": {"enabled_blocks": [*STRUCTURAL_BLOCKS, "temporal"], "temporal_resolution": 10}}),
        warn=False,
    )
    bank = build_neuron_record_bank(fr_model, activity=fr_activity, config=cfg)
    source = build_residual_source(
        bank,
        ResidualSourceConfig(include_functional_response=True, functional_source_dim=16,
                             include_temporal=True),
    )
    return bank, source


def test_residual_input_dimension_and_diagnostics(expanded_source):
    bank, source = expanded_source
    config = ResidualTrainingConfig.from_mapping(
        {"residual_dim": 8, "hidden_dim": 16, "epochs": 3, "batch_size": 16, "mask_mode": "block", "seed": 0}
    )
    residual = train_residual(bank, config=config, source=source)
    assert residual.input_dim == source.n_features
    assert residual.encoder.input_dim == source.n_features
    assert residual.feature_names == source.feature_names
    last = residual.history[-1]
    for group in (SOURCE_GROUP_STRUCTURAL, SOURCE_GROUP_FUNCTIONAL, SOURCE_GROUP_TEMPORAL):
        assert f"train_masked_mse_{group}" in last
        assert f"val_masked_mse_{group}" in last
    assert last["mask_mode"] == "block"
    assert residual.provenance["mask_policy"]["mode"] == "block"
    assert set(residual.provenance["source_groups"]) == {
        SOURCE_GROUP_STRUCTURAL, SOURCE_GROUP_FUNCTIONAL, SOURCE_GROUP_TEMPORAL
    }
    assert residual.provenance["uses_labels"] is False
    diagnostics = residual.provenance["source_diagnostics"]["per_source_history_fields"]
    assert "train_masked_mse_functional_response" in diagnostics


def test_residual_training_does_not_modify_the_snn(fr_model, expanded_source):
    bank, source = expanded_source
    before = {k: v.detach().clone() for k, v in fr_model.state_dict().items()}
    train_residual(
        bank,
        config=ResidualTrainingConfig.from_mapping(
            {"residual_dim": 4, "hidden_dim": 8, "epochs": 2, "batch_size": 16, "seed": 0}
        ),
        source=source,
    )
    after = fr_model.state_dict()
    assert set(before) == set(after)
    for key, value in before.items():
        assert torch.equal(value, after[key]), f"the SNN parameter {key} changed"


def test_residual_persists_and_rejects_a_mismatched_source(expanded_source, tmp_path):
    bank, source = expanded_source
    residual = train_residual(
        bank,
        config=ResidualTrainingConfig.from_mapping(
            {"residual_dim": 4, "hidden_dim": 8, "epochs": 2, "batch_size": 16, "seed": 0}
        ),
        source=source,
    )
    from src.residual import ResidualResult

    path = residual.save(tmp_path / "residual.pt")
    loaded = ResidualResult.load(
        path, expected_feature_names=source.feature_names, expected_residual_dim=4
    )
    assert loaded.feature_names == source.feature_names
    assert loaded.input_dim == source.n_features
    assert loaded.provenance["source_groups"] == residual.provenance["source_groups"]
    payload_groups = loaded.source_schema["provenance"]["source_groups"]
    assert set(payload_groups) == {
        SOURCE_GROUP_STRUCTURAL, SOURCE_GROUP_FUNCTIONAL, SOURCE_GROUP_TEMPORAL
    }

    # a source without the new parts must be rejected (never silently adapted)
    plain = build_residual_source(bank)
    with pytest.raises(ResidualError, match="incompatible residual source"):
        loaded.encode(plain)
    with pytest.raises(ResidualError, match="incompatible residual source"):
        ResidualResult.load(
            path, expected_feature_names=plain.feature_names, expected_residual_dim=4
        )


def test_masks_remain_deterministic_in_both_modes():
    groups = {"structural": [0, 1, 2, 3], "functional_response": [4, 5], "temporal": [6, 7]}
    coordinate = make_source_mask(
        5, 8, mode="coordinate", mask_fraction=0.25, minimum_visible_features=2,
        rng=np.random.default_rng([0, 0]), groups=groups,
    )
    assert np.array_equal(
        coordinate,
        make_source_mask(
            5, 8, mode="coordinate", mask_fraction=0.25, minimum_visible_features=2,
            rng=np.random.default_rng([0, 0]), groups=groups,
        ),
    )
    block = make_block_mask(
        5, 8, groups=groups, mask_fraction=0.25, minimum_visible_features=2,
        rng=np.random.default_rng([0, 0]),
    )
    assert np.array_equal(
        block,
        make_block_mask(
            5, 8, groups=groups, mask_fraction=0.25, minimum_visible_features=2,
            rng=np.random.default_rng([0, 0]),
        ),
    )
    # every masked row keeps at least `minimum_visible_features` coordinates visible
    assert (8 - block.sum(axis=1) >= 2).all()
    # and each row withholds exactly one group's worth of coordinates
    sizes = {len(v) for v in groups.values()}
    assert set(block.sum(axis=1).tolist()) <= sizes
    with pytest.raises(ResidualError, match="groups"):
        make_source_mask(
            3, 8, mode="block", mask_fraction=0.25, minimum_visible_features=2,
            rng=np.random.default_rng(0), groups=None,
        )
    with pytest.raises(ResidualError, match="mask_mode"):
        make_source_mask(
            3, 8, mode="nope", mask_fraction=0.25, minimum_visible_features=2,
            rng=np.random.default_rng(0), groups=groups,
        )


def test_block_mask_falls_back_when_a_group_would_hide_too_much():
    groups = {"only": [0, 1, 2, 3, 4, 5, 6, 7]}
    mask = make_block_mask(
        4, 8, groups=groups, mask_fraction=0.25, minimum_visible_features=3,
        rng=np.random.default_rng(0),
    )
    assert (8 - mask.sum(axis=1) >= 3).all()
    assert bool(mask.any())


def test_training_config_validates_the_mask_mode():
    with pytest.raises(ResidualError, match="mask_mode"):
        ResidualTrainingConfig.from_mapping({"mask_mode": "sources"})
    assert ResidualTrainingConfig.from_mapping({"mask_mode": "block"}).mask_mode == "block"
    assert ResidualTrainingConfig().mask_mode == "coordinate"


# --------------------------------------------------------------------------
# Full vector and configuration
# --------------------------------------------------------------------------
@pytest.mark.parametrize("residual_dim,expected", [(0, 48), (16, 64), (52, 100)])
def test_full_vector_dimensions_still_work(fr_bank, residual_dim, expected):
    residual = None
    if residual_dim:
        residual = train_residual(
            fr_bank,
            config=ResidualTrainingConfig.from_mapping(
                {"residual_dim": residual_dim, "hidden_dim": 8, "epochs": 2, "batch_size": 16, "seed": 0}
            ),
        )
    vectors = build_neuron_vectors(fr_bank, structured_d=48, residual=residual)
    assert vectors.shape == (fr_bank.n_neurons, expected)
    assert vectors.full_dimension == vectors.structured_d + vectors.residual_d


def test_full_vector_with_temporal_source_and_residual(fr_model, fr_activity):
    """structured_48+temporal plus a residual trained on the expanded source."""
    cfg = V2Config.from_config(
        Config({
            "vector": {
                "enabled_blocks": [*STRUCTURAL_BLOCKS, "temporal"], "temporal_resolution": 10,
                "d": 64, "structured_d": 58, "learned_residual_d": 6,
                "residual": {"enabled": True, "source_functional_response": True,
                             "source_temporal": True},
            }
        }),
        warn=False,
    )
    bank = build_neuron_record_bank(fr_model, activity=fr_activity, config=cfg)
    encoder = StructuredVectorEncoder(bank, structured_d=1, enabled_blocks=cfg.vector.enabled_blocks)
    assert encoder.plan.level0_dimension == 58
    source = build_residual_source(bank, ResidualSourceConfig.from_v2_config(cfg))
    assert source.n_features > build_residual_source(bank).n_features
    residual = train_residual(
        bank,
        config=ResidualTrainingConfig.from_v2_config(cfg, residual_dim=6),
        source=source,
    )
    vectors = neuron_vectors_from_config(cfg, bank, residual=residual)
    assert vectors.shape == (bank.n_neurons, 64)
    assert vectors.feature_names[57] == "temporal.bin_09"
    assert vectors.feature_names[-1] == "residual[5]"


def test_new_config_fields_and_validation():
    v = VectorConfig.from_mapping({})
    assert v.functional_source_dim == DEFAULT_FUNCTIONAL_SOURCE_DIM
    assert v.functional_projection_seed == 0
    assert v.functional_source_normalization == "raw"
    assert v.residual.source_functional_response is False
    assert v.residual.source_temporal is False
    assert v.residual.mask_mode == "coordinate"
    mapping = json.loads(json.dumps(v.to_dict()))
    assert mapping["functional_source_dim"] == DEFAULT_FUNCTIONAL_SOURCE_DIM

    with pytest.raises(V2ConfigError, match="functional_source_dim"):
        VectorConfig.from_mapping({"functional_source_dim": 0})
    with pytest.raises(V2ConfigError, match="functional_source_dim"):
        VectorConfig.from_mapping({"functional_source_dim": 10_000})
    with pytest.raises(V2ConfigError, match="functional_projection_seed"):
        VectorConfig.from_mapping({"functional_projection_seed": -1})
    with pytest.raises(V2ConfigError, match="functional_source_normalization"):
        VectorConfig.from_mapping({"functional_source_normalization": "zscore"})
    with pytest.raises(V2ConfigError, match="mask_mode"):
        VectorResidualConfig.from_mapping({"mask_mode": "nope"})
    # a source request without a residual to consume it is a contradiction
    with pytest.raises(V2ConfigError, match="source_functional_response"):
        VectorConfig.from_mapping({"residual": {"source_functional_response": True}})
    with pytest.raises(V2ConfigError, match="source_temporal|learned_residual_d"):
        VectorConfig.from_mapping({"residual": {"enabled": True, "source_temporal": True}})
    # valid combinations are accepted
    ok = VectorConfig.from_mapping(
        {"d": 64, "structured_d": 48, "learned_residual_d": 16,
         "residual": {"enabled": True, "source_temporal": True, "mask_mode": "block"}}
    )
    assert ok.residual.source_temporal is True and ok.residual.mask_mode == "block"


def test_temporal_disabled_and_enabled_are_unambiguous():
    off = VectorConfig.from_mapping({"enabled_blocks": list(STRUCTURAL_BLOCKS)})
    on = VectorConfig.from_mapping({"enabled_blocks": [*STRUCTURAL_BLOCKS, "temporal"]})
    assert "temporal" not in off.enabled_blocks and off.unimplemented_blocks == []
    assert "temporal" in on.enabled_blocks and on.unimplemented_blocks == []
    assert off.temporal_resolution == 10  # the knob is always present and validated
    with pytest.raises(V2ConfigError):
        VectorConfig.from_mapping({"temporal_resolution": 0})