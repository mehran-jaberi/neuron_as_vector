"""Tests for the deterministic variable-dimensional structured encoder.

Coverage: exact 48-D compatibility, lower/higher dimension behaviour, prefix
selection, Level-1 deterministic detail, the fixed seeded projection, block
selection, arbitrary neuron counts, chunking, label-free guarantees, the memory
contract and V2-config integration.
"""

from __future__ import annotations

import copy
import inspect
import json

import numpy as np
import pytest
import torch

from src.data import SHDRecordings
from src.model import SNNConfig, RecurrentLIFSNN, build_model
from src.neuron_record import NeuronRecordBank, build_neuron_record_bank
from src.neurons import extract_structural_representations
from src.structured_vector import (
    ACTIVITY_DETAIL_NAMES,
    DEFAULT_PROJECTION_SEED,
    DETAIL_DEFINITIONS,
    MAX_STRUCTURED_D,
    WEIGHT_DETAIL_NAMES,
    StructuredVectorEncoder,
    StructuredVectorError,
    encode_structured_vectors,
)
from src.utils import Config, PROJECT_ROOT, load_config
from src.v2_config import V2Config

CANONICAL_CHECKPOINTS = ("sweep_l2_0.pt", "nsb_seed1.pt", "nsb_seed2.pt")
DEFAULT_BLOCKS = ("intrinsic", "input_conn", "recurrent_in", "recurrent_out")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _bank(model, *, with_activity=False, rec=None, **kwargs) -> NeuronRecordBank:
    if with_activity and rec is not None:
        return build_neuron_record_bank(model, fit_rec=rec, device="cpu", batch_size=32, **kwargs)
    return build_neuron_record_bank(model, with_activity=False, **kwargs)


def _canonical_architecture_model() -> RecurrentLIFSNN:
    cfg = load_config(PROJECT_ROOT / "configs/neuron_space_baseline.yaml")
    model = build_model(cfg, seed=0)
    with torch.no_grad():
        model.b_hid.copy_(torch.linspace(-0.5, 0.5, model.cfg.n_hidden))
    return model


def _toy_model(n_hidden: int, *, n_input: int = 40) -> RecurrentLIFSNN:
    cfg = SNNConfig(n_input=n_input, n_hidden=n_hidden, n_output=4, n_bins=30, bin_ms=2.0,
                    neuron_param_mode="bias")
    model = build_model(cfg, seed=3)
    with torch.no_grad():
        model.b_hid.copy_(torch.linspace(-0.5, 0.5, n_hidden))
    return model


def _relabelled(rec: SHDRecordings, shift: int = 2) -> SHDRecordings:
    labels = (rec.labels_array + shift) % max(int(rec.labels_array.max()) + 1, 1)
    return SHDRecordings(
        times_ms=rec.times_ms, units=rec.units, offsets=rec.offsets, labels=labels,
        n_channels=rec.n_channels, speakers=rec.speakers, name=rec.name, meta=dict(rec.meta),
    )


# --------------------------------------------------------------------------
# Exact 48-D baseline
# --------------------------------------------------------------------------
def _assert_48d_matches(model, *, expected_features: int = 48):
    bank = _bank(model)
    vectors = StructuredVectorEncoder(bank, structured_d=expected_features).encode()
    X_old, names_old = extract_structural_representations(model).to_matrix()
    assert vectors.shape == X_old.shape == (model.cfg.n_hidden, expected_features)
    assert list(vectors.feature_names) == list(names_old)
    assert float(np.abs(vectors.X - X_old).max()) <= 1e-12


def test_default_48d_matches_existing_representation(trained_like_model):
    _assert_48d_matches(trained_like_model, expected_features=48)


def test_default_48d_matches_for_untrained_model(tiny_model):
    # untrained -> no intrinsic block -> the level-0 anchor is 47-D
    bank = _bank(tiny_model)
    encoder = StructuredVectorEncoder(bank, structured_d=47)
    assert encoder.level0_dimension == 47
    _assert_48d_matches(tiny_model, expected_features=47)


def test_default_48d_matches_for_dynamical_model(dynamical_model):
    _assert_48d_matches(dynamical_model, expected_features=49)


def test_canonical_256_neuron_model_matches_existing_representation():
    model = _canonical_architecture_model()
    assert model.cfg.n_hidden == 256
    bank = _bank(model)
    encoder = StructuredVectorEncoder(bank, structured_d=48)
    assert encoder.level0_dimension == 48
    assert encoder.level1_dimension == len(WEIGHT_DETAIL_NAMES) * 3  # three weighted blocks
    _assert_48d_matches(model, expected_features=48)


@pytest.mark.parametrize("name", CANONICAL_CHECKPOINTS)
def test_canonical_trained_checkpoints_match_at_48d(name):
    path = PROJECT_ROOT / "checkpoints" / name
    if not path.exists():
        pytest.skip(f"{name} is a local training artefact and is not present")
    model, _ = RecurrentLIFSNN.load(str(path), map_location="cpu")
    bank = NeuronRecordBank.from_checkpoint(path, with_activity=False)
    X_new, names_new = None, None
    vectors = StructuredVectorEncoder(bank, structured_d=48).encode()
    X_old, names_old = extract_structural_representations(model).to_matrix()
    assert list(vectors.feature_names) == list(names_old)
    assert vectors.shape == X_old.shape == (model.cfg.n_hidden, 48)
    assert float(np.abs(vectors.X - X_old).max()) <= 1e-12


# --------------------------------------------------------------------------
# Dimension hierarchy
# --------------------------------------------------------------------------
def test_level0_and_level1_dimensions_and_names(trained_like_model):
    bank = _bank(trained_like_model)
    encoder = StructuredVectorEncoder(bank, structured_d=48)
    assert encoder.level0_dimension == 48
    assert encoder.level1_dimension == 93
    assert encoder.source_dimension == 141
    assert encoder.feature_names[:3] == ("intrinsic.learned_bias", "input_conn.entropy", "input_conn.l1")

    full = StructuredVectorEncoder(bank, structured_d=encoder.source_dimension).encode()
    level0 = full.feature_names[:48]
    assert list(level0) == list(bank.to_structured_matrix(blocks=DEFAULT_BLOCKS)[1])
    detail_names = full.feature_names[48:]
    assert detail_names[:3] == ("input_conn." + WEIGHT_DETAIL_NAMES[0], "input_conn." + WEIGHT_DETAIL_NAMES[1],
                                "input_conn." + WEIGHT_DETAIL_NAMES[2])
    assert "input_conn.abs_hist_00" in detail_names
    assert "recurrent_in.top_abs_1" in detail_names


@pytest.mark.parametrize("d", [1, 2, 8, 16, 32, 47])
def test_lower_dimensions_are_the_deterministic_prefix(trained_like_model, d):
    bank = _bank(trained_like_model)
    full = StructuredVectorEncoder(bank, structured_d=48).encode()
    vectors = StructuredVectorEncoder(bank, structured_d=d).encode()
    assert vectors.shape == (bank.n_neurons, d)
    assert vectors.feature_names == full.feature_names[:d]
    assert np.array_equal(vectors.X, full.X[:, :d])  # deterministic coordinate selection
    assert all(c.kind == "level0" for c in vectors.coordinates)
    assert vectors.provenance["projection_used"] is False
    again = StructuredVectorEncoder(bank, structured_d=d).encode()
    assert np.array_equal(again.X, vectors.X)
    assert again.feature_names == vectors.feature_names


def test_dimension_equal_to_level0_returns_the_source_matrix_unchanged(trained_like_model):
    bank = _bank(trained_like_model)
    encoder = StructuredVectorEncoder(bank, structured_d=48)
    vectors = encoder.encode()
    X_expected, names_expected = bank.to_structured_matrix(blocks=DEFAULT_BLOCKS)
    assert encoder.level0_dimension == 48
    assert list(vectors.feature_names) == list(names_expected)
    assert np.array_equal(vectors.X, X_expected)  # no transform, no normalisation, no projection
    assert all(c.kind == "level0" for c in vectors.coordinates)
    assert encoder.provenance["projection_used"] is False


@pytest.mark.parametrize("d", [49, 56, 64, 80, 100, 141])
def test_higher_dimensions_add_interpretable_level1_detail(trained_like_model, d):
    bank = _bank(trained_like_model)
    level0_X, level0_names = bank.to_structured_matrix(blocks=DEFAULT_BLOCKS)
    vectors = StructuredVectorEncoder(bank, structured_d=d).encode()
    assert vectors.shape == (bank.n_neurons, d)
    assert np.array_equal(vectors.X[:, :48], level0_X)  # level 0 is never altered
    kinds = {c.kind for c in vectors.coordinates}
    assert kinds == ({"level0"} if d <= 48 else {"level0", "level1"})
    assert vectors.provenance["projection_used"] is False
    # every Level-1 coordinate is a named, documented deterministic quantity
    for coordinate in vectors.coordinates:
        assert coordinate.name and coordinate.uses_labels is False
        if coordinate.kind == "level1":
            assert coordinate.block in DEFAULT_BLOCKS
            assert coordinate.definition and coordinate.definition != coordinate.name
            detail = coordinate.name.split(".", 1)[1]
            assert detail in DETAIL_DEFINITIONS
    assert vectors.feature_names == StructuredVectorEncoder(bank, structured_d=d).encode().feature_names


# --------------------------------------------------------------------------
# Fixed projection
# --------------------------------------------------------------------------
def test_projection_used_only_beyond_the_source_dimension(trained_like_model):
    bank = _bank(trained_like_model)
    source = StructuredVectorEncoder(bank, structured_d=1).source_dimension
    assert source == 141

    no_projection = StructuredVectorEncoder(bank, structured_d=source).encode()
    assert no_projection.provenance["projection_used"] is False
    assert not any(c.kind == "projection" for c in no_projection.coordinates)

    projected = StructuredVectorEncoder(bank, structured_d=source + 5).encode()
    spec = projected.provenance["projection"]
    assert projected.provenance["projection_used"] is True
    assert spec["seed"] == DEFAULT_PROJECTION_SEED
    assert spec["source_dimension"] == source
    assert spec["output_dimension"] == 5
    assert spec["type"] == "fixed_gaussian" and spec["learned"] is False
    assert projected.coordinates[-1].name == "projection[4]"
    assert projected.feature_names[-5:] == tuple(f"projection[{j}]" for j in range(5))
    # the interpretable source coordinates are untouched
    assert np.array_equal(projected.X[:, :source], no_projection.X)


def test_projection_seed_determinism_and_sensitivity(trained_like_model):
    bank = _bank(trained_like_model)
    source = StructuredVectorEncoder(bank, structured_d=1).source_dimension
    d = source + 8
    a = encode_structured_vectors(bank, structured_d=d, projection_seed=0)
    b = encode_structured_vectors(bank, structured_d=d, projection_seed=0)
    other_seed = encode_structured_vectors(bank, structured_d=d, projection_seed=7)
    assert np.array_equal(a.X, b.X)  # same seed + same bank -> identical
    assert np.array_equal(a.X[:, :source], other_seed.X[:, :source])  # source part is seed-independent
    assert not np.array_equal(a.X[:, source:], other_seed.X[:, source:])  # different seed -> different projection
    assert a.provenance["projection"]["seed"] == 0
    assert other_seed.provenance["projection"]["seed"] == 7
    assert a.provenance["projection_seed"] == 0


def test_projection_matrix_is_two_dimensional_and_matches_provenance(trained_like_model):
    bank = _bank(trained_like_model)
    encoder = StructuredVectorEncoder(bank, structured_d=150)
    matrix = encoder.projection_matrix()
    spec = encoder.provenance["projection"]
    assert matrix is not None and matrix.ndim == 2
    assert matrix.shape == (spec["source_dimension"], spec["output_dimension"])
    assert encoder.output_dimension == spec["source_dimension"] + spec["output_dimension"]
    # a non-projected encoder has no projection matrix at all
    assert StructuredVectorEncoder(bank, structured_d=48).projection_matrix() is None


def test_prefix_property_across_dimension_pairs(trained_like_model):
    """The coordinate system is a prefix: extending d never changes existing coordinates.

    Source coordinates (level 0 / level 1) are bit-identical; projected coordinates are
    identical up to floating-point tolerance because different matrix shapes can make
    BLAS reorder summations.
    """
    bank = _bank(trained_like_model)
    source = StructuredVectorEncoder(bank, structured_d=1).source_dimension
    dimensions = [1, 5, 48, 49, 100, source, source + 3, source + 20]
    matrices = {d: encode_structured_vectors(bank, structured_d=d).X for d in dimensions}
    for small in dimensions:
        for big in dimensions:
            if small > big:
                continue
            if small <= source:
                assert np.array_equal(matrices[big][:, :small], matrices[small]), (small, big)
            else:
                assert np.allclose(matrices[big][:, :small], matrices[small], rtol=1e-12, atol=1e-12), (small, big)


def test_projection_is_prefix_stable_in_the_output_dimension(trained_like_model):
    bank = _bank(trained_like_model)
    source = StructuredVectorEncoder(bank, structured_d=1).source_dimension
    small = encode_structured_vectors(bank, structured_d=source + 3)
    large = encode_structured_vectors(bank, structured_d=source + 20)
    assert small.feature_names == large.feature_names[: source + 3]
    assert np.allclose(large.X[:, : source + 3], small.X, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------
# Block selection
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "blocks",
    [
        ["intrinsic"],
        ["input_conn"],
        ["recurrent_in", "recurrent_out"],
        ("intrinsic", "input_conn", "recurrent_in", "recurrent_out"),
        "intrinsic, input_conn",
    ],
)
def test_block_selection_is_deterministic(trained_like_model, blocks):
    bank = _bank(trained_like_model)
    first = StructuredVectorEncoder(bank, structured_d=16, enabled_blocks=blocks).encode()
    second = StructuredVectorEncoder(bank, structured_d=16, enabled_blocks=blocks).encode()
    assert first.shape == (bank.n_neurons, 16)
    assert first.feature_names == second.feature_names
    assert np.array_equal(first.X, second.X)

    requested = [b for b in blocks] if isinstance(blocks, (list, tuple)) else blocks.split(", ")
    _, expected_names = bank.to_structured_matrix(blocks=[b for b in DEFAULT_BLOCKS if b in requested])
    shared = min(16, len(expected_names))
    assert list(first.feature_names[:shared]) == list(expected_names)[:shared]


def test_activity_block_participates_when_enabled(tiny_model, synthetic_rec):
    bank = _bank(tiny_model, with_activity=True, rec=synthetic_rec)
    blocks = ["intrinsic", "input_conn", "activity"]
    encoder = StructuredVectorEncoder(bank, structured_d=16, enabled_blocks=blocks)
    names = encoder.feature_names
    assert len(names) == 16
    assert any(name.startswith("activity.") for name in names)
    # the default block selection deliberately excludes activity (historical 48-D anchor)
    default_names = StructuredVectorEncoder(bank, structured_d=16).feature_names
    assert not any(name.startswith("activity.") for name in default_names)

    # with activity enabled the activity detail becomes available for larger d
    source = StructuredVectorEncoder(bank, structured_d=1, enabled_blocks=blocks).source_dimension
    full = StructuredVectorEncoder(bank, structured_d=source, enabled_blocks=blocks).encode()
    activity_detail = [n for n in full.feature_names if n.startswith("activity.count_")]
    assert activity_detail[:1] == ["activity." + ACTIVITY_DETAIL_NAMES[0]]


def test_unknown_or_unimplemented_blocks_raise(trained_like_model):
    bank = _bank(trained_like_model)
    for blocks in (["network_context"], ["intrinsic", "network_context"]):
        with pytest.raises(StructuredVectorError, match="not implemented"):
            StructuredVectorEncoder(bank, structured_d=16, enabled_blocks=blocks)
    # `temporal` is implemented but absent from this bank (no FIT activity): it simply
    # contributes nothing, so a temporal-only request has no features to encode
    with pytest.raises(StructuredVectorError, match="no features"):
        StructuredVectorEncoder(bank, structured_d=16, enabled_blocks=["temporal"])
    with pytest.raises(StructuredVectorError, match="unknown block"):
        StructuredVectorEncoder(bank, structured_d=16, enabled_blocks=["not_a_block"])
    with pytest.raises(StructuredVectorError, match="at least one block"):
        StructuredVectorEncoder(bank, structured_d=16, enabled_blocks=[])


def test_absent_block_contributes_nothing(tiny_model):
    # untrained model: intrinsic has no varying parameter, so an intrinsic-only request fails
    bank = _bank(tiny_model)
    assert "intrinsic" not in bank.block_names
    with pytest.raises(StructuredVectorError, match="no features"):
        StructuredVectorEncoder(bank, structured_d=8, enabled_blocks=["intrinsic"])


# --------------------------------------------------------------------------
# Neuron counts
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n_hidden", [32, 64, 128, 256])
def test_arbitrary_neuron_counts(n_hidden):
    model = _toy_model(n_hidden)
    bank = _bank(model)
    vectors = StructuredVectorEncoder(bank, structured_d=64).encode()
    assert vectors.shape == (n_hidden, 64)
    assert vectors.n_neurons == n_hidden
    assert np.array_equal(vectors.X, StructuredVectorEncoder(bank, structured_d=64).encode().X)


# --------------------------------------------------------------------------
# Chunking and memory
# --------------------------------------------------------------------------
@pytest.mark.parametrize("d", [16, 64, 160])
def test_chunked_and_unchunked_outputs_are_identical(trained_like_model, d):
    """Chunking must not change the representation (source coordinates exactly)."""
    bank = _bank(trained_like_model)
    source = StructuredVectorEncoder(bank, structured_d=1).source_dimension
    reference = StructuredVectorEncoder(bank, structured_d=d).encode()
    for chunk in (1, 3, 7, bank.n_neurons, bank.n_neurons + 5):
        chunked = StructuredVectorEncoder(bank, structured_d=d, chunk_size=chunk).encode()
        if d <= source:
            assert np.array_equal(chunked.X, reference.X), chunk
        else:
            assert np.allclose(chunked.X, reference.X, rtol=1e-12, atol=1e-12), chunk
        assert chunked.feature_names == reference.feature_names


def test_output_and_internal_arrays_are_two_dimensional(trained_like_model):
    bank = _bank(trained_like_model)
    vectors = StructuredVectorEncoder(bank, structured_d=200).encode()
    assert vectors.X.ndim == 2
    assert vectors.X.shape == (bank.n_neurons, 200)
    assert StructuredVectorEncoder(bank, structured_d=200).projection_matrix().ndim == 2
    # the source matrix never exists in a forbidden (neurons, samples, d)-style shape
    assert vectors.X.shape[0] == bank.n_neurons
    for name in bank.block_names:
        block = bank.get_block(name)
        for arr in list(block.features.values()) + list(block.flags.values()):
            assert arr.ndim == 1
        if block.weights is not None:
            assert block.weights.ndim == 2


def test_encoder_rejects_invalid_inputs_and_dimensions(trained_like_model):
    bank = _bank(trained_like_model)
    with pytest.raises(StructuredVectorError, match="NeuronRecordBank"):
        StructuredVectorEncoder(bank=None, structured_d=4)  # type: ignore[arg-type]
    for bad in (0, -1, MAX_STRUCTURED_D + 1):
        with pytest.raises(StructuredVectorError, match="structured_d"):
            StructuredVectorEncoder(bank, structured_d=bad)
    with pytest.raises(StructuredVectorError, match="chunk_size"):
        StructuredVectorEncoder(bank, structured_d=8, chunk_size=0)
    with pytest.raises(StructuredVectorError, match="projection_seed"):
        StructuredVectorEncoder(bank, structured_d=8, projection_seed=-1)


# --------------------------------------------------------------------------
# Label-free guarantees
# --------------------------------------------------------------------------
def test_encoder_is_label_free(trained_like_model, synthetic_rec):
    bank = _bank(trained_like_model, with_activity=True, rec=synthetic_rec)
    encoder = StructuredVectorEncoder(bank, structured_d=64)
    assert encoder.uses_labels is False
    assert encoder.provenance["uses_labels"] is False
    assert all(c.uses_labels is False for c in encoder.coordinates)
    assert encoder.provenance["bank"]["uses_labels"] is False
    params = set(inspect.signature(StructuredVectorEncoder).parameters)
    assert not any("label" in p.lower() or p in ("y", "probe", "test") for p in params)
    assert not any("label" in p.lower() for p in inspect.signature(encode_structured_vectors).parameters)


def test_relabelled_fit_data_cannot_change_the_encoded_vectors(tiny_model, synthetic_rec):
    bank_ref = _bank(tiny_model, with_activity=True, rec=synthetic_rec)
    bank_shift = _bank(tiny_model, with_activity=True, rec=_relabelled(synthetic_rec))
    a = StructuredVectorEncoder(bank_ref, structured_d=64).encode()
    b = StructuredVectorEncoder(bank_shift, structured_d=64).encode()
    assert a.feature_names == b.feature_names
    assert np.array_equal(a.X, b.X)


# --------------------------------------------------------------------------
# No fitted state / determinism / provenance
# --------------------------------------------------------------------------
def test_encoder_has_no_fitted_state(trained_like_model):
    bank = _bank(trained_like_model)
    encoder = StructuredVectorEncoder(bank, structured_d=64)
    assert not hasattr(encoder, "fit") and not hasattr(encoder, "train")
    before = copy.deepcopy(encoder.provenance)
    first = encoder.encode()
    second = encoder.encode()
    assert np.array_equal(first.X, second.X)
    assert encoder.provenance == before  # encoding does not mutate the encoder state
    fresh_bank = _bank(trained_like_model)  # an independent bank with identical content
    assert np.array_equal(StructuredVectorEncoder(fresh_bank, structured_d=64).encode().X, first.X)


def test_feature_metadata_and_provenance_are_complete_and_serialisable(trained_like_model):
    bank = _bank(trained_like_model)
    source = StructuredVectorEncoder(bank, structured_d=1).source_dimension
    vectors = StructuredVectorEncoder(bank, structured_d=source + 4).encode()
    assert len(vectors.feature_names) == source + 4
    metadata = vectors.metadata
    assert len(metadata) == source + 4
    assert [m["index"] for m in metadata] == list(range(source + 4))
    assert all({"name", "kind", "block", "source", "definition", "uses_labels"} <= set(m) for m in metadata)
    provenance = vectors.provenance
    for key in (
        "schema", "uses_labels", "deterministic", "fitted_state", "structured_d",
        "level0_dimension", "level1_dimension", "source_dimension", "projection",
        "projection_used", "projection_seed", "selection_policy", "bank", "warnings",
    ):
        assert key in provenance
    assert provenance["residual_implemented"] is False
    assert isinstance(json.dumps(vectors.to_dict()), str)
    assert isinstance(json.dumps(provenance), str)


# --------------------------------------------------------------------------
# V2 configuration integration
# --------------------------------------------------------------------------
def test_from_config_uses_the_structured_d_and_chunk_size(trained_like_model):
    cfg = V2Config.from_config(Config({}), warn=False)
    bank = _bank(trained_like_model)
    encoder = StructuredVectorEncoder.from_config(cfg, bank)
    assert encoder.output_dimension == cfg.vector.structured_d == 48
    assert encoder.chunk_size == cfg.memory.representation_chunk_size
    assert encoder.provenance["configured"]["vector"]["structured_d"] == 48
    assert np.array_equal(encoder.encode().X, StructuredVectorEncoder(bank, structured_d=48).encode().X)
    # explicit chunk size overrides the configuration
    assert StructuredVectorEncoder.from_config(cfg, bank, chunk_size=5).chunk_size == 5


def test_from_config_rejects_a_requested_learned_residual_by_default(trained_like_model):
    cfg = V2Config.from_config(
        Config({"vector": {"d": 64, "structured_d": 48, "learned_residual_d": 16,
                           "residual": {"enabled": True}}}),
        warn=False,
    )
    bank = _bank(trained_like_model)
    with pytest.raises(StructuredVectorError, match="not implemented"):
        StructuredVectorEncoder.from_config(cfg, bank)


def test_from_config_warn_mode_returns_only_the_structured_part(trained_like_model):
    cfg = V2Config.from_config(
        Config({"vector": {"d": 64, "structured_d": 48, "learned_residual_d": 16,
                           "residual": {"enabled": True}}}),
        warn=False,
    )
    bank = _bank(trained_like_model)
    encoder = StructuredVectorEncoder.from_config(cfg, bank, on_residual_request="warn")
    vectors = encoder.encode()
    assert vectors.shape == (bank.n_neurons, 48)  # structured part only, never a fake residual
    assert encoder.warnings and "residual" in encoder.warnings[0]
    assert encoder.provenance["warnings"] == encoder.warnings
    assert encoder.provenance["configured"]["residual_requested"] is True
    assert encoder.provenance["residual_implemented"] is False
    with pytest.raises(StructuredVectorError, match="on_residual_request"):
        StructuredVectorEncoder.from_config(cfg, bank, on_residual_request="ignore")


def test_from_config_rejects_unimplemented_blocks(trained_like_model):
    cfg = V2Config.from_config(
        Config({"vector": {"enabled_blocks": ["intrinsic", "network_context"]}}), warn=False
    )
    bank = _bank(trained_like_model)
    with pytest.raises(StructuredVectorError, match="not implemented"):
        StructuredVectorEncoder.from_config(cfg, bank)