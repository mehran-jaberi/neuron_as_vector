"""Tests for the unified label-free columnar ``NeuronRecordBank`` (``src/neuron_record.py``).

Coverage: columnar structure, deterministic feature naming/ordering, connectivity
orientation (incoming row vs outgoing column), the label-leakage contract, exact
numerical compatibility with the existing 48-D structural representation and the
existing activity representation, and the memory-contract guard.
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest
import torch

from src.data import SHDRecordings
from src.evaluation import collect_activity
from src.model import RecurrentLIFSNN, build_model
from src.neuron_record import (
    RECORD_SCHEMA,
    STRUCTURAL_BLOCKS,
    NeuronRecordBank,
    NeuronRecordError,
    RecordBlock,
    build_neuron_record_bank,
    structural_matrix_from_record_bank,
    validate_record_array,
)
from src.neurons import (
    FeatureBlock,
    activity_features_from_psth,
    add_first_spike_features,
    build_activity_representations,
    extract_structural_representations,
)
from src.utils import Config, PROJECT_ROOT, load_config
from src.v2_config import V2Config, V2ConfigError

ACTIVITY_BLOCK = FeatureBlock.ACTIVITY.value

#: locally produced training artefacts (gitignored); tested when present
CANONICAL_CHECKPOINTS = ("sweep_l2_0.pt", "nsb_seed1.pt", "nsb_seed2.pt")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _unique_weight_model(cfg):
    """A model whose w_in / w_rec contain unique values, so transposes are unambiguous."""
    model = build_model(cfg, seed=0)
    with torch.no_grad():
        w_in = (torch.arange(cfg.n_input * cfg.n_hidden, dtype=torch.float32).reshape(cfg.n_input, cfg.n_hidden) + 1.0) / 1000.0
        w_rec = (torch.arange(cfg.n_hidden * cfg.n_hidden, dtype=torch.float32).reshape(cfg.n_hidden, cfg.n_hidden) + 1.0) / 1000.0
        model.w_in.copy_(w_in)
        model.w_rec.copy_(w_rec)
    return model, w_in.numpy().astype(np.float64), w_rec.numpy().astype(np.float64)


def _activity_result(model, rec, batch_size: int = 32, with_labels: bool = False):
    idx = np.arange(len(rec), dtype=np.int64)
    return collect_activity(
        model, rec, idx, device="cpu", batch_size=batch_size,
        n_classes=int(model.cfg.n_output), with_labels=with_labels, collect_voltage=True,
    )


def _relabelled(rec: SHDRecordings, shift: int = 1) -> SHDRecordings:
    labels = (rec.labels_array + shift) % max(int(rec.labels_array.max()) + 1, 1)
    return SHDRecordings(
        times_ms=rec.times_ms,
        units=rec.units,
        offsets=rec.offsets,
        labels=labels,
        n_channels=rec.n_channels,
        speakers=rec.speakers,
        name=rec.name,
        meta=dict(rec.meta),
    )


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------
def test_bank_is_columnar_and_indexed_by_neuron(trained_like_model, tiny_snn_config):
    bank = build_neuron_record_bank(trained_like_model, with_activity=False)
    assert bank.n_neurons == tiny_snn_config.n_hidden
    assert bank.block_names == ("intrinsic", "input_conn", "recurrent_in", "recurrent_out")
    # features are arrays with a leading neuron axis, not per-neuron objects
    for name in bank.block_names:
        block = bank.get_block(name)
        assert isinstance(block, RecordBlock)
        for arr in block.features.values():
            assert isinstance(arr, np.ndarray) and arr.shape == (bank.n_neurons,)
    assert not hasattr(bank, "neurons")  # no primary per-neuron object collection


def test_connectivity_block_shapes(trained_like_model, tiny_snn_config):
    bank = build_neuron_record_bank(trained_like_model, with_activity=False)
    n = tiny_snn_config.n_hidden
    assert bank.get_block("input_conn").weights.shape == (n, tiny_snn_config.n_input)
    assert bank.get_block("recurrent_in").weights.shape == (n, n)
    assert bank.get_block("recurrent_out").weights.shape == (n, n)


def test_block_feature_counts_match_the_current_representation(trained_like_model):
    bank = build_neuron_record_bank(trained_like_model, with_activity=False)
    assert bank.get_block("intrinsic").n_features == 1  # learned_bias
    assert bank.get_block("input_conn").n_features == 14
    assert bank.get_block("recurrent_in").n_features == 19  # 14 vector + 5 relationship
    assert bank.get_block("recurrent_out").n_features == 14
    assert len(bank.feature_names) == 48


def test_feature_names_are_qualified_and_deterministic(trained_like_model):
    bank_a = build_neuron_record_bank(trained_like_model, with_activity=False)
    bank_b = build_neuron_record_bank(trained_like_model, with_activity=False)
    assert bank_a.feature_names == bank_b.feature_names
    assert all("." in name for name in bank_a.feature_names)
    expected = tuple(
        f"{b}.{f}" for b in bank_a.block_names for f in bank_a.get_block(b).feature_names
    )
    assert bank_a.feature_names == expected


def test_feature_ordering_is_alphabetical_within_each_block(trained_like_model):
    bank = build_neuron_record_bank(trained_like_model, with_activity=False)
    for name in bank.block_names:
        names = bank.get_block(name).feature_names
        assert list(names) == sorted(names)


def test_stored_arrays_are_read_only(trained_like_model):
    bank = build_neuron_record_bank(trained_like_model, with_activity=False)
    block = bank.get_block("input_conn")
    arr = block.features["mean"]
    assert arr.flags.writeable is False
    with pytest.raises(ValueError):
        arr[0] = 123.0  # assignment to read-only array
    with pytest.raises(TypeError):
        block.features["new_feature"] = arr  # mapping is immutable


# --------------------------------------------------------------------------
# Connectivity orientation
# --------------------------------------------------------------------------
def test_input_conn_orientation_is_column_per_neuron(tiny_snn_config):
    model, w_in, _w_rec = _unique_weight_model(tiny_snn_config)
    bank = build_neuron_record_bank(model, with_activity=False)
    weights = bank.get_block("input_conn").weights
    assert np.array_equal(weights, w_in.T)
    for j in (0, 3, tiny_snn_config.n_hidden - 1):
        assert np.array_equal(weights[j], w_in[:, j])


def test_recurrent_in_is_row_and_recurrent_out_is_column(tiny_snn_config):
    model, _w_in, w_rec = _unique_weight_model(tiny_snn_config)
    bank = build_neuron_record_bank(model, with_activity=False)
    incoming = bank.get_block("recurrent_in").weights
    outgoing = bank.get_block("recurrent_out").weights

    assert np.array_equal(incoming, w_rec)  # row i = w_rec[i, :] = incoming onto i
    assert np.array_equal(outgoing, w_rec.T)  # row i = w_rec[:, i] = outgoing from i
    assert np.array_equal(outgoing, incoming.T)
    n = tiny_snn_config.n_hidden
    for i in (0, 1, n // 2, n - 1):
        assert np.array_equal(incoming[i], w_rec[i, :])
        assert np.array_equal(outgoing[i], w_rec[:, i])
        assert not np.array_equal(incoming[i], outgoing[i])  # unique values -> never ambiguous


def test_incoming_orientation_matches_the_forward_pass(tiny_snn_config):
    """Grounding: forward computes s_prev @ w_rec.T, so neuron i receives row i."""
    model, _w_in, _w_rec = _unique_weight_model(tiny_snn_config)
    n = tiny_snn_config.n_hidden
    w_rec32 = model.w_rec.detach().cpu().numpy()  # float32 values as stored in the model
    for j0 in (0, 5, n - 1):
        s_prev = torch.zeros(1, n)
        s_prev[0, j0] = 1.0
        recurrent_input = (s_prev @ model.w_rec.t()).detach()
        for i in (0, 3, n - 1):
            assert float(recurrent_input[0, i]) == pytest.approx(float(w_rec32[i, j0]), rel=0, abs=1e-12)
    bank = build_neuron_record_bank(model, with_activity=False)
    assert "ROW" in bank.get_block("recurrent_in").orientation
    assert "COLUMN" in bank.get_block("recurrent_out").orientation
    assert "w_in[c, j]" in bank.get_block("input_conn").orientation


# --------------------------------------------------------------------------
# Label-leakage contract
# --------------------------------------------------------------------------
def test_bank_is_label_free_by_construction(trained_like_model):
    bank = build_neuron_record_bank(trained_like_model, with_activity=False)
    assert bank.uses_labels is False
    assert bank.provenance["uses_labels"] is False
    assert all(status["uses_labels"] is False for status in bank.block_status().values())
    with pytest.raises(AttributeError):
        bank.uses_labels = True  # read-only property
    with pytest.raises(TypeError):
        NeuronRecordBank(n_neurons=2, blocks={}, uses_labels=True)  # not a constructor argument
    with pytest.raises(NeuronRecordError, match="at least one block"):
        NeuronRecordBank(n_neurons=2, blocks={})


def test_builder_accepts_no_labels_argument():
    params = set(inspect.signature(build_neuron_record_bank).parameters)
    assert not any("label" in p.lower() for p in params)
    assert not any(p in ("y", "probe", "test", "fingerprint") for p in params)
    assert not any("probe" in p.lower() or "test" in p.lower() for p in params)
    assert "fit_idx" in params and "fit_rec" in params  # indices and recordings only
    assert "activity" in params  # or a pre-collected label-free accumulator


def test_scrambled_labels_cannot_change_the_record(tiny_model, synthetic_rec):
    bank_ref = build_neuron_record_bank(tiny_model, fit_rec=synthetic_rec, device="cpu", batch_size=32)
    bank_shift = build_neuron_record_bank(
        tiny_model, fit_rec=_relabelled(synthetic_rec, shift=2), device="cpu", batch_size=32
    )
    assert bank_ref.block_names == bank_shift.block_names
    assert bank_ref.feature_names == bank_shift.feature_names
    X_ref, _ = bank_ref.to_structured_matrix()
    X_shift, _ = bank_shift.to_structured_matrix()
    assert np.array_equal(X_ref, X_shift)
    for name in bank_ref.block_names:
        if bank_ref.get_block(name).weights is not None:
            assert np.array_equal(
                bank_ref.get_block(name).weights, bank_shift.get_block(name).weights
            )


def test_labelled_accumulator_is_rejected(tiny_model, synthetic_rec):
    labelled = _activity_result(tiny_model, synthetic_rec, with_labels=True)
    with pytest.raises(NeuronRecordError, match="label-derived"):
        build_neuron_record_bank(tiny_model, activity=labelled)


def test_activity_block_declares_label_free_provenance(tiny_model, synthetic_rec):
    result = _activity_result(tiny_model, synthetic_rec, with_labels=False)
    bank = build_neuron_record_bank(tiny_model, activity=result, device="cpu")
    block = bank.get_block(ACTIVITY_BLOCK)
    assert block.metadata["split"] == "supplied_accumulator"
    assert bank.provenance["activity"]["label_free"] is True
    assert "never read" in bank.provenance["leakage_contract"]


# --------------------------------------------------------------------------
# Compatibility with the existing 48-D structural representation
# --------------------------------------------------------------------------
def _assert_matches_existing(model, *, include_tonotopic: bool = False, expected_features: int):
    old = extract_structural_representations(model, include_tonotopic=include_tonotopic)
    X_old, names_old = old.to_matrix(blocks=STRUCTURAL_BLOCKS)
    bank = build_neuron_record_bank(model, with_activity=False, include_tonotopic=include_tonotopic)
    X_new, names_new = structural_matrix_from_record_bank(bank)
    assert names_new == names_old
    assert X_new.shape == X_old.shape == (model.cfg.n_hidden, expected_features)
    assert float(np.abs(X_new - X_old).max()) <= 1e-12


def test_structural_matrix_matches_existing_representation(trained_like_model):
    _assert_matches_existing(trained_like_model, expected_features=48)


def test_structural_matrix_matches_for_untrained_model(tiny_model):
    # no varying intrinsic parameter -> no intrinsic block (existing rule)
    bank = build_neuron_record_bank(tiny_model, with_activity=False)
    assert "intrinsic" not in bank.block_names
    _assert_matches_existing(tiny_model, expected_features=47)


def test_structural_matrix_matches_for_dynamical_model(dynamical_model):
    # bias_tau mode emits BOTH tau_mem_ms (dynamical) and learned_bias (generic) -> 49 features
    bank = build_neuron_record_bank(dynamical_model, with_activity=False)
    assert bank.get_block("intrinsic").feature_names == ("learned_bias", "tau_mem_ms")
    _assert_matches_existing(dynamical_model, expected_features=49)


def test_structural_matrix_matches_with_tonotopic_features(trained_like_model):
    _assert_matches_existing(trained_like_model, include_tonotopic=True, expected_features=50)


def test_to_representation_set_reproduces_existing_object_matrix(trained_like_model):
    old = extract_structural_representations(trained_like_model)
    X_old, names_old = old.to_matrix()
    bank = build_neuron_record_bank(trained_like_model, with_activity=False)
    X_new, names_new = bank.to_representation_set().to_matrix()
    assert names_new == names_old
    assert float(np.abs(X_new - X_old).max()) <= 1e-12


def test_structurally_equal_across_chunk_sizes(trained_like_model):
    bank = build_neuron_record_bank(trained_like_model, with_activity=False)
    X_all, names_all = structural_matrix_from_record_bank(bank)
    for chunk in (1, 7, bank.n_neurons):
        X, names = structural_matrix_from_record_bank(bank, chunk_size=chunk)
        assert names == names_all
        assert np.array_equal(X, X_all)


def test_block_selection_matches_existing_matrix(trained_like_model):
    old = extract_structural_representations(trained_like_model)
    bank = build_neuron_record_bank(trained_like_model, with_activity=False)
    for blocks in (["input_conn"], ["recurrent_in", "recurrent_out"], "intrinsic, input_conn"):
        X_old, names_old = old.to_matrix(blocks=blocks)
        X_new, names_new = bank.to_structured_matrix(blocks=blocks)
        assert names_new == names_old
        assert float(np.abs(X_new - X_old).max()) <= 1e-12


# --------------------------------------------------------------------------
# Compatibility on the canonical 256-neuron architecture
# --------------------------------------------------------------------------
def _canonical_architecture_model():
    """The canonical experiment architecture (700 x 2 ms, 256 hidden) from its config."""
    cfg = load_config(PROJECT_ROOT / "configs/neuron_space_baseline.yaml")
    model = build_model(cfg, seed=0)
    with torch.no_grad():
        # mimic a trained network: a varying per-neuron bias makes the intrinsic block exist
        model.b_hid.copy_(torch.linspace(-0.5, 0.5, model.cfg.n_hidden))
    return model


def test_canonical_256_neuron_model_matches_existing_representation():
    model = _canonical_architecture_model()
    assert model.cfg.n_hidden == 256 and model.cfg.n_input == 700 and model.cfg.n_bins == 700

    old = extract_structural_representations(model)
    X_old, names_old = old.to_matrix()
    bank = build_neuron_record_bank(model, with_activity=False)

    assert bank.n_neurons == 256
    assert bank.get_block("input_conn").weights.shape == (256, 700)
    assert bank.get_block("recurrent_in").weights.shape == (256, 256)
    assert bank.get_block("recurrent_out").weights.shape == (256, 256)

    X_new, names_new = bank.to_structured_matrix()
    assert names_new == names_old
    assert X_new.shape == X_old.shape == (256, 48)  # the whole 48-D matrix, not summaries
    assert float(np.abs(X_new - X_old).max()) <= 1e-12


@pytest.mark.parametrize("name", CANONICAL_CHECKPOINTS)
def test_canonical_trained_checkpoints_match_existing_representation(name):
    path = PROJECT_ROOT / "checkpoints" / name
    if not path.exists():
        pytest.skip(f"{name} is a local training artefact and is not present")
    model, _extra = RecurrentLIFSNN.load(str(path), map_location="cpu")
    old = extract_structural_representations(model)
    X_old, names_old = old.to_matrix()
    bank = NeuronRecordBank.from_checkpoint(path, with_activity=False)
    X_new, names_new = bank.to_structured_matrix()
    assert names_new == names_old
    assert X_new.shape == X_old.shape == (model.cfg.n_hidden, 48)
    assert float(np.abs(X_new - X_old).max()) <= 1e-12


# --------------------------------------------------------------------------
# Compatibility with the existing activity representation
# --------------------------------------------------------------------------
def _existing_activity_matrix(model, result):
    features, flags = activity_features_from_psth(
        result.psth, result.counts, bin_ms=result.bin_ms, n_samples=result.n_samples
    )
    features = add_first_spike_features(
        features, result.first_spike_sum, result.first_spike_count, duration_ms=result.duration_ms
    )
    reps = build_activity_representations(features, flags)
    return reps.to_matrix(blocks=[ACTIVITY_BLOCK])


def test_activity_block_matches_existing_activity_representation(tiny_model, synthetic_rec):
    result = _activity_result(tiny_model, synthetic_rec, with_labels=False)
    X_old, names_old = _existing_activity_matrix(tiny_model, result)
    bank = build_neuron_record_bank(tiny_model, activity=result, device="cpu")
    X_new, names_new = bank.to_structured_matrix(blocks=[ACTIVITY_BLOCK])
    assert names_new == names_old
    assert X_new.shape == X_old.shape == (tiny_model.cfg.n_hidden, 12)
    assert float(np.abs(X_new - X_old).max()) <= 1e-12


def test_activity_collected_from_fit_matches_supplied_accumulator(tiny_model, synthetic_rec):
    supplied = _activity_result(tiny_model, synthetic_rec, with_labels=False)
    bank_supplied = build_neuron_record_bank(tiny_model, activity=supplied, device="cpu")
    bank_collected = build_neuron_record_bank(
        tiny_model, fit_rec=synthetic_rec, device="cpu", batch_size=32
    )
    assert bank_collected.feature_names == bank_supplied.feature_names
    X_a, _ = bank_supplied.to_structured_matrix()
    X_b, _ = bank_collected.to_structured_matrix()
    assert float(np.abs(X_a - X_b).max()) <= 1e-12


def test_activity_flags_match_existing_metadata(tiny_model, synthetic_rec):
    result = _activity_result(tiny_model, synthetic_rec, with_labels=False)
    _, flags = activity_features_from_psth(
        result.psth, result.counts, bin_ms=result.bin_ms, n_samples=result.n_samples
    )
    bank = build_neuron_record_bank(tiny_model, activity=result, device="cpu")
    activity = bank.get_block(ACTIVITY_BLOCK)
    assert activity.flags["silent_neuron"].dtype == bool
    assert np.array_equal(activity.flags["silent_neuron"], np.asarray(flags["silent_neuron"], dtype=bool))
    assert np.array_equal(activity.flags["total_spikes"], np.asarray(flags["total_spikes"], dtype=np.float64))
    reps = bank.to_representation_set(blocks=[ACTIVITY_BLOCK])
    assert reps[0].metadata["silent_neuron"] == bool(flags["silent_neuron"][0])
    assert reps[0].metadata["total_spikes"] == float(flags["total_spikes"][0])


def test_activity_sample_counts_shape_and_dtype(tiny_model, synthetic_rec):
    bank = build_neuron_record_bank(tiny_model, fit_rec=synthetic_rec, device="cpu", batch_size=24)
    samples = bank.get_block(ACTIVITY_BLOCK).samples
    assert samples.shape == (len(synthetic_rec), tiny_model.cfg.n_hidden)
    assert samples.dtype == np.float32
    bank_no_counts = build_neuron_record_bank(
        tiny_model, fit_rec=synthetic_rec, device="cpu", batch_size=24, store_sample_counts=False
    )
    assert bank_no_counts.get_block(ACTIVITY_BLOCK).samples is None


def test_full_matrix_has_the_current_full_dimension(trained_like_model, synthetic_rec):
    bank = build_neuron_record_bank(trained_like_model, fit_rec=synthetic_rec, device="cpu", batch_size=32)
    X, names = bank.to_structured_matrix()
    assert X.shape == (trained_like_model.cfg.n_hidden, 60)  # 48 structural + 12 activity
    assert len(names) == 60


# --------------------------------------------------------------------------
# Memory contract
# --------------------------------------------------------------------------
def test_forbidden_tensor_shapes_are_rejected():
    with pytest.raises(NeuronRecordError, match="forbidden shape"):
        RecordBlock(name="x", n_neurons=16, features={"f": np.zeros((16, 120, 30))})
    with pytest.raises(NeuronRecordError, match="forbidden shape"):
        RecordBlock(name="x", n_neurons=16, weights=np.zeros((16, 16, 64)))
    with pytest.raises(NeuronRecordError, match="one per neuron"):
        RecordBlock(name="x", n_neurons=16, features={"f": np.zeros(120)})
    with pytest.raises(NeuronRecordError, match="sample-major"):
        RecordBlock(name="x", n_neurons=16, features={"f": np.zeros((120, 16))})
    with pytest.raises(NeuronRecordError, match="scalar arrays"):
        RecordBlock(name="x", n_neurons=16, features={"f": np.float64(1.0)})


def test_sample_major_counts_orientation_is_enforced():
    good = RecordBlock(name="activity", n_neurons=16, samples=np.zeros((120, 16), dtype=np.float32))
    assert good.samples.shape == (120, 16)
    with pytest.raises(NeuronRecordError):
        RecordBlock(name="activity", n_neurons=16, samples=np.zeros((16, 120)))  # (n, samples) forbidden


def test_validate_record_array_helper():
    ok = validate_record_array(np.zeros(16), "f", 16)
    assert ok.shape == (16,)
    validate_record_array(np.zeros((16, 5)), "w", 16)
    validate_record_array(np.zeros((100, 16)), "counts", 16, allow_sample_major=True)
    with pytest.raises(NeuronRecordError):
        validate_record_array(np.zeros((100, 16)), "f", 16)  # sample-major not allowed here
    with pytest.raises(NeuronRecordError):
        validate_record_array(np.zeros((16, 4, 2)), "f", 16)


def test_bank_has_no_tensor_with_more_than_two_dimensions(trained_like_model, synthetic_rec):
    bank = build_neuron_record_bank(trained_like_model, fit_rec=synthetic_rec, device="cpu", batch_size=32)
    assert bank.assert_no_forbidden_tensors() is True
    for name in bank.block_names:
        block = bank.get_block(name)
        arrays = list(block.features.values()) + list(block.flags.values())
        if block.weights is not None:
            arrays.append(block.weights)
        if block.samples is not None:
            arrays.append(block.samples)
        assert all(arr.ndim <= 2 for arr in arrays)


def test_gpu_storage_config_is_rejected(trained_like_model):
    cfg = V2Config.from_config(Config({"memory": {"storage": "gpu", "device": "cuda"}}), warn=False)
    with pytest.raises(NeuronRecordError, match="GPU storage"):
        build_neuron_record_bank(trained_like_model, with_activity=False, config=cfg)


def test_input_token_budget_is_checked_before_the_activity_pass(tiny_model, synthetic_rec):
    cfg = V2Config.from_config(Config({"memory": {"max_input_tokens": 1000}}), warn=False)
    with pytest.raises(V2ConfigError, match="too large"):
        build_neuron_record_bank(tiny_model, fit_rec=synthetic_rec, device="cpu", config=cfg)


def test_record_batch_size_defaults_to_the_config(tiny_model, synthetic_rec):
    cfg = V2Config.from_config(Config({"memory": {"record_batch_size": 16}}), warn=False)
    bank = build_neuron_record_bank(tiny_model, fit_rec=synthetic_rec, device="cpu", config=cfg)
    assert bank.provenance["activity"]["batch_size"] == 16
    bank_override = build_neuron_record_bank(
        tiny_model, fit_rec=synthetic_rec, device="cpu", config=cfg, batch_size=8
    )
    assert bank_override.provenance["activity"]["batch_size"] == 8


# --------------------------------------------------------------------------
# Declared-but-unimplemented blocks
# --------------------------------------------------------------------------
def test_temporal_and_network_context_are_reported_unimplemented(trained_like_model):
    bank = build_neuron_record_bank(trained_like_model, with_activity=False)
    assert bank.unimplemented_blocks == ("temporal", "network_context")
    status = bank.block_status()
    assert status["temporal"]["implemented"] is False
    assert "not implemented" in status["temporal"]["reason"]
    with pytest.raises(NeuronRecordError, match="not implemented"):
        bank.get_block("temporal")
    with pytest.raises(NeuronRecordError, match="not implemented"):
        bank.to_structured_matrix(blocks=["temporal"])
    with pytest.raises(NeuronRecordError, match="unknown block"):
        bank.get_block("not_a_block")


def test_activity_block_absent_without_fit_data(trained_like_model):
    bank = build_neuron_record_bank(trained_like_model, with_activity=False)
    assert ACTIVITY_BLOCK not in bank.block_names
    with pytest.raises(NeuronRecordError):
        bank.get_block(ACTIVITY_BLOCK)
    assert bank.block_status()[ACTIVITY_BLOCK]["implemented"] is True
    assert bank.block_status()[ACTIVITY_BLOCK]["present"] is False


def test_passing_both_activity_sources_is_rejected(trained_like_model, synthetic_rec):
    result = _activity_result(trained_like_model, synthetic_rec)
    with pytest.raises(NeuronRecordError, match="not both"):
        build_neuron_record_bank(trained_like_model, activity=result, fit_rec=synthetic_rec)


# --------------------------------------------------------------------------
# Provenance / API
# --------------------------------------------------------------------------
def test_provenance_is_machine_readable(trained_like_model, synthetic_rec, tiny_snn_config):
    bank = build_neuron_record_bank(trained_like_model, fit_rec=synthetic_rec, device="cpu", batch_size=32)
    prov = bank.provenance
    assert prov["schema"] == RECORD_SCHEMA
    assert prov["uses_labels"] is False
    assert prov["model"]["n_hidden"] == tiny_snn_config.n_hidden
    assert prov["model"]["n_input"] == tiny_snn_config.n_input
    assert "recurrent_in.weights[i]" in prov["weight_orientation"]
    assert prov["blocks"]["activity"]["uses_labels"] is False
    assert set(prov["unimplemented_blocks"]) == {"temporal", "network_context"}
    assert isinstance(json.dumps(prov), str)  # machine-readable / serialisable


def test_provenance_records_v2_config_when_supplied(trained_like_model, synthetic_rec):
    cfg = V2Config.from_config(Config({}), warn=False)
    bank = build_neuron_record_bank(
        trained_like_model, fit_rec=synthetic_rec, device="cpu", config=cfg, batch_size=32
    )
    snapshot = bank.provenance["v2_config"]
    assert snapshot["vector"]["d"] == cfg.vector.d
    assert snapshot["memory"]["record_batch_size"] == cfg.memory.record_batch_size


def test_summary_reports_blocks_and_status(trained_like_model):
    bank = build_neuron_record_bank(trained_like_model, with_activity=False)
    summary = bank.summary()
    assert summary["n_neurons"] == trained_like_model.cfg.n_hidden
    assert summary["uses_labels"] is False
    assert set(summary["blocks"]) == set(bank.block_names)
    assert summary["blocks"]["recurrent_in"]["n_weights"] == trained_like_model.cfg.n_hidden
    assert "temporal" in summary["block_status"]


def test_from_model_and_from_checkpoint_agree(trained_like_model, tmp_path):
    path = tmp_path / "tiny.pt"
    trained_like_model.save(str(path))
    from_model = build_neuron_record_bank(trained_like_model, with_activity=False)
    from_ckpt = NeuronRecordBank.from_checkpoint(path, with_activity=False)
    X_a, names_a = from_model.to_structured_matrix()
    X_b, names_b = from_ckpt.to_structured_matrix()
    assert names_a == names_b
    assert float(np.abs(X_a - X_b).max()) == 0.0


def test_record_bank_is_deterministic(trained_like_model, synthetic_rec):
    a = build_neuron_record_bank(trained_like_model, fit_rec=synthetic_rec, device="cpu", batch_size=32)
    b = build_neuron_record_bank(trained_like_model, fit_rec=synthetic_rec, device="cpu", batch_size=32)
    assert a.feature_names == b.feature_names
    X_a, _ = a.to_structured_matrix()
    X_b, _ = b.to_structured_matrix()
    assert np.array_equal(X_a, X_b)
    assert np.array_equal(
        a.get_block("input_conn").weights, b.get_block("input_conn").weights
    )
