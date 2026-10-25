"""Tests for the label-free learned residual (``src/residual.py``).

Coverage: the deterministic FIT-only source view, label-free guarantees, masking policy,
architecture and configuration validation, training/logging, reproducibility, persistence
with incompatible-schema rejection, precision fallback and memory guards.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data import SHDRecordings
from src.model import SNNConfig, build_model
from src.neuron_record import NeuronRecordBank, build_neuron_record_bank
from src.residual import (
    EVAL_MASK_OFFSET,
    RESIDUAL_SCHEMA,
    SOURCE_SCHEMA,
    ReconstructionDecoder,
    ResidualEncoder,
    ResidualError,
    ResidualResult,
    ResidualSource,
    ResidualSourceConfig,
    ResidualTrainer,
    ResidualTrainingConfig,
    apply_mask,
    build_residual_source,
    feature_schema_hash,
    make_mask,
    masked_reconstruction_loss,
    train_residual,
)
from src.utils import Config
from src.v2_config import V2Config


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _tiny_config(**overrides) -> ResidualTrainingConfig:
    base = {"residual_dim": 8, "hidden_dim": 16, "epochs": 12, "batch_size": 8}
    base.update(overrides)
    return ResidualTrainingConfig.from_mapping(base)


def _bank(model=None) -> NeuronRecordBank:
    if model is None:
        model = _tiny_model()
    return build_neuron_record_bank(model, with_activity=False)


def _tiny_model(n_hidden: int = 16, *, seed: int = 1):
    cfg = SNNConfig(n_input=20, n_hidden=n_hidden, n_output=4, n_bins=30, bin_ms=2.0)
    model = build_model(cfg, seed=seed)
    with torch.no_grad():
        model.b_hid.copy_(torch.linspace(-0.5, 0.5, n_hidden))
    return model


def _relabelled(rec: SHDRecordings, shift: int = 1) -> SHDRecordings:
    labels = (rec.labels_array + shift) % max(int(rec.labels_array.max()) + 1, 1)
    return SHDRecordings(times_ms=rec.times_ms, units=rec.units, offsets=rec.offsets, labels=labels,
                         n_channels=rec.n_channels, speakers=rec.speakers, name=rec.name,
                         meta=dict(rec.meta))


# --------------------------------------------------------------------------
# Source view
# --------------------------------------------------------------------------
def test_source_shape_names_and_determinism():
    bank = _bank()
    a = build_residual_source(bank)
    b = build_residual_source(bank)
    assert a.X.ndim == 2 and a.X.shape[0] == bank.n_neurons
    assert a.feature_names == b.feature_names
    assert np.array_equal(a.X, b.X)
    assert a.schema_hash == b.schema_hash == feature_schema_hash(a.feature_names)
    assert len(set(a.feature_names)) == a.n_features


def test_source_contains_level0_level1_and_raw_views():
    bank = _bank()
    source = build_residual_source(bank)
    provenance = source.provenance
    assert provenance["schema"] == SOURCE_SCHEMA
    assert provenance["level0_dimension"] == 48
    assert provenance["level1_dimension"] == 93
    assert set(provenance["raw_view_dimensions"]) == {"input_conn", "recurrent_in", "recurrent_out"}
    assert any(name.endswith("raw_view_00") for name in source.feature_names)
    assert "intrinsic.learned_bias" in source.feature_names
    assert "input_conn.abs_q90" in source.feature_names


@pytest.mark.parametrize(
    "config,expect_raw",
    [
        (ResidualSourceConfig(include_level1=False), True),
        (ResidualSourceConfig(include_level0=False), True),
        (ResidualSourceConfig(include_raw_weight_view=False), False),
        (ResidualSourceConfig(raw_view_dim=4), True),
    ],
)
def test_source_config_variants(config, expect_raw):
    bank = _bank()
    source = build_residual_source(bank, config)
    assert any("raw_view_" in n for n in source.feature_names) is expect_raw
    assert source.X.ndim == 2
    if not config.include_level0:
        assert "intrinsic.learned_bias" not in source.feature_names
    if not config.include_level1:
        assert "input_conn.abs_q90" not in source.feature_names
    if config.raw_view_dim < 32:
        assert sum("raw_view_" in n for n in source.feature_names) == 4 * 3  # 4 columns x 3 weighted blocks


def test_source_config_validation():
    with pytest.raises(ResidualError, match="at least one"):
        ResidualSourceConfig(include_level0=False, include_level1=False, include_raw_weight_view=False)
    with pytest.raises(ResidualError, match="raw_view_dim"):
        ResidualSourceConfig(raw_view_dim=0)


def test_source_rejects_unknown_blocks_and_non_bank_inputs():
    bank = _bank()
    with pytest.raises(ResidualError, match="unknown block"):
        build_residual_source(bank, ResidualSourceConfig(enabled_blocks=["nope"]))
    # `temporal` is implemented, but this bank carries no temporal block (no FIT activity),
    # so requesting it as a residual source is an explicit error - not a silent skip
    with pytest.raises(ResidualError, match="temporal"):
        build_residual_source(bank, ResidualSourceConfig(include_temporal=True))
    with pytest.raises(ResidualError, match="functional-response|activity"):
        build_residual_source(bank, ResidualSourceConfig(include_functional_response=True))
    with pytest.raises(ResidualError, match="NeuronRecordBank"):
        build_residual_source(np.zeros((4, 4)))  # type: ignore[arg-type]


def test_source_rejects_probe_split_banks(tiny_model, synthetic_rec):
    probe_named = SHDRecordings(
        times_ms=synthetic_rec.times_ms, units=synthetic_rec.units, offsets=synthetic_rec.offsets,
        labels=synthetic_rec.labels_array, n_channels=synthetic_rec.n_channels,
        speakers=synthetic_rec.speakers, name="probe", meta={},
    )
    bank = build_neuron_record_bank(tiny_model, fit_rec=probe_named, device="cpu", batch_size=32)
    assert bank.provenance["activity"]["split"] == "probe"
    with pytest.raises(ResidualError, match="probe"):
        build_residual_source(bank)


def test_source_rejects_forbidden_shapes_and_non_finite_values():
    with pytest.raises(ResidualError, match="2-D"):
        ResidualSource(X=np.zeros((2, 3, 4)), feature_names=("a",))
    with pytest.raises(ResidualError, match="non-finite"):
        ResidualSource(X=np.array([[1.0, np.inf]]), feature_names=("a", "b"))
    with pytest.raises(ResidualError, match="unique"):
        ResidualSource(X=np.zeros((2, 2)), feature_names=("a", "a"))
    with pytest.raises(ResidualError, match="columns"):
        ResidualSource(X=np.zeros((2, 3)), feature_names=("a",))


# --------------------------------------------------------------------------
# Label-free guarantees
# --------------------------------------------------------------------------
def test_no_labels_parameter_anywhere():
    for callable_object in (
        build_residual_source,
        train_residual,
        ResidualTrainer.fit,
        ResidualSourceConfig,
        ResidualTrainingConfig,
        ResidualResult.encode,
    ):
        params = set(inspect.signature(callable_object).parameters)
        assert not any("label" in p.lower() for p in params), params
        assert not any(p in ("y", "probe", "test", "fingerprint") for p in params), params


def test_provenance_declares_label_free_source_and_training():
    bank = _bank()
    source = build_residual_source(bank)
    result = train_residual(bank, config=_tiny_config())
    assert source.provenance["uses_labels"] is False
    assert source.provenance["fit_only"] is True
    assert result.provenance["uses_labels"] is False
    assert result.provenance["split"]["probe_or_test_used"] is False
    assert result.provenance["snn_checkpoint_modified"] is False
    assert result.provenance["objective"] == "masked_reconstruction_mse_on_withheld_coordinates"
    assert result.uses_labels is False


def test_relabelled_fit_data_cannot_change_source_or_residual(tiny_model, synthetic_rec):
    bank_ref = build_neuron_record_bank(tiny_model, fit_rec=synthetic_rec, device="cpu", batch_size=32)
    bank_shift = build_neuron_record_bank(
        tiny_model, fit_rec=_relabelled(synthetic_rec, shift=3), device="cpu", batch_size=32
    )
    source_ref = build_residual_source(bank_ref)
    source_shift = build_residual_source(bank_shift)
    assert source_ref.feature_names == source_shift.feature_names
    assert np.array_equal(source_ref.X, source_shift.X)
    config = _tiny_config()
    z_ref = train_residual(bank_ref, config=config).encode(bank_ref)
    z_shift = train_residual(bank_shift, config=config).encode(bank_shift)
    assert np.array_equal(z_ref, z_shift)


# --------------------------------------------------------------------------
# Masking
# --------------------------------------------------------------------------
def test_make_mask_is_deterministic_and_seed_sensitive():
    kwargs = dict(mask_fraction=0.25, minimum_visible_features=4)
    a = make_mask(6, 20, rng=np.random.default_rng([0, 1]), **kwargs)
    b = make_mask(6, 20, rng=np.random.default_rng([0, 1]), **kwargs)
    c = make_mask(6, 20, rng=np.random.default_rng([0, 2]), **kwargs)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.shape == (6, 20) and a.dtype == bool


def test_make_mask_respects_fraction_and_minimum_visible():
    mask = make_mask(10, 40, mask_fraction=0.25, minimum_visible_features=8, rng=np.random.default_rng(0))
    realised = mask.sum(axis=1)
    assert set(realised.tolist()) == {10}  # round(0.25 * 40)
    assert ((~mask).sum(axis=1) >= 8).all()
    tight = make_mask(3, 20, mask_fraction=0.9, minimum_visible_features=8, rng=np.random.default_rng(1))
    assert ((~tight).sum(axis=1) >= 8).all()  # clamps so that the minimum stays visible
    assert (tight.sum(axis=1) >= 1).all()


def test_make_mask_validation():
    with pytest.raises(ResidualError, match="minimum_visible_features"):
        make_mask(2, 10, mask_fraction=0.5, minimum_visible_features=10, rng=np.random.default_rng(0))
    with pytest.raises(ResidualError, match="mask_fraction"):
        make_mask(2, 10, mask_fraction=1.0, minimum_visible_features=2, rng=np.random.default_rng(0))
    with pytest.raises(ResidualError, match="shape"):
        make_mask(0, 10, mask_fraction=0.5, minimum_visible_features=2, rng=np.random.default_rng(0))


def test_apply_mask_withholds_only_masked_coordinates():
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    mask = np.zeros((3, 4), dtype=bool)
    mask[0, 1] = True
    mask[2, 3] = True
    masked = apply_mask(x, mask)
    assert masked[0, 1].item() == 0.0 and masked[2, 3].item() == 0.0
    visible = ~torch.as_tensor(mask)
    assert torch.equal(masked[visible], x[visible])


def test_masked_reconstruction_loss_targets_withheld_coordinates_only():
    target = torch.zeros(2, 4)
    mask = np.zeros((2, 4), dtype=bool)
    mask[0, 0] = True  # the single withheld coordinate
    recon = target.clone()
    recon[0, 0] = 2.0  # withheld -> counts
    recon[1, 2] = 5.0  # visible  -> must not count
    assert masked_reconstruction_loss(recon, target, mask).item() == pytest.approx(4.0)

    target_b = target.clone()
    target_b[1, 2] = 100.0  # changing a *visible* target must not change the loss
    assert masked_reconstruction_loss(recon, target_b, mask).item() == pytest.approx(4.0)

    both_masked = mask.copy()
    both_masked[1, 1] = True
    assert masked_reconstruction_loss(recon, target, both_masked).item() == pytest.approx(2.0)  # mean of (4, 0)

    with_visible = masked_reconstruction_loss(recon, target, mask, visible_weight=0.5)
    assert with_visible.item() > 4.0


# --------------------------------------------------------------------------
# Architecture / configuration
# --------------------------------------------------------------------------
def test_encoder_and_decoder_shapes():
    encoder = ResidualEncoder(20, 8, 12)
    decoder = ReconstructionDecoder(20, 8, 12)
    z = encoder(torch.zeros(5, 20))
    x = decoder(z)
    assert z.shape == (5, 8)
    assert x.shape == (5, 20)
    assert encoder.network_config() == {
        "input_dim": 20, "residual_dim": 8, "hidden_dim": 12, "activation": "gelu", "dropout": 0.0
    }
    restored = ResidualEncoder.from_network_config(encoder.network_config())
    assert restored(torch.zeros(3, 20)).shape == (3, 8)


def test_training_config_validation():
    with pytest.raises(ResidualError, match="residual_dim"):
        ResidualTrainingConfig(residual_dim=0)
    with pytest.raises(ResidualError, match="mask_fraction"):
        ResidualTrainingConfig(mask_fraction=0.0)
    with pytest.raises(ResidualError, match="normalization"):
        ResidualTrainingConfig(normalization="zscore")
    with pytest.raises(ResidualError, match="dtype"):
        ResidualTrainingConfig(dtype="float64")
    with pytest.raises(ResidualError, match="device"):
        ResidualTrainingConfig(device="tpu")
    with pytest.raises(ResidualError, match="activation"):
        ResidualTrainingConfig(activation="swish")
    with pytest.raises(ResidualError, match="val_fraction"):
        ResidualTrainingConfig(val_fraction=0.75)
    assert ResidualTrainingConfig.from_mapping({"epochs": "7"}).epochs == 7


def test_residual_dim_is_configurable():
    bank = _bank()
    for dimension in (2, 8, 16, 52):
        result = train_residual(bank, config=_tiny_config(residual_dim=dimension))
        assert result.residual_dim == dimension
        assert result.encode(bank).shape == (bank.n_neurons, dimension)
        assert result.input_dim == build_residual_source(bank).n_features


# --------------------------------------------------------------------------
# Training behaviour
# --------------------------------------------------------------------------
def test_history_records_losses_and_best_epoch():
    bank = _bank()
    result = train_residual(bank, config=_tiny_config(epochs=9))
    assert len(result.history) == 9
    keys = {"epoch", "train_loss", "val_loss", "train_masked_mse", "val_masked_mse"}
    assert keys <= set(result.history[0])
    assert all(np.isfinite(record["val_loss"]) for record in result.history)
    best = min(result.history, key=lambda r: r["val_loss"])
    assert result.best_epoch == best["epoch"]
    assert result.best_val_loss == pytest.approx(best["val_loss"])
    assert 0.0 < result.history[-1]["mask_fraction_realised"] < 1.0


def test_train_validation_split_is_disjoint_and_within_bounds():
    bank = _bank()
    result = train_residual(bank, config=_tiny_config(val_fraction=0.25))
    split = result.provenance["split"]
    train, val = set(split["train_indices"]), set(split["val_indices"])
    assert not (train & val)
    assert train | val == set(range(bank.n_neurons))
    assert split["n_val"] == len(val) == round(0.25 * bank.n_neurons)
    assert split["kind"] == "neuron_examples_within_fit"


def test_training_does_not_modify_the_snn():
    model = _tiny_model()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bank = build_neuron_record_bank(model, with_activity=False)
    result = train_residual(bank, config=_tiny_config())
    result.encode(bank)
    after = model.state_dict()
    assert set(before) == set(after)
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert result.provenance["snn_checkpoint_modified"] is False


def test_reproducibility_for_identical_seed_and_bank():
    bank = _bank()
    config = _tiny_config(seed=3)
    a = train_residual(bank, config=config).encode(bank)
    b = train_residual(bank, config=config).encode(bank)
    assert np.array_equal(a, b)
    other = train_residual(bank, config=_tiny_config(seed=4)).encode(bank)
    assert not np.allclose(a, other)


def test_normalization_is_fit_derived_and_frozen():
    bank = _bank()
    source = build_residual_source(bank)
    result = train_residual(bank, config=_tiny_config())
    normalization = result.normalization
    assert normalization["mode"] == "train_standardise"
    assert len(normalization["mean"]) == source.n_features
    assert len(normalization["std"]) == source.n_features
    assert "FIT-derived" in normalization["provenance"]
    std = np.asarray(normalization["std"], dtype=float)
    assert np.isfinite(std).all() and (std > 0).all()

    none_result = train_residual(bank, config=_tiny_config(normalization="none"))
    assert none_result.normalization["mode"] == "none"
    assert none_result.normalization["mean"] is None
    assert not np.allclose(none_result.encode(bank), result.encode(bank))


def test_train_residual_api_validation():
    bank = _bank()
    source = build_residual_source(bank)
    assert train_residual(bank, source=source, residual_dim=4).residual_dim == 4
    with pytest.raises(ResidualError, match="residual_dim"):
        train_residual(bank)
    with pytest.raises(ResidualError, match="either"):
        train_residual(bank, config=_tiny_config(), residual_dim=4)
    with pytest.raises(ResidualError, match="source"):
        train_residual(bank, source=np.zeros((4, 4)))  # type: ignore[arg-type]
    with pytest.raises(ResidualError, match="bank"):
        train_residual(None, residual_dim=4)


def test_encoder_rejects_incompatible_source_schema():
    bank_a = build_neuron_record_bank(_tiny_model(seed=1), with_activity=False)
    bank_b = build_neuron_record_bank(_tiny_model(seed=2), with_activity=False)
    assert bank_a.n_neurons == bank_b.n_neurons
    result = train_residual(bank_a, config=_tiny_config())
    assert result.encode(bank_b).shape == (bank_b.n_neurons, result.residual_dim)  # same schema, different values
    # a bank with a different block availability has a different schema -> refused
    untrained = build_model(SNNConfig(n_input=20, n_hidden=16, n_output=4, n_bins=30, bin_ms=2.0), seed=1)
    bank_c = build_neuron_record_bank(untrained, with_activity=False)
    with pytest.raises(ResidualError, match="incompatible residual source"):
        result.encode(bank_c)


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def test_save_and_load_round_trip(tmp_path: Path):
    bank = _bank()
    result = train_residual(bank, config=_tiny_config())
    reference = result.encode(bank)
    path = tmp_path / "residual.pt"
    result.save(path)
    loaded = ResidualResult.load(path, expected_feature_names=result.feature_names, expected_residual_dim=result.residual_dim)
    assert loaded.residual_dim == result.residual_dim
    assert loaded.feature_names == result.feature_names
    assert np.array_equal(loaded.encode(bank), reference)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["schema"] == RESIDUAL_SCHEMA
    assert payload["feature_schema_hash"] == result.schema_hash
    assert payload["normalization"]["mode"] == "train_standardise"


def test_load_rejects_incompatible_artifacts(tmp_path: Path):
    bank = _bank()
    result = train_residual(bank, config=_tiny_config(residual_dim=8))
    path = result.save(tmp_path / "r.pt")

    with pytest.raises(ResidualError, match="feature names"):
        ResidualResult.load(path, expected_feature_names=result.feature_names[:-1])
    with pytest.raises(ResidualError, match="residual dimension"):
        ResidualResult.load(path, expected_residual_dim=9)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["feature_schema_hash"] = "0" * 64
    broken = tmp_path / "broken.pt"
    torch.save(payload, broken)
    with pytest.raises(ResidualError, match="integrity"):
        ResidualResult.load(broken)

    payload["schema"] = "learned_residual/v0"
    wrong_schema = tmp_path / "schema.pt"
    torch.save(payload, wrong_schema)
    with pytest.raises(ResidualError, match="schema"):
        ResidualResult.load(wrong_schema)

    with pytest.raises(FileNotFoundError):
        ResidualResult.load(tmp_path / "missing.pt")


def test_load_rejects_corrupted_source_dimension(tmp_path: Path):
    bank = _bank()
    result = train_residual(bank, config=_tiny_config())
    payload = result.to_payload()
    payload["input_dim"] = payload["input_dim"] + 1
    broken = tmp_path / "dim.pt"
    torch.save(payload, broken)
    with pytest.raises(ResidualError, match="inconsistent"):
        ResidualResult.load(broken)


# --------------------------------------------------------------------------
# Precision / memory guards
# --------------------------------------------------------------------------
def test_cpu_precision_fallback_is_documented():
    bank = _bank()
    result = train_residual(bank, config=_tiny_config(dtype="bfloat16", device="cpu"))
    assert result.provenance["dtype"] == "float32"
    assert any("fallback" in w for w in result.provenance["dtype_warnings"])


def test_float32_and_explicit_device_are_recorded():
    bank = _bank()
    result = train_residual(bank, config=_tiny_config(dtype="float32", device="cpu"))
    assert result.provenance["dtype"] == "float32"
    assert result.provenance["dtype_warnings"] == []
    assert result.provenance["device"] == "cpu"


def test_summary_and_provenance_are_json_serialisable():
    bank = _bank()
    result = train_residual(bank, config=_tiny_config())
    summary = result.summary()
    assert summary["schema"] == RESIDUAL_SCHEMA
    assert summary["uses_labels"] is False
    assert summary["residual_dim"] == result.residual_dim
    assert summary["mask"]["mask_fraction"] == result.config.mask_fraction
    assert isinstance(json.dumps(summary), str)


def test_no_forbidden_tensors_enter_the_residual_pipeline():
    bank = _bank()
    source = build_residual_source(bank)
    result = train_residual(bank, config=_tiny_config())
    z = result.encode(bank)
    for array in (source.X, z, np.asarray(result.normalization["mean"]), np.asarray(result.normalization["std"])):
        assert array.ndim <= 2
    # every training example is a feature vector: (n_examples, n_features) / (batch, residual_dim)
    assert source.X.shape[0] == bank.n_neurons
    assert z.shape == (bank.n_neurons, result.residual_dim)
    # masks are 2-D too
    assert make_mask(5, 10, mask_fraction=0.5, minimum_visible_features=2,
                     rng=np.random.default_rng(0)).ndim == 2
    # and the eval mask offset is recorded so it can be reproduced
    assert result.provenance["mask_policy"]["eval_mask_seed_offset"] == EVAL_MASK_OFFSET


# --------------------------------------------------------------------------
# V2 configuration integration
# --------------------------------------------------------------------------
def test_from_v2_config_uses_dimension_and_dtype():
    config = V2Config.from_config(
        Config({"vector": {"d": 64, "structured_d": 48, "learned_residual_d": 16,
                           "residual": {"enabled": True}},
                "precision": {"vector_dtype": "fp32"}}),
        warn=False,
    )
    training = ResidualTrainingConfig.from_v2_config(config, epochs=5)
    assert training.residual_dim == 16
    assert training.dtype == "float32"
    assert training.epochs == 5


def test_from_v2_config_rejects_a_zero_residual_dimension():
    config = V2Config.from_config(Config({}), warn=False)
    assert config.vector.learned_residual_d == 0
    with pytest.raises(ResidualError, match="residual_dim must be > 0"):
        ResidualTrainingConfig.from_v2_config(config)