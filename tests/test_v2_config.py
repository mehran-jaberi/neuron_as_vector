"""Tests for the V2 configuration infrastructure (vector / memory / precision / ...).

These tests cover the configuration surface added for the V2 "neuron as vector"
work. They do not test any representation functionality (none exists yet): the
point is that dimensions, blocks, precision, device, batch/chunk sizes and the
memory-token guard validate correctly *and* that existing configs still load with
the same behaviour.
"""

from __future__ import annotations

import json
import sys
import textwrap

import numpy as np
import pytest

from src.model import SNNConfig
from src.utils import PROJECT_ROOT, Config, load_config
from src.v2_config import (
    DEFAULT_ENABLED_BLOCKS,
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_STRUCTURED_D,
    SUPPORTED_DTYPES,
    V2Config,
    V2ConfigError,
    VectorConfig,
    check_input_token_budget,
    dtype_itemsize,
    estimate_input_tokens,
)

EXISTING_CONFIGS = (
    "baseline.yaml",
    "baseline_repaired.yaml",
    "neuron_space_baseline.yaml",
    "analysis.yaml",
)


def _v2(**sections) -> V2Config:
    """Resolve a V2Config from ad-hoc sections, without printing warnings."""
    return V2Config.from_config(Config(sections), warn=False)


# --------------------------------------------------------------------------
# 1. defaults
# --------------------------------------------------------------------------
def test_default_configuration_loads():
    v2 = V2Config.from_config(Config({}), warn=False)
    assert v2.vector.d == DEFAULT_STRUCTURED_D == 48
    assert v2.vector.structured_d == 48
    assert v2.vector.learned_residual_d == 0
    assert v2.vector.residual.enabled is False
    assert v2.vector.enabled_blocks == list(DEFAULT_ENABLED_BLOCKS)
    assert v2.vector.temporal_resolution == 10
    assert v2.vector.context_depth == 0

    assert v2.memory.train_batch_size == 128
    assert v2.memory.eval_batch_size == 256
    assert v2.memory.record_batch_size == 32
    assert v2.memory.activity_chunk_size == 32
    assert v2.memory.representation_chunk_size == 256
    assert v2.memory.device == "auto"
    assert v2.memory.storage == "cpu"
    assert v2.memory.mixed_precision is False
    assert v2.memory.max_input_tokens == DEFAULT_MAX_INPUT_TOKENS
    assert v2.memory.max_input_tokens_enabled is True

    assert v2.precision.vector_dtype == "float32"
    assert v2.precision.activity_dtype == "float32"
    assert v2.precision.model_dtype == "float32"

    assert v2.network.model_type == "recurrent_lif"
    assert v2.network.n_hidden == SNNConfig().n_hidden == 256
    assert v2.network.n_layers == 1
    assert v2.network.neurons_per_layer == [256]

    assert v2.experiment.dataset == "shd"
    assert v2.experiment.split == "speaker_aware"
    assert v2.experiment.seed == 0

    assert v2.simulation.n_bins == SNNConfig().n_bins
    assert v2.simulation.dt_ms == 2.0

    assert v2.warnings == []


def test_from_config_does_not_mutate_the_input_config():
    cfg = load_config(PROJECT_ROOT / "configs/neuron_space_baseline.yaml")
    before = cfg.to_dict()
    V2Config.from_config(cfg, warn=False)
    assert cfg.to_dict() == before


# --------------------------------------------------------------------------
# 2. backward compatibility with the existing configurations
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", EXISTING_CONFIGS)
def test_existing_baseline_config_still_loads(name):
    cfg = load_config(PROJECT_ROOT / "configs" / name)
    v2 = V2Config.from_config(cfg, warn=False)

    # the existing model/training configuration is untouched
    assert v2.snn.n_hidden == 256
    assert v2.snn.n_bins == 700
    assert v2.snn.bin_ms == 2.0
    assert v2.simulation.n_bins == v2.snn.n_bins == 700
    assert v2.simulation.duration_ms == 1400.0

    # V2 defaults preserve today's representation and add no warnings
    assert v2.vector.d == 48
    assert v2.vector.learned_residual_d == 0
    assert v2.vector.residual.enabled is False
    assert v2.warnings == []

    # memory batch sizes mirror the existing train block when not overridden
    assert v2.memory.train_batch_size == v2.train.batch_size == 128
    assert v2.memory.eval_batch_size == v2.train.eval_batch_size == 256
    assert v2.memory.device == str(cfg.get_path("run.device", "auto"))


def test_existing_recipe_values_survive_the_v2_layer():
    nsb = V2Config.from_config(load_config(PROJECT_ROOT / "configs/neuron_space_baseline.yaml"), warn=False)
    assert nsb.snn.readout_mode == "sum"  # the decisive baseline lever must be preserved
    assert nsb.train.epochs == 20

    repaired = V2Config.from_config(load_config(PROJECT_ROOT / "configs/baseline_repaired.yaml"), warn=False)
    assert repaired.train.l2_spikes == 0.001
    assert repaired.train.target_rate_hz == 10.0


# --------------------------------------------------------------------------
# 3-6. vector.d / structured_d / learned_residual_d and the dimension invariant
# --------------------------------------------------------------------------
def test_vector_d_can_be_overridden():
    # deterministic-only vector of dimension d (residual disabled)
    v2 = _v2(vector={"d": 100})
    assert (v2.vector.d, v2.vector.structured_d, v2.vector.learned_residual_d) == (100, 100, 0)

    # with the residual enabled the split must (and can) be derived
    v2b = _v2(vector={"d": 100, "structured_d": 48, "residual": {"enabled": True}})
    assert (v2b.vector.d, v2b.vector.structured_d, v2b.vector.learned_residual_d) == (100, 48, 52)

    # ambiguous: d alone with a residual requested
    with pytest.raises(V2ConfigError):
        _v2(vector={"d": 100, "residual": {"enabled": True}})


def test_vector_structured_d_can_be_overridden():
    v2 = _v2(vector={"structured_d": 64})
    assert (v2.vector.d, v2.vector.structured_d, v2.vector.learned_residual_d) == (64, 64, 0)
    assert v2.vector.residual.enabled is False


def test_learned_residual_d_can_be_zero():
    v2 = _v2(vector={"d": 48, "structured_d": 48, "learned_residual_d": 0})
    assert v2.vector.learned_residual_d == 0
    assert v2.vector.residual.enabled is False
    assert v2.warnings == []


def test_consistent_dimensions_pass():
    v2 = _v2(vector={"d": 64, "structured_d": 48, "learned_residual_d": 16, "residual": {"enabled": True}})
    assert v2.vector.d == v2.vector.structured_d + v2.vector.learned_residual_d == 64
    # the residual is configured but not implemented in this stage -> warning, no error
    assert any("residual" in w for w in v2.warnings)


@pytest.mark.parametrize(
    "vector",
    [
        {"d": 100, "structured_d": 48, "learned_residual_d": 40},  # 48 + 40 != 100
        {"d": 64, "structured_d": 48, "learned_residual_d": 0},  # 48 + 0 != 64
        {"d": 48, "structured_d": 48, "learned_residual_d": 0, "residual": {"enabled": True}},
        {"d": 64, "structured_d": 48, "learned_residual_d": 16},  # residual capacity but disabled
    ],
)
def test_inconsistent_dimension_configurations_fail(vector):
    with pytest.raises(V2ConfigError):
        VectorConfig.from_mapping(vector)


@pytest.mark.parametrize(
    "vector",
    [
        {"d": -1},
        {"d": 0},
        {"structured_d": -5},
        {"learned_residual_d": -1},
    ],
)
def test_invalid_dimensions_fail(vector):
    with pytest.raises(V2ConfigError):
        VectorConfig.from_mapping(vector)


def test_enabled_blocks_normalization_and_validation():
    v = VectorConfig.from_mapping({"enabled_blocks": "activity, intrinsic ,intrinsic"})
    assert v.enabled_blocks == ["intrinsic", "activity"]  # canonical order, de-duplicated

    with pytest.raises(V2ConfigError):
        VectorConfig.from_mapping({"enabled_blocks": ["intrinsic", "typo_block"]})
    with pytest.raises(V2ConfigError):
        VectorConfig.from_mapping({"enabled_blocks": []})


def test_blocks_are_accepted_and_reported_implemented_or_not():
    v = VectorConfig.from_mapping(
        {"enabled_blocks": ["intrinsic", "temporal", "network_context"], "context_depth": 2}
    )
    assert v.enabled_blocks == ["intrinsic", "temporal", "network_context"]
    # temporal is genuinely implemented now; network_context is still declared only
    assert set(v.unimplemented_blocks) == {"network_context"}
    assert v.context_depth == 2


@pytest.mark.parametrize("value", [0, -1])
def test_invalid_temporal_resolution_fails(value):
    with pytest.raises(V2ConfigError):
        VectorConfig.from_mapping({"temporal_resolution": value})


def test_negative_context_depth_fails_but_zero_is_valid():
    with pytest.raises(V2ConfigError):
        VectorConfig.from_mapping({"context_depth": -1})
    assert VectorConfig.from_mapping({"context_depth": 0}).context_depth == 0


def test_temporal_resolution_cannot_exceed_n_bins():
    with pytest.raises(V2ConfigError):
        _v2(
            model={"n_input": 700, "n_hidden": 256, "n_output": 20, "n_bins": 10, "bin_ms": 2.0},
            vector={"temporal_resolution": 20},
        )


# --------------------------------------------------------------------------
# 9. precision / device / storage validation
# --------------------------------------------------------------------------
def test_invalid_dtype_device_combinations_fail():
    with pytest.raises(V2ConfigError):  # half-precision model on CPU
        _v2(precision={"model_dtype": "fp16"}, memory={"device": "cpu"})
    with pytest.raises(V2ConfigError):  # mixed precision on CPU
        _v2(memory={"mixed_precision": True, "device": "cpu"})
    with pytest.raises(V2ConfigError):  # GPU storage with a CPU device
        _v2(memory={"storage": "gpu", "device": "cpu"})
    with pytest.raises(V2ConfigError):  # bfloat16 vector dtype with NumPy-backed storage
        _v2(precision={"vector_dtype": "bf16"}, memory={"storage": "cpu"})
    with pytest.raises(V2ConfigError):  # mixed precision expects fp32 master weights
        _v2(precision={"model_dtype": "bf16"}, memory={"mixed_precision": True, "device": "auto"})


def test_unsupported_dtype_string_fails():
    with pytest.raises(V2ConfigError):
        _v2(precision={"vector_dtype": "float64"})
    with pytest.raises(V2ConfigError):
        _v2(precision={"model_dtype": "int8"})


def test_invalid_device_and_storage_strings_fail():
    with pytest.raises(V2ConfigError):
        _v2(memory={"device": "tpu"})
    with pytest.raises(V2ConfigError):
        _v2(memory={"storage": "tape"})


def test_dtype_aliases_normalize_to_canonical_names():
    v2 = _v2(precision={"vector_dtype": "fp16", "activity_dtype": "half", "model_dtype": "fp32"})
    assert (v2.precision.vector_dtype, v2.precision.activity_dtype, v2.precision.model_dtype) == (
        "float16",
        "float16",
        "float32",
    )
    assert all(v in SUPPORTED_DTYPES for v in vars(v2.precision).values() if isinstance(v, str))


def test_precision_helper_dtypes():
    v2 = _v2()
    assert v2.precision.to_numpy_dtype("vector") == np.dtype("float32")
    assert str(v2.precision.to_torch_dtype("model")) == "torch.float32"

    # bfloat16 is allowed only with GPU storage, and has no NumPy dtype
    v2b = _v2(
        precision={"vector_dtype": "bf16"},
        memory={"storage": "gpu", "device": "cuda"},
    )
    assert v2b.precision.vector_dtype == "bfloat16"
    assert str(v2b.precision.to_torch_dtype("vector")) == "torch.bfloat16"
    with pytest.raises(V2ConfigError):
        v2b.precision.to_numpy_dtype("vector")
    with pytest.raises(V2ConfigError):
        v2.precision.to_numpy_dtype("not_a_target")


# --------------------------------------------------------------------------
# 10. memory controls
# --------------------------------------------------------------------------
def test_memory_batch_and_chunk_sizes_can_be_overridden():
    v2 = _v2(
        memory={
            "train_batch_size": 64,
            "eval_batch_size": 128,
            "record_batch_size": 16,
            "activity_chunk_size": 8,
            "representation_chunk_size": 32,
        }
    )
    assert v2.memory.train_batch_size == 64
    assert v2.memory.eval_batch_size == 128
    assert v2.memory.record_batch_size == 16
    assert v2.memory.activity_chunk_size == 8
    assert v2.memory.representation_chunk_size == 32


@pytest.mark.parametrize(
    "field",
    [
        "train_batch_size",
        "eval_batch_size",
        "record_batch_size",
        "activity_chunk_size",
        "representation_chunk_size",
    ],
)
@pytest.mark.parametrize("value", [0, -4])
def test_invalid_batch_and_chunk_sizes_fail(field, value):
    with pytest.raises(V2ConfigError):
        _v2(memory={field: value})


def test_memory_defaults_mirror_train_block_and_explicit_values_win():
    cfg = Config({"train": {"batch_size": 32, "eval_batch_size": 64}})
    v2 = V2Config.from_config(cfg, warn=False)
    assert v2.memory.train_batch_size == 32
    assert v2.memory.eval_batch_size == 64

    cfg2 = Config({"train": {"batch_size": 32, "eval_batch_size": 64}, "memory": {"train_batch_size": 8}})
    v2b = V2Config.from_config(cfg2, warn=False)
    assert v2b.memory.train_batch_size == 8  # explicit memory value wins
    assert v2b.memory.eval_batch_size == 64  # the other one is still inherited


def test_memory_device_inherits_run_device():
    v2 = V2Config.from_config(Config({"run": {"device": "cpu"}}), warn=False)
    assert v2.memory.device == "cpu"


# --------------------------------------------------------------------------
# 11. max_input_tokens / dense-input budget guard
# --------------------------------------------------------------------------
def test_max_input_tokens_validation():
    with pytest.raises(V2ConfigError):
        _v2(memory={"max_input_tokens": -1})
    assert _v2(memory={"max_input_tokens": 0}).memory.max_input_tokens_enabled is False


def test_estimate_input_tokens_and_dtype_itemsize():
    assert estimate_input_tokens(32, 700, 700) == 15_680_000
    assert dtype_itemsize("fp32") == 4
    assert dtype_itemsize("fp16") == 2
    assert dtype_itemsize("bf16") == 2
    with pytest.raises(V2ConfigError):
        dtype_itemsize("float64")
    with pytest.raises(V2ConfigError):
        estimate_input_tokens(0, 700, 700)


def test_input_token_budget_guard():
    # disabled guard (max_tokens = 0) never raises
    report = check_input_token_budget(256, 700, 700, max_tokens=0)
    assert report["tokens"] == 125_440_000
    assert report["within_budget"] is True

    # within budget passes and reports the estimate
    ok = check_input_token_budget(32, 700, 700, max_tokens=20_000_000, dtype_name="fp16")
    assert ok["tokens"] == 15_680_000
    assert ok["bytes"] == 15_680_000 * 2
    assert ok["within_budget"] is True

    # exceeded raises with an actionable message
    with pytest.raises(V2ConfigError):
        check_input_token_budget(256, 700, 700, max_tokens=32_000_000)
    with pytest.raises(V2ConfigError):
        check_input_token_budget(32, 700, 700, max_tokens=-1)


def test_memory_config_check_input_budget_uses_configured_limit():
    v2 = _v2(memory={"max_input_tokens": 20_000_000})
    assert v2.memory.check_input_budget(batch_size=32, n_bins=700, n_input=700)["within_budget"] is True
    with pytest.raises(V2ConfigError):
        v2.memory.check_input_budget(batch_size=64, n_bins=700, n_input=700)
    # an explicit override of the limit is honoured
    assert v2.memory.check_input_budget(batch_size=64, n_bins=700, n_input=700, max_tokens=0)["within_budget"] is True


# --------------------------------------------------------------------------
# 12. command-line dotted overrides
# --------------------------------------------------------------------------
def test_dotted_overrides_propagate(tmp_path):
    cfg_file = tmp_path / "v2.yaml"
    cfg_file.write_text(
        textwrap.dedent(
            """
            seed: 3
            model:
              n_input: 700
              n_hidden: 128
              n_output: 20
              n_bins: 700
              bin_ms: 2.0
            """
        ),
        encoding="utf-8",
    )
    cfg = load_config(
        cfg_file,
        overrides=[
            "vector.d=64",
            "vector.structured_d=48",
            "vector.learned_residual_d=16",
            "vector.residual.enabled=true",
            "memory.record_batch_size=16",
            "memory.representation_chunk_size=64",
            "precision.vector_dtype=fp16",
            "memory.device=cpu",
        ],
    )
    v2 = V2Config.from_config(cfg, warn=False)
    assert v2.network.n_hidden == 128
    assert (v2.vector.d, v2.vector.structured_d, v2.vector.learned_residual_d) == (64, 48, 16)
    assert v2.vector.residual.enabled is True
    assert v2.memory.record_batch_size == 16
    assert v2.memory.representation_chunk_size == 64
    assert v2.precision.vector_dtype == "float16"
    assert v2.memory.device == "cpu"
    assert v2.experiment.seed == 3
    assert v2.simulation.n_bins == 700 and v2.simulation.duration_ms == 1400.0


def test_dotted_override_can_raise_the_token_budget(tmp_path):
    cfg_file = tmp_path / "m.yaml"
    cfg_file.write_text("model:\n  n_hidden: 256\n  n_bins: 700\n", encoding="utf-8")
    cfg = load_config(cfg_file, overrides=["memory.max_input_tokens=200000000"])
    assert V2Config.from_config(cfg, warn=False).memory.max_input_tokens == 200_000_000


# --------------------------------------------------------------------------
# Network (including the future multi-layer interface)
# --------------------------------------------------------------------------
def test_network_view_reads_the_model_block():
    v2 = _v2(model={"n_input": 700, "n_hidden": 128, "n_output": 20, "n_bins": 700, "bin_ms": 2.0})
    assert v2.network.n_hidden == 128
    assert v2.network.n_layers == 1
    assert v2.network.neurons_per_layer == [128]
    assert v2.network.multi_layer_implemented is True
    v2.network.require_implemented()  # must not raise


def test_invalid_layer_counts_and_model_types_fail():
    with pytest.raises(V2ConfigError):
        _v2(model={"n_hidden": 256, "n_layers": 0})
    with pytest.raises(V2ConfigError):
        _v2(model={"n_hidden": 256, "model_type": "transformer"})
    with pytest.raises(V2ConfigError):
        _v2(model={"n_hidden": 256, "n_layers": 2, "neurons_per_layer": [128]})
    with pytest.raises(V2ConfigError):
        _v2(model={"n_hidden": 256, "neurons_per_layer": [128]})


def test_multi_layer_is_accepted_but_reported_unimplemented():
    v2 = _v2(model={"n_hidden": 256, "n_layers": 2, "neurons_per_layer": [256, 128]})
    assert v2.network.n_layers == 2
    assert v2.network.neurons_per_layer == [256, 128]
    assert v2.network.multi_layer_implemented is False
    assert any("n_layers" in w for w in v2.warnings)
    with pytest.raises(V2ConfigError):
        v2.require_implemented()


def test_strict_mode_rejects_unimplemented_requests():
    multi = Config({"model": {"n_hidden": 256, "n_layers": 2}})
    V2Config.from_config(multi, warn=False)  # accepted, reported as a warning
    with pytest.raises(V2ConfigError):
        V2Config.from_config(multi, strict=True, warn=False)

    blocks = Config({"vector": {"enabled_blocks": ["intrinsic", "network_context"]}})
    v2 = V2Config.from_config(blocks, warn=False)
    assert v2.vector.unimplemented_blocks == ["network_context"]
    with pytest.raises(V2ConfigError):
        V2Config.from_config(blocks, strict=True, warn=False)

    # temporal is implemented, so a temporal-only request no longer trips strict mode
    temporal = Config({"vector": {"enabled_blocks": ["intrinsic", "temporal"]}})
    assert V2Config.from_config(temporal, warn=False).vector.unimplemented_blocks == []
    V2Config.from_config(temporal, strict=True, warn=False)

    residual = Config({"vector": {"d": 64, "structured_d": 48, "learned_residual_d": 16, "residual": {"enabled": True}}})
    with pytest.raises(V2ConfigError):
        V2Config.from_config(residual, strict=True, warn=False)

    # the default configuration is fully implemented, so strict mode passes
    V2Config.from_config(Config({}), strict=True, warn=False)


# --------------------------------------------------------------------------
# Experiment / simulation views
# --------------------------------------------------------------------------
def test_experiment_view_derives_existing_controls():
    v2 = V2Config.from_config(
        Config({"seed": 7, "run": {"synthetic": True}, "data": {"prefer_speaker_aware": False}}),
        warn=False,
    )
    assert v2.experiment.dataset == "synthetic"
    assert v2.experiment.split == "stratified"
    assert v2.experiment.seed == 7

    with pytest.raises(V2ConfigError):
        _v2(experiment={"dataset": "mnist"})
    with pytest.raises(V2ConfigError):
        _v2(experiment={"split": "random"})
    with pytest.raises(V2ConfigError):
        _v2(experiment={"n_fit": 0})
    with pytest.raises(V2ConfigError):
        _v2(experiment={"seed": -1})


def test_experiment_expected_sizes_are_checked_not_applied():
    v2 = _v2(experiment={"n_fit": 100, "n_probe": 50})
    assert v2.experiment.validate_against_split({"n_train": 100, "n_probe": 50}) == {
        "n_fit": 100,
        "n_probe": 50,
    }
    with pytest.raises(V2ConfigError):
        v2.experiment.validate_against_split({"n_train": 99, "n_probe": 50})
    with pytest.raises(V2ConfigError):
        v2.experiment.validate_against_split({})

    # no declarations -> no checks, no errors
    assert _v2().experiment.validate_against_split({}) == {}


def test_simulation_view_matches_model_block():
    v2 = _v2(model={"n_input": 700, "n_hidden": 256, "n_output": 20, "n_bins": 700, "bin_ms": 2.0})
    assert v2.simulation.n_bins == 700
    assert v2.simulation.dt_ms == 2.0
    assert v2.simulation.bin_ms == 2.0
    assert v2.simulation.duration_ms == 1400.0
    assert v2.simulation.duration_s == 1.4

    with pytest.raises(V2ConfigError):
        _v2(model={"n_input": 700, "n_hidden": 256, "n_output": 20, "n_bins": 0, "bin_ms": 2.0})
    with pytest.raises(V2ConfigError):
        _v2(model={"n_input": 700, "n_hidden": 256, "n_output": 20, "n_bins": 700, "bin_ms": -1.0})


# --------------------------------------------------------------------------
# Serialisation / reporting
# --------------------------------------------------------------------------
def test_to_dict_is_json_serialisable():
    v2 = _v2()
    payload = v2.to_dict()
    assert set(payload) >= {"vector", "memory", "precision", "network", "experiment", "simulation", "snn", "train"}
    assert isinstance(json.dumps(payload), str)


def test_summary_rows_cover_the_demo_knobs():
    rows = dict(_v2().summary_rows())
    for key in (
        "model.n_hidden (n)",
        "vector.d",
        "vector.structured_d",
        "vector.learned_residual_d",
        "vector.enabled_blocks",
        "memory.record_batch_size",
        "memory.representation_chunk_size",
        "memory.device",
        "precision.vector_dtype",
        "simulation.n_bins",
    ):
        assert key in rows


# --------------------------------------------------------------------------
# Small integration demonstration (scripts/show_v2_config.py)
# --------------------------------------------------------------------------
def test_show_v2_config_script_demonstrates_overrides(capsys):
    scripts_dir = str(PROJECT_ROOT / "scripts")
    sys.path.insert(0, scripts_dir)
    try:
        import show_v2_config  # noqa: WPS433 (imported from scripts/ on purpose)
    finally:
        if scripts_dir in sys.path:
            sys.path.remove(scripts_dir)

    example = str(PROJECT_ROOT / "configs/v2_example.yaml")

    # defaults: resolves cleanly and reports the knobs
    assert show_v2_config.main(["--config", example]) == 0
    out = capsys.readouterr().out
    assert "vector.d" in out and "memory.record_batch_size" in out
    assert "no warnings" in out

    # every knob the demonstration promises can be changed from the command line
    rc = show_v2_config.main(
        [
            "--config",
            example,
            "--override",
            "model.n_hidden=128",
            "--override",
            "vector.d=64",
            "--override",
            "vector.structured_d=48",
            "--override",
            "vector.learned_residual_d=16",
            "--override",
            "vector.residual.enabled=true",
            "--override",
            "memory.record_batch_size=16",
            "--override",
            "memory.representation_chunk_size=64",
            "--override",
            "precision.vector_dtype=fp16",
            "--override",
            "memory.device=cpu",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "128" in out and "16" in out and "float16" in out and "cpu" in out

    # contradictory dimensions are rejected with exit code 2
    rc = show_v2_config.main(
        [
            "--config",
            example,
            "--override",
            "vector.d=100",
            "--override",
            "vector.structured_d=48",
            "--override",
            "vector.learned_residual_d=40",
        ]
    )
    assert rc == 2
    assert "error" in capsys.readouterr().err
