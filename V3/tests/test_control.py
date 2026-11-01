"""Tests for the programmatic control panel (``v3.control``).

CPU-only, no training: the training call is replaced by a fake that records the
config it was given, so these tests check the *configuration layer* and the
wiring (which functions are called), never a real run.
"""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path

import pytest
import yaml

from v3 import control
from v3.config import TimingConfig, V3Config
from v3.control import apply_overrides, build_config, run_configured, valid_override_paths

import run_experiment

V3_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = V3_ROOT / "V3_SHD_experiment.ipynb"


def _get(cfg, path: str):
    node = cfg
    for part in path.split("."):
        node = getattr(node, part)
    return node


def _sample(annotation: str, current):
    """A different-but-legal value of the same type as ``current``."""
    if current is not None:
        if isinstance(current, bool):
            return not current
        if isinstance(current, int):
            return current + 1
        if isinstance(current, float):
            return current + 1.0
        if isinstance(current, str):
            return "probe"
        if isinstance(current, list):
            return [*current, "probe"]
        if isinstance(current, dict):
            return {**current, "probe": 1}
        raise AssertionError(f"no sample value for field type {type(current).__name__}")
    # Optional field currently set to None (e.g. max_train_batches)
    snippet = annotation.lower()
    if "int" in snippet:
        return 3
    if "float" in snippet:
        return 0.5
    if "str" in snippet:
        return "probe"
    if "bool" in snippet:
        return True
    raise AssertionError(f"no sample value for annotation {annotation!r}")


def _default() -> V3Config:
    return V3Config.from_yaml(control.DEFAULT_CONFIG)


# ---------------------------------------------------------------------- #
# forms and coercion
# ---------------------------------------------------------------------- #
def test_mapping_and_item_list_produce_the_same_config():
    mapping = {"epochs": 3, "n_neurons": 8, "timing.time_bin_ms": 4.0}
    items = ["epochs=3", "n_neurons=8", "timing.time_bin_ms=4"]
    assert build_config(mapping) == build_config(items)
    expected = _default().with_overrides(
        epochs=3, n_neurons=8,
        timing=TimingConfig(sequence_duration_ms=1000.0, time_bin_ms=4.0),
    )
    assert build_config(mapping) == expected


def test_values_are_coerced_to_the_field_type():
    cfg = build_config({
        "epochs": "3",
        "learning_rate": "5e-4",
        "shuffle_train": "false",
        "tag": "v3_probe",
    })
    assert cfg.epochs == 3 and type(cfg.epochs) is int
    assert cfg.learning_rate == 5e-4 and type(cfg.learning_rate) is float
    assert cfg.shuffle_train is False
    assert cfg.tag == "v3_probe"

    cfg2 = build_config(["epochs=4", "learning_rate=0.002", "shuffle_train=true"])
    assert cfg2.epochs == 4 and type(cfg2.epochs) is int
    assert cfg2.learning_rate == 0.002 and type(cfg2.learning_rate) is float
    assert cfg2.shuffle_train is True


def test_two_level_dotted_keys_reach_the_right_field():
    cfg = build_config({
        "timing.time_bin_ms": 4.0,
        "state_regularization.mode": "noise",
        "state_regularization.noise_std": 0.01,
    })
    assert cfg.timing.time_bin_ms == 4.0
    assert cfg.timing.sequence_duration_ms == _default().timing.sequence_duration_ms
    assert cfg.state_regularization.mode == "noise"
    assert cfg.state_regularization.noise_std == 0.01
    assert cfg.state_regularization.quantize_bits == _default().state_regularization.quantize_bits


def test_augmentation_keys_are_addressable_when_the_block_exists():
    paths = valid_override_paths()
    if not any(p.startswith("augmentation.") for p in paths):
        pytest.skip("the augmentation config block is not present yet")
    assert "augmentation.time_jitter.std_ms" in paths
    cfg = build_config({"augmentation.time_jitter.std_ms": 0.5})
    assert _get(cfg, "augmentation.time_jitter.std_ms") == 0.5


# ---------------------------------------------------------------------- #
# unknown keys
# ---------------------------------------------------------------------- #
def test_unknown_key_names_the_key_and_the_closest_field():
    with pytest.raises(ValueError) as info:
        build_config({"epockz": 2})
    message = str(info.value)
    assert "epockz" in message
    assert "epochs" in message


def test_unknown_nested_key_names_the_closest_dotted_field():
    with pytest.raises(ValueError) as info:
        build_config({"timing.time_bim_ms": 2})
    message = str(info.value)
    assert "timing.time_bim_ms" in message or "time_bim_ms" in message
    assert "timing.time_bin_ms" in message


def test_unknown_key_in_item_form_also_raises():
    with pytest.raises(ValueError):
        build_config(["n_neuronz=8"])


def test_bad_override_container_is_rejected():
    with pytest.raises(TypeError):
        apply_overrides(_default(), "epochs=3")  # a bare string is ambiguous


# ---------------------------------------------------------------------- #
# quick profile, defaults and the YAML file
# ---------------------------------------------------------------------- #
def test_quick_merged_config_equals_the_cli_quick_config():
    from run_experiment import QUICK_OVERRIDES

    expected = V3Config.from_yaml(run_experiment.DEFAULT_CONFIG).with_overrides(**QUICK_OVERRIDES)
    assert build_config(quick=True) == expected


def test_run_configured_with_no_overrides_leaves_defaults_untouched(monkeypatch):
    seen: dict = {}

    def fake_run_variant(vcfg, device, show_progress=True, resume_from=None):
        seen["cfg"] = vcfg
        return {"tag": vcfg.tag, "label": "fake", "history": [], "n_parameters": 0,
                "result": object()}

    monkeypatch.setattr(run_experiment, "run_variant", fake_run_variant)
    result = run_configured(None, register=False, test=False)

    assert seen["cfg"] == _default()
    assert result["config"] == _default()
    assert result["run_dirs"] == {_default().tag: None}


def test_run_configured_applies_overrides_before_the_training_call(monkeypatch):
    seen: dict = {}

    def fake_run_variant(vcfg, device, show_progress=True, resume_from=None):
        seen["cfg"] = vcfg
        return {"tag": vcfg.tag, "label": "fake", "history": [], "n_parameters": 0,
                "result": object()}

    monkeypatch.setattr(run_experiment, "run_variant", fake_run_variant)
    run_configured({"epochs": 7, "seed": "3"}, register=False, test=False)
    assert seen["cfg"].epochs == 7 and type(seen["cfg"].epochs) is int
    assert seen["cfg"].seed == 3

    run_configured(["epochs=9"], register=False, test=False, variant="scalar")
    assert seen["cfg"].epochs == 9
    assert seen["cfg"].state_dim == 1
    assert seen["cfg"].tag.endswith("_scalar_d1")


def test_yaml_config_file_is_never_modified():
    before_bytes = control.DEFAULT_CONFIG.read_bytes()
    before_stat = control.DEFAULT_CONFIG.stat()
    build_config({"epochs": 9, "n_neurons": 8, "timing.time_bin_ms": 4.0})
    assert control.DEFAULT_CONFIG.read_bytes() == before_bytes
    assert control.DEFAULT_CONFIG.stat().st_mtime_ns == before_stat.st_mtime_ns


def test_merged_config_is_what_the_run_config_yaml_records(monkeypatch, tmp_path):
    """The registry row's config.yaml is the merged config, not the YAML defaults."""
    out = tmp_path / "results"

    def fake_run_variant(vcfg, device, show_progress=True, resume_from=None):
        return {
            "tag": vcfg.tag, "label": "fake", "n_parameters": 1,
            "epochs": vcfg.epochs, "fit_accuracy": 0.0, "val_accuracy": 0.0,
            "best_val_accuracy": 0.0, "best_epoch": 1, "train_seconds": 0.0,
            "init_and_data_seconds": 0.0, "checkpoint": "", "history": [],
            "result": object(),
        }

    monkeypatch.setattr(run_experiment, "run_variant", fake_run_variant)
    result = run_configured({"epochs": 3, "out_dir": str(out)}, register=True, test=False)
    ((tag, run_dir),) = result["run_dirs"].items()
    assert run_dir is not None
    written = yaml.safe_load((Path(run_dir) / "config.yaml").read_text(encoding="utf-8"))
    assert written["epochs"] == 3
    assert written["tag"] == tag
    assert Path(run_dir).parent == out


# ---------------------------------------------------------------------- #
# reflection: every CLI-reachable field is reachable through the mapping
# ---------------------------------------------------------------------- #
def test_every_config_field_is_addressable_through_the_mapping():
    base = _default()
    paths = valid_override_paths(base)
    assert len(paths) == len(set(paths)), "duplicate override paths"
    # the CLI's documented one-level dotted keys must be in the same set
    assert {"timing.time_bin_ms", "state_regularization.mode"} <= set(paths)

    for path in paths:
        current = _get(base, path)
        annotation = next(
            f.type for f in dataclasses.fields(_parent(base, path)) if f.name == path.rsplit(".", 1)[-1]
        )
        sample = _sample(str(annotation), current)
        merged = apply_overrides(base, {path: sample})
        got = _get(merged, path)
        assert got == sample, f"override {path}={sample!r} did not reach the field (got {got!r})"
        assert type(got) is type(sample), f"{path}: {type(got)} != {type(sample)}"


def _parent(cfg, path: str):
    node = cfg
    parts = path.split(".")
    for part in parts[:-1]:
        node = getattr(node, part)
    return node


# ---------------------------------------------------------------------- #
# notebook integration (the built artifact, never executed)
# ---------------------------------------------------------------------- #
def test_notebook_has_a_parse_clean_control_panel_cell():
    cells = [
        "".join(cell["source"])
        for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
        if cell["cell_type"] == "code"
    ]
    for i, source in enumerate(cells):
        ast.parse(source, filename=f"notebook cell {i}")  # noqa: F841
    control_cells = [i for i, src in enumerate(cells) if "build_config(" in src]
    assert len(control_cells) == 1, f"expected one control-panel cell, found {control_cells}"
    assert "OVERRIDES" in cells[control_cells[0]]
    assert "run_configured" not in cells[control_cells[0]] or "build_config(" in cells[control_cells[0]]
