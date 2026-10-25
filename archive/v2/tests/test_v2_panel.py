"""Control-panel interface tests: panel state layer + the Jupyter notebook itself.

Two layers are checked:

* ``src/v2_panel.py`` - the notebook-facing state layer (controls, defaults, immediate validation,
  delta overrides, previews, cache inventory, inspection). Everything is hermetic: a tiny synthetic
  config and checkpoint are written to ``tmp_path`` and no real checkpoint or dataset is required.
* ``notebooks/V2_Control_Panel.ipynb`` - structure (valid nbformat, unique ids, every code cell
  compiles, no package installation, no duplicated metric logic) **and execution**: the cells
  tagged ``panel:auto`` (everything except the explicitly triggered build/inspect/evaluate cells)
  are executed in order in-process, with the environment variables the notebook documents. This is
  the lightweight, kernel-free notebook execution test - the environment has no Jupyter kernel and
  none is installed by these tests.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from dataclasses import fields as dataclass_fields

import numpy as np
import pytest
import torch

from src.model import SNNConfig, RecurrentLIFSNN, build_model
from src.neuron_record import build_neuron_record_bank, structural_matrix_from_record_bank
from src.structured_vector import StructuredVectorEncoder
from src.utils import Config
from src.v2_config import (
    RESIDUAL_STANDARDIZATIONS,
    SUPPORTED_DTYPES,
    MemoryConfig,
    PrecisionConfig,
    V2Config,
    VectorConfig,
    VectorResidualConfig,
)
from src import v2_panel as panel
from src.v2_pipeline import (
    BuildResult,
    EvaluationResult,
    ResolvedConfig,
    available_presets,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "V2_Control_Panel.ipynb"

#: A tiny, hermetic stand-in for the scientific configuration (synthetic data, 16 hidden neurons).
TINY_CONFIG = {
    "seed": 0,
    "paths": {"data_dir": "data"},
    "run": {
        "synthetic": True,
        "synthetic_n_samples": 120,
        "synthetic_n_classes": 4,
        "synthetic_n_channels": 20,
    },
    "model": {
        "n_input": 20, "n_hidden": 16, "n_output": 4, "n_bins": 30, "bin_ms": 2.0,
        "neuron_param_mode": "bias", "readout_mode": "sum",
    },
    "train": {"n_classes": 4},
    "data": {"dev_fraction": 0.25, "probe_fraction": 0.25, "prefer_speaker_aware": True, "split_seed": 0},
    "fingerprint": {"n_psth_bins": 5, "min_spikes_for_latency": 1.0},
}


@pytest.fixture(scope="module")
def tiny_config_path(tmp_path_factory) -> Path:
    import yaml

    path = tmp_path_factory.mktemp("panel_cfg") / "tiny.yaml"
    path.write_text(yaml.safe_dump(TINY_CONFIG), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory) -> Path:
    cfg = SNNConfig(
        n_input=20, n_hidden=16, n_output=4, n_bins=30, bin_ms=2.0, neuron_param_mode="bias"
    )
    model = build_model(cfg, seed=1)
    with torch.no_grad():
        model.b_hid.copy_(torch.linspace(-0.5, 0.5, cfg.n_hidden))
    path = tmp_path_factory.mktemp("panel_ckpt") / "tiny.pt"
    model.save(str(path))
    return path


@pytest.fixture(scope="module")
def panel_state(tiny_config_path, tiny_checkpoint, tmp_path_factory) -> dict:
    return panel.default_state(
        preset="historical_48",
        config_path=tiny_config_path,
        checkpoint=tiny_checkpoint,
        out_dir=tmp_path_factory.mktemp("panel_out"),
    )


# --------------------------------------------------------------------------
# Controls: metadata, defaults, immediate validation
# --------------------------------------------------------------------------
def test_controls_cover_every_required_section():
    groups = {c.group for c in panel.panel_controls()}
    assert {
        "panel", "representation", "blocks", "functional_response", "temporal", "residual",
        "model", "memory", "precision", "evaluation",
    } <= groups
    names = {c.name for c in panel.panel_controls()}
    for required in (
        "structured_d", "learned_residual_d", "enabled_blocks", "temporal_resolution",
        "source_functional_response", "functional_source_dim", "functional_projection_seed",
        "functional_source_normalization", "source_temporal",
        "residual_enabled", "residual_seed", "residual_hidden_dim", "residual_epochs",
        "residual_learning_rate", "residual_batch_size", "residual_mask_fraction",
        "residual_mask_mode", "residual_val_fraction", "residual_split_seed", "residual_mask_seed",
        "residual_standardization",
        "device", "train_batch_size", "eval_batch_size", "record_batch_size",
        "activity_chunk_size", "representation_chunk_size", "storage", "mixed_precision",
        "vector_dtype", "activity_dtype", "model_dtype", "max_input_tokens",
        "n_hidden", "model_type", "checkpoint", "preset", "evaluation_target",
    ):
        assert required in names, required


def test_config_backed_controls_exist_in_v2_config():
    """Every control maps to a real configuration field, and every field is exposed."""
    vector_fields = {f.name for f in dataclass_fields(VectorConfig)}
    residual_fields = {f.name for f in dataclass_fields(VectorResidualConfig)}
    memory_fields = {f.name for f in dataclass_fields(MemoryConfig)}
    precision_fields = {f.name for f in dataclass_fields(PrecisionConfig)}
    exposed = panel.config_backed_controls()
    for name, key in exposed.items():
        assert key is not None
        if key.startswith("vector.residual."):
            assert key.split(".")[2] in residual_fields, (name, key)
        elif key.startswith("vector."):
            assert key.split(".")[1] in vector_fields, (name, key)
        elif key.startswith("memory."):
            assert key.split(".")[1] in memory_fields, (name, key)
        elif key.startswith("precision."):
            assert key.split(".")[1] in precision_fields, (name, key)
        elif key.startswith("model."):
            assert key.split(".")[1] in {"n_hidden", "model_type"}, (name, key)
        else:
            assert key == "seed", (name, key)
    # every residual protocol field is exposed (a new field must not stay hidden)
    assert {k.split(".")[2] for k in exposed.values() if k.startswith("vector.residual.")} == residual_fields
    # every other field is exposed except the derived / not-implemented ones and the residual
    # sub-section (which is exposed field-by-field above), both documented
    excluded = {"d", "context_depth", "residual"}
    assert {k.split(".")[1] for k in exposed.values() if k.startswith("vector.") and not k.startswith("vector.residual.")} == vector_fields - excluded
    assert {k.split(".")[1] for k in exposed.values() if k.startswith("memory.")} == memory_fields
    assert {k.split(".")[1] for k in exposed.values() if k.startswith("precision.")} == precision_fields


def test_defaults_come_from_the_backend(tiny_config_path, tiny_checkpoint):
    state = panel.default_state(config_path=tiny_config_path, checkpoint=tiny_checkpoint)
    v2 = V2Config.from_config(Config(TINY_CONFIG), warn=False)
    assert state["structured_d"] == v2.vector.structured_d == 48
    assert state["learned_residual_d"] == v2.vector.learned_residual_d == 0
    assert state["d"] == 48
    assert state["enabled_blocks"] == list(v2.vector.enabled_blocks)
    assert state["residual_epochs"] == v2.vector.residual.epochs == 200
    assert state["residual_hidden_dim"] == v2.vector.residual.hidden_dim == 64
    assert state["residual_batch_size"] == v2.vector.residual.batch_size == 64
    assert state["residual_learning_rate"] == pytest.approx(v2.vector.residual.learning_rate)
    assert state["residual_mask_fraction"] == pytest.approx(v2.vector.residual.mask_fraction)
    assert state["residual_val_fraction"] == pytest.approx(v2.vector.residual.val_fraction)
    assert state["residual_mask_mode"] == v2.vector.residual.mask_mode == "coordinate"
    assert state["residual_standardization"] in RESIDUAL_STANDARDIZATIONS
    assert state["vector_dtype"] in SUPPORTED_DTYPES
    assert state["network_context"] == "not implemented"
    assert state["preset"] is None


def test_immediate_validation_rejects_bad_controls(panel_state):
    for bad in (
        {"structured_d": 0},
        {"structured_d": -1},
        {"learned_residual_d": -5},
        {"enabled_blocks": []},
        {"enabled_blocks": ["nonsense"]},
        {"enabled_blocks": ["intrinsic", "network_context"]},
        {"residual_mask_fraction": 1.5},
        {"residual_val_fraction": 0.9},
        {"residual_epochs": 0},
        {"functional_source_dim": 0},
        {"functional_source_normalization": "zscore"},
        {"residual_standardization": "zscore"},
        {"residual_mask_mode": "diagonal"},
        {"device": "tpu"},
        {"storage": "tape"},
        {"vector_dtype": "int8"},
        {"preset": "not_a_preset"},
        {"d": 99},
        {"n_layers": 3},
        {"nonsense_control": 1},
    ):
        with pytest.raises(panel.PanelError):
            panel.set_controls(dict(panel_state), **bad)


def test_set_controls_is_atomic(panel_state):
    """An invalid request must not leave a half-updated state behind."""
    state = dict(panel_state)
    before = dict(state)
    with pytest.raises(panel.PanelError):
        panel.set_controls(state, residual_epochs=10, structured_d=0)   # 2nd value invalid
    assert state == before


def test_blocks_accept_a_comma_separated_string(panel_state):
    state = dict(panel_state)
    panel.set_controls(state, enabled_blocks="intrinsic, recurrent_in")
    assert state["enabled_blocks"] == ["intrinsic", "recurrent_in"]


def test_dimension_check_reports_the_invariant(panel_state):
    state = dict(panel_state)
    check = panel.dimension_check(state)
    assert check["ok"] and check["expression"] == "48 = 48 structured + 0 residual"
    panel.set_controls(state, structured_d=48, learned_residual_d=16)
    assert not panel.dimension_check(state)["ok"]           # residual capacity, residual disabled
    panel.set_controls(state, residual_enabled=True)
    check = panel.dimension_check(state)
    assert check["ok"] and check["expression"] == "64 = 48 structured + 16 residual"
    panel.set_controls(state, source_functional_response=True)
    assert panel.dimension_check(state)["ok"]
    panel.set_controls(state, residual_enabled=False)
    problems = panel.dimension_check(state)["problems"]
    assert any("functional_response" in p for p in problems)


# --------------------------------------------------------------------------
# Delta overrides and resolution
# --------------------------------------------------------------------------
def test_overrides_are_only_the_changes(panel_state):
    state = dict(panel_state)
    assert panel.state_to_overrides(state) == []                 # preset alone -> no overrides
    panel.set_controls(state, structured_d=64)
    assert panel.state_to_overrides(state) == ["vector.structured_d=64"]
    panel.set_controls(state, residual_enabled=True, learned_residual_d=16, residual_seed=2)
    overrides = panel.state_to_overrides(state)
    assert "vector.structured_d=64" in overrides
    assert "vector.residual.enabled=true" in overrides
    assert "vector.residual.seed=2" in overrides


def test_state_resolves_through_the_backend(panel_state, tiny_config_path, tiny_checkpoint):
    state = dict(panel_state)
    resolved = panel.validate_state(state)
    assert isinstance(resolved, ResolvedConfig)
    assert (resolved.d, resolved.structured_d, resolved.residual_d) == (48, 48, 0)
    assert resolved.run_id == panel.state_run_id(state)
    assert resolved.checkpoint_sha256_16  # the checkpoint was fingerprinted

    # the resolved values are exactly the ones the pure CLI path produces
    from src.v2_pipeline import resolve_config

    control = resolve_config(config_path=tiny_config_path, preset="historical_48",
                             checkpoint=tiny_checkpoint, out_dir=state["out_dir"])
    assert resolved.run_id == control.run_id
    assert resolved.cli_overrides == ()


def test_backend_rejection_surfaces_as_panel_error(panel_state):
    state = dict(panel_state)
    panel.set_controls(state, learned_residual_d=16, residual_enabled=True)
    panel.set_controls(state, source_functional_response=True, functional_source_normalization="raw")
    assert panel.validate_state(state).residual_d == 16
    # a state the backend must refuse: residual capacity with the residual disabled
    panel.set_controls(state, residual_enabled=False)
    with pytest.raises(panel.PanelError, match="requires"):
        panel.validate_state(state)


def test_protocol_drift_is_detected(panel_state):
    state = dict(panel_state)
    assert panel.protocol_drift(state) == {}
    panel.set_controls(state, residual_epochs=20, residual_mask_mode="block")
    drift = panel.protocol_drift(state)
    assert drift["residual_epochs"] == (20, 200)
    assert drift["residual_mask_mode"] == ("block", "coordinate")


def test_preview_and_reproducibility_blocks(panel_state):
    resolved = panel.validate_state(dict(panel_state))
    preview = panel.resolved_preview(resolved)
    for group in ("CHECKPOINT", "MODEL", "FIT / DEV / PROBE / TEST POLICY", "TOTAL DIMENSION",
                  "STRUCTURED BLOCKS", "FUNCTIONAL RESPONSE", "TEMPORAL", "RESIDUAL",
                  "MEMORY / PRECISION", "SEEDS", "EVALUATION"):
        assert group in preview
    text = "\n".join(panel.preview_lines(resolved))
    assert "never accessed" in text
    assert dict(preview["TOTAL DIMENSION"])["total"].startswith("48  (48 + 0)")
    assert dict(preview["STRUCTURED BLOCKS"])["network_context"] == "not implemented (not selectable)"
    repro = "\n".join(panel.reproducibility_lines(resolved))
    assert resolved.run_id in repro and "v2_control_panel.py" in repro
    assert panel.safety_lines()[2].startswith("TEST")


def test_checkpoint_mismatches_anchors_a_relative_config_path(monkeypatch, tmp_path, tiny_checkpoint):
    """Regression: a relative config path is resolved against the repo root, not the cwd."""
    monkeypatch.chdir(tmp_path)
    result = panel.checkpoint_mismatches(tiny_checkpoint, panel.DEFAULT_CONFIG_PATH)
    assert set(result) == {"compatible", "mismatches"}
    assert isinstance(result["mismatches"], dict)


def test_cache_inventory_and_checkpoints(tiny_config_path, tiny_checkpoint):
    inventory = panel.cache_inventory(config_path=tiny_config_path, checkpoint=tiny_checkpoint)
    assert inventory["presets"] == list(available_presets())
    assert len(inventory["checkpoints"]) >= 1
    candidates = panel.checkpoint_candidates()
    assert candidates and all(c["path"].endswith(".pt") for c in candidates)
    details = panel.checkpoint_details(tiny_checkpoint)
    assert details["architecture"]["n_hidden"] == 16
    assert len(details["sha256_16"]) == 16
    assert panel.checkpoint_mismatches(tiny_checkpoint, tiny_config_path)["compatible"] is True
    with pytest.raises(panel.PanelError):
        panel.checkpoint_details("checkpoints/does_not_exist.pt")


# --------------------------------------------------------------------------
# Build / inspect / evaluate through the panel state (hermetic, tiny)
# --------------------------------------------------------------------------
def test_panel_build_48d_matches_the_established_baseline(panel_state, tiny_checkpoint):
    """The key regression anchor: the panel's default == the deterministic 48-D baseline."""
    state = dict(panel_state)
    build = panel.build_from_state(state, use_cache=False)
    assert isinstance(build, BuildResult)
    assert (build.d, build.structured_d, build.residual_d) == (48, 48, 0)
    assert build.residual_cache_path is None

    artifact = panel.load_from_state(state)
    model, _ = RecurrentLIFSNN.load(str(tiny_checkpoint), map_location="cpu")
    from src.v2_pipeline import build_fit_probe_recordings

    bundle = build_fit_probe_recordings(panel.resolve_state(state))
    bank = build_neuron_record_bank(model, fit_rec=bundle["fit"], device="cpu", batch_size=32)
    reference, names = structural_matrix_from_record_bank(bank)
    reference_encoder = StructuredVectorEncoder(bank, structured_d=48).encode()
    assert np.abs(artifact.X - reference).max() == 0.0
    assert np.abs(artifact.X - reference_encoder.X).max() == 0.0
    assert artifact.feature_names == tuple(names)


def test_panel_functional_64_smoke_uses_the_functional_source(tiny_config_path, tiny_checkpoint, tmp_path):
    """functional_64 smoke: 48 + 16, residual input view includes the functional-response source."""
    state = panel.default_state(
        preset="functional_64", config_path=tiny_config_path, checkpoint=tiny_checkpoint,
        out_dir=tmp_path / "functional",
    )
    panel.set_controls(state, residual_epochs=2, residual_hidden_dim=4)   # fast smoke protocol
    assert panel.dimension_check(state)["expression"] == "64 = 48 structured + 16 residual"
    build = panel.build_from_state(state, use_cache=False)
    assert (build.d, build.structured_d, build.residual_d) == (64, 48, 16)

    info = panel.inspect_from_state(state)
    assert info["n_coordinates"] == 64 and info["finite"] and info["n_nan"] == 0 and info["n_inf"] == 0
    assert info["coordinate_groups"]["residual"] == 16
    assert info["residual_source"] is not None
    assert info["residual_source"]["protocol"]["source_functional_response"] is True
    assert "functional_response" in info["residual_source"]["source_groups"]
    assert info["residual_source"]["residual_dim"] == 16
    assert info["provenance"]["dimensions"]["expression"] == "64 = 48 structured + 16 residual"


def test_panel_inspection_reports_the_no_residual_case(panel_state):
    info = panel.inspect_from_state(dict(panel_state))
    assert info["residual_source"] is None
    assert info["d"] == 48 and info["n_coordinates"] == 48
    assert info["coordinate_groups"] == {
        "intrinsic": 1, "input_conn": 14, "recurrent_in": 19, "recurrent_out": 14,
    }
    assert len(info["matrix_sha256"]) == 64


def test_load_before_build_is_a_clear_error(panel_state):
    state = dict(panel_state)
    state["out_dir"] = str(Path(state["out_dir"]) / "never_built")
    with pytest.raises(panel.PanelError, match="build"):
        panel.load_from_state(state)


def test_panel_evaluation_is_separate_and_uses_probe(panel_state):
    state = dict(panel_state)
    panel.set_controls(state, evaluation_target="neuron_zscored", n_perm=10, bootstrap=0)
    panel.build_from_state(state, use_cache=False)
    evaluation = panel.evaluate_from_state(state)
    assert isinstance(evaluation, EvaluationResult)
    assert [row["target_variant"] for row in evaluation.rows] == ["neuron_zscored"]
    assert evaluation.settings["n_perm"] == 10
    assert evaluation.evaluation_path.exists()
    payload = json.loads(evaluation.evaluation_path.read_text(encoding="utf-8"))
    assert payload["data_policy"]["official_test_loaded"] is False
    # the notice the notebook prints before evaluating
    assert panel.evaluation_notice() == [
        "Evaluation uses PROBE.",
        "Representation construction remains FIT-only.",
        "TEST is untouched.",
    ]


def test_dry_run_state_loads_nothing(monkeypatch, panel_state):
    import src.v2_pipeline as pipeline

    monkeypatch.setattr(
        pipeline, "build_fit_probe_recordings",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("dry run must not load data")),
    )
    monkeypatch.setattr(
        pipeline, "RecurrentLIFSNN",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("dry run must not load the model")),
    )
    plan = panel.dry_run_state(dict(panel_state))
    assert plan["dimension"]["expression"] == "48 = 48 structured + 0 residual"
    assert plan["data"]["policy"]["official_test_loaded"] is False


def test_validation_verifies_the_checkpoint_architecture(panel_state):
    """The panel's validation fails fast on a checkpoint/model mismatch (the dry run stays cheap)."""
    state = dict(panel_state)
    assert panel.validate_state(state).d == 48                 # matching checkpoint -> fine
    panel.set_controls(state, n_hidden=64)                     # no such checkpoint
    with pytest.raises(panel.PanelError, match="architecture mismatch"):
        panel.validate_state(state)
    # resolve_state alone stays model-free (it only fingerprints the checkpoint)
    assert panel.resolve_state(state).d == 48
    assert panel.dimension_check(state)["ok"]


def test_widgets_are_optional_and_never_required(panel_state):
    support = panel.widget_support()
    assert set(support) == {"available", "reason", "fallback"}
    widgets = panel.make_widgets(dict(panel_state))
    if support["available"]:                       # pragma: no cover - depends on the environment
        assert widgets is not None
    else:
        assert widgets is None and "ipywidgets" in support["reason"]


# --------------------------------------------------------------------------
# The notebook itself
# --------------------------------------------------------------------------
def _notebook() -> dict:
    assert NOTEBOOK.exists(), f"{NOTEBOOK} is missing"
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells(notebook: dict) -> list[dict]:
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def _tagged(notebook: dict, tag: str) -> list[dict]:
    return [c for c in _code_cells(notebook) if tag in c["metadata"].get("tags", [])]


def test_notebook_is_valid_nbformat():
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    assert notebook["cells"], "the notebook has no cells"
    ids = [c["id"] for c in notebook["cells"]]
    assert len(ids) == len(set(ids)), "cell ids must be unique"
    assert notebook["metadata"]["v2_panel"]["backend"] == "src/v2_pipeline.py"
    assert notebook["metadata"]["v2_panel"]["entry_point"] == "src/v2_panel.py"
    kinds = {c["cell_type"] for c in notebook["cells"]}
    assert kinds == {"markdown", "code"}


def test_notebook_cells_compile_and_do_not_install_packages():
    notebook = _notebook()
    for index, cell in enumerate(_code_cells(notebook), start=1):
        source = "".join(cell["source"])
        assert not source.lstrip().startswith(("%", "!")), f"code cell {index} uses a shell magic"
        compile(source, f"<notebook cell {index}>", "exec")
        for forbidden in ("pip install", "!pip", "%pip", "conda install", "apt-get", "uv add",
                          "uv pip install"):
            assert forbidden not in source, f"code cell {index} tries to install packages"


def test_notebook_delegates_to_the_backend_and_duplicates_no_metrics():
    notebook = _notebook()
    code = "\n".join("".join(c["source"]) for c in _code_cells(notebook))
    # it must go through the panel/backend ...
    assert "from src import v2_panel as panel" in code
    assert "from src.v2_pipeline import" in code
    for needed in ("panel.validate_state", "panel.build_from_state", "panel.inspect_from_state",
                   "panel.evaluate_from_state", "panel.default_state", "panel.set_controls"):
        assert needed in code
    # ... and must not reimplement the science
    for forbidden in ("def mantel", "def geometry", "np.corrcoef", "scipy.stats", "spearmanr",
                      "def normalize", "def encode", "torch.nn", "def train_residual",
                      "def masked_reconstruction"):
        assert forbidden not in code, f"the notebook reimplements {forbidden!r}"


def test_notebook_has_the_required_sections_and_tags():
    notebook = _notebook()
    markdown = "\n".join("".join(c["source"]) for c in notebook["cells"]
                         if c["cell_type"] == "markdown")
    for section in ("## 1. Environment", "## 2. Presets", "## 3. Representation dimensions",
                    "## 4. Structural blocks", "## 5. Functional-response source",
                    "## 6. Temporal source", "## 7. Residual controls",
                    "## 8. Compute / memory / precision", "## 9. Checkpoint",
                    "## 10. Evaluation settings", "## 11. Validate", "## 12. Safety",
                    "## 13. Build / load", "## 14. Inspect", "## 15. Evaluate on PROBE",
                    "## 16. Run identity", "## 17. Reproducibility", "## 18. Parent summary",
                    "## 19. Workflow recipes", "## 20. Notes and limitations"):
        assert section in markdown, section
    assert "never accessed" in markdown or "TEST" in markdown
    assert _tagged(notebook, "panel:auto"), "the auto-executable cells are not tagged"
    assert len(_tagged(notebook, "panel:build")) == 2
    assert len(_tagged(notebook, "panel:evaluate")) == 1


def test_notebook_auto_cells_execute_without_building_anything(monkeypatch, tmp_path,
                                                               tiny_config_path, tiny_checkpoint):
    """Lightweight notebook execution test (kernel-free): runs every ``panel:auto`` cell in order."""
    out_dir = tmp_path / "notebook_out"
    monkeypatch.setenv("V2_PANEL_CONFIG", str(tiny_config_path))
    monkeypatch.setenv("V2_PANEL_CHECKPOINT", str(tiny_checkpoint))
    monkeypatch.setenv("V2_PANEL_OUT_DIR", str(out_dir))

    notebook = _notebook()
    source = "\n\n".join("".join(c["source"]) for c in _tagged(notebook, "panel:auto"))
    namespace: dict = {"__name__": "__notebook__"}
    exec(compile(source, "<notebook auto cells>", "exec"), namespace)  # noqa: S102 - test harness

    # the cells ran and left the expected results behind ...
    assert namespace["state"]["d"] == 48
    assert namespace["resolved"] is not None
    assert namespace["resolved"].run_id == panel.state_run_id(namespace["state"])
    assert namespace["resolved"].d == 48
    # ... without building anything or writing artifacts
    assert not (out_dir / "vectors").exists()
    assert not (out_dir / "resolved_configs").exists()
    assert namespace["state"]["enabled_blocks"] == [
        "intrinsic", "input_conn", "recurrent_in", "recurrent_out"
    ]
    assert namespace["check"]["ok"] is True


def test_notebook_auto_cells_do_not_read_probe_or_test(monkeypatch, tmp_path,
                                                       tiny_config_path, tiny_checkpoint):
    """The auto cells may not collect activity, train a residual, or open the test file."""
    import src.v2_pipeline as pipeline

    def _boom(*args, **kwargs):  # pragma: no cover - must never be called
        raise AssertionError("a panel:auto cell performed an expensive operation")

    monkeypatch.setenv("V2_PANEL_CONFIG", str(tiny_config_path))
    monkeypatch.setenv("V2_PANEL_CHECKPOINT", str(tiny_checkpoint))
    monkeypatch.setenv("V2_PANEL_OUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(pipeline, "collect_activity", _boom)
    monkeypatch.setattr(pipeline, "train_residual", _boom)
    monkeypatch.setattr(pipeline, "build_representation", _boom)

    notebook = _notebook()
    source = "\n\n".join("".join(c["source"]) for c in _tagged(notebook, "panel:auto"))
    exec(compile(source, "<notebook auto cells>", "exec"), {"__name__": "__notebook__"})  # noqa: S102


def test_notebook_build_cell_reaches_the_backend_through_the_state():
    notebook = _notebook()
    build_cell = "".join(_tagged(notebook, "panel:build")[0]["source"])
    assert "panel.build_from_state(state" in build_cell
    assert "use_cache" in build_cell
    evaluate_cell = "".join(_tagged(notebook, "panel:evaluate")[0]["source"])
    assert "panel.evaluate_from_state(state" in evaluate_cell
    assert "evaluation_notice" in evaluate_cell
