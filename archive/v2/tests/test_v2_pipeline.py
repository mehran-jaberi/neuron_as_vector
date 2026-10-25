"""Control-panel tests: presets, dimensions, dry run, run ids, caches, safety, stable API.

Everything runs on a *synthetic* dataset and a tiny checkpoint written to ``tmp_path``, so the
suite needs no downloads and no real checkpoint. The scientific studies are never re-run here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.model import SNNConfig, RecurrentLIFSNN, build_model
from src.neuron_record import build_neuron_record_bank
from src.neuron_vector import NeuronVectorArtifact, NeuronVectorError
from src.residual import (
    ResidualSourceConfig,
    ResidualTrainingConfig,
    build_residual_source,
    residual_training_mismatches,
    train_residual,
)
from src.structured_vector import StructuredVectorEncoder
from src.utils import Config, load_config
from src.v2_config import RESIDUAL_STANDARDIZATIONS, V2Config
from src.v2_pipeline import (
    DATA_POLICY,
    DEFAULT_CONFIG_PATH,
    PRESET_DECOMPOSITIONS,
    SCHEMA,
    PipelineError,
    V2Run,
    _load_or_train_residual,
    available_presets,
    build_fit_probe_recordings,
    build_representation,
    compute_run_id,
    dry_run_report,
    evaluate_representation,
    load_vector_artifact,
    parent_summary,
    residual_cache_path,
    resolve_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def _scripts_on_path():
    """Make the control-panel script importable (the same trick the config test uses)."""
    scripts_dir = str(PROJECT_ROOT / "scripts")
    added = scripts_dir not in sys.path
    if added:
        sys.path.insert(0, scripts_dir)
    yield
    if added and scripts_dir in sys.path:
        sys.path.remove(scripts_dir)

#: A tiny end-to-end configuration (synthetic data, 20 channels, 30 bins, 16 hidden neurons).
PANEL_CONFIG = {
    "seed": 0,
    "paths": {"data_dir": "data"},
    "run": {
        "synthetic": True,
        "synthetic_n_samples": 120,
        "synthetic_n_classes": 4,
        "synthetic_n_channels": 20,
    },
    "model": {
        "n_input": 20,
        "n_hidden": 16,
        "n_output": 4,
        "n_bins": 30,
        "bin_ms": 2.0,
        "neuron_param_mode": "bias",
        "readout_mode": "sum",
    },
    "train": {"n_classes": 4},
    "data": {
        "dev_fraction": 0.25,
        "probe_fraction": 0.25,
        "prefer_speaker_aware": True,
        "split_seed": 0,
    },
    "fingerprint": {"n_psth_bins": 5, "min_spikes_for_latency": 1.0},
}


@pytest.fixture(scope="module")
def panel_config_path(tmp_path_factory) -> Path:
    import yaml

    path = tmp_path_factory.mktemp("panel") / "panel.yaml"
    path.write_text(yaml.safe_dump(PANEL_CONFIG), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def panel_checkpoint(tmp_path_factory) -> Path:
    """A tiny *trained-like* checkpoint: the learned bias varies across neurons."""
    cfg = SNNConfig(
        n_input=20, n_hidden=16, n_output=4, n_bins=30, bin_ms=2.0, neuron_param_mode="bias"
    )
    model = build_model(cfg, seed=1)
    with torch.no_grad():
        model.b_hid.copy_(torch.linspace(-0.5, 0.5, cfg.n_hidden))
    path = tmp_path_factory.mktemp("panel_ckpt") / "panel_checkpoint.pt"
    model.save(str(path))
    return path


def _resolve(panel_config_path, panel_checkpoint, **kwargs):
    kwargs.setdefault("config_path", panel_config_path)
    kwargs.setdefault("checkpoint", panel_checkpoint)
    return resolve_config(**kwargs)


# --------------------------------------------------------------------------
# Defaults and dimension semantics
# --------------------------------------------------------------------------
def test_default_configuration_is_the_historical_48(panel_config_path, panel_checkpoint):
    r = _resolve(panel_config_path, panel_checkpoint)
    assert (r.d, r.structured_d, r.residual_d) == (48, 48, 0)
    assert r.v2.vector.enabled_blocks == [
        "intrinsic", "input_conn", "recurrent_in", "recurrent_out"
    ]
    assert r.residual_enabled is False
    assert r.v2.vector.residual.source_functional_response is False
    assert r.v2.vector.residual.source_temporal is False
    assert r.temporal_block_requested is False
    # the default must not request anything the pipeline cannot build
    assert r.v2.vector.unimplemented_blocks == []
    text = "\n".join(r.summary_lines())
    assert "d = 48" in text and "structured = 48" in text and "residual   = 0" in text


def test_relative_config_path_is_anchored_to_the_repo_root(monkeypatch, tmp_path):
    """Regression: the documented default config must resolve from any working directory.

    A notebook kernel started in ``notebooks/`` has a cwd that is not the repo root. The relative
    default path used to be handed straight to ``load_config``, which raised FileNotFoundError
    there; the checkpoint and the output directory were already anchored to the repo root.
    """
    monkeypatch.chdir(tmp_path)
    r = resolve_config()
    assert Path(r.config_path).is_absolute()
    assert Path(r.config_path) == PROJECT_ROOT / DEFAULT_CONFIG_PATH


@pytest.mark.parametrize(
    "preset,dims",
    [
        ("historical_48", (48, 48, 0)),
        ("structured_64", (64, 64, 0)),
        ("structured_100", (100, 100, 0)),
        ("temporal_structured", (58, 58, 0)),
        ("functional_64", (64, 48, 16)),
        ("functional_100", (100, 48, 52)),
    ],
)
def test_dimension_semantics(panel_config_path, panel_checkpoint, preset, dims):
    r = _resolve(panel_config_path, panel_checkpoint, preset=preset)
    assert (r.d, r.structured_d, r.residual_d) == dims
    assert r.d == r.structured_d + r.residual_d
    assert r.residual_enabled is (dims[2] > 0)


def test_presets_match_their_documented_decomposition(panel_config_path, panel_checkpoint):
    for preset in available_presets():
        r = _resolve(panel_config_path, panel_checkpoint, preset=preset)
        documented = PRESET_DECOMPOSITIONS[preset].split("(")[0]
        expected = f"{r.d} = {r.structured_d} structured + {r.residual_d} residual"
        assert documented.strip().startswith(expected), (preset, documented, expected)


def test_residual_source_presets_select_the_documented_sources(panel_config_path, panel_checkpoint):
    functional = _resolve(panel_config_path, panel_checkpoint, preset="functional_64")
    assert functional.v2.vector.residual.source_functional_response is True
    assert functional.v2.vector.residual.source_temporal is False

    temporal = _resolve(panel_config_path, panel_checkpoint, preset="temporal_structured")
    assert "temporal" in temporal.v2.vector.enabled_blocks
    assert temporal.v2.vector.residual.source_temporal is False
    assert temporal.temporal_block_requested is True


# --------------------------------------------------------------------------
# Overrides
# --------------------------------------------------------------------------
def test_dotted_overrides_propagate(panel_config_path, panel_checkpoint):
    r = _resolve(
        panel_config_path,
        panel_checkpoint,
        preset="historical_48",
        overrides=["vector.structured_d=64", "vector.learned_residual_d=0"],
    )
    assert (r.d, r.structured_d, r.residual_d) == (64, 64, 0)
    assert r.cli_overrides == ("vector.structured_d=64", "vector.learned_residual_d=0")

    r2 = _resolve(
        panel_config_path,
        panel_checkpoint,
        preset="functional_64",
        overrides=["vector.residual.mask_mode=block", "vector.residual.epochs=7"],
    )
    assert r2.v2.vector.residual.mask_mode == "block"
    assert r2.v2.vector.residual.epochs == 7
    assert (r2.d, r2.structured_d, r2.residual_d) == (64, 48, 16)


def test_unknown_override_is_reported(panel_config_path, panel_checkpoint):
    r = _resolve(
        panel_config_path,
        panel_checkpoint,
        overrides=["vector.structured_dim=64", "nonsense.section=1"],
    )
    joined = " ".join(r.warnings)
    assert "structured_dim" in joined and "nonsense.section" in joined
    # the typo must not have changed the resolved representation
    assert (r.d, r.structured_d, r.residual_d) == (48, 48, 0)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "preset,overrides",
    [
        ("historical_48", ["vector.d=64"]),                                   # d != split
        ("historical_48", ["vector.d=-4"]),                                   # negative
        ("historical_48", ["vector.structured_d=-1"]),                        # negative
        (None, ["vector.d=64", "vector.structured_d=48", "vector.learned_residual_d=16"]),  # disabled
        ("functional_64", ["vector.residual.enabled=false"]),                 # residual_d with disabled
        ("historical_48", ["vector.residual.enabled=true"]),                  # enabled with residual_d=0
        ("historical_48", ["vector.enabled_blocks=[intrinsic,network_context]"]),  # not implemented
        ("historical_48", ["vector.residual.source_functional_response=true"]),  # source without residual
        ("historical_48", ["vector.residual.mask_mode=diagonal"]),            # unknown mask mode
        ("historical_48", ["vector.residual.standardization=zscore"]),        # unknown standardization
        ("historical_48", ["vector.residual.epochs=0"]),                      # invalid epoch count
        ("historical_48", ["vector.residual.val_fraction=0.75"]),             # invalid split fraction
        ("historical_48", ["vector.functional_source_dim=0"]),                # invalid source dim
        ("historical_48", ["model.n_layers=2"]),                              # multi-layer
        ("historical_48", ["precision.model_dtype=float16"]),                 # not implemented
        ("historical_48", ["memory.mixed_precision=true"]),                   # not implemented
        ("historical_48", ["memory.record_batch_size=0"]),                    # invalid memory setting
    ],
)
def test_invalid_configurations_are_rejected(panel_config_path, panel_checkpoint, preset, overrides):
    with pytest.raises(PipelineError):
        _resolve(panel_config_path, panel_checkpoint, preset=preset, overrides=overrides)


def test_temporal_encoder_block_resolves(panel_config_path, panel_checkpoint):
    """A temporal *encoder* block is implemented (it needs the FIT temporal block)."""
    r = _resolve(
        panel_config_path, panel_checkpoint, preset="historical_48",
        overrides=["vector.enabled_blocks=[intrinsic,input_conn,recurrent_in,recurrent_out,temporal]"],
    )
    assert "temporal" in r.v2.vector.enabled_blocks
    assert r.temporal_block_requested is True
    assert "temporal" in r.present_block_names()


def test_unsupported_evaluation_mode_is_rejected(panel_config_path, panel_checkpoint):
    with pytest.raises(PipelineError):
        _resolve(panel_config_path, panel_checkpoint, evaluation_mode="fancy")
    with pytest.raises(PipelineError):
        _resolve(panel_config_path, panel_checkpoint, evaluation_target="nonsense")


def test_unknown_preset_is_rejected(panel_config_path, panel_checkpoint):
    with pytest.raises(PipelineError, match="unknown preset"):
        _resolve(panel_config_path, panel_checkpoint, preset="not_a_preset")


def test_missing_checkpoint_is_reported_as_a_note(tmp_path, panel_config_path):
    r = _resolve(panel_config_path, tmp_path / "does_not_exist.pt")
    assert r.checkpoint_exists is False
    assert any("does not exist" in note for note in r.notes)
    with pytest.raises(PipelineError):
        build_representation(r)


# --------------------------------------------------------------------------
# Dry run
# --------------------------------------------------------------------------
def test_dry_run_loads_no_data(monkeypatch, panel_config_path, panel_checkpoint):
    import src.v2_pipeline as pipeline

    def _boom(*args, **kwargs):  # pragma: no cover - must never be called
        raise AssertionError("dry run must not load data or train anything")

    monkeypatch.setattr(pipeline, "build_fit_probe_recordings", _boom)
    monkeypatch.setattr(pipeline, "collect_activity", _boom)
    monkeypatch.setattr(pipeline, "train_residual", _boom)
    monkeypatch.setattr(pipeline, "RecurrentLIFSNN", _boom)

    r = _resolve(panel_config_path, panel_checkpoint, preset="functional_64")
    plan = dry_run_report(r)
    assert plan["run_id"] == r.run_id
    assert plan["dimension"]["expression"] == "64 = 48 structured + 16 residual"
    assert plan["checkpoint"]["exists"] is True
    assert isinstance(plan["phases"]["build"], list) and isinstance(plan["phases"]["evaluate"], list)
    assert plan["data"]["policy"]["official_test_loaded"] is False


def test_dry_run_does_not_write_artifacts(tmp_path, panel_config_path, panel_checkpoint):
    out_dir = tmp_path / "panel_out"
    r = _resolve(panel_config_path, panel_checkpoint, out_dir=out_dir)
    dry_run_report(r)
    assert not out_dir.exists()
    assert not r.vector_artifact_path.exists()


def test_dry_run_cli_prints_the_plan(capsys, panel_config_path, panel_checkpoint, tmp_path):
    import v2_control_panel

    rc = v2_control_panel.main([
        "--config", str(panel_config_path), "--checkpoint", str(panel_checkpoint),
        "--preset", "historical_48", "--dry-run", "--out-dir", str(tmp_path / "out"),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry run: no data is loaded, nothing is trained, PROBE is not evaluated" in out
    assert "48 = 48 structured + 0 residual" in out
    assert "TEST  -> never accessed" in out


def test_cli_without_a_phase_only_inspects(capsys, panel_config_path, panel_checkpoint, tmp_path):
    import v2_control_panel

    rc = v2_control_panel.main([
        "--config", str(panel_config_path), "--checkpoint", str(panel_checkpoint),
        "--preset", "structured_64", "--out-dir", str(tmp_path / "out"),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "no phase selected" in out
    assert not (tmp_path / "out").exists()


def test_cli_json_output_is_machine_readable(capsys, panel_config_path, panel_checkpoint, tmp_path):
    import v2_control_panel

    rc = v2_control_panel.main([
        "--config", str(panel_config_path), "--checkpoint", str(panel_checkpoint),
        "--preset", "functional_100", "--dry-run", "--json", "--out-dir", str(tmp_path / "out"),
    ])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == SCHEMA
    assert payload["phase"] == "inspect"
    assert payload["dry_run"]["dimension"]["expression"] == "100 = 48 structured + 52 residual"
    config = payload["resolved_config"]
    assert config["representation"]["d"] == 100
    assert config["representation"]["structured_d"] == 48
    assert config["representation"]["residual_d"] == 52
    assert config["data_policy"]["test"] == "never accessed"
    assert config["preset_decomposition"].startswith("100 = 48 structured + 52 residual")
    assert payload["data_policy"] == config["data_policy"]


def test_cli_lists_presets(capsys):
    import v2_control_panel

    assert v2_control_panel.main(["--list-presets"]) == 0
    out = capsys.readouterr().out
    for preset in available_presets():
        assert preset in out


# --------------------------------------------------------------------------
# Run identity
# --------------------------------------------------------------------------
def test_run_id_is_deterministic_and_configuration_sensitive(panel_config_path, panel_checkpoint):
    a = _resolve(panel_config_path, panel_checkpoint, preset="functional_64")
    b = _resolve(panel_config_path, panel_checkpoint, preset="functional_64")
    assert a.run_id == b.run_id
    c = _resolve(panel_config_path, panel_checkpoint, preset="functional_100")
    assert a.run_id != c.run_id
    # a protocol change is a meaningful change
    d = _resolve(
        panel_config_path, panel_checkpoint, preset="functional_64",
        overrides=["vector.residual.epochs=13"],
    )
    assert d.run_id != a.run_id
    # evaluation settings are NOT part of the representation identity
    e = _resolve(panel_config_path, panel_checkpoint, preset="functional_64", n_perm=17)
    assert e.run_id == a.run_id
    # the same run id for a different checkpoint
    other = _resolve(panel_config_path, panel_checkpoint, preset="functional_64")
    other_checkpoint_id = compute_run_id(other.v2, checkpoint_id="deadbeefdeadbeef", preset="functional_64")
    assert other_checkpoint_id != a.run_id
    assert len(a.run_id) == 16 and all(ch in "0123456789abcdef" for ch in a.run_id)


def test_run_id_ignores_wall_clock(panel_config_path, panel_checkpoint):
    r = _resolve(panel_config_path, panel_checkpoint, preset="historical_48")
    assert r.created_utc  # recorded separately
    assert r.run_id == compute_run_id(r.v2, checkpoint_id=r.checkpoint_id)


def test_run_id_depends_on_values_not_on_the_preset_label(panel_config_path, panel_checkpoint):
    """Two ways of stating the same resolved configuration share one identity (and cache)."""
    via_preset = _resolve(panel_config_path, panel_checkpoint, preset="structured_64")
    via_overrides = _resolve(
        panel_config_path, panel_checkpoint,
        overrides=["vector.structured_d=64", "vector.learned_residual_d=0"],
    )
    assert via_preset.run_id == via_overrides.run_id
    assert via_preset.to_dict()["representation"] == via_overrides.to_dict()["representation"]
    assert via_preset.preset != via_overrides.preset


# --------------------------------------------------------------------------
# Build phase (FIT only) and the stable API
# --------------------------------------------------------------------------
def test_build_matches_the_lower_level_modules_exactly(tmp_path, panel_config_path, panel_checkpoint):
    r = _resolve(panel_config_path, panel_checkpoint, out_dir=tmp_path / "out")
    build = build_representation(r, use_cache=False)
    assert (build.n_neurons, build.d, build.structured_d, build.residual_d) == (16, 48, 48, 0)
    assert build.residual_trained_now is None and build.residual_cache_path is None

    artifact = load_vector_artifact(r)
    assert artifact.d == 48 and artifact.residual_d == 0

    # rebuild the bank independently and compare with the established 48-D encoder
    bundle = build_fit_probe_recordings(r)
    model, _ = RecurrentLIFSNN.load(str(panel_checkpoint), map_location="cpu")
    bank = build_neuron_record_bank(model, fit_rec=bundle["fit"], device="cpu", batch_size=32)
    reference = StructuredVectorEncoder(bank, structured_d=48).encode()
    assert np.array_equal(artifact.X, reference.X)
    assert artifact.feature_names == tuple(reference.feature_names)
    # residual_d = 0 must not load or instantiate a residual
    assert artifact.provenance["residual_cache"] is None
    assert artifact.provenance["composition"]["residual"]["used"] is False


def test_build_writes_resolved_config_and_artifact(tmp_path, panel_config_path, panel_checkpoint):
    out_dir = tmp_path / "out"
    run = V2Run.from_config(
        config_path=panel_config_path, checkpoint=panel_checkpoint,
        preset="structured_100", out_dir=out_dir,
    )
    build = run.build_representation(use_cache=False)
    assert build.artifact_path.exists()
    assert run.resolved.vector_artifact_path.exists()
    payload = json.loads((out_dir / "resolved_configs" / f"{run.run_id}.json").read_text(encoding="utf-8"))
    assert payload["run_id"] == run.run_id
    assert payload["representation"]["d"] == 100
    assert payload["preset"] == "structured_100"
    assert payload["data_policy"]["official_test_loaded"] is False
    assert payload["component_schemas"]["neuron_vector_artifact"].endswith("/v1")
    # the recorded identity is exactly what the run id hashes
    from hashlib import sha256

    identity = payload["run_identity"]
    assert identity["checkpoint"] == run.resolved.checkpoint_id
    assert "preset" not in identity
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
    assert sha256(canonical.encode("utf-8")).hexdigest()[:16] == run.run_id


def test_build_is_reproducible(tmp_path, panel_config_path, panel_checkpoint):
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    a = build_representation(
        _resolve(panel_config_path, panel_checkpoint, preset="structured_64", out_dir=out_a),
        use_cache=False,
    )
    b = build_representation(
        _resolve(panel_config_path, panel_checkpoint, preset="structured_64", out_dir=out_b),
        use_cache=False,
    )
    assert a.run_id == b.run_id
    assert np.array_equal(load_vector_artifact(_resolve(
        panel_config_path, panel_checkpoint, preset="structured_64", out_dir=out_a
    )).X, load_vector_artifact(_resolve(
        panel_config_path, panel_checkpoint, preset="structured_64", out_dir=out_b
    )).X)


def test_residual_build_composes_structured_and_residual(tmp_path, panel_config_path, panel_checkpoint):
    r = _resolve(
        panel_config_path, panel_checkpoint, preset="functional_64",
        out_dir=tmp_path / "out",
        overrides=["vector.residual.epochs=2", "vector.residual.hidden_dim=4"],
    )
    build = build_representation(r, use_cache=False)
    assert (build.d, build.structured_d, build.residual_d) == (64, 48, 16)
    assert build.residual_trained_now is True
    artifact = load_vector_artifact(r)
    assert artifact.X.shape[1] == 64
    assert artifact.feature_names[48].startswith("residual[")
    # the first 48 coordinates are the historical structural vector
    model, _ = RecurrentLIFSNN.load(str(panel_checkpoint), map_location="cpu")
    bank = build_neuron_record_bank(
        model, fit_rec=build_fit_probe_recordings(r)["fit"], device="cpu", batch_size=32
    )
    reference = StructuredVectorEncoder(bank, structured_d=48).encode()
    assert np.array_equal(artifact.X[:, :48], reference.X)


# --------------------------------------------------------------------------
# Cache integrity
# --------------------------------------------------------------------------
def test_residual_cache_protocol_is_verified(tmp_path, panel_config_path, panel_checkpoint):
    """A cached residual trained under a different protocol must not be reused silently."""
    r = _resolve(
        panel_config_path, panel_checkpoint, preset="functional_64",
        out_dir=tmp_path / "out",
        overrides=["vector.residual.epochs=2", "vector.residual.hidden_dim=4"],
    )
    bundle = build_fit_probe_recordings(r)
    model, _ = RecurrentLIFSNN.load(str(panel_checkpoint), map_location="cpu")
    bank = build_neuron_record_bank(model, fit_rec=bundle["fit"], device="cpu", batch_size=32)
    source_config = ResidualSourceConfig.from_v2_config(r.v2)
    source = build_residual_source(bank, source_config)
    cache_path = tmp_path / "out" / "residuals" / "artifact.pt"

    short = ResidualTrainingConfig.from_v2_config(r.v2)
    short.epochs = 2
    residual = train_residual(bank, config=short, source=source)
    residual.save(cache_path)

    longer = ResidualTrainingConfig.from_v2_config(r.v2)
    longer.epochs = 3
    assert residual_training_mismatches(residual, longer) == {"epochs": (2, 3)}
    assert residual_training_mismatches(residual, short) == {}

    reused, trained_now, mismatches = _load_or_train_residual(
        bank=bank, source_config=source_config, training_config=longer,
        cache_path=cache_path, use_cache=True,
    )
    assert trained_now is True, "a protocol mismatch must retrain instead of reusing"
    assert mismatches == {"epochs": (2, 3)}

    again, trained_now_again, no_mismatch = _load_or_train_residual(
        bank=bank, source_config=source_config, training_config=longer,
        cache_path=cache_path, use_cache=True,
    )
    assert trained_now_again is False and no_mismatch == {}
    assert again.residual_dim == reused.residual_dim


def test_residual_cache_path_depends_on_source_and_seed(tmp_path, panel_config_path, panel_checkpoint):
    base = _resolve(panel_config_path, panel_checkpoint, preset="functional_64", out_dir=tmp_path / "out")
    common = {"checkpoint": base.checkpoint_id, "checkpoint_path": "x", "n_bins": 30, "bin_ms": 2.0,
              "split_seed": 0, "split_strategy": "speaker_aware", "source": "synthetic_train"}
    p = residual_cache_path(base, cache_common=common, source_schema_hash="a" * 64,
                            residual_d=16, seed=0, mask_mode="coordinate")
    assert p.name.endswith("_dres16_seed0.pt") and "_maskblock" not in p.name
    blocked = residual_cache_path(base, cache_common=common, source_schema_hash="a" * 64,
                                 residual_d=16, seed=0, mask_mode="block")
    assert blocked.name.endswith("_dres16_seed0_maskblock.pt")
    other_seed = residual_cache_path(base, cache_common=common, source_schema_hash="a" * 64,
                                     residual_d=16, seed=1, mask_mode="coordinate")
    other_source = residual_cache_path(base, cache_common=common, source_schema_hash="b" * 64,
                                       residual_d=16, seed=0, mask_mode="coordinate")
    assert len({p.name, blocked.name, other_seed.name, other_source.name}) == 4


def test_incompatible_vector_artifacts_are_rejected(tmp_path, panel_config_path, panel_checkpoint):
    r = _resolve(panel_config_path, panel_checkpoint, out_dir=tmp_path / "out")
    build_representation(r, use_cache=False)
    path = r.vector_artifact_path
    with pytest.raises(NeuronVectorError):
        NeuronVectorArtifact.load(path, expected_run_id="deadbeefdeadbeef")
    with pytest.raises(NeuronVectorError):
        NeuronVectorArtifact.load(path, expected_d=99)
    with pytest.raises(NeuronVectorError):
        NeuronVectorArtifact.load(path, expected_residual_d=7)
    with pytest.raises(NeuronVectorError):
        NeuronVectorArtifact.load(path, expected_feature_names=("nope",) * 48)
    # a corrupted matrix fails the content-hash check
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["X"] = arrays["X"] + 1.0
    np.savez_compressed(path, **arrays)
    with pytest.raises(NeuronVectorError, match="integrity"):
        NeuronVectorArtifact.load(path)


def test_evaluation_requires_a_built_artifact(tmp_path, panel_config_path, panel_checkpoint):
    r = _resolve(panel_config_path, panel_checkpoint, out_dir=tmp_path / "out")
    with pytest.raises(FileNotFoundError):
        evaluate_representation(r, use_cache=False)


# --------------------------------------------------------------------------
# Evaluate phase (PROBE only, never TEST)
# --------------------------------------------------------------------------
def test_evaluate_uses_the_frozen_artifact_and_never_rebuilds(tmp_path, monkeypatch,
                                                              panel_config_path, panel_checkpoint):
    import src.v2_pipeline as pipeline

    r = _resolve(panel_config_path, panel_checkpoint, out_dir=tmp_path / "out", n_perm=10, bootstrap=0)
    build_representation(r, use_cache=False)
    monkeypatch.setattr(pipeline, "build_representation", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("evaluate must not rebuild the representation")
    ))
    result = evaluate_representation(r, use_cache=False)
    assert result.rows
    variants = {row["target_variant"] for row in result.rows}
    assert {"raw", "neuron_centered", "neuron_zscored", "mean_rate"} <= variants
    for row in result.rows:
        assert row["n_neurons"] == 16
        assert row["bootstrap_n"] == 0
        assert row["run_id"] == r.run_id
    payload = json.loads(result.evaluation_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == r.run_id
    assert payload["artifact"]["summary"]["run_id"] == r.run_id
    assert payload["data_policy"]["official_test_loaded"] is False
    assert payload["settings"]["n_perm"] == 10


def test_evaluate_single_target_and_secondary_target_selection(tmp_path, panel_config_path, panel_checkpoint):
    r = _resolve(
        panel_config_path, panel_checkpoint, out_dir=tmp_path / "out", n_perm=10, bootstrap=0,
        evaluation_target="neuron_zscored",
    )
    build_representation(r, use_cache=False)
    result = evaluate_representation(r, use_cache=False)
    assert [row["target_variant"] for row in result.rows] == ["neuron_zscored"]

    r2 = _resolve(
        panel_config_path, panel_checkpoint, out_dir=tmp_path / "out2", n_perm=10, bootstrap=0,
        evaluation_target="class_rate_20d",
    )
    build_representation(r2, use_cache=False)
    result2 = evaluate_representation(r2, use_cache=False)
    assert [row["target_variant"] for row in result2.rows] == ["class_rate_20d"]


def test_evaluate_cli_end_to_end(capsys, tmp_path, panel_config_path, panel_checkpoint):
    import v2_control_panel

    out_dir = tmp_path / "out"
    common = ["--config", str(panel_config_path), "--checkpoint", str(panel_checkpoint),
              "--preset", "historical_48", "--out-dir", str(out_dir)]
    assert v2_control_panel.main([*common, "--build"]) == 0
    capsys.readouterr()
    assert v2_control_panel.main([*common, "--evaluate", "--n-perm", "10", "--bootstrap", "0"]) == 0
    out = capsys.readouterr().out
    assert "[evaluate]" in out and "mantel_spearman_r" in out
    assert "TEST  -> never accessed" in out


# --------------------------------------------------------------------------
# Safety: FIT / PROBE / TEST
# --------------------------------------------------------------------------
def test_pipeline_source_never_opens_the_official_test_file():
    for name in ("v2_pipeline.py",):
        source = (PROJECT_ROOT / "src" / name).read_text(encoding="utf-8")
        assert "shd_test" not in source
        assert "test.h5" not in source
    script = (PROJECT_ROOT / "scripts" / "v2_control_panel.py").read_text(encoding="utf-8")
    assert "shd_test" not in script
    assert "test.h5" not in script


def test_recordings_are_fit_and_probe_only(panel_config_path, panel_checkpoint):
    r = _resolve(panel_config_path, panel_checkpoint)
    bundle = build_fit_probe_recordings(r)
    assert bundle["test_loaded"] is False
    assert bundle["source"] == "synthetic_train"
    assert len(bundle["fit"]) > 0 and len(bundle["probe"]) > 0
    assert DATA_POLICY["test"] == "never accessed"
    assert r.to_dict()["data_policy"]["probe"] == "evaluation targets and metrics only"


def test_build_does_not_touch_probe(capsys, tmp_path, monkeypatch, panel_config_path, panel_checkpoint):
    """The build phase must not collect labelled PROBE activity."""
    import src.v2_pipeline as pipeline

    calls: list[str] = []
    original = pipeline._load_or_collect_activity

    def _spy(*, role, **kwargs):
        calls.append(role)
        return original(role=role, **kwargs)

    monkeypatch.setattr(pipeline, "_load_or_collect_activity", _spy)
    r = _resolve(panel_config_path, panel_checkpoint, out_dir=tmp_path / "out")
    build_representation(r, use_cache=False)
    assert calls == ["fit"]


# --------------------------------------------------------------------------
# Parent summary
# --------------------------------------------------------------------------
def test_parent_summary_contains_every_required_section(capsys, panel_config_path, panel_checkpoint):
    import v2_control_panel

    rc = v2_control_panel.main([
        "--config", str(panel_config_path), "--checkpoint", str(panel_checkpoint),
        "--summary-for-parent", "--no-tests",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    required = [
        "=== V2 PARENT SUMMARY ===",
        "STATUS:", "TESTS:", "ACTIVE DOCS:", "CORE MODULES:", "CURRENT ARCHITECTURE:",
        "DIMENSION SEMANTICS:", "DEFAULT:", "AVAILABLE PRESETS:", "SOURCES:", "RESIDUAL:",
        "CONTROL PANEL:", "CACHE:", "SAFETY:", "NETWORK_CONTEXT:", "MULTI_LAYER:",
        "SCIENTIFIC STATE:", "NEXT STEP:", "=== END V2 PARENT SUMMARY ===",
    ]
    for header in required:
        assert header in out, header
    assert "not implemented" in out
    assert "d = structured_d + residual_d" in out
    assert "48 = 48 structured + 0 residual" in out


def test_parent_summary_function_does_not_touch_data(monkeypatch, panel_config_path, panel_checkpoint,
                                                     tmp_path):
    import src.v2_pipeline as pipeline

    monkeypatch.setattr(
        pipeline, "build_fit_probe_recordings",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("summary must not load data")),
    )
    r = _resolve(panel_config_path, panel_checkpoint, out_dir=tmp_path / "out")
    text = parent_summary(r, include_tests=False)
    assert "TEST" in text and "NETWORK_CONTEXT:" in text


def test_test_suite_status_handles_quiet_output(monkeypatch, tmp_path):
    """The repository's quiet pytest output omits the summary line: the exit code decides."""
    import subprocess as subprocess_module

    import src.v2_pipeline as pipeline

    class _Completed:
        def __init__(self, returncode: int, stdout: str):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = ""

    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        pipeline.subprocess, "run",
        lambda *a, **k: _Completed(0, "....................................  [100%]\n"),
    )
    monkeypatch.setattr(pipeline, "_pytest_collected_count", lambda **k: 535)

    status = pipeline.test_suite_status(run_tests=True)
    assert status["exit_code"] == 0
    assert status["collected"] == 535
    assert status["passed"] == 535
    assert status["failed"] == 0
    assert status["result_source"] == "exit_code"
    # the result is recorded so a later invocation can report it without re-running
    recorded = json.loads((tmp_path / "results" / "neuron_vector_capacity" / "last_test_run.json")
                          .read_text(encoding="utf-8"))
    assert recorded["passed"] == 535
    assert subprocess_module is not None

    monkeypatch.setattr(
        pipeline.subprocess, "run",
        lambda *a, **k: _Completed(1, "1 failed, 534 passed in 10.0s\n"),
    )
    failure = pipeline.test_suite_status(run_tests=True)
    assert failure["passed"] == 534 and failure["failed"] == 1
    assert failure["result_source"] == "summary_line"


# --------------------------------------------------------------------------
# Configuration-layer consistency (the residual is implemented, the protocol lives in V2Config)
# --------------------------------------------------------------------------
def test_v2_config_exposes_the_residual_protocol():
    from src.v2_config import VectorConfig

    v = VectorConfig.from_mapping(
        {"structured_d": 48, "learned_residual_d": 16, "residual": {"enabled": True}}
    )
    assert v.residual_implemented is True
    assert (v.residual.epochs, v.residual.hidden_dim, v.residual.batch_size) == (200, 64, 64)
    assert v.residual.learning_rate == pytest.approx(1e-3)
    assert v.residual.standardization in RESIDUAL_STANDARDIZATIONS
    # the established protocol is what ResidualTrainingConfig.from_v2_config now reads
    v2 = V2Config.from_config(
        Config({"vector": {"structured_d": 48, "learned_residual_d": 16, "residual": {"enabled": True}}}),
        warn=False,
    )
    training = ResidualTrainingConfig.from_v2_config(v2)
    assert (training.epochs, training.hidden_dim, training.batch_size) == (200, 64, 64)
    assert training.lr == pytest.approx(1e-3)
    assert training.mask_fraction == pytest.approx(0.25)
    assert training.val_fraction == pytest.approx(0.2)
    assert training.normalization == "train_standardise"
    assert v2.warnings == []  # the learned residual is implemented -> no warning


def test_residual_is_accepted_in_strict_mode():
    v2 = V2Config.from_config(
        Config({"vector": {"structured_d": 48, "learned_residual_d": 16, "residual": {"enabled": True}}}),
        strict=True, warn=False,
    )
    assert v2.vector.residual_implemented is True
    assert v2.warnings == []


def test_load_config_import_is_used(panel_config_path):
    """`load_config` (the repository's loader) resolves the same YAML the panel reads."""
    cfg = load_config(panel_config_path)
    assert cfg.get_path("model.n_hidden") == 16
    assert cfg.get_path("run.synthetic") is True
