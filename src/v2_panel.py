"""Notebook-facing control-surface layer over the stable V2 pipeline.

This module is the **only** code the Jupyter control panel
(``notebooks/V2_Control_Panel.ipynb``) needs. It contains no representation logic, no metric
and no fitting of its own: a *panel state* is a plain ``dict`` of editable values, and every
operation translates it into the existing backend calls

    state  ->  dotted --override list  ->  resolve_config / build_representation / evaluate_representation

so the notebook, the CLI (``scripts/v2_control_panel.py``) and the library all go through
:mod:`src.v2_pipeline` and produce identical configurations, run ids and artifacts.

Design rules

* **Single source of defaults.** :func:`default_state` reads every default from a resolved
  :class:`~src.v2_config.V2Config` (an empty config), so a control can never drift from the
  backend default, and a preset is expressed exactly as "the state that reproduces it".
* **No duplicated validation.** :func:`validate_state` calls the backend; :func:`set_controls`
  only performs immediate *shape* checks (type, allowed choice, known block) so a typo is caught
  before a config object is ever built, as the notebook requires.
* **Delta overrides.** :func:`state_to_overrides` emits only the values that differ from the
  preset baseline, so the notebook's "change one control" workflow produces a minimal,
  auditable override list (and therefore a run id that depends only on resolved values).
* **No widgets required.** The notebook uses clearly marked editable variables and calls
  :func:`make_widgets`, which returns ``None`` when ``ipywidgets`` is unavailable (this
  repository's environment does not include it); nothing here depends on a GUI toolkit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .model import RecurrentLIFSNN, architecture_mismatches
from .neuron_vector import NeuronVectorArtifact, NeuronVectorError
from .residual import ResidualTrainingConfig
from .utils import Config, load_config
from .v2_config import (
    DEFAULT_ENABLED_BLOCKS,
    FUNCTIONAL_SOURCE_NORMALIZATIONS,
    IMPLEMENTED_RECORD_BLOCKS,
    MAX_FUNCTIONAL_SOURCE_DIM,
    RESIDUAL_STANDARDIZATIONS,
    SOURCE_MASK_MODES,
    SUPPORTED_DEVICES,
    SUPPORTED_DTYPES,
    SUPPORTED_MODEL_TYPES,
    SUPPORTED_STORAGE,
    V2Config,
    V2ConfigError,
)
from .v2_pipeline import (
    DATA_POLICY,
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUT_DIR,
    DEFAULT_EVALUATION_TARGET,
    EVALUATION_TARGETS,
    PRESET_DECOMPOSITIONS,
    PRESET_DESCRIPTIONS,
    PROJECT_ROOT,
    SUPPORTED_EVALUATION_MODES,
    BuildResult,
    EvaluationResult,
    PipelineError,
    ResolvedConfig,
    V2Run,
    available_presets,
    build_representation,
    dry_run_report,
    evaluate_representation,
    load_vector_artifact,
    parent_summary,
    resolve_config,
)

#: Provenance schema of a panel state (recorded when a state is exported).
PANEL_SCHEMA = "v2_panel_state/v1"

#: Blocks the notebook may offer as selectable (everything implemented).
SELECTABLE_BLOCKS: tuple[str, ...] = tuple(
    b for b in ("intrinsic", "input_conn", "recurrent_in", "recurrent_out", "activity", "temporal")
    if b in IMPLEMENTED_RECORD_BLOCKS
)

#: Controls that are displayed but cannot be changed (unimplemented or derived).
DISPLAY_ONLY: tuple[str, ...] = ("d", "n_layers", "network_context")


class PanelError(ValueError):
    """Raised for an invalid panel state or a rejected backend request."""


# --------------------------------------------------------------------------
# Control metadata (labels, kinds, choices, grouping)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Control:
    """One editable value of a panel state."""

    name: str                      # state key (also the keyword of set_controls)
    key: str | None                # dotted configuration key (None = panel-only setting)
    kind: str                      # int | float | bool | str | choice | blocks | path
    group: str                     # panel section
    help: str = ""
    choices: tuple[Any, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    label: str | None = None

    @property
    def display(self) -> str:
        return self.label or self.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "config_key": self.key,
            "kind": self.kind,
            "group": self.group,
            "label": self.display,
            "choices": list(self.choices),
            "minimum": self.minimum,
            "maximum": self.maximum,
            "help": self.help,
        }


def _int(name: str, key: str | None, group: str, help: str = "", *, minimum: float | None = 1,
         maximum: float | None = None, label: str | None = None) -> Control:
    return Control(name, key, "int", group, help, (), minimum, maximum, label)


def _float(name: str, key: str | None, group: str, help: str = "", *, minimum: float | None = None,
           maximum: float | None = None, label: str | None = None) -> Control:
    return Control(name, key, "float", group, help, (), minimum, maximum, label)


def _bool(name: str, key: str, group: str, help: str = "", *, label: str | None = None) -> Control:
    return Control(name, key, "bool", group, help, (), None, None, label)


def _choice(name: str, key: str, group: str, choices: Sequence[Any], help: str = "",
            *, label: str | None = None) -> Control:
    return Control(name, key, "choice", group, help, tuple(choices), None, None, label)


#: Every control the panel exposes. ``key=None`` entries are panel-only settings (never
#: translated into configuration overrides); their defaults live in :func:`default_state`.
PANEL_CONTROLS: tuple[Control, ...] = (
    # -- panel settings ------------------------------------------------------
    Control("config_path", None, "path", "panel", "YAML configuration file to start from"),
    Control("preset", None, "str_or_none", "panel",
            "representation preset from the backend (None = the historical default) - see "
            "available_presets()"),
    Control("checkpoint", None, "path", "panel",
            "frozen SNN checkpoint (model.* must match it exactly)"),
    Control("out_dir", None, "path", "panel",
            "artifact directory (vectors / residuals / resolved configs)"),
    Control("strict", None, "bool", "panel", "reject every 'not implemented' request"),
    # -- representation ------------------------------------------------------
    _int("structured_d", "vector.structured_d", "representation",
         "deterministic structured output dimension", label="structured_d"),
    _int("learned_residual_d", "vector.learned_residual_d", "representation",
         "learned residual output dimension (0 disables the residual)",
         minimum=0, label="residual_d (learned_residual_d)"),
    Control(
        name="enabled_blocks",
        key="vector.enabled_blocks",
        kind="blocks",
        group="blocks",
        help="record blocks the structured encoder selects",
        choices=SELECTABLE_BLOCKS,
        label="enabled_blocks (structured encoder)",
    ),
    _int("temporal_resolution", "vector.temporal_resolution", "temporal",
         "coarse FIT PSTH bins of the temporal block", label="temporal_resolution"),
    # -- functional-response source ------------------------------------------
    _bool("source_functional_response", "vector.residual.source_functional_response",
          "functional_response", "residual consumes the functional-response projection",
          label="functional_response (residual source)"),
    _int("functional_source_dim", "vector.functional_source_dim", "functional_response",
         "projection dimension of the FIT per-stimulus response profile",
         maximum=MAX_FUNCTIONAL_SOURCE_DIM, label="functional_source_dim"),
    _int("functional_projection_seed", "vector.functional_projection_seed", "functional_response",
         "fixed projection seed (never shared with other seeds)", minimum=0,
         label="functional_projection_seed"),
    _choice("functional_source_normalization", "vector.functional_source_normalization",
            "functional_response", FUNCTIONAL_SOURCE_NORMALIZATIONS,
            "per-neuron transform of the response profile before the projection",
            label="functional_source_normalization"),
    # -- temporal source -----------------------------------------------------
    _bool("source_temporal", "vector.residual.source_temporal", "temporal",
          "residual consumes the coarse temporal block (independent of the encoder blocks)",
          label="temporal (residual source)"),
    # -- residual protocol ---------------------------------------------------
    _bool("residual_enabled", "vector.residual.enabled", "residual",
          "train a learned residual (requires learned_residual_d > 0)",
          label="residual_enabled"),
    _int("residual_seed", "vector.residual.seed", "residual",
         "training seed of the residual (kept individually per experiment)", minimum=0,
         label="residual_seed"),
    _int("residual_split_seed", "vector.residual.split_seed", "residual",
         "neuron-level train/validation split seed inside FIT", minimum=0,
         label="residual_split_seed"),
    _int("residual_mask_seed", "vector.residual.mask_seed", "residual",
         "mask generation seed", minimum=0, label="residual_mask_seed"),
    _int("residual_hidden_dim", "vector.residual.hidden_dim", "residual",
         "hidden width of the residual MLPs", label="residual_hidden_dim"),
    _int("residual_epochs", "vector.residual.epochs", "residual",
         "training epochs (established protocol: 200)", label="residual_epochs"),
    _int("residual_batch_size", "vector.residual.batch_size", "residual",
         "training batch size (established protocol: 64)", label="residual_batch_size"),
    _float("residual_learning_rate", "vector.residual.learning_rate", "residual",
           "Adam learning rate (established protocol: 1e-3)", minimum=0.0,
           label="residual_learning_rate"),
    _float("residual_mask_fraction", "vector.residual.mask_fraction", "residual",
           "fraction of coordinates withheld per example (established: 0.25)",
           minimum=0.0, maximum=0.999, label="residual_mask_fraction"),
    _int("residual_minimum_visible_features", "vector.residual.minimum_visible_features",
         "residual", "minimum coordinates kept visible per example",
         label="residual_minimum_visible_features"),
    _float("residual_val_fraction", "vector.residual.val_fraction", "residual",
           "validation fraction of the FIT neurons (established: 0.2)",
           minimum=0.0, maximum=0.499, label="residual_val_fraction"),
    _choice("residual_standardization", "vector.residual.standardization", "residual",
            RESIDUAL_STANDARDIZATIONS, "standardization of the residual input view",
            label="residual_standardization"),
    _choice("residual_mask_mode", "vector.residual.mask_mode", "residual", SOURCE_MASK_MODES,
            "coordinate masking (default) or whole-source block masking (diagnostic)",
            label="residual_mask_mode"),
    # -- model ---------------------------------------------------------------
    _int("n_hidden", "model.n_hidden", "model",
         "hidden layer width - must match the checkpoint exactly",
         label="n_hidden (model)"),
    _choice("model_type", "model.model_type", "model", SUPPORTED_MODEL_TYPES,
            "only a single-layer recurrent LIF is implemented", label="model_type"),
    # -- compute / memory ----------------------------------------------------
    _choice("device", "memory.device", "memory", SUPPORTED_DEVICES,
            "device for activity collection and residual training", label="device"),
    _choice("storage", "memory.storage", "memory", SUPPORTED_STORAGE,
            "record-bank storage (only 'cpu' is implemented)", label="storage"),
    _int("train_batch_size", "memory.train_batch_size", "memory", "training batch size",
         label="train_batch_size"),
    _int("eval_batch_size", "memory.eval_batch_size", "memory", "evaluation batch size",
         label="eval_batch_size"),
    _int("record_batch_size", "memory.record_batch_size", "memory",
         "streaming batch of the recorded FIT activity pass", label="record_batch_size"),
    _int("activity_chunk_size", "memory.activity_chunk_size", "memory",
         "activity accumulation chunk", label="activity_chunk_size"),
    _int("representation_chunk_size", "memory.representation_chunk_size", "memory",
         "representation construction chunk", label="representation_chunk_size"),
    _bool("mixed_precision", "memory.mixed_precision", "memory",
          "mixed-precision training (not implemented; rejected)", label="mixed_precision"),
    _int("max_input_tokens", "memory.max_input_tokens", "memory",
         "dense-input token budget (0 disables)", minimum=0, label="max_input_tokens"),
    _choice("vector_dtype", "precision.vector_dtype", "precision", SUPPORTED_DTYPES,
            "storage dtype of the neuron vectors", label="vector_dtype"),
    _choice("activity_dtype", "precision.activity_dtype", "precision", SUPPORTED_DTYPES,
            "storage dtype of the activity statistics", label="activity_dtype"),
    _choice("model_dtype", "precision.model_dtype", "precision", SUPPORTED_DTYPES,
            "SNN compute dtype (only float32 is implemented)", label="model_dtype"),
    # -- seeds / evaluation --------------------------------------------------
    _int("seed", "seed", "evaluation",
         "run seed: split fallback + evaluation permutation/CV seed", minimum=0, label="seed"),
    _choice("evaluation_target", None, "evaluation", tuple(sorted(EVALUATION_TARGETS)),
            "'all' (the four response variants) or one target name", label="evaluation_target"),
    _choice("evaluation_mode", None, "evaluation", SUPPORTED_EVALUATION_MODES,
            "only the response-target mode is implemented", label="evaluation_mode"),
    _int("n_perm", None, "evaluation", "Mantel permutations", minimum=0, label="n_perm"),
    _int("bootstrap", None, "evaluation", "bootstrap resamples (0 disables the CI)",
         minimum=0, label="bootstrap"),
    _int("n_splits", None, "evaluation", "cross-validation folds over neurons",
         label="n_splits"),
)

CONTROL_GROUPS: tuple[str, ...] = (
    "panel", "representation", "blocks", "functional_response", "temporal", "residual",
    "model", "memory", "precision", "evaluation",
)

#: State keys that are *derived* or display-only (never editable, never an override).
DERIVED_KEYS: tuple[str, ...] = ("d", "n_layers", "network_context")


def panel_controls() -> tuple[Control, ...]:
    """All panel controls (metadata only; defaults come from the backend)."""
    return PANEL_CONTROLS


def control_index() -> dict[str, Control]:
    return {c.name: c for c in PANEL_CONTROLS}


def controls_by_group(group: str) -> tuple[Control, ...]:
    return tuple(c for c in PANEL_CONTROLS if c.group == group)


def config_backed_controls() -> dict[str, str]:
    """``{state name: dotted configuration key}`` for the controls that map to config keys."""
    return {c.name: c.key for c in PANEL_CONTROLS if c.key is not None}


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------
def _state_from_v2(v2: V2Config, cfg: Config) -> dict[str, Any]:
    """The panel state that reproduces a resolved configuration exactly."""
    residual = v2.vector.residual
    return {
        # representation
        "structured_d": int(v2.vector.structured_d),
        "learned_residual_d": int(v2.vector.learned_residual_d),
        "enabled_blocks": list(v2.vector.enabled_blocks),
        "temporal_resolution": int(v2.vector.temporal_resolution),
        # functional-response source
        "source_functional_response": bool(residual.source_functional_response),
        "functional_source_dim": int(v2.vector.functional_source_dim),
        "functional_projection_seed": int(v2.vector.functional_projection_seed),
        "functional_source_normalization": v2.vector.functional_source_normalization,
        # temporal source
        "source_temporal": bool(residual.source_temporal),
        # residual protocol
        "residual_enabled": bool(residual.enabled),
        "residual_seed": int(residual.seed),
        "residual_split_seed": int(residual.split_seed),
        "residual_mask_seed": int(residual.mask_seed),
        "residual_hidden_dim": int(residual.hidden_dim),
        "residual_epochs": int(residual.epochs),
        "residual_batch_size": int(residual.batch_size),
        "residual_learning_rate": float(residual.learning_rate),
        "residual_mask_fraction": float(residual.mask_fraction),
        "residual_minimum_visible_features": int(residual.minimum_visible_features),
        "residual_val_fraction": float(residual.val_fraction),
        "residual_standardization": residual.standardization,
        "residual_mask_mode": residual.mask_mode,
        # model (must match the checkpoint; validated by the backend)
        "n_hidden": int(v2.network.n_hidden),
        "model_type": v2.network.model_type,
        # compute / memory / precision
        "device": v2.memory.device,
        "storage": v2.memory.storage,
        "train_batch_size": int(v2.memory.train_batch_size),
        "eval_batch_size": int(v2.memory.eval_batch_size),
        "record_batch_size": int(v2.memory.record_batch_size),
        "activity_chunk_size": int(v2.memory.activity_chunk_size),
        "representation_chunk_size": int(v2.memory.representation_chunk_size),
        "mixed_precision": bool(v2.memory.mixed_precision),
        "max_input_tokens": int(v2.memory.max_input_tokens),
        "vector_dtype": v2.precision.vector_dtype,
        "activity_dtype": v2.precision.activity_dtype,
        "model_dtype": v2.precision.model_dtype,
        # seeds
        "seed": int(cfg.get_path("seed", 0) or 0),
        # display-only
        "d": int(v2.vector.d),
        "n_layers": int(v2.network.n_layers),
        "network_context": "not implemented",
    }


def default_state(
    *,
    preset: str | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    checkpoint: str | Path | None = None,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    evaluation_target: str = DEFAULT_EVALUATION_TARGET,
    evaluation_mode: str = SUPPORTED_EVALUATION_MODES[0],
    n_perm: int = 2000,
    bootstrap: int = 500,
    n_splits: int = 5,
    strict: bool = False,
) -> dict[str, Any]:
    """The baseline state: an empty config's defaults, or exactly the requested preset.

    Every value comes from the backend (a resolved :class:`V2Config`), so the notebook cannot
    drift from the documented defaults.
    """
    resolved = resolve_config(config_path=config_path, preset=preset, checkpoint=checkpoint,
                              out_dir=out_dir)
    state = _state_from_v2(resolved.v2, resolved.cfg)
    state.update(
        config_path=str(config_path),
        preset=preset,
        checkpoint=str(resolved.checkpoint_path) if resolved.checkpoint_path else str(checkpoint or DEFAULT_CHECKPOINT),
        out_dir=str(out_dir),
        strict=bool(strict),
        evaluation_target=evaluation_target,
        evaluation_mode=evaluation_mode,
        n_perm=int(n_perm),
        bootstrap=int(bootstrap),
        n_splits=int(n_splits),
    )
    return state


def _shape_check(name: str, value: Any) -> Any:
    """Immediate shape/type/choice validation of one control value (no backend call)."""
    index = control_index()
    if name in DERIVED_KEYS:
        raise PanelError(f"{name!r} is derived/display-only and cannot be set (available: see panel_controls())")
    control = index.get(name)
    if control is None:
        raise PanelError(
            f"unknown control {name!r}; available: {sorted(index)}"
        )
    if name == "preset" and value is not None and str(value) not in available_presets():
        raise PanelError(f"unknown preset {value!r}; available: {list(available_presets())}")
    if control.kind == "bool":
        if not isinstance(value, (bool, np.bool_)):
            raise PanelError(f"{name} must be a bool, got {value!r}")
        return bool(value)
    if control.kind == "str_or_none":
        if value is None:
            return None
        if not isinstance(value, str):
            raise PanelError(f"{name} must be a string or None, got {value!r}")
        return str(value)
    if control.kind == "path":
        if not isinstance(value, (str, Path)):
            raise PanelError(f"{name} must be a path (string), got {value!r}")
        return str(value)
    if control.kind in ("int", "float"):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise PanelError(f"{name} must be a number, got {value!r}")
        number = float(value) if control.kind == "float" else int(value)
        if control.minimum is not None and number < control.minimum:
            raise PanelError(f"{name} must be >= {control.minimum:g}, got {number:g}")
        if control.maximum is not None and number > control.maximum:
            raise PanelError(f"{name} must be <= {control.maximum:g}, got {number:g}")
        return number
    if control.kind == "blocks":
        if isinstance(value, str):
            value = [part.strip() for part in value.split(",") if part.strip()]
        blocks = [str(b) for b in value]
        if not blocks:
            raise PanelError("enabled_blocks must select at least one block")
        unknown = [b for b in blocks if b not in IMPLEMENTED_RECORD_BLOCKS]
        if unknown:
            raise PanelError(
                f"unknown or not implemented block(s) {unknown}; selectable: {list(SELECTABLE_BLOCKS)}"
            )
        return blocks
    if control.kind == "choice" and control.choices:
        if value not in control.choices:
            raise PanelError(f"{name} must be one of {list(control.choices)}, got {value!r}")
        return value
    if name == "preset" and value is not None and str(value) not in available_presets():
        raise PanelError(
            f"unknown preset {value!r}; available: {list(available_presets())}"
        )
    if not isinstance(value, str):
        raise PanelError(f"{name} must be a string, got {value!r}")
    return str(value)
def set_controls(state: Mapping[str, Any], **values: Any) -> dict[str, Any]:
    """Update a panel state in place after immediate validation; returns the state.

    Unknown names, wrong types, values outside a documented range and not-implemented blocks are
    rejected here (before any configuration object is built), so the notebook reports the problem
    in the cell that made it.
    """
    if not isinstance(state, dict):
        raise PanelError("set_controls needs a mutable state dict (see default_state())")
    # validate everything first: an invalid request never leaves a half-updated state behind
    validated = {name: _shape_check(name, value) for name, value in values.items()}
    state.update(validated)
    if "structured_d" in validated or "learned_residual_d" in validated:
        state["d"] = int(state["structured_d"]) + int(state["learned_residual_d"])
    return state


def dimension_check(state: Mapping[str, Any]) -> dict[str, Any]:
    """The dimension invariant and the residual/enabled consistency, checked immediately."""
    structured = int(state["structured_d"])
    residual = int(state["learned_residual_d"])
    total = structured + residual
    problems: list[str] = []
    if residual > 0 and not bool(state["residual_enabled"]):
        problems.append(
            f"learned_residual_d={residual} > 0 requires residual_enabled=True "
            "(the residual would have no consumer)"
        )
    if bool(state["residual_enabled"]) and residual == 0:
        problems.append("residual_enabled=True requires learned_residual_d > 0")
    if total < 1:
        problems.append("the total dimension must be >= 1")
    if bool(state["source_functional_response"]) and not (bool(state["residual_enabled"]) and residual > 0):
        problems.append(
            "source_functional_response=True requires the residual to be enabled with "
            "learned_residual_d > 0"
        )
    if bool(state["source_temporal"]) and not (bool(state["residual_enabled"]) and residual > 0):
        problems.append(
            "source_temporal=True requires the residual to be enabled with learned_residual_d > 0"
        )
    return {
        "ok": not problems,
        "d": total,
        "structured_d": structured,
        "residual_d": residual,
        "expression": f"{total} = {structured} structured + {residual} residual",
        "problems": problems,
    }


def state_to_overrides(
    state: Mapping[str, Any],
    *,
    baseline: Mapping[str, Any] | None = None,
) -> list[str]:
    """Dotted ``--override`` list for the values that differ from the baseline (in control order)."""
    if baseline is None:
        baseline = default_state(
            preset=state.get("preset"),
            config_path=state.get("config_path", DEFAULT_CONFIG_PATH),
            checkpoint=state.get("checkpoint"),
            out_dir=state.get("out_dir", DEFAULT_OUT_DIR),
        )
    overrides: list[str] = []
    for control in PANEL_CONTROLS:
        if control.key is None:
            continue
        name = control.name
        if name not in state:
            continue
        current, base = state[name], baseline.get(name)
        if control.kind == "float":
            if base is not None and math.isclose(float(current), float(base), rel_tol=0.0, abs_tol=1e-15):
                continue
        elif control.kind == "blocks":
            if [str(b) for b in current] == [str(b) for b in (base or [])]:
                continue
        elif current == base:
            continue
        overrides.append(f"{control.key}={_override_text(current)}")
    return overrides


def _override_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_override_text(v) for v in value) + "]"
    return str(value)


def resolve_state(state: Mapping[str, Any], *, strict: bool | None = None) -> ResolvedConfig:
    """Resolve + validate a panel state through the stable pipeline (the only validation path)."""
    problems = dimension_check(state)
    if not problems["ok"]:
        raise PanelError("; ".join(problems["problems"]))
    try:
        return resolve_config(
            config_path=state.get("config_path", DEFAULT_CONFIG_PATH),
            preset=state.get("preset"),
            overrides=state_to_overrides(state),
            checkpoint=state.get("checkpoint"),
            out_dir=state.get("out_dir", DEFAULT_OUT_DIR),
            evaluation_mode=state.get("evaluation_mode", SUPPORTED_EVALUATION_MODES[0]),
            evaluation_target=state.get("evaluation_target", DEFAULT_EVALUATION_TARGET),
            n_perm=int(state.get("n_perm", 2000)),
            bootstrap=int(state.get("bootstrap", 500)),
            n_splits=int(state.get("n_splits", 5)),
            strict=bool(state.get("strict", False) if strict is None else strict),
        )
    except (PipelineError, V2ConfigError) as exc:
        raise PanelError(str(exc)) from exc


def verify_checkpoint_architecture(resolved: ResolvedConfig) -> None:
    """Fail fast when the selected checkpoint does not match the resolved model block.

    Uses the backend's own mismatch helper (no duplicated rule). The CLI dry run deliberately
    stays model-free; the interactive panel verifies here so a wrong checkpoint is reported in
    the validation cell instead of only when the build starts.
    """
    if not resolved.checkpoint_exists or resolved.checkpoint_path is None:
        return
    try:
        model, _ = RecurrentLIFSNN.load(str(resolved.checkpoint_path), map_location="cpu")
    except Exception as exc:  # noqa: BLE001 - surfaced as a validation error
        raise PanelError(f"could not load checkpoint {resolved.checkpoint_path}: {exc}") from exc
    mismatches = architecture_mismatches(resolved.cfg.get_path("model", {}) or {}, model)
    if mismatches:
        detail = ", ".join(
            f"model.{k}: config={a!r} vs checkpoint={b!r}" for k, (a, b) in mismatches.items()
        )
        raise PanelError(
            "config/checkpoint architecture mismatch: " + detail
            + " (choose a checkpoint that matches the model controls, or fix them)"
        )


def validate_state(state: Mapping[str, Any], *, strict: bool | None = None) -> ResolvedConfig:
    """Resolve + validate a panel state, then verify the checkpoint architecture (fail fast)."""
    resolved = resolve_state(state, strict=strict)
    verify_checkpoint_architecture(resolved)
    return resolved


def state_run_id(state: Mapping[str, Any]) -> str:
    """The run id the current state resolves to (validates first)."""
    return resolve_state(state).run_id


def state_export(state: Mapping[str, Any]) -> dict[str, Any]:
    """Serialisable snapshot of a panel state (for metadata/reproducibility)."""
    return {
        "schema": PANEL_SCHEMA,
        "preset": state.get("preset"),
        "config_path": state.get("config_path"),
        "checkpoint": state.get("checkpoint"),
        "out_dir": state.get("out_dir"),
        "controls": {c.name: state.get(c.name) for c in PANEL_CONTROLS if c.name in state},
        "derived": {k: state.get(k) for k in DERIVED_KEYS if k in state},
    }


def protocol_drift(state: Mapping[str, Any]) -> dict[str, tuple[Any, Any]]:
    """Residual-protocol values that differ from the established (dataclass) defaults."""
    established = ResidualTrainingConfig()
    pairs = {
        "residual_seed": "seed",
        "residual_split_seed": "split_seed",
        "residual_mask_seed": "mask_seed",
        "residual_hidden_dim": "hidden_dim",
        "residual_epochs": "epochs",
        "residual_batch_size": "batch_size",
        "residual_learning_rate": "lr",
        "residual_mask_fraction": "mask_fraction",
        "residual_minimum_visible_features": "minimum_visible_features",
        "residual_val_fraction": "val_fraction",
        "residual_standardization": "normalization",
        "residual_mask_mode": "mask_mode",
    }
    drift: dict[str, tuple[Any, Any]] = {}
    for state_name, field_name in pairs.items():
        if state_name not in state:
            continue
        current = state[state_name]
        default = getattr(established, field_name)
        if isinstance(default, float):
            if math.isclose(float(current), float(default), rel_tol=0.0, abs_tol=1e-15):
                continue
            drift[state_name] = (current, default)
        elif current != default:
            drift[state_name] = (current, default)
    return drift


# --------------------------------------------------------------------------
# Preview / reporting
# --------------------------------------------------------------------------
def safety_lines() -> list[str]:
    """The FIT / PROBE / TEST contract, always displayed by the notebook."""
    return [
        "FIT:   representation construction + residual training",
        "PROBE: evaluation only",
        "TEST:  never accessed",
    ]


def resolved_preview(resolved: ResolvedConfig) -> dict[str, list[tuple[str, Any]]]:
    """Grouped, display-ready view of a resolved configuration (the notebook's dry run)."""
    v = resolved.v2
    r = v.vector.residual
    drift = protocol_drift(
        {
            "residual_seed": r.seed, "residual_split_seed": r.split_seed,
            "residual_mask_seed": r.mask_seed, "residual_hidden_dim": r.hidden_dim,
            "residual_epochs": r.epochs, "residual_batch_size": r.batch_size,
            "residual_learning_rate": r.learning_rate, "residual_mask_fraction": r.mask_fraction,
            "residual_minimum_visible_features": r.minimum_visible_features,
            "residual_val_fraction": r.val_fraction, "residual_standardization": r.standardization,
            "residual_mask_mode": r.mask_mode,
        }
    )
    return {
        "CHECKPOINT": [
            ("path", str(resolved.checkpoint_path)),
            ("sha256_16", resolved.checkpoint_sha256_16),
            ("exists", resolved.checkpoint_exists),
            ("run_id", resolved.run_id),
        ],
        "MODEL": [
            ("model_type", v.network.model_type),
            ("n_hidden", v.network.n_hidden),
            ("n_layers", f"{v.network.n_layers} (multi-layer not implemented)"),
            ("n_input / n_bins", f"{v.snn.n_input} / {v.snn.n_bins} ({v.snn.bin_ms} ms bins)"),
        ],
        "FIT / DEV / PROBE / TEST POLICY": [
            ("FIT", DATA_POLICY["fit"]),
            ("DEV", DATA_POLICY["dev"]),
            ("PROBE", DATA_POLICY["probe"]),
            ("TEST", DATA_POLICY["test"]),
        ],
        "TOTAL DIMENSION": [
            ("structured", resolved.structured_d),
            ("residual", resolved.residual_d),
            ("total", f"{resolved.d}  ({resolved.structured_d} + {resolved.residual_d})"),
        ],
        "STRUCTURED BLOCKS": [
            ("enabled (encoder)", ", ".join(v.vector.enabled_blocks)),
            ("requested blocks present", ", ".join(resolved.present_block_names())),
            ("network_context", "not implemented (not selectable)"),
        ],
        "FUNCTIONAL RESPONSE": [
            ("residual source enabled", r.source_functional_response),
            ("functional_source_dim", v.vector.functional_source_dim),
            ("functional_projection_seed", v.vector.functional_projection_seed),
            ("normalization", v.vector.functional_source_normalization),
        ],
        "TEMPORAL": [
            ("encoder block selected", "temporal" in v.vector.enabled_blocks),
            ("residual source enabled", r.source_temporal),
            ("temporal_resolution", v.vector.temporal_resolution),
            ("bank block requested", resolved.temporal_block_requested),
        ],
        "RESIDUAL": [
            ("enabled", r.enabled),
            ("residual_d", resolved.residual_d),
            ("epochs / batch / lr", f"{r.epochs} / {r.batch_size} / {r.learning_rate}"),
            ("hidden_dim", r.hidden_dim),
            ("mask (mode / fraction / seed)", f"{r.mask_mode} / {r.mask_fraction} / {r.mask_seed}"),
            (
                "protocol drift vs established",
                "none" if not drift else ", ".join(
                    f"{k}: {v_new!r} != {v_def!r}" for k, (v_new, v_def) in drift.items()
                ),
            ),
        ],
        "MEMORY / PRECISION": [
            ("device / storage", f"{v.memory.device} / {v.memory.storage}"),
            (
                "batch (train / eval / record)",
                f"{v.memory.train_batch_size} / {v.memory.eval_batch_size} / "
                f"{v.memory.record_batch_size}",
            ),
            (
                "chunk (activity / representation)",
                f"{v.memory.activity_chunk_size} / {v.memory.representation_chunk_size}",
            ),
            (
                "dtype (vector / activity / model)",
                f"{v.precision.vector_dtype} / {v.precision.activity_dtype} / "
                f"{v.precision.model_dtype}",
            ),
            ("mixed_precision / max_input_tokens",
             f"{v.memory.mixed_precision} / {v.memory.max_input_tokens}"),
        ],
        "SEEDS": [
            ("run seed", resolved.v2.experiment.seed if resolved.v2.experiment.seed is not None else 0),
            ("residual seed / split / mask", f"{r.seed} / {r.split_seed} / {r.mask_seed}"),
            ("functional projection seed", v.vector.functional_projection_seed),
        ],
        "EVALUATION": [
            ("mode / target", f"{resolved.evaluation_mode} / {resolved.evaluation_target}"),
            ("n_perm / bootstrap / n_splits",
             f"{resolved.n_perm} / {resolved.bootstrap} / {resolved.n_splits}"),
        ],
    }


def preview_lines(resolved: ResolvedConfig) -> list[str]:
    """The grouped preview as printable lines."""
    lines: list[str] = []
    for group, rows in resolved_preview(resolved).items():
        lines.append(group)
        for name, value in rows:
            lines.append(f"  {name:34s} {value}")
        lines.append("")
    lines.extend(safety_lines())
    if resolved.warnings:
        lines.append("")
        lines.append("WARNINGS")
        lines.extend(f"  - {w}" for w in resolved.warnings)
    return lines


def reproducibility_lines(resolved: ResolvedConfig, *, extra: Mapping[str, Any] | None = None) -> list[str]:
    """A copy/paste-friendly block: what exactly would run (CLI summary companion)."""
    import sys

    lines = [
        "REPRODUCIBILITY",
        f"  run_id                : {resolved.run_id}",
        f"  config                : {resolved.config_path}",
        f"  preset                : {resolved.preset or '(none)'}",
        f"  cli-equivalent        : uv run python scripts/v2_control_panel.py "
        f"--config {resolved.config_path}" + (f" --preset {resolved.preset}" if resolved.preset else "")
        + "".join(f" --override {o}" for o in resolved.cli_overrides),
        f"  checkpoint            : {resolved.checkpoint_path} (sha256_16={resolved.checkpoint_sha256_16})",
        f"  dimensions            : d={resolved.d} = {resolved.structured_d} structured + "
        f"{resolved.residual_d} residual",
        f"  blocks                : {', '.join(resolved.v2.vector.enabled_blocks)}",
        f"  residual seeds        : seed={resolved.v2.vector.residual.seed}, "
        f"split_seed={resolved.v2.vector.residual.split_seed}, "
        f"mask_seed={resolved.v2.vector.residual.mask_seed}",
        f"  schemas               : pipeline={resolved.schema}, "
        f"components={resolved.to_dict()['component_schemas']}",
        f"  python                : {sys.version.split()[0]} ({sys.executable})",
        f"  repo root             : {PROJECT_ROOT}",
        "  json (paste-ready)    :",
        f"    {resolved.to_dict()['run_identity']}",
    ]
    if extra:
        for name, value in extra.items():
            lines.append(f"  {name:22s}: {value}")
    return lines


def cache_inventory(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    checkpoint: str | Path | None = None,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """What already exists: presets, checkpoints and this run's artifacts (no recomputation)."""
    directory = Path(out_dir)
    if not directory.is_absolute():
        directory = PROJECT_ROOT / directory
    vectors = sorted((directory / "vectors").glob("*.npz")) if (directory / "vectors").exists() else []
    residuals = sorted((directory / "residuals").glob("*.pt")) if (directory / "residuals").exists() else []
    inventory: dict[str, Any] = {
        "presets": list(available_presets()),
        "preset_decompositions": dict(PRESET_DECOMPOSITIONS),
        "preset_descriptions": dict(PRESET_DESCRIPTIONS),
        "checkpoints": checkpoint_candidates(),
        "vectors": [{"name": p.name, "size_bytes": p.stat().st_size} for p in vectors],
        "n_residual_artifacts": len(residuals),
        "out_dir": str(directory),
    }
    if state is not None:
        try:
            resolved = resolve_state(state)
        except PanelError as exc:
            inventory["state_error"] = str(exc)
        else:
            inventory["run_id"] = resolved.run_id
            inventory["vector_artifact"] = str(resolved.vector_artifact_path)
            inventory["vector_artifact_exists"] = resolved.vector_artifact_path.exists()
            inventory["resolved_config_exists"] = resolved.resolved_config_path.exists()
    if checkpoint is not None:
        inventory["checkpoint"] = str(checkpoint)
    return inventory


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------
def checkpoint_candidates(directory: str | Path = "checkpoints") -> list[dict[str, Any]]:
    """Available checkpoints (name, size, mtime) - the notebook's selector source."""
    path = Path(directory)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return []
    out = []
    for item in sorted(path.glob("*.pt")):
        stat = item.stat()
        out.append({
            "name": item.name,
            "path": str(item),
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "mtime": int(stat.st_mtime),
            "is_default": item.name == Path(DEFAULT_CHECKPOINT).name,
        })
    return out


def checkpoint_details(path: str | Path) -> dict[str, Any]:
    """Architecture of a checkpoint plus its identifier (loads the model, never modifies it)."""
    checkpoint = Path(path)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    if not checkpoint.exists():
        raise PanelError(f"checkpoint not found: {checkpoint}")
    model, extra = RecurrentLIFSNN.load(str(checkpoint), map_location="cpu")
    import hashlib

    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    cfg = model.cfg
    return {
        "path": str(checkpoint),
        "sha256_16": digest[:16],
        "architecture": {
            "n_input": int(cfg.n_input),
            "n_hidden": int(cfg.n_hidden),
            "n_output": int(cfg.n_output),
            "n_bins": int(cfg.n_bins),
            "bin_ms": float(cfg.bin_ms),
            "readout_mode": str(cfg.readout_mode),
            "neuron_param_mode": str(cfg.neuron_param_mode),
            "tau_mem_ms": float(cfg.tau_mem_ms),
            "tau_syn_ms": float(cfg.tau_syn_ms),
            "threshold": float(cfg.threshold),
        },
        "extra": {k: v for k, v in dict(extra or {}).items()
                  if isinstance(v, (str, int, float, bool, type(None)))},
    }


def checkpoint_mismatches(path: str | Path, config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Architecture mismatches between a checkpoint and a configuration (empty = compatible)."""
    checkpoint = Path(path)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    model, _ = RecurrentLIFSNN.load(str(checkpoint), map_location="cpu")
    cfg = load_config(config_path)
    mismatches = architecture_mismatches(cfg.get_path("model", {}) or {}, model)
    return {"compatible": not mismatches, "mismatches": {k: list(v) for k, v in mismatches.items()}}


# --------------------------------------------------------------------------
# Operations (the notebook's action cells)
# --------------------------------------------------------------------------
def dry_run_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve + validate and return the plan (loads no data, trains nothing)."""
    return dry_run_report(resolve_state(state))


def build_from_state(
    state: Mapping[str, Any],
    *,
    use_cache: bool = True,
    verbose: bool = False,
) -> BuildResult:
    """Build (or reuse) the frozen representation for a state; FIT only, never evaluates."""
    resolved = resolve_state(state)
    run = V2Run.from_resolved(resolved)
    return run.build_representation(use_cache=use_cache, verbose=verbose)


def load_from_state(state: Mapping[str, Any]) -> NeuronVectorArtifact:
    """Load the frozen artifact of a state (verifying run id, dimensions, names and hash)."""
    resolved = resolve_state(state)
    try:
        return load_vector_artifact(resolved)
    except (NeuronVectorError, FileNotFoundError) as exc:
        raise PanelError(
            f"{exc}\nRun the build cell first (the artifact is {resolved.vector_artifact_path})."
        ) from exc


def inspect_artifact(artifact: NeuronVectorArtifact, *, max_names: int = 8) -> dict[str, Any]:
    """Artifact sanity check: shape, coordinate groups, finiteness and per-group statistics."""
    X = np.asarray(artifact.X, dtype=np.float64)
    names = tuple(artifact.feature_names)
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        group = "residual" if str(name).startswith("residual[") else str(name).split(".")[0]
        groups.setdefault(group, []).append(index)
    per_group = {}
    for group, columns in groups.items():
        block = X[:, columns]
        per_group[group] = {
            "n_coordinates": len(columns),
            "mean": float(block.mean()),
            "std": float(block.std()),
            "absmax": float(np.abs(block).max()),
            "any_nan": bool(np.isnan(block).any()),
            "any_inf": bool(np.isinf(block).any()),
        }
    column_mean = X.mean(axis=0)
    column_std = X.std(axis=0)
    return {
        "run_id": artifact.run_id,
        "n_neurons": artifact.n_neurons,
        "d": artifact.d,
        "structured_d": artifact.structured_d,
        "residual_d": artifact.residual_d,
        "n_coordinates": len(names),
        "feature_names": list(names),
        "feature_names_head": list(names[:max_names]),
        "feature_names_tail": list(names[-max_names:]),
        "coordinate_groups": {g: len(c) for g, c in groups.items()},
        "finite": bool(np.isfinite(X).all()),
        "n_nan": int(np.isnan(X).sum()),
        "n_inf": int(np.isinf(X).sum()),
        "global_mean": float(X.mean()),
        "global_std": float(X.std()),
        "column_mean": {"min": float(column_mean.min()), "median": float(np.median(column_mean)),
                        "max": float(column_mean.max()), "mean": float(column_mean.mean())},
        "column_std": {"min": float(column_std.min()), "median": float(np.median(column_std)),
                       "max": float(column_std.max()), "mean": float(column_std.mean())},
        "per_group": per_group,
        "matrix_sha256": artifact.summary()["matrix_sha256"],
        "provenance": artifact.provenance,
        "residual_source": _residual_source_summary(artifact.provenance),
    }


def _residual_source_summary(provenance: Mapping[str, Any]) -> dict[str, Any] | None:
    composition = dict(provenance.get("composition", {}))
    residual = dict(composition.get("residual", {}))
    if not residual.get("used"):
        return None
    source = dict(provenance.get("residual_protocol") or {})
    residual_provenance = dict(residual.get("residual_provenance") or {})
    return {
        "residual_dim": residual.get("residual_dim"),
        "input_dim": residual.get("input_dim"),
        "source_schema_hash": residual.get("source_schema_hash"),
        "source_groups": (
            {group: len(indices) for group, indices in (residual_provenance.get("source_groups") or {}).items()}
            or None
        ),
        "normalization": residual.get("normalization"),
        "mask": residual.get("mask"),
        "best_epoch": residual.get("best_epoch"),
        "best_val_loss": residual.get("best_val_loss"),
        "protocol": source,
        "cache_path": provenance.get("residual_cache"),
    }


def inspect_from_state(state: Mapping[str, Any], *, max_names: int = 8) -> dict[str, Any]:
    """Inspect the frozen artifact of a state (loads only; never evaluates)."""
    return inspect_artifact(load_from_state(state), max_names=max_names)


def evaluate_from_state(
    state: Mapping[str, Any],
    *,
    artifact: NeuronVectorArtifact | None = None,
    use_cache: bool = True,
    verbose: bool = False,
) -> EvaluationResult:
    """Evaluate the frozen representation on PROBE (never rebuilds it, never touches TEST)."""
    resolved = resolve_state(state)
    return evaluate_representation(
        resolved,
        artifact=artifact if artifact is not None else load_from_state(state),
        use_cache=use_cache,
        verbose=verbose,
    )


def evaluation_notice() -> list[str]:
    """The notice the notebook displays before the evaluation cell."""
    return [
        "Evaluation uses PROBE.",
        "Representation construction remains FIT-only.",
        "TEST is untouched.",
    ]


def evaluation_targets() -> dict[str, str]:
    """Target names the evaluation stack supports, with a short description."""
    return {
        name: (
            "'all' = the four response variants (raw, neuron_centered, neuron_zscored, mean_rate)"
            if name == "all" else description
        )
        for name, description in EVALUATION_TARGETS.items()
    }


# --------------------------------------------------------------------------
# Optional widgets (never required)
# --------------------------------------------------------------------------
def widget_support() -> dict[str, Any]:
    """Whether an interactive widget library is importable (never installed by the panel)."""
    try:
        import ipywidgets  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on the environment
        return {
            "available": False,
            "reason": f"ipywidgets is not importable ({type(exc).__name__}: {exc})",
            "fallback": "use the clearly marked editable variables in each cell",
        }
    return {"available": True, "reason": "", "fallback": ""}


def make_widgets(state: Mapping[str, Any], *, groups: Sequence[str] = ("representation", "residual")):
    """Optional ipywidgets form bound to a state dict; ``None`` when widgets are unavailable.

    The notebook works without this: each control is also an ordinary editable variable.
    """
    support = widget_support()
    if not support["available"]:
        return None
    import ipywidgets as widgets  # pragma: no cover - not installed in this environment

    rows = []
    for group in groups:
        rows.append(widgets.HTML(f"<b>{group}</b>"))
        for control in controls_by_group(group):
            if control.name not in state:
                continue
            value = state[control.name]
            description = control.display
            if control.kind == "bool":
                widget = widgets.Checkbox(value=bool(value), description=description)
            elif control.kind == "choice":
                widget = widgets.Dropdown(options=list(control.choices), value=value, description=description)
            elif control.kind == "int":
                widget = widgets.IntText(value=int(value), description=description)
            elif control.kind == "float":
                widget = widgets.FloatText(value=float(value), description=description)
            elif control.kind == "blocks":
                widget = widgets.Text(value=",".join(value), description=description)
            else:
                widget = widgets.Text(value=str(value), description=description)

            def _sync(change, name=control.name):  # pragma: no cover - widget callback
                try:
                    set_controls(state, **{name: change["new"]})
                except PanelError as exc:
                    print(f"[invalid] {name}: {exc}")

            widget.observe(_sync, names="value")
            rows.append(widget)
    return widgets.VBox(rows)


__all__ = [
    "PANEL_SCHEMA",
    "SELECTABLE_BLOCKS",
    "DISPLAY_ONLY",
    "CONTROL_GROUPS",
    "DERIVED_KEYS",
    "PanelError",
    "Control",
    "PANEL_CONTROLS",
    "panel_controls",
    "control_index",
    "controls_by_group",
    "config_backed_controls",
    "default_state",
    "set_controls",
    "dimension_check",
    "state_to_overrides",
    "resolve_state",
    "validate_state",
    "verify_checkpoint_architecture",
    "state_run_id",
    "state_export",
    "protocol_drift",
    "safety_lines",
    "resolved_preview",
    "preview_lines",
    "reproducibility_lines",
    "cache_inventory",
    "checkpoint_candidates",
    "checkpoint_details",
    "checkpoint_mismatches",
    "dry_run_state",
    "build_from_state",
    "load_from_state",
    "inspect_artifact",
    "inspect_from_state",
    "evaluate_from_state",
    "evaluation_notice",
    "evaluation_targets",
    "widget_support",
    "make_widgets",
    "parent_summary",
    "available_presets",
]
