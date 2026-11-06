"""Tests for ``--help-overrides`` / :func:`v3.config.override_paths`."""

import dataclasses

from v3.config import (
    V3Config,
    apply_override_mapping,
    format_override_help,
    override_paths,
)


def test_every_listed_key_is_accepted_by_the_override_grammar():
    cfg = V3Config()
    paths = override_paths(cfg)
    assert len(paths) > 80
    for path, _kind, value in paths:
        # re-applying the current default must always be a legal override
        out = apply_override_mapping(cfg, {path: value})
        assert isinstance(out, V3Config)


def test_listed_keys_are_scalars_not_blocks():
    cfg = V3Config()
    for path, kind, value in override_paths(cfg):
        assert not dataclasses.is_dataclass(value), path
        # a nested path must point at a field that really exists
        obj = cfg
        for part in path.split("."):
            obj = getattr(obj, part)
        assert obj is value


def test_help_covers_the_nested_blocks_and_known_keys():
    text = format_override_help()
    for needle in (
        "top-level:",
        "epochs",
        "timing.time_bin_ms",
        "augmentation.event_dropout.probability",
        "input_current.enabled",
        "learning_matrix.readout",
        "emission.mode",
        "keys in total.",
    ):
        assert needle in text, needle


def test_help_reflects_a_custom_config():
    cfg = V3Config().with_overrides(epochs=7, tag="custom")
    text = format_override_help(cfg)
    line = next(ln for ln in text.splitlines() if ln.strip().startswith("epochs "))
    assert "7" in line
