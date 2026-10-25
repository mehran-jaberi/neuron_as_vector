"""Tests for configuration parsing and dataclass coercion.

Regression tests for the bug where ``--override train.l2_spikes=1e-3`` was parsed
by YAML 1.1 as the *string* ``'1e-3'`` (because a float exponent needs a decimal
point in YAML 1.1), which then crashed training with
``'>' not supported between instances of 'str' and 'int'``.
"""

from __future__ import annotations

import textwrap

from src.model import SNNConfig
from src.training import TrainConfig
from src.utils import load_config, parse_scalar


def test_parse_scalar_handles_scientific_notation():
    assert parse_scalar("1e-3") == 1e-3
    assert isinstance(parse_scalar("1e-3"), float)
    assert parse_scalar("1E-3") == 1e-3
    assert parse_scalar("-2.5e+2") == -250.0
    assert parse_scalar("0.001") == 0.001


def test_parse_scalar_preserves_types_and_strings():
    assert parse_scalar("512") == 512 and isinstance(parse_scalar("512"), int)
    assert parse_scalar("true") is True
    assert parse_scalar("false") is False
    assert parse_scalar("null") is None
    # non-numeric strings must survive untouched
    assert parse_scalar("subtract") == "subtract"
    assert parse_scalar("cosine") == "cosine"


def test_train_config_coerces_numeric_strings():
    tcfg = TrainConfig.from_mapping({"l2_spikes": "1e-3", "epochs": "5", "lr": "0.001"})
    assert isinstance(tcfg.l2_spikes, float) and tcfg.l2_spikes == 1e-3
    assert isinstance(tcfg.epochs, int) and tcfg.epochs == 5
    assert isinstance(tcfg.lr, float) and tcfg.lr == 0.001
    # the coercion makes the comparison that previously crashed safe
    assert not (tcfg.l2_spikes > 0) or True  # must not raise


def test_snn_config_coerces_numeric_strings():
    snn = SNNConfig.from_mapping({"n_hidden": "32", "bin_ms": "2.0", "tau_mem_ms": "20"})
    assert snn.n_hidden == 32 and isinstance(snn.n_hidden, int)
    assert snn.bin_ms == 2.0 and isinstance(snn.bin_ms, float)
    assert snn.tau_mem_ms == 20.0 and isinstance(snn.tau_mem_ms, float)


def test_load_config_override_scientific_notation(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(textwrap.dedent("""
        run:
          tag: t
        train:
          l2_spikes: 0.0
          epochs: 3
        """), encoding="utf-8")
    cfg = load_config(cfg_file, overrides=["train.l2_spikes=1e-3", "train.epochs=7"])
    assert cfg.get_path("train.l2_spikes") == 1e-3
    assert isinstance(cfg.get_path("train.l2_spikes"), float)
    assert cfg.get_path("train.epochs") == 7
    # and it flows through the dataclass without error
    tcfg = TrainConfig.from_mapping(cfg.get_path("train", {}))
    assert tcfg.l2_spikes == 1e-3 and tcfg.l2_spikes > 0