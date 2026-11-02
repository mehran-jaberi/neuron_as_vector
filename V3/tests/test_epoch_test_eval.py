"""Tests for per-epoch TEST scoring and the long-run checkpoint archive.

CPU-only.  These cover the config switch, the override grammar, the history
columns ``fit`` writes when a ``test_eval`` callable is supplied, and the
archive path naming.  No dataset is needed.
"""

from __future__ import annotations



from v3.config import TimingConfig, V3Config
from v3.control import valid_override_paths


def _cfg() -> V3Config:
    return V3Config(
        n_neurons=4,
        state_dim=8,
        mix_rank=4,
        n_inputs=5,
        n_classes=3,
        timing=TimingConfig(sequence_duration_ms=16.0, time_bin_ms=2.0),
        batch_size=2,
        device="cpu",
        amp=False,
        dtype="float32",
        epochs=1,
    )


def test_flag_defaults_off() -> None:
    assert _cfg().eval_test_each_epoch is False


def test_flag_is_overrideable() -> None:
    assert "eval_test_each_epoch" in valid_override_paths()
    from v3.config import apply_override_items

    cfg = apply_override_items(_cfg(), ["eval_test_each_epoch=true"])
    assert cfg.eval_test_each_epoch is True


def test_history_writer_keeps_test_columns(tmp_path: Path) -> None:
    from v3.experiment import _write_history_csv

    hist = [
        {"epoch": 1, "accuracy": 0.1, "val_accuracy": 0.2, "test_accuracy": 0.3,
         "test_correct": 3, "test_total": 10},
    ]
    p = tmp_path / "history.csv"
    _write_history_csv(hist, p)
    header, row = p.read_text(encoding="utf-8").splitlines()
    cols = header.split(",")
    assert {"test_accuracy", "test_correct", "test_total"} <= set(cols)
    values = dict(zip(cols, row.split(",")))
    assert values["test_accuracy"] == "0.3"
    assert values["test_correct"] == "3"
