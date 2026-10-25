"""Tests for the rewired recurrent-network control."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.rewiring import REWIRE_MODES, rewire_recurrent, rewiring_report


def _w(model):
    return model.w_rec.detach().cpu().numpy().astype(np.float64)


def test_modes_are_the_documented_three():
    assert set(REWIRE_MODES) == {"global", "rowwise", "columnwise"}


def test_global_rewiring_preserves_multiset_only(trained_like_model):
    rewired = rewire_recurrent(trained_like_model, mode="global", seed=3)
    a, b = _w(trained_like_model), _w(rewired)
    assert np.allclose(np.sort(a.ravel()), np.sort(b.ravel()))  # distribution preserved
    report = rewiring_report(trained_like_model, rewired, mode="global")
    assert report["weight_multiset_preserved"] is True
    assert report["row_multisets_preserved"] is False
    assert report["column_multisets_preserved"] is False
    assert report["fraction_positions_changed"] > 0.9
    # "preserved"/"destroyed" are documented in words
    assert "multiset" in report["preserved_summary"]
    assert "relational" in report["destroyed_summary"]


def test_rowwise_rewiring_preserves_each_incoming_multiset(trained_like_model):
    rewired = rewire_recurrent(trained_like_model, mode="rowwise", seed=4)
    a, b = _w(trained_like_model), _w(rewired)
    report = rewiring_report(trained_like_model, rewired, mode="rowwise")
    assert report["weight_multiset_preserved"] is True
    assert report["row_multisets_preserved"] is True  # incoming marginal preserved
    assert report["column_multisets_preserved"] is False
    for i in range(a.shape[0]):
        assert np.allclose(np.sort(a[i, :]), np.sort(b[i, :]))
    # relational structure destroyed: the matrix is not identical
    assert not np.allclose(a, b)


def test_columnwise_rewiring_preserves_each_outgoing_multiset(trained_like_model):
    rewired = rewire_recurrent(trained_like_model, mode="columnwise", seed=5)
    a, b = _w(trained_like_model), _w(rewired)
    report = rewiring_report(trained_like_model, rewired, mode="columnwise")
    assert report["weight_multiset_preserved"] is True
    assert report["column_multisets_preserved"] is True  # outgoing marginal preserved
    assert report["row_multisets_preserved"] is False
    for j in range(a.shape[1]):
        assert np.allclose(np.sort(a[:, j]), np.sort(b[:, j]))


def test_rewiring_changes_only_the_recurrent_matrix(trained_like_model):
    rewired = rewire_recurrent(trained_like_model, mode="global", seed=6)
    assert torch.allclose(trained_like_model.w_in, rewired.w_in)
    assert torch.allclose(trained_like_model.w_out, rewired.w_out)
    assert torch.allclose(trained_like_model.b_out, rewired.b_out)
    if trained_like_model.b_hid is not None:
        assert torch.allclose(trained_like_model.b_hid, rewired.b_hid)
    assert not torch.allclose(trained_like_model.w_rec, rewired.w_rec)


def test_rewiring_changes_the_network_function(trained_like_model, tiny_input):
    x = torch.as_tensor(np.asarray(tiny_input), dtype=trained_like_model.w_in.dtype)
    rewired = rewire_recurrent(trained_like_model, mode="global", seed=7)
    with torch.no_grad():
        a = trained_like_model(x)["logits"]
        b = rewired(x)["logits"]
    assert float((a - b).abs().max()) > 1e-6


def test_rewiring_is_deterministic_and_rejects_unknown_mode(trained_like_model):
    r1 = rewire_recurrent(trained_like_model, mode="global", seed=11)
    r2 = rewire_recurrent(trained_like_model, mode="global", seed=11)
    assert torch.allclose(r1.w_rec, r2.w_rec)
    r3 = rewire_recurrent(trained_like_model, mode="global", seed=12)
    assert not torch.allclose(r1.w_rec, r3.w_rec)
    with pytest.raises(ValueError):
        rewire_recurrent(trained_like_model, mode="not_a_mode")
