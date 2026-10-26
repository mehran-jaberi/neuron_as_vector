"""Fast, static tests for the mechanism ablations (``V3/run_ablation.py``).

No GPU, no dataset, no checkpoint and no training run: tiny synthetic tensors and
a 4-neuron stub model only.  The scientific runs are exercised by the CLI.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import run_ablation as rb
from v3.config import TimingConfig, V3Config
from v3.data import BatchIterator, Split
from v3.model import VectorNeuronPopulation


# ---------------------------------------------------------------------- #
# tiny model / stub data
# ---------------------------------------------------------------------- #
def _tiny_cfg(
    n_neurons: int = 4,
    state_dim: int = 6,
    n_inputs: int = 5,
    n_classes: int = 3,
    n_bins: int = 6,
    seed: int = 0,
) -> V3Config:
    return V3Config(
        n_neurons=n_neurons,
        state_dim=state_dim,
        n_inputs=n_inputs,
        n_classes=n_classes,
        timing=TimingConfig(sequence_duration_ms=float(n_bins), time_bin_ms=1.0),
        dtype="float32",
        device="cpu",
        batch_size=2,
        seed=seed,
    )


def _tiny_model(**kwargs) -> tuple[VectorNeuronPopulation, V3Config]:
    cfg = _tiny_cfg(**kwargs)
    return VectorNeuronPopulation(cfg).eval(), cfg


class _StubStore:
    """Minimal SHDEventStore stand-in: fixed 0/1 inputs, indexed by sample."""

    def __init__(self, n_bins: int, n_inputs: int, labels, n: int = 64, seed: int = 7):
        g = np.random.default_rng(seed)
        self._x = (g.random((n, n_bins, n_inputs)) > 0.8).astype(np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.n_bins, self.n_inputs = int(n_bins), int(n_inputs)

    def batch(self, indices):
        return self._x[np.asarray(indices, dtype=np.int64)]

    def labels_of(self, indices):
        return self.labels[np.asarray(indices, dtype=np.int64)]


# ---------------------------------------------------------------------- #
# knockout: theta is additive, so silencing means a large negative offset
# ---------------------------------------------------------------------- #
def test_knockout_offset_is_negative_because_theta_is_additive():
    # the model computes pot = <z, w_out> + theta, so a *positive* offset pins
    # the neuron above threshold; only a negative one silences it.
    assert rb.SILENCE_OFFSET < 0.0
    theta = torch.zeros(5)
    out = rb.silence_neuron(theta, 2)
    assert out.dtype == theta.dtype
    assert float(out[2]) == pytest.approx(rb.SILENCE_OFFSET)
    assert torch.equal(out[[0, 1, 3, 4]], torch.zeros(4))
    assert torch.equal(theta, torch.zeros(5))  # input untouched
    # twice from the pristine vector is identical: no cumulative drift
    assert torch.equal(out, rb.silence_neuron(theta, 2))


def test_silenced_neuron_never_spikes_while_positive_offset_pins_it_on():
    model, cfg = _tiny_model()
    x = torch.ones(2, cfg.n_bins, cfg.n_inputs)
    theta_orig = torch.full((model.n_neurons,), 50.0)  # force every neuron to fire
    with torch.no_grad():
        model.theta.data.copy_(theta_orig)
        _, base = model(x)
    base_counts = base.float().sum(1)
    assert int(base_counts[:, 1].sum()) == 2 * cfg.n_bins  # not vacuous

    with torch.no_grad():
        model.theta.data.copy_(rb.silence_neuron(theta_orig, 1))
        _, silenced = model(x)
        # the brief's "+1e9" leaves the neuron firing (theta is added, not subtracted)
        theta_plus = theta_orig.clone()
        theta_plus[1] = theta_plus[1] + 1e9
        model.theta.data.copy_(theta_plus)
        _, plus = model(x)
        model.theta.data.copy_(theta_orig)
        _, restored = model(x)

    sil_counts = silenced.float().sum(1)
    assert int(sil_counts[:, 1].sum()) == 0
    assert int(sil_counts[:, 0].sum()) > 0  # neighbours still active
    assert int(plus.float().sum(1)[:, 1].sum()) == 2 * cfg.n_bins
    assert torch.equal(restored, base)  # restoring theta restores the model


def test_silence_neuron_arithmetic_on_nonzero_theta():
    theta = torch.arange(6, dtype=torch.float32) * 0.5
    out = rb.silence_neuron(theta, 4)
    assert float(out[4]) == pytest.approx(2.0 + rb.SILENCE_OFFSET)
    assert torch.equal(out[torch.arange(6) != 4], theta[torch.arange(6) != 4])


# ---------------------------------------------------------------------- #
# counts / rates / readout arithmetic
# ---------------------------------------------------------------------- #
def test_rates_hz_from_counts_units():
    one_second = V3Config(
        timing=TimingConfig(sequence_duration_ms=1000.0, time_bin_ms=2.0),
        dtype="float32",
        device="cpu",
    )
    assert one_second.n_bins == 500
    rates = rb.rates_hz_from_counts(np.full((3, 2), 10.0), one_second)  # 10 spikes / 1 s
    assert rates.shape == (2,)
    assert np.allclose(rates, 10.0)

    short = _tiny_cfg(n_bins=8)  # 8 x 1 ms window
    rates_short = rb.rates_hz_from_counts(np.ones((2, 3)), short)
    assert np.allclose(rates_short, 125.0)


def test_gap_pp_anchor_and_sign():
    assert rb.gap_pp(1868, 2264) == pytest.approx(0.0, abs=1e-9)
    assert rb.gap_pp(1845, 2264) == pytest.approx(-1.0159, abs=1e-3)
    assert rb.gap_pp(1913, 2264) == pytest.approx(2.0, abs=0.05)


def test_readout_accuracy_is_argmax_of_counts_times_w():
    counts = np.array([[5.0, 0.0], [0.0, 5.0], [5.0, 0.0]], dtype=np.float32)
    labels = np.array([0, 1, 0])
    w = np.eye(2, dtype=np.float32)
    got = rb.readout_accuracy(counts, labels, w)
    assert got == {"correct": 3, "total": 3, "accuracy": 1.0}
    flipped = rb.readout_accuracy(counts, labels, w[::-1].copy())  # predicts 1,0,1
    assert flipped["correct"] == 0
    assert flipped["accuracy"] == 0.0


def test_capture_counts_on_stub_store_matches_manual_forward():
    model, cfg = _tiny_model(n_bins=6, n_inputs=5)
    n = 5
    labels = np.array([0, 1, 2, 0, 1], dtype=np.int64)
    split = Split("toy", np.arange(n), labels, np.zeros(n, dtype=np.int64))
    store = _StubStore(cfg.n_bins, cfg.n_inputs, labels)

    out = rb.capture_counts(model, cfg, torch.device("cpu"), store, split, "test", batch_size=2)

    manual_counts, manual_logits = [], []
    with torch.no_grad():
        for x_np, _y in BatchIterator(store, split, 2, shuffle=False):
            logits, spikes = model(torch.from_numpy(x_np))
            manual_counts.append(spikes.float().sum(1).numpy())
            manual_logits.append(logits.float().numpy())
    manual_counts = np.concatenate(manual_counts)
    manual_logits = np.concatenate(manual_logits)

    assert out["counts"].shape == (n, cfg.n_neurons)
    assert out["counts"].dtype == np.float32
    assert np.allclose(out["counts"], manual_counts)
    assert np.array_equal(out["labels"], labels)
    assert np.array_equal(out["indices"], np.arange(n))
    expected = int((manual_logits.argmax(1) == labels).sum())
    assert out["correct"] == expected
    assert out["accuracy"] == pytest.approx(expected / n)
    assert np.allclose(out["rates_hz"], rb.rates_hz_from_counts(out["counts"], cfg))
    assert out["seconds"] >= 0.0
    # the cached counts are exactly what the linear readout reads
    assert rb.test_correct(model, cfg, torch.device("cpu"), store, split, batch_size=2) == expected


# ---------------------------------------------------------------------- #
# readout-only fitting loop (synthetic tensors)
# ---------------------------------------------------------------------- #
def _separable_counts(n: int, seed: int = 0, boost: float = 40.0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    counts = rng.integers(0, 3, (n, 4)).astype(np.float32)
    counts[np.arange(n), y] += boost
    return counts, y


def test_fit_readout_learns_separable_synthetic_and_scores_test():
    cf, yf = _separable_counts(300, seed=0)
    cv, yv = _separable_counts(90, seed=1)
    ct, yt = _separable_counts(90, seed=2)
    cf_before, ct_before = cf.copy(), ct.copy()

    res = rb.fit_readout(cf, yf, cv, yv, ct, yt, epochs=20, batch_size=64)

    assert res["w"].shape == (4, 3)
    assert res["n_features"] == 4 and res["n_classes"] == 3
    assert res["test_total"] == 90
    assert 0.0 <= res["best_val_accuracy"] <= 1.0
    assert res["test_accuracy"] > 0.95
    assert res["epochs_run"] <= rb.READOUT_EPOCHS
    assert len(res["history"]) == res["epochs_run"]
    assert res["epoch_seconds_total"] >= 0.0
    assert np.array_equal(cf, cf_before) and np.array_equal(ct, ct_before)
    scored = rb.readout_accuracy(ct, yt, res["w"])
    assert scored["correct"] == res["test_correct"]
    assert scored["accuracy"] == pytest.approx(res["test_accuracy"])


def test_fit_readout_is_deterministic_and_honours_epoch_cap():
    cf, yf = _separable_counts(200, seed=3)
    cv, yv = _separable_counts(60, seed=4)
    ct, yt = _separable_counts(60, seed=5)

    a = rb.fit_readout(cf, yf, cv, yv, ct, yt, epochs=2, batch_size=64)
    b = rb.fit_readout(cf, yf, cv, yv, ct, yt, epochs=2, batch_size=64)

    assert a["epochs_run"] <= 2
    assert np.array_equal(a["w"], b["w"])
    assert a["test_correct"] == b["test_correct"]
    assert a["best_epoch"] == b["best_epoch"]


def test_fit_readout_rejects_mismatched_feature_dims():
    with pytest.raises(ValueError):
        rb.fit_readout(
            np.zeros((10, 3), dtype=np.float32), np.zeros(10, dtype=np.int64),
            np.zeros((4, 5), dtype=np.float32), np.zeros(4, dtype=np.int64),
            np.zeros((4, 3), dtype=np.float32), np.zeros(4, dtype=np.int64),
            epochs=1,
        )


def test_build_model_freezes_parameters():
    cfg = _tiny_cfg()
    model = rb.build_model(cfg, torch.device("cpu"))
    assert model.training is False
    assert all(not p.requires_grad for p in model.parameters())
    assert model.n_neurons == cfg.n_neurons


def test_a2_seed_changes_the_initial_network():
    assert rb.A2_SEED != 0
    torch.manual_seed(0)
    m0 = VectorNeuronPopulation(_tiny_cfg(seed=0))
    torch.manual_seed(0)
    m1 = VectorNeuronPopulation(_tiny_cfg(seed=rb.A2_SEED))
    assert not torch.equal(m0.w_in, m1.w_in)
    assert not torch.equal(m0.gate, m1.gate)
    torch.manual_seed(0)
    again = VectorNeuronPopulation(_tiny_cfg(seed=rb.A2_SEED))
    assert torch.equal(m1.w_in, again.w_in)  # reproducible


def test_load_payload_rejects_a_non_checkpoint(tmp_path):
    path = tmp_path / "not_a_checkpoint.pt"
    torch.save({"schema": "v3", "note": "not a checkpoint"}, path)
    with pytest.raises(ValueError):
        rb.load_payload(path)


# ---------------------------------------------------------------------- #
# B1 ranking helpers
# ---------------------------------------------------------------------- #
def test_entropy_bits():
    assert rb.entropy_bits(np.array([0, 0, 1, 1, 2, 2, 3, 3])) == pytest.approx(2.0)
    assert rb.entropy_bits(np.zeros(9, dtype=np.int64)) == 0.0
    assert rb.entropy_bits(np.array([], dtype=np.int64)) == 0.0


def test_mi_bits_perfect_and_independent():
    labels = np.array([0, 1, 2] * 100, dtype=np.int64)
    counts = labels.astype(np.float64)  # count == class: a perfect tuning curve
    assert rb.mi_bits(counts, labels) == pytest.approx(np.log2(3.0), abs=1e-9)
    binary = np.repeat([0, 1], 100)
    assert rb.mi_bits(100.0 * binary, binary) == pytest.approx(1.0, abs=1e-9)

    rng = np.random.default_rng(0)
    indep_counts = rng.integers(0, 6, 3000).astype(np.float64)
    indep_labels = rng.integers(0, 2, 3000).astype(np.int64)
    mi = rb.mi_bits(indep_counts, indep_labels, n_bins=8)
    assert 0.0 <= mi < 0.05  # plug-in bias is ~2.5e-3 bits here
    assert rb.mi_bits(np.full(500, 3.0), labels_repeat(500)) == 0.0


def labels_repeat(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.int64) % 3


def test_mi_bits_validation():
    with pytest.raises(ValueError):
        rb.mi_bits(np.zeros(4), np.zeros(5, dtype=np.int64))
    with pytest.raises(ValueError):
        rb.mi_bits(np.zeros(0), np.zeros(0, dtype=np.int64))


def test_mi_per_neuron_finds_the_informative_column():
    n = 400
    y = (np.arange(n) % 2).astype(np.int64)
    counts = np.stack(
        [
            10.0 * y,                       # perfect
            np.full(n, 5.0),                # constant: no information
            np.random.default_rng(1).integers(0, 9, n).astype(np.float64) * 3.0,
        ],
        axis=1,
    )
    mi = rb.mi_per_neuron(counts, y)
    assert mi.shape == (3,)
    assert mi[0] == pytest.approx(1.0, abs=1e-9)
    assert mi[1] == 0.0
    assert mi[2] < 0.05


def test_spearman_monotone_reversed_and_ties():
    x = np.arange(10, dtype=np.float64)
    assert rb.spearman(x, 2.0 * x + 1.0) == pytest.approx(1.0)
    assert rb.spearman(x, -x) == pytest.approx(-1.0)
    assert rb.spearman(x, np.zeros(10)) == 0.0
    assert rb.spearman(np.array([1.0, 2.0, 2.0, 3.0]), np.array([4.0, 4.0, 5.0, 9.0])) > 0.8
    assert rb.spearman(np.array([1.0, 1.0, 2.0, 2.0]), np.array([3.0, 3.0, 1.0, 1.0])) == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        rb.spearman(np.zeros(3), np.zeros(4))


def test_fingerprint_summary_ranking_and_concentration():
    mi = np.array([0.1, 0.4, 0.2, 0.3])
    rates = np.array([1.0, 2.0, 3.0, 4.0])
    s = rb.fingerprint_summary(mi, rates, k=2)
    assert s["n_neurons"] == 4
    assert s["mi_total_bits"] == pytest.approx(1.0)
    assert [row["neuron"] for row in s["top"]] == [1, 3]
    assert [row["neuron"] for row in s["bottom"]] == [0, 2]
    assert s["top"][0]["rate_hz"] == pytest.approx(2.0)
    assert s["top5_share"] == pytest.approx(0.7)
    assert s["effective_informative_neurons"] == pytest.approx(1.0 / 0.30)
    assert s["rank_order"] == [1, 3, 2, 0]
    assert "mi_test_bits" not in s["top"][0]

    with_test = rb.fingerprint_summary(mi, rates, mi_test=1.0 - mi, k=2)
    assert with_test["spearman_mi_fit_vs_test"] == pytest.approx(-1.0)
    assert with_test["top"][0]["mi_test_bits"] == pytest.approx(0.6)


def test_deletion_summary_known_drops():
    drops = np.array([3.0, -0.5, 1.5, 0.2, 0.0])
    s = rb.deletion_summary(drops, threshold_pp=1.0)
    assert s["max_drop_pp"] == 3.0
    assert s["min_drop_pp"] == -0.5
    assert s["most_important_neuron"] == 0
    assert s["n_above_threshold"] == 2
    assert s["n_above_zero"] == 3
    assert s["n_improved"] == 1
    assert s["sorted_neuron_order"] == [0, 2, 3, 4, 1]
    assert s["sorted_drops_pp"] == [3.0, 1.5, 0.2, 0.0, -0.5]
    assert s["mean_drop_pp"] == pytest.approx(float(drops.mean()))


# ---------------------------------------------------------------------- #
# CLI surface
# ---------------------------------------------------------------------- #
def test_parser_subcommands_and_defaults():
    p = rb.build_parser()
    freeze = p.parse_args(["freeze", "--arms", "a1"])
    assert freeze.func is rb.run_freeze
    assert freeze.arms == "a1"
    assert freeze.batch_size == 128
    assert freeze.checkpoint.endswith("v3_vector_d1000.pt")
    fp = p.parse_args(["fingerprints"])
    assert fp.func is rb.run_fingerprints
    assert fp.mi_bins == rb.MI_BINS
    ko = p.parse_args(["knockout"])
    assert ko.func is rb.run_knockout
    with pytest.raises(SystemExit):
        p.parse_args([])


def test_merge_arms_composes_partial_runs(tmp_path):
    path = tmp_path / "freeze.json"
    first = {"protocol": {"checkpoint": "x.pt", "a2_seed": 1}, "arms": {"A2": {"test_correct": 830}}}
    path.write_text(json.dumps(first), encoding="utf-8")
    merged = rb._merge_arms(path, {"protocol": {"a3_zeroed": ["w_rec"]},
                                   "arms": {"A1": {"test_correct": 1857}}})
    assert set(merged["arms"]) == {"A1", "A2"}
    assert merged["protocol"]["a2_seed"] == 1  # prior protocol keys kept
    assert merged["protocol"]["a3_zeroed"] == ["w_rec"]
    fresh = tmp_path / "absent.json"
    assert rb._merge_arms(fresh, {"arms": {"A1": {}}}) == {"arms": {"A1": {}}}


def test_protocol_constants_are_frozen():
    assert rb.REFERENCE_CORRECT == 1868 and rb.REFERENCE_TOTAL == 2264
    assert rb.READOUT_EPOCHS == 20
    assert rb.READOUT_PATIENCE < rb.READOUT_EPOCHS
    assert rb.DROP_THRESHOLD_PP == 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
