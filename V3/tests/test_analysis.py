"""Analysis infrastructure: state capture, ridge probes, state geometry.

CPU-only, synthetic: no dataset, no checkpoint and no training run.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import run_analysis as ra
from v3.config import TimingConfig, V3Config
from v3.data import Split
from v3.model import VectorNeuronPopulation


# ---------------------------------------------------------------------- #
# tiny model / stub data
# ---------------------------------------------------------------------- #
def _tiny_cfg(
    n_neurons: int = 5,
    state_dim: int = 8,
    n_inputs: int = 6,
    n_classes: int = 4,
    n_bins: int = 8,
) -> V3Config:
    return V3Config(
        n_neurons=n_neurons,
        state_dim=state_dim,
        n_inputs=n_inputs,
        n_classes=n_classes,
        timing=TimingConfig(sequence_duration_ms=float(n_bins), time_bin_ms=1.0),
        dtype="float32",
        device="cpu",
        batch_size=3,
    )


def _tiny_model(**kwargs):
    torch.manual_seed(0)
    cfg = _tiny_cfg(**kwargs)
    model = VectorNeuronPopulation(cfg).eval()
    return model, cfg


def _inputs(cfg: V3Config, batch: int = 3) -> torch.Tensor:
    g = torch.Generator().manual_seed(1)
    return (torch.rand(batch, cfg.n_bins, cfg.n_inputs, generator=g) > 0.8).float()


class _StubStore:
    """Minimal stand-in for SHDEventStore (batching only)."""

    def __init__(self, labels: np.ndarray, n_inputs: int, n_bins: int):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.speakers = np.zeros_like(self.labels)
        self.n_inputs = int(n_inputs)
        self.n_bins = int(n_bins)

    def batch(self, indices):
        idx = np.asarray(indices, dtype=np.int64)
        g = np.random.default_rng(7)
        x = (g.random((idx.size, self.n_bins, self.n_inputs)) > 0.8).astype(np.float32)
        return x

    def labels_of(self, indices):
        return self.labels[np.asarray(indices, dtype=np.int64)]


# ---------------------------------------------------------------------- #
# forward(return_states=...)
# ---------------------------------------------------------------------- #
def test_default_forward_arity_unchanged():
    model, cfg = _tiny_model()
    x = _inputs(cfg)
    with torch.no_grad():
        assert len(model(x)) == 2
        assert len(model(x, return_potentials=True)) == 3


def test_return_states_keeps_logits_and_spikes_bit_identical():
    model, cfg = _tiny_model()
    x = _inputs(cfg)
    with torch.no_grad():
        base_logits, base_spikes = model(x)
        logits, spikes, z_final, z_mean = model(x, return_states=True)
        logits_p, spikes_p, pots, z_final_p, z_mean_p = model(
            x, return_potentials=True, return_states=True
        )
    assert torch.equal(base_logits, logits)
    assert torch.equal(base_logits, logits_p)
    assert torch.equal(base_spikes, spikes)
    assert torch.equal(spikes, spikes_p)
    assert torch.equal(z_final, z_final_p)
    assert torch.equal(z_mean, z_mean_p)


def test_return_states_shapes_and_dtype():
    model, cfg = _tiny_model()
    x = _inputs(cfg)
    with torch.no_grad():
        logits, spikes, z_final, z_mean = model(x, return_states=True)
    b = x.shape[0]
    expected = (b, cfg.n_neurons, cfg.state_dim)
    assert z_final.shape == expected
    assert z_mean.shape == expected
    assert z_final.dtype == cfg.torch_dtype
    assert z_mean.dtype == cfg.torch_dtype
    assert torch.isfinite(z_final).all() and torch.isfinite(z_mean).all()
    assert float(z_final.abs().max()) < 1.0  # tanh dynamics
    # a running average over the time loop must differ from the final state
    assert not torch.allclose(z_final, z_mean)
    assert spikes.shape == (b, cfg.n_bins, cfg.n_neurons)


# ---------------------------------------------------------------------- #
# capture_split on a stub store
# ---------------------------------------------------------------------- #
def test_capture_split_shapes_and_readout():
    model, cfg = _tiny_model(n_bins=8)
    n = 6
    labels = np.array([0, 1, 2, 3, 0, 1], dtype=np.int64)
    split = Split("toy", np.arange(n), labels, np.zeros(n, dtype=np.int64))
    store = _StubStore(labels, cfg.n_inputs, cfg.n_bins)
    out = ra.capture_split(
        model, cfg, torch.device("cpu"), store, split, "test", batch_size=2
    )
    assert out["counts"].shape == (n, cfg.n_neurons)
    assert out["counts"].dtype == np.float32
    assert out["final_state"].shape == (n, cfg.n_neurons, cfg.state_dim)
    assert out["final_state"].dtype == np.float16
    assert out["mean_state"].shape == (n, cfg.n_neurons, cfg.state_dim)
    assert out["mean_state"].dtype == np.float16
    assert np.array_equal(out["labels"], labels)
    assert np.array_equal(out["indices"], np.arange(n))
    assert out["correct"] == int((out["logits"].argmax(1) == labels).sum())
    assert 0.0 <= out["accuracy"] <= 1.0


# ---------------------------------------------------------------------- #
# ridge probe helpers
# ---------------------------------------------------------------------- #
def _synthetic_features(n_per_class: int = 40, dim: int = 10, n_classes: int = 3):
    g = torch.Generator().manual_seed(0)
    centers = torch.randn(n_classes, dim, generator=g) * 3.0
    x = torch.cat(
        [centers[c] + torch.randn(n_per_class, dim, generator=g) for c in range(n_classes)]
    )
    y = torch.arange(n_classes).repeat_interleave(n_per_class)
    idx = torch.arange(x.shape[0])
    keep = idx % 5 != 0
    return x[keep], y[keep], x[~keep], y[~keep]


def test_col_stats_handles_constant_column():
    x = torch.ones(10, 3)
    x[:, 1] = torch.arange(10, dtype=torch.float32)
    mean, std = ra._col_stats(x)
    assert torch.allclose(mean, x.mean(0))
    assert torch.isfinite(std).all()
    assert torch.all(std > 0)  # zero-variance columns get a positive floor


def test_ridge_probe_learns_synthetic_classes():
    xf, yf, xt, yt = _synthetic_features()
    mean, std = ra._col_stats(xf)
    xf = ra._standardize_(xf.clone(), mean, std)
    xt = ra._standardize_(xt.clone(), mean, std)
    k_fit = ra._linear_kernel(xf, xf)
    k_test = ra._linear_kernel(xt, xf)
    grid = ra.default_lam_grid(k_fit)
    target = ra._one_hot(yf.numpy(), 3)

    alpha, y_mean, lam, report = ra.fit_ridge_gcv(k_fit, target, grid)
    scores = ra.predict_ridge(k_test, alpha, y_mean)

    assert alpha.shape == (xf.shape[0], 3)
    assert y_mean.shape == (3,)
    assert lam in grid
    assert report["n_fit"] == xf.shape[0]
    assert set(report["gcv"]) == {repr(float(g)) for g in grid}
    accuracy = float((scores.argmax(1) == yt).float().mean())
    assert accuracy > 0.9


def test_predict_ridge_matches_closed_form_on_identity_kernel():
    # K = I, lam chosen by GCV; scores must equal a soft ridge on the targets
    y = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    k = torch.eye(3, dtype=torch.float64) * 2.0
    alpha, y_mean, lam, _ = ra.fit_ridge_gcv(k, y, [0.5])
    scores = ra.predict_ridge(k, alpha, y_mean)
    expected = (
        y - y.mean(0)
    ) * (2.0 / (2.0 + lam)) + y.mean(0)
    assert torch.allclose(scores, expected, atol=1e-9)


# ---------------------------------------------------------------------- #
# geometry helpers
# ---------------------------------------------------------------------- #
def test_spectrum_summary_known_spectrum():
    s = ra.spectrum_summary(torch.tensor([5.0, 5.0, 5.0, 1e-9, 1e-9]))
    assert s["participation_ratio"] == pytest.approx(3.0, abs=1e-6)
    assert s["n_components_90"] == 3
    assert s["n_components_95"] == 3
    assert s["n_components_99"] == 3
    assert s["top_1_variance_fraction"] == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert s["n_components_90"] <= s["n_components_95"] <= s["n_components_99"]


def test_spectrum_summary_rank_one():
    s = ra.spectrum_summary(torch.tensor([7.0, 0.0, 0.0]))
    assert s["participation_ratio"] == pytest.approx(1.0)
    assert s["n_components_90"] == 1
    assert s["top_10_variance_fraction"] == pytest.approx(1.0)


def test_pca_eigenvalues_match_svd():
    g = torch.Generator().manual_seed(2)
    x = torch.randn(40, 6, generator=g)
    eigs = ra.pca_eigenvalues(x.clone())
    centered = x - x.mean(0)
    ref = torch.linalg.svdvals(centered).square().sort(descending=True).values
    assert eigs.shape == (6,)
    assert torch.allclose(eigs, ref.double(), rtol=1e-5, atol=1e-8)
    assert torch.all(eigs[:-1] >= eigs[1:] - 1e-8)


def test_pca_eigenvalues_uses_smaller_gram_for_wide_matrices():
    g = torch.Generator().manual_seed(3)
    x = torch.randn(5, 30, generator=g)  # n < d -> samples x samples Gram
    eigs = ra.pca_eigenvalues(x.clone())
    assert eigs.shape == (5,)
    assert eigs[0] >= eigs[-1]


def test_dimension_variance_summary_isotropic():
    g = torch.Generator().manual_seed(4)
    x = torch.randn(400, 12, generator=g) * 2.0
    s = ra.dimension_variance_summary(x)
    assert s["n_dims"] == 12
    assert s["zero_variance_dims"] == 0
    assert 10.0 < s["participation_ratio"] <= 12.0


def test_load_matrix_flattens_and_casts(tmp_path):
    path = tmp_path / "capture.npz"
    counts = np.arange(12, dtype=np.float32).reshape(4, 3)
    state = np.arange(4 * 3 * 2, dtype=np.float16).reshape(4, 3, 2)
    np.savez(path, counts=counts, mean_state=state)

    c = ra._load_matrix(path, "counts")
    s = ra._load_matrix(path, "mean_state")
    assert c.shape == (4, 3) and c.dtype == torch.float32
    assert s.shape == (4, 6) and s.dtype == torch.float32
    assert torch.equal(c, torch.from_numpy(counts))
    assert torch.equal(s, torch.from_numpy(state.reshape(4, 6)).float())


def test_analysis_paths():
    assert ra.capture_path("", "test").name == "capture_test.npz"
    assert ra.capture_path("scalar_d1", "test").name == "capture_scalar_d1_test.npz"
    assert ra.results_path("geometry").name == "geometry.json"
    assert ra.results_path("geometry", "scalar_d1").name == "geometry_scalar_d1.json"


def test_stratified_folds_balances_every_class():
    labels = np.array([0] * 10 + [1] * 10 + [2] * 10)
    fold = ra.stratified_folds(labels, n_folds=5, seed=0)
    assert fold.shape == labels.shape
    for c in (0, 1, 2):
        counts = np.bincount(fold.numpy()[labels == c], minlength=5)
        assert counts.tolist() == [2, 2, 2, 2, 2]


def test_select_lam_cv_returns_grid_member_and_curve():
    torch.manual_seed(0)
    labels = np.array([0] * 30 + [1] * 30)
    centres = torch.tensor([[1.5, 0.0], [0.0, 1.5]])
    x = centres[torch.from_numpy(labels)] + 0.3 * torch.randn(60, 2)
    k = x @ x.t()
    y = torch.zeros(60, 2)
    y[torch.arange(60), torch.from_numpy(labels)] = 1.0
    grid = [0.01, 0.1, 1.0, 10.0]
    lam, report, meta = ra.select_lam_cv(k, y, labels, grid, n_folds=3, seed=0)
    assert lam in grid
    assert len(report["curve"]) == len(grid)
    assert 0.0 <= report["cv_accuracy"] <= 1.0
    assert meta["n_folds"] == 3
    # the CV-selected probe must still separate this easy problem
    alpha = ra.ridge_alpha(ra.ridge_eig(k, y), lam)
    preds = ra.predict_ridge(k, alpha, torch.zeros(2, dtype=torch.float64))
    assert float((preds.argmax(1) == torch.from_numpy(labels)).float().mean()) > 0.9


def test_gcv_scores_agree_with_fit_ridge_gcv():
    torch.manual_seed(1)
    labels = np.array([0] * 20 + [1] * 20)
    x = torch.randn(40, 5) + 2.0 * torch.tensor([[1.0], [0.0]])[torch.from_numpy(labels)]
    k = x @ x.t()
    y = torch.zeros(40, 2)
    y[torch.arange(40), torch.from_numpy(labels)] = 1.0
    grid = [0.01, 0.1, 1.0, 10.0]
    eig = ra.ridge_eig(k, y)
    curve, lam, _ = ra.gcv_scores(eig, grid)
    alpha, _, lam_wrapper, report = ra.fit_ridge_gcv(k, y, grid)
    assert lam == lam_wrapper
    assert report["lam"] == lam
    assert torch.equal(alpha, ra.ridge_alpha(eig, lam))
    assert len(curve) == len(grid)


# ---------------------------------------------------------------------- #
# primary probe: multinomial logistic regression
# ---------------------------------------------------------------------- #
def _toy_logistic_data(n: int = 120, d: int = 6, k: int = 3, seed: int = 0):
    """Linearly separable-ish Gaussian blobs for the logistic probe."""
    g = torch.Generator().manual_seed(int(seed))
    centers = torch.randn(k, d, generator=g) * 2.0
    y = torch.randint(0, k, (n,), generator=g)
    x = centers[y] + torch.randn(n, d, generator=g)
    return x, y.numpy()


def test_fit_logistic_probe_shapes_and_predict():
    x_fit, y_fit = _toy_logistic_data(120, 6, 3, seed=0)
    x_val, y_val = _toy_logistic_data(48, 6, 3, seed=1)
    state = ra.fit_logistic_probe(
        x_fit.clone(), y_fit, x_val.clone(), y_val, lam=1e-2, epochs=5
    )
    assert state["weight"].shape == (6, 3)
    assert state["bias"].shape == (3,)
    assert state["feature_mean"].shape == (6,)
    assert state["feature_std"].shape == (6,)
    assert state["dim"] == 6 and state["n_fit"] == 120
    assert state["n_classes"] == 3
    assert 0.0 <= state["fit_accuracy"] <= 1.0
    assert list(state["val_curve"]) == ["0.01"]
    x_test, y_test = _toy_logistic_data(40, 6, 3, seed=2)
    assert ra.predict_logits(state, x_test).shape == (40, 3)
    scored = ra.probe_accuracy(state, x_test, y_test)
    assert scored["total"] == 40
    assert 0 <= scored["correct"] <= 40
    assert 0.0 <= scored["accuracy"] <= 1.0


def test_fit_logistic_probe_deterministic_on_fixed_seed():
    def fit():
        x_fit, y_fit = _toy_logistic_data(96, 5, 3, seed=3)
        x_val, y_val = _toy_logistic_data(32, 5, 3, seed=4)
        return ra.fit_logistic_probe(
            x_fit, y_fit, x_val, y_val, lam_grid=(1e-2, 1e-1), epochs=4
        )

    a, b = fit(), fit()
    assert torch.equal(a["weight"], b["weight"])
    assert torch.equal(a["bias"], b["bias"])
    assert a["val_curve"] == b["val_curve"]
    assert a["fit_accuracy"] == b["fit_accuracy"]


def test_fit_logistic_probe_standardises_on_fit_only():
    """Feature statistics must come from FIT, and TEST must reuse them."""
    x_fit, y_fit = _toy_logistic_data(100, 6, 3, seed=5)
    x_val, y_val = _toy_logistic_data(40, 6, 3, seed=6)
    mean, std = ra._col_stats(x_fit)
    # a wildly rescaled VAL must not change the stored FIT statistics
    state = ra.fit_logistic_probe(
        x_fit.clone(), y_fit, (x_val * 50.0), y_val, lam=1e-2, epochs=4
    )
    assert torch.allclose(state["feature_mean"], mean)
    assert torch.allclose(state["feature_std"], std)
    # predict_logits standardises raw TEST with the stored FIT statistics
    x_test = torch.randn(17, 6, generator=torch.Generator().manual_seed(9)) * 3.0 + 1.0
    expected = ((x_test - mean) / std) @ state["weight"] + state["bias"]
    assert torch.allclose(ra.predict_logits(state, x_test), expected, atol=1e-5)


def test_fit_logistic_probe_without_standardisation():
    x_fit, y_fit = _toy_logistic_data(80, 5, 3, seed=7)
    x_val, y_val = _toy_logistic_data(24, 5, 3, seed=8)
    state = ra.fit_logistic_probe(
        x_fit.clone(), y_fit, x_val.clone(), y_val,
        lam=1e-2, epochs=4, standardize=False,
    )
    assert state["feature_mean"] is None and state["feature_std"] is None
    x_test, _ = _toy_logistic_data(10, 5, 3, seed=9)
    logits = ra.predict_logits(state, x_test)
    expected = x_test @ state["weight"] + state["bias"]
    assert torch.allclose(logits, expected)


def test_paired_bootstrap_uses_shared_resamples():
    labels = np.arange(30) % 3
    perfect = labels.copy()
    worse = (labels + 1) % 3  # always wrong on 3 classes
    res = ra.paired_bootstrap(perfect, worse, labels, n_boot=200, seed=0)
    assert res["delta"] == pytest.approx(1.0)
    assert res["n_boot"] == 200
    assert res["excludes_zero"] is True
    assert res["ci95"][0] > 0.0
    same = ra.paired_bootstrap(perfect, perfect, labels, n_boot=50, seed=0)
    assert same["delta"] == 0.0
    assert same["ci95"] == [0.0, 0.0]
    assert same["excludes_zero"] is False


def test_counts_gate_window_matches_brief():
    assert ra._counts_gate_window({"correct": 1868, "total": 2264}) == (1845, 1890)
    assert ra._counts_gate_window({"correct": 900, "total": 1000}, pp=0.01) == (890, 910)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
