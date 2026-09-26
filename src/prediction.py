"""Cross-validated prediction of the functional fingerprint from the representation.

The Mantel/kNN analyses ask whether *distances* agree. This module asks the
stronger, out-of-sample question: can the label-free representation **predict** a
held-out neuron's functional fingerprint?

Protocol (leakage-safe)
-----------------------
* Models are fit on a subset of **neurons** and evaluated on the held-out neurons
  (``KFold`` shuffling over neurons, never over pairs).
* Standardisation and any hyper-parameter selection (ridge ``alpha``) are fit
  **inside the training fold only**; the held-out fold is never touched.
* The representation is *not* tuned against these results: the same fixed
  representation is used for every target.

Two simple regressors are provided:

* **Ridge regression** (``RidgeCV`` selects ``alpha`` by internal CV on the
  training fold).
* **k-nearest-neighbour regression** in representation space.

Reported per target dimension and aggregated: Pearson correlation, :math:`R^2`,
RMSE (and normalised RMSE), MAE, plus the Spearman correlation between the true
and predicted fingerprint *distance matrices* (a geometry-level summary).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

DEFAULT_ALPHAS: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Per-target and aggregate regression metrics for out-of-fold predictions."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    n, m = y_true.shape
    per_target: list[dict[str, Any]] = []
    for j in range(m):
        t = y_true[:, j]
        p = y_pred[:, j]
        ss_res = float(((t - p) ** 2).sum())
        ss_tot = float(((t - t.mean()) ** 2).sum())
        rmse = float(np.sqrt(np.mean((t - p) ** 2)))
        sd = float(t.std())
        per_target.append(
            {
                "target": j,
                "pearson_r": _pearson(t, p),
                "r2": (1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan"),
                "rmse": rmse,
                "nrmse": rmse / sd if sd > 1e-12 else float("nan"),
                "mae": float(np.mean(np.abs(t - p))),
            }
        )
    r2s = np.array([d["r2"] for d in per_target], dtype=np.float64)
    rs = np.array([d["pearson_r"] for d in per_target], dtype=np.float64)
    nrmse = np.array([d["nrmse"] for d in per_target], dtype=np.float64)
    rmse = np.array([d["rmse"] for d in per_target], dtype=np.float64)
    total_res = float(((y_true - y_pred) ** 2).sum())
    total_tot = float(((y_true - y_true.mean(axis=0, keepdims=True)) ** 2).sum())

    # geometry-level agreement of true vs predicted fingerprints
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr

    dist_r = float("nan")
    if n > 2 and m > 0:
        dt = pdist(y_true)
        dp = pdist(y_pred)
        if dt.std() > 1e-12 and dp.std() > 1e-12:
            dist_r = float(spearmanr(dt, dp).statistic)

    return {
        "n_folds_neurons": int(n),
        "n_targets": int(m),
        "pearson_r_mean": float(np.nanmean(rs)) if rs.size else float("nan"),
        "pearson_r_median": float(np.nanmedian(rs)) if rs.size else float("nan"),
        "r2_mean": float(np.nanmean(r2s)) if r2s.size else float("nan"),
        "r2_median": float(np.nanmedian(r2s)) if r2s.size else float("nan"),
        "r2_overall": (1.0 - total_res / total_tot) if total_tot > 1e-12 else float("nan"),
        "nrmse_mean": float(np.nanmean(nrmse)) if nrmse.size else float("nan"),
        "rmse_mean": float(np.nanmean(rmse)) if rmse.size else float("nan"),
        "rmse_median": float(np.nanmedian(rmse)) if rmse.size else float("nan"),
        "mae_mean": float(np.mean([d["mae"] for d in per_target])) if per_target else float("nan"),
        "predicted_vs_true_fingerprint_distance_spearman": dist_r,
        "per_target": per_target,
    }


# --------------------------------------------------------------------------
# Cross-validated predictors
# --------------------------------------------------------------------------
def make_shared_folds(n_samples: int, n_splits: int = 5, seed: int = 0):
    """Build one ``KFold`` splitter so every representation uses the SAME folds.

    Exposed separately from :func:`cross_validated_ridge` so a comparison across
    several representations of the same neurons can pass the *identical* splitter
    to each call (and can assert the fold assignment is identical).
    """
    from sklearn.model_selection import KFold

    n_splits = int(max(2, min(int(n_splits), int(n_samples))))
    return KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))


def _kfold(n_samples: int, n_splits: int, seed: int, cv: Any | None = None):
    if cv is not None:
        return cv, int(getattr(cv, "n_splits", n_splits))
    folds = make_shared_folds(n_samples, n_splits, seed)
    return folds, int(folds.n_splits)


def fold_assignment(n_samples: int, n_splits: int = 5, seed: int = 0, cv: Any | None = None) -> np.ndarray:
    """Fold id (0..n_splits-1) of every neuron for a given splitter."""
    splitter, _ = _kfold(n_samples, n_splits, seed, cv)
    out = np.full(int(n_samples), -1, dtype=np.int64)
    for f, (_, test) in enumerate(splitter.split(np.arange(int(n_samples)))):
        out[np.asarray(test, dtype=np.int64)] = f
    return out


def cross_validated_ridge(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    n_splits: int = 5,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    seed: int = 0,
    cv: Any | None = None,
) -> dict[str, Any]:
    """Out-of-fold ridge predictions of ``Y`` from ``X`` (CV across neurons).

    Pass an explicit ``cv`` splitter (see :func:`make_shared_folds`) to guarantee
    that several representations are compared on identical folds.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of neurons")
    cv, n_splits = _kfold(X.shape[0], n_splits, seed, cv)
    estimator = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.asarray(alphas, dtype=np.float64))),
        ]
    )
    pred = cross_val_predict(estimator, X, Y, cv=cv)
    metrics = regression_metrics(Y, pred)
    metrics.update({"model": "ridge", "n_splits": int(n_splits), "alphas": list(map(float, alphas))})
    return {"predictions": pred, "metrics": metrics}


def cross_validated_knn(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    n_splits: int = 5,
    k: int = 5,
    seed: int = 0,
    cv: Any | None = None,
) -> dict[str, Any]:
    """Out-of-fold k-NN regression predictions of ``Y`` from ``X``."""
    from sklearn.model_selection import cross_val_predict
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of neurons")
    cv, n_splits = _kfold(X.shape[0], n_splits, seed, cv)
    k_eff = int(max(1, min(k, X.shape[0] - 1)))
    estimator = Pipeline(
        [("scale", StandardScaler()), ("knn", KNeighborsRegressor(n_neighbors=k_eff))]
    )
    pred = cross_val_predict(estimator, X, Y, cv=cv)
    metrics = regression_metrics(Y, pred)
    metrics.update({"model": "knn", "n_splits": int(n_splits), "k": k_eff})
    return {"predictions": pred, "metrics": metrics}


def run_prediction_suite(
    representation: np.ndarray,
    fingerprints: Mapping[str, np.ndarray],
    *,
    n_splits: int = 5,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    knn_k: int = 5,
    seed: int = 0,
    random_control_seed: int | None = None,
) -> dict[str, Any]:
    """Ridge + kNN CV prediction of every fingerprint from the representation.

    When ``random_control_seed`` is given, an i.i.d. Gaussian representation of
    matched dimensionality is also predicted as a negative control (it should not
    predict the fingerprint).
    """
    X = np.asarray(representation, dtype=np.float64)
    out: dict[str, Any] = {"n_neurons": int(X.shape[0]), "n_representation_features": int(X.shape[1])}
    for name, Y in fingerprints.items():
        Y = np.asarray(Y, dtype=np.float64)
        ridge = cross_validated_ridge(X, Y, n_splits=n_splits, alphas=alphas, seed=seed)
        knn = cross_validated_knn(X, Y, n_splits=n_splits, k=knn_k, seed=seed)
        out[name] = {"ridge": ridge["metrics"], "knn": knn["metrics"]}
    if random_control_seed is not None:
        rng = np.random.default_rng(int(random_control_seed))
        Xr = rng.standard_normal(X.shape)
        for name, Y in fingerprints.items():
            ridge = cross_validated_ridge(
                Xr, np.asarray(Y, dtype=np.float64), n_splits=n_splits, alphas=alphas, seed=seed
            )
            out.setdefault("random_representation_control", {})[name] = ridge["metrics"]
    return out
