"""Geometry-function analysis: does neuron-space geometry predict function?

Primary hypothesis
------------------
Neurons that are close in the proposed representation space should tend to have
similar functional fingerprints, i.e. the *distance matrices* of the two spaces
should be positively related.

Primary metric (pre-registered in the config, reported everywhere)
-----------------------------------------------------------------
``mantel_spearman_r``: Spearman correlation between the condensed pairwise
distance vector of the neuron representation and the condensed pairwise distance
vector of the functional fingerprint, with a one-sided permutation p-value for
``r > 0`` (Mantel test). Spearman is used instead of Pearson because
distance-distance relationships are monotone but strongly non-linear, and Spearman
is robust to the heavy tail of a few very distant pairs.

Supporting analyses
-------------------
* Pearson correlation (optionally on log-distances).
* kNN analysis: for each neuron, the mean functional distance to its ``k``
  nearest neighbours in representation space, compared against a permutation null
  obtained by randomly relabelling neurons. Answers the *local* question, which a
  single global Mantel statistic can miss.
* Binned distance-distance curve with standard errors (for the figure).
* Partial Mantel controlling for a nuisance variable (used to ask whether the
  relationship survives after accounting for firing-rate differences).

All permutation tests use the same vectorised relabelling scheme: a permutation
``p`` maps the functional matrix ``D`` to ``D[p][:, p]``, which is exactly the
null hypothesis "the representation-neuron pairing is arbitrary".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _pair_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n, 1)


def condensed_to_matrix(condensed: np.ndarray, n: int) -> np.ndarray:
    from scipy.spatial.distance import squareform

    if condensed.size == 0:
        return np.zeros((n, n))
    return squareform(condensed)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3:
        return float("nan")
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def _spearman_from_ranks(rx: np.ndarray, ry: np.ndarray) -> float:
    return _pearson(rx, ry)


def _residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Residual of ``y`` after least-squares projection onto ``[1, x]``."""
    X = np.column_stack([np.ones_like(x), x])
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:  # pragma: no cover
        return y - y.mean()
    return y - X @ beta


@dataclass
class MantelResult:
    """Outcome of a Mantel-style permutation test."""

    statistic: float
    p_value: float
    null_mean: float
    null_std: float
    n_perm: int
    method: str
    alternative: str
    effect_size: float
    n_neurons: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "statistic": self.statistic,
            "p_value": self.p_value,
            "null_mean": self.null_mean,
            "null_std": self.null_std,
            "n_perm": self.n_perm,
            "method": self.method,
            "alternative": self.alternative,
            "effect_size_z": self.effect_size,
            "n_neurons": self.n_neurons,
            "n_pairs": int(self.n_neurons * (self.n_neurons - 1) / 2),
        }


def mantel_test(
    dx: np.ndarray,
    dy: np.ndarray,
    *,
    method: str = "spearman",
    n_perm: int = 1000,
    seed: int = 0,
    alternative: str = "greater",
    n_neurons: int | None = None,
) -> MantelResult:
    """Permutation Mantel test between two condensed distance vectors.

    ``dx`` is held fixed; ``dy`` is relabelled by a random permutation of the
    neurons, which is the standard Mantel null. ``alternative="greater"`` tests
    the primary hypothesis that the two distance matrices are positively related.
    """
    dx = np.asarray(dx, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)
    if dx.shape != dy.shape:
        raise ValueError(f"Distance vectors must have the same length, got {dx.shape} and {dy.shape}")
    m = dx.size
    if m < 3:
        return MantelResult(float("nan"), float("nan"), float("nan"), float("nan"), n_perm, method, alternative, float("nan"), 0)
    n = n_neurons if n_neurons is not None else int(round((1 + np.sqrt(1 + 8 * m)) / 2))
    ii, jj = _pair_indices(n)
    if ii.size != m:
        raise ValueError("Could not infer the number of neurons from the distance vector length")

    if method == "spearman":
        rx = rankdata(dx)
        ry = rankdata(dy)
        observed = _spearman_from_ranks(rx, ry)
    elif method == "pearson":
        rx = dx
        ry = dy
        observed = _pearson(rx, ry)
    elif method == "pearson_log":
        rx = np.log1p(dx)
        ry = np.log1p(dy)
        observed = _pearson(rx, ry)
    else:
        raise ValueError(f"Unknown Mantel method {method!r}")

    dmat = condensed_to_matrix(dy, n)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    for k in range(n_perm):
        p = rng.permutation(n)
        # Relabelled functional distances, computed without materialising D[p][:, p].
        dy_p = dmat[p[ii], p[jj]]
        if method == "spearman":
            null[k] = _spearman_from_ranks(rx, rankdata(dy_p))
        elif method == "pearson":
            null[k] = _pearson(rx, dy_p)
        else:
            null[k] = _pearson(rx, np.log1p(dy_p))

    null_mean = float(np.nanmean(null)) if np.isfinite(null).any() else float("nan")
    null_std = float(np.nanstd(null)) if np.isfinite(null).any() else float("nan")
    if alternative == "greater":
        p_value = float((1 + np.sum(null >= observed)) / (1 + n_perm))
    elif alternative == "less":
        p_value = float((1 + np.sum(null <= observed)) / (1 + n_perm))
    else:
        centre = null_mean
        p_value = float((1 + np.sum(np.abs(null - centre) >= abs(observed - centre))) / (1 + n_perm))
    effect = float((observed - null_mean) / null_std) if null_std > 1e-12 else float("nan")
    return MantelResult(
        statistic=float(observed),
        p_value=p_value,
        null_mean=null_mean,
        null_std=null_std,
        n_perm=int(n_perm),
        method=method,
        alternative=alternative,
        effect_size=effect,
        n_neurons=int(n),
    )


def partial_mantel_test(
    dx: np.ndarray,
    dy: np.ndarray,
    dz: np.ndarray,
    *,
    n_perm: int = 1000,
    seed: int = 0,
    method: str = "spearman",
) -> MantelResult:
    """Partial Mantel test: relationship between ``dx`` and ``dy`` controlling for ``dz``.

    Ranks of all three distance vectors are computed, ``dx`` and ``dy`` are
    residualised on ``rz``, and the correlation of the residuals is tested against
    the same relabelling null. Used to ask whether the representation carries
    information about function *beyond* a nuisance variable such as firing rate.
    """
    dx = np.asarray(dx, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)
    dz = np.asarray(dz, dtype=np.float64)
    if method != "spearman":
        raise ValueError("partial_mantel_test currently supports method='spearman'")
    if dx.size < 3:
        return MantelResult(float("nan"), float("nan"), float("nan"), float("nan"), n_perm, "partial_spearman", "greater", float("nan"), 0)
    n = int(round((1 + np.sqrt(1 + 8 * dx.size)) / 2))
    ii, jj = _pair_indices(n)

    rx = _residualize(rankdata(dx), rankdata(dz))
    ry = _residualize(rankdata(dy), rankdata(dz))
    observed = _pearson(rx, ry)

    dmat = condensed_to_matrix(dy, n)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    for k in range(n_perm):
        p = rng.permutation(n)
        dy_p = dmat[p[ii], p[jj]]
        ry_p = _residualize(rankdata(dy_p), rankdata(dz))
        null[k] = _pearson(rx, ry_p)

    if np.isfinite(null).any():
        null_mean, null_std = float(np.nanmean(null)), float(np.nanstd(null))
    else:
        null_mean = null_std = float("nan")
    p_value = float((1 + np.sum(null >= observed)) / (1 + n_perm))
    effect = float((observed - null_mean) / null_std) if null_std > 1e-12 else float("nan")
    return MantelResult(
        statistic=float(observed),
        p_value=p_value,
        null_mean=null_mean,
        null_std=null_std,
        n_perm=int(n_perm),
        method="partial_spearman",
        alternative="greater",
        effect_size=effect,
        n_neurons=int(n),
    )


# --------------------------------------------------------------------------
# k-nearest-neighbour analysis
# --------------------------------------------------------------------------
@dataclass
class KNNResult:
    """Observed vs null functional similarity of representation-space neighbours."""

    k: int
    observed_mean_func_distance: float
    null_mean: float
    null_std: float
    p_value: float
    effect_size: float
    per_neuron_observed: np.ndarray
    per_neuron_null_mean: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": int(self.k),
            "observed_mean_neighbour_func_distance": self.observed_mean_func_distance,
            "null_mean": self.null_mean,
            "null_std": self.null_std,
            "p_value": self.p_value,
            "effect_size_z": self.effect_size,
            "n_neurons": int(self.per_neuron_observed.size),
        }


def knn_analysis(
    space_matrix: np.ndarray,
    func_distance_matrix: np.ndarray,
    *,
    k_values: Sequence[int] = (3, 5, 10, 20),
    n_perm: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare functional similarity of representation neighbours against a null.

    For every neuron, its ``k`` nearest neighbours in representation space are
    identified and the mean functional-fingerprint distance to them is computed.
    The null distribution is obtained by randomly relabelling neurons, i.e. by
    asking "what if neighbour identity were unrelated to function?".

    A *negative* effect (observed < null) means neighbours are functionally more
    similar than chance - the hypothesis of interest.
    """
    X = np.asarray(space_matrix, dtype=np.float64)
    D = np.asarray(func_distance_matrix, dtype=np.float64)
    n = X.shape[0]
    if D.shape != (n, n):
        raise ValueError(f"Functional distance matrix shape {D.shape} does not match {n} neurons")
    rng = np.random.default_rng(seed)
    results: dict[str, Any] = {"k_values": [int(k) for k in k_values], "per_k": {}}
    if n < 3:
        return results

    # Representation-space neighbour lists (excluding self).
    from scipy.spatial.distance import cdist

    Brep = cdist(X, X)
    np.fill_diagonal(Brep, np.inf)
    order = np.argsort(Brep, axis=1, kind="stable")

    for k in k_values:
        k = int(min(k, n - 1))
        if k < 1:
            continue
        neighbours = order[:, :k]  # (n, k)
        rows = np.repeat(np.arange(n), k)
        cols = neighbours.ravel()

        per_neuron_obs = D[rows, cols].reshape(n, k).mean(axis=1)
        observed = float(per_neuron_obs.mean())

        null_means = np.empty(n_perm, dtype=np.float64)
        per_neuron_null = np.empty((n_perm, n), dtype=np.float64)
        for t in range(n_perm):
            p = rng.permutation(n)
            values = D[p[rows], p[cols]].reshape(n, k).mean(axis=1)
            per_neuron_null[t] = values
            null_means[t] = values.mean()

        null_mean = float(null_means.mean())
        null_std = float(null_means.std())
        p_value = float((1 + np.sum(null_means <= observed)) / (1 + n_perm))
        effect = float((null_mean - observed) / null_std) if null_std > 1e-12 else float("nan")
        res = KNNResult(
            k=k,
            observed_mean_func_distance=observed,
            null_mean=null_mean,
            null_std=null_std,
            p_value=p_value,
            effect_size=effect,
            per_neuron_observed=per_neuron_obs,
            per_neuron_null_mean=per_neuron_null.mean(axis=0),
        )
        results["per_k"][int(k)] = res
    return results


def knn_table(knn_results: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten :func:`knn_analysis` output into rows suitable for CSV/DataFrame."""
    rows: list[dict[str, Any]] = []
    for k, res in sorted(knn_results.get("per_k", {}).items()):
        row = res.to_dict()
        if "per_neuron_observed" in row:
            row.pop("per_neuron_observed")
        if "per_neuron_null_mean" in row:
            row.pop("per_neuron_null_mean")
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Binned distance-distance curve
# --------------------------------------------------------------------------
def distance_correlation_curve(
    dx: np.ndarray,
    dy: np.ndarray,
    *,
    n_bins: int = 20,
    binning: str = "quantile",
) -> dict[str, np.ndarray]:
    """Mean functional distance as a function of representation distance.

    Binning is by quantiles of ``dx`` by default, so every bin contains a
    comparable number of pairs even though distance distributions are skewed.
    Returns bin centres, mean/sem of ``dy``, pair counts and the raw pair cloud
    (subsampled for plotting).
    """
    dx = np.asarray(dx, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)
    if dx.size == 0:
        empty = np.zeros(0)
        return {"bin_center": empty, "bin_mean": empty, "bin_sem": empty, "bin_count": empty, "x": empty, "y": empty}

    if binning == "quantile":
        edges = np.quantile(dx, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        if edges.size < 3:
            edges = np.linspace(dx.min() - 1e-9, dx.max() + 1e-9, 3)
        bin_idx = np.clip(np.digitize(dx, edges[1:-1], right=False), 0, edges.size - 2)
    elif binning == "equal":
        edges = np.linspace(dx.min(), dx.max(), n_bins + 1)
        bin_idx = np.clip(np.digitize(dx, edges[1:-1], right=False), 0, n_bins - 1)
    else:
        raise ValueError(f"Unknown binning {binning!r}")

    n_bins_eff = int(bin_idx.max()) + 1
    centres = np.zeros(n_bins_eff)
    means = np.zeros(n_bins_eff)
    sems = np.zeros(n_bins_eff)
    counts = np.zeros(n_bins_eff, dtype=int)
    for b in range(n_bins_eff):
        sel = bin_idx == b
        counts[b] = int(sel.sum())
        if counts[b] == 0:
            centres[b] = np.nan
            means[b] = np.nan
            sems[b] = np.nan
            continue
        centres[b] = float(dx[sel].mean())
        means[b] = float(dy[sel].mean())
        sems[b] = float(dy[sel].std(ddof=1) / np.sqrt(counts[b])) if counts[b] > 1 else 0.0

    # Subsample the cloud for the scatter panel (keeps figures light).
    max_points = 20000
    if dx.size > max_points:
        rng = np.random.default_rng(0)
        sel = rng.choice(dx.size, size=max_points, replace=False)
        xs, ys = dx[sel], dy[sel]
    else:
        xs, ys = dx, dy
    return {
        "bin_center": centres,
        "bin_mean": means,
        "bin_sem": sems,
        "bin_count": counts,
        "x": xs,
        "y": ys,
    }


# --------------------------------------------------------------------------
# Composite entry point
# --------------------------------------------------------------------------
def geometry_function_analysis(
    space_matrix: np.ndarray,
    fingerprint_matrix: np.ndarray,
    *,
    representation_metric: str = "euclidean",
    fingerprint_metric: str = "euclidean",
    n_perm: int = 1000,
    k_values: Sequence[int] = (3, 5, 10, 20),
    seed: int = 0,
    n_curve_bins: int = 20,
    nuisance_condensed: np.ndarray | None = None,
    include_curves: bool = True,
) -> dict[str, Any]:
    """Run the full primary + supporting analysis for one (representation, fingerprint) pair.

    ``nuisance_condensed`` optionally supplies a distance vector to control for
    (typically the absolute difference in log firing rate), enabling the partial
    Mantel control that answers "is this more than firing rate?".
    """
    from scipy.spatial.distance import pdist

    X = np.asarray(space_matrix, dtype=np.float64)
    Y = np.asarray(fingerprint_matrix, dtype=np.float64)
    n = X.shape[0]

    dx = pdist(X, metric=representation_metric) if n > 1 else np.zeros(0)
    dy = pdist(Y, metric=fingerprint_metric) if n > 1 else np.zeros(0)

    out: dict[str, Any] = {"n_neurons": int(n), "n_pairs": int(dx.size)}

    primary = mantel_test(dx, dy, method="spearman", n_perm=n_perm, seed=seed, alternative="greater")
    out["primary_mantel_spearman"] = primary.to_dict()
    out["secondary_mantel_pearson"] = mantel_test(
        dx, dy, method="pearson", n_perm=n_perm, seed=seed, alternative="greater"
    ).to_dict()
    out["secondary_mantel_pearson_log"] = mantel_test(
        dx, dy, method="pearson_log", n_perm=n_perm, seed=seed, alternative="greater"
    ).to_dict()

    if nuisance_condensed is not None and np.asarray(nuisance_condensed).size == dx.size:
        out["partial_mantel_controlling_for_nuisance"] = partial_mantel_test(
            dx, dy, np.asarray(nuisance_condensed, dtype=np.float64), n_perm=n_perm, seed=seed
        ).to_dict()

    D_func = condensed_to_matrix(dy, n)
    knn = knn_analysis(X, D_func, k_values=k_values, n_perm=n_perm, seed=seed)
    out["knn"] = {"k_values": knn.get("k_values", []), "table": knn_table(knn)}
    out["_knn_raw"] = knn

    if include_curves and dx.size > 0:
        curve = distance_correlation_curve(dx, dy, n_bins=n_curve_bins)
        out["distance_curve"] = {k: v for k, v in curve.items()}
        out["_distance_curve"] = curve

    out["_condensed"] = {"representation": dx, "functional": dy}
    return out


def primary_metric_row(name: str, analysis: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the headline row for a representation, used by the ablation table."""
    primary = analysis.get("primary_mantel_spearman", {})
    partial = analysis.get("partial_mantel_controlling_for_nuisance", {})
    knn_table_rows = analysis.get("knn", {}).get("table", [])
    best_knn = min(knn_table_rows, key=lambda r: r["p_value"]) if knn_table_rows else {}
    return {
        "representation": name,
        "n_neurons": analysis.get("n_neurons"),
        "primary_metric_mantel_spearman_r": primary.get("statistic"),
        "primary_metric_p_value": primary.get("p_value"),
        "primary_metric_effect_size_z": primary.get("effect_size_z"),
        "pearson_r": analysis.get("secondary_mantel_pearson", {}).get("statistic"),
        "pearson_log_r": analysis.get("secondary_mantel_pearson_log", {}).get("statistic"),
        "partial_mantel_r_controlling_firing_rate": partial.get("statistic"),
        "partial_mantel_p_value": partial.get("p_value"),
        "best_knn_k": best_knn.get("k"),
        "best_knn_effect_size_z": best_knn.get("effect_size_z"),
        "best_knn_p_value": best_knn.get("p_value"),
    }
