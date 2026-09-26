"""Geometry-function analysis: does neuron-space geometry predict function?

Primary hypothesis
------------------
Neurons that are close in the proposed representation space should tend to have
similar functional fingerprints, i.e. the *distance matrices* of the two spaces
should be positively related.

Primary metric (PRIMARY ANALYSIS, reported everywhere)
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
* A **rate-matched** stratified Mantel control (pairs are compared only within
  narrow firing-rate-difference strata).
* A **partial** Mantel controlling for a nuisance variable - retained **only as a
  secondary, exploratory statistic**. A partial Mantel is *not* proof that the
  effect is independent of firing rate (see the warning printed in its output).

All permutation tests use the same vectorised relabelling scheme: a permutation
``p`` maps the functional matrix ``D`` to ``D[p][:, p]``, which is exactly the
null hypothesis "the representation-neuron pairing is arbitrary".

Neuron pairs are **not** independent samples: the permutation acts on *neurons*,
not on pairs, and every reported p-value is accompanied by the resolution floor
``1 / (n_perm + 1)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _pair_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n, 1)


def _pair_index_matrix(n: int) -> np.ndarray:
    """``(n, n)`` matrix mapping (i, j) -> condensed-vector index.

    Lets a neuron permutation be applied to a condensed vector by pure indexing:
    if ``dy_p = D[p[ii], p[jj]]`` then ``dy_p = dy[pair_index[p[ii], p[jj]]]``.
    This removes the per-permutation ``rankdata`` from the null loop (a large
    speed-up) while keeping the statistic numerically identical.
    """
    ii, jj = _pair_indices(n)
    mat = np.zeros((n, n), dtype=np.int64)
    mat[ii, jj] = np.arange(ii.size)
    mat[jj, ii] = np.arange(ii.size)
    return mat


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


def statistic_from_vectors(a: np.ndarray, b: np.ndarray, method: str = "spearman") -> float:
    """Mantel-style correlation between two condensed distance vectors.

    ``spearman`` ranks both vectors (the primary metric); ``pearson`` and
    ``pearson_log`` are secondary. Ranks are computed here so that bootstrap and
    stratified controls reuse exactly the same definition as the primary test.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3:
        return float("nan")
    if method == "spearman":
        return _spearman_from_ranks(rankdata(a), rankdata(b))
    if method == "pearson":
        return _pearson(a, b)
    if method == "pearson_log":
        return _pearson(np.log1p(a), np.log1p(b))
    raise ValueError(f"Unknown Mantel method {method!r}")


@dataclass
class MantelResult:
    """Outcome of a Mantel-style permutation test.

    ``p_value`` is one-sided for the configured alternative and is bounded below
    by ``p_value_floor = 1 / (n_perm + 1)``; ``at_resolution_floor`` flags that
    the observed effect saturated that floor (so the true p is only known to be
    *smaller* than reported). ``null`` holds the full permutation null when
    requested (excluded from :meth:`to_dict` to keep JSON small; save it to NPZ).
    """

    statistic: float
    p_value: float
    null_mean: float
    null_std: float
    n_perm: int
    method: str
    alternative: str
    effect_size: float
    n_neurons: int
    p_value_floor: float = float("nan")
    at_resolution_floor: bool = False
    null_quantiles: dict[str, float] = field(default_factory=dict)
    bootstrap_ci_low: float = float("nan")
    bootstrap_ci_high: float = float("nan")
    bootstrap_ci_level: float = 0.95
    null: np.ndarray | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "statistic": self.statistic,
            "p_value": self.p_value,
            "p_value_floor": self.p_value_floor,
            "at_resolution_floor": bool(self.at_resolution_floor),
            "n_perm": self.n_perm,
            "method": self.method,
            "alternative": self.alternative,
            "effect_size_z": self.effect_size,
            "null_mean": self.null_mean,
            "null_std": self.null_std,
            "null_quantiles": dict(self.null_quantiles),
            "null_distribution_included": self.null is not None,
            "n_neurons": self.n_neurons,
            "n_pairs": int(self.n_neurons * (self.n_neurons - 1) / 2),
            "bootstrap_ci": {
                "low": self.bootstrap_ci_low,
                "high": self.bootstrap_ci_high,
                "level": self.bootstrap_ci_level,
            },
            "p_value_interpretation": (
                f"p is bounded below by 1/(n_perm+1) = {self.p_value_floor:.2e}. "
                + (
                    "The observed statistic saturates this floor: report as "
                    f"p < {self.p_value_floor:.1e}, not as the exact value."
                    if self.at_resolution_floor
                    else "The observed statistic does not saturate the floor."
                )
            ),
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
    return_null: bool = False,
    bootstrap: int = 0,
) -> MantelResult:
    """Permutation Mantel test between two condensed distance vectors.

    ``dx`` is held fixed; ``dy`` is relabelled by a random permutation of the
    neurons, which is the standard Mantel null. The permutation acts on *neurons*
    (not on pairs), so neuron pairs are never treated as independent samples.
    ``alternative="greater"`` tests the primary hypothesis that the two distance
    matrices are positively related. ``return_null`` keeps the full null vector.
    """
    dx = np.asarray(dx, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)
    if dx.shape != dy.shape:
        raise ValueError(f"Distance vectors must have the same length, got {dx.shape} and {dy.shape}")
    m = dx.size
    if m < 3:
        return MantelResult(
            float("nan"), float("nan"), float("nan"), float("nan"), n_perm, method,
            alternative, float("nan"), 0,
        )
    n = n_neurons if n_neurons is not None else int(round((1 + np.sqrt(1 + 8 * m)) / 2))
    ii, jj = _pair_indices(n)
    if ii.size != m:
        raise ValueError("Could not infer the number of neurons from the distance vector length")

    observed = statistic_from_vectors(dx, dy, method)

    # Fixed vector `a` and permutable base `b`. For "spearman" we rank once and
    # reuse the rank vector under permutation, because permuting `dy` preserves
    # ties, so rankdata(dy[perm_pair]) == rankdata(dy)[perm_pair].
    if method == "spearman":
        a = rankdata(dx)
        base = rankdata(dy)
    elif method == "pearson":
        a = dx
        base = dy
    elif method == "pearson_log":
        a = np.log1p(dx)
        base = np.log1p(dy)
    else:
        raise ValueError(f"Unknown Mantel method {method!r}")

    a_c = a - a.mean()
    b_c = base - base.mean()
    denom = float(m) * float(a.std()) * float(base.std())
    pair_index = _pair_index_matrix(n)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    for k in range(n_perm):
        p = rng.permutation(n)
        perm_pair = pair_index[p[ii], p[jj]]
        if denom > 1e-12:
            null[k] = float((a_c * b_c[perm_pair]).sum() / denom)
        else:
            null[k] = float("nan")

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

    p_value_floor = 1.0 / (1.0 + n_perm)
    at_floor = bool(np.isfinite(p_value) and p_value <= p_value_floor + 1e-12)
    quantiles: dict[str, float] = {}
    if np.isfinite(null).any():
        for q in (0.5, 0.9, 0.95, 0.99, 0.999):
            quantiles[f"q{q:g}"] = float(np.nanquantile(null, q))

    ci_low = ci_high = float("nan")
    if bootstrap and bootstrap > 0:
        ci_low, ci_high, _ = bootstrap_mantel_ci(
            dx, dy, n_boot=int(bootstrap), seed=seed, method=method, n_neurons=n
        )

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
        p_value_floor=float(p_value_floor),
        at_resolution_floor=at_floor,
        null_quantiles=quantiles,
        bootstrap_ci_low=float(ci_low),
        bootstrap_ci_high=float(ci_high),
        null=null if return_null else None,
    )


def bootstrap_mantel_ci(
    dx: np.ndarray,
    dy: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    method: str = "spearman",
    level: float = 0.95,
    n_neurons: int | None = None,
) -> tuple[float, float, np.ndarray]:
    """Approximate non-parametric CI for the Mantel statistic by resampling neurons.

    Neurons (not pairs) are resampled with replacement, so the dependence between
    pairs is preserved. Resampling with replacement can create duplicated neurons
    (hence zero distance pairs); this is a documented limitation of the naive
    neuron bootstrap and the interval should be read as approximate.

    Returns ``(low, high, bootstrap_samples)``.
    """
    dx = np.asarray(dx, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)
    m = dx.size
    n = n_neurons if n_neurons is not None else int(round((1 + np.sqrt(1 + 8 * m)) / 2))
    ii, jj = _pair_indices(n)
    if ii.size != m:
        raise ValueError("Could not infer the number of neurons from the distance vector length")
    Dx = condensed_to_matrix(dx, n)
    Dy = condensed_to_matrix(dy, n)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[b] = statistic_from_vectors(Dx[idx[ii], idx[jj]], Dy[idx[ii], idx[jj]], method)
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return float("nan"), float("nan"), samples
    lo = float(np.quantile(finite, (1.0 - level) / 2.0))
    hi = float(np.quantile(finite, 1.0 - (1.0 - level) / 2.0))
    return lo, hi, samples


def rate_matched_mantel(
    dx: np.ndarray,
    dy: np.ndarray,
    rate_absdiff: np.ndarray,
    *,
    n_strata: int = 5,
    n_perm: int = 1000,
    seed: int = 0,
    method: str = "spearman",
    n_neurons: int | None = None,
) -> MantelResult:
    """Rate-matched (stratified) Mantel control.

    Neuron pairs are binned into ``n_strata`` quantile strata of the absolute
    firing-rate difference ``|rate_i - rate_j|``. The Mantel statistic is computed
    **within** each stratum (so only neurons with similar firing rates are
    compared) and combined with a sample-size-weighted Fisher z average. The
    permutation null relabels neurons and recomputes the same stratified statistic,
    keeping the rate strata fixed.

    This is a *proper rate-matched control*: unlike a partial Mantel, it does not
    rely on a linear nuisance model and it directly removes the between-rate
    component of the comparison.
    """
    dx = np.asarray(dx, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)
    rate_absdiff = np.asarray(rate_absdiff, dtype=np.float64)
    m = dx.size
    n = n_neurons if n_neurons is not None else int(round((1 + np.sqrt(1 + 8 * m)) / 2))
    ii, jj = _pair_indices(n)
    if rate_absdiff.shape != (m,):
        raise ValueError("rate_absdiff must be a condensed vector matching dx")

    edges = np.unique(np.quantile(rate_absdiff, np.linspace(0, 1, n_strata + 1)))
    if edges.size < 3:
        # Degenerate: cannot stratify -> fall back to the plain Mantel test.
        res = mantel_test(dx, dy, method=method, n_perm=n_perm, seed=seed, n_neurons=n)
        res.method = "rate_matched__degenerate_fallback"
        return res

    stratum = np.clip(np.digitize(rate_absdiff, edges[1:-1], right=False), 0, edges.size - 2)
    masks = [stratum == s for s in range(edges.size - 1)]
    weights = np.array([float(mask.sum()) for mask in masks], dtype=np.float64)

    def stratified(dy_vec: np.ndarray) -> float:
        num = 0.0
        den = 0.0
        for mask, w in zip(masks, weights):
            if w < 3:
                continue
            r = statistic_from_vectors(dx[mask], dy_vec[mask], method)
            if not np.isfinite(r):
                continue
            r = float(np.clip(r, -0.999999, 0.999999))
            num += w * np.arctanh(r)
            den += w
        return float(np.tanh(num / den)) if den > 0 else float("nan")

    observed = stratified(dy)

    pair_index = _pair_index_matrix(n)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    for k in range(n_perm):
        p = rng.permutation(n)
        null[k] = stratified(dy[pair_index[p[ii], p[jj]]])

    null_mean = float(np.nanmean(null)) if np.isfinite(null).any() else float("nan")
    null_std = float(np.nanstd(null)) if np.isfinite(null).any() else float("nan")
    p_value = float((1 + np.sum(null >= observed)) / (1 + n_perm))
    effect = float((observed - null_mean) / null_std) if null_std > 1e-12 else float("nan")
    p_value_floor = 1.0 / (1.0 + n_perm)
    quantiles = {
        f"q{q:g}": float(np.nanquantile(null, q)) for q in (0.5, 0.9, 0.95, 0.99)
    } if np.isfinite(null).any() else {}
    return MantelResult(
        statistic=float(observed),
        p_value=p_value,
        null_mean=null_mean,
        null_std=null_std,
        n_perm=int(n_perm),
        method="rate_matched_stratified",
        alternative="greater",
        effect_size=effect,
        n_neurons=int(n),
        p_value_floor=float(p_value_floor),
        at_resolution_floor=bool(np.isfinite(p_value) and p_value <= p_value_floor + 1e-12),
        null_quantiles=quantiles,
        null=null,
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

    ry_base = rankdata(dy)
    rz = rankdata(dz)
    pair_index = _pair_index_matrix(n)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    for k in range(n_perm):
        p = rng.permutation(n)
        dy_p_ranks = ry_base[pair_index[p[ii], p[jj]]]
        ry_p = _residualize(dy_p_ranks, rz)
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

    A positive ``effect_size_z`` means the observed mean functional distance to a
    neuron's representation-space neighbours is *smaller* than the permutation
    null, i.e. neighbours are functionally more similar than chance - the
    hypothesis of interest. In symbols ``effect_size = (null_mean - observed) /
    null_std`` and ``p_value`` is one-sided for ``observed < null``.
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
    rate_absdiff_condensed: np.ndarray | None = None,
    rate_matched_strata: int = 5,
    primary_bootstrap: int = 0,
    primary_return_null: bool = False,
    include_curves: bool = True,
) -> dict[str, Any]:
    """Run the full primary + supporting analysis for one (representation, fingerprint) pair.

    ``nuisance_condensed`` optionally supplies a distance vector to control for
    (typically the absolute difference in log firing rate), enabling the **partial**
    Mantel control - which is secondary/exploratory only.

    ``rate_absdiff_condensed`` optionally supplies ``|rate_i - rate_j|`` to enable
    the **rate-matched stratified Mantel**, which is the proper rate control.
    """
    from scipy.spatial.distance import pdist

    X = np.asarray(space_matrix, dtype=np.float64)
    Y = np.asarray(fingerprint_matrix, dtype=np.float64)
    n = X.shape[0]

    dx = pdist(X, metric=representation_metric) if n > 1 else np.zeros(0)
    dy = pdist(Y, metric=fingerprint_metric) if n > 1 else np.zeros(0)

    out: dict[str, Any] = {"n_neurons": int(n), "n_pairs": int(dx.size)}

    primary = mantel_test(
        dx, dy, method="spearman", n_perm=n_perm, seed=seed, alternative="greater",
        return_null=primary_return_null, bootstrap=primary_bootstrap,
    )
    out["primary_mantel_spearman"] = primary.to_dict()
    out["_primary_null"] = primary.null
    out["secondary_mantel_pearson"] = mantel_test(
        dx, dy, method="pearson", n_perm=n_perm, seed=seed, alternative="greater"
    ).to_dict()
    out["secondary_mantel_pearson_log"] = mantel_test(
        dx, dy, method="pearson_log", n_perm=n_perm, seed=seed, alternative="greater"
    ).to_dict()

    if rate_absdiff_condensed is not None and np.asarray(rate_absdiff_condensed).size == dx.size:
        out["rate_matched_mantel"] = rate_matched_mantel(
            dx, dy, np.asarray(rate_absdiff_condensed, dtype=np.float64),
            n_strata=int(rate_matched_strata), n_perm=n_perm, seed=seed,
        ).to_dict()

    if nuisance_condensed is not None and np.asarray(nuisance_condensed).size == dx.size:
        partial = partial_mantel_test(
            dx, dy, np.asarray(nuisance_condensed, dtype=np.float64), n_perm=n_perm, seed=seed
        )
        partial_dict = partial.to_dict()
        partial_dict["status"] = "secondary_exploratory"
        partial_dict["warning"] = (
            "A partial Mantel controlling for firing rate is a SECONDARY, exploratory "
            "statistic and is NOT proof that the effect is independent of firing rate. "
            "Use the rate-normalized fingerprint and/or the rate-matched stratified "
            "Mantel for the primary rate control."
        )
        out["partial_mantel_controlling_for_nuisance"] = partial_dict

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
    """Extract the headline row for a representation, used by the ablation table.

    NOTE: the partial Mantel columns are **secondary/exploratory only** and must
    not be reported as proof that the effect is independent of firing rate. The
    proper rate control is the rate-normalized fingerprint and/or the rate-matched
    stratified Mantel (see :func:`rate_matched_mantel`).
    """
    primary = analysis.get("primary_mantel_spearman", {})
    partial = analysis.get("partial_mantel_controlling_for_nuisance", {})
    rate_matched = analysis.get("rate_matched_mantel", {})
    knn_table_rows = analysis.get("knn", {}).get("table", [])
    best_knn = min(knn_table_rows, key=lambda r: r["p_value"]) if knn_table_rows else {}
    return {
        "representation": name,
        "n_neurons": analysis.get("n_neurons"),
        "primary_metric_mantel_spearman_r": primary.get("statistic"),
        "primary_metric_p_value": primary.get("p_value"),
        "primary_metric_p_value_floor": primary.get("p_value_floor"),
        "primary_metric_at_resolution_floor": primary.get("at_resolution_floor"),
        "primary_metric_effect_size_z": primary.get("effect_size_z"),
        "primary_metric_ci_low": (primary.get("bootstrap_ci") or {}).get("low"),
        "primary_metric_ci_high": (primary.get("bootstrap_ci") or {}).get("high"),
        "pearson_r": analysis.get("secondary_mantel_pearson", {}).get("statistic"),
        "pearson_log_r": analysis.get("secondary_mantel_pearson_log", {}).get("statistic"),
        "rate_matched_mantel_r": rate_matched.get("statistic"),
        "rate_matched_mantel_p_value": rate_matched.get("p_value"),
        "rate_matched_mantel_effect_size_z": rate_matched.get("effect_size_z"),
        # secondary / exploratory only:
        "partial_mantel_r_controlling_firing_rate": partial.get("statistic"),
        "partial_mantel_p_value": partial.get("p_value"),
        "partial_mantel_status": "secondary_exploratory__not_a_proof_of_rate_independence",
        "best_knn_k": best_knn.get("k"),
        "best_knn_effect_size_z": best_knn.get("effect_size_z"),
        "best_knn_p_value": best_knn.get("p_value"),
    }
