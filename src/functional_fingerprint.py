"""The functional fingerprint: an *independent* target for the neuron representation.

For every hidden neuron we measure how it responds to held-out stimuli,
class-by-class. Concretely, with ``C`` classes the fingerprint is a
``3C``-dimensional vector

* class-conditioned mean firing rate (Hz),
* class-conditioned mean spike count per sample,
* class-conditioned mean first-spike latency (ms, censored at stimulus end).

Why this is a separate module
----------------------------
The fingerprint is **not** the proposed neuron representation, and it must never
be fed back into it. It is the yardstick against which the representation is
evaluated: the primary hypothesis is that neurons close in representation space
have *similar fingerprints*. Keeping the target in its own module and its own
configuration makes the separation auditable, and the ``uses_labels`` metadata
flag travels with every object so a leakage mistake is easy to detect.

The fingerprint is measured on data the representation never saw: by default the
validation split, optionally the official test set (which must remain untouched
for all modelling decisions, see README).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .utils import Standardizer, sanitize_features

# Feature-set names available for the fingerprint.
FINGERPRINT_FEATURE_SETS = ("class_rate", "class_count", "class_latency")


@dataclass
class FingerprintConfig:
    """Configuration of the functional fingerprint."""

    feature_sets: list[str] = field(default_factory=lambda: list(FINGERPRINT_FEATURE_SETS))
    standardize: str = "column"  # "column" (z-score each class feature) or "none"
    normalize_rows: bool = False  # True -> compare response *patterns* only
    metric: str = "euclidean"  # "euclidean" or "correlation"
    min_spikes_for_latency: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_sets": list(self.feature_sets),
            "standardize": self.standardize,
            "normalize_rows": bool(self.normalize_rows),
            "metric": self.metric,
            "min_spikes_for_latency": float(self.min_spikes_for_latency),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "FingerprintConfig":
        mapping = dict(mapping or {})
        cfg = cls()
        if "feature_sets" in mapping:
            value = mapping["feature_sets"]
            if isinstance(value, str):
                value = [v.strip() for v in value.replace(",", " ").split() if v.strip()]
            unknown = [v for v in value if v not in FINGERPRINT_FEATURE_SETS]
            if unknown:
                raise ValueError(f"Unknown fingerprint feature set(s): {unknown}")
            cfg.feature_sets = list(value)
        for key in ("standardize", "metric"):
            if key in mapping:
                setattr(cfg, key, str(mapping[key]))
        for key in ("normalize_rows",):
            if key in mapping:
                setattr(cfg, key, bool(mapping[key]))
        if "min_spikes_for_latency" in mapping:
            cfg.min_spikes_for_latency = float(mapping["min_spikes_for_latency"])
        return cfg


# --------------------------------------------------------------------------
# Building the fingerprint matrix from class-conditioned statistics
# --------------------------------------------------------------------------
def class_conditioned_fingerprint(
    class_psth: np.ndarray,
    class_counts: np.ndarray,
    class_n: np.ndarray,
    class_first_spike_sum: np.ndarray,
    class_first_spike_count: np.ndarray,
    *,
    n_bins: int,
    bin_ms: float,
    feature_sets: Sequence[str] = FINGERPRINT_FEATURE_SETS,
) -> tuple[np.ndarray, list[str]]:
    """Assemble the ``(n_hidden, n_features)`` fingerprint matrix.

    Parameters
    ----------
    class_psth:
        ``(C, n_hidden, n_bins)`` spike-time histogram summed over the samples of
        each class.
    class_counts:
        ``(C, n_hidden)`` total spike count per class.
    class_n:
        ``(C,)`` number of samples per class.
    class_first_spike_sum, class_first_spike_count:
        ``(C, n_hidden)`` sums/counts used for the (censored) first-spike latency.

    Returns
    -------
    ``(X, feature_names)`` with ``X`` finite everywhere.

    Censoring convention: within a class, if a neuron never spiked for any of
    that class's samples its latency is set to the full stimulus duration. This
    is an explicit, documented choice that avoids NaNs; it is monotone in the
    sense that "no response" maps to the latest possible latency.
    """
    class_psth = np.asarray(class_psth, dtype=np.float64)
    class_counts = np.asarray(class_counts, dtype=np.float64)
    class_n = np.asarray(class_n, dtype=np.float64).reshape(-1)
    class_first_spike_sum = np.asarray(class_first_spike_sum, dtype=np.float64)
    class_first_spike_count = np.asarray(class_first_spike_count, dtype=np.float64)

    C, H, T = class_psth.shape
    duration_s = T * bin_ms / 1000.0
    duration_ms = T * float(bin_ms)
    n_per_class = np.where(class_n > 0, class_n, 1.0).reshape(-1, 1)

    rate = class_psth.sum(axis=2) / (n_per_class * max(duration_s, 1e-9))  # (C, H) Hz
    counts = class_counts / n_per_class  # (C, H)

    latency = np.full((C, H), duration_ms, dtype=np.float64)
    has = class_first_spike_count > 0
    denom = np.where(has, class_first_spike_count, 1.0)
    observed = class_first_spike_sum / denom
    latency[has] = observed[has]

    parts: list[np.ndarray] = []
    names: list[str] = []
    for fs in feature_sets:
        if fs == "class_rate":
            parts.append(rate)
            names += [f"fp_rate.class{c}" for c in range(C)]
        elif fs == "class_count":
            parts.append(counts)
            names += [f"fp_count.class{c}" for c in range(C)]
        elif fs == "class_latency":
            parts.append(latency)
            names += [f"fp_latency.class{c}" for c in range(C)]
        else:
            raise ValueError(f"Unknown fingerprint feature set: {fs}")

    X = np.concatenate([p.T for p in parts], axis=1) if parts else np.zeros((H, 0))
    return sanitize_features(X), names


def class_conditioned_rate_matrix(
    class_psth: np.ndarray, class_n: np.ndarray, *, bin_ms: float
) -> np.ndarray:
    """``(C, n_hidden)`` mean firing-rate matrix - used for the fingerprint heat maps."""
    class_psth = np.asarray(class_psth, dtype=np.float64)
    C, H, T = class_psth.shape
    duration_s = T * bin_ms / 1000.0
    n_per_class = np.where(np.asarray(class_n, dtype=np.float64) > 0, np.asarray(class_n, dtype=np.float64), 1.0)
    return class_psth.sum(axis=2) / (n_per_class.reshape(-1, 1) * max(duration_s, 1e-9))


# --------------------------------------------------------------------------
# Fingerprint space (the independent target)
# --------------------------------------------------------------------------
@dataclass
class FingerprintSpace:
    """The class-conditioned response profiles of all hidden neurons, as a metric space."""

    X_raw: np.ndarray
    feature_names: list[str]
    config: FingerprintConfig = field(default_factory=FingerprintConfig)
    meta: dict[str, Any] = field(default_factory=dict)

    X: np.ndarray = field(init=False, repr=False, default=None)  # type: ignore[assignment]
    standardizer: Standardizer = field(init=False, repr=False, default_factory=Standardizer)

    def __post_init__(self) -> None:
        self.X_raw = sanitize_features(np.asarray(self.X_raw, dtype=np.float64))
        if self.X_raw.ndim != 2:
            raise ValueError(f"Fingerprint matrix must be 2-D, got {self.X_raw.shape}")
        self.feature_names = list(self.feature_names)
        self._fit()

    def _fit(self) -> None:
        if self.config.standardize == "column":
            self.standardizer.fit(self.X_raw)
            Z = self.standardizer.transform(self.X_raw)
        elif self.config.standardize == "none":
            self.standardizer.fit(self.X_raw)  # still records constants
            Z = self.X_raw.copy()
        else:
            raise ValueError(f"Unknown standardize mode {self.config.standardize!r}")
        if self.config.normalize_rows:
            norms = np.linalg.norm(Z, axis=1, keepdims=True)
            Z = np.divide(Z, np.where(norms > 1e-12, norms, 1.0))
        self.X = sanitize_features(Z)
        self.meta.setdefault("uses_labels", True)
        self.meta.setdefault("n_neurons", int(self.X.shape[0]))
        self.meta.setdefault(
            "leakage_warning",
            "This object uses class labels by construction. It is the evaluation target, "
            "never an input to the neuron representation.",
        )

    @property
    def n_neurons(self) -> int:
        return int(self.X.shape[0])

    def condensed(self, metric: str | None = None) -> np.ndarray:
        from scipy.spatial.distance import pdist

        metric = metric or self.config.metric
        if self.n_neurons < 2:
            return np.zeros(0, dtype=np.float64)
        return pdist(self.X, metric=metric)

    def distances(self, metric: str | None = None) -> np.ndarray:
        from scipy.spatial.distance import squareform

        cond = self.condensed(metric)
        if cond.size == 0:
            return np.zeros((self.n_neurons, self.n_neurons))
        return squareform(cond)

    def save(self, path: str) -> None:
        from .utils import save_json

        save_json(
            {
                "config": self.config.to_dict(),
                "feature_names": self.feature_names,
                "X_raw": self.X_raw.tolist(),
                "X": self.X.tolist(),
                "meta": self.meta,
            },
            path,
        )


# --------------------------------------------------------------------------
# Reliability / noise ceiling
# --------------------------------------------------------------------------
def split_half_reliability(
    fingerprints_a: FingerprintSpace,
    fingerprints_b: FingerprintSpace,
) -> dict[str, Any]:
    """Estimate the noise ceiling of the fingerprint.

    Two independent halves of the held-out data each yield a fingerprint. If the
    fingerprints themselves are unreliable, no representation could ever predict
    them, so we report

    * ``matrix_reliability``: Spearman correlation between the two fingerprint
      distance matrices (the Mantel reliability of the target),
    * ``vector_reliability``: mean per-neuron Pearson correlation between the two
      fingerprint vectors,
    * ``attenuation_factor``: ``sqrt(matrix_reliability)``, used to correct an
      observed representation->fingerprint correlation for target noise
      (:math:`r_{true} \\approx r_{obs} / \\sqrt{r_{ceiling}}`).

    This is a *diagnostic*, not a result: it bounds how strong a correlation can
    possibly be.
    """
    from scipy.stats import pearsonr, spearmanr

    da = fingerprints_a.condensed()
    db = fingerprints_b.condensed()
    matrix_r = float(spearmanr(da, db).statistic) if da.size > 2 else float("nan")

    a = fingerprints_a.X_raw
    b = fingerprints_b.X_raw
    per_neuron: list[float] = []
    for i in range(a.shape[0]):
        va, vb = a[i], b[i]
        if np.std(va) < 1e-12 or np.std(vb) < 1e-12:
            continue
        per_neuron.append(float(pearsonr(va, vb).statistic))
    vector_r = float(np.mean(per_neuron)) if per_neuron else float("nan")
    ceiling = max(matrix_r, 0.0)
    return {
        "matrix_reliability_spearman": matrix_r,
        "vector_reliability_pearson": vector_r,
        "n_neurons_with_variance": len(per_neuron),
        "attenuation_factor_sqrt_ceiling": float(np.sqrt(ceiling)),
        "interpretation": (
            "Upper bound on any achievable distance-distance correlation. Reported for "
            "transparency; a low ceiling means the target itself is noisy."
        ),
    }


def attenuation_correct(r_observed: float, reliability: float) -> float:
    """Correct an observed correlation for measurement noise in the target."""
    if reliability is None or not np.isfinite(reliability) or reliability <= 1e-6:
        return float("nan")
    return float(r_observed / np.sqrt(reliability))
