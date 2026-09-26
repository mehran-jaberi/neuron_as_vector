"""The functional fingerprint: an *independent* measurement of what a neuron does.

The fingerprint is the **yardstick**, never an input. It is built from
*class-conditioned, held-out* responses and **must not** be used to construct the
proposed neuron representation (which is label-free). This module and the
``uses_labels`` metadata keep that boundary auditable.

Three fingerprint families are provided, all measured on the held-out
analysis-probe split (see config ``fingerprint.eval_split``):

**A. Class-tuning fingerprint** (``class_rate``)
    A ``C``-dimensional response profile: the mean firing rate of the neuron in
    each of the ``C`` SHD classes. This is the pre-registered primary target.

**B. Rate-normalized tuning fingerprint** (``class_rate_norm``)
    The class profile divided by the neuron's own mean rate over classes, so the
    *overall firing-rate magnitude* cannot dominate similarity: two neurons with
    the same relative tuning have distance 0 regardless of how strongly they fire.

**C. Temporal fingerprint** (``class_psth`` + ``class_temporal_center`` +
``class_temporal_dispersion`` + ``class_latency``)
    Time-resolved response information: a coarse class-conditioned PSTH, the
    temporal centre and dispersion of the response, and the first-spike timing.

Feature-set names (see :data:`FINGERPRINT_FEATURE_SETS`):

===========================  ================================================
``class_rate``               class-conditioned mean firing rate (Hz), ``C`` dims
``class_rate_norm``          the above divided by the per-neuron class-mean
``class_count``              class-conditioned mean spike count (duplicate of
                             ``class_rate`` after z-scoring; retained for
                             backwards compatibility, not used by presets)
``class_latency``            class-conditioned (censored) first-spike latency
``class_psth``               coarse class-conditioned PSTH (``C * n_bins``)
``class_temporal_center``    class-conditioned mean spike time (ms)
``class_temporal_dispersion``class-conditioned spike-time std (ms)
===========================  ================================================

Why this is a separate module
-----------------------------
Keeping the target in its own module and its own configuration makes the
separation auditable, and the ``uses_labels`` metadata flag travels with every
object so a leakage mistake is easy to detect.

The default split is the **analysis-probe** split (``probe``): it is disjoint
from training, and - unlike ``dev`` - it is not used for model selection. The
official **test** set is never used for the analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .utils import Standardizer, sanitize_features

# Feature-set names available for the fingerprint.
FINGERPRINT_FEATURE_SETS = (
    "class_rate",
    "class_rate_norm",
    "class_count",
    "class_latency",
    "class_psth",
    "class_temporal_center",
    "class_temporal_dispersion",
)

# Named presets that map the fingerprint families A / B / C onto feature sets.
FINGERPRINT_PRESETS: dict[str, tuple[str, ...]] = {
    # A. class-tuning fingerprint (the pre-registered primary target)
    "tuning": ("class_rate",),
    # B. rate-normalized tuning fingerprint (primary rate control target)
    "tuning_rate_normalized": ("class_rate_norm",),
    # C. temporal fingerprint
    "temporal": (
        "class_psth",
        "class_temporal_center",
        "class_temporal_dispersion",
        "class_latency",
    ),
    # tuning + temporal combined (reference / secondary target)
    "tuning_plus_temporal": (
        "class_rate",
        "class_psth",
        "class_temporal_center",
        "class_temporal_dispersion",
        "class_latency",
    ),
}


def resolve_feature_sets(value: Any) -> list[str]:
    """Expand a preset name or explicit list into a validated feature-set list."""
    if value is None:
        return list(FINGERPRINT_PRESETS["tuning"])
    if isinstance(value, str):
        raw = [v.strip() for v in value.replace(",", " ").split() if v.strip()]
    else:
        raw = [str(v) for v in value]
    expanded: list[str] = []
    for name in raw:
        if name in FINGERPRINT_PRESETS:
            expanded.extend(FINGERPRINT_PRESETS[name])
        else:
            expanded.append(name)
    unknown = [v for v in expanded if v not in FINGERPRINT_FEATURE_SETS]
    if unknown:
        raise ValueError(
            f"Unknown fingerprint feature set(s): {unknown}. "
            f"Valid: {list(FINGERPRINT_FEATURE_SETS)} or presets {list(FINGERPRINT_PRESETS)}"
        )
    # de-duplicate, keep canonical order
    seen: list[str] = []
    for name in FINGERPRINT_FEATURE_SETS:
        if name in expanded and name not in seen:
            seen.append(name)
    return seen


@dataclass
class FingerprintConfig:
    """Configuration of the functional fingerprint."""

    feature_sets: list[str] = field(default_factory=lambda: list(FINGERPRINT_PRESETS["tuning"]))
    standardize: str = "column"  # "column" (z-score each class feature) or "none"
    normalize_rows: bool = False  # True -> compare response *patterns* only (L2)
    metric: str = "euclidean"  # "euclidean" or "correlation"
    min_spikes_for_latency: float = 1.0
    n_psth_bins: int = 10  # coarse PSTH resolution used by ``class_psth``
    eval_split: str = "probe"  # held-out analysis-probe split (never the test set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_sets": list(self.feature_sets),
            "standardize": self.standardize,
            "normalize_rows": bool(self.normalize_rows),
            "metric": self.metric,
            "min_spikes_for_latency": float(self.min_spikes_for_latency),
            "n_psth_bins": int(self.n_psth_bins),
            "eval_split": self.eval_split,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "FingerprintConfig":
        mapping = dict(mapping or {})
        cfg = cls()
        if "feature_sets" in mapping:
            cfg.feature_sets = resolve_feature_sets(mapping["feature_sets"])
        for key in ("standardize", "metric", "eval_split"):
            if key in mapping:
                setattr(cfg, key, str(mapping[key]))
        for key in ("normalize_rows",):
            if key in mapping:
                setattr(cfg, key, bool(mapping[key]))
        for key in ("min_spikes_for_latency", "n_psth_bins"):
            if key in mapping:
                setattr(cfg, key, int(mapping[key]) if key == "n_psth_bins" else float(mapping[key]))
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
    feature_sets: Sequence[str] = FINGERPRINT_PRESETS["tuning"],
    n_psth_bins: int = 10,
    min_spikes_for_latency: float = 1.0,
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
    feature_sets:
        Any subset of :data:`FINGERPRINT_FEATURE_SETS` (or preset names).
    n_psth_bins:
        Number of coarse time bins for the ``class_psth`` feature set.
    min_spikes_for_latency:
        A class-conditioned latency is only trusted when the neuron spiked at
        least this many times in that class; otherwise it is censored at the full
        stimulus duration. (This makes the previously dead config knob live.)

    Returns
    -------
    ``(X, feature_names)`` with ``X`` finite everywhere.
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

    # B. rate-normalized tuning: remove the neuron's overall magnitude by dividing
    #    its class profile by its mean rate over classes (dimensionless profile).
    mean_over_classes = rate.mean(axis=0, keepdims=True)  # (1, H)
    rate_norm = np.divide(
        rate, np.where(mean_over_classes > 1e-12, mean_over_classes, 1.0)
    )

    # (Censored) class-conditioned first-spike latency.
    latency = np.full((C, H), duration_ms, dtype=np.float64)
    threshold = max(float(min_spikes_for_latency), 1.0)
    has = class_first_spike_count >= threshold
    denom = np.where(has, class_first_spike_count, 1.0)
    observed = class_first_spike_sum / denom
    latency[has] = observed[has]

    # C. temporal summaries from the class-conditioned PSTH.
    t_ms = np.arange(T, dtype=np.float64) * float(bin_ms)
    total = class_psth.sum(axis=2, keepdims=True)  # (C, H, 1)
    active = total[..., 0] > 0
    p_time = np.divide(class_psth, np.where(total > 0, total, 1.0))
    t_center = (p_time * t_ms).sum(axis=2)  # (C, H)
    t_disp = np.sqrt(np.maximum((p_time * (t_ms - t_center[..., None]) ** 2).sum(axis=2), 0.0))
    t_center = np.where(active, t_center, 0.0)
    t_disp = np.where(active, t_disp, 0.0)

    # Coarse class-conditioned PSTH (mean rate in Hz per coarse bin).
    n_psth_bins = max(int(n_psth_bins), 1)
    edges = np.linspace(0, T, n_psth_bins + 1).round().astype(int)
    blocks: list[np.ndarray] = []
    for b in range(n_psth_bins):
        lo = int(edges[b])
        hi = int(max(edges[b + 1], lo + 1))
        block = class_psth[:, :, lo:hi].sum(axis=2)  # (C, H)
        block_dur_s = (hi - lo) * float(bin_ms) / 1000.0
        blocks.append(block / (n_per_class * max(block_dur_s, 1e-9)))
    coarse = np.stack(blocks, axis=2)  # (C, H, n_psth_bins)

    selected = resolve_feature_sets(feature_sets)
    parts: list[np.ndarray] = []
    names: list[str] = []
    for fs in selected:
        if fs == "class_rate":
            parts.append(rate.T)
            names += [f"fp_rate.class{c}" for c in range(C)]
        elif fs == "class_rate_norm":
            parts.append(rate_norm.T)
            names += [f"fp_rate_norm.class{c}" for c in range(C)]
        elif fs == "class_count":
            parts.append(counts.T)
            names += [f"fp_count.class{c}" for c in range(C)]
        elif fs == "class_latency":
            parts.append(latency.T)
            names += [f"fp_latency.class{c}" for c in range(C)]
        elif fs == "class_psth":
            parts.append(coarse.transpose(1, 0, 2).reshape(H, C * n_psth_bins))
            names += [f"fp_psth.class{c}.bin{b}" for c in range(C) for b in range(n_psth_bins)]
        elif fs == "class_temporal_center":
            parts.append(t_center.T)
            names += [f"fp_tcenter.class{c}" for c in range(C)]
        elif fs == "class_temporal_dispersion":
            parts.append(t_disp.T)
            names += [f"fp_tdispersion.class{c}" for c in range(C)]
        else:  # pragma: no cover - resolve_feature_sets validates
            raise ValueError(f"Unknown fingerprint feature set: {fs}")

    X = np.concatenate(parts, axis=1) if parts else np.zeros((H, 0))
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
