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
    # tuning + latency (secondary target)
    "tuning_with_latency": ("class_rate", "class_latency"),
    # tuning + temporal combined (reference / secondary target)
    "tuning_plus_temporal": (
        "class_rate",
        "class_psth",
        "class_temporal_center",
        "class_temporal_dispersion",
        "class_latency",
    ),
}

# --------------------------------------------------------------------------
# The ONE primary functional target (used by every script)
# --------------------------------------------------------------------------
#: Preset name of the pre-registered primary fingerprint:
#: the 20-dimensional class-conditioned firing-rate profile.
PRIMARY_FINGERPRINT_PRESET = "tuning"
#: Secondary target (adds first-spike latency).
SECONDARY_FINGERPRINT_PRESET = "tuning_with_latency"
#: Exploratory target (time-resolved responses).
EXPLORATORY_FINGERPRINT_PRESET = "temporal"

#: Canonical preprocessing/distance settings for the primary metric. Every script
#: must use exactly these (plus the same neurons and the same probe split).
PRIMARY_FINGERPRINT_SETTINGS: dict[str, Any] = {
    "standardize": "column",
    "normalize_rows": False,
    "metric": "euclidean",
    "n_psth_bins": 10,
    "min_spikes_for_latency": 1.0,
    "eval_split": "probe",
}


def preset_fingerprint_config(preset: str, **overrides: Any) -> "FingerprintConfig":
    """Canonical fingerprint configuration for a named preset.

    Uses the shared preprocessing/distance settings unless overridden, so a script
    cannot silently change the metric or the normalisation.
    """
    settings = dict(PRIMARY_FINGERPRINT_SETTINGS)
    settings.update(overrides)
    return FingerprintConfig(feature_sets=resolve_feature_sets(preset), **settings)


def primary_fingerprint_config(**overrides: Any) -> "FingerprintConfig":
    """The canonical primary fingerprint configuration (do not vary it per script)."""
    return preset_fingerprint_config(PRIMARY_FINGERPRINT_PRESET, **overrides)


def fingerprint_definition(config: "FingerprintConfig", *, dimension: int | None = None) -> dict[str, Any]:
    """The complete, unambiguous definition of a fingerprint, for result artifacts."""
    return {
        "feature_sets": list(config.feature_sets),
        "dimension": dimension,
        "standardize": config.standardize,
        "normalize_rows": bool(config.normalize_rows),
        "distance_metric": config.metric,
        "n_psth_bins": int(config.n_psth_bins),
        "min_spikes_for_latency": float(config.min_spikes_for_latency),
        "eval_split": config.eval_split,
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
def _aligned_raw(
    fingerprints_a: "FingerprintSpace",
    fingerprints_b: "FingerprintSpace",
) -> tuple[np.ndarray, np.ndarray, list[str], bool]:
    """Align two half-fingerprints by feature name (they should already match)."""
    names_a = list(fingerprints_a.feature_names)
    names_b = list(fingerprints_b.feature_names)
    aligned = names_a == names_b
    if aligned:
        return fingerprints_a.X_raw, fingerprints_b.X_raw, names_a, True
    common = [n for n in names_a if n in set(names_b)]
    if not common:
        raise ValueError("The two half-fingerprints share no feature names")
    ia = [names_a.index(n) for n in common]
    ib = [names_b.index(n) for n in common]
    return fingerprints_a.X_raw[:, ia], fingerprints_b.X_raw[:, ib], common, False


def split_half_reliability(
    fingerprints_a: "FingerprintSpace",
    fingerprints_b: "FingerprintSpace",
    *,
    common_standardizer: Any | None = None,
    metric: str | None = None,
) -> dict[str, Any]:
    """Estimate the noise ceiling of the fingerprint from two independent halves.

    Audit-critical properties (all reported explicitly so they can be checked):

    * the two halves are separate measurements of the *same neurons* (they must
      contain the same number of neurons and the same feature names);
    * both halves are reduced to distances with the **same metric** - if the two
      spaces disagree, the call raises rather than silently mixing metrics;
    * optionally a **common** :class:`~src.utils.Standardizer` is used so that the
      two halves are scaled identically (otherwise each half standardises itself,
      which conflates measurement noise with scaling differences);
    * the reported ceiling is the **Spearman-Brown corrected full-length**
      reliability ``2r / (1 + r)`` of the half-half distance correlation, and the
      attenuation factor is ``sqrt(full)``. The half-length attenuation
      ``sqrt(r_half)`` is reported separately and is more conservative.

    This is a *diagnostic*, not a result: it bounds how strong a
    representation-function correlation can possibly be.
    """
    from scipy.spatial.distance import pdist
    from scipy.stats import pearsonr, spearmanr

    if fingerprints_a.n_neurons != fingerprints_b.n_neurons:
        raise ValueError(
            "The two half-fingerprints must describe the same neurons; got "
            f"{fingerprints_a.n_neurons} and {fingerprints_b.n_neurons}"
        )

    metric_a = fingerprints_a.config.metric
    metric_b = fingerprints_b.config.metric
    metrics_match = metric_a == metric_b
    if metric is None:
        if not metrics_match:
            raise ValueError(
                f"Half-fingerprints use different metrics ({metric_a!r} vs {metric_b!r}); "
                "pass an explicit `metric` to override."
            )
        metric = metric_a

    warnings: list[str] = []
    if not metrics_match:
        warnings.append(
            f"half fingerprints used different metrics ({metric_a!r} vs {metric_b!r}); "
            f"reliability was computed with the explicit metric {metric!r}"
        )

    raw_a, raw_b, names, aligned = _aligned_raw(fingerprints_a, fingerprints_b)
    if not aligned:
        warnings.append("half-fingerprint feature names differed; aligned on the intersection")

    def _prepare(raw: np.ndarray) -> np.ndarray:
        if common_standardizer is None:
            return raw
        return sanitize_features(common_standardizer.transform(raw))

    Za = _prepare(raw_a)
    Zb = _prepare(raw_b)
    da = pdist(Za, metric=metric) if Za.shape[0] > 1 else np.zeros(0)
    db = pdist(Zb, metric=metric) if Zb.shape[0] > 1 else np.zeros(0)
    matrix_r = float(spearmanr(da, db).statistic) if da.size > 2 and da.std() > 1e-12 and db.std() > 1e-12 else float("nan")

    per_neuron: list[float] = []
    for i in range(Za.shape[0]):
        va, vb = Za[i], Zb[i]
        if np.std(va) < 1e-12 or np.std(vb) < 1e-12:
            continue
        r = float(pearsonr(va, vb).statistic)
        if np.isfinite(r):
            per_neuron.append(r)
    vector_r = float(np.mean(per_neuron)) if per_neuron else float("nan")

    if np.isfinite(matrix_r):
        r_clipped = float(np.clip(matrix_r, -0.999999, 0.999999))
        full_r = 2.0 * r_clipped / (1.0 + r_clipped)
    else:
        full_r = float("nan")
        warnings.append("matrix reliability is not finite; ceiling is undefined")

    ceiling_full = max(full_r, 0.0) if np.isfinite(full_r) else float("nan")
    ceiling_half = max(matrix_r, 0.0) if np.isfinite(matrix_r) else float("nan")
    return {
        "metric": metric,
        "metrics_match": bool(metrics_match),
        "common_standardizer_used": bool(common_standardizer is not None),
        "feature_names_aligned": bool(aligned),
        "n_features": len(names),
        "n_pairs": int(da.size),
        # half-length distance correlation between the two independent halves
        "matrix_reliability_spearman": matrix_r,
        # Spearman-Brown correction to the full-length measurement
        "matrix_reliability_full_spearman_brown": full_r,
        # per-neuron fingerprint vector reproducibility (Pearson, scale-invariant)
        "vector_reliability_pearson": vector_r,
        "n_neurons_with_variance": len(per_neuron),
        # attenuation factors: the full-length one is the correct ceiling divisor
        "attenuation_factor_sqrt_ceiling": float(np.sqrt(ceiling_full)) if np.isfinite(ceiling_full) else float("nan"),
        "attenuation_factor_half_length": float(np.sqrt(ceiling_half)) if np.isfinite(ceiling_half) else float("nan"),
        "warnings": warnings,
        "interpretation": (
            "Upper bound on any achievable distance-distance correlation. "
            "matrix_reliability_spearman is the split-half (shorter measurement) "
            "correlation; matrix_reliability_full_spearman_brown = 2r/(1+r) is the "
            "full-length noise ceiling and attenuation_factor_sqrt_ceiling = "
            "sqrt(full) is the factor by which an observed correlation is attenuated."
        ),
    }


# --------------------------------------------------------------------------
# Reliability audit (independence, class balance, split, metric, leakage)
# --------------------------------------------------------------------------
def audit_split_halves(
    labels: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    *,
    n_classes: int | None = None,
    split_name: str | None = None,
) -> dict[str, Any]:
    """Verify the split-half construction used for the noise ceiling.

    Checks: disjointness, coverage of every evaluated sample, and class balance
    between the two halves.
    """
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    a = np.asarray(idx_a, dtype=np.int64).reshape(-1)
    b = np.asarray(idx_b, dtype=np.int64).reshape(-1)
    set_a, set_b = set(a.tolist()), set(b.tolist())
    all_idx = set(range(labels.size))
    n_classes = int(n_classes if n_classes is not None else (labels.max() + 1 if labels.size else 0))

    counts_full = np.bincount(labels, minlength=n_classes)[:n_classes] if labels.size else np.zeros(n_classes, dtype=int)
    counts_a = np.bincount(labels[a], minlength=n_classes)[:n_classes] if a.size else np.zeros(n_classes, dtype=int)
    counts_b = np.bincount(labels[b], minlength=n_classes)[:n_classes] if b.size else np.zeros(n_classes, dtype=int)
    frac_a = counts_a / np.maximum(counts_full, 1)
    frac_b = counts_b / np.maximum(counts_full, 1)
    deviation = float(np.max(np.abs(frac_a - frac_b))) if n_classes else float("nan")
    min_class = int(counts_full.min()) if counts_full.size else 0
    tol = (1.0 / min_class + 1e-9) if min_class > 0 else 1.0
    return {
        "split_name": split_name,
        "n_samples_total": int(labels.size),
        "n_half_a": int(a.size),
        "n_half_b": int(b.size),
        "halves_disjoint": bool(len(set_a & set_b) == 0),
        "n_overlap": int(len(set_a & set_b)),
        "halves_cover_all_samples": bool((set_a | set_b) == all_idx),
        "n_uncovered": int(len(all_idx - (set_a | set_b))),
        "class_counts_total": counts_full.tolist(),
        "class_counts_half_a": counts_a.tolist(),
        "class_counts_half_b": counts_b.tolist(),
        "max_class_fraction_deviation": deviation,
        "class_stratified": bool(np.isfinite(deviation) and deviation <= tol),
        "all_classes_present_in_both_halves": bool(counts_a.size > 0 and counts_b.size > 0
                                                    and np.all(counts_a > 0) and np.all(counts_b > 0)),
    }


def reliability_audit(
    *,
    labels: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    half_a: "FingerprintSpace",
    half_b: "FingerprintSpace",
    full: "FingerprintSpace | None" = None,
    split_name: str | None = None,
    n_classes: int | None = None,
    common_standardizer: Any | None = None,
) -> dict[str, Any]:
    """Full audit of the fingerprint reliability / noise-ceiling calculation.

    Combines the split construction audit with the reliability estimate and records
    the leakage-relevant provenance (which split was used, which metric, whether a
    common standardiser was applied).
    """
    split = audit_split_halves(labels, idx_a, idx_b, n_classes=n_classes, split_name=split_name)
    rel = split_half_reliability(half_a, half_b, common_standardizer=common_standardizer)
    held_out = split_name is None or str(split_name).lower() in ("probe", "dev", "val", "test")
    checks = {
        "halves_disjoint": split["halves_disjoint"],
        "halves_cover_all_samples": split["halves_cover_all_samples"],
        "class_stratified": split["class_stratified"],
        "split_is_held_out": bool(held_out),
        "split_is_not_train": str(split_name).lower() != "train",
        "same_distance_metric": rel["metrics_match"],
        "feature_names_aligned": rel["feature_names_aligned"],
        "fingerprint_uses_labels_by_design": bool(
            getattr(half_a, "meta", {}).get("uses_labels", True)
        ),
    }
    return {
        "split": split,
        "reliability": rel,
        "checks": checks,
        "passed": bool(all(checks.values())),
        "full_fingerprint_n_features": (len(full.feature_names) if full is not None else None),
        "split_name": split_name,
        "note": (
            "The reliability target is measured on a held-out split; the two halves "
            "are disjoint and class-stratified; both halves are reduced with the same "
            "distance metric; the ceiling is the Spearman-Brown corrected full-length "
            "reliability. This is a diagnostic, not evidence for the hypothesis."
        ),
    }


def attenuation_correct(r_observed: float, reliability: float) -> float:
    """Correct an observed correlation for measurement noise in the target."""
    if reliability is None or not np.isfinite(reliability) or reliability <= 1e-6:
        return float("nan")
    return float(r_observed / np.sqrt(reliability))
