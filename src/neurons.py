"""Structured representation of *individual hidden neurons*.

This is the central object of the project. A :class:`NeuronRepresentation`
stores, for one hidden neuron, several named *feature blocks*, each of which is a
dict of interpretable scalar summaries:

===============  =========================================================
block            content
===============  =========================================================
``intrinsic``    membrane time constant, threshold, reset, per-neuron bias
``input_conn``   compact statistics of the 700-dim input weight vector
``recurrent_in`` statistics of the incoming recurrent weight column
``recurrent_out``statistics of the outgoing recurrent weight row
``activity``     label-free firing statistics measured on training data
===============  =========================================================

Nothing in this module is allowed to use class labels, the official test set, or
the neuron's array index. Blocks are selected by name so that any subset can be
used, which is what makes the ablation analysis in :mod:`src.controls` possible.

``to_vector()`` concatenates the requested blocks in a fixed, reproducible order
and returns a plain float64 vector. The blocks are deliberately kept separate
rather than concatenated blindly, because the scientific question is precisely
*which kind* of information about a neuron is informative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .utils import sanitize_features

# --------------------------------------------------------------------------
# Block registry
# --------------------------------------------------------------------------
class FeatureBlock(str, Enum):
    """Names of the separable feature blocks."""

    INTRINSIC = "intrinsic"
    INPUT_CONN = "input_conn"
    RECURRENT_IN = "recurrent_in"
    RECURRENT_OUT = "recurrent_out"
    ACTIVITY = "activity"

    @classmethod
    def all(cls) -> list[str]:
        return [b.value for b in cls]

    @classmethod
    def structural(cls) -> list[str]:
        """Blocks that depend only on the network's parameters (no data at all)."""
        return [cls.INTRINSIC.value, cls.INPUT_CONN.value, cls.RECURRENT_IN.value, cls.RECURRENT_OUT.value]

    @classmethod
    def connectivity(cls) -> list[str]:
        return [cls.INPUT_CONN.value, cls.RECURRENT_IN.value, cls.RECURRENT_OUT.value]

    @classmethod
    def coerce(cls, blocks: str | Iterable[str] | None, default: Sequence[str] | None = None) -> list[str]:
        """Normalise a block specification to a validated, ordered list of names."""
        if blocks is None:
            return list(default) if default is not None else cls.all()
        if isinstance(blocks, str):
            blocks = [b.strip() for b in blocks.replace(",", " ").split() if b.strip()]
        requested = [str(b) for b in blocks]
        # allow shorthands
        expanded: list[str] = []
        for name in requested:
            if name in ("structural",):
                expanded.extend(cls.structural())
            elif name in ("connectivity", "conn"):
                expanded.extend(cls.connectivity())
            elif name in ("all",):
                expanded.extend(cls.all())
            elif name in ("recurrent", "rec"):
                expanded.extend([cls.RECURRENT_IN.value, cls.RECURRENT_OUT.value])
            else:
                expanded.append(name)
        unknown = [b for b in expanded if b not in cls.all()]
        if unknown:
            raise ValueError(f"Unknown feature block(s): {unknown}. Valid blocks: {cls.all()}")
        # de-duplicate, keep the canonical block order for reproducibility
        seen: list[str] = []
        for b in cls.all():
            if b in expanded and b not in seen:
                seen.append(b)
        return seen


# --------------------------------------------------------------------------
# Per-neuron representation
# --------------------------------------------------------------------------
@dataclass
class NeuronRepresentation:
    """The structured representation of a single hidden neuron."""

    neuron_id: int
    features: dict[str, dict[str, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- named accessors requested by the project spec ----------------------
    @property
    def intrinsic_features(self) -> dict[str, float]:
        return self.features.get(FeatureBlock.INTRINSIC.value, {})

    @property
    def input_connectivity_features(self) -> dict[str, float]:
        return self.features.get(FeatureBlock.INPUT_CONN.value, {})

    @property
    def recurrent_connectivity_features(self) -> dict[str, float]:
        merged: dict[str, float] = {}
        merged.update(self.features.get(FeatureBlock.RECURRENT_IN.value, {}))
        merged.update(self.features.get(FeatureBlock.RECURRENT_OUT.value, {}))
        return merged

    @property
    def activity_features(self) -> dict[str, float]:
        return self.features.get(FeatureBlock.ACTIVITY.value, {})

    # -- vectorisation ------------------------------------------------------
    def available_blocks(self) -> list[str]:
        return [b for b in FeatureBlock.all() if b in self.features and len(self.features[b]) > 0]

    def feature_names(self, blocks: str | Iterable[str] | None = None) -> list[str]:
        """Fully-qualified feature names (``"block.name"``) in vector order."""
        selected = FeatureBlock.coerce(blocks, default=self.available_blocks())
        names: list[str] = []
        for block in selected:
            for name in sorted(self.features.get(block, {})):
                names.append(f"{block}.{name}")
        return names

    def to_vector(
        self,
        blocks: str | Iterable[str] | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Concatenate feature blocks into a float64 vector.

        Parameters
        ----------
        blocks:
            Which blocks to include. ``None`` means "all blocks present".
        feature_names:
            Optional explicit ordered feature list (``"block.name"``). Supplying
            it guarantees that several neurons are vectorised consistently.
        """
        names = list(feature_names) if feature_names is not None else self.feature_names(blocks)
        values = np.empty(len(names), dtype=np.float64)
        for i, qualified in enumerate(names):
            block, _, name = qualified.partition(".")
            values[i] = float(self.features.get(block, {}).get(name, 0.0))
        return sanitize_features(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "neuron_id": int(self.neuron_id),
            "features": {b: dict(sorted(f.items())) for b, f in sorted(self.features.items())},
            "metadata": self.metadata,
        }

    def flat_dict(self) -> dict[str, float]:
        """Flat ``"block.name" -> value`` mapping (convenient for CSV/pandas)."""
        out: dict[str, float] = {}
        for block, feats in self.features.items():
            for name, value in feats.items():
                out[f"{block}.{name}"] = float(value)
        return out


# --------------------------------------------------------------------------
# Collection of neurons
# --------------------------------------------------------------------------
@dataclass
class NeuronRepresentationSet:
    """A set of :class:`NeuronRepresentation` objects with matrix conversion."""

    representations: list[NeuronRepresentation]
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.representations)

    def __iter__(self):
        return iter(self.representations)

    def __getitem__(self, idx: int) -> NeuronRepresentation:
        return self.representations[idx]

    @property
    def neuron_ids(self) -> np.ndarray:
        return np.array([r.neuron_id for r in self.representations], dtype=np.int64)

    def available_blocks(self) -> list[str]:
        blocks: set[str] = set()
        for rep in self.representations:
            blocks.update(rep.available_blocks())
        return [b for b in FeatureBlock.all() if b in blocks]

    def feature_names(self, blocks: str | Iterable[str] | None = None) -> list[str]:
        """Union of feature names across neurons, in a canonical order.

        Names are taken from the union rather than from a single neuron so that
        the matrix layout is stable, and a neuron missing a feature (which should
        not normally happen) contributes an explicit zero.
        """
        selected = FeatureBlock.coerce(blocks, default=self.available_blocks())
        per_block: dict[str, set[str]] = {b: set() for b in selected}
        for rep in self.representations:
            for block in selected:
                per_block[block].update(rep.features.get(block, {}).keys())
        names: list[str] = []
        for block in selected:
            names.extend(f"{block}.{name}" for name in sorted(per_block[block]))
        return names

    def to_matrix(
        self,
        blocks: str | Iterable[str] | None = None,
        feature_names: Sequence[str] | None = None,
        drop_constant: bool = False,
    ) -> tuple[np.ndarray, list[str]]:
        """Return ``(X, feature_names)`` with ``X`` of shape ``(n_neurons, n_features)``."""
        names = list(feature_names) if feature_names is not None else self.feature_names(blocks)
        X = np.zeros((len(self), len(names)), dtype=np.float64)
        for i, rep in enumerate(self.representations):
            X[i] = rep.to_vector(feature_names=names)
        if drop_constant and X.shape[1] > 0:
            keep = X.std(axis=0) > 1e-12
            X = X[:, keep]
            names = [n for n, k in zip(names, keep) if k]
        return X, names

    def subset_neurons(self, neuron_ids: Sequence[int]) -> "NeuronRepresentationSet":
        wanted = set(int(n) for n in neuron_ids)
        keep = [r for r in self.representations if r.neuron_id in wanted]
        return NeuronRepresentationSet(keep, meta=dict(self.meta))

    # -- serialisation ------------------------------------------------------
    def to_records(self) -> list[dict[str, Any]]:
        return [rep.to_dict() for rep in self.representations]

    def save_json(self, path: str) -> None:
        from .utils import save_json

        save_json(
            {"meta": self.meta, "blocks": self.available_blocks(), "neurons": self.to_records()},
            path,
        )


# --------------------------------------------------------------------------
# Structural feature extraction (no data, no labels)
# --------------------------------------------------------------------------
def _concentration_stats(w: np.ndarray) -> dict[str, float]:
    """Concentration / sparsity summaries of an absolute-weight vector."""
    a = np.abs(np.asarray(w, dtype=np.float64)).ravel()
    total = float(a.sum())
    n = a.size
    if total <= 0.0 or n == 0:
        return {
            "entropy": 0.0,
            "participation_ratio_frac": 0.0,
            "top5pct_share": 0.0,
            "top1pct_share": 0.0,
            "n_significant_frac": 0.0,
        }
    p = a / total
    nz = p[p > 0]
    entropy = float(-(nz * np.log(nz)).sum() / np.log(n)) if n > 1 else 0.0
    participation = float(1.0 / np.sum(p**2) / n)  # 1/n (concentrated) .. 1 (uniform)
    k5 = max(1, int(np.ceil(0.05 * n)))
    k1 = max(1, int(np.ceil(0.01 * n)))
    top5 = float(np.sort(a)[-k5:].sum() / total)
    top1 = float(np.sort(a)[-k1:].sum() / total)
    thresh = a.mean() + a.std()
    n_sig = float((a > thresh).mean())
    return {
        "entropy": entropy,
        "participation_ratio_frac": participation,
        "top5pct_share": top5,
        "top1pct_share": top1,
        "n_significant_frac": n_sig,
    }


def _vector_stats(w: np.ndarray) -> dict[str, float]:
    """Basic signed statistics shared by all weight vectors."""
    w = np.asarray(w, dtype=np.float64).ravel()
    abs_w = np.abs(w)
    l1 = float(abs_w.sum())
    pos = float(w[w > 0].sum())
    neg = float(-w[w < 0].sum())
    denom = pos + neg
    return {
        "mean": float(w.mean()) if w.size else 0.0,
        "std": float(w.std()) if w.size else 0.0,
        "l1": l1,
        "l2": float(np.sqrt((w**2).sum())),
        "max_abs": float(abs_w.max()) if w.size else 0.0,
        "pos_frac": float((w > 0).mean()) if w.size else 0.0,
        "neg_frac": float((w < 0).mean()) if w.size else 0.0,
        "pos_neg_balance": float((pos - neg) / denom) if denom > 0 else 0.0,
        "rms": float(np.sqrt((w**2).mean())) if w.size else 0.0,
    }


def input_connectivity_features(w_in_column: np.ndarray) -> dict[str, float]:
    """Compact description of one neuron's input weight vector.

    The 700-dimensional weight vector is indexed by SHD channel. Following the
    dataset documentation, channels are ordered along the cochlea (channel 0 is
    the lowest frequency), so the (normalised) weighted centre of mass and spread
    are meaningful tonotopic summaries. This assumption is documented in the
    README; if the channel order were arbitrary these two features would be
    meaningless, while all other features would remain valid.
    """
    w = np.asarray(w_in_column, dtype=np.float64).ravel()
    feats = _vector_stats(w)
    feats.update(_concentration_stats(w))
    n = w.size
    a = np.abs(w)
    total = float(a.sum())
    if n > 1 and total > 0:
        axis = np.arange(n, dtype=np.float64) / (n - 1)  # normalised tonotopic position
        p = a / total
        com = float((p * axis).sum())
        spread = float(np.sqrt(max(((p * (axis - com) ** 2).sum()), 0.0)))
    else:
        com, spread = 0.0, 0.0
    feats["channel_com"] = com
    feats["channel_spread"] = spread
    return feats


def recurrent_incoming_features(w_rec_column: np.ndarray) -> dict[str, float]:
    """Statistics of the weights *onto* a neuron (a column of ``W_rec``)."""
    feats = _vector_stats(w_rec_column)
    feats.update(_concentration_stats(w_rec_column))
    return feats


def recurrent_outgoing_features(w_rec_row: np.ndarray) -> dict[str, float]:
    """Statistics of the weights *from* a neuron (a row of ``W_rec``)."""
    feats = _vector_stats(w_rec_row)
    feats.update(_concentration_stats(w_rec_row))
    return feats


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2:
        return 0.0
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def recurrent_relationship_features(w_rec: np.ndarray, neuron_id: int) -> dict[str, float]:
    """Features describing the *relationship* between incoming and outgoing weights.

    These capture the neuron's position in the recurrent graph beyond the
    magnitude of its connections:

    ``self_connection``
        the direct autapse weight (constant if self-connections are disabled;
        the standardiser then flags it as non-informative).
    ``in_out_correlation``
        Pearson correlation between the neuron's incoming and outgoing weight
        vectors. High values mean the neuron "talks to" the same neurons it
        "listens to" (reciprocal embedding).
    ``in_out_asymmetry``
        signed relative difference of incoming vs outgoing L2 norms.
    ``reciprocal_strength``
        mean product of |incoming| and |outgoing| weights over shared partners,
        i.e. how strongly reciprocity is expressed in absolute terms.
    ``in_out_cosine``
        cosine similarity between incoming and outgoing weight vectors.
    """
    col = np.asarray(w_rec[:, neuron_id], dtype=np.float64)
    row = np.asarray(w_rec[neuron_id, :], dtype=np.float64)
    l2_in = float(np.sqrt((col**2).sum()))
    l2_out = float(np.sqrt((row**2).sum()))
    denom = l2_in + l2_out
    cosine = float((col * row).sum() / (l2_in * l2_out)) if l2_in > 0 and l2_out > 0 else 0.0
    return {
        "self_connection": float(w_rec[neuron_id, neuron_id]),
        "in_out_correlation": _safe_corr(col, row),
        "in_out_cosine": cosine,
        "in_out_asymmetry": float((l2_in - l2_out) / denom) if denom > 0 else 0.0,
        "reciprocal_strength": float(np.mean(np.abs(col) * np.abs(row))) if col.size else 0.0,
    }


def intrinsic_features_from_model(model: Any, neuron_ids: Sequence[int] | None = None) -> dict[str, np.ndarray]:
    """Per-neuron intrinsic parameters gathered from a model.

    If ``neuron_param_mode`` is ``"none"`` every neuron shares the same threshold,
    time constant and reset. The block is still returned and still usable, but it
    is *constant* and therefore flagged as non-informative by the standardiser and
    reported as such - that is an honest result, not a bug.
    """
    import torch

    cfg = model.cfg
    n = cfg.n_hidden
    ids = np.arange(n) if neuron_ids is None else np.asarray(neuron_ids, dtype=np.int64)
    with torch.no_grad():
        tau = model.alpha_tensor().detach().cpu().numpy()
        # recover the effective time constant from alpha to keep units interpretable
        tau_ms = -cfg.bin_ms / np.log(np.clip(tau, 1e-6, 1 - 1e-6))
        thr = model.effective_threshold().detach().cpu().numpy()
        bias = (
            model.b_hid.detach().cpu().numpy()
            if getattr(model, "b_hid", None) is not None
            else np.zeros(n, dtype=np.float64)
        )
    reset = np.full(n, float(model.effective_reset), dtype=np.float64)
    return {
        "tau_mem_ms": tau_ms[ids].astype(np.float64),
        "threshold": thr[ids].astype(np.float64),
        "reset": reset[ids],
        "bias": bias[ids].astype(np.float64),
    }


def extract_structural_representations(model: Any) -> NeuronRepresentationSet:
    """Build the data-free (structural) representation for every hidden neuron.

    Uses only ``W_in``, ``W_rec`` and the neuron parameters. No dataset, no
    labels, no forward pass. This is the representation that the primary
    hypothesis is tested with in the most conservative setting.
    """
    import torch

    with torch.no_grad():
        w_in = model.w_in.detach().cpu().numpy().astype(np.float64)  # (n_input, n_hidden)
        w_rec = model.w_rec.detach().cpu().numpy().astype(np.float64)  # (n_hidden, n_hidden)
    n_hidden = w_rec.shape[0]
    intrinsic = intrinsic_features_from_model(model)

    reps: list[NeuronRepresentation] = []
    for j in range(n_hidden):
        feats = {
            FeatureBlock.INTRINSIC.value: {k: float(v[j]) for k, v in intrinsic.items()},
            FeatureBlock.INPUT_CONN.value: input_connectivity_features(w_in[:, j]),
            FeatureBlock.RECURRENT_IN.value: recurrent_incoming_features(w_rec[:, j]),
            FeatureBlock.RECURRENT_OUT.value: recurrent_outgoing_features(w_rec[j, :]),
        }
        feats[FeatureBlock.RECURRENT_IN.value].update(recurrent_relationship_features(w_rec, j))
        reps.append(
            NeuronRepresentation(
                neuron_id=j,
                features=feats,
                metadata={"source": "structure"},
            )
        )
    return NeuronRepresentationSet(
        reps,
        meta={
            "kind": "structural",
            "uses_labels": False,
            "uses_data": False,
            "n_hidden": n_hidden,
            "n_input": int(w_in.shape[0]),
            "note": (
                "recurrent_in additionally carries incoming/outgoing relationship features "
                "(self_connection, in_out_correlation, in_out_cosine, in_out_asymmetry, "
                "reciprocal_strength)."
            ),
        },
    )


# --------------------------------------------------------------------------
# Activity features (label-free)
# --------------------------------------------------------------------------
def activity_features_from_psth(
    psth: np.ndarray,
    counts: np.ndarray,
    *,
    bin_ms: float,
    n_samples: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Label-free firing-statistics features for every hidden neuron.

    Parameters
    ----------
    psth:
        ``(n_hidden, n_bins)`` spike-time histogram pooled across *all* samples
        (no class information).
    counts:
        ``(n_samples, n_hidden)`` spike counts per sample.
    bin_ms, n_samples:
        Used to convert to Hz and to compute the Fano factor.

    Returns
    -------
    ``(features, flags)`` where ``features`` maps names to ``(n_hidden,)`` arrays
    and ``flags`` maps names to boolean OR integer ``(n_hidden,)`` arrays (e.g.
    ``silent_neuron``).

    Robustness: neurons that never spike get finite, defined values everywhere
    (rate 0, Fano 0, latency censored at the stimulus duration). Dead neurons are
    flagged in ``flags["silent_neuron"]`` so analyses can exclude or at least
    identify them.

    Note: interspike-interval statistics are computed from the pooled per-neuron
    spike-time histogram rather than from individual spike times. This is an
    approximation (it assumes a renewal-like process) that keeps memory bounded;
    it is documented in the README.
    """
    psth = np.asarray(psth, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    n_hidden, n_bins = psth.shape
    duration_s = n_bins * bin_ms / 1000.0
    total_spikes = counts.sum(axis=0)
    silent = total_spikes <= 0

    rate_per_sample = counts / duration_s  # (n_samples, n_hidden) in Hz
    mean_rate = rate_per_sample.mean(axis=0)
    std_rate = rate_per_sample.std(axis=0)
    silent_fraction = (counts <= 0).mean(axis=0)
    mean_count = counts.mean(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        fano = np.where(mean_count > 0, counts.var(axis=0) / np.where(mean_count > 0, mean_count, 1.0), 0.0)

    # -- ISI statistics from the pooled histogram ---------------------------
    psth_sum = np.where(psth.sum(axis=1, keepdims=True) > 0, psth.sum(axis=1, keepdims=True), 1.0)
    p_time = psth / psth_sum
    isi_mean, isi_cv, burstiness = np.zeros(n_hidden), np.zeros(n_hidden), np.zeros(n_hidden)
    active_bins = (psth > 0).sum(axis=1).astype(np.float64)
    for h in range(n_hidden):
        if total_spikes[h] <= 0:
            continue
        # Approximate mean ISI from spike density: duration / #spike-events.
        n_events = max(active_bins[h], 1.0)
        isi_mean[h] = (duration_s * 1000.0) / n_events
        # Second moment of the ISI distribution approximated from the histogram
        # of activity times (exponential-like assumption for the CV).
        t_ms = np.arange(n_bins, dtype=np.float64) * bin_ms
        mean_t = float((p_time[h] * t_ms).sum())
        var_t = float((p_time[h] * (t_ms - mean_t) ** 2).sum())
        sd_t = float(np.sqrt(max(var_t, 0.0)))
        isi_cv[h] = sd_t / mean_t if mean_t > 0 else 0.0
        burstiness[h] = (isi_cv[h] - 1.0) / (isi_cv[h] + 1.0) if isi_cv[h] > 0 else 0.0

    # -- temporal summaries from the pooled PSTH ---------------------------
    t_ms = np.arange(n_bins, dtype=np.float64) * bin_ms
    temporal_center = np.zeros(n_hidden)
    temporal_dispersion = np.zeros(n_hidden)
    peak_rate = np.zeros(n_hidden)
    active_bin_fraction = np.zeros(n_hidden)
    temporal_entropy = np.zeros(n_hidden)
    for h in range(n_hidden):
        if total_spikes[h] <= 0:
            continue
        p = p_time[h]
        mean_t = float((p * t_ms).sum())
        temporal_center[h] = mean_t
        temporal_dispersion[h] = float(np.sqrt(max((p * (t_ms - mean_t) ** 2).sum(), 0.0)))
        peak_rate[h] = float(psth[h].max() / (n_samples * bin_ms / 1000.0))
        active_bin_fraction[h] = float((psth[h] > 0).mean())
        nz = p[p > 0]
        temporal_entropy[h] = float(-(nz * np.log(nz)).sum() / np.log(n_bins)) if n_bins > 1 else 0.0

    features = {
        "rate_hz": mean_rate,
        "rate_std_hz": std_rate,
        "silent_fraction": silent_fraction,
        "mean_spike_count": mean_count,
        "fano_factor": fano,
        "isi_mean_ms": isi_mean,
        "isi_cv": isi_cv,
        "burstiness": burstiness,
        # First-spike latency is computed elsewhere (per-sample) and merged in by
        # the accumulator; default here is the censored maximum.
        "temporal_center_ms": temporal_center,
        "temporal_dispersion_ms": temporal_dispersion,
        "temporal_entropy": temporal_entropy,
        "peak_rate_hz": peak_rate,
        "active_bin_fraction": active_bin_fraction,
        "log_rate_hz": np.log10(mean_rate + 1e-3),
    }
    flags = {
        "silent_neuron": silent,
        "total_spikes": total_spikes,
    }
    return features, flags


def add_first_spike_features(
    features: dict[str, np.ndarray],
    first_spike_sum_ms: np.ndarray,
    first_spike_count: np.ndarray,
    *,
    duration_ms: float,
) -> dict[str, np.ndarray]:
    """Merge first-spike latency statistics into an activity feature dict.

    Censoring: samples in which a neuron never spiked are treated as having a
    latency equal to the stimulus duration. This is a documented, explicit choice
    (rather than NaN) so that distance computations stay well defined.
    """
    first_spike_sum_ms = np.asarray(first_spike_sum_ms, dtype=np.float64)
    first_spike_count = np.asarray(first_spike_count, dtype=np.float64)
    n = first_spike_sum_ms.shape[0]
    latency = np.full(n, float(duration_ms), dtype=np.float64)
    has = first_spike_count > 0
    denom = np.where(has, first_spike_count, 1.0)
    observed_mean = first_spike_sum_ms / denom
    latency[has] = observed_mean[has]
    features = dict(features)
    features["first_spike_latency_ms"] = latency
    features["first_spike_latency_norm"] = latency / max(duration_ms, 1e-9)
    return features


def build_activity_representations(
    features: dict[str, np.ndarray],
    flags: Mapping[str, np.ndarray],
    *,
    neuron_ids: Sequence[int] | None = None,
) -> NeuronRepresentationSet:
    """Wrap activity feature arrays into a :class:`NeuronRepresentationSet`."""
    any_block = next(iter(features.values()))
    n_hidden = int(np.asarray(any_block).shape[0])
    ids = np.arange(n_hidden) if neuron_ids is None else np.asarray(neuron_ids, dtype=np.int64)
    reps: list[NeuronRepresentation] = []
    for idx, j in enumerate(ids):
        feats = {name: float(sanitize_features(np.asarray(values).ravel()[idx: idx + 1])[0]) for name, values in features.items()}
        meta = {name: (bool(v[idx]) if np.asarray(v).dtype == bool else float(np.asarray(v)[idx])) for name, v in flags.items()}
        reps.append(
            NeuronRepresentation(neuron_id=int(j), features={FeatureBlock.ACTIVITY.value: feats}, metadata=meta)
        )
    return NeuronRepresentationSet(
        reps,
        meta={
            "kind": "activity",
            "uses_labels": False,
            "n_hidden": n_hidden,
            "note": "activity statistics are computed on the reference (training) split only",
        },
    )


def merge_representation_sets(*sets: NeuronRepresentationSet) -> NeuronRepresentationSet:
    """Merge sets that cover the same neurons by unioning their feature blocks."""
    sets = [s for s in sets if len(s) > 0]
    if not sets:
        raise ValueError("No representations to merge")
    n = len(sets[0])
    if any(len(s) != n for s in sets):
        raise ValueError("All representation sets must contain the same number of neurons")
    ids = sets[0].neuron_ids
    for s in sets[1:]:
        if not np.array_equal(s.neuron_ids, ids):
            raise ValueError("All representation sets must refer to the same neuron ids in the same order")
    merged: list[NeuronRepresentation] = []
    for i in range(n):
        features: dict[str, dict[str, float]] = {}
        metadata: dict[str, Any] = {}
        for s in sets:
            rep = s[i]
            for block, feats in rep.features.items():
                features.setdefault(block, {}).update(feats)
            metadata.update(rep.metadata)
        merged.append(NeuronRepresentation(neuron_id=int(ids[i]), features=features, metadata=metadata))
    meta: dict[str, Any] = {"merged_from": [s.meta.get("kind", "unknown") for s in sets]}
    meta["uses_labels"] = any(bool(s.meta.get("uses_labels", False)) for s in sets)
    return NeuronRepresentationSet(merged, meta=meta)
