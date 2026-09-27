"""Structured representation of *individual hidden neurons*.

This is the central object of the project. A :class:`NeuronRepresentation`
stores, for one hidden neuron, several named *feature blocks*, each of which is a
dict of interpretable scalar summaries:

=================  =======================================================
block              content
=================  =======================================================
``intrinsic``      per-neuron parameters actually *learned* for the neuron
``input_conn``     compact statistics of the input weight vector
``recurrent_in``   statistics of the incoming recurrent weights (a **row**)
``recurrent_out``  statistics of the outgoing recurrent weights (a **column**)
``activity``       label-free firing statistics measured on a reference split
=================  =======================================================

Weight orientation (see :mod:`src.model`)
-----------------------------------------
``w_rec[i, j]`` is the weight **from neuron j to neuron i** (``forward`` computes
``s_prev @ w_rec.t()``). Therefore *row* ``i`` holds the weights **onto** neuron
*i* (incoming) and *column* ``j`` holds the weights **from** neuron *j*
(outgoing). The ``recurrent_in`` / ``recurrent_out`` blocks follow this
convention explicitly.

Generic learned bias vs. genuine dynamical parameters
-----------------------------------------------------
Block ``intrinsic`` distinguishes two very different things by feature name and
by metadata:

* :data:`DYNAMICAL_FEATURE_NAMES` -- per-neuron *dynamical* parameters that the
  architecture genuinely learns per neuron (e.g. a per-neuron membrane time
  constant in ``bias_tau`` mode). These may legitimately be called intrinsic
  dynamical parameters.
* :data:`GENERIC_LEARNED_FEATURE_NAMES` -- generic learned additive terms (a
  per-neuron ``learned_bias``). These are **not** biophysical parameters and are
  never described as such. Shared constants (threshold, reset, the base membrane
  time constant when it is not learned per neuron) are *not* emitted as features
  at all, because a constant carries zero neuron-specific information.

Input-channel ordering
----------------------
By default the ``input_conn`` block contains only features that are invariant to
an arbitrary permutation of the input channels. The tonotopic "centre of mass"
and "spread" (which *require* a known channel ordering) are **not** part of the
default/primary representation: nothing in the local documentation or in the SHD
HDF5 files establishes a channel ordering, so assuming one would be an
unsupported assumption. They are available only behind the explicit
``include_tonotopic=True`` opt-in and are listed in :data:`TONOTOPIC_FEATURE_NAMES`.

Nothing in this module uses class labels, the official test set, or the neuron's
array index as a feature. Blocks are selected by name so that any subset can be
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
# Feature semantics (audit-friendly, machine-readable)
# --------------------------------------------------------------------------
#: Generic learned additive terms. A per-neuron bias changes the neuron's
#: excitability but is *not* a biophysical parameter; it must never be reported
#: as "intrinsic biophysics".
GENERIC_LEARNED_FEATURE_NAMES = frozenset({"intrinsic.learned_bias"})

#: Genuine per-neuron *dynamical* parameters (present only when the architecture
#: actually learns them per neuron, e.g. ``neuron_param_mode="bias_tau"``).
DYNAMICAL_FEATURE_NAMES = frozenset({"intrinsic.tau_mem_ms"})

#: Features that assume a known (tonotopic) ordering of the input channels. They
#: are excluded from the default/primary representation because a channel
#: ordering cannot be established from the local documentation or the SHD files.
TONOTOPIC_FEATURE_NAMES = frozenset({"input_conn.channel_com", "input_conn.channel_spread"})


def classify_feature(qualified_name: str) -> str:
    """Return ``"generic_learned"``, ``"dynamical"``, ``"tonotopic"`` or ``"other"``."""
    if qualified_name in GENERIC_LEARNED_FEATURE_NAMES:
        return "generic_learned"
    if qualified_name in DYNAMICAL_FEATURE_NAMES:
        return "dynamical"
    if qualified_name in TONOTOPIC_FEATURE_NAMES:
        return "tonotopic"
    return "other"


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

    def generic_learned_features(self) -> dict[str, float]:
        """Generic learned additive terms (e.g. ``intrinsic.learned_bias``).

        Kept separate from :meth:`dynamical_features` so a learned excitability
        offset is never confused with a genuine biophysical parameter.
        """
        out: dict[str, float] = {}
        for block, feats in self.features.items():
            for name, value in feats.items():
                if f"{block}.{name}" in GENERIC_LEARNED_FEATURE_NAMES:
                    out[f"{block}.{name}"] = float(value)
        return out

    def dynamical_features(self) -> dict[str, float]:
        """Genuine per-neuron dynamical parameters (may be empty)."""
        out: dict[str, float] = {}
        for block, feats in self.features.items():
            for name, value in feats.items():
                if f"{block}.{name}" in DYNAMICAL_FEATURE_NAMES:
                    out[f"{block}.{name}"] = float(value)
        return out

    def feature_kinds(self) -> dict[str, str]:
        """Map every qualified feature name to its :func:`classify_feature` kind."""
        return {f"{b}.{n}": classify_feature(f"{b}.{n}") for b, f in self.features.items() for n in f}

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


def input_connectivity_features(
    w_in_column: np.ndarray,
    *,
    include_tonotopic: bool = False,
) -> dict[str, float]:
    """Compact description of one neuron's input weight vector.

    All features returned by default are *permutation invariant* with respect to
    the ordering of the input channels: they are symmetric functions of the
    multiset of input weights (norms, sparsity, sign balance, ...). Nothing here
    depends on the channel index.

    ``include_tonotopic`` (default ``False``)
        When ``True`` two ordering-dependent features are added:
        ``channel_com`` (normalised weighted centre of mass over the channel
        index) and ``channel_spread``. These are meaningful **only if** channel
        index increases monotonically with cochlear characteristic frequency.
        That ordering is *not* established by the local documentation or by the
        SHD HDF5 files (which contain no channel metadata), so these features are
        deliberately excluded from the primary representation and are opt-in only.
        See :data:`TONOTOPIC_FEATURE_NAMES`.
    """
    w = np.asarray(w_in_column, dtype=np.float64).ravel()
    feats = _vector_stats(w)
    feats.update(_concentration_stats(w))
    if include_tonotopic:
        n = w.size
        a = np.abs(w)
        total = float(a.sum())
        if n > 1 and total > 0:
            axis = np.arange(n, dtype=np.float64) / (n - 1)  # assumed tonotopic position
            p = a / total
            com = float((p * axis).sum())
            spread = float(np.sqrt(max(((p * (axis - com) ** 2).sum()), 0.0)))
        else:
            com, spread = 0.0, 0.0
        feats["channel_com"] = com
        feats["channel_spread"] = spread
    return feats


def recurrent_incoming_features(w_rec_row: np.ndarray) -> dict[str, float]:
    """Statistics of the weights *onto* a neuron.

    ``w_rec[i, j]`` is the weight from ``j`` to ``i`` (see :mod:`src.model`), so
    the weights **onto** neuron ``j`` are the **row** ``w_rec[j, :]``.
    """
    feats = _vector_stats(w_rec_row)
    feats.update(_concentration_stats(w_rec_row))
    return feats


def recurrent_outgoing_features(w_rec_column: np.ndarray) -> dict[str, float]:
    """Statistics of the weights *from* a neuron.

    The weights **from** neuron ``j`` are the **column** ``w_rec[:, j]``.
    """
    feats = _vector_stats(w_rec_column)
    feats.update(_concentration_stats(w_rec_column))
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
    incoming = np.asarray(w_rec[neuron_id, :], dtype=np.float64)  # weights onto this neuron
    outgoing = np.asarray(w_rec[:, neuron_id], dtype=np.float64)  # weights from this neuron
    l2_in = float(np.sqrt((incoming**2).sum()))
    l2_out = float(np.sqrt((outgoing**2).sum()))
    denom = l2_in + l2_out
    cosine = float((incoming * outgoing).sum() / (l2_in * l2_out)) if l2_in > 0 and l2_out > 0 else 0.0
    return {
        "self_connection": float(w_rec[neuron_id, neuron_id]),
        "in_out_correlation": _safe_corr(incoming, outgoing),
        "in_out_cosine": cosine,
        "in_out_asymmetry": float((l2_in - l2_out) / denom) if denom > 0 else 0.0,
        "reciprocal_strength": float(np.mean(np.abs(incoming) * np.abs(outgoing))) if incoming.size else 0.0,
    }


def _varies(values: np.ndarray, rtol: float = 1e-9) -> bool:
    """True if ``values`` actually differ across neurons (not a shared constant)."""
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size < 2:
        return False
    ptp = float(np.ptp(v))
    if ptp == 0.0:
        return False
    return ptp > rtol * (float(np.abs(v).mean()) + rtol)


def intrinsic_features_from_model(model: Any, neuron_ids: Sequence[int] | None = None) -> dict[str, np.ndarray]:
    """Per-neuron parameters that the architecture *actually learns per neuron*.

    Only parameters that genuinely differ across neurons are emitted. Shared
    constants are omitted on purpose: the threshold, the base membrane time
    constant and the reset value are identical for every neuron (unless a
    per-neuron parameterisation is used), so including them would add
    zero-information columns that the standardiser would discard anyway.

    Two kinds of parameter are handled explicitly and never conflated:

    ``tau_mem_ms``
        A genuine per-neuron *dynamical* parameter. It appears only when the
        model learns a per-neuron time constant (``neuron_param_mode="bias_tau"``
        *and* the learned offsets are non-degenerate). Listed in
        :data:`DYNAMICAL_FEATURE_NAMES`.
    ``learned_bias``
        A generic learned per-neuron additive input (a learned excitability
        offset). It is a legitimate learned *intrinsic* parameter but it is **not**
        a biophysical parameter and is never described as one. Listed in
        :data:`GENERIC_LEARNED_FEATURE_NAMES`.

    An untrained network (all-zero bias / offsets) therefore contributes **no**
    intrinsic features; a neuron-specific intrinsic representation only exists
    once the parameter is actually learned to vary.
    """
    import torch

    cfg = model.cfg
    n = cfg.n_hidden
    ids = np.arange(n, dtype=np.int64) if neuron_ids is None else np.asarray(neuron_ids, dtype=np.int64)
    with torch.no_grad():
        alpha = model.alpha_tensor().detach().cpu().numpy().astype(np.float64)
        tau_ms_all = -cfg.bin_ms / np.log(np.clip(alpha, 1e-6, 1 - 1e-6))
        b_hid = getattr(model, "b_hid", None)
        bias_all = b_hid.detach().cpu().numpy().astype(np.float64) if b_hid is not None else None

    feats: dict[str, np.ndarray] = {}
    tau_ms = tau_ms_all[ids]
    if _varies(tau_ms):
        feats["tau_mem_ms"] = tau_ms
    if bias_all is not None:
        bias = bias_all[ids]
        if _varies(bias):
            feats["learned_bias"] = bias
    return feats


def intrinsic_feature_provenance(model: Any) -> dict[str, Any]:
    """Explain, for a given model, which intrinsic parameters exist and why.

    Useful for the audit/README report: it distinguishes genuine per-neuron
    dynamical parameters, generic learned terms, and shared constants that are
    deliberately not emitted as features.
    """
    import torch

    cfg = model.cfg
    n = cfg.n_hidden
    with torch.no_grad():
        alpha = model.alpha_tensor().detach().cpu().numpy().astype(np.float64)
        tau_ms = -cfg.bin_ms / np.log(np.clip(alpha, 1e-6, 1 - 1e-6))
        b_hid = getattr(model, "b_hid", None)
        bias = b_hid.detach().cpu().numpy().astype(np.float64) if b_hid is not None else None
    has_per_neuron_tau = getattr(model, "log_tau_offset", None) is not None
    return {
        "neuron_param_mode": str(getattr(model, "neuron_param_mode", cfg.neuron_param_mode)),
        "n_hidden": int(n),
        "dynamical": {
            "tau_mem_ms": {
                "learned_per_neuron": bool(has_per_neuron_tau),
                "varies_across_neurons": bool(_varies(tau_ms)),
                "emitted": bool(has_per_neuron_tau and _varies(tau_ms)),
                "note": "genuine per-neuron dynamical parameter" if has_per_neuron_tau else "base tau is shared",
            }
        },
        "generic_learned": {
            "learned_bias": {
                "present": bool(bias is not None),
                "varies_across_neurons": bool(_varies(bias)) if bias is not None else False,
                "emitted": bool(bias is not None and _varies(bias)),
                "note": "generic learned additive excitation; NOT a biophysical parameter",
            }
        },
        "shared_constants_not_emitted": {
            "threshold": float(model.effective_threshold().mean().item()),
            "reset": float(model.effective_reset),
            "base_tau_mem_ms": float(cfg.tau_mem_ms),
        },
    }



def extract_structural_representations(
    model: Any,
    *,
    include_tonotopic: bool = False,
) -> NeuronRepresentationSet:
    """Build the data-free (structural) representation for every hidden neuron.

    Uses only ``W_in``, ``W_rec`` and the (learned) neuron parameters. No dataset,
    no labels, no forward pass. This is the representation that the primary
    hypothesis is tested with in the most conservative setting.

    Orientation follows :mod:`src.model`: for neuron ``j`` the **incoming**
    recurrent weights are the row ``w_rec[j, :]`` and the **outgoing** weights are
    the column ``w_rec[:, j]``.

    ``include_tonotopic`` (default ``False``) additionally enables the two
    ordering-dependent input features; they are excluded from the primary
    representation because no local evidence establishes the channel ordering.
    """
    import torch

    with torch.no_grad():
        w_in = model.w_in.detach().cpu().numpy().astype(np.float64)  # (n_input, n_hidden)
        w_rec = model.w_rec.detach().cpu().numpy().astype(np.float64)  # (n_hidden, n_hidden)
    n_hidden = w_rec.shape[0]
    intrinsic = intrinsic_features_from_model(model)
    use_intrinsic = len(intrinsic) > 0

    reps: list[NeuronRepresentation] = []
    for j in range(n_hidden):
        feats: dict[str, dict[str, float]] = {}
        if use_intrinsic:
            feats[FeatureBlock.INTRINSIC.value] = {k: float(v[j]) for k, v in intrinsic.items()}
        feats[FeatureBlock.INPUT_CONN.value] = input_connectivity_features(
            w_in[:, j], include_tonotopic=include_tonotopic
        )
        rec_in = recurrent_incoming_features(w_rec[j, :])  # row = weights onto j
        rec_in.update(recurrent_relationship_features(w_rec, j))
        feats[FeatureBlock.RECURRENT_IN.value] = rec_in
        feats[FeatureBlock.RECURRENT_OUT.value] = recurrent_outgoing_features(w_rec[:, j])  # col = from j
        reps.append(
            NeuronRepresentation(
                neuron_id=j,
                features=feats,
                metadata={"source": "structure"},
            )
        )
    blocks_present = [b for b in FeatureBlock.structural() if use_intrinsic or b != FeatureBlock.INTRINSIC.value]
    return NeuronRepresentationSet(
        reps,
        meta={
            "kind": "structural",
            "uses_labels": False,
            "uses_data": False,
            "n_hidden": n_hidden,
            "n_input": int(w_in.shape[0]),
            "blocks_present": blocks_present,
            "intrinsic_block_present": bool(use_intrinsic),
            "include_tonotopic_features": bool(include_tonotopic),
            "generic_learned_features": sorted(GENERIC_LEARNED_FEATURE_NAMES)
            if FeatureBlock.INTRINSIC.value in {f"{b}" for b in blocks_present}
            else [],
            "dynamical_features": sorted(DYNAMICAL_FEATURE_NAMES) if use_intrinsic else [],
            "note": (
                "recurrent_in = row w_rec[j,:] (weights onto j, includes incoming/outgoing "
                "relationship features); recurrent_out = column w_rec[:,j] (weights from j). "
                "`learned_bias` is a generic learned term, NOT a biophysical parameter. "
                "Shared constants (threshold/reset/base tau) are not emitted."
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

    Audit note (see ``archive/documentation/AUDIT_REPRESENTATION.md``): the earlier ``isi_cv`` /
    ``burstiness`` / ``isi_mean_ms`` names were **mislabelled** - they were
    computed from the pooled per-neuron spike-time histogram, not from
    interspike intervals. ``spike_time_cv`` now states what it actually is (the
    coefficient of variation of the *pooled spike times*, i.e. temporal
    dispersion), and the exact duplicates ``mean_spike_count`` (= ``rate_hz`` up
    to a constant) and ``burstiness`` (= a monotone transform of
    ``spike_time_cv``) are no longer emitted.
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

    # -- temporal dispersion from the pooled histogram ----------------------
    # NOTE: these describe the distribution of *spike times* (pooled over the
    # reference split), NOT interspike intervals. The feature is named
    # ``spike_time_cv`` accordingly.
    psth_sum = np.where(psth.sum(axis=1, keepdims=True) > 0, psth.sum(axis=1, keepdims=True), 1.0)
    p_time = psth / psth_sum
    spike_time_cv = np.zeros(n_hidden)
    for h in range(n_hidden):
        if total_spikes[h] <= 0:
            continue
        t_ms = np.arange(n_bins, dtype=np.float64) * bin_ms
        mean_t = float((p_time[h] * t_ms).sum())
        var_t = float((p_time[h] * (t_ms - mean_t) ** 2).sum())
        sd_t = float(np.sqrt(max(var_t, 0.0)))
        spike_time_cv[h] = sd_t / mean_t if mean_t > 0 else 0.0

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
        "fano_factor": fano,
        "spike_time_cv": spike_time_cv,
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
