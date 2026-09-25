"""Shared model -> representation / fingerprint construction for the analysis scripts.

The two analysis entry points (``extract_representations.py`` and
``run_geometry_analysis.py``) need to turn a trained model plus held-out data into

* the **neuron representation** (the object under study; label-free), built from
  the network parameters and from label-free firing statistics on a reference
  split, and
* the **functional fingerprint** (the independent evaluation target; uses labels),
  built from class-conditioned held-out responses.

Keeping this in one place means the leakage boundary (representation = no labels;
fingerprint = labels, held-out only) is defined exactly once and both scripts are
guaranteed to use the identical construction.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.evaluation import collect_activity
from src.functional_fingerprint import (
    FingerprintConfig,
    FingerprintSpace,
    class_conditioned_fingerprint,
    class_conditioned_rate_matrix,
)
from src.neurons import (
    FeatureBlock,
    add_first_spike_features,
    activity_features_from_psth,
    build_activity_representations,
    extract_structural_representations,
    merge_representation_sets,
)
from src.representations import RepresentationSpace, build_space_from_representations


# --------------------------------------------------------------------------
# Neuron representation (label-free)
# --------------------------------------------------------------------------
def build_activity_representation(
    model: Any,
    ref_rec: Any,
    ref_idx: np.ndarray,
    *,
    device: Any,
    n_classes: int,
    batch_size: int = 256,
) -> tuple[Any, dict[str, Any]]:
    """Label-free activity representation measured on the reference split.

    Returns ``(NeuronRepresentationSet, diagnostics)``. The diagnostics (firing
    rates, silent fraction) are saved so a degenerate, all-silent solution cannot
    be silently analysed.
    """
    res = collect_activity(
        model, ref_rec, ref_idx, device=device, batch_size=batch_size,
        n_classes=n_classes, with_labels=False, collect_voltage=True,
    )
    features, flags = activity_features_from_psth(
        res.psth, res.counts, bin_ms=res.bin_ms, n_samples=res.n_samples
    )
    features = add_first_spike_features(
        features, res.first_spike_sum, res.first_spike_count, duration_ms=res.duration_ms
    )
    reps = build_activity_representations(features, flags)
    duration_s = max(res.duration_s, 1e-9)
    rates = res.counts.sum(axis=0) / (res.n_samples * duration_s)
    diagnostics = {
        "reference_split": getattr(ref_rec, "name", "reference"),
        "n_samples": int(res.n_samples),
        "n_hidden": int(res.n_hidden),
        "mean_rate_hz": float(rates.mean()),
        "silent_neuron_fraction": float((rates <= 0).mean()),
        "low_rate_fraction_lt_0p5hz": float((rates < 0.5).mean()),
    }
    return reps, diagnostics


def build_representation_bundle(
    model: Any,
    ref_rec: Any,
    ref_idx: np.ndarray,
    *,
    device: Any,
    n_classes: int,
    batch_size: int = 256,
    weighting: str = "equal",
    normalize_rows: bool = False,
    include_activity_in_primary: bool = False,
    primary_blocks: list[str] | None = None,
) -> dict[str, Any]:
    """Build structural + activity representations and the derived metric spaces.

    * ``structural``  - from parameters only (no data, no labels).
    * ``activity``    - label-free firing statistics on the reference split.
    * ``full``        - union of the two.

    Returns a dict containing the representation sets, the representations spaces
    (with the *primary* one under key ``"primary"``), and diagnostics.
    """
    structural = extract_structural_representations(model)
    activity, diag = build_activity_representation(
        model, ref_rec, ref_idx, device=device, n_classes=n_classes, batch_size=batch_size
    )
    full = merge_representation_sets(structural, activity)

    if primary_blocks is None:
        primary_blocks = list(FeatureBlock.structural())
        if include_activity_in_primary:
            primary_blocks = primary_blocks + [FeatureBlock.ACTIVITY.value]

    spaces: dict[str, RepresentationSpace] = {
        "structural": build_space_from_representations(
            structural, FeatureBlock.structural(), weighting=weighting, normalize_rows=normalize_rows
        ),
        "activity": build_space_from_representations(
            activity, [FeatureBlock.ACTIVITY.value], weighting=weighting, normalize_rows=normalize_rows
        ),
        "full": build_space_from_representations(
            full, FeatureBlock.all(), weighting=weighting, normalize_rows=normalize_rows
        ),
        "primary": build_space_from_representations(
            full, primary_blocks, weighting=weighting, normalize_rows=normalize_rows
        ),
    }
    return {
        "structural": structural,
        "activity": activity,
        "full": full,
        "spaces": spaces,
        "primary_blocks": primary_blocks,
        "diagnostics": diag,
    }


# --------------------------------------------------------------------------
# Functional fingerprint (uses labels, held-out only)
# --------------------------------------------------------------------------
def build_fingerprint(
    model: Any,
    eval_rec: Any,
    eval_idx: np.ndarray,
    *,
    device: Any,
    n_classes: int,
    batch_size: int = 256,
    fp_config: FingerprintConfig | None = None,
    min_spikes_for_latency: float = 1.0,
) -> dict[str, Any]:
    """Class-conditioned response fingerprint measured on a held-out split.

    Returns a dict with the ``FingerprintSpace``, the class x neuron rate matrix
    (for figures), ``class_n`` and a provenance block that records the split used.
    """
    fp_config = fp_config or FingerprintConfig()
    res = collect_activity(
        model, eval_rec, eval_idx, device=device, batch_size=batch_size,
        n_classes=n_classes, with_labels=True, collect_voltage=False,
    )
    X, names = class_conditioned_fingerprint(
        res.class_psth,
        res.class_counts,
        res.class_n,
        res.class_first_spike_sum,
        res.class_first_spike_count,
        n_bins=res.n_bins,
        bin_ms=res.bin_ms,
        feature_sets=fp_config.feature_sets,
    )
    rate_matrix = class_conditioned_rate_matrix(res.class_psth, res.class_n, bin_ms=res.bin_ms)
    space = FingerprintSpace(
        X_raw=X,
        feature_names=names,
        config=fp_config,
        meta={
            "eval_split": getattr(eval_rec, "name", "eval"),
            "n_samples": int(res.n_samples),
            "n_neurons": int(X.shape[0]),
            "uses_labels": True,
            "n_classes": int(n_classes),
            "min_spikes_for_latency": float(min_spikes_for_latency),
        },
    )
    return {
        "space": space,
        "rate_matrix": rate_matrix,
        "class_n": res.class_n,
        "class_psth": res.class_psth,
        "result": res,
    }
