"""Primary geometry/function analysis, controls and cross-validated prediction.

This module is the single entry point for the pre-registered analysis. It keeps
the central scientific separation explicit:

* **proposed representation** - label-free, built from the network parameters and
  label-free activity (:mod:`src.neurons`, :mod:`src.representations`);
* **functional fingerprint** - an independent, class-conditioned measurement of
  what the neuron does (:mod:`src.functional_fingerprint`), measured on the
  held-out **analysis-probe** split and never fed back into the representation.

Primary statistic
-----------------
Spearman correlation between the upper-triangle (condensed) representation
distances ``D_representation(i, j)`` and functional distances ``D_function(i, j)``,
with a Mantel permutation test. The permutation relabels **neurons**, so neuron
pairs are never treated as independent samples. Every result reports the p-value
resolution floor ``1 / (n_perm + 1)`` and whether the observed effect saturated it.

Controls
--------
1. ``rate_only`` - a single scalar firing rate as the "representation".
2. ``rate_normalized_fingerprint`` - the representation is tested against a
   **rate-normalized** fingerprint (the primary rate control).
3. ``rate_matched`` - a stratified Mantel comparing only neuron pairs with similar
   firing rates (a proper rate-matched control).
4. ``random_representation`` - i.i.d. Gaussian features, matched dimensionality.
5. ``shuffled_neurons`` - the real representation with neuron rows permuted.
6. kNN functional similarity for ``k = 3, 5, 10, 20``.

A partial Mantel controlling for firing rate is retained **only** as a secondary,
exploratory statistic and is explicitly labelled as *not* proof of rate
independence.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .functional_fingerprint import FingerprintSpace
from .geometry_analysis import geometry_function_analysis
from .neurons import FeatureBlock, NeuronRepresentationSet
from .prediction import run_prediction_suite
from .representations import (
    RepresentationSpace,
    random_baseline_space,
    rate_only_space,
    shuffle_control_space,
)


# --------------------------------------------------------------------------
# Rate-distance helper (label-free)
# --------------------------------------------------------------------------
def abs_rate_difference_condensed(activity_reps: NeuronRepresentationSet) -> np.ndarray:
    """Condensed ``|rate_i - rate_j|`` in Hz from the label-free activity block."""
    X, names = activity_reps.to_matrix(blocks=[FeatureBlock.ACTIVITY.value])
    idx = [i for i, n in enumerate(names) if n.endswith(".rate_hz")]
    if not idx:
        idx = [i for i, n in enumerate(names) if n.endswith(".log_rate_hz")]
    rates = X[:, idx[0]] if idx else np.zeros(X.shape[0])
    r = rates.reshape(-1, 1)
    return np.abs(r - r.T)[np.triu_indices(r.size, 1)]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _public(analysis: Mapping[str, Any]) -> dict[str, Any]:
    """Drop private (``_``-prefixed) keys so the dict is JSON-safe."""
    return {k: v for k, v in analysis.items() if not str(k).startswith("_")}


def _analyze(
    space_matrix: np.ndarray,
    fingerprint: FingerprintSpace,
    *,
    n_perm: int,
    k_values: Sequence[int],
    seed: int,
    n_curve_bins: int,
    rate_absdiff: np.ndarray | None,
    nuisance: np.ndarray | None,
    bootstrap: int = 0,
    return_null: bool = False,
    n_strata: int = 5,
) -> dict[str, Any]:
    return geometry_function_analysis(
        space_matrix,
        fingerprint.X,
        n_perm=n_perm,
        k_values=k_values,
        seed=seed,
        n_curve_bins=n_curve_bins,
        rate_absdiff_condensed=rate_absdiff,
        nuisance_condensed=nuisance,
        rate_matched_strata=n_strata,
        primary_bootstrap=bootstrap,
        primary_return_null=return_null,
        include_curves=True,
    )


def _headline(analysis: Mapping[str, Any]) -> dict[str, Any]:
    """Compact, human-facing summary of one analysis (primary + rate controls)."""
    primary = analysis.get("primary_mantel_spearman", {})
    rate_matched = analysis.get("rate_matched_mantel", {})
    knn_rows = analysis.get("knn", {}).get("table", [])
    return {
        "primary_mantel_spearman_r": primary.get("statistic"),
        "p_value": primary.get("p_value"),
        "p_value_floor": primary.get("p_value_floor"),
        "at_resolution_floor": primary.get("at_resolution_floor"),
        "effect_size_z": primary.get("effect_size_z"),
        "null_mean": primary.get("null_mean"),
        "null_std": primary.get("null_std"),
        "n_perm": primary.get("n_perm"),
        "bootstrap_ci": primary.get("bootstrap_ci"),
        "rate_matched_mantel_r": rate_matched.get("statistic"),
        "rate_matched_mantel_p_value": rate_matched.get("p_value"),
        "rate_matched_mantel_effect_size_z": rate_matched.get("effect_size_z"),
        "knn": knn_rows,
    }


# --------------------------------------------------------------------------
# Main orchestrator
# --------------------------------------------------------------------------
def run_function_analysis(
    representation_space: RepresentationSpace,
    fingerprints: Mapping[str, FingerprintSpace],
    *,
    primary_fingerprint: str,
    rate_normalized_fingerprint: str | None = None,
    secondary_fingerprints: Sequence[str] = (),
    activity_reps: NeuronRepresentationSet | None = None,
    n_perm: int = 10000,
    k_values: Sequence[int] = (3, 5, 10, 20),
    seed: int = 0,
    n_curve_bins: int = 20,
    random_repeats: int = 5,
    n_strata: int = 5,
    bootstrap: int = 2000,
    prediction: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the primary analysis, all controls, and cross-validated prediction.

    Returns ``{"summary": <JSON-safe dict>, "arrays": <numpy arrays for NPZ>}``.
    """
    if primary_fingerprint not in fingerprints:
        raise KeyError(f"primary_fingerprint {primary_fingerprint!r} not in {sorted(fingerprints)}")
    fp_primary = fingerprints[primary_fingerprint]

    rate_absdiff = abs_rate_difference_condensed(activity_reps) if activity_reps is not None else None
    nuisance = rate_absdiff  # |Delta rate| as the (secondary) partial-Mantel nuisance

    arrays: dict[str, np.ndarray] = {}

    def _store_null(key: str, analysis: Mapping[str, Any]) -> None:
        null = analysis.get("_primary_null")
        if null is not None:
            arrays[f"null_{key}"] = np.asarray(null, dtype=np.float64)

    # ---- primary -------------------------------------------------------
    primary = _analyze(
        representation_space.X, fp_primary,
        n_perm=n_perm, k_values=k_values, seed=seed, n_curve_bins=n_curve_bins,
        rate_absdiff=rate_absdiff, nuisance=nuisance, bootstrap=bootstrap, return_null=True,
        n_strata=n_strata,
    )
    arrays["dx_primary"] = primary["_condensed"]["representation"]
    arrays["dy_primary"] = primary["_condensed"]["functional"]
    _store_null("primary", primary)

    summary: dict[str, Any] = {
        "meta": {
            "primary_fingerprint": primary_fingerprint,
            "primary_fingerprint_dim": int(fp_primary.X.shape[1]),
            "primary_fingerprint_features": list(fp_primary.feature_names),
            "representation_dim": int(representation_space.X.shape[1]),
            "representation_blocks": list(representation_space.blocks),
            "n_neurons": int(representation_space.n_neurons),
            "n_pairs": int(representation_space.n_neurons * (representation_space.n_neurons - 1) // 2),
            "n_perm": int(n_perm),
            "p_value_floor": 1.0 / (1.0 + n_perm),
            "seed": int(seed),
            "pairs_not_independent": True,
            "pairs_independence_note": (
                "Permutation relabels neurons (not pairs); neuron pairs are NOT treated "
                "as independent samples. All p-values saturate at 1/(n_perm+1)."
            ),
            "leakage_boundary": (
                "representation = label-free; fingerprint = class-conditioned held-out "
                "responses, never used to build the representation."
            ),
        },
        "primary": _public(primary),
        "primary_headline": _headline(primary),
    }

    # ---- secondary targets ---------------------------------------------
    summary["secondary_fingerprints"] = {}
    for name in secondary_fingerprints:
        if name not in fingerprints:
            continue
        res = _analyze(
            representation_space.X, fingerprints[name],
            n_perm=n_perm, k_values=k_values, seed=seed, n_curve_bins=n_curve_bins,
            rate_absdiff=rate_absdiff, nuisance=nuisance, n_strata=n_strata,
        )
        summary["secondary_fingerprints"][name] = {
            "headline": _headline(res),
            "fingerprint_dim": int(fingerprints[name].X.shape[1]),
        }

    # ---- controls ------------------------------------------------------
    controls: dict[str, Any] = {}

    # C1: rate-only representation
    if activity_reps is not None:
        rate_space = rate_only_space(activity_reps)
        res = _analyze(
            rate_space.X, fp_primary, n_perm=n_perm, k_values=k_values, seed=seed,
            n_curve_bins=n_curve_bins, rate_absdiff=rate_absdiff, nuisance=None,
            n_strata=n_strata,
        )
        res["is_control"] = True
        res["description"] = "single scalar firing rate as the representation"
        controls["rate_only"] = _public(res)
        controls["rate_only_headline"] = _headline(res)

    # C2: rate-normalized functional fingerprint (PRIMARY rate control)
    if rate_normalized_fingerprint and rate_normalized_fingerprint in fingerprints:
        fp_norm = fingerprints[rate_normalized_fingerprint]
        res = _analyze(
            representation_space.X, fp_norm, n_perm=n_perm, k_values=k_values, seed=seed,
            n_curve_bins=n_curve_bins, rate_absdiff=None, nuisance=None,
        )
        res["is_control"] = True
        res["description"] = (
            "PRIMARY RATE CONTROL: representation vs a rate-normalized tuning fingerprint, "
            "so overall firing-rate magnitude cannot drive the similarity."
        )
        controls["rate_normalized_fingerprint"] = _public(res)
        controls["rate_normalized_fingerprint_headline"] = _headline(res)

    # C3: random representation (repeated)
    random_rows: list[dict[str, Any]] = []
    for rep in range(int(max(1, random_repeats))):
        rnd = random_baseline_space(
            representation_space.n_neurons,
            representation_space.X.shape[1],
            seed=int(seed) + 1000 * rep,
            weighting="uniform",
        )
        res = _analyze(
            rnd.X, fp_primary, n_perm=n_perm, k_values=k_values, seed=seed + 1000 * rep,
            n_curve_bins=n_curve_bins, rate_absdiff=None, nuisance=None,
        )
        random_rows.append(_headline(res))
    controls["random_representation"] = {
        "is_control": True,
        "n_repeats": len(random_rows),
        "description": "i.i.d. Gaussian features, matched dimensionality (chance level)",
        "repeats": random_rows,
        "primary_r_mean": float(np.nanmean([r["primary_mantel_spearman_r"] for r in random_rows])),
        "primary_r_std": float(np.nanstd([r["primary_mantel_spearman_r"] for r in random_rows])),
    }

    # C4: shuffled-neuron control
    shuffled = shuffle_control_space(representation_space, seed=int(seed))
    res = _analyze(
        shuffled.X, fp_primary, n_perm=n_perm, k_values=k_values, seed=seed,
        n_curve_bins=n_curve_bins, rate_absdiff=None, nuisance=None,
    )
    res["is_control"] = True
    res["description"] = "real representation with neuron rows permuted (should destroy the effect)"
    controls["shuffled_neurons"] = _public(res)
    controls["shuffled_neurons_headline"] = _headline(res)

    summary["controls"] = controls

    # ---- prediction ----------------------------------------------------
    pred_cfg = dict(prediction or {})
    if pred_cfg.get("enabled", True):
        fp_matrices = {name: fp.X for name, fp in fingerprints.items()}
        summary["prediction"] = run_prediction_suite(
            representation_space.X,
            fp_matrices,
            n_splits=int(pred_cfg.get("n_splits", 5)),
            alphas=tuple(pred_cfg.get("alphas", (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0))),
            knn_k=int(pred_cfg.get("knn_k", 5)),
            seed=int(seed),
            random_control_seed=pred_cfg.get("random_control_seed", None),
        )

    return {"summary": summary, "arrays": arrays}
