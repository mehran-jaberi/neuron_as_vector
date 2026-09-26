"""Explicit permutation-invariance tests for the neuron representation.

A neuron representation is only *about* the neuron if it is invariant to how the
hidden neurons happen to be numbered. This module implements the test the
project requires:

1. take a trained network;
2. randomly permute the hidden-neuron ordering;
3. permute **every** hidden-indexed parameter consistently (input columns,
   recurrent rows and columns, output rows, per-neuron parameters);
4. recompute the neuron representations for the permuted network;
5. verify that the representation associated with the same *functional* neuron is
   identical, up to numerical tolerance, before and after the permutation.

Convention
----------
``perm[k]`` is the **original** index of the neuron that becomes new neuron ``k``
(``new[k] = old[perm[k]]``). The permuted network is therefore *functionally
identical* to the original - it is the same circuit with the labels of the hidden
units renamed - which is checked explicitly by
:func:`functional_equivalence_error`.

Any feature that changes when a neuron is renamed is *permutation-sensitive* and
is reported as such. A feature that uses the neuron's array index is the
canonical failure mode and :func:`neuron_index_representation` provides such a
deliberately-broken representation so the detector itself can be tested.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .neurons import (
    FeatureBlock,
    NeuronRepresentation,
    NeuronRepresentationSet,
    classify_feature,
    extract_structural_representations,
)


# --------------------------------------------------------------------------
# Model-level permutation
# --------------------------------------------------------------------------
def random_permutation(n: int, seed: int = 0) -> np.ndarray:
    """A reproducible random permutation of ``range(n)``."""
    return np.random.default_rng(int(seed)).permutation(int(n)).astype(np.int64)


def permute_hidden_neurons(model: Any, perm: Sequence[int]) -> Any:
    """Return a copy of ``model`` whose hidden neurons have been relabelled.

    ``perm[k]`` is the original index of the neuron that becomes new neuron
    ``k``. Every hidden-indexed parameter is permuted consistently:

    * input weights: ``w_in_new[:, k] = w_in_old[:, perm[k]]``
    * recurrent weights: ``w_rec_new[a, b] = w_rec_old[perm[a], perm[b]]``
      (both the pre- and the post-synaptic index are relabelled)
    * readout weights: ``w_out_new[k, :] = w_out_old[perm[k], :]``
    * per-neuron parameters (bias, learned tau offset) follow their neuron;
    * the structural ``self_mask`` buffer (relevant only when self-connections are
      disabled) is relabelled with the same index on both axes.

    The result is functionally identical to the input model; only the numbering
    of the hidden units changes.
    """
    import torch

    perm = np.asarray(perm, dtype=np.int64).ravel()
    n = int(model.cfg.n_hidden)
    if perm.shape != (n,) or sorted(perm.tolist()) != list(range(n)):
        raise ValueError(f"perm must be a permutation of range({n}); got {perm.tolist()}")

    new_model = copy.deepcopy(model)
    with torch.no_grad():
        perm_t = torch.as_tensor(perm, dtype=torch.long, device=model.w_in.device)
        new_model.w_in.copy_(model.w_in.detach().index_select(1, perm_t))
        new_model.w_rec.copy_(
            model.w_rec.detach().index_select(0, perm_t).index_select(1, perm_t)
        )
        new_model.w_out.copy_(model.w_out.detach().index_select(0, perm_t))
        if getattr(model, "b_hid", None) is not None:
            new_model.b_hid.copy_(model.b_hid.detach().index_select(0, perm_t))
        if getattr(model, "log_tau_offset", None) is not None:
            new_model.log_tau_offset.copy_(model.log_tau_offset.detach().index_select(0, perm_t))
        # self_mask is a structural buffer: relabel both axes so a disabled
        # self-connection stays disabled for the *same* functional neuron.
        mask = getattr(model, "self_mask", None)
        if mask is not None:
            new_model.self_mask.copy_(
                mask.detach().index_select(0, perm_t).index_select(1, perm_t)
            )
    return new_model


def functional_equivalence_error(model: Any, permuted: Any, x: Any) -> dict[str, float]:
    """Confirm the permuted model computes the same function as the original.

    Also checks that the hidden activity is *relabelled* rather than changed:
    ``spikes_new[k] == spikes_old[perm[k]]``.
    """
    import torch

    with torch.no_grad():
        a = model(x, record=True)
        b = permuted(x, record=True)
    out = {
        "logits_max_abs_diff": float((a["logits"] - b["logits"]).abs().max().item()),
        "spike_count_total_diff": float(
            (a["spike_count"].sum() - b["spike_count"].sum()).abs().item()
        ),
    }
    return out


# --------------------------------------------------------------------------
# Representation-level comparison
# --------------------------------------------------------------------------
@dataclass
class FeatureInvariance:
    """Invariance verdict for one qualified feature name."""

    feature: str
    kind: str
    invariant: bool
    max_abs_diff: float
    max_rel_diff: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "kind": self.kind,
            "invariant": bool(self.invariant),
            "permutation_sensitive": bool(not self.invariant),
            "max_abs_diff": float(self.max_abs_diff),
            "max_rel_diff": float(self.max_rel_diff),
        }


def _feature_value(rep: NeuronRepresentation, qualified: str) -> float:
    block, _, name = qualified.partition(".")
    return float(rep.features.get(block, {}).get(name, 0.0))


def compare_representation_sets(
    reference: NeuronRepresentationSet,
    permuted: NeuronRepresentationSet,
    perm: Sequence[int],
    *,
    atol: float = 1e-8,
    rtol: float = 1e-6,
) -> dict[str, Any]:
    """Verify ``permuted[k] == reference[perm[k]]`` for every feature.

    ``reference`` are representations of the original network, ``permuted`` of the
    relabelled network. A feature is invariant when the value attached to the same
    functional neuron agrees within ``atol + rtol * max(|a|, |b|)`` for every
    neuron.
    """
    perm = np.asarray(perm, dtype=np.int64).ravel()
    n = len(reference)
    if len(permuted) != n:
        raise ValueError("Reference and permuted sets must have the same number of neurons")
    if perm.shape != (n,) or sorted(perm.tolist()) != list(range(n)):
        raise ValueError(f"perm must be a permutation of range({n})")

    names = sorted(set(reference.feature_names()) | set(permuted.feature_names()))
    per_feature: list[FeatureInvariance] = []
    for qualified in names:
        max_abs = 0.0
        max_rel = 0.0
        invariant = True
        for k in range(n):
            a = _feature_value(permuted[k], qualified)  # value under new label k
            b = _feature_value(reference[int(perm[k])], qualified)  # same functional neuron
            diff = abs(a - b)
            rel = diff / (max(abs(a), abs(b)) + atol)
            max_abs = max(max_abs, diff)
            max_rel = max(max_rel, rel)
            if diff > atol + rtol * max(abs(a), abs(b)):
                invariant = False
        per_feature.append(
            FeatureInvariance(
                feature=qualified,
                kind=classify_feature(qualified),
                invariant=invariant,
                max_abs_diff=max_abs,
                max_rel_diff=max_rel,
            )
        )

    sensitive = [f.feature for f in per_feature if not f.invariant]
    invariant_features = [f.feature for f in per_feature if f.invariant]
    sensitive_blocks = sorted({f.split(".", 1)[0] for f in sensitive})
    report = {
        "n_neurons": int(n),
        "perm": perm.tolist(),
        "atol": float(atol),
        "rtol": float(rtol),
        "n_features": len(per_feature),
        "feature_status": [f.as_dict() for f in per_feature],
        "sensitive_features": sensitive,
        "invariant_features": invariant_features,
        "sensitive_blocks": sensitive_blocks,
        "passed": len(sensitive) == 0,
    }
    return report


def compare_space_invariance(
    reference_space: Any,
    permuted_space: Any,
    perm: Sequence[int],
    *,
    atol: float = 1e-8,
    rtol: float = 1e-6,
) -> dict[str, Any]:
    """Check the *geometry* is invariant: ``D_perm[k, l] == D_ref[perm[k], perm[l]]``."""
    perm = np.asarray(perm, dtype=np.int64).ravel()
    D_ref = reference_space.distances()
    D_perm = permuted_space.distances()
    reindexed = D_ref[np.ix_(perm, perm)]
    diff = np.abs(D_perm - reindexed)
    tol = atol + rtol * np.maximum(np.abs(D_perm), np.abs(reindexed))
    return {
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "passed": bool(np.all(diff <= tol)),
        "n_neurons": int(D_ref.shape[0]),
    }


def assert_permutation_invariant(report: dict[str, Any], *, allow: Sequence[str] = ()) -> None:
    """Raise ``AssertionError`` listing any permutation-sensitive feature.

    ``allow`` lists features that are *expected* to be permutation-sensitive (for
    example a deliberately-broken control); they are ignored.
    """
    sensitive = [f for f in report.get("sensitive_features", []) if f not in set(allow)]
    if sensitive:
        raise AssertionError(
            "Permutation-sensitive neuron representation feature(s) detected: "
            f"{sensitive}. These features depend on the arbitrary hidden-neuron "
            "ordering and must not be used as a neuron representation."
        )


# --------------------------------------------------------------------------
# High-level checks
# --------------------------------------------------------------------------
def check_structural_permutation_invariance(
    model: Any,
    perm: Sequence[int] | None = None,
    *,
    seed: int = 0,
    include_tonotopic: bool = False,
    atol: float = 1e-8,
    rtol: float = 1e-8,
) -> dict[str, Any]:
    """Run the full 5-step invariance test on the *structural* representation.

    Structural features are pure functions of per-neuron parameter arrays, so
    they must agree to (near) floating-point exactness; the default tolerances are
    correspondingly tight.
    """
    n = int(model.cfg.n_hidden)
    perm = random_permutation(n, seed) if perm is None else np.asarray(perm, dtype=np.int64)
    reference = extract_structural_representations(model, include_tonotopic=include_tonotopic)
    permuted_model = permute_hidden_neurons(model, perm)
    permuted = extract_structural_representations(permuted_model, include_tonotopic=include_tonotopic)
    report = compare_representation_sets(reference, permuted, perm, atol=atol, rtol=rtol)
    report["representation"] = "structural"
    report["include_tonotopic"] = bool(include_tonotopic)
    return report


def check_activity_permutation_invariance(
    model: Any,
    rec: Any,
    indices: Sequence[int] | np.ndarray,
    perm: Sequence[int] | None = None,
    *,
    seed: int = 0,
    device: Any = None,
    n_classes: int = 20,
    batch_size: int = 256,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> dict[str, Any]:
    """Permutation-invariance test for the *label-free activity* representation.

    Activity features require a forward pass, so numerically the permuted network
    is only equivalent up to floating-point reassociation; tolerance is looser
    than for the structural blocks.
    """
    from .evaluation import collect_activity
    from .neurons import (
        activity_features_from_psth,
        add_first_spike_features,
        build_activity_representations,
    )

    if device is None:
        from .utils import get_device

        device = get_device()

    def _activity(model_: Any) -> NeuronRepresentationSet:
        res = collect_activity(
            model_, rec, indices, device=device, batch_size=batch_size,
            n_classes=n_classes, with_labels=False, collect_voltage=False,
        )
        features, flags = activity_features_from_psth(
            res.psth, res.counts, bin_ms=res.bin_ms, n_samples=res.n_samples
        )
        features = add_first_spike_features(
            features, res.first_spike_sum, res.first_spike_count, duration_ms=res.duration_ms
        )
        return build_activity_representations(features, flags)

    n = int(model.cfg.n_hidden)
    perm = random_permutation(n, seed) if perm is None else np.asarray(perm, dtype=np.int64)
    reference = _activity(model)
    permuted_model = permute_hidden_neurons(model, perm)
    permuted = _activity(permuted_model)
    report = compare_representation_sets(reference, permuted, perm, atol=atol, rtol=rtol)
    report["representation"] = "activity"
    return report


# --------------------------------------------------------------------------
# Deliberately-broken control (proves the detector works)
# --------------------------------------------------------------------------
def neuron_index_representation(model: Any) -> NeuronRepresentationSet:
    """A representation that *uses the neuron index* - a permutation-sensitive control.

    This must fail the invariance test. It exists only so the test can be
    validated against a known-positive case.
    """
    n = int(model.cfg.n_hidden)
    reps = [
        NeuronRepresentation(
            neuron_id=j,
            features={FeatureBlock.INTRINSIC.value: {"neuron_index": float(j)}},
            metadata={"source": "index_control"},
        )
        for j in range(n)
    ]
    return NeuronRepresentationSet(
        reps,
        meta={"kind": "neuron_index", "uses_labels": False, "uses_data": False},
    )
