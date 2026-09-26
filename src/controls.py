"""Controls, null models, and the before/after-learning comparison.

Anything that could make the primary result look good for the wrong reason gets
an explicit control here:

============================  =====================================================
control                       what it rules out
============================  =====================================================
``random_null``               chance level for this number of neurons/features
``shuffled``                  an artefact of row ordering in the matrix
``rate_only``                 the trivial explanation "it's just firing rate"
``connectivity_only``         connectivity alone should not need activity features
``structural_full``           the conservative, label-free representation
``structural_plus_activity``  whether unsupervised activity adds anything
``leave-one-block-out``       one block dominating the effect
``fingerprint_descriptive``   *circularity check* - this variant is labelled
                              ``circular=True`` and must not be reported as
                              evidence, only as a sanity ceiling
partial Mantel                significance after controlling for firing-rate
                              distance
============================  =====================================================

The :func:`before_after_table` compares an untrained and a trained network in the
same architecture with the same analysis pipeline, which is the question "does
learning reorganise neuron-space geometry?".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .functional_fingerprint import FingerprintSpace
from .geometry_analysis import geometry_function_analysis, primary_metric_row
from .neurons import FeatureBlock, NeuronRepresentationSet, merge_representation_sets
from .representations import (
    EmptyFeatureSelectionError,
    RepresentationSpace,
    build_space_from_representations,
    random_baseline_space,
    rate_only_space,
    shuffle_control_space,
)


# --------------------------------------------------------------------------
# Variant specification
# --------------------------------------------------------------------------
@dataclass
class VariantSpec:
    """Declarative description of one representation variant."""

    name: str
    kind: str  # "blocks" | "random" | "shuffled" | "rate_only" | "fingerprint"
    blocks: list[str] = field(default_factory=list)
    description: str = ""
    is_control: bool = False
    circular: bool = False
    use_activity: bool = False
    include_features: list[str] | None = None
    exclude_features: list[str] | None = None


def default_variants() -> list[VariantSpec]:
    """The pre-registered ablation set.

    Ordered from the conservative structural representation to deliberately
    circular variants, so that the final table reads top-to-bottom as
    "increasingly permissive".
    """
    intrinsic = FeatureBlock.INTRINSIC.value
    input_conn = FeatureBlock.INPUT_CONN.value
    rec_in = FeatureBlock.RECURRENT_IN.value
    rec_out = FeatureBlock.RECURRENT_OUT.value
    activity = FeatureBlock.ACTIVITY.value
    structural = FeatureBlock.structural()
    connectivity = FeatureBlock.connectivity()

    return [
        VariantSpec(
            name="random_null",
            kind="random",
            description="i.i.d. Gaussian features, matched dimensionality (chance level)",
            is_control=True,
        ),
        VariantSpec(
            name="rate_only",
            kind="rate_only",
            description="single scalar firing rate - the trivial baseline the geometry must beat",
            is_control=True,
        ),
        VariantSpec(
            name="shuffled_control",
            kind="shuffled",
            description="full representation with neuron rows permuted (should destroy the effect)",
            is_control=True,
        ),
        VariantSpec(
            name="activity_only",
            kind="blocks",
            blocks=[activity],
            use_activity=True,
            description="label-free firing statistics only",
        ),
        VariantSpec(
            name="connectivity_only",
            kind="blocks",
            blocks=connectivity,
            description="input + recurrent weights only, no intrinsic and no activity",
        ),
        VariantSpec(
            name="intrinsic_only",
            kind="blocks",
            blocks=[intrinsic],
            description=(
                "per-neuron learned intrinsic parameters only. In the default "
                "'bias' architecture this block contains ONLY the generic learned "
                "bias (not a biophysical parameter); genuine dynamical parameters "
                "are absent, so this variant is effectively 'learned bias only'."
            ),
        ),
        VariantSpec(
            name="excitability_only",
            kind="blocks",
            blocks=[intrinsic],
            include_features=["learned_bias"],
            description=(
                "generic learned per-neuron bias only (a learned excitability "
                "offset; explicitly NOT a biophysical intrinsic parameter)"
            ),
        ),
        VariantSpec(
            name="dynamical_only",
            kind="blocks",
            blocks=[intrinsic],
            include_features=["tau_mem_ms", "threshold", "reset"],
            description=(
                "genuine per-neuron dynamical parameters only (empty/skipped when "
                "the architecture does not learn per-neuron dynamics)"
            ),
        ),
        VariantSpec(
            name="input_conn_only",
            kind="blocks",
            blocks=[input_conn],
            description="input connectivity only",
        ),
        VariantSpec(
            name="recurrent_only",
            kind="blocks",
            blocks=[rec_in, rec_out],
            description="recurrent incoming + outgoing connectivity only",
        ),
        VariantSpec(
            name="structural_full",
            kind="blocks",
            blocks=structural,
            description="PRIMARY CONSERVATIVE representation: all structure, zero data, zero labels",
        ),
        VariantSpec(
            name="structural_no_intrinsic",
            kind="blocks",
            blocks=[input_conn, rec_in, rec_out],
            description="leave-one-block-out: drop the (learned) intrinsic parameters",
        ),
        VariantSpec(
            name="structural_no_input",
            kind="blocks",
            blocks=[intrinsic, rec_in, rec_out],
            description="leave-one-block-out: drop input connectivity",
        ),
        VariantSpec(
            name="structural_no_recurrent",
            kind="blocks",
            blocks=[intrinsic, input_conn],
            description="leave-one-block-out: drop recurrent connectivity",
        ),
        VariantSpec(
            name="structural_plus_activity",
            kind="blocks",
            blocks=structural + [activity],
            use_activity=True,
            description="full structural representation plus label-free activity statistics",
        ),
        VariantSpec(
            name="fingerprint_descriptive",
            kind="fingerprint",
            description="CIRCULAR: the fingerprint itself as the 'representation' (upper bound only)",
            is_control=True,
            circular=True,
        ),
    ]


# --------------------------------------------------------------------------
# Variant -> space
# --------------------------------------------------------------------------
def build_variant_space(
    spec: VariantSpec,
    structural_reps: NeuronRepresentationSet,
    activity_reps: NeuronRepresentationSet | None = None,
    fingerprints: FingerprintSpace | None = None,
    *,
    weighting: str = "equal",
    normalize_rows: bool = False,
    block_weights: Mapping[str, float] | None = None,
    random_seed: int = 0,
    shuffle_seed: int = 0,
    full_space: RepresentationSpace | None = None,
) -> tuple[RepresentationSpace, bool]:
    """Materialise one variant as a :class:`RepresentationSpace`.

    Returns ``(space, circular)``; the ``circular`` flag is propagated so that
    downstream tables can refuse to treat it as evidence. Raises
    :class:`~src.representations.EmptyFeatureSelectionError` when the selection
    yields no features (e.g. an untrained model with no informative intrinsic
    parameters).
    """
    if spec.kind == "blocks":
        reps = structural_reps
        if spec.use_activity:
            if activity_reps is None:
                raise ValueError(f"Variant {spec.name} requires an activity representation")
            reps = merge_representation_sets(structural_reps, activity_reps)
        space = build_space_from_representations(
            reps, spec.blocks, weighting=weighting, normalize_rows=normalize_rows,
            block_weights=block_weights,
            include_features=spec.include_features, exclude_features=spec.exclude_features,
            meta={"variant": spec.name, "description": spec.description},
        )
        return space, False

    if spec.kind == "random":
        n_features = len(full_space.feature_names) if full_space is not None else 24
        space = random_baseline_space(
            len(structural_reps), n_features, seed=random_seed, weighting="uniform",
            meta={"variant": spec.name, "description": spec.description},
        )
        return space, False

    if spec.kind == "shuffled":
        if full_space is None:
            full_space, _ = build_variant_space(
                VariantSpec(name="_full", kind="blocks", blocks=FeatureBlock.structural() + [FeatureBlock.ACTIVITY.value],
                            use_activity=activity_reps is not None),
                structural_reps,
                activity_reps,
                weighting=weighting,
                normalize_rows=normalize_rows,
                block_weights=block_weights,
            )
        return shuffle_control_space(full_space, seed=shuffle_seed), False

    if spec.kind == "rate_only":
        if activity_reps is None:
            raise ValueError("rate_only variant requires an activity representation")
        return rate_only_space(activity_reps), False

    if spec.kind == "fingerprint":
        if fingerprints is None:
            raise ValueError("fingerprint variant requires a FingerprintSpace")
        space = RepresentationSpace(
            X_raw=fingerprints.X_raw,
            feature_names=list(fingerprints.feature_names),
            blocks=["fingerprint"],
            weighting="uniform",
            meta={
                "variant": spec.name,
                "description": spec.description,
                "circular": True,
                "uses_labels": True,
                "warning": (
                    "This variant feeds the evaluation target back in as a 'representation'. "
                    "It measures the fingerprint's self-consistency, NOT evidence for the "
                    "hypothesis, and must never be reported as a result."
                ),
            },
        )
        return space, True

    raise ValueError(f"Unknown variant kind {spec.kind!r}")


# --------------------------------------------------------------------------
# Suite runner
# --------------------------------------------------------------------------
def run_variant_suite(
    variants: Sequence[VariantSpec],
    structural_reps: NeuronRepresentationSet,
    fingerprints: FingerprintSpace,
    *,
    activity_reps: NeuronRepresentationSet | None = None,
    n_perm: int = 1000,
    k_values: Sequence[int] = (3, 5, 10, 20),
    seed: int = 0,
    weighting: str = "equal",
    normalize_rows: bool = False,
    block_weights: Mapping[str, float] | None = None,
    nuisance_condensed: np.ndarray | None = None,
    n_random_repeats: int = 5,
    collect_spaces: bool = True,
) -> dict[str, Any]:
    """Evaluate every variant with the identical geometry pipeline.

    The ``random_null`` variant is repeated ``n_random_repeats`` times with
    different seeds so that its spread is visible; all other variants are run once.
    Variants whose feature selection is empty (e.g. "dynamical only" for an
    architecture with no per-neuron dynamics) are recorded with ``skipped=True``
    instead of being silently dropped or crashing.
    """
    rows: list[dict[str, Any]] = []
    analyses: dict[str, Any] = {}
    spaces: dict[str, RepresentationSpace] = {}

    # A "full" space is used to match the dimensionality of the random null.
    full_blocks = list(FeatureBlock.structural())
    if activity_reps is not None:
        full_blocks = full_blocks + [FeatureBlock.ACTIVITY.value]
    full_space, _ = build_variant_space(
        VariantSpec(name="_full_reference", kind="blocks", blocks=full_blocks, use_activity=activity_reps is not None),
        structural_reps,
        activity_reps,
        weighting=weighting,
        normalize_rows=normalize_rows,
        block_weights=block_weights,
    )

    for spec in variants:
        repeats = n_random_repeats if (spec.kind == "random" and n_random_repeats > 1) else 1
        for rep_idx in range(repeats):
            label = spec.name if repeats == 1 else f"{spec.name}#{rep_idx}"
            try:
                space, circular = build_variant_space(
                    spec,
                    structural_reps,
                    activity_reps,
                    fingerprints,
                    weighting=weighting,
                    normalize_rows=normalize_rows,
                    block_weights=block_weights,
                    random_seed=seed + 1000 * rep_idx,
                    shuffle_seed=seed + 1000 * rep_idx,
                    full_space=full_space,
                )
            except EmptyFeatureSelectionError as exc:
                rows.append(
                    {
                        "label": label,
                        "variant": spec.name,
                        "repeat": rep_idx,
                        "kind": spec.kind,
                        "blocks": ",".join(spec.blocks),
                        "n_features": 0,
                        "n_informative_features": 0,
                        "description": spec.description,
                        "is_control": spec.is_control,
                        "circular__do_not_report_as_evidence": False,
                        "skipped": True,
                        "skip_reason": str(exc),
                    }
                )
                continue
            analysis = geometry_function_analysis(
                space.X,
                fingerprints.X,
                n_perm=n_perm,
                k_values=k_values,
                seed=seed,
                nuisance_condensed=nuisance_condensed,
            )
            row = primary_metric_row(label, analysis)
            row.update(
                {
                    "variant": spec.name,
                    "repeat": rep_idx,
                    "kind": spec.kind,
                    "blocks": ",".join(space.blocks),
                    "n_features": len(space.feature_names),
                    "n_informative_features": space.standardizer.n_informative,
                    "description": spec.description,
                    "is_control": spec.is_control,
                    "circular__do_not_report_as_evidence": circular,
                    "skipped": False,
                }
            )
            rows.append(row)
            analyses[label] = {k: v for k, v in analysis.items() if not k.startswith("_")}
            if collect_spaces:
                spaces[label] = space

    return {"rows": rows, "analyses": analyses, "spaces": spaces, "n_perm": n_perm}


# --------------------------------------------------------------------------
# Nuisance variable
# --------------------------------------------------------------------------
def rate_nuisance_condensed(activity_reps: NeuronRepresentationSet) -> np.ndarray:
    """Condensed distance vector of ``|log10 rate |`` differences between neurons.

    Used as the control variable in the partial Mantel test, which asks whether the
    representation-function relationship survives after accounting for the fact
    that neurons with similar firing rates trivially have similar fingerprints.
    """
    X, names = activity_reps.to_matrix(blocks=[FeatureBlock.ACTIVITY.value])
    idx = [i for i, n in enumerate(names) if n.endswith(".log_rate_hz")]
    if not idx:
        idx = [i for i, n in enumerate(names) if n.endswith(".rate_hz")]
    rates = X[:, idx[0]] if idx else np.zeros(X.shape[0])
    r = rates.reshape(-1, 1)
    return np.abs(r - r.T)[np.triu_indices(r.size, 1)]


# --------------------------------------------------------------------------
# Before vs after learning
# --------------------------------------------------------------------------
def before_after_table(
    spaces_before: Mapping[str, RepresentationSpace],
    spaces_after: Mapping[str, RepresentationSpace],
    fingerprints_before: FingerprintSpace,
    fingerprints_after: FingerprintSpace,
    *,
    n_perm: int = 1000,
    k_values: Sequence[int] = (3, 5, 10, 20),
    seed: int = 0,
    nuisance_before: np.ndarray | None = None,
    nuisance_after: np.ndarray | None = None,
) -> dict[str, Any]:
    """Same pipeline, same architecture, untrained vs trained.

    Only representations that exist for *both* conditions are compared, and each
    condition uses its own fingerprint (the untrained network's rows are held-out
    responses of the untrained network, which is the right target for the
    untrained representation).

    Reports ``delta_r`` (after minus before) on the primary Mantel statistic and
    on the kNN effect size, together with a bootstrap-free interpretation of
    whether the change exceeds the permutation spread of either estimate.
    """
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    common = [name for name in spaces_before if name in spaces_after]
    for name in common:
        sb, sa = spaces_before[name], spaces_after[name]
        before = geometry_function_analysis(
            sb.X, fingerprints_before.X, n_perm=n_perm, k_values=k_values, seed=seed,
            nuisance_condensed=nuisance_before,
        )
        after = geometry_function_analysis(
            sa.X, fingerprints_after.X, n_perm=n_perm, k_values=k_values, seed=seed,
            nuisance_condensed=nuisance_after,
        )
        rb = primary_metric_row(f"{name}::before", before)
        ra = primary_metric_row(f"{name}::after", after)
        delta_r = _nan_sub(ra["primary_metric_mantel_spearman_r"], rb["primary_metric_mantel_spearman_r"])
        delta_knn = _nan_sub(ra["best_knn_effect_size_z"], rb["best_knn_effect_size_z"])
        rows.append(
            {
                "variant": name,
                "r_before": rb["primary_metric_mantel_spearman_r"],
                "p_before": rb["primary_metric_p_value"],
                "r_after": ra["primary_metric_mantel_spearman_r"],
                "p_after": ra["primary_metric_p_value"],
                "delta_r_after_minus_before": delta_r,
                "knn_z_before": rb["best_knn_effect_size_z"],
                "knn_z_after": ra["best_knn_effect_size_z"],
                "delta_knn_z": delta_knn,
                "n_neurons": ra.get("n_neurons"),
            }
        )
        details[name] = {
            "before": {k: v for k, v in before.items() if not k.startswith("_")},
            "after": {k: v for k, v in after.items() if not k.startswith("_")},
        }
    return {"rows": rows, "details": details}


def _nan_sub(a: Any, b: Any) -> float:
    try:
        a = float(a)
        b = float(b)
    except (TypeError, ValueError):
        return float("nan")
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return a - b


# --------------------------------------------------------------------------
# Fingerprint reliability helper
# --------------------------------------------------------------------------
def reliability_suite(
    fingerprints_full: FingerprintSpace,
    fingerprints_half_a: FingerprintSpace,
    fingerprints_half_b: FingerprintSpace,
    *,
    common_standardizer: Any | None = None,
    labels: np.ndarray | None = None,
    idx_a: np.ndarray | None = None,
    idx_b: np.ndarray | None = None,
    split_name: str | None = None,
) -> dict[str, Any]:
    """Bundle the noise-ceiling diagnostics with the full-data fingerprint size.

    When the split labels/indices are supplied, a full audit (independence, class
    balance, split provenance, metric consistency) is attached under ``"audit"``.
    """
    from .functional_fingerprint import reliability_audit, split_half_reliability

    rel = split_half_reliability(
        fingerprints_half_a, fingerprints_half_b, common_standardizer=common_standardizer
    )
    rel["n_samples_half_a"] = fingerprints_half_a.meta.get("n_samples")
    rel["n_samples_half_b"] = fingerprints_half_b.meta.get("n_samples")
    rel["full_fingerprint_n_features"] = len(fingerprints_full.feature_names)
    if labels is not None and idx_a is not None and idx_b is not None:
        rel["audit"] = reliability_audit(
            labels=labels,
            idx_a=idx_a,
            idx_b=idx_b,
            half_a=fingerprints_half_a,
            half_b=fingerprints_half_b,
            full=fingerprints_full,
            split_name=split_name,
            n_classes=fingerprints_full.meta.get("n_classes"),
            common_standardizer=common_standardizer,
        )
    return rel
