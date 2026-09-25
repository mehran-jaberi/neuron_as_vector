"""Turning structured neuron representations into a metric *neuron space*.

A :class:`RepresentationSpace` owns

* the raw feature matrix ``X`` (one row per hidden neuron),
* the feature names and their block membership,
* a fitted standardiser,
* and the derived standardised/weighted matrix used for distances.

Two design choices matter scientifically and are therefore explicit and
configurable:

1. **Per-feature standardisation.** Feature blocks mix units (norms, Hz, ms,
   fractions, entropies). Without z-scoring, the norms would dominate every
   distance. Columns with zero variance are mapped to a constant ``0`` and
   reported as non-informative rather than producing NaNs.
2. **Block weighting.** After standardisation, each block is rescaled by
   ``1/sqrt(n_features_in_block)`` (``weighting="equal"``). This gives every
   feature block the same total squared contribution to the distance, so the
   ablation comparison is not decided by whichever block happens to have the
   most features. ``weighting="uniform"`` (every feature equal) is available as
   a robustness check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .neurons import FeatureBlock, NeuronRepresentationSet
from .utils import Standardizer, sanitize_features


def _block_of(feature_name: str) -> str:
    return feature_name.split(".", 1)[0]


@dataclass
class RepresentationSpace:
    """A metric space of hidden neurons induced by a chosen set of feature blocks."""

    X_raw: np.ndarray
    feature_names: list[str]
    blocks: list[str] = field(default_factory=list)
    weighting: str = "equal"
    normalize_rows: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    # fitted state
    X: np.ndarray = field(init=False, repr=False, default=None)  # type: ignore[assignment]
    standardizer: Standardizer = field(init=False, repr=False, default_factory=Standardizer)
    block_scales: np.ndarray = field(init=False, repr=False, default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.X_raw = sanitize_features(np.asarray(self.X_raw, dtype=np.float64))
        if self.X_raw.ndim != 2:
            raise ValueError(f"X_raw must be 2-D, got shape {self.X_raw.shape}")
        self.feature_names = list(self.feature_names)
        if self.X_raw.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X_raw has {self.X_raw.shape[1]} columns but {len(self.feature_names)} feature names"
            )
        if not self.blocks:
            self.blocks = sorted({_block_of(n) for n in self.feature_names})
        self._fit()

    # ------------------------------------------------------------------
    # Fitting / transformation
    # ------------------------------------------------------------------
    def _fit(self) -> None:
        self.standardizer.fit(self.X_raw)
        Z = self.standardizer.transform(self.X_raw)
        if self.weighting == "equal":
            scales = np.ones(len(self.feature_names), dtype=np.float64)
            for block in sorted(set(_block_of(n) for n in self.feature_names)):
                idx = [i for i, n in enumerate(self.feature_names) if _block_of(n) == block]
                if idx:
                    scales[idx] = 1.0 / np.sqrt(len(idx))
            self.block_scales = scales
        elif self.weighting == "uniform":
            self.block_scales = np.ones(len(self.feature_names), dtype=np.float64)
        else:
            raise ValueError(f"Unknown weighting {self.weighting!r}; use 'equal' or 'uniform'")
        Z = Z * self.block_scales
        if self.normalize_rows:
            norms = np.linalg.norm(Z, axis=1, keepdims=True)
            Z = np.divide(Z, np.where(norms > 1e-12, norms, 1.0))
        Z = sanitize_features(Z)
        self.X = Z
        self.meta.setdefault("n_neurons", int(self.X.shape[0]))
        self.meta.setdefault("n_features", int(self.X.shape[1]))
        self.meta.setdefault("n_informative_features", int(self.standardizer.n_informative))
        self.meta.setdefault("weighting", self.weighting)
        self.meta.setdefault("blocks", list(self.blocks))

    @property
    def n_neurons(self) -> int:
        return int(self.X.shape[0])

    @property
    def constant_features(self) -> list[str]:
        if self.standardizer.constant_mask is None:
            return []
        return [n for n, c in zip(self.feature_names, self.standardizer.constant_mask) if c]

    def block_feature_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for name in self.feature_names:
            b = _block_of(name)
            out[b] = out.get(b, 0) + 1
        return out

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def distances(self, metric: str = "euclidean") -> np.ndarray:
        """Full ``(n_neurons, n_neurons)`` distance matrix in neuron space."""
        from scipy.spatial.distance import pdist, squareform

        if self.n_neurons < 2:
            return np.zeros((self.n_neurons, self.n_neurons))
        return squareform(pdist(self.X, metric=metric))

    def condensed(self, metric: str = "euclidean") -> np.ndarray:
        """Upper-triangular distance vector (length ``n(n-1)/2``)."""
        from scipy.spatial.distance import pdist

        if self.n_neurons < 2:
            return np.zeros(0, dtype=np.float64)
        return pdist(self.X, metric=metric)

    def pca(self, n_components: int = 2) -> dict[str, np.ndarray]:
        """PCA of the standardised representation matrix (visually descriptive only)."""
        from sklearn.decomposition import PCA

        n_components = int(min(n_components, min(self.X.shape)))
        pca = PCA(n_components=max(n_components, 1), random_state=0)
        scores = pca.fit_transform(self.X)
        return {
            "scores": scores,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "components": pca.components_,
            "mean": pca.mean_,
        }

    def feature_table(self) -> list[dict[str, float]]:
        """Raw features as a list of dicts (one per neuron), for CSV output."""
        return [dict(zip(self.feature_names, row)) for row in self.X_raw]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            X_raw=self.X_raw,
            X=self.X,
            feature_names=np.array(self.feature_names, dtype=object),
            block_scales=self.block_scales,
            mean=self.standardizer.mean,
            scale=self.standardizer.scale,
            constant_mask=self.standardizer.constant_mask,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RepresentationSpace":
        with np.load(Path(path), allow_pickle=True) as data:
            space = cls(
                X_raw=data["X_raw"],
                feature_names=[str(n) for n in data["feature_names"]],
            )
            space.X = data["X"]
            space.standardizer.mean = data["mean"]
            space.standardizer.scale = data["scale"]
            space.standardizer.constant_mask = data["constant_mask"]
            space.block_scales = data["block_scales"]
        return space

    def summary(self) -> dict[str, Any]:
        return {
            "n_neurons": self.n_neurons,
            "n_features": len(self.feature_names),
            "n_informative_features": self.standardizer.n_informative,
            "constant_features": self.constant_features,
            "block_feature_counts": self.block_feature_counts(),
            "weighting": self.weighting,
            "normalize_rows": bool(self.normalize_rows),
            "meta": self.meta,
        }


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------
def build_space_from_representations(
    reps: NeuronRepresentationSet,
    blocks: str | Iterable[str] | None = None,
    *,
    weighting: str = "equal",
    normalize_rows: bool = False,
    meta: Mapping | None = None,  # type: ignore[valid-type]
) -> RepresentationSpace:
    """Standard entry point: representation set + block selection -> metric space."""
    selected = FeatureBlock.coerce(blocks)
    X, names = reps.to_matrix(blocks=selected)
    if X.shape[1] == 0:
        raise ValueError(f"Selected blocks {selected} carry no features")
    info = dict(reps.meta)
    if meta:
        info.update(dict(meta))
    return RepresentationSpace(
        X_raw=X,
        feature_names=names,
        blocks=selected,
        weighting=weighting,
        normalize_rows=normalize_rows,
        meta=info,
    )


def random_baseline_space(
    n_neurons: int,
    n_features: int,
    *,
    seed: int = 0,
    weighting: str = "uniform",
    meta: Mapping | None = None,  # type: ignore[valid-type]
) -> RepresentationSpace:
    """Null representation of matched dimensionality with i.i.d. Gaussian features.

    This is the "Representation 1" of the ablation study: it has the same number
    of neurons and features as the real space but no structure whatsoever, so it
    calibrates how much geometry-function correlation is expected by chance.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_neurons, max(n_features, 1)))
    names = [f"random.f{i:03d}" for i in range(X.shape[1])]
    info = {"kind": "random_null", "seed": int(seed), "uses_labels": False, "uses_data": False}
    if meta:
        info.update(dict(meta))
    return RepresentationSpace(
        X_raw=X, feature_names=names, blocks=["random"], weighting=weighting, meta=info
    )


def shuffle_control_space(space: RepresentationSpace, *, seed: int = 0) -> RepresentationSpace:
    """Permute which neuron receives which feature vector.

    If the geometry-function relationship disappears under this permutation the
    relationship is genuinely carried by the features and not by an artefact of
    row ordering.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(space.n_neurons)
    return RepresentationSpace(
        X_raw=space.X_raw[perm],
        feature_names=list(space.feature_names),
        blocks=list(space.blocks),
        weighting=space.weighting,
        normalize_rows=space.normalize_rows,
        meta={**space.meta, "kind": "shuffled_control", "seed": int(seed), "permutation": perm.tolist()},
    )


def rate_only_space(activity_reps: NeuronRepresentationSet) -> RepresentationSpace:
    """One-dimensional firing-rate representation.

    This is the critical "trivial baseline" demanded by the project brief: it
    answers whether a multi-feature neuron space tells us anything that a single
    scalar firing rate does not already tell us.
    """
    X, names = activity_reps.to_matrix(blocks=[FeatureBlock.ACTIVITY.value])
    keep = [i for i, n in enumerate(names) if n.endswith(".log_rate_hz")]
    if not keep:
        keep = [i for i, n in enumerate(names) if n.endswith(".rate_hz")]
    if not keep:
        raise ValueError("Activity representation does not contain a firing-rate feature")
    sub = RepresentationSpace(
        X_raw=X[:, keep],
        feature_names=[names[i] for i in keep],
        blocks=[FeatureBlock.ACTIVITY.value],
        weighting="uniform",
        meta={"kind": "rate_only", "uses_labels": False},
    )
    return sub
