"""SHD (Spiking Heidelberg Digits) loading and event binning for V3.

The event-based structure of SHD is preserved: raw ``(time, channel)`` spike
events are binned onto a ``T x C`` grid; nothing is collapsed into a static
feature vector and no per-sample summary statistics are computed.

SHD layout (canonical PyTables form)
------------------------------------
* ``spikes/times``   object dataset, one float array of spike times [s] per sample
* ``spikes/units``   object dataset, one int array of channel ids [0, 700) per sample
* ``labels``         uint16, 20 classes (0-9 English digits, 10-19 German digits)
* ``extra/speaker``  uint16 speaker id

``train`` has 8156 samples over speakers {0,1,2,3,6,7,8,9,10,11}; the official
``test`` has 2264 samples over speakers {0..11} (4 and 5 never appear in train).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .config import V3Config


@dataclass
class Split:
    """Index arrays + bookkeeping for one split of the dataset."""

    name: str
    indices: np.ndarray
    labels: np.ndarray
    speakers: np.ndarray

    def __len__(self) -> int:  # pragma: no cover - trivial
        return int(self.indices.size)

    def summary(self) -> dict:
        counts = np.bincount(self.labels, minlength=20)
        return {
            "name": self.name,
            "n": int(self.indices.size),
            "classes_present": int((counts > 0).sum()),
            "speakers": sorted(int(s) for s in np.unique(self.speakers)),
            "class_counts": [int(c) for c in counts],
        }


class SHDEventStore:
    """Bins SHD events on demand, with an optional RAM cache of the binned codes.

    Only ``(bin, channel)`` pairs are kept; a batch is materialised on the GPU as
    a dense ``(B, T, C)`` tensor.  Two samples' events are never mixed.
    """

    def __init__(self, h5_path: str | Path, cfg: V3Config):
        self.path = Path(h5_path)
        if not self.path.exists():
            raise FileNotFoundError(f"SHD file not found: {self.path}")
        self.cfg = cfg
        self.n_bins = int(cfg.n_bins)
        self.n_inputs = int(cfg.n_inputs)
        self.bin_s = float(cfg.bin_ms) / 1000.0
        self.window_s = float(cfg.window_s)
        self.binary = bool(cfg.binary_input)
        self.cells = self.n_bins * self.n_inputs

        with h5py.File(self.path, "r") as fh:
            self.labels = np.asarray(fh["labels"], dtype=np.int64)
            self.speakers = np.asarray(fh["extra/speaker"], dtype=np.int64)
        self.n_samples = int(self.labels.size)

        self._fh: h5py.File | None = None
        self._cache: dict[int, np.ndarray] | None = {} if cfg.cache_events else None
        # bookkeeping filled in by stats()
        self.n_events_raw = 0
        self.n_events_kept = 0

    # ------------------------------------------------------------------ #
    def _file(self) -> h5py.File:
        if self._fh is None:
            self._fh = h5py.File(self.path, "r")
        return self._fh

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    # ------------------------------------------------------------------ #
    def sample_codes(self, i: int) -> np.ndarray:
        """Flat int32 codes ``bin * C + channel`` for one sample (deduplicated)."""
        if self._cache is not None and i in self._cache:
            return self._cache[i]
        fh = self._file()
        ts = np.asarray(fh["spikes/times"][i], dtype=np.float64)
        us = np.asarray(fh["spikes/units"][i], dtype=np.int64)
        if ts.size == 0:
            codes = np.zeros(0, dtype=np.int32)
        else:
            keep = ts < self.window_s
            ts = ts[keep]
            us = us[keep]
            bins = np.minimum((ts / self.bin_s).astype(np.int64), self.n_bins - 1)
            codes = (bins * self.n_inputs + us).astype(np.int32)
            if self.binary:
                codes = np.unique(codes)
        if self._cache is not None:
            self._cache[i] = codes
        return codes

    # ------------------------------------------------------------------ #
    def batch(self, indices: np.ndarray) -> np.ndarray:
        """Dense float32 input batch ``(B, T, C)`` (counts, or 0/1 when binary)."""
        B = len(indices)
        parts = []
        for b, i in enumerate(indices):
            c = self.sample_codes(int(i))
            if c.size:
                parts.append(c.astype(np.int64) + b * self.cells)
        if parts:
            flat = np.concatenate(parts)
            dense = np.bincount(flat, minlength=B * self.cells)
        else:
            dense = np.zeros(B * self.cells, dtype=np.int64)
        out = dense.reshape(B, self.n_bins, self.n_inputs)
        if self.binary:
            out = (out > 0).astype(np.float32)
        else:
            out = out.astype(np.float32)
        return out

    def labels_of(self, indices: np.ndarray) -> np.ndarray:
        return self.labels[np.asarray(indices, dtype=np.int64)]

    # ------------------------------------------------------------------ #
    def stats(self, sample: int = 2000, rng_seed: int = 0) -> dict:
        """Cheap dataset statistics (used by the notebook sanity-check cell)."""
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(self.n_samples, size=min(sample, self.n_samples), replace=False)
        fh = self._file()
        raw, kept, occ = 0, 0, 0.0
        for i in idx:
            ts = np.asarray(fh["spikes/times"][int(i)], dtype=np.float64)
            raw += ts.size
            kept += int((ts < self.window_s).sum())
            occ += self.sample_codes(int(i)).size
        return {
            "file": str(self.path),
            "n_samples": self.n_samples,
            "n_inputs": self.n_inputs,
            "n_bins": self.n_bins,
            "bin_ms": float(self.cfg.bin_ms),
            "window_ms": float(self.cfg.window_s * 1000),
            "events_sampled": int(raw),
            "events_inside_window": int(kept),
            "event_clip_fraction": float(1.0 - kept / max(raw, 1)),
            "occupied_cells_mean": float(occ / max(len(idx), 1)),
            "input_density": float(occ / max(len(idx), 1) / self.cells),
            "labels": [int(x) for x in np.bincount(self.labels, minlength=20)],
            "speakers": sorted(int(s) for s in np.unique(self.speakers)),
        }


# ---------------------------------------------------------------------- #
def stratified_split(
    labels: np.ndarray, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Class-stratified random split of the *training* file only.

    The official SHD test set is never involved.  This split is used for
    monitoring / model selection on the training distribution; it is **not**
    speaker-independent, which is documented as a limitation.
    """
    if val_fraction <= 0:
        return np.arange(labels.size), np.zeros(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        perm = rng.permutation(idx.size)
        n_val = int(round(val_fraction * idx.size))
        val_idx.extend(idx[perm[:n_val]].tolist())
        train_idx.extend(idx[perm[n_val:]].tolist())
    return (
        np.sort(np.asarray(train_idx, dtype=np.int64)),
        np.sort(np.asarray(val_idx, dtype=np.int64)),
    )


def make_split(store: SHDEventStore, cfg: V3Config) -> tuple[Split, Split]:
    """Return ``(fit, val)`` splits of the SHD training file."""
    tr, va = stratified_split(store.labels, cfg.val_fraction, cfg.split_seed)
    fit = Split("fit", tr, store.labels[tr], store.speakers[tr])
    val = Split("val", va, store.labels[va], store.speakers[va])
    return fit, val


def official_test_split(store: SHDEventStore) -> Split:
    """The official held-out SHD test set (read only, never used for selection)."""
    idx = np.arange(store.n_samples, dtype=np.int64)
    return Split("test", idx, store.labels[idx], store.speakers[idx])


class BatchIterator:
    """Deterministic, seeded minibatch iterator over a :class:`Split`."""

    def __init__(
        self,
        store: SHDEventStore,
        split: Split,
        batch_size: int,
        shuffle: bool,
        seed: int = 0,
        epoch: int = 0,
        drop_last: bool = False,
    ):
        self.store = store
        self.split = split
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.drop_last = bool(drop_last)

    def __len__(self) -> int:
        n = self.split.indices.size
        return n // self.batch_size if self.drop_last else int(np.ceil(n / self.batch_size))

    def __iter__(self):
        idx = self.split.indices
        if self.shuffle:
            rng = np.random.default_rng([self.seed, self.epoch])
            idx = idx[rng.permutation(idx.size)]
        for start in range(0, idx.size, self.batch_size):
            chunk = idx[start : start + self.batch_size]
            if chunk.size < self.batch_size and self.drop_last:
                continue
            x = self.store.batch(chunk)
            y = self.store.labels_of(chunk)
            yield x, y


__all__ = [
    "SHDEventStore",
    "Split",
    "BatchIterator",
    "stratified_split",
    "make_split",
    "official_test_split",
]
