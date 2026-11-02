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
from .event_augmentation import augment


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
        # Optional silence tail: all-zero input bins appended after the stimulus
        # in batch() (never stored in the caches), so the recurrent dynamics can
        # settle before the spike-count readout.
        self.tail_steps = int(cfg.tail_steps)

        with h5py.File(self.path, "r") as fh:
            self.labels = np.asarray(fh["labels"], dtype=np.int64)
            self.speakers = np.asarray(fh["extra/speaker"], dtype=np.int64)
        self.n_samples = int(self.labels.size)

        self._fh: h5py.File | None = None
        # Raw-event augmentation produces a fresh realisation per epoch, so any
        # binned cache (RAM or dense) would freeze one realisation forever.
        # Bypass both caches when augmentation can be active; stats() reports it.
        self.augmentation = cfg.augmentation
        self._aug_requested_cache = bool(
            bool(getattr(cfg, "cache_events", True))
            or bool(getattr(cfg, "binned_cache", False))
        )
        self.cache_warning: str | None = None
        if self.augmentation.active() and self._aug_requested_cache:
            self.cache_warning = (
                "augmentation is enabled: cache_events/binned_cache are bypassed "
                "so every epoch gets a fresh realization"
            )
        self._cache: dict[int, np.ndarray] | None = (
            {} if (cfg.cache_events and not self.augmentation.active()) else None
        )
        # bookkeeping filled in by stats()
        self.n_events_raw = 0
        self.n_events_kept = 0
        # optional pre-binned dense cache (performance only; values identical).
        # Off by default: the on-the-fly path below stays the reference.
        self.binned: object | None = None
        if bool(getattr(cfg, "binned_cache", False)) and not self.augmentation.active():
            from .binned_cache import BinnedCache

            self.binned = BinnedCache.open_or_build(self)
        # augmentation gating: phase is the store-wide switch, the optional index
        # mask keeps a FIT store from augmenting the VAL samples it also serves.
        self.phase = "val"  # safe default: no augmentation outside training
        self.epoch = 0
        self._aug_rng: np.random.Generator | None = None
        self._aug_mask: np.ndarray | None = None
        self._aug_forced_off = False

    # ------------------------------------------------------------------ #
    def _file(self) -> h5py.File:
        if self._fh is None:
            self._fh = h5py.File(self.path, "r")
        return self._fh

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        if self.binned is not None:
            self.binned.close()
            self.binned = None

    # ------------------------------------------------------------------ #
    # raw events (the only place where augmentation may be applied)
    # ------------------------------------------------------------------ #
    def sample_events(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Raw events of one sample: ``(times [s], channel ids)``.

        Times are float64 seconds and channel ids int64 in ``[0, n_inputs)``;
        the arrays are returned as read, without window filtering or binning.
        """
        fh = self._file()
        ts = np.asarray(fh["spikes/times"][int(i)], dtype=np.float64)
        us = np.asarray(fh["spikes/units"][int(i)], dtype=np.int64)
        return ts, us

    def _bin_events(self, ts: np.ndarray, us: np.ndarray) -> np.ndarray:
        """Bin raw events to flat int32 codes; strictly ``0 <= t < window_s``.

        The lower bound is a guard: raw events pushed negative by a time shift or
        jitter must be dropped here, because a negative time would give a negative
        bin index and corrupt neighbouring channels in the bincount.
        """
        keep = (ts >= 0.0) & (ts < self.window_s)
        ts = ts[keep]
        us = us[keep]
        if ts.size == 0:
            return np.zeros(0, dtype=np.int32)
        bins = np.minimum((ts / self.bin_s).astype(np.int64), self.n_bins - 1)
        codes = (bins * self.n_inputs + us).astype(np.int32)
        if self.binary:
            codes = np.unique(codes)
        return codes

    def sample_codes(self, i: int) -> np.ndarray:
        """Flat int32 codes ``bin * C + channel`` for one sample (deduplicated).

        This is the *unaugmented* path; it is what the caches store.  During a
        training epoch with augmentation active, :meth:`batch` uses
        :meth:`_augmented_codes` instead.
        """
        if self._cache is not None and i in self._cache:
            return self._cache[i]
        ts, us = self.sample_events(i)
        codes = self._bin_events(ts, us)
        if self._cache is not None:
            self._cache[i] = codes
        return codes

    # ------------------------------------------------------------------ #
    # augmentation state
    # ------------------------------------------------------------------ #
    def set_phase(self, phase: str) -> None:
        """Store-wide phase for augmentation gating: ``train`` | ``val`` | ``test``.

        Only ``train`` can ever augment; the default is ``val`` so constructing a
        store and probing it cannot perturb data by accident.  The experiment
        additionally forces the TEST store off and masks FIT/VAL sample indices.
        """
        if phase not in ("train", "val", "test"):
            raise ValueError(f"unknown phase {phase!r} (expected train | val | test)")
        self.phase = phase

    def set_epoch(self, epoch: int) -> None:
        """Reseed the augmentation RNG deterministically for one epoch.

        ``default_rng(seed * 1000 + epoch)``: the same seed and epoch always give
        the same realisations; consecutive epochs give fresh ones.  Training and
        evaluation iterators both call this at the start of every pass.
        """
        self.epoch = int(epoch)
        self._aug_rng = np.random.default_rng(int(self.cfg.seed) * 1000 + self.epoch)

    def set_augmentation_indices(self, indices) -> None:
        """Restrict augmentation to these sample indices (use the FIT split)."""
        mask = np.zeros(self.n_samples, dtype=bool)
        idx = np.asarray(indices, dtype=np.int64)
        if idx.size:
            mask[idx] = True
        self._aug_mask = mask

    def force_augmentation_off(self, off: bool = True) -> None:
        """Hard off-switch (VAL/TEST stores); overrides every config flag."""
        self._aug_forced_off = bool(off)

    @property
    def augmentation_forced_off(self) -> bool:
        return self._aug_forced_off

    def augmentation_applies(self, index: int | None = None) -> bool:
        """Whether an augmented realisation is due for ``index`` right now."""
        if self._aug_forced_off or not self.augmentation.active():
            return False
        if self.phase != "train" or not self.augmentation.phase_active("train"):
            return False
        if index is not None and self._aug_mask is not None:
            return bool(self._aug_mask[int(index)])
        return True

    def _augmented_codes(self, i: int) -> np.ndarray:
        """Raw events -> augment -> bin, never cached."""
        if self._aug_rng is None:
            self.set_epoch(self.epoch)
        ts, us = self.sample_events(i)
        ts, us = augment(
            ts,
            us,
            self.augmentation,
            self._aug_rng,
            window_s=self.window_s,
            n_inputs=self.n_inputs,
        )
        return self._bin_events(ts, us)

    # ------------------------------------------------------------------ #
    def batch(self, indices: np.ndarray, include_tail: bool = True) -> np.ndarray:
        """Dense float32 input batch ``(B, T, C)`` (counts, or 0/1 when binary).

        ``T`` is ``timing.total_steps``: when ``timing.silence_tail_ms`` is
        non-zero, the last ``tail_steps`` bins are all-zero and the network keeps
        running after the stimulus in every phase.  ``include_tail=False``
        returns only the stimulus window (used to build the pre-binned cache, so
        the cache stays un-augmented and valid for any tail length).
        """
        if self.binned is not None and not any(
            self.augmentation_applies(int(i)) for i in indices
        ):
            # pre-binned cache: bit-identical values, built from this same path.
            # (augmentation active after the dense cache was built falls through
            # to the on-the-fly path rather than reuse frozen data)
            out = self.binned.batch(indices)
        else:
            B = len(indices)
            parts = []
            for b, i in enumerate(indices):
                if self.augmentation_applies(int(i)):
                    c = self._augmented_codes(int(i))
                else:
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
        if include_tail and self.tail_steps > 0:
            zeros = np.zeros(
                (out.shape[0], self.tail_steps, self.n_inputs), dtype=out.dtype
            )
            out = np.concatenate((out, zeros), axis=1)
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
            kept += int(((ts >= 0.0) & (ts < self.window_s)).sum())
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
            "cache_events_requested": bool(getattr(self.cfg, "cache_events", False)),
            "cache_events_active": self._cache is not None,
            "binned_cache_requested": bool(getattr(self.cfg, "binned_cache", False)),
            "binned_cache_active": self.binned is not None,
            "augmentation": self.augmentation.summary(),
            "warnings": [self.cache_warning] if self.cache_warning else [],
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
        # A fresh augmentation RNG per pass: same epoch -> identical realisations,
        # next epoch -> fresh ones.  Guarded because tests use lightweight stores.
        setter = getattr(self.store, "set_epoch", None)
        if callable(setter):
            setter(self.epoch)
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
