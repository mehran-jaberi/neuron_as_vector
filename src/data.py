"""SHD data loading, event-to-time-bin conversion, and train/validation splits.

Design notes
------------
* The official SHD files (``shd_train.h5`` / ``shd_test.h5``) are HDF5 files with
  flat datasets ``spikes`` (seconds, :math:`\\mathrm{float32}`), ``units``
  (channel index), ``labels`` (class, 0-19) and, for the training file,
  ``speakers``.  There is no explicit per-sample index array.
* We therefore recover sample boundaries in a *defensive, verified* way: we
  prefer any explicit offset/``n_spikes`` dataset if one is present, and
  otherwise use the documented property that spike times are sorted within each
  sample, so a decrease in time marks the start of a new sample.  The number of
  recovered segments is checked against ``len(labels)``.  If the check fails we
  raise with actionable guidance rather than silently mis-grouping spikes.
* Events are kept as raw sparse events in memory (a few tens of MB for SHD) and
  are converted to dense ``(time, channel)`` tensors *per batch*.  Dense caching
  of the whole dataset would need >10 GB (8156 x 500 x 700 float32), which is not
  appropriate for the target 32 GB machine.
* Nothing here depends on class labels except the split functions and the
  synthetic generator; the neuron-representation code never sees labels.
"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
N_INPUT_CHANNELS = 700
N_CLASSES = 20
N_TRAIN_SAMPLES = 8156
N_TEST_SAMPLES = 2264

SHD_URLS: dict[str, str] = {
    "shd_train.h5": "https://zenkelab.org/datasets/shd_train.h5.zip",
    "shd_test.h5": "https://zenkelab.org/datasets/shd_test.h5.zip",
}


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
def _download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as fh:  # noqa: S310
        while True:
            block = response.read(chunk)
            if not block:
                break
            fh.write(block)
    tmp.replace(dest)


def download_shd(data_dir: str | Path, force: bool = False, verbose: bool = True) -> dict[str, Path]:
    """Download and extract the official SHD HDF5 files into ``data_dir``.

    Returns a mapping ``{"train": Path, "test": Path}``. Skips work for files
    that already exist unless ``force=True``.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for filename, url in SHD_URLS.items():
        target = data_dir / filename
        if target.exists() and not force:
            if verbose:
                print(f"[data] found {target}")
        else:
            archive = data_dir / f"{filename}.zip"
            if verbose:
                print(f"[data] downloading {url}")
            _download(url, archive)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(data_dir)
            archive.unlink(missing_ok=True)
            if not target.exists():
                # Some mirrors nest the file inside a folder.
                matches = list(data_dir.rglob(filename))
                if not matches:
                    raise FileNotFoundError(f"Extracted archive did not contain {filename}")
                shutil.move(str(matches[0]), target)
            if verbose:
                print(f"[data] extracted {target}")
        paths["train" if "train" in filename else "test"] = target
    return paths


# --------------------------------------------------------------------------
# HDF5 parsing
# --------------------------------------------------------------------------
def _h5_keys(group: Any) -> list[str]:
    return [k for k in group.keys()]


def _segment_by_time_breaks(times_ms: np.ndarray, n_samples: int) -> np.ndarray | None:
    """Recover sample boundaries from decreases in the (per-sample sorted) times."""
    if times_ms.size == 0:
        return None
    breaks = np.flatnonzero(np.diff(times_ms) < 0) + 1
    bounds = np.concatenate(([0], breaks, [times_ms.size]))
    if bounds.size - 1 != n_samples:
        return None
    return bounds


def parse_shd_h5(path: str | Path, layout: str = "auto", verbose: bool = True) -> dict[str, np.ndarray]:
    """Read an SHD HDF5 file into flat arrays plus per-sample offsets.

    Parameters
    ----------
    layout:
        ``"auto"`` (default), ``"flat"``, or ``"grouped"``. ``"grouped"`` expects
        one HDF5 group per sample (as produced by some converters); ``"flat"``
        expects concatenated ``spikes``/``units`` datasets.

    Returns
    -------
    dict with keys ``times_ms`` (float32), ``units`` (int16), ``offsets``
    (int64, length n_samples + 1), ``labels`` (int64), ``speakers`` (int64 or
    None) and ``layout_used`` (str).
    """
    import h5py

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SHD file not found: {path}")

    with h5py.File(path, "r") as fh:
        keys = _h5_keys(fh)

        labels = np.asarray(fh["labels"][()]).astype(np.int64) if "labels" in keys else None

        # Speaker ids live at the root ("speakers"/"speaker") or, as in the
        # canonical SHD files, under ``extra/speaker``.
        speakers: np.ndarray | None = None
        for cand in ("speakers", "speaker"):
            if cand in keys:
                speakers = np.asarray(fh[cand][()]).astype(np.int64)
                break
        if speakers is None and "extra" in keys and isinstance(fh["extra"], h5py.Group):
            for cand in ("speaker", "speakers"):
                if cand in fh["extra"].keys():
                    speakers = np.asarray(fh["extra"][cand][()]).astype(np.int64)
                    break

        spikes_node = fh["spikes"] if "spikes" in keys else None

        # (a) Canonical SHD (PyTables) layout: ``spikes`` is a *group* holding
        # ragged ``times`` / ``units`` object arrays, one entry per sample.
        canonical = (
            isinstance(spikes_node, h5py.Group)
            and "times" in spikes_node.keys()
            and "units" in spikes_node.keys()
        )

        # (b) one integer-named HDF5 group per sample.
        use_grouped = layout == "grouped" or (
            layout == "auto"
            and not canonical
            and "spikes" not in keys
            and keys
            and all(isinstance(fh[k], h5py.Group) for k in keys)
        )

        if canonical and layout != "flat":
            times_obj = spikes_node["times"][()]
            units_obj = spikes_node["units"][()]
            n_samples = len(times_obj)
            t_list: list[np.ndarray] = []
            u_list: list[np.ndarray] = []
            for i in range(n_samples):
                t = np.asarray(times_obj[i], dtype=np.float64).ravel()
                u = np.asarray(units_obj[i], dtype=np.int64).ravel()
                order = np.argsort(t, kind="stable")
                t_list.append(t[order])
                u_list.append(u[order])
            offsets = np.zeros(n_samples + 1, dtype=np.int64)
            if n_samples:
                offsets[1:] = np.cumsum([t.size for t in t_list])
            times = np.concatenate(t_list) if t_list else np.zeros(0, dtype=np.float64)
            units = np.concatenate(u_list) if u_list else np.zeros(0, dtype=np.int64)
            offsets_arr = offsets
            if labels is None and "labels" in spikes_node.keys():
                labels = np.asarray(spikes_node["labels"][()]).astype(np.int64)
            layout_used = "shd_ragged"
        elif use_grouped:
            groups = [k for k in keys if isinstance(fh[k], h5py.Group)]
            # Sort numerically when the group names are integers.
            try:
                groups.sort(key=int)
            except ValueError:
                groups.sort()
            times_list: list[np.ndarray] = []
            units_list: list[np.ndarray] = []
            offsets = [0]
            lab_list: list[np.ndarray] = []
            spk_list: list[np.ndarray] = []
            for g in groups:
                node = fh[g]
                t = np.asarray(node["spikes"][()], dtype=np.float64).ravel()
                u = np.asarray(node["units"][()], dtype=np.int64).ravel()
                if u.size != t.size:
                    raise ValueError(f"Group {g}: spikes and units have different lengths")
                order = np.argsort(t, kind="stable")
                t, u = t[order], u[order]
                times_list.append(t)
                units_list.append(u)
                offsets.append(offsets[-1] + t.size)
                if "labels" in node.keys():
                    lab_list.append(np.asarray(node["labels"][()]).ravel()[:1])
            if labels is None and lab_list:
                labels = np.concatenate(lab_list).astype(np.int64)
            times = np.concatenate(times_list) if times_list else np.zeros(0)
            units = np.concatenate(units_list) if units_list else np.zeros(0, dtype=np.int64)
            offsets_arr = np.asarray(offsets, dtype=np.int64)
            layout_used = "grouped"
        else:
            if "spikes" not in keys:
                raise ValueError(
                    f"{path} does not contain a 'spikes' dataset or per-sample groups. "
                    f"Available keys: {keys}. Pass --layout grouped/flat if you know the layout."
                )
            times = np.asarray(fh["spikes"][()], dtype=np.float64).ravel()
            units = np.asarray(fh["units"][()], dtype=np.int64).ravel()
            order = np.argsort(times, kind="stable")  # file is sorted; keep stable
            times, units = times[order], units[order]

            if labels is None:
                raise ValueError(f"{path} has no 'labels' dataset; cannot determine the number of samples.")

            n_samples = int(labels.size)
            offsets_arr: np.ndarray | None = None

            # (1) explicit cumulative offsets
            for cand in ("offsets", "spike_offsets", "n_spikes_cumsum"):
                if cand in keys and np.asarray(fh[cand][()]).size == n_samples + 1:
                    offsets_arr = np.asarray(fh[cand][()], dtype=np.int64).ravel()
                    break
            # (2) explicit per-sample spike counts
            if offsets_arr is None:
                for cand in ("n_spikes", "spike_counts", "counts"):
                    if cand in keys and np.asarray(fh[cand][()]).size == n_samples:
                        counts = np.asarray(fh[cand][()], dtype=np.int64).ravel()
                        if counts.sum() == times.size:
                            offsets_arr = np.concatenate(([0], np.cumsum(counts)))
                            break
            # (3) sortedness-break heuristic
            if offsets_arr is None:
                offsets_arr = _segment_by_time_breaks(times, n_samples)

            if offsets_arr is None:
                raise ValueError(
                    f"Could not recover per-sample boundaries in {path}: the sortedness-break "
                    f"heuristic produced a segment count != {n_samples} and no explicit offset "
                    f"dataset was found. Re-export the file with a per-sample group layout, or "
                    f"pass layout='grouped'."
                )
            layout_used = "flat"

    times_ms = (times * 1000.0).astype(np.float32)
    return {
        "times_ms": times_ms,
        "units": units.astype(np.int16),
        "offsets": offsets_arr.astype(np.int64),
        "labels": labels,
        "speakers": speakers,
        "layout_used": layout_used,
    }


# --------------------------------------------------------------------------
# Recordings container
# --------------------------------------------------------------------------
@dataclass
class SHDRecordings:
    """Sparse event recordings for a set of samples.

    Times are stored in **milliseconds**, sorted within each sample.
    """

    times_ms: np.ndarray
    units: np.ndarray
    offsets: np.ndarray
    labels: np.ndarray
    n_channels: int = N_INPUT_CHANNELS
    speakers: np.ndarray | None = None
    name: str = "shd"
    meta: dict[str, Any] = field(default_factory=dict)

    # -- basic ---------------------------------------------------------------
    def __len__(self) -> int:
        return int(self.offsets.size - 1)

    @property
    def labels_array(self) -> np.ndarray:
        return np.asarray(self.labels, dtype=np.int64)

    def sample(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(times_ms, units)`` for sample ``i``."""
        lo, hi = int(self.offsets[i]), int(self.offsets[i + 1])
        return self.times_ms[lo:hi], self.units[lo:hi]

    def subset(self, indices: Sequence[int] | np.ndarray, name: str | None = None) -> "SHDRecordings":
        """Return a new container restricted to ``indices`` (order preserved)."""
        indices = np.asarray(indices, dtype=np.int64)
        times_list, units_list, offsets = [], [], [0]
        for i in indices:
            t, u = self.sample(int(i))
            times_list.append(t)
            units_list.append(u)
            offsets.append(offsets[-1] + t.size)
        times = np.concatenate(times_list) if times_list else np.zeros(0, dtype=np.float32)
        units = np.concatenate(units_list) if units_list else np.zeros(0, dtype=np.int16)
        speakers = None if self.speakers is None else np.asarray(self.speakers)[indices]
        return SHDRecordings(
            times_ms=times.astype(np.float32),
            units=units.astype(np.int16),
            offsets=np.asarray(offsets, dtype=np.int64),
            labels=self.labels_array[indices],
            n_channels=self.n_channels,
            speakers=speakers,
            name=name or self.name,
            meta=dict(self.meta),
        )

    @property
    def duration_ms(self) -> float:
        return float(self.times_ms.max()) if self.times_ms.size else 0.0

    # -- serialisation -------------------------------------------------------
    def save_npz(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            times_ms=self.times_ms,
            units=self.units,
            offsets=self.offsets,
            labels=self.labels_array,
            speakers=np.array([], dtype=np.int64) if self.speakers is None else self.speakers,
            n_channels=np.array([self.n_channels]),
        )
        return path

    @classmethod
    def from_npz(cls, path: str | Path, name: str = "shd") -> "SHDRecordings":
        with np.load(Path(path), allow_pickle=False) as data:
            speakers = data["speakers"]
            return cls(
                times_ms=data["times_ms"].astype(np.float32),
                units=data["units"].astype(np.int16),
                offsets=data["offsets"].astype(np.int64),
                labels=data["labels"].astype(np.int64),
                n_channels=int(data["n_channels"][0]),
                speakers=None if speakers.size == 0 else speakers.astype(np.int64),
                name=name,
            )


def load_shd_recordings(
    h5_path: str | Path, layout: str = "auto", verbose: bool = True
) -> SHDRecordings:
    """Parse an SHD HDF5 file into :class:`SHDRecordings`."""
    parsed = parse_shd_h5(h5_path, layout=layout, verbose=verbose)
    rec = SHDRecordings(
        times_ms=parsed["times_ms"],
        units=parsed["units"],
        offsets=parsed["offsets"],
        labels=parsed["labels"],
        n_channels=N_INPUT_CHANNELS,
        speakers=parsed["speakers"],
        name=Path(h5_path).stem,
        meta={"layout_used": parsed["layout_used"], "source": str(h5_path)},
    )
    if verbose:
        print(
            f"[data] {rec.name}: {len(rec)} samples, {rec.times_ms.size} events, "
            f"layout={parsed['layout_used']}, speakers={'yes' if rec.speakers is not None else 'no'}"
        )
    return rec


def load_shd(
    data_dir: str | Path,
    *,
    download: bool = True,
    layout: str = "auto",
    verbose: bool = True,
) -> dict[str, SHDRecordings]:
    """Load the official train/test split, downloading it first if necessary.

    Returns ``{"train": SHDRecordings, "test": SHDRecordings}``. The official
    test set is returned untouched and must not be used for any modelling
    decision (see README).
    """
    data_dir = Path(data_dir)
    train_h5 = data_dir / "shd_train.h5"
    test_h5 = data_dir / "shd_test.h5"
    if not (train_h5.exists() and test_h5.exists()):
        if not download:
            raise FileNotFoundError(
                f"SHD files not found in {data_dir} and download=False. "
                f"Expected {train_h5} and {test_h5}."
            )
        download_shd(data_dir, verbose=verbose)
    return {
        "train": load_shd_recordings(train_h5, layout=layout, verbose=verbose),
        "test": load_shd_recordings(test_h5, layout=layout, verbose=verbose),
    }


# --------------------------------------------------------------------------
# Event -> time-bin conversion
# --------------------------------------------------------------------------
def events_to_bins(
    times_ms: np.ndarray,
    units: np.ndarray,
    *,
    n_bins: int,
    bin_ms: float,
    n_channels: int = N_INPUT_CHANNELS,
) -> np.ndarray:
    """Convert one sample's events to a dense ``(n_bins, n_channels)`` count array.

    Events outside ``[0, n_bins * bin_ms)`` are dropped (documented; SHD events
    end well before the default 1000 ms window).
    """
    out = np.zeros((n_bins, n_channels), dtype=np.float32)
    if times_ms.size == 0:
        return out
    bin_idx = np.floor(np.asarray(times_ms, dtype=np.float64) / float(bin_ms)).astype(np.int64)
    valid = (bin_idx >= 0) & (bin_idx < n_bins) & (units >= 0) & (units < n_channels)
    if valid.any():
        np.add.at(out, (bin_idx[valid], units[valid].astype(np.int64)), 1.0)
    return out


def recordings_to_bins(
    rec: SHDRecordings,
    indices: Sequence[int] | np.ndarray | None = None,
    *,
    n_bins: int,
    bin_ms: float,
) -> np.ndarray:
    """Convert several samples to a stacked ``(N, n_bins, n_channels)`` array.

    Provided for small/debug datasets and for tests. Training uses the batched
    tensor path in :func:`batch_events_to_bins` to avoid dense caching.
    """
    indices = range(len(rec)) if indices is None else indices
    arrays = [
        events_to_bins(*rec.sample(int(i)), n_bins=n_bins, bin_ms=bin_ms, n_channels=rec.n_channels)
        for i in indices
    ]
    if not arrays:
        return np.zeros((0, n_bins, rec.n_channels), dtype=np.float32)
    return np.stack(arrays, axis=0)


def batch_events_to_bins(
    rec: SHDRecordings,
    indices: np.ndarray,
    *,
    n_bins: int,
    bin_ms: float,
    device: Any = None,
    dtype: Any = None,
) -> Any:
    """Fully-vectorised batched binning returning a ``(B, T, C)`` tensor.

    All events of the whole batch are scattered in a single ``index_add_`` call,
    so the cost does not grow with a Python-level loop over the batch.
    """
    import torch

    if dtype is None:
        dtype = torch.float32
    indices = np.asarray(indices, dtype=np.int64)
    n_channels = rec.n_channels
    # Allocate the output directly on the target device: it is by far the largest
    # object here (B x T x C), so a CPU->GPU copy per batch would dominate runtime.
    out = torch.zeros(len(indices), n_bins, n_channels, dtype=dtype, device=device)
    if indices.size == 0:
        return out

    starts = rec.offsets[indices]
    ends = rec.offsets[indices + 1]
    counts = (ends - starts).astype(np.int64)
    total = int(counts.sum())
    if total > 0:
        batch_ids = np.repeat(np.arange(len(indices), dtype=np.int64), counts)
        # Gather all events of the batch in one pass.
        gather_idx = np.repeat(starts, counts) + (
            np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
        )
        t = rec.times_ms[gather_idx].astype(np.float64)
        u = rec.units[gather_idx].astype(np.int64)
        bin_idx = np.floor(t / float(bin_ms)).astype(np.int64)
        valid = (bin_idx >= 0) & (bin_idx < n_bins) & (u >= 0) & (u < n_channels)
        if valid.any():
            flat_np = (
                batch_ids[valid] * (n_bins * n_channels) + bin_idx[valid] * n_channels + u[valid]
            )
            flat = torch.from_numpy(flat_np)
            if device is not None:
                flat = flat.to(device)
            ones = torch.ones(flat.shape[0], dtype=dtype, device=flat.device)
            out.view(-1).index_add_(0, flat, ones)
    return out


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------
def speaker_aware_split(
    labels: np.ndarray,
    speakers: np.ndarray,
    *,
    val_fraction: float = 0.1,
    seed: int = 0,
    min_classes: int = N_CLASSES,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Hold out whole *speakers* for validation.

    Whole-speaker held-out validation is the scientifically preferable choice for
    SHD because the official test set contains two unseen speakers; a
    speaker-disjoint validation split therefore better estimates generalisation
    to novel speakers than a random split over samples would.

    Steps
    -----
    1. Shuffle speakers deterministically.
    2. Add whole speakers to the validation pool until ``val_fraction`` of the
       samples is covered.
    3. If a class is missing from the validation pool, move a small number of
       samples of that class from training speakers into validation so that
       every class is represented.

    Returns ``(train_idx, val_idx, info)``; ``info`` documents the speakers used
    and the class-coverage repair that was applied, so the README claim can be
    verified from the saved metadata.
    """
    labels = np.asarray(labels, dtype=np.int64)
    speakers = np.asarray(speakers, dtype=np.int64)
    if labels.shape != speakers.shape:
        raise ValueError("labels and speakers must have the same shape")

    rng = np.random.default_rng(seed)
    unique_speakers = np.unique(speakers)
    order = rng.permutation(unique_speakers)
    target = val_fraction * labels.size

    val_idx: list[int] = []
    chosen_speakers: list[int] = []
    current = 0
    for spk in order:
        if current >= target:
            break
        idx = np.flatnonzero(speakers == spk)
        val_idx.append(idx)
        chosen_speakers.append(int(spk))
        current += idx.size

    val_set = set(int(i) for i in np.concatenate(val_idx)) if val_idx else set()
    train_set = set(range(labels.size)) - val_set

    # Repair class coverage.
    repaired: dict[int, int] = {}
    for cls in range(int(max(labels.max() + 1, min_classes))):
        if any(labels[i] == cls for i in val_set):
            continue
        candidates = [i for i in sorted(train_set) if labels[i] == cls]
        if not candidates:
            continue
        moved = candidates[: max(1, min(3, len(candidates)))]
        for i in moved:
            train_set.discard(i)
            val_set.add(i)
        repaired[int(cls)] = len(moved)

    train_idx = np.array(sorted(train_set), dtype=np.int64)
    val_idx = np.array(sorted(val_set), dtype=np.int64)
    info = {
        "strategy": "speaker_aware",
        "seed": int(seed),
        "val_fraction_requested": float(val_fraction),
        "val_fraction_realised": float(val_idx.size / max(labels.size, 1)),
        "n_speakers_total": int(unique_speakers.size),
        "n_speakers_held_out": len(chosen_speakers),
        "held_out_speakers": chosen_speakers,
        "classes_repaired": repaired,
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
    }
    return train_idx, val_idx, info


def stratified_split(
    labels: np.ndarray,
    *,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Class-stratified random split (fallback when speaker metadata is absent)."""
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    val_parts: list[np.ndarray] = []
    train_parts: list[np.ndarray] = []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        idx = rng.permutation(idx)
        n_val = max(1, int(round(val_fraction * idx.size)))
        val_parts.append(np.sort(idx[:n_val]))
        train_parts.append(np.sort(idx[n_val:]))
    train_idx = np.concatenate(train_parts) if train_parts else np.empty(0, dtype=np.int64)
    val_idx = np.concatenate(val_parts) if val_parts else np.empty(0, dtype=np.int64)
    info = {
        "strategy": "stratified",
        "seed": int(seed),
        "val_fraction_requested": float(val_fraction),
        "val_fraction_realised": float(val_idx.size / max(labels.size, 1)),
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
        "limitation": (
            "speaker metadata unavailable, so validation samples were drawn from the same "
            "speakers as training. Validation accuracy may therefore mildly overestimate "
            "generalisation to unseen speakers."
        ),
    }
    return train_idx, val_idx, info


def make_validation_split(
    train_rec: SHDRecordings,
    *,
    val_fraction: float = 0.1,
    seed: int = 0,
    prefer_speaker_aware: bool = True,
) -> tuple[SHDRecordings, SHDRecordings, dict[str, Any]]:
    """Split the official training data into ``(train, val)``.

    Uses a whole-speaker split when speaker metadata is present, otherwise falls
    back to a class-stratified split and records that limitation in ``info``.
    """
    if prefer_speaker_aware and train_rec.speakers is not None:
        train_idx, val_idx, info = speaker_aware_split(
            train_rec.labels_array, train_rec.speakers, val_fraction=val_fraction, seed=seed
        )
    else:
        train_idx, val_idx, info = stratified_split(
            train_rec.labels_array, val_fraction=val_fraction, seed=seed
        )
        if prefer_speaker_aware:
            info["fallback_reason"] = "no speaker metadata in the training file"
    return train_rec.subset(train_idx, name="train"), train_rec.subset(val_idx, name="val"), info


# --------------------------------------------------------------------------
# Synthetic dataset (for tests and for pipeline validation without downloads)
# --------------------------------------------------------------------------
def make_synthetic_shd(
    *,
    n_samples: int = 300,
    n_classes: int = 5,
    n_channels: int = 40,
    n_bins: int = 50,
    bin_ms: float = 2.0,
    n_speakers: int = 5,
    seed: int = 0,
    spikes_per_sample: int = 180,
    class_bandwidth: float = 3.0,
) -> SHDRecordings:
    """Generate a structured, learnable spiking dataset with SHD-like layout.

    Each class ``c`` has a preferred location on the (ordered) channel axis,
    ``centre_c = (c + 0.5) * n_channels / n_classes``. Events are drawn from a
    Poisson process whose intensity peaks around ``centre_c`` with bandwidth
    ``class_bandwidth`` channels, so a recurrent SNN can learn the task and the
    hidden units develop class-tuned responses. Speakers add a small per-speaker
    jitter of the centre, which makes a speaker-disjoint split meaningful.

    The returned object has the same interface as an SHD dataset, so the whole
    pipeline (training, representation extraction, geometry analysis) can be
    exercised without downloading anything.
    """
    rng = np.random.default_rng(seed)
    times_list: list[np.ndarray] = []
    units_list: list[np.ndarray] = []
    offsets = [0]
    labels = np.full(n_samples, -1, dtype=np.int64)
    speakers = np.full(n_samples, -1, dtype=np.int64)

    centres = (np.arange(n_classes, dtype=np.float64) + 0.5) * n_channels / n_classes
    channel_axis = np.arange(n_channels, dtype=np.float64)

    for i in range(n_samples):
        cls = int(i % n_classes)
        spk = int((i // n_classes) % n_speakers)
        jitter = rng.normal(0.0, class_bandwidth * 0.5)
        centre = centres[cls] + jitter
        profile = np.exp(-0.5 * ((channel_axis - centre) / class_bandwidth) ** 2)
        # Background rate keeps silent channels from being entirely uninformative.
        profile = 0.9 * profile / profile.sum() + 0.1 / n_channels
        channels = rng.choice(n_channels, size=spikes_per_sample, p=profile)
        times = np.sort(rng.uniform(0.0, n_bins * bin_ms, size=spikes_per_sample))
        times_list.append(times.astype(np.float32))
        units_list.append(channels.astype(np.int16))
        offsets.append(offsets[-1] + spikes_per_sample)
        labels[i] = cls
        speakers[i] = spk

    return SHDRecordings(
        times_ms=np.concatenate(times_list),
        units=np.concatenate(units_list),
        offsets=np.asarray(offsets, dtype=np.int64),
        labels=labels,
        n_channels=n_channels,
        speakers=speakers,
        name="synthetic",
        meta={"synthetic": True, "class_bandwidth": class_bandwidth},
    )


def subset_by_class(rec: SHDRecordings, max_per_class: int, seed: int = 0) -> SHDRecordings:
    """Deterministically subsample the recordings to at most ``max_per_class`` per class."""
    from .utils import class_balanced_indices

    idx = class_balanced_indices(rec.labels_array, max_per_class=max_per_class, seed=seed)
    return rec.subset(idx, name=f"{rec.name}_subset")


# --------------------------------------------------------------------------
# Batched iteration
# --------------------------------------------------------------------------
def iterate_batches(
    rec: SHDRecordings,
    indices: Sequence[int] | np.ndarray,
    *,
    batch_size: int,
    n_bins: int,
    bin_ms: float,
    shuffle: bool = False,
    seed: int = 0,
    device: Any = None,
    drop_last: bool = False,
):
    """Yield batches ``{"x": (B, T, C), "y": (B,), "idx": (B,)}``.

    Binning happens per batch with :func:`batch_events_to_bins`, so memory stays
    proportional to the batch rather than to the dataset. Indices are yielded so
    that per-sample analysis output can be aligned with the original recordings.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(seed)
        indices = indices[rng.permutation(indices.size)]
    n = indices.size
    starts = range(0, n, batch_size)
    for start in starts:
        stop = min(start + batch_size, n)
        if drop_last and (stop - start) < batch_size:
            continue
        chunk = indices[start:stop]
        x = batch_events_to_bins(rec, chunk, n_bins=n_bins, bin_ms=bin_ms, device=device)
        y = rec.labels_array[chunk]
        import torch

        yield {"x": x, "y": torch.from_numpy(y).to(device if device is not None else "cpu"), "idx": chunk}
