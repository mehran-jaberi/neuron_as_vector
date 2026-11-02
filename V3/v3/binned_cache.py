"""One-off pre-binned dense input cache for SHD (performance, not science).

The training loop spends host time rebuilding the dense ``(B, T, C)`` input
tensor from event codes every batch (``np.bincount`` over a
``B * n_bins * n_inputs`` array).  :class:`BinnedCache` materialises that dense
tensor **once per file**, in file order, on disk as a raw ``uint8`` array
(``binary_input=True``) or ``uint16`` array (counts), and serves batches from a
``numpy.memmap`` instead.

Correctness contract
--------------------
The cache is built *through* :meth:`SHDEventStore.batch`, i.e. the current
on-the-fly implementation is the only producer of the values.  Per sample the
dense array contains only exact small integers (0/1, or counts), so the
round-trip ``float32 -> uint8/uint16 -> float32`` is exact and a cached batch is
bit-identical to the on-the-fly batch for the same indices in the same order
(asserted in ``V3/tests/test_binned_cache.py`` with ``torch.equal``).

Storage / layout
----------------
``V3/cache/binned/<key>/dense.<dt>``  raw little-endian dense array, shape
``(n_samples, n_bins, n_inputs)``, C order
``V3/cache/binned/<key>/meta.json``   written last; its presence marks a
complete cache (the directory is otherwise ignored / rebuilt).

``<key>`` hashes the binned layout plus the h5 path, size and mtime, so changing
``time_bin_ms``, ``sequence_duration_ms``, ``n_inputs``, ``binary_input`` or the
data file never silently mixes caches.  The directory is git-ignored and must
never be committed.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np

V3_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = V3_ROOT / "cache" / "binned"
ENV_ROOT = "V3_BINNED_CACHE_DIR"   # override the cache location (tests, other drives)
SCHEMA = "v3_binned_cache/v1"


def default_root() -> Path:
    """Cache root: ``$V3_BINNED_CACHE_DIR`` if set, else ``V3/cache/binned``."""
    raw = os.environ.get(ENV_ROOT)
    return Path(raw).expanduser() if raw else DEFAULT_ROOT


def _file_fingerprint(h5_path: str | Path) -> dict:
    st = Path(h5_path).stat()
    return {"path": str(Path(h5_path).resolve()), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def cache_key(h5_path: str | Path, cfg) -> str:
    """Stable key for (binned layout, data file) — see module docstring."""
    meta = {
        "schema": SCHEMA,
        "file": _file_fingerprint(h5_path),
        "n_bins": int(cfg.n_bins),
        "n_inputs": int(cfg.n_inputs),
        "binary_input": bool(cfg.binary_input),
        "window_s": float(cfg.window_s),
        "bin_s": float(cfg.bin_ms) / 1000.0,
    }
    return hashlib.sha1(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:16]


def dense_dtype(cfg) -> np.dtype:
    """0/1 input -> uint8; event counts -> uint16."""
    return np.dtype(np.uint8 if bool(cfg.binary_input) else np.uint16)


class BinnedCache:
    """Read path over a pre-binned dense memmap."""

    def __init__(self, directory: str | Path, meta: dict):
        self.dir = Path(directory)
        self.meta = dict(meta)
        self.n_samples = int(meta["n_samples"])
        self.n_bins = int(meta["n_bins"])
        self.n_inputs = int(meta["n_inputs"])
        self.dtype = np.dtype(meta["dtype"])
        self.path = self.dir / f"dense.{self.dtype.name}"
        self.shape = (self.n_samples, self.n_bins, self.n_inputs)
        expected = int(np.prod(self.shape)) * self.dtype.itemsize
        if not self.path.exists() or self.path.stat().st_size != expected:
            raise FileNotFoundError(
                f"incomplete binned cache {self.dir}: expected {expected} bytes in {self.path}"
            )
        self._mm = np.memmap(self.path, dtype=self.dtype, mode="r", shape=self.shape)

    # ------------------------------------------------------------------ #
    @property
    def nbytes(self) -> int:
        return self.path.stat().st_size

    def batch(self, indices: np.ndarray) -> np.ndarray:
        """Dense ``(B, T, C)`` float32 batch, bit-identical to ``store.batch``."""
        idx = np.asarray(indices, dtype=np.int64)
        out = np.asarray(self._mm[idx])
        # exact: the stored values are small integers, exactly representable in
        # float32; this is the same conversion the on-the-fly path performs.
        return out.astype(np.float32, copy=False)

    def close(self) -> None:
        self._mm = None

    def summary(self) -> dict:
        return {
            "dir": str(self.dir),
            "n_samples": self.n_samples,
            "shape": list(self.shape),
            "dtype": str(self.dtype),
            "nbytes": self.nbytes,
        }

    # ------------------------------------------------------------------ #
    @classmethod
    def open(cls, h5_path: str | Path, cfg, root: str | Path | None = None) -> "BinnedCache | None":
        """Open a complete cache for ``(h5_path, cfg)``; ``None`` if absent."""
        directory = Path(root or default_root()) / cache_key(h5_path, cfg)
        meta_path = directory / "meta.json"
        if not meta_path.exists():
            return None
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        return cls(directory, meta)

    @classmethod
    def build(
        cls,
        store,
        root: str | Path | None = None,
        force: bool = False,
        verbose: bool = True,
    ) -> "BinnedCache":
        """Materialise the cache from ``store`` (the authoritative on-the-fly path).

        Writes to a temporary file and renames it only after the dense array is
        complete; ``meta.json`` is written last, so a partial build is never
        mistaken for a complete cache.
        """
        h5_path, cfg = store.path, store.cfg
        directory = Path(root or default_root()) / cache_key(h5_path, cfg)
        existing = cls.open(h5_path, cfg, root=root)
        if existing is not None and not force:
            if verbose:
                print(f"[binned-cache] reusing {existing.path} ({existing.nbytes / 2**30:.2f} GiB)",
                      flush=True)
            return existing

        dt = dense_dtype(cfg)
        shape = (int(store.n_samples), int(cfg.n_bins), int(cfg.n_inputs))
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / f"dense.{dt.name}.tmp"
        final = directory / f"dense.{dt.name}"
        if tmp.exists():
            tmp.unlink()
        mm = np.memmap(tmp, dtype=dt, mode="w+", shape=shape)
        t0 = time.perf_counter()
        bs = max(1, int(getattr(cfg, "batch_size", 128)))
        for start in range(0, shape[0], bs):
            idx = np.arange(start, min(start + bs, shape[0]), dtype=np.int64)
            dense = store.batch(idx, include_tail=False)   # authoritative values,
            # without the empty silence tail: the cache stores the stimulus
            # window only, and SHDEventStore.batch appends the tail on read.
            if dt == np.uint8:
                block = (dense > 0).astype(np.uint8)
            else:
                mx = float(dense.max()) if dense.size else 0.0
                if mx > np.iinfo(dt).max:
                    raise ValueError(
                        f"event count {mx} does not fit {dt}; the counts cache is not lossless"
                    )
                block = dense.astype(dt)
            mm[start : start + idx.size] = block
            if verbose and (start // bs) % 16 == 0:
                done = start + idx.size
                print(
                    f"[binned-cache] {Path(h5_path).name}: {done}/{shape[0]} samples "
                    f"({100.0 * done / shape[0]:5.1f}%) {time.perf_counter() - t0:6.1f}s",
                    flush=True,
                )
        mm.flush()
        del mm
        os.replace(tmp, final)                              # atomic on Windows too
        meta = {
            "schema": SCHEMA,
            "key": cache_key(h5_path, cfg),
            "file": _file_fingerprint(h5_path),
            "n_samples": shape[0],
            "n_bins": shape[1],
            "n_inputs": shape[2],
            "binary_input": bool(cfg.binary_input),
            "dtype": dt.name,
            "window_s": float(cfg.window_s),
            "bin_ms": float(cfg.bin_ms),
            "dense_file": final.name,
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "build_seconds": round(time.perf_counter() - t0, 2),
        }
        with open(directory / "meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, sort_keys=True)
        if verbose:
            print(
                f"[binned-cache] built {final} ({final.stat().st_size / 2**30:.2f} GiB, "
                f"{meta['build_seconds']:.1f}s)",
                flush=True,
            )
        return cls(directory, meta)

    @classmethod
    def open_or_build(cls, store, root: str | Path | None = None, verbose: bool = True) -> "BinnedCache":
        cache = cls.open(store.path, store.cfg, root=root)
        if cache is not None:
            if verbose:
                print(f"[binned-cache] using {cache.path} ({cache.nbytes / 2**30:.2f} GiB)", flush=True)
            return cache
        return cls.build(store, root=root, verbose=verbose)


__all__ = ["BinnedCache", "cache_key", "dense_dtype", "DEFAULT_ROOT", "SCHEMA"]
