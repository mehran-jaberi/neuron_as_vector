"""Raw-event augmentation for SHD, applied **before** the 2-ms binning.

The transform acts on the raw ``(time [s], channel)`` event stream of one
sample.  It never touches the binned ``(T, C)`` tensor, never touches labels and
never mutates its inputs (copies are returned).  The pipeline is::

    raw events -> channel dropout -> event dropout -> insertion
               -> global time shift -> timestamp jitter -> window filter
               -> 2-ms binning (v3.data)

The order is fixed and documented because the augmenters are not commutative
(e.g. inserted events are not subject to event dropout, and jitter is the last
perturbation before the window filter).  Every step draws from one
``numpy.random.Generator``, so a fixed seed/epoch gives a deterministic
realisation (the caller is responsible for seeding, see
:meth:`v3.data.SHDEventStore.set_epoch`).

Units and parameters
--------------------
* ``time_jitter``       per event: ``t <- t + N(0, std_ms^2)``; ``std_ms`` in ms.
* ``event_dropout``     per event: keep with probability ``1 - probability``.
* ``event_insertion``   add ``Poisson(rate_hz * window_s)`` events; ``rate_hz`` is
  **inserted events per second of the window** (1000 ms + 50 Hz -> ~50 events
  per sample).  Channels are uniform over ``[0, n_inputs)``, times uniform over
  ``[0, window_s)``.  Insertion happens after channel/event dropout, so inserted
  events are drawn from the full channel range, not only the kept channels.
* ``global_time_shift`` one shift per sample: ``t <- t + U(-max_ms, +max_ms)``;
  ``max_ms`` in ms, symmetric, sequence duration unchanged.
* ``channel_dropout``   one Bernoulli per input channel; every event of a dropped
  channel disappears.  A draw that would drop **all** channels is repaired by
  keeping one uniformly chosen channel, so the population signal never vanishes
  for that reason.

Window filter
-------------
The final step keeps exactly ``0 <= t < window_s``.  This is a guard, not a
formality: jitter/shift can push raw events to negative times, and the old
binning code would have turned those into negative channel indices and corrupted
the binned tensor.  Negative times are impossible in the output.
"""

from __future__ import annotations

import numpy as np

from .config import EventAugmentationConfig

__all__ = ["augment"]


def _resolve_config(cfg, window_s: float | None, n_inputs: int | None):
    """Accept either the augmentation block or the whole ``V3Config``."""
    if isinstance(cfg, EventAugmentationConfig):
        acfg = cfg
    else:
        acfg = getattr(cfg, "augmentation", None)
        if not isinstance(acfg, EventAugmentationConfig):
            raise TypeError(
                "cfg must be an EventAugmentationConfig or a V3Config, "
                f"got {type(cfg).__name__}"
            )
        if window_s is None:
            window_s = getattr(cfg, "window_s", None)
        if n_inputs is None:
            n_inputs = getattr(cfg, "n_inputs", None)
    if window_s is None or n_inputs is None:
        raise ValueError(
            "window_s and n_inputs must be provided when cfg is an "
            "EventAugmentationConfig (only the whole V3Config carries them)"
        )
    window_s = float(window_s)
    n_inputs = int(n_inputs)
    if window_s <= 0.0:
        raise ValueError(f"window_s must be > 0, got {window_s}")
    if n_inputs < 1:
        raise ValueError(f"n_inputs must be >= 1, got {n_inputs}")
    return acfg, window_s, n_inputs


def augment(
    times: np.ndarray,
    units: np.ndarray,
    cfg,
    rng: np.random.Generator,
    *,
    window_s: float | None = None,
    n_inputs: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment one sample's raw events and return ``(times, units)``.

    ``times`` are seconds (float64), ``units`` are channel ids (int64) in
    ``[0, n_inputs)``.  ``cfg`` is an :class:`EventAugmentationConfig` or a
    ``V3Config`` (which supplies ``window_s``/``n_inputs`` automatically).
    ``rng`` must be a ``numpy.random.Generator``.  With the master switch or every
    sub-augmenter disabled the inputs are returned unchanged (as copies).

    The returned arrays satisfy ``0 <= times < window_s``; the input arrays are
    never modified.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be numpy.random.Generator, got {type(rng).__name__}")
    acfg, window_s, n_inputs = _resolve_config(cfg, window_s, n_inputs)

    ts = np.array(times, dtype=np.float64, copy=True)
    us = np.array(units, dtype=np.int64, copy=True)
    if ts.ndim != 1 or us.ndim != 1 or ts.shape != us.shape:
        raise ValueError(
            f"times and units must be 1-D arrays of equal length, got {ts.shape} and {us.shape}"
        )
    if not acfg.active():
        return ts, us

    if us.size and (int(us.min()) < 0 or int(us.max()) >= n_inputs):
        raise ValueError(
            f"channel ids must be in [0, {n_inputs}), got [{int(us.min())}, {int(us.max())}]"
        )

    # 1. channel dropout: one Bernoulli per channel, never all channels.
    if acfg.channel_dropout.enabled:
        keep = rng.random(n_inputs) >= float(acfg.channel_dropout.probability)
        if not keep.any():  # pragma: no cover - only reachable for probability == 1
            keep[int(rng.integers(0, n_inputs))] = True
        if us.size:
            event_keep = keep[us]
            ts, us = ts[event_keep], us[event_keep]

    # 2. event dropout: independent Bernoulli per remaining event.
    if acfg.event_dropout.enabled:
        keep = rng.random(ts.size) >= float(acfg.event_dropout.probability)
        ts, us = ts[keep], us[keep]

    # 3. insertion: Poisson rate, uniform channel and time inside the window.
    if acfg.event_insertion.enabled:
        n_new = int(rng.poisson(float(acfg.event_insertion.rate_hz) * window_s))
        if n_new > 0:
            new_t = rng.uniform(0.0, window_s, size=n_new)
            new_u = rng.integers(0, n_inputs, size=n_new, dtype=np.int64)
            ts = np.concatenate([ts, new_t])
            us = np.concatenate([us, new_u])

    # 4. global time shift: one symmetric shift per sample.
    if acfg.global_time_shift.enabled:
        delta_s = float(rng.uniform(-acfg.global_time_shift.max_ms,
                                    acfg.global_time_shift.max_ms)) / 1000.0
        ts = ts + delta_s

    # 5. timestamp jitter: independent per event.
    if acfg.time_jitter.enabled:
        ts = ts + rng.normal(0.0, float(acfg.time_jitter.std_ms) / 1000.0, size=ts.size)

    # 6. window filter: strictly positive times only (see module docstring).
    keep = (ts >= 0.0) & (ts < window_s)
    return ts[keep], us[keep]
