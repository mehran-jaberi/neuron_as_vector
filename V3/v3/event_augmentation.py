"""Raw-event augmentation for SHD, applied **before** the 2-ms binning.

The transform acts on the raw ``(time [s], channel)`` event stream of one
sample.  It never touches the binned ``(T, C)`` tensor, never touches labels and
never mutates its inputs (copies are returned).  The pipeline is::

    raw events -> same-class event blend -> channel dropout -> event dropout
               -> insertion -> global time shift -> timestamp jitter
               -> window filter -> 2-ms binning (v3.data)

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
* ``event_blend``       M7 same-class "event mixup".  With probability
  ``probability`` the sample is merged with a **same-class partner** supplied by
  the caller (:func:`blend_events`): both trains are shifted so their centres of
  mass meet at the midpoint, each spike of either source is kept independently
  with ``keep_probability``, and the merged train is clipped to the window.  It
  runs **first**, so every later step sees the merged train as one sample.  No
  partner (or ``probability == 0``) makes it a no-op that draws nothing.  Only the
  training store ever supplies a partner, and partners are FIT indices only.

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

__all__ = ["augment", "blend_events"]


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


def _check_event_arrays(times: np.ndarray, units: np.ndarray, what: str) -> None:
    if times.ndim != 1 or units.ndim != 1 or times.shape != units.shape:
        raise ValueError(
            f"{what}: times and units must be 1-D arrays of equal length, "
            f"got {times.shape} and {units.shape}"
        )


def blend_events(
    times_a: np.ndarray,
    units_a: np.ndarray,
    times_b: np.ndarray,
    units_b: np.ndarray,
    rng: np.random.Generator,
    *,
    window_s: float,
    keep_probability: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge two **same-class** raw event trains into one (M7 "event mixup").

    Steps, in this fixed order:

    1. **Centre-of-mass alignment.**  Each train is shifted by half the gap
       between the two mean event times, so both centres of mass land on the
       midpoint and coincide.  The shift is deterministic.
    2. **Per-spike keeping.**  Each spike of ``a`` (draws first) and then of ``b``
       is kept independently with ``keep_probability``.
    3. **Merge, sort, clip.**  The kept spikes are concatenated, ordered by time
       (stable sort) and clipped to ``0 <= t < window_s``.

    Keeping an empty source is undefined, so when either train is empty the
    *unmodified* ``a`` is returned (as copies): no blend, no draws.  Inputs are
    never mutated.  The class label is not involved at all -- the caller is
    responsible for choosing a same-class partner.
    """
    ta = np.array(times_a, dtype=np.float64, copy=True)
    ua = np.array(units_a, dtype=np.int64, copy=True)
    tb = np.array(times_b, dtype=np.float64, copy=True)
    ub = np.array(units_b, dtype=np.int64, copy=True)
    _check_event_arrays(ta, ua, "a")
    _check_event_arrays(tb, ub, "b")
    if not 0.0 <= float(keep_probability) <= 1.0:
        raise ValueError(f"keep_probability must be in [0, 1], got {keep_probability}")
    if ta.size == 0 or tb.size == 0:
        return ta, ua

    # 1. align the centres of mass on their midpoint (both trains move by half)
    target = 0.5 * (float(ta.mean()) + float(tb.mean()))
    ta = ta + (target - float(ta.mean()))
    tb = tb + (target - float(tb.mean()))

    # 2. keep each spike independently (a first, then b -- pinned draw order)
    keep_a = rng.random(ta.size) < float(keep_probability)
    keep_b = rng.random(tb.size) < float(keep_probability)
    ts = np.concatenate([ta[keep_a], tb[keep_b]])
    us = np.concatenate([ua[keep_a], ub[keep_b]])

    # 3. time order, then clip to the window
    order = np.argsort(ts, kind="stable")
    ts, us = ts[order], us[order]
    inside = (ts >= 0.0) & (ts < window_s)
    return ts[inside], us[inside]


def augment(
    times: np.ndarray,
    units: np.ndarray,
    cfg,
    rng: np.random.Generator,
    *,
    window_s: float | None = None,
    n_inputs: int | None = None,
    partner: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment one sample's raw events and return ``(times, units)``.

    ``times`` are seconds (float64), ``units`` are channel ids (int64) in
    ``[0, n_inputs)``.  ``cfg`` is an :class:`EventAugmentationConfig` or a
    ``V3Config`` (which supplies ``window_s``/``n_inputs`` automatically).
    ``rng`` must be a ``numpy.random.Generator``.  With the master switch or every
    sub-augmenter disabled the inputs are returned unchanged (as copies).

    ``partner`` is an optional **same-class** raw event train ``(times, units)``.
    When ``augmentation.event_blend`` is enabled it is aligned on its centre of
    mass and blended in with probability ``event_blend.probability`` (see
    :func:`blend_events`); the label is untouched.  Without a partner, or with
    ``probability == 0``, the blend step draws no random numbers at all and the
    output equals the un-blended pipeline.  The data store supplies the partner.

    The returned arrays satisfy ``0 <= times < window_s``; the input arrays are
    never modified.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be numpy.random.Generator, got {type(rng).__name__}")
    acfg, window_s, n_inputs = _resolve_config(cfg, window_s, n_inputs)

    ts = np.array(times, dtype=np.float64, copy=True)
    us = np.array(units, dtype=np.int64, copy=True)
    _check_event_arrays(ts, us, "times/units")
    if not acfg.active():
        return ts, us

    if us.size and (int(us.min()) < 0 or int(us.max()) >= n_inputs):
        raise ValueError(
            f"channel ids must be in [0, {n_inputs}), got [{int(us.min())}, {int(us.max())}]"
        )

    # 0. same-class event blend: overlay a same-class partner first, so every
    #    later step treats the merged event train as one sample.
    if acfg.event_blend.enabled and partner is not None:
        p_times, p_units = partner
        p_ts = np.array(p_times, dtype=np.float64, copy=True)
        p_us = np.array(p_units, dtype=np.int64, copy=True)
        _check_event_arrays(p_ts, p_us, "partner")
        if p_us.size and (int(p_us.min()) < 0 or int(p_us.max()) >= n_inputs):
            raise ValueError(
                f"partner channel ids must be in [0, {n_inputs}), "
                f"got [{int(p_us.min())}, {int(p_us.max())}]"
            )
        blend_p = float(acfg.event_blend.probability)
        # ``blend_p == 0`` draws nothing: an enabled block at probability 0 is
        # identical to the off path, RNG stream included.
        if blend_p > 0.0 and rng.random() < blend_p:
            ts, us = blend_events(
                ts, us, p_ts, p_us, rng,
                window_s=window_s,
                keep_probability=float(acfg.event_blend.keep_probability),
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
