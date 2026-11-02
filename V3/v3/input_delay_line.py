"""Input delay line: K time-shifted copies of the already-binned SHD input.

Several strong SHD papers feed the network a delay line -- the input at time
``t`` is presented together with the input from a few strides earlier
(Nowotny 2025: 10 shifts x 30 ms; D'Agostino: fixed random delays).  This
module implements the fixed-stride, uniformly spaced version, applied to the
dense ``(B, T, C)`` tensor produced by :mod:`v3.data` *after* binning and
*before* the input projection ``w_in`` in :mod:`v3.model`.

Copy ``k`` (``k = 0 .. K-1``) is the input shifted right along the time axis by
``k * stride_bins`` bins::

    copy_k[b, t, c] = x[b, t - k*stride_bins, c]   if t >= k*stride_bins
                    = 0                            otherwise

so the first ``k * stride_bins`` steps of copy ``k`` are zero (zeros enter at
the boundary).  There is no re-binning of raw events and no random delay: the
delays are deterministic and equidistant, which keeps the configuration small
and the forward cheap.

The block is applied **once per batch**, before the temporal loop, and never
inside the gradient-checkpointed segment: the cost is one vectorised shift per
copy and one extra input buffer, not a per-timestep operation.

Modes (see :class:`v3.config.InputDelayLineConfig` for the parameter/memory
cost at K=4, C=700, D=1000):

``concat``
    Copies are stacked along the channel axis.  The model's ``w_in`` is
    ``(K*C, D)``, so each delay tap gets its own input weights: +(K-1)*C*D
    parameters (+2.1 M at K=4, C=700, D=1000).  The buffer is filled one whole
    copy at a time -- the forward never materialises a list of K shifted
    tensors -- and is consumed by a single matmul.
``sum``
    Copies are summed back into the C original channels.  ``w_in`` is
    unchanged: 0 extra parameters, a smoothing / fan-in effect.

``copies=1`` in either mode returns the input tensor itself (same object) and
the caller short-circuits to the original, unmodified forward path.
"""

from __future__ import annotations

import torch

from .config import InputDelayLineConfig


# ---------------------------------------------------------------------- #
def delay_line(
    x: torch.Tensor,
    *,
    copies: int,
    stride_bins: int,
    mode: str,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Stack/sum ``copies`` shifted copies of ``x`` (``(B, T, C)``).

    Parameters
    ----------
    x:
        already-binned input ``(B, T, C)``; never modified in place.
    copies:
        K >= 1.  ``K == 1`` returns ``x`` itself (identity).
    stride_bins:
        delay between successive copies, in whole time bins >= 0.
    mode:
        ``"concat"`` -> ``(B, T, K*C)``, ``"sum"`` -> ``(B, T, C)``.
    dtype:
        optional dtype for the concat buffer (e.g. the model's fp16 state
        dtype, which halves the buffer and avoids a full-size cast at the
        projection).  Ignored by ``sum``, which stays in ``x.dtype`` until the
        projection casts as usual.
    """
    if x.dim() != 3:
        raise ValueError(f"delay line expects a (B, T, C) tensor, got {tuple(x.shape)}")
    K = int(copies)
    stride = int(stride_bins)
    if K < 1:
        raise ValueError(f"copies must be >= 1, got {copies}")
    if stride < 0:
        raise ValueError(f"stride_bins must be >= 0, got {stride_bins}")
    if mode not in ("concat", "sum"):
        raise ValueError(f"mode must be 'concat' or 'sum', got {mode!r}")
    if K == 1:
        return x

    B, T, C = x.shape

    if mode == "sum":
        # K shifted copies summed into the C original channels.  `clone()` +
        # in-place adds on the leading-different slices keeps the peak at two
        # (B, T, C) buffers; the boundary stays zero because the slices only
        # ever add real values where `t - k*stride >= 0`.
        out = x.clone()
        for k in range(1, K):
            shift = k * stride
            if shift >= T:
                break
            out[:, shift:] = out[:, shift:] + x[:, : T - shift]
        return out

    # concat: one (B, T, K*C) buffer, filled one copy at a time.  This avoids
    # the `torch.cat([shift(x, k) for k in range(K)], -1)` peak of 2x the
    # delayed input; the only temporaries are per-copy casts of an x slice.
    out_dtype = x.dtype if dtype is None else dtype
    out = x.new_zeros((B, T, K * C), dtype=out_dtype)
    for k in range(K):
        shift = k * stride
        if shift >= T:
            break
        lo = k * C
        out[:, shift:, lo : lo + C] = x[:, : T - shift].to(out_dtype)
    return out


# ---------------------------------------------------------------------- #
def apply_input_delay_line(
    x: torch.Tensor,
    cfg: InputDelayLineConfig,
    *,
    bin_ms: float,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Config-driven entry point used by the model forward.

    Returns ``x`` itself when the block is disabled or ``copies == 1``, so the
    baseline path is bit-identical (the caller may then skip every delay-line
    tensor).  Otherwise returns the delayed input described by ``cfg``.
    """
    if not cfg.enabled or int(cfg.copies) <= 1:
        return x
    return delay_line(
        x,
        copies=int(cfg.copies),
        stride_bins=cfg.stride_bins(bin_ms),
        mode=cfg.mode,
        dtype=dtype,
    )


__all__ = ["delay_line", "apply_input_delay_line"]
