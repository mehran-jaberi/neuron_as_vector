"""Controlled imprecision for the neuron internal state ``z(t)``.

These are the only new scientific manipulations in the imprecision stage.  They
act on the **actual internal state used by the subsequent dynamics** -- i.e. right
after the leaky update inside ``VectorNeuronPopulation._segment`` -- so the
perturbed ``z`` feeds the next timestep's spike generation, population signal and
intra-neuron mixing.  They are *not* applied to the classifier input, to the
labels, or to the input data.

Design notes
------------
* The perturbed state stays bounded by the ``tanh`` bound ``|z| <= 1`` (a final
  ``clamp`` enforces it as a safety guard), so the fp16 path never overflows.
* Quantization uses a **symmetric, dynamic-range** quantizer with a
  straight-through estimator (STE): the forward pass is genuinely quantized but
  the gradient of the identity is passed through, so training stays end-to-end.
* Both operations are no-ops when the corresponding mode/phase is disabled, so the
  default configuration is bit-identical to the baseline.
"""

from __future__ import annotations

import torch

from .config import StateRegularizationConfig

PHASES = ("train", "val", "test")


def add_state_noise(z: torch.Tensor, std: float) -> torch.Tensor:
    """``z + eps`` with ``eps ~ N(0, std^2)``.  No-op when ``std <= 0``."""
    std = float(std)
    if std <= 0.0:
        return z
    noise = torch.randn(z.shape, device=z.device, dtype=torch.float32) * std
    return (z.float() + noise).to(z.dtype)


def quantize_state(z: torch.Tensor, bits: int, clip: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """Symmetric dynamic-range uniform quantizer with a straight-through gradient.

    Forward (genuinely quantized)::

        scale = clamp(max|z|, eps, clip)          # dynamic range, bounded by `clip`
        levels = 2**(bits-1) - 1                  # positive levels, so 2**bits total
        q     = round(clamp(z / scale * levels, -levels, levels))
        z_q   = q / levels * scale

    Backward: identity (STE), i.e. ``z + (z_q - z).detach()``.  This keeps the
    gradient path alive so the model remains trainable end-to-end.

    ``scale`` is detached, so the (non-differentiable) range estimate never
    contributes a gradient.  The range is bounded by ``clip`` (the ``tanh`` bound
    ``|z| <= 1`` is a hard mathematical invariant of the state update) and floored
    at ``eps`` so an all-zero / constant state cannot amplify float noise.
    """
    bits = int(bits)
    if bits >= 16:  # 16-bit range is not a meaningful restriction; keep exact
        return z
    levels = float(2 ** (bits - 1) - 1)
    if levels < 1.0:
        return z
    z32 = z.float()
    scale = z32.detach().abs().amax().clamp(min=float(eps)).clamp(max=float(clip))
    q = torch.round(z32 / scale * levels).clamp(-levels, levels)
    zq = q / levels * scale
    return (z32 + (zq - z32).detach()).to(z.dtype)


def regularize_state(z: torch.Tensor, reg: StateRegularizationConfig, phase: str) -> torch.Tensor:
    """Apply the configured perturbation to ``z`` for the given phase.

    Returns ``z`` unchanged (same object) when disabled, so the baseline is
    exactly preserved.
    """
    if not reg.phase_active(phase):
        return z
    out = z
    if reg.uses_quantization:
        out = quantize_state(out, reg.quantize_bits, reg.quantize_clip)
    if reg.uses_noise and float(reg.noise_std) > 0.0:
        out = add_state_noise(out, reg.noise_std)
    # preserve the tanh bound |z| <= 1 that the fp16 path relies on
    return out.clamp(-1.0, 1.0) if out is not z else z


__all__ = ["regularize_state", "add_state_noise", "quantize_state", "PHASES"]
