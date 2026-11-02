"""V3 configuration: the single place where every experiment parameter lives.

V3 tests the *original* "C" version of the neuron-as-vector idea: a neuron's
internal state is genuinely ``D``-dimensional and the ``D`` dimensions
participate directly in its own recurrent dynamics, in its interaction with the
other neurons, and in spike generation.  There is no scalar neuron anywhere in
the vector model and no post-hoc embedding of anything.

The whole experiment is described by :class:`V3Config`.  Nothing in the model,
the data loader, the training loop or the notebook hard-codes ``N`` or ``D``.

Primary configuration (see ``configs/v3_default.yaml``)::

    N = 64     D = 1000     CUDA     float16
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

V3_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = V3_ROOT.parent

#: Spike surrogate gradients understood by :mod:`v3.model`.
#:
#: ``fast_sigmoid`` (default) is ``gamma / (1 + beta|u|)^2``, the derivative of
#: ``gamma*u / (1 + beta|u|)``; it is the recorded baseline and must stay
#: bit-identical.  ``slayer`` (M11a) is ``1 / (1 + alpha|u|)^2`` with its own
#: ``surrogate_alpha`` width whose default is stability-driven (see
#: ``V3Config.surrogate_alpha``): at ``alpha=2`` the far-field derivative is at
#: most 4x the baseline's, whereas the old ``alpha=0.5`` reached 63x and
#: overflowed the fp16 BPTT.
SURROGATE_KINDS = ("fast_sigmoid", "slayer")

#: Readout modes.  ``sum``/``mean`` are the historical count readouts;
#: ``temporal_attention`` is selected through the ``attention`` block instead
#: (M11b) and keeps the same classifier weights.
READOUTS = ("sum", "mean")


@dataclass
class StateRegularizationConfig:
    """Controlled imprecision applied to the neuron internal state ``z(t)``.

    This is **not** about the numerical datatype of the model.  ``V3Config.dtype``
    / ``amp`` decide how CUDA stores and multiplies numbers (fp16 state, fp32
    masters, autocast/GradScaler); this object decides whether the *internal
    vector state itself* is deliberately perturbed or quantized.

    Three separate knobs, kept explicit:

    1. training arithmetic dtype  -> ``V3Config.dtype`` / ``amp``
    2. precision of the state      -> ``quantize_bits`` here
    3. stochastic perturbation     -> ``noise_std`` here

    Modes
    -----
    ``none``               state untouched (baseline; default).
    ``noise``              ``z <- z + N(0, std^2)``.
    ``quantization``       ``z <- Q(z)`` symmetric straight-through quantizer.
    ``noise_quantization`` quantize first, then add noise.

    The ``apply_during_*`` flags choose the phase.  Default is **training only**:
    the perturbation acts as a training-time regularizer and inference (validation
    / the official test set) stays at full precision unless explicitly requested.
    """

    mode: str = "none"
    noise_std: float = 0.0        # 0.001 .. 0.05 for the screening sweep
    quantize_bits: int = 8        # 16 / 12 / 8 / 6 / 4 for the screening sweep
    quantize_clip: float = 1.0    # max magnitude of z (tanh bound; safety guard)
    apply_during_training: bool = True
    apply_during_validation: bool = False
    apply_during_test: bool = False

    MODES = ("none", "noise", "quantization", "noise_quantization")

    @property
    def uses_noise(self) -> bool:
        return self.mode in ("noise", "noise_quantization")

    @property
    def uses_quantization(self) -> bool:
        return self.mode in ("quantization", "noise_quantization")

    @property
    def noise_enabled(self) -> bool:
        """Noise is genuinely active (mode selects it and the std is positive)."""
        return self.uses_noise and float(self.noise_std) > 0.0

    @property
    def quantization_enabled(self) -> bool:
        return self.uses_quantization

    def phase_active(self, phase: str) -> bool:
        """Whether the perturbation is applied in ``phase`` (train | val | test)."""
        if self.mode == "none":
            return False
        if phase == "train":
            return bool(self.apply_during_training)
        if phase == "val":
            return bool(self.apply_during_validation)
        if phase == "test":
            return bool(self.apply_during_test)
        raise ValueError(f"unknown phase {phase!r} (expected train | val | test)")

    def validate(self) -> None:
        if self.mode not in self.MODES:
            raise ValueError(f"state_regularization.mode must be one of {self.MODES}, got {self.mode!r}")
        if float(self.noise_std) < 0.0:
            raise ValueError("state_regularization.noise_std must be >= 0")
        if not 2 <= int(self.quantize_bits) <= 16:
            raise ValueError("state_regularization.quantize_bits must be in [2, 16]")
        if float(self.quantize_clip) <= 0.0:
            raise ValueError("state_regularization.quantize_clip must be > 0")

    @classmethod
    def from_dict(cls, data: dict, strict: bool = False) -> "StateRegularizationConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown and strict:
            raise ValueError(f"unknown state_regularization keys: {sorted(unknown)}")
        return cls(**{k: v for k, v in data.items() if k in known})


def _block_from_dict(cls, data: dict, strict: bool, label: str):
    """Build a nested config block from a mapping (ignores unknown keys unless strict)."""
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown and strict:
        raise ValueError(f"unknown {label} keys: {sorted(unknown)}")
    return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class AdaptationConfig:
    """Spike-frequency adaptation channel of the vector neuron.

    Each neuron owns one scalar adaptive gain ``w_ad_i`` that reads a decaying
    spike-count trace ``b_i(t)`` and subtracts a recovery current from the
    spike-generation potential::

        pot_i(t) = <z_i(t), w_out_i> + theta_i - w_ad_i * b_i(t)
        s_i(t)   = H(pot_i(t))
        b_i(t+1) = rho * b_i(t) + s_i(t)

    ``b`` is a **state**, not a parameter: it is zero at the start of every
    sequence and is carried across the model's (gradient-checkpointed) time
    segments exactly like the vector state ``z``.  ``w_ad`` is initialised to
    ``gain_init`` (default 0.1, i.e. a tenth of a spike-trace unit of threshold
    shift, so the mechanism can actually move the threshold); with
    ``gain_init = 0`` the adaptive current ``w_ad * b`` vanishes identically and
    the *enabled* forward pass is bit-identical to the baseline, which is why 0
    left the first implementation inert.

    All defaults are off: with ``enabled = False`` neither the gain nor the
    buffer is registered and the forward pass is unchanged (bit-identical
    baseline).  ``rho`` must be in ``[0, 1)`` (``rho = 1`` would integrate
    without bound), ``learnable_gain = False`` registers ``w_ad`` as a fixed
    (non-trainable) buffer instead of a parameter.
    """

    enabled: bool = False
    rho: float = 0.95            # per-step decay: b <- rho * b + s
    gain_init: float = 0.1       # initial w_ad (0 => adaptation silent at t = 0)
    learnable_gain: bool = True  # False => w_ad is a buffer, no gradient

    def validate(self) -> None:
        if not 0.0 <= float(self.rho) < 1.0:
            raise ValueError("adaptation.rho must be in [0, 1)")
        if not math.isfinite(float(self.gain_init)):
            raise ValueError("adaptation.gain_init must be finite")

    @classmethod
    def from_dict(cls, data: dict, strict: bool = False) -> "AdaptationConfig":
        return _block_from_dict(cls, data, strict, "adaptation")


@dataclass
class StateDynamicsConfig:
    """Optional trainable state dynamics: per-neuron leak and initial state.

    Both features are **off by default**, so the baseline -- a fixed scalar leak
    ``alpha = dt/tau`` and ``z(0) = 0`` -- stays bit-identical unless requested.

    ``enabled`` is the master switch: the two sub-flags are inert while it is
    ``False``, so a sweep may stage them in an override file without changing
    behaviour.  When active:

    * ``trainable_alpha`` -- one free parameter per neuron, ``raw_i``, whose
      sigmoid is that neuron's leak ``alpha_i = sigmoid(raw_i)``.  ``raw_i`` is
      initialised to ``logit(alpha_init)``, so every neuron starts at exactly
      ``alpha_init``; the leak is then learned per neuron and multiplies the
      whole ``D``-dimensional state of that neuron.
    * ``learnable_z0`` -- the initial state ``z_i(0) in R^D`` becomes an
      ``(N, D)`` parameter, initialised exactly to zero.

    ``alpha_init`` replaces the scalar ``V3Config.alpha`` *only* when
    ``trainable_alpha`` is in force; the scalar field keeps its meaning for the
    baseline path and as the initialisation reference.
    """

    enabled: bool = False
    trainable_alpha: bool = False
    learnable_z0: bool = False
    alpha_init: float = 0.1

    @property
    def alpha_trainable(self) -> bool:
        """``trainable_alpha`` is actually in force (master switch on)."""
        return bool(self.enabled and self.trainable_alpha)

    @property
    def z0_learnable(self) -> bool:
        """``learnable_z0`` is actually in force (master switch on)."""
        return bool(self.enabled and self.learnable_z0)

    @property
    def active(self) -> bool:
        """At least one of the two trainable state features is in force."""
        return self.alpha_trainable or self.z0_learnable

    def alpha_init_logit(self) -> float:
        """``logit(alpha_init)``: the initial value of each raw alpha parameter."""
        p = float(self.alpha_init)
        return math.log(p / (1.0 - p))

    def validate(self) -> None:
        if not math.isfinite(float(self.alpha_init)):
            raise ValueError("state_dynamics.alpha_init must be finite")
        if self.alpha_trainable and not 0.0 < float(self.alpha_init) < 1.0:
            raise ValueError(
                "state_dynamics.alpha_init must be strictly between 0 and 1 when "
                f"trainable_alpha is enabled, got {self.alpha_init!r}"
            )

    @classmethod
    def from_dict(cls, data: dict, strict: bool = False) -> "StateDynamicsConfig":
        return _block_from_dict(cls, data, strict, "state_dynamics")


@dataclass
class InputDelayConfig:
    """Learnable per-input-channel conduction delays.

    One delay per input channel (``n_inputs`` channels; 700 for SHD) in
    milliseconds, applied to the binned input sequence *before* the shared input
    projection.  The delay is **fractional**: channel ``c``'s sequence is
    linearly interpolated between the two neighbouring bins, so the delay is
    differentiable with respect to a genuine time shift.

    The stored model parameter is the unconstrained pre-sigmoid value ``raw`` and
    the effective delay is ``max_ms * sigmoid(raw)``: it is strictly inside
    ``(0, max_ms)`` by construction, never sitting on a clamp boundary where the
    gradient would be zero.  Delays start uniformly inside
    ``[init_ms, max_ms / 2]`` (defaults ``[0.5, 10]`` ms, drawn from a dedicated
    seeded generator), so they are strictly interior from the first step.

    Defaults are off, so the input path is untouched and the baseline stays
    bit-identical (no extra parameter, same ``state_dict`` keys).

    ``learnable=False`` keeps the delays in the model (so a fixed delay map can
    be supplied with :meth:`v3.model.VectorNeuronPopulation.set_input_delays_ms`)
    but freezes them.
    """

    enabled: bool = False
    max_ms: float = 20.0
    learnable: bool = True
    init_ms: float = 0.5        # lower bound of the interior uniform init [ms]

    def validate(self) -> None:
        if float(self.max_ms) < 0.0:
            raise ValueError("input_delays.max_ms must be >= 0")
        if float(self.init_ms) < 0.0:
            raise ValueError("input_delays.init_ms must be >= 0")
        if self.enabled and float(self.max_ms) <= 0.0:
            # the forward map is delay = max_ms * sigmoid(raw)
            raise ValueError("input_delays.max_ms must be > 0 when enabled")
        if self.enabled and self.learnable:
            # a learnable delay initialised on a bound receives no gradient
            if not 0.0 < float(self.init_ms) < float(self.max_ms):
                raise ValueError(
                    "input_delays.init_ms must be strictly inside (0, max_ms) "
                    "for learnable delays"
                )

    @classmethod
    def from_dict(cls, data: dict, strict: bool = False) -> "InputDelayConfig":
        return _block_from_dict(cls, data, strict, "input_delays")


@dataclass
class AdlifConfig:
    """Two-timescale spike-triggered adaptation currents (adLIF-style).

    With ``enabled = True`` every neuron carries two scalar currents that are
    driven by its own spike train and subtracted from the pre-activation of the
    vector state update (inside the ``tanh``, i.e. as part of ``mix + drive``)::

        u_fast(t) = rho_fast * u_fast(t-1) + s(t-1)
        u_slow(t) = rho_slow * u_slow(t-1) + s(t-1)
        pre(t)    = mix(t) + drive(t) - gain_fast * u_fast(t) - gain_slow * u_slow(t)
        z(t+1)    = z(t) + alpha * (tanh(pre(t)) - z(t))

    ``s(t)`` is the spike vector generated from the pre-update state, so the
    current entering step ``t`` is driven by the *previous* step's spike and the
    first step has ``u = 0``.  ``rho_fast = 0.9`` / ``rho_slow = 0.99`` are the
    per-step leaks (one bin == one dynamical step); the gains are one learnable
    scalar per neuron each and start at ``gain_init = 0``, so the mechanism is
    exactly inert at initialisation while the currents still accumulate from
    spikes -- from the second step on the currents are non-zero, which is what
    gives the gains a non-zero gradient (a gain that only multiplied a zero
    current would be frozen; see the M1 history).

    ``use_fast`` / ``use_slow`` enable the two timescales separately (at least
    one must be on when ``enabled = True``).  The default is **off**: with
    ``enabled = False`` no current and no parameter exists and the baseline
    update rule / ``state_dict`` are untouched.
    """

    enabled: bool = False
    use_fast: bool = True
    use_slow: bool = True
    rho_fast: float = 0.9
    rho_slow: float = 0.99
    gain_init: float = 0.0

    def validate(self) -> None:
        if self.enabled and not (bool(self.use_fast) or bool(self.use_slow)):
            raise ValueError("adlif.enabled requires adlif.use_fast and/or adlif.use_slow")
        if not 0.0 <= float(self.rho_fast) < 1.0:
            raise ValueError("adlif.rho_fast must be in [0, 1)")
        if not 0.0 <= float(self.rho_slow) < 1.0:
            raise ValueError("adlif.rho_slow must be in [0, 1)")
        if not math.isfinite(float(self.gain_init)):
            raise ValueError("adlif.gain_init must be finite")

    @classmethod
    def from_dict(cls, data: dict, strict: bool = False) -> "AdlifConfig":
        return _block_from_dict(cls, data, strict, "adlif")


@dataclass
class TwoStageConfig:
    """A second, independent rank-R mixing stage inside one timestep (M12).

    Off by default.  With ``enabled = False`` no parameter is created and the
    timestep update is exactly the single-mixer baseline, so the default model is
    bit-identical.  With ``enabled = True`` each step becomes::

        mix2  = (z @ w_down2) @ w_up2       rank R2, independent of w_down / w_up
        inner = tanh(mix2 + drive)
        z    <- z + alpha * (tanh(mix + lam * inner) - z)

    The gain is **not** a free scalar.  The optimised quantity is a raw scalar
    ``lam_raw`` and the gain is::

        lam = lam_min + (lam_max - lam_min) * sigmoid(lam_raw)

    with ``0 < lam_min < lam_init < lam_max``.  ``lam_raw`` is warm-started at
    ``logit((lam_init - lam_min) / (lam_max - lam_min))``, so the stage starts at
    ``lam = lam_init`` (1.0 by default).  Since ``lam_min > 0`` and the sigmoid
    lies in ``(0, 1)``, ``lam`` is always finite and inside
    ``(lam_min, lam_max)``: training can scale the inner drive, but it can never
    multiply it away (the previous free ``lam`` could reach 0 or turn negative
    and silently switch the stage -- and the spikes -- off).

    ``w_down2`` is initialised **exactly like** ``w_down``
    (``randn(D, R2) / sqrt(D)``, same generator), and ``w_up2`` like ``w_up``
    (``randn(R2, D) * 0.5 / sqrt(R2)``), so the inner stage starts with the same
    mixing gain as the outer one: a zero or scaled-down init would leave it --
    and ``lam`` -- inert from step 0.
    """

    enabled: bool = False
    rank: int = 64
    lam_min: float = 0.1
    lam_max: float = 2.0
    lam_init: float = 1.0

    @property
    def active(self) -> bool:
        """The second mixing stage is in force."""
        return bool(self.enabled)

    def validate(self) -> None:
        if int(self.rank) < 1:
            raise ValueError("two_stage.rank must be >= 1")
        for name in ("lam_min", "lam_init", "lam_max"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"two_stage.{name} must be finite")
        if float(self.lam_min) <= 0.0:
            raise ValueError("two_stage.lam_min must be > 0")
        if not float(self.lam_min) < float(self.lam_init) < float(self.lam_max):
            raise ValueError("two_stage must satisfy lam_min < lam_init < lam_max")

    @classmethod
    def from_dict(cls, data: dict, strict: bool = False) -> "TwoStageConfig":
        return _block_from_dict(cls, data, strict, "two_stage")


@dataclass
class TimeJitterConfig:
    """Per-event timestamp jitter: ``t_i' = t_i + eps_i``, ``eps_i ~ N(0, std_ms^2)``."""

    enabled: bool = False
    std_ms: float = 0.5

    def validate(self) -> None:
        if float(self.std_ms) < 0.0:
            raise ValueError("augmentation.time_jitter.std_ms must be >= 0")

    @classmethod
    def from_dict(cls, data: dict, strict: bool = False) -> "TimeJitterConfig":
        return _block_from_dict(cls, data, strict, "augmentation.time_jitter")


@dataclass
class EventDropoutConfig:
    """Drop each raw event independently with ``probability`` (labels are preserved)."""

    enabled: bool = False
    probability: float = 0.1

    def validate(self) -> None:
        if not 0.0 <= float(self.probability) <= 1.0:
            raise ValueError("augmentation.event_dropout.probability must be in [0, 1]")

    @classmethod
    def from_dict(cls, data: dict, strict: bool = False) -> "EventDropoutConfig":
        return _block_from_dict(cls, data, strict, "augmentation.event_dropout")


@dataclass
class EventInsertionConfig:
    """Insert Poisson(``rate_hz * window_s``) artificial events on valid SHD channels.

    ``rate_hz`` is the **inserted events per second of the window** (not a
    probability and not a count): a 1000 ms window with ``rate_hz: 50`` inserts
    50 events per sample on average.
    """

    enabled: bool = False
    rate_hz: float = 50.0

    def validate(self) -> None:
        if float(self.rate_hz) < 0.0:
            raise ValueError("augmentation.event_insertion.rate_hz must be >= 0")

    @classmethod
    def from_dict(cls, data: dict, strict: bool = False) -> "EventInsertionConfig":
        return _block_from_dict(cls, data, strict, "augmentation.event_insertion")


@dataclass
class GlobalTimeShiftConfig:
    """Shift every timestamp by ``U(-max_ms, +max_ms)`` (sequence duration unchanged)."""

    enabled: bool = False
    max_ms: float = 100.0

    def validate(self) -> None:
        if float(self.max_ms) < 0.0:
            raise ValueError("augmentation.global_time_shift.max_ms must be >= 0")

    @classmethod
    def from_dict(cls, data: dict, strict: bool = False) -> "GlobalTimeShiftConfig":
        return _block_from_dict(cls, data, strict, "augmentation.global_time_shift")


@dataclass
class ChannelDropoutConfig:
    """Drop whole input channels for one sample (one Bernoulli per channel).

    The all-channels-dropped outcome is impossible by construction: if every
    channel were drawn as dropped, one channel is kept at random.
    """

    enabled: bool = False
    probability: float = 0.1

    def validate(self) -> None:
        if not 0.0 <= float(self.probability) <= 1.0:
            raise ValueError("augmentation.channel_dropout.probability must be in [0, 1]")

    @classmethod
    def from_dict(cls, data: dict, strict: bool = False) -> "ChannelDropoutConfig":
        return _block_from_dict(cls, data, strict, "augmentation.channel_dropout")


@dataclass
class EventBlendConfig:
    """Same-class raw-event blend ("event mixup"): two samples of one class merged.

    With probability ``probability`` a training sample is merged with a randomly
    chosen **same-class** partner from the FIT split: both event trains are
    shifted so their centres of mass (mean event time) coincide, each spike of
    either source is kept independently with ``keep_probability``, and the merged
    train is clipped to the window.  The label is unchanged.

    Partners come from an index the data store builds **once** on the training
    split, so a VAL or TEST sample can never become (or receive) a partner.
    Disabled by default; ``probability: 0`` draws no random numbers at all, so
    the enabled-but-zero path is identical to the off path.
    """

    enabled: bool = False
    probability: float = 0.5
    keep_probability: float = 0.5

    def validate(self) -> None:
        if not 0.0 <= float(self.probability) <= 1.0:
            raise ValueError("augmentation.event_blend.probability must be in [0, 1]")
        if not 0.0 <= float(self.keep_probability) <= 1.0:
            raise ValueError("augmentation.event_blend.keep_probability must be in [0, 1]")

    @classmethod
    def from_dict(cls, data: dict, strict: bool = False) -> "EventBlendConfig":
        return _block_from_dict(cls, data, strict, "augmentation.event_blend")


@dataclass
class EventAugmentationConfig:
    """Raw-event augmentation, applied **before** the 2-ms binning.

    The transform operates on the raw ``(time, channel)`` event stream, never on
    the binned ``(T, C)`` tensor.  Fixed operation order (see
    :func:`v3.event_augmentation.augment`)::

        same-class event blend -> channel dropout -> event dropout -> insertion
        -> global time shift -> timestamp jitter -> window filter

    ``enabled`` is the master switch and every sub-block has its own ``enabled``
    flag; **all defaults are off**, so the unaugmented path is bit-identical to
    the baseline.  The ``apply_during_*`` flags choose the phase exactly like
    :class:`StateRegularizationConfig`; default is training only, and the
    experiment additionally forces VAL/TEST stores off.

    ``event_blend`` is the only op that needs another sample: the data store
    supplies a same-class FIT partner, otherwise it is a no-op.
    """

    enabled: bool = False
    event_blend: EventBlendConfig = field(default_factory=EventBlendConfig)
    time_jitter: TimeJitterConfig = field(default_factory=TimeJitterConfig)
    event_dropout: EventDropoutConfig = field(default_factory=EventDropoutConfig)
    event_insertion: EventInsertionConfig = field(default_factory=EventInsertionConfig)
    global_time_shift: GlobalTimeShiftConfig = field(default_factory=GlobalTimeShiftConfig)
    channel_dropout: ChannelDropoutConfig = field(default_factory=ChannelDropoutConfig)
    apply_during_training: bool = True
    apply_during_validation: bool = False
    apply_during_test: bool = False

    AUGMENTERS = (
        "event_blend",
        "time_jitter",
        "event_dropout",
        "event_insertion",
        "global_time_shift",
        "channel_dropout",
    )

    @property
    def any_augmenter_enabled(self) -> bool:
        """At least one sub-augmenter is switched on (the master switch aside)."""
        return any(getattr(self, name).enabled for name in self.AUGMENTERS)

    def active(self) -> bool:
        """The master switch is on and at least one sub-augmenter is enabled."""
        return bool(self.enabled) and self.any_augmenter_enabled

    def phase_active(self, phase: str) -> bool:
        """Whether the augmentation is applied in ``phase`` (train | val | test)."""
        if not self.active():
            return False
        if phase == "train":
            return bool(self.apply_during_training)
        if phase == "val":
            return bool(self.apply_during_validation)
        if phase == "test":
            return bool(self.apply_during_test)
        raise ValueError(f"unknown phase {phase!r} (expected train | val | test)")

    def validate(self) -> None:
        for name in self.AUGMENTERS:
            getattr(self, name).validate()

    def summary(self) -> dict:
        """Plain-dict snapshot used by ``SHDEventStore.stats`` / logs."""
        return {
            "enabled": bool(self.enabled),
            "active": self.active(),
            "apply_during_training": bool(self.apply_during_training),
            "apply_during_validation": bool(self.apply_during_validation),
            "apply_during_test": bool(self.apply_during_test),
            **{
                name: asdict(getattr(self, name))
                for name in self.AUGMENTERS
            },
        }

    @classmethod
    def from_dict(cls, data: dict, strict: bool = False) -> "EventAugmentationConfig":
        block_classes = {
            "event_blend": EventBlendConfig,
            "time_jitter": TimeJitterConfig,
            "event_dropout": EventDropoutConfig,
            "event_insertion": EventInsertionConfig,
            "global_time_shift": GlobalTimeShiftConfig,
            "channel_dropout": ChannelDropoutConfig,
        }
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown and strict:
            raise ValueError(f"unknown augmentation keys: {sorted(unknown)}")
        kwargs = {}
        for k, v in data.items():
            if k not in known:
                continue
            if k in block_classes and isinstance(v, dict):
                v = block_classes[k].from_dict(v, strict=strict)
            kwargs[k] = v
        return cls(**kwargs)


@dataclass
class AttentionConfig:
    """Learned temporal attention over the readout (M11b).

    A spike-count readout collapses all ``T`` bins into ``(B, N)``; the linear
    classifier therefore cannot weight *when* a spike happened.  When this block
    is enabled the model keeps per-bin counts ``c_t`` (B, T, N) and combines them
    with a learned probability distribution over bins::

        a = softmax(attn_logits / temperature)        # (T,), sums to 1
        counts_att = sum_t (T * a_t) * c_t            # (B, N), count scale
        logits = counts_att @ W_cls

    The ``T *`` factor restores the plain spike-count scale: without it the
    uniform distribution would give ``1/T`` of the historical count readout
    (T=500), suppress every classifier gradient by ~500x and leave the
    zero-initialised readout stuck.  With it, uniform attention reproduces the
    plain count sum and the classifier starts with the baseline gradient
    (M11b), while the attention logits see the full, un-suppressed gradient as
    soon as ``W_cls`` leaves zero.

    The distribution is parameterised directly by one logit per bin (``T``
    parameters, no hidden MLP) and the logits are zero-initialised, so the
    initial forward reuses the plain spike-count readout.  ``temperature``
    sharpens (``< 1``) or flattens (``> 1``) the distribution.

    ``enabled=False`` (the default) leaves the readout exactly as before and
    adds **no parameters** to the model.
    """

    enabled: bool = False
    temperature: float = 1.0

    def validate(self) -> None:
        if float(self.temperature) <= 0.0:
            raise ValueError("attention.temperature must be > 0")

    @classmethod
    def from_dict(cls, data: dict, strict: bool = False) -> "AttentionConfig":
        return _block_from_dict(cls, data, strict, "attention")


@dataclass
class TimingConfig:
    """Temporal discretization of the SHD experiment.

    Two independent quantities define the timing; the number of dynamical steps
    follows from them::

        num_time_steps = sequence_duration_ms / time_bin_ms     (must be an integer)

    ``time_bin_ms`` is simultaneously the SHD bin width **and** the model's
    integration step ``dt`` (one bin == one dynamical step), so the leak factor
    ``alpha = dt/tau`` changes with it: 2 ms -> ``alpha = 2/tau``,
    4 ms -> ``alpha = 4/tau``.

    Defaults are the **2 ms** discretization
    (1000 ms / 2 ms -> 500 steps, ``alpha = 0.1`` at ``tau = 20 ms``).  The
    historical 4 ms reference (250 steps, ``alpha = 0.2``) is still available by
    setting ``time_bin_ms: 4``.
    """

    sequence_duration_ms: float = 1000.0
    time_bin_ms: float = 2.0

    @property
    def num_time_steps(self) -> int:
        return int(round(float(self.sequence_duration_ms) / float(self.time_bin_ms)))

    @property
    def simulation_dt_ms(self) -> float:
        return float(self.time_bin_ms)

    @property
    def duration_s(self) -> float:
        return float(self.sequence_duration_ms) / 1000.0

    def validate(self) -> None:
        if float(self.time_bin_ms) <= 0:
            raise ValueError("timing.time_bin_ms must be > 0")
        if float(self.sequence_duration_ms) <= 0:
            raise ValueError("timing.sequence_duration_ms must be > 0")
        steps = float(self.sequence_duration_ms) / float(self.time_bin_ms)
        if abs(steps - round(steps)) > 1e-9:
            raise ValueError(
                "timing.sequence_duration_ms / time_bin_ms must be an integer number of "
                f"steps, got {self.sequence_duration_ms}/{self.time_bin_ms} = {steps}"
            )
        if round(steps) < 1:
            raise ValueError("timing must yield at least one time step")

    @classmethod
    def from_dict(cls, data: dict, strict: bool = False) -> "TimingConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown and strict:
            raise ValueError(f"unknown timing keys: {sorted(unknown)}")
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class InputDelayLineConfig:
    """K time-shifted copies of the *already-binned* input (M10).

    The binned input ``x`` of shape ``(B, T, C)`` is presented to the shared
    input projection through a delay line: copy ``k`` (``k = 0 .. K-1``) is
    ``x`` shifted right along the time axis by ``k * stride`` bins, with zeros
    entering at the leading boundary.  Nothing is re-binned: this sits between
    the dense (B, T, C) input and ``w_in``, inside :class:`v3.model.VectorNeuronPopulation`,
    and is applied **once per batch before** the checkpointed temporal loop.

    Modes
    -----
    ``concat``
        the K shifted copies are stacked along the channel axis, so the tensor
        entering ``w_in`` is ``(B, T, K*C)`` and ``w_in`` becomes
        ``(K*C, D)``: **+(K-1)*C*D parameters** (+2.1 M at K=4, C=700,
        D=1000).  Every delay tap gets its own input weights.
    ``sum``
        the K shifted copies are summed back into the original C channels; the
        tensor stays ``(B, T, C)`` and ``w_in`` is unchanged -- no extra
        parameters, no learning per tap.  A smoothing / fan-in transform: each
        channel sees a short window of its own recent past.

    ``enabled=False`` (the default) is a no-op *and* adds no parameters: the
    model builds the same ``(C, D)`` ``w_in`` and the forward is bit-identical
    to the baseline.  ``copies=1`` is likewise a no-op in both modes (the code
    short-circuits to the original path).

    Cost at K=4, C=700, D=1000 (fp16 state), B=32, T=500
    ---------------------------------------------------
    * ``concat``: ``w_in`` grows 0.7 M -> 2.8 M parameters (+2.1 M; the whole
      model grows 2,025,440 -> 4,125,440, i.e. +2.1 M / +104 % at
      N=64, D=1000, R=64) and the forward holds one extra
      ``(B, T, K*C)`` input buffer -- 4*700*500*32 values = 89.6 MB in fp16
      (179 MB if kept in fp32).  It is filled one copy at a time, so no list of
      K full-size shifted tensors is ever built.
    * ``sum``: 0 parameters, one ``(B, T, C)`` fp32 buffer (44.8 MB).
    * ``copies`` is therefore capped at 4 in ``concat`` mode (validation):
      beyond that the buffer and the ``(K*C, D)`` weight/gradient no longer fit
      comfortably next to the ~2.3 GiB peak of the baseline at D=1000.

    ``stride_ms`` must convert to a whole number of ``timing.time_bin_ms``
    bins and must not exceed the sequence duration.  The timing-dependent
    checks only run when the block is enabled, so an inert block can never
    invalidate an otherwise unchanged run.
    """

    enabled: bool = False
    copies: int = 4          # K
    stride_ms: float = 10.0  # delay between successive copies (Nowotny: 30 ms)
    mode: str = "concat"     # "concat" (K*C channels) | "sum" (C channels)

    MODES = ("concat", "sum")
    MAX_CONCAT_COPIES = 4

    @property
    def effective_copies(self) -> int:
        """K actually presented (1 when the block is switched off)."""
        return int(self.copies) if self.enabled else 1

    @property
    def widens_input(self) -> bool:
        """True iff the delay line must widen ``w_in`` to ``(K*C, D)``."""
        return self.enabled and self.mode == "concat" and int(self.copies) > 1

    def input_channels(self, n_inputs: int) -> int:
        """Width of the tensor handed to ``w_in``: ``K*C`` for an active concat, else ``C``."""
        return int(n_inputs) * self.effective_copies if self.widens_input else int(n_inputs)

    def stride_bins(self, bin_ms: float) -> int:
        """Delay step in bins; rejects a stride that is not a whole number of bins."""
        step = float(self.stride_ms) / float(bin_ms)
        bins = int(round(step))
        if abs(step - bins) > 1e-9:
            raise ValueError(
                f"input_delay_line.stride_ms={self.stride_ms:g} is not an integer "
                f"multiple of timing.time_bin_ms={float(bin_ms):g}"
            )
        return bins

    def validate(
        self,
        bin_ms: float | None = None,
        sequence_duration_ms: float | None = None,
    ) -> None:
        if self.mode not in self.MODES:
            raise ValueError(
                f"input_delay_line.mode must be one of {self.MODES}, got {self.mode!r}"
            )
        if int(self.copies) < 1:
            raise ValueError("input_delay_line.copies must be >= 1")
        if float(self.stride_ms) < 0.0:
            raise ValueError("input_delay_line.stride_ms must be >= 0")
        if not self.enabled:
            return  # inert block: timing-dependent checks would be noise
        if self.mode == "concat" and int(self.copies) > self.MAX_CONCAT_COPIES:
            raise ValueError(
                "input_delay_line.copies is capped at "
                f"{self.MAX_CONCAT_COPIES} for mode='concat' (each copy adds "
                "C*D parameters and a full (B, T, C) input slice); use mode='sum' "
                "for more copies"
            )
        if int(self.copies) > 1 and float(self.stride_ms) <= 0.0:
            raise ValueError("input_delay_line.stride_ms must be > 0 when copies > 1")
        if sequence_duration_ms is not None and float(self.stride_ms) > float(sequence_duration_ms):
            raise ValueError(
                f"input_delay_line.stride_ms={self.stride_ms:g} exceeds the sequence "
                f"duration {float(sequence_duration_ms):g} ms"
            )
        if bin_ms is not None:
            self.stride_bins(bin_ms)  # raises unless the stride is a whole number of bins

    @classmethod
    def from_dict(cls, data: dict, strict: bool = False) -> "InputDelayLineConfig":
        return _block_from_dict(cls, data, strict, "input_delay_line")

    def summary(self) -> dict:
        return asdict(self)


@dataclass
class V3Config:
    """All V3 experiment parameters."""

    # ---- run identity ----------------------------------------------------
    tag: str = "v3"
    seed: int = 0

    # ---- data (SHD) ------------------------------------------------------
    train_h5: str = "data/shd_train.h5"
    test_h5: str = "data/shd_test.h5"
    n_inputs: int = 700
    n_classes: int = 20
    # temporal discretization: the step count T is *derived*, never stored
    timing: TimingConfig = field(default_factory=TimingConfig)
    binary_input: bool = True    # >=1 event in a bin -> 1 (standard SHD encoding)
    val_fraction: float = 0.10   # stratified validation split, taken from the TRAIN file
    split_seed: int = 0
    cache_events: bool = True    # cache binned event codes in RAM (faster epochs)
    binned_cache: bool = False   # read the dense (n,T,C) input from the pre-binned
    #                              disk cache (V3/v3/binned_cache.py) instead of
    #                              rebuilding it per batch; opt-in, built once
    max_train_samples: int = 0   # 0 = use the whole training split (subset only for quick runs)
    shuffle_train: bool = True   # shuffle FIT batches between epochs
    shuffle_val: bool = False    # validation order is deterministic

    # ---- model -----------------------------------------------------------
    n_neurons: int = 64          # N
    state_dim: int = 1000        # D
    mix_rank: int = 64           # rank of the shared intra-neuron mixing map
    tau_ms: float = 20.0         # state leak time constant [ms]
    input_gain: float = 5.0      # scale of the shared input projection init
    surrogate: str = "fast_sigmoid"  # "fast_sigmoid" (baseline) | "slayer"
    surrogate_beta: float = 4.0  # fast-sigmoid surrogate slope
    surrogate_gamma: float = 1.0 # fast-sigmoid surrogate gain
    surrogate_alpha: float = 2.0 # SLAYER width: 1/(1 + alpha|u|)^2; 2.0 keeps the
    #                              far-field gradient <= 4x fast_sigmoid so the fp16
    #                              BPTT keeps the baseline's loss-scale headroom
    readout: str = "sum"         # "sum" (spike count) | "mean"
    rate_reg: float = 1.0e-4     # weight of the firing-rate regulariser (0 disables)
    rate_target_hz: float = 10.0
    emit_identity_init: bool = True  # init W_emit ~ I (+ small noise)

    # ---- learnable per-input-channel conduction delays (off by default) --
    # One delay per input channel [ms], applied to the binned input sequence
    # *before* the shared input projection (see V3/v3/model.py).  Disabled ->
    # the input path is untouched (bit-identical baseline).  Enabled -> the
    # delays start strictly interior (see InputDelayConfig) so the parameter
    # receives gradient from the first step.
    input_delays: InputDelayConfig = field(default_factory=InputDelayConfig)

    # ---- training --------------------------------------------------------
    batch_size: int = 32
    epochs: int = 5             # primary control of how much training is done
    max_train_batches: int | None = None  # None = all batches; int = cap per epoch
    learning_rate: float = 2.0e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    optimizer: str = "adam"
    lr_schedule: str = "cosine"  # "cosine" | "none"
    warmup_frac: float = 0.03
    label_smoothing: float = 0.0

    # ---- precision / device ---------------------------------------------
    device: str = "cuda"
    dtype: str = "float16"       # dtype of the neural state / major model tensors
    amp: bool = True             # autocast + GradScaler for fp32 master params
    grad_checkpoint_chunks: int = 10  # split T into this many checkpointed chunks
    select_best_val: bool = True      # keep the best-validation-accuracy epoch as secondary
    eval_test_each_epoch: bool = False  # score the official TEST split at the end of every epoch

    # ---- controlled state imprecision (disabled by default) --------------
    # This is the *only* new scientific manipulation in the imprecision stage: it
    # deliberately perturbs/quantizes the neuron's internal vector state z(t).
    # It is independent of `dtype`/`amp` (the training arithmetic).  Default
    # `mode="none"` reproduces the baseline exactly.
    state_regularization: StateRegularizationConfig = field(
        default_factory=StateRegularizationConfig
    )

    # ---- trainable state dynamics (disabled by default) -------------------
    # Per-neuron leak (tau) and per-neuron initial state z(0).  Both flags are
    # off, so the default forward pass is exactly the fixed-alpha, zero-state
    # baseline.
    state_dynamics: StateDynamicsConfig = field(
        default_factory=StateDynamicsConfig
    )
    # ---- two-timescale adaptation currents (disabled by default) ---------
    # adLIF-style spike-triggered currents that subtract from the pre-activation
    # of the state update (inside the tanh).  Off by default: no parameter is
    # created and the baseline update rule stays bit-identical.
    adlif: AdlifConfig = field(default_factory=AdlifConfig)
    # ---- second nonlinear mixing stage (disabled by default) --------------
    # Extra depth *inside* one timestep: an independent rank-R2 mixer feeding a
    # second tanh before the state update.  Off by default, and the parameters
    # are created only when enabled, so the baseline model is unchanged.
    two_stage: TwoStageConfig = field(default_factory=TwoStageConfig)

    # ---- raw-event augmentation (disabled by default) ---------------------
    # Applied to the raw (time, channel) events *before* the 2-ms binning; the
    # binned tensor is never perturbed.  All defaults are off, so the default
    # path stays bit-identical.  Training only unless a phase flag is flipped.
    augmentation: EventAugmentationConfig = field(
        default_factory=EventAugmentationConfig
    )

    # ---- spike-frequency adaptation (disabled by default) -----------------
    # A per-neuron slow negative feedback on the spike-generation potential:
    # pot = <z, w_out> + theta - w_ad * b with b <- rho*b + s carried like the
    # state.  Off by default and nothing is registered when off, so the
    # baseline forward pass and parameter set are unchanged.
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    # ---- input delay line on the binned input (disabled by default) --------
    # K time-shifted copies of the *already-binned* (B, T, C) input, applied
    # once per batch before the checkpointed temporal loop.  Off by default:
    # no extra parameters and a bit-identical forward (see
    # InputDelayLineConfig for the K=4, D=1000 parameter/memory cost).
    input_delay_line: InputDelayLineConfig = field(
        default_factory=InputDelayLineConfig
    )
    # ---- learned temporal attention readout (disabled by default) --------
    # When enabled, the readout uses one learned weight per time bin instead of
    # a plain count sum; see AttentionConfig.  The default (off) path is
    # bit-identical and adds no parameters.
    attention: AttentionConfig = field(default_factory=AttentionConfig)

    # ---- io --------------------------------------------------------------
    out_dir: str = "results"        # resolved relative to V3/ when not absolute
    ckpt_dir: str = "checkpoints"
    figure_dir: str = "figures"
    log_every: int = 0           # >0 prints every k batches (in addition to tqdm)

    # ------------------------------------------------------------------ #
    # derived quantities
    # ------------------------------------------------------------------ #
    @property
    def bin_ms(self) -> float:
        """Width of one temporal bin [ms] (== ``timing.time_bin_ms``)."""
        return float(self.timing.time_bin_ms)

    @property
    def n_bins(self) -> int:
        """Number of temporal steps T, derived from the timing block."""
        return int(self.timing.num_time_steps)

    @property
    def time_bin_ms(self) -> float:
        return self.bin_ms

    @property
    def sequence_duration_ms(self) -> float:
        return float(self.timing.sequence_duration_ms)

    @property
    def num_time_steps(self) -> int:
        return self.n_bins

    @property
    def dt_ms(self) -> float:
        """Integration step.  One bin == one dynamical step."""
        return float(self.timing.simulation_dt_ms)

    @property
    def simulation_dt_ms(self) -> float:
        return self.dt_ms

    @property
    def alpha(self) -> float:
        """Discrete leak factor ``dt / tau`` of the state update.

        Baseline path only: when ``state_dynamics.alpha_trainable`` is in force,
        the model uses the per-neuron ``sigmoid(raw_i)`` leak instead.
        """
        return min(1.0, float(self.dt_ms) / float(self.tau_ms))

    @property
    def duration_s(self) -> float:
        return self.timing.duration_s

    @property
    def window_s(self) -> float:
        return self.timing.duration_s

    @property
    def effective_input_channels(self) -> int:
        """Width of the tensor entering ``w_in``: ``C``, or ``K*C`` for an active concat."""
        return self.input_delay_line.input_channels(self.n_inputs)

    @property
    def torch_dtype(self):
        import torch

        return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
            str(self.dtype).lower()
        ]

    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        if self.n_neurons < 1 or self.state_dim < 1:
            raise ValueError("n_neurons and state_dim must be >= 1")
        self.timing.validate()
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError("val_fraction must be in [0, 1)")
        if self.surrogate not in SURROGATE_KINDS:
            raise ValueError(f"surrogate must be one of {SURROGATE_KINDS}, got {self.surrogate!r}")
        if float(self.surrogate_alpha) <= 0.0:
            raise ValueError("surrogate_alpha (SLAYER width) must be > 0")
        if self.readout not in READOUTS:
            raise ValueError(f"readout must be one of {READOUTS}, got {self.readout!r}")
        if float(self.grad_clip) < 0.0:
            raise ValueError("grad_clip must be >= 0 (0 disables global-norm clipping)")
        if self.optimizer not in ("adam", "adamw", "sgd"):
            raise ValueError("optimizer must be adam | adamw | sgd")
        if self.lr_schedule not in ("cosine", "none"):
            raise ValueError("lr_schedule must be cosine | none")
        if self.mix_rank < 1:
            raise ValueError("mix_rank must be >= 1")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha = dt/tau must be in (0, 1]")
        if self.grad_checkpoint_chunks < 1:
            raise ValueError("grad_checkpoint_chunks must be >= 1")
        if self.max_train_batches is not None and int(self.max_train_batches) < 1:
            raise ValueError("max_train_batches must be null (all batches) or >= 1")
        self.state_regularization.validate()
        self.state_dynamics.validate()
        self.adlif.validate()
        self.two_stage.validate()
        self.augmentation.validate()
        self.adaptation.validate()
        self.input_delays.validate()
        self.input_delay_line.validate(
            bin_ms=self.bin_ms, sequence_duration_ms=self.sequence_duration_ms
        )
        self.attention.validate()

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        d = asdict(self)
        d["derived"] = {
            "sequence_duration_ms": self.sequence_duration_ms,
            "time_bin_ms": self.time_bin_ms,
            "num_time_steps": self.num_time_steps,
            "simulation_dt_ms": self.simulation_dt_ms,
            "dt_ms": self.dt_ms,
            "alpha": self.alpha,
            "duration_s": self.duration_s,
            "window_s": self.window_s,
        }
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def resolved(self) -> "V3Config":
        """Validated copy of this configuration."""
        self.validate()
        return dataclasses.replace(self)

    def path(self, value: str) -> Path:
        """Resolve a data path relative to the repository root."""
        p = Path(value)
        return p if p.is_absolute() else (REPO_ROOT / p)

    def v3_path(self, value: str) -> Path:
        """Resolve an output path relative to the ``V3/`` folder."""
        p = Path(value)
        return p if p.is_absolute() else (V3_ROOT / p)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, data: dict, strict: bool = False) -> "V3Config":
        data = dict(data)
        # backward compatibility: derive a timing block from the legacy flat
        # `bin_ms` / `n_bins` keys if no explicit `timing` block is present
        if "timing" not in data and ("bin_ms" in data or "n_bins" in data):
            bin_ms = float(data.pop("bin_ms", 4.0))
            n_bins = int(data.pop("n_bins", 250))
            data["timing"] = {
                "time_bin_ms": bin_ms,
                "sequence_duration_ms": bin_ms * n_bins,
            }
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known - {"derived"}
        if unknown and strict:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        kwargs = {}
        for k, v in data.items():
            if k not in known:
                continue
            if k == "state_regularization" and isinstance(v, dict):
                v = StateRegularizationConfig.from_dict(v, strict=strict)
            elif k == "state_dynamics" and isinstance(v, dict):
                v = StateDynamicsConfig.from_dict(v, strict=strict)
            elif k == "adlif" and isinstance(v, dict):
                v = AdlifConfig.from_dict(v, strict=strict)
            elif k == "two_stage" and isinstance(v, dict):
                v = TwoStageConfig.from_dict(v, strict=strict)
            elif k == "augmentation" and isinstance(v, dict):
                v = EventAugmentationConfig.from_dict(v, strict=strict)
            elif k == "adaptation" and isinstance(v, dict):
                v = AdaptationConfig.from_dict(v, strict=strict)
            elif k == "input_delays" and isinstance(v, dict):
                v = InputDelayConfig.from_dict(v, strict=strict)
            elif k == "input_delay_line" and isinstance(v, dict):
                v = InputDelayLineConfig.from_dict(v, strict=strict)
            elif k == "attention" and isinstance(v, dict):
                v = AttentionConfig.from_dict(v, strict=strict)
            elif k == "timing" and isinstance(v, dict):
                v = TimingConfig.from_dict(v, strict=strict)
            kwargs[k] = v
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path, strict: bool = True) -> "V3Config":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data, strict=strict)

    def with_overrides(self, **overrides) -> "V3Config":
        """Copy the configuration with ``overrides`` applied.

        Plain field names (``epochs=3``, ``input_delays=InputDelayConfig(...)``)
        replace the field directly.  Dotted keys (``input_delays.enabled``) go
        through the same grammar as the CLI overrides, so a single leaf can be
        changed without rebuilding its block.
        """
        plain = {k: v for k, v in overrides.items() if "." not in k}
        nested = {k: v for k, v in overrides.items() if "." in k}
        out = dataclasses.replace(self, **plain)
        return apply_override_mapping(out, nested) if nested else out


# ---------------------------------------------------------------------- #
# override grammar (CLI ``--override key=value`` and programmatic mappings)
# ---------------------------------------------------------------------- #
NESTED_BLOCKS: dict[str, type] = {
    "timing": TimingConfig,
    "state_regularization": StateRegularizationConfig,
    "state_dynamics": StateDynamicsConfig,
    "adlif": AdlifConfig,
    "two_stage": TwoStageConfig,
    "augmentation": EventAugmentationConfig,
    "adaptation": AdaptationConfig,
    "input_delays": InputDelayConfig,
    "input_delay_line": InputDelayLineConfig,
    "attention": AttentionConfig,
}


def _coerce_override(annotation: object, raw: str):
    """Coerce a CLI string to the field's annotated scalar type."""
    anno = str(annotation)
    if "bool" in anno:
        text = raw.strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"cannot parse {raw!r} as bool")
    if "int" in anno and "float" not in anno:
        return int(raw)
    if "float" in anno:
        return float(raw)
    return raw


def _resolve_override(cfg: V3Config, key: str):
    """Walk ``a.b.c`` against the dataclass schema; return (leaf field, value)."""
    parts = key.split(".")
    if not key or any(not part for part in parts):
        raise ValueError(f"invalid override key {key!r}")
    obj: object = cfg
    for index, part in enumerate(parts):
        if not dataclasses.is_dataclass(obj):
            raise ValueError(f"unknown configuration key {key!r}")
        fields = {f.name: f for f in dataclasses.fields(obj)}
        if part not in fields:
            raise ValueError(f"unknown configuration key {key!r}")
        value = getattr(obj, part)
        if index == len(parts) - 1:
            return fields[part], value
        if not dataclasses.is_dataclass(value):
            raise ValueError(
                f"unknown configuration key {key!r}: {part!r} is not a nested block"
            )
        obj = value
    raise ValueError(f"unknown configuration key {key!r}")  # pragma: no cover


def _merge_override_dict(base: dict, updates: dict, prefix: str) -> dict:
    """Recursively merge a nested mapping into a config block, rejecting unknown keys."""
    for key, value in updates.items():
        if key not in base:
            raise ValueError(f"unknown configuration key {f'{prefix}.{key}'}")
        if isinstance(value, dict) and isinstance(base[key], dict):
            base[key] = _merge_override_dict(base[key], value, f"{prefix}.{key}")
        else:
            base[key] = value
    return base


def _set_override_path(cfg: V3Config, parts: list[str], value) -> V3Config:
    if len(parts) == 1:
        return dataclasses.replace(cfg, **{parts[0]: value})
    child = getattr(cfg, parts[0])
    return dataclasses.replace(
        cfg, **{parts[0]: _set_override_path(child, parts[1:], value)}
    )


def _apply_override(cfg: V3Config, key: str, value) -> V3Config:
    field_def, current = _resolve_override(cfg, key)
    parts = key.split(".")
    if dataclasses.is_dataclass(current):
        if isinstance(value, dict):
            merged = _merge_override_dict(asdict(current), value, key)
            value = type(current).from_dict(merged, strict=True)
        elif isinstance(value, type(current)):
            value = value
        else:
            raise ValueError(
                f"override {key!r} expects a mapping (or {type(current).__name__}), "
                f"got {type(value).__name__}"
            )
    elif isinstance(value, str):
        value = _coerce_override(field_def.type, value)
    return _set_override_path(cfg, parts, value)


def apply_override_mapping(cfg: V3Config, mapping: dict[str, object]) -> "V3Config":
    """Apply programmatic overrides, including two levels of dotted nesting.

    ``{"augmentation.time_jitter.std_ms": 0.5, "epochs": 7}`` updates the leaf
    field; a nested mapping (``{"augmentation": {"time_jitter": {"std_ms": 0.5}}}``)
    is merged into the existing block, so unspecified fields keep their values.
    Unknown keys raise :class:`ValueError`; the returned object is a new config
    (the input is never mutated).
    """
    if not isinstance(mapping, dict):
        raise TypeError(f"overrides must be a dict, got {type(mapping).__name__}")
    out = cfg
    for key, value in mapping.items():
        if not isinstance(key, str):
            raise ValueError(f"override keys must be strings, got {type(key).__name__}")
        out = _apply_override(out, key, value)
    return out


def apply_override_items(cfg: V3Config, items: list[str]) -> "V3Config":
    """Apply CLI-style ``key=value`` overrides (two levels of dotted nesting).

    ``timing.time_bin_ms=4``, ``state_regularization.mode=noise`` and
    ``augmentation.time_jitter.std_ms=0.5`` are all valid.  Raises
    :class:`ValueError` for malformed items and unknown keys.
    """
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"override must be key=value, got {item!r}")
        parsed[key] = raw
    return apply_override_mapping(cfg, parsed)


def describe(cfg: V3Config) -> str:
    """Short human-readable summary used in logs and notebook output."""
    reg = cfg.state_regularization
    dl = cfg.input_delay_line
    delay = (
        f", delay_line={dl.mode}x{dl.copies}@{dl.stride_ms:g}ms" if dl.enabled else ""
    )
    readout = "temporal_attention" if cfg.attention.enabled else cfg.readout
    return (
        f"V3Config(tag={cfg.tag!r}, N={cfg.n_neurons}, D={cfg.state_dim}, "
        f"T={cfg.num_time_steps}x{cfg.time_bin_ms:g}ms={cfg.sequence_duration_ms:g}ms, "
        f"batch={cfg.batch_size}, epochs={cfg.epochs}, lr={cfg.learning_rate:g}, "
        f"device={cfg.device}, dtype={cfg.dtype}, surrogate={cfg.surrogate}, "
        f"readout={readout}, state_reg={reg.mode}{delay})"
    )


def parameter_groups(cfg: V3Config) -> dict:
    """Analytic parameter budget of the primary model (used for reporting)."""
    N, D, C, R = cfg.n_neurons, cfg.state_dim, cfg.n_inputs, min(cfg.mix_rank, cfg.state_dim)
    groups = {
        # C, or K*C when the delay line stacks shifted copies along the channel axis
        "w_in": cfg.effective_input_channels * D,
        "w_down": D * R,
        "w_up": R * D,
        "w_emit": D * D,
        "w_rec": N * N,
        "gate": N * D,
        "bias": N * D,
        "w_out": N * D,
        "theta": N,
        "w_cls": N * cfg.n_classes,
    }
    if cfg.adaptation.enabled and cfg.adaptation.learnable_gain:
        groups["w_ad"] = N
    if cfg.input_delays.enabled:
        groups["input_delays"] = C
    if cfg.adlif.enabled:
        # one learnable scalar gain per neuron per enabled timescale
        if cfg.adlif.use_fast:
            groups["adlif_gain_fast"] = N
        if cfg.adlif.use_slow:
            groups["adlif_gain_slow"] = N
    if cfg.attention.enabled:
        # one learned logit per time bin; absent (and budget-neutral) when off
        groups["attn_logits"] = cfg.n_bins
    if cfg.two_stage.active:
        # the second mixer is an independent rank-R2 pair plus the raw scalar
        # lam_raw (the effective lam is a sigmoid of it, so still one parameter);
        # absent by default, so the baseline budget is unchanged.
        R2 = min(int(cfg.two_stage.rank), D)
        groups["w_down2"] = D * R2
        groups["w_up2"] = R2 * D
        groups["lam_raw"] = 1
    groups["total"] = sum(groups.values())
    return groups


__all__ = [
    "V3Config",
    "TimingConfig",
    "StateRegularizationConfig",
    "AdaptationConfig",
    "StateDynamicsConfig",
    "InputDelayConfig",
    "AdlifConfig",
    "AttentionConfig",
    "TwoStageConfig",
    "TimeJitterConfig",
    "EventBlendConfig",
    "EventDropoutConfig",
    "EventInsertionConfig",
    "GlobalTimeShiftConfig",
    "ChannelDropoutConfig",
    "EventAugmentationConfig",
    "InputDelayLineConfig",
    "SURROGATE_KINDS",
    "READOUTS",
    "NESTED_BLOCKS",
    "apply_override_items",
    "apply_override_mapping",
    "V3_ROOT",
    "REPO_ROOT",
    "describe",
    "parameter_groups",
    "math",
]
