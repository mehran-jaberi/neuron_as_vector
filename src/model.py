"""A deliberately simple recurrent LIF spiking neural network (SHD baseline).

Design intent
-------------
The classifier is *not* the point of the project. It exists to produce a trained
spiking circuit whose individual hidden neurons can then be analysed. We
therefore keep the architecture conventional and expose every internal quantity
(input weights, recurrent weights, output weights, hidden membrane potentials,
hidden spikes, output activity) as a first-class object.

Dynamics (discrete time, current-based synapses, implicit-Euler LIF)
------------------------------------------------------------------
.. math::

    I_t &= \\beta I_{t-1} + W_{in} x_t + W_{rec} S_{t-1} + b_{hid} \\\\
    V_t &= \\alpha V_{t-1} + (1-\\alpha) I_t \\\\
    S_t &= \\Theta(V_t - \\vartheta) \\quad \\text{(surrogate gradient)} \\\\
    V_t &\\leftarrow V_t - \\vartheta S_t \\quad \\text{(soft reset)}

    O_t &= \\rho O_{t-1} + W_{out} S_t + b_{out}, \\qquad
    \\text{logits} = \\frac{1}{T}\\sum_t O_t

with :math:`\\alpha=\\exp(-\\Delta t/\\tau_{mem})` and
:math:`\\beta=\\exp(-\\Delta t/\\tau_{syn})`.

The output layer is a leaky linear integrator over hidden spikes rather than a
second spiking layer. This is the standard SHD readout, it trains more reliably
than a spiking readout, and it keeps the *hidden* layer as the only spiking
population of interest - which is exactly the population we analyse.

Surrogate gradient
------------------
A SuperSpike-style fast-sigmoid derivative :math:`\\gamma / (1+\\beta|u|)^2` with
:math:`u = V-\\vartheta`, injected with the subtract-detach trick so that the
forward value is exactly the Heaviside step and only the backward pass is
smooth.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn

from .utils import Config


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class SNNConfig:
    """All model hyper-parameters. Serialisable and fully configurable."""

    n_input: int = 700
    n_hidden: int = 256
    n_output: int = 20

    # temporal discretisation
    n_bins: int = 500
    bin_ms: float = 2.0

    # neuron dynamics
    tau_mem_ms: float = 20.0
    tau_syn_ms: float = 5.0
    threshold: float = 1.0
    reset: str = "subtract"  # "subtract" (soft) or "zero" (hard)
    readout_leak: float = 0.0  # rho; 0 => pure mean-rate readout

    # surrogate gradient
    surrogate_beta: float = 5.0
    surrogate_gamma: float = 0.3

    # initialisation
    input_weight_scale: float = 10.0
    recurrent_weight_scale: float = 2.0
    recurrent_density: float = 1.0  # fraction of non-zero recurrent connections
    zero_self_connection: bool = False
    signed_input_weights: bool = True
    signed_recurrent_weights: bool = True

    # learnable, neuron-specific parameters (makes the intrinsic block informative)
    neuron_param_mode: str = "bias"  # "none" | "bias" | "bias_tau"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "SNNConfig":
        valid = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in dict(mapping).items() if k in valid}
        return cls(**kwargs)

    @classmethod
    def from_config(cls, cfg: Config, prefix: str = "model") -> "SNNConfig":
        """Build from a nested project config, e.g. ``cfg["model"]["n_hidden"]``."""
        node = cfg.get_path(prefix, {}) or {}
        merged = Config({**node, **{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__}})  # type: ignore[attr-defined]
        return cls.from_mapping(merged.to_dict())

    # -- derived quantities -------------------------------------------------
    @property
    def alpha(self) -> float:
        return float(math.exp(-self.bin_ms / self.tau_mem_ms))

    @property
    def beta(self) -> float:
        return float(math.exp(-self.bin_ms / self.tau_syn_ms))

    @property
    def duration_ms(self) -> float:
        return self.n_bins * self.bin_ms


# --------------------------------------------------------------------------
# Surrogate spike nonlinearity
# --------------------------------------------------------------------------
def surrogate_spike(u: torch.Tensor, beta: float = 5.0, gamma: float = 0.3) -> torch.Tensor:
    """Heaviside step in the forward pass, fast-sigmoid derivative in the backward pass.

    ``u`` is the *thresholded* membrane potential :math:`V-\\vartheta`.
    """
    heaviside = (u > 0).to(u.dtype)
    # Differentiable surrogate whose derivative is the fast-sigmoid surrogate
    # gradient ``gamma / (1 + beta |u|)^2`` (equal to ``gamma`` at u = 0).
    surrogate = gamma * u / (1.0 + beta * u.abs())
    # Forward value == heaviside; gradient only flows through `surrogate`.
    return heaviside.detach() - surrogate.detach() + surrogate


def lif_voltage_update(
    v: torch.Tensor,
    i_syn: torch.Tensor,
    alpha: torch.Tensor | float,
) -> torch.Tensor:
    """One implicit-Euler LIF membrane update: ``V <- alpha V + (1 - alpha) I``."""
    return alpha * v + (1.0 - alpha) * i_syn


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class RecurrentLIFSNN(nn.Module):
    """Recurrent LIF network with a linear leaky-integrator readout.

    Exposed internals (used by the representation modules)
    -----------------------------------------------------
    ``w_in``   : (n_input, n_hidden)   input weights, column ``j`` = neuron ``j``
    ``w_rec``  : (n_hidden, n_hidden)  ``w_rec[i, j]`` = weight from ``j`` to ``i``
    ``w_out``  : (n_hidden, n_output)
    ``b_hid``  : (n_hidden, 1) or None learnable per-neuron excitability
    ``tau_mem_ms`` property / ``alpha`` tensor : per-neuron membrane constants
    """

    def __init__(self, cfg: SNNConfig):
        super().__init__()
        self.cfg = cfg

        self.w_in = nn.Parameter(torch.empty(cfg.n_input, cfg.n_hidden))
        self.w_rec = nn.Parameter(torch.empty(cfg.n_hidden, cfg.n_hidden))
        self.w_out = nn.Parameter(torch.empty(cfg.n_hidden, cfg.n_output))
        self.b_out = nn.Parameter(torch.zeros(cfg.n_output))

        self.neuron_param_mode = cfg.neuron_param_mode
        if self.neuron_param_mode in ("bias", "bias_tau"):
            self.b_hid = nn.Parameter(torch.zeros(cfg.n_hidden))
        else:
            self.register_parameter("b_hid", None)

        if self.neuron_param_mode == "bias_tau":
            # Unconstrained per-neuron log-time-constant offsets (init: 0 -> tau_mem).
            self.log_tau_offset = nn.Parameter(torch.zeros(cfg.n_hidden))
        else:
            self.register_parameter("log_tau_offset", None)

        self.register_buffer("self_mask", torch.ones(cfg.n_hidden, cfg.n_hidden))
        self.reset_parameters()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def reset_parameters(self, generator: torch.Generator | None = None) -> None:
        cfg = self.cfg
        with torch.no_grad():
            fan_in = float(cfg.n_input)
            bound = cfg.input_weight_scale / math.sqrt(fan_in)
            w_in = (torch.rand(cfg.n_input, cfg.n_hidden, generator=generator) * 2.0 - 1.0) * bound
            if not cfg.signed_input_weights:
                w_in = w_in.abs()
            self.w_in.copy_(w_in)

            rec_std = cfg.recurrent_weight_scale / math.sqrt(cfg.n_hidden)
            w_rec = torch.randn(cfg.n_hidden, cfg.n_hidden, generator=generator) * rec_std
            if cfg.recurrent_density < 1.0:
                mask = (torch.rand(cfg.n_hidden, cfg.n_hidden, generator=generator) < cfg.recurrent_density).float()
                w_rec = w_rec * mask
            if not cfg.signed_recurrent_weights:
                w_rec = w_rec.abs()
            if cfg.zero_self_connection:
                self.self_mask.fill_(1.0)
                self.self_mask.fill_diagonal_(0.0)
            w_rec = w_rec * self.self_mask
            self.w_rec.copy_(w_rec)

            out_bound = 1.0 / math.sqrt(max(cfg.n_hidden, 1))
            self.w_out.copy_(
                (torch.rand(cfg.n_hidden, cfg.n_output, generator=generator) * 2.0 - 1.0) * out_bound
            )
            self.b_out.zero_()
            if self.b_hid is not None:
                self.b_hid.zero_()
            if self.log_tau_offset is not None:
                self.log_tau_offset.zero_()

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------
    @property
    def n_hidden(self) -> int:
        return self.cfg.n_hidden

    def alpha_tensor(self) -> torch.Tensor:
        """Per-neuron membrane decay, shape ``(n_hidden,)`` (or scalar 0-dim)."""
        base = torch.full(
            (self.cfg.n_hidden,), self.cfg.tau_mem_ms, device=self.w_in.device, dtype=self.w_in.dtype
        )
        if self.log_tau_offset is not None:
            tau = base * torch.exp(self.log_tau_offset)
        else:
            tau = base
        return torch.exp(-self.cfg.bin_ms / tau.clamp(min=self.cfg.bin_ms * 1.01))

    def effective_threshold(self) -> torch.Tensor:
        return torch.full(
            (self.cfg.n_hidden,), float(self.cfg.threshold), device=self.w_in.device, dtype=self.w_in.dtype
        )

    @property
    def effective_reset(self) -> float:
        return float(self.cfg.threshold if self.cfg.reset == "subtract" else 0.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        *,
        record: bool = False,
        state: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        """Run the network over ``x`` of shape ``(B, T, n_input)``.

        Parameters
        ----------
        record:
            When ``True`` the full per-timestep traces (hidden membrane potentials,
            hidden spikes, output activity) are returned. During training this is
            left ``False`` to keep the autograd graph small.

        Returns
        -------
        dict with ``logits`` (B, n_output), ``spike_count`` (B, n_hidden) and, if
        ``record``, ``hidden_v``/``hidden_spikes`` (B, T, n_hidden) and
        ``output_activity`` (B, T, n_output).
        """
        cfg = self.cfg
        if x.dim() != 3:
            raise ValueError(f"Expected input of shape (B, T, n_input), got {tuple(x.shape)}")
        B, T, _ = x.shape
        if T != cfg.n_bins:
            raise ValueError(
                f"Input has T={T} time steps but the model is configured for "
                f"n_bins={cfg.n_bins}. The binning window and the model must agree; "
                f"refusing to silently run with a mismatched window."
            )
        device, dtype = x.device, x.dtype

        if state is None:
            v = torch.zeros(B, cfg.n_hidden, device=device, dtype=dtype)
            i_syn = torch.zeros(B, cfg.n_hidden, device=device, dtype=dtype)
            s_prev = torch.zeros(B, cfg.n_hidden, device=device, dtype=dtype)
            o = torch.zeros(B, cfg.n_output, device=device, dtype=dtype)
        else:
            v = state["v"]
            i_syn = state["i_syn"]
            s_prev = state["s_prev"]
            o = state["o"]

        alpha = self.alpha_tensor().to(device=device, dtype=dtype)
        thr = self.effective_threshold().to(device=device, dtype=dtype)
        reset = self.effective_reset
        beta = cfg.beta
        rho = cfg.readout_leak
        w_in, w_rec, w_out = self.w_in, self.w_rec, self.w_out
        b_hid = self.b_hid
        b_out = self.b_out

        spike_count = torch.zeros(B, cfg.n_hidden, device=device, dtype=dtype)
        # Time-accumulated readout: logits = (1/T) * sum_t O_t (see class docstring).
        o_sum = torch.zeros(B, cfg.n_output, device=device, dtype=dtype)
        rec_spikes = torch.zeros(B, T, cfg.n_hidden, device=device, dtype=dtype) if record else None
        rec_v = torch.zeros(B, T, cfg.n_hidden, device=device, dtype=dtype) if record else None
        rec_o = torch.zeros(B, T, cfg.n_output, device=device, dtype=dtype) if record else None

        for t in range(T):
            recurrent_input = s_prev @ w_rec.t()
            i_syn = beta * i_syn + x[:, t, :] @ w_in + recurrent_input
            if b_hid is not None:
                i_syn = i_syn + b_hid
            v = lif_voltage_update(v, i_syn, alpha)

            s = surrogate_spike(v - thr, cfg.surrogate_beta, cfg.surrogate_gamma)
            # Reset uses the *spike indicator* detached from the autograd graph, so
            # the surrogate gradient does not leak a second gradient path through the
            # reset. Forward behaviour is unchanged: subtract the threshold (soft
            # reset) or zero the membrane (hard reset) whenever the neuron spiked.
            s_reset = s.detach()
            if reset == 0.0:
                v = v * (1.0 - s_reset)
            else:
                v = v - reset * s_reset

            o = rho * o + s @ w_out + b_out
            o_sum = o_sum + o

            s_prev = s
            spike_count = spike_count + s
            if record:
                rec_spikes[:, t, :] = s
                rec_v[:, t, :] = v
                rec_o[:, t, :] = o

        logits = o_sum / float(max(T, 1))

        out: dict[str, Any] = {
            "logits": logits,
            "output_activity": o,
            "spike_count": spike_count,
            "state": {"v": v, "i_syn": i_syn, "s_prev": s_prev, "o": o},
        }
        if record:
            out["hidden_v"] = rec_v
            out["hidden_spikes"] = rec_spikes
            out["output_activity_trace"] = rec_o
        return out

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def save(self, path: str, extra: Mapping[str, Any] | None = None) -> None:
        payload = {
            "model_config": self.cfg.to_dict(),
            "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
        }
        if extra:
            payload["extra"] = dict(extra)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, map_location: Any = "cpu") -> tuple["RecurrentLIFSNN", dict[str, Any]]:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        cfg = SNNConfig.from_mapping(payload["model_config"])
        model = cls(cfg)
        model.load_state_dict(payload["state_dict"])
        return model, payload.get("extra", {})


def build_model(cfg: Config | SNNConfig, *, seed: int | None = None, device: Any = None) -> RecurrentLIFSNN:
    """Construct a model, optionally with a dedicated RNG seed for its init.

    Passing the same ``seed`` to this function yields identical initial weights,
    which is what makes the untrained-vs-trained comparison in
    :mod:`src.controls` a fair comparison.
    """
    snn_cfg = cfg if isinstance(cfg, SNNConfig) else SNNConfig.from_config(cfg)
    model = RecurrentLIFSNN(snn_cfg)
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        model.reset_parameters(generator=generator)
    if device is not None:
        model = model.to(device)
    return model


ARCHITECTURE_KEYS: tuple[str, ...] = ("n_input", "n_hidden", "n_output", "n_bins", "bin_ms")


def architecture_mismatches(
    expected_model_block: Mapping[str, Any],
    model: RecurrentLIFSNN | SNNConfig,
) -> dict[str, tuple[Any, Any]]:
    """Compare an analysis config's ``model`` block against a model's architecture.

    Returns ``{key: (expected_value, actual_value)}`` for every architecture-defining
    key that disagrees (``{}`` means they agree). Used to refuse a silent mismatch
    between an analysis config and a checkpoint, which would otherwise make
    downstream comparisons (e.g. an untrained-vs-trained model) scientifically
    invalid without any error.
    """
    snn = model if isinstance(model, SNNConfig) else model.cfg
    have = snn.to_dict()
    expected = dict(expected_model_block)
    return {
        key: (expected.get(key), have.get(key))
        for key in ARCHITECTURE_KEYS
        if key in expected and expected.get(key) != have.get(key)
    }


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Parameter counts, split into the blocks that the analysis cares about."""
    out = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            out[name] = int(param.numel())
    out["total"] = int(sum(out.values()))
    return out
