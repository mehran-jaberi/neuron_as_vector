"""Manual per-step timing of the fused V3 timestep loop (``_segment_any``).

Times ``VectorNeuronPopulation._segment_any`` directly at the performance
protocol of the task brief -- ``B=64``, ``N=64``, ``D=1000``, fp16, CUDA when
available -- and prints milliseconds per timestep.  Inference only
(``torch.no_grad``): no BPTT graph is built, so this measures the kernel chain of
one step, not training throughput.  The test suite never runs this tool.

    .venv/Scripts/python.exe V3/tools/bench_step.py
    .venv/Scripts/python.exe V3/tools/bench_step.py --device cuda --repeats 5

One call covers ``--steps`` timesteps, exactly like one chunk of ``forward()``;
the printed number is per timestep.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

V3_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(V3_ROOT), str(V3_ROOT.parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from v3.config import TimingConfig, V3Config          # noqa: E402
from v3.model import VectorNeuronPopulation           # noqa: E402


def _config(args: argparse.Namespace, device: str) -> V3Config:
    return V3Config(
        n_neurons=args.neurons,
        state_dim=args.dim,
        mix_rank=args.rank,
        n_inputs=args.inputs,
        n_classes=20,
        timing=TimingConfig(sequence_duration_ms=2.0 * args.steps, time_bin_ms=2.0),
        batch_size=args.batch,
        device=device,
        dtype=args.dtype,
        amp=False,
        grad_checkpoint_chunks=1,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--neurons", type=int, default=64)
    parser.add_argument("--dim", type=int, default=1000)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--inputs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=10,
                        help="timesteps per timed call (one forward chunk)")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dtype", default="float16", choices=("float16", "float32"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args(argv)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("cuda requested but unavailable -- falling back to cpu")
        device = "cpu"

    torch.manual_seed(0)
    cfg = _config(args, device)
    model = VectorNeuronPopulation(cfg).eval().to(device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    # (B, steps, D): exactly what forward() hands to _segment_any for one chunk
    a_chunk = torch.randn(args.batch, args.steps, args.dim, dtype=dtype, device=device)
    z = model.init_state(args.batch, torch.device(device))

    def one_call() -> None:
        model._segment_any(z, a_chunk)

    with torch.no_grad():
        for _ in range(max(0, args.warmup)):
            one_call()
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(max(1, args.repeats)):
            one_call()
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    per_call_ms = 1e3 * elapsed / max(1, args.repeats)
    print(
        f"B={args.batch} N={args.neurons} D={args.dim} R={args.rank} "
        f"dtype={args.dtype} device={device} steps/call={args.steps}"
    )
    print(
        f"{per_call_ms / max(1, args.steps):.3f} ms/step "
        f"({per_call_ms:.3f} ms per {args.steps}-step call, "
        f"{max(1, args.repeats)} repeats)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
