"""torch.compile feasibility gate for the fused ``_segment`` kernel (item 5).

Microbenchmarks one ``_segment`` call (B=128, 10 timesteps) on the reference
geometry (N=64, D=1000, C=700, fp16 state) with and without torch.compile.
Prints the eager time, the compiled time (if it compiles at all), the first-call
compile latency, and the failure mode otherwise.  Never writes the registry.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v3.config import TimingConfig, V3Config  # noqa: E402
from v3.model import VectorNeuronPopulation  # noqa: E402


def make_cfg() -> V3Config:
    return V3Config(
        n_neurons=64, state_dim=1000, mix_rank=64, n_inputs=700, n_classes=20,
        timing=TimingConfig(1000.0, 2.0), dtype="float16", amp=False, device="cuda",
    )


def step(segment, z0, a):
    z, s, p = segment(z0, a)
    loss = z.float().pow(2).mean() + s.float().pow(2).mean() + p.float().pow(2).mean()
    loss.backward()
    return float(loss.detach())


def timed(fn, iters: int, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args(argv)
    print(f"[compile-probe] torch {torch.__version__}, cuda {torch.version.cuda}, "
          f"device {torch.cuda.get_device_name(0)}", flush=True)

    cfg = make_cfg()
    model = VectorNeuronPopulation(cfg).to("cuda")
    # forward() projects the input to D before chunking, so _segment sees (B, S, D)
    a = (torch.rand(args.batch, args.steps, cfg.state_dim, device="cuda") < 0.02).half()
    z0 = model.init_state(args.batch, torch.device("cuda"))

    eager = timed(lambda: step(model._segment, z0, a), args.iters)
    print(f"[compile-probe] eager      {eager * 1e3:8.2f} ms/step-call", flush=True)

    for label, kwargs in (
        ("default", dict(dynamic=False)),
        ("fullgraph", dict(dynamic=False, fullgraph=True)),
    ):
        try:
            t0 = time.perf_counter()
            fn = torch.compile(model._segment, **kwargs)
            step(fn, z0, a)
            torch.cuda.synchronize()
            compile_s = time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001 - the failure mode *is* the result
            print(f"[compile-probe] {label:9s} FAILED to compile in {time.perf_counter() - t0:.1f}s: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            tb = traceback.format_exc().splitlines()[-6:]
            print("\n".join("    " + line for line in tb), flush=True)
            continue
        compiled = timed(lambda: step(fn, z0, a), args.iters)
        print(f"[compile-probe] {label:9s} {compiled * 1e3:8.2f} ms/step-call "
              f"({eager / compiled:.2f}x, compile {compile_s:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
