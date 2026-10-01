"""V3 smoke test: verify the mechanics before any real training run.

Checks, in order:

1. surrogate-gradient maths (autograd of the surrogate primitive == our backward)
2. state shape / dtype (``z`` is genuinely ``(B, N, D)`` and genuinely fp16)
3. vector coupling (changing one coordinate of ``z`` moves other coordinates)
4. real SHD: a tiny end-to-end fit on a handful of samples (loss/acc computed)
5. scalar baseline (``D = 1``) runs through the same code path
6. full-size timing/memory probe for the primary configuration

Run::

    .venv\\Scripts\\python.exe V3\\run_smoke_test.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

V3_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(V3_ROOT))

from v3.config import V3Config, parameter_groups  # noqa: E402
from v3.data import BatchIterator, SHDEventStore, make_split, official_test_split  # noqa: E402
from v3.model import VectorNeuronPopulation, surrogate_spike  # noqa: E402
from v3.train import _rate_stats, evaluate, fit, set_seed  # noqa: E402

OK = "[OK]  "
FAIL = "[FAIL]"
_failures: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    print(f"{OK if cond else FAIL} {name}{(' -- ' + extra) if extra else ''}")
    if not cond:
        _failures.append(name)


def banner(text: str) -> None:
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


# ---------------------------------------------------------------------- #
def test_surrogate() -> None:
    banner("1. surrogate gradient")
    u = torch.linspace(-4, 4, 41, dtype=torch.float64, requires_grad=True)
    beta, gamma = 4.0, 1.0
    prim = gamma * u / (1 + beta * u.abs())
    g_auto = torch.autograd.grad(prim.sum(), u, create_graph=False)[0]
    g_ours = torch.autograd.grad(surrogate_spike(u, beta, gamma).sum(), u)[0]
    err = float((g_auto - g_ours).abs().max())
    check("backward == d/du [gamma*u/(1+beta|u|)]", err < 1e-9, f"max abs diff {err:.3e}")
    u0 = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    at0 = float(torch.autograd.grad(surrogate_spike(u0, beta, gamma).sum(), u0)[0])
    check("gradient at u=0 equals gamma", abs(at0 - gamma) < 1e-12, f"grad={at0}")
    s = surrogate_spike(torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64), beta, gamma)
    check("forward is a Heaviside", bool((s == torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)).all()))


# ---------------------------------------------------------------------- #
def test_state_dims(device: torch.device, dtype: torch.dtype) -> None:
    banner("2/3. state dimensions, dtype and vector coupling")
    cfg = V3Config(n_neurons=4, state_dim=16, n_inputs=32, n_bins=12, bin_ms=4.0, mix_rank=5,
                   device=str(device), dtype=str(dtype).replace("torch.", ""))
    model = VectorNeuronPopulation(cfg).to(device)
    B, T, C = 3, 12, 32
    x = (torch.rand(B, T, C, device=device) < 0.02).to(cfg.torch_dtype)
    z = model.init_state(B, device)
    check("initial state shape", tuple(z.shape) == (B, 4, 16), str(tuple(z.shape)))
    check("initial state dtype is fp16", z.dtype == torch.float16, str(z.dtype))

    _, spikes, pots = model.forward(x, return_potentials=True)
    check("spike tensor shape (B,T,N)", tuple(spikes.shape) == (B, T, 4), str(tuple(spikes.shape)))
    check("spikes are 0/1", bool(((spikes == 0) | (spikes == 1)).all()))
    check("potentials shape (B,T,N)", tuple(pots.shape) == (B, T, 4))

    # run a single segment and inspect the resulting state
    a = x.reshape(B * T, C).to(cfg.torch_dtype) @ model.w_in.to(cfg.torch_dtype)
    a = a.reshape(B, T, cfg.state_dim)
    z2, s2, _ = model._segment(model.init_state(B, device), a)
    check("updated state shape (B,N,D)", tuple(z2.shape) == (B, 4, 16), str(tuple(z2.shape)))
    check("state stays fp16 after the update", z2.dtype == torch.float16, str(z2.dtype))

    # vector coupling: perturbing ONE coordinate of the state must move others
    model.eval()
    z_before, _, _ = model._segment(torch.zeros_like(z2), a[:, :1])
    z0 = torch.zeros_like(z2)
    z0[:, :, 0] = 0.9                                # only coordinate 0 is non-zero
    z_after, _, _ = model._segment(z0, a[:, :1])
    moved = (z_after[:, :, 1:] - z_before[:, :, 1:]).abs().max()
    check("D>1 coordinates are coupled (rank-R mixing)", float(moved) > 1e-4, f"max |delta| on coords 1..D-1 = {float(moved):.3e}")


# ---------------------------------------------------------------------- #
def test_scalar_baseline(device: torch.device) -> None:
    banner("5. scalar baseline (D = 1) uses the same code path")
    cfg = V3Config(n_neurons=8, state_dim=1, n_inputs=32, n_bins=10, mix_rank=4, bin_ms=4.0,
                   device=str(device), dtype="float16")
    model = VectorNeuronPopulation(cfg).to(device)
    check("D=1 marked as scalar baseline", model.is_scalar_baseline)
    x = (torch.rand(2, 10, 32, device=device) < 0.05).to(cfg.torch_dtype)
    logits, spikes = model(x)
    check("scalar baseline forward", tuple(logits.shape) == (2, 20) and tuple(spikes.shape) == (2, 10, 8))


# ---------------------------------------------------------------------- #
def test_real_shd(device: torch.device) -> None:
    banner("4. real SHD, tiny end-to-end fit")
    cfg = V3Config(
        tag="smoke", n_neurons=8, state_dim=64, mix_rank=8, n_bins=250, bin_ms=4.0,
        batch_size=8, epochs=2, learning_rate=2e-3, val_fraction=0.25, split_seed=0,
        device=str(device), dtype="float16", cache_events=True, log_every=0,
    )
    store = SHDEventStore(cfg.path(cfg.train_h5), cfg)
    fit_split, val_split = make_split(store, cfg)
    sub = fit_split.indices[:64]
    from v3.data import Split

    small = Split("fit_small", sub, store.labels[sub], store.speakers[sub])
    check("split sizes", len(small) == 64 and len(val_split) > 0, f"{len(small)} / {len(val_split)}")
    xb = store.batch(sub[:4])
    check("binning produces (B,T,C)", xb.shape == (4, 250, 700), str(xb.shape))
    check("binned input is binary", set(np.unique(xb).tolist()) <= {0.0, 1.0})

    model = VectorNeuronPopulation(cfg).to(device)
    model.calibrate_threshold(
        torch.from_numpy(xb).to(device), target_rate_hz=cfg.rate_target_hz
    )
    with torch.no_grad():
        _, sp = model(torch.from_numpy(xb).to(device))
        r = float(model.rate_hz(sp).mean())
    check("threshold calibration gives a sane initial rate", 1.0 < r < 60.0, f"{r:.1f} Hz")
    res = fit(model, store, small, None, cfg, device, show_progress=True)
    check("training ran 2 epochs", len(res.history) == 2)
    check("loss is finite and recorded", all(np.isfinite(h["loss"]) for h in res.history))
    check("accuracy recorded", all(0.0 <= h["accuracy"] <= 1.0 for h in res.history))

    grad_missing = [n for n, p in model.named_parameters() if p.grad is None]
    check("every parameter received a gradient", not grad_missing, str(grad_missing[:4]))
    ev = evaluate(model, store, small, cfg, device)
    check("evaluate() returns accuracy", 0.0 <= ev["accuracy"] <= 1.0, f"acc={ev['accuracy']:.3f}")
    check("evaluate() returns rate stats", ev["rate_max_hz"] > 0, f"rate_mean={ev['rate_mean_hz']:.1f} Hz")
    store.close()


def test_parameter_update(device: torch.device) -> None:
    banner("6. parameters actually move (fp16 fwd + fp32 master weights)")
    cfg = V3Config(n_neurons=6, state_dim=32, n_inputs=64, n_bins=8, mix_rank=4, bin_ms=4.0,
                   device=str(device), dtype="float16", batch_size=4, learning_rate=1e-2,
                   rate_reg=0.0, grad_checkpoint_chunks=2)
    set_seed(0)
    model = VectorNeuronPopulation(cfg).to(device)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    x = (torch.rand(4, 8, 64, device=device) < 0.05).to(cfg.torch_dtype)
    y = torch.randint(0, 20, (4,), device=device)
    model.train()
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        logits, spikes = model(x)
        loss, _ = model.loss(logits, spikes, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    moved = {n: float((p.detach() - before[n]).abs().max()) for n, p in model.named_parameters()}
    n_moved = sum(1 for v in moved.values() if v > 0)
    check("all parameter tensors updated", n_moved == len(moved), f"{n_moved}/{len(moved)} moved")
    check("all updates finite", all(np.isfinite(v) for v in moved.values()))
    check("master weights remain fp32", all(p.dtype == torch.float32 for p in model.parameters()))


# ---------------------------------------------------------------------- #
def test_rate_stats() -> None:
    banner("8. firing-rate statistics are per-sample normalised")
    cfg = V3Config(n_bins=250, bin_ms=4.0)
    # 10 samples, 250 steps, 8 neurons; neuron j fires every (j+1)-th step
    s = torch.zeros(10, 250, 8)
    for j in range(8):
        s[:, :: (j + 1), j] = 1.0
    st = _rate_stats(s, cfg)
    st1 = _rate_stats(s[:1], cfg)
    # a firing *rate* is intensive: it must not change with the number of samples.
    # (the earlier bug summed over the split but divided only by T, so the 10-sample
    #  result was 10x the 1-sample result)
    check("mean rate is invariant to the number of samples",
          abs(st["rate_mean_hz"] - st1["rate_mean_hz"]) < 1e-6,
          f"{st['rate_mean_hz']:.4f} Hz vs {st1['rate_mean_hz']:.4f} Hz")
    check("spikes per sample is invariant to the number of samples",
          abs(st["spikes_per_sample"] - st1["spikes_per_sample"]) < 1e-6,
          f"{st['spikes_per_sample']:.3f} vs {st1['spikes_per_sample']:.3f}")
    # hand-computed from the construction: counts are 250,125,84,63,50,42,36,32
    # (tolerance is fp32: the rates are computed as count/T/dt in float32)
    check("mean rate matches the hand-computed value",
          abs(st["rate_mean_hz"] - 85.25) < 1e-4, f"{st['rate_mean_hz']:.4f} Hz (expected 85.25)")
    check("spikes per sample matches the hand-computed value",
          abs(st["spikes_per_sample"] - 682.0) < 1e-6, str(st["spikes_per_sample"]))
    check("no neuron is silent in this construction", st["silent_neurons"] == 0)
    check("max rate is the always-firing neuron", abs(st["rate_max_hz"] - 250.0) < 1e-4,
          f"{st['rate_max_hz']:.2f} Hz")


# ---------------------------------------------------------------------- #
def test_full_size_probe(device: torch.device) -> None:
    banner("7. full-size probe (N=64, D=1000, T=250, B=32) -- timing + memory")
    if device.type != "cuda":
        print("no CUDA -- skipped")
        return
    cfg = V3Config(n_neurons=64, state_dim=1000, mix_rank=64, n_bins=250, bin_ms=4.0,
                   batch_size=32, device="cuda", dtype="float16", grad_checkpoint_chunks=10)
    model = VectorNeuronPopulation(cfg).to(device)
    groups = parameter_groups(cfg)
    check("analytic parameter count matches", groups["total"] == model.n_parameters(),
          f"{model.n_parameters():,} params")
    x = (torch.rand(32, 250, 700, device=device) < 0.045).to(cfg.torch_dtype)
    y = torch.randint(0, 20, (32,), device=device)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    logits, spikes = model(x)
    loss, _ = model.loss(logits, spikes, y)
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**20
    n_batches = int(np.ceil(8156 * 0.9 / 32))
    print(f"  fwd+bwd one batch: {dt * 1000:.0f} ms  peak alloc {peak:.0f} MiB")
    print(f"  -> ~{dt * n_batches:.1f} s/epoch for {n_batches} batches")
    check("full-size step fits in 6 GB", peak < 5500, f"peak {peak:.0f} MiB")
    check("forward produced finite logits", bool(torch.isfinite(logits).all()))
    check("spike tensor is fp16", spikes.dtype == cfg.torch_dtype, str(spikes.dtype))
    check("logits are computed in fp32", logits.dtype == torch.float32, str(logits.dtype))


# ---------------------------------------------------------------------- #
def main() -> int:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__}  device {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    test_surrogate()
    test_state_dims(device, torch.float16)
    test_scalar_baseline(device)
    test_parameter_update(device)
    test_real_shd(device)
    test_rate_stats()
    test_full_size_probe(device)
    banner("summary")
    if _failures:
        print(f"{len(_failures)} check(s) FAILED: {_failures}")
        return 1
    print("all smoke checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
