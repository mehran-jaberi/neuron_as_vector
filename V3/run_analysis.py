"""Mechanism analysis of the trained V3 vector-neuron state.

This is the infrastructure for the first two items of ``V3/TODO.md`` §Analysis:

* does the raw ``(N, D)`` state carry class information beyond the spike-count
  readout?  (linear probe on captured state features)
* how many of the ``D = 1000`` state dimensions are actually used?
  (PCA / participation ratio)

Three subcommands, none of which trains or modifies the model:

    python V3/run_analysis.py capture      # frozen checkpoint -> per-sample features
    python V3/run_analysis.py probe        # logistic probes (fitted on FIT, L2 on VAL)
    python V3/run_analysis.py geometry     # PCA / effective-dimensionality diagnostics

``capture`` writes ``V3/results/analysis/capture_{split}.npz`` for the vector
checkpoint (``--tag`` prefixes the file name, e.g. ``scalar_d1``) with

    counts       (n, N)        spike counts, exactly the readout feature
    final_state  (n, N, D)     state after the last update (float16)
    mean_state   (n, N, D)     time-mean of the state read at each timestep (float16)
    labels       (n,)          class labels
    logits       (n, K)        the model's own readout scores
    indices      (n,)          row indices into the corresponding SHD file

The primary probe is a multinomial logistic regression (linear softmax) in torch:
features are standardised on the fitting split, the L2 strength is chosen on the
probe's validation split, and the official TEST split is only ever scored.  A
legacy ridge probe on one-hot targets is kept (`probe --estimator ridge`) solely
to reproduce the earlier estimator-limited table.  TEST labels are never used to
fit or to select anything.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from v3.config import V3Config
from v3.data import (
    BatchIterator,
    SHDEventStore,
    make_split,
    official_test_split,
)
from v3.experiment import effective_config, resolve_device
from v3.model import VectorNeuronPopulation

V3_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = V3_DIR / "results" / "analysis"
CHECKPOINT_DIR = V3_DIR / "checkpoints"
VECTOR_CHECKPOINT = CHECKPOINT_DIR / "v3_vector_d1000.pt"
SCALAR_CHECKPOINT = CHECKPOINT_DIR / "v3_vector_scalar_d1.pt"

SPLITS = ("fit", "val", "test")
# evaluation phase passed to the model (state imprecision is off in the baseline
# checkpoint anyway; "test" is the documented phase for the official test file).
SPLIT_PHASE = {"fit": "val", "val": "val", "test": "test"}

#: Each feature set is a concatenation of per-source blocks; every block is
#: standardised independently on the fitting split (equivalent to concatenating
#: the standardised blocks and fitting one ridge).
FEATURE_SETS = {
    "counts": ("counts",),
    "final_state": ("final_state",),
    "mean_state": ("mean_state",),
    "counts+mean_state": ("counts", "mean_state"),
}

#: Feature sets of the primary (logistic) protocol, counts first because the
#: counts result is the estimator-validity gate.
LOGISTIC_FEATURE_ORDER = ("counts", "mean_state", "final_state", "counts+mean_state")

#: One estimator for every feature set: multinomial logistic regression,
#: Adam, batch 256, 30 epochs, all hyper-parameters fixed except the L2
#: strength, which is chosen on the probe's validation split (VAL).
LOGISTIC_LR = 0.01
LOGISTIC_EPOCHS = 30
LOGISTIC_BATCH = 256
LOGISTIC_LAM_GRID = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
LOGISTIC_SEED = 0

#: Estimator-validity gate: the counts-only probe must land within 1 pp of the
#: model's own counts-only readout (1868/2264 = 82.51 % on the official TEST
#: split), i.e. 1845-1890 correct out of 2264.  If it does not, every
#: state-vs-counts number is estimator-limited and must not be interpreted.
COUNTS_GATE_CORRECT = (1845, 1890)


# ---------------------------------------------------------------------- #
# paths / checkpoint loading
# ---------------------------------------------------------------------- #
def capture_path(tag: str, split: str) -> Path:
    name = f"capture_{tag}_{split}.npz" if tag else f"capture_{split}.npz"
    return ANALYSIS_DIR / name


def results_path(stem: str, tag: str = "") -> Path:
    name = f"{stem}_{tag}.json" if tag else f"{stem}.json"
    return ANALYSIS_DIR / name


def load_checkpoint(path: str | Path, device: str = "auto"):
    """Rebuild a frozen model from a ``v3_checkpoint/v1`` file."""
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if "config" not in payload or "state_dict" not in payload:
        raise ValueError(f"{path}: not a v3 checkpoint (missing config/state_dict)")
    cfg = V3Config.from_dict(payload["config"])
    if device in (None, "", "auto"):
        dev = resolve_device(cfg)
    else:
        dev = torch.device(device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
    cfg = effective_config(cfg, dev)
    model = VectorNeuronPopulation(cfg)
    model.load_state_dict(payload["state_dict"])
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg, dev, payload


def _to_device(x: np.ndarray, cfg: V3Config, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(x)
    if cfg.dtype != "float32":
        t = t.to(cfg.torch_dtype)  # 0/1 input: lossless
    return t.to(device, non_blocking=True)


# ---------------------------------------------------------------------- #
# capture
# ---------------------------------------------------------------------- #
@torch.no_grad()
def capture_split(
    model: VectorNeuronPopulation,
    cfg: V3Config,
    device: torch.device,
    store: SHDEventStore,
    split,
    phase: str,
    batch_size: int,
) -> dict:
    """Run one split, return the feature arrays + readout accuracy."""
    from tqdm.auto import tqdm

    n, N, D = len(split), model.n_neurons, model.state_dim
    K = int(cfg.n_classes)
    counts = np.zeros((n, N), dtype=np.float32)
    final_state = np.zeros((n, N, D), dtype=np.float16)
    mean_state = np.zeros((n, N, D), dtype=np.float16)
    logits = np.zeros((n, K), dtype=np.float32)
    labels = np.asarray(split.labels, dtype=np.int64)
    indices = np.asarray(split.indices, dtype=np.int64)

    model.eval()
    model.set_phase(phase)
    it = BatchIterator(store, split, batch_size, shuffle=False)
    tic = time.perf_counter()
    pos = 0
    for x_np, y_np in tqdm(it, total=len(it), desc=split.name, leave=False):
        x = _to_device(x_np, cfg, device)
        logits_b, spikes_b, z_final, z_mean = model(x, return_states=True)
        b = int(y_np.size)
        counts[pos : pos + b] = spikes_b.float().sum(1).cpu().numpy()
        final_state[pos : pos + b] = z_final.cpu().numpy()
        mean_state[pos : pos + b] = z_mean.cpu().numpy()
        logits[pos : pos + b] = logits_b.float().cpu().numpy()
        pos += b
    seconds = time.perf_counter() - tic

    preds = logits.argmax(1)
    correct = int((preds == labels).sum())
    total = int(labels.size)
    return {
        "counts": counts,
        "final_state": final_state,
        "mean_state": mean_state,
        "labels": labels,
        "logits": logits,
        "indices": indices,
        "split": split.name,
        "n": total,
        "correct": correct,
        "accuracy": correct / max(total, 1),
        "seconds": seconds,
    }


def run_capture(args: argparse.Namespace) -> dict:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(args.checkpoint)
    model, cfg, device, payload = load_checkpoint(path, args.device)
    batch_size = int(args.batch_size or cfg.batch_size)
    print(
        f"[capture] {path.name}: tag={cfg.tag} epochs={cfg.epochs} "
        f"N={model.n_neurons} D={model.state_dim} T={cfg.n_bins} "
        f"device={device} dtype={cfg.dtype} batch={batch_size}"
    )

    store_train = SHDEventStore(cfg.path(cfg.train_h5), cfg)
    store_test = SHDEventStore(cfg.path(cfg.test_h5), cfg)
    fit_split, val_split = make_split(store_train, cfg)
    test_split = official_test_split(store_test)
    available = {
        "fit": (store_train, fit_split),
        "val": (store_train, val_split),
        "test": (store_test, test_split),
    }

    wanted = [s.strip() for s in args.splits.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in available]
    if unknown:
        raise SystemExit(f"unknown splits {unknown}; choose from {SPLITS}")

    meta = {
        "checkpoint": str(path),
        "checkpoint_tag": cfg.tag,
        "epochs": int(cfg.epochs),
        "seed": int(cfg.seed),
        "n_neurons": int(model.n_neurons),
        "state_dim": int(model.state_dim),
        "n_bins": int(cfg.n_bins),
        "readout": str(cfg.readout),
        "device": str(device),
        "dtype": str(cfg.dtype),
        "batch_size": batch_size,
        "state_note": (
            "final_state = state after the last update; mean_state = fp32 accumulation "
            "of the state read at each of the T timesteps, divided by T"
        ),
        "splits": {},
    }
    for name in wanted:
        store, split = available[name]
        out = capture_split(
            model, cfg, device, store, split, SPLIT_PHASE[name], batch_size
        )
        out_path = capture_path(args.tag, name)
        np.savez(
            out_path,
            counts=out["counts"],
            final_state=out["final_state"],
            mean_state=out["mean_state"],
            labels=out["labels"],
            logits=out["logits"],
            indices=out["indices"],
        )
        meta["splits"][name] = {
            "n": out["n"],
            "correct": out["correct"],
            "accuracy": out["accuracy"],
            "seconds": out["seconds"],
            "file": str(out_path),
        }
        print(
            f"[capture] {name}: {out['correct']}/{out['n']} = "
            f"{out['accuracy'] * 100:.2f}%  ({out['seconds']:.1f}s) -> {out_path.name}"
        )

    store_train.close()
    store_test.close()
    meta_path = results_path("capture_meta", args.tag)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[capture] meta -> {meta_path}")
    return meta


# ---------------------------------------------------------------------- #
# ridge probe helpers (torch only; CPU by default)
# ---------------------------------------------------------------------- #
def _load_labels(path: Path) -> np.ndarray:
    with np.load(path) as z:
        return np.asarray(z["labels"], dtype=np.int64)


def _one_hot(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    y = torch.zeros(labels.size, n_classes, dtype=torch.float64)
    y[torch.arange(labels.size), torch.from_numpy(labels).long()] = 1.0
    return y


def _load_matrix(path: Path, key: str, dtype=torch.float32) -> torch.Tensor:
    """Load one ``(n, ...)`` array from a capture file as a 2-D fp32 tensor."""
    with np.load(path) as z:
        arr = z[key]
    n = int(arr.shape[0])
    d = int(np.prod(arr.shape[1:], dtype=np.int64))
    target = 64 * 2**20  # elements per staging block (~128 MB fp16 / 256 MB fp32)
    block = max(1, min(n, target // max(d, 1)))
    out = torch.empty(n, d, dtype=dtype)
    for i in range(0, n, block):
        chunk = arr[i : i + block].reshape(-1, d)
        out[i : i + block] = torch.from_numpy(chunk).to(dtype)
    return out


def _col_stats(x: torch.Tensor, block: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    """Column mean / standard deviation without a full-size temporary."""
    n, d = x.shape
    s = torch.zeros(d, dtype=torch.float32, device=x.device)
    q = torch.zeros(d, dtype=torch.float32, device=x.device)
    for i in range(0, n, block):
        c = x[i : i + block]
        s += c.sum(0)
        q += c.square().sum(0)
    mean = s / n
    var = (q / n - mean.square()).clamp_min_(0.0)
    std = var.sqrt()
    # a silent feature must not blow up: floor at 1e-3 of the median scale
    floor = max(float(std.median()) * 1e-3, 1e-6)
    return mean, std.clamp_min_(floor)


def _standardize_(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x.sub_(mean).div_(std)


def _linear_kernel(a: torch.Tensor, b: torch.Tensor, block: int = 1024) -> torch.Tensor:
    """``a @ b.T`` in row blocks, fp32."""
    out = torch.empty(a.shape[0], b.shape[0], dtype=torch.float32)
    bt = b.t()
    for i in range(0, a.shape[0], block):
        out[i : i + block] = a[i : i + block] @ bt
    return out


def _build_source_kernels(
    fit_path: Path, test_path: Path, key: str
) -> dict:
    """Standardise one feature source on the fitting split and build kernels.

    Returns ``K_fit = Xf Xf^T`` and ``K_test = Xt Xf^T`` (fp32, CPU) plus the
    shape information.  Only the fitting split's own statistics are used.
    """
    xf = _load_matrix(fit_path, key)
    xt = _load_matrix(test_path, key)
    mean, std = _col_stats(xf)
    xf = _standardize_(xf, mean, std)
    xt = _standardize_(xt, mean, std)
    k_fit = _linear_kernel(xf, xf)
    k_test = _linear_kernel(xt, xf)
    info = {
        "source": key,
        "n_fit": int(xf.shape[0]),
        "n_test": int(xt.shape[0]),
        "dim": int(xf.shape[1]),
        "trace_over_n": float(torch.diagonal(k_fit).mean()),
    }
    return {"K_fit": k_fit, "K_test": k_test, "info": info}


def default_lam_grid(k_fit: torch.Tensor, exponents=range(-10, 2)) -> list[float]:
    """Ridge grid relative to the mean diagonal (mean squared row norm)."""
    scale = float(torch.diagonal(k_fit).mean())
    return [scale * 10.0**e for e in exponents]


def ridge_eig(
    k_fit: torch.Tensor, y_fit: torch.Tensor, dtype=torch.float64
) -> dict:
    """One eigendecomposition of the fitting kernel, shared by every criterion.

    With ``K = X X^T = U diag(s) U^T`` the dual ridge solution for a strength
    ``lam`` is ``alpha = U diag(1/(s+lam)) U^T (y - y_mean)`` and the scores on
    new samples are ``K_new @ alpha + y_mean`` (centred features, no explicit
    bias).  ``u2`` / ``yc`` are cached because GCV re-uses them for every grid
    point.
    """
    k = k_fit.to(dtype)
    y = y_fit.to(dtype)
    y_mean = y.mean(0)
    yc = y - y_mean
    s, u = torch.linalg.eigh(k)
    s = s.clamp_min_(0.0)
    return {
        "s": s,
        "u": u,
        "uty": u.t() @ yc,
        "yc": yc,
        "u2": u.square(),
        "y_mean": y_mean,
        "n_fit": int(k.shape[0]),
        "trace_over_n": float(s.mean()),
    }


def ridge_alpha(eig: dict, lam: float) -> torch.Tensor:
    """Dual coefficients for one ridge strength."""
    return eig["u"] @ (eig["uty"] * (1.0 / (eig["s"] + float(lam)))[:, None])


def gcv_scores(eig: dict, lam_grid) -> tuple[dict, float, float]:
    """Generalised cross-validation curve (squared-error criterion)."""
    s, u, uty, u2, yc = eig["s"], eig["u"], eig["uty"], eig["u2"], eig["yc"]
    gcv = {}
    best_lam, best_score = None, float("inf")
    for lam in lam_grid:
        lam = float(lam)
        coef = 1.0 / (s + lam)
        resid = yc - u @ (uty * coef[:, None])
        h = u2 @ (s * coef)
        denom = (1.0 - h).clamp_min(1e-12)
        score = float((resid.square().sum(1) / denom.square()).mean())
        gcv[repr(lam)] = score
        if score < best_score:
            best_lam, best_score = lam, score
    return gcv, float(best_lam), best_score


def fit_ridge_gcv(
    k_fit: torch.Tensor,
    y_fit: torch.Tensor,
    lam_grid: list[float],
) -> tuple[torch.Tensor, torch.Tensor, float, dict]:
    """Ridge on one-hot targets with the ridge strength chosen by GCV.

    Returns ``(alpha, y_mean, lam, report)``.
    """
    eig = ridge_eig(k_fit, y_fit)
    gcv, best_lam, _ = gcv_scores(eig, lam_grid)
    alpha = ridge_alpha(eig, best_lam)
    report = {
        "lam": best_lam,
        "lam_over_trace_mean": best_lam / eig["trace_over_n"]
        if eig["trace_over_n"]
        else float("nan"),
        "gcv": gcv,
        "n_fit": eig["n_fit"],
        "trace_over_n": eig["trace_over_n"],
        "rank": int((eig["s"] > eig["s"].max() * 1e-10).sum())
        if float(eig["s"].max())
        else 0,
    }
    return alpha, eig["y_mean"], best_lam, report


def predict_ridge(
    kernel: torch.Tensor, alpha: torch.Tensor, y_mean: torch.Tensor
) -> torch.Tensor:
    """Class scores for samples whose kernel against the fitting set is given."""
    return kernel.to(alpha.dtype) @ alpha + y_mean


def stratified_folds(labels, n_folds: int, seed: int = 0) -> torch.Tensor:
    """Fold id per sample, balanced within every class."""
    y = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    g = torch.Generator().manual_seed(int(seed))
    fold = torch.zeros(y.numel(), dtype=torch.long)
    for c in torch.unique(y):
        idx = torch.nonzero(y == c, as_tuple=False).flatten()
        perm = idx[torch.randperm(idx.numel(), generator=g)]
        fold[perm] = torch.arange(perm.numel()) % n_folds
    return fold


def select_lam_cv(
    k_fit: torch.Tensor,
    y_fit: torch.Tensor,
    labels,
    lam_grid: list[float],
    n_folds: int = 5,
    seed: int = 0,
) -> tuple[float, dict, dict]:
    """K-fold cross-validated *accuracy* over ``lam_grid`` inside the fitting split.

    The criterion is classification accuracy (not squared error), and the held
    out fold is always a subset of the fitting split; TEST is never involved.
    Folds use fp32 eigen-decompositions (selection only).
    """
    y = torch.as_tensor(np.asarray(y_fit), dtype=torch.float32)
    y_lab = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    fold = stratified_folds(labels, n_folds, seed)
    correct = torch.zeros(len(lam_grid), dtype=torch.float64)
    for f in range(n_folds):
        tr = fold != f
        va = fold == f
        k_tr = k_fit[tr][:, tr].float()
        k_va = k_fit[va][:, tr].float()
        eig = ridge_eig(k_tr, y[tr], dtype=torch.float32)
        for i, lam in enumerate(lam_grid):
            alpha = ridge_alpha(eig, lam)
            pred = (k_va @ alpha + eig["y_mean"]).argmax(1)
            correct[i] += int((pred == y_lab[va]).sum())
        del k_tr, k_va
    curve = (correct / float(k_fit.shape[0])).tolist()
    best = int(torch.as_tensor(curve).argmax())
    return float(lam_grid[best]), {"cv_accuracy": curve[best], "curve": curve}, {
        "n_folds": int(n_folds),
        "seed": int(seed),
    }


# ---------------------------------------------------------------------- #
# multinomial logistic probe (primary estimator)
# ---------------------------------------------------------------------- #
def _load_concat(path: Path, keys: tuple[str, ...], dtype=torch.float32) -> torch.Tensor:
    """Load several ``(n, ...)`` capture arrays as one ``(n, sum(d))`` matrix.

    Blocked so that a 64k-dim state split never needs two full-size buffers.
    """
    with np.load(path) as z:
        parts = [z[k] for k in keys]
    n = int(parts[0].shape[0])
    cols = [int(np.prod(p.shape[1:], dtype=np.int64)) for p in parts]
    out = torch.empty(n, sum(cols), dtype=dtype)
    target = 64 * 2**20  # elements per staging block
    off = 0
    for p, d in zip(parts, cols):
        block = max(1, min(n, target // max(d, 1)))
        for i in range(0, n, block):
            out[i : i + block, off : off + d] = torch.from_numpy(
                p[i : i + block].reshape(-1, d)
            ).to(dtype)
        off += d
    return out


def load_split_features(
    path: Path, sources: tuple[str, ...], device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Load one feature set for one split, ``(n, dim)`` fp32 on ``device``."""
    x = (
        _load_matrix(path, sources[0])
        if len(sources) == 1
        else _load_concat(path, sources)
    )
    return x.to(torch.device(device), non_blocking=False)


def _train_logistic(
    x_fit: torch.Tensor,
    y_fit: np.ndarray | torch.Tensor,
    *,
    lam: float,
    lr: float = LOGISTIC_LR,
    epochs: int = LOGISTIC_EPOCHS,
    batch_size: int = LOGISTIC_BATCH,
    seed: int = LOGISTIC_SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One multinomial logistic regression at a fixed L2 strength.

    Loss is mean cross-entropy plus ``lam * sum(W**2)`` (bias unpenalised);
    ``lam`` is the only tuned quantity.  Deterministic for a fixed seed:
    zero initialisation and one seeded shuffle generator.
    """
    n, d = x_fit.shape
    k = int(np.asarray(y_fit).max()) + 1
    device = x_fit.device
    torch.manual_seed(int(seed))
    w = torch.zeros(d, k, dtype=torch.float32, device=device, requires_grad=True)
    b = torch.zeros(k, dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=float(lr))
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    yt = torch.as_tensor(np.asarray(y_fit), dtype=torch.long, device=device)
    for _ in range(int(epochs)):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, int(batch_size)):
            idx = perm[i : i + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(x_fit[idx] @ w + b, yt[idx])
            if lam:
                loss = loss + float(lam) * w.square().sum()
            loss.backward()
            opt.step()
    return w.detach(), b.detach()


def _linear_accuracy(
    w: torch.Tensor, b: torch.Tensor, x: torch.Tensor, y: np.ndarray | torch.Tensor
) -> tuple[int, int]:
    with torch.no_grad():
        pred = (x @ w + b).argmax(1)
        yt = torch.as_tensor(np.asarray(y), dtype=torch.long, device=x.device)
        return int((pred == yt).sum()), int(yt.numel())


def fit_logistic_probe(
    x_fit: torch.Tensor,
    y_fit: np.ndarray,
    x_val: torch.Tensor,
    y_val: np.ndarray,
    *,
    lam: float | None = None,
    lam_grid=LOGISTIC_LAM_GRID,
    standardize: bool = True,
    lr: float = LOGISTIC_LR,
    epochs: int = LOGISTIC_EPOCHS,
    batch_size: int = LOGISTIC_BATCH,
    seed: int = LOGISTIC_SEED,
) -> dict:
    """Fit the primary probe; the L2 strength is selected on VAL.

    Structural no-leakage contract: this function never sees TEST.  Feature
    standardisation uses only the fitting split ``x_fit`` and the same
    statistics are stored in the returned state, so TEST can be standardised
    later by :func:`predict_logits` without affecting the fit.  Note that
    ``x_fit`` and ``x_val`` are standardised **in place** (the 64k state
    matrices are too large to copy on the GPU); pass copies if the caller
    needs the raw values afterwards.

    ``lam`` given -> skip selection (tests); otherwise every value of
    ``lam_grid`` is trained and VAL accuracy picks the winner (ties go to the
    stronger penalty).
    """
    x_fit = x_fit.float()
    x_val = x_val.float()
    device = x_fit.device
    labels = np.asarray(y_fit)
    n_classes = int(
        max(labels.max(), int(np.asarray(y_val).max())) + 1
    )
    feature_mean = feature_std = None
    if standardize:
        feature_mean, feature_std = _col_stats(x_fit)
        x_fit = _standardize_(x_fit, feature_mean, feature_std)
        x_val = _standardize_(x_val, feature_mean, feature_std)

    candidates = [float(v) for v in ([lam] if lam is not None else lam_grid)]
    curve: dict[str, float] = {}
    best = None
    for cand in candidates:
        w, b = _train_logistic(
            x_fit,
            labels,
            lam=cand,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
        )
        correct, total = _linear_accuracy(w, b, x_val, y_val)
        acc = correct / max(total, 1)
        curve[repr(cand)] = acc
        if (
            best is None
            or acc > best["val_accuracy"]
            or (acc == best["val_accuracy"] and cand > best["lam"])
        ):
            best = {
                "lam": cand,
                "val_accuracy": acc,
                "val_correct": correct,
                "val_total": total,
                "weight": w,
                "bias": b,
            }
    best_w, best_b = best.pop("weight"), best.pop("bias")
    fit_correct, fit_total = _linear_accuracy(best_w, best_b, x_fit, labels)
    out = dict(best)
    out.update(
        {
            "weight": best_w.cpu(),
            "bias": best_b.cpu(),
            "fit_correct": int(fit_correct),
            "fit_total": int(fit_total),
            "fit_accuracy": fit_correct / max(fit_total, 1),
            "n_fit": int(x_fit.shape[0]),
            "dim": int(x_fit.shape[1]),
            "n_classes": int(n_classes),
            "lam_grid": candidates,
            "val_curve": curve,
            "standardized": bool(standardize),
            "feature_mean": feature_mean.cpu() if feature_mean is not None else None,
            "feature_std": feature_std.cpu() if feature_std is not None else None,
            "lr": float(lr),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "seed": int(seed),
            "device": str(device),
        }
    )
    return out


def standardize_with(state: dict, x: torch.Tensor) -> torch.Tensor:
    """Apply the probe's fitting-split standardisation to raw features."""
    mean, std = state.get("feature_mean"), state.get("feature_std")
    if mean is None or std is None:
        return x
    return (x - mean.to(x.device)) / std.to(x.device)


def predict_logits(state: dict, x: torch.Tensor) -> torch.Tensor:
    """Class scores for raw (unstandardised) features; TEST never fits anything."""
    return standardize_with(state, x.float()) @ state["weight"].to(x.device) + state[
        "bias"
    ].to(x.device)


def probe_accuracy(state: dict, x: torch.Tensor, y: np.ndarray) -> dict:
    """Accuracy of a fitted probe state on a held-out split (labels untouched)."""
    pred = predict_logits(state, x).argmax(1).cpu().numpy()
    y = np.asarray(y)
    return {
        "correct": int((pred == y).sum()),
        "total": int(y.size),
        "accuracy": float((pred == y).mean()) if y.size else 0.0,
        "predictions": pred,
    }


def paired_bootstrap(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    labels: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Paired bootstrap CI for ``accuracy(a) - accuracy(b)`` on shared samples.

    The same resample indices are applied to every model, so the CI is for the
    paired difference, not for two independent accuracies.
    """
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    labels = np.asarray(labels)
    n = int(labels.size)
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, n, size=(int(n_boot), n), dtype=np.int64)
    hit_a = (pred_a[idx] == labels[idx]).mean(1)
    hit_b = (pred_b[idx] == labels[idx]).mean(1)
    delta = hit_a - hit_b
    lo, hi = np.percentile(delta, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    observed = float(
        (pred_a == labels).mean() - (pred_b == labels).mean()
    )
    return {
        "delta": observed,
        "ci95": [float(lo), float(hi)],
        "alpha": float(alpha),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
        "mean_bootstrap_delta": float(delta.mean()),
        "p_fraction_delta_le_zero": float((delta <= 0.0).mean()),
    }


# ---------------------------------------------------------------------- #
# probe
# ---------------------------------------------------------------------- #
def _readout_accuracy(path: Path) -> dict:
    with np.load(path) as z:
        preds = np.asarray(z["logits"]).argmax(1)
        labels = np.asarray(z["labels"])
    correct = int((preds == labels).sum())
    total = int(labels.size)
    return {"correct": correct, "total": total, "accuracy": correct / max(total, 1)}


def run_probe_ridge(args: argparse.Namespace) -> dict:
    """Legacy ridge probe (kept only to reproduce the estimator-limited table)."""
    tag = args.tag or ""
    paths = {s: capture_path(tag, s) for s in SPLITS}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit(f"missing capture files: {missing}; run `capture` first")

    readout = {s: _readout_accuracy(paths[s]) for s in SPLITS}
    labels = {s: _load_labels(paths[s]) for s in SPLITS}
    n_classes = int(max(int(labels[s].max()) for s in SPLITS)) + 1
    y = {s: _one_hot(labels[s], n_classes) for s in ("fit", "val", "test")}
    folds = int(args.folds)

    # kernels for one source fit on FIT or VAL, always against TEST
    kernels: dict[str, dict[str, dict]] = {}
    for protocol in ("val", "fit"):
        kernels[protocol] = {}
        for key in ("counts", "final_state", "mean_state"):
            tic = time.perf_counter()
            kernels[protocol][key] = _build_source_kernels(
                paths[protocol], paths["test"], key
            )
            info = kernels[protocol][key]["info"]
            print(
                f"[probe] kernels {protocol:>3}-fitted {key:<12} dim={info['dim']:>6} "
                f"({time.perf_counter() - tic:.1f}s)"
            )

    results: dict[str, dict] = {}
    preds_by_entry: dict[str, np.ndarray] = {}
    best = None
    for feat, sources in FEATURE_SETS.items():
        results[feat] = {}
        for protocol in ("val", "fit"):
            k_fit = sum(kernels[protocol][src]["K_fit"] for src in sources)
            k_test = sum(kernels[protocol][src]["K_test"] for src in sources)
            lam_grid = default_lam_grid(k_fit)
            eig = ridge_eig(k_fit, y[protocol])
            gcv_curve, lam_gcv, _ = gcv_scores(eig, lam_grid)
            lam_cv, cv_report, cv_meta = select_lam_cv(
                k_fit, y[protocol], labels[protocol], lam_grid,
                n_folds=folds, seed=0,
            )
            key = f"{protocol}_fitted"
            entry = {
                "protocol": key,
                "n_fit": int(labels[protocol].size),
                "n_test": int(labels["test"].size),
                "dim": int(
                    sum(kernels[protocol][src]["info"]["dim"] for src in sources)
                ),
                "primary": "cv",
                "cv": {
                    "lam": lam_cv,
                    "lam_over_trace_mean": lam_cv / eig["trace_over_n"],
                    "cv_accuracy": cv_report["cv_accuracy"],
                    "cv_curve": cv_report["curve"],
                    "selection": cv_meta,
                },
                "gcv": {
                    "lam": lam_gcv,
                    "lam_over_trace_mean": lam_gcv / eig["trace_over_n"],
                    "lam_grid": [float(x) for x in lam_grid],
                    "gcv_curve": gcv_curve,
                },
            }
            for criterion in ("cv", "gcv"):
                lam = entry[criterion]["lam"]
                alpha = ridge_alpha(eig, lam)
                scores = predict_ridge(k_test, alpha, eig["y_mean"])
                preds = scores.argmax(1).numpy()
                fit_preds = predict_ridge(k_fit, alpha, eig["y_mean"]).argmax(1).numpy()
                entry[criterion].update({
                    "test_correct": int((preds == labels["test"]).sum()),
                    "test_total": int(labels["test"].size),
                    "test_accuracy": float((preds == labels["test"]).mean()),
                    "fit_accuracy": float((fit_preds == labels[protocol]).mean()),
                })
                if criterion == "cv":
                    preds_by_entry[f"{feat}/{key}"] = preds
            results[feat][key] = entry
            cv, gcv = entry["cv"], entry["gcv"]
            print(
                f"[probe] {feat:<17} {key:<12} n_fit={entry['n_fit']:<5} "
                f"CV lam/trace={cv['lam_over_trace_mean']:.2e} "
                f"cv={cv['cv_accuracy'] * 100:.2f}% "
                f"TEST={cv['test_accuracy'] * 100:.2f}% "
                f"(gcv lam/trace={gcv['lam_over_trace_mean']:.2e} "
                f"TEST={gcv['test_accuracy'] * 100:.2f}%)"
            )
            if best is None or cv["test_accuracy"] > best["test_accuracy"]:
                best = {
                    "feature": feat,
                    "protocol": key,
                    "criterion": "cv",
                    "test_accuracy": cv["test_accuracy"],
                    "test_correct": cv["test_correct"],
                    "test_total": cv["test_total"],
                }

    if best is not None:
        preds = preds_by_entry[f"{best['feature']}/{best['protocol']}"]
        cm = np.zeros((n_classes, n_classes), dtype=np.int64)
        np.add.at(cm, (labels["test"], preds), 1)
        best["per_class_accuracy"] = (
            cm.diagonal() / np.maximum(cm.sum(1), 1)
        ).tolist()
        best["classes_present"] = [
            int(x) for x in np.unique(labels["test"]) if int(x) < n_classes
        ]

    out = {
        "capture_tag": tag,
        "protocol": (
            "Ridge regression on one-hot targets; features standardised on the fitting "
            "split. Primary ridge strength chosen by 5-fold stratified CV accuracy "
            "inside the fitting split; GCV (squared error) recorded as a secondary "
            "criterion. FIT-fitted probes are in-sample for the model (disclose); TEST "
            "is never used to fit or select, only to score."
        ),
        "n": {s: int(labels[s].size) for s in SPLITS},
        "model_readout": readout,
        "feature_sets": {k: list(v) for k, v in FEATURE_SETS.items()},
        "probe": results,
        "best": best,
        "best_selection_note": (
            "best feature set / protocol selected post hoc on TEST accuracy of the "
            "primary (CV-selected) probe; the corresponding test number is therefore "
            "optimistic"
        ),
    }
    out_path = results_path("probe_results", tag)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"[probe] -> {out_path}")
    return out


# ---------------------------------------------------------------------- #
# primary probe: multinomial logistic regression, one protocol per feature set
# ---------------------------------------------------------------------- #
def probe_device(name: str = "auto") -> torch.device:
    """Respect the ``--device`` probe option; never silently fall back to CPU."""
    if name in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return dev


def _counts_gate_window(readout: dict, pp: float = 0.01) -> tuple[int, int]:
    """Correct-answer window ``readout ± pp`` for the counts-only gate."""
    total = int(readout["total"])
    acc = float(readout["correct"]) / max(total, 1)
    return (
        int(np.floor((acc - pp) * total)),
        int(np.floor((acc + pp) * total)),
    )


def _logistic_entry(
    feat: str,
    sources: tuple[str, ...],
    state: dict,
    scored: dict,
    labels: np.ndarray,
    n_classes: int,
    seconds: float,
) -> dict:
    """JSON-safe result row for one feature set / one probe fit."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(labels), np.asarray(scored["predictions"])), 1)
    per_class = (cm.diagonal() / np.maximum(cm.sum(1), 1)).tolist()
    present = [int(c) for c in np.unique(labels) if int(c) < n_classes]
    return {
        "feature": feat,
        "sources": list(sources),
        "dim": int(state["dim"]),
        "n_fit": int(state["n_fit"]),
        "n_val": int(state["val_total"]),
        "n_test": int(scored["total"]),
        "standardized": bool(state["standardized"]),
        "lam": float(state["lam"]),
        "val_accuracy": float(state["val_accuracy"]),
        "val_curve": state["val_curve"],
        "fit_accuracy": float(state["fit_accuracy"]),
        "test_correct": int(scored["correct"]),
        "test_total": int(scored["total"]),
        "test_accuracy": float(scored["accuracy"]),
        "per_class_accuracy": per_class,
        "classes_present": present,
        "seconds": float(seconds),
        "lr": float(state["lr"]),
        "epochs": int(state["epochs"]),
        "batch_size": int(state["batch_size"]),
        "seed": int(state["seed"]),
    }


def _run_logistic_feature_set(
    paths: dict[str, Path],
    feat: str,
    sources: tuple[str, ...],
    labels: dict[str, np.ndarray],
    n_classes: int,
    device: torch.device,
    standardize: bool,
    lr: float = LOGISTIC_LR,
    epochs: int = LOGISTIC_EPOCHS,
) -> tuple[dict, np.ndarray]:
    """Load features, fit on FIT with L2 selected on VAL, score TEST once."""
    tic = time.perf_counter()
    x_fit = load_split_features(paths["fit"], sources, device)
    x_val = load_split_features(paths["val"], sources, device)
    state = fit_logistic_probe(
        x_fit,
        labels["fit"],
        x_val,
        labels["val"],
        standardize=standardize,
        lr=lr,
        epochs=epochs,
    )
    del x_fit, x_val
    x_test = load_split_features(paths["test"], sources, device)
    scored = probe_accuracy(state, x_test, labels["test"])
    del x_test
    if device.type == "cuda":
        torch.cuda.empty_cache()
    entry = _logistic_entry(
        feat,
        sources,
        state,
        scored,
        labels["test"],
        n_classes,
        time.perf_counter() - tic,
    )
    print(
        f"[probe-logistic] {feat:<17} dim={entry['dim']:>6} "
        f"lam={entry['lam']:.2e} VAL={entry['val_accuracy'] * 100:.2f}% "
        f"TEST={entry['test_correct']}/{entry['test_total']} = "
        f"{entry['test_accuracy'] * 100:.2f}% "
        f"(fit {entry['fit_accuracy'] * 100:.2f}%, {entry['seconds']:.1f}s)",
        flush=True,
    )
    return entry, np.asarray(scored["predictions"])


def run_probe_logistic(args: argparse.Namespace) -> dict:
    """Primary probe: one multinomial-logistic protocol for every feature set.

    Counts are probed first and gated against the model's own counts-only
    readout (1868/2264 = 82.51 % -> window 1845-1890).  A failing gate stops
    the run before any state feature is interpreted.
    """
    tag = args.tag or ""
    paths = {s: capture_path(tag, s) for s in SPLITS}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit(f"missing capture files: {missing}; run `capture` first")

    device = probe_device(getattr(args, "device", "auto"))
    readout = {s: _readout_accuracy(paths[s]) for s in SPLITS}
    labels = {s: _load_labels(paths[s]) for s in SPLITS}
    n_classes = int(max(int(labels[s].max()) for s in SPLITS)) + 1
    low, high = _counts_gate_window(readout["test"])
    print(
        f"[probe-logistic] device={device} n={ {s: int(labels[s].size) for s in SPLITS} } "
        f"readout TEST={readout['test']['correct']}/{readout['test']['total']}; "
        f"counts gate window {low}-{high}",
        flush=True,
    )

    probe_lr = float(getattr(args, "lr", LOGISTIC_LR))
    probe_epochs = int(getattr(args, "epochs", LOGISTIC_EPOCHS))
    results: dict[str, dict] = {}
    preds: dict[str, np.ndarray] = {}
    gate: dict = {}
    for feat in LOGISTIC_FEATURE_ORDER:
        sources = FEATURE_SETS[feat]
        entry, pred = _run_logistic_feature_set(
            paths, feat, sources, labels, n_classes, device, standardize=True,
            lr=probe_lr, epochs=probe_epochs,
        )
        results[feat] = entry
        preds[feat] = pred
        if feat == "counts":
            correct = int(entry["test_correct"])
            gate = {
                "readout_test_correct": int(readout["test"]["correct"]),
                "readout_test_total": int(readout["test"]["total"]),
                "readout_test_accuracy": float(readout["test"]["accuracy"]),
                "window_correct": [low, high],
                "window_pp": 1.0,
                "counts_test_correct": correct,
                "counts_test_total": int(entry["test_total"]),
                "counts_test_accuracy": float(entry["test_accuracy"]),
                "passed": bool(low <= correct <= high),
            }
            if not gate["passed"]:
                out = {
                    "capture_tag": tag,
                    "estimator": "multinomial logistic regression",
                    "protocol": LOGISTIC_PROTOCOL,
                    "n": {s: int(labels[s].size) for s in SPLITS},
                    "model_readout": readout,
                    "gate": gate,
                    "results": results,
                    "stopped": "counts gate failed; state probes not run",
                }
                out_path = results_path("probe_logistic", tag)
                with open(out_path, "w", encoding="utf-8") as fh:
                    json.dump(out, fh, indent=2)
                raise SystemExit(
                    f"GATE FAILED: counts probe {correct}/{entry['test_total']} "
                    f"outside {low}-{high} (readout "
                    f"{readout['test']['correct']}/{readout['test']['total']}); "
                    f"estimator bug -- state-vs-counts numbers not interpretable. "
                    f"See {out_path}"
                )

    # does per-feature standardisation matter for the 64k-dim state features?
    standardization = {}
    for feat in ("mean_state", "final_state"):
        raw, _ = _run_logistic_feature_set(
            paths, feat, FEATURE_SETS[feat], labels, n_classes, device,
            standardize=False, lr=probe_lr, epochs=probe_epochs,
        )
        standardization[feat] = {"standardized": results[feat], "raw": raw}
        print(
            f"[probe-logistic] standardisation {feat}: TEST "
            f"standardized={results[feat]['test_correct']}/{results[feat]['test_total']} "
            f"raw={raw['test_correct']}/{raw['test_total']}",
            flush=True,
        )

    boot = paired_bootstrap(
        preds["mean_state"], preds["counts"], labels["test"],
        n_boot=int(getattr(args, "bootstrap", 1000)), seed=LOGISTIC_SEED,
    )
    print(
        f"[probe-logistic] paired bootstrap (mean_state - counts): "
        f"delta={boot['delta'] * 100:+.2f} pp 95% CI "
        f"[{boot['ci95'][0] * 100:+.2f}, {boot['ci95'][1] * 100:+.2f}] "
        f"excludes_zero={boot['excludes_zero']}",
        flush=True,
    )

    out = {
        "capture_tag": tag,
        "estimator": (
            "multinomial logistic regression in torch; Adam lr=0.01, batch 256, "
            "30 epochs; L2 coefficient selected on VAL from "
            f"{list(LOGISTIC_LAM_GRID)}; fitted on FIT; TEST scored once"
        ),
        "protocol": LOGISTIC_PROTOCOL,
        "n": {s: int(labels[s].size) for s in SPLITS},
        "device": str(device),
        "model_readout": readout,
        "gate": gate,
        "feature_sets": {k: list(v) for k, v in FEATURE_SETS.items()},
        "results": results,
        "standardization": standardization,
        "bootstrap": {
            "contrast": "mean_state - counts",
            "paired": True,
            "n_test": int(labels["test"].size),
            "result": boot,
        },
    }
    out_path = results_path("probe_logistic", tag)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    pred_path = results_path("probe_logistic_predictions", tag).with_suffix(".npz")
    np.savez_compressed(
        pred_path,
        labels=labels["test"],
        **{f"pred_{k}": v for k, v in preds.items()},
    )
    print(f"[probe-logistic] -> {out_path}")
    print(f"[probe-logistic] -> {pred_path}")
    return out


LOGISTIC_PROTOCOL = (
    "Multinomial logistic regression (linear softmax) in torch. Features are "
    "standardised per dimension with FIT mean/std (fit only; TEST standardised "
    "with the same statistics). Adam lr=0.01, batch 256, 30 epochs, zero init, "
    "fixed seed 0. L2 strength lam*sum(W^2) selected on VAL (816 samples, never "
    "trained on by the model) from a decades-wide grid {1e-8..1}; ties go to the "
    "stronger penalty. "
    "Fitted on FIT (7340), TEST (2264) scored once. FIT-fitted probes are "
    "in-sample for the model (disclose); TEST is never used to fit or select."
)


def run_probe(args: argparse.Namespace) -> dict:
    """Probe entry point; ``--estimator`` switches legacy ridge / primary logistic."""
    if getattr(args, "estimator", "logistic") == "ridge":
        return run_probe_ridge(args)
    return run_probe_logistic(args)


# ---------------------------------------------------------------------- #
# geometry / effective dimensionality
# ---------------------------------------------------------------------- #
def pca_eigenvalues(x: torch.Tensor) -> torch.Tensor:
    """Descending eigenvalues of the covariance of ``x`` (n, d), float64.

    Uses the smaller Gram matrix and does not divide by ``n-1`` (all reported
    diagnostics are ratios, so the constant is irrelevant).
    """
    n, d = x.shape
    x = x.float()
    x.sub_(x.mean(0))
    gram = x @ x.t() if n <= d else x.t() @ x
    s = torch.linalg.eigvalsh(gram.double())
    return s.clamp_min_(0.0).flip(0)


def spectrum_summary(eigs: torch.Tensor) -> dict:
    """Participation ratio and component counts for a descending spectrum."""
    eigs = eigs.double().clamp_min(0.0)
    total = float(eigs.sum())
    if total <= 0.0:
        return {
            "participation_ratio": float("nan"),
            "n_components_90": 0,
            "n_components_95": 0,
            "n_components_99": 0,
            "top_1_variance_fraction": 0.0,
            "top_10_variance_fraction": 0.0,
        }
    var = eigs / total
    cum = var.cumsum(0)

    def n_for(threshold: float) -> int:
        return int((cum < threshold).sum().item()) + 1

    return {
        "participation_ratio": total**2 / float(eigs.square().sum()),
        "n_components_90": n_for(0.90),
        "n_components_95": n_for(0.95),
        "n_components_99": n_for(0.99),
        "top_1_variance_fraction": float(var[0]),
        "top_10_variance_fraction": float(var[:10].sum()),
    }


def _distribution(values: list[float]) -> dict:
    v = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "min": float(v.min()),
        "max": float(v.max()),
        "p10": float(np.percentile(v, 10)),
        "p90": float(np.percentile(v, 90)),
    }


def dimension_variance_summary(x: torch.Tensor) -> dict:
    """Participation ratio of the per-coordinate variance (diagonal only)."""
    var = x.float().var(0, unbiased=False)
    total = float(var.sum())
    if total <= 0.0:
        return {"participation_ratio": float("nan"), "n_dims": int(var.numel())}
    return {
        "participation_ratio": total**2 / float(var.square().sum()),
        "n_dims": int(var.numel()),
        "zero_variance_dims": int((var <= 0.0).sum()),
    }


def run_geometry(args: argparse.Namespace) -> dict:
    tag = args.tag or ""
    split = args.split
    path = capture_path(tag, split)
    if not path.exists():
        raise SystemExit(f"missing capture file: {path}; run `capture` first")

    with np.load(path) as z:
        mean_state = np.asarray(z["mean_state"])
        labels = np.asarray(z["labels"])
    n = int(mean_state.shape[0])
    n_neurons = int(mean_state.shape[1])
    state_dim = int(mean_state.shape[2])
    print(f"[geometry] {path.name}: n={n} N={n_neurons} D={state_dim}")

    flat = torch.from_numpy(mean_state.reshape(n, -1).copy()).float()
    overall = spectrum_summary(pca_eigenvalues(flat))
    overall["n_dims"] = state_dim * n_neurons
    overall["dimension_variance"] = dimension_variance_summary(flat)

    # per-neuron subspace across samples
    per_neuron_pr, per_neuron_90, per_neuron_95, per_neuron_99 = [], [], [], []
    per_neuron_dimvar_pr = []
    for i in range(n_neurons):
        xi = torch.from_numpy(mean_state[:, i, :].copy()).float()
        sm = spectrum_summary(pca_eigenvalues(xi))
        per_neuron_pr.append(sm["participation_ratio"])
        per_neuron_90.append(sm["n_components_90"])
        per_neuron_95.append(sm["n_components_95"])
        per_neuron_99.append(sm["n_components_99"])
        per_neuron_dimvar_pr.append(
            dimension_variance_summary(xi)["participation_ratio"]
        )

    per_neuron = {
        "n_neurons": n_neurons,
        "state_dim": state_dim,
        "participation_ratio": _distribution(per_neuron_pr),
        "dimension_variance_participation_ratio": _distribution(per_neuron_dimvar_pr),
        "n_components_90": _distribution(per_neuron_90),
        "n_components_95": _distribution(per_neuron_95),
        "n_components_99": _distribution(per_neuron_99),
    }
    if state_dim == 1:
        per_neuron["note"] = (
            "D=1: each neuron's state is one-dimensional by construction, so its "
            "per-neuron participation ratio is trivially 1; only the population-level "
            "geometry (64 dims across neurons) is informative"
        )
    out = {
        "capture_tag": tag,
        "capture_file": str(path),
        "split": split,
        "n_samples": n,
        "n_neurons": n_neurons,
        "state_dim": state_dim,
        "mean_state_geometry": overall,
        "per_neuron_geometry": per_neuron,
        "method": (
            "PCA of the captured mean state; participation ratio "
            "(sum(lambda)^2 / sum(lambda^2)) counts the effective number of "
            "dimensions (1 = rank-1, d = isotropic in d dims)"
        ),
    }
    out_path = results_path("geometry", tag)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"[geometry] overall PR={overall['participation_ratio']:.2f} "
          f"(n90/95/99={overall['n_components_90']}/{overall['n_components_95']}/"
          f"{overall['n_components_99']}); per-neuron PR median="
          f"{per_neuron['participation_ratio']['median']:.2f}")
    print(f"[geometry] -> {out_path}")
    return out


# ---------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threads", type=int, default=8,
                   help="torch CPU threads (default 8: under the 60%% CPU cap)")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("capture", help="run the frozen checkpoint and save state features")
    c.add_argument("--checkpoint", default=str(VECTOR_CHECKPOINT))
    c.add_argument("--tag", default="", help="prefix for output files (e.g. scalar_d1)")
    c.add_argument("--splits", default="fit,val,test")
    c.add_argument("--device", default="auto")
    c.add_argument("--batch-size", type=int, default=0, help="0 = checkpoint config")
    c.set_defaults(func=run_capture)

    pr = sub.add_parser(
        "probe", help="linear probes: primary logistic regression (ridge = legacy)"
    )
    pr.add_argument("--tag", default="")
    pr.add_argument(
        "--estimator", choices=("logistic", "ridge"), default="logistic",
        help="logistic = validated protocol; ridge = legacy estimator-limited table",
    )
    pr.add_argument("--device", default="auto", help="cpu / cuda / auto")
    pr.add_argument("--bootstrap", type=int, default=1000,
                    help="paired bootstrap resamples for the mean-state vs counts delta")
    pr.add_argument("--folds", type=int, default=5, help="CV folds inside the fit split (ridge)")
    pr.add_argument(
        "--epochs", type=int, default=LOGISTIC_EPOCHS,
        help=f"probe training epochs (protocol default {LOGISTIC_EPOCHS}); raise it "
        "only when the counts gate fails, and report the deviation",
    )
    pr.add_argument(
        "--lr", type=float, default=LOGISTIC_LR,
        help=f"probe learning rate (protocol default {LOGISTIC_LR})",
    )
    pr.set_defaults(func=run_probe)

    g = sub.add_parser("geometry", help="PCA / effective dimensionality")
    g.add_argument("--tag", default="")
    g.add_argument("--split", default="test", choices=list(SPLITS))
    g.set_defaults(func=run_geometry)
    return p


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(int(args.threads))
    return args.func(args)


if __name__ == "__main__":
    main()
