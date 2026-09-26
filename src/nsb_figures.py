"""The nine figures for the neuron-space baseline experiment.

All figures are *descriptive*: PCA (Figure 2) and any projection are never used as
inferential evidence. The inferential content lives in the Mantel statistics,
the controls, the cross-validated prediction and their permutation nulls, which
are drawn as panels here only for readability.

Figures
-------
1. network and neuron-representation schematic
2. PCA of the (standardised) structural neuron representation
3. representation distance vs functional distance (with a binned trend)
4. raw vs rate-normalized functional relationship
5. representation/control comparison
6. kNN functional similarity
7. before vs after training
8. rewired-recurrent control
9. cross-validated prediction performance
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

DEFAULT_FORMATS = ("png", "pdf")
DEFAULT_DPI = 200

_PALETTE = {
    "structural": "#1f77b4",
    "rate_only": "#d62728",
    "random": "#7f7f7f",
    "neuron_shuffle": "#bcbd22",
}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _finish(fig, path_stem: Path, *, formats: Sequence[str], dpi: int) -> list[Path]:
    out: list[Path] = []
    for fmt in formats:
        p = path_stem.with_suffix(f".{fmt}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        out.append(p)
    return out


def _clean(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _fmt(v: Any) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(f) else f"{f:+.3f}"


# --------------------------------------------------------------------------
# Figure 1: schematic
# --------------------------------------------------------------------------
def figure1_schematic(bundle, out: Path, *, formats=DEFAULT_FORMATS, dpi=DEFAULT_DPI) -> list[Path]:
    plt = _plt()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # left: the network
    def box(x, y, w, h, label, color):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="k", lw=1.1, alpha=0.85))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9, wrap=True)

    box(0.04, 0.35, 0.14, 0.30, "input\n700 channels\n(2 ms bins)", "#cfe2f3")
    box(0.42, 0.22, 0.22, 0.56, "hidden\n256 recurrent\nLIF neurons", "#f6b26b")
    box(0.80, 0.40, 0.14, 0.20, "readout\n20 classes", "#93c47d")
    ax.annotate("", xy=(0.42, 0.5), xytext=(0.18, 0.5),
                arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#38761d"))
    ax.annotate("", xy=(0.80, 0.5), xytext=(0.64, 0.5),
                arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#38761d"))
    # recurrent loop
    ax.annotate("", xy=(0.53, 0.80), xytext=(0.53, 0.80),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#cc0000",
                                connectionstyle="arc3,rad=1.6"))
    ax.text(0.53, 0.88, "dense recurrence\nW_rec (256x256)", ha="center", fontsize=8, color="#cc0000")
    ax.text(0.5, 0.05, "trained checkpoint: sweep_l2_0 (accumulated readout, ~5 Hz hidden)",
            ha="center", fontsize=8, style="italic")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(a) network", fontsize=11)

    # right: the separation
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.set_title("(b) label-free representation  vs  independent fingerprint", fontsize=11)
    ax2.add_patch(plt.Rectangle((0.02, 0.30), 0.40, 0.58, facecolor="#eaf3ff",
                                edgecolor="#1f77b4", lw=1.4))
    ax2.text(0.22, 0.83, "NEURON REPRESENTATION", ha="center", fontsize=9, color="#1f77b4", weight="bold")
    ax2.text(0.22, 0.77, "label-free", ha="center", fontsize=8, style="italic", color="#1f77b4")
    for i, blk in enumerate(["intrinsic", "input_conn", "recurrent_in", "recurrent_out"]):
        ax2.text(0.22, 0.69 - i * 0.08, f"• {blk}", ha="center", fontsize=9)
    ax2.text(0.22, 0.33, "no labels  •  no test set", ha="center", fontsize=8, color="#1f77b4")

    ax2.add_patch(plt.Rectangle((0.58, 0.30), 0.40, 0.58, facecolor="#fdeeee",
                                edgecolor="#cc0000", lw=1.4))
    ax2.text(0.78, 0.83, "FUNCTIONAL FINGERPRINT", ha="center", fontsize=9, color="#cc0000", weight="bold")
    ax2.text(0.78, 0.77, "uses labels (PROBE only)", ha="center", fontsize=8, style="italic", color="#cc0000")
    for i, line in enumerate(["20-d class-rate profile", "(secondary: latency,", "temporal PSTH)"]):
        ax2.text(0.78, 0.68 - i * 0.09, line, ha="center", fontsize=9)
    ax2.text(0.78, 0.33, "evaluation target only", ha="center", fontsize=8, color="#cc0000")

    ax2.plot([0.50, 0.50], [0.28, 0.90], color="k", lw=2, ls="--")
    ax2.text(0.50, 0.18, "Mantel / CV comparison\n(never fed back)", ha="center", fontsize=8)
    fig.tight_layout()
    return _finish(fig, out / "figure1_schematic", formats=formats, dpi=dpi)


# --------------------------------------------------------------------------
# Figure 2: PCA
# --------------------------------------------------------------------------
def figure2_pca(bundle, out: Path, *, formats=DEFAULT_FORMATS, dpi=DEFAULT_DPI) -> list[Path]:
    plt = _plt()
    X = np.asarray(bundle["structural_X"], dtype=np.float64)
    rates = np.asarray(bundle["neuron_rate_hz"], dtype=np.float64)
    from sklearn.decomposition import PCA

    n_comp = int(min(2, X.shape[0], X.shape[1]))
    pca = PCA(n_components=n_comp, random_state=0)
    scores = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    sc = ax.scatter(scores[:, 0], scores[:, 1] if n_comp > 1 else np.zeros_like(scores[:, 0]),
                    c=rates, cmap="viridis", s=32, edgecolor="k", lw=0.3)
    fig.colorbar(sc, ax=ax, label="mean firing rate (Hz)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
    if n_comp > 1:
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")
    ax.set_title("PCA of the label-free structural neuron representation\n(descriptive only - not evidence)", fontsize=10)
    _clean(ax)
    fig.tight_layout()
    return _finish(fig, out / "figure2_representation_pca", formats=formats, dpi=dpi)


# --------------------------------------------------------------------------
# Figure 3: distance vs distance
# --------------------------------------------------------------------------
def figure3_distance_scatter(bundle, out: Path, *, formats=DEFAULT_FORMATS, dpi=DEFAULT_DPI) -> list[Path]:
    plt = _plt()
    dx = np.asarray(bundle["structural_dx"], dtype=np.float64)
    dy = np.asarray(bundle["tuning_dy"], dtype=np.float64)
    structural = next((r for r in bundle["after_rows"]
                       if r.get("representation") == "structural"), {})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.0))
    hb = ax.hexbin(dx, dy, gridsize=40, bins="log", cmap="Blues", mincnt=1)
    fig.colorbar(hb, ax=ax, label="log10(pair count)")
    ax.set_xlabel("structural representation distance")
    ax.set_ylabel("functional (class-rate) distance")
    ax.set_title(f"(a) all {dx.size} neuron pairs\nMantel r = {_fmt(structural.get('raw_r'))}", fontsize=10)
    _clean(ax)

    # binned trend by quantiles of dx
    n_bins = 20
    edges = np.unique(np.quantile(dx, np.linspace(0, 1, n_bins + 1)))
    if edges.size >= 3:
        idx = np.clip(np.digitize(dx, edges[1:-1]), 0, edges.size - 2)
        centres, means, sems = [], [], []
        for b in range(edges.size - 1):
            sel = idx == b
            if sel.sum() < 2:
                continue
            centres.append(float(dx[sel].mean()))
            means.append(float(dy[sel].mean()))
            sems.append(float(dy[sel].std(ddof=1) / np.sqrt(sel.sum())))
        ax2.errorbar(centres, means, yerr=sems, fmt="o-", color="#1f77b4", ms=4, capsize=3)
        ax2.set_xlabel("structural representation distance (quantile bin)")
        ax2.set_ylabel("mean functional distance")
        ax2.set_title("(b) binned trend (mean ± SEM)", fontsize=10)
        _clean(ax2)
    fig.tight_layout()
    return _finish(fig, out / "figure3_distance_vs_function", formats=formats, dpi=dpi)


# --------------------------------------------------------------------------
# Figure 4: raw vs rate-normalized
# --------------------------------------------------------------------------
def figure4_raw_vs_ratenorm(bundle, out: Path, *, formats=DEFAULT_FORMATS, dpi=DEFAULT_DPI) -> list[Path]:
    plt = _plt()
    rows = [r for r in bundle["after_rows"] if not r.get("skipped")]
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    lim = 1.0
    ax.plot([-lim, lim], [-lim, lim], color="k", lw=0.8, ls="--", zorder=0)
    for r in rows:
        x, y = _num(r.get("raw_r")), _num(r.get("rate_normalized_r"))
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        name = r["representation"]
        color = _PALETTE.get(name, "#9467bd")
        marker = "s" if r.get("is_control") else "o"
        ax.scatter(x, y, color=color, marker=marker, s=70, edgecolor="k", lw=0.5, zorder=3)
        ax.annotate(name, (x, y), fontsize=7.5, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlim(-0.1, 0.7)
    ax.set_ylim(-0.1, 0.7)
    ax.set_xlabel("Mantel r  vs raw functional fingerprint")
    ax.set_ylabel("Mantel r  vs rate-normalized fingerprint")
    ax.set_title("Does the relationship survive rate normalization?\n(points below the diagonal lose evidence)", fontsize=10)
    _clean(ax)
    fig.tight_layout()
    return _finish(fig, out / "figure4_raw_vs_rate_normalized", formats=formats, dpi=dpi)


# --------------------------------------------------------------------------
# Figure 5: control comparison
# --------------------------------------------------------------------------
def figure5_control_comparison(bundle, out: Path, *, formats=DEFAULT_FORMATS, dpi=DEFAULT_DPI) -> list[Path]:
    plt = _plt()
    rows = [r for r in bundle["after_rows"] if not r.get("skipped")]
    names = [r["representation"] for r in rows]
    raw = np.array([_num(r.get("raw_r")) for r in rows], dtype=np.float64)
    norm = np.array([_num(r.get("rate_normalized_r")) for r in rows], dtype=np.float64)
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.6, 0.62 * len(names) + 2.2))
    ax.barh(y + 0.2, raw, height=0.38, color="#1f77b4", label="vs raw fingerprint")
    ax.barh(y - 0.2, norm, height=0.38, color="#ff7f0e", label="vs rate-normalized fingerprint")
    for i, n in enumerate(names):
        if n == "structural":
            ax.get_yticklabels()
            ax.axhspan(i - 0.5, i + 0.5, color="#1f77b4", alpha=0.08, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(["* " + n if n == "structural" else n for n in names])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Mantel Spearman r")
    ax.set_title("representation / control comparison (PROBE fingerprint)\n* = PRIMARY representation", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    _clean(ax)
    fig.tight_layout()
    return _finish(fig, out / "figure5_control_comparison", formats=formats, dpi=dpi)


# --------------------------------------------------------------------------
# Figure 6: kNN
# --------------------------------------------------------------------------
def figure6_knn(bundle, out: Path, *, formats=DEFAULT_FORMATS, dpi=DEFAULT_DPI) -> list[Path]:
    plt = _plt()
    knn = bundle.get("structural_knn", [])
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    if knn:
        ks = [r["k"] for r in knn]
        z = [_num(r.get("effect_size_z")) for r in knn]
        p = [_num(r.get("p_value")) for r in knn]
        ax.plot(ks, z, "o-", color="#1f77b4", label="structural")
        for x, zz, pp in zip(ks, z, p):
            ax.annotate(f"p={pp:.1e}" if np.isfinite(pp) else "p=n/a", (x, zz),
                        fontsize=7, xytext=(3, 5), textcoords="offset points")
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("k (representation-space nearest neighbours)")
        ax.set_ylabel("effect size z  (positive = neighbours more similar)")
        ax.set_title("kNN functional similarity of representation neighbours", fontsize=10)
        ax.set_xticks(ks)
        _clean(ax)
    else:
        ax.text(0.5, 0.5, "no kNN results", ha="center")
        ax.axis("off")
    fig.tight_layout()
    return _finish(fig, out / "figure6_knn_functional_similarity", formats=formats, dpi=dpi)


# --------------------------------------------------------------------------
# Figure 7: before vs after
# --------------------------------------------------------------------------
def figure7_before_after(bundle, out: Path, *, formats=DEFAULT_FORMATS, dpi=DEFAULT_DPI) -> list[Path]:
    plt = _plt()
    ba = [b for b in bundle.get("before_after", []) if b.get("variant") == "structural"]
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    if ba and (np.isfinite(_num(ba[0].get("raw_r_before"))) or np.isfinite(_num(ba[0].get("raw_r_after")))):
        b = ba[0]
        labels = ["raw", "rate-normalized", "rate-matched"]
        before = [_num(b.get("raw_r_before")), _num(b.get("rate_normalized_r_before")), _num(b.get("rate_matched_r_before"))]
        after = [_num(b.get("raw_r_after")), _num(b.get("rate_normalized_r_after")), _num(b.get("rate_matched_r_after"))]
        x = np.arange(len(labels))
        ax.bar(x - 0.2, before, 0.4, label="before learning (untrained)", color="#bbbbbb")
        ax.bar(x + 0.2, after, 0.4, label="after learning (trained)", color="#1f77b4")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("Mantel Spearman r")
        ax.set_title("Does learning strengthen organization in neuron-space?", fontsize=10)
        ax.legend(fontsize=8)
        _clean(ax)
    else:
        ax.text(0.5, 0.5, "before-learning structural representation unavailable\n(empty intrinsic block)", ha="center", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    return _finish(fig, out / "figure7_before_vs_after", formats=formats, dpi=dpi)


# --------------------------------------------------------------------------
# Figure 8: rewiring
# --------------------------------------------------------------------------
def figure8_rewiring(bundle, out: Path, *, formats=DEFAULT_FORMATS, dpi=DEFAULT_DPI) -> list[Path]:
    plt = _plt()
    rewired = bundle.get("rewired", {})
    struct_after = next((r for r in bundle["after_rows"] if r.get("representation") == "structural"), {})
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    modes = ["original"] + list(rewired.keys())
    raw, norm = [], []
    raw.append(_num(struct_after.get("raw_r")))
    norm.append(_num(struct_after.get("rate_normalized_r")))
    for mode in rewired:
        row = next((r for r in rewired[mode]["rows"] if r.get("representation") == "structural"), {})
        raw.append(_num(row.get("raw_r")))
        norm.append(_num(row.get("rate_normalized_r")))
    x = np.arange(len(modes))
    ax.bar(x - 0.2, raw, 0.4, label="raw fingerprint", color="#1f77b4")
    ax.bar(x + 0.2, norm, 0.4, label="rate-normalized fingerprint", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=15)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Mantel Spearman r (structural representation)")
    ax.set_title("Rewired-recurrent control (weight distribution preserved)", fontsize=10)
    ax.legend(fontsize=8)
    _clean(ax)
    fig.tight_layout()
    return _finish(fig, out / "figure8_rewiring_control", formats=formats, dpi=dpi)


# --------------------------------------------------------------------------
# Figure 9: CV prediction
# --------------------------------------------------------------------------
def figure9_cv(bundle, out: Path, *, formats=DEFAULT_FORMATS, dpi=DEFAULT_DPI) -> list[Path]:
    plt = _plt()
    cv = bundle.get("cv_representations", {})
    names = [r["representation"] for r in bundle["after_rows"] if not r.get("skipped")]
    names = [n for n in names if n in cv]
    r2 = [_num(cv.get(n, {}).get("cv_r2_mean")) for n in names]
    rr = [_num(cv.get(n, {}).get("cv_pearson_r_mean")) for n in names]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.4, 0.6 * len(names) + 2.2))
    ax.barh(y + 0.2, r2, height=0.38, color="#2ca02c", label="ridge CV R²")
    ax.barh(y - 0.2, rr, height=0.38, color="#9467bd", label="ridge CV correlation")
    ax.set_yticks(y)
    ax.set_yticklabels(["* " + n if n == "structural" else n for n in names])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("out-of-fold (across neurons, identical folds)")
    ax.set_title("Cross-validated prediction of the 20-d fingerprint", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    _clean(ax)
    fig.tight_layout()
    return _finish(fig, out / "figure9_cv_prediction", formats=formats, dpi=dpi)


def _num(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def render_all(
    out_dir: str | Path,
    *,
    formats: Sequence[str] = DEFAULT_FORMATS,
    dpi: int = DEFAULT_DPI,
) -> list[Path]:
    """Render all nine figures from ``out_dir/figure_bundle.{json,npz}``."""
    import json

    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "figure_bundle.json", "r", encoding="utf-8") as fh:
        bundle: dict[str, Any] = json.load(fh)
    with np.load(str(out_dir / "figure_bundle.npz"), allow_pickle=True) as data:
        for key in ("structural_X", "structural_dx", "tuning_dy", "tuning_rate_normalized_dy",
                    "neuron_rate_hz", "class_rate_matrix", "structural_feature_names"):
            bundle[key] = data[key]

    written: list[Path] = []
    for fn in (figure1_schematic, figure2_pca, figure3_distance_scatter, figure4_raw_vs_ratenorm,
               figure5_control_comparison, figure6_knn, figure7_before_after, figure8_rewiring,
               figure9_cv):
        try:
            written.extend(fn(bundle, fig_dir, formats=formats, dpi=dpi))
        except Exception as exc:  # pragma: no cover - a figure must never break the run
            print(f"[figure][warn] {fn.__name__} failed: {type(exc).__name__}: {exc}")
    return written