"""Render the publication figures from a saved analysis bundle.

Stage 8 (final, and cheap). This script never runs the model or touches data: it
reads the JSON bundle written by ``scripts/run_geometry_analysis.py`` (plus the
optional training history) and draws Figures 1-6 via
:func:`src.visualization.make_all_figures`.

=============================================  =====================================
Figure                                          Question it answers
=============================================  =====================================
``figure1_training_curves``                     did the network actually learn?
``figure2_representation_pca``                  where do hidden neurons live?
``figure3_distance_correlation``                geometry vs function (PRIMARY)
``figure4_knn_effect``                          are neighbours functionally similar?
``figure5_ablation_summary``                    which representation content matters?
``figure6_fingerprint_heatmap``                 what does the fingerprint look like?
=============================================  =====================================

Example
-------
    uv run python scripts/make_figures.py --config configs/analysis.yaml
"""

from __future__ import annotations

from _common import (  # noqa: E402
    figure_dir,
    load_run_config,
    parse_args,
    result_path,
)
from src.utils import load_json  # noqa: E402
from src.visualization import make_all_figures  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = parse_args("Render the project figures from a saved analysis bundle.", "configs/analysis.yaml")
    cfg = load_run_config(args)
    tag = str(cfg.get_path("run.tag", "baseline"))

    formats = [str(f) for f in cfg.get_path("figures.formats", ["png", "pdf"])]
    dpi = int(cfg.get_path("figures.dpi", 200))

    bundle_path = result_path(cfg, f"{tag}_figure_bundle.json")
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Figure bundle not found: {bundle_path}. Run "
            f"`uv run python scripts/run_geometry_analysis.py --config configs/analysis.yaml` first."
        )
    bundle = load_json(bundle_path)

    out = figure_dir(cfg)
    written = make_all_figures(bundle, out, formats=formats, dpi=dpi)

    if not written:
        print("[warn] no figures were produced (the bundle had no plottable sections).")
    else:
        for path in written:
            print(f"[figure] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
