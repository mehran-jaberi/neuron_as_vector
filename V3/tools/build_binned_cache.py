"""Build the pre-binned dense input cache for SHD (one-off, performance only).

Writes ``V3/cache/binned/<key>/`` for the train and test files.  The cache is
git-ignored, is built *through* the on-the-fly binning path (so its values are
bit-identical by construction), and is reused by any run with
``binned_cache=True`` (``--override binned_cache=true``).

    .venv/Scripts/python.exe V3/tools/build_binned_cache.py
    .venv/Scripts/python.exe V3/tools/build_binned_cache.py --only train --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

V3_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = V3_ROOT.parent
for p in (str(V3_ROOT), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_experiment import DEFAULT_CONFIG, apply_overrides  # noqa: E402
from v3.binned_cache import BinnedCache  # noqa: E402
from v3.config import V3Config  # noqa: E402
from v3.data import SHDEventStore  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="build the V3 pre-binned SHD input cache")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--only", choices=["train", "test", "both"], default="both")
    ap.add_argument("--force", action="store_true", help="rebuild even if a cache exists")
    args = ap.parse_args(argv)

    base = apply_overrides(V3Config.from_yaml(args.config), args.override)
    # binned_cache=False: build from the authoritative on-the-fly path
    base = base.with_overrides(binned_cache=False)
    files = []
    if args.only in ("train", "both"):
        files.append(("train", base.path(base.train_h5)))
    if args.only in ("test", "both"):
        files.append(("test", base.path(base.test_h5)))

    for name, path in files:
        if not Path(path).exists():
            print(f"[binned-cache] {name}: {path} not found, skipped", flush=True)
            continue
        store = SHDEventStore(path, base)
        print(f"[binned-cache] {name}: {store.n_samples} samples from {path}", flush=True)
        cache = BinnedCache.build(store, force=args.force)
        print(f"[binned-cache] {name}: {cache.summary()}", flush=True)
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
