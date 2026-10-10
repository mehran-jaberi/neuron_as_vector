"""Make the isolated ``v3`` package importable for the V3 test module.

V3 lives outside the repository's ``tests/`` package on purpose (isolation), so it
carries its own conftest that puts ``V3/`` on ``sys.path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

V3_ROOT = Path(__file__).resolve().parents[1]
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))
