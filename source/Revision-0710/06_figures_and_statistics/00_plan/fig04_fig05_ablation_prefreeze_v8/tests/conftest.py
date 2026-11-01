from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]
for path in (ROOT, REPO_ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

