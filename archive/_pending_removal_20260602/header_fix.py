"""
header_fix.py — 向後相容 shim（與 path_fix.py 等價）
====================================================
歷史上同時存在 header_fix.py 與 path_fix.py 兩個近乎相同的 sys.path 補丁。
v2 將兩者統一轉發至 core.path_setup，避免邏輯漂移。
"""

from __future__ import annotations
import sys
from pathlib import Path

_scripts_dir = Path(__file__).resolve().parent.parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

try:
    import core.path_setup  # noqa: F401
except Exception:
    for _sub in ["fetchers", "pipeline", "training", "monitor", "models", "utils"]:
        _p = str(_scripts_dir / _sub)
        if _p not in sys.path:
            sys.path.append(_p)
