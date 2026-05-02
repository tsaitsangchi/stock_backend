"""
core/path_setup.py — 統一的 sys.path 設定工具
================================================
取代散落於多個 fetcher / training 腳本頂部、重複 2~4 次的 sys.path 區塊。

呼叫方式（在每支腳本最頂部）：
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)

或更精簡的單行寫法：
    import sys; from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import core.path_setup  # noqa: F401  # side-effect: ensure all sub-paths
"""

from __future__ import annotations

import sys
from pathlib import Path

_SUBDIRS = ("fetchers", "pipeline", "training", "monitor", "models", "utils", "tests")


def ensure_scripts_on_path(caller_file: str | Path | None = None) -> Path:
    """
    把 scripts/ 與其各子目錄加入 sys.path（冪等）。

    Parameters
    ----------
    caller_file : 通常傳 __file__；若為 None 則由本模組推算 scripts/ 根目錄

    Returns
    -------
    Path  scripts/ 根目錄
    """
    if caller_file is None:
        scripts_dir = Path(__file__).resolve().parent.parent
    else:
        # 任何位於 scripts/<subdir>/foo.py 的呼叫者都能正確推得根目錄
        path = Path(caller_file).resolve()
        scripts_dir = path.parent
        while scripts_dir.name != "scripts" and scripts_dir.parent != scripts_dir:
            scripts_dir = scripts_dir.parent
        if scripts_dir.name != "scripts":
            scripts_dir = Path(__file__).resolve().parent.parent

    candidates = [scripts_dir] + [scripts_dir / s for s in _SUBDIRS]
    for p in candidates:
        s = str(p)
        if p.exists() and s not in sys.path:
            sys.path.insert(0, s)
    return scripts_dir


# 模組被 import 時自動執行一次（提供「import core.path_setup」的副作用語意）
ensure_scripts_on_path()
