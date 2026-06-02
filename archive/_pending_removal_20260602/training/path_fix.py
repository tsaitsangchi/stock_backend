"""
path_fix.py — 向後相容 shim
============================
此檔案歷史上被部分腳本以 `import path_fix` 方式引用，作用是把
scripts/ 各子目錄加入 sys.path。

新版改由統一的 core.path_setup 提供同樣的副作用，本 shim 僅做轉發
以保持向後相容；新檔案請直接使用：

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import core.path_setup  # noqa: F401
"""

import sys
from pathlib import Path

# 確保 scripts/ 已在 sys.path 後再 import core.path_setup
_scripts_dir = Path(__file__).resolve().parent.parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

try:
    import core.path_setup  # noqa: F401  # side-effect: ensure all sub-paths
except Exception:
    # core.path_setup 不存在時退回原本行為（極端 bootstrap 場景）
    for _sub in ["fetchers", "pipeline", "training", "monitor", "models", "utils"]:
        _p = str(_scripts_dir / _sub)
        if _p not in sys.path:
            sys.path.append(_p)
