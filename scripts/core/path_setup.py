"""
core/path_setup.py — 統一的 sys.path 設定工具 v2.0
================================================
取代散落於多個 fetcher / training 腳本頂部、重複 2~4 次的 sys.path 區塊。

v2.0 改進（與 db_utils v3.0 / model_metadata v2.0 一致的「完整性」精神）：
  ★ get_scripts_dir()   : 統一取得 scripts/ 根目錄。
  ★ get_outputs_dir()   : 統一取得 scripts/outputs（預設失敗清單 / metadata 寫入位置）。
  ★ get_models_dir()    : 統一取得 scripts/outputs/models。
  ★ get_archive_dir()   : 統一取得 scripts/outputs/models/archive。
  ★ get_logs_dir()      : 統一取得 scripts/logs（自動建立）。
  ★ ensure_dirs_exist() : 啟動時一次建立所有必要目錄。
  ★ 所有 helper 都包 mkdir(parents=True, exist_ok=True)，避免「目錄不存在導致
    失敗清單寫不出來」的隱性問題。

呼叫方式（在每支腳本最頂部）：
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import core.path_setup  # noqa: F401  # side-effect: ensure all sub-paths

或更精簡：
    from core.path_setup import ensure_scripts_on_path, get_outputs_dir
    ensure_scripts_on_path(__file__)
    OUTPUT_DIR = get_outputs_dir()
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_SUBDIRS = ("fetchers", "pipeline", "training", "monitor", "models", "utils", "tests", "core")

# 模組層級快取（避免重複計算）
_scripts_dir_cache: Path | None = None


# ─────────────────────────────────────────────
# scripts 根目錄推算
# ─────────────────────────────────────────────
def _resolve_scripts_dir(caller_file: str | Path | None = None) -> Path:
    """從 caller 路徑往上找出 scripts/ 根目錄。"""
    if caller_file is None:
        # core/path_setup.py 自己在 scripts/core/ 下，parent.parent = scripts/
        return Path(__file__).resolve().parent.parent

    path = Path(caller_file).resolve()
    cursor = path.parent
    while cursor.name != "scripts" and cursor.parent != cursor:
        cursor = cursor.parent
    if cursor.name != "scripts":
        # fallback：以 path_setup 自己的位置推算
        cursor = Path(__file__).resolve().parent.parent
    return cursor


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
    global _scripts_dir_cache

    scripts_dir = _resolve_scripts_dir(caller_file)
    _scripts_dir_cache = scripts_dir

    candidates = [scripts_dir] + [scripts_dir / s for s in _SUBDIRS]
    for p in candidates:
        s = str(p)
        if p.exists() and s not in sys.path:
            sys.path.insert(0, s)
    return scripts_dir


# ─────────────────────────────────────────────
# 標準資料夾位置（v2.0 新增）
# ─────────────────────────────────────────────
def get_scripts_dir() -> Path:
    """取得 scripts/ 根目錄（會 cache）。"""
    global _scripts_dir_cache
    if _scripts_dir_cache is None:
        _scripts_dir_cache = _resolve_scripts_dir()
    return _scripts_dir_cache


def get_outputs_dir(create: bool = True) -> Path:
    """
    取得 scripts/outputs 目錄。
    所有 fetcher 的失敗清單、integrity_gaps.json、backfill_failures.json 都應寫到這裡。
    """
    p = get_scripts_dir() / "outputs"
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def get_models_dir(create: bool = True) -> Path:
    """取得 scripts/outputs/models 目錄（current 模型存放處）。"""
    p = get_outputs_dir(create=create) / "models"
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def get_archive_dir(create: bool = True) -> Path:
    """取得 scripts/outputs/models/archive 目錄（歷史模型 + metadata 封存處）。"""
    p = get_models_dir(create=create) / "archive"
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def get_logs_dir(create: bool = True) -> Path:
    """取得 scripts/logs 目錄（fetcher / training log 寫入處）。"""
    p = get_scripts_dir() / "logs"
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def get_checkpoints_dir(create: bool = True) -> Path:
    """取得 scripts/outputs/checkpoints 目錄（fetcher 進度檔）。"""
    p = get_outputs_dir(create=create) / "checkpoints"
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_dirs_exist() -> dict[str, Path]:
    """
    啟動時一次建立所有必要目錄，並回傳 dict 方便檢查。
    建議在每支 fetcher / training 腳本啟動時呼叫一次，
    避免「跑到一半才發現 outputs 目錄不存在」。
    """
    dirs = {
        "scripts":     get_scripts_dir(),
        "outputs":     get_outputs_dir(),
        "models":      get_models_dir(),
        "archive":     get_archive_dir(),
        "logs":        get_logs_dir(),
        "checkpoints": get_checkpoints_dir(),
    }
    return dirs


# ─────────────────────────────────────────────
# 模組載入時自動執行：ensure paths
# ─────────────────────────────────────────────
ensure_scripts_on_path()


__all__ = [
    "ensure_scripts_on_path",
    "get_scripts_dir", "get_outputs_dir", "get_models_dir",
    "get_archive_dir", "get_logs_dir", "get_checkpoints_dir",
    "ensure_dirs_exist",
]