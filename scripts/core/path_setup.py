"""
core/path_setup.py v2.0 (Integrity Edition)
===========================================
提供統一的 sys.path 配置與專案目錄結構管理，確保執行期寫入路徑的完整性。
取代過去散落於各個腳本頂部中冗餘且重複的環境設定代碼。
"""

import sys
from pathlib import Path
from typing import Dict, Optional

_scripts_dir_cache: Optional[Path] = None

def _resolve_scripts_dir(caller_file: str) -> Path:
    """從呼叫者的路徑往上尋找 scripts 根目錄。"""
    current_path = Path(caller_file).resolve()
    for parent in [current_path] + list(current_path.parents):
        if parent.name == "scripts":
            return parent
        # 若在上層目錄找到 core 或 fetchers，代表該目錄即為 root
        if (parent / "core").is_dir() and (parent / "fetchers").is_dir():
            return parent
    # 預設回傳當前腳本所在的目錄
    return current_path.parent

def get_scripts_dir(caller_file: str = __file__) -> Path:
    """獲取專案的 scripts 根目錄，並具備快取機制。"""
    global _scripts_dir_cache
    if _scripts_dir_cache is None:
        _scripts_dir_cache = _resolve_scripts_dir(caller_file)
    return _scripts_dir_cache

def ensure_scripts_on_path(caller_file: str):
    """將專案根目錄及其子目錄加入 sys.path。"""
    scripts_dir = get_scripts_dir(caller_file)
    scripts_str = str(scripts_dir)
    
    if scripts_str not in sys.path:
        sys.path.insert(0, scripts_str)
        
    # 自動加入專案常用子目錄
    subdirs = ["fetchers", "pipeline", "training", "monitor", "models", "utils", "tests", "core"]
    for subdir in subdirs:
        subdir_path = str(scripts_dir / subdir)
        if subdir_path not in sys.path:
            sys.path.insert(1, subdir_path)

def get_outputs_dir(caller_file: str = __file__) -> Path:
    """獲取並確保 outputs 目錄存在。"""
    path = get_scripts_dir(caller_file) / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_models_dir(caller_file: str = __file__) -> Path:
    """獲取並確保 models 目錄存在。"""
    path = get_outputs_dir(caller_file) / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_archive_dir(caller_file: str = __file__) -> Path:
    """獲取並確保 archive 目錄存在。"""
    path = get_models_dir(caller_file) / "archive"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_logs_dir(caller_file: str = __file__) -> Path:
    """獲取並確保 logs 目錄存在。"""
    path = get_scripts_dir(caller_file) / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_checkpoints_dir(caller_file: str = __file__) -> Path:
    """獲取並確保 checkpoints 目錄存在。"""
    path = get_outputs_dir(caller_file) / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path

def ensure_dirs_exist(caller_file: str = __file__) -> Dict[str, Path]:
    """初始化並回傳所有必備的專案目錄字典。"""
    return {
        "scripts": get_scripts_dir(caller_file),
        "outputs": get_outputs_dir(caller_file),
        "models": get_models_dir(caller_file),
        "archive": get_archive_dir(caller_file),
        "logs": get_logs_dir(caller_file),
        "checkpoints": get_checkpoints_dir(caller_file)
    }

# 預設行為：當模組被其他腳本載入時，嘗試以自身路徑進行環境初始化
ensure_scripts_on_path(__file__)