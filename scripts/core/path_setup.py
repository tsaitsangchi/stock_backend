"""
path_setup.py v3.0 (MLOps Modular Edition)
量化系統核心：專案路徑與環境變數管理
================================================================================
v3.0 重大升級：
  · 支援標準化套件安裝：為未來遷移至 `pip install -e .` (pyproject.toml) 做準備，
    減少對 sys.path 強行注入的依賴。
  · 環境變數優先：支援以環境變數 TRINITY_ROOT 強制指定專案根目錄，完美適配 Docker/Airflow 生產環境。
  · 智慧載入 (Smart Path Injection)：自動偵測模組是否已可用，若尚未安裝才啟動 sys.path fallback。

執行範例與最佳實踐：
    # 在任何腳本的開頭，只需要呼叫 ensure_dirs_exist 即可確保輸出目錄健全
    from core.path_setup import ensure_dirs_exist
    dirs = ensure_dirs_exist()
    print(f"訓練好的模型將存檔於: {dirs['models']}")
    
    # 生產環境下 (如 Linux Server / Docker)，可透過環境變數直接鎖定根目錄
    # export TRINITY_ROOT="/opt/stock_backend/scripts"
"""

import os
import sys
import importlib.util
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

_project_root_cache: Optional[Path] = None

def get_project_root(caller_file: str = __file__) -> Path:
    """
    獲取專案的根目錄 (原 scripts 目錄)，並具備快取與環境變數優先機制。
    優先順序: 1. TRINITY_ROOT 環境變數 -> 2. 向上尋找特徵目錄 -> 3. 當前目錄
    """
    global _project_root_cache
    if _project_root_cache is not None:
        return _project_root_cache

    # 1. 檢查環境變數 (最高優先權，適合生產環境與 Docker)
    env_root = os.environ.get("TRINITY_ROOT")
    if env_root and Path(env_root).exists():
        _project_root_cache = Path(env_root).resolve()
        return _project_root_cache

    # 2. 從呼叫者的路徑往上尋找根目錄
    current_path = Path(caller_file).resolve()
    for parent in [current_path] + list(current_path.parents):
        if parent.name == "scripts":
            _project_root_cache = parent
            return _project_root_cache
        # 若在上層目錄找到 core 或 fetchers，代表該目錄即為 root
        if (parent / "core").is_dir() and (parent / "fetchers").is_dir():
            _project_root_cache = parent
            return _project_root_cache
            
    # 3. 預設回傳當前腳本所在的目錄
    _project_root_cache = current_path.parent
    return _project_root_cache

def ensure_scripts_on_path(caller_file: str = __file__):
    """
    智慧型環境變數注入：
    若專案已經透過 package 形式安裝，則不干擾 sys.path。
    否則，將專案根目錄及其核心子目錄加入 sys.path 以向下相容舊版開發模式。
    """
    root_dir = get_project_root(caller_file)
    root_str = str(root_dir)
    
    # 智慧偵測：檢查是否已經能順利 import core
    is_core_installed = importlib.util.find_spec("core") is not None
    
    if is_core_installed and root_str not in sys.path:
        # 系統已經作為標準套件安裝，無需 Hack，直接返回
        return

    # Fallback：舊版 sys.path 強行注入模式
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
        
    # 自動加入專案常用子目錄
    subdirs = ["fetchers", "pipeline", "training", "monitor", "models", "utils", "tests", "core"]
    for subdir in subdirs:
        subdir_path = str(root_dir / subdir)
        if subdir_path not in sys.path:
            sys.path.insert(1, subdir_path)

def get_outputs_dir(caller_file: str = __file__) -> Path:
    """獲取並確保 outputs 目錄存在。"""
    path = get_project_root(caller_file) / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_models_dir(caller_file: str = __file__) -> Path:
    """獲取並確保 models 目錄存在。"""
    path = get_outputs_dir(caller_file) / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_archive_dir(caller_file: str = __file__) -> Path:
    """獲取並確保 archive 目錄存在 (用於歷史模型封存)。"""
    path = get_models_dir(caller_file) / "archive"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_logs_dir(caller_file: str = __file__) -> Path:
    """獲取並確保 logs 目錄存在。"""
    path = get_project_root(caller_file) / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_checkpoints_dir(caller_file: str = __file__) -> Path:
    """獲取並確保 checkpoints 目錄存在 (用於斷點續傳)。"""
    path = get_outputs_dir(caller_file) / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path

def ensure_dirs_exist(caller_file: str = __file__) -> Dict[str, Path]:
    """初始化並回傳所有必備的專案資料目錄字典。"""
    return {
        "root": get_project_root(caller_file),
        "outputs": get_outputs_dir(caller_file),
        "models": get_models_dir(caller_file),
        "archive": get_archive_dir(caller_file),
        "logs": get_logs_dir(caller_file),
        "checkpoints": get_checkpoints_dir(caller_file)
    }

# 預設行為：當模組被其他腳本載入時，嘗試以自身路徑進行環境初始化
ensure_scripts_on_path(__file__)