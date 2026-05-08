"""
core/model_metadata.py v3.0 (Full Lifecycle Management Edition)
===============================================================
提供量化模型的特徵指紋比對、原子寫入保障、歷史版本回滾支援，
以及動態環境套件依賴追蹤的完整生命週期管理。
"""

import os
import json
import shutil
import hashlib
import subprocess
import threading
import importlib
import glob
import platform
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    """量化模型詮釋資料結構。"""
    stock_id: str
    model_path: str
    timestamp: str
    git_hash: Optional[str]
    python_version: str
    feature_count: int
    feature_fingerprint: str
    oof_da: float
    oof_sharpe: float
    oof_ic: float
    oof_n_samples: int
    n_trades_per_fold: float
    max_drawdown: float
    train_end_date: str
    horizon_days: int
    calibration_method: str
    calibrator_cv: str
    notes: str
    package_versions: Dict[str, str]

    def to_dict(self) -> Dict:
        return asdict(self)

# ─────────────────────────────────────────────
# 路徑鎖定機制 (Path-level Mutex)
# ─────────────────────────────────────────────
_locks_dict_lock = threading.Lock()
_path_locks: Dict[str, threading.Lock] = {}

class _path_lock:
    """提供針對單一路徑的執行緒安全互斥鎖，防範平行訓練競爭。"""
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
    def __enter__(self):
        with _locks_dict_lock:
            if self.path not in _path_locks:
                _path_locks[self.path] = threading.Lock()
            self.lock = _path_locks[self.path]
        self.lock.acquire()
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.release()

# ─────────────────────────────────────────────
# 核心工具函式
# ─────────────────────────────────────────────
def get_git_hash(short: bool = True) -> Optional[str]:
    """獲取當前程式碼庫的 Git Commit Hash 以進行版控追蹤。"""
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception as e:
        logger.warning(f"無法獲取 Git Hash: {e}")
        return None

def fingerprint_features(features: List[str]) -> str:
    """生成經過 SHA-256 雜湊的特徵指紋，防範推論期資料漂移。"""
    sorted_features = sorted(features)
    feature_str = ",".join(sorted_features)
    return hashlib.sha256(feature_str.encode("utf-8")).hexdigest()[:12]

def get_package_versions(packages: Optional[List[str]] = None) -> Dict[str, str]:
    """動態擷取訓練環境下的關鍵套件版本。"""
    if packages is None:
        packages = ["numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "joblib", "torch", "polars"]
    
    versions = {}
    for pkg in packages:
        try:
            module_name = "sklearn" if pkg == "scikit-learn" else pkg
            mod = importlib.import_module(module_name)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not installed"
    return versions

# ─────────────────────────────────────────────
# 原子寫入與檔案操作 (Atomic I/O)
# ─────────────────────────────────────────────
def atomic_write_json(path: str, data: dict):
    """利用暫存檔與 os.replace 實現防止斷電損毀的原子寫入機制。"""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    tmp_path = path + ".tmp"
    with _path_lock(path):
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False, default=str)
        os.replace(tmp_path, path)

def atomic_copy_file(src: str, dst: str):
    """原子檔案複製，防止複製過程中的多進程競爭。"""
    dst_obj = Path(dst)
    dst_obj.parent.mkdir(parents=True, exist_ok=True)
    
    tmp_dst = dst + ".tmp"
    with _path_lock(dst):
        shutil.copy2(src, tmp_dst)
        os.replace(tmp_dst, dst)

# ─────────────────────────────────────────────
# 詮釋資料持久化與生命週期管理
# ─────────────────────────────────────────────
def save_metadata(metadata: ModelMetadata, archive_dir: str, current_pkl_path: str, also_archive_pkl: bool = True):
    """保存詮釋資料，並可選擇性建立實體模型封存檔。"""
    os.makedirs(archive_dir, exist_ok=True)
    filename_base = f"ensemble_{metadata.stock_id}_{metadata.timestamp}_{metadata.git_hash or 'nogit'}"
    json_path = os.path.join(archive_dir, f"{filename_base}.metadata.json")
    
    atomic_write_json(json_path, metadata.to_dict())
    
    if also_archive_pkl and os.path.exists(current_pkl_path):
        archive_pkl_path = os.path.join(archive_dir, f"{filename_base}.pkl")
        atomic_copy_file(current_pkl_path, archive_pkl_path)
        logger.info(f"模型與詮釋資料已封存至 {archive_dir}")

def list_history(stock_id: str, archive_dir: str, limit: Optional[int] = None) -> List[ModelMetadata]:
    """依時間逆序回傳特定股票的歷史模型版本清單。"""
    if not os.path.exists(archive_dir):
        return []
        
    pattern = os.path.join(archive_dir, f"ensemble_{stock_id}_*.metadata.json")
    files = glob.glob(pattern)
    
    # 根據檔案修改時間 (mtime) 進行反向排序 (最新的在最前)
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    history = []
    for f in files:
        if limit and len(history) >= limit:
            break
        try:
            with open(f, "r", encoding="utf-8") as file:
                data = json.load(file)
                history.append(ModelMetadata(**data))
        except Exception as e:
            logger.error(f"無法載入詮釋資料 {f}: {e}")
            
    return history

def load_latest_metadata(stock_id: str, archive_dir: str) -> Optional[ModelMetadata]:
    """尋找並載入特定股票最新版本的詮釋資料。"""
    history = list_history(stock_id, archive_dir, limit=1)
    return history[0] if history else None

def rollback_to_metadata(metadata: ModelMetadata, archive_dir: str, current_pkl_path: str) -> bool:
    """將指定的歷史封存模型檔還原至當前的作用路徑。"""
    filename_base = f"ensemble_{metadata.stock_id}_{metadata.timestamp}_{metadata.git_hash or 'nogit'}"
    archive_pkl_path = os.path.join(archive_dir, f"{filename_base}.pkl")
    
    if not os.path.exists(archive_pkl_path):
        logger.error(f"無法回滾：在 {archive_pkl_path} 找不到封存模型檔案")
        return False
        
    try:
        atomic_copy_file(archive_pkl_path, current_pkl_path)
        logger.info(f"成功將 {metadata.stock_id} 回滾至版本 {metadata.timestamp}")
        return True
    except Exception as e:
        logger.error(f"回滾失敗 {metadata.stock_id}: {e}")
        return False

def assert_feature_schema_match(metadata: ModelMetadata, current_features: List[str], strict: bool = True, allow_extra: bool = False):
    """於推論階段驗證特徵集合，確保模型部署之相容性。"""
    current_fingerprint = fingerprint_features(current_features)
    if current_fingerprint != metadata.feature_fingerprint:
        msg = f"特徵綱要不匹配！預期 {metadata.feature_fingerprint}, 實際 {current_fingerprint}。"
        if strict and not allow_extra:
            raise RuntimeError(msg)
        elif allow_extra:
            logger.warning(f"{msg} 由於 allow_extra=True，允許額外特徵。")
        else:
            logger.warning(msg)