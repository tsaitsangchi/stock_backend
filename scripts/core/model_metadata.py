"""
model_metadata.py v4.0 (MLOps Registry Edition)
================================================================================
量化系統核心：模型生命週期管理與註冊表
此模組負責管理機器學習模型的元資料 (Metadata)，確保模型版本、特徵指紋、
以及回測指標 (Sharpe, DA) 的一致性與原子性存儲。

核心功能：
  · Pydantic 校驗    ─ 嚴格限制模型指標範圍，拒絕異常模型落地。
  · Model Registry   ─ 將模型元資料同步寫入 PostgreSQL，支援 SQL 追蹤與比對。
  · 特徵指紋 (Fingerprint) ─ 紀錄訓練時的特徵清單，防止預測時發生特徵漂移 (Skew)。
  · 原子性寫入        ─ 採用 .tmp 中轉機制，確保 JSON 與模型檔在存檔過程不毀損。

修訂歷程：
  v4.0 (2026-04-20):
    - [重大] 引入 Pydantic (Data Contracts)，強制進行評估指標校驗。
    - [重大] 實作 Model Registry，支援與 PostgreSQL 同步。
    - [修正] 修復 Path 拼接錯誤導致的原子寫入崩潰。
  v3.0 (2026-03-15):
    - [基礎] 實作特徵指紋功能。

執行範例：
    # 範例 1：建立並儲存模型元資料 (含自動 Git Hash 綁定)
    from core.model_metadata import ModelMetadata, save_metadata
    meta = ModelMetadata(
        stock_id="2330", model_path="ensemble.pkl", feature_count=150,
        feature_fingerprint="abc123hash", oof_da=0.55, oof_sharpe=1.8,
        oof_ic=0.08, oof_n_samples=1200, n_trades_per_fold=45.5,
        max_drawdown=-0.15, train_end_date="2024-01-01", horizon_days=5,
        calibration_method="isotonic", calibrator_cv="prefit"
    )
    save_metadata(meta, archive_dir="outputs/models/archive")
"""

import os
import json
import shutil
import hashlib
import subprocess
import threading
import glob
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# 導入 Pydantic 進行嚴格資料契約校驗
from pydantic import BaseModel, Field, field_validator, ValidationError

# 嘗試對接內部資料庫模組，若無則降級為純檔案模式
try:
    from core.db_utils import db_transaction
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False

logger = logging.getLogger(__name__)

# =====================================================================
# 資料庫 DDL (Model Registry)
# =====================================================================
DDL_MODEL_REGISTRY = """
CREATE TABLE IF NOT EXISTS model_registry (
    stock_id VARCHAR(20) NOT NULL,
    timestamp VARCHAR(50) NOT NULL,
    model_path TEXT,
    git_hash VARCHAR(50),
    python_version VARCHAR(20),
    feature_count INTEGER,
    feature_fingerprint VARCHAR(64),
    oof_da NUMERIC(10, 4),
    oof_sharpe NUMERIC(10, 4),
    oof_ic NUMERIC(10, 4),
    max_drawdown NUMERIC(10, 4),
    train_end_date DATE,
    horizon_days INTEGER,
    notes TEXT,
    package_versions JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (stock_id, timestamp)
);
"""

# =====================================================================
# 領域實體與資料契約 (Data Contracts)
# =====================================================================
def _get_git_hash() -> str:
    """自動獲取當前 Git Hash 以綁定模型版本"""
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.STDOUT)
        return h.decode("utf-8").strip()
    except Exception:
        return "nogit"

class ModelMetadata(BaseModel):
    """
    量化模型詮釋資料結構 (具備嚴格校驗)
    任何不符合量化物理意義的數據都會在此 be 攔截。
    """
    stock_id: str = Field(..., min_length=1)
    model_path: str
    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    git_hash: str = Field(default_factory=_get_git_hash)
    python_version: str = Field(default_factory=platform.python_version)
    
    # 特徵資訊
    feature_count: int = Field(..., ge=1, description="特徵數量必須大於 0")
    feature_fingerprint: str
    
    # 回測與表現指標 (OOF: Out-Of-Fold)
    oof_da: float = Field(..., ge=0.0, le=1.0, description="方向準確率 (DA) 必須介於 0 與 1 之間")
    oof_sharpe: float
    oof_ic: float
    oof_n_samples: int = Field(..., ge=0)
    n_trades_per_fold: float
    max_drawdown: float = Field(..., le=0.0, description="最大回撤必須為負數或零")
    
    # 訓練邊界
    train_end_date: str
    horizon_days: int = Field(..., ge=1, description="預測窗期必須大於等於 1 天")
    
    # 校準器與其他設定
    calibration_method: str
    calibrator_cv: str
    notes: str = ""
    package_versions: Dict[str, str] = Field(default_factory=dict)

# =====================================================================
# 路徑鎖定與原子操作機制
# =====================================================================
_locks_dict_lock = threading.Lock()
_path_locks: Dict[str, threading.Lock] = {}

def _get_lock_for_path(path_str: str) -> threading.Lock:
    with _locks_dict_lock:
        if path_str not in _path_locks:
            _path_locks[path_str] = threading.Lock()
        return _path_locks[path_str]

def atomic_write_json(path: Path | str, data: BaseModel | dict) -> None:
    """原子化寫入 JSON，修正了 Path 拼接問題，並支援 Pydantic BaseModel"""
    path_obj = Path(path)
    # 若傳入的是 Pydantic 模型，自動轉為 dict
    dump_data = data.model_dump() if isinstance(data, BaseModel) else data
    
    lock = _get_lock_for_path(str(path_obj.resolve()))
    with lock:
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        # 【核心修復】使用 .with_name 避免 Path 物件不支援 + 運算子的錯誤
        temp_path = path_obj.with_name(f"{path_obj.name}.tmp")
        
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(dump_data, f, indent=2, ensure_ascii=False, default=str)
        
        # os.replace 確保在 POSIX 系統中是原子性的覆蓋
        os.replace(str(temp_path), str(path_obj))

def atomic_copy_file(src: str, dst: str) -> None:
    """原子化複製檔案 (如 .pkl 模型檔)"""
    src_obj, dst_obj = Path(src), Path(dst)
    lock = _get_lock_for_path(str(dst_obj.resolve()))
    with lock:
        dst_obj.parent.mkdir(parents=True, exist_ok=True)
        temp_dst = dst_obj.with_name(f"{dst_obj.name}.tmp")
        shutil.copy2(src_obj, temp_dst)
        os.replace(str(temp_dst), str(dst_obj))

# =====================================================================
# 特徵指紋 (Feature Fingerprint)
# =====================================================================
def fingerprint_features(feature_names: List[str]) -> str:
    """
    為特徵列表產生加密指紋。
    用於預測階段比對「當前組建的特徵」與「模型訓練時的特徵」是否完美吻合。
    """
    sorted_features = sorted(list(feature_names))
    encoded = json.dumps(sorted_features).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()

def assert_feature_schema_match(metadata: ModelMetadata, current_features: List[str]):
    """於推論階段驗證特徵集合，確保模型部署之相容性。"""
    current_fingerprint = fingerprint_features(current_features)
    if current_fingerprint != metadata.feature_fingerprint:
        logger.error(
            f"❌ 特徵指紋不符！模型 {metadata.stock_id} 拒絕預測。\n"
            f"預期: {metadata.feature_fingerprint}\n"
            f"實際: {current_fingerprint}"
        )
        raise ValueError(f"Feature Schema Mismatch for stock {metadata.stock_id}")

# =====================================================================
# 生命週期操作：存檔、讀取與資料庫註冊
# =====================================================================

def register_model_in_db(metadata: ModelMetadata):
    """將模型詮釋資料同步註冊至 PostgreSQL"""
    if not _DB_AVAILABLE:
        return
    
    sql = """
        INSERT INTO model_registry (
            stock_id, timestamp, model_path, git_hash, python_version,
            feature_count, feature_fingerprint, oof_da, oof_sharpe, oof_ic,
            max_drawdown, train_end_date, horizon_days, notes, package_versions
        ) VALUES (
            %(stock_id)s, %(timestamp)s, %(model_path)s, %(git_hash)s, %(python_version)s,
            %(feature_count)s, %(feature_fingerprint)s, %(oof_da)s, %(oof_sharpe)s, %(oof_ic)s,
            %(max_drawdown)s, %(train_end_date)s, %(horizon_days)s, %(notes)s, %(package_versions)s
        )
        ON CONFLICT (stock_id, timestamp) DO NOTHING;
    """
    
    data_dict = metadata.model_dump()
    data_dict["package_versions"] = json.dumps(data_dict["package_versions"])
    
    try:
        with db_transaction() as cur:
            cur.execute(DDL_MODEL_REGISTRY)
            cur.execute(sql, data_dict)
        logger.debug(f"已將模型 {metadata.stock_id} (TS:{metadata.timestamp}) 註冊至資料庫。")
    except Exception as e:
        logger.error(f"寫入 Model Registry 失敗 (不影響本地存檔): {e}")

def save_metadata(metadata: ModelMetadata, archive_dir: str):
    """儲存詮釋資料：同時寫入本地端與資料庫註冊表"""
    archive_path = Path(archive_dir)
    archive_path.mkdir(parents=True, exist_ok=True)
    
    # 1. 寫入本地 JSON
    filename_base = f"ensemble_{metadata.stock_id}_{metadata.timestamp}_{metadata.git_hash}"
    json_path = archive_path / f"{filename_base}.json"
    atomic_write_json(json_path, metadata)
    
    # 2. 註冊至資料庫 (Model Registry)
    register_model_in_db(metadata)

def list_history(stock_id: str, archive_dir: str, limit: int = 5) -> List[ModelMetadata]:
    """獲取特定股票的模型歷史紀錄 (依時間倒序)"""
    search_pattern = os.path.join(archive_dir, f"ensemble_{stock_id}_*.json")
    files = glob.glob(search_pattern)
    
    history = []
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
                history.append(ModelMetadata(**data))
        except Exception as e:
            logger.warning(f"讀取封存檔 {f} 失敗: {e}")
            
    history.sort(key=lambda x: x.timestamp, reverse=True)
    return history[:limit]

def rollback_to_metadata(metadata: ModelMetadata, archive_dir: str, current_pkl_path: str) -> bool:
    """將指定的歷史封存模型檔還原至當前的作用路徑 (Active Model Path)"""
    filename_base = f"ensemble_{metadata.stock_id}_{metadata.timestamp}_{metadata.git_hash}"
    archive_pkl_path = Path(archive_dir) / f"{filename_base}.pkl"
    
    if not archive_pkl_path.exists():
        logger.error(f"無法回滾：在 {archive_pkl_path} 找不到封存模型檔案")
        return False
        
    try:
        atomic_copy_file(str(archive_pkl_path), current_pkl_path)
        logger.info(f"🔄 成功將 {metadata.stock_id} 實盤模型回滾至版本 {metadata.timestamp}")
        return True
    except Exception as e:
        logger.error(f"回滾失敗 {metadata.stock_id}: {e}")
        return False