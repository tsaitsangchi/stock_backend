"""
model_metadata.py v2.3 (Quantum Finance Edition)
================================================================================
模型詮釋資料與版本控制 — 原子寫入完整性版 (Quantum v5.1 標準)
提供量化模型的特徵指紋比對、原子寫入保障、以及歷史版本回滾支援。

修訂歷程：
  v2.3 (2026-05-10): [核心] 整合 Active Symlinks 自動更新機制。
  v2.2 (2026-05-10): [修正] 強化 Bootstrap 邏輯。

【執行範例矩陣 — 模型管理方案】
1. 模型註冊與封存 (Python)：
   meta = ModelMetadata(...); save_model_registry(meta)
2. 特徵指紋校驗 (Python)：
   assert_feature_schema_match(meta, ["price", "volume"])
3. 模型註冊中心稽核 (SQL)：
   SELECT stock_id, model_name, feature_count, oof_da FROM model_metadata ORDER BY trained_at DESC;
4. 原子 JSON 封存檢查 (Shell)：
   ls scripts/outputs/models/archive/*.json
================================================================================
"""
import os, sys, json, shutil, hashlib, subprocess, threading, logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

# ── 終極路徑自癒 Bootstrap (核心自救版) ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_SCRIPTS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR.parent))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import db_transaction, write_pipeline_log
except ImportError:
    import path_setup
    path_setup.ensure_scripts_on_path(__file__)
    from db_utils import db_transaction, write_pipeline_log

logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    stock_id: str
    model_name: str
    model_path: str
    timestamp: str
    git_hash: Optional[str]
    python_version: str = sys.version.split()[0]
    feature_count: int = 0
    feature_fingerprint: str = ""
    oof_da: float = 0.0
    oof_sharpe: float = 0.0
    oof_ic: float = 0.0
    oof_n_samples: int = 0
    params: Dict = None
    notes: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

_locks_dict_lock = threading.Lock()
_path_locks: Dict[str, threading.Lock] = {}

class PathLock:
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

def get_git_hash(short: bool = True) -> Optional[str]:
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return None

def fingerprint_features(features: List[str]) -> str:
    sorted_features = sorted(features)
    feature_str = ",".join(sorted_features)
    return hashlib.sha256(feature_str.encode("utf-8")).hexdigest()[:12]

def atomic_write_json(path: str, data: dict):
    tmp_path = str(path) + ".tmp"
    with PathLock(path):
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        os.replace(tmp_path, path)

def atomic_copy_file(src: str, dst: str):
    tmp_dst = str(dst) + ".tmp"
    with PathLock(dst):
        shutil.copy2(src, tmp_dst)
        os.replace(tmp_dst, dst)

def save_model_registry(metadata: ModelMetadata):
    sql = """
        INSERT INTO model_metadata (
            stock_id, model_name, model_path, accuracy, oof_da, oof_sharpe, 
            feature_count, params, trained_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (stock_id, model_name) DO UPDATE SET
            model_path = EXCLUDED.model_path,
            accuracy = EXCLUDED.accuracy,
            oof_da = EXCLUDED.oof_da,
            oof_sharpe = EXCLUDED.oof_sharpe,
            feature_count = EXCLUDED.feature_count,
            params = EXCLUDED.params,
            trained_at = CURRENT_TIMESTAMP
    """
    try:
        with db_transaction() as cur:
            cur.execute(sql, (
                metadata.stock_id, metadata.model_name, metadata.model_path, 
                metadata.oof_da, metadata.oof_da, metadata.oof_sharpe,
                metadata.feature_count, json.dumps(metadata.params)
            ))
        logger.info(f"✅ [Registry] {metadata.stock_id} 註冊成功 (DB)")
    except Exception as e:
        logger.error(f"❌ [Registry] {metadata.stock_id} 註冊失敗: {e}")

    # 2. 本地原子 JSON 封存與檔案備份
    try:
        from core.path_setup import get_archive_dir, update_latest_link
    except ImportError:
        from path_setup import get_archive_dir, update_latest_link
        
    archive_dir = get_archive_dir()
    filename_base = f"model_{metadata.stock_id}_{metadata.timestamp}_{metadata.git_hash or 'nogit'}"
    json_path = os.path.join(archive_dir, f"{filename_base}.metadata.json")
    
    # 寫入詮釋資料
    atomic_write_json(json_path, metadata.to_dict())
    
    # 複製模型實體檔案
    archive_pkl_path = os.path.join(archive_dir, f"{filename_base}.pkl")
    if metadata.model_path and os.path.exists(metadata.model_path):
        atomic_copy_file(metadata.model_path, archive_pkl_path)
    
    # 3. 更新 Active Symlinks 指標 (僅適用於本地路徑)
    link_name = f"{metadata.stock_id}_latest.pkl"
    if os.path.exists(archive_pkl_path):
        update_latest_link(archive_pkl_path, link_name)
    
    logger.info(f"✅ [Archive] {metadata.stock_id} 模型封存至 {archive_dir}")

def assert_feature_schema_match(metadata: ModelMetadata, current_features: List[str]):
    current_fingerprint = fingerprint_features(current_features)
    if current_fingerprint != metadata.feature_fingerprint:
        raise RuntimeError(f"Feature schema mismatch! Expected {metadata.feature_fingerprint}, got {current_fingerprint}.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    # 測試用模型檔案
    test_file = "test_model_2330.pkl"
    Path(test_file).write_text("dummy model")
    
    meta = ModelMetadata(stock_id="2330", model_name="Ensemble_ML", model_path=test_file, timestamp="20260510", git_hash=get_git_hash())
    save_model_registry(meta)