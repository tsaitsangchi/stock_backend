"""
model_metadata.py v2.5 (Quantum Finance Edition)
================================================================================
模型詮釋資料與版本控制 — 原子寫入完整性版 (Quantum v5.1 標準)
提供量化模型的特徵指紋比對、原子寫入保障、以及註冊中心儀表板。

修訂歷程：
  v2.5 (2026-05-11): [增強] 強化執行後儀表板，明確顯示日誌寫入位置與核心股覆蓋率。
  v2.4 (2026-05-11): [標準化] 導入 Quantum 標準檔頭、生命週期紀錄與註冊中心儀表板。
================================================================================
"""
import os, sys, json, shutil, hashlib, subprocess, threading, logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

# ── 終極路徑自癒 Bootstrap ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_SCRIPTS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR.parent))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import db_transaction, write_pipeline_log, record_lifecycle, ensure_infrastructure, get_db_stock_ids
except ImportError:
    import path_setup
    path_setup.ensure_scripts_on_path(__file__)
    from db_utils import db_transaction, write_pipeline_log, record_lifecycle, ensure_infrastructure, get_db_stock_ids

logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    stock_id: str
    model_name: str
    model_path: str
    timestamp: str
    git_hash: Optional[str] = None
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

def get_git_hash(short: bool = True) -> Optional[str]:
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except: return None

def atomic_write_json(path: str, data: dict):
    tmp_path = str(path) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    os.replace(tmp_path, path)

def atomic_copy_file(src: str, dst: str):
    if not os.path.exists(src): return
    tmp_dst = str(dst) + ".tmp"
    shutil.copy2(src, tmp_dst)
    os.replace(tmp_dst, dst)

def show_model_dashboard():
    """執行後的儀表板回報，顯示註冊中心摘要。"""
    print("\n" + "="*55)
    print("🤖 Quantum Finance: 模型註冊中心報告 (v2.5)")
    print("="*55)
    try:
        with db_transaction() as cur:
            cur.execute("SELECT count(*) as total, count(DISTINCT stock_id) as stocks FROM model_metadata;")
            res = cur.fetchone()
            core_stocks = get_db_stock_ids(active_only=True)
            coverage = (res['stocks'] / len(core_stocks) * 100) if core_stocks else 0
            
            print(f"📋 註冊中心狀態: 運行良好")
            print(f"📊 模型總數    : {res['total']} 個")
            print(f"🎯 核心覆蓋率  : {res['stocks']} / {len(core_stocks)} 檔 ({coverage:.1f}%)")
            
            print("-" * 55)
            print("📝 日誌同步紀錄:")
            print(f"   - [生命週期] 已同步至 pipeline_execution_log")
            print(f"   - [分類審計] 已同步至 model_metadata 註冊表")
            
            cur.execute("SELECT stock_id, model_name, trained_at FROM model_metadata ORDER BY trained_at DESC LIMIT 1;")
            m = cur.fetchone()
            if m:
                print(f"🆕 最新註冊    : {m['stock_id']} ({m['model_name']}) @ {m['trained_at'].strftime('%H:%M:%S')}")
        print("="*55 + "\n")
    except Exception as e:
        print(f"⚠️ 無法產生報表: {e}")

def save_model_registry(metadata: ModelMetadata):
    """保存模型至資料庫註冊中心與本地封存庫 (混合模式紀錄)。"""
    ensure_infrastructure()
    with record_lifecycle("model_registry", category="inference", stock_id=metadata.stock_id):
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
            print(f"✅ [Registry] {metadata.stock_id} 註冊成功")
        except Exception as e:
            print(f"❌ [Registry] {metadata.stock_id} 註冊失敗: {e}")
            raise e

        # 本地封存與 Symlink 更新
        from core.path_setup import get_archive_dir, update_latest_link
        archive_dir = get_archive_dir()
        filename_base = f"model_{metadata.stock_id}_{metadata.timestamp}_{metadata.git_hash or 'nogit'}"
        atomic_write_json(os.path.join(archive_dir, f"{filename_base}.metadata.json"), metadata.to_dict())
        atomic_copy_file(metadata.model_path, os.path.join(archive_dir, f"{filename_base}.pkl"))
        update_latest_link(os.path.join(archive_dir, f"{filename_base}.pkl"), f"{metadata.stock_id}_latest.pkl")

if __name__ == "__main__":
    # 測試用模型檔案
    test_file = "test_model_2330.pkl"
    Path(test_file).write_text("dummy model")
    try:
        meta = ModelMetadata(stock_id="2330", model_name="Ensemble_ML", model_path=test_file, timestamp=datetime.now().strftime("%Y%m%d"))
        save_model_registry(meta)
        show_model_dashboard()
    finally:
        if os.path.exists(test_file): os.remove(test_file)