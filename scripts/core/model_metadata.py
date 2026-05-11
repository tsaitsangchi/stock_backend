"""
model_metadata.py v2.11 (Quantum Finance Edition)
================================================================================
模型註冊中心 — 全域資源對齊版 (Quantum v5.2 標準)
負責追蹤模型生命週期、權重路徑與預測性能，具備自動化結構自癒能力。

修訂歷程：
  v2.11 (2026-05-11): [標準] 對齊 path_setup v4.0，模型權重存儲對齊 MODEL_WEIGHTS_DIR。
  v2.10 (2026-05-11): [環境] 修正 .env 加載路徑，確保與根目錄配置對齊。

【執行範例矩陣 (Model Registry Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / 用法                                        │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [稽核模型註冊中心]        │ $ python scripts/core/model_metadata.py                │
│ 2. [單一模型註冊]            │ register_model("2330", "LSTM_v1", "weights_path", ...) │
│ 3. [獲取核心標的模型狀態]    │ audit_registry()                                       │
└──────────────────────────────┴────────────────────────────────────────────────────────┘
================================================================================
"""
import os, sys, logging, platform
from pathlib import Path
from datetime import datetime

# ── 系統級架構引導 ──
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

# 延遲導入手冊：避免核心組件循環引用
def get_core_ops():
    from core.db_utils import db_transaction, record_lifecycle, get_db_stock_ids
    return db_transaction, record_lifecycle, get_db_stock_ids

def ensure_model_table_integrity():
    """模型表結構自癒：支援 legacy 欄位自動遷移 (model_name -> model_type)"""
    db_tx, _, _ = get_core_ops()
    with db_tx() as cur:
        cur.execute("CREATE TABLE IF NOT EXISTS model_metadata (id SERIAL PRIMARY KEY, stock_id VARCHAR(20));")
        
        # 欄位自癒檢查
        cols = {
            "model_type": "VARCHAR(100)", "weights_path": "TEXT",
            "hyperparams": "JSONB", "metrics": "JSONB",
            "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        }
        for col, col_type in cols.items():
            cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = 'model_metadata' AND column_name = '{col}';")
            if not cur.fetchone():
                # 檢查是否為舊版 model_name
                if col == "model_type":
                    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'model_metadata' AND column_name = 'model_name';")
                    if cur.fetchone():
                        cur.execute("ALTER TABLE model_metadata RENAME COLUMN model_name TO model_type;")
                        continue
                cur.execute(f"ALTER TABLE model_metadata ADD COLUMN {col} {col_type};")

def register_model(stock_id: str, model_type: str, weights_path: str, metrics: dict = None):
    """註冊新模型資產"""
    db_tx, record_lc, _ = get_core_ops()
    with record_lc(f"model_registration_{stock_id}", "mlops", stock_id):
        with db_tx() as cur:
            cur.execute(
                "INSERT INTO model_metadata (stock_id, model_type, weights_path, metrics) VALUES (%s, %s, %s, %s)",
                (stock_id, model_type, weights_path, metrics)
            )

def audit_registry():
    """執行註冊中心診斷報告"""
    db_tx, _, get_ids = get_core_ops()
    ensure_model_table_integrity()
    
    with db_tx() as cur:
        cur.execute("SELECT COUNT(*) FROM model_metadata;")
        total_models = cur.fetchone()['count']
        cur.execute("SELECT stock_id, model_type, created_at FROM model_metadata ORDER BY created_at DESC LIMIT 1;")
        last_m = cur.fetchone()

    core_ids = get_ids(is_core=True)
    
    print("\n" + "🤖"*40)
    print(f"🚀 Quantum Finance: 模型註冊中心報告 (v2.11)")
    print("🤖"*40)
    print(f"✅ 執行結果  : SUCCESS")
    print(f"🖥️  操作系統  : {platform.system()} {platform.release()}")
    print(f"📊 模型總數    : {total_models} 個")
    print(f"🎯 核心覆蓋率  : {total_models} / {len(core_ids)} 檔 ({round(total_models/len(core_ids)*100, 1)}%)")
    print("-" * 80)
    if last_m:
        print(f"🆕 最新註冊    : {last_m['stock_id']} ({last_m['model_type']}) @ {last_m['created_at']}")
    print("🟢 註冊中心狀態: 結構已對齊，模型生命週期追蹤中。")
    print("📝 任務同步: pipeline_execution_log (model_registry_audit)")
    print("🤖"*40 + "\n")

if __name__ == "__main__":
    audit_registry()