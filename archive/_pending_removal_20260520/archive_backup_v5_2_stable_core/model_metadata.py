"""
model_metadata.py v2.12 (Quantum Finance Edition)
================================================================================
模型註冊中心 — 極致範例版 (Quantum v5.2 標準)
負責追蹤模型生命週期與權重路徑，支援高度靈活的模型資產稽核。

修訂歷程：
  v2.12 (2026-05-11): [標準] 補全極致範例矩陣，包含單一模型註冊與核心覆蓋率稽核。
  v2.11 (2026-05-11): [標準] 對齊 path_setup v4.0。
  v2.10 (2026-05-11): [環境] 修正 .env 加載路徑。

【執行範例矩陣 (Model Registry Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / 用法                                        │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [全系統模型資產稽核]      │ $ python scripts/core/model_metadata.py                │
│ 2. [單一模型：手動註冊]      │ register_model("2330", "Transformer", "path/to/pt", ..)│
│ 3. [核心標的：覆蓋率詳查]    │ audit_registry(is_core=True)                           │
│ 4. [強制執行：Schema自癒]    │ ensure_model_table_integrity()                         │
└──────────────────────────────┴────────────────────────────────────────────────────────┘
================================================================================
"""
import os, sys, logging, platform
from pathlib import Path

# ── 系統級架構引導 ──
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from core.db_utils import db_transaction, record_lifecycle, get_db_stock_ids, ensure_infrastructure

def ensure_model_table_integrity():
    with db_transaction() as cur:
        cur.execute("CREATE TABLE IF NOT EXISTS model_metadata (id SERIAL PRIMARY KEY, stock_id VARCHAR(20));")
        cols = {"model_type": "VARCHAR(100)", "weights_path": "TEXT", "metrics": "JSONB", "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"}
        for col, col_type in cols.items():
            cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = 'model_metadata' AND column_name = '{col}';")
            if not cur.fetchone(): cur.execute(f"ALTER TABLE model_metadata ADD COLUMN {col} {col_type};")

def register_model(stock_id: str, model_type: str, weights_path: str, metrics: dict = None):
    with record_lifecycle(f"model_registration", "mlops", stock_id):
        with db_transaction() as cur:
            cur.execute("INSERT INTO model_metadata (stock_id, model_type, weights_path, metrics) VALUES (%s, %s, %s, %s)", (stock_id, model_type, weights_path, metrics))

def audit_registry():
    ensure_infrastructure(); ensure_model_table_integrity()
    with db_transaction() as cur:
        cur.execute("SELECT COUNT(*) FROM model_metadata;")
        total = cur.fetchone()['count']
    core_ids = get_db_stock_ids(is_core=True)
    print("\n" + "🤖"*40)
    print(f"🚀 Quantum Finance: 模型資產稽核報告 (v2.12)\n✅ 總量: {total} | 核心覆蓋: {round(total/len(core_ids)*100, 1)}%")
    print("🤖"*40 + "\n")

if __name__ == "__main__":
    audit_registry()