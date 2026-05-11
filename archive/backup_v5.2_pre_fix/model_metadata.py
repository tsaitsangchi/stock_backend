"""
model_metadata.py v2.12 (Quantum Finance Edition)
================================================================================
模型註冊中心 — 極致範例版 (Quantum v5.2 標準)
負責追蹤模型生命週期與權重路徑。

修訂歷程：
  v2.12 (2026-05-11): [標準] 補全極致範例矩陣。
  v2.11 (2026-05-11): [標準] 對齊 path_setup v4.0。

【執行範例矩陣 (Model Registry Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / 用法                                        │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [全系統模型資產稽核]      │ $ python scripts/core/model_metadata.py                │
│ 2. [單一模型：手動註冊]      │ register_model("2330", "Transformer", "path", {})      │
└──────────────────────────────┴────────────────────────────────────────────────────────┘
================================================================================
"""
import os, sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from core.db_utils import db_transaction, record_lifecycle, ensure_infrastructure

def ensure_model_table_integrity():
    with db_transaction() as cur:
        cur.execute("CREATE TABLE IF NOT EXISTS model_metadata (id SERIAL PRIMARY KEY, stock_id VARCHAR(20), model_type VARCHAR(100), weights_path TEXT, metrics JSONB, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);")

def register_model(stock_id: str, model_type: str, weights_path: str, metrics: dict = None):
    with record_lifecycle("model_registration", "mlops", stock_id):
        with db_transaction() as cur:
            cur.execute("INSERT INTO model_metadata (stock_id, model_type, weights_path, metrics) VALUES (%s, %s, %s, %s)", (stock_id, model_type, weights_path, metrics))

def audit_registry():
    ensure_infrastructure(); ensure_model_table_integrity()
    with db_transaction() as cur:
        cur.execute("SELECT COUNT(*) FROM model_metadata;")
        print(f"Total Models: {cur.fetchone()['count']}")

if __name__ == "__main__":
    audit_registry()