"""
model_metadata.py v2.10 (Quantum Finance Edition)
================================================================================
模型註冊中心 — 最終穩定版 (Quantum v5.2 標準)
負責追蹤模型訓練進度，並具備自動修正舊版表格結構的能力。

修訂歷程：
  v2.10 (2026-05-11): [修復] 補回遺漏的 register_model 函式，恢復完整功能。
  v2.9 (2026-05-11): [自癒] 實作結構自癒與欄位對齊。
================================================================================
"""
import sys, logging, time, platform, json
from pathlib import Path
from datetime import datetime

# ── 系統級架構引導 ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_ROOT = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
if str(_SCRIPTS_ROOT) not in sys.path: sys.path.insert(0, str(_SCRIPTS_ROOT))

# 採用延遲導入防止循環引用
def get_core_ops():
    from core.db_utils import db_transaction, record_lifecycle, get_db_stock_ids
    return db_transaction, record_lifecycle, get_db_stock_ids

def register_model(stock_id: str, model_type: str, metrics: dict):
    """手動註冊模型元數據。"""
    db_tx, _, _ = get_core_ops()
    with db_tx() as cur:
        cur.execute("""
            INSERT INTO model_metadata (stock_id, model_type, metrics)
            VALUES (%s, %s, %s)
        """, (stock_id, model_type, json.dumps(metrics)))
    print(f"✅ 模型註冊成功: {stock_id} ({model_type})")

def ensure_model_table_integrity():
    """深度檢查並修正 model_metadata 表格結構。"""
    db_tx, _, _ = get_core_ops()
    with db_tx() as cur:
        cur.execute("CREATE TABLE IF NOT EXISTS model_metadata (id SERIAL PRIMARY KEY);")
        cols_to_check = {
            "stock_id": "VARCHAR(20)", "model_type": "VARCHAR(50)",
            "metrics": "JSONB", "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        }
        for col, col_type in cols_to_check.items():
            cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = 'model_metadata' AND column_name = '{col}';")
            if not cur.fetchone():
                if col == "model_type":
                    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'model_metadata' AND column_name = 'model_name';")
                    if cur.fetchone():
                        cur.execute("ALTER TABLE model_metadata RENAME COLUMN model_name TO model_type;"); continue
                cur.execute(f"ALTER TABLE model_metadata ADD COLUMN {col} {col_type};")

def show_registry_dashboard(stats: dict):
    print("\n" + "🤖"*40)
    print("🚀 Quantum Finance: 模型註冊中心報告 (v2.10)")
    print("🤖"*40)
    print(f"✅ 執行結果  : SUCCESS")
    print(f"🖥️  操作系統  : {platform.system()} {platform.release()}")
    print(f"📊 模型總數    : {stats['model_count']} 個")
    print(f"🎯 核心覆蓋率  : {stats['model_count']} / {stats['core_count']} 檔 ({stats['coverage']:.1f}%)")
    print("-" * 80)
    if stats['latest']:
        print(f"🆕 最新註冊    : {stats['latest']['stock_id']} ({stats['latest']['model_type']}) @ {stats['latest']['created_at']}")
    print("🟢 註冊中心狀態: 結構已對齊，模型生命週期追蹤中。")
    print("🤖"*40 + "\n")

def audit_registry():
    db_tx, record_lc, get_ids = get_core_ops()
    ensure_model_table_integrity()
    with record_lc("model_registry_audit", category="ml", stock_id="ML_SYSTEM"):
        with db_tx() as cur:
            cur.execute("SELECT COUNT(DISTINCT stock_id) FROM model_metadata;")
            model_count = cur.fetchone()['count']
            cur.execute("SELECT stock_id, model_type, created_at FROM model_metadata ORDER BY created_at DESC LIMIT 1;")
            latest = cur.fetchone()
        core_ids = get_ids(is_core=True)
        core_count = len(core_ids) if core_ids else 128
        show_registry_dashboard({
            "model_count": model_count, "core_count": core_count, "coverage": (model_count / core_count * 100) if core_count > 0 else 0, "latest": latest
        })

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    audit_registry()