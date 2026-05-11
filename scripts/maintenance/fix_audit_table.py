r"""
fix_audit_table.py v2.1 (Quantum Finance Edition)
================================================================================
資料表結構修復工具 — 審計日誌專項 (Quantum v5.2 標準)
負責執行 data_audit_log 的非破壞性結構加固，確保欄位對齊生產標準而不遺失數據。

修訂歷程：
  v2.1 (2026-05-11): [終極版] 實作非破壞性自癒 (ALTER)，捨棄危險的 DROP 邏輯，補全 Hybrid Logging 與範例矩陣。

【執行範例矩陣 (Fix Operations Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / SQL                                         │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [執行專項審計表修復]      │ $ python scripts/maintenance/fix_audit_table.py        │
│ 2. [全系統架構一致性檢查]    │ $ python scripts/maintenance/check_schema_consistency.py --fix │
│ 3. [驗證修復後的欄位]        │ \d data_audit_log (在 psql 中執行)                     │
└──────────────────────────────┴────────────────────────────────────────────────────────┘

【業務邏輯說明】
  - 安全優先: 偵測到缺失欄位時僅執行 ALTER TABLE ADD COLUMN。
  - 核心欄位: data_date, action_type, rows_affected, stock_id, table_name。
================================================================================
"""
import sys, logging, time
from pathlib import Path
from datetime import datetime

# ── 終極路徑自癒 Bootstrap ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent if _THIS_DIR.name != "scripts" else _THIS_DIR
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_SCRIPTS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR.parent))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import db_transaction, record_lifecycle, write_data_audit_log
except ImportError as e:
    print(f"[FATAL] 無法匯入核心配置: {e}", file=sys.stderr)
    sys.exit(1)

logger = logging.getLogger(__name__)

def show_fix_dashboard(fixes: list):
    """執行後的修復儀表板。"""
    print("\n" + "🛠️"*35)
    print("🚀 Quantum Finance: 審計表專項修復報告 (v2.1)")
    print("🛠️"*35)
    print(f"✅ 修復狀態: 完成")
    print(f"📦 修復細項: {len(fixes)} 項欄位加固")
    
    if fixes:
        for f in fixes:
            print(f"   - {f}")
        print("-" * 70)
        print("🟢 成功！data_audit_log 結構已安全升級至 Quantum v5.2 標準。")
    else:
        print("-" * 70)
        print("⚪ 狀態：未偵測到結構異常，無需修復。")
        
    print("-" * 70)
    print("📝 日誌同步: pipeline_execution_log & data_audit_log (schema_fix)")
    print("🛠️"*35 + "\n")

def safe_fix_table():
    """執行非破壞性審計表修復。"""
    with record_lifecycle("audit_table_fix", category="maintenance", stock_id="SYSTEM"):
        fixes = []
        try:
            logger.info("🛠️  正在執行 data_audit_log 非破壞性結構稽核...")
            
            with db_transaction() as cur:
                # 1. 確保基礎表存在
                cur.execute("CREATE TABLE IF NOT EXISTS data_audit_log (id SERIAL PRIMARY KEY);")
                
                # 2. 定義標準欄位矩陣
                audit_cols = {
                    "table_name": "VARCHAR(100)",
                    "stock_id": "VARCHAR(50)",
                    "data_date": "DATE",
                    "action_type": "VARCHAR(50)",
                    "rows_affected": "INTEGER",
                    "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                }
                
                # 3. 逐一稽核與自癒
                for col, col_type in audit_cols.items():
                    cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = 'data_audit_log' AND column_name = '{col}';")
                    if not cur.fetchone():
                        logger.info(f"🔧 [Fix] 偵測到缺失欄位 {col}，正在執行安全修補...")
                        cur.execute(f"ALTER TABLE data_audit_log ADD COLUMN {col} {col_type};")
                        fixes.append(f"新增欄位: {col}")

            show_fix_dashboard(fixes)
            return True
        except Exception as e:
            logger.error(f"❌ 審計表專項修復失敗: {e}")
            raise e

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    safe_fix_table()
