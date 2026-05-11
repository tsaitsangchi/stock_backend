"""
sync_stocks_from_config.py v2.2 (Quantum Finance Edition)
================================================================================
配置同步工具 — Config.py 與資料庫標的名單 1:1 對齊 (Quantum v5.2 標準)
負責根據 config.py 中的 STOCK_LIST 更新資料庫中的 is_core 標記。

修訂歷程：
  v2.2 (2026-05-11): [標準化] 補全「128 檔標的 1:1 對齊、強制配置重置、核心標記同步」之範例矩陣。
  v2.1 (2026-05-11): [標準化] 導入同步報告儀表板與 Hybrid Logging。

【執行範例矩陣 (Sync Operations Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令                                               │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [128 檔核心標的 1:1 同步] │ $ python scripts/maintenance/sync_stocks_from_config.py │
│ 2. [強制重置所有非核心標記]  │ 腳本會自動將不在 config.py 中的標記為 is_core = FALSE。│
│ 3. [驗證核心標記狀態 (SQL)]  │ SELECT stock_id FROM stocks WHERE is_core = TRUE;      │
└──────────────────────────────┴────────────────────────────────────────────────────────┘
================================================================================
"""
import sys, logging, time
from pathlib import Path
from datetime import datetime

_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent if _THIS_DIR.name != "scripts" else _THIS_DIR
if str(_SCRIPTS_DIR) not in sys.path: sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from core.db_utils import db_transaction, record_lifecycle, write_data_audit_log
    from config import STOCK_LIST
except ImportError:
    print("[FATAL] 無法匯入核心配置。")
    sys.exit(1)

def show_sync_dashboard(core_count: int):
    print("\n" + "🔄"*35)
    print("🚀 Quantum Finance: 配置同步報告 (v2.2)")
    print("🔄"*35)
    print(f"✅ 同步狀態  : 完成")
    print(f"📊 配置標的數: {len(STOCK_LIST)} 檔")
    print(f"💎 標記核心數: {core_count} 檔")
    print("-" * 70)
    print("🟢 成功！資料庫 is_core 狀態已根據 config.py 完成 1:1 同步。")
    print("📝 日誌同步: pipeline_execution_log & data_audit_log (config_sync)")
    print("🔄"*35 + "\n")

def run_sync():
    with record_lifecycle("config_sync", category="maintenance", stock_id="SYSTEM"):
        with db_transaction() as cur:
            # 1. 重置所有核心標記
            cur.execute("UPDATE stocks SET is_core = FALSE;")
            # 2. 根據 config.py 更新核心標記
            cur.execute("UPDATE stocks SET is_core = TRUE WHERE stock_id = ANY(%s);", (STOCK_LIST,))
            cur.execute("SELECT count(*) FROM stocks WHERE is_core = TRUE;")
            count = cur.fetchone()['count']
            
        write_data_audit_log("stocks", "SYSTEM", datetime.now().strftime("%Y-%m-%d"), "CONFIG_SYNC", count)
        show_sync_dashboard(count)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_sync()
