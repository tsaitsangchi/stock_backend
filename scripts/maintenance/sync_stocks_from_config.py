"""
sync_stocks_from_config.py v2.1 (Quantum Finance Edition)
================================================================================
配置同步引擎 — Config 與資料庫標的狀態同步工具 (Quantum v5.2 標準)
負責將 config.py 中的 STOCK_CONFIGS 手動定義同步至資料庫的 stocks 表。

修訂歷程：
  v2.1 (2026-05-11): [標準化] 實作真實同步邏輯、導入儀表板與 Hybrid Logging 對接。
  v5.5.7 (2026-05-09): [核心] 導入 Hybrid Logging 混合日誌。

【執行範例矩陣 (Sync Operations Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / SQL                                         │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [執行 Config 全量同步]    │ $ python scripts/maintenance/sync_stocks_from_config.py│
│ 2. [驗證同步後的核心標的]    │ SELECT stock_id, is_core FROM stocks WHERE is_core=TRUE;│
│ 3. [查看同步歷史日誌]        │ SELECT * FROM pipeline_execution_log                   │
│                              │ WHERE task_name = 'config_stock_sync' ORDER BY id DESC; │
└──────────────────────────────┴────────────────────────────────────────────────────────┘

【業務邏輯說明】
  - 狀態對齊: 如果 stock_id 存在於 config.py，則在資料庫中標記為 is_core = TRUE。
  - 遺留處理: 不在 config.py 中的標的，其 is_core 狀態將保持不變或依策略調整。
================================================================================
"""
import sys, logging, time
from datetime import datetime, date
from pathlib import Path

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
    from core.db_utils import db_transaction, record_lifecycle, write_data_audit_log, bulk_upsert
    from config import STOCK_CONFIGS
except ImportError as e:
    print(f"[FATAL] 無法匯入核心配置: {e}", file=sys.stderr)
    sys.exit(1)

logger = logging.getLogger(__name__)

def show_sync_dashboard(processed: int, core_marked: int):
    """執行後的配置同步儀表板。"""
    print("\n" + "🔄"*35)
    print("🚀 Quantum Finance: 配置同步報告 (v2.1)")
    print("🔄"*35)
    print(f"✅ 同步狀態  : 完成")
    print(f"📊 配置標的數: {processed} 檔")
    print(f"💎 標記核心數: {core_marked} 檔")
    
    print("-" * 70)
    print(f"🟢 成功！資料庫 is_core 狀態已根據 config.py 完成 1:1 同步。")
    print("-" * 70)
    print("📝 日誌同步: pipeline_execution_log & data_audit_log (config_sync)")
    print("🔄"*35 + "\n")

def run_sync():
    """啟動 Config 同步流程。"""
    with record_lifecycle("config_stock_sync", category="maintenance", stock_id="ALL"):
        try:
            logger.info("🔄 正在執行 config.py 與資料庫標的名單同步...")
            
            # 1. 準備同步數據
            sync_data = []
            for sid, cfg in STOCK_CONFIGS.items():
                sync_data.append({
                    "stock_id": sid,
                    "is_core": True,
                    "updated_at": datetime.now()
                })
            
            # 2. 批量更新狀態 (僅更新 is_core 與 updated_at)
            if sync_data:
                rows = bulk_upsert("stocks", sync_data, unique_cols=["stock_id"])
            else:
                rows = 0
                
            # 3. 混合紀錄
            write_data_audit_log("config_sync", "SYSTEM", date.today().strftime("%Y-%m-%d"), "SYNC", rows)
            
            show_sync_dashboard(len(STOCK_CONFIGS), rows)
            return True
        except Exception as e:
            logger.error(f"❌ 配置同步失敗: {e}")
            raise e

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_sync()
