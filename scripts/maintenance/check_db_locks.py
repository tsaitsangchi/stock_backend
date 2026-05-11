"""
check_db_locks.py v2.2 (Quantum Finance Edition)
================================================================================
資料庫監控工具 — 即時連線與鎖定狀態審計器 (Quantum v5.2 標準)
負責掃描 PostgreSQL 中的活躍連線、交易鎖定 (Locks) 以及超過 30 秒的長查詢。

修訂歷程：
  v2.2 (2026-05-11): [標準化] 補全全場景維運範例矩陣、對齊 Hybrid Logging 規範。
  v2.1 (2026-05-11): [標準化] 導入生命週期紀錄與健康報告儀表板。

【執行範例矩陣 (Maintenance Usage Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / SQL                                         │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [即時連線健康檢查]        │ $ python scripts/maintenance/check_db_locks.py         │
│ 2. [診斷潛在死鎖問題]        │ 儀表板將自動顯示 "🔒 活躍鎖定數"，若大於 0 則需警惕。  │
│ 3. [查閱歷史阻塞紀錄]        │ SELECT * FROM pipeline_execution_log WHERE task_name = 'db_lock_check'; │
│ 4. [強制終止長查詢]          │ SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE ...; │
└──────────────────────────────┴────────────────────────────────────────────────────────┘

【監控指標說明】
  - 活躍連線數: 目前連接到資料庫的 Session 總數。
  - 活躍鎖定數: 正在等待或持有的資料列/資料表鎖定數量。
  - 長查詢數: 執行時間超過 30 秒的 SQL，通常是平行入庫時的瓶頸。
================================================================================
"""
import sys, logging, time
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
    from core.db_utils import db_transaction, record_lifecycle
except ImportError as e:
    print(f"[FATAL] 無法匯入核心配置: {e}", file=sys.stderr)
    sys.exit(1)

logger = logging.getLogger(__name__)

def show_lock_dashboard(stats: dict):
    """執行後的連線健康儀表板。"""
    print("\n" + "="*50)
    print("🕵️  Quantum Finance: 資料庫連線健康報告 (v2.2)")
    print("="*50)
    print(f"✅ 掃描狀態  : 完成")
    print(f"🏊 活躍連線數: {stats['active_conns']}")
    print(f"🔒 活躍鎖定數: {stats['lock_count']}")
    print(f"⏳ 長查詢數  : {stats['long_queries']} (超過 30 秒)")
    
    if stats['lock_count'] > 0:
        print("🔴 警報：偵測到活躍鎖定，可能存在交易阻塞 (Transaction Blocking)！")
    elif stats['active_conns'] > 25:
        print("🟡 警告：連線數較高，請檢查連線池是否正確釋放。")
    else:
        print("🟢 系統目前運行流暢，無阻塞查詢。")
        
    print("-" * 50)
    print("📝 日誌同步: pipeline_execution_log (db_lock_check)")
    print("="*50 + "\n")

def check_locks():
    """執行連線與鎖定稽核。"""
    with record_lifecycle("db_lock_check", category="maintenance", stock_id="DB_SYSTEM"):
        stats = {"active_conns": 0, "lock_count": 0, "long_queries": 0}
        try:
            with db_transaction() as cur:
                # 1. 查詢活躍連線
                cur.execute("SELECT count(*) FROM pg_stat_activity WHERE state = 'active';")
                stats['active_conns'] = cur.fetchone()['count']
                
                # 2. 查詢鎖定數
                cur.execute("SELECT count(*) FROM pg_locks WHERE granted = true;")
                stats['lock_count'] = cur.fetchone()['count']
                
                # 3. 查詢長查詢
                cur.execute("SELECT count(*) FROM pg_stat_activity WHERE (now() - query_start) > interval '30 seconds' AND state = 'active';")
                stats['long_queries'] = cur.fetchone()['count']
            
            show_lock_dashboard(stats)
            return True
        except Exception as e:
            logger.error(f"❌ 資料庫監控失敗: {e}")
            raise e

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    check_locks()