"""
check_db_locks.py v2.1 (Quantum Finance Edition)
================================================================================
資料庫鎖定掃描器 — 系統穩定性工具 (Quantum v5.2 標準)
負責監測資料庫連線狀態、活躍鎖定 (Locks) 以及掛起的長時間查詢。

修訂歷程：
  v2.1 (2026-05-11): [標準化] 導入 Quantum 標準檔頭、生命週期監測與連線儀表板。
  v5.5.7 (2026-05-09): [核心] 整合基礎混合日誌。

執行範例 (Comprehensive Usage Examples):
  1. [系統健康掃描] 掃描目前所有活躍鎖定與掛起查詢:
     python scripts/maintenance/check_db_locks.py

  2. [系統監控] 查看過去 24 小時的鎖定紀錄 (SQL):
     SELECT * FROM pipeline_execution_log WHERE task_name = 'db_lock_check' ORDER BY created_at DESC;

  3. [強制終止異常連線] 若發現長查詢，可手動終止 (SQL 參考):
     SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'active' AND now() - query_start > interval '5 minutes';
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

def show_lock_dashboard(stats: dict):
    """執行後的連線儀表板回報。"""
    print("\n" + "="*50)
    print("🕵️ Quantum Finance: 資料庫連線健康報告 (v2.1)")
    print("="*50)
    print(f"✅ 掃描狀態  : 完成")
    print(f"🏊 活躍連線數: {stats.get('total_active', 0)}")
    print(f"🔒 活躍鎖定數: {stats.get('lock_count', 0)}")
    print(f"⏳ 長查詢數  : {stats.get('long_queries', 0)} (超過 30 秒)")
    
    if stats.get('lock_count', 0) > 0:
        print("⚠️ 警告：偵測到活躍鎖定，請注意數據寫入衝突！")
    else:
        print("🟢 系統目前運行流暢，無阻塞查詢。")
    
    print("-" * 50)
    print("📝 日誌同步: pipeline_execution_log (db_lock_check)")
    print("="*50 + "\n")

def check_locks():
    """執行多維度鎖定與連線掃描。"""
    stats = {}
    with record_lifecycle("db_lock_check", category="sys", stock_id="SYSTEM"):
        try:
            with db_transaction() as cur:
                # 1. 掃描活躍鎖定
                cur.execute("""
                    SELECT count(*) as cnt FROM pg_stat_activity 
                    WHERE wait_event_type IS NOT NULL AND state = 'active';
                """)
                stats['lock_count'] = cur.fetchone()['cnt']
                
                # 2. 掃描總活躍連線
                cur.execute("SELECT count(*) as cnt FROM pg_stat_activity WHERE state = 'active';")
                stats['total_active'] = cur.fetchone()['cnt']
                
                # 3. 掃描長查詢 (超過 30 秒)
                cur.execute("""
                    SELECT count(*) as cnt FROM pg_stat_activity 
                    WHERE state = 'active' AND now() - query_start > interval '30 seconds';
                """)
                stats['long_queries'] = cur.fetchone()['cnt']
                
                # 4. 混合模式：系統維護審計
                write_data_audit_log("pg_stat_activity", "SYSTEM", datetime.now().strftime("%Y-%m-%d"), "SCAN", stats['lock_count'])
                
            show_lock_dashboard(stats)
            return True
        except Exception as e:
            logger.error(f"❌ 鎖定掃描失敗: {e}")
            raise e

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    check_locks()