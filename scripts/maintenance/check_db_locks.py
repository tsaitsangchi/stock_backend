"""
check_db_locks.py v2.5 (Quantum Finance Edition)
================================================================================
資料庫連線與鎖定診斷工具 — 系統穩定性哨兵 (Quantum v5.2 標準)
負責監控連線池狀態、事務鎖定與死鎖風險。

修訂歷程：
  v2.5 (2026-05-11): [標準化] 對齊 v3.7 視覺報告標準，加入執行結果與系統資訊。
  v2.4 (2026-05-11): [最佳架構] 接入 core 統一接口。

【執行範例矩陣 (Lock Audit Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令                                               │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [全系統資料庫連線診斷]    │ $ python scripts/maintenance/check_db_locks.py         │
│ 2. [診斷特定資料表鎖定]      │ $ python scripts/maintenance/check_db_locks.py --table_name stock_price │
└──────────────────────────────┴────────────────────────────────────────────────────────┘
================================================================================
"""
import sys, logging, argparse, platform
from pathlib import Path

# ── 最佳架構引導 ──
try:
    import core
    from core import db_transaction, record_lifecycle
except ImportError:
    print("[FATAL] 核心架構引導失敗。")
    sys.exit(1)

def show_lock_dashboard(conns: list, locks: list):
    print("\n" + "🔒"*40)
    print("🚀 Quantum Finance: 資料庫連線與鎖定報告 (v2.5)")
    print("🔒"*40)
    print(f"✅ 執行結果  : SUCCESS")
    print(f"🖥️  操作系統  : {platform.system()} {platform.release()}")
    print(f"🔌 活躍連線  : {len(conns)} 筆")
    print(f"⚠️  待處理鎖定: {len(locks)} 筆")
    
    if locks:
        print("\n[危險訊號] 偵測到活動鎖定：")
        for l in locks:
            print(f"   - Table: {l['relname']:<20} | Mode: {l['mode']:<15} | PID: {l['pid']}")
    else:
        print("\n🟢 系統運行平穩：未偵測到任何阻塞鎖定。")
        
    print("-" * 80)
    print("📝 任務同步: pipeline_execution_log (db_lock_audit)")
    print("🔒"*40 + "\n")

def audit_locks(table_name=None):
    with record_lifecycle("db_lock_audit", category="maintenance", stock_id="DB_SERVER"):
        with db_transaction() as cur:
            cur.execute("SELECT pid, usename, state, query FROM pg_stat_activity WHERE state != 'idle';")
            conns = cur.fetchall()
            sql = "SELECT t.relname, l.locktype, l.mode, l.pid, l.granted FROM pg_locks l JOIN pg_stat_all_tables t ON l.relation = t.relid WHERE t.schemaname = 'public'"
            if table_name: sql += f" AND t.relname = '{table_name}'"
            cur.execute(sql)
            locks = cur.fetchall()
        show_lock_dashboard(conns, locks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--table_name", type=str)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    audit_locks(table_name=args.table_name)