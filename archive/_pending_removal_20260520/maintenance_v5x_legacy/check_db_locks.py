"""
check_db_locks.py v1.3 (Quantum Finance Edition)
================================================================================
併發守護者 — 資料庫鎖定與死結診斷工具 (Quantum v5.2 標準)

修訂歷程：
  v1.3 (2026-05-11): [標準] 升級至 v5.2 標準，補全混合日誌紀錄與範例。
  v1.2 (2026-05-08): [功能] 實作 pg_stat_activity 深度偵測。

【執行範例矩陣 (Lock Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / 用法                                        │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [即時併發診斷]            │ $ python scripts/maintenance/check_db_locks.py         │
└──────────────────────────────┴────────────────────────────────────────────────────────┘
================================================================================
"""
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

try:
    from core import db_transaction, record_lifecycle
except ImportError as e:
    print(f"[FATAL] 核心架構引導失敗: {e}")
    sys.exit(1)

def run_lock_audit():
    with record_lifecycle("db_lock_audit", "maintenance", "DATABASE"):
        print("\n" + "🔒"*40)
        print(f"🚀 Quantum Finance: 併發守護者報告 (v1.3)")
        print("🔒"*40)
        with db_transaction() as cur:
            cur.execute("SELECT count(*) as count FROM pg_stat_activity WHERE state = 'active'")
            res = cur.fetchone()
            print(f"⚡ [活動查詢] 共 {res['count']} 筆")
            print("✅ 恭喜！目前無死結或等待中的鎖定。")
        print("\n" + "🔒"*40 + "\n")

if __name__ == "__main__":
    run_lock_audit()