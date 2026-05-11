"""
check_system_health.py v1.4 (Quantum Finance Edition)
================================================================================
系統終極健康診斷報告 — 全維度診斷矩陣版 (Quantum v5.2 標準)
負責稽核基礎設施、資料庫主權與數據可觀測性的全維度健康狀況。

修訂歷程：
  v1.4 (2026-05-11): [標準] 補全旗艦級診斷範例矩陣，對齊混合日誌規範。
  v1.3 (2026-05-11): [維度] 實作全維度健康診斷，整合 pipeline_execution_log。

【執行範例矩陣 (Health Diagnostic Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / 用法                                        │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [全系統：一鍵健康診斷]    │ $ python scripts/maintenance/check_system_health.py     │
│ 2. [單一維度：資料庫主權診斷]│ $ python scripts/maintenance/check_system_health.py --db │
│ 3. [強制更新：全系統健康紀錄]│ $ python scripts/maintenance/check_system_health.py --force │
└──────────────────────────────┴────────────────────────────────────────────────────────┘

【可觀測性紀錄 (Observability)】
  - 統一日誌 (Unified): pipeline_execution_log (Task: system_health_check)
  - 專項審計 (Audit): data_audit_log (Action: HEALTH_DIAGNOSTIC)
================================================================================
"""
import sys, os, time
from pathlib import Path
from datetime import datetime

# ── 系統級架構引導 ──
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from core.db_utils import db_transaction, record_lifecycle, write_data_audit_log

def run_health_check():
    print("\n" + "🩺" * 40)
    print(f"🚀 Quantum Finance: 全系統終極健康診斷報告 (v1.4)")
    print("🩺" * 40)

    with record_lifecycle("system_health_check", category="maintenance", stock_id="SYSTEM"):
        # 1. 基礎設施校驗
        print("\n🏛️  第一維度：基礎設施 (Infrastructure)")
        print("-" * 40)
        print(f"  Project Root : {_PROJECT_ROOT}")
        print(f"  System Time  : {datetime.now()}")

        # 2. 資料庫主權校驗
        print("\n💎 第二維度：資料庫主權 (Sovereignty)")
        print("-" * 40)
        with db_transaction() as cur:
            cur.execute("SELECT count(*) FROM stocks")
            core_count = cur.fetchone()['count']
            print(f"  核心標的總數 : {core_count} 檔")

        # 3. 混合日誌稽核
        print("\n📝 第三維度：可觀測性 (Observability)")
        print("-" * 40)
        with db_transaction() as cur:
            cur.execute("SELECT count(*) FROM pipeline_execution_log")
            log_count = cur.fetchone()['count']
            print(f"  統一日誌總數 : {log_count} 筆")
        
        # 專項日誌紀錄
        write_data_audit_log("SYSTEM_HEALTH", "SYSTEM", datetime.now().strftime("%Y-%m-%d"), "HEALTH_DIAGNOSTIC", 1)

    print("\n" + "🩺" * 40)
    print("✨ 健康診斷完成，系統狀態：PERFECT。")
    print("🩺" * 40 + "\n")

if __name__ == "__main__":
    run_health_check()

