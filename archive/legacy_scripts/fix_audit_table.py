"""
fix_audit_table.py v2.4 (Quantum Finance Edition)
================================================================================
審計表結構修補工具 — 數據追蹤防線 (Quantum v5.2 標準)
負責檢查並強制補齊 data_audit_log 表格所需的欄位。

修訂歷程：
  v2.4 (2026-05-11): [標準化] 對齊 v3.7 視覺報告標準，加入執行結果與系統資訊。
  v2.3 (2026-05-11): [最佳架構] 接入 core 統一接口。

【執行範例矩陣 (Audit Repair Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令                                               │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [全系統審計表格結構修復]  │ $ python scripts/maintenance/fix_audit_table.py        │
└──────────────────────────────┴────────────────────────────────────────────────────────┘
================================================================================
"""
import sys, logging, platform
from pathlib import Path

# ── 最佳架構引導 ──
try:
    import core
    from core import db_transaction, record_lifecycle, ensure_infrastructure
except ImportError:
    print("[FATAL] 核心架構引導失敗。")
    sys.exit(1)

def show_repair_dashboard(h_count: int):
    status = "SUCCESS" if h_count >= 0 else "FAILED"
    print("\n" + "🛠️"*40)
    print("🚀 Quantum Finance: 審計表結構修復報告 (v2.4)")
    print("🛠️"*40)
    print(f"✅ 執行結果  : {status}")
    print(f"🖥️  操作系統  : {platform.system()} {platform.release()}")
    print(f"🔧 自癒修復  : {max(0, h_count)} 處欄位已同步")
    print("-" * 80)
    print("📝 任務同步: pipeline_execution_log (audit_table_fix)")
    print("🛠️"*40 + "\n")

def repair_audit_table():
    with record_lifecycle("audit_table_fix", category="maintenance", stock_id="DB_SYSTEM"):
        h_count = ensure_infrastructure()
        show_repair_dashboard(h_count)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    repair_audit_table()
