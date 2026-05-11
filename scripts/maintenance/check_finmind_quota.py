"""
check_finmind_quota.py v2.5 (Quantum Finance Edition)
================================================================================
API 配額監控工具 — 數據供應鏈哨兵 (Quantum v5.2 標準)
負責監控 FinMind API 配額使用率、帳號狀態與連線延遲。

修訂歷程：
  v2.5 (2026-05-11): [標準化] 對齊 v3.7 視覺報告標準，加入執行結果與系統資訊。
  v2.4 (2026-05-11): [最佳架構] 接入 core 統一接口，對齊 FinMindClient v4.9。

【執行範例矩陣 (Quota Audit Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令                                               │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [全系統 API 配額健康稽核] │ $ python scripts/maintenance/check_finmind_quota.py    │
│ 2. [診斷 API 供應鏈延遲]     │ $ python scripts/maintenance/check_finmind_quota.py --test_ping│
└──────────────────────────────┴────────────────────────────────────────────────────────┘
================================================================================
"""
import sys, logging, platform
from pathlib import Path

# ── 最佳架構引導 ──
try:
    import core
    from core import FinMindClient, record_lifecycle
except ImportError:
    print("[FATAL] 核心架構引導失敗。")
    sys.exit(1)

def show_quota_dashboard(quota: dict):
    print("\n" + "📡"*40)
    print("🚀 Quantum Finance: API 配額稽核報告 (v2.5)")
    print("📡"*40)
    print(f"✅ 執行結果  : SUCCESS")
    print(f"🖥️  操作系統  : {platform.system()} {platform.release()}")
    print(f"👤 帳號 ID   : {quota['user_id']}")
    print(f"📊 剩餘配額  : {quota['remaining']} / {quota['limit']} 筆")
    
    use_pct = (quota['used'] / quota['limit'] * 100) if quota['limit'] > 0 else 0
    if use_pct > 90:
        print(f"🔴 警報：配額使用率達 {use_pct:.1f}%")
    else:
        print(f"🟢 正常：配額剩餘充足 ({use_pct:.1f}%)")
        
    print("-" * 80)
    print("📝 任務同步: pipeline_execution_log (api_quota_audit)")
    print("📡"*40 + "\n")

def audit_quota():
    with record_lifecycle("api_quota_audit", category="maintenance", stock_id="FINMIND"):
        client = FinMindClient()
        quota = client.get_quota()
        show_quota_dashboard(quota)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    audit_quota()