"""
check_finmind_quota.py v1.3 (Quantum Finance Edition)
================================================================================
配額監控哨兵 — API 供應鏈預警工具 (Quantum v5.2 標準)

修訂歷程：
  v1.3 (2026-05-11): [標準] 升級至 v5.2 標準，對齊 FinMindClient v4.19 自檢邏輯。

【執行範例矩陣 (Quota Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / 用法                                        │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [標準配額狀態查詢]        │ $ python scripts/maintenance/check_finmind_quota.py    │
└──────────────────────────────┴────────────────────────────────────────────────────────┘
================================================================================
"""
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

try:
    from core import FinMindClient, record_lifecycle
except ImportError as e:
    print(f"[FATAL] 核心架構引導失敗: {e}")
    sys.exit(1)

def run_quota_check():
    client = FinMindClient()
    with record_lifecycle("api_quota_check", "maintenance", "SYSTEM"):
        print("\n" + "💎"*40)
        print(f"🚀 Quantum Finance: API 供應鏈配額報告 (v1.3)")
        print("💎"*40)
        
        info = client.get_quota()
        print(f"\n👤 帳號 (ID) : {info.get('user_id')}")
        print(f"📧 電子郵件  : {info.get('email')}")
        print(f"📊 使用情況  : {info.get('used')} / {info.get('limit')}")
        print(f"🛡️  診斷結果  : {info.get('diag')}")
        print("\n" + "💎"*40 + "\n")

if __name__ == "__main__":
    run_quota_check()