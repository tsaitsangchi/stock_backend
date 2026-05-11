"""
check_finmind_datalist.py v3.1 (Quantum Finance Edition)
================================================================================
連通性哨兵 — API 通訊品質與延遲稽核工具 (Quantum v5.2 標準)

修訂歷程：
  v3.1 (2026-05-11): [標準] 升級至 v5.2 標準，補全延遲稽核與配額對齊範例。
  v3.0 (2026-05-11): [重構] 轉型為通訊品質哨兵，實作 check_latency。

【執行範例矩陣 (Connectivity Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / 用法                                        │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [即時 API 品質稽核]       │ $ python scripts/maintenance/check_finmind_datalist.py │
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

def run_connectivity_audit():
    client = FinMindClient()
    with record_lifecycle("api_connectivity_check", "maintenance", "SYSTEM"):
        print("\n" + "📡"*40)
        print(f"🚀 Quantum Finance: API 連通性與品質報告 (v3.1)")
        print("📡"*40)
        
        latency = client.check_latency()
        quota = client.get_quota()
        status_icon = "🟢" if latency != -1 else "🔴"
        
        print(f"\n✅ 連通狀態  : {status_icon} {'正常' if latency != -1 else '失敗'}")
        print(f"⏱️  響應延遲  : {latency if latency != -1 else 'N/A'} ms")
        print(f"👤 帳號 (ID) : {quota.get('user_id')}")
        print(f"📊 每小時配額: {quota.get('used')} / {quota.get('limit')}")
        
        print("\n" + "📡"*40 + "\n")

if __name__ == "__main__":
    run_connectivity_audit()