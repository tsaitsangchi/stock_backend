"""
check_finmind_quota.py v1.2 (Quantum Finance Edition)
================================================================================
API 供應鏈守護者 — 配額監控診斷工具 (Quantum v5.2 標準)
負責即時偵測 FinMind API 配額、Token 狀態與通訊品質。

修訂歷程：
  v1.2 (2026-05-11): [標準] 升級至 v5.2 標準，整合 core.FinMindClient 與混合模式日誌。
  v1.1 (2026-05-08): [修復] 修正 Token 讀取邏輯。
  v1.0 (2026-04-25): [首發] 實作基礎 API 配額查詢。

【執行範例矩陣 (Quota Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / 用法                                        │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [即時配額診斷]            │ $ python scripts/maintenance/check_finmind_quota.py    │
└──────────────────────────────┴────────────────────────────────────────────────────────┘
================================================================================
"""
import sys, logging, time, platform
from pathlib import Path

# ── 系統級架構引導 ──
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

try:
    from core import FinMindClient, record_lifecycle
except ImportError as e:
    print(f"[FATAL] 核心架構引導失敗: {e}")
    sys.exit(1)

def run_quota_audit():
    """執行 API 供應鏈稽核"""
    client = FinMindClient()
    
    with record_lifecycle("api_quota_audit", "maintenance", "SYSTEM"):
        q = client.get_quota()
        
        print("\n" + "📡"*40)
        print(f"🚀 Quantum Finance: API 供應鏈診斷報告 (v1.2)")
        print("📡"*40)
        
        status_icon = "🟢" if q.get("diag") == "Success" else "🔴"
        print(f"\n👤 帳號 ID     : {q.get('user_id')}")
        print(f"📧 電子郵件    : {q.get('email')}")
        print(f"📊 每小時上限  : {q.get('limit')} 筆")
        print(f"📉 目前已使用  : {q.get('used')} 筆")
        print(f"🔋 剩餘可用    : {int(q.get('limit', 0)) - int(q.get('used', 0))} 筆")
        print(f"🔍 診斷狀態    : {status_icon} {q.get('diag')}")
        
        if q.get("diag") != "Success":
            print("-" * 80)
            print("⚠️  警報：偵測到通訊異常，請立即檢查 .env 中的 FINMIND_TOKEN！")
            
        print("\n" + "📡"*40 + "\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_quota_audit()