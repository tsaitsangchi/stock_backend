"""
check_finmind_quota.py v2.3 (Quantum Finance Edition)
================================================================================
API 配額審計器 — 外部資源監控工具 (Quantum v5.2 標準)
負責監測 FinMind API 的即時配額使用量、剩餘次數與帳號限制。

修訂歷程：
  v2.3 (2026-05-11): [標準化] 補全維運範例矩陣、優化儀表板視覺效果與 Hybrid Logging 深度對接。
  v2.2 (2026-05-11): [核心] 導入 record_lifecycle 與 Client v3.9 對接。

【執行範例矩陣 (Quota Audit Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / SQL                                         │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [即時配額診斷]            │ $ python scripts/maintenance/check_finmind_quota.py    │
│ 2. [查看配額使用趨勢]        │ SELECT rows_affected, created_at FROM data_audit_log   │
│                              │ WHERE table_name = 'api_quota_usage' ORDER BY id DESC; │
│ 3. [任務生命週期稽核]        │ SELECT status, duration_ms FROM pipeline_execution_log │
│                              │ WHERE task_name = 'api_quota_audit' ORDER BY id DESC;  │
└──────────────────────────────┴────────────────────────────────────────────────────────┘

【業務邏輯說明】
  - 數據來源: 直接調用 FinMindClient.get_quota()。
  - 預警機制: 使用率 > 90% 時主動輸出紅色警報。
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
    from core.db_utils import record_lifecycle, write_data_audit_log
    from core.finmind_client import FinMindClient
except ImportError as e:
    print(f"[FATAL] 無法匯入核心配置: {e}", file=sys.stderr)
    sys.exit(1)

logger = logging.getLogger(__name__)

def show_quota_dashboard(quota: dict):
    """執行後的配額審計儀表板。"""
    remaining = quota.get('remaining', 0)
    used = quota.get('used', 0)
    limit = quota.get('limit', 6000)
    usage_rate = (used / limit * 100) if limit > 0 else 0
    
    print("\n" + "🎟️"*35)
    print("🚀 Quantum Finance: FinMind 配額審計報告 (v2.3)")
    print("🎟️"*35)
    print(f"✅ 審計狀態  : 執行完成")
    print(f"📊 已使用次數: {used} 筆")
    print(f"📉 剩餘配額  : {remaining} 筆")
    print(f"🔥 當前使用率: {usage_rate:.1f}%")
    
    if usage_rate > 90:
        print("-" * 70)
        print("🔴 警報：API 配額已接近上限，請立即停止非必要抓取！")
    elif usage_rate > 70:
        print("-" * 70)
        print("🟡 警告：配額消耗較快，建議減緩抓取速度。")
    else:
        print("-" * 70)
        print("🟢 狀態良好：配額充足，可執行大規模 Ingestion 任務。")
        
    print("-" * 70)
    print("📝 日誌同步: pipeline_execution_log & data_audit_log (api_quota_usage)")
    print("🎟️"*35 + "\n")

def check_quota():
    """執行真實 API 配額審計並紀錄日誌。"""
    with record_lifecycle("api_quota_audit", category="sys", stock_id="FINMIND"):
        try:
            client = FinMindClient()
            quota = client.get_quota()
            
            # 分類紀錄：將配額使用量紀錄到 data_audit_log
            write_data_audit_log("api_quota_usage", "FINMIND", 
                                 datetime.now().strftime("%Y-%m-%d"), 
                                 "AUDIT", quota['used'])
            
            show_quota_dashboard(quota)
            return True
        except Exception as e:
            logger.error(f"❌ 配額審計任務執行失敗: {e}")
            raise e

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    check_quota()