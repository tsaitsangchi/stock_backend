"""
check_finmind_quota.py v2.2 (Quantum Finance Edition)
================================================================================
API 配額審計器 — 外部資源監控工具 (Quantum v5.2 標準)
負責監測 FinMind API 的即時配額使用量、剩餘次數與帳號限制。

修訂歷程：
  v2.2 (2026-05-11): [標準化] 導入 record_lifecycle、配額診斷儀表板並對接 Client v3.9。
  v5.5.7 (2026-05-09): [核心] 整合基礎混合日誌。

執行範例 (Comprehensive Usage Examples):
  1. [即時配額檢查] 查看目前的 API 消耗狀態:
     python scripts/maintenance/check_finmind_quota.py

  2. [消耗趨勢監控] 查看過去 24 小時的 API 使用紀錄 (SQL):
     SELECT rows_processed as used_count, created_at FROM pipeline_execution_log 
     WHERE task_name = 'api_quota_audit' ORDER BY created_at DESC;

  3. [資產審計] 將配額消耗情況寫入審計日誌 (SQL):
     SELECT * FROM data_audit_log WHERE table_name = 'api_quota_usage' ORDER BY created_at DESC;
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
    
    print("\n" + "="*50)
    print("🎟️  Quantum Finance: FinMind 配額審計報告 (v2.2)")
    print("="*50)
    print(f"✅ 審計狀態  : 執行完成")
    print(f"📊 使用量 (Used)  : {used} 筆")
    print(f"📉 剩餘量 (Left)  : {remaining} 筆")
    print(f"🔥 使用率 (Usage) : {usage_rate:.1f}%")
    
    if usage_rate > 90:
        print("🔴 警報：API 配額已接近上限，請立即停止非必要抓取！")
    elif usage_rate > 70:
        print("🟡 警告：配額消耗較快，建議減緩抓取速度。")
    else:
        print("🟢 狀態：配額充足，系統運行環境優良。")
        
    print("-" * 50)
    print("📝 日誌同步: pipeline_execution_log & data_audit_log")
    print("="*50 + "\n")

def check_quota():
    """執行真實 API 配額審計。"""
    with record_lifecycle("api_quota_audit", category="sys", stock_id="FINMIND"):
        try:
            client = FinMindClient()
            quota = client.get_quota()
            
            # 混合模式：審計紀錄 (紀錄已使用次數)
            write_data_audit_log("api_quota_usage", "FINMIND", datetime.now().strftime("%Y-%m-%d"), "SCAN", quota['used'])
            
            show_quota_dashboard(quota)
            return True
        except Exception as e:
            logger.error(f"❌ 配額審計失敗: {e}")
            raise e

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    check_quota()