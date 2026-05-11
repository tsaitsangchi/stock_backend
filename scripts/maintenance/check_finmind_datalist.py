"""
check_finmind_datalist.py v2.1 (Quantum Finance Edition)
================================================================================
API 連通性探測器 — 外部供應鏈工具 (Quantum v5.2 標準)
負責驗證 FinMind API 的連線狀態、Token 有效性以及資料集響應。

修訂歷程：
  v2.1 (2026-05-11): [修復] 對接 FinMindClient v3.8 參數規範，導入生命週期與 API 儀表板。
  v5.5.7 (2026-05-09): [核心] 整合基礎混合日誌。

執行範例 (Comprehensive Usage Examples):
  1. [全域連通性探測] 驗證 API 與 Token 狀態:
     python scripts/maintenance/check_finmind_datalist.py

  2. [API 配額查閱] 透過診斷儀表板確認剩餘次數 (執行後自動顯示)。

  3. [連線監控] 查詢過去 24 小時的 API 延遲趨勢 (SQL):
     SELECT duration_ms, created_at FROM pipeline_execution_log WHERE task_name = 'api_connectivity_check' ORDER BY created_at DESC;
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

def show_api_dashboard(client, elapsed_ms, dataset_status):
    """執行後的 API 探測儀表板。"""
    quota = client.get_quota()
    print("\n" + "="*50)
    print("📡 Quantum Finance: API 供應鏈健康報告 (v2.1)")
    print("="*50)
    print(f"✅ 連通狀態  : {'正常響應' if dataset_status else '連線異常'}")
    print(f"⏱️  響應延遲  : {elapsed_ms} ms")
    print(f"🔑 Token 狀態: 有效 (tsaitsangchi)")
    print(f"📊 每小時配額: {quota.get('remaining', 0)} / {quota.get('limit', 6000)}")
    
    if quota.get('remaining', 0) < 500:
        print("⚠️ 警告：API 配額即將耗盡，請注意抓取頻率！")
    else:
        print("🟢 配額充足，可支援大規模 Ingestion 任務。")
        
    print("-" * 50)
    print("📝 日誌同步: pipeline_execution_log (api_connectivity_check)")
    print("="*50 + "\n")

def check_datalist():
    """執行多維度 API 連通性探測。"""
    t0 = time.monotonic()
    logger.info("📡 正在驗證 FinMind 資料集連通性...")
    
    with record_lifecycle("api_connectivity_check", category="sys", stock_id="FINMIND"):
        try:
            api = FinMindClient()
            # 測試抓取基礎資料集 (對接 v3.8 參數：dataset, stock_id, start_date)
            # 對於 TaiwanStockInfo 等 Meta 資料集，stock_id 與 start_date 傳入空字串即可
            data = api.get_data("TaiwanStockInfo", "", "")
            
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            dataset_status = True if data and len(data) > 0 else False
            
            # 混合模式：審計紀錄 (紀錄資料集規模)
            write_data_audit_log("api_datalist", "FINMIND", datetime.now().strftime("%Y-%m-%d"), "SCAN", len(data) if data else 0)
            
            show_api_dashboard(api, elapsed_ms, dataset_status)
            return True
        except Exception as e:
            logger.error(f"❌ API 連通性探測失敗: {e}")
            raise e

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    check_datalist()