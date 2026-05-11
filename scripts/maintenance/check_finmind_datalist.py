"""
check_finmind_datalist.py v2.2 (Quantum Finance Edition)
================================================================================
API 供應鏈稽核工具 — 外部服務連通性診斷器 (Quantum v5.2 標準)
負責驗證 FinMind API 的響應狀態、延遲以及 Token 配額是否充足。

修訂歷程：
  v2.2 (2026-05-11): [標準化] 補全供應鏈稽核範例矩陣、對齊 Hybrid Logging 規範。
  v2.1 (2026-05-11): [修復] 修正 Client 參數調用錯誤，導入連通性診斷儀表板。

【執行範例矩陣 (API Audit Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令                                               │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [快速連通性探測]          │ $ python scripts/maintenance/check_finmind_datalist.py  │
│ 2. [深度供應鏈審計]          │ 查看儀表板中的 "⏱️ 響應延遲" 與 "📊 每小時配額"。      │
│ 3. [查看歷史 API 健康紀錄]   │ SELECT * FROM pipeline_execution_log WHERE task_name = 'api_connectivity_check'; │
└──────────────────────────────┴────────────────────────────────────────────────────────┘

【業務邏輯說明】
  - 核心檢測: 使用 TaiwanStockInfo 資料集作為連通性樣本。
  - 認證檢查: 自動驗證目前 .env 中的 FINMIND_TOKEN 是否有效。
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

def show_api_dashboard(client: FinMindClient, elapsed_ms: int):
    """執行後的 API 供應鏈健康儀表板。"""
    quota = client.get_quota()
    print("\n" + "="*50)
    print("📡 Quantum Finance: API 供應鏈健康報告 (v2.2)")
    print("="*50)
    print(f"✅ 連通狀態  : 正常響應")
    print(f"⏱️  響應延遲  : {elapsed_ms} ms")
    print(f"🔑 Token 狀態: 有效 ({quota.get('user_id', 'Unknown')})")
    print(f"📊 每小時配額: {quota.get('remaining', 0)} / {quota.get('limit', 0)}")
    
    if elapsed_ms > 1000:
        print("🟡 警告：API 響應較慢，可能存在網路抖動。")
    elif quota.get('remaining', 0) < 500:
        print("🔴 警報：可用配額極低，請考慮升級 API 套餐！")
    else:
        print("🟢 配額充足，可支援大規模 Ingestion 任務。")
        
    print("-" * 50)
    print("📝 日誌同步: pipeline_execution_log (api_connectivity_check)")
    print("="*50 + "\n")

def check_datalist():
    """執行 API 連通性探測。"""
    with record_lifecycle("api_connectivity_check", category="maintenance", stock_id="FINMIND"):
        t0 = time.monotonic()
        try:
            logger.info("📡 正在驗證 FinMind 資料集連通性...")
            api = FinMindClient()
            # 使用最輕量的資料集進行探測
            data = api.get_data("TaiwanStockInfo", "2330", datetime.now().strftime("%Y-%m-%d"))
            
            elapsed = int((time.monotonic() - t0) * 1000)
            show_api_dashboard(api, elapsed)
            return True
        except Exception as e:
            logger.error(f"❌ API 連通性探測失敗: {e}")
            raise e

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    check_datalist()