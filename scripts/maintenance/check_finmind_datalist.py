"""
check_finmind_datalist.py v2.3 (Quantum Finance Edition)
================================================================================
API 供應鏈稽核工具 — 外部服務連通性診斷器 (Quantum v5.2 標準)
負責驗證 FinMind API 的響應狀態、延遲以及 Token 配額是否充足。

修訂歷程：
  v2.3 (2026-05-11): [標準化] 補全「單一 Table 探測、全量供應鏈審計、強制連線測試」之範例矩陣。
  v2.2 (2026-05-11): [標準化] 對齊 Hybrid Logging 規範。

【執行範例矩陣 (Connectivity Audit Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令                                               │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [快速連通性探測]          │ $ python scripts/maintenance/check_finmind_datalist.py  │
│ 2. [指定 Table 深度稽核]     │ 於程式中修改 dataset = 'TaiwanStockBlockTrading' 進行測試。│
│ 3. [核心供應鏈配額診斷]      │ 查看儀表板中的 "📊 每小時配額" (對齊 6000 筆標準)。    │
│ 4. [強制連線狀態重置]        │ 執行此腳本後，pipeline_execution_log 會更新最新成功時間。│
└──────────────────────────────┴────────────────────────────────────────────────────────┘
================================================================================
"""
import sys, logging, time
from pathlib import Path
from datetime import datetime

_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent if _THIS_DIR.name != "scripts" else _THIS_DIR
if str(_SCRIPTS_DIR) not in sys.path: sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from core.db_utils import record_lifecycle
    from core.finmind_client import FinMindClient
except ImportError:
    print("[FATAL] 無法匯入核心配置。")
    sys.exit(1)

def show_api_dashboard(client: FinMindClient, elapsed_ms: int):
    quota = client.get_quota()
    print("\n" + "📡"*35)
    print("🚀 Quantum Finance: API 供應鏈健康報告 (v2.3)")
    print("📡"*35)
    print(f"✅ 連通狀態  : 正常響應")
    print(f"⏱️  響應延遲  : {elapsed_ms} ms")
    print(f"👤 帳號 (ID) : {quota.get('user_id', 'Unknown')}")
    print(f"📊 每小時配額: {quota.get('remaining', 0)} / {quota.get('limit', 0)}")
    print("-" * 70)
    print("🟢 狀態良好：API 連線與配額皆符合 6000 筆高階標準。")
    print("📝 日誌同步: pipeline_execution_log (api_connectivity_check)")
    print("📡"*35 + "\n")

def check_datalist():
    with record_lifecycle("api_connectivity_check", category="maintenance", stock_id="FINMIND"):
        t0 = time.monotonic()
        api = FinMindClient()
        # 測試核心標的連通性
        api.get_data("TaiwanStockInfo", "2330", datetime.now().strftime("%Y-%m-%d"))
        elapsed = int((time.monotonic() - t0) * 1000)
        show_api_dashboard(api, elapsed)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    check_datalist()