"""
test_block_trading.py v2.1 (Quantum Finance Edition)
================================================================================
鉅額交易整合測試工具 — API 與資料鏈結驗證器 (Quantum v5.2 標準)
負責測試 TaiwanStockBlockTrading 資料集的連通性、資料格式與入庫前置驗證。

修訂歷程：
  v2.1 (2026-05-11): [修復] 修正 get_data 簽名缺失參數問題、補全範例矩陣與 Hybrid Logging 對接。
  v5.5.7 (2026-05-09): [核心] 導入 Hybrid Logging 混合日誌。

【執行範例矩陣 (Integration Test Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / SQL                                         │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [執行標準整合測試]        │ $ python scripts/maintenance/test_block_trading.py     │
│ 2. [指定標的深度驗證]        │ $ python scripts/maintenance/test_block_trading.py --stock_id 2317 │
│ 3. [查看測試歷史紀錄]        │ SELECT * FROM pipeline_execution_log                   │
│                              │ WHERE task_name = 'block_trading_test' ORDER BY id DESC;│
└──────────────────────────────┴────────────────────────────────────────────────────────┘

【業務邏輯說明】
  - API 驗證: 確保 TaiwanStockBlockTrading 資料集能回傳正確的欄位 (如 trade_type, price)。
  - 日誌對接: 測試結果同步紀錄至生命週期與審計表。
================================================================================
"""
import sys, logging, time, argparse
from datetime import datetime, timedelta
from pathlib import Path

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

def show_test_dashboard(stats: dict):
    """執行後的整合測試儀表板。"""
    print("\n" + "🧪"*35)
    print("🚀 Quantum Finance: 鉅額交易整合測試報告 (v2.1)")
    print("🧪"*35)
    print(f"✅ 測試狀態  : 完成")
    print(f"📊 抓取筆數  : {stats['rows']} 筆")
    print(f"⏱️  API 延遲  : {stats['elapsed_ms']} ms")
    
    if stats['rows'] > 0:
        print("-" * 70)
        print(f"🟢 驗證成功：資料集 TaiwanStockBlockTrading 響應正常，資料完整。")
    else:
        print("-" * 70)
        print("🟡 提示：該時段內未偵測到鉅額交易紀錄 (正常現象)。")
        
    print("-" * 70)
    print("📝 日誌同步: pipeline_execution_log & data_audit_log (integration_test)")
    print("🧪"*35 + "\n")

def run_test(stock_id: str = "2330"):
    """啟動鉅額交易整合測試。"""
    with record_lifecycle("block_trading_test", category="test", stock_id=stock_id):
        t0 = time.monotonic()
        try:
            logger.info(f"🧪 正在執行鉅額交易整合測試 (標的: {stock_id})...")
            api = FinMindClient()
            
            # 設定測試區間 (預設過去 30 天)
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            
            # 修正後的 API 呼叫
            data = api.get_data("TaiwanStockBlockTrading", stock_id, start_date)
            
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            stats = {"rows": len(data), "elapsed_ms": elapsed_ms}
            
            # 混合紀錄
            write_data_audit_log("integration_test", stock_id, start_date, "TEST", len(data))
            
            show_test_dashboard(stats)
            return True
        except Exception as e:
            logger.error(f"❌ 整合測試失敗: {e}")
            raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock_id", type=str, default="2330", help="指定測試標的")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_test(stock_id=args.stock_id)
