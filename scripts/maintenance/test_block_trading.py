"""
test_block_trading.py v5.5 (Trinity Core Edition)
================================================================================
鉅額交易資料測試工具 — 混合模式日誌實作版
此模組負責驗證 FinMind 的鉅額交易資料可用性，並識別帳戶權限是否足夠。

核心功能：
  · 權限偵測       ─ 自動識別 400 (權限不足) 錯誤。
  · 執行紀錄       ─ 對接 write_pipeline_log，標記為 test_v5.1 (Testing)。

修訂歷程：
  v5.5 (2026-05-09):
    - [規範] 導入混合模式日誌，紀錄於 pipeline_execution_log。
    - [核心] 對接 path_setup v3.0 與 FinMindClient v5.1 標準。
"""

import sys
import logging
import time
from pathlib import Path

# ── 系統路徑修復 (對接 path_setup v3.0) ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import write_pipeline_log
    from core.finmind_client import FinMindClient
except ImportError as e:
    print(f"[FATAL] 無法匯入核心模組: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_test(dataset: str, data_id: str = ""):
    t0 = time.monotonic()
    client = FinMindClient()
    logger.info(f"🚀 正在測試鉅額交易資料集: {dataset}")
    
    try:
        data = client.get_data(dataset, data_id, "2024-03-04", "2024-03-04")
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        status = "success" if data else "no_new_data"
        
        write_pipeline_log(dataset, data_id or "TEST", status, "test", elapsed_ms, len(data) if data else 0)
        logger.info(f"✅ 測試完成：{status}")
    except Exception as e:
        logger.error(f"❌ 測試失敗: {e}")
        write_pipeline_log(dataset, data_id or "TEST", "failed", "test", 0, 0, str(e))

if __name__ == "__main__":
    run_test("TaiwanStockBlockTradingDailyReport")
