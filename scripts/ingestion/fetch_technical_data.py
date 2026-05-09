"""
fetch_technical_data.py v5.5.1 (Trinity Core Final)
================================================================================
量價資料抓取器 — 混合模式日誌實作版
負責同步 TaiwanStockPrice 資料至資料庫的 stock_price 表。

修訂歷程：
  v5.5.1 (2026-05-09):
    - [規範] 導入混合模式日誌。
    - [核心] 資料表對齊：由 daily_prices 遷移至 stock_price。

執行範例：
  python scripts/ingestion/fetch_technical_data.py
"""

import sys
import logging
import time
from pathlib import Path

# ── 系統路徑修復 ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import write_pipeline_log, get_latest_date
    from core.finmind_client import FinMindClient
except ImportError as e:
    print(f"[FATAL] 無法匯入核心配置: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def fetch_tech(stock_id: str):
    t0 = time.monotonic()
    api = FinMindClient()
    
    # 🔍 資料表對齊：stock_price
    last_date = get_latest_date("stock_price", stock_id) or "2010-01-01"
    
    logger.info(f"📈 正在抓取 {stock_id} 量價資料 (Since: {last_date})...")
    data = api.get_data("TaiwanStockPrice", stock_id, start_date=last_date)
    
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    
    write_pipeline_log(
        task_name="fetch_technical",
        stock_id=stock_id,
        status="success",
        category="ingestion",
        duration_ms=elapsed_ms,
        rows=len(data)
    )
    return len(data)

if __name__ == "__main__":
    fetch_tech("2330")