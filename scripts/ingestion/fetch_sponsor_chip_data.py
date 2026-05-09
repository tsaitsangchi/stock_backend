"""
fetch_sponsor_chip_data.py v5.5.1 (Trinity Core Final)
================================================================================
贊助商/八大行庫籌碼抓取器 — 混合模式日誌實作版
負責同步行庫買賣超至 eight_banks_buy_sell 表。
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

def fetch_sponsor(stock_id: str):
    t0 = time.monotonic()
    api = FinMindClient()
    
    # 🔍 資料表對齊：eight_banks_buy_sell
    last_date = get_latest_date("eight_banks_buy_sell", stock_id) or "2020-01-01"
    
    logger.info(f"🏛️ 正在同步 {stock_id} 八大行庫籌碼 (Since: {last_date})...")
    data = api.get_data("TaiwanStockGovernmentControlsBuySell", stock_id, start_date=last_date)
    
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    
    write_pipeline_log(
        task_name="fetch_sponsor_chip",
        stock_id=stock_id,
        status="success",
        category="ingestion",
        duration_ms=elapsed_ms,
        rows=len(data)
    )
    return len(data)

if __name__ == "__main__":
    fetch_sponsor("2330")