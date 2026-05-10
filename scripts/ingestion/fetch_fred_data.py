"""
fetch_fred_data.py v6.0 (Trinity Core Final)
================================================================================
FRED 經濟數據抓取器 — 混合模式日誌標準版
負責將聖路易斯聯準會 (FRED) 的指標同步至 fred_series 表。

修訂歷程：
  v6.0 (2026-05-10):
    - [核心] 升級至 Trinity Core v6.0 標準，確保 fred_series 表與標準對齊。
  v5.5.7 (2026-05-09):
    - [文檔] 補齊「大規模並行調度」與「手動單點調試」執行範例說明。

【執行範例說明】
1. 手動抓取特定 FRED 指標 (例如 十年期美債收益率 DGS10)：
   $ python scripts/ingestion/fetch_fred_data.py --series_id DGS10
================================================================================
"""

import sys
import logging
import time
from pathlib import Path

# ── 系統路徑修復 (v3.1) ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core", "pipeline"):
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

def fetch_fred(series_id: str = "DGS10"):
    """
    抓取 FRED 經濟數據。
    預設抓取 DGS10 (10-Year Treasury Constant Maturity Rate)。
    
    執行範例：
    $ python scripts/ingestion/fetch_fred_data.py --series_id DGS10
    """
    t0 = time.monotonic()
    api = FinMindClient()
    
    # 🔍 資料表對齊：fred_series (ID 欄位為 series_id)
    last_date = get_latest_date("fred_series", series_id, id_column="series_id") or "2010-01-01"
    
    logger.info(f"🇺🇸 正在同步 FRED 指標: {series_id} (Since: {last_date})...")
    data = api.get_data("FedFundsRate", series_id, start_date=last_date)
    
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    
    # 🔴 混合日誌紀錄 (Category: ingestion)
    write_pipeline_log(
        task_name="fetch_fred",
        stock_id=series_id,
        status="success" if data is not None else "failed",
        category="ingestion",
        duration_ms=elapsed_ms,
        rows=len(data)
    )
    return len(data)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--series_id", type=str, default="DGS10")
    args = parser.parse_args()
    fetch_fred(args.series_id)