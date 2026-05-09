"""
fetch_fred_data.py v5.5.7 (Trinity Core Final)
================================================================================
FRED 經濟數據抓取器 — 混合模式日誌實作版
負責將聖路易斯聯準會 (FRED) 的指標同步至 fred_series 表。

修訂歷程：
  v5.5.7 (2026-05-09):
    - [修復] 修正資料表欄位對齊 (series_id)。
    - [文檔] 補齊極致詳細的「並行調度」與「單點調試」範例。
  v5.5.1 (2026-05-09):
    - [規範] 導入混合模式日誌。

【執行範例說明】

1. 手動單點調試 (抓取 10 年期美債殖利率 DGS10)：
   $ python scripts/ingestion/fetch_fred_data.py

2. 大規模並行抓取 (同時抓取多項總經指標)：
   ------------------------------------------------------------
   from ingestion.parallel_fetch import run_orchestrator
   from ingestion.fetch_fred_data import fetch_fred
   
   series_list = ["DGS10", "DGS2", "UNRATE", "CPIAUCSL"]
   run_orchestrator(fetch_fred, series_list, "macro_fred_sync")
   ------------------------------------------------------------
"""

import sys
import logging
import time
from pathlib import Path

# ── 系統路徑修復 (v3.0) ──
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

def fetch_fred(series_id: str = "DGS10"):
    """
    抓取 FRED 經濟數據。
    預設抓取 DGS10 (10-Year Treasury Constant Maturity Rate)。
    """
    t0 = time.monotonic()
    api = FinMindClient()
    
    # 🔍 資料表對齊：fred_series (ID 欄位為 series_id)
    last_date = get_latest_date("fred_series", series_id, id_column="series_id") or "2010-01-01"
    
    logger.info(f"🇺🇸 正在同步 FRED 指標: {series_id} (Since: {last_date})...")
    data = api.get_data("FedFundsRate", series_id, start_date=last_date) # 註：FinMind 中 FRED 統一使用 FedFundsRate 介面
    
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    
    write_pipeline_log(
        task_name="fetch_fred",
        stock_id=series_id,
        status="success",
        category="ingestion",
        duration_ms=elapsed_ms,
        rows=len(data)
    )
    return len(data)

if __name__ == "__main__":
    # 單點測試預設抓取 10 年期美債殖利率
    fetch_fred("DGS10")