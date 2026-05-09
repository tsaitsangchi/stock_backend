"""
fetch_fred_data.py v5.5.1 (Trinity Core Final)
================================================================================
FRED 宏觀經濟抓取器 — 混合模式日誌實作版
負責同步 FRED 系列指標至 fred_series 表。
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

def fetch_fred():
    t0 = time.monotonic()
    api = FinMindClient()
    
    # 🔍 資料表對齊：fred_series
    last_date = get_latest_date("fred_series", "DGS10") or "2010-01-01"
    
    logger.info(f"🇺🇸 正在同步 FRED 宏觀經濟指標 (Series: DGS10, Since: {last_date})...")
    data = api.get_data("US_Treasury_Yield_10Y", start_date=last_date)
    
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    
    write_pipeline_log(
        task_name="fetch_fred",
        stock_id="USA_MACRO",
        status="success",
        category="ingestion",
        duration_ms=elapsed_ms,
        rows=len(data)
    )
    return len(data)

if __name__ == "__main__":
    fetch_fred()