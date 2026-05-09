"""
fetch_derivative_data.py v5.5.1 (Trinity Core Final)
================================================================================
資料抓取模組 — 混合模式日誌實作版
對應資料庫實體表並支援並行調度。

修訂歷程：
  v5.5.1 (2026-05-09):
    - [規範] 導入混合模式日誌。
    - [核心] 對接 v5.5 並行調度規範與路徑修復 v3.0。

執行範例：
  python scripts/ingestion/fetch_derivative_data.py
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

def fetch_derivative():
    t0 = time.monotonic()
    api = FinMindClient()
    
    # 🔍 資料表對齊：futures_ohlcv
    last_date = get_latest_date("futures_ohlcv", "TX") or "2010-01-01"
    
    logger.info(f"🎭 正在同步期貨資料 (Since: {last_date})...")
    data = api.get_data("TaiwanFuturesDaily", "TX", start_date=last_date)
    
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    
    write_pipeline_log(
        task_name="fetch_derivative",
        stock_id="TX",
        status="success",
        category="ingestion",
        duration_ms=elapsed_ms,
        rows=len(data)
    )
    return len(data)

if __name__ == "__main__":
    fetch_derivative()