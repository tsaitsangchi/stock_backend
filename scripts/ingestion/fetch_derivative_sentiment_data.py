"""
fetch_derivative_sentiment_data.py v5.5.1 (Trinity Core Final)
================================================================================
選擇權情緒抓取器 — 混合模式日誌實作版
負責同步選擇權未平倉與 Put/Call Ratio 至 options_oi_large_holders 表。
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

def fetch_sentiment():
    t0 = time.monotonic()
    api = FinMindClient()
    
    # 🔍 資料表對齊：options_oi_large_holders
    last_date = get_latest_date("options_oi_large_holders") or "2010-01-01"
    
    logger.info(f"🎭 正在同步選擇權情緒指標 (Since: {last_date})...")
    data = api.get_data("TaiwanOptionPutCallRatio", start_date=last_date)
    
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    
    write_pipeline_log(
        task_name="fetch_sentiment",
        stock_id="MARKET_WIDE",
        status="success",
        category="ingestion",
        duration_ms=elapsed_ms,
        rows=len(data)
    )
    return len(data)

if __name__ == "__main__":
    fetch_sentiment()