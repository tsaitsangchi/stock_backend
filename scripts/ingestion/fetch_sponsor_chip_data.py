"""
fetch_sponsor_chip_data.py v5.5 (Trinity Core Edition)
================================================================================
贊助商/大戶籌碼資料抓取器 — 混合模式日誌實作版
此模組抓取「八大行庫買賣超」、「大戶持股比例」等關鍵籌碼因子。
"""

import sys
import logging
import time
from pathlib import Path

# ── 系統路徑修復 (對接 path_setup v3.0) ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core", "ingestion"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import write_pipeline_log
    from core.finmind_client import FinMindClient
except ImportError as e:
    print(f"[FATAL] 無法匯入核心配置: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def fetch_sponsor(stock_id: str):
    t0 = time.monotonic()
    api = FinMindClient()
    logger.info(f"🏛️ 正在抓取 {stock_id} 大戶與行庫籌碼...")
    
    data = api.get_data("TaiwanStockGovernmentControlsBuySell", stock_id)
    
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    
    # 🔴 混合日誌紀錄 (Category: ingestion)
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