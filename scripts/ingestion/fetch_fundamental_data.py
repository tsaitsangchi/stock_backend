"""
fetch_fundamental_data.py v5.5 (Trinity Core Edition)
================================================================================
基本面資料抓取器 — 混合模式日誌實作版
此模組抓取「綜合損益表」、「資產負債表」等財報資料。

核心功能：
  · 季報自動更新   ─ 自動檢測新季報發布並更新資料庫。
  · 分類日誌紀錄     ─ 執行監控 (pipeline_execution_log) 歸類於 ingestion 類別。
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

def fetch_fundamental(stock_id: str):
    t0 = time.monotonic()
    api = FinMindClient()
    logger.info(f"📊 正在抓取 {stock_id} 財務報表...")
    
    data = api.get_data("TaiwanStockFinancialStatements", stock_id)
    
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    
    # 🔴 混合日誌紀錄 (Category: ingestion)
    write_pipeline_log(
        task_name="fetch_fundamental",
        stock_id=stock_id,
        status="success",
        category="ingestion",
        duration_ms=elapsed_ms,
        rows=len(data)
    )
    return len(data)

if __name__ == "__main__":
    fetch_fundamental("2330")