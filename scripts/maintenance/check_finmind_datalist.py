"""
check_finmind_datalist.py v5.5.2 (Trinity Core Final)
================================================================================
資料集可用性監測器 — 混合模式日誌實作版
驗證 FinMind 資料集（如 TaiwanStockPrice）是否正常服務中。
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
    from core.db_utils import write_pipeline_log
    from core.finmind_client import FinMindClient
except ImportError as e:
    print(f"[FATAL] 無法匯入核心配置: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def check_datalist():
    t0 = time.monotonic()
    logger.info("📡 正在驗證 FinMind 資料集連通性...")
    
    api = FinMindClient()
    # 測試抓取少量資料以驗證連通性
    data = api.get_data("TaiwanStockInfo")
    
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    write_pipeline_log("api_datalist_check", "SYSTEM", "success", "sys", elapsed_ms, len(data))
    logger.info(f"✅ 連通性驗證成功，資料集正常響應。")

if __name__ == "__main__":
    check_datalist()