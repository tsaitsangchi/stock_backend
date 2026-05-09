"""
fetch_missing_stocks_data.py v5.5.2 (Trinity Core Final)
================================================================================
缺失標的救援抓取器 — 混合模式日誌實作版
比對 config.py 與資料庫，自動執行空白標底的全量補齊。
"""

import sys
import logging
import time
from pathlib import Path

# ── 系統路徑修復 ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core", "ingestion"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import db_transaction, write_pipeline_log
    from core.finmind_client import FinMindClient
    from config import STOCK_CONFIGS
except ImportError as e:
    print(f"[FATAL] 無法匯入核心配置: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def rescue_missing_stocks():
    t0 = time.monotonic()
    api = FinMindClient()
    rescued_count = 0
    
    try:
        # 1. 找出資料庫中完全沒有資料的 stock_id
        with db_transaction() as cur:
            cur.execute("SELECT DISTINCT stock_id FROM stock_price")
            existing_ids = {r['stock_id'] for r in cur.fetchall()}
            
        config_ids = set(STOCK_CONFIGS.keys())
        missing_ids = config_ids - existing_ids
        
        if not missing_ids:
            logger.info("✅ 所有配置標的皆已有資料。")
        else:
            logger.info(f"🆘 發現 {len(missing_ids)} 個缺失標的，啟動救援...")
            for mid in list(missing_ids)[:5]: # 每次救援 5 支，防止配額耗盡
                data = api.get_data("TaiwanStockPrice", mid, start_date="2020-01-01")
                rescued_count += 1
                
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        write_pipeline_log("rescue_missing", "SYSTEM", "success", "ingestion", elapsed_ms, rescued_count)
        return rescued_count
        
    except Exception as e:
        logger.error(f"❌ 救援行動失敗: {e}")
        write_pipeline_log("rescue_missing", "SYSTEM", "failed", "ingestion", 0, 0, str(e))
        return 0

if __name__ == "__main__":
    rescue_missing_stocks()