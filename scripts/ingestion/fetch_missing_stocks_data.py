"""
fetch_missing_stocks_data.py v6.2 (Trinity Core Final)
================================================================================
資料救援工具 — 混合模式日誌標準版
負責檢測配置中缺失數據的標的，並啟動救援抓取任務。

修訂歷程：
  v6.2 (2026-05-10):
    - [文檔] 補齊「全量救援 (--all)」與「自定義限額 (--limit)」執行範例說明。
  v6.1 (2026-05-10):
    - [功能] 新增 --limit 與 --all 參數，支援自定義救援數量。

【執行範例說明】
1. 救援所有缺失標的 (自動補齊所有數據)：
   $ python scripts/ingestion/fetch_missing_stocks_data.py --all

2. 僅救援前 10 檔缺失標的 (限額模式)：
   $ python scripts/ingestion/fetch_missing_stocks_data.py --limit 10
================================================================================
"""

import sys
import logging
import time
import argparse
from pathlib import Path

# ── 系統路徑修復 (v3.1) ──
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

def rescue_missing_stocks(limit=5):
    """
    掃描資料庫，針對無量價紀錄的核心標的進行緊急抓取。
    
    執行範例：
    $ python scripts/ingestion/fetch_missing_stocks_data.py --all
    $ python scripts/ingestion/fetch_missing_stocks_data.py --limit 20
    """
    t0 = time.monotonic()
    api = FinMindClient()
    rescued_count = 0
    
    try:
        # 1. 找出資料庫中已有量價紀錄的標的
        with db_transaction() as cur:
            cur.execute("SELECT DISTINCT stock_id FROM stock_price")
            existing_ids = {r['stock_id'] for r in cur.fetchall()}
            
        config_ids = set(STOCK_CONFIGS.keys())
        missing_ids = config_ids - existing_ids
        
        if not missing_ids:
            logger.info("✅ 所有配置標的皆已有資料。")
        else:
            total_missing = len(missing_ids)
            fetch_count = total_missing if limit == -1 else min(total_missing, limit)
            logger.info(f"🆘 發現 {total_missing} 個缺失標的，啟動救援 (本次目標 {fetch_count} 檔)...")
            
            # 2. 針對目標標的進行救援
            for mid in list(missing_ids)[:fetch_count]:
                logger.info(f"🚀 正在救援 {mid}...")
                data = api.get_data("TaiwanStockPrice", mid, start_date="2020-01-01")
                if data: rescued_count += 1
                
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        
        # 🔴 混合日誌紀錄 (Category: ingestion)
        write_pipeline_log(
            task_name="rescue_missing",
            stock_id="SYSTEM",
            status="success",
            category="ingestion",
            duration_ms=elapsed_ms,
            rows=rescued_count
        )
        logger.info(f"🏁 救援行動完成！成功補齊 {rescued_count} 檔標的。")
        return rescued_count
        
    except Exception as e:
        logger.error(f"❌ 救援行動失敗: {e}")
        write_pipeline_log(
            task_name="rescue_missing",
            stock_id="SYSTEM",
            status="failed",
            category="ingestion",
            duration_ms=0,
            rows=0,
            err=str(e)
        )
        return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5, help="限制救援標的數量，-1 為全部救援")
    parser.add_argument("--all", action="store_true", help="救援所有缺失標的 (等同 --limit -1)")
    args = parser.parse_args()
    
    limit = -1 if args.all else args.limit
    rescue_missing_stocks(limit=limit)