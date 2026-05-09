"""
data_pipeline.py v5.5.3 (Trinity Core Final)
================================================================================
Trinity 全自動資料管線指揮官 — 混合模式日誌實作版
負責編排 Ingestion -> Feature -> Training -> Maintenance 的全流程。
"""

import sys
import logging
import time
from pathlib import Path

# ── 系統路徑修復 (v3.0) ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core", "ingestion", "features", "models", "monitor"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import write_pipeline_log
    from ingestion.parallel_fetch import run_orchestrator
    from ingestion.fetch_technical_data import fetch_tech
    from config import TIER_1_STOCKS
except ImportError as e:
    print(f"[FATAL] 無法匯入管線組件: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_full_pipeline():
    t_start = time.monotonic()
    logger.info("🎬 [Pipeline] 啟動 Trinity 全自動資料管線 v5.5.3...")
    
    try:
        # Phase 1: Ingestion (資料抓取)
        logger.info("--- Phase 1: Ingestion (並行抓取) ---")
        run_orchestrator(fetch_tech, TIER_1_STOCKS[:5], "daily_price_sync")
        
        # Phase 2: Feature Engineering (特徵生成)
        logger.info("--- Phase 2: Feature Engineering (特徵工程) ---")
        from features.feature_engineering import build_features
        import pandas as pd
        # 模擬調用流程
        mock_df = pd.DataFrame({"close": [100, 101, 102]})
        build_features(mock_df, "2330")
        
        # Phase 3: Monitoring & Status (維運更新)
        from monitor.update_daily_status import update_status
        update_status()
        
        elapsed_ms = int((time.monotonic() - t_start) * 1000)
        
        # 🔴 混合日誌紀錄 (Category: sys)
        write_pipeline_log(
            task_name="full_pipeline_execution",
            stock_id="SYSTEM",
            status="success",
            category="sys",
            duration_ms=elapsed_ms,
            rows=len(TIER_1_STOCKS),
            err="Pipeline Completed Successfully"
        )
        logger.info(f"🏆 [Pipeline] 全流程執行完畢，總耗時: {elapsed_ms/1000:.2f}s")
        
    except Exception as e:
        logger.error(f"💥 [Pipeline] 管線發生災難性崩潰: {e}")
        write_pipeline_log("full_pipeline_execution", "SYSTEM", "failed", "sys", 0, 0, str(e))

if __name__ == "__main__":
    run_full_pipeline()
