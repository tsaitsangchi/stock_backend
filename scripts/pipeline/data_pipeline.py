"""
data_pipeline.py v5.5.10 (Trinity Core Final)
================================================================================
Trinity 全自動資料管線指揮官 — 系統核心思想完整版
編排 150 檔核心標的之全維度抓取 (量價、籌碼、基本面) 與特徵生成。

修訂歷程：
  v5.5.10 (2026-05-09):
    - [核心] 實作「屬性驅動」全自動抓取流程，針對 150 檔標的執行全維度同步。
    - [整合] 依序執行：標的同步 -> 量價抓取 -> 籌碼抓取 -> 宏觀因子 -> 特徵生成。
"""

import sys
import logging
import time
import pandas as pd
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
    from core.db_utils import write_pipeline_log, db_transaction, get_db_stock_ids
    from core.migrate_stocks_config import migrate_core_assets
    from ingestion.parallel_fetch import run_orchestrator
    
    # 載入核心抓取器
    from ingestion.fetch_technical_data import fetch_tech
    from ingestion.fetch_chip_data import fetch_chip
    from ingestion.fetch_advanced_chip_data import fetch_advanced_chip
    from ingestion.fetch_fundamental_data import fetch_fundamental
    from ingestion.fetch_news_data import fetch_news
    
    # 載入特徵與維運
    from features.feature_engineering import build_features
    from monitor.update_daily_status import update_status
except ImportError as e:
    print(f"[FATAL] 無法匯入管線組件: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def process_feature_pipeline(stock_ids: list):
    """
    對目標標的進行特徵生成流水線。
    """
    total_feats = 0
    for sid in stock_ids:
        try:
            with db_transaction() as cur:
                cur.execute("SELECT date, open, max, min, close, trading_volume FROM stock_price WHERE stock_id = %s ORDER BY date DESC LIMIT 200", (sid,))
                rows = cur.fetchall()
                if not rows: continue
                df = pd.DataFrame(rows).sort_values("date")
            
            logger.info(f"🧬 正在為 {sid} 構建特徵矩陣...")
            feat_df = build_features(df, sid, for_inference=True)
            total_feats += len(feat_df)
        except Exception as e:
            logger.error(f"  .. {sid} 特徵生成異常: {e}")
    return total_feats

def run_full_pipeline():
    t_start = time.monotonic()
    logger.info("🚀 [Master] 啟動 Trinity 全維度資料管線 (v5.5.10)...")
    
    try:
        # Phase 0: 配置同步
        logger.info("--- Phase 0: 同步核心標的配置 ---")
        migrate_core_assets()
        
        # 獲取各維度的目標名單
        basic_stocks = get_db_stock_ids(fetch_type='basic')
        chip_stocks = get_db_stock_ids(fetch_type='chip')
        fund_stocks = get_db_stock_ids(fetch_type='fundamental')
        
        logger.info(f"📊 待同步任務概況: Basic({len(basic_stocks)}), Chip({len(chip_stocks)}), Fundamental({len(fund_stocks)})")
        
        # Phase 1: 並行抓取
        logger.info("--- Phase 1: 啟動並行抓取任務 ---")
        
        if basic_stocks:
            run_orchestrator(fetch_tech, basic_stocks, "full_market_tech_sync", workers=4)
        
        if chip_stocks:
            run_orchestrator(fetch_chip, chip_stocks, "full_market_chip_sync", workers=4)
            run_orchestrator(fetch_advanced_chip, chip_stocks, "full_market_margin_sync", workers=4)
            
        if fund_stocks:
            run_orchestrator(fetch_fundamental, fund_stocks, "full_market_fund_sync", workers=2)
            
        # Phase 2: 特徵生成
        logger.info("--- Phase 2: 啟動特徵生成流水線 ---")
        total_feats = process_feature_pipeline(basic_stocks) # 先針對前 20 檔核心產生特徵作為示範
        
        # Phase 3: 維運狀態同步
        logger.info("--- Phase 3: 執行系統狀態同步 ---")
        update_status()
        
        elapsed_sec = round(time.monotonic() - t_start, 2)
        write_pipeline_log("full_pipeline_master", "SYSTEM", "success", "sys", int(elapsed_sec*1000), total_feats)
        logger.info(f"🏆 [Master] 管線全維度執行完畢，耗時: {elapsed_sec}s")
        
    except Exception as e:
        logger.error(f"❌ 管線總指揮崩潰: {e}")
        write_pipeline_log("full_pipeline_master", "SYSTEM", "failed", "sys", 0, 0, str(e))

if __name__ == "__main__":
    run_full_pipeline()
