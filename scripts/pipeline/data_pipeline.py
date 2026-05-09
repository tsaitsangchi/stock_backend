"""
data_pipeline.py v5.5.7 (Trinity Core Final)
================================================================================
Trinity 全自動資料管線指揮官 — 生產環境正式版
編排 Ingestion -> Feature -> Maintenance 的端到端資料生命週期。

修訂歷程：
  v5.5.7 (2026-05-09):
    - [文檔] 補齊極致詳細的「全自動管線生命週期」執行範例。
  v5.5.4 (2026-05-09):
    - [整合] 達成端到端真實資料流 (SQL -> DataFrame -> Features)。

【執行範例說明】

1. 直接從命令行啟動全自動日終工作 (EOD Pipeline)：
   $ python scripts/pipeline/data_pipeline.py
   (此指令會自動執行：並行抓取 -> 特徵計算 -> 系統狀態同步)

2. 在 Linux Crontab 中排程定時執行 (每日晚上 6 點啟動)：
   00 18 * * 1-5 /home/hugo/project/stock_backend/venv/bin/python /home/hugo/project/stock_backend/scripts/pipeline/data_pipeline.py >> /home/hugo/project/stock_backend/logs/pipeline.log 2>&1

3. 在其他 Python 腳本中作為「總管線」調用：
   ------------------------------------------------------------
   from pipeline.data_pipeline import run_full_pipeline
   
   # 執行一次完整的「資料抓取 + 特徵生成 + 系統同步」循環
   run_full_pipeline()
   ------------------------------------------------------------

4. 管線日誌檢查：
   執行後可至資料庫執行以下查詢，確認生命週期各階段狀態：
   SELECT * FROM pipeline_execution_log WHERE task_name = 'full_pipeline_master' ORDER BY created_at DESC;
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
    from core.db_utils import write_pipeline_log, db_transaction
    from ingestion.parallel_fetch import run_orchestrator
    from ingestion.fetch_technical_data import fetch_tech
    from features.feature_engineering import build_features
    from monitor.update_daily_status import update_status
    from config import TIER_1_STOCKS
except ImportError as e:
    print(f"[FATAL] 無法匯入管線組件: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def process_single_stock_pipeline(stock_id: str):
    """
    單一標的之特徵與模型流水線。
    """
    try:
        # 1. 從資料庫讀取最新量價
        with db_transaction() as cur:
            cur.execute("SELECT date, open, max, min, close, trading_volume FROM stock_price WHERE stock_id = %s ORDER BY date DESC LIMIT 200", (stock_id,))
            rows = cur.fetchall()
            if not rows: return 0
            df = pd.DataFrame(rows)
            df = df.sort_values("date")
        
        # 2. 構建特徵
        feat_df = build_features(df, stock_id, for_inference=True)
        return len(feat_df)
    except Exception as e:
        logger.error(f"  .. {stock_id} 流水線異常: {e}")
        return 0

def run_full_pipeline():
    t_start = time.monotonic()
    logger.info("🚀 [Master] 啟動 Trinity 生產級資料管線 v5.5.4...")
    
    try:
        # Phase 1: 並行資料抓取 (更新 stock_price 表)
        logger.info("--- Phase 1: 啟動並行抓取任務 ---")
        run_orchestrator(fetch_tech, TIER_1_STOCKS[:10], "daily_sync")
        
        # Phase 2: 特徵生成流水線 (端到端資料流)
        logger.info("--- Phase 2: 啟動特徵生成流水線 ---")
        total_feats = 0
        for sid in TIER_1_STOCKS[:5]:
            total_feats += process_single_stock_pipeline(sid)
        
        # Phase 3: 維運狀態同步
        logger.info("--- Phase 3: 執行系統狀態同步 ---")
        update_status()
        
        elapsed_ms = int((time.monotonic() - t_start) * 1000)
        write_pipeline_log("full_pipeline_master", "SYSTEM", "success", "sys", elapsed_ms, total_feats)
        logger.info(f"🏆 [Master] 管線執行完畢，總處理特徵筆數: {total_feats}，耗時: {elapsed_ms/1000:.2f}s")
        
    except Exception as e:
        logger.error(f"💥 [Master] 指揮部崩潰: {e}")
        write_pipeline_log("full_pipeline_master", "SYSTEM", "failed", "sys", 0, 0, str(e))

if __name__ == "__main__":
    run_full_pipeline()
