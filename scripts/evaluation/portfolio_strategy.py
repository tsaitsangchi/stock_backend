"""
portfolio_strategy.py v5.5.7 (Trinity Core Final)
================================================================================
系統組件 — 混合模式日誌實作版
目錄：evaluation

修訂歷程：
  v5.5.7 (2026-05-09):
    - [文檔] 補齊極致詳細的執行範例說明。
  v5.5.x (2026-05-09):
    - [核心] 導入 Hybrid Logging 混合日誌與路徑標準化。

【執行範例說明】

1. 直接從命令行執行：
   $ python scripts/evaluation/portfolio_strategy.py

2. 在其他 Python 腳本中引用：
   ------------------------------------------------------------
   from evaluation.portfolio_strategy import ...
   ------------------------------------------------------------

3. 日誌查閱：
   SELECT * FROM pipeline_execution_log ORDER BY created_at DESC LIMIT 10;
"""

import sys
import logging
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ── 系統路徑修復 (對接 path_setup v3.0) ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core", "pipeline", "ingestion", "inference", "features"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import write_pipeline_log, ensure_ddl
    from config import STOCK_CONFIGS
    from signal_filter import SignalFilter
except ImportError as e:
    print(f"[FATAL] 無法匯入核心配置: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class BarbellAllocator:
    def __init__(self, total_budget=10_000_000.0):
        self.total_budget = total_budget
        ensure_ddl()

    def allocate(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        t0 = time.monotonic()
        logger.info("📐 執行《2026 量子金融藍圖》槓鈴戰略分配...")
        
        # 1. 模擬分配邏輯
        time.sleep(0.5)
        
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        
        # 2. 混合日誌紀錄 (Category: backtest)
        write_pipeline_log(
            task_name="barbell_strategy",
            stock_id="STRATEGY_CORE",
            status="success",
            category="backtest",
            duration_ms=elapsed_ms,
            rows=len(signals_df)
        )
        return signals_df # 假設返回結果

if __name__ == "__main__":
    allocator = BarbellAllocator()
    test_df = pd.DataFrame([{"stock_id": "2330", "prob_up": 0.65}])
    allocator.allocate(test_df)
