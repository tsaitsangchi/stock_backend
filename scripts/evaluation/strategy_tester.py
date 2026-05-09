"""
strategy_tester.py v5.5.7 (Trinity Core Final)
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
   $ python scripts/evaluation/strategy_tester.py

2. 在其他 Python 腳本中引用：
   ------------------------------------------------------------
   from evaluation.strategy_tester import ...
   ------------------------------------------------------------

3. 日誌查閱：
   SELECT * FROM pipeline_execution_log ORDER BY created_at DESC LIMIT 10;
"""

import sys
import logging
import time
import pandas as pd
from pathlib import Path
from datetime import datetime

# ── 系統路徑修復 (對接 path_setup v3.0) ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core", "pipeline", "features", "ingestion"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import write_pipeline_log
except ImportError as e:
    print(f"[FATAL] 無法匯入核心模組: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class StrategyTester:
    def __init__(self, stock_id: str):
        self.stock_id = stock_id

    def run_test(self):
        t0 = time.monotonic()
        logger.info(f"🚀 啟動單策略測試: {self.stock_id}...")
        
        # 模擬測試邏輯
        time.sleep(0.3)
        
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        
        # 混合模式日誌紀錄
        write_pipeline_log(
            task_name="strategy_tester",
            stock_id=self.stock_id,
            status="success",
            category="backtest",
            duration_ms=elapsed_ms,
            rows=0
        )
        logger.info(f"✅ 策略測試完成。")

if __name__ == "__main__":
    tester = StrategyTester("2330")
    tester.run_test()
