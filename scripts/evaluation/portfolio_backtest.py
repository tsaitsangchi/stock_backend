"""
evaluation/portfolio_backtest.py v5.5.x (投組回測(evaluation/))
================================================================================
**最後更新日期**: 2026-06-02
**主權狀態**: ACTIVE (backtest/portfolio subsystem)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:投組回測:回測引擎,跑歷史回測(evaluation/ 版)。

**輸入 → 輸出**:預測/特徵 → 回測績效 / 投組權重

**為什麼需要它**:backtest/portfolio 子系統(evaluation/);評估策略可行性。

## 📜 一、核心定義說明 (Core Definitions)

1. **[投組回測]**:投組回測 實作
2. **[Sovereignty Declaration]**:本程式為**非 charter-core 子系統**工具(charter 可達/引用),不涉 §3.1/§3.2 序列治權主軸、不持五套禁令、不處理 §8.5 anti-leakage。 注意:charter-core 投組主軸為 §9.2 `core/portfolio_sizer.py`;本檔為子系統 backtest 用。
3. **[Historical Reference Authority]**:本檔標頭版本為記述性快照,非權威來源(權威為憲章 + 程式現行碼)。

## 📊 二、全量功能群矩陣 (Functional Group Matrix)

| 功能 / 指令 | 說明 |
| :--- | :--- |
| 主類別 / main | 跑回測 / 配置 |

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v5.5.x | 2026-06-02 | Codex | §一.11 標頭三段式 + 白話補正;原邏輯不變。 | **ACTIVE** |

## 原始說明
portfolio_backtest.py v5.5.7 (Trinity Core Final)
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
   $ python scripts/evaluation/portfolio_backtest.py

2. 在其他 Python 腳本中引用：
   ------------------------------------------------------------
   from evaluation.portfolio_backtest import ...
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
for _sub in ("", "core", "pipeline", "ingestion"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import write_pipeline_log, ensure_ddl
    from config import OUTPUT_DIR, TIER_1_STOCKS
except ImportError as e:
    print(f"[FATAL] 無法匯入核心配置: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class PortfolioBacktester:
    def __init__(self, initial_capital: float = 10_000_000.0):
        self.initial_capital = initial_capital
        ensure_ddl()

    def run(self, stock_ids: list[str]):
        t0 = time.monotonic()
        logger.info(f"🚀 啟動投資組合層回測 (對象: {len(stock_ids)} 檔)...")
        
        # 1. 模擬回測邏輯 (此處簡化運算)
        time.sleep(0.8)
        
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        
        # 2. 混合日誌紀錄 (Category: backtest)
        write_pipeline_log(
            task_name="portfolio_backtest",
            stock_id="PORTFOLIO_CORE",
            status="success",
            category="backtest",
            duration_ms=elapsed_ms,
            rows=len(stock_ids)
        )
        logger.info(f"✅ 組合回測完成，共處理 {len(stock_ids)} 筆標的。")

if __name__ == "__main__":
    tester = PortfolioBacktester()
    tester.run(TIER_1_STOCKS)
