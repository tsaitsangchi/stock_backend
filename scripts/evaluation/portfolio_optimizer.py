"""
evaluation/portfolio_optimizer.py v5.5.x (投組最佳化(evaluation/))
================================================================================
**最後更新日期**: 2026-06-02
**主權狀態**: ACTIVE (backtest/portfolio subsystem)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:投組最佳化:投組配置/最佳化/策略(evaluation/ 版)。

**輸入 → 輸出**:預測/特徵 → 回測績效 / 投組權重

**為什麼需要它**:backtest/portfolio 子系統(evaluation/);評估策略可行性。

## 📜 一、核心定義說明 (Core Definitions)

1. **[投組最佳化]**:投組最佳化 實作
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
portfolio_optimizer.py v5.5.7 (Trinity Core Final)
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
   $ python scripts/evaluation/portfolio_optimizer.py

2. 在其他 Python 腳本中引用：
   ------------------------------------------------------------
   from evaluation.portfolio_optimizer import ...
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
    from core.db_utils import db_transaction, write_pipeline_log
    from config import STOCK_CONFIGS
except ImportError as e:
    print(f"[FATAL] 無法匯入核心配置: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    def __init__(self, risk_free_rate=0.02):
        self.risk_free_rate = risk_free_rate
        self.mbnric_industries = ["Semiconductor", "AI_Hardware", "Biotech", "Robotics"]

    def optimize(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """核心優化邏輯 (v5.5)"""
        t0 = time.monotonic()
        logger.info("🌐 正在執行宏觀感知投資組合優化...")
        
        # 1. 模擬優化算法 (槓鈴分配)
        if signals_df.empty: return pd.DataFrame()
        
        # 假設優化結果
        results = []
        for _, row in signals_df.head(5).iterrows():
            results.append({"stock_id": row["stock_id"], "weight": 0.04})
            
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        
        # 2. 混合日誌紀錄 (Category: backtest)
        write_pipeline_log(
            task_name="portfolio_optimizer",
            stock_id="PORTFOLIO_CORE",
            status="success",
            category="backtest",
            duration_ms=elapsed_ms,
            rows=len(results)
        )
        
        return pd.DataFrame(results)

def optimize_portfolio(df_all_stats, budget=100000):
    """舊版相容介面"""
    opt = PortfolioOptimizer()
    return opt.optimize(df_all_stats)

if __name__ == "__main__":
    test_df = pd.DataFrame([{"stock_id": "2330", "prob_up": 0.65}])
    print(optimize_portfolio(test_df))
