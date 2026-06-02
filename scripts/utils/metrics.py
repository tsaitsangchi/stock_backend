"""
utils/metrics.py v5.5.26 (績效指標工具)
================================================================================
**最後更新日期**: 2026-06-02
**主權狀態**: ACTIVE (utility helper)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:計算績效指標(Sharpe、最大回撤 MDD 等)的純函式工具。

**輸入 → 輸出**:報酬序列 → Sharpe / MDD

**為什麼需要它**:backtest/portfolio 子系統共用之指標計算。

## 📜 一、核心定義說明 (Core Definitions)

1. **[Metrics Helper]**:純函式績效指標
2. **[Sovereignty Declaration]**:本程式為**非 charter-core 子系統**工具(charter 可達/引用),不涉 §3.1/§3.2 序列治權主軸、不持五套禁令、不處理 §8.5 anti-leakage。
3. **[Historical Reference Authority]**:本檔標頭版本為記述性快照,非權威來源(權威為憲章 + 程式現行碼)。

## 📊 二、全量功能群矩陣 (Functional Group Matrix)

| 功能 / 指令 | 說明 |
| :--- | :--- |
| calculate_sharpe_ratio() | 年化 Sharpe |
| calculate_mdd() | 最大回撤 |

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v5.5.26 | 2026-06-02 | Codex | §一.11 標頭三段式 + 白話補正;原邏輯不變。 | **ACTIVE** |

## 原始說明
metrics.py v5.5.26 (Trinity Core Final)
================================================================================
績效指標工具 — 混合日誌整合版
負責計算投資組合與單一標的的績效指標 (ROI, Sharpe, MDD, Win Rate)。

修訂歷程：
  v5.5.26 (2026-05-10):
    - [核心] 標準化運算指標邏輯，支援與 evaluation_log 對接。
    - [文檔] 補齊極致詳細的執行範例說明。

【執行範例說明】

1. 程式內引用計算指標：
   ------------------------------------------------------------
   from utils.metrics import calculate_sharpe_ratio
   sharpe = calculate_sharpe_ratio(returns_list)
   ------------------------------------------------------------

2. 日誌查閱 (確認指標計算任務)：
   SELECT * FROM pipeline_execution_log WHERE category = 'backtest' ORDER BY created_at DESC;
"""

import sys
import numpy as np
from pathlib import Path

# ── 系統路徑修復 ──
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
except ImportError:
    pass

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """計算夏普比率"""
    if len(returns) < 2: return 0.0
    avg_ret = np.mean(returns)
    std_ret = np.std(returns)
    if std_ret == 0: return 0.0
    return (avg_ret - risk_free_rate / 252) / std_ret * np.sqrt(252)

def calculate_mdd(cum_returns):
    """計算最大回撤"""
    if len(cum_returns) < 2: return 0.0
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - peak) / peak
    return np.min(drawdown)

if __name__ == "__main__":
    test_rets = [0.01, -0.005, 0.02, 0.015, -0.01]
    print(f"測試夏普值: {calculate_sharpe_ratio(test_rets):.2f}")
