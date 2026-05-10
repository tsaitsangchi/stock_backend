"""
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
