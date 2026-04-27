"""
portfolio_backtest.py — 組合層回測引擎
整合多標的持倉、Beta 暴露分析、換手率與壓力測試。
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Optional
from scipy.stats import linregress

# 導入專案路徑
sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.config import (
    OUTPUT_DIR, STOCK_CONFIGS, RISK_CONFIG, PORTFOLIO_EVAL_TARGETS,
    FRICTION_CONFIG, LARGE_CAP_TICKERS
)
from scripts.data_pipeline import load_total_return_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class PortfolioBacktester:
    def __init__(self, initial_capital: float = 10_000_000.0):
        self.initial_capital = initial_capital
        self.output_dir = OUTPUT_DIR

    def load_all_oof_predictions(self, stock_ids: List[str]) -> pd.DataFrame:
        """
        載入所有標的的 OOF 預測。
        """
        all_preds = []
        for sid in stock_ids:
            path = self.output_dir / f"oof_predictions_partial_{sid}.csv"
            if not path.exists():
                # 嘗試正式版名稱
                path = self.output_dir / f"oof_predictions_{sid}.csv"
            
            if path.exists():
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                df.columns = [f"{sid}_prob"]
                all_preds.append(df)
            else:
                logger.warning(f"  找不到 {sid} 的 OOF 預測檔案，略過。")
        
        if not all_preds:
            return pd.DataFrame()
        
        return pd.concat(all_preds, axis=1).sort_index()

    def load_all_prices(self, stock_ids: List[str], start_date: str) -> pd.DataFrame:
        """
        載入所有標的的收盤價與報酬。
        """
        from scripts.data_pipeline import load_stock_price
        all_prices = []
        for sid in stock_ids:
            try:
                df = load_stock_price(sid)
                if not df.empty:
                    df = df[['close']].rename(columns={'close': f"{sid}_close"})
                    all_prices.append(df)
            except Exception as e:
                logger.error(f"  載入 {sid} 股價失敗: {e}")
        
        if not all_prices:
            return pd.DataFrame()
        
        return pd.concat(all_prices, axis=1).sort_index()

    def run_backtest(self, stock_ids: List[str], start_date: str = "2018-01-01"):
        """
        執行組合層回測。
        """
        logger.info(f"=== 啟動組合層回測 (共 {len(stock_ids)} 檔標的) ===")
        
        # 1. 準備數據
        pred_df = self.load_all_oof_predictions(stock_ids)
        if pred_df.empty:
            logger.error("無法執行回測：無預測數據。")
            return
        
        price_df = self.load_all_prices(stock_ids, start_date)
        taiex = load_total_return_index()
        if not taiex.empty:
            taiex = taiex[['TAIEX']].pct_change().rename(columns={'TAIEX': 'taiex_ret'})
        
        # 對齊日期
        common_idx = pred_df.index.intersection(price_df.index)
        pred_df = pred_df.loc[common_idx]
        price_df = price_df.loc[common_idx]
        
        ret_df = price_df.pct_change()
        
        # 2. 模擬持倉邏輯 (等權重分配給 prob > 0.75 的標的)
        threshold = 0.75
        weights = (pred_df > threshold).astype(float)
        
        # 歸一化權重 (每日總權重 = 1.0)
        row_sums = weights.sum(axis=1)
        # 防止除以 0
        weights = weights.div(row_sums.replace(0, 1), axis=0)
        
        # 3. 計算組合表現
        port_ret_gross = (weights.shift(1) * ret_df).sum(axis=1)
        
        # 4. 交易成本與換手率
        # 換手率 = 權重絕對變動之和 / 2
        weight_diff = weights.diff().abs().sum(axis=1) / 2.0
        turnover_annual = weight_diff.mean() * 252
        
        # 平均成本 (簡化版：買賣各 0.3%~0.5%)
        # 假設平均成本為 0.4%
        avg_tc = (FRICTION_CONFIG["commission"] * 2 + FRICTION_CONFIG["securities_tax"] + 0.002)
        tc_series = weight_diff * avg_tc
        
        port_ret_net = port_ret_gross - tc_series
        
        # 5. 指標計算
        cum_ret = (1 + port_ret_net).cumprod()
        total_ret = cum_ret.iloc[-1] - 1
        
        # 年化報酬 (複利)
        days = (cum_ret.index[-1] - cum_ret.index[0]).days
        ann_ret = (1 + total_ret) ** (365.25 / days) - 1
        
        ann_vol = port_ret_net.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        
        # 最大回撤
        peak = cum_ret.cummax()
        dd = (cum_ret - peak) / peak
        mdd = dd.min()
        
        calmar = ann_ret / abs(mdd) if mdd < 0 else 0
        
        # 最差單月
        monthly_ret = port_ret_net.resample('M').apply(lambda x: (1 + x).prod() - 1)
        worst_month = monthly_ret.min()
        
        # Beta to TAIEX
        beta = 0.0
        if not taiex.empty:
            merged = pd.DataFrame({'port': port_ret_net, 'mkt': taiex['taiex_ret']}).dropna()
            if len(merged) > 30:
                slope, _, _, _, _ = linregress(merged['mkt'], merged['port'])
                beta = slope

        # 6. 輸出報告
        print("\n" + "═"*50)
        print(" 《2026 量子金融藍圖》組合層壓力測試報告")
        print("─"*50)
        print(f" 回測區間: {common_idx[0].date()} ~ {common_idx[-1].date()}")
        print(f" 標的數量: {len(stock_ids)} 檔")
        print(f" 初始資金: {self.initial_capital:,.0f} TWD")
        print("─"*50)
        
        metrics = {
            "portfolio_sharpe": sharpe,
            "max_drawdown":    mdd,
            "calmar_ratio":    calmar,
            "beta_to_taiex":   beta,
            "turnover_rate":   turnover_annual,
            "worst_month_ret": worst_month,
            "total_return":    total_ret,
            "annual_return":   ann_ret
        }
        
        for k, v in metrics.items():
            tgt = PORTFOLIO_EVAL_TARGETS.get(k, None)
            flag = ""
            if tgt is not None:
                # 注意：max_drawdown 和 worst_month_ret 是越接近 0 越好（值越大）
                if k in ["max_drawdown", "worst_month_ret"]:
                    flag = "✅" if v >= tgt else "⚠️ (高風險)"
                elif k == "beta_to_taiex":
                    flag = "✅" if v <= tgt else "⚠️ (高暴露)"
                else:
                    flag = "✅" if v >= tgt else "⚠️ (低預期)"
            
            unit = "%" if k in ["max_drawdown", "worst_month_ret", "total_return", "annual_return", "turnover_rate"] else ""
            val = v * 100 if unit == "%" else v
            print(f" {k:18s}: {val:8.2f}{unit}  {flag}")
        
        print("═"*50)
        
        # 儲存組合權益曲線
        cum_ret.to_csv(self.output_dir / "portfolio_equity_curve.csv")
        logger.info(f"權益曲線已儲存至 {self.output_dir}/portfolio_equity_curve.csv")
        
        return metrics

if __name__ == "__main__":
    from scripts.config import TIER_1_STOCKS
    tester = PortfolioBacktester()
    tester.run_backtest(TIER_1_STOCKS)
