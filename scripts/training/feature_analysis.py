"""
feature_analysis.py — 因子穩定性與有效性分析工具
==========================================
計算各類特徵的 Information Coefficient (IC) 與 IC Information Ratio (IC IR)，
識別具備穩定預測力的「強健因子」。
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict

logger = logging.getLogger(__name__)

class FactorAnalyzer:
    def __init__(self, df: pd.DataFrame, target_col: str = "target_30d"):
        self.df = df.copy()
        self.target_col = target_col
        if target_col not in self.df.columns:
            logger.warning(f"Target column {target_col} not found in DataFrame.")

    def analyze_robustness(self, features: List[str], window: int = 252) -> pd.DataFrame:
        """
        計算特徵的 Rolling IC, IC IR 與 因子衰減。
        
        參數:
            window: Rolling 視窗大小 (252 代表約一年)
        """
        if self.target_col not in self.df.columns:
            return pd.DataFrame()

        results = []
        y = self.df[self.target_col]
        
        logger.info(f"開始分析 {len(features)} 個特徵的穩定性 (Window={window})...")
        
        for f in features:
            if f not in self.df.columns:
                continue
            
            x = self.df[f]
            
            # 1. Rolling Correlation (IC)
            ic_series = x.rolling(window).corr(y)
            
            # 2. 基本指標
            mean_ic = ic_series.mean()
            std_ic = ic_series.std()
            ic_ir = mean_ic / std_ic if (std_ic and not np.isnan(std_ic)) else 0
            
            # 3. 衰減率 (Decay): 最近 60 天 IC 與全域均值的偏差
            recent_ic = ic_series.tail(60).mean()
            # 若 recent_ic < mean_ic (絕對值縮小)，代表衰減
            decay = (np.abs(recent_ic) - np.abs(mean_ic)) / np.abs(mean_ic) if mean_ic != 0 else 0
            
            results.append({
                "feature": f,
                "mean_ic": round(mean_ic, 4),
                "ic_ir": round(ic_ir, 4),
                "decay": round(decay, 4),
                "is_robust": (np.abs(mean_ic) > 0.02) and (np.abs(ic_ir) >= 0.5)
            })
            
        res_df = pd.DataFrame(results)
        if not res_df.empty:
            res_df = res_df.sort_values("ic_ir", ascending=False).reset_index(drop=True)
            
        robust_count = res_df[res_df["is_robust"]].shape[0] if not res_df.empty else 0
        logger.info(f"分析完成。共識別出 {robust_count} 個強健因子 (IC IR >= 0.5)。")
        
        return res_df

def print_factor_report(report_df: pd.DataFrame):
    """
    列印因子健壯性報告。
    """
    if report_df.empty:
        print("查無有效的因子分析數據。")
        return

    print("\n" + "="*80)
    print("【特徵穩定性報告 (Factor Robustness Report)】")
    print("="*80)
    
    robust = report_df[report_df["is_robust"]]
    print(f"總特徵數: {len(report_df)} | 強健因子數 (IC IR >= 0.5): {len(robust)}")
    print("-" * 80)
    
    if not robust.empty:
        print("TOP 10 強健因子:")
        print(robust.head(10)[["feature", "mean_ic", "ic_ir", "decay"]].to_string(index=False))
    else:
        print("警告: 查無任何特徵達到強健門檻 (IC IR >= 0.5)。")
        
    print("\n衰減最嚴重因子 (Decay < -0.3):")
    decaying = report_df[report_df["decay"] < -0.3].sort_values("decay")
    if not decaying.empty:
        print(decaying.head(5)[["feature", "mean_ic", "ic_ir", "decay"]].to_string(index=False))
    else:
        print("目前因子表現穩定，無顯著衰減。")
    print("="*80 + "\n")
