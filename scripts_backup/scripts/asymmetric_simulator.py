"""
asymmetric_simulator.py — 《2026 量子金融藍圖》物理脈搏診斷儀
"""

import os
import sys
import pandas as pd
import numpy as np
import logging

sys.path.append(os.getcwd())
from scripts.data_pipeline import build_daily_frame
from scripts.feature_engineering import build_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_diagnostics(stock_id="2330"):
    logger.info(f"=== 啟動物理脈搏診斷: {stock_id} (2022-2026) ===")
    
    raw = build_daily_frame(stock_id=stock_id, start_date="2022-01-01")
    df = build_features(raw, stock_id=stock_id, for_inference=True)
    
    # 診斷核心物理指標
    physics_metrics = {
        "引力拉力 (Gravity Pull)": df["gravity_pull"],
        "資訊衝擊 (Info Force)": df["info_force_per_mass"],
        "量子動能 (Momentum)": df["quantum_momentum"]
    }
    
    print("\n" + "="*50)
    print(f" 台積電 (2330) 物理脈搏診斷報告 (2022-2026)")
    print("-" * 50)
    for name, series in physics_metrics.items():
        print(f" {name}:")
        print(f"   - 最大值: {series.max():.4f}")
        print(f"   - 最小值: {series.min():.4f}")
        print(f"   - 平均值: {series.mean():.4f}")
        print(f"   - 標準差: {series.std():.4f}")
    print("="*50)

    # 模擬一次「寬鬆版」物理套利
    # 門檻設為 1 個標準差
    pull_thresh = df["gravity_pull"].mean() - df["gravity_pull"].std()
    force_thresh = df["info_force_per_mass"].mean() + df["info_force_per_mass"].std()
    
    capital = 1000000.0
    initial_capital = capital
    position = 0
    equity_curve = []
    
    for i in range(126, len(df)):
        date = df.index[i]
        latest = df.iloc[i]
        price = latest["close"]
        
        if position == 0 and latest["gravity_pull"] < pull_thresh and latest["info_force_per_mass"] > force_thresh:
            position = (capital * 0.2) / price
            capital *= 0.8
            logger.info(f"[{date.date()}] ⚡ 物理脈搏觸發 | 價格: {price:.1f}")
        elif position > 0 and latest["gravity_pull"] > 0:
            capital += position * price * (1 - 0.003)
            position = 0
            logger.info(f"[{date.date()}] ✅ 引力平衡出場 | 價格: {price:.1f}")
            
        equity_curve.append(capital + (position * price))
        
    print(f" 寬鬆版物理模擬報酬: {(equity_curve[-1]/initial_capital-1):.2%}")
    print("="*50)

if __name__ == "__main__":
    run_diagnostics("2330")
