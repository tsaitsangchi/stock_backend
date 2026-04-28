"""
singularity_layout_simulator.py — 《2026 奇點佈局》戰略模擬器
核心任務：模擬在 2025-2026 週期低谷，如何利用「重力井底部」信號戰略性建倉。
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime

sys.path.append(os.getcwd())
from scripts.data_pipeline import build_daily_frame
from scripts.feature_engineering import build_features
from scripts.signal_filter import SignalFilter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_singularity_simulation(stock_id="2330"):
    logger.info(f"=== 啟動 2026 奇點佈局模擬: {stock_id} ===")
    
    # 讀取數據 (模擬接近 2026 的環境)
    raw = build_daily_frame(stock_id=stock_id, start_date="2024-01-01")
    df = build_features(raw, stock_id=stock_id, for_inference=True)
    
    capital = 1000000.0
    initial_capital = capital
    position = 0
    trades = []
    
    # 模擬 2026 佈局邏輯
    for i in range(126, len(df)):
        date = df.index[i]
        latest = df.iloc[i]
        price = latest["close"]
        
        # 物理信號偵測
        gravity_pull = latest.get("gravity_pull", 0)
        info_force = latest.get("info_force_per_mass", 0)
        singularity_dist = latest.get("singularity_dist", 2.0)
        
        # 戰略建倉條件：重力井極度底部 (-1.5σ) + 資訊力初步注入
        pull_thresh = -10.0 # 針對 2330 的物理感應門檻
        
        if position == 0 and gravity_pull < pull_thresh and info_force > 0:
            # 觸發 20% 量子端建倉
            position = (capital * 0.2) / price
            capital *= 0.8
            trades.append({"date": date, "type": "ENTRY", "price": price, "pull": gravity_pull})
            logger.info(f"[{date.date()}] 🌌 奇點佈局啟動 | 價格: {price:.1f} | 重力拉力: {gravity_pull:.2f} | 奇點距離: {singularity_dist:.2f}yr")
            
        # 奇點噴發/引力回歸出場
        elif position > 0 and gravity_pull > 0:
            capital += position * price * (1 - 0.003)
            position = 0
            trades.append({"date": date, "type": "EXIT", "price": price, "pull": gravity_pull})
            logger.info(f"[{date.date()}] 🚀 奇點動能噴發 | 價格: {price:.1f} | 獲利出場")
            
    final_equity = capital + (position * df["close"].iloc[-1])
    
    print("\n" + "="*60)
    print(f" 《2026 奇點佈局》模擬戰報 — {stock_id}")
    print("-" * 60)
    print(f" 戰略目標: 重力井底部左側建倉")
    print(f" 模擬區間: 2024 - 2026 奇點前夕")
    print(f" 最終總資產: {final_equity:,.0f} ({((final_equity/initial_capital)-1):.2%})")
    print(f" 執行次數: {len(trades)//2} 次完整循環")
    if trades:
        for t in trades:
            print(f"  - {t['date'].date()} | {t['type']} | 價格: {t['price']:.1f} | 引力: {t['pull']:.2f}")
    print("="*60)

if __name__ == "__main__":
    # 選擇第六波賽道代表：台積電 (AI)、台達電 (綠能)、聯發科 (邊緣 AI)、世芯 (ASIC)
    for sid in ["2330", "2308", "2454", "3661"]:
        try:
            run_singularity_simulation(sid)
        except Exception as e:
            logger.error(f"標的 {sid} 模擬失敗: {e}")
