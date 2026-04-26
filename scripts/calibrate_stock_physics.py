"""
calibrate_stock_physics.py — 快速物理校準儀 (v2.0 Optimized)
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import psycopg2
from pathlib import Path

# 修正路徑
sys.path.append(os.getcwd())
from scripts.data_pipeline import build_daily_frame, _query
from scripts.feature_engineering import build_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def calibrate_stock(stock_id: str):
    logger.info(f"===> 快速校準中: {stock_id}")
    
    # 使用較短的歷史資料 (500日) 以加速
    raw = build_daily_frame(stock_id=stock_id, start_date="2023-01-01")
    if len(raw) < 100: return
        
    df = build_features(raw, stock_id=stock_id, for_inference=True)
    
    # 優化計算：使用標準滾動統計而非 rolling.apply(rank)
    avg_mass = df["inertial_mass"].median() if "inertial_mass" in df.columns else 12.0
    gravity_elasticity = df["gravity_pull"].std() if "gravity_pull" in df.columns else 0.05
    
    info_sensitivity = 0.5
    if "info_force_per_mass" in df.columns and "displacement_pct" in df.columns:
        # 只取最後 126 日的相關性
        sub = df.tail(126)
        info_sensitivity = sub["info_force_per_mass"].corr(sub["displacement_pct"].shift(-1))
        if np.isnan(info_sensitivity): info_sensitivity = 0.5
        
    fat_tail_index = df["log_return_1d"].tail(252).kurtosis() if "log_return_1d" in df.columns else 3.0
    
    # 寫入資料庫 (帶密碼環境變數)
    try:
        conn = psycopg2.connect(dbname="stock", user="stock", password="stock", host="localhost")
        cur = conn.cursor()
        upsert_query = """
        INSERT INTO stock_physics_registry (
            stock_id, avg_mass, gravity_elasticity, info_sensitivity, 
            fat_tail_index, updated_at
        ) VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (stock_id) DO UPDATE SET
            avg_mass = EXCLUDED.avg_mass,
            gravity_elasticity = EXCLUDED.gravity_elasticity,
            info_sensitivity = EXCLUDED.info_sensitivity,
            fat_tail_index = EXCLUDED.fat_tail_index,
            updated_at = CURRENT_TIMESTAMP;
        """
        cur.execute(upsert_query, (stock_id, float(avg_mass), float(gravity_elasticity), float(info_sensitivity), float(fat_tail_index)))
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"  [OK] {stock_id} | Sens: {info_sensitivity:.3f} | Mass: {avg_mass:.1f}")
    except Exception as e:
        logger.error(f"  資料庫寫入失敗: {e}")

if __name__ == "__main__":
    target_stocks = ["2330", "2317", "2454", "2376", "2382", "3661", "6669"]
    for sid in target_stocks:
        calibrate_stock(sid)
