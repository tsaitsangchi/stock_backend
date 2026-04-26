"""
calibrate_stock_physics.py — 物理常數與 80/20 策略校準儀 (v3.0 Strategic)
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
    logger.info(f"===> 80/20 戰略校準中: {stock_id}")
    
    raw = build_daily_frame(stock_id=stock_id, start_date="2023-01-01")
    if len(raw) < 100: return
        
    df = build_features(raw, stock_id=stock_id, for_inference=True)
    
    # 1. 核心物理常數
    avg_mass = df["inertial_mass"].median() if "inertial_mass" in df.columns else 12.0
    gravity_elasticity = df["gravity_pull"].std() if "gravity_pull" in df.columns else 0.05
    
    # 2. 80/20 戰略參數
    convexity_score = 0.0
    tail_risk_score = 0.0
    
    if "info_force_per_mass" in df.columns and "displacement_pct" in df.columns:
        force = df["info_force_per_mass"]
        disp = df["displacement_pct"]
        
        # 右側 20%：極端正向力下的表現 (Convexity)
        upper_force_threshold = force.quantile(0.8)
        convexity_score = disp[force > upper_force_threshold].mean() * 100
        
        # 左側 20%：極端負向力下的表現 (Tail Risk)
        lower_force_threshold = force.quantile(0.2)
        tail_risk_score = disp[force < lower_force_threshold].mean() * 100

    # 3. 寫入資料庫
    try:
        conn = psycopg2.connect(dbname="stock", user="stock", password="stock", host="localhost")
        cur = conn.cursor()
        upsert_query = """
        INSERT INTO stock_physics_registry (
            stock_id, avg_mass, gravity_elasticity, 
            convexity_score, tail_risk_score, updated_at
        ) VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (stock_id) DO UPDATE SET
            avg_mass = EXCLUDED.avg_mass,
            gravity_elasticity = EXCLUDED.gravity_elasticity,
            convexity_score = EXCLUDED.convexity_score,
            tail_risk_score = EXCLUDED.tail_risk_score,
            updated_at = CURRENT_TIMESTAMP;
        """
        cur.execute(upsert_query, (
            stock_id, float(avg_mass), float(gravity_elasticity), 
            float(convexity_score or 0), float(tail_risk_score or 0)
        ))
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"  [OK] {stock_id} | Convexity: {convexity_score:.2f} | TailRisk: {tail_risk_score:.2f}")
    except Exception as e:
        logger.error(f"  資料庫寫入失敗: {e}")

if __name__ == "__main__":
    target_stocks = ["2330", "2317", "2454", "2376", "2382", "3661", "6669"]
    for sid in target_stocks:
        calibrate_stock(sid)
