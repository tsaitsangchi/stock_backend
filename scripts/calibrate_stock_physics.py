"""
calibrate_stock_physics.py — 物理常數與 MBNRIC 賽道校準小區 (v4.0 Matrix)

修改摘要（第三輪審查修復）：
  [P1 2.6] 移除硬編碼 DB 憑證（psycopg2.connect 績效 hardcode）
           改由 config.DB_CONFIG 統一引用
  [P2 3.2] 修正資料表名稱： stock_physics_registry → stock_dynamics_registry
           以對齊 signal_filter.py 的 _load_dynamics_registry() 讀取逻輯
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import psycopg2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.data_pipeline import build_daily_frame
from scripts.feature_engineering import build_features
from scripts.config import DB_CONFIG  # [P1 修復] 統一來源

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def calibrate_stock(stock_id: str):
    logger.info(f"===> MBNRIC 矩陣校準: {stock_id}")
    
    raw = build_daily_frame(stock_id=stock_id, start_date="2023-01-01")
    if len(raw) < 100: return
        
    df = build_features(raw, stock_id=stock_id, for_inference=True)
    
    # 1. 核心物理常數
    avg_mass = df["inertial_mass"].median() if "inertial_mass" in df.columns else 12.0
    gravity_elasticity = df["gravity_pull"].std() if "gravity_pull" in df.columns else 0.05
    
    # 2. 創新速度 (Innovation Velocity)
    # 邏輯：計算標的在「資訊衝擊 (Force)」與「能量釋放 (Energy)」之間的轉化率
    if "info_force_per_mass" in df.columns and "kinetic_energy" in df.columns:
        # 使用滾動相關性與斜率作為速度代理
        sub = df.tail(252)
        velocity = sub["info_force_per_mass"].corr(sub["kinetic_energy"])
        innovation_velocity = max(0.5, min(3.0, velocity * 5.0)) # 正規化至 0.5~3.0
    else:
        innovation_velocity = 1.0

    # 3. 寫入資料庫
    # [P2 3.2] 資料表改為 stock_dynamics_registry，對齊 signal_filter.py 的 _load_dynamics_registry()
    try:
        conn = psycopg2.connect(**DB_CONFIG)  # [P1] 統一引用 config.DB_CONFIG
        cur = conn.cursor()
        upsert_query = """
        INSERT INTO stock_dynamics_registry
            (stock_id, avg_mass, gravity_elasticity, innovation_velocity, updated_at)
        VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (stock_id) DO UPDATE SET
            avg_mass            = EXCLUDED.avg_mass,
            gravity_elasticity  = EXCLUDED.gravity_elasticity,
            innovation_velocity = EXCLUDED.innovation_velocity,
            updated_at          = CURRENT_TIMESTAMP;
        """
        cur.execute(upsert_query, (stock_id, float(avg_mass), float(gravity_elasticity), float(innovation_velocity)))
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"  [OK] {stock_id} | avg_mass={avg_mass:.2f} | innovation_velocity={innovation_velocity:.2f}")
    except Exception as e:
        logger.error(f"  資料庫更新失敗: {e}")

if __name__ == "__main__":
    target_stocks = ["2330", "2317", "2454", "2376", "2382", "3661", "6669", "2308", "1513"]
    for sid in target_stocks:
        calibrate_stock(sid)
