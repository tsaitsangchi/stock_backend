import logging
import numpy as np
import pandas as pd

# ── 1. 統一環境初始化 ──
from core.path_setup import ensure_scripts_on_path
ensure_scripts_on_path(__file__)

from scripts.data_pipeline import build_daily_frame
from core.db_utils import get_db_conn, safe_commit_rows, ensure_ddl
from config import STOCK_CONFIGS

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DDL = """
CREATE TABLE IF NOT EXISTS stock_dynamics_registry (
    stock_id          VARCHAR(20) PRIMARY KEY,
    info_sensitivity  NUMERIC(8,4),
    gravity_elasticity NUMERIC(8,4),
    fat_tail_index    NUMERIC(8,2),
    convexity_score   NUMERIC(8,4),
    tail_risk_score   NUMERIC(8,4),
    wave_track        VARCHAR(50),
    innovation_velocity NUMERIC(8,4),
    last_updated      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# ... (保留 compute_dynamics 原始邏輯不變) ...
def compute_dynamics(stock_id: str):
    # (此處保留原程式碼的特徵計算邏輯)
    pass

def main():
    conn = get_db_conn()
    try:
        ensure_ddl(conn, DDL)
        
        results = []
        for sid in STOCK_CONFIGS.keys():
            dyn = compute_dynamics(sid)
            if dyn:
                # 轉為 tuple 列表以適應 bulk_upsert
                results.append((
                    dyn["stock_id"], dyn["info_sensitivity"], dyn["gravity_elasticity"],
                    dyn["fat_tail_index"], dyn["convexity_score"], dyn["tail_risk_score"],
                    dyn["wave_track"], dyn["innovation_velocity"]
                ))
        
        if not results:
            logger.warning("沒有產出任何動力學參數。")
            return
            
        # ── 2. 使用 core 的安全批次寫入 ──
        sql = """
            INSERT INTO stock_dynamics_registry 
            (stock_id, info_sensitivity, gravity_elasticity, fat_tail_index, 
             convexity_score, tail_risk_score, wave_track, innovation_velocity)
            VALUES %s
            ON CONFLICT (stock_id) DO UPDATE SET
                info_sensitivity = EXCLUDED.info_sensitivity,
                gravity_elasticity = EXCLUDED.gravity_elasticity,
                fat_tail_index = EXCLUDED.fat_tail_index,
                convexity_score = EXCLUDED.convexity_score,
                tail_risk_score = EXCLUDED.tail_risk_score,
                wave_track = EXCLUDED.wave_track,
                innovation_velocity = EXCLUDED.innovation_velocity,
                last_updated = CURRENT_TIMESTAMP
        """
        template = "(%s, %s, %s, %s, %s, %s, %s, %s)"
        rows_committed = safe_commit_rows(conn, sql, results, template, label="stock_dynamics")
        logger.info(f"✅ 成功更新 {rows_committed} 支標的的動力學參數。")
        
    finally:
        conn.close()

if __name__ == "__main__":
    main()