import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
sys.path.append(str(base_dir / "pipeline"))

import logging
import numpy as np
import pandas as pd
from datetime import date, timedelta
from data_pipeline import build_daily_frame
from core.db_utils import get_db_conn

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

def compute_dynamics(stock_id: str):
    logger.info(f"正在計算 {stock_id} 的動力學參數...")
    try:
        # 抓取最近 3 年資料以計算穩健參數
        df = build_daily_frame(stock_id=stock_id)
        if df.empty or len(df) < 252:
            logger.warning(f"{stock_id} 資料不足，跳過。")
            return None
        
        returns = df["close"].pct_change().dropna()
        
        # 1. info_sensitivity (資訊力敏感度)
        # 定義：價格對成交量衝擊的響應程度。使用 (Abs Returns / Log Turnover) 的相關性或比值
        turnover = df["turnover_value"].replace(0, 1)
        impact = np.abs(returns) / np.log1p(turnover.pct_change().abs())
        info_sensitivity = np.clip(impact.median() * 100, 0.1, 1.0)
        
        # 2. gravity_elasticity (重力彈性 / 均值回歸速度)
        # 定義：1 - 滯後一階自相關。自相關越低，回歸越快。
        autocorr = returns.autocorr(lag=1)
        gravity_elasticity = np.clip(1.0 - abs(autocorr), 0.1, 0.9)
        
        # 3. fat_tail_index (厚尾指數 / 峰度)
        fat_tail_index = np.clip(returns.kurtosis(), 1.0, 50.0)
        
        # 4. convexity_score (凸性分數 / Spear)
        # 定義：(上行標準差 / 下行標準差)。越高代表上行潛力越大（偏態正向）。
        pos_rets = returns[returns > 0]
        neg_rets = returns[returns < 0]
        if len(neg_rets) > 0 and neg_rets.std() > 0:
            convexity_score = np.clip(pos_rets.std() / neg_rets.std(), 0.5, 2.0)
        else:
            convexity_score = 1.0
            
        # 5. tail_risk_score (尾部風險分數 / Shield)
        # 定義：CVaR (95%)。數值越負，風險越大。
        var_95 = returns.quantile(0.05)
        tail_risk_score = np.clip(returns[returns <= var_95].mean(), -0.1, 0.0)
        
        # 6. wave_track (康波軌道)
        # 根據產業分類
        from config import STOCK_CONFIGS
        industry = STOCK_CONFIGS.get(stock_id, {}).get("industry", "Traditional")
        if industry in ["Semiconductor", "AI_Hardware"]:
            wave_track = "MBNRIC_CYBERNETIC" # 第六波驅動
        elif industry in ["Energy", "Shipping"]:
            wave_track = "TRANSITION_HUB"
        else:
            wave_track = "LEGACY_IT"
            
        # 7. innovation_velocity (創新速度)
        # 代理指標：營收增長率的二階導數（加速度）的穩定性，或簡單用 industry 權重
        rev_growth = df["rev_yoy"].dropna() if "rev_yoy" in df.columns else pd.Series([0])
        innovation_velocity = 1.2 if wave_track == "MBNRIC_CYBERNETIC" else 1.0
        
        return {
            "stock_id": stock_id,
            "info_sensitivity": float(info_sensitivity),
            "gravity_elasticity": float(gravity_elasticity),
            "fat_tail_index": float(fat_tail_index),
            "convexity_score": float(convexity_score),
            "tail_risk_score": float(tail_risk_score),
            "wave_track": wave_track,
            "innovation_velocity": float(innovation_velocity)
        }
    except Exception as e:
        logger.error(f"計算 {stock_id} 失敗: {e}")
        return None

def main():
    from config import STOCK_CONFIGS
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(DDL)
        conn.commit()
        
        results = []
        for sid in STOCK_CONFIGS.keys():
            dyn = compute_dynamics(sid)
            if dyn:
                results.append(dyn)
        
        if not results:
            logger.warning("沒有產出任何動力學參數。")
            return
            
        with conn.cursor() as cur:
            for d in results:
                cur.execute("""
                    INSERT INTO stock_dynamics_registry 
                    (stock_id, info_sensitivity, gravity_elasticity, fat_tail_index, 
                     convexity_score, tail_risk_score, wave_track, innovation_velocity)
                    VALUES (%(stock_id)s, %(info_sensitivity)s, %(gravity_elasticity)s, %(fat_tail_index)s, 
                            %(convexity_score)s, %(tail_risk_score)s, %(wave_track)s, %(innovation_velocity)s)
                    ON CONFLICT (stock_id) DO UPDATE SET
                        info_sensitivity = EXCLUDED.info_sensitivity,
                        gravity_elasticity = EXCLUDED.gravity_elasticity,
                        fat_tail_index = EXCLUDED.fat_tail_index,
                        convexity_score = EXCLUDED.convexity_score,
                        tail_risk_score = EXCLUDED.tail_risk_score,
                        wave_track = EXCLUDED.wave_track,
                        innovation_velocity = EXCLUDED.innovation_velocity,
                        last_updated = CURRENT_TIMESTAMP
                """, d)
        conn.commit()
        logger.info(f"✅ 成功更新 {len(results)} 支標的的動力學參數。")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
