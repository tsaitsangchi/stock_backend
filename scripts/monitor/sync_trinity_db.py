import sys
from pathlib import Path
import logging
from datetime import date
import pandas as pd

# Setup paths
base_dir = Path(__file__).resolve().parent.parent
for sub in ["fetchers", "pipeline", "training", "monitor"]:
    sys.path.append(str(base_dir / sub))
sys.path.append(str(base_dir))

from core.db_utils import get_db_conn
from data_pipeline import _query
from data_integrity_audit import IntegrityAuditor
from model_health_check import check_model_files_df

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# DDL (Trinity Tables)
# ─────────────────────────────────────────────
DDL_HEALTH_MATRIX = """
CREATE TABLE IF NOT EXISTS system_health_matrix (
    stock_id VARCHAR(10) NOT NULL,
    audit_date DATE NOT NULL,
    data_coverage_pct NUMERIC(5,2),
    model_status VARCHAR(50),
    prediction_status VARCHAR(50) DEFAULT '🔴 NO_PRED',
    health_score NUMERIC(5,2),
    last_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (stock_id, audit_date)
);
"""

DDL_INVESTMENT_RECOMM = """
CREATE TABLE IF NOT EXISTS investment_recommendations (
    stock_id VARCHAR(10) NOT NULL,
    prediction_date DATE NOT NULL,
    prob_up NUMERIC(5,4),
    signal_level VARCHAR(50),
    recommended_weight NUMERIC(5,4),
    last_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (stock_id, prediction_date)
);
"""

def sync_health_matrix():
    logger.info("同步全系統健康度矩陣...")
    
    # 確保 DDL 存在並更新欄位 (Idempotent)
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(DDL_HEALTH_MATRIX)
    cur.execute(DDL_INVESTMENT_RECOMM)
    # 補丁：若 prediction_status 欄位不存在則新增
    cur.execute("""
        DO $$ 
        BEGIN 
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='system_health_matrix' AND column_name='prediction_status') THEN
                ALTER TABLE system_health_matrix ADD COLUMN prediction_status VARCHAR(50) DEFAULT '🔴 NO_PRED';
            END IF;
        END $$;
    """)
    conn.commit()

    auditor = IntegrityAuditor(days_window=60)
    df = auditor.audit_coverage_matrix()
    
    # 獲取模型狀態
    model_df = check_model_files_df(auditor.stock_ids)
    
    # 獲取今日預測狀態
    pred_df = _query("SELECT stock_id FROM stock_forecast_daily WHERE predict_date = (SELECT MAX(predict_date) FROM stock_forecast_daily) AND day_offset = 30")
    pred_ready_ids = set(pred_df["stock_id"].astype(str).tolist()) if not pred_df.empty else set()

    today = date.today()
    for _, row in df.iterrows():
        sid = str(row["stock_id"])
        # 解析百分比字串
        vals = row.iloc[1:].apply(lambda x: float(str(x).rstrip('%')) if isinstance(x, str) else float(x))
        avg = float(vals.mean())
        
        # 狀態對應
        m_status = model_df[model_df["stock_id"] == sid]["status"].iloc[0] if sid in model_df["stock_id"].values else "🔴 MISSING"
        p_status = "🟢 READY" if sid in pred_ready_ids else "🔴 NO_PRED"
        
        cur.execute("""
            INSERT INTO system_health_matrix (stock_id, audit_date, data_coverage_pct, model_status, prediction_status, health_score)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (stock_id, audit_date) DO UPDATE SET
                data_coverage_pct = EXCLUDED.data_coverage_pct,
                model_status = EXCLUDED.model_status,
                prediction_status = EXCLUDED.prediction_status,
                health_score = EXCLUDED.health_score,
                last_updated_at = CURRENT_TIMESTAMP;
        """, (sid, today, avg, m_status, p_status, avg/100.0))
    
    conn.commit()
    conn.close()
    logger.info("健康度矩陣同步完成。")

def sync_investment_recommendations():
    logger.info("同步投資建議矩陣...")
    # 從預測結果表提取最新建議
    query = """
    WITH latest_pred AS (
        SELECT stock_id, predict_date, prob_up, 
               RANK() OVER (PARTITION BY stock_id ORDER BY predict_date DESC) as rk
        FROM stock_forecast_daily
    )
    SELECT * FROM latest_pred WHERE rk = 1
    """
    df = _query(query)
    if df.empty:
        logger.warning("尚無預測資料可同步。")
        return
        
    conn = get_db_conn()
    cur = conn.cursor()
    
    for _, row in df.iterrows():
        sid = str(row["stock_id"])
        prob = float(row["prob_up"])
        p_date = row["predict_date"]
        
        # 簡易信號分級邏輯 (可根據需求微調)
        if prob >= 0.8: level = "🚀 極致攻擊 (Strong Buy)"
        elif prob >= 0.65: level = "🏹 戰略突擊 (Buy)"
        elif prob >= 0.55: level = "🛡️ 陣地防禦 (Hold)"
        elif prob >= 0.45: level = "🌫️ 混沌中立 (Neutral)"
        else: level = "🧊 零度真空 (Avoid)"
        
        weight = max(0, (prob - 0.5) * 2 * 0.1) # 簡易倉位建議
        
        cur.execute("""
            INSERT INTO investment_recommendations (stock_id, prediction_date, prob_up, signal_level, recommended_weight)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (stock_id, prediction_date) DO UPDATE SET
                prob_up = EXCLUDED.prob_up,
                signal_level = EXCLUDED.signal_level,
                recommended_weight = EXCLUDED.recommended_weight,
                last_updated_at = CURRENT_TIMESTAMP;
        """, (sid, p_date, prob, level, weight))
        
    conn.commit()
    conn.close()
    logger.info("投資建議矩陣同步完成。")

if __name__ == "__main__":
    sync_health_matrix()
    sync_investment_recommendations()
    logger.info("✅ 全系統 Trinity 資料庫同步完成！")
