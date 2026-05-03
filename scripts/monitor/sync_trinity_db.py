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
    investment_amount NUMERIC(15,2) DEFAULT 0,
    recommended_shares BIGINT DEFAULT 0,
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

def sync_investment_recommendations(total_capital: float = 100000.0):
    logger.info(f"同步投資建議矩陣 (總資金: ${total_capital:,.0f})...")
    
    # 確保 DDL 並補償新欄位
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(DDL_INVESTMENT_RECOMM)
    for col in ["investment_amount", "recommended_shares"]:
        cur.execute(f"""
            DO $$ 
            BEGIN 
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='investment_recommendations' AND column_name='{col}') THEN
                    ALTER TABLE investment_recommendations ADD COLUMN {col} NUMERIC;
                END IF;
            END $$;
        """)
    conn.commit()

    # 從預測結果表提取最新建議 (限制最近 7 天，避免推薦過期訊號)
    query = """
    WITH latest_pred AS (
        SELECT f.stock_id, f.predict_date, f.prob_up, f.current_close,
               RANK() OVER (PARTITION BY f.stock_id ORDER BY f.predict_date DESC) as rk
        FROM stock_forecast_daily f
        WHERE f.day_offset = 30
          AND f.predict_date >= (SELECT MAX(predict_date) FROM stock_forecast_daily) - INTERVAL '7 days'
    )
    SELECT * FROM latest_pred WHERE rk = 1
    ORDER BY prob_up DESC
    """
    df = _query(query)
    if df.empty:
        logger.warning("尚無最近 7 天的預測資料可同步。")
        conn.close()
        return
        
    # 1. 篩選最強前三支 (且必須具備看漲信號 prob_up > 0.55)
    top_picks = df[df["prob_up"] > 0.55].head(3).copy()
    logger.info(f"符合建議門檻 (>0.55) 的標的數: {len(top_picks)}")
    
    if top_picks.empty:
        logger.info("⚠️ 目前無足夠強度的投資標的，建議全部保留現金。")
    else:
        # 2. 依訊號強度 (prob_up - 0.5) 進行權重分配
        # 權重公式：(P - 0.5) / Sum(P_i - 0.5)
        top_picks["excess_prob"] = top_picks["prob_up"] - 0.5
        total_excess = top_picks["excess_prob"].sum()
        top_picks["alloc_ratio"] = top_picks["excess_prob"] / total_excess
        top_picks["alloc_amount"] = top_picks["alloc_ratio"] * total_capital
        
        # 檢查 current_close 是否有效
        top_picks["current_close"] = top_picks["current_close"].fillna(0)
        
        # 計算股數 (取整數，無條件捨去以免超額)
        def calc_shares(row):
            if row["current_close"] <= 0: return 0
            return int(row["alloc_amount"] / row["current_close"])
            
        top_picks["shares"] = top_picks.apply(calc_shares, axis=1)
        # 修正實際金額
        top_picks["actual_amount"] = top_picks["shares"] * top_picks["current_close"]
        logger.info(f"配置標的: {top_picks[['stock_id', 'prob_up', 'actual_amount']].to_dict('records')}")

    # 3. 遍歷所有最新預測並寫入
    top_ids_set = set(top_picks["stock_id"].astype(str).tolist()) if not top_picks.empty else set()
    
    for _, row in df.iterrows():
        sid = str(row["stock_id"])
        prob = float(row["prob_up"])
        p_date = row["predict_date"]
        # 處理可能缺失的收盤價
        price_val = row.get("current_close")
        price = float(price_val) if pd.notnull(price_val) else 0.0
        
        # 信號分級
        if prob >= 0.8: level = "🚀 極致攻擊 (Strong Buy)"
        elif prob >= 0.65: level = "🏹 戰略突擊 (Buy)"
        elif prob >= 0.55: level = "🛡️ 陣地防禦 (Hold)"
        elif prob >= 0.45: level = "🌫️ 混沌中立 (Neutral)"
        else: level = "🧊 零度真空 (Avoid)"
        
        # 建議倉位 (原有邏輯作為基準)
        weight = max(0, (prob - 0.5) * 2 * 0.1)
        
        # 判斷是否為前三名分配標的，若是則將推薦權重對齊分配比例
        invest_amt = 0.0
        shares = 0
        if sid in top_ids_set:
            match = top_picks[top_picks["stock_id"].astype(str) == sid]
            if not match.empty:
                invest_amt = float(match["actual_amount"].iloc[0])
                shares = int(match["shares"].iloc[0])
                weight = float(match["alloc_ratio"].iloc[0])  # 對齊分配比例
                level = f"⭐ 核心配置 | {level}"
        
        cur.execute("""
            INSERT INTO investment_recommendations (
                stock_id, prediction_date, prob_up, signal_level, 
                recommended_weight, investment_amount, recommended_shares
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (stock_id, prediction_date) DO UPDATE SET
                prob_up = EXCLUDED.prob_up,
                signal_level = EXCLUDED.signal_level,
                recommended_weight = EXCLUDED.recommended_weight,
                investment_amount = EXCLUDED.investment_amount,
                recommended_shares = EXCLUDED.recommended_shares,
                last_updated_at = CURRENT_TIMESTAMP;
        """, (sid, p_date, prob, level, weight, invest_amt, shares))
        
    conn.commit()
    conn.close()
    logger.info("投資建議矩陣同步完成。")

if __name__ == "__main__":
    sync_health_matrix()
    sync_investment_recommendations()
    logger.info("✅ 全系統 Trinity 資料庫同步完成！")
