import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
for sub in ["fetchers", "pipeline", "training", "monitor"]: sys.path.append(str(base_dir / sub))
sys.path.append(str(base_dir))
import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
for sub in ["fetchers", "pipeline", "training", "monitor"]: sys.path.append(str(base_dir / sub))
sys.path.append(str(base_dir))
import sys
from pathlib import Path
# Copyright (c) 2026 Antigravity Quant Research. All rights reserved.
# Proprietary and Confidential.
# ... (Skill License Header)

"""
portfolio_optimizer.py v2.1 — 宏觀感知優化器
新增功能：
  1. 康波週期時代溢價 (K-Wave Era Premium)
  2. 美元流動性過濾器 (USD Liquidity Filter)
"""

import argparse
import logging
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from config import STOCK_CONFIGS
from data_pipeline import DB_CONFIG, load_exchange_rate
import psycopg2
import psycopg2.extras

# 設定日誌
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def get_db_conn():
    return psycopg2.connect(**DB_CONFIG)

def fetch_latest_predictions():
    """從資料庫讀取所有個股最新的預測軌跡。"""
    conn = get_db_conn()
    try:
        query = """
        WITH latest_predict AS (
            SELECT stock_id, MAX(predict_date) as max_pdate
            FROM stock_forecast_daily
            GROUP BY stock_id
        )
        SELECT 
            f.stock_id, f.predict_date, f.current_close::FLOAT as current_price, f.prob_up::FLOAT as prob_up,
            ((f.ensemble_price::FLOAT / NULLIF(f.current_close::FLOAT, 0)) - 1.0) as expected_return_mid,
            f.confidence_level
        FROM stock_forecast_daily f
        JOIN latest_predict l ON f.stock_id = l.stock_id AND f.predict_date = l.max_pdate
        WHERE f.day_offset = (SELECT MAX(day_offset) FROM stock_forecast_daily WHERE stock_id = f.stock_id AND predict_date = f.predict_date)
        """
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query)
            return cur.fetchall()
    finally:
        conn.close()

def get_liquidity_multiplier() -> float:
    """根據美元指數/匯率計算流動性修正因子。"""
    try:
        usd_df = load_exchange_rate()
        if usd_df.empty: return 1.0
        
        latest_usd = usd_df["usd_twd_mid"].iloc[-1]
        logger.info(f"偵測到當前美元匯率: {latest_usd:.2f}")
        
        # 邏輯：美金強於 32.2 代表流動性吃緊；低於 31.5 代表寬鬆
        if latest_usd > 32.2:
            return 0.85  # 壓低成長股權重
        elif latest_usd < 31.5:
            return 1.15  # 拉高成長股權重
        return 1.0
    except Exception as e:
        logger.warning(f"無法取得匯率資料進行流動性修正: {e}")
        return 1.0

def calculate_market_breadth(df_all):
    """
    計算全市場寬度指標 (Market Breadth)。
    市場邏輯：當超過 55% 的標的轉弱，代表多頭結構受損，應全面減碼避險。
    """
    if df_all.empty: return 1.0
    
    # 指標 1: 預測機率 > 0.5 的佔比
    bull_pct = (df_all["prob_up"] > 0.5).mean()
    
    # 指標 2: 預期報酬為正的佔比
    positive_ret_pct = (df_all["expected_return_mid"] > 0).mean()
    
    combined_breadth = (bull_pct + positive_ret_pct) / 2
    logger.info(f"📊 全市場寬度感應器: {combined_breadth:.1%} (多頭結構強度)")
    
    # 風險閘門邏輯
    if combined_breadth < 0.40:
        logger.warning("🚨 [崩盤模式] 市場寬度極差，觸發強制避險 (Haircut 80%)")
        return 0.2
    elif combined_breadth < 0.50:
        logger.warning("⚠️ [警示模式] 多空易位，減碼 50%")
        return 0.5
    return 1.0

def optimize_portfolio(df_all_stats, budget, max_stock_pct=0.05):
    """
    執行 20/60/20 資本過濾與不對稱槓鈴策略。
    
    1.  20/60/20 過濾：只保留表現最優異的前 20% 標的。
    2.  不對稱槓鈴：80% 配置於防禦端 (現金/中性)，20% 配置於攻擊端 (Top 20% 標的)。
    """
    if df_all_stats.empty: return pd.DataFrame()
    
    df = df_all_stats.copy()
    
    # [Step 1] 市場寬度感應與流動性修正
    risk_gate_mult = calculate_market_breadth(df)
    liq_mult = get_liquidity_multiplier()
    
    # [Step 2] 綜合評分 (第一性原則 + 時代溢價)
    confidence_map = {"🔥 強烈買進 (STRONG_BUY)": 2.0, "🟢 高信心": 1.5, "🟡 中等信心": 1.0, "🔴 低信心（建議觀望）": 0.2}
    df["conf_mult"] = df["confidence_level"].map(lambda x: confidence_map.get(x, 1.0))
    
    # 時代溢價 (MBNRIC 產業)
    mbnric_industries = ["Semiconductor", "AI_Hardware", "Biotech", "Robotics"]
    df["era_mult"] = df["industry"].apply(lambda x: 1.3 if x in mbnric_industries else 1.0)
    
    # 物理得分：機率 x 預期報酬 x 信心 x 時代
    df["quantum_score"] = (df["prob_up"] - 0.5) * df["expected_return_mid"] * df["conf_mult"] * df["era_mult"]
    
    # [Step 3] 20/60/20 資本過濾 (Pareto Filtering)
    # 依分數排序，只保留前 20% 的標的 (右側發光綠色區)
    df = df.sort_values("quantum_score", ascending=False)
    n_top = max(1, int(len(df) * 0.2))
    df_top_20 = df.head(n_top).copy()
    
    logger.info(f"🎯 [八二法則] 從 {len(df)} 檔中篩選出前 {n_top} 檔「發光綠色區」標的。")
    
    # [Step 4] 不對稱槓鈴分配 (Barbell Strategy)
    # 攻擊端 (Spear)：佔總預算 20%
    attack_budget_ratio = 0.20 * risk_gate_mult # 受風險閘門控管
    defense_budget_ratio = 1.0 - attack_budget_ratio
    
    if df_top_20.empty or df_top_20["quantum_score"].max() <= 0:
        logger.warning("⚠️ 無法找到符合攻擊條件的標的，全數退守防禦端。")
        return pd.DataFrame()

    # 攻擊端內的權重分配 (依分數比例)
    pos_scores = df_top_20["quantum_score"].clip(lower=0)
    if pos_scores.sum() > 0:
        df_top_20["attack_weight"] = pos_scores / pos_scores.sum()
    else:
        df_top_20["attack_weight"] = 1.0 / len(df_top_20)
        
    # 最終權重 = 攻擊預算 x 攻擊端內權重
    # 單一標的上限控管 (Anti-fragility)
    df_top_20["weight"] = (df_top_20["attack_weight"] * attack_budget_ratio).clip(upper=max_stock_pct)
    
    logger.info(f"🛡️ [槓鈴策略] 防禦端 (現金/安全資產): {defense_budget_ratio:.1%}, 攻擊端 (標的): {df_top_20['weight'].sum():.1%}")
    
    return df_top_20[df_top_20["weight"] > 0]

def main():
    parser = argparse.ArgumentParser(description="宏觀感知優化器 v2.1")
    parser.add_argument("--budget", type=float, default=100000)
    args = parser.parse_args()

    raw_preds = fetch_latest_predictions()
    if not raw_preds: return
    
    df_preds = pd.DataFrame(raw_preds)
    df_preds["industry"] = df_preds["stock_id"].apply(lambda x: STOCK_CONFIGS.get(x, {}).get("industry", "Unknown"))
    df_preds["name"] = df_preds["stock_id"].apply(lambda x: STOCK_CONFIGS.get(x, {}).get("name", "Unknown"))
    
    # 執行優化 (槓鈴策略)
    df_final = optimize_portfolio(df_preds, args.budget)
    
    # 輸出簡報
    defense_ratio = 1.0 - (df_final["weight"].sum() if not df_final.empty else 0.0)
    print("\n" + "═" * 60)
    print(f" 🌐 宏觀感知投資組合建議 ({datetime.now().strftime('%Y-%m-%d')})")
    print("═" * 60)
    print(f" 🛡️  防禦端 (現金/安全資產): {defense_ratio:>6.1%}")
    for _, row in df_final.sort_values("weight", ascending=False).iterrows():
        print(f"  {row['stock_id']:<6} {row['name']:<8} | 攻擊端權重: {row['weight']:>6.1%} | 產業: {row['industry']}")
    print("═" * 60 + "\n")

if __name__ == "__main__":
    main()
