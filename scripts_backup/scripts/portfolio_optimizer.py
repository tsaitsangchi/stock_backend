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

def optimize_portfolio(df_candidates, df_all_stats, budget, max_industry_pct=0.4, max_stock_pct=0.25):
    """執行資產配置優化。"""
    df = df_candidates.copy()
    
    # 1. 取得市場寬度乘數 (Risk Gate)
    risk_gate_mult = calculate_market_breadth(df_all_stats)
    
    df["raw_score"] = (df["prob_up"] - 0.5) * df["expected_return_mid"]
    
    # 1. 信心等級加權
    confidence_map = {"高信心 (Green)": 1.5, "中信心 (Yellow)": 1.0, "低信心 (Red)": 0.3}
    df["weight_mult"] = df["confidence_level"].map(lambda x: confidence_map.get(x, 1.0))
    
    # 2. 康波週期時代溢價 (K-Wave)
    def get_era_multiplier(industry):
        if industry in ["AI_Hardware", "Semiconductor"]: return 1.2
        return 1.0
    df["era_mult"] = df["industry"].apply(get_era_multiplier)
    
    # 3. 美元流動性修正
    liq_mult = get_liquidity_multiplier()
    # 只有成長型產業受流動性因子顯著影響
    df["liq_mult"] = df["industry"].apply(lambda x: liq_mult if x in ["AI_Hardware", "Semiconductor"] else 1.0)
    
    df["score"] = df["raw_score"] * df["weight_mult"] * df["era_mult"] * df["liq_mult"]
    
    df = df[df["score"] > 0].sort_values("score", ascending=False)
    if df.empty: return pd.DataFrame()

    # 貪婪分配法
    df["weight"] = 0.0
    industry_usage = {ind: 0.0 for ind in df["industry"].unique()}
    total_allocated = 0.0
    scores_sum = df["score"].sum()
    df["ideal_weight"] = df["score"] / scores_sum
    
    for idx, row in df.iterrows():
        ind = row["industry"]
        actual_w = min(row["ideal_weight"], max_stock_pct, max(0, max_industry_pct - industry_usage[ind]))
        df.at[idx, "weight"] = actual_w
        industry_usage[ind] += actual_w
        total_allocated += actual_w
        if total_allocated >= 1.0: break
            
    # 應用風險閘門 (Global Exposure Control)
    df["weight"] = df["weight"] * risk_gate_mult
    
    if df["weight"].sum() > 0:
        # 在風險閘門範圍內進行歸一化 (若沒觸發避險則維持 100%)
        # 若觸發避險，則總權重會低於 1.0，剩餘為現金
        pass 
        
    return df[df["weight"] > 0]

def main():
    parser = argparse.ArgumentParser(description="宏觀感知優化器 v2.1")
    parser.add_argument("--budget", type=float, default=100000)
    args = parser.parse_args()

    raw_preds = fetch_latest_predictions()
    if not raw_preds: return
    
    df_preds = pd.DataFrame(raw_preds)
    df_preds["industry"] = df_preds["stock_id"].apply(lambda x: STOCK_CONFIGS.get(x, {}).get("industry", "Unknown"))
    df_preds["name"] = df_preds["stock_id"].apply(lambda x: STOCK_CONFIGS.get(x, {}).get("name", "Unknown"))
    
    # 傳入全市場資料進行寬度計算，並過濾出可交易標的
    df_final = optimize_portfolio(df_preds[df_preds["prob_up"] >= 0.55], df_preds, args.budget)
    
    # 輸出簡報 (僅顯示權重與宏觀狀態)
    print("\n" + "═" * 60)
    print(f" 🌐 宏觀感知投資組合建議 ({datetime.now().strftime('%Y-%m-%d')})")
    print("═" * 60)
    for _, row in df_final.sort_values("weight", ascending=False).iterrows():
        print(f"  {row['stock_id']:<6} {row['name']:<8} | 權重: {row['weight']:>6.1%} | 產業: {row['industry']}")
    print("═" * 60 + "\n")

if __name__ == "__main__":
    main()
