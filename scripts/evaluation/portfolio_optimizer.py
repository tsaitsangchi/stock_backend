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

class PortfolioOptimizer:
    def __init__(self, risk_free_rate=0.02):
        self.risk_free_rate = risk_free_rate
        self.confidence_map = {
            "🔥 強烈買進 (STRONG_BUY)": 2.0,
            "🟢 高信心": 1.5,
            "🟡 中等信心": 1.0,
            "🔴 低信心（建議觀望）": 0.2
        }
        self.mbnric_industries = ["Semiconductor", "AI_Hardware", "Biotech", "Robotics"]

    def optimize_v4(self, signals: dict, exchange_rate: float = 32.0) -> dict:
        """
        v4.0 核心優化邏輯：實踐不對稱槓鈴策略與 20/60/20 資本過濾。
        """
        if not signals:
            return {"CASH": 1.0}

        # 1. 轉換為 DataFrame 進行運算
        data = []
        for sid, s in signals.items():
            # 關鍵修正：只有被 SignalFilter 允許交易 (LONG) 的標的才能進入權重分配
            if s.get('decision') == 'LONG':
                industry = STOCK_CONFIGS.get(sid, {}).get("industry", "Unknown")
                era_mult = 1.3 if industry in self.mbnric_industries else 1.0
                conf_mult = self.confidence_map.get(s.get('confidence_level', "🟡 中等信心"), 1.0)
                
                data.append({
                    "stock_id": sid,
                    "prob_up": s['prob_up'],
                    "overall_score": (s.get('overall_score', 50.0) / 100.0),
                    "era_mult": era_mult,
                    "conf_mult": conf_mult,
                    "expected_return_mid": s.get('expected_return_mid', 0.05)
                })
        
        if not data:
            return {"CASH": 1.0}
            
        df = pd.DataFrame(data)
        
        # 2. 市場寬度
        risk_gate_mult = calculate_market_breadth(df)
        
        # 3. 量子得分 (Quantum Score)
        df["quantum_score"] = (df["prob_up"] - 0.5) * df["overall_score"] * df["era_mult"] * df["conf_mult"]
        
        # 4. 20/60/20 篩選
        df = df.sort_values("quantum_score", ascending=False)
        n_top = max(1, int(len(df) * 0.2))
        df_top = df.head(n_top).copy()
        
        # 5. 槓鈴分配
        attack_ratio = 0.20 * risk_gate_mult
        total_q = df_top["quantum_score"].clip(lower=1e-6).sum()
        
        weights = {}
        for _, row in df_top.iterrows():
            w = (row["quantum_score"] / total_q) * attack_ratio
            weights[row["stock_id"]] = round(float(w), 4)
            
        weights["CASH"] = round(1.0 - sum(weights.values()), 4)
        return weights

def optimize_portfolio(df_all_stats, budget, max_stock_pct=0.05):
    """舊版入口封裝 (Backward Compatibility)"""
    optimizer = PortfolioOptimizer()
    signals = {}
    for _, row in df_all_stats.iterrows():
        signals[row['stock_id']] = {
            'prob_up': row['prob_up'],
            'expected_return_mid': row.get('expected_return_mid', 0.05),
            'confidence_level': row.get('confidence_level', "🟡 中等信心"),
            'overall_score': row.get('overall_score', 80.0)
        }
    
    weights = optimizer.optimize_v4(signals)
    results = []
    for sid, w in weights.items():
        if sid != "CASH":
            results.append({"stock_id": sid, "weight": w})
    return pd.DataFrame(results)

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
