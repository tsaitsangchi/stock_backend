"""
portfolio_strategy.py — 針對 30,000 TWD 的最佳投資建議
"""

import logging
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from config import MODEL_DIR, STOCK_CONFIGS, get_all_features, ALL_FEATURES
from data_pipeline import build_daily_frame, _query
from feature_engineering import build_features

logger = logging.getLogger(__name__)

def load_model(stock_id: str):
    path = MODEL_DIR / f"ensemble_{stock_id}.pkl"
    if not path.exists():
        return None
    return joblib.load(path)

def get_latest_price(stock_id: str):
    query_str = f"SELECT close FROM stock_price WHERE stock_id='{stock_id}' ORDER BY date DESC LIMIT 1"
    df = _query(query_str)
    if df.empty:
        return None
    return float(df['close'].iloc[0])

def main():
    logging.basicConfig(level=logging.INFO)
    budget = 30000
    results = []
    
    # 預計涵蓋的股票 (已有模型的)
    stock_ids = ["2330", "2317", "2454", "2881", "2382", "2603", "2002", "3037", "2308", "6669", "9958"]
    
    logger.info(f"正在為 {len(stock_ids)} 支股票生成預測...")
    
    for sid in stock_ids:
        model = load_model(sid)
        if model is None:
            continue
            
        try:
            # 取得特徵 (限制回測長度以加速)
            raw = build_daily_frame(stock_id=sid, start_date="2023-01-01")
            df_feat = build_features(raw, stock_id=sid, for_inference=True)
            
            all_feats_sid = get_all_features(sid)
            feat_cols = [c for c in all_feats_sid if c in df_feat.columns]
            X_latest = df_feat[feat_cols].fillna(0).iloc[[-1]]
            
            # 預測
            pred_dict = model.predict(X_latest)
            prob_up = float(pred_dict["ensemble"][0])
            
            current_price = float(df_feat["close"].iloc[-1])
            
            results.append({
                "stock_id": sid,
                "name": STOCK_CONFIGS.get(sid, {}).get("name", sid),
                "price": current_price,
                "prob_up": prob_up,
            })
        except Exception as e:
            logger.error(f"股票 {sid} 預測失敗: {e}")

    # 排序：按上漲機率
    if not results:
        logger.error("沒有任何股票成功生成預測。請檢查模型文件是否存在且特徵工程是否正常。")
        return

    df_res = pd.DataFrame(results).sort_values("prob_up", ascending=False)
    
    print("\n" + "="*70)
    print(f"  30,000 TWD 投資策略建議 (基準日: {datetime.now().strftime('%Y-%m-%d')})")
    print("="*70)
    print(f"{'代號':<6} {'名稱':<10} {'現價':>8} {'上漲機率':>10} {'建議股數(零股)':>15}")
    print("-"*70)
    
    for _, row in df_res.head(5).iterrows():
        # 簡單策略：如果是高信心股票，分配約 1/3 資金
        # 由於資金只有 3 萬，建議分散在 2-3 支股票，或集中在最強的一支
        shares = int(budget / 3 / row['price']) if row['prob_up'] > 0.55 else 0
        if shares == 0 and row['prob_up'] > 0.55:
            shares = int(budget / row['price']) # 集中投資
            
        print(f"{row['stock_id']:<6} {row['name']:<10} {row['price']:>8.2f} {row['prob_up']:>10.2%} {shares:>15d}")

    print("\n* 註：由於資金為 3 萬元，建議透過「盤中零股」交易。")
    print("  推薦優先關注上漲機率 > 55% 且信心度高的個股。")
    print("="*70)

if __name__ == "__main__":
    main()
