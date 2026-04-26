"""
portfolio_strategy.py — 《2026 量子金融藍圖》槓鈴資產分配器
核心邏輯：80% 核心防禦 (Core) + 20% 量子進取 (Quantum)
"""

import logging
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# 導入專案路徑
sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.config import MODEL_DIR, STOCK_CONFIGS, get_all_features, ALL_FEATURES
from scripts.data_pipeline import build_daily_frame, _query
from scripts.feature_engineering import build_features
from scripts.signal_filter import SignalFilter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- 標的分類 (DNA Classification) ---
CORE_STOCKS = ["2330", "2317", "2881", "2882", "0050"] # 權值、高流動性、重力中心
QUANTUM_STOCKS = ["2454", "2376", "3661", "2382", "6669"] # 高驚奇值、高衝量、AI/技術驅動

class BarbellAllocator:
    def __init__(self, total_budget=1000000.0):
        self.total_budget = total_budget
        self.sf = SignalFilter()
        
        # 1. 執行戰略槓鈴分配 (Strategic Barbell)
        # 核心原則：80-90% 安全端 (Safety), 10-20% 凸性端 (Quantum)
        safety_ratio = 0.85 if market_entropy > 0.05 else 0.80
        quantum_total_ratio = 1.0 - safety_ratio
        
        safety_budget = self.total_budget * safety_ratio
        quantum_total_budget = self.total_budget * quantum_total_ratio
        
        logger.info(f">>> 執行戰略槓鈴 | 市場熵值: {market_entropy:.2f}")
        logger.info(f"    - 🛡️ 安全防禦端 (Safety): {safety_ratio:.0%} | 預算: {safety_budget:,.0f}")
        logger.info(f"    - 🚀 凸性量子端 (Quantum): {quantum_total_ratio:.0%} | 預算: {quantum_total_budget:,.0f}")
        
        final_allocation = []
        
        # 2. 核心端分配 (從安全預算中撥出少量進行權值股配置，其餘保留現金)
        # 核心端僅限於重力井極度穩定的標的
        core_picks = results[results['stock_id'].isin(CORE_STOCKS) & (results['prob_up'] >= 0.7)]
        if not core_picks.empty:
            # 即使是核心，也只動用安全預算的 30% (即總資產的 24%)，其餘 56% 必須是現金
            core_invest_budget = safety_budget * 0.3
            per_stock_budget = core_invest_budget / len(core_picks)
            for _, row in core_picks.iterrows():
                shares = int(per_stock_budget / row['price'])
                final_allocation.append({**row.to_dict(), "type": "CORE_STABLE", "shares": shares, "allocation": shares * row['price']})
        
        # 3. 量子凸性端分配 (10-20% 預算)
        # 嚴格篩選：必須具備高上漲機率 + 高凸性得分
        quantum_picks = results[results['stock_id'].isin(QUANTUM_STOCKS) & (results['prob_up'] >= 0.65)]
        if not quantum_picks.empty:
            # 尋找最具備「右側 20%」特質的標的
            per_stock_budget = quantum_total_budget / min(2, len(quantum_picks))
            for _, row in quantum_picks.head(2).iterrows():
                shares = int(per_stock_budget / row['price'])
                final_allocation.append({**row.to_dict(), "type": "QUANTUM_ALPHA", "shares": shares, "allocation": shares * row['price']})
                
        return pd.DataFrame(final_allocation)

def main():
    budget = 1000000.0 # 預設一百萬
    results = []
    
    # 合併所有核心與量子標的
    target_stocks = CORE_STOCKS + QUANTUM_STOCKS
    
    for sid in target_stocks:
        model_path = MODEL_DIR / f"ensemble_{sid}.pkl"
        if not model_path.exists(): continue
        
        try:
            model = joblib.load(model_path)
            raw = build_daily_frame(stock_id=sid, start_date="2024-01-01")
            df_feat = build_features(raw, stock_id=sid, for_inference=True)
            
            latest = df_feat.iloc[[-1]]
            pred_dict = model.predict(latest)
            prob_up = float(pred_dict["ensemble"][0])
            
            # 取得市場熵值 (作為風險度量)
            market_entropy = float(df_feat["market_entropy"].iloc[-1]) if "market_entropy" in df_feat.columns else 0.05
            
            results.append({
                "stock_id": sid,
                "name": STOCK_CONFIGS.get(sid, {}).get("name", sid),
                "price": float(df_feat["close"].iloc[-1]),
                "prob_up": prob_up,
                "entropy": market_entropy
            })
        except Exception as e:
            logger.error(f"標的 {sid} 預測失敗: {e}")

    if not results:
        logger.error("無可用預測。")
        return

    df_res = pd.DataFrame(results).sort_values("prob_up", ascending=False)
    
    # 執行分配
    allocator = BarbellAllocator(total_budget=budget)
    # 取平均熵值作為市場環境參考
    avg_entropy = df_res['entropy'].mean()
    final_df = allocator.allocate(df_res, market_entropy=avg_entropy)
    
    print("\n" + "="*80)
    print(f" 《2026 量子金融藍圖》槓鈴投資策略建議 (基準日: {datetime.now().strftime('%Y-%m-%d')})")
    print("="*80)
    print(f"{'類別':<8} {'代號':<6} {'名稱':<10} {'現價':>8} {'上漲機率':>10} {'建議股數':>10} {'配置金額':>12}")
    print("-" * 80)
    
    for _, row in final_df.iterrows():
        print(f"{row['type']:<8} {row['stock_id']:<6} {row['name']:<10} {row['price']:>8.2f} {row['prob_up']:>10.2%} {row['shares']:>10d} {row['allocation']:>12,.0f}")
    
    print("="*80)
    print(f"  總配置金額: {final_df['allocation'].sum():,.0f} / 現金保留: {budget - final_df['allocation'].sum():,.0f}")
    print("="*80)

if __name__ == "__main__":
    main()
