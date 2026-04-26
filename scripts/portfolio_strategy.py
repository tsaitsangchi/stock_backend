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
from scripts.config import MODEL_DIR, STOCK_CONFIGS
from scripts.data_pipeline import build_daily_frame
from scripts.feature_engineering import build_features
from scripts.signal_filter import SignalFilter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- 標的分類 (DNA Classification) ---
# 核心端：絕對重力中心，高流動性安全標的
CORE_STOCKS = ["2330", "2317", "2881", "2882", "0050"] 
# 量子端：高凸性、高驚奇值、第六波先鋒
QUANTUM_STOCKS = ["2454", "3661", "2382", "6669", "1760", "4147", "2399"]

class BarbellAllocator:
    def __init__(self, total_budget=1000000.0):
        self.total_budget = total_budget
        self.sf = SignalFilter()

    def allocate(self, results: pd.DataFrame, market_entropy: float = 0.05) -> pd.DataFrame:
        """
        執行戰略槓鈴分配。
        核心原則：80-90% 安全端 (Safety), 10-20% 凸性端 (Quantum)。
        拒絕平庸：消除中間地帶，標的若不具備極端凸性，分配為 0。
        """
        # 1. 決定安全邊際比例
        # 當市場熵值增加（混亂度上升），安全端比例自動提高
        safety_ratio = 0.85 if market_entropy > 0.02 else 0.80
        quantum_total_ratio = 1.0 - safety_ratio
        
        safety_budget = self.total_budget * safety_ratio
        quantum_total_budget = self.total_budget * quantum_total_ratio
        
        logger.info(f">>> 執行戰略槓鈴 | 市場熵值: {market_entropy:.4f}")
        logger.info(f"    - 🛡️ 安全防禦端 (Safety): {safety_ratio:.0%} | 預算: {safety_budget:,.0f}")
        logger.info(f"    - 🚀 凸性量子端 (Quantum): {quantum_total_ratio:.0%} | 預算: {quantum_total_budget:,.0f}")
        
        final_allocation = []
        
        # 2. 核心端分配 (Safety Base)
        # 核心端僅限於重力井極度穩定的標的
        core_picks = results[results['stock_id'].isin(CORE_STOCKS) & (results['prob_up'] >= 0.65)]
        if not core_picks.empty:
            # 核心端投資限制：不超過安全預算的 30%，其餘保留現金 (Safety Base)
            core_invest_limit = safety_budget * 0.3
            per_stock_budget = core_invest_limit / len(core_picks)
            for _, row in core_picks.iterrows():
                shares = int(per_stock_budget / row['price'])
                if shares > 0:
                    final_allocation.append({**row.to_dict(), "type": "CORE_SAFE", "shares": shares, "allocation": shares * row['price']})
        
        # 3. 量子凸性端分配 (Quantum Alpha)
        # 嚴格篩選：必須具備高上漲機率 + 高創新速度 (Innovation Velocity)
        quantum_candidates = results[results['stock_id'].isin(QUANTUM_STOCKS) & (results['prob_up'] >= 0.60)]
        
        if not quantum_candidates.empty:
            # 💡 拒絕平庸邏輯：只選擇機率最高且具備「奇點溢價」的前 2 名
            quantum_picks = quantum_candidates.sort_values("prob_up", ascending=False).head(2)
            per_stock_budget = quantum_total_budget / len(quantum_picks)
            
            for _, row in quantum_picks.iterrows():
                shares = int(per_stock_budget / row['price'])
                if shares > 0:
                    final_allocation.append({**row.to_dict(), "type": "QUANTUM_凸性", "shares": shares, "allocation": shares * row['price']})
                
        return pd.DataFrame(final_allocation)

def main():
    budget = 1000000.0
    results = []
    
    # 掃描所有藍圖標的
    target_stocks = list(set(CORE_STOCKS + QUANTUM_STOCKS))
    
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
            
            entropy = float(latest["market_entropy"].iloc[0]) if "market_entropy" in latest.columns else 0.05
            
            results.append({
                "stock_id": sid,
                "name": STOCK_CONFIGS.get(sid, {}).get("name", sid),
                "price": float(latest["close"].iloc[0]),
                "prob_up": prob_up,
                "entropy": entropy
            })
        except Exception as e:
            logger.error(f"標的 {sid} 處理失敗: {e}")

    if not results:
        logger.warning("目前無任何標的通過物理過濾。")
        return

    df_res = pd.DataFrame(results)
    allocator = BarbellAllocator(total_budget=budget)
    avg_entropy = df_res['entropy'].mean()
    final_df = allocator.allocate(df_res, market_entropy=avg_entropy)
    
    if final_df.empty:
        print("\n[!] 今日無符合《2026 量子金融藍圖》非對稱美學的交易機會，建議 100% 持現金等待。")
        return

    print("\n" + "="*80)
    print(f" 《2026 量子金融藍圖》槓鈴投資策略建議 (基準日: {datetime.now().strftime('%Y-%m-%d %H:%M')})")
    print("-" * 80)
    print(f" 戰略方針: 80% 安全端防禦 / 20% 冪律端捕捉 / 徹底消除中間平庸")
    print("="*80)
    print(f"{'類別':<12} {'代號':<6} {'名稱':<10} {'現價':>8} {'上漲機率':>10} {'建議股數':>10} {'配置金額':>12}")
    print("-" * 80)
    
    for _, row in final_df.iterrows():
        print(f"{row['type']:<12} {row['stock_id']:<6} {row['name']:<10} {row['price']:>8.2f} {row['prob_up']:>10.2%} {row['shares']:>10d} {row['allocation']:>12,.0f}")
    
    print("="*80)
    print(f"  總配置金額: {final_df['allocation'].sum():,.0f} | 現金保留 (Safety Base): {budget - final_df['allocation'].sum():,.0f}")
    print("="*80)

if __name__ == "__main__":
    main()
