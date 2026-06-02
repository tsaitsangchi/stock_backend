import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm

# Setup paths
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))

from scripts.config import OUTPUT_DIR, TIER_1_STOCKS, STOCK_CONFIGS, RISK_CONFIG
from scripts.pipeline.signal_filter import SignalFilter
from scripts.pipeline.portfolio_optimizer import optimize_portfolio
from scripts.pipeline.data_pipeline import load_stock_price, load_features_from_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class FaithfulBacktester:
    def __init__(self, initial_capital=10_000_000):
        self.initial_capital = initial_capital
        self.sf = SignalFilter()
        self.output_dir = OUTPUT_DIR

    def run(self, stock_ids: list[str], start_date="2022-01-01"):
        logger.info(f"🚀 啟動「真實生產級」組合回測 (共 {len(stock_ids)} 檔)")
        
        # 1. 載入資料
        all_oof = {}
        all_feat = {}
        all_prices = {}
        
        for sid in stock_ids:
            oof_path = self.output_dir / f"oof_predictions_{sid}.csv"
            logger.info(f"Checking path: {oof_path}")
            if not oof_path.exists():
                oof_path = self.output_dir / f"oof_predictions_partial_{sid}.csv"
                logger.info(f"Checking partial path: {oof_path}")
            
            if oof_path.exists():
                oof = pd.read_csv(oof_path, index_col=0, parse_dates=True)
                all_oof[sid] = oof
                
                # 載入特徵 (用於 SignalFilter)
                feat = load_features_from_store(sid)
                if not feat.empty:
                    all_feat[sid] = feat
                
                # 載入價格 (用於計算 P&L)
                price = load_stock_price(sid)
                if not price.empty:
                    all_prices[sid] = price['close']
            else:
                logger.warning(f"  找不到 {sid} 的預測資料，略過。")

        if not all_oof:
            logger.error("❌ 無可用預測資料，無法執行回測。")
            return

        # 2. 找出共同日期區間
        common_dates = None
        for sid, oof in all_oof.items():
            if common_dates is None:
                common_dates = oof.index
            else:
                common_dates = common_dates.intersection(oof.index)
        
        common_dates = common_dates[common_dates >= pd.to_datetime(start_date)].sort_values()
        logger.info(f"  回測區間: {common_dates[0].date()} ~ {common_dates[-1].date()} (共 {len(common_dates)} 天)")

        # 3. 逐日模擬
        portfolio_value = self.initial_capital
        equity_curve = []
        current_weights = pd.Series(0.0, index=stock_ids)
        
        for d in tqdm(common_dates, desc="Backtesting"):
            # A. 取得當日所有標的的訊號
            daily_signals = []
            for sid in all_oof.keys():
                if d not in all_oof[sid].index or d not in all_feat[sid].index:
                    continue
                
                prob_up = float(all_oof[sid].loc[d, 'prob_up'])
                
                # 構造模擬 Report 供 SignalFilter 使用
                # 注意：這裡簡化了 Report，主要提供機率和方向
                mock_report = {
                    "stock_id": sid,
                    "prob_up": prob_up,
                    "current_close": float(all_prices[sid].loc[d]),
                    "confidence": "🟢 高信心" if prob_up > 0.65 else "🟡 中等信心",
                    "model_agreement": 0.8,
                    "expected_return": {"mid": 0.05 if prob_up > 0.5 else -0.05},
                    "target_price": {"mid": float(all_prices[sid].loc[d]) * 1.05},
                    "warnings": {"macro_shock": False, "data_drift": False, "low_confidence": False}
                }
                
                # 使用真實 SignalFilter 進行過濾 (P2-4 核心)
                res = self.sf.evaluate(mock_report, all_feat[sid].loc[:d])
                
                if res.is_tradeable:
                    daily_signals.append({
                        "stock_id": sid,
                        "prob_up": prob_up,
                        "expected_return_mid": mock_report["expected_return"]["mid"],
                        "confidence_level": mock_report["confidence"],
                        "industry": STOCK_CONFIGS.get(sid, {}).get("industry", "Unknown")
                    })
                else:
                    if d == common_dates[0]: # 只印第一天的診斷
                         logger.info(f"  [Diagnostic] {sid} rejected: {res.blocking_reasons} | Score: {res.overall_score}")

            # B. 執行組合優化 (80/20 槓鈴策略)
            if daily_signals:
                df_signals = pd.DataFrame(daily_signals)
                optimized_df = optimize_portfolio(df_signals, budget=portfolio_value)
                
                new_weights = pd.Series(0.0, index=stock_ids)
                if not optimized_df.empty:
                    for _, row in optimized_df.iterrows():
                        new_weights[row['stock_id']] = row['weight']
            else:
                new_weights = pd.Series(0.0, index=stock_ids)

            # C. 計算當日 P&L
            # 假設以當日收盤價換倉（簡化），下一日計算報酬
            if len(equity_curve) > 0:
                daily_ret = 0.0
                for sid, w in current_weights.items():
                    if w > 0 and d in all_prices[sid].index:
                        # 取得前一交易日價格
                        prev_d = common_dates[common_dates.get_loc(d)-1]
                        ret = (all_prices[sid].loc[d] / all_prices[sid].loc[prev_d]) - 1
                        daily_ret += w * ret
                
                # 扣除交易成本 (換倉部分)
                turnover = (new_weights - current_weights).abs().sum() / 2.0
                tc = turnover * 0.003 # 預設 0.3%
                
                portfolio_value *= (1 + daily_ret - tc)
            
            current_weights = new_weights
            equity_curve.append({"date": d, "value": portfolio_value})

        # 4. 產出報告
        df_equity = pd.DataFrame(equity_curve).set_index("date")
        df_equity['ret'] = df_equity['value'].pct_change()
        
        total_ret = (df_equity['value'].iloc[-1] / self.initial_capital) - 1
        ann_ret = (1 + total_ret) ** (252 / len(df_equity)) - 1
        ann_vol = df_equity['ret'].std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        
        mdd = (df_equity['value'] / df_equity['value'].cummax() - 1).min()
        
        print("\n" + "═"*60)
        print(" 🏆 「量子金融 v4.0」真實組合回測報告 (Faithful)")
        print("═"*60)
        print(f" 總報酬率   : {total_ret*100:.2f}%")
        print(f" 年化報酬率 : {ann_ret*100:.2f}%")
        print(f" 年化波動度 : {ann_vol*100:.2f}%")
        print(f" 夏普比率   : {sharpe:.2f}")
        print(f" 最大回撤   : {mdd*100:.2f}%")
        print("═"*60)
        
        df_equity.to_csv(self.output_dir / "faithful_backtest_equity.csv")
        logger.info(f"回測結果已儲存至 {self.output_dir}/faithful_backtest_equity.csv")

if __name__ == "__main__":
    from scripts.config import LARGE_CAP_TICKERS
    tester = FaithfulBacktester()
    tester.run(LARGE_CAP_TICKERS)
