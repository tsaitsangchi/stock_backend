import os
import pandas as pd
import numpy as np
import logging
import sys
from datetime import datetime

# 導入專案路徑
sys.path.append(os.getcwd())
from scripts.config import OUTPUT_DIR, MODEL_DIR, CONFIDENCE_THRESHOLD

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self, stock_id="2330", initial_capital=1000000.0):
        self.stock_id = stock_id
        self.initial_capital = initial_capital
        self.transaction_cost = 0.004425 # 手續費(0.1425%) + 證交稅(0.3%)
        
    def run(self):
        logger.info(f"=== 啟動 {self.stock_id} 二十年長征回測 (第一性原則版) ===")
        
        # 1. 讀取 OOF 預測數據 (這是最真實的回測來源)
        # 如果有最新的 wf_fold_metrics.csv 與 oof_predictions，我們優先使用
        # 暫時假設我們從訓練日誌中產出的 OOF 序列中讀取
        # 為了演示，我們建立一個模擬回測邏輯，整合預測機率與實際價格
        
        from scripts.data_pipeline import build_daily_frame
        from scripts.feature_engineering import build_features
        
        logger.info(">>> 讀取全歷史數據 (2006-2026)...")
        raw = build_daily_frame(stock_id=self.stock_id, start_date="2006-01-01")
        df = build_features(raw, stock_id=self.stock_id, for_inference=True)
        
        # 2. 獲取模型預測 (在實際環境中，這應該是 Cross-Validation 的 OOF 結果)
        # 這裡我們模擬一個「決策序列」
        # 真正的回測應讀取訓練過程中產出的 oof_df.csv
        oof_path = os.path.join(OUTPUT_DIR, "oof_predictions.csv")
        if not os.path.exists(oof_path):
            logger.error("找不到 OOF 預測文件，請先執行完整訓練以產出回測序列。")
            return
            
        preds = pd.read_csv(oof_path, index_col=0, parse_dates=True)
        
        # 合併價格與預測
        bt_df = df[["close", "kwave_score"]].join(preds[["prob_up"]], how="inner")
        
        # 3. 模擬交易
        capital = self.initial_capital
        position = 0
        shares = 0
        equity_curve = []
        trades = []
        
        for i in range(len(bt_df)):
            date = bt_df.index[i]
            price = bt_df["close"].iloc[i]
            prob = bt_df["prob_up"].iloc[i]
            kwave = bt_df["kwave_score"].iloc[i]
            
            # 出場邏輯 (持有滿 30 天或機率跌破 0.4)
            if position > 0:
                entry_date, entry_price = trades[-1]["entry_date"], trades[-1]["entry_price"]
                days_held = (date - entry_date).days
                if days_held >= 30 or prob < 0.4:
                    capital = shares * price * (1 - 0.003) # 扣印花稅
                    trades[-1].update({
                        "exit_date": date,
                        "exit_price": price,
                        "profit": capital - trades[-1]["entry_capital"],
                        "return": (price / entry_price) - 1
                    })
                    position = 0
                    shares = 0
                    logger.info(f"[{date.date()}] 出場 | 價格: {price:.1f} | 持有: {days_held}天 | 收益: {trades[-1]['return']:.2%}")

            # 入場邏輯 (STRONG_BUY 且 康波不在冬天的極端區)
            if position == 0 and prob >= CONFIDENCE_THRESHOLD:
                # 第一性：康波過濾器 (如果 kwave_score 太高代表系統性風險極大，略過)
                if kwave > 1.5:
                    logger.warning(f"[{date.date()}] 訊號觸發但康波分數過高 ({kwave:.2f})，放棄入場。")
                    continue
                    
                shares = (capital * 0.95) / (price * 1.001425)
                entry_capital = capital
                trades.append({
                    "entry_date": date,
                    "entry_price": price,
                    "entry_capital": entry_capital,
                    "prob": prob,
                    "kwave": kwave
                })
                position = 1
                logger.info(f"[{date.date()}] 入場 | 價格: {price:.1f} | 機率: {prob:.2f} | 康波: {kwave:.2f}")

            current_equity = capital if position == 0 else shares * price
            equity_curve.append(current_equity)

        bt_df["equity"] = equity_curve
        
        # 4. 計算指標
        total_return = (equity_curve[-1] / self.initial_capital) - 1
        max_drawdown = (bt_df["equity"] / bt_df["equity"].cummax() - 1).min()
        win_rate = len([t for t in trades if t.get("profit", 0) > 0]) / len([t for t in trades if "exit_date" in t])
        
        print("\n" + "="*40)
        print(f"二十年回測報告 ({self.stock_id})")
        print("-" * 40)
        print(f"總報酬率: {total_return:.2%}")
        print(f"最大回測: {max_drawdown:.2%}")
        print(f"勝率: {win_rate:.2%}")
        print(f"交易次數: {len(trades)}")
        print("="*40)
        
        # 儲存結果
        bt_df.to_csv(os.path.join(OUTPUT_DIR, f"backtest_result_{self.stock_id}.csv"))
        logger.info(f"回測完成，結果已儲存至 outputs/backtest_result_{self.stock_id}.csv")

if __name__ == "__main__":
    engine = BacktestEngine(stock_id="2330")
    engine.run()
