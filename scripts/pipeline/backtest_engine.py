"""
backtest_engine.py — 第一性原則長波段回測引擎 (整合 Core v3.0)
============================================================
整合了 path_setup (環境隔離) 與 model_metadata (版控追蹤)。
確保留單邏輯與摩擦成本計算符合真實交易環境。
"""

import os
import pandas as pd
import logging
from datetime import datetime

# ─────────────────────────────────────────────
# 1. 統一的環境與路徑初始化 (使用 core.path_setup)
# ─────────────────────────────────────────────
# 只需要這兩行，就能取代原本重複 3 次的 sys.path.append 邏輯
from core.path_setup import ensure_scripts_on_path, get_outputs_dir, get_archive_dir
ensure_scripts_on_path(__file__)

# 初始化動態路徑
OUTPUT_DIR = get_outputs_dir(__file__)
ARCHIVE_DIR = get_archive_dir(__file__)

# 載入專案配置與其他核心模組
from config import CONFIDENCE_THRESHOLD, FRICTION_CONFIG
from core.model_metadata import load_latest_metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 交易摩擦成本結構 (買出: 手續費+撇點 | 賣出: 手續費+證交稅+撇點)
BUY_COST_RATE  = 1 + FRICTION_CONFIG.get("commission", 0.001425)
SELL_COST_RATE = 1 - (FRICTION_CONFIG.get("commission", 0.001425) + FRICTION_CONFIG.get("securities_tax", 0.003))

class BacktestEngine:
    def __init__(self, stock_id="2330", initial_capital=1000000.0):
        self.stock_id = stock_id
        self.initial_capital = initial_capital
        
    def run(self):
        logger.info(f"=== 啟動 {self.stock_id} 二十年長征回測 (第一性原則版) ===")
        
        # ─────────────────────────────────────────────
        # 2. 聯動 Model Metadata (確保回測對齊模型版本)
        # ─────────────────────────────────────────────
        metadata = load_latest_metadata(self.stock_id, str(ARCHIVE_DIR))
        if metadata:
            logger.info(f"[詮釋資料] 關聯模型版本: {metadata.timestamp} (Git: {metadata.git_hash})")
            logger.info(f"[詮釋資料] 訓練期預期 OOF 績效 -> DA: {metadata.oof_da:.4f} | Sharpe: {metadata.oof_sharpe:.4f}")
        else:
            logger.warning("[詮釋資料] 找不到該標的之歷史模型封存紀錄，將進行無版控盲測。")

        # ─────────────────────────────────────────────
        # 3. 讀取全歷史數據與特徵
        # ─────────────────────────────────────────────
        from scripts.data_pipeline import build_daily_frame
        from scripts.feature_engineering import build_features
        
        logger.info(">>> 讀取全歷史數據與構建特徵...")
        raw = build_daily_frame(stock_id=self.stock_id, start_date="2006-01-01")
        df = build_features(raw, stock_id=self.stock_id, for_inference=True)
        
        # ─────────────────────────────────────────────
        # 4. 獲取模型 OOF 預測 (Out-Of-Fold)
        # ─────────────────────────────────────────────
        oof_path = OUTPUT_DIR / "oof_predictions.csv"
        if not oof_path.exists():
            logger.error(f"找不到 OOF 預測文件: {oof_path}，請先執行模型訓練。")
            return
            
        preds = pd.read_csv(oof_path, index_col=0, parse_dates=True)
        
        # 合併價格與預測 (Inner Join 確保對齊)
        bt_df = df[["close", "kwave_score"]].join(preds[["prob_up"]], how="inner")
        if bt_df.empty:
            logger.error("數據合併後為空，請檢查時間軸索引。")
            return
            
        # ─────────────────────────────────────────────
        # 5. 執行模擬交易
        # ─────────────────────────────────────────────
        capital = self.initial_capital
        position = 0
        shares = 0
        equity_curve = []
        trades = []
        
        logger.info(">>> 開始模擬交易迭代...")
        for date, row in bt_df.iterrows():
            price = row["close"]
            prob = row["prob_up"]
            kwave = row["kwave_score"]
            
            # [出場邏輯] 持有滿 30 天或機率跌破 0.4
            if position > 0:
                entry_date = trades[-1]["entry_date"]
                entry_price = trades[-1]["entry_price"]
                days_held = (date - entry_date).days
                
                if days_held >= 30 or prob < 0.4:
                    capital = shares * price * SELL_COST_RATE
                    gross_ret = (price / entry_price) - 1
                    net_ret = gross_ret - (FRICTION_CONFIG.get("commission", 0.001425) * 2 + FRICTION_CONFIG.get("securities_tax", 0.003))
                    
                    trades[-1].update({
                        "exit_date": date,
                        "exit_price": price,
                        "profit": capital - trades[-1]["entry_capital"],
                        "return": gross_ret,
                        "net_return": net_ret,
                    })
                    position = 0
                    shares = 0
                    logger.info(
                        f"[{date.date()}] 出場 | 價格: {price:.1f} | 持有: {days_held:2d}天 "
                        f"| 毛利: {gross_ret:6.2%} | 淨利: {net_ret:6.2%}"
                    )

            # [入場邏輯] 空倉 + 機率過閥值 + 康波濾網
            if position == 0 and prob >= CONFIDENCE_THRESHOLD:
                if kwave > 1.5:
                    logger.debug(f"[{date.date()}] 訊號觸發但康波分數過高 ({kwave:.2f})，放棄入場。")
                    continue

                shares = (capital * 0.95) / (price * BUY_COST_RATE)
                trades.append({
                    "entry_date": date,
                    "entry_price": price,
                    "entry_capital": capital,
                    "prob": prob,
                    "kwave": kwave,
                })
                position = 1
                logger.info(f"[{date.date()}] 入場 | 價格: {price:.1f} | 機率: {prob:.2f} | 康波: {kwave:.2f}")

            # 記錄每日權益
            current_equity = capital if position == 0 else shares * price
            equity_curve.append(current_equity)

        bt_df["equity"] = equity_curve
        
        # ─────────────────────────────────────────────
        # 6. 計算績效指標與儲存
        # ─────────────────────────────────────────────
        total_return = (equity_curve[-1] / self.initial_capital) - 1
        max_drawdown = (bt_df["equity"] / bt_df["equity"].cummax() - 1).min()
        completed_trades = [t for t in trades if "exit_date" in t]
        win_rate = len([t for t in completed_trades if t.get("profit", 0) > 0]) / len(completed_trades) if completed_trades else 0
        
        print("\n" + "="*45)
        print(f" 📊 二十年回測報告 ({self.stock_id})")
        print("-" * 45)
        print(f" 初始資金 : {self.initial_capital:,.0f}")
        print(f" 最終資金 : {equity_curve[-1]:,.0f}")
        print(f" 總報酬率 : {total_return:+.2%}")
        print(f" 最大回測 : {max_drawdown:.2%}")
        print(f" 勝率     : {win_rate:.2%}")
        print(f" 交易次數 : {len(completed_trades)}")
        print("="*45 + "\n")
        
        # 動態寫入統一的 outputs 目錄
        result_path = OUTPUT_DIR / f"backtest_result_{self.stock_id}.csv"
        bt_df.to_csv(result_path)
        logger.info(f"✅ 回測完成，明細已儲存至: {result_path}")

if __name__ == "__main__":
    engine = BacktestEngine(stock_id="2330")
    engine.run()