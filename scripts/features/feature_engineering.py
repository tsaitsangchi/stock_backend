"""
feature_engineering.py v6.5 (Quantum Finance Edition)
================================================================================
特徵工程核心引擎 — 物理資訊動力學版 (Quantum v5.1 標準)
此模組將市場視為物理系統，基於 $F = M \times a$ 理念轉化特徵。

核心功能：
  · 市場質量 (M)   ─ 滾動流動性估算 (Liquidity Mass)。
  · 資訊力 (F)     ─ 資訊衝擊帶來的價格位移。
  · 資訊力場 (Theta) ─ 系統不穩定性臨界指標。
  · 宏觀整合       ─ FRED 關鍵指標 (T10Y2Y, VIX)。

修訂歷程：
  v6.5 (2026-05-10): [核心] 導入物理資訊特徵 (Force, Mass, Theta)。
  v6.1 (2026-05-10): [核心] 整合 FRED 數據。
================================================================================
"""

import sys
import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path

# ── 系統路徑修復 (對接 path_setup v2.0) ──
try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
except ImportError:
    _THIS_DIR = Path(__file__).resolve().parent
    if str(_THIS_DIR) not in sys.path: sys.path.insert(0, str(_THIS_DIR))

try:
    from core.db_utils import write_pipeline_log, get_db_stock_ids, db_transaction
except ImportError as e:
    print(f"[FATAL] 無法匯入核心模組: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def save_features_to_db(stock_id: str, df: pd.DataFrame):
    """將特徵資料持久化至 features 表格"""
    with db_transaction() as cur:
        for date, row in df.iterrows():
            feature_cols = [c for c in row.index if c != "target_20d"]
            feature_json = row[feature_cols].to_json()
            target = float(row["target_20d"]) if "target_20d" in row and not pd.isna(row["target_20d"]) else 0.0
            
            cur.execute('''
                INSERT INTO features (stock_id, date, feature_data, target_value)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (stock_id, date) DO UPDATE 
                SET feature_data = EXCLUDED.feature_data, 
                    target_value = EXCLUDED.target_value
            ''', (stock_id, date, feature_json, target))

def fetch_raw_data(stock_id: str) -> pd.DataFrame:
    """從資料庫抓取多維度原始數據"""
    with db_transaction() as cur:
        # 1. 價量
        cur.execute("SELECT date, open, max as high, min as low, close, trading_volume as volume FROM stock_price WHERE stock_id = %s ORDER BY date ASC", (stock_id,))
        df_price = pd.DataFrame(cur.fetchall())
        if df_price.empty: return pd.DataFrame()
        df_price['date'] = pd.to_datetime(df_price['date'])
        df_price.set_index('date', inplace=True)
        df_price = df_price.astype(float)
        
        # 2. 籌碼 (法人持股比)
        cur.execute("SELECT date, percent as inst_percent FROM shareholding WHERE stock_id = %s AND holdclass = 'Institutional' ORDER BY date ASC", (stock_id,))
        df_chip = pd.DataFrame(cur.fetchall())
        if not df_chip.empty:
            df_chip['date'] = pd.to_datetime(df_chip['date'])
            df_chip.set_index('date', inplace=True)
            df_price = df_price.join(df_chip.astype(float), how='left')
            
        # 3. 營收
        cur.execute("SELECT date, revenue FROM month_revenue WHERE stock_id = %s ORDER BY date ASC", (stock_id,))
        df_rev = pd.DataFrame(cur.fetchall())
        if not df_rev.empty:
            df_rev['date'] = pd.to_datetime(df_rev['date'])
            df_rev.set_index('date', inplace=True)
            df_price = df_price.join(df_rev.astype(float), how='left')
            
    df_price.ffill(inplace=True)
    return df_price

def fetch_macro_data() -> pd.DataFrame:
    """抓取 FRED 總經數據"""
    with db_transaction() as cur:
        cur.execute("SELECT date, series_id, value FROM fred_series WHERE series_id IN ('T10Y2Y', 'BAMLH0A0HYM2', 'UNRATE', 'VIXCLS')")
        df = pd.DataFrame(cur.fetchall())
        if df.empty: return pd.DataFrame()
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.pivot(index='date', columns='series_id', values='value').astype(float)
        df.ffill(inplace=True)
        return df

def build_features(df_raw: pd.DataFrame, stock_id: str, df_macro: pd.DataFrame = None, for_inference: bool = False) -> pd.DataFrame:
    """
    特徵生成核心 (v6.5 Quantum Edition)
    """
    t0 = time.monotonic()
    if df_raw is None or df_raw.empty:
        df_raw = fetch_raw_data(stock_id)
        
    if df_raw.empty: return pd.DataFrame()

    try:
        df = df_raw.copy()
        
        # ── Group A: 量子物理特徵 ($F = M \times a$) ──
        # 1. 質量 (Mass): 20 日滾動平均成交量 (對數化)
        df["mass_M"] = np.log1p(df["volume"].rolling(20).median())
        
        # 2. 位移 (Displacement): 價格自然對數差值
        df["log_ret"] = np.log(df["close"]).diff()
        
        # 3. 資訊力 (Force): $F = M \times \Delta \ln P$
        df["force_F"] = df["mass_M"] * df["log_ret"]
        
        # 4. 資訊力場 (Theta): 短期力與長期波動之比 (系統不穩定性)
        vol_60 = df["log_ret"].rolling(60).std()
        df["force_theta"] = df["force_F"].abs() / (vol_60 + 1e-9)
        
        # ── Group B: 傳統動能與乖離 ──
        df["ma_20"] = df["close"].rolling(20).mean()
        df["ma_60"] = df["close"].rolling(60).mean()
        df["gravity_well"] = (df["close"] - df["ma_20"]) / df["ma_20"] # 重力井偏離
        
        # ── Group C: 籌碼與營收 ──
        if "inst_percent" in df.columns:
            df["inst_mom"] = df["inst_percent"].diff(5)
        if "revenue" in df.columns:
            df["rev_yoy"] = df["revenue"].pct_change(12)
            
        # ── Group D: 宏觀環境 ──
        if df_macro is not None and not df_macro.empty:
            df = df.join(df_macro, how='left').ffill()
        
        # 清理無效資料
        df.dropna(subset=["ma_60", "force_theta"], inplace=True)
        
        # 目標變數 (未來 20 日報酬)
        if not for_inference:
            df["target_20d"] = df["close"].shift(-20) / df["close"] - 1
            save_features_to_db(stock_id, df.dropna(subset=["target_20d"]))
        
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        write_pipeline_log("feature_engineering", stock_id, "success", "feature", elapsed_ms, len(df))
        return df
        
    except Exception as e:
        logger.error(f"❌ {stock_id} 特徵工程失敗: {e}")
        write_pipeline_log("feature_engineering", stock_id, "failed", "feature", 0, 0, str(e))
        return pd.DataFrame()

def run_all_feature_engineering():
    target_ids = get_db_stock_ids()
    logger.info(f"🚀 [Feature] 啟動量子全量特徵工程 (共 {len(target_ids)} 檔)...")
    df_macro = fetch_macro_data()
    success_count = 0
    t_start = time.monotonic()
    
    for sid in target_ids:
        res = build_features(None, sid, df_macro=df_macro)
        if not res.empty: success_count += 1
        
    duration = int(time.monotonic() - t_start)
    logger.info(f"🏁 [Feature] 任務完成！成功: {success_count}/{len(target_ids)}，總耗時: {duration}s")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock_id", type=str)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    
    if args.all:
        run_all_feature_engineering()
    else:
        sid = args.stock_id or "2330"
        df_macro = fetch_macro_data()
        res = build_features(None, sid, df_macro=df_macro)
        if not res.empty:
            print(f"\n[Quantum 特徵預覽] {sid}")
            print(res.tail(5)[["close", "force_F", "force_theta", "gravity_well", "target_20d"]])
