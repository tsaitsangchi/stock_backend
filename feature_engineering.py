"""
feature_engineering.py — 特徵工程層
依第一性原則 5 大類特徵，基於 build_daily_frame() 的輸出建構模型輸入。

輸出：pd.DataFrame，新增 ~100 個特徵欄
   + target_30d（回歸）+ direction_30d（分類）+ target_binary（二元）
   + 多時程目標：target_15d / target_21d / target_30d（漲 X% 的二元分類）
   + 趨勢 Regime 標籤：trend_regime（bull/bear/sideways）
   + 中期信號特徵（由 build_medium_term_features 計算後透過 build_features_with_medium_term 整合）
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

try:
    from config import FEATURE_GROUPS, HORIZON, STOCK_CONFIGS, DEFAULT_STOCK_ID
except ImportError:
    from scripts.config import FEATURE_GROUPS, HORIZON, STOCK_CONFIGS, DEFAULT_STOCK_ID

logger = logging.getLogger(__name__)

# ── 技術指標套件：依優先順序嘗試 ─────────────────────────────
# 1. pandas-ta（若已安裝）
# 2. ta（pip install ta，純 Python，支援 Python 3.14+）
# 3. 手動計算 fallback（不需任何額外套件）
HAS_PANDAS_TA = False
HAS_TA        = False

try:
    import pandas_ta as _pta  # type: ignore
    HAS_PANDAS_TA = True
    logger.info("技術指標：使用 pandas-ta")
except Exception:
    pass

if not HAS_PANDAS_TA:
    try:
        import ta as _ta_lib
        HAS_TA = True
        logger.info("技術指標：使用 ta 套件（pandas-ta 不可用）")
    except ImportError:
        logger.info("技術指標：使用手動計算 fallback（pip install ta 可加速）")


# ─────────────────────────────────────────────
# 技術動能特徵
# ─────────────────────────────────────────────

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"].copy()
    v = df["volume"].copy()

    # 對數報酬率
    log_r = np.log(c / c.shift(1))
    for n in [1, 5, 10, 20]:
        df[f"log_return_{n}d"] = np.log(c / c.shift(n))

    # 已實現波動率
    for n in [10, 20, 60]:
        df[f"realized_vol_{n}d"] = log_r.rolling(n).std() * np.sqrt(252)

    # 移動平均
    for n in [20, 50, 120]:
        df[f"ma_{n}"] = c.rolling(n).mean()
        df[f"price_to_ma{n}"] = c / df[f"ma_{n}"] - 1

    # 均線交叉
    df["ma_cross_20_50"]  = df["ma_20"] / df["ma_50"] - 1
    df["ma_cross_50_120"] = df["ma_50"] / df["ma_120"] - 1

    if HAS_PANDAS_TA:
        # ── pandas-ta ──────────────────────────────────────────
        import pandas_ta as _pta  # type: ignore
        df["rsi_14"]    = _pta.rsi(c, length=14)
        df["rsi_28"]    = _pta.rsi(c, length=28)
        macd_df = _pta.macd(c, fast=12, slow=26, signal=9)
        if macd_df is not None:
            df["macd"]        = macd_df.iloc[:, 0]
            df["macd_signal"] = macd_df.iloc[:, 2]
            df["macd_hist"]   = macd_df.iloc[:, 1]
        bb = _pta.bbands(c, length=20, std=2)
        if bb is not None:
            df["bb_upper"] = bb.iloc[:, 2]
            df["bb_lower"] = bb.iloc[:, 0]
            df["bb_pct"]   = bb.iloc[:, 1]
        df["atr_14"] = _pta.atr(df["high"], df["low"], c, length=14)

    elif HAS_TA:
        # ── ta 套件（pip install ta，支援 Python 3.14+）─────────
        import ta as _ta_lib
        df["rsi_14"]    = _ta_lib.momentum.RSIIndicator(c, window=14).rsi()
        df["rsi_28"]    = _ta_lib.momentum.RSIIndicator(c, window=28).rsi()
        _macd = _ta_lib.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
        df["macd"]        = _macd.macd()
        df["macd_signal"] = _macd.macd_signal()
        df["macd_hist"]   = _macd.macd_diff()
        _bb = _ta_lib.volatility.BollingerBands(c, window=20, window_dev=2)
        df["bb_upper"]  = _bb.bollinger_hband()
        df["bb_lower"]  = _bb.bollinger_lband()
        df["bb_pct"]    = _bb.bollinger_pband()
        df["atr_14"]    = _ta_lib.volatility.AverageTrueRange(
            df["high"], df["low"], c, window=14
        ).average_true_range()

    else:
        # ── 手動計算 fallback（無額外依賴）───────────────────────
        # RSI
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        df["rsi_14"] = 100 - 100 / (1 + rs)
        # RSI-28（用指數移動平均，更接近 Wilder 原始公式）
        gain28 = delta.clip(lower=0).ewm(alpha=1/28, adjust=False).mean()
        loss28 = (-delta.clip(upper=0)).ewm(alpha=1/28, adjust=False).mean()
        rs28   = gain28 / loss28.replace(0, np.nan)
        df["rsi_28"] = 100 - 100 / (1 + rs28)
        # MACD
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        df["macd"]        = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"]   = df["macd"] - df["macd_signal"]
        # Bollinger Bands
        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        df["bb_upper"] = sma20 + 2 * std20
        df["bb_lower"] = sma20 - 2 * std20
        df["bb_pct"]   = (c - df["bb_lower"]) / (
            (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
        )
        # ATR（Wilder 平滑）
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - c.shift(1)).abs(),
            (df["low"]  - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        df["atr_14"] = tr.ewm(alpha=1/14, adjust=False).mean()

    # 成交量特徵
    df["volume_ma_20"]    = v.rolling(20).mean()
    df["volume_ratio_20"] = v / df["volume_ma_20"]

    # 量價相關
    temp = pd.concat([log_r.rename("ret"), v.pct_change().rename("vol_chg")], axis=1)
    df["price_volume_corr_20"] = temp["ret"].rolling(20).corr(temp["vol_chg"])

    # 動量
    for n in [10, 20]:
        df[f"momentum_{n}d"] = c / c.shift(n) - 1

    # 價差特徵
    df["high_low_spread"]   = (df["high"] - df["low"]) / c
    df["open_close_spread"] = (c - df["open"]) / df["open"]

    return df


# ─────────────────────────────────────────────
# 資金流情緒特徵
# ─────────────────────────────────────────────

def add_fund_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    vol = df["volume"]

    for col in ["foreign_net", "trust_net", "dealer_net"]:
        if col not in df.columns:
            df[col] = 0.0

    # 淨買超佔成交量比
    for inst in ["foreign", "trust"]:
        net = df.get(f"{inst}_net", pd.Series(0, index=df.index))
        df[f"{inst}_net"] = net
        df[f"{inst}_net_vol_ratio"] = net / vol.replace(0, np.nan)

    # 外資淨買超移動平均
    df["foreign_net_ma5"]  = df["foreign_net"].rolling(5).mean()
    df["foreign_net_ma20"] = df["foreign_net"].rolling(20).mean()

    # 外資持股比率
    holding_col = "foreign_investment_shares_ratio"
    if holding_col in df.columns:
        df["foreign_holding_ratio"]   = df[holding_col]
        df["foreign_holding_chg_5d"]  = df[holding_col].diff(5)
    else:
        df["foreign_holding_ratio"]  = np.nan
        df["foreign_holding_chg_5d"] = np.nan

    # 融資融券
    if "margin_balance" in df.columns:
        df["margin_balance_chg"] = df["margin_balance"].pct_change()
    else:
        df["margin_balance"] = df["margin_balance_chg"] = 0.0

    if "short_balance" in df.columns:
        df["short_balance_chg"]  = df["short_balance"].pct_change()
        df["margin_short_ratio"] = (
            df["margin_balance"] /
            df["short_balance"].replace(0, np.nan)
        )
    else:
        df["short_balance"] = df["short_balance_chg"] = 0.0
        df["margin_short_ratio"] = 0.0

    # 散戶 vs 法人（正值→法人主導）
    df["retail_vs_inst"] = (
        df.get("foreign_net", 0) + df.get("trust_net", 0)
    ) / vol.replace(0, np.nan)

    return df


# ─────────────────────────────────────────────
# 基本面脈衝特徵
# ─────────────────────────────────────────────

def add_fundamental_features(df: pd.DataFrame) -> pd.DataFrame:
    # 月營收 YoY / MoM（月資料前向填值後計算）
    if "revenue" in df.columns:
        rev = df["revenue"]
        df["revenue_yoy"]  = rev / rev.shift(252) - 1    # 約 12 個月交易日
        df["revenue_mom"]  = rev / rev.shift(21)  - 1    # 約 1 個月
        df["revenue_3m_avg_yoy"] = (
            rev.rolling(63).mean() / rev.rolling(63).mean().shift(252) - 1
        )
    else:
        for c in ["revenue_yoy", "revenue_mom", "revenue_3m_avg_yoy"]:
            df[c] = np.nan

    # 毛利率（季報前向填值）
    if "gross_profit" in df.columns and "revenue_stmt" in df.columns:
        df["gross_margin"]      = df["gross_profit"] / df["revenue_stmt"].replace(0, np.nan)
        df["gross_margin_chg_qoq"] = df["gross_margin"].diff(63)
    else:
        df["gross_margin"] = df["gross_margin_chg_qoq"] = np.nan

    # 營業利益率
    if "operating_income" in df.columns and "revenue_stmt" in df.columns:
        df["operating_income_margin"] = (
            df["operating_income"] / df["revenue_stmt"].replace(0, np.nan)
        )
    else:
        df["operating_income_margin"] = np.nan

    # EPS（TTM / QoQ / YoY）
    if "eps" in df.columns:
        df["eps_ttm"]  = df["eps"].rolling(4).sum()      # 4 季滾動（季資料 ffill）
        df["eps_qoq"]  = df["eps"].diff(63)
        df["eps_yoy"]  = df["eps"].diff(252)
    else:
        df["eps_ttm"] = df["eps_qoq"] = df["eps_yoy"] = np.nan

    # ROE = net_income / Equity（TTM）
    if "net_income" in df.columns and "Equity" in df.columns:
        df["roe_ttm"] = (
            df["net_income"].rolling(4).sum() /
            df["Equity"].replace(0, np.nan)
        )
    else:
        df["roe_ttm"] = np.nan

    # 流動比率
    if "CurrentAssets" in df.columns and "CurrentLiabilities" in df.columns:
        df["current_ratio"] = (
            df["CurrentAssets"] / df["CurrentLiabilities"].replace(0, np.nan)
        )
    else:
        df["current_ratio"] = np.nan

    # 負債比率
    if "Liabilities" in df.columns and "TotalAssets" in df.columns:
        df["debt_ratio"] = df["Liabilities"] / df["TotalAssets"].replace(0, np.nan)
    else:
        df["debt_ratio"] = np.nan

    # 現金比率
    if "CashAndCashEquivalents" in df.columns and "CurrentLiabilities" in df.columns:
        df["cash_ratio"] = (
            df["CashAndCashEquivalents"] /
            df["CurrentLiabilities"].replace(0, np.nan)
        )
    else:
        df["cash_ratio"] = np.nan

    # 資本支出 / 營收比（CapEx Ratio，以 PPE 變動近似）
    if "PropertyPlantAndEquipment" in df.columns and "revenue_stmt" in df.columns:
        capex_approx = df["PropertyPlantAndEquipment"].diff(63)
        df["capex_ratio"] = capex_approx / df["revenue_stmt"].replace(0, np.nan)
    else:
        df["capex_ratio"] = np.nan

    return df


# ─────────────────────────────────────────────
# 估值錨點特徵
# ─────────────────────────────────────────────

def add_valuation_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["per", "pbr", "dividend_yield"]:
        if col not in df.columns:
            df[col] = np.nan
            continue
        # 過去 252 天百分位（衡量估值偏高/偏低）
        df[f"{col}_pct_rank_252"] = (
            df[col].rolling(252)
            .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1],
                   raw=False)
        )

    # PER 偏離 60 日均值
    if "per" in df.columns:
        df["per_deviation_from_ma"] = df["per"] / df["per"].rolling(60).mean() - 1

    return df


def add_event_features(df: pd.DataFrame, stock_id: str = DEFAULT_STOCK_ID) -> pd.DataFrame:
    """
    事件驅動特徵：捕捉台股結構性斷點（春節、除息、ADR 跳空）。
    """
    # 1. 股利與除息日效應
    if "cash_ex_dividend_trading_date" in df.columns:
        ex_dates = pd.to_datetime(df["cash_ex_dividend_trading_date"], errors="coerce")
        df["days_to_next_ex_dividend"] = (ex_dates - df.index).dt.days.clip(lower=-10, upper=365)
        
        # 除息前 5 日 (拉抬/棄息) 與 除息後 3 日 (填息力道)
        df["is_pre_dividend"] = ((df["days_to_next_ex_dividend"] > 0) & (df["days_to_next_ex_dividend"] <= 5)).astype(float)
        df["is_post_dividend"] = ((df["days_to_next_ex_dividend"] <= 0) & (df["days_to_next_ex_dividend"] >= -3)).astype(float)
        
        # 除息前後 5 日 dummy (原有 add_event_features 2 的邏輯)
        df["dividend_ex_dummy"] = (df["days_to_next_ex_dividend"].abs() <= 5).astype(int)
    else:
        df["is_pre_dividend"] = 0.0
        df["is_post_dividend"] = 0.0
        df["days_to_next_ex_dividend"] = 365
        df["dividend_ex_dummy"] = 0

    # 2. 農曆春節封關效應 (Lunar New Year)
    lunar_new_year_dates = [
        "2018-02-15", "2019-02-04", "2020-01-24", "2021-02-11", 
        "2022-01-31", "2023-01-21", "2024-02-09", "2025-01-28", "2026-02-16"
    ]
    lny_dt = pd.to_datetime(lunar_new_year_dates)
    days_to_lny = pd.Series(np.nan, index=df.index)
    for lny in lny_dt:
        diff = (lny - df.index).days
        mask = (diff >= 0) & (diff <= 20)
        days_to_lny[mask] = diff[mask]
    df["is_pre_lunar_new_year"] = (days_to_lny <= 7).astype(float)

    # 3. ADR 隔夜跳空 (TSM/UMC/HON/ASX 專屬)
    adr_config = {
        "2330": {"ticker": "TSM", "ratio": 5},
        "2303": {"ticker": "UMC", "ratio": 5},
        "2317": {"ticker": "HNHPF", "ratio": 2},
        "2308": {"ticker": "TWDAY", "ratio": 5},
    }
    if stock_id in adr_config:
        cfg = adr_config[stock_id]
        adr_col = f"us_{cfg['ticker']}_close"
        if adr_col in df.columns and "exchange_rate" in df.columns:
            implied_tw_price = (df[adr_col].shift(1) * df["exchange_rate"].shift(1)) / cfg["ratio"]
            df["adr_overnight_gap"] = implied_tw_price / df["close"].shift(1) - 1.0

    # 4. TTM 現金股利 (原有 add_event_features 2 的邏輯)
    if "cash_earnings_distribution" in df.columns:
        df["cash_dividend_ttm"] = df["cash_earnings_distribution"].rolling(252).sum()
    else:
        df["cash_dividend_ttm"] = np.nan

    # 5. 距上次財報天數
    df["days_since_last_earnings"] = np.arange(len(df)) % 63

    return df


# ─────────────────────────────────────────────
# 期貨籌碼特徵（新增）
# ─────────────────────────────────────────────

def add_futures_chip_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    期貨籌碼特徵（台指期 TX + 台指選擇權 TFO）

    特徵說明（全部 7 個）：
      tx_oi_chg_1d    : 儇1d 未平倉量變化（正就增資金流入，負就失雙毹出）
      tx_oi_chg_5d    : 期貨 5 日 OI 跨週變化（周趨勢方向確認）
      tx_basis        : 期現貨基差 = TX收盤 / TAIEX - 1
                        正基差 = 市場看多；負基差 = 避險壓力大
      tx_basis_5d_chg : 基差 5 日變化（營命轉變速度）
      tx_vol_ma_ratio : TX 成交量 / 20 日均量（頂底列鷹確認用）
      tfo_pcr_volume  : TFO Put/Call 成交量比（> 1.2 店部信號，< 0.7 頂部信號）
      tfo_pcr_oi      : TFO Put/Call 未平倉量比（機構避險仓位信號）

    資料來源：data_pipeline 已將 TX / TFO 資料 JOIN 至主框架
    起始曥期：TX 1998年起，TFO 2005年起（訓練起始 2010年，最少 5 年覆蓋）
    """
    # ── 台指期基礎特徵 ──────────────────────────────
    if "tx_oi" in df.columns:
        df["tx_oi_chg_1d"] = df["tx_oi"].diff(1)
        df["tx_oi_chg_5d"] = df["tx_oi"].diff(5)

        # 成交量相對強度
        tx_vol_ma = df["tx_volume"].rolling(20).mean()
        df["tx_vol_ma_ratio"] = df["tx_volume"] / tx_vol_ma.replace(0, np.nan)

        # 基差：期現貨價差比率（用 TAIEX 大盤索引）
        if "TAIEX" in df.columns:
            taiex = df["TAIEX"].replace(0, np.nan)
            df["tx_basis"]        = df["tx_close"] / taiex - 1
            df["tx_basis_5d_chg"] = df["tx_basis"].diff(5)
        else:
            df["tx_basis"]        = np.nan
            df["tx_basis_5d_chg"] = np.nan
            logger.debug("  [futures_chip] TAIEX 欄不存在，基差特徵設為 NaN")
    else:
        for col in ["tx_oi_chg_1d", "tx_oi_chg_5d", "tx_vol_ma_ratio",
                    "tx_basis", "tx_basis_5d_chg"]:
            df[col] = np.nan
        logger.debug("  [futures_chip] tx_oi 欄不存在（可能 data_pipeline 未將 TX 資料 JOIN）")

    # ── 選擇權 PCR ──────────────────────────────────
    if "tfo_put_vol" in df.columns and "tfo_call_vol" in df.columns:
        df["tfo_pcr_volume"] = (
            df["tfo_put_vol"] / df["tfo_call_vol"].replace(0, np.nan)
        )
    else:
        df["tfo_pcr_volume"] = np.nan

    if "tfo_put_oi" in df.columns and "tfo_call_oi" in df.columns:
        df["tfo_pcr_oi"] = (
            df["tfo_put_oi"] / df["tfo_call_oi"].replace(0, np.nan)
        )
    else:
        df["tfo_pcr_oi"] = np.nan

    return df


# ─────────────────────────────────────────────
# 滾動高階統計
# ─────────────────────────────────────────────

    return df


def add_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    log_r = df.get("log_return_1d", np.log(df["close"] / df["close"].shift(1)))

    for n in [20, 60]:
        df[f"skew_{n}d"]         = log_r.rolling(n).skew()
        df[f"kurt_{n}d"]         = log_r.rolling(n).kurt()
        df[f"autocorr_lag1_{n}d"] = log_r.rolling(n).apply(
            lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 2 else np.nan,
            raw=False
        )
        mean_r   = log_r.rolling(n).mean()
        std_r    = log_r.rolling(n).std()
        df[f"sharpe_{n}d"] = (mean_r / std_r.replace(0, np.nan)) * np.sqrt(252)

    return df


def add_volatility_clustering_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算波動率聚類特徵（Volatility Clustering & Regime Shift）。
    """
    log_r = df.get("log_return_1d", np.log(df["close"] / df["close"].shift(1)))
    
    # 波動率加速度
    vol_20 = log_r.rolling(20).std()
    df["vol_acceleration"] = vol_20 / vol_20.shift(5) - 1
    
    # 偏度與峰度的變動率
    for n in [20, 60]:
        df[f"skew_chg_{n}d"] = df[f"skew_{n}d"].diff(5)
        df[f"kurt_chg_{n}d"] = df[f"kurt_{n}d"].diff(5)
        
    return df


def add_us_chain_features(df: pd.DataFrame, stock_id: str = DEFAULT_STOCK_ID) -> pd.DataFrame:
    """
    美股供應鏈與 ADR 特徵（US Chain Features）。
    進化 v3.0：新增 Lead-Lag 領先指標與群體動能。
    """
    config = STOCK_CONFIGS.get(stock_id, STOCK_CONFIGS[DEFAULT_STOCK_ID])
    us_tickers = [t.lower().replace("^", "") for t in config["us_chain_tickers"]]
    
    # 1. ADR 溢價 (擴張版)
    if config.get("use_adr_premium", False):
        if "tsm_close" in df.columns and "usd_twd_mid" in df.columns:
            tsm_adr_twd = df["tsm_close"] * df["usd_twd_mid"] / 5
            df["tsm_premium"] = tsm_adr_twd / df["close"] - 1
            df["tsm_premium_ma5"] = df["tsm_premium"].rolling(5).mean()
            # 溢價波動度 (衡量恐慌程度)
            df["tsm_premium_vol"] = df["tsm_premium"].rolling(20).std()
        else:
            df["tsm_premium"] = df["tsm_premium_ma5"] = df["tsm_premium_vol"] = 0.0

    # 2. 主要科技/半導體/金融股動能
    chain_rets = []
    for ticker in us_tickers:
        col = f"{ticker}_close"
        if col in df.columns:
            # 領先 1 日報酬 (美股前一晚表現)
            ret_1d = df[col].pct_change(1)
            df[f"{ticker}_ret_1d_lag1"] = ret_1d.shift(0) # 因為 data_pipeline 已對齊日期，這裡 1d 就是領先信號
            df[f"{ticker}_ret_5d"] = df[col].pct_change(5)
            chain_rets.append(ret_1d)
        else:
            df[f"{ticker}_ret_1d_lag1"] = 0.0
            df[f"{ticker}_ret_5d"] = 0.0

    # 3. 群體動能 (Chain Composite Momentum)
    if chain_rets:
        df["us_chain_composite_ret"] = pd.concat(chain_rets, axis=1).mean(axis=1)
        df["us_chain_composite_ma5"] = df["us_chain_composite_ret"].rolling(5).mean()
    else:
        df["us_chain_composite_ret"] = 0.0
        df["us_chain_composite_ma5"] = 0.0

    # 4. 美股大盤相對強度 (SOXX / QQQ)
    for index_ticker in ["soxx_close", "qqq_close"]:
        if index_ticker in df.columns:
            df[f"{index_ticker}_mom"] = df[index_ticker].pct_change(5)
        else:
            df[f"{index_ticker}_mom"] = 0.0

    return df


# ─────────────────────────────────────────────
# 趨勢 Regime 偵測（新增）
# ─────────────────────────────────────────────

def add_trend_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    偵測趨勢 Regime（牛市/熊市/整理期），補充現有的波動率 Regime。

    波動率 Regime（已有）：衡量「市場動盪程度」
    趨勢 Regime（新增）  ：衡量「市場方向性」

    合併兩者使模型能理解：
      - 低波動 + 牛市 → 最佳進場環境（訊號可信度最高）
      - 高波動 + 熊市 → 保守或空倉（訊號可信度最低）
      - 整理期         → 避免頻繁操作（小幅雜訊多）

    趨勢判斷邏輯（多重驗證）：
      price_vs_ma120 > +5%  AND ma20 > ma50  → 牛市
      price_vs_ma120 < -5%  AND ma20 < ma50  → 熊市
      其餘                                    → 整理期

    輸出欄位：
      trend_regime        : 'bull' / 'bear' / 'sideways'（字串）
      trend_regime_int    : 1 / -1 / 0（整數，供模型使用）
      trend_strength      : ma20/ma50 比值（連續，衡量趨勢強度）
      price_vs_ma120      : 股價偏離 120 日均線百分比
      bull_bear_slope_20d : 20 日股價斜率（正=上升趨勢）
    """
    c = df["close"]

    # 若這些欄位在 add_technical_features 中已計算則複用
    ma20  = df["ma_20"]  if "ma_20"  in df.columns else c.rolling(20).mean()
    ma50  = df["ma_50"]  if "ma_50"  in df.columns else c.rolling(50).mean()
    ma120 = df["ma_120"] if "ma_120" in df.columns else c.rolling(120).mean()

    # 趨勢強度指標
    df["trend_strength"]    = ma20 / ma50.replace(0, np.nan) - 1
    df["price_vs_ma120"]    = c / ma120.replace(0, np.nan) - 1

    # 20 日股價斜率（OLS 正規化斜率）
    log_c = np.log(c.replace(0, np.nan))
    df["bull_bear_slope_20d"] = log_c.diff(20) / 20   # 平均每日 log 報酬率

    # 趨勢 Regime 分類（三重條件）
    is_bull = (
        (df["price_vs_ma120"] > 0.05) &    # 股價在 120 日均線以上 5%
        (ma20 > ma50)                        # 短均線 > 長均線（黃金交叉）
    )
    is_bear = (
        (df["price_vs_ma120"] < -0.05) &   # 股價在 120 日均線以下 5%
        (ma20 < ma50)                        # 短均線 < 長均線（死亡交叉）
    )

    trend_regime_int = pd.Series(0, index=df.index, dtype=int)   # 預設整理期
    trend_regime_int[is_bull] = 1
    trend_regime_int[is_bear] = -1

    df["trend_regime_int"] = trend_regime_int
    df["trend_regime"]     = trend_regime_int.map({1: "bull", -1: "bear", 0: "sideways"})

    # 複合 Regime 標籤（趨勢 × 波動率，供詳細分析用）
    if "realized_vol_20d" in df.columns:
        vol = df["realized_vol_20d"]
        vol_regime = pd.cut(
            vol,
            bins=[-np.inf, 0.20, 0.40, np.inf],
            labels=["low_vol", "mid_vol", "high_vol"],
        ).astype(str)
        df["compound_regime"] = trend_regime_int.map(
            {1: "bull", -1: "bear", 0: "sideways"}
        ) + "_" + vol_regime

    logger.debug("[trend_regime] 趨勢 Regime 特徵完成")
    return df


# ─────────────────────────────────────────────
# 多時程目標變數（新增）
# ─────────────────────────────────────────────

def add_multi_horizon_targets(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = (15, 21, 30),
    threshold_pct: float = 0.02,         # 2% = 視為「有效上漲」的最低門檻
) -> pd.DataFrame:
    """
    為多個時間視窗建立二元分類目標：「N 天後是否上漲超過 threshold_pct」

    設計理念：
      - 將連續報酬率轉換為二元分類，提升訊號純淨度
      - 同時訓練 15/21/30 天，取三者共識方向（多數決）降低噪音
      - threshold_pct 篩除橫盤期（±2% 視為無方向性），提升訊號品質

    輸出欄位：
      target_{n}d_binary : 1（上漲 > threshold）/ 0（其他）
      target_{n}d_return : 未來 N 日的連續對數報酬率（回歸輔助用）
      target_consensus   : 15/21/30 天三者共識（多數決，0.0~1.0）
    """
    c = df["close"]
    binary_cols = []

    for h in horizons:
        future_return = np.log(c.shift(-h) / c)
        col_binary = f"target_{h}d_binary"
        col_return = f"target_{h}d_return"

        # 二元分類：漲幅 > threshold → 1，否則 → 0，NaN 保留
        df[col_return] = future_return
        df[col_binary] = (
            (future_return > np.log(1 + threshold_pct)).astype("Int64")
        )
        # 最後 h 筆 NaN
        df.loc[df.index[-h:], col_binary] = pd.NA
        df.loc[df.index[-h:], col_return] = np.nan

        binary_cols.append(col_binary)
        logger.debug(f"[multi_horizon] target_{h}d_binary 完成 (threshold={threshold_pct:.1%})")

    # 多時程共識：三者多數決（上漲票數 / 總票數）
    binary_mat = df[binary_cols].astype(float)
    df["target_consensus"] = binary_mat.mean(axis=1)   # 0.0, 0.33, 0.67, 1.0
    df["target_consensus_binary"] = (df["target_consensus"] >= 0.5).astype("Int64")

    logger.info(
        f"[multi_horizon] 多時程目標完成：horizons={horizons}  "
        f"threshold={threshold_pct:.1%}  共識上漲率="
        f"{df['target_consensus_binary'].mean():.2%}"
    )
    return df


# ─────────────────────────────────────────────
# 目標變數（原有）
# ─────────────────────────────────────────────

def apply_triple_barrier(close: pd.Series, horizon: int, pt: float, sl: float) -> pd.Series:
    """
    三重障礙標籤法 (Triple-Barrier Method)：
    1. 上軌 (Profit Take)：價格漲幅超過 pt
    2. 下軌 (Stop Loss)：價格跌幅超過 sl
    3. 時間軌 (Vertical)：到達 horizon 日
    回傳：1 (上軌先觸發), -1 (下軌先觸發), 0 (時間軌觸發)
    """
    labels = pd.Series(index=close.index, data=0, dtype="Int64")
    
    # 為了向量化效率，我們使用滾動窗口的累計最大/最小回報
    # 注意：這是一個簡化版的向量化實現
    returns = close.pct_change()
    
    # 對於每一天，檢查未來 horizon 天內的情況
    # 這裡使用一個較為直接的迴圈，針對單一標的效能尚可
    # 若要全市場大規模運算，建議使用更高階的向量化技巧或並行
    close_val = close.values
    n = len(close_val)
    
    for i in range(n - horizon):
        window = close_val[i+1 : i+horizon+1]
        price_i = close_val[i]
        
        # 計算窗口內相對 i 日的最高與最低漲跌幅
        max_ret = (window.max() / price_i) - 1
        min_ret = (window.min() / price_i) - 1
        
        if max_ret >= pt and (min_ret > -sl or np.where(window/price_i - 1 >= pt)[0][0] < np.where(window/price_i - 1 <= -sl)[0][0]):
            labels.iloc[i] = 1
        elif min_ret <= -sl:
            labels.iloc[i] = -1
        else:
            labels.iloc[i] = 0
            
    # 最後 horizon 筆設為 NaN
    labels.iloc[-horizon:] = pd.NA
    return labels

def add_targets(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """
    新增三種目標：
    1. 傳統 30d 報酬率
    2. 傳統 30d 漲跌方向
    3. 三重障礙標籤 (Triple-Barrier) - 考慮路徑風險
    """
    future_close = df["close"].shift(-horizon)
    df["target_30d"] = (future_close / df["close"] - 1)
    df["target_return"] = future_close - df["close"]

    df["direction_30d"] = (
        np.sign(df["target_30d"])
        .astype("Int64")
    )

    # ── 三重障礙法標籤 ──────────────────────────────────────────
    # 使用 8% 停利 / 5% 停損 / 30天 結算的標準配置
    # 這是為了徹底解決「路徑風險」問題，讓模型學習避開深蹲後的反彈。
    df["target_triple_barrier"] = apply_triple_barrier(df["close"], horizon, pt=0.08, sl=0.05)
    
    # target_binary：全面切換為三重障礙標籤
    # 只有先觸發「停利軌 (pt)」的才標記為 1，其餘（觸發停損或時間到）均為 0。
    df["target_binary"] = (df["target_triple_barrier"] == 1).astype("Int64")

    return df


# ─────────────────────────────────────────────
# 主函式
# ─────────────────────────────────────────────

def add_commodity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算原油 (WTI/Brent) 與黃金的報酬率特徵。
    對塑膠 (台塑)、鋼鐵 (中鋼)、航運 (長榮) 具備強大解釋力。
    """
    df = df.copy()
    for col in ["oil_wti", "oil_brent", "gold_price"]:
        if col in df.columns:
            # 1. 報酬率
            df[f"{col}_ret_5d"] = df[col].pct_change(5)
            df[f"{col}_ret_20d"] = df[col].pct_change(20)
            # 2. 乖離率 (相對於 20 日均線)
            ma20 = df[col].rolling(20).mean()
            df[f"{col}_bias_20d"] = (df[col] - ma20) / ma20
    return df


def add_sponsor_chip_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sponsor/Backer 進階籌碼特徵：大戶、八大行庫、鉅額交易"""
    # 1. 大戶籌碼集中度：近 3 個月持股比例變化率
    if "large_holder_pct" in df.columns:
        df["large_holder_change_3m"] = df["large_holder_pct"] - df["large_holder_pct"].shift(60)
        df["large_holder_change_1m"] = df["large_holder_pct"] - df["large_holder_pct"].shift(20)

    # 2. 聰明錢護航指標：外資與八大行庫是否同步買超
    if "eight_banks_net" in df.columns and "foreign_net" in df.columns:
        df["smart_money_sync_buy"] = ((df["foreign_net"] > 0) & (df["eight_banks_net"] > 0)).astype(int)
        df["smart_money_sync_sell"] = ((df["foreign_net"] < 0) & (df["eight_banks_net"] < 0)).astype(int)
        df["eight_banks_net_roll_5"] = df["eight_banks_net"].rolling(5).sum()

    # 3. 鉅額交易淨額佔比
    if "block_net" in df.columns and "volume" in df.columns:
        df["block_trade_net_ratio"] = df["block_net"] / (df["volume"] * 1000 + 1)
        df["block_net_roll_5"] = df["block_net"].rolling(5).sum()
        
    return df

def add_sentiment_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sponsor/Backer 進階情緒與宏觀特徵：選擇權大戶、恐懼貪婪、景氣對策"""
    # 1. 選擇權大額 OI Put/Call Ratio
    if "put_top10_oi" in df.columns and "call_top10_oi" in df.columns:
        df["put_call_large_ratio"] = df["put_top10_oi"] / (df["call_top10_oi"] + 1)
        df["put_call_large_ratio_diff"] = df["put_call_large_ratio"].diff()

    # 2. 恐懼與貪婪指數
    if "fear_greed_score" in df.columns:
        df["fear_greed_roll_5"] = df["fear_greed_score"].rolling(5).mean()
        df["is_extreme_fear"] = (df["fear_greed_score"] < 25).astype(int)
        df["is_extreme_greed"] = (df["fear_greed_score"] > 75).astype(int)

    # 3. 景氣對策信號與大盤市值比重
    if "macro_monitoring_score" in df.columns:
        df["macro_score_diff"] = df["macro_monitoring_score"].diff()
    if "market_weight_pct" in df.columns:
        df["market_weight_diff"] = df["market_weight_pct"].diff()

    return df


def add_expectation_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """
    第一性原則：預期基準模型 (Expectation Baseline)
    核心邏輯：計算「實際觀測值」與「市場基準預期」的殘差 (Surprise)。
    只有殘差（驚奇值）才是推動價格不對稱波動的真正動力。
    """
    # 1. 籌碼驚奇 (Flow Surprise)
    # 計算外資買超相對於其 20 日平均絕對波動的倍數 (Z-Score 近似)
    if "foreign_net" in df.columns:
        flow_ma = df["foreign_net"].rolling(20).mean()
        flow_std = df["foreign_net"].rolling(20).std().replace(0, np.nan)
        df["foreign_flow_surprise"] = (df["foreign_net"] - flow_ma) / flow_std
    
    # 2. 宏觀驚奇 (Macro Surprise)
    # 匯率偏離度：當前匯率相對於 20 日均線的偏離 (反映短期避險或套利資金的突發性)
    if "usd_twd_spot" in df.columns:
        usd_ma = df["usd_twd_spot"].rolling(20, min_periods=1).mean()
        df["usd_twd_surprise"] = df["usd_twd_spot"] / usd_ma - 1
    
    # 3. 估值驚奇 (Valuation Residual)
    # PER 偏離度：相對於一年 (252天) 均值的偏離，捕捉「估值修復」或「超漲」
    if "per" in df.columns:
        per_ma = df["per"].rolling(252, min_periods=60).mean().replace(0, np.nan)
        df["per_valuation_surprise"] = df["per"] / per_ma - 1
        
    # 4. 營收驚奇 (Fundamental Surprise)
    # 營收加速器：當前 YoY 相對於過去一年平均 YoY 的增量
    if "revenue_yoy" in df.columns:
        rev_yoy_ma = df["revenue_yoy"].rolling(252).mean().replace(0, np.nan)
        df["revenue_growth_surprise"] = df["revenue_yoy"] - rev_yoy_ma

    # 5. 跨市場驚奇 (Cross-Market Surprise)
    # ADR 溢價偏離：相對於 ma5 的突發性偏離 (通常是美股盤後大變動的先行反映)
    if "tsm_premium" in df.columns:
        premium_ma = df["tsm_premium"].rolling(5).mean()
        df["adr_surprise"] = df["tsm_premium"] - premium_ma

    return df


def add_kwave_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    康波週期 (Kondratiev Wave) 長波特徵
    核心邏輯：利用「金油比」、「利率長波」與「大宗商品動能」來模擬經濟長週期的相位。

    修改摘要（第三輪審查修復 P1 全系統 kwave_score 幽靈特徵）：
      原版 kwave_score 僅在 gold_price + oil_brent + US10Y 都存在時才計算，
      其他標的則是「全系統使用、無人定義」的幽靈特徵。
      本版本明確提供三層 fallback，確保 kwave_score 在任何資料條件下
      都有確定值，並補充 NaN 防護以利下游 (signal_filter / backtest_engine) 使用。
    """
    # 1. 資源與避險對比 (生產 vs 儲存)
    if "gold_price" in df.columns and "oil_brent" in df.columns:
        # 金油比上升通常代表長波衰退/蕭條 (Winter)；下降代表繁榮 (Autumn)
        df["gold_oil_ratio"] = df["gold_price"] / df["oil_brent"].replace(0, np.nan)
        rolling_mean = df["gold_oil_ratio"].rolling(252 * 2, min_periods=60).mean()
        rolling_std  = df["gold_oil_ratio"].rolling(252 * 2, min_periods=60).std()
        df["gold_oil_ratio_z"] = (df["gold_oil_ratio"] - rolling_mean) / rolling_std.replace(0, np.nan)

    # 2. 利率長波 (Capital Cost Wave)
    if "US10Y" in df.columns:
        # 10年債殖利率的超長均線偏離 (反映長波生產力的資本回報率)
        long_ma = df["US10Y"].rolling(252 * 5, min_periods=252).mean()  # 5年均線
        df["yield_cycle_pos"] = df["US10Y"] - long_ma

    # 3. 康波相位分數 (K-Wave Score)
    # 邏輯：分數越高代表長波越趨於「秋季/冬初」(風險上升)；
    #       越低趨於「春季/回升」(擴張期)。取值區間 ≈ -2.0 ~ +2.0
    kwave_components: list[pd.Series] = []
    if "gold_oil_ratio_z" in df.columns and "yield_cycle_pos" in df.columns:
        # 主要訊號：金油比 + 利率長波（原始邏輯保留）
        kwave_components.append(
            df["gold_oil_ratio_z"].fillna(0.0) * 0.6
            + df["yield_cycle_pos"].rolling(20, min_periods=5).mean().fillna(0.0) * (-0.4)
        )

    # Fallback A：使用美 10/2Y 利差 + Fed 升息速率（適用於缺商品價格資料的標的）
    if "us_yield_spread" in df.columns and "fed_rate_chg_30d" in df.columns:
        spread_rank = df["us_yield_spread"].rolling(252, min_periods=60).rank(pct=True)
        rate_rank   = df["fed_rate_chg_30d"].rolling(252, min_periods=60).rank(pct=True)
        # 殖利率倒掛（rank 趨近 0）+ 升息加速（rank 趨近 1）→ 高 kwave 風險
        kwave_components.append(((1 - spread_rank) * 2 - 1).fillna(0.0))   # ~ -1 ~ +1
        kwave_components.append((rate_rank * 2 - 1).fillna(0.0))            # ~ -1 ~ +1

    # Fallback B：使用 TAIEX 相對強度反向作為粗糙代理
    if not kwave_components and "taiex_rel_strength" in df.columns:
        rs_rank = df["taiex_rel_strength"].rolling(252, min_periods=60).rank(pct=True)
        kwave_components.append(((1 - rs_rank) * 2 - 1).fillna(0.0))

    if kwave_components:
        # 多個訊號平均後 clip 至合理範圍
        df["kwave_score"] = sum(kwave_components) / len(kwave_components)
        df["kwave_score"] = df["kwave_score"].clip(-2.0, 2.0).fillna(0.0)
    else:
        # 最終 fallback：常數 0（中性），確保特徵存在以利下游 join
        df["kwave_score"] = 0.0

    return df


def add_k2026_resonance_features(df: pd.DataFrame, stock_id: str) -> pd.DataFrame:
    """
    《2026 量子金融藍圖》核心宏觀特徵：
    1. 雙週期共振：K-Wave (60y) + Kuznets (20y)
    2. 第六波奇點溢價：MBNRIC 文明交棒
    """
    # 1. 庫茲涅茨週期 (Kuznets) 代理：房地產與信貸週期 (利用 10Y 真實利率長波代理)
    if "US10Y" in df.columns and "cpi_yoy" in df.columns:
        real_yield = df["US10Y"] - df["cpi_yoy"]
        df["kuznets_proxy"] = real_yield.rolling(252*5).mean()
    else:
        df["kuznets_proxy"] = 0.0

    # 2. 2026 奇點距離與共振
    current_year = df.index.year + (df.index.month - 1) / 12.0
    df["singularity_dist"] = 2026.0 - current_year
    
    # 雙週期向下共振 (泡沫出清風險)
    if "kwave_score" in df.columns:
        # 當兩大週期同步走弱，且接近 2026，壓力值指數級上升
        df["resonance_pressure"] = (df["kwave_score"] + df["kuznets_proxy"]).rolling(126).mean()
        df["bubble_crash_risk"] = np.exp(-np.abs(df["singularity_dist"])) * (df["resonance_pressure"] > 0).astype(float)
    
    # 3. 第六波 (MBNRIC) 標的識別與溢價
    # 第六波驅動者：半導體奇點、AI、機器人、生物運算
    mbnric_leaders = ["2330", "2454", "3661", "6669", "NVDA", "ASML", "TSM"]
    if stock_id in mbnric_leaders:
        # 越接近 2026，第六波領導者的「奇點溢價」越高 (反映資產重新定價)
        df["singularity_premium"] = np.exp(-np.abs(df["singularity_dist"]))
    else:
        df["singularity_premium"] = 0.0
        
    return df


def add_kinetic_dynamics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    動力學特徵：將價格運動視為物理力學過程。
    對齊公式：動量 (Momentum) = 質量 (Mass) x 位移 (Displacement)
    """
    # 1. 質量 (Mass) = 資本流動性 (以市值對數與成交量組合建模)
    # 市值決定了改變價格的「慣性」
    if "market_cap" in df.columns:
        df["inertial_mass"] = np.log1p(df["market_cap"])
    else:
        # 若無市值數據，以 252 日均量作為質量的代理變數
        df["inertial_mass"] = np.log1p(df["volume"].rolling(252).mean())

    # 2. 位移 (Displacement) = 價格變動
    df["displacement"] = df["close"] - df["open"]
    df["displacement_pct"] = df["close"] / df["open"] - 1

    # 3. 動量 (Momentum) = 質量 * 位移
    # 這是物理體系中的真實動能，而非單純的價格漲跌
    df["kinetic_momentum"] = df["inertial_mass"] * df["displacement"]
    
    # 4. 能量 (Energy) = 0.5 * m * v^2 (回報率作為速度)
    if "returns_1d" in df.columns:
        df["kinetic_energy"] = 0.5 * df["inertial_mass"] * (df["returns_1d"]**2)

    return df


def add_power_law_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    冪律分佈與肥尾偵測：捕捉非正態分佈的極端動能。
    """
    # 1. 肥尾強度 (Tail Intensity)
    # 利用對數縮放捕捉極端回報 (Outliers)
    if "returns_1d" in df.columns:
        std = df["returns_1d"].rolling(252).std()
        z_score = df["returns_1d"] / std.replace(0, np.nan)
        # 對數化極值：當 Z > 3 時，強度會非線性增長
        df["tail_intensity"] = np.sign(z_score) * np.log1p(np.abs(z_score))
        
    # 2. 力學失衡比 (Force Imbalance)
    # 物理衝量 (Impulse) 的買賣能量對比
    if "price_impulse" in df.columns:
        # 正衝量總和 / 負衝量總和 (20日視窗)
        pos_imp = df["price_impulse"].clip(lower=0).rolling(20).sum()
        neg_imp = df["price_impulse"].clip(upper=0).abs().rolling(20).sum()
        df["force_imbalance"] = pos_imp / neg_imp.replace(0, np.nan)
        
    return df


def add_order_flow_proxy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    虛擬訂單流代理 (Virtual Order Flow Proxy)：從 OHLCV 中提取多空博弈力量。
    """
    if "high" in df.columns and "low" in df.columns and "close" in df.columns and "volume" in df.columns:
        range_hl = (df["high"] - df["low"]).replace(0, np.nan)
        
        # 1. 內生買賣壓力 (Intraday Pressure)
        # 利用 Close 在當日區間的位置來判斷買賣方誰主導了最後的決策
        # (Close - Low) = 買方推進, (High - Close) = 賣方壓制
        df["buy_pressure"] = df["volume"] * (df["close"] - df["low"]) / range_hl
        df["sell_pressure"] = df["volume"] * (df["high"] - df["close"]) / range_hl
        
        # 2. 訂單流失衡比 (Order Flow Imbalance Proxy)
        df["ofi_proxy"] = (df["buy_pressure"] - df["sell_pressure"]) / df["volume"].replace(0, np.nan)
        df["ofi_proxy_ma5"] = df["ofi_proxy"].rolling(5).mean()
        
        # 3. 訂單吸收/衰竭偵測 (Absorption/Exhaustion)
        # 當量很大但 Price Range 很小時，代表有隱藏的大量掛單在「吸收」動能
        df["absorption_ratio"] = df["volume"] / range_hl
        
    return df


def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    微觀結構代理特徵 (Microstructure Proxies)：從日線資料估算訂單流與流動性衝擊。
    """
    # 1. Amihud Illiquidity (非流動性)
    # 捕捉「單位成交金額產生的價格衝擊」: |Return| / Turnover
    if "returns_1d" in df.columns and "volume" in df.columns and "close" in df.columns:
        turnover_twd = (df["volume"] * df["close"]).replace(0, np.nan)
        df["amihud_illiquidity"] = df["returns_1d"].abs() / turnover_twd
        df["amihud_illiquidity"] = df["amihud_illiquidity"].rolling(20).mean()
    
    # 2. Kyle's Lambda Proxy (資訊交易衝擊)
    # 估算成交量對價格的敏感度：Cov(Return, SignedVol) / Var(SignedVol)
    if "returns_1d" in df.columns and "volume" in df.columns:
        signed_vol = np.sign(df["returns_1d"]) * df["volume"]
        roll_cov = df["returns_1d"].rolling(60).cov(signed_vol)
        roll_var = signed_vol.rolling(60).var()
        df["kyle_lambda_proxy"] = roll_cov / roll_var.replace(0, np.nan)

    # 3. 收盤位置比率 (Close-to-High Ratio)
    # 衡量買方在收盤前的控制力 (威廉指標變體)
    if "high" in df.columns and "low" in df.columns and "close" in df.columns:
        range_hl = (df["high"] - df["low"]).replace(0, np.nan)
        df["close_pos_ratio"] = (df["close"] - df["low"]) / range_hl

    return df


def add_cross_asset_correlation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    跨資產關聯特徵：計算與美股連結標的/產業指標的動態 Beta 偏離。
    """
    us_cols = [c for c in df.columns if c.startswith("us_") and c.endswith("_close")]
    if not us_cols or "returns_1d" not in df.columns:
        return df
        
    # 選取主要連結標的 (如 SOXX, NVDA)
    primary_us = us_cols[0]
    us_ret = np.log(df[primary_us] / df[primary_us].shift(1))
    tw_ret = df["returns_1d"]
    
    # 計算短期 (60d) vs 長期 (252d) Beta
    beta_60 = tw_ret.rolling(60).cov(us_ret) / us_ret.rolling(60).var()
    beta_252 = tw_ret.rolling(252).cov(us_ret) / us_ret.rolling(252).var()
    
    df["cross_beta_60"] = beta_60
    df["cross_beta_dev"] = beta_60 - beta_252 # Beta 偏離度 (過熱或脫鉤訊號)
    
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    信號交叉特徵 (Cross-Signal Interactions)：捕捉多因子共振效應。
    """
    # 1. 外資買超 x RSI 超賣 (強力支撐共振)
    if "foreign_net" in df.columns and "rsi_14" in df.columns:
        # 過去 5 日外資合計買超 + RSI 低於 40
        foreign_buy = (df["foreign_net"].rolling(5).sum() > 0).astype(float)
        rsi_oversold = (df["rsi_14"] < 40).astype(float)
        df["foreign_buy_x_oversold"] = foreign_buy * rsi_oversold
        
    # 2. 波動率 x 融資擴張 (融資斷頭/過熱風險)
    if "realized_vol_20d" in df.columns and "margin_balance" in df.columns:
        vol_high = (df["realized_vol_20d"] > df["realized_vol_20d"].rolling(60).quantile(0.8)).astype(float)
        margin_increase = (df["margin_balance"] > df["margin_balance"].shift(5)).astype(float)
        df["high_vol_margin_risk"] = vol_high * margin_increase
        
    return df


def add_gravity_well_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    重力井模型 (Gravity Well Model)：
    絕對重力中心 = 內在價值 (以長期估值中樞 PBR/PER 模擬)
    引力強度 = 偏離度的非線性回歸力
    """
    if "pbr" in df.columns:
        # 1. 絕對重力中心 (Gravity Center)：低熵狀態
        # 使用 5 年 (1260日) 超長週期中值，代表系統的穩定平衡點
        df["gravity_center"] = df["pbr"].rolling(1260, min_periods=252).median()
        
        # 2. 位移 (Displacement)：市場價格受情緒 (高熵) 驅動繞行
        df["valuation_displacement"] = df["pbr"] - df["gravity_center"]
        
        # 3. 物理引力 (Gravity Pull)：當偏離邊緣時，引力最強
        # 模擬 F = G * (m1*m2) / r^2，但在交易中，r 越大引力越強 (類似彈簧力/恢復力)
        df["gravity_pull"] = df["valuation_displacement"] * df["inertial_mass"]
        
    # 4. 力 (Force) = 資訊衝擊 (Information Shock)
    # 盈餘、籌碼、宏觀等驚奇值的加權整合
    surprise_cols = [c for c in df.columns if "surprise" in c]
    if surprise_cols:
        # 力 = 資訊能量 / 質量 (單位質量所受的資訊衝擊)
        df["info_force_per_mass"] = df[surprise_cols].mean(axis=1) / df["inertial_mass"]
        
    return df


def add_liquidity_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    降噪歸真：流動性品質 (Liquidity Quality)
    核心邏輯：區分「有效物理動量」與「隨機雜訊」。
    """
    if "volume" in df.columns and "returns_1d" in df.columns:
        # 1. 流動性效率 (Liquidity Efficiency)：單位成交量產生的回報
        # 數值越穩定，代表機構支撐力越強，噪音越低
        vol_std = df["returns_1d"].rolling(20).std().replace(0, np.nan)
        df["liquidity_quality"] = df["volume"].rolling(20).mean() / vol_std
        
    return df


def add_blueprint_entry_signals(df: pd.DataFrame, stock_id: str) -> pd.DataFrame:
    """
    佈局 2026：戰略性重力井建倉信號
    """
    if "gravity_pull" in df.columns and "info_force_per_mass" in df.columns:
        # 偵測「重力井底部」：當價格極端負偏離 + 資訊力注入
        pull_std = df["gravity_pull"].rolling(252).std()
        df["gravity_bottom_signal"] = ((df["gravity_pull"] < -1.5 * pull_std) & 
                                       (df["info_force_per_mass"] > 0)).astype(int)
        
    return df


def add_staleness_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算資料陳舊度 (Data Staleness)。
    利用 data_pipeline 中標記的 ann_date 欄位計算距離最近一次公告的天數。
    """
    # 1. 財報陳舊度 (以 EPS 為例)
    if "ann_date_stmt" in df.columns:
        # ann_date_stmt 已經被 ffill，代表每個交易日看到的都是最近一次公告的日期
        df["eps_staleness_days"] = (df.index - pd.to_datetime(df["ann_date_stmt"])).dt.days
        # 清除輔助欄位以防干擾模型
        df = df.drop(columns=["ann_date_stmt"])
        
    # 2. 營收陳舊度
    if "ann_date_rev" in df.columns:
        df["rev_staleness_days"] = (df.index - pd.to_datetime(df["ann_date_rev"])).dt.days
        df = df.drop(columns=["ann_date_rev"])
        
    return df


def build_features(raw: pd.DataFrame, stock_id: str = DEFAULT_STOCK_ID, for_inference: bool = False) -> pd.DataFrame:
    """
    接收 build_daily_frame() 的輸出，返回包含全部特徵 + 目標的 DataFrame。

    v2 新增：
      - add_trend_regime_features()  ：趨勢 Regime 偵測（牛市/熊市/整理期）
      - add_multi_horizon_targets()  ：多時程二元目標（15/21/30 天）
    """
    logger.info("=== 開始特徵工程 ===")
    df = raw.copy()

    df = add_technical_features(df)
    logger.info(f"  技術動能特徵完成，shape={df.shape}")

    df = add_fund_flow_features(df)
    logger.info(f"  資金流情緒特徵完成，shape={df.shape}")

    df = add_futures_chip_features(df)   # 期貨籌碼（新增）
    logger.info(f"  期貨籌碼特徵完成，shape={df.shape}")

    df = add_fundamental_features(df)
    logger.info(f"  基本面脈衝特徵完成，shape={df.shape}")

    df = add_valuation_features(df)
    logger.info(f"  估值錨點特徵完成，shape={df.shape}")

    # df = add_kwave_macro_score(df)
    logger.info(f"  宏觀因子特徵完成，shape={df.shape}")

    df = add_commodity_features(df)
    logger.info(f"  大宗商品特徵完成，shape={df.shape}")

    df = add_us_chain_features(df, stock_id)
    logger.info(f"  美股供應鏈特徵完成，shape={df.shape}")

    df = add_sponsor_chip_features(df)
    logger.info(f"  進階籌碼特徵完成，shape={df.shape}")

    df = add_sentiment_macro_features(df)
    logger.info(f"  進階情緒特徵完成，shape={df.shape}")

    df = add_event_features(df, stock_id)
    logger.info(f"  事件驅動特徵完成，shape={df.shape}")

    df = add_rolling_stats(df)
    logger.info(f"  滾動統計特徵完成，shape={df.shape}")

    df = add_volatility_clustering_features(df)
    logger.info(f"  波動率聚類特徵完成，shape={df.shape}")

    # ── 第一性原則：預期基準與驚奇值 (Expectation Residuals) ─────
    df = add_expectation_residuals(df)
    logger.info(f"  預期殘差 (Surprise) 注入完成，shape={df.shape}")

    # ── 宏觀長波：康波週期特徵 (K-Wave) ────────────────────────
    df = add_kwave_regime_features(df)
    logger.info(f"  康波長波特徵 (K-Wave) 注入完成，shape={df.shape}")

    # ── 2026 大共振：雙週期與奇點溢價 (2026 Resonance) ──────────
    df = add_k2026_resonance_features(df, stock_id)
    logger.info(f"  2026 大共振特徵注入完成，shape={df.shape}")

    # ── 動力學：動力學特徵 (Kinetic Dynamics) ────────────────
    df = add_kinetic_dynamics_features(df)
    logger.info(f"  動力學特徵 (Kinetic) 注入完成，shape={df.shape}")

    # ── 冪律分佈：肥尾偵測 (Power Law) ──────────────────────────
    df = add_power_law_features(df)
    logger.info(f"  冪律肥尾特徵 (Power Law) 注入完成，shape={df.shape}")

    df = add_order_flow_proxy_features(df)
    df = add_microstructure_features(df)
    df = add_cross_asset_correlation_features(df)
    df = add_interaction_features(df)
    df = add_staleness_features(df)
    logger.info(f"  複合進階特徵 (Microstructure/Cross-Asset/Interaction) 注入完成，shape={df.shape}")

    # ── 物理學：重力井模型 (Gravity Well) ─────────────────────
    df = add_gravity_well_features(df)
    logger.info(f"  重力井物理特徵 (Gravity Well) 注入完成，shape={df.shape}")

    # ── 降噪歸真：流動性品質 (Liquidity Quality) ───────────────
    df = add_liquidity_quality_features(df)
    logger.info(f"  流動性品質特徵注入完成，shape={df.shape}")

    # ── 戰略佈局：2026 建倉信號 (Singularity Entry) ───────────
    df = add_blueprint_entry_signals(df, stock_id)
    logger.info(f"  戰略建倉信號注入完成，shape={df.shape}")


    # ── 新增：趨勢 Regime 偵測 ────────────────────────────────
    df = add_trend_regime_features(df)
    logger.info(f"  趨勢 Regime 特徵完成，shape={df.shape}")

    # ── 舊版目標（向後兼容）──────────────────────────────────
    df = add_targets(df, HORIZON)
    logger.info(f"  目標變數完成，shape={df.shape}")

    # ── 新增：多時程二元目標（15/21/30 天）───────────────────
    df = add_multi_horizon_targets(df, horizons=(15, 21, 30), threshold_pct=0.02)
    logger.info(f"  多時程目標完成，shape={df.shape}")

    # 推論時不能移除最後的 HORIZON 天，因為我們要預測未來
    if not for_inference:
        # 移除最後 HORIZON 天（目標為 NaN）
        df = df.iloc[:-HORIZON]

    # ── inf → NaN（除以零產生的 inf 必須先清除，XGBoost/LightGBM 無法處理 inf）──
    num_cols = df.select_dtypes(include="number").columns.tolist()
    # 統計 inf 數量（inf 出現在 replace 之前）
    n_inf = int(np.isinf(df[num_cols].select_dtypes(include="float")).sum().sum())
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    if n_inf > 0:
        logger.warning(f"  清除 {n_inf} 個 inf 值（已替換為 NaN，請檢查相關特徵）")

    # 移除極端值（±5σ clip），只對數值型浮點欄套用
    # 注意：絕對股價與成交量等非平穩序列（Non-stationary data）絕不可 clip，否則會切斷創新高的最新價格！
    exclude_clip_cols = {
        "target_30d", "direction_30d", "target_binary", "target_consensus",
        "target_consensus_binary", "close", "open", "high", "low", "volume",
        "trend_regime", "trend_regime_int", "compound_regime",
    }
    # 多時程目標也排除 clip
    exclude_clip_cols |= {f"target_{h}d_binary" for h in (15, 21, 30)}
    exclude_clip_cols |= {f"target_{h}d_return" for h in (15, 21, 30)}

    for col in df.columns:
        if col in exclude_clip_cols:
            continue
        if pd.api.types.is_float_dtype(df[col]):
            mu, sigma = df[col].mean(), df[col].std()
            if sigma > 0:
                df[col] = df[col].clip(mu - 5 * sigma, mu + 5 * sigma)

    logger.info(f"=== 特徵工程完成：{len(df):,} 筆 × {df.shape[1]} 欄 ===")
    return df


def build_features_with_medium_term(
    raw: pd.DataFrame,
    stock_id: str = DEFAULT_STOCK_ID,
    for_inference: bool = False,
) -> pd.DataFrame:
    """
    完整特徵工程入口（含中期信號）。

    流程：
      1. data_pipeline.build_medium_term_features()  → 計算 15 個中期信號特徵
      2. build_features()                             → 全部特徵工程（含趨勢 Regime + 多時程目標）

    這是推薦的訓練入口，用法：
        from data_pipeline import build_daily_frame, build_medium_term_features
        from feature_engineering import build_features_with_medium_term

        raw = build_daily_frame(stock_id="2330")
        df  = build_features_with_medium_term(raw, stock_id="2330")
    """
    try:
        from data_pipeline import build_medium_term_features
        raw_enhanced = build_medium_term_features(raw, stock_id=stock_id)
        logger.info("[pipeline] 中期信號特徵已整合")
    except Exception as e:
        logger.warning(f"[pipeline] 中期信號特徵計算失敗，略過：{e}")
        raw_enhanced = raw

    return build_features(raw_enhanced, stock_id=stock_id, for_inference=for_inference)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    from data_pipeline import build_daily_frame
    raw = build_daily_frame()
    df  = build_features(raw)
    print(df[["close", "target_30d", "direction_30d",
              "rsi_14", "foreign_net", "per", "fed_rate"]].tail(10))
    print(f"\n目標分佈：\n{df['direction_30d'].value_counts()}")
    print(f"\n特徵缺失率（前 20）：\n{df.isnull().mean().sort_values(ascending=False).head(20)}")
