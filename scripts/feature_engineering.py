"""
feature_engineering.py — 特徵工程層
依第一性原則 5 大類特徵，基於 build_daily_frame() 的輸出建構模型輸入。

輸出：pd.DataFrame，新增 ~100 個特徵欄 + target_30d（回歸）+ direction_30d（分類）
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import FEATURE_GROUPS, HORIZON

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
    df["price_volume_corr_20"] = (
        pd.concat([log_r, v.pct_change()], axis=1)
        .rolling(20).corr().unstack().iloc[:, 1]
    )

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


# ─────────────────────────────────────────────
# 宏觀因子特徵
# ─────────────────────────────────────────────

def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    # FED 利率
    fed_col = "FED"
    if fed_col in df.columns:
        df["fed_rate"]      = df[fed_col]
        df["fed_rate_chg_30d"] = df[fed_col].diff(21)
    else:
        df["fed_rate"] = df["fed_rate_chg_30d"] = np.nan

    for country, col in [("BOJ", "boj_rate"), ("ECB", "ecb_rate")]:
        df[col] = df[country] if country in df.columns else np.nan

    # 匯率（USD/TWD）
    if "usd_twd_mid" in df.columns:
        df["usd_twd_spot"]    = df["usd_twd_mid"]
        df["usd_twd_chg_10d"] = df["usd_twd_mid"].pct_change(10)
    else:
        df["usd_twd_spot"] = df["usd_twd_chg_10d"] = np.nan

    # 大盤指數報酬
    for idx_name, prefix in [("TAIEX", "taiex"), ("TPEx", "tpex")]:
        if idx_name in df.columns:
            idx_r = np.log(df[idx_name] / df[idx_name].shift(1))
            df[f"{prefix}_ret_5d"]  = np.log(df[idx_name] / df[idx_name].shift(5))
            df[f"{prefix}_ret_20d"] = np.log(df[idx_name] / df[idx_name].shift(20))
        else:
            for suf in ["ret_5d", "ret_20d"]:
                df[f"{prefix}_{suf}"] = np.nan

    # 相對強弱（個股 vs 大盤）
    if "TAIEX" in df.columns:
        stock_ret  = df["log_return_20d"] if "log_return_20d" in df.columns else 0
        market_ret = df["taiex_ret_20d"]  if "taiex_ret_20d"  in df.columns else 0
        df["taiex_rel_strength"] = stock_ret - market_ret
    else:
        df["taiex_rel_strength"] = np.nan

    return df


# ─────────────────────────────────────────────
# 事件驅動特徵
# ─────────────────────────────────────────────

def add_event_features(df: pd.DataFrame) -> pd.DataFrame:
    # 距下次除息日天數
    if "cash_ex_dividend_trading_date" in df.columns:
        ex_dates = pd.to_datetime(
            df["cash_ex_dividend_trading_date"], errors="coerce"
        )
        df["days_to_next_ex_dividend"] = (
            ex_dates - df.index
        ).dt.days.clip(lower=-5, upper=365)
        # 除息前後 5 日 dummy
        df["dividend_ex_dummy"] = (
            df["days_to_next_ex_dividend"].abs() <= 5
        ).astype(int)
    else:
        df["days_to_next_ex_dividend"] = 365
        df["dividend_ex_dummy"]        = 0

    # TTM 現金股利
    if "cash_earnings_distribution" in df.columns:
        df["cash_dividend_ttm"] = (
            df["cash_earnings_distribution"].rolling(252).sum()
        )
    else:
        df["cash_dividend_ttm"] = np.nan

    # 距上次財報天數（以季報為準，每 63 天一次近似）
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


def add_us_chain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    美股供應鏈與 ADR 特徵（US Chain Features）。
    整合台積電 ADR (TSM)、輝達 (NVDA)、蘋果 (AAPL) 與半導體指數 (SOXX) 的外圍市場動能。
    """
    # 1. TSM 現貨溢價差距 (TSM Premium)
    # TSM ADR 1 單位 = 5 股 2330 零股。需先轉換成台幣
    if "tsm_close" in df.columns and "usd_twd_mid" in df.columns:
        tsm_adr_twd = df["tsm_close"] * df["usd_twd_mid"] / 5
        df["tsm_premium"] = tsm_adr_twd / df["close"] - 1
        df["tsm_premium_ma5"] = df["tsm_premium"].rolling(5).mean()
    else:
        df["tsm_premium"] = 0.0
        df["tsm_premium_ma5"] = 0.0

    # 2. 主要科技/半導體股動能 (NVIDIA, AAPL, SOXX)
    for ticker in ["nvda", "aapl", "soxx"]:
        col = f"{ticker}_close"
        if col in df.columns:
            df[f"{ticker}_ret_1d"] = df[col].pct_change(1)
            df[f"{ticker}_ret_5d"] = df[col].pct_change(5)
            df[f"{ticker}_ret_20d"] = df[col].pct_change(20)
        else:
            df[f"{ticker}_ret_1d"] = 0.0
            df[f"{ticker}_ret_5d"] = 0.0
            df[f"{ticker}_ret_20d"] = 0.0

    return df


# ─────────────────────────────────────────────
# 目標變數
# ─────────────────────────────────────────────

def add_targets(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """
    target_30d   ：未來 horizon 日收盤報酬率（回歸目標，float，最後 horizon 筆為 NaN）
    direction_30d：漲跌方向 +1 / -1 / 0（Int64 nullable integer，保留 NaN）
    target_binary：上漲為 1，其餘為 0（Int64 nullable integer）

    注意：最後 horizon 筆 target_30d 為 NaN（無未來收盤價）。
    pandas 2.x 禁止帶 NaN 的浮點欄直接 .astype(int)，
    改用 pandas nullable integer 型態（Int64）保留 NaN。
    build_features() 後續以 df.iloc[:-HORIZON] 移除這些 NaN 列。
    """
    future_close = df["close"].shift(-horizon)
    df["target_30d"] = (future_close / df["close"] - 1)
    df["target_return"] = future_close - df["close"]

    # np.sign 對 NaN 仍回傳 NaN（float）；用 Int64 保留 NaN，不強制轉 int
    df["direction_30d"] = (
        np.sign(df["target_30d"])
        .astype("Int64")          # pandas nullable integer，支援 NaN
    )

    # target_binary：上漲(direction==1) → 1，其餘 → 0，NaN 保留
    df["target_binary"] = (df["direction_30d"] == 1).astype("Int64")

    return df


# ─────────────────────────────────────────────
# 主函式
# ─────────────────────────────────────────────

def build_features(raw: pd.DataFrame, for_inference: bool = False) -> pd.DataFrame:
    """
    接收 build_daily_frame() 的輸出，返回包含全部特徵 + 目標的 DataFrame。
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

    df = add_macro_features(df)
    logger.info(f"  宏觀因子特徵完成，shape={df.shape}")

    df = add_us_chain_features(df)
    logger.info(f"  美股供應鏈特徵完成，shape={df.shape}")

    df = add_event_features(df)
    logger.info(f"  事件驅動特徵完成，shape={df.shape}")

    df = add_rolling_stats(df)
    logger.info(f"  滾動統計特徵完成，shape={df.shape}")

    df = add_targets(df, HORIZON)
    logger.info(f"  目標變數完成，shape={df.shape}")

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
    exclude_clip_cols = {"target_30d", "direction_30d", "target_binary", "close", "open", "high", "low", "volume"}
    for col in df.columns:
        if col in exclude_clip_cols:
            continue
        if pd.api.types.is_float_dtype(df[col]):
            mu, sigma = df[col].mean(), df[col].std()
            if sigma > 0:
                df[col] = df[col].clip(mu - 5 * sigma, mu + 5 * sigma)

    logger.info(f"=== 特徵工程完成：{len(df):,} 筆 × {df.shape[1]} 欄 ===")
    return df


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
