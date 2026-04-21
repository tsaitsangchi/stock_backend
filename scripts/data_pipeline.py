"""
data_pipeline.py — 資料層（PostgreSQL 17 版）
直接從 PostgreSQL 讀取所有資料表，合併為每日寬格式 DataFrame。

DB 連線設定與 fetch_technical_data.py 一致（psycopg2）。
輸出：pd.DataFrame，index=date（DatetimeIndex，台股交易日），含前向填值。

使用方式：
    from data_pipeline import build_daily_frame
    df = build_daily_frame()                            # 全歷史
    df = build_daily_frame(start_date="2015-01-01")    # 指定起始日
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from typing import Optional

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

from config import STOCK_ID, STOCK_CONFIGS, DEFAULT_STOCK_ID

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# PostgreSQL 連線設定（與 fetch_technical_data.py 相同）
# ─────────────────────────────────────────────
DB_CONFIG: dict = {
    "dbname":   "stock",
    "user":     "stock",
    "password": "stock",
    "host":     "localhost",
    "port":     "5432",
}


@contextmanager
def get_conn():
    """取得 psycopg2 連線，自動關閉，異常時 rollback。"""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _query(sql: str, params: tuple = ()) -> pd.DataFrame:
    """
    執行 SELECT，以 DictCursor 回傳 pd.DataFrame。
    psycopg2 的 Decimal 型態自動轉為 float。
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            if not rows:
                return pd.DataFrame()
            cols = [d.name for d in cur.description]
            df   = pd.DataFrame([dict(r) for r in rows], columns=cols)
    # Decimal / object → float（psycopg2 回傳 Decimal 型態）
    # pandas 2.x 已移除 errors="ignore"，改用 coerce 後還原非數值欄
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            # 只有當轉換後 NaN 數量未增加時才套用（確保純數值欄被轉換）
            orig_na = df[col].isna().sum()
            if converted.isna().sum() <= orig_na:
                df[col] = converted
    return df


# ─────────────────────────────────────────────
# 各資料表載入函式
# ─────────────────────────────────────────────

def load_stock_price(stock_id: str = STOCK_ID) -> pd.DataFrame:
    """
    stock_price：日線股價
    DB schema: date, stock_id, trading_volume, trading_money,
               open, max, min, close, spread, trading_turnover
    → max/min 對應特徵工程的 high/low
    """
    sql = """
        SELECT date,
               open::float,
               max::float             AS high,
               min::float             AS low,
               close::float,
               spread::float,
               trading_volume::bigint AS volume,
               trading_money::bigint  AS turnover_value,
               trading_turnover::int
        FROM   stock_price
        WHERE  stock_id = %s
        ORDER  BY date
    """
    df = _query(sql, (stock_id,))
    if df.empty:
        logger.warning(f"[stock_price] 無資料（stock_id={stock_id}）")
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    logger.debug(f"[stock_price] {len(df):,} 筆  "
                 f"{df.index[0].date()} ~ {df.index[-1].date()}")
    return df


def load_stock_per(stock_id: str = STOCK_ID) -> pd.DataFrame:
    """
    stock_per：每日 PER / PBR / 股利殖利率
    DB schema: date, stock_id, dividend_yield, per, pbr
    """
    sql = """
        SELECT date,
               dividend_yield::float,
               per::float,
               pbr::float
        FROM   stock_per
        WHERE  stock_id = %s
        ORDER  BY date
    """
    df = _query(sql, (stock_id,))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    logger.debug(f"[stock_per] {len(df):,} 筆")
    return df


def load_institutional(stock_id: str = STOCK_ID) -> pd.DataFrame:
    """
    institutional_investors_buy_sell：三大法人買賣超
    DB schema: date, stock_id, buy, name, sell
    name 值：Foreign_Investor / Investment_Trust / Dealers（及其子項）
    → long → wide，並計算各機構淨買賣超
    """
    sql = """
        SELECT date,
               name,
               buy::bigint  AS buy,
               sell::bigint AS sell
        FROM   institutional_investors_buy_sell
        WHERE  stock_id = %s
        ORDER  BY date, name
    """
    df = _query(sql, (stock_id,))
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])

    wide = df.pivot_table(
        index="date", columns="name",
        values=["buy", "sell"], aggfunc="last",
    )
    wide.columns = [f"{agg}_{nm.lower()}" for agg, nm in wide.columns]
    wide = wide.sort_index()

    # 計算三大機構淨買賣超
    for inst, col_key in [
        ("foreign", "foreign_investor"),
        ("trust",   "investment_trust"),
        ("dealer",  "dealers"),
    ]:
        bc, sc = f"buy_{col_key}", f"sell_{col_key}"
        if bc in wide.columns and sc in wide.columns:
            wide[f"{inst}_net"] = wide[bc].fillna(0) - wide[sc].fillna(0)

    logger.debug(f"[institutional] {len(wide):,} 筆  欄：{wide.columns.tolist()}")
    return wide


def load_margin(stock_id: str = STOCK_ID) -> pd.DataFrame:
    """
    margin_purchase_short_sale：融資融券餘額
    DB schema: date, stock_id, margin_purchase_today_balance,
               short_sale_today_balance, offset_loan_and_short …
    """
    sql = """
        SELECT date,
               margin_purchase_today_balance::int    AS margin_balance,
               margin_purchase_yesterday_balance::int,
               short_sale_today_balance::int         AS short_balance,
               short_sale_yesterday_balance::int,
               offset_loan_and_short::int
        FROM   margin_purchase_short_sale
        WHERE  stock_id = %s
        ORDER  BY date
    """
    df = _query(sql, (stock_id,))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    logger.debug(f"[margin] {len(df):,} 筆")
    return df


def load_shareholding(stock_id: str = STOCK_ID) -> pd.DataFrame:
    """
    shareholding：外資持股比率
    DB schema: date, stock_id, foreign_investment_shares_ratio,
               foreign_investment_remain_ratio, …
    """
    sql = """
        SELECT date,
               foreign_investment_shares_ratio::float,
               foreign_investment_remain_ratio::float,
               foreign_investment_shares::bigint,
               number_of_shares_issued::bigint
        FROM   shareholding
        WHERE  stock_id = %s
        ORDER  BY date
    """
    df = _query(sql, (stock_id,))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    logger.debug(f"[shareholding] {len(df):,} 筆")
    return df


def load_financial_statements(stock_id: str = STOCK_ID) -> pd.DataFrame:
    """
    financial_statements：損益表（季資料，long → wide）
    DB schema: date, stock_id, type, value, origin_name
    只保留預測最相關的 7 個科目。
    """
    KEEP = (
        "Revenue", "GrossProfit", "OperatingIncome",
        "IncomeAfterTax", "EPS", "TAX",
        "TotalNonoperatingIncomeAndExpense",
    )
    ph = ",".join(["%s"] * len(KEEP))
    sql = f"""
        SELECT date, type, value::float
        FROM   financial_statements
        WHERE  stock_id = %s
          AND  type IN ({ph})
        ORDER  BY date, type
    """
    df = _query(sql, (stock_id, *KEEP))
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot_table(
        index="date", columns="type", values="value", aggfunc="last"
    ).sort_index()
    wide = wide.rename(columns={
        "Revenue":         "revenue_stmt",
        "GrossProfit":     "gross_profit",
        "OperatingIncome": "operating_income",
        "IncomeAfterTax":  "net_income",
        "EPS":             "eps",
        "TAX":             "tax",
    })
    logger.debug(f"[financial_statements] {len(wide):,} 季期")
    return wide


def load_balance_sheet(stock_id: str = STOCK_ID) -> pd.DataFrame:
    """
    balance_sheet：資產負債表（季資料，long → wide）
    DB schema: date, stock_id, type, value, origin_name
    """
    KEEP = (
        "TotalAssets",            "TotalAssets_per",
        "Liabilities",            "Liabilities_per",
        "Equity",                 "Equity_per",
        "CashAndCashEquivalents", "CashAndCashEquivalents_per",
        "CurrentAssets",          "CurrentAssets_per",
        "CurrentLiabilities",     "CurrentLiabilities_per",
        "PropertyPlantAndEquipment",
        "LongtermBorrowings",     "ShorttermBorrowings",
        "Inventories",
    )
    ph = ",".join(["%s"] * len(KEEP))
    sql = f"""
        SELECT date, type, value::float
        FROM   balance_sheet
        WHERE  stock_id = %s
          AND  type IN ({ph})
        ORDER  BY date, type
    """
    df = _query(sql, (stock_id, *KEEP))
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot_table(
        index="date", columns="type", values="value", aggfunc="last"
    ).sort_index()
    logger.debug(f"[balance_sheet] {len(wide):,} 季期")
    return wide


def load_dividend(stock_id: str = STOCK_ID) -> pd.DataFrame:
    """
    dividend：股利資料（除息日、現金股利）
    DB schema: date, stock_id, cash_earnings_distribution,
               cash_ex_dividend_trading_date, cash_dividend_payment_date …
    """
    sql = """
        SELECT date,
               cash_earnings_distribution::float,
               cash_statutory_surplus::float,
               cash_ex_dividend_trading_date,
               cash_dividend_payment_date
        FROM   dividend
        WHERE  stock_id = %s
        ORDER  BY date
    """
    df = _query(sql, (stock_id,))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["cash_ex_dividend_trading_date"] = pd.to_datetime(
        df["cash_ex_dividend_trading_date"], errors="coerce"
    )
    df["cash_dividend_payment_date"] = pd.to_datetime(
        df["cash_dividend_payment_date"], errors="coerce"
    )
    df = df.set_index("date").sort_index()
    logger.debug(f"[dividend] {len(df):,} 筆")
    return df


def load_month_revenue(stock_id: str = STOCK_ID) -> pd.DataFrame:
    """
    month_revenue：月營收
    DB schema: date, stock_id, country, revenue, revenue_month, revenue_year
    """
    sql = """
        SELECT date,
               revenue::bigint,
               revenue_month::int,
               revenue_year::int
        FROM   month_revenue
        WHERE  stock_id = %s
        ORDER  BY date
    """
    df = _query(sql, (stock_id,))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    logger.debug(f"[month_revenue] {len(df):,} 筆")
    return df


def load_interest_rate() -> pd.DataFrame:
    """
    interest_rate：各央行利率（FED / BOJ / ECB / PBOC 等）
    DB schema: date, country, full_country_name, interest_rate
    → pivot(country) → 每央行獨立欄
    """
    sql = """
        SELECT date, country, interest_rate::float
        FROM   interest_rate
        ORDER  BY date, country
    """
    df = _query(sql)
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot_table(
        index="date", columns="country",
        values="interest_rate", aggfunc="last",
    ).sort_index()
    logger.debug(f"[interest_rate] {len(wide):,} 日期 × {wide.shape[1]} 央行")
    return wide


def load_exchange_rate(currencies: tuple = ("USD", "JPY", "EUR")) -> pd.DataFrame:
    """
    exchange_rate：匯率（取得 USD/TWD, JPY/TWD, EUR/TWD 中間價）
    DB schema: date, currency, cash_buy, cash_sell, spot_buy, spot_sell
    """
    ph = ",".join(["%s"] * len(currencies))
    sql = f"""
        SELECT date, currency,
               spot_buy::float,
               spot_sell::float
        FROM   exchange_rate
        WHERE  currency IN ({ph})
        ORDER  BY date, currency
    """
    df = _query(sql, currencies)
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])

    # 計算中間價並 pivot
    df["mid"] = (df["spot_buy"] + df["spot_sell"]) / 2
    wide = df.pivot_table(index="date", columns="currency", values="mid", aggfunc="last")
    wide.columns = [f"{c.lower()}_twd_mid" for c in wide.columns]
    
    logger.debug(f"[exchange_rate] {len(wide):,} 筆，幣別：{wide.columns.tolist()}")
    return wide.sort_index()


def load_bond_yield() -> pd.DataFrame:
    """
    bond_yield：公債殖利率（US10Y, US2Y 等）
    並計算利差 (Spread)
    """
    sql = """
        SELECT date, bond_id, value::float
        FROM   bond_yield
        ORDER  BY date, bond_id
    """
    df = _query(sql)
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot_table(index="date", columns="bond_id", values="value", aggfunc="last").sort_index()
    
    # 計算利差
    if "US10Y" in wide.columns and "US2Y" in wide.columns:
        wide["us_yield_spread"] = wide["US10Y"] - wide["US2Y"]
        
    logger.debug(f"[bond_yield] {len(wide):,} 筆")
    return wide


def load_total_return_index() -> pd.DataFrame:
    """
    total_return_index：TAIEX / TPEx 大盤指數
    DB schema: date, stock_id, price
    → pivot(stock_id) → TAIEX / TPEx 各一欄
    """
    sql = """
        SELECT date, stock_id, price::float
        FROM   total_return_index
        ORDER  BY date, stock_id
    """
    df = _query(sql)
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot_table(
        index="date", columns="stock_id",
        values="price", aggfunc="last",
    ).sort_index()
    logger.debug(f"[total_return_index] {len(wide):,} 筆  "
                 f"指數：{wide.columns.tolist()}")
    return wide


def load_commodities() -> pd.DataFrame:
    """
    載入 crude_oil_prices (Brent/WTI) 與 gold_price。
    """
    # ── 原油 ──
    sql_oil = """
        SELECT date, name, price::float
        FROM crude_oil_prices
        ORDER BY date, name
    """
    df_oil = _query(sql_oil)
    oil_wide = pd.DataFrame()
    if not df_oil.empty:
        df_oil["date"] = pd.to_datetime(df_oil["date"])
        oil_wide = df_oil.pivot_table(
            index="date", columns="name", values="price", aggfunc="last"
        ).sort_index()
        oil_wide.columns = [f"oil_{c.lower()}" for c in oil_wide.columns]

    # ── 黃金 ──
    sql_gold = """
        SELECT date, price::float AS gold_price
        FROM gold_price
        ORDER BY date
    """
    df_gold = _query(sql_gold)
    gold_df = pd.DataFrame()
    if not df_gold.empty:
        df_gold["date"] = pd.to_datetime(df_gold["date"])
        gold_df = df_gold.set_index("date").sort_index()

    if oil_wide.empty and gold_df.empty:
        return pd.DataFrame()

    if oil_wide.empty: return gold_df
    if gold_df.empty: return oil_wide

    return oil_wide.join(gold_df, how="outer").sort_index()


# ─────────────────────────────────────────────
# 期貨 / 選擇權載入函式（新增）
# ─────────────────────────────────────────────

def load_tx_futures(
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
) -> pd.DataFrame:
    """
    台指期 TX 近月合約每日資料（依未平倉量選主力合約）。
    回傳欄位：tx_close, tx_volume, tx_oi

    原理：台積電佔加權指數權重 ~30%，TX 跟台積電相關性 > 0.85，
    外資台指期淨部位是資金自由化後最強的方向宣示指標之一。
    """
    params: list = []
    date_filter = ""
    if start_date:
        date_filter += " AND date >= %s"
        params.append(start_date)
    if end_date:
        date_filter += " AND date <= %s"
        params.append(end_date)

    sql = f"""
        WITH ranked AS (
            SELECT date,
                   close::float        AS tx_close,
                   volume::int         AS tx_volume,
                   open_interest::int  AS tx_oi,
                   ROW_NUMBER() OVER (
                       PARTITION BY date
                       ORDER BY open_interest DESC
                   ) AS rn
            FROM futures_daily
            WHERE futures_id = 'TX'
              AND trading_session = 'position'
              AND open_interest > 0
              AND close > 0
              {date_filter}
        )
        SELECT date, tx_close, tx_volume, tx_oi
        FROM ranked WHERE rn = 1
        ORDER BY date
    """
    df = _query(sql, tuple(params))
    if df.empty:
        logger.warning("[tx_futures] TX 期貨資料為空")
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    logger.debug(f"[tx_futures] {len(df):,} 筆  "
                 f"{df.index[0].date()} ~ {df.index[-1].date()}")
    return df


def load_tfo_options(
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
) -> pd.DataFrame:
    """
    台指選擇權 TFO 每日彙總：Put/Call 成交量與未平倉量。
    回傳欄位：tfo_put_vol, tfo_call_vol, tfo_put_oi, tfo_call_oi

    用途：計算 PCR（Put/Call Ratio），衡量機構避險壓力和市場恐慌程度。
    PCR > 1.2 歷史上展示市場過度恐慌（底部信號）；
    PCR < 0.7 展示市場貪婪過度（頂部信號）。
    """
    params: list = []
    date_filter = ""
    if start_date:
        date_filter += " AND date >= %s"
        params.append(start_date)
    if end_date:
        date_filter += " AND date <= %s"
        params.append(end_date)

    sql = f"""
        SELECT date,
               SUM(CASE WHEN call_put = 'put'  THEN volume        ELSE 0 END)::float AS tfo_put_vol,
               SUM(CASE WHEN call_put = 'call' THEN volume        ELSE 0 END)::float AS tfo_call_vol,
               SUM(CASE WHEN call_put = 'put'  THEN open_interest ELSE 0 END)::float AS tfo_put_oi,
               SUM(CASE WHEN call_put = 'call' THEN open_interest ELSE 0 END)::float AS tfo_call_oi
        FROM option_daily
        WHERE option_id = 'TFO'
          AND trading_session = 'position'
          {date_filter}
        GROUP BY date
        ORDER BY date
    """
    df = _query(sql, tuple(params))
    if df.empty:
        logger.warning("[tfo_options] TFO 選擇權資料為空")
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    logger.debug(f"[tfo_options] {len(df):,} 筆  "
                 f"{df.index[0].date()} ~ {df.index[-1].date()}")
    return df


def load_us_stocks(
    target_stocks: list[str] = ["TSM", "NVDA", "AAPL", "SOXX"],
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
) -> pd.DataFrame:
    """
    載入 us_stock_price 美股收盤價（adj_close），
    包含台積電 ADR、輝達、蘋果、半導體指數等供應鏈指標。
    """
    if not target_stocks:
        return pd.DataFrame()

    params = list(target_stocks)
    ph = ",".join(["%s"] * len(target_stocks))
    
    date_filter = ""
    if start_date:
        date_filter += " AND date >= %s"
        params.append(start_date)
    if end_date:
        date_filter += " AND date <= %s"
        params.append(end_date)
        
    sql = f"""
        SELECT date, stock_id, adj_close::float
        FROM us_stock_price
        WHERE stock_id IN ({ph})
        {date_filter}
        ORDER BY date, stock_id
    """
    df = _query(sql, tuple(params))
    if df.empty:
        logger.warning("[us_stocks] 美股資料為空")
        return pd.DataFrame()
        
    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot_table(
        index="date", columns="stock_id",
        values="adj_close", aggfunc="last"
    ).sort_index()
    
    # 重新命名欄位 (tsm_close, nvda_close, aapl_close, soxx_close)
    wide.columns = [f"{c.lower()}_close" for c in wide.columns]
    
    logger.debug(f"[us_stocks] {len(wide):,} 筆  "
                 f"{wide.index[0].date()} ~ {wide.index[-1].date()}")
    return wide


# ─────────────────────────────────────────────
# 主合併函式
# ─────────────────────────────────────────────

def build_daily_frame(
    stock_id:   str = STOCK_ID,
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
) -> pd.DataFrame:
    """
    以 stock_price 的交易日為主軸，將所有資料表 LEFT JOIN 後前向填值。

    參數
    ----
    stock_id   : 股票代碼（預設 config.STOCK_ID = "2330"）
    start_date : 過濾起始日 YYYY-MM-DD（None 表示不限）
    end_date   : 過濾結束日 YYYY-MM-DD（None 表示不限）

    回傳
    ----
    pd.DataFrame
        index   = date（DatetimeIndex，台股交易日）
        columns = 所有原始欄位（前向填值後）
    """
    logger.info(f"=== 建立每日特徵框架（stock_id={stock_id}）===")

    # ── 主軸：日線股價 ──────────────────────────────────────
    price = load_stock_price(stock_id)
    if price.empty:
        raise ValueError(
            f"stock_price 無資料（stock_id={stock_id}）。"
            "請確認 PostgreSQL 連線與資料是否存在。"
        )
    base = price.copy()

    # ── 每日頻率資料（直接 left join，以 date index 對齊） ──
    daily_sources = [
        ("per",            load_stock_per(stock_id)),
        ("inst",           load_institutional(stock_id)),
        ("margin",         load_margin(stock_id)),
        ("shareholding",   load_shareholding(stock_id)),
        ("interest_rate",  load_interest_rate()),
        ("exchange_rate",  load_exchange_rate()),
        ("bond_yield",     load_bond_yield()),
        ("return_index",   load_total_return_index()),
        # ── 期貨籌碼（新增）────────────────────────────────
        ("tx_futures",     load_tx_futures(start_date, end_date)),
        ("tfo_options",    load_tfo_options(start_date, end_date)),
        # ── 美股特徵（新增）────────────────────────────────
        ("us_stocks",      load_us_stocks(
            target_stocks = STOCK_CONFIGS.get(stock_id, STOCK_CONFIGS[DEFAULT_STOCK_ID])["us_chain_tickers"],
            start_date    = start_date, 
            end_date      = end_date
        )),
        # ── 大宗商品（新增）────────────────────────────────
        ("commodities",    load_commodities()),
    ]
    for name, df in daily_sources:
        if df.empty:
            logger.warning(f"  [{name}] 查無資料，跳過")
            continue
        base = base.join(df, how="left", rsuffix=f"_{name}")
        logger.debug(f"  JOIN [{name}] → shape={base.shape}")

    # ── 季資料（財報/資產負債表）：reindex 前向填值 ─────────
    quarterly_sources = [
        ("financial_stmt", load_financial_statements(stock_id)),
        ("balance_sheet",  load_balance_sheet(stock_id)),
    ]
    for name, df in quarterly_sources:
        if df.empty:
            logger.warning(f"  [{name}] 查無資料，跳過")
            continue
        base = base.join(
            df.reindex(base.index, method="ffill"),
            how="left", rsuffix=f"_{name}",
        )
        logger.debug(f"  JOIN [{name}] (ffill) → shape={base.shape}")

    # ── 月營收：reindex 前向填值 ─────────────────────────────
    rev = load_month_revenue(stock_id)
    if not rev.empty:
        base = base.join(
            rev.reindex(base.index, method="ffill"),
            how="left", rsuffix="_rev",
        )

    # ── 股利：reindex 前向填值 ────────────────────────────────
    div = load_dividend(stock_id)
    if not div.empty:
        base = base.join(
            div.reindex(base.index, method="ffill"),
            how="left", rsuffix="_div",
        )

    # ── 全域前向填值（補齊遺漏值） ───────────────────────────
    base = base.ffill()

    # ── 日期範圍過濾 ─────────────────────────────────────────
    if start_date:
        base = base.loc[start_date:]
    if end_date:
        base = base.loc[:end_date]

    logger.info(
        f"=== 每日框架完成：{len(base):,} 交易日 × {base.shape[1]} 欄  "
        f"（{base.index[0].date()} ~ {base.index[-1].date()}）==="
    )
    return base

# ─────────────────────────────────────────────
# 預測逐日軌跡寫入 (Insert Daily Trajectory)
# ─────────────────────────────────────────────

def save_forecast_daily(report: dict) -> None:
    """
    每次 predict.py 執行後，將未來 30 天的逐日預測軌跡寫入
    stock_forecast_daily 資料表（30 筆/次，以日曆日推算交易日偏移量）。

    report 需包含 daily_trajectory 欄位（由 run_prediction 組裝）。
    """
    trajectory: list = report.get("daily_trajectory")
    if not trajectory:
        logger.warning("報告中無 daily_trajectory，略過逐日寫入")
        return

    sql = """
        INSERT INTO stock_forecast_daily (
            predict_date, stock_id, forecast_date, day_offset,
            price_q10, price_q25, price_q50, price_q75, price_q90,
            ensemble_price,
            current_close, prob_up, confidence_level, model_agreement,
            xgb_prob, lgb_prob, tft_prob,
            extreme_valuation, macro_shock
        ) VALUES (
            %(predict_date)s, %(stock_id)s, %(forecast_date)s, %(day_offset)s,
            %(price_q10)s, %(price_q25)s, %(price_q50)s, %(price_q75)s, %(price_q90)s,
            %(ensemble_price)s,
            %(current_close)s, %(prob_up)s, %(confidence_level)s, %(model_agreement)s,
            %(xgb_prob)s, %(lgb_prob)s, %(tft_prob)s,
            %(extreme_valuation)s, %(macro_shock)s
        )
        ON CONFLICT (predict_date, stock_id, forecast_date) DO UPDATE SET
            day_offset        = EXCLUDED.day_offset,
            price_q10         = EXCLUDED.price_q10,
            price_q25         = EXCLUDED.price_q25,
            price_q50         = EXCLUDED.price_q50,
            price_q75         = EXCLUDED.price_q75,
            price_q90         = EXCLUDED.price_q90,
            ensemble_price    = EXCLUDED.ensemble_price,
            current_close     = EXCLUDED.current_close,
            prob_up           = EXCLUDED.prob_up,
            confidence_level  = EXCLUDED.confidence_level,
            model_agreement   = EXCLUDED.model_agreement,
            xgb_prob          = EXCLUDED.xgb_prob,
            lgb_prob          = EXCLUDED.lgb_prob,
            tft_prob          = EXCLUDED.tft_prob,
            extreme_valuation = EXCLUDED.extreme_valuation,
            macro_shock       = EXCLUDED.macro_shock,
            created_at        = CURRENT_TIMESTAMP;
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, trajectory)
        conn.commit()

    logger.info(
        f"逐日預測寫入成功：predict_date={trajectory[0]['predict_date']} "
        f"× {len(trajectory)} 天 (stock_id={trajectory[0]['stock_id']})"
    )



# ─────────────────────────────────────────────
# 快速診斷（直接執行此檔時）
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info("── 連線測試 ──")
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                ver = cur.fetchone()[0]
        logger.info(f"連線成功：{ver}")
    except Exception as e:
        logger.error(f"連線失敗：{e}")
        sys.exit(1)

    df = build_daily_frame()

    PREVIEW = ["close", "FED", "usd_twd_mid", "jpy_twd_mid", "US10Y", "US2Y", "us_yield_spread"]
    avail = [c for c in PREVIEW if c in df.columns]

    print(f"\nshape      : {df.shape}")
    print(f"date range : {df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"\n最後 5 筆（關鍵欄）：")
    print(df[avail].tail(5).to_string())
    print(f"\n缺失率（前 15）：")
    print(df.isnull().mean().sort_values(ascending=False).head(15).to_string())
