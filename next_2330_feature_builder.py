"""
next_2330_feature_builder.py  v4.0
「下一支2330」特徵計算 + 寫入資料庫

v4.0 修正清單（對應評估報告）：
  ① 工程 Bug 修復
     - stock_id VARCHAR(10) → VARCHAR(20)（修正 ETF 代碼寫入失敗）
     - 自動遷移舊表欄位（ALTER COLUMN）
     - 相似度計算改為「兩階段」：
         Phase 1：全市場分批計算特徵 → 寫 DB（similarity = NULL）
         Phase 2：從 DB 讀全量特徵 → 全市場 Z-score → 算相似度 → 回寫
       （消除跨批次 Z-score 不一致 + 基準 2330 similarity 為 NaN 的問題）
     - 確保 2330 基準股票必在 Phase 1 首批計算

  ② 特徵升級（新增 10 個更「第一性」的特徵）
     - gross_margin_vs_industry   毛利率 vs 同產業中位數溢價（定價權代理）
     - ppe_growth_rate            固定資產年增率（資本密集壁壘趨勢）
     - foreign_bullish_years      外資連續買超月數（生態鎖定深度）
     - rev_acceleration           營收加速度（近3年CAGR / 前3年CAGR）
     - roic_proxy                 保留盈餘/總資產（資本再投資效率）
     - gross_margin_10y_stability 毛利率長期穩定度（10年護城河深度）
     - net_margin_trend           淨利率趨勢斜率（是否持續改善）
     - eps_rev_correlation        EPS 與營收相關性（獲利品質）
     - debt_trend                 負債比率趨勢（是否持續去槓桿）
     - rev_vs_market_premium      營收超額成長（相對全市場均值）

  ③ XGBoost 機率模型（由 next_2330_predictor_light.py 執行）
     - Feature Builder 只負責計算/儲存特徵
     - 模型訓練/預測由輕量預測器執行，避免 Builder 過重

執行方式：
    python next_2330_feature_builder.py               # 全市場（建議）
    python next_2330_feature_builder.py --stocks 2330 2454 2317
    python next_2330_feature_builder.py --batch-size 30   # 記憶體不足時
    python next_2330_feature_builder.py --resume          # 斷點續算
    python next_2330_feature_builder.py --force           # 強制重算
    python next_2330_feature_builder.py --phase1-only     # 只算特徵，跳過相似度
    python next_2330_feature_builder.py --phase2-only     # 只更新相似度（特徵已算完）
"""

import argparse
import sys
import warnings
from datetime import datetime, date

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════
# 設定
# ══════════════════════════════════════════════
DB_CONFIG = {
    "dbname":   "stock",
    "user":     "stock",
    "password": "stock",
    "host": "localhost",
    "port":     "5432",
}
BATCH_SIZE = 50


def get_conn():
    return psycopg2.connect(**DB_CONFIG)


def read_sql(sql: str, params=None) -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql(sql, conn, params=params if params else None)


def test_connection() -> bool:
    try:
        read_sql("SELECT 1")
        print("  ✓  PostgreSQL 連線成功")
        return True
    except Exception as e:
        print(f"  ✗  PostgreSQL 連線失敗：{e}")
        return False


# ══════════════════════════════════════════════
# DDL + 自動遷移
# ══════════════════════════════════════════════
DDL_FEATURES = """
CREATE TABLE IF NOT EXISTS next_2330_features (
    date                        DATE          NOT NULL,
    stock_id                    VARCHAR(20)   NOT NULL,   -- ★ 改為 VARCHAR(20) 支援 ETF 代碼
    stock_name                  VARCHAR(100),
    industry_category           VARCHAR(50),

    -- ① 護城河：技術規模壁壘（原有）
    rev_cagr_5y                 NUMERIC(20,6),
    rev_yoy_mean                NUMERIC(20,6),
    rev_yoy_stability           NUMERIC(20,6),
    rev_season_cv               NUMERIC(20,6),
    rev_latest_bn               NUMERIC(20,4),
    gross_margin_mean           NUMERIC(20,6),
    gross_margin_stability      NUMERIC(20,6),
    gross_margin_latest         NUMERIC(20,6),
    op_margin_mean              NUMERIC(20,6),
    op_margin_stability         NUMERIC(20,6),
    net_margin_mean             NUMERIC(20,6),
    eps_cagr                    NUMERIC(20,6),
    eps_mean                    NUMERIC(20,6),
    eps_latest                  NUMERIC(20,6),
    eps_stability               NUMERIC(20,6),
    asset_cagr                  NUMERIC(20,6),
    asset_latest                NUMERIC(20,4),
    equity_cagr                 NUMERIC(20,6),
    retained_earnings_cagr      NUMERIC(20,6),
    retained_earnings_latest    NUMERIC(20,4),
    ppe_to_assets_mean          NUMERIC(20,6),
    roe_mean                    NUMERIC(20,6),
    roe_latest                  NUMERIC(20,6),
    roa_mean                    NUMERIC(20,6),
    roa_latest                  NUMERIC(20,6),
    debt_equity_ratio_mean      NUMERIC(20,6),
    debt_equity_ratio_latest    NUMERIC(20,6),
    current_ratio_mean          NUMERIC(20,6),
    current_ratio_latest        NUMERIC(20,6),

    -- ② 護城河：需求爆發生態鎖定（原有）
    foreign_ownership_mean      NUMERIC(20,6),
    foreign_ownership_latest    NUMERIC(20,6),
    foreign_ownership_trend     NUMERIC(20,6),
    foreign_ownership_stability NUMERIC(20,6),
    insti_foreign_net_mean      NUMERIC(20,2),
    insti_foreign_net_positive  NUMERIC(20,6),
    insti_trust_net_mean        NUMERIC(20,2),
    insti_trust_net_positive    NUMERIC(20,6),
    insti_total_net_mean        NUMERIC(20,2),
    insti_total_net_positive    NUMERIC(20,6),
    price_1y_return             NUMERIC(20,6),
    price_3y_return             NUMERIC(20,6),
    price_latest                NUMERIC(20,4),
    price_annualized_vol        NUMERIC(20,6),
    avg_turnover_vol_mn         NUMERIC(20,4),
    per_mean                    NUMERIC(20,6),
    per_latest                  NUMERIC(20,6),
    pbr_mean                    NUMERIC(20,6),
    pbr_latest                  NUMERIC(20,6),
    div_yield_mean              NUMERIC(20,6),
    div_yield_latest            NUMERIC(20,6),
    margin_ratio_mean           NUMERIC(20,6),
    short_ratio_mean            NUMERIC(20,6),
    ls_ratio_mean               NUMERIC(20,6),

    -- ③ 護城河：財務紀律現金流（原有）
    cash_div_cagr               NUMERIC(20,6),
    cash_div_mean               NUMERIC(20,6),
    cash_div_latest             NUMERIC(20,6),
    cash_div_stability          NUMERIC(20,6),
    cash_div_years              NUMERIC(20,0),
    cash_div_positive_ratio     NUMERIC(20,6),
    employee_stock_div_ratio    NUMERIC(20,6),

    -- ★ 新增 10 個更「第一性」的特徵
    gross_margin_vs_industry    NUMERIC(20,6),  -- 毛利率溢價（vs 同產業中位數）
    ppe_growth_rate             NUMERIC(20,6),  -- 固定資產年增率（資本壁壘趨勢）
    foreign_bullish_months      NUMERIC(20,0),  -- 外資連續淨買超月數
    rev_acceleration            NUMERIC(20,6),  -- 營收加速度（近3年/前3年 CAGR）
    roic_proxy                  NUMERIC(20,6),  -- 保留盈餘/總資產（再投資效率）
    gross_margin_10y_stability  NUMERIC(20,6),  -- 毛利率 10 年長期穩定度
    net_margin_trend            NUMERIC(20,6),  -- 淨利率趨勢斜率（OLS 斜率）
    eps_rev_correlation         NUMERIC(20,6),  -- EPS 與營收相關性（獲利品質）
    debt_trend                  NUMERIC(20,6),  -- 負債比率趨勢斜率（去槓桿程度）
    rev_vs_market_premium       NUMERIC(20,6),  -- 營收 YoY 相對全市場超額成長

    -- 相似度、模型輸出與元資料
    similarity_to_2330          NUMERIC(20,6),   -- Phase 2 更新：加權歐氏距離
    xgb_prob                    NUMERIC(20,6),   -- XGBoost「成為下一支2330的機率」
    start_date_used             DATE,
    update_time                 TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_next2330_stock ON next_2330_features(stock_id);
CREATE INDEX IF NOT EXISTS idx_next2330_date  ON next_2330_features(date);
CREATE INDEX IF NOT EXISTS idx_next2330_sim   ON next_2330_features(similarity_to_2330 DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_next2330_xgb   ON next_2330_features(xgb_prob DESC NULLS LAST);
"""

# 自動遷移：舊表 stock_id VARCHAR(10) → VARCHAR(20) + 新增欄位
MIGRATE_SQL = """
DO $$
BEGIN
    -- 修正 stock_id 長度
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='next_2330_features'
          AND column_name='stock_id'
          AND character_maximum_length < 20
    ) THEN
        ALTER TABLE next_2330_features ALTER COLUMN stock_id TYPE VARCHAR(20);
        RAISE NOTICE 'Migrated: stock_id → VARCHAR(20)';
    END IF;

    -- 新增 10 個特徵欄位（若不存在）
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='next_2330_features' AND column_name='gross_margin_vs_industry') THEN
        ALTER TABLE next_2330_features
            ADD COLUMN gross_margin_vs_industry  NUMERIC(20,6),
            ADD COLUMN ppe_growth_rate           NUMERIC(20,6),
            ADD COLUMN foreign_bullish_months    NUMERIC(20,0),
            ADD COLUMN rev_acceleration          NUMERIC(20,6),
            ADD COLUMN roic_proxy               NUMERIC(20,6),
            ADD COLUMN gross_margin_10y_stability NUMERIC(20,6),
            ADD COLUMN net_margin_trend          NUMERIC(20,6),
            ADD COLUMN eps_rev_correlation       NUMERIC(20,6),
            ADD COLUMN debt_trend               NUMERIC(20,6),
            ADD COLUMN rev_vs_market_premium    NUMERIC(20,6),
            ADD COLUMN xgb_prob                 NUMERIC(20,6);
        RAISE NOTICE 'Migrated: added 11 new columns';
    END IF;
END
$$;
"""

# ══════════════════════════════════════════════
# FEATURE_COLS（決定 Upsert 欄位順序）
# ══════════════════════════════════════════════
FEATURE_COLS = [
    # 原有特徵
    "rev_cagr_5y","rev_yoy_mean","rev_yoy_stability","rev_season_cv","rev_latest_bn",
    "gross_margin_mean","gross_margin_stability","gross_margin_latest",
    "op_margin_mean","op_margin_stability","net_margin_mean",
    "eps_cagr","eps_mean","eps_latest","eps_stability",
    "asset_cagr","asset_latest","equity_cagr",
    "retained_earnings_cagr","retained_earnings_latest",
    "ppe_to_assets_mean","roe_mean","roe_latest","roa_mean","roa_latest",
    "debt_equity_ratio_mean","debt_equity_ratio_latest",
    "current_ratio_mean","current_ratio_latest",
    "foreign_ownership_mean","foreign_ownership_latest",
    "foreign_ownership_trend","foreign_ownership_stability",
    "insti_foreign_net_mean","insti_foreign_net_positive",
    "insti_trust_net_mean","insti_trust_net_positive",
    "insti_total_net_mean","insti_total_net_positive",
    "price_1y_return","price_3y_return","price_latest",
    "price_annualized_vol","avg_turnover_vol_mn",
    "per_mean","per_latest","pbr_mean","pbr_latest",
    "div_yield_mean","div_yield_latest",
    "margin_ratio_mean","short_ratio_mean","ls_ratio_mean",
    "cash_div_cagr","cash_div_mean","cash_div_latest",
    "cash_div_stability","cash_div_years","cash_div_positive_ratio",
    "employee_stock_div_ratio",
    # ★ 新增特徵
    "gross_margin_vs_industry","ppe_growth_rate","foreign_bullish_months",
    "rev_acceleration","roic_proxy","gross_margin_10y_stability",
    "net_margin_trend","eps_rev_correlation","debt_trend","rev_vs_market_premium",
]

UPSERT_SQL = f"""
INSERT INTO next_2330_features (
    date, stock_id, stock_name, industry_category,
    {', '.join(FEATURE_COLS)},
    similarity_to_2330, start_date_used, update_time
) VALUES %s
ON CONFLICT (date, stock_id) DO UPDATE SET
    stock_name          = EXCLUDED.stock_name,
    industry_category   = EXCLUDED.industry_category,
    {', '.join(f'{c} = EXCLUDED.{c}' for c in FEATURE_COLS)},
    similarity_to_2330  = EXCLUDED.similarity_to_2330,
    start_date_used     = EXCLUDED.start_date_used,
    update_time         = CURRENT_TIMESTAMP;
"""

# 相似度更新（Phase 2）
UPDATE_SIM_SQL = """
UPDATE next_2330_features
SET similarity_to_2330 = %s, update_time = CURRENT_TIMESTAMP
WHERE date = %s AND stock_id = %s;
"""

# XGBoost 機率更新
UPDATE_XGB_SQL = """
UPDATE next_2330_features
SET xgb_prob = %s, update_time = CURRENT_TIMESTAMP
WHERE date = %s AND stock_id = %s;
"""


# ══════════════════════════════════════════════
# 全市場基準數據（供新特徵計算使用）
# ══════════════════════════════════════════════
_MARKET_CACHE = {}


def get_market_rev_yoy(start_date: str, calc_date: str) -> float:
    """全市場月營收 YoY 均值（用於計算超額成長）。"""
    key = f"market_rev_yoy_{start_date}_{calc_date}"
    if key in _MARKET_CACHE:
        return _MARKET_CACHE[key]
    try:
        df = read_sql(
            "SELECT date, revenue FROM month_revenue WHERE date >= %s AND date <= %s ORDER BY date",
            params=[start_date, calc_date]
        )
        if df.empty:
            return np.nan
        df["date"] = pd.to_datetime(df["date"])
        df = df.groupby("date")["revenue"].sum()
        yoy = df.pct_change(12).dropna()
        val = yoy.mean()
        _MARKET_CACHE[key] = val
        return val
    except Exception:
        return np.nan


def get_industry_gross_margin(industry: str, start_date: str, calc_date: str) -> float:
    """同產業所有股票毛利率中位數（用於計算定價權溢價）。"""
    key = f"ind_gm_{industry}_{start_date}_{calc_date}"
    if key in _MARKET_CACHE:
        return _MARKET_CACHE[key]
    try:
        # 取同產業股票清單
        stocks_df = read_sql(
            "SELECT stock_id FROM stock_info WHERE industry_category = %s",
            params=[industry]
        )
        if stocks_df.empty:
            return np.nan
        sids = stocks_df["stock_id"].astype(str).tolist()
        ph = ",".join(["%s"] * len(sids))
        df = read_sql(
            f"""
            SELECT f.stock_id,
                   SUM(CASE WHEN f.type='GrossProfit' THEN f.value ELSE 0 END) /
                   NULLIF(SUM(CASE WHEN f.type='Revenue' THEN f.value ELSE 0 END), 0) AS gm
            FROM financial_statements f
            WHERE f.date >= %s
              AND f.date <= %s
              AND f.type IN ('GrossProfit','Revenue')
              AND f.stock_id IN ({ph})
            GROUP BY f.stock_id
            HAVING SUM(CASE WHEN f.type='Revenue' THEN f.value ELSE 0 END) > 0
            """,
            params=[start_date, calc_date] + sids
        )
        if df.empty:
            return np.nan
        val = float(df["gm"].median())
        _MARKET_CACHE[key] = val
        return val
    except Exception:
        return np.nan


# ══════════════════════════════════════════════
# 分批資料載入
# ══════════════════════════════════════════════
def load_batch(batch_ids: list, start_date: str, start_date_10y: str, calc_date: str) -> dict:
    ph = ",".join(["%s"] * len(batch_ids))
    sp = batch_ids

    def _q(sql, params):
        try:
            with get_conn() as conn:
                return pd.read_sql(sql, conn, params=params if params else None)
        except Exception as e:
            print(f"\n    ⚠  SQL: {e}")
            return pd.DataFrame()

    dfs = {}
    dfs["stock_info"] = _q(
        f"SELECT stock_id, stock_name, industry_category FROM stock_info WHERE stock_id IN ({ph})", sp
    )
    dfs["stock_price"] = _q(
        f"SELECT date, stock_id, close, trading_volume FROM stock_price WHERE date >= %s AND date <= %s AND stock_id IN ({ph}) ORDER BY stock_id, date",
        [start_date, calc_date] + sp
    )
    dfs["stock_per"] = _q(
        f"SELECT date, stock_id, dividend_yield, per, pbr FROM stock_per WHERE date >= %s AND date <= %s AND stock_id IN ({ph}) ORDER BY stock_id, date",
        [start_date, calc_date] + sp
    )
    fin_types = ["Revenue","GrossProfit","OperatingIncome","IncomeAfterTaxes","EPS","NetIncome"]
    ft_ph = ",".join(["%s"] * len(fin_types))
    # 財報取 10 年以計算 gross_margin_10y_stability
    dfs["fin_stmt"] = _q(
        f"SELECT date, stock_id, type, value FROM financial_statements WHERE date >= %s AND date <= %s AND type IN ({ft_ph}) AND stock_id IN ({ph}) ORDER BY stock_id, date, type",
        [start_date_10y, calc_date] + fin_types + sp
    )
    bs_types = [
        "TotalAssets","Equity","Liabilities","CurrentAssets","CurrentLiabilities",
        "RetainedEarnings","PropertyPlantAndEquipment",
        "UnappropriatedRetainedEarningsAaccumulatedDeficit",
    ]
    bs_ph = ",".join(["%s"] * len(bs_types))
    dfs["balance"] = _q(
        f"SELECT date, stock_id, type, value FROM balance_sheet WHERE date >= %s AND date <= %s AND type IN ({bs_ph}) AND stock_id IN ({ph}) ORDER BY stock_id, date, type",
        [start_date_10y, calc_date] + bs_types + sp
    )
    dfs["month_rev"] = _q(
        f"SELECT date, stock_id, revenue FROM month_revenue WHERE date >= %s AND date <= %s AND stock_id IN ({ph}) ORDER BY stock_id, date",
        [start_date, calc_date] + sp
    )
    dfs["dividend"] = _q(
        f"SELECT date, stock_id, cash_earnings_distribution, cash_statutory_surplus, ratio_of_employee_stock_dividend FROM dividend WHERE date <= %s AND stock_id IN ({ph}) ORDER BY stock_id, date",
        [calc_date] + sp
    )
    dfs["insti"] = _q(
        f"SELECT date, stock_id, name, buy, sell FROM institutional_investors_buy_sell WHERE date >= %s AND date <= %s AND stock_id IN ({ph}) ORDER BY stock_id, date, name",
        [start_date, calc_date] + sp
    )
    dfs["margin"] = _q(
        f"SELECT date, stock_id, margin_purchase_today_balance, margin_purchase_limit, short_sale_today_balance, short_sale_limit FROM margin_purchase_short_sale WHERE date >= %s AND date <= %s AND stock_id IN ({ph}) ORDER BY stock_id, date",
        [start_date, calc_date] + sp
    )
    dfs["shareholding"] = _q(
        f"SELECT date, stock_id, foreign_investment_shares_ratio FROM shareholding WHERE date >= %s AND date <= %s AND stock_id IN ({ph}) ORDER BY stock_id, date",
        [start_date, calc_date] + sp
    )

    for key, df in dfs.items():
        if not df.empty:
            if "stock_id" in df.columns:
                df["stock_id"] = df["stock_id"].astype(str)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
        dfs[key] = df
    return dfs


# ══════════════════════════════════════════════
# 特徵工程輔助
# ══════════════════════════════════════════════
def _cagr(s: pd.Series, freq: float) -> float:
    s = s.dropna()
    if len(s) < 2: return np.nan
    v0, v1 = s.iloc[0], s.iloc[-1]
    if v0 <= 0 or v1 <= 0: return np.nan
    yrs = (len(s) - 1) / freq
    return np.nan if yrs <= 0 else (v1 / v0) ** (1 / yrs) - 1

def _stab(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 2 or s.mean() == 0: return np.nan
    return 1 - s.std() / abs(s.mean())

def _ts(df: pd.DataFrame, t: str) -> pd.Series:
    return (df[df["type"] == t].sort_values("date")
            .set_index("date")["value"].astype(float).dropna())

def _ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    df = df[df["a"] != 0]
    return (df["b"] / df["a"]) if not df.empty else pd.Series(dtype=float)

def _ols_slope(s: pd.Series) -> float:
    """OLS 線性趨勢斜率（正值=上升趨勢）。"""
    s = s.dropna()
    if len(s) < 3: return np.nan
    x = np.arange(len(s), dtype=float)
    try:
        slope = np.polyfit(x, s.values.astype(float), 1)[0]
        return float(slope)
    except Exception:
        return np.nan


# ══════════════════════════════════════════════
# 特徵工程輔助
# ══════════════════════════════════════════════
def _cagr(s: pd.Series, freq: float) -> float:
    s = s.dropna()
    if len(s) < 2: return np.nan
    v0, v1 = s.iloc[0], s.iloc[-1]
    if v0 <= 0 or v1 <= 0: return np.nan
    yrs = (len(s) - 1) / freq
    return np.nan if yrs <= 0 else (v1 / v0) ** (1 / yrs) - 1

def _stab(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 2 or s.mean() == 0: return np.nan
    return 1 - s.std() / abs(s.mean())

def _ts(df: pd.DataFrame, t: str) -> pd.Series:
    return (df[df["type"] == t].sort_values("date")
            .set_index("date")["value"].astype(float).dropna())

def _ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    df = df[df["a"] != 0]
    return (df["b"] / df["a"]) if not df.empty else pd.Series(dtype=float)

def _ols_slope(s: pd.Series) -> float:
    """OLS 線性趨勢斜率（正值=上升趨勢）。"""
    s = s.dropna()
    if len(s) < 3: return np.nan
    x = np.arange(len(s), dtype=float)
    try:
        slope = np.polyfit(x, s.values.astype(float), 1)[0]
        return float(slope)
    except Exception:
        return np.nan


# ══════════════════════════════════════════════
# 原有特徵計算
# ══════════════════════════════════════════════
def feat_revenue(df: pd.DataFrame, sid: str) -> dict:
    sub = df[df["stock_id"] == sid].sort_values("date").dropna(subset=["revenue"])
    if sub.empty: return {}
    rev = sub["revenue"].astype(float)
    yoy = rev.pct_change(12)
    sub2 = sub.copy(); sub2["month"] = sub2["date"].dt.month
    season_cv = sub2.groupby("month")["revenue"].mean().std() / rev.mean() if rev.mean() != 0 else np.nan
    return {
        "rev_cagr_5y":       _cagr(rev, 12),
        "rev_yoy_mean":      yoy.mean(),
        "rev_yoy_stability": _stab(yoy),
        "rev_season_cv":     season_cv,
        "rev_latest_bn":     rev.iloc[-1] / 1e9,
    }

def feat_income(df: pd.DataFrame, sid: str) -> dict:
    sub = df[df["stock_id"] == sid]
    if sub.empty: return {}
    rev   = _ts(sub, "Revenue")
    gross = _ts(sub, "GrossProfit")
    opinc = _ts(sub, "OperatingIncome")
    net   = _ts(sub, "IncomeAfterTaxes")
    eps   = _ts(sub, "EPS")
    feats = {}
    def _add(a, b, pfx):
        r = _ratio(a, b)
        if not r.empty:
            feats[f"{pfx}_mean"] = r.mean()
            feats[f"{pfx}_latest"] = r.iloc[-1]
            feats[f"{pfx}_stability"] = _stab(r)
    _add(rev, gross, "gross_margin")
    _add(rev, opinc, "op_margin")
    _add(rev, net,   "net_margin")
    if len(eps) > 0:
        feats.update({"eps_cagr":_cagr(eps,4), "eps_mean":eps.mean(),
                      "eps_latest":eps.iloc[-1], "eps_stability":_stab(eps)})
    return feats

def feat_balance(df_b: pd.DataFrame, df_f: pd.DataFrame, sid: str) -> dict:
    sub_b = df_b[df_b["stock_id"] == sid]
    sub_f = df_f[df_f["stock_id"] == sid]
    if sub_b.empty: return {}
    assets   = _ts(sub_b, "TotalAssets")
    equity   = _ts(sub_b, "Equity")
    liab     = _ts(sub_b, "Liabilities")
    cur_a    = _ts(sub_b, "CurrentAssets")
    cur_l    = _ts(sub_b, "CurrentLiabilities")
    retained = _ts(sub_b, "RetainedEarnings")
    ppe      = _ts(sub_b, "PropertyPlantAndEquipment")
    net_inc  = _ts(sub_f, "IncomeAfterTaxes")
    feats = {}
    if len(assets) > 0:
        feats["asset_cagr"] = _cagr(assets, 4)
        feats["asset_latest"] = assets.iloc[-1] / 1e9
    if len(equity) > 0:
        feats["equity_cagr"] = _cagr(equity, 4)
    if len(retained) > 0:
        feats["retained_earnings_cagr"] = _cagr(retained, 4)
        feats["retained_earnings_latest"] = retained.iloc[-1] / 1e9
    def _rp(a, b, name):
        r = _ratio(a, b)
        if not r.empty:
            feats[f"{name}_mean"] = r.mean()
            feats[f"{name}_latest"] = r.iloc[-1]
    _rp(equity, liab,    "debt_equity_ratio")
    _rp(cur_l,  cur_a,   "current_ratio")
    _rp(equity, net_inc, "roe")
    _rp(assets, net_inc, "roa")
    _rp(assets, ppe,     "ppe_to_assets")
    return feats

def feat_shareholding(df: pd.DataFrame, sid: str) -> dict:
    sub = df[df["stock_id"] == sid].sort_values("date")
    if sub.empty: return {}
    ratio = sub["foreign_investment_shares_ratio"].astype(float).dropna()
    if ratio.empty: return {}
    recent = ratio.tail(60).mean()
    prev   = ratio.iloc[:-60].mean() if len(ratio) > 60 else ratio.mean()
    return {
        "foreign_ownership_mean":      ratio.mean(),
        "foreign_ownership_latest":    ratio.iloc[-1],
        "foreign_ownership_trend":     recent - prev,
        "foreign_ownership_stability": _stab(ratio),
    }

def feat_institutional(df: pd.DataFrame, sid: str) -> dict:
    sub = df[df["stock_id"] == sid].copy()
    if sub.empty: return {}
    sub["net"] = sub["buy"].astype(float) - sub["sell"].astype(float)
    feats = {}
    for nk, lb in [("Foreign_Investor","foreign"),("Investment_Trust","trust")]:
        g = sub[sub["name"] == nk]["net"]
        if not g.empty:
            feats[f"insti_{lb}_net_mean"]     = g.mean()
            feats[f"insti_{lb}_net_positive"] = (g > 0).mean()
    total = sub[sub["name"].isin(["Foreign_Investor","Investment_Trust","Dealer_self"])].groupby("date")["net"].sum()
    if not total.empty:
        feats["insti_total_net_mean"]     = total.mean()
        feats["insti_total_net_positive"] = (total > 0).mean()
    return feats

def feat_margin(df: pd.DataFrame, sid: str) -> dict:
    sub = df[df["stock_id"] == sid].sort_values("date")
    if sub.empty: return {}
    mg_b = sub["margin_purchase_today_balance"].astype(float)
    mg_l = sub["margin_purchase_limit"].astype(float)
    ss_b = sub["short_sale_today_balance"].astype(float)
    ss_l = sub["short_sale_limit"].astype(float)
    feats = {}
    r = _ratio(mg_l, mg_b)
    if not r.empty: feats["margin_ratio_mean"] = r.mean()
    r = _ratio(ss_l, ss_b)
    if not r.empty: feats["short_ratio_mean"] = r.mean()
    r = _ratio(mg_b, ss_b)
    if not r.empty: feats["ls_ratio_mean"] = r.mean()
    return feats

def feat_price(df_p: pd.DataFrame, df_r: pd.DataFrame, sid: str) -> dict:
    sub_p = df_p[df_p["stock_id"] == sid].sort_values("date")
    sub_r = df_r[df_r["stock_id"] == sid].sort_values("date")
    feats = {}
    if not sub_p.empty:
        c = sub_p["close"].astype(float).dropna()
        lr = np.log(c / c.shift(1)).dropna()
        feats["price_annualized_vol"] = lr.std() * np.sqrt(252)
        feats["price_1y_return"]      = (c.iloc[-1] / c.iloc[max(-252,-len(c))]) - 1
        feats["price_3y_return"]      = (c.iloc[-1] / c.iloc[max(-756,-len(c))]) - 1
        feats["price_latest"]         = c.iloc[-1]
        feats["avg_turnover_vol_mn"]  = sub_p["trading_volume"].astype(float).dropna().mean() / 1e6
    if not sub_r.empty:
        for col in ["per","pbr","dividend_yield"]:
            s = sub_r[col].astype(float).dropna()
            if len(s) > 0:
                lb = "div_yield" if col == "dividend_yield" else col
                feats[f"{lb}_mean"]   = s.mean()
                feats[f"{lb}_latest"] = s.iloc[-1]
    return feats

def feat_dividend(df: pd.DataFrame, sid: str) -> dict:
    sub = df[df["stock_id"] == sid].sort_values("date")
    if sub.empty: return {}
    cash = (sub["cash_earnings_distribution"].fillna(0).astype(float) +
            sub["cash_statutory_surplus"].fillna(0).astype(float))
    if cash.sum() == 0: return {}
    return {
        "cash_div_cagr":           _cagr(cash, 1),
        "cash_div_mean":           cash.mean(),
        "cash_div_latest":         cash.iloc[-1],
        "cash_div_stability":      _stab(cash),
        "cash_div_years":          float(len(cash)),
        "cash_div_positive_ratio": (cash > 0).mean(),
        "employee_stock_div_ratio":sub["ratio_of_employee_stock_dividend"].fillna(0).astype(float).mean(),
    }


# ══════════════════════════════════════════════
# ★ 新增 10 個更「第一性」特徵
# ══════════════════════════════════════════════
def feat_advanced(dfs: dict, sid: str, industry: str, start_date: str, calc_date: str) -> dict:
    """
    10 個更第一性的特徵，直接對應「2330 為何是護國神山」的本質。
    """
    feats = {}
    sub_b  = dfs["balance"][dfs["balance"]["stock_id"] == sid]
    sub_f  = dfs["fin_stmt"][dfs["fin_stmt"]["stock_id"] == sid]
    sub_mr = dfs["month_rev"][dfs["month_rev"]["stock_id"] == sid].sort_values("date")
    sub_sh = dfs["shareholding"][dfs["shareholding"]["stock_id"] == sid].sort_values("date")
    sub_in = dfs["insti"][dfs["insti"]["stock_id"] == sid].copy()

    # 1. 毛利率溢價（vs 同產業中位數）── 定價權代理
    #    2330 的毛利率 60%+ 遠高於半導體業均值 ~35%，溢價 = 真實壁壘
    if not sub_f.empty and industry:
        rev   = _ts(sub_f, "Revenue")
        gross = _ts(sub_f, "GrossProfit")
        r     = _ratio(rev, gross)
        if not r.empty:
            ind_median = get_industry_gross_margin(industry, start_date, calc_date)
            if not np.isnan(ind_median):
                feats["gross_margin_vs_industry"] = r.mean() - ind_median

    # 2. 固定資產年增率（資本壁壘趨勢）── PPE 持續成長 = 護城河在加深
    if not sub_b.empty:
        ppe = _ts(sub_b, "PropertyPlantAndEquipment")
        if len(ppe) > 4:
            feats["ppe_growth_rate"] = _cagr(ppe, 4)

    # 3. 外資連續淨買超月數（生態鎖定深度）── 越長代表外資越難撤退
    if not sub_in.empty:
        sub_in["net"] = sub_in["buy"].astype(float) - sub_in["sell"].astype(float)
        foreign = sub_in[sub_in["name"] == "Foreign_Investor"].groupby("date")["net"].sum()
        if not foreign.empty:
            # 從最新日期往前數連續買超月數
            monthly = foreign.resample("ME").sum() if hasattr(foreign.index, 'freq') else foreign
            consecutive = 0
            for v in monthly.sort_index(ascending=False):
                if v > 0:
                    consecutive += 1
                else:
                    break
            feats["foreign_bullish_months"] = float(consecutive)

    # 4. 營收加速度（近3年 CAGR / 前3年 CAGR）── 成長是加速還是減速
    if not sub_mr.empty:
        rev = sub_mr["revenue"].astype(float)
        if len(rev) >= 72:  # 至少6年月資料
            recent3 = _cagr(rev.iloc[-36:], 12)
            prev3   = _cagr(rev.iloc[-72:-36], 12)
            if not np.isnan(recent3) and not np.isnan(prev3) and prev3 != 0:
                feats["rev_acceleration"] = recent3 / prev3

    # 5. ROIC 代理（保留盈餘/總資產）── 資本再投資效率
    if not sub_b.empty:
        retained = _ts(sub_b, "RetainedEarnings")
        assets   = _ts(sub_b, "TotalAssets")
        r = _ratio(assets, retained)
        if not r.empty:
            feats["roic_proxy"] = r.mean()

    # 6. 毛利率 10 年長期穩定度（護城河深度的關鍵指標）
    if not sub_f.empty:
        rev   = _ts(sub_f, "Revenue")
        gross = _ts(sub_f, "GrossProfit")
        r = _ratio(rev, gross)
        if len(r) >= 20:  # 至少 5 年季資料
            feats["gross_margin_10y_stability"] = _stab(r)

    # 7. 淨利率趨勢斜率（OLS）── 正值 = 持續改善
    if not sub_f.empty:
        rev = _ts(sub_f, "Revenue")
        net = _ts(sub_f, "IncomeAfterTaxes")
        r   = _ratio(rev, net)
        if len(r) >= 4:
            feats["net_margin_trend"] = _ols_slope(r)

    # 8. EPS 與營收相關性（獲利品質）── 高相關 = 獲利真實反映營運
    if not sub_f.empty and not sub_mr.empty:
        eps   = _ts(sub_f, "EPS")
        # 月營收轉季度（取同期）
        sub_mr2 = sub_mr.copy()
        sub_mr2["date"] = pd.to_datetime(sub_mr2["date"])
        sub_mr2 = sub_mr2.set_index("date")["revenue"].resample("QE").sum()
        common = eps.index.intersection(sub_mr2.index)
        if len(common) >= 8:
            feats["eps_rev_correlation"] = float(eps[common].corr(sub_mr2[common]))

    # 9. 負債比率趨勢斜率（去槓桿 = 越穩越好）
    if not sub_b.empty:
        equity = _ts(sub_b, "Equity")
        liab   = _ts(sub_b, "Liabilities")
        de = _ratio(equity, liab)
        if len(de) >= 4:
            feats["debt_trend"] = _ols_slope(de)

    # 10. 營收超額成長（YoY 相對全市場均值）── 超出越多定價權越強
    if not sub_mr.empty:
        rev = sub_mr["revenue"].astype(float)
        yoy = rev.pct_change(12).dropna()
        if not yoy.empty:
            market_avg = get_market_rev_yoy(start_date, calc_date)
            if not np.isnan(market_avg):
                feats["rev_vs_market_premium"] = yoy.mean() - market_avg

    return feats


# ══════════════════════════════════════════════
# 整合所有特徵
# ══════════════════════════════════════════════
def compute_all_features(sid: str, dfs: dict, start_date: str, calc_date: str) -> dict:
    feats = {"stock_id": sid}
    info = dfs["stock_info"]
    industry = ""
    if not info.empty and sid in info["stock_id"].values:
        row = info[info["stock_id"] == sid].iloc[0]
        feats["stock_name"]        = str(row.get("stock_name", ""))
        feats["industry_category"] = str(row.get("industry_category", ""))
        industry = feats["industry_category"]

    feats.update(feat_revenue(dfs["month_rev"],   sid))
    feats.update(feat_income(dfs["fin_stmt"],     sid))
    feats.update(feat_balance(dfs["balance"], dfs["fin_stmt"], sid))
    feats.update(feat_shareholding(dfs["shareholding"], sid))
    feats.update(feat_institutional(dfs["insti"], sid))
    feats.update(feat_margin(dfs["margin"],       sid))
    feats.update(feat_price(dfs["stock_price"], dfs["stock_per"], sid))
    feats.update(feat_dividend(dfs["dividend"],   sid))
    feats.update(feat_advanced(dfs, sid, industry, start_date, calc_date))

    return feats


# ══════════════════════════════════════════════
# Phase 2：全市場相似度計算（修正版）
# ══════════════════════════════════════════════
FEATURE_WEIGHTS = {
    # ══ 核心三大驅動（v4.4 再提升，與 predictor_light 同步）══
    "gross_margin_vs_industry":   7.0,   # 定價權溢價（5.0→7.0）
    "rev_acceleration":           6.0,   # 成長加速度（4.5→6.0）
    "foreign_bullish_months":     5.5,   # 外資連續買超（4.0→5.5）

    # ══ 第一性護城河特徵（中高權重）══
    "rev_cagr_5y":                4.5,   # 長期複合成長（4.0→4.5）
    "gross_margin_10y_stability": 4.0,   # 10年毛利率穩定（3.5→4.0）護城河深度
    "gross_margin_mean":          3.5,
    "op_margin_mean":             3.0,
    "eps_cagr":                   3.0,
    "rev_vs_market_premium":      3.0,   # 超額成長（2.5→3.0）
    "roe_mean":                   2.5,
    "roic_proxy":                 2.5,
    "ppe_growth_rate":            2.5,
    "gross_margin_stability":     2.0,
    "net_margin_trend":           2.0,
    "eps_rev_correlation":        2.0,

    # ══ 資本壁壘特徵（中等權重）══
    "ppe_to_assets_mean":         2.0,
    "asset_cagr":                 2.0,
    "retained_earnings_cagr":     2.0,
    "roa_mean":                   1.5,

    # ══ 外資籌碼特徵（中等）══
    "foreign_ownership_mean":     2.5,
    "foreign_ownership_trend":    2.0,
    "foreign_ownership_stability":1.5,
    "insti_total_net_positive":   2.0,
    "insti_foreign_net_mean":     1.5,
    "insti_trust_net_mean":       1.0,
    "price_3y_return":            1.5,
    "price_1y_return":            1.0,

    # ══ 財務紀律（偏低）══
    "cash_div_cagr":              1.5,
    "cash_div_stability":         1.5,
    "op_margin_stability":        1.5,
    "eps_stability":              1.0,
    "current_ratio_mean":         1.0,
    "debt_equity_ratio_mean":    -1.5,   # 負債越低越好（負號）
    "debt_trend":                -1.0,   # 去槓桿趨勢（負號）
    "ls_ratio_mean":             -0.5,   # 空多比越低越好（負號）
}


def compute_global_similarity(calc_date: str, benchmark_id: str = "2330") -> int:
    """
    Phase 2：從 DB 讀取全量特徵 → 全市場 Z-score → 更新 similarity_to_2330

    改進要點：
    1. 用全市場（所有股票）的 mean/std 做 Z-score，不受批次影響
    2. 基準 2330 必在計算範圍內（已在 Phase 1 首批計算）
    3. 相似度為 1 / (1 + 加權歐氏距離)，2330 自身 = 1.0
    """
    print(f"\n  Phase 2：從 DB 讀取全量特徵…", end="", flush=True)

    try:
        use_cols = list(FEATURE_WEIGHTS.keys())
        cols_in_db = ", ".join(["stock_id"] + use_cols)
        df = read_sql(
            f"SELECT {cols_in_db} FROM next_2330_features WHERE date = %s",
            params=[calc_date]
        )
    except Exception as e:
        print(f"\n  ✗  Phase 2 讀取失敗：{e}")
        return 0

    print(f"  {len(df)} 支", flush=True)

    df["stock_id"] = df["stock_id"].astype(str)

    # 確認基準存在
    if benchmark_id not in df["stock_id"].values:
        print(f"  ✗  基準 {benchmark_id} 不在特徵表中，無法計算相似度")
        return 0

    use_cols = [c for c in FEATURE_WEIGHTS if c in df.columns]
    weights  = np.array([FEATURE_WEIGHTS[c] for c in use_cols])

    numeric = df[use_cols].apply(pd.to_numeric, errors="coerce")

    # 全市場 Z-score（解決跨批次不一致）
    mu  = numeric.mean()
    std = numeric.std().replace(0, 1)
    norm = (numeric - mu) / std

    bench_idx  = df[df["stock_id"] == benchmark_id].index[0]

    # ★ NaN 修正：將 NaN 填為 0（即「落在均值上」），對距離貢獻為 0（中性）
    # 原做法：NaN 傳播 → dist=NaN → similarity=NaN（導致大量股票相似度為空）
    # 新做法：缺失特徵視為「與基準距離為0的維度」，不懲罰也不獎勵
    norm_filled  = norm.fillna(0)
    bench_filled = norm_filled.loc[bench_idx].values

    # 計算加權歐氏距離
    diff = norm_filled.values - bench_filled
    dist = np.sqrt(((diff * weights) ** 2).sum(axis=1))

    # 轉換為 0~1 相似度（基準自身 = 1.0）
    similarities = 1 / (1 + dist)

    # 診斷：印出 2330 自身相似度（應為 1.0）與非空比率
    bench_sim    = float(similarities[df.index.get_loc(bench_idx)])
    nan_before   = int(numeric.isna().any(axis=1).sum())
    print(f"  NaN 修正前有 {nan_before}/{len(df)} 支存在缺失特徵（已填0處理）")

    # 批次更新 DB
    print(f"  更新相似度…", end="", flush=True)
    with get_conn() as conn:
        with conn.cursor() as cur:
            rows = [
                (float(sim), calc_date, str(sid))
                for sim, sid in zip(similarities, df["stock_id"])
            ]
            psycopg2.extras.execute_batch(cur, UPDATE_SIM_SQL, rows, page_size=500)
        conn.commit()

    print(f"  完成（基準 {benchmark_id} 相似度 = {bench_sim:.4f}）")
    return len(rows)


# ══════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="next_2330 feature builder")
    parser.add_argument("--calc-date",  default=str(date.today()),
                        help="計算基準日（預設今日）")
    parser.add_argument("--start-date", default=None,
                        help="資料起始日（預設 calc-date 往前 10 年）")
    parser.add_argument("--stocks",     nargs="+", default=None,
                        help="只計算指定股票代號（空白分隔）")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"每批股票數（預設 {BATCH_SIZE}）")
    parser.add_argument("--force",      action="store_true",
                        help="強制重算（即使已有資料）")
    parser.add_argument("--resume",     action="store_true",
                        help="斷點續算（跳過已計算的股票）")
    parser.add_argument("--phase1-only", action="store_true",
                        help="只執行 Phase 1（計算特徵），跳過相似度更新")
    parser.add_argument("--phase2-only", action="store_true",
                        help="只執行 Phase 2（更新相似度），跳過特徵計算")
    args = parser.parse_args()

    calc_date  = args.calc_date
    start_date = args.start_date or str(
        datetime.strptime(calc_date, "%Y-%m-%d").replace(year=datetime.strptime(calc_date, "%Y-%m-%d").year - 10).date()
    )
    # 10y 視窗（用於財報 10 年穩定度）
    start_date_10y = str(
        datetime.strptime(calc_date, "%Y-%m-%d").replace(year=datetime.strptime(calc_date, "%Y-%m-%d").year - 11).date()
    )

    print(f"\n{'='*60}")
    print(f"  next_2330 feature builder")
    print(f"  calc-date  : {calc_date}")
    print(f"  start-date : {start_date}")
    print(f"{'='*60}")

    if not test_connection():
        sys.exit(1)

    # DDL + 自動遷移
    with get_conn() as conn:
        with conn.cursor() as cur:
            for stmt in DDL_FEATURES.strip().split(";"):
                s = stmt.strip()
                if s:
                    cur.execute(s)
            cur.execute(MIGRATE_SQL)
        conn.commit()
    print("  ✓ Schema 確認完成")

    # Phase 2 only
    if args.phase2_only:
        compute_global_similarity(calc_date)
        return

    # ── 取得股票清單 ─────────────────────
    if args.stocks:
        all_ids = [str(s) for s in args.stocks]
    else:
        df_ids = read_sql("SELECT DISTINCT stock_id FROM stock_info ORDER BY stock_id")
        all_ids = df_ids["stock_id"].astype(str).tolist()

    # 確保 2330 在首批（Phase 2 需要基準）
    if "2330" in all_ids:
        all_ids.remove("2330")
        all_ids.insert(0, "2330")

    # 斷點續算
    if args.resume and not args.force:
        done_df = read_sql(
            "SELECT DISTINCT stock_id FROM next_2330_features WHERE date = %s",
            params=[calc_date]
        )
        done_set = set(done_df["stock_id"].astype(str).tolist())
        all_ids = [s for s in all_ids if s not in done_set]
        print(f"  斷點續算：剩餘 {len(all_ids)} 支")

    total   = len(all_ids)
    batch_n = args.batch_size
    n_done  = 0
    n_err   = 0

    print(f"  共 {total} 支股票，每批 {batch_n} 支")
    print(f"  開始 Phase 1（計算特徵並寫入 DB）…")

    for i in range(0, total, batch_n):
        batch_ids = all_ids[i: i + batch_n]
        prog = f"[{i+1}-{min(i+batch_n, total)}/{total}]"
        print(f"  {prog} 載入資料…", end="", flush=True)

        try:
            dfs = load_batch(batch_ids, start_date, start_date_10y, calc_date)
        except Exception as e:
            print(f"\n    ✗ 資料載入失敗：{e}")
            n_err += len(batch_ids)
            continue

        rows = []
        for sid in batch_ids:
            try:
                feats = compute_all_features(sid, dfs, start_date, calc_date)
                row_vals = [calc_date, sid,
                            feats.get("stock_name", ""),
                            feats.get("industry_category", "")]
                for col in FEATURE_COLS:
                    val = feats.get(col, None)
                    row_vals.append(None if val is None or (isinstance(val, float) and np.isnan(val)) else float(val))
                row_vals.extend([None, start_date, datetime.now()])  # similarity, start_date_used, update_time
                rows.append(tuple(row_vals))
                n_done += 1
            except Exception as e:
                print(f"\n    ✗ {sid} 特徵計算失敗：{e}")
                n_err += 1

        if rows:
            try:
                with get_conn() as conn:
                    with conn.cursor() as cur:
                        psycopg2.extras.execute_values(cur, UPSERT_SQL, rows, page_size=100)
                    conn.commit()
                print(f"  寫入 {len(rows)} 筆")
            except Exception as e:
                print(f"\n    ✗ 批次寫入失敗：{e}")
                n_err += len(rows)
                n_done -= len(rows)
        else:
            print()

    print(f"\n  Phase 1 完成：成功 {n_done} 支 / 失敗 {n_err} 支")

    # Phase 2：全市場相似度
    if not args.phase1_only:
        print(f"\n  Phase 2：計算全市場相似度…")
        compute_global_similarity(calc_date)

    print(f"\n{'='*60}")
    print(f"  全部完成！calc-date={calc_date}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
