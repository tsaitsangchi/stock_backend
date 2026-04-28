"""
next_2330_predictor_light.py  v4.9（預測可信度強化）
「下一支2330」中小型成長軌跡版

Bug 修正記錄（2026-03-20）：
  ① Track B（Isolation Forest）全部顯示 ETF/DR
     根本原因：舊濾鏡邏輯 ~asset_valid | (asset <= 上限)，
               ETF 的 asset_latest=NaN → asset_valid=False → ~asset_valid=True → 被保留！
     修正：新增 _is_real_stock() 判斷函式，在資產篩選之前先過濾掉
           ETF / DR / 特別股 / 受益證券（代號規則判斷）；
           同時改為 asset_latest.notna() 強制要求資產資料存在。

  ② KeyError: 'similarity_to_2330' in 交集排名
     根本原因：df_if 本身已含 similarity_to_2330 欄位，
               再 merge df_filtered 的同名欄位 → Pandas 自動加上 _x/_y suffix
     修正：merge 前先判斷欄位是否已存在，若存在則跳過 merge。

Bug 修正記錄（2026-03-26）v4.5：
  ③ CAGR / 毛利率濾鏡邏輯反轉
     根本原因：~cagr_valid | (cagr >= min) 的語義是「NaN 就放行」，
               導致毛利率 N/A 的金融股、財報不完整的公司全部通過濾鏡。
     修正：直接用 df2["rev_cagr_5y"] >= min_rev_cagr，
           Pandas 中 NaN 比較回傳 False，NaN 股票自動排除，語義正確。

新增功能（v4.5）：
  ④ --exclude-industries：排除非科技護城河產業（金融、航運、建設等）
  ⑤ --exclude-ky：排除 KY 境外股（財報透明度與治理風險）
  ⑥ --min-foreign-bullish：設定外資連續買超月數下限
  ⑦ 可信度評級：根據候選集大小與分數分布自動輸出 HIGH / MEDIUM / LOW

預設值調整（v4.6）：
  ★ 直接執行 python next_2330_predictor_light.py 即等同於最嚴格模式：
    - 模式：純 Isolation Forest（--no-pure-if 切回加權相似度）
    - 產業排除：使用預設名單（--all-industries 關閉；自訂：--exclude-industries 金融保險業 ...）
    - KY 排除：開啟（--no-exclude-ky 關閉）
    - 雙軌模式：加 --isolation-forest 開啟

修正（v4.8）：
  ⑪ 基準訓練集改用完整資料期（預設 2012-12-31 後）
     根本原因：IsolationForest / OneClassSVM 訓練時混入 2012 年前的不完整快照
               （gross_margin_10y_stability、foreign_bullish_months 等新特徵大量 NaN），
               fillna(0) 把「資料缺失」當成「與 2330 一致的中性值」，
               嚴重污染訓練集，導致模型分數壓在 0.50x 邊界附近。
     修正：
       ① 新增 --bench-min-date 參數（預設 2012-12-31），只讀該日期之後的基準快照
       ② 讀入後額外剔除「核心特徵 NaN 比例 > 30%」的列（雙重保護）
       ③ 兩個訓練函式（run_pure_isolation_forest、run_isolation_forest）同步套用
     預期效果：訓練快照從 145 期降至約 54 期（2012→2026），
               但每一期的特徵完整率從 ~62% 提升到 ~98%，信號更純淨。
"""

import argparse
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════
# PostgreSQL 連線設定
# ══════════════════════════════════════════════
DB_CONFIG = {
    "dbname":   "stock",
    "user":     "stock",
    "password": "stock",
    "host": "localhost",
    "port":     "5432",
}

# ══════════════════════════════════════════════
# 濾鏡預設值
# ══════════════════════════════════════════════
DEFAULT_MAX_ASSET_BN     = 800
DEFAULT_MIN_REV_CAGR     = 0.12
DEFAULT_MIN_GROSS_MARGIN = 0.25

DEFAULT_EXCLUDE_STOCKS = {
    "2330", "2308", "2454", "3711", "6669",
    "2382", "2317", "2303", "2412",
    "2881", "2882", "2886", "2891", "2892",
}

# ★ 新增（v4.5）：非科技護城河產業預設排除名單
# 這些產業的高毛利率通常來自週期性、政策性或資源型因素，
# 而非可複製的技術護城河，不適合作為「下一支 2330」的候選。
DEFAULT_EXCLUDE_INDUSTRIES = {
    "金融保險業",
    "銀行業",
    "證券期貨業",
    "建材營造業",
    "航運業",
    "觀光餐旅業",
    "貿易百貨業",
}

# ══════════════════════════════════════════════
# 特徵權重（與 feature_builder 同步）
# ══════════════════════════════════════════════
FEATURE_WEIGHTS = {
    # ══ 核心三大驅動（與 feature_builder v4.4 同步）══
    "gross_margin_vs_industry":   7.0,   # 定價權溢價（5.0→7.0）
    "rev_acceleration":           7.0,   # 成長加速度（6.0→7.0）★v4.9
    "foreign_bullish_months":     5.5,   # 外資連續買超（4.0→5.5）

    # ══ 第一性護城河特徵（中高權重）══
    "rev_cagr_5y":                4.5,   # 長期複合成長（4.0→4.5）
    "gross_margin_10y_stability": 5.0,   # 10年毛利率穩定（4.0→5.0）★v4.9
    "gross_margin_mean":          3.5,
    "op_margin_mean":             3.0,
    "eps_cagr":                   4.0,   # EPS CAGR（3.0→4.0）★v4.9
    "rev_vs_market_premium":      4.0,   # 超額成長（3.0→4.0）★v4.9
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

SIM_FEATURE_COLS = list(FEATURE_WEIGHTS.keys())


# ══════════════════════════════════════════════
# DB 工具
# ══════════════════════════════════════════════
def get_conn():
    return psycopg2.connect(**DB_CONFIG)


def read_sql(sql: str, params=None) -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql(sql, conn, params=params if params else None)


def test_connection() -> bool:
    try:
        read_sql("SELECT 1")
        return True
    except Exception as e:
        print(f"✗  連線失敗：{e}")
        return False


# ══════════════════════════════════════════════
# 讀取特徵表
# ══════════════════════════════════════════════
def get_available_dates() -> list:
    df = read_sql("SELECT DISTINCT date FROM next_2330_features ORDER BY date DESC")
    return df["date"].tolist()


def load_features(calc_date: str = None) -> tuple:
    if not calc_date:
        df_d = read_sql("SELECT MAX(date) as d FROM next_2330_features")
        calc_date = str(df_d["d"].iloc[0])
        print(f"  使用最新計算基準日：{calc_date}")
    df = read_sql(
        "SELECT * FROM next_2330_features WHERE date = %s ORDER BY similarity_to_2330 DESC NULLS LAST",
        params=[calc_date]
    )
    df["stock_id"] = df["stock_id"].astype(str)
    for col in ["similarity_to_2330", "asset_latest", "rev_cagr_5y",
                "gross_margin_mean", "foreign_ownership_mean", "roe_mean", "eps_cagr",
                "gross_margin_vs_industry", "rev_acceleration", "foreign_bullish_months"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, calc_date


def load_features_multi_date(stock_ids: list) -> pd.DataFrame:
    ph = ",".join(["%s"] * len(stock_ids))
    df = read_sql(
        f"""
        SELECT date, stock_id, stock_name, industry_category,
               similarity_to_2330, asset_latest, rev_latest_bn,
               rev_cagr_5y, gross_margin_mean, gross_margin_vs_industry,
               rev_acceleration, foreign_bullish_months,
               foreign_ownership_mean, roe_mean, eps_cagr
        FROM next_2330_features
        WHERE stock_id IN ({ph})
        ORDER BY stock_id, date
        """,
        params=stock_ids
    )
    df["stock_id"] = df["stock_id"].astype(str)
    return df


# ══════════════════════════════════════════════
# ★ 修正 ① ：ETF/DR 過濾判斷
# ══════════════════════════════════════════════
def _is_real_stock(stock_id: str) -> bool:
    """
    判斷是否為真實個股，排除 ETF、DR、特別股、受益證券等。
    只有通過判斷的股票才納入候選。

    排除邏輯：
      - 長度 > 6         → ETF（00677U 等）
      - 以 "00" 開頭     → ETF（00929、006208 等）
      - 以 "0" 開頭 ≥4碼 → ETF（0050、0056 等）
      - 以 "9" 開頭 6碼  → 存託憑證（911619 等）
      - 含字母且非 KY 結尾 → 特別股/受益證券（2883B、01008T 等）
    """
    sid = str(stock_id).strip()
    if len(sid) > 6:
        return False
    if sid.startswith("00"):
        return False
    if sid.startswith("0") and len(sid) >= 4:
        return False
    if sid.startswith("9") and len(sid) == 6:
        return False
    if any(c.isalpha() for c in sid) and not sid.upper().endswith("KY"):
        return False
    return True


# ══════════════════════════════════════════════
# 中小型企業濾鏡（v4.5 強化版）
# ══════════════════════════════════════════════
def apply_smallcap_filter(
    df, benchmark, max_asset_bn, min_rev_cagr, min_gross_margin,
    extra_exclude,
    exclude_industries: set = None,
    exclude_ky: bool = False,
    min_foreign_bullish: float = None,
    min_roe: float = 0.0,
) -> tuple:
    """
    多重濾鏡：

    1. 排除基準 + 已知大公司
    2. ★ 排除 ETF / DR / 特別股（代號規則判斷）
    3. ★ 排除 asset_latest 為 NaN（財務資料缺漏）
    4. 排除 asset_latest > max_asset_bn
    5. ★★ 最低成長性（NaN 自動排除）
    6. ★★ 最低毛利率（NaN 自動排除）
    7. ★★ 排除指定產業（v4.7：改用關鍵字模糊比對，修正標籤尾綴不一致問題）
    8. ★ 排除 KY 境外股
    9. ★ 外資連續買超月數下限
   10. ★ 新增（v4.7）：最低 ROE 均值（預設 0.0，排除持續虧損公司）
    """
    exclude_set = DEFAULT_EXCLUDE_STOCKS | extra_exclude | {benchmark}
    df2 = df.copy()
    total_before = len(df2)

    # 1. 排除指定股票
    df2 = df2[~df2["stock_id"].isin(exclude_set)]
    n_after_exclude = len(df2)

    # 2. ★ 排除 ETF / DR / 特別股 / 受益證券
    df2 = df2[df2["stock_id"].apply(_is_real_stock)]
    n_after_real = len(df2)

    # 3+4. ★ asset_latest 必須存在且 <= 上限（NaN 一律排除）
    df2 = df2[df2["asset_latest"].notna() & (df2["asset_latest"] <= max_asset_bn)]
    n_after_asset = len(df2)

    # 5. ★★ 最低成長性
    if min_rev_cagr is not None:
        df2 = df2[df2["rev_cagr_5y"] >= min_rev_cagr]

    # 6. ★★ 最低毛利率
    if min_gross_margin is not None:
        df2 = df2[df2["gross_margin_mean"] >= min_gross_margin]

    n_after_growth = len(df2)

    # 7. ★★ 排除指定產業（v4.7 修正：關鍵字模糊比對）
    # 舊版：isin(exclude_industries) → 完全比對，"建材營造" ≠ "建材營造業" 就漏掉
    # 新版：任意一個關鍵字出現在 industry_category 字串內即排除
    n_before_industry = len(df2)
    if exclude_industries:
        def _industry_excluded(cat):
            if not isinstance(cat, str):
                return False
            return any(kw in cat for kw in exclude_industries)
        df2 = df2[~df2["industry_category"].apply(_industry_excluded)]
    n_after_industry = len(df2)

    # 8. ★ 排除 KY 境外股
    n_before_ky = len(df2)
    if exclude_ky and "stock_name" in df2.columns:
        df2 = df2[~df2["stock_name"].fillna("").astype(str).str.upper().str.contains("KY")]
    n_after_ky = len(df2)

    # 9. ★ 外資連續買超月數下限
    n_before_fb = len(df2)
    if min_foreign_bullish is not None and "foreign_bullish_months" in df2.columns:
        df2 = df2[df2["foreign_bullish_months"] >= min_foreign_bullish]
    n_after_fb = len(df2)

    # 10. ★ 新增（v4.7）：最低 ROE 均值（排除持續虧損公司）
    n_before_roe = len(df2)
    if min_roe is not None and "roe_mean" in df2.columns:
        df2 = df2[df2["roe_mean"] >= min_roe]
    n_after_roe = len(df2)

    summary = {
        "total_before":        total_before,
        "after_filter":        len(df2),
        "removed_exclude":     total_before - n_after_exclude,
        "removed_etf":         n_after_exclude - n_after_real,
        "removed_asset":       n_after_real - n_after_asset,
        "removed_growth":      n_after_asset - n_after_growth,
        "removed_industry":    n_before_industry - n_after_industry,
        "removed_ky":          n_before_ky - n_after_ky,
        "removed_fb":          n_before_fb - n_after_fb,
        "removed_roe":         n_before_roe - n_after_roe,
        "max_asset_bn":        max_asset_bn,
        "min_rev_cagr":        min_rev_cagr,
        "min_gross_margin":    min_gross_margin,
        "min_roe":             min_roe,
        "exclude_set_size":    len(exclude_set),
        "exclude_industries":  exclude_industries or set(),
        "exclude_ky":          exclude_ky,
        "min_foreign_bullish": min_foreign_bullish,
    }
    return df2, summary


# ══════════════════════════════════════════════
# Isolation Forest / OneClassSVM 模式
# ══════════════════════════════════════════════
def run_isolation_forest(df: pd.DataFrame, df_filtered: pd.DataFrame,
                         benchmark: str = "2330",
                         bench_min_date: str = "2012-12-31") -> pd.DataFrame:
    """
    OneClassSVM 異常檢測：
      訓練：2330 的歷史快照（bench_min_date 之後，確保特徵完整性）
      測試：中小型濾鏡後的所有候選股（已排除 ETF/DR）
      輸出：if_score（0~1，越高越像 2330 的特徵空間）
    """
    try:
        from sklearn.svm import OneClassSVM
        from sklearn.preprocessing import RobustScaler
    except ImportError:
        print("  ✗  需安裝：pip install scikit-learn")
        return df_filtered.copy().assign(if_score=np.nan)

    use_cols = [c for c in SIM_FEATURE_COLS if c in df.columns]
    if not use_cols:
        print("  ✗  無可用特徵欄位")
        return df_filtered.copy().assign(if_score=np.nan)

    # ★ v4.8：只讀 bench_min_date 之後的快照（剔除早期不完整資料）
    try:
        df_bench_all = read_sql(
            "SELECT * FROM next_2330_features WHERE stock_id = %s AND date >= %s ORDER BY date",
            params=[benchmark, bench_min_date]
        )
        df_bench_all["stock_id"] = df_bench_all["stock_id"].astype(str)
    except Exception as e:
        print(f"  ✗  讀取基準歷史失敗：{e}")
        return df_filtered.copy().assign(if_score=np.nan)

    avail_cols = [c for c in use_cols if c in df_bench_all.columns]

    # ★ v4.8：剔除核心特徵 NaN 比例 > 30% 的列（雙重保護）
    bench_numeric = df_bench_all[avail_cols].apply(pd.to_numeric, errors="coerce")
    nan_ratio = bench_numeric.isna().mean(axis=1)
    df_bench_clean = df_bench_all[nan_ratio <= 0.30]
    n_dropped = len(df_bench_all) - len(df_bench_clean)
    if n_dropped > 0:
        print(f"\n    （剔除 {n_dropped} 期低品質快照，NaN > 30%）", end="")

    X_bench = bench_numeric.loc[df_bench_clean.index].fillna(0).values

    if len(X_bench) == 0:
        print(f"  ✗  {benchmark} 無足夠歷史特徵資料（bench_min_date={bench_min_date}）")
        return df_filtered.copy().assign(if_score=np.nan)

    X_all  = df[avail_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    X_cand = df_filtered[avail_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values

    scaler = RobustScaler()
    scaler.fit(X_all)

    X_bench_s = scaler.transform(X_bench)
    X_cand_s  = scaler.transform(X_cand)

    n_bench = len(X_bench_s)
    nu = min(0.5, max(0.05, 1.0 / (n_bench + 1)))

    print(f"  OneClassSVM：{n_bench} 期 {benchmark} 快照（≥{bench_min_date}）  nu={nu:.3f}", end="", flush=True)
    ocsvm = OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
    ocsvm.fit(X_bench_s)

    scores = ocsvm.decision_function(X_cand_s)
    scores_norm = 1 / (1 + np.exp(-scores))

    df_out = df_filtered.copy()
    df_out["if_score"] = scores_norm
    print(f"  完成（{len(avail_cols)} 個特徵）")
    return df_out


# ══════════════════════════════════════════════
# 純 Isolation Forest 模式（最正確長期解法）
# ══════════════════════════════════════════════
def run_pure_isolation_forest(df: pd.DataFrame, df_filtered: pd.DataFrame,
                              benchmark: str = "2330",
                              bench_min_date: str = "2012-12-31",
                              extra_benchmarks: list = None,
                              fixed_contamination: float = 0.10) -> pd.DataFrame:
    """
    純 Isolation Forest 模式（v4.9 預測可信度強化版）：

    設計邏輯：
      - 訓練集：2330 bench_min_date 之後的歷史快照（特徵完整率高）
      - 測試集：中小型濾鏡後的候選股（已排除 ETF/DR）
      - 演算法：sklearn IsolationForest，fixed_contamination 固定值
      - 輸出：pure_if_score（0~1）+ combo_score（IF%×0.6 + 相似度%×0.4）

    v4.9 強化：
      - extra_benchmarks：補充正樣本（例：['2454', '3034']），擴充訓練集
      - fixed_contamination：固定污染率 0.10（比動態計算更嚴格，提升分化度）
      - 特徵品質篩選：只用訓練集中 NaN ≤ 20% 的特徵（提升信號純度）
    """
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import RobustScaler
    except ImportError:
        print("  ✗  需安裝：pip install scikit-learn")
        return df_filtered.copy().assign(pure_if_score=np.nan)

    all_feature_cols = [
        c for c in SIM_FEATURE_COLS
        if c in df.columns and c in df_filtered.columns
    ]
    if not all_feature_cols:
        print("  ✗  無可用特徵欄位")
        return df_filtered.copy().assign(pure_if_score=np.nan)

    # ★ v4.8：只讀 bench_min_date 之後的快照
    try:
        df_bench_all = read_sql(
            "SELECT * FROM next_2330_features WHERE stock_id = %s AND date >= %s ORDER BY date",
            params=[benchmark, bench_min_date]
        )
        df_bench_all["stock_id"] = df_bench_all["stock_id"].astype(str)
    except Exception as e:
        print(f"  ✗  讀取基準歷史失敗：{e}")
        return df_filtered.copy().assign(pure_if_score=np.nan)

    # ★ v4.9：補充正樣本（extra_benchmarks），擴充訓練集樣本量
    n_main_bench = len(df_bench_all)
    if extra_benchmarks:
        extra_parts = []
        for eb in extra_benchmarks:
            try:
                df_eb = read_sql(
                    "SELECT * FROM next_2330_features WHERE stock_id = %s AND date >= %s ORDER BY date",
                    params=[str(eb), bench_min_date]
                )
                df_eb["stock_id"] = df_eb["stock_id"].astype(str)
                extra_parts.append(df_eb)
                print(f"\n    ✓ 補充正樣本 {eb}：{len(df_eb)} 期快照", end="")
            except Exception as eb_err:
                print(f"\n    ⚠  補充正樣本 {eb} 讀取失敗：{eb_err}", end="")
        if extra_parts:
            df_bench_all = pd.concat([df_bench_all] + extra_parts, ignore_index=True)
            n_extra = len(df_bench_all) - n_main_bench
            print(f"\n    訓練集合計：{n_main_bench}（主）+ {n_extra}（補充）= {len(df_bench_all)} 期", end="")

    avail_cols = [c for c in all_feature_cols if c in df_bench_all.columns]

    # ★ v4.8：剔除核心特徵 NaN 比例 > 30% 的列
    bench_numeric = df_bench_all[avail_cols].apply(pd.to_numeric, errors="coerce")
    nan_ratio = bench_numeric.isna().mean(axis=1)
    df_bench_clean = df_bench_all[nan_ratio <= 0.30]
    n_dropped = len(df_bench_all) - len(df_bench_clean)
    if n_dropped > 0:
        print(f"\n    （剔除 {n_dropped} 期低品質快照，NaN > 30%）", end="")

    # ★ v4.9：特徵品質篩選（只用訓練集中 NaN ≤ 20% 的特徵，提升信號純度）
    feat_nan_ratio = bench_numeric.loc[df_bench_clean.index].isna().mean()
    good_features  = feat_nan_ratio[feat_nan_ratio <= 0.20].index.tolist()
    if len(good_features) >= 5:
        n_dropped_feats = len(avail_cols) - len(good_features)
        if n_dropped_feats > 0:
            print(f"\n    （排除 {n_dropped_feats} 個高缺失率特徵，保留 {len(good_features)} 個）", end="")
        avail_cols = good_features

    X_bench = bench_numeric.loc[df_bench_clean.index, avail_cols].fillna(0).values

    if len(X_bench) == 0:
        print(f"  ✗  {benchmark} 無足夠歷史快照（bench_min_date={bench_min_date}）")
        return df_filtered.copy().assign(pure_if_score=np.nan)

    X_all  = df[avail_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    X_cand = df_filtered[avail_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values

    scaler = RobustScaler()
    scaler.fit(X_all)

    X_bench_s = scaler.transform(X_bench)
    X_cand_s  = scaler.transform(X_cand)

    n_bench = len(X_bench_s)
    # ★ v4.9：固定污染率（比動態計算 1/(n+1)≈0.018 更嚴格，提升核心空間分化度）
    contamination = fixed_contamination

    extra_label = f"+{len(extra_benchmarks)}股補充" if extra_benchmarks else ""
    print(
        f"\n  IsolationForest：{n_bench} 期快照（{benchmark}{extra_label}，≥{bench_min_date}）  "
        f"contamination={contamination:.3f}  特徵數={len(avail_cols)}",
        end="", flush=True
    )

    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_bench_s)

    raw_scores = iso.decision_function(X_cand_s)
    scores_norm = 1 / (1 + np.exp(-raw_scores * 3))

    df_out = df_filtered.copy()
    df_out["pure_if_score"] = scores_norm

    # ★ v4.9：組合分數（IF 百分位排名 × 0.6 + 相似度百分位排名 × 0.4）
    # 結合無監督 IF 的資料驅動視角 + 加權相似度的領域知識，提升綜合可信度
    if "similarity_to_2330" in df_out.columns and df_out["similarity_to_2330"].notna().any():
        if_pct  = df_out["pure_if_score"].rank(pct=True, na_option="bottom")
        sim_pct = pd.to_numeric(df_out["similarity_to_2330"], errors="coerce").rank(
            pct=True, na_option="bottom"
        )
        df_out["combo_score"] = if_pct * 0.6 + sim_pct * 0.4

    print(f"  完成")
    return df_out


def print_pure_if_ranking(df_pure: pd.DataFrame, calc_date: str,
                          top_n: int, benchmark: str = "2330"):
    """
    列印純 Isolation Forest 排名報告。

    格式與 Track A 相似，但以 pure_if_score 為主排序欄位，
    並額外顯示新增的第一性特徵以利人工判讀。
    """
    print(f"\n{SEP}")
    print(f"  下一支 2330 — 純 Isolation Forest 排名（基準日：{calc_date}）")
    print(f"  模式：無人工權重，完全資料驅動，{benchmark} 全歷史快照訓練")
    print(SEP)

    if df_pure.empty or "pure_if_score" not in df_pure.columns:
        print("  ⚠  無純 IF 分數資料")
        return

    valid = df_pure.dropna(subset=["pure_if_score"])
    if valid.empty:
        print("  ⚠  所有候選股的 pure_if_score 均為 NaN")
        return

    sort_col = "combo_score" if "combo_score" in valid.columns else "pure_if_score"
    sort_label = "組合分數" if sort_col == "combo_score" else "IF分數"
    print(f"\n  ━━ 純 IF Top {top_n}（排序依據：{sort_label}，最落在 {benchmark} 歷史特徵空間）━━\n")
    print("  【欄位說明】"
          "IF分數=落在2330特徵空間的機率(0~1)  "
          "毛利溢價=vs同產業差值  "
          "成長加速=近/前3年CAGR\n")

    cols_def = [
        ("排名",    None,                          4,  ""),
        ("代號",    "stock_id",                    6,  ""),
        ("名稱",    "stock_name",                  8,  ""),
        ("產業",    "industry_category",           12, ""),
        ("IF分數",  "pure_if_score",               8,  ".4f"),
        ("組合分",  "combo_score",                 6,  ".4f"),   # ★ v4.9
        ("資產(億)", "asset_latest",               8,  ",.0f"),
        ("營收CAGR","rev_cagr_5y",                 9,  ".2%"),
        ("毛利率",  "gross_margin_mean",            7,  ".2%"),
        ("毛利溢價","gross_margin_vs_industry",     8,  "+.2%"),
        ("成長加速","rev_acceleration",             8,  ".2f"),
        ("外資月數","foreign_bullish_months",       8,  ".0f"),
        ("ROE",    "roe_mean",                     6,  ".2%"),
    ]

    header = "  " + "  ".join(f"{name:{w}}" for name, _, w, _ in cols_def)
    print(header)
    print("  " + "─" * (len(header) - 2))

    df_top = (
        valid
        .sort_values(sort_col, ascending=False)
        .head(top_n)
    )

    for rank, (_, row) in enumerate(df_top.iterrows(), 1):
        parts = []
        for name, col, w, fmt in cols_def:
            if col is None:
                parts.append(f"{rank:{w}}")
            elif fmt == "":
                parts.append(f"{str(row.get(col, '')):<{w}}")
            else:
                parts.append(f"{_fmt(row.get(col, np.nan), fmt):>{w}}")
        print("  " + "  ".join(parts))

    # ★ v4.9：三欄對照（IF分數 / 組合分數 / 加權相似度）
    if "similarity_to_2330" in df_top.columns:
        print(f"\n  📌 純IF Top {top_n} 三欄對照（IF分數 / 組合分數 / 加權相似度）：")
        print(f"  {'代號':<6}  {'IF分數':>8}  {'組合分數':>8}  {'加權相似度':>10}  名稱")
        print("  " + "─" * 50)
        for _, row in df_top.iterrows():
            print(
                f"  {str(row.get('stock_id','')):<6}  "
                f"{_fmt(row.get('pure_if_score', np.nan), '.4f'):>8}  "
                f"{_fmt(row.get('combo_score',   np.nan), '.4f'):>8}  "
                f"{_fmt(row.get('similarity_to_2330', np.nan), '.4f'):>10}  "
                f"{str(row.get('stock_name', ''))}"
            )

    # ★ 可信度評級（v4.9：同時顯示組合分數資訊）
    top_score = valid["pure_if_score"].max()
    combo_top = valid["combo_score"].max() if "combo_score" in valid.columns else None
    rating    = _confidence_rating(valid, top_score, "pure_if_score")
    print(f"\n  📊 模型可信度評級：{rating}")
    score_info = f"IF最高分 {_fmt(top_score, '.4f')}"
    if combo_top is not None:
        score_info += f"  組合最高分 {_fmt(combo_top, '.4f')}"
    print(f"     （{score_info}  候選集 {len(valid)} 支）")


# ══════════════════════════════════════════════
# 報告輸出
# ══════════════════════════════════════════════
SEP = "=" * 75


def _fmt(val, fmt: str) -> str:
    if pd.isna(val):
        return "N/A"
    try:
        return format(float(val), fmt)
    except Exception:
        return str(val)


def _confidence_rating(df_filtered: pd.DataFrame, top_score: float, score_col: str) -> str:
    """
    ★ 修正（v4.7）：可信度評級門檻區分 pure_if vs 加權相似度。

    純 IF（pure_if_score）：sigmoid 輸出，0.5 = 邊界，0.6+ 才有顯著信號
    加權相似度（similarity_to_2330）：最高約 0.10，門檻完全不同

    評分依據：
      - 候選集大小（越多越可靠）
      - 最高分是否超過顯著門檻
      - 前 5 名分數標準差（分化度，越大越有鑑別力）
    """
    n = len(df_filtered)
    scores = df_filtered[score_col].dropna()
    if scores.empty or n < 5:
        return "⚠  LOW     候選集過小（< 5 支），結果不具統計意義"

    top5_std = scores.nlargest(5).std()

    is_pure_if = (score_col == "pure_if_score")
    if is_pure_if:
        # 純 IF 用 sigmoid：0.5 = 邊界無信號，0.60+ 才代表明顯落在 2330 特徵空間內
        if top_score >= 0.60 and n >= 20 and top5_std >= 0.01:
            return "★★★ HIGH    分數高於邊界門檻，候選集充足"
        elif top_score >= 0.52 and n >= 10:
            return "★★  MEDIUM  略高於隨機邊界，建議搭配基本面驗證"
        else:
            return "★   LOW     分數接近隨機邊界（≈0.5），鑑別力不足，僅供 idea generator"
    else:
        # 加權相似度：全市場最高約 0.10，基準 2330 為 1.00
        if top_score >= 0.15 and n >= 30 and top5_std >= 0.02:
            return "★★★ HIGH    分數鑑別力高，候選集充足"
        elif top_score >= 0.08 and n >= 15:
            return "★★  MEDIUM  可作初步篩選參考，建議搭配基本面驗證"
        else:
            return "★   LOW     相似度分數普遍偏低，適合作 idea generator，不建議直接操作"


def _print_rank_table(df: pd.DataFrame, score_col: str, score_label: str, top_n: int):
    """通用排名表格列印（已套濾鏡的 df）。"""
    cols_def = [
        ("排名",   None,                         4,  ""),
        ("代號",   "stock_id",                   6,  ""),
        ("名稱",   "stock_name",                 8,  ""),
        ("產業",   "industry_category",          12, ""),
        (score_label, score_col,                 8, ".4f"),
        ("資產(億)", "asset_latest",             8,  ",.0f"),
        ("營收CAGR", "rev_cagr_5y",              9,  ".2%"),
        ("毛利率",  "gross_margin_mean",          7,  ".2%"),
        ("毛利溢價", "gross_margin_vs_industry",  8,  "+.2%"),
        ("成長加速", "rev_acceleration",          8,  ".2f"),
        ("外資月數", "foreign_bullish_months",    8,  ".0f"),
        ("ROE",    "roe_mean",                   6,  ".2%"),
    ]

    header = "  " + "  ".join(f"{name:{w}}" for name, _, w, _ in cols_def)
    print(header)
    print("  " + "─" * (len(header) - 2))

    df_top = (
        df.dropna(subset=[score_col])
        .sort_values(score_col, ascending=False)
        .head(top_n)
    )

    for rank, (_, row) in enumerate(df_top.iterrows(), 1):
        parts = []
        for name, col, w, fmt in cols_def:
            if col is None:
                parts.append(f"{rank:{w}}")
            elif fmt == "":
                parts.append(f"{str(row.get(col,'')):<{w}}")
            else:
                parts.append(f"{_fmt(row.get(col, np.nan), fmt):>{w}}")
        print("  " + "  ".join(parts))

    return df_top


def print_main_ranking(df, df_filtered, filter_summary, calc_date, top_n, benchmark,
                       df_if=None):
    """主排名報告。"""
    fs = filter_summary
    print(f"\n{SEP}")
    print(f"  下一支 2330 — 中小型成長軌跡排名 v4.9（基準日：{calc_date}）")
    print(SEP)

    print(f"\n  📐 濾鏡條件：")
    print(f"     排除大公司/基準 {fs['exclude_set_size']} 支  移除 {fs['removed_exclude']} 支")
    print(f"     排除 ETF/DR/特別股  移除 {fs['removed_etf']} 支")
    print(f"     資產 ≤ {fs['max_asset_bn']} 億元（NaN 也排除）  移除 {fs['removed_asset']} 支")
    if fs["min_rev_cagr"]:
        print(f"     月營收 CAGR ≥ {fs['min_rev_cagr']:.0%}  再移除 {fs['removed_growth']} 支低成長/NaN 股")
    if fs["min_gross_margin"]:
        print(f"     毛利率均值 ≥ {fs['min_gross_margin']:.0%}  再移除（含 NaN 者）")
    if fs.get("removed_industry", 0) > 0:
        inds = sorted(fs["exclude_industries"])
        print(f"     排除產業 {inds}  移除 {fs['removed_industry']} 支")
    if fs.get("removed_ky", 0) > 0:
        print(f"     排除 KY 境外股  移除 {fs['removed_ky']} 支")
    if fs.get("removed_fb", 0) > 0:
        print(f"     外資連續買超 ≥ {fs['min_foreign_bullish']:.0f} 個月  移除 {fs['removed_fb']} 支")
    if fs.get("removed_roe", 0) > 0:
        print(f"     ROE 均值 ≥ {fs['min_roe']:.1%}  移除 {fs['removed_roe']} 支虧損/低報酬公司")
    print(f"     → 最終剩 {fs['after_filter']} 支真實個股候選")

    # 基準摘要
    bench = df[df["stock_id"] == benchmark]
    if not bench.empty:
        b = bench.iloc[0]
        print(f"\n  📌 基準 {benchmark} {b.get('stock_name','')}：")
        print(
            f"     相似度={_fmt(b.get('similarity_to_2330'), '.4f')}  "
            f"資產={_fmt(b.get('asset_latest'), ',.0f')}億  "
            f"毛利率={_fmt(b.get('gross_margin_mean'), '.2%')}  "
            f"毛利溢價={_fmt(b.get('gross_margin_vs_industry'), '+.2%')}  "
            f"營收CAGR={_fmt(b.get('rev_cagr_5y'), '.2%')}"
        )

    if df_filtered.empty:
        print(f"\n  ⚠  濾鏡過嚴，無候選。建議調低 --min-rev-cagr 或 --min-gross-margin。")
        return

    # Track A：相似度排名
    print(f"\n  ━━ Track A：加權相似度 Top {top_n}（最像早期台積電成長軌跡）━━\n")
    print("  【欄位說明】毛利溢價=毛利率 vs 同產業差值  成長加速=近3年/前3年CAGR  外資月數=連續淨買超月數\n")
    top_a = _print_rank_table(df_filtered, "similarity_to_2330", "相似度", top_n)

    # ★ 新增（v4.5）：可信度評級
    top_score = df_filtered["similarity_to_2330"].dropna().max() if "similarity_to_2330" in df_filtered.columns else 0
    rating = _confidence_rating(df_filtered, top_score, "similarity_to_2330")
    print(f"\n  📊 模型可信度評級：{rating}")
    print(f"     （最高分 {_fmt(top_score, '.4f')}，候選集 {fs['after_filter']} 支）")

    # Track B：Isolation Forest（若有）
    if df_if is not None and "if_score" in df_if.columns:
        print(f"\n  ━━ Track B：Isolation Forest Top {top_n}（最不像其他中小型股）━━\n")
        top_b = _print_rank_table(df_if, "if_score", "IF分數", top_n)

        # ★ 修正 ②：交集排名 KeyError 修正
        ids_a = set(top_a["stock_id"])
        ids_b = set(top_b["stock_id"])
        both  = ids_a & ids_b
        if both:
            df_both = df_if[df_if["stock_id"].isin(both)].copy()

            if "similarity_to_2330" not in df_both.columns:
                df_both = df_both.merge(
                    df_filtered[["stock_id", "similarity_to_2330"]],
                    on="stock_id", how="left"
                )
            elif df_both["similarity_to_2330"].isna().all():
                df_both = df_both.drop(columns=["similarity_to_2330"])
                df_both = df_both.merge(
                    df_filtered[["stock_id", "similarity_to_2330"]],
                    on="stock_id", how="left"
                )

            df_both["combined"] = (
                pd.to_numeric(df_both["similarity_to_2330"], errors="coerce")
                  .rank(pct=True, ascending=True, na_option="bottom") +
                pd.to_numeric(df_both["if_score"], errors="coerce")
                  .rank(pct=True, ascending=True, na_option="bottom")
            )
            df_both = df_both.sort_values("combined", ascending=False)
            print(f"\n  ━━ ★ 雙軌交集候選（{len(df_both)} 支，兩個模型共同選出）★ ━━\n")
            _print_rank_table(df_both, "similarity_to_2330", "相似度", len(df_both))
        else:
            print(f"\n  （兩軌前 {top_n} 名無交集，建議放寬 --top 參數）")


def print_benchmark_detail(df: pd.DataFrame, benchmark: str = "2330"):
    bench = df[df["stock_id"] == benchmark]
    if bench.empty:
        print(f"  ⚠  {benchmark} 不在特徵表中")
        return
    b = bench.iloc[0]
    print(f"\n{SEP}")
    print(f"  {benchmark} {b.get('stock_name','')} 完整特徵摘要 v4.6")
    print(SEP)
    groups = [
        ("★ 核心三大驅動", [
            ("毛利率溢價（vs 產業中位數）", "gross_margin_vs_industry",  "+.2%"),
            ("營收加速度（近/前 3 年）",   "rev_acceleration",           ".3f"),
            ("外資連續買超月數",            "foreign_bullish_months",     ".0f"),
        ]),
        ("🏭 護城河① 技術規模壁壘", [
            ("月營收 CAGR（5年）",         "rev_cagr_5y",            ".2%"),
            ("最新月營收（億元）",          "rev_latest_bn",           ".1f"),
            ("毛利率均值",                 "gross_margin_mean",       ".2%"),
            ("毛利率 10 年穩定度",         "gross_margin_10y_stability",".3f"),
            ("毛利率穩定度",               "gross_margin_stability",  ".3f"),
            ("毛利率最新",                 "gross_margin_latest",     ".2%"),
            ("營業利益率均值",              "op_margin_mean",         ".2%"),
            ("淨利率均值",                 "net_margin_mean",         ".2%"),
            ("淨利率趨勢斜率",             "net_margin_trend",        ".6f"),
            ("EPS CAGR",                  "eps_cagr",                ".2%"),
            ("EPS 與營收相關性",           "eps_rev_correlation",     ".3f"),
            ("ROE 均值",                  "roe_mean",                ".2%"),
            ("ROA 均值",                  "roa_mean",                ".2%"),
            ("ROIC 代理",                 "roic_proxy",              ".3f"),
            ("資產 CAGR",                 "asset_cagr",              ".2%"),
            ("資產最新（億元）",            "asset_latest",           ",.0f"),
            ("固定資產年增率",             "ppe_growth_rate",         ".2%"),
            ("固定資產/資產比",            "ppe_to_assets_mean",      ".2%"),
            ("保留盈餘 CAGR",             "retained_earnings_cagr",  ".2%"),
            ("負債權益比",                "debt_equity_ratio_mean",  ".3f"),
            ("負債趨勢斜率",              "debt_trend",              ".6f"),
            ("流動比率",                  "current_ratio_mean",      ".2f"),
        ]),
        ("📡 護城河② 需求爆發生態鎖定", [
            ("外資持股比率均值（%）",      "foreign_ownership_mean",      ".2f"),
            ("外資持股最新（%）",          "foreign_ownership_latest",    ".2f"),
            ("外資持股趨勢（+加碼）",     "foreign_ownership_trend",     ".3f"),
            ("外資持股穩定度",             "foreign_ownership_stability", ".3f"),
            ("外資淨買超均值（股）",       "insti_foreign_net_mean",      ",.0f"),
            ("投信淨買超均值（股）",       "insti_trust_net_mean",        ",.0f"),
            ("三大法人買超日比率",         "insti_total_net_positive",    ".2%"),
            ("近 1 年股價報酬",            "price_1y_return",             ".2%"),
            ("近 3 年股價報酬",            "price_3y_return",             ".2%"),
            ("本益比均值",                "per_mean",                     ".1f"),
            ("股價淨值比均值",             "pbr_mean",                    ".2f"),
            ("營收超額成長 vs 市場",       "rev_vs_market_premium",       ".2%"),
        ]),
        ("💰 護城河③ 財務紀律", [
            ("現金股利 CAGR",             "cash_div_cagr",           ".2%"),
            ("現金股利穩定度",             "cash_div_stability",      ".3f"),
            ("殖利率均值（%）",            "div_yield_mean",          ".2f"),
            ("融資餘額比率",               "margin_ratio_mean",       ".4f"),
            ("空多比",                    "ls_ratio_mean",             ".4f"),
        ]),
        ("📊 模型分數", [
            ("加權相似度（Phase 2）",      "similarity_to_2330",  ".4f"),
        ]),
    ]
    for gname, items in groups:
        print(f"\n  {gname}")
        print(f"  {'─'*62}")
        for label, col, fmt in items:
            print(f"    {label:<32} {_fmt(b.get(col, np.nan), fmt)}")


def print_industry_analysis(df_filtered: pd.DataFrame, top_n: int):
    print(f"\n{SEP}")
    print("  📊 產業別分析（中小型真實個股候選集中度）")
    print(SEP)
    top = (
        df_filtered
        .dropna(subset=["similarity_to_2330"])
        .sort_values("similarity_to_2330", ascending=False)
        .head(top_n * 2)
    )
    if top.empty:
        print("  ⚠  無資料")
        return
    by_ind = (
        top.groupby("industry_category")
        .agg(
            count     = ("stock_id",           "count"),
            avg_sim   = ("similarity_to_2330",  "mean"),
            avg_asset = ("asset_latest",         "mean"),
            avg_gm    = ("gross_margin_mean",    "mean"),
            avg_cagr  = ("rev_cagr_5y",          "mean"),
            top_stock = ("stock_id",             "first"),
            top_name  = ("stock_name",           "first"),
        )
        .sort_values("count", ascending=False)
        .reset_index()
    )
    print(f"\n  {'產業':<14}  {'股數':>4}  {'相似度':>8}  "
          f"{'資產(億)':>8}  {'毛利率':>7}  {'CAGR':>7}  {'代表股':>6}  名稱")
    print("  " + "─" * 72)
    for _, row in by_ind.iterrows():
        print(
            f"  {str(row['industry_category']):<14}  {int(row['count']):>4}  "
            f"{_fmt(row['avg_sim'], '.4f'):>8}  "
            f"{_fmt(row['avg_asset'], ',.0f'):>8}  "
            f"{_fmt(row['avg_gm'], '.2%'):>7}  "
            f"{_fmt(row['avg_cagr'], '.2%'):>7}  "
            f"{str(row['top_stock']):>6}  {row['top_name']}"
        )


def print_track_report(stock_ids: list, benchmark: str):
    df = load_features_multi_date(stock_ids)
    if df.empty:
        print("  ⚠  找不到歷史特徵（需多次執行 feature_builder 累積快照）")
        return
    print(f"\n{SEP}")
    print("  📈 相似度趨勢追蹤")
    print(SEP)
    df["date"] = pd.to_datetime(df["date"])
    for sid in stock_ids:
        sub = df[df["stock_id"] == sid].sort_values("date")
        if sub.empty:
            print(f"\n  {sid}：無資料")
            continue
        name  = sub.iloc[-1].get("stock_name", "")
        asset = sub.iloc[-1].get("asset_latest", np.nan)
        print(f"\n  {sid} {name}（資產：{_fmt(asset, ',.0f')} 億元）")
        print(f"  {'日期':<12}  {'相似度':>7}  {'進度':>42}  "
              f"{'CAGR':>7}  {'毛利率':>7}  {'溢價':>7}  {'加速':>6}")
        print("  " + "─" * 98)
        for _, row in sub.iterrows():
            sim  = row.get("similarity_to_2330", np.nan)
            cagr = row.get("rev_cagr_5y", np.nan)
            gm   = row.get("gross_margin_mean", np.nan)
            gm_p = row.get("gross_margin_vs_industry", np.nan)
            acc  = row.get("rev_acceleration", np.nan)
            bar  = "█" * int((sim or 0) * 40) + "░" * max(0, 40 - int((sim or 0) * 40))
            print(
                f"  {str(row['date'])[:10]:<12}  {_fmt(sim,'.4f'):>7}  [{bar}]  "
                f"{_fmt(cagr,'.2%'):>7}  {_fmt(gm,'.2%'):>7}  "
                f"{_fmt(gm_p,'+.2%'):>7}  {_fmt(acc,'.2f'):>6}"
            )


# ══════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="下一支 2330 — 中小型成長軌跡版 v4.9（預測可信度強化）"
    )
    p.add_argument("--date",             default=None)
    p.add_argument("--top",              type=int, default=20)
    p.add_argument("--benchmark",        default="2330")
    p.add_argument("--max-asset",        type=float, default=DEFAULT_MAX_ASSET_BN)
    p.add_argument("--min-rev-cagr",     type=float, default=DEFAULT_MIN_REV_CAGR)
    p.add_argument("--min-gross-margin", type=float, default=DEFAULT_MIN_GROSS_MARGIN)
    p.add_argument("--exclude-stocks",   nargs="*", default=[])

    # ══ 模式選擇 ══
    # v4.6：pure-if 改為預設開啟；--no-pure-if 切回加權相似度
    p.add_argument("--pure-if",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="純 IF 模式（預設開啟）；--no-pure-if 切回加權相似度")
    p.add_argument("--isolation-forest", action="store_true",
                   help="雙軌模式：加權相似度 + OneClassSVM（需 scikit-learn）")

    p.add_argument("--benchmark-detail", action="store_true")
    p.add_argument("--by-industry",      action="store_true")
    p.add_argument("--track",            nargs="+", default=[])
    p.add_argument("--output",           default=None)

    # ══ 濾鏡（v4.6：全部預設開啟）══
    # --exclude-industries：
    #   未傳（預設） → 使用預設名單
    #   --exclude-industries（不帶值） → 使用預設名單（同上）
    #   --exclude-industries A B → 自訂名單
    #   --all-industries → 完全不排除任何產業
    p.add_argument("--exclude-industries", nargs="*", default=None,
                   metavar="INDUSTRY",
                   help="排除產業（預設使用內建名單）。"
                        "自訂：--exclude-industries 金融保險業 航運業；"
                        "關閉：--all-industries")
    p.add_argument("--all-industries", action="store_true", default=False,
                   help="關閉產業排除（保留所有產業進入候選集）")
    # --exclude-ky：v4.6 預設 True；--no-exclude-ky 關閉
    p.add_argument("--exclude-ky",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="排除 KY 境外股（預設開啟）；--no-exclude-ky 關閉")
    p.add_argument("--min-foreign-bullish", type=float, default=None,
                   metavar="MONTHS",
                   help="外資連續淨買超月數下限（例：--min-foreign-bullish 3）")
    # ★ 新增（v4.7）：ROE 門檻，預設排除負 ROE（虧損公司）
    p.add_argument("--min-roe", type=float, default=0.0,
                   metavar="ROE",
                   help="ROE 均值下限（預設 0.0，排除持續虧損公司）。"
                        "關閉：--min-roe -999；更嚴格：--min-roe 0.05")
    # ★ 新增（v4.8）：基準訓練集起始日
    p.add_argument("--bench-min-date", default="2012-12-31",
                   metavar="DATE",
                   help="基準訓練集起始日（預設 2012-12-31，只用完整特徵期）。"
                        "若要用全歷史：--bench-min-date 1990-01-01")
    # ★ v4.9 新增
    p.add_argument("--extra-benchmarks", nargs="*", default=[],
                   metavar="STOCK_ID",
                   help="補充正樣本（例：--extra-benchmarks 2454 3034 6415）。"
                        "將這些股票的歷史快照合併入訓練集，擴充樣本量以提升 IF 分數鑑別力。")
    p.add_argument("--contamination", type=float, default=0.10,
                   metavar="RATIO",
                   help="IsolationForest 固定污染率（預設 0.10，比動態計算更嚴格）。"
                        "更嚴格：0.15；更寬鬆：0.05")
    p.add_argument("--use-market-share", action="store_true", default=False,
                   help="整合全球市佔率評分（讀取 company_market_power_score）。"
                        "最終分數 = combo_score×0.65 + ms_score×0.35")
    return p.parse_args()


def main():
    args = parse_args()
    min_rev = args.min_rev_cagr     if args.min_rev_cagr     > 0 else None
    min_gm  = args.min_gross_margin if args.min_gross_margin > 0 else None
    extra   = set(str(s) for s in args.exclude_stocks)

    # ★ v4.6：解析 --exclude-industries / --all-industries
    # --all-industries          → 完全不排除
    # --exclude-industries A B  → 自訂名單
    # --exclude-industries（不帶值）或 未傳參數 → 使用預設名單
    if args.all_industries:
        exc_industries = set()
    elif args.exclude_industries is not None and len(args.exclude_industries) > 0:
        exc_industries = set(args.exclude_industries)
    else:
        exc_industries = DEFAULT_EXCLUDE_INDUSTRIES   # 預設：使用內建名單

    print(f"\n{SEP}")
    print("  下一支 2330 — 中小型成長軌跡版 v4.9（預測可信度強化）")
    print(f"  執行時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  濾鏡：資產≤{args.max_asset}億"
          + (f"  CAGR≥{min_rev:.0%}" if min_rev else "  CAGR不限")
          + (f"  毛利率≥{min_gm:.0%}" if min_gm else "  毛利率不限"))
    if exc_industries:
        print(f"  排除產業：{sorted(exc_industries)}")
    else:
        print(f"  產業排除：已關閉（--all-industries）")
    print(f"  排除 KY 境外股：{'是' if args.exclude_ky else '否（--no-exclude-ky）'}")
    if args.min_foreign_bullish:
        print(f"  外資買超下限：{args.min_foreign_bullish:.0f} 個月")
    min_roe = args.min_roe if args.min_roe > -999 else None
    if min_roe is not None:
        print(f"  ROE 下限：{min_roe:.1%}（--min-roe -999 可關閉）")
    print(f"  基準訓練集：{args.bench_min_date} 之後（--bench-min-date 可調整）")
    # ★ v4.6：pure-if 預設開啟；isolation-forest 優先於 pure-if
    if args.isolation_forest:
        print("  模式：加權相似度 + Isolation Forest 雙軌")
    elif args.pure_if:
        print("  模式：純 Isolation Forest（無人工權重，純資料驅動）[預設]")
    else:
        print("  模式：加權相似度（--no-pure-if 模式）")
    print(SEP + "\n")

    if not test_connection():
        sys.exit(1)

    try:
        dates = get_available_dates()
    except Exception as e:
        print(f"  ✗  讀取 next_2330_features 失敗：{e}")
        print("     請先執行：python next_2330_feature_builder.py")
        sys.exit(1)

    if not dates:
        print("  ✗  next_2330_features 無資料，請先執行 feature_builder")
        sys.exit(1)

    print(f"  可用計算日（共 {len(dates)} 期）：{[str(d)[:10] for d in dates[:5]]}")

    if args.track:
        print_track_report([str(s) for s in args.track], args.benchmark)
        return

    print(f"\n讀取特徵表…")
    df, calc_date = load_features(args.date)
    print(f"  ✓  {len(df)} 支股票 × {len(df.columns)} 個欄位")

    # ★ 更新（v4.7）：傳入 min_roe
    df_filtered, fs = apply_smallcap_filter(
        df, args.benchmark, args.max_asset, min_rev, min_gm, extra,
        exclude_industries=exc_industries,
        exclude_ky=args.exclude_ky,
        min_foreign_bullish=args.min_foreign_bullish,
        min_roe=min_roe,
    )
    print(f"  ✓  濾鏡後剩 {fs['after_filter']} 支真實個股候選"
          f"（排除 ETF/DR {fs['removed_etf']} 支）")

    if args.benchmark_detail:
        print_benchmark_detail(df, args.benchmark)

    df_if = None
    df_pure = None

    # ── 雙軌模式（isolation-forest 優先）────────────────────
    if args.isolation_forest:
        print(f"\n計算 Isolation Forest 分數…")
        df_if = run_isolation_forest(df, df_filtered, args.benchmark,
                                     bench_min_date=args.bench_min_date)
        print_main_ranking(df, df_filtered, fs, calc_date, args.top, args.benchmark, df_if)
        if args.by_industry:
            print_industry_analysis(df_filtered, args.top)

    # ── 純 IF 模式（預設）──────────────────────────────────
    elif args.pure_if:
        extra_bm = [str(s) for s in args.extra_benchmarks] if args.extra_benchmarks else []
        print(f"\n計算純 Isolation Forest 分數（無人工權重）…")
        if extra_bm:
            print(f"  補充正樣本：{extra_bm}（訓練集擴充中）")
        df_pure = run_pure_isolation_forest(
            df, df_filtered, args.benchmark,
            bench_min_date=args.bench_min_date,
            extra_benchmarks=extra_bm,
            fixed_contamination=args.contamination,
        )

        # ★ v5.0：整合全球市佔率評分
        if getattr(args, "use_market_share", False):
            try:
                df_ms = read_sql(
                    "SELECT stock_id, weighted_ms_score, market_position, "
                    "       best_market_rank, top_product_name, ai_server_exposure "
                    "FROM company_market_power_score"
                )
                df_ms["stock_id"] = df_ms["stock_id"].astype(str)
                ms_map = df_ms.set_index("stock_id")["weighted_ms_score"].to_dict()
                df_pure["ms_score"] = (
                    df_pure["stock_id"].astype(str).map(ms_map).fillna(0.15)
                )
                # ms_score 百分位排名（在候選池內）
                df_pure["ms_pct"] = df_pure["ms_score"].rank(pct=True, na_option="bottom")
                # combo_score 百分位
                if "combo_score" in df_pure.columns:
                    base_pct = df_pure["combo_score"].rank(pct=True, na_option="bottom")
                else:
                    base_pct = df_pure["pure_if_score"].rank(pct=True, na_option="bottom")
                # 最終綜合分數
                df_pure["ms_final_score"] = (
                    base_pct * 0.65 + df_pure["ms_pct"] * 0.35
                ).round(4)
                print(f"  ✓ 全球市佔率整合完成（{len(ms_map)} 支有市佔資料）")
                print(f"    已依 ms_final_score = combo(65%) + ms_score(35%) 重新排名")
                # 排序改用 ms_final_score
                if "ms_final_score" in df_pure.columns:
                    df_pure = df_pure.sort_values("ms_final_score", ascending=False)
            except Exception as e:
                print(f"  ⚠ 市佔率整合失敗（{e}），繼續使用原排名")

        print_pure_if_ranking(df_pure, calc_date, args.top, args.benchmark)
        if args.by_industry:
            sort_col = "ms_final_score" if "ms_final_score" in df_pure.columns \
                       else ("combo_score" if "combo_score" in df_pure.columns else "pure_if_score")
            df_top_pure = df_pure.nlargest(args.top * 2, sort_col)
            print_industry_analysis(df_top_pure, args.top)

    # ── 加權相似度模式（--no-pure-if）────────────────────
    else:
        print_main_ranking(df, df_filtered, fs, calc_date, args.top, args.benchmark)
        if args.by_industry:
            print_industry_analysis(df_filtered, args.top)

    if args.output:
        df_out = df.copy()
        df_out["passed_filter"] = df_out["stock_id"].isin(df_filtered["stock_id"])
        if df_if is not None and "if_score" in df_if.columns:
            df_out = df_out.merge(df_if[["stock_id", "if_score"]], on="stock_id", how="left")
        if df_pure is not None and "pure_if_score" in df_pure.columns:
            pure_out_cols = [c for c in ["pure_if_score", "combo_score"] if c in df_pure.columns]
            df_out = df_out.merge(df_pure[["stock_id"] + pure_out_cols], on="stock_id", how="left")
        df_out.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"\n  ✅ 已輸出：{args.output}（{len(df_out)} 支）")

    print(f"\n{SEP}")
    print("  常用指令（v5.0）：")
    print()
    print("  # ★ 預設模式（組合分數雙軌）")
    print("  python next_2330_predictor_light.py --top 30 --by-industry")
    print()
    print("  # ★ v5.0 整合全球市佔率（需先執行 market_share_builder --init）")
    print("  python next_2330_predictor_light.py --use-market-share --extra-benchmarks 2454 3034 --top 20")
    print()
    print("  # ★ v4.9 補充正樣本（擴充訓練集，提升 IF 分數鑑別力）")
    print("  python next_2330_predictor_light.py --extra-benchmarks 2454 3034 --top 30 --by-industry")
    print()
    print("  # 更嚴格訓練集（只用 2015 年後完整資料）")
    print("  python next_2330_predictor_light.py --bench-min-date 2015-01-01 --top 30 --by-industry")
    print()
    print("  # 加外資買超 + ROE 門檻")
    print("  python next_2330_predictor_light.py --min-foreign-bullish 3 --min-roe 0.05 --top 30 --by-industry")
    print()
    print("  # 完整報告 + 市佔率整合 + 輸出 CSV")
    print("  python next_2330_predictor_light.py --use-market-share --extra-benchmarks 2454 3034 \\")
    print("      --by-industry --output result_with_ms.csv")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
