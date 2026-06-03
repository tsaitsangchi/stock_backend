"""
market_regime_timing_validation.py v0.1 (Market-Regime / Bear-Probability TIMING Model + Walk-Forward Validation · per CLAUDE.md §一.11 三段式)
================================================================================
**最後更新日期**: 2026-06-03
**主權狀態**: MARKET-REGIME TIMING(P(bear) 擇時模型)WALK-FORWARD VALIDATION + §3.2 evaluation 橫切(NON stock-selection)+ §14.7-DC SOURCE-PURE(real FRED/price only, no impute, no hardcoded knowledge)+ §8.5 ANTI-LEAKAGE RIGOROUS(features ≤ t / label forward t+1..t+H / macro release-lag)+ §一.10 SOURCE-TRACEABLE(全 DB)+ §一.11 三段式合規 + ⚠️ HONEST CAVEAT(market timing is genuinely hard / probabilistic / US-macro-proxy / few-bear-events)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:用「總體經濟訊號(VIX 恐慌指數、殖利率曲線倒掛、失業率、工業生產…)+ 市場自身趨勢(波動度、相對均線位置、回檔幅度)」,
預測「未來 H 個交易日,整個台股市場會不會大跌(累積跌幅超過門檻)」的**機率**;機率高時建議**空手保本**,沒有非進場不可。

**它怎麼做(步驟)**:
1. 用「全部數字代號股票」的每日報酬,做一條**等權重市場指數**(這是我們要擇時的標的,也是計算多空的基準)。
2. 對每個「決策日 t」(每月月底交易日),計算 t 當下**看得到的特徵**:總經(FRED)+ 市場自身趨勢(只用 t 以前的資料)。
3. **標籤(答案)**:t 之後 H 個交易日,市場累積報酬是否 < −門檻(例如 −8%)→ 是=熊市(1),否=非熊市(0)。標籤用未來資料(t 之後)是合法的;特徵絕不可。
4. **逐月往前走(walk-forward,防洩漏核心)**:在 t 訓練時,**只用標籤已完整觀測**的樣本(即樣本決策日 ≤ t − H 個交易日,確保它的「未來 H 日」全落在 t 之前),再預測 t 當下的 P(bear)。擴張窗。
5. 兩個模型:(1) LightGBM 分類器(主)、(2) 邏輯迴歸(基準,可解釋)。兩者都報;若 LightGBM 沒贏基準,就誠實說出來。
6. 算三類成績:**精準度**(AUC / PR-AUC / Brier / 各門檻 precision-recall-命中率-假警報率 / 領先天數)、
   **信任度**(機率校準 reliability + ECE / 前後半期 AUC 穩定度 / 多 seed AUC 散布)、
   **能不能保本賺錢**(擇時策略:P(bear)<門檻就持有市場、否則空手現金,含切換成本 vs 買進持有;CAGR/波動/Sharpe/MaxDD/Calmar/在市場時間% + 各大熊市[2000/2008/2020/2022]的表現)。
7. 誠實結論:擇時相對「買進持有」**有沒有真的降低最大回檔 / 改善 Calmar 或 Sharpe**。

**輸入**:資料庫(`TaiwanStockPriceAdj` 股價 + `fred_series` 總經)。**輸出**:JSON(各 horizon 三類成績 + 誠實警語)+ 可讀 stdout 摘要表。
**它不做的事**:不選股(這跟既有的 multi_cycle_*_validation cross-sectional 選股驗證**本質不同** — 它判定「要不要在市場裡」而非「買哪幾支」);
不改資料庫(純讀取評估;§3.2 evaluation 角色);不偽造/不補值(缺資料留 NaN);不寫死任何 domain knowledge。
**為什麼需要它**:選股模型回答「買哪些」,擇時模型回答「現在該不該在場內」;兩者正交,合起來才完整。但擇時本質困難,本工具只做**誠實的 walk-forward 研究驗證,非保證獲利**。

## 📜 一、核心定義說明 (Core Definitions / The Constitution)

1. **[Sovereignty Declaration]** (v0.1, 憲法 §3.2 橫切分析模組): 本程式為 evaluation 後處理工具,**非** §9.1 cross-sectional 選股落地鏈;預測 P(bear)=「要不要在市場裡」,非「買哪幾支股」。
   **治權邊界**:(a) §3.2 evaluation 模組;(b) 不涉五套禁令(§0.1/§0.2/§0.3);(c) T1-T3 不分層;(d) §8.5 anti-leakage(見第 4 條);
   (e) **不訓練 production model**(不寫 model_registry / model_training_run);(f) **read-only**(不改 `TaiwanStockPriceAdj` / `fred_series` / universe);
   (g) 唯一職責:market-regime P(bear) walk-forward 預測 + precision/trust/economic 三類 metrics + JSON 持久化。
2. **[Source Purity]** (v0.1, CLAUDE.md §一.10 + §一.13 / 憲法 §14.7-DC): 全特徵 = 真實 FRED(`fred_series`)/ price(`TaiwanStockPriceAdj`)raw value 經 mathematical transform(log-return / z-score / rolling vol / MA ratio / yoy % / drawdown)。
   **無 hardcoded knowledge dict**(無 THEME_KEYWORDS 類);**無 imputed / 無 fabricate**:某 series 在 t 缺值 → 留 NaN(LightGBM 原生處理 NaN;LogisticRegression 分支以訓練集 median 填補 **僅限該 baseline**,且 fit-time median 不洩漏未來);**FRED 為美國總經 → 台股之 proxy**(見誠實警語)。
3. **[Forward-Looking Label, No Leakage]** (v0.1): `bear_ahead[t] = 1{ market_index[t+H]/market_index[t] − 1 < −threshold }`。標籤刻意使用**未來** H 個交易日(t+1..t+H)— 對 LABEL 合法;但特徵(第 4 條)絕不可觸及 > t 之資料。H 以**交易日位置**精確計(非 calendar-day 近似)。
4. **[Anti-Leakage RIGOROUS — #1 correctness]** (v0.1, 憲法 §8.5): 三道結構性防線:
   (i) **features ≤ t**:決策日 t 之特徵僅用 ≤ t 之 price + ≤ release-lagged macro;
   (ii) **macro release-lag**:月頻 FRED 序列(UNRATE/UMCSENT/INDPRO/M2SL)有發布延遲 → as-of 月 M 之值僅自 **M+1 月** 起可見(LAG +1 month);日頻序列(VIX/殖利率)forward-fill 至日,不前視;
   (iii) **train/predict 邊界**:在決策日 t(交易日位置 `p_t`)訓練時,**僅納入標籤已完整觀測之樣本** — 樣本決策日位置 `p_s` 須滿足 `p_s + H ≤ p_t`(該樣本之未來 H 日標籤窗 `[p_s+1..p_s+H]` 完全落在 t 當下可見之歷史內);predict P(bear) at t;expanding window。**絕不**以 > 樣本自身日期之資料算其特徵,**絕不**訓練於標籤窗延伸超過 t 之樣本。首次預測須滿足最小歷史(≥ 2002 + ≥ ~120 訓練月,確保 ≥1 先前熊市)。
5. **[Two Models, Honest Comparison]** (v0.1): (1) PRIMARY = `lightgbm.LGBMClassifier`(modest:n_est~200/num_leaves~15/lr~0.03/正則/`scale_pos_weight=n_neg/n_pos` 處理不平衡/seed);
   (2) BASELINE = sklearn `LogisticRegression`(標準化 + L2,robust/interpretable)。**兩者皆報**;若 LightGBM out-of-sample 沒贏 Logistic,**誠實說明**(這本身是有價值的訊號)。
6. **[Precision / Trust / Economic 三分]** (v0.1): 精準度(ROC-AUC / PR-AUC / Brier / 各 operating threshold precision-recall-hit-FPR / lead-time)、
   信任度(10-bin reliability + ECE / 前後半期 AUC 穩定度 / 多 seed AUC mean-spread)、
   經濟性(擇時 vs 買進持有:CAGR/vol/Sharpe/MaxDD/Calmar/在市場%/切換次數 + 各已知熊市分段表現)分開報告 → 回答「真的能保本/賺錢嗎?」。headline 誠實裁決 = 擇時相對買進持有**是否降低 MaxDD 且/或改善 Calmar/Sharpe**。
7. **[Zero Hardcoded Verdict]** (v0.1, 憲法 §5.6.3): significance / verdict 全動態計算;operating thresholds {0.3,0.5,0.7} 與 bear 門檻為 CLI 可配置之 transparent reference(Tier 3 disclosure,非 feature data,非硬編 PASS/FAIL)。
   已知熊市分段窗(2000/2008/2020/2022)為**歷史事實標記**(供分段報告),非預測輸入、非 feature。
8. **[§一.10 #3 Multi-Run]** (v0.1): LightGBM 含 stochasticity → 多 seed {5422,7331,1009} 報 AUC mean/spread;single-run 不得作為 deterministic fact。
9. **[Conservative Economic Assumption]** (v0.1): 空手 = 現金 0% 報酬(無 risk-free 收益)→ 對擇時策略**保守**(低估其報酬);切換含小額成本(預設 0.1%/次)。
10. **[⚠️ HONEST CAVEAT — Market Timing Is Hard]** (v0.1): market timing 本質困難;輸出為**機率非確定**;FRED 總經為**美國** proxy(台股無對等公開高頻總經);**熊市事件少**(34 年僅數次)→ 估計**不確定性大**;本工具為 **walk-forward 研究驗證,非獲利保證**;cash=0 假設。所有數據 trace 回 (a) program output / (b) DB query。
11. **[Historical Reference Authority]** (v0.1): TOOL_VER = "v0.1" 為記述性快照,非權威來源。
12. **[Idempotency]** (v0.1): 不寫 DB(--output 寫 JSON);可重跑;horizon / threshold / seed / smoke 全可配置。

## 📊 二、全量功能群矩陣 (The Ultimate Functional Group Matrix)

### Group A. Market Series & Label(source-pure 擇時標的 + 前瞻標籤)

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| A.1 Equal-weight market | build_market_series() → 全 numeric stock 每日 log-return → 跨股等權平均 → cumprod 指數 | §一.10 (b) DB / §14.7-DC source-pure |
| A.2 Trading-day index | market_dates(ordered)→ 整數位置做 H-交易日算術(非 calendar 近似)| §8.5 精確 horizon |
| A.3 Forward label | bear_ahead[t]=1{ fwd-cum-ret over t+1..t+H < −threshold } | 第 3/4 條(label forward, features ≤ t)|

### Group B. Features(macro release-lagged + market-self trailing,全 ≤ t)

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| B.1 Macro daily | VIX level+60d-z / T10Y2Y,T10Y3M level+inverted / DGS2,DGS3MO / T10YIE(ffill 至日,不前視)| §一.10 / §8.5 |
| B.2 Macro monthly(LAG +1m)| UNRATE level+12m-chg / UMCSENT / INDPRO 12m% / M2SL 12m%(發布延遲)| 第 4 條(ii)release-lag |
| B.3 Market-self trailing | realized vol 20/60d / trailing ret 20/60/120d / close/MA50 / close/MA200 / drawdown-from-252d-peak | 僅 ≤ t |
| B.4 NaN policy | 缺值留 NaN(LightGBM 原生);Logistic 分支用 fit-time train-median 填補(不洩漏)| §一.13 no-impute(主模型)|

### Group C. Walk-Forward & Models(anti-leakage 核心)

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| C.1 Decision dates | 月底交易日(monthly)| expanding window |
| C.2 Train boundary | 樣本 p_s + H ≤ p_t 才納入訓練(標籤完整觀測)| 第 4 條(iii)|
| C.3 LightGBM | LGBMClassifier(scale_pos_weight / 多 seed)| 第 5/8 條 |
| C.4 Logistic | StandardScaler(fit on train)+ LogisticRegression(L2)| 第 5 條 baseline |

### Group D. Metrics & Persistence

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| D.1 Precision | ROC-AUC / PR-AUC / Brier / per-thr precision-recall-hit-FPR / lead-time | 第 6 條 |
| D.2 Trust | reliability(10-bin)+ ECE / sub-period AUC / multi-seed spread | 第 6/8 條 |
| D.3 Economic | timed vs buy-hold:CAGR/vol/Sharpe/MaxDD/Calmar/in-mkt%/switches + per-episode | 第 6/9 條 |
| D.4 JSON | --output {per_horizon:{precision,trust,economic},_meta(含 honest caveat)} | §一.10 / 第 10 條 |
| D.5 stdout | 可讀摘要表(AUC / calibration / timed-vs-buyhold / per-episode)| §一.12 |

### 對齊憲章 §二 維運矩陣

| 場景 | 命令 |
| :--- | :--- |
| Smoke(plumbing 驗證)| `python scripts/evaluation/market_regime_timing_validation.py --smoke --output reports/market_regime_timing_smoke.json` |
| 完整(全 horizon 20/60/120)| `python scripts/evaluation/market_regime_timing_validation.py --output reports/market_regime_timing.json` |
| 單 horizon | `python scripts/evaluation/market_regime_timing_validation.py --horizon 60 --threshold 0.08 --output reports/mrt_h60.json` |

### 不提供之旗標 (Intentionally Omitted)

- `--commit`:本工具不寫 evaluation_log / model_registry(屬 model_trainer 治權;§3.2 橫切只讀)。`--dry-run`/`--commit` 接受但語意 = 純計算列印;僅 `--output` 寫 JSON。
- cross-sectional 選股旗標(--n-top 等):本工具為 market-timing,**非選股**,刻意不提供。

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.1 | 2026-06-03 | Codex | **首版**:market-regime / bear-probability 擇時模型 + walk-forward 驗證。等權重市場序列(全 numeric stock 1992-2026)為擇時標的 + 前瞻 bear 標籤(t+1..t+H cum-ret < −threshold)。特徵 = FRED 總經(release-lagged)+ 市場自身趨勢(全 ≤ t)。LightGBM(主)+ Logistic(基準)。三道 anti-leakage 防線(features ≤ t / macro release-lag / 訓練樣本 p_s+H ≤ p_t)。三類 metrics:精準度(AUC/PR-AUC/Brier/per-thr/lead-time)+ 信任度(reliability+ECE / sub-period AUC / multi-seed)+ 經濟性(timed vs buy-hold MaxDD/Sharpe/Calmar + per-episode 2000/2008/2020/2022)。§一.10 全 DB source-traceable;§一.11 三段式合規;⚠️ 誠實警語入 _meta。 | **ACTIVE** |
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")  # NumPy log(NaN) + sklearn/lightgbm non-fatal warnings
import os
import sys, argparse, math, json, logging, time
from datetime import date, datetime
from pathlib import Path

_base_dir = Path(__file__).resolve().parent.parent  # scripts/
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

PROJECT_ROOT = _base_dir.parent  # repo root (parent of scripts/)
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")  # explicit env load (db_utils also loads at import; idempotent)
except Exception:
    pass

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from core.db_utils import get_db_conn  # noqa: E402  (env-load side effect intended)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

CONSTITUTION_VER = "v6.1.0"
TOOL_VER = "v0.1"

# ── Reference constants (Tier 3 transparent disclosure — NOT feature data, NOT hardcoded verdict) ──
DEFAULT_HORIZONS = [20, 60, 120]          # forward trading-day windows for bear label / eval
DEFAULT_THRESHOLD = 0.08                  # bear if forward cum-return < −8% over the window
OPERATING_THRESHOLDS = [0.3, 0.5, 0.7]    # probability cut-points for precision/recall + timing
SWITCH_COST = 0.001                       # 0.1% per market<->cash switch (timing strategy)
MIN_TRAIN_MONTHS = 120                    # ≥ ~10yr expanding history before first OOS prediction
MIN_OOS_YEAR = 2002                       # first OOS decision year (≥1 prior bear episode in train)
TRADING_DAYS_YEAR = 252
SEEDS_MULTI = [5422, 7331, 1009]
ECE_BINS = 10

# Monthly FRED series carry a publication delay → an as-of month M value is only
# observable from month M+1. We LAG these by +1 month before forward-filling to daily.
MACRO_MONTHLY_LAGGED = {"UNRATE", "UMCSENT", "INDPRO", "M2SL"}
# Daily FRED series: forward-fill to the trading-day grid (no look-ahead; last published value).
MACRO_DAILY = ["VIXCLS", "T10Y2Y", "T10Y3M", "DGS2", "DGS3MO", "T10YIE"]
MACRO_ALL = MACRO_DAILY + ["UNRATE", "UMCSENT", "INDPRO", "M2SL"]

# Known bear episodes (historical FACTS for per-episode reporting only — NOT a feature, NOT a predictor input).
BEAR_EPISODES = [
    ("2000_dotcom", date(2000, 1, 1), date(2002, 12, 31)),
    ("2008_gfc", date(2007, 10, 1), date(2009, 6, 30)),
    ("2020_covid", date(2020, 1, 1), date(2020, 6, 30)),
    ("2022_tightening", date(2022, 1, 1), date(2022, 12, 31)),
]


# ════════════════════════════════════════════════════════════════════════════
# Group A — Market series (equal-weight, source-pure) + forward bear label
# ════════════════════════════════════════════════════════════════════════════
def build_market_series(cur):
    """全 numeric stock 每日 log-return → 跨股等權平均 → cumprod 指數 (source-pure, §一.10 (b) DB).

    每股 daily log-return = ln(close_t/close_{t-1}) 僅在連續交易兩日皆有 close>0 時定義(groupby-diff,
    NaN at each stock's first row + 跨缺口處)→ market_ret[d] = 該日有有效 return 之全股等權平均
    (= 同時在 d-1 與 d 交易之股票)。市場序列無 leakage(每日僅用 d 與 d-1 之 close)。

    Returns:
        market_dates: np.ndarray[date]  排序後之市場交易日(有 market_ret 之日)
        market_ret:   np.ndarray[float] 每日等權 log-return
        market_index: np.ndarray[float] cumprod 指數(= exp(cumsum(log-return)),起點正規化為 1.0 之前一日)
    """
    cur.execute('SELECT stock_id, date, close FROM "TaiwanStockPriceAdj" '
                "WHERE stock_id ~ '^[0-9]' AND close>0 ORDER BY stock_id, date")
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["stock_id", "date", "close"])
    df["logc"] = np.log(df["close"].astype(float).values)
    # per-stock consecutive-day log return (diff stays within each stock_id; NaN at first row of each stock)
    df["ret"] = df.groupby("stock_id", sort=False)["logc"].diff()
    g = df.dropna(subset=["ret"]).groupby("date")["ret"].mean().sort_index()
    market_dates = np.array(list(g.index))
    market_ret = g.values.astype(float)
    market_index = np.exp(np.cumsum(market_ret))   # index level (relative; level on first market day)
    return market_dates, market_ret, market_index


def make_bear_label(market_index, horizon):
    """bear_ahead[i] = 1 if forward cumulative return over next `horizon` trading days < −threshold.

    Returns a function-friendly array of forward cumulative SIMPLE returns over [i+1 .. i+horizon]
    measured in trading-day positions (NaN where the full window extends beyond the last observation).
    The caller compares against −threshold. Using FUTURE data here is intentional and correct for a LABEL.
    """
    n = len(market_index)
    fwd_cum_ret = np.full(n, np.nan)
    for i in range(n):
        j = i + horizon
        if j < n:
            fwd_cum_ret[i] = market_index[j] / market_index[i] - 1.0
    return fwd_cum_ret


# ════════════════════════════════════════════════════════════════════════════
# Group B — Features (macro release-lagged + market-self trailing), all ≤ t
# ════════════════════════════════════════════════════════════════════════════
def load_macro_daily_frame(cur, market_dates):
    """Build a daily macro frame aligned to market trading dates (forward-fill, release-lag for monthly).

    - Daily series: as published; forward-filled to the trading-day grid (last value ≤ date → no look-ahead).
    - Monthly series in MACRO_MONTHLY_LAGGED: shift observation date +1 month (publication delay) THEN ffill.
    Missing → NaN (not imputed; source-pure). Returns a DataFrame indexed by market trading date.
    """
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in market_dates])
    frame = pd.DataFrame(index=idx)
    for sid in MACRO_ALL:
        cur.execute("SELECT date, value FROM fred_series WHERE series_id=%s ORDER BY date", (sid,))
        recs = cur.fetchall()
        if not recs:
            frame[sid] = np.nan
            continue
        s = pd.Series([float(v) if v is not None else np.nan for _, v in recs],
                      index=pd.DatetimeIndex([pd.Timestamp(d) for d, _ in recs])).sort_index()
        if sid in MACRO_MONTHLY_LAGGED:
            # publication delay: month-M value only visible from month M+1 → shift the observation date forward 1 month
            s.index = s.index + pd.DateOffset(months=1)
            s = s[~s.index.duplicated(keep="last")].sort_index()
        # forward-fill onto the trading-day grid: each market date sees the latest value dated ≤ that date
        aligned = s.reindex(s.index.union(idx)).sort_index().ffill().reindex(idx)
        frame[sid] = aligned.values
    return frame


def _zscore_60d(series):
    m = series.rolling(60, min_periods=20).mean()
    sd = series.rolling(60, min_periods=20).std(ddof=1)
    return (series - m) / sd.replace(0.0, np.nan)


def _yoy_pct(series_daily, lookback_days):
    """trailing YoY/period % change on a forward-filled daily series (value_t / value_{t-lookback} − 1)."""
    return series_daily / series_daily.shift(lookback_days) - 1.0


def build_feature_frame(market_dates, market_ret, market_index, macro_frame):
    """Compute the full as-of feature matrix indexed by market trading date.

    Every feature uses ONLY data dated ≤ the row's own trading date:
      - market-self: trailing realized vol / trailing returns / MA ratios / current drawdown (all backward windows)
      - macro: release-lagged (monthly) + forward-filled (daily) levels & transforms
    No future leakage: all rolling/shift windows look backward only.
    """
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in market_dates])
    ret = pd.Series(market_ret, index=idx)
    px = pd.Series(market_index, index=idx)
    feat = pd.DataFrame(index=idx)

    # ── market-self (trailing, ≤ t) ──
    feat["mkt_vol_20d"] = ret.rolling(20, min_periods=10).std(ddof=1) * math.sqrt(TRADING_DAYS_YEAR)
    feat["mkt_vol_60d"] = ret.rolling(60, min_periods=30).std(ddof=1) * math.sqrt(TRADING_DAYS_YEAR)
    feat["mkt_ret_20d"] = px / px.shift(20) - 1.0
    feat["mkt_ret_60d"] = px / px.shift(60) - 1.0
    feat["mkt_ret_120d"] = px / px.shift(120) - 1.0
    feat["mkt_close_ma50"] = px / px.rolling(50, min_periods=25).mean() - 1.0
    feat["mkt_close_ma200"] = px / px.rolling(200, min_periods=100).mean() - 1.0
    roll_peak_252 = px.rolling(TRADING_DAYS_YEAR, min_periods=60).max()
    feat["mkt_drawdown_252d"] = px / roll_peak_252 - 1.0   # ≤ 0 (current drawdown from trailing 1yr peak)

    # ── macro levels + transforms (release-lagged via macro_frame; backward windows only) ──
    m = macro_frame
    feat["vix_level"] = m["VIXCLS"]
    feat["vix_z60"] = _zscore_60d(m["VIXCLS"])
    feat["t10y2y_level"] = m["T10Y2Y"]
    feat["t10y2y_inverted"] = (m["T10Y2Y"] < 0).astype(float).where(m["T10Y2Y"].notna(), np.nan)
    feat["t10y3m_level"] = m["T10Y3M"]
    feat["t10y3m_inverted"] = (m["T10Y3M"] < 0).astype(float).where(m["T10Y3M"].notna(), np.nan)
    feat["dgs2_level"] = m["DGS2"]
    feat["dgs3mo_level"] = m["DGS3MO"]
    feat["t10yie_level"] = m["T10YIE"]
    feat["unrate_level"] = m["UNRATE"]
    feat["unrate_chg_12m"] = m["UNRATE"] - m["UNRATE"].shift(TRADING_DAYS_YEAR)   # ~12m change on ffilled daily grid
    feat["umcsent_level"] = m["UMCSENT"]
    feat["indpro_yoy"] = _yoy_pct(m["INDPRO"], TRADING_DAYS_YEAR)
    feat["m2sl_yoy"] = _yoy_pct(m["M2SL"], TRADING_DAYS_YEAR)
    return feat


FEATURE_COLS = [
    "mkt_vol_20d", "mkt_vol_60d", "mkt_ret_20d", "mkt_ret_60d", "mkt_ret_120d",
    "mkt_close_ma50", "mkt_close_ma200", "mkt_drawdown_252d",
    "vix_level", "vix_z60", "t10y2y_level", "t10y2y_inverted", "t10y3m_level", "t10y3m_inverted",
    "dgs2_level", "dgs3mo_level", "t10yie_level",
    "unrate_level", "unrate_chg_12m", "umcsent_level", "indpro_yoy", "m2sl_yoy",
]


def month_end_decision_positions(market_dates):
    """Monthly decision dates = the last trading-day position within each calendar month."""
    ser = pd.Series(range(len(market_dates)),
                    index=pd.DatetimeIndex([pd.Timestamp(d) for d in market_dates]))
    last_pos = ser.groupby([ser.index.year, ser.index.month]).max()
    return sorted(int(p) for p in last_pos.values)


# ════════════════════════════════════════════════════════════════════════════
# Group C — Walk-forward (ANTI-LEAKAGE CORE) + models
# ════════════════════════════════════════════════════════════════════════════
def _fit_lightgbm(X_tr, y_tr, seed, n_estimators, smoke):
    n_pos = int(y_tr.sum()); n_neg = int(len(y_tr) - n_pos)
    spw = (n_neg / n_pos) if n_pos > 0 else 1.0
    clf = lgb.LGBMClassifier(
        n_estimators=(50 if smoke else n_estimators), num_leaves=15, learning_rate=0.03,
        min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.5, scale_pos_weight=spw, random_state=seed,
        n_jobs=-1, verbosity=-1,
    )
    clf.fit(X_tr, y_tr)
    return clf


def _fit_logistic(X_tr, y_tr):
    """Standardized L2 logistic baseline. NaN → train-median imputation (fit-time only, no future leak)."""
    med = np.nanmedian(X_tr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xf = np.where(np.isfinite(X_tr), X_tr, med)
    scaler = StandardScaler().fit(Xf)
    Xs = scaler.transform(Xf)
    clf = LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", max_iter=2000)
    clf.fit(Xs, y_tr)
    return clf, scaler, med


def _predict_logistic(model_tuple, X):
    clf, scaler, med = model_tuple
    Xf = np.where(np.isfinite(X), X, med)
    return clf.predict_proba(scaler.transform(Xf))[:, 1]


def walk_forward(feat, fwd_cum_ret, market_dates, decision_positions, horizon, threshold,
                 seed, n_estimators, smoke, multi_seed=False):
    """Expanding-window walk-forward with RIGOROUS anti-leakage.

    For each decision position p_t (monthly, ≥ MIN_OOS_YEAR & ≥ MIN_TRAIN_MONTHS history):
      TRAIN set = all earlier decision samples p_s with  p_s + horizon ≤ p_t
                  (label window [p_s+1 .. p_s+horizon] fully observed using only data dated ≤ t),
                  features taken as-of each p_s (≤ p_s data only).
      PREDICT P(bear) at p_t (features as-of p_t; its own label is NOT used / not yet observed).
    Returns OOS arrays: dates, y_true (realized bear), p_lgbm, p_logit, (+per-seed p_lgbm if multi_seed).
    """
    feat_vals = feat[FEATURE_COLS].values  # row i ↔ market_dates[i]
    label = (fwd_cum_ret < -threshold).astype(float)  # bear_ahead per trading-day position
    label_observed = np.isfinite(fwd_cum_ret)         # True where forward window fully exists

    samples = [p for p in decision_positions]
    oos_dates, y_true = [], []
    p_lgbm, p_logit = [], []
    p_seed = {s: [] for s in SEEDS_MULTI} if multi_seed else None

    for p_t in samples:
        t_date = pd.Timestamp(market_dates[p_t])
        if t_date.year < MIN_OOS_YEAR:
            continue
        if not label_observed[p_t]:
            continue  # cannot evaluate this decision date (its forward window not yet realized)
        # TRAIN: earlier monthly samples whose label window ends at/before p_t (fully observed as-of t)
        train_pos = [p_s for p_s in samples
                     if p_s + horizon <= p_t and label_observed[p_s]]
        if len(train_pos) < MIN_TRAIN_MONTHS:
            continue
        Xtr = feat_vals[train_pos]
        ytr = label[train_pos]
        # require both classes present and feature rows with at least some finite values
        row_ok = np.isfinite(Xtr).any(axis=1)
        Xtr, ytr = Xtr[row_ok], ytr[row_ok]
        if len(np.unique(ytr)) < 2 or len(ytr) < MIN_TRAIN_MONTHS:
            continue
        Xte = feat_vals[p_t:p_t + 1]
        if not np.isfinite(Xte).any():
            continue

        # PRIMARY lightgbm
        clf = _fit_lightgbm(Xtr, ytr, seed, n_estimators, smoke)
        p_l = float(clf.predict_proba(Xte)[:, 1][0])
        # BASELINE logistic
        lt = _fit_logistic(Xtr, ytr)
        p_lr = float(_predict_logistic(lt, Xte)[0])

        oos_dates.append(market_dates[p_t])
        y_true.append(float(label[p_t]))
        p_lgbm.append(p_l)
        p_logit.append(p_lr)
        if multi_seed:
            for s in SEEDS_MULTI:
                cs = _fit_lightgbm(Xtr, ytr, s, n_estimators, smoke)
                p_seed[s].append(float(cs.predict_proba(Xte)[:, 1][0]))

    return {
        "dates": oos_dates, "y_true": np.array(y_true),
        "p_lgbm": np.array(p_lgbm), "p_logit": np.array(p_logit),
        "p_seed": ({s: np.array(v) for s, v in p_seed.items()} if multi_seed else None),
    }


# ════════════════════════════════════════════════════════════════════════════
# Group D — Metrics: precision / trust / economic
# ════════════════════════════════════════════════════════════════════════════
def _safe_auc(y, p):
    if len(np.unique(y)) < 2 or len(y) < 3:
        return None
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return None


def precision_block(y, p):
    out = {
        "n_oos": int(len(y)), "n_bear": int(y.sum()), "bear_base_rate": float(np.mean(y)) if len(y) else None,
        "roc_auc": _safe_auc(y, p),
        "pr_auc": (float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else None),
        "brier": (float(brier_score_loss(y, p)) if len(y) else None),
        "by_threshold": {},
    }
    for thr in OPERATING_THRESHOLDS:
        pred = (p >= thr).astype(float)
        tp = float(np.sum((pred == 1) & (y == 1))); fp = float(np.sum((pred == 1) & (y == 0)))
        fn = float(np.sum((pred == 0) & (y == 1))); tn = float(np.sum((pred == 0) & (y == 0)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None      # = hit-rate for bear class
        fpr = fp / (fp + tn) if (fp + tn) > 0 else None
        hit_rate = float(np.mean(pred == y)) if len(y) else None  # overall accuracy
        out["by_threshold"][str(thr)] = {
            "alarm_count": int(tp + fp), "precision": precision, "recall_hit_rate": recall,
            "false_positive_rate": fpr, "overall_accuracy": hit_rate,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        }
    return out


def lead_time_days(dates, p, y_true_arr, market_dates, market_index, horizon, threshold, thr):
    """Avg trading days between first alarm (P≥thr) and the bear-window drawdown trough, per realized bear OOS date.

    For each OOS decision date that is a realized bear (label=1), find the trough position within its
    forward window [t+1..t+horizon]; lead = (trough_pos − first_alarm_pos) in trading days, where first_alarm_pos
    is the earliest OOS date ≤ t with P≥thr in the run-up. Returns mean lead (days) or None.
    """
    if len(dates) == 0:
        return None
    pos_of = {pd.Timestamp(d): i for i, d in enumerate(market_dates)}
    alarms = [(pd.Timestamp(d), pi) for d, pi in zip(dates, p) if pi >= thr]
    leads = []
    for di, (d, yv) in enumerate(zip(dates, y_true_arr)):
        if yv != 1.0:
            continue
        t_pos = pos_of.get(pd.Timestamp(d))
        if t_pos is None:
            continue
        win = market_index[t_pos + 1:t_pos + horizon + 1]
        if len(win) == 0:
            continue
        trough_pos = t_pos + 1 + int(np.argmin(win))
        prior_alarms = [pos_of[a] for a, _ in alarms if pos_of.get(a) is not None and pos_of[a] <= t_pos]
        if not prior_alarms:
            continue
        first_alarm_pos = min(prior_alarms)
        leads.append(trough_pos - first_alarm_pos)
    return float(np.mean(leads)) if leads else None


def trust_block(y, p, dates, p_seed):
    # 10-bin reliability + ECE
    bins = np.linspace(0.0, 1.0, ECE_BINS + 1)
    reliability = []; ece = 0.0; n = len(y)
    for b in range(ECE_BINS):
        lo, hi = bins[b], bins[b + 1]
        mask = (p >= lo) & (p < hi) if b < ECE_BINS - 1 else (p >= lo) & (p <= hi)
        cnt = int(np.sum(mask))
        if cnt == 0:
            reliability.append({"bin": [round(lo, 2), round(hi, 2)], "count": 0,
                                "pred_mean": None, "obs_freq": None})
            continue
        pm = float(np.mean(p[mask])); of = float(np.mean(y[mask]))
        reliability.append({"bin": [round(lo, 2), round(hi, 2)], "count": cnt,
                            "pred_mean": pm, "obs_freq": of})
        ece += (cnt / n) * abs(pm - of)
    # sub-period AUC stability (first vs second half of OOS by time)
    half = n // 2
    auc_first = _safe_auc(y[:half], p[:half]) if half >= 3 else None
    auc_second = _safe_auc(y[half:], p[half:]) if (n - half) >= 3 else None
    # multi-seed AUC mean/spread
    seed_aucs = None
    if p_seed:
        vals = [_safe_auc(y, p_seed[s]) for s in p_seed]
        vals = [v for v in vals if v is not None]
        if vals:
            seed_aucs = {"per_seed": {str(s): _safe_auc(y, p_seed[s]) for s in p_seed},
                         "mean": float(np.mean(vals)), "spread": float(max(vals) - min(vals)),
                         "min": float(min(vals)), "max": float(max(vals))}
    return {
        "ece": float(ece) if n else None,
        "reliability_10bin": reliability,
        "auc_first_half": auc_first, "auc_second_half": auc_second,
        "auc_subperiod_gap": (abs(auc_first - auc_second) if (auc_first is not None and auc_second is not None) else None),
        "multi_seed_lgbm_auc": seed_aucs,
    }


def _series_stats(daily_log_rets):
    """CAGR / ann-vol / Sharpe / MaxDD / Calmar from a daily LOG-return stream (cash days = 0.0)."""
    r = np.asarray(daily_log_rets, dtype=float)
    if len(r) == 0:
        return {"cagr": None, "ann_vol": None, "sharpe": None, "max_drawdown": None, "calmar": None}
    cum = np.exp(np.cumsum(r))
    years = len(r) / TRADING_DAYS_YEAR
    cagr = (cum[-1]) ** (1.0 / years) - 1.0 if years > 0 and cum[-1] > 0 else None
    ann_vol = float(np.std(r, ddof=1) * math.sqrt(TRADING_DAYS_YEAR)) if len(r) > 1 else None
    sharpe = (float(np.mean(r)) / np.std(r, ddof=1) * math.sqrt(TRADING_DAYS_YEAR)) \
        if len(r) > 1 and np.std(r, ddof=1) > 0 else None
    peak = np.maximum.accumulate(cum)
    mdd = float(np.max((peak - cum) / peak)) if len(cum) else None  # as positive fraction
    calmar = (cagr / mdd) if (cagr is not None and mdd and mdd > 0) else None
    return {"cagr": cagr, "ann_vol": ann_vol, "sharpe": sharpe, "max_drawdown": mdd, "calmar": calmar}


def economic_block(run, market_dates, market_ret, horizon):
    """Timing strategy (hold market when P(bear)<thr else cash@0, with switch cost) vs buy-and-hold.

    The signal at decision date t holds until the next monthly decision date (step-function exposure).
    Conservative: cash earns 0%. Switch cost SWITCH_COST charged (as a log drag) on each exposure change.
    Buy-and-hold = full equal-weight market exposure over the SAME OOS window.
    Per-episode: timed vs buy-hold cumulative return within each known bear window (drawdown reduction test).
    """
    dates = run["dates"]; p = run["p_lgbm"]
    if len(dates) == 0:
        return {"note": "no OOS decisions", "by_threshold": {}, "by_episode": {}}
    pos_of = {pd.Timestamp(d): i for i, d in enumerate(market_dates)}
    dec_pos = [pos_of[pd.Timestamp(d)] for d in dates]
    oos_start = dec_pos[0]
    oos_end = min(dec_pos[-1] + horizon, len(market_dates) - 1)  # extend to realize the last decision's window
    # daily market log returns over the OOS span (market_ret[i] is the return realized ON day i)
    span = list(range(oos_start + 1, oos_end + 1))
    bh_daily = [market_ret[i] for i in span]

    # buy-and-hold over identical span
    out = {"by_threshold": {}, "by_episode": {}}
    out["buy_and_hold"] = _series_stats(bh_daily)
    out["buy_and_hold"]["span_days"] = len(span)
    out["buy_and_hold"]["span"] = [str(market_dates[oos_start + 1]), str(market_dates[oos_end])]

    for thr in OPERATING_THRESHOLDS:
        # build a step-function exposure (1=in market, 0=cash) over `span`, set by the most recent decision ≤ day
        exposure = np.zeros(len(span), dtype=float)
        switches = 0; prev_exp = 0.0; di = 0
        # map each decision position to its in-market flag (P(bear) < thr → in market)
        dec_flags = [(dp, 1.0 if pv < thr else 0.0) for dp, pv in zip(dec_pos, p)]
        timed_daily = []
        for k, day in enumerate(span):
            # advance to the latest decision whose position < day (decision known at end-of-day t acts from t+1)
            while di + 1 < len(dec_flags) and dec_flags[di + 1][0] < day:
                di += 1
            cur_exp = dec_flags[di][1] if dec_flags[di][0] < day else 0.0
            if k == 0:
                prev_exp = cur_exp
            if cur_exp != prev_exp:
                switches += 1
            r = market_ret[day] if cur_exp == 1.0 else 0.0
            # apply switch cost as a log drag on the day exposure changes
            if cur_exp != prev_exp:
                r += math.log(1.0 - SWITCH_COST)
            timed_daily.append(r)
            exposure[k] = cur_exp
            prev_exp = cur_exp
        stats = _series_stats(timed_daily)
        stats["pct_time_in_market"] = float(np.mean(exposure))
        stats["n_switches"] = int(switches)
        # honest verdict vs buy-hold
        bh = out["buy_and_hold"]
        stats["reduces_maxdd_vs_buyhold"] = (
            bool(stats["max_drawdown"] is not None and bh["max_drawdown"] is not None
                 and stats["max_drawdown"] < bh["max_drawdown"]))
        stats["improves_calmar_vs_buyhold"] = (
            bool(stats["calmar"] is not None and bh["calmar"] is not None and stats["calmar"] > bh["calmar"]))
        stats["improves_sharpe_vs_buyhold"] = (
            bool(stats["sharpe"] is not None and bh["sharpe"] is not None and stats["sharpe"] > bh["sharpe"]))
        out["by_threshold"][str(thr)] = stats

    # per-episode (use the 0.5 threshold exposure as the representative timing line)
    rep_thr = 0.5
    dec_flags = [(dp, 1.0 if pv < rep_thr else 0.0) for dp, pv in zip(dec_pos, p)]
    for name, e_start, e_end in BEAR_EPISODES:
        e_idx = [i for i, d in enumerate(market_dates) if e_start <= d <= e_end]
        if len(e_idx) < 5:
            out["by_episode"][name] = {"in_oos": False, "note": "episode outside market-date range"}
            continue
        # only meaningful if the episode overlaps the OOS span
        ep_span = [i for i in e_idx if (oos_start + 1) <= i <= oos_end]
        in_oos = len(ep_span) >= 5
        bh_ep = float(np.exp(np.sum([market_ret[i] for i in e_idx])) - 1.0)  # buy-hold simple return in window
        rec = {"window": [str(e_start), str(e_end)], "in_oos": in_oos,
               "buy_hold_return": bh_ep, "buy_hold_maxdd": _series_stats([market_ret[i] for i in e_idx])["max_drawdown"]}
        if in_oos:
            di = 0; prev_exp = 0.0; timed_daily = []; exp_list = []
            for day in ep_span:
                while di + 1 < len(dec_flags) and dec_flags[di + 1][0] < day:
                    di += 1
                cur_exp = dec_flags[di][1] if dec_flags[di][0] < day else 0.0
                r = market_ret[day] if cur_exp == 1.0 else 0.0
                if exp_list and cur_exp != exp_list[-1]:
                    r += math.log(1.0 - SWITCH_COST)
                timed_daily.append(r); exp_list.append(cur_exp)
            rec["timed_return"] = float(np.exp(np.sum(timed_daily)) - 1.0)
            rec["timed_maxdd"] = _series_stats(timed_daily)["max_drawdown"]
            rec["pct_time_in_market"] = float(np.mean(exp_list)) if exp_list else None
            rec["reduced_drawdown"] = (bool(rec["timed_maxdd"] is not None and rec["buy_hold_maxdd"] is not None
                                            and rec["timed_maxdd"] < rec["buy_hold_maxdd"]))
        out["by_episode"][name] = rec
    return out


# ════════════════════════════════════════════════════════════════════════════
# Orchestration
# ════════════════════════════════════════════════════════════════════════════
def run_horizon(feat, fwd_cum_ret_cache, market_dates, market_ret, market_index, decision_positions,
                horizon, threshold, seed, n_estimators, smoke):
    fwd = fwd_cum_ret_cache[horizon]
    logger.info(f"  [H={horizon}] walk-forward (anti-leakage: train sample p_s + {horizon} ≤ p_t)...")
    wf = walk_forward(feat, fwd, market_dates, decision_positions, horizon, threshold,
                      seed, n_estimators, smoke, multi_seed=True)
    y = wf["y_true"]; pl = wf["p_lgbm"]; plr = wf["p_logit"]
    if len(y) == 0:
        logger.warning(f"  [H={horizon}] 0 OOS decisions (insufficient history); skipping")
        return None
    prec_lgbm = precision_block(y, pl)
    prec_logit = precision_block(y, plr)
    for thr in OPERATING_THRESHOLDS:
        lt = lead_time_days(wf["dates"], pl, y, market_dates, market_index, horizon, threshold, thr)
        prec_lgbm["by_threshold"][str(thr)]["lead_time_days_to_trough"] = lt
    trust = trust_block(y, pl, wf["dates"], wf["p_seed"])
    econ = economic_block(wf, market_dates, market_ret, horizon)
    auc_l = prec_lgbm["roc_auc"]; auc_r = prec_logit["roc_auc"]
    lgbm_beats_logit = (auc_l is not None and auc_r is not None and auc_l > auc_r)
    return {
        "horizon": horizon, "threshold": threshold,
        "n_oos_decisions": int(len(y)), "n_realized_bear": int(y.sum()),
        "oos_window": [str(wf["dates"][0]), str(wf["dates"][-1])],
        "precision": {"lightgbm": prec_lgbm, "logistic": prec_logit,
                      "lgbm_beats_logistic_auc": bool(lgbm_beats_logit)},
        "trust": trust,
        "economic": econ,
    }


def run(args):
    conn = get_db_conn()
    cur = conn.cursor()
    t0 = time.monotonic()
    logger.info("Building equal-weight market series (all numeric stocks, 1992-2026)...")
    market_dates, market_ret, market_index = build_market_series(cur)
    logger.info(f"Market series: {len(market_dates)} trading days "
                f"({market_dates[0]} → {market_dates[-1]}; load {time.monotonic()-t0:.1f}s)")

    if args.smoke:
        cutoff = date(2015, 1, 1)
        keep = np.array([d >= cutoff for d in market_dates])
        if keep.sum() > 60:
            market_dates = market_dates[keep]; market_ret = market_ret[keep]
            market_index = np.exp(np.cumsum(market_ret))
            logger.info(f"  [SMOKE] limited to ≥2015: {len(market_dates)} trading days")

    macro_frame = load_macro_daily_frame(cur, market_dates)
    logger.info(f"Macro frame: {macro_frame.shape[1]} series × {macro_frame.shape[0]} days "
                f"(monthly release-lag +1m on {sorted(MACRO_MONTHLY_LAGGED)})")
    feat = build_feature_frame(market_dates, market_ret, market_index, macro_frame)
    decision_positions = month_end_decision_positions(market_dates)
    logger.info(f"Features: {len(FEATURE_COLS)} cols; decision dates: {len(decision_positions)} month-ends")

    horizons = [args.horizon] if args.horizon else DEFAULT_HORIZONS
    fwd_cache = {h: make_bear_label(market_index, h) for h in horizons}
    n_est = 50 if args.smoke else 200

    results = {}
    for h in horizons:
        hr = run_horizon(feat, fwd_cache, market_dates, market_ret, market_index, decision_positions,
                         h, args.threshold, args.seed, n_est, args.smoke)
        if hr:
            results[f"H{h}"] = hr
            logger.info(f"  [H={h}] done: {hr['n_oos_decisions']} OOS decisions, "
                        f"{hr['n_realized_bear']} realized bear; "
                        f"lgbm AUC={hr['precision']['lightgbm']['roc_auc']} "
                        f"logit AUC={hr['precision']['logistic']['roc_auc']} "
                        f"(elapsed {time.monotonic()-t0:.0f}s)")
    conn.close()
    return results, market_dates, market_index


def _fmt(v, p="{:+.3f}"):
    return p.format(v) if isinstance(v, (int, float)) else "  n/a"


def print_summary(results):
    logger.info("=" * 110)
    logger.info("Market-Regime TIMING — OOS Summary (PRIMARY=LightGBM; baseline=Logistic)")
    logger.info("=" * 110)
    logger.info(f"  {'H':>4} {'N_oos':>6} {'Bear':>5} {'AUC_lgbm':>9} {'AUC_logit':>9} {'PR-AUC':>7} "
                f"{'Brier':>7} {'ECE':>6} {'AUCgap':>7} {'beats?':>7}")
    for k, r in results.items():
        pl = r["precision"]["lightgbm"]; pr = r["precision"]["logistic"]; tr = r["trust"]
        logger.info(f"  {r['horizon']:>4} {r['n_oos_decisions']:>6} {r['n_realized_bear']:>5} "
                    f"{_fmt(pl['roc_auc'],'{:.3f}'):>9} {_fmt(pr['roc_auc'],'{:.3f}'):>9} "
                    f"{_fmt(pl['pr_auc'],'{:.3f}'):>7} {_fmt(pl['brier'],'{:.3f}'):>7} "
                    f"{_fmt(tr['ece'],'{:.3f}'):>6} {_fmt(tr['auc_subperiod_gap'],'{:.3f}'):>7} "
                    f"{('YES' if r['precision']['lgbm_beats_logistic_auc'] else 'no'):>7}")
    logger.info("-" * 110)
    logger.info("  Economic — timing (hold mkt if P(bear)<thr else CASH, +switch cost) vs BUY-HOLD:")
    for k, r in results.items():
        bh = r["economic"].get("buy_and_hold", {})
        logger.info(f"   [H={r['horizon']}] buy&hold: Sharpe={_fmt(bh.get('sharpe'),'{:.2f}')} "
                    f"MaxDD={_fmt(bh.get('max_drawdown'),'{:.3f}')} Calmar={_fmt(bh.get('calmar'),'{:.2f}')} "
                    f"CAGR={_fmt(bh.get('cagr'),'{:.3f}')}")
        for thr, s in r["economic"].get("by_threshold", {}).items():
            flags = ("MDD↓" if s.get("reduces_maxdd_vs_buyhold") else "    ") + \
                    (" Calmar↑" if s.get("improves_calmar_vs_buyhold") else "       ") + \
                    (" Sharpe↑" if s.get("improves_sharpe_vs_buyhold") else "       ")
            logger.info(f"       thr={thr}: Sharpe={_fmt(s.get('sharpe'),'{:.2f}')} "
                        f"MaxDD={_fmt(s.get('max_drawdown'),'{:.3f}')} Calmar={_fmt(s.get('calmar'),'{:.2f}')} "
                        f"CAGR={_fmt(s.get('cagr'),'{:.3f}')} inMkt={_fmt(s.get('pct_time_in_market'),'{:.2f}')} "
                        f"sw={s.get('n_switches')} | {flags}")
    logger.info("-" * 110)
    logger.info("  Per-episode (representative thr=0.5; did timing reduce the drawdown?):")
    for k, r in results.items():
        for name, e in r["economic"].get("by_episode", {}).items():
            if not e.get("in_oos"):
                logger.info(f"   [H={r['horizon']}] {name:18}: (not in OOS span)")
                continue
            logger.info(f"   [H={r['horizon']}] {name:18}: buy&hold_ret={_fmt(e.get('buy_hold_return'),'{:+.2%}')} "
                        f"timed_ret={_fmt(e.get('timed_return'),'{:+.2%}')} "
                        f"bh_MaxDD={_fmt(e.get('buy_hold_maxdd'),'{:.3f}')} "
                        f"timed_MaxDD={_fmt(e.get('timed_maxdd'),'{:.3f}')} "
                        f"{'(reduced DD)' if e.get('reduced_drawdown') else '(did NOT reduce)'}")


def main():
    ap = argparse.ArgumentParser(description=f"Market-Regime / Bear-Probability TIMING Validation {TOOL_VER}")
    ap.add_argument("--output", type=str, default=None, help="write results JSON to repo-relative path")
    ap.add_argument("--horizon", type=int, default=None, help="single forward horizon in trading days (default: run 20/60/120)")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="bear label: forward cum-return < -threshold (default 0.08)")
    ap.add_argument("--seed", type=int, default=5422)
    ap.add_argument("--smoke", action="store_true", help="plumbing check: ≥2015 + n_estimators 50")
    ap.add_argument("--dry-run", action="store_true", help="compute+print only (default behavior; accepted for symmetry)")
    ap.add_argument("--commit", action="store_true", help="no DB write occurs; accepted for symmetry (use --output to persist JSON)")
    args = ap.parse_args()

    logger.info("=" * 110)
    logger.info(f"Market-Regime / Bear-Probability TIMING Validation {TOOL_VER}")
    logger.info(f"  Predicts P(bear) = WHETHER to be in market (NOT which stocks) — §3.2 evaluation, source-pure, anti-leakage")
    logger.info(f"  horizon={'all 20/60/120' if not args.horizon else args.horizon} threshold=-{args.threshold:.0%} "
                f"seed={args.seed} smoke={args.smoke}")
    logger.info(f"  ⚠️  CAVEAT: market timing is HARD / probabilistic / FRED macro is US-proxy-for-TW / few bear events → wide uncertainty")
    logger.info("=" * 110)

    t_global = time.monotonic()
    results, market_dates, market_index = run(args)

    if not results:
        logger.warning("No horizon produced OOS decisions (need more history; try without --smoke).")
    else:
        print_summary(results)
    logger.info(f"\n  Total elapsed: {time.monotonic()-t_global:.1f}s")

    if args.output:
        out = dict(results)
        out["_meta"] = {
            "tool": "market_regime_timing_validation.py", "tool_ver": TOOL_VER,
            "model_primary": "lightgbm.LGBMClassifier (n_est~200, num_leaves 15, lr 0.03, scale_pos_weight=n_neg/n_pos)",
            "model_baseline": "sklearn LogisticRegression (StandardScaler + L2, class_weight balanced)",
            "run_at": datetime.now().isoformat(), "constitution_ver": CONSTITUTION_VER,
            "seed": args.seed, "threshold_bear": args.threshold,
            "horizons": ([args.horizon] if args.horizon else DEFAULT_HORIZONS),
            "operating_thresholds": OPERATING_THRESHOLDS, "switch_cost": SWITCH_COST,
            "min_oos_year": MIN_OOS_YEAR, "min_train_months": MIN_TRAIN_MONTHS,
            "n_market_days": int(len(market_dates)),
            "market_span": [str(market_dates[0]), str(market_dates[-1])],
            "macro_series": MACRO_ALL, "macro_monthly_release_lagged": sorted(MACRO_MONTHLY_LAGGED),
            "feature_cols": FEATURE_COLS,
            "anti_leakage": ("RIGOROUS: (i) features computed as-of each date use ONLY data dated <= that date; "
                             "(ii) monthly FRED series lagged +1 month for publication delay; "
                             "(iii) training samples restricted to decision positions p_s with p_s + horizon <= p_t "
                             "(label window fully observed using only data <= decision date t). Expanding window."),
            "source_traceability": ("per CLAUDE.md §一.10 — all data from (b) DB query "
                                    "(TaiwanStockPriceAdj close + fred_series); features = real value -> math transform; "
                                    "no fabrication, no imputation (LightGBM handles NaN; Logistic uses fit-time train-median "
                                    "for its standardization only)."),
            "smoke": args.smoke,
            "HONEST_CAVEAT": ("Market timing is genuinely HARD. Outputs are PROBABILISTIC, not certain. FRED macro is "
                              "US data used as a PROXY for Taiwan (no equivalent high-frequency TW macro is published). "
                              "Bear episodes are FEW over 1992-2026, so estimates carry WIDE uncertainty. This is a "
                              "walk-forward RESEARCH validation, NOT a guarantee of future performance. The economic "
                              "backtest assumes cash earns 0% (conservative; no risk-free yield) and a small per-switch cost. "
                              "If LightGBM does not beat the Logistic baseline out-of-sample, that is reported honestly."),
        }
        op = (PROJECT_ROOT / args.output) if not os.path.isabs(args.output) else Path(args.output)
        op.parent.mkdir(parents=True, exist_ok=True)
        with open(op, "w") as f:
            json.dump(out, f, indent=2, default=str)
        logger.info(f"  Results persisted: {op}")


if __name__ == "__main__":
    main()
