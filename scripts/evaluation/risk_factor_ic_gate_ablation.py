"""
risk_factor_ic_gate_ablation.py v0.1 (系統/特異風險分解特徵 — 正交性 + IC 雙 Gate Ablation · §9.9 / §一.15 / §0.3-E)
================================================================================
**最後更新日期**: 2026-06-07
**主權狀態**: RESEARCH-MODE DUAL-GATE ABLATION(正交性 gate + IC gate)+ SOURCE-PURE(純價格)+ §一.10 SOURCE-TRACEABLE + §一.8 HONEST + 不動 DB / 不 commit / 不改 SPEC
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:實測兩個「真正交」候選特徵 ——
**市場 beta**(個股跟大盤一起漲跌的程度)與 **特異波動**(扣掉大盤後個股自己亂跳的程度)——
到底能不能在現有 37 特徵之外**多帶來新的選股預測力**,用兩道關卡(先正交、後 IC)嚴格篩。

**為什麼是這兩個(正交性 litmus)**:樹模型手上只有「總波動」,**沒有大盤報酬序列 → 算不出 beta、也分不出系統 vs 特異風險**。
所以這是 37 特徵推導不出來的「全新資訊」(對比之前被否決的 cycle_phase:它只是價格位置,樹早能從動能/均線推出 → 冗餘)。
學理:Ang et al. (2006) 低特異波動異常,跨市場(含亞洲)最穩健的橫斷面異常之一。

**它怎麼做(步驟)**:
1. 取 397 乾淨核心股 + canonical panels(§14.7-DE 單一源)。
2. 每 panel:大盤報酬 r_mkt = 當日 397 股**等權日報酬均值**(純價格,≤as_of);每股算
   `beta_market_252d` = Cov(r_i,r_mkt)/Var(r_mkt)(重用 `FeatureStoreBuilder._ols_slope`,SSOT);
   `idio_vol_60d` = r_i 對 r_mkt 局部回歸後**殘差的標準差**。
3. **Gate-1 正交性**:候選與 37 特徵的橫斷面 |corr| 須 **< 0.5**(否則判冗餘淘汰,不進 IC)。
4. **Gate-2 IC**:過正交者,walk-forward 比較 [37] vs [37+候選] 的橫斷面 IC,lift ≥ 0.01 才 KEEP(§一.15 iv)。
5. 4 種持有期各做;季(60d)最穩健,權重最高。

**source-pure(不可有 AI 幻像值)**:全部從 `TaiwanStockPriceAdj` 收盤價經數學轉換;無 hardcoded、無 FRED 依賴、
任一值未定義(歷史不足 / Var=0)→ None → 該股該 panel 從兩模型同時剔除,**絕不補值**(§一.13)。

**輸入**:DB(feature_values + TaiwanStockPriceAdj)。**輸出**:JSON(reports/)+ log。
**它不做的事**:不寫 feature_values、不改 FEATURE_DEFINITIONS / SPEC、不 commit、不 retrain(§3.1 evaluation)。
promote 與否、rebuild、retrain 為後續獨立授權步驟。

## 📜 一、核心定義說明 (Core Definitions / The Constitution)

1. **[Orthogonality Gate]** (v0.1, 用戶 2026-06-07 directive「重新設計成正交特徵」): 候選與 37 canonical 各特徵之橫斷面平均 |Pearson corr| < 0.5 才算正交;≥0.5 判冗餘淘汰(防再現 cycle_phase 之動能冗餘)。
2. **[IC-Gated Retention]** (v0.1, §一.15 iv / §0.3-E): 過正交 gate 者,walk-forward 橫斷面 IC 提升 ≥0.01(`drop_minus_full < −0.01`)才 KEEP。
3. **[Source-Pure Price-Only]** (v0.1, §一.10 / §一.13): beta/idio-vol 全由 adjusted close 經 OLS 數學轉換;市場因子 = 397 等權報酬均值;無 hardcoded / 無 impute(None→drop)。
4. **[SSOT OLS Reuse]** (v0.1, §0.0-I): beta = Cov/Var 重用 `FeatureStoreBuilder._ols_slope`;idio 殘差 std 為新運算(本檔為其 reference 實作,過 gate 後原樣移植 feature_store_builder)。
5. **[Anti-Leakage]** (v0.1, §8.5): 全 ≤ as_of;市場因子當日同期(contemporaneous regression,無未來洩漏);walk-forward train=過去 panels / test=當前。
6. **[Honest Statistical Disclosure]** (v0.1, §一.8): 報正交 corr + per-horizon IC lift + 配對 t;週(5d)IC≈0 視為雜訊明示。
7. **[Research-Mode Sovereignty]** (v0.1, §3.1): read-only;不寫 DB / 不 commit / 不改 SPEC。
8. **[Zero Hardcoded Verdict]** (v0.1, §5.6.3): KEEP/REVOKE 依雙 gate 結果動態判定。

## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix)

### A. 主要運行模式
| 子項 | CLI | 治權契約 |
| :--- | :--- | :--- |
| A.1 全 4-horizon 雙 gate | `python scripts/evaluation/risk_factor_ic_gate_ablation.py` | §一.15 iv |
| A.2 指定 horizon | `--horizons 60,252` | §14.7-CY |
| A.3 指定 seed | `--seed 5422` | §一.10 #3 |

### B. 補充運行模式
| 子項 | CLI | 治權契約 |
| :--- | :--- | :--- |
| B.1 快速 smoke(末 N panels;主驗正交性) | `--limit-panels 8` | 開發驗證 |

### 不提供之旗標 (Intentionally Omitted)
- `--commit`:research-mode,永不寫 DB / SPEC。

### 對齊憲章 §二 維運矩陣
| 場景 | 命令 |
| :--- | :--- |
| 風險因子正交+IC 裁決 | `python scripts/evaluation/risk_factor_ic_gate_ablation.py` |

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.1 | 2026-06-07 | Claude | **首版**:用戶 directive「重新設計成正交特徵再測」(否決動能冗餘之 cycle_phase 後)。F1 候選 = `beta_market_252d` + `idio_vol_60d`(系統/特異風險分解,Ang 2006 低特異波動異常);純價格 source-pure;市場因子=397 等權報酬均值;新增**正交性 gate**(|corr|<0.5)前置於 IC-gate。研究模式:不動 DB / 不 commit / 不改 SPEC。 | **ACTIVE** |
"""

from __future__ import annotations
import sys, argparse, math, json, logging, time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

import numpy as np
import lightgbm as lgb
from core.db_utils import get_db_conn, get_canonical_panel_dates
from core.feature_store_builder import FeatureStoreBuilder  # §0.0-I: 重用 _ols_slope(Cov/Var SSOT)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

CONSTITUTION_VER = "v6.1.0"
TOOL_VER = "v0.1"
SEED = 5422
FEATURE_SET_VERSION = "feature_set_v0.5"

LGB_PARAMS = {"learning_rate": 0.05, "max_depth": 5, "num_leaves": 20, "min_child_samples": 30,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
              "reg_alpha": 0.1, "reg_lambda": 0.1, "objective": "regression", "metric": "rmse",
              "verbose": -1, "seed": SEED}
N_ESTIMATORS = 200

SPEC_37 = [
    "log_return_20d", "log_return_60d", "log_return_252d", "ma_ratio_20", "ma_ratio_60", "max_drawdown_252d",
    "upside_volatility_60d", "downside_volatility_60d", "convexity_60d", "volatility_60d", "volatility_252d",
    "upside_capture_60d", "downside_capture_60d",
    "avg_daily_value_log_60d", "avg_daily_value_log_252d", "zero_volume_ratio_252d", "turnover_mean_60d",
    "pe_ratio", "pb_ratio", "dividend_yield",
    "roe_ttm", "operating_margin_ttm", "eps_sum_4q", "net_income_positive_ratio_8q",
    "revenue_yoy_3m_log", "asset_growth_yoy", "revenue_yoy_3m", "revenue_yoy_12m",
    "preferential_attachment_60d",
    "right_tail_returns_skew_252d",
    "liquidity_rank_pct_sector_60d", "size_log_zscore_sector",
    "foreign_net_20d", "foreign_net_60d", "trust_net_20d", "trust_net_60d", "margin_ratio_60d",
]

# F1 候選:系統/特異風險分解
CANDIDATE_FEATURES = ["beta_market_252d", "idio_vol_60d"]
BETA_WIN = 252
IDIO_WIN = 60
MIN_OBS_BETA = 60
MIN_OBS_IDIO = 40

IC_GATE_LIFT = 0.01      # §一.15 iv
ORTHO_CORR_MAX = 0.50    # 正交性 gate:與 37 之 max |corr| 須 < 此值


def load_universe(cur):
    cur.execute(
        "SELECT m.stock_id FROM core_universe_membership m JOIN core_universe_snapshot s ON s.snapshot_id=m.snapshot_id "
        "WHERE s.status='committed' AND m.core_tier='core_universe' "
        "AND s.snapshot_id=(SELECT snapshot_id FROM core_universe_snapshot WHERE status='committed' ORDER BY created_at DESC LIMIT 1)"
    )
    return list({r[0] for r in cur.fetchall()})


def load_features_37(cur, fs_id, universe):
    cur.execute(
        "SELECT stock_id, feature_name, feature_value::numeric FROM feature_values "
        "WHERE feature_set_id=%s AND stock_id=ANY(%s) AND is_null_imputed IS NOT TRUE",
        (fs_id, list(universe)),
    )
    feat = defaultdict(dict)
    for sid, fname, val in cur.fetchall():
        if val is not None and fname in SPEC_37:
            feat[sid][fname] = float(val)
    return feat


def load_close_series(cur, as_of, universe, lookback_days=450):
    start = as_of - timedelta(days=lookback_days)
    cur.execute(
        "SELECT stock_id, date, close::numeric FROM \"TaiwanStockPriceAdj\" "
        "WHERE stock_id=ANY(%s) AND date<=%s AND date>=%s AND close>0 ORDER BY stock_id, date",
        (list(universe), str(as_of), str(start)),
    )
    out = defaultdict(list)
    for sid, d, c in cur.fetchall():
        out[sid].append((d, float(c)))
    return out


def compute_market_and_stock_returns(close_series):
    """大盤因子 r_mkt[date] = 當日全股等權日 log 報酬均值;回傳 (stock_rets {sid:[(date,ret)]}, mkt {date:ret})。"""
    ret_by_date = defaultdict(list)
    stock_rets = {}
    for sid, pairs in close_series.items():
        rets = []
        for i in range(1, len(pairs)):
            (_, c0), (d1, c1) = pairs[i - 1], pairs[i]
            if c0 > 0 and c1 > 0:
                r = math.log(c1 / c0)
                rets.append((d1, r))
                ret_by_date[d1].append(r)
        stock_rets[sid] = rets
    mkt = {d: (sum(v) / len(v)) for d, v in ret_by_date.items() if v}
    return stock_rets, mkt


def compute_f1_for_stock(stock_ret_pairs, mkt):
    """beta_market_252d = Cov/Var(重用 _ols_slope);idio_vol_60d = 局部回歸殘差 std。None→drop 不補。"""
    aligned = [(r, mkt[d]) for d, r in stock_ret_pairs if d in mkt]
    res = {"beta_market_252d": None, "idio_vol_60d": None}
    if len(aligned) < MIN_OBS_BETA:
        return res
    win_b = aligned[-BETA_WIN:]
    ys_b = [a[0] for a in win_b]; xs_b = [a[1] for a in win_b]
    res["beta_market_252d"] = FeatureStoreBuilder._ols_slope(xs_b, ys_b, min_obs=MIN_OBS_BETA)  # Cov/Var SSOT
    win_i = aligned[-IDIO_WIN:]
    if len(win_i) >= MIN_OBS_IDIO:
        yi = [a[0] for a in win_i]; xi = [a[1] for a in win_i]
        b = FeatureStoreBuilder._ols_slope(xi, yi, min_obs=MIN_OBS_IDIO)
        if b is not None:
            mx = sum(xi) / len(xi); my = sum(yi) / len(yi)
            a_int = my - b * mx
            resid = [yi[k] - (a_int + b * xi[k]) for k in range(len(yi))]
            if len(resid) > 1:
                res["idio_vol_60d"] = float(np.std(resid, ddof=1))
    return res


def load_forward_returns(cur, as_of, horizon_days):
    cur.execute(
        "SELECT MIN(date) FROM \"TaiwanStockPriceAdj\" WHERE date >= (%s::date + INTERVAL '%s days') "
        "AND stock_id ~ '^[0-9]' AND date <= (%s::date + INTERVAL '%s days')",
        (str(as_of), horizon_days, str(as_of), horizon_days + 14),
    )
    r = cur.fetchone()
    label_date = r[0] if r and r[0] else None
    if not label_date:
        return {}, None
    cur.execute(
        "WITH t0 AS (SELECT stock_id, close FROM \"TaiwanStockPriceAdj\" WHERE date=%s AND close>0), "
        "t1 AS (SELECT stock_id, close FROM \"TaiwanStockPriceAdj\" WHERE date=%s AND close>0) "
        "SELECT t0.stock_id, LN(t1.close::numeric/t0.close::numeric) FROM t0 JOIN t1 ON t0.stock_id=t1.stock_id",
        (str(as_of), str(label_date)),
    )
    return {sid: float(r) for sid, r in cur.fetchall()}, label_date


def spearman_ic(pred, y):
    pred = np.array(pred); y = np.array(y)
    rp = pred.argsort().argsort().astype(float); ry = y.argsort().argsort().astype(float)
    if np.std(rp) < 1e-10 or np.std(ry) < 1e-10:
        return 0.0
    return float(np.corrcoef(rp, ry)[0, 1])


def winsorize(arr, lo_q=0.01, hi_q=0.99):
    return np.clip(arr, np.quantile(arr, lo_q), np.quantile(arr, hi_q))


def train_lgb(X, y, feat_names):
    data = lgb.Dataset(np.array(X), label=y, feature_name=feat_names)
    return lgb.train(LGB_PARAMS, data, num_boost_round=N_ESTIMATORS)


def build_feature_cache(cur, panels, universe):
    cache = {}
    drop_log = []
    t0 = time.monotonic()
    for idx, (fs_id, as_of) in enumerate(panels):
        feat37 = load_features_37(cur, fs_id, universe)
        if not feat37:
            continue
        close_series = load_close_series(cur, as_of, universe)
        stock_rets, mkt = compute_market_and_stock_returns(close_series)
        sids, X37, Xcand = [], [], []
        n_drop = 0
        for sid in universe:
            row37 = feat37.get(sid)
            if not row37 or any(f not in row37 for f in SPEC_37):
                n_drop += 1
                continue
            cand = compute_f1_for_stock(stock_rets.get(sid, []), mkt)
            if any(cand[f] is None for f in CANDIDATE_FEATURES):  # no-impute(§一.13)
                n_drop += 1
                continue
            base = [row37[f] for f in SPEC_37]
            sids.append(sid)
            X37.append(base)
            Xcand.append([cand[f] for f in CANDIDATE_FEATURES])
        if sids:
            cache[as_of] = (sids, np.array(X37), np.array(Xcand))
            drop_log.append((as_of, len(sids), n_drop))
        if (idx + 1) % 20 == 0:
            logger.info(f"  feature cache: {idx+1}/{len(panels)} panels ({time.monotonic()-t0:.0f}s)")
    return cache, drop_log


def orthogonality_gate(feat_cache):
    """Gate-1:候選 vs 37 之橫斷面 |Pearson corr|,跨 panel 平均,取每候選對 37 之 max。"""
    logger.info(f"\n{'='*100}\nGate-1 正交性(候選 vs 37 canonical,|corr| 須 < {ORTHO_CORR_MAX})\n{'='*100}")
    per_cand = {c: defaultdict(list) for c in CANDIDATE_FEATURES}
    for as_of, (sids, X37, Xcand) in feat_cache.items():
        if len(sids) < 30:
            continue
        for ci, cname in enumerate(CANDIDATE_FEATURES):
            cv = Xcand[:, ci]
            if np.std(cv) < 1e-12:
                continue
            for fi, fname in enumerate(SPEC_37):
                fv = X37[:, fi]
                if np.std(fv) < 1e-12:
                    continue
                r = float(np.corrcoef(cv, fv)[0, 1])
                if not math.isnan(r):
                    per_cand[cname][fname].append(abs(r))
    ortho_result = {}
    for cname in CANDIDATE_FEATURES:
        avg_abs = {f: float(np.mean(v)) for f, v in per_cand[cname].items() if v}
        if not avg_abs:
            ortho_result[cname] = {"max_abs_corr": None, "top_corr_feature": None, "pass": False, "top5": []}
            continue
        top = sorted(avg_abs.items(), key=lambda kv: kv[1], reverse=True)
        max_corr = top[0][1]
        passed = max_corr < ORTHO_CORR_MAX
        ortho_result[cname] = {
            "max_abs_corr": max_corr, "top_corr_feature": top[0][0],
            "pass": bool(passed), "top5": [(f, round(c, 3)) for f, c in top[:5]],
        }
        verdict = "✅ 正交" if passed else "❌ 冗餘"
        logger.info(f"  {cname:22} max|corr|={max_corr:.3f}(vs {top[0][0]}) → {verdict}")
        logger.info(f"      top5 |corr|: {ortho_result[cname]['top5']}")
    return ortho_result


def evaluate_horizon(cur, panels, feat_cache, horizon_days, label, cand_indices):
    """Gate-2 IC:[37] vs [37 + 過正交 gate 之候選] walk-forward IC。cand_indices = 要納入的候選欄 index。"""
    logger.info(f"\n{'='*100}\nHorizon: {label}({horizon_days}d) | 納入候選 idx={cand_indices}\n{'='*100}")
    panel_data = {}
    for fs_id, as_of in panels:
        if as_of not in feat_cache:
            continue
        sids, X37, Xcand = feat_cache[as_of]
        returns, _ = load_forward_returns(cur, as_of, horizon_days)
        if not returns:
            continue
        idxs = [i for i, sid in enumerate(sids) if sid in returns]
        if not idxs:
            continue
        y = np.array([returns[sids[i]] for i in idxs])
        x37 = X37[idxs, :]
        xc = Xcand[idxs, :][:, cand_indices] if cand_indices else None
        xfull = np.hstack([x37, xc]) if cand_indices else x37
        panel_data[as_of] = (x37, xfull, y)

    keys = sorted(panel_data.keys())
    ic37_list, icfull_list = [], []
    cand_names = [CANDIDATE_FEATURES[i] for i in cand_indices]
    for i in range(1, len(keys)):
        te37, tefull, te_y = panel_data[keys[i]]
        tr37, trfull, tr_y = [], [], []
        for j in range(i):
            a37, afull, ay = panel_data[keys[j]]
            tr37.append(a37); trfull.append(afull); tr_y.append(ay)
        X37tr = np.vstack(tr37); Xfulltr = np.vstack(trfull); ytr = np.concatenate(tr_y)
        if len(X37tr) < 100:
            continue
        y_tr = winsorize(ytr)
        m37 = train_lgb(X37tr, y_tr, SPEC_37)
        mfull = train_lgb(Xfulltr, y_tr, SPEC_37 + cand_names)
        ic37_list.append(spearman_ic(m37.predict(te37), te_y))
        icfull_list.append(spearman_ic(mfull.predict(tefull), te_y))

    if not ic37_list:
        return None
    ic37 = np.array(ic37_list); icf = np.array(icfull_list)
    diff = icf - ic37
    n = len(diff)
    sd = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    t_stat = float(np.mean(diff) / (sd / math.sqrt(n))) if sd > 1e-12 and n > 1 else 0.0
    mean37 = float(np.mean(ic37)); meanf = float(np.mean(icf))
    lift = meanf - mean37
    keep = (mean37 - meanf) < -IC_GATE_LIFT
    sig = "顯著" if abs(t_stat) > 1.997 else "不顯著"
    logger.info(f"  IC 37={mean37:+.4f} | 37+cand={meanf:+.4f} | lift={lift:+.4f} | 配對t={t_stat:+.2f}({sig}) | gate={'✅ KEEP' if keep else '❌ REVOKE'}")
    return {"horizon": label, "horizon_days": horizon_days, "n_panels": n,
            "mean_ic_37": mean37, "mean_ic_37plus_cand": meanf, "lift": lift,
            "drop_minus_full": mean37 - meanf, "paired_diff_t_stat": t_stat,
            "gate_keep": bool(keep), "candidates_included": cand_names}


def main():
    parser = argparse.ArgumentParser(description=f"Risk-Factor IC-Gate Ablation {TOOL_VER}")
    parser.add_argument("--horizons", type=str, default="5,20,60,252")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--limit-panels", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    LGB_PARAMS["seed"] = args.seed; globals()["SEED"] = args.seed

    horizon_days_list = [int(d.strip()) for d in args.horizons.split(",")]
    labels = []
    for d in horizon_days_list:
        if d <= 7: labels.append(("weekly", d))
        elif d <= 30: labels.append(("monthly", d))
        elif d <= 90: labels.append(("quarterly", d))
        else: labels.append(("annual", d))

    logger.info("=" * 100)
    logger.info(f"系統/特異風險分解特徵 — 正交性 + IC 雙 Gate Ablation {TOOL_VER}(§9.9 / §一.15 iv / §0.3-E)")
    logger.info("=" * 100)
    logger.info(f"  候選: {CANDIDATE_FEATURES} | baseline: {len(SPEC_37)} | probe: lgbm | seed: {args.seed}")
    logger.info(f"  正交 gate: |corr| < {ORTHO_CORR_MAX} | IC gate: lift ≥ {IC_GATE_LIFT}")

    conn = get_db_conn()
    try:
        cur = conn.cursor()
        universe = load_universe(cur)
        logger.info(f"  Universe: {len(universe)} stocks")
        panels = get_canonical_panel_dates(FEATURE_SET_VERSION)
        if args.limit_panels and args.limit_panels < len(panels):
            panels = panels[-args.limit_panels:]
        logger.info(f"  Panels: {len(panels)} ({panels[0][1]} ~ {panels[-1][1]}, §14.7-DE)")

        logger.info(f"\n建 feature cache(37 from DB + F1 risk factors live)...")
        feat_cache, drop_log = build_feature_cache(cur, panels, universe)
        if drop_log:
            avg_sids = np.mean([d[1] for d in drop_log])
            avg_drop = np.mean([d[2] for d in drop_log])
            logger.info(f"  cache: {len(feat_cache)} panels | 平均 common-support {avg_sids:.0f} 股 | 平均剔除 {avg_drop:.1f} 股/panel(§一.8)")

        # Gate-1 正交性
        ortho = orthogonality_gate(feat_cache)
        ortho_pass_idx = [i for i, c in enumerate(CANDIDATE_FEATURES) if ortho.get(c, {}).get("pass")]
        ortho_pass_names = [CANDIDATE_FEATURES[i] for i in ortho_pass_idx]
        logger.info(f"\n  Gate-1 結果:通過正交 = {ortho_pass_names or '(無)'}")

        results = {"_ortho": ortho}
        if not ortho_pass_idx:
            logger.info("  ⚠️ 無候選通過正交 gate → 不進 IC-gate(判冗餘,維持 37)")
        else:
            logger.info(f"\n{'='*100}\nGate-2 IC(僅納入通過正交者: {ortho_pass_names})\n{'='*100}")
            t_global = time.monotonic()
            for label, days in labels:
                r = evaluate_horizon(cur, panels, feat_cache, days, label, ortho_pass_idx)
                if r:
                    results[label] = r
            logger.info(f"\n  Total IC-gate elapsed: {time.monotonic()-t_global:.1f}s")
            # 綜合
            logger.info(f"\n{'='*100}\n雙 Gate 綜合裁決\n{'='*100}")
            for label, _ in labels:
                r = results.get(label)
                if r:
                    v = "KEEP" if r["gate_keep"] else "REVOKE"
                    sig = "✅" if abs(r["paired_diff_t_stat"]) > 1.997 else " "
                    logger.info(f"  {label:10} IC37={r['mean_ic_37']:+.4f} 37+cand={r['mean_ic_37plus_cand']:+.4f} lift={r['lift']:+.4f} t={r['paired_diff_t_stat']:+.2f}{sig} {v}")
            robust_keep = any(results.get(h, {}).get("gate_keep") for h in ("quarterly", "annual"))
            logger.info(f"\n  穩健 horizon(季/年)KEEP: {'是 → 候選 promote(須後續授權)' if robust_keep else '否 → 維持 37'}")
            results["_robust_keep"] = bool(robust_keep)

        out_path = args.output or f"reports/risk_factor_ic_gate_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out_full = Path(_base_dir).parent / out_path
        out_full.parent.mkdir(parents=True, exist_ok=True)
        results["_meta"] = {
            "tool": "risk_factor_ic_gate_ablation.py", "tool_ver": TOOL_VER,
            "candidate_features": CANDIDATE_FEATURES, "baseline_n_features": len(SPEC_37),
            "ortho_corr_max": ORTHO_CORR_MAX, "ic_gate_lift_threshold": IC_GATE_LIFT,
            "ortho_pass": ortho_pass_names, "probe_model": "lgbm",
            "run_at": datetime.now().isoformat(), "constitution_ver": CONSTITUTION_VER,
            "seed": SEED, "horizons": horizon_days_list, "n_universe": len(universe),
            "n_panels_input": len(panels),
            "source_traceability": "per §一.10 — (b) DB query + price-only OLS (reuse FeatureStoreBuilder._ols_slope)",
            "governance": "research-mode; no DB write; no commit; no SPEC change (§3.1)",
        }
        with open(out_full, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\n  Results persisted: {out_full}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
