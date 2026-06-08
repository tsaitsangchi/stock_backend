"""
kwave_ic_gate_ablation.py v0.1 (K-wave 多尺度循環特徵 IC-Gate Ablation · §0.3-A / §一.15 / §14.7-DC T_DC-30 / §0.3-E)
================================================================================
**最後更新日期**: 2026-06-07
**主權狀態**: RESEARCH-MODE IC-GATE ABLATION + SOURCE-PURE FAITHFUL-REUSE(§0.0-I 單一源)+ §一.10 SOURCE-TRACEABLE + §一.15 (iv) IC-GATED RETENTION + §一.8 HONEST + 不動 DB / 不 commit / 不改 production SPEC
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:實測「§0.3-A K-wave 7 個循環特徵(股價循環位置 + 景氣敏感度 beta)到底有沒有用」——
把它們加進現有 37 個特徵,看選股的預測力(IC)有沒有真的提升,**用數據決定該不該收回模型**,而不是憑感覺。

**背景(為什麼需要它)**:這 7 個特徵 2026-05-31 寫好過(v0.7),2026-06-01 又被治權 directive「死重應排除」移除(v0.8)——
因為**從沒實跑過 IC-gate**(§一.15 第 iv 條要求:特徵要證明能提升 IC ≥0.01 才保留)。本程式補跑這個從缺的 gate。

**它怎麼做(步驟)**:
1. 取 397 乾淨核心股 + canonical panels(§14.7-DE 單一源,非寫死)。
2. 每個 panel:從 DB 載入 37 個 canonical 特徵;**即時**從 `TaiwanStockPriceAdj`+`fred_series` 算出 7 個 K-wave 特徵
   (忠實重用 `feature_store_builder.py` 的 `_cycle_phase` / `_compute_macro_beta_features`,值與真正 build 出來的一致)。
3. **walk-forward**(防洩漏):每個時間點只用之前資料訓練,比較兩個模型 ——
   (A) 只用 37 特徵;(B) 用 37+7 特徵 —— 對**同一批 test 股票**算橫斷面 IC。
4. **IC-gate 判定**(§一.15 iv / §0.3-E):若 (B) 的 IC 比 (A) 高 ≥0.01(`drop_minus_full < −0.01`)→ 這 7 特徵值得保留;
   否則確認 2026-06-01 移除正確。並附 per-panel IC 差的配對 t 檢定(看提升是否只是雜訊)。
5. 4 種持有期(週/月/季/年)各做一遍;季(60d)為最穩健 horizon,權重最高。

**source-pure 保證(不可有 AI 幻像值)**:cycle_phase = (close−min)/(max−min) 純股價;macro_beta = 個股報酬對 FRED 因子
首階差分的 rolling-OLS 斜率(as-of ≤ 設定日,無 forward-fill 補造);無 hardcoded dict、無 hardcoded 40/60 年週期;
任一 K-wave 值未定義(None)→ 該股在該 panel 從兩個模型**同時剔除**(common support),**不補值**。

**輸入**:DB(feature_values + TaiwanStockPriceAdj + fred_series)。**輸出**:JSON 報告(reports/)+ log。
**它不做的事**:不寫 feature_values、不改 FEATURE_DEFINITIONS / SPEC、不 commit、不 retrain production
(§3.1 evaluation 角色、純讀取 + 暫態運算)。promote 與否、rebuild、retrain 為後續獨立授權步驟。

## 📜 一、核心定義說明 (Core Definitions / The Constitution)

1. **[IC-Gated Retention]** (v0.1, §一.15 iv / §0.3-E): 7 K-wave 特徵保留條件 = 加入後 walk-forward 橫斷面 IC 提升 ≥0.01(`drop_minus_full = IC_37 − IC_37+7 < −0.01`);否則撤銷。
2. **[Source-Pure Faithful Reuse]** (v0.1, §0.0-I / §一.10 / §一.13): K-wave 值由 import `FeatureStoreBuilder._cycle_phase` / `_compute_macro_beta_features` / `_load_macro_factor_series` 計算 → 與真正 build bit-faithful;無 hardcoded dict / 無 hardcoded 年限 / 無 imputed。
3. **[Common-Support No-Impute]** (v0.1, §一.13): 任一 K-wave 值 None → 該股該 panel 從 (A)(B) 同時剔除;絕不補值;每 panel 剔除數須 log(§一.8 no silent caps)。
4. **[Anti-Leakage Walk-Forward]** (v0.1, §8.5): 每 panel 只用 ≤ as_of 資料;train pool = 過去 panels;test = 當前 panel;K-wave 內建 ≤ as_of gate。
5. **[Same-Baseline Comparison]** (v0.1, §14.7-DG / §一.16): 397 核心 × canonical panels × seed × winsorize × 同模型 recipe;(A)(B) 唯一差異 = 7 K-wave 欄位。
6. **[Honest Statistical Disclosure]** (v0.1, §一.8 / §一.10): 報 per-horizon IC 差 + 配對 t 檢定;週(5d)IC≈0 之 ablation 視為雜訊明示;不誇大。
7. **[Research-Mode Sovereignty]** (v0.1, §3.1 evaluation): read-only;不寫 DB;不 commit;不改 production SPEC;唯一職責 = IC-gate 裁決 + JSON 持久化。
8. **[Zero Hardcoded Verdict]** (v0.1, §5.6.3): KEEP/REVOKE 依 ablation 結果動態判定。

## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix)

### A. 主要運行模式
| 子項 | CLI | 治權契約 |
| :--- | :--- | :--- |
| A.1 全 4-horizon IC-gate | `python scripts/evaluation/kwave_ic_gate_ablation.py` | §一.15 iv |
| A.2 指定 horizon | `--horizons 60,252` | §14.7-CY |
| A.3 指定 seed | `--seed 5422` | §一.10 #3 |
| A.4 輸出路徑 | `--output reports/kwave_ic_gate_<ts>.json` | §一.10 |

### B. 補充運行模式
| 子項 | CLI | 治權契約 |
| :--- | :--- | :--- |
| B.1 快速 smoke(限末 N panels) | `--limit-panels 20` | 開發驗證 |

### 不提供之旗標 (Intentionally Omitted)
- `--commit`:本程式為 research-mode,**永不寫 DB / SPEC**(promote 為獨立授權步驟)。

### 對齊憲章 §二 維運矩陣
| 場景 | 命令 |
| :--- | :--- |
| K-wave IC-gate 裁決 | `python scripts/evaluation/kwave_ic_gate_ablation.py` |

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.1 | 2026-06-07 | Claude | **首版**:補跑 §一.15 (iv) 從缺之 K-wave IC-gate ablation。7 特徵(cycle_phase_{5,20,60,252}d + macro_beta_{t10y2y,unrate,ipg3344s})2026-06-01 因「死重應排除 + IC-gate 從未跑」被移除;本程式忠實重用 `feature_store_builder.py` source-pure 函式即時計算 K-wave,walk-forward 比較 37 vs 37+7 之橫斷面 IC + 配對 t 檢定,依 §一.15(iv) gate 裁決 KEEP/REVOKE。研究模式:不動 DB / 不 commit / 不改 SPEC。 | **ACTIVE** |
"""

from __future__ import annotations
import sys, argparse, math, json, logging, time
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

import numpy as np
import lightgbm as lgb
from core.db_utils import get_db_conn, get_canonical_panel_dates
from core.feature_store_builder import FeatureStoreBuilder  # §0.0-I 單一源:忠實重用 K-wave source-pure 函式

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

CONSTITUTION_VER = "v6.1.0"
TOOL_VER = "v0.1"
SEED = 5422
FEATURE_SET_VERSION = "feature_set_v0.6"

# probe model = lgbm(快速且具代表性之 ablation probe;與 multi_cycle_ensemble_validation 同 recipe)
LGB_PARAMS = {"learning_rate": 0.05, "max_depth": 5, "num_leaves": 20, "min_child_samples": 30,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
              "reg_alpha": 0.1, "reg_lambda": 0.1, "objective": "regression", "metric": "rmse",
              "verbose": -1, "seed": SEED}
N_ESTIMATORS = 200

# 37 canonical source-pure features(= multi_cycle_*_validation SPEC_43,amihud 移除後實 37)
SPEC_37 = [
    "log_return_20d", "log_return_60d", "log_return_252d", "ma_ratio_20", "ma_ratio_60", "max_drawdown_252d",
    "upside_volatility_60d", "downside_volatility_60d", "convexity_60d", "volatility_60d", "volatility_252d",
    "upside_capture_60d", "downside_capture_60d",
    "avg_daily_value_log_60d", "avg_daily_value_log_252d", "turnover_mean_60d",
    "pe_ratio", "pb_ratio", "dividend_yield",
    "roe_ttm", "operating_margin_ttm", "eps_sum_4q", "revenue_yoy_3m_log", "asset_growth_yoy", "revenue_yoy_3m", "revenue_yoy_12m",
    "right_tail_returns_skew_252d",
    "liquidity_rank_pct_sector_60d", "size_log_zscore_sector",
    "foreign_net_20d", "foreign_net_60d", "margin_ratio_60d",
]

# §0.3-A / §一.15 候選 7 特徵(cycle_phase 4 尺度 + macro_beta 3 因子)
CYCLE_PHASE_WINDOWS = [("cycle_phase_5d", 5), ("cycle_phase_20d", 20), ("cycle_phase_60d", 60), ("cycle_phase_252d", 252)]
MACRO_BETA_KEYS = ["macro_beta_t10y2y", "macro_beta_unrate", "macro_beta_ipg3344s"]
KWAVE_FEATURES = [k for k, _ in CYCLE_PHASE_WINDOWS] + MACRO_BETA_KEYS

IC_GATE_LIFT = 0.01  # §一.15 iv / §0.3-E:IC 須提升 ≥ 此值才 KEEP


def load_universe(cur):
    cur.execute(
        "SELECT m.stock_id FROM core_universe_membership m JOIN core_universe_snapshot s ON s.snapshot_id=m.snapshot_id "
        "WHERE s.status='committed' AND m.core_tier='core_universe' "
        "AND s.snapshot_id=(SELECT snapshot_id FROM core_universe_snapshot WHERE status='committed' ORDER BY created_at DESC LIMIT 1)"
    )
    return list({r[0] for r in cur.fetchall()})


def load_features_37(cur, fs_id, universe):
    """37 canonical features ← feature_values(is_null_imputed IS NOT TRUE,同 validators)。回傳 {sid: {fname: val}}。"""
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


def load_close_series(cur, as_of, universe, lookback_days=400):
    """每股 ≤ as_of 之 (date, close) 序列(asc);one query。cycle_phase_252d 需 ≥252、macro_beta 需 ≥61 closes。"""
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


def compute_kwave_for_stock(close_pairs, fsb, factor_series):
    """忠實重用 feature_store_builder source-pure 函式。回傳 {fname: val 或 None}。"""
    closes = [c for _, c in close_pairs]
    kw = {}
    for fname, n in CYCLE_PHASE_WINDOWS:
        kw[fname] = FeatureStoreBuilder._cycle_phase(closes, n)  # static;(close−min)/(max−min) or None
    mb = fsb._compute_macro_beta_features(close_pairs, factor_series)  # {macro_beta_*: slope or None}
    for k in MACRO_BETA_KEYS:
        kw[k] = mb.get(k)
    return kw


def load_forward_returns(cur, as_of, horizon_days):
    """forward log return(同 multi_cycle_ensemble_validation)。回傳 {sid: ret}, label_date。"""
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
    """每 panel 計算 common-support 之 (sids, X37, X44)。X44 = X37 + 7 K-wave(同 row 順序)。"""
    cache = {}
    drop_log = []
    t0 = time.monotonic()
    for idx, (fs_id, as_of) in enumerate(panels):
        feat37 = load_features_37(cur, fs_id, universe)
        if not feat37:
            continue
        close_series = load_close_series(cur, as_of, universe)
        fsb = FeatureStoreBuilder(as_of, FEATURE_SET_VERSION)  # 輕量;僅為 as_of context
        factor_series = fsb._load_macro_factor_series(cur)     # FRED ≤ as_of(§8.5)
        sids, X37, X44 = [], [], []
        n_drop_feat37, n_drop_kwave = 0, 0
        for sid in universe:
            row37 = feat37.get(sid)
            if not row37 or any(f not in row37 for f in SPEC_37):
                n_drop_feat37 += 1
                continue
            pairs = close_series.get(sid, [])
            if len(pairs) < 1:
                n_drop_kwave += 1
                continue
            kw = compute_kwave_for_stock(pairs, fsb, factor_series)
            if any(kw[f] is None for f in KWAVE_FEATURES):  # common-support no-impute(§一.13)
                n_drop_kwave += 1
                continue
            base = [row37[f] for f in SPEC_37]
            sids.append(sid)
            X37.append(base)
            X44.append(base + [kw[f] for f in KWAVE_FEATURES])
        if sids:
            cache[as_of] = (sids, X37, X44)
            drop_log.append((as_of, len(sids), n_drop_feat37, n_drop_kwave))
        if (idx + 1) % 20 == 0:
            logger.info(f"  feature cache: {idx+1}/{len(panels)} panels ({time.monotonic()-t0:.0f}s)")
    return cache, drop_log


def evaluate_horizon(cur, panels, feat_cache, horizon_days, label):
    logger.info(f"\n{'='*100}\nHorizon: {label}({horizon_days}d)\n{'='*100}")
    # attach forward returns per panel
    panel_data = {}
    for fs_id, as_of in panels:
        if as_of not in feat_cache:
            continue
        sids, X37, X44 = feat_cache[as_of]
        returns, _ = load_forward_returns(cur, as_of, horizon_days)
        if not returns:
            continue
        m37, m44, my = [], [], []
        for i, sid in enumerate(sids):
            if sid in returns:
                m37.append(X37[i]); m44.append(X44[i]); my.append(returns[sid])
        if m37:
            panel_data[as_of] = (m37, m44, my)

    keys = sorted(panel_data.keys())
    ic37_list, ic44_list = [], []
    for i in range(1, len(keys)):
        te37, te44, te_y = panel_data[keys[i]]
        tr37, tr44, tr_y = [], [], []
        for j in range(i):
            a37, a44, ay = panel_data[keys[j]]
            tr37.extend(a37); tr44.extend(a44); tr_y.extend(ay)
        if len(tr37) < 100:
            continue
        y_tr = winsorize(np.array(tr_y))
        m37 = train_lgb(tr37, y_tr, SPEC_37)
        m44 = train_lgb(tr44, y_tr, SPEC_37 + KWAVE_FEATURES)
        p37 = m37.predict(np.array(te37))
        p44 = m44.predict(np.array(te44))
        ic37_list.append(spearman_ic(p37, te_y))
        ic44_list.append(spearman_ic(p44, te_y))

    if not ic37_list:
        return None
    ic37 = np.array(ic37_list); ic44 = np.array(ic44_list)
    diff = ic44 - ic37
    n = len(diff)
    sd = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    t_stat = float(np.mean(diff) / (sd / math.sqrt(n))) if sd > 1e-12 and n > 1 else 0.0
    mean_ic37 = float(np.mean(ic37)); mean_ic44 = float(np.mean(ic44))
    lift = mean_ic44 - mean_ic37
    drop_minus_full = mean_ic37 - mean_ic44  # §一.15 iv
    keep = drop_minus_full < -IC_GATE_LIFT
    res = {
        "horizon": label, "horizon_days": horizon_days, "n_panels": n,
        "mean_ic_37": mean_ic37, "mean_ic_37plus7": mean_ic44,
        "lift": lift, "drop_minus_full": drop_minus_full,
        "paired_diff_t_stat": t_stat,
        "gate_keep": bool(keep), "gate_threshold_lift": IC_GATE_LIFT,
    }
    sig = "顯著" if abs(t_stat) > 1.997 else "不顯著"
    verdict = "✅ KEEP" if keep else "❌ REVOKE"
    logger.info(f"  IC 37={mean_ic37:+.4f} | 37+7={mean_ic44:+.4f} | lift={lift:+.4f} | drop_minus_full={drop_minus_full:+.4f} | 配對t={t_stat:+.2f}({sig}) | gate={verdict}")
    return res


def main():
    parser = argparse.ArgumentParser(description=f"K-wave IC-Gate Ablation {TOOL_VER}")
    parser.add_argument("--horizons", type=str, default="5,20,60,252")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--limit-panels", type=int, default=0, help="僅用末 N panels(smoke;0=全部)")
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
    logger.info(f"K-wave 多尺度循環特徵 IC-Gate Ablation {TOOL_VER}(§0.3-A / §一.15 iv / §0.3-E)")
    logger.info("=" * 100)
    logger.info(f"  候選 7 特徵: {KWAVE_FEATURES}")
    logger.info(f"  baseline: {len(SPEC_37)} canonical features | probe model: lgbm | seed: {args.seed}")

    conn = get_db_conn()
    try:
        cur = conn.cursor()
        universe = load_universe(cur)
        logger.info(f"  Universe: {len(universe)} stocks")
        panels = get_canonical_panel_dates(FEATURE_SET_VERSION)  # §14.7-DE 單一源
        if args.limit_panels and args.limit_panels < len(panels):
            panels = panels[-args.limit_panels:]
        logger.info(f"  Panels: {len(panels)} ({panels[0][1]} ~ {panels[-1][1]}, data-driven §14.7-DE)")

        logger.info(f"\n建 feature cache(37 from DB + 7 K-wave live faithful-reuse)...")
        feat_cache, drop_log = build_feature_cache(cur, panels, universe)
        if drop_log:
            avg_sids = np.mean([d[1] for d in drop_log])
            avg_drop_kw = np.mean([d[3] for d in drop_log])
            logger.info(f"  feature cache: {len(feat_cache)} panels | 平均 common-support {avg_sids:.0f} 股 | 平均因 K-wave None 剔除 {avg_drop_kw:.1f} 股/panel(§一.8 no silent caps)")

        results = {}
        t_global = time.monotonic()
        for label, days in labels:
            r = evaluate_horizon(cur, panels, feat_cache, days, label)
            if r:
                results[label] = r

        # 綜合裁決(§一.8 honest:季 60d 為最穩健 horizon,權重最高;週 5d IC≈0 視為雜訊)
        logger.info(f"\n{'='*100}\nIC-Gate 綜合裁決(§一.15 iv / §0.3-E)\n{'='*100}")
        logger.info(f"  {'Horizon':10} {'IC_37':>8} {'IC_37+7':>9} {'lift':>8} {'配對t':>7} {'gate':>10}")
        for label, r in results.items():
            sig = "✅" if abs(r["paired_diff_t_stat"]) > 1.997 else " "
            v = "KEEP" if r["gate_keep"] else "REVOKE"
            logger.info(f"  {label:10} {r['mean_ic_37']:>+8.4f} {r['mean_ic_37plus7']:>+9.4f} {r['lift']:>+8.4f} {r['paired_diff_t_stat']:>+6.2f}{sig} {v:>10}")
        robust_keep = any(results.get(h, {}).get("gate_keep") for h in ("quarterly", "annual"))
        logger.info(f"\n  穩健 horizon(季/年)是否有 KEEP: {'是 → 候選 promote(須後續授權 rebuild+retrain)' if robust_keep else '否 → 確認 2026-06-01 移除正確,維持 37'}")
        logger.info(f"  Total elapsed: {time.monotonic()-t_global:.1f}s")

        out_path = args.output or f"reports/kwave_ic_gate_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out_full = Path(_base_dir).parent / out_path
        out_full.parent.mkdir(parents=True, exist_ok=True)
        payload = {label: r for label, r in results.items()}
        payload["_meta"] = {
            "tool": "kwave_ic_gate_ablation.py", "tool_ver": TOOL_VER,
            "candidate_features": KWAVE_FEATURES, "baseline_n_features": len(SPEC_37),
            "probe_model": "lgbm", "run_at": datetime.now().isoformat(),
            "constitution_ver": CONSTITUTION_VER, "seed": SEED,
            "horizons": horizon_days_list, "n_universe": len(universe),
            "n_panels_input": len(panels), "ic_gate_lift_threshold": IC_GATE_LIFT,
            "robust_horizon_keep": bool(robust_keep),
            "source_traceability": "per CLAUDE.md §一.10 — (b) DB query + faithful-reuse feature_store_builder source-pure fns",
            "governance": "research-mode; no DB write; no commit; no SPEC change (§3.1 evaluation)",
        }
        with open(out_full, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        logger.info(f"  Results persisted: {out_full}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
