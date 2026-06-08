"""
feature_pruning_ablation.py v0.1 (嚴格有用性 gate 精簡集驗證 — 37 vs 26 · §14.7-DI / §14.7-DC T_DC-28 / §一.15)
================================================================================
**最後更新日期**: 2026-06-07
**主權狀態**: RESEARCH-MODE PRUNED-SET ABLATION(嚴格 gate 26 vs 全 37 + 分解 trust_net / 共線)+ §一.10 SOURCE-TRACEABLE + 不動 DB / 不 commit
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:驗證「嚴格有用性 gate」精簡掉 11 個特徵(trust_net×2 + 4 共線群各留 1 砍 9)後,
選股預測力(IC)會不會掉 —— **證明精簡不傷(預期持平或微升)才值得真的 rebuild+retrain**。

**為什麼**:用戶原則「特徵須具嚴格有用性;沒用特徵只造成預測錯誤」。但實際移除前須先**實證**精簡集不損 IC
(五鏡 + 資料正確性已指出 trust_net 兩軸皆敗、共線群冗餘)。此為「先驗證再決定」之驗證步。

**它怎麼做**:walk-forward(防洩漏)比較 4 個特徵集 ——
- `full_37`(全 37,baseline)
- `pruned_26`(嚴格 gate:砍 11)
- `trustnet_drop_35`(只砍 trust_net×2 → 隔離噪音移除效果)
- `collinear_drop_28`(只砍 9 共線 → 隔離冗餘移除效果)
對每 horizon 算橫斷面 IC + 與 full_37 之 delta + per-panel 配對 t。delta ≥ −0.005 = 不傷可砍。

**輸入**:feature_values(397 核心 + canonical panels)。**輸出**:JSON + log。
**它不做的事**:不寫 DB、不改 SPEC、不 commit、不 retrain production(§3.1);實際精簡須另經 §14.7-DC T_DC-28 + 授權 + rebuild + retrain。

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
1. **[Strict-Usefulness Gate Verification]** (v0.1, §14.7-DI/§一.15): 精簡集須先實證 walk-forward IC 不掉,方可移除(non-degradation gate)。
2. **[Effect Decomposition]** (v0.1): 分 trust_net-only / collinear-only / 全砍,隔離各來源效果(避免混淆)。
3. **[Source-Traceable]** (v0.1, §一.10): 全 (b) DB query;probe = lgbm(與 dead_weight ablation 同 recipe)。
4. **[Anti-Leakage]** (v0.1, §8.5): walk-forward train=過去 / test=當前;DELTA 為相對比較(洩漏對各集等影響,relative 仍有效)。
5. **[Research-Mode]** (v0.1, §3.1): read-only;不寫 DB / 不 commit / 不改 SPEC。
6. **[Zero Hardcoded Verdict]** (v0.1, §5.6.3): KEEP/REVERT 依 delta 動態判定。

## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix)
### A. 主要運行模式
| 子項 | CLI | 治權契約 |
| :--- | :--- | :--- |
| A.1 全 4-horizon 精簡驗證 | `python scripts/evaluation/feature_pruning_ablation.py` | §14.7-DI |
| A.2 指定 horizon | `--horizons 60,252` | §14.7-CY |
| A.3 threads(對並行客氣) | `--threads 3` | 維運 |
### 不提供之旗標 (Intentionally Omitted)
- `--commit`:research-mode,永不寫 DB / SPEC。

## 📜 三、全修訂歷程 (Full Revision History)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.1 | 2026-06-07 | Claude | 首版:用戶「嚴格有用性 gate → 先驗證再決定」。比較 full_37 / pruned_26(砍 trust_net×2 + 共線×9)/ trustnet-only / collinear-only,4-horizon walk-forward IC + 配對 t,證明精簡不傷。research-mode。 | **ACTIVE** |
"""

from __future__ import annotations
import sys, argparse, math, json, logging, time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

import numpy as np
import lightgbm as lgb
from core.db_utils import get_db_conn, get_canonical_panel_dates

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

CONSTITUTION_VER = "v6.1.0"; TOOL_VER = "v0.1"; SEED = 5422
FEATURE_SET_VERSION = "feature_set_v0.6"; N_ESTIMATORS = 200

LGB_PARAMS = {"learning_rate": 0.05, "max_depth": 5, "num_leaves": 20, "min_child_samples": 30,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5, "reg_alpha": 0.1,
              "reg_lambda": 0.1, "objective": "regression", "metric": "rmse", "verbose": -1, "seed": SEED}

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

# 嚴格 gate 移除清單(§14.7-DI:預測力 fail + 共線冗餘)
DROP_NOISE = ["trust_net_20d", "trust_net_60d", "net_income_positive_ratio_8q"]  # 預測力 fail(兩穩健 horizon |t|<1.997)
DROP_COLLINEAR = ["avg_daily_value_log_252d", "turnover_mean_60d",        # 群1 留 avg_daily_value_log_60d
                  "preferential_attachment_60d", "liquidity_rank_pct_sector_60d", "size_log_zscore_sector",
                  "upside_volatility_60d", "upside_capture_60d",          # 群2 留 volatility_60d
                  "downside_capture_60d",                                 # 群3 留 downside_volatility_60d
                  "revenue_yoy_3m"]                                       # 群4 留 revenue_yoy_3m_log
DROP_ALL = DROP_NOISE + DROP_COLLINEAR  # 共 12

NONDEGRADE = -0.005  # delta vs full ≥ 此 → 不傷可砍


def variant_cols():
    def idx(drop): return [i for i, f in enumerate(SPEC_37) if f not in drop]
    return {
        "full_37": list(range(len(SPEC_37))),
        "pruned_25": idx(DROP_ALL),
        "noise_drop_34": idx(DROP_NOISE),
        "collinear_drop_28": idx(DROP_COLLINEAR),
    }


def load_universe(cur):
    cur.execute(
        "SELECT m.stock_id FROM core_universe_membership m JOIN core_universe_snapshot s ON s.snapshot_id=m.snapshot_id "
        "WHERE s.status='committed' AND m.core_tier='core_universe' "
        "AND s.snapshot_id=(SELECT snapshot_id FROM core_universe_snapshot WHERE status='committed' ORDER BY created_at DESC LIMIT 1)"
    )
    return list({r[0] for r in cur.fetchall()})


def load_features(cur, fs_id, universe):
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


def load_forward_returns(cur, as_of, horizon_days):
    cur.execute(
        "SELECT MIN(date) FROM \"TaiwanStockPriceAdj\" WHERE date >= (%s::date + INTERVAL '%s days') "
        "AND stock_id ~ '^[0-9]' AND date <= (%s::date + INTERVAL '%s days')",
        (str(as_of), horizon_days, str(as_of), horizon_days + 14),
    )
    r = cur.fetchone(); ld = r[0] if r and r[0] else None
    if not ld:
        return {}
    cur.execute(
        "WITH t0 AS (SELECT stock_id, close FROM \"TaiwanStockPriceAdj\" WHERE date=%s AND close>0), "
        "t1 AS (SELECT stock_id, close FROM \"TaiwanStockPriceAdj\" WHERE date=%s AND close>0) "
        "SELECT t0.stock_id, LN(t1.close::numeric/t0.close::numeric) FROM t0 JOIN t1 ON t0.stock_id=t1.stock_id",
        (str(as_of), str(ld)),
    )
    return {sid: float(r) for sid, r in cur.fetchall()}


def spearman_ic(pred, y):
    pred = np.array(pred); y = np.array(y)
    rp = pred.argsort().argsort().astype(float); ry = y.argsort().argsort().astype(float)
    if np.std(rp) < 1e-10 or np.std(ry) < 1e-10:
        return 0.0
    return float(np.corrcoef(rp, ry)[0, 1])


def winsorize(arr, lo_q=0.01, hi_q=0.99):
    return np.clip(arr, np.quantile(arr, lo_q), np.quantile(arr, hi_q))


def build_cache(cur, panels, universe):
    cache = {}
    for fs_id, as_of in panels:
        feat = load_features(cur, fs_id, universe)
        sids = [s for s in universe if feat.get(s) and all(f in feat[s] for f in SPEC_37)]
        if sids:
            cache[as_of] = (sids, np.array([[feat[s][f] for f in SPEC_37] for s in sids]))
    return cache


def evaluate_horizon(cur, panels, cache, horizon_days, label, variants, threads, purge=False, embargo_days=5):
    logger.info(f"\n{'='*100}\nHorizon {label}({horizon_days}d){' [PURGED embargo='+str(embargo_days)+'d]' if purge else ' [raw]'}\n{'='*100}")
    pdata = {}
    for fs_id, as_of in panels:
        if as_of not in cache:
            continue
        sids, X = cache[as_of]
        ret = load_forward_returns(cur, as_of, horizon_days)
        idx = [i for i, s in enumerate(sids) if s in ret]
        if not idx:
            continue
        pdata[as_of] = (X[idx, :], np.array([ret[sids[i]] for i in idx]))

    keys = sorted(pdata.keys())
    ic = {v: [] for v in variants}
    params = dict(LGB_PARAMS); params["num_threads"] = threads
    purge_cal = int(horizon_days * 1.45) + embargo_days  # §8.5 embargo:trading→calendar 近似
    for i in range(1, len(keys)):
        t_i = keys[i]
        Xte, yte = pdata[t_i]
        tr_js = [j for j in range(i) if (not purge) or (t_i - keys[j]).days > purge_cal]
        if not tr_js:
            continue
        trX = [pdata[keys[j]][0] for j in tr_js]; trY = [pdata[keys[j]][1] for j in tr_js]
        Xtr = np.vstack(trX); ytr = winsorize(np.concatenate(trY))
        if len(Xtr) < 100:
            continue
        for v, cols in variants.items():
            names = [SPEC_37[c] for c in cols]
            m = lgb.train(params, lgb.Dataset(Xtr[:, cols], label=ytr, feature_name=names), num_boost_round=N_ESTIMATORS)
            ic[v].append(spearman_ic(m.predict(Xte[:, cols]), yte))

    base = np.array(ic["full_37"])
    if len(base) == 0:
        return None
    out = {"horizon": label, "horizon_days": horizon_days, "n_panels": len(base),
           "full_37_ic": float(np.mean(base)), "variants": {}}
    logger.info(f"  full_37 IC = {np.mean(base):+.4f}")
    for v in variants:
        if v == "full_37":
            continue
        arr = np.array(ic[v]); diff = arr - base; n = len(diff)
        sd = float(np.std(diff, ddof=1)) if n > 1 else 0.0
        t = float(np.mean(diff) / (sd / math.sqrt(n))) if sd > 1e-12 else 0.0
        delta = float(np.mean(arr) - np.mean(base))
        verdict = "✅ 不傷可砍" if delta >= NONDEGRADE else ("⚠️ 邊界" if delta >= -0.01 else "❌ 有損保留")
        out["variants"][v] = {"ic": float(np.mean(arr)), "delta_vs_full": delta, "paired_t": t, "verdict": verdict}
        logger.info(f"  {v:20} IC={np.mean(arr):+.4f} delta={delta:+.4f} t={t:+.2f} → {verdict}")
    return out


def main():
    parser = argparse.ArgumentParser(description=f"Feature Pruning Ablation {TOOL_VER}")
    parser.add_argument("--horizons", type=str, default="5,20,60,252")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--purge", action="store_true", help="purged-CV(§8.5 embargo 剔重疊訓練 panel)")
    parser.add_argument("--embargo-days", type=int, default=5)
    parser.add_argument("--limit-panels", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    hs = [int(d) for d in args.horizons.split(",")]
    labels = [("weekly" if d <= 7 else "monthly" if d <= 30 else "quarterly" if d <= 90 else "annual", d) for d in hs]
    variants = variant_cols()
    logger.info("=" * 100)
    logger.info(f"嚴格有用性 gate 精簡集驗證 {TOOL_VER}(§14.7-DI)")
    logger.info(f"  full_37 / pruned_25(砍 {len(DROP_ALL)}) / noise_drop_34 / collinear_drop_28 | probe=lgbm(threads={args.threads})")
    logger.info(f"  砍 預測力fail: {DROP_NOISE}")
    logger.info(f"  砍 共線(各群留1): {DROP_COLLINEAR}")
    logger.info("=" * 100)

    conn = get_db_conn()
    try:
        cur = conn.cursor()
        universe = load_universe(cur)
        panels = get_canonical_panel_dates(FEATURE_SET_VERSION)
        if args.limit_panels and args.limit_panels < len(panels):
            panels = panels[-args.limit_panels:]
        logger.info(f"  Universe {len(universe)} | Panels {len(panels)}")
        cache = build_cache(cur, panels, universe)
        logger.info(f"  cache {len(cache)} panels\n")

        results = {}
        t0 = time.monotonic()
        for label, d in labels:
            r = evaluate_horizon(cur, panels, cache, d, label, variants, args.threads, args.purge, args.embargo_days)
            if r:
                results[label] = r

        logger.info(f"\n{'='*100}\n精簡集綜合裁決(delta vs full_37;季/年穩健權重)\n{'='*100}")
        logger.info(f"  {'Horizon':10} {'pruned_25':>22} {'noise_34':>22} {'collinear_28':>22}")
        for label, _ in labels:
            r = results.get(label)
            if not r:
                continue
            def cell(v):
                x = r["variants"].get(v, {})
                return f"{x.get('delta_vs_full',0):+.4f}({x.get('paired_t',0):+.1f})"
            logger.info(f"  {label:10} {cell('pruned_25'):>22} {cell('noise_drop_34'):>22} {cell('collinear_drop_28'):>22}")
        # pruned_25 在穩健 horizon 是否不傷
        robust_ok = all(results.get(h, {}).get("variants", {}).get("pruned_25", {}).get("delta_vs_full", -1) >= NONDEGRADE
                        for h in ("quarterly", "annual") if h in results)
        logger.info(f"\n  pruned_25 在季/年是否「不傷」(delta≥{NONDEGRADE}): {'是 → 精簡集可採,進 rebuild+retrain(需授權)' if robust_ok else '否 → 須保守保留'}")
        logger.info(f"  Total elapsed: {time.monotonic()-t0:.1f}s")

        out_path = args.output or f"reports/feature_pruning_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out_full = Path(_base_dir).parent / out_path
        out_full.parent.mkdir(parents=True, exist_ok=True)
        results["_meta"] = {"tool": "feature_pruning_ablation.py", "tool_ver": TOOL_VER,
                            "full_n": len(SPEC_37), "pruned_n": len(SPEC_37) - len(DROP_ALL),
                            "drop_noise": DROP_NOISE, "drop_collinear": DROP_COLLINEAR,
                            "nondegrade_thresh": NONDEGRADE, "probe": "lgbm", "threads": args.threads,
                            "run_at": datetime.now().isoformat(), "constitution_ver": CONSTITUTION_VER,
                            "seed": SEED, "horizons": hs, "n_universe": len(universe), "n_panels": len(panels),
                            "robust_pruned_ok": bool(robust_ok),
                            "source_traceability": "per §一.10 (b) DB query",
                            "governance": "research-mode; no DB write; no commit (§3.1)"}
        with open(out_full, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\n  Results persisted: {out_full}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
