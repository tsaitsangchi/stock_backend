"""
dead_weight_feature_ablation.py v0.1 (死重特徵 Leave-One-Out Ablation · §14.7-DC T_DC-28 / §一.15 / §0.3-E)
================================================================================
**最後更新日期**: 2026-06-07
**主權狀態**: RESEARCH-MODE LEAVE-ONE-OUT NECESSITY ABLATION + §一.10 SOURCE-TRACEABLE + §一.8 HONEST + 不動 DB / 不 commit / 不改 SPEC
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:檢驗 3 個「LightGBM gain=0」的疑似死重特徵(`zero_volume_ratio_252d` / `revenue_yoy_3m` /
`preferential_attachment_60d`),**把它們一個一個拿掉,看選股預測力(IC)會不會變差** —— 不變差就是死重、可移除(降噪)。

**為什麼**:單一 pooled 模型的 gain=0 只是「線索」(可能被相關特徵掩蓋真實貢獻);要判定能不能移除,
正規測試 = **leave-one-out**(留一法):37 全特徵 vs 拿掉某一個的 36 特徵,比 walk-forward IC。
移除後 IC 不掉(或更好)→ 死重,走 §14.7-DC / T_DC-28 移除流程;IC 明顯掉 → 其實有用,保留。

**它怎麼做**:
1. 397 核心 + canonical panels(§14.7-DE)。2. 每 panel 載 37 特徵 + forward return。
3. walk-forward,每步訓練 5 個模型:baseline(37)、各拿掉 1 個(36)×3、全拿掉 3 個(34);對同一 test 算 IC。
4. 每 horizon 報:各變體 IC、與 baseline 的 delta、per-panel 配對 t。5. 4 種持有期;季(60d)最穩健權重最高。

**判定**:delta = IC(移除後) − IC(baseline)。delta ≥ −0.005(噪音內)→ 安全可移除;delta > 0 → 移除反而更好(該特徵是雜訊);delta < −0.01 → 移除有損,保留。

**輸入**:DB(feature_values + TaiwanStockPriceAdj)。**輸出**:JSON + log。
**它不做的事**:不寫 DB、不改 SPEC、不 commit、不 retrain(§3.1 evaluation);移除決定 + rebuild + retrain 為後續獨立授權。

## 📜 一、核心定義說明 (Core Definitions / The Constitution)

1. **[Leave-One-Out Necessity Test]** (v0.1, §0.3-E / §一.15): 特徵必要性 = 移除後 walk-forward IC 是否下降;不降即非必要(死重)。
2. **[Dead-Weight Removal Doctrine]** (v0.1, §14.7-DC T_DC-28): 移除後 IC 不傷之 source-pure 特徵 = 死重(zero discrimination / zero contribution),可移除以降噪降過擬合(amihud 先例)。
3. **[Source-Traceable]** (v0.1, §一.10): 全 (b) DB query;IC 數字可 trace。
4. **[Anti-Leakage Walk-Forward]** (v0.1, §8.5): 每 panel ≤ as_of;train=過去 / test=當前。
5. **[Shared-Train Fair Compare]** (v0.1): 同一 (train_X, train_y) 下訓練各欄位子集,唯一差異 = 移除的特徵欄。
6. **[Honest Disclosure]** (v0.1, §一.8): 報 delta + 配對 t;週 IC≈0 視雜訊明示;單模型 gain=0 為線索非定論。
7. **[Research-Mode]** (v0.1, §3.1): read-only;不寫 DB / 不 commit / 不改 SPEC。
8. **[Zero Hardcoded Verdict]** (v0.1, §5.6.3): REMOVE/KEEP 依 ablation delta 動態判定。

## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix)

### A. 主要運行模式
| 子項 | CLI | 治權契約 |
| :--- | :--- | :--- |
| A.1 全 4-horizon 死重 ablation | `python scripts/evaluation/dead_weight_feature_ablation.py` | §14.7-DC T_DC-28 |
| A.2 指定 horizon | `--horizons 60,252` | §14.7-CY |
| A.3 限執行緒(對並行任務客氣) | `--threads 3` | 維運 |

### B. 補充運行模式
| 子項 | CLI | 治權契約 |
| :--- | :--- | :--- |
| B.1 快速 smoke | `--limit-panels 12 --horizons 60` | 開發驗證 |

### 不提供之旗標 (Intentionally Omitted)
- `--commit`:research-mode,永不寫 DB / SPEC。

### 對齊憲章 §二 維運矩陣
| 場景 | 命令 |
| :--- | :--- |
| 死重特徵必要性裁決 | `python scripts/evaluation/dead_weight_feature_ablation.py` |

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.1 | 2026-06-07 | Claude | **首版**:用戶 directive,對 3 個 LightGBM gain=0 疑似死重特徵(zero_volume_ratio_252d / revenue_yoy_3m / preferential_attachment_60d)做 leave-one-out 必要性 ablation。baseline(37)vs 各移除 1(36)×3 + 全移除 3(34),walk-forward IC + 配對 t。研究模式:不動 DB / 不 commit / 不改 SPEC。 | **ACTIVE** |
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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

CONSTITUTION_VER = "v6.1.0"
TOOL_VER = "v0.1"
SEED = 5422
FEATURE_SET_VERSION = "feature_set_v0.6"

LGB_PARAMS = {"learning_rate": 0.05, "max_depth": 5, "num_leaves": 20, "min_child_samples": 30,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
              "reg_alpha": 0.1, "reg_lambda": 0.1, "objective": "regression", "metric": "rmse",
              "verbose": -1, "seed": SEED}
N_ESTIMATORS = 200

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

DROP_CANDIDATES = ["zero_volume_ratio_252d", "revenue_yoy_3m", "preferential_attachment_60d"]
REMOVE_SAFE_DELTA = -0.005   # delta ≥ 此值 → 移除安全(IC 未明顯下降)
KEEP_DELTA = -0.01           # delta < 此值 → 移除有損,保留


def build_variants():
    """回傳 {變體名: [納入的欄 index]}。"""
    variants = {"baseline_37": list(range(len(SPEC_37)))}
    for c in DROP_CANDIDATES:
        ci = SPEC_37.index(c)
        variants[f"drop_{c}"] = [i for i in range(len(SPEC_37)) if i != ci]
    drop3 = {SPEC_37.index(c) for c in DROP_CANDIDATES}
    variants["drop_all3"] = [i for i in range(len(SPEC_37)) if i not in drop3]
    return variants


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


def load_forward_returns(cur, as_of, horizon_days):
    cur.execute(
        "SELECT MIN(date) FROM \"TaiwanStockPriceAdj\" WHERE date >= (%s::date + INTERVAL '%s days') "
        "AND stock_id ~ '^[0-9]' AND date <= (%s::date + INTERVAL '%s days')",
        (str(as_of), horizon_days, str(as_of), horizon_days + 14),
    )
    r = cur.fetchone()
    label_date = r[0] if r and r[0] else None
    if not label_date:
        return {}
    cur.execute(
        "WITH t0 AS (SELECT stock_id, close FROM \"TaiwanStockPriceAdj\" WHERE date=%s AND close>0), "
        "t1 AS (SELECT stock_id, close FROM \"TaiwanStockPriceAdj\" WHERE date=%s AND close>0) "
        "SELECT t0.stock_id, LN(t1.close::numeric/t0.close::numeric) FROM t0 JOIN t1 ON t0.stock_id=t1.stock_id",
        (str(as_of), str(label_date)),
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


def build_feature_cache(cur, panels, universe):
    cache = {}
    for fs_id, as_of in panels:
        feat = load_features_37(cur, fs_id, universe)
        if not feat:
            continue
        sids, X = [], []
        for sid in universe:
            row = feat.get(sid)
            if row and all(f in row for f in SPEC_37):
                sids.append(sid)
                X.append([row[f] for f in SPEC_37])
        if sids:
            cache[as_of] = (sids, np.array(X))
    return cache


def evaluate_horizon(cur, panels, feat_cache, horizon_days, label, variants, threads):
    logger.info(f"\n{'='*100}\nHorizon: {label}({horizon_days}d)\n{'='*100}")
    panel_data = {}
    for fs_id, as_of in panels:
        if as_of not in feat_cache:
            continue
        sids, X = feat_cache[as_of]
        returns = load_forward_returns(cur, as_of, horizon_days)
        if not returns:
            continue
        idxs = [i for i, sid in enumerate(sids) if sid in returns]
        if not idxs:
            continue
        y = np.array([returns[sids[i]] for i in idxs])
        panel_data[as_of] = (X[idxs, :], y)

    keys = sorted(panel_data.keys())
    ic_lists = {v: [] for v in variants}
    params = dict(LGB_PARAMS); params["num_threads"] = threads
    for i in range(1, len(keys)):
        Xte, yte = panel_data[keys[i]]
        trX, trY = [], []
        for j in range(i):
            aX, ay = panel_data[keys[j]]
            trX.append(aX); trY.append(ay)
        Xtr = np.vstack(trX); ytr = np.concatenate(trY)
        if len(Xtr) < 100:
            continue
        y_tr = winsorize(ytr)
        for vname, cols in variants.items():
            names = [SPEC_37[c] for c in cols]
            m = lgb.train(params, lgb.Dataset(Xtr[:, cols], label=y_tr, feature_name=names), num_boost_round=N_ESTIMATORS)
            ic_lists[vname].append(spearman_ic(m.predict(Xte[:, cols]), yte))

    base = ic_lists["baseline_37"]
    if not base:
        return None
    base = np.array(base)
    out = {"horizon": label, "horizon_days": horizon_days, "n_panels": len(base),
           "mean_ic_baseline_37": float(np.mean(base)), "variants": {}}
    logger.info(f"  baseline(37) IC = {np.mean(base):+.4f}")
    for vname in variants:
        if vname == "baseline_37":
            continue
        v = np.array(ic_lists[vname])
        diff = v - base  # 移除後 − baseline
        n = len(diff)
        sd = float(np.std(diff, ddof=1)) if n > 1 else 0.0
        t = float(np.mean(diff) / (sd / math.sqrt(n))) if sd > 1e-12 and n > 1 else 0.0
        delta = float(np.mean(v) - np.mean(base))
        if delta >= REMOVE_SAFE_DELTA:
            verdict = "✅ 可移除(IC 未降)"
        elif delta < KEEP_DELTA:
            verdict = "❌ 保留(移除有損)"
        else:
            verdict = "⚠️ 邊界"
        out["variants"][vname] = {"mean_ic": float(np.mean(v)), "delta_vs_baseline": delta,
                                  "paired_t": t, "verdict": verdict}
        logger.info(f"  {vname:42} IC={np.mean(v):+.4f} delta={delta:+.4f} t={t:+.2f} → {verdict}")
    return out


def main():
    parser = argparse.ArgumentParser(description=f"Dead-Weight Feature Ablation {TOOL_VER}")
    parser.add_argument("--horizons", type=str, default="5,20,60,252")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--threads", type=int, default=3, help="lgbm num_threads(對並行任務客氣)")
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

    variants = build_variants()
    logger.info("=" * 100)
    logger.info(f"死重特徵 Leave-One-Out Ablation {TOOL_VER}(§14.7-DC T_DC-28 / §一.15)")
    logger.info("=" * 100)
    logger.info(f"  死重候選: {DROP_CANDIDATES}")
    logger.info(f"  變體: {list(variants.keys())} | probe: lgbm(threads={args.threads}) | seed: {args.seed}")

    conn = get_db_conn()
    try:
        cur = conn.cursor()
        universe = load_universe(cur)
        logger.info(f"  Universe: {len(universe)} stocks")
        panels = get_canonical_panel_dates(FEATURE_SET_VERSION)
        if args.limit_panels and args.limit_panels < len(panels):
            panels = panels[-args.limit_panels:]
        logger.info(f"  Panels: {len(panels)} ({panels[0][1]} ~ {panels[-1][1]}, §14.7-DE)")

        logger.info(f"\n建 feature cache(37 from DB)...")
        feat_cache = build_feature_cache(cur, panels, universe)
        logger.info(f"  cache: {len(feat_cache)} panels")

        results = {}
        t_global = time.monotonic()
        for label, days in labels:
            r = evaluate_horizon(cur, panels, feat_cache, days, label, variants, args.threads)
            if r:
                results[label] = r

        # 綜合裁決(per 候選,跨 horizon;季/年為穩健權重)
        logger.info(f"\n{'='*100}\n死重綜合裁決(每候選逐 horizon delta;季/年穩健權重)\n{'='*100}")
        for c in DROP_CANDIDATES:
            vk = f"drop_{c}"
            logger.info(f"\n  ▸ {c}")
            for label, _ in labels:
                r = results.get(label)
                if r and vk in r["variants"]:
                    vv = r["variants"][vk]
                    logger.info(f"      {label:10} delta={vv['delta_vs_baseline']:+.4f} t={vv['paired_t']:+.2f} {vv['verdict']}")
        logger.info(f"\n  ▸ drop_all3(同時移除 3 個)")
        for label, _ in labels:
            r = results.get(label)
            if r and "drop_all3" in r["variants"]:
                vv = r["variants"]["drop_all3"]
                logger.info(f"      {label:10} delta={vv['delta_vs_baseline']:+.4f} t={vv['paired_t']:+.2f} {vv['verdict']}")
        logger.info(f"\n  Total elapsed: {time.monotonic()-t_global:.1f}s")

        out_path = args.output or f"reports/dead_weight_feature_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out_full = Path(_base_dir).parent / out_path
        out_full.parent.mkdir(parents=True, exist_ok=True)
        results["_meta"] = {
            "tool": "dead_weight_feature_ablation.py", "tool_ver": TOOL_VER,
            "drop_candidates": DROP_CANDIDATES, "baseline_n_features": len(SPEC_37),
            "remove_safe_delta": REMOVE_SAFE_DELTA, "keep_delta": KEEP_DELTA,
            "probe_model": "lgbm", "threads": args.threads,
            "run_at": datetime.now().isoformat(), "constitution_ver": CONSTITUTION_VER,
            "seed": SEED, "horizons": horizon_days_list, "n_universe": len(universe),
            "n_panels_input": len(panels),
            "source_traceability": "per §一.10 — (b) DB query",
            "governance": "research-mode; no DB write; no commit; no SPEC change (§3.1)",
        }
        with open(out_full, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\n  Results persisted: {out_full}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
