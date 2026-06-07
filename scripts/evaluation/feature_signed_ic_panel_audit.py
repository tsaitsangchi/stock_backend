"""
feature_signed_ic_panel_audit.py v0.1 (逐特徵有號 IC × 全 96-panel 平均 + sign 穩定度 · §14.7-CM / §14.7-DD PHASE9 / §0.3-E)
================================================================================
**最後更新日期**: 2026-06-07
**主權狀態**: RESEARCH-MODE PER-FEATURE SIGNED-IC PANEL AUDIT + §一.10 SOURCE-TRACEABLE + §一.8 HONEST + 不動 DB / 不 commit
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:逐一檢驗 37 個特徵「**單獨**對未來報酬的正負相關(有號 IC)」,並看這個方向**穩不穩定**(跨時間是否一致),
用來判斷**每個特徵存在的價值** —— IC 越大(不論正負)且方向越穩 = 越有價值;IC≈0 或方向亂跳 = 價值低。

**和既有 `audit_feature_ic_vs_future_return.py` 差別**:那支是 single-snapshot(單一時點,易受該期雜訊誤導);
本支對**全 96 canonical panels** 各算一次再平均 + 給 sign 穩定度 + 跨 panel 配對 t,結論穩健得多。

**它怎麼做**:
1. 397 核心 + canonical panels(§14.7-DE)。2. 每 panel:對每個特徵,算它與未來報酬的橫斷面 Spearman IC(有號)。
3. 跨 panel 聚合:平均 IC(有號)、sign 穩定度(同號 panel 比例)、配對 t、顯著性。4. 4 種持有期各做,依季(60d)|IC| 排名。

**判讀**:|mean IC| 大 + sign 穩定度高 + |t|>2 = 有價值特徵;|mean IC|≈0 或 sign 穩定度≈50% = 價值存疑(候選精簡)。

**source-pure**:全 (b) DB query;Spearman 自實作(無 scipy);純讀取、不訓練、不寫 DB、不 commit(§3.1 evaluation)。

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
1. **[Per-Feature Signed IC]** (v0.1, §14.7-CM): 每特徵 vs forward return 之橫斷面 Spearman rank IC(保留正負號)。
2. **[Panel-Averaged Robustness]** (v0.1, §14.7-DD PHASE9): 全 96 panels 各算後平均,非 single-snapshot。
3. **[Sign Stability]** (v0.1, audit_feature_sign_stability 精神): 同號 panel 比例;判方向可靠性。
4. **[Source-Traceable]** (v0.1, §一.10): 全 (b) DB query;0 scipy。
5. **[Anti-Leakage]** (v0.1, §8.5): forward return ≤ as_of+horizon;特徵 ≤ as_of。
6. **[Research-Mode]** (v0.1, §3.2 audit): read-only;不寫 DB / 不 commit。
7. **[Zero Hardcoded Verdict]** (v0.1, §5.6.3): IC / 顯著性動態計算。

## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix)
### A. 主要運行模式
| 子項 | CLI | 治權契約 |
| :--- | :--- | :--- |
| A.1 全 4-horizon 逐特徵有號 IC | `python scripts/evaluation/feature_signed_ic_panel_audit.py` | §14.7-CM |
| A.2 指定 horizon | `--horizons 60` | §14.7-CY |
### 不提供之旗標 (Intentionally Omitted)
- `--commit`:research-mode,永不寫 DB。

## 📜 三、全修訂歷程 (Full Revision History)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.1 | 2026-06-07 | Claude | 首版:用戶 directive「逐特徵算在 ensemble 中的正負相關係數判斷價值」。37 特徵 × 全 96-panel 平均有號 Spearman IC + sign 穩定度 + 配對 t,補既有 single-snapshot 工具之穩健性缺口。research-mode。 | **ACTIVE** |
"""

from __future__ import annotations
import sys, argparse, math, json, logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

import numpy as np
from core.db_utils import get_db_conn, get_canonical_panel_dates

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

FEATURE_SET_VERSION = "feature_set_v0.5"
SPEC_37 = [
    "log_return_20d", "log_return_60d", "log_return_252d", "ma_ratio_20", "ma_ratio_60", "max_drawdown_252d",
    "upside_volatility_60d", "downside_volatility_60d", "convexity_60d", "volatility_60d", "volatility_252d",
    "upside_capture_60d", "downside_capture_60d",
    "avg_daily_value_log_60d", "avg_daily_value_log_252d", "zero_volume_ratio_252d", "turnover_mean_60d",
    "pe_ratio", "pb_ratio", "dividend_yield",
    "roe_ttm", "operating_margin_ttm", "eps_sum_4q", "net_income_positive_ratio_8q",
    "revenue_yoy_3m_log", "asset_growth_yoy", "revenue_yoy_3m", "revenue_yoy_12m",
    "preferential_attachment_60d", "right_tail_returns_skew_252d",
    "liquidity_rank_pct_sector_60d", "size_log_zscore_sector",
    "foreign_net_20d", "foreign_net_60d", "trust_net_20d", "trust_net_60d", "margin_ratio_60d",
]


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
    r = cur.fetchone()
    ld = r[0] if r and r[0] else None
    if not ld:
        return {}
    cur.execute(
        "WITH t0 AS (SELECT stock_id, close FROM \"TaiwanStockPriceAdj\" WHERE date=%s AND close>0), "
        "t1 AS (SELECT stock_id, close FROM \"TaiwanStockPriceAdj\" WHERE date=%s AND close>0) "
        "SELECT t0.stock_id, LN(t1.close::numeric/t0.close::numeric) FROM t0 JOIN t1 ON t0.stock_id=t1.stock_id",
        (str(as_of), str(ld)),
    )
    return {sid: float(r) for sid, r in cur.fetchall()}


def spearman_ic(x, y):
    x = np.array(x); y = np.array(y)
    rx = x.argsort().argsort().astype(float); ry = y.argsort().argsort().astype(float)
    if np.std(rx) < 1e-10 or np.std(ry) < 1e-10:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    parser = argparse.ArgumentParser(description="Per-Feature Signed-IC Panel Audit v0.1")
    parser.add_argument("--horizons", type=str, default="5,20,60,252")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    horizon_days_list = [int(d.strip()) for d in args.horizons.split(",")]

    conn = get_db_conn()
    try:
        cur = conn.cursor()
        universe = load_universe(cur)
        panels = get_canonical_panel_dates(FEATURE_SET_VERSION)
        logger.info(f"Universe {len(universe)} | Panels {len(panels)} ({panels[0][1]}~{panels[-1][1]})")

        # feature cache(每 panel:sids + {feature: [values 對齊 sids]})
        feat_cache = {}
        for fs_id, as_of in panels:
            feat = load_features(cur, fs_id, universe)
            sids = [s for s in universe if feat.get(s) and all(f in feat[s] for f in SPEC_37)]
            if sids:
                feat_cache[as_of] = (sids, {f: np.array([feat[s][f] for s in sids]) for f in SPEC_37})

        all_results = {}
        for hd in horizon_days_list:
            label = "weekly" if hd <= 7 else "monthly" if hd <= 30 else "quarterly" if hd <= 90 else "annual"
            per_feat_ics = defaultdict(list)
            for fs_id, as_of in panels:
                if as_of not in feat_cache:
                    continue
                sids, fmap = feat_cache[as_of]
                ret = load_forward_returns(cur, as_of, hd)
                idx = [i for i, s in enumerate(sids) if s in ret]
                if len(idx) < 30:
                    continue
                y = np.array([ret[sids[i]] for i in idx])
                for f in SPEC_37:
                    ic = spearman_ic(fmap[f][idx], y)
                    if ic is not None:
                        per_feat_ics[f].append(ic)
            rows = []
            for f in SPEC_37:
                ics = np.array(per_feat_ics[f])
                if len(ics) == 0:
                    continue
                mean_ic = float(np.mean(ics)); n = len(ics)
                sd = float(np.std(ics, ddof=1)) if n > 1 else 0.0
                t = float(mean_ic / (sd / math.sqrt(n))) if sd > 1e-12 else 0.0
                same_sign = float(np.mean(np.sign(ics) == np.sign(mean_ic))) if mean_ic != 0 else 0.5
                rows.append({"feature": f, "mean_ic": mean_ic, "abs_ic": abs(mean_ic),
                             "sign_stability": same_sign, "t_stat": t, "n_panels": n})
            rows.sort(key=lambda r: r["abs_ic"], reverse=True)
            all_results[label] = rows
            logger.info(f"\n{'='*108}\nHorizon {label}({hd}d) — 逐特徵有號 IC(全 {len(panels)}-panel 平均,依 |IC| 排名)\n{'='*108}")
            logger.info(f"  {'#':>2} {'特徵':30} {'有號IC':>9} {'|IC|':>7} {'sign穩定':>8} {'配對t':>8} {'顯著':>5}")
            for i, r in enumerate(rows, 1):
                sig = "✅" if abs(r["t_stat"]) > 1.997 else ""
                logger.info(f"  {i:>2} {r['feature']:30} {r['mean_ic']:>+9.4f} {r['abs_ic']:>7.4f} {r['sign_stability']*100:>7.0f}% {r['t_stat']:>+8.2f} {sig:>5}")

        out_path = args.output or f"reports/feature_signed_ic_panel_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out_full = Path(_base_dir).parent / out_path
        out_full.parent.mkdir(parents=True, exist_ok=True)
        all_results["_meta"] = {"tool": "feature_signed_ic_panel_audit.py", "tool_ver": "v0.1",
                                "n_universe": len(universe), "n_panels": len(panels),
                                "horizons": horizon_days_list, "run_at": datetime.now().isoformat(),
                                "source_traceability": "per §一.10 (b) DB query", "governance": "research-mode; no DB write"}
        with open(out_full, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"\n  Results persisted: {out_full}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
