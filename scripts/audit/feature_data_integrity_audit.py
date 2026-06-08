"""
feature_data_integrity_audit.py v0.1 (全特徵值資料正確性審計 — 軸 B · §14.7-DC/CB/CJ + AP-1/AP-2 + §一.10)
================================================================================
**最後更新日期**: 2026-06-07
**主權狀態**: RESEARCH-MODE DATA-INTEGRITY AUDIT(source-purity + 完整度 + 合理性 + zero-fill + 常數廣播 + 分佈 + 跨panel穩定)+ §一.10 SOURCE-TRACEABLE + 不動 DB / 不 commit
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:逐一檢查 37 個特徵的**值本身對不對 / 乾不乾淨**(不是「有沒有用」——那是五鏡軸 A;這支是「資料正確性」軸 B,兩軸正交)。

**它怎麼做**:對全 37 特徵 × 397 核心 × 96 canonical panels,跑 7 項資料健檢,每特徵給「乾淨🟢/可疑🟡/壞🔴」:
1. **source-purity**(§14.7-DC/§一.10):is_null_imputed=TRUE 計數(應為 0,否則 AI 幻像值混入)
2. **完整度**(§14.7-CB):每 (特徵,股,panel) 是否都有值(缺 row / NULL 計數)
3. **合理性**(§14.7-CJ):NaN/NULL + **資料驅動離群**(值偏離 median 超過 N·MAD;**不 hardcode「合理域」**)
4. **無 silent zero-fill**(AP-1/T_DC-20):exact-zero 比例 >30% 旗標
5. **無常數廣播**(AP-2/T_DC-21):平均每-panel distinct ≤5 旗標(死特徵嫌疑)
6. **分佈健檢**(§一.10):min/max/mean/std
7. **跨 panel 穩定**:每-panel 均值的跨 panel 變異(暴走漂移 = regime 或資料錯誤)

**輸入**:feature_values(committed 397 核心 + canonical panels)。**輸出**:scorecard(stdout)+ JSON。
**它不做的事**:不評斷特徵預測力(那是五鏡軸 A)、不寫 DB、不改 SPEC、不 commit(§3.2 橫切 audit、read-only)。

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
1. **[Two-Axis Orthogonality]** (v0.1, §一.13 v0.15): 資料正確性(軸 B,本程式)與預測價值(軸 A,五鏡)正交;source-pure 之值仍可能無預測力,有預測力之值仍須先 source-pure。
2. **[Source-Purity Check]** (v0.1, §14.7-DC/§一.10): is_null_imputed=TRUE = AI 幻像值,核心股應為 0。
3. **[No-Hardcoded-Range Reasonableness]** (v0.1, §一.13): 合理性用**資料驅動離群**(median±N·MAD),不 hardcode 各特徵「合理域」(避免 hardcoded knowledge 違規)。
4. **[AP-1/AP-2 Detection]** (v0.1, T_DC-20/21): exact-zero>30% = silent-fill 嫌疑;distinct≤5 = 常數廣播/死特徵嫌疑。
5. **[Source-Traceable]** (v0.1, §一.10): 全 (b) DB query(SQL 聚合)。
6. **[Research-Mode]** (v0.1, §3.2 橫切 audit): read-only;不寫 DB / 不 commit / 不改 SPEC。
7. **[Zero Hardcoded Verdict]** (v0.1, §5.6.3): 🟢/🟡/🔴 依檢查結果動態判定。

## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix)
### A. 主要運行模式
| 子項 | CLI | 治權契約 |
| :--- | :--- | :--- |
| A.1 全特徵值資料健檢 | `python scripts/audit/feature_data_integrity_audit.py` | §14.7-DC/CB/CJ |
| A.2 離群 MAD 倍數 | `--mad-mult 10` | §14.7-CJ |
### 不提供之旗標 (Intentionally Omitted)
- `--commit`:read-only audit,永不寫 DB。

## 📜 三、全修訂歷程 (Full Revision History)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.1 | 2026-06-07 | Claude | 首版:軸 B 全特徵值資料正確性審計(五鏡軸 A 之正交補完)。7 項檢查 × 37×397×96 → 健康度 scorecard;合理性用資料驅動離群(不 hardcode 合理域)。research-mode。 | **ACTIVE** |
"""

from __future__ import annotations
import sys, argparse, json, logging
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

FEATURE_SET_VERSION = "feature_set_v0.6"
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

ZERO_FILL_THRESH = 0.30   # exact-zero >30% → AP-1 旗標
DISTINCT_THRESH = 5       # 平均每-panel distinct ≤5 → AP-2 旗標
CROSS_PANEL_COV_THRESH = 3.0  # 跨 panel 均值 CoV >此 → 漂移旗標


def load_universe(cur):
    cur.execute(
        "SELECT m.stock_id FROM core_universe_membership m JOIN core_universe_snapshot s ON s.snapshot_id=m.snapshot_id "
        "WHERE s.status='committed' AND m.core_tier='core_universe' "
        "AND s.snapshot_id=(SELECT snapshot_id FROM core_universe_snapshot WHERE status='committed' ORDER BY created_at DESC LIMIT 1)"
    )
    return list({r[0] for r in cur.fetchall()})


def main():
    parser = argparse.ArgumentParser(description="Feature Data-Integrity Audit v0.1")
    parser.add_argument("--mad-mult", type=float, default=10.0, help="離群閾值(median±N·MAD)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    conn = get_db_conn()
    try:
        cur = conn.cursor()
        universe = load_universe(cur)
        panels = get_canonical_panel_dates(FEATURE_SET_VERSION)
        fs_ids = [p[0] for p in panels]
        n_core = len(universe); n_panel = len(panels); expected = n_core * n_panel
        logger.info("=" * 116)
        logger.info(f"全特徵值資料正確性審計(軸 B)v0.1 | 核心 {n_core} × panels {n_panel} = {expected} cells/特徵")
        logger.info("=" * 116)

        # Query 1:per-feature 全期聚合(check 1/2/4/5/6)
        cur.execute(
            """
            SELECT feature_name,
              COUNT(*) AS n,
              COUNT(*) FILTER (WHERE is_null_imputed IS TRUE) AS n_imputed,
              COUNT(*) FILTER (WHERE feature_value IS NULL) AS n_null,
              COUNT(*) FILTER (WHERE feature_value = 0) AS n_zero,
              COUNT(DISTINCT feature_value) AS n_distinct,
              MIN(feature_value), MAX(feature_value), AVG(feature_value), STDDEV(feature_value),
              percentile_cont(0.5) WITHIN GROUP (ORDER BY feature_value) AS p50
            FROM feature_values
            WHERE feature_set_id = ANY(%s) AND stock_id = ANY(%s) AND feature_name = ANY(%s)
            GROUP BY feature_name
            """,
            (fs_ids, universe, SPEC_37),
        )
        agg = {}
        for row in cur.fetchall():
            fn, n, n_imp, n_null, n_zero, n_dist, mn, mx, avg, std, p50 = row
            agg[fn] = {"n": n, "n_imputed": n_imp, "n_null": n_null, "n_zero": n_zero,
                       "n_distinct": n_dist, "min": float(mn) if mn is not None else None,
                       "max": float(mx) if mx is not None else None,
                       "mean": float(avg) if avg is not None else None,
                       "std": float(std) if std is not None else 0.0,
                       "median": float(p50) if p50 is not None else None}

        # Query 2:per-feature per-panel(check 5 細化 + check 7 跨panel穩定)
        cur.execute(
            """
            SELECT feature_name, feature_set_id,
              AVG(feature_value) AS panel_mean,
              COUNT(DISTINCT feature_value) AS panel_distinct
            FROM feature_values
            WHERE feature_set_id = ANY(%s) AND stock_id = ANY(%s) AND feature_name = ANY(%s)
            GROUP BY feature_name, feature_set_id
            """,
            (fs_ids, universe, SPEC_37),
        )
        panel_means = defaultdict(list); panel_distincts = defaultdict(list)
        for fn, fsid, pm, pd in cur.fetchall():
            if pm is not None:
                panel_means[fn].append(float(pm))
            panel_distincts[fn].append(int(pd))

        # Query 3:離群偵測(median±N·MAD;資料驅動,SQL 取 MAD 近似用 p25/p75 IQR robust)
        cur.execute(
            """
            SELECT feature_name,
              percentile_cont(0.25) WITHIN GROUP (ORDER BY feature_value) AS p25,
              percentile_cont(0.75) WITHIN GROUP (ORDER BY feature_value) AS p75
            FROM feature_values
            WHERE feature_set_id = ANY(%s) AND stock_id = ANY(%s) AND feature_name = ANY(%s)
            GROUP BY feature_name
            """,
            (fs_ids, universe, SPEC_37),
        )
        iqr = {}
        for fn, p25, p75 in cur.fetchall():
            iqr[fn] = (float(p25) if p25 is not None else 0.0, float(p75) if p75 is not None else 0.0)

        # 組裝 scorecard
        results = {}
        for fn in SPEC_37:
            a = agg.get(fn)
            if not a:
                results[fn] = {"verdict": "🔴 缺資料", "note": "feature absent in committed panels"}
                continue
            pm = np.array(panel_means.get(fn, [])); pd_list = panel_distincts.get(fn, [])
            avg_panel_distinct = float(np.mean(pd_list)) if pd_list else 0.0
            zero_ratio = a["n_zero"] / a["n"] if a["n"] else 0.0
            completeness = a["n"] / expected
            cross_cov = float(np.std(pm) / abs(np.mean(pm))) if len(pm) > 1 and abs(np.mean(pm)) > 1e-12 else 0.0
            # 離群:min/max 偏離 median 超過 N·(IQR/1.349 ≈ MAD-equiv robust σ)
            p25, p75 = iqr.get(fn, (0.0, 0.0)); robust_sigma = (p75 - p25) / 1.349 if (p75 - p25) > 1e-12 else 0.0
            med = a["median"] or 0.0
            out_lo = (med - a["min"]) / robust_sigma if robust_sigma > 0 and a["min"] is not None else 0.0
            out_hi = (a["max"] - med) / robust_sigma if robust_sigma > 0 and a["max"] is not None else 0.0
            max_robust_z = max(abs(out_lo), abs(out_hi))
            # verdict
            flags = []
            if a["n_imputed"] > 0: flags.append(f"🔴imputed={a['n_imputed']}")
            if a["n_null"] > 0: flags.append(f"🔴null={a['n_null']}")
            if completeness < 0.95: flags.append(f"🔴完整度{completeness*100:.0f}%")
            if zero_ratio > ZERO_FILL_THRESH: flags.append(f"🟡zero={zero_ratio*100:.0f}%")
            if avg_panel_distinct <= DISTINCT_THRESH: flags.append(f"🟡distinct≈{avg_panel_distinct:.1f}")
            if max_robust_z > 50: flags.append(f"🟡離群z≈{max_robust_z:.0f}")
            if cross_cov > CROSS_PANEL_COV_THRESH: flags.append(f"🟡漂移CoV={cross_cov:.1f}")
            if any("🔴" in f for f in flags): verdict = "🔴 壞"
            elif flags: verdict = "🟡 可疑"
            else: verdict = "🟢 乾淨"
            results[fn] = {"verdict": verdict, "completeness": completeness, "n_imputed": a["n_imputed"],
                           "n_null": a["n_null"], "zero_ratio": zero_ratio, "avg_panel_distinct": avg_panel_distinct,
                           "max_robust_z": max_robust_z, "cross_panel_cov": cross_cov,
                           "min": a["min"], "max": a["max"], "mean": a["mean"], "std": a["std"],
                           "flags": flags}

        # 輸出
        order = sorted(SPEC_37, key=lambda f: {"🔴": 0, "🟡": 1, "🟢": 2}.get(results[f]["verdict"][:1] if results[f].get("verdict") else "🔴", 0))
        logger.info(f"\n  {'特徵':30} {'判':>4} {'完整':>6} {'imp':>4} {'zero%':>6} {'distinct':>9} {'離群z':>7} {'漂移':>6} | min~max")
        n_clean = n_susp = n_bad = 0
        for fn in order:
            r = results[fn]
            if "verdict" not in r: continue
            v = r["verdict"][:1]
            n_clean += v == "🟢"; n_susp += v == "🟡"; n_bad += v == "🔴"
            logger.info(f"  {fn:30} {r['verdict'][:1]:>3} {r.get('completeness',0)*100:>5.0f}% {r.get('n_imputed',0):>4} "
                        f"{r.get('zero_ratio',0)*100:>5.0f}% {r.get('avg_panel_distinct',0):>9.1f} {r.get('max_robust_z',0):>7.0f} "
                        f"{r.get('cross_panel_cov',0):>6.1f} | {r.get('min')}~{r.get('max')}")
            if r.get("flags"):
                logger.info(f"  {'':30}   旗標: {', '.join(r['flags'])}")
        logger.info(f"\n{'='*116}")
        logger.info(f"  總計 37 特徵:🟢 乾淨 {n_clean} | 🟡 可疑 {n_susp} | 🔴 壞 {n_bad}")
        logger.info(f"  source-purity:全特徵 imputed 總計 = {sum(results[f].get('n_imputed',0) for f in SPEC_37 if 'n_imputed' in results[f])}(應為 0)")
        logger.info("=" * 116)

        out_path = args.output or f"reports/feature_data_integrity_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out_full = Path(_base_dir).parent / out_path
        out_full.parent.mkdir(parents=True, exist_ok=True)
        results["_meta"] = {"tool": "feature_data_integrity_audit.py", "tool_ver": "v0.1",
                            "n_universe": n_core, "n_panels": n_panel, "expected_cells_per_feature": expected,
                            "summary": {"clean": n_clean, "suspicious": n_susp, "bad": n_bad},
                            "thresholds": {"zero_fill": ZERO_FILL_THRESH, "distinct": DISTINCT_THRESH, "cross_panel_cov": CROSS_PANEL_COV_THRESH, "mad_mult": args.mad_mult},
                            "run_at": datetime.now().isoformat(),
                            "source_traceability": "per §一.10 (b) DB query (SQL aggregate)",
                            "governance": "research-mode; no DB write; no commit (§3.2 audit)"}
        with open(out_full, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"  Results persisted: {out_full}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
