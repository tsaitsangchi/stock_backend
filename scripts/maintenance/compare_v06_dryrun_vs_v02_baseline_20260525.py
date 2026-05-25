"""
compare_v06_dryrun_vs_v02_baseline_20260525.py v0.1
================================================================================
最後更新日期: 2026-05-25
主權狀態: ABLATION-COMPARISON (對齊 §14.7-BH ablation 之 universe 名單實證)
最高原則: Empirical Universe Diff Witness

對 builder v0.7.1 (RMS) 跑 dry-run 取出 v0.6 policy universe;
從 DB 讀 v0.2 baseline universe;
比對 core_universe / convex_universe 之 stock_id 集合差異(churn rate)。

純 SELECT-only + 純 stdout + 不寫 DB;對映 §0.0-G 跑通 + §14.7-AX 公式層揭露第 7 次驗證落地。

執行: python scripts/maintenance/compare_v06_dryrun_vs_v02_baseline_20260525.py
================================================================================
"""
import sys
from datetime import date
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_SCRIPTS_DIR = _THIS_FILE.parent.parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from core.db_utils import get_db_connection
from core.core_universe_builder import CoreUniverseBuilder


AS_OF_DATE = date(2026, 5, 21)
V02_SNAPSHOT_ID = "core_universe_20260521_core_universe_policy_v0_2"


def fetch_v02_baseline():
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            '''
            SELECT stock_id, stock_name, industry_category, core_tier
            FROM "core_universe_membership"
            WHERE "snapshot_id" = %s
              AND "core_tier" IN ('core_universe', 'convex_universe')
            ORDER BY core_tier, stock_id
            ''',
            (V02_SNAPSHOT_ID,),
        )
        return cur.fetchall()
    finally:
        cur.close()
        conn.close()


def run_v06_dryrun():
    print(f"🔬 [v0.7.1 dry-run] 計算 v0.6 policy universe @ {AS_OF_DATE}...")
    builder = CoreUniverseBuilder(
        as_of_date=AS_OF_DATE,
        policy_version="core_universe_policy_v0.6",
        commit=False,
        special_rebalance_reason="v0.7.1 RMS 對齊後比對 v0.2 baseline universe(§14.7-BH ablation 落地驗證)",
        include_emerging=False,
    )
    if not builder.preflight_check():
        print("❌ Preflight failed")
        sys.exit(1)
    builder._market_data = builder._load_market_data()
    candidates = builder.load_candidates()
    return candidates


def main():
    print("=" * 80)
    print("v0.7.1 RMS dry-run vs v0.2 baseline universe — 實證對照")
    print(f"as_of_date={AS_OF_DATE} / v0.2 snapshot={V02_SNAPSHOT_ID}")
    print("=" * 80)

    # v0.2 baseline
    v02_rows = fetch_v02_baseline()
    v02_core = {r[0] for r in v02_rows if r[3] == "core_universe"}
    v02_convex = {r[0] for r in v02_rows if r[3] == "convex_universe"}
    v02_stock_info = {r[0]: (r[1], r[2]) for r in v02_rows}
    print(f"\n📌 v0.2 baseline: core={len(v02_core)}, convex={len(v02_convex)}")

    # v0.7.1 dry-run
    candidates = run_v06_dryrun()
    v06_core = {c.stock_id for c in candidates if c.core_tier == "core_universe"}
    v06_convex = {c.stock_id for c in candidates if c.core_tier == "convex_universe"}
    v06_quarantine = {c.stock_id for c in candidates if c.core_tier == "quarantine_universe"}
    v06_research = {c.stock_id for c in candidates if c.core_tier == "research_universe"}
    print(f"\n📌 v0.7.1 dry-run: core={len(v06_core)}, convex={len(v06_convex)}, "
          f"quarantine={len(v06_quarantine)}, research={len(v06_research)}")

    # candidate stock info lookup
    v06_info = {c.stock_id: (c.stock_name, c.industry_category, c.core_score, c.theme_score,
                              c.fundamental_score, c.institutional_flow_score,
                              c.volatility_control_score)
                for c in candidates}

    # core_universe diff
    print("\n" + "=" * 80)
    print("📊 CORE UNIVERSE (top-120) 差異")
    print("=" * 80)
    core_intersection = v02_core & v06_core
    core_v02_only = v02_core - v06_core  # v0.2 有 v0.7.1 沒有(被踢出)
    core_v06_only = v06_core - v02_core  # v0.7.1 有 v0.2 沒有(新進)
    churn_rate = (len(core_v02_only) + len(core_v06_only)) / (len(v02_core) + len(v06_core)) * 2
    print(f"  Intersection: {len(core_intersection)}/{len(v02_core)} ({len(core_intersection)/len(v02_core)*100:.1f}%)")
    print(f"  v0.2 only (被踢出): {len(core_v02_only)}")
    print(f"  v0.7.1 only (新進): {len(core_v06_only)}")
    print(f"  Churn rate (對稱):  {churn_rate:.4f} (= 2 × |Δ| / |A|+|B|)")
    print(f"  Jaccard (overlap):  {len(core_intersection) / len(v02_core | v06_core):.4f}")

    # core_universe NEW IN top-20 (by core_score)
    print("\n🆕 v0.7.1 新進 top-20 (by core_score)")
    print(f"  {'stock_id':<10} {'name':<20} {'industry':<10} {'cscore':>7} {'theme':>6} {'FG':>6} {'IF':>6} {'VC':>6}")
    new_sorted = sorted(core_v06_only, key=lambda sid: -v06_info[sid][2])[:20]
    for sid in new_sorted:
        n, ind, cs, ts, fg, iif, vc = v06_info[sid]
        ind_s = (ind or "")[:8]
        n_s = (n or "")[:18]
        print(f"  {sid:<10} {n_s:<20} {ind_s:<10} {cs:>7.2f} {ts:>6.2f} {fg:>6.1f} {iif:>6.1f} {vc:>6.1f}")

    # core_universe DROPPED top-20 (by v0.2 baseline score lookup)
    print("\n❌ v0.2 被踢出 top-20 (依 stock_id sorted; original score 從 v0.2 DB)")
    print(f"  {'stock_id':<10} {'name':<20} {'industry':<10} {'now_v07_tier':<20}")
    dropped_sorted = sorted(core_v02_only)[:20]
    for sid in dropped_sorted:
        n, ind = v02_stock_info[sid]
        ind_s = (ind or "")[:8]
        n_s = (n or "")[:18]
        # find new tier
        new_tier = "—"
        if sid in v06_convex: new_tier = "convex_universe"
        elif sid in v06_quarantine: new_tier = "quarantine_universe"
        elif sid in v06_research: new_tier = "research_universe"
        else: new_tier = "MISSING(no candidate)"
        print(f"  {sid:<10} {n_s:<20} {ind_s:<10} {new_tier:<20}")

    # convex_universe diff
    print("\n" + "=" * 80)
    print("📊 CONVEX UNIVERSE (top-30) 差異")
    print("=" * 80)
    conv_intersection = v02_convex & v06_convex
    print(f"  Intersection: {len(conv_intersection)}/{len(v02_convex)} ({len(conv_intersection)/len(v02_convex)*100:.1f}%)")
    print(f"  v0.2 convex only: {len(v02_convex - v06_convex)}")
    print(f"  v0.7.1 convex only (新進): {len(v06_convex - v02_convex)}")
    print(f"  Jaccard:           {len(conv_intersection) / len(v02_convex | v06_convex):.4f}")

    # 1303 南亞 specific(§14.7-BH 揭露之 sign flip case)
    print("\n" + "=" * 80)
    print("📊 1303 南亞 specific check (§14.7-BH RMS vs STDDEV sign flip case)")
    print("=" * 80)
    nan_ya = "1303"
    nan_ya_v02_tier = "core_universe" if nan_ya in v02_core else "convex_universe" if nan_ya in v02_convex else "research/quarantine/missing"
    nan_ya_v06_tier = "core_universe" if nan_ya in v06_core else "convex_universe" if nan_ya in v06_convex else "research_universe" if nan_ya in v06_research else "quarantine"
    print(f"  v0.2 baseline tier: {nan_ya_v02_tier}")
    print(f"  v0.7.1 RMS tier:    {nan_ya_v06_tier}")
    if nan_ya in v06_info:
        n, ind, cs, ts, fg, iif, vc = v06_info[nan_ya]
        print(f"  v0.7.1 scores:      core={cs:.2f} / theme={ts:.2f} / FG={fg:.1f} / IF={iif:.1f} / VC={vc:.1f}")

    # 統計 quarantine_universe
    print("\n" + "=" * 80)
    print("📊 Quarantine reasons 統計 (top 10)")
    print("=" * 80)
    quar_reasons = {}
    for c in candidates:
        if c.core_tier == "quarantine_universe":
            reason = c.exclusion_reason or "unknown"
            quar_reasons[reason] = quar_reasons.get(reason, 0) + 1
    for reason, count in sorted(quar_reasons.items(), key=lambda x: -x[1])[:10]:
        print(f"  {count:>4}  {reason}")

    print("\n" + "=" * 80)
    print("✅ 對照完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
