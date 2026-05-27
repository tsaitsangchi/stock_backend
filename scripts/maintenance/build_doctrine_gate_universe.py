"""
build_doctrine_gate_universe.py v0.9 (§14.7-BV Phase C — Doctrine-Gate-First Universe Builder)
================================================================================
最後更新日期: 2026-05-26
主權狀態: IMPLEMENTED (憲法 v6.1.0 §14.7-BV Phase C 落地 / Path C: doctrine-gate first, score-rank for tier / 同次完成 §14.7-BU Phase E data layer hook coupling)
最高原則: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 📜 一、核心定義說明 (Core Definitions)
1. [Doctrine-Gate-First Authority]: 本工具落地 §14.7-BV Phase B 入憲之 Path C algorithm — 三基柱資料源依據為 selection 必要條件;N 完全動態;CoreScore 降為 gate-pass 內 tier ranking。
2. [Stage Pipeline]: 4 stages — (1) §0.3 K-wave market prerequisite 5/5 / (2) per-stock §0.1 5-source check + §0.2 by-def / (3) CoreScore composite (INFO display only per §14.7-BW) / (4) all doctrine-pass → core_universe(無 cap/floor/tier % per §14.7-BW pure doctrine)。
3. [Score Reuse]: 直接從既有 core_universe_scores 表讀 6 sub-scores(DQ/LM/FG/TR/IF/VC/RP)計算 composite,不重 compute(避免 v0.7 builder 完整跑 ~4-6h);此為 Path C 之 minimal-viable execution(builder full rewrite 為 v1.0 升版選項)。
4. [§14.7-BU Phase E Data Layer Coupling]: per §14.7-BV charter 之 execution coupling 紀律,本 builder 同次完成 §14.7-BU Phase E 之 data layer hook — 為新 snapshot 之每支 gate-pass stock 寫 universe_completeness_snapshot 之 data layer 3 records(per pillar)。
5. [Existing Scores Authority]: 複用 core_universe_scores 之 sub-scores 屬「v0.2-v0.7 score 仍為治權產物」之延續(per §14.7-BT precedent — algorithm 升版不撤銷 historical scoring data)。
6. [Zero Hardcoded N]: N 完全 doctrine-derived(per §14.7-BW pure doctrine 第二十一輪 + 2026-05-27 用戶 directive「排除所有固定的核心股數量」);無 N_min / N_max / tier % / 任何 fixed bound。
7. [Sovereignty Declaration]: 本工具屬 §14.7-BV Phase C 落地;不修改 §6.4 CoreScore 公式 / §6.7 SQL contract / §6.7.1 annex / §0.1-A / §0.2-A / §0.3-A 三套禁令 / §8.5 anti-leakage(由 builders 主管)。

## 📊 二、執行指令
| 場景 | 指令 |
| :--- | :--- |
| Dry-run(只顯示 gate-pass 數量 + tier split,不寫 DB) | `$ python scripts/maintenance/build_doctrine_gate_universe.py --dry-run` |
| Commit(寫 v0.9 snapshot + membership + §14.7-BU completeness data layer) | `$ python scripts/maintenance/build_doctrine_gate_universe.py --commit` |

## 📜 三、修訂歷程
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **v0.10 + §14.7-BX support** | 2026-05-26 | Codex | **§14.7-BX Phase C-3 + C-4 同次升版**:加 `--weekly-mode` CLI flag(per §14.7-BX 第二十二輪 canonical spec)— atomic supersede 前週 committed + `_weekly` suffix snapshot_id;**§6.7 SSOT 任一時點 ≤ 1 committed 不變式維持**。docstring 對齊 §14.7-BW(第二十一輪 Path D pure doctrine)+ §14.7-BX(第二十二輪 weekly recommit T-axis 純化)雙治權層;CoreScore 為 INFO display only(§14.7-BW 第 5 條 治權位階降階);N 完全 doctrine-derived 無 cap/floor/tier %(§14.7-BW Path D)。對既有 builder 邏輯影響:零(只新增 `weekly_mode` 參數;default False = legacy one-shot mode 不動);對既有 snapshot 影響:零(weekly_mode=True 觸發時才動既有 committed snapshot)。 | **ACTIVE** |
| **v0.10** | 2026-05-26 | Codex | **§14.7-BW Phase B → v0.9 doctrine_gate → v0.10 pure_doctrine 重構**:對映用戶 5 次明示之最強讀法「取消所有 200 支及 150 支」+「§14.7-BV Path C 內 4 hidden hardcode 揭露」。取消 N_MAX cap / N_MIN floor / 70-30 tier split / CoreScore as ranking gate(4 hidden hardcode 全 removed)。POLICY_VERSION → `core_universe_policy_v0.10_pure_doctrine`;Stage 3 + Stage 4 重寫:CoreScore INFO display / 全 doctrine-pass → core_universe(convex 棄用)。DB committed evidence:`core_universe_20260526_core_universe_policy_v0_10_pure_doctrine`(N=1862 / 0 convex / 941 research)。 | SUPERSEDED by v0.10+§14.7-BX |
| **v0.9** | 2026-05-26 | Codex | **§14.7-BV Phase C 落地首版 / minimal-viable doctrine-gate runner**:依憲章 v6.1.0-patch 第二十輪 §14.7-BV Phase B 入憲(charter L9462)+ Phase A 設計研究(`reports/doctrine_gate_selection_phase_a_research_20260526.md` 566 行 §14)落地。Path C 4-stage pipeline 完整(含 N_MAX=200 cap / 70-30 tier split / CoreScore ranking gate;後 §14.7-BW 揭露皆為 hidden hardcode)。複用 core_universe_scores 之 sub-scores 計算 composite。同次完成 §14.7-BU Phase E data layer hook。 | SUPERSEDED by v0.10 |
================================================================================
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from core.path_setup import ensure_scripts_on_path
ensure_scripts_on_path(__file__)

from core.db_utils import get_db_connection


CONSTITUTION_VER = "v6.1.0"
TOOL_VER = "v0.10_pure_doctrine"
POLICY_VERSION = "core_universe_policy_v0.10_pure_doctrine"
POLICY_VERSION_WEEKLY = "core_universe_policy_v0.10_pure_doctrine_weekly"  # §14.7-BX Phase C: weekly mode 獨立 policy(避免 UNIQUE constraint on (as_of_date, policy_version)衝突)

# CoreScore weights (per §6.4 / inherited from v0.7.1 builder) — kept for INFO display only, not for selection
WEIGHTS = {'DQ': 0.25, 'LM': 0.25, 'FG': 0.20, 'TR': 0.15, 'IF': 0.10, 'VC': 0.05}

# §14.7-BV v0.10 pure-doctrine: NO N bound, NO tier % split
# (per user 2026-05-26 第 5 次明示「取消所有 200 支及 150 支」之 doctrine 強化:
#  N_MAX / N_MIN / CORE_PCT 任何 fixed bound 皆為 hardcode 違反 doctrine)
# 既有 §6.7.1 annex 之 N_min=100 / N_max=200 / core_pct=0.70 在此 v0.10 不適用;
# 所有過 doctrine 之 stocks 皆進入 `core_universe`(不分 convex);
# N = doctrine-pass set 大小,無任何 trim / cap / floor / quantile threshold。

# §0.1 first-principle 5 raw sources
FP_SOURCES = [
    'TaiwanStockPriceAdj',
    'TaiwanStockFinancialStatements',
    'TaiwanStockMonthRevenue',
    'TaiwanStockInstitutionalInvestorsBuySell',
    'TaiwanStockMarginPurchaseShortSale',
]

# §0.3 K-wave 13 indicators 之 Path C 分層(per §14.7-BZ Phase F 2026-05-27)
# 拆 §0.3.1 K-wave / §0.3.2 Multi-cycle / §0.3.3 Microstructure 三 sub-pillars
# 對映 Kondratiev/Schumpeter/Mensch/Perez 學派之多週期 hierarchy

# §0.3.1 K-wave 純 macro structural(40-60 年 Kondratiev wave)— 7 indicators / avg 80%
KW_INDICATORS_PURE = [
    ('PATENTUSALLTOTAL', 'fred_series', 'series_id'),   # Tech: US Patents(85%)
    ('B985RC1Q027SBEA', 'fred_series', 'series_id'),    # Tech: IP Products Investment(80%)
    ('TCMDO', 'fred_series', 'series_id'),              # Credit: US Total Credit(75%)
    ('QUSPAM770A', 'fred_series', 'series_id'),         # Credit: BIS Credit-to-GDP(80%)
    ('LFWA64TTUSA647N', 'fred_series', 'series_id'),    # Demographics: Working-age %(85%)
    ('SPPOPDPNDOLUSA', 'fred_series', 'series_id'),     # Demographics: Old-age dependency(80%)
    ('PALLFNFINDEXQ', 'fred_series', 'series_id'),      # Commodity: CRB Index(75%)
]

# §0.3.2 Multi-cycle Context(7-25 年 Juglar + Kuznets + Kitchin K5 edge)— 5 indicators / avg 49%
MC_INDICATORS = [
    ('M2SL', 'fred_series', 'series_id'),               # Monetary regime(70%)
    ('WTISPLC', 'fred_series', 'series_id'),            # Energy/Commodity Juglar(70%)
    ('T10Y2Y', 'fred_series', 'series_id'),             # Yield curve Juglar(30%)
    ('TW_SEMI_VWAP_YOY', 'kwave_supply_cycle_proxy', 'proxy_id'),    # Sector Kitchin/K5(40%)
    ('TW_SHIPPING_VWAP_YOY', 'kwave_supply_cycle_proxy', 'proxy_id'),# Juglar trade(35%)
]

# §0.3.3 Microstructure(月 ~ 季 sentiment / regime)— 1 indicator / avg 10%
MS_INDICATORS = [
    ('VIXCLS', 'fred_series', 'series_id'),             # Volatility regime(10%)
]

# Backward compat:KW_INDICATORS = 全 13 indicators(per §14.7-BY 之 indicator-axis 純化)
# 但 Stage 1 現在拆 3 sub-stages,KW_INDICATORS 主要供 documentation 與 audit reference
KW_INDICATORS = KW_INDICATORS_PURE + MC_INDICATORS + MS_INDICATORS
KW_INDICATOR_COUNT = len(KW_INDICATORS)  # = 13 per §14.7-BZ Phase F(7 + 5 + 1)
KW_PURE_COUNT = len(KW_INDICATORS_PURE)  # = 7
MC_COUNT = len(MC_INDICATORS)            # = 5
MS_COUNT = len(MS_INDICATORS)            # = 1


def check_indicators(cur, indicator_list):
    """Helper: check binary presence(rows > 0)of indicator list."""
    present = []
    for name, table, col in indicator_list:
        cur.execute(f"SELECT to_regclass('public.\"{table}\"')")
        if cur.fetchone()[0] is None:
            continue
        cur.execute(f'SELECT COUNT(*) FROM "{table}" WHERE {col}=%s', (name,))
        if cur.fetchone()[0] > 0:
            present.append(name)
    return present


def check_kwave_market_context(cur):
    """Stage 1: §0.3 K-wave market-level prerequisite(backward-compat / 全 13 indicators)."""
    return check_indicators(cur, KW_INDICATORS)


def check_macro_pillars(cur):
    """Stage 1A/1B/1C(§14.7-BZ Phase F):三 sub-pillar binary gate.

    Returns:
        (kw_pure_present, mc_present, ms_present) — 三 sub-pillar 各自之 present indicator list
    """
    kw_pure = check_indicators(cur, KW_INDICATORS_PURE)
    mc = check_indicators(cur, MC_INDICATORS)
    ms = check_indicators(cur, MS_INDICATORS)
    return kw_pure, mc, ms


def get_doctrine_gate_pass(cur):
    """Stage 2: per-stock §0.1 5/5 source existence (§0.2 by-def via candidate set)."""
    cur.execute("""
        WITH src AS (
            SELECT i.stock_id, i.stock_name, i.type, i.industry_category,
                (p.stock_id IS NOT NULL)::int + (fs.stock_id IS NOT NULL)::int +
                (mr.stock_id IS NOT NULL)::int + (ii.stock_id IS NOT NULL)::int +
                (mm.stock_id IS NOT NULL)::int AS coverage
            FROM "TaiwanStockInfo" i
            LEFT JOIN (SELECT DISTINCT stock_id FROM "TaiwanStockPriceAdj") p ON p.stock_id=i.stock_id
            LEFT JOIN (SELECT DISTINCT stock_id FROM "TaiwanStockFinancialStatements") fs ON fs.stock_id=i.stock_id
            LEFT JOIN (SELECT DISTINCT stock_id FROM "TaiwanStockMonthRevenue") mr ON mr.stock_id=i.stock_id
            LEFT JOIN (SELECT DISTINCT stock_id FROM "TaiwanStockInstitutionalInvestorsBuySell") ii ON ii.stock_id=i.stock_id
            LEFT JOIN (SELECT DISTINCT stock_id FROM "TaiwanStockMarginPurchaseShortSale") mm ON mm.stock_id=i.stock_id
        )
        SELECT stock_id, stock_name, type, industry_category
        FROM src
        WHERE coverage = 5
        ORDER BY stock_id
    """)
    return cur.fetchall()


def compute_composite_score(cur, stock_id):
    """Stage 3 helper: composite = sum(w_i * sub_i) - risk_penalty;use most recent score row."""
    cur.execute("""
        SELECT data_quality_score, liquidity_score, fundamental_score,
               theme_score, institutional_flow_score, volatility_control_score,
               risk_penalty
        FROM core_universe_scores s
        JOIN core_universe_snapshot ss ON s.snapshot_id = ss.snapshot_id
        WHERE s.stock_id = %s
        ORDER BY ss.as_of_date DESC NULLS LAST, ss.created_at DESC NULLS LAST
        LIMIT 1
    """, (stock_id,))
    row = cur.fetchone()
    if not row:
        return None, None
    dq, lm, fg, tr, if_, vc, rp = [float(x) if x is not None else 0.0 for x in row]
    composite = (WEIGHTS['DQ'] * dq + WEIGHTS['LM'] * lm + WEIGHTS['FG'] * fg +
                 WEIGHTS['TR'] * tr + WEIGHTS['IF'] * if_ + WEIGHTS['VC'] * vc - rp)
    return composite, {'DQ': dq, 'LM': lm, 'FG': fg, 'TR': tr, 'IF': if_, 'VC': vc, 'RP': rp}


def write_snapshot_and_membership(conn, cur, scored_results, n_total, n_core, n_convex, as_of, all_candidates, weekly_mode=False):
    """Stage Commit: write v0.10 snapshot + membership + universe_completeness_snapshot data layer.

    If weekly_mode=True(per §14.7-BX Phase C):
      (a) Atomically supersede any current committed snapshot (status committed → superseded)
      (b) 使用獨立 policy_version `POLICY_VERSION_WEEKLY`(避免與 legacy v0.10 同日衝突 UNIQUE constraint)
      (c) Add `_weekly` suffix to snapshot_id
      (d) Preserves §6.7 SSOT invariant(at most 1 committed at any time)
    """
    active_policy = POLICY_VERSION_WEEKLY if weekly_mode else POLICY_VERSION
    suffix = "_weekly" if weekly_mode else ""
    snapshot_id = f"core_universe_{as_of.replace('-', '')}_{active_policy.replace('.', '_')}"

    # §14.7-BX Phase C atomic supersede: in weekly mode, mark any currently-committed snapshot as superseded
    if weekly_mode:
        cur.execute("""
            UPDATE core_universe_snapshot
            SET status='superseded',
                notes = COALESCE(notes, '') || ' | SUPERSEDED ' || %s::date || ' by new weekly recommit per §14.7-BX'
            WHERE status='committed'
        """, (as_of,))
        if cur.rowcount > 0:
            print(f"  ⤴️  Superseded {cur.rowcount} prior committed snapshot(s) per §14.7-BX weekly mode")

    # 1. Register policy(weekly_mode 用 POLICY_VERSION_WEEKLY)
    policy_name = ('Core Universe Policy v0.10 Pure Doctrine WEEKLY(§14.7-BX)' if weekly_mode
                   else 'Core Universe Policy v0.10 Pure Doctrine')
    policy_desc = ('§14.7-BX Phase C weekly recommit: doctrine-gate-first selection + atomic supersede prior committed. '
                   'Per-stock §0.1 5/5 raw source + §0.2 by-def + §0.3 K-wave market 5/5 prerequisite. '
                   'N dynamic per doctrine; no cap/floor/tier %; weekly auto-recommit per §14.7-BX 第二十二輪.'
                   if weekly_mode else
                   '§14.7-BV/BW Phase C: doctrine-gate-first one-shot. Per-stock §0.1 5/5 + §0.2 by-def + §0.3 market 5/5. N dynamic.')
    policy_notes = ('§14.7-BX Phase C-3 weekly mode 落地' if weekly_mode
                    else '§14.7-BW Phase B v0.10 pure doctrine 落地')
    cur.execute("""
        INSERT INTO core_universe_policy (
            policy_version, policy_name, description,
            weight_config, eligibility_config, risk_config,
            effective_from, active, notes
        )
        SELECT %s, %s, %s,
            weight_config, eligibility_config, risk_config,
            %s::date, TRUE, %s
        FROM core_universe_policy WHERE policy_version='core_universe_policy_v0.2'
        ON CONFLICT (policy_version) DO UPDATE SET active=TRUE, updated_at=NOW()
    """, (active_policy, policy_name, policy_desc, as_of, policy_notes))

    # 2. Snapshot
    snap_note = (f'§14.7-BX Phase C weekly recommit:N={n_total} doctrine-pass(§0.1 5/5 + §0.2 + §0.3 market 5/5);atomic-supersede prior committed;policy={active_policy}'
                 if weekly_mode else
                 f'§14.7-BW Phase B v0.10 pure doctrine one-shot:N={n_total} doctrine-pass;{n_core} core + {n_convex} convex;policy={active_policy}')
    cur.execute("""
        INSERT INTO core_universe_snapshot (
            snapshot_id, as_of_date, source_data_cutoff, policy_version,
            total_candidates, research_count, core_count, convex_count, quarantine_count,
            status, notes
        ) VALUES (%s, %s::date, %s::date, %s, %s, %s, %s, %s, %s, 'committed', %s)
        ON CONFLICT (snapshot_id) DO UPDATE SET
            status='committed',
            total_candidates=EXCLUDED.total_candidates,
            research_count=EXCLUDED.research_count,
            core_count=EXCLUDED.core_count,
            convex_count=EXCLUDED.convex_count,
            notes=EXCLUDED.notes
    """, (
        snapshot_id, as_of, as_of, active_policy,
        len(all_candidates), len(all_candidates) - n_total, n_core, n_convex, 0,
        snap_note,
    ))

    # 3. Membership rows
    gate_pass_ids = {s['stock']['stock_id'] for s in scored_results}
    for idx, item in enumerate(scored_results):
        stock = item['stock']
        score = item['score']
        score_detail = item['score_detail']
        tier = 'core_universe' if idx < n_core else 'convex_universe'
        cur.execute("""
            INSERT INTO core_universe_membership (
                snapshot_id, stock_id, stock_name, type, industry_category,
                core_tier, core_score, selected_at, effective_from, review_cycle,
                active, selection_reason,
                train_eligible, predict_eligible, backtest_eligible, downstream_ready,
                label_horizon, policy_version
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, NOW(), %s::date, 'annual',
                TRUE, %s,
                TRUE, TRUE, TRUE, TRUE,
                20, %s
            )
            ON CONFLICT DO NOTHING
        """, (
            snapshot_id, stock['stock_id'], stock['stock_name'], stock['type'], stock['industry_category'],
            tier, score, as_of,
            f"doctrine_gate pass (5/5 §0.1 + §0.2 by-def + §0.3 5/5 market context); composite_score={score:.2f}; rank={idx+1}/{n_total}",
            active_policy,
        ))

    # 4. Add non-gate-pass candidates as research_universe
    for cand in all_candidates:
        if cand['stock_id'] in gate_pass_ids:
            continue
        cur.execute("""
            INSERT INTO core_universe_membership (
                snapshot_id, stock_id, stock_name, type, industry_category,
                core_tier, selected_at, effective_from, review_cycle,
                active, exclusion_reason,
                train_eligible, predict_eligible, backtest_eligible, downstream_ready,
                label_horizon, policy_version
            ) VALUES (
                %s, %s, %s, %s, %s,
                'research_universe', NOW(), %s::date, 'annual',
                TRUE, %s,
                FALSE, FALSE, FALSE, FALSE,
                20, %s
            )
            ON CONFLICT DO NOTHING
        """, (
            snapshot_id, cand['stock_id'], cand['stock_name'], cand['type'], cand['industry_category'],
            as_of,
            "doctrine_gate not pass (<5/5 §0.1 raw source coverage per §14.7-BV Path C)",
            active_policy,
        ))

    # 5. §14.7-BU Phase E data-layer hook: write per-stock × per-pillar completeness records
    completeness_snapshot_id = f"completeness_{as_of.replace('-', '')}_{active_policy.replace('.', '_')}_data_layer"
    for item in scored_results:
        stock_id = item['stock']['stock_id']
        # §0.1 first_principle data layer: 5/5 expected/actual
        cur.execute("""
            INSERT INTO universe_completeness_snapshot
                (snapshot_id, universe_snapshot_id, as_of_date, stock_id, pillar, layer,
                 expected_items, actual_items, completeness_pct, evidence_source_table)
            VALUES (%s, %s, %s::date, %s, 'first_principle', 'data', 5, 5, 100.00, %s)
            ON CONFLICT (snapshot_id, stock_id, pillar, layer) DO UPDATE SET
                expected_items=EXCLUDED.expected_items,
                actual_items=EXCLUDED.actual_items,
                completeness_pct=EXCLUDED.completeness_pct,
                evidence_source_table=EXCLUDED.evidence_source_table
        """, (completeness_snapshot_id, snapshot_id, as_of, stock_id, ','.join(FP_SOURCES)))
        # §0.2 pareto data layer: 1/1 (in core/convex tier by definition)
        cur.execute("""
            INSERT INTO universe_completeness_snapshot
                (snapshot_id, universe_snapshot_id, as_of_date, stock_id, pillar, layer,
                 expected_items, actual_items, completeness_pct, evidence_source_table)
            VALUES (%s, %s, %s::date, %s, 'pareto', 'data', 1, 1, 100.00, 'core_universe_membership')
            ON CONFLICT (snapshot_id, stock_id, pillar, layer) DO UPDATE SET
                expected_items=EXCLUDED.expected_items,
                actual_items=EXCLUDED.actual_items,
                completeness_pct=EXCLUDED.completeness_pct,
                evidence_source_table=EXCLUDED.evidence_source_table
        """, (completeness_snapshot_id, snapshot_id, as_of, stock_id))
        # §0.3 拆 3 sub-pillars(§14.7-BZ Phase F Path C):market-level broadcast
        # §0.3.1 kondratiev_kwave: 7/7
        cur.execute("""
            INSERT INTO universe_completeness_snapshot
                (snapshot_id, universe_snapshot_id, as_of_date, stock_id, pillar, layer,
                 expected_items, actual_items, completeness_pct, evidence_source_table)
            VALUES (%s, %s, %s::date, %s, 'kondratiev_kwave', 'data', %s, %s, 100.00, 'fred_series')
            ON CONFLICT (snapshot_id, stock_id, pillar, layer) DO UPDATE SET
                expected_items=EXCLUDED.expected_items,
                actual_items=EXCLUDED.actual_items,
                completeness_pct=EXCLUDED.completeness_pct,
                evidence_source_table=EXCLUDED.evidence_source_table
        """, (completeness_snapshot_id, snapshot_id, as_of, stock_id, KW_PURE_COUNT, KW_PURE_COUNT))
        # §0.3.2 kondratiev_multicycle: 5/5
        cur.execute("""
            INSERT INTO universe_completeness_snapshot
                (snapshot_id, universe_snapshot_id, as_of_date, stock_id, pillar, layer,
                 expected_items, actual_items, completeness_pct, evidence_source_table)
            VALUES (%s, %s, %s::date, %s, 'kondratiev_multicycle', 'data', %s, %s, 100.00, 'fred_series,kwave_supply_cycle_proxy')
            ON CONFLICT (snapshot_id, stock_id, pillar, layer) DO UPDATE SET
                expected_items=EXCLUDED.expected_items,
                actual_items=EXCLUDED.actual_items,
                completeness_pct=EXCLUDED.completeness_pct,
                evidence_source_table=EXCLUDED.evidence_source_table
        """, (completeness_snapshot_id, snapshot_id, as_of, stock_id, MC_COUNT, MC_COUNT))
        # §0.3.3 kondratiev_microstructure: 1/1
        cur.execute("""
            INSERT INTO universe_completeness_snapshot
                (snapshot_id, universe_snapshot_id, as_of_date, stock_id, pillar, layer,
                 expected_items, actual_items, completeness_pct, evidence_source_table)
            VALUES (%s, %s, %s::date, %s, 'kondratiev_microstructure', 'data', %s, %s, 100.00, 'fred_series')
            ON CONFLICT (snapshot_id, stock_id, pillar, layer) DO UPDATE SET
                expected_items=EXCLUDED.expected_items,
                actual_items=EXCLUDED.actual_items,
                completeness_pct=EXCLUDED.completeness_pct,
                evidence_source_table=EXCLUDED.evidence_source_table
        """, (completeness_snapshot_id, snapshot_id, as_of, stock_id, MS_COUNT, MS_COUNT))

    conn.commit()
    return snapshot_id, completeness_snapshot_id


def build(commit=False, weekly_mode=False):
    conn = get_db_connection()
    cur = conn.cursor()
    as_of = datetime.now().strftime('%Y-%m-%d')

    print("=" * 72)
    print(f"§14.7-BV Phase C — Doctrine-Gate Universe Builder ({TOOL_VER})")
    print("=" * 72)

    # Stage 1 三 sub-stages(§14.7-BZ Phase F Path C 分層):1A K-wave / 1B Multi-cycle / 1C Microstructure
    kw_pure_present, mc_present, ms_present = check_macro_pillars(cur)

    print(f"\n[Stage 1A] §0.3.1 K-wave pure(40-60 年): {len(kw_pure_present)}/{KW_PURE_COUNT}")
    for n in kw_pure_present: print(f"    ✅ {n}")
    if len(kw_pure_present) < KW_PURE_COUNT:
        missing = [n for n, _, _ in KW_INDICATORS_PURE if n not in kw_pure_present]
        print(f"\n❌ Stage 1A FAIL: §0.3.1 K-wave({len(kw_pure_present)}/{KW_PURE_COUNT});missing: {missing}")
        conn.close()
        return False
    print(f"  ✅ Stage 1A PASS")

    print(f"\n[Stage 1B] §0.3.2 Multi-cycle(7-25 年): {len(mc_present)}/{MC_COUNT}")
    for n in mc_present: print(f"    ✅ {n}")
    if len(mc_present) < MC_COUNT:
        missing = [n for n, _, _ in MC_INDICATORS if n not in mc_present]
        print(f"\n❌ Stage 1B FAIL: §0.3.2 Multi-cycle({len(mc_present)}/{MC_COUNT});missing: {missing}")
        conn.close()
        return False
    print(f"  ✅ Stage 1B PASS")

    print(f"\n[Stage 1C] §0.3.3 Microstructure(月 ~ 季): {len(ms_present)}/{MS_COUNT}")
    for n in ms_present: print(f"    ✅ {n}")
    if len(ms_present) < MS_COUNT:
        missing = [n for n, _, _ in MS_INDICATORS if n not in ms_present]
        print(f"\n❌ Stage 1C FAIL: §0.3.3 Microstructure({len(ms_present)}/{MS_COUNT});missing: {missing}")
        conn.close()
        return False
    print(f"  ✅ Stage 1C PASS")

    # Backward-compat alias:kw_present = 全 13 indicators
    kw_present = kw_pure_present + mc_present + ms_present

    # Stage 2: per-stock §0.1 + §0.2 gate
    gate_pass_rows = get_doctrine_gate_pass(cur)
    print(f"\n[Stage 2] per-stock §0.1 5/5 + §0.2 by-def gate:")
    print(f"  Gate-pass set: {len(gate_pass_rows)} stocks")

    # Load all candidates for non-pass research_universe
    cur.execute('SELECT stock_id, stock_name, type, industry_category FROM "TaiwanStockInfo" ORDER BY stock_id')
    all_candidates = [
        {'stock_id': r[0], 'stock_name': r[1], 'type': r[2], 'industry_category': r[3]}
        for r in cur.fetchall()
    ]
    print(f"  Non-gate candidates → research_universe: {len(all_candidates) - len(gate_pass_rows)}")

    # v0.10 pure doctrine: NO N_MIN gate (would be implicit floor hardcode)
    # — if gate-pass is small, downstream handles per its own logic

    # Stage 3 (informational only): composite CoreScore for INFO display;does NOT affect selection
    print(f"\n[Stage 3] Composite CoreScore (INFO-only — NOT used for selection)")
    scored = []
    missing_score_count = 0
    for r in gate_pass_rows:
        stock = {'stock_id': r[0], 'stock_name': r[1], 'type': r[2], 'industry_category': r[3]}
        composite, score_detail = compute_composite_score(cur, stock['stock_id'])
        if composite is None:
            missing_score_count += 1
            composite, score_detail = 0.0, {'note': 'no historical score'}
        scored.append({'stock': stock, 'score': composite, 'score_detail': score_detail})
    if missing_score_count > 0:
        print(f"  ⚠️ {missing_score_count} stocks with no historical score (score=0;不影響選擇,僅 INFO 缺值)")
    # Sort by score ONLY for INFO display ordering (not for cap / tier split)
    scored.sort(key=lambda x: x['score'], reverse=True)
    print(f"  Top 5 by composite (INFO):")
    for s in scored[:5]:
        print(f"    {s['stock']['stock_id']:6} {s['stock']['stock_name']:<14} score={s['score']:.2f} industry={s['stock']['industry_category']}")
    print(f"  Bottom 5 by composite (INFO):")
    for s in scored[-5:]:
        print(f"    {s['stock']['stock_id']:6} {s['stock']['stock_name']:<14} score={s['score']:.2f} industry={s['stock']['industry_category']}")

    # Stage 4 v0.10: NO N_MAX cap, NO tier % split — all doctrine-pass → core_universe
    n_total = len(scored)
    n_core = n_total  # all pass = core
    n_convex = 0  # no convex tier in pure doctrine
    print(f"\n[Stage 4] Pure doctrine assignment(per §14.7-BV v0.10):")
    print(f"  All {n_total} doctrine-pass stocks → core_universe")
    print(f"  convex_universe = 0(tier 概念在 v0.10 不適用;待 portfolio_sizer 邊界另立)")
    print(f"  N 完全 doctrine-derived,無 cap / floor / tier %")

    if not commit:
        print("\n--- DRY-RUN ONLY (use --commit to write) ---")
        conn.close()
        return True

    # Commit
    mode_label = "WEEKLY(§14.7-BX atomic supersede + _weekly suffix)" if weekly_mode else "ONE-SHOT(legacy mode)"
    print(f"\n[Commit] Mode: {mode_label}")
    print(f"        Writing v0.10 snapshot + membership + universe_completeness_snapshot data layer...")
    snapshot_id, completeness_id = write_snapshot_and_membership(
        conn, cur, scored, n_total, n_core, n_convex, as_of, all_candidates, weekly_mode=weekly_mode
    )
    print(f"  ✅ snapshot: {snapshot_id}")
    print(f"  ✅ membership: {len(all_candidates)} rows ({n_total} gate-pass + {len(all_candidates)-n_total} research)")
    print(f"  ✅ universe_completeness_snapshot: {3*n_total} rows (3 pillars × {n_total} stocks @ data layer)")
    print(f"     completeness_snapshot_id: {completeness_id}")

    # Verify
    print(f"\n[Verify]")
    cur.execute("SELECT status FROM core_universe_snapshot WHERE snapshot_id=%s", (snapshot_id,))
    print(f"  snapshot status: {cur.fetchone()[0]}")
    cur.execute("SELECT core_tier, COUNT(*) FROM core_universe_membership WHERE snapshot_id=%s GROUP BY core_tier ORDER BY COUNT(*) DESC", (snapshot_id,))
    for r in cur.fetchall():
        print(f"  tier {r[0]}: {r[1]}")
    cur.execute("SELECT pillar, layer, COUNT(*) FROM universe_completeness_snapshot WHERE snapshot_id=%s GROUP BY pillar, layer ORDER BY pillar", (completeness_id,))
    print(f"  universe_completeness records:")
    for r in cur.fetchall():
        print(f"    {r[0]} × {r[1]}: {r[2]}")

    conn.close()
    print(f"\n🎯 §14.7-BV Phase C COMPLETE — doctrine-gate-first universe committed (N={n_total} dynamic)")
    return True


def main():
    parser = argparse.ArgumentParser(description=f"Doctrine-Gate Universe Builder ({TOOL_VER})")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Show gate-pass + tier split without writing DB")
    mode.add_argument("--commit", action="store_true", help="Write new v0.10 snapshot + membership + completeness data layer")
    parser.add_argument("--weekly-mode", action="store_true",
                        help="§14.7-BX Phase C weekly mode:atomic-supersede prior committed + add _weekly suffix to snapshot_id")
    args = parser.parse_args()
    ok = build(commit=args.commit, weekly_mode=args.weekly_mode)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
