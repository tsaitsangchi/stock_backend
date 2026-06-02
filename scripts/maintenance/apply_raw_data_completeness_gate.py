"""
apply_raw_data_completeness_gate.py — §14.7-CD Raw Data Completeness Gate
================================================================================
最後更新日期: 2026-05-27
主權狀態: DEPRECATED (§14.7-CG v6.5.0 native gate 整合;邏輯已併入 scripts/core/core_universe_builder.py `DoctrineNativeGateBuilder` Stage 2+3;本 script 保留為歷史 audit trail)
最高原則: 「全資料來源須從 FinMind / FRED API 直接抓取且每股 source 完整到位」

## ⚠️ §14.7-CG 取代備註 (2026-05-27)

- **新 SSOT**: `scripts/core/core_universe_builder.py --mode doctrine-native --commit`(§14.7-CG native integration;包含 §14.7-CD raw gate + §14.7-CC source authority + §0.3 K-wave macro)
- **本 script 狀態**: 邏輯 100% 移植到 `DoctrineNativeGateBuilder.P1_THRESHOLDS + P2_THRESHOLDS`;calling site 已移除
- **歷史用途**: v0.12 N=1,543 snapshot 為本 script 之 historical evidence(已 superseded by v0.13)
- **下架時點**: 預計 v6.5.x 後完全移除(per §14.7-CG Phase E migration)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:套用 §14.7-CD Raw Data Completeness Gate:檢查每股原始資料是否完整,不齊者排除。

**輸入 → 輸出**:raw 資料 + universe → 過 gate 之 snapshot

**為什麼需要它**:確保核心股的 raw 資料完整(模型前置)。

## 一、核心定義說明
- [Raw Data Completeness Gate]: 個股 raw API 資料任何 source 缺漏 → 排除核心股
- [Source-Level Audit]: 在 RAW table layer 直接 check(stricter than feature-level gate)
- [9 Required Sources]: PriceAdj(252d)/ PER(latest)/ MonthRevenue(12m)/
  FinStmt 3 types(4Q each)/ BalanceSheet 2 types(2Q+)/ Institutional(60d)/
  Margin(60d)/ Info(latest)

## 二、CLI 範例
    python scripts/maintenance/apply_raw_data_completeness_gate.py --dry-run
    python scripts/maintenance/apply_raw_data_completeness_gate.py --commit
## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.2 | 2026-06-02 | Codex | §一.11 標頭三段式 + 白話段補正;原邏輯不變。 | **ACTIVE** |

## 📊 二、全量維運指令總矩陣 (Operational Matrix)

| 指令 / 模式 | 行為 | 治權對應 |
| :--- | :--- | :--- |
| --dry-run(預設) | 只稽核 | §14.7-CD |
| --commit | 套 gate + 寫 DB | §14.7-CD |

"""
from __future__ import annotations

import sys
import argparse
import logging
import json
from datetime import date, timedelta
from pathlib import Path

_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

from core.db_utils import get_db_conn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

# Raw data requirements per source(per §14.7-CD)
THRESHOLDS = {
    "price_252d": 200,        # PriceAdj 過去 365 天 >= 200 trading days
    "per_recent": 1,           # PER recent ≥ 1 row(2026)且 PER/PBR/dividend_yield 三欄非空
    "monthrev_12m": 12,        # MonthRevenue 過去 18 個月 >= 12 row
    "finstmt_rev_4q": 4,       # Revenue 4 quarter
    "finstmt_op_4q": 4,        # OperatingIncome 4 quarter
    "finstmt_iat_4q": 4,       # IncomeAfterTaxes 4 quarter
    "bs_ta_2q": 2,             # TotalAssets 2 quarter(for YoY)
    "bs_eq_1q": 1,             # EquityAttributableToOwnersOfParent 1 quarter
    "inst_60d": 40,            # Institutional 60 天 >= 40 row
    "margin_60d": 40,          # Margin 60 天 >= 40 row
    "info_1": 1,               # Info industry_category >= 1 row
}


def run_audit(cur, today):
    """One-pass aggregation per source — fast."""
    d_365 = today - timedelta(days=365)
    d_18m = today - timedelta(days=18 * 30)
    d_24m = today - timedelta(days=24 * 30)
    d_60d = today - timedelta(days=90)
    d_year = today.replace(month=1, day=1)

    # PriceAdj
    cur.execute(
        'SELECT stock_id, COUNT(*) FROM "TaiwanStockPriceAdj" WHERE date >= %s GROUP BY stock_id',
        (d_365,),
    )
    price_count = dict(cur.fetchall())

    # PER recent(2026)
    cur.execute(
        '''SELECT stock_id, COUNT(*) FROM "TaiwanStockPER"
           WHERE date >= %s AND "PER" IS NOT NULL AND "PBR" IS NOT NULL AND "dividend_yield" IS NOT NULL
           GROUP BY stock_id''',
        (d_year,),
    )
    per_count = dict(cur.fetchall())

    # MonthRevenue 18m
    cur.execute(
        'SELECT stock_id, COUNT(*) FROM "TaiwanStockMonthRevenue" WHERE date >= %s GROUP BY stock_id',
        (d_18m,),
    )
    monthrev_count = dict(cur.fetchall())

    # FinStmt by type
    cur.execute(
        '''SELECT stock_id, type, COUNT(DISTINCT date) FROM "TaiwanStockFinancialStatements"
           WHERE date >= %s AND type IN ('Revenue','OperatingIncome','IncomeAfterTaxes')
           GROUP BY stock_id, type''',
        (d_24m,),
    )
    finstmt_rev, finstmt_op, finstmt_iat = {}, {}, {}
    for sid, ttype, n in cur.fetchall():
        if ttype == 'Revenue':
            finstmt_rev[sid] = n
        elif ttype == 'OperatingIncome':
            finstmt_op[sid] = n
        elif ttype == 'IncomeAfterTaxes':
            finstmt_iat[sid] = n

    # BalanceSheet by type
    cur.execute(
        '''SELECT stock_id, type, COUNT(DISTINCT date) FROM "TaiwanStockBalanceSheet"
           WHERE date >= %s AND type IN ('TotalAssets','EquityAttributableToOwnersOfParent')
           GROUP BY stock_id, type''',
        (d_24m,),
    )
    bs_ta, bs_eq = {}, {}
    for sid, ttype, n in cur.fetchall():
        if ttype == 'TotalAssets':
            bs_ta[sid] = n
        elif ttype == 'EquityAttributableToOwnersOfParent':
            bs_eq[sid] = n

    # Institutional
    cur.execute(
        'SELECT stock_id, COUNT(DISTINCT date) FROM "TaiwanStockInstitutionalInvestorsBuySell" WHERE date >= %s GROUP BY stock_id',
        (d_60d,),
    )
    inst_count = dict(cur.fetchall())

    # Margin
    cur.execute(
        'SELECT stock_id, COUNT(DISTINCT date) FROM "TaiwanStockMarginPurchaseShortSale" WHERE date >= %s GROUP BY stock_id',
        (d_60d,),
    )
    margin_count = dict(cur.fetchall())

    # Info
    cur.execute(
        'SELECT stock_id, COUNT(*) FROM "TaiwanStockInfo" WHERE industry_category IS NOT NULL GROUP BY stock_id',
    )
    info_count = dict(cur.fetchall())

    return {
        "price_252d": price_count,
        "per_recent": per_count,
        "monthrev_12m": monthrev_count,
        "finstmt_rev_4q": finstmt_rev,
        "finstmt_op_4q": finstmt_op,
        "finstmt_iat_4q": finstmt_iat,
        "bs_ta_2q": bs_ta,
        "bs_eq_1q": bs_eq,
        "inst_60d": inst_count,
        "margin_60d": margin_count,
        "info_1": info_count,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--commit", action="store_true")
    p.add_argument("--dry-run", action="store_true", default=True)
    args = p.parse_args()
    if args.commit:
        args.dry_run = False

    mode = "COMMIT" if not args.dry_run else "DRY-RUN"
    logger.info(f"=== §14.7-CD Raw Data Completeness Gate / mode={mode} ===")

    conn = get_db_conn()
    try:
        cur = conn.cursor()

        # 1. Get current snapshot
        cur.execute(
            """SELECT snapshot_id FROM core_universe_snapshot
               WHERE status='committed' ORDER BY as_of_date DESC LIMIT 1"""
        )
        cur_snap = cur.fetchone()[0]
        logger.info(f"  Current snapshot: {cur_snap}")

        cur.execute(
            'SELECT stock_id FROM core_universe_membership WHERE snapshot_id = %s AND core_tier=%s',
            (cur_snap, 'core_universe'),
        )
        current_universe = sorted([r[0] for r in cur.fetchall()])
        logger.info(f"  Current universe: {len(current_universe)} stocks")

        # 2. Run audit
        today = date.today()
        logger.info(f"  Running raw data audit(audit date: {today})...")
        counts = run_audit(cur, today)
        logger.info(f"  Audit complete")

        # 3. Apply gate per stock
        qualified = []
        rejected = {}  # sid -> [reasons]
        for sid in current_universe:
            reasons = []
            for source, threshold in THRESHOLDS.items():
                n = counts[source].get(sid, 0)
                if n < threshold:
                    reasons.append(f"{source}={n}<{threshold}")
            if reasons:
                rejected[sid] = reasons
            else:
                qualified.append(sid)

        # 4. Report
        logger.info(f"\n📊 §14.7-CD Audit Result:")
        logger.info(f"  Current universe : {len(current_universe):4d}")
        logger.info(f"  ✅ QUALIFIED     : {len(qualified):4d} ({100.0*len(qualified)/len(current_universe):.1f}%)")
        logger.info(f"  ❌ REJECTED      : {len(rejected):4d}")

        # Reject reason histogram
        reason_hist = {}
        for sid, reasons in rejected.items():
            for r in reasons:
                src = r.split('=')[0]
                reason_hist[src] = reason_hist.get(src, 0) + 1
        if reason_hist:
            logger.info(f"\n📋 Top rejection reasons:")
            for src, n in sorted(reason_hist.items(), key=lambda x: -x[1]):
                logger.info(f"  {src:20s}: {n:3d} stocks fail")

        if args.dry_run:
            sample = list(rejected.items())[:5]
            logger.info(f"\n[DRY-RUN] sample rejected:")
            for sid, reasons in sample:
                logger.info(f"  {sid}: {', '.join(reasons[:3])}")
            return

        # 5. COMMIT mode
        today_str = today.strftime("%Y%m%d")
        new_policy = "core_universe_policy_v0.12_raw_data_completeness_gate"
        new_snap = f"core_universe_{today_str}_{new_policy.replace('.', '_')}"

        logger.info(f"\n📝 Creating new snapshot: {new_snap}")

        cur.execute(
            """INSERT INTO core_universe_policy (policy_version, policy_name, description, active, effective_from)
               VALUES (%s, %s, %s, TRUE, CURRENT_DATE)
               ON CONFLICT (policy_version) DO NOTHING""",
            (new_policy, "Raw Data Completeness Gate v6.4.2",
             "§14.7-CD:enforce raw API source data completeness per stock"),
        )

        cur.execute(
            """INSERT INTO core_universe_snapshot
                (snapshot_id, as_of_date, source_data_cutoff, policy_version,
                 total_candidates, core_count, status, notes, created_at)
               VALUES (%s, CURRENT_DATE, CURRENT_DATE, %s, %s, %s, 'committed',
                 %s, NOW())""",
            (new_snap, new_policy, len(current_universe), len(qualified),
             f"§14.7-CD Raw Data Completeness Gate;superseded {cur_snap}"),
        )

        cur.executemany(
            """INSERT INTO core_universe_membership
                (snapshot_id, stock_id, core_tier, active, selected_at, selection_reason)
               VALUES (%s, %s, 'core_universe', TRUE, NOW(), %s)""",
            [(new_snap, sid, "§14.7-CD raw data completeness verified") for sid in qualified],
        )

        cur.execute(
            "UPDATE core_universe_snapshot SET status='superseded' WHERE snapshot_id=%s",
            (cur_snap,),
        )

        detail = json.dumps({
            "step": "§14.7-CD",
            "n_before": len(current_universe),
            "n_after": len(qualified),
            "n_rejected": len(rejected),
            "reason_hist": reason_hist,
        })
        cur.execute(
            """INSERT INTO universe_revision_log
                (revision_time, actor, action_type, object_type, object_id, policy_version, snapshot_id, detail, note)
               VALUES (NOW(), 'apply_raw_data_completeness_gate', 'raw_data_completeness_gate', 'snapshot',
                       %s, %s, %s, %s::jsonb,
                       '§14.7-CD:每股 raw API source data 完整性 hard gate')""",
            (new_snap, new_policy, new_snap, detail),
        )

        conn.commit()
        logger.info(f"✅ COMMIT done")
        logger.info(f"   New snapshot: {new_snap} ({len(qualified)} stocks)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
