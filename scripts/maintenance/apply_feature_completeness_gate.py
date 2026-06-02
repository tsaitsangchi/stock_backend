"""
apply_feature_completeness_gate.py — §14.7-CB Feature Completeness Gate(Steps 2+3)
================================================================================
最後更新日期: 2026-05-27
主權狀態: DEPRECATED (§14.7-CG v6.5.0 native gate 整合;邏輯已併入 scripts/core/core_universe_builder.py `DoctrineNativeGateBuilder` Stage 4 optional;本 script 保留為歷史 audit trail)
最高原則: Core Universe Doctrinal Feature Completeness Gate

## ⚠️ §14.7-CG 取代備註 (2026-05-27)

- **新 SSOT**: `scripts/core/core_universe_builder.py --mode doctrine-native --with-feature-gate`(§14.7-CG Stage 4 optional feature gate;integrated with §14.7-CD raw gate)
- **本 script 狀態**: 邏輯保留但 calling site 已移除(run_weekly_doctrine_recommit.py Step 4 切換至 native builder)
- **歷史用途**: v0.11 N=1,640 snapshot 為本 script 之 historical evidence
- **下架時點**: 預計 v6.5.x 後完全移除(per §14.7-CG Phase E migration)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:套用 §14.7-CB Feature Completeness Gate:檢查每股 37 特徵是否齊全,不齊者排除核心。

**輸入 → 輸出**:feature_set + universe → 過 gate 之新 snapshot

**為什麼需要它**:確保進模型的核心股特徵完整(無缺 row)。

## 一、核心定義說明
- [Feature Completeness Gate]: 依用戶治權原則「特徵值不全到位之股票不應列入核心股」,
  本 script 對 current core_universe_snapshot 套用 hard gate:
  1. Step 2 data_recency_check:as_of 前 90 天無 PriceAdj 交易資料 → 排除
  2. Step 3 feature_completeness_check:37 spec features 不全 → 排除
- [Atomic Supersede]: 建立 NEW snapshot(status='committed' + supersedes 舊 snapshot),
  舊 snapshot 自動降級為 'superseded',per §14.7-BX 治理機制。
- [Audit Trail]: 同時寫 universe_revision_log 記錄變更原因 + delta count。
- [Reversible]: 不刪除 raw data;若 gate 結果不理想可回退至舊 snapshot。

## 二、CLI 範例
    python scripts/maintenance/apply_feature_completeness_gate.py --dry-run
    python scripts/maintenance/apply_feature_completeness_gate.py --commit
## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.2 | 2026-06-02 | Codex | §一.11 標頭三段式 + 白話段補正;原邏輯不變。 | **ACTIVE** |

## 📊 二、全量維運指令總矩陣 (Operational Matrix)

| 指令 / 模式 | 行為 | 治權對應 |
| :--- | :--- | :--- |
| --dry-run(預設) | 只稽核,不寫 DB | §14.7-CB |
| --commit | 套 gate + 寫新 snapshot | §14.7-CB |
| --feature-set-id <id> | 指定特徵集版本 | 維運 |

"""
from __future__ import annotations

import sys
import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

from core.db_utils import get_db_conn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

SPEC_37_FEATURES = [
    "log_return_20d", "log_return_60d", "log_return_252d",
    "upside_volatility_60d", "downside_volatility_60d", "convexity_60d",
    "avg_daily_value_log_60d", "zero_volume_ratio_252d",
    "pe_ratio", "pb_ratio", "dividend_yield",
    "roe_ttm", "operating_margin_ttm",
    "revenue_yoy_3m_log", "asset_growth_yoy",
    "right_tail_concentration_60d", "barbell_balance_60d", "preferential_attachment_60d",
    "fitness_signal_60d", "right_tail_returns_skew_252d", "liquidity_rank_pct_sector_60d",
    "size_log_zscore_sector",
    "kwave_tech_paradigm_strength", "kwave_credit_cycle_phase", "kwave_credit_to_gdp_gap",
    "kwave_demographics_trend", "kwave_commodity_supercycle", "kwave_phase_indicator",
    "mc_monetary_regime", "mc_yield_curve_inversion", "mc_oil_juglar_phase", "mc_semi_kitchin", "mc_shipping_juglar",
    "ms_volatility_regime", "ms_vix_term_structure", "ms_market_stress",
]


def identify_qualified_stocks(cur, feature_set_id: str, recency_days: int = 90):
    """Identify stocks passing both gates:data recency + feature completeness。"""
    # Step 2: data_recency — stocks with at least one trading day in last `recency_days`
    cur.execute(
        f"""
        SELECT DISTINCT stock_id
        FROM "TaiwanStockPriceAdj"
        WHERE date >= CURRENT_DATE - INTERVAL '{recency_days} days'
        """,
    )
    recent_traders = {r[0] for r in cur.fetchall()}

    # Step 3: feature_completeness — stocks with all 37 spec features
    cur.execute(
        """
        SELECT stock_id, COUNT(DISTINCT feature_name) AS n_features
        FROM feature_values
        WHERE feature_set_id = %s AND feature_name = ANY(%s)
        GROUP BY stock_id
        HAVING COUNT(DISTINCT feature_name) = 37
        """,
        (feature_set_id, SPEC_37_FEATURES),
    )
    feature_complete = {r[0] for r in cur.fetchall()}

    return recent_traders, feature_complete


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--commit", action="store_true", help="apply gate + write new snapshot")
    p.add_argument("--dry-run", action="store_true", default=True, help="audit only")
    p.add_argument("--feature-set-id", default="fs_20260527_feature_set_v0_4")
    p.add_argument("--recency-days", type=int, default=90)
    args = p.parse_args()

    if args.commit:
        args.dry_run = False

    mode = "COMMIT" if not args.dry_run else "DRY-RUN"
    logger.info(f"=== §14.7-CB Feature Completeness Gate / mode={mode} ===")

    conn = get_db_conn()
    try:
        cur = conn.cursor()

        # 1. 取 current committed core_universe_snapshot
        cur.execute(
            """
            SELECT snapshot_id, as_of_date, policy_version
            FROM core_universe_snapshot
            WHERE status = 'committed'
            ORDER BY as_of_date DESC LIMIT 1
            """,
        )
        row = cur.fetchone()
        if not row:
            logger.error("❌ No committed core_universe_snapshot found")
            sys.exit(1)
        cur_snapshot_id, as_of_date, policy_version = row
        logger.info(f"  Current snapshot: {cur_snapshot_id}")

        # 2. 取當前 universe
        cur.execute(
            """
            SELECT m.stock_id, m.core_tier
            FROM core_universe_membership m
            WHERE m.snapshot_id = %s AND m.core_tier IN ('core_universe', 'convex_universe')
            ORDER BY m.stock_id
            """,
            (cur_snapshot_id,),
        )
        current_members = cur.fetchall()
        current_set = {r[0] for r in current_members}
        logger.info(f"  Current universe: {len(current_set)} stocks")

        # 3. Run two gates
        recent, complete = identify_qualified_stocks(cur, args.feature_set_id, args.recency_days)
        logger.info(f"  Recent traders (last {args.recency_days}d): {len(recent)}")
        logger.info(f"  Feature complete (37/37): {len(complete)}")

        # 4. Intersection = qualified
        qualified = current_set & recent & complete
        stranded = current_set - recent
        incomplete = (current_set & recent) - complete

        logger.info(f"\n📊 Gate Audit Result:")
        logger.info(f"  Current universe          : {len(current_set):4d}")
        logger.info(f"  ❌ Stranded (no recency)  : {len(stranded):4d}")
        logger.info(f"  ❌ Incomplete features    : {len(incomplete):4d}")
        logger.info(f"  ✅ QUALIFIED              : {len(qualified):4d}")
        logger.info(f"  Reduction                 : {len(current_set) - len(qualified):4d} (-{100.0 * (len(current_set) - len(qualified)) / len(current_set):.1f}%)")

        if args.dry_run:
            logger.info(f"\n[DRY-RUN] no DB changes;showing first 10 stocks per category")
            logger.info(f"  Sample stranded: {sorted(stranded)[:10]}")
            logger.info(f"  Sample incomplete: {sorted(incomplete)[:10]}")
            logger.info(f"\nTo commit gate:python {Path(__file__).name} --commit")
            return

        # 5. COMMIT mode:建立 new snapshot
        today_str = date.today().strftime("%Y%m%d")
        new_policy_version = "core_universe_policy_v0.11_feature_completeness_gate"
        new_snapshot_id = f"core_universe_{today_str}_{new_policy_version.replace('.', '_')}"

        logger.info(f"\n📝 Creating new snapshot: {new_snapshot_id}")

        # 5a. Ensure policy_version exists(FK requirement)
        cur.execute(
            """
            INSERT INTO core_universe_policy
              (policy_version, policy_name, description, active, effective_from)
            VALUES (%s, %s, %s, TRUE, CURRENT_DATE)
            ON CONFLICT (policy_version) DO NOTHING
            """,
            (new_policy_version,
             "Feature Completeness Gate v6.4.0",
             "§14.7-CB:enforce 37/37 spec features + 90d data recency as hard gate for core_universe entry"),
        )

        # 5b. Insert new snapshot
        cur.execute(
            """
            INSERT INTO core_universe_snapshot
              (snapshot_id, as_of_date, source_data_cutoff, policy_version,
               total_candidates, core_count, status, notes, created_at)
            VALUES (%s, CURRENT_DATE, CURRENT_DATE, %s, %s, %s, 'committed',
              '§14.7-CB Feature Completeness Gate applied(Step 2+3);superseded ' || %s, NOW())
            """,
            (new_snapshot_id, new_policy_version, len(current_set), len(qualified), cur_snapshot_id),
        )

        # 5c. Insert qualified stocks
        qualified_sorted = sorted(qualified)
        cur.executemany(
            """
            INSERT INTO core_universe_membership
              (snapshot_id, stock_id, core_tier, active, selected_at, selection_reason)
            VALUES (%s, %s, 'core_universe', TRUE, NOW(), %s)
            """,
            [
                (new_snapshot_id, sid, "§14.7-CB gate passed:37/37 features + 90d recency")
                for sid in qualified_sorted
            ],
        )

        # 5d. Supersede old snapshot
        cur.execute(
            """
            UPDATE core_universe_snapshot SET status = 'superseded' WHERE snapshot_id = %s
            """,
            (cur_snapshot_id,),
        )

        # 5e. Log revision(per schema)
        import json
        detail = json.dumps({
            "step": "§14.7-CB Step 2+3",
            "n_before": len(current_set),
            "n_after": len(qualified),
            "n_stranded_dropped": len(stranded),
            "n_incomplete_dropped": len(incomplete),
            "reduction_pct": round(100.0 * (len(current_set) - len(qualified)) / len(current_set), 2),
        })
        cur.execute(
            """
            INSERT INTO universe_revision_log
              (revision_time, actor, action_type, object_type, object_id, policy_version, snapshot_id, detail, note)
            VALUES (NOW(), 'apply_feature_completeness_gate', 'feature_completeness_gate', 'snapshot', %s, %s, %s, %s::jsonb,
                    '§14.7-CB:Step 2 data_recency + Step 3 feature_completeness 套用')
            """,
            (new_snapshot_id, new_policy_version, new_snapshot_id, detail),
        )

        conn.commit()
        logger.info(f"✅ COMMIT done")
        logger.info(f"   New snapshot: {new_snapshot_id} ({len(qualified)} stocks)")
        logger.info(f"   Superseded:   {cur_snapshot_id}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
