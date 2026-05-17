"""
migrate_anomaly_registry.py v0.1 (Quantum Finance §6.8.8-C D1+D2 Migration)
================================================================================
**最後更新日期**: 2026-05-17
**主權狀態**: ACTIVE (DRAFT) — §6.8.8-C D1 (DB table) + D2 (baseline seed) 載體
**最高原則**: §6.8.8-B 從文件治權升至機器強制治權

對應憲章條文：
- §6.8.8-C (II) DB schema 強制契約
- §6.8.8-C (VII) Registry Mutation Audit Trail
- §6.8.8-C (VIII) D1 + D2 deliverables

執行：
  python scripts/maintenance/migrate_anomaly_registry.py            # 套用 D1+D2
  python scripts/maintenance/migrate_anomaly_registry.py --check    # 只檢查現況，不寫入

Idempotent：可重複執行；不會產生重複 audit trail，不會重覆 insert baseline。
"""
import argparse
import sys
from datetime import date
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_MAINT_DIR = _THIS_FILE.parent
_SCRIPTS_DIR = _MAINT_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from core.db_utils import get_db_connection, record_lifecycle


# §6.8.8-B (II) 首批基線清單（charter ↔ code single source of truth）
# effective_from 採用「事實生效日」語意（非 registry 寫入日）：
# - class A: 該股最後交易日的次日（即「事實上殭屍化」之起點）
# - class D: 該股 IPO 後可投入觀測之早期日（結構性 NA 為近永久事實，採 1990-01-01 為下限保守上界）
BASELINE_ZOMBIES = [
    {"stock_id": "1701", "stock_name": "中化",
     "effective_from": date(2024, 8, 21),
     "reason": "TaiwanStockPrice.MAX(date)=2024-08-20，停止交易 ≈ 9 個月（effective_from = 末交易日+1）"},
    {"stock_id": "1729", "stock_name": "必翔",
     "effective_from": date(2017, 5, 18),
     "reason": "TaiwanStockPrice.MAX(date)=2017-05-17，停止交易 ≈ 9 年（effective_from = 末交易日+1）"},
    {"stock_id": "3559", "stock_name": "全智科",
     "effective_from": date(2017, 8, 24),
     "reason": "TaiwanStockPrice.MAX(date)=2017-08-23，停止交易 ≈ 9 年（effective_from = 末交易日+1）"},
]
_STRUCTURAL_NA_EFFECTIVE_FROM = date(1990, 1, 1)  # 結構性 NA 為近永久事實，採保守早期日
BASELINE_STRUCTURAL_NA = [
    {"stock_id": "6907", "dataset": "TaiwanStockDividend",
     "effective_from": _STRUCTURAL_NA_EFFECTIVE_FROM,
     "reason": "雅特力-KY（KY 公司）結構性無股利資料"},
] + [
    {"stock_id": sid, "dataset": "TaiwanStockMarginPurchaseShortSale",
     "effective_from": _STRUCTURAL_NA_EFFECTIVE_FROM,
     "reason": "tpex 半導體業未開放信用交易或 FinMind 資料集涵蓋限制"}
    for sid in ("6708", "6907", "7751", "7770", "7772", "7810", "7828", "8102")
]

CHARTER_REF_INITIAL = "charter §6.8.8-B (II) initial baseline 2026-05-17"
COMMITTED_BY = "migrate_anomaly_registry.py v0.1"


# §6.8.8-C (II) 補強 DDL（schema framework 不支援之 partial unique + composite CHECK）
DDL_COMPOSITE_CHECK = """
ALTER TABLE universe_anomaly_registry
    DROP CONSTRAINT IF EXISTS ck_anomaly_class_dataset;
ALTER TABLE universe_anomaly_registry
    ADD CONSTRAINT ck_anomaly_class_dataset CHECK (
        (anomaly_class = 'A' AND dataset IS NULL) OR
        (anomaly_class = 'D' AND dataset IS NOT NULL)
    );
"""

DDL_EFFECTIVE_RANGE_CHECK = """
ALTER TABLE universe_anomaly_registry
    DROP CONSTRAINT IF EXISTS ck_anomaly_effective_range;
ALTER TABLE universe_anomaly_registry
    ADD CONSTRAINT ck_anomaly_effective_range CHECK (
        effective_to IS NULL OR effective_to >= effective_from
    );
"""

DDL_PARTIAL_UNIQUE_INDEX = """
CREATE UNIQUE INDEX IF NOT EXISTS uq_anomaly_active
    ON universe_anomaly_registry (anomaly_class, stock_id, COALESCE(dataset, ''))
    WHERE effective_to IS NULL;
"""


def _table_exists(cur, table_name: str) -> bool:
    cur.execute("SELECT to_regclass(%s);", (f'public."{table_name}"',))
    return cur.fetchone()[0] is not None


def _active_entry_exists(cur, class_: str, stock_id: str, dataset) -> bool:
    """檢查是否已存在 effective 之相同 (class, stock_id, dataset) 條目。"""
    if dataset is None:
        cur.execute(
            'SELECT 1 FROM universe_anomaly_registry '
            'WHERE anomaly_class=%s AND stock_id=%s AND dataset IS NULL '
            'AND effective_to IS NULL LIMIT 1',
            (class_, stock_id),
        )
    else:
        cur.execute(
            'SELECT 1 FROM universe_anomaly_registry '
            'WHERE anomaly_class=%s AND stock_id=%s AND dataset=%s '
            'AND effective_to IS NULL LIMIT 1',
            (class_, stock_id, dataset),
        )
    return cur.fetchone() is not None


def _ensure_constraints_and_indexes(cur) -> dict:
    """套用 partial unique index 與 composite CHECK；冪等。"""
    counters = {"checks_applied": 0, "indexes_applied": 0}
    cur.execute(DDL_COMPOSITE_CHECK)
    cur.execute(DDL_EFFECTIVE_RANGE_CHECK)
    counters["checks_applied"] = 2
    cur.execute(DDL_PARTIAL_UNIQUE_INDEX)
    counters["indexes_applied"] = 1
    return counters


def _seed_baseline(cur) -> dict:
    """套用 §6.8.8-B (II) 首批基線；冪等（跳過已存在之 active 條目）。"""
    counters = {"class_A_inserted": 0, "class_A_skipped": 0,
                "class_D_inserted": 0, "class_D_skipped": 0,
                "revision_log_written": 0}

    inserted_class_a = []
    for z in BASELINE_ZOMBIES:
        if _active_entry_exists(cur, "A", z["stock_id"], None):
            counters["class_A_skipped"] += 1
            continue
        cur.execute(
            'INSERT INTO universe_anomaly_registry '
            '(anomaly_class, stock_id, dataset, effective_from, effective_to, '
            ' reason, committed_by, audit_trail_ref, notes) '
            'VALUES (%s, %s, NULL, %s, NULL, %s, %s, %s, %s) '
            'RETURNING registry_id',
            ("A", z["stock_id"], z["effective_from"], z["reason"],
             COMMITTED_BY, CHARTER_REF_INITIAL, f'stock_name={z["stock_name"]}'),
        )
        registry_id = cur.fetchone()[0]
        inserted_class_a.append((z["stock_id"], registry_id))
        counters["class_A_inserted"] += 1

    inserted_class_d = []
    for d in BASELINE_STRUCTURAL_NA:
        if _active_entry_exists(cur, "D", d["stock_id"], d["dataset"]):
            counters["class_D_skipped"] += 1
            continue
        cur.execute(
            'INSERT INTO universe_anomaly_registry '
            '(anomaly_class, stock_id, dataset, effective_from, effective_to, '
            ' reason, committed_by, audit_trail_ref, notes) '
            'VALUES (%s, %s, %s, %s, NULL, %s, %s, %s, NULL) '
            'RETURNING registry_id',
            ("D", d["stock_id"], d["dataset"], d["effective_from"], d["reason"],
             COMMITTED_BY, CHARTER_REF_INITIAL),
        )
        registry_id = cur.fetchone()[0]
        inserted_class_d.append((d["stock_id"], d["dataset"], registry_id))
        counters["class_D_inserted"] += 1

    # §6.8.8-C (VII) Registry Mutation Audit Trail — 寫入 universe_revision_log
    # object_id 受 VARCHAR(255) 限制；批次條目細節走 detail JSONB
    import json
    if inserted_class_a:
        detail_a = {"class": "A",
                    "entries": [{"stock_id": s, "registry_id": r}
                                for s, r in inserted_class_a]}
        cur.execute(
            'INSERT INTO universe_revision_log '
            '(actor, action_type, object_type, object_id, detail, note) '
            'VALUES (%s, %s, %s, %s, %s::jsonb, %s)',
            (COMMITTED_BY,
             "zombie_exclusion 2026-05-17 baseline",
             "universe_anomaly_registry",
             f"baseline_batch_A_n{len(inserted_class_a)}",
             json.dumps(detail_a),
             f"§6.8.8-C D2 baseline seed: {len(inserted_class_a)} class-A zombies"),
        )
        counters["revision_log_written"] += 1

    if inserted_class_d:
        detail_d = {"class": "D",
                    "entries": [{"stock_id": s, "dataset": d, "registry_id": r}
                                for s, d, r in inserted_class_d]}
        cur.execute(
            'INSERT INTO universe_revision_log '
            '(actor, action_type, object_type, object_id, detail, note) '
            'VALUES (%s, %s, %s, %s, %s::jsonb, %s)',
            (COMMITTED_BY,
             "structural_na_registry_add 2026-05-17 baseline",
             "universe_anomaly_registry",
             f"baseline_batch_D_n{len(inserted_class_d)}",
             json.dumps(detail_d),
             f"§6.8.8-C D2 baseline seed: {len(inserted_class_d)} class-D structural NA entries"),
        )
        counters["revision_log_written"] += 1

    return counters


def report_state(cur) -> dict:
    """檢查目前 registry 狀態，回傳統計。"""
    cur.execute(
        'SELECT anomaly_class, COUNT(*) FILTER (WHERE effective_to IS NULL) AS active, '
        '       COUNT(*) AS total '
        'FROM universe_anomaly_registry GROUP BY anomaly_class ORDER BY anomaly_class'
    )
    rows = cur.fetchall()
    state = {"class_A_active": 0, "class_A_total": 0,
             "class_D_active": 0, "class_D_total": 0}
    for cls, active, total in rows:
        state[f"class_{cls}_active"] = active
        state[f"class_{cls}_total"] = total
    return state


def run(check_only: bool = False) -> int:
    conn = get_db_connection()
    cur = conn.cursor()
    with record_lifecycle("migrate_anomaly_registry_v0.1",
                          category="governance",
                          stock_id="SYSTEM") as lifecycle:
        try:
            if not _table_exists(cur, "universe_anomaly_registry"):
                msg = ("universe_anomaly_registry 不存在；請先執行 "
                       "`python scripts/core/core_universe_schema.py --init "
                       "--table universe_anomaly_registry`")
                print(f"❌ {msg}")
                if hasattr(lifecycle, "mark_failed"):
                    lifecycle.mark_failed(msg)
                return 1

            if check_only:
                state = report_state(cur)
                print("📊 universe_anomaly_registry 現況：")
                for k, v in state.items():
                    print(f"   {k:24s} = {v}")
                return 0

            ddl_stats = _ensure_constraints_and_indexes(cur)
            seed_stats = _seed_baseline(cur)
            conn.commit()

            state = report_state(cur)
            print("✅ universe_anomaly_registry migration 完成 (§6.8.8-C D1+D2)")
            print(f"   DDL: {ddl_stats['checks_applied']} CHECK 套用, "
                  f"{ddl_stats['indexes_applied']} partial unique index 套用")
            print(f"   Seed class A (zombies):       "
                  f"inserted={seed_stats['class_A_inserted']}, "
                  f"skipped={seed_stats['class_A_skipped']}")
            print(f"   Seed class D (structural NA): "
                  f"inserted={seed_stats['class_D_inserted']}, "
                  f"skipped={seed_stats['class_D_skipped']}")
            print(f"   universe_revision_log writes: {seed_stats['revision_log_written']}")
            print("📊 現況：")
            for k, v in state.items():
                print(f"   {k:24s} = {v}")

            if seed_stats["class_A_inserted"] + seed_stats["class_D_inserted"] == 0:
                marker = ("class A/D baseline already present; "
                          "no insert performed (idempotent)")
                if hasattr(lifecycle, "mark_warning"):
                    lifecycle.mark_warning(marker)
            return 0
        except Exception as exc:
            conn.rollback()
            msg = f"migration failed: {type(exc).__name__}: {exc}"
            print(f"❌ {msg}")
            if hasattr(lifecycle, "mark_failed"):
                lifecycle.mark_failed(msg)
            return 1
        finally:
            cur.close()
            conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="§6.8.8-C D1+D2: universe_anomaly_registry migration"
    )
    parser.add_argument("--check", action="store_true",
                        help="only report current state; no DDL or seed writes")
    args = parser.parse_args()
    sys.exit(run(check_only=args.check))
