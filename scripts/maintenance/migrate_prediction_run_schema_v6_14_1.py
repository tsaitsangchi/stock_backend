"""
migrate_prediction_run_schema_v6_14_1.py — Schema drift fix for prediction_run
================================================================================
最後更新日期: 2026-05-28
治權: §14.7-CS Model Training Landing + §14.7-CT Prediction Production Closure

修補 prediction_run table 之 schema drift(prediction_engine.py v0.3 DDL vs DB 舊 schema):

ALTER TABLE prediction_run:
  + ADD COLUMN IF NOT EXISTS prediction_policy_version VARCHAR(50)
  + ADD COLUMN IF NOT EXISTS rows_written INTEGER
  + ALTER COLUMN label_horizon DROP NOT NULL

Idempotent — 安全多次 run。
"""
from __future__ import annotations
import sys, logging
from pathlib import Path

_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

from core.db_utils import get_db_conn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])


def main():
    conn = get_db_conn()
    try:
        cur = conn.cursor()
        logger.info("=" * 80)
        logger.info("Schema Migration v6.14.1 — prediction_run column alignment")
        logger.info("=" * 80)

        # Check current columns
        cur.execute("""
            SELECT column_name, is_nullable FROM information_schema.columns
            WHERE table_schema='public' AND table_name='prediction_run'
            ORDER BY ordinal_position
        """)
        before = {r[0]: r[1] for r in cur.fetchall()}
        logger.info(f"Before migration:{len(before)} columns")

        # Migration 1: add prediction_policy_version
        cur.execute("""
            ALTER TABLE prediction_run
            ADD COLUMN IF NOT EXISTS prediction_policy_version VARCHAR(50)
        """)
        # Migration 2: add rows_written
        cur.execute("""
            ALTER TABLE prediction_run
            ADD COLUMN IF NOT EXISTS rows_written INTEGER
        """)
        # Migration 3: make label_horizon nullable(prediction_engine v0.3 不提供此欄)
        cur.execute("""
            ALTER TABLE prediction_run
            ALTER COLUMN label_horizon DROP NOT NULL
        """)
        conn.commit()

        # Verify
        cur.execute("""
            SELECT column_name, is_nullable FROM information_schema.columns
            WHERE table_schema='public' AND table_name='prediction_run'
            ORDER BY ordinal_position
        """)
        after = {r[0]: r[1] for r in cur.fetchall()}
        logger.info(f"After migration:{len(after)} columns")

        added = set(after) - set(before)
        nullability_changed = [c for c in before if c in after and before[c] != after[c]]
        logger.info(f"\nColumns added:{sorted(added)}")
        logger.info(f"Nullability changed:{nullability_changed}")

        # Verify code-DB sync via prediction_engine DDL
        logger.info("\n" + "=" * 80)
        logger.info("Verification — code(prediction_engine.py DDL)expects:")
        logger.info("=" * 80)
        expected = ["run_id", "model_id", "feature_set_id", "as_of_date",
                    "universe_snapshot_id", "prediction_policy_version",
                    "rows_written", "status", "notes", "created_at"]
        for col in expected:
            present = col in after
            logger.info(f"  {'✅' if present else '❌'} {col}")

        missing = set(expected) - set(after)
        if missing:
            logger.warning(f"\n  ⚠️ Still missing:{missing}")
        else:
            logger.info(f"\n  🎯 All expected columns present — schema drift resolved")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
