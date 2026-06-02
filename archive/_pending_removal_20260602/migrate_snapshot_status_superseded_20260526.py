"""
migrate_snapshot_status_superseded_20260526.py v0.1 (§14.7-BX Phase C-1 schema migration)
================================================================================
最後更新日期: 2026-05-26
主權狀態: IMPLEMENTED (憲法 v6.1.0 §14.7-BX Phase C-1 落地 / status enum 升版加 superseded)
最高原則: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 📜 一、目的
依憲章 §14.7-BX(第二十二輪)Phase C-1 / canonical spec 之 status enum 升版要求,
為 `core_universe_snapshot.status` 加 CHECK constraint enforcing:
    {committed, superseded, deprecated, draft}

`superseded` 為新值,對映 weekly auto-recommit 之「前週 snapshot 自動 supersede」治權契約。

## 📊 二、執行
| 場景 | 指令 |
| :--- | :--- |
| Dry-run(顯示 SQL 不執行)| `$ python scripts/maintenance/migrate_snapshot_status_superseded_20260526.py --dry-run` |
| Commit | `$ python scripts/maintenance/migrate_snapshot_status_superseded_20260526.py --commit` |

## 📜 三、修訂歷程
| v0.1 | 2026-05-26 | Codex | §14.7-BX Phase C-1 落地;加 CHECK constraint(冪等;若 constraint 已存在則 skip)| ACTIVE |
================================================================================
"""
import argparse
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from core.db_utils import get_db_connection

CONSTRAINT_NAME = "ck_core_universe_snapshot_status"
ALLOWED_VALUES = ('committed', 'superseded', 'deprecated', 'draft')


def main():
    parser = argparse.ArgumentParser(description="§14.7-BX Phase C-1 status enum migration")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--commit", action="store_true")
    args = parser.parse_args()

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Step 1: Check existing status values in DB
        cur.execute("SELECT status, COUNT(*) FROM core_universe_snapshot GROUP BY status ORDER BY status")
        print("=== Pre-migration: existing status values ===")
        invalid = []
        for r in cur.fetchall():
            ok = r[0] in ALLOWED_VALUES
            print(f"  {r[0]}: {r[1]} {'✅' if ok else '❌ NOT IN ALLOWED'}")
            if not ok:
                invalid.append(r[0])
        if invalid:
            print(f"❌ Cannot proceed: existing values {invalid} not in allowed enum {ALLOWED_VALUES}")
            sys.exit(1)

        # Step 2: Check if constraint already exists (idempotent)
        cur.execute("""
            SELECT conname FROM pg_constraint
            WHERE conrelid = 'public.core_universe_snapshot'::regclass
              AND conname = %s
        """, (CONSTRAINT_NAME,))
        existing = cur.fetchone()
        if existing:
            print(f"\n⚠️ CHECK constraint '{CONSTRAINT_NAME}' already exists — idempotent skip")
            conn.close()
            return

        # Step 3: Add CHECK constraint
        values_sql = ", ".join(f"'{v}'" for v in ALLOWED_VALUES)
        ddl = f"ALTER TABLE core_universe_snapshot ADD CONSTRAINT \"{CONSTRAINT_NAME}\" CHECK (status IN ({values_sql}))"
        print(f"\n=== Migration SQL ===")
        print(f"  {ddl}")

        if args.dry_run:
            print("\n--- DRY-RUN; use --commit to execute ---")
            conn.close()
            return

        cur.execute(ddl)
        conn.commit()
        print(f"\n✅ Constraint '{CONSTRAINT_NAME}' added to core_universe_snapshot")

        # Verify
        cur.execute("""
            SELECT pg_get_constraintdef(oid)
            FROM pg_constraint
            WHERE conrelid = 'public.core_universe_snapshot'::regclass
              AND conname = %s
        """, (CONSTRAINT_NAME,))
        print(f"  Verified: {cur.fetchone()[0]}")

    except Exception as e:
        conn.rollback()
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
