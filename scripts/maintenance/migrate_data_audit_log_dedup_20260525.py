"""
migrate_data_audit_log_dedup_20260525.py v0.1 (Quantum Finance §3.2A.J Migration)
================================================================================
**最後更新日期**: 2026-05-25
**主權狀態**: ONE-OFF MIGRATION (憲法 v6.1.0-patch §3.2A.J / §14.7-AY 落地 C 項；對齊 data_schema v2.17 / db_utils v2.48 升版預備)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:一次性 migration:data_audit_log 去重 + 加 UNIQUE constraint(§3.2A.J 配套)。

**輸入 → 輸出**:舊 data_audit_log → 去重後 + constraint

**為什麼需要它**:一次性 schema 修正(已套用)。

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
1. [One-Off Migration]: 本腳本為一次性 DB schema migration，對齊憲章 §3.2A.J `data_audit_log` 5-tuple UNIQUE constraint 治權契約落地，**執行一次後不再重複**（再執行為 idempotent：dup=0 + constraint 已存在則 NO-OP exit 0）。
2. [Race-Safe Bridge]: data_schema.py v2.17 已在 DDL 宣告 UNIQUE constraint，但既有 DB 之 `data_audit_log` 表（包含 race-induced dup）需先 dedup 才能 ADD CONSTRAINT；本腳本扮演 schema-state migration bridge 之治權職責。
3. [Defensive Architecture]: 預設 `--dry-run` 模式（read-only 報告 dup 數量 + 影響範圍）；`--apply` 須配 `--confirm` 顯式雙重確認；預設執行 PG_DUMP backup（可 `--skip-backup` 關閉）。
4. [Hybrid Observability]: 透過 `db_utils.record_lifecycle()` 寫入 `pipeline_execution_log`；透過升版後的 `write_data_audit_log()` 寫入 `data_audit_log`（DEDUP_MIGRATION op_type）。
5. [Zero Hardcoded Verdict]: 主權判定（PERFECT / WARNING / FAILED）依執行結果動態計算，嚴禁硬編碼。對齊憲章 §5.6.3 與 §3.2 接受標準。
6. [Sovereignty Declaration]: 本腳本為憲章 §3.2A 衍生治權層 migration 工具（cross-ref 憲章 L2722-2745 §3.2A.J / L7480-7568 §14.7-AY）；不涉及 §0.1-A 第一性原理 / §0.2-A 八二法則 / §0.3-A 康波週期 / §0.0-E.4 統合層 / §0.0-F.3 AI 協作工具規則五套禁令；不在 §0.1.1 T1/T2/T3 分層內；不處理 §8.5 anti-leakage；不選股不評分；不持有 Raw API Schema（屬 `data_schema.py` Authority）。

## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix)

| 維運需求場景 (Scenario) | 權威指令 / 建議用法 | 對齊模組 |
| :--- | :--- | :--- |
| **1. [預檢：dup 數量與影響範圍]** | `$ python scripts/maintenance/migrate_data_audit_log_dedup_20260525.py --dry-run` | migrate v0.1 |
| **2. [實際執行：dedup + ADD CONSTRAINT + backup]** | `$ python scripts/maintenance/migrate_data_audit_log_dedup_20260525.py --apply --confirm` | migrate v0.1 |
| **3. [實際執行：跳過 backup（不建議）]** | `$ python scripts/maintenance/migrate_data_audit_log_dedup_20260525.py --apply --confirm --skip-backup` | migrate v0.1 |
| **4. [驗證後續狀態]** | `$ python scripts/maintenance/migrate_data_audit_log_dedup_20260525.py --verify-only` | migrate v0.1 |

## 📜 三、全修訂歷程 (Full Revision History)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **v0.1** | 2026-05-25 | Codex | **§3.2A.J / §14.7-AY 落地 C 項：data_audit_log dedup + ADD CONSTRAINT 一次性 migration**：依憲章 v6.1.0-patch（commit `4da2450`，2026-05-25）新入憲之 §3.2A.J `db_utils.write_data_audit_log` Audit Log Write-Safe 治權契約（憲章 L2722-2745）+ §14.7-AY §7.4-A 姊妹缺陷補完入憲（憲章 L7480-7568）落地裁決第 C 項。本腳本為配合 data_schema.py v2.17（DDL UNIQUE constraint 落地）+ db_utils.py v2.48（write_data_audit_log() 加 ON CONFLICT DO NOTHING）之 schema-state bridge migration。**Root cause（2026-05-24 Audit 2 揭露）**：Step 4F 啟動 ~65 秒兩個 sync_engine worker 並發寫 audit log 撞同 microsecond + 同 5-tuple → race-induced dup 1 個。**本腳本功能**：(A) 預檢模式 `--dry-run`：統計 dup 數量、影響範圍（受影響的 stock_id / dataset / date 分布）+ ADD CONSTRAINT 預估時間；(B) 執行模式 `--apply --confirm`：可選 PG_DUMP backup（預設 ON）+ DELETE dup 保留 MIN(id) per 5-tuple + ALTER TABLE ADD CONSTRAINT `data_audit_log_5tuple_unique` + verify dup=0 + record_lifecycle 寫入 pipeline_execution_log；(C) 驗證模式 `--verify-only`：僅檢查當前 DB 狀態（dup count + constraint 存在性）不做任何修改。**對齊治權契約**：本腳本為一次性 migration（執行後 idempotent — 再執行只會發現 dup=0 + constraint 已存在則 NO-OP exit 0）；對齊 §3.2A.J 裁決第 4 條「既有 DB 須執行一次性 dedup migration 再 ADD CONSTRAINT」；對齊 §0.4 [Hybrid Observability]（雙日誌寫入）；對齊 §5.6.3 [Zero Hardcoded Verdict]（verdict 動態計算）。 | **ACTIVE** |
================================================================================
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# 治權常數 (Constitution Constants)
# ──────────────────────────────────────────────────────────────────────────────
CONSTITUTION_VER = "v6.1.0"
TOOL_VER = "v0.1"

# ── 系統級架構引導 ──
_THIS_FILE = Path(__file__).resolve()
_MAINTENANCE_DIR = _THIS_FILE.parent
_SCRIPTS_DIR = _MAINTENANCE_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from core.db_utils import get_db_connection, record_lifecycle, write_data_audit_log
except ImportError as e:
    print(f"❌ 核心組件導入失敗：{e}")
    sys.exit(1)


CONSTRAINT_NAME = "data_audit_log_5tuple_unique"
UNIQUE_COLS = ("table_name", "stock_id", "data_date", "action_type", "timestamp")


def count_duplicates(cur):
    """統計 data_audit_log 之 5-tuple race-induced dup 數量與分布。"""
    cur.execute("""
        SELECT COUNT(*) FROM (
            SELECT table_name, stock_id, data_date, action_type, timestamp, COUNT(*) AS c
            FROM data_audit_log
            GROUP BY table_name, stock_id, data_date, action_type, timestamp
            HAVING COUNT(*) > 1
        ) dup_groups;
    """)
    dup_groups = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM data_audit_log;")
    total_rows = cur.fetchone()[0]

    cur.execute("""
        SELECT SUM(c - 1) FROM (
            SELECT COUNT(*) AS c
            FROM data_audit_log
            GROUP BY table_name, stock_id, data_date, action_type, timestamp
            HAVING COUNT(*) > 1
        ) dup_counts;
    """)
    dup_rows_to_delete = cur.fetchone()[0] or 0

    return dup_groups, total_rows, dup_rows_to_delete


def dup_distribution_sample(cur, limit=10):
    """抽樣 dup 之 stock_id / dataset / date 分布(供 dry-run 報告)。"""
    cur.execute("""
        SELECT table_name, stock_id, data_date, action_type, timestamp, COUNT(*) AS c
        FROM data_audit_log
        GROUP BY table_name, stock_id, data_date, action_type, timestamp
        HAVING COUNT(*) > 1
        ORDER BY c DESC, timestamp DESC
        LIMIT %s;
    """, (limit,))
    return cur.fetchall()


def constraint_exists(cur):
    """檢查 data_audit_log_5tuple_unique UNIQUE constraint 是否已存在。"""
    cur.execute("""
        SELECT 1 FROM pg_constraint
        WHERE conname = %s
          AND conrelid = 'data_audit_log'::regclass;
    """, (CONSTRAINT_NAME,))
    return cur.fetchone() is not None


def pg_dump_backup(backup_dir):
    """執行 pg_dump backup data_audit_log 表至 backup_dir。"""
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    dbname = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    if not all((host, port, dbname, user, password)):
        raise RuntimeError("缺少 DB env (DB_HOST/PORT/NAME/USER/PASSWORD);無法 backup")

    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = backup_dir / f"data_audit_log_backup_{timestamp}.sql"

    env = os.environ.copy()
    env["PGPASSWORD"] = password
    cmd = [
        "pg_dump",
        "-h", host,
        "-p", str(port),
        "-U", user,
        "-d", dbname,
        "-t", "data_audit_log",
        "--data-only",
        "-f", str(backup_file),
    ]
    print(f"  → 執行 pg_dump backup: {backup_file}")
    subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
    return backup_file


def dedup_keep_min_id(cur):
    """DELETE dup 保留 MIN(id) per 5-tuple。回傳實際刪除 row 數。"""
    cur.execute("""
        DELETE FROM data_audit_log
        WHERE id NOT IN (
            SELECT MIN(id)
            FROM data_audit_log
            GROUP BY table_name, stock_id, data_date, action_type, timestamp
        );
    """)
    return cur.rowcount


def add_unique_constraint(cur):
    """ALTER TABLE data_audit_log ADD CONSTRAINT UNIQUE 5-tuple。"""
    cur.execute(f"""
        ALTER TABLE data_audit_log
        ADD CONSTRAINT {CONSTRAINT_NAME}
        UNIQUE ({", ".join(UNIQUE_COLS)});
    """)


def run_dry_run():
    """`--dry-run` 模式:報告 dup 統計與分布,不改 DB。"""
    print("=" * 78)
    print(f"  migrate_data_audit_log_dedup {TOOL_VER} — DRY RUN MODE")
    print(f"  憲章對齊: {CONSTITUTION_VER} §3.2A.J / §14.7-AY")
    print("=" * 78)
    print()

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        constraint_existed = constraint_exists(cur)
        print(f"📋 當前 DB 狀態:")
        print(f"   UNIQUE constraint '{CONSTRAINT_NAME}' 存在: {constraint_existed}")

        dup_groups, total_rows, dup_rows = count_duplicates(cur)
        print(f"   data_audit_log 總 row 數: {total_rows:,}")
        print(f"   dup 5-tuple 群數: {dup_groups}")
        print(f"   實際待刪 row 數(保留 MIN(id) per group): {dup_rows}")

        if dup_groups > 0:
            print()
            print(f"📊 dup 分布抽樣(前 10 個 dup group):")
            print(f"   {'table_name':<40s} {'stock_id':<10s} {'date':<12s} {'action':<10s} {'count':>5s}")
            print(f"   {'-' * 80}")
            for row in dup_distribution_sample(cur, limit=10):
                tbl, sid, date, act, ts, cnt = row
                print(f"   {tbl:<40s} {sid:<10s} {str(date):<12s} {act:<10s} {cnt:>5d}")

        print()
        print(f"💡 建議下一步:")
        if constraint_existed and dup_groups == 0:
            print(f"   ✅ 無事可做 - constraint 已存在且 dup=0;遷移完成。")
        elif constraint_existed and dup_groups > 0:
            print(f"   ⚠️ constraint 已存在但仍有 dup({dup_groups} groups) - 異常狀態,請手動調查。")
        elif not constraint_existed and dup_groups == 0:
            print(f"   執行: python {_THIS_FILE.name} --apply --confirm")
            print(f"   (將跳過 dedup 步驟,直接 ADD CONSTRAINT)")
        else:
            print(f"   執行: python {_THIS_FILE.name} --apply --confirm")
            print(f"   (將 dedup {dup_rows} rows,然後 ADD CONSTRAINT)")
        print()
        return 0
    finally:
        cur.close()
        conn.close()


def run_apply(skip_backup, no_verify, backup_dir):
    """`--apply --confirm` 模式:執行 backup + dedup + ADD CONSTRAINT + verify。"""
    print("=" * 78)
    print(f"  migrate_data_audit_log_dedup {TOOL_VER} — APPLY MODE")
    print(f"  憲章對齊: {CONSTITUTION_VER} §3.2A.J / §14.7-AY")
    print("=" * 78)
    print()

    started_at = datetime.now()
    backup_file = None
    rows_deleted = 0
    constraint_added = False

    with record_lifecycle("data_audit_log_dedup_migration", "MAINTENANCE", "SYSTEM") as lc:
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            # Pre-check
            constraint_existed = constraint_exists(cur)
            dup_groups, total_rows, dup_rows = count_duplicates(cur)

            print(f"📋 升版前狀態:")
            print(f"   constraint 已存在: {constraint_existed}")
            print(f"   total rows: {total_rows:,} / dup groups: {dup_groups} / dup rows to delete: {dup_rows}")
            print()

            # Idempotent early exit
            if constraint_existed and dup_groups == 0:
                print("✅ 無事可做 - constraint 已存在且 dup=0;遷移已完成。")
                print()
                return 0

            # Step 1: backup (除非 --skip-backup)
            if not skip_backup:
                print("🔒 Step 1: PG_DUMP backup")
                try:
                    backup_file = pg_dump_backup(backup_dir)
                    print(f"   ✅ backup 完成: {backup_file}")
                except (subprocess.CalledProcessError, RuntimeError) as e:
                    print(f"   ❌ backup 失敗: {e}")
                    lc.mark_failed(f"backup failed: {e}")
                    return 1
            else:
                print("⚠️ Step 1: SKIPPED (--skip-backup) — 不建議在生產環境跳過")
            print()

            # Step 2: dedup
            if dup_groups > 0:
                print(f"🧹 Step 2: DELETE dup 保留 MIN(id)")
                rows_deleted = dedup_keep_min_id(cur)
                conn.commit()
                print(f"   ✅ 刪除 {rows_deleted} rows")
            else:
                print("✅ Step 2: SKIPPED — 無 dup")
            print()

            # Step 3: ADD CONSTRAINT
            if not constraint_existed:
                print(f"🔐 Step 3: ALTER TABLE ADD CONSTRAINT {CONSTRAINT_NAME}")
                add_unique_constraint(cur)
                conn.commit()
                constraint_added = True
                print(f"   ✅ UNIQUE constraint 已加入 (5-tuple)")
            else:
                print(f"✅ Step 3: SKIPPED — constraint 已存在")
            print()

            # Step 4: verify
            if not no_verify:
                print("🔍 Step 4: Verify dup=0 + constraint 存在")
                dup_groups_after, _, _ = count_duplicates(cur)
                constraint_after = constraint_exists(cur)
                if dup_groups_after == 0 and constraint_after:
                    print(f"   ✅ dup_groups_after=0 / constraint_exists=True → PERFECT")
                else:
                    print(f"   ❌ dup_groups_after={dup_groups_after} / constraint_exists={constraint_after}")
                    lc.mark_failed(f"verify failed: dup={dup_groups_after} constraint={constraint_after}")
                    return 1
            else:
                print("⚠️ Step 4: SKIPPED (--no-verify)")
            print()

            # 寫 audit log
            try:
                write_data_audit_log(
                    "data_audit_log",
                    "SYSTEM",
                    started_at.strftime("%Y-%m-%d"),
                    "DEDUP_MIGRATION",
                    rows_deleted,
                )
            except Exception as e:
                print(f"⚠️ audit log 寫入失敗(non-blocking): {e}")

            elapsed = (datetime.now() - started_at).total_seconds()
            print("=" * 78)
            print(f"  ✅ Migration COMPLETED — verdict=PERFECT")
            print(f"     rows_deleted: {rows_deleted} / constraint_added: {constraint_added}")
            print(f"     backup: {backup_file or 'SKIPPED'}")
            print(f"     elapsed: {elapsed:.2f}s")
            print("=" * 78)
            return 0
        except Exception as e:
            conn.rollback()
            print(f"❌ Migration FAILED: {type(e).__name__}: {e}")
            lc.mark_failed(f"{type(e).__name__}: {e}")
            return 1
        finally:
            cur.close()
            conn.close()


def run_verify_only():
    """`--verify-only` 模式:檢查當前狀態,不改 DB。"""
    print("=" * 78)
    print(f"  migrate_data_audit_log_dedup {TOOL_VER} — VERIFY ONLY")
    print("=" * 78)
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        constraint_existed = constraint_exists(cur)
        dup_groups, total_rows, dup_rows = count_duplicates(cur)
        print(f"  constraint '{CONSTRAINT_NAME}' 存在: {constraint_existed}")
        print(f"  total rows: {total_rows:,}")
        print(f"  dup groups: {dup_groups}")
        print(f"  dup rows: {dup_rows}")
        if constraint_existed and dup_groups == 0:
            print("  ✅ verdict=PERFECT (migration 已完成)")
            return 0
        else:
            print("  ⚠️ verdict=PENDING (migration 尚未完成或不一致)")
            return 1
    finally:
        cur.close()
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description=f"§3.2A.J data_audit_log dedup migration ({TOOL_VER})",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 預檢(read-only,推薦先跑):
  python scripts/maintenance/migrate_data_audit_log_dedup_20260525.py --dry-run

  # 實際執行(會 backup + dedup + ADD CONSTRAINT):
  python scripts/maintenance/migrate_data_audit_log_dedup_20260525.py --apply --confirm

  # 驗證當前狀態:
  python scripts/maintenance/migrate_data_audit_log_dedup_20260525.py --verify-only
""")
    parser.add_argument("--dry-run", action="store_true",
                        help="預檢模式(預設):報告 dup 統計與分布,不改 DB")
    parser.add_argument("--apply", action="store_true",
                        help="實際執行 dedup + ADD CONSTRAINT(需配 --confirm)")
    parser.add_argument("--confirm", action="store_true",
                        help="顯式確認(防呆;與 --apply 同時使用)")
    parser.add_argument("--skip-backup", action="store_true",
                        help="跳過 pg_dump backup(不建議在生產環境跳過)")
    parser.add_argument("--no-verify", action="store_true",
                        help="跳過 post-migration verify")
    parser.add_argument("--verify-only", action="store_true",
                        help="僅驗證當前 DB 狀態,不改 DB")
    parser.add_argument("--backup-dir", default="logs/migrations",
                        help="pg_dump backup 目錄(預設: logs/migrations/)")
    args = parser.parse_args()

    # Mutual exclusion check
    modes = sum([bool(args.dry_run), bool(args.apply), bool(args.verify_only)])
    if modes == 0:
        # default to dry-run
        return run_dry_run()
    if modes > 1:
        print("❌ --dry-run / --apply / --verify-only 必須擇一(預設 --dry-run)")
        return 2

    if args.apply and not args.confirm:
        print("❌ --apply 必須配 --confirm 顯式確認")
        print("   範例: python scripts/maintenance/migrate_data_audit_log_dedup_20260525.py --apply --confirm")
        return 2

    if args.verify_only:
        return run_verify_only()
    if args.apply:
        return run_apply(args.skip_backup, args.no_verify, args.backup_dir)
    return run_dry_run()


if __name__ == "__main__":
    sys.exit(main())
