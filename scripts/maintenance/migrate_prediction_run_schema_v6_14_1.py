"""
migrate_prediction_run_schema_v6_14_1.py v0.2 (prediction_run schema migration)
================================================================================
**最後更新日期**: 2026-06-02
**主權狀態**: ACTIVE (one-off / maintenance)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:**一次性** DB migration:prediction_run schema 升至 v6.14.1(已套用)。

**輸入 → 輸出**:既有 DB → 一次性處理結果

**為什麼需要它**:記述性保留(已執行);非常態流程。

## 📜 一、核心定義說明 (Core Definitions)

1. **[One-off Script]**:一次性/維運腳本
2. **[Sovereignty Declaration]**:本程式為**非 charter-core 子系統**工具(charter 可達/引用),不涉 §3.1/§3.2 序列治權主軸、不持五套禁令、不處理 §8.5 anti-leakage。 本檔為**一次性**腳本,非常態 pipeline;保留作 audit trail(亦為 C-隔離候選)。
3. **[Historical Reference Authority]**:本檔標頭版本為記述性快照,非權威來源(權威為憲章 + 程式現行碼)。

## 📊 二、全量功能群矩陣 (Functional Group Matrix)

| 功能 / 指令 | 說明 |
| :--- | :--- |
| python <此檔> | 執行一次性處理 |

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.2 | 2026-06-02 | Codex | §一.11 標頭三段式 + 白話補正;標示一次性。原邏輯不變。 | **ACTIVE** |

## 原始說明
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
