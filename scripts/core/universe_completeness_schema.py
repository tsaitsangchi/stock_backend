"""
universe_completeness_schema.py v0.1 (§14.7-BU Phase C+D — Cross-Layer × Cross-Pillar Universe Completeness Governance + §8 Prediction Layer Schema)
================================================================================
最後更新日期: 2026-05-26
主權狀態: IMPLEMENTED (憲法 v6.1.0 §14.7-BU Phase C+D 落地 / §0.4 數位孿生完整性 implicit→explicit / §8 Prediction Layer 0% gap closure / Path C hybrid)
最高原則: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 📜 一、核心定義說明 (Core Definitions)
1. [Universe Completeness Schema Authority]: 本工具只建立 §14.7-BU Path C 之 3 new tables + 1 materialized view（prediction_run / predictions / universe_completeness_snapshot / universe_completeness_matrix_current）;不管理 raw API schema、不管理 core_universe_* 治理表（由 core_universe_schema.py 主管）、不管理 feature_store_* 表（由 feature_store_schema.py 主管）、不管理 model_registry（由 data_schema.py 主管）。
2. [Cross-Layer SSOT Boundary]: universe_completeness_snapshot 為 trinity × layer 跨層完整性唯一查詢入口;但既有 28 tables 之 layer authority 紀律維持 — 每層 builder 仍主管自己 audit（fetch_log / data_audit_log / feature_store_* / model_*）,不破。
3. [§8 Prediction Layer Closure]: prediction_run + predictions 為 §8 ACTIVE DRAFT 自 v6.0.0 起 0% gap 之 schema 落地（per §14.7-BU Phase B 入憲）;此後 §8 升正式條文之 schema 預備完成。
4. [Purely Additive]: 本路徑只新增,不修改既有 28 tables。任一既有 schema 變動為 §14.7-BU 治權邊界破壞 → preflight FAIL。
5. [Trinity Enum Discipline]: pillar ∈ {first_principle, pareto, kondratiev}（對映 §0.1 / §0.2 / §0.3）;layer ∈ {data, feature, model, prediction}（對映 §8 三層 + raw）。違反 enum 為 DB CHECK constraint 阻擋,不需 application 層驗證。
6. [Materialized View Refresh Discipline]: universe_completeness_matrix_current 為 reporting layer;CONCURRENTLY refresh 需 UNIQUE INDEX（已建立）;refresh 不阻塞 reads。Refresh trigger 為(a) 新 core_universe_snapshot commit (b) builders 寫入後 (c) 手動 REFRESH。
7. [Zero Hardcoded Verdict]: 主權判定動態計算（§5.6.3）;verdict ∈ {PERFECT, WARNING, FAILED}。
8. [Sovereignty Declaration]: 本工具屬 §3.2 橫切 library + §14.7-BU Schema Authority（per §14.7-BU Phase C+D 落地）;不選股（不涉 §6 CoreScore）、不訓練模型（不涉 §8.3）、不預測（不涉 §9.1）、不分配資金（不涉 §9.2）、不涉 §0.1-A / §0.2-A / §0.3-A 三套禁令、不在 T1-T3 分層內、不處理 §8.5 anti-leakage（由 builders 主管）。
9. [Historical Reference Authority]: v0.1 為首版落地（§14.7-BU Phase C+D 同次入憲）;後續升版保留歷程。
10. [Hybrid Observability]: 維運觸發 record_lifecycle;DDL 寫入 data_audit_log。

## 📊 二、全量維運指令總矩陣
| 場景 | 指令 | 對齊 |
| :--- | :--- | :--- |
| 1. 初始化 universe_completeness 治理層（3 tables + 1 view） | `$ python scripts/core/universe_completeness_schema.py --init` | universe_completeness_schema v0.1 |
| 2. 強制重鑄（drop + recreate） | `$ python scripts/core/universe_completeness_schema.py --init --force` | universe_completeness_schema v0.1 |
| 3. 單表重鑄 | `$ python scripts/core/universe_completeness_schema.py --init --table universe_completeness_snapshot` | universe_completeness_schema v0.1 |
| 4. 離線復原（略過 preflight） | `$ python scripts/core/universe_completeness_schema.py --init --skip-preflight` | universe_completeness_schema v0.1 |
| 5. Materialized view refresh（手動） | `$ python scripts/core/universe_completeness_schema.py --refresh-view` | universe_completeness_schema v0.1 |

## 📜 三、全修訂歷程 (Full Revision History)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.1-§一.11 | 2026-05-29 | Codex | **§一.11 三段式標頭規範對齊**:標題從「修訂歷程」改為「全修訂歷程 (Full Revision History)」對齊 CLAUDE.md §一.11 強制格式。原 v0.1 邏輯不變。 | **ACTIVE** |
| **v0.1** | 2026-05-26 | Codex | **§14.7-BU Phase C+D 落地首版**:依憲章 v6.1.0-patch 第十九輪 §14.7-BU Phase B 入憲（charter L9378 新子節）+ Phase A 設計研究（commit `73228ba` 523 行 §6 schemas）;建 3 new tables + 1 materialized view。Path C hybrid 之治權邊界:(I) prediction_run + predictions 補 §8 prediction layer 0% gap;(II) universe_completeness_snapshot 為 cross-layer × cross-pillar SSOT（PK = snapshot_id × stock_id × pillar × layer / CHECK enums for pillar + layer + pct）;(III) universe_completeness_matrix_current 為 reporting materialized view（CONCURRENTLY refresh ready）。對既有 28 tables 影響:零 schema 變動（purely additive）。前置依賴:core_universe_snapshot / core_universe_policy / model_registry / feature_store_snapshot / TaiwanStockInfo / pipeline_execution_log / data_audit_log。對應 §0.4 數位孿生完整性 implicit→explicit;對應 §8 ACTIVE DRAFT 升正式條文 schema 預備完成。 | **ACTIVE** |
================================================================================
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_CORE_DIR = _THIS_FILE.parent
_SCRIPTS_DIR = _CORE_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from core.db_utils import get_db_connection, record_lifecycle, write_data_audit_log
except ImportError as exc:
    print(f"❌ 核心組件導入失敗,請確認 core/ 目錄: {exc}")
    sys.exit(1)


CONSTITUTION_VER = "v6.1.0"
TOOL_VER = "v0.1"


UNIVERSE_COMPLETENESS_REGISTRY = {
    "prediction_run": {
        "columns": {
            "run_id": "VARCHAR(255) PRIMARY KEY",
            "model_id": "VARCHAR(255) NOT NULL",
            "feature_set_id": "VARCHAR(255) NOT NULL",
            "universe_snapshot_id": "VARCHAR(255) NOT NULL",
            "as_of_date": "DATE NOT NULL",
            "label_horizon": "INTEGER NOT NULL",
            "total_predictions": "INTEGER NOT NULL DEFAULT 0",
            "null_predictions": "INTEGER NOT NULL DEFAULT 0",
            "status": "VARCHAR(64) NOT NULL DEFAULT 'draft'",
            "sector_balance_applied": "BOOLEAN DEFAULT FALSE",
            "sector_balance_params": "JSONB",
            "notes": "TEXT",
            "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        },
        "unique_constraints": [["model_id", "as_of_date"]],
        "foreign_keys": [
            {"columns": ["model_id"], "ref_table": "model_registry", "ref_columns": ["model_id"], "on_delete": "RESTRICT"},
            {"columns": ["feature_set_id"], "ref_table": "feature_store_snapshot", "ref_columns": ["feature_set_id"], "on_delete": "RESTRICT"},
            {"columns": ["universe_snapshot_id"], "ref_table": "core_universe_snapshot", "ref_columns": ["snapshot_id"], "on_delete": "RESTRICT"},
        ],
        "indexes": [
            ("idx_prediction_run_as_of", ["as_of_date"]),
            ("idx_prediction_run_status", ["status", "as_of_date"]),
        ],
    },
    "predictions": {
        "columns": {
            "run_id": "VARCHAR(255) NOT NULL",
            "stock_id": "VARCHAR(255) NOT NULL",
            "predicted_value": "NUMERIC",
            "prediction_rank": "INTEGER",
            "raw_value": "NUMERIC",
            "sector_penalty": "NUMERIC",
            "industry_category": "VARCHAR(255)",
            "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        },
        "primary_key": ["run_id", "stock_id"],
        "foreign_keys": [
            {"columns": ["run_id"], "ref_table": "prediction_run", "ref_columns": ["run_id"], "on_delete": "CASCADE"},
        ],
        "indexes": [
            ("idx_predictions_stock", ["stock_id", "run_id"]),
            ("idx_predictions_rank", ["run_id", "prediction_rank"]),
        ],
    },
    "universe_completeness_snapshot": {
        "columns": {
            "snapshot_id": "VARCHAR(255) NOT NULL",
            "universe_snapshot_id": "VARCHAR(255) NOT NULL",
            "as_of_date": "DATE NOT NULL",
            "stock_id": "VARCHAR(255) NOT NULL",
            "pillar": "VARCHAR(32) NOT NULL",
            "layer": "VARCHAR(32) NOT NULL",
            "expected_items": "INTEGER NOT NULL",
            "actual_items": "INTEGER NOT NULL",
            "completeness_pct": "NUMERIC(5,2) NOT NULL",
            "missing_items": "JSONB",
            "evidence_source_table": "VARCHAR(255)",
            "computed_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        },
        "primary_key": ["snapshot_id", "stock_id", "pillar", "layer"],
        "check_constraints": [
            ("ck_completeness_pillar", "pillar IN ('first_principle', 'pareto', 'kondratiev')"),
            ("ck_completeness_layer", "layer IN ('data', 'feature', 'model', 'prediction')"),
            ("ck_completeness_pct", "completeness_pct BETWEEN 0 AND 100"),
        ],
        "foreign_keys": [
            {"columns": ["universe_snapshot_id"], "ref_table": "core_universe_snapshot", "ref_columns": ["snapshot_id"], "on_delete": "RESTRICT"},
        ],
        "indexes": [
            ("idx_completeness_universe", ["universe_snapshot_id"]),
            ("idx_completeness_query", ["pillar", "layer", "completeness_pct"]),
            ("idx_completeness_stock", ["stock_id", "snapshot_id"]),
        ],
    },
}

MATERIALIZED_VIEW_NAME = "universe_completeness_matrix_current"
MATERIALIZED_VIEW_DDL = """
CREATE MATERIALIZED VIEW "universe_completeness_matrix_current" AS
SELECT
    c.stock_id,
    c.pillar,
    c.layer,
    c.completeness_pct,
    c.actual_items,
    c.expected_items,
    c.missing_items,
    c.evidence_source_table,
    u.policy_version,
    u.as_of_date AS universe_as_of_date,
    c.as_of_date AS completeness_as_of_date,
    c.computed_at
FROM "universe_completeness_snapshot" c
JOIN "core_universe_snapshot" u ON c.universe_snapshot_id = u.snapshot_id
WHERE u.status = 'committed'
  AND u.as_of_date = (
      SELECT MAX(as_of_date) FROM "core_universe_snapshot" WHERE status = 'committed'
  )
""".strip()

MATERIALIZED_VIEW_UNIQUE_INDEX = (
    "idx_completeness_mv_unique",
    ["stock_id", "pillar", "layer"],
)

PREREQUISITE_TABLES = [
    "pipeline_execution_log",
    "data_audit_log",
    "TaiwanStockInfo",
    "core_universe_snapshot",
    "core_universe_policy",
    "model_registry",
    "feature_store_snapshot",
]

DROP_ORDER = [
    "predictions",
    "prediction_run",
    "universe_completeness_snapshot",
]


class UniverseCompletenessSchemaManager:
    def __init__(self):
        self.constitution_ver = CONSTITUTION_VER
        self.tool_ver = TOOL_VER
        self.stats = {"success": 0, "failed": 0, "warning": 0, "details": []}
        self.preflight = {"pass": 0, "failed": 0, "warning": 0, "details": []}

    def _detail(self, bucket, message):
        self.stats[bucket] += 1
        icon = {"success": "✅", "warning": "⚠️", "failed": "❌"}[bucket]
        self.stats["details"].append(f"{icon} {message}")

    def _preflight_detail(self, bucket, message):
        self.preflight[bucket] += 1
        icon = {"pass": "✅", "warning": "⚠️", "failed": "❌"}[bucket]
        self.preflight["details"].append(f"{icon} [PREFLIGHT-{bucket.upper()}] {message}")

    def _mark_lifecycle(self, lifecycle, level, message):
        if lifecycle is None:
            return
        method_name = "mark_failed" if level == "failed" else "mark_warning"
        marker = getattr(lifecycle, method_name, None)
        if callable(marker):
            marker(message)

    def _quote_columns(self, columns):
        return ", ".join([f'"{column}"' for column in columns])

    def _constraint_name(self, prefix, table_name, columns_or_suffix):
        if isinstance(columns_or_suffix, list):
            suffix = "_".join(columns_or_suffix)
        else:
            suffix = str(columns_or_suffix)
        base = f"{prefix}_{table_name}_{suffix}".lower()
        return base[:60]

    def _table_exists(self, cur, table_name):
        cur.execute("SELECT to_regclass(%s);", (f'public."{table_name}"',))
        return cur.fetchone()[0] is not None

    def preflight_check(self):
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            prerequisites_ok = True
            for table_name in PREREQUISITE_TABLES:
                if self._table_exists(cur, table_name):
                    self._preflight_detail("pass", f"{table_name} exists")
                else:
                    prerequisites_ok = False
                    self._preflight_detail(
                        "failed",
                        f"{table_name} missing; run data_schema.py / core_universe_schema.py / feature_store_schema.py --init first",
                    )
        finally:
            cur.close()
            conn.close()
        return prerequisites_ok

    def _create_table(self, cur, table_name, config):
        cols_def = ", ".join([f'"{column}" {definition}' for column, definition in config["columns"].items()])
        pk = config.get("primary_key")
        if pk:
            cols_def += f', PRIMARY KEY ({self._quote_columns(pk)})'
        cur.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({cols_def})')

    def _apply_unique_constraints(self, cur, table_name, config):
        for columns in config.get("unique_constraints", []):
            constraint_name = self._constraint_name("uq", table_name, columns)
            cur.execute(f'ALTER TABLE "{table_name}" DROP CONSTRAINT IF EXISTS "{constraint_name}"')
            cur.execute(
                f'ALTER TABLE "{table_name}" ADD CONSTRAINT "{constraint_name}" UNIQUE ({self._quote_columns(columns)})'
            )

    def _apply_check_constraints(self, cur, table_name, config):
        for constraint_name, expression in config.get("check_constraints", []):
            cur.execute(f'ALTER TABLE "{table_name}" DROP CONSTRAINT IF EXISTS "{constraint_name}"')
            cur.execute(
                f'ALTER TABLE "{table_name}" ADD CONSTRAINT "{constraint_name}" CHECK ({expression})'
            )

    def _apply_foreign_keys(self, cur, table_name, config):
        for fk in config.get("foreign_keys", []):
            columns = fk["columns"]
            constraint_name = self._constraint_name("fk", table_name, columns)
            ref_cols = self._quote_columns(fk["ref_columns"])
            on_delete = fk.get("on_delete", "NO ACTION")
            cur.execute(f'ALTER TABLE "{table_name}" DROP CONSTRAINT IF EXISTS "{constraint_name}"')
            cur.execute(
                f'ALTER TABLE "{table_name}" ADD CONSTRAINT "{constraint_name}" '
                f'FOREIGN KEY ({self._quote_columns(columns)}) '
                f'REFERENCES "{fk["ref_table"]}" ({ref_cols}) ON DELETE {on_delete}'
            )

    def _apply_indexes(self, cur, table_name, config):
        for index_name, columns in config.get("indexes", []):
            cur.execute(
                f'CREATE INDEX IF NOT EXISTS "{index_name}" ON "{table_name}" ({self._quote_columns(columns)})'
            )

    def _create_materialized_view(self, cur):
        cur.execute(f'DROP MATERIALIZED VIEW IF EXISTS "{MATERIALIZED_VIEW_NAME}"')
        cur.execute(MATERIALIZED_VIEW_DDL)
        idx_name, idx_columns = MATERIALIZED_VIEW_UNIQUE_INDEX
        cur.execute(
            f'CREATE UNIQUE INDEX IF NOT EXISTS "{idx_name}" '
            f'ON "{MATERIALIZED_VIEW_NAME}" ({self._quote_columns(idx_columns)})'
        )

    def init_tables(self, target_table=None, force=False, skip_preflight=False):
        start_time = time.time()
        tables = [target_table] if target_table else list(UNIVERSE_COMPLETENESS_REGISTRY.keys())
        ddl_executed = False

        with record_lifecycle("universe_completeness_schema_init_v0.1", category="schema", stock_id="SYSTEM") as lifecycle:
            if target_table and target_table not in UNIVERSE_COMPLETENESS_REGISTRY:
                msg = f'表名 "{target_table}" 未登錄於 UNIVERSE_COMPLETENESS_REGISTRY'
                self._detail("failed", msg)
                self._mark_lifecycle(lifecycle, "failed", msg)
                self.report_results(start_time, ddl_executed=False)
                return False

            if skip_preflight:
                msg = "--skip-preflight used; universe_completeness DDL executed without prerequisite validation"
                self._preflight_detail("warning", msg)
                self._mark_lifecycle(lifecycle, "warning", msg)
            elif not self.preflight_check():
                for line in self.preflight["details"]:
                    if "FAILED" in line:
                        self._mark_lifecycle(lifecycle, "failed", line)
                self._detail("failed", "前置表驗證失敗;已停止 universe_completeness DDL")
                self.report_results(start_time, ddl_executed=False)
                return False

            conn = get_db_connection()
            cur = conn.cursor()
            try:
                print("🛠️  正在啟動 universe_completeness 治理資料層初始化程序...")
                ddl_executed = True

                if force:
                    cur.execute(f'DROP MATERIALIZED VIEW IF EXISTS "{MATERIALIZED_VIEW_NAME}"')
                    drop_targets = DROP_ORDER if not target_table else [target_table]
                    for table_name in drop_targets:
                        cur.execute(f'DROP TABLE IF EXISTS "{table_name}" CASCADE')

                for table_name in tables:
                    config = UNIVERSE_COMPLETENESS_REGISTRY[table_name]
                    self._create_table(cur, table_name, config)

                for table_name in tables:
                    config = UNIVERSE_COMPLETENESS_REGISTRY[table_name]
                    self._apply_unique_constraints(cur, table_name, config)
                    self._apply_check_constraints(cur, table_name, config)

                for table_name in tables:
                    config = UNIVERSE_COMPLETENESS_REGISTRY[table_name]
                    self._apply_foreign_keys(cur, table_name, config)
                    self._apply_indexes(cur, table_name, config)

                if not target_table:
                    self._create_materialized_view(cur)
                    self._detail("success", f'物化視圖: "{MATERIALIZED_VIEW_NAME}" - reporting 層封印完成')

                conn.commit()

                for table_name in tables:
                    self._detail("success", f'表名: "{table_name}" - universe_completeness 容器封印完成')
                    try:
                        write_data_audit_log(
                            table_name,
                            "SYSTEM",
                            datetime.now().strftime("%Y-%m-%d"),
                            "UNIVERSE_COMPLETENESS_SCHEMA_INIT",
                            1,
                        )
                    except Exception as exc:
                        msg = f'{table_name} data_audit_log 寫入失敗: {type(exc).__name__}: {exc}'
                        self._detail("warning", msg)
                        self._mark_lifecycle(lifecycle, "warning", msg)

            except Exception as exc:
                conn.rollback()
                msg = f"universe_completeness DDL 失敗: {type(exc).__name__}: {exc}"
                self._detail("failed", msg)
                self._mark_lifecycle(lifecycle, "failed", msg)
            finally:
                cur.close()
                conn.close()

            self.report_results(start_time, ddl_executed=ddl_executed)
            return self.stats["failed"] == 0 and self.preflight["failed"] == 0

    def refresh_view(self, concurrently=True):
        start_time = time.time()
        with record_lifecycle("universe_completeness_view_refresh_v0.1", category="schema", stock_id="SYSTEM") as lifecycle:
            conn = get_db_connection()
            cur = conn.cursor()
            try:
                if not self._table_exists(cur, MATERIALIZED_VIEW_NAME):
                    msg = f'物化視圖 "{MATERIALIZED_VIEW_NAME}" 不存在;先執行 --init'
                    self._detail("failed", msg)
                    self._mark_lifecycle(lifecycle, "failed", msg)
                    self.report_results(start_time, ddl_executed=False)
                    return False

                mode = "CONCURRENTLY" if concurrently else ""
                try:
                    cur.execute(f'REFRESH MATERIALIZED VIEW {mode} "{MATERIALIZED_VIEW_NAME}"')
                except Exception as exc:
                    conn.rollback()
                    msg = f"CONCURRENTLY refresh 失敗,fallback 為阻塞式 refresh: {type(exc).__name__}: {exc}"
                    self._detail("warning", msg)
                    self._mark_lifecycle(lifecycle, "warning", msg)
                    cur.execute(f'REFRESH MATERIALIZED VIEW "{MATERIALIZED_VIEW_NAME}"')

                conn.commit()
                self._detail("success", f'物化視圖 "{MATERIALIZED_VIEW_NAME}" 已 refresh')
            except Exception as exc:
                conn.rollback()
                msg = f"refresh 失敗: {type(exc).__name__}: {exc}"
                self._detail("failed", msg)
                self._mark_lifecycle(lifecycle, "failed", msg)
            finally:
                cur.close()
                conn.close()

            self.report_results(start_time, ddl_executed=True)
            return self.stats["failed"] == 0

    def compute_verdict(self):
        if self.stats["failed"] > 0 or self.preflight["failed"] > 0:
            return "FAILED"
        if self.stats["warning"] > 0 or self.preflight["warning"] > 0:
            return "WARNING"
        return "PERFECT"

    def report_results(self, start_time, ddl_executed=True):
        print("\n" + "🛡️" * 40)
        print(f"🚀 Quantum Finance: Universe Completeness 治理資料層報告 ({self.tool_ver})")
        print("🛡️" * 40)
        print(f"治權基準 : 系統架構大憲章_{self.constitution_ver}.md §14.7-BU Phase C+D")
        print("治理權責 : Universe Completeness Schema Authority")
        print("邊界封印 : prediction_run + predictions + universe_completeness_snapshot + matview only; 既有 28 tables 零變動")
        print("─" * 80)
        for line in self.preflight["details"]:
            print(line)
        print("─" * 80)
        for line in self.stats["details"]:
            print(line)
        print("─" * 80)
        print(f"🔎 PREFLIGHT PASS/WARN/FAIL : {self.preflight['pass']}/{self.preflight['warning']}/{self.preflight['failed']}")
        print(f"📈 治理表總數 : {len(UNIVERSE_COMPLETENESS_REGISTRY)} tables + 1 materialized view")
        print(f"✅ 成功項目   : {self.stats['success']}")
        print(f"⚠️  警告項目   : {self.stats['warning']}")
        print(f"❌ 失敗項目   : {self.stats['failed']}")
        print(f"🧱 DDL 執行   : {'YES' if ddl_executed else 'NO'}")
        print(f"🕒 總計耗時   : {(time.time() - start_time)*1000:.2f} ms")
        print(f"⚖️  主權判定   : {self.compute_verdict()}")
        print("🛡️" * 40 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Quantum Finance Universe Completeness 治理資料層工具 (v0.1)")
    parser.add_argument("--init", action="store_true", help="啟動 universe_completeness 治理層初始化（3 tables + 1 view）")
    parser.add_argument("--force", action="store_true", help="強制重置現有 universe_completeness 表 + 物化視圖")
    parser.add_argument("--table", type=str, help="指定單一治理表名（跳過 materialized view 建立）")
    parser.add_argument("--skip-preflight", action="store_true", help="離線/災難復原:略過前置表檢查")
    parser.add_argument("--refresh-view", action="store_true", help="手動 refresh universe_completeness_matrix_current（CONCURRENTLY）")
    args = parser.parse_args()

    manager = UniverseCompletenessSchemaManager()
    if args.refresh_view:
        ok = manager.refresh_view()
        sys.exit(0 if ok else 1)
    if args.init:
        ok = manager.init_tables(target_table=args.table, force=args.force, skip_preflight=args.skip_preflight)
        sys.exit(0 if ok else 1)
    parser.print_help()


if __name__ == "__main__":
    main()
