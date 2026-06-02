"""
feature_store_schema.py v0.1 (Quantum Finance Feature Store Schema Authority)
================================================================================
**最後更新日期**: 2026-05-16
**主權狀態**: IMPLEMENTED (憲法 v6.0.0 §8.2 下游治理草案 v0.1 對齊)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:建立**特徵庫**的資料表 DDL(feature_values / feature_set)。

**輸入 → 輸出**:(無)→ feature store 表

**為什麼需要它**:特徵計算結果要有地方存,這支建那些表。

## 📜 一、核心定義說明 (Core Definitions)
1. [Feature Store Schema Authority]: 本工具只建立 §8.2 三張 feature_store_* 表，不管理 raw API schema、不管理 core_universe_* 治理表、不管理 model/prediction 表。
2. [Downstream Boundary]: 對齊憲章 §8.1 三層職責邊界 — 不得保存 labels、不得保存 model output。
3. [As-of-Strict]: 所有特徵值僅得由 builder 以 `WHERE date <= as_of_date` 嚴格過濾後寫入（§8.5）。
4. [Column Inheritance]: `feature_values.stock_id` 必須繼承 `TaiwanStockInfo.stock_id` 之 SQL 型別寬度。
5. [Snapshot Governance]: Feature Set 必須以 `feature_set_id` / `feature_set_version` / `as_of_date` 版本化。
6. [Hybrid Observability]: 建表行為必須接入 record_lifecycle 與 data_audit_log，並輸出完整終端摘要。
7. [Zero Hardcoded Verdict]: 主權判定必須動態計算 (§5.6.3)。
8. **[Sovereignty Declaration]** (2026-05-29 §一.11 補入, 憲法 §3.2 橫切 schema 模組 / §8.2 Feature Store): 本程式為 **§8.2 Feature Store 之 3 governance tables(feature_store_snapshot / feature_definition / feature_values)唯一 schema 建立載體**(§3.2 橫切;與 data_schema.py / core_universe_schema.py 並列)。**治權邊界**:(a) §3.2 橫切 schema;(b) **不管理 raw API schema**(由 data_schema.py 負責);(c) **不管理 core_universe_* 表**(由 core_universe_schema.py 負責);(d) **不管理 model / prediction 表**(由 data_schema.py / universe_completeness_schema.py 負責);(e) **不計算 features**(由 feature_store_builder.py 負責);(f) **不選股**;(g) 唯一職責:建立 §8.2 三 governance tables + 欄位繼承 + as-of-strict 治權邊界 enforcement。

## 📊 二、全量維運指令總矩陣
| 場景 | 指令 | 對齊 |
| :--- | :--- | :--- |
| 1. 初始化 feature_store 治理層 | `$ python scripts/core/feature_store_schema.py --init` | feature_store_schema v0.1 |
| 2. 強制重鑄 | `$ python scripts/core/feature_store_schema.py --init --force` | feature_store_schema v0.1 |
| 3. 單表重鑄 | `$ python scripts/core/feature_store_schema.py --init --table feature_values` | feature_store_schema v0.1 |
| 4. 離線復原 | `$ python scripts/core/feature_store_schema.py --init --skip-preflight` | feature_store_schema v0.1 |

## 📜 三、全修訂歷程 (Full Revision History)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.1-§一.11 | 2026-05-29 | Codex | **§一.11 三段式標頭規範對齊**:(I) §一 補入 [Sovereignty Declaration] 治權邊界宣告(原 7 條 → 8 條);(II) §三 標題從「修訂歷程」改為「全修訂歷程 (Full Revision History)」對齊 CLAUDE.md §一.11 強制格式。原 v0.1 邏輯不變。 | **ACTIVE** |
| **v0.1** | 2026-05-16 | Codex | **Feature Store 治理層初版**：依憲章 §8.2.1 建立 feature_store_snapshot / feature_definition / feature_values 三表，並落地欄位繼承與 as-of-strict 治權邊界。 | **ACTIVE (DRAFT)** |
================================================================================
"""
import argparse
import re
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
    from core.data_schema import DATASET_REGISTRY
except ImportError as exc:
    print(f"❌ 核心組件導入失敗，請確認 core/ 目錄: {exc}")
    sys.exit(1)


FEATURE_STORE_REGISTRY = {
    "feature_store_snapshot": {
        "columns": {
            "feature_set_id": "VARCHAR(255) PRIMARY KEY",
            "feature_set_version": "VARCHAR(255) NOT NULL",
            "as_of_date": "DATE NOT NULL",
            "source_data_cutoff": "DATE NOT NULL",
            "universe_snapshot_id": "VARCHAR(255) NOT NULL",
            "policy_version": "VARCHAR(255) NOT NULL",
            "total_stocks": "INTEGER NOT NULL DEFAULT 0",
            "feature_count": "INTEGER NOT NULL DEFAULT 0",
            "label_horizon": "INTEGER NOT NULL DEFAULT 20",
            "status": "VARCHAR(64) NOT NULL DEFAULT 'draft'",
            "notes": "TEXT",
            "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        },
        "unique_constraints": [["as_of_date", "feature_set_version"]],
        "foreign_keys": [
            {
                "columns": ["universe_snapshot_id"],
                "ref_table": "core_universe_snapshot",
                "ref_columns": ["snapshot_id"],
                "on_delete": "RESTRICT",
            },
            {
                "columns": ["policy_version"],
                "ref_table": "core_universe_policy",
                "ref_columns": ["policy_version"],
                "on_delete": "SET NULL",
            },
        ],
        "indexes": [
            ("idx_feature_store_snapshot_status", ["status"]),
            ("idx_feature_store_snapshot_as_of", ["as_of_date"]),
        ],
    },
    "feature_definition": {
        "columns": {
            "feature_set_id": "VARCHAR(255) NOT NULL",
            "feature_name": "VARCHAR(255) NOT NULL",
            "feature_group": "VARCHAR(64) NOT NULL",
            "source_table": "VARCHAR(255) NOT NULL",
            "derivation_window": "VARCHAR(64) NOT NULL",
            "value_type": "VARCHAR(32) NOT NULL",
            "null_strategy": "VARCHAR(64) NOT NULL",
            "as_of_strict": "BOOLEAN NOT NULL DEFAULT TRUE",
            "description": "TEXT",
            "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        },
        "primary_key": ["feature_set_id", "feature_name"],
        "foreign_keys": [
            {
                "columns": ["feature_set_id"],
                "ref_table": "feature_store_snapshot",
                "ref_columns": ["feature_set_id"],
                "on_delete": "CASCADE",
            }
        ],
        "indexes": [
            ("idx_feature_definition_group", ["feature_set_id", "feature_group"]),
        ],
    },
    "feature_values": {
        "columns": {
            "feature_set_id": "VARCHAR(255) NOT NULL",
            "stock_id": "VARCHAR(255) NOT NULL",
            "as_of_date": "DATE NOT NULL",
            "feature_name": "VARCHAR(255) NOT NULL",
            "feature_value": "NUMERIC(24, 8)",
            "is_null_imputed": "BOOLEAN NOT NULL DEFAULT FALSE",
            "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        },
        "primary_key": ["feature_set_id", "stock_id", "as_of_date", "feature_name"],
        "foreign_keys": [
            {
                "columns": ["feature_set_id"],
                "ref_table": "feature_store_snapshot",
                "ref_columns": ["feature_set_id"],
                "on_delete": "CASCADE",
            }
        ],
        "indexes": [
            ("idx_feature_values_stock_date", ["stock_id", "as_of_date"]),
            ("idx_feature_values_feature_name", ["feature_set_id", "feature_name"]),
        ],
    },
}

RAW_COLUMN_INHERITANCE = {
    "feature_values": {
        "stock_id": ("TaiwanStockInfo", "stock_id"),
    },
}

PREREQUISITE_TABLES = [
    "pipeline_execution_log",
    "data_audit_log",
    "TaiwanStockInfo",
    "core_universe_snapshot",
    "core_universe_policy",
]
DROP_ORDER = [
    "feature_values",
    "feature_definition",
    "feature_store_snapshot",
]


class FeatureStoreSchemaManager:
    def __init__(self):
        self.constitution_ver = "v6.0.0"
        self.tool_ver = "v0.1"
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

    def _constraint_name(self, prefix, table_name, columns):
        base = f"{prefix}_{table_name}_{'_'.join(columns)}".lower()
        return base[:60]

    def _table_exists(self, cur, table_name):
        cur.execute("SELECT to_regclass(%s);", (f'public."{table_name}"',))
        return cur.fetchone()[0] is not None

    def _parse_sized_type(self, sql_type, type_name):
        match = re.search(rf"{type_name}\((\d+)(?:,\s*(\d+))?\)", sql_type.upper())
        if not match:
            return None
        values = [int(value) for value in match.groups() if value is not None]
        return tuple(values)

    def _is_type_compatible(self, derived_type, raw_type):
        derived = derived_type.upper()
        raw = raw_type.upper()
        if raw.startswith("VARCHAR"):
            raw_size = self._parse_sized_type(raw, "VARCHAR")
            derived_size = self._parse_sized_type(derived, "VARCHAR")
            if derived.startswith("TEXT"):
                return True
            return bool(raw_size and derived_size and derived_size[0] >= raw_size[0])
        if raw.startswith("NUMERIC"):
            raw_size = self._parse_sized_type(raw, "NUMERIC")
            derived_size = self._parse_sized_type(derived, "NUMERIC")
            return bool(raw_size and derived_size and derived_size[0] >= raw_size[0] and derived_size[1] >= raw_size[1])
        if raw.startswith("DATE"):
            return derived.startswith("DATE")
        if raw.startswith("TIMESTAMP"):
            return derived.startswith("TIMESTAMP")
        return derived.split()[0] == raw.split()[0]

    def validate_column_inheritance(self):
        for table_name, mappings in RAW_COLUMN_INHERITANCE.items():
            table_config = FEATURE_STORE_REGISTRY.get(table_name)
            if not table_config:
                self._preflight_detail("failed", f"inheritance table missing in FEATURE_STORE_REGISTRY: {table_name}")
                continue
            for derived_col, (raw_table, raw_col) in mappings.items():
                derived_type = table_config["columns"].get(derived_col)
                raw_type = DATASET_REGISTRY.get(raw_table, {}).get("columns", {}).get(raw_col)
                if not derived_type:
                    self._preflight_detail("failed", f"{table_name}.{derived_col} missing; expected to inherit {raw_table}.{raw_col}")
                    continue
                if not raw_type:
                    self._preflight_detail("failed", f"raw column missing in DATASET_REGISTRY: {raw_table}.{raw_col}")
                    continue
                if not self._is_type_compatible(derived_type, raw_type):
                    self._preflight_detail(
                        "failed",
                        f"{table_name}.{derived_col} type={derived_type} incompatible with {raw_table}.{raw_col} type={raw_type}",
                    )
                    continue
                self._preflight_detail("pass", f"{table_name}.{derived_col} inherits {raw_table}.{raw_col} ({raw_type})")

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
                        f"{table_name} missing; run data_schema.py / core_universe_schema.py --init first",
                    )
        finally:
            cur.close()
            conn.close()
        if not prerequisites_ok:
            self._preflight_detail(
                "warning",
                "RAW_COLUMN_INHERITANCE validation skipped because prerequisite raw/governance tables are missing",
            )
            return False
        self.validate_column_inheritance()
        return self.preflight["failed"] == 0

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

    def init_tables(self, target_table=None, force=False, skip_preflight=False):
        start_time = time.time()
        tables = [target_table] if target_table else list(FEATURE_STORE_REGISTRY.keys())
        ddl_executed = False

        with record_lifecycle("feature_store_schema_init_v0.1", category="schema", stock_id="SYSTEM") as lifecycle:
            if target_table and target_table not in FEATURE_STORE_REGISTRY:
                msg = f'表名 "{target_table}" 未登錄於 FEATURE_STORE_REGISTRY'
                self._detail("failed", msg)
                self._mark_lifecycle(lifecycle, "failed", msg)
                self.report_results(start_time, ddl_executed=False)
                return False

            if skip_preflight:
                msg = "--skip-preflight used; feature_store DDL executed without prerequisite validation"
                self._preflight_detail("warning", msg)
                self._mark_lifecycle(lifecycle, "warning", msg)
            elif not self.preflight_check():
                for line in self.preflight["details"]:
                    if "FAILED" in line:
                        self._mark_lifecycle(lifecycle, "failed", line)
                self._detail("failed", "前置表驗證失敗；已停止 feature_store DDL")
                self.report_results(start_time, ddl_executed=False)
                return False

            conn = get_db_connection()
            cur = conn.cursor()
            try:
                print("🛠️  正在啟動 feature_store 治理資料層初始化程序...")
                ddl_executed = True

                if force:
                    drop_targets = DROP_ORDER if not target_table else [target_table]
                    for table_name in drop_targets:
                        cur.execute(f'DROP TABLE IF EXISTS "{table_name}" CASCADE')

                for table_name in tables:
                    config = FEATURE_STORE_REGISTRY[table_name]
                    self._create_table(cur, table_name, config)

                for table_name in tables:
                    config = FEATURE_STORE_REGISTRY[table_name]
                    self._apply_unique_constraints(cur, table_name, config)

                for table_name in tables:
                    config = FEATURE_STORE_REGISTRY[table_name]
                    self._apply_foreign_keys(cur, table_name, config)
                    self._apply_indexes(cur, table_name, config)

                conn.commit()

                for table_name in tables:
                    self._detail("success", f'表名: "{table_name}" - feature_store 容器封印完成')
                    try:
                        write_data_audit_log(
                            table_name,
                            "SYSTEM",
                            datetime.now().strftime("%Y-%m-%d"),
                            "FEATURE_STORE_SCHEMA_INIT",
                            1,
                        )
                    except Exception as exc:
                        msg = f'{table_name} data_audit_log 寫入失敗: {type(exc).__name__}: {exc}'
                        self._detail("warning", msg)
                        self._mark_lifecycle(lifecycle, "warning", msg)

            except Exception as exc:
                conn.rollback()
                msg = f"feature_store DDL 失敗: {type(exc).__name__}: {exc}"
                self._detail("failed", msg)
                self._mark_lifecycle(lifecycle, "failed", msg)
            finally:
                cur.close()
                conn.close()

            self.report_results(start_time, ddl_executed=ddl_executed)
            return self.stats["failed"] == 0 and self.preflight["failed"] == 0

    def compute_verdict(self):
        if self.stats["failed"] > 0 or self.preflight["failed"] > 0:
            return "FAILED"
        if self.stats["warning"] > 0 or self.preflight["warning"] > 0:
            return "WARNING"
        return "PERFECT"

    def report_results(self, start_time, ddl_executed=True):
        print("\n" + "🛡️" * 40)
        print(f"🚀 Quantum Finance: Feature Store 治理資料層初始化報告 ({self.tool_ver})")
        print("🛡️" * 40)
        print(f"治權基準 : 系統架構大憲章_{self.constitution_ver}.md §8.2")
        print("治理權責 : Feature Store Schema Authority")
        print("邊界封印 : feature_set governance + as-of-strict feature values only; no labels/model output/predictions")
        print("─" * 80)
        for line in self.preflight["details"]:
            print(line)
        print("─" * 80)
        for line in self.stats["details"]:
            print(line)
        print("─" * 80)
        print(f"🔎 PREFLIGHT PASS/WARN/FAIL : {self.preflight['pass']}/{self.preflight['warning']}/{self.preflight['failed']}")
        print(f"📈 治理表總數 : {len(FEATURE_STORE_REGISTRY)}")
        print(f"✅ 成功項目   : {self.stats['success']}")
        print(f"⚠️  警告項目   : {self.stats['warning']}")
        print(f"❌ 失敗項目   : {self.stats['failed']}")
        print(f"🧱 DDL 執行   : {'YES' if ddl_executed else 'NO'}")
        print(f"🕒 總計耗時   : {(time.time() - start_time)*1000:.2f} ms")
        print(f"⚖️  主權判定   : {self.compute_verdict()}")
        print("🛡️" * 40 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Quantum Finance Feature Store 治理資料層工具 (v0.1)")
    parser.add_argument("--init", action="store_true", help="啟動 feature_store 治理層初始化")
    parser.add_argument("--force", action="store_true", help="強制重置現有 feature_store 表")
    parser.add_argument("--table", type=str, help="指定單一治理表名")
    parser.add_argument("--skip-preflight", action="store_true", help="離線/災難復原：略過前置表檢查")
    args = parser.parse_args()

    manager = FeatureStoreSchemaManager()
    if args.init:
        ok = manager.init_tables(target_table=args.table, force=args.force, skip_preflight=args.skip_preflight)
        sys.exit(0 if ok else 1)
    parser.print_help()


if __name__ == "__main__":
    main()
