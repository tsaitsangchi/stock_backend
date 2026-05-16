"""
core_universe_schema.py v0.2 (Quantum Finance Derived Governance Schema Authority)
================================================================================
**最後更新日期**: 2026-05-16
**主權狀態**: IMPLEMENTED (憲法 v5.4.22 核心股治理分層對齊 + bootstrap preflight clarity)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
1. [Derived Governance Schema Authority]: 本工具只建立核心股治理資料層，不管理 FinMind/FRED raw API schema。
2. [Raw API Separation]: `data_schema.py` 是 Raw API Schema Authority；本工具不得取代 API contract probe。
3. [Column Inheritance]: 凡保存 raw API 欄位語意，欄位名稱大小寫與 SQL 型別寬度必須繼承 `DATASET_REGISTRY`。
4. [Downstream Bridge Boundary]: 預留 feature/model/prediction/backtest eligibility 與 version 欄位，但不得保存實際 feature / label / prediction values。
5. [Snapshot Governance]: Universe 必須以 snapshot_id / as_of_date / policy_version 版本化，避免未來函數與倖存者偏誤。
6. [Hybrid Observability]: 建表行為必須接入 record_lifecycle 與 data_audit_log，並輸出完整終端摘要。

## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix)
| 維運需求場景 (Scenario) | 權威指令 / 建議用法 (Exhaustive Examples) | 對齊模組 |
| :--- | :--- | :--- |
| **1. [核心股治理層初始化]** | `$ python scripts/core/core_universe_schema.py --init` | core_universe_schema v0.2 |
| **2. [核心股治理層強制重鑄]** | `$ python scripts/core/core_universe_schema.py --init --force` | core_universe_schema v0.2 |
| **3. [單一治理表重鑄]** | `$ python scripts/core/core_universe_schema.py --init --table core_universe_membership` | core_universe_schema v0.2 |
| **4. [離線災難復原]** | `$ python scripts/core/core_universe_schema.py --init --skip-preflight` | core_universe_schema v0.2 |

## 📜 三、全修訂歷程 (Full Revision History)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **v0.2** | 2026-05-14 | Codex | **欄位繼承對齊版 + 2026-05-16 bootstrap preflight clarity**：依憲章 Derived Schema 欄位繼承原則，將 `market` 修正為 raw API 欄位 `type`；新增 `RAW_COLUMN_INHERITANCE` 與 preflight 型別相容檢查，確保 derived table 引用 raw 欄位時名稱大小寫與 SQL 型別繼承 `DATASET_REGISTRY`。若 raw / infra prerequisite tables 尚未建立，preflight 必須回報明確 WARNING 並跳過繼承驗證，不得輸出誤導性的 PASS。 | **ACTIVE** |
| v0.1 | 2026-05-14 | Codex | **核心股治理資料層初始版**：依憲章 v5.4.19 建立 Derived Governance Schema；新增 policy、snapshot、membership、scores、theme taxonomy、stock-theme map、revision log；只保存治理銜接欄位，不保存實際 feature/label/prediction values。 | SUPERSEDED |
================================================================================
"""
import argparse
import sys
import time
import re
from pathlib import Path
from datetime import datetime

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


CORE_UNIVERSE_REGISTRY = {
    "core_universe_policy": {
        "columns": {
            "policy_version": "VARCHAR(255) PRIMARY KEY",
            "policy_name": "VARCHAR(255) NOT NULL",
            "description": "TEXT",
            "weight_config": "JSONB",
            "eligibility_config": "JSONB",
            "risk_config": "JSONB",
            "effective_from": "DATE",
            "effective_to": "DATE",
            "active": "BOOLEAN DEFAULT TRUE",
            "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            "notes": "TEXT",
        },
        "unique_constraints": [],
        "indexes": [
            ("idx_core_universe_policy_active", ["active"]),
            ("idx_core_universe_policy_effective", ["effective_from", "effective_to"]),
        ],
    },
    "core_universe_snapshot": {
        "columns": {
            "snapshot_id": "VARCHAR(255) PRIMARY KEY",
            "as_of_date": "DATE NOT NULL",
            "source_data_cutoff": "DATE",
            "policy_version": "VARCHAR(255)",
            "feature_set_version": "VARCHAR(255)",
            "model_policy_version": "VARCHAR(255)",
            "prediction_policy_version": "VARCHAR(255)",
            "total_candidates": "INTEGER DEFAULT 0",
            "research_count": "INTEGER DEFAULT 0",
            "core_count": "INTEGER DEFAULT 0",
            "convex_count": "INTEGER DEFAULT 0",
            "quarantine_count": "INTEGER DEFAULT 0",
            "status": "VARCHAR(255) DEFAULT 'draft'",
            "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            "notes": "TEXT",
        },
        "unique_constraints": [["as_of_date", "policy_version"]],
        "foreign_keys": [
            {
                "columns": ["policy_version"],
                "ref_table": "core_universe_policy",
                "ref_columns": ["policy_version"],
                "on_delete": "SET NULL",
            }
        ],
        "indexes": [
            ("idx_core_universe_snapshot_as_of", ["as_of_date"]),
            ("idx_core_universe_snapshot_policy", ["policy_version"]),
            ("idx_core_universe_snapshot_status", ["status"]),
        ],
    },
    "core_universe_membership": {
        "columns": {
            "snapshot_id": "VARCHAR(255) NOT NULL",
            "stock_id": "VARCHAR(255) NOT NULL",
            "stock_name": "VARCHAR(255)",
            "type": "VARCHAR(255)",
            "industry_category": "VARCHAR(255)",
            "core_tier": "VARCHAR(255) NOT NULL",
            "core_score": "NUMERIC(20, 6)",
            "selected_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            "effective_from": "DATE",
            "effective_to": "DATE",
            "review_cycle": "VARCHAR(255)",
            "active": "BOOLEAN DEFAULT TRUE",
            "selection_reason": "TEXT",
            "exclusion_reason": "TEXT",
            "train_eligible": "BOOLEAN DEFAULT FALSE",
            "predict_eligible": "BOOLEAN DEFAULT FALSE",
            "backtest_eligible": "BOOLEAN DEFAULT FALSE",
            "downstream_ready": "BOOLEAN DEFAULT FALSE",
            "min_history_days": "INTEGER",
            "price_coverage_252d": "NUMERIC(20, 6)",
            "revenue_coverage_24m": "NUMERIC(20, 6)",
            "financial_coverage_8q": "NUMERIC(20, 6)",
            "label_horizon": "INTEGER",
            "policy_version": "VARCHAR(255)",
            "feature_set_version": "VARCHAR(255)",
            "model_policy_version": "VARCHAR(255)",
            "prediction_policy_version": "VARCHAR(255)",
            "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        },
        "unique_constraints": [["snapshot_id", "stock_id"]],
        "foreign_keys": [
            {
                "columns": ["snapshot_id"],
                "ref_table": "core_universe_snapshot",
                "ref_columns": ["snapshot_id"],
                "on_delete": "CASCADE",
            }
        ],
        "indexes": [
            ("idx_core_universe_membership_stock", ["stock_id"]),
            ("idx_core_universe_membership_tier", ["core_tier"]),
            ("idx_core_universe_membership_active", ["active"]),
            ("idx_core_universe_membership_train", ["train_eligible"]),
            ("idx_core_universe_membership_predict", ["predict_eligible"]),
        ],
    },
    "core_universe_scores": {
        "columns": {
            "snapshot_id": "VARCHAR(255) NOT NULL",
            "stock_id": "VARCHAR(255) NOT NULL",
            "as_of_date": "DATE NOT NULL",
            "source_data_cutoff": "DATE",
            "policy_version": "VARCHAR(255)",
            "core_score": "NUMERIC(20, 6)",
            "data_quality_score": "NUMERIC(20, 6)",
            "liquidity_score": "NUMERIC(20, 6)",
            "fundamental_score": "NUMERIC(20, 6)",
            "theme_score": "NUMERIC(20, 6)",
            "institutional_flow_score": "NUMERIC(20, 6)",
            "volatility_control_score": "NUMERIC(20, 6)",
            "risk_penalty": "NUMERIC(20, 6)",
            "score_detail": "JSONB",
            "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        },
        "unique_constraints": [["snapshot_id", "stock_id"]],
        "foreign_keys": [
            {
                "columns": ["snapshot_id"],
                "ref_table": "core_universe_snapshot",
                "ref_columns": ["snapshot_id"],
                "on_delete": "CASCADE",
            }
        ],
        "indexes": [
            ("idx_core_universe_scores_stock", ["stock_id"]),
            ("idx_core_universe_scores_as_of", ["as_of_date"]),
            ("idx_core_universe_scores_core_score", ["core_score"]),
        ],
    },
    "theme_taxonomy": {
        "columns": {
            "theme_code": "VARCHAR(255) PRIMARY KEY",
            "theme_name": "VARCHAR(255) NOT NULL",
            "theme_group": "VARCHAR(255)",
            "description": "TEXT",
            "parent_theme_code": "VARCHAR(255)",
            "active": "BOOLEAN DEFAULT TRUE",
            "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        },
        "unique_constraints": [],
        "foreign_keys": [
            {
                "columns": ["parent_theme_code"],
                "ref_table": "theme_taxonomy",
                "ref_columns": ["theme_code"],
                "on_delete": "SET NULL",
            }
        ],
        "indexes": [
            ("idx_theme_taxonomy_group", ["theme_group"]),
            ("idx_theme_taxonomy_active", ["active"]),
        ],
    },
    "stock_theme_map": {
        "columns": {
            "stock_id": "VARCHAR(255) NOT NULL",
            "theme_code": "VARCHAR(255) NOT NULL",
            "confidence_score": "NUMERIC(20, 6)",
            "mapping_source": "VARCHAR(255)",
            "effective_from": "DATE",
            "effective_to": "DATE",
            "active": "BOOLEAN DEFAULT TRUE",
            "version": "VARCHAR(255) NOT NULL DEFAULT 'v0.2'",
            "notes": "TEXT",
            "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        },
        "unique_constraints": [["stock_id", "theme_code", "version"]],
        "foreign_keys": [
            {
                "columns": ["theme_code"],
                "ref_table": "theme_taxonomy",
                "ref_columns": ["theme_code"],
                "on_delete": "CASCADE",
            }
        ],
        "indexes": [
            ("idx_stock_theme_map_stock", ["stock_id"]),
            ("idx_stock_theme_map_theme", ["theme_code"]),
            ("idx_stock_theme_map_active", ["active"]),
        ],
    },
    "universe_revision_log": {
        "columns": {
            "revision_id": "SERIAL PRIMARY KEY",
            "revision_time": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            "actor": "VARCHAR(255)",
            "action_type": "VARCHAR(255) NOT NULL",
            "object_type": "VARCHAR(255)",
            "object_id": "VARCHAR(255)",
            "policy_version": "VARCHAR(255)",
            "snapshot_id": "VARCHAR(255)",
            "detail": "JSONB",
            "note": "TEXT",
        },
        "unique_constraints": [],
        "foreign_keys": [
            {
                "columns": ["snapshot_id"],
                "ref_table": "core_universe_snapshot",
                "ref_columns": ["snapshot_id"],
                "on_delete": "SET NULL",
            }
        ],
        "indexes": [
            ("idx_universe_revision_log_time", ["revision_time"]),
            ("idx_universe_revision_log_snapshot", ["snapshot_id"]),
            ("idx_universe_revision_log_action", ["action_type"]),
        ],
    },
}

RAW_COLUMN_INHERITANCE = {
    "core_universe_membership": {
        "stock_id": ("TaiwanStockInfo", "stock_id"),
        "stock_name": ("TaiwanStockInfo", "stock_name"),
        "type": ("TaiwanStockInfo", "type"),
        "industry_category": ("TaiwanStockInfo", "industry_category"),
    },
    "core_universe_scores": {
        "stock_id": ("TaiwanStockInfo", "stock_id"),
    },
    "stock_theme_map": {
        "stock_id": ("TaiwanStockInfo", "stock_id"),
    },
}

PREREQUISITE_TABLES = ["pipeline_execution_log", "data_audit_log", "TaiwanStockInfo"]
DROP_ORDER = [
    "universe_revision_log",
    "stock_theme_map",
    "theme_taxonomy",
    "core_universe_scores",
    "core_universe_membership",
    "core_universe_snapshot",
    "core_universe_policy",
]


class CoreUniverseSchemaManager:
    def __init__(self):
        self.constitution_ver = "v5.4.22"
        self.tool_ver = "v0.2"
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
            table_config = CORE_UNIVERSE_REGISTRY.get(table_name)
            if not table_config:
                self._preflight_detail("failed", f"inheritance table missing in CORE_UNIVERSE_REGISTRY: {table_name}")
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
                    self._preflight_detail("failed", f"{table_name} missing; run data_schema.py --init first")
        finally:
            cur.close()
            conn.close()
        if not prerequisites_ok:
            self._preflight_detail(
                "warning",
                "RAW_COLUMN_INHERITANCE validation skipped because prerequisite raw/infra tables are missing",
            )
            return False
        self.validate_column_inheritance()
        return self.preflight["failed"] == 0

    def _create_table(self, cur, table_name, config):
        cols_def = ", ".join([f'"{column}" {definition}' for column, definition in config["columns"].items()])
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
        tables = [target_table] if target_table else list(CORE_UNIVERSE_REGISTRY.keys())
        ddl_executed = False

        with record_lifecycle("core_universe_schema_init_v0.2", category="schema", stock_id="SYSTEM") as lifecycle:
            if target_table and target_table not in CORE_UNIVERSE_REGISTRY:
                msg = f'表名 "{target_table}" 未登錄於 CORE_UNIVERSE_REGISTRY'
                self._detail("failed", msg)
                self._mark_lifecycle(lifecycle, "failed", msg)
                self.report_results(start_time, ddl_executed=False)
                return False

            if skip_preflight:
                msg = "--skip-preflight used; governance DDL executed without prerequisite validation"
                self._preflight_detail("warning", msg)
                self._mark_lifecycle(lifecycle, "warning", msg)
            elif not self.preflight_check():
                for line in self.preflight["details"]:
                    if "FAILED" in line:
                        self._mark_lifecycle(lifecycle, "failed", line)
                self._detail("failed", "前置表驗證失敗；已停止核心股治理 DDL")
                self.report_results(start_time, ddl_executed=False)
                return False

            conn = get_db_connection()
            cur = conn.cursor()
            try:
                print("🛠️  正在啟動核心股治理資料層初始化程序...")
                ddl_executed = True

                if force:
                    drop_targets = DROP_ORDER if not target_table else [target_table]
                    for table_name in drop_targets:
                        cur.execute(f'DROP TABLE IF EXISTS "{table_name}" CASCADE')

                for table_name in tables:
                    config = CORE_UNIVERSE_REGISTRY[table_name]
                    self._create_table(cur, table_name, config)

                for table_name in tables:
                    config = CORE_UNIVERSE_REGISTRY[table_name]
                    self._apply_unique_constraints(cur, table_name, config)

                for table_name in tables:
                    config = CORE_UNIVERSE_REGISTRY[table_name]
                    self._apply_foreign_keys(cur, table_name, config)
                    self._apply_indexes(cur, table_name, config)

                conn.commit()

                for table_name in tables:
                    self._detail("success", f'表名: "{table_name}" - 核心股治理容器封印完成')
                    try:
                        write_data_audit_log(
                            table_name,
                            "SYSTEM",
                            datetime.now().strftime("%Y-%m-%d"),
                            "CORE_UNIVERSE_SCHEMA_INIT",
                            1,
                        )
                    except Exception as exc:
                        msg = f'{table_name} data_audit_log 寫入失敗: {type(exc).__name__}: {exc}'
                        self._detail("warning", msg)
                        self._mark_lifecycle(lifecycle, "warning", msg)

            except Exception as exc:
                conn.rollback()
                msg = f"核心股治理 DDL 失敗: {type(exc).__name__}: {exc}"
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
        print(f"🚀 Quantum Finance: 核心股治理資料層初始化報告 ({self.tool_ver})")
        print("🛡️" * 40)
        print(f"治權基準 : 系統架構大憲章_{self.constitution_ver}.md")
        print("治理權責 : Derived Governance Schema Authority")
        print("邊界封印 : raw column inheritance + eligibility/version/cutoff/coverage only; no feature/label/prediction values")
        print("─" * 80)
        for line in self.preflight["details"]:
            print(line)
        print("─" * 80)
        for line in self.stats["details"]:
            print(line)
        print("─" * 80)
        print(f"🔎 PREFLIGHT PASS/WARN/FAIL : {self.preflight['pass']}/{self.preflight['warning']}/{self.preflight['failed']}")
        print(f"📈 治理表總數 : {len(CORE_UNIVERSE_REGISTRY)}")
        print(f"✅ 成功項目   : {self.stats['success']}")
        print(f"⚠️  警告項目   : {self.stats['warning']}")
        print(f"❌ 失敗項目   : {self.stats['failed']}")
        print(f"🧱 DDL 執行   : {'YES' if ddl_executed else 'NO'}")
        print(f"🕒 總計耗時   : {(time.time() - start_time)*1000:.2f} ms")
        print(f"⚖️  主權判定   : {self.compute_verdict()}")
        print("🛡️" * 40 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Quantum Finance 核心股治理資料層工具 (v0.2)")
    parser.add_argument("--init", action="store_true", help="啟動核心股治理資料層初始化")
    parser.add_argument("--force", action="store_true", help="強制重置現有核心股治理表")
    parser.add_argument("--table", type=str, help="指定單一治理表名")
    parser.add_argument("--skip-preflight", action="store_true", help="離線/災難復原：略過 data_schema 前置表檢查")
    args = parser.parse_args()

    manager = CoreUniverseSchemaManager()
    if args.init:
        ok = manager.init_tables(target_table=args.table, force=args.force, skip_preflight=args.skip_preflight)
        sys.exit(0 if ok else 1)
    parser.print_help()


if __name__ == "__main__":
    main()
