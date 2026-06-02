"""
audit_universe_completeness.py v0.1 (§14.7-BU Phase F — Cross-Layer × Cross-Pillar Universe Completeness Audit)
================================================================================
最後更新日期: 2026-05-26
主權狀態: IMPLEMENTED (憲法 v6.1.0 §14.7-BU Phase F 落地 / §0.4 數位孿生完整性 audit-only verdict / INFO-only non-blocking)
最高原則: Audit-Only Authority (讀;不寫;不作 FAIL gate;對齊 §11C 治權檢驗延伸)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:跨層×跨支柱 universe 完整度稽核(§14.7-BU Phase F)。

**輸入 → 輸出**:core_universe snapshot → 完整度 check 報告

**為什麼需要它**:驗證核心股在各資料層/支柱的完整度。

## 📜 一、核心定義說明 (Core Definitions)
1. [Audit-Only Authority]: 本工具屬 §11C 治權檢驗延伸 + L1 audit-only;只讀 universe_completeness_snapshot / universe_completeness_matrix_current / core_universe_snapshot / core_universe_membership;不寫 DB;不執行 sync / fetcher;不作 FAIL gate(對齊 audit_kwave_transition.py 同模式)。
2. [Cross-Layer × Cross-Pillar Audit Scope]: 12 checks(C1-C12)涵蓋 schema integrity(C1-C4)+ data integrity(C5-C8)+ coverage(C9-C11)+ trinity dashboard(C12)。Phase E builders 補 hook 前,data-side checks 多為 INFO(records 為空);Phase E 後變實質驗證。
3. [Schema Integrity Strict]: C1-C4 為 schema 結構檢驗(3 tables + 1 view + FK + CHECK enum + matview index)— 任一缺失為 FAIL(因 §14.7-BU Phase C/D 必須 100% 落地;此為前置條件不容忍)。
4. [Data Integrity INFO]: C5-C12 為 data-side 檢驗(records 存在性、enum 違反、coverage 完整性);Phase E 前皆為 INFO(records 為空為預期),不阻斷;Phase E 後預期實質驗證 ≥ 95% PASS。
5. [Trinity Dashboard Output]: C12 為 119 × 12 矩陣化輸出(per stock × per pillar × per layer),對映憲章 §0.4 數位孿生完整性之 explicit 化(per §14.7-BU 治權新特性 #1)。
6. [Zero Hardcoded Verdict]: 主權判定動態計算(§5.6.3);verdict ∈ {PERFECT, WARNING, FAILED, EMPTY_AWAITING_PHASE_E}。
7. [Sovereignty Declaration]: 本工具屬 §11C audit;不選股(不涉 §6 CoreScore)、不訓練模型(不涉 §8.3)、不預測(不涉 §9.1)、不分配資金(不涉 §9.2)、不涉 §0.1-A / §0.2-A / §0.3-A 五套禁令、不在 T1-T3 分層內、不處理 §8.5 anti-leakage(由 builders 主管)。
8. [Historical Reference Authority]: v0.1 為首版落地(§14.7-BU Phase F 入憲);後續升版保留歷程。
9. [Read-Only Sovereignty]: 只讀 universe_completeness_*  / core_universe_* / model_registry / feature_store_snapshot;不寫任何表;不執行 REFRESH(由 schema 工具或 builders 觸發)。
10. [Hybrid Observability]: 維運觸發 record_lifecycle(若可用);主權判定動態計算;output 支援 console / json 雙模式。

## 📊 二、執行指令
| 場景 | 指令 |
| :--- | :--- |
| **Default audit(latest committed snapshot)** | `$ python scripts/maintenance/audit_universe_completeness.py` |
| **指定 snapshot_id** | `$ python scripts/maintenance/audit_universe_completeness.py --snapshot-id <snapshot_id>` |
| **Output JSON(machine-readable)** | `$ python scripts/maintenance/audit_universe_completeness.py --output-format json` |
| **Verbose(顯示每 cell completeness)** | `$ python scripts/maintenance/audit_universe_completeness.py --verbose` |

## 📊 二、全量維運指令總矩陣 (Operational Matrix)

| 指令 / 模式 | 行為 | 治權對應 |
| :--- | :--- | :--- |
| --snapshot-id <id> | 指定 snapshot(預設 latest committed) | §14.7-BU |
| --output-format console|json | 輸出格式 | 維運 |
| --verbose | 顯示每 check 細節 | 維運 |

## 📜 三、修訂歷程
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **v0.1** | 2026-05-26 | Codex | **§14.7-BU Phase F 落地首版**:依憲章 v6.1.0-patch 第十九輪 §14.7-BU Phase B 入憲(charter L9378 新子節)+ Phase A 設計研究(commit `73228ba` §8 audit tool design)+ Phase C/D schema 落地(`universe_completeness_schema.py v0.1` 同次)。**12 checks**:C1 3 tables existence / C2 materialized view + unique index / C3 FK 整體完整 / C4 CHECK constraints(pillar / layer / completeness_pct)/ C5 universe_completeness_snapshot record count / C6 pillar enum 合法率 / C7 layer enum 合法率 / C8 completeness_pct 邊界 / C9 universe snapshot coverage(預期 119 stocks × 12 cells = 1,428 records)/ C10 cross-layer rollup(per stock avg)/ C11 cross-pillar rollup(per layer avg)/ C12 trinity dashboard 矩陣輸出。**Phase E builders 補 hook 前**:C5-C12 為 INFO(records 空);Phase E 後預期實質驗證 ≥ 95% PASS。**INFO-only 治權**:不作 FAIL gate(對齊 audit_kwave_transition.py / §0.3.8.4 治權邊界 / §11C audit-only 慣例);僅 C1-C4 schema integrity 為 FAIL(前置條件)。對既有 DB 影響:零(read-only)。 | **ACTIVE** |
================================================================================
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from core.path_setup import ensure_scripts_on_path
ensure_scripts_on_path(__file__)

from core.db_utils import get_db_connection


CONSTITUTION_VER = "v6.1.0"
TOOL_VER = "v0.1"

EXPECTED_TABLES = ["prediction_run", "predictions", "universe_completeness_snapshot"]
EXPECTED_VIEW = "universe_completeness_matrix_current"
EXPECTED_PILLARS = {
    "first_principle",
    "pareto",
    "kondratiev",  # backward-compat / historical(per §14.7-BZ Phase F:超前 §0.3 mix pillar)
    # §14.7-BZ Phase F(2026-05-27)新加 3 sub-pillars(對映 §0.3.1/§0.3.2/§0.3.3)
    "kondratiev_kwave",         # §0.3.1 K-wave pure(40-60 年)
    "kondratiev_multicycle",    # §0.3.2 Multi-cycle(7-25 年)
    "kondratiev_microstructure", # §0.3.3 Microstructure(月 ~ 季)
}
EXPECTED_LAYERS = {"data", "feature", "model", "prediction"}
EXPECTED_CELLS_PER_STOCK = len(EXPECTED_PILLARS) * len(EXPECTED_LAYERS)  # 24 per §14.7-BZ Phase F
# Note: per-stock 實際只寫 5 pillars × 4 layers = 20 cells(`kondratiev` enum 為 backward-compat
# 不再新 insert;只在歷史 records 出現),audit 仍接受 6 enum 為合法值


def _row_exists(cur, query, params=None):
    cur.execute(query, params or ())
    return cur.fetchone()


def check_c1_tables_exist(conn):
    """C1: Schema integrity — 3 tables existence."""
    cur = conn.cursor()
    results = []
    all_exist = True
    for table in EXPECTED_TABLES:
        cur.execute("SELECT to_regclass(%s)", (f'public."{table}"',))
        exists = cur.fetchone()[0] is not None
        results.append({"table": table, "exists": exists})
        if not exists:
            all_exist = False
    cur.close()
    return {
        "check": "C1_tables_exist",
        "status": "PASS" if all_exist else "FAIL",
        "details": results,
    }


def check_c2_matview_exists(conn):
    """C2: Materialized view + unique index existence."""
    cur = conn.cursor()
    cur.execute(
        "SELECT relname FROM pg_class WHERE relkind='m' AND relname=%s",
        (EXPECTED_VIEW,),
    )
    matview_exists = cur.fetchone() is not None

    cur.execute(
        "SELECT indexname FROM pg_indexes WHERE tablename=%s AND indexname=%s",
        (EXPECTED_VIEW, "idx_completeness_mv_unique"),
    )
    unique_idx_exists = cur.fetchone() is not None
    cur.close()

    status = "PASS" if matview_exists and unique_idx_exists else "FAIL"
    return {
        "check": "C2_matview_exists",
        "status": status,
        "details": {
            "matview_exists": matview_exists,
            "unique_index_exists": unique_idx_exists,
            "concurrently_refresh_ready": matview_exists and unique_idx_exists,
        },
    }


def check_c3_foreign_keys(conn):
    """C3: Foreign key integrity per Phase A §6 design."""
    expected_fks = [
        ("prediction_run", "model_id", "model_registry"),
        ("prediction_run", "feature_set_id", "feature_store_snapshot"),
        ("prediction_run", "universe_snapshot_id", "core_universe_snapshot"),
        ("predictions", "run_id", "prediction_run"),
        ("universe_completeness_snapshot", "universe_snapshot_id", "core_universe_snapshot"),
    ]
    cur = conn.cursor()
    missing = []
    found = []
    for table, col, ref_table in expected_fks:
        cur.execute(
            """
            SELECT conname FROM pg_constraint
            WHERE conrelid = %s::regclass
              AND contype = 'f'
              AND confrelid = %s::regclass
              AND %s = ANY (SELECT attname FROM pg_attribute WHERE attrelid = conrelid AND attnum = ANY (conkey))
            """,
            (f'public."{table}"', f'public."{ref_table}"', col),
        )
        row = cur.fetchone()
        if row:
            found.append({"table": table, "column": col, "ref": ref_table, "constraint": row[0]})
        else:
            missing.append({"table": table, "column": col, "ref": ref_table})
    cur.close()
    return {
        "check": "C3_foreign_keys",
        "status": "PASS" if not missing else "FAIL",
        "details": {"found": found, "missing": missing, "total_expected": len(expected_fks)},
    }


def check_c4_check_constraints(conn):
    """C4: CHECK constraints on universe_completeness_snapshot (pillar / layer / pct)."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT conname, pg_get_constraintdef(oid)
        FROM pg_constraint
        WHERE conrelid = 'public."universe_completeness_snapshot"'::regclass
          AND contype = 'c'
        ORDER BY conname
        """
    )
    rows = cur.fetchall()
    cur.close()

    found_names = {r[0] for r in rows}
    expected = {"ck_completeness_pillar", "ck_completeness_layer", "ck_completeness_pct"}
    missing = expected - found_names
    return {
        "check": "C4_check_constraints",
        "status": "PASS" if not missing else "FAIL",
        "details": {
            "found": [{"name": n, "def": d} for n, d in rows],
            "missing": list(missing),
            "total_expected": len(expected),
        },
    }


def check_c5_record_count(conn, snapshot_id):
    """C5: universe_completeness_snapshot record count for given snapshot."""
    cur = conn.cursor()
    if snapshot_id:
        cur.execute(
            'SELECT COUNT(*) FROM "universe_completeness_snapshot" WHERE universe_snapshot_id = %s',
            (snapshot_id,),
        )
    else:
        cur.execute('SELECT COUNT(*) FROM "universe_completeness_snapshot"')
    count = cur.fetchone()[0]
    cur.close()

    status = "INFO_EMPTY" if count == 0 else "INFO_DATA_PRESENT"
    return {
        "check": "C5_record_count",
        "status": status,
        "details": {
            "snapshot_id": snapshot_id,
            "record_count": count,
            "note": "Phase E builders 補 hook 前為 0 屬預期(non-blocking)",
        },
    }


def check_c6_pillar_enum(conn):
    """C6: pillar enum 合法率 (DB CHECK enforces;此檢驗為 redundancy)."""
    cur = conn.cursor()
    cur.execute(
        'SELECT DISTINCT pillar FROM "universe_completeness_snapshot"'
    )
    rows = [r[0] for r in cur.fetchall()]
    cur.close()

    if not rows:
        return {
            "check": "C6_pillar_enum",
            "status": "INFO_EMPTY",
            "details": {"distinct_pillars": [], "expected": list(EXPECTED_PILLARS)},
        }
    invalid = [p for p in rows if p not in EXPECTED_PILLARS]
    return {
        "check": "C6_pillar_enum",
        "status": "PASS" if not invalid else "FAIL",
        "details": {
            "distinct_pillars": rows,
            "invalid": invalid,
            "expected": list(EXPECTED_PILLARS),
        },
    }


def check_c7_layer_enum(conn):
    """C7: layer enum 合法率 (DB CHECK enforces;此檢驗為 redundancy)."""
    cur = conn.cursor()
    cur.execute(
        'SELECT DISTINCT layer FROM "universe_completeness_snapshot"'
    )
    rows = [r[0] for r in cur.fetchall()]
    cur.close()

    if not rows:
        return {
            "check": "C7_layer_enum",
            "status": "INFO_EMPTY",
            "details": {"distinct_layers": [], "expected": list(EXPECTED_LAYERS)},
        }
    invalid = [l for l in rows if l not in EXPECTED_LAYERS]
    return {
        "check": "C7_layer_enum",
        "status": "PASS" if not invalid else "FAIL",
        "details": {
            "distinct_layers": rows,
            "invalid": invalid,
            "expected": list(EXPECTED_LAYERS),
        },
    }


def check_c8_pct_bounds(conn):
    """C8: completeness_pct 邊界 [0, 100] (DB CHECK enforces;此檢驗為 redundancy)."""
    cur = conn.cursor()
    cur.execute(
        'SELECT MIN(completeness_pct), MAX(completeness_pct), COUNT(*) FROM "universe_completeness_snapshot"'
    )
    min_val, max_val, count = cur.fetchone()
    cur.close()

    if count == 0:
        return {
            "check": "C8_pct_bounds",
            "status": "INFO_EMPTY",
            "details": {"min": None, "max": None, "count": 0},
        }
    out_of_bounds = (min_val is not None and min_val < 0) or (max_val is not None and max_val > 100)
    return {
        "check": "C8_pct_bounds",
        "status": "FAIL" if out_of_bounds else "PASS",
        "details": {
            "min": float(min_val) if min_val is not None else None,
            "max": float(max_val) if max_val is not None else None,
            "count": count,
        },
    }


def check_c9_universe_coverage(conn, snapshot_id):
    """C9: universe snapshot stocks 覆蓋率 (per snapshot, expected = N stocks × 12 cells)."""
    cur = conn.cursor()
    if not snapshot_id:
        cur.execute(
            "SELECT snapshot_id FROM core_universe_snapshot WHERE status='committed' ORDER BY as_of_date DESC LIMIT 1"
        )
        row = cur.fetchone()
        snapshot_id = row[0] if row else None

    if not snapshot_id:
        cur.close()
        return {
            "check": "C9_universe_coverage",
            "status": "INFO_NO_SNAPSHOT",
            "details": {"snapshot_id": None, "note": "no committed core_universe_snapshot found"},
        }

    cur.execute(
        'SELECT COUNT(*) FROM "core_universe_membership" WHERE snapshot_id = %s',
        (snapshot_id,),
    )
    universe_n = cur.fetchone()[0]

    cur.execute(
        'SELECT COUNT(DISTINCT stock_id), COUNT(*) FROM "universe_completeness_snapshot" WHERE universe_snapshot_id = %s',
        (snapshot_id,),
    )
    distinct_stocks, total_records = cur.fetchone()
    cur.close()

    expected_records = universe_n * EXPECTED_CELLS_PER_STOCK
    if total_records == 0:
        status = "INFO_EMPTY"
    elif distinct_stocks == universe_n and total_records == expected_records:
        status = "PASS"
    else:
        status = "INFO_PARTIAL"

    return {
        "check": "C9_universe_coverage",
        "status": status,
        "details": {
            "snapshot_id": snapshot_id,
            "universe_n": universe_n,
            "distinct_stocks_in_completeness": distinct_stocks,
            "total_records": total_records,
            "expected_records": expected_records,
            "coverage_pct": round(100.0 * total_records / expected_records, 2) if expected_records > 0 else 0.0,
        },
    }


def check_c10_per_stock_rollup(conn, snapshot_id):
    """C10: per stock cross-layer rollup (avg completeness)."""
    cur = conn.cursor()
    if not snapshot_id:
        cur.execute("SELECT snapshot_id FROM core_universe_snapshot WHERE status='committed' ORDER BY as_of_date DESC LIMIT 1")
        row = cur.fetchone()
        snapshot_id = row[0] if row else None

    if not snapshot_id:
        cur.close()
        return {"check": "C10_per_stock_rollup", "status": "INFO_NO_SNAPSHOT", "details": {}}

    cur.execute(
        """
        SELECT stock_id, AVG(completeness_pct)::float AS avg_pct, COUNT(*) AS cells
        FROM "universe_completeness_snapshot"
        WHERE universe_snapshot_id = %s
        GROUP BY stock_id
        ORDER BY avg_pct ASC
        LIMIT 5
        """,
        (snapshot_id,),
    )
    bottom_5 = [{"stock_id": r[0], "avg_pct": round(r[1], 2), "cells": r[2]} for r in cur.fetchall()]
    cur.close()

    if not bottom_5:
        return {"check": "C10_per_stock_rollup", "status": "INFO_EMPTY", "details": {"snapshot_id": snapshot_id}}
    return {
        "check": "C10_per_stock_rollup",
        "status": "INFO_DATA_PRESENT",
        "details": {"snapshot_id": snapshot_id, "bottom_5_stocks_by_avg_completeness": bottom_5},
    }


def check_c11_per_layer_rollup(conn, snapshot_id):
    """C11: per layer × per pillar rollup (avg completeness)."""
    cur = conn.cursor()
    if not snapshot_id:
        cur.execute("SELECT snapshot_id FROM core_universe_snapshot WHERE status='committed' ORDER BY as_of_date DESC LIMIT 1")
        row = cur.fetchone()
        snapshot_id = row[0] if row else None

    if not snapshot_id:
        cur.close()
        return {"check": "C11_per_layer_rollup", "status": "INFO_NO_SNAPSHOT", "details": {}}

    cur.execute(
        """
        SELECT pillar, layer, AVG(completeness_pct)::float AS avg_pct, COUNT(*) AS records
        FROM "universe_completeness_snapshot"
        WHERE universe_snapshot_id = %s
        GROUP BY pillar, layer
        ORDER BY pillar, layer
        """,
        (snapshot_id,),
    )
    rollup = [
        {"pillar": r[0], "layer": r[1], "avg_pct": round(r[2], 2), "records": r[3]}
        for r in cur.fetchall()
    ]
    cur.close()

    if not rollup:
        return {"check": "C11_per_layer_rollup", "status": "INFO_EMPTY", "details": {"snapshot_id": snapshot_id}}
    return {
        "check": "C11_per_layer_rollup",
        "status": "INFO_DATA_PRESENT",
        "details": {"snapshot_id": snapshot_id, "trinity_layer_rollup": rollup},
    }


def check_c12_trinity_dashboard(conn, snapshot_id):
    """C12: Trinity dashboard (3 × 4 = 12 cell matrix) summary."""
    cur = conn.cursor()
    if not snapshot_id:
        cur.execute("SELECT snapshot_id FROM core_universe_snapshot WHERE status='committed' ORDER BY as_of_date DESC LIMIT 1")
        row = cur.fetchone()
        snapshot_id = row[0] if row else None

    if not snapshot_id:
        cur.close()
        return {"check": "C12_trinity_dashboard", "status": "INFO_NO_SNAPSHOT", "details": {}}

    cur.execute(
        """
        SELECT pillar, layer,
               COUNT(*) AS records,
               AVG(completeness_pct)::float AS avg_pct,
               MIN(completeness_pct)::float AS min_pct,
               MAX(completeness_pct)::float AS max_pct
        FROM "universe_completeness_snapshot"
        WHERE universe_snapshot_id = %s
        GROUP BY pillar, layer
        """,
        (snapshot_id,),
    )
    cell_data = {(r[0], r[1]): r for r in cur.fetchall()}
    cur.close()

    matrix = {}
    for pillar in sorted(EXPECTED_PILLARS):
        matrix[pillar] = {}
        for layer in sorted(EXPECTED_LAYERS):
            data = cell_data.get((pillar, layer))
            if data:
                matrix[pillar][layer] = {
                    "records": data[2],
                    "avg_pct": round(data[3], 2) if data[3] is not None else None,
                    "min_pct": round(data[4], 2) if data[4] is not None else None,
                    "max_pct": round(data[5], 2) if data[5] is not None else None,
                }
            else:
                matrix[pillar][layer] = {"records": 0, "avg_pct": None}

    total_records = sum(
        m["records"] for pillar_data in matrix.values() for m in pillar_data.values()
    )
    return {
        "check": "C12_trinity_dashboard",
        "status": "INFO_EMPTY" if total_records == 0 else "INFO_DATA_PRESENT",
        "details": {"snapshot_id": snapshot_id, "matrix": matrix, "total_records": total_records},
    }


def compute_verdict(results):
    schema_checks = ["C1_tables_exist", "C2_matview_exists", "C3_foreign_keys", "C4_check_constraints"]
    schema_fails = [r for r in results if r["check"] in schema_checks and r["status"] == "FAIL"]
    if schema_fails:
        return "FAILED"

    data_fails = [r for r in results if r["check"] not in schema_checks and r["status"] == "FAIL"]
    if data_fails:
        return "WARNING"

    empty_count = sum(1 for r in results if r["status"].startswith("INFO_EMPTY"))
    if empty_count >= 6:
        return "EMPTY_AWAITING_PHASE_E"
    return "PERFECT"


def run_all_checks(snapshot_id=None):
    conn = get_db_connection()
    try:
        results = [
            check_c1_tables_exist(conn),
            check_c2_matview_exists(conn),
            check_c3_foreign_keys(conn),
            check_c4_check_constraints(conn),
            check_c5_record_count(conn, snapshot_id),
            check_c6_pillar_enum(conn),
            check_c7_layer_enum(conn),
            check_c8_pct_bounds(conn),
            check_c9_universe_coverage(conn, snapshot_id),
            check_c10_per_stock_rollup(conn, snapshot_id),
            check_c11_per_layer_rollup(conn, snapshot_id),
            check_c12_trinity_dashboard(conn, snapshot_id),
        ]
    finally:
        conn.close()
    return results


def render_console(results, verbose=False):
    print("\n" + "🛡️" * 40)
    print(f"🚀 Quantum Finance: Universe Completeness Audit Report ({TOOL_VER})")
    print("🛡️" * 40)
    print(f"治權基準 : 系統架構大憲章_{CONSTITUTION_VER}.md §14.7-BU Phase F")
    print(f"治理權責 : Audit-Only Authority (§11C 治權檢驗延伸 / read-only / INFO-only)")
    print(f"執行時刻 : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("─" * 80)

    icon_map = {
        "PASS": "✅",
        "FAIL": "❌",
        "INFO_EMPTY": "ℹ️",
        "INFO_DATA_PRESENT": "📊",
        "INFO_NO_SNAPSHOT": "⚠️",
        "INFO_PARTIAL": "🟡",
    }
    for result in results:
        icon = icon_map.get(result["status"], "•")
        print(f"{icon} [{result['status']:<22s}] {result['check']}")
        if verbose or result["status"] in ("FAIL", "INFO_PARTIAL"):
            details_str = json.dumps(result["details"], ensure_ascii=False, indent=2, default=str)
            for line in details_str.split("\n"):
                print(f"    {line}")
        elif result["check"] == "C12_trinity_dashboard" and result["status"] == "INFO_DATA_PRESENT":
            matrix = result["details"]["matrix"]
            print("    Trinity Dashboard (3 pillars × 4 layers):")
            for pillar in sorted(matrix):
                for layer in sorted(matrix[pillar]):
                    cell = matrix[pillar][layer]
                    print(f"      {pillar:<16s} × {layer:<10s}: records={cell['records']:<4d} avg_pct={cell.get('avg_pct')}")

    print("─" * 80)
    verdict = compute_verdict(results)
    verdict_icon = {"PERFECT": "🎯", "WARNING": "⚠️", "FAILED": "❌", "EMPTY_AWAITING_PHASE_E": "⏸️"}.get(verdict, "•")
    print(f"⚖️  主權判定 : {verdict_icon} {verdict}")
    if verdict == "EMPTY_AWAITING_PHASE_E":
        print(f"📝 註記      : Phase E builders 補 hook 前 universe_completeness_snapshot 為空為預期")
        print(f"             schema integrity(C1-C4)PASS 即為本階段健康狀態")
    print("🛡️" * 40 + "\n")


def render_json(results):
    output = {
        "tool": "audit_universe_completeness",
        "tool_ver": TOOL_VER,
        "constitution_ver": CONSTITUTION_VER,
        "audit_time": datetime.now().isoformat(),
        "verdict": compute_verdict(results),
        "checks": results,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(description="Quantum Finance Universe Completeness Audit (v0.1)")
    parser.add_argument("--snapshot-id", type=str, default=None, help="指定 core_universe_snapshot.snapshot_id;預設為 latest committed")
    parser.add_argument("--output-format", choices=["console", "json"], default="console", help="輸出格式")
    parser.add_argument("--verbose", action="store_true", help="顯示每 check 完整 details")
    args = parser.parse_args()

    results = run_all_checks(snapshot_id=args.snapshot_id)
    if args.output_format == "json":
        render_json(results)
    else:
        render_console(results, verbose=args.verbose)

    verdict = compute_verdict(results)
    sys.exit(0 if verdict in ("PERFECT", "WARNING", "EMPTY_AWAITING_PHASE_E") else 1)


if __name__ == "__main__":
    main()
