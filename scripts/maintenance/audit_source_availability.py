"""
audit_source_availability.py v0.1 (Core 150 Strict Source Availability Audit)
================================================================================
**最後更新日期**: 2026-05-18
**主權狀態**: ACTIVE (憲法 v6.0.0 §14.7-L 對齊)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE

目的：
  驗證 core+convex 150 個股資料是否符合新版大憲章之嚴格定義：
  每一個 FinMind `stock_id + dataset` 必須從 API 最早可得日期完整對齊 DB。

驗收口徑：
  - FinMind API 以 `start_date=1990-01-01` 作為來源端可得下界。
  - 對每個 `stock_id + dataset` 比對 API 與 DB 的 row_count / min(date) / max(date)。
  - API source-empty 時，DB 必須為 0 rows 才算 `SOURCE_EMPTY_OK`。
  - 任何 row_count/min/max 差異均為 mismatch，嚴格模式 exit 1。

常用指令：
  python scripts/maintenance/audit_source_availability.py --universe core --all --include-fred --strict
  python scripts/maintenance/audit_source_availability.py --id 2330 --all --strict
  python scripts/maintenance/audit_source_availability.py --universe core --all --strict --snapshot-out /tmp/api_start_dates_core150.json
  python scripts/maintenance/audit_source_availability.py --universe core --all --strict --snapshot-in /tmp/api_start_dates_core150.json
================================================================================
"""
import argparse
import json
import os
import sys
import time
from collections import deque
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import requests

_THIS_FILE = Path(__file__).resolve()
_MAINTENANCE_DIR = _THIS_FILE.parent
_SCRIPTS_DIR = _MAINTENANCE_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from core.path_setup import get_report_dir
    from core.db_utils import get_db_connection, get_core_stocks_from_db, record_lifecycle
    from core.data_schema import DATASET_REGISTRY, FINMIND_API_TABLES
    from core.finmind_client import FinMindClient
except ImportError as exc:
    print(f"❌ 核心組件導入失敗: {exc}")
    sys.exit(1)


STRICT_SOURCE_START_DATE = "1990-01-01"
DEFAULT_THROTTLE_PER_HOUR = 5500
FRED_PAGE_LIMIT = 100000
FRED_SERIES = ["DFF", "UNRATE", "T10Y2Y", "VIXCLS"]
STOCK_LEVEL_DATASETS = [
    name for name in FINMIND_API_TABLES
    if name != "TaiwanStockInfo"
]


class SimpleThrottle:
    """Small local throttle for audit probes; keeps audit under §7 request policy."""

    def __init__(self, max_per_hour=DEFAULT_THROTTLE_PER_HOUR):
        self.max_per_hour = max_per_hour
        self.window = deque()

    def acquire(self):
        now = time.time()
        while self.window and self.window[0] < now - 3600:
            self.window.popleft()
        if len(self.window) >= self.max_per_hour:
            sleep_for = 3600 - (now - self.window[0]) + 1
            print(f"⏸  audit throttle sleep {sleep_for:.0f}s ({len(self.window)}/{self.max_per_hour})")
            time.sleep(sleep_for)
        self.window.append(time.time())


class SourceAvailabilityAuditor:
    def __init__(self, start_date=STRICT_SOURCE_START_DATE, throttle_per_hour=DEFAULT_THROTTLE_PER_HOUR,
                 snapshot_in=None, snapshot_out=None):
        self.constitution_ver = "v6.0.0"
        self.tool_ver = "v0.1"
        self.start_date = start_date
        self.client = FinMindClient()
        self.throttle = SimpleThrottle(max_per_hour=throttle_per_hour)
        self.snapshot_in = Path(snapshot_in) if snapshot_in else None
        self.snapshot_out = Path(snapshot_out) if snapshot_out else None
        self.api_snapshot = self._load_snapshot(self.snapshot_in)
        self.results = []
        self.fred_results = []
        self.api_errors = 0
        self.checked = 0
        self.mismatch = 0
        self.source_empty_ok = 0
        self.fred_checked = 0
        self.fred_mismatch = 0
        self.fred_api_errors = 0

    def _load_snapshot(self, path):
        if not path:
            return {}
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "items" in payload:
            items = payload["items"]
        elif isinstance(payload, dict) and "finmind" in payload:
            items = payload["finmind"]
        elif isinstance(payload, list):
            items = payload
        else:
            raise ValueError(f"Unsupported snapshot format: {path}")
        return {
            (str(item["stock_id"]), item.get("dataset") or item.get("table")): item
            for item in items
        }

    def _dump_snapshot(self):
        if not self.snapshot_out:
            return
        self.snapshot_out.parent.mkdir(parents=True, exist_ok=True)
        items = [
            {
                "stock_id": r["stock_id"],
                "dataset": r["dataset"],
                "api_rows": r["api_rows"],
                "api_min": r["api_min"],
                "api_max": r["api_max"],
            }
            for r in self.results
            if r["api_status"] == "OK"
        ]
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "start_date": self.start_date,
            "items": items,
        }
        with open(self.snapshot_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _resolve_stocks(self, stock_id=None, universe="core"):
        if stock_id:
            return [str(stock_id)]
        if universe != "core":
            raise ValueError("v0.1 only authorizes --universe core for §6.7 core+convex scope")
        return list(get_core_stocks_from_db(tiers=("core_universe", "convex_universe")))

    def _resolve_datasets(self, dataset=None, all_datasets=False):
        if dataset:
            if dataset not in STOCK_LEVEL_DATASETS:
                raise ValueError(f"Unsupported stock-level dataset: {dataset}")
            return [dataset]
        if all_datasets:
            return list(STOCK_LEVEL_DATASETS)
        return ["TaiwanStockPrice", "TaiwanStockPriceAdj", "TaiwanStockFinancialStatements"]

    def _api_summary_from_snapshot(self, stock_id, dataset):
        item = self.api_snapshot.get((str(stock_id), dataset))
        if not item:
            return None
        return {
            "api_status": "OK",
            "api_rows": int(item.get("api_rows") or 0),
            "api_min": item.get("api_min"),
            "api_max": item.get("api_max"),
        }

    def _fetch_api_summary(self, stock_id, dataset):
        cached = self._api_summary_from_snapshot(stock_id, dataset)
        if cached is not None:
            return cached

        self.throttle.acquire()
        params = {
            "dataset": dataset,
            "data_id": stock_id,
            "start_date": self.start_date,
        }
        if self.client.token:
            params["token"] = self.client.token
        res = requests.get(self.client.api_url, params=params, headers=self.client.headers, timeout=30)
        res.raise_for_status()
        payload = res.json()
        if payload.get("msg") not in (None, "", "success"):
            raise RuntimeError(f"FinMind app-level error: {payload.get('msg')}")
        data = payload.get("data", [])
        if not data:
            return {"api_status": "OK", "api_rows": 0, "api_min": None, "api_max": None}

        df = pd.DataFrame(data)
        if "date" not in df.columns:
            raise ValueError(f"{dataset} API response missing date column")
        dates = pd.to_datetime(df["date"], errors="coerce").dt.date.dropna()
        if dates.empty:
            return {"api_status": "OK", "api_rows": 0, "api_min": None, "api_max": None}
        return {
            "api_status": "OK",
            "api_rows": int(len(df[df["date"].notna()])),
            "api_min": str(min(dates)),
            "api_max": str(max(dates)),
        }

    def _db_summary(self, stock_id, dataset, conn):
        if dataset not in DATASET_REGISTRY:
            raise ValueError(f"Dataset not in DATASET_REGISTRY: {dataset}")
        columns = DATASET_REGISTRY[dataset]["columns"]
        if "stock_id" not in columns or "date" not in columns:
            raise ValueError(f"Dataset is not stock-level/date-level: {dataset}")
        with conn.cursor() as cur:
            cur.execute(
                f'''
                SELECT COUNT(*), MIN("date"), MAX("date")
                FROM "{dataset}"
                WHERE "stock_id" = %s
                ''',
                (stock_id,),
            )
            rows, min_date, max_date = cur.fetchone()
        return {
            "db_rows": int(rows or 0),
            "db_min": str(min_date) if isinstance(min_date, (date, datetime)) else None,
            "db_max": str(max_date) if isinstance(max_date, (date, datetime)) else None,
        }

    def _fetch_fred_summary(self, series_id):
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            raise RuntimeError("FRED_API_KEY missing")
        observations = []
        offset = 0
        while True:
            params = {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "limit": FRED_PAGE_LIMIT,
                "offset": offset,
                "sort_order": "asc",
            }
            res = requests.get("https://api.stlouisfed.org/fred/series/observations", params=params, timeout=30)
            res.raise_for_status()
            page = res.json().get("observations", [])
            observations.extend(page)
            if len(page) < FRED_PAGE_LIMIT:
                break
            offset += FRED_PAGE_LIMIT

        valid_dates = []
        for item in observations:
            value = item.get("value")
            if value in (None, "."):
                continue
            try:
                float(value)
            except (TypeError, ValueError):
                continue
            valid_dates.append(pd.to_datetime(item.get("date"), errors="coerce").date())
        valid_dates = [d for d in valid_dates if pd.notna(d)]
        return {
            "series_id": series_id,
            "api_valid_rows": len(valid_dates),
            "api_valid_min": str(min(valid_dates)) if valid_dates else None,
            "api_valid_max": str(max(valid_dates)) if valid_dates else None,
        }

    def _db_fred_summary(self, series_id, conn):
        with conn.cursor() as cur:
            cur.execute(
                '''
                SELECT COUNT(*), MIN("date"), MAX("date")
                FROM "FredData"
                WHERE "series_id" = %s AND "value" IS NOT NULL
                ''',
                (series_id,),
            )
            rows, min_date, max_date = cur.fetchone()
        return {
            "db_valid_rows": int(rows or 0),
            "db_valid_min": str(min_date) if isinstance(min_date, (date, datetime)) else None,
            "db_valid_max": str(max_date) if isinstance(max_date, (date, datetime)) else None,
        }

    def _classify(self, row):
        if row["api_status"] != "OK":
            return "API_ERROR"
        if row["api_rows"] == 0:
            return "SOURCE_EMPTY_OK" if row["db_rows"] == 0 else "SOURCE_EMPTY_DB_HAS_ROWS"
        if (
            row["api_rows"] == row["db_rows"]
            and row["api_min"] == row["db_min"]
            and row["api_max"] == row["db_max"]
        ):
            return "OK"
        return "MISMATCH"

    def _classify_fred(self, row):
        if row["api_status"] != "OK":
            return "API_ERROR"
        if (
            row["api_valid_rows"] == row["db_valid_rows"]
            and row["api_valid_min"] == row["db_valid_min"]
            and row["api_valid_max"] == row["db_valid_max"]
        ):
            return "OK"
        return "MISMATCH"

    def run(self, stock_id=None, universe="core", dataset=None, all_datasets=False, strict=True, include_fred=False):
        stocks = self._resolve_stocks(stock_id=stock_id, universe=universe)
        datasets = self._resolve_datasets(dataset=dataset, all_datasets=all_datasets)
        task_name = f"audit_source_availability_{stock_id or universe}"

        with record_lifecycle(task_name, category="maintenance", stock_id=stock_id or "SYSTEM") as lifecycle:
            conn = get_db_connection()
            try:
                for sid in stocks:
                    for ds in datasets:
                        base = {"stock_id": str(sid), "dataset": ds}
                        try:
                            api = self._fetch_api_summary(str(sid), ds)
                            db = self._db_summary(str(sid), ds, conn)
                            row = {**base, **api, **db}
                        except Exception as exc:
                            self.api_errors += 1
                            row = {
                                **base,
                                "api_status": "ERROR",
                                "api_rows": None,
                                "api_min": None,
                                "api_max": None,
                                "db_rows": None,
                                "db_min": None,
                                "db_max": None,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        row["status"] = self._classify(row)
                        self.results.append(row)
                        self.checked += 1
                        if row["status"] == "SOURCE_EMPTY_OK":
                            self.source_empty_ok += 1
                        elif row["status"] not in {"OK"}:
                            self.mismatch += 1
                if include_fred:
                    for series_id in FRED_SERIES:
                        base = {"series_id": series_id}
                        try:
                            api = self._fetch_fred_summary(series_id)
                            db = self._db_fred_summary(series_id, conn)
                            row = {**base, "api_status": "OK", **api, **db}
                        except Exception as exc:
                            self.fred_api_errors += 1
                            row = {
                                **base,
                                "api_status": "ERROR",
                                "api_valid_rows": None,
                                "api_valid_min": None,
                                "api_valid_max": None,
                                "db_valid_rows": None,
                                "db_valid_min": None,
                                "db_valid_max": None,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        row["status"] = self._classify_fred(row)
                        self.fred_results.append(row)
                        self.fred_checked += 1
                        if row["status"] != "OK":
                            self.fred_mismatch += 1
            finally:
                conn.close()

            self._dump_snapshot()
            verdict = self._verdict(strict=strict)
            if verdict == "FAILED" and hasattr(lifecycle, "mark_failed"):
                lifecycle.mark_failed(f"strict source availability mismatch={self.mismatch}, api_errors={self.api_errors}")
            elif verdict == "WARNING" and hasattr(lifecycle, "mark_warning"):
                lifecycle.mark_warning(f"source availability mismatch={self.mismatch}, api_errors={self.api_errors}")
            self._write_report(verdict, stocks, datasets)
            self._print_summary(verdict)
            return verdict

    def _verdict(self, strict=True):
        if self.api_errors > 0 or self.fred_api_errors > 0:
            return "FAILED"
        if self.mismatch > 0 or self.fred_mismatch > 0:
            return "FAILED" if strict else "WARNING"
        return "PERFECT"

    def _write_report(self, verdict, stocks, datasets):
        report_path = get_report_dir() / f"source_availability_audit_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        mismatches = [r for r in self.results if r["status"] not in {"OK", "SOURCE_EMPTY_OK"}]
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Core 150 strict source availability audit\n\n")
            f.write(f"- **time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **constitution**: 系統架構大憲章_{self.constitution_ver}.md §14.7-L\n")
            f.write(f"- **tool**: audit_source_availability {self.tool_ver}\n")
            f.write(f"- **start_date**: {self.start_date}\n")
            f.write(f"- **scope**: stocks={len(stocks)}, datasets={len(datasets)}\n")
            f.write(f"- **verdict**: **{verdict}**\n")
            f.write(f"- **summary**: checked={self.checked}, source_empty_ok={self.source_empty_ok}, mismatch={self.mismatch}, api_errors={self.api_errors}\n\n")
            if self.fred_results:
                f.write(f"- **fred_summary**: checked={self.fred_checked}, mismatch={self.fred_mismatch}, api_errors={self.fred_api_errors}\n\n")
            f.write("## Mismatches\n\n")
            if not mismatches:
                f.write("None.\n\n")
            else:
                f.write("| stock_id | dataset | status | api_rows | api_min | api_max | db_rows | db_min | db_max |\n")
                f.write("|---|---|---|---:|---|---|---:|---|---|\n")
                for r in mismatches:
                    f.write(
                        f"| {r['stock_id']} | {r['dataset']} | {r['status']} | "
                        f"{r.get('api_rows')} | {r.get('api_min')} | {r.get('api_max')} | "
                        f"{r.get('db_rows')} | {r.get('db_min')} | {r.get('db_max')} |\n"
                    )
                f.write("\n## Targeted Backfill Commands\n\n")
                for r in mismatches:
                    if r["status"] == "API_ERROR":
                        continue
                    f.write(
                        "```bash\n"
                        f".venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id {r['stock_id']} "
                        f"--dataset {r['dataset']} --strict-source-history\n"
                        "```\n"
                    )
            if self.fred_results:
                f.write("\n## FRED Valid Observation Alignment\n\n")
                f.write("| series_id | status | api_valid_rows | api_valid_min | api_valid_max | db_valid_rows | db_valid_min | db_valid_max |\n")
                f.write("|---|---|---:|---|---|---:|---|---|\n")
                for r in self.fred_results:
                    f.write(
                        f"| {r['series_id']} | {r['status']} | {r.get('api_valid_rows')} | "
                        f"{r.get('api_valid_min')} | {r.get('api_valid_max')} | "
                        f"{r.get('db_valid_rows')} | {r.get('db_valid_min')} | {r.get('db_valid_max')} |\n"
                    )
        self.report_path = report_path

    def _print_summary(self, verdict):
        print("\n" + "🛡️" * 40)
        print(f"🚀 Quantum Finance: strict source availability audit ({self.tool_ver})")
        print("🛡️" * 40)
        print(f"report : {self.report_path}")
        print(f"checked={self.checked}")
        print(f"source_empty_ok={self.source_empty_ok}")
        print(f"mismatch={self.mismatch}")
        print(f"api_errors={self.api_errors}")
        if self.fred_results:
            print(f"fred_checked={self.fred_checked}")
            print(f"fred_mismatch={self.fred_mismatch}")
            print(f"fred_api_errors={self.fred_api_errors}")
        print(f"verdict={verdict}")
        print("🛡️" * 40 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Core 150 FinMind strict source availability audit")
    parser.add_argument("--id", type=str, help="single stock_id")
    parser.add_argument("--universe", choices=["core"], default="core", help="authorized universe scope")
    parser.add_argument("--dataset", type=str, help="single FinMind stock-level dataset")
    parser.add_argument("--all", action="store_true", help="audit all FinMind stock-level datasets")
    parser.add_argument("--start-date", default=STRICT_SOURCE_START_DATE,
                        help=f"FinMind source lower bound (default {STRICT_SOURCE_START_DATE})")
    parser.add_argument("--strict", action="store_true", help="exit 1 on any mismatch")
    parser.add_argument("--snapshot-in", help="reuse prior API source snapshot JSON")
    parser.add_argument("--snapshot-out", help="write API source snapshot JSON")
    parser.add_argument("--include-fred", action="store_true",
                        help="also verify FRED DFF/UNRATE/T10Y2Y/VIXCLS valid numeric observations against DB")
    parser.add_argument("--throttle", type=int, default=DEFAULT_THROTTLE_PER_HOUR,
                        help="API requests per hour cap for audit probe")
    args = parser.parse_args()

    auditor = SourceAvailabilityAuditor(
        start_date=args.start_date,
        throttle_per_hour=args.throttle,
        snapshot_in=args.snapshot_in,
        snapshot_out=args.snapshot_out,
    )
    verdict = auditor.run(
        stock_id=args.id,
        universe=args.universe,
        dataset=args.dataset,
        all_datasets=args.all,
        strict=args.strict,
        include_fred=args.include_fred,
    )
    sys.exit(0 if verdict in {"PERFECT", "WARNING"} else 1)


if __name__ == "__main__":
    main()
