"""
sovereign_sync_engine.py v1.8 (Quantum Finance Market Universe Seed Engine)
================================================================================
**最後更新日期**: 2026-05-14
**主權狀態**: MARKET UNIVERSE SEED VERIFIED (憲法 v5.4.19 對齊)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
1. [Market Universe Seed]: 第 4 步取得 `TaiwanStockInfo` 市場個股清單，並同步 FRED 核心宏觀資料。
2. [Schema Sovereignty]: 所有寫入欄位必須 100% 對齊 `data_schema.py v2.11` 的 `DATASET_REGISTRY`。
3. [Hybrid Observability]: 使用 `record_lifecycle(... ) as lc`，將 warning / failed 回寫 pipeline lifecycle。
4. [Zero Silent Drop]: API 空回應、HTTP 4xx/5xx、dropna 全空、DB upsert 失敗皆必須記入 stats 與 terminal report。
5. [Idempotency]: 使用 ON CONFLICT upsert，確保 seed / sync 可重跑。

## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix)
| 維運需求場景 (Scenario) | 權威指令 / 建議用法 (Exhaustive Examples) | 對齊模組 |
| :--- | :--- | :--- |
| **1. [市場個股資料取得與種子灌溉]** | `$ python scripts/ingestion/sovereign_sync_engine.py --seed` | sync_engine v1.8 |
| **2. [單一標的指定數據集同步]** | `$ python scripts/ingestion/sovereign_sync_engine.py --id 2330 --dataset TaiwanStockPrice` | sync_engine v1.8 |
| **3. [FRED 宏觀指標全譜同步]** | `$ python scripts/ingestion/sovereign_sync_engine.py --source fred` | sync_engine v1.8 |

## 📜 三、全修訂歷程 (Full Revision History)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **v1.8** | 2026-05-14 | Codex | **市場個股資料取得與種子灌溉治理**：對齊憲章 v5.4.19 與 data_schema v2.11；接上 lifecycle context；修正全域 args.seed；欄位鏡像與 upsert 前清洗；動態 PERFECT/WARNING/FAILED 判定。 | **ACTIVE** |
| v1.7 | 2026-05-13 | Auto-patch | Bug #1 修補：sync_fred() 補完 empty-data 失敗分支與 dropna 後空集合分支。 | SUPERSEDED |
| v1.6 | 2026-05-13 | Antigravity | 創世圓滿：對齊憲法 v5.4.18。 | ARCHIVED |
================================================================================
"""
import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import requests

_THIS_FILE = Path(__file__).resolve()
_INGESTION_DIR = _THIS_FILE.parent
_SCRIPTS_DIR = _INGESTION_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from core.db_utils import get_db_connection, record_lifecycle, write_data_audit_log, get_core_stocks_from_db
    from core.data_schema import DATASET_REGISTRY, FINMIND_API_TABLES
    from core.finmind_client import FinMindClient
except ImportError as exc:
    print(f"❌ 核心組件導入失敗，請確認 core/ 目錄: {exc}")
    sys.exit(1)


class SovereignSyncEngine:
    FRED_LIST = ["DFF", "UNRATE", "T10Y2Y", "VIXCLS"]
    DEFAULT_FINMIND_DATASETS = [
        "TaiwanStockPrice",
        "TaiwanStockInstitutionalInvestorsBuySell",
        "TaiwanStockMarginPurchaseShortSale",
        "TaiwanStockPER",
    ]

    def __init__(self):
        self.fm_client = FinMindClient()
        self.fred_key = os.getenv("FRED_API_KEY")
        self.constitution_ver = "v5.4.19"
        self.schema_ver = "v2.11"
        self.tool_ver = "v1.8"
        self.stats = {"success": 0, "warning": 0, "failed": 0, "rows": 0, "details": []}

    def _detail(self, status, message):
        self.stats[status] += 1
        icon = {"success": "✅", "warning": "⚠️", "failed": "❌"}[status]
        self.stats["details"].append(f"{icon} {message}")

    def _convert_type(self, series, sql_type):
        if sql_type.startswith("DATE"):
            return pd.to_datetime(series, errors="coerce").dt.date
        if sql_type.startswith("TIMESTAMP"):
            return pd.to_datetime(series, errors="coerce")
        if sql_type.startswith("NUMERIC") or sql_type.startswith("INTEGER") or sql_type.startswith("BIGINT"):
            return pd.to_numeric(series, errors="coerce")
        return series.where(series.notna(), None)

    def _align_to_schema(self, table_name, df):
        config = DATASET_REGISTRY.get(table_name)
        if not config:
            raise ValueError(f"憲章未定義表名: {table_name}")

        expected_cols = list(config["columns"].keys())
        actual_cols = list(df.columns)
        missing = [col for col in expected_cols if col not in actual_cols]
        extra = [col for col in actual_cols if col not in expected_cols]
        if missing or extra:
            problems = []
            if missing:
                problems.append(f"missing={missing}")
            if extra:
                problems.append(f"extra={extra}")
            raise ValueError(f"{table_name} API/schema 欄位不一致: {'; '.join(problems)}")

        aligned = df[expected_cols].copy()
        for col, sql_type in config["columns"].items():
            aligned[col] = self._convert_type(aligned[col], sql_type)

        unique_cols = config.get("unique_constraints", [])
        if unique_cols:
            before = len(aligned)
            aligned = aligned.dropna(subset=unique_cols)
            dropped = before - len(aligned)
            if dropped:
                self._detail("warning", f"{table_name}: 已丟棄 {dropped} 筆 unique key 欄位為空的資料")

        aligned = aligned.where(pd.notnull(aligned), None)
        if aligned.empty:
            raise ValueError(f"{table_name} 清洗後資料為空")
        return aligned

    def _upsert_to_db(self, table_name, df):
        if df.empty:
            raise ValueError(f"{table_name} 無可寫入資料")

        config = DATASET_REGISTRY[table_name]
        unique_cols = config.get("unique_constraints", [])
        if not unique_cols:
            raise ValueError(f"{table_name} 未定義 unique_constraints，禁止 upsert")

        cols = list(df.columns)
        quoted_cols = [f'"{col}"' for col in cols]
        placeholders = ", ".join(["%s"] * len(cols))
        conflict_cols = ", ".join([f'"{col}"' for col in unique_cols])
        update_cols = ", ".join([f'"{col}" = EXCLUDED."{col}"' for col in cols if col not in unique_cols])
        if not update_cols:
            update_clause = "DO NOTHING"
        else:
            update_clause = f"DO UPDATE SET {update_cols}"

        sql = f'''
            INSERT INTO "{table_name}" ({", ".join(quoted_cols)})
            VALUES ({placeholders})
            ON CONFLICT ({conflict_cols})
            {update_clause}
        '''

        conn = get_db_connection()
        cur = conn.cursor()
        try:
            data = [tuple(row) for row in df.itertuples(index=False, name=None)]
            cur.executemany(sql, data)
            conn.commit()
            rows = len(df)
            audit_date = df.iloc[0].get("date", datetime.now().date()) if hasattr(df.iloc[0], "get") else datetime.now().date()
            write_data_audit_log(table_name, "SYNC", audit_date, "UPSERT", rows)
            return rows
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()
            conn.close()

    def sync_finmind(self, stock_id, dataset_name, start_date):
        try:
            print(f"📡 正在獲取 FinMind: {stock_id} / {dataset_name}...")
            params = {"dataset": dataset_name, "start_date": start_date}
            if stock_id is not None:
                params["data_id"] = stock_id
            if self.fm_client.token:
                params["token"] = self.fm_client.token

            res = requests.get(self.fm_client.api_url, params=params, headers=self.fm_client.headers, timeout=30)
            res.raise_for_status()
            payload = res.json()
            if payload.get("msg") not in (None, "success"):
                raise RuntimeError(payload.get("msg"))
            data = payload.get("data", [])
            if not data:
                self._detail("failed", f"{dataset_name} ({stock_id or 'MARKET'}): API 回傳 0 筆")
                return

            df = self._align_to_schema(dataset_name, pd.DataFrame(data))
            rows = self._upsert_to_db(dataset_name, df)
            self.stats["rows"] += rows
            self._detail("success", f"{dataset_name} ({stock_id or 'MARKET'}): {rows} 筆 UPSERT 成功")
        except Exception as exc:
            self._detail("failed", f"{dataset_name} ({stock_id or 'MARKET'}) 失敗: {type(exc).__name__}: {exc}")

    def sync_fred(self, series_id):
        try:
            print(f"📡 正在獲取 FRED: {series_id}...")
            if not self.fred_key:
                raise RuntimeError("FRED_API_KEY missing")
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series_id,
                "api_key": self.fred_key,
                "file_type": "json",
                "limit": 1000,
                "sort_order": "desc",
            }
            res = requests.get(url, params=params, timeout=30)
            res.raise_for_status()
            payload = res.json()
            data = payload.get("observations", [])
            if not data:
                raise RuntimeError(payload.get("error_message") or "API 回傳 0 個 observation")

            df = pd.DataFrame(data)
            df["series_id"] = series_id
            df = self._align_to_schema("FredData", df)
            df = df.dropna(subset=["value"])
            if df.empty:
                raise ValueError("全部 observation 在數據聖潔清洗後為空")

            rows = self._upsert_to_db("FredData", df)
            self.stats["rows"] += rows
            self._detail("success", f"FRED/{series_id}: {rows} 筆 UPSERT 成功")
        except Exception as exc:
            self._detail("failed", f"FRED/{series_id} 失敗: {type(exc).__name__}: {exc}")

    def _resolve_stocks(self, stock_id, universe):
        if stock_id:
            return [stock_id]
        if universe == "core":
            try:
                stocks = get_core_stocks_from_db()
            except Exception as exc:
                self._detail("failed", f"core universe 讀取失敗: {type(exc).__name__}: {exc}")
                return []
            if not stocks:
                self._detail("warning", "core universe 無標的")
            return stocks
        return []

    def _target_datasets(self, dataset, all_datasets):
        if dataset:
            return [dataset]
        if all_datasets:
            return [name for name in FINMIND_API_TABLES if name != "TaiwanStockInfo"]
        return self.DEFAULT_FINMIND_DATASETS

    def _apply_lifecycle_verdict(self, lifecycle):
        if lifecycle is None:
            return
        if self.stats["failed"] > 0 and hasattr(lifecycle, "mark_failed"):
            lifecycle.mark_failed("; ".join(self.stats["details"][:5]))
        elif self.stats["warning"] > 0 and hasattr(lifecycle, "mark_warning"):
            lifecycle.mark_warning("; ".join(self.stats["details"][:5]))

    def run(self, stock_id=None, universe=None, source=None, dataset=None, days=30, seed=False, all_datasets=False):
        start_time = time.time()
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        task_name = f"sync_{source or 'all'}_{stock_id or universe or ('seed' if seed else 'macro')}"

        with record_lifecycle(task_name, category="ingestion", stock_id=stock_id or "SYSTEM") as lifecycle:
            if source in (None, "finmind"):
                if seed or dataset == "TaiwanStockInfo":
                    self.sync_finmind("", "TaiwanStockInfo", start_date)

                stocks = self._resolve_stocks(stock_id, universe)
                for sid in stocks:
                    for ds in self._target_datasets(dataset, all_datasets):
                        if ds == "TaiwanStockInfo":
                            continue
                        self.sync_finmind(sid, ds, start_date)

            if source in (None, "fred") and not stock_id:
                for series_id in self.FRED_LIST:
                    self.sync_fred(series_id)

            self._apply_lifecycle_verdict(lifecycle)
            self.report_results(start_time)

        return self.stats["failed"] == 0

    def report_results(self, start_time):
        if self.stats["failed"] > 0:
            verdict = "FAILED"
        elif self.stats["warning"] > 0:
            verdict = "WARNING"
        else:
            verdict = "PERFECT"

        print("\n" + "🛡️" * 40)
        print(f"🚀 Quantum Finance: 主權同步引擎執行摘要 ({self.tool_ver})")
        print("🛡️" * 40)
        print(f"治權基準 : 系統架構大憲章_{self.constitution_ver}.md")
        print(f"schema 基準 : data_schema {self.schema_ver}")
        print("核心技術 : API Contract Mirror + Absolute Case Sovereignty + Lifecycle Context")
        print("─" * 80)
        for detail in self.stats["details"]:
            print(detail)
        print("─" * 80)
        print(f"📈 成功同步項目 : {self.stats['success']}")
        print(f"⚠️  警告同步項目 : {self.stats['warning']}")
        print(f"❌ 失敗同步項目 : {self.stats['failed']}")
        print(f"📝 總計寫入筆數 : {self.stats['rows']}")
        print(f"🕒 總計耗時     : {(time.time() - start_time):.2f} s")
        print(f"⚖️  主權判定     : {verdict}")
        print("🛡️" * 40 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum Finance 主權同步引擎 (v1.8)")
    parser.add_argument("--id", type=str, help="指定標的 ID (如 2330)")
    parser.add_argument("--universe", type=str, choices=["core"], help="指定標的範圍")
    parser.add_argument("--source", type=str, choices=["finmind", "fred"], help="指定數據源")
    parser.add_argument("--dataset", type=str, help="指定數據集")
    parser.add_argument("--seed", action="store_true", help="市場個股資料取得與種子灌溉模式")
    parser.add_argument("--all", action="store_true", help="全數據灌溉模式")
    parser.add_argument("--days", type=int, default=30, help="同步天數 (預設 30 天)")
    args = parser.parse_args()

    engine = SovereignSyncEngine()
    ok = engine.run(
        stock_id=args.id,
        universe=args.universe,
        source=args.source,
        dataset=args.dataset,
        days=args.days,
        seed=args.seed,
        all_datasets=args.all,
    )
    sys.exit(0 if ok else 1)