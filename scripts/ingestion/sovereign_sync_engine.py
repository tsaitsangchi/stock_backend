"""
sovereign_sync_engine.py v1.9 (Quantum Finance Market Universe Seed Engine)
================================================================================
**最後更新日期**: 2026-05-14
**主權狀態**: MARKET UNIVERSE SEED VERIFIED (憲法 v5.4.21 對齊)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
1. [Market Universe Seed]: 第 4 步取得 `TaiwanStockInfo` 市場個股清單，並同步 FRED 核心宏觀資料。
2. [Schema Sovereignty]: 所有寫入欄位必須 100% 對齊 `data_schema.py v2.11` 的 `DATASET_REGISTRY`。
3. [Hybrid Observability]: 使用 `record_lifecycle(... ) as lc`，將 warning / failed 回寫 pipeline lifecycle。
4. [Zero Silent Drop]: API 空回應、HTTP 4xx/5xx、dropna 全空、DB upsert 失敗皆必須記入 stats 與 terminal report。
5. [Idempotency]: 使用 ON CONFLICT upsert，確保 seed / sync 可重跑。
6. [Phase-Appropriate Lookback]: 預設 `--days 30` 為日常增量；CoreScore v0.2 選股 phase 建議 `--days 730`
   （對應憲章 6.4 price_coverage_252d + revenue_coverage_24m + financial_coverage_8q）。
   全歷史模式（自 2000-01-01）暫不提供，待 v0.3+ 模型訓練 phase 評估後另行授權。

## 📊 二、全量維運指令總矩陣 (per 憲法 v5.4.19 Section 二 / 5.5.3 五大標準)

### A. 5.5.3 五大標準場景 (Core 5 Scenarios)
| 維運需求場景 (Scenario)                       | 權威指令 / 建議用法                                                                                          | 對齊模組 |
| :-------------------------------------------- | :---------------------------------------------------------------------------------------------------------- | :--- |
| **1. [個股同步]**                              | `$ python scripts/ingestion/sovereign_sync_engine.py --id 2330`                                              | sovereign_sync_engine v1.9 |
| **2. [單一 Table 同步]**                       | `$ python scripts/ingestion/sovereign_sync_engine.py --id 2330 --dataset TaiwanStockPrice`                   | sovereign_sync_engine v1.9 |
| **3. [單一個股所有 Table 同步]**                | `$ python scripts/ingestion/sovereign_sync_engine.py --id 2330 --all`                                        | sovereign_sync_engine v1.9 |
| **4. [研究宇宙第一階段灌溉]**                  | `$ python scripts/ingestion/sovereign_sync_engine.py --universe research --all --days 730`                   | sovereign_sync_engine v1.9 |
| **5. [核心宇宙最終同步]**                      | `$ python scripts/ingestion/sovereign_sync_engine.py --universe core --all --days 730`                        | sovereign_sync_engine v1.9 |

### B. 補充運行模式 (Auxiliary Modes)
| 場景                          | 指令                                                                                            | 用途 |
| :---------------------------- | :--------------------------------------------------------------------------------------------- | :--- |
| A. 種子灌溉 (TaiwanStockInfo)  | `$ python scripts/ingestion/sovereign_sync_engine.py --seed`                                    | 第 4 步全市場資產名冊重灌 |
| B. 宏觀指標 (FRED 全譜)        | `$ python scripts/ingestion/sovereign_sync_engine.py --source fred`                             | DFF/UNRATE/T10Y2Y/VIXCLS |
| C. 選股 phase 全宇宙補刷       | `$ python scripts/ingestion/sovereign_sync_engine.py --all --days 730`                          | 為 CoreScore v0.2 準備 2 年歷史 |

### C. 旗標語意 (Flag Semantics, v1.9 整理)
- `--id <stock_id>`：指定單一標的 ID（如 2330）。
- `--universe research|convex|core`：依 `research_universe` / `convex_universe` / `core_universe` + `convex_universe` 分段抓取名單。
- `--dataset <name>`：只同步單一 dataset；不傳則用 `DEFAULT_FINMIND_DATASETS`（4 核心表）。
- **`--all`**：取代「DEFAULT 4 表」為 `FINMIND_API_TABLES`（除 `TaiwanStockInfo` 外的全部 stock-data tables）。
- `--days <N>`：增量同步天數（預設 30）。**研究宇宙 / 選股 phase 建議 730**。
- `--seed`：執行 `TaiwanStockInfo` 種子灌溉（與其他旗標可組合）。
- `--source [finmind|fred]`：限定數據源。

### D. 不提供之旗標 (Intentionally Omitted)
- `--force` (全歷史 from 2000-01-01)：v1.9 不提供，原因：
  - `core_universe_builder v0.1/v0.2` 最大歷史需求 = 730 天（憲章 6.4 證實）
  - 26 年歷史 × 2796 檔 × 9 表 將耗盡 FinMind API quota
  - 待 v0.3+ 模型訓練/回測 phase 啟動時，由 builder 提請新議題並重新授權

## 📜 三、全修訂歷程 (Full Revision History)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **v1.9** | 2026-05-14 | Auto-patch | **5.5.3 對齊版**：(1) 矩陣表補滿憲章 5.5.3 五大標準場景與 v5.4.19 Section 二之 sync 場景；(2) `--all` 旗標語意正式登錄文件（v1.8 已實作但未銘刻）；(3) 新增「Phase-Appropriate Lookback」條款，明定 CoreScore v0.2 選股 phase 建議 `--days 730`；(4) 加入 runtime hint，使用 `--universe research|convex|core` 而未調整 `--days` 時提示選股最佳實務；(5) 明文不提供 `--force`，避免在 v0.2 builder 完工前誘發 API quota 浪費。本版不更動任何 sync logic，僅文件對齊 + UX 強化。 | **ACTIVE** |
| v1.8 | 2026-05-14 | Codex | **市場個股資料取得與種子灌溉治理**：對齊憲章 v5.4.19 與 data_schema v2.11；接上 lifecycle context；修正全域 args.seed；欄位鏡像與 upsert 前清洗；動態 PERFECT/WARNING/FAILED 判定。 | SUPERSEDED |
| v1.7 | 2026-05-13 | Auto-patch | Bug #1 修補：sync_fred() 補完 empty-data 失敗分支與 dropna 後空集合分支。 | ARCHIVED |
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


# v1.9 phase-aware constants
SELECTION_PHASE_DAYS = 730   # CoreScore v0.2 建議窗 (~2 年)
UNIVERSE_TIERS = {
    "research": ("research_universe",),
    "convex": ("convex_universe",),
    "core": ("core_universe", "convex_universe"),
}


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
        self.constitution_ver = "v5.4.21"
        self.schema_ver = "v2.11"
        self.tool_ver = "v1.9"
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

    def _db_value(self, value):
        if pd.isna(value):
            return None
        return value

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
            data = [
                tuple(self._db_value(value) for value in row)
                for row in df.itertuples(index=False, name=None)
            ]
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
                self._detail("warning", f"{dataset_name} ({stock_id or 'MARKET'}): API 回傳 0 筆")
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
        if universe in UNIVERSE_TIERS:
            try:
                stocks = get_core_stocks_from_db(tiers=UNIVERSE_TIERS[universe])
            except Exception as exc:
                self._detail("failed", f"{universe} universe 讀取失敗: {type(exc).__name__}: {exc}")
                return []
            if not stocks:
                self._detail("warning", f"{universe} universe 無標的")
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

    def _phase_appropriate_hint(self, stock_id, universe, days, dataset, seed, all_datasets):
        """v1.9 新增：選股 phase 引導提示，依憲章 6.4 lookback 建議。"""
        # 不對 seed / 單檔 / FRED-only 等情境發提示
        if seed or stock_id or dataset:
            return
        if universe in UNIVERSE_TIERS and days < SELECTION_PHASE_DAYS:
            print(f"💡 [Phase Hint] 對 `--universe {universe}` 使用 `--days 30`（預設）只取增量。")
            print(f"   若你正在執行 research/core staged irrigation，建議改用 `--days {SELECTION_PHASE_DAYS}` "
                  f"（~2 年），對應憲章 6.4 之 price_coverage_252d + revenue_coverage_24m + financial_coverage_8q。")
            print(f"   範例：python scripts/ingestion/sovereign_sync_engine.py --universe {universe} --all --days {SELECTION_PHASE_DAYS}")
            print("─" * 80)

    def run(self, stock_id=None, universe=None, source=None, dataset=None, days=30, seed=False, all_datasets=False):
        start_time = time.time()
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        task_name = f"sync_{source or 'all'}_{stock_id or universe or ('seed' if seed else 'macro')}"

        # v1.9: 選股 phase hint (純提示，不阻斷)
        self._phase_appropriate_hint(stock_id, universe, days, dataset, seed, all_datasets)

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
            self.report_results(start_time, days, universe)

        return self.stats["failed"] == 0

    def report_results(self, start_time, days, universe):
        if self.stats["failed"] > 0:
            verdict = "FAILED"
        elif self.stats["warning"] > 0:
            verdict = "WARNING"
        else:
            verdict = "PERFECT"

        # v1.9: phase 標籤
        if days >= SELECTION_PHASE_DAYS:
            phase_label = f"選股 phase ({days} 天)"
        elif days <= 60:
            phase_label = f"增量 phase ({days} 天)"
        else:
            phase_label = f"自訂窗 ({days} 天)"

        print("\n" + "🛡️" * 40)
        print(f"🚀 Quantum Finance: 主權同步引擎執行摘要 ({self.tool_ver})")
        print("🛡️" * 40)
        print(f"治權基準 : 系統架構大憲章_{self.constitution_ver}.md")
        print(f"schema 基準 : data_schema {self.schema_ver}")
        if universe:
            print(f"執行 universe : {universe}")
        print(f"執行 phase : {phase_label}")
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
    parser = argparse.ArgumentParser(
        description="Quantum Finance 主權同步引擎 (v1.9 — 5.5.3 對齊版)",
        epilog="選股 phase 範例：python scripts/ingestion/sovereign_sync_engine.py --universe research --all --days 730；"
               "核心 phase 範例：python scripts/ingestion/sovereign_sync_engine.py --universe core --all --days 730",
    )
    parser.add_argument("--id", type=str, help="指定標的 ID (如 2330)")
    parser.add_argument("--universe", type=str, choices=["research", "convex", "core"],
                        help="指定標的範圍（research = 研究宇宙、convex = 凸性宇宙、core = 已提交核心宇宙）")
    parser.add_argument("--source", type=str, choices=["finmind", "fred"], help="指定數據源")
    parser.add_argument("--dataset", type=str, help="指定單一 dataset 名稱")
    parser.add_argument("--seed", action="store_true",
                        help="種子灌溉模式（同步 TaiwanStockInfo 全市場資產名冊）")
    parser.add_argument("--all", action="store_true",
                        help="使用 FINMIND_API_TABLES 全表 (除 TaiwanStockInfo) 取代 DEFAULT_FINMIND_DATASETS (4 表)")
    parser.add_argument("--days", type=int, default=30,
                        help="同步天數 (預設 30；CoreScore v0.2 選股 phase 建議 730)")
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
