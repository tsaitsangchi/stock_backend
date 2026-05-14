"""
sovereign_sync_engine.py v1.7 (Quantum Finance Sovereign Sync Engine)
================================================================================
**最後更新日期**: 2026-05-13
**主權狀態**: GENESIS COMPLETED (憲法 v5.4.18 對齊)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
1. [Sovereign Alignment]: 數據寫入必須嚴格遵循 data_schema.py 定義之大小寫與型態。
2. [Idempotency]: 使用 ON CONFLICT 邏輯，確保數據重刷之安全性。
3. [Hybrid Observability]: 同步行為必須觸發全量日誌，並顯示執行摘要；
   任何分支（成功 / 無新資料 / 失敗）皆必須計數並輸出明細，禁止靜默丟失。
4. [Universal Ingestion]: 統一管理 FinMind 與 FRED 等多元供應鏈數據流入。

## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix)
| 維運需求場景 (Scenario)   | 權威指令 / 建議用法 (Exhaustive Examples)                             | 對齊模組 |
| :----------------------- | :-------------------------------------------------------------------- | :--- |
| **1. [同步：種子資產名冊灌溉]** | `$ python scripts/ingestion/sovereign_sync_engine.py --seed`        | sync_engine v1.7 |
| **2. [同步：單一標的全數據]** | `$ python scripts/ingestion/sovereign_sync_engine.py --id 2330 --all` | sync_engine v1.7 |
| **3. [同步：宏觀指標同步]**   | `$ python scripts/ingestion/sovereign_sync_engine.py --source fred`  | sync_engine v1.7 |
| **4. [同步：指定數據集同步]** | `$ python scripts/ingestion/sovereign_sync_engine.py --id 2330 --dataset TaiwanStockPrice`| sync_engine v1.7 |

## 📜 三、全修訂歷程 (Full Revision History)
| **v1.7** | 2026-05-13 | Auto-patch | **Bug #1 修補**：sync_fred() 補完 empty-data 失敗分支與 dropna 後空集合分支；新增 `res.raise_for_status()` 與 `timeout=30`；阻斷 UNRATE / VIXCLS 等靜默丟資料；符合憲法第二條 Hybrid Observability。 | **ACTIVE** |
| v1.6 | 2026-05-13 | Antigravity | **創世圓滿**：對齊憲法 v5.4.18；對齊「大憲章」命名體系。 | ARCHIVED |
| v1.5 | 2026-05-13 | Antigravity | **崩潰修復**：對齊憲法 v5.4.17；實作 FRED 數據清洗。 | ARCHIVED |
| v1.4 | 2026-05-13 | Antigravity | **全量實證對齊**：對齊憲法 v5.4.16；確立全譜系大同步地位。 | ARCHIVED |
| v1.3 | 2026-05-13 | Antigravity | **全量大同步**：對齊憲法 v5.4.15；達成檔案治理全量收斂。 | ARCHIVED |
| v1.2 | 2026-05-13 | Antigravity | **治權完備**：對齊憲法 v5.4.14；確立歷史歷程追加規範與灌溉標準。 | ARCHIVED |
| v1.1 | 2026-05-13 | Antigravity | **治權對齊**：對齊憲法 v5.4.13；整合治權基準報告邏輯。 | ARCHIVED |
| v1.0 | 2026-05-13 | Antigravity | **革命性重構**：取代 template_fetcher，實現 1:1 大小寫主權同步。 | ARCHIVED |
================================================================================
"""
import sys, os, time
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime, timedelta
import argparse

# ── 系統級架構引導 ──
_THIS_FILE = Path(__file__).resolve()
_INGESTION_DIR = _THIS_FILE.parent
_SCRIPTS_DIR = _INGESTION_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path: sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from core.db_utils import get_db_connection, record_lifecycle, write_data_audit_log, get_core_stocks_from_db
    from core.data_schema import DATASET_REGISTRY
    from core.finmind_client import FinMindClient
except ImportError:
    print("❌ 核心組件導入失敗，請確認 core/ 目錄")
    sys.exit(1)

class SovereignSyncEngine:
    def __init__(self):
        self.fm_client = FinMindClient()
        self.fred_key = os.getenv("FRED_API_KEY")
        self.stats = {"success": 0, "failed": 0, "rows": 0, "details": []}
        self.constitution_ver = "大憲章_v5.4.18"

    def _upsert_to_db(self, table_name, df):
        """核心主權寫入邏輯：1:1 大小寫對齊 + Idempotency"""
        if df.empty: return 0

        config = DATASET_REGISTRY.get(table_name)
        if not config:
            raise ValueError(f"❌ 憲法未定義表名: {table_name}")

        conn = get_db_connection()
        cur = conn.cursor()

        # 遵循 [Absolute Case Sovereignty]：欄位名稱加雙引號
        cols = [f'"{c}"' for c in df.columns]
        placeholders = ", ".join(["%s"] * len(df.columns))

        # 建立 ON CONFLICT 語句
        unique_cols = [f'"{c}"' for c in config["unique_constraints"]]
        update_cols = [f'{c} = EXCLUDED.{c}' for c in cols if c not in unique_cols]

        sql = f"""
            INSERT INTO "{table_name}" ({", ".join(cols)})
            VALUES ({placeholders})
            ON CONFLICT ({", ".join(unique_cols)})
            DO UPDATE SET {", ".join(update_cols)}
        """

        try:
            # 批量寫入以提升性能
            data = [tuple(x) for x in df.values]
            cur.executemany(sql, data)
            conn.commit()
            rows = len(df)
            write_data_audit_log(table_name, "SYNC", df.iloc[0].get("date", "N/A"), "UPSERT", rows)
            return rows
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cur.close(); conn.close()

    def sync_finmind(self, stock_id, dataset_name, start_date):
        """同步 FinMind 數據集"""
        try:
            print(f"📡 正在獲取 FinMind: {stock_id} / {dataset_name}...")
            params = {
                "dataset": dataset_name,
                "data_id": stock_id,
                "start_date": start_date,
                "token": self.fm_client.token
            }
            res = requests.get(self.fm_client.api_url, params=params)
            data = res.json().get("data", [])

            if data:
                df = pd.DataFrame(data)
                # 強制轉換日期格式以對齊 DATE 型態
                # 實施 [Data Sanctity] 數據聖潔原則：處理無效日期
                if "date" in df.columns:
                    df = df[df["date"].notna() & (df["date"] != "None")]
                    df["date"] = pd.to_datetime(df["date"]).dt.date

                rows = self._upsert_to_db(dataset_name, df)
                self.stats["rows"] += rows
                self.stats["success"] += 1
                self.stats["details"].append(f"✅ {dataset_name} ({stock_id}): {rows} 筆對齊成功")
            else:
                self.stats["details"].append(f"⚠️ {dataset_name} ({stock_id}): 無新數據")
        except Exception as e:
            self.stats["failed"] += 1
            self.stats["details"].append(f"❌ {dataset_name} ({stock_id}) 失敗: {str(e)}")

    def sync_fred(self, series_id):
        """同步 FRED 宏觀數據 (v1.7 修補：補完 empty-data 失敗分支，杜絕靜默丟失)"""
        try:
            print(f"📡 正在獲取 FRED: {series_id}...")
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series_id,
                "api_key": self.fred_key,
                "file_type": "json",
                "limit": 1000,           # 獲取最近 1000 點
                "sort_order": "desc"
            }
            res = requests.get(url, params=params, timeout=30)
            res.raise_for_status()       # [v1.7] 4xx/5xx 強制 raise 進 except
            payload = res.json()
            data = payload.get("observations", [])

            # [v1.7 Bug#1 修補] API 回 0 個 observation - 走 failed 分支
            # 違反 Hybrid Observability 之靜默吐零情境，由本分支封堵。
            if not data:
                err = payload.get("error_message") or "API 回傳 0 個 observation"
                self.stats["failed"] += 1
                self.stats["details"].append(f"❌ FRED/{series_id} 失敗: {err}")
                return

            df = pd.DataFrame(data)
            df["series_id"] = series_id
            # 只保留對齊 FredData 表結構的欄位
            target_cols = ["date", "series_id", "value", "realtime_start", "realtime_end"]
            df = df[target_cols]
            # 實施 [Data Sanctity] 數據聖潔原則：處理 FRED 缺失值符號 '.'
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["value"])

            # [v1.7 Bug#1 修補] 數據聖潔清洗後 df 全空 - 走 failed 分支
            if df.empty:
                self.stats["failed"] += 1
                self.stats["details"].append(f"❌ FRED/{series_id} 失敗: 全部 observation 在數據聖潔清洗後為空")
                return

            df["date"] = pd.to_datetime(df["date"]).dt.date

            rows = self._upsert_to_db("FredData", df)
            self.stats["rows"] += rows
            self.stats["success"] += 1
            self.stats["details"].append(f"✅ FRED/{series_id}: {rows} 筆主權同步完成")
        except Exception as e:
            self.stats["failed"] += 1
            self.stats["details"].append(f"❌ FRED/{series_id} 失敗: {str(e)}")

    def run(self, stock_id=None, universe=None, source=None, dataset=None, days=30):
        """主權同步引擎入口"""
        start_time = time.time()
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        task_name = f"sync_{source or 'all'}_{stock_id or universe or 'macro'}"

        with record_lifecycle(task_name, category="ingestion", stock_id=stock_id or "SYSTEM"):
            # 1. 處理標的
            stocks = []
            if stock_id: stocks = [stock_id]
            elif universe == "core": stocks = get_core_stocks_from_db()

            # 2. 處理 FinMind 同步
            if source == "finmind" or not source:
                # [種子模式]：如果指定 TaiwanStockInfo，則執行全市場清單同步
                if dataset == "TaiwanStockInfo" or args.seed:
                    self.sync_finmind("", "TaiwanStockInfo", start_date)

                # [數據模式]：同步特定標的之數據集
                target_datasets = [dataset] if dataset else [
                    "TaiwanStockPrice", "TaiwanStockInstitutionalInvestorsBuySell",
                    "TaiwanStockMarginPurchaseShortSale", "TaiwanStockPER"
                ]
                # 排除種子表，避免在個股迴圈中重複執行
                target_datasets = [d for d in target_datasets if d != "TaiwanStockInfo"]

                for sid in stocks:
                    for ds in target_datasets:
                        self.sync_finmind(sid, ds, start_date)

            # 3. 處理 FRED 同步
            if (source == "fred" or not source) and not stock_id:
                # 引用憲法 v5.4.5 第九章之核心指標
                fred_list = ["DFF", "UNRATE", "T10Y2Y", "VIXCLS"]
                for fsid in fred_list:
                    self.sync_fred(fsid)

            self.report_results(start_time)

    def report_results(self, start_time):
        """同步結果詳細報告 (旗艦輸出)"""
        print("\n" + "🛡️" * 40)
        print("🚀 Quantum Finance: 主權同步引擎執行摘要")
        print("🛡️" * 40)
        print(f"治權基準 : 系統架構{self.constitution_ver}.md")
        print(f"核心技術 : Absolute Case Sovereignty (1:1 鏡像)")
        print("─" * 80)
        for d in self.stats["details"]: print(d)
        print("─" * 80)
        print(f"📈 成功同步項目 : {self.stats['success']}")
        print(f"❌ 失敗同步項目 : {self.stats['failed']}")
        print(f"📝 總計寫入筆數 : {self.stats['rows']}")
        print(f"🕒 總計耗時     : {(time.time() - start_time):.2f} s")
        print(f"⚖️  主權判定     : {'PERFECT' if self.stats['failed'] == 0 else 'WARNING'}")
        print("🛡️" * 40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum Finance 主權同步引擎 (v1.7)")
    parser.add_argument("--id", type=str, help="指定標的 ID (如 2330)")
    parser.add_argument("--universe", type=str, choices=["core"], help="指定標的範圍")
    parser.add_argument("--source", type=str, choices=["finmind", "fred"], help="指定數據源")
    parser.add_argument("--dataset", type=str, help="指定數據集")
    parser.add_argument("--seed", action="store_true", help="種子灌溉模式 (獲取全市場清單)")
    parser.add_argument("--all", action="store_true", help="全數據灌溉模式")
    parser.add_argument("--days", type=int, default=30, help="同步天數 (預設 30 天)")

    args = parser.parse_args()
    engine = SovereignSyncEngine()
    engine.run(stock_id=args.id, universe=args.universe, source=args.source, dataset=args.dataset, days=args.days)