"""
data_schema.py v1.7 (Quantum Finance Edition)
================================================================================
數據契約字典與全系統執行矩陣 — 旗艦終極版 (Quantum v5.2 標準)
負責定義與 API 1:1 對齊的資料表映射，並作為全系統維運指令的「權威索引」。

修訂歷程：
  v1.7 (2026-05-11): [極致] 補全全鏈路維運矩陣，涵蓋個股、全表、核心股、強制更新等所有情境。
  v1.6 (2026-05-11): [標準] 恢復 CLI 介面，導入延遲匯入機制解決循環引用。

【全系統維運指令矩陣 (The Comprehensive Master Matrix)】
┌──────────────────────────────────────┬────────────────────────────────────────────────────────┐
│ 維運場景                             │ 建議執行指令                                           │
├──────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [單一實體表：結構稽核]            │ $ python scripts/core/data_schema.py --check TaiwanStockPrice │
│ 2. [全系統：資料表初始化/自癒]       │ $ python scripts/core/data_schema.py --init            │
│ 3. [個股 x 單一表：數據同步]         │ $ python scripts/ingestion/template_fetcher.py --id 2330 --dataset TaiwanStockPrice │
│ 4. [個股 x 所有表：全量同步]         │ $ python scripts/ingestion/template_fetcher.py --id 2330 --all_datasets │
│ 5. [核心股 x 所有表：全量同步]       │ $ python scripts/ingestion/template_fetcher.py --universe core --all_datasets │
│ 6. [核心股 x 所有表：強制重刷]       │ $ python scripts/ingestion/template_fetcher.py --universe core --all_datasets --force │
│ 7. [全系統：實體表強制重鑄(含數據)]  │ $ python scripts/core/data_schema.py --init --force     │
└──────────────────────────────────────┴────────────────────────────────────────────────────────┘

【可觀測性紀錄 (Observability)】
  - 統一日誌 (Unified): pipeline_execution_log (Task: schema_init_v1.7)
  - 專項審計 (Audit): data_audit_log (Action: SCHEMA_INIT / SCHEMA_FORCE_REBUILD)
================================================================================
"""
import sys, argparse
from pathlib import Path
from datetime import datetime

# ── 靜像數據契約定義 (Static Registry) ──
DATASET_SCHEMA_MAP = {
    "TaiwanStockInfo": {
        "table": "TaiwanStockInfo",
        "unique_cols": ["stock_id"],
        "sql": """
            CREATE TABLE IF NOT EXISTS "TaiwanStockInfo" (
                "industry_category" VARCHAR(100),
                "stock_id" VARCHAR(50) PRIMARY KEY,
                "stock_name" VARCHAR(100),
                "type" VARCHAR(50),
                "date" DATE,
                "last_updated" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """
    },
    "TaiwanStockPrice": {
        "table": "TaiwanStockPrice",
        "unique_cols": ["stock_id", "date"],
        "sql": """
            CREATE TABLE IF NOT EXISTS "TaiwanStockPrice" (
                "date" DATE,
                "stock_id" VARCHAR(50),
                "open" NUMERIC(20, 6),
                "max" NUMERIC(20, 6),
                "min" NUMERIC(20, 6),
                "close" NUMERIC(20, 6),
                "spread" NUMERIC(20, 6),
                "Trading_Volume" BIGINT,
                "Trading_money" BIGINT,
                "Trading_turnover" BIGINT,
                PRIMARY KEY ("stock_id", "date")
            );
        """
    },
    "TaiwanStockInstitutionalInvestorsBuySell": {
        "table": "TaiwanStockInstitutionalInvestorsBuySell",
        "unique_cols": ["stock_id", "date", "name"],
        "sql": """
            CREATE TABLE IF NOT EXISTS "TaiwanStockInstitutionalInvestorsBuySell" (
                "date" DATE,
                "stock_id" VARCHAR(50),
                "buy" BIGINT,
                "sell" BIGINT,
                "name" VARCHAR(100),
                PRIMARY KEY ("stock_id", "date", "name")
            );
        """
    },
    "TaiwanStockMarginPurchaseShortSale": {
        "table": "TaiwanStockMarginPurchaseShortSale",
        "unique_cols": ["stock_id", "date"],
        "sql": """
            CREATE TABLE IF NOT EXISTS "TaiwanStockMarginPurchaseShortSale" (
                "date" DATE,
                "stock_id" VARCHAR(50),
                "MarginPurchaseBuy" BIGINT,
                "MarginPurchaseSell" BIGINT,
                "MarginPurchaseCashRepayment" BIGINT,
                "MarginPurchaseYesterdayBalance" BIGINT,
                "MarginPurchaseTodayBalance" BIGINT,
                "MarginPurchaseLimit" BIGINT,
                "ShortSaleBuy" BIGINT,
                "ShortSaleSell" BIGINT,
                "ShortSaleCashRepayment" BIGINT,
                "ShortSaleYesterdayBalance" BIGINT,
                "ShortSaleTodayBalance" BIGINT,
                "ShortSaleLimit" BIGINT,
                "OffsetLoanAndShort" BIGINT,
                "Note" VARCHAR(255),
                PRIMARY KEY ("stock_id", "date")
            );
        """
    },
    "FredData": {
        "table": "FredData",
        "unique_cols": ["series_id", "date"],
        "sql": """
            CREATE TABLE IF NOT EXISTS "FredData" (
                "date" DATE,
                "series_id" VARCHAR(100),
                "value" NUMERIC(20, 6),
                PRIMARY KEY ("series_id", "date")
            );
        """
    }
}

def ensure_all_schemas(drop_existing=False):
    """執行全量資料表初始化 (Lazy Import 解決循環引用)"""
    _THIS_DIR = Path(__file__).resolve().parent
    _PROJECT_ROOT = _THIS_DIR.parent.parent
    if str(_PROJECT_ROOT / "scripts") not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

    from core.db_utils import db_transaction, record_lifecycle, write_data_audit_log
    task_name = "schema_init_v1.7"
    
    with record_lifecycle(task_name, "maintenance", "DATABASE"):
        print("\n" + "="*60)
        print(f"🚀 Quantum Finance: 終極數據契約初始化任務 (v1.7)")
        print("="*60)
        with db_transaction() as cur:
            for dataset, config in DATASET_SCHEMA_MAP.items():
                if drop_existing:
                    cur.execute(f'DROP TABLE IF EXISTS "{config["table"]}" CASCADE')
                    action = "SCHEMA_FORCE_REBUILD"
                else:
                    action = "SCHEMA_INIT"
                cur.execute(config["sql"])
                write_data_audit_log(config["table"], "SYSTEM", datetime.now().strftime("%Y-%m-%d"), action, 0)
                print(f"  ✅ [{dataset}] -> {config['table']} 對齊成功")
        print("-" * 60)
        print(f"✨ 混合日誌已同步完成。")
        print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true", help="初始化所有資料表")
    parser.add_argument("--force", action="store_true", help="強制重鑄表結構")
    parser.add_argument("--check", help="檢查特定表結構")
    args = parser.parse_args()
    
    if args.init:
        ensure_all_schemas(drop_existing=args.force)
    elif args.check:
        if args.check in DATASET_SCHEMA_MAP:
            print(f"📄 Dataset: {args.check}\n📜 SQL: \n{DATASET_SCHEMA_MAP[args.check]['sql']}")
        else:
            print(f"❌ 找不到數據集: {args.check}")
    else:
        print(f"Quantum Data Schema v1.7 | Active Datasets: {list(DATASET_SCHEMA_MAP.keys())}")
