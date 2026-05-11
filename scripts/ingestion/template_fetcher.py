"""
template_fetcher.py v1.1 (Quantum Finance Edition)
================================================================================
數據採集標準模板 — Schema 自癒與增量更新版 (Quantum v5.2 標準)
負責執行 FinMind 數據抓取，具備自動建表 (Auto-Healing) 與物理 Parquet 存檔功能。

修訂歷程：
  v1.1 (2026-05-11): [結構] 整合 Schema 哨兵，抓取前自動檢查並建立缺失資料表。
  v1.0 (2026-05-11): [首發] 整合 Polars 與混合模式日誌。

【執行範例矩陣 (Ingestion Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / 用法                                        │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [全自動建表與抓取]        │ $ python scripts/ingestion/template_fetcher.py --id 2330│
│ 2. [全核心標的批量更新]      │ $ python scripts/ingestion/template_fetcher.py --all   │
└──────────────────────────────┴────────────────────────────────────────────────────────┘
================================================================================
"""
import sys, logging, time, argparse
from pathlib import Path
from datetime import datetime, timedelta
import polars as pl

# ── 系統級架構引導 ──
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from core import (
    FinMindClient, bulk_upsert, get_latest_date, get_db_stock_ids,
    record_lifecycle, write_data_audit_log, db_transaction,
    get_raw_data_dir
)

# ── Schema 哨兵：硬核保底定義 ──
DATASET_SCHEMA_MAP = {
    "TaiwanStockPrice": {
        "table": "stock_price_day",
        "sql": """
            CREATE TABLE IF NOT EXISTS stock_price_day (
                date DATE, stock_id VARCHAR(20), 
                open FLOAT, max FLOAT, min FLOAT, close FLOAT, 
                spread FLOAT, Trading_Volume BIGINT, Trading_money BIGINT, 
                Trading_turnover BIGINT, 
                PRIMARY KEY (stock_id, date)
            );
        """,
        "unique_cols": ["stock_id", "date"]
    }
}

def ensure_dataset_table(dataset: str):
    """Schema 哨兵：確保資料庫中存在對應的表格"""
    if dataset not in DATASET_SCHEMA_MAP:
        logging.warning(f"⚠️ 警告：未定義 {dataset} 的保底 Schema，將跳過建表檢查。")
        return
    
    config = DATASET_SCHEMA_MAP[dataset]
    with db_transaction() as cur:
        cur.execute(config["sql"])
    logging.info(f"🛡️ [Schema Sentinel] {config['table']} 結構校驗通過。")

def fetch_and_store(stock_id: str, dataset: str, start_date: str = None):
    """標準化採集、轉換、存儲 (ETL) 流程"""
    client = FinMindClient()
    table_name = DATASET_SCHEMA_MAP.get(dataset, {}).get("table", dataset.lower())
    unique_cols = DATASET_SCHEMA_MAP.get(dataset, {}).get("unique_cols", ["stock_id", "date"])

    # 1. 增量偵測
    if not start_date:
        last_date = get_latest_date(table_name, stock_id)
        start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d") if last_date else "2020-01-01"

    # 2. 啟動生命週期監控
    with record_lifecycle(f"fetch_{dataset}_{stock_id}", category="ingestion", stock_id=stock_id):
        # A. 執行 API 抓取
        data = client.get_data(dataset, stock_id, start_date)
        if not data:
            logging.info(f"[{stock_id}] 無新數據可供下載 (自 {start_date} 起)")
            return 0

        # B. 使用 Polars 進行標準化處理
        df = pl.DataFrame(data)
        
        # C. 物理存檔 (Parquet)
        raw_save_path = get_raw_data_dir() / f"{dataset}_{stock_id}.parquet"
        df.write_parquet(raw_save_path)
        
        # D. 資料庫批量寫入
        rows = bulk_upsert(table_name, data, unique_cols)
        
        # E. 分類稽核日誌
        write_data_audit_log(table_name, stock_id, df["date"][-1], "FETCH_INSERT", rows)
        
        logging.info(f"✅ [{stock_id}] 同步 {rows} 筆數據 -> {table_name} & {raw_save_path.name}")
        return rows

def run_pipeline(target_id: str = None, run_all: bool = False, manual_start: str = None):
    dataset = "TaiwanStockPrice"
    
    # 預先執行 Schema 哨兵校驗
    ensure_dataset_table(dataset)
    
    if run_all:
        stock_ids = get_db_stock_ids(is_core=True)
        logging.info(f"🚀 啟動批量更新任務：共 {len(stock_ids)} 檔標的")
    elif target_id:
        stock_ids = [target_id]
    else:
        logging.error("❌ 錯誤：請指定 --id 或 --all 參數。")
        return

    total_rows = 0
    for sid in stock_ids:
        try:
            total_rows += fetch_and_store(sid, dataset, manual_start)
            time.sleep(0.5)
        except Exception as e:
            logging.error(f"❌ [{sid}] 任務失敗: {e}")
            continue
            
    print(f"\n✨ 任務完成！總計同步 {total_rows} 筆數據。")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", help="指定抓取單一標的 ID (如 2330)")
    parser.add_argument("--all", action="store_true", help="執行全核心標的批量更新")
    parser.add_argument("--start", help="手動指定起始日期 (YYYY-MM-DD)")
    args = parser.parse_args()

    run_pipeline(target_id=args.id, run_all=args.all, manual_start=args.start)
