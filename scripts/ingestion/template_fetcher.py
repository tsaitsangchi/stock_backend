"""
template_fetcher.py v1.4 (Quantum Finance Edition)
================================================================================
數據採集標準模板 — 嚴格去重安全版 (Quantum v5.2 標準)
負責執行 FinMind 數據抓取，具備批次資料去重邏輯，防止 ON CONFLICT 衝突。

修訂歷程：
  v1.4 (2026-05-11): [修復] 加入資料去重邏輯，防止同一批次內重複 ID 導致的寫入失敗。
  v1.3 (2026-05-11): [修復] 支援動態 ID 欄位 (如 series_id) 與非時間序列數據集。
================================================================================
"""
import sys, logging, time, argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from core import (
    FinMindClient, bulk_upsert, get_latest_date, get_db_stock_ids,
    record_lifecycle, write_data_audit_log
)
from core.data_schema import DATASET_SCHEMA_MAP

def fetch_and_store(stock_id: str, dataset: str, start_date: str = None, force: bool = False):
    client = FinMindClient()
    if dataset not in DATASET_SCHEMA_MAP: return 0

    config = DATASET_SCHEMA_MAP[dataset]
    table_name = config["table"]
    unique_cols = config["unique_cols"]

    id_col = "series_id" if dataset == "FredData" else "stock_id"
    if force:
        start_date = "2010-01-01"
    elif not start_date:
        last_date = get_latest_date(table_name, stock_id, id_column=id_col)
        start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d") if last_date else ("2020-01-01" if dataset != "TaiwanStockInfo" else "")

    with record_lifecycle(f"fetch_{dataset}", category="ingestion", stock_id=stock_id):
        data = client.get_data(dataset, stock_id, start_date)
        if not data:
            logging.info(f"[{stock_id}] {dataset} 無新數據")
            return 0

        # ── 🛡️ 嚴格去重邏輯 ──
        # 防止同一批次內出現重複鍵導致 ON CONFLICT 失敗
        df = pd.DataFrame(data)
        df = df.drop_duplicates(subset=unique_cols, keep='last')
        clean_data = df.to_dict('records')

        rows = bulk_upsert(table_name, clean_data, unique_cols)
        
        audit_date = clean_data[-1].get("date", datetime.now().strftime("%Y-%m-%d"))
        write_data_audit_log(table_name, stock_id, audit_date, "FETCH_SYNC", rows)
        
        logging.info(f"✅ [{stock_id}] {dataset} 同步 {rows} 筆")
        return rows

def run_pipeline(args):
    if args.id: stock_ids = [args.id]
    elif args.universe == "core": stock_ids = get_db_stock_ids(core_only=True)
    else: return

    datasets = list(DATASET_SCHEMA_MAP.keys()) if args.all_datasets else [args.dataset]
    total_rows = 0
    for sid in stock_ids:
        for ds in datasets:
            try:
                total_rows += fetch_and_store(sid, ds, args.start, args.force)
                time.sleep(0.1)
            except Exception as e:
                logging.error(f"❌ [{sid}] {ds} 失敗: {e}")
    print(f"\n✨ 同步任務完成！總計入庫 {total_rows} 筆。")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", help="指定標的 ID")
    parser.add_argument("--universe", choices=["core"])
    parser.add_argument("--dataset", default="TaiwanStockPrice")
    parser.add_argument("--all_datasets", action="store_true")
    parser.add_argument("--start")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_pipeline(args)
