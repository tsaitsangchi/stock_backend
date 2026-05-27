"""
data_pipeline.py v7.0 (Quantum Finance Edition)
================================================================================
數據管線總指揮官 — 混合日誌標準版 (Quantum v5.1)
負責協調整個核心宇宙全集(dynamic per §14.7-BW,無 hardcoded 128)之數據同步、稽核與初步處理。

修訂歷程：
  v7.0 (2026-05-10): [核心] 導入 --table 參數，支援全量核心標的之特定資料表同步。
  v6.9 (2026-05-10): [核心] 升級 Quantum Finance 物理特徵相容性。

【執行範例矩陣 — 數據同步方案】
1. 增量同步 (全核心宇宙 dynamic per §14.7-BW x 所有表):
   python scripts/pipeline/data_pipeline.py --all
2. 全核心 x 指定表 (僅籌碼面)：
   python scripts/pipeline/data_pipeline.py --all --table chip
3. 全核心 x 指定表 x 強制全量更新 (從 2010 年起)：
   python scripts/pipeline/data_pipeline.py --all --table price --start 2010-01-01
4. 單一個股同步 (所有表)：
   python scripts/pipeline/data_pipeline.py --stock_id 2330
5. 單一個股 x 指定表 x 強制全量更新：
   python scripts/pipeline/data_pipeline.py --stock_id 2330 --table revenue --start 2015-01-01
6. 系統強制全宇宙全資料表強制更新 (慎用)：
   python scripts/pipeline/data_pipeline.py --all --start 2000-01-01
================================================================================
"""
import sys, logging, time, argparse
from pathlib import Path

# ── 終極路徑自癒 Bootstrap ──
try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
except ImportError:
    _THIS_DIR = Path(__file__).resolve().parent
    if str(_THIS_DIR) not in sys.path: sys.path.insert(0, str(_THIS_DIR))

try:
    from core.db_utils import get_db_stock_ids, write_pipeline_log
    from ingestion.parallel_fetch import fetch_stock_data_unit
except ImportError:
    from db_utils import get_db_stock_ids, write_pipeline_log
    from parallel_fetch import fetch_stock_data_unit

def run_master_pipeline(stock_id: str = None, start_date: str = None, table: str = None):
    t_start = time.monotonic()
    target_ids = [stock_id] if stock_id else get_db_stock_ids()
    
    logging.info(f"🚀 [Pipeline] 同步開始 (標的: {len(target_ids)} 檔, Table: {table or 'ALL'}, Start: {start_date or 'Auto'})...")
    success_count = 0; total_rows = 0
    
    for sid in target_ids:
        try:
            rows = fetch_stock_data_unit(sid, start_date=start_date, table=table)
            success_count += 1; total_rows += (rows or 0)
        except Exception as e:
            logging.error(f"❌ {sid} 同步異常: {e}")
            
    elapsed_ms = int((time.monotonic() - t_start) * 1000)
    write_pipeline_log(f"master_pipeline_{table or 'all'}", "SYSTEM", "success", "pipeline", elapsed_ms, total_rows)
    logging.info(f"🏁 [Pipeline] 完成！成功: {success_count}/{len(target_ids)}，總量: {total_rows}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock_id", type=str)
    parser.add_argument("--table", type=str, help="price, chip, fundamental, revenue")
    parser.add_argument("--start", type=str)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    
    if args.all:
        run_master_pipeline(start_date=args.start, table=args.table)
    elif args.stock_id:
        run_master_pipeline(stock_id=args.stock_id, start_date=args.start, table=args.table)
    else:
        print("💡 請指定 --stock_id <id> 或 --all 以啟動同步任務。")
