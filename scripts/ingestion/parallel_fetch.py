"""
parallel_fetch.py v7.6 (Quantum Finance Edition)
================================================================================
旗艦級並行編排器 — 數據同步核心 (Quantum v5.1 標準)
負責協調多進程抓取任務，支援標的、資料表與宇宙 (Universe) 的多維度同步。

修訂歷程：
  v7.6 (2026-05-10): [修復] 實作「市場 vs 個股」任務分離，修復 ReturnIndex 參數錯誤。
  v7.5 (2026-05-10): [核心] 優化執行範例矩陣，整合 v5.1 混合日誌規範。
  v7.4 (2026-05-10): [擴充] 完善五級度執行矩陣，支援 universe=all 全量同步。
  v7.3 (2026-05-10): [修復] 實作 PID-Aware Pool 處理多進程 DB 連線衝突。

【執行範例矩陣 — 終極編排方案】
1. 單一標的、所有表格同步 (最常用)：
   python scripts/ingestion/parallel_fetch.py --stock_id 2330 --table ALL
2. 單一標的、特定表格同步：
   python scripts/ingestion/parallel_fetch.py --stock_id 2330 --table Technical
3. 核心宇宙標的 (140+ 檔)、所有表格同步：
   python scripts/ingestion/parallel_fetch.py --universe core --table ALL
4. 全市場標的 (2000+ 檔)、所有表格「強制」回刷歷史：
   python scripts/ingestion/parallel_fetch.py --universe all --table ALL --force
5. 特定表格之缺失標的自動回填：
   python scripts/ingestion/fetch_missing_stocks_data.py (內部調用此編排器)
================================================================================
"""
import os, sys, logging, time, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ── 終極路徑自癒 Bootstrap ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent if _THIS_DIR.name != "scripts" else _THIS_DIR
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_SCRIPTS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR.parent))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import get_db_stock_ids, write_pipeline_log
except ImportError:
    import path_setup
    path_setup.ensure_scripts_on_path(__file__)
    from db_utils import get_db_stock_ids, write_pipeline_log

logger = logging.getLogger(__name__)

# 表格與腳本映射矩陣 (個股級)
STOCK_TABLE_MAP = {
    "Technical": "fetch_technical_data.py",
    "PriceAdj": "fetch_price_adj_data.py",
    "Chip": "fetch_chip_data.py",
    "AdvancedChip": "fetch_advanced_chip_data.py",
    "Fundamental": "fetch_fundamental_data.py",
    "CashFlows": "fetch_cash_flows_data.py",
    "Revenue": "fetch_month_revenue.py",
    "PER": "fetch_macro_fundamental_data.py",
    "News": "fetch_news_data.py",
    "Dividend": "fetch_event_risk_data.py",
    "Sponsor": "fetch_sponsor_chip_data.py",
    "BlockTrading": "fetch_block_trading.py"
}

# 市場級表格映射 (僅需執行一次)
MARKET_TABLE_MAP = {
    "ReturnIndex": "fetch_total_return_index.py"
}

def run_task(cmd):
    return os.system(cmd)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock_id", type=str)
    parser.add_argument("--universe", type=str, choices=["core", "all"])
    parser.add_argument("--table", type=str, default="ALL")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    start_time = time.time()
    
    # 標的選取邏輯
    if args.stock_id:
        target_ids = [args.stock_id]
    elif args.universe == "core":
        target_ids = get_db_stock_ids(active_only=True)
    elif args.universe == "all":
        target_ids = get_db_stock_ids(active_only=False)
    else:
        logger.error("❌ 必須指定 --stock_id 或 --universe")
        return

    # 取得目標表格列表
    if args.table == "ALL":
        target_stock_tables = STOCK_TABLE_MAP.keys()
        target_market_tables = MARKET_TABLE_MAP.keys()
    else:
        target_stock_tables = [args.table] if args.table in STOCK_TABLE_MAP else []
        target_market_tables = [args.table] if args.table in MARKET_TABLE_MAP else []
    
    tasks = []
    
    # 1. 產生個股任務
    for sid in target_ids:
        for tbl in target_stock_tables:
            script = STOCK_TABLE_MAP[tbl]
            cmd = f"{sys.executable} {_THIS_DIR}/{script} --stock_id {sid}"
            if args.force: cmd += " --force"
            tasks.append(cmd)

    # 2. 產生市場任務 (僅執行一次)
    for tbl in target_market_tables:
        script = MARKET_TABLE_MAP[tbl]
        cmd = f"{sys.executable} {_THIS_DIR}/{script}"
        if args.force: cmd += " --force"
        tasks.append(cmd)

    logger.info(f"🚀 啟動並行編排: 標的數={len(target_ids)}, 任務總數={len(tasks)}, 平行度={args.workers}")
    
    success_count = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_task, cmd): cmd for cmd in tasks}
        for future in as_completed(futures):
            if future.result() == 0:
                success_count += 1

    duration = int((time.time() - start_time) * 1000)
    write_pipeline_log("ParallelOrchestrator", f"Universe:{args.universe or args.stock_id}", 
                       "SUCCESS" if success_count == len(tasks) else "PARTIAL", 
                       "Orchestration", duration_ms=duration, rows=success_count)
    
    logger.info(f"🏁 編排任務結束。成功率: {success_count}/{len(tasks)}，耗時: {duration/1000:.2f}s")

if __name__ == "__main__":
    main()
