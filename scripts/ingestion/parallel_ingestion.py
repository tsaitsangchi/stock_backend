"""
parallel_ingestion.py v8.0 (Quantum Finance Edition)
================================================================================
旗艦級並行編排器 — 數據入庫總管 (Quantum v5.2 標準)
負責協調多進程 Ingestion 任務，支援標的、資料表與宇宙 (Universe) 的多維度同步。

修訂歷程：
  v8.0 (2026-05-11): [封裝] 啟動「全量更名計畫」，將所有 fetch_ 腳本更名為 ingest_。
  v7.6 (2026-05-10): [核心] 實作「市場 vs 個股」任務分離。

執行範例 (Comprehensive Usage Examples):
  1. [單一標的全量入庫] 對台積電(2330)執行所有資料表同步:
     python scripts/ingestion/parallel_ingestion.py --stock_id 2330 --table ALL

  2. [核心宇宙同步] 對 128 檔核心股票執行籌碼(Chip)與技術(Technical)同步:
     python scripts/ingestion/parallel_ingestion.py --universe core --table Chip
     python scripts/ingestion/parallel_ingestion.py --universe core --table Technical

  3. [全市場回刷] 對所有標的執行「強制」歷史回刷 (慎用):
     python scripts/ingestion/parallel_ingestion.py --universe all --table PriceAdj --force

  4. [系統監測] 透過儀表板查看昨晚的批量執行狀態 (SQL):
     SELECT * FROM pipeline_execution_log WHERE category = 'Orchestration' ORDER BY created_at DESC;
================================================================================
"""
import os, sys, logging, time, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime

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
    from core.db_utils import get_db_stock_ids, write_pipeline_log, record_lifecycle, write_data_audit_log
except ImportError:
    import path_setup
    path_setup.ensure_scripts_on_path(__file__)
    from db_utils import get_db_stock_ids, write_pipeline_log, record_lifecycle, write_data_audit_log

logger = logging.getLogger(__name__)

# 表格與腳本映射矩陣
STOCK_TABLE_MAP = {
    "Technical": "ingest_technical_data.py",
    "PriceAdj": "ingest_price_adj_data.py",
    "Chip": "ingest_chip_data.py",
    "AdvancedChip": "ingest_advanced_chip_data.py",
    "Fundamental": "ingest_fundamental_data.py",
    "CashFlows": "ingest_cash_flows_data.py",
    "Revenue": "ingest_month_revenue.py",
    "PER": "ingest_macro_fundamental_data.py",
    "News": "ingest_news_data.py",
    "Dividend": "ingest_event_risk_data.py",
    "Sponsor": "ingest_sponsor_chip_data.py",
    "BlockTrading": "ingest_block_trading.py"
}

MARKET_TABLE_MAP = {
    "ReturnIndex": "ingest_total_return_index.py"
}

def run_task(cmd):
    """執行單一子進程任務並返回代碼。"""
    return os.system(cmd)

def show_orchestration_dashboard(args, tasks_count, success_count, duration):
    """批量任務後的旗艦級儀表板回報。"""
    rate = (success_count / tasks_count * 100) if tasks_count > 0 else 0
    print("\n" + "█"*60)
    print("🚀 Quantum Finance: 大數據入庫編排報告 (v8.0)")
    print("█" * 60)
    print(f"📡 執行模式  : {'個股' if args.stock_id else '宇宙'} ({args.stock_id or args.universe})")
    print(f"📊 任務清單  : {args.table}")
    print(f"🔄 總任務數  : {tasks_count}")
    print(f"✅ 成功完成  : {success_count}")
    print(f"🚀 總體成功率: {rate:.1f}%")
    print(f"⏱️  執行總耗時: {duration/1000:.2f} s")
    print("-" * 60)
    print("📝 系統日誌提示:")
    print("   - [Orchestration] 已記錄至 pipeline_execution_log")
    print("   - [SubTasks] 各子任務已獨立寫入生命週期與審計日誌")
    print("█" * 60 + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock_id", type=str)
    parser.add_argument("--universe", type=str, choices=["core", "all"])
    parser.add_argument("--table", type=str, default="ALL")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    t_start = datetime.now()
    
    # 使用生命週期監測器封裝整個編排任務
    task_id = args.stock_id or f"Universe_{args.universe}"
    with record_lifecycle("parallel_ingestion", category="Orchestration", stock_id=task_id):
        # 標的選取
        if args.stock_id: target_ids = [args.stock_id]
        elif args.universe == "core": target_ids = get_db_stock_ids(active_only=True)
        elif args.universe == "all": target_ids = get_db_stock_ids(active_only=False)
        else:
            logger.error("❌ 錯誤：必須指定 --stock_id 或 --universe")
            return

        # 任務清單構建
        if args.table == "ALL":
            target_stock_tables = STOCK_TABLE_MAP.keys()
            target_market_tables = MARKET_TABLE_MAP.keys()
        else:
            target_stock_tables = [args.table] if args.table in STOCK_TABLE_MAP else []
            target_market_tables = [args.table] if args.table in MARKET_TABLE_MAP else []
        
        tasks = []
        # 1. 產生個股級任務
        for sid in target_ids:
            for tbl in target_stock_tables:
                script = STOCK_TABLE_MAP[tbl]
                cmd = f"{sys.executable} {_THIS_DIR}/{script} --stock_id {sid}"
                if args.force: cmd += " --force"
                tasks.append(cmd)
        # 2. 產生市場級任務
        for tbl in target_market_tables:
            script = MARKET_TABLE_MAP[tbl]
            cmd = f"{sys.executable} {_THIS_DIR}/{script}"
            if args.force: cmd += " --force"
            tasks.append(cmd)

        if not tasks:
            logger.warning("⚠️ 沒有符合條件的任務可執行。")
            return

        logger.info(f"🚀 啟動並行入庫編排: 總任務數={len(tasks)}, 平行度={args.workers}")
        
        success_count = 0
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_task, cmd): cmd for cmd in tasks}
            for future in as_completed(futures):
                if future.result() == 0:
                    success_count += 1
                else:
                    logger.error(f"❌ 任務失敗: {futures[future]}")

        duration = int((datetime.now() - t_start).total_seconds() * 1000)
        
        # 寫入整體編排審計
        write_data_audit_log("ingestion_orchestration", task_id, t_start.strftime("%Y-%m-%d"), "BATCH", success_count)
        
        # 顯示儀表板
        show_orchestration_dashboard(args, len(tasks), success_count, duration)

if __name__ == "__main__":
    main()
