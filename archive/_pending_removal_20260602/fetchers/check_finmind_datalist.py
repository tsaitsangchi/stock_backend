"""
check_finmind_datalist.py — FinMind 資料可用性與連線診斷工具 (Quantum Finance v5.1 Edition)
================================================================================
v3.0 改進：
  ★ 整合核心模組：使用 `core.finmind_client` 進行 API 請求，具備自動重試與統計功能。
  ★ 彈性查詢：支援透過 CLI 參數指定資料集 (dataset)、個股代碼 (stock_id) 及日期區間。
  ★ 資料庫連動：整合 `core.db_utils` 檢查本地資料表狀態，對比 API 回傳資料筆數。
  ★ 效能監控：程式結束時印出統一的 `RequestStats` 報表。

執行範例：
    # 檢查特定資料集（市場層級）
    python scripts/fetchers/check_finmind_datalist.py --dataset TaiwanStockTotalMarginPurchaseShortSale
    
    # 檢查特定個股的特定資料集（個股層級）
    python scripts/fetchers/check_finmind_datalist.py --dataset TaiwanStockCashFlowsStatement --stock-id 2330 --start 2020-01-01
    
    # 檢查高流量資料集（如大額期權未平倉）
    python scripts/fetchers/check_finmind_datalist.py --dataset TaiwanOptionOpenInterestLargeTraders --start 2018-01-02
"""

import sys
import argparse
import logging
import time
from pathlib import Path
from datetime import date, datetime

# ── 1. 啟動引導 (Bootstrap)：確保能找到 scripts/ ──
_scripts_dir = Path(__file__).resolve().parent.parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from core.path_setup import ensure_scripts_on_path
ensure_scripts_on_path(__file__)

from core.finmind_client import finmind_get, get_request_stats
from core.db_utils import get_db_conn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def check_db_table_exists(conn, table_name: str) -> bool:
    """檢查本地資料庫是否存在該資料表。"""
    sql = "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = %s);"
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (table_name,))
            return cur.fetchone()[0]
    except Exception as e:
        logger.warning(f"檢查資料表 {table_name} 失敗：{e}")
        return False

def check_dataset(dataset: str, stock_id: str | None, start_date: str, end_date: str, delay: float):
    """執行資料集可用性檢查。"""
    params = {
        "start_date": start_date,
        "end_date": end_date,
    }
    if stock_id:
        params["data_id"] = stock_id

    logger.info(f"正在檢查資料集: {dataset} | 目標: {stock_id or '市場全量'} | 區間: {start_date} ~ {end_date}")
    
    t0 = time.time()
    try:
        # 使用核心 finmind_get，內含自動重試與統計
        data = finmind_get(
            dataset=dataset,
            params=params,
            delay=delay,
            raise_on_error=True
        )
        elapsed = time.time() - t0
        
        if not data:
            logger.warning(f"結果：[無資料] (耗時 {elapsed:.2f}s)")
            return
        
        logger.info(f"結果：[成功] 取得 {len(data)} 筆資料 (耗時 {elapsed:.2f}s)")
        
        # 顯示前 2 筆範例資料
        if len(data) > 0:
            logger.info("範例數據 (首筆)：")
            print(f"  {data[0]}")
        if len(data) > 1:
            logger.info("範例數據 (末筆)：")
            print(f"  {data[-1]}")
            
    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"結果：[失敗] {e} (耗時 {elapsed:.2f}s)")

def main():
    p = argparse.ArgumentParser(description="FinMind API 資料診斷工具")
    p.add_argument("--dataset", type=str, default="TaiwanOptionOpenInterestLargeTraders", help="FinMind 資料集名稱")
    p.add_argument("--stock-id", type=str, default=None, help="個股代碼 (選填)")
    p.add_argument("--start", type=str, default=None, help="起始日期 (YYYY-MM-DD)")
    p.add_argument("--end", type=str, default=date.today().strftime("%Y-%m-%d"), help="結束日期 (YYYY-MM-DD)")
    p.add_argument("--delay", type=float, default=1.0, help="請求延遲秒數")
    p.add_argument("--check-db", action="store_true", help="是否同時檢查本地資料庫對應表")
    args = p.parse_args()

    # 預設起始日期處理
    if not args.start:
        if args.dataset == "TaiwanOptionOpenInterestLargeTraders":
            args.start = "2018-01-02"
        else:
            args.start = (date.today().replace(year=date.today().year - 1)).strftime("%Y-%m-%d")

    conn = get_db_conn()
    try:
        # 1. 執行 API 檢查
        check_dataset(args.dataset, args.stock_id, args.start, args.end, args.delay)
        
        # 2. 執行 DB 檢查 (選填)
        if args.check_db:
            # 嘗試猜測資料表名稱 (通常為 dataset 轉小寫)
            # 這裡僅為簡單邏輯，實際可能需要對應表
            table_guess = args.dataset.lower()
            exists = check_db_table_exists(conn, table_guess)
            logger.info(f"本地資料庫檢查：資料表 [{table_guess}] {'✅ 已存在' if exists else '❌ 不存在'}")
            
    finally:
        conn.close()
        logger.info("診斷完成。")
        # 印出 FinMind 統計報表
        get_request_stats().summary()

if __name__ == "__main__":
    main()
