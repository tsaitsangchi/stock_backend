"""
check_finmind_datalist.py v4.0 (Diagnostics & MLOps Edition)
FinMind 資料可用性與連線診斷工具
================================================================================
v4.0 重大升級：
  · 完美對接 finmind_client v4.0：改用 FinMindClient 類別 (支援 SQLite 快取與斷路器)。
  · 完美對接 db_utils v4.3：改用 db_session 上下文管理器進行安全的 DB 查詢。
  · 修正參數錯誤：嚴格遵守 get_data(dataset, data_id, start_date, end_date) 簽名，
    若為市場全量資料，data_id 會自動代入空字串 "" 而非報錯。

執行範例：
    # 檢查特定資料集（市場層級，無 stock_id）
    python scripts/fetchers/check_finmind_datalist.py --dataset TaiwanOptionOpenInterestLargeTraders --start 2024-01-01
    
    # 檢查特定個股的特定資料集（個股層級）
    python scripts/fetchers/check_finmind_datalist.py --dataset TaiwanStockCashFlowsStatement --stock-id 2330 --start 2023-01-01
    
    # 結合資料庫檢查 (比對 API 回傳筆數與 DB 本地筆數)
    python scripts/fetchers/check_finmind_datalist.py --dataset TaiwanStockInstitutionalInvestorsBuySell --stock-id 2317 --check-db
"""

import sys
import argparse
import logging
import time
from pathlib import Path
from datetime import date

# ── 系統路徑修復 (對接 path_setup v3.0) ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for sub in ("", "core"):
    p = (_SCRIPTS_DIR / sub) if sub else _SCRIPTS_DIR
    sp = str(p)
    if p.exists() and sp not in sys.path:
        sys.path.insert(0, sp)

try:
    from core.db_utils import db_session
    from core.finmind_client import FinMindClient, get_request_stats
except ImportError as e:
    print(f"無法匯入核心模組: {e}", file=sys.stderr)
    sys.exit(1)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def check_dataset(dataset: str, stock_id: str, start: str, end: str, delay: float):
    """執行 API 拉取測試"""
    target_name = stock_id if stock_id else "市場全量"
    logger.info(f"🔍 正在檢查資料集: {dataset} | 目標: {target_name} | 區間: {start} ~ {end}")

    # 使用 v4.0 的單例客戶端
    client = FinMindClient()
    t0 = time.time()
    
    try:
        # 【核心修復】即使沒有 stock_id，也必須傳遞空字串以符合新版函數簽名
        data_id_param = stock_id if stock_id else ""

        data = client.get_data(
            dataset=dataset,
            data_id=data_id_param,
            start_date=start,
            end_date=end
        )
        elapsed = time.time() - t0

        if not data:
            logger.warning(f"⚠️ 結果：[成功但無資料] 回傳 0 筆 (耗時 {elapsed:.2f}s)")
        else:
            logger.info(f"✅ 結果：[成功] 取得 {len(data)} 筆資料 (耗時 {elapsed:.2f}s)")
            # 預覽第一筆資料結構，幫助開發者確認欄位
            logger.info(f"   💡 第一筆資料預覽: {data[0]}")

        # 保護 API 配額的延遲
        time.sleep(delay)
        return True, data

    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"❌ 結果：[失敗] {e} (耗時 {elapsed:.2f}s)")
        return False, None

def check_db_table(dataset: str, stock_id: str):
    """(選填) 自動映射並檢查本地資料庫的狀態"""
    # Dataset 到 Table 的映射表 (可依業務需求擴充)
    mapping = {
        "TaiwanOptionOpenInterestLargeTraders": "options_oi_large_holders",
        "TaiwanStockInstitutionalInvestorsBuySell": "institutional_investors_buy_sell",
        "TaiwanStockMarginPurchaseShortSale": "margin_purchase_short_sale",
        "TaiwanStockShareholding": "shareholding"
    }

    table_name = mapping.get(dataset)
    if not table_name:
        logger.info(f"   [DB 檢查] 找不到對應 '{dataset}' 的資料表映射，跳過 DB 檢查。")
        return

    try:
        # 使用 v4.3 的連線池上下文管理器，保證安全歸還連線
        with db_session() as conn:
            with conn.cursor() as cur:
                if stock_id:
                    cur.execute(f"SELECT COUNT(*) FROM {table_name} WHERE stock_id = %s", (stock_id,))
                else:
                    cur.execute(f"SELECT COUNT(*) FROM {table_name}")

                count = cur.fetchone()[0]
                logger.info(f"   [DB 檢查] 本地資料表 '{table_name}' 目前共有 {count} 筆記錄。")
    except Exception as e:
        logger.warning(f"   [DB 檢查失敗] 無法查詢表 {table_name}: {e}")

def main():
    p = argparse.ArgumentParser(description="FinMind API 資料診斷工具 (v4.0)")
    p.add_argument("--dataset", type=str, default="TaiwanOptionOpenInterestLargeTraders", help="FinMind 資料集名稱")
    p.add_argument("--stock-id", type=str, default="", help="個股代碼 (選填，若是市場級資料請留空)")
    p.add_argument("--start", type=str, default=None, help="起始日期 (YYYY-MM-DD)")
    p.add_argument("--end", type=str, default=date.today().strftime("%Y-%m-%d"), help="結束日期 (YYYY-MM-DD)")
    p.add_argument("--delay", type=float, default=1.0, help="請求延遲秒數")
    p.add_argument("--check-db", action="store_true", help="是否同時檢查本地資料庫對應表")
    args = p.parse_args()

    # 預設起始日期處理 (依據資料集特性給定合理的預設區間)
    if not args.start:
        if args.dataset == "TaiwanOptionOpenInterestLargeTraders":
            args.start = "2018-01-02"
        else:
            args.start = (date.today().replace(year=date.today().year - 1)).strftime("%Y-%m-%d")

    logger.info("============================================================")
    logger.info("🚀 啟動 FinMind API 診斷工具...")

    # 1. 檢查 API 端點
    success, data = check_dataset(args.dataset, args.stock_id, args.start, args.end, args.delay)

    # 2. 檢查 Database 同步狀況
    if args.check_db:
        check_db_table(args.dataset, args.stock_id)

    logger.info("診斷完成。")

    # 3. 印出全域請求統計 (受惠於 SQLite 快取，重複執行將會看到 Cache Hit)
    try:
        stats = get_request_stats()
        stats.summary()
    except Exception:
        pass

if __name__ == "__main__":
    main()