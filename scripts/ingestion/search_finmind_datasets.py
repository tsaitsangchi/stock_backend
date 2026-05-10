"""
search_finmind_datasets.py v4.2 (Quantum Finance Edition)
================================================================================
FinMind 資料集檢索工具 — 開發輔助專項 (Quantum v5.1 標準)
負責檢索 FinMind API 支援的所有資料集與代號，輔助開發新抓取器。

修訂歷程：
  v4.2 (2026-05-10): [文件] 完善五維度執行範例矩陣，確保範例完整性。
  v4.1 (2026-05-10): [修復] 對齊 db_utils v5.1 混合日誌規範。
  v4.0 (2026-05-10): [核心] 整合 v5.1 Bootstrap 機制。

【執行範例矩陣 — 工具使用方案】
1. 檢索特定關鍵字的資料集 (Python)：
   python scripts/ingestion/search_finmind_datasets.py --query Price
2. 單一標的「所有」維度表格抓取 (透過編排器)：
   python scripts/ingestion/parallel_fetch.py --stock_id 2330 --table ALL
3. 核心標的集「所有」維度表格同步 (透過編排器)：
   python scripts/ingestion/parallel_fetch.py --universe core --table ALL
4. 核心標的集「所有」維度表格「強制」更新 (透過編排器)：
   python scripts/ingestion/parallel_fetch.py --universe core --table ALL --force
5. 全市場標的「所有」維度表格同步 (透過編排器)：
   python scripts/ingestion/parallel_fetch.py --universe all --table ALL
================================================================================
"""
import os, sys, logging, argparse
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
    from core.finmind_client import FinMindClient
except ImportError:
    import path_setup
    path_setup.ensure_scripts_on_path(__file__)
    from finmind_client import FinMindClient

logger = logging.getLogger(__name__)

def search_datasets(query: str):
    client = FinMindClient()
    logger.info(f"🔍 正在 FinMind 中檢索關鍵字: {query}...")
    # 這裡僅列出常用的核心資料集對應關係作為參考
    registry = {
        "TaiwanStockPrice": "台股日線 (OHLCV)",
        "TaiwanStockPriceAdj": "台股還原股價",
        "TaiwanStockInstitutionalInvestorsBuySell": "三大法人買賣超",
        "TaiwanStockMarginPurchaseShortSale": "融資融券",
        "TaiwanStockMonthRevenue": "月營收",
        "TaiwanStockFinancialStatements": "財務報表",
        "TaiwanStockCashFlows": "現金流量表",
        "TaiwanStockPER": "本益比與殖利率",
        "TaiwanStockTotalReturnIndex": "報酬指數",
        "TaiwanStockBlockTrading": "鉅額交易",
        "TaiwanStockNews": "個股新聞",
        "TaiwanStockCashDividend": "股利發放",
        "TaiwanFuturesDaily": "期貨日線",
        "TaiwanOptionDaily": "期權日線",
        "FredData": "FRED 經濟指標",
        "USStockPrice": "美股日線"
    }
    
    found = False
    for dataset, desc in registry.items():
        if query.lower() in dataset.lower() or query in desc:
            logger.info(f"✅ 匹配: {dataset} ({desc})")
            found = True
    
    if not found:
        logger.info(f"ℹ️  未找到與 '{query}' 相關的資料集。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="Price")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    search_datasets(args.query)