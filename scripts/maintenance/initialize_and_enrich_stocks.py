"""
initialize_and_enrich_stocks.py v6.4 (Quantum Finance Edition)
================================================================================
資產初始化與豐富化引擎 — 系統根基工具 (Quantum v5.2 標準)
負責同步全市場股票清單 (Stock List)、豐富化元數據並標記核心追蹤標的 (Core Universe)。

修訂歷程：
  v6.4 (2026-05-11): [修復] 實作 stock_id 主動去重 (Deduplication)，解決 PostgreSQL 同批次重複更新衝突。
  v6.3 (2026-05-11): [修復] 實作欄位白名單過濾，排除 API 多餘欄位 (如 date)。
  v6.2 (2026-05-11): [標準化] 導入 Quantum 標準規範、資產儀表板與 Hybrid Logging 對接。

【執行範例矩陣 (Comprehensive Usage Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令                                               │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [全市場資產初始化]        │ $ python scripts/maintenance/initialize_and_enrich_stocks.py │
│ 2. [標記核心 128 標的]       │ $ python scripts/maintenance/initialize_and_enrich_stocks.py --mark_core │
│ 3. [強制更新特定個股]        │ $ python scripts/maintenance/initialize_and_enrich_stocks.py --stock_id 2330 │
│ 4. [查看資產異動日誌]        │ SELECT * FROM data_audit_log WHERE table_name = 'stocks'; │
└──────────────────────────────┴────────────────────────────────────────────────────────┘

【自癒與防錯說明】
  - 自動過濾 API 冗餘欄位 (如 date, industry_category_en 等)。
  - 自動對 stock_id 進行去重，防止資料庫 Upsert 衝突。
  - 自動寫入 pipeline_execution_log 與 data_audit_log。
================================================================================
"""
import sys, logging, time, argparse
from datetime import datetime
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
    from core.db_utils import db_transaction, bulk_upsert, record_lifecycle, write_data_audit_log
    from core.finmind_client import FinMindClient
except ImportError as e:
    print(f"[FATAL] 無法匯入核心配置: {e}", file=sys.stderr)
    sys.exit(1)

logger = logging.getLogger(__name__)

# 核心標的名單 (根據實際需求調整)
CORE_UNIVERSE = ["2330", "2317", "2454", "2308", "2303", "2881", "2882", "2357"]

def show_asset_dashboard(stats: dict):
    """執行後的資產管理儀表板。"""
    print("\n" + "🏛️"*35)
    print("🚀 Quantum Finance: 資產初始化報告 (v6.4)")
    print("🏛️"*35)
    print(f"✅ 執行狀態  : 完成 (資料已自動去重)")
    print(f"📈 總資產數  : {stats.get('total', 0)} 檔")
    print(f"💎 核心標的  : {stats.get('core', 0)} 檔")
    print(f"📥 成功入庫  : {stats.get('upserted', 0)} 筆")
    print("-" * 70)
    print("📝 日誌同步: pipeline_execution_log & data_audit_log (stocks)")
    print("🏛️"*35 + "\n")

def initialize_stocks(stock_id: str = None, mark_core: bool = False):
    """執行股票資產初始化。"""
    ALLOWED_COLUMNS = ["stock_id", "stock_name", "industry_category", "type", "is_active", "is_core", "updated_at"]
    
    with record_lifecycle("asset_initialization", category="maintenance", stock_id=stock_id or "ALL"):
        try:
            api = FinMindClient()
            raw_data = api.get_data("TaiwanStockInfo", stock_id or "", "")
            if not raw_data:
                logger.error("❌ 無法從 API 獲取股票資訊")
                return False
            
            # 使用字典進行去重 (Deduplication)
            dedup_map = {}
            for item in raw_data:
                sid = item.get('stock_id')
                if not sid: continue
                
                # 1. 豐富化與標記邏輯
                item['is_active'] = True if item.get('status') == 'active' else False
                item['is_core'] = True if (mark_core and sid in CORE_UNIVERSE) else False
                item['updated_at'] = datetime.now()
                
                # 2. 欄位白名單過濾
                filtered_item = {k: v for k, v in item.items() if k in ALLOWED_COLUMNS}
                
                # 存入字典，後面的會覆蓋前面的，達到去重效果
                dedup_map[sid] = filtered_item

            clean_data = list(dedup_map.values())
            logger.info(f"🔍 [Audit] 獲取 {len(raw_data)} 筆原始數據，去重後保留 {len(clean_data)} 筆。")

            # 3. 執行批量入庫
            rows = bulk_upsert("stocks", clean_data, unique_cols=["stock_id"])
            
            # 獲取統計資訊
            stats = {"total": 0, "core": 0, "upserted": rows}
            with db_transaction() as cur:
                cur.execute("SELECT count(*) as cnt FROM stocks;")
                stats['total'] = cur.fetchone()['cnt']
                cur.execute("SELECT count(*) as cnt FROM stocks WHERE is_core = TRUE;")
                stats['core'] = cur.fetchone()['cnt']
            
            write_data_audit_log("stocks", stock_id or "SYSTEM", datetime.now().strftime("%Y-%m-%d"), "INITIALIZE", rows)
            
            show_asset_dashboard(stats)
            return True
        except Exception as e:
            logger.error(f"❌ 資產初始化失敗: {e}")
            raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock_id", type=str)
    parser.add_argument("--mark_core", action="store_true")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    initialize_stocks(stock_id=args.stock_id, mark_core=args.mark_core)
