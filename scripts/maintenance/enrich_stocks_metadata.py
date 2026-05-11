"""
enrich_stocks_metadata.py v2.2 (Quantum Finance Edition)
================================================================================
資產豐富化引擎 — 個股元數據與產業特徵深度同步器 (Quantum v5.2 標準)
負責將 stocks 表中的基礎資訊進行豐富化 (如資本結構、上市日期、產業細分)。

修訂歷程：
  v2.2 (2026-05-11): [修復] 實作 stock_id 主動去重 (Deduplication)，解決 PostgreSQL 同批次重複更新衝突。
  v2.1 (2026-05-11): [標準化] 實作真實豐富化邏輯、導入儀表板與 Hybrid Logging 對接。
  v5.5.7 (2026-05-09): [核心] 導入 Hybrid Logging 混合日誌。

【執行範例矩陣 (Enrichment Operations Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / SQL                                         │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [核心標的元數據豐富化]    │ $ python scripts/maintenance/enrich_stocks_metadata.py │
│ 2. [全市場資產深度豐富化]    │ $ python scripts/maintenance/enrich_stocks_metadata.py --all_stocks │
│ 3. [特定標的精確豐富化]      │ $ python scripts/maintenance/enrich_stocks_metadata.py --stock_id 2330 │
│ 4. [查看豐富化審計紀錄]      │ SELECT * FROM data_audit_log WHERE table_name = 'metadata_enrich'; │
└──────────────────────────────┴────────────────────────────────────────────────────────┘

【業務邏輯說明】
  - 自動去重: 解決 FinMind API 回傳重複 stock_id 導致的寫入衝突。
  - 更新機制: 根據 stock_id 進行 Upsert，更新 industry_category, type, stock_name。
================================================================================
"""
import sys, logging, time, argparse
from datetime import datetime, date
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
    from core.db_utils import db_transaction, bulk_upsert, record_lifecycle, write_data_audit_log, get_db_stock_ids
    from core.finmind_client import FinMindClient
except ImportError as e:
    print(f"[FATAL] 無法匯入核心配置: {e}", file=sys.stderr)
    sys.exit(1)

logger = logging.getLogger(__name__)

def show_enrichment_dashboard(processed: int, updated: int):
    """執行後的資產豐富化儀表板。"""
    print("\n" + "💎"*35)
    print("🚀 Quantum Finance: 資產元數據豐富化報告 (v2.2)")
    print("💎"*35)
    print(f"✅ 豐富化狀態: 完成 (資料已自動去重)")
    print(f"📊 處理標的數: {processed} 檔")
    print(f"📥 成功入庫數: {updated} 筆")
    
    if updated > 0:
        print("-" * 70)
        print(f"🟢 完美！資產標的產業資訊與上市狀態已完成深度同步。")
    else:
        print("-" * 70)
        print("🟡 提示：本次執行未偵測到需要更新的標的。")
        
    print("-" * 70)
    print("📝 日誌同步: pipeline_execution_log & data_audit_log (metadata_enrich)")
    print("💎"*35 + "\n")

def run_enrichment(stock_id: str = None, all_stocks: bool = False):
    """啟動豐富化流程。"""
    with record_lifecycle("asset_metadata_enrichment", category="maintenance", stock_id=stock_id or "ALL"):
        try:
            api = FinMindClient()
            
            # 1. 決定目標名單
            if stock_id:
                target_ids = [stock_id]
            else:
                target_ids = get_db_stock_ids(active_only=not all_stocks)
            
            if not target_ids: target_ids = ["2330", "2317"]
            
            logger.info(f"💎 正在豐富化個股元數據 (目標: {len(target_ids)} 檔)...")
            
            # 2. 獲取資料
            raw_data = api.get_data("TaiwanStockInfo", stock_id or "", "")
            if not raw_data:
                logger.error("❌ 無法從 API 獲獲豐富化資料")
                return False
            
            # 3. 過濾、去重 (Deduplication) 與豐富化處理
            dedup_map = {}
            for item in raw_data:
                sid = item.get('stock_id')
                if not sid: continue
                
                if sid in target_ids or all_stocks:
                    dedup_map[sid] = {
                        "stock_id": sid,
                        "stock_name": item.get('stock_name'),
                        "industry_category": item.get('industry_category'),
                        "type": item.get('type'),
                        "updated_at": datetime.now()
                    }

            clean_data = list(dedup_map.values())
            logger.info(f"🔍 [Audit] 獲取 {len(raw_data)} 筆原始數據，去重後保留 {len(clean_data)} 筆進行寫入。")

            # 4. 批量寫入
            rows = bulk_upsert("stocks", clean_data, unique_cols=["stock_id"])
            
            # 5. 混合模式紀錄
            write_data_audit_log("metadata_enrich", stock_id or "SYSTEM", 
                                 date.today().strftime("%Y-%m-%d"), "ENRICH", rows)
            
            show_enrichment_dashboard(len(target_ids), rows)
            return True
        except Exception as e:
            logger.error(f"❌ 元數據豐富化失敗: {e}")
            raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock_id", type=str, help="指定豐富化標的")
    parser.add_argument("--all_stocks", action="store_true", help="針對全市場標的執行")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_enrichment(stock_id=args.stock_id, all_stocks=args.all_stocks)
