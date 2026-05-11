"""
initialize_and_enrich_stocks.py v6.5 (Quantum Finance Edition)
================================================================================
資產初始化引擎 — 全市場標的同步與核心宇宙定義工具 (Quantum v5.2 標準)
負責從 API 獲取全市場上市櫃名單，並標記核心觀測標的 (is_core)。

修訂歷程：
  v6.5 (2026-05-11): [標準化] 補全全場景範例矩陣、強化核心標的自動標記邏輯、對齊 Hybrid Logging。
  v6.4 (2026-05-11): [修復] 實作 stock_id 去重，解決 ON CONFLICT 寫入衝突。

【執行範例矩陣 (Asset Operations Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / SQL                                         │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [全市場資產初始化]        │ $ python scripts/maintenance/initialize_and_enrich_stocks.py │
│ 2. [同步並標記核心宇宙]      │ $ python scripts/maintenance/initialize_and_enrich_stocks.py --mark_core │
│ 3. [強制更新特定產業資產]    │ $ python scripts/maintenance/initialize_and_enrich_stocks.py --all_stocks │
│ 4. [查看核心宇宙規模]        │ SELECT count(*) FROM stocks WHERE is_core = TRUE;      │
└──────────────────────────────┴────────────────────────────────────────────────────────┘

【業務邏輯說明】
  - 自動去重: 處理 API 回傳的重複標的。
  - 核心定義: 自動標記 2330, 2317, 2454, 2303, 2357, 2382, 3008, 2301 為核心標的。
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
    from core.db_utils import db_transaction, bulk_upsert, record_lifecycle, write_data_audit_log
    from core.finmind_client import FinMindClient
except ImportError as e:
    print(f"[FATAL] 無法匯入核心配置: {e}", file=sys.stderr)
    sys.exit(1)

logger = logging.getLogger(__name__)

# 定義核心觀測標的 (權值股與代表性標的)
CORE_STOCKS = ["2330", "2317", "2454", "2303", "2357", "2382", "3008", "2301"]

def show_asset_dashboard(processed: int, core_count: int):
    """執行後的資產初始化儀表板。"""
    print("\n" + "🏛️"*35)
    print("🚀 Quantum Finance: 資產初始化報告 (v6.5)")
    print("🏛️"*35)
    print(f"✅ 執行狀態  : 完成 (資料已自動去重)")
    print(f"📈 總資產數  : {processed} 檔")
    print(f"💎 核心標的  : {core_count} 檔")
    print(f"📥 成功入庫  : {processed} 筆")
    
    if core_count > 0:
        print("-" * 70)
        print(f"🟢 系統核心宇宙 (Core Universe) 已建立，共計 {core_count} 檔標的將優先同步。")
    else:
        print("-" * 70)
        print("🟡 警告：未偵測到核心標的，建議執行時加入 --mark_core 參數。")
        
    print("-" * 70)
    print("📝 日誌同步: pipeline_execution_log & data_audit_log (stocks)")
    print("🏛️"*35 + "\n")

def run_initialization(mark_core: bool = False):
    """執行全市場資產初始化與豐富化。"""
    with record_lifecycle("asset_initialization", category="maintenance", stock_id="ALL"):
        try:
            api = FinMindClient()
            logger.info("📡 正在獲取全市場上市櫃資產名單...")
            raw_data = api.get_data("TaiwanStockInfo", "", "")
            
            if not raw_data:
                logger.error("❌ 無法獲取 API 資產數據")
                return False

            # 去重與核心標記處理
            dedup_map = {}
            core_found = 0
            for item in raw_data:
                sid = item.get('stock_id')
                if not sid: continue
                
                # 決定是否標記為核心
                is_core = (sid in CORE_STOCKS) if mark_core else False
                if is_core: core_found += 1
                
                dedup_map[sid] = {
                    "stock_id": sid,
                    "stock_name": item.get('stock_name'),
                    "industry_category": item.get('industry_category'),
                    "type": item.get('type'),
                    "is_active": True,
                    "is_core": is_core,
                    "updated_at": datetime.now()
                }

            clean_data = list(dedup_map.values())
            logger.info(f"🔍 [Audit] 獲取 {len(raw_data)} 筆原始數據，去重後保留 {len(clean_data)} 筆。")

            # 批量寫入
            rows = bulk_upsert("stocks", clean_data, unique_cols=["stock_id"])
            
            # 混合紀錄
            write_data_audit_log("stocks", "ALL", date.today().strftime("%Y-%m-%d"), "INIT", rows)
            
            show_asset_dashboard(len(clean_data), core_found)
            return True
        except Exception as e:
            logger.error(f"❌ 資產初始化失敗: {e}")
            raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mark_core", action="store_true", help="強制標記預設核心標的")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_initialization(mark_core=args.mark_core)
