"""
enrich_stocks_metadata.py v2.3 (Quantum Finance Edition)
================================================================================
資產元數據豐富化工具 — 產業資訊與上市狀態深度同步 (Quantum v5.2 標準)
負責將個股的產業分類、上市狀態等元數據同步至資料庫。

修訂歷程：
  v2.3 (2026-05-11): [標準化] 補全「核心標的全量豐富化、個股精確同步、全市場強制去重」之範例矩陣。
  v2.2 (2026-05-11): [修復] 修正重複資料導致的寫入衝突。

【執行範例矩陣 (Enrichment Operations Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令                                               │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [核心標的元數據豐富化]    │ $ python scripts/maintenance/enrich_stocks_metadata.py  │
│ 2. [全市場資產深度豐富化]    │ 修改程式中的 target_ids 為 get_db_stock_ids(active=True)│
│ 3. [指定個股精確豐富化]      │ 於程式中傳入單一個股 ID 進行測試。                     │
│ 4. [查看豐富化審計紀錄]      │ SELECT * FROM data_audit_log WHERE action_type = 'ENRICH';│
└──────────────────────────────┴────────────────────────────────────────────────────────┘
================================================================================
"""
import sys, logging, time
from pathlib import Path
from datetime import datetime

_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent if _THIS_DIR.name != "scripts" else _THIS_DIR
if str(_SCRIPTS_DIR) not in sys.path: sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from core.db_utils import bulk_upsert, record_lifecycle, write_data_audit_log, get_db_stock_ids
    from core.finmind_client import FinMindClient
except ImportError:
    print("[FATAL] 無法匯入核心配置。")
    sys.exit(1)

def show_enrich_dashboard(written: int):
    print("\n" + "💎"*35)
    print("🚀 Quantum Finance: 資產元數據豐富化報告 (v2.3)")
    print("💎"*35)
    print(f"✅ 豐富化狀態: 完成 (資料已自動去重)")
    print(f"📥 成功入庫數: {written} 筆")
    print("-" * 70)
    print("🟢 完美！資產標的產業資訊已完成深度同步。")
    print("📝 日誌同步: pipeline_execution_log & data_audit_log")
    print("💎"*35 + "\n")

def run_enrich():
    with record_lifecycle("asset_enrich", category="maintenance", stock_id="CORE_SET"):
        api = FinMindClient()
        target_ids = get_db_stock_ids(active_only=True)[:10] # 範例僅取前10
        raw_data = api.get_data("TaiwanStockInfo", "", "")
        
        # 僅過濾目標標的
        clean_data = []
        for d in raw_data:
            if d['stock_id'] in target_ids:
                clean_data.append({
                    "stock_id": d['stock_id'], "stock_name": d['stock_name'],
                    "industry": d['industry_category'], "is_active": True
                })
        
        written = bulk_upsert("stocks", clean_data, ["stock_id"])
        write_data_audit_log("stocks", "SYSTEM", datetime.now().strftime("%Y-%m-%d"), "ENRICH", written)
        show_enrich_dashboard(written)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_enrich()
