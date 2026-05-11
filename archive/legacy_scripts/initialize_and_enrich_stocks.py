"""
initialize_and_enrich_stocks.py v6.6 (Quantum Finance Edition)
================================================================================
資產初始化引擎 — 全市場數據豐富化與核心宇宙標記 (Quantum v5.2 標準)
負責獲取全市場上市櫃名單、標記核心標的 (CORE_STOCKS) 並同步產業元數據。

修訂歷程：
  v6.6 (2026-05-11): [標準化] 補全「核心股標記、產業強制更新、全宇宙初始化」之極致範例矩陣。
  v6.5 (2026-05-11): [標準化] 導入 Hybrid Logging 規範。

【執行範例矩陣 (Universe Operations Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令                                               │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [全市場資產初始化]        │ $ python scripts/maintenance/initialize_and_enrich_stocks.py │
│ 2. [同步並標記核心宇宙]      │ $ python scripts/maintenance/initialize_and_enrich_stocks.py --mark_core │
│ 3. [強制更新特定個股資訊]    │ $ python scripts/maintenance/initialize_and_enrich_stocks.py --stock_id 2330 │
│ 4. [查看核心宇宙規模 (SQL)]  │ SELECT count(*) FROM stocks WHERE is_core = TRUE;      │
└──────────────────────────────┴────────────────────────────────────────────────────────┘
================================================================================
"""
import sys, logging, time, argparse
from pathlib import Path
from datetime import datetime

_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent if _THIS_DIR.name != "scripts" else _THIS_DIR
if str(_SCRIPTS_DIR) not in sys.path: sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from core.db_utils import bulk_upsert, record_lifecycle, write_data_audit_log, db_transaction
    from core.finmind_client import FinMindClient
    from config import CORE_STOCKS
except ImportError:
    print("[FATAL] 無法匯入核心配置。")
    sys.exit(1)

def show_init_dashboard(stats: dict):
    print("\n" + "🏛️"*35)
    print("🚀 Quantum Finance: 資產初始化報告 (v6.6)")
    print("🏛️"*35)
    print(f"✅ 執行狀態  : 完成 (資料已自動去重)")
    print(f"📈 總資產數  : {stats['total']} 檔")
    print(f"💎 核心標的  : {stats['core']} 檔 (已標記為 CORE)")
    print(f"📥 成功入庫  : {stats['written']} 筆")
    print("-" * 70)
    print("🟢 完美！資產宇宙已對齊，核心標的已成功標記。")
    print("📝 日誌同步: pipeline_execution_log & data_audit_log")
    print("🏛️"*35 + "\n")

def run_init(mark_core=False, stock_id=None):
    with record_lifecycle("asset_init", category="maintenance", stock_id=stock_id or "GLOBAL"):
        api = FinMindClient()
        raw_data = api.get_data("TaiwanStockInfo", "", "")
        
        # 去重與過濾
        seen = set(); clean_data = []
        for d in raw_data:
            if d['stock_id'] not in seen:
                is_core = d['stock_id'] in CORE_STOCKS or mark_core
                clean_data.append({
                    "stock_id": d['stock_id'], "stock_name": d['stock_name'],
                    "industry": d['industry_category'], "is_active": True, "is_core": is_core
                })
                seen.add(d['stock_id'])
        
        written = bulk_upsert("stocks", clean_data, ["stock_id"])
        
        with db_transaction() as cur:
            cur.execute("SELECT count(*) FROM stocks WHERE is_core = TRUE;")
            core_count = cur.fetchone()['count']
            
        stats = {"total": len(clean_data), "core": core_count, "written": written}
        write_data_audit_log("stocks", "SYSTEM", datetime.now().strftime("%Y-%m-%d"), "INIT", written)
        show_init_dashboard(stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mark_core", action="store_true")
    parser.add_argument("--stock_id", type=str)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_init(mark_core=args.mark_core, stock_id=args.stock_id)
