"""
check_data_integrity.py v3.4 (Quantum Finance Edition)
================================================================================
數據品質稽核報告 — 全維度稽核矩陣版 (Quantum v5.2 標準)
負責稽核與 API 1:1 對齊之鏡像表數據品質，整合混合日誌。

修訂歷程：
  v3.4 (2026-05-11): [標準] 移除硬編碼，引用 data_schema 鏡像契約，補全全維度稽核矩陣。
  v3.3 (2026-05-11): [結構] 修正 table 引用為 TaiwanStockPrice，對齊 v5.2 標準。

【執行範例矩陣 (Integrity Comprehensive Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / 用法                                        │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [單一個股：單一表稽核]    │ $ python scripts/maintenance/check_data_integrity.py --id 2330 --dataset TaiwanStockPrice │
│ 2. [單一個股：所有表稽核]    │ $ python scripts/maintenance/check_data_integrity.py --id 2330 --all_datasets │
│ 3. [所有核心股：單一表稽核]  │ $ python scripts/maintenance/check_data_integrity.py --universe core --dataset TaiwanStockPrice │
│ 4. [所有核心股：所有表稽核]  │ $ python scripts/maintenance/check_data_integrity.py --universe core --all_datasets │
│ 5. [強制更新：全量稽核]      │ $ python scripts/maintenance/check_data_integrity.py --universe core --force │
└──────────────────────────────┴────────────────────────────────────────────────────────┘

【可觀測性紀錄 (Observability)】
  - 統一日誌 (Unified): pipeline_execution_log (Task: audit_{dataset}_{stock_id})
  - 專項審計 (Audit): data_audit_log (Action: QUALITY_CHECK)
================================================================================
"""
import sys, argparse, logging
from pathlib import Path
from datetime import datetime

# ── 系統級架構引導 ──
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from core import db_transaction, get_db_stock_ids, record_lifecycle, write_data_audit_log
from core.data_schema import DATASET_SCHEMA_MAP

def run_audit(target_id: str, dataset: str, universe: str = None, force: bool = False):
    """執行數據品質稽核"""
    print("\n" + "🎯" * 40)
    print(f"🚀 Quantum Finance: 數據品質稽核報告 (v3.4)")
    print("🎯" * 40)

    # 決定標的
    if target_id:
        stock_ids = [target_id]
    elif universe == "core":
        stock_ids = get_db_stock_ids(core_only=True)
    else:
        print("❌ 錯誤：請指定 --id 或 --universe core")
        return

    # 決定數據集
    dataset_list = list(DATASET_SCHEMA_MAP.keys()) if dataset == "ALL" else [dataset]

    for ds in dataset_list:
        if ds not in DATASET_SCHEMA_MAP:
            print(f"⚠️ 警告：跳過未定義數據集 {ds}")
            continue
            
        config = DATASET_SCHEMA_MAP[ds]
        table_name = config["table"]
        
        print(f"\n📡 正在稽核 {len(stock_ids)} 檔標的之 [{table_name}] 數據品質...")
        
        for sid in stock_ids:
            with record_lifecycle(f"audit_{ds}", category="maintenance", stock_id=sid):
                with db_transaction() as cur:
                    # 注意：這裡使用引號保護大小寫敏感表名
                    cur.execute(f'SELECT COUNT(*) as cnt FROM "{table_name}" WHERE "stock_id" = %s', (sid,))
                    res = cur.fetchone()
                    rows = res['cnt'] if res else 0
                    
                    status = "✅ OK" if rows > 0 else "❌ MISSING"
                    print(f"  📍 {sid:<6} | Rows: {rows:>8} | 狀態: {status}")
                    
                    # 🔴 專項稽核日誌
                    write_data_audit_log(table_name, sid, datetime.now().strftime("%Y-%m-%d"), "QUALITY_CHECK", rows)

    print("\n" + "🎯" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", help="指定稽核單一標的 (如 2330)")
    parser.add_argument("--universe", choices=["core"], help="選取標的宇宙")
    parser.add_argument("--dataset", default="TaiwanStockPrice", help="指定稽核數據集")
    parser.add_argument("--all_datasets", action="store_true", help="稽核所有已定義數據集")
    parser.add_argument("--force", action="store_true", help="強制重新稽核並記錄")
    
    args = parser.parse_args()
    
    run_audit(
        target_id=args.id, 
        dataset="ALL" if args.all_datasets else args.dataset, 
        universe=args.universe, 
        force=args.force
    )
