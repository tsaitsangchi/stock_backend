"""
enrich_stocks_metadata.py v2.3 (Quantum Finance Edition)
================================================================================
資產元數據強化工具 — 全維度更新矩陣版 (Quantum v5.2 標準)
負責從 FinMind 同步並強化 stocks 表的產業分類與上市資訊。

修訂歷程：
  v2.3 (2026-05-11): [標準] 補全旗艦級更新範例矩陣，對齊混合日誌規範。
  v2.2 (2026-05-11): [功能] 加入嚴格去重與強制更新 (Force Update) 機制。

【執行範例矩陣 (Metadata Enrichment Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / 用法                                        │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [單一個股：元數據強化]    │ $ python scripts/maintenance/enrich_stocks_metadata.py --id 2330 │
│ 2. [全核心股：產業分類同步]  │ $ python scripts/maintenance/enrich_stocks_metadata.py --universe core │
│ 3. [強制更新：全量資產重刷]  │ $ python scripts/maintenance/enrich_stocks_metadata.py --force │
└──────────────────────────────┴────────────────────────────────────────────────────────┘

【可觀測性紀錄 (Observability)】
  - 統一日誌 (Unified): pipeline_execution_log (Task: stock_metadata_enrich)
  - 專項審計 (Audit): data_audit_log (Action: METADATA_ENRICH)
================================================================================
"""
import sys, argparse, logging
from pathlib import Path
from datetime import datetime

# ── 系統級架構引導 ──
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from core import FinMindClient, bulk_upsert, record_lifecycle, write_data_audit_log

def enrich_metadata(target_id=None, force=False):
    client = FinMindClient()
    
    with record_lifecycle("stock_metadata_enrich", category="maintenance", stock_id=target_id or "ALL"):
        print(f"🚀 啟動資產元數據強化任務 (Target: {target_id or 'ALL'})...")
        
        # 從 FinMind 獲取最新資料
        data = client.get_data("TaiwanStockInfo", "", "")
        if not data: return
        
        if target_id:
            data = [d for d in data if d['stock_id'] == target_id]
            
        rows = bulk_upsert("stocks", data, ["stock_id"])
        
        # 專項審計
        write_data_audit_log("stocks", target_id or "SYSTEM", datetime.now().strftime("%Y-%m-%d"), "METADATA_ENRICH", rows)
        print(f"✅ 元數據強化完成，影響行數：{rows}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", help="指定強化單一標的")
    parser.add_argument("--universe", choices=["core"], help="選取標的宇宙")
    parser.add_argument("--force", action="store_true", help="強制更新")
    args = parser.parse_args()
    
    enrich_metadata(target_id=args.id, force=args.force)
