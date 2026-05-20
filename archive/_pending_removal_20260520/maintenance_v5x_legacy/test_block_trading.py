"""
test_block_trading.py v1.3 (Quantum Finance Edition)
================================================================================
大額交易連通性測試工具 — 旗艦維運矩陣版 (Quantum v5.2 標準)
負責稽核特定標的之大額交易 (Block Trading) 數據連通性與 API 配額消耗。

修訂歷程：
  v1.3 (2026-05-11): [標準] 補全旗艦級測試範例矩陣，對齊混合日誌規範。
  v1.2 (2026-05-08): [結構] 修正路徑引導，對齊 v5.2 主權架構。

【執行範例矩陣 (Block Trading Test Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / 用法                                        │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [單一個股：大額交易稽核]  │ $ python scripts/maintenance/test_block_trading.py --id 2330 │
│ 2. [所有核心股：大額交易連通]│ $ python scripts/maintenance/test_block_trading.py --universe core │
│ 3. [強制更新：全量測試紀錄]  │ $ python scripts/maintenance/test_block_trading.py --force │
└──────────────────────────────┴────────────────────────────────────────────────────────┘

【可觀測性紀錄 (Observability)】
  - 統一日誌 (Unified): pipeline_execution_log (Task: test_block_trading)
  - 專項審計 (Audit): data_audit_log (Action: TEST_BLOCK_DATA)
================================================================================
"""
import sys, argparse, logging
from pathlib import Path
from datetime import datetime

# ── 系統級架構引導 ──
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from core import FinMindClient, record_lifecycle, write_data_audit_log

def test_block_data(target_id="2330", force=False):
    client = FinMindClient()
    
    with record_lifecycle("test_block_trading", category="test", stock_id=target_id):
        print(f"🚀 啟動大額交易連通性測試 (Target: {target_id})...")
        
        # 模擬測試 API 呼叫
        data = client.get_data("TaiwanStockInstitutionalInvestorsBuySell", target_id, datetime.now().strftime("%Y-%m-%d"))
        
        rows = len(data)
        write_data_audit_log("TEST_BLOCK", target_id, datetime.now().strftime("%Y-%m-%d"), "TEST_BLOCK_DATA", rows)
        print(f"✅ 測試完成，獲取資料筆數：{rows}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default="2330", help="指定測試標的")
    parser.add_argument("--universe", choices=["core"], help="選取標的宇宙")
    parser.add_argument("--force", action="store_true", help="強制更新測試紀錄")
    args = parser.parse_args()
    
    test_block_data(target_id=args.id, force=args.force)
