
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.finmind_client import finmind_get
import logging

logging.basicConfig(level=logging.INFO)

def test_access():
    print("=== FinMind 權限深度測試 ===")
    
    # 1. 測試八大行庫 (個股模式)
    print("\n[1] 測試：八大行庫 (1101 個股模式)")
    res1 = finmind_get("TaiwanStockGovernmentBankBuySell", {"data_id": "1101", "start_date": "2024-04-01", "end_date": "2024-04-02"})
    print(f"    結果：成功獲取 {len(res1)} 筆" if res1 else "    結果：失敗 (403 or empty)")

    # 2. 測試八大行庫 (全市場模式)
    print("\n[2] 測試：八大行庫 (全市場模式)")
    res2 = finmind_get("TaiwanStockGovernmentBankBuySell", {"start_date": "2024-04-01", "end_date": "2024-04-01"})
    print(f"    結果：成功獲取 {len(res2)} 筆" if res2 else "    結果：失敗 (403 or empty)")

    # 3. 測試還原股價 (個股模式)
    print("\n[3] 測試：還原股價 (1101)")
    res3 = finmind_get("TaiwanStockPriceAdj", {"data_id": "1101", "start_date": "2024-04-01", "end_date": "2024-04-01"})
    print(f"    結果：成功獲取 {len(res3)} 筆" if res3 else "    結果：失敗 (403 or empty)")

if __name__ == "__main__":
    test_access()
