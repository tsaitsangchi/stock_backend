"""
test_logging_v1.py
================================================================================
驗證 Quantum Finance 混合模式日誌系統是否正常運作。
1. 生命週期紀錄 (pipeline_execution_log)
2. 專屬分類紀錄 (data_audit_log)
================================================================================
"""
from core.db_utils import record_lifecycle, write_data_audit_log, check_db_health
from core.finmind_client import FinMindClient
import logging

def test_system():
    logging.basicConfig(level=logging.INFO)
    print("🚀 開始 Quantum 混合日誌整合測試...")
    
    client = FinMindClient()
    stock_id = "2330"
    dataset = "TaiwanStockPrice"
    
    # 模擬完整生命週期：從抓取到寫入
    with record_lifecycle(f"Test_Fetch_{dataset}", category="ingestion", stock_id=stock_id):
        print(f"📡 正在模擬抓取 {stock_id} 的 {dataset}...")
        # 實際抓取一小段數據以產生真實 IO
        data = client.get_data(dataset, stock_id, "2024-05-01")
        
        if data:
            print(f"✅ 抓取成功，共 {len(data)} 筆資料。")
            # 模擬寫入後的專屬審計紀錄
            print(f"📝 正在寫入專屬分類紀錄 (Data Audit Log)...")
            write_data_audit_log(dataset, stock_id, "2024-05-01", "2024-05-10", len(data))
        else:
            print("⚠️ 未抓取到數據，請檢查 API 配額或網路。")

if __name__ == "__main__":
    test_system()
    # 最後執行健康檢查，確認日誌總量是否增加
    check_db_health()
