"""
check_logs.py
================================================================================
檢查並驗證日誌紀錄。
新增：手動插入一筆測試資料以驗證寫入權限。
================================================================================
"""
from core.db_utils import db_transaction, write_pipeline_log
import json

def test_and_check():
    print("🧪 執行手動寫入測試...")
    write_pipeline_log("Manual_Test", "TEST_ID", "success", "diagnostic", 999, 1)
    
    print("📋 正在檢查生命週期日誌 (pipeline_execution_log)...")
    with db_transaction() as cur:
        cur.execute("SELECT * FROM pipeline_execution_log ORDER BY created_at DESC LIMIT 5;")
        logs = cur.fetchall()
        if logs:
            print(f"✅ 找到 {len(logs)} 筆紀錄：")
            print(json.dumps(logs, indent=2, default=str))
        else:
            print("❌ 依然找不到任何紀錄，請檢查資料庫實體。")
        
    print("\n📋 正在檢查專屬分類日誌 (data_audit_log)...")
    with db_transaction() as cur:
        cur.execute("SELECT * FROM data_audit_log ORDER BY created_at DESC LIMIT 3;")
        audits = cur.fetchall()
        print(json.dumps(audits, indent=2, default=str))

if __name__ == "__main__":
    test_and_check()
