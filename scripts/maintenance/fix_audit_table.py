"""
fix_audit_table.py (Quantum Maintenance Utility)
================================================================================
資料表結構修復工具 — 審計日誌專項
負責解決 data_audit_log 結構不一致的問題。
================================================================================
"""
import os, sys
from pathlib import Path

# ── 終極路徑自癒 Bootstrap ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from core.db_utils import get_db_connection

def fix_table():
    conn = get_db_connection()
    cur = conn.cursor()
    print("🛠️  正在檢查並修復 data_audit_log 資料表...")
    
    try:
        # 如果表存在，直接刪除重來（因為這是審計紀錄，初期重建最保險）
        cur.execute("DROP TABLE IF EXISTS data_audit_log CASCADE;")
        
        # 創建正確結構的表
        cur.execute("""
            CREATE TABLE data_audit_log (
                id SERIAL PRIMARY KEY,
                table_name VARCHAR(100) NOT NULL,
                stock_id VARCHAR(50) NOT NULL,
                start_date DATE,
                end_date DATE,
                rows_affected INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        print("✅ data_audit_log 資料表已成功修復並升級至 v5.1 標準！")
    except Exception as e:
        conn.rollback()
        print(f"❌ 修復失敗: {e}")
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    fix_table()
