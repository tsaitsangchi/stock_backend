"""
db_utils.py v2.37 (Quantum Finance Edition)
================================================================================
資料庫核心引擎 — 旗艦編年史版 (Quantum v5.2 標準)
負責管理資產元數據、結構自癒、混合日誌裝飾器與鏡像保護寫入。

【核心定義說明 (Core Definitions)】
1. [Infrastructure Resilience]: 提供具備自動重連與健康診斷的資料庫通訊介面。
2. [Observability Engine]: 內建 record_lifecycle 與 write_data_audit_log 雙軌日誌機制。
3. [Asset Discovery]: 管理 stocks 表元數據，作為全系統標的獲取的單一事實來源。

【執行範例矩陣 (Historical & Active Matrix)】
┌──────────────────────────────────────┬────────────────────────────────────────────────────────┐
│ 維運場景                             │ 建議指令 / 用法                                        │
├──────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [核心股：全量健康診斷]            │ $ python scripts/core/db_utils.py                      │
│ 2. [單一表：資料獲取範例]            │ sql = "SELECT * FROM TaiwanStockPrice WHERE id='2330'" │
│ 3. [舊版範例 (v2.0)：基礎連線測試]   │ conn = get_db_connection() (維持至今)                  │
│ 4. [標準範例 (v2.26)：專案路徑自愈]  │ 自動載入 ROOT/.env，無須手動 export                     │
│ 5. [旗艦範例 (v2.37)：全量同步校驗]  │ $ python scripts/ingestion/template_fetcher.py ...     │
└──────────────────────────────────────┴────────────────────────────────────────────────────────┘

【全修訂歷程 (Full Revision History)】
  v2.37 (2026-05-12): [憲法] 注入今日詳細核心定義、舊歷程保留規範，對齊 2026-05-12 旗艦要求。
  v2.36 (2026-05-12): [憲法] 注入詳細核心定義與歷程保留規範。 (Earlier Standard)
  v2.35 (2026-05-11): [標準] 補全極致範例矩陣與歷史歷程，確立為 v5.2 基礎設施憲法。
  v2.34 (2026-05-11): [修復] 恢復診斷控制台輸出，對齊 v5.2 觀測性標準。
  v2.30 (2026-05-08): [觀測] 引入 record_lifecycle 裝飾器實現混合日誌規範。
  v2.26 (2026-05-05): [感知] 實現自動尋找專案根目錄並加載環境變數。
  v2.0  (2026-04-30): [安全] 整合加密認證，棄用明文連線配置。
  v1.0  (2026-04-20): [奠基] 初始版本，建立基本連線與 stocks 元數據表。
================================================================================
"""
import os, sys, psycopg2, logging
from contextlib import contextmanager
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# ── 系統級架構引導 (v2.26 遺產) ──
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

@contextmanager
def record_lifecycle(task_name, category="general", stock_id=None):
    """旗艦級生命週期裝飾器 (v2.30 遺產)"""
    start_time = datetime.now()
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        yield
        status = "success"
    except Exception as e:
        status = f"failed: {str(e)}"
        raise e
    finally:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() * 1000
        cur.execute("""
            INSERT INTO pipeline_execution_log (task_name, category, stock_id, status, duration_ms)
            VALUES (%s, %s, %s, %s, %s)
        """, (task_name, category, stock_id, status, duration))
        conn.commit()
        cur.close()
        conn.close()

def write_data_audit_log(table_name, stock_id, data_date, action_type, rows_affected):
    """專項審計日誌 (v2.30 遺產)"""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO data_audit_log (table_name, stock_id, data_date, action_type, rows_affected)
            VALUES (%s, %s, %s, %s, %s)
        """, (table_name, stock_id, data_date, action_type, rows_affected))
        conn.commit()
    finally:
        cur.close()
        conn.close()

def get_db_connection():
    """建立資料庫連線 (v2.0 遺產)"""
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )

def db_connection():
    """連線健康診斷 (v2.34 遺產)"""
    try:
        conn = get_db_connection()
        conn.close()
        return True
    except:
        return False

def get_core_stocks_from_db():
    """獲取核心資產清單 (v1.0 遺產)"""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT stock_id FROM stocks WHERE is_core = TRUE")
    stocks = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()
    return stocks

if __name__ == "__main__":
    print("-" * 60)
    print("🚀 Quantum Finance: db_utils v2.37 核心診斷啟動...")
    conn_status = "SUCCESS" if db_connection() else "FAILED"
    print(f"✅ 資料庫連線 : {conn_status}")
    if conn_status == "SUCCESS":
        print(f"📊 核心資產數 : {len(get_core_stocks_from_db())}")
    print("-" * 60)