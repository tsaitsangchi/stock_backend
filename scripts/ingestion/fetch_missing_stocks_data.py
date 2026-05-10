"""
fetch_missing_stocks_data.py v4.3 (Quantum Finance Edition)
================================================================================
數據治理工具 — 缺失標的偵測 (Quantum v5.1 標準)
負責對比 stocks 定義表與各行情表，找出並回填尚未同步數據的標的。

修訂歷程：
  v4.3 (2026-05-10): [核心] 實作混合模式日誌，對齊 v5.1 治理紀錄規範。
  v4.2 (2026-05-10): [文件] 完善五維度執行範例矩陣，確保範例完整性。
  v4.1 (2026-05-10): [修復] 對齊 db_utils v5.1 混合日誌規範。

【執行範例矩陣 — 數據治理方案】
1. 偵測全市場缺失數據並回填 (Python)：
   python scripts/ingestion/fetch_missing_stocks_data.py
2. 單一標的「所有」維度表格抓取 (透過編排器)：
   python scripts/ingestion/parallel_fetch.py --stock_id 2330 --table ALL
3. 核心標的集「所有」維度表格同步 (透過編排器)：
   python scripts/ingestion/parallel_fetch.py --universe core --table ALL
4. 核心標的集「所有」維度表格「強制」更新 (透過編排器)：
   python scripts/ingestion/parallel_fetch.py --universe core --table ALL --force
5. 全市場標的「所有」維度表格同步 (透過編排器)：
   python scripts/ingestion/parallel_fetch.py --universe all --table ALL
================================================================================
"""
import os, sys, logging, time
from pathlib import Path
from datetime import datetime

# ── 終極路徑自癒 Bootstrap ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent if _THIS_DIR.name != "scripts" else _THIS_DIR
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_SCRIPTS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR.parent))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import get_db_connection, write_pipeline_log, write_data_audit_log
except ImportError:
    import path_setup
    path_setup.ensure_scripts_on_path(__file__)
    from db_utils import get_db_connection, write_pipeline_log, write_data_audit_log

logger = logging.getLogger(__name__)

def detect_and_fetch_missing():
    start_time = time.time()
    conn = get_db_connection()
    cur = conn.cursor()
    
    logger.info("🔍 正在偵測缺失數據標的...")
    
    try:
        # 找出 stocks 表中有但 stock_price 表中無的標的
        query = """
        SELECT s.stock_id 
        FROM stocks s
        LEFT JOIN stock_price p ON s.stock_id = p.stock_id
        WHERE p.stock_id IS NULL
        """
        cur.execute(query)
        missing_ids = [row[0] for row in cur.fetchall()]
        
        if missing_ids:
            logger.info(f"🚨 發現 {len(missing_ids)} 檔缺失數據標的，啟動回填...")
            for sid in missing_ids:
                cmd = f"{sys.executable} {_THIS_DIR}/parallel_fetch.py --stock_id {sid} --table ALL"
                os.system(cmd)
            
            duration = int((time.time() - start_time) * 1000)
            # 🔴 治理層混合日誌
            write_pipeline_log("Governance_MissingCheck", "MARKET", "SUCCESS", "Governance", duration_ms=duration, rows=len(missing_ids))
            write_data_audit_log("system_governance", "MISSING_SYNC", "N/A", datetime.now().strftime("%Y-%m-%d"), len(missing_ids))
        else:
            logger.info("✅ 目前無缺失數據的標的。")
            
    except Exception as e:
        logger.error(f"❌ 缺失偵測失敗: {e}")
        write_pipeline_log("Governance_MissingCheck", "MARKET", "FAILED", "Governance", err=str(e))
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    detect_and_fetch_missing()