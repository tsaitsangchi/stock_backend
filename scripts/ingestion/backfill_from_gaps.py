"""
backfill_from_gaps.py v4.3 (Quantum Finance Edition)
================================================================================
數據治理工具 — 歷史斷層回填 (Quantum v5.1 標準)
負責偵測資料表中日期不連續的斷層 (Gaps)，並精準發起回填任務。

修訂歷程：
  v4.3 (2026-05-10): [核心] 實作混合模式日誌，對齊 v5.1 治理紀錄規範。
  v4.2 (2026-05-10): [文件] 完善五維度執行範例矩陣，確保範例完整性。
  v4.1 (2026-05-10): [修復] 對齊 db_utils v5.1 混合日誌規範。

【執行範例矩陣 — 數據治理方案】
1. 偵測全市場價格數據斷層並回填 (Python)：
   python scripts/ingestion/backfill_from_gaps.py --table stock_price
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
import os, sys, logging, time, argparse
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

def detect_and_backfill_gaps(table_name: str):
    start_time = time.time()
    conn = get_db_connection()
    cur = conn.cursor()
    
    logger.info(f"🔍 正在偵測 {table_name} 中的數據斷層...")
    
    try:
        # 使用 LEAD 窗口函數偵測日期跳躍 (大於 4 天視為斷層)
        query = f"""
        SELECT stock_id, date, next_date
        FROM (
            SELECT stock_id, date, LEAD(date) OVER (PARTITION BY stock_id ORDER BY date) as next_date
            FROM {table_name}
        ) t
        WHERE next_date - date > 4
        """
        cur.execute(query)
        gaps = cur.fetchall()
        
        if gaps:
            logger.info(f"🚨 發現 {len(gaps)} 處數據斷層，啟動精準回填...")
            for sid, start, end in gaps:
                logger.info(f"🔄 回填 {sid}: {start} -> {end}")
                cmd = f"{sys.executable} {_THIS_DIR}/parallel_fetch.py --stock_id {sid} --table ALL"
                os.system(cmd)
            
            duration = int((time.time() - start_time) * 1000)
            # 🔴 治理層混合日誌
            write_pipeline_log("Governance_GapFill", table_name, "SUCCESS", "Governance", duration_ms=duration, rows=len(gaps))
            write_data_audit_log("system_governance", f"GAP_FILL_{table_name.upper()}", "N/A", datetime.now().strftime("%Y-%m-%d"), len(gaps))
        else:
            logger.info(f"✅ {table_name} 目前無數據斷層。")
            
    except Exception as e:
        logger.error(f"❌ 斷層偵測失敗: {e}")
        write_pipeline_log("Governance_GapFill", table_name, "FAILED", "Governance", err=str(e))
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=str, default="stock_price")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    detect_and_backfill_gaps(args.table)