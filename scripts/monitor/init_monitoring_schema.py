"""
init_monitoring_schema.py v5.5.7 (Trinity Core Final)
================================================================================
系統組件 — 混合模式日誌實作版
目錄：monitor

修訂歷程：
  v5.5.7 (2026-05-09):
    - [文檔] 補齊極致詳細的執行範例說明。
  v5.5.x (2026-05-09):
    - [核心] 導入 Hybrid Logging 混合日誌與路徑標準化。

【執行範例說明】

1. 直接從命令行執行：
   $ python scripts/monitor/init_monitoring_schema.py

2. 在其他 Python 腳本中引用：
   ------------------------------------------------------------
   from monitor.init_monitoring_schema import ...
   ------------------------------------------------------------

3. 日誌查閱：
   SELECT * FROM pipeline_execution_log ORDER BY created_at DESC LIMIT 10;
"""

import sys
import logging
import time
from pathlib import Path

# ── 系統路徑修復 (v3.0) ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import db_transaction, write_pipeline_log
except ImportError as e:
    print(f"[FATAL] 無法匯入核心配置: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def init_schema():
    t0 = time.monotonic()
    logger.info("🏗️ 正在初始化監控架構 DDL (Pipeline & Evaluation Logs)...")
    
    try:
        with db_transaction() as cur:
            # 1. 建立核心生命週期日誌表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_execution_log (
                    id SERIAL PRIMARY KEY,
                    task_name VARCHAR(100),
                    stock_id VARCHAR(20),
                    status VARCHAR(20),
                    category VARCHAR(20),
                    duration_ms INTEGER,
                    rows_processed INTEGER,
                    error_msg TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            # 🔴 補丁：確保 created_at 欄位存在
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='pipeline_execution_log' AND column_name='created_at';")
            if not cur.fetchone():
                cur.execute("ALTER TABLE pipeline_execution_log ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;")

            # 🔴 補丁：確保 stocks.last_sync_at 欄位存在
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='stocks' AND column_name='last_sync_at';")
            if not cur.fetchone():
                cur.execute("ALTER TABLE stocks ADD COLUMN last_sync_at TIMESTAMP;")
            # 2. 建立業務指標評估表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_log (
                    id SERIAL PRIMARY KEY,
                    strategy_name VARCHAR(100),
                    stock_id VARCHAR(20),
                    metric_name VARCHAR(50),
                    metric_value FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        write_pipeline_log("init_monitoring_schema", "SYSTEM", "success", "sys", elapsed_ms, 2)
        logger.info("✅ 監控架構 DDL 初始化完成。")
        
    except Exception as e:
        logger.error(f"❌ 初始化失敗: {e}")
        write_pipeline_log("init_monitoring_schema", "SYSTEM", "failed", "sys", 0, 0, str(e))

if __name__ == "__main__":
    init_schema()
