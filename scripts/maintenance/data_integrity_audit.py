"""
data_integrity_audit.py v5.5.7 (Trinity Core Final)
================================================================================
系統組件 — 混合模式日誌實作版
目錄：maintenance

修訂歷程：
  v5.5.7 (2026-05-09):
    - [文檔] 補齊極致詳細的執行範例說明。
  v5.5.x (2026-05-09):
    - [核心] 導入 Hybrid Logging 混合日誌與路徑標準化。

【執行範例說明】

1. 直接從命令行執行：
   $ python scripts/maintenance/data_integrity_audit.py

2. 在其他 Python 腳本中引用：
   ------------------------------------------------------------
   from maintenance.data_integrity_audit import ...
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

def run_audit():
    t0 = time.monotonic()
    logger.info("⚖️ 啟動全系統資料完整性審計...")
    
    error_count = 0
    try:
        with db_transaction() as cur:
            # 模擬審計邏輯：檢查關鍵表是否有空值
            cur.execute("SELECT count(*) as cnt FROM stock_price WHERE close IS NULL;")
            res = cur.fetchone()
            error_count = res['cnt'] if res else 0
            
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        write_pipeline_log("data_integrity_audit", "SYSTEM", "success", "sys", elapsed_ms, error_count)
        logger.info(f"✅ 審計完成，發現異常筆數: {error_count}")
        
    except Exception as e:
        logger.error(f"❌ 審計失敗: {e}")
        write_pipeline_log("data_integrity_audit", "SYSTEM", "failed", "sys", 0, 0, str(e))

if __name__ == "__main__":
    run_audit()
