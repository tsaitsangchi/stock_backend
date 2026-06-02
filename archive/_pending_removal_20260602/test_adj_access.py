"""
test_adj_access.py v5.5.26 (Trinity Core Final)
================================================================================
測試工具 — 混合模式日誌實作版
負責驗證資料庫中 price_adj 表格的存取權限與數據完整性。

修訂歷程：
  v5.5.26 (2026-05-10):
    - [文檔] 補齊極致詳細的執行範例說明。
  v5.5.25 (2026-05-10):
    - [標準化] 對接 db_utils (v4.13)。

【執行範例說明】

1. 直接從命令行執行（執行 DB 存取測試）：
   $ python scripts/training/test_adj_access.py

2. 日誌查閱 (確認測試結果紀錄)：
   SELECT task_name, status, error_message, created_at 
   FROM pipeline_execution_log 
   WHERE task_name = 'db_access_test' 
   ORDER BY created_at DESC LIMIT 5;
"""

import sys
import logging
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
    print(f"[FATAL] 無法匯入核心組件: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def test_db_access():
    logger.info("🔍 [DB Test] 正在驗證 price_adj 資料表存取...")
    try:
        with db_transaction() as cur:
            cur.execute("SELECT COUNT(*) FROM price_adj LIMIT 1")
            count = cur.fetchone()['count']
            logger.info(f"✅ 成功！price_adj 表格正常 (紀錄數: {count})")
            write_pipeline_log("db_access_test", "SYSTEM", "success", "sys")
    except Exception as e:
        logger.error(f"❌ 存取失敗: {e}")
        write_pipeline_log("db_access_test", "SYSTEM", "failed", "sys", 0, 0, str(e))

if __name__ == "__main__":
    test_db_access()
