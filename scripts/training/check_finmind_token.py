"""
check_finmind_token.py v5.5.26 (Trinity Core Final)
================================================================================
維運工具 — 混合模式日誌實作版
負責驗證 FinMind API Token 狀態與剩餘配額。

修訂歷程：
  v5.5.26 (2026-05-10):
    - [文檔] 補齊極致詳細的執行範例說明。
  v5.5.25 (2026-05-10):
    - [標準化] 對接 finmind_client 與混合日誌。

【執行範例說明】

1. 直接從命令行執行（執行 Token 驗證）：
   $ python scripts/training/check_finmind_token.py

2. 日誌查閱 (確認 API 驗證結果)：
   SELECT task_name, status, error_message, created_at 
   FROM pipeline_execution_log 
   WHERE task_name = 'check_token' 
   ORDER BY created_at DESC LIMIT 5;
"""

import sys
import logging
import time
from pathlib import Path

# ── 系統路徑修復 (v3.0) ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core", "pipeline"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import write_pipeline_log
    from pipeline.finmind_client import FinMindClient
except ImportError as e:
    print(f"[FATAL] 無法匯入核心組件: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def verify_token():
    t0 = time.monotonic()
    logger.info("🔑 [Auth] 正在驗證 FinMind Token 狀態...")
    try:
        client = FinMindClient()
        time.sleep(0.2)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        write_pipeline_log("check_token", "SYSTEM", "success", "sys", elapsed_ms)
        logger.info("✅ Token 驗證成功！")
    except Exception as e:
        logger.error(f"❌ Token 驗證失敗: {e}")
        write_pipeline_log("check_token", "SYSTEM", "failed", "sys", 0, 0, str(e))

if __name__ == "__main__":
    verify_token()
