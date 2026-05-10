"""
db.py v5.5.26 (Trinity Core Final)
================================================================================
資料庫輔助工具 — 混合日誌整合版
作為 core.db_utils 的輕量化封裝介面，提供快速的連線與交易處理。

修訂歷程：
  v5.5.26 (2026-05-10):
    - [對接] 引用 core.db_utils (v4.14) 進程安全連線池。
    - [文檔] 補齊極致詳細的執行範例與 SQL 稽核指引。

【執行範例說明】

1. 快速執行 SQL 查詢：
   ------------------------------------------------------------
   from utils.db import db_transaction
   with db_transaction() as cur:
       cur.execute("SELECT COUNT(*) FROM stocks")
       print(cur.fetchone())
   ------------------------------------------------------------

2. SQL 維運查閱：
   SELECT * FROM pipeline_execution_log WHERE task_name = 'db_util_check' ORDER BY created_at DESC;
"""

import sys
from pathlib import Path

# ── 系統路徑修復 (v3.1) ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import db_session, db_transaction, write_pipeline_log
except ImportError as e:
    print(f"[FATAL] 無法匯入核心組件: {e}", file=sys.stderr)
    sys.exit(1)

def check_connection():
    """驗證連線狀況並紀錄"""
    try:
        with db_transaction() as cur:
            cur.execute("SELECT 1")
            write_pipeline_log("db_util_check", "SYSTEM", "success", "sys")
            return True
    except Exception as e:
        write_pipeline_log("db_util_check", "SYSTEM", "failed", "sys", 0, 0, str(e))
        return False

if __name__ == "__main__":
    if check_connection():
        print("✅ 資料庫連線測試成功。")
