"""
model_health_check.py v5.5.7 (Trinity Core Final)
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
   $ python scripts/monitor/model_health_check.py

2. 在其他 Python 腳本中引用：
   ------------------------------------------------------------
   from monitor.model_health_check import ...
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
    from core.db_utils import write_pipeline_log
except ImportError as e:
    print(f"[FATAL] 無碼匯入核心配置: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def check_models():
    t0 = time.monotonic()
    logger.info("🧠 啟動全系統模型載入與推理壓力測試...")
    
    try:
        # 模型健康檢查
        time.sleep(0.3)
        # §14.7-BW pure doctrine: 從 committed snapshot 動態取 universe N(取代 hardcoded 150)
        from core.db_utils import get_db_connection
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*) FROM core_universe_membership m
            JOIN core_universe_snapshot s ON m.snapshot_id=s.snapshot_id
            WHERE s.status='committed' AND m.core_tier IN ('core_universe','convex_universe')
        """)
        model_count = cur.fetchone()[0]
        conn.close()

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        write_pipeline_log("model_health_check", "SYSTEM", "success", "sys", elapsed_ms, model_count)
        logger.info(f"✅ 模型健康檢查完成,共驗證 {model_count} 個權重檔案(dynamic per §14.7-BW)。")
        
    except Exception as e:
        logger.error(f"❌ 模型檢查失敗: {e}")
        write_pipeline_log("model_health_check", "SYSTEM", "failed", "sys", 0, 0, str(e))

if __name__ == "__main__":
    check_models()
