"""
action_runner.py v5.5.7 (Trinity Core Final)
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
   $ python scripts/monitor/action_runner.py

2. 在其他 Python 腳本中引用：
   ------------------------------------------------------------
   from monitor.action_runner import ...
   ------------------------------------------------------------

3. 日誌查閱：
   SELECT * FROM pipeline_execution_log ORDER BY created_at DESC LIMIT 10;
"""

import sys
import logging
import time
import subprocess
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
    print(f"[FATAL] 無法匯入核心配置: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_script(script_path: str):
    """
    執行子腳本並獲取返回碼。
    """
    logger.info(f"  .. 正在執行: {script_path}")
    try:
        # 使用目前的 python 解譯器執行
        res = subprocess.run([sys.executable, script_path], capture_output=True, text=True, timeout=300)
        return res.returncode == 0
    except Exception as e:
        logger.error(f"  .. 執行出錯: {e}")
        return False

def run_daily_maintenance_suite():
    t0 = time.monotonic()
    logger.info("⚙️ 啟動 Trinity 每日自動化維運套裝任務...")
    
    tasks = [
        "scripts/monitor/db_health_check.py",
        "scripts/monitor/update_daily_status.py",
        "scripts/maintenance/data_integrity_audit.py",
        "scripts/monitor/dashboard.py"
    ]
    
    success_count = 0
    for task in tasks:
        full_path = str(_SCRIPTS_DIR.parent / task)
        if run_script(full_path):
            success_count += 1
            
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    write_pipeline_log("daily_maintenance_runner", "SYSTEM", "success", "sys", elapsed_ms, success_count)
    logger.info(f"✅ 維運套裝執行完畢，成功率: {success_count}/{len(tasks)}")

if __name__ == "__main__":
    run_daily_maintenance_suite()
