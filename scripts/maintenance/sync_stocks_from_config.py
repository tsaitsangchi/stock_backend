"""
sync_stocks_from_config.py v5.5 (Trinity Core Edition)
================================================================================
股票清單同步工具 — 混合模式日誌實作版
此模組負責將 scripts/config.py 中的股票清單與設定同步至資料庫 stocks 表中。

核心功能：
  · 配置同步       ─ 自動更新個股名稱、產業分類與美股聯動標籤。
  · 執行紀錄       ─ 對接 write_pipeline_log，標記為 sys_v5.1 (Maintenance)。

修訂歷程：
  v5.5 (2026-05-09):
    - [規範] 導入混合模式日誌。
    - [核心] 對接 path_setup v3.0 與 db_utils v4.7 標準。
"""

import sys
import logging
import time
from pathlib import Path

# ── 系統路徑修復 (對接 path_setup v3.0) ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import db_transaction, write_pipeline_log, ensure_ddl
    from config import STOCK_CONFIGS
except ImportError as e:
    print(f"[FATAL] 無法匯入核心配置: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_sync():
    t0 = time.monotonic()
    logger.info(f"🔄 正在從 config.py 同步 {len(STOCK_CONFIGS)} 支標的至資料庫...")
    
    count = 0
    try:
        with db_transaction() as cur:
            for sid, cfg in STOCK_CONFIGS.items():
                cur.execute("""
                    INSERT INTO stocks (stock_id, stock_name, industry, is_core, updated_at)
                    VALUES (%s, %s, %s, TRUE, CURRENT_TIMESTAMP)
                    ON CONFLICT (stock_id) DO UPDATE SET
                        stock_name = EXCLUDED.stock_name,
                        industry = EXCLUDED.industry,
                        updated_at = CURRENT_TIMESTAMP;
                """, (sid, cfg.get("name", ""), cfg.get("industry", "Unknown")))
                count += 1
        
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        write_pipeline_log("sync_stocks", "SYSTEM", "success", "sys", elapsed_ms, count)
        logger.info(f"✅ 同步完成，共處理 {count} 筆標的。")
        
    except Exception as e:
        logger.error(f"❌ 同步失敗: {e}")
        write_pipeline_log("sync_stocks", "SYSTEM", "failed", "sys", 0, 0, str(e))

if __name__ == "__main__":
    run_sync()
