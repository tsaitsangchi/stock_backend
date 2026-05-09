"""
migrate_stocks_config.py v5.5.9 (Trinity Core Final)
================================================================================
資產矩陣同步引擎 — 系統核心思想落地版
負責將 config.py 中的 151 檔核心標的同步至資料庫，並重置非核心標的。

修訂歷程：
  v5.5.9 (2026-05-09):
    - [核心] 實作「核心標的」清洗邏輯：全市場 3096 -> 核心 151。
    - [規範] 導入 v5.5.x 混合日誌與極致文檔標準。
"""

import sys
import logging
import time
from pathlib import Path

# ── 系統路徑修復 ──
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
    from config import STOCK_CONFIGS
except ImportError as e:
    print(f"[FATAL] 無法匯入核心配置: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def migrate_core_assets():
    """執行資產矩陣同步，確保資料庫標的符合系統核心思想"""
    t0 = time.monotonic()
    logger.info(f"🔄 正在將系統核心思想 (151 檔標的) 同步至資料庫...")
    
    try:
        with db_transaction() as cur:
            # 1. 先將全市場標的設為「非監控」狀態
            cur.execute("UPDATE stocks SET is_active = FALSE, is_core = FALSE")
            
            # 2. 逐一啟用 config.py 中的核心標的
            for sid, cfg in STOCK_CONFIGS.items():
                cur.execute("""
                    UPDATE stocks 
                    SET is_active = TRUE, 
                        is_core = TRUE,
                        stock_name = %s,
                        industry = %s,
                        fetch_basic = TRUE,
                        fetch_chip = TRUE,
                        fetch_fundamental = TRUE,
                        fetch_news = TRUE
                    WHERE stock_id = %s
                """, (cfg['name'], cfg.get('industry', 'N/A'), sid))
                
                # 如果資料庫裡還沒有這檔股票 (例如新加入核心的)，則執行 INSERT
                if cur.rowcount == 0:
                    cur.execute("""
                        INSERT INTO stocks (stock_id, stock_name, industry, is_active, is_core, fetch_basic, fetch_chip, fetch_fundamental, fetch_news)
                        VALUES (%s, %s, %s, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE)
                    """, (sid, cfg['name'], cfg.get('industry', 'N/A')))
        
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        write_pipeline_log("migrate_stocks_config", "SYSTEM", "success", "sys", elapsed_ms, len(STOCK_CONFIGS))
        logger.info(f"✅ 核心資產同步完成！目前監控中心標的數: {len(STOCK_CONFIGS)}")
        
    except Exception as e:
        logger.error(f"❌ 同步失敗: {e}")
        write_pipeline_log("migrate_stocks_config", "SYSTEM", "failed", "sys", 0, 0, str(e))

if __name__ == "__main__":
    migrate_core_assets()