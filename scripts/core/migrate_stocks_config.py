"""
migrate_stocks_config.py v5.8 (Trinity Core Final)
================================================================================
修訂歷程：
  v5.8 (2026-05-10): [修正] 強化路徑自癒 Bootstrap，解決 No module named 'core'。
"""
import sys, logging, time
from pathlib import Path

# ── 終極路徑自癒 Bootstrap ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS = None
for p in [_THIS_DIR, _THIS_DIR.parent, _THIS_DIR.parent.parent]:
    if p.name == "scripts" or (p / "scripts").exists():
        _SCRIPTS = p if p.name == "scripts" else (p / "scripts")
        break
if _SCRIPTS:
    if str(_SCRIPTS) not in sys.path: sys.path.insert(0, str(_SCRIPTS))
    if str(_SCRIPTS.parent) not in sys.path: sys.path.insert(0, str(_SCRIPTS.parent))

try:
    from core.db_utils import db_transaction, write_pipeline_log
except ImportError:
    from db_utils import db_transaction, write_pipeline_log

try:
    from config import STOCK_CONFIGS
except ImportError:
    # 嘗試向上尋找 config
    sys.path.append(str(_SCRIPTS))
    from config import STOCK_CONFIGS

def migrate_core_assets():
    t0 = time.monotonic()
    logging.info(f"🔄 正在同步核心資產 ({len(STOCK_CONFIGS)} 檔) 並清理元數據...")
    try:
        with db_transaction() as cur:
            # 1. 重置所有標的狀態與抓取旗標
            cur.execute("""
                UPDATE stocks SET 
                    is_active = FALSE, 
                    is_core = FALSE,
                    fetch_basic = FALSE,
                    fetch_chip = FALSE,
                    fetch_fundamental = FALSE,
                    fetch_derivative = FALSE,
                    fetch_news = FALSE
            """)
            # 2. 啟動核心標的並更新資訊
            for sid, cfg in STOCK_CONFIGS.items():
                cur.execute("""
                    UPDATE stocks SET 
                        is_active = TRUE, 
                        is_core = TRUE, 
                        fetch_basic = TRUE,
                        fetch_chip = TRUE,
                        fetch_fundamental = TRUE,
                        fetch_derivative = TRUE,
                        fetch_news = TRUE,
                        stock_name = %s, 
                        industry = %s 
                    WHERE stock_id = %s
                """, (cfg['name'], cfg.get('industry','N/A'), sid))
                if cur.rowcount == 0:
                    cur.execute("""
                        INSERT INTO stocks (stock_id, stock_name, industry, is_active, is_core, fetch_basic, fetch_chip, fetch_fundamental, fetch_derivative, fetch_news)
                        VALUES (%s, %s, %s, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE)
                    """, (sid, cfg['name'], cfg.get('industry','N/A')))
            
            # 3. 清理完全不屬於核心的冗餘資料 (保持 table 潔淨)
            cur.execute("DELETE FROM stocks WHERE is_active = FALSE")
            
        write_pipeline_log("migrate_stocks_config", "SYSTEM", "success", "sys", int((time.monotonic()-t0)*1000), len(STOCK_CONFIGS))
        logging.info("✅ 核心資產元數據同步完成")
    except Exception as e: logging.error(f"❌ 同步失敗: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    migrate_core_assets()