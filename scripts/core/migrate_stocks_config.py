"""
migrate_stocks_config.py v6.0 (Quantum Finance Edition)
================================================================================
資產配置遷移工具 — 混合日誌標準版 (Quantum v5.1 標準)
將 config.py 中的 128 檔核心資產清單同步至 PostgreSQL 資料庫，並清理冗餘標的。

修訂歷程：
  v6.0 (2026-05-10): [文檔] 補齊「核心股」與「強制清理」執行範例矩陣。
  v5.9 (2026-05-10): [文檔] 補齊基礎執行範例。

【執行範例矩陣 — 資產管理方案】
1. 同步 128 檔全量核心清單：
   python scripts/core/migrate_stocks_config.py
2. 強制清理非核心標的：
   本腳本會自動執行 DELETE FROM stocks WHERE is_active = FALSE，保持 Table 潔淨。
3. 核心資產狀態檢查 (SQL)：
   SELECT stock_id, stock_name, industry FROM stocks WHERE is_core = TRUE ORDER BY stock_id;
4. 遷移執行日誌稽核 (SQL)：
   SELECT * FROM pipeline_execution_log WHERE task_name = 'migrate_stocks_config';
================================================================================
"""
import sys, logging, time
from pathlib import Path

# ── 終極路徑自癒 Bootstrap (核心自救版) ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_SCRIPTS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR.parent))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
except ImportError:
    import path_setup
    path_setup.ensure_scripts_on_path(__file__)

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