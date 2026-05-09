"""
migrate_stocks_config.py v3.0 (Trinity Core Edition)
================================================================================
量化系統核心：資產矩陣與資料庫同步引擎
此模組負責將 config.py 中的股票配置 (150+ 檔) 同步至 PostgreSQL 資料庫，
並自動處理資料表欄位擴展 (Schema Expansion) 與資料更新 (Upsert)。

核心功能：
  · 冪等性同步       ─ 無論執行幾次，Schema 與資料都會保持在最新狀態。
  · 自動 Schema 擴展 ─ 自動檢測並補齊 stocks 資料表缺失的控制欄位。
  · 極速批次寫入     ─ 以 execute_batch 實作，效能優於傳統迴圈寫入。

修訂歷程：
  v3.0 (2026-04-20):
    - [重大] 完美對接 db_utils v4.0，全面採用 db_transaction 確保原子性。
    - [效能] 導入 psycopg2.extras.execute_batch，同步時間縮短至毫秒級。
  v2.0 (2026-04-10):
    - [基礎] 實作 DDL 自動生成功能。

執行範例：
    # 常規同步 (當 config.py 中的股票清單有增減，或欄位有擴增時執行)
    python scripts/core/migrate_stocks_config.py
"""

import sys
import json
import logging
from pathlib import Path

# ── 系統路徑修復與標準化 ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for sub in ("", "core"):
    p = (_SCRIPTS_DIR / sub) if sub else _SCRIPTS_DIR
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import psycopg2.extras

try:
    from core.db_utils import db_transaction
    from config import STOCK_CONFIGS
except ImportError as e:
    print(f"無法匯入核心模組: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def migrate():
    """執行資產矩陣的同步遷移"""
    logger.info("開始執行資產矩陣同步 (Schema Expansion & Data Upsert)...")
    
    records = []
    for stock_id, metadata in STOCK_CONFIGS.items():
        name = metadata.get("name", "")
        industry = metadata.get("industry", "")
        us_chain_list = metadata.get("us_chain_tickers", [])
        us_chain_json = json.dumps(us_chain_list) if us_chain_list else None
        records.append((stock_id, name, industry, us_chain_json))

    required_columns = {
        "is_core": "BOOLEAN DEFAULT FALSE",
        "us_chain": "TEXT",
        "is_active": "BOOLEAN DEFAULT TRUE",
        "fetch_basic": "BOOLEAN DEFAULT TRUE",
        "fetch_chip": "BOOLEAN DEFAULT TRUE",
        "fetch_fundamental": "BOOLEAN DEFAULT TRUE",
        "fetch_derivative": "BOOLEAN DEFAULT FALSE",
        "fetch_news": "BOOLEAN DEFAULT FALSE",
    }

    upsert_query = """
        INSERT INTO stocks (
            stock_id, stock_name, industry, us_chain, 
            is_core, is_active, fetch_basic, fetch_chip, fetch_fundamental, fetch_news
        ) VALUES (
            %s, %s, %s, %s, 
            TRUE, TRUE, TRUE, TRUE, TRUE, FALSE
        )
        ON CONFLICT (stock_id) 
        DO UPDATE SET 
            stock_name = EXCLUDED.stock_name,
            industry   = EXCLUDED.industry,
            us_chain   = EXCLUDED.us_chain,
            is_core    = TRUE,
            is_active  = TRUE;
    """

    try:
        with db_transaction() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS stocks (
                    stock_id VARCHAR(20) PRIMARY KEY,
                    stock_name VARCHAR(100),
                    industry VARCHAR(100)
                );
            """)
            
            for col_name, col_type in required_columns.items():
                try:
                    cur.execute(f"ALTER TABLE stocks ADD COLUMN IF NOT EXISTS {col_name} {col_type};")
                except Exception as e:
                    logger.error(f"擴增欄位 {col_name} 失敗: {e}")
                    raise

            psycopg2.extras.execute_batch(cur, upsert_query, records)
            
        logger.info(f"✅ 成功同步 {len(records)} 檔股票配置與綱要結構。")

    except Exception as e:
        logger.error(f"❌ 同步失敗，已全數回滾 (Rollback): {e}")
        raise

if __name__ == "__main__":
    migrate()