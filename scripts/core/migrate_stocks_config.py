"""
Asset Matrix Configuration Sync Engine v2.0
===========================================
負責將 Python 定義的 150 檔資產矩陣與美股關聯設定，原子性地同步至 PostgreSQL 資料庫。
支援欄位綱要自動擴展（Schema Expansion）與冪等 UPSERT 操作。
"""

import sys
import json
import logging
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_BASE_DIR = _THIS_DIR.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from core.db_utils import get_db_conn
from config import STOCK_CONFIGS 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def migrate():
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            # 1. 動態資料表綱要擴展 (Schema Expansion)
            logger.info("Verifying and expanding 'stocks' table schema...")
            
            # 定義必要欄位及其類型
            required_columns = {
                "is_core": "BOOLEAN DEFAULT FALSE",
                "us_chain": "TEXT",
                "is_active": "BOOLEAN DEFAULT TRUE",
                "fetch_basic": "BOOLEAN DEFAULT TRUE",
                "fetch_chip": "BOOLEAN DEFAULT TRUE",
                "fetch_fundamental": "BOOLEAN DEFAULT TRUE",
                "fetch_derivative": "BOOLEAN DEFAULT FALSE",
                "fetch_news": "BOOLEAN DEFAULT FALSE",
                "description_zh": "TEXT"
            }
            
            for col, col_type in required_columns.items():
                cur.execute(f"ALTER TABLE stocks ADD COLUMN IF NOT EXISTS {col} {col_type};")
            
            # 2. 核心資產矩陣同步 (Idempotent Upsert)
            logger.info(f"Synchronizing {len(STOCK_CONFIGS)} core assets to database...")
            
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
            
            migrated_count = 0
            for stock_id, metadata in STOCK_CONFIGS.items():
                name = metadata.get("name", "")
                industry = metadata.get("industry", "")
                # 將美股連動列表序列化為 JSON 儲存
                us_chain_list = metadata.get("us_chain_tickers", [])
                us_chain_json = json.dumps(us_chain_list) if us_chain_list else None
                
                cur.execute(upsert_query, (stock_id, name, industry, us_chain_json))
                migrated_count += 1
                
        conn.commit()
        logger.info(f"✅ Successfully migrated {migrated_count} stock configurations.")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"❌ Migration failed: {e}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
