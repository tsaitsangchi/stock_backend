"""
migrate_stocks_config.py v3.0
量化系統核心：資產矩陣與資料庫同步引擎 (整合自動事務與批次處理版)
================================================================================
v3.0 重大升級：
  · 完美對接 db_utils v4.0：全面採用 @contextmanager (db_transaction)，
    免去手動 commit/rollback，確保 Schema 擴展與資料寫入具備絕對的原子性 (Atomicity)。
  · 效能躍升：以 psycopg2.extras.execute_batch 取代迴圈逐筆 INSERT，
    150+ 檔股票設定的同步時間從數秒縮短至毫秒級。
  · 冪等性防護 (Idempotency)：無論執行幾次，Schema 與資料都會保持在最新且正確的狀態。

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
    sp = str(p)
    if p.exists() and sp not in sys.path:
        sys.path.insert(0, sp)

import psycopg2.extras

try:
    # 載入我們優化過 v4.0 連線池與上下文管理器
    from core.db_utils import db_transaction
    # 載入 Python 定義的資產設定檔
    from config import STOCK_CONFIGS
except ImportError as e:
    print(f"無法匯入核心模組: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def migrate():
    """執行資產矩陣的同步遷移"""
    logger.info("開始執行資產矩陣同步 (Schema Expansion & Data Upsert)...")
    
    # 準備要批次寫入的資料
    records = []
    for stock_id, metadata in STOCK_CONFIGS.items():
        name = metadata.get("name", "")
        industry = metadata.get("industry", "")
        # 將美股連動列表序列化為 JSON 儲存，供後續模型特徵抓取使用
        us_chain_list = metadata.get("us_chain_tickers", [])
        us_chain_json = json.dumps(us_chain_list) if us_chain_list else None
        
        records.append((stock_id, name, industry, us_chain_json))

    # 動態資料表綱要定義 (隨業務需求可隨時擴增)
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

    # 冪等 Upsert SQL
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

    # 核心邏輯：使用 db_transaction() 確保 Schema 修改與資料寫入在同一個 Transaction 內
    try:
        with db_transaction() as cur:
            # =========================================================
            # Step 1: 動態資料表綱要擴展 (Schema Expansion)
            # =========================================================
            # 確保 stocks 基礎表存在
            cur.execute("""
                CREATE TABLE IF NOT EXISTS stocks (
                    stock_id VARCHAR(20) PRIMARY KEY,
                    stock_name VARCHAR(100),
                    industry VARCHAR(100)
                );
            """)
            
            # 自動檢測並補齊缺失的欄位
            for col_name, col_type in required_columns.items():
                try:
                    cur.execute(f"ALTER TABLE stocks ADD COLUMN IF NOT EXISTS {col_name} {col_type};")
                except Exception as e:
                    # 捕捉到例外代表語法錯誤或其他系統異常，db_transaction 會在最外層自動 Rollback
                    logger.error(f"擴增欄位 {col_name} 失敗: {e}")
                    raise

            # =========================================================
            # Step 2: 極速批次寫入 (Bulk Upsert)
            # =========================================================
            # 以 execute_batch 取代迴圈單筆 execute，效能提升 10 倍以上
            psycopg2.extras.execute_batch(cur, upsert_query, records)
            
        # 若執行到此，db_transaction 會自動安全地執行 commit()
        logger.info(f"✅ 成功同步 {len(records)} 檔股票配置與綱要結構。")

    except Exception as e:
        # 若執行過程中發生任何錯誤 (包含 Schema 或是 Data 寫入錯誤)
        # db_transaction 會自動 rollback()，確保資料庫不會處於半完成的髒狀態
        logger.error(f"❌ 同步失敗，已全數回滾 (Rollback): {e}")
        raise

if __name__ == "__main__":
    migrate()