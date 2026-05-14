import sys
from pathlib import Path

_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

from core.db_utils import get_db_conn, ensure_ddl
from config import STOCK_CONFIGS

DDL_STOCKS = """
CREATE TABLE IF NOT EXISTS stocks (
    stock_id    VARCHAR(20) PRIMARY KEY,
    stock_name  VARCHAR(100),
    industry    VARCHAR(50),
    is_core     BOOLEAN DEFAULT TRUE,
    us_chain    TEXT[],
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_stocks_core ON stocks (is_core);
"""

def sync_stocks():
    conn = get_db_conn()
    try:
        print("正在建立/檢查 stocks 資料表...")
        ensure_ddl(conn, DDL_STOCKS)
        
        print(f"正在從 config.py 同步 {len(STOCK_CONFIGS)} 支股票...")
        with conn.cursor() as cur:
            for sid, cfg in STOCK_CONFIGS.items():
                name = cfg.get("name", "")
                industry = cfg.get("industry", "Unknown")
                us_chain = cfg.get("us_chain_tickers", [])
                
                cur.execute("""
                    INSERT INTO stocks (stock_id, stock_name, industry, is_core, us_chain, updated_at)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (stock_id) DO UPDATE SET
                        stock_name = EXCLUDED.stock_name,
                        industry = EXCLUDED.industry,
                        is_core = EXCLUDED.is_core,
                        us_chain = EXCLUDED.us_chain,
                        updated_at = NOW();
                """, (sid, name, industry, True, us_chain))
        
        conn.commit()
        print("同步完成！")
    finally:
        conn.close()

if __name__ == "__main__":
    sync_stocks()
