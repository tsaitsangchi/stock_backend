"""
fetch_stock_info.py
從 FinMind API 抓取 TaiwanStockInfo（台股總覽）並寫入 PostgreSQL stock_info 資料表。

需求套件：
    pip install requests psycopg2-binary
"""

import logging
import sys

import psycopg2
import psycopg2.extras
import requests

# ======================
# 設定 logging
# ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ======================
# FinMind API 設定
# ======================
FINMIND_API_URL = "https://api.finmindtrade.com/api/v4/data"
FINMIND_TOKEN = (
    "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9"
    ".eyJkYXRlIjoiMjAyNi0wMy0xNCAxODoxNTo1NCIsInVzZXJfaWQiOiJ0c2FpdHNhbmdjaGkiLCJlbWFpbCI6InRzYWl0c2FuZ2NoaUBnbWFpbC5jb20iLCJpcCI6IjIyMC4xMzQuMjYuNzAifQ"
    ".muoHEMMLiiRQoxZj7evq-9hclsVRXE3IfLNZWDZ6PQE"
)

# ======================
# PostgreSQL 連線設定
# ======================
DB_CONFIG = {
    "dbname": "stock",
    "user": "stock",
    "password": "stock",
    "host": "localhost",
    "port": "5432",
}

# ======================
# DDL（若資料表尚未存在則建立）
# ======================
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS stock_info (
    stock_id          VARCHAR(50) PRIMARY KEY,
    stock_name        VARCHAR(100),
    industry_category VARCHAR(100),
    type              VARCHAR(50),
    date              DATE
);
CREATE INDEX IF NOT EXISTS idx_stock_info_date ON stock_info (date);
"""

# ======================
# Upsert SQL
# ======================
UPSERT_SQL = """
INSERT INTO stock_info (stock_id, stock_name, industry_category, type, date)
VALUES %s
ON CONFLICT (stock_id)
DO UPDATE SET
    stock_name        = EXCLUDED.stock_name,
    industry_category = EXCLUDED.industry_category,
    type              = EXCLUDED.type,
    date              = EXCLUDED.date;
"""


# ──────────────────────────────────────────────
# 1. 從 FinMind 抓資料
# ──────────────────────────────────────────────
def fetch_stock_info() -> list[dict]:
    """
    呼叫 FinMind TaiwanStockInfo API，回傳原始 data list。
    TaiwanStockInfo 不需要帶 data_id / start_date / end_date。
    """
    headers = {"Authorization": f"Bearer {FINMIND_TOKEN}"}
    params = {"dataset": "TaiwanStockInfo"}

    logger.info("呼叫 FinMind API：TaiwanStockInfo ...")
    resp = requests.get(FINMIND_API_URL, headers=headers, params=params, timeout=60)

    # HTTP 層錯誤
    resp.raise_for_status()

    payload = resp.json()

    # API 業務層錯誤（如 402 用量超出）
    status = payload.get("status")
    if status != 200:
        raise RuntimeError(
            f"FinMind API 回傳非成功狀態：status={status}, msg={payload.get('msg')}"
        )

    data = payload.get("data", [])
    logger.info(f"共取得 {len(data)} 筆股票資料")
    return data


# ──────────────────────────────────────────────
# 2. 轉換為 tuple list（對應 DB 欄位順序）
# ──────────────────────────────────────────────
def transform(records: list[dict]) -> list[tuple]:
    """
    將 API 回傳的 dict list 轉換成
    (stock_id, stock_name, industry_category, type, date) tuple list。
    date 欄位若為空字串則設為 None。
    """
    seen = {}
    for r in records:
        stock_id = r.get("stock_id")
        if not stock_id:
            continue
        # 同一 stock_id 保留日期最新的那筆
        _date = r.get("date", "")
        date_val = _date if (_date and _date.upper() != "NONE") else None
        if stock_id not in seen or (date_val and date_val > (seen[stock_id][4] or "")):
            seen[stock_id] = (
                stock_id,
                r.get("stock_name"),
                r.get("industry_category"),
                r.get("type"),
                date_val,
            )

    rows = list(seen.values())
    logger.info(f"去重後剩 {len(rows)} 筆（原始 {len(records)} 筆）")
    return rows


# ──────────────────────────────────────────────
# 3. 寫入 PostgreSQL
# ──────────────────────────────────────────────
def upsert_to_db(rows: list[tuple]) -> None:
    """
    連線 PostgreSQL，確保 DDL 存在後，批次 upsert 所有資料。
    """
    if not rows:
        logger.warning("沒有資料可寫入，程式結束。")
        return

    logger.info(f"連線 PostgreSQL（{DB_CONFIG['host']}:{DB_CONFIG['port']} / {DB_CONFIG['dbname']}）...")
    conn = psycopg2.connect(**DB_CONFIG)

    try:
        with conn:
            with conn.cursor() as cur:
                # 建立資料表 & 索引（冪等）
                logger.info("確認 stock_info 資料表與索引是否存在 ...")
                cur.execute(CREATE_TABLE_SQL)

                # 批次 upsert（execute_values 效率遠優於逐筆 execute）
                logger.info(f"開始 upsert {len(rows)} 筆資料 ...")
                psycopg2.extras.execute_values(
                    cur,
                    UPSERT_SQL,
                    rows,
                    template="(%s, %s, %s, %s, %s::date)",
                    page_size=1000,
                )

        logger.info("✅ 資料寫入完成（已 commit）。")

    except Exception as exc:
        logger.error(f"寫入 DB 時發生錯誤：{exc}")
        raise

    finally:
        conn.close()
        logger.info("PostgreSQL 連線已關閉。")


# ──────────────────────────────────────────────
# 主程式
# ──────────────────────────────────────────────
def main():
    try:
        # Step 1：抓資料
        raw_data = fetch_stock_info()

        # Step 2：轉換
        rows = transform(raw_data)

        # Step 3：寫入 DB
        upsert_to_db(rows)

    except requests.HTTPError as e:
        logger.error(f"HTTP 錯誤：{e}")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)
    except psycopg2.OperationalError as e:
        logger.error(f"無法連線 PostgreSQL：{e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"未預期的錯誤：{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
