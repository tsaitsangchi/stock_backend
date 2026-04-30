import psycopg2
from config import DB_CONFIG

def get_db_connection():
    """
    統一取得資料庫連線。
    """
    return psycopg2.connect(**DB_CONFIG)

def run_query(query: str, params: tuple = None, fetch: bool = True):
    """
    執行查詢並自動管理連線與游標。
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            if fetch:
                return cur.fetchall()
            conn.commit()
    finally:
        conn.close()
