import sys
import os
from pathlib import Path
_base = Path(__file__).resolve().parent.parent
sys.path.append(str(_base))
from core.db_utils import get_db_conn

def check_db_locks():
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            print("--- 執行中的長時間查詢 ---")
            cur.execute("""
                SELECT pid, now() - query_start AS duration, query, state
                FROM pg_stat_activity
                WHERE state != 'idle' AND now() - query_start > interval '1 minute'
                ORDER BY duration DESC;
            """)
            rows = cur.fetchall()
            for r in rows:
                print(f"PID: {r[0]}, Duration: {r[1]}, State: {r[3]}\nQuery: {r[2][:200]}...")
                
            print("\n--- 等待鎖定的查詢 ---")
            cur.execute("""
                SELECT pid, wait_event_type, wait_event, query 
                FROM pg_stat_activity 
                WHERE wait_event IS NOT NULL AND state != 'idle';
            """)
            rows = cur.fetchall()
            for r in rows:
                print(f"PID: {r[0]}, Wait: {r[1]}/{r[2]}\nQuery: {r[3][:200]}...")
    finally:
        conn.close()

if __name__ == "__main__":
    check_db_locks()
