import psycopg2
from scripts.data_pipeline import DB_CONFIG

try:
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT stock_id, COUNT(*) FROM stock_forecast_daily GROUP BY stock_id;")
    rows = cur.fetchall()
    print("Stock Predictions in DB:")
    for row in rows:
        print(f"  {row[0]}: {row[1]} records")
    conn.close()
except Exception as e:
    print(f"Error: {e}")
