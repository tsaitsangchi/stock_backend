import psycopg2
from scripts.config import DB_CONFIG

def create_users_table():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        sql = """
        CREATE TABLE IF NOT EXISTS public.users (
            id SERIAL PRIMARY KEY,
            gmail VARCHAR(255) UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name VARCHAR(255),
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL
        );
        """
        cur.execute(sql)
        conn.commit()
        print("Successfully created 'users' table or it already exists.")
        
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error creating users table: {e}")

if __name__ == "__main__":
    create_users_table()
