"""
fetch_missing_stocks_data.py — 自動補齊 config.py 中新增個股的歷史數據
"""
import subprocess
import sys
import psycopg2
from pathlib import Path
from config import STOCK_CONFIGS

VENV_PYTHON = "/home/hugo/project/stock_backend/venv/bin/python3"
SCRIPTS_DIR = Path(__file__).parent
DB_CONFIG = {
    "dbname": "stock",
    "user": "stock",
    "password": "stock",
    "host": "172.31.122.166",
    "port": "5432",
}

def run_script(script_name, args=[]):
    cmd = [VENV_PYTHON, str(SCRIPTS_DIR / script_name)] + args
    print(f"執行指令: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def get_missing_data_stocks():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    missing = []
    for sid in STOCK_CONFIGS.keys():
        cur.execute("SELECT COUNT(*) FROM stock_price WHERE stock_id = %s", (sid,))
        if cur.fetchone()[0] < 100:  # 如果資料少於 100 筆，視為需要補齊
            missing.append(sid)
    
    conn.close()
    return missing

def main():
    # 1. 更新股票基本資訊
    print("=== 更新股票基本資訊 (stock_info) ===")
    run_script("fetch_stock_info.py")

    # 2. 找出需要補資料的個股
    missing_stocks = get_missing_data_stocks()
    if not missing_stocks:
        print("所有個股資料皆已充足。")
        return

    print(f"發現需要補資料的個股：{', '.join(missing_stocks)}")
    
    data_scripts = [
        "fetch_technical_data.py",
        "fetch_fundamental_data.py",
        "fetch_chip_data.py",
        "fetch_derivative_data.py",
        "fetch_macro_data.py",
        "fetch_international_data.py"
    ]

    for sid in missing_stocks:
        print(f"\n=== 正在準備個股 {sid} ({STOCK_CONFIGS[sid]['name']}) 的資料更新 ===")
        
    # 直接執行全量更新腳本，它們會自動偵測 stock_info 中新增的股票並補齊資料
    print("\n=== 執行全量增量更新 ===")
    for script in data_scripts:
        try:
            run_script(script)
        except Exception as e:
            print(f"執行 {script} 時發生錯誤: {e}")

    print("\n✅ 所有新個股資料抓取完成。")

if __name__ == "__main__":
    main()
