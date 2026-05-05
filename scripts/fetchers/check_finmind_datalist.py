import requests
import json
import sys
import os
import time
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.finmind_client import FINMIND_TOKEN

def check_options_large_oi():
    dataset = "TaiwanOptionOpenInterestLargeTraders"
    # 測試 2018-01-01 (使用 Header)
    url = f"https://api.finmindtrade.com/api/v4/data?dataset={dataset}&start_date=2018-01-01"
    headers = {"Authorization": f"Bearer {FINMIND_TOKEN}"}
    
    try:
        print(f"Fetching {dataset} for 2018-01-01 (Using Header)...")
        t0 = time.time()
        resp = requests.get(url, headers=headers)
        elapsed = time.time() - t0
        print(f"Status: {resp.status_code}")
        print(f"Time taken: {elapsed:.2f} seconds")
        
        res_json = resp.json()
        data = res_json.get("data", [])
        print(f"Total rows returned: {len(data)}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_options_large_oi()
