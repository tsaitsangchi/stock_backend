import requests
import json
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.finmind_client import FINMIND_TOKEN

def search_datasets():
    # 搜尋包含 Option 的資料集
    url = f"https://api.finmindtrade.com/api/v4/datalist?token={FINMIND_TOKEN}"
    
    try:
        resp = requests.get(url)
        all_datasets = resp.json().get("data", [])
        option_datasets = [d for d in all_datasets if "Option" in d]
        print(f"Found {len(option_datasets)} option-related datasets:")
        for d in option_datasets:
            print(f" - {d}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    search_datasets()
