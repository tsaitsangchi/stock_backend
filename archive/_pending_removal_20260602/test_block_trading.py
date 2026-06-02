import sys
import os
from pathlib import Path

# 加入 scripts 目錄
_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

from core.finmind_client import finmind_get

import requests
import json

def test_raw():
    url = "https://api.finmindtrade.com/api/v4/data"
    token = os.environ.get("FINMIND_TOKEN")
    
    datasets = ["TaiwanStockBlockTradingDailyReport", "TaiwanStockBlockTrading"]
    params_list = [
        {"dataset": "TaiwanStockBlockTradingDailyReport", "start_date": "2024-03-04", "end_date": "2024-03-04"},
        {"dataset": "TaiwanStockBlockTrade", "data_id": "2330", "start_date": "2024-03-04", "end_date": "2024-03-10"}
    ]
    
    for p in params_list:
        if token: p["token"] = token
        print(f"Testing {p['dataset']} with {p}...")
        resp = requests.get(url, params=p)
        print(f"  Status: {resp.status_code}")
        try:
            print(f"  Body: {resp.json().get('msg', resp.text)}")
        except:
            print(f"  Body: {resp.text}")

if __name__ == "__main__":
    test_raw()
