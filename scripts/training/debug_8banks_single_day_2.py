import os
import requests
from dotenv import load_dotenv

load_dotenv()
token = os.environ.get("FINMIND_TOKEN")

url = "https://api.finmindtrade.com/api/v4/data"
params = {
    "dataset": "TaiwanStockGovernmentBankBuySell",
    "start_date": "2025-07-02",
    "end_date": "2025-07-02",
    "token": token
}
resp = requests.get(url, params=params)
print(f"Status Code: {resp.status_code}")
try:
    data = resp.json()
    print(f"Data count: {len(data.get('data', []))}")
except:
    print(f"Response: {resp.text}")
