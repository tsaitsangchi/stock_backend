import os
import requests
from dotenv import load_dotenv

load_dotenv()
token = os.environ.get("FINMIND_TOKEN")

url = "https://api.finmindtrade.com/api/v4/data"
params = {
    "dataset": "TaiwanStockGovernmentBankBuySell",
    "start_date": "2025-07-01",
    "end_date": "2025-07-01", # Try just one day
    "token": token
}
resp = requests.get(url, params=params)
print(f"Status Code: {resp.status_code}")
try:
    data = resp.json()
    print(f"JSON Response: {data}")
except:
    print(f"Text Response: {resp.text}")
