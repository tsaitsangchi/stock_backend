import requests
import time

FINMIND_API_URL = "https://api.finmindtrade.com/api/v4/data"
FINMIND_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyNi0wMy0xNCAxODoxNTo1NCIsInVzZXJfaWQiOiJ0c2FpdHNhbmdjaGkiLCJlbWFpbCI6InRzYWl0c2FuZ2NoaUBnbWFpbC5jb20iLCJpcCI6IjIyMC4xMzQuMjYuNzAifQ.muoHEMMLiiRQoxZj7evq-9hclsVRXE3IfLNZWDZ6PQE"

def test_api(dataset, sid, start, end):
    headers = {"Authorization": f"Bearer {FINMIND_TOKEN}"}
    params = {
        "dataset": dataset,
        "data_id": sid,
        "start_date": start,
        "end_date": end
    }
    resp = requests.get(FINMIND_API_URL, headers=headers, params=params)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.text}")

print("Testing 2016-01-01 to 2016-01-10")
test_api("TaiwanStockShareholdingByBroker", "2330", "2016-01-01", "2016-01-10")

print("\nTesting 2024-01-01 to 2024-01-10")
test_api("TaiwanStockShareholdingByBroker", "2330", "2024-01-01", "2024-01-10")
