import requests

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
    if resp.status_code == 200:
        data = resp.json().get("data", [])
        if data:
            print(f"Keys: {data[0].keys()}")
            print(f"First 10 records: {data[:10]}")
        else:
            print("No data")
    else:
        print(f"Response: {resp.text}")

print("Testing TaiwanStockTradingDailyReport for 2330")
test_api("TaiwanStockTradingDailyReport", "2330", "2024-04-01", "2024-04-01")
