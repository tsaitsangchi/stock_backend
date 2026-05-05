import requests
import json
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.finmind_client import FINMIND_TOKEN, FINMIND_USER_INFO_URL

def check_account_status():
    headers = {"Authorization": f"Bearer {FINMIND_TOKEN}"}
    try:
        print("Checking FinMind user info...")
        resp = requests.get(FINMIND_USER_INFO_URL, headers=headers)
        data = resp.json()
        print(json.dumps(data, indent=2, ensure_ascii=False))
        
        # 檢查配額
        api_count = data.get("api_count", 0)
        api_limit = data.get("api_limit", 600)
        print(f"\nAPI Usage: {api_count} / {api_limit}")
        if api_count >= api_limit:
            print("⚠️ 配額已耗盡！")
        else:
            print("✅ 配額尚充足。")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_account_status()
