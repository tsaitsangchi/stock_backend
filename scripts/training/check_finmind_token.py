import os
import requests
from dotenv import load_dotenv

load_dotenv()
token = os.environ.get("FINMIND_TOKEN")
print(f"Token: {token[:5]}...{token[-5:] if token else ''}")

url = "https://api.web.finmindtrade.com/v2/user_info"
headers = {"Authorization": f"Bearer {token}"}
resp = requests.get(url, headers=headers)
print(f"Status Code: {resp.status_code}")
try:
    print(f"User Info: {resp.json()}")
except:
    print(f"Response Text: {resp.text}")
