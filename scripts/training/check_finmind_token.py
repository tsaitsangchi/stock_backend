import os
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# 加載環境變數
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

TOKEN = os.environ.get("FINMIND_TOKEN", "")

def check_token():
    print("="*50)
    print("🔍 FinMind Token 權限診斷工具")
    print("="*50)
    
    if not TOKEN:
        print("❌ [錯誤] 未偵測到 FINMIND_TOKEN。請檢查 scripts/.env 檔案。")
        return

    print(f"📡 使用 Token: {TOKEN[:5]}...{TOKEN[-5:] if len(TOKEN)>10 else ''}")
    
    # 1. 檢查帳號基本資訊與使用量
    user_url = f"https://api.finmindtrade.com/api/v4/user/info?token={TOKEN}"
    try:
        res = requests.get(user_url)
        data = res.json()
        if data.get("msg") == "success":
            print(f"✅ [帳號] 狀態正常")
            print(f"   · 剩餘請求次數: {data.get('api_request_limit', 'Unknown')}")
        else:
            print(f"❌ [帳號] API 回報錯誤: {data.get('msg')}")
    except Exception as e:
        print(f"❌ [帳號] 連線失敗: {e}")

    # 2. 測試不同資料集的權限 (以 2330 為例)
    datasets = {
        "TaiwanStockPrice": "台股日收盤價 (基礎)",
        "TaiwanStockInstitutionalInvestorsBuySell": "法人買賣超 (進階籌碼)",
        "TaiwanStockMarginPurchaseShortSale": "融資融券 (信用交易)",
        "TaiwanStockShareholding": "股權分佈 (大戶持股)"
    }
    
    print("\n📊 資料集權限測試:")
    for dataset, desc in datasets.items():
        test_url = f"https://api.finmindtrade.com/api/v4/data?dataset={dataset}&data_id=2330&start_date=2024-04-01&token={TOKEN}"
        try:
            r = requests.get(test_url)
            if r.status_code == 200:
                print(f"   ✅ {desc:<15} : 正常 (200 OK)")
            elif r.status_code == 403:
                print(f"   🚫 {desc:<15} : 拒絕存取 (403 Forbidden) - 可能需要付費或 Token 等級不足")
            else:
                print(f"   ⚠️ {desc:<15} : 異常 ({r.status_code}) - {r.text[:50]}")
        except Exception as e:
            print(f"   ❌ {desc:<15} : 請求失敗 ({e})")

    print("\n" + "="*50)
    print("💡 建議：若看到 403 Forbidden，請登入 FinMind 官網確認您的訂閱方案")
    print("   或是更換一個有效的 Token。")
    print("="*50)

if __name__ == "__main__":
    check_token()
