import os, requests, sys
from pathlib import Path
from dotenv import load_dotenv

# ── 物理基準定位 ──
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

def test_raw_supply_chain():
    token = os.getenv("FINMIND_TOKEN")
    
    print("\n" + "🔍" * 20)
    print("Quantum Finance: 供應鏈原生通訊探測 (Raw Sense)")
    print("🔍" * 20)
    
    # 1. 物理 Token 檢查
    print(f"\n[1. 物理層稽核]")
    if not token:
        print("❌ 警告: .env 中未偵測到 FINMIND_TOKEN！")
    else:
        # 僅顯示前 5 碼與末 5 碼以示安全與主權
        masked_token = f"{token[:5]}...{token[-5:]}" if len(token) > 10 else "***"
        print(f"✅ 偵測到 Token: {masked_token}")
        print(f"📏 Token 長度: {len(token)}")

    # 2. 原始 API 調用 (User Info)
    print(f"\n[2. 協議層稽核 - /user_info]")
    # 修正端點: 根據文檔，配額查詢位於 api.web 子域名下
    url_info = "https://api.web.finmindtrade.com/v2/user_info"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    
    try:
        res = requests.get(url_info, headers=headers, timeout=10)
        print(f"📡 HTTP 狀態碼: {res.status_code}")
        print(f"📦 原始 JSON 回傳: {res.text}")
        
        # 結構解析測試
        data = res.json()
        if "user_id" in data:
            print(f"✨ 成功感測到 user_id: {data.get('user_id')}")
        else:
            print("⚠️ 注意: 回傳中缺失 user_id 鍵值。")
            
    except Exception as e:
        print(f"❌ 通訊崩潰: {e}")

    # 3. 核心模組對比 (FinMindClient)
    print(f"\n[3. 封裝層稽核 - FinMindClient]")
    sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
    try:
        from core.finmind_client import FinMindClient
        client = FinMindClient()
        client_res = client.get_user_info()
        print(f"🛠️ FinMindClient 回傳對象: {client_res}")
        
        if client_res == data:
            print("✅ 封裝層與原始通訊 100% 對齊 (Sovereign Matched)")
        else:
            print("❌ 偏差偵測: 封裝解析邏輯與原始回傳不一致！")
    except Exception as e:
        print(f"❌ 核心導入失敗: {e}")

    print("\n" + "─" * 40)
    print("探測完成。請根據以上「原始數據」判定認證失敗之主因。")
    print("─" * 40 + "\n")

if __name__ == "__main__":
    test_raw_supply_chain()
