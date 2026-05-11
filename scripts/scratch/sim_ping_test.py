
import requests, time

def simulate_ping():
    print("🚀 [模擬實驗] 正在測試 API Ping 邏輯...")
    url = "https://api.finmindtrade.com/api/v4/data"
    start = time.time()
    try:
        # 僅做 HEAD 請求或最小量請求模擬 Ping
        res = requests.head(url, timeout=5)
        latency = int((time.time() - start) * 1000)
        print(f"✅ 模擬成功！狀態碼: {res.status_code}, 延遲: {latency} ms")
        return True
    except Exception as e:
        print(f"❌ 模擬失敗: {e}")
        return False

if __name__ == "__main__":
    simulate_ping()
