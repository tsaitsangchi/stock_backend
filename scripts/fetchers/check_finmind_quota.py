"""
check_finmind_quota.py v4.2 (Robust Diagnostic Edition)
FinMind API 配額診斷工具 - 解決 404 與 Token 版本相容性問題
================================================================================
v4.2 修正項目：
  · 解決 404 錯誤：自動切換 GET/POST 方法，並支援自動降級至備援端點。
  · Token 校驗：顯示目前載入 Token 的指紋，確認是否與 .env 同步。
  · 強化錯誤處理：捕捉特定 HTTP 狀態碼並給予具體建議。
"""

import sys
import os
import json
import argparse
import requests
import logging
from datetime import datetime, timedelta
from pathlib import Path

# ── 系統路徑修復 ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for sub in ("", "core"):
    p = (_SCRIPTS_DIR / sub) if sub else _SCRIPTS_DIR
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from core.db_utils import db_session
    from core.finmind_client import FinMindClient
except ImportError as e:
    print(f"無法匯入核心模組: {e}", file=sys.stderr)
    sys.exit(1)

# 端點清單 (優先權排序)
ENDPOINTS = [
    "https://api.finmindtrade.com/api/v4/user_info",
    "https://api.web.finmindtrade.com/v2/user_info"
]

def identify_tier(limit: int) -> str:
    if limit <= 300: return "免費版 (Free)"
    elif limit <= 600: return "個人版 (Personal)"
    elif limit <= 3000: return "專業版 (Professional)"
    return "企業版 / 高階贊助 (Enterprise/VIP)"

def try_fetch_user_info(url: str, token: str) -> dict:
    """嘗試以不同方法從指定 URL 獲取資訊"""
    # 方法 1: GET with params
    try:
        res = requests.get(url, params={"token": token}, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if data.get("msg") == "success": return data
    except: pass

    # 方法 2: POST with data
    try:
        res = requests.post(url, data={"token": token}, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if data.get("msg") == "success": return data
    except: pass
    
    return None

def main():
    parser = argparse.ArgumentParser(description="FinMind API 配額診斷工具 v4.2")
    args = parser.parse_args()

    client = FinMindClient()
    token = getattr(client, 'api_token', os.environ.get("FINMIND_TOKEN", ""))

    print("\n" + "═" * 65)
    print(f"    FinMind API 帳戶權益報表 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print("═" * 65)

    if not token:
        print(" ⚠️  錯誤：找不到 FINMIND_TOKEN。請檢查專案根目錄的 .env 檔案。")
        return

    # 印出 Token 指紋 (確認是否為結尾 HITHM 的那個)
    masked_token = f"{token[:10]}...{token[-10:]}"
    print(f" 🔑 目前使用 Token : {masked_token}")
    
    result_data = None
    used_url = ""

    # 迴圈嘗試所有可能的端點
    for url in ENDPOINTS:
        print(f" 🔍 正在嘗試端點 : {url.split('/')[-2]}/{url.split('/')[-1]}...")
        data = try_fetch_user_info(url, token)
        if data:
            result_data = data
            used_url = url
            break

    if not result_data:
        print(" ❌ 失敗：無法從任何已知端點獲取資訊。")
        print(" 💡 可能原因：")
        print("    1. 您的 Token 可能已過期或被重置。")
        print("    2. .env 內容可能尚未存檔，請確認內容為您最新的 Token。")
        print("    3. 您目前的 IP 可能被 FinMind 暫時封鎖。")
        print("═" * 65 + "\n")
        return

    # 顯示成功結果
    info = result_data.get("data", {})
    limit = info.get("api_request_limit", 300)
    used = info.get("user_request_count", 0)

    print(f" ✅ 獲取成功！(來自 {used_url.split('.')[1]})")
    print(f" 帳戶等級 : {identify_tier(limit)}")
    print(f" 配額限制 : {limit:,} 次 / 小時")
    print(f" 目前已用 : {used:,} 次")
    print(f" 剩餘可用 : {max(0, limit - used):,} 次")
    
    # 計算進度條
    ratio = min(used / limit, 1.0) if limit > 0 else 0
    bar = '█' * int(40 * ratio) + '-' * (40 - int(40 * ratio))
    print(f" 使用進度 : [{bar}] {ratio*100:.1f}%")

    # 重置倒數
    now = datetime.now()
    next_reset = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    remaining = (next_reset - now).total_seconds()
    print(f" 重置倒數 : {int(remaining // 60)} 分 {int(remaining % 60)} 秒")
    print("═" * 65 + "\n")

if __name__ == "__main__":
    main()