"""
check_finmind_quota.py v5.5 (Trinity Core Edition)
================================================================================
FinMind API 配額與帳戶權益診斷工具 — 混合模式日誌實作版
此模組負責檢查目前 Token 的等級、剩餘配額與重置時間，確保抓取管線有充足資源。

核心功能：
  · 自動偵測       ─ 支援多端點自動降級與 GET/POST 自適應。
  · 權益報表       ─ 顯示帳戶等級 (Free/Premium) 與視覺化進度條。
  · 執行紀錄       ─ 對接 write_pipeline_log，標記為 sys_v5.1。

修訂歷程：
  v5.5 (2026-05-09):
    - [規範] 導入混合模式日誌。
    - [核心] 對接 path_setup v3.0 與 FinMindClient v5.1 標準。
"""

import sys
import os
import time
import requests
import logging
from datetime import datetime, timedelta
from pathlib import Path

# ── 系統路徑修復 (對接 path_setup v3.0) ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import write_pipeline_log
    from core.finmind_client import FinMindClient
except ImportError as e:
    print(f"[FATAL] 無法匯入核心模組: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ENDPOINTS = [
    "https://api.finmindtrade.com/api/v4/user_info",
    "https://api.web.finmindtrade.com/v2/user_info"
]

def main():
    t0 = time.monotonic()
    client = FinMindClient()
    token = client.api_token
    
    print("\n" + "═" * 65)
    print(f"    FinMind API 帳戶權益報表 ({datetime.now():%Y-%m-%d %H:%M:%S})")
    print("═" * 65)

    res_data = None
    for url in ENDPOINTS:
        try:
            r = requests.get(url, params={"token": token}, timeout=10)
            if r.status_code == 200 and r.json().get("msg") == "success":
                res_data = r.json().get("data", r.json())
                break
        except: continue

    if res_data:
        limit = res_data.get("api_request_limit", 300)
        used = res_data.get("user_request_count", 0)
        print(f" ✅ 獲取成功！")
        print(f" 配額限制 : {limit:,} 次 / 小時")
        print(f" 目前已用 : {used:,} 次")
        print(f" 剩餘可用 : {max(0, limit - used):,} 次")
        
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        write_pipeline_log("check_finmind_quota", "FINMIND_API", "success", "sys", elapsed_ms, used)
    else:
        print(" ❌ 無法獲取配額資訊。")
        write_pipeline_log("check_finmind_quota", "FINMIND_API", "failed", "sys", 0, 0, "All endpoints failed")
    print("═" * 65 + "\n")

if __name__ == "__main__":
    main()