"""
search_finmind_datasets.py v5.1 (Trinity Core Edition)
================================================================================
FinMind 資料集搜尋工具 — 完美對接 core/ 五大核心模組
此模組為維運輔助工具，協助開發者快速查詢 FinMind API 目前支援的所有資料集清單。

核心功能：
  · 全域搜尋         ─ 呼叫 datalist 接口，獲取 FinMind 完整 100+ 資料集清單。
  · 關鍵字過濾       ─ 支援大小寫不敏感的關鍵字過濾，快速定位目標 Dataset。
  · 權限自動帶入     ─ 對接 FinMindClient v5.1，自動從環境變數載入 Bearer Token。

對接核心模組 (scripts/core/)：
  · finmind_client v5.1      ─ Singleton + 自動 Token 管理
  · path_setup v3.0          ─ 跨環境路徑自動導引

修訂歷程：
  v5.1 (2026-05-09):
    - [修復] 解決 ImportError: cannot import name 'FINMIND_TOKEN' 問題。
    - [整合] 全面對接 FinMindClient 單例模式獲取 Token 與連線配置。
    - [路徑] 對接 path_setup v3.0 標準。

執行範例：
    # 搜尋包含 "Option" (選擇權) 的資料集
    python scripts/fetchers/search_finmind_datasets.py --keyword Option
    
    # 列出所有可用的 FinMind 資料集
    python scripts/fetchers/search_finmind_datasets.py
"""

import sys
import logging
import argparse
import requests
from pathlib import Path
from typing import Optional

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
    from core.finmind_client import FinMindClient
except ImportError as e:
    print(f"[FATAL] 無法匯入 core 模組: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FINMIND_DATALIST_URL = "https://api.finmindtrade.com/api/v4/datalist"

def search_datasets(keyword: Optional[str] = None) -> None:
    """向 FinMind 查詢所有可用的 dataset，並可選擇性以關鍵字過濾。"""
    client = FinMindClient()
    logger.info("正在向 FinMind API 請求資料集清單...")
    
    headers = {}
    if client.api_token:
        headers["Authorization"] = f"Bearer {client.api_token}"
    
    try:
        resp = requests.get(FINMIND_DATALIST_URL, headers=headers, timeout=15)
        resp.raise_for_status()
        
        data = resp.json()
        if data.get("msg") != "success":
            logger.error(f"API 回傳錯誤：{data.get('msg')}")
            return
            
        all_datasets = data.get("data", [])
        logger.info(f"✅ 成功獲取！FinMind 目前共提供 {len(all_datasets)} 個資料集。")
        
        if keyword:
            keyword_lower = keyword.lower()
            filtered = [d for d in all_datasets if keyword_lower in d.lower()]
            logger.info(f"🔍 包含關鍵字 '{keyword}' 的資料集共有 {len(filtered)} 個：")
            for d in filtered:
                logger.info(f"  - {d}")
        else:
            logger.info("📝 列出前 30 個資料集範例：")
            for d in all_datasets[:30]:
                logger.info(f"  - {d}")
            if len(all_datasets) > 30:
                logger.info(f"  ... (還有 {len(all_datasets) - 30} 個，請使用 --keyword 縮小搜尋範圍)")
                
    except Exception as e:
        logger.error(f"連線 FinMind API 發生錯誤：{e}")

def main():
    parser = argparse.ArgumentParser(description="FinMind 資料集搜尋工具 v5.1 (Trinity Core Edition)")
    parser.add_argument("--keyword", "-k", type=str, default=None, help="搜尋關鍵字")
    args = parser.parse_args()

    search_datasets(keyword=args.keyword)

if __name__ == "__main__":
    main()