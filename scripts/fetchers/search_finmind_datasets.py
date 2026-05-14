"""
search_finmind_datasets.py — FinMind 資料集搜尋工具 (v3.2 核心模組升級版)
================================================================================
v3.2 改進：
  ★ 導入 `core.path_setup` 統一處理路徑，確保跨環境執行的穩定性。
  ★ 統一 Token 傳遞規範：全面改用 `Authorization: Bearer` 標頭，符合 FinMind v4 安全標準。
  ★ 支援動態搜尋：導入 `argparse`，可透過 `--keyword` 參數自由搜尋任何主題的資料集。
  ★ 標準化輸出：使用 `logging` 模組取代 `print`，提升日誌層級管理與閱讀體驗。

執行範例：
    # 搜尋包含 "Option" (選擇權) 的資料集 (不區分大小寫)
    python scripts/fetchers/search_finmind_datasets.py --keyword Option
    
    # 搜尋包含 "Stock" (股票) 的資料集
    python scripts/fetchers/search_finmind_datasets.py --keyword stock
    
    # 搜尋總經相關資料集
    python scripts/fetchers/search_finmind_datasets.py -k Taiwan
    
    # 列出所有可用的 FinMind 資料集 (顯示部分預覽)
    python scripts/fetchers/search_finmind_datasets.py
"""

from __future__ import annotations

import sys
import logging
import argparse
import requests
from pathlib import Path

# ── 1. 統一的環境與路徑設定 (path_setup v2.0) ──
_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

from core.path_setup import ensure_scripts_on_path
ensure_scripts_on_path(__file__)

# ── 2. 引入核心模組 ──
from core.finmind_client import FINMIND_TOKEN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

FINMIND_DATALIST_URL = "https://api.finmindtrade.com/api/v4/datalist"


def search_datasets(keyword: str | None = None) -> None:
    """
    向 FinMind 查詢所有可用的 dataset，並可選擇性以關鍵字過濾。
    """
    logger.info("正在向 FinMind API 請求資料集清單...")
    
    # v4 標準：使用 HTTP Header 傳遞 Bearer Token
    headers = {}
    if FINMIND_TOKEN:
        headers["Authorization"] = f"Bearer {FINMIND_TOKEN}"
    
    try:
        resp = requests.get(FINMIND_DATALIST_URL, headers=headers, timeout=10)
        resp.raise_for_status()
        
        data = resp.json()
        if data.get("msg") != "success":
            logger.error(f"API 回傳錯誤：{data.get('msg')}")
            return
            
        all_datasets = data.get("data", [])
        logger.info(f"✅ 成功獲取！FinMind 目前共提供 {len(all_datasets)} 個資料集。")
        
        # 關鍵字過濾邏輯 (不區分大小寫)
        if keyword:
            filtered_datasets = [d for d in all_datasets if keyword.lower() in d.lower()]
            logger.info(f"🔍 包含關鍵字 '{keyword}' 的資料集共有 {len(filtered_datasets)} 個：")
            for d in filtered_datasets:
                logger.info(f"  - {d}")
        else:
            logger.info("📝 列出所有資料集 (前 30 筆作範例，如需查詢特定目標請使用 --keyword)：")
            for d in all_datasets[:30]:
                logger.info(f"  - {d}")
            if len(all_datasets) > 30:
                logger.info(f"  ... (還有 {len(all_datasets) - 30} 個，請加上 -k 參數縮小搜尋範圍)")
                
    except requests.exceptions.RequestException as e:
        logger.error(f"連線 FinMind API 發生錯誤：{e}")


def main():
    p = argparse.ArgumentParser(description="FinMind 資料集搜尋工具 (v3.2 升級版)")
    p.add_argument("--keyword", "-k", type=str, default=None, help="搜尋資料集的關鍵字 (例如 Option, Stock)")
    args = p.parse_args()

    search_datasets(keyword=args.keyword)


if __name__ == "__main__":
    main()