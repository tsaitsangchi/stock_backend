"""
check_finmind_quota.py — FinMind API 配額與帳戶狀態診斷工具 (Quantum Finance v5.1 Edition)
================================================================================
v3.1 改進：
  ★ 統一 Token 傳遞規範：全面改用 `Authorization: Bearer` 標頭，符合 FinMind v4 API 標準。
  ★ 整合核心模組：使用 `core.finmind_client` 的 `check_api_quota` 進行標準化查詢。
  ★ 帳戶等級識別：根據 `api_request_limit` 自動判定帳戶等級（免費、個人版、專業版、機構版）。
  ★ 視覺化呈現：提供配額使用進度條與重置倒數提醒。
  ★ 支援強制重新整理：繞過 60 秒快取直接向 API 索取最新數據。

執行範例：
    # 執行基本檢查 (使用 60 秒快取，適合高頻呼叫觀測)
    python scripts/fetchers/check_finmind_quota.py
    
    # 強制重新整理 (繞過快取，直接發送請求更新配額)
    python scripts/fetchers/check_finmind_quota.py --force
    
    # 顯示完整 API 回應內容 (除配額外，印出原始 user_info JSON 資料)
    python scripts/fetchers/check_finmind_quota.py --verbose
    
    # 強制重新整理並顯示完整 JSON
    python scripts/fetchers/check_finmind_quota.py --force --verbose
"""

from __future__ import annotations

import sys
import argparse
import logging
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path

# ── 1. 統一的環境與路徑設定 (path_setup v2.0) ──
_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

from core.path_setup import ensure_scripts_on_path
ensure_scripts_on_path(__file__)

# ── 2. 引入核心模組 ──
from core.finmind_client import check_api_quota, USER_INFO_URL, FINMIND_TOKEN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def print_progress_bar(used: int, limit: int, length: int = 40) -> str:
    """繪製視覺化進度條。"""
    if limit <= 0:
        return "N/A"
    
    percent = (used / limit)
    filled_length = int(length * percent)
    bar = "█" * filled_length + "░" * (length - filled_length)
    
    # 顏色控制：綠(安全) -> 黃(警告) -> 紅(危險)
    color_code = "\033[92m" # Green
    if percent > 0.7:
        color_code = "\033[93m" # Yellow
    if percent > 0.9:
        color_code = "\033[91m" # Red
    
    reset_code = "\033[0m"
    return f"{color_code}|{bar}| {percent*100:.1f}%{reset_code}"


def identify_tier(limit: int) -> str:
    """根據限制次數判定帳戶等級。"""
    if limit <= 0: return "未知"
    if limit <= 600: return "免費版 (Free)"
    if limit <= 6000: return "個人版 (Personal / Premium)"
    if limit <= 30000: return "專業版 (Pro)"
    return "機構版 (Enterprise)"


def main():
    p = argparse.ArgumentParser(description="FinMind API 配額檢查與診斷工具 v3.1")
    p.add_argument("--force", action="store_true", help="強制重新整理 (繞過 60 秒快取)")
    p.add_argument("--verbose", action="store_true", help="顯示完整 API 回應內容")
    args = p.parse_args()

    logger.info("正在連線至 FinMind 獲取帳戶狀態...")
    
    try:
        # 1. 呼叫核心函式獲取基本配額
        used, limit = check_api_quota(force=args.force)
        
        # 2. 如果需要 verbose，手動多叫一次 user_info 獲取更多資訊
        extra_info = {}
        if args.verbose:
            # 修正：全面改用 HTTP Header 傳遞 Token，符合 v4 安全規範
            headers = {"Authorization": f"Bearer {FINMIND_TOKEN}"}
            res = requests.get(USER_INFO_URL, headers=headers, timeout=10)
            res.raise_for_status()
            extra_info = res.json()

        # 3. 輸出美化報表
        print("\n" + "=" * 65)
        print(f"    FinMind API 帳戶權益報表 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
        print("=" * 65)
        
        tier = identify_tier(limit)
        print(f" 帳戶等級 : {tier}")
        print(f" 配額限制 : {limit:,} 次 / 小時")
        print(f" 目前已用 : {used:,} 次")
        print(f" 剩餘可用 : {max(0, limit - used):,} 次")
        print(f" 使用進度 : {print_progress_bar(used, limit)}")
        
        # 4. 計算重置時間 (每小時整點)
        now = datetime.now()
        next_reset = (now + timedelta(hours=1)).replace(minute=0, second=5, microsecond=0)
        remaining = (next_reset - now).total_seconds()
        print(f" 重置倒數 : {int(remaining // 60)} 分 {int(remaining % 60)} 秒")
        
        # 5. 詳細模式輸出
        if args.verbose and extra_info:
            print("-" * 65)
            print(" 詳細資訊 (User Info Raw Response):")
            print(json.dumps(extra_info, indent=2, ensure_ascii=False))
            
        print("=" * 65 + "\n")
        
        # 6. 狀態警示
        if used >= limit > 0:
            logger.warning("⚠️ 偵測到配額已耗盡！部分依賴自動等待的抓取腳本將會進入休眠模式直到下一整點。")
        elif used > limit * 0.9:
            logger.warning("⚠️ 配額即將用盡 (已超過 90%)，請謹慎執行全量抓取。")
            
    except Exception as e:
        logger.error(f"帳戶狀態檢查失敗: {e}")


if __name__ == "__main__":
    main()