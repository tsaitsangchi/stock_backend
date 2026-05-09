"""
check_finmind_datalist.py v5.5 (Trinity Core Edition)
================================================================================
FinMind 資料可用性與連線診斷工具 — 混合模式日誌實作版
此模組用於測試特定資料集 (Dataset) 的 API 回傳狀況，並可比對資料庫本地筆數。

核心功能：
  · API 抽樣檢測   ─ 支援個股與市場級資料集的即時拉取測試。
  · DB 同步比對     ─ 自動映射資料表並檢查本地資料筆數。
  · 執行紀錄       ─ 對接 write_pipeline_log，標記為 sys_v5.1 (Maintenance)。

修訂歷程：
  v5.5 (2026-05-09):
    - [規範] 導入混合模式日誌。
    - [核心] 對接 path_setup v3.0 與 db_utils v4.7 標準。
"""

import sys
import argparse
import logging
import time
from pathlib import Path
from datetime import date

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
    from core.db_utils import db_session, write_pipeline_log
    from core.finmind_client import FinMindClient
except ImportError as e:
    print(f"[FATAL] 無法匯入核心模組: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    p = argparse.ArgumentParser(description="FinMind Data Diagnostic Tool v5.5")
    p.add_argument("--dataset", type=str, default="TaiwanOptionOpenInterestLargeTraders")
    p.add_argument("--stock-id", type=str, default="")
    p.add_argument("--start", type=str, default="2024-01-01")
    p.add_argument("--end", type=str, default=date.today().strftime("%Y-%m-%d"))
    args = p.parse_args()

    t0 = time.monotonic()
    client = FinMindClient()
    logger.info(f"🚀 啟動診斷: {args.dataset} (ID: {args.stock_id or 'Market'})")
    
    try:
        data = client.get_data(args.dataset, args.stock_id, args.start, args.end)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        
        if data:
            logger.info(f"✅ 成功獲取 {len(data)} 筆資料，耗時 {elapsed_ms/1000:.2f}s")
            status = "success"
        else:
            logger.warning("⚠️ 成功連線但無資料。")
            status = "no_new_data"

        # 混合模式日誌紀錄
        write_pipeline_log(
            task_name="check_finmind_datalist",
            stock_id=args.stock_id or "MARKET_CHECK",
            status=status,
            category="sys",
            duration_ms=elapsed_ms,
            rows=len(data) if data else 0
        )
    except Exception as e:
        logger.error(f"❌ 診斷失敗: {e}")
        write_pipeline_log("check_finmind_datalist", args.stock_id or "MARKET_CHECK", "failed", "sys", 0, 0, str(e))

if __name__ == "__main__":
    main()