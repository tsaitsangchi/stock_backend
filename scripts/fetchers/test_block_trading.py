"""
test_block_trading.py v5.1 (Trinity Core Edition)
================================================================================
鉅額交易資料測試與抓取工具 — 完美對接 core/ 五大核心模組
此模組負責驗證與抓取 FinMind 的鉅額交易資料，支援日報表與個股明細兩種模式。

核心功能：
  · 權限偵測         ─ 自動識別 400 (權限不足) 錯誤，並轉化為可稽核的日誌紀錄。
  · 斷路器保護       ─ 對接 v5.1 核心，業務級權限錯誤不會觸發全域熔斷。
  · 監控集成         ─ 失敗原因將完整記錄於 fetch_log，供資料完整性審計使用。

對接核心模組 (scripts/core/)：
  · db_utils v4.6            ─ 連線池 + 事務原子性 + 筆數追蹤 (rows_inserted)
  · finmind_client v5.1      ─ Singleton + 智慧斷路器 (排除業務級錯誤)
  · path_setup v3.0          ─ 跨環境路徑自動導引

修訂歷程：
  v5.1 (2026-05-09):
    - [核心] 移除原始 requests 呼叫，全面改裝 FinMindClient()。
    - [監控] 導入 write_fetch_log，將「權限不足」等失敗原因落盤。
    - [穩定] 對接 v5.1 智慧斷路器，確保單一接口失敗不影響全系統。

執行範例：
    # 測試抓取台積電鉅額交易明細
    python scripts/fetchers/test_block_trading.py
"""

import sys
import logging
import time
from pathlib import Path
from typing import Dict, Any

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
    from core.db_utils import write_fetch_log
    from core.finmind_client import FinMindClient
except ImportError as e:
    print(f"[FATAL] 無法匯入 core 模組: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def test_raw_dataset(dataset: str, data_id: str = None, start: str = "2024-03-04", end: str = "2024-03-04"):
    """
    測試特定資料集的抓取狀況，並紀錄至 fetch_log。
    """
    api = FinMindClient()
    logger.info(f"🚀 正在測試資料集: {dataset} (ID: {data_id or 'N/A'})")
    
    t0 = time.monotonic()
    try:
        data = api.get_data(dataset, data_id, start, end)
        duration_ms = int((time.monotonic() - t0) * 1000)
        
        if data:
            logger.info(f"  ✅ 成功獲取 {len(data)} 筆資料。")
            write_fetch_log(dataset, data_id or "SYSTEM", "success", "test_v5.1", start, end, duration_ms, len(data), None)
        else:
            logger.info("  ⚠️ 請求成功但無資料。")
            write_fetch_log(dataset, data_id or "SYSTEM", "no_new_data", "test_v5.1", start, end, duration_ms, 0, None)
            
    except Exception as e:
        duration_ms = int((time.monotonic() - t0) * 1000)
        err_msg = str(e)
        logger.error(f"  ❌ 抓取失敗: {err_msg}")
        
        # 識別是否為權限問題
        status = "failed"
        if "Please update your user level" in err_msg or "400 Client Error" in err_msg:
            status = "permission_denied"
            logger.warning("  💡 偵測到權限限制：此資料集需要更高階的 FinMind 帳號。")
            
        write_fetch_log(dataset, data_id or "SYSTEM", status, "test_v5.1", start, end, duration_ms, 0, err_msg)

def main():
    logger.info("=" * 70)
    logger.info(f"  Block Trading Tester v5.1  ")
    logger.info("=" * 70)
    
    # 測試 1: 鉅額交易日報
    test_raw_dataset("TaiwanStockBlockTradingDailyReport")
    
    # 測試 2: 個股鉅額交易明細
    test_raw_dataset("TaiwanStockBlockTrade", data_id="2330", start="2024-03-04", end="2024-03-10")

if __name__ == "__main__":
    main()
