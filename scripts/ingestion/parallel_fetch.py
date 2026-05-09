"""
parallel_fetch.py v5.5.2 (Trinity Core Final)
================================================================================
並行資料抓取引擎 — 混合模式日誌實作版
負責調度多個抓取器並執行任務匯總，支援真實執行緒池。

核心功能：
  · 並行調度       ─ 真實調用傳入的任務函式 (task_func)。
  · 指標歸集       ─ 統計總成功數與總抓取筆數。
  · 混合日誌紀錄     ─ 對接 pipeline_execution_log (Category: ingestion)。
"""

import sys
import logging
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── 系統路徑修復 ──
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
    from config import TIER_1_STOCKS
except ImportError as e:
    print(f"[FATAL] 無法匯入核心配置: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_orchestrator(task_func, stock_ids: list, task_label: str = "batch_fetch", workers: int = 4):
    """
    真正的並行執行調度器。
    :param task_func: 接收單一 stock_id 的函式。
    :param stock_ids: 標的名單。
    """
    t0 = time.monotonic()
    logger.info(f"🚀 [Orchestrator] 啟動並行任務: {task_label} (Targets: {len(stock_ids)})")
    
    total_rows = 0
    success_count = 0
    
    if not task_func:
        logger.warning("⚠️ 未提供 task_func，執行空跑測試。")
        time.sleep(0.3)
        return

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_sid = {executor.submit(task_func, sid): sid for sid in stock_ids}
            for future in as_completed(future_to_sid):
                sid = future_to_sid[future]
                try:
                    rows = future.result()
                    total_rows += (rows if isinstance(rows, int) else 0)
                    success_count += 1
                except Exception as exc:
                    logger.error(f"❌ {sid} 執行失敗: {exc}")
        
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        
        write_pipeline_log(
            task_name=f"parallel_{task_label}",
            stock_id="SYSTEM_WIDE",
            status="success",
            category="ingestion",
            duration_ms=elapsed_ms,
            rows=total_rows,
            err=f"Success: {success_count}/{len(stock_ids)}"
        )
        logger.info(f"✅ {task_label} 並行執行完成，總筆數: {total_rows}")
        
    except Exception as e:
        logger.error(f"❌ 調度器崩潰: {e}")
        write_pipeline_log(f"parallel_{task_label}", "SYSTEM", "failed", "ingestion", 0, 0, str(e))

if __name__ == "__main__":
    # 範例：並行執行全市場基本資訊抓取 (測試用)
    from ingestion.fetch_technical_data import fetch_tech
    run_orchestrator(fetch_tech, TIER_1_STOCKS[:5], "tech_sync_v5.5")