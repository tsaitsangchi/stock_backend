"""
check_data_integrity.py v7.1 (Trinity Core Master Audit)
================================================================================
數據完整性稽核工具 — 高階資產可信度引擎
負責檢查資料庫內的數據密度與同步進度，並將結果紀錄於 Hybrid Logging 體系。

修訂歷程：
  v7.1 (2026-05-10):
    - [核心] 實作「混合模式日誌」：統一 pipeline_execution_log 與專門 data_audit_log 分類紀錄。
    - [優化] 調整輸出格式，預設僅顯示異常標的，提升稽核效率。
  v7.0 (2026-05-10):
    - [核心] 建立基礎數據完整性掃描框架。

【執行範例說明 — 數據稽核矩陣】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令                                               │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [全域數據健康檢查]        │ $ python scripts/maintenance/check_data_integrity.py     │
│ 2. [指定標的深度稽核]        │ $ python scripts/maintenance/check_data_integrity.py --stock_id 2330 │
│ 3. [顯示所有標的狀態]        │ $ python scripts/maintenance/check_data_integrity.py --verbose  │
└──────────────────────────────┴────────────────────────────────────────────────────────┘
================================================================================
"""

import sys
import logging
import time
import argparse
from datetime import datetime, date
from pathlib import Path

# ── 系統路徑修復 (v3.1) ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core", "ingestion", "pipeline"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import db_transaction, get_db_stock_ids, get_latest_date, write_pipeline_log
except ImportError as e:
    print(f"[FATAL] 無法匯入核心配置: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def update_data_audit_log(stock_id: str, missing_days: int, health_score: float, status: str):
    """
    更新專門的分類紀錄表 data_audit_log。
    """
    sql = """
        INSERT INTO data_audit_log (stock_id, missing_days, health_score, status, last_checked_at)
        VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (stock_id) DO UPDATE SET
            missing_days = EXCLUDED.missing_days,
            health_score = EXCLUDED.health_score,
            status = EXCLUDED.status,
            last_checked_at = CURRENT_TIMESTAMP;
    """
    try:
        with db_transaction() as cur:
            cur.execute(sql, (stock_id, missing_days, health_score, status))
    except Exception as e:
        logger.error(f"❌ [Log] 更新 data_audit_log 失敗 ({stock_id}): {e}")

def audit_integrity(stock_id: str = None, verbose: bool = False):
    """
    執行數據完整性稽核報告。
    """
    t_master = time.monotonic()
    today = date.today()
    
    METRICS = {
        "Price": "stock_price",
        "Chip": "institutional_investors_buy_sell",
        "Fundamental": "financial_statements",
        "Revenue": "month_revenue"
    }

    target_ids = [stock_id] if stock_id else get_db_stock_ids()
    logger.info(f"🔍 [Audit] 啟動數據完整性稽核 (目標: {len(target_ids)} 檔標的)...")

    total_issues = 0
    
    print("\n" + "="*85)
    print(f"{'Stock ID':<10} | {'Metric':<12} | {'Last Date':<12} | {'Status':<15}")
    print("-" * 85)

    for sid in target_ids:
        t_unit = time.monotonic()
        stock_issues = 0
        max_lag = 0
        dimensions_ok = 0
        
        results = []
        for dim_name, table_name in METRICS.items():
            last_dt_str = get_latest_date(table_name, sid)
            
            status = "✅ OK"
            last_dt_display = "None"
            days_diff = 999 # Default for missing
            
            if not last_dt_str:
                status = "❌ MISSING"
                stock_issues += 1
            else:
                last_dt = datetime.strptime(last_dt_str, "%Y-%m-%d").date()
                last_dt_display = last_dt_str
                days_diff = (today - last_dt).days
                
                # 判斷邏輯
                if dim_name == "Price" and days_diff > 3:
                    status = f"⚠️ LAG {days_diff}d"
                    stock_issues += 1
                elif dim_name == "Revenue" and days_diff > 45:
                    status = f"⚠️ LAG {days_diff}d"
                    stock_issues += 1
                else:
                    dimensions_ok += 1
                
                max_lag = max(max_lag, days_diff)

            results.append((dim_name, last_dt_display, status))

        # 寫入分類日誌
        health_score = (dimensions_ok / len(METRICS)) * 100
        overall_status = "Healthy" if stock_issues == 0 else ("Warning" if health_score >= 50 else "Critical")
        update_data_audit_log(sid, max_lag if max_lag < 999 else 0, health_score, overall_status)

        # 輸出處理
        if verbose or stock_issues > 0:
            for dim, dt, stat in results:
                print(f"{sid:<10} | {dim:<12} | {dt:<12} | {stat:<15}")
            if verbose: print("-" * 85)

        total_issues += stock_issues

    elapsed_ms = int((time.monotonic() - t_master) * 1000)
    
    # 🔴 寫入生命週期日誌 (Category: maintenance)
    write_pipeline_log(
        task_name="data_integrity_audit_master",
        stock_id="SYSTEM" if not stock_id else stock_id,
        status="success" if total_issues == 0 else "warning",
        category="maintenance",
        duration_ms=elapsed_ms,
        rows=total_issues
    )
    
    print("="*85)
    logger.info(f"🏆 稽核完成！發現 {total_issues} 項異常，紀錄已同步至 Hybrid Logs，耗時: {elapsed_ms}ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trinity Data Integrity Auditor")
    parser.add_argument("--stock_id", type=str, help="指定稽核標的")
    parser.add_argument("--verbose", action="store_true", help="顯示所有標的狀態 (預設僅顯示異常)")
    args = parser.parse_args()
    
    audit_integrity(stock_id=args.stock_id, verbose=args.verbose)
