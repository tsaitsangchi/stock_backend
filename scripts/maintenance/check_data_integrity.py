"""
check_data_integrity.py v7.2 (Quantum Finance Edition)
================================================================================
數據完整性稽核工具 — 高階資產可信度引擎 (Quantum v5.2 標準)
負責檢查資料庫內的數據密度與同步進度，並將結果紀錄於混合日誌體系。

修訂歷程：
  v7.2 (2026-05-11): [標準化] 導入 Quantum 標準檔頭、生命週期監測與數據健康儀表板。
  v7.1 (2026-05-10): [優化] 調整輸出格式，預設僅顯示異常標的。

執行範例 (Comprehensive Usage Examples):
  1. [全域數據稽核] 掃描所有核心標的的數據同步狀態:
     python scripts/maintenance/check_data_integrity.py

  2. [指定標的稽核] 深度掃描台積電(2330)的所有維度:
     python scripts/maintenance/check_data_integrity.py --stock_id 2330

  3. [完整模式] 顯示所有標的狀態 (包含正常者):
     python scripts/maintenance/check_data_integrity.py --verbose

  4. [稽核結果查閱] 查看最近一次完整性掃描發現的異常數 (SQL):
     SELECT rows_processed as issues_count FROM pipeline_execution_log WHERE task_name = 'data_integrity_audit' ORDER BY created_at DESC LIMIT 1;
================================================================================
"""
import sys, logging, time, argparse
from datetime import datetime, date
from pathlib import Path

# ── 終極路徑自癒 Bootstrap ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent if _THIS_DIR.name != "scripts" else _THIS_DIR
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_SCRIPTS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR.parent))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import db_transaction, get_db_stock_ids, get_latest_date, record_lifecycle, write_data_audit_log
except ImportError as e:
    print(f"[FATAL] 無法匯入核心配置: {e}", file=sys.stderr)
    sys.exit(1)

logger = logging.getLogger(__name__)

def show_integrity_dashboard(stats: dict):
    """稽核任務後的健康儀表板回報。"""
    print("\n" + "="*70)
    print("🔍 Quantum Finance: 數據完整性稽核報告 (v7.2)")
    print("="*70)
    print(f"✅ 稽核狀態  : 執行完成")
    print(f"📊 掃描標的  : {stats['total_stocks']} 檔")
    print(f"⚠️ 發現異常  : {stats['total_issues']} 項資料缺漏")
    print(f"🌡️  平均健康度: {stats['avg_health']:.1f}%")
    
    if stats['total_issues'] > 0:
        print("-" * 70)
        print(f"📢 待辦提醒：請檢查 ingestion 體系是否正常運行，或使用 parallel_ingestion 回補。")
    
    print("-" * 70)
    print("📝 日誌同步: pipeline_execution_log & data_audit_log")
    print("="*70 + "\n")

def audit_integrity(stock_id: str = None, verbose: bool = False):
    """執行全系統或特定標的的數據完整性稽核。"""
    t0 = time.monotonic()
    today = date.today()
    
    # 定義稽核維度
    METRICS = {
        "Price": "stock_price",
        "Chip": "institutional_investors_buy_sell",
        "Fundamental": "financial_statements",
        "Revenue": "month_revenue"
    }

    with record_lifecycle("data_integrity_audit", category="maintenance", stock_id=stock_id or "GLOBAL"):
        target_ids = [stock_id] if stock_id else get_db_stock_ids(active_only=True)
        if not target_ids: 
            target_ids = ["2330", "2317", "2454"] # 降級至基礎清單
            
        logger.info(f"🔍 [Audit] 啟動數據完整性稽核 (目標: {len(target_ids)} 檔標的)...")

        total_issues = 0
        sum_health = 0
        
        if verbose or stock_id:
            print("\n" + "-"*85)
            print(f"{'Stock ID':<10} | {'Metric':<12} | {'Last Date':<12} | {'Status':<15}")
            print("-" * 85)

        for sid in target_ids:
            stock_issues = 0
            dimensions_ok = 0
            
            stock_results = []
            for dim_name, table_name in METRICS.items():
                last_dt_str = get_latest_date(table_name, sid)
                
                status = "✅ OK"
                last_dt_display = last_dt_str if last_dt_str else "None"
                
                if not last_dt_str:
                    status = "❌ MISSING"
                    stock_issues += 1
                else:
                    last_dt = datetime.strptime(last_dt_str, "%Y-%m-%d").date()
                    days_diff = (today - last_dt).days
                    
                    # 容錯邏輯
                    if dim_name == "Price" and days_diff > 3:
                        status = f"⚠️ LAG {days_diff}d"; stock_issues += 1
                    elif dim_name == "Revenue" and days_diff > 45:
                        status = f"⚠️ LAG {days_diff}d"; stock_issues += 1
                    else:
                        dimensions_ok += 1
                
                stock_results.append((dim_name, last_dt_display, status))

            health_score = (dimensions_ok / len(METRICS)) * 100
            sum_health += health_score
            total_issues += stock_issues

            if verbose or stock_issues > 0 or stock_id:
                for dim, dt, stat in stock_results:
                    print(f"{sid:<10} | {dim:<12} | {dt:<12} | {stat:<15}")
                if verbose: print("-" * 85)

        # 混合模式：審計紀錄
        write_data_audit_log("integrity_audit", stock_id or "SYSTEM", today.strftime("%Y-%m-%d"), "SCAN", total_issues)
        
        # 顯示儀表板
        stats = {
            "total_stocks": len(target_ids),
            "total_issues": total_issues,
            "avg_health": sum_health / len(target_ids) if target_ids else 0
        }
        show_integrity_dashboard(stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock_id", type=str)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    audit_integrity(stock_id=args.stock_id, verbose=args.verbose)
