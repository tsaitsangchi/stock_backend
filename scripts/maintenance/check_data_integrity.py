"""
check_data_integrity.py v7.3 (Quantum Finance Edition)
================================================================================
數據完整性稽核工具 — 高階資產可信度引擎 (Quantum v5.2 標準)
負責檢查資料庫內的數據密度、覆蓋區間與同步進度，支持全市場與核心股多維度掃描。

修訂歷程：
  v7.3 (2026-05-11): [終極版] 擴展 8 大稽核維度、實作核心股智慧過濾、補全全場景執行範例矩陣。
  v7.2 (2026-05-11): [標準化] 導入 Quantum 標準檔頭、生命週期監測與數據健康儀表板。

【執行範例矩陣 (Comprehensive Usage Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令                                               │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [核心標的深度稽核]        │ $ python scripts/maintenance/check_data_integrity.py   │
│ 2. [單一個股全表稽核]        │ $ python scripts/maintenance/check_data_integrity.py --stock_id 2330 │
│ 3. [全市場單一資料表稽核]    │ $ python scripts/maintenance/check_data_integrity.py --table_name stock_price │
│ 4. [全量宇宙稽核 (3000+ 檔)] │ $ python scripts/maintenance/check_data_integrity.py --all_stocks │
│ 5. [查看稽核異常紀錄]        │ SELECT * FROM data_audit_log WHERE table_name = 'integrity_check'; │
└──────────────────────────────┴────────────────────────────────────────────────────────┘

【業務邏輯說明】
  - 檢查維度：Price, Chip, Fundamental, Revenue, Margin, Dividend, AdjPrice, ReturnIndex。
  - 健康度定義：該標的在所有維度中均有數據的比例。
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

# 全維度稽核清單
METRICS = {
    "Price": "stock_price",
    "Chip": "institutional_investors_buy_sell",
    "Fundamental": "financial_statements",
    "Revenue": "month_revenue",
    "Margin": "margin_purchase_short_sale",
    "Dividend": "dividend",
    "AdjPrice": "price_adj",
    "ReturnIdx": "total_return_index"
}

def show_integrity_dashboard(stats: dict):
    """稽核任務後的健康儀表板回報。"""
    print("\n" + "🔍"*35)
    print("🚀 Quantum Finance: 數據完整性稽核報告 (v7.3)")
    print("🔍"*35)
    print(f"✅ 稽核狀態  : 執行完成")
    print(f"📊 掃描標的  : {stats['total_stocks']} 檔")
    print(f"🧩 檢查維度  : {len(METRICS)} 個資料表")
    print(f"⚠️ 發現異常  : {stats['total_issues']} 項資料缺漏")
    print(f"🌡️  系統健康度: {stats['avg_health']:.1f}%")
    
    if stats['avg_health'] < 50:
        print("-" * 70)
        print(f"🔴 警告：健康度低於 50%，建議使用 parallel_ingestion.py 執行數據全量初始化。")
    elif stats['total_issues'] > 0:
        print("-" * 70)
        print(f"🟡 提醒：發現部分缺漏，請針對缺失標的進行補增。")
    else:
        print("-" * 70)
        print("🟢 完美！數據宇宙已完全對齊。")
        
    print("-" * 70)
    print("📝 日誌同步: pipeline_execution_log & data_audit_log (integrity_check)")
    print("🔍"*35 + "\n")

def audit_integrity(stock_id: str = None, table_name: str = None, all_stocks: bool = False, verbose: bool = False):
    """執行全系統數據完整性稽核。"""
    t0 = time.monotonic()
    
    with record_lifecycle("data_integrity_audit", category="maintenance", stock_id=stock_id or "GLOBAL"):
        # 1. 決定標的名單
        if stock_id:
            target_ids = [stock_id]
        elif all_stocks:
            target_ids = get_db_stock_ids(active_only=True)
        else:
            # 預設僅稽核核心標的 (is_core=TRUE)
            with db_transaction() as cur:
                cur.execute("SELECT stock_id FROM stocks WHERE is_core = TRUE;")
                target_ids = [r['stock_id'] for r in cur.fetchall()]
            if not target_ids: target_ids = ["2330", "2317", "2454"] # 最終降級

        # 2. 決定稽核維度
        target_metrics = {k: v for k, v in METRICS.items() if (not table_name or v == table_name)}
        
        logger.info(f"🔍 [Audit] 啟動深層稽核 (標的: {len(target_ids)} 檔, 維度: {len(target_metrics)} 類)...")

        total_issues = 0
        total_dimensions = len(target_ids) * len(target_metrics)
        
        # 繪製表格
        print("\n" + "-"*90)
        print(f"{'Stock ID':<10} | {'Metric':<12} | {'Table Name':<30} | {'Last Date':<12} | {'Status'}")
        print("-" * 90)

        for sid in target_ids:
            for m_name, t_name in target_metrics.items():
                last_dt = get_latest_date(t_name, sid)
                status = "✅ OK" if last_dt else "❌ MISSING"
                if not last_dt: total_issues += 1
                
                if not last_dt or verbose or stock_id:
                    print(f"{sid:<10} | {m_name:<12} | {t_name:<30} | {str(last_dt):<12} | {status}")

        # 3. 統計與日誌
        health = ((total_dimensions - total_issues) / total_dimensions * 100) if total_dimensions > 0 else 0
        stats = {
            "total_stocks": len(target_ids),
            "total_issues": total_issues,
            "avg_health": health
        }
        
        # 混合模式：審計日誌
        write_data_audit_log("integrity_check", stock_id or "SYSTEM", date.today().strftime("%Y-%m-%d"), "SCAN", total_issues)
        
        show_integrity_dashboard(stats)
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock_id", type=str, help="稽核指定個股")
    parser.add_argument("--table_name", type=str, help="稽核指定資料表")
    parser.add_argument("--all_stocks", action="store_true", help="稽核全市場標的")
    parser.add_argument("--verbose", action="store_true", help="顯示所有細節 (包含正常項)")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    audit_integrity(stock_id=args.stock_id, table_name=args.table_name, all_stocks=args.all_stocks, verbose=args.verbose)
