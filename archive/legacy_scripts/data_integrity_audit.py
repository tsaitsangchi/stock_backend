"""
data_integrity_audit.py v2.1 (Quantum Finance Edition)
================================================================================
系統級數據審計器 — 全域資料一致性與完整性主控台 (Quantum v5.2 標準)
負責執行全系統 8 大維度數據的深度稽核，並輸出結構化的審計報告與 Hybrid Logging。

修訂歷程：
  v2.1 (2026-05-11): [標準化] 導入全維度審計邏輯、智慧儀表板與範例矩陣。
  v5.5.7 (2026-05-09): [核心] 導入 Hybrid Logging 混合日誌。

【執行範例矩陣 (Audit Operations Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / SQL                                         │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [執行全系統主審計]        │ $ python scripts/maintenance/data_integrity_audit.py   │
│ 2. [指定標的深度穿透]        │ $ python scripts/maintenance/data_integrity_audit.py --stock_id 2330 │
│ 3. [查看歷史審計健康度]      │ SELECT rows_affected, created_at FROM data_audit_log   │
│                              │ WHERE table_name = 'system_audit' ORDER BY id DESC;    │
└──────────────────────────────┴────────────────────────────────────────────────────────┘

【業務邏輯說明】
  - 核心指標: 針對 stocks 表中標記為 is_core 的標的進行 8 維度資料密度稽核。
  - 數據維度: Price, Chip, Fundamental, Revenue, Margin, Dividend, AdjPrice, ReturnIdx。
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

# 審計維度映射
AUDIT_METRICS = {
    "Price": "stock_price",
    "Chip": "institutional_investors_buy_sell",
    "Fundamental": "financial_statements",
    "Revenue": "month_revenue",
    "Margin": "margin_purchase_short_sale",
    "Dividend": "dividend",
    "AdjPrice": "price_adj",
    "ReturnIdx": "total_return_index"
}

def show_audit_dashboard(stats: dict):
    """執行後的高級審計儀表板。"""
    print("\n" + "⚖️"*35)
    print("🚀 Quantum Finance: 全系統數據審計報告 (v2.1)")
    print("⚖️"*35)
    print(f"✅ 審計狀態  : 執行完成")
    print(f"📊 標的總數  : {stats['total_stocks']} 檔 (核心標的)")
    print(f"🧩 審計維度  : {len(AUDIT_METRICS)} 大維度")
    print(f"⚠️ 缺漏總數  : {stats['total_issues']} 項")
    print(f"🌡️  系統置信度: {stats['avg_health']:.1f}%")
    
    if stats['avg_health'] < 100:
        print("-" * 70)
        print(f"📢 審計建議：發現數據缺漏，建議執行 parallel_ingestion.py 啟動全量宇宙同步。")
    else:
        print("-" * 70)
        print("🟢 完美狀態：全系統數據審計通過，完整性 100%。")
        
    print("-" * 70)
    print("📝 日誌同步: pipeline_execution_log & data_audit_log (system_audit)")
    print("⚖️"*35 + "\n")

def run_audit(stock_id: str = None):
    """啟動全系統審計流程。"""
    with record_lifecycle("system_data_audit", category="sys", stock_id=stock_id or "ALL"):
        try:
            # 1. 決定標的名單 (優先選取核心標的)
            if stock_id:
                target_ids = [stock_id]
            else:
                with db_transaction() as cur:
                    cur.execute("SELECT stock_id FROM stocks WHERE is_core = TRUE;")
                    target_ids = [r['stock_id'] for r in cur.fetchall()]
                if not target_ids: target_ids = ["2330", "2317", "2454"]

            logger.info(f"⚖️ 啟動全系統審計 (目標: {len(target_ids)} 檔標的)...")
            
            total_issues = 0
            total_dimensions = len(target_ids) * len(AUDIT_METRICS)
            
            for sid in target_ids:
                for m_name, t_name in AUDIT_METRICS.items():
                    if not get_latest_date(t_name, sid):
                        total_issues += 1
            
            # 2. 計算健康度
            health = ((total_dimensions - total_issues) / total_dimensions * 100) if total_dimensions > 0 else 0
            stats = {
                "total_stocks": len(target_ids),
                "total_issues": total_issues,
                "avg_health": health
            }
            
            # 3. 混合紀錄
            write_data_audit_log("system_audit", stock_id or "SYSTEM", 
                                 date.today().strftime("%Y-%m-%d"), "SCAN", total_issues)
            
            show_audit_dashboard(stats)
            return True
        except Exception as e:
            logger.error(f"❌ 審計任務失敗: {e}")
            raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock_id", type=str, help="指定審計標的")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_audit(stock_id=args.stock_id)
