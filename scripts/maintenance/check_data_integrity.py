"""
check_data_integrity.py v7.7 (Quantum Finance Edition)
================================================================================
數據完整性稽核工具 — 最佳架構示範版 (Quantum v5.2 標準)
負責檢查資料庫內的數據密度。本版本採用 core 統一接口導入。

修訂歷程：
  v7.7 (2026-05-11): [架構最佳化] 採用 core 統一調度接口導入，簡化路徑管理。
  v7.6 (2026-05-11): [修復] 修正儀表板邏輯 Bug。
================================================================================
"""
import sys, logging, time, argparse
from datetime import datetime, date
from pathlib import Path

# ── 最佳架構：透過 core 統一引導 ──
try:
    import core
    # 這裡我們展示 core 統一暴露的好處：直接從 core 導入，不需要寫 core.db_utils
    from core import (
        db_transaction, get_db_stock_ids, get_latest_date, 
        record_lifecycle, write_data_audit_log
    )
except ImportError as e:
    print(f"\n[FATAL] 核心架構引導失敗！錯誤: {e}")
    sys.exit(1)

logger = logging.getLogger(__name__)

METRICS = {
    "Price": "stock_price", "Chip": "institutional_investors_buy_sell", "Fundamental": "financial_statements",
    "Revenue": "month_revenue", "Margin": "margin_purchase_short_sale", "Dividend": "dividend",
    "AdjPrice": "price_adj", "ReturnIdx": "total_return_index"
}

def show_integrity_dashboard(stats: dict):
    print("\n" + "🔍"*35)
    print("🚀 Quantum Finance: 數據完整性稽核報告 (v7.7)")
    print("🔍"*35)
    print(f"✅ 稽核狀態  : 執行完成")
    print(f"🌡️  系統健康度: {stats['avg_health']:.1f}%")
    print("-" * 70)
    if stats['avg_health'] < 100:
        print(f"🔴 警報：偵測到 {stats['total_issues']} 項缺漏！")
    else:
        print("🟢 完美！系統已完全對齊。")
    print("-" * 70)
    print("📝 日誌同步: pipeline_execution_log & data_audit_log")
    print("🔍"*35 + "\n")

def audit_integrity(stock_id=None, table_name=None, all_stocks=False, verbose=False):
    with record_lifecycle("data_integrity_audit", category="maintenance", stock_id=stock_id or "GLOBAL"):
        if stock_id: target_ids = [stock_id]
        elif all_stocks: target_ids = get_db_stock_ids(active_only=True)
        else:
            with db_transaction() as cur:
                cur.execute("SELECT stock_id FROM stocks WHERE is_core = TRUE;")
                target_ids = [r['stock_id'] for r in cur.fetchall()]
        
        target_metrics = {k: v for k, v in METRICS.items() if (not table_name or v == table_name)}
        total_issues = 0
        total_dimensions = len(target_ids) * len(target_metrics)

        for sid in target_ids:
            for m_name, t_name in target_metrics.items():
                last_dt = get_latest_date(t_name, sid)
                if not last_dt: total_issues += 1
                if not last_dt or verbose or stock_id:
                    print(f"[{sid}] {m_name:<12} | {str(last_dt):<12} | {'✅ OK' if last_dt else '❌ MISSING'}")

        health = ((total_dimensions - total_issues) / total_dimensions * 100) if total_dimensions > 0 else 0
        stats = {"total_stocks": len(target_ids), "total_issues": total_issues, "avg_health": health}
        write_data_audit_log("integrity_check", stock_id or "SYSTEM", date.today().strftime("%Y-%m-%d"), "SCAN", total_issues)
        show_integrity_dashboard(stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock_id", type=str)
    parser.add_argument("--table_name", type=str)
    parser.add_argument("--all_stocks", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    audit_integrity(stock_id=args.stock_id, table_name=args.table_name, all_stocks=args.all_stocks, verbose=args.verbose)
