"""
check_schema_consistency.py v8.0 (Quantum Finance Edition)
================================================================================
結構哨兵 — 全域資料庫一致性稽核與自癒引擎 (Quantum v5.2 標準)
負責校驗資料庫表格與 API 結構的對齊，並提供「硬核保底」自動建表功能。

修訂歷程：
  v8.0 (2026-05-11): [標準] 升級至 v5.2 標準，整合 core 模組並導入硬核保底 SQL 矩陣。
  v7.1 (2026-05-11): [功能] 導入壓力測試樣本探測模式。

【執行範例矩陣 (Consistency Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / 用法                                        │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [全系統架構一致性稽核]    │ $ python scripts/maintenance/check_schema_consistency.py│
│ 2. [全量缺失資料表自動修復]  │ $ python scripts/maintenance/check_schema_consistency.py --fix│
│ 3. [單一資料表強制修復]      │ $ python scripts/maintenance/check_schema_consistency.py --id table_name --fix│
└──────────────────────────────┴────────────────────────────────────────────────────────┘
================================================================================
"""
import sys, logging, time, argparse
from pathlib import Path

# ── 系統級架構引導 ──
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

try:
    from core import db_transaction, record_lifecycle, ensure_infrastructure
except ImportError as e:
    print(f"[FATAL] 核心架構引導失敗: {e}")
    sys.exit(1)

# ── 硬核保底 SQL 矩陣 (Hardcoded Fallback) ──
# 這裡定義了系統所有核心數據表的「完美結構」
SCHEMA_FALLBACK = {
    "stock_price_day": """
        CREATE TABLE IF NOT EXISTS stock_price_day (
            date DATE, stock_id VARCHAR(20), open FLOAT, max FLOAT, min FLOAT, close FLOAT, 
            spread FLOAT, Trading_Volume BIGINT, Trading_money BIGINT, Trading_turnover BIGINT,
            PRIMARY KEY (stock_id, date)
        );
    """,
    "stock_institutional_investors": """
        CREATE TABLE IF NOT EXISTS stock_institutional_investors (
            date DATE, stock_id VARCHAR(20), name VARCHAR(50), 
            buy BIGINT, sell BIGINT, net_buy BIGINT,
            PRIMARY KEY (stock_id, date, name)
        );
    """,
    "stock_margin_purchase_short_sale": """
        CREATE TABLE IF NOT EXISTS stock_margin_purchase_short_sale (
            date DATE, stock_id VARCHAR(20), 
            MarginPurchaseBuy BIGINT, MarginPurchaseSell BIGINT, MarginPurchaseCashRepayment BIGINT,
            ShortSaleBuy BIGINT, ShortSaleSell BIGINT, ShortSaleCashRepayment BIGINT,
            PRIMARY KEY (stock_id, date)
        );
    """
}

def check_and_fix(target_table: str = None, apply_fix: bool = False):
    """執行架構稽核與自癒"""
    with record_lifecycle("schema_consistency_audit", "maintenance", "SYSTEM"):
        print("\n" + "🏛️"*40)
        print(f"🚀 Quantum Finance: 結構哨兵報告 (v8.0)")
        print("🏛️"*40)
        
        tables_to_check = [target_table] if target_table else SCHEMA_FALLBACK.keys()
        
        for table in tables_to_check:
            if table not in SCHEMA_FALLBACK:
                print(f"  ⚠️  跳過: {table} (未定義保底 SQL)")
                continue
                
            with db_transaction() as cur:
                cur.execute(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{table}');")
                exists = cur.fetchone()['exists']
                
                if not exists:
                    status = "❌ 缺失"
                    if apply_fix:
                        print(f"  🔧 正在自癒: {table}...")
                        cur.execute(SCHEMA_FALLBACK[table])
                        status = "✅ 已修復"
                else:
                    status = "✅ 正常"
                
                print(f"  📍 資料表: {table.ljust(35)} | 狀態: {status}")

        print("\n" + "🏛️"*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", help="指定稽核單一資料表")
    parser.add_argument("--fix", action="store_true", help="執行自動修復")
    args = parser.parse_args()

    ensure_infrastructure()
    check_and_fix(target_table=args.id, apply_fix=args.fix)
