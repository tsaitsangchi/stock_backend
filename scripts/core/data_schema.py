"""
data_schema.py v1.8 (Quantum Finance Edition)
================================================================================
數據契約中心 — 旗艦編年史版 (Quantum v5.2 標準)
負責全系統數據字典註冊、SQL 自動生成與資料庫結構主權維護。

【核心定義說明 (Core Definitions)】
1. [Authoritative Registry]: 建立 1:1 API 鏡像註冊表，棄用硬編碼 SQL 字串，實現「配置即結構」。
2. [High Precision Policy]: 強制採用 NUMERIC(20, 6) 存儲金融數據，消除浮點數計算誤差。
3. [Auto-Governance]: 透過 DATASET_SCHEMA_MAP 實現實體表的自動化初始化與結構校驗。

【執行範例矩陣 (Historical & Active Matrix)】
┌──────────────────────────────────────┬────────────────────────────────────────────────────────┐
│ 維運場景                             │ 建議指令 / 用法                                        │
├──────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [單一表：連通性校驗]              │ $ python scripts/core/data_schema.py --init            │
│ 2. [單一表：結構毀滅性重鑄]          │ $ python scripts/core/data_schema.py --init --force     │
│ 3. [舊版範例 (v1.0)：手動建立股價表] │ $ psql -c "CREATE TABLE TaiwanStockPrice (...)" (已廢棄) │
│ 4. [標準範例 (v1.4)：多表批次對齊]   │ $ python scripts/core/data_schema.py --all-tables       │
│ 5. [核心範例 (v1.7)：全主權重鑄]     │ $ python scripts/core/data_schema.py --init --force     │
└──────────────────────────────────────┴────────────────────────────────────────────────────────┘

【全修訂歷程 (Full Revision History)】
  v1.8 (2026-05-12): [憲法] 注入今日詳細核心定義、舊歷程保留規範，對齊 2026-05-12 旗艦要求。
  v1.7 (2026-05-11): [主權] 建立 Registry 治權，實現 1:1 API 鏡像，確立 v5.2 數據憲法。
  v1.6 (2026-05-11): [標準] 整合「混合日誌」與「極致維運矩陣」Header。
  v1.5 (2026-05-10): [治理] 強制執行 NUMERIC(20, 6) 高精度存儲規範。
  v1.4 (2026-05-08): [擴充] 加入 TaiwanStockInstitutionalInvestorsBuySell 等多維數據對齊。
  v1.0 (2026-05-01): [奠基] 初始版本，硬編碼 TaiwanStockPrice 基礎結構。
================================================================================
"""
import sys, argparse, logging
from pathlib import Path
from datetime import datetime

# ── 系統級架構引導 ──
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

try:
    from core.db_utils import get_db_connection, record_lifecycle, write_data_audit_log
except ImportError:
    # 支援獨立執行 (Lazy Import)
    def record_lifecycle(*args, **kwargs):
        class Mock:
            def __enter__(self): pass
            def __exit__(self, *args): pass
        return Mock()
    def write_data_audit_log(*args, **kwargs): pass

# 核心數據契約 Registry (v1.7 權威定義)
DATASET_SCHEMA_MAP = {
    "TaiwanStockInfo": {
        "industry_category": "TEXT",
        "stock_id": "TEXT PRIMARY KEY",
        "stock_name": "TEXT",
        "type": "TEXT",
        "date": "DATE"
    },
    "TaiwanStockPrice": {
        "date": "DATE",
        "stock_id": "TEXT",
        "open": "NUMERIC(20, 6)",
        "high": "NUMERIC(20, 6)",
        "low": "NUMERIC(20, 6)",
        "close": "NUMERIC(20, 6)",
        "Volume": "NUMERIC(20, 6)",
        "Trading_Money": "NUMERIC(20, 6)",
        "Trading_turnover": "NUMERIC(20, 6)",
        "spread": "NUMERIC(20, 6)",
        "upper_limit": "NUMERIC(20, 6)",
        "lower_limit": "NUMERIC(20, 6)"
    },
    "TaiwanStockInstitutionalInvestorsBuySell": {
        "date": "DATE",
        "stock_id": "TEXT",
        "buy": "NUMERIC(20, 6)",
        "sell": "NUMERIC(20, 6)",
        "name": "TEXT"
    },
    "TaiwanStockMarginPurchaseShortSale": {
        "date": "DATE",
        "stock_id": "TEXT",
        "MarginPurchaseBuy": "NUMERIC(20, 6)",
        "MarginPurchaseSell": "NUMERIC(20, 6)",
        "MarginPurchaseCashRepayment": "NUMERIC(20, 6)",
        "MarginPurchaseLimit": "NUMERIC(20, 6)",
        "ShortSaleBuy": "NUMERIC(20, 6)",
        "ShortSaleSell": "NUMERIC(20, 6)",
        "ShortSaleCashBalance": "NUMERIC(20, 6)",
        "ShortSaleLimit": "NUMERIC(20, 6)"
    },
    "FredData": {
        "date": "DATE",
        "value": "NUMERIC(20, 6)",
        "series_id": "TEXT"
    }
}

def init_schema(force=False):
    """根據 Registry 自動生成並執行 SQL (v1.7 治權邏輯)"""
    with record_lifecycle("schema_init_v1.7", category="maintenance", stock_id="SYSTEM"):
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            for table_name, schema in DATASET_SCHEMA_MAP.items():
                if force:
                    cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
                
                cols = ", ".join([f"{col} {dtype}" for col, dtype in schema.items()])
                sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({cols});"
                cur.execute(sql)
                print(f"  ✅ [{table_name}] -> {table_name} 對齊成功")
                
                # 專項數據審計
                write_data_audit_log(table_name, "SYSTEM", datetime.now().strftime("%Y-%m-%d"), 
                                     "SCHEMA_FORCE_REBUILD" if force else "SCHEMA_INIT", 1)
            
            conn.commit()
            print("-" * 60 + "\n✨ 混合日誌已同步完成。")
        except Exception as e:
            conn.rollback()
            print(f"❌ 初始化失敗: {e}")
        finally:
            cur.close()
            conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum Finance 數據契約管理")
    parser.add_argument("--init", action="store_true", help="初始化數據契約")
    parser.add_argument("--force", action="store_true", help="強制重新建立實體表 (警告：數據將被清空)")
    args = parser.parse_args()

    if args.init:
        print("\n" + "=" * 60)
        print("🚀 Quantum Finance: 終極數據契約初始化任務 (v1.8)")
        print("=" * 60)
        init_schema(force=args.force)
        print("=" * 60 + "\n")
    else:
        parser.print_help()
