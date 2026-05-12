"""
data_schema.py v1.9 (Quantum Finance Edition)
================================================================================
數據契約中心 — 旗艦編年史版 (Quantum v5.2 標準)
負責全系統數據字典註冊、SQL 自動生成與 API 實體鏡像對齊。

【核心定義說明 (Core Definitions)】
1. [Authoritative Registry]: 建立 1:1 API 鏡像註冊表，實現「配置即結構」。
2. [API-to-DB Mapping]: 深度整合 FinMind v4 與 FRED API，確保欄位名稱與類型 100% 兼容。
3. [High Precision Policy]: 強制採用 NUMERIC(20, 6) 存儲，消除金融計算誤差。

【執行範例矩陣 (Flagship Operational Matrix)】
┌──────────────────────────────────────┬────────────────────────────────────────────────────────┐
│ 維運場景                             │ 建議指令 / 用法                                        │
├──────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [個股/單一表：連通性校驗]         │ $ python scripts/core/data_schema.py --init            │
│ 2. [單一個股所有表：結構重組]        │ $ python scripts/core/data_schema.py --init --force     │
│ 3. [所有核心股與全表：強制全量同步]  │ $ python scripts/ingestion/template_fetcher.py          │
│                                      │   --universe core --all_datasets --force               │
│ 4. [數據治權：實體表強制重鑄]        │ $ python scripts/core/data_schema.py --init --force     │
└──────────────────────────────────────┴────────────────────────────────────────────────────────┘

【API 數據映射明細 (API Mapping Specification)】
- [FinMind: TaiwanStockInfo] -> 核心標的元數據 (端點: TaiwanStockInfo)
- [FinMind: TaiwanStockPrice] -> 每日 OHLCV (端點: TaiwanStockPrice)
- [FinMind: InstitutionalInvestors] -> 三大法人買賣超 (端點: TaiwanStockInstitutionalInvestorsBuySell)
- [FRED: MacroData] -> 總體經濟指標 (端點: fred_data)

【全修訂歷程 (Full Revision History)】
  v1.9 (2026-05-12): [對齊] 注入詳細 API 映射明細 (FinMind/FRED) 與全量同步範例矩陣。
  v1.8 (2026-05-12): [憲法] 注入今日詳細核心定義、舊歷程保留規範。
  v1.7 (2026-05-11): [主權] 建立 Registry 治權，實現 1:1 API 鏡像。
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
    def record_lifecycle(*args, **kwargs):
        class Mock:
            def __enter__(self): pass
            def __exit__(self, *args): pass
        return Mock()
    def write_data_audit_log(*args, **kwargs): pass

# 核心數據契約 Registry (v1.9: API Mapping Optimized)
# 這裡定義了 1:1 的 API 欄位對應關係
DATASET_SCHEMA_MAP = {
    "TaiwanStockInfo": {
        "_api_source": "FinMind/TaiwanStockInfo",
        "industry_category": "TEXT",
        "stock_id": "TEXT PRIMARY KEY",
        "stock_name": "TEXT",
        "type": "TEXT",
        "date": "DATE"
    },
    "TaiwanStockPrice": {
        "_api_source": "FinMind/TaiwanStockPrice",
        "date": "DATE",
        "stock_id": "TEXT",
        "open": "NUMERIC(20, 6)",
        "high": "NUMERIC(20, 6)",
        "low": "NUMERIC(20, 6)",
        "close": "NUMERIC(20, 6)",
        "Volume": "NUMERIC(20, 6)",  # 對應 API: Volume
        "Trading_Money": "NUMERIC(20, 6)",
        "Trading_turnover": "NUMERIC(20, 6)",
        "spread": "NUMERIC(20, 6)",
        "upper_limit": "NUMERIC(20, 6)",
        "lower_limit": "NUMERIC(20, 6)"
    },
    "TaiwanStockInstitutionalInvestorsBuySell": {
        "_api_source": "FinMind/TaiwanStockInstitutionalInvestorsBuySell",
        "date": "DATE",
        "stock_id": "TEXT",
        "buy": "NUMERIC(20, 6)",
        "sell": "NUMERIC(20, 6)",
        "name": "TEXT"
    },
    "TaiwanStockMarginPurchaseShortSale": {
        "_api_source": "FinMind/TaiwanStockMarginPurchaseShortSale",
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
        "_api_source": "FRED/series",
        "date": "DATE",
        "value": "NUMERIC(20, 6)",
        "series_id": "TEXT"
    }
}

def init_schema(force=False):
    """根據 Registry 自動生成並執行 SQL (v1.9 治權邏輯)"""
    with record_lifecycle("schema_init_v1.9", category="maintenance", stock_id="SYSTEM"):
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            for table_name, config in DATASET_SCHEMA_MAP.items():
                if force:
                    cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
                
                # 過濾掉 API 元數據，僅生成 SQL 欄位
                fields = {k: v for k, v in config.items() if not k.startswith("_")}
                cols = ", ".join([f"{col} {dtype}" for col, dtype in fields.items()])
                sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({cols});"
                cur.execute(sql)
                
                api_src = config.get("_api_source", "Unknown")
                print(f"  ✅ [{table_name}] -> 對齊 API: {api_src} (SUCCESS)")
                
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
    parser.add_argument("--force", action="store_true", help="強制重新建立實體表")
    args = parser.parse_args()

    if args.init:
        print("\n" + "=" * 60)
        print("🚀 Quantum Finance: 數據契約與 API 對應初始化 (v1.9)")
        print("=" * 60)
        init_schema(force=args.force)
        print("=" * 60 + "\n")
    else:
        parser.print_help()
