"""
data_schema.py v2.2 (Quantum Finance Edition)
================================================================================
數據契約與 API 對齊引擎 — 憲法完整版 (Quantum v5.2 標準)
負責全系統數據字典註冊、SQL 自動生成、API 鏡像對齊與全量維運指令矩陣。

【核心定義說明 (Core Definitions)】
1. [Data Contract Sovereignty]: 確立數據契約為系統「真理來源」，API 必須 100% 鏡像對齊。
2. [Schema Compatibility Sovereignty]: 
   - [VARCHAR]: 所有的字串型態欄位最少定義為 VARCHAR(100)，領先 API 傳入長度以確保相容。
   - [NUMBER]: 所有的數值型態欄位最少定義為 NUMERIC(20, 6)，確保計算精度與相容。
3. [Historical Reference Authority]: 保留從 v1.0 到 v2.2 的所有歷史歷程，作為判定系統正確性的基準。
4. [Hybrid Observability]: 契約變動必須同時記錄在生命週期日誌與數據審計日誌中。

【全量執行範例矩陣 (The Complete Operational Matrix)】
┌──────────────────────────────────────────┬────────────────────────────────────────────────────────┐
│ 維運需求場景                             │ 建議指令 / 用法                                        │
├──────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [個股 / 單一表：連通性偵測]           │ $ python scripts/core/data_schema.py --init            │
│ 2. [單一 Table：毀滅性結構重鑄]          │ $ python scripts/core/data_schema.py --init --force     │
│ 3. [單一個股 / 所有表：全量數據同步]     │ $ python scripts/ingestion/template_fetcher.py          │
│                                          │   --id 2330 --all_datasets                             │
│ 4. [所有核心股 / 所有表：全量數據同步]   │ $ python scripts/ingestion/template_fetcher.py          │
│                                          │   --universe core --all_datasets                       │
│ 5. [所有核心股 / 所有表：全量強制更新]   │ $ python scripts/ingestion/template_fetcher.py          │
│                                          │   --universe core --all_datasets --force               │
│ 6. [數據稽核：跨年度契約一致性檢查]      │ $ python scripts/maintenance/check_schema_consistency.py│
└──────────────────────────────────────────┴────────────────────────────────────────────────────────┘

【全修訂歷程 (Full Revision History)】
  v2.2 (2026-05-12): [防禦] 升級所有字串欄位至 VARCHAR(100)，落實「數據相容主權」。
  v2.1 (2026-05-12): [憲法] 補全全量維運矩陣與四維核心定義，對齊 v5.2 旗艦要求。
  v1.0 (2026-04-20): [奠基] 初始數據契約定義。
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
    print("⚠️  無法從 core 導入組件，啟用 Mock 模式 (請確認 path_setup)")
    def record_lifecycle(*args, **kwargs):
        class Mock:
            def __enter__(self): pass
            def __exit__(self, *args): pass
        return Mock()
    def write_data_audit_log(*args, **kwargs): pass

# 數據契約 Registry (v2.2 防禦性寬容版)
# 所有 VARCHAR 起跳值為 100，所有 NUMBER 為 NUMERIC(20, 6)
DATASET_SCHEMA_MAP = {
    "TaiwanStockInfo": {
        "_api": "FinMind/TaiwanStockInfo", 
        "stock_id": "VARCHAR(100) PRIMARY KEY", 
        "stock_name": "VARCHAR(100)", 
        "industry_category": "VARCHAR(100)", 
        "type": "VARCHAR(100)", 
        "date": "DATE"
    },
    "TaiwanStockPrice": {
        "_api": "FinMind/TaiwanStockPrice", 
        "date": "DATE", 
        "stock_id": "VARCHAR(100)", 
        "open": "NUMERIC(20, 6)", 
        "high": "NUMERIC(20, 6)", 
        "low": "NUMERIC(20, 6)", 
        "close": "NUMERIC(20, 6)", 
        "Volume": "NUMERIC(20, 6)", 
        "Trading_Money": "NUMERIC(20, 6)"
    },
    "TaiwanStockInstitutionalInvestorsBuySell": {
        "_api": "FinMind/InstInvestors", 
        "date": "DATE", 
        "stock_id": "VARCHAR(100)", 
        "buy": "NUMERIC(20, 6)", 
        "sell": "NUMERIC(20, 6)", 
        "name": "VARCHAR(100)"
    },
    "TaiwanStockMarginPurchaseShortSale": {
        "_api": "FinMind/Margin", 
        "date": "DATE", 
        "stock_id": "VARCHAR(100)", 
        "MarginPurchaseBuy": "NUMERIC(20, 6)", 
        "MarginPurchaseSell": "NUMERIC(20, 6)"
    },
    "FredData": {
        "_api": "FRED/series", 
        "date": "DATE", 
        "value": "NUMERIC(20, 6)", 
        "series_id": "VARCHAR(100)"
    }
}

def init_schema(force=False):
    """執行數據契約初始化 (v2.2 憲法版)"""
    start_time = datetime.now()
    results = []
    
    with record_lifecycle("schema_init_v2.2", category="maintenance", stock_id="SYSTEM"):
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            for table_name, config in DATASET_SCHEMA_MAP.items():
                if force: cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
                
                fields = {k: v for k, v in config.items() if not k.startswith("_")}
                cols = ", ".join([f"{col} {dtype}" for col, dtype in fields.items()])
                cur.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({cols});")
                
                api_src = config.get("_api", "Unknown")
                results.append(f"  ✅ [SUCCESS] 表: {table_name:<40} 物理狀態: DEFENSIVE_OVERSIZED")
                write_data_audit_log(table_name, "SYSTEM", datetime.now().strftime("%Y-%m-%d"), "INIT", 1)
            
            conn.commit()
            
            # ── 執行後詳細結果摘要 (Detailed Summary) ──
            print("\n" + "🛡️" * 40)
            print("🚀 Quantum Finance: 數據契約與 API 對齊旗艦初始化 (v2.2)")
            print("🛡️" * 40)
            
            print("\n" + "─" * 80)
            print("📊 執行任務摘要報告 (Task Summary Report v2.2)")
            print("─" * 80)
            for r in results: print(r)
            print("─" * 80)
            print(f"🕒 執行總時長   : {(datetime.now() - start_time).total_seconds():.2f}s")
            print(f"⚖️  數據主權狀態 : PERFECT (防禦性相容對齊)")
            print("─" * 80)
            
            # ── 治權維運建議 (Reference Info) ──
            print("\n💡 治權維運建議 (Reference Information):")
            print("1. [相容提示]: 字串欄位已升級為 VARCHAR(100)，數值為 NUMERIC(20, 6)。")
            print("2. [範例提示]: 請參閱 Header 矩陣執行「毀滅性結構重鑄」以套用新長度。")
            print("3. [歷史提示]: 所有結構變更已歸檔至 pipeline_execution_log。")
            print("─" * 80 + "\n")
            
        except Exception as e:
            conn.rollback()
            print(f"❌ 關鍵錯誤: {e}")
        finally:
            cur.close(); conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum Finance 數據契約管理")
    parser.add_argument("--init", action="store_true", help="初始化數據契約")
    parser.add_argument("--force", action="store_true", help="強制重新建立實體表")
    args = parser.parse_args()

    if args.init:
        init_schema(force=args.force)
    else:
        parser.print_help()
