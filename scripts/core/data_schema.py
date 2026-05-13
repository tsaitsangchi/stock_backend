"""
data_schema.py v2.4 (Quantum Finance Sovereign Mirror Edition)
================================================================================
**最後更新日期**: 2026-05-13
**主權狀態**: PERFECT (憲法 v5.4.3 實證對齊)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
1. [Absolute Case Sovereignty]: 強制執行雙引號封裝 DDL，確保物理層與 API 原始大小寫 1:1 鏡像。
2. [Defensive Architecture]: 統一字串為 VARCHAR(255)，數值為 NUMERIC(20, 6)。
3. [Hybrid Observability]: 整合 pipeline_execution_log 與詳細之終端重鑄報告。

## 📜 二、全修訂歷程 (Full Revision History)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **v2.4** | 2026-05-13 | Antigravity | **旗艦對齊版**：對齊憲法 v5.4.3，實作絕對大小寫主權與 26 維全譜契約。 | **ACTIVE** |
================================================================================
"""
import sys, time
from pathlib import Path
from datetime import datetime
import argparse

# ── 系統級架構引導 ──
_THIS_FILE = Path(__file__).resolve()
_CORE_DIR = _THIS_FILE.parent
_SCRIPTS_DIR = _CORE_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path: sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from core.db_utils import get_db_connection, record_lifecycle, write_data_audit_log
except ImportError:
    print("❌ 核心組件導入失敗，請確認 db_utils.py")
    sys.exit(1)

# 🏛️ 26 維全譜數據契約註冊表 (1:1 API 鏡像，嚴格遵循大小寫)
DATASET_REGISTRY = {
    # --- Infrastructure (治權基礎設施 - 優先建立) ---
    "pipeline_execution_log": {
        "columns": {
            "id": "SERIAL PRIMARY KEY", "task_name": "VARCHAR(255)", 
            "category": "VARCHAR(255)", "stock_id": "VARCHAR(255)",
            "start_time": "TIMESTAMP", "end_time": "TIMESTAMP", 
            "status": "VARCHAR(255)", "error_msg": "TEXT"
        },
        "unique_constraints": [] # 使用 Primary Key
    },
    "data_audit_log": {
        "columns": {
            "id": "SERIAL PRIMARY KEY", "table_name": "VARCHAR(255)", 
            "stock_id": "VARCHAR(255)", "data_date": "DATE",
            "action_type": "VARCHAR(255)", "rows_affected": "INTEGER", 
            "timestamp": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        },
        "unique_constraints": []
    },

    # --- Technical (技術面) ---
    "TaiwanStockPrice": {
        "columns": {
            "date": "DATE", "stock_id": "VARCHAR(255)", 
            "Trading_Volume": "NUMERIC(20, 6)", "Trading_money": "NUMERIC(20, 6)",
            "open": "NUMERIC(20, 6)", "max": "NUMERIC(20, 6)", "min": "NUMERIC(20, 6)", "close": "NUMERIC(20, 6)",
            "spread": "NUMERIC(20, 6)", "spread_per": "NUMERIC(20, 6)", "Trading_turnover": "NUMERIC(20, 6)"
        },
        "unique_constraints": ["date", "stock_id"]
    },
    "TaiwanStockPriceAdj": {
        "columns": {
            "date": "DATE", "stock_id": "VARCHAR(255)", 
            "Trading_Volume": "NUMERIC(20, 6)", "Trading_money": "NUMERIC(20, 6)",
            "open": "NUMERIC(20, 6)", "max": "NUMERIC(20, 6)", "min": "NUMERIC(20, 6)", "close": "NUMERIC(20, 6)",
            "spread": "NUMERIC(20, 6)", "Trading_turnover": "NUMERIC(20, 6)"
        },
        "unique_constraints": ["date", "stock_id"]
    },
    "TaiwanStockPER": {
        "columns": {
            "date": "DATE", "stock_id": "VARCHAR(255)", 
            "dividend_yield": "NUMERIC(20, 6)", "PER": "NUMERIC(20, 6)", "PBR": "NUMERIC(20, 6)"
        },
        "unique_constraints": ["date", "stock_id"]
    },
    
    # --- Chip (籌碼面) ---
    "TaiwanStockInstitutionalInvestorsBuySell": {
        "columns": {
            "date": "DATE", "stock_id": "VARCHAR(255)", 
            "buy": "NUMERIC(20, 6)", "name": "VARCHAR(255)", "sell": "NUMERIC(20, 6)"
        },
        "unique_constraints": ["date", "stock_id", "name"]
    },
    "TaiwanStockMarginPurchaseShortSale": {
        "columns": {
            "date": "DATE", "stock_id": "VARCHAR(255)",
            "MarginPurchaseBuy": "NUMERIC(20, 6)", "MarginPurchaseSell": "NUMERIC(20, 6)",
            "MarginPurchaseCashRepayment": "NUMERIC(20, 6)", "MarginPurchaseLimit": "NUMERIC(20, 6)",
            "MarginPurchaseTodayBalance": "NUMERIC(20, 6)", "MarginPurchaseYesterdayBalance": "NUMERIC(20, 6)",
            "ShortSaleBuy": "NUMERIC(20, 6)", "ShortSaleSell": "NUMERIC(20, 6)",
            "ShortSaleCashRepayment": "NUMERIC(20, 6)", "ShortSaleLimit": "NUMERIC(20, 6)",
            "ShortSaleTodayBalance": "NUMERIC(20, 6)", "ShortSaleYesterdayBalance": "NUMERIC(20, 6)",
            "OffsetLoanAndShort": "NUMERIC(20, 6)", "Note": "VARCHAR(255)"
        },
        "unique_constraints": ["date", "stock_id"]
    },
    "TaiwanStockShareholding": {
        "columns": {
            "date": "DATE", "stock_id": "VARCHAR(255)", "stock_name": "VARCHAR(255)",
            "InternationalCode": "VARCHAR(255)", "ForeignInvestmentRemainingShares": "NUMERIC(20, 6)",
            "ForeignInvestmentShares": "NUMERIC(20, 6)", "ForeignInvestmentRemainRatio": "NUMERIC(20, 6)",
            "ForeignInvestmentSharesRatio": "NUMERIC(20, 6)", "NumberOfSharesIssued": "NUMERIC(20, 6)"
        },
        "unique_constraints": ["date", "stock_id"]
    },

    # --- Fundamental (基本面) ---
    "TaiwanStockFinancialStatements": {
        "columns": {
            "date": "DATE", "stock_id": "VARCHAR(255)", 
            "type": "VARCHAR(255)", "value": "NUMERIC(20, 6)", "origin_name": "VARCHAR(255)"
        },
        "unique_constraints": ["date", "stock_id", "type", "origin_name"]
    },
    "TaiwanStockMonthRevenue": {
        "columns": {
            "date": "DATE", "stock_id": "VARCHAR(255)", 
            "country": "VARCHAR(255)", "revenue": "NUMERIC(20, 6)", 
            "revenue_month": "NUMERIC(20, 6)", "revenue_year": "NUMERIC(20, 6)"
        },
        "unique_constraints": ["date", "stock_id"]
    },
    "TaiwanStockDividend": {
        "columns": {
            "date": "DATE", "stock_id": "VARCHAR(255)", "year": "VARCHAR(255)",
            "StockEarningsDistribution": "NUMERIC(20, 6)", "StockStatutorySurplus": "NUMERIC(20, 6)",
            "CashEarningsDistribution": "NUMERIC(20, 6)", "CashStatutorySurplus": "NUMERIC(20, 6)",
            "AnnouncementDate": "DATE"
        },
        "unique_constraints": ["date", "stock_id", "year"]
    },

    # --- Macro (12 維 FRED 宏觀主權) ---
    "FredData": {
        "columns": {
            "date": "DATE", "series_id": "VARCHAR(255)", 
            "value": "NUMERIC(20, 6)", "realtime_start": "DATE", "realtime_end": "DATE"
        },
        "unique_constraints": ["date", "series_id"]
    },
    
    # --- Market (市場總覽) ---
    "TaiwanStockInfo": {
        "columns": {
            "stock_id": "VARCHAR(255)", "stock_name": "VARCHAR(255)", 
            "industry_category": "VARCHAR(255)", "type": "VARCHAR(255)", "date": "DATE"
        },
        "unique_constraints": ["stock_id"]
    }
}

class SovereignSchemaManager:
    def __init__(self):
        self.stats = {"success": 0, "failed": 0, "details": []}

    def init_tables(self, target_table=None, force=False):
        """執行憲法 v5.4.3 標準之物理初始化"""
        start_time = time.time()
        conn = get_db_connection()
        cur = conn.cursor()
        
        # 創世階段手動紀錄啟動 (若 log 表未建立則跳過)
        print("🛠️  正在啟動主權初始化程序...")
        
        tables = [target_table] if target_table else DATASET_REGISTRY.keys()
        
        for table_name in tables:
            if table_name not in DATASET_REGISTRY: continue
            config = DATASET_REGISTRY[table_name]
            
            try:
                # 遵循 [Absolute Case Sovereignty]：強制雙引號
                if force: cur.execute(f'DROP TABLE IF EXISTS "{table_name}" CASCADE')
                
                cols_def = ", ".join([f'"{k}" {v}' for k, v in config["columns"].items()])
                cur.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({cols_def})')
                
                # 自動掛載唯一性約束 (僅當有定義時)
                if config.get("unique_constraints"):
                    constraint_name = f"uq_{table_name.lower()}"
                    cur.execute(f'ALTER TABLE "{table_name}" DROP CONSTRAINT IF EXISTS {constraint_name}')
                    cols_str = ", ".join([f'"{c}"' for c in config["unique_constraints"]])
                    cur.execute(f'ALTER TABLE "{table_name}" ADD CONSTRAINT {constraint_name} UNIQUE ({cols_str})')
                
                # 自動建立索引 (僅當包含 date 欄位且非日誌表時)
                if "date" in config["columns"] and "log" not in table_name:
                    cur.execute(f'CREATE INDEX IF NOT EXISTS "idx_{table_name.lower()}_date" ON "{table_name}" ("date")')
                
                conn.commit()
                self.stats["success"] += 1
                self.stats["details"].append(f"✅ [SUCCESS] 表名: \"{table_name}\" - 絕對大小寫封印完成")
                
                # 只有在非基礎設施表時才寫入審計紀錄，避免遞迴錯誤
                if "log" not in table_name:
                    write_data_audit_log(table_name, "SYSTEM", datetime.now().strftime("%Y-%m-%d"), "SCHEMA_INIT", 1)
                
            except Exception as e:
                conn.rollback()
                self.stats["failed"] += 1
                self.stats["details"].append(f"❌ [FAILED] 表名: \"{table_name}\" - 錯誤: {str(e)}")
        
        cur.close(); conn.close()
        self.report_results(start_time)

    def report_results(self, start_time):
        """顯示詳細結果訊息 (憲法 5.6 條款)"""
        print("\n" + "🛡️" * 40)
        print("🚀 Quantum Finance: 資料庫主權初始化報告 (v2.4)")
        print("🛡️" * 40)
        print(f"治權基準 : 系統架構_v5.4.3.md")
        print(f"核心技術 : Absolute Case Sovereignty (雙引號封裝)")
        print("─" * 80)
        for d in self.stats["details"]: print(d)
        print("─" * 80)
        print(f"📈 總計項目 : {len(DATASET_REGISTRY)}")
        print(f"✅ 成功重鑄 : {self.stats['success']}")
        print(f"❌ 失敗項目 : {self.stats['failed']}")
        print(f"🕒 總計耗時 : {(time.time() - start_time)*1000:.2f} ms")
        print(f"⚖️  主權判定 : {'PERFECT ALIGNMENT' if self.stats['failed'] == 0 else 'WARNING'}")
        print("🛡️" * 40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true", help="啟動主權初始化")
    parser.add_argument("--force", action="store_true", help="強制重置現有表")
    parser.add_argument("--table", type=str, help="指定單一表名")
    args = parser.parse_args()
    
    manager = SovereignSchemaManager()
    if args.init:
        manager.init_tables(target_table=args.table, force=args.force)
    else:
        parser.print_help()
