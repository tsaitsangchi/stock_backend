"""
migrate_stocks_config.py v6.5 (Quantum Finance Edition)
================================================================================
資產宇宙同步器 — 終極數據遷移版 (Quantum v5.2 標準)
負責將 config.py 中的全量元數據(名稱、產業、美股連動)永久遷移至資料庫。

修訂歷程：
  v6.5 (2026-05-11): [主權化] 遷移全量元數據至 stocks 欄位，標誌著 DB 主權化完成。
  v6.4 (2026-05-11): [修復] 智能偵測 STOCK_CONFIGS。

【執行範例矩陣 (Sovereign Sync Matrix)】
  1. [執行全量元數據主權化遷移]  │ $ python scripts/core/migrate_stocks_config.py
================================================================================
"""
import sys, logging, platform, json
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_ROOT = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
if str(_SCRIPTS_ROOT) not in sys.path: sys.path.insert(0, str(_SCRIPTS_ROOT))

try:
    from core import db_transaction, record_lifecycle, ensure_infrastructure
    import config
except ImportError as e:
    print(f"[FATAL] 引導失敗: {str(e)}"); sys.exit(1)

def show_sovereign_dashboard(stats: dict):
    print("\n" + "🏛️"*40)
    print("🚀 Quantum Finance: 資料庫主權化遷移報告 (v6.5)")
    print("🏛️"*40)
    print(f"✅ 執行結果  : SUCCESS")
    print(f"📊 遷移規模  : {stats['count']} 檔標的")
    print(f"💎 屬性同步  : 名稱、產業、美股連動、ADR 溢價")
    print("-" * 80)
    print("🟢 遷移成功：資料庫已具備獨立治理能力，config.py 現可作為備份或種子使用。")
    print("📝 任務同步: pipeline_execution_log (sovereign_migration)")
    print("🏛️"*40 + "\n")

def run_sovereign_migration():
    with record_lifecycle("sovereign_migration", category="infra", stock_id="FS_SYSTEM"):
        # 確保資料庫結構已擴充
        ensure_infrastructure()
        
        stock_configs = getattr(config, "STOCK_CONFIGS", {})
        if not stock_configs:
            raise ValueError("在 config.py 中找不到 STOCK_CONFIGS")

        with db_transaction() as cur:
            # 重置 is_core
            cur.execute("UPDATE stocks SET is_core = FALSE;")
            
            for sid, info in stock_configs.items():
                us_chain = ",".join(info.get("us_chain_tickers", []))
                use_adr = info.get("use_adr_premium", False)
                
                # 執行 UPSERT (更新所有元數據)
                cur.execute("""
                    INSERT INTO stocks (stock_id, name, industry, us_chain_tickers, use_adr_premium, is_core, is_active)
                    VALUES (%s, %s, %s, %s, %s, TRUE, TRUE)
                    ON CONFLICT (stock_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        industry = EXCLUDED.industry,
                        us_chain_tickers = EXCLUDED.us_chain_tickers,
                        use_adr_premium = EXCLUDED.use_adr_premium,
                        is_core = TRUE;
                """, (sid, info.get("name"), info.get("industry"), us_chain, use_adr))
                    
        show_sovereign_dashboard({"count": len(stock_configs)})

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_sovereign_migration()