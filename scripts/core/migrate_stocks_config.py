"""
migrate_stocks_config.py v6.1 (Quantum Finance Edition)
================================================================================
資產配置遷移工具 — 混合日誌標準版 (Quantum v5.1 標準)
將 config.py 中的核心資產清單同步至資料庫，並清理冗餘標的，提供執行後狀態回報。

修訂歷程：
  v6.1 (2026-05-11): [標準化] 導入 Quantum 標準檔頭、生命週期監測器與執行後儀表板回報。
  v6.0 (2026-05-10): [文檔] 補齊「核心股」與「強制清理」執行範例矩陣。

執行範例 (Comprehensive Usage Examples):
  1. [資產同步] 同步 config.py 中的 128 檔核心資產並查看回報:
     python scripts/core/migrate_stocks_config.py

  2. [核心股 + 所有表] 配合後續抓取任務 (示意):
     # 同步完後，可配合 db_utils 獲取清單
     from core.db_utils import get_db_stock_ids
     stocks = get_db_stock_ids(active_only=True)

  3. [單一表重置] 若需強制更新特定資產的抓取旗標 (SQL):
     UPDATE stocks SET fetch_basic = TRUE WHERE stock_id = '2330';

  4. [系統監測] 查看遷移任務的執行歷史與耗時 (SQL):
     SELECT * FROM pipeline_execution_log WHERE task_name = 'migrate_stocks_config' ORDER BY created_at DESC;
================================================================================
"""
import sys, logging, time
from pathlib import Path
from datetime import datetime

# ── 終極路徑自癒 Bootstrap ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_SCRIPTS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR.parent))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
except ImportError:
    import path_setup
    path_setup.ensure_scripts_on_path(__file__)

try:
    from core.db_utils import db_transaction, write_pipeline_log, record_lifecycle, write_data_audit_log
except ImportError:
    from db_utils import db_transaction, write_pipeline_log, record_lifecycle, write_data_audit_log

try:
    from config import STOCK_CONFIGS
except ImportError:
    from scripts.config import STOCK_CONFIGS

def show_migration_report():
    """執行後的儀表板回報，顯示目前資產庫狀態。"""
    print("\n" + "="*50)
    print("📊 Quantum Finance: 資產清單同步報告")
    print("="*50)
    try:
        with db_transaction() as cur:
            cur.execute("SELECT count(*) as total, count(*) FILTER (WHERE is_core = TRUE) as core FROM stocks;")
            res = cur.fetchone()
            print(f"✅ 同步狀態  : 執行完成")
            print(f"📈 總資產數  : {res['total']} 檔")
            print(f"⭐ 核心資產  : {res['core']} 檔 (已標記為核心)")
            
            # 檢查最近一筆日誌
            cur.execute("SELECT duration_ms FROM pipeline_execution_log WHERE task_name = 'migrate_stocks_config' ORDER BY created_at DESC LIMIT 1;")
            log = cur.fetchone()
            if log:
                print(f"⏱️ 任務耗時  : {log['duration_ms']} ms")
        print("="*50 + "\n")
    except Exception as e:
        print(f"⚠️ 無法產生報表: {e}")

def migrate_core_assets():
    """執行資產遷移與元數據清理。"""
    logging.info(f"🔄 正在同步核心資產 ({len(STOCK_CONFIGS)} 檔) 並清理元數據...")
    
    # 使用混合模式生命週期監測器
    with record_lifecycle("migrate_stocks_config", category="sys", stock_id="SYSTEM"):
        with db_transaction() as cur:
            # 1. 預防性重置 (確保只有 config.py 中存在的標的是活躍的)
            cur.execute("""
                UPDATE stocks SET 
                    is_active = FALSE, is_core = FALSE,
                    fetch_basic = FALSE, fetch_chip = FALSE,
                    fetch_fundamental = FALSE, fetch_derivative = FALSE,
                    fetch_news = FALSE
            """)
            
            # 2. 啟動核心標的並更新資訊
            for sid, cfg in STOCK_CONFIGS.items():
                cur.execute("""
                    UPDATE stocks SET 
                        is_active = TRUE, is_core = TRUE, 
                        fetch_basic = TRUE, fetch_chip = TRUE,
                        fetch_fundamental = TRUE, fetch_derivative = TRUE,
                        fetch_news = TRUE,
                        stock_name = %s, industry = %s 
                    WHERE stock_id = %s
                """, (cfg['name'], cfg.get('industry','N/A'), sid))
                
                if cur.rowcount == 0:
                    cur.execute("""
                        INSERT INTO stocks (stock_id, stock_name, industry, is_active, is_core, fetch_basic, fetch_chip, fetch_fundamental, fetch_derivative, fetch_news)
                        VALUES (%s, %s, %s, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE)
                    """, (sid, cfg['name'], cfg.get('industry','N/A')))
            
            # 3. 清理完全不屬於核心的冗餘資料 (混合模式：清理也是一種變動，記錄到 Audit)
            cur.execute("DELETE FROM stocks WHERE is_active = FALSE")
            deleted_count = cur.rowcount
            
        # 專屬分類紀錄：資產異動審計
        write_data_audit_log("stocks", "SYSTEM", datetime.now().strftime("%Y-%m-%d"), datetime.now().strftime("%Y-%m-%d"), len(STOCK_CONFIGS))
        logging.info("✅ 核心資產元數據同步完成")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    migrate_core_assets()
    # 執行完畢後主動顯示結果儀表板
    show_migration_report()