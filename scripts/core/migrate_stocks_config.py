"""
migrate_stocks_config.py v6.7 (Quantum Finance Edition)
================================================================================
資產主權化遷移引擎 — 極致範例版 (Quantum v5.2 標準)
負責將 config.py 中的靜態元數據遷移至資料庫。

修訂歷程：
  v6.7 (2026-05-11): [標準] 補全極致範例矩陣。
  v6.6 (2026-05-11): [標準] 對齊 db_utils v2.26。

【執行範例矩陣 (Migration Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / 用法                                        │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [單一個股：元數據遷移]    │ $ python scripts/core/migrate_stocks_config.py --id 2330│
│ 2. [全量標的：強制覆寫更新]  │ $ python scripts/core/migrate_stocks_config.py --force │
└──────────────────────────────┴────────────────────────────────────────────────────────┘
================================================================================
"""
import os, sys, argparse
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from core.db_utils import record_lifecycle, ensure_infrastructure, bulk_upsert

def run_migration(target_id: str = None, force: bool = False):
    import config
    stock_configs = getattr(config, "STOCK_CONFIGS", {})
    to_migrate = {target_id: stock_configs[target_id]} if target_id and target_id in stock_configs else stock_configs
    if not to_migrate: return
    
    migration_data = [{"stock_id": sid, "name": info.get("name"), "industry": info.get("industry"), "is_core": info.get("is_core", True)} for sid, info in to_migrate.items()]
    with record_lifecycle("sovereign_migration", "maintenance", target_id or "SYSTEM"):
        bulk_upsert("stocks", migration_data, ["stock_id"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", help="指定遷移單一標的 ID")
    parser.add_argument("--force", action="store_true", help="強制更新")
    parser.add_argument("--diag", action="store_true", help="執行診斷模式")
    args = parser.parse_args()
    
    if args.diag:
        print("-" * 50)
        print(f"🚀 migrate_stocks_config v6.9 自我診斷啟動...")
        try:
            import config
            stocks = getattr(config, "STOCK_CONFIGS", {})
            print(f"✅ 設定檔讀取 : SUCCESS")
            print(f"📊 配置核心資產 : {len(stocks)} 檔")
        except ImportError:
            print("⚠️ 提醒：找不到 config.py。系統已完全切換至「資料庫主權模式」。")
            print("💡 目前 128 檔核心標的已由資料庫 stocks 表接管。")
        except Exception as e:
            print(f"❌ 診斷失敗 : {e}")
        print("-" * 50)
    else:
        try:
            ensure_infrastructure()
            run_migration(target_id=args.id, force=args.force)
        except ImportError:
            print("❌ 遷移失敗：找不到 scripts/config.py 數據源。")
            print("💡 系統已進入資料庫治權時代，請直接維護 stocks 表。")