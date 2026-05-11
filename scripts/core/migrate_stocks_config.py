"""
migrate_stocks_config.py v6.6 (Quantum Finance Edition)
================================================================================
資產主權化遷移引擎 — 全方位標定版 (Quantum v5.2 標準)
負責將 config.py 中的靜態元數據遷移至資料庫，達成系統治理的主權化。

修訂歷程：
  v6.6 (2026-05-11): [標準] 對齊 db_utils v2.26，支援單一標定更新與全量強制更新。
  v6.5 (2026-05-11): [環境] 修正 .env 加載路徑，確保與根目錄配置對齊。

【執行範例矩陣 (Migration Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / 用法                                        │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [全系統標的遷移]          │ $ python scripts/core/migrate_stocks_config.py         │
│ 2. [單一核心標的更新]        │ $ python scripts/core/migrate_stocks_config.py --id 2330│
│ 3. [強制全量屬性同步]        │ $ python scripts/core/migrate_stocks_config.py --force  │
└──────────────────────────────┴────────────────────────────────────────────────────────┘
================================================================================
"""
import os, sys, logging, platform, argparse
from pathlib import Path

# ── 系統級架構引導 ──
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from core.db_utils import db_transaction, record_lifecycle, ensure_infrastructure, bulk_upsert

def run_migration(target_id: str = None, force: bool = False):
    """執行主權化遷移任務"""
    # 延遲導入 config 避免循環依賴
    import config
    stock_configs = getattr(config, "STOCK_CONFIGS", {})
    
    # 1. 過濾目標
    if target_id:
        if target_id not in stock_configs:
            print(f"❌ 錯誤：標的 {target_id} 不存在於 config.py 中。")
            return
        to_migrate = {target_id: stock_configs[target_id]}
    else:
        to_migrate = stock_configs

    # 2. 準備遷移數據
    migration_data = []
    for sid, info in to_migrate.items():
        migration_data.append({
            "stock_id": sid,
            "name": info.get("name", "Unknown"),
            "industry": info.get("industry", "N/A"),
            "us_chain_tickers": ",".join(info.get("us_chain_tickers", [])),
            "use_adr_premium": info.get("use_adr_premium", False),
            "is_core": True,
            "is_active": True
        })

    # 3. 執行寫入
    with record_lifecycle(f"sovereign_migration_{target_id or 'all'}", "maintenance", target_id or "SYSTEM"):
        rows = bulk_upsert("stocks", migration_data, ["stock_id"])
        
    print("\n" + "🏛️"*40)
    print(f"🚀 Quantum Finance: 資料庫主權化遷移報告 (v6.6)")
    print("🏛️"*40)
    print(f"✅ 執行結果  : SUCCESS")
    print(f"📊 遷移規模  : {len(migration_data)} 檔標的")
    print(f"🎯 目標模式  : {'SINGLE' if target_id else 'FULL'}")
    print(f"💎 影響行數  : {rows} 行已同步")
    print("-" * 80)
    print("🟢 遷移成功：資料庫主權狀態已更新。")
    print("📝 任務同步: pipeline_execution_log (sovereign_migration)")
    print("🏛️"*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", help="指定遷移單一標的 ID (例如 2330)")
    parser.add_argument("--force", action="store_true", help="強制更新所有屬性")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    ensure_infrastructure()
    run_migration(target_id=args.id, force=args.force)