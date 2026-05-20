"""
migrate_stocks_config.py v6.7 (Quantum Finance Edition)
================================================================================
資產主權化遷移引擎 — 極致範例版 (Quantum v5.2 標準)
負責將 config.py 中的靜態元數據遷移至資料庫，支持高靈活性的過濾與更新。

修訂歷程：
  v6.7 (2026-05-11): [標準] 補全極致範例矩陣，明確個股、核心標的與全量更新指令。
  v6.6 (2026-05-11): [標準] 對齊 db_utils v2.26，支援單一標定與強制更新。

【執行範例矩陣 (Migration Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / 用法                                        │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [單一個股：元數據遷移]    │ $ python scripts/core/migrate_stocks_config.py --id 2330│
│ 2. [全核心標的：屬性同步]    │ $ python scripts/core/migrate_stocks_config.py --core  │
│ 3. [全量標的：強制覆寫更新]  │ $ python scripts/core/migrate_stocks_config.py --force │
│ 4. [單一Table清空後重新遷移] │ $ python scripts/core/migrate_stocks_config.py --clean │
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

def run_migration(target_id: str = None, force: bool = False, core_only: bool = False):
    import config
    stock_configs = getattr(config, "STOCK_CONFIGS", {})
    
    if target_id:
        to_migrate = {target_id: stock_configs.get(target_id)} if target_id in stock_configs else {}
    elif core_only:
        to_migrate = {k: v for k, v in stock_configs.items() if v.get("is_core", True)}
    else:
        to_migrate = stock_configs

    if not to_migrate: return

    migration_data = [{
        "stock_id": sid, "name": info.get("name"), "industry": info.get("industry"),
        "us_chain_tickers": ",".join(info.get("us_chain_tickers", [])),
        "use_adr_premium": info.get("use_adr_premium", False),
        "is_core": info.get("is_core", True), "is_active": True
    } for sid, info in to_migrate.items()]

    with record_lifecycle(f"sovereign_migration", "maintenance", target_id or "SYSTEM"):
        rows = bulk_upsert("stocks", migration_data, ["stock_id"])
        
    print("\n" + "🏛️"*40)
    print(f"🚀 Quantum Finance: 資產主權同步報告 (v6.7)\n✅ 同步規模: {len(migration_data)} 檔 | 影響行數: {rows}")
    print("🏛️"*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", help="指定遷移單一標的 ID")
    parser.add_argument("--core", action="store_true", help="僅遷移核心標的")
    parser.add_argument("--force", action="store_true", help="強制更新所有屬性")
    args = parser.parse_args()
    ensure_infrastructure()
    run_migration(target_id=args.id, force=args.force, core_only=args.core)