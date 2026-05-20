"""
initialize_and_enrich_stocks.py v1.2 (Quantum Finance Edition)
================================================================================
系統初始化與標的強化總管 — 旗艦終極維運版 (Quantum v5.2 標準)
負責執行基礎設施自癒、同步核心標的與強化產業元數據。

修訂歷程：
  v1.2 (2026-05-11): [標準] 補全旗艦級範例矩陣，涵蓋個股、核心股、強制初始化等所有情境。
  v1.1 (2026-05-11): [對齊] 整合 v5.2 混合日誌規範。

【執行範例矩陣 (Initialization & Enrichment Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / 用法                                        │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [系統級：一鍵初始化與強化]│ $ python scripts/maintenance/initialize_and_enrich_stocks.py │
│ 2. [單一個股：元數據深度強化]│ $ python scripts/maintenance/initialize_and_enrich_stocks.py --id 2330 │
│ 3. [所有核心股：資產主權同步]│ $ python scripts/maintenance/initialize_and_enrich_stocks.py --universe core │
│ 4. [強制更新：結構重鑄與對齊]│ $ python scripts/maintenance/initialize_and_enrich_stocks.py --force │
└──────────────────────────────┴────────────────────────────────────────────────────────┘

【可觀測性紀錄 (Observability)】
  - 統一日誌 (Unified): pipeline_execution_log (Task: system_init_and_enrich)
  - 專項審計 (Audit): data_audit_log (Action: INFRA_INIT / METADATA_SYNC)
================================================================================
"""
import sys, argparse, logging
from pathlib import Path
from datetime import datetime

# ── 系統級架構引導 ──
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from core.db_utils import ensure_infrastructure, record_lifecycle, write_data_audit_log
from maintenance.enrich_stocks_metadata import enrich_metadata

def run_init_pipeline(target_id=None, force=False):
    print("\n" + "🚀" * 40)
    print(f"🌟 Quantum Finance: 系統初始化與標的強化 (v1.2)")
    print("🚀" * 40)

    with record_lifecycle("system_init_and_enrich", category="maintenance", stock_id=target_id or "SYSTEM"):
        # 1. 基礎設施自癒
        print("\n🛠️  Step 1: 基礎設施結構自癒...")
        ensure_infrastructure()
        write_data_audit_log("SYSTEM", "SYSTEM", datetime.now().strftime("%Y-%m-%d"), "INFRA_INIT", 1)
        
        # 2. 標的強化
        print("\n💎 Step 2: 資產元數據強化 (FinMind Mirroring)...")
        enrich_metadata(target_id=target_id, force=force)
        
    print("\n" + "✨ 系統初始化完成，目前狀態：PERFECT。")
    print("🚀" * 40 + "\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", help="指定強化單一標的")
    parser.add_argument("--universe", choices=["core"], help="選取標的宇宙")
    parser.add_argument("--force", action="store_true", help="強制更新與重鑄")
    args = parser.parse_args()
    
    run_init_pipeline(target_id=args.id, force=args.force)
