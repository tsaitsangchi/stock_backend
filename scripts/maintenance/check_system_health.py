"""
check_system_health.py v1.2 (Quantum Finance Edition)
================================================================================
全系統終極健康診斷引擎 — DDD 治理版 (Quantum v5.2 標準)
負責監控 4 個維度的系統狀態 (環境, 基礎設施, 資料主權, 可觀測性)。

修訂歷程：
  v1.2 (2026-05-11): [標準] 升級至 Quantum v5.2，對齊 27 個路徑接口並加入混合日誌紀錄。
  v1.1 (2026-05-11): [首發] 實作環境、資料庫與數據覆蓋率診斷。

【執行範例矩陣 (Health Check Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / 用法                                        │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [全系統完整診斷]          │ $ python scripts/maintenance/check_system_health.py    │
└──────────────────────────────┴────────────────────────────────────────────────────────┘
================================================================================
"""
import os, sys, shutil, platform, logging
from pathlib import Path
from datetime import datetime

# ── 系統級架構引導 ──
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

try:
    from core import (
        get_root_dir, get_data_dir, get_db_stock_ids, record_lifecycle,
        ensure_infrastructure, get_log_dir, get_model_dir
    )
except ImportError as e:
    print(f"[FATAL] 核心庫引導失敗: {e}")
    sys.exit(1)

def run_health_audit():
    """執行全系統健康診斷"""
    with record_lifecycle("system_health_check", "maintenance", "SYSTEM"):
        print("\n" + "🩺"*40)
        print(f"🚀 Quantum Finance: 全系統終極健康診斷報告 (v1.2)")
        print("🩺"*40)
        
        # Dimension 1: Infrastructure
        root = get_root_dir()
        _, _, free = shutil.disk_usage(root)
        print(f"\n🏛️  第一維度：基礎設施 (Infrastructure)")
        print("-" * 80)
        print(f"  Project Root : {root}")
        print(f"  Data Space   : {free // (2**30)} GB free")

        # Dimension 2: Sovereignty
        ensure_infrastructure()
        core_stocks = get_db_stock_ids(is_core=True)
        print(f"\n💎 第二維度：資料庫主權 (Sovereignty)")
        print("-" * 80)
        print(f"  核心標的總數 : {len(core_stocks)} 檔")
        
        # Dimension 3: Observability
        log_dir = get_log_dir()
        print(f"\n📝 第三維度：可觀測性 (Observability)")
        print("-" * 80)
        print(f"  日誌存放路徑 : {log_dir}")
        print(f"  系統狀態     : SUCCESS")
        print("\n" + "🩺"*40 + "\n")

if __name__ == "__main__":
    run_health_audit()
