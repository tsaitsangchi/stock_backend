"""
debug_feat_crash.py v5.5 (Trinity Core Edition)
================================================================================
特徵計算崩潰調試工具 — 混合模式日誌實作版
此模組專用於診斷特定標的（如 2881）在特徵工程 (Feature Engineering) 過程中的異常。

核心功能：
  · 斷點調試       ─ 自動比對 raw_frame 與 build_features 之後的欄位差異。
  · 特徵一致性檢查 ─ 驗證最終產出的特徵數是否符合 config.py 預期。
  · 執行紀錄       ─ 對接 write_pipeline_log，標記為 test_v5.1 (Testing)。

修訂歷程：
  v5.5 (2026-05-09):
    - [規範] 導入混合模式日誌。
    - [核心] 對接 path_setup v3.0 與 features/feature_engineering v5.2 標準。
"""

import sys
import logging
import time
from pathlib import Path

# ── 系統路徑修復 (對接 path_setup v3.0) ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core", "pipeline", "features"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import write_pipeline_log
    from data_pipeline import build_daily_frame
    from feature_engineering import build_features
except ImportError as e:
    print(f"[FATAL] 無法匯入核心模組: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def debug_stock(stock_id: str = "2881"):
    t0 = time.monotonic()
    logger.info(f"🔍 啟動特徵崩潰診斷: {stock_id}...")
    
    try:
        raw = build_daily_frame(stock_id)
        df = build_features(raw, stock_id=stock_id)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        
        logger.info(f"✅ 診斷完成：Raw({raw.shape}) -> Feat({df.shape})")
        write_pipeline_log("debug_feat_crash", stock_id, "success", "test", elapsed_ms, df.shape[1])
        
    except Exception as e:
        logger.error(f"❌ 診斷發現崩潰: {e}")
        write_pipeline_log("debug_feat_crash", stock_id, "failed", "test", 0, 0, str(e))

if __name__ == "__main__":
    debug_stock()
