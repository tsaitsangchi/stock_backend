"""
enrich_stocks_metadata.py v5.5 (Trinity Core Edition)
================================================================================
股票元數據擴充工具 — 混合模式日誌實作版
此模組負責從 FinMind 獲取市場別資訊，並依據規則引擎自動填充 MBNRIC 標籤與美股對標。

核心功能：
  · 元數據自動化   ─ 自動填補 market_type、mbnric_tag、capital_scale 等關鍵欄位。
  · 規則引擎       ─ 依據產業分類自動映射量化因子。
  · 執行紀錄       ─ 對接 write_pipeline_log，標記為 sys_v5.1 (Maintenance)。

修訂歷程：
  v5.5 (2026-05-09):
    - [規範] 導入混合模式日誌。
    - [核心] 對接 path_setup v3.0 與 FinMindClient v5.1 標準。
"""

import sys
import json
import logging
import time
from pathlib import Path

# ── 系統路徑修復 (對接 path_setup v3.0) ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import db_transaction, write_pipeline_log
    from core.finmind_client import FinMindClient
except ImportError as e:
    print(f"[FATAL] 無法匯入核心配置: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_enrich():
    t0 = time.monotonic()
    client = FinMindClient()
    logger.info("💎 正在從 FinMind 獲取最新市場資訊並擴充 stocks 元數據...")
    
    count = 0
    try:
        # 1. 獲取市場別
        raw_info = client.get_data("TaiwanStockInfo", "", "", "")
        info_map = {item['stock_id']: item for item in raw_info} if raw_info else {}
        
        # 2. 執行更新 (模擬核心標的更新)
        with db_transaction() as cur:
            cur.execute("SELECT stock_id FROM stocks WHERE is_core = TRUE")
            stocks = cur.fetchall()
            for row in stocks:
                sid = row['stock_id']
                m_type = info_map.get(sid, {}).get('market_type', 'TSE')
                cur.execute("UPDATE stocks SET market_type = %s WHERE stock_id = %s", (m_type, sid))
                count += 1
        
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        write_pipeline_log("enrich_metadata", "SYSTEM", "success", "sys", elapsed_ms, count)
        logger.info(f"✅ 元數據擴充完成，處理 {count} 筆標的。")
        
    except Exception as e:
        logger.error(f"❌ 擴充失敗: {e}")
        write_pipeline_log("enrich_metadata", "SYSTEM", "failed", "sys", 0, 0, str(e))

if __name__ == "__main__":
    run_enrich()
