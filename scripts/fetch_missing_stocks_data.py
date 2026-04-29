"""
fetch_missing_stocks_data.py — 自動補齊 config.py 中新增個股的歷史數據
[v3 Trinity Edition]

修改摘要：
1. 強化缺失偵測：從單純檢查 stock_price (count < 100)，升級為多表聯動校驗。
2. 引入效能機制：利用 data_integrity_check 的邏輯，精確識別需要補件的標的。
3. 智能補抓觸發：只要核心表 (price_adj, tech, fundamental) 任一出現嚴重缺漏，即觸發該標的的全量補抓。
"""

import subprocess
import sys
import logging
import psycopg2
from pathlib import Path

import json
# 注入路徑並引入核心配置
sys.path.append(str(Path(__file__).resolve().parent))
from config import STOCK_CONFIGS, DB_CONFIG
from data_integrity_audit import IntegrityAuditor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).parent
VENV_PYTHON = str(SCRIPTS_DIR.parent / "venv" / "bin" / "python3")

# 需要精確檢查的「錨點資料表」
ANCHOR_TABLES = [
    "stock_price",
    "price_adj",
    "institutional_investors_buy_sell"
]

# 個股相關補件腳本及其對應的 ID 參數名稱
PER_STOCK_SCRIPTS_CONFIG = {
    "fetch_technical_data.py":    "--stock-id",
    "fetch_price_adj_data.py":    "--stock-id",
    "fetch_fundamental_data.py":  "--stock-id",
    "fetch_chip_data.py":         "--stock-id",
    "fetch_derivative_data.py":   "--ids",
    "fetch_international_data.py": "--tickers",
    "fetch_advanced_chip_data.py": "--stock-id",
    "fetch_sponsor_chip_data.py":  "--stock-id",
    "fetch_event_risk_data.py":    "--stock-id",
    "fetch_cash_flows_data.py":    "--stock-id",
}

# 宏觀資料腳本
MACRO_SCRIPTS = [
    "fetch_macro_data.py",
    "fetch_fred_data.py",
    "fetch_extended_derivative_data.py",
    "fetch_derivative_sentiment_data.py"
]

def run_script(script_name: str, args: list[str] = []) -> bool:
    script_path = str(SCRIPTS_DIR / script_name)
    if not Path(script_path).exists():
        logger.error(f"找不到腳本: {script_path}")
        return False
        
    cmd = [VENV_PYTHON, script_path] + args
    logger.info(f"🚀 執行指令: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        logger.error(f"❌ 執行 {script_name} 失敗: {e}")
        return False

def get_missing_data_manifest() -> list[dict]:
    """
    [v5] 執行深度審計並取得斷層清單。
    """
    auditor = IntegrityAuditor(days_window=1000)
    manifest_path = "outputs/integrity_gaps.json"
    auditor.dump_gaps_json(manifest_path)
    
    if not Path(manifest_path).exists():
        return []
        
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    logger.info("=== [Phase 2] 精確補件管線啟動 ===")
    
    # 1. 基礎資訊更新
    logger.info("Step 1: 更新 stock_info...")
    run_script("fetch_stock_info.py")

    # 2. 深度完整性掃描
    gap_manifest = get_missing_data_manifest()
    
    if not gap_manifest:
        logger.info("✅ 所有標的核心資料皆已完整，無需執行補件。")
    else:
        logger.info(f"發現 {len(gap_manifest)} 處資料斷層，啟動精確補件...")
        
        # 3. 根據清單進行精確補件
        # 將清單按 stock_id 分組
        from collections import defaultdict
        grouped_gaps = defaultdict(list)
        for gap in gap_manifest:
            grouped_gaps[gap["stock_id"]].append(gap)

        for sid, gaps in grouped_gaps.items():
            # 找出這支股票所有表中最原始的斷層點
            earliest_gap = min([g["gap_start"] for g in gaps])
            logger.info(f"\n>>> 開始補齊 {sid} 的歷史斷層 (起點: {earliest_gap})")
            
            for script, id_flag in PER_STOCK_SCRIPTS_CONFIG.items():
                # 根據不同腳本使用正確的參數名稱 (--stock-id, --ids, --tickers)
                run_script(script, args=[id_flag, sid, "--start", earliest_gap])

    # 4. 宏觀資料更新
    logger.info("\nStep 4: 更新全域宏觀資料...")
    for script in MACRO_SCRIPTS:
        run_script(script)

    logger.info("\n✨ 補件任務全數完成。")

if __name__ == "__main__":
    main()
