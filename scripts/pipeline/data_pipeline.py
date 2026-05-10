"""
data_pipeline.py v6.7 (Trinity Core Final)
================================================================================
修訂歷程：
  v6.7 (2026-05-10): [修正] 強化路徑自癒 Bootstrap，解決 No module named 'core'。
"""
import sys, logging, time, argparse
from pathlib import Path

# ── 終極路徑自癒 Bootstrap ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS = None
for p in [_THIS_DIR, _THIS_DIR.parent, _THIS_DIR.parent.parent]:
    if p.name == "scripts" or (p / "scripts").exists():
        _SCRIPTS = p if p.name == "scripts" else (p / "scripts")
        break
if _SCRIPTS:
    if str(_SCRIPTS) not in sys.path: sys.path.insert(0, str(_SCRIPTS))
    if str(_SCRIPTS.parent) not in sys.path: sys.path.insert(0, str(_SCRIPTS.parent))

try:
    from core.db_utils import get_db_stock_ids, write_pipeline_log
    from ingestion.parallel_fetch import fetch_stock_data_unit
except ImportError:
    try:
        from db_utils import get_db_stock_ids, write_pipeline_log
        from parallel_fetch import fetch_stock_data_unit
    except ImportError:
        print("[FATAL] 無法匯入核心配置，請確認 PYTHONPATH 或 scripts 目錄存在。")
        sys.exit(1)

def run_master_pipeline(stock_id: str = None, start_date: str = None):
    t_start = time.monotonic(); target_ids = [stock_id] if stock_id else get_db_stock_ids()
    logging.info(f"🚀 [Pipeline] 同步開始 (共 {len(target_ids)} 檔)...")
    success_count = 0; total_rows = 0
    for sid in target_ids:
        try:
            rows = fetch_stock_data_unit(sid, start_date=start_date)
            success_count += 1; total_rows += (rows or 0)
        except Exception as e: logging.error(f"❌ {sid} 同步異常: {e}")
    write_pipeline_log("data_pipeline_master", "SYSTEM", "success", "pipeline", int((time.monotonic()-t_start)*1000), total_rows)
    logging.info(f"🏁 [Pipeline] 完成！成功: {success_count}/{len(target_ids)}，總量: {total_rows}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(); parser.add_argument("--stock_id", type=str); parser.add_argument("--start", type=str); parser.add_argument("--all", action="store_true"); args = parser.parse_args()
    run_master_pipeline(stock_id=args.stock_id, start_date=args.start)
