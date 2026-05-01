import os
import shutil
import logging
from pathlib import Path
from datetime import datetime, timedelta

# 設定日誌
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIRS = [
    BASE_DIR / "outputs",
    BASE_DIR / "training" / "outputs",
    BASE_DIR / "fetchers" / "logs",
]

def rotate_logs(keep_days=7):
    """
    清理舊日誌，將超過 keep_days 的 .log 檔案刪除或壓縮。
    """
    logger.info(f"=== 啟動日誌維護 (保留 {keep_days} 天) ===")
    now = datetime.now()
    cutoff = now - timedelta(days=keep_days)

    count = 0
    for log_dir in LOG_DIRS:
        if not log_dir.exists():
            continue
            
        logger.info(f"檢查目錄: {log_dir}")
        for f in log_dir.glob("*.log*"):
            # 取得檔案最後修改時間
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            if mtime < cutoff:
                try:
                    logger.info(f"  正在清理陳舊日誌: {f.name} (日期: {mtime.date()})")
                    f.unlink()
                    count += 1
                except Exception as e:
                    logger.error(f"  清理 {f.name} 失敗: {e}")
                    
    logger.info(f"=== 日誌維護完成，共清理 {count} 個檔案 ===")

if __name__ == "__main__":
    rotate_logs(keep_days=7)
