"""
model_loader.py — 安全的模型讀寫工具 (File Locking)
==============================================
防止 auto_train_manager.py 寫入模型時，predict.py 同時讀取導致損毀。
採用 fcntl (Linux) 實施建議的檔案鎖。
"""

import os
import joblib
import fcntl
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

def safe_load(path: Path | str) -> Any:
    """ 使用共享鎖 (LOCK_SH) 安全讀取模型 """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"模型檔案不存在: {path}")

    start_time = time.time()
    with open(path, "rb") as f:
        # 獲取共享鎖（多個進程可以同時讀，但寫入時會被阻斷）
        try:
            fcntl.flock(f, fcntl.LOCK_SH)
            model = joblib.load(f)
            return model
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
            duration = time.time() - start_time
            if duration > 0.5:
                logger.debug(f"[Lock] 讀取模型 {path.name} 耗時 {duration:.2f}s (等待鎖)")

def safe_dump(model: Any, path: Path | str):
    """ 使用互斥鎖 (LOCK_EX) 安全寫入模型 """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    # 先寫入臨時檔案，再 rename，這是最安全的原子操作
    temp_path = path.with_suffix(".tmp")
    
    with open(temp_path, "wb") as f:
        try:
            # 獲取互斥鎖
            fcntl.flock(f, fcntl.LOCK_EX)
            joblib.dump(model, f)
            # 確保資料寫入磁碟
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

    # 原子替換
    os.replace(temp_path, path)
    duration = time.time() - start_time
    logger.info(f"[Lock] 模型已安全儲存至 {path.name} (耗時 {duration:.2f}s)")
