# Copyright (c) 2026 Antigravity Quant Research. All rights reserved.
# Proprietary and Confidential.
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Written by Antigravity Virtual Colleague for the 80-Stock Predictive Project.
# Description: 統一的模型載入與快取工具，支援 ensemble 模型的自動化加載 + 檔案鎖。
# Version: 3.0.0

"""
model_loader.py — 高效能模型載入器（含檔案鎖保護）

修改摘要（第三輪審查修復 P2 3.1）：
  原版：auto_train_manager.py 訓練中寫入 ensemble_{sid}.pkl 同時，
        automate_daily.py 的推論進程可能正在讀同一個檔案，
        造成 pickle.load 取到「半寫狀態」的損毀檔。
  本版：
    - load_ensemble_model() 使用 fcntl shared lock（讀鎖）
    - save_ensemble_model() 使用 fcntl exclusive lock（寫鎖），
      並透過 atomic-write（先寫 .tmp 再 rename）確保原子性
    - 兩者皆有超時保護，避免無限等待
    - Windows 平台無 fcntl，自動降級為「無鎖」模式並 logger.warning
"""

import logging
import os
import pickle
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import joblib

from config import MODEL_DIR

logger = logging.getLogger(__name__)

# 使用快取避免重複載入大型模型
_MODEL_CACHE: dict[str, Any] = {}

# fcntl 僅 Linux/macOS 提供；Windows 退化為 no-op
try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False
    logger.warning("fcntl 不可用（非 POSIX 平台），檔案鎖降級為 no-op")


# ─────────────────────────────────────────────
# 檔案鎖 context manager（[P2 修復 3.1]）
# ─────────────────────────────────────────────


@contextmanager
def _file_lock(file_obj, exclusive: bool = False, timeout: float = 30.0):
    """
    取得檔案鎖（fcntl flock）。
      exclusive=True → 寫鎖（X-lock，獨占）
      exclusive=False → 讀鎖（S-lock，共享）
    若超時則拋出 TimeoutError。Windows 自動降級為 no-op。
    """
    if not _HAS_FCNTL:
        yield
        return

    lock_type = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    start = time.time()
    while True:
        try:
            fcntl.flock(file_obj.fileno(), lock_type | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.time() - start > timeout:
                raise TimeoutError(f"檔案鎖等待逾時 ({timeout}s)")
            time.sleep(0.1)
    try:
        yield
    finally:
        try:
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            logger.warning(f"釋放檔案鎖失敗: {e}")


# ─────────────────────────────────────────────
# 載入 ensemble 模型（含讀鎖）
# ─────────────────────────────────────────────


def load_ensemble_model(stock_id: str, use_cache: bool = True) -> Optional[Any]:
    """
    載入指定個股的 Ensemble 模型 (XGB/LGB/TFT)，使用 shared lock 避免讀到
    訓練進程正在寫入的半寫狀態。

    Args:
        stock_id : 台灣股市代碼
        use_cache: 是否使用 in-process 快取（預設 True）

    Returns:
        載入的模型物件，若不存在則回傳 None。
    """
    global _MODEL_CACHE

    if use_cache and stock_id in _MODEL_CACHE:
        logger.debug(f"Using cached model for {stock_id}")
        return _MODEL_CACHE[stock_id]

    model_path = Path(MODEL_DIR) / f"ensemble_{stock_id}.pkl"
    if not model_path.exists():
        logger.warning(f"Model file not found: {model_path}")
        return None

    try:
        with open(model_path, "rb") as f:
            with _file_lock(f, exclusive=False, timeout=30.0):
                # joblib 比 pickle 對 sklearn / xgboost 物件更友善
                try:
                    model = joblib.load(f)
                except Exception:
                    f.seek(0)
                    model = pickle.load(f)

        if use_cache:
            _MODEL_CACHE[stock_id] = model
        logger.info(f"Successfully loaded model for {stock_id} from {model_path}")
        return model

    except TimeoutError as e:
        logger.error(f"[{stock_id}] 等待模型讀鎖逾時: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load model for {stock_id}: {e}")
        return None


# ─────────────────────────────────────────────
# 儲存 ensemble 模型（含寫鎖 + atomic write）
# ─────────────────────────────────────────────


def save_ensemble_model(stock_id: str, model: Any, atomic: bool = True) -> bool:
    """
    儲存 Ensemble 模型至 MODEL_DIR / ensemble_{stock_id}.pkl，使用 exclusive
    lock 防止其他進程同時寫入；同時 atomic-write 確保即使中途崩潰也不會
    產生半寫檔（讀進程要嘛看到舊版、要嘛看到新版，不會看到中間狀態）。

    Args:
        stock_id: 台灣股市代碼
        model   : 要儲存的模型物件
        atomic  : 是否使用 .tmp + rename 原子寫入（預設 True）

    Returns:
        True 表示成功，False 表示失敗（log 已記錄錯誤）。
    """
    target_path = Path(MODEL_DIR) / f"ensemble_{stock_id}.pkl"
    write_path = target_path.with_suffix(".pkl.tmp") if atomic else target_path

    try:
        write_path.parent.mkdir(parents=True, exist_ok=True)
        with open(write_path, "wb") as f:
            with _file_lock(f, exclusive=True, timeout=60.0):
                joblib.dump(model, f)
                f.flush()
                os.fsync(f.fileno())   # 強制刷新到磁碟，避免 atomic rename 後資料未落地

        if atomic:
            # POSIX rename 是原子操作；同 filesystem 內保證 swap 不會丟資料
            shutil.move(str(write_path), str(target_path))

        # 失效快取（讓下次 load 取到新版）
        _MODEL_CACHE.pop(stock_id, None)
        logger.info(f"Successfully saved model for {stock_id} to {target_path}")
        return True

    except TimeoutError as e:
        logger.error(f"[{stock_id}] 等待模型寫鎖逾時: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to save model for {stock_id}: {e}")
        # 清理失敗的 .tmp 檔
        if atomic and write_path.exists():
            try:
                write_path.unlink()
            except Exception:
                pass
        return False


# ─────────────────────────────────────────────
# 雜項
# ─────────────────────────────────────────────


def clear_cache() -> None:
    """清除全域模型快取，釋放記憶體。"""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    logger.info("Model cache cleared.")


# 向後相容：舊呼叫者可能 import load_model_with_lock
load_model_with_lock = load_ensemble_model
save_model_with_lock = save_ensemble_model
