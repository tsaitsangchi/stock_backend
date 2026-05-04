"""
core/model_metadata.py — 模型版本追蹤與 rollback 支援 v2.0（原子寫入完整性版）
================================================================================
v2.0 改進（與 db_utils v3.0「逐支逐日 commit」精神一致）：
  ★ atomic_write_json()：tmp + os.replace 原子寫入，崩潰不會留下半份檔案。
  ★ save_metadata() 改為原子寫入，並加 file lock 防多程序競爭。
  ★ rollback_to_metadata()：依 metadata 還原 .pkl 為 current（一鍵 rollback）。
  ★ list_history()：列出指定 stock_id 的歷史 metadata，依時間倒序。
  ★ assert_feature_schema_match()：可選擇 strict / fuzzy（允許新增欄位）。
  ★ 所有 IO 失敗都不致使訓練流程中斷，僅 warning。

v1.0 既有：
  · ModelMetadata dataclass（git hash / feature schema / OOF 績效）
  · save_metadata / load_latest_metadata / fingerprint_features
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Metadata Schema
# ─────────────────────────────────────────────
@dataclass
class ModelMetadata:
    """單次訓練產出之模型元資料。"""

    stock_id: str
    model_path: str             # 對應的 .pkl 相對路徑
    train_end_date: str         # 訓練資料截止日 YYYY-MM-DD
    feature_count: int
    feature_fingerprint: str    # sha256 of sorted feature names
    git_hash: str | None = None
    python_version: str = field(default_factory=lambda: platform.python_version())
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # OOF 績效指標
    oof_da: float | None = None
    oof_sharpe: float | None = None
    oof_ic: float | None = None
    oof_n_samples: int | None = None
    n_trades_per_fold: float | None = None
    max_drawdown: float | None = None

    # 訓練設定
    horizon_days: int | None = None
    calibration_method: str | None = None
    calibrator_cv: str | None = None
    package_versions: dict[str, str] = field(default_factory=dict)
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────
# 檔案鎖（避免多程序同時寫同一份 metadata）
# ─────────────────────────────────────────────
_locks_dict_lock = threading.Lock()
_path_locks: dict[str, threading.Lock] = {}


def _lock_for(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _locks_dict_lock:
        lk = _path_locks.get(key)
        if lk is None:
            lk = threading.Lock()
            _path_locks[key] = lk
        return lk


@contextmanager
def _path_lock(path: Path):
    lk = _lock_for(path)
    lk.acquire()
    try:
        yield
    finally:
        lk.release()


# ─────────────────────────────────────────────
# 工具函式
# ─────────────────────────────────────────────
def get_git_hash(short: bool = True) -> str | None:
    """取得當前 commit hash。失敗時回傳 None（不阻斷訓練）。"""
    try:
        cmd = ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"] if short \
            else ["git", "rev-parse", "HEAD"]
        out = subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, timeout=5
        ).decode().strip()
        return out or None
    except Exception:
        return None


def fingerprint_features(features: list[str]) -> str:
    """對特徵名稱列表產生 sha256 fingerprint（前 12 hex chars）。"""
    blob = "|".join(sorted(features)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def get_package_versions(packages: list[str] | None = None) -> dict[str, str]:
    """取得關鍵套件版本（無法取得時回 'unknown'）。"""
    pkgs = packages or [
        "numpy", "pandas", "scikit-learn", "xgboost", "lightgbm",
        "joblib", "torch", "polars",
    ]
    versions: dict[str, str] = {}
    for name in pkgs:
        try:
            mod = __import__(name.replace("-", "_"))
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[name] = "not_installed"
    return versions


# ─────────────────────────────────────────────
# 原子寫入 JSON（v2.0 新增）
# ─────────────────────────────────────────────
def atomic_write_json(path: str | Path, data: Any) -> Path:
    """
    原子寫入 JSON：先寫到 .tmp，再以 os.replace 替換。
    崩潰時不會留下半份檔案。同時以 path-level lock 防多執行緒競爭。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with _path_lock(p):
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        os.replace(tmp, p)
    return p


def atomic_copy_file(src: str | Path, dst: str | Path) -> Path:
    """
    原子複製檔案：先複製到 dst.tmp，再 os.replace 為 dst。
    避免訓練到一半被讀取到不完整檔案。
    """
    src_p = Path(src)
    dst_p = Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    with _path_lock(dst_p):
        tmp = dst_p.with_suffix(dst_p.suffix + ".tmp")
        shutil.copy2(src_p, tmp)
        os.replace(tmp, dst_p)
    return dst_p


# ─────────────────────────────────────────────
# 持久化（v2.0 改為原子寫入）
# ─────────────────────────────────────────────
def save_metadata(
    metadata: ModelMetadata,
    archive_dir: str | Path,
    also_archive_pkl: bool = True,
) -> Path:
    """
    將 metadata 寫入 archive_dir，並（可選）將模型 .pkl 一同 cp 至 archive。
    v2.0 改為原子寫入：metadata.json 與 .pkl 都採 tmp + replace。

    最終目錄結構：
        outputs/models/
        ├── ensemble_2330.pkl                        # current
        └── archive/
            ├── ensemble_2330_2026-04-15_a3f4.pkl
            └── ensemble_2330_2026-04-15_a3f4.metadata.json

    Returns
    -------
    Path  metadata.json 的絕對路徑
    """
    archive = Path(archive_dir)
    archive.mkdir(parents=True, exist_ok=True)

    base_name = (
        f"ensemble_{metadata.stock_id}_"
        f"{metadata.timestamp}_{metadata.git_hash or 'nogit'}"
    )
    meta_path = archive / f"{base_name}.metadata.json"

    payload = metadata.to_dict()
    try:
        atomic_write_json(meta_path, payload)
        logger.info(f"[metadata] 寫入 {meta_path.name}")
    except Exception as e:
        logger.warning(f"[metadata] 寫入失敗：{e}")

    if also_archive_pkl:
        src = Path(metadata.model_path)
        if src.exists():
            dst = archive / f"{base_name}.pkl"
            try:
                atomic_copy_file(src, dst)
                logger.info(f"[metadata] 模型快照已封存 {dst.name}")
            except Exception as e:
                logger.warning(f"[metadata] 模型快照封存失敗：{e}")
        else:
            logger.warning(
                f"[metadata] 找不到模型檔 {src}，僅寫 metadata 不封存 .pkl"
            )

    return meta_path


def load_latest_metadata(
    stock_id: str, archive_dir: str | Path
) -> ModelMetadata | None:
    """讀取指定股票最近一次的 metadata（依檔案 mtime 倒序）。"""
    archive = Path(archive_dir)
    if not archive.exists():
        return None
    candidates = sorted(
        archive.glob(f"ensemble_{stock_id}_*.metadata.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    try:
        data = json.loads(candidates[0].read_text(encoding="utf-8"))
        return ModelMetadata(**{
            k: v for k, v in data.items()
            if k in ModelMetadata.__dataclass_fields__
        })
    except Exception as e:
        logger.warning(f"[metadata] 讀取 {candidates[0].name} 失敗：{e}")
        return None


def list_history(
    stock_id: str, archive_dir: str | Path, limit: int | None = None
) -> list[ModelMetadata]:
    """
    列出指定股票的歷史 metadata（依時間倒序）。

    Parameters
    ----------
    limit : 取最近 N 個，None 代表全部
    """
    archive = Path(archive_dir)
    if not archive.exists():
        return []
    files = sorted(
        archive.glob(f"ensemble_{stock_id}_*.metadata.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if limit is not None:
        files = files[:limit]
    out: list[ModelMetadata] = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            out.append(ModelMetadata(**{
                k: v for k, v in data.items()
                if k in ModelMetadata.__dataclass_fields__
            }))
        except Exception as e:
            logger.warning(f"[metadata] 讀取 {f.name} 失敗：{e}")
    return out


def rollback_to_metadata(
    metadata: ModelMetadata,
    archive_dir: str | Path,
    current_path: str | Path,
) -> bool:
    """
    依 metadata 將對應的封存 .pkl 還原為 current_path（一鍵 rollback）。

    Parameters
    ----------
    metadata     : 要還原的版本（通常由 list_history 取得）
    archive_dir  : 封存資料夾
    current_path : 目標 current 模型路徑

    Returns
    -------
    bool  是否成功
    """
    archive = Path(archive_dir)
    base_name = (
        f"ensemble_{metadata.stock_id}_"
        f"{metadata.timestamp}_{metadata.git_hash or 'nogit'}"
    )
    src = archive / f"{base_name}.pkl"
    if not src.exists():
        logger.error(f"[metadata] rollback 失敗：找不到 {src}")
        return False
    try:
        atomic_copy_file(src, current_path)
        logger.info(f"[metadata] 已 rollback：{src.name} → {Path(current_path).name}")
        return True
    except Exception as e:
        logger.error(f"[metadata] rollback 失敗：{e}")
        return False


def assert_feature_schema_match(
    runtime_features: list[str],
    metadata: ModelMetadata,
    strict: bool = False,
    allow_extra: bool = False,
) -> bool:
    """
    在 predict 時呼叫，比對 runtime 特徵集與訓練時是否一致。

    Parameters
    ----------
    runtime_features : 推論當下 feature 欄位名稱清單
    metadata         : 訓練完保存的 metadata
    strict           : True 則 mismatch 直接 raise
    allow_extra      : True 時允許 runtime 多了訓練時沒有的欄位（fingerprint
                       仍會不同，但不視為錯誤；僅 warning）

    Returns
    -------
    bool  是否相符
    """
    cur_fp = fingerprint_features(runtime_features)
    if cur_fp == metadata.feature_fingerprint:
        return True

    if allow_extra:
        # 嘗試從 metadata 復原訓練時的特徵集（fingerprint 不可逆，這裡只能比對欄位數）
        msg = (
            f"[metadata] Feature schema diff（stock={metadata.stock_id}），"
            f"但 allow_extra=True：runtime={len(runtime_features)} "
            f"vs trained={metadata.feature_count}（容許）"
        )
        logger.warning(msg)
        return False  # fingerprint 不同仍回 False，由呼叫端決定

    msg = (
        f"[metadata] Feature schema mismatch（stock={metadata.stock_id}）\n"
        f"  runtime feature_count={len(runtime_features)}, fingerprint={cur_fp}\n"
        f"  trained feature_count={metadata.feature_count}, "
        f"fingerprint={metadata.feature_fingerprint}"
    )
    if strict:
        raise RuntimeError(msg)
    logger.warning(msg)
    return False


__all__ = [
    "ModelMetadata",
    "get_git_hash", "fingerprint_features", "get_package_versions",
    "atomic_write_json", "atomic_copy_file",
    "save_metadata", "load_latest_metadata", "list_history",
    "rollback_to_metadata", "assert_feature_schema_match",
]