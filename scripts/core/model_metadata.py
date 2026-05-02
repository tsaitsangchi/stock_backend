"""
core/model_metadata.py — 模型版本追蹤與 rollback 支援
======================================================
回應系統檢核報告 P3-2「模型 versioning + rollback」建議。

每次訓練完成存模型時，同步寫一份 metadata.json 至 outputs/models/archive/，
記錄訓練當下的：
  · git commit hash（讓問題可被精確 reproduce）
  · feature schema 版本（特徵欄位數量 + sha256 fingerprint）
  · 訓練資料截止日期（防範看未來偏差）
  · OOF 績效指標（DA / Sharpe / IC）
  · Python 與關鍵套件版本

支援 rollback：當本週模型表現劣於上週時，可由 metadata 快速找回上一檔
.pkl 並還原為 current。
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import shutil
import subprocess
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
    n_trades_per_fold: float | None = None  # 對應 P0-2 中波動高頻訊號警告
    max_drawdown: float | None = None

    # 訓練設定
    horizon_days: int | None = None
    calibration_method: str | None = None
    calibrator_cv: str | None = None        # "TimeSeriesSplit-5" / "OOF" / "none"
    package_versions: dict[str, str] = field(default_factory=dict)
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    """
    對特徵名稱列表產生 sha256 fingerprint（前 12 hex chars）。
    特徵集若有變動（新增/刪除/重命名），fingerprint 會立刻不同，
    可作為「模型 vs 推論時 feature schema 是否一致」的快速健檢。
    """
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
# 持久化
# ─────────────────────────────────────────────
def save_metadata(
    metadata: ModelMetadata,
    archive_dir: str | Path,
    also_archive_pkl: bool = True,
) -> Path:
    """
    將 metadata 寫入 archive_dir，並（可選）將模型 .pkl 一同 cp 至 archive。

    最終目錄結構：
        outputs/models/
        ├── ensemble_2330.pkl                        # current (覆蓋寫)
        └── archive/
            ├── ensemble_2330_2026-04-15_a3f4.pkl    # 歷史快照
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
    meta_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info(f"[metadata] 寫入 {meta_path.name}")

    if also_archive_pkl:
        src = Path(metadata.model_path)
        if src.exists():
            dst = archive / f"{base_name}.pkl"
            try:
                shutil.copy2(src, dst)
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
    """讀取指定股票最近一次的 metadata（依 timestamp 排序）。"""
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
        return ModelMetadata(**{k: v for k, v in data.items() if k in ModelMetadata.__dataclass_fields__})
    except Exception as e:
        logger.warning(f"[metadata] 讀取 {candidates[0].name} 失敗：{e}")
        return None


def assert_feature_schema_match(
    runtime_features: list[str],
    metadata: ModelMetadata,
    strict: bool = False,
) -> bool:
    """
    在 predict 時呼叫，比對 runtime 特徵集與訓練時是否一致。

    Parameters
    ----------
    runtime_features : 推論當下構建出的 feature 欄位名稱清單
    metadata         : 訓練完保存的 metadata
    strict           : True 則 mismatch 直接 raise；False 僅警告

    Returns
    -------
    bool  是否相符
    """
    cur_fp = fingerprint_features(runtime_features)
    if cur_fp == metadata.feature_fingerprint:
        return True

    msg = (
        f"[metadata] Feature schema mismatch（stock={metadata.stock_id}）\n"
        f"  runtime feature_count={len(runtime_features)}, fingerprint={cur_fp}\n"
        f"  trained feature_count={metadata.feature_count}, fingerprint={metadata.feature_fingerprint}"
    )
    if strict:
        raise RuntimeError(msg)
    logger.warning(msg)
    return False
