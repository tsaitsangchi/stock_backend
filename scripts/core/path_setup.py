"""
path_setup.py v2.7 (Quantum Finance Edition)
================================================================================
系統路徑配置器 — 混合日誌與數據治理版 (Quantum v5.1 標準)
整合 Active Symlinks 指標、LRU 自動清理機制，並將治理事件同步至 Hybrid Logs。

修訂歷程：
  v2.7 (2026-05-10): [核心] 導入混合模式日誌，將治理事件同步至 pipeline_execution_log。
  v2.6 (2026-05-10): [修正] 補回遺失的 get_archive_dir 與 get_checkpoints_dir。

【執行範例矩陣 — 數據治理方案】
1. 更新最新模型指標 (Python)：
   update_latest_link("archive/model_2330_v1.pkl", "2330_latest.pkl")
2. 清理過期日誌 (Python)：
   cleanup_directory(get_logs_dir(), max_size_gb=1.0, keep_days=30)
3. 查看目前路徑對應：
   python scripts/core/path_setup.py
================================================================================
"""
import sys
import os
import time
import shutil
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

# ── 終極路徑自癒 Bootstrap (核心自救版) ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_SCRIPTS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR.parent))

# ── 環境變數與抽象層載入 ──
try:
    from dotenv import load_dotenv
    _ENV_PATH = _SCRIPTS_DIR.parent / ".env"
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH)
    else:
        load_dotenv()
except ImportError:
    pass

try:
    import fsspec
except ImportError:
    fsspec = None

_scripts_dir_cache: Optional[Path] = None

def _resolve_scripts_dir(caller_file: str) -> Path:
    current_path = Path(caller_file).resolve()
    for parent in [current_path] + list(current_path.parents):
        if parent.name == "scripts":
            return parent
        if (parent / "core").is_dir() and (parent / "fetchers").is_dir():
            return parent
        if (parent / "scripts").is_dir():
            return parent / "scripts"
    return current_path.parent

def get_scripts_dir(caller_file: str = __file__) -> Path:
    global _scripts_dir_cache
    if _scripts_dir_cache is None:
        _scripts_dir_cache = _resolve_scripts_dir(caller_file)
    return _scripts_dir_cache

def get_fs_and_path(path_str: str) -> Tuple[Any, str]:
    if fsspec is None:
        return None, path_str
    fs, path = fsspec.core.url_to_fs(path_str)
    return fs, path

def get_env_path(key: str, default_subpath: str, caller_file: str = __file__) -> str:
    scripts_dir = get_scripts_dir(caller_file)
    env_val = os.getenv(key)
    if env_val:
        return env_val
    data_root = os.getenv("TRINITY_DATA_ROOT")
    if data_root:
        if "://" in data_root:
            return f"{data_root.rstrip('/')}/{default_subpath}"
        target_path = Path(data_root).resolve() / default_subpath
    else:
        target_path = scripts_dir / default_subpath
    if "://" not in str(target_path):
        Path(target_path).mkdir(parents=True, exist_ok=True)
    return str(target_path)

def ensure_scripts_on_path(caller_file: str):
    scripts_dir = get_scripts_dir(caller_file)
    scripts_str = str(scripts_dir)
    if scripts_str not in sys.path:
        sys.path.insert(0, scripts_str)
    if str(scripts_dir.parent) not in sys.path:
        sys.path.insert(0, str(scripts_dir.parent))

def get_outputs_dir(caller_file: str = __file__) -> str:
    return get_env_path("TRINITY_OUTPUT_DIR", "outputs", caller_file)

def get_models_dir(caller_file: str = __file__) -> str:
    return get_env_path("TRINITY_MODEL_DIR", "outputs/models", caller_file)

def get_archive_dir(caller_file: str = __file__) -> str:
    m_dir = get_models_dir(caller_file)
    if "://" in m_dir:
        return f"{m_dir.rstrip('/')}/archive"
    path = Path(m_dir) / "archive"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)

def get_logs_dir(caller_file: str = __file__) -> str:
    return get_env_path("TRINITY_LOG_DIR", "logs", caller_file)

def get_checkpoints_dir(caller_file: str = __file__) -> str:
    o_dir = get_outputs_dir(caller_file)
    if "://" in o_dir:
        return f"{o_dir.rstrip('/')}/checkpoints"
    path = Path(o_dir) / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)

def update_latest_link(target_path: str, link_name: str):
    """建立符號鏈接，指向最新版本，並寫入 Hybrid Logs。"""
    if "://" in target_path:
        logging.info(f"ℹ️  [Link] 雲端路徑暫不支援本地 Symlink: {target_path}")
        return
    
    target = Path(target_path).resolve()
    active_dir = Path(get_models_dir()) / "active_models"
    active_dir.mkdir(parents=True, exist_ok=True)
    
    link_path = active_dir / link_name
    
    try:
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        os.symlink(target, link_path)
        logging.info(f"🔗 [Link] 已更新指標: {link_name} -> {target.name}")
        
        # 混合日誌紀錄 (延遲匯入避免循環引用)
        try:
            from core.db_utils import write_pipeline_log
            write_pipeline_log("Governance", link_name.split('_')[0], "SUCCESS", "PathLink", err=f"Linked to {target.name}")
        except: pass
        
    except Exception as e:
        logging.error(f"❌ [Link] 更新指標失敗: {e}")

def cleanup_directory(directory: str, max_size_gb: float = 10.0, keep_days: int = 30):
    """磁碟治理：清理舊檔案以維持磁碟健康，並寫入 Hybrid Logs。"""
    if "://" in directory: return
    
    path = Path(directory)
    if not path.exists(): return
    
    now = time.time()
    seconds_to_keep = keep_days * 86400
    
    files = []
    for f in path.glob("**/*"):
        if f.is_file():
            files.append((f, f.stat().st_mtime, f.stat().st_size))
            
    deleted_count = 0
    for f, mtime, size in files:
        if (now - mtime) > seconds_to_keep:
            try:
                f.unlink()
                deleted_count += 1
            except: pass
            
    files = sorted([f for f in path.glob("**/*") if f.is_file()], key=os.path.getmtime)
    current_size = sum(os.path.getsize(f) for f in files)
    max_bytes = max_size_gb * 1024**3
    
    while current_size > max_bytes and files:
        f = files.pop(0)
        size = os.path.getsize(f)
        try:
            f.unlink()
            current_size -= size
            deleted_count += 1
        except: pass
        
    if deleted_count > 0:
        msg = f"🧹 [Cleanup] 已從 {directory} 清理 {deleted_count} 個過期檔案。"
        logging.info(msg)
        
        # 混合日誌紀錄 (延遲匯入避免循環引用)
        try:
            from core.db_utils import write_pipeline_log
            write_pipeline_log("Cleanup", "SYSTEM", "SUCCESS", "DiskHealth", rows=deleted_count, err=f"Dir: {directory}")
        except: pass

def ensure_dirs_exist(caller_file: str = __file__) -> Dict[str, str]:
    ensure_scripts_on_path(caller_file)
    return {
        "scripts": str(get_scripts_dir(caller_file)),
        "outputs": get_outputs_dir(caller_file),
        "models": get_models_dir(caller_file),
        "archive": get_archive_dir(caller_file),
        "logs": get_logs_dir(caller_file),
        "checkpoints": get_checkpoints_dir(caller_file)
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    dirs = ensure_dirs_exist(__file__)
    print("✅ Quantum Finance 專案目錄結構 (Phase 3 治理版) 初始化完成：")
    for k, v in dirs.items():
        print(f" - {k.capitalize()}: {v}")
    
    # 治理功能測試
    test_file = Path(dirs['logs']) / "test_cleanup.tmp"
    test_file.write_text("dummy")
    cleanup_directory(dirs['logs'], max_size_gb=0.000001, keep_days=0)