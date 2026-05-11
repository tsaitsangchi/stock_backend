"""
path_setup.py v3.1 (Quantum Finance Edition)
================================================================================
系統路徑配置器 — 混合日誌與數據治理版 (Quantum v5.1 標準)
整合 Active Symlinks 指標、LRU 自動清理機制，並將治理事件同步至系統儀表板。

修訂歷程：
  v3.1 (2026-05-11): [標準化] 導入 Quantum 標準檔頭、生命週期監測與磁碟空間回報儀表板。
  v3.0 (2026-05-11): [治理] 整合清理事件至 data_audit_log 專屬紀錄。
  v2.7 (2026-05-10): [核心] 導入基礎混合模式日誌。

執行範例 (Comprehensive Usage Examples):
  1. [系統初始化] 初始化專案目錄結構並清理過期日誌:
     python scripts/core/path_setup.py

  2. [核心路徑獲取] 在其他程式中獲取模型存放路徑:
     from core.path_setup import get_models_dir
     path = get_models_dir()

  3. [最新連結更新] 將訓練好的模型連結至 'latest' 指標:
     from core.path_setup import update_latest_link
     update_latest_link("archive/model_2330.pkl", "2330_latest.pkl")

  4. [磁碟治理] 定期清理過期 30 天且超過 10GB 的目錄:
     from core.path_setup import cleanup_directory
     cleanup_directory("scripts/logs", max_size_gb=10.0, keep_days=30)
================================================================================
"""
import sys, os, time, shutil, logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

# ── 終極路徑自癒 Bootstrap ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_SCRIPTS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR.parent))

try:
    from dotenv import load_dotenv
    _ENV_PATH = _SCRIPTS_DIR.parent / ".env"
    if _ENV_PATH.exists(): load_dotenv(_ENV_PATH)
    else: load_dotenv()
except: pass

_scripts_dir_cache: Optional[Path] = None

def get_scripts_dir(caller_file: str = __file__) -> Path:
    global _scripts_dir_cache
    if _scripts_dir_cache is None:
        current_path = Path(caller_file).resolve()
        for parent in [current_path] + list(current_path.parents):
            if parent.name == "scripts": _scripts_dir_cache = parent; break
            if (parent / "core").is_dir() and (parent / "fetchers").is_dir(): _scripts_dir_cache = parent; break
            if (parent / "scripts").is_dir(): _scripts_dir_cache = parent / "scripts"; break
        if not _scripts_dir_cache: _scripts_dir_cache = current_path.parent
    return _scripts_dir_cache

def ensure_scripts_on_path(caller_file: str):
    s_dir = get_scripts_dir(caller_file)
    for p in [str(s_dir), str(s_dir.parent)]:
        if p not in sys.path: sys.path.insert(0, p)

def get_env_path(key: str, default_subpath: str) -> str:
    val = os.getenv(key)
    if val: return val
    path = get_scripts_dir() / default_subpath
    if "://" not in str(path): path.mkdir(parents=True, exist_ok=True)
    return str(path)

def get_outputs_dir() -> str: return get_env_path("TRINITY_OUTPUT_DIR", "outputs")
def get_models_dir() -> str: return get_env_path("TRINITY_MODEL_DIR", "outputs/models")
def get_archive_dir() -> str:
    p = Path(get_models_dir()) / "archive"
    p.mkdir(parents=True, exist_ok=True); return str(p)
def get_logs_dir() -> str: return get_env_path("TRINITY_LOG_DIR", "logs")

def show_path_dashboard(dirs: Dict[str, str]):
    """基礎設施儀表板回報。"""
    print("\n" + "="*55)
    print("🚀 Quantum Finance: 基礎設施路徑儀表板 (v3.1)")
    print("="*55)
    for k, v in dirs.items():
        print(f" - {k.capitalize():<12}: {v}")
    
    print("-" * 55)
    # 額外參考訊息：磁碟空間
    try:
        total, used, free = shutil.disk_usage(dirs['scripts'])
        print(f"💾 磁碟狀態  : 剩餘 {free // (2**30)} GB / 總量 {total // (2**30)} GB")
        log_size = sum(f.stat().st_size for f in Path(dirs['logs']).glob('**/*') if f.is_file())
        print(f"📋 日誌佔用  : {log_size / (2**20):.2f} MB")
    except: pass
    print("="*55 + "\n")

def update_latest_link(target_path: str, link_name: str):
    """建立符號鏈接，並寫入生命週期日誌。"""
    if "://" in target_path: return
    target = Path(target_path).resolve()
    active_dir = Path(get_models_dir()) / "active_models"
    active_dir.mkdir(parents=True, exist_ok=True)
    link_path = active_dir / link_name
    try:
        if link_path.exists() or link_path.is_symlink(): link_path.unlink()
        os.symlink(target, link_path)
        logging.info(f"🔗 [Link] {link_name} -> {target.name}")
        # 延遲匯入避免循環
        from core.db_utils import write_pipeline_log
        write_pipeline_log("PathLink", link_name.split('_')[0], "success", "governance")
    except: pass

def cleanup_directory(directory: str, max_size_gb: float = 10.0, keep_days: int = 30):
    """治理功能：清理舊檔案並寫入審計日誌。"""
    if "://" in directory or not Path(directory).exists(): return
    path = Path(directory)
    now = time.time(); deleted_count = 0
    files = sorted([f for f in path.glob("**/*") if f.is_file()], key=os.path.getmtime)
    for f in files:
        if (now - f.stat().st_mtime) > (keep_days * 86400):
            try: f.unlink(); deleted_count += 1
            except: pass
    
    current_size = sum(f.stat().st_size for f in path.glob("**/*") if f.is_file())
    if current_size > (max_size_gb * 1024**3):
        files = sorted([f for f in path.glob("**/*") if f.is_file()], key=os.path.getmtime)
        while current_size > (max_size_gb * 1024**3) and files:
            f = files.pop(0); sz = f.stat().st_size
            try: f.unlink(); current_size -= sz; deleted_count += 1
            except: pass
    
    if deleted_count > 0:
        logging.info(f"🧹 [Cleanup] 從 {directory} 清理 {deleted_count} 個檔案。")
        from core.db_utils import write_pipeline_log, write_data_audit_log
        write_pipeline_log("Cleanup", "SYSTEM", "success", "governance", rows=deleted_count)
        write_data_audit_log("filesystem", "LOGS", datetime.now().strftime("%Y-%m-%d"), "CLEANUP", deleted_count)

def ensure_dirs_exist(caller_file: str = __file__) -> Dict[str, str]:
    ensure_scripts_on_path(caller_file)
    dirs = {
        "scripts": str(get_scripts_dir(caller_file)),
        "outputs": get_outputs_dir(),
        "models": get_models_dir(),
        "archive": get_archive_dir(),
        "logs": get_logs_dir()
    }
    # 確保本地路徑存在
    for d in dirs.values():
        if "://" not in d: Path(d).mkdir(parents=True, exist_ok=True)
    return dirs

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    # 整合生命週期紀錄
    try:
        from core.db_utils import record_lifecycle
        with record_lifecycle("path_setup", category="sys", stock_id="SYSTEM"):
            dirs = ensure_dirs_exist(__file__)
            cleanup_directory(dirs['logs'], max_size_gb=1.0, keep_days=30)
    except:
        dirs = ensure_dirs_exist(__file__)
    
    show_path_dashboard(dirs)