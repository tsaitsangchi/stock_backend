"""
utils/db.py v5.5.26 (DB 連線工具)
================================================================================
**最後更新日期**: 2026-06-02
**主權狀態**: ACTIVE (utility helper)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:資料庫連線檢查小工具(check_connection)。

**輸入 → 輸出**:DB 參數 → 連線狀態

**為什麼需要它**:子系統共用之輕量 DB helper。

## 📜 一、核心定義說明 (Core Definitions)

1. **[DB Helper]**:提供連線檢查
2. **[Sovereignty Declaration]**:本程式為**非 charter-core 子系統**工具(charter 可達/引用),不涉 §3.1/§3.2 序列治權主軸、不持五套禁令、不處理 §8.5 anti-leakage。
3. **[Historical Reference Authority]**:本檔標頭版本為記述性快照,非權威來源(權威為憲章 + 程式現行碼)。

## 📊 二、全量功能群矩陣 (Functional Group Matrix)

| 功能 / 指令 | 說明 |
| :--- | :--- |
| check_connection() | 檢查 DB 連線 |

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v5.5.26 | 2026-06-02 | Codex | §一.11 標頭三段式 + 白話補正;原邏輯不變。 | **ACTIVE** |

## 原始說明
db.py v5.5.26 (Trinity Core Final)
================================================================================
資料庫輔助工具 — 混合日誌整合版
作為 core.db_utils 的輕量化封裝介面，提供快速的連線與交易處理。

修訂歷程：
  v5.5.26 (2026-05-10):
    - [對接] 引用 core.db_utils (v4.14) 進程安全連線池。
    - [文檔] 補齊極致詳細的執行範例與 SQL 稽核指引。

【執行範例說明】

1. 快速執行 SQL 查詢：
   ------------------------------------------------------------
   from utils.db import db_transaction
   with db_transaction() as cur:
       cur.execute("SELECT COUNT(*) FROM stocks")
       print(cur.fetchone())
   ------------------------------------------------------------

2. SQL 維運查閱：
   SELECT * FROM pipeline_execution_log WHERE task_name = 'db_util_check' ORDER BY created_at DESC;
"""

import sys
from pathlib import Path

# ── 系統路徑修復 (v3.1) ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import db_session, db_transaction, write_pipeline_log
except ImportError as e:
    print(f"[FATAL] 無法匯入核心組件: {e}", file=sys.stderr)
    sys.exit(1)

def check_connection():
    """驗證連線狀況並紀錄"""
    try:
        with db_transaction() as cur:
            cur.execute("SELECT 1")
            write_pipeline_log("db_util_check", "SYSTEM", "success", "sys")
            return True
    except Exception as e:
        write_pipeline_log("db_util_check", "SYSTEM", "failed", "sys", 0, 0, str(e))
        return False

if __name__ == "__main__":
    if check_connection():
        print("✅ 資料庫連線測試成功。")
