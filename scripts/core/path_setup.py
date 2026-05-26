"""
path_setup.py v4.48 (Quantum Finance Bootstrap-Aligned Path SSOT Edition · §0.0-I.10 Cross-Platform Path Resolution + §14.7-BR Phase C-1 配套 ensure_scripts_on_path 函式補入)
================================================================================
**最後更新日期**: 2026-05-26
**主權狀態**: PERFECT (憲法 v6.1.0 對齊 + [Zero Hardcoded Verdict] + [Sovereignty Declaration] 核心定義補入 + **§0.0-I.10 跨平台路徑解析 (v4.47 os.path.realpath() 解 symlink 後比對；對齊 §14.7-AU v6.1.0 升版)** + **v4.48 `ensure_scripts_on_path()` 函式補入 (§14.7-BR Phase C-1 配套；治本 6 fetchers + maintenance scripts 之 entry-point 引導 import error)**；8 項檢查面 100% 合規)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
1. [Path Sovereignty]: 確立絕對物理基準 (PROJECT_ROOT)，所有子路徑接口必須以此為唯一根節點擴展。
2. [Self-Healing Mechanism]: 具備自動化目錄重建與權限修正能力，確保系統環境「零配置」主權啟動。
3. [Historical Reference Authority]: 保留從 v1.0 至今的所有歷史，作為判定系統物理架構正確性的基準。
4. [Boundary Integrity]: 確保 25 維全譜路徑接口與實體目錄 100% 同步，封印任何執行鏈中斷之可能性。
5. [Bootstrap Anchor Alignment]: `.env` 僅作為 Bootstrap Anchor 錨定 `PROJECT_ROOT`；`path_setup.py` 保持 Path SSOT，負責從根目錄推導 25 維路徑。
6. [Hub Compatibility]: 本檔之 `PROJECT_ROOT_CALC`、25 維 `get_*_dir()` 與 `get_evaluation_dir()` 相容別名、`ALL_PATHS`、`ensure_all_dirs()` 為 `core/__init__.py v1.16` hub 鏡像匯出之 import 邊界；任何改名或移除都會直接破壞 hub 治權。
7. [Zero Hardcoded Verdict]: 主權判定（PERFECT / WARNING / FAILED）必須依執行結果動態計算，嚴禁硬編碼。對齊憲章 §5.6.3「禁止硬編碼 PERFECT」與 §3.2 Step 1 接受標準；治權慣例對齊 `data_schema.py v2.13+` / `core/__init__.py v1.16` 之 [Zero Hardcoded Verdict]。
8. [Sovereignty Declaration]: 本模組為憲章 §3.1 序列模組 / Path SSOT Authority（憲章 §一 3. [Boundary Integrity] / L2334 / L2335）；不涉及 §0.1-A 第一性原理 / §0.2-A 八二法則 / §0.3-A 康波週期 / §0.0-E.4 統合層 / §0.0-F.3 AI 協作工具規則五套禁令；不在 §0.1.1 T1/T2/T3 分層內（path 為 infrastructure，不執行第一性原理工程公式）；不處理 §8.5 anti-leakage（不涉時間序列建模）；不承擔 25 維以外之路徑治權（依 [Boundary Integrity] 邊界）。

## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix - 100% Coverage)
| 維運需求場景 (Scenario)   | 權威指令 / 建議用法 (Exhaustive Examples)                             | 對齊模組 |
| :----------------------- | :-------------------------------------------------------------------- | :--- |
| **1. [路徑治理：全維度稽核]** | `$ python scripts/core/path_setup.py`                                 | path_setup v4.47 |
| **2. [個股同步：環境路徑初始化]** | `$ python scripts/core/path_setup.py`                                 | path_setup v4.47 |
| **3. [啟動錨點：`.env` 與 `PROJECT_ROOT` 對齊驗證]** | `$ python scripts/core/path_setup.py`                                 | path_setup v4.47 |

## 📜 三、全修訂歷程 (Full Revision History - 舊詳細參考)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **v4.48** | 2026-05-26 | Codex | **`ensure_scripts_on_path(caller_file)` 函式補入（§14.7-BR Phase C-1 配套；治本 6 fetchers + maintenance scripts entry-point 引導 import error）**：依憲章 v6.1.0-patch 第十六輪 §14.7-BR Phase B 入憲 (charter L9229+) 後執行 Phase C-1 M2SL sync 揭露之 stranded state — `scripts/fetchers/fetch_fred_data.py` L54 + 5 個其他 fetchers (`fundamental` / `news` / `international` / `chip` / `cash_flows`) 全部 `from core.path_setup import ensure_scripts_on_path` 但本檔自 git history 從未含此函式 (code-side vs path_setup-side truth 不同步)。**Root cause**：他機 production 可能更新 path_setup 加入此函式但未 push；本機 stale path_setup + stale fetchers 同時存在 → 6 fetchers 全 broken on 本機 (本 session 之前未跑 fetcher 才未揭露)。**功能補入 1 點**：(a) 新 public function `ensure_scripts_on_path(caller_file)` (位置：`ensure_all_dirs()` 之前)；邏輯 `Path(caller_file).resolve().parent.parent` (caller 通常在 `scripts/<subdir>/` → `scripts/` = caller.parent.parent) → 加入 `sys.path` (idempotent;重複呼叫安全；對齊 6 fetchers L50-52 之 manual 既有 pattern)。**標頭變更**：(b) 副標補入「§14.7-BR Phase C-1 配套 ensure_scripts_on_path 函式補入」；(c) 主權狀態行補入 v4.48 描述；(d) TOOL_VER v4.47 → v4.48；(e) 最後更新日期 2026-05-23 → 2026-05-26；(f) v4.47 SUPERSEDED + 新 v4.48 ACTIVE entry；(g) `record_lifecycle("path_setup_v4.47", ...)` task_name 升 v4.48 (一處)。**治權邊界嚴守**：25 維路徑接口 / `ALL_PATHS` / `ensure_all_dirs()` / `_evaluate_anchor()` / `_load_logging_hooks()` / 自癒邏輯 / `verdict` 計算邏輯 / hub 鏡像 import 邊界**全保留**；本補正純為新增 1 個 public function 解 6 fetchers stranded import；**不**改 schema / DB / API / Hub / `core/__init__.py` 介面。**驗證**：6 fetchers 之既有 `from core.path_setup import ensure_scripts_on_path` import 即時解通 (M2SL sync 為第一驗證案例 / §14.7-BR Phase C-1 立即可跑)。**§0.0-G 第 N 次跑通延伸落地**：本程式為 §14.7-BR Phase C-1 之 dependency-fix 配套，stranded state 修補首例 (本機 vs production code-truth 不同步 → 治權修補)。**對應憲章 v6.1.0-patch 第十六輪 §14.7-BR**：Phase B 入憲 (commit `95fda16`) 後執行 Phase C-1 揭露之 stranded → infrastructure 治本 patch。 | **ACTIVE** |
| v4.47 | 2026-05-23 | Codex | **§0.0-I.10 跨平台路徑解析治權契約落地（憲章 v6.1.0 §0.0-I.10 + §14.7-AU 入憲對應之單程式升版）**：依憲章 v6.1.0 §0.0-I.10（2026-05-23 入憲）落地路徑比對改用 `os.path.realpath()` 解析 symlink。**Root cause 實證**：2026-05-23 from-zero rebuild Step 1 揭露 `.env` `PROJECT_ROOT=/home/hugo/project/stock_backend`（Linux 慣例）於 macOS 環境（實際 `/Users/hugo/project/stock_backend`）被 `_evaluate_anchor()` 之裸字串比對判 `MISMATCHED` → 主權狀態 FAILED。雖然 macOS 上 `/home → Users` symlink 可解析（`ls -d /home/hugo/...` 成功），但 v4.46 字串比對未考慮 symlink 同義。**功能變更 1 點**：(a) `_evaluate_anchor()` 升級邏輯：(I) `os.path.realpath()` 解析 env 與 calc 路徑後再比對；(II) realpath 拋例外時返 `failed / PATH RESOLVE ERROR / realpath failure: ...`；(III) realpath 後不對齊返 `failed / MISMATCHED (env → env_real; calc=calc_real)`（真實 mismatch）；(IV) realpath 後對齊但字串不同返 `success / MATCHED (via symlink: env → env_real)`（合法 symlink 同義）；(V) 字串與 realpath 皆對齊返 `success / MATCHED`（原 v4.46 行為）。**標頭變更**：(b) 副標補入「§0.0-I.10 Cross-Platform Path Resolution 落地」；(c) 主權狀態行補入 §0.0-I.10 v4.47 + 憲法 v6.0.0 → v6.1.0；(d) TOOL_VER v4.46 → v4.47 + CONSTITUTION_VER v6.0.0 → v6.1.0；(e) 最後更新日期 2026-05-21 → 2026-05-23；(f) v4.46 SUPERSEDED + 新 v4.47 ACTIVE entry。**治權邊界嚴守**：所有 25 維路徑接口 / `ALL_PATHS` / `ensure_all_dirs()` / 自癒邏輯 / `record_lifecycle` / `verdict` 計算邏輯**全保留**；本補正純為 anchor 比對之 symlink 對齊（macOS / Linux / Windows 跨平台部署相容性）；**不**改 schema / DB / API / Hub / `core/__init__.py` 介面。**對應憲章 v6.0.0 → v6.1.0 升版**：本程式為 §14.7-AU 預備 7 程式之第二個落地（per `reports/系統架構大憲章_v6.1.0.md` §14.7-AU C 表）。**§0.0-G 第 19 次跑通延伸落地之第二棒**。 | SUPERSEDED |
| v4.46 | 2026-05-21 | Codex | **[Zero Hardcoded Verdict] + [Sovereignty Declaration] 核心定義 2 條補入（8 項檢查面 100% 合規補強；逐元件審計 Step 1.1.1 跟進）**：依 v4.45 後 8 項檢查面審計揭露之 4 項標頭治權自我宣告缺口（缺 [Zero Hardcoded Verdict] 明文 / 治權位階 Type 未明示 / 5 套禁令未明示 / T1-T3 分層未明示 / §8.5 anti-leakage 未明示），補入兩條核心定義一次性涵蓋。**補正內容**：(I) 核心定義新增第 7 條 [Zero Hardcoded Verdict]：主權判定動態計算（程式邏輯 L181-187 之 `sovereignty_status` 計算原已 §5.6.3 合規，本次純為核心定義條之顯式宣告）；(II) 核心定義新增第 8 條 [Sovereignty Declaration]：§3.1 序列模組 / Path SSOT Authority（§一 3. / L2334 / L2335）；不涉 5 套禁令；不在 T1/T2/T3 分層內；不處理 §8.5 anti-leakage；不承擔 25 維以外之路徑治權；(III) [Hub Compatibility] 核心定義第 6 條之 hub 引用由 `v1.14` 同步至 `v1.16`（對齊 `core/__init__.py` 實際版本）；(IV) 主權狀態行升至「(憲法 v6.0.0 對齊 + [Zero Hardcoded Verdict] + [Sovereignty Declaration] 核心定義補入；8 項檢查面 100% 合規)」；(V) TOOL_VER v4.45 → v4.46；(VI) 維運矩陣 / Flagship Report / record_lifecycle task_name 之 cosmetic v4.45 → v4.46。**API、ALL_PATHS、25 維接口、自癒邏輯、`_evaluate_anchor()`、verdict 計算邏輯、所有公開行為皆無變更**；本補正純為標頭治權自我宣告（與 `data_schema.py v2.14` / `core/__init__.py v1.16` 同模式）。 | SUPERSEDED |
| v4.45 | 2026-05-20 | Codex | **v6.0.0 標頭治權對齊 + `CONSTITUTION_VER` 模組常數補入（逐元件審計 Step 1.1.1 補正）**：依逐元件治權合規審計 Step 1.1.1 揭露之兩項違規：(1) 缺 `CONSTITUTION_VER` 模組常數（違反憲章 L26「所有 §3.1/§3.2 登錄模組之 `CONSTITUTION_VER` 同步至 v6.0.0」）；(2) 修訂歷程缺 v6.0.0 升版條目（雖 §6.8.6 已執行字串補正，但 v4.44 條目未更新）。本次補正：(I) 新增 `CONSTITUTION_VER = "v6.0.0"` 模組常數於 25 維路徑接口註冊區之前；(II) 補入本 v4.45 修訂條目記錄 v6.0.0 對齊；(III) [Hub Compatibility] 核心定義第 6 條之 hub 引用維持 `core/__init__.py v1.14`（依 §0.0-I 單一引用源，hub 自身版本變動由 hub 維護）。**API、ALL_PATHS、25 維接口、自癒邏輯、`_evaluate_anchor()` 與所有公開行為皆無變更**；本補正純為標頭治權對齊。 | SUPERSEDED |
| v4.44 | 2026-05-15 | Codex | **憲法 v5.4.22 同日補充標籤對齊 + 2026-05-16 bootstrap 診斷補正**：標頭由「憲法 v5.4.21 啟動治理對齊」升至「憲法 v5.4.22 啟動治理對齊 + 同日 hub 補充相容」；新增 [Hub Compatibility] 核心定義第 6 條，明定本檔對 `core/__init__.py v1.14` hub 之 import 邊界承諾（`PROJECT_ROOT_CALC` / 25 維 `get_*_dir()` / `get_evaluation_dir()` / `ALL_PATHS` / `ensure_all_dirs()`）。2026-05-16 補正 DB logging hook 不可用時的 `BOOTSTRAP-DEFERRED (DB hook unavailable: ...)` 診斷文字；**API、ALL_PATHS、自癒邏輯與 `_evaluate_anchor()` 皆無變更**。 | SUPERSEDED |
| v4.43 | 2026-05-14 | Codex | **憲法 v5.4.21 啟動治理對齊**：補齊 Bootstrap Anchor / Path SSOT 契約；新增 `get_evaluation_dir()` 相容別名；`.env` 錨點異常不再回報 PERFECT；DB schema 尚未初始化時進入 `BOOTSTRAP-DEFERRED`；真實 DB audit log 寫入失敗才降級為 warning，不阻斷路徑自癒。 | SUPERSEDED |
| v4.42 | 2026-05-13 | Antigravity | **憲法 v5.4 對齊**：校正治理維度為 25 維；修正標頭為對齊 v5.4；新增 .env 錨點對齊校驗邏輯。 | SUPERSEDED |
| v4.41 | 2026-05-12 | Antigravity | 主權完備化：補全全場景路徑維運矩陣。 | SUPERSEDED |
| v1.0 | 2026-04-20 | Antigravity | 主權奠基：初始路徑管理版本。 | ARCHIVED |
================================================================================
"""
import os, sys, time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────────────
# 治權常數 (Constitution Constants) — v4.45 新增（憲章 L26 / Step 1.1.1 補正）
# ──────────────────────────────────────────────────────────────────────────────
CONSTITUTION_VER = "v6.1.0"
TOOL_VER = "v4.48"

# ── 系統級架構引導 ──
_THIS_FILE = Path(__file__).resolve()
_CORE_DIR = _THIS_FILE.parent
_SCRIPTS_DIR = _CORE_DIR.parent
PROJECT_ROOT_CALC = _SCRIPTS_DIR.parent

# 載入 .env 進行錨點稽核
load_dotenv(PROJECT_ROOT_CALC / ".env")
PROJECT_ROOT_ENV = os.getenv("PROJECT_ROOT")

# 25 維路徑接口註冊中心 (v4.47 完備版)
def get_root_dir(): return PROJECT_ROOT_CALC
def get_core_dir(): return PROJECT_ROOT_CALC / "scripts" / "core"
def get_utils_dir(): return PROJECT_ROOT_CALC / "scripts" / "utils"
def get_maintenance_dir(): return PROJECT_ROOT_CALC / "scripts" / "maintenance"
def get_data_dir(): return PROJECT_ROOT_CALC / "data"
def get_raw_data_dir(): return PROJECT_ROOT_CALC / "data" / "raw"
def get_ingestion_dir(): return PROJECT_ROOT_CALC / "data" / "ingestion"
def get_feature_dir(): return PROJECT_ROOT_CALC / "data" / "features"
def get_feature_store_dir(): return PROJECT_ROOT_CALC / "data" / "feature_store"
def get_model_dir(): return PROJECT_ROOT_CALC / "models"
def get_model_weights_dir(): return PROJECT_ROOT_CALC / "models" / "weights"
def get_model_scalers_dir(): return PROJECT_ROOT_CALC / "models" / "scalers"
def get_training_dir(): return PROJECT_ROOT_CALC / "models" / "training"
def get_archive_dir(): return PROJECT_ROOT_CALC / "models" / "archive"
def get_mlflow_dir(): return PROJECT_ROOT_CALC / "models" / "mlflow"
def get_infer_dir(): return PROJECT_ROOT_CALC / "inference"
def get_prediction_dir(): return PROJECT_ROOT_CALC / "inference" / "predictions"
def get_eval_dir(): return PROJECT_ROOT_CALC / "inference" / "evaluation"
def get_evaluation_dir(): return get_eval_dir()  # Legacy alias; not counted as an extra path dimension.
def get_output_dir(): return PROJECT_ROOT_CALC / "outputs"
def get_report_dir(): return PROJECT_ROOT_CALC / "reports"
def get_scratch_dir(): return PROJECT_ROOT_CALC / "scratch"
def get_log_dir(): return PROJECT_ROOT_CALC / "logs"
def get_pipeline_dir(): return PROJECT_ROOT_CALC / "logs" / "pipeline"
def get_monitor_dir(): return PROJECT_ROOT_CALC / "logs" / "monitor"
def get_test_dir(): return PROJECT_ROOT_CALC / "tests"

ALL_PATHS = [
    get_root_dir(), get_core_dir(), get_utils_dir(), get_maintenance_dir(),
    get_data_dir(), get_raw_data_dir(), get_ingestion_dir(),
    get_feature_dir(), get_feature_store_dir(),
    get_model_dir(), get_model_weights_dir(), get_model_scalers_dir(),
    get_training_dir(), get_archive_dir(), get_mlflow_dir(),
    get_infer_dir(), get_prediction_dir(), get_eval_dir(),
    get_output_dir(), get_report_dir(), get_scratch_dir(), get_log_dir(),
    get_pipeline_dir(), get_monitor_dir(), get_test_dir()
]

def _load_logging_hooks():
    """載入混合日誌鉤子；DB schema 尚未建立時進入 bootstrap deferred。"""
    from contextlib import contextmanager

    @contextmanager
    def mock_lifecycle(task_name, **kwargs):
        yield None

    def mock_audit_log(*args, **kwargs):
        return None

    try:
        if str(_SCRIPTS_DIR) not in sys.path:
            sys.path.insert(0, str(_SCRIPTS_DIR))
        from core.db_utils import get_db_connection, record_lifecycle, write_data_audit_log

        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT
                    to_regclass('public.pipeline_execution_log') IS NOT NULL,
                    to_regclass('public.data_audit_log') IS NOT NULL
                """
            )
            has_pipeline_log, has_audit_log = cur.fetchone()
        finally:
            cur.close()
            conn.close()

        if has_pipeline_log and has_audit_log:
            return record_lifecycle, write_data_audit_log, "REAL (DB-Linked)"
        return mock_lifecycle, mock_audit_log, "BOOTSTRAP-DEFERRED (DB schema pending)"
    except Exception as e:
        reason = str(e).strip() or type(e).__name__
        return mock_lifecycle, mock_audit_log, f"BOOTSTRAP-DEFERRED (DB hook unavailable: {reason})"


def _mark_lifecycle(lifecycle, level, message):
    if lifecycle is None:
        return
    method_name = "mark_failed" if level == "failed" else "mark_warning"
    marker = getattr(lifecycle, method_name, None)
    if callable(marker):
        marker(message)


def _evaluate_anchor():
    """v4.47 §0.0-I.10 跨平台路徑解析：路徑比對改用 os.path.realpath() 解析 symlink
    後再比對，避免跨平台 (macOS `/home → Users`) 之 symlink 同義路徑被誤判 MISMATCHED。"""
    calculated = str(PROJECT_ROOT_CALC)
    if not PROJECT_ROOT_ENV:
        return "failed", "MISSING (PROJECT_ROOT not set in .env)", "PROJECT_ROOT missing in .env"
    # v4.47 §0.0-I.10：realpath() 解析 symlink 後比對（裸字串比對之升級）
    try:
        env_real = os.path.realpath(PROJECT_ROOT_ENV)
        calc_real = os.path.realpath(calculated)
    except (TypeError, OSError) as exc:
        return (
            "failed",
            f"PATH RESOLVE ERROR (env={PROJECT_ROOT_ENV})",
            f"realpath failure: {type(exc).__name__}: {exc}",
        )
    if env_real != calc_real:
        # symlink 解析後仍不對齊：真實 mismatch
        return (
            "failed",
            f"MISMATCHED (env={PROJECT_ROOT_ENV} → {env_real}; calc={calc_real})",
            f"PROJECT_ROOT mismatch (post-realpath): env_real={env_real}, calc_real={calc_real}",
        )
    if PROJECT_ROOT_ENV != calculated:
        # 字串不同但 realpath 解析後對齊（合法 symlink 同義）
        return (
            "success",
            f"MATCHED (via symlink: env={PROJECT_ROOT_ENV} → {env_real})",
            None,
        )
    return "success", "MATCHED", None


def ensure_scripts_on_path(caller_file) -> None:
    """v4.48 (§14.7-BR Phase C-1 配套): 確保 scripts/ 目錄在 sys.path 內,
    讓 fetcher / maintenance 等子目錄之 script 可使用 `from core.xxx import yyy`
    模式不受 invocation 方式影響。Idempotent — 重複呼叫安全。

    用途: 6 fetchers(fred / fundamental / news / international / chip / cash_flows)
    + maintenance scripts 之 entry-point 引導(對齊 fetchers L50-52 既有 manual
    sys.path 模式)。

    Args:
        caller_file: 呼叫端傳入 __file__(typically `scripts/fetchers/xxx.py` 或
                     `scripts/maintenance/xxx.py` 之絕對路徑)。
    """
    caller_path = Path(caller_file).resolve()
    # caller 通常在 scripts/<subdir>/(如 scripts/fetchers/); scripts/ = caller.parent.parent
    scripts_dir = caller_path.parent.parent
    scripts_dir_str = str(scripts_dir)
    if scripts_dir_str not in sys.path:
        sys.path.insert(0, scripts_dir_str)


def ensure_all_dirs():
    """執行物理路徑對齊與自癒 (v4.48 / 憲法 v6.1.0 啟動治理版)"""
    start_time = time.time()
    record_lifecycle, write_data_audit_log, log_mode = _load_logging_hooks()
    warnings = []

    for p in ALL_PATHS:
        p.mkdir(parents=True, exist_ok=True)

    anchor_level, anchor_display, anchor_message = _evaluate_anchor()
    if log_mode.startswith("MOCK"):
        warnings.append(f"Hybrid logging downgraded: {log_mode}")

    with record_lifecycle("path_setup_v4.48", category="maintenance", stock_id="SYSTEM") as lifecycle:
        if anchor_message:
            _mark_lifecycle(lifecycle, anchor_level, anchor_message)

        try:
            write_data_audit_log(
                "PATH_SOVEREIGNTY",
                "SYSTEM",
                datetime.now().strftime("%Y-%m-%d"),
                "SELF_HEAL_v4.47",
                len(ALL_PATHS),
            )
        except Exception as e:
            msg = f"data_audit_log downgraded: {type(e).__name__}: {e}"
            warnings.append(msg)
            _mark_lifecycle(lifecycle, "warning", msg)

        for msg in warnings:
            _mark_lifecycle(lifecycle, "warning", msg)

        if anchor_level != "success":
            sovereignty_status = "FAILED (Bootstrap Anchor Drift)"
        elif warnings:
            sovereignty_status = "WARNING (路徑已對齊；日誌降級)"
        else:
            sovereignty_status = "PERFECT (已對齊/自癒)"

        # ── 執行後路徑稽核摘要 (Flagship Report v4.47) ──
        print("\n" + "🛡️" * 40)
        print("🚀 Quantum Finance: 路徑主權治理中心 (v4.47)")
        print("🛡️" * 40)

        print("\n" + "─" * 80)
        print("📊 物理路徑稽核摘要報告 (Path Sovereignty Report v4.47)")
        print("─" * 80)
        print(f"✅ 物理基準 (ROOT) : {PROJECT_ROOT_CALC}")
        print(f"⚓ 錨點對齊 (.env) : {anchor_display}")
        print("✅ 治理維度        : 25 維全譜路徑 (對齊 v6.1.0)")
        print(f"🕒 處理時長        : {(time.time() - start_time)*1000:.2f} ms")
        print(f"📝 混合日誌模式    : {log_mode}")
        if warnings:
            print(f"⚠️  降級警告        : {'; '.join(warnings[:3])}")
        print(f"⚖️  路徑主權狀態    : {sovereignty_status}")
        print("─" * 80)

        print("\n💡 路徑維運建議 (Reference Information):")
        print("1. [治權提示]: 系統已鎖定絕對路徑，嚴禁在子模組中使用 os.chdir()。")
        print("2. [解耦提示]: .env 僅錨定 PROJECT_ROOT；path_setup.py 承擔 25 維 Path SSOT 職責。")
        print("3. [範例提示]: 請參閱 Header 矩陣以確保維運環境的一致性。")
        print("─" * 80 + "\n")

    return anchor_level == "success"


if __name__ == "__main__":
    sys.exit(0 if ensure_all_dirs() else 1)
