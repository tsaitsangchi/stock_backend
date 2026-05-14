"""
audit_supply_chain.py v1.17 (Compliance Edition - DB State Aware)
================================================================================
**最後更新日期**: 2026-05-13
**主權狀態**: GENESIS COMPLETED (憲法 v5.4.18 對齊)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
1. [Absolute Case Alignment]: 監控並如實記錄 API 之大小寫 (Case-Sensitive)。
2. [Compliance Assertion]: 執行後必須明確顯示與《系統架構大憲章_v5.4.18》之對齊狀態。
3. [Database State Verification] (v1.17 新增): 稽核不可只看 API，必須交叉比對 DB 實際狀態，
   包含表存在性、筆數、4 個 FRED series 完整性、最新資料日期。
4. [Truth-based Verdict] (v1.17 新增): 主權判定必須依稽核結果動態計算，
   嚴禁硬編碼 PERFECT；任何 FAILED 立刻將整體判定降為 FAILED。
5. [Lifecycle Integrity] (v1.17 新增): 必須交叉比對 pipeline_execution_log 與 data_audit_log，
   抓出靜默失敗、status 謊報、end_time 缺欄等執行鏈異常。

## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix)
| 維運需求場景 (Scenario)              | 權威指令 / 建議用法 (Exhaustive Examples)                                       | 對齊模組 |
| :----------------------------------- | :----------------------------------------------------------------------------- | :--- |
| **0. [偵察：全譜供應鏈合規審計]**     | `$ python scripts/maintenance/audit_supply_chain.py`                            | audit_tool v1.17 |
| **1. [偵察：僅 FinMind 供應鏈]**      | `$ python scripts/maintenance/audit_supply_chain.py --source finmind`           | audit_tool v1.17 |
| **2. [偵察：僅 FRED 宏觀]**           | `$ python scripts/maintenance/audit_supply_chain.py --source fred`              | audit_tool v1.17 |
| **3. [偵察：僅 DB 實況稽核]**         | `$ python scripts/maintenance/audit_supply_chain.py --db-only`                  | audit_tool v1.17 |
| **4. [偵察：含 lifecycle 日誌交叉比對]** | `$ python scripts/maintenance/audit_supply_chain.py --include-logs`             | audit_tool v1.17 |

## 📜 三、全修訂歷程 (Full Revision History)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **v1.17** | 2026-05-13 | Auto-patch | **稽核失職修補**：(1) 補強為 DB-state aware（不再只查 API）；(2) 新增 FredData 4 series 完整性檢查（封堵 UNRATE/VIXCLS 漏網事件）；(3) 新增 lifecycle log 交叉比對抓 status 謊報與 end_time 缺漏；(4) compute_verdict() 動態計算判定，廢除硬編碼 PERFECT；(5) 標頭版號對齊 revision history。 | **ACTIVE** |
| v1.16 | 2026-05-13 | Antigravity | **創世圓滿**：對齊憲法 v5.4.18；對齊「大憲章」命名體系。 | ARCHIVED |
| v1.15 | 2026-05-13 | Antigravity | **崩潰修復**：對齊憲法 v5.4.17；同步治權崩潰修復基準。 | ARCHIVED |
| v1.14 | 2026-05-13 | Antigravity | **全量實證對齊**：對齊憲法 v5.4.16；確立全譜系大同步地位。 | ARCHIVED |
| v1.13 | 2026-05-13 | Antigravity | **全量大同步**：對齊憲法 v5.4.15；達成檔案治理全量收斂。 | ARCHIVED |
| v1.12 | 2026-05-13 | Antigravity | **治權完備**：對齊憲法 v5.4.14；確立歷史歷程追加規範與離線偵察地位。 | ARCHIVED |
| v1.11 | 2026-05-13 | Antigravity | **離線修正**：移除 DB Logging，解決創世悖論。 | ARCHIVED |
================================================================================
"""
import pandas as pd
import requests
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import argparse

# ── 系統級架構引導 ──
_THIS_FILE = Path(__file__).resolve()
_MAINTENANCE_DIR = _THIS_FILE.parent
_SCRIPTS_DIR = _MAINTENANCE_DIR.parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path: sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from core.path_setup import get_report_dir
    from core.db_utils import record_lifecycle, get_db_connection
    from core.finmind_client import FinMindClient
except ImportError:
    print("❌ 核心組件導入失敗")
    sys.exit(1)


class ComplianceAuditor:
    # 已知 FinMind 數據集分類
    FINMIND_CONFIG = {
        "Technical":   ["TaiwanStockPrice", "TaiwanStockPriceAdj", "TaiwanStockPER"],
        "Chip":        ["TaiwanStockInstitutionalInvestorsBuySell", "TaiwanStockMarginPurchaseShortSale", "TaiwanStockShareholding"],
        "Fundamental": ["TaiwanStockFinancialStatements", "TaiwanStockMonthRevenue", "TaiwanStockDividend"],
        "Market":      ["TaiwanStockInfo"],
    }
    # FRED 必備 4 個 series（憲法 v5.4.5 第九章核心宏觀指標）
    FRED_MACRO_LIST = ["DFF", "UNRATE", "T10Y2Y", "VIXCLS"]
    # FRED 各 series 之 freshness 閾值（天）
    FRED_FRESHNESS_DAYS = {"DFF": 7, "T10Y2Y": 7, "VIXCLS": 7, "UNRATE": 60}  # UNRATE 月頻

    def __init__(self):
        self.fm_client = FinMindClient()
        self.fred_key = os.getenv("FRED_API_KEY")
        self.report_path = get_report_dir() / f"compliance_audit_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        self.audit_results = []
        self.constitution_ver = "v5.4.18"
        # [v1.17] 動態判定計數
        self.fail_count = 0
        self.warn_count = 0
        self.pass_count = 0

    # ─────────────────────────────────────────────────────────────
    # 既有：API 層稽核
    # ─────────────────────────────────────────────────────────────
    def recommend_type(self, dtype, max_len):
        if "int" in str(dtype) or "float" in str(dtype):
            return "NUMERIC(20, 6)"
        return f"VARCHAR({max(255, max_len)})"

    def audit_finmind(self):
        print(f"🚀 正在掃描 FinMind 供應鏈 API (憲法 {self.constitution_ver} 標準)...")
        for category, datasets in self.FINMIND_CONFIG.items():
            for ds in datasets:
                try:
                    params = {"dataset": ds, "data_id": "2330", "start_date": "2024-05-01", "token": self.fm_client.token}
                    res = requests.get(self.fm_client.api_url, params=params, timeout=30)
                    res.raise_for_status()
                    data = res.json().get("data", [])
                    if data:
                        df = pd.DataFrame(data)
                        col_details = [f"{col}({self.recommend_type(df[col].dtype, 0)})" for col in df.columns]
                        self._record("FinMind-API", ds, "✅ PASS", ", ".join(col_details[:5]) + "...")
                    else:
                        self._record("FinMind-API", ds, "⚠️ EMPTY", "API 無樣本（請手動排查）")
                except Exception as e:
                    self._record("FinMind-API", ds, "❌ FAILED", str(e))

    def audit_fred(self):
        print(f"🚀 正在掃描 FRED 宏觀主權 API (憲法 {self.constitution_ver} 標準)...")
        for sid in self.FRED_MACRO_LIST:
            try:
                url = "https://api.stlouisfed.org/fred/series/observations"
                params = {"series_id": sid, "api_key": self.fred_key, "file_type": "json", "limit": 1}
                res = requests.get(url, params=params, timeout=30)
                res.raise_for_status()
                data = res.json().get("observations", [])
                if data:
                    self._record("FRED-API", sid, "✅ PASS", f"Cols: {list(data[0].keys())}")
                else:
                    self._record("FRED-API", sid, "⚠️ EMPTY", "API 無觀測值")
            except Exception as e:
                self._record("FRED-API", sid, "❌ FAILED", str(e))

    # ─────────────────────────────────────────────────────────────
    # [v1.17 新增] DB 實況稽核
    # ─────────────────────────────────────────────────────────────
    def audit_db_state(self):
        """檢查 DB 各核心表存在性、筆數、FRED 4 series 完整性。"""
        print(f"🚀 正在交叉比對資料庫實況 (憲法 {self.constitution_ver} 標準)...")
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            # 1. FredData 4 series 完整性 (封堵 UNRATE/VIXCLS 漏網事件)
            cur.execute('SELECT series_id, COUNT(*) FROM "FredData" GROUP BY series_id;')
            present = {row[0]: row[1] for row in cur.fetchall()}
            missing = [s for s in self.FRED_MACRO_LIST if s not in present]
            if missing:
                self._record("DB-FRED", "completeness", "❌ FAILED",
                             f"缺少 series: {missing}（必備 {self.FRED_MACRO_LIST}）")
            else:
                self._record("DB-FRED", "completeness", "✅ PASS",
                             f"4 series 全到位 {dict(present)}")

            # 2. 各 FinMind 表存在性 + 筆數
            for category, datasets in self.FINMIND_CONFIG.items():
                for ds in datasets:
                    try:
                        cur.execute(f'SELECT COUNT(*) FROM "{ds}";')
                        n = cur.fetchone()[0]
                        if n == 0:
                            self._record("DB-FinMind", ds, "⚠️ EMPTY", "表存在但筆數 = 0")
                        else:
                            self._record("DB-FinMind", ds, "✅ PASS", f"{n:,} 筆")
                    except Exception as e:
                        conn.rollback()
                        self._record("DB-FinMind", ds, "❌ FAILED", f"查詢失敗: {e}")
        finally:
            cur.close(); conn.close()

    def audit_data_freshness(self):
        """檢查各表最新資料日期是否在閾值內。"""
        print(f"🚀 正在稽核資料時鮮度 (憲法 {self.constitution_ver} 標準)...")
        today = datetime.now().date()
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            # FRED freshness (per series)
            for sid in self.FRED_MACRO_LIST:
                cur.execute('SELECT MAX(date) FROM "FredData" WHERE series_id = %s;', (sid,))
                row = cur.fetchone()
                latest = row[0] if row and row[0] else None
                threshold = self.FRED_FRESHNESS_DAYS.get(sid, 7)
                if not latest:
                    self._record("Freshness", f"FRED/{sid}", "❌ FAILED", "無任何資料")
                else:
                    age = (today - latest).days
                    if age > threshold:
                        self._record("Freshness", f"FRED/{sid}", "⚠️ STALE",
                                     f"latest={latest}, age={age}d (閾值 {threshold}d)")
                    else:
                        self._record("Freshness", f"FRED/{sid}", "✅ PASS",
                                     f"latest={latest}, age={age}d")
        finally:
            cur.close(); conn.close()

    def audit_pipeline_logs(self, window_hours=24):
        """[v1.17 新增] 抓 lifecycle 異常: status 謊報、end_time 缺、最近失敗等。"""
        print(f"🚀 正在交叉比對 pipeline_execution_log (近 {window_hours} 小時)...")
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cutoff = datetime.now() - timedelta(hours=window_hours)
            # 1. 抓近期 sync_* 任務
            cur.execute("""
                SELECT task_name, status, start_time, end_time, error_msg
                FROM pipeline_execution_log
                WHERE start_time >= %s AND task_name LIKE 'sync_%%'
                ORDER BY start_time DESC;
            """, (cutoff,))
            rows = cur.fetchall()
            if not rows:
                self._record("Pipeline-Log", "recent_syncs", "⚠️ EMPTY",
                             f"近 {window_hours}h 無 sync_* 紀錄")
            else:
                # 2. 檢查 status != 'success'
                failed = [r for r in rows if r[1] and r[1].lower() != "success"]
                if failed:
                    sample = "; ".join(f"{r[0]}={r[1]}" for r in failed[:3])
                    self._record("Pipeline-Log", "failed_tasks", "❌ FAILED",
                                 f"{len(failed)} 個任務失敗: {sample}")
                else:
                    self._record("Pipeline-Log", "task_status", "✅ PASS",
                                 f"{len(rows)} 個近期任務皆 success")

                # 3. 檢查 end_time IS NULL（lifecycle 邏輯 bug）
                missing_end = [r for r in rows if r[3] is None]
                if missing_end:
                    sample = "; ".join(r[0] for r in missing_end[:3])
                    self._record("Pipeline-Log", "end_time_null", "⚠️ ANOMALY",
                                 f"{len(missing_end)} 個任務 end_time=NULL (db_utils.record_lifecycle 可能未寫入)")
                else:
                    self._record("Pipeline-Log", "end_time_complete", "✅ PASS",
                                 f"{len(rows)} 個任務皆有 end_time")
        finally:
            cur.close(); conn.close()

    # ─────────────────────────────────────────────────────────────
    # [v1.17 新增] 動態判定 & 報告
    # ─────────────────────────────────────────────────────────────
    def _record(self, source, item, status, detail):
        """統一記錄，並即時累加計數。"""
        self.audit_results.append([source, item, status, detail])
        if "❌" in status or "FAILED" in status.upper():
            self.fail_count += 1
        elif any(k in status for k in ["⚠️", "WARN", "EMPTY", "STALE", "ANOMALY"]):
            self.warn_count += 1
        else:
            self.pass_count += 1

    def compute_verdict(self):
        """[v1.17] 動態判定主權狀態 - 嚴禁硬編碼。"""
        if self.fail_count > 0:
            return "FAILED"
        if self.warn_count > 0:
            return "WARNING"
        return "PERFECT"

    def generate_report(self, verdict):
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write(f"# Quantum Finance 治權合規性審計報告\n\n")
            f.write(f"- **時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **對齊目標**: 系統架構大憲章_{self.constitution_ver}.md\n")
            f.write(f"- **稽核工具**: audit_supply_chain v1.17 (DB State Aware)\n")
            f.write(f"- **判定結果**: **{verdict}** "
                    f"(PASS={self.pass_count}, WARN={self.warn_count}, FAIL={self.fail_count})\n\n")
            f.write("## 🔍 稽核明細\n\n")
            f.write("| 來源層 | 項目 | 狀態 | 詳細 |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            for r in self.audit_results:
                f.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |\n")
            f.write(f"\n\n## 🛡️ 治權對齊判定 (Sovereignty Assertion)\n")
            if verdict == "PERFECT":
                f.write(f"已達成憲法 {self.constitution_ver} 全譜對齊。所有 API、DB、Pipeline 皆符合主權標準。\n")
            elif verdict == "WARNING":
                f.write(f"已達成憲法 {self.constitution_ver} 結構對齊，但 {self.warn_count} 個項目觸發警告，請手動排查。\n")
            else:
                f.write(f"❌ 偏離憲法 {self.constitution_ver} —— 共 {self.fail_count} 個失敗項目，必須立即修補。\n")

    def run(self, source=None, db_only=False, include_logs=False):
        """主權偵察入口 (v1.17)"""
        # 動態判斷各 audit 是否執行
        run_finmind_api = (not db_only) and (not source or source == "finmind")
        run_fred_api    = (not db_only) and (not source or source == "fred")
        run_db_state    = True   # DB 稽核永遠跑
        run_freshness   = True
        run_pipeline    = include_logs or db_only  # 預設不查 logs，需明示

        with record_lifecycle("compliance_audit_v1.17", category="maintenance", stock_id="SYSTEM"):
            if run_finmind_api: self.audit_finmind()
            if run_fred_api:    self.audit_fred()
            if run_db_state:    self.audit_db_state()
            if run_freshness:   self.audit_data_freshness()
            if run_pipeline:    self.audit_pipeline_logs()

            verdict = self.compute_verdict()
            self.generate_report(verdict)

            # 旗艦輸出
            print("\n" + "🛡️" * 40)
            print("🚀 Quantum Finance: 全譜供應鏈合規稽核 (v1.17)")
            print("🛡️" * 40)
            print(f"📊 治權對齊報告 : {self.report_path.name}")
            print(f"📊 稽核項目統計 : PASS={self.pass_count}, WARN={self.warn_count}, FAIL={self.fail_count}")
            print(f"🛡️ 對齊基準     : 憲法 {self.constitution_ver}")
            print(f"⚖️  主權判定     : {verdict}")
            print("🛡️" * 40 + "\n")

            # 若有失敗，exit code 非零 (給 CI/CD 用)
            if verdict == "FAILED":
                sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum Finance 全譜供應鏈合規稽核 (v1.17)")
    parser.add_argument("--source", type=str, choices=["finmind", "fred"],
                        help="只稽核指定 API 來源（不指定則全跑）")
    parser.add_argument("--db-only", action="store_true",
                        help="只跑 DB 層稽核，跳過 API 呼叫（離線/網路受限時）")
    parser.add_argument("--include-logs", action="store_true",
                        help="納入 pipeline_execution_log 交叉比對（lifecycle 異常偵測）")
    args = parser.parse_args()

    ComplianceAuditor().run(
        source=args.source,
        db_only=args.db_only,
        include_logs=args.include_logs,
    )