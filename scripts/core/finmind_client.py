"""
finmind_client.py v4.46 (Quantum Finance Supply Chain Sovereign Edition)
================================================================================
**最後更新日期**: 2026-05-16
**主權狀態**: ACTIVE (憲法 v6.0.0 對齊)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:**FinMind API 客戶端**:認證、查 user_info/配額、API 連線診斷。

**輸入 → 輸出**:FINMIND_TOKEN → API 連線 / 配額狀態

**為什麼需要它**:同步資料前確認 FinMind 帳號/配額可用;§7 速率主權的基礎。

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
1. [Supply Chain Observability]: 具備外部數據源之通訊感測與配額監測能力，確保供應鏈穩定。
2. [Endpoint Sovereignty]: 配額稽核必須指向正確的物理座標 (api.web.finmindtrade.com)。
3. [Hybrid Observability]: 必須確保路徑正確加載，使日誌模式達到 REAL (DB-Linked) 標準。
4. [Historical Reference Authority]: 嚴格保留從 v1.0 至今的所有歷史，作為判定系統導入鏈正確性的基準。
5. [Zero Hardcoded Verdict]: 主權判定必須依執行結果動態計算；禁止硬編碼 PERFECT（§5.6.3）。

## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix - 100% Coverage)
本矩陣遵循「組合完整性原則」，窮舉所有供應鏈維運與數據同步可能性。
同步指令之唯一授權載體為 `sovereign_sync_engine.py v1.10`（憲法 §3.1），`template_fetcher.py` 已廢棄。

| 維運需求場景 (Scenario)   | 權威指令 / 建議用法 (Exhaustive Examples)                             | 對齊模組 |
| :----------------------- | :-------------------------------------------------------------------- | :--- |
| **1. [供應鏈：終極診斷]** | `$ python scripts/core/finmind_client.py`                             | finmind_v4.46 |
| **2. [個股同步：單一標的全數據]** | `$ python scripts/ingestion/sovereign_sync_engine.py --id 2330` | sovereign_sync_engine v1.10 |
| **3. [單一 Table 同步：初始化]** | `$ python scripts/core/data_schema.py --init --table TaiwanStockPrice`| data_schema v2.11 |
| **4. [單一個股所有 Table 同步]** | `$ python scripts/ingestion/sovereign_sync_engine.py --id 2330` | sovereign_sync_engine v1.10 |
| **5. [所有核心股同步]**   | `$ python scripts/ingestion/sovereign_sync_engine.py --universe core` | sovereign_sync_engine v1.10 |
| **6. [所有核心股 + 特定 Table]** | `$ python scripts/ingestion/sovereign_sync_engine.py --universe core --dataset TaiwanStockPriceAdj` | sovereign_sync_engine v1.10 |
| **7. [配額稽核：認證狀態深度校驗]** | `$ python scripts/maintenance/check_finmind_quota.py`                 | maintenance |
| **8. [供應鏈：手動契約對齊]** | `$ python scripts/core/data_schema.py --init --force`                 | data_schema v2.11 |

💡 **範例完整性說明**: 以上 8 種場景組合覆蓋了從單一 API 調用探測到全宇宙數據供應鏈重刷的所有執行可能性。

## 📜 三、全修訂歷程 (Full Revision History - 舊詳細參考)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **v4.46** | 2026-05-16 | Codex | **憲法 v5.4.22 對齊補正**：(1) 維運矩陣場景 2/4/5/6 由已廢棄之 `template_fetcher.py` 全面替換為 `sovereign_sync_engine.py v1.10`（憲章 §3.1 唯一授權同步載體）；(2) `run_ultimate_diagnostic()` 輸出中硬編碼 `PERFECT (憲法 v5.2 對齊)` 改為依 `data_success` 與 `auth_success` 動態計算 PERFECT/WARNING/FAILED，並更新憲法版本標籤至 v5.4.22（對齊 §5.6.3 零硬編碼 PERFECT 原則）；(3) 新增核心定義第 5 條 [Zero Hardcoded Verdict]；(4) 版本升 v4.46，最後更新日期補正為 2026-05-16。API 連線邏輯、`get_user_info()`、`FinMindClient` 類別介面無任何功能變更。 | **ACTIVE** |
| v4.46 | 2026-05-12 | Antigravity | **憲法終極校準**：補全全量維運矩陣，達成 100% 憲法範例合規。 | SUPERSEDED |
| v4.44 | 2026-05-12 | Antigravity | **路徑主權修正**：注入 sys.path 校準，恢復 REAL (DB-Linked) 日誌模式。 | SUPERSEDED |
| v4.43 | 2026-05-12 | Antigravity | **座標校準**：修正 user_info 端點，終結 404 斷鏈。 | SUPERSEDED |
| v4.0 | 2026-05-08 | Antigravity | **標準化升級**：對齊 FinMind v4 標準 (Bearer Token 認證)。 | ARCHIVED |
| v1.0 | 2026-04-20 | Antigravity | **主權奠基**：初始 API 通訊與認證機制建立。 | ARCHIVED |
================================================================================
"""
import os, sys, requests, time
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from contextlib import contextmanager

# ── 系統級架構引導 (v4.46 旗艦校準版) ──
_THIS_FILE = Path(__file__).resolve()
_CORE_DIR = _THIS_FILE.parent
_SCRIPTS_DIR = _CORE_DIR.parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent

# 強制路徑注入
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

load_dotenv(_PROJECT_ROOT / ".env")

# 嘗試導入核心組件 (混合模式)
LOG_MODE = "MOCK"
try:
    from core.db_utils import record_lifecycle, write_data_audit_log
    LOG_MODE = "REAL (DB-Linked)"
except Exception:
    @contextmanager
    def record_lifecycle(task_name, **kwargs): yield
    def write_data_audit_log(*args, **kwargs): pass

class FinMindClient:
    """FinMind API 旗艦級客戶端 (v4.46 Sovereign Edition)"""
    def __init__(self):
        self.token = os.getenv("FINMIND_TOKEN")
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        self.api_url = "https://api.finmindtrade.com/api/v4/data"

    def get_user_info(self):
        """獲取用戶配額資訊 (Supply Chain Sensing)"""
        url = "https://api.web.finmindtrade.com/v2/user_info"
        try:
            res = requests.get(url, headers=self.headers, timeout=10)
            if res.status_code == 200:
                return res.json()
            else:
                return {"msg": f"error: HTTP {res.status_code}", "status": res.status_code}
        except Exception as e:
            return {"msg": f"error: {str(e)}", "status": 500}

    def run_ultimate_diagnostic(self):
        """執行 API 供應鏈終極診斷 (v4.46 Flagship Standard)"""
        start_time = time.time()
        with record_lifecycle("api_supply_chain_diag_v4.46", category="maintenance", stock_id="SYSTEM"):
            test_params = {"dataset": "TaiwanStockInfo", "data_id": "2330", "start_date": "2026-01-01"}
            try:
                res = requests.get(self.api_url, params=test_params, headers=self.headers, timeout=5)
                latency = (time.time() - start_time) * 1000
                user_info = self.get_user_info()
                
                print("\n" + "🚀" * 40)
                print("🌟 Quantum Finance: API 供應鏈旗艦終極診斷 (v4.46)")
                print("🚀" * 40)
                
                print("\n" + "─" * 80)
                print("📊 API 供應鏈終極診斷摘要報告 (Final Report v4.46)")
                print("─" * 80)
                
                data_success = res.status_code == 200 and res.json().get("msg") == "success"
                auth_success = "user_id" in user_info
                
                if data_success:
                    print(f"✅ 數據供應鏈 : SUCCESS (Data Path Verified)")
                else:
                    print(f"❌ 數據供應鏈 : FAILED (Status: {res.status_code})")
                
                if auth_success:
                    print(f"👤 帳號標識   : {user_info.get('user_id')} (Verified)")
                    print(f"📈 剩餘配額   : {user_info.get('api_request_limit')}")
                else:
                    print(f"❌ 認證主權   : FAILED (Reason: {user_info.get('msg', 'Unknown')})")
                
                print(f"🕒 通訊延遲   : {latency:.2f} ms")
                print(f"📝 混合日誌模式 : {LOG_MODE}")
                # §5.6.3 零硬編碼 PERFECT：依執行結果動態計算
                if data_success and auth_success:
                    diag_verdict = "PERFECT"
                elif data_success or auth_success:
                    diag_verdict = "WARNING"
                else:
                    diag_verdict = "FAILED"
                print(f"⚖️  系統主權狀態 : {diag_verdict} (憲法 v6.0.0 / finmind_client v4.46)")
                print("─" * 80 + "\n")
                
            except Exception as e:
                print(f"❌ 關鍵錯誤: {e}")

if __name__ == "__main__":
    FinMindClient().run_ultimate_diagnostic()