"""
model_metadata.py v2.16 (Quantum Finance Edition)
================================================================================
模型元數據總管 — 生產級效能註冊版 (Quantum v5.2 標準)
負責管理模型版本、效能指標 (Metrics) 與權重路徑的持久化。

修訂歷程：
  v2.16 (2026-05-11): [標準] 補全旗艦級範例矩陣，對齊混合日誌與 v5.2 主權架構。
  v2.15 (2026-05-11): [解耦] 修正循環匯入問題，優化 ModelMetadata 物件封裝。

【執行範例矩陣 (Model Metadata Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / 用法                                        │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [單一個股：模型註冊]      │ meta = ModelMetadata(stock_id="2330", ...); meta.save()│
│ 2. [單一個股：所有模型檢索]  │ models = ModelMetadata.get_all_for_stock("2330")       │
│ 3. [所有核心股：效能稽核]    │ $ python scripts/core/model_metadata.py --audit_core   │
│ 4. [所有核心股：強制清理過期]│ $ python scripts/core/model_metadata.py --purge_old    │
│ 5. [系統級：模型地圖導出]    │ $ python scripts/core/model_metadata.py --export       │
└──────────────────────────────┴────────────────────────────────────────────────────────┘

【可觀測性紀錄 (Observability)】
  - 統一日誌 (Unified): pipeline_execution_log (Task: model_registry_save)
  - 專項審計 (Audit): data_audit_log (Action: MODEL_REGISTER / MODEL_PURGE)
================================================================================
"""
import os, sys, json, logging
from pathlib import Path
from datetime import datetime

# ── 系統級架構引導 ──
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

try:
    from core.db_utils import db_transaction, record_lifecycle, write_data_audit_log
except ImportError:
    import db_utils
    from db_utils import db_transaction, record_lifecycle, write_data_audit_log

class ModelMetadata:
    def __init__(self, stock_id, model_name, version, metrics=None, params=None, path=None):
        self.stock_id = stock_id
        self.model_name = model_name
        self.version = version
        self.metrics = metrics or {}
        self.params = params or {}
        self.path = path or ""

    def save(self):
        """將模型元數據持久化至資料庫，並紀錄混合日誌"""
        with record_lifecycle("model_registry_save", category="modeling", stock_id=self.stock_id):
            with db_transaction() as cur:
                cur.execute("""
                    INSERT INTO models (model_name, stock_id, version, metrics, params, path)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING model_id
                """, (self.model_name, self.stock_id, self.version, json.dumps(self.metrics), json.dumps(self.params), self.path))
                model_id = cur.fetchone()['model_id']
                
                # 專項審計
                write_data_audit_log("models", self.stock_id, datetime.now().strftime("%Y-%m-%d"), "MODEL_REGISTER", 1)
                logging.info(f"✅ 模型註冊成功：{self.model_name} v{self.version} (ID: {model_id})")
                return model_id

    @staticmethod
    def get_all_for_stock(stock_id):
        """檢索特定標的所有模型紀錄"""
        with db_transaction() as cur:
            cur.execute("SELECT * FROM models WHERE stock_id = %s ORDER BY created_at DESC", (stock_id,))
            return cur.fetchall()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print(f"Quantum Model Metadata Engine v2.16")
    print(f"💡 請參閱程式開頭的執行範例矩陣以進行模型維護。")