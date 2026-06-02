# `data_schema.py v2.14 --init --force` 執行實證紀錄

**執行日期**: 2026-05-20 (Asia/Taipei)
**執行版本**: `data_schema v2.14` (commit `0f6268d`)
**對應憲章**: `reports/系統架構大憲章_v6.0.0.md` §二 Step 2 / §3.1 L2440 / §14.7-AH
**裁決**: ✅ **PERFECT ALIGNMENT**

---

## 一、執行背景

本次執行為 v2.14 標頭治權對齊（[Sovereignty Declaration] 核心定義第 6 條補入；
8 項檢查面 100% 合規；commit `0f6268d`）後之**第一次動態實證**，目的：

1. 驗證 v2.14 之 verdict 動態計算邏輯（L393-396）對齊 §5.6.3 [Zero Hardcoded Verdict]
2. 驗證標頭治權對齊（v2.13 → v2.14）未破壞任何 API / DDL / CLI 行為
3. 提供接力指引 §7.7「換機接力第 1 步」之實證證據

---

## 二、執行環境

| 項目 | 值 |
|---|---|
| 工作目錄 | `/home/hugo/project/stock_backend` |
| Python 解譯器 | `venv/bin/python` (Python 3.12) |
| 注意 | 接力指引 §7.7 預設 `.venv/`，但本機實際為 `venv/`，無功能差異 |
| `.env` | 1644 bytes / 6 個關鍵 env 變數（DB_* + FRED_API_KEY + FINMIND_API_TOKEN）present |
| Git HEAD | `e3b1497` (含 `__init__.py v1.16` + `data_schema.py v2.14`) |

---

## 三、執行命令

```bash
venv/bin/python scripts/core/data_schema.py --init --force
```

---

## 四、執行結果（完整輸出）

```
🔎 正在執行 API-first 契約探測 (v6.0.0)...
🛠️  正在啟動主權初始化程序...

🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️
🚀 Quantum Finance: 資料庫主權初始化報告 (v2.14)
🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️
治權基準 : 系統架構大憲章_v6.0.0.md
核心技術 : API Contract First + Absolute Case Sovereignty
────────────────────────────────────────────────────────────────────────────────
✅ [API-PASS] TaiwanStockPrice - 10 columns matched
✅ [API-PASS] TaiwanStockPriceAdj - 10 columns matched
✅ [API-PASS] TaiwanStockPER - 5 columns matched
✅ [API-PASS] TaiwanStockInstitutionalInvestorsBuySell - 5 columns matched
✅ [API-PASS] TaiwanStockMarginPurchaseShortSale - 16 columns matched
✅ [API-PASS] TaiwanStockShareholding - 13 columns matched
✅ [API-PASS] TaiwanStockFinancialStatements - 5 columns matched
✅ [API-PASS] TaiwanStockMonthRevenue - 7 columns matched
✅ [API-PASS] TaiwanStockDividend - 22 columns matched
✅ [API-PASS] TaiwanStockInfo - 5 columns matched
✅ [API-PASS] FredData - 4+1 derived columns matched
────────────────────────────────────────────────────────────────────────────────
✅ [SUCCESS] 表名: "pipeline_execution_log" - 絕對大小寫封印完成
✅ [SUCCESS] 表名: "data_audit_log" - 絕對大小寫封印完成
✅ [SUCCESS] 表名: "TaiwanStockPrice" - 絕對大小寫封印完成
✅ [SUCCESS] 表名: "TaiwanStockPriceAdj" - 絕對大小寫封印完成
✅ [SUCCESS] 表名: "TaiwanStockPER" - 絕對大小寫封印完成
✅ [SUCCESS] 表名: "TaiwanStockInstitutionalInvestorsBuySell" - 絕對大小寫封印完成
✅ [SUCCESS] 表名: "TaiwanStockMarginPurchaseShortSale" - 絕對大小寫封印完成
✅ [SUCCESS] 表名: "TaiwanStockShareholding" - 絕對大小寫封印完成
✅ [SUCCESS] 表名: "TaiwanStockFinancialStatements" - 絕對大小寫封印完成
✅ [SUCCESS] 表名: "TaiwanStockMonthRevenue" - 絕對大小寫封印完成
✅ [SUCCESS] 表名: "TaiwanStockDividend" - 絕對大小寫封印完成
✅ [SUCCESS] 表名: "FredData" - 絕對大小寫封印完成
✅ [SUCCESS] 表名: "TaiwanStockInfo" - 絕對大小寫封印完成
────────────────────────────────────────────────────────────────────────────────
🔎 API PASS/WARN/FAIL : 11/0/0
📈 總計項目 : 13
✅ 成功重鑄 : 13
❌ 失敗項目 : 0
🧱 DDL 執行 : YES
🕒 總計耗時 : 6719.94 ms
⚖️  主權判定 : PERFECT ALIGNMENT
🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️🛡️

=== EXIT CODE: 0 ===
```

---

## 五、結果裁決

### 5.1 API Contract First Probe（Step A）

| 指標 | 結果 | 對應憲章 |
|---|---|---|
| FinMind 10 dataset probe | **10/10 PASS** | §二 Step 2 + §3.1 L2440 |
| FRED `DFF` series probe | **1/1 PASS**（4 API 欄位 + 1 local derived `series_id`）| §6.0A L2706+ |
| 總計 PASS/WARN/FAIL | **11/0/0** | §5.6.3 動態判定 |
| 欄位大小寫對齊 | 全 13 張表 1:1 對齊 API 原始大小寫 | §1.4 Absolute Case Sovereignty |

### 5.2 DDL 主權重鑄（Step B）

| 表類別 | 表數 | 結果 |
|---|---|---|
| Infrastructure | 2 (`pipeline_execution_log` + `data_audit_log`) | 全 SUCCESS |
| FinMind Raw | 10 | 全 SUCCESS |
| FRED Macro | 1 (`FredData`) | SUCCESS |
| **合計** | **13** | **13/0 SUCCESS** |

每張表完成「絕對大小寫封印」+ unique constraint（依 `DATASET_REGISTRY`）+ `idx_*_date`
索引（非 log 表）。

### 5.3 Verdict 動態計算（§5.6.3 對齊驗證）

依 L393-396 之 verdict 邏輯：

```python
verdict = "PERFECT ALIGNMENT" if self.stats["failed"] == 0 and self.contract_stats["failed"] == 0 else "FAILED"
if verdict == "PERFECT ALIGNMENT" and self.contract_stats["warn"] > 0:
    verdict = "WARNING"
```

實況 `failed=0` + `warn=0` → 動態計算為 **PERFECT ALIGNMENT**（非硬編碼）。
**[Zero Hardcoded Verdict] 條款落地驗證通過**（v2.13 第 5 條 + v2.14 第 6 條對齊）。

### 5.4 Exit Code 對齊 §3.2 Step 2 接受標準

| Verdict | 預期 exit code | 實際 exit code |
|---|---|---|
| PERFECT ALIGNMENT | 0 | **0** ✅ |

對齊 §3.2 Step 2 / Step 2C「PERFECT/WARNING → exit 0，FAILED → exit 1」之分流契約。

### 5.5 耗時

| 項目 | 數值 |
|---|---|
| 總計耗時 | **6,719.94 ms**（6.7 秒） |
| API contract probe | ~6 秒（11 個 HTTP 請求，timeout 30s） |
| DDL 重鑄 + commit | ~0.7 秒（13 張表，每張含 DROP/CREATE/UNIQUE/INDEX） |

效能與 §14.1〜§14.3 歷次空 DB 重建紀錄（PERFECT ALIGNMENT 區段 2,800〜10,000 ms）一致。

---

## 六、v2.14 標頭治權對齊驗證

| 標頭聲明 | 實況驗證 |
|---|---|
| `data_schema.py v2.14` (報告 header) | ✅ 顯示「資料庫主權初始化報告 (v2.14)」 |
| `CONSTITUTION_VER = "v6.0.0"` | ✅ 顯示「治權基準 : 系統架構大憲章_v6.0.0.md」 |
| `self.constitution_ver = "v6.0.0"` | ✅ probe 訊息「(v6.0.0)」 |
| `TOOL_VER = "v2.14"` | ✅ 與 header 同步 |
| 第 1 條 [API Contract First] | ✅ Step A 先於 Step B 執行 |
| 第 2 條 [Absolute Case Sovereignty] | ✅ 全 13 表「絕對大小寫封印完成」 |
| 第 3 條 [Defensive Architecture] | ✅ VARCHAR(255) / NUMERIC(20,6) DDL 重鑄 |
| 第 4 條 [Hybrid Observability] | ✅ pipeline_execution_log + 詳細終端報告 |
| 第 5 條 [Zero Hardcoded Verdict] | ✅ verdict 動態計算 = PERFECT ALIGNMENT（非硬編碼） |
| 第 6 條 [Sovereignty Declaration] | ✅ 執行未觸發任何 §0.1-A / §0.2-A / §0.3-A / §0.0-E.4 / §0.0-F.3 禁令；未介入時間序列；未調度 universe |

---

## 七、後續執行建議

依憲章 §二 + per_program_audit §7.7 之九步序列，本 Step 2 已完成，後續：

| 順位 | 指令 | 對應 |
|---|---|---|
| **Step 2B** | `venv/bin/python scripts/core/core_universe_schema.py --init` | 7 張 §6.7 治理 schema |
| Step 2C | `venv/bin/python scripts/core/db_utils.py` | 前置依賴檢查（預期 bootstrap WARNING：committed snapshot 0 rows） |
| Step 3 | `venv/bin/python scripts/maintenance/audit_supply_chain.py --include-logs` | 供應鏈驗收 |
| HUB（非必須） | `venv/bin/python scripts/core/__init__.py` | 四層動態稽核（驗證 v1.16 hub）|

---

## 八、結論

✅ **`data_schema.py v2.14` 動態實證通過**：

1. v2.14 標頭治權對齊（[Sovereignty Declaration] 核心定義第 6 條 + TOOL_VER v2.14）
   未破壞任何 API / DDL / CLI 行為
2. verdict 動態計算邏輯（L393-396）對齊 §5.6.3 [Zero Hardcoded Verdict]，
   PERFECT ALIGNMENT 為實況計算結果而非硬編碼
3. 11/0/0 + 13/0 + DDL=YES + exit=0 滿足 §3.2 Step 2 接受標準
4. 本實證可作為 §14.7-AH 之動態落地證據，亦為跨平台接力指引 §7.7「換機接力第 1 步」
   之正式回報

**HEAD commit**: `e3b1497`（含 `__init__.py v1.16` + `data_schema.py v2.14`）
**下一步建議**: Step 2B `core_universe_schema.py --init` 或進入 Step 1.1.3 元件審計
