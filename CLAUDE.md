# CLAUDE.md — AI 協作工具規則 (Single Source of Truth)

**文件性質**：本檔為本專案 AI 協作工具規則之單一事實來源 (SSOT)。
**位階聲明**：所有**治權禁令、工程契約、架構治權**以 `reports/系統架構大憲章_v6.0.0.md`（以下簡稱「憲章」）為準；本檔僅承載「**如何使用 AI 工具編輯本專案**」之短半衰期協作規則。
**最後更新**：2026-05-30(§一.13 v0.15 amihud Dead-Feature Removal 入憲[用戶授權「移除 amihud(最乾淨)→ SPEC 38 → 37」+ 兩 AskUserQuestion 確認「全系統一致」/「刪除 amihud rows」;amihud = zero-variance dead feature **非 AI 幻像**,398 core DB-verified distinct=1 / 全 0.00000000;全 28 code files surgical removal[py_compile pass]+ DB v0.5-family DELETE 95840 rows + feature_count 38→37[guarded-transaction];T_DC-28 Dead-Feature Removal Treaty 入憲;既有 model 未 retrain + 無 git commit 之 honest disclosure;雙層治權鎖 主憲章 §14.7-DC v0.15 + §一.13 v0.15 同次入憲];前序 §一.13 v0.14 Trainer-Feature-Set-Binding Corrective 入憲[用戶授權 Task C「修正 code + 入憲」,**非 strictest re-check 循環**];Finding:§14.7-DC 治權契約 3「model_trainer inherent 已對齊」為 incomplete —— universe-membership 繼承乾淨 ≠ feature-value-source 乾淨;10 production trainers 之 `DEFAULT_PANELS` 原預設 legacy `feature_set_v0_4`(含 5 已移除 AI 幻像 feature + 3 imputed core)→ repoint recent-8 v0_5 + 載入 query 補 `is_null_imputed IS NOT TRUE` 防禦過濾[10/10 grep + py_compile 驗證];committed snapshot + 1001 membership rows explicit bind `feature_set_v0_5`[psql 0 residual NULL 驗證];8 v0_5 panel source-purity psql 驗證 398 core / 38 features / 0 imputed each;T_DC-27 Trainer-Feature-Set-Binding Treaty 入憲;heavy retrain 未授權不 auto-run[§一.12 + §二.6];雙層治權鎖 主憲章 §14.7-DC v0.14 + CLAUDE.md §一.13 v0.14 同次入憲)

依憲章 §0.0-F（AI 協作工具規則治權位階錨點），本檔受憲章 §0.0-E.4 統合層治權元規則間接約束，但不上升至憲章治權位階。

---

## 一、通用基礎規則 (General Rules)

### 1. Read Before Edit
- 任何 `Edit` / `Write` 工具呼叫前，**必須先用 `Read` 工具讀過該檔案**。
- 不可憑記憶或推測編輯，包含小修改。
- 編輯時必須精確匹配既有內容，含縮排與空白。

### 2. Edit 優先於 Write
- 既有檔案修改一律使用 `Edit`（只送 diff，安全可審查）。
- `Write` 會整個覆蓋檔案，**只用於建立新檔案或極端情況下的完整重寫**。

### 3. 最小邊界原則
- 只改使用者要求的部分。
- **不順手重構、不順便清理、不額外加註解**。
- 不為「假想的未來需求」加 abstraction / helper。
- 修一個 bug 不順便美化周圍程式碼。

### 4. 註解最小化
- 預設**不寫註解**。
- 只在「why 非顯而易見」時寫一行（規避某個 bug、隱藏的不變式、特殊邊界）。
- 不寫「what」註解（程式碼本身已說明）。
- 不寫有時效性的註解（「為此任務新增」「給 X 使用」）。

### 5. 不引入安全漏洞
- 注意 OWASP Top 10（SQL injection、XSS、command injection 等）。
- 寫入外部系統前驗證輸入。
- 不在 commit 中含 `.env` / credentials / API key。

### 6. 不確定就停手問
- 任何破壞性操作（`rm -rf`、`git push --force`、`git reset --hard`、刪除分支、改 CI）**必須先確認**。
- 任何影響共用狀態的操作（推送、發 PR、改基礎設施）**必須先確認**。
- 使用者授權某次操作 ≠ 授權所有同類操作。

### 7. UI / Frontend 改動需實測
- 不能只靠 type check / unit test 宣稱完成。
- 必須實際啟動 dev server、在瀏覽器測試。
- 若無法測 UI，**明確說明「未測試」**，不可佯稱成功。

### 8. 報告誠實
- 描述「做了什麼」而非「打算做什麼」。
- 工具呼叫失敗要明說。
- 不假裝完成未完成的工作。

### 9. 外部資源驗證 protocol(2026-05-25 §14.7-AX(E) / §14.7-BI/BJ 雙重實證入憲)

任何認定**外部資源不可用**(API paywall / quota cap / tier insufficient)之前,**必須**:

1. **直接調用 user_info / quota_info API**(例如 FinMind `/api/v4/user_info`),取得實際 tier + quota + subscription status;**不可從 sync error 直接推測 tier**(error 可能來自 hourly window timing)
2. **若 API 無 user_info endpoint**:跑單股探測 + 等候 hourly window 重置(通常 1 hour)後重試,仍 fail 才認定資源不可用
3. **明確區分「tier 不足(永久 blocked)」vs「quota 暫時耗盡(可重試)」**
4. **入憲記述須區分上述兩類**(Path D vs Path A retry)

**證據基礎**:§14.7-BJ ROE Path A 第 1 次嘗試誤將 sponsor 認為 free user → 入憲 Path D dropped;§14.7-BI 第 2 次 user_info verify 後確認 sponsor 仍 active → Path A' SUCCESS / V 動員度 64% → 73%。違反 #9 protocol 將導致治權誤判 + 重複入憲成本。

### 10. 資料真實性 / No Data Hallucination(2026-05-28 §14.7-CW v6.17.1 patch precedent 入憲)

任何 AI 產出之數據(metrics / IC / Sharpe / Win rate / α / MDD / 統計量 / 表格 / 引用值 / 對比基準)**必須具備真實來源依據**,**不可 AI 自動生成數據幻像**(用戶 2026-05-28 explicit directive)。

**1. 真實來源依據 — 三類唯一允許 source**

全部產出數據必須出自以下三類來源之一,**不可從第四類產生**:

- **(a) 程式輸出**:scripts 跑出之 stdout / JSON file / log file(如 `data/models/<model_id>/metrics.json` / dry-run terminal output)
- **(b) DB query**:活 PostgreSQL query 結果(SELECT statement actual output rows)
- **(c) API response**:FinMind(`api.finmindtrade.com`)/ FRED(`api.stlouisfed.org`)/ Anthropic API JSON response

**禁止來源(第四類)**:從記憶 / 推測 / 估算 / "合理猜測" / 從相似 model 推估 / "為 charter 完整性" 補 placeholder 數字。

**2. 數據來源 traceability — 每數字可 trace 回 (a)(b)(c)**

Charter inscription / commit message / 報告 / 對話回覆 中之 **每個量化數字** 必須能 trace 回 (a)(b)(c) 三類其中之一;不可:

- 從記憶 / 推測填數字(即使數字看起來合理)
- "為了 charter 完整性" 補 placeholder 數字
- 從相似 model 推估數字(必須 actual run)
- 對比表中 baseline 與 new 兩邊任一邊缺真實來源

**3. Stochastic metrics 多次跑取 statistics**(per §14.7-CW T_CW-6 / v6.17.1 patch 之直接 enforcement)

任何含 stochasticity(LGBM bagging / sklearn random_state / dropout / multi-thread sub-sampling)之 production metric:

- **必須 ≥ 3 runs** 取得 statistics(min / median / max / mean / spread)
- **不得以 single-run anchor 寫入 charter 為 deterministic fact**
- Charter inscription 須含 multi-run range + median + mean,**single anchor 須註明為 max / min outlier**

**4. 違反入憲先例(2026-05-28)— §14.7-CW v6.17.0 → v6.17.1 patch**

§14.7-CW v6.17.0 將 LGBM 單次 commit run Sharpe 4.74 / IR 5.86 / α +16.22% 寫入 charter 為 deterministic fact;**v6.17.1 patch honest audit 後揭露真實 6-run range Sharpe 3.71-4.74 / median 3.90**;committed anchor 為 distribution 之 max outlier(非 mean);charter inscription 修正為 reproducibility-aware + T_CW-6 補入(治權條 5 → 6)+ 入憲規則 4 → 5 條。

**此為違反 #10 之公開治權先例**,future session 不得重蹈;違反者 charter inscription deemed misleading + 必須 reproducibility patch correction(同 v6.17.1 流程)。

**5. Hallucination 警示信號(self-audit checklist)**

寫任何量化數字前自問:

- 「這個 `4.74` 是誰跑出來的?哪個 file 寫過?」
- 若無 source file / DB row / API response → **STOP**,跑實證再寫
- 若 source 為 single run 而 system 含 stochasticity → **補跑 ≥ 3 runs 再 inscribe**
- 對比表中 "v0.1 vs v0.2" 數字必須兩邊都有真實來源(不可 v0.1 從記憶 / v0.2 從程式跑 — 此 asymmetry 為 hallucination 高風險區)

**6. Inscription 執行檢查清單(每次量化數字寫入 charter / commit / 報告之強制清單)**

- [ ] 數字 source file 已可讀(指向具體 file path 或 DB query / API endpoint)
- [ ] 若 stochastic,已跑 ≥ 3 runs(min / median / max / mean 全有)
- [ ] Charter inscription 含 range + median(**非** single anchor)
- [ ] 對比基準(baseline)也有真實來源(非估算)
- [ ] 異常值已揭露(commit anchor 為 max / min outlier 須註明)
- [ ] 分母 / 評估方法 / 樣本大小 對等(不對等須揭露)

**證據基礎(本條入憲)**:用戶 2026-05-28 explicit directive「Win rate 83.3% 是否真實?」之追問之後揭露 v6.17.0 charter Sharpe 4.74 為 single-run max anchor(實測 6-run range 3.71-4.74 / median 3.90),違反數據真實性;v6.17.1 reproducibility transparency patch 修正 charter + T_CW-6 入憲(commit `6da6110` / tag `v6.17.1-section14-7-CW-reproducibility-transparency-patch-20260528`)為直接 precedent;**用戶 explicit directive 「所有產出的數據都具備真實來源依據,不要 AI 自動產生數據幻像」即本條 #10 之入憲動因**。

### 11. 程式標頭強制三段式(Mandatory Header Three-Section Convention)(2026-05-29 用戶 explicit directive 入憲)

任何新建 Python script 之 file-level docstring **必須**包含以下三段式(per sovereign_sync_engine.py 範本):

**三段式強制內容**:

#### 第一段:📜 一、核心定義說明 (Core Definitions / The Constitution)

- 程式之治權邊界 / 設計原則 numbered list(typically 5-15 條)
- 每條格式:`N. **[Concept Name]** (vX.Y, 憲法 §X.Y): 一句話描述 + 治權契約引用`
- 必須包含治權慣例 5 條:
  - **[Sovereignty Declaration]**:本程式對應 §3.1 序列模組 / §3.2 橫切模組 / §3.2A 等之身分自我宣告 + 治權邊界 6-8 點
  - **[Zero Hardcoded Verdict]**(per §5.6.3):verdict 動態計算,不硬編 PERFECT/WARNING/FAILED
  - **[Historical Reference Authority]**:此程式之 schema_ver / TOOL_VER 屬於記述性快照,非權威來源
  - **[Idempotency]**:支援重跑(ON CONFLICT upsert / status='committed' 等保護)
  - **模組特有條**:對映本程式核心功能(如 [Schema Sovereignty] / [Zero Silent Drop] / [Anti-Leakage] 等)

#### 第二段:📊 二、全量功能群矩陣 OR 全量維運指令總矩陣 (Functional Group Matrix OR Operational Matrix)

允許二擇一(per 既有 production scripts 慣例兩種 valid format 並存):

**Variant A**:`## 📊 二、全量功能群矩陣 (The Ultimate Functional Group Matrix)`
- 依「業務功能群」(非依 method 結構)拆分為 6-10 大 Group(A, B, C, D...)
- 適用於:功能多元化之 production engine(sovereign_sync_engine.py / multi_cycle_validation.py)

**Variant B**:`## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix)`
- 依「維運指令場景」(A. 主要運行模式 / B. 補充運行模式 / ...)拆分
- 適用於:CLI-mode-driven scripts(core_universe_builder.py / feature_store_builder.py / model_trainer.py / prediction_engine.py / portfolio_sizer.py)

**共同要求**(兩 variant 皆須):
- 每段對應一個表格,columns:`子項 | 對應方法/行為 / CLI / 治權契約`
- 必須含對齊「憲章 §二 維運矩陣 標準場景索引」
- 不提供之旗標 / 模式 sub-section:明示 intentionally omitted + 理由

#### 第三段:📜 三、全修訂歷程 (Full Revision History)

- 完整 Markdown 表格 columns:`| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |`
- v1.0 → latest 全保留(舊版改 SUPERSEDED / ARCHIVED 狀態,不刪除)
- 修訂說明須含:**功能變更 N 點** + 治權契約引用 + 介面相容性說明

**標頭頂部 6 行強制 metadata**:

```python
"""
<program_name>.py vX.Y (<descriptive subtitle>)
================================================================================
**最後更新日期**: YYYY-MM-DD
**主權狀態**: <ALL CAPS SUMMARY 含所有重要治權契約引用>
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)
```

**為何強制此格式**(per 用戶 2026-05-29 directive):

1. **治權自我宣告**:每個 production script 須清楚標示其在 §3.1/§3.2 序列/橫切位置,以及對應憲章 §
2. **功能可發現性**:後續 maintainer / 跨機接續 / AI 協作均可從標頭立即理解程式之全功能範圍
3. **修訂可追溯**:全 revision history 保留為 audit trail,對齊 §一.10 「真實來源依據」治權
4. **8 項標頭評估**(per §四.4)依本條為創建基準,非僅評估基準
5. **AI 創建 script 違反 §一.11 視為治權違規**,須補正後才可 commit/push

**現行範本檔案**(per sovereign_sync_engine.py v1.22):`scripts/ingestion/sovereign_sync_engine.py`(L1-230 為標頭三段式 + 修訂歷程)。任何新 script 應 cross-reference 此檔以校準格式。

**證據基礎(本條入憲)**:用戶 2026-05-29 explicit directive「為什麼你現在所產生的程式又没有核心定義說明、全量功能群矩陣、全修訂歷程,像 sovereign_sync_engine.py 這支程式寫的這麼詳細,例如 multi_cycle_xgboost_validation.py。請重新檢查依目前憲章內所有需要執行的程式確實的補入...並把所有程式開頭要寫清楚完整的核心定義說明、全量功能群矩陣、全修訂歷程 這個作法寫到 claude.md」之直接入憲。違反者:本 session 5 個 scripts(model_trainer_lgbm_v2.py / model_trainer_xgboost.py / build_historical_panels.py / multi_cycle_validation.py / multi_cycle_xgboost_validation.py)均違反本條,須補正。

### 12. Long-Running 訓練 / Multi-cycle 工作流程 5 分鐘回報節奏(2026-05-29 用戶 explicit directive 入憲)

任何 AI 觸發之 **estimated runtime ≥ 5 分鐘 之 long-running workflow**(model training / multi-cycle validation / large backtest / from-zero rebuild)**必須每 5 分鐘對用戶回報一次進度**,避免用戶等待中誤判 AI 已 stall 或失敗。

**1. 強制適用範圍**

凡 estimated runtime ≥ 5 分鐘之以下任何類別,須啟動 5-min 回報:

- **Model training**:8-panel walk-forward / 95-panel walk-forward / cross-machine rebuild / hyperparameter sweep
- **Multi-cycle validation**:4-horizon 全跑(weekly + monthly + quarterly + annual)/ 任一 horizon × ≥ 60 panels
- **Walk-forward backtest**:跨年 expanding window evaluation
- **From-zero rebuild**:全市場 sync / feature_store rebuild / 大規模 audit
- **Cross-model comparison run**:同 SPEC × 多 model 串跑

**2. 回報內容強制最低(每 5 分鐘)**

每 5 分鐘回報須含:

- **已完成階段**:已跑完哪些 step / horizon / fold
- **當前進行階段** + **已 elapsed time**(分鐘精度)
- **剩餘 estimated time**(基於既往 horizons 之 per-horizon 速度推估)
- **已知 metrics**:若 horizons 已完成,列已完成之 Sharpe / Eff t / Win Rate / 其他 standardized metrics
- **任何 warning / error 即時提報**(不等下一個 5-min cycle)

**3. 實作方式**

優先順序:

1. **`Monitor` tool 之 sleep 300 loop**(per CLAUDE.md §五 implicit)— 每 300s emit 一次 progress line + grep filter 避免 burn context
2. **`Bash run_in_background`** 啟動主 task + 對 log file 之週期 grep status report
3. **`ScheduleWakeup` delay 300s**(REPL idle 才 fire,作為 fallback)
4. ❌ **不可** `tail -f` 整個 log(會 burn context)
5. ❌ **不可** 在 ≥ 5 分鐘 task 後完全靜默(violation)

**4. 違反治權**

- 若 ≥ 5 分鐘 task 無 5-min 回報,視為 **§一.8「報告誠實」violation**
- 若用戶詢問「仍在跑嗎?」「進度如何?」之後,回報未含具體 elapsed time + completed steps + remaining estimate = **額外 violation**

**5. 與 §二.6 SHMM 之關係(雙層治權並存)**

| 層 | 規則 | 觸發門檻 | 目標 |
| :--- | :--- | :--- | :--- |
| 系統層 | §二.6 SHMM | ≥ 30 分鐘 long-running workflow | self-healing / system uptime / agent revival |
| 用戶體驗層 | **§一.12 5-min progress report** | **≥ 5 分鐘 任何 AI-triggered task** | **user-facing feedback / 防 stall 誤判** |

兩規則並存無衝突:SHMM 為 system 自身存活機制,§一.12 為 user-facing 進度可視性。

**證據基礎(本條入憲)**:用戶 2026-05-29 explicit directive「之後所有的模型訓練都每 5 分鐘回報一次,寫入憲章」之直接入憲。**precedent**:本 session LightGBM dedicated multi-cycle 跑 ~4 分鐘 用戶詢問 "仍在做模型訓練嗎?" 觸發此入憲。違反者:**本 session 之 RF / ET multi-cycle 跑 7.8 / 4.6 分鐘均靜默無回報**,違反本條(回溯適用,未來不得重蹈)。

### 13. Source-Pure Universe Doctrine — Imputed Stocks 必須排除 core_universe(§14.7-DC / 2026-05-29 用戶 explicit directive 入憲)

任何 stock 之任一 feature 為 **imputed / zero-fill / null-strategy 補值**(即 `feature_values.is_null_imputed=True`)→ **該 stock 必須從 core_universe 排除**。Imputed value 沒有 (a)(b)(c) source-traceable origin,**等同 §一.10 #1 之 AI 幻像值**(第四類禁止來源),**不可進入 model training universe**。

**1. Imputed / 任何 AI 自補值 = AI 幻像值之等價判定**

per §一.10 「全部產出數據必須出自以下三類來源」:(a) 程式輸出 / (b) DB query / (c) API response。**任何不屬於此三類之 value 即為第四類 AI 幻像值之違規**。

- ❌ **Imputed zero-fill**:無 FinMind/FRED API source(計算 NaN 後系統補 0)→ **AI 幻像值**
- ❌ **Median-fill / mean-fill**:統計值非 API origin → **AI 幻像值**
- ❌ **Forward-fill / backward-fill**:source 仍是來自其他時間點的真實數據,但 **as_of point 之 source 仍為缺失** → **AI 幻像值**
- ❌ **AI-generated / system-derived(any)** ⭐:任何由系統或 AI **自己補上 / 算出 / 推測** 之 value 而非由 source API value 經 mathematical transformation 得到 → **AI 幻像值**
- ❌ **「為了完整度」之合理推估**:即使數字看起來合理,只要不是來自 (a)(b)(c) 真實 source → **AI 幻像值**
- ✅ **唯一允許**:**真實 FinMind / FRED API source 之 raw value 經 mathematical transformation(log/ratio/z-score/winsorize/rolling-window-agg/etc.)而得之 numeric value** 才為 source-pure

⭐ **核心 doctrine**:**「imputed 值或是你自己補的值,就是這種没有 finmind api 與 fred api 來源的值」**(per 2026-05-29 用戶 explicit directive 之入憲原文)→ **此種值之 stock 必須排除核心股之外**(per 用戶 explicit directive)。理由:**這樣的特徵值跑模型是没意義的**(用戶原文)。

**第二次 strict re-check directive(2026-05-29)補強三類 AI 幻像值揭露**:

per 用戶 explicit re-check directive「再重新檢查 database 內所有個股的值...請寫入憲章」之 strict 補強:

- ❌ **Hardcoded knowledge** ⭐ NEW:任何 system / AI / human 在 code 中 hardcode 之 lookup table / dict / classification rule(e.g., `THEME_KEYWORDS = {"半導體": 100, ...}` in `feature_store_builder.py` L203)→ 由此 derived 之 feature value 為 AI 幻像 → 整支 feature 應從 SPEC 移除或重新設計
- ❌ **Silent fallback** ⭐ NEW:builder 中 `or 0` / `ELSE 0` / `COALESCE(_, 0)` 之 silent NULL replacement 而未設 `is_null_imputed=True` flag(已 audit 9 處位置:L345/L364/L377/L391/L420-421/L442/L496/L845)→ NULL silent zero-fill 為 AI 幻像值
- ❌ **Derived / Proxy 表 chain** ⭐ NEW:任何 derived table(e.g., `kwave_supply_cycle_proxy`)若被 production feature 引用 → 整條 source chain 須 trace 至原始 API(若鏈中含 hardcoded knowledge → 整條 chain 違 §一.10)
- ✅ **唯一允許**:**直接從 FinMind / FRED API raw value 經 mathematical transformation(log/ratio/z-score/winsorize/rolling-window-agg/etc.)而得之 numeric value**;**無 hardcoded knowledge 注入**;**無 silent fallback**

**Real example findings(2026-05-29 v0.2 audit)**:
- ⚠️ `theme_strength`:scores from `THEME_KEYWORDS = {"半導體": 100, "生技": 95, ..., "汽車": 60}` hardcoded 14 個 entries → **AI 幻像 feature**(已移除 SPEC_43 v0.2)
- ⚠️ 9 處 silent fallback in `feature_store_builder.py`:除 `margin_ratio_60d` 有 `is_null_imputed=True` flag(已 corrective)外,其他 8 處皆無 flag → 須補 explicit NULL detection

**第三次 strictest re-check directive(2026-05-29)補強 — TRANSITIVELY-TAINTED + HARDCODED KEYWORD CHOICE**:

per 用戶 explicit re-check directive(2026-05-29 第三次):「**再以最嚴格的模式進行重新檢查 database 內所有個股的值與所有程式中給的值**」之 strictest 補強:

- ❌ **Transitively-Tainted Features**(v0.3 NEW):任一 feature 之計算 chain 中含 **直接或間接** hardcoded AI knowledge → 整支 feature 為 AI 幻像 feature
  - **Real example**:`fitness_signal_60d = (avg_value × (theme_strength+0.01) × (foreign_ratio+0.01))^(1/3)` 使用 `theme_strength`(hardcoded scores)→ **整支 transitively tainted → 移除 SPEC v0.3**
- ❌ **Hardcoded Keyword Choice**(v0.3 NEW):即使 feature value 為 binary deterministic,只要 **「挑哪個 keyword」之選擇** 為 AI / human domain knowledge → 該 feature 為 AI 幻像 feature
  - **Real example**:`theme_is_semiconductor = 1 if industry contains '半導體' else 0` — 雖 binary 機械式,**「挑半導體不挑食品」為 AI domain knowledge**(非 FinMind API 賦予之 semantic)→ **移除 SPEC v0.3**

**SPEC 演進**:
- 原 SPEC_43(含 211 imputed + theme_strength + fitness_signal_60d + theme_is_semiconductor + barbell + right_tail_concentration)
- v0.17 SPEC_43 - 211 imputed stocks
- v0.18 SPEC_42 - theme_strength
- v0.19 SPEC_40 - fitness_signal_60d - theme_is_semiconductor
- **v0.20 SPEC_38 - barbell_balance_60d(hardcoded 0.80 Pareto)- right_tail_concentration_60d(hardcoded 10% decile)**(strictest-of-strictest)
- **§14.7-DC v0.15 SPEC_38 → SPEC_37 - amihud_illiquidity_60d**(⚠️ **dead-feature 軸,非 AI 幻像軸**:amihud 為 genuine FinMind source[`|return|/dollar-volume`]但 398 core DB-verified distinct=1 / 全 0.00000000 zero-variance dead feature;與上述 v0.18-v0.20 之 hallucination removal **正交**;見下方 v0.15 subsection + T_DC-28)

**Empirical 反論(corrective bundle III + IV)**:
- Bundle III v0.19:LightGBM ded +0.64 / CatBoost ded +0.39 / RF Win 100%
- Bundle IV v0.20:XGBoost ded 4.55(全 v0.20 最高)/ LGBM v0.2 +0.47 / ET +0.39 / RF +0.26
- **doctrine purity → 更乾淨 signal,不損 production 表現**

**v0.15 amihud_illiquidity_60d Dead-Feature Removal(2026-05-30 用戶授權「移除 amihud(最乾淨):承認在這個核心股 universe 下它是死特徵 → SPEC 38 → 37」+ 兩 AskUserQuestion 確認[scope「全系統一致」/ DB「刪除 amihud rows」]入憲 — SPEC_38 → SPEC_37 + T_DC-28 Dead-Feature Removal Treaty)**:

> ⚠️ **非 strictest re-check 循環 + 非 hallucination 移除**(per §一.8 honest labeling):amihud 為 genuine FinMind-source feature(`|return|/dollar-volume` mathematical transformation),**非 §一.10 第四類 AI 幻像**;移除原因為 **zero-variance dead feature**(398 core DB-verified distinct=1),與前 5 doctrine-removed AI 幻像 feature(theme_strength 等)**為正交兩軸**(source-purity vs discrimination)。

**Finding(DB-verified,isolated psql READ-ONLY,against v0_4 panel `fs_20260528_feature_set_v0_4` × committed core membership;v0_5 amihud rows 已 DELETE 故以同源 v0_4 重現)**:committed 398 core 之 amihud = distinct=1 / min=max=0.00000000 / exact-zero 398/398(100%);non-core overall max 0.00000013(~1.3e-7)。**Root cause**:Amihud = |return|/dollar-volume,core 選股偏高流動性(LiquidityMass 30% CoreScore)→ dollar-volume 極大 → amihud sub-1e-8 → `feature_values.feature_value numeric(24,8)` 截斷為 0。分類 = AP-2 Constant Global Broadcast / T_DC-21 最極端 case(distinct=1),用戶選 **removal 分支**(非 rescale)。

**全系統一致 removal(用戶授權 scope「全系統一致」— 28 code files,grep-exhaustive per T_DC-26)**:

| 群 | 修正 |
|---|---|
| SSOT `feature_store_builder.py` | 移除 FEATURE_DEFINITIONS entry + `_amihud_illiquidity()` staticmethod + assignment line |
| 10 trainers + 10 validators | 各 `SPEC_43` list 移除 amihud → 37 entries |
| 5 audit files | 移除 amihud + FEATURE_RANGES / LITERATURE_REF / LITERATURE_SIGN entry + group label (5)→(4)|
| `core_universe_builder.py`(`SPEC_38_FEATURES` completeness gate,**functionally required**)+ `apply_feature_completeness_gate.py`(`SPEC_37_FEATURES`,cosmetic)| 移除 amihud |

grep:全 `scripts/` 0 amihud 殘留,唯一 residual = `scripts/pipeline/feature_engineering.py:1041-1042` legacy `amihud_illiquidity`(無 `_60d` / rolling(20)/ 不在 v0_5 / 不同 feature → 正確保留 per §一.3)。28 files py_compile pass。

**DB removal(用戶授權「刪除 amihud rows(最乾淨)」— atomic guarded transaction LANDED)**:DELETE 95840 rows(v0.5-family 96 panel = 95 production + 1 diag_test,`feature_set_version LIKE 'feature_set_v0.5%'`)+ UPDATE 96 panel feature_count 38→37;v0_4 legacy 100436 rows **untouched**(歷史)。post-verify:v0.5-family amihud=**0** / feature_count=**37** / spot panel `fs_20260415_feature_set_v0_5` distinct features=**37**。

**deferred naming debt(per §一.3)**:`SPEC_43`(20 files,實持 37)/ `SPEC_38_FEATURES`(`core_universe_builder.py`,實持 37)未改名;`SPEC_37_FEATURES`(`apply_feature_completeness_gate.py`,DEPRECATED)移除 amihud 後實持 **36** ≠ 名稱 37(名稱從不 match content:移除前之 37 = **19 真實 v0_5 feature + 18 非 v0_5 entry**[amihud + 14 個 kwave_*/mc_*/ms_* macros(僅存 legacy fs_v0_4)+ 3 個已移除 doctrine feature:right_tail_concentration_60d / barbell_balance_60d / fitness_signal_60d],且**缺 18 個真實 v0_5 feature** → 該 37 非 v0_5 之 37;DB-verified vs fs_20260415_feature_set_v0_5);5 audit files 仍列已移除 doctrine features — 皆 out of scope 留待後續。

**未授權項(honest disclosure §一.8)**:❌ 既有 model artifacts **未 retrain**(retrain on SPEC_37 為獨立授權項,觸發 §一.12 + §二.6,不 auto-run);❌ **無 git commit**(全 28 code edits + charter/CLAUDE.md inscription 尚未 commit,per §二.2 需 explicit 授權)。

**T_DC-28(Dead-Feature Removal Treaty)— v0.15 NEW**:任一 feature 在**全部 committed-core stocks** 下 DB-verified zero-variance(`COUNT(DISTINCT value)=1`)→ dead feature(zero discrimination)→ 必須移除(或若 precision-truncation 遮蔽真實 signal 則 rescale-and-rebuild)。為 AP-2 / T_DC-21 最極端 case(distinct=1)。**關鍵 distinction vs §一.10 / T_DC-7**:dead feature 之 source-purity test **PASS**(genuine source)但 discrimination test **FAIL**(zero variance)→ 與 hallucination removal **正交**;選 removal 時 evidence = **DB-verified distinct=1 across committed-core**(非僅 all-stocks 高 exact-zero ratio,該為較弱 AP-1 signal)。amihud 為 first concrete instance。

**雙層治權鎖更新(v0.15)**:主憲章 §14.7-DC v0.15 + CLAUDE.md §一.13 v0.15 同次入憲(per T_DC-6)。T_DC accumulation:T_DC-1 ~ T_DC-28(v0.15 +T_DC-28 Dead-Feature Removal)。

**證據基礎(本條入憲)**:用戶 2026-05-30 explicit「**移除 amihud(最乾淨):承認在這個核心股 universe 下它是死特徵 → SPEC 38 → 37**」+ 兩 AskUserQuestion 確認(「全系統一致(建議)」/「刪除 amihud rows(最乾淨)」)之直接入憲。code grep(28 files / 0 殘留除 legacy / py_compile pass);DB guarded-transaction(DELETE 95840 / UPDATE 96 / v0.5-family amihud=0 / feature_count 37 / v0_4 保留 100436);dead-feature psql READ-ONLY(398 core distinct=1 / 全 0.00000000 / non-core max 1.3e-7)。**precedent**:首個 **dead-feature(非 hallucination)removal** 入憲 case;確立 source-purity 軸(§一.10 / T_DC-7)與 discrimination 軸(AP-2 / T_DC-21 / T_DC-28)正交;codify T_DC-28。

---

**v0.14 Trainer-Feature-Set-Binding Corrective(2026-05-30 用戶授權 Task C「修正 code + 入憲」入憲 — 10 production trainers 之 `DEFAULT_PANELS` 原預設 legacy `feature_set_v0_4` → repoint recent-8 v0_5 + `is_null_imputed` 防禦過濾 + snapshot/membership explicit bind `feature_set_v0_5` + T_DC-27)**:

> ⚠️ **本條非 strictest re-check 循環**(無第十四次 re-check directive);為用戶 explicit 授權之 corrective action(Task C「修正 code + 入憲」)。per §一.8 honest labeling 與 re-check cycle 區分。

**Finding — universe-membership 乾淨 ≠ feature-value-source 乾淨**:§14.7-DC 治權契約 3 原宣稱 trainers「inherent 已對齊 §14.7-DC」為 **incomplete** —— trainers query universe SQL 繼承乾淨股票清單(398 core),但各 trainer `DEFAULT_PANELS` 原 hardcode 8 個 `feature_set_v0_4` panel(43-feature legacy,含 5 已移除 AI 幻像 feature + 3 imputed core 1532/1533/4953 之 margin_ratio_60d)→ 即使股票清單乾淨,經 v0_4 載入仍注入 AI 幻像 / imputed feature values → §14.7-DC + §一.10 之 latent 注入路徑。**membership(哪些 stock)與 feature-value-source(哪一版 feature values)為兩個正交 gate**。

**治權效果(用戶 2026-05-30 explicit 框定「移除 AI 幻像 feature + imputed core 注入源」)**:10 trainers 經 v0_4→v0_5 repoint + `is_null_imputed IS NOT TRUE` 防禦,**自訓練資料路徑切斷兩類注入源** —— (1) 5 個 AI 幻像 feature(theme_strength / theme_is_semiconductor / fitness_signal_60d / barbell_balance_60d / right_tail_concentration_60d,v0_5 已 ABSENT);(2) imputed-core 注入源(v0_4 `margin_ratio_60d` imputed 涉 3 core stock 1532/1533/4953;v0_5 imputed-among-core = 0)。

> ⚠️ **scope 釐清(per §一.10 + §一.6)**:此為 trainer **feature-value-source gate** 切斷,**不更動 universe membership**(哪些 stock 入核心由 §14.7-DC hard gate + `core_universe_builder` 決定)。imputed-among-core DB-verified = **3(v0_4)→ 0(v0_5)**,**非 10**;若 inscribe「特定 N 支 stock 不入核心」之 membership 變更,須附 source-traceable stock 清單 + `core_universe_builder` 重 build,不在本 v0.14 scope。

**P-A Corrective(全 10 trainer,grep + psql 驗證 LANDED)**:

| 修正 | 內容 | 驗證 |
|---|---|---|
| `DEFAULT_PANELS` repoint | 8 × v0_4 → recent-8 v0_5(fs_20250915 ~ fs_20260415)| 10/10:0 v0_4 / 8 v0_5 each |
| `is_null_imputed` 防禦過濾 | 載入 query 補 `AND is_null_imputed IS NOT TRUE` | 10/10:1 filter each |
| snapshot/membership binding | NULL → `feature_set_v0_5` | snapshot 1 row + membership 1001 rows(398+603)全 bound / 0 NULL |

- **P-A 選法(per §一.3 最小邊界)**:保留 trainers 既有「8-panel walk-forward」語意;DB 驗證 8 panel 全部 398 core / 38 features / **imputed-among-core = 0**。
- **deferred cleanup debt(per §一.3)**:`SPEC_43` 變數名(實持 38 features)+「43 canonical」comment 未改名;zero-fill `get(f, 0.0)` 未動(對齊 clean validators)。
- **未授權項(honest disclosure §一.8)**:❌ heavy retrain 未執行(10 model × panel = 獨立授權項,觸發 §一.12 + §二.6,不 auto-run);v0.14 僅修正 default 綁定 + 載入防禦 + snapshot binding,不回溯 retrain;charter v0.22 metrics re-inscription 仍 BLOCKED by retrain。

**T_DC-27(Trainer-Feature-Set-Binding Treaty)— v0.14 NEW**:production trainers(及任何載入 feature values 之 universe-aware 下游模組)feature-set 必須預設 doctrine-target `feature_set_v0_5`(不得 legacy v0_4);載入 query 必須補 `is_null_imputed IS NOT TRUE`;committed snapshot 必須 explicit bind `feature_set_version`(不留 NULL)。核心 insight:membership 繼承 ≠ feature-value-source 綁定,雙 gate must-both-hold。違反 = §14.7-DC + §一.10。

**雙層治權鎖更新(v0.14)**:主憲章 §14.7-DC v0.14 + CLAUDE.md §一.13 v0.14 同次入憲(per T_DC-6)。T_DC accumulation:T_DC-1 ~ T_DC-27。

**證據基礎(本條入憲)**:用戶 2026-05-30 explicit Task C 授權「**改 10 支 trainer 預設改綁 v0_5、補 is_null_imputed 防禦過濾、snapshot 綁 feature_set_version,再把 gap + corrective 入憲 §14.7-DC**」之直接入憲。Code grep 驗證(10/10 trainer:0 v0_4 / 8 v0_5 / 1 filter / py_compile pass);DB binding psql READ-ONLY 驗證(snapshot + 1001 membership bound feature_set_v0_5 / 0 NULL);8 v0_5 panel source-purity psql 驗證(398 core / 38 features / 0 imputed each)。**precedent**:v0.14 揭露治權契約 3「universe-membership 繼承乾淨 → inherent 已對齊」宣稱 incomplete;T_DC-27 codify membership 與 feature-value-source 雙 gate must-both-hold。

---

**第十三次 strictest re-check directive(2026-05-30)— AP-5 FOURTH instance(§0.3.3 L2279 + §0.3.9 L2457-2525 numbered-區 doctrine mandate — v0.11/v0.12 anchor-navigated sweep 遺漏)→ doctrine-region inline marker + banner LANDED + T_DC-26 Grep-Exhaustive-Not-Anchor-Nav Treaty + v0_5 38-feature / 0-imputed-core DB 四度再確認 + code 0.80/0.20 operational-value Tier 3 透明揭露**:

per 用戶 explicit 第十三次 directive「**再一次以最嚴格的模式進行重新檢查 database 內所有個股的值與所有程式中給的值又或者在憲章要求系統給特定的值...八二法則是一種思想...不是給特定的值...康波週期就給 40 或 60 這樣的值...是一種循環思想,不是給特定的值...imputed 值或是你自己補的值,就是這種没有 finmind api 與 fred api 來源的值,而有這些值的個股就要排除在核心股之外,因為這樣的特徵值跑模型是没意義的。請寫入憲章。**」之 strictest 入憲。本次為 v0.12 → v0.13 incremental 三 Phase 三方獨立驗證(Code grep + 本人 DB query via isolated venv READ-ONLY + Charter **grep-traverse 全文** re-sweep)。

### 三 Phase v0.13 verdict(2026-05-30 DB+code+charter 三方驗證)

| Phase | v0.13 verdict | vs v0.12 |
|---|---|---|
| **1. Code** | **0 active Tier 4-5 feature violation**(全 production scripts + 9 multi_cycle validators 維持 CLEAN;THEME_KEYWORDS / 5 doctrine-removed feature 引用皆為 removal comment;`audit_doctrine_compliance.py` L366-383 actively enforce `^THEME_KEYWORDS\s*=\s*\{` 不存在 → PASS);**NEW 透明揭露**:全文 grep `0.80`/`0.20` hardcoded values 逐一分類確認**全屬 Tier 3 operational/portfolio 層,非 feature value**(`config.py:62` target_core/agg_ratio 0.80/0.20 + `config.py:809` PARETO_RATIO 0.2 + `portfolio_sizer.py:22-23` barbell sleeve caps + `core_universe_builder.py:1471` foreign-remain CoreScore 分桶)→ 不在 `feature_values`、不入模型 feature → **非 §14.7-DC violation**;cleanup debt 同 v0.11/v0.12 | ⬆️ 維持 CLEAN + NEW 0.80/0.20 Tier 3 classification |
| **2. DB** | committed snapshot 仍 policy v0.18(398 core / convex 0 / quarantine 603 / total 2803 / feature_set_version=NULL / created 2026-05-29 23:08);**v0_5 = 38 features / imputed-among-398-core = 0 / 5 doctrine-removed 全 ABSENT**(四度再確認);macro(18)仍僅 fs_v0_4 / v0_5=0;AP-2(distinct≤5)=0;AP-1 top:zero_volume_ratio_252d 89.9% / amihud 80.8% / trust_net_20d 61.8% / trust_net_60d 56.1% / margin_ratio_60d 20.6% | ⬆️ 四度再確認(invariants fresh,無 drift)|
| **3. Charter** | **AP-5 FOURTH instance**:v0.11 T_DC-24 + v0.12「exhaustive sweep」實為 **anchor-navigation**(僅掃 §0.3-A〜§0.3-E lettered + §6.3/§6.4),未 grep-traverse 全文 → 遺漏 numbered 區 **§0.3.3 L2279**(§6/§8 對映 blockquote 列 THEME_KEYWORDS 半導體 100/生技 95/... 為 active 治理實踐)+ **§0.3.9 L2457-2525**(整節 THEME_KEYWORDS 字典維護框架 + hardcoded 權重 [30,100])→ **本 cycle inline marker + head banner LANDED** | ❌ **AP-5 FOURTH instance**(anchor-navigated gap)→ marker 本 cycle LANDED + T_DC-26 入憲 |

### AP-5 FOURTH instance + T_DC-26(Grep-Exhaustive-Not-Anchor-Nav Treaty)

v0.10 first(§0.3-A/C/D/E prose 缺 banner);v0.11 second(§6.3 六層 + cells named-only);v0.12 third(cells L2555/L2590 inline marker landed);**v0.13 fourth**:grep-driven 全文 sweep 揭露前序所有「exhaustive」實為 **anchor-navigation**(僅導航已知主節點 §0.3-A〜§0.3-E + §6.3/§6.4,從未對全文 grep `THEME_KEYWORDS`/`ThemeResonance`)→ 遺漏 §0.3.1-0.3.10 **numbered 區**(排版位於 lettered 區之前、敘事與主 locus 遙遠)之 **§0.3.3 L2279**(列 hardcoded score 45-100 為 active)+ **§0.3.9 L2457-2525**(整節 THEME_KEYWORDS 字典維護治權,類比 §0.3-D 但無 banner)。

**T_DC-26(Grep-Exhaustive-Not-Anchor-Nav Treaty)**:T_DC-24 之 exhaustive sweep 必須以**全文 grep-traverse** 執行,不得僅 anchor-navigate 至已知主節點;**boundary 釐清(防 v0.14 re-litigate)**:concept-level layer-placement 提及(`ThemeResonance`/`MBNRIC` 作為 universe 層載體名稱,無 hardcoded score / 無 15% 權重宣稱 / 不治理 dict)為 K-wave 循環思想之合憲表述**不需 marker**;僅 (i) 列 hardcoded score 45-100、(ii) 宣稱 15% CoreScore 權重為現行、(iii) 治理 THEME_KEYWORDS dict 為 active 三者需 marker。為 T_DC-24 之 search-methodology enforcement。

### Code 0.80 / 0.20 operational-value Tier 3 透明揭露(用戶第十三次「系統內給的特定值如八二法則」直接回應)

全文 grep `0.80`/`0.20` 經逐一 5-tier 分類,確認**全屬 Tier 3 operational/portfolio/universe-governance 層,非 feature value**,故 §0.2-A 概念層 barbell / §9.2 sizing 已 sanction,**不違反 §14.7-DC**(該 doctrine 限 `feature_values` 個股特徵值):

| 位置 | 值 | 性質 | 裁決 |
|---|---|---|---|
| `config.py:62` target_core/agg_ratio | 0.80 / 0.20 | §9.2 portfolio core/agg sleeve 配比 | Tier 3 operational,非 feature → ✅ |
| `config.py:809` PARETO_RATIO | 0.2 | feature-selection meta(保留前 20% 特徵) | Tier 3 meta,非 feature → ✅ |
| `portfolio_sizer.py:22-23` attack/safety caps | 0.20 / 0.80 | §0.2 槓鈴 sleeve caps | Tier 3 risk-mgmt(§0.2-A sanction)→ ✅ |
| `portfolio_strategy.py:54` safety_ratio | 0.85 / 0.80 | regime-conditional 防禦配比 | Tier 3 risk-mgmt → ✅ |
| `core_universe_builder.py:1471` foreign-remain 分桶 | 0.80/0.50/0.20 | CoreScore sub-score 評分閾值 | Tier 3 universe-gov,非 feature → ✅ |

**關鍵 distinction**:已移除之 Tier 4 violation(`barbell_balance_60d` = abs(top20%vol − 0.80))將 0.80 注入 **FEATURE 層**(進 `feature_values` → 入模型)→ AI 幻像 → v0.20 移除;上表 0.80/0.20 為**配置/風控/選股層 operational params**(不進 `feature_values`、不入模型 feature)→ 同 cost_per_rebal=0.006 屬 Tier 3 disclosed,合憲。此即 v0.5 「Pareto 思想層允許 vs 特定值入 feature 層禁止」doctrine 之 operational-layer 對應。

### DB-verified facts(本條全數字 source = isolated venv psql READ-ONLY query per §一.10 (b))

- committed snapshot:policy v0.18 / 398 core / convex 0 / quarantine 603 / total 2803 / feature_set_version=NULL / created 2026-05-29 23:08
- v0_5:38 distinct features(SPEC_38)/ imputed-among-398-core = 0 ✅(四度再確認)/ 5 doctrine-removed 全 ABSENT
- macro features(kwave_* / mc_* / ms_*):18 個僅存於 legacy `fs_v0_4`;v0_5 = 0 rows
- AP-2(distinct ≤ 5):0 features in v0_5 ✅
- AP-1 exact-zero top(> T_DC-20 30% audit 對象):zero_volume_ratio_252d 89.9% / amihud 80.8% / trust_net_20d 61.8% / trust_net_60d 56.1% / margin_ratio_60d 20.6%
- v0_4 legacy:仍含 3 imputed core(1532/1533/4953 margin_ratio_60d)— corrective 維持 pending(snapshot 應 explicit bind feature_set_version='feature_set_v0_5')

### Pareto / K-wave / Imputed doctrine re-affirmation(用戶第十三次 emphasis — 與前 12 次一致)

- **Pareto(八二法則)**:思想層 ✅(inspire portfolio barbell / universe governance);特定值層分兩類 —— 入 **feature 層**(barbell_balance_60d 之 0.80)= Tier 4 AI 幻像(已 v0.20 移除);入 **operational/portfolio 層**(config/portfolio_sizer 之 0.80/0.20)= Tier 3 disclosed 合憲(§0.2-A sanction)
- **K-wave(康波週期)**:循環思想層 ✅;特定值層 ❌(40/60 年週期長度 + THEME_KEYWORDS keyword score 45-100 = Tier 4-5 AI 幻像)— ThemeResonance + THEME_KEYWORDS 已 §6.4-DC v0.12 移除;kwave_*(FRED-derived)無 hardcoded 40/60 且不在 v0_5
- **Imputed / self-filled**:無 FinMind / FRED API source 之值 = AI 幻像(§一.10 第四類)→ 該 stock 強制排除 core_universe(v0_5 已 enforced 0 imputed core,DB 四度確認)

### 雙層治權鎖更新(v0.13)

主憲章 §14.7-DC v0.13 + CLAUDE.md §一.13 v0.13 同次入憲(雙層治權鎖 v0.13 layer)。T_DC accumulation:T_DC-1 ~ T_DC-26(v0.13 +T_DC-26 Grep-Exhaustive-Not-Anchor-Nav)。AP accumulation:AP-1 / AP-2(RESOLVED ✅ in v0_5)/ AP-3 / AP-4 / AP-5(v0.13 FOURTH instance §0.3.3 L2279 + §0.3.9 numbered 區 — 本 cycle marker + banner LANDED)。

**證據基礎(本條入憲)**:用戶 2026-05-30 第十三次 explicit「**再一次以最嚴格的模式進行重新檢查...請寫入憲章**」之直接入憲。Phase 1 code grep(全 production scripts + 9 validators 0 active Tier 4-5 feature violation;`audit_doctrine_compliance.py` enforce THEME_KEYWORDS 不存在;NEW 0.80/0.20 operational-value Tier 3 逐一分類);Phase 2 DB SQL via isolated venv READ-ONLY(committed v0.18/398 + v0_5 38 features / imputed-among-core=0 四度確認 / 5 doctrine-removed absent / macro 僅 fs_v0_4 / AP-2=0 / AP-1 zero_volume 89.9% + amihud 80.8%);Phase 3 charter **grep-traverse 全文**(揭露 v0.11/v0.12 為 anchor-navigation → §0.3.3 L2279 + §0.3.9 L2457-2525 numbered 區 doctrine mandate 遺漏 → inline marker + banner LANDED + T_DC-26 入憲)。**precedent**:v0.13 為 v0.11 T_DC-24 / v0.12「exhaustive」claim 之 search-methodology 驗證 —— 揭露 anchor-navigation ≠ grep-traverse;確立 sweep 必須全文 grep + concept-vs-mandate boundary(防 v0.14 re-litigate concept 提及)。違反者:任何 future mandate-removal sweep 若以 anchor-navigation 替代全文 grep-traverse 即違反 T_DC-26。

---

**第十二次 strictest re-check directive(2026-05-30)— AP-5 THIRD instance(§0.3-A/C 表格 cell L2555/L2590 inline marker — v0.11 T_DC-24 named-but-not-landed)→ inline cell marker LANDED + T_DC-25 Inline-Cell-Annotation Treaty + v0_5 38-feature / 0-imputed-core / macro-absent DB 三度再確認**:

per 用戶 explicit 第十二次 directive「**再一次以最嚴格的模式進行重新檢查 database 內所有個股的值與所有程式中給的值又或者在憲章要求系統給特定的值...八二法則是一種思想...不是給特定的值...康波週期就給 40 或 60 這樣的值...是一種循環思想,不是給特定的值...imputed 值或是你自己補的值,就是這種没有 finmind api 與 fred api 來源的值,而有這些值的個股就要排除在核心股之外,因為這樣的特徵值跑模型是没意義的。請寫入憲章。**」之 strictest 入憲。本次為 v0.11 → v0.12 incremental 三 Phase 三方獨立驗證(Code grep + 本人 DB query via isolated venv READ-ONLY + Charter exhaustive re-sweep)。

### 三 Phase v0.12 verdict(2026-05-30 DB+code+charter 三方驗證)

| Phase | v0.12 verdict | vs v0.11 |
|---|---|---|
| **1. Code** | **0 active Tier 4-5 violation**(全 production scripts + 9 multi_cycle validators 維持 CLEAN);所有 doctrine-removed feature 引用皆為 removal comment;`kwave_*` / `mc_*` / `ms_*` 為 FRED API-derived source-pure(`kwave_phase_indicator = sum(valid_components)/len` equal-weight = Tier 0,**無 hardcoded 40 / 60 年週期長度**);cleanup debt 維持(`core_universe_builder.py:2212` stale 六層 print + `feature_store_builder.py:208` stale 註解 + theme_score no-op L1706/L1850/L1863)| ⬆️ 維持 CLEAN(無新 violation)|
| **2. DB** | committed snapshot 仍 policy v0.18(398 core / feature_set_version=NULL / created 2026-05-29 23:08);**v0_5 = 38 features / imputed-among-core = 0 / 5 doctrine-removed 全 ABSENT**(三度再確認);**NEW: macro features(kwave/mc/ms — 18 個)僅存於 legacy `fs_v0_4`(1583 stocks / 28494 rows),v0_5 完全 ABSENT(0 rows)** | ⬆️ 三度再確認 + NEW macro-absent-in-v0_5 |
| **3. Charter** | **AP-5 THIRD instance**:v0.11 T_DC-24 treaty text 已 NAME 表格 cell L2555/L2590 為遺漏點,但 v0.11 僅 land §6.3 + §0.3 上方 banner,**未 inline annotate cell 本身** → out-of-context / grep table-row 仍讀為 active mandate → **本 cycle inline cell marker LANDED**(⚠️ SUPERSEDED + strikethrough per §一.11)| ❌ **AP-5 THIRD instance**(named-but-not-landed)→ inline marker LANDED + T_DC-25 入憲 |

### AP-5 THIRD instance + T_DC-25(Inline-Cell-Annotation Treaty)

v0.11 inscribe T_DC-24(exhaustive sweep)且 treaty text 已 **explicitly 列出表格 cell L2555/L2590** 為遺漏點,然 v0.11 實際僅 land §6.3 SUPERSEDED banner + §0.3-A/C 表格上方 banner blockquote,**未對 cell 本身 inline annotate** — banner 不隨 table row 移動,故 grep table-row / out-of-context 閱讀 cell 時仍讀為 active「✅ 允許且強制」/「✅ K-wave 唯一有效投影 channel」mandate。此為 **AP-5(Stale-Mandate Drift)之 THIRD concrete instance**(v0.10 first = §0.3-A/C/D/E prose 全缺 banner;v0.11 second = §6.3 banner landed but cells named-only;v0.12 third = cells inline marker LANDED)。本 cycle 已對 §0.3-A L2555 + §0.3-C L2590 cell 置入 inline ⚠️ SUPERSEDED + strikethrough(歷史文字保留 per §一.11)。

**T_DC-25(Inline-Cell-Annotation Treaty)**:當 superseded mandate 以**表格 cell** 形式存在時,SUPERSEDED marker 必須 **inline 置於該 cell 內**(strikethrough + ⚠️ marker),不得僅依賴表格上方之 banner blockquote;因 table row 可被 grep / 獨立 out-of-context 閱讀而 banner 不隨 row 移動,故 **banner-above-table ≠ inline-cell-annotation**。為 T_DC-24 之 granularity enforcement。

### DB-verified facts(本條全數字 source = isolated venv psql READ-ONLY query per §一.10 (b))

- committed snapshot:policy v0.18 / 398 core / convex 0 / feature_set_version=NULL / created 2026-05-29 23:08
- v0_5:38 distinct features(SPEC_38)/ imputed-among-398-core = 0 ✅ / 5 doctrine-removed(theme_strength / theme_is_semiconductor / fitness_signal_60d / barbell_balance_60d / right_tail_concentration_60d)全 ABSENT
- macro features(kwave_* / mc_* / ms_*):18 個僅存於 legacy `fs_v0_4`(1583 stocks / 28494 rows);v0_5 = 0 rows(完全 ABSENT)→ v0_5 為純 38-feature 集合,即使 source-pure 之 macro features 亦不在其內
- v0_4 legacy:仍含 3 imputed core(1532/1533/4953 margin_ratio_60d)— snapshot 若不 explicit bind `feature_set_version='feature_set_v0_5'` 則 model train on v0_4 仍注入 imputed(corrective 維持 pending per v0.11 Step E caveat)

### Pareto / K-wave / Imputed doctrine re-affirmation(用戶第十二次 emphasis — 與前 11 次一致)

- **Pareto(八二法則)**:思想層 ✅;特定值層 ❌(0.80 / 0.20 / 10% decile hardcode = Tier 4 AI 幻像)— barbell_balance_60d + right_tail_concentration_60d 已 v0.20 移除 + v0_5 ABSENT(DB 三度確認)
- **K-wave(康波週期)**:循環思想層 ✅;特定值層 ❌(40 / 60 年週期長度 hardcode = Tier 4 AI 幻像)— ThemeResonance + THEME_KEYWORDS 已 §6.4-DC v0.12 移除;`kwave_*` features(FRED-derived)無 hardcoded 40/60 且不在 v0_5
- **Imputed / self-filled**:無 FinMind / FRED API source 之值 = AI 幻像(§一.10 第四類)→ 該 stock 強制排除 core_universe(v0_5 已 enforced 0 imputed core,DB 三度確認)

### 雙層治權鎖更新(v0.12)

主憲章 §14.7-DC v0.12 + CLAUDE.md §一.13 v0.12 同次入憲(雙層治權鎖 v0.12 layer)。T_DC accumulation:T_DC-1 ~ T_DC-25(v0.12 +T_DC-25 Inline-Cell-Annotation)。AP accumulation:AP-1 / AP-2(RESOLVED ✅ in v0_5)/ AP-3 / AP-4 / AP-5(v0.12 THIRD instance §0.3-A/C 表格 cell — 本 cycle inline marker LANDED)。

**證據基礎(本條入憲)**:用戶 2026-05-30 第十二次 explicit「**再一次以最嚴格的模式進行重新檢查...請寫入憲章**」之直接入憲。Phase 1 code grep(全 production scripts + 9 validators 0 active Tier 4-5;kwave_* FRED-derived equal-weight Tier 0;cleanup debt 同 v0.11);Phase 2 DB SQL via isolated venv READ-ONLY(committed v0.18/398 + v0_5 38 features / imputed-among-core=0 / 5 doctrine-removed absent / macro 18 features 僅 fs_v0_4 1583 stocks 28494 rows / v0_5 macro=0);Phase 3 charter exhaustive re-sweep(發現 v0.11 T_DC-24 named L2555/L2590 但未 inline land → 本 cycle inline cell marker LANDED + T_DC-25 入憲)。**precedent**:v0.12 為 v0.11 T_DC-24 自身之 completeness 驗證 — 揭露 v0.11 named-but-not-landed gap → 確立 banner-above-table ≠ inline-cell-annotation 之 T_DC-25 granularity 原則。違反者:任何 future table-cell mandate removal 若僅 land banner 而未 inline annotate cell 即違反 T_DC-25。

---

**第十一次 strictest re-check directive(2026-05-30)— NEW AP-5 residual instance(§6.3 六層 CoreScore stale mandate)+ T_DC-24 Exhaustive-Mandate-Removal-Sweep Treaty + v0_5 0-imputed-core DB 再確認 + 全 production scripts + 9 validators code CLEAN**:

per 用戶 explicit 第十一次 directive「**再一次以最嚴格的模式進行重新檢查 database 內所有個股的值與所有程式中給的值又或者在憲章要求系統給特定的值...八二法則是一種思想...不是給特定的值...康波週期就給 40 或 60 這樣的值...是一種循環思想,不是給特定的值...imputed 值或是你自己補的值,就是這種没有 finmind api 與 fred api 來源的值,而有這些值的個股就要排除在核心股之外,因為這樣的特徵值跑模型是没意義的。請寫入憲章。**」之 strictest 入憲。本次為 v0.10 → v0.11 incremental 三 Phase 三方獨立驗證(Code grep + Charter read + 本人 DB query via isolated venv READ-ONLY)。

### 三 Phase v0.11 verdict(2026-05-30 DB+code+charter 三方驗證)

| Phase | v0.11 verdict | vs v0.10 |
|---|---|---|
| **1. Code** | **0 active Tier 4-5 violation**(全 production scripts + 9 multi_cycle validators CLEAN);NEW finding = `core_universe_builder.py:2212` stale print echo 舊六層 formula(`CoreScore v0.2 = 0.25*DQ + 0.25*LM + 0.20*FG + 0.15*TR + 0.10*IF + 0.05*VC - RP`)+ theme_score no-op kill-switch(L1706 theme_score=0.0 / L1850/L1863 `>= 70.0` filter → convex_pool 永遠 empty)→ **cleanup debt(非 doctrine violation)** | ⬆️ 維持 CLEAN(v0.10 已確認 code-side LANDED);v0.11 新增 9-validator + 全 production scripts sweep 確認 |
| **2. DB** | committed snapshot 仍 policy v0.18(398 core / convex 0 / quarantine 603 / total 2803 / feature_set_version=NULL / created 2026-05-29 23:08);**v0_5(SPEC_38)= 38 features exact / 5 doctrine-removed features 全 ABSENT / AP-2(distinct≤5)= 0 features / imputed among 398 core under v0_5 = 0** ✅;v0_4 legacy 仍 3 imputed core(1532/1533/4953 margin_ratio_60d)+ 4 AP-2;AP-1 exact-zero in v0_5:amihud 76.7% / zero_volume_ratio_252d 84.8%(NEW)/ trust_net 53-63% | ⬆️ 再確認 v0_5 0-imputed-core;NEW AP-1 zero_volume_ratio_252d 84.8% 揭露 |
| **3. Charter** | §0.3-A/C/D/E SUPERSEDED banners 已 landed(v0.10)✅;**NEW AP-5 residual instance**:§6.3 L3217 六層 CoreScore(含 `15% ThemeResonance`)無 banner + 表格 cell L2555/L2590 — v0.10 sweep 僅 annotate §0.3-A/C/D/E prose 而遺漏 §6.3 + 表格 cell → **residual Stale-Mandate Drift**;K-wave / Pareto 全 concept-level(PASS)| ❌ **NEW AP-5 residual 揭露**(v0.10 sweep 非 exhaustive)→ §6.3 SUPERSEDED banner 本 cycle LANDED(L3218)+ T_DC-24 入憲 |

### NEW AP-5 residual instance(§6.3 六層 CoreScore)+ §6.3 banner LANDED(本 cycle)

v0.10 揭露 AP-5(Stale-Mandate Drift)並 annotate §0.3-A/C/D/E 四處 SUPERSEDED banner;**v0.11 exhaustive sweep 發現 v0.10 遺漏 §6.3 L3217(六層 CoreScore 公式含 `15% ThemeResonance`)+ §0.3-A 表格 cell L2555 + §0.3-D 表格 cell L2590** — 為 AP-5 之 **SECOND concrete instance(residual)**。本 cycle 已 land §6.3 SUPERSEDED banner(主憲章 L3218)per T_DC-23 + T_DC-24:標示六層為 v0.2 歷史口徑,現行五層 CoreScore = **30% DataQuality + 30% LiquidityMass + 20% FundamentalGravity + 15% InstitutionalFlow + 5% VolatilityControl − RiskPenalty**(`core_universe_builder.py` v0.12 L1620 為實作 SSOT;§6.4-DC v0.12 L3318 為 charter SSOT)。

### T_DC-24(Exhaustive-Mandate-Removal-Sweep Treaty)— v0.11 NEW

**T_DC-24**:任何 charter-mandated 機制之 removal annotation(per T_DC-23)必須為 **exhaustive sweep** — 對全憲章 grep 該機制(關鍵字 / 公式 / 表格 cell / 歷史 narrative)之全部出現點逐一 annotate SUPERSEDED,不得僅 annotate 高頻引用節點而遺漏低頻 / 表格 / 歷史段;遺漏任一即為 AP-5 residual Stale-Mandate Drift。為 T_DC-23 之 completeness enforcement(T_DC-23 要求 same-cycle annotate;T_DC-24 要求 annotate 必須涵蓋全部出現點)。

### DB-verified facts(本條全數字 source = isolated venv psql READ-ONLY query per §一.10 (b))

- committed snapshot:policy v0.18 / 398 core / 0 convex / 603 quarantine / 2803 total / feature_set_version=NULL / created 2026-05-29 23:08
- v0_5 feature_set:38 distinct features(SPEC_38);5 doctrine-removed(theme_strength / theme_is_semiconductor / fitness_signal_60d / barbell_balance_60d / right_tail_concentration_60d)全 ABSENT
- v0_5 AP-2(distinct value ≤ 5):0 features ✅(v0_4 legacy 仍 4)
- imputed among 398 core under v0_5:**0** ✅(§14.7-DC gate 對 v0_5 滿足)
- imputed among all stocks under v0_5:604 stocks(非 core → 已 quarantine,合 doctrine)
- v0_4 legacy:3 imputed core(1532/1533/4953 margin_ratio_60d)— 任何 model train on v0_4 仍注入 imputed(§14.7-DC violation;corrective:snapshot 應 explicit bind feature_set_version='feature_set_v0_5')
- AP-1 exact-zero in v0_5(> T_DC-20 30% → 須 audit legitimate-0 vs silent-fallback):amihud 76.7% / zero_volume_ratio_252d 84.8%(NEW)/ trust_net_20d ~63% / trust_net_60d ~53%

### Pareto / K-wave doctrine re-affirmation(用戶第十一次 emphasis)

- **Pareto(八二法則)**:思想層 ✅(right-tail concentration 原則 inspire universe governance);特定值層 ❌(0.80 / 0.20 / 10% decile hardcode = Tier 4 AI 幻像)— v0.20 已移除 barbell_balance_60d + right_tail_concentration_60d
- **K-wave(康波週期)**:循環思想層 ✅;特定值層 ❌(40 / 60 年週期長度 hardcode = Tier 4 AI 幻像)— ThemeResonance + THEME_KEYWORDS(K-wave CoreScore 投影載體)已 §6.4-DC v0.12 移除
- **Imputed / self-filled**:無 FinMind / FRED API source 之值 = AI 幻像(§一.10 第四類)→ 該 stock 強制排除 core_universe(v0_5 已 enforced 0 imputed core)

### 雙層治權鎖更新(v0.11)

主憲章 §14.7-DC v0.11 + CLAUDE.md §一.13 v0.11 同次入憲(雙層治權鎖 v0.11 layer)。T_DC accumulation:T_DC-1 ~ T_DC-24(v0.11 +T_DC-24)。AP accumulation:AP-1/AP-2(RESOLVED ✅ in v0_5)/AP-3/AP-4/AP-5(v0.11 SECOND instance §6.3 residual — 本 cycle banner LANDED)。

**證據基礎(本條入憲)**:用戶 2026-05-30 第十一次 explicit「**再一次以最嚴格的模式進行重新檢查...請寫入憲章**」之直接入憲。Phase 1 code grep(全 production scripts + 9 multi_cycle validators 0 active Tier 4-5 violation;`core_universe_builder.py:2212` stale print + theme_score no-op kill-switch L1706/L1850/L1863 為 cleanup debt);Phase 2 DB SQL via isolated venv READ-ONLY(committed v0.18/398 + v0_5 38 features / 5 doctrine-removed absent / AP-2=0 / imputed-among-core=0 / imputed-all=604 / AP-1 amihud 76.7% + zero_volume_ratio_252d 84.8%);Phase 3 charter read(§0.3-A/C/D/E banners landed + NEW §6.3 L3217 六層 CoreScore residual AP-5 + 表格 cell L2555/L2590)。**precedent**:v0.11 為 v0.10 AP-5 揭露後之 exhaustive completeness sweep — 發現 v0.10 sweep 非 exhaustive(遺漏 §6.3 + 表格 cell)→ §6.3 SUPERSEDED banner 本 cycle LANDED + T_DC-24 入憲。違反者:任何 future mandate removal 若僅 annotate 部分出現點即違反 T_DC-24。

---

**第十次 strictest re-check directive(2026-05-30)— Charter-Code Contradiction 揭露(AP-5 Stale-Mandate Drift)+ Code Phase Step C 補 LANDED 確認 + DB 398 source-pure under v0_5 驗證 + T_DC-23 Charter-Mandate-Removal Sync Treaty**:

per 用戶 explicit 第十次 directive「**再一次以最嚴格的模式進行重新檢查 database 內所有個股的值與所有程式中給的值又或者在憲章要求系統給特定的值...八二法則是一種思想...康波週期...不是給特定的值...imputed 值或是你自己補的值...有這些值的個股就要排除在核心股之外...請寫入憲章。**」之 strictest 入憲。三 Phase 並行 audit(Code agent + Charter agent + 本人 DB query 三方獨立驗證):

### 三 Phase v0.10 verdict(2026-05-30 DB+code+charter 三方驗證)

| Phase | v0.10 verdict | vs v0.9 |
|---|---|---|
| **1. Code** | `core_universe_builder.py` **NOW CLEAN**(TOOL_VER v0.12;THEME_KEYWORDS L196-208 已移除為註解 / `_theme_resonance_score()` L1332-1333 已移除 / CoreScore 五層 DQ 0.30+LM 0.30+FG 0.20+IF 0.15+VC 0.05);`feature_store_builder.py` 仍 CLEAN(5 features + THEME_KEYWORDS 移除)| ⬆️ **UPGRADE**:v0.9 標記 core_universe side STILL VIOLATING → v0.10 確認 Step C code-side LANDED |
| **2. DB** | committed `core_universe_20260529_..._policy_v0_18_source_pure_panhistorical_gate`:**398 core / convex 0 / quarantine 603 / total 2803 / feature_set_version=NULL / created 2026-05-29 23:08**;imputed among 398 core = **ONLY margin_ratio_60d / ONLY in superseded v0_4 / ONLY 3 stocks(1532/1533/4953)**;**v0_5(SPEC_38)= 0 imputed among core** ✅ | v0.9 universe 未 commit(v0.17/910 plan)→ v0.10 actual committed v0.18/398(stricter)|
| **3. Charter** | **NEW VIOLATION**(推翻先前「charter prose clean」結論):§0.3-A L2553/L2564 + §0.3-C L2586 + §0.3-D L2598/L2601 + §0.3-E L2613 仍 MANDATE ThemeResonance 15% + THEME_KEYWORDS 為「✅ 允許且強制」/「唯一參數化載體」,而 code v0.12 已移除 → **Charter-Code 自相矛盾** | ❌ **REGRESSION 揭露**:prior iterations 認定 charter prose clean(violations 只在 code/DB);v0.10 證實 charter 自身 stale mandate |

### AP-5(Stale-Mandate Drift Anti-Pattern)— v0.10 NEW

**AP-5**:**code 已移除某 charter-mandated 機制,但 charter 上游 doctrine 條文未同步更新,仍 mandate 該機制 → charter 自相矛盾**。為 **AP-3 之逆向**(AP-3 = charter inscribe 領先 code;AP-5 = code 領先 charter,charter 留 stale mandate)。

- **具體 instance**:`core_universe_builder.py` v0.12 移除 ThemeResonance(theme_score=0.0 no-op)+ THEME_KEYWORDS;但 charter §0.3-A L2553(TR 15% 權重「允許且強制」)/ §0.3-D L2598(THEME_KEYWORDS「唯一參數化載體」)/ §0.3-E L2613(15% 權重 fallback 10%)未更新 → future machine 若 from charter 重建 code 將 re-introduce THEME_KEYWORDS Tier 5 violation
- **風險**:charter 為 SSOT;stale mandate 比 code violation 更危險(charter 驅動 code,非反向)

### T_DC-23(Charter-Mandate-Removal Sync Treaty)— v0.10 NEW

**T_DC-23**:**當 code 移除某 charter-mandated 機制時,SAME inscription cycle 必須同步 annotate 上游 charter mandate 為 SUPERSEDED**;不得單側 lag。對齊 T_DC-19 Charter-Code-DB Triple Sync(三層同步)之 removal-direction enforcement。

### Code 結構 residual(非 doctrine violation,cleanup debt)

- `core_universe_builder.py`:`theme_score` field(L219)+ `theme_score=0.0`(L1706)+ `theme_score >= 70.0` convex filter(L1850/L1863)+ sort key(L1839)仍物理存在 → 中性化為 no-op,但 **convex_pool 永遠 empty**(committed snapshot convex_count=0 confirm)→ convex tier 當前無 member
- `feature_store_builder.py:208`:stale FALSE 註解(宣稱「core_universe_builder.py THEME_KEYWORDS 仍存(v0.22 follow-up)」)— v0.12 已移除,註解已過時

### v0.21 plan progress matrix update(per T_DC-22 step-by-step disclosure)

| Step | Action | v0.10 status(vs v0.9)|
|---|---|---|
| **A** | Commit v0.17 build SQL(imputed-exclusion filter)| ✅ LANDED(code,unchanged)|
| **B** | feature_store_builder 移除 5 features + THEME_KEYWORDS | ✅ LANDED(unchanged)|
| **C** | core_universe_builder 移除 THEME_KEYWORDS + FG audit | ✅ **FULLY LANDED**(code-side TOOL_VER v0.12;**C-charter sub-step LANDED 2026-05-30** — §0.3-A/C/D/E SUPERSEDED banners per T_DC-23,用戶 explicit 授權)|
| **D** | Rebuild fs_v0_5(38 features)| ✅ LANDED(0 imputed among 398 core)|
| **E** | Rebuild core_universe | ⬆️ **NOW LANDED at v0.18/398**(v0.9 NOT-landed/v0.17-910 plan → actual committed v0.18/398,stricter);⚠️ caveat:feature_set_version=NULL(binding gap)+ convex_count=0 |
| **F** | 補 amihud / trust_net explicit NULL detection | ❌ NOT-LANDED(institutional `or 0`/`ELSE 0` L405-422 仍在,SPEC zero_fill by design 但 present-NULL 未 flag)|
| **G** | Retrain 全 models on v0_5 + v0.18 | 🚧 **PARTIALLY**:foundation models(Chronos/TFT/iTransformer/PatchTST)on v0.18/398 in progress;tree models 待 verify retrained on v0.18/398 |
| **H** | Re-inscribe charter v0.22 with DB-verified metrics | 🚧 BLOCKED by G |

**進度估計**:~58-62% complete(A+B+C-code+C-charter+D+E landed;F NOT;G partial;H blocked)

### 3-stock imputed 之 honest interpretation(per §14.7-DC core doctrine)

DB query 證實 committed 398 core 中,**3 stocks(1532/1533/4953)之 margin_ratio_60d 為 imputed,但僅在 superseded feature_set v0_4**;在 doctrine-target v0_5(SPEC_38)中**0 imputed**。裁決:
- ✅ **398 core 在 v0_5(doctrine-aligned set)下 100% source-pure** → §14.7-DC gate 對 v0_5 滿足
- ⚠️ **snapshot feature_set_version=NULL** → 未正式 bind 至 v0_5;且 v0_4(43-feature legacy set)仍含 3 imputed core stocks → **任何 model 若 train on v0_4 將注入 imputed margin_ratio_60d**(§14.7-DC violation)
- **corrective**:snapshot 應 explicit set `feature_set_version='feature_set_v0_5'`;tree models 須確認 train on v0_5 而非 v0_4

### 雙層治權鎖更新(v0.10)

主憲章 §14.7-DC v0.10 + CLAUDE.md §一.13 v0.10 同次入憲(雙層治權鎖 v0.10 layer)。T_DC accumulation:T_DC-1 ~ T_DC-23(v0.10 +T_DC-23)。AP accumulation:AP-1/AP-2(RESOLVED ✅)/AP-3/AP-4/AP-5(v0.10 NEW)。

**證據基礎(本條入憲)**:用戶 2026-05-30 第十次 explicit「**再一次以最嚴格的模式進行重新檢查...請寫入憲章**」之直接入憲。三 Phase 三方驗證證據鏈:Phase 1 code grep(`core_universe_builder.py` TOOL_VER v0.12 + L196-208/L1332-1333 removal comments + L1706 theme_score=0.0 + L1850/L1863 `>= 70.0` no-op);Phase 2 DB SQL(committed snapshot 398 core/convex 0/feature_set_version NULL + imputed among core = 3 stocks 1532/1533/4953 margin_ratio_60d ONLY v0_4 / v0_5 = 0);Phase 3 charter read(§0.3-A L2553「允許且強制」TR 15% + §0.3-D L2598「唯一參數化載體」+ §0.3-E L2613 15%→10% mandate 仍 active)。**precedent**:v0.10 為首次揭露 charter 自身 stale mandate(AP-5),推翻 prior iterations「charter prose clean」結論;ThemeResonance code 移除(v0.12)與 charter mandate(§0.3-A/D/E)desync 為 T_DC-23 first concrete instance。違反者:任何 future from-charter code rebuild 將 re-introduce THEME_KEYWORDS Tier 5 violation。

---

**第九次 strictest-of-strictest re-check directive(2026-05-29)— Pareto/K-wave/Imputed 三 Doctrine Re-affirmation + v0.21 Plan Partial-Land Disclosure + AP-4 Aspirational-Reality Drift + T_DC-22 Step-by-Step Land Disclosure Treaty**:

per 用戶 explicit 第九次 directive「**再一次以最嚴格的模式進行重新檢查 database 內所有個股的值與所有程式中給的值又或者在憲章要求系統給特定的值,又或著在此專案系統內給的特定值如八二法則...等(因為八二法則是一種思想,是指大部份的結果來自少部份的資源,不是給特定的值),又或著康波週期就給 40 或 60 這樣的值(因為康波週期是一種循環思想,不是給特定的值),是否有之前說過不可有 AI 幻像值嗎?imputed 值或是你自己補的值,就是這種没有 finmind api 與 fred api 來源的值,而有這些值的個股就要排除在核心股之外,因為這樣的特徵值跑模型是没意義的。請寫入憲章。**」之 strictest-of-strictest 入憲。

### 第九次 user explicit 3 doctrine clarifications

| Doctrine | 用戶 explicit clarification | 治權治理 |
|---|---|---|
| **A. Pareto / 八二法則** | 「思想層,大部份結果來自少部份資源,**不是給特定的值**」| hardcode 0.80 / 0.20 / 10% / 20% 為 feature 參數 → AI 幻像 → 移除 |
| **B. 康波週期 / K-wave** | 「循環思想,**不是給 40 或 60 這樣的值**」| hardcode 40-year / 60-year cycle length → AI 幻像 → 移除 |
| **C. Imputed / 你自己補的值** | 「**没有 finmind api 與 fred api 來源的值,有這些值的個股要排除在核心股之外,跑模型是没意義的**」| 任一 feature `is_null_imputed=TRUE` → 該 stock 強制從 core_universe 排除(v0.2 hard gate reaffirmation)|

### v0.8 → v0.9 三 Phase 並行 audit 增量發現

| Phase | v0.9 verdict(2026-05-29 audit) |
|---|---|
| **1. Code** | `feature_store_builder.py` **CLEANED**(THEME_KEYWORDS / Pareto 0.80 / 5 features 移除;TOOL_VER v0.6 標示);`core_universe_builder.py` **STILL VIOLATING**(THEME_KEYWORDS 30 entries L197 + `_theme_resonance_score()` L1360-1367 + L1634 仍 active called from CoreScore)|
| **2. DB** | **`fs_20190815_feature_set_v0_5`(38 features × 1002 stocks)COMMITTED at 2026-05-29 20:11** ✅(全 5 v0.18-v0.20 doctrine removal landed)+ AP-2 RESOLVED ✅(0 features distinct_vals ≤ 5);**236 stocks 仍 has is_null_imputed=TRUE in fs_v0.5**(margin_ratio_60d 236 / foreign_net 27×2 / trust_net 27×2 / eps 3 / ni_pos 3);AP-1 部分改善(amihud 87% → 73.8% 仍違反 T_DC-20 30% threshold)|
| **3. Charter** | Charter STILL clean ✅(no new Tier 4-5 mandates);Pareto §0.2 / K-wave §0.3 doctrine 已 fully aligned 用戶第九次 emphasis;v0.8 aspirational marker placement adequate |

### v0.21 plan progress matrix(per T_DC-22 step-by-step disclosure)

| Step | Action | v0.9 status |
|---|---|---|
| **A** | Commit v0.17 build SQL(imputed-exclusion filter) | ✅ **LANDED**(code)— `core_universe_builder.py` v0.11 之 `_apply_source_pure_filter()` 已加 + auto-called for all builds + `build_historical_panels.py` CLI argparse 支援 Step D rebuild |
| **B** | feature_store_builder.py 移除 5 Tier 4-5 features + THEME_KEYWORDS | ✅ **LANDED** |
| **C** | core_universe_builder.py 移除 THEME_KEYWORDS + audit FG thresholds | ⚠️ **PARTIALLY landed**(feature_store side done / core_universe side not)|
| **D** | Rebuild fs_v0_5(38 features × 1002 stocks) | ✅ **LANDED** |
| **E** | Rebuild core_universe v0.17(910 + 236 quarantine) | ❌ NOT landed(ready-to-execute) |
| **F** | 補 amihud / trust_net explicit NULL detection | ❌ NOT landed |
| **G** | Retrain 全 9-10 models on v0.5 + v0.17 | 🚧 **BLOCKED by E** |
| **H** | Re-inscribe charter v0.22 with DB-verified metrics | 🚧 **BLOCKED by E+G** |

**進度估計**:~37-42% complete(A code + B 完整 + D 完整 + C 部分 landed);剩餘 ~5-6 hr workflow(待用戶 explicit authorize)

### AP-4 + T_DC-22(v0.9 NEW)

**AP-4(Aspirational-Reality Drift Anti-Pattern)**:charter inscription 之 corrective plan 部分執行 + 部分未執行而無 step-by-step disclosure → governance gap(extending AP-3)

**T_DC-22(Step-by-Step Land Disclosure Treaty)**:任何 corrective plan(≥ 3 steps)後續 audit 入憲時 **必須 step-by-step 個別 disclose** completion status,**禁止 binary summary**;status taxonomy: ✅ LANDED / ⚠️ PARTIALLY-LANDED / ❌ NOT-LANDED / 🚧 BLOCKED-by-X

### Pareto / K-wave 用戶 emphasis 與 charter 對齊驗證

- **Pareto**:思想層(right-tail concentration principle)允許 inspire universe governance / right_tail_returns_skew_252d 等 Tier 0-2 features;特定值層(0.80 / 0.20 / alpha=1.16)禁止 hardcode → 對齊 charter §0.2 L1916-1920 + §0.2-A 7 禁令
- **K-wave**:循環思想層允許 inspire ThemeResonance L1 universe weight;特定值層(45-60 年週期 / 2026 共振時點)禁止 hardcode → 對齊 charter §0.3-A L2560 禁令 #2 + §0.3.6 SWRD「Hypothesis non Law」
- **注意**:核心股 score 之 ThemeResonance(15%)仍依賴 `core_universe_builder.py` THEME_KEYWORDS 30-entry dict(Step C PARTIALLY-LANDED)→ K-wave/Pareto 思想層 charter clean,**TR sub-score code path 仍 Tier 5 violation**,須 Step C 完成 land 後 TR 才 source-pure

### v0.7 marker incremental update

v0.8 第 10 條 retroactive marker 仍 valid 但 **partial-land 補充**:SPEC_38 feature_store 層已 land(fs_v0.5 / 38 features / 1002 stocks);universe 層仍 aspirational(910 / 211 quarantine 未 commit);v0.7 metrics 須 partial-land aware 直至 v0.22 inscription。

### 雙層治權鎖更新(v0.9)

主憲章 §14.7-DC v0.9 + CLAUDE.md §一.13 v0.9 同次入憲(雙層治權鎖 v0.9 layer)。T_DC accumulation:T_DC-1 ~ T_DC-22(v0.9 +T_DC-22)。AP accumulation:AP-1/AP-2(RESOLVED ✅)/AP-3/AP-4(v0.9 NEW)。

**證據基礎(本條入憲)**:用戶 2026-05-29 第九次 explicit「**再一次以最嚴格的模式進行重新檢查...請寫入憲章**」之直接入憲。三 Phase 增量 audit 證據鏈完整:Phase 1 grep evidence(feature_store_builder.py TOOL_VER v0.6 + L202 explicit comment;core_universe_builder.py L197/L1360/L1634 仍 active);Phase 2 DB SQL evidence(fs_v0.5 38×1002 committed at 2026-05-29 20:11 + 236 imputed stocks 仍 active members + AP-2 broadcast features distinct_vals > 5);Phase 3 charter scan(§0.2 / §0.3 doctrine 與用戶 doctrine 完全對齊;無 charter inscription correction needed)。**precedent**:v0.21 plan partial-land 狀態(B+D landed / C partially / E NOT / F NOT / G+H BLOCKED)→ AP-4 first concrete instance + T_DC-22 first concrete implementation。違反者:任何後續 audit 若以 binary summary disclose 即違反 T_DC-22。

---

**第八次 strictest-of-strictest re-check directive(2026-05-29)— Charter-Code-DB Triple Sync Treaty + 3 New Anti-Patterns + v0.7 Aspirational Inscription 之 honest retroactive 揭露**:

per 用戶 explicit 第八次 directive「**再一次以最嚴格的模式進行重新檢查 database 內所有個股的值與所有程式中給的值又或者在憲章要求系統給特定的值,又或著在此專案系統內給的特定值如八二法則...等...請寫入憲章**」之 strictest-of-strictest 入憲。三 Phase 並行 audit(Code / DB / Charter):

### 三 Phase audit summary

| Phase | Scope | Method | Verdict |
|---|---|---|---|
| 1 | 全 production scripts hardcoded value | Explore agent grep + tier classify | **Tier 4-5 violations 仍在 committed code**(THEME_KEYWORDS / Pareto 0.80 / Bianconi-Barabási 0.01+1/3 / VIX 30 / ROE 0.30 / FG sub-score 等)|
| 2 | DB 全 individual stock values | psql query feature_values / snapshots | **v0.7 SPEC_38 doctrine 未 land 在 DB**(committed: v0.16 / 43 features / 1002 stocks / 181 imputed)|
| 3 | 憲章自身 hardcoded value | Explore agent scan §6/§7/§8/§9/§14.7-X | **Charter itself clean,但 v0.7 inscription 與 Code+DB 脫節** |

### v0.7 charter claim vs 本機 committed reality(2026-05-29 DB query 證據)

| v0.7 charter claim | 本機 committed reality | 性質 |
|---|---|---|
| SPEC_38(38 features) | feature_set_v0.4(43 features)| code/DB 未 land |
| 910 stocks | 1002 stocks(policy v0.16)| code/DB 未 land |
| 211 imputed excluded(quarantine_count=211)| 181 imputed 仍在 active universe(quarantine_count=0)| enforcement 未執行 |
| 0 AI 幻像 | 5 supposedly-removed features 仍 alive in DB | doctrine 未 land |
| Sharpe 4.55 / Eff t 4.37(charter)| reproduce 結果 annual Sharpe 5.879 / Eff t 4.518(本機 9-tree v0.16 run)| prior-machine ad-hoc artifacts |

### 三 New Anti-Patterns(v0.8 NEW — orthogonal to 5-tier)

- **AP-1 Silent Zero-Fill Risk**:`amihud_illiquidity_60d` 87% exact-zero — illiquidity proxy 不應如此高 0,疑似 silent zero-fill 未設 `is_null_imputed=True`(同 `trust_net_20d` 59% / `trust_net_60d` 54.6%)
- **AP-2 Constant Global Broadcast**:`barbell_balance_60d` / `right_tail_concentration_60d` / `theme_strength` / `theme_is_semiconductor` 全 1002 stocks 為單一值(zero discrimination = noise injection)
- **AP-3 Aspirational Inscription**:charter inscribed 但 code / DB 未 land 之 case;v0.7 inscription 為此 pattern 之 first instance

### 三 New Treaties(v0.8 NEW)

- **T_DC-19 Charter-Code-DB Triple Sync Treaty**:任何 inscription 必須三層(Charter ∧ Code ∧ DB)同時驗證;缺一須明確標示為 aspirational(Charter-only / Code-aspirational / DB-aspirational)
- **T_DC-20 Silent Zero-Fill Detection Treaty**:exact-zero ratio > 30% 須執行 audit(legitimate 0 / silent fallback / 混合);silent fallback 須補 explicit `is_null_imputed=True` flag
- **T_DC-21 Constant Global Broadcast Anti-Pattern Treaty**:`COUNT(DISTINCT value) ≤ 5` 之 feature 為 zero-discriminatory,須 removal 或 fix-and-rebuild

### v0.7 inscription 之 honest retroactive disclosure(per T_DC-19)

v0.7 SPEC_38 / 910 / Sharpe 4.55 等 metrics 來自 **prior machine ad-hoc run**(v0.17 SQL 未 commit / v0.17 snapshot 未 commit / feature 移除未 commit);**本機 committed code+DB 不能 reproduce**(per `reports/9tree_multi_cycle_local_v016_vs_charter_v6_23_7_comparison_20260529.md` L25 明示「v0.17 build SQL 不在 committed code(只在前機 ad-hoc 跑)」)。

### v0.21 Corrective Action Plan(待用戶 explicit authorize — AI 不自行執行)

8 steps / ~10-12 hr workflow:(A) Commit v0.17 SQL → (B) 移除 5 Tier 4-5 features + THEME_KEYWORDS from feature_store_builder → (C) 同上 core_universe_builder + FG audit → (D) Rebuild fs_v0_5(38 × 910)→ (E) Rebuild policy_v0.17(910 stocks)→ (F) 補 amihud + trust_net explicit NULL detection → (G) Retrain 9-10 models → (H) Re-inscribe v0.21。⚠️ Steps D+F+G 觸發 §二.6 SHMM + §一.12 5-min report 強制。

### 雙層治權鎖更新

主憲章 §14.7-DC v0.8 + CLAUDE.md §一.13 v0.8 同次入憲(雙層治權鎖)。T_DC accumulation: T_DC-1 ~ T_DC-21。AP accumulation: AP-1/2/3。

**證據基礎(本條入憲)**:用戶 2026-05-29 第八次 explicit「**再一次以最嚴格的模式進行重新檢查 database 內所有個股的值與所有程式中給的值又或者在憲章要求系統給特定的值...請寫入憲章**」之直接入憲。三 Phase 並行 audit + DB SQL query + Code grep 證據鏈完整。**precedent**:v0.7 ratification 之 Sharpe 4.55 / 910 stocks 等 metrics 因本機 committed code 未含 v0.17 SQL 而不能 reproduce,違反 §一.10 #2 數據來源 traceability 之邊界 case;v0.21 corrective action plan 為 retroactive 修正路徑。

---

**第七次 final-convergence-ratification re-check directive(2026-05-29)— SPEC_38 = Strictest Convergence Point Ratification**:

per 用戶 explicit 第七次 directive(continued strictest re-check + "請寫入憲章")之 **convergence ratification**:

### 7 user directives → 6 corrective bundles → SPEC_38 convergence

| Iteration | Action | SPEC |
|---|---|---|
| 1 | Bundle I: 排除 211 imputed stocks → 910 universe | SPEC_43 |
| 2 | Bundle II: -theme_strength | SPEC_42 |
| 3 | Bundle III: -fitness_signal_60d -theme_is_semiconductor | SPEC_40 |
| 4 | Bundle IV: -barbell_balance_60d -right_tail_concentration_60d | **SPEC_38** ⭐ |
| 5 | v0.5 Concept vs Specific Value clarification | SPEC_38 unchanged |
| 6 | v0.6 5-tier Hardcoded Value Catalog | SPEC_38 unchanged |
| **7** | **v0.7 Final Convergence Ratification** | **SPEC_38 unchanged**(convergence point)|

### Per-Feature Tier 分類最終 audit

✅ **全 38 features 僅使用 Tier 0-2**:
- Tier 0(universal math)= 36 features use
- Tier 1(calendar)= 32 features use  
- Tier 2(statistical conventions)= 7 features use
- **Tier 4(concept-specific predicted values)= 0** ✅
- **Tier 5(hardcoded knowledge dicts)= 0** ✅

### 新 T_DC-18 Convergence Treaty

**T_DC-18**:**SPEC_38 v0.20 = strict source-pure convergence point**。任何未來 system 新增必須維持 SPEC_38 strictness baseline(Tier 0-2 only)。

### Convergence Practical Limit

進一步移除 features:
- 無 time windows → 無法計算 multi-day metrics
- 無 statistical conventions → 無法計算 statistical measures
- ⚠️ **系統將完全無法產生 features**

⭐ **SPEC_38 v0.20 = practical strictest convergence point**(不可再 reduce 而仍保持 functional system)

### v0.7 Final Convergence empirical confirmation

⭐ 910 × 38 = 34,580 rows / 100% complete / 0 imputed / 0 AI 幻像
⭐ XGBoost dedicated v0.20 8-panel Sharpe 4.55(全 model 最強)
⭐ XGBoost dedicated v0.20 annual Sharpe 5.64 / Eff t 4.37(過 T_CZ-6 4.20)
⭐ **strictest 與 production 表現完全不衝突**

**第六次 ultimate-final re-check directive(2026-05-29)— Comprehensive 5-tier Hardcoded Value Catalog + Transparency Disclosure**:

per 用戶 explicit「再一次以最嚴格的模式進行重新檢查...請寫入憲章」之 ultimate-final 入憲。系統性 catalog 全 hardcoded values:

### 5-Tier Hardcoded Value Hierarchy

| Tier | 類型 | 治權 | 例證 |
|---|---|---|---|
| **Tier 0** | Universal math constants | ✅ ALWAYS OK | π / e / log base / sqrt() / 0 (positive/negative natural split) |
| **Tier 1** | Calendar conventions | ✅ OK | 252d=year / 60d=quarter / 20d=month / 4q=year / sqrt(12) |
| **Tier 2** | Universal statistical conventions | ✅ OK | t=1.997 for p<0.05 / percentile / z-score / std formula |
| **Tier 3** | Empirical market parameters | ⚠️ **TRANSPARENT DISCLOSURE REQUIRED** | cost_per_rebal=0.006(broker fee)/ panel_spacing=30 |
| **Tier 4** | Concept-specific predicted values | ❌ **AI 幻像 REMOVE** | Pareto 0.80 / K-wave 40-60y / 10% decile |
| **Tier 5** | Hardcoded knowledge dictionaries | ❌ **AI 幻像 REMOVE** | THEME_KEYWORDS scores / sector maps |

### Charter-Mandated Tier 3 Transparent Disclosure 全 catalog

**Multi-cycle validators**:
- `cost_per_rebal = 0.006`(TW broker fee estimate / **empirical not from API**)
- `panel_spacing = 30`(calendar month convention)

**Treaty Gates(§14.7-CZ T_CZ-6)**:
- Eff t ≥ **4.20** / Sharpe ≥ **2.40** / Win ≥ **79%**(production criteria,**not feature data**)

**Model Hyperparameters(§14.7-CW T_CW-4)**:
- 200 / 5 / 0.05 / 5422 等(model config,not feature value)

**Deprecated Dead Code**:
- `feature_store_builder.py` L791 ms_stress=30.0 VIX threshold(per §14.7-CK 已 removed feature 之 dead code)

### SPEC_38 v0.20 final compliance verification

| Feature class | Used values | Tier | OK? |
|---|---|---|---|
| All 38 SPEC_38 features | Time windows + math + standard stats | **Tier 0-2 only** | ✅ |
| Already removed(v0.18-v0.20)| THEME_KEYWORDS / 0.80 / 10% etc. | Tier 4-5 | ❌ removed |

✅ **SPEC_38 v0.20 全 features 僅使用 Tier 0-2(universal math + calendar + statistical conventions)** — strictest valid + production-ready

**第五次 ultimate-strictest re-check directive(2026-05-29)— Concept vs Specific Value Critical Clarification**:

per 用戶 explicit directive「**八二法則是一種思想,是指大部份的結果來自少部份的資源,不是給特定的值;又或著康波週期就給40或60這樣的值(因為康波週期是一種循環思想,不是給特定的值)**」:

- ✅ **概念 / 思想 / Framework(可用)**:八二法則 / 康波週期 / Pareto theory / K-wave theory / preferential attachment theory — **這些 INSPIRE feature engineering 為 OK**
- ❌ **概念之特定預測值(AI 幻像)**:
  - Pareto's specific 0.80 (80/20 split predicted value) → AI 幻像
  - K-wave specific 40/60 year cycle length → AI 幻像
  - 10% decile / 5% threshold 等 hardcoded numeric thresholds → AI 幻像
  - THEME_KEYWORDS hardcoded scores(100/95/.../60)→ AI 幻像

**Concept vs Specific Value 判定指引**:

| 類型 | OK / NOT OK | 例證 |
|---|---|---|
| 時間 windows | ✅ OK | 20d / 60d / 252d / 4q / 8q / TTM(standard aggregation periods)|
| Universal math constants | ✅ OK | π / e / log base / 0 (positive vs negative split)|
| 比例 ratios | ✅ OK | pe_ratio / pb_ratio / yoy comparisons |
| 統計 normalization | ✅ OK | z-score / percentile rank / log transform |
| Standard statistical thresholds | ✅ OK | 0 for positive/negative split |
| **Concept-specific predicted values** | ❌ **AI 幻像** | **Pareto 0.80 / K-wave 40-60y / decile 10% / THEME_KEYWORDS scores** |

**v0.5 retroactive validation**:用戶 doctrine clarification **retroactively VALIDATE** 全 corrective bundle II-IV 移除決策正確
- ✅ theme_strength (v0.18) — uses hardcoded specific scores
- ✅ theme_is_semiconductor (v0.19) — uses hardcoded specific keyword choice
- ✅ fitness_signal_60d (v0.19) — transitively uses theme_strength
- ✅ barbell_balance_60d (v0.20) — uses Pareto's specific 0.80 value
- ✅ right_tail_concentration_60d (v0.20) — uses hardcoded 10% decile value

**SPEC_38 v0.20 已 fully aligned 用戶 doctrine clarification — NO 進一步 feature removal needed**

**第四次 strictest-of-strictest re-check directive(2026-05-29)— Charter-Mandated Hardcoded Threshold**:

per 用戶 explicit「**又或者在憲章要求系統給特定的值**」之揭露:

- ❌ **Charter-Mandated Hardcoded Threshold**(v0.4 NEW):任一 feature 公式中使用 charter / doctrine inscribed 之 specific reference constant(非 universal math const)→ AI 幻像 feature
  - **Real example**:
    - `barbell_balance_60d = abs((top 20% vol share) - 0.80)` — **0.80 為 Pareto 80/20 charter-mandated constant** → 移除 SPEC v0.4
    - `right_tail_concentration_60d`:**10% decile cutoff** charter-mandated → 移除 SPEC v0.4
- ✅ **kwave_supply_cycle_proxy 已 deprecated**:production 已用 real FRED API(`IPG3344S` 取代 `TW_SEMI_VWAP_YOY` / `PCU4831114831115` 取代 `TW_SHIPPING_VWAP_YOY`)

**2. 三層判定 strict gate**

任何 stock 進入 core_universe 之前須通過 strict triple-gate:

- **§14.7-CB Feature Completeness Gate**:全 43 features 必須有 row(no missing row)
- **§14.7-CJ Feature Reasonableness Gate**:value 必須合理範圍內(no extreme outlier / no NaN)
- **§14.7-DC Source-Pure Doctrine(本條)**:**任一 feature 若 `is_null_imputed=True` → stock 排除**(per §一.10 violation)

✅ 三 gate 皆通過 → core_tier='core_universe' / train_eligible=True
❌ §14.7-DC fail → core_tier='quarantine' / train_eligible=False / exclusion_reason 註明「§14.7-DC Source-Pure Doctrine: <feature_name> imputed (no FinMind API source)」

**3. 強制適用範圍**

- **core_universe_builder**:必須在 stock eligibility 評估階段強制 §14.7-DC 過濾
- **model_trainer 系列**(全 10 model + 未來新增):query universe SQL 已自動排除(只取 status='committed' + core_tier='core_universe' 之 stocks)
- **prediction_engine**:同上
- **portfolio_sizer**:同上
- **multi_cycle validators**:同上

**4. Imputed stocks 處理 protocol**

當發現 imputed stock(無論單一或多個 features):

1. **不可直接 zero-fill 通過**(此為 §一.10 violation)
2. **若 root cause = source API 真的缺資料**:該 stock 為 fundamental 不可用 → quarantine
3. **若 root cause = 計算邏輯 NaN(e.g., 0/0 in margin_ratio_60d)**:可以兩種處置:
   - **A.** 修正 feature 計算邏輯為 robust(e.g., 改用其他 raw signal as proxy)→ 新跑 feature_store_builder → 該 stock 可恢復
   - **B.** 直接 quarantine(per 本條 default)→ universe 縮小
4. **不可第三選項**:以 zero-fill 通過(此為 §14.7-DC violation)

**5. 違反治權**

- 訓練 / 預測 / backtest 任一階段使用含 imputed stock 之 universe → 結果 **deemed tainted**(per §一.10 violation)
- 須 **rebuild universe + retrain models + 更新所有 reports + 公開揭露 retroactive correction**
- 違反 + 不揭露 → 額外 §一.8 報告誠實 violation

**6. precedent 入憲案例(2026-05-29)**

本 session 9-tree CCF + Transformer dedicated **全部訓練於 1,121 stocks universe**(含 211 個 `margin_ratio_60d` zero-fill imputed)→ **全部結果 tainted**。Per §14.7-DC inscription:

- ❌ tainted models:LGBM v0.2 / LightGBM dedicated / XGBoost v0.1 既存 / XGBoost dedicated / CatBoost v0.1 既存 / CatBoost dedicated / Ensemble / Random Forest / Extra Trees / Transformer dedicated(共 10 model)
- ✅ corrective action:rebuild universe v0.17(910 stocks)+ retrain 全 10 model + retroactive doctrine inscription
- 治權 stamp:本 §14.7-DC 入憲 + 後續所有 model results 須 trace 至 v0.17 universe

**7. 雙層治權鎖(主憲章 + CLAUDE.md)**

本條 §一.13 + 主憲章 §14.7-DC 同次入憲,雙層治權鎖避免單點失效:

- **主憲章 §14.7-DC**:`reports/系統架構大憲章_v6.1.0.md`(L11953+)— 7 治權契約 + 5 新特性 + 6 證偽承諾(T_DC-1~6)+ corrective bundle precedent disclosure
- **CLAUDE.md §一.13(本條)**:AI 協作工具規則之 enforcement — 6 處置 protocol + 5 違反罰則

任一側更新但另側未對齊 → **§14.7-DC T_DC-6 治權結構性 violation**。

**證據基礎(本條入憲)**:用戶 2026-05-29 explicit directive(連續兩次):
1. **第一次**:「imputed 值就是這種没有 finmind api 與 fred api 來源的值,而有這些值的個股就要排除在核心股之外」之直接入憲
2. **第二次 strict re-emphasis**:「**imputed 值或是你自己補的值**,就是這種没有 finmind api 與 fred api 來源的值,而有這些值的個股就要排除在核心股之外,因為這樣的特徵值跑模型是没意義的。**請寫入憲章**」→ 加強對「AI 自己補的值」之 explicit ban + 強制 dual-layer 入憲

AI 之前錯誤合理化「zero-fill 為 doctrine-compliant treatment」為 §一.10 violation 之 retroactive 揭露。違反者:**本 session 全 10 model dedicated/baseline 均 retroactive tainted,須 rebuild on 910 universe**(已執行 corrective bundle)。

### 14. Tree-Family From-Zero → Model → Validation 12-PHASE 全序列治權範本(§14.7-DD / 2026-05-31 用戶 explicit directive 入憲)

任何 **tree-family(production 主軸)從零重建到模型驗證** 之 workflow,**必須**依主憲章 §14.7-DD 之 12-PHASE canonical sequence 執行。本條為 §14.7-DD 之 CLAUDE.md(AI 協作工具規則)雙層治權鎖伙伴(同次入憲);主憲章 §14.7-DD 為治權 SSOT,本條為 AI 工具 enforcement。

**1. 12-PHASE 序列(不可跳 phase / 不可改 order;每 PHASE audit gate 通過才進下一)**

| PHASE | 階段 | Script(摘要)|
|---|---|---|
| 0 | 環境建立 | OS deps + venv + .env + PostgreSQL role/db + import smoke test |
| 1 | Schema DDL | data_schema → initialize_market_data → core_universe_schema → feature_store_schema |
| 2 | Genesis 名冊 | sovereign_sync_engine.py --seed |
| 3 | Bootstrap 宇宙 | core_universe_builder.py bootstrap_init |
| 4 | 全市場全歷史 + FRED | sovereign_sync_engine.py --universe full --all + fetch_fred_data.py |
| 5 | Raw audit | audit_full_db_vs_api_reconcile.py(DB↔API 對帳)|
| 6 | 最終宇宙(real-data)| core_universe_builder.py bootstrap_final --commit |
| 7 | Feature Store(v0_5 / **37 features** / current+95)| feature_store_builder.py --commit + build_historical_panels.py |
| 8 | 宇宙加 gate 收緊(source-pure)| core_universe_builder.py --with-feature-gate --commit |
| 9 | Feature audit(訓練前)| audit_feature_ic_vs_future_return + sign_stability + necessity |
| 10 | 模型訓練 | **base model_trainer.py --commit(建表)→ 9 tree family trainers** |
| 11 | 模型驗證 | universe_completeness_schema.py → 9 tree-family multi_cycle validators |

> PHASE 12(週度 doctrine recommit)為維運層,**不在從零序列內**。

**2. Tree-family inventory(per §14.7-DD T_DD-4)**

- **9 trainers**:`model_trainer_{xgboost, xgboost_dedicated, lightgbm, lgbm_v2, catboost, catboost_dedicated, random_forest, extra_trees, ensemble}.py`
- **9 multi-cycle validators**:`multi_cycle_{validation, lightgbm, xgboost, xgboost_dedicated, catboost, catboost_dedicated, random_forest, extra_trees, ensemble}_validation.py`
- **不在範圍**:foundation/transformer(chronos / tft / itransformer / patchtst / transformer_dedicated)

**3. B1-B4 code-level blockers 強制 enforcement(per §14.7-DD + §一.8 誠實)**

執行 tree-family 從零重建前,AI **必須**確認以下 4 項(B1+B2 本 session 已 code-fix 但**未 commit**,future session 須先驗證 fix 仍在):

- **B1(PHASE 8)**:`core_universe_builder.py` feature gate 必為 `n >= len(self.SPEC_38_FEATURES)`(=37),**不可** hardcode `>= 38`(amihud 移除後 SPEC=37 之 self-introduced regression,已修)。若 gate 仍 `>= 38` → `--with-feature-gate` 會清空核心宇宙
- **B2(PHASE 9)**:5 個 feature audit 之 SPEC list 必為 **37**(set-match SSOT `feature_store_builder.py` FEATURE_DEFINITIONS),snapshot 須 dynamic 讀 latest committed(`pick_forward_window()`),**不可** hardcode `fs_v0_4`(已修)
- **B3(PHASE 10)**:`model_registry` / `model_training_run` 由 **base `model_trainer.py`** 建(DDL IF NOT EXISTS,L266/L352-353);9 family trainer 只 INSERT → **從零序列必先跑一次 base trainer --commit 建表**,再跑 9 family
- **B4(PHASE 11)**:multi-cycle validator 之 §14.7-CZ T_CZ-6 gate(Eff t≥4.20 / Sharpe≥2.40 / Win≥79%)**僅 docstring,非 code-enforced**;code 唯一硬判定 `abs(eff_t)>1.997`(p<0.05)→ 「PASS T_CZ-6」須**人工**對照印出 metrics 裁決,不可佯稱 code 自動 gate

**4. 雙軌獨立(per §14.7-DD T_DD-3)**

PHASE 10(trainer 產 artifacts + registry)與 PHASE 11(validator self-contained walk-forward,**不讀 `data/models/` 不讀 `model_registry`**)為獨立雙軌;validator 不需 PHASE 10 模型先完成;唯一共同前置 = PHASE 7 feature_values committed。

**5. Long-running 治權(PHASE 4 / 10 / 11)**

- PHASE 4 全市場同步(≥30 min)→ §二.6 SHMM 強制(N≥3 Monitor heartbeat + sentinel + watchdog + self-healing)
- PHASE 4/10/11(≥5 min)→ §一.12 5-min 回報強制(已完成階段 + elapsed + 剩餘估計 + 已知 metrics + warning)

**6. 資料真實性(per §一.10)**

時間預算全為 ESTIMATE(PHASE 4 全市場 ~6-10 hr 未實跑 profiling),非 deterministic fact;實跑以各程式 stdout `Total elapsed` 為準。任何 metrics 寫入須 trace 回 (a) 程式輸出 / (b) DB query / (c) API response。

**7. 雙層治權鎖(主憲章 + CLAUDE.md)**

本條 §一.14 + 主憲章 §14.7-DD 同次入憲;任一側更新另側未對齊 → §14.7-DC T_DC-6 結構性 violation。配套報告 `reports/tree_based_from_zero_build_runbook_20260531.md`(implementation reference)。

**證據基礎(本條入憲)**:用戶 2026-05-31 explicit directive「Tree-based(production 主軸)從零開始到模型產生及之後的模型驗証...請依目前的憲章來進行產生,如果有不足的需要先進行憲章的修訂後再修改此報告」+ AskUserQuestion Q1=「現在入憲 §14.7-DD」+ Q2=「修正 B1-B3」之直接入憲。B1+B2 已依 Q2 授權**僅改 code 修正**(py_compile PASS)**未 commit、未 DB write、未 retrain**(commit 仍另需 §二.2 授權)。

---

## 二、本專案編輯規則 (Project-Specific Edit Rules)

### 1. 編輯位置
- **所有編輯指向 `/home/hugo/project/stock_backend/`**。
- **不要寫入 `.claude/worktrees/` 內**（worktree 為鏡像目錄，會丟失）。
- 即使當前工作目錄在 worktree 中，仍應使用絕對路徑寫入主目錄。

### 2. Commit / Push 必須使用者明示授權
- 不自行 `git commit`。
- 不自行 `git push`。
- 使用者要求 commit 時，遵循憲章與一般 git 安全協議（不 `--amend` 已 push 提交、不 `--force` push 至主分支、不跳過 hooks）。

### 3. PR / 遠端互動
- 不自行建立 PR、不關閉 PR、不在 issue 留言。
- 任何 `gh` 指令影響遠端狀態的操作須先確認。

### 4. 研究報告寫入位置
- 逐程式研究報告、跨層分析、實證裁決一律寫入 `reports/`。
- 命名規範參照既有檔案（如 `<module>_<topic>_research_<YYYYMMDD>.md`）。

### 5. 程式變更前審視優先級
- 任何程式碼修改提案先對照憲章 **§0.0-E.6** 升版優先級（P0/P1/P2/P3）。
- 若不在優先級內，明文說明為何例外。

### 6. Long-running workflow 心跳可靠性 SHMM(憲章 v6.1.0 §14.7-AX)

任何 Claude session 跑 ≥ 30 min long-running workflow(如 from-zero rebuild、全市場 audit、~9-15h 工作流)時,**必須**建立 **Self-Healing Monitor Mesh (SHMM)**:

1. **N ≥ 3 個獨立 Monitor heartbeat**,週期非整數倍對齊(如 15/20/25/30 min 組合)
2. **Sentinel timestamp 檔案**(如 `/tmp/claude_loop_last_fire.txt`),任何 wake event 觸發 agent 時必須更新
3. **Watchdog Monitor**:每 60s 檢查 sentinel age,> 35 min 警報
4. **Self-healing protocol**:任何 wake event 觸發時必須執行 (a) 更新 sentinel (b) `ps` 驗證 SHMM 對應 shell 存活 (c) 重掛任何死亡 Monitor (d) 報告 workflow 狀態 (e) watchdog 警報則完整健康診斷
5. **`CronCreate` 不得作為**唯一觸發機制(REPL idle 才 fire,long-running workflow 中經常 miss);Monitor sleep loop 為主軸
6. **User manual ping 為最後底線**,不依賴用戶記憶觸發

### 7. 跨平台環境前置（憲章 v6.1.0 §0.0-I.9 / §0.0-I.10 / §14.7-AU）
- AI 協作工具於首次 setup 時須執行下列檢查:
  - **OS 原生依賴**(per §0.0-I.9):`xgboost` / `lightgbm` 需 OpenMP runtime;`psycopg2-binary` 在某些系統需 PostgreSQL client headers。
    - macOS: `brew install libomp` + `brew install postgresql@17`(可選)
    - Linux: `sudo apt-get install -y libgomp1 libpq-dev`
    - Windows: 通常無需(內含 vcomp140.dll)
  - **跨平台路徑**(per §0.0-I.10):`.env PROJECT_ROOT` 必須對齊本機物理路徑;macOS `/home/<user>` 為 `/Users/<user>` symlink,`path_setup.py v4.47+` 用 `os.path.realpath()` 解析後比對。
  - **Import smoke test**(必須通過才能進入後續執行):
    ```bash
    python -c "import psycopg2, pandas, polars, numpy, requests, sklearn, xgboost, lightgbm; print('✅ all imports OK')"
    ```
- 若 import smoke test 失敗,**先補 OS 層依賴**再執行任何 sync / audit 指令(避免在中途撞錯後 rollback DB)。
- 對應 `reports/full_market_sync_<時戳>.md` 之 Issue #1 / #2 修正建議。

---

## 三、憲章治權索引 (Constitutional Index)

本檔不重複憲章內容。需查詢治權規則時，依下表跳轉至憲章對應節：

| 工程動作類型 | 憲章節點 |
|---|---|
| 第一性原理隱喻禁令（不可硬寫 F=M×ΔlnP / 重力井 / SOC） | §0.1-A（6 條禁令）+ §0.1.1（T1/T2/T3 分層）|
| 八二法則治權邊界（20/60/20 / 槓鈴 / α 不得固定） | §0.2-A（7 條禁令）|
| 康波週期治權邊界（K-wave 不得進入 L2/L3） | §0.3-A（7 條禁令）|
| 統合層元規則（統合層不得下沉到工程公式） | §0.0-E.4（5 條禁令）|
| 三基柱跨層完整度基線 | §0.0-B / §0.0-C / §0.0-D |
| 升版優先級 P0/P1/P2/P3 | §0.0-E.6 |
| Universe SSOT（核心股治理唯一入口） | §6.7 |
| 三層防禦 | §7 |
| 五支落地鏈治權邊界（不得越權） | §8.4 / §0.0-A.1〜§0.0-A.5 |
| Anti-leakage（時間邊界） | §8.5 |
| 機率預測契約（30 日 horizon） | §9.1 |
| Portfolio sizing 契約 | §9.2 |
| 證偽承諾（共 21 項） | §0.1-E / §0.2-E / §0.3-E |

---

## 四、AI 協作工作流程 (Workflow)

### 1. 逐程式研究流程
1. 先用 `Read` / `Grep` 理解程式現況。
2. 對照憲章 §0.0-A.1〜§0.0-A.5 確認治權角色。
3. 對照憲章對應 §（如 §6.4 CoreScore、§8.5 anti-leakage）確認契約。
4. 對照 §0.0-B / §0.0-C / §0.0-D 確認跨層完整度位置。
5. 產出研究報告寫入 `reports/`。
6. 若需修訂入憲，依 §14.7-X 模式新增子節。
7. 程式碼修改提案對照 §0.0-E.6 優先級審查。

### 2. 計畫先行
- 多步驟或破壞性任務先寫計畫，使用者確認後執行。
- 計畫檔可存於 `.claude/plans/`。

### 3. 跨基柱影響評估
- 任何 §0.1 / §0.2 / §0.3 相關工程動作，須查 §0.0-B / §0.0-C / §0.0-D 升版觸發條件，確認是否影響完整度百分比。
- 跨基柱動作（如修改 portfolio_sizer 同時影響三件套）須在報告中明示。

### 4. 完整度評估必須先檢驗標頭 docstring

依使用者 2026-05-21 明示之治權原則：「**完整度包含所有程式開頭的註解說明，這很重要。**」

**8 項標頭強制檢驗項目**（任一缺即非 100% 合規）：

1. **主權狀態行**：含「(憲法 v6.0.0 對齊 + 最新治權修補摘要)」
2. **最後更新日期**：與 git history 一致
3. **最高原則**：固定為 `THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)`
4. **核心定義說明**：含治權慣例 5 條
   - 模組職責定義（如 [Path Sovereignty] / [API Contract First] / [Hub Sovereignty]）
   - **[Zero Hardcoded Verdict]**（§5.6.3 動態判定）
   - **[Sovereignty Declaration]**（§3.1/§3.2/§3.2A 治權位階 + 五套禁令不涉 + T1-T3 不分層 + §8.5 anti-leakage 不處理）
   - [Historical Reference Authority] 或類似歷史保留條
   - 模組特有條（如 [Hub Compatibility] / [Schema SSOT] / [Defensive Architecture]）
5. **全量維運指令總矩陣**：場景齊全 + 對齊模組版本
6. **全修訂歷程**：v1.0 → 最新版本完整保留（v5.4.x 為歷史記述）
7. **cross-reference 精確行號**：不可用 placeholder（如 `L24XX` / `第 N 行` / `待補`）
8. **cosmetic 字串對齊**：版本號 / 行號 / 治權引用須與最新狀態同步

**規則**：

- 進行「完整度 / 100% 合規」評估時，**必須先逐項檢驗標頭 docstring，再看程式邏輯**
- 任一標頭項缺失，即非 100% 合規（即使程式邏輯完美）
- 修訂歷程內之歷史記述（v5.4.x 等）保留為歷史記述，不更動（依憲章 §0.0-I.7 / L26）
- 對應 memory: `feedback_header_docstring_completeness.md`

**實證**（2026-05-21 內 9 次升版皆源自標頭缺口，0 項與程式邏輯本身有關）：

- `path_setup.py` v4.44 → v4.45 → v4.46
- `core/__init__.py` v1.14 → v1.15 → v1.16 → v1.17
- `data_schema.py` v2.11 → v2.12 → v2.13 → v2.14 → v2.15
- `audit_api_schema_compliance.py` v0.1 → v0.2

---

## 五、本檔升版規則

- 本檔半衰期約 **6-12 個月**（隨 Claude Code 工具演進而更新）。
- 工具升級或團隊協作慣例變更時更新本檔，**不需動憲章**。
- 若工具變更影響到憲章治權邊界（例：新工具能否進入 L2 / L3），須同步升版憲章 §0.0-E.4 / §0.0-F。
- 本檔升版不需通過 §6.8 同步治權審計（屬工具層，非系統治權）。

---

## 六、相關文件

| 文件 | 角色 |
|---|---|
| `README.md` | 對外專案說明 |
| `CLAUDE.md`（本檔） | AI 協作工具規則 SSOT |
| `reports/系統架構大憲章_v6.0.0.md` | 系統架構治權 SSOT |
| `reports/系統核心完整度評估報告.md` | 完整度評估參考 |
| `reports/*_research_*.md` | 逐程式 / 跨層研究報告 |
| `.claude/plans/` | 計畫檔暫存目錄 |
