# CLAUDE.md — AI 協作工具規則 (Single Source of Truth)

**文件性質**：本檔為本專案 AI 協作工具規則之單一事實來源 (SSOT)。
**位階聲明**：所有**治權禁令、工程契約、架構治權**以 `reports/系統架構大憲章_v6.0.0.md`（以下簡稱「憲章」）為準；本檔僅承載「**如何使用 AI 工具編輯本專案**」之短半衰期協作規則。
**最後更新**：2026-05-29(§一.11 程式標頭強制三段式入憲;對應 sovereign_sync_engine.py 標頭範本 + §四.4 8 項標頭評估治權慣例)

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

**Empirical 反論(corrective bundle III + IV)**:
- Bundle III v0.19:LightGBM ded +0.64 / CatBoost ded +0.39 / RF Win 100%
- Bundle IV v0.20:XGBoost ded 4.55(全 v0.20 最高)/ LGBM v0.2 +0.47 / ET +0.39 / RF +0.26
- **doctrine purity → 更乾淨 signal,不損 production 表現**

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
