# CLAUDE.md — AI 協作工具規則 (Single Source of Truth)

**文件性質**：本檔為本專案 AI 協作工具規則之單一事實來源 (SSOT)。
**位階聲明**：所有**治權禁令、工程契約、架構治權**以 `reports/系統架構大憲章_v6.0.0.md`（以下簡稱「憲章」）為準；本檔僅承載「**如何使用 AI 工具編輯本專案**」之短半衰期協作規則。
**最後更新**：2026-05-21

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
