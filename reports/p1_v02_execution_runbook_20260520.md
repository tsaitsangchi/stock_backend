# P1 v0.2 執行 Runbook（model retrain + ablation IC 驗證）

- **generated_at**: 2026-05-20 Asia/Taipei
- **scope**: 對 P1 v0.1（feature_store v0.3）之 model retrain + ablation 實證驗證
- **憲章基準**: `reports/系統架構大憲章_v6.0.0.md` §9.9 + §14.7-AD + §14.7-AE
- **執行環境需求**: 主環境 venv + PostgreSQL + psycopg2 + 既有 raw 資料表
- **預估時間**: 約 5-10 分鐘（feature build ~2min + model train ~3min + ablation ~1min）

---

## 一、前置條件確認

執行前須確認以下狀態：

```bash
# 1. venv 已啟用且 psycopg2 可載入
venv/bin/python -c "import psycopg2; print('OK')"

# 2. DB 連線可用
venv/bin/python -c "from scripts.core.db_utils import get_db_connection; print(get_db_connection())"

# 3. feature_store_builder.py 為 v0.3
grep 'TOOL_VER = "v0.3"' scripts/core/feature_store_builder.py

# 4. 既有 v0.1 / v0.2 feature_set 與 model 仍存在（作為 baseline）
psql -c "SELECT feature_set_id, feature_count FROM feature_store_snapshot
         WHERE as_of_date = '2026-04-25' ORDER BY feature_set_id;"
```

預期：
- feature_set_v0.1（27）、feature_set_v0.2（31，含 interaction）皆已 committed
- v0.3 尚未建立（本 runbook 第一個動作即為建立）

---

## 二、執行步驟

### Step 1：Commit v0.3 feature set（~2 分鐘）

```bash
venv/bin/python scripts/core/feature_store_builder.py \
  --commit \
  --as-of-date 2026-04-25 \
  --feature-set-version feature_set_v0.3 \
  --label-horizon 20
```

**預期結果**：
- feature_set_id = `fs_20260425_feature_set_v0_3`
- feature_definition rows = 31（27 base + 4 upside/downside；v0.2 interaction 排除）
- feature_values rows ≈ 4500-4600（150 stocks × ~30 features，部分 drop）
- null_imputed 紀錄與 v0.1 相當
- source_cutoff / as_of_date 符合 as-of-strict
- verdict = PERFECT

**驗證指令**：
```bash
psql -c "SELECT feature_set_id, feature_count, value_count, null_imputed_count, status
         FROM feature_store_snapshot
         WHERE feature_set_id = 'fs_20260425_feature_set_v0_3';"
```

預期：feature_count=31, status='committed'。

### Step 2：訓練 v0.3 model（~3 分鐘）

```bash
venv/bin/python scripts/core/model_trainer.py \
  --commit \
  --feature-set-id fs_20260425_feature_set_v0_3 \
  --label-horizon 20
```

**預期結果**：
- model_id = `mdl_20260425_lgbm_h20_<hash>_v0_1`（hash 由 trainer 計算）
- feature_count = 31
- ic_mean 預期 ≥ 0.35（與 v0.1 27 features 之 0.3712 相當或更佳）
- model_registry 寫入新 entry

**取得 model_id**：
```bash
psql -tA -c "SELECT model_id FROM model_registry
             WHERE feature_set_id = 'fs_20260425_feature_set_v0_3'
             ORDER BY created_at DESC LIMIT 1;"
```

### Step 3：填入 model_id 並執行 ablation

編輯 `scripts/maintenance/_oneoff_v03_upside_downside_ablation.py`，將 `MODEL_ID_V03 = None` 改為 Step 2 取得的實際 model_id。

```bash
venv/bin/python scripts/maintenance/_oneoff_v03_upside_downside_ablation.py
```

**預期輸出格式**：

```
======================================================================
v0.3 ablation @ as_of=2026-04-25, label_date=2026-05-15
Model: mdl_20260425_lgbm_h20_<hash>_v0_1
Feature set: fs_20260425_feature_set_v0_3 (31 features: 27 base + 4 upside/downside)
======================================================================
  stocks: ~147 (intersection of features × labels)

  full_ic = 0.XXXX
  n_stocks = ~147

  Group                | n   | IC w/o   | Δ vs full  | Verdict
  -------------------- | --- | -------- | ---------- | -------
  liquidity            | 4   | 0.XXXX   | -0.0XXX    | ✅ HELPFUL
  price                | 12  | 0.XXXX   | -0.0XXX    | ✅ HELPFUL (← v0.3 含 4 個新)
  institutional        | 5   | 0.XXXX   | -0.0XXX    | ✅ HELPFUL
  fundamental          | 4   | 0.XXXX   | -0.0XXX    | ✅/➖
  macro                | 4   | 0.XXXX   | -0.0XXX    | ➖ neutral
  theme                | 2   | 0.XXXX   | -0.0XXX    | ➖ neutral

  --- v0.3 個別新特徵 ablation ---
  Feature                        | IC w/o   | Δ vs full  | Verdict
  ------------------------------ | -------- | ---------- | -------
  upside_volatility_60d          | 0.XXXX   | -0.0XXX    | ✅/➖
  upside_capture_60d             | 0.XXXX   | -0.0XXX    | ✅/➖
  downside_volatility_60d        | 0.XXXX   | -0.0XXX    | ✅/➖
  downside_capture_60d           | 0.XXXX   | -0.0XXX    | ✅/➖

======================================================================
§9.9 P1 v0.1 升版條件驗證裁決
======================================================================
  ...verdict 行...

  📁 JSON 結果儲存於：reports/v03_upside_downside_ablation_result.json
```

**裁決規則**：

| helpful 個別特徵數 | 裁決 | 下一步 |
|---|---|---|
| ≥ 2/4 HELPFUL | ✅ v0.3 升版條件達成 | 啟動 P1 v0.3（修改 VolatilityControl）|
| 1/4 HELPFUL | ⚠️ 部分達成 | 保留 v0.3 為證據，暫不修改 VolatilityControl |
| 0/4 HELPFUL | ❌ 否決 | 同 P2 教訓，需非線性 trainer |

---

## 三、執行後動作（依裁決結果）

### 情境 A：≥ 2/4 HELPFUL（升版條件達成）

1. 將 `reports/v03_upside_downside_ablation_result.json` 內容轉為 §14.7-AF 入憲報告
2. 起草 §9.10 VolatilityControl Upside/Downside Score Contract v0.1
3. 修改 `core_universe_builder.py` 之 `_volatility_control_score()` 為 upside-tilted 公式
4. 重跑 core_universe_builder 產出新 snapshot（注意：會觸發 150 檔組成變動）
5. 依序重跑 feature_store_builder / model_trainer / prediction_engine / portfolio_sizer

### 情境 B：1/4 HELPFUL（部分達成）

1. 入憲 §14.7-AF 紀錄部分達成事實
2. 保留 v0.3 feature_set 為 production-current（與 v0.1 並存）
3. 暫不啟動 P1 v0.3 VolatilityControl 修改
4. 評估是否需要 P1 v0.4：採非線性 trainer（與 P2 教訓對應）

### 情境 C：0/4 HELPFUL（否決）

1. 入憲 §14.7-AF 紀錄否決事實（與 §0.0-D.6 #1 P2 否決並列）
2. 不部署 v0.3 為 production-current
3. 重新評估 P1 路徑——可能需跳過 v0.3，直接進入「非線性 trainer」途徑
4. 更新 §9.9-G 跨層影響預期

---

## 四、§14.7-AF 入憲報告框架（待填入實證）

```markdown
# §14.7-AF P1 v0.2 model retrain + ablation IC 實證

- **generated_at**: <YYYY-MM-DD>
- **scope**: 對 §9.9 P1 v0.1 feature_store v0.3 之 model retrain + ablation 實證
- **as_of_date**: 2026-04-25
- **label_date**: 2026-05-15
- **feature_set_id**: fs_20260425_feature_set_v0_3
- **model_id**: <由 Step 2 取得>
- **verdict**: <達成 / 部分達成 / 否決>

## 一、執行結果摘要

| 指標 | v0.1 baseline | v0.3 |
|---|---|---|
| full_ic | 0.3716 | <實際值> |
| 特徵數 | 27 | 31 |
| n_stocks | <實際值> | <實際值> |

## 二、Group ablation 結果

<從 ablation 輸出複製>

## 三、v0.3 個別新特徵 ablation

<從 ablation 輸出複製>

## 四、§9.9 升版條件驗證

<裁決規則套用>

## 五、與 P2 interaction features 之實證比較

| 維度 | P2 (已否決) | P1 v0.1 (本次) |
|---|---|---|
| 整組 ablation IC | +0.0131 (HARMFUL) | <實際值> |
| Stock-specific 性 | 偽 (broadcast) | 真實 |
| 結論 | 否決 | <達成/部分/否決> |

## 六、下一階段建議

<依情境 A/B/C 之動作清單>
```

---

## 五、回滾與安全網

若任一步驟異常：

```bash
# 回滾 v0.3 feature_set（如有需要）
psql -c "UPDATE feature_store_snapshot SET status='deprecated'
         WHERE feature_set_id = 'fs_20260425_feature_set_v0_3';"

# 回滾 v0.3 model（如有需要）
psql -c "UPDATE model_registry SET status='deprecated'
         WHERE model_id = '<MODEL_ID_V03>';"

# 既有 v0.1 production-current 仍維持，不受影響
```

---

## 六、驗證 checklist

執行完成後檢查：

- [ ] feature_set_v0.3 committed，feature_count=31
- [ ] model_id_v03 在 model_registry 為 committed
- [ ] ablation 腳本執行無誤，產出 reports/v03_upside_downside_ablation_result.json
- [ ] §14.7-AF 報告依範本填入實證數據
- [ ] §0.0-A.2 / §0.0-B 配置層百分比依結果更新（Level 2）
- [ ] commit + push + tag v6.1.4-P1-v02-ablation-completed

---

## 七、為何當前 worktree 無法執行

當前 Claude Code worktree（`.claude/worktrees/elated-volhard-131e58`）缺少：

- venv（無 psycopg2 / pandas / lightgbm 等依賴）
- DB 連線設定

因此本 runbook 僅準備腳本與框架，**實際執行需在主環境**（`/home/hugo/project/stock_backend/` + venv）。

**Claude Code 在當前環境已完成的事**：

1. ✅ feature_store_builder.py v0.3 程式實作（11 項單元測試通過）
2. ✅ scripts/maintenance/_oneoff_v03_upside_downside_ablation.py 撰寫
3. ✅ 本 runbook
4. ✅ §14.7-AF 框架草擬

**user 在主環境須完成的事**：

1. ⬜ 執行 Step 1-3 取得實證
2. ⬜ 填入 §14.7-AF 報告
3. ⬜ 依裁決進行 Level 2 內容同步
4. ⬜ commit + push + tag

---

**本 runbook 入憲為 §14.7-AF 之前置文件**。
