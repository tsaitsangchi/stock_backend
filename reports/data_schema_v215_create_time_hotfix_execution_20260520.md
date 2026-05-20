# `data_schema v2.15` create_time Hotfix 落地實證紀錄

**執行日期**: 2026-05-20 (Asia/Taipei)
**對應憲章**: §二 Step 2 / §3.1 L2440 / **§14.7-AK Step 3 落地實證**
**對應 commit**: Step 2 入憲 = `98f1707`；Step 3 落地（本實證 + 後續 commit）
**裁決**: ✅ **PERFECT**（首次「audit 揭露 → DDL 修補 → 重測 PERFECT」治權閉環完整完成）

---

## 一、Hotfix 緣由

`audit_api_schema_compliance v0.1` 首次實測（§14.7-AJ Step 3 / commit `608c5e8` /
reports/api_schema_compliance_audit_20260520_2059.md）揭露：

| 議題 | 內容 |
|---|---|
| 表 / 欄位 | `TaiwanStockMonthRevenue.create_time` |
| DDL 宣告 | `TIMESTAMP`（預期 `YYYY-MM-DD HH:MM:SS`，19 字元）|
| FinMind API 實際回傳 | `'2026-04-21'` 等 10 字元 DATE 字串 |
| Layer B cast 失敗 | 3/3（樣本中所有非空值皆為 DATE 格式而非 TIMESTAMP）|
| Verdict | FAILED；exit 1 |

依使用者裁決「**audit 工具須嚴格、不得包容違規**」，採**甲案**修 DDL 對齊 API 物理本質。

---

## 二、Step 2 入憲（commit `98f1707`）

| 變動位置 | 內容 |
|---|---|
| `data_schema.py` `DATASET_REGISTRY` | `TaiwanStockMonthRevenue.create_time`: `"TIMESTAMP"` → `"DATE"` |
| `data_schema.py` 標頭 / TOOL_VER / 維運矩陣 / report_results | v2.14 → v2.15 |
| `data_schema.py` 修訂歷程 | 新增 v2.15 條目；v2.14 標記 SUPERSEDED |
| 憲章 §二 L2408-2409 | data_schema v2.14 → v2.15 |
| 憲章 §3.1 L2440 模組清單 | v2.14 → v2.15 + 補加 hotfix 描述 |
| 憲章 §3.2 L2483 對齊註記 | 補登 v2.15 hotfix 紀錄 |
| 憲章 §14.7-AK | 新增整節（hotfix 起源 + 治權裁決 + 物理依據 + Step 2 入憲 + Step 3 預告）|

---

## 三、Step 3 落地實證（本實證）

### 3.1 重建 13 表（含 DDL DATE）

**命令**：
```bash
venv/bin/python scripts/core/data_schema.py --init --force
```

**結果**：
| 指標 | 值 |
|---|---|
| API contract probe | 11/0/0（全 PASS） |
| DDL 主權重鑄 | 13/13 SUCCESS |
| Verdict | **PERFECT ALIGNMENT**（動態計算） |
| Exit code | **0** |
| 耗時 | **2,704.88 ms** |

對齊 §3.2 Step 2 接受標準。

### 3.2 重跑 9 層 audit

**命令**：
```bash
venv/bin/python scripts/maintenance/audit_api_schema_compliance.py --include-fred
```

**結果（完整 9 層 PASS / FAIL）**：

| Layer | 名稱 | PASS | FAIL | 對比 v2.14 |
|---|---|---:|---:|---|
| A | DDL ↔ DB Physical Consistency | 119 | 0 | ✅ 持平 |
| **B** | **API Sample ↔ DDL Type Compatibility** | **102** | **0** | ✅ **101 → 102 / 1 → 0**（hotfix 解決）|
| C | API Sample Length / Precision Range | 83 | 0 | ✅ 持平 |
| D | NULL Ratio Sanity | 103 | 0 | ✅ 持平 |
| E | PK / Unique Constraint Uniqueness | 11 | 0 | ✅ 持平 |
| F | Duplicate Row Detection | 13 | 0 | ✅ 持平 |
| G | Date Series Continuity | 11 | 0 | ✅ 持平 |
| H | Referential Integrity | 9 | 0 | ✅ 持平 |
| I | Value Range Sanity | 8 | 0 | ✅ 持平 |
| **合計** | — | **459** | **0** | **458/1 → 459/0** |

**總計指標**：
| 指標 | v2.14 之首測 | v2.15 之 hotfix 後 |
|---|---|---|
| Total PASS | 458 | **459** |
| Total FAIL | 1 | **0** |
| Verdict | FAILED | **PERFECT** |
| Exit code | 1 | **0** |
| 耗時 | 3,087 ms | **2,793 ms** |
| 報告 | reports/api_schema_compliance_audit_20260520_2059.md | **reports/api_schema_compliance_audit_20260520_2119.md** |

---

## 四、Layer B 治權閉環驗證

| 階段 | 動作 | 結果 |
|---|---|---|
| 1. Audit 首測（§14.7-AJ Step 3） | 跑 v0.1 工具 9 層 | Layer B 揭露 1 FAIL（create_time TIMESTAMP↔DATE mismatch） |
| 2. 使用者裁決 | 「audit 工具須嚴格、不得包容違規」 | 採甲案修 DDL |
| 3. Step 2 入憲（commit `98f1707`） | 修 DDL 為 DATE + 憲章 §14.7-AK | v2.15 |
| 4. Step 3 重建（本實證 3.1） | `--init --force` 用新 DDL 重建 13 表 | PERFECT ALIGNMENT |
| 5. Step 3 重測（本實證 3.2） | 跑 v0.1 工具 9 層 | **Layer B 0 FAIL；Total 459/0；PERFECT** |
| 6. **治權閉環完成** | — | **§0.0-G 第 8 次跑通；首次「audit 揭露 → DDL 修補」閉環完成** |

---

## 五、Layer B 之 102 / v2.14 之 101 之差

PASS 數從 101 升至 102 — 多了 1 個。原因：

| 樣本欄位 | v2.14 結果 | v2.15 結果 |
|---|---|---|
| `TaiwanStockMonthRevenue.create_time` | cast 失敗 3/3（DDL TIMESTAMP）| cast 通過 3/3（DDL DATE）|

Layer B 之計數方式為「每個 (table, column) 為一個 PASS / FAIL 單元」；v2.14 此單元為 FAIL，
v2.15 為 PASS，故 Total PASS +1，Total FAIL -1。

---

## 六、邏輯動量裁決

| 治權項 | 變動 |
|---|---|
| `DATASET_REGISTRY` 表數 | 13 → 13（不變） |
| `TaiwanStockMonthRevenue` 欄位數 | 10 → 10（不變） |
| `TaiwanStockMonthRevenue.unique_constraints` | `["date", "stock_id"]` → 不變 |
| `probe_api_contracts()` 邏輯 | 不變 |
| Verdict 動態計算（L393-396） | 不變 |
| CLI `--init / --force / --table / --skip-api-contract` | 不變 |
| §1〜§7 強制契約 | 不變 |
| §6.7 SQL | 不變 |
| 25 維路徑 | 不變 |
| 九步合法執行序 | 不變 |
| §7 三層防禦 | 不變 |
| §8 / §9 條文 | 不變 |
| **僅一處變動** | **`create_time` DDL：`TIMESTAMP` → `DATE`** |

---

## 七、自我合規檢驗

| §3.2A 治權邊界要求 | 實證 |
|---|---|
| §5.6.3 零硬編 PERFECT | ✅ v0.1 之 `_compute_verdict()` 動態回 PERFECT（前次回 FAILED 也是動態計算） |
| §3.2 接受標準 | ✅ PERFECT → exit 0；前次 FAILED → exit 1 |
| §0.4 可觀察性 | ✅ `record_lifecycle` + `write_data_audit_log` 接線正常 |
| §0.0-I 單一引用源 | ✅ DATASET_REGISTRY 為唯一 schema SSOT；v2.15 hotfix 也只在 DATASET_REGISTRY 修一處 |
| §0.0-G 憲章先行紀律 | ✅ Step 2 入憲 commit `98f1707` 先於本 Step 3 落地（嚴格分離雙 commit 模式之第 2 次） |

---

## 八、§0.0-G 第 8 次跑通完整實證表

| Step | 動作 | 狀態 | Commit |
|---|---|---|---|
| Step 1（研究） | 揭露起源（§14.7-AJ Step 3 落地實證之 Layer B FAIL） | ✅ 含於 §14.7-AJ |
| 使用者裁決 | 「audit 工具須嚴格、不得包容違規」採甲案 | ✅ | — |
| Step 2（入憲） | 修 DDL + 憲章 §3.1 / §3.2 / §14.7-AK 補登 | ✅ | `98f1707` |
| Step 3（落地） | `--init --force` 重建 + audit 重測 PERFECT + 本實證報告 + §14.7-AK Step 3 補登 | ✅ **本 commit** | （後續 commit） |

**§0.0-G 第 8 次跑通**（前七次：§9.2 / §9.9 / §9.1-A〜I / §14.7-AG / §14.7-AG 自我修正 /
§0.1-F / §14.7-AJ）。**第 2 次嚴格分離雙 commit 模式**。

---

## 九、結論

✅ **「audit 揭露 → DDL 修補 → 重測 PERFECT」治權閉環完整完成**：

1. `audit_api_schema_compliance v0.1` 工具首次執行（§14.7-AJ Step 3）即揭露真實 schema-API
   mismatch — 證明工具治權能力
2. 依使用者治權原則裁決（採甲案修 DDL）
3. v2.14 → v2.15 hotfix 入憲（§14.7-AK Step 2）
4. 本實證重建 + 重測通過（Layer B 1 FAIL → 0 FAIL；459/0 全 PASS；PERFECT）
5. 工具自身嚴格性保留（未放寬 Layer B cast 邏輯，符合 §3.2A 治權邊界）

**HEAD commit at execution time**：`98f1707`（v2.15 hotfix Step 2 入憲）
**本實證將觸發**：§14.7-AK Step 3 實證段補登 + 下一 commit 收尾

**下一步**：依憲章 §二 接續九步序列（Step 2B `core_universe_schema.py --init` 或 Step 1.1.3 元件審計）
