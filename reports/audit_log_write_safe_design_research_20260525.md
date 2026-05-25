# Audit Log Write-Safe Research — §7.4-A 姊妹缺陷治權閉環提案

- **撰寫日期**: 2026-05-25
- **對應憲章**: `reports/系統架構大憲章_v6.1.0.md`(7,344 行)
- **對應實證**: `reports/api_schema_compliance_audit_20260524_1425.md`(Audit 2 verdict=FAILED;Layer F dup=1)
- **性質**: 入憲提案 + 程式預備升版計畫(分階段授權)
- **撰寫角色**: Claude(Opus 4.7 1M context)

---

## 一、執行摘要(Headline)

| 項目 | 結果 |
|---|---|
| **缺口本質** | `data_audit_log` 表 DDL `unique_constraints=[]` + `write_data_audit_log()` 純 INSERT,**無 race-safe 保護** |
| **實證來源** | 2026-05-24 Audit 2 Layer F 揭露:`data_audit_log` 表有 1 個 race-induced duplicate(同 microsecond 兩 worker 並發寫入)|
| **對稱性** | §7.4-A multi-worker **讀側**(sync_engine)已治權閉環;**寫側**(audit log)為對稱姊妹缺陷 |
| **業務影響** | **無** — 業務 dataset 全部 PK uniqueness PASS;此 dup 只在 infra 觀測表 |
| **治權影響** | **有** — `data_audit_log` 為 §0.4 可觀察性唯一真理來源(憲章 L2469),race-induced dup 直接侵蝕觀測層 |
| **嚴重度** | Low(業務不影響)+ Medium(治權對稱性缺口)|
| **建議入憲位階** | **v6.1.0-patch** 單議題治權閉環(比照 §14.7-AO / AP / AT 模式)|
| **建議新契約編號** | **§3.2A.J** Audit Log Write-Safe(對應 §14.7-AY 治權閉環記錄)|

---

## 二、Root Cause 詳細分析

### 2.1 觀察事實

**Audit 2 v0.6 報告**(`reports/api_schema_compliance_audit_20260524_1425.md` L471-475):

```
### Layer F: Duplicate Row Detection
- PASS=12, FAIL=1
- ✅ [F] pipeline_execution_log.row: 無重複:10 (full(10))
- ❌ [F] data_audit_log.row: duplicate row:sample_total=22035, sample_distinct=22034, dup=1 (full(22035))
- ✅ [F] TaiwanStockPrice.row: 無重複:100020 (sampled)
  ...(其他 11 個業務表 + FRED 全 PASS)
- ✅ [F] TaiwanStockInfo.row: 無重複:2803 (full(2803))

verdict: FAILED
```

**EXECUTION_LOG** 描述(`reports/rebuild_logs/item3_v6.1.0_recursive/EXECUTION_LOG.md` L108-114):

```
### Phase 6 audit 2 v0.6 with `--db-sample-size 100000`(14:24-14:26)
- ⚠️ FAILED 1 個 Layer F dup
- root cause:`data_audit_log` 表 1 個 multi-worker race condition dup
  - `TaiwanStockPrice / SYNC / 2024-05-09 / UPSERT / 494 rows / timestamp 2026-05-24 09:48:42.236195`
  - Step 4F 啟動 ~65s 兩個 worker 並發寫 audit log → 同 microsecond 撞 race
- 不影響業務資料(TaiwanStockPrice 本身 UPSERT idempotent 正確)
```

### 2.2 程式現況

**`scripts/core/data_schema.py` L78-86**(現行 v2.16):

```python
"data_audit_log": {
    "columns": {
        "id": "SERIAL PRIMARY KEY",
        "table_name": "VARCHAR(255)",
        "stock_id": "VARCHAR(255)",
        "data_date": "DATE",
        "action_type": "VARCHAR(255)",
        "rows_affected": "INTEGER",
        "timestamp": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    },
    "unique_constraints": []   # ⚠️ 空白 — 無 UNIQUE constraint
}
```

**`scripts/core/db_utils.py` L294-305**(現行 v2.47):

```python
def write_data_audit_log(table_name, stock_id, data_date, action_type, rows_affected):
    """專項審計日誌 (v2.43 unchanged in v2.44) - 混合模式 B: data_audit_log"""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO data_audit_log (table_name, stock_id, data_date, action_type, rows_affected)
            VALUES (%s, %s, %s, %s, %s)
        """, (table_name, stock_id, data_date, action_type, rows_affected))
        conn.commit()
    finally:
        cur.close(); conn.close()
```

→ **純 INSERT,無冪等保護**;`timestamp` 由 PostgreSQL `CURRENT_TIMESTAMP` 自動填,microsecond 精度。

### 2.3 Race Condition 機制

```
時間軸(2026-05-24 Step 4F 啟動 ~65 秒):

   Worker A                       Worker B                    PostgreSQL
   ────────                       ────────                    ──────────
T+0    fetch TaiwanStockPrice
       (stock=2330 dataset=...)
       UPSERT 494 rows ✅
T+1    write_data_audit_log(
         TaiwanStockPrice, SYNC,
         2024-05-09, UPSERT, 494)
       ─────────────────────────────────────────────────────→ INSERT row id=2034
                                  fetch TaiwanStockPrice
                                  (stock=2330 dataset=...)
                                  UPSERT 494 rows
                                  (idempotent: 0 net change)
T+1.000236  (同 microsecond)      write_data_audit_log(
                                    TaiwanStockPrice, SYNC,
                                    2024-05-09, UPSERT, 494)
                                  ───────────────────────────→ INSERT row id=2035
                                                              (timestamp 同為
                                                               2026-05-24 09:48:42.236195)
T+2     Audit 2 Layer F scan:
        sample_distinct=22034 (5-tuple)
        sample_total=22035    (id 唯一)
        → dup=1 → FAIL
```

**核心問題**:
- PostgreSQL `CURRENT_TIMESTAMP` 在同 transaction 內為固定值,但兩個 worker 各自 connection 各自 transaction,撞同 microsecond 即產生看似「完全相同」的兩列(只有 SERIAL id 不同)
- Audit Layer F 用 DISTINCT 統計所有非 id 欄位 → 偵測到 dup

### 2.4 為何業務資料不受影響

業務 dataset(TaiwanStockPrice / PriceAdj / PER ...)的 DDL 全部有 `unique_constraints=["date", "stock_id"]`(見 `data_schema.py` L96, 105, 112, ...),且寫入經 `bulk_upsert()` 用 `ON CONFLICT DO UPDATE` 保證冪等。兩個 worker 撞同一筆業務資料只會產生**一筆**最終 row。

但 audit log 表設計上**允許重複記錄**(理論上每次寫入都是新事件,SERIAL id 自然 unique),DDL 未約束 5-tuple uniqueness;**INSERT 也沒有 ON CONFLICT 保護** — 因此 race 自然產生 dup。

---

## 三、§7.4-A 治權對稱性論證

| 治權面向 | §7.4-A 讀側(sync_engine) | 當前寫側(audit log) |
|---|---|---|
| **物件** | FinMind API HTTP 402 多 worker cascade | DB `data_audit_log` 表多 worker 並發寫入 |
| **治權邊界** | global_402_cooldown_until + cascade-skip | **無** — 純 INSERT |
| **多 worker 模式** | ✅ 治權閉環(2026-05-23 v1.22 落地)| ⚠️ 治權缺口 |
| **典型情境** | 4 worker 同 quota cap → cascade detection | 2 worker 同 microsecond → 同 5-tuple 寫入 |
| **mitigation** | 第一個撞 402 設 global cool-down,其他立即 skip | (待入憲)UNIQUE constraint + ON CONFLICT DO NOTHING |
| **實證** | Step 4F **0 cascade**(vs baseline 4 cascade)| Audit 2 dup=1(揭露於 2026-05-24)|

**對稱性結論**:§7.4-A 完成了「**讀側**多 worker 治權閉環」,§3.2A.J(本提案)補完「**寫側**多 worker 治權閉環」,合計達成 §7.4-A 之**多 worker 全閉環治權對稱**。

依憲章 §0.0-G「憲章先行紀律」+ §0.0-H「Level 1 強制契約通用模板」,本提案符合單議題治權閉環模式(§14.7-AO/AP/AT)。

---

## 四、§3.2A.J 完整條文草擬(放入憲章 L2717 之後)

```markdown
**§3.2A.J `db_utils.write_data_audit_log` Audit Log Write-Safe 治權契約
       (Race-Safe Audit Log) — *2026-05-25 v6.1.0-patch 入憲*(對應 §14.7-AY)**:

**觸發**:2026-05-24 從零遞迴驗證執行後置 Audit 2
(`audit_api_schema_compliance v0.6 --db-sample-size 100000 --include-fred`)
(`reports/api_schema_compliance_audit_20260524_1425.md`)實證:
**Layer F Duplicate Row Detection 發現 `data_audit_log` 表有 1 個 race-induced duplicate row**
(`TaiwanStockPrice / SYNC / 2024-05-09 / UPSERT / 494 rows / timestamp 2026-05-24 09:48:42.236195`),
導致 audit 整體 verdict=FAILED;root cause:Step 4F 啟動約 65 秒時,
兩個 sync_engine worker 在同 microsecond 並發呼叫
`db_utils.write_data_audit_log()` → INSERT 同 5-tuple → race condition 產生 dup。

**治權缺口本質**:`data_audit_log` 表現行 DDL `unique_constraints=[]`(僅 SERIAL `id` 為 PK,
無業務鍵 UNIQUE constraint);`db_utils.write_data_audit_log()` 為純 INSERT 無 ON CONFLICT 保護。
§7.4-A 已治權多 worker **讀側**(sync_engine 對 FinMind API HTTP 402 cascade),
但 multi-worker **寫側**(audit log)為對稱姊妹缺陷,構成 §7.4-A 治權閉環之未閉合邊界。
業務 dataset(TaiwanStockPrice 等 12 表)早有 `unique_constraints=["date", "stock_id"]`
+ `bulk_upsert()` ON CONFLICT DO UPDATE,本缺口僅影響 infra 觀測表。

**裁決**:
1. **`data_schema.py v2.16 → v2.17`**:`data_audit_log.unique_constraints`
   從 `[]` 改為 `["table_name", "stock_id", "data_date", "action_type", "timestamp"]`
   (5-tuple,timestamp 為 microsecond 精度作 race boundary).
2. **`db_utils.write_data_audit_log() v2.47 → v2.48`**:INSERT 改為
   `INSERT ... ON CONFLICT (table_name, stock_id, data_date, action_type, timestamp) DO NOTHING`,
   保證 multi-worker 並發呼叫不產生 dup,同時不阻斷 caller(冪等寫入).
3. **與 §7.4-A 對稱性**:本契約 + §7.4-A 共同完成「multi-worker 讀寫雙側治權閉環」.
4. **追溯適用 §3.2A.F Layer F Duplicate Row Detection**:既有 audit_api_schema_compliance
   對 `data_audit_log` 之 PASS 標準仍為「dup=0」;v6.1.0-patch 落地後,race-induced dup 應視為
   **已治權閉環**之歷史殘留,既有 DB 須執行一次性 dedup migration 再 ADD CONSTRAINT.
5. **不擴張至業務表**:本契約僅影響 `data_audit_log` infra 表;`pipeline_execution_log`
   不入此契約(SERIAL `id` 為 PK 自然 unique,且未發現 race-induced 問題);
   業務 dataset 已有業務鍵 UNIQUE constraint,不重複治權.

**裁決依據**:
(a) `data_audit_log` 為 §0.4 可觀察性「唯一真理來源」(憲章 L2469),
    race-induced dup 直接侵蝕觀測層治權,違反 §0.4;
(b) Audit Layer F dup 偵測機制有效(PROVEN 2026-05-24 揭露 race);但若 race 為已知容許情況,
    每次 multi-worker 跑後 audit 即會 FAIL,治權契約之 PASS 標準不可達,陷入 stalemate;
(c) `INSERT ... ON CONFLICT DO NOTHING` 為 PostgreSQL 標準冪等模式,
    效能成本極低(UNIQUE index 維持 O(log n) 查找),不影響長跑寫入吞吐量;
(d) timestamp microsecond 精度作為 UNIQUE 鍵第 5 元素,足以區分非 race 之合法重複寫入
    (同 dataset 同 date 之多次 sync 必落於不同 microsecond);
    若兩 worker 真撞同 microsecond + 同 4-tuple,視為「同一筆 audit 紀錄」自然合併,
    不影響觀測語義;
(e) `pipeline_execution_log` 不入此契約之依據:該表 SERIAL `id` 為唯一識別,
    且每次 lifecycle 必有獨立 start_time / end_time,自然 race-safe.

**§3.2A.J 自 v6.1.0-patch 起追溯適用**:既有 DB `data_audit_log` 表必須在升版時執行
一次性 dedup migration(保留 `MIN(id) GROUP BY (table_name, stock_id, data_date, action_type, timestamp)`),
否則 `ALTER TABLE ... ADD CONSTRAINT` 會因現有 dup 而失敗.

**v2.17 / v2.48 升版預備**(詳見 §14.7-AY):
- `data_schema.py`:`unique_constraints` 改為 5-tuple;`--init --force` 模式 idempotent(既有 UNIQUE constraint 同名時跳過)
- `db_utils.py`:`write_data_audit_log()` SQL 改 ON CONFLICT DO NOTHING
- 新增 `scripts/maintenance/migrate_data_audit_log_dedup_20260525.py`:
  一次性 migration 含 `--dry-run` 模式 + 顯式確認 prompt + dedup SQL + ADD CONSTRAINT
- 重跑 audit_api_schema_compliance v0.6 → 預期 Layer F dup=0 → verdict=PERFECT
```

---

## 五、§14.7-AY 治權閉環子節草擬(放入憲章 §14.7-AX 之後)

```markdown
### §14.7-AY 2026-05-25 v6.1.0 Operations Reality 二輪治權閉環:
data_audit_log race-induced duplicate 揭露 → §3.2A.J 寫側治權契約入憲
(§7.4-A 對稱姊妹缺陷補完)

**背景**:2026-05-24 從零遞迴驗證 Phase 6 後置 Audit 2
(`audit_api_schema_compliance v0.6 --include-fred --db-sample-size 100000`)
verdict=FAILED;唯一 FAIL 在 Layer F `data_audit_log.row: duplicate row dup=1`,
業務 dataset 全 PASS.

**Root Cause**:Step 4F 啟動 ~65s 兩個 sync_engine worker 並發呼叫
`db_utils.write_data_audit_log()`,撞同 microsecond + 同 5-tuple
→ 純 INSERT 無 ON CONFLICT → race condition 產生 dup
(`TaiwanStockPrice / SYNC / 2024-05-09 / UPSERT / 494 rows / 2026-05-24 09:48:42.236195`).

**治權對稱性論證**:§7.4-A multi-worker 讀側(sync_engine HTTP 402 cascade)已治權閉環;
寫側(audit log 並發寫入)為對稱姊妹缺陷.補完 §3.2A.J 即完成
multi-worker 讀寫雙側治權對稱.

**程式預備升版**:
| 程式 | 現行 | 升版後 | 變更 |
|---|---|---|---|
| `data_schema.py` | v2.16 | **v2.17** | `data_audit_log.unique_constraints` 加 5-tuple |
| `db_utils.py` | v2.47 | **v2.48** | `write_data_audit_log()` 加 ON CONFLICT DO NOTHING |
| 新檔 `scripts/maintenance/migrate_data_audit_log_dedup_20260525.py` | — | v0.1 | 一次性 dedup + ADD CONSTRAINT |

**§0.0-G 第 20 次跑通**(前 19 次:§9.2 / §9.9 / §9.1-A〜I / §14.7-AG / §14.7-AG 自我修正
/ §0.1-F / §14.7-AJ / §14.7-AK / §14.7-AL / §14.7-AM / §14.7-AM 雞與蛋補強 / §14.7-AN
/ §14.7-AO / §14.7-AP / §14.7-AQ / §14.7-AR / §14.7-AS / §14.7-AT / §14.7-AU).
本次為「v6.1.0 二輪實證(2026-05-24 從零遞迴驗證)→ 單議題治權閉環(§7.4-A 對稱補完)
→ v6.1.0-patch 升版」之模式,對比 §14.7-AU 為「多議題彙整升版 v6.1.0 minor bump」,
本次回歸 §14.7-AO 之單議題模式.

**實證關聯**:
- `reports/api_schema_compliance_audit_20260524_1425.md`(Audit 2 verdict=FAILED 原始實證)
- `reports/rebuild_logs/item3_v6.1.0_recursive/EXECUTION_LOG.md`(root cause 分析 L108-114)
- `reports/audit_log_write_safe_research_20260525.md`(本提案研究報告)
```

---

## 六、程式 diff 預覽

### 6.1 `scripts/core/data_schema.py`(L78-86 diff)

```diff
 "data_audit_log": {
     "columns": {
         "id": "SERIAL PRIMARY KEY",
         "table_name": "VARCHAR(255)",
         "stock_id": "VARCHAR(255)",
         "data_date": "DATE",
         "action_type": "VARCHAR(255)",
         "rows_affected": "INTEGER",
         "timestamp": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
     },
-    "unique_constraints": []
+    "unique_constraints": ["table_name", "stock_id", "data_date", "action_type", "timestamp"]
 },
```

外加:標頭 `TOOL_VER = "v2.16"` → `"v2.17"` + 修訂歷程加 v2.17 entry。

### 6.2 `scripts/core/db_utils.py`(L294-305 diff)

```diff
 def write_data_audit_log(table_name, stock_id, data_date, action_type, rows_affected):
-    """專項審計日誌 (v2.43 unchanged in v2.44) - 混合模式 B: data_audit_log"""
+    """專項審計日誌 (v2.48: §3.2A.J race-safe upgrade) - 混合模式 B: data_audit_log"""
     conn = get_db_connection()
     cur = conn.cursor()
     try:
         cur.execute("""
             INSERT INTO data_audit_log (table_name, stock_id, data_date, action_type, rows_affected)
             VALUES (%s, %s, %s, %s, %s)
+            ON CONFLICT (table_name, stock_id, data_date, action_type, timestamp) DO NOTHING
         """, (table_name, stock_id, data_date, action_type, rows_affected))
         conn.commit()
     finally:
         cur.close(); conn.close()
```

外加:標頭 `TOOL_VER = "v2.47"` → `"v2.48"` + 修訂歷程加 v2.48 entry + `CONSTITUTION_VER = "v6.0.0"` → `"v6.1.0"`(對齊現行憲章)。

⚠️ **注意 timestamp 隱含問題**:由於 `timestamp` 由 PostgreSQL `CURRENT_TIMESTAMP` 自動填,INSERT 沒明示 timestamp 值,ON CONFLICT 的 conflict_target 需在 INSERT 端先計算 — 實際上 PostgreSQL 會 evaluate `DEFAULT` 然後判斷是否衝突。雙 worker 真撞同 microsecond + 同 4-tuple 時:
- 兩個 transaction 各自 `INSERT` 評估 `DEFAULT CURRENT_TIMESTAMP`(同 microsecond)
- 第一個 transaction commit 後,UNIQUE index 已有此 5-tuple
- 第二個 transaction 嘗試 INSERT 同 5-tuple → ON CONFLICT 觸發 → DO NOTHING

理論上可行,但需在 migration 後實測驗證 ON CONFLICT 是否正確攔截 microsecond-level race。

### 6.3 新增 `scripts/maintenance/migrate_data_audit_log_dedup_20260525.py` 設計

```python
#!/usr/bin/env python3
"""
一次性 migration: data_audit_log dedup + ADD UNIQUE CONSTRAINT
對應憲章 §3.2A.J / §14.7-AY(v6.1.0-patch 2026-05-25 入憲)

執行流程:
  1. --dry-run: 報告 dup 數量、影響範圍,不改 DB
  2. --apply:
     (a) DELETE dup 保留 MIN(id) per 5-tuple
     (b) ALTER TABLE data_audit_log ADD CONSTRAINT ... UNIQUE (...)
     (c) record_lifecycle 寫入 pipeline_execution_log
     (d) verify: 重 query dup 應 = 0

安全機制:
  - 預設 --dry-run
  - --apply 需傳 --confirm 才實際執行
  - 在 ALTER 前自動 PG_DUMP backup data_audit_log 至 logs/
"""

import argparse, sys, os, subprocess, datetime
from pathlib import Path

# ...(完整實作見 Stage 3 落地時)
```

主要 SQL:

```sql
-- Step 1: 計數 dup
SELECT COUNT(*) FROM (
    SELECT table_name, stock_id, data_date, action_type, timestamp, COUNT(*) c
    FROM data_audit_log
    GROUP BY table_name, stock_id, data_date, action_type, timestamp
    HAVING COUNT(*) > 1
) dup_groups;

-- Step 2: dedup (保留 MIN(id))
DELETE FROM data_audit_log
WHERE id NOT IN (
    SELECT MIN(id)
    FROM data_audit_log
    GROUP BY table_name, stock_id, data_date, action_type, timestamp
);

-- Step 3: ADD CONSTRAINT
ALTER TABLE data_audit_log
ADD CONSTRAINT data_audit_log_5tuple_unique
UNIQUE (table_name, stock_id, data_date, action_type, timestamp);

-- Step 4: verify
SELECT COUNT(*) FROM (
    SELECT table_name, stock_id, data_date, action_type, timestamp, COUNT(*) c
    FROM data_audit_log
    GROUP BY table_name, stock_id, data_date, action_type, timestamp
    HAVING COUNT(*) > 1
) dup_check;  -- 應為 0
```

---

## 七、Stage 1-5 完整授權路線

| Stage | 內容 | 邊界 | 可逆性 | 估時 | 風險 |
|---|---|---|---|---|---|
| **1** | 寫本研究報告 → `reports/` | 純寫 reports/ | 100%(刪檔) | 8-10 min | ✅ 已完成本檔 |
| **2** | 升 charter v6.1.0-patch:加 v6.1.0-patch entry + §3.2A.J + §14.7-AY | 動 charter md | High(git 可回退)| 10-15 min | 治權檔變動,但 git 可回退 |
| **3** | 升 `data_schema.py v2.17` + `db_utils.py v2.48` + 新建 migration script | 動程式碼,不執行 migration | High(git 可回退)| 20-25 min | 程式變動,但未動 DB |
| **4** | 跑 migration(dry-run → apply)+ 重跑 audit 2 | **動 DB schema + 資料** | LOW(dedup 不可逆,但 ADD CONSTRAINT 可 DROP) | 5-10 min | ⚠️ **真正破壞性** — 需 DB 備份 |
| **5** | 更新 charter v6.1.0 memory + 接力點 memory + 建議 commit | 動 memory + 可選 git | High | 5 min | — |

---

## 八、風險與回退方案

### 8.1 主要風險

| 風險 | 機率 | 影響 | 緩解 |
|---|---|---|---|
| ON CONFLICT 對 microsecond race 攔截不完整 | Low | Audit 仍有 dup | Stage 4 驗證後若仍有 dup,改用 application-level lock |
| migration 時 dedup 誤刪 | Low | 觀測層歷史不準 | --dry-run 預檢 + PG_DUMP backup |
| 升 charter 後與下游模組 metadata 不同步 | Low | 後續 audit 工具讀錯版本 | Stage 5 同步更新 memory + 全模組 audit 重跑 |
| ON CONFLICT DO NOTHING 隱藏其他 bug(如真正不該重複的 audit log)| Very Low | 觀測層漏記 | DO NOTHING 寫入 stderr WARN(後續 v2.49 可加) |

### 8.2 回退方案

| Stage 失敗 | 回退步驟 |
|---|---|
| Stage 1 | 刪 `reports/audit_log_write_safe_research_20260525.md` |
| Stage 2 | `git checkout reports/系統架構大憲章_v6.1.0.md` |
| Stage 3 | `git checkout scripts/core/data_schema.py scripts/core/db_utils.py` + 刪 migration script |
| Stage 4 | `ALTER TABLE data_audit_log DROP CONSTRAINT data_audit_log_5tuple_unique`(但 dedup 不可逆 — 需從 PG_DUMP restore)|
| Stage 5 | `git checkout` memory 檔 |

---

## 九、決策點(待你裁定)

### 9.1 推進粒度

| 選項 | 操作邊界 | 適用情境 |
|---|---|---|
| **A. 停在 Stage 1** | ✅ 已完成 | 只想看提案,後續另排 |
| **B. 推進到 Stage 2** | + 升 charter | 確認治權契約成立,延後程式升版 |
| **C. 推進到 Stage 3** | + 升程式但不動 DB | 程式 ready,但 DB migration 你想手動或晚點再做 |
| **D. 五階段一氣呵成** | + 動 DB + 驗證 | 全套治權閉環,需 DB 備份意願 |

### 9.2 憲章修訂簽署身份

| 選項 | 模式 |
|---|---|
| **甲. Claude 補登** | Claude 以 charter 既有先例(audit_doctrine_compliance v0.4 等)直接補登;你 commit 時以你身份簽署 |
| **乙. 等 Codex 接手** | 我只寫 reports/(本檔已完成),charter 修訂等 Codex 處理(保持「多議題彙整治權升版」由 Codex 簽署的歷史慣例)|

### 9.3 v6.1.0-patch 編號

| 選項 | 含義 |
|---|---|
| **編 v6.1.0-patch §3.2A.J / §14.7-AY** | 與本提案一致 |
| **編 v6.1.0-patch §14.7-AY only**(不開 §3.2A.J)| 治權閉環記錄但不新增 §3.2A 子節 |
| **升 v6.1.0.1 minor patch number**(對齊 SemVer)| 程式對齊度更高,但 charter 沒先例 |

---

## 十、結論與建議

**Claude 建議**:
- 推進粒度:**B(Stage 1+2)**— 先完成治權契約入憲,程式升版另行排序;Stage 3 前可加碼決定 ON CONFLICT 攔截實測結論
- 修訂身份:**甲(Claude 補登)**— 比照 audit_doctrine_compliance v0.4 等既有先例,降低跨人 handoff 摩擦
- patch 編號:**v6.1.0-patch §3.2A.J / §14.7-AY** — 符合既有 §14.7-AO/AP/AT 模式

**最大風險**:Stage 4 的 dedup 不可逆,且 ON CONFLICT 對 microsecond race 攔截需實證。若你選 D(五階段一氣呵成),強烈建議先 PG_DUMP backup `data_audit_log` 表。

**最小可行路徑**(極保守):停在 Stage 1,把這份報告掛入接力點 memory 的「治權 follow-up」list,等下次 from-zero rebuild 時自然 race 重現再決定升版。但這會讓「§7.4-A 治權閉環不完整」這個事實**持續存在**。

---

*Research drafted 2026-05-25 by Claude(Opus 4.7 1M context)*
*Next step: 等使用者裁定 9.1 / 9.2 / 9.3 三決策點 → 進 Stage 2-5 或停在 Stage 1*
