# Universe Completeness Governance — Cross-Layer × Cross-Pillar Tracking Architecture Design Research

**日期**: 2026-05-26
**Phase**: A(設計研究 / pre-charter inscription)
**對應軌道**: v6.3.2 或 v6.4.0 (cross-layer completeness governance milestone)
**對應憲章基礎**: §0.4 數位孿生完整性 / §8 三層職責邊界 / §6.7.1 dynamic universe / §14.7-BT closure
**Status**: ✅ Phase A 完整(15 章 / non-destructive / 不動 DB 不動 code)

---

## 1. 觸發背景

用戶於 2026-05-26 提問:

> 「這119支個股依第一性原理、八二法則、康波週期核心思想都具有對應的資料來源來進行核心股,需要另外建立 table 來進行儲存嗎? 如果未來需記錄資料的抓取完整性、特徵產生完整性、模型產生完整性、預測產生完整性。」

此問題揭露:
- 既有 28 個 DB tables 雖各層 audit 既有(fetch_log / data_audit_log / feature_store_* / model_*),但 **缺跨層 SSOT** 回答「某 stock 在某 as_of_date 三基柱 × 四層完整性?」
- §8 預測層 schema **0% 缺失**(prediction tables 不存在)
- §0.4 數位孿生完整性需 explicit cross-layer contract,目前 implicit

---

## 2. Pillar × Layer 矩陣(3 × 4 = 12 cells)

| Layer ↓ \ Pillar → | §0.1 第一性原理 | §0.2 八二法則 | §0.3 康波週期 |
|---|---|---|---|
| **資料抓取** | M(price) / V(financial) / F(institutional) / ΔlnP(price) 共 5 sources | universe membership 119 stocks(已實 = 100%) | M2SL / T10Y2Y / VIXCLS / TW_SEMI / TW_SHIPPING 共 5 sources |
| **特徵產生** | 27 base features(price/liquidity/fundamental/institutional) | core/convex tier 標籤(via membership) | 4 macro + 2 theme(共 6 features) |
| **模型產生** | model 是否 train 該 stock(coverage 是否含此 119 之 stock) | sector_balance Lagrangian 是否生效 | (隱式;通過 features 進入 model) |
| **預測產生** | per-stock prediction 是否 generated | top-K rank coverage(per Pareto) | (隱式) |

**12 cells × 119 stocks = 1,428 應追蹤 (stock, pillar, layer) 三元組**;每元組需 0-100% completeness 標記。

---

## 3. 既有 28 Tables Inventory(per 2026-05-26)

### 3.1 Raw Data Tables (12)
| Table | 三基柱對應 | 完整性追蹤現況 |
|---|---|---|
| TaiwanStockPrice / PriceAdj | §0.1 M + ΔlnP | fetch_log per fetch |
| TaiwanStockFinancialStatements | §0.1 V | fetch_log |
| TaiwanStockMonthRevenue | §0.1 V | fetch_log |
| TaiwanStockInstitutionalInvestorsBuySell | §0.1 F | fetch_log |
| TaiwanStockMarginPurchaseShortSale | §0.1 F | fetch_log |
| TaiwanStockDividend | §0.1 V/M | fetch_log |
| TaiwanStockPER | §0.1 V | fetch_log |
| TaiwanStockShareholding | §0.1 F | fetch_log |
| TaiwanStockInfo | §0.3 I(industry tag) | (statically populated;少 sync) |
| FredData | §0.3 M/B/R(DFF/T10Y2Y/VIX/UNRATE) | fetch_log |
| fred_series | §0.3 M(M2SL) | fetch_log |

### 3.2 Governance Tables (5)
| Table | 角色 |
|---|---|
| core_universe_snapshot | universe 元資料(committed snapshot) |
| core_universe_membership | per stock × per snapshot tier |
| core_universe_scores | per stock × per snapshot CoreScore |
| core_universe_policy | policy_version 治權 |
| universe_revision_log | snapshot 變更歷史 |

### 3.3 Feature Tables (3)
| Table | 角色 |
|---|---|
| feature_store_snapshot | per feature_set_id 元資料 |
| feature_definition | per feature_set × per feature 定義 |
| feature_values | per feature_set × per stock × per feature value |

### 3.4 Model Tables (2)
| Table | 角色 |
|---|---|
| model_registry | per model_id 元資料(含 metrics JSONB) |
| model_training_run | per training run 記錄 |

### 3.5 Prediction Tables (**0** ⚠️)
- **完全空缺**(per §8 三層職責邊界,應有 prediction_run + predictions 兩表)
- 此為治權 0% gap;必須補建

### 3.6 Audit / Lifecycle Tables (3)
| Table | 角色 |
|---|---|
| data_audit_log | per table × per stock × per data_date audit |
| pipeline_execution_log | per task_name × per stock 任務記錄 |
| fetch_log | per fetch granular log(per stock × table × date_range) |

### 3.7 K-wave / Other (3)
| Table | 角色 |
|---|---|
| kwave_supply_cycle_proxy | TW_SEMI + TW_SHIPPING VWAP YoY proxies |
| theme_taxonomy | THEME_KEYWORDS taxonomy |
| stock_theme_map | per stock theme tagging |

---

## 4. Gap Analysis (4 Layers)

### 4.1 資料抓取層 Gap

| 需求 | 既有支撐 | Gap |
|---|---|---|
| Per fetch 細粒度 log | ✅ fetch_log (12 cols) | — |
| Per stock × per source × per day 覆蓋率 | 🟡 data_audit_log (aggregated) | **缺 per-stock × per-pillar rollup view** |
| Pipeline 任務狀態 | ✅ pipeline_execution_log | — |
| 三基柱跨表 cross-source completeness | ❌ 需即時 JOIN 5+ tables | **缺 SSOT** |

**評分**: ~75% covered;缺 trinity rollup。

### 4.2 特徵產生層 Gap

| 需求 | 既有支撐 | Gap |
|---|---|---|
| Feature set 元資料 | ✅ feature_store_snapshot | — |
| 31 features 定義 | ✅ feature_definition | — |
| Per stock × per feature 值 | ✅ feature_values | — |
| Per stock × per feature_group rollup | 🟡 GROUP BY 即時計算 | **缺 materialized view** |
| 三基柱對應 feature_set 對齊度 | ❌ feature 未明示 pillar tag | **缺 pillar_tag 欄** |

**評分**: ~85% covered;缺 pillar tagging。

### 4.3 模型產生層 Gap

| 需求 | 既有支撐 | Gap |
|---|---|---|
| Model 元資料 | ✅ model_registry (16 cols / metrics JSONB) | — |
| Training run 記錄 | ✅ model_training_run | — |
| Per stock × per model training coverage | 🟡 metrics JSON 內(隱埋) | **缺 explicit table** |
| Walk-forward panel × IC 表格化 | 🟡 metrics JSON | **缺 explicit table** |
| sector_balance audit | 🟡 metrics JSON | **缺 explicit table** |

**評分**: ~70% covered;walk-forward / sector_balance 隱埋 JSON。

### 4.4 預測產生層 Gap

| 需求 | 既有支撐 | Gap |
|---|---|---|
| Prediction run snapshot | ❌ 不存在 | **整層空缺** |
| Per stock predictions | ❌ 不存在 | **整層空缺** |
| Per stock × per as_of_date 預測值 | ❌ 不存在(可能寫 JSON file) | **整層空缺** |

**評分**: **0% covered**;治權 critical gap。

---

## 5. 3 個 Architecture Path 對比評估

### 5.1 Path A: 最小增量(reuse + view)

**新建**:
- prediction_run + predictions(2 表 / 補 §8 0% gap)
- materialized view `universe_completeness_matrix_current`(JOIN 既有 5 tables)

**Pros**: 最少 schema 變動 / DRY 原則
**Cons**: View 需 refresh / 無 explicit completeness contract / drift 風險

### 5.2 Path B: 完整 governance 表新建

**新建**:
- data_completeness_matrix
- feature_completeness_matrix
- model_completeness_matrix
- prediction_completeness_matrix
- prediction_run + predictions
- universe_completeness_rollup
- (共 6 表)

**Pros**: 每層完整性 first-class
**Cons**: 高寫入成本 / 與既有 audit 重複 / drift 風險 / 違 §一 #3

### 5.3 Path C: Hybrid(★推薦★)

**新建**:
- prediction_run + predictions(2 表 / 補 §8)
- universe_completeness_snapshot(1 表 / 三基柱 × 四層 SSOT)
- materialized view `universe_completeness_matrix_current`(1 view)
- 共 3 表 + 1 view

**Pros**:
- 每層 builder 仍主管自己 audit(layer authority 不破)
- 新表 1 個 SSOT 統一 trinity 視角
- 補預測層 0% gap
- 符合 §一 #3
- 對齊 §0.4 數位孿生完整性

**Cons**:
- 4 builders 需各補 completeness write hook(寫入紀律)
- Materialized view 需 refresh 機制

**評估結論**: **Path C 為最佳平衡**(empirically 評估三向 trade-off)。

---

## 6. Path C 提案 Schema 草案

### 6.1 prediction_run

```sql
CREATE TABLE prediction_run (
    run_id                  VARCHAR(255) PRIMARY KEY,
    model_id                VARCHAR(255) NOT NULL,
    feature_set_id          VARCHAR(255) NOT NULL,
    universe_snapshot_id    VARCHAR(255) NOT NULL,
    as_of_date              DATE NOT NULL,
    label_horizon           INTEGER NOT NULL,
    total_predictions       INTEGER NOT NULL DEFAULT 0,
    null_predictions        INTEGER NOT NULL DEFAULT 0,
    status                  VARCHAR(64) NOT NULL DEFAULT 'draft',
    sector_balance_applied  BOOLEAN DEFAULT FALSE,
    sector_balance_params   JSONB,
    notes                   TEXT,
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (model_id, as_of_date),
    FOREIGN KEY (model_id) REFERENCES model_registry(model_id),
    FOREIGN KEY (feature_set_id) REFERENCES feature_store_snapshot(feature_set_id),
    FOREIGN KEY (universe_snapshot_id) REFERENCES core_universe_snapshot(snapshot_id)
);

CREATE INDEX idx_prediction_run_as_of ON prediction_run(as_of_date DESC);
CREATE INDEX idx_prediction_run_status ON prediction_run(status, as_of_date);
```

### 6.2 predictions

```sql
CREATE TABLE predictions (
    run_id              VARCHAR(255) NOT NULL,
    stock_id            VARCHAR(255) NOT NULL,
    predicted_value     NUMERIC,
    prediction_rank     INTEGER,
    raw_value           NUMERIC,
    sector_penalty      NUMERIC,
    industry_category   VARCHAR(255),
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (run_id, stock_id),
    FOREIGN KEY (run_id) REFERENCES prediction_run(run_id) ON DELETE CASCADE
);

CREATE INDEX idx_predictions_stock ON predictions(stock_id, run_id);
CREATE INDEX idx_predictions_rank ON predictions(run_id, prediction_rank);
```

### 6.3 universe_completeness_snapshot(治權新表)

```sql
CREATE TABLE universe_completeness_snapshot (
    snapshot_id              VARCHAR(255) NOT NULL,
    universe_snapshot_id     VARCHAR(255) NOT NULL,
    as_of_date               DATE NOT NULL,
    stock_id                 VARCHAR(255) NOT NULL,
    pillar                   VARCHAR(32) NOT NULL,
    layer                    VARCHAR(32) NOT NULL,
    expected_items           INTEGER NOT NULL,
    actual_items             INTEGER NOT NULL,
    completeness_pct         NUMERIC(5,2) NOT NULL,
    missing_items            JSONB,
    evidence_source_table    VARCHAR(255),
    computed_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (snapshot_id, stock_id, pillar, layer),
    CHECK (pillar IN ('first_principle', 'pareto', 'kondratiev')),
    CHECK (layer IN ('data', 'feature', 'model', 'prediction')),
    CHECK (completeness_pct BETWEEN 0 AND 100),
    FOREIGN KEY (universe_snapshot_id) REFERENCES core_universe_snapshot(snapshot_id)
);

CREATE INDEX idx_completeness_universe ON universe_completeness_snapshot(universe_snapshot_id);
CREATE INDEX idx_completeness_query ON universe_completeness_snapshot(pillar, layer, completeness_pct DESC);
CREATE INDEX idx_completeness_stock ON universe_completeness_snapshot(stock_id, snapshot_id);
```

### 6.4 Materialized View: universe_completeness_matrix_current

```sql
CREATE MATERIALIZED VIEW universe_completeness_matrix_current AS
SELECT 
    c.stock_id,
    c.pillar,
    c.layer,
    c.completeness_pct,
    c.actual_items,
    c.expected_items,
    c.missing_items,
    c.evidence_source_table,
    u.policy_version,
    u.as_of_date AS universe_as_of_date,
    c.as_of_date AS completeness_as_of_date,
    c.computed_at
FROM universe_completeness_snapshot c
JOIN core_universe_snapshot u ON c.universe_snapshot_id = u.snapshot_id
WHERE u.status = 'committed'
  AND u.as_of_date = (
      SELECT MAX(as_of_date) FROM core_universe_snapshot WHERE status = 'committed'
  );

CREATE UNIQUE INDEX idx_completeness_mv_unique 
ON universe_completeness_matrix_current(stock_id, pillar, layer);
```

---

## 7. 4 Builders Audit Hook 設計

### 7.1 Data Layer Hook(任何 fetcher)

```python
def write_data_completeness(stock_id, as_of_date, universe_snapshot_id):
    """
    對每支 stock 寫入 3 records(per pillar):
    - first_principle: 5 sources / actual = count of fetched sources / pct = actual/5
    - pareto: 1 (univ membership) / actual = 1 if in core/convex / pct = 100 or 0
    - kondratiev: 5 sources / actual = count of macro sources fetched / pct = actual/5
    """
    pass
```

### 7.2 Feature Layer Hook(feature_store_builder)

```python
def write_feature_completeness(stock_id, feature_set_id, universe_snapshot_id):
    """
    對每支 stock 寫入 3 records:
    - first_principle: 22 features expected / actual = non-null count / pct = actual/22
    - pareto: 1 (membership status) / actual = 1 if in core/convex / pct = 100 or 0
    - kondratiev: 6 features expected / actual = non-null count / pct = actual/6
    """
    pass
```

### 7.3 Model Layer Hook(model_trainer)

```python
def write_model_completeness(stock_id, model_id, universe_snapshot_id):
    """
    對每支 stock 寫入 1-3 records:
    - first_principle: 1 / actual = 1 if stock in training set / pct = 100 or 0
    - pareto: 1 / actual = 1 if sector_balance applied / pct = 100 or 0
    - kondratiev: (implicit; same as first_principle)
    """
    pass
```

### 7.4 Prediction Layer Hook(prediction_engine)

```python
def write_prediction_completeness(stock_id, run_id, universe_snapshot_id):
    """
    對每支 stock 寫入 1-3 records:
    - first_principle: 1 / actual = 1 if prediction generated / pct = 100 or 0
    - pareto: 1 / actual = 1 if in top-K rank / pct = 100 or 0
    - kondratiev: (implicit)
    """
    pass
```

---

## 8. Audit Tool 設計: audit_universe_completeness.py

```python
"""
audit_universe_completeness.py v0.1
==================================
Cross-layer × Cross-pillar Universe Completeness Audit Authority

對映憲章 §0.4 數位孿生完整性 + §8 三層職責邊界

主要 checks (≥10):
1. Schema 完整性(3 new tables exist)
2. Foreign key 完整性
3. Materialized view 可 refresh
4. Per pillar × per layer 至少 1 record per snapshot
5. completeness_pct 邊界 [0, 100]
6. Pillar enum 合法(first_principle/pareto/kondratiev)
7. Layer enum 合法(data/feature/model/prediction)
8. Per stock × per (pillar, layer) 唯一(check PK)
9. Universe snapshot 119 stocks coverage 完整
10. 跨層 rollup verdict(per stock 平均 completeness)
11. Trinity dashboard 輸出 119 × 12 矩陣
12. INFO-only(non-blocking)或 WARN(< threshold)
"""
```

---

## 9. Materialized View Refresh 策略

### 9.1 Trigger 時機

- 每次新 `core_universe_snapshot` commit 後(per §6.8 annual rebalance + special override)
- 每次 4 builders 完成寫入後(可選 cascade refresh)
- 手動 `REFRESH MATERIALIZED VIEW universe_completeness_matrix_current` 指令

### 9.2 Refresh Hook

```python
def refresh_completeness_view():
    cur.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY universe_completeness_matrix_current")
```

`CONCURRENTLY` 需要 UNIQUE INDEX(已在 6.4 設計);允許 refresh 不阻塞 reads。

---

## 10. 對既有 Tables 影響評估

| 既有 table | 影響 | 修改範圍 |
|---|---|---|
| All 28 tables | 0(本路徑只新增,不修改) | 無 |
| fetch_log / data_audit_log / feature_store_* / model_* | 0 | 仍為各層 layer authority |
| core_universe_snapshot | 新 FK reference 來自 3 new tables / 0 schema change | 無 |

**結論**: Path C 為 **purely additive**;0 個既有 schema 動到;0 個既有程式被 break。

---

## 11. 治權邊界與既有契約整合

### 11.1 與 §0.4 數位孿生完整性對齊

§0.4 既有「原子性事務 + 高粒度寫入 + 低熵狀態」原則;本路徑將「完整性」首次表格化為 DB record,使 §0.4 從 implicit 變 explicit。

### 11.2 與 §8 三層職責邊界對齊

§8 既有 Feature Store / Model Registry / Prediction Table 三層分離;本路徑:
- 補建 Prediction Table 層(0% gap 治權義務)
- 新增 cross-layer rollup table(不破 layer authority;只 reporting)

### 11.3 與 §6.7.1 dynamic universe 對齊

§6.7.1 dynamic universe size annex 已確認 N 變動;completeness_snapshot 之 per-stock granularity 自動隨 N 適應(無 hardcode);對齊 §14.7-BT closure。

### 11.4 與 §14.7-BR / §14.7-BT 對齊

- §14.7-BR Phase D 已明示 §0.3 5/5 leading indicators 100%;本表將提供 trinity dashboard 視角的 explicit cross-pillar visualization
- §14.7-BT closure 之 119 stocks dynamic universe;本表為 119 × 12 cells 完整性 SSOT

---

## 12. 入憲建議

### 12.1 新節 §11.x 或 §14.7-BU 子節

**建議: 新建 §14.7-BU(per §14.7-X 模式)**

子節內容大綱:
1. 觸發背景(用戶 2026-05-26 提問)
2. 既有 28 tables inventory + 4 layer gap analysis
3. 3 Architecture path 對比 + Path C 推薦
4. 3 new tables + 1 view schema 草案
5. 4 builders audit hooks 設計
6. Materialized view refresh 策略
7. 與既有 §0.4 / §8 / §6.7.1 / §14.7-BR / §14.7-BT 整合
8. Phase A→G roadmap

### 12.2 修訂歷程 第十九輪 entry

```
v6.1.0-patch 第十九輪: §14.7-BU Phase B 入憲 — Cross-Layer × Cross-Pillar Universe Completeness Governance(2026-05-26;Path C hybrid;補 §8 0% prediction layer gap + 新 universe_completeness_snapshot SSOT + 1 materialized view;Phase A 設計研究先行)
```

### 12.3 升版優先級評估(per §0.0-E.6)

- **P0**: 補 §8 prediction layer 0% gap 為治權義務 → **P0 必補**
- **P1**: universe_completeness_snapshot 為 §0.4 數位孿生完整性具體化 → **P1 強烈推薦**
- **P2**: 4 builders audit hooks → **P2 implementation**
- **P3**: audit_universe_completeness.py + materialized view → **P3 quality**

---

## 13. Phase A→G Roadmap

| Phase | 內容 | 時間 | Status |
|---|---|---|---|
| **A** | Design research(本報告) | ~1.5h | ✅ **本 entry** |
| **B** | 入憲 §14.7-BU + 修訂歷程第十九輪 + universe_completeness 治權邊界 | ~30 min | ⏸ next |
| **C** | universe_completeness_schema.py --init(3 new tables + 1 view) | ~30 min | ⏸ next |
| **D** | prediction layer schema 補建(prediction_run + predictions per §8) | ~20 min | ⏸ next |
| **E** | 4 builders 補 audit hooks(fetcher / feature_store / model_trainer / prediction_engine) | ~2-3h | ⏸ cross-session |
| **F** | audit_universe_completeness.py v0.1 + materialized view refresh hook | ~1h | ⏸ cross-session |
| **G** | v6.3.2(或 v6.4.0) milestone tag + handoff v8 | ~30 min | ⏸ closure |

**總計**: ~6-8h(可分跨 2-3 sessions)

---

## 14. Risks / Mitigation

| Risk | 機率 | 影響 | Mitigation |
|---|---|---|---|
| R1: 4 builders 補 hook 工作量超預期 | 中 | 中 | Phase E 拆細;先補 feature_store(已 production-current); prediction_engine 跟進;data/model 最後 |
| R2: Materialized view refresh 失敗 | 低 | 中 | CONCURRENTLY + retry; fallback 為非 CONCURRENTLY |
| R3: pillar/layer enum 漏 cell 標記 | 中 | 低 | Audit tool default verdict = INFO(non-blocking) |
| R4: 既有 audit tables 重複 | 高 | 低 | 由 Path C 設計顯式區分:既有 tables = layer authority 寫;新表 = SSOT 寫;不互改 |
| R5: 寫入紀律破壞(builder 忘寫) | 中 | 中 | 設計 `audit_self_completeness_write()` 為各 builder 強制 step;non-write → preflight FAIL |
| R6: 跨 session 跨機 schema drift | 中 | 高 | universe_completeness_schema.py 為 SSOT;init 強制 |

---

## 15. 結論

本 design research 提案 **Path C(hybrid)** 為跨層 × 跨基柱 universe completeness governance 之最佳平衡:

- ✅ **補 §8 治權義務**(prediction layer 0% gap → 100% with prediction_run + predictions)
- ✅ **新 SSOT**(universe_completeness_snapshot 為 trinity × layer 唯一查詢入口)
- ✅ **既有 audit 完全不變**(layer authority 紀律維持)
- ✅ **對齊 §0.4 / §8 / §6.7.1 / §14.7-BR / §14.7-BT**
- ✅ **3 new tables + 1 view**(符合 §一 #3 最小邊界)
- ✅ **Phase A→G 拆解** 跨 2-3 sessions(可控)

**下一步**(待用戶授權):
1. Phase B 入憲(~30 min;本 session 可推)
2. Phase C-D schema 建立(~50 min;本 session 可推)
3. Phase E 跨 session(builders 補 hooks)
4. Phase F-G 跨 session(audit tool + milestone tag)

---

**設計研究作者**: Claude
**Status**: ✅ Phase A complete / pre-charter / non-destructive / 不動 DB 不動 code
**Cross-References**:
- 對應憲章 §0.4 / §8 / §6.7.1 / §14.7-BT / §14.7-BR
- 對應 evidence: `reports/core_universe_audit_20260526_1603.md` / `reports/feature_store_v08_implementation_audit_20260526.md`
- 對應 handoff: `reports/session_handoff_20260526_v7.md`
- 對應 user query: 2026-05-26「需要另外建立 table 來進行儲存嗎?」
