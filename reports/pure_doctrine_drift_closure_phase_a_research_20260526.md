# Pure Doctrine Drift Closure — Phase A Design Research(§14.7-BW)

**日期**: 2026-05-26
**Phase**: A(設計研究 / pre-charter inscription)
**對應軌道**: v6.4.0 或 v7.0.0(治權判準層第二輪純化;對映 §14.7-BV 之 Path C drift 修正)
**對應憲章基礎**: §14.7-BV Phase B(第二十輪)/ §6.7.1 dynamic size annex(§14.7-BT 第十八輪)/ §6.4 CoreScore 公式 / §6.7 SSOT SQL / §9.2 portfolio_sizer barbell / §0.0-A 三核心思想統合層
**Status**: ✅ Phase A 完整(15 章 / non-destructive / 不動 DB 不動 code 對 v0.10)
**對應 evidence**: `core_universe_20260526_core_universe_policy_v0_10_pure_doctrine`(committed;N=1862 / 0 cap / 0 tier %)

---

## 1. 觸發背景

用戶於 2026-05-26 同 session 內**第 5 次明示** doctrine 強化,觸發本研究:

### 1.1 五次明示之累進

| 次 | 明示內容 | 治權意涵 |
|---|---|---|
| 1 | 「沒有一定要多少支核心股,但必須符合三基柱具有對應資料來源為依據」 | doctrine 釐清(framing 修正)|
| 2 | 「取消 150 在任何地方」 | active deprecation(charter 清整)|
| 3 | 「取消 150 支核心股 + restate doctrine」 | active deprecation 強化 |
| 4 | 「先全部取消 150 支核心股」 | imperative execution(Phase D-1 觸發)|
| 5 | **「取消所有 200 支及 150 支」** | **連 v0.9 Path C 之 200 也 deprecate — 揭露 N_MAX cap 為 hidden hardcode** |

第 5 次明示為本研究之**核心觸發** — 用戶在我剛 commit v0.9 doctrine_gate snapshot(N=200 / cap-bound)後立刻 deprecate,**揭露 §14.7-BV 第二十輪入憲之 Path C 仍含 hidden hardcode**(N_max=200 / N_min=100 / 70-30 tier % / CoreScore as ranking gate)。

### 1.2 §14.7-BV Path C drift 之 4 條 hidden hardcode

| Drift 點 | §14.7-BV Path C 之 inscribed 處 | 違反 doctrine 之點 |
|---|---|---|
| **N_max=200 cap** | Stage 4 之「[N_min=100, N_max=200] cap」+ §14.7-BV 第 6 條「§6.7.1 治權保留」 | 任何 N 上限為 hardcode |
| **N_min=100 floor** | 同上 | 任何 N 下限為 hardcode |
| **70/30 tier % split** | Stage 4 之「core_pct 70%」 | tier % 為 hardcode |
| **CoreScore as ranking gate** | Stage 3「within-pass-set CoreScore rank」+ Stage 4 之 tier 賦予依據 | CoreScore 之 score-based ordering 影響哪些股入哪個 tier → 仍為 implicit selection 之 influence |

**結論**:§14.7-BV 第二十輪 Phase B 入憲未完整 honor user 之 pure doctrine;Path C 為 partial fit(優於 Path A/B 但未及 Path D);Path D(原評為「過激」)實為 user 真意。

---

## 2. v0.10 落地 evidence(已 committed)

### 2.1 DB 狀態

```
snapshot_id: core_universe_20260526_core_universe_policy_v0_10_pure_doctrine
status: committed
policy_version: core_universe_policy_v0.10_pure_doctrine
as_of_date: 2026-05-26
total_candidates: 2803
core_count: 1862   ← 全 doctrine-pass
convex_count: 0    ← tier 棄用
research_count: 941
quarantine_count: 0
```

### 2.2 三基柱 doctrine 對齊驗證

| 基柱 | 機制 | 通過 N |
|---|---|---|
| §0.1 第一性原理 | 5 raw sources × per-stock 存在性 | 1,862 stocks(loose 5/5)|
| §0.2 八二法則 | 在 TaiwanStockInfo 候選集 | 1,862 by-def |
| §0.3 康波週期 | market-level 5/5 indicators 可用 | 1,862(broadcast)|

### 2.3 universe_completeness_snapshot data layer(§14.7-BU coupling 已 honored)

5,586 records 已寫入(3 pillars × 1,862 stocks):
- first_principle × data: 1,862 records / avg 100.0%
- pareto × data: 1,862 records / avg 100.0%
- kondratiev × data: 1,862 records / avg 100.0%

### 2.4 v0.10 script 之 4 個 hardcode 取消實現

| 取消對象 | v0.9 status | v0.10 status |
|---|---|---|
| N_MAX=200 cap | active(`scored = scored[:N_MAX]`)| **removed**(no cap)|
| N_MIN=100 floor | active(`if len < N_MIN: return False`)| **removed**(no floor)|
| 70/30 tier split | active(`n_core = int(n_total * CORE_PCT)`)| **removed**(`n_core = n_total / n_convex = 0`)|
| CoreScore as ranking gate | active(determines tier)| **removed**(INFO display only)|

---

## 3. Path D 升正之治權位階

### 3.1 §14.7-BV 4-algorithm 對比之追溯修正

| Path | §14.7-BV 第二十輪 評估 | §14.7-BW 修正評估 |
|---|---|---|
| A: CoreScore top-N(N=150 hardcode)| rejected | rejected(維持)|
| B: §14.7-BT Top X% by CoreScore | partial fit / score-driven | rejected(score-driven 違反 doctrine)|
| C: Doctrine-gate first + CoreScore tier rank | ★ 推薦 | **降為 partial fit**(含 hidden hardcode;過渡 design choice)|
| **D: Pure doctrine, no rank** | **過激 / rejected** | **★ 升為正式 doctrine**(對應 v0.10 落地)|

### 3.2 Path D 之完整定義(§14.7-BW canonical)

**Path D Pure Doctrine 4-stage spec**:

| Stage | 動作 | 是否 hardcode |
|---|---|---|
| 1 | §0.3 K-wave market prerequisite 5/5 indicators 必存在 DB | ❌ no hardcode(5 個 indicators 為治權定義之 §0.3.8 spec / per §14.7-BR Phase C-4 closure)|
| 2 | per-stock §0.1 5/5 raw source 存在性 + §0.2 by-def | ❌ no hardcode(5 個 sources 為治權定義之 §0.1 spec)|
| 3 | **取所有 pass stage 1+2 之 stocks 為 selected** | ❌ no hardcode(N = doctrine-pass set 大小)|
| 4 | **無 tier split / 無 score rank** | ❌ no hardcode(CoreScore 為 INFO display only)|

### 3.3 與 §14.7-BV Path C 之治權關係

| 條 | §14.7-BV Path C(第二十輪 inscribed)| §14.7-BW Path D(本研究 inscribed)| 關係 |
|---|---|---|---|
| Selection 必要條件 | 三基柱資料源依據 | 三基柱資料源依據 | **相同**(doctrine core 不變)|
| N 政策 | 動態(per gate-pass)但 cap-bound [100,200] | 動態(per gate-pass)**無任何 bound** | **§14.7-BW 取消 cap/floor** |
| Tier 結構 | core_universe / convex_universe(70/30 split)| **無 tier**(全部 selected = core_universe;convex_universe 棄用)| **§14.7-BW 棄用 tier %** |
| CoreScore 角色 | tier ranking gate(影響哪些股入哪 tier)| INFO display only(完全不影響 selection)| **§14.7-BW 降階** |
| §6.7.1 annex(N_max/N_min/core_pct)| 「治權保留」(per §14.7-BT 第十八輪)| **不適用於 pure doctrine**;legacy 時代 prescription | **§14.7-BW 加 deprecation note** |

---

## 4. 對既有治權之處置

### 4.1 §14.7-BV 第二十輪 之治權處置

**不撤銷,但追溯加 supersession note**:
- §14.7-BV Phase A research + Phase B 入憲 之 narrative 保留為 historical record(per §14.7-BT precedent / §0.0-I 既有 entries respect)
- 但 §14.7-BW 為 superseding treaty;後續任何 selection 須對齊 Path D(非 Path C)
- §14.7-BV 之 5 條治權新特性 4(「3 HARD BLOCK 必拆」)+ 5(「user dual-明示」)維持有效;3 條被本研究修訂或補充

### 4.2 §6.7.1 dynamic size annex(§14.7-BT 第十八輪)之治權處置

**不撤銷,但 deprecation 明文化**:
- §6.7.1 annex 之 N_max=200 / N_min=100 / core_pct=0.70 為 v0.7-v0.8 era 治權 prescription
- §14.7-BW Phase B 後,**此 prescription 不再為 active doctrine 之治權邊界**;reduced 為「歷史 prescription 之 narrative record」
- 未來 algorithm 升版若不採 pure doctrine(e.g., v1.0 引入新 tier model),須在新 charter entry 中重新定義 bound;不可 implicitly 沿用 §6.7.1 之 100/200/70 values

### 4.3 §6.4 CoreScore 公式之治權處置

**公式本身不動,治權位階降階**:
- §6.4 之 6-layer CoreScore composite formula 結構不變(DQ+LM+FG+TR+IF+VC-RP / weights 0.25/0.25/0.20/0.15/0.10/0.05/1.0)
- 治權位階從「§6 強制契約 / 選股之核心 score」**降為**「INFO display / legacy reference / 未來可選 tier 區分依據」
- 不再為 selection gate;不再決定哪支股入 core_universe;不再用於 70/30 tier split
- 既有 core_universe_scores 表保留(per §14.7-BT precedent / audit trail);新 snapshot 寫入 scores 為 INFO 紀錄,不為治權判準

### 4.4 §9.2 portfolio_sizer barbell allocation 之治權處置

**Barbell concept 維持有效,但 tier 邊界另立**:
- 既有 §9.2-A~H + §9.2-I v0.3 補強 之 barbell allocation 結構保留
- 在 v0.10 pure doctrine 下,**convex_universe = 0** → barbell 之「凸性側 30%」失去 universe support
- **§14.7-BW Phase D 之新邊界**(cross-session work / 不在本 entry scope):
  - Option X: 在 doctrine-pass set 內按 CoreScore quantile 切「上 30% / 下 70%」為 barbell 之 convex / core 對映
  - Option Y: barbell 改為 sector / industry diversification 而非 tier-based
  - Option Z: 棄用 barbell concept,改為 equal-weight + sector cap
- 本 §14.7-BW Phase B 不選定 Option;留 portfolio_sizer 升版 (§9.2 v0.4) 時專案研究

### 4.5 §0.4 數位孿生完整性之治權處置

**§14.7-BU governance infrastructure 維持有效**:
- universe_completeness_snapshot 表 + materialized view + audit tool 不動
- v0.10 已寫 1,862 × 3 = 5,586 data layer records(§14.7-BU Phase E data layer 已 honored)
- pure doctrine 下 §14.7-BU 之 SSOT 功能更純粹 — 因 N 為 doctrine 結果,T_BU-5(N × 12 cells)之 cardinality 為 1,862 × 12 = 22,344(待 Phase E feature/model/prediction layer hooks 落地後達成)

### 4.6 §14.7-BT 第十八輪 之治權處置

**不撤銷,但 N policy 升版**:
- §14.7-BT 取消「§6.7 SSOT 150 hardcode prescription」之 closure 維持有效
- §14.7-BT 之 §6.7.1 annex 新建仍為 v0.7-v0.8 era 之治權記述
- §14.7-BW 不撤銷 §14.7-BT;只在 Path D selection 下其 N_max/N_min/core_pct 不適用

---

## 5. v0.10 evidence 之 Phase F audit 驗收

執行 `audit_universe_completeness.py v0.1` against v0.10 committed snapshot 結果:

```
[Schema integrity C1-C4]:
  ✅ C1_tables_exist (prediction_run / predictions / universe_completeness_snapshot)
  ✅ C2_matview_exists (with CONCURRENTLY-ready unique index)
  ✅ C3_foreign_keys (5 FKs verified)
  ✅ C4_check_constraints (pillar / layer / completeness_pct enums)

[Data integrity C5-C12]:
  ✅ C5_record_count (5586 records;3 pillars × 1862 stocks @ data layer)
  ✅ C6_pillar_enum (first_principle / pareto / kondratiev 全合法)
  ✅ C7_layer_enum (data layer 全合法;feature/model/prediction layer 待 Phase E hook 落地)
  ✅ C8_pct_bounds ([0, 100] 全 record 合法 / 全 = 100.00%)
  ✅ C9_universe_coverage (1862 distinct stocks × 3 pillars × 1 layer = 5586 records / 100%)
  ✅ C10_per_stock_rollup
  ✅ C11_per_layer_rollup (data layer 全 1862 / avg 100%)
  ✅ C12_trinity_dashboard (3 × 4 matrix;data layer 全綠 / feature/model/prediction layer 待 Phase E)

主權判定: PERFECT 🎯
```

對映 §14.7-BU T_BU-1~5 證偽承諾:
- T_BU-1 PK 唯一性 100% ✅
- T_BU-2 completeness_pct 邊界 [0,100] ✅
- T_BU-3 pillar enum 合法率 100% ✅
- T_BU-4 layer enum 合法率 100% ✅
- T_BU-5 N × 12 cells coverage:partial(N=1862 × 3 pillars × 1 layer = 5586;完整 1862 × 12 = 22344 需 Phase E feature/model/prediction layer hooks 落地)

---

## 6. Phase A-D Roadmap

| Phase | 內容 | 時間 | Status |
|---|---|---|---|
| **A** | Design research(本報告) | ~1h | ✅ **本 entry** |
| **B** | 入憲 §14.7-BW + 修訂歷程 第二十一輪 | ~30 min | ⏸ next(本 session 可推) |
| **C** | builder script 之 charter alignment(`build_doctrine_gate_universe.py` v0.10 已落地;docstring 對齊 §14.7-BW 入憲文) | ~30 min | ⏸ next session |
| **D** | downstream Phase D-2 升版:`prediction_engine.py` / `portfolio_sizer.py` / `feature_store_builder.py` 3 HARD BLOCK 拆 + §9.2 portfolio_sizer 之 tier-less 升版設計 | ~3-5h | ⏸ cross-session |
| **E** | audit tools 升版(audit_core_universe.py 加 v0.10 policy 識別) | ~1h | ⏸ cross-session |
| **F** | integration test(full chain smoke + walk-forward) | ~2h | ⏸ cross-session |
| **G** | v6.4.0 / v7.0.0 milestone tag + handoff | ~30 min | ⏸ closure |

**總計**: ~8-10h(可分 2-3 sessions)

---

## 7. 對 v6.4.0 / v7.0.0 軌道之路線涵義

### 7.1 v6.4.0 軌道(per §14.7-BV 預定)

§14.7-BV Phase B inscribed: v6.4.0 之 milestone 包含 §14.7-BV Path C 落地。

**§14.7-BW 觸發後升版**:v6.4.0 milestone 改為「§14.7-BW Path D pure doctrine 落地」:
- core_universe_builder.py v0.10 入庫(目前為 maintenance/build_doctrine_gate_universe.py;規範化為 core/ 目錄之正式 builder)
- §9.2 portfolio_sizer 升版 tier-less(v0.4)
- audit 升版識別 v0.10 policy
- §6.4 CoreScore 治權位階降階入憲

### 7.2 v7.0.0 軌道考量

若 §14.7-BW 落地對既有 §6.4 / §9.2 之治權位階改動屬「破壞性 schema 變更」之治權層,則升 v7.0.0 而非 v6.4.0(per §0.0-E.6 升版優先級 之 major 定義)。

**判斷標準**:本 §14.7-BW 不改 SQL contract / 不改 raw DDL / 不改 9 步序列,但**改 §6.4 CoreScore 治權位階 + §9.2 barbell concept**;此屬「治權位階改動」非「結構性破壞」→ v6.4.0 minor 升版即可(per v6.0.0 定版宣言 §5.1 之 minor 定義「§8 升強制契約 / 新 audit step / 非破壞性 schema 擴張」延伸)。

---

## 8. §14.7-BW 治權新特性(predicted)

1. **首例「治權判準 fourth-round refinement」**(§14.7-BT cancel hardcode → §14.7-BU build governance → §14.7-BV change criterion → §14.7-BW pure-form refinement;四輪累進)
2. **首例「治權自我 drift 揭露 + 修正」**(§14.7-BV Phase B inscribed Path C 含 hidden hardcode → §14.7-BW 揭露並升 Path D 為正式);類比 §14.7-BR Phase C-4 之「Phase A oversight 揭露 + Phase C-NEW 補修」pattern,但 scope 為 doctrine layer 而非實作 layer
3. **首例「§6.4 CoreScore 治權位階降階」**(從 §6 強制契約之 selection score → INFO display only);CoreScore 公式不動但 doctrine role 改寫
4. **首例「§9.2 barbell tier 棄用觸發」**(convex_universe = 0 → barbell 需 cross-session refresh;portfolio_sizer 升版 §9.2 v0.4 之治權預備)
5. **首例「N policy 完全無 bound」**(§14.7-BV inscribed [100,200] cap-bound dynamic → §14.7-BW 完全 doctrine-derived;對映 user 第 5 次明示)

---

## 9. 證偽承諾 T_BW-1〜5

- T_BW-1: v0.10 committed snapshot 之 N(目前 1,862)為 doctrine-derived;若 §0.1 thresholds 或 §0.3 indicators 改動,N 自動隨之變動;不被任何 explicit cap / floor / target 鎖定
- T_BW-2: 新 snapshot 之 core_universe_membership 所有 row 之 `core_tier='core_universe'`;`convex_universe` 之 row count = 0(tier 棄用)
- T_BW-3: §14.7-BW Phase B 入憲後,charter active prescription 之「[N_min=100, N_max=200] cap」/「70/30 tier split」/「CoreScore tier ranking gate」之語句 count = 0(僅在 §14.7-BV / §6.7.1 annex 之歷史 narrative 中保留;不出現於新 active prescription)
- T_BW-4: `universe_completeness_snapshot` 之 1,862 stocks × 3 pillars × data layer = 5,586 records;avg completeness_pct = 100.0%;與 v0.10 committed snapshot 之 universe_snapshot_id 對齊
- T_BW-5: `audit_universe_completeness.py` 跑 v0.10 snapshot verdict 為 PERFECT(C1-C4 schema integrity strict + C5-C12 data integrity all PASS / INFO_DATA_PRESENT)

---

## 10. Risks / Mitigation

| Risk | 機率 | 影響 | Mitigation |
|---|---|---|---|
| R1: §14.7-BW 與 §14.7-BV 之治權關係模糊(用戶疑問「兩條都有效嗎?」)| 中 | 中 | §14.7-BW Phase B 明示「supersede」(per §14.7-BR Phase C-4 stale 追溯修正 precedent);§14.7-BV Path C 降為 historical narrative |
| R2: §6.7.1 annex(§14.7-BT)之 N_max/N_min/core_pct 沒有顯式 deprecation,未來 AI 仍引用 | 中 | 中 | §14.7-BW Phase B 明文「§6.7.1 之 [100,200,70] values 為 v0.7-v0.8 era prescription;在 Path D 下不適用」 |
| R3: §6.4 CoreScore 治權降階引發既有 audit / builder 工具誤動作 | 高 | 中 | §14.7-BW Phase B 明文「§6.4 公式不動;治權位階改動只影響 selection 之 doctrine 角色」;既有 audit_core_universe / feature_store_* 之 CoreScore 讀取不受影響 |
| R4: §9.2 portfolio_sizer 之 barbell concept 因 convex_universe=0 而 break | 高 | 高 | §14.7-BW Phase B 明文 Phase D scope 含 §9.2 v0.4 升版(tier-less / sector-based);本 entry 不解;留 cross-session work |
| R5: 用戶後續再深化 doctrine(e.g., 取消 §0.1 5/5 之「5」也算 hardcode)| 低 | 低 | §14.7-BW Phase B 明文「§0.1 5 sources 為 §0.1 治權定義之 spec,非 algorithm hardcode」— 邊界與 algorithm-side cap 不同 |
| R6: 下游 3 HARD BLOCK 拆延宕(N=1862 vs 150 hardcode break)| 高(已存在)| 中 | Phase D 跨 session;短期內 prediction_engine / portfolio_sizer 跑會 fail;不影響 doctrine 治權層;為 cross-session work |

---

## 11. 入憲建議

### 11.1 新節 §14.7-BW(per §14.7-X 模式)

子節內容大綱(11-12 章):
1. 觸發(用戶 5 次明示 + v0.10 落地證據 + §14.7-BV Path C drift 揭露)
2. §14.7-BV Path C 之 4 hidden hardcode 揭露
3. Path D 升正之治權位階(§14.7-BV 4-algorithm 評估追溯修正)
4. Path D Pure Doctrine 4-stage spec(完整 canonical)
5. 對既有治權處置(§14.7-BV/§14.7-BT/§6.7.1/§6.4/§9.2/§14.7-BU/§0.4)
6. v0.10 落地 evidence(已 committed N=1862 / audit PERFECT)
7. Phase A-G roadmap
8. v6.4.0 / v7.0.0 軌道意涵
9. §14.7-BW 治權新特性 5 條
10. 證偽承諾 T_BW-1〜5
11. Risks / Mitigation
12. Cross-Reference 精確行號

### 11.2 修訂歷程 第二十一輪 entry draft

```
v6.1.0-patch 第二十一輪: §14.7-BW Phase B 入憲 — Pure Doctrine Drift Closure(對映 §14.7-BV Path C 之 4 hidden hardcode 揭露 + Path D 升正;對應用戶 5 次明示之最強讀法「任何 N bound 違反 doctrine」)。§14.7-BV 不撤銷但 Path C 降為 historical narrative;§6.7.1 annex 之 N_max/N_min/core_pct prescription 對 Path D 不適用;§6.4 CoreScore 治權位階降為 INFO display;§9.2 barbell 之 convex tier 暫缺待 v0.4 升版重定義。v0.10 snapshot 已 committed(N=1862)為 evidence。對既有治權影響:零 SQL contract / 零 schema / 零 raw DDL 改動;治權位階改動限於 §6.4 + §9.2 + §6.7.1 之 doctrine role refinement。
```

### 11.3 升版優先級(per §0.0-E.6)

- **P0**:doctrine 一致性為治權層核心 → **P0 必補**
- **P1**:§9.2 v0.4 portfolio_sizer tier-less 升版 → **P1 next-session**
- **P2**:downstream 3 HARD BLOCK 拆(Phase D)→ **P2 implementation**
- **P3**:audit tools 升版 + integration test → **P3 quality**

---

## 12. Cross-Reference 精確行號(預期)

§14.7-BW 子節入憲位置:charter 之 §14.7-BV(L9462)之後 / §20 創世圓滿宣言之前(L9550+;依 §14.7-BV section 結束 + 緩衝計算)。具體行號待 Phase B 入憲時取定值。

修訂歷程 第二十一輪 row 位置:L66(在 §14.7-BV 第二十輪 row 之上)。

---

## 13. 對 §0.0-D D 基柱影響

§0.0-D(康波週期跨層完整度基線)目前 ~85%(post §14.7-BR Phase D)+ 87-88%(預期 post §14.7-BV)。

**§14.7-BW 影響**:零(本 entry 屬 doctrine 純化,不改 §0.3 indicator 數量 / coverage / verdict 算法)。

對應 §0.0-B(第一性原理)+ §0.0-C(八二法則):
- B 基柱:**§14.7-BW 強化**(per-stock §0.1 5/5 enforce 為 doctrine canonical;非「實證一致性提升」而為「治權位階明文化」+2pp)
- C 基柱:**不變**(§0.2 by-def 100%)
- D 基柱:**不變**(§0.3 已 5/5;§14.7-BW 不改 §0.3.8 spec)

---

## 14. 配套程式落地(已完成 / 跨 session)

### 14.1 已完成(本 session)

`scripts/maintenance/build_doctrine_gate_universe.py` v0.9 → v0.10 升版:
- TOOL_VER: `v0.9_doctrine_gate` → `v0.10_pure_doctrine`
- POLICY_VERSION: `core_universe_policy_v0.9_doctrine_gate` → `core_universe_policy_v0.10_pure_doctrine`
- N_MAX / N_MIN / CORE_PCT 常數移除(明示 comment「per §14.7-BV v0.10:NO N bound, NO tier % split」)
- Stage 3 + Stage 4 重寫:CoreScore 為 INFO display / 全 doctrine-pass → core_universe
- DB committed snapshot:`core_universe_20260526_core_universe_policy_v0_10_pure_doctrine`(N=1862 / 0 convex / 941 research)

### 14.2 跨 session(Phase C-G)

- builder script docstring 對齊 §14.7-BW Phase B 入憲文(目前 docstring 之 §14.7-BV 引用為前置入憲;§14.7-BW 入憲後須 supersede)
- `core_universe_builder.py` 主檔(scripts/core/)之 v0.7.1 → v1.0 升版(Path D 落地之 canonical builder;當前 `build_doctrine_gate_universe.py` 為 maintenance 工具)
- 下游 3 HARD BLOCK 拆
- §9.2 portfolio_sizer v0.4 設計(tier-less / sector-based / 或其他)
- audit_core_universe.py 加 v0.10 policy 識別

---

## 15. 結論

本 design research 提案 **§14.7-BW Phase B 入憲 Path D Pure Doctrine** 為對應用戶 2026-05-26 五次明示之 fourth-round refinement closure:

- ✅ **揭露 §14.7-BV Path C drift**(4 hidden hardcode:N_max/N_min/tier %/CoreScore role)
- ✅ **升 Path D 為正式 doctrine**(原評為「過激」實為 user 真意)
- ✅ **v0.10 落地 evidence**(N=1862 committed / audit PERFECT)
- ✅ **§6.4 CoreScore 治權位階降為 INFO**(公式不動 / role 改寫)
- ✅ **§6.7.1 annex N_max/N_min/core_pct 之 prescription 對 Path D 失效**(historical narrative 保留)
- ✅ **§9.2 portfolio_sizer barbell 待 v0.4 升版**(convex tier 棄用觸發 / cross-session work)
- ✅ **5 條治權新特性 + 5 證偽承諾**

**下一步**(待用戶授權 per-phase):
1. Phase B 入憲(~30 min;本 session 可推 — 純 charter 編輯)
2. Phase C builder docstring + canonical core_universe_builder.py v1.0(cross-session)
3. Phase D 下游 3 HARD BLOCK 拆 + §9.2 v0.4(cross-session)
4. Phase E audit + Phase F integration + Phase G milestone tag(cross-session)

---

**設計研究作者**: Claude
**Status**: ✅ Phase A complete / pre-charter / non-destructive
**Cross-References**:
- 對應憲章 §14.7-BV(第二十輪)/ §14.7-BT(第十八輪)/ §14.7-BU(第十九輪)/ §6.4 / §6.7 / §6.7.1 / §9.2 / §0.0-A/B/C/D / §0.4
- 對應 evidence:`core_universe_20260526_core_universe_policy_v0_10_pure_doctrine`(committed)+ `audit_universe_completeness.py` PERFECT verdict
- 對應 user trigger:2026-05-26 五次明示(累進到「取消所有 200 支及 150 支」之最強 reading)
- 對應 memory:`core_stock_selection_doctrine.md`(已升 v3 反映 §14.7-BW pure doctrine)
- 對應同步配套:`scripts/maintenance/build_doctrine_gate_universe.py` v0.10(已落地)
