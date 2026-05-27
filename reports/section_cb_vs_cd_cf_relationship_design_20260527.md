# §14.7-CB ↔ §14.7-CD/CF 治權關係 — 設計研究

**對齊**: charter §14.7-CB(Feature Completeness Gate)/ §14.7-CD(Raw Data Completeness Gate)/ §14.7-CF(Unified SSOT)/ §14.7-CG(Native Implementation;Phase C dry-run pending)
**作者**: Codex
**日期**: 2026-05-27
**Trigger**: §14.7-CG v0.13 dry-run 揭露 N=1,583 ⊃ v0.12 N=1,543(+40 stocks 為 raw layer 通過但 feature_values 為空)

---

## §1 問題陳述

§14.7-CG Phase C 落地 `DoctrineNativeGateBuilder` v0.13 並 dry-run 後揭露:

- **v0.13 N=1,583**(11 raw source thresholds × 13 FRED series gate)
- **v0.12 N=1,543**(active committed snapshot)
- **v0.13 ⊃ v0.12 嚴格包含**(0 stocks lost,40 stocks added)
- **+40 stocks 特性**:raw 11 sources 全通過,feature_values 表中 0 features

此違反原 §14.7-CG T_CG-1「v0.13 dry-run N=1,543 + 0 member diff」承諾;deeper analysis 揭露 charter doctrine 之內部一致性需釐清:**§14.7-CB feature gate 是否屬 §14.7-CF Invariant 1 之一部分**?

---

## §2 §14.7-CB / CD / CF 三節 doctrine 對比

### 2.1 三節原文摘要(per charter inscribed)

| 節 | 治權層 | enforcement scope | enforce 之 source |
|---|---|---|---|
| **§14.7-CB**(2026-05-27 v6.4.0 / 第二十六輪)| **feature-level** completeness | 37 spec features 全到位 | `feature_values` 表(builder 衍生)|
| **§14.7-CD**(2026-05-27 v6.4.2 / 第二十八輪)| **source-level** completeness | 11 raw API sources × thresholds | 9 FinMind tables(raw API)|
| **§14.7-CF**(2026-05-27 v6.4.3 / 第三十輪)| **SSOT entry** | 三 invariant 統合 | mapping table 引 §BW/CC/CD/CE(無 §CB)|

### 2.2 §14.7-CD 對 §14.7-CB 之文字描述(charter L10214-10216)

> **§14.7-CD vs §14.7-CB 差異(深度治權純化)**:
> - §14.7-CB:feature-level gate(37/37 features 非 None);但 `roe_ttm` 之 PBR/PER identity fallback 可 mask 缺漏(feature 算得出 ≠ raw source 完整)
> - **§14.7-CD**:**source-level gate**(9 raw tables × thresholds);**廢棄所有 fallback**;raw source 不全則嚴格排除

→ charter 明示 §14.7-CD 為 **§14.7-CB 之嚴格化**,且 §14.7-CD「廢棄所有 fallback」。

### 2.3 §14.7-CD Phase B 對既有治權影響(charter L10240)

> Phase B 對既有治權影響:§14.7-CB/CC 不撤銷,§14.7-CD 為其嚴格化;[...] v0.11 N=1,729(§14.7-CB)升至 v0.12 N=**1,541**(§14.7-CD;-188 / -10.9%)

→ §14.7-CD 明文「§14.7-CB 不撤銷」,但 v0.12 將 §14.7-CD apply 在 §14.7-CB 之上(composite enforcement)。

### 2.4 §14.7-CF Invariant 1 之精確措辭(charter L10354)

> 1. **三基柱 source-existence prerequisite**:核心股挑選之**唯一前置條件**為 §0.1 第一性原理 + §0.2 八二法則 + §0.3 康波週期皆具備對應 raw data source(per §14.7-CD 之 11 source thresholds);任一基柱 source 不到位即剔除

→ Invariant 1 明示「per §14.7-CD 之 11 source thresholds」**為唯一前置條件**。**未引 §14.7-CB**。

### 2.5 §14.7-CF mapping 表(charter L10358-10365)

| 既有節 | 治權 scope | §14.7-CF SSOT 之中對應 invariant |
|---|---|---|
| §14.7-BW(2026-05-26)| N pure doctrine 廢棄 hardcode | Invariant 2 |
| §14.7-CC(2026-05-27)| Source authority 全 API-fetched | Invariant 3 |
| §14.7-CD(2026-05-27)| Raw data completeness gate 11 sources | Invariant 1 |
| §14.7-CE(2026-05-27)| Per-stock empirical proof | Invariant 1+3 之 empirical closure |

→ **§14.7-CB 不在表中**。§14.7-CF 之 SSOT 統合**未涵蓋** §14.7-CB feature gate。

---

## §3 邏輯 inference

### 3.1 §14.7-CB 與 §14.7-CD 之邏輯關係(超集 / 子集 / 獨立)

設 S_CB = {stocks pass 37/37 features in feature_values}, S_CD = {stocks pass 11 raw thresholds}。

**若 feature_store_builder 完美運作(per §14.7-CD 之「廢棄 fallback,真實或 None」)**:
- 每股 11 raw sources 完整 ⟹ 37 features 全可計算 ⟹ S_CD ⊆ S_CB
- 每股 raw sources 缺漏 ⟹ 對應 feature = None ⟹ stock fails §14.7-CB ⟹ S_CB ⊆ S_CD
- 雙向蘊含 ⟹ **S_CB = S_CD**(理論上)

**實證(v0.13 dry-run 揭露)**:
- v0.12 = S_CD ∩ S_CB(via apply_raw_data_completeness_gate on v0.11)= 1,543
- v0.13 = S_CD(raw only)= 1,583
- +40 stocks ∈ S_CD \ S_CB(raw 通過 / feature_values 空)

→ **理論等價但實證有 +40 gap**。Gap 來自 feature_store_builder 對這 40 股**尚未跑過**(feature_values 表為空 = 沒被 build 過),非 feature 算不出來。

### 3.2 +40 stocks 之 root cause

驗證樣本(5/40):1225 / 1340 / 1418 / 1436 / 1442 等(食品 / 紡織業 / 工業電腦相關)

可能原因:
1. feature_store_builder 上次跑時這 40 股未進入 v0.10/v0.11 universe(被早期 gate 排除)
2. v0.10/v0.11 universe 在 build_doctrine_gate_universe Stage 2 之「5-source EXISTS」邏輯有量化盲區(已被 §14.7-CD 嚴格化捕捉)
3. 這些股後續被新 §14.7-CD 接受,但 feature_store_builder 尚未 re-run 於 expanded universe

→ **Gap 為 stale feature_values 之 artifact**,非治權結構性問題。

### 3.3 §14.7-CB vs §14.7-CD 治權位階解讀

**Reading A:§14.7-CD 已嚴格化覆蓋 §14.7-CB(supersede in practice)**
- 依據:§14.7-CD 廢棄所有 fallback,feature 從此「真實或 None」
- 推論:raw 完整 ⟹ feature 完整(若 builder 跑過);§14.7-CB 之 37/37 gate 變成 trivial
- 治權結果:§14.7-CF SSOT 只需 §14.7-CD;§14.7-CB 為歷史 narrative

**Reading B:§14.7-CB / §14.7-CD 為獨立層,§14.7-CF 應補完 mapping**
- 依據:§14.7-CD 明示「§14.7-CB 不撤銷」
- 推論:兩 gate 為 AND-composition(stock 必同時通過 raw 11 + feature 37/37)
- 治權結果:v0.13 native 應 enforce 兩 gate;§14.7-CF mapping 應補 §14.7-CB → Invariant 1.2(feature-level extension)

**Reading C:§14.7-CD 為 raw-layer prerequisite,§14.7-CB 為 downstream feature-layer 要求(§8 範圍)**
- 依據:§14.7-CB 之 source 為 `feature_values`(builder 衍生),非 raw API
- 推論:§14.7-CB 之 enforcement 屬於 §10 model_trainer 之 input gate(per §14.7-CD 「真實或 None」cascade)
- 治權結果:v0.13 native 只 enforce raw(§14.7-CD);feature-layer gate 移至 model_trainer

---

## §4 治權裁決推薦

### 4.1 推薦採用 Reading A + C 混合

依據:
1. **charter §14.7-CD 明文「廢棄所有 fallback」**:理論上 raw 完整 ⟹ feature 完整,§14.7-CB 之 37/37 gate 變為「feature_store_builder 跑過 = pass」之冗餘 check
2. **+40 stocks 之 root cause 為 stale feature_values**(非 feature 算不出來;是 builder 對這些股未 re-run)
3. **§14.7-CF 之 mapping 已明示「per §14.7-CD」**:SSOT 治權選 §CD 為 Invariant 1 enforcement,§CB 為 historical artifact

### 4.2 治權閉環調整

**現行(v0.12,§14.7-CB ∧ §14.7-CD composite):**
```
Pipeline:  raw sync → feature_store_builder → §CB feature gate (1,729→1,640)
        → §CD raw gate (1,640→1,543)
治權位階:  §CB-then-§CD 雙 gate
```

**新治權(v0.13 native,§14.7-CD pure):**
```
Pipeline:  raw sync → §CD raw gate (2,803→1,583) ← v0.13 native
        → [downstream] feature_store_builder per §14.7-CD「真實或 None」
        → [downstream] model_trainer drops stocks with None features
治權位階:  §CD primary;§CB 降為 downstream feature-layer 之自然 cascade
```

### 4.3 對 +40 stocks 之處置

- v0.13 commit N=1,583 → +40 stocks 加入 universe
- 同次配套 trigger `feature_store_builder` rebuild on N=1,583 universe → 補建這 40 股之 feature_values
- 預期結果:rebuild 後 N=1,583 之 feature_values 應 ≥ 37 features per stock(per §14.7-CD「真實或 None」)
- 若 rebuild 後仍有 stocks feature_values < 37 → 揭露 feature_store_builder 之 bug(non-§CD source dependency)

### 4.4 charter 配套修訂

1. **§14.7-CG entry 更新**:
   - T_CG-1 改寫:從「N=1,543 + 0 member diff」→「N=1,583 ⊃ v0.12 1,543(+40 = stale feature_values rebuild target)」
   - 加註:Phase D/E commit 前須 trigger feature_store_builder rebuild on N=1,583 → 補建 40 股 features

2. **§14.7-CF mapping 補註**(optional,僅清晰化非結構性):
   - §14.7-CB 為 §14.7-CD 之 feature-layer 自然 cascade(per §14.7-CD「廢棄 fallback;真實或 None」)
   - §14.7-CB 不入 §14.7-CF mapping 為 by-design(SSOT 只 enforce source-level,feature-level 由 model_trainer 自然 drop)

3. **§14.7-CB 治權狀態**:
   - 仍 ACTIVE(不撤銷 / 不 supersede)
   - 但實質 cascade 於 §14.7-CD「真實或 None」+ model_trainer drop;非 universe-selection gate
   - apply_feature_completeness_gate.py 標為 LEGACY(per Phase E)

---

## §5 證偽承諾(Reading A+C 採用後)

- **T_CB-CF-1**:v0.13 commit N=1,583 後,跑 feature_store_builder 重 build on N=1,583 → 預期 ≥ 1,543 stocks 有 ≥ 37 features(對應 v0.12 已 build)
- **T_CB-CF-2**:若 feature_store_builder rebuild 後仍有 stocks features < 37(扣除 §14.7-CD 之 None 容忍)→ 揭露 feature_store_builder bug 須修補
- **T_CB-CF-3**:v0.13 universe(N=1,583)為 §14.7-CF Invariant 1 之 pure raw-layer 實現
- **T_CB-CF-4**:§14.7-CB 之 feature gate(37/37)cascade 至 model_trainer 之 input gate(per §0.0-E.6 priority)
- **T_CB-CF-5**:charter §14.7-CB / CD / CE / CF / CG / 本研究報告 inscribed 構成完整 source-completeness ↔ feature-completeness 治權閉環

---

## §6 結論與下一步

### 結論

1. **v0.13 N=1,583 為正確 §14.7-CF 治權實現**(per Reading A+C)
2. **+40 stocks 為 stale feature_values artifact**,非 doctrine 違規
3. **§14.7-CB 不撤銷**,但實質降為 feature_store_builder 之自然 cascade + model_trainer input gate;不再為 universe-selection gate

### 下一步(Phase D-E 解凍前)

1. **用戶 review 本研究報告**並裁決 Reading A+C 是否正確
2. **若 accept Reading A+C**:
   - Phase D commit v0.13 snapshot N=1,583
   - 同次 trigger feature_store_builder rebuild on 1,583 universe
   - Verify rebuild 後 1,583 stocks feature_values ≥ 37(per T_CB-CF-1)
   - charter §14.7-CG entry 更新(T_CG-1 改寫;Reading A+C inscribed)
   - apply_feature_completeness_gate.py 標 LEGACY
3. **若 user 要求 Reading B**(§CB / §CD AND-composition):
   - v0.13 加 Stage 4 feature_completeness gate → N=1,543
   - charter §14.7-CF mapping 補 §CB → Invariant 1.2

### 已就位之 artifact(本 session)

- `reports/v6_5_0_native_gate_design_research_20260527.md` (Phase A 設計研究)
- `reports/section_cb_vs_cd_cf_relationship_design_20260527.md` (本研究)
- charter §14.7-CG inscribed(將被 T_CG-1 改寫升版)
- `scripts/core/core_universe_builder.py` `DoctrineNativeGateBuilder` v0.13(code 已在 working tree)
- v0.13 dry-run 實證 N=1,583 ⊃ v0.12 1,543(實證閉環)

**狀態**:Phase A/B/C 完成;Phase D-E 暫凍待裁決。
