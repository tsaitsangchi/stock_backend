# Portfolio Sizer v0.3 設計研究報告 — Pareto 集中右尾補強 + ROE-aware weighting + sector count 強化

- **產出日期**: 2026-05-26
- **產出者**: Claude Sonnet 4.7 session(承接 v6.1.18.2 cross-machine handoff)
- **觸發**: 用戶 2026-05-26「CoreScore BY DESIGN 平均化（需 portfolio_sizer 補救）」+ 「v0.3 升版對齊 v6.1.18+(ROE/v0.7/V 73%)」
- **scope**: 治標 — 純 sizer 層 ROE-aware weighting + sector count 5→3 + 對齊 v6.1.18+ 治權；root cause(upstream prediction 100% 半導體)需 §10 model_trainer 治本(另案)
- **對映**: §9.2 v0.2 → v0.3 補強 + §0.2-A 八二法則治權分層 + §14.7-AA Part C 揭露 + §14.7-BI ROE 解鎖
- **§0.0-G Level 1 紀律**: 本研究為治權先行步驟;入憲(§9.2-I 或 §14.7-BL)後始可程式落地

---

## 一、觸發背景

### 1.1 用戶 anchor 之 echo 鏈

用戶 2026-05-26 上午 echo 三次 anchor:
1. 「先看資料庫內的資料運用在核心股的挑選時在八二法則思想是否有資料依據」
2. 「⚠️ CoreScore 公式 BY DESIGN 平均化（需 portfolio_sizer 補救）」
3. 選甲「portfolio_sizer v0.2 → v0.3 升版對齊 v6.1.18+(ROE/v0.7/V 73%)」

第二個 echo 揭露**結構性治權設計問題** — CoreScore 平均化是 charter 已預知之 L1 selection 邊界,需 L3 sizing 補救。

### 1.2 治權層三分權責(charter 已明文)

| 層 | 章節 | 治權職責 | 對 Pareto 處置 |
|---|---|---|---|
| **L1 Universe**(builder) | §6.4 + §6.7 + §0.2-A | selection-oriented(取 top 5.42%) | **必須平均化**(6 維 → 1 score)|
| **L2 Tactical**(prediction)| §9.1 | 預測 + rank + IC | 不直接 weighting,只 rank |
| **L3 Sizing**(portfolio_sizer) | §9.2 + §0.2-A | **weighting-oriented**(執行 Pareto) | **必須集中右尾**(12 FAIL gate)|

L1 必須平均化 BY DESIGN — 反證若用 Pareto-weighted 公式,TSMC raw Trading_money 占 6.36% → core_score 變 ~95 → universe 變「TSMC + 7 半導體 + 113 邊緣股」。所以 L1 取「全面合格 candidate」是治權必要。

---

## 二、v0.1 / v0.2 實證狀況

### 2.1 v0.1 proposal 100% 半導體實證(2026-05-19)

`reports/portfolio_allocation_proposal_2025-04-25.md`:

```
攻擊端 20% / CASH 80% / 配置 6 stocks
6/6 = 100% 半導體業
20 個 long candidates 全是半導體業
Rank 7-20 (14 支)attack_cap_exhausted

配置明細:
6643 M31 / 6237 驊訊 / 3443 創意 / 3661 世芯-KY / 4971 IET-KY / 3374 精材
全部 3-5% cap 配置(FLAT;rank 1 與 rank 6 weight 相同)
```

### 2.2 v0.2 audit 97.5% 合規(2026-05-20)

`reports/portfolio_sizer_v02_implementation_audit_20260520.md`:

- v0.2 對齊度 **97.5% (78/80)**
- 已實作:ConstitutionalViolationError 類別 + 4 audit hooks 獨立 + G11 + G12 (single_sector_count_max=5)
- 14 unit tests 全通過
- CONSTITUTION_VER = v6.0.0
- TOOL_VER = v0.2

### 2.3 §14.7-AA Part C 已揭露(2026-05-20)

> 「首份 allocation proposal 揭露 100% 半導體集中：所有 top 20 long 訊號皆來自半導體業，使 sector_weight_max=0.40 失去實質約束力 — sector cap 字面合規但『槓鈴跨域分散』精神未實現」

**修正版 4 子條件 AND 之 #1a**: **stock-specific sector_exposure**(這是 model_trainer 層之事,非 sizer 層)

---

## 三、Root Cause 限制聲明(治權誠實)

### 3.1 Root cause 不在 sizer

`portfolio_sizer.py` 是 **cap-based control 之 downstream consumer**:
- 只能在 upstream prediction 給的 long candidates 內挑
- 若 prediction 100% 半導體 → sizer 無法選非半導體
- v0.2 加 G12 single_sector_count_max=5 → 即便 6 支仍 100% 半導體

### 3.2 v0.3 之範圍誠實聲明

**v0.3 只能 part-fix**:

| 問題 | v0.3 能解 | v0.3 無法解 |
|---|---|---|
| Pareto 集中右尾(高 ROE 股 over-weight) | ✅ ROE-weighted | — |
| Prediction value-weighted(差距大者 over-weight) | ✅ value-weighted | — |
| sector 多元化(候選內 sector count 強化) | ✅ count 5→3 | ❌ 若候選全同 sector,無解 |
| 100% 單一 sector 集中 | ❌ candidate pool 限制 | **需 §10 model_trainer 治本** |
| 跨 sector 槓鈴跨域分散 | ❌ 同 candidate pool 限制 | **需 §10 治本** |

### 3.3 §10 model_trainer 治本另案

- 在 model_trainer 層加 sector-balanced loss / penalty
- training time enforce sector exposure cap
- walk-forward IC > 0 + sharpe gate
- v6.2.0 軌道(預估 2-3 週)

**v0.3 = 治標 / §10 = 治本** — 兩者並行不衝突;v0.3 設計研究中明示此限制以避免誤解。

---

## 四、v0.3 設計目標

### 4.1 治標 4 個 enhancements

1. **ROE-aware weighting**(對映 §14.7-BI ROE 解鎖,V 73%):高 ROE 股 over-weight,低 ROE 股 under-weight
2. **Prediction value-weighted**(對映 §9.1 prediction layer):差距大者 over-weight,非僅 rank-based
3. **Sector count 強化**(對映 §14.7-AA Part C):single_sector_count_max 5 → 3,強制 diversify
4. **對齊 v6.1.18+ 治權**(對映 §14.7-BI / §14.7-BJ / §14.7-BK):DEFAULT_PREDICTION_POLICY_VERSION 升至 v0.2

### 4.2 治權邊界嚴守

- **不改** §9.2-A~H 既有 12 FAIL gate 公式(G1-G12)
- **不改** §0.2-A 7 禁令
- **不改** 攻擊端 20% / 防護端 80% / 單股 5% / convex 3% 既有 cap
- 僅**新增** G13/G14/G15(v0.3 補強)+ apply_policy 內 ROE-aware weighting

---

## 五、DEFAULT_POLICY 升版細節

### 5.1 v0.2 → v0.3 變更表

| Parameter | v0.2 | v0.3 | 動機 |
|---|---|---|---|
| `single_sector_count_max` | 5 | **3** | §14.7-AA Part C 強化(降低同 sector 集中)|
| `roe_weight_alpha` | — | **0.5** | 新增 — ROE-weighted Pareto 強度 |
| `prediction_value_weight_beta` | — | **0.3** | 新增 — raw value-weighted 強度 |
| `roe_multiplier_clamp_min` | — | **0.5** | 新增 — 避免 ROE 極端低 weight 過小 |
| `roe_multiplier_clamp_max` | — | **1.5** | 新增 — 避免 ROE 極端高 weight 過大 |
| `DEFAULT_PREDICTION_POLICY_VERSION` | prediction_policy_v0.1 | **prediction_policy_v0.2** | 對齊 v6.1.18+ |
| `DEFAULT_SIZING_POLICY_VERSION` | sizing_policy_v0.2 | **sizing_policy_v0.3** | 升版標記 |
| 其他 (attack_max / safety_min / single_max / convex_max / sector_max) | 不變 | 不變 | §9.2-A~H 治權邊界 |

---

## 六、ROE-aware weighting 公式設計

### 6.1 公式定義

```
為每個 long candidate:
1. 取 ROE(從 v0.7 snapshot 的 fg_roe;若 None 設 sector median)
2. 標準化:roe_z = (roe - roe_mean) / roe_std  # over long candidates
3. ROE multiplier:roe_mult = 1 + roe_weight_alpha × roe_z
4. Clamp:roe_mult = max(0.5, min(1.5, roe_mult))
5. 計算 proposed weight:proposed = base_cap × roe_mult
6. 仍受 G5 cap:proposed = min(proposed, single_stock_weight_max)
```

### 6.2 數值例

假設 6 個 long candidates 之 ROE:
```
TSMC: 32.72%   z=+1.5  → mult=1.50(clamp)  → proposed=5%×1.5=7.5% → cap to 5%
台達電: 26.62% z=+0.8  → mult=1.40         → proposed=5%×1.4=7.0% → cap to 5%
聯發科: 25.87% z=+0.7  → mult=1.35         → proposed=5%×1.35=6.75% → cap to 5%
鴻海: 12.71%   z=-0.5  → mult=0.75         → proposed=5%×0.75=3.75%
中華電: 10.33% z=-0.7  → mult=0.65         → proposed=5%×0.65=3.25%
1303 南亞: -4%  z=-1.8  → mult=0.50(clamp) → proposed=5%×0.5=2.5%
```

**effect**: 高 ROE 股仍 cap 在 5%(已是 G5 上限),低 ROE 股自動降至 2.5-3.75%。總配置：5+5+5+3.75+3.25+2.5 = 24.5% > 20% attack_cap → 仍會觸發 attack_total cap;rank 順序 ROE-aware 後重新計算。

### 6.3 治權對齊

- 不違 G5 single_stock_max = 5%(因為 cap 應用後仍 ≤ 5%)
- 不違 G4 attack_total_max = 20%(超出仍由 G4 cap)
- 不違 §0.2-A 第 5 條(單股 cap)

---

## 七、Prediction value-weighted 公式設計

### 7.1 公式定義

```
類似 ROE-weighted:
1. value_z = (prediction_value - value_mean) / value_std
2. value_mult = 1 + prediction_value_weight_beta × value_z
3. Clamp [0.7, 1.3](beta 較小,clamp 較窄)
4. 與 ROE multiplier 相乘:
   final_mult = roe_mult × value_mult
   final_mult = clamp(final_mult, 0.5, 1.5)
5. proposed = base_cap × final_mult,再 cap by G5
```

### 7.2 治權對齊

- 兩個 multiplier 設計分權(ROE 反映「基本面 quality」/ value 反映「predicted alpha 強度」)
- 不違 §0.2-A 7 禁令(α 不是固定,是 data-driven)

---

## 八、新增 audit gates 規格

### 8.1 G13 ROE-weighted Pareto compliance

```
條件:在配置完成後,top 1 ROE 股之 weight ≥ median ROE 股 weight × 1.0
意涵:高 ROE 股至少不應 < 中位 ROE 股
FAIL 動作:raise ConstitutionalViolationError(gate_id="G13", ...)
```

### 8.2 G14 score_scope v0.6/v0.7 對齊

```
條件:upstream prediction_run 必須對應 v0.6 或 v0.7 snapshot
意涵:確保 sizer 對齊 ROE 解鎖後 production
FAIL 動作:raise ConstitutionalViolationError(gate_id="G14", ...)
```

### 8.3 G15 ROE coverage gate (WARN-only)

```
條件:core+convex 之 ROE 覆蓋度 < 90%
意涵:金融業 BS 對齊問題揭露(§14.7-BL 候選);WARN 不 FAIL
WARN 動作:log warning(不阻塞配置)
```

---

## 九、治權對齊度檢驗

### 9.1 對 §9.2-A~H 既有 12 FAIL gate

| Gate | v0.3 動作 | 對齊 |
|---|---|---|
| G1 唯一 delivery | 不改 | ✅ |
| G2 覆蓋度 150 | 不改 | ✅ |
| G3-G8 槓鈴與限額 | 不改 | ✅ |
| G9-G10 治權純度 | 不改 | ✅ |
| G11 as_of_date 一致(v0.2) | 不改 | ✅ |
| G12 single_sector_count_max | 5 → **3** | ✅(降參數,不改邏輯)|
| **G13/G14/G15(v0.3 新增)** | 新增 | ✅(擴張,不違既有)|

### 9.2 對 §0.2-A 7 禁令

- α 不得固定:roe_weight_alpha = 0.5 是 tunable 預設值,不是 hard-code = ✅
- universe 不得動態切換:不改 universe = ✅
- 中段不得字面化:仍只配 long signal = ✅
- 不得跳 §0.4:不繞 §0.4 = ✅
- 攻擊端不得 > 20%:不改 attack_cap = ✅
- 不得納入防護端:不改 safety_min = ✅
- 長尾理論不得改變 §6.7 鎖定:不改 §6.7 = ✅

### 9.3 對 v6.1.18+ 新治權

- §14.7-BI ROE 解鎖:✅ 對齊(用 fg_roe weighted)
- §14.7-BJ Path D 認賠:✅ 對齊(未升 sponsor 時 fallback 至 ROE = None)
- §14.7-BK F 升 T1 預備:✅ 不衝突(IF 仍 T2,不影響)
- §14.7-AX(E) 外部資源 protocol:✅ 不適用(本子節無外部 API)

---

## 十、證偽承諾 T_PS_v0.3-1〜5

| ID | 證偽指標 | 通過門檻 | 失敗反應 |
|---|---|---|---|
| **T_PS_v0.3-1** | v0.3 vs v0.2 配置 weight 變化 | 高 ROE 股 weight ≥ 低 ROE 股 × 1.3 | 公式重新調 alpha |
| **T_PS_v0.3-2** | v0.3 G13 ROE-weighted compliance | 100% allocation 通過 G13 | 公式 bug 修補 |
| **T_PS_v0.3-3** | v0.3 G12 single_sector_count_max=3 強化 | 100% allocation 通過 G12 (max 3 per sector) | 強制重組或 fallback to 5 |
| **T_PS_v0.3-4** | v0.3 對齊 v0.6/v0.7 snapshot | G14 通過 100% | upstream prediction policy 配套升版 |
| **T_PS_v0.3-5** | walk-forward IC ≥ v0.2 baseline | 跨 12 期 IC mean ≥ v0.2 mean | revert to v0.2 |

---

## 十一、對既有 snapshot / 治權影響

| 項目 | 影響 |
|---|---|
| 既有 v0.2 snapshot | **零**(v0.2 配置不重 build)|
| §9.2-A~H 既有 12 FAIL gate | **零**(僅擴張 G13-G15)|
| §0.2-A 7 禁令 | **零** |
| §6.4 CoreScore 公式 | **零**(不改 builder)|
| §6.7 universe SSOT | **零** |
| upstream prediction policy | 需配套升至 v0.2(對齊 v6.1.18+ v0.7 snapshot)|
| audit_doctrine_compliance.py | 需更新識別 sizing_policy_v0.3(配套小升版)|

---

## 十二、v0.3 落地路徑

### Phase 1: 治權先行(本研究 + 入憲)

```
1. 寫設計研究 ← 本文件
2. 入憲 §9.2-I (v0.3 補強條款) 或 §14.7-BL(治權升版預備記述)
3. commit charter
```

### Phase 2: 程式落地

```
4. portfolio_sizer.py v0.2 → v0.3:
   - DEFAULT_POLICY 升版(5 個新 params)
   - apply_policy 加 ROE-aware + value-weighted multiplier
   - 加 G13/G14/G15 audit gates
   - load_inputs 加 ROE lookup(讀 v0.7 snapshot fg_roe)
   - 標頭主權狀態行 + 修訂歷程加 v0.3 entry
5. audit_doctrine_compliance.py(配套小升版,識別 sizing_policy_v0.3)
6. commit code
```

### Phase 3: smoke test + commit

```
7. dry-run v0.3 對 2025-04-25 prediction_run
8. 比對 v0.2 vs v0.3 配置差異(預期:仍 100% 半導體 但 ROE 高的 stock 拿更多 weight)
9. 入憲 §14.7-BM(v0.3 落地完成記述 + 證偽承諾啟動)
10. commit 全部
```

---

## 十三、對下游影響

### audit_core_universe.py:零

- audit 是看 core_universe builder,不看 sizer
- 不需配套升版

### audit_doctrine_compliance.py:小升版

- 加識別 sizing_policy_v0.3
- 加識別 sizing v0.3 新 audit gates(G13/G14/G15)
- 不改既有檢驗邏輯

### feature_store_builder.py:零

### model_trainer.py:零(§10 是 root cause 治本另案)

### prediction_engine.py:零(只 upstream prediction policy 升至 v0.2)

---

## 十四、Cross-Reference 精確行號

| 項目 | 位置 |
|---|---|
| §9.2-A~H 強制契約 | charter L102 (entry) + 後續主章 |
| §14.7-AA Part C(100% 半導體揭露) | charter L98 |
| §14.7-BI ROE 解鎖 | charter L8662 |
| §14.7-BJ ROE Path A blocked | charter L8794 |
| §14.7-BK F 升 T1 Phase A | charter L8721 |
| §14.7-AX(E) 外部資源 protocol | charter L7708 |
| v0.1 proposal | `reports/portfolio_allocation_proposal_2025-04-25.md` |
| v0.2 audit | `reports/portfolio_sizer_v02_implementation_audit_20260520.md` |
| 本 v0.3 設計研究 | `reports/portfolio_sizer_v03_design_research_20260526.md`(本檔)|

---

## 十五、治權邊界嚴守 + 結論

### 本 v0.3 不改:

- §9.2-A~H 既有 12 FAIL gate(G1-G12)
- §0.2-A 7 禁令
- 攻擊端 20% / 防護端 80% / 單股 5% / convex 3% / sector 40%
- §6.4 CoreScore 公式
- §6.7 universe SSOT
- §0.1-A / §0.3-A 治權禁令
- raw DDL
- CLI 參數結構(不增加 user-tunable 參數,roe_alpha 等列為 DEFAULT_POLICY 內部)

### 本 v0.3 新增:

- G13/G14/G15 audit gates
- ROE-aware weighting 公式
- prediction value-weighted 公式
- single_sector_count_max 5 → 3
- 對齊 v6.1.18+ snapshot(v0.6/v0.7)

### 結論

v0.3 是 **§9.2 portfolio_sizer 治權層之治標升版** — 在不改既有 12 FAIL gate 前提下,引入 ROE-aware Pareto 集中右尾 + sector count 強化。**Root cause 限制誠實聲明**: upstream prediction 100% 半導體之 candidate pool 仍需 §10 model_trainer 治本(另案 v6.2.0 軌道)。

本研究為治權先行步驟;**入憲 §9.2-I (v0.3 補強條款) 後始可程式落地**(對映 §0.0-G Level 1 紀律)。

---

*Report generated 2026-05-26 by Claude Sonnet 4.7 session*
*基於 v0.7 production snapshot + v6.1.18.2 cross-machine handoff context*
*Pending charter inscription before code landing*
