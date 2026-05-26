# §14.7-BT Phase A 設計研究 — 取消 150 Hardcode + 動態 Universe Selection

- **產出日期**: 2026-05-26 late late late evening
- **產出者**: Claude Sonnet 4.7 session
- **觸發**: 用戶第 31+ 次 anchor「取消所有 hardcode + 依 DB 動態挑選核心股」
- **scope**: §6.7 SSOT 150 hardcode 移除 + dynamic universe selection algorithm 設計
- **位階**: §14.7-BT Phase A 設計研究(0 程式改 / 0 charter 改 / 類比 §14.7-BR Phase A pattern)
- **defensible defaults**(用戶 Q1-Q4 dismissed → Auto Mode 採):
  - Q1 Scope = L1 only(150 size hardcode)
  - Q2 Algorithm = B Top X% by composite CoreScore(X 預設 5%)
  - Q3 Phase = Phase A only(本封存)
  - Q4 Backward-compat = 完全 backward-compat(opt-in via new policy version)
- **後續 dependency**: Phase B-D 需用戶 explicit confirmation Q1-Q4
- **Session context**: 已 5× FINAL closed;本 Phase A 為 disciplined minimal scope

---

## 一、150 hardcode 之 5 層 inventory(完整 audit trail)

### 1.1 L1: Size limit(150 = 120 + 30)

```python
# scripts/core/core_universe_builder.py L249-250
class CoreUniverseBuilder:
    def __init__(
        self,
        ...
        core_limit=120,         ← HARDCODE A
        convex_limit=30,        ← HARDCODE B
        ...
    )

# scripts/core/core_universe_builder.py L2154-2155
parser.add_argument("--core-limit", type=int, default=120, ...)    ← HARDCODE C
parser.add_argument("--convex-limit", type=int, default=30, ...)   ← HARDCODE D
```

→ **4 處 hardcode** in 1 file

### 1.2 L2: Sub-score weights(6-layer)

```
DQ 25% / LM 25% / FG 20% / TR 15% / IF 10% / VC 5% - RP
hardcoded in:
  - builder _compute_corescore() method
  - charter §6.4 第 1-7 條(各 sub-score 之 weight 規定)
```

→ **本 Phase A scope 不涉**(L2 留 future research)

### 1.3 L3: Tier thresholds(5 階梯 score mapping)

```
每 sub-score(FG / IF / VC / LM / TR / DQ)各 5 階梯 thresholds:
e.g. VC: > +0.10 → 95 / > +0.05 → 85 / > 0 → 75 / > -0.05 → 60 / > -0.10 → 40 / 其他 → 20
hardcoded in:
  - builder 各 _score 方法
  - charter §6.3 第 1-7 條
```

→ **本 Phase A scope 不涉**(L3 留 future research)

### 1.4 L4: Statistical params(各模組之 lambda / threshold)

```
winsor 0.05/0.95 (model_trainer / portfolio_sizer)
sector_balance_lambda 0.3 (model_trainer milestone #3)
sector_entropy_min 0.5 (DEFAULT_TRAINING_POLICY)
walk_forward_panel_size 8 (DEFAULT_TRAINING_POLICY)
```

→ **本 Phase A scope 不涉**(L4 留 future research)

### 1.5 L5: Timeout / frequency(operational)

```
training_max_time_seconds 3600 (DEFAULT_TRAINING_POLICY)
60d window (volatility) / 252 trading days (annualize)
```

→ **本 Phase A scope 不涉**(L5 留 future research)

### 1.6 本 Phase A scope 鎖定

**Only L1**(150 size limit 4 處 hardcode)→ minimal disruption + defensible Phase A

---

## 二、Charter 治權衝突(§6.7 SSOT 150 鎖定)

### 2.1 Conflict scope

charter `§6.7 SSOT 150 鎖定` 為治權慣例(出現於 30+ entries):

```
v0.7 entry: 「本版不改 §6.7 SSOT 150 鎖定」
v0.6 entry: 「本版不改 §6.7 SSOT 150 鎖定」
v0.5 entry: 「本版不改 §6.7 SSOT 150 鎖定」
...(30+ 次)

Rebuild logs(20+ 次):
「§6.7 核心同步資產數 = 150」
```

### 2.2 §6.7 條文本身

charter §6.7 條文(L3380)**未明文「150」**;只規定 SQL contract:

```sql
SELECT m.stock_id
FROM core_universe_membership m
JOIN core_universe_snapshot s
WHERE s.status = 'committed'
  AND m.core_tier IN ('core_universe', 'convex_universe')
  AND s.as_of_date = (SELECT MAX(...) FROM ... WHERE status='committed')
```

→ §6.7 為「**SQL contract SSOT**」,**非 size lock**

### 2.3 「150 鎖定」治權升版策略

**3 個 escalating options**:

| Option | 治權成本 | 對既有 entries 影響 |
|---|---|---|
| **α. §6.7 不動 / 30+ entries 不動 / 加 §6.7.1 dynamic size annex** | 🟢 低 | 既有「150 鎖定」維持 valid 至 v0.7 snapshot;新 dynamic policy 起獨立 |
| β. §6.7 升版 + 30+ entries 全部追溯修正 | 🔴 高 | charter 散修 / 治權成本高 |
| γ. v6.3.0 重大升版 / charter major reset | 🔴 極高 | major version bump / charter 重寫 |

**本 Phase A 推薦 Option α**(類比 §14.7-BR Phase B 之 stale 追溯修正 pattern;最小 charter disruption + clear forward path)

---

## 三、Algorithm 5 方法 evaluation(Q2 decision matrix)

### 3.1 Comparison matrix

| Algorithm | 邏輯 | N 變動性 | 治權依據 | Complexity |
|---|---|---|---|---|
| **A. Top 5% by Trading_money** | §0.2 直接 / 純 liquidity-driven | ~140(隨日變)| §0.2-A 禁令 #2 流動性 | 🟢 低 |
| **B. Top X% by composite CoreScore**(預設 X=5%)| 既有 6-layer scoring + dynamic cap | ~140-150 | §0.0-A.1 + §6.4 既有 framework | 🟢 低(本 phase 推薦)|
| **C. Threshold-based**(score > X) | 絕對 score threshold;score 高即入選 | 60-280(隨日大幅波動)| 治權無對應條 | 🟡 中 |
| **D. Sector-balanced per MBNRIC 6 支柱**(per sector top N)| §0.3 MBNRIC 配額制 | 6 × N(若 N=25 → 150)| §0.3.9 MBNRIC × TWSE | 🔴 高 |
| **E. Walk-forward IC-tuned N**(per panel optimal N)| 跨歷史 panel 之 IC > 0 反推 N | 隨 panel 變 | §10-E walk-forward | 🔴 極高(需 §10 production validation 完成)|

### 3.2 推薦:Algorithm B(Top X% by composite CoreScore)

**理由**:
1. **保留既有 6-layer CoreScore design**(§6.4 charter 不動)
2. **對齊 §0.2 八二法則**(X=5% 對應「top 5% concentration」hypothesis)
3. **Charter conflict 最小**(只升 §6.7 size 部分,不動 §6.4 公式)
4. **Backward-compat 自然**(post X 計算 N 可逼近 150 / 與 v0.2-v0.7 snapshots 比對連續)
5. **Dynamic + adaptive**(X 為 design parameter;隨 universe 規模調整)

### 3.3 X% 之 default 選擇

| X 候選 | N 估計 | Rationale |
|---|---|---|
| **5%**(default) | ~140 stocks | §0.2 八二法則直接對應;與 historical 150 接近 |
| 3% | ~84 | 更集中 / 高 alpha 候選 / 風險:N 過小 IC noise |
| 7% | ~196 | 更寬鬆 / 風險:dilution of signal |
| 10% | ~280 | 過寬 / 不對應八二 |

→ **X=5% 為 defensible default**(對齊 §0.2 + 接近 historical 150)

### 3.4 N min/max safety guard

| Guard | 預設 | 理由 |
|---|---|---|
| `N_min` | 100 stocks | 避免 universe 過小 IC noise;對應 §0.2-E 證偽之 minimal critical mass |
| `N_max` | 200 stocks | 避免 signal dilution;對應 attention bandwidth constraint |

→ Algorithm B + N_min/N_max guards 為 robust design

---

## 四、新 Policy Version 設計(Backward-compat)

### 4.1 Policy version naming

```
既有 versions:
  core_universe_policy_v0.2(本機 / 150 hardcode)
  core_universe_policy_v0.6/v0.7(他機 production / 150 hardcode)

新 version:
  core_universe_policy_v0.8_dynamic  ← Phase A 提案
```

### 4.2 Dynamic policy metadata

新 v0.8 之 selection_policy JSON(寫入 `core_universe_snapshot.metadata`):

```json
{
  "selection_algorithm": "B_top_pct_composite_corescore",
  "selection_pct": 5.0,
  "selection_N_min": 100,
  "selection_N_max": 200,
  "actual_N_selected": 142,
  "actual_X_pct": 5.07,
  "core_convex_split": "70/30",
  "core_N": 99,
  "convex_N": 43,
  "rationale": "§0.2 top 5% + §6.4 composite CoreScore + N_min/max guards"
}
```

### 4.3 Core / Convex 之 dynamic split

| 既有 v0.7 | 新 v0.8 提案 |
|---|---|
| core_limit=120 / convex_limit=30(80/20 split)| **core_pct=70% / convex_pct=30%**(於 N 內動態切)|

→ 若 N=140:core ≈ 98 / convex ≈ 42(approximate 既有 120/30 ratio)

---

## 五、Builder 程式落地設計(Phase C 預備 / 本 phase 純設計)

### 5.1 Constructor change(L249-250)

```python
# BEFORE (v0.7)
def __init__(self, as_of_date, policy_version, commit=False,
             core_limit=120, convex_limit=30, ...):

# AFTER (v0.8 提案;backward-compat)
def __init__(self, as_of_date, policy_version, commit=False,
             core_limit=None, convex_limit=None,         # legacy:可設絕對數
             selection_pct=None,                          # NEW:dynamic %(if set,override core/convex_limit)
             selection_N_min=100, selection_N_max=200,   # NEW:safety guards
             core_pct_within_selected=0.70,              # NEW:core/convex split %
             ...):
    # 若 policy_version 起頭含 'v0.8_dynamic',強制 dynamic mode
    if policy_version.startswith('core_universe_policy_v0.8'):
        if selection_pct is None:
            selection_pct = 5.0  # default
        self.core_limit = None  # disabled
        self.convex_limit = None  # disabled
        self.selection_pct = selection_pct
    else:
        # legacy mode:用 core/convex_limit hardcode
        self.core_limit = core_limit or 120
        self.convex_limit = convex_limit or 30
```

### 5.2 Selection logic change

```python
# NEW: _apply_dynamic_selection() method
def _apply_dynamic_selection(self, scored_candidates):
    """v0.8 dynamic selection:Algorithm B Top X% by composite CoreScore."""
    n_total = len(scored_candidates)
    target_n = int(n_total * self.selection_pct / 100)
    target_n = max(self.selection_N_min, min(self.selection_N_max, target_n))
    
    sorted_by_score = sorted(scored_candidates, key=lambda s: s['core_score'], reverse=True)
    selected = sorted_by_score[:target_n]
    
    core_n = int(target_n * self.core_pct_within_selected)
    convex_n = target_n - core_n
    
    return {
        'core': selected[:core_n],
        'convex': selected[core_n:core_n + convex_n],
    }
```

### 5.3 CLI flags addition

```python
# NEW flags(backward-compat)
parser.add_argument("--selection-pct", type=float, default=None,
                    help="v0.8 dynamic top X% selection(if set,override --core-limit/--convex-limit)")
parser.add_argument("--selection-n-min", type=int, default=100,
                    help="v0.8 dynamic N min guard")
parser.add_argument("--selection-n-max", type=int, default=200,
                    help="v0.8 dynamic N max guard")
parser.add_argument("--core-pct", type=float, default=0.70,
                    help="v0.8 core_pct within selected(0.70 = 70/30 core/convex split)")
```

---

## 六、對 §0.1 / §0.2 / §0.3 三柱之 alignment

| 基柱 | Dynamic universe 對映 | Conflict / Reinforcement |
|---|---|---|
| §0.1 第一性原理 | 不直接影響(M/V/F/ΔlnP 仍透過 6-layer CoreScore 評分)| ✅ Reinforce(scoring 不動;只改 selection cap)|
| §0.2 八二法則 | **直接對齊**(Algorithm B X=5% = §0.2 之 top 5% concentration hypothesis 之 explicit 落地)| ✅✅ Reinforce strongly(從 implicit 150 約 5% → explicit 5%)|
| §0.3 康波週期 | 不直接影響(MBNRIC 字典 / theme 仍在 CoreScore 內)| ✅ Reinforce(若選 Algorithm D 則直接落地 §0.3.9 配額制)|

→ Algorithm B 對三柱皆 reinforcing(無 conflict)

---

## 七、對 downstream(Feature Store / Model / Prediction / Sizer)影響

| Downstream | 影響 |
|---|---|
| **Feature Store**(`feature_store_builder.py v0.5`)| 🟡 minor — feature_set_snapshot 之 `total_stocks` 從 fixed 150 → dynamic;DDL 已含此欄位(non-fixed) |
| **Model Trainer**(`model_trainer.py v0.2.4`)| 🟢 zero — `load_inputs` 之 SQL 用 §6.7 contract(query JOIN tier),已 dynamic |
| **Prediction Engine**(`prediction_engine.py v0.3`)| 🟡 minor — `coverage` audit 之 expected 150 → expected N(per snapshot metadata)|
| **Portfolio Sizer**(`portfolio_sizer.py v0.3`)| 🟢 zero — 從 prediction_run 之 N stocks 為 input,不依賴 fixed 150 |
| **audit_core_universe**(`audit_core_universe.py v0.2`)| 🔴 major — expected `§6.7 = 150` 之 invariant 需動態化 |

→ **3 個下游需 minor patch**(feature_store + prediction_engine + audit_core_universe);**Phase C 範圍含這 3 個 patches**

---

## 八、§6.7 Charter 升版 draft

### 8.1 推薦:Option α(加 §6.7.1 dynamic size annex)

```markdown
### 6.7.1 動態 Universe Size Annex — *§14.7-BT Phase B 入憲 / 2026-XX*

§6.7 之 SQL contract 為 SSOT;**size 不在 §6.7 SQL contract 內鎖定**。
歷史 v0.2-v0.7 snapshots 採 `core_universe=120 + convex_universe=30 = 150` 為 hardcode default;
v0.8+ snapshots 採 **dynamic size**(per policy metadata 之 `selection_pct` + N_min/max guards)。

**Dynamic Size 之治權邊界**:
1. **N_min ≥ 100**(避免 IC noise;對應 §0.2-E 證偽 minimal critical mass)
2. **N_max ≤ 200**(避免 signal dilution;對應 attention bandwidth)
3. **selection_pct ∈ [3%, 10%]**(對應 §0.2 八二法則之 reasonable concentration band)
4. **core / convex split**:預設 70/30(per N);可 override
5. **policy metadata 強制宣告**:`selection_algorithm` + `selection_pct` + `actual_N` 寫入 snapshot.metadata
6. **既有 v0.2-v0.7 snapshots 治權維持**:hardcode 150 為 legacy mode;dynamic 為 v0.8+ opt-in
```

### 8.2 「30+ entries 之 150 鎖定」追溯策略

```
保留(legacy 慣例):
  v0.7 之前 entries 之「§6.7 SSOT 150 鎖定」維持 valid(代表 hardcode mode)
  
新增解釋(v0.8+):
  v0.8 起 policy_version startswith 'v0.8_dynamic'時,size 為 dynamic
  per snapshot.metadata 之 selection_pct + actual_N
  
治權成本:零(legacy entries 不追溯;只加新 §6.7.1 annex)
```

---

## 九、Backward-compat design pattern

### 9.1 模式 1:Policy-version-driven mode dispatch

```python
if policy_version.startswith('core_universe_policy_v0.8'):
    # NEW dynamic mode
    self._apply_dynamic_selection(...)
else:
    # LEGACY hardcode mode(v0.2-v0.7)
    self._apply_legacy_selection(core_limit=120, convex_limit=30)
```

→ **0 risk to 既有 v0.2-v0.7 snapshots**(同 builder code 跑既有 policy 仍 → 150)

### 9.2 模式 2:CLI flag opt-in(secondary)

```bash
# Legacy mode(default 0.7):
python core_universe_builder.py --commit --policy-version core_universe_policy_v0.7

# NEW dynamic mode(opt-in):
python core_universe_builder.py --commit --policy-version core_universe_policy_v0.8_dynamic --selection-pct 5.0
```

### 9.3 模式 3:Audit tool dual mode

`audit_core_universe.py v0.3`(Phase C):
```python
if snapshot.policy_version.startswith('v0.8'):
    expected_n = snapshot.metadata['actual_N_selected']  # dynamic
else:
    expected_n = 150  # legacy invariant
```

---

## 十、Phase B-D Roadmap

### 10.1 Phase A(本 phase ✅)

- 本封存(~3-4h)
- 0 program 改 / 0 charter 改
- Defensible defaults Q1=L1 / Q2=B / Q3=Phase A only / Q4=完全 backward-compat
- Deliverable:本 design research

### 10.2 Phase B(charter 入憲;~2h;**需用戶 explicit 確認**)

- §6.7.1 dynamic size annex 新建(Option α)
- 新 §14.7-BT 子節(類比 §14.7-BR Phase B pattern)
- charter v6.1.0-patch 第十八輪(或 v6.3.0 開始,per Q4 decision)
- **Pre-requisite**:用戶確認 Q1-Q4 之 final values(本 phase 採 defensible defaults 待確認)

### 10.3 Phase C(程式落地;~4-6h)

- `core_universe_builder.py v0.7 → v0.8`:
  - L249-250 constructor 加 dynamic mode 分支
  - L2154-2155 CLI 加 --selection-pct 等 flags
  - `_apply_dynamic_selection()` 新方法
  - DEFAULT_POLICY_VERSION = 'core_universe_policy_v0.8_dynamic'
- `audit_core_universe.py v0.2 → v0.3`(dual mode + expected N 動態化)
- `feature_store_builder.py`(minor patch / total_stocks 接受 dynamic)
- `prediction_engine.py`(minor patch / coverage 接受 dynamic)

### 10.4 Phase D(production smoke + tag;~1h)

- Dry-run v0.8 dynamic with X=5%
- 對比 v0.7 之 150(historical baseline)vs v0.8 dynamic N
- Evidence report
- Tag `v6.3.0-dynamic-universe-selection-phase-d`(若 Q4=major upgrade)
  或 `v6.2.1-dynamic-universe-selection-phase-d`(若 Q4=minor)

### 10.5 Total estimated effort

```
Phase A(本封存)               ~3-4h ✅
Phase B(charter 入憲)          ~2h
Phase C(builder + audit)       ~4-6h
Phase D(smoke + tag)            ~1h
─────────────────────────────────────
Total                          ~10-13h 跨 2-3 sessions
```

---

## 十一、Risks / Mitigation

| Risk | 等級 | Mitigation |
|---|---|---|
| **R1**: 既有 v0.7 snapshot 之 IC validation 不可比(150 vs dynamic N)| 🟡 中 | dual mode 保留 legacy;新 evidence 報告對比 150 vs dynamic 之 IC |
| **R2**: §6.7 30+ entries 之 治權慣例 inconsistency | 🟢 低 | Option α 不追溯;只加 §6.7.1 annex |
| **R3**: dynamic N 過小(< 100)IC noise | 🟢 低 | N_min=100 guard |
| **R4**: dynamic N 過大(> 200)signal dilution | 🟢 低 | N_max=200 guard |
| **R5**: core/convex split 70/30 之 reasonableness | 🟡 中 | 對應 historical 120/30 = 80/20;Phase C dry-run 可調 |
| **R6**: §6.8 年度重選契約之 size 變動性影響制度穩定性 | 🟡 中 | dynamic N 仍在 [100, 200] band 內;年度重選頻率不動 |

---

## 十二、§0.0-G Level 1 對齊

```
Phase A 設計研究 ✅ 本封存(0 program 改 / 0 charter 改)
Phase B 治權先行入憲(charter §6.7.1 + §14.7-BT 子節)
Phase C 程式落地(builder + audit + minor downstream patches)
Phase D production smoke + tag
─────────────────────────────────────────────────────────
本 phase 對齊 §0.0-G Level 1 紀律之第 N 次跑通候選(類比 §14.7-BR pattern)
```

---

## 十三、§0.0-H 八子節模板不適用

§14.7-BT 為 cross-layer sync 任務(類比 §14.7-BR):
- scope 涵蓋 builder + charter + audit + downstream patches(4 件)
- 非單一程式之治權契約
- 不適用 §0.0-H 八子節模板

---

## 十四、Cross-Reference

- 本機 v0.2 snapshot:`core_universe_20260521_core_universe_policy_v0_2`(150 hardcode)
- 他機 v0.7 production snapshot(150 hardcode;§14.7-BI ROE 解鎖)
- builder hardcode 位置:`scripts/core/core_universe_builder.py` L249-250 + L2154-2155
- charter §6.7 條文:`reports/系統架構大憲章_v6.1.0.md` L3380(SQL contract)
- charter 30+ entries 之「§6.7 SSOT 150 鎖定」:跨 v0.3-v0.7 多輪修訂歷程
- 類比 pattern:§14.7-BR Phase A(`reports/kwave_leading_indicators_phase_a_research_20260526.md`)
- 類比 Phase B 入憲 pattern:§14.7-BR Phase B(commit 95fda16 / 第十六輪)
- §0.2 八二法則:`reports/pareto_4_dimensions_evidence_v02_baseline_20260526.md`(87548f1)
- §6.8 年度重選契約:`reports/系統架構大憲章_v6.1.0.md` L3419

---

## 十五、結語 + Next session 推進建議

### 15.1 本 Phase A 之 deliverable

- **本封存** = 完整 design research(15 chapters / 5 algorithm evaluation / 5 hardcode 層 inventory / charter conflict 策略 / backward-compat pattern / Phase B-D roadmap)
- **defensible defaults**:Q1=L1 only / Q2=B / Q3=Phase A only / Q4=完全 backward-compat
- **0 program 改 / 0 charter 改**

### 15.2 Next session 必先問用戶之 critical questions

```
Q1. Scope:確認 L1 only?或擴 L1-L3?L1-L5?
Q2. Algorithm:確認 B Top X% by composite CoreScore?X 取 5%?
Q3. Phase:Phase B 同次入憲?或拆?
Q4. Backward-compat:確認 opt-in via new policy version 之 dual mode pattern?
Q5. Version:v6.2.1 patch(minor)?或 v6.3.0 milestone(major)?
```

### 15.3 Phase B-D 推進條件

- Phase B 啟動 = 用戶確認 Q1-Q5
- Phase C 啟動 = Phase B charter 已入憲
- Phase D 啟動 = Phase C 程式 dry-run 通過

### 15.4 預期 outcome(post Phase D)

```
v0.7 hardcode 150 → v0.8 dynamic ~140(X=5% / N_min=100 / N_max=200)
charter §6.7.1 dynamic size annex 入憲
治權慣例「150 鎖定」維持為 legacy(v0.7 之前 snapshots)
新 v0.8+ snapshots 採 dynamic
backward-compat 100%(既有 snapshots 不變)
```

---

*Report generated 2026-05-26 late late late evening by Claude Sonnet 4.7 session*
*基於 Q1-Q4 dismissed → Auto Mode defensible defaults(L1 only / Algorithm B / Phase A only / 完全 backward-compat)*
*類比 §14.7-BR Phase A pattern(commit f07ba16)— 5 algorithm evaluation + charter conflict 策略 + Phase B-D roadmap*
*Session 已 5× FINAL closed;本 Phase A 為 disciplined minimal scope;Phase B-D 留 next session(需用戶 explicit Q1-Q5 確認)*
