# Doctrine-Gate-First Universe Selection — Phase A Design Research

**日期**: 2026-05-26
**Phase**: A(設計研究 / pre-charter inscription)
**對應軌道**: v6.4.0 或 v7.0.0(治權判準層升版;比 v6.3.0 §14.7-BT 更深)
**對應憲章基礎**: §0.0-A 三核心思想統合層 / §0.0-B/C/D 三基柱跨層完整度基線 / §6.4 CoreScore / §6.7 SSOT SQL / §6.7.1 dynamic size annex(§14.7-BT 第十八輪)/ §6.8 制度穩定性 / §14.7-BU completeness governance(第十九輪)
**Status**: ✅ Phase A 完整(15 章 / non-destructive / 不動 DB 不動 code)

---

## 1. 觸發背景

用戶於 2026-05-26 連續兩次明示之 doctrine 升級:

### 1.1 第一次明示(doctrine 釐清)

> 「核心股的挑選時,在第一性原理、八二法則、康波週期是否都具有對應的資料來源依據進行核心股挑選,沒有一定要多少支核心股,但必須符合第一性原理、八二法則、康波週期都具有對應的資料來源為依據。」

意涵:核心股挑選之**真正判準**為「per-candidate × 三基柱資料來源依據」,N(支數)為判準結果非目標。

### 1.2 第二次明示(active deprecation)

> 「取消目前核心股為 150 支在任何地方。」

意涵:強化第一次之釐清為**治權清整動作** — 不只 framing 修正,而是 active 刪除「150 = 目前核心股數量」之 prescription 語句。

### 1.3 與 §14.7-BT 之差異

| 治權層 | §14.7-BT(第十八輪)| §14.7-BV(本研究)|
|---|---|---|
| 取消對象 | §6.7 SSOT 150 hardcode **prescription** | 「150 = 目前核心股」之 **statement** 在任何地方 |
| Selection algorithm | Top X% by composite CoreScore(X=5%)| **Doctrine-gate first**,然後 score-rank for tier |
| N policy | N_min=100 / N_max=200 / 3-10% dynamic range | N **完全動態**;由 doctrine-gate 結果決定 |
| 核心判準 | CoreScore composite(綜合 6 sub-scores)| 三基柱資料來源依據(per Phase A §2 矩陣)|
| Charter narrative | 不追溯修正既有 entries | **追溯範圍待界定**(本研究分析)|

---

## 2. 既有「150」分布 Inventory

### 2.1 Charter(系統架構大憲章_v6.1.0.md)

**Total**: 217 occurrences

| 類型 | 估計分布 | 處置建議 |
|---|---|---|
| 歷史 entries 之描述語(v5.4.x / v6.0.x 落地紀錄)| ~70% | **保留**(per §14.7-BT precedent / 對齊 §0.0-I 既有 entries respect)|
| §6.7 SQL contract 註解 / 例子 | ~10% | **保留**(SQL contract 為 SSOT 不含 size)|
| §6.7.1 annex 之 dynamic range(N_min/N_max)| ~5% | **保留**(§14.7-BT 已取消 hardcode prescription;range 為 cap)|
| 「150 為目前核心股」之 active prescription | ~15% | **追溯修正候選**(本研究主要 scope)|

### 2.2 Production Scripts(scripts/core/)

**Total**: 32 occurrences across 5 files

| File | Line | 類型 | 阻力等級 |
|---|---|---|---|
| `feature_store_builder.py` | L35 | docstring(`[Universe Lock]: 範圍鎖定 150 支`)| 🟡 soft prescription |
| `feature_store_builder.py` | L271 | preflight WARN if N ≠ 150 | 🟡 warning(不阻斷)|
| `core_universe_builder.py` | L52 | docstring(`as-of 候選 < 150 bootstrap minimum`)| 🟡 soft |
| `core_universe_builder.py` | L57+ | 修訂歷程紀錄(v0.5-v0.7 之多 entries) | 🟢 歷史 |
| `prediction_engine.py` | L238-239 | **HARD FAIL** if `len(self.rows) != 150` | 🔴 **HARD BLOCK** |
| `portfolio_sizer.py` | L116 | `"required_coverage": 150` constant | 🔴 **HARD CONSTRAINT** |
| `portfolio_sizer.py` | L174-175 | G2 gate `if prediction_rows != 150: return FAIL` | 🔴 **HARD BLOCK** |
| `portfolio_sizer.py` | L467 | log message `"G2 prediction × membership join complete (150/150)"` | 🟡 cosmetic |

**3 HARD BLOCK 點位**(`prediction_engine.py` + `portfolio_sizer.py` × 2)為 production 落地之**必拆障礙**;不拆則新 snapshot(N ≠ 150)無法進 pipeline。

### 2.3 Audit Tools(scripts/maintenance/)

| File | 影響 |
|---|---|
| `audit_core_universe.py` | `POLICY_SCORE_SCOPE_MAP` 含 v0.2-v0.7.1 之預期 / `EXPECTED_SCORE_DETAIL_KEYS` per policy / `check_v01_boundary` 依 policy 版本分流 — 需加 v0.9 識別 |
| `audit_downstream_readiness.py` | 預期 150 prediction rows | 需 dynamic threshold |
| `audit_doctrine_compliance.py` | 既有 dual-track promotion gate(§14.7-AV)— 需加 doctrine-gate 規則 |

### 2.4 DB Snapshots

| Snapshot | Policy | N | 處置 |
|---|---|---|---|
| `core_universe_20260524_core_universe_policy_v0_2` | v0.2 | 120 + 30 = 150 | **legacy preserve**(audit trail) |
| `core_universe_20260522_core_universe_policy_v0_3` | v0.3 | 120 + 30 = 150 | legacy preserve |
| `core_universe_20260522_core_universe_policy_v0_7` | v0.7 | 120 + 30 = 150 | legacy preserve |
| 待 build:`core_universe_20260527+_core_universe_policy_v0_9` | v0.9 | doctrine-derived(預期 ~119-146)| **本研究 Phase D-1 commit** |

---

## 3. 4 Selection Algorithm 對比評估

### 3.1 Path A: Current(CoreScore Top-N + hardcode N=150)

**Status**: v0.2-v0.7.1 在用;§14.7-BT 已取消 hardcode prescription 但 algorithm 仍 score-driven。

```python
candidates = filter_by_basic_criteria(all_stocks)  # ~2800
scores = compute_corescore(candidates)
selected = top_n(scores, n=150, core_n=120, convex_n=30)
```

**Pros**: Simple / production-proven
**Cons**: N 為 target 而非結果 / 不直接 enforce doctrine / 可能選入 doctrine-不全之 stocks

### 3.2 Path B: §14.7-BT v0.8 Proposed(Top X% by CoreScore)

**Status**: §14.7-BT Phase A 推薦;Phase C 程式落地未跑(roadmap 留)。

```python
candidates = filter_by_basic_criteria(all_stocks)
scores = compute_corescore(candidates)
top_pct = 5.0  # X = top 5%
n_target = max(N_min, min(N_max, int(len(candidates) * top_pct / 100)))
selected = top_n(scores, n=n_target, core_pct=0.70)
```

**Pros**: Dynamic N / 對齊 §0.2 八二法則 top 5% / N_min/N_max cap 保護
**Cons**: 仍 score-driven / 不 enforce doctrine / 5% threshold 為新 hardcode(只是更細)

### 3.3 Path C: Doctrine-Gate First, Score-Rank Second(★ 推薦 ★)

**Status**: 本研究設計;對應用戶 doctrine 直接 enforce。

```python
candidates = filter_by_basic_criteria(all_stocks)

# Stage 1: Market-level §0.3 prerequisite
if not check_kwave_market_context(min_indicators=5):
    raise PreconditionError("§0.3 K-wave market context insufficient")

# Stage 2: Per-stock doctrine gate
gate_pass = []
for stock in candidates:
    fp_coverage = check_first_principle_coverage(stock)  # 5 sources
    pareto_coverage = stock in candidates  # by-def True
    if fp_coverage == 5 and pareto_coverage:
        gate_pass.append(stock)

# Stage 3: Within gate-pass set, CoreScore rank for tier assignment
scores = compute_corescore(gate_pass)
ranked = sorted(scores, key=lambda x: x.score, reverse=True)

# Stage 4: Tier split (core 70% / convex 30% within gate-pass set)
n_total = len(gate_pass)
n_core = int(n_total * 0.70)
selected = {
    'core': ranked[:n_core],
    'convex': ranked[n_core:]
}
```

**Pros**:
- ✅ Doctrine-direct enforcement(每支 selected stock 必過 3 基柱資料源依據)
- ✅ N 完全動態(per gate-pass set 大小)
- ✅ tier 結構保留(下游 prediction_engine / portfolio_sizer 預期 core/convex 分層仍 work)
- ✅ §6.4 CoreScore 不廢棄(降為 tier 內 ranking 而非 selection gate)
- ✅ 對齊 §0.0-A 三基柱統合層 doctrine

**Cons**:
- ⚠️ Algorithm 更動較大(Path B 為 score-driven 之 incremental;Path C 為 paradigm shift)
- ⚠️ 下游 hardcode N=150 必須拆(3 HARD BLOCK 點位)
- ⚠️ Audit tools 需加 doctrine-gate rule(v0.9 policy 識別)

### 3.4 Path D: Pure Doctrine Pass(無 score rank)

**Status**: 最 maximalist 之 doctrine 讀法。

```python
# Just stage 1 + 2 of Path C; no rank, no tier
selected = gate_pass  # all 146 or 119 (per probe)
```

**Pros**: 最純粹 doctrine
**Cons**:
- ❌ 不分 core/convex(下游需大改 — portfolio_sizer 之 barbell allocation 依賴 tier 區分)
- ❌ §6.4 CoreScore 完全棄用(浪費 v0.2-v0.7 累積之 6 sub-score 設計)
- ❌ N 完全失控(若 ≥ 200 將觸 §6.7.1 N_max cap;若 ≤ 100 觸 N_min)

### 3.5 評估結論

| Path | Doctrine fit | Implementation cost | Downstream impact | Recommendation |
|---|---|---|---|---|
| A(current)| ❌ N-target 與 doctrine 牴觸 | 0(已 active)| 0 | rejected |
| B(§14.7-BT)| 🟡 partial(score-driven 不 enforce gate)| ~4-6h(已 designed)| 中(N dynamic 但仍 score-cap)| Phase B 已入憲但**未足以 fulfill** 用戶 doctrine |
| **C(★)** | ✅ direct enforcement | ~6-8h(builder + audit + downstream)| 中-高(3 HARD BLOCK 需拆)| **推薦** |
| D(pure)| ✅ 純粹 | ~10-15h(下游 barbell 大改)| **極高** | 過激 |

**Path C** 為平衡 doctrine + algorithm 結構 + downstream 衝擊之最佳路徑。

---

## 4. Path C Algorithm 詳細設計

### 4.1 §0.1 First-Principle 判準(per-stock)

**5 raw sources 存在性(per Phase A §7 hook 設計 / §14.7-BU 已對齊)**:

| Source | 對映 §0.1 維度 | Pass threshold |
|---|---|---|
| TaiwanStockPriceAdj | M(price)+ ΔlnP | ≥ 240 rows / 380d window(realistic Taiwan trading calendar)|
| TaiwanStockFinancialStatements | V(financial)| ≥ 8 rows / 24m(8 quarters)|
| TaiwanStockMonthRevenue | V(revenue)| ≥ 24 rows / 25m |
| TaiwanStockInstitutionalInvestorsBuySell | F(institutional)| ≥ 240 rows / 380d |
| TaiwanStockMarginPurchaseShortSale | F(margin)| ≥ 240 rows / 380d |

**Pass**: 5/5(strict)

**Per probe 2026-05-26 evidence**: 150 current → 146 pass strict(4 fail 小型股 margin / 新上市)

### 4.2 §0.2 Pareto 判準(per-stock)

**By-definition**: stock 在 candidate set(已過 basic filtering;in TaiwanStockInfo 上市/上櫃)→ §0.2 pass。

亦可加 score-based threshold(e.g., CoreScore ≥ some percentile);保留為 v1.0+ 升版選項。

### 4.3 §0.3 K-wave 判準(market-level)

**Global prerequisite**: market K-wave indicators ≥ N_min in DB

| Indicator | Source | Required |
|---|---|---|
| M2SL | FredData(per §14.7-BR Phase C-1)| ✅ |
| T10Y2Y | FredData | ✅ |
| VIXCLS | FredData | ✅ |
| TW_SEMI_VWAP_YOY | kwave_supply_cycle_proxy(§14.7-BR Phase C-2)| ✅ |
| TW_SHIPPING_VWAP_YOY | kwave_supply_cycle_proxy(§14.7-BR Phase C-4)| ✅ |

**Pass**: 5/5(strict;對齊 §14.7-BR Phase C-4 ceiling)

**意義**: §0.3 為 market signal,broadcast to all stocks。若 market context 不足,整個 selection 停擺(對齊 §0.3-A 禁令 #6 春初訊號合議制)。

### 4.4 Tier 分配(within gate-pass set)

```python
n_total = len(gate_pass)
n_core = int(n_total * 0.70)  # 70% to core
n_convex = n_total - n_core   # 30% to convex (rounded up)

scores_ranked = sorted(gate_pass, key=lambda x: x.corescore, reverse=True)
core_universe = scores_ranked[:n_core]
convex_universe = scores_ranked[n_core:]
```

**70/30 split 保留之 rationale**:
- 對齊 §6.7.1 annex(§14.7-BT 第十八輪 core-pct 70%)
- 對齊 §9.2 portfolio_sizer barbell allocation 結構
- 下游 prediction_engine / portfolio_sizer / model_trainer 假設 tier 分層存在

### 4.5 N 之動態邊界

依 §6.7.1 annex 之 N_min=100 / N_max=200 cap:

```python
if n_total < 100:
    raise InsufficientGatePassError(f"only {n_total} stocks pass doctrine gate; need ≥ 100")
if n_total > 200:
    # Take top 200 by CoreScore from gate_pass (degrade to score-cap mode)
    gate_pass = sorted(gate_pass, key=lambda x: x.corescore, reverse=True)[:200]
    n_total = 200
```

per probe evidence(2026-05-26)gate_pass ≈ 146(realistic threshold)~ 119(feature-derivable)— 皆在 [100, 200] cap 內,不觸發 boundary。

---

## 5. 對既有治權影響

### 5.1 治權層保留(不動)

- §0.1-A 6 禁令 / §0.2-A 7 禁令 / §0.3-A 7 禁令(三基柱核心 doctrine)
- §6.4 CoreScore 公式(降為 tier 內 ranking 而非 selection gate)
- §6.7 SQL contract(SSOT;size 不在 contract 內;per §14.7-BT 已 clarify)
- §6.7.1 dynamic size annex(§14.7-BT 第十八輪 / N_min=100, N_max=200, core-pct 70% 仍適用)
- §6.8 制度穩定性(年度頻率 / 不日週月頻 re-commit)
- §8.5 anti-leakage(builder layer 自管)
- §0.0-A/B/C/D 三基柱統合層 / 跨層完整度基線
- §14.7-BU(第十九輪)universe_completeness_snapshot — **直接成為 doctrine-gate 之驗證載體**

### 5.2 治權層升版(本研究 scope)

- **§14.7-BV 新建子節**(per §14.7-X 模式)
- 修訂歷程 v6.1.0-patch **第二十輪**
- 30+ legacy 「150 鎖定」entries 之**追溯範圍**:active prescription 處(~15%)考慮逐條評估;歷史記述(~70%)per §14.7-BT precedent 保留

### 5.3 Builder code 升版(Phase C 跨 session)

- `core_universe_builder.py` v0.7.1 → **v0.9**(Path C algorithm)
- `DEFAULT_POLICY_VERSION` v0.6 → `core_universe_policy_v0.9_doctrine_gate`
- 新方法:`check_first_principle_coverage(stock)` / `check_kwave_market_context()` / `apply_doctrine_gate(candidates)` / `tier_split_by_score(gate_pass)`
- CoreScore 公式不變(降階為 tier ranking)

### 5.4 Downstream code 升版(Phase D-2 跨 session)

| File | 變更 |
|---|---|
| `prediction_engine.py` L238-239 | `!= 150` → `< N_min(=100)`;assert universe matches snapshot N |
| `portfolio_sizer.py` L116 | `"required_coverage": 150` → `"required_coverage_min": 100` |
| `portfolio_sizer.py` L174-175 G2 | `!= 150` → `< snapshot.n_total`;dynamic per active snapshot |
| `portfolio_sizer.py` L467 log | `(150/150)` → `({actual}/{expected})` |
| `feature_store_builder.py` L271 | `expected ~150` → `expected = snapshot.n_total` |

### 5.5 Audit code 升版(Phase E 跨 session)

- `audit_core_universe.py`:加 `core_universe_policy_v0.9_doctrine_gate` 識別 + score_detail keys 升版
- `audit_downstream_readiness.py`:N expectation 改 dynamic per snapshot
- `audit_doctrine_compliance.py`:加 doctrine-gate-first 規則

### 5.6 DB snapshot 升版(Phase D-1 跨 session)

- v0.2 snapshot(committed)→ `deprecated`(但保留 audit trail)
- 新 commit `core_universe_20260527+_core_universe_policy_v0_9`(N = doctrine-derived)
- `universe_completeness_snapshot` 同次 backfill(Phase E builders 補 hook 後);本 snapshot 之 119 × 12 = 1,428 records 為 §14.7-BU T_BU-5 之首次實質驗收

---

## 6. 下游硬阻拆解詳細

### 6.1 prediction_engine.py L238-239

```python
# CURRENT (v?)
if len(self.rows) != 150:
    self._detail("fail", f"prediction universe rows={len(self.rows)}, expected 150")

# NEW (Path C)
expected_n = self.snapshot.total_stocks  # read from core_universe_snapshot
if len(self.rows) != expected_n:
    self._detail("fail", f"prediction universe rows={len(self.rows)}, expected {expected_n} per snapshot {self.snapshot.snapshot_id}")
```

### 6.2 portfolio_sizer.py L116, L174-175, L467

```python
# CURRENT
DEFAULT_POLICY = {
    "required_coverage": 150,
    ...
}

def check_g2_coverage(prediction_rows):
    if prediction_rows != 150:
        return False, f"G2: prediction rows = {prediction_rows}, expected 150"
    ...

# NEW (Path C)
DEFAULT_POLICY = {
    "required_coverage_min": 100,  # per §6.7.1 N_min
    "required_coverage_max": 200,  # per §6.7.1 N_max
    ...
}

def check_g2_coverage(prediction_rows, snapshot):
    expected_n = snapshot.total_stocks
    if prediction_rows != expected_n:
        return False, f"G2: prediction rows = {prediction_rows}, expected {expected_n} per snapshot {snapshot.snapshot_id}"
    if expected_n < DEFAULT_POLICY["required_coverage_min"]:
        return False, f"G2: snapshot N = {expected_n} below N_min = {DEFAULT_POLICY['required_coverage_min']}"
    ...
```

### 6.3 feature_store_builder.py L35, L271

```python
# CURRENT docstring
# 5. [Universe Lock]: 範圍鎖定 `core_universe ∪ convex_universe` 150 支

# NEW docstring  
# 5. [Universe Lock]: 範圍鎖定 `core_universe ∪ convex_universe`;N = doctrine-gate 結果(per §14.7-BV)
```

### 6.4 影響分析

- 拆 hardcode 之 risk 為**新 N 進入 production 流程**;若新 snapshot 已 build 完成 + 下游 hardcode 已升版 + audit 已 align,則無 production block
- 拆順序紀律:**snapshot commit → downstream hardcode 升版 → audit 升版** 為穩順序(downstream 早於 snapshot 升版會 break)
- 對齊 §14.7-BT Phase D-2 cascading rebuild 模式

---

## 7. 與 §14.7-BT / §14.7-BU 之治權關係

### 7.1 三輪 treaty change 治權堆疊

```
§14.7-BT(第十八輪): 取消 §6.7 SSOT 150 hardcode prescription
   ↓
§14.7-BU(第十九輪): 建 universe_completeness_snapshot per-stock × per-pillar × per-layer SSOT(governance)
   ↓
§14.7-BV(第二十輪;本研究): 改 selection algorithm 為 doctrine-gate first
```

§14.7-BV **無法獨立落地**:依賴 §14.7-BU 之 universe_completeness governance(per-stock pillar coverage check)。本 Path C 之 Stage 2 doctrine gate 即透過 §14.7-BU 之 universe_completeness_snapshot 表查 per-stock coverage。

### 7.2 §14.7-BU Phase E builders 補 hook 之 alignment

§14.7-BV Phase C builder rewrite **同次**完成 §14.7-BU Phase E 之 data layer + feature layer hook(於 builder 內部寫 universe_completeness_snapshot;一次 commit 含兩 phase)。對齊 §14.7-BT Phase A 之 multi-phase major treaty change pattern。

---

## 8. Phase A-G Roadmap

| Phase | 內容 | 時間 | Status |
|---|---|---|---|
| **A** | Design research(本報告) | ~1.5h | ✅ **本 entry** |
| **B** | 入憲 §14.7-BV + 修訂歷程 第二十輪 | ~30 min | ⏸ next(本 session 可推) |
| **C** | `core_universe_builder.py` v0.9 落地(Path C algorithm + §14.7-BU Phase E data/feature hook 合併) | ~4-6h | ⏸ cross-session |
| **D-1** | DB destructive:v0.2 deprecate + 新 v0.9 snapshot commit + universe_completeness_snapshot backfill(119 × 12 records) | ~1-2h | ⏸ cross-session |
| **D-2** | Downstream hardcode 拆:`prediction_engine` / `portfolio_sizer` / `feature_store_builder` | ~2-3h | ⏸ cross-session |
| **E** | Audit tools 升版:`audit_core_universe.py` / `audit_doctrine_compliance.py` / `audit_downstream_readiness.py` | ~2h | ⏸ cross-session |
| **F** | Integration test + dry-run smoke(從 builder → feature_store → model → prediction → sizer 全鏈)| ~1-2h | ⏸ cross-session |
| **G** | v6.4.0(或 v7.0.0)milestone tag + handoff | ~30 min | ⏸ closure |

**總計**: ~12-17h(可分 3-4 sessions)

---

## 9. Risks / Mitigation

| Risk | 機率 | 影響 | Mitigation |
|---|---|---|---|
| R1: doctrine-gate 結果 N < 100(< N_min) | 低 | 高 | Pre-build dry-run 驗證;若 N < 100 升 §0.1 threshold 重評(loosen);告警入 audit |
| R2: doctrine-gate 結果 N > 200(> N_max) | 低 | 中 | Stage 4 之 score-cap mode 自動 trim(已設計)|
| R3: 下游 hardcode 拆漏(某處仍 = 150)| 中 | 高 | Phase F integration test 全鏈 smoke;`grep -rn "= 150\|== 150\|: 150" scripts/` 為 final check |
| R4: §0.3 K-wave market context 不足(本機 2/5)| 高(本機)| 中 | Phase D-1 前 catch-up sync(M2SL / TW_SEMI / TW_SHIPPING);或允許 graceful degradation(WARN 而非 FAIL)|
| R5: v0.2 snapshot 重建期間 production gap | 中 | 高 | Phase D-1 前確保 v0.9 build 完成 + audit PASS;v0.2 維持 active 直到 cutover commit |
| R6: §14.7-BV 與 §14.7-BT 治權重疊 | 低 | 低 | §14.7-BV 明示「BT cancel prescription / BU build governance / BV change criterion」之治權堆疊關係;不撤銷 BT 結論 |
| R7: charter 既有 entries 之追溯修正範圍失控 | 中 | 中 | 依 §14.7-BT precedent 之 narrative continuity 紀律;只追溯 active prescription 處(~15%);歷史記述保留 |
| R8: §14.7-BU Phase E hook 與 §14.7-BV Phase C builder 改寫之 coupling | 高 | 中 | Phase C 同次完成 BU 之 data + feature layer hook;明示 dependency in §14.7-BV charter entry |

---

## 10. 入憲建議

### 10.1 新節 §14.7-BV(per §14.7-X 模式)

子節內容大綱:
1. 觸發(用戶 2026-05-26 dual明示)
2. 對 §14.7-BT 之深化關係(BT cancel hardcode / BV change criterion)
3. 4 algorithm 對比 + Path C 推薦
4. Path C 詳細 spec(Stage 1-4)
5. 對既有治權影響(治權保留 / 治權升版 / code 升版 / DB 升版)
6. Phase A-G roadmap
7. 與 §14.7-BU 之合併 Phase E hook coupling 明示
8. 治權新特性(predicted)

### 10.2 修訂歷程 第二十輪 entry draft

```
v6.1.0-patch 第二十輪: §14.7-BV Phase B 入憲 — Doctrine-Gate-First Universe Selection(治權判準層升版;比 §14.7-BT 更深;對應用戶 2026-05-26 dual 明示「N 為判準結果非目標」+ 「取消 150 在任何地方」)。Path C(doctrine-gate first → score-rank for tier within passing set)。對既有治權影響:零(Phase B 純 charter 入憲;Phase C-G 跨 sessions 取 per-phase 授權)。
```

### 10.3 升版優先級(per §0.0-E.6)

- **P0**:doctrine 一致性為治權核心;§14.7-BV 為 §14.7-BT/BU 之 doctrine closure → **P0 必補**
- **P1**:Path C algorithm 落地為 v6.4.0/v7.0.0 軌道 milestone → **P1 強烈推薦**
- **P2**:Downstream hardcode 拆 + audit 升版 → **P2 implementation**
- **P3**:integration test + milestone tag → **P3 quality**

---

## 11. §14.7-BV 治權新特性(predicted)

1. **首例「Doctrine-as-gate vs Score-as-rank」二分明文化**:doctrine = selection 之必要條件 / score = selection 之 ranking 排序工具;此區分為 §14.7-BV 之 paradigm shift
2. **首例「治權判準層 third-round refinement」**:§14.7-BT(cancel hardcode)→ §14.7-BU(build governance)→ §14.7-BV(change criterion);三輪累進為「同一 doctrine 在不同 surface 之逐輪 explicit 化」
3. **首例「跨 entry coupling」**:§14.7-BV Phase C 同次完成 §14.7-BU Phase E 之 data + feature layer hook;明示治權 entry 間之 execution coupling
4. **首例「3 HARD BLOCK 下游 hardcode 必拆」之 cascading destructive scope**:類比 §14.7-BT Phase D-2 但更深(BT 僅改 builder;BV 改 builder + downstream 3 files)
5. **首例「user dual-明示 trigger」**:用戶於同一 session 內 doctrinal 釐清 + active deprecation 兩次明示之 treaty change trigger(類比 §14.7-BT 之 4-confirm 但 dimensions 不同)

---

## 12. 證偽承諾 T_BV-1〜5

- T_BV-1: Path C algorithm 落地後,doctrine-gate pass set N 在 [100, 200] 範圍內(per probe 預期 ~119-146)
- T_BV-2: 新 v0.9 snapshot 之 universe_completeness_snapshot backfill 後,T_BU-5(N × 12 records coverage 完整)實質達標
- T_BV-3: 下游 3 HARD BLOCK 拆除後,全鏈 smoke test 不再因 N ≠ 150 而 FAIL
- T_BV-4: §14.7-BV 落地後,「150」在 scripts/ 之 active prescription 處(non-docstring)為 0 / scripts 之 32 → 預期 < 10(僅歷史紀錄)
- T_BV-5: charter 之「150 為目前核心股」之 active statement count: 217 → 預期 < 50(歷史 entries 保留 / active prescription 清整)

---

## 13. 對 §0.0-D D 基柱影響

§0.0-D(康波週期跨層完整度基線)目前 ~85%(post §14.7-BR Phase D)。本 §14.7-BV 落地:
- §0.3 K-wave 升為 selection prerequisite(market-level)→ 提升 D 基柱在 selection layer 之治權地位 +2-3pp
- 預期 D 基柱完整度:**87-88%**(post §14.7-BV Phase C-D)

對應 §0.0-B(第一性原理)+ §0.0-C(八二法則):
- B 基柱 +1-2pp(per-stock §0.1 5/5 強制 enforce)
- C 基柱 不變(§0.2 by-def 已 100%)

---

## 14. 配套程式落地(Phase C 跨 session)

`core_universe_builder.py` v0.7.1 → v0.9:

```python
# 新增 method
def apply_doctrine_gate(self, candidates):
    """Stage 2 of Path C: per-stock 三基柱資料源依據 gate."""
    
    # § 0.3 market-level prerequisite
    kw_indicators = self._check_kwave_market_context()
    if kw_indicators < 5:
        raise PreconditionError(
            f"§0.3 K-wave market context insufficient: {kw_indicators}/5"
        )
    
    # Per-stock §0.1 5/5 check + §0.2 by-def
    gate_pass = []
    for stock in candidates:
        fp_score = self._check_first_principle_coverage(stock)
        if fp_score == 5:  # §0.2 by-def True since stock in candidates
            gate_pass.append(stock)
    
    return gate_pass

def select_doctrine_gate_universe(self, candidates):
    """Path C main entry: doctrine-gate → score-rank → tier split."""
    gate_pass = self.apply_doctrine_gate(candidates)
    
    if len(gate_pass) < N_MIN:  # § 6.7.1 N_min
        raise InsufficientGatePassError(
            f"only {len(gate_pass)} pass doctrine; need ≥ {N_MIN}"
        )
    if len(gate_pass) > N_MAX:
        gate_pass = sorted(gate_pass, key=lambda x: x.corescore, reverse=True)[:N_MAX]
    
    scores = self._compute_corescore(gate_pass)
    ranked = sorted(gate_pass, key=lambda s: scores[s], reverse=True)
    
    n_core = int(len(ranked) * 0.70)
    return {
        'core': ranked[:n_core],
        'convex': ranked[n_core:],
    }
```

---

## 15. 結論

本 design research 提案 **Path C(Doctrine-Gate-First, Score-Rank Second)** 為對應用戶 2026-05-26 dual 明示之最佳治權路徑:

- ✅ **Doctrine 直接 enforcement**(三基柱資料源依據為 selection gate)
- ✅ **N 完全動態**(per gate-pass set;不再 150 hardcode)
- ✅ **tier 結構保留**(70/30 split within gate-pass;下游 barbell 相容)
- ✅ **§6.4 CoreScore 不廢棄**(降為 tier 內 ranking)
- ✅ **與 §14.7-BU 直接耦合**(透過 universe_completeness_snapshot 查 coverage;Phase C 同次完成 BU Phase E hook)
- ✅ **§14.7-BT precedent 對齊**(narrative continuity / 既有 entries 不過度追溯)
- ✅ **Phase A→G 拆解**(可控跨 3-4 sessions)
- ✅ **5 條治權新特性 + 5 條證偽承諾**

**下一步**(待用戶授權 per-phase):
1. Phase B 入憲(~30 min;本 session 可推 — 對應用戶授權 Full Phase A-G)
2. Phase C builder rewrite(~4-6h;cross-session;與 §14.7-BU Phase E 合併)
3. Phase D-1 DB destructive + D-2 downstream(~3-5h;cross-session)
4. Phase E audit + Phase F integration test(~3-4h;cross-session)
5. Phase G v6.4.0 / v7.0.0 milestone tag + handoff(~30 min)

---

**設計研究作者**: Claude
**Status**: ✅ Phase A complete / pre-charter / non-destructive / 不動 DB 不動 code
**Cross-References**:
- 對應憲章 §0.0-A / §0.0-B/C/D / §6.4 / §6.7 / §6.7.1 / §6.8 / §8.5 / §14.7-BT / §14.7-BU
- 對應 evidence:[`reports/universe_completeness_phase_e_preview_probe_20260526.md`](universe_completeness_phase_e_preview_probe_20260526.md)(146/119 數字之 evidence)
- 對應 user trigger:2026-05-26 雙重明示(doctrine 釐清 + active deprecation)
- 對應 memory:`core_stock_selection_doctrine.md`(已升 v2 反映本研究 doctrine 強化)
