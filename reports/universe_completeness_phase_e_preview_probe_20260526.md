# Universe Completeness Phase E Preview — One-shot 3-Pillar Coverage Probe

**日期**: 2026-05-26
**性質**: 一次性 evidence probe(非 §14.7-BU Phase E 正式 hook;不寫 DB;不改 code)
**對應憲章**: §0.4 數位孿生完整性 / §14.7-BU Phase B 入憲(charter L9378)/ Phase A research §2 三基柱 × 四層矩陣
**對應 doctrine**: 核心股挑選之治權判準 = 三基柱皆具備對應資料來源依據(N 不固定;源自用戶 2026-05-26 明示)
**對應 snapshot**: `core_universe_20260524_core_universe_policy_v0_2`(as_of=2026-05-24;committed)

---

## 1. 觸發

§14.7-BU Phase B closure 後,文件多處引用「119 stocks(83 core + 36 convex)」與 DB committed snapshot 之 **150 (120 core + 30 convex)** 不一致。用戶釐清:此差異非 doctrine 衝突,而是不同 selection algorithm 對「滿足三基柱資料依據」之 stocks 數目的計算差。本 probe 用現有 raw tables,bracket 該差到「raw data availability vs feature-derivable」之邊界。

---

## 2. 判準對映

| 基柱 | 判準層次 | 本 probe 採用 |
|---|---|---|
| §0.1 第一性原理 | per-stock × 5 raw sources(M / V × 2 / F × 2) | TaiwanStockPriceAdj / FinancialStatements / MonthRevenue / InstitutionalInvestorsBuySell / MarginPurchaseShortSale |
| §0.2 八二法則 | by-definition(in core/convex tier) | core_universe_membership.core_tier IN ('core_universe','convex_universe') |
| §0.3 康波週期 | market-level(uniform across stocks)| FredData.series_id + (kwave_supply_cycle_proxy if 存在) |

---

## 3. §0.1 First-Principle Coverage 結果

### 3.1 Loose criterion(per Phase A §7 hook 設計;source 存在性)

```
5/5 sources exist: 150 / 150 stocks (100.0%)
```

**結論**:依 Phase A research 設計之 hook(actual = count of fetched sources / pct = actual/5),**所有 150 支皆 100% 過 §0.1**。

### 3.2 Realistic threshold criterion(近似 production usable)

| Source threshold | 對應拉取 |
|---|---|
| M_price_240d | ≥ 240 rows in TaiwanStockPriceAdj(380d window;~年度交易日) |
| V_fs_8q | ≥ 8 rows in TaiwanStockFinancialStatements(24m;~8 季 × 1 metric)|
| V_rev_24m | ≥ 24 rows in TaiwanStockMonthRevenue(25m)|
| F_inst_240d | ≥ 240 rows in TaiwanStockInstitutionalInvestorsBuySell(380d)|
| F_margin_240d | ≥ 240 rows in TaiwanStockMarginPurchaseShortSale(380d)|

```
5/5 thresholded: 146 / 150 stocks (97.3%)
≥ 4/5:          149 / 150 stocks (99.3%)
≥ 3/5:          150 / 150 stocks (100.0%)
```

### 3.3 失敗 4 支詳情

| Stock | 名稱 | Tier / Industry | Missing |
|---|---|---|---|
| 3131 | 弘塑 | core / 其他電子類 | F_margin_240d |
| 4749 | 新應材 | convex / 半導體業 | V_rev_24m + F_margin_240d(新上市 / 報表期數不足)|
| 6640 | 均華 | core / 半導體業 | F_margin_240d |
| 6683 | 雍智科技 | core / 半導體業 | F_margin_240d |

**觀察**:3/4 缺 F_margin(融資券交易資料)— 推測為較小型股流動性不足之 margin trading 不活躍導致。**非資料抓取問題**(table 有資料但該股本身少 margin 活動)。新應材(4749)額外缺 V_rev_24m 反映**新上市股本身月報期數 < 24 個月**。

---

## 4. §0.2 Pareto Coverage 結果

```
All 150 stocks in core_universe / convex_universe tier: 100% (by snapshot definition)
```

依 doctrine,§0.2 判準為「stock 是否在 core/convex tier」— 此為 `core_universe_membership` snapshot 之 definitional 屬性,本 probe 之 150 stocks 全數 by-def 滿足。

---

## 5. §0.3 K-wave Market-Level Coverage(本機 DB 落後狀態)

### 5.1 預期 5 indicators(per §14.7-BR Phase C-4 = 5-of-5)

| Indicator | 預期來源 | 本機 DB 狀態 |
|---|---|---|
| M2SL | FredData(per §14.7-BR Phase C-1)| ❌ 缺 |
| T10Y2Y | FredData | ✅ 12,491 rows |
| VIXCLS | FredData | ✅ 9,191 rows |
| TW_SEMI_VWAP_YOY | kwave_supply_cycle_proxy(per §14.7-BR Phase C-2)| ❌ table 不存在 |
| TW_SHIPPING_VWAP_YOY | kwave_supply_cycle_proxy(per §14.7-BR Phase C-4)| ❌ table 不存在 |

```
本機 DB market-level §0.3: 2 / 5 = 40%
```

### 5.2 解釋

§14.7-BR Phase C-1 / C-2 / C-4 之資料 sync + table 建置在**另一台機**完成(charter inscribed;commits in repo);**本機 DB 尚未 catch-up sync**。對齊 §14.7-AX(E) 外部資源 protocol — 此為 timing 而非 tier 問題,跑相應 sync script 即補齊。

### 5.3 §0.3 判準對 stock-level 之意義

§0.3 為 market-level signal(K-wave 春初訊號共振等),不應 per-stock 評估。本 probe 將其報為「market context 是否存在」之 binary 維度。若採嚴格 5-of-5,本機尚不及格;若採 partial(至少 1 indicator),通過。

---

## 6. 119 vs 150 Bracket

| 數字 | 來源 | 判準 | 解釋 |
|---|---|---|---|
| 150 | DB committed v0.2(2026-05-24)| CoreScore top-N(hardcode 150 era)| Current selection algorithm 產出 |
| 146 | 本 probe 5/5 realistic threshold | raw data 240d/8q/24m availability | 150 中 4 支因 margin / 月報期不足被排除 |
| 119 | feature_store v0.8 dry-run([feature_store_v08_implementation_audit_20260526.md](feature_store_v08_implementation_audit_20260526.md))| feature_store v0.8 builder 之 31 features × per-stock 最小 history | 比 raw existence 嚴(需 derivable features;含 ΔlnP 變異數 / FG 11-sub-score / IF 9-sub-score 等) |

**Bracket 推論**:
- 150 → 146(− 4):**realistic raw-source threshold** 過濾(4 支 missing margin / revenue)
- 146 → 119(− 27):**feature-derivable threshold** 過濾(更嚴 builder 計算性要求,如歷史長度、滯後窗口、NaN 容忍度)
- 從 150 到 119 之差(31 支)分布於兩個 quality layer:**raw availability(4 支)+ feature derivability(27 支)**

---

## 7. 結論

### 7.1 答用戶判準問題

依用戶 doctrine「三基柱皆具備對應資料來源依據」之嚴格 reading:

- §0.1 first-principle raw source: 150 / 150 全過(Phase A §7 hook 設計)或 146 / 150 全過(realistic threshold)
- §0.2 Pareto: 150 / 150 by-def 全過
- §0.3 K-wave market context: 本機 2/5(timing gap;另機已 5/5);doctrine 為「市場有 K-wave 訊號可用」非「per-stock K-wave 資料」

**Strict 三基柱皆過(本機)**:
- 若 §0.3 partial(≥ 1 indicator)acceptable:**146 stocks pass realistic / 150 pass loose**
- 若 §0.3 strict 5/5:0 pass(因 timing gap)

### 7.2 對 §14.7-BU Phase E 落地之 implications

1. **per-stock 完整性追蹤 layer 設計合理**:本 probe 已驗證 §0.1 source 存在性與 threshold-based derivability 為兩層獨立 criterion,Phase E 4 builders hook 設計可分別寫入(data layer hook 採 source 存在性 / feature layer hook 採 derivability)
2. **§0.3 layer hook 應採 market-level**(non-per-stock);per-stock 寫入時 §0.3 pillar 之 expected_items / actual_items 應為「market 指標 N 個中 actual 幾個」之 broadcast 值
3. **本機 vs production sync 落後**為跨機 timing,不影響 schema / doctrine 設計;Phase E 落地前可選擇先 catch-up sync 或 accept partial 狀態

### 7.3 對 charter cross-references 之 implications

§14.7-BU 第十九輪 entry T_BU-5(119 stocks × 12 cells = 1,428 records)反映**v0.8 dry-run 之 feature-derivable 數**;若採 raw existence(Phase A §7 設計),expected = current N × 12 = **150 × 12 = 1,800 records**(committed snapshot 階段)或 **119 × 12 = 1,428**(v0.8 commit 後)。Phase E 落地時 cardinality 須 match 當下 committed snapshot 之 N。

**非 charter 錯誤**;為「不同 N policy / 不同 derivability layer」並存之自然多態。

---

## 8. 配套

本 probe 為**一次性 evidence**,非 permanent script;若日後欲 Phase E formal 落地,參見:
- [universe_completeness_governance_design_research_20260526.md](universe_completeness_governance_design_research_20260526.md) §7 hook 設計
- [scripts/maintenance/audit_universe_completeness.py](../scripts/maintenance/audit_universe_completeness.py)(已落地 v0.1;C5-C12 待 Phase E 寫入後實質驗證)
- [scripts/core/universe_completeness_schema.py](../scripts/core/universe_completeness_schema.py)(schema 已 init 完成)

---

**作者**: Claude
**Status**: ✅ 一次性 evidence preview / non-destructive / 不改 schema / 不改 code
