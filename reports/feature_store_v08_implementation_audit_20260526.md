# Feature Store v0.8 Universe Rebuild — Empirical Evidence Report

**日期**: 2026-05-26
**對應**: handoff v6 §七 priority #2 / Path A 路徑落地 / v6.3.1 軌道
**模組**: `scripts/core/feature_store_builder.py` v0.5
**對接 universe**: `core_universe_20260521_core_universe_policy_v0_8_dynamic` (119 stocks; 83 core + 36 convex)
**Feature Set**: `feature_set_v0.4` → `feature_set_id = fs_20260521_feature_set_v0_4`
**主權判定**: ✅ **PERFECT** (14/0/0 preflight + 35 features defined + 3689 value rows / 4.3s)

---

## 一、執行摘要

本報告記錄 v6.3.0 `core_universe_20260521_core_universe_policy_v0_8_dynamic` snapshot landing 後,Feature Store 對接 v0.8 動態 universe 之首次 dry-run 實證。**0 行程式碼修改** — feature_store_builder v0.5 透過 `SELECT MAX(as_of_date)` dispatch 自動拾取 v0.8 snapshot,driver 119 stocks(83 core + 36 convex)無縫對接。

### 關鍵讀數

| 指標 | 值 | 對映 |
|---|---|---|
| Preflight PASS/WARN/FAIL | **14/0/0** | §5.6.3 動態判定 |
| Universe snapshot resolved | `core_universe_20260521_core_universe_policy_v0_8_dynamic` | §6.7.1 dynamic annex |
| Policy version | `core_universe_policy_v0.8_dynamic` | §14.7-BT Phase D-2 |
| Stocks loaded | **119** (= 83 core + 36 convex) | §0.2 八二法則 top 5% |
| Features defined | **35** (31 active v0.3 set + 4 v0.2 interaction archived) | §8.2.2 |
| Value rows (would write) | **3,689** (= 119 × 31 active) | §8.2 治理表 |
| Null imputed | 0 | §8.5 anti-leakage |
| Verdict | PERFECT | §5.6.3 |
| Total time | 4,296 ms | — |

---

## 二、Feature Set v0.4 結構(35 features × 7 groups × 11 sources)

### 2.1 Group 分布

| Group | Active | Sources | 對應憲章基柱 |
|---|---|---|---|
| **price** | 12 | TaiwanStockPriceAdj | §0.1 ΔlnP 元素 + §9.9 v0.3 upside/downside |
| **liquidity** | 4 | TaiwanStockPriceAdj | §0.1 M 元素(成交量/市值代理) |
| **fundamental** | 4 | TaiwanStockMonthRevenue / TaiwanStockFinancialStatements | §0.1 V 元素 |
| **institutional** | 5 | TaiwanStockInstitutionalInvestorsBuySell / TaiwanStockMarginPurchaseShortSale | §0.1 F 元素 |
| **theme** | 2 | TaiwanStockInfo × THEME_KEYWORDS | §0.3 MBNRIC I/N tag |
| **macro** | 4 | FredData (DFF/VIX/T10Y2Y/UNRATE) | §0.3 康波週期 4 條 |
| **interaction** (archived) | 4 | FRED × PriceAdj / FRED × FinStmt / Info × PriceAdj / Info × Institutional | §0.3 × §0.1 cross-pillar |

### 2.2 Source 分布(11 distinct sources)

| Source | Feature count |
|---|---|
| TaiwanStockPriceAdj | 16(price 12 + liquidity 4) |
| TaiwanStockInstitutionalInvestorsBuySell | 4 |
| FredData | 4 |
| TaiwanStockFinancialStatements | 2 |
| TaiwanStockMonthRevenue | 2 |
| TaiwanStockInfo | 2 |
| TaiwanStockMarginPurchaseShortSale | 1 |
| FredData × TaiwanStockPriceAdj (interaction) | 1 |
| FredData × TaiwanStockFinancialStatements (interaction) | 1 |
| TaiwanStockInfo × TaiwanStockPriceAdj (interaction) | 1 |
| TaiwanStockInfo × TaiwanStockInstitutionalInvestorsBuySell (interaction) | 1 |

---

## 三、三基柱 DB 對應(feature_set_v0.4 level)

### 3.1 §0.1 第一性原理 — **覆蓋 ~85%**

| 元素 | feature_set_v0.4 對應 | 狀態 |
|---|---|---|
| **M(市值/規模)** | liquidity 群: `avg_daily_value_log_60d/252d`(美元成交量代理 log10) | ✅ FULL |
| **V(週轉/盈利)** | liquidity: `turnover_mean_60d`, `zero_volume_ratio_252d`;fundamental: `revenue_yoy_12m/3m`, `eps_sum_4q`, `net_income_positive_ratio_8q` | 🟡 ~70%(CashFlow ⏸) |
| **F(外力/籌碼)** | institutional: `foreign_net_20d/60d`, `trust_net_20d/60d`, `margin_ratio_60d` | ✅ FULL |
| **ΔlnP(價格變化)** | price: `log_return_20d/60d/252d`, `volatility_60d/252d`, `ma_ratio_20/60`, `max_drawdown_252d`, `upside_volatility_60d`, `downside_volatility_60d`, `upside_capture_60d`, `downside_capture_60d` | ✅ FULL(v0.3 upside/downside 已落地) |
| **時間單向性** | SQL gate via `build_publication_date_gate()`(§8.5-9 Phase 2;v0.4+) | ✅ FULL |

### 3.2 §0.2 八二法則 — **覆蓋 100%**

| 機制 | 對應 | 狀態 |
|---|---|---|
| **Top 5% 集中** | Universe input itself: 119 stocks(top ~4.3% × 2,767 candidates) | ✅ via v0.8 snapshot |
| **Core/Convex 70/30 split** | core_tier filtering(`core_universe ∪ convex_universe`) | ✅ via builder L262 |

註: §0.2 不需 feature 層級對應(已 embedded in universe selection 上游)。

### 3.3 §0.3 康波週期 — **覆蓋 ~60%(feature_set lag)**

| 元素 | feature_set_v0.4 對應 | DB 可用 | feature 層級狀態 |
|---|---|---|---|
| **M (Monetary) - DFF** | macro: `macro_dff_level` | ✅ FredData (26,258) | ✅ FULL |
| **M (Monetary) - M2SL_YoY** | ❌ 未加入 v0.4 | ✅ fred_series (435) | ⏸ feature lag(§14.7-BR DB 已備但 feature_set 未升 v0.5) |
| **B (Bonds) - T10Y2Y** | macro: `macro_t10y2y_level` | ✅ FredData (12,491) | ✅ FULL |
| **N (Natural) - TW_SEMI_VWAP_YOY** | ❌ 未加入 v0.4 | ✅ kwave_supply_cycle_proxy (369) | ⏸ feature lag |
| **N (Natural) - TW_SHIPPING_VWAP_YOY** | ❌ 未加入 v0.4 | ✅ kwave_supply_cycle_proxy (401) | ⏸ feature lag |
| **R (Risk) - VIXCLS** | macro: `macro_vix_level` | ✅ FredData (9,191) | ✅ FULL |
| **I (Industry) - MBNRIC tag** | theme: `theme_strength`, `theme_is_semiconductor` | ✅ TaiwanStockInfo + THEME_KEYWORDS | ✅ FULL |
| **C (Civilization)** | (interaction archived: theme × log_return, theme × foreign_net) | — | 🟡 PARTIAL(cross-pillar interactions in v0.2 archived,not active) |

**Empirical 治權現況**: DB 已有 5/5 leading indicators(per §14.7-BR Phase A-D complete),但 **feature_set_v0.4 只 consume 2/5**(VIXCLS + T10Y2Y)+ DFF/UNRATE 兩個 K-wave 不直接相關之指標。**Feature set lag** = next session priority #3 candidate(升 v0.5 feature_set + ablation)。

---

## 四、Backward-Compat 驗證(關鍵設計成功點)

### 4.1 0 行程式碼修改

**feature_store_builder.py v0.5** 對接 v0.8 dynamic snapshot **無需任何修改**,原因:

```python
# scripts/core/feature_store_builder.py L235-273
cur.execute("""
    SELECT s.snapshot_id, s.policy_version, s.source_data_cutoff
    FROM "core_universe_snapshot" s
    WHERE s.status = 'committed'
      AND s.as_of_date = (
          SELECT MAX(as_of_date) FROM "core_universe_snapshot" WHERE status = 'committed'
      )
    ORDER BY s.created_at DESC LIMIT 1
""")
# ... 自動拾取 v0.8 snapshot(2026-05-21 為唯一 committed snapshot;
# v0.2-v0.7 legacy 已於 §14.7-BT Phase D-1 deprecate)

cur.execute("""
    SELECT DISTINCT m.stock_id
    FROM "core_universe_membership" m
    JOIN "core_universe_snapshot" s ON s.snapshot_id = m.snapshot_id
    WHERE s.status = 'committed'
      AND m.core_tier IN ('core_universe', 'convex_universe')
      ...
""")
# ... 自動取 83 + 36 = 119 stocks(non-hardcoded)
```

### 4.2 動態 universe size 容差

- L270-273: `if len(self.core_stocks) < 100`: WARN 觸發(threshold = 100,non-hardcode 150)
- 119 stocks > 100 → **PASS** (non-WARN)
- 設計上對齊 §6.7.1 dynamic annex 之 N_min ≥ 100 邊界

### 4.3 對 audit_core_universe v0.3 之同步

audit v0.3 已認得 `core_universe_policy_v0.8_dynamic` policy 對應之 score_scope `v0.10_dynamic_universe_via_top_pct_composite_corescore`(per `reports/core_universe_audit_20260526_1603.md`)。Feature Store builder 對該 snapshot 之 read 行為與 audit 之 read 行為一致(同樣走 `SELECT MAX(as_of_date)` dispatch)。

---

## 五、治權誠實三軸

依 CLAUDE.md §一 #8 報告誠實 + §0.0-G Level 1 紀律:

| 軸 | 狀態 | 說明 |
|---|---|---|
| **Implementation completion** | ✅ **100%** | feature_store_builder v0.5 + feature_set_v0.4 對接 v0.8 universe → PERFECT dry-run(14/0/0,4.3 秒,0 程式碼修改) |
| **Feature-set ↔ Charter level alignment** | 🟡 **~80%** | §0.1 V 元素 CashFlow ⏸ + §0.3 5/5 leading indicators 中 3/5 未進 feature_set(M2SL/TW_SEMI/TW_SHIPPING DB 已備 but feature lag) |
| **Production-empirical IC validation** | ⏸ **0%** | 需後續 cascading: feature_store commit + model retrain + walk-forward + IC ≥ baseline 才能宣稱有效 |
| **Epistemological ceiling** | ~95% | §0.3 K-wave decades-level constraint(per §14.7-BR honest amendment) |

---

## 六、已 inventoried 之 known gaps(non-FAIL but pending)

| Gap | 影響 | 對應 next session priority | 對應憲章 |
|---|---|---|---|
| G1: M2SL_YoY 未進 v0.4 feature_set | §0.3 M 元素 feature 層級空缺 | next++ priority(升 feature_set_v0.5) | §14.7-BR Phase C-1 |
| G2: TW_SEMI_VWAP_YOY 未進 v0.4 | §0.3 N 元素 feature 層級空缺(半導體 supply cycle) | next++ priority(升 feature_set_v0.5) | §14.7-BR Phase C-2 |
| G3: TW_SHIPPING_VWAP_YOY 未進 v0.4 | §0.3 N 元素 feature 層級空缺(BDI proxy) | next++ priority(升 feature_set_v0.5) | §14.7-BR Phase C-4 |
| G4: CashFlow features 未進 v0.4 | §0.1 V 元素覆蓋 ~70% 非 100% | next priority(待 §14.7-BO Phase B sync) | §14.7-BO |
| G5: Feature commit (`--commit`) 尚未跑 | feature_set 仍 dry-run only;下游 model 無法 read | next priority #3 | handoff v6 §七 #3 |

**Gap 處置 framework**: G1-G3 為 feature-set 層升版(需 ablation),G4 為 V 補強(需 DB sync),G5 為本路徑後續(commit + cascading rebuild)。

---

## 七、Phase 路徑 closure / open

| Phase | 狀態 | 證據 |
|---|---|---|
| Phase 1: 探查 feature_store_builder v0.5 現況 | ✅ COMPLETED | 本報告 §四 |
| Phase 2: init feature_store_schema(3 表) | ✅ COMPLETED | preflight 14/0/0 |
| Phase 3: dry-run on v0.8 universe(119 stocks) | ✅ COMPLETED | verdict PERFECT |
| Phase 4: 寫 evidence report | ✅ COMPLETED | 本報告 |
| Phase 5: commit + tag v6.3.1-feature-store-v08-rebuilt | ⏳ pending(本 session) | — |
| Phase 6: production commit(`--commit`)+ cascading rebuild | ⏸ next session | handoff v7 |
| Phase 7: feature_set v0.4 → v0.5 升版(加 M2SL + TW proxies + CashFlow) | ⏸ next++ session | next++ |

---

## 八、結論

**Quantum Finance Feature Store v0.8 universe rebuild(dry-run level)正式達成 PERFECT 主權判定**。

- ✅ 0 行程式碼修改(feature_store_builder v0.5 backward-compat 設計成功)
- ✅ 119 stocks 自動 resolve(via `SELECT MAX(as_of_date)` dispatch)
- ✅ 35 features defined / 3,689 value rows / 4.3 秒
- ✅ §0.1 ~85%(M ✅ / F ✅ / ΔlnP ✅ / V 🟡 ~70%)
- ✅ §0.2 100%(via universe input)
- 🟡 §0.3 ~60%(macro+theme covered;5/5 leading indicators 中 2/5 在 feature_set,3/5 DB ready 但 feature lag)

**Implementation milestone v6.3.1 達成**;production-empirical 與 feature_set v0.5 升版為 next/next++ session priority。

---

## 九、Cross-references

- 對應憲章 §6.7.1 / §14.7-BT(dynamic universe annex)
- 對應憲章 §8.2 / §8.5-9(feature store + publication-date discipline)
- 對應憲章 §14.7-BR(K-wave 5/5 leading indicators DB)
- 對應憲章 §14.7-BO(CashFlow Phase B ⏸)
- handoff: `reports/session_handoff_20260526_v6_3_0.md` §七 #2
- 設計研究: `reports/feature_store_builder_first_principles_research_20260519.md`
- v0.3 audit: `reports/feature_store_v03_implementation_audit_20260520.md`
- core_universe audit: `reports/core_universe_audit_20260526_1603.md`

---

**報告作者**: Claude(per Path A 本 session 延長 marathon)
**主權判定**: ✅ PERFECT dry-run(implementation level)
**Next milestone**: production commit + cascading model retrain(handoff v7 priority #1)
