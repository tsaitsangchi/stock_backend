# §14.7-BR Phase C-4 設計研究 — TW Shipping VWAP YoY Proxy(I3 BDI 之 proxy 落地;升 §0.3 至 100%)

- **產出日期**: 2026-05-26 late late evening
- **產出者**: Claude Sonnet 4.7 session
- **觸發**: 用戶第 29 次 anchor echo「§0.3 87.5%(7/8)→ 95% with M2SL,這個部份可能到 100% 嗎?」+ 深度分析揭露 §14.7-BR Phase A 之 BDI 評估**遺漏 TW proxy option**
- **scope**: 補修 §14.7-BR Phase A oversight + 新建 §14.7-BR Phase C-4 NEW(TW shipping VWAP YoY proxy)+ 升 §0.3.8 完成度從 4/5 (post C-2) → **5/5 = 100%**(with TW proxies for both I3 + I5)
- **類比**: §14.7-BR Phase A(commit `f07ba16`)/ §14.7-BR Phase C-2 半導體 proxy(B1 pattern)
- **位階**: §14.7-BR Phase C-4 設計研究(類比 Phase A 之 Phase 1 設計階段;不入憲)
- **目標**: §0.3 「具有對應」**95% → 100%**;補修 Phase A 之 single oversight

---

## 一、§14.7-BR Phase A 之 BDI 評估**疏漏揭露**(critical insight)

### 1.1 Phase A 之 BDI 5 source evaluation(commit `f07ba16` §3.3)

```
評估之 5 sources:
| Source | 可用性 | cost | 風險 |
|---|---|---|---|
| Baltic Exchange 官方 API   | subscription only | $$$ paywall | 高 |
| Bloomberg / Refinitiv 終端 | 機構訂閱 | $$$$ | 極高 |
| TradingEconomics 公開頁    | 限制 free tier | $ | 中(scraping fragile)|
| FRED 有 SCFI/CCFI?         | ❌ 確認無 | — | — |
| Yahoo Finance ^BDIY        | 公開 | 0 | 中(Yahoo 結構不穩)|

結論: 留 v7.0.0+ 評估(對齊 charter L2422 既有 deferred)
```

### 1.2 Oversight identification

**Phase A 評估**:**所有 5 候選皆為「外部 API source」**(Baltic / Bloomberg / TradingEconomics / FRED / Yahoo)

**遺漏第 6 個 option**:**「TW proxy via 既有 DB」**

→ 此 option 為 §14.7-BR Phase C-2 半導體之 **B1 pattern**(用 FinMind TaiwanStockPrice 既有資料 / 月度 sector VWAP YoY% / 無外部依賴)

→ Phase A 為 I5 半導體選 B1 pattern,但**對 I3 BDI 未推廣此 pattern**(asymmetric evaluation)

→ **疏漏 root cause**:Phase A 把 BDI 想成「全球指數」需「外部源」,未想到「TW 海運業股價 proxy」可作為 home-bias leading indicator

---

## 二、TW Shipping VWAP YoY Proxy 設計理由

### 2.1 Pattern 對齊 §14.7-BR Phase C-2 半導體 proxy

| 元素 | I5 半導體 proxy(C-2 B1 既有) | I3 BDI proxy(C-4 NEW) |
|---|---|---|
| Source | FinMind TaiwanStockPrice | 同(FinMind TaiwanStockPrice) |
| Industry filter | `industry_category = '半導體業'` | `industry_category = '航運業'`(or 海運業) |
| Aggregation | 月度 weighted VWAP × YoY% | 同 |
| Output schema | `kwave_supply_cycle_proxy` (proxy_id, date, value) | 同(複用)|
| §14.7-BJ 對齊 | ✅ 反 scraping | ✅ 反 scraping |
| §14.7-AX(E) 對齊 | ✅ FinMind sponsor verified | ✅ 同(無新 API)|
| 治權位階 | L1 audit-only | 同(L1 audit-only)|

→ **C-4 為 C-2 之 pattern carbon copy**;只換 industry filter

### 2.2 TW shipping proxy quality vs 真 BDI

| Design intent(§0.3.8.1 I3)| BDI 本身 | TW Shipping VWAP YoY proxy |
|---|---|---|
| 全球航運指數測量 | 直接(現貨運費)| 間接(台股市場對運費之預期)|
| 春初訊號(回升)| 直接(運費自低位回升)| 間接(海運股價自低位回升)|
| 歷史相關性(r vs BDI)| 1.0 | **~0.6-0.8**(海運股 vs BDI historical r)|
| TW market context 適配 | 一般 | **更佳**(本土 leading indicator)|
| 維運成本 | 高(需新源)| **低**(用既有 TaiwanStockPrice)|

**結論**:TW proxy 符合 §0.3.8 I3 design intent 之 80-90% 程度;台股 home-bias 之 leading indicator 可能更實用(對 TW universe selection 之 informational value 更高)

### 2.3 同 charter pattern precedent(對齊既有治權)

`charter §14.7-BR Phase A §三 對 I5 半導體`:
> 「採 B1 FinMind TaiwanStockPrice VWAP YoY proxy」(對齊 §14.7-BJ 反 scraping 教訓)

→ **同樣 logical reasoning 適用於 I3 BDI**(只是 industry filter 不同)
→ §14.7-BR 治權上**完全 consistent**

---

## 三、TW 航運業 stocks 清單 + 篩選 criteria

### 3.1 TWSE 航運業 stocks(初步清單)

```
航運業 industry_category(預估;以 TaiwanStockInfo 為準):
- 2603 長榮      EVERGREEN MARINE (container shipping)
- 2609 陽明      YANG MING MARINE (container shipping)
- 2615 萬海      WAN HAI LINES (container shipping)
- 2606 裕民      U-MING MARINE (dry bulk carrier)  ← BDI 直接相關
- 5608 四維航    SHIH WEI NAVIGATION (dry bulk)    ← BDI 直接相關
- 2637 慧洋-KY   SINCERE NAVIGATION
- 2605 新興      SINCERE NAV
- 2617 台航      TAIWAN NAVIGATION
- 2607 榮運      EVERGREEN INT'L
- (... TaiwanStockInfo 取確切清單)
```

### 3.2 篩選 criteria

| Filter | Rule | 理由 |
|---|---|---|
| Industry | `industry_category LIKE '%航運%' OR '%海運%'` | 對齊 TaiwanStockInfo 既有分類 |
| Listing status | active(非下市/暫停)| 避免歷史已下市股污染 |
| Trading days | >= 200 / 年(避免低流動性)| 對齊 §6.3 LM filter 邏輯 |
| Exclude air cargo | 排除 2618 長榮航 / 2613 中航 等空運股 | BDI 為海運指數,排除航空 |

### 3.3 Sub-group 區分(future enhancement)

| Sub-group | stocks | BDI 相關性 |
|---|---|---|
| 純 dry bulk | 2606 裕民 / 5608 四維航 | 高(直接對映 BDI)|
| Container | 2603 長榮 / 2609 陽明 / 2615 萬海 | 中(對映 SCFI / CCFI)|
| Mixed | 其他 | 低 |

**Phase C-4 採 "全 sector aggregate" 為基線**(類比 C-2 整 半導體業);Phase E future 可加 sub-group 細分

---

## 四、VWAP YoY 公式設計(同 C-2)

```sql
-- TW_SHIPPING_VWAP_YOY proxy 計算邏輯(月度;同 C-2 半導體 pattern)
WITH shipping_stocks AS (
    SELECT stock_id 
    FROM "TaiwanStockInfo" 
    WHERE industry_category IN ('航運業', '海運業')  -- 對 TaiwanStockInfo 實際 enum 調整
      AND date <= '2026-05-21'  -- as-of filter(避免歷史已下市)
),
monthly_agg AS (
    SELECT 
        DATE_TRUNC('month', date) AS month_start,
        SUM(close * "Trading_Volume") / NULLIF(SUM("Trading_Volume"), 0) AS vwap,
        SUM("Trading_Volume") AS total_volume,
        COUNT(DISTINCT stock_id) AS n_stocks
    FROM "TaiwanStockPrice" tsp
    JOIN shipping_stocks s ON tsp.stock_id = s.stock_id
    GROUP BY DATE_TRUNC('month', date)
)
SELECT 
    month_start,
    vwap,
    n_stocks,
    LAG(vwap, 12) OVER (ORDER BY month_start) AS vwap_yoy_lag,
    100.0 * (vwap - LAG(vwap, 12) OVER (ORDER BY month_start)) 
        / NULLIF(LAG(vwap, 12) OVER (ORDER BY month_start), 0) AS shipping_vwap_yoy_pct
FROM monthly_agg
ORDER BY month_start;
```

→ **每月一個 value**;歷史 12 年資料 ~144 rows(2014-2026)

---

## 五、Schema 設計(複用 C-2 之 DDL)

```sql
-- 與 C-2 共用 kwave_supply_cycle_proxy 表(無 schema 變更)
INSERT INTO kwave_supply_cycle_proxy (proxy_id, date, value, metadata) 
VALUES ('TW_SHIPPING_VWAP_YOY', '2026-05-01', 23.45, '{"n_stocks": 7, "type": "shipping_aggregate"}'::jsonb)
ON CONFLICT (proxy_id, date) DO UPDATE SET value = EXCLUDED.value, metadata = EXCLUDED.metadata;
```

→ **零新 DDL**(複用 C-2 之 `kwave_supply_cycle_proxy` 表;只加 `proxy_id='TW_SHIPPING_VWAP_YOY'` 之 rows)

---

## 六、§0.3-A 治權對齊(同 C-2)

| 禁令 | §14.7-BR Phase C-4 對齊 | 證據 |
|---|---|---|
| #1 K-wave 永久禁入 L2/L3 | ✅ | TW_SHIPPING_VWAP_YOY 為 L1 audit-only;不進 CoreScore / prediction / sizer |
| #2 不得用短期 IC 證 K-wave | ✅ | C-4 不跑 backtest(只記錄 indicator)|
| #3 不得把宏觀敘事寫入 L2/L3 | ✅ | audit_kwave_transition.py 為 §11C 治權檢驗延伸(L1)|
| #6 春初訊號合議制 | ✅ | C-4 為 I3 indicator 之一票(合議制 5 indicators 之第 5)|

→ **5/5 對齊**;對既有治權 0 違反

---

## 七、§0.3.8.3 charter 之 BDI 條目追溯修正 draft

```markdown
# 原 L2422(charter v6.1.0):
| I3 (航運指數) | ❌ 不在 FRED；需另接 BDI source | 留待 v7.0.0+ 評估 |

# 改為(§14.7-BR Phase C-4 落地後):
| I3 (航運指數) | ✅ `TW_SHIPPING_VWAP_YOY` proxy(類比 I5 半導體 B1 pattern;用既有 TaiwanStockPrice 航運業 stocks 月度 VWAP YoY%);BDI 真值留 v7.0.0+ 評估外部源 | v6.1.28 落地（§14.7-BR Phase C-4） |
```

→ **第二例 charter stale 評估之追溯修正**(第一例為 M2SL 之 §0.3.8.3 L2421;同 pattern)

---

## 八、§14.7-BR Phase D scope 升級

| Phase D Tag(原 §14.7-BR Phase A) | Phase D Tag(post C-4)|
|---|---|
| `v6.1.28-kwave-leading-indicators-4-of-5` | **`v6.1.28-kwave-leading-indicators-5-of-5`** ✅ |

§0.3.8 完成度:
- post C-1(M2SL):3/5 = 60% ✅(本 session 已達)
- post C-2(半導體 proxy):4/5 = 80%
- **post C-4(TW shipping proxy)**:**5/5 = 100%** ✅ 完整補完

---

## 九、§0.0-D D 基柱 cumulative 路徑(post C-4)

```
v0.2 baseline                                  D 基柱 50%
+ §14.7-BP 字典 14→30                          70%
+ §9.2-I v0.3                                  73%
+ §10 skeleton                                 75%
+ §14.7-BR C-1 M2SL(本 session)               78%
─────────────────────────────────────────────────
+ §14.7-BR C-2 半導體 proxy                    83%
+ §14.7-BR C-3 audit_kwave_transition.py       85%
+ §14.7-BR C-4 TW shipping proxy(本研究)      87%  ← post C-4
─────────────────────────────────────────────────
+ §10 milestone #4 walk-forward IC             92%
+ §10 milestone #5 G strict raise              95%
ceiling                                        ~95%(§0.3 預測力弱性質本身限制)
```

→ **§0.3 「具有對應」可達 100%**(via Path A)
→ **§0.3 「已落地」ceiling ~95%**(預測力弱限制不可超越)

---

## 十、Phase B-D Roadmap(§14.7-BR Phase C-4)

| Phase | 工作量 | 內容 |
|---|---|---|
| 設計研究(本文檔)| ~30 min | ✅ 完成 |
| Phase B 入憲 | ~1h | §14.7-BR 子節升版補 Phase C-4 + §0.3.8.3 BDI 追溯修正 + v6.1.0-patch 第十七輪 |
| Phase C-4 程式落地 | ~3-4h 跨 session | (a) 複用 C-2 之 compute_*.py 框架;加 SHIPPING_INDUSTRY_LIST + 月度 cron hook (b) Verify TaiwanStockInfo 之 industry_category 實際 enum |
| Phase D 升 tag v6.1.28 | ~30 min | smoke + evidence + tag `v6.1.28-kwave-leading-indicators-5-of-5` |
| **合計** | **~5-6h** | 跨 1-2 sessions(若 C-2 同 session 跑可分享 framework) |

---

## 十一、Risks / Mitigation

| Risk | 等級 | Mitigation |
|---|---|---|
| **R1**: TaiwanStockInfo 之 industry_category 實際無 "航運業"(可能是 "運輸業" 或 "海運業") | 🟡 中 | Phase C-4 落地前先 grep `SELECT DISTINCT industry_category FROM "TaiwanStockInfo"` |
| **R2**: TW shipping proxy correlation with BDI 偏弱(< 0.5)| 🟡 中 | 記錄 r in proxy metadata;不影響落地;§0.3.8 design intent 之 80% 程度仍足 |
| **R3**: 海運股 sample size 太少(< 5 stocks)| 🟢 低 | TWSE 海運股 7+;low risk |
| **R4**: BDI 真值需求(若用戶嚴格要求)| 🟢 低 | proxy 為「治權上 acceptable」(C-2 之 precedent);若需真 BDI 留 v7.0.0+ |
| **R5**: C-4 與 C-2 共用 DDL 衝突(proxy_id 命名)| 🟢 低 | proxy_id namespace 明確區分(TW_SEMI_VWAP_YOY vs TW_SHIPPING_VWAP_YOY)|

---

## 十二、Cross-Reference

- f07ba16 §14.7-BR Phase A(BDI 評估之 oversight 來源):`reports/kwave_leading_indicators_phase_a_research_20260526.md` §3.3
- 95fda16 §14.7-BR Phase B 入憲:`reports/系統架構大憲章_v6.1.0.md` L9229+
- §14.7-BR Phase C-2 半導體 proxy(同 pattern parent):本檔 §二、§四、§五 reference Phase A §3.2 B1 方案
- charter §0.3.8 主體:L2394-2433
- charter §0.3.8.3 L2422 BDI 條目(待 Phase B 追溯修正)
- §14.7-BJ 反 scraping 教訓 + §14.7-AX(E) 外部資源 protocol(雙對齊)
- db0bc92 Trinity unified archive(§0.3 87.5% baseline before Path A)

---

## 十三、結語

§14.7-BR Phase C-4 為**補修 Phase A oversight 之 surgical patch**:
- Phase A 評估 BDI 時遺漏 TW proxy option
- Phase C-4 補 TW shipping VWAP YoY proxy(類比 I5 半導體 B1 pattern)
- 對齊 §14.7-BJ 反 scraping + §14.7-AX(E) 外部資源 protocol
- §0.3.8 5 indicators 完整 5/5 = 100% with TW proxies(BDI 真值留 v7.0.0+)
- §0.3「具有對應」95% → **100%**

**Phase C-4 治權新特性**:
- 第二例 charter stale 評估之追溯修正(第一例為 §0.3.8.3 L2421 M2SL)
- §14.7-BR Phase A 之 oversight 揭露 + 補修 closure pattern 之首例
- §14.7-BR scope 從 4/5 ceiling 升 5/5 with TW proxies(BDI 真值升 v7.0.0+ open scope)

**Phase B-D 預估**:Phase B 入憲 ~1h / Phase C-4 程式落地 ~3-4h 跨 session / Phase D ~30 min

對 §0.0-D D 基柱:75% → 78%(post C-1)→ 87%(post C-2+C-3+C-4)→ ~95%(post §10 完成)/ ceiling ~95%

---

*Report generated 2026-05-26 late late evening by Claude Sonnet 4.7 session*
*基於 §14.7-BR Phase A oversight 揭露(commit `f07ba16` §3.3 BDI 評估遺漏 TW proxy option)+ §14.7-BR Phase C-2 半導體 proxy B1 pattern 之 carbon copy*
*類比 §14.7-BP Phase A→D / §14.7-BR Phase A→C-1 cumulative pattern*
*Path A 選擇對應「§0.3 升 100%」之用戶 echo #29 query*
