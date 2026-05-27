# Macro Pillar Split — Phase A Design Research(§14.7-BY Phase F)

**日期**: 2026-05-27
**Phase**: A(設計研究 / pre-charter inscription)
**對應軌道**: v6.2.0 或 v7.0.0(macro infrastructure 層 / 三層 multi-cycle hierarchy 升 charter pillars)
**對應憲章基礎**: §0.3 康波週期 / §0.3-A 7 禁令 / §14.7-BR Phase C-4 / §14.7-BU governance / §14.7-BY Path A 已落地(11 P0 + 2 P1 = 13 indicators / 5 大 driver category 5/5 完整)
**Status**: ✅ Phase A 完整(15 章 / non-destructive / 不動 DB 不動 code)
**對應 user trigger**: 2026-05-27 Phase E closure 後選 Path C 分層升版(§0.3 / §0.4 / §0.5 三 sub-pillars)
**前置基線**: v6.1.32(Phase E closure / K-wave 13 indicators / 5 大 driver coverage 5/5)

---

## 1. 觸發

§14.7-BY Phase E(v6.1.32 / 2026-05-27)落地後,**Path A+(strict K-wave + P1 補強)已 closure**:13 indicators / 5 大 driver coverage 5/5。但 Phase A research §8.3 中之 **Path C 分層升版**仍為 future option;Phase E 之 trade-off(multiplier 8.80 → 8.15 因 P1 拉低 avg)實際上指向 Path C 之根本解:**將 mix indicators 拆為三層獨立 sub-pillars,各層名實相符,各自 audit + governance**。

### 1.1 Phase E 揭露之 trade-off 之根本原因

| 觀察 | 解釋 |
|---|---|
| 加 2 P1 indicators 後 avg K-wave % 從 80% 降至 63% | P1 之 QUSPAM770A(80%)+ WTISPLC(70%)拉低均值;但更深層原因為「**K-wave + non-K-wave indicators 混在同一 pillar**」之 mix |
| Simple avg 不公平 | M2SL(70%)與 VIXCLS(10%)在同一 §0.3 之 "kondratiev" pillar 內 avg → 失去 K-wave 與 microstructure 之區分 |
| Path E 治權價值在 5/5 driver coverage 而非 multiplier | 因為 Path A+ 之**精神**仍是「single pillar 內 mix」,multiplier 本就不是純 K-wave 純度 |

### 1.2 Path C 之 first-principle 解決方案

**將 §0.3 之 mix 拆為三層 sub-pillars,每層獨立 K-wave / Juglar/Kuznets / Microstructure**:
- §0.3 **K-wave**(40-60 年 / 純 macro structural)
- §0.4 **Multi-cycle**(7-25 年 / Juglar + Kuznets 中期循環)
- §0.5 **Microstructure**(月內 ~ 季 / short-term sentiment / market regime)

各層名實相符,各自 100% pure(K-wave pillar 平均 ≥ 75% / Multi-cycle pillar 平均 ≥ 70% / Microstructure 平均 ≥ 60%);各自獨立 builder gate / audit / governance。

---

## 2. 治權位階對齊(§14.7-BY Phase F 預定子節)

### 2.1 對映既有憲章節

| 節 | 對應內容 |
|---|---|
| §0.3 康波週期(原條文)| 名義為 40-60 年 K-wave;當前 mix 13 indicators(K-wave + Juglar + Kitchin + Microstructure)|
| §0.3-A 7 條禁令 | K-wave 不下沉 L2/L3 / 不作為 per-stock score |
| §14.7-BR Phase C-4 | 5 leading indicators baseline(M2SL/T10Y2Y/VIXCLS/TW_SEMI/TW_SHIPPING)|
| §14.7-BU universe_completeness_snapshot | 3 pillars × N stocks × 4 layers schema |
| §14.7-BY Phase A-E | 13 indicators / 5 大 driver coverage 5/5 |

### 2.2 §14.7-BY Phase F 預定子節之治權含義

本 Phase F research 若進 Phase B-style 入憲,將**新增憲章 §0.4 + §0.5 sub-pillars**:
- §0.3 升版為**純 K-wave**(只保留 ≥75% 對應度之 indicators)
- §0.4 新建為 **Multi-cycle Context**(收納 Juglar/Kuznets 中期循環 indicators)
- §0.5 新建為 **Microstructure**(收納短期 sentiment / market regime indicators)

對映 §14.7-BY 之 Indicator-axis 純化更深一層:從「indicator-level purity」升至「**pillar-level purity**」。

### 2.3 治權邊界(本 Phase F research 不踩之線)

- ❌ 不改 §0.1 / §0.2 / §0.3-A 7 條禁令字面
- ❌ 不改 §14.7-BW Pure Doctrine 之 N dynamic 邏輯
- ❌ 不改 §14.7-BX Weekly recommit 之 T-axis
- ❌ 不動 DB committed snapshot(v0.10_pure_doctrine_weekly / N=1,857)
- ✅ 提案 §0.4 / §0.5 新 sub-pillars(charter 升版)
- ✅ 提案 universe_completeness_snapshot schema 升 5 pillars(ALTER ENUM CHECK constraint)
- ✅ 提案 builder Stage 1 binary gate 拆 3 sub-stages

---

## 3. 學術 SSOT — 多週期 hierarchy 各層性質

### 3.1 完整 6 層週期 hierarchy(per Schumpeter 1939 / Perez 2002 + 後人延伸)

| 週期名 | 長度 | 驅動因素 | 性質 | 學派出處 |
|---|---|---|---|---|
| **Kondratiev wave** | **40-60 年** | 科技革命 / 信用大循環 / 人口結構 / 能源典範 / 長週期商品 | 結構性 | Kondratiev 1925 / Schumpeter 1939 / Perez 2002 |
| **Kuznets cycle** | 15-25 年 | 基建投資 / 不動產 / 人口遷移 / global trade | 半結構性 | Kuznets 1958 |
| **Juglar cycle** | 7-11 年 | 設備投資 / 信用循環 / business cycle | 中期 | Juglar 1862 / Modigliani 1961 |
| **Kitchin cycle** | 3-5 年 | 庫存週期 / 訂單堆積 | 短期 | Kitchin 1923 |
| **Trading cycles** | 月 ~ 季 | 季節性 / OPEX / 月底 rebalance | 微結構 | Roll 1984 / Lo-MacKinlay 1988 |
| **Microstructure** | 日內 ~ 分鐘 | Open/close auction / 流動性 / order flow | Market noise | Kyle 1985 / Glosten-Milgrom 1985 |

### 3.2 Path C 三層 sub-pillar 之學術對映

| Sub-pillar | 對應週期 | 學術 anchor |
|---|---|---|
| **§0.3 K-wave**(40-60 年) | Kondratiev only | 純 structural;對應 K-wave 學術定義 |
| **§0.4 Multi-cycle**(7-25 年) | Juglar + Kuznets | 中期循環;business cycle leading indicators |
| **§0.5 Microstructure**(月 ~ 季) | Kitchin + Trading + Microstructure | 短期 sentiment / sector cycle / market regime |

---

## 4. 三個 sub-pillar 之 boundary 定義

### 4.1 §0.3 K-wave(純 strucutural / 40-60 年)

**包括之 driver categories**(per §14.7-BY Phase E 之 5 大 driver):
- Technological(科技革命)
- Credit Long-cycle(信用大循環,~50-75 年)
- Demographics(人口結構,25-30 年代際)
- Energy paradigm(能源典範,50-70 年)
- Long-cycle Commodity(超週期商品,30-40 年)

**Threshold**:indicator 之 K-wave correspondence ≥ **75%**

### 4.2 §0.4 Multi-cycle(7-25 年)

**包括之 driver categories**:
- Juglar business cycle(7-11 年信用 / 設備投資)
- Kuznets cycle(15-25 年 基建 / 不動產)
- Global trade cycle(7-11 年 BDI 等貿易循環)
- Yield curve regime(7-11 年 inversion-to-normalization)

**Threshold**:indicator 之 K-wave correspondence **30-75%**(中期循環級別)

### 4.3 §0.5 Microstructure(月 ~ 季 / sentiment / regime)

**包括之 driver categories**:
- Volatility regime(VIX / market sentiment)
- Sector inventory(Kitchin 3-5 年 sector 庫存)
- Short-term flow(ETF / 法人 / 散戶情緒)
- Intraday liquidity(本 charter 不涉,留作 future expansion)

**Threshold**:indicator 之 K-wave correspondence **< 30%**(短期級別)

---

## 5. 當前 13 indicators 之 Path C 分配(實證)

依各 indicator 之真實所屬週期級別 + K-wave correspondence,分配到 3 sub-pillars:

| Indicator | K-wave % | 所屬週期 | Path C sub-pillar |
|---|---:|---|---|
| **§0.3 K-wave**(≥75%)|||
| PATENTUSALLTOTAL | 85% | Tech innovation / multi-decade | §0.3 K-wave |
| LFWA64TTUSA647N | 85% | Demographics / multi-decade | §0.3 K-wave |
| B985RC1Q027SBEA | 80% | Tech IP investment / multi-decade | §0.3 K-wave |
| SPPOPDPNDOLUSA | 80% | Demographics / multi-decade | §0.3 K-wave |
| QUSPAM770A | 80% | BIS Credit-to-GDP long cycle | §0.3 K-wave |
| TCMDO | 75% | US Total Credit long cycle | §0.3 K-wave |
| PALLFNFINDEXQ | 75% | CRB long-cycle commodity | §0.3 K-wave |
| **§0.4 Multi-cycle**(30-75%)|||
| M2SL | 70% | Monetary regime(Kuznets/Kondratiev edge)| §0.4 Multi-cycle |
| WTISPLC | 70% | WTI Oil(Juglar + commodity cycle)| §0.4 Multi-cycle |
| TW_SEMI_VWAP_YOY | 40% | Kitchin sector inventory(K5 edge)| §0.4 Multi-cycle |
| TW_SHIPPING_VWAP_YOY | 35% | Juglar global trade cycle | §0.4 Multi-cycle |
| T10Y2Y | 30% | Juglar yield curve leading | §0.4 Multi-cycle |
| **§0.5 Microstructure**(<30%)|||
| VIXCLS | 10% | Short-term volatility regime | §0.5 Microstructure |

**分配統計**:
- §0.3 K-wave: **7 indicators**(avg 80%)
- §0.4 Multi-cycle: **5 indicators**(avg 49%)
- §0.5 Microstructure: **1 indicator**(avg 10%)

### 5.1 Path C 之 pillar-level purity(VS Path A+ 之 single pillar mix)

| Pillar | indicators | avg K-wave % | Purity |
|---|---:|---:|---|
| **§0.3 K-wave**(post Path C)| 7 | **80%** | **🟢 純 K-wave** |
| **§0.4 Multi-cycle** | 5 | 49% | 🟡 中期 mix(by definition)|
| **§0.5 Microstructure** | 1 | 10% | 🔴 by definition |

**Path C 之治權精神**:**每層名實相符**;§0.3 之 K-wave 純度為 80%(不再被低週期 indicators 拉低);§0.4 / §0.5 各自獨立(不冒充 K-wave)。

---

## 6. universe_completeness_snapshot schema 升 5 pillars

### 6.1 現行 schema(per §14.7-BU)

```sql
-- pillar enum
CHECK (pillar IN ('first_principle', 'pareto', 'kondratiev'))
```

### 6.2 Path C 升版 schema

```sql
-- 升版 5 pillars(adds 'multi_cycle' + 'microstructure')
ALTER TABLE universe_completeness_snapshot
DROP CONSTRAINT IF EXISTS check_pillar_enum;

ALTER TABLE universe_completeness_snapshot
ADD CONSTRAINT check_pillar_enum 
CHECK (pillar IN ('first_principle', 'pareto', 'kondratiev', 'multi_cycle', 'microstructure'));
```

### 6.3 Materialized view 升版

```sql
-- universe_completeness_matrix_current 升 5 × 4 = 20 cells per stock
DROP MATERIALIZED VIEW universe_completeness_matrix_current;
CREATE MATERIALIZED VIEW universe_completeness_matrix_current AS
  SELECT stock_id, pillar, layer, expected_items, actual_items, completeness_pct
  FROM universe_completeness_snapshot
  WHERE universe_snapshot_id = (SELECT snapshot_id FROM core_universe_snapshot WHERE status='committed');
CREATE UNIQUE INDEX ON universe_completeness_matrix_current (stock_id, pillar, layer);
REFRESH MATERIALIZED VIEW universe_completeness_matrix_current;
```

### 6.4 對 §14.7-BU governance 之影響

- universe_completeness_snapshot per stock 從 12 cells(3 pillars × 4 layers)升 **20 cells**(5 pillars × 4 layers)
- 既有 data 之 backfill 處理:
  - 既有 `first_principle / pareto / kondratiev` records 保留不動
  - Phase F 程式碼升版後,新 builder commit 寫入 **5 pillars × N stocks** records
- Storage 估計:1857 stocks × 20 cells × 1 weekly snapshot = 37,140 rows/week(per §14.7-BX weekly recommit)

---

## 7. builder Stage 1 binary gate 拆 3 sub-stages

### 7.1 現行 Stage 1(post §14.7-BY Phase E)

```python
# 單一 binary gate:13/13 indicators
def check_kwave_market_context(cur):
    """Stage 1: §0.3 K-wave market-level prerequisite(11+2 indicators)."""
    # 13/13 binary gate(混合 K-wave + Juglar + Kitchin + Microstructure)
```

### 7.2 Path C 升版:Stage 1A/1B/1C 三 sub-stages

```python
# §0.3 K-wave 7 indicators(strict ≥75% correspondence)
KW_INDICATORS_PURE = [
    ('PATENTUSALLTOTAL', 'fred_series', 'series_id'),
    ('LFWA64TTUSA647N', 'fred_series', 'series_id'),
    ('B985RC1Q027SBEA', 'fred_series', 'series_id'),
    ('SPPOPDPNDOLUSA', 'fred_series', 'series_id'),
    ('QUSPAM770A', 'fred_series', 'series_id'),
    ('TCMDO', 'fred_series', 'series_id'),
    ('PALLFNFINDEXQ', 'fred_series', 'series_id'),
]

# §0.4 Multi-cycle 5 indicators(7-25 年中期循環)
MC_INDICATORS = [
    ('M2SL', 'fred_series', 'series_id'),
    ('WTISPLC', 'fred_series', 'series_id'),
    ('TW_SEMI_VWAP_YOY', 'kwave_supply_cycle_proxy', 'proxy_id'),
    ('TW_SHIPPING_VWAP_YOY', 'kwave_supply_cycle_proxy', 'proxy_id'),
    ('T10Y2Y', 'fred_series', 'series_id'),
]

# §0.5 Microstructure 1 indicator
MS_INDICATORS = [
    ('VIXCLS', 'fred_series', 'series_id'),
]

def check_macro_pillars(cur):
    """Stage 1A/1B/1C: 三層 macro pillar prerequisite(per §14.7-BY Phase F Path C)。"""
    # Stage 1A: §0.3 K-wave 7/7 binary gate
    kw_present = check_indicators(cur, KW_INDICATORS_PURE)
    if len(kw_present) < len(KW_INDICATORS_PURE):
        return False, "Stage 1A K-wave fail", kw_present
    
    # Stage 1B: §0.4 Multi-cycle 5/5 binary gate
    mc_present = check_indicators(cur, MC_INDICATORS)
    if len(mc_present) < len(MC_INDICATORS):
        return False, "Stage 1B Multi-cycle fail", mc_present
    
    # Stage 1C: §0.5 Microstructure 1/1 binary gate
    ms_present = check_indicators(cur, MS_INDICATORS)
    if len(ms_present) < len(MS_INDICATORS):
        return False, "Stage 1C Microstructure fail", ms_present
    
    return True, "All 3 sub-stages PASS", kw_present + mc_present + ms_present
```

### 7.3 universe_completeness_snapshot insert 升 5 pillars

```python
# Phase F 升版:write per-stock × per-pillar(5 pillars)× per-layer(4 layers)
# §0.3 kondratiev: 7/7 (was 13/13)
# §0.4 multi_cycle: 5/5 (new)
# §0.5 microstructure: 1/1 (new)
```

---

## 8. Charter §0.4 + §0.5 新 sub-pillars outline

### 8.1 §0.4 Multi-cycle Context(新節)

```markdown
### §0.4 Multi-cycle Context(中期循環 macro context)

#### §0.4-A 治權邊界
1. **時間尺度**:7-25 年(Juglar + Kuznets)
2. **驅動因素**:business cycle(信用/設備投資)+ 基建 + 中期商品
3. **與 §0.3 K-wave 之區分**:
   - §0.3 為 40-60 年結構性;§0.4 為中期週期性
   - §0.4 indicators 之 K-wave correspondence 30-75%
   - §0.4 為 K-wave 之 sub-cycle hierarchy 內之中層
4. **不下沉 L2/L3**:同 §0.3-A 之治權邊界(Macro reference only)
5. **per-stock 影響**:不直接 per-stock 計算;類比 §0.3 為 market-level prerequisite gate

#### §0.4-B Indicator set(§14.7-BY Phase F Path C 初版)
- M2SL(US M2 monetary regime / Kuznets)
- WTISPLC(WTI oil Juglar cycle)
- T10Y2Y(yield curve Juglar leading)
- TW_SEMI_VWAP_YOY(Kitchin/K5 sector)
- TW_SHIPPING_VWAP_YOY(Juglar global trade)
```

### 8.2 §0.5 Microstructure(新節)

```markdown
### §0.5 Microstructure(短期 sentiment / market regime)

#### §0.5-A 治權邊界
1. **時間尺度**:月 ~ 季(下至 daily volatility)
2. **驅動因素**:volatility regime / short-term sentiment / market microstructure
3. **與 §0.4 Multi-cycle 之區分**:
   - §0.4 為 7-25 年中期;§0.5 為 月 ~ 季短期
   - §0.5 indicators 之 K-wave correspondence < 30%
4. **不直接驅動 selection**:Microstructure 為 reference signal,不作為 per-stock 之 selection gate(只 market-level binary gate)
5. **per-stock 影響**:不直接 per-stock 計算;類比 §0.3/§0.4

#### §0.5-B Indicator set(§14.7-BY Phase F Path C 初版)
- VIXCLS(VIX volatility regime / Microstructure)

#### §0.5-C Future expansion
- 月底 OPEX seasonality(待補)
- ETF 流向 / 法人籌碼 short-term flow(待補)
- Intraday liquidity proxy(待研究)
```

---

## 9. Audit 工具升版範圍

### 9.1 audit_universe_completeness.py 升 5 pillars

| Check | Pre Path C | Post Path C |
|---|---|---|
| C6 pillar_enum | 3 enum(first_principle/pareto/kondratiev)| 5 enum(+ multi_cycle/microstructure)|
| C9 universe_coverage | 1857 × 12 cells = 22,284 expected | 1857 × 20 cells = **37,140 expected** |
| C12 trinity_dashboard | 3 × 4 grid | **5 × 4 grid** |

### 9.2 audit_kwave_transition.py 升版(per Path C scope 重組)

當前 audit_kwave_transition 評估 5 indicators 之 spring/winter transition signals。Path C 後:
- 改名為 audit_macro_pillar_transition.py(或保留名稱,內容拆 3 sub-pillars 之 transition)
- §0.3 K-wave: 7 indicators 之 long-cycle 上升訊號
- §0.4 Multi-cycle: 5 indicators 之 mid-cycle 上升訊號(原本之 spring transition 主要適用此層)
- §0.5 Microstructure: 1 indicator 之 vol regime 上升訊號

### 9.3 universe_completeness_schema.py 升版

```python
# universe_completeness_schema.py L? CREATE TABLE statement:
DDL_UPDATE = """
ALTER TABLE universe_completeness_snapshot DROP CONSTRAINT IF EXISTS check_pillar_enum;
ALTER TABLE universe_completeness_snapshot 
  ADD CONSTRAINT check_pillar_enum 
  CHECK (pillar IN ('first_principle', 'pareto', 'kondratiev', 'multi_cycle', 'microstructure'));

REFRESH MATERIALIZED VIEW universe_completeness_matrix_current;
"""
```

### 9.4 build_doctrine_gate_universe.py 升版(per §7.2/§7.3)

- KW_INDICATORS_PURE / MC_INDICATORS / MS_INDICATORS 3 個 list
- check_kwave_market_context → check_macro_pillars(三 sub-stages)
- universe_completeness_snapshot insert 升 5 pillars

---

## 10. Cross-Reference 影響面(charter v6.1.0 + code)

### 10.1 Charter 升版範圍

| Section | 改動類型 |
|---|---|
| §0.3 條文 | 主體升 "K-wave purity" 之 strict definition;indicator set 從 13 mix 升 7 pure |
| §0.3-A 7 禁令 | 不動(同樣適用 §0.4 / §0.5)|
| §0.4 新節 | **新建**(Multi-cycle Context)|
| §0.4-A 治權邊界 | **新建** |
| §0.4-B Indicator set | **新建** |
| §0.5 新節 | **新建**(Microstructure)|
| §0.5-A 治權邊界 | **新建** |
| §0.5-B Indicator set | **新建** |
| §0.5-C Future expansion | **新建** |
| §14.7-BR Phase C-4 | 補註「Path C 後 indicators 拆 3 sub-pillars」 |
| §14.7-BU universe_completeness_snapshot | 升 5 pillars enum + 升 5 × 4 = 20 cells per stock |
| §14.7-BY | 補註 Phase F 升版 |
| §14.7-BZ 新節(預定) | **新建**(Phase F 入憲記述)|

### 10.2 Code 升版範圍

| File | 改動 |
|---|---|
| `scripts/maintenance/build_doctrine_gate_universe.py` | KW_INDICATORS → 3 lists / Stage 1 → 3 sub-stages / universe_completeness insert 升 5 pillars |
| `scripts/maintenance/audit_universe_completeness.py` | C6/C9/C12 升版識別 5 pillars |
| `scripts/maintenance/audit_kwave_transition.py` | 升版 / 拆 3 sub-pillars 之 transition logic |
| `scripts/core/universe_completeness_schema.py` | ALTER pillar enum CHECK constraint |

### 10.3 DB 改動

- ALTER TABLE universe_completeness_snapshot 之 CHECK constraint
- 既有 records 不動(`first_principle` / `pareto` / `kondratiev` 保留)
- 新 builder commit 時寫入 `multi_cycle` / `microstructure` records

---

## 11. 證偽承諾 T_BY-F-1〜5

依憲章 §0.3-E 之證偽承諾架構:

| ID | 證偽命題 | 失敗條件 |
|---|---|---|
| T_BY-F-1 | §0.3 K-wave pillar 升 strict 後,avg K-wave correspondence ≥ 75%(實證 ≥80%)| 若 < 70% → rollback Path C / re-evaluate indicator allocation |
| T_BY-F-2 | §0.4 Multi-cycle pillar avg K-wave correspondence 30-75%(實證 49%)| 若 > 75% 或 < 25% → 重新分類 |
| T_BY-F-3 | §0.5 Microstructure pillar avg K-wave correspondence < 30% | 若 ≥ 30% → 重新分類 |
| T_BY-F-4 | Stage 1A/1B/1C 三 sub-stages 全 PASS 後,§6.7 SSOT N=1857 不變(只 indicator 重組,不影響 stock selection)| 若 N 變動 > 10% → 重 evaluate |
| T_BY-F-5 | universe_completeness_snapshot 升 5 pillars 後,audit_universe_completeness 保持 PERFECT | 若 FAIL → schema 升版 bug |

---

## 12. 風險評估

### 12.1 工程風險

| Risk | Mitigation |
|---|---|
| ALTER TABLE 之 ENUM CHECK 升版可能 break 既有 records | 用 `DROP CONSTRAINT + ADD CONSTRAINT` pattern(per §14.7-BX C-1 之 status enum 升版 precedent)|
| Materialized view DROP + RECREATE 期間 audit 失效 | 跑 maintenance window;<1 分鐘 |
| 既有 audit 工具之 hardcoded `kondratiev` reference 需全清 | grep + 升版 audit_universe_completeness/audit_kwave_transition |

### 12.2 治權風險

| Risk | Mitigation |
|---|---|
| 新建 §0.4 / §0.5 charter section 影響面大 | 對齊 §14.7-BU additive governance pattern;不撤銷既有 §0.3 |
| §0.3 從 13 indicators 縮到 7 indicators 可能讓 multiplier 看似下降 | 但**pillar-level purity** 升至 80% pure;trade-off 在 charter 中明示 |
| §0.5 Microstructure 只 1 indicator(VIXCLS)看似稀薄 | Future expansion(intraday liquidity / 法人籌碼 / ETF flow)留作 v6.3.0+ scope |

### 12.3 Doctrine risk

| Risk | Mitigation |
|---|---|
| §0.3-A 「K-wave 不下沉 L2/L3」之邏輯延伸至 §0.4 / §0.5 是否合適? | §0.4 中期循環本身已是 L2 級別,但仍為 macro reference(不作為 per-stock score),所以邏輯一致 |
| §0.5 Microstructure 從學術上看可能屬 L3(intraday) | 但本 charter 之 §0.5 限為 monthly/quarterly indicators(VIXCLS daily close 仍視為 monthly aggregation),不踩 intraday L3 |

---

## 13. Phase F-1/F-2/F-3 sub-phases roadmap

### 13.1 Phase F-1(charter 入憲,~0.5 人天)

- §0.4 Multi-cycle Context 新節 inscription
- §0.5 Microstructure 新節 inscription
- §14.7-BZ 第二十四輪 Phase B 入憲 entry(類比 §14.7-BY 之 Phase B 模式)
- §14.7-BY 補註 "Phase F 升版"

### 13.2 Phase F-2(schema 升 5 pillars,~0.5 人天)

- universe_completeness_schema.py:加 ALTER constraint script
- 跑 DB migration:`python universe_completeness_schema.py --upgrade-pillars-5`
- Verify pg_constraint check 通過

### 13.3 Phase F-3(builder 升版,~1.5 人天)

- KW_INDICATORS 拆 3 lists(KW_INDICATORS_PURE / MC_INDICATORS / MS_INDICATORS)
- check_kwave_market_context → check_macro_pillars(三 sub-stages)
- universe_completeness_snapshot insert 升 5 pillars(per-stock × 5 × layer)

### 13.4 Phase F-4(audit 工具升版,~1 人天)

- audit_universe_completeness.py:C6/C9/C12 升 5 pillars
- audit_kwave_transition.py:拆 3 sub-pillars transition logic

### 13.5 Phase F-5(driver run + verify,~0.5 人天)

- 跑 builder commit + audit verify
- universe_completeness 之 5 pillars × 1857 stocks × 4 layers = 37,140 expected records
- audit_universe_completeness verdict 持續 PERFECT

---

## 14. 工程 effort 與時程估計

| Phase | 動作 | Effort | Cumulative |
|---|---|---|---|
| **Phase F-1** | Charter §0.4 + §0.5 + §14.7-BZ 入憲 | 0.5 人天 | 0.5 |
| **Phase F-2** | universe_completeness schema 升 5 pillars | 0.5 人天 | 1.0 |
| **Phase F-3** | builder 升版 + 3 sub-stages | 1.5 人天 | 2.5 |
| **Phase F-4** | audit 工具升版 | 1.0 人天 | 3.5 |
| **Phase F-5** | dry-run + commit + audit verify | 0.5 人天 | 4.0 |
| **Phase F 全 closure** | — | **~4 人天** | — |

時程:**全 closure 約 4 人天(跨 session;可分 2-3 sessions)**

---

## 15. 結論 + Phase G milestone path

### 15.1 結論

1. **Path C 為 §14.7-BY doctrine 之根本解**:從 indicator-level purity 升至 pillar-level purity;每層名實相符;**§0.3 K-wave 真正純化至 80% avg**
2. **13 indicators 分配**:7 K-wave + 5 Multi-cycle + 1 Microstructure(per 學術級別)
3. **新建 §0.4 / §0.5 charter sections**:對映 Schumpeter 多週期 hierarchy
4. **schema 升 5 pillars**:universe_completeness_snapshot 從 12 cells 升 20 cells per stock
5. **builder 升 3 sub-stages**:Stage 1A/1B/1C 各層獨立 binary gate
6. **effort ~4 人天**:跨 session 可分 2-3 sessions 完成
7. **Phase G v6.2.0 milestone tag** 為 Phase F 完成後之 closure

### 15.2 推薦 Phase B-style 入憲

依 §14.7-BY 之 Phase A → Phase B 模式,本 Phase F research(本文件)可作為 §14.7-BZ 第二十四輪入憲之 Phase A research,Phase F-1 為 Phase B-style charter inscription。

### 15.3 治權判準 seventh-round refinement

依 §14.7-BY 之六輪累進(BT/BU/BV/BW/BX/BY),Path C 成為:

| 輪次 | §14.7-Bx | 軸 |
|---|---|---|
| 18 | BT | N-axis cancel hardcode |
| 19 | BU | Governance schema |
| 20 | BV | Selection criterion |
| 21 | BW | N-axis pure form |
| 22 | BX | T-axis temporal |
| 23 | BY | Indicator-axis(本 §14.7-BY)|
| **24** | **BZ** | **Pillar-axis(本 Path C 升版)** ✅ |

**七輪累進完成**:N-axis + T-axis + Indicator-axis + **Pillar-axis** 四純化。

### 15.4 下一步(待用戶 explicit auth)

| Step | 動作 | Effort |
|---|---|---|
| 1 | Phase F-1 charter 入憲(需用戶 explicit auth)| 0.5 人天 |
| 2 | Phase F-2 schema 升 5 pillars | 0.5 人天 |
| 3 | Phase F-3 builder 升 3 sub-stages | 1.5 人天 |
| 4 | Phase F-4 audit 升版 | 1.0 人天 |
| 5 | Phase F-5 dry-run + commit + verify | 0.5 人天 |
| **Total** | — | **~4 人天** |

---

**Phase F 設計研究作者**: Claude(Opus 4.7)
**Session ID**: 2026-05-27
**Charter base**: v6.1.0 + 第二十三輪 patch(§14.7-BY)
**HEAD commit at Phase F Phase A 完成**: `f259a34`(v6.1.32-kondratiev-phase-e-p1-augmentation-20260527)
**Status**: ✅ Phase F Phase A research 完整 / 15 章 / non-destructive(不動 DB 不動 code)/ 待用戶 explicit auth 進 Phase F-1 charter 入憲

---

## 16. Namespace Correction Note(2026-05-27 同日 Phase F-1 入憲前修正)

執行 Phase F-1 charter inscription 前發現 charter §0.4 / §0.5 namespace **已被既有 v6.0.0 inscribed section 佔用**:
- §0.4 = 數位孿生完整性(per v6.0.0 charter L162)
- §0.5 = 三維度系統架構(per v6.0.0 charter L162)
- §0.6 = 戰略結論與行動指引
- §0.7 = 治權位階圖

**用戶 2026-05-27 explicit auth 修正**:採用 **§0.3 sub-section namespace** 取代:

| 本 Phase A research 原 proposal | Phase F-1 入憲實際 namespace |
|---|---|
| §0.3 K-wave(主節升版) | **§0.3.1 K-wave 純 macro structural(40-60 年)** |
| §0.4 Multi-cycle Context | **§0.3.2 Multi-cycle Context(7-25 年 Juglar+Kuznets)** |
| §0.5 Microstructure | **§0.3.3 Microstructure(月 ~ 季 sentiment/regime)** |

**對應 pillar enum 修正**(per §6 schema 升版):
- 原 proposal: `kondratiev` → 拆 `multi_cycle` + `microstructure`(獨立 enum values)
- 修正後: `kondratiev` → 拆 **`kondratiev_kwave` / `kondratiev_multicycle` / `kondratiev_microstructure`**(prefix 對齊 §0.3 hierarchy)
- 既有 `kondratiev` enum value **保留為 backward compat**(historical records 不動;新 commit 寫入 3 sub-pillar enum)

**§7-§9 程式碼之 list 命名修正**:
- KW_INDICATORS_PURE → 保留(對映 §0.3.1)
- MC_INDICATORS → 保留(對映 §0.3.2)
- MS_INDICATORS → 保留(對映 §0.3.3)

**修正影響**:
- §6 schema 升版:enum 從 5 升 **6**(加 3 sub-pillar enum,保留 `kondratiev` 為 backward compat)
- §8 charter outline:§0.4/§0.5 改為 §0.3.2/§0.3.3 sub-sections
- §11 T_BY-F-1〜5 證偽承諾:adjust pillar 名稱對映,核心邏輯不變

**Phase F-1 charter 入憲將以本修正後 namespace 落地**;design research 主體保留為 Phase A initial proposal 之 audit trail。
