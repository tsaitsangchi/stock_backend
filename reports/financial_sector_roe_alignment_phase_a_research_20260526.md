# §14.7-BM 金融業 ROE 對齊 Phase A 設計研究 — 4 方案評估 + Theoretical Analysis(類比 §14.7-BK 模式)

- **產出日期**: 2026-05-26 evening(本機 v6.1.19.1 closure 後)
- **產出者**: Claude Sonnet 4.7 session
- **觸發**: 用戶 echo「金融業 ROE 對齊(等 §14.7-BL 候選)」+ 第 8 次 anchor 回應
- **§14.7 編號釐清**: §14.7-BL 已被 §14.7-BK Phase D 預定為 F 升 T1 marker;本主題用 **§14.7-BM**
- **scope**: Phase A 治權先行設計研究 — 不動程式;類比 §14.7-BK F 升 T1 Phase A 模式
- **執行限制**: 本機 DB 無 BS data(handoff §二);Phase A 純 theoretical;Phase B-D 待 BS sync 後跑

---

## 一、觸發背景

### 1.1 用戶 echo 鏈

| 順序 | echo |
|---|---|
| 1 | 「先看資料庫內的資料運用在核心股的挑選時在八二法則思想是否有資料依據」 |
| 2 | 「⚠️ CoreScore 公式 BY DESIGN 平均化(需 portfolio_sizer 補救)」 |
| 3 | 「reports/系統架構大憲章_v6.1.0.md(新增 §14.7-BI/BJ/BK + §14.7-AX(E)」 |
| 4 | 「完整封存 v6.1.19」 |
| 5 | 「v0.6 snapshot 是否 commit DB(仍 v0.2 baseline)」 |
| 6 | 「金融業 ROE 對齊(等 §14.7-BL 候選)」← **本研究觸發** |

### 1.2 已有 evidence 之金融業 ROE 缺口

從 `reports/pareto_law_evidence_and_v07_universe_diff_20260526.md` L134:

| Stock | 名稱 | v0.3 score | v0.7 score | Δ | v0.7 ROE |
|---|---|---|---|---|---|
| 2882 | 國泰金 | 75.04 | 77.39 | **+2.35** | **(no BS)** |
| 2891 | 中信金 | 81.31 | 82.66 | **+1.35** | **(no BS)** |

對比高 ROE 製造業:
| Stock | 名稱 | Δ | ROE |
|---|---|---|---|
| 2308 | 台達電 | +4.45 | 26.62% |
| 2454 | 聯發科 | +4.30 | 25.87% |
| 2330 | TSMC | +2.55 | 32.72% |

→ 金融業 score 升幅 +1.35 〜 +2.35,遠低於高 ROE 製造業 +4 範圍 → **金融業在 v0.7 ROE 解鎖後仍未獲對應評分**

### 1.3 已 acknowledge unfinished item

v6.1.18.2 handoff doc §六 已列:
> 「金融業 ROE 對齊(國泰/中信無 ROE 值)— 已知 — BS 對金融業 EAOP 對應需特殊 case」

本研究為**該 unfinished item 之 Phase A 治權先行設計研究**。

---

## 二、§14.7 編號歸屬釐清(治權整理)

### 2.1 §14.7-BL 已被預定

從 charter L8765 (§14.7-BK F 升 T1 Phase D 描述):
> 「Step 10-15. 入憲 §0.1.4 + **§14.7-BL**(類比 §0.1.3)+ 更新 §0.1.1/§0.1-A/§0.1-B/§0.1-E」

→ §14.7-BL 是 F 升 T1 之 future marker(等 walk-forward IC 通過後啟動)

### 2.2 本研究使用 §14.7-BM 編號

可用 slot:§14.7-BM / §14.7-BN(均未預定)
本研究使用 **§14.7-BM**(下一個自然連續編號)

### 2.3 治權編號表(post v6.1.19.1)

```
v6.1.18+(已入憲):
- §14.7-BG/BH/BI/BJ/BK
- §14.7-AX(E)

待入憲(Phase 進行中):
- §9.2-I 或 §14.7-BL 待 portfolio_sizer v0.3 Phase B(下個 session)
- §0.1.4 + §14.7-BL — F 升 T1 Phase D(等 walk-forward IC)

本研究新增候選:
- §14.7-BM — 金融業 ROE 對齊 Phase A(本文件;治權預備)
- §14.7-BM 升正式條文 — 等 BS sync + 4 方案 ablation
```

---

## 三、現有 charter 既有金融業相關條文清查

| Charter 行 | 條文 | 內容 |
|---|---|---|
| L3738 | §3.X | 金融業 8 家 8.1% revenue<0 |
| L3746 | §6.8.8-B 預備 | revenue 範圍 -32.7B 〜 -2K(含金控大額負值)|
| L3751 | §6.8.8-B 預備 | 金融業利息淨損 + 公平價值變動 + 保險衝抵 |
| L6914 | §6.8.8-B | revenue<0 合憲業務裁決(金融業 31%) |
| **L8107** | **§14.7-BC FG sub-score #2** | **PBR 估值(industry-relative)4 階梯 + 金融業特殊處理 ±15** |

### 重要發現:§14.7-BC PBR 已有金融業特殊處理之預備,但 ROE 沒有

§14.7-BC FG v0.5 升版時,**已預備 PBR 之金融業特殊處理(±15)**,但未明文化:
- 哪些 industry_category 屬於金融業?(金融業 / 銀行業 / 證券業 / 保險業 4 類?)
- 金融業 PBR 比較 base 是什麼?(同業 median / 銀行業 median?)
- 為什麼 PBR ±15 但沒 ROE 對應?

**本 §14.7-BM 應與 §14.7-BC 之金融業特殊處理對齊** — 一致定義「金融業」範圍 + 一致 industry-relative 邏輯。

---

## 四、金融業會計特殊性 — 為何 ROE 對齊困難

### 4.1 金融業 BalanceSheet 結構差異

| 維度 | 一般企業(製造/科技) | 金融業 |
|---|---|---|
| Assets - Liabilities | = Equity | = Equity + Reserves + 技術準備金 |
| Equity 組成 | Common Stock + RE + OCI | + AT1 (Additional Tier 1) + Tier 2 |
| ROE 標準 | NI / Equity | **Banking ROE = NI / Tier 1 Capital** (BIS) |
| 資本適足率 | 無強制 | **BIS 8% / Tier 1 6%** 強制 |
| OCI 影響 | 中等 | 極大(投資未實現損益占淨值大宗)|

### 4.2 EAOP (EquityAttributableToOwnersOfParent) 對金融業可能歧義

`EAOP` 是 IFRS 用語,對金融業可能:
- **A.** 含 OCI 中投資未實現損益 → 隨市場大幅波動 → ROE 失真
- **B.** 不含 AT1 / Tier 2 → 與 Banking ROE 標準偏差
- **C.** 含技術準備金(保險業)→ 與真實淨值不對應
- **D.** 完全不含 BS data(本機 query 顯示)→ 直接 None

### 4.3 國泰金 / 中信金 case study

```
Stock 2882 國泰金:
  - industry_category: 金融業
  - v0.3 score 75.04 → v0.7 77.39 (Δ +2.35)
  - ROE: (no BS) — 可能因 BS table 中沒有國泰金資料
       OR EAOP 對銀行業意義不對應 builder 邏輯

Stock 2891 中信金:
  - industry_category: 金融業
  - v0.3 score 81.31 → v0.7 82.66 (Δ +1.35)
  - ROE: (no BS) — 同上

對比高 ROE 製造業 score 升 +4 範圍:
  - 國泰金 / 中信金 之 ROE 缺值 → score 升幅僅 +1.35 〜 +2.35
  - 此 +1.35〜+2.35 為其他 sub-score (LM/TR/IF/VC) 之自然調整,非 ROE 貢獻
```

### 4.4 金融業 vs 銀行業 vs 證券業 vs 保險業之歸類

TWSE/TPEx industry_category 之金融類包含:
- **金融業**(主要 — 8 家金控:2880 華南金 / 2881 富邦金 / 2882 國泰金 / 2885 元大金 / 2886 兆豐金 / 2887 台新金 / 2888 新光金 / 2891 中信金 等)
- **銀行業**(獨立銀行 — 2812 台中銀 / 2820 華票 / 2834 臺企銀 等)
- **證券業**(2855 統一證 / 6005 群益期 等)
- **保險業**(2851 中再保 / 2867 三商壽 等)
- **其他金融業**(2849 安泰銀 等)

每子分類會計處理略有不同,本 §14.7-BM 應該明文化 4 子分類之處理。

---

## 五、4 種金融業 ROE 對齊方案 evaluation

### 方案 1:Banking ROE(金融業特殊公式)

**公式**: `Banking ROE = Net Income / Tier 1 Capital`

**前提條件**:
- FinMind FinancialStatements 須含 Tier 1 capital 欄位(本機無 BS 無法驗;另一機可查)
- 對銀行業 / 金控 計算
- 證券業 / 保險業 仍需獨立公式

**優點**:
- 對齊 BIS 國際標準
- 反映金融業真實獲利能力(扣 capital adequacy 之 leverage 限制)

**缺點**:
- 複雜度高 — 需 Tier 1 capital 欄位
- 跨金融子分類仍需多套公式
- 與其他產業 ROE 公式不同,無法直接 industry-relative compare

**Phase B 實證需要**:
- 查 BS 中是否有 Tier 1 capital 欄位
- 8 家金控 Tier 1 capital 完整度

### 方案 2:Simplified — 用 OperatingIncome / Equity

**公式**: `金融業 simplified ROE = OperatingIncome / EquityAttributableToOwnersOfParent`

**前提條件**:
- 金融業有 OperatingIncome 欄位(FinStmt 中已有)
- 排除 OCI 不實現損益之噪音

**優點**:
- 較簡單 — 用 FinStmt 既有欄位
- 排除 OCI 波動影響

**缺點**:
- 仍依賴 EAOP 對金融業正確含義
- OperatingIncome 對保險業可能含技術準備金提存

**Phase B 實證需要**:
- 8 家金控 + 14 家銀行業 OpInc / EAOP 計算
- vs 一般 ROE (NI / EAOP) 之差異 / 對齊度

### 方案 3:純排除 — 金融業 ROE = None(保守)

**公式**:
```python
if industry_category in ['金融業', '銀行業', '證券業', '保險業', '其他金融業']:
    roe = None  # 跳過 ROE sub-score, 不評分
```

**優點**:
- 最保守 / 最透明
- 不誤判 — 不亂套公式
- 程式最簡單

**缺點**:
- 金融業選股完全無 ROE 訊號
- 國泰金 / 中信金 score 仍只能靠 LM/TR/IF/VC 拉抬
- 跟現狀(BS data missing 自動 None)效果相同

**Phase B 實證需要**:
- 確認 industry_category 範圍 4-5 類
- 金融業 universe 在 v0.7 baseline 之 score 分佈

### 方案 4:Beta proxy — 用 ROA × Leverage

**公式**: `金融業 alt_ROE = ROA × (Total Assets / Equity)`
其中:
- `ROA = NetIncome / TotalAssets`
- `Leverage = TotalAssets / Equity`(金融業典型 8x-15x)

**優點**:
- 利用金融業高槓桿特性反推 ROE
- 對齊 DuPont 拆解之治權含義

**缺點**:
- 需 TotalAssets 欄位(BS 中應有)
- 槓桿倍數計算可能放大誤差
- 對保險業效果差(技術準備金影響大)

**Phase B 實證需要**:
- BS 中 TotalAssets 欄位完整度
- 8 家金控 leverage 計算
- vs 方案 1/2 之差異

---

## 六、4 方案治權成本對比表

| 維度 | 方案 1 Banking ROE | 方案 2 Simplified | 方案 3 純排除 | 方案 4 ROA × Leverage |
|---|---|---|---|---|
| 程式複雜度 | 高 | 中 | 低 | 中 |
| 資料依賴 | Tier 1 capital 欄位 | OpInc + EAOP | industry_category | NI + Assets + EAOP |
| 治權影響 | 大(改公式) | 中(只改金融業) | 小(增 if-branch) | 中 |
| 國際標準對齊 | ✅ BIS | ⚠️ 簡化 | — | ⚠️ DuPont |
| ROE 有值率(8 金控) | 待 BS 查實 | 待 BS 查實 | 0%(永遠 None) | 待 BS 查實 |
| Charter 章節結構 | 新公式 §6.3-N | §6.3 第 4 條 sub-clause | §6.3 第 4 條 sub-clause | §6.3-N |
| 配套 audit gates | 新增 G16 | 新增 G16 | 無 | 新增 G16 |
| **Phase A 治權成本** | **高** | **中** | **低** | **中** |

**Phase A 建議**: 方案 3(純排除)為 Phase A 治標 / 方案 1 為 Phase D 治本

---

## 七、Phase A / B / C / D 路線圖

### Phase A:本研究(治權先行設計研究)

- ✅ Theoretical analysis(本文件第 4 章)
- ✅ 4 方案 evaluation(第 5/6 章)
- ✅ Charter 編號歸屬釐清(§14.7-BM 而非 §14.7-BL)
- ✅ 既有 charter 條文清查(§14.7-BC PBR 金融業特殊處理對齊)
- ✅ 本研究 commit + push + tag v6.1.20

### Phase B:等本機 BS sync(per handoff §二 方向甲/丙)

- ⏸ 查實 8 家金控 + 14 家銀行業 BS data 完整度
- ⏸ Tier 1 capital 欄位是否存在(方案 1 前提)
- ⏸ EAOP 對金融業實際含義(方案 2 前提)
- ⏸ TotalAssets 欄位完整度(方案 4 前提)

### Phase C:4 方案 ablation(BS sync 後)

- ⏸ 4 方案各跑 dry-run + 比對 universe 差異
- ⏸ 8 家金控 score 升幅之 4 方案比較
- ⏸ 跟製造業 ROE 標準對齊度分析
- ⏸ §14.7-BC PBR 金融業特殊處理之一致性檢驗

### Phase D:選一方案落地 + 入憲

- ⏸ 依 Phase C ablation 結果選方案
- ⏸ 入憲 §14.7-BM 升正式條文 + 配套 §6.3 第 4 條 sub-clause
- ⏸ builder v0.8 → v0.9 程式落地
- ⏸ audit_core_universe v0.2 配套
- ⏸ 證偽承諾啟動

---

## 八、配套 §14.7-BC 金融業特殊處理之一致性整理

### 8.1 §14.7-BC L8107 已預備 PBR 金融業 ±15

> 「PBR 估值(industry-relative)`rel = PBR / median(industry_PBR)` 4 階梯 + 金融業特殊處理 ±15」

但**未明文化**:
- 「金融業」具體範圍?(僅金融業 1 類 / 含銀行業 / 含 4 子分類?)
- 「特殊處理 ±15」具體公式?(如:rel < 1.0 → +15 等)
- 為什麼 PBR 對金融業需特殊處理?(因金融業 PBR 通常 0.5-1.5,vs 製造業 2-5 不同基準)

### 8.2 本 §14.7-BM 應同時補完 §14.7-BC 金融業條款

**v0.3 或 v0.4 升版時建議**:
- §14.7-BC 金融業範圍明文化(配合 §14.7-BM)
- §14.7-BC PBR 金融業特殊處理公式明文化
- §14.7-BM ROE 金融業特殊處理(per 選定方案)
- 配套 §6.3 第 4 條 FG sub-clause 明文化「金融業特殊處理」之 4-5 子產業範圍

→ 此一致性整理為 Phase B/C 之配套工作

---

## 九、治權邊界嚴守

### 本 §14.7-BM Phase A 不改:

- §6.4 CoreScore 公式總結構(VC 5% 維持)
- §6.7 SSOT 150
- §0.1-A 6 條禁令
- §9.9 P1 v0.1 強制契約 RMS 公式
- §14.7-BC PBR 金融業特殊處理現狀(未明文化但已存在;不撤回)
- §14.7-BI ROE 解鎖 SUCCESS(主流製造業 ROE 仍用 NI / EAOP)
- builder v0.8 既有 `_roe_score()` helper(不改)
- audit_core_universe v0.2 既有檢查
- raw DDL
- CLI 介面

### 本 §14.7-BM Phase A 新增:

- 本研究 reports/ 之 §14.7-BM Phase A 設計研究文件
- 4 方案 evaluation + theoretical analysis
- Phase A-D 路線圖
- §14.7-BC 金融業條款一致性整理建議

### 本 §14.7-BM Phase D 未來 (等 BS sync + ablation 後):

- §6.3 第 4 條 FG sub-clause 「金融業 ROE 特殊處理」明文化
- §14.7-BC 金融業範圍明文化(配套)
- builder v0.8 → v0.9 程式落地(加 if industry_category in [...]: alt_ROE)
- audit_core_universe v0.2 配套(若新 EXPECTED_KEYS)

---

## 十、證偽承諾 T_FIN_ROE_v0.1-1〜5(等 Phase D)

| ID | 證偽指標 | 通過門檻 | 失敗反應 |
|---|---|---|---|
| **T_FIN_ROE_v0.1-1** | 金融業 ROE 有值率 | 4 家金控 / 8 家有 ROE 值 ≥ 50% | 換方案 |
| **T_FIN_ROE_v0.1-2** | 金融業 score 升幅(v0.7 baseline vs after §14.7-BM) | Δ ≥ +2 分平均 | 公式重算 |
| **T_FIN_ROE_v0.1-3** | 金融業 ROE 與製造業 ROE 之 z-score 對齊 | 跨產業 z 分佈 [-2, +2] 之 95% | 標準化重組 |
| **T_FIN_ROE_v0.1-4** | universe selection 之金融業 count 變化 | v0.7 vs v0.8 金融業 count 變化 ≤ 3 | 過度反應檢驗 |
| **T_FIN_ROE_v0.1-5** | walk-forward IC 對齊(v6.2.0)| 金融業 universe 之 IC ≥ baseline | 撤回方案 |

---

## 十一、Cross-Reference 精確行號

| 項目 | 位置 |
|---|---|
| §14.7-BC PBR 金融業特殊處理 | charter L8107 |
| §14.7-BK F 升 T1 Phase D §14.7-BL reservation | charter L8765 |
| §6.8.8-B revenue<0 金融業合憲業務 | charter L6914 |
| §3.X 金融業 8 家 8.1% revenue<0 | charter L3738 |
| 國泰金 / 中信金 v0.7 evidence | `reports/pareto_law_evidence_and_v07_universe_diff_20260526.md` L134 |
| §14.7-BI ROE 解鎖 SUCCESS | charter L8662 |
| Handoff §六 unfinished item | `reports/session_handoff_20260526.md` |
| 本 §14.7-BM Phase A | `reports/financial_sector_roe_alignment_phase_a_research_20260526.md`(本檔) |

---

## 十二、對下游影響

### Phase A(本研究):零

- 不動程式
- 不動 charter §6.3 / §6.4
- 不動 universe / snapshot / audit

### Phase D(未來 落地時):

- builder v0.8 → v0.9 加金融業 ROE 特殊邏輯
- audit_core_universe v0.2 配套(可能加 EXPECTED_SCORE_DETAIL_KEYS v0.7 之 fg_roe_method)
- portfolio_sizer v0.3+(若已落地)可加 G15 之配套金融業特殊處理
- feature_store_builder 可能不動(屬 §9.9 RMS 已對齊)

---

## 十三、與 §14.7-BK F 升 T1 Phase A 模式對齊度

| 特徵 | §14.7-BK F 升 T1 Phase A | §14.7-BM 金融業 ROE Phase A |
|---|---|---|
| 治權先行設計研究 | ✅ | ✅ |
| Not 改程式 | ✅ | ✅ |
| 4 方案 / Option 比較 | ✅ (5 條治權支持 / 5 條反對) | ✅ (4 方案 evaluation) |
| 治權成本評估 | ✅ | ✅ |
| Phase A-D 路線圖 | ✅ | ✅ |
| 等下游(walk-forward IC) | ✅ | ⚠️ 等 BS sync 才能 Phase B |
| 配套既有條款(§14.7-BC) | ❌ | ✅ |

**結論**: §14.7-BM 與 §14.7-BK 同屬 Phase A 治權先行設計研究模式;§14.7-BM 額外有「§14.7-BC PBR 金融業特殊處理一致性整理」之配套價值。

---

## 十四、Phase A 結論

### 14.1 Phase A 主要結論

1. **§14.7-BL 已被 §14.7-BK F 升 T1 預定** — 本主題用 **§14.7-BM**
2. **金融業 ROE 對齊存在根本性會計差異** — 不能直接套用一般企業 ROE 公式
3. **4 方案 evaluation 完成** — 方案 3(純排除)為 Phase A 治標 / 方案 1(Banking ROE)為治本
4. **§14.7-BC PBR 金融業特殊處理已預備但未明文化** — 本 §14.7-BM 應配套整理
5. **Phase B 阻塞於本機 DB BS sync**(per handoff §二)— 待 sync 後才能 ablation 4 方案
6. **完整治權路線圖 Phase A-D 已就緒** — 4 phases 各對應明確 deliverable

### 14.2 Phase A 治權成本

- 本研究時間: ~2 小時
- 寫入 DB: 0
- Charter 入憲: 0(本研究不入憲;只 commit + push + tag;入憲在 Phase D)
- 程式變更: 0
- 對 production 影響: 0

### 14.3 Phase B 建議啟動時點

- 本機 DB sync 至 v0.7(含 BS table)後
- 或在另一機(production)直接做 Phase B
- 或等下次主題切換時(用戶決定)

---

## 十五、結語

**金融業 ROE 對齊**是 v6.1.18.2 handoff §六 列為 unfinished item 之主題,本 §14.7-BM Phase A 設計研究**完整 closure 其治權預備**:

- 4 方案完整 evaluation(Banking ROE / Simplified / 純排除 / ROA × Leverage)
- 金融業會計特殊性 theoretical analysis
- §14.7-BL slot 釐清(已被 §14.7-BK 預定;本主題用 §14.7-BM)
- §14.7-BC PBR 金融業特殊處理一致性整理建議
- Phase A-D 路線圖 + 證偽承諾框架

**Root cause 限制**:本機 DB 無 BS table,Phase B-D 阻塞於 sync(per handoff §二);Phase A 純 theoretical,**可即時 commit + push + tag v6.1.20**。

未來 Phase B-D 啟動時,可基於本 Phase A 直接展開 + 跑實證 + 入憲。類比 §14.7-BK 模式(Phase A 設計研究 + 等下游條件 + Phase B-D 後續)。

---

*Report generated 2026-05-26 evening by Claude Sonnet 4.7 session*
*基於 v6.1.19.1 evening closure + 用戶第 8 次 anchor echo*
*Pending Phase B 啟動(等 BS sync 或 production 機操作)*
