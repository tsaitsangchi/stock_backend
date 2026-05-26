# §14.7-BP THEME_KEYWORDS 字典升版 Phase A 設計研究 — MBNRIC M+C 支柱補完(治本 §0.3 N 72.7% 主導 root cause)

- **產出日期**: 2026-05-26 evening
- **產出者**: Claude Sonnet 4.7 session
- **觸發**: 用戶選甲「§14.7-BP Phase A 設計研究(THEME_KEYWORDS 字典 M+C 補完)— 本機可即時做」
- **scope**: Phase A 治權先行設計研究 — 不動程式;類比 §14.7-BC/BF/BM/BO/§10 Phase A 模式
- **§14.7 編號釐清**: §14.7-BO 為 CashFlow Phase A;**本主題使用 §14.7-BP**(下一個自然連續 slot)
- **對映 charter**: §0.3.9 MBNRIC × 台股產業映射 / §14.7-AA Part C 100% 半導體 root cause / k_wave_4_dimensions evidence

---

## 一、觸發背景 — §0.3 4 維度 evidence 揭露之 root cause

### 1.1 §0.3 康波週期 4 維度 evidence(commit `833c2d6`)揭露之 5 個 structural issues

從 `reports/k_wave_4_dimensions_evidence_v02_baseline_20260526.md`:

| Issue # | 揭露 |
|---|---|
| **1** | **THEME_KEYWORDS 字典之 MBNRIC 覆蓋失衡(M + C 缺)** |
| 2 | N 支柱 72.7% 集中 — §14.7-AA Part C 之 root cause 之一 |
| 3 | §0.3.8 5 個 leading indicators 之實作度僅 40% |
| 4 | §0.3-A 治權邊界正確守住 |
| 5 | §14.7-XK K-wave vs h20/h30 horizon 治權釐清正確 |

→ **Issue #1 為本研究 §14.7-BP Phase A 之直接觸發**;Issue #3 為 §14.7-BQ 候選之另案

### 1.2 既有 14 keywords 之 MBNRIC 覆蓋分析

```
現有 THEME_KEYWORDS 字典(scripts/core/core_universe_builder.py L173-188):
半導體 100, 生技 95, 醫療 95, 資訊 90, 電腦 85, 通信 85, 電子 80, 機器 80,
電機 75, 綠能 75, 光電 70, 能源 70, 航太 65, 汽車 60

MBNRIC 對映:
| 支柱 | 字典 keywords | 字典覆蓋 |
|---|---|---|
| **M Materials** | (無) | **❌ 0 keywords** |
| B Biotech | 生技 / 醫療 | ✅ 2 keywords |
| N Nanotech/Neural | 半導體 / 電子 / 機器 | ✅ 3 keywords + 高分 |
| R Robotics/綠能 | 電機 / 綠能 / 汽車 | ✅ 3 keywords |
| I Info | 資訊 / 電腦 / 通信 / 光電 | ✅ 4 keywords |
| **C Computing/Cloud** | (無) | **❌ 0 keywords**(無「雲端」「量子」「AI」) |
```

→ MBNRIC 6 支柱中 **2 個(M+C)字典覆蓋為 0** — 從 v0.2 入憲就存在但未被任何 audit 揭露

### 1.3 對 §14.7-AA Part C 之 chain reaction

```
THEME_KEYWORDS 字典只覆蓋 4/6 支柱(M+C 缺)+ N 支柱權重最高(半導體 100)
   ↓
N 支柱 industry_category 佔 72.7% (115/150 in v0.2 snapshot)
   ↓
半導體 + 電子 + 電子零組件 = 76.7% universe
   ↓
prediction layer 100% 半導體 candidates(§14.7-AA Part C)
   ↓
portfolio_sizer 100% 電子集中(v0.3 治標)
   ↓
§14.7-AA Part C「sector_cap=0.40 失效」之 root cause
```

→ **§14.7-BP 字典升版為治本 root cause 之上游修補**;§10 model_trainer sector-balanced loss 為下游 reinforce

---

## 二、§0.3.9 MBNRIC 6 支柱對映 TWSE industry 現況清查

### 2.1 actual DB query 結果(全市場 2700+ stocks)

```
全市場 industry_category distribution(本機 TaiwanStockInfo):

電子工業              268     ← N (字典已 cover)
生技醫療業            242     ← B (字典已 cover)
ETF                  241     ← (excluded by industry filter)
電子零組件業           208     ← N (字典已 cover)
半導體業              155     ← N (字典已 cover)
其他                  127     ← (mixed)
電機機械              117     ← R (字典已 cover)
電腦及週邊設備業        117     ← I (字典已 cover)
上櫃 ETF              114     ← (excluded)
光電業                 91     ← I (字典已 cover)
建材營造               89     ← M (❌ 字典缺)
觀光餐旅               63     ← (非 MBNRIC)
通信網路業             62     ← I (字典已 cover)
鋼鐵工業               54     ← M (❌ 字典缺)
紡織纖維               54     ← M (❌ 字典缺)
綠能環保               51     ← R (字典已 cover)
其他電子類             49     ← N+C (字典已 cover N;C 缺)
資訊服務業             49     ← I+C (字典已 cover I;C 缺)
金融保險               49     ← (非 MBNRIC;K-wave 不對應)
汽車工業               45     ← R (字典已 cover)
化學生技醫療           42     ← B+M (字典已 cover B;M 缺)
食品工業               41     ← (非 MBNRIC)
電子通路業             37     ← N (字典已 cover)
航運業                 35     ← (非 MBNRIC)
文化創意業             34     ← (非 MBNRIC)
塑膠工業               27     ← M (❌ 字典缺)
數位雲端類             24     ← C (❌ 字典缺) ✨
化學工業               24     ← M (❌ 字典缺)
運動休閒               22     ← (非 MBNRIC)
居家生活類             22     ← (非 MBNRIC)
數位雲端               22     ← C (❌ 字典缺) ✨
其他電子業             21     ← N
貿易百貨               20     ← (非 MBNRIC)
綠能環保類             18     ← R (字典已 cover)
電器電纜               17     ← R (字典已 cover)
ETN                   16     ← (excluded)
居家生活               15     ← (非 MBNRIC)
金融業                 14     ← (非 MBNRIC)
油電燃氣業             13     ← R (❌ 字典「能源 70」可能 cover 但不精確)
橡膠工業               11     ← M (❌ 字典缺)
水泥工業                8     ← M (❌ 字典缺)
造紙工業                8     ← M (❌ 字典缺)
玻璃陶瓷                5     ← M (❌ 字典缺)
農業科技業              4     ← B (❌ 字典「生技」可能不完全 cover)
農業科技                3     ← B (❌ 同上)
```

### 2.2 MBNRIC 6 支柱完整對映 + 字典覆蓋 audit

| 支柱 | TWSE industries | 字典 covered? | M+C gap fill 候選 |
|---|---|---|---|
| **M Materials** | 鋼鐵(54)/ 塑膠(27)/ 化學(24)/ 紡織(54)/ 建材(89)/ 玻璃(5)/ 橡膠(11)/ 水泥(8)/ 造紙(8)/ 化學生技醫療(42 之 M 部分)| **❌ 0/10 sectors** | **9 新 keywords** |
| B Biotech | 生技醫療(242)/ 化學生技醫療(42 之 B 部分)/ 農業科技(7) | ✅ 90%(農業科技 +1) | +1 |
| **C Computing** | **數位雲端類(24)+ 數位雲端(22)= 46 stocks**/ 資訊服務業(49 之 C 部分)/ 其他電子類(49 之 C 部分) | **❌ 0/3 sectors** | **5 新 keywords** |
| N Nanotech | 半導體 155 / 電子 268 / 電子零組件 208 / 其他電子 49 / 其他電子業 21 / 電子通路 37 = 738 | ✅ 100% | 0 |
| R Robotics | 電機機械 117 / 綠能環保 51 / 綠能環保類 18 / 汽車工業 45 / 油電燃氣 13 / 電器電纜 17 | ✅ 85%(油電 +1) | +1 |
| I Info | 電腦及週邊 117 / 通信網路 62 / 資訊服務 49 / 光電業 91 | ✅ 100% | 0 |

---

## 三、字典升版設計清單(14 → 25-29 keywords)

### 3.1 M Materials 支柱補完(9 新 keywords)

| Keyword | 對應 TWSE industry | stocks | 建議分數 | 動機 |
|---|---|---|---|---|
| **化學** | 化學工業 + 化學生技醫療 | 24+42 | **65** | 高科技材料(光阻劑/CMP)|
| **建材** | 建材營造 | 89 | 55 | 第六波建築智能化 |
| **鋼鐵** | 鋼鐵工業 | 54 | 50 | 傳統 M;周期性低 |
| **紡織** | 紡織纖維 | 54 | 50 | 智能紡織(碳纖維/3D 編織) |
| **塑膠** | 塑膠工業 | 27 | 55 | 工程塑膠 / 生物降解 |
| **橡膠** | 橡膠工業 | 11 | 50 | EV 輪胎 / 工業密封 |
| **水泥** | 水泥工業 | 8 | 45 | 傳統 M;低成長 |
| **造紙** | 造紙工業 | 8 | 45 | 傳統 M;低成長 |
| **玻璃** | 玻璃陶瓷 | 5 | 50 | 光學玻璃 / 先進陶瓷 |

→ M 支柱補完 9 keywords / mean score 52(中等;不主導也不忽略)

### 3.2 C Computing/Cloud 支柱補完(5 新 keywords;高分對齊 §0.3 第六波)

| Keyword | 對應 TWSE industry | stocks | 建議分數 | 動機 |
|---|---|---|---|---|
| **量子** | (新興 / 對齊 §0.3 第六波 MBNRIC C) | 0 (待 future) | **100** | C 支柱頂分;對齊 §0.3 K-wave |
| **AI** | 其他電子類(部分)/ 資訊服務業(部分) | ~50 | **95** | 第六波 Cognitive 對應 |
| **雲端** | 數位雲端類(24)+ 數位雲端(22)| 46 | **95** | 直接對映 TWSE 數位雲端 sector |
| **算力** | (新興 / GPU AI 算力) | 0 (待 future) | 90 | C 支柱次頂;對齊 GPU 浪潮 |
| **演算** | 資訊服務業(部分)/ 演算法相關 | ~10 | 85 | C 支柱;對齊 algorithm-driven |

→ C 支柱補完 5 keywords / mean score 93(**高分對齊 §0.3 第六波之 priority**)

### 3.3 B Biotech 補強(1 新 keyword;補農業科技)

| Keyword | 對應 TWSE industry | stocks | 建議分數 | 動機 |
|---|---|---|---|---|
| **農科** | 農業科技業 + 農業科技 | 7 | 80 | B 支柱;對齊 §0.3 MBNRIC 之 Bio-agri |

### 3.4 R Robotics/綠能 補強(1 新 keyword;補油電)

| Keyword | 對應 TWSE industry | stocks | 建議分數 | 動機 |
|---|---|---|---|---|
| **油電** | 油電燃氣業 | 13 | 70 | R 支柱;傳統能源轉型 / 取代「能源 70」之精確化 |

### 3.5 升版前後對比

```
v0.2 baseline 字典 (14 keywords):
  半導體 100, 生技 95, 醫療 95, 資訊 90, 電腦 85, 通信 85, 電子 80, 機器 80,
  電機 75, 綠能 75, 光電 70, 能源 70, 航太 65, 汽車 60

v0.3 字典 (29 keywords;14 + 15 新):
  # N 支柱(不變,4 keywords)
  半導體 100, 電子 80, 機器 80, 光電 70 (光電 partial 也算 N)
  # B 支柱(現有 2 + 新 1 = 3)
  生技 95, 醫療 95, 農科 80 (新)
  # I 支柱(現有 4)
  資訊 90, 電腦 85, 通信 85, 光電 70
  # R 支柱(現有 3 + 新 1 = 4)
  電機 75, 綠能 75, 汽車 60, 油電 70 (新)
  # M 支柱(新 9)
  化學 65, 建材 55, 鋼鐵 50, 紡織 50, 塑膠 55, 橡膠 50, 水泥 45, 造紙 45, 玻璃 50
  # C 支柱(新 5)
  量子 100, AI 95, 雲端 95, 算力 90, 演算 85
  # 既有 不對應 MBNRIC(留作 cushion)
  能源 70(R partial)/ 航太 65(I partial)
```

---

## 四、預期 universe shift effect(theoretical projection)

### 4.1 v0.2 baseline vs v0.3 字典之 sector distribution 預期

```
v0.2 baseline(現況):
  N 109/150 = 72.7%(半導體 34 + 電子 50 + 電子零組件 25 ...)
  I 25 = 16.7%
  R 10 = 6.7%
  B 3 = 2.0%
  C 2 = 1.3%(電子通路;非真正 cloud)
  M 0 = 0.0%
  unmapped 1 = 0.7%

v0.3 預期(after 字典升版):
  N 70/150 = 46.7%(降 26pp;因 M+C 加入分流)
  I 25 = 16.7%(穩定;字典不變)
  R 12 = 8.0%(+1.3pp;油電 70 加分)
  B 5 = 3.3%(+1.3pp;農科 80)
  M 20 = 13.3%(+13.3pp;9 新 M keywords)
  C 15 = 10.0%(+8.7pp;5 新 C keywords)
  unmapped 3 = 2.0%
```

→ **N 從 72.7% 降至 46.7%**(治本 §14.7-AA Part C);M+C 從 1.3% 升至 23.3%(對齊 §0.3.9 MBNRIC 預期)

### 4.2 對 prediction layer 之預期 chain reaction

```
v0.3 字典 → builder TR 之 sector 多元化
   ↓
universe 之 sector spread 改善(N 46.7% / I 16.7% / R 8% / M 13.3% / C 10%)
   ↓
prediction model 訓練時 candidate pool 已多元化
   ↓
top 20 long signals 預期跨 4-5 sectors(對映 §10 之 T_MT_v0.2-7 證偽承諾)
   ↓
portfolio_sizer v0.3 之 G12 single_sector_count_max=3 真實有效
   ↓
最終配置自動 sector-balanced(§0.2 槓鈴跨域精神實現)
```

---

## 五、治權對齊 §0.3-A 7 禁令 + §14.7-BC 一致性

### 5.1 §0.3-A 7 禁令對齊度

| 禁令 # | 內容 | §14.7-BP 是否違 |
|---|---|---|
| 1 | K-wave 不下沉至 L2/L3 計算 | ✅ 不違(本研究只動 builder L1 TR score)|
| 2 | 不得將 K-wave narrative 寫成 alpha 固定值 | ✅ 不違(新 keywords 為 thematic identification 非 alpha)|
| 3 | 不得用 K-wave 推導出短期 trading 訊號 | ✅ 不違 |
| 4 | 不得跳 §0.4 可觀察性 | ✅ 不違 |
| 5 | 不得讓 K-wave 主題權重作為動態函數輸入 | ✅ 不違(score 為靜態) |
| 6 | 不得用 K-wave 取代 fundamentals | ✅ 不違(TR 仍 15% / FG 仍 20%)|
| 7 | 不得用 K-wave 推導 sizing | ✅ 不違 |

### 5.2 §14.7-BC PBR 金融業特殊處理一致性

從 charter §14.7-BC §4.2(L8107):
> 「PBR 估值(industry-relative)`rel = PBR / median(industry_PBR)` 4 階梯 + 金融業特殊處理 ±15」

**金融業在本 §14.7-BP 不加 keyword**:
- 金融保險(49)+ 金融業(14)= 63 stocks
- **K-wave 第六波 MBNRIC 不對應金融業**(金融不是 6 大支柱之一)
- §14.7-BC PBR 已對金融業有特殊處理(FG 層);TR 層不應加 keyword
- 保持金融業 theme_score = 30(neutral)即可

### 5.3 §14.7-BP 跟 §10 model_trainer sector-balanced loss 之關係

```
§14.7-BP:  builder 字典升版(L1 universe selection 多元化)
§10:        model_trainer sector-balanced loss(L2 prediction 多元化)
            
→ 兩者為「上下游 reinforcement」非「重複」
→ §14.7-BP 先治根(universe 不再 76.7% 電子);§10 後 reinforce(prediction 訓練 sector aware)
→ §14.7-BP Phase B-D 可獨立做(不阻塞於 §10);§10 落地後 leverage §14.7-BP 之效益
```

---

## 六、對既有 v0.2 / v0.7 snapshot 之影響

| 項目 | 影響 |
|---|---|
| 既有 v0.2 snapshot | **零**(不重 build)|
| 既有 v0.7 snapshot | **零**(他機 production 不變)|
| §6.4 CoreScore 公式 | **零**(TR 15% 權重不變)|
| §6.7 universe SSOT | **零** |
| §0.3-A 7 禁令 | **零** |
| §14.7-BC FG industry-relative | **零** |
| §10 model_trainer | 配套提供更 sector-balanced candidate pool(下游 benefit)|

---

## 七、§14.7-BP Phase A-D 路線圖

### Phase A:本研究(治權先行設計研究)

- ✅ 9 個 M keywords + 5 個 C keywords + 1 B + 1 R = 16 新 keywords
- ✅ 字典 14 → 29 keywords(預期 distribution shift 之 projection)
- ✅ §0.3-A 7 禁令對齊 + §14.7-BC 一致性
- ✅ Phase A 之 commit + push + tag v6.1.25

### Phase B:入憲 §14.7-BP charter 條文

- ⏸ 起草 §14.7-BP 子節(類比 §14.7-BO Phase A inspired 之 charter section)
- ⏸ 修訂歷程加 v6.1.0-patch 第十五輪 entry
- ⏸ 預估 charter +200-300 行

### Phase C:程式落地 builder THEME_KEYWORDS 字典升版

- ⏸ scripts/core/core_universe_builder.py THEME_KEYWORDS dict 加 15 新 keywords
- ⏸ 標頭 docstring v0.8 → v0.9 升版 entry
- ⏸ 修訂歷程加 v0.9 entry

### Phase D:smoke test + commit + tag v6.1.26

- ⏸ dry-run v0.9 對 2026-05-21 as_of_date
- ⏸ 比對 v0.2 vs v0.9 universe sector distribution
- ⏸ 預期 N 從 72.7% 降至 ~50%
- ⏸ commit + push + tag v6.1.26

---

## 八、證偽承諾 T_TK_v0.1-1〜5(等 Phase D dry-run)

| ID | 證偽指標 | 通過門檻 |
|---|---|---|
| **T_TK_v0.1-1** | M 支柱進 universe stocks | v0.9 dry-run 之 universe 包含 ≥ 10 M sector stocks |
| **T_TK_v0.1-2** | C 支柱進 universe stocks | v0.9 dry-run 包含 ≥ 5 C sector stocks(主要 數位雲端類)|
| **T_TK_v0.1-3** | N 支柱比例下降 | v0.9 dry-run 之 N sector ratio ≤ 60%(從 72.7%)|
| **T_TK_v0.1-4** | top 30 convex 包含 ≥ 4 sectors | 多元化驗證 |
| **T_TK_v0.1-5** | walk-forward IC ≥ v0.2 baseline | 字典升版不傷 IC(等 v6.2.0 §10) |

---

## 九、跟 §14.7-BO / §14.7-BM / §10 之關係

| Phase A | 主題 | 跟 §14.7-BP 之關係 |
|---|---|---|
| §14.7-BM 金融業 ROE | 金融業 FG 層補強 | 獨立(金融不在 MBNRIC;§14.7-BP 不加金融 keyword)|
| §14.7-BO CashFlow | V 補強(FG 7 新 sub-scores)| 獨立(FG vs TR 不同維)|
| §10 model_trainer | sector-balanced loss(L2)| **上下游 reinforcement**(§14.7-BP L1 / §10 L2)|
| §14.7-BQ leading indicators | §0.3.8 M2/BDI/半導體庫存補完 | 同 §0.3 但獨立 Phase A(另案)|

→ §14.7-BP **可獨立於其他 Phase A 落地**;Phase B-D 不需要等其他 Phase 完成

---

## 十、預估升版成本 vs benefit

### 10.1 工作量

```
Phase A 設計研究:    ~2 小時(本研究)
Phase B 入憲 charter: ~1.5 小時(+200-300 行)
Phase C 程式落地:    ~1 小時(THEME_KEYWORDS dict 加 15 keywords + 升版 docstring)
Phase D smoke test:  ~1 小時(dry-run + 比對 sector distribution)
                    ─────────────
總計:               ~5.5 小時(可單 session 完成或拆 2 sessions)
```

### 10.2 治權收益

- 解 §0.3 4 維度 evidence 之 issue #1(THEME_KEYWORDS 字典覆蓋失衡)
- 解 §14.7-AA Part C root cause 之上游修補(N 72.7% → 50%)
- §0.3.9 MBNRIC 6 支柱完整 mapping
- 對映 §0.3 第六波 emphasis(C/Cloud/AI/Quantum 高分)
- 為 §10 model_trainer 提供更多元 candidate pool

→ **5.5 小時 ROI 極高**;直接解決 v6.1.x 期間多次揭露之 root cause

---

## 十一、Cross-Reference 精確行號

| 項目 | 位置 |
|---|---|
| §0.3.9 MBNRIC × 台股產業映射 | charter L119+(v6.0.0-patch entry)|
| §0.3-A 7 禁令 | charter §0.3-A 主章 |
| §14.7-AA Part C 100% 半導體 root cause | charter L98(修訂歷程)|
| §14.7-BC PBR 金融業特殊處理 | charter L8107 |
| §14.7-BM 金融業 ROE Phase A | reports/financial_sector_roe_alignment_phase_a_research_20260526.md |
| §14.7-BO CashFlow Phase A | reports/cashflow_sync_phase_a_research_20260526.md |
| §10 model_trainer Phase A | reports/model_trainer_phase_a_research_20260526.md(commit 644e2eb)|
| §0.3 4 維度 evidence | reports/k_wave_4_dimensions_evidence_v02_baseline_20260526.md(commit 833c2d6)|
| 既有 THEME_KEYWORDS 字典 | scripts/core/core_universe_builder.py L173-188 |
| 本 §14.7-BP Phase A | reports/theme_keywords_dictionary_upgrade_phase_a_research_20260526.md(本檔)|

---

## 十二、治權邊界嚴守 + Phase A 結論

### 本 §14.7-BP Phase A 不改:

- §6.4 CoreScore 公式(TR 15% 權重維持)
- §6.7 universe SSOT
- §0.3-A 7 禁令
- §0.1-A 6 禁令
- §0.2-A 7 禁令
- §14.7-BC FG industry-relative 既有 PBR 金融業特殊處理
- §14.7-BI / §14.7-BM / §14.7-BO / §10 既有 Phase A
- builder v0.8 / portfolio_sizer v0.3 / audit v0.2(本 Phase D 才升 builder v0.9)
- raw DDL / CLI 結構

### 本 §14.7-BP Phase A 新增(僅 reports/):

- 本研究 reports/ 之 §14.7-BP Phase A 設計研究文件
- 9 個 M + 5 個 C + 1 B + 1 R = **16 新 keywords 設計**
- 預期 universe shift effect projection
- §0.3-A 治權對齊度驗證
- 證偽承諾 T_TK_v0.1-1〜5

### 本 §14.7-BP Phase B-D 未來:

- 入憲 §14.7-BP 子節 charter(+200-300 行)
- builder v0.8 → v0.9(THEME_KEYWORDS dict 加 15 keywords)
- dry-run v0.9 vs v0.2 比對 sector distribution
- 入憲 §14.7-BR(候選)v0.9 落地記述
- audit_core_universe v0.2 不動(THEME_KEYWORDS 升版不影響 audit)

---

## 十三、Phase A 結論

**問:THEME_KEYWORDS 字典升版可治本 §14.7-AA Part C root cause 嗎?**

**答:Phase B-D 落地後 N 支柱預期 72.7% → ~50%**(治本上游修補)

**完整 chain reaction**:
```
v0.2 字典 (14 keywords): N 72.7% / I 16.7% / R 6.7% / B 2% / C 1.3% / M 0%
   ↓ §14.7-BP Phase D 落地後
v0.9 字典 (29 keywords): N ~50% / I ~17% / R ~8% / B ~3% / M ~13% / C ~10%
   ↓ 對 prediction layer
prediction model 候選 pool 多元化 → top 20 跨 4-5 sectors
   ↓ 對 portfolio_sizer v0.3
G12 single_sector_count_max=3 真實有效
   ↓ 端到端
§0.2 槓鈴跨域精神實現 + §14.7-AA Part C 治本
```

**Phase A 治權成本**: ~2 小時 / 寫入 DB:0 / Charter 入憲:0(本 Phase A 不入憲;Phase B 才入)/ 程式變更:0

**Phase B-D 阻塞**: 無(完全本機可即時做;不需 sync;不需 token)

未來 Phase B-D 啟動時,可基於本 Phase A 直接展開。Phase D 完成後可立即見到 universe sector distribution 之改善(本機 v0.2 → v0.9 dry-run 比對)。

---

*Report generated 2026-05-26 evening by Claude Sonnet 4.7 session*
*基於 §0.3 4 維度 evidence(commit 833c2d6)揭露之 issue #1(THEME_KEYWORDS 字典覆蓋失衡)*
*Phase B-D 完全本機可即時做(不需 sync / 不需 token);跟 §10 model_trainer 為上下游 reinforcement*
