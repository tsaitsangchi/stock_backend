# §0.3 康波週期 — §0.0-D 4 維度實證(本機 v0.2 baseline)

- **產出日期**: 2026-05-26 evening
- **產出者**: Claude Sonnet 4.7 session
- **觸發**: 用戶第 10 次 anchor echo「先看資料庫內的資料運用在核心股的挑選時在康波週期思想是否有資料依據」
- **執行**: 本機 v0.2 snapshot(`core_universe_20260521_core_universe_policy_v0_2`)
- **scope**: 從 §0.0-D 4 維度 framework(戰略定錨 / MBNRIC 對映 / leading indicators / 區域異步)看 actual DB 對齊
- **對映**: §0.0-D 跨層完整度基線 / §0.3.7-9 / §14.7-XK / §14.7-AA Part C(揭露之 root cause 之一)
- **類比**: `reports/pareto_4_dimensions_evidence_v02_baseline_20260526.md`(§0.2 八二法則 4 維度實證之姊妹文件)

---

## 維度 1:theme_score 全 150 stocks 全部 ≥ 70 ✅

```
sector breakdown(core+convex 150):
  半導體業       34 stocks  theme=100(字典最高)
  化學生技醫療     2          theme=95
  生技醫療業      1          theme=95
  電腦及週邊      16         theme=85
  通信網路業      4          theme=85
  電子工業       50         theme=80
  電子零組件業    25         theme=80
  其他電子類      5          theme=80
  電子通路業      2          theme=80
  其他電子業      1          theme=80
  電機機械       10         theme=75
  
→ 100% 達 convex_universe gate(theme ≥ 70)
→ 字典直接對應 industry_category(機械對齊)
```

---

## 維度 2:MBNRIC 6 支柱對映 — **重大結構失衡**

```
N Nanotech/Neural (半導體+電子+電子零組件)   109/150 = 72.7%
I Info/資訊      (電腦+通信+其他電子)         25/150 = 16.7%
R Robotics/綠能  (電機機械)                  10/150 =  6.7%
B Biotech/生技   (生技醫療+化學生技)            3/150 =  2.0%
C Computing      (電子通路)                    2/150 =  1.3%
M Materials      —                            0/150 =  0.0%  ← 完全缺席!
unmapped                                       1/150 =  0.7%
                                              ──────────────────
                                              149/150 = 99.3%
```

**🚨 重大揭露**: MBNRIC 6 支柱**字典覆蓋失衡** + **N 支柱 72.7%** 主導

### THEME_KEYWORDS 字典之 MBNRIC 對映分析

字典 14 條 keywords:
```
半導體 100, 生技 95, 醫療 95, 資訊 90, 電腦 85, 通信 85, 電子 80, 機器 80,
電機 75, 綠能 75, 光電 70, 能源 70, 航太 65, 汽車 60
```

| MBNRIC 支柱 | 字典 keywords | 字典覆蓋 |
|---|---|---|
| **M Materials** | — | **❌ 0 keywords** |
| B Biotech | 生技 / 醫療 | ✅ 2 keywords |
| N Nanotech/Neural | 半導體 / 電子 / 機器 | ✅ 3 keywords + 高分 |
| R Robotics/綠能 | 電機 / 綠能 / 汽車 | ✅ 3 keywords |
| I Info | 資訊 / 電腦 / 通信 / 光電 | ✅ 4 keywords |
| **C Computing/Cloud** | — | **❌ 0 keywords**(無「雲端」「量子」「AI」) |

→ 字典本身就**沒有 M(材料)和 C(雲端/量子)**之 keyword!此治權缺口從 v0.2 入憲就一直存在但未被揭露。

---

## 維度 3:FRED 4 series K-wave leading indicators ⚠️ 缺 3 個 §0.3.8 預期

| 本機 DB 實際 | rows | latest value | §0.3.8 對應 |
|---|---|---|---|
| **T10Y2Y** | 12,491 | 0.43 | ✅ I2 yield curve |
| **VIXCLS** | 9,191 | 16.76 | ✅ I4 恐慌指數 |
| DFF | 26,258 | 3.62 | ❌ 不在 §0.3.8 預期內 |
| UNRATE | 939 | 4.30 | ❌ 不在 §0.3.8 預期內 |
| ❌ M2SL(缺) | — | — | I1 春初訊號 |
| ❌ BDI(缺) | — | — | I3 全球景氣 |
| ❌ 半導體庫存(缺) | — | — | I5 supply cycle |

**§0.3.8 5 個 leading indicators 之實際完成度:2/5 = 40%**

---

## 維度 4:§0.3.7 區域異步 🔵 UNKNOWN(治權宣告 / 本機無證據)

§0.3.7 治權層宣告「台股 K-wave 滯後美股 5-10 年(半導體子賽道滯後 1-3 年)」;本機 DB v0.2 snapshot 不含跨年 sector rotation 對比,需 §10 walk-forward(v6.2.0)才能驗。

---

## 🚨 5 個新揭露 structural issues

### 1. THEME_KEYWORDS 字典之 MBNRIC 覆蓋失衡

- M(材料/微化)、C(雲端/量子/AI)**完全缺席**
- 字典 14 條全集中在 B/N/R/I 4 支柱
- 從 v0.2 入憲就存在但**未被任何 audit 揭露**

### 2. N 支柱 72.7% 集中 — §14.7-AA Part C 之 root cause 之一

之前認為「100% 半導體集中」是 sizer sector_cap 失效問題,但實際 **builder THEME_KEYWORDS 字典本身就天然偏向 N 支柱**:

```
N 支柱 keywords:    3 條(半導體 100 / 電子 80 / 機器 80)
   ↓ 對應 industry_category
   半導體業 34 + 電子工業 50 + 電子零組件業 25 + 其他電子類 5 + 其他電子業 1 = 115 stocks (76.7%)
```

→ prediction 之 100% 半導體 candidates 之上游 root cause **不只 model_trainer,也包含 builder 之字典天然偏好**

### 3. §0.3.8 5 個 leading indicators 之實作度僅 40%

- T10Y2Y ✅ / VIXCLS ✅
- M2SL ❌ / BDI ❌ / 半導體庫存 ❌
- K-wave 春初訊號之**判斷依據不足 60%**

### 4. §0.3-A 治權邊界正確守住

K-wave 在 L1 builder 落地(TR 15%)但**不**進 L2 prediction / L3 sizing — 對齊 §0.3-A 7 禁令之「不得把宏觀敘事直接寫入 L2/L3 計算」

### 5. §14.7-XK K-wave vs h20/h30 horizon 治權釐清

charter §0.3.10 已釐清:K-wave 為長期方向(50-60 年),h=30 為短期執行(30 交易日) — 兩者不衝突,K-wave 透過 universe selection 限定範圍,prediction 在範圍內做短期排名。

---

## §0.3 康波週期 4 層 verdict

| 層 | Verdict | 證據 |
|---|---|---|
| **資料層** | 🟢 STRONG | 字典 + MBNRIC + FRED 全有對應 raw data |
| **治權層** | 🟢 STRONG | §0.0-D / §0.3.7-9 完整 |
| **實作 L1** | 🟡 PARTIAL | TR 15% 字典實作 ✅ / MBNRIC M+C 字典缺口 ⚠️ |
| **實作 L2 (feature_store)** | 🔵 PARTIAL | FRED 4 series sync / 缺 3 indicators |
| **實作 L3 (sizing)** | ⚪ N/A | §0.3-A 禁令禁止 K-wave 進 L3 |
| **證偽層** | ⏸ PENDING | walk-forward 跨 K-wave 階段 IC 等 v6.2.0 |

---

## 跟 §0.2 八二法則之 4 維度實證對比

| §0.0-X 維度 | §0.2 八二法則(已 archive)| §0.3 康波週期(本檔)|
|---|---|---|
| 維度 1 | 左尾隔離(378 quarantine) | theme_score 100% ≥ 70 |
| 維度 2 | 右尾集中(6 維 sub-score)| MBNRIC 6 支柱失衡(N 72.7%)|
| 維度 3 | 上行凸性(v0.2 legacy 壓抑)| FRED leading indicators 40% |
| 維度 4 | 槓鈴資金(100% 電子集中)| 區域異步 UNKNOWN |
| 結論 | 資料層 STRONG / 實作 PARTIAL | 資料層 STRONG / 實作 PARTIAL |
| 新揭露 issues | 6 個 | 5 個 |

→ §0.2 跟 §0.3 在實作層皆有 PARTIAL issue,但 root cause 之 N 支柱集中為兩者**共通結構性問題**

---

## 對應未來治權升版方向(Phase A 設計研究候選)

| 候選 §14.7-X | 主題 | 工作量 |
|---|---|---|
| **§14.7-BO**(候選)| THEME_KEYWORDS 字典升版(MBNRIC M+C 補完)| Phase A 設計研究 ~2 小時 |
| **§14.7-BP**(候選)| §0.3.8 leading indicators 補完(M2+BDI+半導體庫存)| Phase A 設計研究 ~2 小時 |
| §14.7-BQ(候選)| §0.3.7 區域異步 walk-forward 驗證(等 §10) | v6.2.0 軌道 |
| 已存在 §14.7-BM | 金融業 ROE 對齊 Phase A | 已 commit `9f64755` |
| 已存在 §14.7-BN | portfolio_sizer v0.3 Phase B | 已 commit `9ea41ce` |

→ 本檔不入憲(類比 pareto_4_dimensions evidence 模式);純 archive 揭露;未來若需要 §14.7-BO / §14.7-BP 入憲再走 Phase A 流程。

---

## Cross-Reference

- 姊妹文件: `reports/pareto_4_dimensions_evidence_v02_baseline_20260526.md`(§0.2 八二法則 4 維度)
- 既有 §0.3 evidence: `reports/k_wave_evidence_and_l1_implementation_20260526.md`(他機 203 行 commit `0b84284`)
- Charter §0.0-D 跨層完整度: 系統架構大憲章_v6.1.0.md L557-621
- Charter §0.3.7-9: L243+ / L262+ / L272+
- THEME_KEYWORDS 字典: scripts/core/core_universe_builder.py L173-188

---

*Report generated 2026-05-26 evening by Claude Sonnet 4.7 session*
*基於本機 v0.2 snapshot (core_universe_20260521_core_universe_policy_v0_2)*
*v6.1.22 之後本 session 第 10 次 anchor echo 之 deep-dive closure*
