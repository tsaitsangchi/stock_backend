# §0.2 八二法則 — §0.0-C 4 維度實證(本機 v0.2 baseline)

- **產出日期**: 2026-05-26 evening
- **產出者**: Claude Sonnet 4.7 session
- **觸發**: 用戶第 9 次 anchor echo「先看資料庫內的資料運用在核心股的挑選時在八二法則思想是否有資料依據」
- **執行**: 本機 v0.2 snapshot(`core_universe_20260521_core_universe_policy_v0_2`)
- **scope**: 從 §0.0-C 4 維度 framework(左尾隔離 / 右尾集中 / 上行凸性 / 槓鈴資金)看 actual data 對齊
- **對映**: §0.0-C 跨層完整度基線 / §14.7-AA Part C 100% 半導體揭露 / §14.7-BH RMS

---

## 維度 1:左尾隔離(Left-tail Isolation)

```
quarantine_universe: 378 stocks
100% exclusion_reason: "non_equity_or_fund_like_industry"
```

**揭露**: 左尾隔離**只靠 industry filter**(排除 ETF/REIT/受益憑證);**沒有 quality-based 隔離**(score floor / coverage threshold / low LM / high RP 沒被 quarantine,只 rank 排後段)。

---

## 維度 2:右尾集中(Right-tail Concentration)

Core+Convex 150 stocks 之 sub-score 強度比較:

| Sub-score | top 1 | rank 75 (median) | rank 150 (last) | Δ |
|---|---|---|---|---:|
| **core_score** | 2382 廣達 **92.7** | 8112 至上 85.2 | 3013 晟銘電 82.1 | **10.6** |
| theme_score | 8299 群聯 100.0 | 8021 尖點 80.0 | 2049 上銀 75.0 | 25.0 |
| **FG** | 2382 廣達 100.0 | 2301 光寶科 90.0 | 3661 世芯-KY 55.0 | 45.0 |
| **IF** | 2344 華邦電 90.0 | 3167 大量 60.0 | 2354 鴻準 30.0 | 60.0 |
| **VC** | 2412 中華電 95.0 | 2441 超豐 50.0 | 2486 一詮 20.0 | 75.0 |
| LM | 2344 華邦電 99.5 | 3131 弘塑 84.2 | 4739 康普 61.9 | 37.6 |

**結構性發現**: 6 維 sub-score 之 top 1 是**不同 stocks**(廣達/群聯/華邦電/中華電 各最強一維)→ 證明**平均化是必要設計**(若 top 一致則平均化 redundant)。

---

## 維度 3:上行凸性(Upside Convexity)

Core+Convex 之 VC ≥ 85(stable 型)15 stocks:
- 電子工業 7 / 電腦及週邊 4 / 電機機械 3 / 其他電子 1
- **100% 電子業 + 全 stable 股**

**§0.0-C.3 揭露之 root cause**: v0.2 VC 用 legacy cv_close → 壓抑上下行波動 → 得高分是穩定股不是爆發股。v0.7.1 builder 已修(RMS / §14.7-BH)但本機 v0.2 snapshot 為 legacy 評分。

---

## 維度 4:槓鈴資金落地(Barbell Cash Distribution)

模擬 portfolio_sizer v0.2 對本機 v0.2 universe 跑配置(取 top 20 by core_score 為 long 訊號):

```
攻擊端總權重: 20.00% ✅(cap 20%)
CASH safety:  80.00% ✅(min 80%)
Core 配置: 0 stocks / Convex 配置: 7 stocks

Sector 集中度(攻擊端內):
  半導體業         9.00% of total /  45.0% of attack
  電子工業         6.00% of total /  30.0% of attack
  電腦及週邊設備業    5.00% of total /  25.0% of attack
  
→ 100% 電子業(完美驗證 §14.7-AA Part C 之揭露)
```

**重要 finding**: top 20 by core_score 全部 theme_score ≥ 70 → 進 convex tier(0 core,7 convex)。**core tier 反而很難進攻擊端**。

---

## §0.2 八二法則 4 層 verdict

| 層 | Verdict | 證據 |
|---|---|---|
| **資料層** | 🟢 STRONG | Raw Trading_money top 5% 拿 74.46% |
| **治權層** | 🟢 STRONG | §0.0-C / §0.2-A 7 禁令 / §9.2-A~H 12 FAIL gate 完整 |
| **實作 L1 Universe** | 🟡 PARTIAL | 5.42% selection ✅ / CoreScore 平均化壓縮冪律 ⚠️ |
| **實作 L2 Tactical** | 🔵 UNKNOWN | model_trainer 未落地(等 §10) |
| **實作 L3 Sizing** | 🟡 PARTIAL | sizer v0.2 cap ✅ / 100% 電子集中持續 ⚠️ |
| **證偽層** | ⏸ PENDING | walk-forward IC 等 v6.2.0 |

---

## 6 個新揭露 structural issues

1. **左尾隔離只靠 industry filter** — 沒 quality-based(low LM / high RP 沒被 quarantine,只 rank 排後段)
2. **6 維 sub-score 各自不同 top stocks** — 平均化設計合理(若不平均化會被單一維度劫持)
3. **本機 v0.2 VC 仍 legacy cv_close** — v0.7.1 RMS 修正在他機 DB production,本機未對齊
4. **模擬配置 100% 電子業** — 跟另一機 prediction-driven 100% 半導體結構相同
5. **top 20 by core_score 全進 convex tier** — 因 theme ≥ 70 觸發 convex 歸類;core tier 反而很難進攻擊端
6. **§0.0-C.3 之三層共同壓制** — L1 VC / Feature Store volatility_60d / Model Trainer 線性合成,都壓抑上行凸性

---

## Cross-Reference

- 之前 evidence: `reports/pareto_law_evidence_and_v07_universe_diff_20260526.md`(v0.7 snapshot 5 tests)
- v0.6 dry-run: `reports/v06_dryrun_vs_v02_baseline_universe_diff_20260525.md`
- portfolio_sizer v0.3 設計: `reports/portfolio_sizer_v03_design_research_20260526.md`
- 金融業 ROE 對齊 Phase A: `reports/financial_sector_roe_alignment_phase_a_research_20260526.md`
- Charter §0.0-C: 系統架構大憲章_v6.1.0.md L459-535
- Charter §0.2-A 7 禁令: L116 修訂歷程 + 主章節

---

*Report generated 2026-05-26 evening by Claude Sonnet 4.7 session*
*基於本機 v0.2 snapshot (core_universe_20260521_core_universe_policy_v0_2)*
*v6.1.20 之後本 session 第 9 次 anchor echo 之 deep-dive closure*
