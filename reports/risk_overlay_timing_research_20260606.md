# 空頭風控 Overlay 研究報告 — vol-target×趨勢「地板」(Phase 1) + Gaussian-HMM regime「增強」(Phase 2)

**日期**: 2026-06-06
**作者**: Claude (AI 協作)
**治權對齊**: 憲章 §14.7-DG(regime-risk / 經濟價值口徑)、§8.5(anti-leakage)、§一.10(資料真實性 / source-traceable)、§一.11(三段式標頭)、§14.7-CY/CZ(walk-forward / 通過門檻精神)
**性質**: 研究驗證(walk-forward),**非獲利保證**;所有數據 trace 回 (b) DB query(`TaiwanStockPriceAdj`)。
**對應程式**: `scripts/evaluation/risk_overlay_timing_validation.py`(Phase 1)、`scripts/evaluation/risk_overlay_hmm_validation.py`(Phase 2)
**對應輸出**: `reports/risk_overlay_timing_20260606_181753.json`、`reports/risk_overlay_hmm_20260606_182818.json`

---

## 一、動機與設計框架

橫斷面選股模型回答「買哪些」;**空頭防護**回答「現在該在場內多少」。先前驗證顯示現行 `market_regime_timing`(LightGBM P(bear) 擇時)recall 低(門檻 0.3→30% / 0.5→13% / 0.7→0%)、precision 16-21%,且**對單日外生閃崩(如 2026-06-05 費半 SOX −10.26%)無預警力**。

據此採兩層設計,核心原則:**不把保命押在「預測準」上**——
- **(B) 風控地板(不需預測)**:波動目標(vol-target)+ 趨勢過濾,穩健、自動減碼。
- **(A) regime 增強(預測)**:校準後的 regime 機率當乘數疊加,只當「錦上添花」。

研究問題:**(A) 在 (B) 之上,到底加不加值?**

## 二、方法

### Phase 1 — 規則型 vol-target×趨勢 overlay(無 ML)
- **市場序列**:複用 `market_regime_timing` 之 `build_market_series`(全 numeric stock 等權 log-return 指數,1992-2026)→ 與現行擇時模型**同一條序列**(apples-to-apples)。
- **訊號(≤ t)**:近 60 日年化 realized vol;close/MA200(趨勢)。
- **曝險**:`vol_scaler = clip(目標波動 0.15 / 近期波動, 0, 1.0)`;`trend = 1 if 收盤>MA200 else 0`;三配置 = voltarget_only / trend_only / **combined(相乘)**。月度再平衡,曝險**上限 1.0(只減碼、不開槓桿)**。
- **Anti-leakage(§8.5)**:曝險在月底 t 僅用 ≤ t 資料,**t+1 起套用**;比例換手成本(SWITCH_COST 複用)。

### Phase 2 — Gaussian-HMM regime 增強(hmmlearn 0.3.3)
- **HMM**:2-態高斯 HMM,觀測 = [日 log-return, 近 5 日 vol];**每 12 個月在 ≤ t 報酬上重擬合**(擴張窗),bear 態 = 報酬均值較低者。
- **P(bear)(≤ t)**:`predict_proba(X[≤t])[-1]`(序列止於 t → 無未來洩漏);**isotonic 校準**(只用標籤已完整觀測之訓練樣本 s:s+20 ≤ t)。
- **疊加**:`floor_x_hmm = Phase1 combined × (1 − P(bear)校準)`;另比 `hmm_only`。
- **可重現(§一.10 #3)**:HMM EM init 固定 `random_state=5422`。

## 三、結果(全史 1992-2026,同口徑可比)

| 方法 | MaxDD | Calmar | Sharpe | CAGR | avgExp |
|---|---|---|---|---|---|
| buy & hold | 0.898 | −0.01 | −0.06 | −1.2% | 1.00 |
| `market_regime_timing`(LightGBM thr0.3,OOS~2002+)| 0.569 | 0.08 | — | — | — |
| **Phase 1 `floor`(vol-target×趨勢)** | **0.291** | 0.14 | 0.40 | +4.1% | 0.51 |
| **Phase 2 `floor×HMM`** | **0.259** | 0.15 | 0.44 | +3.8% | 0.44 |
| `hmm_only`(HMM 單獨)| 0.842 | −0.01 | −0.06 | −0.9% | 0.84 |

> ⚠️ 窗略不同(regime 模型 OOS~2002+ vs overlay 全 1992+);方向明確。

**增量裁決(`floor×HMM − floor`)**:
- 全史:ΔMaxDD **−0.032**、ΔCalmar +0.01、ΔSharpe +0.04 → **YES(邊際)**
- ≥2010 近期:ΔMaxDD −0.005、ΔCalmar ≈0 → **NO**

**逐空頭(`floor×HMM`,MaxDD,全部降回撤 ✅)**:
| 事件 | buy-hold | overlay |
|---|---|---|
| 2000 dotcom | 0.781 | **0.161** |
| 2008 GFC | 0.645 | **0.157** |
| 2020 covid | 0.305 | **0.050** |
| 2022 tightening | 0.200 | **0.093** |

**分層歸因**:trend_only 是主力(MaxDD 0.911→0.411);vol-target 單獨幫助小(成本拖累);combined 最佳(→0.291)。

## 四、誠實綜合裁決

**實證確認「(B) 風控地板是主力、(A) regime 預測是配角」**:
1. Phase 1 規則地板**獨力**把 34 年 MaxDD 91%→29%、翻正 Sharpe(−0.11→+0.40)/CAGR、4 空頭全降、且**勝過現行 LightGBM 擇時模型**——**不需任何預測**。
2. Phase 2 HMM 增強層**只加邊際價值(全史 −3pp MaxDD,主要修 2000/2008),近期幾乎為零**;`hmm_only` 單獨更差。
3. → **最佳實踐 = 以規則型 vol-target×趨勢為地板,regime 訊號頂多錦上添花、不可當主角。**

## 五、誠實警語(§一.10 / §14.7-DG)
- buy-hold 91% MaxDD 部分為「等權全市場含大量小型/下市股」之長史特性;**overlay 相對改善 + 逐空頭降幅**才是可信證據。
- overlay 是「波動噴出**後**減碼」→ **無法躲過單日外生閃崩第一天**(如 6/5 SOX −10%),只限制後續傷害;趨勢過濾在快速 V 轉會晚進晚出。
- 熊市事件少(34 年數次)→ HMM 校準不確定性大。
- cash=0%(保守,無無風險收益);比較窗未完全對齊(已揭露)。

## 六、Source-traceability / 可重現
- 全數據出自 (b) DB query `TaiwanStockPriceAdj`(複用 `build_market_series`);訊號 = real value → math transform;無 fabricate / 無 impute(warmup-NaN 決策略過)。
- Phase 2 HMM 固定 `random_state=5422` → 同參數同 seed 可重現;Phase 1 純規則無隨機性。

## 七、Deliverables 與後續
- **程式**:`risk_overlay_timing_validation.py`(Phase 1)、`risk_overlay_hmm_validation.py`(Phase 2),皆 §一.11 三段式 + anti-leakage + read-only。
- **相依**:`hmmlearn==0.3.3`(已加入 requirements)。
- **後續(未做,待授權)**:(a) 把 overlay 套到實際 top-N 選股組合(非僅市場指數)的曝險縮放;(b) 加信用利差 / VIX 期限結構等領先特徵的 regime 變體;(c) 與 production portfolio_sizer 整合之經濟價值再驗。

---

## 修訂歷程
| 版本 | 日期 | 說明 |
|---|---|---|
| v1.0 | 2026-06-06 | 首版。Phase 1(規則 overlay)+ Phase 2(HMM 增強)walk-forward 研究;確立 (B)>(A) 之經濟價值實證。全數據 DB source-traceable。 |
