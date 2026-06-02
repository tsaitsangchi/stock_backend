# 模型比較共同基準規格 SSOT v1.0 (Model Comparison Baseline Specification)

**文件性質**：跨模型「精準度 / 信任度」比較之**單一事實來源 (SSOT)**。本系統所有股價預測模型（tree-family / transformer / foundation / stock-specific / ensemble）**必須**依本規格產生驗證結果並納入同一 leaderboard，否則比較不可靠。
**位階**：承載憲章 §14.7-CY（4-horizon walk-forward 框架）+ §14.7-CZ T_CZ-6（production 通過門檻）之比較層落地;治權以憲章為準。
**建立日期**：2026-06-01
**基準框架實作 SSOT**：`scripts/evaluation/multi_cycle_validation.py` v0.1（base validator）;各家族 validator 沿用相同聚合邏輯,僅替換 fold 模型。
**資料真實性**：本規格定義之全部指標皆由程式對 PostgreSQL `feature_values` / `daily_prices` 實算產出（§一.10 (b) DB query）;**不得手填、不得估算**。

---

## 一、為何需要共同基準（可靠性前提）

不同模型若用不同 universe / 不同歷史長度 / 不同 horizon 定義 / 不同成本假設 / 不同 random seed 來各自驗證,得到的 Sharpe / Win rate **不可互比**（apples-to-oranges）。要回答「哪個模型預測股價較能賺錢、較可信」,**必須**所有模型站在**完全相同的比較基礎**上。本規格即鎖定該基礎。

---

## 二、共同驗證框架 — 4-Horizon Walk-Forward Expanding-Window OOS

每個模型對**四個預測週期**各跑一次完整 walk-forward（對齊用戶「by 週、月、季、年多重週期」需求）：

| horizon | label | horizon_days | 意義 |
|---|---|---|---|
| 週 | weekly | 5 | 短線 |
| 月 | monthly | 20 | 波段 |
| 季 | quarterly | 60 | 中期（憲章 §14.7-CY sweet spot）|
| 年 | annual | 252 | 長期 |

**Walk-forward 程序**（`multi_cycle_validation.py:225-255`,嚴格無洩漏）：
1. 面板（panel）依時間排序,每個面板 = 一個 as-of 日的全核心股特徵橫切面 + 其 H 日後的真實 forward return。
2. 對第 `i` 個面板（i≥1）：**只用 [0..i-1] 全部歷史面板訓練**,預測第 `i` 面板（true out-of-sample, expanding window）。
3. 訓練 label 先 winsorize 1%/99%;每 fold 最少 100 筆訓練樣本。
4. 「每一支個股在資料庫存在的個別過去最久的實際數據」即由 expanding window 自然體現 —— 早期面板訓練樣本少,隨時間擴張。

**Portfolio 建構**（每面板）：依模型預測值取 **top-20 個股（等權 long-only）**;`top20_ret` = 該 20 股實際 H 日 forward return 均值;`univ_ret` = 全核心股均值;`alpha = top20_ret − univ_ret`。

---

## 三、評估宇宙與資料基礎（apples-to-apples 鎖定項）

所有納入同一 leaderboard 的模型**必須共用**以下基礎,且報告須明列：

| 項目 | 鎖定值 | 來源 |
|---|---|---|
| 核心股 universe | committed snapshot（policy v0.18）| `core_universe_membership` core_tier='core_universe' |
| 特徵集 | `feature_set_v0_5`,37 source-pure features | `feature_values`（`is_null_imputed IS NOT TRUE` 過濾,per T_DC-27）|
| 面板數 | 95 historical panels（月度 grid,spacing=30d）| `feature_set_id` v0_5 panels |
| random seed | 5422 | validator `_meta.seed` |
| 成本模型 | cost_per_rebal = 0.006（台股來回手續費估計,Tier 3 揭露）| `multi_cycle_validation.py:289` |
| panel 間距 | 30 日 | `multi_cycle_validation.py:303` |

> ⚠️ **跨機 universe 分歧（誠實揭露）**：目前存在兩個各自內部一致的 universe 批次 —— macOS build = **914 核心股**、Linux from-zero 重建 = **397 核心股**。**同一張 leaderboard 內絕不可混批**;比較報告須分別呈現,並可用「同模型於兩 universe 是否同序」作為強健性（信任度）佐證。

---

## 四、標準化指標集（精準度 + 信任度,全部由框架實算）

每個 `模型 × horizon` cell 必須輸出下列指標（`multi_cycle_validation.py:313-330`）：

### A. 精準度（Precision / 預測力）
| 指標 | 定義 | 解讀 |
|---|---|---|
| `mean_ic` | 各面板 Spearman rank IC（預測值 vs 實際 forward return）之均值 | >0 有方向預測力;IC 越高排序越準 |
| `std_ic` | IC 跨面板標準差 | 穩定度 |
| `win_rate` | top20_ret > 0 的面板比例 | 賺錢面板佔比 |
| `top20_ret` / `univ_ret` | top-20 組合 vs 全宇宙均值報酬 | 選股是否優於平均 |

### B. 信任度（Trust / 統計可信度）
| 指標 | 定義 | 解讀 |
|---|---|---|
| `mean_alpha_per_panel` | mean(top20_ret − univ_ret) | 超額報酬 |
| `ir` | mean_alpha/std_alpha × √12 | 資訊比率 |
| `t_stat` | mean_alpha/(std_alpha/√n) | 原始 t（**未**重疊校正）|
| `n_effective` | 重疊校正後有效樣本數（見下）| 越小越不可信 |
| `effective_t_stat` | t_stat × √(n_eff/n) | **重疊校正後 t,信任度主指標** |
| `is_significant_p05` | \|eff_t\| > 1.997 | p<0.05 顯著性最低門檻 |
| `sharpe` | mean_ret/std_ret × √12 | 風險調整報酬 |
| `mdd_per_panel` | 累積報酬最大回撤 | 下行風險 |
| `annualized_simple_net` | 扣成本年化報酬（見下）| 「真的賺多少」|

### C. 重疊校正（為何 horizon 越長越不可信）
面板每 30 日取樣,但持有 H 日 → H>30 時相鄰面板報酬重疊 → 有效樣本縮水（`:300-311`）：
- H ≤ 30（週/月）：`n_eff = n`,無重疊
- H > 30（季/年）：`n_eff = n × (30/H)`

實測（95 panels,可用 ≈65）：週 n_eff≈65 / 月≈65 / 季≈32 / **年≈7.3**。
> ⚠️ **年 horizon 信任度警示**：n_eff≈7.3 → 自由度極低 → 信賴區間極寬;即使 `effective_t_stat` 過 4.20,該「顯著」仍**統計脆弱**（以 1.997 large-df 臨界值判定在 n_eff≈7 時偏寬鬆）。**年 horizon 的 PASS 必須附此 caveat,不可單獨宣稱「穩定賺錢」**。

### D. 誠實年化（`:280-298`）
`rebals_per_year = 252/H` → 短 horizon 重配次數多 → 成本拖累大。`annualized_log_net = mean_ret×rebals_per_year − 0.006×rebals_per_year`;`annualized_simple_net = exp(log_net) − 1`。

---

## 五、通過門檻（T_CZ-6 + 顯著性）

| 層級 | 條件 | 用途 |
|---|---|---|
| **最低顯著性** | `is_significant_p05`（\|eff_t\|>1.997, p<0.05）| 是否「不只是雜訊」 |
| **production 門檻 T_CZ-6** | **eff_t ≥ 4.20 且 Sharpe ≥ 2.40 且 Win ≥ 79%**（三者同時）| 是否「可信地賺錢」 |

> ⚠️ **T_CZ-6 非 code-enforced（finding B4）**：validator 程式僅硬判 `is_significant_p05`;完整 T_CZ-6 三條 AND 由本基準之比較產生器 / 人工依印出 metrics 裁決,不可佯稱 validator 自動 gate。

---

## 六、Apples-to-Apples 比較協議（鐵則）

1. 同一 leaderboard 內所有模型 **universe / 特徵集 / 面板 / seed / 成本 完全相同**。
2. **絕不混批**（914 與 397 分表）。
3. 每個指標必 trace 回 validator JSON（§一.10 (b)）;leaderboard 數字一律由程式解析 JSON 產生,**不手抄**。
4. 排名以「過 T_CZ-6 的 horizon 數」為主、`effective_t_stat` 為輔;同時呈現精準度（IC）與信任度（eff_t / n_eff）。
5. caveat（年 n_eff 脆弱、long-only top-20、log 年化假設）必隨報告揭露。

---

## 七、未來模型接入規範（Step 2/3 及後續）

新模型欲納入比較,只需：
1. 提供一支 `multi_cycle_<model>_validation.py`,沿用本框架（相同 universe/panels/seed/cost/4-horizon/top-20/IC/eff_t），僅替換 fold 模型。
2. 輸出符合 §四 指標鍵的 JSON 至 `reports/multi_cycle_<model>_<timestamp>.json`,`_meta` 含 `model_family / seed / n_universe / n_panels_input / source_traceability`。
3. 由 `scripts/evaluation/build_model_comparison_report.py` 自動掃描納入 leaderboard（依 model_key + n_universe 分組,不混批）。
4. 標頭依 §一.11 三段式;≥5min 觸發 §一.12 回報;≥30min 觸發 §二.6 SHMM。

---

## 修訂歷程

| 版本 | 日期 | 修訂說明 |
|---|---|---|
| v1.0 | 2026-06-01 | 首版。鎖定 T_CZ-6 + 4-horizon walk-forward 為跨模型比較唯一 SSOT;定義精準度（mean_ic/win/ir）+ 信任度（eff_t/sig/n_eff/sharpe/mdd/ann_net）標準指標集 + apples-to-apples 協議 + 年 horizon n_eff 脆弱 caveat + 跨機 914/397 分批揭露。全方法學 trace 至 `multi_cycle_validation.py:225-330`。|
