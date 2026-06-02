# LightGBM Multi-Cycle 驗證報告(參考基準模型 #1)

**模型**：LightGBM 4.6.0 | **類別**：Tree | **角色**：本標準化框架**首個參考基準模型**(建立報告範本)
**最後更新**：2026-06-02
**比較基準**：依 `common_model_comparison_baseline_20260602.md`(同一 universe / 窗 / horizon / metric / gate)
**資料來源**：全 (b) DB query(`source_traceability="all (b) DB query"`，§一.10)
**Seeds**：5422 / 1009 / 7331(3 runs，取 median + range，§一.10 #3)

---

## 一、運行配置(= 共同比較基準)

| 項目 | 值 |
| :--- | :--- |
| Universe | 397 source-pure 核心股(§14.7-DC PAN-HISTORICAL gate) |
| 特徵集 | `feature_set_v0.5`，37 source-pure 特徵 |
| 驗證窗 | **157 panels / 2013-05-15 ~ 2026-06-01**(資料驅動，`get_canonical_panel_dates()`，§14.7-DE) |
| Horizons | weekly 5d / monthly 20d / quarterly 60d / annual 252d |
| 驗證法 | walk-forward expanding window；top-20 等權多頭 vs universe 等權 |
| 成本 | log cost drag `cost_per_rebal=0.006`(Tier-3 揭露) |

---

## 二、4-Horizon × 3-Seed 聚合結果(median，[min–max] range)

| Horizon | net 年化 | Sharpe | Eff t ⭐ | Win | IC | IR | MDD/panel |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| weekly 5d | +11.4% [6.3–13.6] | 0.906 [0.82–1.00] | 1.704 [1.23–2.17] | 67.9% [67.0–69.8] | 0.010 [0.008–0.013] | 0.573 | 0.152 |
| monthly 20d | +26.8% [25.2–28.5] | 1.256 [1.23–1.29] | 3.484 [3.35–3.74] | 66.0% [62.3–67.0] | 0.037 [0.036–0.038] | 1.172 | 0.298 |
| quarterly 60d | +22.6% [22.2–23.1] | 2.039 [2.02–2.11] | 6.065 [6.01–6.51] | 73.3% [72.4–77.1] | 0.113 [0.112–0.117] | 2.900 | 0.399 |
| **annual 252d** | **+30.2% [29.8–30.5]** | **4.624 [4.56–4.73]** | **6.020 [5.52–6.11]** | **89.2% [89.2–90.2]** | **0.276 [0.275–0.279]** | **5.984** | 0.458 |

> **趨勢**:net / Sharpe / Eff t / IC / Win **全部隨 horizon 拉長單調上升** — LightGBM 在本特徵集上是**中長週期**模型,週線幾乎無 edge。

---

## 三、精準度 (Precision)

| Horizon | 方向命中率 | top-20 重疊率 | 解讀 |
| :--- | :--- | :--- | :--- |
| weekly | 50.0% | 9.3% | ≈ 隨機,**無方向 edge** |
| monthly | 52.0% | 10.2% | 微弱 edge |
| quarterly | 53.3% | 11.5% | 中等 edge |
| **annual** | **62.9%** | **16.8%** | **明顯 edge**(命中率顯著 >50%) |

---

## 四、信任度 / 可靠度 (Trust / Reliability)

- **跨 seed 穩定性**:極高 — annual net range 僅 **29.8–30.5%**、IC range **0.2754–0.2788**(隨機性極低 → 結果可信,非單次僥倖)。
- **`reliability_ic_stability_cov`**(IC 跨 panel CoV,越低越穩):annual **0.388** / quarterly 0.743 / monthly 2.54 / weekly 10.29 → 長 horizon IC 穩定度遠優於短 horizon。
- **顯著性**:annual / quarterly / monthly 之 `eff_t` 皆 >1.997(p<0.05 顯著);weekly eff_t=1.704 **未達顯著**。

---

## 五、T_CZ-6 Production Gate(median;eff_t≥4.20 ∧ Sharpe≥2.40 ∧ Win≥79%)

| Horizon | Eff t | Sharpe | Win | 裁決 |
| :--- | :--- | :--- | :--- | :--- |
| weekly | 1.704 ❌ | 0.906 ❌ | 67.9% ❌ | 🔴 FAIL |
| monthly | 3.484 ❌ | 1.256 ❌ | 66.0% ❌ | 🔴 FAIL |
| quarterly | 6.065 ✅ | 2.039 ❌ | 73.3% ❌ | 🔴 FAIL(僅 Eff t 過) |
| **annual** | **6.020 ✅** | **4.624 ✅** | **89.2% ✅** | 🟢 **PASS** |

---

## 六、「能否賺錢」裁決

| Horizon | 等級 | 理由 |
| :--- | :--- | :--- |
| weekly | 🟠 微弱 | net +11.4% 但方向命中 ≈50%、eff_t 未顯著 → 不可靠 |
| monthly | 🟡 獲利未達門檻 | net +26.8%、eff_t 3.48 顯著,但未過 T_CZ-6 |
| quarterly | 🟡 獲利未達門檻 | net +22.6%、eff_t 6.07 高度顯著,僅差 Sharpe(2.04<2.40)/ Win(73%<79%) |
| **annual** | 🟢 **Production-grade 獲利** | net **+30.2%**/yr、Sharpe **4.62**、Win **89.2%**、IC **0.276**,**過 T_CZ-6** |

### 總裁決
> **能賺錢 — 且在 annual(252d)horizon 達 production-grade**:扣成本年化 +30.2%、Sharpe 4.62、勝率 89%、方向命中 62.9%,三 seed 結果高度一致(net 僅 ±0.4% 波動)。
> Quarterly / monthly **顯著獲利**(net +22~27%)但未過嚴格 T_CZ-6(Sharpe/Win 不足)。**Weekly 無可靠 edge**,不建議。
> LightGBM 的賺錢能力來自**中長週期排序能力**(IC 隨 horizon 由 0.01 → 0.28),非短線擇時。

---

## 七、來源可追溯 (§一.10)

- 全部數字 trace 至 3 個 seed JSON:`/tmp/lgbm_ref_2013_s{5422,1009,7331}.json`(`_meta.source_traceability="per CLAUDE.md §一.10 — all (b) DB query"`)。
- 特徵 / 報酬 / universe 全部 (b) DB query(`feature_values` source-pure + `TaiwanStockPriceAdj`);**無估算、無 AI 補值**。
- 窗由 `get_canonical_panel_dates()` 資料驅動產生(§14.7-DE 單一引用源)。

## 八、重現指令

```bash
cd /home/hugo/project/stock_backend
for s in 5422 1009 7331; do
  ./venv/bin/python scripts/evaluation/multi_cycle_lightgbm_validation.py \
      --commit --seed $s --output /tmp/lgbm_ref_2013_s${s}.json
done
```

> **作為基準範本**:後續 XGBoost / CatBoost / Ensemble / Transformer / Foundation / 股票專用模型一律產出**本結構**報告(同 universe/窗/metric/gate),最後彙整 master comparison 並排比較。
