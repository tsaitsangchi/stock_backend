# XGBoost Multi-Cycle 驗證報告(模型 #2)

**模型**：XGBoost(`xgb.train` hist / 200 trees / depth 5 / lr 0.05)| **類別**：Tree
**最後更新**：2026-06-02
**比較基準**：依 `common_model_comparison_baseline_20260602.md`(同一 universe / 窗 / horizon / metric / gate)
**Metric 來源**：`core.db_utils.summarize_horizon_metrics()`(§14.7-DF 單一計算引用源 — 與 LightGBM **同一 helper**,完全可比)
**資料來源**：全 (b) DB query(§一.10)| **Seeds**：5422 / 1009 / 7331(取 median + range，§一.10 #3)

---

## 一、運行配置(= 共同比較基準)

| 項目 | 值 |
| :--- | :--- |
| Universe | 397 source-pure 核心股(§14.7-DC) |
| 特徵集 | `feature_set_v0.5`，37 source-pure 特徵 |
| 驗證窗 | 157 panels / 2013-05-15 ~ 2026-06-01(`get_canonical_panel_dates()`，§14.7-DE) |
| Horizons | weekly 5d / monthly 20d / quarterly 60d / annual 252d |
| 組合 | top-20 等權多頭 vs universe 等權；成本 `cost_per_rebal=0.006` |

---

## 二、4-Horizon × 3-Seed 聚合結果(median，[min–max] range)

| Horizon | net 年化 | Sharpe | Eff t ⭐ | Win | IC | IR | MDD/panel |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| weekly 5d | +9.4% [9.1–13.5] | 0.878 [0.86–0.98] | 1.611 [1.44–2.02] | 67.9% [66.0–69.8] | 0.011 [0.009–0.012] | 0.542 | 0.166 |
| monthly 20d | +22.8% [22.2–24.3] | 1.178 [1.12–1.20] | 3.015 [2.66–3.13] | 63.2% [61.3–67.0] | 0.037 [0.034–0.038] | 1.015 | 0.286 |
| quarterly 60d | +23.1% [22.4–24.1] | 2.071 [2.03–2.25] | 6.050 [5.94–7.15] | 74.3% [74.3–74.3] | 0.123 [0.117–0.125] | 2.892 | 0.394 |
| **annual 252d** | **+32.7% [32.3–33.3]** | **4.631 [4.57–4.72]** | **5.768 [5.76–5.86]** | **90.2% [89.2–91.2]** | **0.297 [0.296–0.298]** | **5.734** | 0.591 |

> 趨勢同 LightGBM:net / Sharpe / Eff t / IC / Win 隨 horizon 拉長單調上升;XGBoost 為**中長週期**排序模型。

---

## 三、精準度 + 信任度

| Horizon | 方向命中 | top-20 重疊 | IC-CoV(越低越穩)| 解讀 |
| :--- | :--- | :--- | :--- | :--- |
| weekly | 50.1% | 9.3% | 9.00 | 無方向 edge |
| monthly | 51.9% | 10.0% | 2.51 | 微弱 |
| quarterly | 53.6% | 11.3% | 0.675 | 中等 |
| **annual** | **63.3%** | **18.3%** | **0.395** | 明顯 edge + 高穩定 |

**跨 seed 穩定性極高**:annual net range 僅 **32.3–33.3%**、Sharpe **4.57–4.72**、IC **0.296–0.298** → 隨機性極低,可信。

---

## 四、T_CZ-6 Gate(median)+ 賺錢裁決

| Horizon | Eff t | Sharpe | Win | Gate | 賺錢裁決 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| weekly | 1.611 ❌ | 0.878 ❌ | 67.9% ❌ | 🔴 | 🟠 微弱(hit≈50%、不顯著) |
| monthly | 3.015 ❌ | 1.178 ❌ | 63.2% ❌ | 🔴 | 🟡 獲利未達門檻(net +22.8%、eff_t 顯著) |
| quarterly | 6.050 ✅ | 2.071 ❌ | 74.3% ❌ | 🔴 | 🟡 獲利未達門檻(net +23.1%、eff_t 高度顯著) |
| **annual** | **5.768 ✅** | **4.631 ✅** | **90.2% ✅** | 🟢 **PASS** | 🟢 **Production-grade**(net +32.7%/yr) |

### 總裁決
> **能賺錢 — annual(252d)達 production-grade**:扣成本年化 **+32.7%**、Sharpe **4.63**、勝率 **90.2%**、方向命中 63.3%,3 seed 高度一致。Quarterly/monthly 顯著獲利但未過嚴格 T_CZ-6;weekly 無 edge。

---

## 五、Head-to-Head：LightGBM vs XGBoost(模型 #1 vs #2,同基準)

| Horizon | metric | LightGBM | XGBoost | 領先 |
| :--- | :--- | :--- | :--- | :--- |
| annual | net 年化 | +30.2% | **+32.7%** | XGB |
| annual | Sharpe | 4.624 | **4.631** | ≈(XGB 微勝) |
| annual | Eff t | **6.020** | 5.768 | LGBM |
| annual | Win | 89.2% | **90.2%** | XGB |
| annual | IC | 0.276 | **0.297** | XGB |
| quarterly | Sharpe | 2.039 | **2.071** | XGB |
| quarterly | Eff t | 6.065 | 6.050 | ≈ |
| monthly | net | **+26.8%** | +22.8% | LGBM |
| weekly | (both 無 edge) | — | — | — |

> **兩模型 annual 皆過 T_CZ-6**,表現同級。XGBoost annual 報酬/IC/勝率略高,LightGBM annual 顯著性(Eff t)與 monthly 報酬略高。兩者高度一致(同特徵、同窗、同 helper)→ 印證訊號來自**資料 + 中長週期排序能力**,非單一模型 artifact。

---

## 六、來源可追溯 + 重現

- 數字 trace 至 `/tmp/xgb_ref_s{5422,1009,7331}.json`(全 (b) DB query);metric 由 `summarize_horizon_metrics()`(§14.7-DF 單一源,與 LightGBM 同 helper)。
- ⚠️ **非決定性註記**(§一.10 #3 / T_DF-4):tree 模型多執行緒訓練 run-to-run 微異 → 已用 3 seed 取 range/median,非單次 anchor。

```bash
for s in 5422 1009 7331; do
  ./venv/bin/python scripts/evaluation/multi_cycle_xgboost_validation.py --commit --seed $s --output /tmp/xgb_ref_s${s}.json
done
```

> 下一步:CatBoost(模型 #3)→ Ensemble → 其餘,最後彙整 master comparison。
