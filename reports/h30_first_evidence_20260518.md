# First H30 Historical Evidence Report (2026-05-18)

- **generated_at**: 2026-05-18 14:03 Asia/Taipei
- **constitution**: `reports/系統架構大憲章_v6.0.0.md` §9.1 / §14.7-R
- **purpose**: 朝 §9.1「30 個交易日 forward-return 預測」終極目標前進的第一筆 h30 evidence
- **verdict**: READY_FOR_DRAFT_EVIDENCE

## Scope

- as_of_date: `2026-03-31`
- label_horizon: 30 (calendar days, 對齊現行 trainer 實作)
- predicted label_date: `2026-04-30`
- max DB price date: `2026-05-15` (≥ label_date ✅)
- universe_snapshot: `core_universe_20260515_core_universe_policy_v0_2`

## Pipeline Steps

| Step | Result |
|---|---|
| Step 9 Feature Store | `fs_20260331_feature_set_v0_1_h30_historical_20260331` committed；preflight 14/0/0；150 stocks × 27 features × 3975 rows / 58 imputed / 671ms / **PERFECT** |
| Step 10 Model Trainer | `mdl_20260331_lgbm_h30_e5d97ec0_v0_1` committed；rows_trained=147；**IC_mean=0.4090**；RMSE=0.4467；label_date=2026-04-30 / 1389ms / **PERFECT** |
| Step 11 Prediction | `pred_20260331_mdl_20260331_lgbm_h30_e5d97ec0_v0_1` committed 後依 §8.8.8 標記 deprecated；150 predictions / 122ms |

## Top Rank-IC Features (h30 vs h20 比較)

| Feature | rank_IC (h30) | 比較 h20（2026-04-25 model） |
|---|---:|---|
| volatility_252d | **0.3590** | 流動性/波動性主導 h30 |
| avg_daily_value_log_252d | 0.3429 | 同上 |
| avg_daily_value_log_60d | 0.3219 | — |
| volatility_60d | 0.3124 | — |
| revenue_yoy_3m | 0.2281 | 月營收成長 |
| log_return_252d | 0.2208 | 趨勢 |
| turnover_mean_60d | 0.2206 | 流動性 |
| eps_sum_4q | 0.2183 | 獲利能力 |
| log_return_60d | 0.2142 | 中期趨勢 |
| max_drawdown_252d | 0.2030 | 風險 |

**觀察**：h30 horizon 下，年度視窗（252d）特徵之 rank IC 普遍高於 h20，顯示長視窗 horizon 更受 long-term liquidity / volatility 特徵主導。

## Compliance Check (after tool upgrades)

| Audit | Result | Notes |
|---|---|---|
| `audit_leakage.py` **v0.2** | ✅ 18/0/0 PERFECT | 新增 ALLOWED_LABEL_HORIZONS = {20, 30} |
| `audit_downstream_readiness.py` **v0.2** | ✅ 29/1/0 READY_FOR_DRAFT_EVIDENCE | 同上 |
| Sole WARN | production-current label window | 等 2026-06-04 |

## §8.8.8 Compliance Pattern

- **model_registry**: 1 個 h30 model committed (作為 v6.2.0 evidence)
- **prediction_run**: deprecated h30 prediction（避免「prediction-backed 數量 > 1」FAIL）
- **唯一 prediction-backed delivery**: 仍為 `pred_20260425_mdl_20260425_lgbm_h20_d969ffb1_v0_1` (h20, IC=0.3716)
- 對齊 §8.8.8 與 §8.8.10「Final Delivery Index」模式

## Decision

1. **§9.1 v6.2.0 預備證據成立**：第一個 h30 horizon model 證明 trainer/feature/prediction 三層 pipeline 對 horizon=30 工作正常
2. **Top features 揭露**：h30 horizon 下流動性/波動性主導；對未來 §9.2 portfolio sizing 之凸性配置具參考價值
3. **§9 終極目標可路徑驗證**：從 h20 (v6.1.0) → h30 (v6.2.0) 跨度已有第一筆乾淨實證
4. **不取代 h20 production-current delivery**：仍等 2026-06-04 完成 v6.1.0 升版
5. **下個 §9 里程碑**：v6.2.0 升版前須補 walk-forward h30 panel（建議 6-12 點）建立 IC 穩定性證據

## Tool Upgrades This Run

| Tool | Version | Change |
|---|---|---|
| `audit_leakage.py` | v0.1 → **v0.2** | 新增 ALLOWED_LABEL_HORIZONS = {20, 30}；relax `missing_or_bad_horizon` 至 multi-horizon |
| `audit_downstream_readiness.py` | v0.1 → **v0.2** | 同上；保留 FORMAL_LABEL_HORIZON=20 為 production-current gate |
