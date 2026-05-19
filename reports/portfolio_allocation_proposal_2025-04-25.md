# Portfolio Allocation Proposal (v0.1)

- **generated_at**: 2026-05-19 21:03:23 Asia/Taipei
- **tool**: portfolio_sizer.py v0.1
- **constitution**: 系統架構大憲章_v6.0.0.md §9.2 / §0.2 / §0.0-A.5
- **prediction_run_id**: `pred_20250425_mdl_20250425_lgbm_h20_5c7f36c2_v0_1`
- **model_id**: `mdl_20250425_lgbm_h20_5c7f36c2_v0_1`
- **as_of_date**: 2025-04-25
- **universe_snapshot_id**: `core_universe_20260514_core_universe_policy_v0_2`
- **prediction_policy**: prediction_policy_v0.1
- **sizing_policy**: sizing_policy_v0.1

## 1. 配置摘要 (Allocation Summary)

| 指標 | 值 |
|---|---:|
| 攻擊端總權重 | 0.2000 (20.00%) |
| 防禦端 (CASH) | 0.8000 (80.00%) |
| 配置股票數 | 6 |
| 候選但未配置 | 14 |

### Tier 配置分布

| Tier | 總權重 | 股數 |
|---|---:|---:|
| convex_universe | 0.1500 | 5 |
| core_universe | 0.0500 | 1 |
| **CASH (safety)** | **0.8000** | n/a |

### Sector 配置分布

| Sector | 總權重 | 股數 | cap (≤0.4) |
|---|---:|---:|---:|
| 半導體業 | 0.2000 | 6 | ✅ |

## 2. 配置明細 (Allocation Details)

| Rank | Stock | Name | Tier | Sector | Pred Value | Signal | Conf | Weight | Reason | Risk Flags |
|---:|---|---|---|---|---:|---|---:|---:|---|---|
| - | **CASH** | safety sleeve | safety | - | - | - | - | **0.8000** | §0.2 防禦端 (≥80%) | - |
| 1 | `6643` | M31 | convex_universe | 半導體業 | 0.6585 | long | 0.9867 | 0.0300 | tier=convex_universe | cap=3% | - |
| 2 | `6237` | 驊訊 | core_universe | 半導體業 | 0.6182 | long | 0.9733 | 0.0500 | tier=core_universe | cap=5% | - |
| 3 | `3443` | 創意 | convex_universe | 半導體業 | 0.6039 | long | 0.9600 | 0.0300 | tier=convex_universe | cap=3% | - |
| 4 | `3661` | 世芯-KY | convex_universe | 半導體業 | 0.5875 | long | 0.9467 | 0.0300 | tier=convex_universe | cap=3% | - |
| 5 | `4971` | IET-KY | convex_universe | 半導體業 | 0.5569 | long | 0.9333 | 0.0300 | tier=convex_universe | cap=3% | - |
| 6 | `3374` | 精材 | convex_universe | 半導體業 | 0.5562 | long | 0.9200 | 0.0300 | tier=convex_universe | cap=3% | - |
| 7 | `4749` | 新應材 | convex_universe | 半導體業 | 0.5475 | long | 0.9067 | 0.0000 | attack_cap_exhausted | attack_budget_full |
| 8 | `6683` | 雍智科技 | core_universe | 半導體業 | 0.5459 | long | 0.8933 | 0.0000 | attack_cap_exhausted | attack_budget_full |
| 9 | `8054` | 安國 | core_universe | 半導體業 | 0.5414 | long | 0.8800 | 0.0000 | attack_cap_exhausted | attack_budget_full |
| 10 | `6568` | 宏觀 | core_universe | 半導體業 | 0.5392 | long | 0.8667 | 0.0000 | attack_cap_exhausted | attack_budget_full |
| 11 | `6679` | 鈺太 | core_universe | 半導體業 | 0.5275 | long | 0.8533 | 0.0000 | attack_cap_exhausted | attack_budget_full |
| 12 | `8086` | 宏捷科 | convex_universe | 半導體業 | 0.5272 | long | 0.8400 | 0.0000 | attack_cap_exhausted | attack_budget_full |
| 13 | `4966` | 譜瑞-KY | core_universe | 半導體業 | 0.5271 | long | 0.8267 | 0.0000 | attack_cap_exhausted | attack_budget_full |
| 14 | `3707` | 漢磊 | core_universe | 半導體業 | 0.5154 | long | 0.8133 | 0.0000 | attack_cap_exhausted | attack_budget_full |
| 15 | `6462` | 神盾 | core_universe | 半導體業 | 0.4920 | long | 0.8000 | 0.0000 | attack_cap_exhausted | attack_budget_full |
| 16 | `4991` | 環宇-KY | convex_universe | 半導體業 | 0.4896 | long | 0.7867 | 0.0000 | attack_cap_exhausted | attack_budget_full |
| 17 | `6533` | 晶心科 | core_universe | 半導體業 | 0.4668 | long | 0.7733 | 0.0000 | attack_cap_exhausted | attack_budget_full |
| 18 | `5443` | 均豪 | convex_universe | 半導體業 | 0.4244 | long | 0.7600 | 0.0000 | attack_cap_exhausted | attack_budget_full |
| 19 | `5351` | 鈺創 | core_universe | 半導體業 | 0.3920 | long | 0.7467 | 0.0000 | attack_cap_exhausted | attack_budget_full |
| 20 | `3105` | 穩懋 | convex_universe | 半導體業 | 0.3814 | long | 0.7333 | 0.0000 | attack_cap_exhausted | attack_budget_full |

## 3. Sizing Policy v0.1 規則

- attack_total_weight_max: 0.2
- safety_total_weight_min: 0.8
- single_stock_weight_max: 0.05
- convex_tier_weight_max: 0.03
- sector_weight_max: 0.4
- required_coverage: 150
- max_committed_runs: 1

## 4. 治權邊界宣告

- 本程式只讀 committed prediction_run / prediction_values / core_universe_membership；
  **不**重選 universe、**不**重訓 model、**不**重算 prediction、**不**修改任何 raw 表。
- 本配置為 dry-run / report proposal，**非**投資建議；signal_label='long' 為訊號標籤，
  不等於買賣指令。
- §0.1-A 永久禁令 #2/#3 守住：本程式未實作 IFF Θ / SOC / 重力井邊緣觸發。
- §0.2 槓鈴策略：攻擊端 ≤20% + 防禦端 ≥80% + 單股 ≤5% (convex ≤3%) + sector ≤40%。
- §0.0-G 憲章先行紀律對映：本程式為 §0.0-A.5 第五個轉換器；
  治權邊界明文限定於「formal prediction → allocation proposal」。

## 5. 執行紀錄

- ✅ [PASS] committed prediction run located: pred_20250425_mdl_20250425_lgbm_h20_5c7f36c2_v0_1 (model=mdl_20250425_lgbm_h20_5c7f36c2_v0_1)
- ✅ [PASS] prediction_values loaded: rows=150
- ✅ [PASS] membership loaded: rows=150 (snapshot=core_universe_20260514_core_universe_policy_v0_2)
- ✅ [PASS] prediction × membership join complete (150/150)
- ✅ [PASS] candidates filtered: 20 long signals
- ✅ [PASS] sizing policy v0.1 applied: attack_total=0.2000, cash=0.8000, allocated_stocks=6
