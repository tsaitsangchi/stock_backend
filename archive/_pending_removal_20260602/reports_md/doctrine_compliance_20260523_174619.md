# §0 系統核心思想治權檢驗報告 (Doctrine Compliance Audit)

- generated_at: 2026-05-23 17:46:19
- tool: audit_doctrine_compliance.py v0.4
- constitution: 系統架構大憲章_v6.1.0.md §0
- verdict: **FAILED**
- PASS/WARN/FAIL: 21/2/1
- elapsed: 437.75 ms

- for_promotion: `v6.1.0`

## 四大支柱檢驗摘要

| 支柱 | PASS | WARN | FAIL |
|---|---:|---:|---:|
| §0.1 第一性原則與市場物理學 | 9 | 1 | 1 |
| §0.2 八二法則與不對稱槓鈴 | 4 | 0 | 0 |
| §0.3 康波週期與 2026 雙重共振 | 2 | 1 | 0 |
| §0.4 可觀察性與數位孿生完整性 | 6 | 0 | 0 |

## 檢驗項目明細


### §0.4 可觀察性與數位孿生完整性

- **PASS** `charter_doctrine`: 憲章 §0.1〜§0.4 四大支柱完整存在

### §0.1 第一性原則與市場物理學

- **PASS** `corescore_six_layers`: 六層 CoreScore (DQ/LM/FG/TR/IF/VC) 完整存在於 active policy
- **WARN** `physics_features`: feature_definition 缺少物理量化群 (price/liquidity/institutional): ['price', 'liquidity', 'institutional']
- **PASS** `t3_leakage`: scripts/core/feature_store_builder.py 無 T3 元素洩漏（§0.1-A 禁令 #2/#3 守住）
- **PASS** `t3_leakage`: scripts/core/model_trainer.py 無 T3 元素洩漏（§0.1-A 禁令 #2/#3 守住）
- **PASS** `t3_leakage`: scripts/core/prediction_engine.py 無 T3 元素洩漏（§0.1-A 禁令 #2/#3 守住）
- **PASS** `t3_leakage`: scripts/core/portfolio_sizer.py 無 T3 元素洩漏（§0.1-A 禁令 #2/#3 守住）
- **PASS** `t3_leakage`: scripts/pipeline/portfolio_optimizer.py 無 T3 元素洩漏（§0.1-A 禁令 #2/#3 守住）
- **PASS** `t3_leakage`: scripts/pipeline/portfolio_strategy.py 無 T3 元素洩漏（§0.1-A 禁令 #2/#3 守住）
- **PASS** `t3_leakage`: scripts/pipeline/portfolio_backtest.py 無 T3 元素洩漏（§0.1-A 禁令 #2/#3 守住）
- **PASS** `t3_leakage_summary`: §0.1-A 禁令 #2/#3 全部守住；掃描 7 個模組無 T3 元素實作
- **FAIL** `proxy_transparency`: 無 committed feature_definition；§0.1 proxy 透明度無從驗證

### §0.2 八二法則與不對稱槓鈴

- **PASS** `right_tail_size`: core+convex=150 (core 120 + convex 30) 符合槓鈴右尾規模
- **PASS** `left_tail_isolation`: quarantine_universe=378 已執行左尾剔除
- **PASS** `middle_observation`: research_universe=2243 中段觀測池規模合理
- **PASS** `no_drift_guard`: §6.8 annual rebalance guard 已實作於 core_universe_builder

### §0.3 康波週期與 2026 雙重共振

- **PASS** `theme_keywords`: THEME_KEYWORDS 涵蓋第六波 MBNRIC 核心主題（半導體/生技/醫療/綠能）
- **WARN** `macro_features`: feature_definition macro 群缺少: ['macro_dff_level', 'macro_vix_level', 'macro_t10y2y_level', 'macro_unrate_yoy']
- **PASS** `fred_macro_data`: FredData 完整含四核心序列 (DFF/VIXCLS/T10Y2Y/UNRATE)

### §0.4 可觀察性與數位孿生完整性

- **PASS** `log_tables`: 混合日誌活躍: pipeline_execution_log=28, data_audit_log=22089
- **PASS** `lifecycle_usage`: 所有運行型 charter 模組皆使用 record_lifecycle
- **PASS** `no_hardcoded_perfect`: charter 模組無硬編 PERFECT 違憲字串
- **PASS** `sql_ssot`: §6.7 SQL 由 db_utils.get_core_stocks_from_db 集中提供
- **PASS** `promotion_gate`: target=v6.1.0 (Operations Reality) §0.0-I.9 跨平台依賴宣告已在 requirements.txt 標頭實作

## §0.7 升版規則對照

本工具實作憲章 §0.7「升版提案必須附 §0 四大支柱治理檢驗報告；無法明示對映即不得進入正式 review」。

- `PERFECT` (FAIL=0, WARN=0)：升版可進入正式 review。
- `WARNING` (FAIL=0, WARN>0)：升版可進入 review，但需明文解釋每項 WARN。
- `FAILED` (FAIL>0)：升版必須阻擋；任一 FAIL 即違反 §0.7。