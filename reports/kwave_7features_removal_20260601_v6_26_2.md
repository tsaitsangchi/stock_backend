# §0.3-A K-wave 7 IC-pending 特徵移除記述(2026-06-01 v6.26.2 sealed)

**Subject**:用戶治權 directive「7 個特徵是『死重』應排除,没用的特徵值反而導致模型失準」之落地
**前序**:`v6.26.1`(`3c5d021`,本機從零重建);本次為其後續特徵簡化
**對齊**:§0.3-A 多尺度循環思想修正案 / §14.7-DC T_DC-30 / CLAUDE.md §一.13(source-purity + 簡約)
**資料真實性**:全部數字 source = 本機 live psql query / 程式 stdout(§一.10)

---

## 一、移除標的:7 個 K-wave IC-pending 特徵

| 群 | 特徵 |
|---|---|
| cycle_phase(4)| `cycle_phase_5d` / `cycle_phase_20d` / `cycle_phase_60d` / `cycle_phase_252d` |
| macro_beta(3)| `macro_beta_t10y2y` / `macro_beta_unrate` / `macro_beta_ipg3344s` |

## 二、為何移除(治權依據)

- 這 7 個特徵由 v6.25.0(§0.3-A / T_DC-30)加入 feature store,**設計身分為「IC-pending 候選」**:`feature_store_builder` 標頭明載「canonical SPEC 仍 37(gate/trainer 不選),7 新特徵俟 PHASE 9 IC-gate(ablation drop_minus_full<-0.01)始 promote」。
- **但 canonical SPEC 始終 37**(trainer/validator/gate/audit 皆不選),**IC-gate ablation 從未實跑** → 7 特徵長期懸置為「死重」:白算、佔 store、模型從未使用。
- 依 §0.3-A **自身 promote-only-if-proven 設計**:未通過 IC-gate 之候選不應 promote。本次將「未升格」formalize 為「移除」,避免死重 + 多特徵過擬合風險(尤其小宇宙 × 8-panel 訓練資料量有限)。
- **§0.3-A K-wave 多尺度循環思想本身保留**;`core_universe_builder` Stage 1 macro 存在性 gate(查 `fred_series`)保留(屬 selection 層、非特徵)。此為「7 個特定個股級投影特徵未 earn 到位置而移除」,**非推翻 §0.3-A doctrine**。

## 三、做法(改 writer code + DELETE 過期列,per [[no-manual-data-fill]])

1. **Code**:`feature_store_builder.py` v0.7 → **v0.8**:移除 7 個 FEATURE_DEFINITIONS entries + 3 處計算呼叫(`_cycle_phase` ×4 / `_load_macro_factor_series` / `_compute_macro_beta_features`);保留 audit-trail 註解 + dead-code 函式(符合本檔 deprecated 慣例)。py_compile PASS;dry-run 確認 features defined = 37。
2. **DB(交易式,刪除後驗證列數精確符合才 commit)**:
   - `feature_values` DELETE K-wave 列:**1,015,359**
   - `feature_definition` DELETE K-wave 列:**672**
   - `feature_store_snapshot` feature_count 44 → 37:**96 panels**
3. **為何用 DELETE 而非「重建」**:(a) commit 為 upsert,重建清不掉既有 K-wave 列;(b) 重建會讀最新 v0.18 宇宙、將 panels 從 1584 縮成 397 股(超出「只移除 K-wave」範圍)。DELETE 精準只動 K-wave。

## 四、最終狀態(DB-verified)

| 項目 | 移除前 | 移除後 |
|---|---|---|
| feature_set_v0_5 distinct features | 44 | **37** ✅ |
| 殘留 K-wave 列 | 1,015,359 | **0** ✅ |
| feature_store_snapshot.feature_count | 44 | **37** ✅ |
| panels 股數(current)| 1584 | **1584(保留)** ✅ |
| v0_5 panels | 96 | **96(不變)** |
| model_registry | 17 | **17(不變)** |

## 五、重要:模型完全不受影響

模型/validator 本來就只用 canonical SPEC 37、從未選取這 7 個 K-wave。故:
- **17 models + 9 validators 之 v6.26.1 實證結果仍完全有效**(未被推翻);
- **無需 retrain / re-validate**(訓練資料 byte-identical);
- 本次純為 store 簡化 + 了結 IC-pending 死重 + 誠實 metadata。

## 六、尚未做(governance follow-up,需用戶決定)

- 主憲章 §0.3-A / T_DC-30 是否補述「7 IC-pending 特徵 IC-gate 未跑 → 移除」之記述(本 reports 檔已為 audit record;charter inscription 為更深治權步驟)。

---

**封存印記**:v6.26.2 = v6.26.1 之後續特徵簡化 sealed checkpoint;§0.3-A doctrine 保留,7 個未升格 IC-pending 特徵移除;模型不變;DB DELETE 過期列(交易驗證)+ code v0.8。
