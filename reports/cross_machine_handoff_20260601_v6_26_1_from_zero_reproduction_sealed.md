# 跨機接續 Handoff:本機 Linux 完全從零重建 + 3 runbook 順序缺口補正(2026-06-01 v6.26.1 sealed)

**Subject**:用戶換機(Linux/WSL)後繼續本專案 → 程式碼已是 v6.26.0 但本機 DB 為舊 v0.17 → 執行「完全從零」全市場重建至 v0.18 sealed checkpoint
**前序 HEAD**:`9d8c9ec`(v6.26.0 跨機接續快照,另一台機器之 12-PHASE 完成)
**Remote**:`https://github.com/tsaitsangchi/stock_backend`(master)
**Charter**:`reports/系統架構大憲章_v6.1.0.md`(§14.7-DD 12-PHASE / §14.7-DC v0.18 pan-historical source-pure / §0.3-A K-wave)
**Runbook**:`reports/tree_based_from_zero_build_runbook_20260531.md`(本次新增 §〇.A B5-B7)
**AI 工具規則**:`CLAUDE.md`(§一.10 數據真實性 / §一.12 5-min 回報 / §一.13 source-pure / §二.6 SHMM)

> 本文件全部數字 source = 本機 live psql READ-ONLY query 或 程式 stdout / git(per CLAUDE.md §一.10);無記憶推測值。

---

## 一、本封存點核心狀態(DB / git 實證)

### 1.1 重建緣由
- 用戶換機後 `git pull` 將程式碼同步至 v6.26.0(HEAD `9d8c9ec`),但**本機 PostgreSQL DB 仍停在舊 v0.17**(910 core / feature_set_v0_4),非 v6.26.0 之 v0.18。
- v6.24→v6.26 之 DB 重建在另一台機器完成,只有 git 程式碼/報告經 pull 過來,**DB 產物未跟隨**。
- 用戶 explicit 選擇「完全從零(~8-14hr)」:`pg_dump` 備份 → DROP 整個 DB → 依 §14.7-DD PHASE 1→11 從 FinMind 全市場全史重抓重建。

### 1.2 最終 DB 狀態(committed,DB-verified)
```
universe   : core_universe_policy_v0.18_source_pure_panhistorical_gate
             core_count=397 / quarantine_count=730
feature    : feature_set_v0_5 = 96 panels (95 historical 2018-06-15~2026-04-15 + 1 current fs_20260601) × 44 features
models     : model_registry 17 rows
             (lgbm 9 / xgboost 2 / catboost 2 / lightgbm 1 / random_forest 1 / extra_trees 1 / ensemble_tree 1)
raw        : 10 FinMind 表 ~80,966,706 rows (2774 stocks × 全史 1990→2026) + fred_series 24 series / 70,641 rows
DB size    : 14 GB
```

### 1.3 PHASE 11 驗證 + T_CZ-6(9 validators / 9 pass;source = reports/multi_cycle_*_20260601_15*.json)
完整三閾值 T_CZ-6(Eff t≥4.20 ∧ Sharpe≥2.40 ∧ Win≥79%,用 `effective_t_stat` overlap-corrected)**annual horizon 通過 3 model**:

| Validator | Annual Eff t | Annual Sharpe | Annual Win | T_CZ-6 |
|---|---:|---:|---:|:--:|
| xgboost | 4.57 | 4.97 | 90.2% | ✅ |
| xgboost_dedicated | 4.47 | 5.09 | 90.2% | ✅ |
| lightgbm | 4.44 | 4.98 | 91.8% | ✅ |
| lgbm_base / catboost / catboost_ded / ensemble / random_forest / extra_trees | 3.06–4.11 | 4.21–4.92 | 90–92% | · |

- quarterly horizon 全未過(win ~69-77% 略低於 79%;eff_t/sharpe 部分達標)。
- 與 v6.26.0 結論一致:**xgboost 家族最強、3 cell 過 gate**;xgboost annual eff_t 4.57 ≈ v6.26.0 handoff 之 4.526。
- ⚠️ **B4 揭露**:T_CZ-6 非 code-enforced;validator 唯一硬判 `is_significant_p05 = abs(effective_t_stat)>1.997`(全 9 model annual 皆 True)。上表為人工三閾值裁決。

---

## 二、3 個 runbook 空-DB 順序缺口(B5-B7,本次實證 + 已 inscribe runbook §〇.A)

B1-B4(runbook 原有)在「有殘留表」之機器寫成,未在乾淨空 DB 跑過 bootstrap 起手。本次真正空 DB 揭露 3 個**僅空 DB 才觸發**之順序缺口,皆 resolved-by-ordering(未改 production code):

| # | 缺陷 | PHASE | 修法 |
|---|---|---|---|
| **B5** | Stage 1 K-wave gate 查 `fred_series`(≠`FredData`),無人在 PHASE 3 前建 | 3 | PHASE 2-3 間插 `fetch_fred_data.py` |
| **B6** | PHASE 3 無 `--bootstrap` → commit 0 成員 → PHASE 4「無標的」 | 3→4 | PHASE 3 加 `--bootstrap` |
| **B7** | feature_store_builder 寫 `universe_completeness_snapshot`,但該表 + `model_registry` 在 PHASE 11 才建(循環依賴) | 7 | PHASE 7 前先建 `model_registry`(DDL-only)+ `universe_completeness_schema.py --init` |

---

## 三、PHASE 4 全市場同步實證(§二.6 SHMM + §一.12 5-min 回報)

- 耗時 **5h19m**(19,150s);2774 股 × 10 表 × 全史,逐表 dataset-batched / workers=4 / dynamic-quota。
- 配額行為(皆 by-design,非故障):**§7.6 A5 300s 短暫停 ~20 次**(window 達 5500/5500 讓配額自然回收)+ **1 次 §7.4-A 402 cascade 1800s cooldown**(workers 撞硬上限)。實證 FinMind sponsor tier 有效(單股探測 200/success)。
- §一.9 protocol:`/api/v4/user_info` 回 404 → 改單股探測確認 token 有效(非從 error 推測 tier)。

---

## 四、honest disclosure — core=397 vs v6.26.0 handoff 之 914

- 本次 v0.18 = **397 core / 730 quarantine**,**對齊本 runbook §六記載之 PHASE 8 = 398 core / 603 quarantine**;**不對齊** v6.26.0 handoff 之 914 core(914 疑為 v0.17 之 910 誤標,或非 pan-historical 計數)。
- 730 quarantine 全為 §14.7-DC **PAN-HISTORICAL** source-pure gate:任一股在 96 panels 任一個有 imputed 特徵(主要 margin / foreign / eps)即排除。藍籌如 1101 台泥(margin_ratio_60d=859.74 未 imputed)在 core。
- `margin_ratio_60d` 當前 panel imputed 333 檔 ≈ 歷史 panel 321 檔(穩定);730 quarantine 為跨 96 panel 累積之嚴格 pan-historical 結果。
- **模型訓練於此 397 source-pure 宇宙。** 若需對齊 handoff 914,須另查 v6.26.0 當時 imputed 較少之成因(未授權不做)。

---

## 五、產物 / 備份 / 未竟事項

- **備份**:`/tmp/rebuild_from_zero/pre_rebuild_backup_20260601_083200.dump`(DROP 前 v0.17 DB,-Fc 1GB,pg_restore 可還原;33 tables verified)。
- **operational drivers**(非 production,不 commit):`/tmp/rebuild_from_zero/rebuild_from_zero.sh`(PHASE 0-6)+ `resume_7_11.sh`(PHASE 7-11)+ `monitor.sh`。
- **model artifacts**:`data/models/<id>/`(gitignored,本機產物,不在 repo)。
- **未竟(皆未授權不 auto-run)**:
  1. **K-wave 7 特徵未接入 trainer**:已建進 feature store(44 features),但 trainer/validator SPEC=37 不含 → 模型未實際用到 K-wave(與 v6.26.0 一致)。啟用須各 trainer/validator SPEC 補 7 名 → retrain。
  2. **core 397 vs 914 成因**:若要對齊 handoff,須深查 v6.26.0 imputation 差異。
  3. **B5-B7 修法**已 inscribe runbook §〇.A;主憲章 §14.7-DD / CLAUDE.md §一.14 是否補述 B5-B7 待用戶決定。

---

**封存印記**:本 handoff = v6.26.1 本機從零重建 sealed checkpoint;承接 v6.26.0(`9d8c9ec`)。DB 全程無手動補值(只 DROP + 從 API 重建),3 缺口 resolved-by-ordering 未改 production code,core 397 vs handoff 914 已 honest disclosure(§一.8)。
