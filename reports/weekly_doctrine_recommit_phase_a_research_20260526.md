# Weekly Doctrine-Driven Recommit — Phase A Design Research(§14.7-BX)

**日期**: 2026-05-26
**Phase**: A(設計研究 / pre-charter inscription)
**對應軌道**: v6.4.0 或 v7.0.0(治權頻率層升版;與 §6.8 年度頻率 institutional stability 衝突解析)
**對應憲章基礎**: §14.7-BW(第二十一輪 Path D Pure Doctrine)/ §6.8 制度穩定性 / §6.8.1 frequency spirit / §14.7-BT 第十八輪 §6.7.1 annex / §8 三層職責邊界
**Status**: ✅ Phase A 完整(13 章 / non-destructive / 不動 DB 不動 code)
**對應 user trigger**: 2026-05-26 「每週跑 builder v0.10 + commit 新 snapshot 為 current committed。改憲章」

---

## 1. 觸發

§14.7-BW Phase A+B closure(本 session 同次)後,用戶提問「可以寫一支程式每週收盤後來重新跑核心股嗎?」並選 **Path B(weekly auto-recommit + 改憲章)**,觸發本 §14.7-BX 設計研究。

### 1.1 用戶選擇之治權含義

| 路徑 | 內容 | User 選定 |
|---|---|---|
| Path A | Weekly 純 scan(讀,不寫)+ 年度 commit | ❌ rejected |
| **Path B** | **Weekly auto recommit(違 §6.8;需改憲章)** | ✅ **selected** |
| Path C | Weekly scan + drift-triggered special commit | ❌ rejected |

Path B 為**最 pure-doctrine consistent** 路徑:doctrine 為「per-stock × 三基柱資料源依據」,資料源狀態為連續變動(stock 新上市 / 財報新發布 / FRED 指標新 sync 都改變 pass-set),annual 頻率讓中年新滿足 doctrine 之 stocks 無法進入,屬於「frozen for a year」之 implicit fixed-state hardcode。**Path B 解此 hidden hardcode**。

---

## 2. §6.8 vs §14.7-BW 之 first-principles 衝突分析

### 2.1 §6.8 制度穩定性 / §6.8.1 spirit(既有 inscribed treaty)

§6.8 之 「每年最後一個交易日後一次性」+ §6.8.1 之「非日/週/月頻 re-commit」設立 rationale:

| 設立理由 | 治權考量 |
|---|---|
| 避免 universe churn 破壞下游 | model train 用 universe X / predict 變 Y → distribution shift |
| 治權穩定性 institutional stability | 治理者 review 負擔降低 |
| 減少 decision noise | 年度 review 累積足夠 signal-to-noise |
| 對齊 §6.8 制度穩定性精神 | 不可日/週/月頻 churn |

### 2.2 §14.7-BW Pure Doctrine(本 session 第二十一輪 inscribed)

§14.7-BW 之 doctrine:per-stock × 三基柱資料源依據;N 為 doctrine 結果非預設。

**隱含時間維度**:doctrine 是 **per timestamp** 概念 — 某 t 時點之 pass-set 是 t 時點之 data 狀態決定。t1 vs t2 之 pass-set 可能不同(新股上市 / 新財報 / 新 sync)。

### 2.3 衝突的核心

**§6.8 將 t1 時點之 doctrine result 凍結至 t1+365**;§14.7-BW 之 doctrine 在 t1+x(0 < x < 365)之 pass-set 可能與 t1 不同,但 frozen state 不允許更新 → **§6.8 violates §14.7-BW(per timestamp doctrine)**。

換言之:**§6.8 之年度 freeze 本身為一種 hidden hardcode**(凍結 t1 之 pass-set 為 fixed-state up to t1+365);與 §14.7-BW 取消所有 hardcode 之 doctrine 不一致。

### 2.4 衝突解法

| 解法 | 含義 | 衝突殘留 |
|---|---|---|
| A: 升 §6.8 為 SSOT(年度 freeze)| doctrine 退讓;current pass-set 可凍結 | doctrine 之 timestamp dimension 失效 |
| B: 升 §14.7-BW 為 SSOT(weekly recommit)| §6.8 之 frequency spirit 退讓;但 institutional stability 之治權考量需另解 | downstream churn 風險(需另設 mitigation)|
| C: 並存(special_rebalance trigger)| 折衷;但 trigger 規則本身為 implicit hardcode(threshold)| 部分對齊 / 部分 hardcode |

**Path B 用戶選定**:升 §14.7-BW 為 SSOT;§6.8 之 frequency 降為「v0.7-v0.8 era prescription」(historical record);新 frequency 為 weekly。

---

## 3. §14.7-BX Weekly Doctrine-Driven Recommit canonical spec

### 3.1 頻率定義

**Recommit 頻率**:每週(weekly);具體時點為**每週收盤後**(對齊台股交易日;通常週五 13:30 後 / 若週五為國定假則順延至最近交易日)。

**為何 weekly 而非 daily**:
- Daily 之 raw data update 仍未必同步(financial statements 季度;monthly revenue 月度;institutional / margin 日度)
- Weekly 給 raw data sync 之自然 buffer
- 仍滿足 doctrine 之「per timestamp 變動敏感」

**為何不是 monthly / quarterly**:
- Monthly 之 latency 對 doctrine-state-change(新股上市 / 新財報)反應慢
- Weekly 為「最低粒度滿足 doctrine」+「最高粒度尚可控制下游 churn」之折衷

### 3.2 Recommit 動作

每週末跑:
1. `fetch_*` syncing(若 cron 未獨立跑)— 補齊本週新 raw data
2. `compute_semi_supply_cycle_proxy.py` + (shipping) — 補齊 §0.3 K-wave proxies
3. `build_doctrine_gate_universe.py --commit` — 跑 Path D doctrine gate + commit 新 weekly snapshot
4. `audit_universe_completeness.py` — 驗收 verdict 為 PERFECT
5. 生 `reports/weekly_universe_recommit_<YYYYMMDD>.md` — drift report(對比上週 snapshot)
6. (optional)若 drift > threshold:trigger downstream re-train pipeline

### 3.3 Snapshot 命名 convention

```
core_universe_<YYYYMMDD>_core_universe_policy_v0.10_pure_doctrine_weekly
```

- `YYYYMMDD`:當週收盤日
- `_weekly` suffix:標記此為 weekly recommit snapshot

### 3.4 既有 snapshot 處置

每週新 snapshot commit 後:
- 前一週 snapshot:status 從 `committed` → **`superseded`**(新狀態值;區別於 `deprecated`/`draft`)
- 更早歷史 snapshots:不動(維持 `superseded` 或 `deprecated`)
- 治權層只承認**最新一個 `committed`**(per §6.7 SSOT SQL contract)

**新 status 值「superseded」之治權含義**:
- `committed`:current active universe
- `superseded`:被新 commit 取代之 historical(audit trail)
- `deprecated`:被治權主動 cancel(per §14.7-BT/BW precedent)
- `draft`:WIP / 未 commit

需於 §14.7-BX Phase C 之 schema 升版加 `superseded` 狀態值(目前 enum 為 {committed, deprecated, draft})。

---

## 4. 對既有治權之處置

### 4.1 §6.8 制度穩定性 之治權處置

**不撤銷,但 frequency clause 追溯修正為「v0.7-v0.8 era prescription」**:
- §6.8 之「每年最後一個交易日後一次性」/「非日/週/月頻 re-commit」之條文保留為 charter narrative
- 在 §14.7-BX 入憲後,**此 frequency rule 在 Path D pure doctrine 下不適用**;reduced 為 historical record
- §6.8.7 全市場全天數限定治理例外、§6.8.7-B 長跑監控規範等其他 §6.8 sub-clauses **不動**(屬 sync infrastructure 治權,非 commit frequency 治權)

### 4.2 §6.8.1 之治權處置

同 §6.8 — 「frequency spirit」之 specific clause 在 Path D 下不適用;保留為 historical record。

### 4.3 §14.7-BT 第十八輪 §6.7.1 annex 之治權處置

§14.7-BT 之「dynamic N 仍服從年度頻率;§6.8.1 spirit 維持」段:
- 在 §14.7-BX 入憲後,**此段在 Path D 下不適用**;reduced 為 v0.7-v0.8 era prescription
- §14.7-BT 主體(取消 §6.7 SSOT 150 hardcode prescription)**不撤銷**
- §6.7.1 annex 結構保留;只其 frequency clause 失效

### 4.4 §14.7-BW Phase D 之 §9.2 portfolio_sizer barbell 議題

**weekly recommit 對 §9.2 v0.4 升版需求加深**:每週 universe 變動 → portfolio_sizer 每週重 size → barbell tier 之 stability 需求更高 → §9.2 v0.4 tier-less 設計成為更迫切之 cross-session work

### 4.5 §8 ACTIVE DRAFT 之治權處置

§8 三層職責邊界(feature_store / model_registry / prediction)在 weekly universe shift 下需:
- Feature Store v0.9+:支援 weekly feature_set 升版(每週 universe 變 → feature 重 build)
- Model Trainer v0.3+:支援 weekly model retrain(每週 universe 變 → model retrain;否則 universe 與 model 不一致)
- Prediction Engine v0.2+:對齊 weekly snapshot

§8 升強制契約之時機推遲至 §14.7-BX Phase C-G 落地後一併處理(per §14.7-BV 第二十輪預定之 v6.4.0/v7.0.0 milestone 範圍延伸)。

---

## 5. 對下游 churn 之影響評估

### 5.1 Model retrain 工作量

| 頻率 | 每年 retrain 次數 | 累積 compute |
|---|---:|---|
| 年度(§6.8 既有)| 1 | 1x |
| **Weekly(§14.7-BX)** | **52** | **52x** |

每週 model retrain → walk-forward IC validation panel 之 8 時點 8x 工作量(每週 8 panels × 52 weeks = 416 panels/year vs 8/year)。

### 5.2 Mitigation(downstream churn 控制)

**Option M1**(推薦):**Incremental retrain mode**
- 每週 universe 變動通常為小幅度(per probe:current N=1862;新增/移除預期 < 10/week)
- Model 採 incremental fine-tune from prior week's checkpoint,非 from-scratch
- 計算成本約 1/10 of full retrain × 52 weeks = 5.2x cost vs annual(可接受)

**Option M2**:**Lazy retrain trigger**
- 每週跑 universe recommit;但 model retrain 僅在 drift > threshold(e.g., 5% pass-set 變動)時觸發
- 平均 retrain 頻率約 quarterly(若 universe 穩定)

**Option M3**:**Parallel walk-forward**
- 維持每週 walk-forward IC 計算(small N panels);不重 train from-scratch model
- Model 持續但 universe shift

§14.7-BX Phase B 入憲不選定 sub-option;留 Phase C-D 之 model_trainer / feature_store 升版時專案研究(類比 §14.7-BV Phase D 之 §9.2 v0.4 sub-option 待定 pattern)。

### 5.3 Compute / Storage 成本

| 維度 | 年度 | **Weekly** |
|---|---:|---:|
| Snapshots 累積/year | 1 | 52 |
| Membership rows 累積/year | ~2,800 | ~145,600 |
| universe_completeness_snapshot rows/year(3 pillars × N × data layer)| ~5,586 | ~290,500 |
| DB storage/year | 既有 | +50x snapshot rows / +50x completeness rows |

**Mitigation**:
- 舊 `superseded` snapshots 之 membership 可 archive(壓縮儲存或移至 cold storage)
- universe_completeness_snapshot 可只保留最近 12 週 + monthly summary
- 屬 Phase E audit / storage 升版 scope

---

## 6. 對 §0.0-B/C/D 跨層完整度基線影響

§14.7-BX 入憲後,§0.0-B 第一性原理 / §0.0-C 八二法則 / §0.0-D 康波週期 之跨層完整度基線:

| 基柱 | 影響 |
|---|---|
| §0.0-B(第一性原理)| +1pp(weekly enforce per-stock §0.1 doctrine 強化)|
| §0.0-C(八二法則)| 不變 |
| §0.0-D(康波週期)| +0.5pp(weekly K-wave market prerequisite enforce 強化)|

預期 §0.0-D 完整度:目前 85-88%(post §14.7-BV)→ 89-91%(post §14.7-BX Phase C-G)。

---

## 7. Phase A-G Roadmap

| Phase | 內容 | 時間 | Status |
|---|---|---|---|
| **A** | Design research(本報告) | ~1h | ✅ **本 entry** |
| **B** | 入憲 §14.7-BX + 修訂歷程 第二十二輪 | ~30 min | ⏸ next(本 session 可推) |
| **C** | 建 weekly recommit cron / launchd script + Snapshot schema 加 `superseded` status enum | ~2-3h | ⏸ next session |
| **D** | model_trainer / feature_store_builder weekly mode 升版(per §5.2 Mitigation sub-option 選定後)| ~5-8h | ⏸ cross-session |
| **E** | audit_universe_completeness.py + audit_core_universe.py 加 weekly snapshot 識別 | ~1h | ⏸ cross-session |
| **F** | First 4-week integration test(觀察 drift / model IC stability)| ~1 month(real-time)| ⏸ post-deploy |
| **G** | v6.4.0 / v7.0.0 milestone tag(若 §14.7-BV/BW/BX 共構 minor → v6.4.0;若 major → v7.0.0)| ~30 min | ⏸ closure |

**總計**: ~10-15h scripting + 1 month observation

---

## 8. §14.7-BX 治權新特性(predicted)

1. **首例「治權頻率層升版」**(§6.8 之 commit frequency 從年度 → weekly;類比 §14.7-BW 之 doctrine 純化但 scope 為 temporal dimension)
2. **首例「§6.8 frequency clause 追溯為 historical narrative」**(類比 §14.7-BW 之 §6.7.1 cap deprecation pattern)
3. **首例「治權判準 fifth-round refinement」**(BT cancel hardcode → BU build governance → BV change criterion → BW pure-form refinement → BX temporal-dimension refinement;五輪累進)
4. **首例「snapshot status enum 升版」**(committed/deprecated/draft → +superseded;為 weekly auto-recommit 之治權預備)
5. **首例「downstream churn vs doctrine purity 治權平衡」**(下游 mitigation sub-option 待 Phase D 選定;類比 §14.7-BV Phase D 之 §9.2 v0.4 模式)

---

## 9. 證偽承諾 T_BX-1〜5

- T_BX-1: §14.7-BX Phase C 落地後,weekly recommit cron 每週執行一次,連續觀察 ≥ 4 週皆有新 committed snapshot(與上週 different snapshot_id)
- T_BX-2: 每週 commit 後,前週 snapshot status 為 `superseded`,當週為 `committed`,total `committed` 始終 = 1(per §6.7 SSOT)
- T_BX-3: 連續 4 週 universe pass-set 變動率 < 5%(穩定性 sanity check;若 > 5% 表 §0.1 source 急劇變動或 §0.3 indicator sync 不穩)
- T_BX-4: weekly recommit 之 audit_universe_completeness.py verdict 持續 PERFECT(C1-C4 schema strict + C5-C12 data integrity)
- T_BX-5: charter 之 active prescription 中,「年度頻率」/「annual rebalance」/「non-weekly」之語句 count 對 weekly recommit 文脈 = 0(僅在 §6.8 / §6.8.1 / §14.7-BT 之 historical narrative 中保留)

---

## 10. Risks / Mitigation

| Risk | 機率 | 影響 | Mitigation |
|---|---|---|---|
| R1: Model IC stability 受 weekly universe churn 破壞 | 高 | 高 | Phase D 之 sub-option(M1 incremental / M2 lazy / M3 parallel)選定 |
| R2: DB storage 爆炸(weekly snapshots × N rows)| 高 | 中 | Phase E 加 storage archive policy / monthly summary |
| R3: cron / launchd 失敗 → 某週未 commit | 中 | 中 | SHMM 心跳監控(per §14.7-AX);failure detect & alert |
| R4: §6.8 既有 inscribed treaty 之追溯衝突 | 中 | 低 | §14.7-BX Phase B 明文 supersession;§6.8 條文不撤銷;只 frequency clause 失效 |
| R5: 用戶後續再深化 doctrine(e.g., daily recommit)| 低 | 中 | weekly 已為「最低粒度滿足 doctrine」+「最高粒度可控 churn」之平衡;再升 daily 需新 §14.7-BY 入憲 |
| R6: Phase D 之 sub-option 選定延宕,weekly recommit 跑了但下游 model 沒重 train → 不一致 | 高 | 高 | Phase C 落地 weekly cron 之前須先 §5.2 Mitigation sub-option 治權選定;不可顛倒順序 |

---

## 11. 入憲建議

### 11.1 新節 §14.7-BX(per §14.7-X 模式)

子節內容大綱(10-11 章):
1. 觸發(user 選 Path B + 改憲章)
2. §6.8 vs §14.7-BW 衝突解析
3. Path B canonical spec(weekly frequency / 收盤後 / snapshot_weekly suffix / superseded status)
4. 對既有治權處置(§6.8 / §6.8.1 / §14.7-BT 之 frequency clause / §9.2 / §8)
5. 對下游 churn 影響評估 + Mitigation sub-options
6. Phase A-G roadmap
7. 治權新特性 5 條
8. 證偽承諾 T_BX-1〜5
9. Risks / Mitigation
10. Cross-Reference 精確行號

### 11.2 修訂歷程 第二十二輪 entry draft

```
v6.1.0-patch 第二十二輪: §14.7-BX Phase B 入憲 — Weekly Doctrine-Driven Recommit(治權頻率層升版;§6.8 年度頻率 → weekly;對應用戶選定 Path B 之「每週跑 builder v0.10 + commit 新 snapshot」直接 directive)。Doctrine 升 SSOT;§6.8 / §6.8.1 / §14.7-BT 之 frequency clause 在 Path D 下不適用(reduced 為 v0.7-v0.8 era prescription)。Weekly recommit canonical spec:每週收盤後 / snapshot_id 加 `_weekly` suffix / 新 status enum `superseded` / 前週 snapshot 自動 supersede。Phase C 落地 cron + schema enum 升版;Phase D 待 sub-option 選定後 model_trainer / feature_store 升版;Phase E audit / storage archive policy;Phase F-G observation + milestone tag。對既有治權影響:§6.8 條文保留為 historical narrative;§9.2 v0.4 升版需求加深;§8 升強制契約推遲至 §14.7-BX C-G 落地後一併。
```

---

## 12. Cross-Reference 精確行號(預期)

§14.7-BX 子節入憲位置:charter 之 §14.7-BW(L9550)之後 / §20 創世圓滿宣言之前。具體行號待 Phase B 入憲時取定值。

修訂歷程 第二十二輪 row 位置:L66(在 §14.7-BW 第二十一輪 row 之上)。

---

## 13. 結論

本 design research 提案 **§14.7-BX Phase B 入憲 Weekly Doctrine-Driven Recommit** 為對應用戶 Path B 之頻率治權升版:

- ✅ **doctrine 升 SSOT**(per timestamp 變動敏感性)
- ✅ **§6.8 / §6.8.1 / §14.7-BT frequency clause 追溯為 historical narrative**(條文不撤銷)
- ✅ **Weekly canonical spec**(每週收盤後 / `_weekly` suffix / `superseded` status enum)
- ✅ **Downstream churn 治權保留**(Mitigation sub-option M1/M2/M3 待 Phase D 選定)
- ✅ **與 §14.7-BV/BW Phase D-G 之 §9.2 v0.4 / model_trainer / feature_store 升版**一併規劃
- ✅ **5 治權新特性 + 5 證偽承諾**

**下一步**(待用戶授權 per-phase):
1. Phase B 入憲(~30 min;本 session 可推)
2. Phase C cron + schema enum 升版(~2-3h;cross-session)
3. Phase D model_trainer / feature_store weekly mode(~5-8h;Mitigation sub-option 待選)
4. Phase E audit + storage archive policy(~1h;cross-session)
5. Phase F first 4-week integration observation(~1 month real-time;post-deploy)
6. Phase G v6.4.0/v7.0.0 milestone tag(~30 min)

---

**設計研究作者**: Claude
**Status**: ✅ Phase A complete / pre-charter / non-destructive
**Cross-References**:
- 對應憲章 §14.7-BW(第二十一輪)/ §14.7-BV(第二十輪)/ §14.7-BT(第十八輪)/ §6.8 / §6.8.1 / §6.7.1 / §9.2 / §8 / §0.0-A/B/C/D
- 對應 user trigger:2026-05-26「每週跑 builder v0.10 + commit 新 snapshot 為 current committed。改憲章」(Path B explicit selection)
- 對應 evidence:`core_universe_20260526_core_universe_policy_v0_10_pure_doctrine`(v0.10 為 weekly 之 base policy)
- 對應 memory:`core_stock_selection_doctrine.md`(已升 v3;Phase B 後升 v4 反映 weekly recommit frequency)
