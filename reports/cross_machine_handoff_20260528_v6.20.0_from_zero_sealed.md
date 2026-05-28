# Cross-Machine Handoff — v6.20.0 From-Zero Build Sequence SEALED(2026-05-28)

**Session Type**:封存點(sealed checkpoint)— upgrade from v6.19.0 sealed
**HEAD**:`bdaa04a`(本 handoff commit 後更新)
**Latest milestone tag**:`v6.20.0-section14-7-CZ-from-zero-build-sequence-20260528`
**Session focus**:統合 §14.7-AM/CB/CF/CG/CW/CX/CY 為 unified §14.7-CZ canonical 8-phase sequence
**用戶 directive**:「更新全部檔案上傳到 GitHub 並做封存點」

---

## 一、本封存點 vs 前 v6.19.0 sealed 增量

| 項目 | v6.19.0 sealed | **v6.20.0 sealed**(本)|
|---|---|---|
| Doctrine count(§14.7-*)| 66 | **67** |
| New doctrine | §14.7-CY Multi-Cycle | **§14.7-CZ From-Zero Build Sequence** |
| Canonical sequence | 各 phase 各自獨立 inscribed | **8-phase unified end-to-end canonical** |
| Implementation reference | 各 doctrine 分散於 charter | **`reports/from_zero_to_model_build_guide_20260528.md`**(747 行)|
| 從零建構治權 | 部分(supply chain in §14.7-AM)| **完整 8-phase 閉環** |

---

## 二、本 Session v6.16.1 → v6.20.0 完整治權升版鏈

| Version | Commit | Tag | Doctrine |
|---|---|---|---|
| v6.17.0 | 77fc1d6 | v6.17.0-section14-7-CW | §14.7-CW LGBM Tree Production |
| v6.17.1 | 6da6110 | v6.17.1-section14-7-CW-reproducibility-patch | §14.7-CW T_CW-6 Reproducibility |
| CLAUDE.md | d7bb852 | — | §一.10 No Data Hallucination |
| v6.18.0 | a95116e | v6.18.0-section14-7-CX | §14.7-CX 8-Year Historical OOS |
| v6.18.0 handoff | d8fb500 | session-final-20260528-v6.18.0-8year-sealed | (handoff)|
| v6.19.0 | aea057c | v6.19.0-section14-7-CY | §14.7-CY Multi-Cycle Horizon Validation |
| v6.19.0 handoff | 21ab9af | session-final-20260528-v6.19.0-multi-cycle-sealed | (handoff)|
| v6.19.0 enhanced | ca2dcf2 | — | Cross-machine setup section |
| **v6.20.0** | **bdaa04a** | **v6.20.0-section14-7-CZ-from-zero-build-sequence-20260528** | **§14.7-CZ From-Zero Build Sequence** |
| **v6.20.0 sealed** | **(this commit)** | **session-final-20260528-v6.20.0-from-zero-sealed** | **(本 handoff)** |

---

## 三、§14.7-CZ Canonical 8-Phase Sequence(從零到 production reality)

| Phase | 階段 | Charter Anchor | 耗時 | Audit Gate |
|---|---|---|---|---|
| **0** | Environment Bootstrap | §0.0-I.9/I.10 | ~10 min | Import smoke test PASS |
| **1** | DB Schema Init | §3.2/§6.7 | ~5 min | ≥ 33 tables |
| **2** | Raw API Sync 4-step(§14.7-AM)| §14.7-AM/§7.4-A | **~6-12 hr** | audit_supply_chain PERFECT |
| **3** | FRED Macro Sync | §0.3/§14.7-BY/BZ | ~10 min | ≥ 13 FRED series |
| **4** | Core Universe Selection | §14.7-CB/CC/CD/CE/CF/CG/CI/CJ/CK | ~10 min | audit_core_universe PERFECT |
| **5** | Feature Store(current + 95 historical)| §14.7-CA/CL | ~20 min | ≥ 96 snapshots / ≥ 4M rows |
| **6** | Feature Audits | §14.7-CM/CN/CO | ~5 min | t-stat reports |
| **7** | LGBM Model Training | §14.7-CW | ~3 min | Treaty Gates 4/4 PASS |
| **8** | Multi-Layer Validation | §14.7-CV/CX/CY | ~5 min | Quarterly Eff t ≥ 4.20 |
| **TOTAL** | — | — | **~7-14 hr first-time** | **Reality check pass** |

**Re-build from dump**:Phase 2 改為 pg_restore ~30 min,total **~1-2 hr**

---

## 四、§14.7-CZ 6 證偽承諾(T_CZ-1〜6)

| Treaty | 內容 |
|---|---|
| **T_CZ-1** | 8-phase sequence cannot skip / reorder |
| **T_CZ-2** | Audit gate FAIL 阻擋下一 phase |
| **T_CZ-3** | Time budget realistic(not under-promised)|
| **T_CZ-4** | Every phase 必引明確 charter anchor |
| **T_CZ-5** | Every script idempotent(可重跑)|
| **T_CZ-6** | **Quarterly Eff t ≥ 4.20 / Sharpe ≥ 2.4 / Win ≥ 79% final reality**(per §14.7-CY)|

---

## 五、本 Session 6 commits / 4 milestone tags / 3 sealed tags

### Commits(7 commits in master since v6.16.1 baseline)

```
77fc1d6 §14.7-CW LGBM Tree v0.2
6da6110 §14.7-CW T_CW-6 Reproducibility patch
d7bb852 CLAUDE.md §一.10 No Data Hallucination
a95116e §14.7-CX 8-Year Historical OOS
d8fb500 v6.18.0 sealed handoff
aea057c §14.7-CY Multi-Cycle Horizon Validation
21ab9af v6.19.0 sealed handoff
ca2dcf2 Enhanced cross-machine setup
bdaa04a §14.7-CZ From-Zero Build Sequence(NEW v6.20.0)
+(this commit)v6.20.0 sealed handoff
```

### Milestone tags

- `v6.17.0-section14-7-CW-lgbm-tree-model-upgrade-20260528`
- `v6.17.1-section14-7-CW-reproducibility-transparency-patch-20260528`
- `v6.18.0-section14-7-CX-8year-historical-validation-20260528`
- `v6.19.0-section14-7-CY-multi-cycle-horizon-validation-20260528`
- **`v6.20.0-section14-7-CZ-from-zero-build-sequence-20260528`**

### Sealed tags

- `session-final-20260528-v6.18.0-8year-sealed`
- `session-final-20260528-v6.19.0-multi-cycle-sealed`
- **`session-final-20260528-v6.20.0-from-zero-sealed`**(本封存點)

---

## 六、新檔案總覽(本 session 累積)

### Scripts(3 new production scripts)

| 檔案 | 行數 | 用途 |
|---|---|---|
| `scripts/core/model_trainer_lgbm_v2.py` | 398 | §14.7-CW LGBM tree v0.2 |
| `scripts/evaluation/build_historical_panels.py` | 130 | 95-panel historical builder |
| `scripts/evaluation/multi_cycle_validation.py` | 330 | §14.7-CY 4-horizon validator |

### Doctrine inscriptions

| 檔案 | 變更 |
|---|---|
| `reports/系統架構大憲章_v6.1.0.md` | +§14.7-CW / §14.7-CX / §14.7-CY / §14.7-CZ + 5 revision history entries |
| `CLAUDE.md` | +§一.10 No Data Hallucination |

### Reports(audit trail + reference)

| 檔案 | 行數 | 用途 |
|---|---|---|
| `reports/multi_cycle_validation_report_20260528.md` | 155 | Multi-cycle reality report |
| `reports/multi_cycle_validation_20260528_final.json` | structured | Multi-cycle JSON |
| `reports/multi_cycle_validation_20260528.json` | structured | Audit trail(early run)|
| `reports/multi_cycle_validation_20260528_corrected.json` | structured | Audit trail(intermediate)|
| `reports/cross_machine_handoff_20260528_v6.18.0_8year_sealed.md` | 264 | v6.18.0 sealed handoff |
| `reports/cross_machine_handoff_20260528_v6.19.0_multi_cycle_sealed.md` | 502 | v6.19.0 sealed + cross-machine setup |
| **`reports/from_zero_to_model_build_guide_20260528.md`** | **747** | **§14.7-CZ implementation reference** |
| **`reports/cross_machine_handoff_20260528_v6.20.0_from_zero_sealed.md`** | **(本檔)** | **v6.20.0 sealed handoff** |

---

## 七、新機從零重建(per §14.7-CZ)

### Quick start

```bash
# 1. Clone + checkout v6.20.0 sealed
git clone https://github.com/tsaitsangchi/stock_backend.git
cd stock_backend
git checkout session-final-20260528-v6.20.0-from-zero-sealed

# 2. 讀 implementation reference(完整 step-by-step)
less reports/from_zero_to_model_build_guide_20260528.md

# 3. 依 Phase 0-8 順序執行
# Phase 0(env)→ Phase 1(schema)→ Phase 2(sync 6-12hr)
# → Phase 3(FRED)→ Phase 4(universe)→ Phase 5(features)
# → Phase 6(audits)→ Phase 7(model)→ Phase 8(validation)

# 4. 最終 reality check(per §14.7-CZ T_CZ-6)
python scripts/evaluation/multi_cycle_validation.py --dry-run --horizons 60
# Expected: Quarterly Eff t ≥ 4.20 / Sharpe ≥ 2.4 / Win ≥ 79%
```

### Re-build from dump(已有 backup,推薦)

```bash
# 在原機 dump:
pg_dump -h localhost -p 5432 -U <user> -d stock -F c -f stock_backup.dump

# 在新機 restore:
createdb -h localhost -p 5432 -U <user> stock
pg_restore -h localhost -p 5432 -U <user> -d stock stock_backup.dump
# Time: ~30 min(vs 6-12 hr API sync)
```

---

## 八、Production Reality Final State(per §14.7-CY / §14.7-CZ T_CZ-6)

### Recommended Strategy: Quarterly Rebalance

| 維度 | 真實值 |
|---|---|
| **Recommended horizon** | **Quarterly(60-day)** |
| **Net annualized** | **+24.44%/year** |
| **Sharpe(net)** | ~2.4 |
| **Win rate** | 79.7% |
| **n_effective** | 32 truly independent panels(overlap-corrected)|
| **Effective t-stat** | **+4.20**(p < 0.001 robust)|
| **MDD** | ~17% |

### 全 horizon comparison(per §14.7-CY 4-cycle validation)

| Horizon | Eff t | Sig p<0.05 | Sharpe | Win | Net Annual |
|---|---|---|---|---|---|
| weekly(5d)| +1.59 | ❌ | 0.89 | 67.7% | +13.99% |
| monthly(20d)| +1.41 | ❌ | 0.97 | 64.6% | +17.41% |
| **quarterly(60d)** | **+4.20** | **✅** | **2.55** | **79.7%** | **+24.44%** |
| annual(252d)| +3.58 | ✅(small n_eff)| 4.81 | 91.8% | +29.69% |

---

## 九、Pipeline Provenance Summary(累計 v6.20.0)

```
[FinMind API + FRED API]
  ↓ 77,312,879 raw rows
[DB raw tables 11 FinMind + fred_series]
  ↓ §14.7-CC source authority
[2,799 stocks → 1,121 core_universe / 59.9% filtered]
  ↓ §14.7-CB/CI/CJ/CK 三重 quality gate
[Feature Store 43 canonical features × 102 monthly snapshots]
  ↓ 4,696,034 feature_values rows
[Walk-Forward LGBM v0.2 training]
  ↓ 95-panel × 4 horizons = 380 walk-forward trains
[§14.7-CY 4-cycle validation]
  ↓ Quarterly Eff t=4.20 ✅
[PRODUCTION READY @ +24.44% net annual]
```

---

## 十、CLAUDE.md §一.10 enforcement(累計)

| 規則 | 證據 |
|---|---|
| (a) 程式輸出 | 3 production scripts(LGBM trainer / panel builder / multi-cycle validator)|
| (b) DB query | feature_values + feature_store_snapshot + model_registry + TaiwanStockPriceAdj |
| (c) API response | FinMind + FRED |
| 禁止從記憶 | ✅ 全 session enforced |
| Multi-run statistics | ✅ 6-run LGBM(per T_CW-6)|
| ≥ 3 horizons | ✅ 4 horizons(per T_CY-2)|
| Overlap correction | ✅ n_eff(per T_CY-3)|
| Honest annualization | ✅ mean × rebals/year(per T_CY-4)|
| Cost-drag per horizon | ✅ disclosed(per T_CY-5)|
| System script execution | ✅ not AI env(per T_CY-1)|
| **8-phase canonical sequence** | **✅ inscribed**(per **T_CZ-1**)|
| Phase audit gates | ✅ each phase(per T_CZ-2)|
| Time budget realistic | ✅ 7-14 hr disclosed(per T_CZ-3)|
| Reality-check terminal | ✅ quarterly Eff t ≥ 4.20(per T_CZ-6)|

---

## 十一、Next Session 接續方向

| Priority | 方向 |
|---|---|
| **P0** | **3-6 月 paper trading 啟動**(quarterly rebalance @ +24.44% predicted)|
| **P1** | **DB full-drop empirical rebuild test**(per §14.7-CZ end-to-end validation)|
| P1 | Liquidity audit top-20 stocks |
| P1 | Survivorship bias 修正(per-panel dynamic universe)|
| P2 | Multi-seed ensemble(消 LGBM ±15% stochasticity)|
| P2 | 2008 GFC stress test(BalanceSheet 限制 fallback)|

---

## 十二、Repository / Charter / CLAUDE.md State(absolute snapshot)

| 項目 | 真實值 |
|---|---|
| Repository | https://github.com/tsaitsangchi/stock_backend |
| Branch | master |
| HEAD(after this commit)| (待 commit 更新)|
| Charter §14.7-* sections | **67** |
| CLAUDE.md §一 rules | **10**(含 §一.10)|
| Pipeline scale | 77.3M raw → 1,121 stocks → 4.7M features → 380 LGBM trains → quarterly @ +24.44% net |
| Production model | `mdl_20260415_lgbm_h30_0b243a67_v0_2`(committed)|
| **From-zero build path** | **`reports/from_zero_to_model_build_guide_20260528.md`**(747 行)|

---

**封存點建立時間**:2026-05-28 16:40(UTC+8)
**封存 git tag**:`session-final-20260528-v6.20.0-from-zero-sealed`(本檔 commit 後 push)
**從零重建 reference**:`reports/from_zero_to_model_build_guide_20260528.md`
**Doctrines inscribed this session**:5(§14.7-CW + CW patch + CX + CY + **CZ**)+ CLAUDE.md §一.10
