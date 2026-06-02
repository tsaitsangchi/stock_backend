# reports/ 殘留阻擋 .md 之 Load-Bearing 分析(2026-06-02)

**性質**：清理收尾分析報告(§二.4)。記錄 reports/ .md 清理(v6.26.12 / v6.26.13)後**剩餘 ~16 個阻擋檔**之引用源逐一追查結論 —— **0 個可乾淨清除**,皆為「有理由被引用」之 load-bearing 檔。
**目的**：durably 記錄此結論,**防止未來 session 重複分析**(re-litigation),並界定唯一的自然退場路徑。
**方法**：in-memory 引用圖固定點閉包(active corpus = 全 repo 非 archive/.claude 之 .md+.py;referrer = textual basename 出現)。全數字 source = code/grep 實證(§一.10 (a)(b))。

---

## 一、背景

reports/ .md 清理兩批已封存:
- **v6.26.12**(HEAD `ac8cc1a`):隔離 89 沒用 .md;**23 檔阻擋**(被未隔離活躍檔引用)。
- **v6.26.13**(HEAD `5ebab3d`):解除其中 7 檔(project_structure dump + Quantum_Finance v5×3 + 系統架構 v5.x×3);完整度報告加 §一.11 SUPERSEDED banner。

剩餘 **~16 阻擋檔**:用戶 directive「先處理其引用源」。本報告為該追查之結論。

## 二、追查結論:剩餘阻擋檔 0 個可乾淨清除

逐一追完每個阻擋檔的活躍引用源,全部落入兩類 **load-bearing** 牆,無一可在不竄改記錄 / 不弱化憲章審計軌跡下移除。

### 類別 A — `common_model_comparison_baseline_v1.md`:現行 torch 報告的事實比較基準

| 項 | 內容 |
|---|---|
| 引用源 | `{chronos,tft,itransformer,patchtst}_multi_cycle_validation_report_2026053x.md`(4 報告)+ `multi_cycle_{chronos,itransformer,patchtst}_validation.py`(3 validator docstring) |
| 為何不能移 | `baseline_v1` = **38 特徵 / v0.18-398 舊宇宙**;現行 SSOT `baseline_20260602` = **397 / 37 特徵重建後宇宙**(不同 universe)。torch 模型**實際就是對 v1 基準評估**的 |
| 若強行 repoint | 等於竄改「torch 模型用哪個基準」→ 違 **§一.10**(資料真實性) |
| 裁決 | **保留**,直到 torch 模型在新宇宙(397/37)重跑 |

### 類別 B — 憲章 T2 §14 audit-trail allowlist 刻意保留之證據鏈

grep 證實以下牆檔被 `系統架構大憲章_v6.1.0.md` 引用,屬其 `.gitignore` 之 **「# T2 重大實證報告（憲章 §14 / §8.8.x 引用之 audit trail）」** 白名單 —— 即憲章 §14.7-X 入憲之「證據基礎」刻意保留的審計軌跡:

| 牆檔(charter T2) | 擋住之阻擋檔 |
|---|---|
| `full_market_sync_20260523_1640.md` | api_schema/compliance/core_universe `_20260523` + `rebuild_execution_20260522_from_zero` |
| `session_handoff_20260526_{evening,final,v8_cross_machine}.md` | session_handoff_20260526 系列 + core_universe_audit_0525 |
| `k_wave_cumulative_state_post_session_20260526.md` | first_principles / pareto cumulative_state |
| `universe_completeness_governance_design_research_20260526.md`、`universe_completeness_phase_e_preview_probe_20260526.md` | core_universe_audit_0526、feature_store_v08_audit、session_handoff_v7 |
| `from_zero_to_model_build_guide_20260528.md` | cross_machine_handoff_20260528_v6.19.0 |
| `tree_based_from_zero_build_runbook_20260531.md`(CLAUDE.md + memory 亦引用) | cross_machine_handoff_20260601_v6_26_1(本即保留之**最新**封存點,moot) |
| `portfolio_sizer_v03_design_research_20260526.md`、`v6_2_0_honest_amendment_20260526.md` | portfolio_sizer_v02_audit、trinity_cross_pillar_audit |
| `系統核心完整度評估報告.md`(CLAUDE.md §六) | compliance_audit_20260519(已於 v6.26.13 加 SUPERSEDED banner 涵蓋) |

**裁決**:**保留**。移除將弱化憲章 §14 入憲之證據審計軌跡;且其證據快照(audit snapshots)由產生它們的 sync/rebuild 日誌記述,應與日誌同進退。

## 三、結論與建議

1. **封閉安全集已抽乾**:cleanly-removable 之 .md 已於 v6.26.12+v6.26.13 全數隔離(96 檔)。剩餘 ~16 為 load-bearing,**非「沒用」而是「有理由被引用」**。
2. **建議停在此處**。強行清除唯二路徑皆不可取:(a) repoint torch 基準 → 竄改記錄(§一.10);(b) 編輯憲章 T2 allowlist 去引用 → 弱化 §14 審計軌跡。
3. **唯一自然退場路徑**:**torch 模型(chronos/tft/itransformer/patchtst)在新宇宙 397/37 重跑後**(pipeline 已排隊),其 4 份報告改引用 `baseline_20260602`,屆時 `baseline_v1` + 4 報告**一起退役**,無需竄改。類別 B 之審計軌跡則隨憲章生命週期自然管理。

## 四、累計清理狀態(本封存點)

- 隔離區累計:**120 .py + 96 .md**(`archive/_pending_removal_20260602/`,git mv 可逆)。
- `scripts/`:~207 → 101;`reports/` .md:265 → 169。
- 0 活躍引用斷裂;101/101 py_compile PASS。

**證據基礎**：本報告全數字出自 code/grep 實證(in-memory 引用圖閉包 + memory/charter grep);baseline_v1 vs baseline_20260602 universe 差異經 grep 對照(38-feat/398 vs 397/37);charter T2 allowlist 歸屬經 grep `系統架構大憲章_v6.1.0.md` 確認。無 AI 幻像數據(§一.10)。
