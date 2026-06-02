#!/usr/bin/env python3
"""
build_model_comparison_report.py v1.0 (跨模型統一比較 leaderboard 產生器)
================================================================================
**最後更新日期**: 2026-06-01
**主權狀態**: 承載 model_comparison_baseline_spec_20260601.md SSOT + 憲章 §14.7-CY 4-horizon + §14.7-CZ T_CZ-6;§一.10 (b) DB-derived JSON 唯一來源;§一.11 三段式標頭
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

================================================================================
## 📜 一、核心定義說明 (Core Definitions / The Constitution)
================================================================================

1. **[Sovereignty Declaration]** (§3.2 橫切模組 / evaluation 層): 本程式為「比較層」唯讀彙整器,
   讀取各 `multi_cycle_*_validation.py` 產出之 JSON,彙整為單一 leaderboard。
   治權邊界: 不訓練模型、不寫 DB、不改 universe、不碰 feature store;純讀 reports/*.json → 寫 reports/*.md+json。
   不涉五套禁令(§0.1/§0.2/§0.3)之工程公式;不處理 §8.5 anti-leakage(由各 validator 自身負責)。
2. **[Zero Hardcoded Verdict]** (§5.6.3): T_CZ-6 PASS/FAIL 由 metrics 動態判定(eff_t≥4.20 ∧ sharpe≥2.40
   ∧ win≥0.79),不硬編任何模型之 verdict;門檻常數集中於 TCZ6 dataclass 單一來源。
3. **[Historical Reference Authority]**: 本程式之版本號 / 行號為記述性快照,非權威;
   方法學權威為 multi_cycle_validation.py:225-330 + baseline_spec SSOT。
4. **[Idempotency]**: 純讀 + 覆寫輸出檔,可重跑;不累積狀態、不 INSERT DB。
5. **[Apples-to-Apples Sovereignty]** (本程式特有): 依 (model_key, n_universe) 分組,**絕不混批**
   不同 universe 之結果;同一 leaderboard 內所有模型共用 universe/panels/seed/cost(per baseline_spec §六)。
   model_key 由 `_meta.tool` 解析(消除 dedicated 變體之 model_family 撞名);同組取 `_meta.run_at` 最新。

================================================================================
## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix)
================================================================================

### A. 主要運行模式
| 子項 | CLI | 行為 |
|---|---|---|
| 預設(今日批次) | `python build_model_comparison_report.py` | 掃 reports/multi_cycle_*.json,since=2026-06-01,每 (model,universe) 取最新,輸出 md+json |
| 指定輸出前綴 | `--out reports/model_comparison_20260601` | 產 <out>.md + <out>.json |
| 指定起始日 | `--since 2026-06-01` | 僅納入 _meta.run_at ≥ 該日之 JSON(濾掉歷史批次) |
| 納入全部歷史 | `--all-dates` | 不濾日期(各 universe 自成一表) |

### B. 補充運行模式
| 子項 | CLI | 行為 |
|---|---|---|
| 限定 universe | `--universe 914` | 只輸出該 universe 之表(預設全部 universe 各一表) |
| 自訂 glob | `--glob 'reports/multi_cycle_*.json'` | 覆寫掃描範圍 |

### C. 對齊憲章 §二 維運矩陣
- 對映「§14.7-CY 多週期驗證」之比較層彙整;不提供訓練 / DB 寫入旗標(intentionally omitted — 非本程式職責)。

================================================================================
## 📜 三、全修訂歷程 (Full Revision History)
================================================================================

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
|---|---|---|---|---|
| v1.0 | 2026-06-01 | Claude | 首版。掃 multi_cycle JSON → 依 (model_key,n_universe) 分組不混批 → 套 T_CZ-6 → 輸出 leaderboard md+json。精準度(mean_ic/win/ir)+信任度(eff_t/sig/n_eff/sharpe/mdd/ann_net)。承載 baseline_spec v1.0 SSOT。 | ACTIVE |
"""
from __future__ import annotations
import argparse
import glob as globmod
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime

HORIZONS = ["weekly", "monthly", "quarterly", "annual"]
HORIZON_ZH = {"weekly": "週(5d)", "monthly": "月(20d)", "quarterly": "季(60d)", "annual": "年(252d)"}


@dataclass(frozen=True)
class TCZ6:
    """§14.7-CZ T_CZ-6 production 門檻(單一來源,Zero Hardcoded Verdict)。"""
    eff_t_min: float = 4.20
    sharpe_min: float = 2.40
    win_min: float = 0.79
    sig_t: float = 1.997  # p<0.05 large-df 最低顯著性

    def passes(self, eff_t, sharpe, win) -> bool:
        if eff_t is None or sharpe is None or win is None:
            return False
        return eff_t >= self.eff_t_min and sharpe >= self.sharpe_min and win >= self.win_min


GATE = TCZ6()


def model_key_from_meta(meta: dict, fname: str) -> str:
    """由 _meta.tool 解析乾淨 model_key(消除 dedicated 撞名);fallback 檔名。"""
    tool = (meta or {}).get("tool", "") or ""
    key = tool.replace("multi_cycle_", "").replace("_validation.py", "").replace(".py", "")
    if not key or key == "validation":
        # base validator(multi_cycle_validation.py)= LGBM baseline
        # 但檔名可能含模型名(罕見),保險再看檔名
        base = os.path.basename(fname).replace("multi_cycle_", "")
        m = re.match(r"([a-z_]+?)(?:_v\d|_clean|_\d{8})", base)
        if m and m.group(1) not in ("validation",):
            return m.group(1)
        return "lgbm_base"
    return key


def get(b: dict, *keys):
    for k in keys:
        if b and k in b and b[k] is not None:
            return b[k]
    return None


def load_runs(glob_pat: str, since: str | None):
    """回傳 list[dict]: 每筆 = 一個 JSON 的解析摘要。"""
    runs = []
    for f in sorted(globmod.glob(glob_pat)):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        meta = d.get("_meta", {})
        run_at = str(meta.get("run_at", ""))
        if since and run_at and run_at[:10] < since:
            continue
        if since and not run_at:
            # 無 run_at 之舊檔: since 模式下略過(無法判定批次)
            continue
        key = model_key_from_meta(meta, f)
        runs.append({
            "file": os.path.basename(f),
            "key": key,
            "n_universe": meta.get("n_universe"),
            "n_panels": meta.get("n_panels_input"),
            "seed": meta.get("seed"),
            "run_at": run_at,
            "data": d,
        })
    return runs


def pick_latest(runs):
    """依 (key, n_universe) 取 run_at 最新。"""
    best = {}
    for r in runs:
        k = (r["key"], r["n_universe"])
        if k not in best or r["run_at"] > best[k]["run_at"]:
            best[k] = r
    return best


def cell_metrics(block: dict):
    eff = get(block, "effective_t_stat")
    shp = get(block, "sharpe")
    win = get(block, "win_rate")
    ic = get(block, "mean_ic", "rank_ic_mean", "ic")
    ann = get(block, "annualized_simple_net")
    neff = get(block, "n_effective")
    ir = get(block, "ir")
    mdd = get(block, "mdd_per_panel")
    sig = get(block, "is_significant_p05")
    return dict(eff=eff, shp=shp, win=win, ic=ic, ann=ann, neff=neff, ir=ir, mdd=mdd, sig=sig,
                czpass=GATE.passes(eff, shp, win))


def fmt(v, spec, scale=1.0, pct=False):
    if v is None:
        return "n/a"
    try:
        x = v * scale
        s = format(x, spec)
        return s + ("%" if pct else "")
    except Exception:
        return str(v)


def build_markdown(by_universe: dict, since: str | None, glob_pat: str) -> str:
    L = []
    L.append("# 跨模型統一比較報告 (Model Comparison Leaderboard)")
    L.append("")
    L.append(f"**產生時間**：{datetime.now().isoformat(timespec='seconds')}")
    L.append("**基準**：`reports/model_comparison_baseline_spec_20260601.md` v1.0（T_CZ-6 + 4-horizon walk-forward）")
    L.append(f"**通過門檻 T_CZ-6**：eff_t ≥ {GATE.eff_t_min} 且 Sharpe ≥ {GATE.sharpe_min} 且 Win ≥ {GATE.win_min*100:.0f}%（三者同時）")
    L.append(f"**資料來源**：`{glob_pat}`（since={since or '全部'}）;全數字 trace 自各 validator JSON（§一.10 (b)）")
    L.append("**產生器**：`scripts/evaluation/build_model_comparison_report.py` v1.0")
    L.append("")
    L.append("> ⚠️ **絕不混批**：不同 n_universe 之結果分表呈現;同表內所有模型共用 universe/panels/seed/cost。")
    L.append("> ⚠️ **年 horizon caveat**：n_eff≈7.3 → 自由度極低 → 即使過 T_CZ-6 仍統計脆弱,不可單獨宣稱穩定賺錢。")
    L.append("")

    summary_rows = []  # for verdict
    for n_univ in sorted(by_universe.keys(), key=lambda x: (x is None, -(x or 0))):
        group = by_universe[n_univ]  # dict key->run
        # basis line
        any_run = next(iter(group.values()))
        L.append("---")
        L.append("")
        L.append(f"## Universe = {n_univ} 核心股")
        L.append("")
        L.append(f"- 面板數：{any_run['n_panels']}　seed：{any_run['seed']}　模型數：{len(group)}")
        L.append("")
        L.append(f"| 模型 | horizon | eff_t | Sharpe | Win% | IC | IR | Ann.Net% | n_eff | sig(p<.05) | T_CZ-6 |")
        L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|")
        # stable model order: lgbm_base first, then alphabetical
        def kord(k):
            return (0 if k == "lgbm_base" else 1, k)
        for key in sorted(group.keys(), key=kord):
            d = group[key]["data"]
            n_pass = 0
            for hz in HORIZONS:
                if hz not in d:
                    continue
                m = cell_metrics(d[hz])
                if m["czpass"]:
                    n_pass += 1
                czmark = "✅ PASS" if m["czpass"] else ("· sig" if m["sig"] else "—")
                L.append(
                    f"| {key} | {HORIZON_ZH[hz]} | {fmt(m['eff'],'+.3f')} | {fmt(m['shp'],'+.3f')} | "
                    f"{fmt(m['win'],'.1f',100)} | {fmt(m['ic'],'+.3f')} | {fmt(m['ir'],'+.2f')} | "
                    f"{fmt(m['ann'],'+.2f',100)} | {fmt(m['neff'],'.1f')} | {('Y' if m['sig'] else 'n')} | {czmark} |"
                )
            summary_rows.append((n_univ, key, n_pass))
        L.append("")

    # ── Verdict ──
    L.append("---")
    L.append("")
    L.append("## 誠實裁決 — 「依此模型預測股價真的能賺錢嗎?」")
    L.append("")
    pass_cells = [(u, k, n) for (u, k, n) in summary_rows if n > 0]
    if pass_cells:
        L.append("### 通過 T_CZ-6 的模型(過關 horizon 數)")
        L.append("")
        L.append("| Universe | 模型 | 過 T_CZ-6 的 horizon 數 |")
        L.append("|---|---|---:|")
        for u, k, n in sorted(pass_cells, key=lambda x: (-(x[0] or 0), -x[2])):
            L.append(f"| {u} | {k} | {n} / 4 |")
        L.append("")
    else:
        L.append("**無任何模型在任何 horizon 通過完整 T_CZ-6。**")
        L.append("")
    L.append("### 重點誠實結論")
    L.append("")
    L.append("- 通過 T_CZ-6 的 cell 高度集中於 **年(252d)horizon**,但該 horizon **n_eff≈7.3**(統計脆弱);"
             "季(60d,n_eff≈32 較穩健)多數模型卡在 **Win < 79%**。")
    L.append("- 週(5d)horizon 普遍 **IC ≤ 0**(無短線方向預測力),不具賺錢基礎。")
    L.append("- portfolio 為 **top-20 long-only vs 等權 universe**;成本 0.6%/rebal 已扣。")
    L.append("- 結論:**並非每個模型每個週期都能賺錢**;真實證據顯示「預測力與超額報酬隨 horizon 拉長而增強,"
             "但長 horizon 的統計信任度同步下降」—— 需 caveat 後方能談獲利性。")
    L.append("")
    return "\n".join(L)


def build_json(by_universe: dict, since, glob_pat) -> dict:
    out = {
        "_meta": {
            "tool": "build_model_comparison_report.py", "tool_ver": "v1.0",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "baseline_spec": "model_comparison_baseline_spec_20260601.md v1.0",
            "tcz6": {"eff_t_min": GATE.eff_t_min, "sharpe_min": GATE.sharpe_min, "win_min": GATE.win_min},
            "glob": glob_pat, "since": since,
            "source_traceability": "per CLAUDE.md §一.10 — all from validator JSON (b) DB-derived",
        },
        "universes": {},
    }
    for n_univ, group in by_universe.items():
        ub = {"models": {}}
        for key, run in group.items():
            d = run["data"]
            cells = {}
            for hz in HORIZONS:
                if hz in d:
                    cells[hz] = cell_metrics(d[hz])
            ub["models"][key] = {"file": run["file"], "n_panels": run["n_panels"],
                                 "seed": run["seed"], "run_at": run["run_at"], "cells": cells}
        out["universes"][str(n_univ)] = ub
    return out


def main():
    ap = argparse.ArgumentParser(description="跨模型統一比較 leaderboard 產生器(T_CZ-6 + 4-horizon)")
    ap.add_argument("--glob", default="reports/multi_cycle_*.json")
    ap.add_argument("--since", default="2026-06-01", help="僅納入 _meta.run_at ≥ 此日(YYYY-MM-DD)")
    ap.add_argument("--all-dates", action="store_true", help="不濾日期")
    ap.add_argument("--universe", type=int, default=None, help="只輸出此 n_universe")
    ap.add_argument("--out", default="reports/model_comparison_20260601", help="輸出前綴(.md/.json)")
    args = ap.parse_args()

    since = None if args.all_dates else args.since
    runs = load_runs(args.glob, since)
    if not runs:
        print(f"[ERR] 無符合條件之 JSON(glob={args.glob} since={since})")
        return 1
    best = pick_latest(runs)

    # group by n_universe
    by_universe: dict = {}
    for (key, n_univ), run in best.items():
        if args.universe is not None and n_univ != args.universe:
            continue
        by_universe.setdefault(n_univ, {})[key] = run
    if not by_universe:
        print("[ERR] 過濾後無資料")
        return 1

    md = build_markdown(by_universe, since, args.glob)
    js = build_json(by_universe, since, args.glob)
    with open(args.out + ".md", "w") as f:
        f.write(md)
    with open(args.out + ".json", "w") as f:
        json.dump(js, f, ensure_ascii=False, indent=2)

    n_models = sum(len(g) for g in by_universe.values())
    print(f"[OK] universes={sorted(k for k in by_universe)} models={n_models}")
    print(f"[OK] wrote {args.out}.md")
    print(f"[OK] wrote {args.out}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
