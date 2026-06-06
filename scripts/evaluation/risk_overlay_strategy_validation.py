"""
risk_overlay_strategy_validation.py v0.1 (Step 1 — 把 vol-target×趨勢 overlay 套到「真實 top-20 策略」報酬流,非市場 proxy)
================================================================================
**最後更新日期**: 2026-06-06
**主權狀態**: RISK-OVERLAY-ON-REAL-STRATEGY VALIDATION(§14.7-DH Step 1 gate)+ §3.2 evaluation 橫切 + §14.7-DC SOURCE-PURE + §8.5 ANTI-LEAKAGE(walk-forward 選股 [0..i-1] 訓練→預測 i;overlay 曝險 ≤as_of 設定)+ §一.10 SOURCE-TRACEABLE + §一.11 三段式 + §14.7-DE Canonical Panel(get_canonical_panel_dates,非硬編)+ ⚠️ HONEST CAVEAT
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:先前 overlay 是套在「等權全市場指數」(proxy)上;這支把同一個 vol-target×趨勢 overlay **套到我們真正的策略(最佳模型的 top-20 多頭組合)報酬流**上,回答「**對我們的策略,overlay 到底有沒有降低回撤?**」——這是 §14.7-DH「Step 2 整合進 portfolio_sizer」之前必過的驗證 gate。

**它怎麼做(步驟)**:
1. 用 canonical panels(`get_canonical_panel_dates`,§一.16 非硬編)+ 397 核心宇宙,跑月度 walk-forward(複用 base LightGBM,[0..i-1] 訓練→預測 i,top-20 等權)→ 得到**真實策略每月報酬流** `top20_ret(as_of)`。
2. 用等權市場序列(複用 market_regime_timing / Phase1)算每個 as_of 當下(≤as_of)的 overlay 曝險(combined = vol-target×趨勢)。
3. `overlay'd_strategy(as_of) = 曝險(as_of) × top20_ret(as_of)`(其餘現金 0)。
4. 比 **raw 策略 vs overlay'd 策略**的 MaxDD / Calmar / Sharpe / CAGR(月度複利 equity 曲線)+ 逐空頭(2020/2022,落在 panel 範圍內者)。
5. **裁決**:overlay 是否降低**真實策略**的 MaxDD(§14.7-DH Step 1 gate)。

**輸入 / 輸出**:輸入 = `feature_values` + `TaiwanStockPriceAdj`(§一.10 (b) DB)。輸出 = JSON(raw vs overlay'd 策略 + 裁決 + per-episode)+ stdout。

**它不做的事**:不改 production(portfolio_sizer 整合是 Step 2、須另授權 + treaty);不挑新股(複用既有選股邏輯);不寫 DB。

**為什麼需要它**:市場指數 proxy ≠ top-20 long-only 策略(回撤輪廓不同);Step 2 動 production 前,必須先在**真實策略**上證明 overlay 有效。

## 📜 一、核心定義說明 (Core Definitions / The Constitution)

1. **[Sovereignty Declaration]** (v0.1, §3.2 橫切): overlay-on-strategy 研究驗證;read-only;不寫 model_registry;不改 production。唯一職責:真實策略報酬流 × overlay 曝險 → 經濟價值對比 + JSON。
2. **[Source Purity]** (§14.7-DC / §一.10): 選股特徵 = feature_values(`is_null_imputed IS NOT TRUE`);overlay 訊號 = price→math transform;market 序列複用 build_market_series;無 impute/無 fabricate。
3. **[Anti-Leakage]** (§8.5): 選股 walk-forward [0..i-1] 訓練→預測 i(複用 base validator);overlay 曝險在 as_of 僅用 ≤as_of 之 market vol/趨勢;無未來洩漏。
4. **[Canonical Panel]** (§14.7-DE / §一.16): panels 來自 `get_canonical_panel_dates`(單一引用源,資料驅動,**非硬編** — 修正 base validator `get_panel_dates` 之硬編)。
5. **[Economic-Value Comparability]** (§14.7-DG/DH): MaxDD/Calmar/Sharpe 月度複利 equity;raw vs overlay'd 同法對比 → 公允。
6. **[Zero Hardcoded Verdict]** (§5.6.3): overlay_reduces_strategy_maxdd 動態計算。
7. **[⚠️ HONEST CAVEAT]** (v0.1): top20_ret 為 H=20 月度取樣(輕微重疊);策略用 base LightGBM 為代表(非逐一最佳模型);panel 範圍(~2018-2026)只涵蓋 2020/2022 空頭(2000/2008 在 panel 前);overlay 仍擋不住單日閃崩第一天;cash=0%。

## 📊 二、全量維運指令總矩陣

| 子項 | CLI | 治權契約 |
|---|---|---|
| Smoke | `... risk_overlay_strategy_validation.py --smoke --output reports/risk_overlay_strategy_smoke.json` | 第 3 條;限 panel 數 |
| 完整 + 輸出 | `... risk_overlay_strategy_validation.py --output reports/risk_overlay_strategy_<ts>.json` | 第 5 條 |
| `--commit`/`--dry-run` | 接受;僅 `--output` 寫 JSON(不寫 DB)| 第 1 條 |

## 📜 三、全修訂歷程

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
|---|---|---|---|---|
| v0.1 | 2026-06-06 | Claude | **首版(§14.7-DH Step 1)**:把 Phase1 combined overlay 套到真實 top-20 策略(base LightGBM walk-forward,canonical panels,§一.16 非硬編)報酬流;比 raw vs overlay'd 策略 MaxDD/Calmar/Sharpe + per-episode + 動態裁決。複用 mc(load_features/load_forward_returns/LGB_PARAMS/SPEC_43)+ ro(overlay)+ mrt(market 序列/經濟口徑)。§一.10 全 DB;§一.11 三段式;⚠️ caveat 入 _meta。 | **ACTIVE** |
"""
import sys, os, argparse, math, json, logging, time
from datetime import date, datetime
from pathlib import Path

_base_dir = Path(__file__).resolve().parent.parent
for _p in (str(_base_dir), str(_base_dir / "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
PROJECT_ROOT = _base_dir.parent

import numpy as np
import pandas as pd
import lightgbm as lgb

import multi_cycle_validation as mc
from multi_cycle_validation import load_features, load_forward_returns, winsorize, spearman_ic, LGB_PARAMS, N_ESTIMATORS, SPEC_43
import risk_overlay_timing_validation as ro
import market_regime_timing_validation as mrt
from market_regime_timing_validation import build_market_series, BEAR_EPISODES, TRADING_DAYS_YEAR
from core.db_utils import get_db_conn, get_canonical_panel_dates  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
CONSTITUTION_VER = "v6.1.0"; TOOL_VER = "v0.1"
STRAT_HORIZON = 20   # monthly strategy (top-20 over 20-day forward), monthly rebalance


def strategy_returns(cur, panels, universe, horizon, seed):
    """Walk-forward base-LightGBM top-20 strategy → per-panel (as_of, top20_ret, univ_ret). (mc logic, anti-leakage.)"""
    LGB_PARAMS["seed"] = seed
    panel_data = {}
    for fs_id, as_of in panels:
        X, sids = load_features(cur, fs_id, universe)
        if not X:
            continue
        returns, _ = load_forward_returns(cur, as_of, horizon)
        if not returns:
            continue
        XX, yy = [], []
        for i, sid in enumerate(sids):
            if sid in returns:
                XX.append(X[i]); yy.append(returns[sid])
        if XX:
            panel_data[as_of] = (XX, yy)
    keys = sorted(panel_data.keys())
    strat = []
    for i in range(1, len(keys)):
        X_test, y_test = panel_data[keys[i]]
        tr_X, tr_y = [], []
        for j in range(i):
            xj, yj = panel_data[keys[j]]; tr_X.extend(xj); tr_y.extend(yj)
        if len(tr_X) < 100:
            continue
        X_tr = np.array(tr_X); y_tr = winsorize(np.array(tr_y), 0.01, 0.99)
        model = lgb.train(LGB_PARAMS, lgb.Dataset(X_tr, label=y_tr, feature_name=SPEC_43), num_boost_round=N_ESTIMATORS)
        pred = model.predict(np.array(X_test))
        n_top = min(20, len(pred))
        top_idx = np.argsort(pred)[-n_top:]
        top20 = float(np.mean([y_test[k] for k in top_idx])); univ = float(np.mean(y_test))
        strat.append((keys[i], top20, univ))
    return strat


def overlay_exposure_at(as_of, mdates, realized_vol, above_trend, valid_pos):
    """combined overlay exposure using market vol/trend at the last market date <= as_of (anti-leakage)."""
    ts = np.array([pd.Timestamp(d) for d in mdates])
    p = int(np.searchsorted(ts, pd.Timestamp(as_of), side="right") - 1)
    if p < 0 or p >= len(valid_pos) or not valid_pos[p]:
        return 1.0   # warmup / no signal → stay fully invested (conservative: overlay only de-risks when it has signal)
    return ro.decision_exposure("combined", realized_vol[p], above_trend[p], ro.DEFAULT_TARGET_VOL)


def eq_stats(rets):
    """月度 simple-return 流 → 複利 equity 曲線之 MaxDD/Calmar/Sharpe/CAGR."""
    r = np.asarray(rets, dtype=float)
    if len(r) == 0:
        return {"n": 0, "maxdd": None, "calmar": None, "sharpe": None, "cagr": None, "cum_return": None}
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    mdd = float(np.max((peak - eq) / peak)) if len(eq) else None
    years = len(r) / 12.0
    cagr = float(eq[-1] ** (1.0 / years) - 1.0) if years > 0 and eq[-1] > 0 else None
    sharpe = float(np.mean(r) / np.std(r, ddof=1) * math.sqrt(12)) if len(r) > 1 and np.std(r, ddof=1) > 0 else None
    calmar = (cagr / mdd) if (cagr is not None and mdd and mdd > 0) else None
    return {"n": len(r), "maxdd": mdd, "calmar": calmar, "sharpe": sharpe, "cagr": cagr, "cum_return": float(eq[-1] - 1.0)}


def run(args):
    conn = get_db_conn(); cur = conn.cursor(); t0 = time.monotonic()
    cur.execute("""SELECT m.stock_id FROM core_universe_membership m
                   JOIN core_universe_snapshot s ON s.snapshot_id=m.snapshot_id
                   WHERE s.status='committed' AND m.core_tier='core_universe'
                   AND s.snapshot_id=(SELECT snapshot_id FROM core_universe_snapshot
                                      WHERE status='committed' ORDER BY created_at DESC LIMIT 1)""")
    universe = list({r[0] for r in cur.fetchall()})
    panels = get_canonical_panel_dates("feature_set_v0.5")
    if args.smoke:
        panels = panels[-24:]
    logger.info(f"Universe: {len(universe)} core; panels: {len(panels)} (canonical, {panels[0][1]}→{panels[-1][1]})")
    logger.info(f"Strategy walk-forward (base LightGBM top-20, H={STRAT_HORIZON}, seed {args.seed})...")
    strat = strategy_returns(cur, panels, universe, STRAT_HORIZON, args.seed)
    logger.info(f"  Strategy panels (OOS): {len(strat)} (elapsed {time.monotonic()-t0:.0f}s)")
    if len(strat) < 12:
        logger.warning("Too few strategy panels."); conn.close(); return {}

    logger.info("Building market series for overlay exposure (reuse)...")
    mdates, mret, midx = build_market_series(cur); conn.close()
    realized_vol, above_trend, valid_pos = ro.compute_overlay_signals(mdates, mret, midx, ro.VOL_LOOKBACK, ro.TREND_MA)

    raw, overl, exps = [], [], []
    for as_of, top20, univ in strat:
        e = overlay_exposure_at(as_of, mdates, realized_vol, above_trend, valid_pos)
        raw.append(top20); overl.append(e * top20); exps.append(e)

    raw_s = eq_stats(raw); ov_s = eq_stats(overl)
    ov_s["avg_exposure"] = float(np.mean(exps))
    verdict = {
        "overlay_reduces_strategy_maxdd": bool(ov_s["maxdd"] is not None and raw_s["maxdd"] is not None and ov_s["maxdd"] < raw_s["maxdd"] - 1e-6),
        "overlay_improves_calmar": bool(ov_s["calmar"] is not None and raw_s["calmar"] is not None and ov_s["calmar"] > raw_s["calmar"] + 1e-6),
        "delta_maxdd": (None if ov_s["maxdd"] is None or raw_s["maxdd"] is None else ov_s["maxdd"] - raw_s["maxdd"]),
        "delta_calmar": (None if ov_s["calmar"] is None or raw_s["calmar"] is None else ov_s["calmar"] - raw_s["calmar"]),
    }
    # per-episode (panels whose as_of falls in the bear window)
    by_ep = {}
    aods = [s[0] for s in strat]
    for name, e_start, e_end in BEAR_EPISODES:
        idx = [k for k, ao in enumerate(aods) if e_start <= ao <= e_end]
        if len(idx) < 2:
            by_ep[name] = {"in_range": False}; continue
        rr = [raw[k] for k in idx]; oo = [overl[k] for k in idx]
        by_ep[name] = {"in_range": True, "n_panels": len(idx),
                       "raw_maxdd": eq_stats(rr)["maxdd"], "overlay_maxdd": eq_stats(oo)["maxdd"],
                       "raw_cum": eq_stats(rr)["cum_return"], "overlay_cum": eq_stats(oo)["cum_return"],
                       "avg_exposure": float(np.mean([exps[k] for k in idx])),
                       "reduced_drawdown": bool(eq_stats(oo)["maxdd"] is not None and eq_stats(rr)["maxdd"] is not None and eq_stats(oo)["maxdd"] < eq_stats(rr)["maxdd"])}
    return {"raw_strategy": raw_s, "overlay_strategy": ov_s, "verdict": verdict, "by_episode": by_ep,
            "n_strategy_panels": len(strat), "strategy_span": [str(strat[0][0]), str(strat[-1][0])]}


def main():
    ap = argparse.ArgumentParser(description=f"Risk Overlay on REAL strategy (Step 1) {TOOL_VER}")
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--seed", type=int, default=5422)
    ap.add_argument("--smoke", action="store_true", help="last 24 panels only")
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--commit", action="store_true")
    args = ap.parse_args()
    logger.info("=" * 110)
    logger.info(f"Risk Overlay on REAL top-20 strategy (§14.7-DH Step 1 gate) {TOOL_VER}")
    logger.info("  ⚠️  market-index proxy ≠ top-20 strategy; this tests the overlay on the ACTUAL strategy return stream. cash=0%")
    logger.info("=" * 110)
    tg = time.monotonic()
    res = run(args)
    if res:
        rs, os_, v = res["raw_strategy"], res["overlay_strategy"], res["verdict"]
        logger.info(f"  strategy span {res['strategy_span']} ({res['n_strategy_panels']} monthly panels)")
        logger.info(f"  RAW strategy    : MaxDD={mrt._fmt(rs['maxdd'],'{:.3f}')} Calmar={mrt._fmt(rs['calmar'],'{:.2f}')} Sharpe={mrt._fmt(rs['sharpe'],'{:.2f}')} CAGR={mrt._fmt(rs['cagr'],'{:.3f}')}")
        logger.info(f"  OVERLAY strategy: MaxDD={mrt._fmt(os_['maxdd'],'{:.3f}')} Calmar={mrt._fmt(os_['calmar'],'{:.2f}')} Sharpe={mrt._fmt(os_['sharpe'],'{:.2f}')} CAGR={mrt._fmt(os_['cagr'],'{:.3f}')} avgExp={mrt._fmt(os_['avg_exposure'],'{:.2f}')}")
        logger.info(f"  ⮕ VERDICT: overlay reduces strategy MaxDD? {'YES' if v['overlay_reduces_strategy_maxdd'] else 'NO'} "
                    f"(ΔMaxDD={mrt._fmt(v['delta_maxdd'],'{:+.3f}')} ΔCalmar={mrt._fmt(v['delta_calmar'],'{:+.2f}')})")
        for name, e in res["by_episode"].items():
            if e.get("in_range"):
                logger.info(f"   {name:18}: raw_MaxDD={mrt._fmt(e['raw_maxdd'],'{:.3f}')} overlay_MaxDD={mrt._fmt(e['overlay_maxdd'],'{:.3f}')} "
                            f"avgExp={mrt._fmt(e['avg_exposure'],'{:.2f}')} {'(reduced)' if e['reduced_drawdown'] else '(did NOT)'}")
    logger.info(f"\n  Total elapsed: {time.monotonic()-tg:.1f}s")
    if args.output and res:
        out = {"result": res, "_meta": {
            "tool": "risk_overlay_strategy_validation.py", "tool_ver": TOOL_VER, "run_at": datetime.now().isoformat(),
            "constitution_ver": CONSTITUTION_VER, "strategy_model": "base LightGBM top-20 (multi_cycle_validation reuse)",
            "strategy_horizon_days": STRAT_HORIZON, "overlay": f"combined vol-target({ro.DEFAULT_TARGET_VOL})×trend(MA{ro.TREND_MA})",
            "seed": args.seed, "smoke": args.smoke,
            "anti_leakage": "stock pick walk-forward [0..i-1]→i; overlay exposure uses market vol/trend at last market date <= as_of.",
            "source_traceability": "per §一.10 — feature_values (is_null_imputed IS NOT TRUE) + TaiwanStockPriceAdj; canonical panels via get_canonical_panel_dates (非硬編).",
            "HONEST_CAVEAT": ("Step-1 gate for §14.7-DH Step-2 (portfolio_sizer integration). Strategy = representative base "
                              "LightGBM top-20 (not per-model best). Panel range ~2018-2026 covers 2020/2022 bears only. "
                              "Overlay de-risks AFTER vol/trend signal; cannot avoid single-day exogenous crash first day. cash=0%."),
        }}
        op = (PROJECT_ROOT / args.output) if not os.path.isabs(args.output) else Path(args.output)
        op.parent.mkdir(parents=True, exist_ok=True)
        with open(op, "w") as f:
            json.dump(out, f, indent=2, default=str)
        logger.info(f"  Results persisted: {op}")


if __name__ == "__main__":
    main()
