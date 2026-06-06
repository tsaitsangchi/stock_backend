"""
risk_overlay_tuning_validation.py v0.1 (Step 1b — overlay 調參掃描:找「降真實策略 MaxDD 又不傷 Calmar」甜蜜點)
================================================================================
**最後更新日期**: 2026-06-06
**主權狀態**: RISK-OVERLAY PARAM-TUNING SWEEP on REAL strategy(§14.7-DH Step 1b)+ §3.2 evaluation 橫切 + §14.7-DC SOURCE-PURE + §8.5 ANTI-LEAKAGE + §一.10 SOURCE-TRACEABLE + §一.11 三段式 + §14.7-DE Canonical Panel + ⚠️ HONEST CAVEAT
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:Step 1 發現「基準 overlay(target_vol 0.15)在真實策略上降 MaxDD 但傷 Calmar(過度減碼)」;這支**掃多組參數**(提高 target_vol、只用趨勢、用 HMM regime 當開關)套到**同一條真實 top-20 策略報酬流**上,找出「**降 MaxDD 且 Calmar 不變差**」的甜蜜點 config(§14.7-DH T_DH-3 乾淨 gate)。

**它怎麼做(步驟)**:
1. 算一次真實策略報酬流(複用 Step 1:base LightGBM top-20 walk-forward,canonical panels)。
2. 算一次市場 vol/趨勢訊號 + HMM 校準 P(bear)(複用 Phase1/Phase2,皆 ≤t anti-leakage)。
3. 對多組 config 各算曝險→套策略→MaxDD/Calmar/Sharpe vs raw 策略(每組只是不同縮放,秒級)。
4. 標出**通過乾淨 gate(MaxDD↓ ∧ Calmar 不變差)**的甜蜜點。

**它不做的事**:不改 production(找到甜蜜點後是否進 Step 2 由用戶 + treaty 決定);不挑新股;不寫 DB。

**為什麼需要它**:Step 1 證明基準 overlay 接 production 會傷 Calmar;Step 1b 找有沒有一組參數能「只在該減碼時減碼」,既降尾部回撤又不犧牲風險調整報酬——這是 Step 2 整合的前提。

## 📜 一、核心定義說明 (Core Definitions / The Constitution)

1. **[Sovereignty Declaration]** (v0.1, §3.2 橫切): overlay 調參研究;read-only;不寫 model_registry;不改 production。
2. **[Source Purity]** (§14.7-DC/§一.10): 全 DB(feature_values + TaiwanStockPriceAdj);訊號 = price→math transform;無 impute/fabricate。
3. **[Anti-Leakage]** (§8.5): 策略選股 walk-forward [0..i-1]→i;overlay vol/趨勢 ≤as_of;HMM 末點後驗 + 校準只用 label 已觀測樣本。
4. **[Canonical Panel]** (§14.7-DE/§一.16): panels 由 get_canonical_panel_dates 單一源。
5. **[Economic-Value Gate]** (§14.7-DH T_DH-3): 甜蜜點 = MaxDD < raw **且** Calmar ≥ raw(動態裁決,不硬編)。
6. **[Reproducible]** (§一.10 #3): LightGBM/HMM 固定 seed。
7. **[⚠️ HONEST CAVEAT]** (v0.1): 多組 config 之擇優屬 in-sample 調參(2018-2026 同一窗)→ 有過擬合風險;甜蜜點須註明「需 OOS/前進測試再確認」;monthly 解析度漏單日閃崩;cash=0%;策略用 base LightGBM 為代表。

## 📊 二、全量維運指令總矩陣

| 子項 | CLI | 治權契約 |
|---|---|---|
| Smoke | `... risk_overlay_tuning_validation.py --smoke --output reports/risk_overlay_tuning_smoke.json` | 限 panel |
| 完整 + 輸出 | `... risk_overlay_tuning_validation.py --output reports/risk_overlay_tuning_<ts>.json` | 全 config |
| `--commit`/`--dry-run` | 接受;僅 `--output` 寫 JSON | read-only |

## 📜 三、全修訂歷程

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
|---|---|---|---|---|
| v0.1 | 2026-06-06 | Claude | **首版(§14.7-DH Step 1b)**:overlay 調參掃描(target_vol 0.15-0.40 / trend-only / voltarget-only / regime-gated HMM / trend×regime)套真實 top-20 策略;找 MaxDD↓∧Calmar 不傷之甜蜜點。複用 Step1(strategy_returns/eq_stats)+ Phase1(overlay)+ Phase2(HMM)。§一.10 全 DB;§一.11 三段式;in-sample 調參 caveat 入 _meta。 | **ACTIVE** |
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

import market_regime_timing_validation as mrt
from market_regime_timing_validation import build_market_series, make_bear_label, BEAR_EPISODES, DEFAULT_THRESHOLD
import risk_overlay_timing_validation as ro
import risk_overlay_hmm_validation as rh
import risk_overlay_strategy_validation as rs
from core.db_utils import get_db_conn, get_canonical_panel_dates  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
CONSTITUTION_VER = "v6.1.0"; TOOL_VER = "v0.1"
REGIME_GATE_THR = 0.5   # regime_gated_step: 全投資 unless P(bear) > thr


def _vt(vol, target):   # vol-target scaler clipped
    return float(min(max(target / max(vol, ro.MIN_VOL), 0.0), ro.MAX_EXPOSURE))


def exposure_for(name, vol, trend, pbear):
    """各 config 之曝險 ∈ [0, MAX_EXPOSURE]. vol=年化波動, trend=bool(>MA200), pbear=校準 P(bear)."""
    tflag = 1.0 if bool(trend) else 0.0
    if name == "combined_tv0.15":   return min(_vt(vol, 0.15) * tflag, ro.MAX_EXPOSURE)
    if name == "combined_tv0.25":   return min(_vt(vol, 0.25) * tflag, ro.MAX_EXPOSURE)
    if name == "combined_tv0.40":   return min(_vt(vol, 0.40) * tflag, ro.MAX_EXPOSURE)
    if name == "trend_only":        return tflag
    if name == "voltarget_tv0.25":  return _vt(vol, 0.25)
    if name == "regime_only":       return float(min(max(1.0 - pbear, 0.0), ro.MAX_EXPOSURE))
    if name == "trend_x_regime":    return float(min(tflag * (1.0 - pbear), ro.MAX_EXPOSURE))
    if name == "regime_gated_step": return 1.0 if pbear <= REGIME_GATE_THR else 0.0
    raise ValueError(name)


CONFIGS = ["combined_tv0.15", "combined_tv0.25", "combined_tv0.40", "trend_only",
           "voltarget_tv0.25", "regime_only", "trend_x_regime", "regime_gated_step"]


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
        panels = panels[-30:]
    logger.info(f"Universe {len(universe)} core; panels {len(panels)}; strategy walk-forward (base LightGBM top-20)...")
    strat = rs.strategy_returns(cur, panels, universe, rs.STRAT_HORIZON, args.seed)
    logger.info(f"  strategy OOS panels: {len(strat)} ({time.monotonic()-t0:.0f}s)")
    if len(strat) < 12:
        conn.close(); return {}

    mdates, mret, midx = build_market_series(cur); conn.close()
    realized_vol, above_trend, valid_pos = ro.compute_overlay_signals(mdates, mret, midx, ro.VOL_LOOKBACK, ro.TREND_MA)
    ts_arr = np.array([pd.Timestamp(d) for d in mdates])

    # HMM calibrated P(bear) at each strategy as_of (anti-leakage walk-forward on those market positions)
    aods = [s[0] for s in strat]
    dec_pos = [int(np.searchsorted(ts_arr, pd.Timestamp(ao), side="right") - 1) for ao in aods]
    valid_full = np.ones(len(mdates), dtype=bool)  # use all (HMM has own MIN_TRAIN)
    p_raw, n_fits = rh.hmm_walk_forward(mret, dec_pos, valid_full, rh.HMM_N_STATES, rh.HMM_REFIT_MONTHS, args.seed)
    fwd = make_bear_label(midx, rh.CALIB_HORIZON)
    p_cal, _ = rh.calibrate_walk_forward(p_raw, fwd, DEFAULT_THRESHOLD, rh.CALIB_HORIZON)

    # per-as_of signals
    raw = [s[1] for s in strat]
    raw_stats = rs.eq_stats(raw)
    sig = []  # (as_of, top20, vol, trend, pbear)
    for (ao, top20, _u), dp in zip(strat, dec_pos):
        p = dp if (0 <= dp < len(valid_pos)) else len(valid_pos) - 1
        vol = realized_vol[p] if not (p < 0) else np.nan
        trend = above_trend[p] if (0 <= p < len(above_trend)) else True
        valid = bool(0 <= p < len(valid_pos) and valid_pos[p])
        pb = p_cal.get(p, 0.0)
        sig.append((ao, top20, vol, trend, pb, valid))

    out = {"raw_strategy": raw_stats, "by_config": {}, "sweet_spots": []}
    for name in CONFIGS:
        ov, exps = [], []
        for (ao, top20, vol, trend, pb, valid) in sig:
            e = exposure_for(name, vol, trend, pb) if valid else 1.0  # no signal → fully invested
            ov.append(e * top20); exps.append(e)
        s = rs.eq_stats(ov)
        s["avg_exposure"] = float(np.mean(exps))
        s["reduces_maxdd"] = bool(s["maxdd"] is not None and raw_stats["maxdd"] is not None and s["maxdd"] < raw_stats["maxdd"] - 1e-6)
        s["calmar_not_worse"] = bool(s["calmar"] is not None and raw_stats["calmar"] is not None and s["calmar"] >= raw_stats["calmar"] - 1e-6)
        s["sweet_spot"] = bool(s["reduces_maxdd"] and s["calmar_not_worse"])
        s["delta_maxdd"] = (None if s["maxdd"] is None else s["maxdd"] - raw_stats["maxdd"])
        s["delta_calmar"] = (None if s["calmar"] is None else s["calmar"] - raw_stats["calmar"])
        out["by_config"][name] = s
        if s["sweet_spot"]:
            out["sweet_spots"].append(name)
    out["hmm_fits"] = n_fits; out["n_strategy_panels"] = len(strat)
    out["strategy_span"] = [str(strat[0][0]), str(strat[-1][0])]
    return out


def main():
    ap = argparse.ArgumentParser(description=f"Risk Overlay tuning sweep on real strategy (Step 1b) {TOOL_VER}")
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--seed", type=int, default=5422)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--commit", action="store_true")
    args = ap.parse_args()
    logger.info("=" * 110)
    logger.info(f"Risk Overlay PARAM-TUNING sweep on REAL top-20 strategy (§14.7-DH Step 1b) {TOOL_VER}")
    logger.info("  目標:找 MaxDD↓ ∧ Calmar 不變差 之甜蜜點。⚠️ in-sample 調參,甜蜜點須 OOS 再確認。cash=0%")
    logger.info("=" * 110)
    tg = time.monotonic()
    res = run(args)
    if res:
        rs_ = res["raw_strategy"]
        logger.info(f"  RAW strategy ({res['strategy_span']}, {res['n_strategy_panels']} panels): "
                    f"MaxDD={mrt._fmt(rs_['maxdd'],'{:.3f}')} Calmar={mrt._fmt(rs_['calmar'],'{:.2f}')} "
                    f"Sharpe={mrt._fmt(rs_['sharpe'],'{:.2f}')} CAGR={mrt._fmt(rs_['cagr'],'{:.3f}')}")
        logger.info("  " + "-" * 100)
        for name, s in res["by_config"].items():
            tag = "★甜蜜點" if s["sweet_spot"] else ("MDD↓" if s["reduces_maxdd"] else "    ")
            logger.info(f"  {name:18}: MaxDD={mrt._fmt(s['maxdd'],'{:.3f}')}({mrt._fmt(s['delta_maxdd'],'{:+.3f}')}) "
                        f"Calmar={mrt._fmt(s['calmar'],'{:.2f}')}({mrt._fmt(s['delta_calmar'],'{:+.2f}')}) "
                        f"Sharpe={mrt._fmt(s['sharpe'],'{:.2f}')} CAGR={mrt._fmt(s['cagr'],'{:.3f}')} "
                        f"avgExp={mrt._fmt(s['avg_exposure'],'{:.2f}')} | {tag}")
        logger.info("  " + "-" * 100)
        logger.info(f"  ⮕ 甜蜜點(MaxDD↓ ∧ Calmar 不傷): {res['sweet_spots'] if res['sweet_spots'] else '無 — 無 config 能既降回撤又不傷 Calmar'}")
    logger.info(f"\n  Total elapsed: {time.monotonic()-tg:.1f}s")
    if args.output and res:
        out = {"result": res, "_meta": {
            "tool": "risk_overlay_tuning_validation.py", "tool_ver": TOOL_VER, "run_at": datetime.now().isoformat(),
            "constitution_ver": CONSTITUTION_VER, "configs": CONFIGS, "seed": args.seed, "smoke": args.smoke,
            "regime_gate_thr": REGIME_GATE_THR, "gate": "sweet_spot = MaxDD < raw AND Calmar >= raw (§14.7-DH T_DH-3)",
            "strategy_model": "base LightGBM top-20", "anti_leakage": "stock pick [0..i-1]→i; overlay/HMM ≤as_of; calib label-observed-only.",
            "source_traceability": "per §一.10 — feature_values + TaiwanStockPriceAdj; canonical panels via get_canonical_panel_dates.",
            "HONEST_CAVEAT": ("IN-SAMPLE tuning over one window (2018-2026); picking the best config risks overfitting → "
                              "any sweet spot MUST be re-confirmed out-of-sample / forward before Step-2 production integration. "
                              "monthly-panel resolution misses single-day crashes; cash=0%; representative base LightGBM strategy."),
        }}
        op = (PROJECT_ROOT / args.output) if not os.path.isabs(args.output) else Path(args.output)
        op.parent.mkdir(parents=True, exist_ok=True)
        with open(op, "w") as f:
            json.dump(out, f, indent=2, default=str)
        logger.info(f"  Results persisted: {op}")


if __name__ == "__main__":
    main()
