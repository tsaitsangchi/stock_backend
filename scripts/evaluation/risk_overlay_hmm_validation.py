"""
risk_overlay_hmm_validation.py v0.1 (Phase 2 — Gaussian-HMM regime as ENHANCER on the Phase-1 vol-target+trend floor)
================================================================================
**最後更新日期**: 2026-06-06
**主權狀態**: HMM-REGIME ENHANCER over RISK-OVERLAY FLOOR (incremental-value 量測) + §3.2 evaluation 橫切 (NON stock-selection) + §14.7-DC SOURCE-PURE (real price only) + §8.5 ANTI-LEAKAGE RIGOROUS (HMM 在 ≤t 擬合 / posterior 取 ≤t 序列末點 / 校準只用 label 已完整觀測之訓練樣本) + §一.10 SOURCE-TRACEABLE + §一.11 三段式 + §14.7-DG ECONOMIC-VALUE 口徑 (與 market_regime_timing / Phase-1 可比) + ⚠️ HONEST CAVEAT
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:它先用一個「高斯 HMM」從市場報酬裡**自動辨識「現在處在平靜多頭 or 動盪空頭 regime」的機率 P(bear)**,把它**校準**後,當成乘數疊到 Phase 1 那個「vol-target×趨勢」風控地板上(P(bear) 越高、曝險砍越多),然後回答最關鍵的問題:**這個 HMM regime 訊號,在已經很強的 Phase 1 地板「之上」,到底有沒有額外加值?**

**它怎麼做(步驟)**:
1. 用同一條 source-pure 等權市場序列(複用 market_regime_timing / Phase 1)。
2. **HMM regime**:每隔一段時間(預設 12 個月)用「當下看得到的過去報酬(≤ t)」重新擬合一個 2-態高斯 HMM;在每個月底 t,取「序列末點(=t)」的後驗機率 → P(bear)=報酬均值較低那一態的機率(只用 ≤ t 資料,不偷看未來)。
3. **校準**:用「標籤已完整觀測」的訓練樣本(決策日 s 滿足 s+H ≤ t)做 isotonic 校準,把 P(bear) 對齊「未來真的 bear 的機率」。
4. **疊加**:Phase 2 曝險 = Phase 1 combined 地板 × (1 − P(bear)校準後)。
5. **量測增量**:同口徑 backtest 比較 — buy&hold vs Phase1 地板 vs HMM 單獨 vs 地板×HMM → 看「地板×HMM」有沒有比「地板」更低 MaxDD / 更高 Calmar(= HMM 是否加值),並逐 2000/08/20/22 空頭檢視。

**輸入 / 輸出**:輸入 = `TaiwanStockPriceAdj`(§一.10 (b) DB)。輸出 = JSON(各 config vs buy-hold + 增量裁決 + per-episode + HMM 校準可靠度)+ stdout 摘要。

**它不做的事**:不挑股;不保證躲過單日外生閃崩(HMM 是 regime 訊號,反應慢於單日 shock);不寫 DB(read-only);不開槓桿(曝險上限沿用 Phase 1 = 1.0)。

**為什麼需要它**:Phase 1 已證「不需預測的風控地板」很有效;Phase 2 誠實檢驗「加上一個 regime 預測訊號到底值不值得」——若加值有限,正是「(B) 風控 > (A) 預測」最誠實的證據;若加值,則量化它多在哪。

## 📜 一、核心定義說明 (Core Definitions / The Constitution)

1. **[Sovereignty Declaration]** (v0.1, §3.2 evaluation 橫切): 擇時/風控 regime-enhancer 研究驗證,非 §3.1 選股。read-only;不訓練 production model;唯一職責:HMM regime 信號 → 疊 Phase 1 地板 → 增量經濟價值量測 + JSON。
2. **[Source Purity]** (§14.7-DC / §一.10): HMM 觀測 = 真實 `TaiwanStockPriceAdj` 日 log-return(+ 滾動波動),經 mathematical transform;無 hardcoded knowledge;無 imputed(warmup/缺值決策略過)。
3. **[Anti-Leakage RIGOROUS]** (§8.5): (i) HMM 僅在報酬序列 **≤ t** 擬合(每 REFIT_MONTHS 重擬合,擴張窗);(ii) 決策 t 之 P(bear) = predict_proba(X[≤t]) **末點**後驗(序列止於 t → 無未來);(iii) isotonic 校準只用 **label 已完整觀測** 之訓練樣本(決策 s 滿足 p_s + H ≤ p_t);(iv) 曝險 t 設定、**t+1 起套用**。
4. **[Economic-Value Comparability]** (§14.7-DG): 複用 market_regime_timing 之 `build_market_series` / `_series_stats` / `BEAR_EPISODES` / `SWITCH_COST` / `make_bear_label` + Phase 1 (`risk_overlay_timing`) 之 overlay 函式 → 同序列/同口徑/同事件/同成本/同地板 → buy-hold / Phase1 / market_regime_timing **三方可比**。
5. **[Incremental-Value 量測]** (v0.1 特有): 核心輸出 = 「floor×hmm」vs「floor」之 MaxDD/Calmar/Sharpe 差 → 動態裁決 `hmm_adds_value`(不硬編)。
6. **[Zero Hardcoded Verdict]** (§5.6.3): 所有 improves/reduces/adds_value 皆計算得出。
7. **[Tier-3 Operational Params]** (§一.10): HMM_N_STATES / REFIT_MONTHS / CALIB_HORIZON / random_state 為 model-config(非 feature value),透明揭露即合憲。
8. **[Reproducibility]** (§一.10 #3): HMM EM init 對 random_state 敏感 → 固定 `random_state` 求可重現;_meta 揭露;非 stochastic-metric-as-fact。
9. **[Idempotency]**: read-only;同參數+同 seed → 同結果;僅 `--output` 寫 JSON。
10. **[⚠️ HONEST CAVEAT]** (v0.1): HMM 是 **regime**(緩變)訊號 → 對單日外生閃崩反應慢;posterior 在快速轉折會滯後;熊市事件少 → 校準不確定性大;若 HMM 在 Phase 1 地板之上**加值有限或為負**,如實報(此即 (B)>(A) 之證據);cash=0%;過去不保證未來。

## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix)

### A. 主要運行模式
| 子項 | CLI | 治權契約 |
|---|---|---|
| Smoke | `python scripts/evaluation/risk_overlay_hmm_validation.py --smoke --output reports/risk_overlay_hmm_smoke.json` | 第 3 條;≥2010 子集 |
| 完整 + 輸出 | `python scripts/evaluation/risk_overlay_hmm_validation.py --output reports/risk_overlay_hmm_<ts>.json` | 第 4/5 條 |

### B. 補充
| 子項 | CLI | 治權契約 |
|---|---|---|
| HMM 態數 | `--n-states 3` | 第 7 條 |
| 重擬合週期 | `--refit-months 6` | 第 7 條 |
| seed | `--seed 5422` | 第 8 條 |
| `--commit`/`--dry-run` | 接受;僅 `--output` 寫 JSON(不寫 DB)| 第 1 條 |

### C. 輸出
| 子項 | 行為 | 治權契約 |
|---|---|---|
| JSON | `{result:{buy_and_hold, by_config:{floor,hmm_only,floor_x_hmm}, incremental, by_episode, hmm_calibration}, _meta}` | 第 4/5/9 條 |
| stdout | 各 config vs buy-hold + 增量裁決 + per-episode | §一.12 |

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
|---|---|---|---|---|
| v0.1 | 2026-06-06 | Claude | **首版(Phase 2)**:Gaussian-HMM regime(hmmlearn)當增強層疊 Phase 1 地板。**功能**:(a) walk-forward HMM(每 REFIT_MONTHS 在 ≤t 重擬合 / 末點後驗 P(bear) / 固定 seed 可重現);(b) isotonic 校準(只用 label 已觀測訓練樣本,anti-leakage);(c) 三 config(floor / hmm_only / floor_x_hmm)vs buy-hold;(d) **增量裁決**(floor×hmm vs floor 之 MaxDD/Calmar/Sharpe 差);(e) per-episode 2000/08/20/22;(f) 複用市場序列/經濟口徑/Phase1 overlay(可比)。§一.10 全 DB;§一.11 三段式;⚠️ caveat 入 _meta。 | **ACTIVE** |
"""
import sys, os, argparse, math, json, logging, time
from datetime import date, datetime
from pathlib import Path

_base_dir = Path(__file__).resolve().parent.parent  # scripts/
for _p in (str(_base_dir), str(_base_dir / "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
PROJECT_ROOT = _base_dir.parent

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.isotonic import IsotonicRegression

import market_regime_timing_validation as mrt
from market_regime_timing_validation import (
    build_market_series, month_end_decision_positions, _series_stats, make_bear_label,
    BEAR_EPISODES, SWITCH_COST, TRADING_DAYS_YEAR, DEFAULT_THRESHOLD,
)
import risk_overlay_timing_validation as ro
from core.db_utils import get_db_conn  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

CONSTITUTION_VER = "v6.1.0"
TOOL_VER = "v0.1"

# ── Tier-3 operational params (transparent disclosure — NOT feature data) ──
HMM_N_STATES = 2          # 2-state Gaussian HMM on returns (clean bull/bear); bear = lower-mean state
HMM_REFIT_MONTHS = 12     # re-fit HMM every N monthly decisions (expanding ≤ t window)
HMM_MIN_TRAIN = 252       # ≥1yr of returns before first HMM fit / OOS decision
HMM_N_ITER = 100          # EM iterations
CALIB_HORIZON = 20        # forward window (trading days) for the isotonic-calibration bear label
VOL_OBS_WINDOW = 5        # trailing-vol observation dimension window


def _hmm_observations(market_ret, idx_upto):
    """2-D observation: [daily log-return, trailing short realized vol] over positions 0..idx_upto (≤ t)."""
    r = market_ret[:idx_upto + 1]
    s = pd.Series(r)
    vol = s.rolling(VOL_OBS_WINDOW, min_periods=2).std(ddof=0).bfill().fillna(0.0).values
    X = np.column_stack([r, vol]).astype(float)
    return X


def hmm_walk_forward(market_ret, decision_positions, valid_pos, n_states, refit_months, seed):
    """Anti-leakage walk-forward HMM P(bear) per decision date.

    Re-fit GaussianHMM on returns ≤ t every `refit_months` decisions (expanding window). At each
    decision t, P(bear) = posterior (filtered at the endpoint) of the LOWER-MEAN state, computed on
    X[≤t] only (sequence ends at t → no look-ahead). Returns dict pos -> p_bear_raw.
    """
    p_raw = {}
    model = None; bear_state = None; fits = 0; since_fit = 10 ** 9
    dec = [dp for dp in decision_positions if dp < len(valid_pos) and valid_pos[dp] and dp >= HMM_MIN_TRAIN]
    for dp in dec:
        if since_fit >= refit_months or model is None:
            Xtr = _hmm_observations(market_ret, dp)   # data ≤ t only
            try:
                m = GaussianHMM(n_components=n_states, covariance_type="diag",
                                n_iter=HMM_N_ITER, random_state=seed, tol=1e-3)
                m.fit(Xtr)
                model = m
                bear_state = int(np.argmin(model.means_[:, 0]))   # lower mean-return state = bear
                fits += 1; since_fit = 0
            except Exception as e:
                logger.warning(f"  HMM fit failed at pos {dp}: {e}; keep previous params")
        if model is None:
            continue
        try:
            Xt = _hmm_observations(market_ret, dp)
            post = model.predict_proba(Xt)[-1]   # posterior at the endpoint (= t), uses ≤ t only
            p_raw[dp] = float(post[bear_state])
        except Exception:
            p_raw[dp] = p_raw.get(dp, 0.0)
        since_fit += 1
    return p_raw, fits


def calibrate_walk_forward(p_raw, fwd_cum_ret, threshold, horizon):
    """Walk-forward isotonic calibration of raw P(bear) → P(forward-bear-outcome).

    For each decision t, fit isotonic on TRAIN decisions s whose label window is fully observed
    relative to t (s + horizon <= t), mapping raw posterior -> realized bear (fwd_cum_ret[s] < -thr).
    Apply to t. Anti-leakage: only past, fully-realized labels train the calibrator. Falls back to raw
    when too few train points / one class only. Also returns reliability (binned) over all OOS decisions.
    """
    pos_sorted = sorted(p_raw.keys())
    p_cal = {}
    for t in pos_sorted:
        xs, ys = [], []
        for s in pos_sorted:
            if s + horizon <= t and not math.isnan(fwd_cum_ret[s]):
                xs.append(p_raw[s]); ys.append(1.0 if fwd_cum_ret[s] < -threshold else 0.0)
        if len(xs) >= 30 and len(set(ys)) == 2:
            try:
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                iso.fit(xs, ys)
                p_cal[t] = float(iso.predict([p_raw[t]])[0])
            except Exception:
                p_cal[t] = p_raw[t]
        else:
            p_cal[t] = p_raw[t]
    # reliability (10-bin) of calibrated prob vs realized bear, over decisions with observed label
    rel = []
    obs = [(t, p_cal[t], (1.0 if (not math.isnan(fwd_cum_ret[t]) and fwd_cum_ret[t] < -threshold) else 0.0))
           for t in pos_sorted if not math.isnan(fwd_cum_ret[t])]
    if obs:
        arr = np.array([(p, y) for _, p, y in obs])
        bins = np.clip((arr[:, 0] * 10).astype(int), 0, 9)
        for b in range(10):
            mask = bins == b
            if mask.sum() > 0:
                rel.append({"bin": b / 10.0, "n": int(mask.sum()),
                            "mean_pred": float(arr[mask, 0].mean()), "obs_freq": float(arr[mask, 1].mean())})
    return p_cal, rel


def backtest_exposures(market_dates, market_ret, dec_exposures):
    """Backtest a step-function exposure (list of (pos, exposure)) → daily LOG-returns + stats.

    Exposure set at decision pos acts from the NEXT day (pos < day); proportional switch cost on change.
    (Mirrors risk_overlay_timing / market_regime_timing economic 口徑 exactly.)
    """
    dec = sorted(dec_exposures, key=lambda x: x[0])
    if len(dec) < 2:
        return None
    oos_start = dec[0][0]; oos_end = len(market_dates) - 1
    span = list(range(oos_start + 1, oos_end + 1))
    di = 0; prev = dec[0][1]; timed = []; expo = np.zeros(len(span)); nsw = 0
    for k, day in enumerate(span):
        while di + 1 < len(dec) and dec[di + 1][0] < day:
            di += 1
        cur = dec[di][1] if dec[di][0] < day else dec[0][1]
        r = cur * market_ret[day]
        d = abs(cur - prev)
        if d > 1e-9:
            nsw += 1; r += math.log(max(1.0 - SWITCH_COST * d, 1e-9))
        timed.append(r); expo[k] = cur; prev = cur
    return {"timed_daily": timed, "exposure": expo, "span": span, "oos_start": oos_start,
            "oos_end": oos_end, "n_switches": int(nsw)}


def _episode_stats(market_dates, market_ret, dec_exposures):
    dec = sorted(dec_exposures, key=lambda x: x[0])
    out = {}
    for name, e_start, e_end in BEAR_EPISODES:
        e_idx = [i for i, d in enumerate(market_dates) if e_start <= d <= e_end]
        if len(e_idx) < 5 or not dec:
            out[name] = {"in_range": False}; continue
        bh = [market_ret[i] for i in e_idx]
        di = 0; prev = dec[0][1]; timed = []; el = []
        for day in e_idx:
            while di + 1 < len(dec) and dec[di + 1][0] < day:
                di += 1
            cur = dec[di][1] if dec[di][0] < day else dec[0][1]
            r = cur * market_ret[day]
            if el and abs(cur - el[-1]) > 1e-9:
                r += math.log(max(1.0 - SWITCH_COST * abs(cur - el[-1]), 1e-9))
            timed.append(r); el.append(cur)
        bh_mdd = _series_stats(bh)["max_drawdown"]; t_mdd = _series_stats(timed)["max_drawdown"]
        out[name] = {"in_range": True, "buy_hold_maxdd": bh_mdd, "timed_maxdd": t_mdd,
                     "buy_hold_return": float(np.exp(np.sum(bh)) - 1.0),
                     "timed_return": float(np.exp(np.sum(timed)) - 1.0),
                     "avg_exposure": float(np.mean(el)) if el else None,
                     "reduced_drawdown": bool(t_mdd is not None and bh_mdd is not None and t_mdd < bh_mdd)}
    return out


def _verdict(s, bh):
    s["reduces_maxdd_vs_buyhold"] = bool(s.get("max_drawdown") is not None and bh.get("max_drawdown") is not None and s["max_drawdown"] < bh["max_drawdown"])
    s["improves_calmar_vs_buyhold"] = bool(s.get("calmar") is not None and bh.get("calmar") is not None and s["calmar"] > bh["calmar"])
    s["improves_sharpe_vs_buyhold"] = bool(s.get("sharpe") is not None and bh.get("sharpe") is not None and s["sharpe"] > bh["sharpe"])
    return s


def run(args):
    conn = get_db_conn(); cur = conn.cursor(); t0 = time.monotonic()
    logger.info("Building equal-weight market series (reuse market_regime_timing)...")
    market_dates, market_ret, market_index = build_market_series(cur); conn.close()
    logger.info(f"Market series: {len(market_dates)} days ({market_dates[0]} → {market_dates[-1]}; {time.monotonic()-t0:.1f}s)")
    if args.smoke:
        cutoff = date(2010, 1, 1)
        keep = np.array([d >= cutoff for d in market_dates])
        if keep.sum() > 600:
            market_dates = market_dates[keep]; market_ret = market_ret[keep]
            market_index = np.exp(np.cumsum(market_ret))
            logger.info(f"  [SMOKE] ≥2010: {len(market_dates)} days")

    # Phase-1 floor signals (reuse) + decision positions
    realized_vol, above_trend, valid_pos = ro.compute_overlay_signals(
        market_dates, market_ret, market_index, ro.VOL_LOOKBACK, ro.TREND_MA)
    decision_positions = month_end_decision_positions(market_dates)

    # HMM walk-forward P(bear) + isotonic calibration (anti-leakage)
    logger.info(f"HMM walk-forward: {args.n_states}-state, refit every {args.refit_months} mo, seed {args.seed} ...")
    p_raw, n_fits = hmm_walk_forward(market_ret, decision_positions, valid_pos,
                                     args.n_states, args.refit_months, args.seed)
    fwd = make_bear_label(market_index, CALIB_HORIZON)
    p_cal, reliability = calibrate_walk_forward(p_raw, fwd, DEFAULT_THRESHOLD, CALIB_HORIZON)
    logger.info(f"  HMM fits={n_fits}; OOS decisions with P(bear)={len(p_raw)}; calibrated.")

    # decisions common to floor + hmm (need valid floor signal AND an hmm posterior)
    common = sorted([dp for dp in p_cal.keys() if dp < len(valid_pos) and valid_pos[dp]])
    if len(common) < 12:
        logger.warning("Too few common decisions; abort."); return {}, market_dates
    floor_exp = {dp: ro.decision_exposure("combined", realized_vol[dp], above_trend[dp], ro.DEFAULT_TARGET_VOL) for dp in common}

    configs = {
        "floor":        [(dp, floor_exp[dp]) for dp in common],
        "hmm_only":     [(dp, float(min(max(1.0 - p_cal[dp], 0.0), ro.MAX_EXPOSURE))) for dp in common],
        "floor_x_hmm":  [(dp, float(min(max(floor_exp[dp] * (1.0 - p_cal[dp]), 0.0), ro.MAX_EXPOSURE))) for dp in common],
    }
    ref = backtest_exposures(market_dates, market_ret, configs["floor"])
    span = ref["span"]
    bh = _series_stats([market_ret[i] for i in span]); bh["span_days"] = len(span)
    bh["span"] = [str(market_dates[ref["oos_start"] + 1]), str(market_dates[ref["oos_end"]])]

    result = {"buy_and_hold": bh, "by_config": {}, "by_episode": {}}
    cfg_stats = {}
    for name, de in configs.items():
        bt = backtest_exposures(market_dates, market_ret, de)
        s = _series_stats(bt["timed_daily"])
        s["avg_exposure"] = float(np.mean(bt["exposure"])); s["n_switches"] = bt["n_switches"]
        s = _verdict(s, bh)
        result["by_config"][name] = s; cfg_stats[name] = s
        result["by_episode"][name] = _episode_stats(market_dates, market_ret, de)

    # INCREMENTAL value of HMM enhancer over the Phase-1 floor (the core question)
    fl, fx = cfg_stats["floor"], cfg_stats["floor_x_hmm"]
    result["incremental"] = {
        "delta_maxdd_floorxhmm_minus_floor": (None if fx["max_drawdown"] is None or fl["max_drawdown"] is None else fx["max_drawdown"] - fl["max_drawdown"]),
        "delta_calmar": (None if fx["calmar"] is None or fl["calmar"] is None else fx["calmar"] - fl["calmar"]),
        "delta_sharpe": (None if fx["sharpe"] is None or fl["sharpe"] is None else fx["sharpe"] - fl["sharpe"]),
        "hmm_adds_value": bool(fx["max_drawdown"] is not None and fl["max_drawdown"] is not None
                               and fx["calmar"] is not None and fl["calmar"] is not None
                               and (fx["max_drawdown"] < fl["max_drawdown"] - 1e-6) and (fx["calmar"] > fl["calmar"] + 1e-6)),
    }
    result["hmm_calibration"] = {"n_fits": n_fits, "n_oos": len(p_raw), "reliability_bins": reliability,
                                 "avg_p_bear_raw": float(np.mean(list(p_raw.values()))) if p_raw else None,
                                 "avg_p_bear_cal": float(np.mean(list(p_cal.values()))) if p_cal else None}
    return result, market_dates


def print_summary(res):
    logger.info("=" * 110)
    logger.info("HMM-REGIME ENHANCER over Phase-1 floor — economic value (same 口徑 as Phase1 / market_regime_timing)")
    logger.info("=" * 110)
    bh = res.get("buy_and_hold", {})
    logger.info(f"  buy&hold: Sharpe={mrt._fmt(bh.get('sharpe'),'{:.2f}')} MaxDD={mrt._fmt(bh.get('max_drawdown'),'{:.3f}')} "
                f"Calmar={mrt._fmt(bh.get('calmar'),'{:.2f}')} CAGR={mrt._fmt(bh.get('cagr'),'{:.3f}')} span={bh.get('span')}")
    for name, s in res.get("by_config", {}).items():
        flags = ("MDD↓" if s.get("reduces_maxdd_vs_buyhold") else "    ") + (" Calmar↑" if s.get("improves_calmar_vs_buyhold") else "       ") + (" Sharpe↑" if s.get("improves_sharpe_vs_buyhold") else "       ")
        logger.info(f"   {name:14}: Sharpe={mrt._fmt(s.get('sharpe'),'{:.2f}')} MaxDD={mrt._fmt(s.get('max_drawdown'),'{:.3f}')} "
                    f"Calmar={mrt._fmt(s.get('calmar'),'{:.2f}')} CAGR={mrt._fmt(s.get('cagr'),'{:.3f}')} "
                    f"avgExp={mrt._fmt(s.get('avg_exposure'),'{:.2f}')} sw={s.get('n_switches')} | {flags}")
    inc = res.get("incremental", {})
    logger.info("-" * 110)
    logger.info(f"  ⮕ INCREMENTAL (floor_x_hmm − floor): ΔMaxDD={mrt._fmt(inc.get('delta_maxdd_floorxhmm_minus_floor'),'{:+.3f}')} "
                f"ΔCalmar={mrt._fmt(inc.get('delta_calmar'),'{:+.2f}')} ΔSharpe={mrt._fmt(inc.get('delta_sharpe'),'{:+.2f}')} "
                f"→ HMM adds value? {'YES' if inc.get('hmm_adds_value') else 'NO'}")
    cal = res.get("hmm_calibration", {})
    logger.info(f"  HMM: fits={cal.get('n_fits')} OOS={cal.get('n_oos')} avg P(bear) raw={mrt._fmt(cal.get('avg_p_bear_raw'),'{:.3f}')} cal={mrt._fmt(cal.get('avg_p_bear_cal'),'{:.3f}')}")
    logger.info("-" * 110)
    logger.info("  Per-episode (config=floor_x_hmm):")
    for name, e in res.get("by_episode", {}).get("floor_x_hmm", {}).items():
        if not e.get("in_range"):
            logger.info(f"   {name:18}: (not in range)"); continue
        logger.info(f"   {name:18}: bh_MaxDD={mrt._fmt(e.get('buy_hold_maxdd'),'{:.3f}')} timed_MaxDD={mrt._fmt(e.get('timed_maxdd'),'{:.3f}')} "
                    f"avgExp={mrt._fmt(e.get('avg_exposure'),'{:.2f}')} {'(reduced DD)' if e.get('reduced_drawdown') else '(did NOT)'}")


def main():
    ap = argparse.ArgumentParser(description=f"Risk Overlay + HMM Regime Enhancer (Phase 2) {TOOL_VER}")
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--n-states", type=int, default=HMM_N_STATES, dest="n_states")
    ap.add_argument("--refit-months", type=int, default=HMM_REFIT_MONTHS, dest="refit_months")
    ap.add_argument("--seed", type=int, default=5422)
    ap.add_argument("--smoke", action="store_true", help="≥2010 subset")
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--commit", action="store_true")
    args = ap.parse_args()

    logger.info("=" * 110)
    logger.info(f"Phase-2 HMM-Regime Enhancer over vol-target+trend floor {TOOL_VER}")
    logger.info(f"  ⚠️  CAVEAT: HMM is a REGIME (slow) signal; won't avoid single-day exogenous crash; few bear events → wide uncertainty; cash=0%")
    logger.info("=" * 110)
    tg = time.monotonic()
    res, market_dates = run(args)
    if res:
        print_summary(res)
    logger.info(f"\n  Total elapsed: {time.monotonic()-tg:.1f}s")

    if args.output and res:
        out = {"result": res, "_meta": {
            "tool": "risk_overlay_hmm_validation.py", "tool_ver": TOOL_VER,
            "model": f"GaussianHMM({args.n_states}-state, diag, refit/{args.refit_months}mo) P(bear) → isotonic-calibrated → "
                     f"× Phase-1 combined(vol-target {ro.DEFAULT_TARGET_VOL} × trend MA{ro.TREND_MA}) floor",
            "run_at": datetime.now().isoformat(), "constitution_ver": CONSTITUTION_VER,
            "n_states": args.n_states, "refit_months": args.refit_months, "seed": args.seed,
            "calib_horizon": CALIB_HORIZON, "switch_cost": SWITCH_COST, "max_exposure": ro.MAX_EXPOSURE,
            "n_market_days": int(len(market_dates)), "market_span": [str(market_dates[0]), str(market_dates[-1])],
            "bear_episodes": [e[0] for e in BEAR_EPISODES],
            "comparable_to": "risk_overlay_timing_validation.py (Phase 1) + market_regime_timing_validation.py (same series/口徑)",
            "anti_leakage": ("HMM fit on returns <= t (refit every N months, expanding); P(bear) = predict_proba(X[<=t])[-1] "
                             "(sequence ends at t, no look-ahead); isotonic calibrator fit only on train decisions s with "
                             "s + CALIB_HORIZON <= t (label fully observed); exposure set at t applied from t+1."),
            "source_traceability": "per §一.10 — TaiwanStockPriceAdj close → log-return/vol; no fabrication/imputation.",
            "reproducibility": f"HMM EM init fixed via random_state={args.seed} (per §一.10 #3); deterministic given seed.",
            "smoke": args.smoke,
            "HONEST_CAVEAT": ("HMM is a REGIME (slow-moving) signal: it will NOT avoid the first day of a sudden exogenous "
                              "crash and lags fast reversals. Bear episodes are FEW → calibration carries wide uncertainty. "
                              "If the HMM enhancer does NOT improve MaxDD/Calmar over the Phase-1 rule floor, that is reported "
                              "honestly (hmm_adds_value=false) and is itself evidence that risk-management (B) dominates "
                              "prediction (A). cash=0%; past results do not guarantee future."),
        }}
        op = (PROJECT_ROOT / args.output) if not os.path.isabs(args.output) else Path(args.output)
        op.parent.mkdir(parents=True, exist_ok=True)
        with open(op, "w") as f:
            json.dump(out, f, indent=2, default=str)
        logger.info(f"  Results persisted: {op}")


if __name__ == "__main__":
    main()
