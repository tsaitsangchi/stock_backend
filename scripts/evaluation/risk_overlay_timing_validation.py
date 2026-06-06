"""
risk_overlay_timing_validation.py v0.1 (Phase 1 — Volatility-Target + Trend Risk Overlay, rule-based / no-ML)
================================================================================
**最後更新日期**: 2026-06-06
**主權狀態**: RISK-MANAGEMENT OVERLAY (vol-target + trend-filter) RULE BACKTEST + §3.2 evaluation 橫切 (NON stock-selection) + §14.7-DC SOURCE-PURE (real price only, no impute, no hardcoded knowledge) + §8.5 ANTI-LEAKAGE (exposure 在決策日 t 僅用 ≤ t 資料設定,t+1 起套用) + §一.10 SOURCE-TRACEABLE (全 DB) + §一.11 三段式合規 + §14.7-DG ECONOMIC-VALUE 口徑 (與 market_regime_timing 可比) + ⚠️ HONEST CAVEAT (overlay 限制傷害,非躲過第一天)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:它不挑「買哪些股」,而是回答「現在該把多少比例的錢放在市場裡」——當市場波動變大、或跌破長期均線時,自動把曝險降下來,藉此**降低大跌時的回撤**;這是一個「風險管理外掛(overlay)」,不是預測模型。

**它怎麼做(步驟)**:
1. 用資料庫全市場股價建一條「等權市場指數」(與 market_regime_timing 完全同一條,保證可比)。
2. 每個月底,只用「當天看得到的過去資料」算兩個訊號:(a) 近 60 日已實現波動率;(b) 收盤價相對 200 日均線(在不在多頭趨勢)。
3. 依規則決定下個月的市場曝險比例:
   - **波動目標**:曝險 = 目標波動 ÷ 近期波動(波動越大、曝險越低),上限 100%(只減碼、不開槓桿)。
   - **趨勢過濾**:跌破 200 日均線就把曝險降到 0(或設定值)。
   - **合成**:兩者相乘。
4. 把這個曝險套用到「下個月」的實際市場報酬(t 設定、t+1 起生效 → 不偷看未來),換手扣成本。
5. 跟「always 滿倉 buy-and-hold」比:**最大回撤(MaxDD)/ Calmar / Sharpe / 在場時間 / 換手次數**,並逐一檢視 2000/2008/2020/2022 四次空頭有沒有真的降低回撤。

**輸入 / 輸出**:輸入 = `TaiwanStockPriceAdj`(股價,§一.10 (b) DB)。輸出 = JSON(各 overlay 配置 vs buy-hold 的經濟價值 + per-episode)+ 可讀 stdout 摘要。

**它不做的事**:不挑股(那是橫斷面選股模型的事);不預測「哪一天會崩」(單日外生閃崩擋不住,只能在波動噴出後減碼、限制傷害);不寫 model_registry / 不改任何 DB(read-only);不開槓桿(曝險上限 100%)。

**為什麼需要它**:這是「先建穩健保護地板」的 Phase 1——不需任何預測就能降低回撤;之後 Phase 2 再把校準後的 regime 機率當「增強層」疊上去,量測它到底有沒有在這個地板之上額外加值。

## 📜 一、核心定義說明 (Core Definitions / The Constitution)

1. **[Sovereignty Declaration]** (v0.1, 憲法 §3.2 evaluation 橫切): 本程式為**擇時/風控 overlay 研究驗證**,非 §3.1 序列選股模組。治權邊界:(a) 不挑股(回答「曝險多少」非「買哪些」);(b) 不涉五套禁令之 feature 層(只用 price-derived vol/trend);(c) 不分 T1-T3;(d) read-only(不改 `TaiwanStockPriceAdj`);(e) 不訓練 production model(不寫 model_registry / model_training_run);(f) 唯一職責:rule-based overlay 曝險 → 經濟價值 backtest vs buy-hold + per-episode + JSON。
2. **[Source Purity]** (v0.1, §14.7-DC / §一.10 / §一.13): 全訊號 = 真實 `TaiwanStockPriceAdj` close 經 mathematical transform(log-return / rolling realized-vol / MA ratio)。**無 hardcoded knowledge dict;無 imputed / 無 fabricate**;NaN(warmup 未足)→ 該決策日略過(不前視填補)。
3. **[Anti-Leakage]** (v0.1, §8.5): exposure 在決策日 t 僅用 **≤ t** 之 close 計算(rolling 窗皆 backward);套用於 **t+1 起**之日報酬(mirror market_regime_timing economic_block 之 `decision_pos < day` 推進邏輯)。無未來洩漏。
4. **[Economic-Value Comparability]** (v0.1, §14.7-DG): 複用 `market_regime_timing_validation` 之 `build_market_series` / `_series_stats` / `BEAR_EPISODES` / `SWITCH_COST` / `TRADING_DAYS_YEAR` → **同一市場序列 + 同一 MaxDD/Calmar/Sharpe 口徑 + 同一空頭事件 + 同一換手成本** → 與 regime 擇時模型**可直接並列比較**(apples-to-apples)。
5. **[Zero Hardcoded Verdict]** (§5.6.3): reduces_maxdd / improves_calmar / improves_sharpe 等裁決皆**動態計算**(overlay 數值 vs buy-hold 比較),不硬編 PASS/FAIL。
6. **[Tier-3 Operational Params]** (§一.10 五階): TARGET_VOL / VOL_LOOKBACK / TREND_MA / MAX_EXPOSURE / 月度再平衡 為**風控/配置層 operational params**(類比 cost_per_rebal),**非 feature value、不入模型特徵層** → 透明揭露即合憲(同 §14.7-DG 之 0.6% 成本揭露)。
7. **[Historical Reference Authority]** (v0.1): TOOL_VER 屬記述性快照;buy-hold / regime 對比值來自既有 `reports/*` JSON(權威來源)。
8. **[Idempotency]** (v0.1): read-only + 純計算;重跑同參數 → 同結果(無隨機性;規則型無 seed 依賴)。僅 `--output` 寫 JSON。
9. **[⚠️ HONEST CAVEAT — Overlay 限制傷害非躲過第一天]** (v0.1): vol-target overlay 是「波動噴出**後**減碼」→ 對**單日外生閃崩(如 2026-06-05 SOX −10%)無法事前避開第一天**,只能限制後續傷害;趨勢過濾在快速 V 轉時會晚進晚出;cash=0 假設(保守,無無風險收益);過去降低回撤**不保證**未來。

## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix)

### A. 主要運行模式
| 子項 | 對應 CLI | 治權契約 |
|---|---|---|
| Smoke(plumbing)| `python scripts/evaluation/risk_overlay_timing_validation.py --smoke --output reports/risk_overlay_smoke.json` | 第 2/3 條;≥2015 子集 |
| 完整 + 輸出 | `python scripts/evaluation/risk_overlay_timing_validation.py --output reports/risk_overlay_timing_<ts>.json` | 第 4 條 |

### B. 補充運行模式
| 子項 | 對應 CLI | 治權契約 |
|---|---|---|
| 自訂目標波動 | `--target-vol 0.12` | 第 6 條 Tier 3 |
| 自訂趨勢均線 | `--trend-ma 150` | 第 6 條 Tier 3 |
| `--commit`/`--dry-run` | 接受但語意 = 純計算列印;僅 `--output` 寫 JSON(不寫 DB)| 第 1 條 read-only |

### C. 輸出
| 子項 | 對應行為 | 治權契約 |
|---|---|---|
| JSON | `{overlay:{buy_and_hold, by_config:{voltarget_only,trend_only,combined}, by_episode}, _meta(含 caveat)}` | 第 4/9 條 |
| stdout | 各 config vs buy-hold 經濟價值表 + per-episode | §一.12 |

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
|---|---|---|---|---|
| v0.1 | 2026-06-06 | Claude | **首版(Phase 1)**:vol-target + trend 規則型風控 overlay。複用 market_regime_timing 之 source-pure 市場序列 + 經濟口徑(可比)。**功能**:(a) 月度再平衡曝險 = clip(目標波動/近期波動 × 趨勢旗標, 0, 1);(b) 三配置(voltarget_only / trend_only / combined)vs buy-hold;(c) MaxDD/Calmar/Sharpe/in-mkt%/換手 + 比例換手成本;(d) per-episode 2000/08/20/22 降回撤檢驗;(e) anti-leakage(t 設定 / t+1 套用);(f) read-only / 無 ML / 無 seed。§一.10 全 DB source-traceable;§一.11 三段式;⚠️ caveat 入 _meta。為 Phase 2(HMM/校準 regime 增量)之穩健地板。 | **ACTIVE** |
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

# Reuse source-pure building blocks from the regime-timing validator → identical market series +
# identical economic-value 口徑 + identical bear episodes + identical switch cost (apples-to-apples).
import market_regime_timing_validation as mrt
from market_regime_timing_validation import (
    build_market_series, month_end_decision_positions, _series_stats,
    BEAR_EPISODES, SWITCH_COST, TRADING_DAYS_YEAR,
)
from core.db_utils import get_db_conn  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

CONSTITUTION_VER = "v6.1.0"
TOOL_VER = "v0.1"

# ── Tier-3 operational params (transparent disclosure — NOT feature data, NOT hardcoded verdict) ──
DEFAULT_TARGET_VOL = 0.15     # annualized vol target for the vol-targeting scaler
VOL_LOOKBACK = 60            # trailing trading days for realized-vol forecast
TREND_MA = 200              # trend filter moving-average length (trading days)
MAX_EXPOSURE = 1.0          # long-only, no leverage → overlay can only DE-risk
TREND_OFF_EXPOSURE = 0.0    # exposure when price < trend MA (full de-risk)
MIN_VOL = 0.02              # floor on realized vol to avoid div-by-tiny blow-up of vol scaler


def compute_overlay_signals(market_dates, market_ret, market_index, vol_lookback, trend_ma):
    """As-of signals (all ≤ each row's own date): annualized trailing realized vol + price/MA ratio.

    Anti-leakage: rolling windows look BACKWARD only; the value at date t uses only data dated ≤ t.
    """
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in market_dates])
    ret = pd.Series(market_ret, index=idx)
    px = pd.Series(market_index, index=idx)
    realized_vol = ret.rolling(vol_lookback, min_periods=vol_lookback // 2).std(ddof=1) * math.sqrt(TRADING_DAYS_YEAR)
    ma = px.rolling(trend_ma, min_periods=trend_ma // 2).mean()
    above_trend = (px > ma)                          # True/False; NaN-safe handled at decision time
    return realized_vol.values, above_trend.values, ma.notna().values & realized_vol.notna().values


def decision_exposure(config, realized_vol_t, above_trend_t, target_vol):
    """Map as-of signals at a decision date → target market exposure in [0, MAX_EXPOSURE] for `config`."""
    vt = target_vol / max(realized_vol_t, MIN_VOL)          # vol-targeting scaler
    vt = min(max(vt, 0.0), MAX_EXPOSURE)
    trend = 1.0 if bool(above_trend_t) else TREND_OFF_EXPOSURE
    if config == "voltarget_only":
        e = vt
    elif config == "trend_only":
        e = trend
    elif config == "combined":
        e = vt * trend
    else:
        raise ValueError(config)
    return float(min(max(e, 0.0), MAX_EXPOSURE))


def backtest_config(config, market_dates, market_ret, decision_positions, valid_pos,
                    realized_vol, above_trend, target_vol):
    """Step-function exposure (set at each month-end decision ≤ t, applied from t+1) vs buy-hold.

    Returns daily timed LOG-returns over the OOS span + exposure stats. Switch cost applied as a log
    drag PROPORTIONAL to |Δexposure| on the day exposure changes (generalizes the binary regime cost).
    """
    # decisions that have valid signals (after warmup); each carries its target exposure
    dec = [(dp, decision_exposure(config, realized_vol[dp], above_trend[dp], target_vol))
           for dp in decision_positions if dp < len(valid_pos) and valid_pos[dp]]
    if len(dec) < 2:
        return None
    oos_start = dec[0][0]
    oos_end = len(market_dates) - 1
    span = list(range(oos_start + 1, oos_end + 1))
    timed_daily = []
    exposure = np.zeros(len(span))
    di = 0; prev_exp = dec[0][1]; n_switches = 0; turnover = 0.0
    for k, day in enumerate(span):
        while di + 1 < len(dec) and dec[di + 1][0] < day:
            di += 1
        cur_exp = dec[di][1] if dec[di][0] < day else dec[0][1]
        r = cur_exp * market_ret[day]
        d_exp = abs(cur_exp - prev_exp)
        if d_exp > 1e-9:
            n_switches += 1; turnover += d_exp
            r += math.log(max(1.0 - SWITCH_COST * d_exp, 1e-9))  # proportional switch cost as log drag
        timed_daily.append(r); exposure[k] = cur_exp; prev_exp = cur_exp
    return {"timed_daily": timed_daily, "exposure": exposure, "span": span,
            "oos_start": oos_start, "oos_end": oos_end,
            "n_switches": int(n_switches), "avg_turnover_per_change": (turnover / n_switches) if n_switches else 0.0}


def per_episode(config, market_dates, market_ret, decision_positions, valid_pos,
                realized_vol, above_trend, target_vol):
    """For each known bear episode: did the overlay reduce the drawdown vs buy-hold within the window?"""
    dec = [(dp, decision_exposure(config, realized_vol[dp], above_trend[dp], target_vol))
           for dp in decision_positions if dp < len(valid_pos) and valid_pos[dp]]
    out = {}
    for name, e_start, e_end in BEAR_EPISODES:
        e_idx = [i for i, d in enumerate(market_dates) if e_start <= d <= e_end]
        if len(e_idx) < 5 or not dec:
            out[name] = {"in_range": False, "note": "episode outside market-date / signal range"}
            continue
        bh_daily = [market_ret[i] for i in e_idx]
        bh_ret = float(np.exp(np.sum(bh_daily)) - 1.0)
        bh_mdd = _series_stats(bh_daily)["max_drawdown"]
        di = 0; prev_exp = dec[0][1]; timed_daily = []; exp_list = []
        for day in e_idx:
            while di + 1 < len(dec) and dec[di + 1][0] < day:
                di += 1
            cur_exp = dec[di][1] if dec[di][0] < day else dec[0][1]
            r = cur_exp * market_ret[day]
            if exp_list and abs(cur_exp - exp_list[-1]) > 1e-9:
                r += math.log(max(1.0 - SWITCH_COST * abs(cur_exp - exp_list[-1]), 1e-9))
            timed_daily.append(r); exp_list.append(cur_exp)
        t_mdd = _series_stats(timed_daily)["max_drawdown"]
        out[name] = {
            "in_range": True, "window": [str(e_start), str(e_end)],
            "buy_hold_return": bh_ret, "buy_hold_maxdd": bh_mdd,
            "timed_return": float(np.exp(np.sum(timed_daily)) - 1.0), "timed_maxdd": t_mdd,
            "avg_exposure": float(np.mean(exp_list)) if exp_list else None,
            "reduced_drawdown": bool(t_mdd is not None and bh_mdd is not None and t_mdd < bh_mdd),
        }
    return out


CONFIGS = ["voltarget_only", "trend_only", "combined"]


def run(args):
    conn = get_db_conn(); cur = conn.cursor()
    t0 = time.monotonic()
    logger.info("Building equal-weight market series (reuse market_regime_timing; all numeric stocks)...")
    market_dates, market_ret, market_index = build_market_series(cur)
    conn.close()
    logger.info(f"Market series: {len(market_dates)} trading days "
                f"({market_dates[0]} → {market_dates[-1]}; load {time.monotonic()-t0:.1f}s)")
    if args.smoke:
        cutoff = date(2015, 1, 1)
        keep = np.array([d >= cutoff for d in market_dates])
        if keep.sum() > 300:
            market_dates = market_dates[keep]; market_ret = market_ret[keep]
            market_index = np.exp(np.cumsum(market_ret))
            logger.info(f"  [SMOKE] limited to ≥2015: {len(market_dates)} trading days")

    realized_vol, above_trend, valid_pos = compute_overlay_signals(
        market_dates, market_ret, market_index, args.vol_lookback, args.trend_ma)
    decision_positions = month_end_decision_positions(market_dates)
    logger.info(f"Decision dates: {len(decision_positions)} month-ends; "
                f"vol_lookback={args.vol_lookback} trend_ma={args.trend_ma} target_vol={args.target_vol:.0%}")

    overlay = {"by_config": {}, "by_episode": {}}
    # buy-and-hold over the FULL valid span (first valid decision → end) for reference
    ref = backtest_config("trend_only", market_dates, market_ret, decision_positions, valid_pos,
                          realized_vol, above_trend, args.target_vol)
    if ref is None:
        logger.warning("Insufficient valid decisions (warmup too long for span).")
        return {}, market_dates
    span = ref["span"]
    bh = _series_stats([market_ret[i] for i in span])
    bh["span_days"] = len(span)
    bh["span"] = [str(market_dates[ref["oos_start"] + 1]), str(market_dates[ref["oos_end"]])]
    overlay["buy_and_hold"] = bh

    for cfg in CONFIGS:
        bt = backtest_config(cfg, market_dates, market_ret, decision_positions, valid_pos,
                             realized_vol, above_trend, args.target_vol)
        if bt is None:
            overlay["by_config"][cfg] = {"note": "insufficient decisions"}; continue
        s = _series_stats(bt["timed_daily"])
        s["pct_time_in_market"] = float(np.mean(bt["exposure"]))
        s["avg_exposure"] = float(np.mean(bt["exposure"]))
        s["n_switches"] = bt["n_switches"]
        s["avg_turnover_per_change"] = bt["avg_turnover_per_change"]
        s["reduces_maxdd_vs_buyhold"] = bool(s["max_drawdown"] is not None and bh["max_drawdown"] is not None
                                             and s["max_drawdown"] < bh["max_drawdown"])
        s["improves_calmar_vs_buyhold"] = bool(s["calmar"] is not None and bh["calmar"] is not None
                                               and s["calmar"] > bh["calmar"])
        s["improves_sharpe_vs_buyhold"] = bool(s["sharpe"] is not None and bh["sharpe"] is not None
                                               and s["sharpe"] > bh["sharpe"])
        overlay["by_config"][cfg] = s
        overlay["by_episode"][cfg] = per_episode(cfg, market_dates, market_ret, decision_positions,
                                                 valid_pos, realized_vol, above_trend, args.target_vol)
    return overlay, market_dates


def print_summary(overlay):
    logger.info("=" * 110)
    logger.info("RISK OVERLAY (vol-target + trend) — economic value vs BUY-HOLD (same series/口徑 as market_regime_timing)")
    logger.info("=" * 110)
    bh = overlay.get("buy_and_hold", {})
    logger.info(f"  buy&hold: Sharpe={mrt._fmt(bh.get('sharpe'),'{:.2f}')} "
                f"MaxDD={mrt._fmt(bh.get('max_drawdown'),'{:.3f}')} Calmar={mrt._fmt(bh.get('calmar'),'{:.2f}')} "
                f"CAGR={mrt._fmt(bh.get('cagr'),'{:.3f}')} span={bh.get('span')}")
    for cfg, s in overlay.get("by_config", {}).items():
        if "sharpe" not in s:
            logger.info(f"   {cfg:16}: {s.get('note')}"); continue
        flags = ("MDD↓" if s.get("reduces_maxdd_vs_buyhold") else "    ") + \
                (" Calmar↑" if s.get("improves_calmar_vs_buyhold") else "       ") + \
                (" Sharpe↑" if s.get("improves_sharpe_vs_buyhold") else "       ")
        logger.info(f"   {cfg:16}: Sharpe={mrt._fmt(s.get('sharpe'),'{:.2f}')} "
                    f"MaxDD={mrt._fmt(s.get('max_drawdown'),'{:.3f}')} Calmar={mrt._fmt(s.get('calmar'),'{:.2f}')} "
                    f"CAGR={mrt._fmt(s.get('cagr'),'{:.3f}')} avgExp={mrt._fmt(s.get('avg_exposure'),'{:.2f}')} "
                    f"sw={s.get('n_switches')} | {flags}")
    logger.info("-" * 110)
    logger.info("  Per-episode (config=combined; did overlay reduce the drawdown?):")
    for name, e in overlay.get("by_episode", {}).get("combined", {}).items():
        if not e.get("in_range"):
            logger.info(f"   {name:18}: (not in range)"); continue
        logger.info(f"   {name:18}: bh_ret={mrt._fmt(e.get('buy_hold_return'),'{:+.2%}')} "
                    f"timed_ret={mrt._fmt(e.get('timed_return'),'{:+.2%}')} "
                    f"bh_MaxDD={mrt._fmt(e.get('buy_hold_maxdd'),'{:.3f}')} "
                    f"timed_MaxDD={mrt._fmt(e.get('timed_maxdd'),'{:.3f}')} "
                    f"avgExp={mrt._fmt(e.get('avg_exposure'),'{:.2f}')} "
                    f"{'(reduced DD)' if e.get('reduced_drawdown') else '(did NOT reduce)'}")


def main():
    ap = argparse.ArgumentParser(description=f"Risk Overlay (vol-target + trend) Timing Validation {TOOL_VER}")
    ap.add_argument("--output", type=str, default=None, help="write results JSON to repo-relative path")
    ap.add_argument("--target-vol", type=float, default=DEFAULT_TARGET_VOL, dest="target_vol")
    ap.add_argument("--vol-lookback", type=int, default=VOL_LOOKBACK, dest="vol_lookback")
    ap.add_argument("--trend-ma", type=int, default=TREND_MA, dest="trend_ma")
    ap.add_argument("--smoke", action="store_true", help="plumbing check: ≥2015 subset")
    ap.add_argument("--dry-run", action="store_true", help="compute+print only (default; accepted for symmetry)")
    ap.add_argument("--commit", action="store_true", help="no DB write; accepted for symmetry (use --output to persist)")
    args = ap.parse_args()

    logger.info("=" * 110)
    logger.info(f"Risk Overlay (vol-target + trend) Timing Validation {TOOL_VER} — answers HOW MUCH in market (not which stocks)")
    logger.info(f"  ⚠️  CAVEAT: overlay LIMITS damage (de-risks AFTER vol spikes); cannot avoid a single-day exogenous crash; cash=0%")
    logger.info("=" * 110)
    t_global = time.monotonic()
    overlay, market_dates = run(args)
    if overlay:
        print_summary(overlay)
    logger.info(f"\n  Total elapsed: {time.monotonic()-t_global:.1f}s")

    if args.output and overlay:
        out = {"overlay": overlay}
        out["_meta"] = {
            "tool": "risk_overlay_timing_validation.py", "tool_ver": TOOL_VER,
            "model": f"rule-based vol-target(target={args.target_vol}, vol_lookback={args.vol_lookback}) "
                     f"× trend-filter(MA{args.trend_ma}); monthly rebalance; long-only cap {MAX_EXPOSURE}",
            "run_at": datetime.now().isoformat(), "constitution_ver": CONSTITUTION_VER,
            "configs": CONFIGS, "target_vol": args.target_vol, "vol_lookback": args.vol_lookback,
            "trend_ma": args.trend_ma, "max_exposure": MAX_EXPOSURE, "switch_cost": SWITCH_COST,
            "n_market_days": int(len(market_dates)),
            "market_span": [str(market_dates[0]), str(market_dates[-1])],
            "bear_episodes": [e[0] for e in BEAR_EPISODES],
            "comparable_to": "market_regime_timing_validation.py (same build_market_series + _series_stats + BEAR_EPISODES + SWITCH_COST)",
            "anti_leakage": ("exposure at month-end t uses ONLY data <= t (backward rolling vol/MA); applied from t+1 "
                             "(decision_pos < day). No future leakage. Rule-based (no training, no seed)."),
            "source_traceability": ("per CLAUDE.md §一.10 — all data from (b) DB query (TaiwanStockPriceAdj close); "
                                    "signals = real value -> math transform (log-return / rolling vol / MA ratio); "
                                    "no fabrication, no imputation (warmup-NaN decisions skipped)."),
            "smoke": args.smoke,
            "HONEST_CAVEAT": ("This is Phase-1 RULE-BASED protection (no predictor). It de-risks AFTER volatility rises "
                              "or trend breaks, so it LIMITS damage rather than avoiding the first day of a sudden "
                              "exogenous crash (e.g. a single-day −10% sector shock). Cash earns 0% (conservative). "
                              "Past drawdown reduction does NOT guarantee future results. Phase-2 will add a calibrated "
                              "regime probability as an ENHANCER on top of this floor and measure its INCREMENTAL value."),
        }
        op = (PROJECT_ROOT / args.output) if not os.path.isabs(args.output) else Path(args.output)
        op.parent.mkdir(parents=True, exist_ok=True)
        with open(op, "w") as f:
            json.dump(out, f, indent=2, default=str)
        logger.info(f"  Results persisted: {op}")


if __name__ == "__main__":
    main()
