"""
multi_cycle_tft_validation.py v0.2 (Temporal Fusion Transformer · Google/Lim et al. · Multi-Cycle Stock-Price Validation · per CLAUDE.md §一.11 三段式)
================================================================================
**最後更新日期**: 2026-06-02
**主權狀態**: TFT(TEMPORAL FUSION TRANSFORMER)4-HORIZON WALK-FORWARD VALIDATION + §14.7-CY HORIZON-DOCTRINE 第二族(neural)+ §14.7-DC v0.18 SOURCE-PURE UNIVERSE + §一.10 SOURCE-TRACEABLE(全 DB)+ §一.10 #3 MULTI-RUN + 共同比較基準(COMMON COMPARISON BASELINE)第一實作 + §一.11 三段式合規 + §14.7-DE §0.0-I panel-date helper 切換(2026-06-02)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:用 TFT(Temporal Fusion Transformer 時序融合) 序列模型,吃每支股票的「歷史價格序列」,預測未來報酬,評估「靠它選股能不能賺錢、準不準、可不可信」。

**它怎麼做(步驟)**:
1. 取 397 支「乾淨核心股」,載入每支的歷史價格序列(序列模型看時間走勢,非橫斷面特徵)。
2. 把 2013-05 ~ 2026-06 切成月度時間點(panel)。
3. **逐點往前走(walk-forward)**:每點只用那之前的序列訓練,預測之後報酬,不偷看未來(防洩漏)。
4. 依預測挑最看好的股票做多,跟全市場平均比,算賺賠。
5. 在 **4 種持有期**各做一遍:週(5 天)/ 月(20 天)/ 季(60 天)/ 年(252 天)。
6. 算成績:報酬率、Sharpe、勝率,加上**排序 IC、淨 Sharpe、機率校準覆蓋率**(此類序列模型用自有 calibration 導向 `aggregate_horizon`;§14.7-DF 註明 torch 暫不套樹模型 metric helper,各模型 rework 時再對齊共同欄位後與樹模型並比)。
7. 判定這模型在哪個週期「真的能賺錢且可信」。

**輸入**:資料庫(股價序列)。**輸出**:JSON(各週期成績)+ log。
**它不做的事**:不改資料庫(純讀取評估;§3.1 evaluation 角色)。
**為什麼需要它**:序列/基礎模型路線的實證裁判,與樹模型並列比較(共同欄位對齊後)。

## 📜 一、核心定義說明 (Core Definitions / The Constitution)

1. **[Sovereignty Declaration]** (v0.1, 憲法 §3.2 橫切分析模組): 本程式為 evaluation 後處理工具,非序列落地鏈模組。
   **治權邊界**:(a) §3.2 evaluation 模組;(b) 不涉五套禁令(§0.1/§0.2/§0.3);(c) T1-T3 不分層;
   (d) §8.5 anti-leakage:encoder 僅見 as_of 之前之 weekly bars,decoder forecast 未來 → 結構性無洩漏;
   (e) **不訓練 production model**(不寫 model_registry);(f) **read-only**(不改 feature_values / TaiwanStockPriceAdj / universe);
   (g) 唯一職責:TFT 4-horizon walk-forward 預測 + 共同比較基準 metrics + JSON 持久化。
2. **[Common Comparison Baseline]** (v0.1, CLAUDE.md 用戶 2026-05-29 directive): 本程式定義「**所有模型共用之比較基準**」第一實作 —
   universe v0.18(398 source-pure)× 95 monthly panels × 真實 forward log returns(TaiwanStockPriceAdj)× 4 horizons(5/20/60/252d)×
   top-20 equal-weight long × 0.6% cost × {Sharpe, Win, Eff-t, T_CZ-6 gate}。任何模型(tree / neural / 未來)套用同一 protocol +
   同一 realized targets → 精準度(precision)/ 信任度(trust)比較才 apples-to-apples。模型可用各自 natural representation
   (tree=38 cross-sectional features;TFT=每股最長 weekly price 序列),比較點在 OUTPUT 預測之品質,非 input。
3. **[Source Traceability]** (v0.1, CLAUDE.md §一.10): 全數據 (b) DB query(TaiwanStockPriceAdj close/Trading_money + core_universe_*)+
   (a) program output(本 JSON / log);**0 AI memory reuse**;weekly features 全為 price/volume 之 source-pure mathematical transform
   (log return / rolling std / ma-ratio / log dollar volume),**無 imputed / 無 hardcoded knowledge / 無 §一.13 第四類幻像值**。
4. **[Longest-Available Per-Stock History]** (v0.1, 用戶 2026-05-29 directive): 每股取其在 DB 之最長 daily 歷史(resample → weekly bars);
   不齊頭裁切;TimeSeriesDataSet group_ids=["stock_id"] + allow_missing_timesteps 容許各股不同起點。
5. **[Real Temporal Fusion Transformer]** (v0.1): 使用 pytorch_forecasting.TemporalFusionTransformer(Lim et al. 2019 / Google)—
   variable selection + LSTM encoder/decoder + interpretable multi-head attention + QuantileLoss;非 surrogate。
   單一 pooled panel model(非每股一個);target=weekly log return;decoder forecast 未來 weekly returns → cumulative-sum
   → 5/20/60/252-day forward-return scores(一個 model 服務 4 horizons,leakage-safe by construction)。
6. **[Precision / Trust / Profitability 三分]** (v0.1): 精準度(rank-IC / directional accuracy / RMSE / MAE / R²)、
   信任度(Eff-t significance / 多 seed 穩定度 per §一.10 #3 / P10-P90 calibration coverage)、
   賺錢能力(net-of-cost Sharpe / Eff-t / Win / annualized net return / T_CZ-6 gate)分開報告 → 回答「真的能賺錢嗎?」。
7. **[Zero Hardcoded Verdict]** (v0.1, 憲法 §5.6.3): significance 動態(abs(eff_t) > 1.997);T_CZ-6 gate(4.20/2.40/0.79)
   為 charter-mandated reference threshold(Tier 3 transparent disclosure,非 feature data,非硬編 verdict)。
8. **[§一.10 #3 Multi-Run]** (v0.1): stochastic(torch init / dropout / sgd)→ 須 ≥3 seeds {5422,7331,1009};
   single-run 不得作為 deterministic charter fact;median 為 inscription central estimate(由 _aggregate.py 跨 seed 聚合)。
9. **[Schema-Compatible Output]** (v0.1): per-horizon JSON keys 與 multi_cycle_validation.py 對齊(sharpe / win_rate /
   effective_t_stat / is_significant_p05 / annualized_simple_net / ...)+ precision/trust 數值 keys → _aggregate.py 可直接 roll up。
10. **[Historical Reference Authority]** (v0.1): TOOL_VER = "v0.1" 為記述性快照,非權威來源。
11. **[Idempotency]** (v0.1): 不寫 DB(--output 寫 JSON);可重跑;refit cadence / TFT size / epochs 全可配置 → 跑期可控。

## 📊 二、全量功能群矩陣 (The Ultimate Functional Group Matrix)

### Group A. Common Comparison Baseline(共同比較基準 — 與全模型一致)

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| A.1 Universe | get_universe() → 最新 committed snapshot core_tier='core_universe'(398 v0.18)| §14.7-DC v0.18 |
| A.2 Panels | get_panel_dates() → 95 monthly mid-month(2018-06-15~2026-04-15)| 與 baseline 同 grid |
| A.3 Forward returns | load_forward_returns() → 真實 log return(TaiwanStockPriceAdj)| §一.10 (b) DB |
| A.4 Portfolio | top-20 equal-weight long(np.argsort[-20:])| 與 baseline 同 |
| A.5 Profitability | sharpe / win / mdd / eff_t / annualized_net / T_CZ-6 | §14.7-CY / §14.7-CZ |

### Group B. TFT Model(每股最長 weekly 序列 → forecast)

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| B.1 Price series | load_price_series() → daily close/money → weekly bars + source-pure features | §一.10 / §一.13 |
| B.2 Dataset | build_timeseries_dataset() → TimeSeriesDataSet(group=stock_id)| pf 1.7 |
| B.3 Train | train_tft() → TemporalFusionTransformer + QuantileLoss + EarlyStopping(CPU)| Lim et al. |
| B.4 Predict | predict_forward() → decode 未來 weekly returns → cumsum → 4-horizon scores | §8.5 leakage-safe |
| B.5 Walk-forward refit | --refit-every N panels(default annual≈12)| expanding window |

### Group C. Precision / Trust(標準化 block — 全模型共用定義)

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| C.1 Precision | rank_ic / directional_accuracy / rmse / mae / r2(pred vs 真實 realized)| 共同基準 |
| C.2 Trust | effective_t_stat / is_significant_p05 / calibration_p10_p90_coverage | 共同基準 |
| C.3 Multi-seed | --seed {5422,7331,1009} → _aggregate.py min/median/max/mean | §一.10 #3 |

### Group D. Persistence

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| D.1 JSON | --output reports/tft_v0/<...>.json(schema-compatible + precision/trust)| §一.10 / §二.4 |
| D.2 stdout | cross-cycle comparison matrix + precision/trust 摘要 | §一.12 進度 |

### 對齊憲章 §二 維運矩陣

| 場景 | 命令 |
| :--- | :--- |
| Smoke(plumbing 驗證)| `python scripts/evaluation/multi_cycle_tft_validation.py --smoke --output reports/tft_v0/_smoke.json` |
| 完整單 seed | `... --seed 5422 --output reports/tft_v0/tft_s5422.json` |
| 3-run 教義全合規 | 對 {5422,7331,1009} 各跑一次 → _aggregate.py |

### 不提供之旗標 (Intentionally Omitted)

- `--commit`:本工具不寫 evaluation_log / model_registry(屬 model_trainer 治權;§3.2 橫切只讀)。
- `--cost-per-rebal`:0.6% standard per §14.7-CY T_CY-5(與 baseline 對齊,不可變更否則破壞比較基準)。
- `--horizons`:固定 5/20/60/252(共同比較基準定義之一部分,不可變更)。

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.2 | 2026-06-02 | Codex | **§0.0-I panel-date helper 切換(§14.7-DE)**:panel 窗改用 `get_canonical_panel_dates()`(§14.7-DE 單一引用源,移除寫死 date(2018,6,15) → 資料驅動 157 panels)+ 修正 stale log 字串(2018-2026/95→data-driven)。metric 仍用自有 `aggregate_horizon`(calibration 框架,§14.7-DF deferred,各模型 rework 時對齊)。**未改模型、未 retrain**。 | **ACTIVE** |
| v0.1 | 2026-05-29 | Claude | **首版**:TFT(Temporal Fusion Transformer / Google)multi-cycle 股價預測驗證 + 共同比較基準(common comparison baseline)第一實作。每股最長 weekly 序列 → pooled TFT(QuantileLoss)→ 4-horizon cumsum scores → 與 baseline 同 universe(v0.18/398)/ panels(95)/ forward returns(TaiwanStockPriceAdj)/ portfolio(top-20)/ cost(0.6%)/ gate(T_CZ-6)。新增標準化 precision(rank-IC / dir-acc / RMSE / MAE / R²)+ trust(Eff-t / 多 seed / P10-P90 calibration)block。§一.10 全 DB source-traceable;§一.10 #3 multi-seed;§8.5 leakage-safe by encoder/decoder split;§一.11 三段式合規。 | SUPERSEDED |
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")  # NumPy 2.x vs torch-compiled-1.x non-fatal warning + pf deprecations
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import sys, argparse, math, json, logging, time
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

_base_dir = Path(__file__).resolve().parent.parent  # scripts/
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

import numpy as np
import pandas as pd
from core.db_utils import get_db_conn, get_canonical_panel_dates, summarize_horizon_metrics

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

CONSTITUTION_VER = "v6.1.0"
TOOL_VER = "v0.2"

# ── Common Comparison Baseline constants (FIXED — 不可變更,否則破壞跨模型比較) ──
HORIZONS = [("weekly", 5), ("monthly", 20), ("quarterly", 60), ("annual", 252)]
N_TOP = 20                 # top-k long portfolio (baseline 對齊)
COST_PER_REBAL = 0.006     # TW broker round-trip (§14.7-CY T_CY-5)
PANEL_SPACING = 30         # monthly grid (for overlap-corrected n_eff)
GATE = {"effective_t_stat": 4.20, "sharpe": 2.40, "win_rate": 0.79}  # §14.7-CZ T_CZ-6 annual
CRIT_T = 1.997             # p<0.05 large-df

# ── TFT / weekly-bar constants ──
BAR_DAYS = 5               # weekly bar = every 5 trading days
ENC_WEEKS = 104            # encoder lookback (~2 yr)
PRED_WEEKS = 52            # decoder horizon (covers annual 252d ≈ 50 wk)
HORIZON_STEPS = {5: 1, 20: 4, 60: 12, 252: 50}  # trading days → weekly decoder step index
P10_IDX, P50_IDX, P90_IDX = 1, 3, 5  # QuantileLoss [.02,.1,.25,.5,.75,.9,.98]


# ════════════════════════════════════════════════════════════════════════════
# Group A — Common Comparison Baseline DB loaders
# ════════════════════════════════════════════════════════════════════════════
def get_universe(cur):
    cur.execute("""SELECT m.stock_id FROM core_universe_membership m
                   JOIN core_universe_snapshot s ON s.snapshot_id=m.snapshot_id
                   WHERE s.status='committed' AND m.core_tier='core_universe'
                   AND s.snapshot_id=(SELECT snapshot_id FROM core_universe_snapshot
                                      WHERE status='committed' ORDER BY created_at DESC LIMIT 1)""")
    return sorted({r[0] for r in cur.fetchall()})


def load_forward_returns(cur, as_of, horizon_days):
    """真實 forward log return at horizon (per (b) DB query) — 與 baseline 完全相同."""
    cur.execute("""SELECT MIN(date) FROM "TaiwanStockPriceAdj"
                   WHERE date >= (%s::date + INTERVAL '%s days')
                     AND stock_id ~ '^[0-9]' AND date <= (%s::date + INTERVAL '%s days')""",
                (str(as_of), horizon_days, str(as_of), horizon_days + 14))
    r = cur.fetchone()
    label_date = r[0] if r and r[0] else None
    if not label_date:
        return {}, None
    cur.execute("""WITH t0 AS (SELECT stock_id, close FROM "TaiwanStockPriceAdj" WHERE date=%s AND close>0),
                        t1 AS (SELECT stock_id, close FROM "TaiwanStockPriceAdj" WHERE date=%s AND close>0)
                   SELECT t0.stock_id, LN(t1.close::numeric/t0.close::numeric)
                   FROM t0 JOIN t1 ON t0.stock_id=t1.stock_id""", (str(as_of), str(label_date)))
    return {sid: float(v) for sid, v in cur.fetchall()}, label_date


# ════════════════════════════════════════════════════════════════════════════
# Group B — TFT price-series construction
# ════════════════════════════════════════════════════════════════════════════
def load_price_series(cur, universe, max_stocks=None):
    """每股最長 daily 歷史 → weekly bars + source-pure features → long DataFrame.

    Columns: stock_id, time_idx, bar_date, y(weekly log ret), vol_4w, ma_ratio, log_dollar_vol.
    全為 TaiwanStockPriceAdj close/Trading_money 之 mathematical transform(§一.10 / §一.13 source-pure)."""
    if max_stocks:
        universe = universe[:max_stocks]
    rows = []
    for sid in universe:
        cur.execute("""SELECT date, close, "Trading_money" FROM "TaiwanStockPriceAdj"
                       WHERE stock_id=%s AND close>0 ORDER BY date""", (sid,))
        rec = cur.fetchall()
        if len(rec) < (ENC_WEEKS + PRED_WEEKS + 20) * BAR_DAYS:
            continue  # 不足以構成 encoder+decoder 之最短序列
        dts = [r[0] for r in rec]
        close = np.array([float(r[1]) for r in rec])
        money = np.array([float(r[2]) if r[2] is not None else 0.0 for r in rec])
        # weekly bars: 每 BAR_DAYS 取 block 末日 close + block 內 money 加總
        bdate, bclose, bmoney = [], [], []
        for k in range(BAR_DAYS - 1, len(close), BAR_DAYS):
            bdate.append(dts[k]); bclose.append(close[k]); bmoney.append(money[k - BAR_DAYS + 1:k + 1].sum())
        bclose = np.array(bclose); bmoney = np.array(bmoney)
        if len(bclose) < ENC_WEEKS + PRED_WEEKS + 5:
            continue
        y = np.diff(np.log(bclose))                       # weekly log return
        s = pd.Series(y)
        vol = s.rolling(4).std().to_numpy()               # 4-week realized vol
        ma13 = pd.Series(bclose).rolling(13).mean().to_numpy()
        ma_ratio = bclose / ma13 - 1.0                    # close vs 13-wk MA
        ldv = np.log(bmoney + 1.0)                        # log dollar volume
        # align (y has len-1; index i corresponds to bar i+1)
        for i in range(1, len(bclose)):
            j = i - 1
            if not np.isfinite(y[j]) or not np.isfinite(vol[j]) or not np.isfinite(ma_ratio[i]):
                continue
            rows.append((sid, bdate[i], float(y[j]), float(vol[j]), float(ma_ratio[i]), float(ldv[i])))
    df = pd.DataFrame(rows, columns=["stock_id", "bar_date", "y", "vol_4w", "ma_ratio", "log_dollar_vol"])
    # per-stock consecutive time_idx
    df = df.sort_values(["stock_id", "bar_date"]).reset_index(drop=True)
    df["time_idx"] = df.groupby("stock_id").cumcount().astype(int)
    return df


def build_timeseries_dataset(df, max_time_idx_per_stock=None):
    """Training TimeSeriesDataSet up to cutoff (per-stock time_idx ≤ cutoff)."""
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder
    work = df
    if max_time_idx_per_stock is not None:
        work = df[df["time_idx"] <= df["stock_id"].map(max_time_idx_per_stock)].copy()
    # keep stocks with enough length for ≥1 encoder+decoder window
    lens = work.groupby("stock_id")["time_idx"].count()
    keep = lens[lens >= ENC_WEEKS + PRED_WEEKS + 1].index
    work = work[work["stock_id"].isin(keep)].copy()
    if work.empty:
        return None
    ds = TimeSeriesDataSet(
        work,
        time_idx="time_idx", target="y", group_ids=["stock_id"],
        max_encoder_length=ENC_WEEKS, min_encoder_length=ENC_WEEKS // 2,
        max_prediction_length=PRED_WEEKS, min_prediction_length=1,
        static_categoricals=["stock_id"],
        time_varying_unknown_reals=["y", "vol_4w", "ma_ratio", "log_dollar_vol"],
        target_normalizer=GroupNormalizer(groups=["stock_id"]),
        categorical_encoders={"stock_id": NaNLabelEncoder(add_nan=True)},
        add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    return ds


def train_tft(training_ds, seed, epochs, lr, hidden, batch_size):
    import torch, lightning.pytorch as pl
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    train_dl = training_ds.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    tft = TemporalFusionTransformer.from_dataset(
        training_ds, learning_rate=lr, hidden_size=hidden, attention_head_size=1,
        dropout=0.1, hidden_continuous_size=min(8, hidden), loss=QuantileLoss(),
        log_interval=-1, optimizer="adam",
    )
    trainer = pl.Trainer(
        max_epochs=epochs, accelerator="cpu", devices=1, gradient_clip_val=0.1,
        enable_progress_bar=False, enable_checkpointing=False, logger=False,
        enable_model_summary=False,
    )
    trainer.fit(tft, train_dl)
    return tft


def predict_forward(tft, training_ds, df, cutoff_idx_per_stock, batch_size):
    """For each stock, encoder window ending at its cutoff bar → decode PRED_WEEKS →
    cumulative-sum median weekly returns → forward-return score per horizon.
    Returns: dict horizon_days → {stock_id: pred_fwd_log_ret}, plus weekly P10/P50/P90 for calibration."""
    from pytorch_forecasting import TimeSeriesDataSet
    # build per-stock slice: encoder rows [cut-ENC+1 .. cut] (real) + future placeholder [cut+1 .. cut+PRED]
    slices = []
    last_feat = {}
    for sid, g in df.groupby("stock_id"):
        cut = cutoff_idx_per_stock.get(sid)
        if cut is None:
            continue
        enc = g[(g.time_idx <= cut) & (g.time_idx > cut - ENC_WEEKS)]
        if len(enc) < ENC_WEEKS:
            continue
        slices.append(enc)
        lf = enc.iloc[-1]
        last_feat[sid] = lf
        fut = pd.DataFrame({
            "stock_id": sid,
            "time_idx": np.arange(cut + 1, cut + 1 + PRED_WEEKS),
            "bar_date": [lf["bar_date"]] * PRED_WEEKS,
            "y": np.nan, "vol_4w": float(lf["vol_4w"]),
            "ma_ratio": float(lf["ma_ratio"]), "log_dollar_vol": float(lf["log_dollar_vol"]),
        })
        slices.append(fut)
    if not slices:
        return {h: {} for _, h in HORIZONS}, {}
    pred_df = pd.concat(slices, ignore_index=True).sort_values(["stock_id", "time_idx"]).reset_index(drop=True)
    pred_df["y"] = pred_df["y"].fillna(0.0)  # decoder target placeholder; not used as input
    pred_ds = TimeSeriesDataSet.from_dataset(training_ds, pred_df, predict=True, stop_randomization=True)
    pred_dl = pred_ds.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    out = tft.predict(pred_dl, mode="quantiles", return_index=True)
    output = out.output if hasattr(out, "output") else out[0]
    index = out.index if hasattr(out, "index") else out[1]
    quant = np.asarray(output.cpu() if hasattr(output, "cpu") else output)  # [n, PRED, n_quantiles]
    idx_df = index.reset_index(drop=True)
    fwd = {h: {} for _, h in HORIZONS}
    weekly_q = {}
    for r in range(len(idx_df)):
        sid = idx_df.iloc[r]["stock_id"]
        med = quant[r, :, P50_IDX]            # median weekly log returns over decoder
        cum = np.cumsum(med)
        for _, h in HORIZONS:
            step = HORIZON_STEPS[h]
            if step <= len(cum):
                fwd[h][sid] = float(cum[step - 1])
        weekly_q[sid] = (float(quant[r, 0, P10_IDX]), float(quant[r, 0, P50_IDX]), float(quant[r, 0, P90_IDX]))
    return fwd, weekly_q


# ════════════════════════════════════════════════════════════════════════════
# Group A/C — metrics (portfolio + precision + trust)
# ════════════════════════════════════════════════════════════════════════════
def spearman_ic(pred, y):
    pred = np.asarray(pred); y = np.asarray(y)
    if len(pred) < 3:
        return 0.0
    rp = pred.argsort().argsort().astype(float); ry = y.argsort().argsort().astype(float)
    if np.std(rp) < 1e-10 or np.std(ry) < 1e-10:
        return 0.0
    return float(np.corrcoef(rp, ry)[0, 1])


def aggregate_horizon(label, horizon_days, panel_top_rets, panel_univ_rets,
                      panel_ics, panel_diracc, all_pred, all_real, calib_cov):
    """共同比較基準 metric 計算(與 multi_cycle_validation.py 完全相同之 profitability 數學)+ precision/trust."""
    n = len(panel_top_rets)
    if n == 0:
        return None
    pr = np.array(panel_top_rets); ur = np.array(panel_univ_rets)
    mean_ret = float(np.mean(pr)); std_ret = float(np.std(pr, ddof=1)) if n > 1 else 0.0
    sharpe = mean_ret / std_ret * math.sqrt(12) if std_ret > 0 else 0.0
    win_rate = float(np.mean(pr > 0))
    alphas = pr - ur
    mean_alpha = float(np.mean(alphas)); std_alpha = float(np.std(alphas, ddof=1)) if n > 1 else 0.0
    ir = mean_alpha / std_alpha * math.sqrt(12) if std_alpha > 0 else 0.0
    t_stat = mean_alpha / (std_alpha / math.sqrt(n)) if std_alpha > 0 else 0.0
    running = peak = mdd = 0.0
    for r in pr:
        running += r; peak = max(peak, running); mdd = max(mdd, peak - running)
    rebals_per_year = 252.0 / horizon_days
    ann_log_gross = mean_ret * rebals_per_year
    ann_cost = COST_PER_REBAL * rebals_per_year
    ann_log_net = ann_log_gross - ann_cost
    ann_simple_net = math.exp(ann_log_net) - 1
    net_rets = pr - COST_PER_REBAL
    net_sharpe = float(np.mean(net_rets)) / (np.std(net_rets, ddof=1) * math.sqrt(1)) * math.sqrt(12) \
        if n > 1 and np.std(net_rets, ddof=1) > 0 else 0.0
    if horizon_days <= PANEL_SPACING:
        n_eff = float(n); overlap_pct = 0.0
    else:
        n_eff = n * (PANEL_SPACING / horizon_days)
        overlap_pct = (horizon_days - PANEL_SPACING) / horizon_days * 100
    eff_t = t_stat * math.sqrt(n_eff / n) if n > 0 else 0.0
    # precision block
    ap = np.array(all_pred); ar = np.array(all_real)
    rmse = float(np.sqrt(np.mean((ap - ar) ** 2))) if len(ap) else None
    mae = float(np.mean(np.abs(ap - ar))) if len(ap) else None
    ss_res = float(np.sum((ar - ap) ** 2)); ss_tot = float(np.sum((ar - np.mean(ar)) ** 2)) if len(ar) else 0.0
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else None
    return {
        "horizon": label, "horizon_days": horizon_days, "n_panels": n,
        "n_effective": n_eff, "overlap_pct": overlap_pct, "rebals_per_year": rebals_per_year,
        "mean_ret_per_panel": mean_ret, "std_ret_per_panel": std_ret,
        "sharpe": sharpe, "win_rate": win_rate, "mdd_per_panel": mdd,
        "mean_alpha_per_panel": mean_alpha, "ir": ir, "t_stat": t_stat,
        "effective_t_stat": eff_t, "is_significant_p05": bool(abs(eff_t) > CRIT_T),
        "annualized_log_gross": ann_log_gross, "annualized_simple_net": ann_simple_net,
        "annual_cost_drag_log": ann_cost, "net_sharpe_per_panel": net_sharpe,
        # ── standardized PRECISION block ──
        "rank_ic_mean": float(np.mean(panel_ics)) if panel_ics else None,
        "rank_ic_std": float(np.std(panel_ics, ddof=1)) if len(panel_ics) > 1 else None,
        "rank_ic_ir": (float(np.mean(panel_ics)) / float(np.std(panel_ics, ddof=1)) * math.sqrt(12))
                      if len(panel_ics) > 1 and np.std(panel_ics, ddof=1) > 0 else None,
        "directional_accuracy": float(np.mean(panel_diracc)) if panel_diracc else None,
        "rmse": rmse, "mae": mae, "r2": r2,
        # ── standardized TRUST block ──
        "calibration_p10_p90_coverage": calib_cov,  # weekly-level proper coverage (None for non-weekly)
    }


# ════════════════════════════════════════════════════════════════════════════
# Orchestration
# ════════════════════════════════════════════════════════════════════════════
def run(args):
    conn = get_db_conn()
    cur = conn.cursor()
    universe = get_universe(cur)
    logger.info(f"Universe: {len(universe)} stocks (v0.18 source-pure)")
    panels = [d for _fsid, d in get_canonical_panel_dates("feature_set_v0.5")]  # §14.7-DE / §0.0-I 單一引用源
    if args.max_panels:
        panels = panels[:args.max_panels]
    logger.info(f"Panels: {len(panels)} monthly as_of dates")

    t0 = time.monotonic()
    df = load_price_series(cur, universe, max_stocks=args.max_stocks)
    logger.info(f"Price series: {df['stock_id'].nunique()} stocks × weekly bars = {len(df)} rows "
                f"(load {time.monotonic()-t0:.1f}s)")

    # bar_date → per-stock time_idx lookup (for mapping as_of → cutoff bar)
    df_sorted = df.sort_values(["stock_id", "time_idx"])
    per_stock = {sid: g[["time_idx", "bar_date"]].to_numpy() for sid, g in df_sorted.groupby("stock_id")}
    per_stock_y = {sid: dict(zip(g["time_idx"].tolist(), g["y"].tolist())) for sid, g in df_sorted.groupby("stock_id")}

    def cutoff_for(as_of):
        cut = {}
        for sid, arr in per_stock.items():
            mask = arr[:, 1] <= as_of
            if mask.any():
                cut[sid] = int(arr[mask][-1, 0])
        return cut

    # refit schedule
    refit_every = args.refit_every
    panel_idx_list = list(range(len(panels)))
    refit_points = panel_idx_list[::refit_every] if refit_every > 0 else [0]
    logger.info(f"TFT refit at panel indices {refit_points} (every {refit_every}); "
                f"enc={ENC_WEEKS}wk pred={PRED_WEEKS}wk hidden={args.hidden} epochs={args.epochs} seed={args.seed}")

    # accumulators per horizon
    acc = {h: {"top": [], "univ": [], "ic": [], "diracc": [], "pred": [], "real": [], "pa": []} for _, h in HORIZONS}
    calib_hits, calib_tot = 0, 0
    current_tft = None
    current_ds = None
    n_pred_panels = 0

    for pi, as_of in enumerate(panels):
        if pi in refit_points or current_tft is None:
            cutoff = cutoff_for(as_of)
            if not cutoff:
                continue
            tr0 = time.monotonic()
            training_ds = build_timeseries_dataset(df, max_time_idx_per_stock=cutoff)
            if training_ds is None:
                logger.warning(f"  panel {pi} {as_of}: empty training set, skip refit")
                continue
            current_tft = train_tft(training_ds, args.seed, args.epochs, args.lr, args.hidden, args.batch_size)
            current_ds = training_ds
            logger.info(f"  [refit @ panel {pi} {as_of}] trained in {time.monotonic()-tr0:.1f}s")

        cutoff = cutoff_for(as_of)
        fwd, weekly_q = predict_forward(current_tft, current_ds, df, cutoff, args.batch_size)
        # realized forward returns per horizon (共同基準 truth)
        for label, h in HORIZONS:
            real, _ = load_forward_returns(cur, as_of, h)
            preds = fwd.get(h, {})
            common = [s for s in preds if s in real]
            if len(common) < N_TOP + 5:
                continue
            p = np.array([preds[s] for s in common]); y = np.array([real[s] for s in common])
            top_idx = np.argsort(p)[-N_TOP:]
            acc[h]["top"].append(float(np.mean(y[top_idx])))
            acc[h]["univ"].append(float(np.mean(y)))
            acc[h]["ic"].append(spearman_ic(p, y))
            acc[h]["diracc"].append(float(np.mean(np.sign(p) == np.sign(y))))
            acc[h]["pred"].extend(p.tolist()); acc[h]["real"].extend(y.tolist())
            acc[h]["pa"].append((p, y))  # §14.7-DF: per-panel (pred,actual) → 共用 metric helper
        # weekly calibration: realized next-week y vs predicted [P10,P90]
        for sid, (p10, p50, p90) in weekly_q.items():
            c = cutoff.get(sid)
            if c is None:
                continue
            ry = per_stock_y.get(sid, {}).get(c + 1)
            if ry is not None:
                calib_tot += 1
                if p10 <= ry <= p90:
                    calib_hits += 1
        n_pred_panels += 1
        if n_pred_panels % 6 == 0:
            logger.info(f"  predicted {n_pred_panels} panels (latest as_of={as_of}, elapsed {time.monotonic()-t0:.0f}s)")

    calib_cov = (calib_hits / calib_tot) if calib_tot else None
    results = {}
    for label, h in HORIZONS:
        a = acc[h]
        cc = calib_cov if h == 5 else None
        r = aggregate_horizon(label, h, a["top"], a["univ"], a["ic"], a["diracc"], a["pred"], a["real"], cc)
        if r:
            # §14.7-DF 補共用 canonical metric keys(單一計算源 summarize_horizon_metrics)
            # → 與 6 樹 + FT-Transformer 同 metric 碼可比;tft 自有 calibration/r2 等 extra 保留
            s = summarize_horizon_metrics(label, h, a["pa"])
            if s:
                r.update(s)
            results[label] = r

    conn.close()
    return results, universe, panels, df


def main():
    ap = argparse.ArgumentParser(description=f"Multi-Cycle TFT Validation {TOOL_VER}")
    ap.add_argument("--seed", type=int, default=5422)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--refit-every", type=int, default=12, help="refit TFT every N monthly panels (default annual≈12)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-stocks", type=int, default=None, help="limit universe for smoke")
    ap.add_argument("--max-panels", type=int, default=None, help="limit #panels for smoke")
    ap.add_argument("--smoke", action="store_true", help="tiny config: 40 stocks, 6 panels, 1 refit, 2 epochs")
    args = ap.parse_args()
    if args.smoke:
        args.max_stocks = args.max_stocks or 40
        args.max_panels = args.max_panels or 6
        args.epochs = 2
        args.refit_every = 0  # single refit at panel 0
        args.batch_size = 64

    logger.info("=" * 100)
    logger.info(f"Multi-Cycle TFT Validation {TOOL_VER} (Temporal Fusion Transformer / Google)")
    logger.info(f"  COMMON COMPARISON BASELINE: source-pure universe (data-driven §14.7-DE) × 4 horizons × top-{N_TOP} × cost {COST_PER_REBAL}")
    logger.info(f"  seed={args.seed} smoke={args.smoke} max_stocks={args.max_stocks}")
    logger.info("=" * 100)

    t_global = time.monotonic()
    results, universe, panels, df = run(args)

    # cross-cycle matrix
    logger.info(f"\n{'='*100}\nCross-Cycle Comparison Matrix (TFT)\n{'='*100}")
    logger.info(f"  {'Horizon':10} {'N':>4} {'Eff t':>7} {'Sig':>4} {'Sharpe':>7} {'Win':>6} {'NetAnn':>9} "
                f"{'rankIC':>7} {'DirAcc':>7}")
    for label, r in results.items():
        sig = "✅" if r["is_significant_p05"] else "❌"
        ic = r.get("rank_ic_mean"); da = r.get("directional_accuracy")
        logger.info(f"  {label:10} {r['n_panels']:>4} {r['effective_t_stat']:>+7.3f} {sig:>4} "
                    f"{r['sharpe']:>+7.3f} {r['win_rate']*100:>5.1f}% {r['annualized_simple_net']*100:>+8.2f}% "
                    f"{(ic if ic is not None else float('nan')):>+7.3f} "
                    f"{(da*100 if da is not None else float('nan')):>6.1f}%")
    ann = results.get("annual")
    if ann:
        gv = {k: (ann.get(k), thr, (ann.get(k) is not None and ann.get(k) >= thr)) for k, thr in GATE.items()}
        verdict = all(v[2] for v in gv.values())
        logger.info(f"\n  T_CZ-6 annual gate: {'✅ PASS' if verdict else '❌ FAIL'} — "
                    + " | ".join(f"{k}={ann.get(k)}(≥{thr})" for k, thr in GATE.items()))
    logger.info(f"\n  Total elapsed: {time.monotonic()-t_global:.1f}s")

    if args.output:
        out = {label: {k: v for k, v in r.items() if k != "panel_records"} for label, r in results.items()}
        out["_meta"] = {
            "tool": "multi_cycle_tft_validation.py", "tool_ver": TOOL_VER,
            "model": "TemporalFusionTransformer (pytorch_forecasting)",
            "run_at": datetime.now().isoformat(), "constitution_ver": CONSTITUTION_VER,
            "seed": args.seed, "horizons": [h for _, h in HORIZONS],
            "n_universe": len(universe), "n_stocks_with_series": int(df["stock_id"].nunique()),
            "n_panels_input": len(panels), "enc_weeks": ENC_WEEKS, "pred_weeks": PRED_WEEKS,
            "refit_every": args.refit_every, "epochs": args.epochs, "hidden": args.hidden,
            "smoke": args.smoke,
            "common_baseline": f"universe v0.18 × {len(panels)} panels × {HORIZONS} × top-{N_TOP} × cost {COST_PER_REBAL} × T_CZ-6",
            "source_traceability": "per CLAUDE.md §一.10 — all data from (b) DB query TaiwanStockPriceAdj/core_universe_*",
        }
        op = Path(_base_dir).parent / args.output
        op.parent.mkdir(parents=True, exist_ok=True)
        with open(op, "w") as f:
            json.dump(out, f, indent=2, default=str)
        logger.info(f"  Results persisted: {op}")


if __name__ == "__main__":
    main()
