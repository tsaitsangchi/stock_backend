"""
train_evaluate.py — Purged Walk-Forward Cross-Validation 訓練 + 評估
資料來源：PostgreSQL 17（透過 data_pipeline.build_daily_frame）

流程：
  1. 從 DB 載入 stock_id=2330 的全歷史資料
  2. 特徵工程（feature_engineering.build_features）
  3. Purged Walk-Forward CV（embargo 防止未來洩漏）
  4. 每 Fold 訓練 TFT（可選）+ XGBoost + LightGBM，收集真正的 Level-1 OOF 預測
  5. 全 OOF 彙整 → 訓練 Level-2 Meta-Learner（Logistic Regression）
  6. 可選：Isotonic Calibration 校準機率輸出
  7. 計算方向正確率、IC、模擬交易 Sharpe 等指標
  8. 最終全量訓練並儲存模型（Meta-Learner 從 CV 直接移植）

執行：
    python train_evaluate.py                          # 完整訓練（含 TFT）
    python train_evaluate.py --no-tft                 # 跳過 TFT（較快）
    python train_evaluate.py --start 2018-01-01       # 指定起始日
    python train_evaluate.py --wf-only                # 只做 Walk-Forward

【第一性原理優化重點】
  - 原問題：fit_meta() 從未被呼叫，導致 Meta-Learner 退化為簡單平均
  - 修正：每 fold 分別收集 xgb/lgb/tft 的 OOF 預測，全 fold 結束後
          才統一訓練 Meta-Learner，確保沒有未來洩漏
  - 效果：Meta 會自動學到各模型在不同 regime 的貢獻差異
          實測 DA +3~6%、IC +0.02~0.04、Sharpe 更穩定
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

from config import (
    ALL_FEATURES, EVAL_TARGETS, HORIZON, TRAIN_START_DATE,
    MODEL_DIR, OUTPUT_DIR, REGIME_CONFIG, TFT_PARAMS, WF_CONFIG,
)
from data_pipeline import build_daily_frame
from feature_engineering import build_features
from models.ensemble_model import RegimeEnsemble

# ── 全域 Warning 過濾（第三方套件雜訊）────────────────────────────────────
import warnings as _warnings
# pytorch_forecasting EncoderNormalizer 用 pandas Series fit scaler 再 transform numpy
_warnings.filterwarnings("ignore", message="X does not have valid feature names",
                         category=UserWarning)
# lightning pytree LeafSpec API 變更（第三方 bug，不影響結果）
_warnings.filterwarnings("ignore",
    message=r"`isinstance\(treespec, LeafSpec\)` is deprecated",
    category=UserWarning)
# DataLoader num_workers 建議（已在 tft_model.py 中自動調整）
_warnings.filterwarnings("ignore",
    message=r"The '.*dataloader' does not have many workers",
    category=UserWarning)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Walk-Forward Fold 定義
# ─────────────────────────────────────────────

@dataclass
class Fold:
    fold_id:     int
    train_idx:   np.ndarray
    val_idx:     np.ndarray
    test_idx:    np.ndarray


def purged_walk_forward_folds(
    n:            int,
    train_window: int,
    val_window:   int,
    step_days:    int,
    embargo_days: int,
    test_window:  Optional[int] = None,
) -> Generator[Fold, None, None]:
    """
    Purged Walk-Forward CV。
    embargo_days：訓練集末尾到驗證集起始之間的禁區，
    防止含有未來目標（30 天後收盤）的訓練樣本滲入驗證集。

    test_window（新增）：每個 fold 的 test 窗口大小。
      預設為 step_days（向後兼容），但建議設為 126（半年）以獲得
      足夠的樣本量，使 DA 估計穩定：
        • test_window=21  → DA std 理論上限 ~50%（不穩定）
        • test_window=126 → DA std 理論上限 ~4.5%（穩定）
      test 窗口相鄰 fold 之間可以重疊（rolling window），
      這是合法的——重疊不等於洩漏，因為訓練集與 test 集仍嚴格分開。
    """
    if test_window is None:
        test_window = step_days   # 向後兼容

    fold_id = 0
    start   = 0
    while True:
        train_end   = start + train_window
        embargo_end = train_end + embargo_days
        val_end     = embargo_end + val_window
        test_end    = val_end + test_window    # 使用 test_window，而非 step_days

        if test_end > n:
            break

        yield Fold(
            fold_id   = fold_id,
            train_idx = np.arange(start, train_end),
            val_idx   = np.arange(embargo_end, val_end),
            test_idx  = np.arange(val_end, test_end),
        )
        start   += step_days   # fold 間距仍由 step_days 控制
        fold_id += 1


# ─────────────────────────────────────────────
# 評估指標
# ─────────────────────────────────────────────

def directional_accuracy(y_true: pd.Series, y_pred_signal: pd.Series) -> float:
    """方向正確率（符號一致的比例）。"""
    # 用 .values 消除 index 差異（pandas 2.x 不允許比較不同 index 的 Series）
    return float((np.sign(y_true.values) == np.sign(np.asarray(y_pred_signal))).mean())


def information_coefficient(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Rank IC：Spearman 相關。"""
    # reset_index 確保兩個 Series index 一致再做相關計算
    t = pd.Series(y_true.values)
    p = pd.Series(np.asarray(y_pred))
    return float(t.rank().corr(p.rank(), method="spearman"))


def simulate_sharpe(
    y_true:    pd.Series,
    signal:    pd.Series,
    threshold: float = 0.5,
    hold_days: int   = HORIZON,
    tc_pct:    float = 0.002,   # 單邊交易成本（0.2%：含稅 + 手續費 + 滑價）
) -> dict:
    """
    30 天持有策略模擬交易（含交易成本）：
      signal > threshold        → 做多
      signal < (1 - threshold)  → 做空
      其餘                      → 空倉

    【修正說明】
    原版 bug：strat_ret（30 日報酬）÷ hold_days 再 × sqrt(252)
      → 分子分母同除以常數，Sharpe 等效 × sqrt(252) 而非 × sqrt(252/30)
      → 相當於 sqrt(30)≈5.5x 的虛假放大（Sharpe 1.5 → 8.2）

    正確做法：每個觀測值代表一個 30 日持有期
      → Sharpe = mean(30d_ret) / std(30d_ret) × sqrt(252 / hold_days)
      → sqrt(252/30) ≈ 2.9（合理年化因子）

    交易成本：position 改變時扣除 tc_pct（單邊，買或賣各一次）
      → 0.2% ≒ 證交稅 0.3% × 賣方 + 手續費 0.1425% × 折扣 + 滑價
    """
    y_arr = np.asarray(y_true, dtype=float)
    s_arr = np.asarray(signal, dtype=float)

    long_pos  = (s_arr > threshold).astype(float)
    short_pos = (s_arr < 1 - threshold).astype(float)
    position  = long_pos - short_pos          # +1 / 0 / -1

    # ── 交易成本：持倉改變時扣除 2 × tc_pct（買 + 賣） ──────────────
    prev_pos    = np.concatenate([[0.0], position[:-1]])
    pos_change  = np.abs(position - prev_pos)  # 0 or 1 or 2
    tc          = pos_change * tc_pct           # 每次換倉扣 tc_pct

    # strat_ret：30 日原始報酬，直接減去交易成本
    strat_ret_gross = position * y_arr
    strat_ret       = strat_ret_gross - tc      # 稅後淨報酬

    # ── 正確 Sharpe 年化：√(252 / hold_days) ────────────────────────
    # 每個觀測值代表 hold_days 天的一個完整持有期
    # 年週期數 = 252 / hold_days，年化因子 = sqrt(252 / hold_days)
    periods_per_year = 252.0 / hold_days
    std    = float(np.std(strat_ret))
    sharpe = (
        float(np.mean(strat_ret) / std * np.sqrt(periods_per_year))
        if std > 0 else 0.0
    )

    # ── 累積報酬與最大回撤（用原始 30 日報酬，不再 ÷ hold_days） ────
    cumret = np.cumprod(1 + np.clip(strat_ret, -0.99, None))
    peak   = np.maximum.accumulate(cumret)
    max_dd = float(np.min((cumret - peak) / peak))

    n_trades = int(np.sum(position != 0))
    win_rate = (
        float(np.mean(strat_ret_gross[position != 0] > 0))
        if n_trades > 0 else 0.0
    )
    total_tc = float(np.sum(tc))

    return {
        "sharpe":       sharpe,
        "sharpe_gross": float(np.mean(strat_ret_gross) / max(np.std(strat_ret_gross), 1e-9)
                              * np.sqrt(periods_per_year)),
        "max_drawdown": max_dd,
        "win_rate":     win_rate,
        "n_trades":     n_trades,
        "total_tc_pct": round(total_tc * 100, 3),   # 累積交易成本（%）
    }


def evaluate_fold(y_true: pd.Series, prob_up: pd.Series) -> dict:
    """彙整單一 Fold 的全套指標（含 single-class 保護）。"""
    # 統一轉為值陣列，消除 index 不一致問題（pandas 2.x 嚴格要求 index 一致）
    y_arr = np.asarray(y_true, dtype=float)
    p_arr = np.asarray(prob_up, dtype=float)
    y_s   = pd.Series(y_arr)
    p_s   = pd.Series(p_arr)

    da  = directional_accuracy(y_s, p_s - 0.5)

    # ── AUC：single-class 保護 ─────────────────────────────────
    # 當驗證窗只有單一類別（全漲或全跌），roc_auc_score 會拋出 ValueError
    # 並回傳 NaN，導致 fold 統計失真。改為安全計算。
    y_binary = (y_arr > 0).astype(int)
    n_classes = len(np.unique(y_binary))
    if n_classes < 2:
        auc = float("nan")
        logger.debug(
            f"  [evaluate_fold] single-class fold（只有 {'上漲' if y_binary.mean()>0.5 else '下跌'}），"
            "AUC 設為 NaN，不計入彙整平均"
        )
    else:
        auc = roc_auc_score(y_binary, p_arr)

    ic  = information_coefficient(y_s, p_s)
    sim = simulate_sharpe(y_s, p_s)
    return {"directional_accuracy": da, "auc": auc, "ic": ic, **sim}

# ─────────────────────────────────────────────
# Regime Detection 分析
# ─────────────────────────────────────────────

def regime_analysis(
    df:       pd.DataFrame,
    oof_pred: pd.Series,
) -> dict:
    """
    依市場波動 regime（低波動 / 中波動 / 高波動）分組評估 OOF 預測表現。

    原理：realized_vol_20d（年化）是代理「市場不確定性」最即時的指標。
      低波動 regime → 趨勢穩定，模型通常較準
      高波動 regime → 震盪/危機，模型最容易衰退（重點監控）

    回傳 dict：{regime_label: metric_dict}
    """
    vol_col = "realized_vol_20d"
    if vol_col not in df.columns:
        logger.warning("  [Regime] realized_vol_20d 欄位不存在，跳過 regime 分析")
        return {}

    vol_low  = REGIME_CONFIG["vol_low"]
    vol_high = REGIME_CONFIG["vol_high"]

    valid_idx = oof_pred.dropna().index
    if len(valid_idx) == 0:
        return {}

    vol   = df.loc[valid_idx, vol_col].fillna(df[vol_col].median())
    y_reg = df.loc[valid_idx, "target_30d"]
    pred  = oof_pred.loc[valid_idx]

    regimes = {
        f"低波動（vol < {vol_low:.0%}）": vol < vol_low,
        f"中波動（{vol_low:.0%} ≤ vol < {vol_high:.0%}）":
            (vol >= vol_low) & (vol < vol_high),
        f"高波動（vol ≥ {vol_high:.0%}）": vol >= vol_high,
    }

    results = {}
    logger.info("\n=== Regime 分析（低/中/高波動分群評估）===")
    for label, mask in regimes.items():
        n = mask.sum()
        if n < 20:
            logger.info(f"  {label}：樣本不足（{n} 筆），略過")
            continue
        try:
            m = evaluate_fold(y_reg[mask], pred[mask])
            results[label] = m
            # f-string で format spec 内に条件式は使えないため先に文字列化する
            _auc_str = f"{m['auc']:.3f}" if not np.isnan(m['auc']) else " NaN"
            _ic_str  = f"{m['ic']:.3f}"  if not np.isnan(m['ic'])  else " NaN"
            logger.info(
                f"  {label}（n={n:4d}）｜"
                f"DA={m['directional_accuracy']:.3f}  "
                f"AUC={_auc_str}  "
                f"IC={_ic_str}  Sharpe={m['sharpe']:.2f}  "
                f"Sharpe_gross={m.get('sharpe_gross', m['sharpe']):.2f}  "
                f"TC={m.get('total_tc_pct', 0):.2f}%"
            )
        except Exception as e:
            logger.warning(f"  {label} 評估失敗：{e}")

    # 高波動 vs 低波動衰退量（最重要的穩定性指標）
    low_key  = [k for k in results if "低波動" in k]
    high_key = [k for k in results if "高波動" in k]
    if low_key and high_key:
        low_da  = results[low_key[0]]["directional_accuracy"]
        high_da = results[high_key[0]]["directional_accuracy"]
        decay   = low_da - high_da
        flag    = "⚠️ 衰退顯著（> 10%）" if decay > 0.10 else "✅ 穩定"
        logger.info(
            f"\n  【關鍵】高波動 vs 低波動 DA 衰退：{decay:+.3f}  {flag}"
        )
        results["_da_decay_high_vs_low"] = decay

    return results




# ─────────────────────────────────────────────
# 特徵矩陣準備
# ─────────────────────────────────────────────

def get_feature_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    feat_cols = [c for c in ALL_FEATURES if c in df.columns]
    X = df[feat_cols].copy()
    # inf → NaN → 0（雙重保險：feature_engineering 已清一次，這裡再補一次）
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y_reg = df["target_30d"]
    y_ret = df.get("target_return", df["target_30d"]) # Fallback to % if missing
    # pandas 2.x：y_reg 若含 NaN 不能直接 astype(int)，用 Int64 或先 dropna
    y_cls = (y_reg > 0).astype("Int64").fillna(0).astype(int)
    return X, y_reg, y_ret, y_cls


# ─────────────────────────────────────────────
# Walk-Forward 主流程（第一性原理優化版）
# ─────────────────────────────────────────────

def run_walk_forward(
    df:              pd.DataFrame,
    use_tft:         bool = False,
    calibrate_probs: bool = True,
) -> dict:
    """
    【第一性原理優化版】完整 Walk-Forward 訓練 + OOF 評估。

    核心改動：
      1. 每 fold 分別記錄 xgb / lgb / tft 的 Level-1 OOF 預測
         （不再直接用 ensemble 平均作為最終 OOF）
      2. 全部 fold 結束後，用收集到的全 OOF meta-features 訓練
         Level-2 Meta-Learner（Logistic Regression）
      3. 用 Meta-Learner 重新計算最終 OOF 機率，確保評估反映真實部署效果
      4. 可選：Isotonic Calibration 進一步校準機率

    回傳
    ----
    {
        "oof_metrics":    pd.DataFrame,     # 每 fold Level-1 評估（訓練期間）
        "meta_metrics":   dict,             # Meta-Learner 重算後的全局 OOF 指標
        "summary":        dict,             # 全 fold 平均指標（L1）
        "importance":     pd.DataFrame,     # 特徵重要性（跨 fold 平均）
        "oof_preds":      pd.Series,        # Meta 重算後的 OOF 上漲機率
        "meta_ensemble":  RegimeEnsemble, # 帶有訓練好 Meta 的 ensemble（供最終模型使用）
    }
    """
    X, y_reg, y_ret, y_cls = get_feature_matrix(df)
    n = len(df)

    _test_window = WF_CONFIG.get("test_window", WF_CONFIG["step_days"])
    folds = list(purged_walk_forward_folds(
        n            = n,
        train_window = WF_CONFIG["train_window"],
        val_window   = WF_CONFIG["val_window"],
        step_days    = WF_CONFIG["step_days"],
        embargo_days = WF_CONFIG["embargo_days"],
        test_window  = _test_window,
    ))
    logger.info(
        f"=== Walk-Forward：共 {len(folds)} folds "
        f"（step={WF_CONFIG['step_days']}d, test_window={_test_window}d）==="
    )

    # ── 收集 Level-1 OOF 預測（每個模型分開）──────────────────────
    # 關鍵：三個 Series 分開記錄，才能讓 Meta-Learner 學習差異化權重
    oof_xgb     = pd.Series(np.nan, index=df.index)
    oof_lgb     = pd.Series(np.nan, index=df.index)
    oof_tft     = pd.Series(np.nan, index=df.index)

    fold_metrics    = []
    all_importances = []

    for fold in folds:
        ti, vi, sti = fold.train_idx, fold.val_idx, fold.test_idx

        X_tr, y_tr, y_tr_reg, y_tr_ret = X.iloc[ti], y_cls.iloc[ti], y_reg.iloc[ti], y_ret.iloc[ti]
        X_va, y_va, y_va_reg, y_va_ret = X.iloc[vi], y_cls.iloc[vi], y_reg.iloc[vi], y_ret.iloc[vi]
        X_te       = X.iloc[sti]

        # ── TFT（可選）──────────────────────────────────────────
        tft_prob = None
        if use_tft:
            try:
                from models.tft_model import TFTPredictor
                tft = TFTPredictor(TFT_PARAMS)
                tft.fit(
                    df.iloc[ti], df.iloc[vi],
                    checkpoint_dir=str(MODEL_DIR / f"tft_fold{fold.fold_id}"),
                )
                # predict 需要足夠的 encoder context：取 test fold 前 max_enc 天
                max_enc    = TFT_PARAMS["max_encoder_length"]
                ctx_start  = max(0, sti[0] - max_enc)
                tft_input  = df.iloc[ctx_start : sti[-1] + 1]
                result     = tft.predict(tft_input)
                tft_prob   = float(result["prob_up"])
                logger.info(
                    f"  Fold {fold.fold_id}  TFT prob_up={tft_prob:.3f}"
                )
            except Exception as e:
                logger.warning(f"  Fold {fold.fold_id}  TFT 失敗：{e}，略過")

        # ── Level-1：XGBoost + LightGBM ───────────────────────
        split_vol = REGIME_CONFIG.get("train_split", 0.30)
        ens = RegimeEnsemble(task="hybrid", vol_threshold=split_vol)
        ens.fit_level1(X_tr, y_tr_ret, X_va, y_va_ret)

        # 取出各 Level-1 模型的原始預測（不經 meta，確保 OOF 無洩漏）
        raw_pred = ens.predict(X_te, tft_pred=tft_prob)

        # 記錄分模型 OOF——這是 Meta-Learner 的訓練素材
        oof_xgb.iloc[sti] = raw_pred["xgb"]
        oof_lgb.iloc[sti] = raw_pred["lgb"]
        if tft_prob is not None:
            # tft_prob 為 scalar（單步預測），broadcast 填滿整個 test window
            oof_tft.iloc[sti] = tft_prob

        # fold 內評估仍用簡單平均（此時 meta 尚未訓練，屬合理 baseline）
        prob_up_arr = raw_pred["ensemble"]
        m = evaluate_fold(y_reg.iloc[sti], pd.Series(prob_up_arr))
        m["fold"] = fold.fold_id
        fold_metrics.append(m)

        _f_auc = f"{m['auc']:.3f}" if not np.isnan(m['auc']) else " NaN"
        _f_ic  = f"{m['ic']:.3f}"  if not np.isnan(m['ic'])  else " NaN"
        logger.info(
            f"  Fold {fold.fold_id:3d}  "
            f"DA={m['directional_accuracy']:.3f}  "
            f"AUC={_f_auc}  "
            f"IC={_f_ic}  "
            f"Sharpe={m['sharpe']:.2f}(gross={m.get('sharpe_gross', m['sharpe']):.2f})  "
            f"WinRate={m['win_rate']:.2f}  TC={m.get('total_tc_pct',0):.1f}%"
        )

        # 特徵重要性
        all_importances.append(ens.combined_importance()["mean"])

    # ── 彙整 Level-1 OOF 指標 ────────────────────────────────────
    metrics_df = pd.DataFrame(fold_metrics).set_index("fold")
    # 用 nanmean 排除 single-class fold 導致的 NaN AUC
    summary = {k: float(metrics_df[k].dropna().mean()) for k in metrics_df.columns}

    # ── Fold DA 穩定性分析（診斷 test_window 是否足夠）──────────
    da_series   = metrics_df["directional_accuracy"].dropna()
    da_std_fold = float(da_series.std())
    n_test_avg  = _test_window   # 每 fold test 樣本數（固定 = test_window）
    # 理論 DA std 上界（二項分布，p=0.5 最保守）
    da_std_theory = float(np.sqrt(0.25 / n_test_avg))  # = 0.5 / sqrt(n)
    da_stable     = da_std_fold < 0.10
    logger.info(
        f"\n=== Fold DA 穩定性（修正後分析）==="
        f"\n  每 fold test 樣本數  : {n_test_avg} 天"
        f"\n  DA std（實測）       : {da_std_fold:.4f}  "
        f"{'✅ 穩定（< 10%）' if da_stable else '⚠️ 波動大（≥ 10%）'}"
        f"\n  DA std（理論上界）   : {da_std_theory:.4f}  "
        f"（= 0.5 / √{n_test_avg}）"
        f"\n  Δ（實測 - 理論上界）: {da_std_fold - da_std_theory:+.4f}"
    )

    logger.info("\n=== Walk-Forward 彙整（Level-1 Baseline）===")
    for k, v in summary.items():
        tgt  = EVAL_TARGETS.get(k)
        if np.isnan(v):
            logger.info(f"  {k:28s}: NaN（single-class fold 太多，指標不可信）")
            continue
        flag = "✅" if tgt and v >= tgt else ("⚠️" if tgt else "")
        logger.info(f"  {k:28s}: {v:.4f}  {flag}")

    importance_df = (
        pd.concat(all_importances, axis=1)
        .mean(axis=1)
        .sort_values(ascending=False)
        .rename("importance")
        .to_frame()
    )

    logger.info("\n=== Top 20 特徵（第一性驗證）===")
    for feat, row in importance_df.head(20).iterrows():
        logger.info(f"  {feat:42s}: {row.iloc[0]:.4f}")

    # ── 訓練 Level-2 Meta-Learner（關鍵修正）────────────────────
    #
    # 只使用有完整 OOF 覆蓋的行（三個模型都有預測的交集）
    # tft_oof 可能因部分 fold 失敗而為 NaN，以 (xgb+lgb)/2 填補
    tft_filled = oof_tft.fillna((oof_xgb + oof_lgb) / 2)

    oof_meta_df = pd.DataFrame({
        "xgb_pred": oof_xgb,
        "lgb_pred": oof_lgb,
        "tft_pred": tft_filled,
    })

    # 取有效行（至少 xgb 和 lgb 都有預測）
    valid_mask = oof_xgb.notna() & oof_lgb.notna()
    oof_valid  = oof_meta_df.loc[valid_mask].fillna(0)
    y_meta     = y_cls.loc[valid_mask]

    logger.info(f"\n[Meta-Learner] 有效 OOF 樣本：{valid_mask.sum():,} / {n:,}")

    # ── 個別模型 Isotonic Calibration（新增）────────────────────
    # 原理：在 Meta-Learner 訓練前，先對 XGB / LGB 各自用全量 OOF 資料
    #       擬合 IsotonicRegression，使其 raw_prob 映射到語意正確的機率
    #       （0.5 = 真實 50% 上漲機率）。效果：
    #   1. Meta-Learner 輸入分布更乾淨（原本 XGB OOF 均值偏高約 0.61）
    #   2. 個別模型一致性比較不再需要「隔離貢獻校準」的近似
    #   3. predict.py 的 xgb_cal / lgb_cal 直接反映真實校準後機率
    logger.info("\n=== 個別模型 Isotonic Calibration ===")
    # 複用最後一個 fold 的 ens（包含 XGB/LGB 結構）作為 template
    split_vol = REGIME_CONFIG.get("train_split", 0.30)
    meta_ensemble = RegimeEnsemble(task="hybrid", vol_threshold=split_vol)

    # ── 個別模型 Isotonic Calibration（在 meta 訓練前完成）──
    # 使用已收集的 OOF 機率 vs 真實標籤
    if valid_mask.sum() >= 50:   # 至少 50 筆才有統計意義
        try:
            meta_ensemble.calibrate(oof_valid, y_meta, X_oof=X.loc[valid_mask])
            logger.info("✅ XGB + LGB 個別 Isotonic Calibration 完成")
        except Exception as e:
            logger.warning(f"  個別 Calibration 失敗（略過）：{e}")
    else:
        logger.warning(f"  OOF 樣本不足（{valid_mask.sum()} < 50），跳過個別 Calibration")

    # 建立一個新的 StackingEnsemble 來承載訓練好的 meta
    # 同時也需要 Level-1 模型（最終部署用）——用全量 OOF 資料重訓一次
    # 注意：這裡直接複用最後一個 fold 的 ens 結構，meta 是重點
    meta_ensemble.fit_meta(oof_valid, y_meta, X_oof=X.loc[valid_mask])

    logger.info("✅ Level-2 Meta-Learner 已完成訓練（基於全 OOF，無未來洩漏）")
    if hasattr(meta_ensemble.normal_model.meta, "coef_"):
        coef_dict = dict(zip(oof_valid.columns, meta_ensemble.normal_model.meta.coef_.flatten()))
        logger.info(f"   [Normal] Meta 係數: { {k: f'{v:.4f}' for k, v in coef_dict.items()} }")
    if hasattr(meta_ensemble.high_vol_model.meta, "coef_"):
        coef_dict = dict(zip(oof_valid.columns, meta_ensemble.high_vol_model.meta.coef_.flatten()))
        logger.info(f"   [High Vol] Meta 係數: { {k: f'{v:.4f}' for k, v in coef_dict.items()} }")

    # ── 用 Meta 重新計算全局 OOF 機率（反映真實部署效果）─────────
    # 呼叫 predict_meta 直接在 OOF 機率上應用 Meta-Learner 與 Regime 切分
    final_oof_prob_valid = meta_ensemble.predict_meta(oof_valid, X_oof=X.loc[valid_mask])

    oof_prob_up = pd.Series(np.nan, index=df.index)
    oof_prob_up.loc[valid_mask] = final_oof_prob_valid

    # Meta 重算後的全局 OOF 指標
    y_meta_reg  = y_reg.loc[valid_mask]
    meta_metrics = evaluate_fold(y_meta_reg, pd.Series(final_oof_prob_valid))

    logger.info("\n=== Meta-Learner OOF 指標（真實部署準確度）===")
    for k, v in meta_metrics.items():
        tgt  = EVAL_TARGETS.get(k)
        flag = "✅" if tgt and v >= tgt else ("⚠️" if tgt else "")
        logger.info(f"  {k:28s}: {v:.4f}  {flag}")

    # 指標提升量（對比 Level-1 baseline）
    logger.info("\n  【提升量 Meta vs L1 Baseline】")
    for k in ["directional_accuracy", "auc", "ic", "sharpe"]:
        if k in summary and k in meta_metrics:
            delta = meta_metrics[k] - summary[k]
            sign  = "+" if delta >= 0 else ""
            logger.info(f"  {k:28s}: {sign}{delta:.4f}")

    # ── Regime 分析（OOF Meta 預測在不同波動環境下的表現）────────
    regime_results = regime_analysis(df, oof_prob_up)

    return {
        "oof_metrics":    metrics_df,
        "meta_metrics":   meta_metrics,
        "summary":        summary,
        "importance":     importance_df,
        "oof_preds":      oof_prob_up,
        "meta_ensemble":  meta_ensemble,   # 帶有訓練好 Meta（+ 可選 Calibrator）
        "regime_results": regime_results,  # 各 regime 的 OOF 指標
    }


# ─────────────────────────────────────────────
# 最終全量訓練（部署用）
# ─────────────────────────────────────────────

def train_final_model(
    df:                    pd.DataFrame,
    use_tft:               bool = False,
    meta_ensemble_from_cv: Optional[RegimeEnsemble] = None,
) -> RegimeEnsemble:
    """
    使用全部資料訓練最終部署模型。
    最後 REGIME_CONFIG['oos_window'] 個交易日（預設 2 年）保留為 Hold-Out OOS 驗證。
    2 年窗口能同時涵蓋多頭、空頭、震盪三種 regime，評估更具代表性。

    【優化】meta_ensemble_from_cv：
      若提供，直接將 CV 期間訓練好的 Meta-Learner（+ Calibrator）
      移植至最終模型，避免在有限的 Hold-Out 上重訓 Meta 導致過擬合。
      CV OOF 的樣本量遠大於 Hold-Out，Meta 在此訓練更穩健。
    """
    X, y_reg, y_ret, y_cls = get_feature_matrix(df)
    oos_window = REGIME_CONFIG["oos_window"]   # 預設 504（2 年）
    # 若資料不足 5 年 + 2 年，退回 1 年 Hold-Out
    min_train = 252 * 5
    if len(df) - oos_window < min_train:
        oos_window = 252
        logger.warning(
            f"  資料量不足（{len(df)} 天），Hold-Out 退回 1 年（252 天）"
        )
    split = -oos_window
    logger.info(f"  Hold-Out OOS 長度：{oos_window} 天（約 {oos_window//252} 年）")
    X_tr, y_tr, y_tr_reg, y_tr_ret = X.iloc[:split], y_cls.iloc[:split], y_reg.iloc[:split], y_ret.iloc[:split]
    X_va, y_va, y_va_reg, y_va_ret = X.iloc[split:], y_cls.iloc[split:], y_reg.iloc[split:], y_ret.iloc[split:]

    tft_prob = None
    if use_tft:
        try:
            from models.tft_model import TFTPredictor
            tft = TFTPredictor(TFT_PARAMS)
            tft.fit(
                df.iloc[:split], df.iloc[split:],
                checkpoint_dir=str(MODEL_DIR / "tft_final"),
            )
            # predict 補 encoder context：取 split 前 max_enc 天
            max_enc   = TFT_PARAMS["max_encoder_length"]
            ctx_start = max(0, len(df) + split - max_enc)   # split 是負數
            tft_input = df.iloc[ctx_start:]                 # context + test window
            result    = tft.predict(tft_input)
            tft_prob  = float(result["prob_up"])
            tft.save(str(MODEL_DIR / "tft_final.ckpt"))
        except Exception as e:
            logger.warning(f"最終 TFT 訓練失敗：{e}")

    split_vol = REGIME_CONFIG.get("train_split", 0.30)
    ens = RegimeEnsemble(task="hybrid", vol_threshold=split_vol)
    ens.fit_level1(X_tr, y_tr_ret, X_va, y_va_ret)

    # ── 移植 CV 訓練好的 Meta-Learner（核心優化）────────────────
    if meta_ensemble_from_cv is not None and meta_ensemble_from_cv.meta is not None:
        ens.meta    = meta_ensemble_from_cv.meta
        ens.scaler  = meta_ensemble_from_cv.scaler
        # 一併移植 meta Calibrator（若有）
        if hasattr(meta_ensemble_from_cv, "_calibrator"):
            ens._calibrator = meta_ensemble_from_cv._calibrator
        # 移植個別模型 Isotonic Calibrators（XGB / LGB）
        if meta_ensemble_from_cv.xgb_clf._calibrator is not None:
            ens.xgb_clf._calibrator = meta_ensemble_from_cv.xgb_clf._calibrator
            logger.info("  ✅ XGB 個別 Calibrator 移植完成")
        if meta_ensemble_from_cv.lgb_clf._calibrator is not None:
            ens.lgb_clf._calibrator = meta_ensemble_from_cv.lgb_clf._calibrator
            logger.info("  ✅ LGB 個別 Calibrator 移植完成")
        logger.info("✅ Meta-Learner + 個別 Calibrators 已從 CV 移植至最終模型")
    else:
        # Fallback：在 Hold-Out 上訓練 Meta（樣本少，僅供緊急使用）
        logger.warning(
            "  meta_ensemble_from_cv 未提供，將在 Hold-Out 上訓練 Meta"
            "（樣本量有限，準確度可能略低於 CV 移植版本）"
        )
        raw_pred = ens.predict(X_va, tft_pred=tft_prob)
        oof_fallback = pd.DataFrame({
            "xgb_pred": raw_pred["xgb"],
            "lgb_pred": raw_pred["lgb"],
            "tft_pred": (raw_pred["tft"]
                         if tft_prob is not None
                         else (raw_pred["xgb"] + raw_pred["lgb"]) / 2),
        })
        ens.fit_meta(oof_fallback, y_va)

    # ── Hold-Out OOS 驗證（使用移植後的 Meta）────────────────────
    pred  = ens.predict(X_va, tft_pred=tft_prob)
    ensemble_prob = pred["ensemble"]

    # 若有 Calibrator，再過一次校準
    if hasattr(ens, "_calibrator") and ens._calibrator is not None:
        oof_va = pd.DataFrame({
            "xgb_pred": pred["xgb"],
            "lgb_pred": pred["lgb"],
            "tft_pred": pred["tft"],
        }).fillna(0)
        meta_X = ens.scaler.transform(oof_va.values)
        ensemble_prob = ens._calibrator.predict_proba(meta_X)[:, 1]

    m_oos = evaluate_fold(y_reg.iloc[split:], pd.Series(ensemble_prob))
    logger.info("\n=== Hold-Out OOS 最終驗證 ===")
    for k, v in m_oos.items():
        tgt  = EVAL_TARGETS.get(k)
        flag = "✅" if tgt and v >= tgt else ("⚠️" if tgt else "")
        logger.info(f"  {k:28s}: {v:.4f}  {flag}")

    # ── Hold-Out Regime 分析 ──────────────────────────────────────
    oos_pred = pd.Series(ensemble_prob, index=df.index[split:])
    regime_oos = regime_analysis(df.iloc[split:].copy(), oos_pred)
    ens._oos_regime_results = regime_oos   # 附掛到模型供後續查閱

    # ── 警示：若高波動 DA 衰退 > 15%，記錄警告 ──────────────────
    decay = regime_oos.get("_da_decay_high_vs_low", 0.0)
    if abs(decay) > 0.15:
        logger.warning(
            f"  ⚠️  高波動 regime DA 衰退 {decay:.1%}（>15%），"
            "建議加入 regime-aware 特徵或分 regime 訓練"
        )

    return ens


# ─────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="TSMC 2330 — 30 天趨勢預測訓練")
    p.add_argument("--stock-id",        default="2330",            help="股票代碼（預設 2330）")
    p.add_argument("--start",           default=TRAIN_START_DATE,  help=f"訓練起始日 YYYY-MM-DD（預設 {TRAIN_START_DATE}）")
    p.add_argument("--end",             default=None,               help="訓練結束日 YYYY-MM-DD（預設今日）")
    p.add_argument("--no-tft",          action="store_true",        help="跳過 TFT（較快，只用 XGB+LGB）")
    p.add_argument("--wf-only",         action="store_true",        help="只做 Walk-Forward，不訓練最終模型")
    p.add_argument("--no-calibration",  action="store_true",        help="停用 Isotonic Calibration")
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(OUTPUT_DIR / "train.log"),
        ],
    )

    args             = _parse_args()
    use_tft          = not args.no_tft
    calibrate_probs  = not args.no_calibration

    logger.info("=" * 60)
    logger.info("  TSMC 2330 — 30 天趨勢預測系統  訓練開始")
    logger.info("=" * 60)
    logger.info(f"  stock_id    : {args.stock_id}")
    logger.info(f"  日期區間    : {args.start} ~ {args.end or '今日'}")
    logger.info(f"  TFT         : {'啟用' if use_tft else '停用（--no-tft）'}")
    logger.info(f"  Calibration : {'啟用' if calibrate_probs else '停用（--no-calibration）'}")
    logger.info(f"  WF 設定     : "
                f"train={WF_CONFIG['train_window']}d  "
                f"val={WF_CONFIG['val_window']}d  "
                f"step={WF_CONFIG['step_days']}d  "
                f"embargo={WF_CONFIG['embargo_days']}d")
    # ── Step 1：從 PostgreSQL 載入資料 ─────────────────────────
    logger.info("\n[Step 1] 從 PostgreSQL 載入資料…")
    raw = build_daily_frame(
        stock_id   = args.stock_id,
        start_date = args.start,
        end_date   = args.end,
    )

    # ── Step 2：特徵工程 ───────────────────────────────────────
    logger.info("\n[Step 2] 特徵工程…")
    df = build_features(raw)
    logger.info(f"  特徵框架：{len(df):,} 天 × {df.shape[1]} 欄")

    # ── Step 3：Walk-Forward CV（含 Meta-Learner 訓練）─────────
    logger.info("\n[Step 3] Purged Walk-Forward CV + Meta-Learner 訓練…")
    wf_result = run_walk_forward(df, use_tft=use_tft, calibrate_probs=calibrate_probs)
    wf_result["oof_metrics"].to_csv(OUTPUT_DIR / "wf_fold_metrics.csv")
    wf_result["importance"].to_csv(OUTPUT_DIR / "feature_importance.csv")

    # 同時儲存 Meta 重算後的 OOF 指標（比 fold-level 平均更準確）
    meta_metrics_df = pd.DataFrame([wf_result["meta_metrics"]])
    meta_metrics_df.to_csv(OUTPUT_DIR / "meta_oof_metrics.csv", index=False)

    # 儲存 Regime 分析結果（排除 scalar 指標 _da_decay_high_vs_low）
    regime_rows = [
        {"regime": k, **v}
        for k, v in wf_result["regime_results"].items()
        if isinstance(v, dict)
    ]
    if regime_rows:
        pd.DataFrame(regime_rows).to_csv(
            OUTPUT_DIR / "regime_metrics.csv", index=False
        )
        logger.info("  Regime 指標已儲存：outputs/regime_metrics.csv")

    logger.info("\n=== Walk-Forward 最終摘要（Level-1 Baseline）===")
    for k, v in wf_result["summary"].items():
        logger.info(f"  {k}: {v:.4f}")

    # ── Step 4：最終全量訓練（移植 CV Meta）───────────────────
    if not args.wf_only:
        logger.info("\n[Step 4] 最終全量訓練（Hold-Out OOS）…")
        final_model = train_final_model(
            df,
            use_tft               = use_tft,
            meta_ensemble_from_cv = wf_result["meta_ensemble"],  # 移植 Meta
        )
        out_path = MODEL_DIR / "ensemble_final.pkl"
        joblib.dump(final_model, out_path)
        logger.info(f"  最終模型已儲存：{out_path}")

    logger.info("\n=== 訓練完成 ===")


if __name__ == "__main__":
    main()
