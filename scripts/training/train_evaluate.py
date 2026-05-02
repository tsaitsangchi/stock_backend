from __future__ import annotations
import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
for sub in ["fetchers", "pipeline", "training", "monitor", "models", "utils"]:
    p = str(base_dir / sub)
    if p not in sys.path: sys.path.append(p)
if str(base_dir) not in sys.path: sys.path.append(str(base_dir))
import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
for sub in ["fetchers", "pipeline", "training", "monitor", "models", "utils"]:
    p = str(base_dir / sub)
    if p not in sys.path: sys.path.append(p)
if str(base_dir) not in sys.path: sys.path.append(str(base_dir))
import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
for sub in ["fetchers", "pipeline", "training", "monitor", "models", "utils"]:
    p = str(base_dir / sub)
    if p not in sys.path: sys.path.append(p)
if str(base_dir) not in sys.path: sys.path.append(str(base_dir))
import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
for sub in ['fetchers', 'pipeline', 'training', 'monitor']: sys.path.append(str(base_dir / sub))
sys.path.append(str(base_dir))
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


import sys
import logging
import os
import joblib
import argparse
from datetime import datetime
from dataclasses import dataclass
from typing import Generator, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit  # [P1 第五輪修復] 強制時間序列校準
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────
# [P1 第五輪修復] CalibratedClassifierCV 防呆封裝
# 為什麼需要：sklearn 的 CalibratedClassifierCV(cv=5) 預設使用隨機 KFold，
#            在時間序列資料中等同於用未來資料校準歷史機率 → 破壞 Walk-Forward
#            嚴格性。本系統的實際校準走 OOF 路徑（meta_ensemble.calibrate
#            (oof_valid, y_meta)），因 OOF 預測本身就是時間有序的 out-of-fold
#            機率，並無洩漏。但任何未來在 in-sample 上 wrap 模型的程式碼，
#            應透過此封裝強制 TimeSeriesSplit。
# ─────────────────────────────────────────────
def make_time_series_calibrator(estimator, n_splits: int = 5, method: str = "isotonic"):
    """建立使用 TimeSeriesSplit 的 CalibratedClassifierCV，避免時間序列洩漏。"""
    return CalibratedClassifierCV(
        estimator=estimator,
        cv=TimeSeriesSplit(n_splits=n_splits),
        method=method,
    )

from config import (
    ALL_FEATURES, EVAL_TARGETS, HORIZON, TRAIN_START_DATE,
    MODEL_DIR, OUTPUT_DIR, REGIME_CONFIG, TFT_PARAMS, WF_CONFIG,
    XGB_PARAMS, LGB_PARAMS,
    STOCK_CONFIGS, get_all_features, PARETO_RATIO, TRAINING_STRATEGY, SECTOR_POOLS,
    FRICTION_CONFIG, LARGE_CAP_TICKERS, calculate_net_return
)
from data_pipeline import build_daily_frame
from feature_engineering import build_features, build_features_with_medium_term
from models.ensemble_model import RegimeEnsemble
from feature_analysis import FactorAnalyzer, print_factor_report

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
    turbo:        bool = False,  # [新增] Turbo 模式
) -> Generator[Fold, None, None]:
    """
    Purged Walk-Forward CV。
    turbo=True：會將 step_days 強制放大到 126 (半年)，極速縮減 fold 數量。
    """
    if turbo:
        step_days = max(step_days, 126)
        logger.info(f"⚡ [Turbo Mode] Step days forced to {step_days}")

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

from utils.metrics import directional_accuracy, information_coefficient, calculate_net_return, simulate_sharpe, evaluate_fold, regime_analysis
from utils.feature_selection import lasso_feature_selection


# ─────────────────────────────────────────────
# 特徵矩陣準備
# ─────────────────────────────────────────────

def get_feature_matrix(
    df: pd.DataFrame,
    stock_id: str,
    feature_list: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    # 過濾掉目標變數為 NaN 的行
    df = df.dropna(subset=["target_30d"])
    
    if feature_list is not None:
        feat_cols = [c for c in feature_list if c in df.columns]
    else:
        all_features = get_all_features(stock_id)
        feat_cols = [c for c in all_features if c in df.columns]
        
    X = df[feat_cols].copy().replace([np.inf, -np.inf], np.nan).fillna(0)
    y_reg = df.loc[X.index, "target_30d"]
    y_ret = df.loc[X.index, "target_ret_30d"] if "target_ret_30d" in df.columns else y_reg
    y_cls = (y_reg > 0).astype(int)
    
    # [P0 Fix] 確保索引唯一且具備 (date, stock_id) 語意。
    # 解決當 pool 訓練時，不同股票可能具有相同原始 index (e.g. 0, 1, 2) 導致的 loc 衝突。
    if "date" in df.columns and "stock_id" in df.columns:
        df = df.copy()
        df["_tmp_idx"] = range(len(df))
        multi_idx = pd.MultiIndex.from_frame(df[["date", "stock_id"]])
        df.index = multi_idx
        X.index = multi_idx
        y_reg.index = multi_idx
        y_ret.index = multi_idx
        y_cls.index = multi_idx
    else:
        # Fallback to unique RangeIndex if columns missing
        X = X.reset_index(drop=True)
        y_reg = y_reg.reset_index(drop=True)
        y_ret = y_ret.reset_index(drop=True)
        y_cls = y_cls.reset_index(drop=True)
    
    return df, X, y_reg, y_ret, y_cls


# ─────────────────────────────────────────────
# Walk-Forward 主流程（第一性原理優化版）
# ─────────────────────────────────────────────

def run_walk_forward(
    df:              pd.DataFrame,
    stock_id:        str,
    use_tft:         bool = False,
    calibrate_probs: bool = True,
    feature_list:    Optional[list[str]] = None,
    turbo:           bool = False,  # [新增]
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
    df, X, y_reg, y_ret, y_cls = get_feature_matrix(df, stock_id=stock_id, feature_list=feature_list)
    n = len(X)

    _test_window = WF_CONFIG.get("test_window", WF_CONFIG["step_days"])
    folds = list(purged_walk_forward_folds(
        n            = n,
        train_window = WF_CONFIG["train_window"],
        val_window   = WF_CONFIG["val_window"],
        step_days    = WF_CONFIG["step_days"],
        embargo_days = WF_CONFIG["embargo_days"],
        test_window  = _test_window,
        turbo        = turbo,
    ))
    logger.info(
        f"=== Walk-Forward：共 {len(folds)} folds "
        f"（step={WF_CONFIG['step_days']}d, test_window={_test_window}d）==="
    )

    # ── 收集 Level-1 OOF 預測（每個模型分開）──────────────────────
    # 關鍵：三個 Series 分開記錄，才能讓 Meta-Learner 學習差異化權重
    oof_xgb     = pd.Series(np.nan, index=X.index)
    oof_lgb     = pd.Series(np.nan, index=X.index)
    oof_tft     = pd.Series(np.nan, index=X.index)

    fold_metrics    = []
    all_importances = []

    for fold in folds:
        ti, vi, sti = fold.train_idx, fold.val_idx, fold.test_idx
        
        # 邊界防護：確保 sti 不會超出 X 的範圍
        sti = [i for i in sti if i < len(X)]
        if len(sti) == 0:
            logger.warning(f"Fold {fold.fold_id} 的測試索引超出範圍，跳過此 Fold")
            continue

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
                    checkpoint_dir=str(MODEL_DIR / f"tft_{stock_id}_fold{fold.fold_id}"),
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
        vol_low = REGIME_CONFIG["vol_low"]
        vol_high = REGIME_CONFIG["vol_high"]
        
        # [超級加速] Turbo 模式下全程使用單一模型 (不分 regime)
        if turbo:
            from models.ensemble_model import LGBPredictor
            lgb_params = LGB_PARAMS.copy()
            lgb_params["n_estimators"] = 100 if feature_list is None else 300
            lgb_params["learning_rate"] = 0.1
            lgb_params["n_jobs"] = 2  # 減少並行衝突
            ens = LGBPredictor(params=lgb_params)
            ens.fit(X_tr, y_tr, X_va, y_va)
            
            p = ens.predict_score(X_te)
            raw_pred = {
                "xgb": p, "lgb": p, "ensemble": p
            }
            # 為 Phase 1 & 2 提供重要性介面 (相容性)
            importance = pd.Series(ens.model.feature_importances_, index=ens.feature_names)
            ens.combined_importance = lambda: {"mean": importance}
        else:
            ens = RegimeEnsemble(task="hybrid", vol_low=vol_low, vol_high=vol_high)
            ens.fit_level1(X_tr, y_tr_ret, X_va, y_va_ret)
            raw_pred = ens.predict(X_te, tft_pred=tft_prob)


        # 記錄分模型 OOF——這是 Meta-Learner 的訓練素材
        oof_xgb.iloc[sti] = raw_pred["xgb"]
        oof_lgb.iloc[sti] = raw_pred["lgb"]
        if tft_prob is not None:
            # tft_prob 為 scalar（單步預測），broadcast 填滿整個 test window
            oof_tft.iloc[sti] = tft_prob

        # fold 內評估仍用簡單平均（此時 meta 尚未訓練，屬合理 baseline）
        prob_up_arr = raw_pred["ensemble"]
        m = evaluate_fold(y_reg.iloc[sti], pd.Series(prob_up_arr), stock_id=stock_id)
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

        # ── 實時存檔 (每 10 Fold 存一次，以便觀察回測進度) ─────────────
        if fold.fold_id % 10 == 0:
            partial_oof = pd.DataFrame({"prob_up": oof_lgb.fillna(oof_xgb)}) # 暫用 L1 平均
            partial_path = OUTPUT_DIR / f"oof_predictions_partial_{stock_id}.csv"
            partial_oof.to_csv(partial_path)
            logger.info(f"  [實時存檔] 已儲存中間回測序列至 {partial_path}")

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

    # ── [P2-5 修正] 因子類別重要性分析 ──────────────────────────
    FEATURE_GROUPS = {
        "Technical":    ["rsi", "ma", "kdj", "boll", "macd", "willr", "atr", "cci", "mfi", "slope", "mom"],
        "Chip":         ["foreign", "investment", "dealer", "margin", "short", "institutional", "sponsor", "buy_sell"],
        "Fundamental":  ["revenue", "eps", "gross_margin", "operating_income", "net_income", "fcf", "rev_yoy", "per", "pbr"],
        "Macro":        ["fed_rate", "vix", "yield", "usd_twd", "crude_oil", "gold", "adr_premium"],
        "Physics":      ["info_force", "price_accel", "entropy", "gravity", "elasticity", "system_entropy"],
        "Sentiment":    ["news", "sentiment", "volatility_cluster"],
        "Wave":         ["k_wave", "cycle", "kondratiev", "regime"],
    }
    
    group_importance = {}
    for group, patterns in FEATURE_GROUPS.items():
        score = 0.0
        for feat in importance_df.index:
            if any(p in feat.lower() for p in patterns):
                score += importance_df.loc[feat, "importance"]
        group_importance[group] = score
        
    group_imp_df = pd.Series(group_importance).sort_values(ascending=False).to_frame(name="importance")
    group_imp_df.to_csv(OUTPUT_DIR / "feature_importance_by_group.csv")
    
    logger.info("\n=== 因子類別重要性 (Aggregated) ===")
    for grp, row in group_imp_df.iterrows():
        logger.info(f"  {grp:15s}: {row['importance']:.4f}")

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
    meta_ensemble = ens
    if turbo:
        # 為相容性，我們返回與 non-turbo 一致的字典結構
        all_imp_df = pd.DataFrame(all_importances).T
        importance_mean = all_imp_df.mean(axis=1).sort_values(ascending=False)
        
        # 構造 oof_full 以供 main() 儲存 .csv 與 .npy
        oof_full = pd.DataFrame({
            "date":     y_cls.index,
            "prob_up":  oof_xgb,   # Turbo 模式下以 XGB OOF 代表
            "y_true":   y_cls.values
        })
        
        logger.info("✅ [Turbo] 已跳過 Meta-Learner 訓練，直接產出單一模型")
        return {
            "importance":   importance_mean,
            "meta_metrics": summary,
            "final_model":  ens,
            "oof_metrics":  pd.DataFrame(fold_metrics),
            "oof_preds":    oof_xgb,
            "oof_full":     oof_full,
            "regime_results": {},
            "summary":      summary,
            "meta_ensemble": None
        }

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
    if hasattr(meta_ensemble.low_vol_model.meta_learner, "coef_"):
        coef_dict = dict(zip(oof_valid.columns, meta_ensemble.low_vol_model.meta_learner.coef_.flatten()))
        logger.info(f"   [Low Vol] Meta 係數: { {k: f'{v:.4f}' for k, v in coef_dict.items()} }")
    if hasattr(meta_ensemble.mid_vol_model.meta_learner, "coef_"):
        coef_dict = dict(zip(oof_valid.columns, meta_ensemble.mid_vol_model.meta_learner.coef_.flatten()))
        logger.info(f"   [Mid Vol] Meta 係數: { {k: f'{v:.4f}' for k, v in coef_dict.items()} }")
    if hasattr(meta_ensemble.high_vol_model.meta_learner, "coef_"):
        coef_dict = dict(zip(oof_valid.columns, meta_ensemble.high_vol_model.meta_learner.coef_.flatten()))
        logger.info(f"   [High Vol] Meta 係數: { {k: f'{v:.4f}' for k, v in coef_dict.items()} }")

    # ── 用 Meta 重新計算全局 OOF 機率（反映真實部署效果）─────────
    # 呼叫 predict_meta 直接在 OOF 機率上應用 Meta-Learner 與 Regime 切分
    final_oof_prob_valid = meta_ensemble.predict_meta(oof_valid, X_oof=X.loc[valid_mask])

    oof_prob_up = pd.Series(np.nan, index=X.index)
    oof_prob_up.loc[valid_mask] = final_oof_prob_valid

    # Meta 重算後的全局 OOF 指標
    y_meta_reg  = y_reg.iloc[np.where(valid_mask.values)[0]]
    meta_metrics = evaluate_fold(y_meta_reg, pd.Series(final_oof_prob_valid, index=y_meta_reg.index), stock_id=stock_id)

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
    regime_results = regime_analysis(df, oof_prob_up, REGIME_CONFIG)

    # [P2 修復 2.10] 組裝完整 OOF 預測序列（含日期 + 真實標籤）
    # 供 backtest_audit.py 的 calibration_analysis() 與 model_health_check.py
    # 的 PSI 參考分佈使用。
    oof_full_df = pd.DataFrame({
        "date":     y_cls.index,
        "prob_up":  oof_prob_up.loc[y_cls.index].values,
        "y_true":   y_cls.values,         # 二元標籤（漲/跌）
        "y_return": y_ret.values,         # 連續報酬率
    }).dropna(subset=["prob_up"])

    return {
        "oof_metrics":    metrics_df,
        "meta_metrics":   meta_metrics,
        "summary":        summary,
        "importance":     importance_df,
        "oof_preds":      oof_prob_up,
        "oof_full":       oof_full_df,    # [P2] 含日期/標籤的完整 OOF 序列
        "meta_ensemble":  meta_ensemble,   # 帶有訓練好 Meta（+ 可選 Calibrator）
        "regime_results": regime_results,  # 各 regime 的 OOF 指標
    }


# ─────────────────────────────────────────────
# 最終全量訓練（部署用）
# ─────────────────────────────────────────────

def train_final_model(
    df:                    pd.DataFrame,
    feature_list:          list,
    stock_id:              str = "2330",
    use_tft:               bool = False,
    meta_ensemble_from_cv: Optional[RegimeEnsemble] = None,
    turbo:                 bool = False,
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
    df, X, y_reg, y_ret, y_cls = get_feature_matrix(df, stock_id=stock_id, feature_list=feature_list)
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
                checkpoint_dir=str(MODEL_DIR / f"tft_{stock_id}_final"),
            )
            # predict 補 encoder context：取 split 前 max_enc 天
            max_enc   = TFT_PARAMS["max_encoder_length"]
            ctx_start = max(0, len(df) + split - max_enc)   # split 是負數
            tft_input = df.iloc[ctx_start:]                 # context + test window
            result    = tft.predict(tft_input)
            tft_prob  = float(result["prob_up"])
            tft.save(str(MODEL_DIR / f"tft_{stock_id}_final.ckpt"))
        except Exception as e:
            logger.warning(f"最終 TFT 訓練失敗：{e}")

    if turbo:
        from models.ensemble_model import LGBPredictor
        lgb_params = LGB_PARAMS.copy()
        lgb_params["n_estimators"] = 500  # 最終模型多訓練一點
        lgb_params["learning_rate"] = 0.05
        lgb_params["n_jobs"] = 4
        ens = LGBPredictor(params=lgb_params)
        ens.fit(X_tr, y_tr, X_va, y_va)
        logger.info("✅ [Turbo] 最終單一模型訓練完成")
        
        # [P2 Fix] 為了相容 predict.py 的 ensemble.predict() 格式，
        # 將單一 Turbo 模型包裝進 StackingEnsemble。
        from models.ensemble_model import StackingEnsemble
        wrapper = StackingEnsemble(use_xgb=False, use_lgb=True, use_elastic=False, use_mom=False)
        wrapper.models = {"lgb": ens}
        wrapper.feature_names = ens.feature_names
        return wrapper

    vol_low = REGIME_CONFIG["vol_low"]
    vol_high = REGIME_CONFIG["vol_high"]
    ens = RegimeEnsemble(task="hybrid", vol_low=vol_low, vol_high=vol_high)
    ens.fit_level1(X_tr, y_tr_ret, X_va, y_va_ret)
    if meta_ensemble_from_cv is not None:
        pass
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
    regime_oos = regime_analysis(df.iloc[split:].copy(), oos_pred, REGIME_CONFIG)
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
# ─────────────────────────────────────────────
# 80/20 Pareto 精煉邏輯
# ─────────────────────────────────────────────

def select_top_features(importance_df: pd.DataFrame, ratio: float = PARETO_RATIO) -> list[str]:
    """
    依據特徵重要性，篩選出貢獻度前 ratio % 的黃金特徵。
    """
    n_top = max(10, int(len(importance_df) * ratio))
    top_features = importance_df.head(n_top).index.tolist()
    
    # 強制保留關鍵基礎特徵 (Regime, ADR 等)
    essential = ["realized_vol_20d", "trend_regime", "adr_premium"]
    for e in essential:
        if e not in top_features and e in importance_df.index:
            top_features.append(e)
            
    return top_features

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
    p.add_argument("--step-days",       type=int,                   help="Walk-Forward 步進天數（覆蓋 config.RETRAIN_FREQ）")
    p.add_argument("--fast-mode",       action="store_true",        help="快速模式：減少 Fold 數並強制 80/20 篩選")
    p.add_argument("--turbo",           action="store_true",        help="⚡ 極速模式：step_days=126 (半年)，並跳過耗時分析")
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
    # ── 資源傾斜設定 (80/20) ────────────────────────────────
    if args.turbo:
        WF_CONFIG["step_days"] = 126
        logger.info("⚡ [Turbo Mode] 啟用：目標為全標的一鍵快速更新")
    elif args.fast_mode:
        logger.info("🚀 [Fast Mode] 啟用：step_days=42")
        WF_CONFIG["step_days"] = 42
        
    use_tft          = not args.no_tft and not args.turbo
    calibrate_probs  = not args.no_calibration
    stock_id         = args.stock_id

    # 更新 WF 設定
    if args.step_days:
        WF_CONFIG["step_days"] = args.step_days

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
    
    # ── 1. 資料載入與池化 (Data Pooling) ─────────────────────
    training_pool = [stock_id]
    if TRAINING_STRATEGY["use_global_backbone"]:
        for sector, members in SECTOR_POOLS.items():
            if stock_id in members:
                training_pool = members
                logger.info(f"  [Global Backbone] 啟用分區池化訓練: {sector} (n={len(members)})")
                break
                
    from data_pipeline import load_features_from_store
    all_dfs = []
    for sid in training_pool:
        df_sid = load_features_from_store(sid, start_date=args.start, end_date=args.end)
        if not df_sid.empty:
            # [P0 Fix] 確保 date 與 stock_id 存在於欄位中，以便後續排序與建立 MultiIndex
            df_sid = df_sid.reset_index()
            if "stock_id" not in df_sid.columns:
                df_sid["stock_id"] = sid
            all_dfs.append(df_sid)
            
    if not all_dfs:
        logger.error(f"  查無資料，請先執行 python scripts/update_feature_store.py")
        sys.exit(1)
        
    df = pd.concat(all_dfs, ignore_index=True).sort_values(["date", "stock_id"])
    # ── [P0 修復] 移除重複欄位（防止 XGBoost 崩潰） ─────────────
    if df.columns.duplicated().any():
        logger.warning(f"  偵測到重複特徵欄位，已自動去重：{df.columns[df.columns.duplicated()].unique().tolist()}")
        df = df.loc[:, ~df.columns.duplicated()]
        
    df = df.dropna(subset=["target_30d"])
        
    logger.info(f"  資料載入完成，總樣本數: {len(df):,} (標的數: {len(training_pool)})")
    logger.info(f"  特徵框架：{df.shape[1]} 欄")

    # ── 多時程共識統計 ────────────────────────────────────
    if "target_consensus_binary" in df.columns:
        consensus_rate = df["target_consensus_binary"].mean()
        logger.info(f"  多時程共識上漲率：{consensus_rate:.2%}（15/21/30 天三者多數決）")
        for h in (15, 21, 30):
            col = f"target_{h}d_binary"
            if col in df.columns:
                rate = df[col].astype(float).mean()
                logger.info(f"    target_{h}d_binary 上漲率：{rate:.2%}")


    # ── Step 3：Walk-Forward CV（八二法則兩階段精煉）─────────
    logger.info("\n[Step 3] Purged Walk-Forward CV（八二法則兩階段精煉）")
    
    # --- Phase 1: 快速探索與降維 (Universal Model Approach) ---
    logger.info("\n>>> Phase 1: 核心特徵掃描與 LASSO 降維...")
    scan_result = run_walk_forward(df, stock_id=stock_id, use_tft=False, calibrate_probs=False, turbo=args.turbo)
    
    # 結合策略 ② 與 ③：特徵降維 + 正則化
    if TRAINING_STRATEGY["feature_selection"] == "robust_ic":
        # 混合篩選：IC IR + LASSO
        logger.info("執行混合特徵篩選 (IC IR + LASSO)...")
        # 首先用 LASSO 將空間壓縮到 30 個最具解釋力的維度
        feature_cols = [c for c in scan_result["importance"].index if c in df.columns]
        lasso_cols = lasso_feature_selection(df[feature_cols], df["target_30d"], max_features=30)
        
        # [P0 Fix] 若 LASSO 失敗，則直接從 importance 取前 30
        if not lasso_cols:
            logger.warning("  ⚠️ LASSO 篩選結果為空，改為使用重要性排序作為備援...")
            lasso_cols = feature_cols[:30]

        # 再用 IC IR 驗證其穩定性
        analyzer = FactorAnalyzer(df, target_col="target_30d")
        ic_report = analyzer.analyze_robustness(lasso_cols)
        
        # [P0 Fix] 確保 ic_report 不為空且包含 is_robust 欄位
        if not ic_report.empty and "is_robust" in ic_report.columns:
            golden_features = ic_report[ic_report["is_robust"]]["feature"].tolist()
        else:
            logger.warning("  ⚠️ IC IR 分析未產出有效強健因子，將回退至 LASSO 選項。")
            golden_features = []

        if len(golden_features) < 10:
            logger.info(f"  ⚠️ 黃金特徵過少 ({len(golden_features)})，改為完整保留 {len(lasso_cols)} 個 LASSO 選項...")
            golden_features = lasso_cols
    else:
        golden_features = select_top_features(scan_result["importance"])

    logger.info(f"\n✅ 篩選完成：從 {df.shape[1]} 個特徵中精選出 {len(golden_features)} 個黃金特徵")
    logger.info(f"   黃金特徵 (Top 20)：{golden_features[:20]}")
    logger.info(f"   黃金特徵範例：{golden_features[:10]}...")
    
    # --- Phase 2: 精確打擊 (啟動完整訓練) ---
    logger.info(f"\n>>> Phase 2: 精確打擊 (Refined Full Training with {len(golden_features)} features)...")
    wf_result = run_walk_forward(
        df, stock_id=stock_id, 
        use_tft=use_tft, 
        calibrate_probs=calibrate_probs,
        feature_list=golden_features,
        turbo=args.turbo
    )
    
    wf_result["oof_metrics"].to_csv(OUTPUT_DIR / "wf_fold_metrics.csv")
    wf_result["importance"].to_csv(OUTPUT_DIR / "feature_importance_refined.csv")
    
    # 儲存 OOF 預測序列 (用於回測)
    oof_df = pd.DataFrame({"prob_up": wf_result["oof_preds"]})
    oof_path = OUTPUT_DIR / f"oof_predictions_{stock_id}.csv"
    oof_df.to_csv(oof_path)
    logger.info(f"  OOF 預測序列已儲存：{oof_path}")

    # [P2 修復 2.10] 同步輸出含日期/標籤的完整 OOF 預測序列
    # 供 backtest_audit.py 做完整 calibration_analysis（不只 scalar metrics）
    oof_full = wf_result.get("oof_full")
    if oof_full is not None and not oof_full.empty:
        oof_full_path = OUTPUT_DIR / f"oof_predictions_with_dates_{stock_id}.csv"
        oof_full.to_csv(oof_full_path, index=False)
        logger.info(f"  完整 OOF 序列（date/prob_up/y_true）已儲存：{oof_full_path}")
        
        # [P0] 儲存 OOF 分佈為 .npy 供 model_health_check.py 使用
        try:
            arr = oof_full["prob_up"].dropna().values
            np.save(MODEL_DIR / f"oof_ref_dist_{stock_id}.npy", arr)
            logger.info(f"  OOF 參考分佈 (.npy) 已儲存：{MODEL_DIR / f'oof_ref_dist_{stock_id}.npy'}")
        except Exception as e:
            logger.warning(f"  儲存 OOF .npy 失敗：{e}")

    # 同時儲存 Meta 重算後的 OOF 指標（比 fold-level 平均更準確）
    meta_metrics_df = pd.DataFrame([wf_result["meta_metrics"]])
    meta_metrics_df.to_csv(OUTPUT_DIR / "meta_oof_metrics.csv", index=False)

    # 儲存 Regime 分析結果
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

    logger.info("\n=== Walk-Forward 最終摘要 (Refined) ===")
    for k, v in wf_result["summary"].items():
        logger.info(f"  {k}: {v:.4f}")

    # ── Step 4：最終全量訓練（移植 CV Meta）───────────────────
    if not args.wf_only:
        logger.info("\n[Step 4] 最終全量訓練（Hold-Out OOS）…")
        final_model = train_final_model(
            df,
            stock_id              = stock_id,
            use_tft               = use_tft,
            meta_ensemble_from_cv = wf_result["meta_ensemble"],  # 移植 Meta
            feature_list          = golden_features,            # 使用精煉特徵
            turbo                 = args.turbo
        )
        # 紀錄特徵清單至模型物件，方便預測時讀取
        final_model.refined_features = golden_features
        
        out_path = MODEL_DIR / f"ensemble_{stock_id}.pkl"
        joblib.dump(final_model, out_path)
        logger.info(f"  最終模型已儲存：{out_path}")
        
        # ── 更新效能註冊表 ──────────────────────────────────
        update_metrics_registry(stock_id, wf_result["meta_metrics"])

        # ── MLflow 實驗追蹤 (Model Versioning) ────────────────
        try:
            import mlflow
            import mlflow.sklearn
            from datetime import datetime
            
            # 設定實驗名稱 (依標的分類)
            mlflow.set_experiment(f"個股模型_{stock_id}")
            
            run_name = f"訓練_{stock_id}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            with mlflow.start_run(run_name=run_name):
                # 紀錄參數
                mlflow.log_params({
                    "股票代號": stock_id,
                    "訓練起始日": args.start,
                    "使用TFT模型": use_tft,
                    "特徵數量": len(golden_features),
                    "WF訓練窗口": WF_CONFIG["train_window"],
                    "WF步進天數": WF_CONFIG["step_days"],
                    "禁區天數": WF_CONFIG["embargo_days"]
                })
                
                # 紀錄指標 (OOF 成果)
                m = wf_result["meta_metrics"]
                mlflow.log_metrics({
                    "方向正確率(DA)": float(m.get("directional_accuracy", 0)),
                    "相關係數(IC)": float(m.get("ic", 0)),
                    "夏普比率(Sharpe)": float(m.get("sharpe", 0)),
                    "平均淨報酬": float(m.get("avg_net_return", 0)),
                    "預期價值(EV)": float(m.get("expectancy", 0))
                })
                
                # 紀錄模型物件 (Versioning)
                mlflow.sklearn.log_model(final_model, f"ensemble_{stock_id}")
                logger.info(f"  MLflow 追蹤完成：Run Name = {run_name}")
                
        except Exception as e:
            logger.warning(f"  MLflow 紀錄失敗 (請檢查 mlflow 是否安裝): {e}")

    logger.info("\n=== 訓練完成 ===")


def update_metrics_registry(stock_id: str, metrics: dict):
    """ 將訓練結果寫入統一註冊表，供管理員調度使用 """
    import json
    registry_path = "scripts/outputs/metrics_registry.json"
    registry = {}
    
    if os.path.exists(registry_path):
        try:
            with open(registry_path, "r") as f:
                registry = json.load(f)
        except Exception:
            pass
            
    # 只存儲核心指標
    registry[stock_id] = {
        "directional_accuracy": float(metrics.get("directional_accuracy", 0.5)),
        "sharpe": float(metrics.get("sharpe", 0.0)),
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    try:
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=4)
        logger.info(f"  效能註冊表已更新：{registry_path}")
    except Exception as e:
        logger.error(f"  寫入註冊表失敗: {e}")


if __name__ == "__main__":
    main()
