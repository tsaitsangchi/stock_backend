"""
models/ensemble_model.py — XGBoost + LightGBM + CatBoost Ensemble
Level-1 表格學習器，與 TFT 並列組成 Stacking Ensemble。

輸出：
  - 回歸：target_30d 點預測
  - 分類：direction_30d 機率
  - Stacking meta-learner：整合 TFT + Tree 預測
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.warning("xgboost 未安裝：pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    logger.warning("lightgbm 未安裝：pip install lightgbm")

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CAT = True
except Exception:
    # catboost 尚未支援 Python 3.14+；底層 C 擴充可能拋出 ValueError 而非 ImportError
    HAS_CAT = False
    logger.warning("catboost 不可用（Python 3.14 尚不支援），已略過")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.warning("shap 未安裝：pip install shap")


from config import XGB_PARAMS, LGB_PARAMS, ALL_FEATURES


# ─────────────────────────────────────────────
# Level-1：XGBoost
# ─────────────────────────────────────────────

class XGBPredictor:
    def __init__(self, params: dict = None, task: str = "classification"):
        """
        task: 'classification'（方向） 或 'regression'（報酬率）
        """
        self.params = params or XGB_PARAMS.copy()
        self.task = task
        self.model = None
        self.feature_names: list[str] = []
        # 個別 Calibrator（預設 Platt Scaling，保序回歸備用）
        self._calibrator = None
        self._cal_type   = "none"   # 'sigmoid' | 'isotonic' | 'none'

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame, y_val: pd.Series) -> "XGBPredictor":
        if not HAS_XGB:
            raise ImportError("xgboost 未安裝")

        self.feature_names = X_train.columns.tolist()
        p = self.params.copy()
        # XGBoost 2.x：early_stopping_rounds 移至建構子，不再是 fit() 參數
        es = p.pop("early_stopping_rounds", 50)

        if self.task == "classification":
            self.model = xgb.XGBClassifier(
                **p,
                early_stopping_rounds=es,   # ← 放在建構子
                objective="binary:logistic",
                eval_metric="auc",
            )
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=100,
            )
        else:
            self.model = xgb.XGBRegressor(
                **p,
                early_stopping_rounds=es,   # ← 放在建構子
                objective="reg:squarederror",
                eval_metric="rmse",
            )
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=100,
            )
        logger.info(f"  [XGB-{self.task}] 最佳迭代：{self.model.best_iteration}")
        return self

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """回傳原始（未校準）機率或回歸分數，meta-learner 訓練時使用此輸出。"""
        if self.task == "classification":
            return self.model.predict_proba(X)[:, 1]
        else:
            return self.model.predict(X)

    def predict_score_cal(self, X: pd.DataFrame) -> np.ndarray:
        """
        回傳 Platt/Isotonic 校準後機率（語意對齊）。
        預設使用 Platt Scaling（LogisticRegression），提供平滑的 S 型映射。
        IsotonicRegression 在稀疏區間（如 0.44~0.50）會優化 step jump；
        Platt 則透過 sigmoid 平滑插値，對稀疏區間更穩健。
        若校準器尚未安裝，回退到 predict_score()。
        """
        raw = self.predict_score(X)
        if self._calibrator is None:
            return raw
        # LogisticRegression（Platt Scaling）有 predict_proba；
        # IsotonicRegression 僅有 predict/transform
        if hasattr(self._calibrator, "predict_proba"):
            return self._calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        return self._calibrator.predict(raw)

    def calibrate(
        self,
        oof_probs: np.ndarray,
        y_true: np.ndarray,
        method: str = "isotonic",
    ) -> None:
        """
        用 Walk-Forward 全量 OOF 訓練個別 Calibrator。

        method:
          'isotonic' — IsotonicRegression（預設）：保序回歸，正確捕捉
                       XGB/LGB 過度自信的非線性映射（如 raw>0.70 實際準確率
                       反而下降的模式）。OOF 樣本 > 500 時推薦。
          'sigmoid'  — Platt Scaling：LogisticRegression on raw probs。
                       若 OOF 顯示線性關係時較穩定，但本模型 XGB coef<0
                       （反向映射），使用 sigmoid 會得到錯誤方向。

        Args:
            oof_probs: OOF 各行的原始預測機率（shape: [N]）
            y_true:    真實方向標籤 0/1（shape: [N]）
            method:    'isotonic'（預設）或 'sigmoid'
        """
        if method == "sigmoid":
            from sklearn.linear_model import LogisticRegression
            cal = LogisticRegression(
                C=1.0, max_iter=500, solver="lbfgs"
            )
            # 直接以 raw_prob 為張量，括號內有完整的 1D Platt S-curve
            cal.fit(oof_probs.reshape(-1, 1), y_true)
            self._calibrator = cal
            self._cal_type   = "sigmoid"
            cal_mean = float(cal.predict_proba(oof_probs.reshape(-1, 1))[:, 1].mean())
        else:
            from sklearn.isotonic import IsotonicRegression
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(oof_probs, y_true)
            self._calibrator = cal
            self._cal_type   = "isotonic"
            cal_mean = float(cal.predict(oof_probs).mean())

        logger.info(
            f"  [XGB Calibrator/{method}] 擬合完成："
            f"OOF 均值 {oof_probs.mean():.4f} → 校準後均值 {cal_mean:.4f}  "
            f"（樣本 {len(oof_probs):,} 筆）"
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def feature_importance(self) -> pd.Series:
        if self.model is None:
            return pd.Series(dtype=float)
        imp = self.model.feature_importances_
        return pd.Series(imp, index=self.feature_names).sort_values(ascending=False)

    def shap_values(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        if not HAS_SHAP or self.model is None:
            return None
        explainer = shap.TreeExplainer(self.model)
        return explainer.shap_values(X)


# ─────────────────────────────────────────────
# Level-1：LightGBM
# ─────────────────────────────────────────────

class LGBPredictor:
    def __init__(self, params: dict = None, task: str = "classification"):
        self.params = params or LGB_PARAMS.copy()
        self.task = task
        self.model = None
        self.feature_names: list[str] = []
        # 個別 Calibrator（預設 Platt Scaling，同 XGBPredictor）
        self._calibrator = None
        self._cal_type   = "none"

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame, y_val: pd.Series) -> "LGBPredictor":
        if not HAS_LGB:
            raise ImportError("lightgbm 未安裝")

        self.feature_names = X_train.columns.tolist()
        p = self.params.copy()
        es = p.pop("early_stopping_rounds", 50)

        callbacks = [
            lgb.early_stopping(es, verbose=False),
            lgb.log_evaluation(100),
        ]

        if self.task == "classification":
            self.model = lgb.LGBMClassifier(**p, objective="binary", metric="auc")
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks,
            )
        elif self.task == "ranking":
            y_train_rank = pd.qcut(y_train, 5, labels=False, duplicates='drop')
            y_val_rank = pd.qcut(y_val, 5, labels=False, duplicates='drop')
            self.model = lgb.LGBMRanker(**p, objective="rank_xendcg")
            self.model.fit(
                X_train, y_train_rank,
                eval_set=[(X_val, y_val_rank)],
                group=[len(X_train)],
                eval_group=[[len(X_val)]],
                callbacks=callbacks,
            )
        else:
            self.model = lgb.LGBMRegressor(**p, objective="regression", metric="rmse")
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks,
            )
        logger.info(f"  [LGB-{self.task}] 最佳迭代：{self.model.best_iteration_}")
        return self

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """回傳原始（未校準）機率、回歸或排序分數，meta-learner 訓練時使用此輸出。"""
        if self.task == "classification":
            return self.model.predict_proba(X)[:, 1]
        else:
            return self.model.predict(X)

    def predict_score_cal(self, X: pd.DataFrame) -> np.ndarray:
        """Platt/Isotonic 校準後機率（同 XGBPredictor，若無校準器則回退原始值）。"""
        raw = self.predict_score(X)
        if self._calibrator is None:
            return raw
        if hasattr(self._calibrator, "predict_proba"):
            return self._calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        return self._calibrator.predict(raw)

    def calibrate(
        self,
        oof_probs: np.ndarray,
        y_true: np.ndarray,
        method: str = "isotonic",
    ) -> None:
        """用 Walk-Forward 全量 OOF 訓練個別 Calibrator（同 XGBPredictor）。"""
        if method == "sigmoid":
            from sklearn.linear_model import LogisticRegression
            cal = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
            cal.fit(oof_probs.reshape(-1, 1), y_true)
            self._calibrator = cal
            self._cal_type   = "sigmoid"
            cal_mean = float(cal.predict_proba(oof_probs.reshape(-1, 1))[:, 1].mean())
        else:
            from sklearn.isotonic import IsotonicRegression
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(oof_probs, y_true)
            self._calibrator = cal
            self._cal_type   = "isotonic"
            cal_mean = float(cal.predict(oof_probs).mean())

        logger.info(
            f"  [LGB Calibrator/{method}] 擬合完成："
            f"OOF 均值 {oof_probs.mean():.4f} → 校準後均值 {cal_mean:.4f}  "
            f"（樣本 {len(oof_probs):,} 筆）"
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def feature_importance(self) -> pd.Series:
        if self.model is None:
            return pd.Series(dtype=float)
        imp = self.model.feature_importances_
        return pd.Series(imp, index=self.feature_names).sort_values(ascending=False)

    def shap_values(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        if not HAS_SHAP or self.model is None:
            return None
        explainer = shap.TreeExplainer(self.model)
        return explainer.shap_values(X)


# ─────────────────────────────────────────────
# Level-2：Stacking Meta-Learner
# ─────────────────────────────────────────────

class StackingEnsemble:
    """
    Level-1：TFT + XGBoost + LightGBM（各出一個預測）
    Level-2：Logistic Regression（分類）或 Ridge（回歸）作為 meta-learner

    訓練流程（Purged Walk-Forward）：
      1. 用 OOF（Out-of-Fold）預測生成 meta-features
      2. 在 meta-features 上訓練 meta-learner
      3. 推論時：先過三個 L1 模型 → 拼接 → 過 L2
    """

    def __init__(self, task: str = "classification"):
        self.task = task
        if task == "hybrid":
            self.xgb_clf    = XGBPredictor(task="regression")
            self.lgb_clf    = LGBPredictor(task="ranking")
        else:
            self.xgb_clf    = XGBPredictor(task=task)
            self.lgb_clf    = LGBPredictor(task=task)
        self.meta: Optional[LogisticRegression | Ridge] = None
        self.scaler     = StandardScaler()
        self.tft_col    = "tft_pred"    # TFT 預測欄位名稱
        self._calibrator = None         # Isotonic Calibrator（joblib 必須初始化才會序列化）

    def _get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feat_cols = [c for c in ALL_FEATURES if c in df.columns]
        X = df[feat_cols].fillna(0)
        return X

    # ── Level-1 訓練 ──
    def fit_level1(self,
                   X_train: pd.DataFrame, y_train: pd.Series,
                   X_val:   pd.DataFrame, y_val:   pd.Series):
        """分別訓練 XGB 和 LGB。"""
        logger.info("  [L1] 訓練 XGBoost…")
        self.xgb_clf.fit(X_train, y_train, X_val, y_val)

        logger.info("  [L1] 訓練 LightGBM…")
        self.lgb_clf.fit(X_train, y_train, X_val, y_val)

    # ── Level-2 訓練 ──
    def fit_meta(self,
                 oof_df: pd.DataFrame,
                 y_meta: pd.Series):
        """
        oof_df：包含 'xgb_pred', 'lgb_pred', 'tft_pred' 三欄（OOF 預測）
        """
        # 用 numpy array 訓練 scaler，避免後續 transform 時的 feature name 警告
        self._meta_columns = list(oof_df.columns)
        meta_X = self.scaler.fit_transform(oof_df.fillna(0).values)

        if self.task == "classification" or self.task == "hybrid":
            self.meta = LogisticRegression(C=1.0, max_iter=1000)
        else:
            self.meta = Ridge(alpha=1.0)

        self.meta.fit(meta_X, y_meta)
        logger.info(f"  [L2 Meta] 係數：{dict(zip(oof_df.columns, self.meta.coef_.flatten()))}")

    # ── 推論 ──
    def predict(self, X: pd.DataFrame, tft_pred: Optional[float] = None) -> dict:
        """
        X         ：特徵 DataFrame（單行或多行）
        tft_pred  ：TFT 的點預測或機率（可為 None，此時以 tree 平均替代）

        回傳 dict 包含：
          ensemble  : 校準後的整體機率（主要輸出）
          xgb/lgb/tft : 各子模型原始機率（僅供參考）
          xgb_cal/lgb_cal/tft_cal : 各子模型「隔離貢獻校準機率」
            → 透過 scaler.mean_ 固定其餘模型於 OOF 均值，隔離單一模型貢獻
            → 語意等同「若只用這個模型，校準後的上漲機率是多少」
            → 可直接用於方向一致性比較（相互間語意對齊）
        """
        feat_X = self._get_features(X)

        xgb_p = self.xgb_clf.predict_score(feat_X)
        lgb_p = self.lgb_clf.predict_score(feat_X)

        tft_p = np.full(len(feat_X), tft_pred if tft_pred is not None
                        else (xgb_p + lgb_p) / 2)

        oof = pd.DataFrame({
            "xgb_pred": xgb_p,
            "lgb_pred": lgb_p,
            "tft_pred": tft_p,
        })

        if self.meta is not None:
            meta_X = self.scaler.transform(oof.fillna(0).values)
            if self.task in ["classification", "hybrid"]:
                # 優先使用 Isotonic Calibrator（機率語意更準確）
                if self._calibrator is not None:
                    ensemble_prob = self._calibrator.predict_proba(meta_X)[:, 1]
                else:
                    ensemble_prob = self.meta.predict_proba(meta_X)[:, 1]
            else:
                ensemble_prob = self.meta.predict(meta_X)
        else:
            ensemble_prob = oof.mean(axis=1).values    # 等權平均 fallback

        # ── 個別校準機率（直接使用各子模型的 Isotonic Calibrator）──────
        # 原理：各子模型經 OOF 全量資料擬合包序回歸，
        #       raw_prob → calibrated_prob 使 0.5 = 真實 50% 上漲機率。
        # 比「雔離貢獻校準」更精準：直接擬合而非透過 meta 指 chain 近似。
        xgb_cal = self.xgb_clf.predict_score_cal(feat_X)
        lgb_cal = self.lgb_clf.predict_score_cal(feat_X)
        tft_cal = tft_p.copy()    # TFT 已有內部分位校準，不需另外校準

        return {
            "ensemble": ensemble_prob,
            "xgb":      xgb_p,
            "lgb":      lgb_p,
            "tft":      tft_p,
            "xgb_cal":  xgb_cal,   # 個別 Isotonic 校準後機率
            "lgb_cal":  lgb_cal,
            "tft_cal":  tft_cal,
        }

    def predict_meta(self, oof_df: pd.DataFrame) -> np.ndarray:
        """直接使用 OOF 預測矩陣過 Meta Learner（不跑 Level-1）"""
        if self.meta is None:
            return oof_df.mean(axis=1).values
        meta_X = self.scaler.transform(oof_df.fillna(0).values)
        if self.task in ["classification", "hybrid"]:
            if self._calibrator is not None:
                return self._calibrator.predict_proba(meta_X)[:, 1]
            else:
                return self.meta.predict_proba(meta_X)[:, 1]
        else:
            return self.meta.predict(meta_X)

    # ── SHAP 解釋 ──
    def shap_analysis(self, X: pd.DataFrame) -> dict:
        feat_X = self._get_features(X)
        return {
            "xgb_shap": self.xgb_clf.shap_values(feat_X),
            "lgb_shap": self.lgb_clf.shap_values(feat_X),
        }

    # ── Feature Importance 彙整 ──
    def combined_importance(self) -> pd.DataFrame:
        xgb_imp = self.xgb_clf.feature_importance().rename("xgb")
        lgb_imp = self.lgb_clf.feature_importance().rename("lgb")
        df = pd.concat([xgb_imp, lgb_imp], axis=1).fillna(0)
        df["mean"] = df.mean(axis=1)
        return df.sort_values("mean", ascending=False)

    # ── 評估 ──
    def evaluate(self, X: pd.DataFrame, y: pd.Series,
                 tft_pred: Optional[np.ndarray] = None) -> dict:
        pred_dict = self.predict(X, tft_pred)
        ensemble  = pred_dict["ensemble"]

        if self.task == "classification":
            direction_pred = (ensemble > 0.5).astype(int)
            direction_true = (y > 0).astype(int)
            da  = accuracy_score(direction_true, direction_pred)
            auc = roc_auc_score(direction_true, ensemble)
            return {"directional_accuracy": da, "auc": auc}
        else:
            ic = pd.Series(ensemble).corr(y)
            return {"ic": ic, "rmse": float(np.sqrt(((ensemble - y)**2).mean()))}


# ─────────────────────────────────────────────
# Level-3：Regime Ensemble (自動切換 Normal / High Volatility 模型)
# ─────────────────────────────────────────────

class RegimeEnsemble:
    """
    分 Regime 訓練：依據 realised_vol_20d 切分兩套 StackingEnsemble。
    預設以 30% 做切分，解決高/低波動期特徵重要性截然不同的問題。
    """
    def __init__(self, task: str = "classification", vol_threshold: float = 0.30):
        self.task = task
        self.vol_threshold = vol_threshold
        self.vol_col = "realized_vol_20d"
        
        self.normal_model = StackingEnsemble(task)
        self.high_vol_model = StackingEnsemble(task)
        self.feature_names = None
        
    def _split_mask(self, X: pd.DataFrame) -> pd.Series:
        if self.vol_col in X.columns:
            return X[self.vol_col] >= self.vol_threshold
        else:
            logger.warning(f"[RegimeEnsemble] 缺少 {self.vol_col}，全部視為 normal regime。")
            return pd.Series(False, index=X.index)

    def fit_level1(self, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series):
        self.feature_names = X_train.columns.tolist()
        
        tr_mask = self._split_mask(X_train)
        va_mask = self._split_mask(X_val)
        
        n_high_tr = tr_mask.sum()
        n_norm_tr = len(tr_mask) - n_high_tr

        logger.info(f"  [Regime] 訓練集切分：Normal={n_norm_tr}, High-Vol={n_high_tr}")
        
        # 訓練 Normal Model
        self.normal_model.fit_level1(
            X_train[~tr_mask], y_train[~tr_mask],
            X_val[~va_mask] if (~va_mask).sum() > 0 else X_val, 
            y_val[~va_mask] if (~va_mask).sum() > 0 else y_val
        )
        
        # 訓練 High Vol Model，若樣本太少(如剛開始)可能 fail，加保護
        if n_high_tr > 20:
            self.high_vol_model.fit_level1(
                X_train[tr_mask], y_train[tr_mask],
                X_val[va_mask] if va_mask.sum() > 0 else X_val, 
                y_val[va_mask] if va_mask.sum() > 0 else y_val
            )
        else:
            logger.warning(f"  [Regime] High-Vol 樣本過少 ({n_high_tr})，借用 normal model 權重")
            self.high_vol_model = self.normal_model

    def fit_meta(self, oof_df: pd.DataFrame, y_meta: pd.Series, X_oof: pd.DataFrame = None):
        """
        傳入 X_oof 讓 meta 知道 OOF 的 vol 狀態
        """
        if X_oof is None:
            logger.warning("[RegimeEnsemble.fit_meta] X_oof 為空，無法切分 oof，使用 normal_model 訓練 meta")
            self.normal_model.fit_meta(oof_df, y_meta)
            self.high_vol_model.fit_meta(oof_df, y_meta)
            return

        mask = self._split_mask(X_oof)
        n_high = mask.sum()
        
        self.normal_model.fit_meta(oof_df[~mask], y_meta[~mask])
        
        if n_high > 10:
            self.high_vol_model.fit_meta(oof_df[mask], y_meta[mask])
        else:
            self.high_vol_model.meta = self.normal_model.meta

    def calibrate(self, oof_df: pd.DataFrame, y_meta: pd.Series, X_oof: pd.DataFrame = None):
        """分別對 normal 和 high vol 模型做 isotonic calibration"""
        if X_oof is None:
            self.normal_model.xgb_clf.calibrate(oof_df["xgb_pred"].values, y_meta.values)
            self.normal_model.lgb_clf.calibrate(oof_df["lgb_pred"].values, y_meta.values)
            self.high_vol_model.xgb_clf.calibrate(oof_df["xgb_pred"].values, y_meta.values)
            self.high_vol_model.lgb_clf.calibrate(oof_df["lgb_pred"].values, y_meta.values)
            return
            
        mask = self._split_mask(X_oof)
        if (~mask).sum() >= 50:
            self.normal_model.xgb_clf.calibrate(oof_df.loc[~mask, "xgb_pred"].values, y_meta[~mask].values)
            self.normal_model.lgb_clf.calibrate(oof_df.loc[~mask, "lgb_pred"].values, y_meta[~mask].values)
        if mask.sum() >= 50:
            self.high_vol_model.xgb_clf.calibrate(oof_df.loc[mask, "xgb_pred"].values, y_meta[mask].values)
            self.high_vol_model.lgb_clf.calibrate(oof_df.loc[mask, "lgb_pred"].values, y_meta[mask].values)

    def predict(self, X: pd.DataFrame, tft_pred: Optional[Union[np.ndarray, float]] = None) -> dict:
        if isinstance(tft_pred, (float, int)):
            tft_pred = np.full(len(X), float(tft_pred))
            
        mask = self._split_mask(X)
        
        # 就算空 Series 也要有基礎結構
        pred_norm = self.normal_model.predict(X[~mask], tft_pred[~mask] if tft_pred is not None else None) if (~mask).sum() > 0 else None
        pred_high = self.high_vol_model.predict(X[mask], tft_pred[mask] if tft_pred is not None else None) if mask.sum() > 0 else None
        
        res = {}
        # 合併所有 key 的 Dict
        keys = ["ensemble", "xgb", "lgb", "tft", "xgb_cal", "lgb_cal", "tft_cal"]
        
        for k in keys:
            merged = pd.Series(index=X.index, dtype=float)
            if pred_norm is not None and k in pred_norm:
                merged.loc[~mask] = pred_norm[k]
            if pred_high is not None and k in pred_high:
                merged.loc[mask] = pred_high[k]
            res[k] = merged.values
            
        return res

    def predict_meta(self, oof_df: pd.DataFrame, X_oof: pd.DataFrame) -> np.ndarray:
        """只用 Meta Learner 推論（處理 OOF 時使用）"""
        mask = self._split_mask(X_oof)
        ensemble_prob = pd.Series(index=oof_df.index, dtype=float)
        
        if (~mask).sum() > 0:
            ensemble_prob.loc[~mask] = self.normal_model.predict_meta(oof_df[~mask])
        if mask.sum() > 0:
            ensemble_prob.loc[mask] = self.high_vol_model.predict_meta(oof_df[mask])
            
        return ensemble_prob.values

    def combined_importance(self) -> pd.DataFrame:
        """回傳加權平均或並列的特徵重要性"""
        df_norm = self.normal_model.combined_importance().rename(columns={"mean": "mean_norm"})
        if self.high_vol_model != self.normal_model:
            df_high = self.high_vol_model.combined_importance().rename(columns={"mean": "mean_high"})
            # 合併
            df = df_norm[['mean_norm']].join(df_high[['mean_high']], how='outer').fillna(0)
            df['mean'] = (df['mean_norm'] + df['mean_high']) / 2
            return df.sort_values("mean", ascending=False)
        else:
            return df_norm

    def evaluate(self, X: pd.DataFrame, y: pd.Series,
                 tft_pred: Optional[np.ndarray] = None) -> dict:
        pred_dict = self.predict(X, tft_pred)
        ensemble = pred_dict["ensemble"]

        if self.task == "classification":
            direction_pred = (ensemble > 0.5).astype(int)
            direction_true = (y > 0).astype(int)
            da = accuracy_score(direction_true, direction_pred)
            auc = roc_auc_score(direction_true, ensemble)
            return {"directional_accuracy": da, "auc": auc}
        else:
            ic = pd.Series(ensemble).corr(y)
            return {"ic": ic, "rmse": float(np.sqrt(((ensemble - y)**2).mean()))}

