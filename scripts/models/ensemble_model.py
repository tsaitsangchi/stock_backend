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
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.special import softmax

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


        return self.model.feature_importances_


# ─────────────────────────────────────────────
# Level-1：異質性模型 (ElasticNet & Momentum)
# ─────────────────────────────────────────────

class ElasticNetPredictor:
    """
    線性因子模型：高可解釋性，在低波動/線性 Regime 下穩定。
    """
    def __init__(self, task: str = "classification"):
        self.task = task
        # 使用 ElasticNet 正則化處理高維共線性
        if task == "classification":
            self.model = LogisticRegression(
                penalty="elasticnet", solver="saga", l1_ratio=0.5, max_iter=3000
            )
        else:
            self.model = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
        self.feature_names = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, *args, **kwargs):
        self.feature_names = X_train.columns.tolist()
        X_tmp = X_train.fillna(X_train.median()).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.fit_transform(X_tmp)
        self.model.fit(X_scaled, y_train)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_tmp = X.fillna(X.median()).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.transform(X_tmp)
        if self.task == "classification":
            return self.model.predict_proba(X_scaled)[:, 1]
        return self.model.predict(X_scaled)

    def feature_importance(self) -> pd.Series:
        if hasattr(self.model, "coef_"):
            coefs = np.abs(self.model.coef_.flatten())
            return pd.Series(coefs, index=self.feature_names).sort_values(ascending=False)
        return pd.Series(dtype=float)


class SimpleMomentumModel:
    """
    基於規則的動量模型：不受樣本量限制，提供與 ML 完全異質的信號。
    """
    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def fit(self, *args, **kwargs):
        return self # 規則模型無需訓練

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # 核心邏輯：捕捉價格慣性
        if "returns_1d" in X.columns:
            mom = X["returns_1d"].rolling(self.lookback).sum().fillna(0)
            # 映射至 [0, 1] 機率空間 (Sigmoid 近似)
            return 1 / (1 + np.exp(-mom * 5))
        return np.full(len(X), 0.5)

    def feature_importance(self) -> pd.Series:
        return pd.Series({"returns_1d": 1.0})


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

    def __init__(
        self,
        use_xgb: bool = True,
        use_lgb: bool = True,
        use_elastic: bool = True,
        use_mom: bool = True,
        task: str = "classification"
    ):
        self.task = task
        self.models = {}
        if use_xgb:     self.models["xgb"] = XGBPredictor(task=task)
        if use_lgb:     self.models["lgb"] = LGBPredictor(task=task)
        if use_elastic: self.models["elastic"] = ElasticNetPredictor()
        if use_mom:     self.models["mom"] = SimpleMomentumModel()
        
        self.meta_learner = LogisticRegression() # Level-2 Stacking
        self.weights = {} # 動態加權
        self.scaler     = StandardScaler()
        self.feature_names: Optional[list[str]] = None

    def _get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.feature_names is not None:
            feat_cols = [c for c in self.feature_names if c in df.columns]
        else:
            from config import ALL_FEATURES
            feat_cols = [c for c in ALL_FEATURES if c in df.columns]
        X = df[feat_cols].fillna(0)
        return X

    # ── Level-1 訓練 ──
    def fit_level1(self,
                   X_train: pd.DataFrame, y_train: pd.Series,
                   X_val:   pd.DataFrame, y_val:   pd.Series):
        """分別訓練各子模型。"""
        self.feature_names = X_train.columns.tolist()
        for name, model in self.models.items():
            logger.info(f"  [L1] 訓練 {name}…")
            if hasattr(model, "fit"):
                if name in ["xgb", "lgb"]:
                    model.fit(X_train, y_train, X_val, y_val)
                else:
                    model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame, recent_performance: dict = None) -> dict:
        """
        產出各模型預測，並根據最近表現進行動態加權。
        """
        feat_X = self._get_features(X)
        preds = {}
        for name, model in self.models.items():
            preds[name] = model.predict(feat_X)
        
        # ── 動態加權 (Softmax weighting) ──────────────────────
        if recent_performance:
            # 依據各模型最近 60 天的 DA/IC 評分
            scores = np.array([recent_performance.get(name, 0.5) for name in self.models.keys()])
            # Simple softmax implementation
            exp_scores = np.exp(scores / 0.05)
            w = exp_scores / np.sum(exp_scores)
            self.weights = dict(zip(self.models.keys(), w))
            
            ensemble_prob = np.zeros(len(X))
            for i, name in enumerate(self.models.keys()):
                ensemble_prob += preds[name] * w[i]
            preds["ensemble"] = ensemble_prob
        else:
            # 預設等權重
            preds["ensemble"] = np.mean(list(preds.values()), axis=0)
            
        return preds

    def combined_importance(self) -> pd.DataFrame:
        importances = []
        for name, model in self.models.items():
            if hasattr(model, "feature_importance"):
                importances.append(model.feature_importance().rename(name))
        
        df = pd.concat(importances, axis=1).fillna(0)
        df["mean"] = df.mean(axis=1)
        return df.sort_values("mean", ascending=False)

    # ── 評估 ──
    def evaluate(self, X: pd.DataFrame, y: pd.Series,
                 recent_performance: dict = None) -> dict:
        pred_dict = self.predict(X, recent_performance)
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
    分 Regime 訓練：依據 realised_vol_20d 切分三套 StackingEnsemble。
    預設以 20% 和 40% 做切分，解決高/中/低波動期特徵重要性截然不同的問題。
    """
    def __init__(self, task: str = "classification", vol_low: float = 0.20, vol_high: float = 0.40):
        self.task = task
        self.vol_low = vol_low
        self.vol_high = vol_high
        self.vol_col = "realized_vol_20d"
        
        self.low_vol_model = StackingEnsemble(task)
        self.mid_vol_model = StackingEnsemble(task)
        self.high_vol_model = StackingEnsemble(task)
        self.feature_names = None
        
    def _split_mask(self, X: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
        if self.vol_col in X.columns:
            v = X[self.vol_col]
            return (v < self.vol_low), ((v >= self.vol_low) & (v < self.vol_high)), (v >= self.vol_high)
        else:
            logger.warning(f"[RegimeEnsemble] 缺少 {self.vol_col}，全部視為 mid regime。")
            f = pd.Series(False, index=X.index)
            t = pd.Series(True, index=X.index)
            return f, t, f

    def fit_level1(self, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series):
        self.feature_names = X_train.columns.tolist()
        
        tr_low, tr_mid, tr_high = self._split_mask(X_train)
        va_low, va_mid, va_high = self._split_mask(X_val)
        
        n_low_tr = tr_low.sum()
        n_mid_tr = tr_mid.sum()
        n_high_tr = tr_high.sum()

        logger.info(f"  [Regime] 訓練集切分：Low-Vol={n_low_tr}, Mid-Vol={n_mid_tr}, High-Vol={n_high_tr}")
        
        # 訓練 Low Vol Model
        if n_low_tr > 20:
            self.low_vol_model.fit_level1(
                X_train[tr_low], y_train[tr_low],
                X_val[va_low] if va_low.sum() > 0 else X_val, 
                y_val[va_low] if va_low.sum() > 0 else y_val
            )
        else:
            logger.warning(f"  [Regime] Low-Vol 樣本過少 ({n_low_tr})，將借用 mid model 權重")
            
        # 訓練 Mid Vol Model
        if n_mid_tr > 20:
            self.mid_vol_model.fit_level1(
                X_train[tr_mid], y_train[tr_mid],
                X_val[va_mid] if va_mid.sum() > 0 else X_val, 
                y_val[va_mid] if va_mid.sum() > 0 else y_val
            )
        else:
            logger.warning(f"  [Regime] Mid-Vol 樣本過少 ({n_mid_tr})，將借用其他模型權重")

        # 處理樣本過少的情況 (權重借用)
        if n_low_tr <= 20:
            self.low_vol_model = self.mid_vol_model
        if n_mid_tr <= 20:
            self.mid_vol_model = self.low_vol_model if n_low_tr > 20 else self.high_vol_model
        
        # 訓練 High Vol Model
        if n_high_tr > 20:
            self.high_vol_model.fit_level1(
                X_train[tr_high], y_train[tr_high],
                X_val[va_high] if va_high.sum() > 0 else X_val, 
                y_val[va_high] if va_high.sum() > 0 else y_val
            )
        else:
            logger.warning(f"  [Regime] High-Vol 樣本過少 ({n_high_tr})，借用 mid model 權重")
            self.high_vol_model = self.mid_vol_model

    def fit_meta(self, oof_df: pd.DataFrame, y_meta: pd.Series, X_oof: pd.DataFrame = None):
        if X_oof is None:
            logger.warning("[RegimeEnsemble.fit_meta] X_oof 為空，無法切分 oof，使用 mid_model 訓練 meta")
            self.low_vol_model.fit_meta(oof_df, y_meta)
            self.mid_vol_model.fit_meta(oof_df, y_meta)
            self.high_vol_model.fit_meta(oof_df, y_meta)
            return

        m_low, m_mid, m_high = self._split_mask(X_oof)
        
        if m_low.sum() > 10:
            self.low_vol_model.fit_meta(oof_df[m_low], y_meta[m_low])
        else:
            self.low_vol_model.meta = self.mid_vol_model.meta
            
        if m_mid.sum() > 10:
            self.mid_vol_model.fit_meta(oof_df[m_mid], y_meta[m_mid])
        else:
            self.mid_vol_model.meta = self.low_vol_model.meta if m_low.sum() > 10 else self.high_vol_model.meta
            
        if m_high.sum() > 10:
            self.high_vol_model.fit_meta(oof_df[m_high], y_meta[m_high])
        else:
            self.high_vol_model.meta = self.mid_vol_model.meta

    def calibrate(self, oof_df: pd.DataFrame, y_meta: pd.Series, X_oof: pd.DataFrame = None):
        if X_oof is None:
            for model in [self.low_vol_model, self.mid_vol_model, self.high_vol_model]:
                model.xgb_clf.calibrate(oof_df["xgb_pred"].values, y_meta.values)
                model.lgb_clf.calibrate(oof_df["lgb_pred"].values, y_meta.values)
            return
            
        m_low, m_mid, m_high = self._split_mask(X_oof)
        
        for mask, model in [(m_low, self.low_vol_model), (m_mid, self.mid_vol_model), (m_high, self.high_vol_model)]:
            if mask.sum() >= 50:
                model.xgb_clf.calibrate(oof_df.loc[mask, "xgb_pred"].values, y_meta[mask].values)
                model.lgb_clf.calibrate(oof_df.loc[mask, "lgb_pred"].values, y_meta[mask].values)

    def predict(self, X: pd.DataFrame, tft_pred: Optional[Union[np.ndarray, float]] = None) -> dict:
        if isinstance(tft_pred, (float, int)):
            tft_pred = np.full(len(X), float(tft_pred))
            
        m_low, m_mid, m_high = self._split_mask(X)
        
        p_low = self.low_vol_model.predict(X[m_low], tft_pred[m_low] if tft_pred is not None else None) if m_low.sum() > 0 else None
        p_mid = self.mid_vol_model.predict(X[m_mid], tft_pred[m_mid] if tft_pred is not None else None) if m_mid.sum() > 0 else None
        p_high = self.high_vol_model.predict(X[m_high], tft_pred[m_high] if tft_pred is not None else None) if m_high.sum() > 0 else None
        
        res = {}
        for k in ["ensemble", "xgb", "lgb", "tft", "xgb_cal", "lgb_cal", "tft_cal"]:
            merged = pd.Series(index=X.index, dtype=float)
            if p_low is not None and k in p_low: merged.loc[m_low] = p_low[k]
            if p_mid is not None and k in p_mid: merged.loc[m_mid] = p_mid[k]
            if p_high is not None and k in p_high: merged.loc[m_high] = p_high[k]
            res[k] = merged.values
            
        return res

    def predict_meta(self, oof_df: pd.DataFrame, X_oof: pd.DataFrame) -> np.ndarray:
        m_low, m_mid, m_high = self._split_mask(X_oof)
        ensemble_prob = pd.Series(index=oof_df.index, dtype=float)
        
        if m_low.sum() > 0: ensemble_prob.loc[m_low] = self.low_vol_model.predict_meta(oof_df[m_low])
        if m_mid.sum() > 0: ensemble_prob.loc[m_mid] = self.mid_vol_model.predict_meta(oof_df[m_mid])
        if m_high.sum() > 0: ensemble_prob.loc[m_high] = self.high_vol_model.predict_meta(oof_df[m_high])
            
        return ensemble_prob.values

    def combined_importance(self) -> pd.DataFrame:
        df_low = self.low_vol_model.combined_importance().rename(columns={"mean": "mean_low"})
        df_mid = self.mid_vol_model.combined_importance().rename(columns={"mean": "mean_mid"})
        df_high = self.high_vol_model.combined_importance().rename(columns={"mean": "mean_high"})
        
        df = df_low[['mean_low']].join(df_mid[['mean_mid']], how='outer').join(df_high[['mean_high']], how='outer').fillna(0)
        df['mean'] = (df['mean_low'] + df['mean_mid'] + df['mean_high']) / 3
        return df.sort_values("mean", ascending=False)

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

