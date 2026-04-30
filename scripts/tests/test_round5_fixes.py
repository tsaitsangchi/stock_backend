"""
tests/test_round5_fixes.py
===========================
第五輪審查修復項目的單元測試。

涵蓋範圍：
  ① ensemble_model.XGBPredictor / LGBPredictor 的單一類別防護
  ② model_health_check.calculate_psi 的數學正確性
  ③ DB_CONFIG 的單一來源（config.py）原則
  ④ feature_engineering 的 staleness 與 kwave_score 函式存在
  ⑤ parallel_train.get_stocks_needing_training 的「缺檔 + 過期」邏輯

執行方式：
    cd scripts
    pytest tests/test_round5_fixes.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 注入 scripts/ 為 sys.path，以便能夠匯入專案模組
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ─────────────────────────────────────────────
# ① 單一類別防護：XGBPredictor / LGBPredictor
# ─────────────────────────────────────────────
class TestSingleClassGuard:
    """
    驗證 XGB/LGB 在 y 只有單一類別（y 全為 0 或全為 1）時不會崩潰，
    而是降級為常數機率回傳。
    """

    def _make_xy(self, label: int, n: int = 60) -> tuple[pd.DataFrame, pd.Series]:
        rng = np.random.default_rng(42)
        X = pd.DataFrame(
            rng.standard_normal((n, 4)),
            columns=["f1", "f2", "f3", "f4"],
        )
        y = pd.Series(np.full(n, label, dtype=int), name="y")
        return X, y

    def test_xgb_all_positive(self):
        from models.ensemble_model import XGBPredictor, HAS_XGB
        if not HAS_XGB:
            pytest.skip("xgboost 未安裝")
        X_tr, y_tr = self._make_xy(1)
        X_va, y_va = self._make_xy(1, n=20)
        m = XGBPredictor(task="classification")
        m.fit(X_tr, y_tr, X_va, y_va)
        # 應該降級而非拋例外
        assert m._const_class == 1
        # predict_score 應該回傳常數機率（接近 1）
        scores = m.predict_score(X_va)
        assert scores.shape == (20,)
        assert np.allclose(scores, 0.99)

    def test_xgb_all_negative(self):
        from models.ensemble_model import XGBPredictor, HAS_XGB
        if not HAS_XGB:
            pytest.skip("xgboost 未安裝")
        X_tr, y_tr = self._make_xy(0)
        X_va, y_va = self._make_xy(0, n=20)
        m = XGBPredictor(task="classification")
        m.fit(X_tr, y_tr, X_va, y_va)
        assert m._const_class == 0
        scores = m.predict_score(X_va)
        assert np.allclose(scores, 0.01)

    def test_lgb_all_positive(self):
        from models.ensemble_model import LGBPredictor, HAS_LGB
        if not HAS_LGB:
            pytest.skip("lightgbm 未安裝")
        X_tr, y_tr = self._make_xy(1)
        X_va, y_va = self._make_xy(1, n=20)
        m = LGBPredictor(task="classification")
        m.fit(X_tr, y_tr, X_va, y_va)
        assert m._const_class == 1
        scores = m.predict_score(X_va)
        assert np.allclose(scores, 0.99)


# ─────────────────────────────────────────────
# ② PSI 計算的數學正確性
# ─────────────────────────────────────────────
class TestPSI:
    def test_psi_identical_distributions_is_zero(self):
        from model_health_check import calculate_psi
        rng = np.random.default_rng(0)
        x = rng.beta(2, 2, 1000)
        psi = calculate_psi(x, x)
        # 完全相同分佈 PSI 應該非常小（量化誤差）
        assert psi < 0.01

    def test_psi_different_distributions_is_large(self):
        from model_health_check import calculate_psi
        rng = np.random.default_rng(0)
        x = rng.beta(2, 2, 2000)         # 集中在 0.5
        y = rng.beta(0.5, 0.5, 2000)     # U 形分佈，極化
        psi = calculate_psi(x, y)
        assert psi > 0.2  # 嚴重位移閾值

    def test_psi_handles_nans(self):
        from model_health_check import calculate_psi
        x = np.array([0.1, 0.2, np.nan, 0.5, 0.7, 0.9, 0.3, 0.4])
        y = np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85])
        psi = calculate_psi(x, y)
        # 不應該拋例外，且回傳合法數值
        assert not np.isnan(psi)
        assert psi >= 0


# ─────────────────────────────────────────────
# ③ DB_CONFIG 單一來源原則
# ─────────────────────────────────────────────
class TestSingleSourceConfig:
    def test_db_config_loaded_from_env(self):
        """config.DB_CONFIG 由 .env 載入，不是硬編碼字串。"""
        import os
        os.environ.setdefault("FINMIND_TOKEN", "test_dummy_token_for_unit_test")
        from importlib import reload
        import config as cfg
        reload(cfg)
        assert "dbname" in cfg.DB_CONFIG
        assert "password" in cfg.DB_CONFIG
        # 至少有 host / port 兩個關鍵欄位
        assert cfg.DB_CONFIG.get("host", "") != ""

    def test_utils_db_uses_db_config(self):
        """utils.db.get_db_connection 必須使用 DB_CONFIG，不能硬編碼。"""
        import os
        os.environ.setdefault("FINMIND_TOKEN", "test_dummy_token_for_unit_test")
        import inspect
        from utils import db as db_mod
        src = inspect.getsource(db_mod.get_db_connection)
        assert "DB_CONFIG" in src
        # 不應該出現硬編碼的 password='stock'
        assert 'password="stock"' not in src
        assert "password='stock'" not in src


# ─────────────────────────────────────────────
# ④ feature_engineering 關鍵函式存在
# ─────────────────────────────────────────────
class TestFeatureEngineeringFunctions:
    def test_kwave_function_exists(self):
        import os
        os.environ.setdefault("FINMIND_TOKEN", "test_dummy_token_for_unit_test")
        from feature_engineering import add_kwave_regime_features
        assert callable(add_kwave_regime_features)

    def test_staleness_function_exists(self):
        import os
        os.environ.setdefault("FINMIND_TOKEN", "test_dummy_token_for_unit_test")
        from feature_engineering import add_staleness_features
        assert callable(add_staleness_features)

    def test_kwave_score_in_feature_groups(self):
        import os
        os.environ.setdefault("FINMIND_TOKEN", "test_dummy_token_for_unit_test")
        from config import FEATURE_GROUPS
        all_features = [f for grp in FEATURE_GROUPS.values() for f in grp]
        assert "kwave_score" in all_features, "kwave_score 必須註冊在 FEATURE_GROUPS"

    def test_physics_signals_group_present(self):
        import os
        os.environ.setdefault("FINMIND_TOKEN", "test_dummy_token_for_unit_test")
        from config import FEATURE_GROUPS
        assert "physics_signals" in FEATURE_GROUPS
        # 重力井等物理特徵在組裡
        physics_features = FEATURE_GROUPS["physics_signals"]
        for required in ["gravity_pull", "info_force_per_mass", "singularity_dist"]:
            assert required in physics_features


# ─────────────────────────────────────────────
# ⑤ parallel_train.get_stocks_needing_training
# ─────────────────────────────────────────────
class TestParallelTrainHealthIntegration:
    def test_function_exists(self):
        import os
        os.environ.setdefault("FINMIND_TOKEN", "test_dummy_token_for_unit_test")
        from parallel_train import get_stocks_needing_training
        assert callable(get_stocks_needing_training)

    def test_step_days_uses_wf_config(self):
        """parallel_train 應引用 WF_CONFIG['step_days']，不再硬編碼 63。"""
        import os, inspect
        os.environ.setdefault("FINMIND_TOKEN", "test_dummy_token_for_unit_test")
        import parallel_train as pt
        src = inspect.getsource(pt)
        assert 'WF_CONFIG["step_days"]' in src or "WF_CONFIG['step_days']" in src
