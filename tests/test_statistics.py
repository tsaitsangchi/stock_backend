import numpy as np
from scripts.model_health_check import calculate_psi

def test_psi_identical_distribution():
    # 相同分佈，PSI 應接近 0
    ref = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    act = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    # 由於函式內部會轉直方圖，我們直接餵模擬分佈
    psi = calculate_psi(ref, act)
    assert psi < 0.01

def test_psi_drifted_distribution():
    # 明顯漂移，PSI 應較大
    ref = np.random.normal(0.5, 0.1, 1000).clip(0, 1)
    act = np.random.normal(0.7, 0.1, 1000).clip(0, 1)
    psi = calculate_psi(ref, act)
    assert psi > 0.1
