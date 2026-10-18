import numpy as np
import pandas as pd

from eplanes.analysis.context import bh, holm
from eplanes.analysis.features import capacity_window, charge_phase_stats
from eplanes.config import load_settings

CFG = load_settings().raw
DT = 0.02


def test_holm_and_bh_known_values():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(holm(p), [0.03, 0.06, 0.06, 0.02])
    assert np.allclose(bh(p), [0.02, 0.04, 0.04, 0.02])


def test_capacity_window_recovers_known_capacity():
    t = np.arange(0, 90, DT)
    cur = np.full(len(t), -20.0)             # 20 A charging
    q_true = 24.0
    soc = np.clip(10 + (20 * t / 60) / q_true * 100, 0, 100)
    soc = np.round(soc)
    q, why, _ = capacity_window(t, soc, cur, +1, CFG, 0.1)
    assert why == "ok" and abs(q - q_true) / q_true < 0.05


def test_capacity_window_rejects_partial_coverage():
    t = np.arange(0, 10, DT)
    cur = np.full(len(t), -20.0)
    soc = np.round(30 + (20 * t / 60) / 24 * 100)
    q, why, _ = capacity_window(t, soc, cur, +1, CFG, 0.1)
    assert np.isnan(q) and why.startswith("window_coverage")


def _ccv(n_cc=60, n_tail=60):
    t = np.arange(0, n_cc + n_tail, DT)
    i = np.where(t < n_cc, 25.0, 25.0 * np.exp(-(t - n_cc) / 12.0))
    v = np.where(t < n_cc, 3600 + 10 * t, 4200.0)
    return t, i, v


def test_charge_phase_positive_control_finds_cc_cv():
    t, i, v = _ccv()
    r = charge_phase_stats(t, i + np.random.default_rng(0).normal(0, 0.1, len(t)), v, DT)
    assert r["has_cc"] and r["has_taper"]


def test_charge_phase_negative_control_smooth_ramp():
    t = np.arange(0, 80, DT)
    i = 28 * np.sin(np.pi * t / 80) ** 1.0
    r = charge_phase_stats(t, i, 3600 + 7 * t, DT)
    assert not r["has_cc"] and not r["has_taper"] and r["ramp_like"]
