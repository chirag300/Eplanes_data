import numpy as np

from eplanes.config import load_settings
from eplanes.quality.init_artifacts import settle_info

CQ = load_settings().raw["quality"]
t = np.arange(0, 6, 0.02)


def test_zero_fill_detected():
    x = np.full(len(t), 25.0) + np.random.default_rng(0).normal(0, 0.05, len(t))
    x[0] = 0.0
    r = settle_info(t, x, CQ, "rest_start", "bat_1_max_cell_temp")
    assert r["cls"] == "zero_fill" and r["settle_idx"] == 1


def test_bin_average_detected():
    x = np.full(len(t), 16.0)
    x[0], x[1] = 0.0, 16.0 * 2 / 3
    r = settle_info(t, x, CQ, "rest_start", "bat_1_max_cell_temp")
    assert r["cls"] == "bin_avg_zero"


def test_stable_signal_not_flagged():
    x = np.full(len(t), 3600.0) + np.random.default_rng(1).normal(0, 1, len(t))
    r = settle_info(t, x, CQ, "rest_start", "bat_1_min_cell_volt")
    assert r["cls"] == "stable" and r["settle_idx"] == 0


def test_real_trend_is_not_start_up():
    x = 90.0 - 0.5 * t + np.random.default_rng(2).normal(0, 0.05, len(t))
    r = settle_info(t, x, CQ, "rest_start", "bat_1_max_cell_temp")
    assert r["settle_idx"] == 0


def test_load_start_never_flags_physical_transient():
    x = np.where(t < 1.0, 160.0, 70.0).astype(float)
    r = settle_info(t, x, CQ, "load_start", "bat_1_current")
    assert r["cls"] in ("load_onset", "stable")


def test_slow_relaxation_is_monotone_settling():
    x = 3600 - 40 * np.exp(-t / 0.15) + np.random.default_rng(3).normal(0, 0.3, len(t))
    r = settle_info(t, x, CQ, "rest_start", "bat_1_min_cell_volt")
    assert r["cls"] == "monotone_settling" and r["settle_idx"] > 5


def test_on_grid_integer_values_are_not_bin_average():
    x = np.full(len(t), 21.0)
    x[0], x[1] = 0.0, 18.0          # 18/21 = 6/7 but on the 1-degree grid
    r = settle_info(t, x, CQ, "rest_start", "bat_1_max_cell_temp")
    assert r["cls"] != "bin_avg_zero"


def test_gps_clock_hhmmss():
    from eplanes.build.clean import gps_clock
    tm = np.arange(0, 30, 0.1 / 60)                    # 10 Hz for 30 min
    sec = 12 * 3600 + 42 * 60 + 20 + np.floor(tm * 60)  # starts 12:42:20
    ts = (sec // 3600) * 10000 + (sec % 3600) // 60 * 100 + sec % 60
    r = gps_clock(tm, ts, np.ones(len(tm), bool))
    assert abs(r["gps_tod_start_s"] - (12 * 3600 + 42 * 60 + 20)) <= 1 and abs(r["gps_clock_slope"] - 1) < 1e-3
