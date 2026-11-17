"""Tests of the SOH pipeline. Each one guards a specific way the science could go wrong without any error being raised."""
import json

import numpy as np
import pandas as pd
import pytest

from eplanes.soh.capacity import OcvFunction, Reference, fit_ocv
from eplanes.soh.common import load_ctx
from eplanes.soh.dataset import BMS, CURVE, HISTORY, STRESS, LabelEngine, feature_list
from eplanes.soh.label import Hyper, Series, detect_steps, fit_hyper, kalman
from eplanes.soh.observables import charger_step, extract_session, takeoff_ramp
from eplanes.soh.simulate import ocv

EC = dict(n_series=96, step_from_min_a=15.0, step_min_di_a=5.0, ramp_min_di_a=30.0, ramp_window_s=20.0)
LC = dict(step_window=8, step_z=6.0, step_min_pts=2.0, max_steps_per_pack=4)


def _series(n=200, seed=0, sigma=0.004, step_at=None, step=0.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 3.0
    truth = 1 - 0.0001 * t
    if step_at:
        truth[step_at:] += step
    y = truth + rng.normal(0, sigma, n)
    lam = np.full(n, 1 / sigma ** 2)
    return Series(("a", 1), t, np.arange(n) * 14.0, lam, y * lam, np.zeros(n)), truth


def test_kalman_tracks_the_true_state():
    s, truth = _series()
    _, _, xs, ps, _ = kalman(s, Hyper(0.02, 0.01, 1e-6), 30.0)
    assert np.sqrt(np.mean((xs - truth) ** 2)) < 0.002
    assert np.all(ps > 0)


def test_leave_one_out_equals_really_deleting_the_session():
    """The closed form used for the label must match rerunning the smoother without the session."""
    s, _ = _series(n=60)
    h = Hyper(0.02, 0.01, 1e-6)
    _, _, xs, ps, _ = kalman(s, h, 30.0)
    k = 30
    prec = 1 / ps[k]
    loo = (xs[k] * prec - s.eta[k]) / (prec - s.lam[k])
    lam, eta = s.lam.copy(), s.eta.copy()
    lam[k], eta[k] = 0.0, 0.0
    _, _, xd, _, _ = kalman(Series(s.key, s.t, s.ah, lam, eta, s.extra_q), h, 30.0)
    assert loo == pytest.approx(xd[k], abs=1e-9)


def test_step_detector_finds_a_real_step_and_ignores_noise():
    h = Hyper(0.0, 0.0, 1e-6)
    s, _ = _series(step_at=100, step=0.04)
    found = detect_steps(s, h, 30.0, LC)
    assert len(found) == 1 and abs(found[0]["k"] - 100) <= 3 and found[0]["delta"] == pytest.approx(0.04, abs=0.01)
    for seed in range(5):
        assert detect_steps(_series(seed=seed)[0], h, 30.0, LC) == []


def test_pack_without_label_readings_does_not_crash():
    """Real data: VSW1280107 has flight readings only, which are not in the label."""
    from eplanes.soh.label import robust_pass
    grid = pd.DataFrame(dict(flight_id=[1, 2, 3], aircraft_label="a", battery=1, t_days=[0.0, 1.0, 2.0], ah_cum=[0.0, 10.0, 20.0]))
    d = pd.DataFrame(dict(flight_id=[1, 2], modality="dis", y_c=[1.0, 1.01]), index=[7, 8])
    params = {"dis": dict(sigma=0.05), "chg": dict(sigma=0.02)}
    s, w = robust_pass(grid, d, params, ["chg"], Hyper(), 30.0, 4.0, 3)
    assert (s.lam == 0).all() and (w == 1.0).all()


def test_charger_step_recovers_resistance_at_constant_state():
    n = 400
    i = np.r_[np.full(200, -23.0), np.full(200, -9.5)]
    r_cell = 2.6e-3
    v = 96 * (3.95 + np.linspace(0, 0.01, n) + (-i) * r_cell)
    out = charger_step(i, v, np.full(n, 25.0), np.linspace(0, 8, n), (0, n), EC)
    assert out["step_found"] and out["r_step_mohm"] == pytest.approx(2.6, abs=0.05)


def test_no_step_is_invented_on_a_smooth_ramp():
    n = 800
    i = -np.linspace(27, 23, n)
    v = 96 * (3.7 + np.linspace(0, 0.3, n))
    assert not charger_step(i, v, np.full(n, 25.0), np.linspace(0, 12, n), (0, n), EC)["step_found"]


def test_charge_starting_at_time_zero_is_kept():
    """Zero is a valid start time; it must not read as missing."""
    from eplanes.soh.capacity import charge_samples
    n = 120
    t = np.arange(n) * 10.0
    c = {"1_t": t, "1_i": np.full(n, -25.0), "1_q": t * 25 / 3600, "1_v": np.linspace(3.7, 4.0, n), "1_seg": np.zeros(n, int)}
    rc = dict(charge_skip_s=180.0, charge_i_min_a=15.0)
    got = charge_samples(c, "1", dict(charge_start_s=0.0, charge_stop_s=1190.0), 0.0025, rc)
    assert got is not None and len(got[0]) > 50


def test_takeoff_ramp_recovers_resistance():
    i = np.r_[np.full(100, 4.0), np.linspace(4, 100, 10), np.full(100, 100.0)]
    v = 96 * (4.1 - i * 2.8e-3)
    out = takeoff_ramp(i, v, np.full(len(i), 20.0), EC)
    assert out["ramp_found"] and out["r_ramp_mohm"] == pytest.approx(2.8, abs=0.05)


def test_registration_recovers_scale_from_any_partial_window():
    """Equation (1): sessions covering different windows of one curve must give the capacity they were built with."""
    edges = np.arange(3.5, 4.1, 0.005)
    th = np.linspace(0, 1, 3000)
    v = ocv(th)
    q = lambda vv: np.interp(vv, v, th) * 28.0        # Ah from empty at scale 1
    rng = np.random.default_rng(0)
    scales = rng.uniform(0.8, 1.0, 80)
    M = np.full((80, len(edges)), np.nan)
    for k, s in enumerate(scales):
        a, b = rng.uniform(3.62, 3.72), rng.uniform(3.98, 4.05)
        m = (edges >= a) & (edges <= b)
        M[k, m] = s * q(edges[m]) + rng.uniform(0, 5)
    early = np.ones(80, bool)
    ref = Reference(edges).fit(M, 4, early, 4.0)
    sc = ref.register(M, 4.0)
    est = sc.s.to_numpy() / np.nanmedian(sc.s.to_numpy()) * np.median(scales)
    assert np.nanmax(np.abs(est - scales)) < 0.01


def test_ocv_self_calibration_needs_no_soc():
    th = np.linspace(0, 1, 3000)
    v = ocv(th)
    rng = np.random.default_rng(1)
    rows = []
    for k in range(300):
        s = 1 - 0.15 * k / 300
        lo, hi = rng.uniform(0.25, 0.55), rng.uniform(0.9, 1.0)
        rows.append(dict(aircraft_label="a", battery=1, v_a=np.interp(hi, th, v), v_c=np.interp(lo, th, v), dq=-(hi - lo) * 28 * s))
    p = pd.DataFrame(rows)
    f, raw = fit_ocv(p, 0.04)
    ratio = raw / np.nanmedian(raw[:30])
    true = (1 - 0.15 * np.arange(300) / 300)
    assert np.nanmean(np.abs(ratio - true / true[:30].mean())) < 0.012


def test_rest_pairs_skip_short_rest_after_a_charge():
    from eplanes.soh.capacity import rest_pairs
    rc = dict(r2r_min_rest_min=20.0, r2r_max_rest_days=21.0, r2r_soc_gate_pts=3.0, r2r_min_ah=5.0, r2r_min_rest_after_charge_min=240.0)
    t0 = pd.Timestamp("2026-01-01 10:00")
    def sess(fid, start, q_end):
        return dict(flight_id=fid, aircraft_label="a", battery=1, usable_electrical=True, start_datetime=start, duration_min=60.0, rest_start_v=3.8,
                    rest_start_i_a=0.0, r_eff=0.0025, rest_start_q_ah=0.0, q_end_ah=q_end, soc_first=50.0, soc_last=50.0, n_segments_q=1)
    for q_end, gap_h, kept in ((15.0, 2, False), (15.0, 6, True), (-15.0, 2, True)):   # charge then 2 h, charge then 6 h, flight then 2 h
        obs = pd.DataFrame([sess(1, t0, q_end), sess(2, t0 + pd.Timedelta(hours=1 + gap_h), 0.0)])
        assert (len(rest_pairs(obs, rc)) == 1) == kept, (q_end, gap_h)


def test_process_noise_respects_its_cap():
    s, _ = _series(n=80)
    s.eta[::7] += 0.08 * s.lam[::7]          # gross outliers would push an uncapped fit to a large q
    assert fit_hyper([s], 30.0, q_max=2e-6).q <= 2e-6 * 1.0001


def test_ocv_function_inverts():
    f = OcvFunction(np.array([3.5, 3.7, 3.9, 4.1]), np.array([5.0, 10.0, 8.0]))
    v = np.array([3.55, 3.8, 4.05])
    assert f.inverse(f(v)) == pytest.approx(v, abs=1e-3)


def test_feature_ladder_keeps_contaminants_out():
    cfg = load_ctx().cfg
    cols = BMS + CURVE + HISTORY + ["temp_mean_c", "r_step25_mohm", "ah_out"]
    df = pd.DataFrame(np.ones((2, len(cols))), columns=cols)
    assert not set(feature_list(cfg, "run_B", df)) & set(BMS)
    assert not set(feature_list(cfg, "run_C", df)) & (set(BMS) | set(HISTORY))
    assert not set(feature_list(cfg, "run_D", df)) & (set(BMS) | set(HISTORY) | set(CURVE))


def test_future_voltages_cannot_reach_a_forecaster():
    """Planned-usage inputs must not contain anything measured from the battery's own response."""
    assert not [c for c in STRESS if "v_" in c or "volt" in c or c.startswith("y_") or "rest" in c]


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    from eplanes.soh.label import run_label
    from eplanes.soh.observables import run_extract
    from eplanes.soh.simulate import make_sim
    d = tmp_path_factory.mktemp("soh")
    make_sim(d / "data", 0.1, log=lambda *a: None)
    c = load_ctx(d / "data", d / "out")
    q = lambda *a: None
    run_extract(c, q)
    run_label(c, q)
    return c


def test_label_beats_bms_on_known_truth(ctx):
    from eplanes.soh.validate import v0_truth
    r = v0_truth(ctx, pd.read_parquet(ctx.cache / "labels.parquet"), lambda *a: None)
    assert r["rmse_label"] < 1.5 and r["rmse_bms"] > 4 * r["rmse_label"]


def test_label_ignores_bms_jumps_but_soc_window_follows_them(ctx):
    from eplanes.soh.validate import v1_bms_steps
    lab = pd.read_parquet(ctx.cache / "labels.parquet")
    r = v1_bms_steps(ctx, lab, pd.read_parquet(ctx.cache / "observations_calibrated.parquet"), lambda *a: None)
    assert abs(r["slope_physics"]) < 0.15 and r["slope_soc_window"] > 0.7


def test_extraction_is_blind_to_reported_soh_and_soc(ctx):
    """Scrambling every BMS output must not change any physics measurement."""
    import pyarrow.parquet as pq
    from eplanes.soh.observables import COLS
    g = pq.ParquetFile(ctx.proc / "analysis_ready_timeseries.parquet").read_row_group(1, columns=["flight_id"] + COLS).to_pandas()
    a, ca = extract_session(g, ctx.settings.raw, ctx.cfg)
    h = g.copy()
    rng = np.random.default_rng(0)
    for b in (1, 2):
        for c in ("soc", "soh"):
            h[f"bat_{b}_{c}"] = rng.uniform(1, 100, len(h))
    z, cz = extract_session(h, ctx.settings.raw, ctx.cfg)
    for k in ("r_step_mohm", "ah_in", "ah_out", "rest_start_v", "charge_min"):
        x, y = a["packs"]["1"].get(k), z["packs"]["1"].get(k)
        assert x == y or (x is not None and np.isnan(x) and np.isnan(y))
    for k in ("q", "v", "i"):
        assert np.array_equal(ca["1"][k], cz["1"][k], equal_nan=True)


def test_labels_before_a_cutoff_do_not_see_the_future(ctx):
    e = LabelEngine(ctx)
    key = next(k for k in e.grids if len(e.grids[k]) > 60)
    grid = e.grids[key]
    cut = grid.start_datetime.iloc[len(grid) // 2]
    before = e.upto(key, cut).copy()
    late = grid.flight_id[grid.start_datetime > cut]
    d = e.d[key].copy()
    d.loc[d.flight_id.isin(late), "y_c"] = 0.3            # destroy everything after the cutoff
    e.d[key] = d
    e._cache.clear()
    after = e.upto(key, cut)
    assert np.allclose(before.x_s, after.x_s) and np.allclose(before.x_f, after.x_f)


def test_held_out_fleet_is_scored_once(ctx, monkeypatch):
    import eplanes.soh.evaluate as ev
    (ctx.out / "FINAL_TEST_LOCK.json").write_text(json.dumps(dict(first_run_utc="2026-01-01", runs=1)))
    with pytest.raises(SystemExit, match="already scored"):
        ev.run_final(ctx, log=lambda *a: None)
    (ctx.out / "FINAL_TEST_LOCK.json").unlink()
