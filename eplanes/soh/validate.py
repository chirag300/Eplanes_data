"""Tests of the physics label that need no ground truth.

V1  BMS step test      At each large BMS jump, did the voltage-anchored capacity move too? Same question for SOC-window capacity.
V2  Twin packs         Two packs of one aircraft share every flight and charge. How closely do their labels track, compared with the BMS?
V3  Cross-modality     Charge, flight and rest-to-rest capacity come from different data. Do they give the same fade?
V4  Voltage prediction Using capacity known BEFORE a session, predict the rest voltage AFTER it, from the measured Ah only.
                       A correct capacity predicts the battery's own future voltage; a capacity with false jumps cannot.
V5  Coulomb closure    Charge in over charge out per cycle. A drifting current sensor would look like fade, so this is checked.
V6  Curve shape        The label assumes the voltage curve keeps its shape with age. Capacity read from the lower and the upper half of the
                       voltage window must then fade alike. A gap warns that the shape is changing. An even stretch of the curve would pass
                       this test: only a reference capacity test can rule that out.
V0  Truth check        Simulated data only: error against the true capacity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .capacity import fit_ocv
from .common import Ctx, Log, write_json


def _rmse(a, b=0.0):
    e = np.asarray(a, float) - b
    return float(np.sqrt(np.nanmean(e ** 2)))


def theil(t, y):
    from scipy.stats import theilslopes
    ok = np.isfinite(t) & np.isfinite(y)
    if ok.sum() < 6 or np.ptp(t[ok]) <= 0:
        return (np.nan, np.nan, np.nan)
    r = theilslopes(y[ok], t[ok])
    return (r.slope, r.low_slope, r.high_slope)


def monthly(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.dropna(subset=[col, "start_datetime"])
    return g.assign(month=g.start_datetime.dt.to_period("M")).groupby("month")[col].median().reset_index()


def v0_truth(ctx: Ctx, lab: pd.DataFrame, log) -> dict:
    p = ctx.proc / "sim_truth.csv"
    if not p.exists():
        return {}
    lc = ctx.cfg["label"]
    tr = pd.read_csv(p)
    m = lab.merge(tr[["flight_id", "battery", "x_true", "soh_bms", "weak_true"]], on=["flight_id", "battery"])
    lo, hi = lc["settle_days"], lc["settle_days"] + lc["baseline_days"]
    base = m[(m.t_days >= lo) & (m.t_days <= hi)].groupby(["aircraft_label", "battery"]).x_true.median().rename("x_base")
    m = m.join(base, on=["aircraft_label", "battery"])
    m["true"] = 100 * m.x_true / m.x_base
    m = m[m.t_days >= lo].dropna(subset=["true"])
    m = m[~m.aircraft_label.isin(["VSW1280107"])]
    rows = []
    for k, g in m.groupby(["aircraft_label", "battery"]):
        r = dict(aircraft=k[0], pack=k[1], n=len(g), rmse_label=_rmse(g.soh_phys, g.true), rmse_causal=_rmse(g.soh_filt, g.true), rmse_bms=_rmse(g.soh_bms_median, g.true),
                 bias_label=float((g.soh_phys - g.true).mean()), cover_2sd=float((np.abs(g.soh_phys - g.true) <= 2 * g.soh_sd).mean()), sd_median=float(g.soh_sd.median()))
        for c in ("chg", "dis", "r2r"):
            if f"soh_{c}" in g:
                r[f"rmse_{c}"] = _rmse(g[f"soh_{c}"], g.true)
        if "q_soc_window_ah" in g and g.q_soc_window_ah.notna().sum() > 5:
            q = g.dropna(subset=["q_soc_window_ah"])
            b = q.q_soc_window_ah.iloc[: 30].median()
            r["rmse_soc_window"] = _rmse(100 * q.q_soc_window_ah / b, q.true)
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(ctx.tables / "V0_truth_check_SIMULATION.csv", index=False)
    log("[V0] SIMULATED data: error against true capacity (points)\n" + t.round(2).to_string(index=False))
    return dict(rmse_label=float(t.rmse_label.mean()), rmse_bms=float(t.rmse_bms.mean()), cover_2sd=float(t.cover_2sd.mean()))


def v1_bms_steps(ctx: Ctx, lab: pd.DataFrame, obs_cal: pd.DataFrame, log) -> dict:
    vc = ctx.cfg["validation"]
    W = vc["bms_side_sessions"]
    prim = obs_cal[obs_cal.modality.isin(ctx.cfg["label"]["modalities"])].groupby(["flight_id", "battery"]).y_c.median().rename("phys_meas").reset_index()
    d = lab.merge(prim, on=["flight_id", "battery"], how="left").sort_values(["aircraft_label", "battery", "start_datetime"])
    rows = []
    for (ac, b), g in d.groupby(["aircraft_label", "battery"], sort=False):
        g = g.reset_index(drop=True)
        base_q = g.q_soc_window_ah.dropna().iloc[:30].median() if g.q_soc_window_ah.notna().any() else np.nan
        s = g.soh_bms_median.to_numpy(float)
        ds = np.r_[np.nan, np.diff(s)]
        last = -10 ** 6
        for k in np.flatnonzero(np.abs(ds) >= vc["bms_jump_pts"]):
            if k - last < W:           # a burst of steps counts as one event
                continue
            last = k
            lo, hi = g.iloc[max(0, k - W):k], g.iloc[k:k + W]
            if hi.phys_meas.notna().sum() < 2 or lo.phys_meas.notna().sum() < 2:
                continue
            rows.append(dict(aircraft=ac, pack=b, when=g.start_datetime.iloc[k], flight_id=int(g.flight_id.iloc[k]), d_bms=float(hi.soh_bms_median.median() - lo.soh_bms_median.median()),
                             d_phys=100 * float(hi.phys_meas.median() - lo.phys_meas.median()),
                             d_soc_window=100 * float(hi.q_soc_window_ah.median() - lo.q_soc_window_ah.median()) / base_q if np.isfinite(base_q) else np.nan))
    t = pd.DataFrame(rows)
    t.to_csv(ctx.tables / "V1_bms_step_events.csv", index=False)
    if t.empty:
        log("[V1] no BMS jumps large enough")
        return {}
    t = t[np.abs(t.d_bms) >= vc["bms_jump_pts"]]

    def slope(x, y):
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3:
            return np.nan, np.nan, np.nan
        x, y = x[ok], y[ok]
        est = float(np.sum(x * y) / np.sum(x * x))
        rng = np.random.default_rng(1)
        boots = []
        for _ in range(1000):
            i = rng.integers(0, len(x), len(x))
            if np.sum(x[i] ** 2) > 0:
                boots.append(np.sum(x[i] * y[i]) / np.sum(x[i] ** 2))
        return est, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))

    out = dict(n_events=len(t))
    for c, nm in (("d_phys", "physics"), ("d_soc_window", "soc_window")):
        e, lo, hi = slope(t.d_bms.to_numpy(float), t[c].to_numpy(float))
        out[f"slope_{nm}"], out[f"slope_{nm}_lo"], out[f"slope_{nm}_hi"] = e, lo, hi
    log(f"[V1] {len(t)} BMS jumps of at least {vc['bms_jump_pts']} points. Fraction of each jump that also appears (1 = moves fully with the BMS, 0 = ignores it): "
        f"voltage-anchored {out['slope_physics']:.2f} [{out['slope_physics_lo']:.2f}, {out['slope_physics_hi']:.2f}]; SOC-window {out['slope_soc_window']:.2f} [{out['slope_soc_window_lo']:.2f}, {out['slope_soc_window_hi']:.2f}]")
    return out


def v2_twins(ctx: Ctx, lab: pd.DataFrame, log) -> dict:
    rows = []
    for ac, g in lab.groupby("aircraft_label"):
        p = g.pivot_table(index="flight_id", columns="battery", values=["soh_phys", "soh_bms_median", "start_datetime"], aggfunc="first")
        if p.empty or len(p) < 20 or 2 not in p["soh_phys"].columns:
            continue
        d = pd.DataFrame(dict(t=pd.to_datetime(p["start_datetime"][1]), a=p["soh_phys"][1].astype(float), b=p["soh_phys"][2].astype(float), ba=p["soh_bms_median"][1].astype(float), bb=p["soh_bms_median"][2].astype(float))).sort_values("t")
        step = d.resample("30D", on="t").median().dropna()
        if len(step) < 4:
            continue
        rows.append(dict(aircraft=ac, windows=len(step), rms_gap_label=_rmse(step.a - step.b - (step.a - step.b).median()), rms_gap_bms=_rmse(step.ba - step.bb - (step.ba - step.bb).median()),
                         corr_change_label=float(np.corrcoef(step.a.diff().dropna(), step.b.diff().dropna())[0, 1]), corr_change_bms=float(np.corrcoef(step.ba.diff().dropna(), step.bb.diff().dropna())[0, 1])))
    t = pd.DataFrame(rows)
    t.to_csv(ctx.tables / "V2_twin_pack_agreement.csv", index=False)
    if len(t):
        log("[V2] twin packs, 30-day medians (RMS gap after removing a constant offset, points)\n" + t.round(2).to_string(index=False))
    return dict(rms_gap_label=float(t.rms_gap_label.mean()), rms_gap_bms=float(t.rms_gap_bms.mean())) if len(t) else {}


def v3_modalities(ctx: Ctx, lab: pd.DataFrame, log) -> dict:
    cols = [(c, c.replace("soh_", "")) for c in ("soh_phys", "soh_chg", "soh_dis", "soh_r2r", "soh_chg_vmax", "soh_bms_median") if c in lab]
    rows = []
    for (ac, b), g in lab.groupby(["aircraft_label", "battery"]):
        if g.t_days.max() < 120:
            continue
        for c, nm in cols:
            m = monthly(g, c)
            if len(m) < 4:
                continue
            ty = (m.month.dt.start_time - m.month.dt.start_time.iloc[0]).dt.days.to_numpy() / 365.25
            sl, lo, hi = theil(ty, m[c].to_numpy(float))
            rows.append(dict(aircraft=ac, pack=b, indicator=nm, months=len(m), pts_per_year=sl, ci_low=lo, ci_high=hi))
    t = pd.DataFrame(rows)
    t.to_csv(ctx.tables / "V3_fade_by_indicator.csv", index=False)
    if len(t):
        log("[V3] fade on monthly medians (points per year)\n" + t.pivot_table(index=["aircraft", "pack"], columns="indicator", values="pts_per_year").round(1).to_string())
    piv = {}
    a, b = [c for c in ("soh_chg", "soh_r2r") if c in lab], None
    if len(a) == 2:
        d = lab.dropna(subset=a)
        d = d[d.groupby(["aircraft_label", "battery"]).soh_chg.transform("size") > 30]
        piv["rms_gap_chg_vs_r2r"] = _rmse(d[a[0]] - d[a[1]])
    return piv


def v4_voltage_prediction(ctx: Ctx, lab: pd.DataFrame, log) -> dict:
    """Predict the rest voltage at the start of the next session from the previous one and the Ah moved between them."""
    p = ctx.cache / "rest_pairs.csv"
    if not p.exists():
        return {}
    pairs = pd.read_csv(p, parse_dates=["start_datetime"])
    if len(pairs) < 30:
        return {}
    rc = ctx.cfg["registration"]
    use = [c for c in ("soh_f_chg", "soh_filt") if c in lab]
    phys_col = use[0]
    k = lab[["flight_id", "battery", phys_col, "soh_bms_median"]].rename(columns={phys_col: "phys"})
    d = pairs.merge(k, on=["flight_id", "battery"], how="inner").dropna(subset=["phys", "soh_bms_median"])
    d["fleet_fit"] = d.aircraft_label.isin(lab[lab.fleet.astype(str) == str(ctx.cfg["label"]["fit_fleet"])].aircraft_label.unique())
    if d.fleet_fit.sum() < 30:
        d["fleet_fit"] = True
    d = d.reset_index(drop=True)
    res, rows = {}, []
    for name, c in (("constant_capacity", np.full(len(d), 1.0)), ("reported_soh", d.soh_bms_median.to_numpy() / 100.0), ("physics_label_causal", d.phys.to_numpy() / 100.0)):
        f, _ = fit_ocv(d[d.fleet_fit].reset_index(drop=True), rc["r2r_knot_v"], iters=1, s_init=c[d.fleet_fit.to_numpy()])
        pred = f.inverse(f(d.v_a.to_numpy()) + d.dq.to_numpy() / np.maximum(c, 0.2))
        e = (pred - d.v_c.to_numpy()) * 1000.0
        ok = np.abs(e) < 300
        res[name] = dict(rmse_mv=_rmse(e[ok]), mae_mv=float(np.mean(np.abs(e[ok]))))
        for fleet_fit, g in pd.DataFrame(dict(e=e, f=d.fleet_fit, ac=d.aircraft_label)).groupby("ac"):
            rows.append(dict(aircraft=fleet_fit, capacity_source=name, n=len(g), rmse_mv=_rmse(g.e[np.abs(g.e) < 300])))
    t = pd.DataFrame(rows)
    t.to_csv(ctx.tables / "V4_voltage_prediction.csv", index=False)
    log("[V4] rest voltage after a session predicted from capacity known before it (RMSE, mV per cell)\n" + t.pivot_table(index="aircraft", columns="capacity_source", values="rmse_mv").round(1).to_string())
    return {k: v["rmse_mv"] for k, v in res.items()}


def v5_closure(ctx: Ctx, log) -> dict:
    obs = pd.read_parquet(ctx.cache / "session_observables_r.parquet")
    u = obs[obs.usable_electrical].sort_values(["aircraft_label", "battery", "start_datetime"])
    rows = []
    for (ac, b), g in u.groupby(["aircraft_label", "battery"]):
        g = g.reset_index(drop=True)
        full = g.index[(g.charge_min >= 5) & (g.v_max_cell > g.v_max_cell.quantile(0.5))].to_numpy()
        for a, c in zip(full[:-1], full[1:]):
            mid = g.iloc[a + 1:c + 1]
            if len(mid) < 2 or len(mid) > 4:
                continue
            ain, aout = mid.ah_in.sum(), mid.ah_out.sum()
            if aout >= 5:
                rows.append(dict(aircraft=ac, pack=b, when=g.start_datetime.iloc[c], ratio=ain / aout))
    t = pd.DataFrame(rows)
    if t.empty:
        return {}
    t["half"] = t.when.dt.year.astype(str) + "H" + ((t.when.dt.month > 6) + 1).astype(str)
    s = t.groupby(["aircraft", "pack", "half"]).ratio.agg(["median", "size"]).reset_index()
    s.to_csv(ctx.tables / "V5_coulomb_closure.csv", index=False)
    log(f"[V5] charge in / charge out per cycle: median {t.ratio.median():.4f}, range of half-year medians {s['median'].min():.4f}-{s['median'].max():.4f}")
    return dict(closure_median=float(t.ratio.median()), closure_min=float(s["median"].min()), closure_max=float(s["median"].max()))


def v6_shape(ctx: Ctx, log) -> dict:
    long = pd.read_parquet(ctx.cache / "capacity_observations.parquet")
    d = long[long.modality == ctx.cfg["label"]["primary"]]
    if not {"s_low", "s_high"} <= set(d.columns):
        return {}
    rows = []
    for (ac, b), g in d.groupby(["aircraft_label", "battery"]):
        g = g.sort_values("start_datetime")
        if len(g) < 30 or (g.start_datetime.iloc[-1] - g.start_datetime.iloc[0]).days < 120:
            continue
        ty = (g.start_datetime - g.start_datetime.iloc[0]).dt.total_seconds().to_numpy() / 86400 / 365.25
        out = dict(aircraft=ac, pack=b)
        for c in ("s_low", "s_high", "s"):
            y = 100 * g[c].to_numpy(float) / np.nanmedian(g[c].to_numpy(float)[: 20])
            out[f"{c}_pts_per_year"] = theil(ty[np.isfinite(y)], y[np.isfinite(y)])[0]
        rows.append(out)
    t = pd.DataFrame(rows)
    t.to_csv(ctx.tables / "V6_curve_shape_check.csv", index=False)
    if t.empty:
        return {}
    gap = (t.s_low_pts_per_year - t.s_high_pts_per_year).abs()
    log("[V6] fade read from the lower and upper half of the voltage window (points per year)\n" + t.round(2).to_string(index=False))
    return dict(mean_gap_pts_per_year=float(gap.mean()), max_gap_pts_per_year=float(gap.max()))


def run_validate(ctx: Ctx, log=None) -> dict:
    log = log or Log(ctx)
    lab = pd.read_parquet(ctx.cache / "labels.parquet")
    obs_cal = pd.read_parquet(ctx.cache / "observations_calibrated.parquet")
    out = {}
    out["V0"] = v0_truth(ctx, lab, log)
    out["V1"] = v1_bms_steps(ctx, lab, obs_cal, log)
    out["V2"] = v2_twins(ctx, lab, log)
    out["V3"] = v3_modalities(ctx, lab, log)
    out["V4"] = v4_voltage_prediction(ctx, lab, log)
    out["V5"] = v5_closure(ctx, log)
    out["V6"] = v6_shape(ctx, log)
    write_json(ctx.out / "validation_summary.json", out)
    return out
