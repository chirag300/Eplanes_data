"""The physics SOH label: a state-space smoother over the capacity measurements.

State, one per pack and session, relative capacity x_k:
    x_{k+1} = x_k - mu_k + w_k
    mu_k = a [sqrt(tau_{k+1}) - sqrt(tau_k)] + b dAh_k      calendar plus cycling fade
    w_k ~ N(0, q dt_k)  (plus a large variance at an accepted step)
Observations, one per measurement:
    y_mk = x_k + beta_m' z_mk + e_mk                         z = temperature and window covariates
Solved with a Kalman filter and RTS smoother; Student-t weights make single bad sessions harmless.

Why this form
  * The gap-aware process noise lets the state move further across long idle periods and little between same-day sessions.
  * Fade is the prior, not a constraint. If the measurements show a sustained rise the smoother follows it and the step is
    reported, because a pack's usable capacity can truly rise (balancing, a replaced module). Whether a rise is real is an
    empirical question for the maintenance log, not something to assume away.
  * Each session gets a smoothed value, a causal (filtered) value, and a leave-one-out value that excludes that session's own
    measurements. Models that take features from a session are trained on the leave-one-out value, so the label cannot leak
    the session's own curve back into its inputs.

All hyperparameters come from the fit fleet. Other aircraft are only passed through the fitted model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

from .common import Ctx, Log, mad_sigma, write_json
from .capacity import run_capacity

DAY = 86400.0


@dataclass
class Hyper:
    a: float = 0.02           # per sqrt(year)
    b: float = 0.01           # per 1000 Ah discharged
    q: float = 1e-6           # per day
    extra: dict = field(default_factory=dict)


@dataclass
class Series:
    """One pack on its session grid; obs are merged per session into a precision and an information term."""
    key: tuple
    t: np.ndarray             # days since the pack's first session
    ah: np.ndarray            # cumulative discharged Ah (logged sessions)
    lam: np.ndarray           # sum of 1/R at each session (0 = no measurement)
    eta: np.ndarray           # sum of y/R at each session
    extra_q: np.ndarray       # additional process variance entering session k (accepted steps)


def _basis(s: Series, tau0: float):
    root = np.sqrt((s.t + tau0) / 365.25)
    return np.r_[0.0, np.diff(root)], np.r_[0.0, np.diff(s.ah)] / 1000.0, np.r_[0.0, np.diff(s.t)]


def kalman(s: Series, h: Hyper, tau0: float, x0: float = 1.0, p0: float = 0.04 ** 2, smooth: bool = True):
    """Scalar Kalman filter and RTS smoother. Returns x_f, P_f, x_s, P_s, log-likelihood."""
    b1, b2, dt = _basis(s, tau0)
    mu = h.a * b1 + h.b * b2
    qq = h.q * dt + s.extra_q
    n = len(s.t)
    xp, pp, xf, pf = np.empty(n), np.empty(n), np.empty(n), np.empty(n)
    x, p, ll = x0, p0, 0.0
    lam, eta = s.lam, s.eta
    for k in range(n):
        if k:
            x, p = x - mu[k], p + qq[k]
        xp[k], pp[k] = x, p
        if lam[k] > 0:
            r = 1.0 / lam[k]
            y = eta[k] * r
            sv = p + r
            e = y - x
            ll += -0.5 * (np.log(2 * np.pi * sv) + e * e / sv)
            g = p / sv
            x, p = x + g * e, (1 - g) * p
        xf[k], pf[k] = x, p
    if not smooth:
        return xf, pf, None, None, ll
    xs, ps = xf.copy(), pf.copy()
    for k in range(n - 2, -1, -1):
        c = pf[k] / max(pp[k + 1], 1e-14)
        xs[k] = xf[k] + c * (xs[k + 1] - xp[k + 1])
        ps[k] = pf[k] + c * c * (ps[k + 1] - pp[k + 1])
    return xf, pf, xs, ps, ll


def fit_hyper(series: list[Series], tau0: float, start: Hyper | None = None, q_max: float = 2e-6) -> Hyper:
    """Maximum likelihood for (a, b, q) on the fit fleet. q is capped: left free, real-data outliers drive it to its limit and the
    label then follows noise (first real run, Nov 2026)."""
    start = start or Hyper()

    def nll(p):
        h = Hyper(np.exp(p[0]), np.exp(p[1]), np.exp(p[2]))
        return -sum(kalman(s, h, tau0, smooth=False)[4] for s in series)

    x0 = np.log([max(start.a, 1e-3), max(start.b, 1e-3), max(start.q, 1e-8)])
    bounds = [(np.log(1e-5), np.log(0.3)), (np.log(1e-5), np.log(0.3)), (np.log(1e-9), np.log(q_max))]
    r = minimize(nll, x0, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=60, eps=1e-2))
    return Hyper(float(np.exp(r.x[0])), float(np.exp(r.x[1])), float(np.exp(r.x[2])))


def local_residual(df: pd.DataFrame, col: str, window: int = 21) -> pd.Series:
    """Deviation of a measurement from its own neighbours in time, within a pack and modality."""
    r = pd.Series(np.nan, index=df.index)
    for _, g in df.sort_values("start_datetime").groupby(["aircraft_label", "battery", "modality"], sort=False):
        r.loc[g.index] = g[col] - g[col].rolling(window, center=True, min_periods=5).median()
    return r


def normalise(long: pd.DataFrame, lc: dict) -> pd.DataFrame:
    """Divide each modality by its own early-life level, per pack, after skipping the break-in period."""
    d = long.dropna(subset=["s", "start_datetime"]).copy()
    first = d.groupby(["aircraft_label", "battery"]).start_datetime.transform("min")
    d["t_days"] = (d.start_datetime - first).dt.total_seconds() / DAY
    lo, hi = lc["settle_days"], lc["settle_days"] + lc["baseline_days"]
    rows = []
    for (ac, b, m), g in d.groupby(["aircraft_label", "battery", "modality"], sort=False):
        w = g[(g.t_days >= lo) & (g.t_days <= hi)]
        if len(w) < 5:
            w = g.sort_values("t_days").head(max(5, len(g) // 5))
        rows.append(dict(aircraft_label=ac, battery=b, modality=m, base=float(w.s.quantile(lc["baseline_q"])), n_base=len(w)))
    base = pd.DataFrame(rows)
    d = d.merge(base, on=["aircraft_label", "battery", "modality"], how="left")
    d["y"] = d.s / d.base
    return d


def calibrate(d: pd.DataFrame, fit_mask: pd.Series, lc: dict, log) -> tuple[pd.DataFrame, dict]:
    """Remove temperature and measurement-window effects, per modality.

    Estimated from how each measurement departs from its neighbours in time, so slow fade and season cannot be mistaken for a
    temperature effect. Also returns each modality's noise level."""
    d = d.copy()
    params = {}
    d["e_local"] = local_residual(d, "y")
    for m, g in d.groupby("modality"):
        fit = g[fit_mask.reindex(g.index).fillna(False)]
        z = {"temp": g.temp_mean_c - lc["t_ref_c"]}
        for c in ("v_lo", "v_hi"):
            if c in g and g[c].notna().sum() > 20 and g[c].std() > 0:
                z[c] = g[c] - g[c].median()
        Z = pd.DataFrame(z).fillna(0.0)
        use = fit.index[fit.e_local.notna()]
        beta = np.zeros(Z.shape[1])
        if len(use) >= 30:
            e = fit.loc[use, "e_local"].to_numpy()
            ok = np.abs(e) < 4 * max(mad_sigma(e), 1e-4)
            A = Z.loc[use].to_numpy()[ok]
            A = A - A.mean(0)
            scale = A.std(0) + 1e-9
            beta = np.linalg.solve((A / scale).T @ (A / scale) + 1.0 * np.eye(A.shape[1]), (A / scale).T @ e[ok]) / scale
        d.loc[g.index, "y_c"] = g.y - Z.to_numpy() @ beta
        params[m] = dict(beta=dict(zip(Z.columns, beta.tolist())))
    d["e_local"] = local_residual(d, "y_c")
    for m, g in d.groupby("modality"):
        fit = g[fit_mask.reindex(g.index).fillna(False)]
        s = mad_sigma(fit.e_local.to_numpy()) if fit.e_local.notna().sum() >= 20 else mad_sigma(g.e_local.to_numpy())
        params[m]["sigma"] = float(max(s if np.isfinite(s) else 0.02, lc["sigma_floor"]))
        params[m]["n"] = int(len(g))
    log("[label] per-session noise: " + ", ".join(f"{m} {100 * p['sigma']:.2f}%" for m, p in params.items()))
    return d, params


def make_series(grid: pd.DataFrame, d: pd.DataFrame, params: dict, mods: list[str], weights: pd.Series | None = None) -> Series:
    """Merge the measurements of the chosen modalities onto a pack's session grid."""
    pos = {f: i for i, f in enumerate(grid.flight_id)}
    lam, eta = np.zeros(len(grid)), np.zeros(len(grid))
    g = d[d.modality.isin(mods)]
    for m, f, y, idx in zip(g.modality, g.flight_id, g.y_c, g.index):
        k = pos.get(f)
        if k is None or not np.isfinite(y):
            continue
        w = 1.0 if weights is None else float(weights.get(idx, 1.0))
        lam[k] += w / params[m]["sigma"] ** 2
        eta[k] += w * y / params[m]["sigma"] ** 2
    return Series((grid.aircraft_label.iloc[0], int(grid.battery.iloc[0])), grid.t_days.to_numpy(float), grid.ah_cum.to_numpy(float), lam, eta, np.zeros(len(grid)))


def robust_pass(s_grid, d, params, mods, h, tau0, nu, iters, extra_q=None):
    """Kalman smoothing with Student-t reweighting of the individual measurements."""
    weights = pd.Series(1.0, index=d.index)
    s = make_series(s_grid, d, params, mods)
    if extra_q is not None:
        s.extra_q = extra_q.copy()
    g = d[d.modality.isin(mods)]
    if g.empty:   # a pack with no reading of the label's types (e.g. flight readings only): nothing to weight
        return s, weights
    for _ in range(iters):
        _, _, xs, _, _ = kalman(s, h, tau0)
        at = dict(zip(s_grid.flight_id, xs))
        e = g.y_c.to_numpy() - g.flight_id.map(at).to_numpy()
        sig = g.modality.map(lambda m: params[m]["sigma"]).to_numpy()
        w = (nu + 1) / (nu + (e / sig) ** 2)
        weights.loc[g.index] = np.nan_to_num(w, nan=0.0)
        extra = s.extra_q
        s = make_series(s_grid, d, params, mods, weights)
        s.extra_q = extra
    return s, weights


def detect_steps(s: Series, h: Hyper, tau0: float, lc: dict) -> list[dict]:
    """Sustained level shifts that the fade prior cannot explain.

    Compares robust means of the sessions either side of each point after removing the expected fade. An accepted step gets a
    large process variance at that session so the smoother can follow it; it is listed in label_steps.csv for checking against
    maintenance records."""
    has = np.flatnonzero(s.lam > 0)
    if len(has) < 3 * lc["step_window"]:
        return []
    b1, b2, _ = _basis(s, tau0)
    drift = np.cumsum(h.a * b1 + h.b * b2)
    z = (s.eta[has] / s.lam[has]) + drift[has]
    sig = max(mad_sigma(np.diff(z)) / np.sqrt(2), 1e-3)
    W = lc["step_window"]
    stat = np.zeros(len(z))
    dlt = np.zeros(len(z))
    for j in range(W, len(z) - W + 1):
        dlt[j] = np.median(z[j:j + W]) - np.median(z[j - W:j])
        stat[j] = dlt[j] / (1.2533 * sig * np.sqrt(2.0 / W))
    out = []
    a = np.abs(stat).copy()
    for _ in range(lc["max_steps_per_pack"]):
        j = int(np.argmax(a))
        if a[j] < lc["step_z"]:
            break
        if abs(dlt[j]) * 100 >= lc["step_min_pts"]:
            out.append(dict(k=int(has[j]), delta=float(dlt[j]), z=float(stat[j])))
        a[max(0, j - 2 * W): j + 2 * W] = 0
    return out


def build_grid(obs: pd.DataFrame) -> dict:
    """Session grid per pack: every usable session, in time order, with cumulative discharged charge."""
    grids = {}
    u = obs[obs.usable_electrical].dropna(subset=["start_datetime"])
    for key, g in u.sort_values(["start_datetime", "flight_id"]).groupby(["aircraft_label", "battery"], sort=False):
        g = g.copy()
        g["t_days"] = (g.start_datetime - g.start_datetime.iloc[0]).dt.total_seconds() / DAY
        g["ah_cum"] = g.ah_out.fillna(0).cumsum().shift(1).fillna(0.0)
        grids[key] = g.reset_index(drop=True)
    return grids


def smooth_label(grids, d, params, mods, h, lc, steps=None, find_steps=False):
    """Run the robust smoother on every pack. Returns a tidy frame and accepted steps."""
    tau0 = lc["tau0_days"]
    rows, stp = [], []
    for key, grid in grids.items():
        dp = d[(d.aircraft_label == key[0]) & (d.battery == key[1])]
        if not len(dp[dp.modality.isin(mods)]):
            continue
        s, weights = robust_pass(grid, dp, params, mods, h, tau0, lc["t_nu"], lc["irls_iters"])
        found = (steps or {}).get(key, [])
        if find_steps:
            found = detect_steps(s, h, tau0, lc)
        for st in found:
            s.extra_q[st["k"]] = (2 * st["delta"]) ** 2
            stp.append(dict(aircraft_label=key[0], battery=key[1], k=st["k"], flight_id=int(grid.flight_id.iloc[st["k"]]), when=grid.start_datetime.iloc[st["k"]],
                            delta=st["delta"], delta_pts=100 * st["delta"], z=st["z"]))
        if found:
            s, weights = robust_pass(grid, dp, params, mods, h, tau0, lc["t_nu"], lc["irls_iters"], s.extra_q)
        xf, pf, xs, ps, _ = kalman(s, h, tau0)
        prec = 1.0 / ps
        loo_prec = np.maximum(prec - s.lam, 1e-3)
        xloo = (xs * prec - s.eta) / loo_prec
        rows.append(pd.DataFrame(dict(flight_id=grid.flight_id, battery=key[1], aircraft_label=key[0], x_s=xs, sd_s=np.sqrt(ps), x_f=xf, sd_f=np.sqrt(pf),
                                      x_loo=xloo, sd_loo=np.sqrt(1.0 / loo_prec), lam=s.lam, t_days=grid.t_days, ah_cum=grid.ah_cum)))
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), stp


def isotonic_view(lab: pd.DataFrame, steps: list[dict]) -> pd.Series:
    """Monotone non-increasing version between accepted steps: a sensitivity check, never the default label."""
    out = pd.Series(np.nan, index=lab.index)
    for (ac, b), g in lab.groupby(["aircraft_label", "battery"], sort=False):
        ks = sorted(s["k"] for s in steps if s["aircraft_label"] == ac and s["battery"] == b)
        idx = g.index.to_numpy()
        cuts = [0] + ks + [len(idx)]
        for a, c in zip(cuts[:-1], cuts[1:]):
            if c - a < 2:
                out.loc[idx[a:c]] = g.x_s.to_numpy()[a:c]
                continue
            ir = IsotonicRegression(increasing=False)
            out.loc[idx[a:c]] = ir.fit_transform(np.arange(c - a), g.x_s.to_numpy()[a:c], sample_weight=1.0 / g.sd_s.to_numpy()[a:c] ** 2)
    return out


def resistance_label(grids: dict, obs: pd.DataFrame, lc: dict, fit_mask: pd.Series, log) -> pd.DataFrame:
    """Resistance growth relative to early life, smoothed the same way. Lower-quantile baseline, because break-in biases early values."""
    col = next((c for c in ("r_step25_mohm", "r_ramp25_mohm") if c in obs and obs[c].notna().sum() >= 30), None)
    if col is None:
        return pd.DataFrame(columns=["flight_id", "battery", "r_growth", "r_growth_sd"])
    d = obs.dropna(subset=[col])[["flight_id", "battery", "aircraft_label", "start_datetime", col]].rename(columns={col: "s"})
    d["modality"] = "r"
    lc2 = dict(lc, baseline_q=lc["r_baseline_q"])
    d = normalise(d, lc2)
    d["y_c"] = np.log(d.y)
    d["e_local"] = local_residual(d, "y_c")
    fitm = obs.fleet.astype(str) == str(lc["fit_fleet"])
    sig = float(max(mad_sigma(d.loc[d.aircraft_label.isin(obs[fitm].aircraft_label.unique()), "e_local"].to_numpy()), 0.005))
    params = {"r": dict(sigma=sig)}
    h = Hyper(0.0, 0.0, 2e-6)
    lab, _ = smooth_label(grids, d, params, ["r"], h, lc)
    lab["r_growth"] = np.exp(lab.x_s - lab.groupby(["aircraft_label", "battery"]).x_s.transform("first"))
    lab["r_growth_sd"] = lab.sd_s
    log(f"[label] resistance measurement {col}: noise {100 * sig:.1f}% per session")
    return lab[["flight_id", "battery", "r_growth", "r_growth_sd"]]


def diagnostics(ctx: Ctx, obs: pd.DataFrame, long: pd.DataFrame, st: dict) -> None:
    """Tables that show whether the physics assumptions hold on this data, so a run can be judged from its results alone."""
    rows = []
    for name, ref in st["refs"].items():
        if ref is None:
            continue
        q = ref.q_ref()
        rows.append(pd.DataFrame(dict(modality=name, v=ref.edges, q_ref_ah=q, dq_dv_ah_per_v=np.r_[np.diff(q) / np.diff(ref.edges), np.nan], valid=ref.valid)))
    if rows:
        pd.concat(rows).to_csv(ctx.tables / "E2_reference_curves.csv", index=False)
    if st["ocv"] is not None:
        f = st["ocv"]
        pd.DataFrame(dict(v=f.knots, f_ah=f(f.knots))).to_csv(ctx.tables / "E3_rest_voltage_function.csv", index=False)
    g = long.groupby(["aircraft_label", "battery", "modality"])
    q = g.agg(n=("s", "size"), s_first=("s", "first"), s_median=("s", "median"), s_last=("s", "last"), span_ah=("span_ah", "median"), se=("se", "median"),
              v_lo=("v_lo", "median"), v_hi=("v_hi", "median"), temp_c=("temp_mean_c", "median")).reset_index()
    q.to_csv(ctx.tables / "E4_capacity_reading_quality.csv", index=False)
    u = obs[obs.usable_electrical].copy()
    u["quarter"] = u.start_datetime.dt.to_period("Q").astype(str)
    cols = [c for c in ("r_step25_mohm", "r_step3025_mohm", "r_ramp25_mohm", "temp_mean_c", "spread_mv_end") if c in u]
    u.groupby(["aircraft_label", "battery", "quarter"])[cols].median().reset_index().to_csv(ctx.tables / "E5_resistance_by_quarter.csv", index=False)
    p = st["pairs"]
    if len(p):
        ps = p.assign(gap=pd.cut(p.rest_min, [0, 60, 240, 1440, 1e6], labels=["<1 h", "1-4 h", "4-24 h", ">1 day"]), kind=np.where(p.dq > 0, "charge", "discharge"))
        ps.groupby(["aircraft_label", "battery", "kind", "gap"], observed=True).agg(n=("s", "size"), s_median=("s", "median"), s_sd=("s", "std")).reset_index().to_csv(ctx.tables / "E6_rest_pairs.csv", index=False)


def run_label(ctx: Ctx, log=None) -> pd.DataFrame:
    log = log or Log(ctx)
    lc = ctx.cfg["label"]
    obs0 = pd.read_parquet(ctx.cache / "session_observables.parquet")
    long, st = run_capacity(ctx, obs0, log)
    obs = st["obs"]
    diagnostics(ctx, obs, long, st)
    long.to_parquet(ctx.cache / "capacity_observations.parquet", index=False)
    obs.to_parquet(ctx.cache / "session_observables_r.parquet", index=False)
    d = normalise(long, lc)
    fit_mask = d.fleet.astype(str) == str(lc["fit_fleet"])
    d, params = calibrate(d, fit_mask, lc, log)
    grids = build_grid(obs)
    mods = [m for m in lc["modalities"] if m in params]
    prim = lc["primary"] if lc["primary"] in params else mods[0]
    # no per-modality gain rescaling: when a reading type disagrees, rescaling amplifies it (gains hit 0.5 on real data and doubled
    # the deviations). Reading types are combined as measured; their disagreement goes into the systematic uncertainty instead.
    fit_keys = [k for k in grids if k[0] in set(d[fit_mask].aircraft_label)]
    h = Hyper()
    tau0 = lc["tau0_days"]
    for _ in range(2):   # weights from the current fit, then refit: single bad sessions must not inflate the process noise
        fs = [robust_pass(grids[k], d[(d.aircraft_label == k[0]) & (d.battery == k[1])], params, mods, h, tau0, lc["t_nu"], lc["irls_iters"])[0] for k in fit_keys]
        fs = [s for s in fs if (s.lam > 0).sum() >= 10]
        if fs:
            h = fit_hyper(fs, tau0, h, lc["q_max_per_day"])
    log(f"[label] fitted on fleet {lc['fit_fleet']}: calendar a={h.a:.4f}/sqrt(yr), cycling b={h.b:.4f}/1000 Ah, process q={h.q:.2e}/day")
    lab, steps = smooth_label(grids, d, params, mods, h, lc, find_steps=True)
    lab["x_iso"] = isotonic_view(lab, steps)
    step_map = {}
    for s in steps:
        step_map.setdefault((s["aircraft_label"], s["battery"]), []).append(s)
    views = {}
    for m in sorted(params):
        v, _ = smooth_label(grids, d, params, [m], h, lc, steps=step_map)
        if len(v):
            views[m] = v.rename(columns={"x_s": f"x_{m}", "x_f": f"xf_{m}", "sd_s": f"sd_{m}"})[["flight_id", "battery", f"x_{m}", f"xf_{m}", f"sd_{m}"]]
            lab = lab.merge(views[m], on=["flight_id", "battery"], how="left")
    rg = resistance_label(grids, obs, lc, fit_mask, log)
    lab = lab.merge(rg, on=["flight_id", "battery"], how="left")
    bms = obs[["flight_id", "battery", "soh_bms_median", "soh_bms_first", "soh_bms_last", "q_soc_window_ah", "fleet", "flight_type", "start_datetime", "temp_mean_c"]]
    lab = lab.merge(bms, on=["flight_id", "battery"], how="left")
    for c in [c for c in lab.columns if c.startswith(("x_", "xf_"))]:
        lab[c.replace("x_", "soh_", 1) if c.startswith("x_") else c.replace("xf_", "soh_f_", 1)] = 100 * lab[c]
    lab["soh_phys"], lab["soh_filt"], lab["soh_loo"] = 100 * lab.x_s, 100 * lab.x_f, 100 * lab.x_loo
    # The Kalman variance only knows about random noise. Disagreement between independent measurement types is the honest,
    # truth-free measure of systematic error, so it is added to every reported uncertainty.
    each = [f"soh_{m}" for m in mods if f"soh_{m}" in lab]
    sys_sd = lab[each].sub(lab.soh_phys, axis=0).abs().mean(axis=1) * 1.2533 if len(each) >= 2 else pd.Series(0.0, index=lab.index)
    sys_sd = sys_sd.groupby([lab.aircraft_label, lab.battery]).transform(lambda z: z.rolling(41, center=True, min_periods=5).median()).fillna(sys_sd.median()).clip(lower=lc["label_sd_floor_pts"])
    lab["soh_sd_random"] = 100 * lab.sd_s
    lab["soh_sd_systematic"] = sys_sd
    lab["soh_sd"] = np.hypot(100 * lab.sd_s, sys_sd)
    lab["soh_filt_sd"] = np.hypot(100 * lab.sd_f, sys_sd)
    lab["soh_loo_sd"] = np.hypot(100 * lab.sd_loo, sys_sd)
    lab.to_parquet(ctx.cache / "labels.parquet", index=False)
    lab.to_csv(ctx.tables / "L1_physics_soh_label.csv", index=False)
    pd.DataFrame(steps).to_csv(ctx.tables / "L2_detected_steps.csv", index=False)
    d.to_parquet(ctx.cache / "observations_calibrated.parquet", index=False)
    write_json(ctx.cache / "label_hyper.json", dict(a=h.a, b=h.b, q=h.q, params=params, mods=mods, primary=prim, fit_fleet=lc["fit_fleet"], resistance=st["rparams"]))
    if steps:
        log("[label] steps accepted (check against maintenance records):\n" + pd.DataFrame(steps).drop(columns=["k", "delta"]).round({"delta_pts": 2, "z": 1}).to_string(index=False))
    else:
        log("[label] no step accepted")
    summary = lab.groupby(["aircraft_label", "battery"]).agg(sessions=("flight_id", "size"), measured=("lam", lambda s: int((s > 0).sum())), first=("soh_phys", "first"), last=("soh_phys", "last"),
                                                             sd_median=("soh_sd", "median"), bms_first=("soh_bms_median", "first"), bms_last=("soh_bms_median", "last"), days=("t_days", "max")).reset_index()
    summary["fade_pts_per_year"] = (summary["first"] - summary["last"]) / (summary.days / 365.25).replace(0, np.nan)
    summary.to_csv(ctx.tables / "L3_label_summary.csv", index=False)
    log(summary.round(2).to_string(index=False))
    return lab


def as_of(grid: pd.DataFrame, d: pd.DataFrame, params: dict, mods: list[str], h: Hyper, lc: dict, cutoff_days: float) -> tuple[float, float]:
    """Causal state of one pack at a cutoff, from measurements up to the cutoff only."""
    g = grid[grid.t_days <= cutoff_days]
    if not len(g):
        return np.nan, np.nan
    s = make_series(g, d[d.flight_id.isin(g.flight_id)], params, mods)
    xf, pf, *_ = kalman(s, h, lc["tau0_days"], smooth=False)
    return float(xf[-1]), float(np.sqrt(pf[-1]))
