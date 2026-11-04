"""Capacity from voltage and charge alone, with no BMS SOC or SOH in the calculation.

Equivalent-circuit view. A pack's rest-corrected cell voltage v is a fixed function of how full the cells are:
    v(t) = U(theta(t)),   theta(t) = theta_0 + Q(t) / C
where Q is the integrated current and C the capacity. U is a property of the chemistry, shared by the fleet.
Inverting U gives Q as a function of v:
    Q_k(v) = c_k + s_k * q_ref(v)          (1)
Every session's Q(v) curve is a shifted and scaled copy of one reference curve. The scale s_k is the session's capacity
relative to the reference; the shift c_k only says how full the pack was when the session started.

Three independent readings of s:
  chg   charge curve, high-current part before the charger steps down                (equation 1)
  dis   steady in-flight load                                                         (equation 1)
  r2r   rest-to-rest: ah_net / [f(v_rest_1) - f(v_rest_2)], f self-calibrated from the data

The reference curves are built from the fit fleet only; other aircraft are registered onto them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from .common import Ctx, Log, mad_sigma


def load_curves(ctx: Ctx, fid: int) -> dict:
    p = ctx.cache / "sessions" / f"flight_{fid}.npz"
    return dict(np.load(p)) if p.exists() else {}


# ------------------------------------------------------------------------------ resistance model
def fit_resistance(obs: pd.DataFrame, fit_mask: pd.Series, log) -> tuple[pd.DataFrame, dict]:
    """Temperature law log r = log r25(pack, quarter) + E (1/T - 1/298.15), one E per resistance type.

    Season and age are confounded, so E comes from within-pack, within-quarter variation only."""
    out = obs.copy()
    params = {}
    q = out.start_datetime.dt.to_period("Q").astype(str)
    for col, tcol in (("r_step_mohm", "step_temp_c"), ("r_step30_mohm", "step_temp_c"), ("r_ramp_mohm", "ramp_temp_c")):
        for c in (col, tcol):
            if c not in out:
                out[c] = np.nan
        y = np.log(out[col].where((out[col] > 0.2) & (out[col] < 12)))
        ik = 1.0 / (out[tcol] + 273.15) - 1.0 / 298.15
        m = y.notna() & ik.notna() & fit_mask & out.usable_electrical
        e = np.nan
        if m.sum() >= 20:
            grp = out.aircraft_label.astype(str) + "_" + out.battery.astype(str) + "_" + q
            dy = y[m] - y[m].groupby(grp[m]).transform("median")
            dx = ik[m] - ik[m].groupby(grp[m]).transform("median")
            keep = (dy.abs() < 3 * max(mad_sigma(dy.to_numpy()), 1e-3)) & (dx.abs() > 0)
            if keep.sum() >= 15 and dx[keep].std() > 0:
                e = float(np.sum(dx[keep] * dy[keep]) / np.sum(dx[keep] ** 2))
        e = e if np.isfinite(e) and 0 <= e <= 8000 else 0.0
        params[col] = dict(E_kelvin=e, n=int(m.sum()))
        out[col.replace("_mohm", "25_mohm")] = np.exp(y - e * ik)
    log(f"[resistance] temperature law (E in K, 2500-4000 is typical of Li-ion): {params}")
    return out, params


def effective_r(obs: pd.DataFrame, params: dict) -> pd.Series:
    """Per-cell resistance used for the IR correction of one session (Ohm per pack A).

    Slowly varying pack value (rolling median of the 25 degC 30 s step resistance, falling back to the ramp and the short step)
    scaled to the session temperature. A rolling median keeps one bad step from distorting a whole session."""
    use = []
    for col in ("r_step3025_mohm", "r_step25_mohm", "r_ramp25_mohm"):
        if col in obs and obs[col].notna().sum() >= 10:
            use.append(col)
    res = pd.Series(np.nan, index=obs.index)
    e = params.get("r_step30_mohm", {}).get("E_kelvin") or params.get("r_ramp_mohm", {}).get("E_kelvin") or 0.0
    for _, g in obs.groupby(["aircraft_label", "battery"], sort=False):
        s = pd.Series(np.nan, index=g.index)
        for c in use:
            sm = g[c].rolling(31, center=True, min_periods=3).median().ffill().bfill()
            s = s.fillna(sm)
        res.loc[g.index] = s
    res = res.fillna(res.median()).fillna(2.5)
    ik = 1.0 / (obs.temp_mean_c.fillna(obs.temp_mean_c.median()) + 273.15) - 1.0 / 298.15
    return res * np.exp(e * ik) / 1000.0


# ------------------------------------------------------------------------------ session curves
def charge_samples(c: dict, b: str, row, r_eff: float, rc: dict, signal: str = "v"):
    """(v, q) during the high-current part of the main charge, IR-corrected; polarisation start-up skipped."""
    t, i, q, v, seg = c[f"{b}_t"], c[f"{b}_i"], c[f"{b}_q"], c[f"{b}_{signal}"], c[f"{b}_seg"]
    start, stop = row.get("charge_start_s", np.nan), row.get("charge_stop_s", np.nan)
    if not (np.isfinite(start) and np.isfinite(stop)):
        return None
    if np.isfinite(row.get("step_t_s", np.nan)):
        stop = row["step_t_s"] - 30.0
    m = (t >= start + rc["charge_skip_s"]) & (t <= stop) & (i <= -rc["charge_i_min_a"]) & np.isfinite(v) & np.isfinite(i) & np.isfinite(q)
    if m.sum() < 12:
        return None
    s = seg[m]
    keep = s == np.bincount(s - s.min()).argmax() + s.min()
    vv, qq = (v[m] + i[m] * r_eff)[keep], q[m][keep]
    vv = pd.Series(vv).rolling(3, center=True, min_periods=1).median().to_numpy()   # one voltage glitch must not bend the curve
    vv = np.maximum.accumulate(vv)                          # charging voltage only rises; removes sensor noise
    return vv, qq


def discharge_samples(c: dict, b: str, r_eff: float, rc: dict, signal: str = "v"):
    """(v, q) on steady in-flight load: same IR correction, only after the current has been steady for a while."""
    t, i, q, v, seg = c[f"{b}_t"], c[f"{b}_i"], c[f"{b}_q"], c[f"{b}_{signal}"], c[f"{b}_seg"]
    win = max(2, int(rc["dis_steady_s"] / max(np.median(np.diff(t)), 1)))
    s = pd.Series(i)
    lo, hi = s.rolling(win, min_periods=win).min().to_numpy(), s.rolling(win, min_periods=win).max().to_numpy()
    m = (i >= rc["dis_i_min_a"]) & (i <= rc["dis_i_max_a"]) & (hi - lo <= 0.25 * np.maximum(np.abs(i), 1)) & np.isfinite(v) & np.isfinite(q)
    if m.sum() < 8:
        return None
    sg = seg[m]
    keep = sg == np.bincount(sg - sg.min()).argmax() + sg.min()
    return (v[m] + i[m] * r_eff)[keep], q[m][keep]


def to_grid(vv, qq, edges, monotone):
    """Charge at each voltage bin. A monotone curve is interpolated; a scattered one is averaged per bin."""
    n = len(edges)
    if monotone:
        u, idx = np.unique(vv, return_index=True)
        if len(u) < 3:
            return np.full(n, np.nan)
        out = np.interp(edges, u, qq[idx], left=np.nan, right=np.nan)
        return out
    centres = np.digitize(vv, edges) - 1
    out = np.full(n, np.nan)
    for j in np.unique(centres):
        if 0 <= j < n:
            out[j] = np.median(qq[centres == j])
    return out


class Reference:
    """Fleet reference curve q_ref(v): built by iterating (median local slope -> scale of each session)."""

    def __init__(self, edges):
        self.edges = edges
        self.d = None

    def q_ref(self):
        return np.r_[0.0, np.cumsum(self.d)]

    def fit(self, M: np.ndarray, iterations: int, early: np.ndarray, min_window_ah: float):
        """M[k, j] = charge of session k at voltage bin j (NaN = not covered). early marks sessions that fix the scale."""
        s = np.ones(len(M))
        for _ in range(iterations):
            D = np.diff(M, axis=1) / s[:, None]
            with np.errstate(all="ignore"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    d = np.nanmedian(D, axis=0)
            cnt = np.isfinite(D).sum(0)
            d = np.where(cnt >= 5, d, np.nan)
            pos = np.flatnonzero(np.isfinite(d))
            if len(pos) < 3:
                raise RuntimeError("too few sessions to build a reference curve")
            d = np.interp(np.arange(len(d)), pos, d[pos])
            self.d = np.clip(d, 1e-4, None)
            self.valid = np.zeros(len(self.d) + 1, bool)
            self.valid[pos[0]: pos[-1] + 2] = True
            sc = self.register(M, min_window_ah)
            scale = np.nanmedian(sc.s[early]) if np.isfinite(sc.s[early]).any() else 1.0
            self.d = self.d * scale
            s = np.where(np.isfinite(sc.s), sc.s / scale, 1.0)
        return self

    def register(self, M: np.ndarray, min_window_ah: float) -> pd.DataFrame:
        """Least-squares fit of equation (1) for each session."""
        qr = self.q_ref()
        rows = []
        for k in range(len(M)):
            y = M[k]
            ok = np.isfinite(y) & self.valid
            if ok.sum() < 5 or np.ptp(qr[ok]) < min_window_ah:
                rows.append((np.nan, np.nan, np.nan, np.nan, np.nan, int(ok.sum())))
                continue
            x = qr[ok]
            xm, ym = x.mean(), y[ok].mean()
            sxx = np.sum((x - xm) ** 2)
            s = np.sum((x - xm) * (y[ok] - ym)) / sxx
            res = y[ok] - ym - s * (x - xm)
            neff = max(ok.sum() / 4.0, 3.0)
            se = float(np.std(res) / np.sqrt(sxx * neff / ok.sum()))
            rows.append((float(s), se, float(self.edges[ok].min()), float(self.edges[ok].max()), float(np.ptp(x)), int(ok.sum())))
        return pd.DataFrame(rows, columns=["s", "se", "v_lo", "v_hi", "span_ah", "n_bins"])


def _early(df: pd.DataFrame, days: float) -> np.ndarray:
    first = df.groupby(["aircraft_label", "battery"]).start_datetime.transform("min")
    return ((df.start_datetime - first).dt.total_seconds() / 86400 <= days).to_numpy()


def register_mode(ctx: Ctx, obs: pd.DataFrame, mode: str, signal: str, fit_mask: pd.Series, log) -> tuple[pd.DataFrame, Reference | None]:
    rc = ctx.cfg["registration"]
    bin_v = rc["chg_bin_v"] if mode == "chg" else rc["dis_bin_v"]
    edges = np.arange(3.40, 4.2001, bin_v)
    rows, grids = [], []
    want = obs[obs.usable_electrical & (obs.charge_min >= 5 if mode == "chg" else obs.discharge_min >= 3)]
    for r in want.to_dict("records"):
        c = load_curves(ctx, int(r["flight_id"]))
        b = str(int(r["battery"]))
        if f"{b}_t" not in c:
            continue
        got = charge_samples(c, b, r, r["r_eff"], rc, signal) if mode == "chg" else discharge_samples(c, b, r["r_eff"], rc, signal)
        if got is None:
            continue
        g = to_grid(got[0], got[1], edges, mode == "chg")
        if np.isfinite(g).sum() >= 4:
            grids.append(g)
            rows.append(dict(flight_id=r["flight_id"], battery=r["battery"], aircraft_label=r["aircraft_label"], start_datetime=r["start_datetime"]))
    if len(rows) < 10:
        log(f"[capacity] {mode}/{signal}: only {len(rows)} usable sessions, skipped")
        return pd.DataFrame(columns=["flight_id", "battery", "s", "se"]), None
    meta = pd.DataFrame(rows)
    M = np.vstack(grids)
    fit = pd.merge(meta[["flight_id", "battery"]], obs.assign(_f=fit_mask)[["flight_id", "battery", "_f"]], on=["flight_id", "battery"], how="left")._f.fillna(False).to_numpy(bool)
    if fit.sum() < 10:
        log(f"[capacity] {mode}/{signal}: only {fit.sum()} sessions in the fit fleet; using all aircraft for the reference")
        fit = np.ones(len(meta), bool)
    ref = Reference(edges)
    early = _early(meta[fit], ctx.cfg["label"]["settle_days"] + ctx.cfg["label"]["baseline_days"])
    ref.fit(M[fit], rc["iterations"], early, rc["min_window_ah"])
    sc = ref.register(M, rc["min_window_ah"])
    res = pd.concat([meta[["flight_id", "battery"]].reset_index(drop=True), sc], axis=1)
    # the same fit on the lower and upper half of the voltage window: if the curve keeps its shape with age, both fade alike
    mid = float(np.nanmedian(edges[np.isfinite(M).any(axis=0) & ref.valid])) if ref.valid.any() else np.nan
    for name, keep in (("s_low", edges <= mid), ("s_high", edges >= mid)):
        M2 = M.copy()
        M2[:, ~keep] = np.nan
        res[name] = ref.register(M2, rc["min_window_ah"] / 2).s.to_numpy()
    ok = res.s.notna()
    log(f"[capacity] {mode}/{signal}: {int(ok.sum())} of {len(res)} sessions registered; median span {res.span_ah.median():.1f} Ah; "
        f"reference covers {edges[ref.valid].min():.3f}-{edges[ref.valid].max():.3f} V")
    return res[ok], ref


# ------------------------------------------------------------------------------ rest-to-rest
def rest_pairs(obs: pd.DataFrame, rc: dict) -> pd.DataFrame:
    """Pairs of rest voltages with only logged current between them. BMS SOC is used only to detect an unlogged session in between."""
    rows = []
    for (ac, b), g in obs[obs.usable_electrical].groupby(["aircraft_label", "battery"], sort=False):
        g = g.sort_values("start_datetime")
        rec = g.to_dict("records")
        for a, c in zip(rec[:-1], rec[1:]):
            if not (np.isfinite(a.get("rest_start_v", np.nan)) and np.isfinite(c.get("rest_start_v", np.nan))):
                continue
            end = a["start_datetime"] + pd.Timedelta(minutes=float(a["duration_min"]) if np.isfinite(a["duration_min"]) else 0.0)
            rest_min = (c["start_datetime"] - end).total_seconds() / 60.0
            if not (rc["r2r_min_rest_min"] <= rest_min <= rc["r2r_max_rest_days"] * 1440):
                continue
            if abs((c.get("soc_first") or np.nan) - (a.get("soc_last") or np.nan)) > rc["r2r_soc_gate_pts"]:
                continue
            if a.get("n_segments_q", 1) > 1:
                continue
            dq = a["q_end_ah"] - a["rest_start_q_ah"]
            if not np.isfinite(dq) or abs(dq) < rc["r2r_min_ah"]:
                continue
            va = a["rest_start_v"] + a["rest_start_i_a"] * a["r_eff"]
            vc = c["rest_start_v"] + c["rest_start_i_a"] * c["r_eff"]
            rows.append(dict(aircraft_label=ac, battery=b, flight_id=a["flight_id"], next_flight_id=c["flight_id"], start_datetime=a["start_datetime"],
                             v_a=va, v_c=vc, dq=dq, rest_min=rest_min))
    return pd.DataFrame(rows)


class OcvFunction:
    """Monotone piecewise-linear f(v), in Ah per unit of relative capacity. f(v2) - f(v1) = Ah needed between the two rest voltages."""

    def __init__(self, knots, w):
        self.knots, self.w = knots, w

    def __call__(self, v):
        return np.interp(v, self.knots, np.r_[0.0, np.cumsum(self.w)])

    def inverse(self, f):
        y = np.r_[0.0, np.cumsum(self.w)] + np.arange(len(self.knots)) * 1e-9
        return np.interp(f, y, self.knots)


def fit_ocv(p: pd.DataFrame, knot_v: float, iters: int = 6, s_init: np.ndarray | None = None) -> tuple[OcvFunction, np.ndarray]:
    """Bilinear self-calibration: dq = s * [f(v_hi) - f(v_lo)]. Alternates non-negative least squares for f with a smooth s per pack."""
    lo, hi = np.minimum(p.v_a, p.v_c).to_numpy(), np.maximum(p.v_a, p.v_c).to_numpy()
    y = np.abs(p.dq.to_numpy())
    knots = np.arange(np.floor(lo.min() / knot_v) * knot_v, np.ceil(hi.max() / knot_v) * knot_v + 1e-9, knot_v)
    width = np.diff(knots)
    A = np.vstack([np.clip(np.minimum(hi, knots[j + 1]) - np.maximum(lo, knots[j]), 0, None) / width[j] for j in range(len(width))]).T
    lam = 0.05 * np.median(y)
    Dm = np.zeros((len(width) - 2, len(width)))
    for j in range(len(width) - 2):
        Dm[j, j:j + 3] = [lam, -2 * lam, lam]
    s = np.ones(len(p)) if s_init is None else s_init.copy()
    w = None
    keep = np.ones(len(p), bool)
    for _ in range(iters):
        X = np.vstack([A[keep] * s[keep][:, None], Dm])
        w, _ = nnls(X, np.r_[y[keep], np.zeros(len(Dm))], maxiter=20 * X.shape[1])
        pred = A @ w
        raw = np.where(pred > 1, y / pred, np.nan)
        keep = np.abs(raw - np.nanmedian(raw)) < 5 * max(mad_sigma(raw), 1e-3)
        keep = keep & np.isfinite(raw)
        sm = pd.Series(raw).groupby([p.aircraft_label.to_numpy(), p.battery.to_numpy()]).transform(lambda z: z.rolling(21, center=True, min_periods=3).median()).to_numpy()
        s = np.where(np.isfinite(sm), sm, np.nanmedian(raw))
    return OcvFunction(knots, w), raw


def run_capacity(ctx: Ctx, obs: pd.DataFrame, log=None) -> tuple[pd.DataFrame, dict]:
    """Adds resistance columns to obs and returns the long table of capacity observations."""
    log = log or Log(ctx)
    rc = ctx.cfg["registration"]
    fit_fleet = ctx.cfg["label"]["fit_fleet"]
    fit_mask = obs.fleet.astype(str) == str(fit_fleet)
    log(f"[capacity] reference curves and laws fitted on fleet {fit_fleet} only ({int(fit_mask.sum())} pack-sessions)")
    obs, rparams = fit_resistance(obs, fit_mask, log)
    obs["r_eff"] = effective_r(obs, rparams)
    out = []
    refs = {}
    for name, mode, signal in (("chg", "chg", "v"), ("dis", "dis", "v"), ("chg_vmax", "chg", "vmax"), ("dis_vmin", "dis", "vmin")):
        res, ref = register_mode(ctx, obs, mode, signal, fit_mask, log)
        refs[name] = ref
        if len(res):
            out.append(res.assign(modality=name))
    pairs = rest_pairs(obs, rc)
    ocv = None
    if len(pairs) >= 20:
        pf = pairs[pairs.aircraft_label.isin(obs[fit_mask].aircraft_label.unique())]
        if len(pf) < 20:
            pf = pairs
        ocv, _ = fit_ocv(pf.reset_index(drop=True), rc["r2r_knot_v"])
        pred = np.abs(ocv(pairs.v_a) - ocv(pairs.v_c))
        pairs["s"] = np.where(pred > 1, np.abs(pairs.dq) / pred, np.nan)
        pairs["se"] = np.nan
        early = _early(pf, ctx.cfg["label"]["settle_days"] + ctx.cfg["label"]["baseline_days"])
        base = np.nanmedian(np.abs(pf.dq.to_numpy())[early] / np.maximum(np.abs(ocv(pf.v_a) - ocv(pf.v_c))[early], 1)) if early.any() else 1.0
        pairs["s"] = pairs.s / base
        ocv.w = ocv.w * base
        out.append(pairs[["flight_id", "battery", "s", "se"]].dropna(subset=["s"]).assign(modality="r2r"))
        log(f"[capacity] rest-to-rest: {len(pairs)} pairs, {int(pairs.s.notna().sum())} usable")
    else:
        log(f"[capacity] rest-to-rest: only {len(pairs)} pairs, skipped")
    pairs.to_csv(ctx.cache / "rest_pairs.csv", index=False)
    long = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["flight_id", "battery", "s", "se", "modality"])
    long = long.merge(obs[["flight_id", "battery", "aircraft_label", "fleet", "start_datetime", "temp_mean_c"]], on=["flight_id", "battery"], how="left")
    return long, dict(obs=obs, refs=refs, ocv=ocv, pairs=pairs, rparams=rparams)
