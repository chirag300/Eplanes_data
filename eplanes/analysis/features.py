"""Flight x battery feature extraction under an explicit cleaning rule.

Cleaning rules (what is excluded from statistics; raw rows are never deleted):
  none        use every row as logged
  row0        mask the all-zero first row only
  settle      mask each signal group until its data-driven settle index (+ row 0)
  fixed:<m>   mask every signal for the first <m> minutes
Segmentation / timestamp variants are handled by `seg_current_a` and `time_rule`
arguments so the sensitivity analysis can reuse the same code.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..build.clean import FLAG_BITS, SIGNALS, iter_flights
from ..quality.init_artifacts import GROUP_BITS, group_of

DEFAULT_RULE = "leadzero"
warnings.filterwarnings("ignore", message="All-NaN slice")
warnings.filterwarnings("ignore", message="Mean of empty slice")


def apply_rule(d: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Return a copy where masked sample values are NaN."""
    d = d.copy()
    if rule == "none":
        return d
    t = d["time_min"].to_numpy(float)
    if rule == "leadzero":
        # mask, per signal, the run of exact zeros that opens the record before the channel first reports a value
        # (real data: temperatures stay 0 for several samples after row 0, so masking row 0 alone is not enough;
        #  the data-driven 'settle' window instead masks real charge/discharge onsets)
        for c in SIGNALS:
            if c in d:
                x = d[c].to_numpy(float)
                nz = np.flatnonzero(np.isfinite(x) & (x != 0))
                if len(nz) and nz[0] > 0:
                    d.loc[d.index[: nz[0]], c] = np.nan
        return d
    if rule == "row0":
        for c in SIGNALS:
            if c in d:
                d.loc[d["quality_flag"].to_numpy() & FLAG_BITS["row0_zero"] > 0, c] = np.nan
        return d
    if rule == "settle":
        r0 = (d["quality_flag"].to_numpy() & FLAG_BITS["row0_zero"]) > 0
        ug = d["init_unsettled_groups"].to_numpy()
        for c in SIGNALS:
            g = group_of(c)
            if g and c in d:
                m = r0 | ((ug & GROUP_BITS[g]) > 0)
                d.loc[m, c] = np.nan
        return d
    if rule.startswith("fixed:"):
        m = t < float(rule.split(":")[1])
        for c in SIGNALS:
            if c in d:
                d.loc[m, c] = np.nan
        return d
    raise ValueError(rule)


def _integrate(t, y, max_dt):
    """Trapezoid integral of y over t (minutes -> hours) skipping gaps and NaNs."""
    if len(t) < 2:
        return 0.0
    dt = np.diff(t)
    ym = 0.5 * (y[1:] + y[:-1])
    ok = np.isfinite(ym) & (dt > 0) & (dt <= max_dt)
    return float(np.sum(ym[ok] * dt[ok]) / 60.0)


def monotone_runs(s, tol):
    runs, start, lo = [], 0, s[0]
    for k in range(1, len(s)):
        if s[k] >= lo - tol:
            lo = max(lo, s[k])
        else:
            if k - 1 > start:
                runs.append((start, k - 1))
            start, lo = k, s[k]
    if len(s) - 1 > start:
        runs.append((start, len(s) - 1))
    return runs


def capacity_window(t, soc, cur, direction, cfg, max_dt):
    """Implied capacity from ONE monotone SOC pass inside the SOC window (never the union of passes).
    direction=+1 for charging (SOC rising), -1 for discharging."""
    lo, hi = cfg["analysis"]["soc_window"]
    cov = cfg["analysis"]["min_window_coverage"]
    tol = cfg["analysis"]["soc_monotone_tol"]
    ok = np.isfinite(t) & np.isfinite(soc) & np.isfinite(cur)
    if ok.sum() < 50:
        return np.nan, "too_few_samples", 0
    t, soc, cur = t[ok], soc[ok], cur[ok]
    s = soc * direction
    # SOC is integer and sample-and-hold at 10 Hz: search monotone passes on the change points only
    chg = np.r_[0, np.where(np.diff(s) != 0)[0] + 1]
    ends = np.r_[chg[1:] - 1, len(s) - 1]
    best = None
    n_pass = 0
    for ra, rb in monotone_runs(s[chg], tol):
        a, b = chg[ra], ends[rb]
        sl = slice(a, b + 1)
        m = (soc[sl] >= lo) & (soc[sl] <= hi)
        if m.sum() < 25:
            continue
        n_pass += 1
        span = float(soc[sl][m].max() - soc[sl][m].min())
        if best is None or span > best[0]:
            best = (span, a, b, m)
    if best is None:
        return np.nan, "window_unpopulated", n_pass
    span, a, b, m = best
    if span < (hi - lo) * cov:
        return np.nan, f"window_coverage_{span:.0f}pct", n_pass
    sl = slice(a, b + 1)
    idx = np.where(m)[0]
    tt, ii = t[sl][idx], np.abs(cur[sl][idx])
    ah = _integrate(tt, ii, max_dt)
    return ah / (span / 100.0), "ok", n_pass


def r0_proxy(t, vmid_mv, cur, soc, cfg, med_dt):
    """Per-cell DC resistance proxy (mOhm) from windowed dV/dI; needs current variation and stable SOC."""
    n = max(30, int(60 / 60 / max(med_dt, 1e-6)))  # ~60 s window
    vals = []
    for a in range(0, len(t) - n, n):
        sl = slice(a, a + n)
        i, v, s = cur[sl], vmid_mv[sl], soc[sl]
        if not (np.isfinite(i).all() and np.isfinite(v).all()):
            continue
        if np.nanstd(i) < 2.0 or np.nanmax(s) - np.nanmin(s) > 1.0:
            continue
        c = np.corrcoef(i, v)[0, 1]
        if not np.isfinite(c) or c ** 2 < 0.5:
            continue
        slope = np.polyfit(i, v, 1)[0]  # mV/A = mOhm
        vals.append(-slope if slope < 0 else np.nan)
    vals = np.array([x for x in vals if np.isfinite(x)])
    return (float(np.median(vals)), len(vals)) if len(vals) else (np.nan, 0)


def charge_phase_stats(t, cur_abs, vmax, med_dt):
    """Look for a CC plateau followed by a CV-like taper in a charging segment.

    CC plateau  : longest run within +/-1% of the 90th-percentile current lasting >= max(5 min, 15% of the segment).
    CV taper    : AFTER the plateau, current decays monotonically to < 50% of plateau level while the max cell
                  voltage stays within 1% of its segment maximum.
    ramp_like   : charging current exists but no such plateau (smooth ramp / bell-shaped profile).
    Reports what the data support; a negative result is a result."""
    out = dict(has_cc=False, has_taper=False, plateau_min=0.0, taper_min=0.0, ramp_like=False)
    if len(t) < 100:
        return out
    k = max(5, int(0.5 / max(med_dt, 1e-6)))
    sm = pd.Series(cur_abs).rolling(k, center=True, min_periods=3).median().to_numpy()
    p90 = np.nanpercentile(sm, 90)
    if not np.isfinite(p90) or p90 < 3:
        return out
    near = np.abs(sm - p90) <= 0.01 * p90
    best, best_end, cur = 0, -1, 0
    for i, v in enumerate(near):
        cur = cur + 1 if v else 0
        if cur > best:
            best, best_end = cur, i
    seg_min = len(t) * med_dt
    out["plateau_min"] = best * med_dt
    out["has_cc"] = bool(out["plateau_min"] >= max(5.0, 0.15 * seg_min))
    if out["has_cc"]:
        tail = sm[best_end:]
        if len(tail) > 100 and np.isfinite(tail).all():
            ratio_ok = tail[-1] < 0.5 * p90
            mono = np.mean(np.diff(tail) <= 0.01 * p90) > 0.9
            vmx = np.nanmax(vmax)
            v_ok = np.nanmin(vmax[best_end:][-max(10, len(tail) // 4):]) >= vmx - 0.01 * vmx
            out["has_taper"] = bool(ratio_ok and mono and v_ok)
            out["taper_min"] = float(len(tail) * med_dt) if out["has_taper"] else 0.0
    out["ramp_like"] = bool((not out["has_cc"]) and np.nanmax(sm) > 3)
    return out


def flight_state(x: pd.DataFrame, t: np.ndarray, med_dt: float, cfg: dict) -> dict:
    """Ambient temperature and flight phases from the avionics block (only where it is alive: ACC_NORM ~ 1 g).
    Phases: ground (below 30 m above the session's ground level), climb (smoothed vertical rate > +1 m/s),
    descent (< -1 m/s), cruise (airborne otherwise). Altitude is assumed to be in metres."""
    out = dict(avionics_frac=0.0, oat_c_median=np.nan, oat_c_min=np.nan, oat_c_max=np.nan, airborne_min=0.0,
               climb_min=0.0, cruise_min=0.0, descent_min=0.0, alt_gain_m=np.nan, ias_max=np.nan, has_flight_state=False)
    if "avionics_valid" not in x or not np.isfinite(med_dt):
        return out
    av = x["avionics_valid"].to_numpy().astype(bool)
    out["avionics_frac"] = float(av.mean())
    if av.sum() < max(20, int(0.5 / med_dt)):
        return out
    if "oat" in x:
        oat = pd.to_numeric(x["oat"], errors="coerce").to_numpy(float)[av]
        if np.isfinite(oat).any():
            out.update(oat_c_median=float(np.nanmedian(oat)), oat_c_min=float(np.nanmin(oat)), oat_c_max=float(np.nanmax(oat)))
    if "ias" in x:
        out["ias_max"] = float(np.nanmax(pd.to_numeric(x["ias"], errors="coerce").to_numpy(float)[av]))
    if "pressure_alt" in x:
        alt = pd.to_numeric(x["pressure_alt"], errors="coerce").to_numpy(float)
        alt = np.where(av, alt, np.nan)
        if np.isfinite(alt).sum() > 50:
            ground = np.nanpercentile(alt, 5)
            agl = alt - ground
            k = max(3, int(10 / 60 / med_dt))  # 10 s smoothing
            sm = pd.Series(agl).rolling(k, center=True, min_periods=2).median().to_numpy()
            rate = np.gradient(sm, t * 60.0)  # m/s
            air = sm > 30
            out["alt_gain_m"] = float(np.nanmax(agl))
            out["airborne_min"] = float(air.sum() * med_dt)
            out["climb_min"] = float((air & (rate > 1)).sum() * med_dt)
            out["descent_min"] = float((air & (rate < -1)).sum() * med_dt)
            out["cruise_min"] = float((air & (np.abs(rate) <= 1)).sum() * med_dt)
            out["has_flight_state"] = bool(out["airborne_min"] > 1)
    return out


FEATURE_COLS = (["time_min", "quality_flag", "init_unsettled_groups", "segment_type", "avionics_valid"] +
                [f"bat_{b}_{s}" for b in (1, 2) for s in ("current", "soc", "soh", "min_cell_volt", "max_cell_volt",
                                                           "max_cell_temp", "min_cell_temp", "avg_cell_temp", "voltage")] +
                ["oat", "pressure_alt", "ias", "acc_norm", "motor_power"])


def extract_flight(d: pd.DataFrame, cfg: dict, rule: str = DEFAULT_RULE, seg_current_a: float | None = None,
                   time_rule: str = "logged") -> list[dict]:
    """One dict per (flight, battery)."""
    cs = cfg["charge_sign"]
    x = apply_rule(d, rule)
    t = x["time_min"].to_numpy(float)
    ok_t = np.isfinite(t)
    dts = np.diff(t)
    pos = dts[np.isfinite(dts) & (dts > 0)]
    med_dt = float(np.median(pos)) if len(pos) else np.nan
    # an interval counts as a logging gap only if longer than gap_factor x median step AND gap_min_s
    max_dt = max(cfg["quality"]["gap_factor"] * med_dt, cfg["quality"].get("gap_min_s", 2.0) / 60.0) if np.isfinite(med_dt) else np.inf
    # timestamp-alignment rules (sensitivity): logged | uniform (index x median dt) | gapbridge (integrate across gaps)
    if time_rule == "uniform" and np.isfinite(med_dt):
        t = np.arange(len(t)) * med_dt
    elif time_rule == "gapbridge":
        max_dt = np.inf
    st = x["segment_type"].to_numpy()
    if seg_current_a is not None:
        from ..build.clean import segment_activity
        cur_all = np.nanmean(np.vstack([d["bat_1_current"], d["bat_2_current"]]), axis=0)
        c2 = dict(cfg)
        c2["segmentation"] = dict(cfg["segmentation"], active_current_a=seg_current_a)
        _, st = segment_activity(d["time_min"].to_numpy(float), cur_all, c2)
    # drivetrain cross-check: sum of both packs' V x I against the logged motor power
    ptot = np.zeros(len(x))
    okp = np.ones(len(x), bool)
    for bb in (1, 2):
        vb = x[f"bat_{bb}_voltage"].to_numpy(float) if f"bat_{bb}_voltage" in x else np.full(len(x), np.nan)
        ib = x[f"bat_{bb}_current"].to_numpy(float)
        pb = -cs * ib * np.where(vb > 50, vb, np.nan) / 1000.0
        ptot = ptot + np.nan_to_num(pb)
        okp &= np.isfinite(pb)
    ptot = np.where(okp, ptot, np.nan)
    flight_level = dict(pack_total_power_peak_kw=float(np.nanmax(ptot)) if np.isfinite(ptot).any() else np.nan)
    if "motor_power" in x:
        mp = pd.to_numeric(x["motor_power"], errors="coerce").to_numpy(float)
        flight_level["motor_power_peak_kw"] = float(np.nanmax(mp)) if np.isfinite(mp).any() else np.nan
        m = np.isfinite(mp) & np.isfinite(ptot) & (ptot > 5) & (mp > 0)
        flight_level["motor_over_pack_power_median"] = float(np.median(mp[m] / ptot[m])) if m.sum() > 50 else np.nan
    out = []
    for b in (1, 2):
        cur = x[f"bat_{b}_current"].to_numpy(float)
        soc = x[f"bat_{b}_soc"].to_numpy(float)
        soh = x[f"bat_{b}_soh"].to_numpy(float)
        vmin = x[f"bat_{b}_min_cell_volt"].to_numpy(float)
        vmax = x[f"bat_{b}_max_cell_volt"].to_numpy(float)
        tmax = x[f"bat_{b}_max_cell_temp"].to_numpy(float)
        tavg = x[f"bat_{b}_avg_cell_temp"].to_numpy(float) if f"bat_{b}_avg_cell_temp" in x else np.full(len(x), np.nan)
        dis = -cs * cur  # positive when discharging
        r = dict(flight_id=int(d["flight_id"].iloc[0]), battery=b, rule=rule, **flight_level,
                 duration_min=float(np.nanmax(t) - np.nanmin(t)) if ok_t.any() else np.nan, n_rows=len(x))
        fin = np.isfinite(soc)
        r["soc_start"] = float(soc[fin][0]) if fin.any() else np.nan
        r["soc_end"] = float(soc[fin][-1]) if fin.any() else np.nan
        r["soc_min"] = float(np.nanmin(soc)) if fin.any() else np.nan
        r["soc_max"] = float(np.nanmax(soc)) if fin.any() else np.nan
        r["delta_soc"] = r["soc_end"] - r["soc_start"]
        r["soh_start"] = float(soh[np.isfinite(soh)][0]) if np.isfinite(soh).any() else np.nan
        r["soh_end"] = float(soh[np.isfinite(soh)][-1]) if np.isfinite(soh).any() else np.nan
        r["soh_median"] = float(np.nanmedian(soh)) if np.isfinite(soh).any() else np.nan
        r["v_min_mv"] = float(np.nanmin(vmin)) if np.isfinite(vmin).any() else np.nan
        r["v_max_mv"] = float(np.nanmax(vmax)) if np.isfinite(vmax).any() else np.nan
        r["v_spread_mv_median"] = float(np.nanmedian(vmax - vmin)) if np.isfinite(vmax - vmin).any() else np.nan
        r["i_peak_discharge_a"] = float(np.nanmax(dis)) if np.isfinite(dis).any() else np.nan
        r["i_peak_charge_a"] = float(np.nanmax(-dis)) if np.isfinite(dis).any() else np.nan
        r["i_mean_abs_a"] = float(np.nanmean(np.abs(cur))) if np.isfinite(cur).any() else np.nan
        r["ah_discharged"] = _integrate(t, np.clip(dis, 0, None), max_dt)
        r["ah_charged"] = _integrate(t, np.clip(-dis, 0, None), max_dt)
        r["ah_throughput"] = r["ah_discharged"] + r["ah_charged"]
        r["ah_net_discharge"] = r["ah_discharged"] - r["ah_charged"]
        r["temp_max_c"] = float(np.nanmax(tmax)) if np.isfinite(tmax).any() else np.nan
        r["temp_start_c"] = float(tmax[np.isfinite(tmax)][0]) if np.isfinite(tmax).any() else np.nan
        r["temp_delta_c"] = r["temp_max_c"] - r["temp_start_c"]
        r["temp_avg_mean_c"] = float(np.nanmean(tavg)) if np.isfinite(tavg).any() else np.nan
        r["frac_charge"] = float((st == "charge").mean())
        r["frac_discharge"] = float((st == "discharge").mean())
        n_ser = cfg.get("n_series_cells")
        vmean = 0.5 * (vmin + vmax) / 1000.0
        vpack = x[f"bat_{b}_voltage"].to_numpy(float) if f"bat_{b}_voltage" in x else np.full(len(x), np.nan)
        vpack = np.where(vpack > 50, vpack, np.nan)  # zeros = not yet logged
        if np.isfinite(vpack).any():
            r["power_source"] = "pack voltage x current"
        elif n_ser:
            vpack = vmean * n_ser
            r["power_source"] = f"mean cell voltage x {n_ser} x current"
        else:
            r["power_source"] = "unavailable"
        p = dis * vpack / 1000.0  # kW, positive = discharge
        r["power_peak_kw"] = float(np.nanmax(p)) if np.isfinite(p).any() else np.nan
        r["power_mean_discharge_kw"] = float(np.nanmean(p[p > 0])) if np.isfinite(p).any() and (p > 0).any() else np.nan
        r["energy_discharged_kwh"] = _integrate(t, np.clip(p, 0, None), max_dt)
        r["energy_charged_kwh"] = _integrate(t, np.clip(-p, 0, None), max_dt)
        r["v_pack_start"] = float(vpack[np.isfinite(vpack)][0]) if np.isfinite(vpack).any() else np.nan
        r["v_pack_min"] = float(np.nanmin(vpack)) if np.isfinite(vpack).any() else np.nan
        r["temp_min_cell_min_c"] = float(np.nanmin(x[f"bat_{b}_min_cell_temp"])) if f"bat_{b}_min_cell_temp" in x and x[f"bat_{b}_min_cell_temp"].notna().any() else np.nan
        r.update(flight_state(x, t, med_dt, cfg))
        # rate-weighted power index (per-cell W) always available
        pi = np.clip(dis, 0, None) * vmean
        r["power_index_peak_w_per_cell"] = float(np.nanmax(pi)) if np.isfinite(pi).any() else np.nan
        # capacity proxies
        direction = 1 if r["frac_charge"] > r["frac_discharge"] else -1
        q, why, npass = capacity_window(t, soc, cur, direction, cfg, max_dt)
        r.update(implied_capacity_ah=q, capacity_reason=why, capacity_n_passes=npass)
        r0, nr0 = r0_proxy(t, 0.5 * (vmin + vmax), cur, soc, cfg, med_dt)
        r.update(r0_proxy_mohm_cell=r0, r0_n_windows=nr0)
        if np.isfinite(vpack).any():
            r0p, nr0p = r0_proxy(t, vpack * 1000.0, cur, soc, cfg, med_dt)  # mV/A = mOhm on the pack
            r.update(r0_pack_mohm=r0p, r0_pack_n_windows=nr0p)
        # charging phases
        cm = (st == "charge")
        if cm.sum() > 100:
            ph = charge_phase_stats(t[cm], np.abs(cur[cm]), vmax[cm], med_dt)
            r.update({f"chg_{k}": v for k, v in ph.items()})
            r["charge_time_min"] = float(cm.sum() * med_dt)
            r["charge_soc_start"] = float(soc[cm][np.isfinite(soc[cm])][0]) if np.isfinite(soc[cm]).any() else np.nan
            r["charge_soc_end"] = float(soc[cm][np.isfinite(soc[cm])][-1]) if np.isfinite(soc[cm]).any() else np.nan
            r["charge_i_mean_a"] = float(np.nanmean(np.abs(cur[cm])))
            r["charge_i_peak_a"] = float(np.nanmax(np.abs(cur[cm])))
            sm_i = pd.Series(np.abs(cur[cm])).rolling(max(5, int(0.5 / med_dt)), center=True, min_periods=3).median()
            vmid_c = 0.5 * (vmin[cm] + vmax[cm])
            r["dqdv_current_cv"] = float(sm_i.std() / sm_i.mean()) if sm_i.mean() > 0 else np.nan
            r["dqdv_n_vbins"] = int(len(np.unique(np.round(vmid_c[np.isfinite(vmid_c)] / 10.0))))
            r["dqdv_usable"] = bool(ph["has_cc"] and r["dqdv_current_cv"] < 0.1 and
                                    (r["charge_soc_end"] - r["charge_soc_start"]) >= 30 and r["dqdv_n_vbins"] >= 20)
        out.append(r)
    return out


def extract_all(settings, rule: str = DEFAULT_RULE, seg_current_a=None, flight_ids=None, time_rule: str = "logged") -> pd.DataFrame:
    path = settings.data_dir / "processed" / "analysis_ready_timeseries.parquet"
    rows = []
    for g in iter_flights(path, FEATURE_COLS):
        if flight_ids is not None and int(g["flight_id"].iloc[0]) not in flight_ids:
            continue
        rows.extend(extract_flight(g, settings.raw, rule, seg_current_a, time_rule))
    return pd.DataFrame(rows)


def read_flight(settings, fid: int) -> pd.DataFrame:
    import pyarrow.dataset as ds
    path = settings.data_dir / "processed" / "analysis_ready_timeseries.parquet"
    return ds.dataset(path).to_table(filter=ds.field("flight_id") == int(fid)).to_pandas()
