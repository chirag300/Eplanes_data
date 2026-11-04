"""Per-session physics observables. Input: the analysis-ready 10 Hz table. Nothing here uses reported SOH or SOC to compute
a health quantity; both are only copied through so later stages can compare against them or detect unlogged sessions.

For every (session, pack) this stage stores
  * the charger's step-down and the take-off current ramp, which give resistance from voltage and current alone
  * rest voltages at the start and end of the session (low-current windows, later IR-corrected)
  * charge in and out, energy, temperature, cell-voltage spread and load statistics
  * a 10 s curve (time, current, charge, voltage, min/max cell voltage, temperature) for the capacity stages
Results are cached per session, so an interrupted run resumes and later stages never touch the 10 Hz table again.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from ..analysis.features import apply_rule, capacity_window
from ..build.clean import FLAG_BITS, iter_flights
from .common import Ctx, Log, load_sessions, write_json

VERSION = 4
# every column later stages read, so a dataset where (say) no charger step is ever found gives empty columns rather than a crash
EXPECTED = ["step_found", "step_t_s", "step_i_pre_a", "step_i_post_a", "r_step_mohm", "r_step30_mohm", "step_temp_c", "ramp_found", "r_ramp_mohm", "ramp_temp_c", "ramp_r2",
            "rest_start_v", "rest_start_i_a", "rest_start_q_ah", "rest_end_v", "rest_end_i_a", "charge_start_s", "charge_stop_s", "v_load_min_cell", "kwh_out", "kwh_in",
            "q_soc_window_ah", "q_end_ah", "ah_in", "ah_out", "soc_first", "soc_last", "soh_bms_median", "soh_bms_first", "soh_bms_last", "temp_mean_c", "charge_min", "discharge_min",
            "v_max_cell", "n_segments_q"]
BAD_BITS = FLAG_BITS["out_of_range"] | FLAG_BITS["nonmonotonic_time"] | FLAG_BITS["sentinel_value"] | FLAG_BITS["zero_sample_after_init"]
COLS = ["time_min", "quality_flag", "init_unsettled_groups", "aircraft_label"] + \
    [f"bat_{b}_{s}" for b in (1, 2) for s in ("current", "voltage", "min_cell_volt", "max_cell_volt", "soc", "soh",
                                              "max_cell_temp", "min_cell_temp", "avg_cell_temp")]


def _fine_grid(t, y_dict, bin_s):
    """Mean of every signal in fixed time bins; bins without samples stay NaN."""
    idx = np.floor((t - t[0]) / bin_s).astype(np.int64)
    n = int(idx[-1]) + 1
    out = {}
    cnt = np.bincount(idx, minlength=n).astype(float)
    for k, y in y_dict.items():
        ok = np.isfinite(y)
        s = np.bincount(idx[ok], weights=y[ok], minlength=n)
        c = np.bincount(idx[ok], minlength=n).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[k] = np.where(c > 0, s / c, np.nan)
    out["_count"] = cnt
    return t[0] + np.arange(n) * bin_s, out


def _bridge(y, max_bins):
    """Linear interpolation across NaN runs up to max_bins long. Returns the filled series and a mask of longer gaps."""
    y = y.copy()
    fin = np.isfinite(y)
    if fin.sum() < 2:
        return y, ~fin
    idx = np.arange(len(y))
    filled = np.interp(idx, idx[fin], y[fin])
    gap = np.zeros(len(y), bool)
    change = np.flatnonzero(np.diff(np.r_[0, (~fin).astype(int), 0]))
    for a, b in zip(change[::2], change[1::2]):
        if b - a <= max_bins and a > 0 and b < len(y):
            y[a:b] = filled[a:b]
        else:
            gap[a:b] = True
    return y, gap


def _rolling_median(y, n):
    return pd.Series(y).rolling(n, center=True, min_periods=max(1, n // 3)).median().to_numpy()


def _longest_run(mask, allow_gap=0):
    """Longest run of True, allowing short False interruptions. Returns (start, stop) or None."""
    m = mask.copy()
    if allow_gap:
        change = np.flatnonzero(np.diff(np.r_[0, (~m).astype(int), 0]))
        for a, b in zip(change[::2], change[1::2]):
            if b - a <= allow_gap and a > 0 and b < len(m):
                m[a:b] = True
    change = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
    if not len(change):
        return None
    runs = list(zip(change[::2], change[1::2]))
    return max(runs, key=lambda r: r[1] - r[0])


def charger_step(i, v, temp, q, run, ec):
    """Resistance at the charger's step-down: same state of charge either side, so dV/dI is purely resistive."""
    out = dict(step_found=False)
    a, b = run
    ia = np.abs(i)
    fw = 4
    for k in range(a + 60, b - 40):
        if not (np.isfinite(ia[k - fw:k + fw + 1]).all()):
            continue
        pre, post = ia[k - fw - 2:k - 1], ia[k + 2:k + fw + 3]
        if len(pre) < 3 or len(post) < 3:
            continue
        pm, qm = np.median(pre), np.median(post)
        if pm < ec["step_from_min_a"] or pm - qm < ec["step_min_di_a"] or pre.max() - pre.min() > 1.5 or post.max() - post.min() > 1.5:
            continue
        vp, vq = v[k - fw - 2:k - 1], v[k + 2:k + fw + 3]
        if not (np.isfinite(vp).sum() >= 3 and np.isfinite(vq).sum() >= 3):
            continue
        di = pm - qm
        r0 = (np.nanmedian(vp) - np.nanmedian(vq)) / di / ec["n_series"] * 1000.0
        out.update(step_found=True, step_t_s=float(k), step_i_pre_a=float(pm), step_i_post_a=float(qm), r_step_mohm=float(r0),
                   step_v_pre=float(np.nanmedian(vp)), step_v_post=float(np.nanmedian(vq)), step_q_ah=float(q[k]),
                   step_temp_c=float(np.nanmedian(temp[max(0, k - 30):k + 30])))
        if k + 34 < len(v) and np.isfinite(v[k + 26:k + 35]).sum() >= 4 and np.nanmax(ia[k + 2:k + 35]) - np.nanmin(ia[k + 2:k + 35]) < 2.0:
            out["r_step30_mohm"] = float((np.nanmedian(vp) - np.nanmedian(v[k + 26:k + 35])) / di / ec["n_series"] * 1000.0)
        break
    return out


def takeoff_ramp(i, v, temp, ec):
    """Resistance from the biggest current rise of the session: voltage regressed on current during the ramp."""
    out = dict(ramp_found=False)
    n = int(ec["ramp_window_s"])
    fin = np.isfinite(i) & np.isfinite(v)
    if fin.sum() < n + 10:
        return out
    ii = np.where(fin, i, 0.0)
    rise = np.r_[np.full(n, -np.inf), ii[n:] - ii[:-n]]
    k = int(np.argmax(rise))
    if rise[k] < ec["ramp_min_di_a"]:
        return out
    a, b = max(0, k - n - 2), min(len(i), k + 3)
    m = fin[a:b]
    if m.sum() < 6:
        return out
    x, y = i[a:b][m], v[a:b][m]
    if np.ptp(x) < ec["ramp_min_di_a"]:
        return out
    slope, icpt = np.polyfit(x, y, 1)
    pred = slope * x + icpt
    ss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - float(np.sum((y - pred) ** 2)) / ss if ss > 0 else 0.0
    if slope >= 0 or r2 < 0.8:
        return out
    out.update(ramp_found=True, r_ramp_mohm=float(-slope / ec["n_series"] * 1000.0), ramp_r2=r2, ramp_di_a=float(np.ptp(x)), ramp_t_s=float(k),
               ramp_temp_c=float(np.nanmedian(temp[max(0, k - 60):k + 30])), ramp_i_peak_a=float(np.nanmax(x)),
               ramp_v_before=float(np.nanmedian(v[a:a + 3])))
    return out


def rest_window(i, v, q, temp, ec, end=False):
    """First (or last) low-current window of the session. Voltage is stored with its current so it can be IR-corrected later."""
    keep = np.isfinite(i) & np.isfinite(v)
    idx = np.flatnonzero(keep)
    if not len(idx):
        return {}
    low = np.abs(i) <= ec["rest_max_a"]
    if end:
        k = idx[-1]
        if not low[k]:
            return {}
        a = k
        while a > 0 and k - a < 30 and low[a - 1] and keep[a - 1]:
            a -= 1
        sl = slice(a, k + 1)
    else:
        k = idx[0]
        if not low[k]:
            return {}
        b = k
        while b + 1 < len(i) and b - k < 30 and low[b + 1] and keep[b + 1]:
            b += 1
        sl = slice(k, b + 1)
    n = sl.stop - sl.start
    if n < ec["rest_min_s"]:
        return {}
    tag = "rest_end" if end else "rest_start"
    return {f"{tag}_v": float(np.median(v[sl])), f"{tag}_i_a": float(np.mean(i[sl])), f"{tag}_s": float(n), f"{tag}_q_ah": float(np.median(q[sl])),
            f"{tag}_t_s": float(sl.start), f"{tag}_temp_c": float(np.nanmedian(temp[sl])) if np.isfinite(temp[sl]).any() else np.nan}


def pack_session(tg, sig, base_cfg, ec, t_min, soc_raw, cur_raw):
    """All observables of one pack in one session."""
    bin_s = ec["bin_s"]
    i, gap = _bridge(sig["i"], int(ec["max_bridge_s"] / bin_s))
    v = sig["v"] / ec["n_series"]                 # mean-cell voltage, V
    vmin, vmax = sig["vmin"] / 1000.0, sig["vmax"] / 1000.0
    temp = pd.Series(sig["tavg"]).ffill().bfill().to_numpy()
    tmax = pd.Series(sig["tmax"]).ffill().bfill().to_numpy()
    dq = np.where(np.isfinite(i), -i, 0.0) * bin_s / 3600.0     # + = charge going in
    q = np.cumsum(dq)
    seg = np.cumsum(np.r_[0, (gap[1:] & ~gap[:-1]).astype(int)])  # a long gap starts a new charge-counting segment
    is_ = _rolling_median(i, max(5, int(15 / bin_s)))
    r = dict(n_bins=len(i), gap_long_s=float(gap.sum() * bin_s), n_segments_q=int(seg.max()) + 1)
    if not np.isfinite(i).any() or not np.isfinite(v).any():
        return r, None
    fin = np.isfinite(i)
    r["ah_in"] = float(np.sum(np.clip(-i[fin], 0, None)) * bin_s / 3600)
    r["ah_out"] = float(np.sum(np.clip(i[fin], 0, None)) * bin_s / 3600)
    r["q_end_ah"] = float(q[-1])
    r["i_rms_a"] = float(np.sqrt(np.mean(i[fin] ** 2)))
    r["i_peak_dis_a"] = float(np.max(i[fin]))
    r["i_peak_chg_a"] = float(np.max(-i[fin]))
    p = np.nan_to_num(i * sig["v"]) / 1000.0
    r["kwh_out"] = float(np.sum(np.clip(p, 0, None)) * bin_s / 3600)
    r["kwh_in"] = float(np.sum(np.clip(-p, 0, None)) * bin_s / 3600)
    r["temp_mean_c"] = float(np.nanmean(temp)) if np.isfinite(temp).any() else np.nan
    r["temp_max_c"] = float(np.nanmax(tmax)) if np.isfinite(tmax).any() else np.nan
    r["temp_start_c"] = float(temp[0]) if np.isfinite(temp[0]) else np.nan
    r["temp_rise_c"] = float(np.nanmax(temp) - temp[0]) if np.isfinite(temp).any() else np.nan
    r["v_min_cell"] = float(np.nanmin(vmin)) if np.isfinite(vmin).any() else np.nan
    r["v_max_cell"] = float(np.nanmax(vmax)) if np.isfinite(vmax).any() else np.nan
    sp = vmax - vmin
    r["spread_mv_median"] = float(np.nanmedian(sp) * 1000) if np.isfinite(sp).any() else np.nan
    r["spread_mv_max"] = float(np.nanmax(sp) * 1000) if np.isfinite(sp).any() else np.nan
    r["spread_mv_end"] = float(np.nanmedian(sp[-60:]) * 1000) if np.isfinite(sp[-60:]).any() else np.nan
    chg = _longest_run(is_ < -ec["charge_on_a"], int(60 / bin_s))
    dis = _longest_run(is_ > ec["charge_on_a"], int(120 / bin_s))
    r["charge_min"] = float((chg[1] - chg[0]) * bin_s / 60) if chg else 0.0
    r["discharge_min"] = float((dis[1] - dis[0]) * bin_s / 60) if dis else 0.0
    if chg and r["charge_min"] >= 5:
        r["charge_start_s"], r["charge_stop_s"] = float(chg[0] * bin_s), float(chg[1] * bin_s)
        r["i_chg_start_a"] = float(np.nanmedian(-i[chg[0] + 60:chg[0] + 120])) if chg[1] - chg[0] > 130 else np.nan
        r.update(charger_step(i, sig["v"], temp, q, chg, ec))
    if dis and r["discharge_min"] >= 3:
        r.update(takeoff_ramp(i, sig["v"], temp, ec))
        r["v_load_min_cell"] = float(np.nanmin(v[dis[0]:dis[1]]))
    for e in (False, True):
        r.update(rest_window(i, v, q, temp, ec, e))
    # copied through for comparison and for detecting unlogged sessions only
    so = sig["soc"][np.isfinite(sig["soc"])]
    sh = sig["soh"][np.isfinite(sig["soh"]) & (sig["soh"] > 0)]
    r["soc_first"] = float(so[0]) if len(so) else np.nan
    r["soc_last"] = float(so[-1]) if len(so) else np.nan
    r["soh_bms_first"] = float(sh[0]) if len(sh) else np.nan
    r["soh_bms_last"] = float(sh[-1]) if len(sh) else np.nan
    r["soh_bms_median"] = float(np.median(sh)) if len(sh) else np.nan
    direction = 1 if r["charge_min"] > r["discharge_min"] else -1
    try:
        qw, why, _ = capacity_window(t_min, soc_raw, cur_raw, direction, base_cfg, 2.0 / 60.0)
    except Exception:  # noqa: BLE001
        qw, why = np.nan, "error"
    r["q_soc_window_ah"], r["q_soc_window_reason"] = (float(qw) if np.isfinite(qw) else np.nan), why
    # coarse curve
    nb = max(1, int(ec["curve_bin_s"] / bin_s))
    m = len(i) // nb
    if m < 3:
        return r, None
    def red(y, fn=np.nanmean):
        z = y[: m * nb].reshape(m, nb)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return fn(z, axis=1)
    curve = dict(t=(np.arange(m) + 0.5) * nb * bin_s, i=red(i), q=q[nb - 1::nb][:m], v=red(v), vmin=red(vmin), vmax=red(vmax), temp=red(temp),
                 seg=seg[nb - 1::nb][:m].astype(np.int16), soc=red(sig["soc"]))
    curve = {k: (x.astype(np.float32) if x.dtype.kind == "f" else x) for k, x in curve.items()}
    return r, curve


def extract_session(g: pd.DataFrame, base_cfg: dict, cfg: dict) -> tuple[dict, dict]:
    ec = cfg["extraction"]
    x = apply_rule(g, ec["rule"])
    t = x["time_min"].to_numpy(float) * 60.0
    keep = np.isfinite(t)
    bad = (x["quality_flag"].to_numpy(np.int64) & BAD_BITS) > 0
    keep &= t - np.nanmin(t) < ec.get("max_session_h", 12.0) * 3600.0     # a corrupt time stamp must not create billions of empty bins
    t_ok = t[keep]
    if len(t_ok) < 30:
        return {"packs": {}}, {}
    order = np.argsort(t_ok, kind="stable")
    res, curves = {}, {}
    tt = t_ok[order]
    for b in (1, 2):
        sig_raw = {"i": x[f"bat_{b}_current"], "v": x[f"bat_{b}_voltage"], "vmin": x[f"bat_{b}_min_cell_volt"], "vmax": x[f"bat_{b}_max_cell_volt"],
                   "soc": x[f"bat_{b}_soc"], "soh": x[f"bat_{b}_soh"], "tmax": x[f"bat_{b}_max_cell_temp"], "tavg": x.get(f"bat_{b}_avg_cell_temp", x[f"bat_{b}_max_cell_temp"])}
        sig = {}
        for k, s in sig_raw.items():
            y = pd.to_numeric(s, errors="coerce").to_numpy(float).copy()
            y[bad] = np.nan
            if k == "v":
                y[y < 50] = np.nan
            if k in ("vmin", "vmax"):
                y[y < 1000] = np.nan
            sig[k] = y[keep][order]      # a temperature of exactly 0 degC is real here; leading start-up zeros are already masked by the rule
        if not np.isfinite(sig["tavg"]).any():
            sig["tavg"] = sig["tmax"]    # older exports lack the average for pack 2
        grid, fine = _fine_grid(tt, sig, ec["bin_s"])
        fine.pop("_count")
        tmin_full = tt / 60.0
        rec, curve = pack_session(grid, fine, base_cfg, ec, tmin_full, sig["soc"], sig["i"])
        res[str(b)] = rec
        if curve is not None:
            curves[str(b)] = curve
    return {"packs": res}, curves


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps([VERSION, cfg["extraction"]], sort_keys=True).encode()).hexdigest()[:12]


def run_extract(ctx: Ctx, log=None, limit=None) -> pd.DataFrame:
    log = log or Log(ctx)
    path = ctx.proc / "analysis_ready_timeseries.parquet"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run `python -m eplanes build-clean` first.")
    sessions = load_sessions(ctx)
    meta = sessions.set_index("flight_id")
    h = config_hash(ctx.cfg)
    sdir = ctx.cache / "sessions"
    done, new = 0, 0
    for g in iter_flights(path, COLS):
        fid = int(g["flight_id"].iloc[0])
        jp, np_ = sdir / f"flight_{fid}.json", sdir / f"flight_{fid}.npz"
        if jp.exists() and json.loads(jp.read_text()).get("hash") == h:
            done += 1
            continue
        if limit and new >= limit:
            break
        rec, curves = extract_session(g, ctx.settings.raw, ctx.cfg)
        rec.update(flight_id=fid, hash=h)
        np.savez_compressed(np_, **{f"{b}_{k}": v for b, c in curves.items() for k, v in c.items()})
        write_json(jp, rec)
        new += 1
        if new % 100 == 0:
            log(f"  extracted {new} new sessions ({done} cached)")
    rows = []
    for jp in sorted(sdir.glob("flight_*.json")):
        rec = json.loads(jp.read_text())
        fid = rec["flight_id"]
        if fid not in meta.index:
            continue
        m = meta.loc[fid]
        for b, r in rec.get("packs", {}).items():
            rows.append(dict(flight_id=fid, battery=int(b), aircraft_label=m.aircraft_label, fleet=m.fleet, flight_type=m.flight_type,
                             start_datetime=m.start_datetime, duration_min=m.duration_min, usable_electrical=bool(m.usable_electrical), **r))
    if not rows:
        raise SystemExit("no session could be extracted: check that flights_with_qc.csv and the parquet file come from the same run")
    obs = pd.DataFrame(rows).sort_values(["aircraft_label", "battery", "start_datetime", "flight_id"]).reset_index(drop=True)
    for c in EXPECTED:
        if c not in obs:
            obs[c] = np.nan
    obs.to_parquet(ctx.cache / "session_observables.parquet", index=False)
    u = obs[obs.usable_electrical]
    log(f"[extract] {len(obs)} pack-sessions ({new} new, {done} cached); usable {len(u)}")
    tab = u.groupby(["aircraft_label", "battery"]).agg(pack_sessions=("flight_id", "size"), charges=("charge_min", lambda s: int((s >= 5).sum())),
                                                       charger_step=("step_found", lambda s: int(s.fillna(False).astype(bool).sum())),
                                                       takeoff_ramp=("ramp_found", lambda s: int(s.fillna(False).astype(bool).sum())),
                                                       rest_start=("rest_start_v", lambda s: int(s.notna().sum())),
                                                       r_step_mohm=("r_step_mohm", "median"), r_ramp_mohm=("r_ramp_mohm", "median"),
                                                       step_i_pre=("step_i_pre_a", "median"), step_i_post=("step_i_post_a", "median")).reset_index()
    tab.to_csv(ctx.tables / "E1_extraction_coverage.csv", index=False)
    log(tab.to_string(index=False))
    return obs
