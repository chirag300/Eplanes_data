"""Systematic analysis of unreliable readings at the start of a session.

Nothing here deletes data. For every flight x signal we measure how the first
samples relate to the signal's own "settled" value, and classify the pattern
with explicit, falsifiable rules:

  zero_fill           first sample is exactly 0 while the settled value is clearly non-zero
                      (logger/BMS had not populated the channel yet)
  bin_avg_zero        an early sample equals (k-1)/k of the settled value for small k
                      (consistent with a resampling bin that averaged in a zero-filled sample)
  monotone_settling   non-zero deviation that decays towards the settled value
                      (relaxation, filter warm-up or sensor warm-up; NOT separable by data alone
                      when the session starts at rest)
  load_onset          session starts under load: deviations coincide with real current changes,
                      so a physical transient cannot be excluded -> never flagged as artefact
  stable              nothing beyond (at most) the first sample deviates
  not_assessable      too few valid samples in the reference window

The labels say "consistent with", they do not prove a cause. Cause attribution
that needs BMS internals is listed as unresolved in the quality report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FRACS = {k: (k - 1) / k for k in range(2, 8)}


def robust_sigma(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


def resolution(x: np.ndarray) -> float:
    """Smallest non-zero step between distinct values (quantisation)."""
    u = np.unique(x[np.isfinite(x)])
    d = np.diff(u)
    d = d[d > 1e-9]
    return float(d.min()) if len(d) else 0.0


def start_state(t, current, active_a, win_min=1.0) -> str:
    """'rest_start' if |I| stays below the activity threshold in the first minute
    (excluding row 0), else 'load_start'."""
    m = (t > 0) & (t <= win_min)
    c = np.abs(current[m & np.isfinite(current)])
    if len(c) < 5:
        return "unknown"
    return "rest_start" if np.median(c) < active_a else "load_start"


def settle_info(t: np.ndarray, x: np.ndarray, cfg_q: dict, state: str, signal: str) -> dict:
    """Characterise the start of one signal in one flight."""
    out = dict(signal=signal, n=len(x), x0=np.nan, x1=np.nan, ref=np.nan, tol=np.nan,
               settle_idx=np.nan, settle_min=np.nan, dev0_rel=np.nan, ratio1=np.nan,
               cls="not_assessable", frac_k=np.nan)
    if len(x) < 10:
        return out
    w0, w1 = cfg_q["settle_window_min"]
    m = (t >= w0) & (t <= w1) & np.isfinite(x)
    if m.sum() < 20:  # short flight: use the second half of the first 3 minutes
        m = (t >= t.max() * 0.4) & (t <= t.max() * 0.7) & np.isfinite(x)
    if m.sum() < 20:
        return out
    ref = float(np.median(x[m]))
    # noise from first differences so a real trend inside the window does not inflate the tolerance
    sig = robust_sigma(np.diff(x[m])) / np.sqrt(2)
    k0 = min(8, len(x) - 1)
    res = resolution(x[k0:][np.isfinite(x[k0:])])  # exclude the first samples so a contaminated start does not inflate the step
    tol = max(cfg_q["settle_k_mad"] * (sig if np.isfinite(sig) else 0.0), res, 1e-9)
    # linear reference fitted inside the window and extrapolated back: a genuine trend (SOC falling,
    # temperature rising) must not be mistaken for start-up settling
    coef = np.polyfit(t[m], x[m], 1)
    refv = np.polyval(coef, t)
    early = (t <= w1) & np.isfinite(x)
    viol = early & (np.abs(x - refv) > tol)
    # only the CONTIGUOUS run of deviating samples that starts the record counts as start-up behaviour
    # (up to 3 clean samples tolerated inside it); later deviations are real signal dynamics
    last, gap = -1, 0
    for i, v in enumerate(viol):
        if v:
            last, gap = i, 0
        else:
            gap += 1
            if gap > 3:
                break
    idx = last + 1
    out.update(x0=float(x[0]) if np.isfinite(x[0]) else np.nan,
               x1=float(x[1]) if np.isfinite(x[1]) else np.nan,
               ref=ref, tol=tol, settle_idx=idx, settle_min=float(t[min(idx, len(t) - 1)]) if idx else 0.0)
    scale = max(abs(ref), tol)
    out["dev0_rel"] = float((x[0] - ref) / scale) if np.isfinite(x[0]) else np.nan
    out["ratio1"] = float(x[1] / ref) if abs(ref) > 5 * tol and np.isfinite(x[1]) else np.nan

    nonzero_ref = abs(ref) > max(5 * tol, 1e-6)
    zero0 = np.isfinite(x[0]) and x[0] == 0.0 and nonzero_ref
    # bin-average contamination: any of the first 6 samples ~ (k-1)/k * ref
    frac_k = np.nan
    if nonzero_ref:
        for i in range(1, min(7, len(x))):
            r = x[i] / ref
            # averaging a zero into a bin produces a value OFF the sensor's resolution grid (e.g. 10.667 for an
            # integer-degree sensor); an on-grid value (18 vs 21) is ordinary signal, not bin contamination
            grid = res if res > 0 else 0.1  # constant signal: fall back to the 0.1 logging precision
            off_grid = abs(x[i] / grid - round(x[i] / grid)) > 0.05
            for k, f in FRACS.items():
                if abs(r - f) < 0.015 and abs(x[i] - ref) > tol and off_grid:
                    frac_k = float(k)
        out["frac_k"] = frac_k
    if state == "load_start" and signal in ("current", "min_cell_volt", "max_cell_volt", "soc"):
        # real load onset: only the all-zero first row is attributable to logging
        cls = "zero_fill" if zero0 else ("load_onset" if idx > 1 else "stable")
    elif zero0 and np.isfinite(frac_k):
        cls = "bin_avg_zero"
    elif zero0:
        cls = "zero_fill"
    elif np.isfinite(frac_k):
        cls = "bin_avg_zero"
    elif idx > 1:
        seg = np.abs(x[:idx] - ref)
        ok = np.isfinite(seg)
        if ok.sum() >= 4:
            rho = pd.Series(seg[ok]).corr(pd.Series(np.arange(ok.sum())), method="spearman")
            cls = "monotone_settling" if rho is not None and rho < -0.5 else "load_onset" if state == "load_start" else "irregular_start"
        else:
            cls = "irregular_start"
    else:
        cls = "stable"
    out["cls"] = cls
    out["not_settled_in_window"] = bool(idx > 0 and t[min(idx, len(t) - 1)] >= w1 - 0.05)
    out["extended_settling"] = bool(idx > 2)  # settling beyond the first couple of samples, whatever the primary label
    return out


def init_records(fid: int, t: np.ndarray, cols: dict[str, np.ndarray], cfg: dict) -> list[dict]:
    """cols: canonical column -> array. Returns one record per (signal) present."""
    cq = cfg["quality"]
    cur = cols.get("bat_1_current", np.full_like(t, np.nan))
    state = start_state(t, cur, cfg["segmentation"]["active_current_a"])
    recs = []
    for name, x in cols.items():
        if name == "time_min" or not np.isfinite(x).any():
            continue
        r = settle_info(t, x, cq, state, name)
        r.update(flight_id=fid, start_state=state)
        recs.append(r)
    return recs


GROUP_BITS = {"volt": 1, "soc": 2, "soh": 4, "current": 8, "temp": 16}


def group_of(col: str) -> str | None:
    if "volt" in col:
        return "volt"
    if col.endswith("_soc"):
        return "soc"
    if col.endswith("_soh"):
        return "soh"
    if col.endswith("current"):
        return "current"
    if "temp" in col:
        return "temp"
    return None
