"""Raw-curve nowcast: can a 1-D CNN read capacity straight off one session's voltage-charge curve?

Two input views, per session:
  flight   dQ/dv of the discharge, binned on voltage
  charge   dQ/dv of the charge, binned on voltage
Flight curves predicting a label dominated by charge and rest measurements is the useful case: a health reading from a flight,
with no charge needed. The target is the leave-one-out label, so a session's own reading is never in its own target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .capacity import charge_samples, discharge_samples, load_curves

EDGES = np.arange(3.45, 4.2001, 0.0125)


def curve_row(ctx, r, mode):
    rc = ctx.cfg["registration"]
    c = load_curves(ctx, int(r["flight_id"]))
    b = str(int(r["battery"]))
    if f"{b}_t" not in c:
        return None
    got = charge_samples(c, b, r, r["r_eff"], rc) if mode == "charge" else discharge_samples(c, b, r["r_eff"], rc)
    if got is None or len(got[0]) < 8:
        return None
    v, q = got
    i = np.argsort(v)
    v, q = v[i], q[i]
    grid = np.interp(EDGES, v, q, left=np.nan, right=np.nan)
    d = np.diff(grid) / np.diff(EDGES) / 100.0
    mask = np.isfinite(d).astype(float)
    return np.stack([np.nan_to_num(d), mask])


def curve_set(ctx, table, mode):
    want = table[(table.charge_min >= 5) if mode == "charge" else (table.discharge_min >= 3)]
    want = want[want.usable_electrical]
    X, meta = [], []
    for r in want.to_dict("records"):
        x = curve_row(ctx, r, mode)
        if x is not None and x[1].sum() >= 6:
            X.append(x)
            meta.append(r)
    if not X:
        return np.empty((0, 2, len(EDGES) - 1)), want.iloc[0:0]
    return np.stack(X), pd.DataFrame(meta)


def curve_cv(ctx, table, keys, cuts, engine, log, seeds=None) -> pd.DataFrame:
    from .evaluate import score_now
    from .models.neural import CurveNow
    t = table[[(a, b) in set(keys) for a, b in zip(table.aircraft_label, table.battery)]]
    out = []
    for mode in ("flight", "charge"):
        X, meta = curve_set(ctx, t, mode)
        if len(meta) < 40:
            log(f"[curves] {mode}: only {len(meta)} sessions, skipped")
            continue
        meta = meta.reset_index(drop=True)
        ps = []
        for a, b in cuts:
            lab = pd.concat([engine.upto(k, a)[["flight_id", "x_loo", "sd_loo"]].assign(battery=k[1]) for k in keys])
            m = meta.merge(lab, on=["flight_id", "battery"], how="left", suffixes=("", "_cut"))
            tr = m.index[(m.start_datetime <= a) & m.x_loo_cut.notna()].to_numpy() if "x_loo_cut" in m else np.array([])
            te = m.index[(m.start_datetime > a) & (m.start_datetime <= b)].to_numpy()
            if len(tr) < 30 or len(te) < 5:
                continue
            w = 1.0 / np.maximum(m.sd_loo_cut.to_numpy(float)[tr], 2e-3) ** 2
            net = CurveNow(ctx.cfg["models"], seeds).fit(X[tr], 100 * m.x_loo_cut.to_numpy(float)[tr], w)
            ps.append(meta.iloc[te][["aircraft_label", "battery", "flight_id", "soh_loo", "soh_phys"]].assign(pred=net.predict(X[te])))
        if ps:
            p = pd.concat(ps)
            out.append(score_now(p, f"curve_{mode}", "curve_cnn", X.shape[2]))
    r = pd.DataFrame(out)
    if len(r):
        log("[curves] CNN on raw session curves\n" + r.round(3).to_string(index=False))
    return r
