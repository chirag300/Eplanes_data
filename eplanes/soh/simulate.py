"""Truth-known physics simulator: the test bed for the SOH label.

The real fleet has no capacity reference test, so a method cannot be proven on it. This simulator generates sessions
from an equivalent-circuit pack model whose TRUE capacity is known, driven by a BMS that behaves like the one described
in the dataset paper: reported SOH is revised in discrete steps during charges, has large episodes unrelated to capacity,
and reported SOC is scaled by the BMS's own capacity estimate (so SOC-window capacity inherits the BMS errors).

Results on this data say the method works in principle. They say nothing about the real fleet.

Files are written in the formats the real pipeline produces (analysis_ready_timeseries.parquet, flights_with_qc.csv,
bms_session.csv), plus sim_truth.csv. Every bit of pack physics lives here and nowhere in the estimators.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.interpolate import PchipInterpolator
from scipy.signal import lfilter

DT = 1.0                       # s
N_SERIES = 96
C_NOM = 28.5                   # Ah between 0 and 100 % of the BMS scale
THETA = np.array([0, .03, .06, .1, .15, .2, .3, .4, .5, .6, .7, .8, .9, .95, 1.0])
OCV_V = np.array([3.30, 3.40, 3.47, 3.53, 3.57, 3.60, 3.64, 3.68, 3.74, 3.82, 3.90, 3.98, 4.07, 4.12, 4.175])
_GRID = np.linspace(-0.2, 1.2, 2801)
_OCV = PchipInterpolator(THETA, OCV_V, extrapolate=True)(np.clip(_GRID, 0, 1)) + np.where(_GRID < 0, 2.0 * _GRID, 0) + \
    np.where(_GRID > 1, 0.8 * (_GRID - 1), 0)
N_REP = 8                      # representative cells simulated per pack
AIRCRAFT = {"C-GAUW": ("A", "2023-05-16", "2026-09-10", [("2024-07-01", "2025-01-27")]),
            "C-GMUW": ("A", "2025-06-03", "2026-09-12", [("2025-07-05", "2026-05-20")]),
            "VSW1280102": ("B", "2024-03-11", "2026-10-16", []),
            "VSW1280107": ("B", "2025-03-13", "2025-03-14", [])}


def ocv(theta):
    return np.interp(theta, _GRID, _OCV)


def arrhenius(t_c, e_k=2800.0):
    return np.exp(e_k * (1.0 / (np.asarray(t_c, float) + 273.15) - 1.0 / 298.15))


class Pack:
    def __init__(self, rng, label, b, t0):
        self.label, self.b, self.t0 = label, b, t0
        self.c0 = C_NOM * (1 + rng.normal(0, 0.012))
        self.a = rng.normal(0.040, 0.006)        # per sqrt(year), calendar
        self.bc = rng.normal(0.012, 0.002)       # per 1000 Ah discharged, cycling
        self.r25 = rng.normal(2.5, 0.12) * 1e-3  # V per pack A, per cell
        self.delta = np.sort(rng.normal(0, 0.012, N_REP))
        self.delta[0] -= 0.012                   # one weak cell
        self.dr = rng.normal(0, 0.04, N_REP)
        self.ah = 0.0
        self.jump = 0.0                          # real capacity events
        self.p = np.zeros(2)                     # polarisation carried between sessions (V)
        self.q_out = 0.01 * self.c0              # Ah below full
        self.soh_bms = 100.0
        self.bias = rng.normal(0, 0.03)          # current sensor offset, A

    def x(self, t):
        age_y = max((t - self.t0).total_seconds() / 86400 / 365.25, 0)
        return float(np.clip(1 - self.a * np.sqrt(age_y) - self.bc * self.ah / 1000 + self.jump, 0.4, 1.05))

    def resist(self, x):
        return self.r25 * (1 + 2.0 * (1 - x))     # resistance grows as capacity fades


def respond(pack: Pack, cur, temp, x, slip: float = 0.0, shape: float = 0.0):
    """Terminal voltages for a current trace (A, + = discharge). Updates nothing."""
    n = len(cur)
    q = pack.q_out + np.cumsum(cur) * DT / 3600.0
    caps = pack.c0 * x * (1 + pack.delta * (1 + 2 * (1 - x)))
    th = 1.0 - q[:, None] / caps[None, :]
    r0 = pack.resist(x) * arrhenius(temp)
    r1, tau1, r2, tau2 = 0.35 * pack.resist(x), 60.0, 0.25 * pack.resist(x), 1200.0
    a1, a2 = np.exp(-DT / tau1), np.exp(-DT / tau2)
    p1, _ = lfilter([r1 * (1 - a1)], [1, -a1], cur, zi=[pack.p[0] * a1])
    p2, _ = lfilter([r2 * (1 - a2)], [1, -a2], cur, zi=[pack.p[1] * a2])
    v = ocv(th + slip * (1 - x)) + shape * (1 - x) * (np.clip(th, 0, 1) - 0.6) - cur[:, None] * r0[:, None] * (1 + pack.dr[None, :]) - (p1 + p2)[:, None]
    return v.mean(1), v.min(1), v.max(1), q, p1, p2, n


def flight_profile(rng, target_ah):
    def hold(level, s, noise=0.5):
        return level + rng.normal(0, noise, int(s))

    seg = [hold(0.0, rng.integers(8, 25), 0.05), hold(rng.uniform(2.5, 4.5), rng.integers(240, 500), 0.2)]
    peak = rng.uniform(92, 104)
    seg += [np.linspace(4, peak, int(rng.integers(6, 12))), hold(peak, rng.integers(50, 90), 1.5),
            np.linspace(peak, 65, 10), hold(rng.uniform(60, 70), rng.integers(180, 300), 1.0)]
    used = sum(np.sum(s) for s in seg) / 3600
    tail = [hold(4.0, rng.integers(120, 240), 0.5), hold(3.0, rng.integers(60, 180), 0.2), hold(0.0, rng.integers(15, 60), 0.05)]
    used += sum(np.sum(s) for s in tail) / 3600
    while used < target_ah:
        lvl, s = (rng.uniform(26, 40), rng.integers(180, 500)) if rng.random() < 0.7 else (rng.uniform(55, 70), rng.integers(40, 90))
        s_ = hold(lvl, s, 0.8)
        seg.append(np.linspace(seg[-1][-1], s_[0], 8))
        seg.append(s_)
        used += (np.sum(s_) + 8 * 0.5 * (seg[-2][0] + s_[0])) / 3600
    return np.concatenate(seg + tail)


def charge_profile(rng, pack: Pack, temp, x, slip: float = 0.0, shape: float = 0.0):
    """Slow ramp 27 -> 23 A, step down when the top cell reaches a voltage limit, then a taper (as in the real logs)."""
    hz = int(3 * 3600)
    head = np.zeros(int(rng.integers(5, 25)))
    qs = np.arange(hz) * DT
    k = 4.0 / 14.0 / 3600.0
    i = 27.0 * np.exp(-k * qs)  # dI/dt = -k I gives I falling with charge in
    cur = -np.r_[head, i]
    tt = np.full(len(cur), temp)
    _, _, vmax, *_ = respond(pack, cur, tt, x, slip, shape)
    limit = 4.10 + rng.normal(0, 0.002)
    hit = np.flatnonzero(vmax >= limit)
    a = int(hit[0]) if len(hit) else len(cur) - 1
    ramp = cur[:a + 1]
    lo = -rng.uniform(9.2, 9.8)
    step = np.full(hz, lo)
    c2 = np.r_[ramp, step]
    _, _, vmax2, *_ = respond(pack, c2, np.full(len(c2), temp), x, slip, shape)
    hit2 = np.flatnonzero((vmax2 >= 4.17) & (np.arange(len(c2)) > a))
    b = int(hit2[0]) if len(hit2) else a + 1800
    flat = c2[a + 1:b + 1]
    taper = lo * np.exp(-np.arange(int(rng.integers(300, 900))) / 300.0)
    tail = np.zeros(int(rng.integers(15, 45)))
    return np.r_[ramp, flat, taper, tail]


def bms_after_session(rng, pack: Pack, kind, soc_start, when, events):
    """The BMS's reported SOH: small revisions on low-SOC charges, overstated fade, plus large episodes unrelated to capacity."""
    old = pack.soh_bms
    if kind == "charging" and soc_start < 55 and rng.random() < 0.5:
        pack.soh_bms += rng.normal(-0.35, 1.4)
    for d, kind_ev, val in events.get((pack.label, pack.b), []):
        if pack._ev_done.get(d) is None and when >= d:
            pack._ev_done[d] = True
            pack.soh_bms = {"reset": 100.0, "up": pack.soh_bms + val, "down": pack.soh_bms - val}[kind_ev]
    if kind == "charging" and pack.soh_bms > 85 and old == 100.0:  # after a reset the estimator walks back down
        pack.soh_bms -= rng.uniform(1, 3)
    pack.soh_bms = float(np.clip(pack.soh_bms, 35, 100))


def make_sim(data_dir, scale: float = 0.45, seed: int = 11, log=print, ocv_slip: float = 0.0, ocv_shape: float = 0.0) -> Path:
    """Stress tests of the fixed-curve assumption: ocv_slip shifts the open-circuit curve with age, ocv_shape stretches it."""
    rng = np.random.default_rng(seed)
    proc = Path(data_dir) / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    (Path(data_dir) / "SYNTHETIC").write_text("Simulated data from eplanes.soh.simulate, not real flights.\n")
    bms_ev = {("C-GAUW", 1): [(pd.Timestamp("2024-05-02"), "reset", 0), (pd.Timestamp("2026-05-01"), "up", 22)],
              ("C-GAUW", 2): [(pd.Timestamp("2024-05-02"), "reset", 0), (pd.Timestamp("2026-05-01"), "up", 20)],
              ("VSW1280102", 1): [(pd.Timestamp("2025-08-10"), "up", 30), (pd.Timestamp("2025-10-01"), "down", 25),
                                  (pd.Timestamp("2026-02-01"), "up", 28)],
              ("VSW1280102", 2): [(pd.Timestamp("2025-08-10"), "up", 30), (pd.Timestamp("2025-10-01"), "down", 24),
                                  (pd.Timestamp("2026-02-01"), "up", 27)],
              ("C-GMUW", 1): [(pd.Timestamp("2026-05-22"), "down", 20), (pd.Timestamp("2026-06-05"), "up", 22)],
              ("C-GMUW", 2): [(pd.Timestamp("2026-05-22"), "down", 20), (pd.Timestamp("2026-06-05"), "up", 21)]}
    real_ev = {("C-GAUW", 2): (pd.Timestamp("2025-06-15"), 0.035)}   # one real capacity gain, e.g. a replaced module
    writer, schema = None, None
    flights, truth, bms_rows = [], [], []
    fid = 4000
    for label, (fleet, a, b, idle) in AIRCRAFT.items():
        start, end = pd.Timestamp(a) + pd.Timedelta(hours=14), pd.Timestamp(b) + pd.Timedelta(hours=14)
        idle = [(pd.Timestamp(x), pd.Timestamp(y)) for x, y in idle]
        packs = {k: Pack(rng, label, k, start) for k in (1, 2)}
        for p in packs.values():
            p._ev_done = {}
        t, session_n = start, 0
        while t < end:
            if any(x <= t < y for x, y in idle):
                t = max(y for x, y in idle if x <= t < y) + pd.Timedelta(hours=14)
                continue
            cycles = int(rng.integers(1, 4))
            for _ in range(cycles):
                dod = float(np.clip(rng.normal(0.62 if fleet == "B" else 0.54, 0.09), 0.15, 0.78))
                x_now = packs[1].x(t)
                seq = [("flight test", dod), ("charging", None)]
                if rng.random() < 0.06:
                    seq.insert(1, ("ground test", 0.04))
                for kind, d in seq:
                    temp0 = 18 + 9 * np.sin(2 * np.pi * (t.day_of_year - 110) / 365.25) + rng.normal(0, 2.0) + (6 if kind == "charging" else 4)
                    cur0 = None
                    if kind == "flight test":
                        cur0 = flight_profile(rng, d * packs[1].c0 * x_now)
                    elif kind == "ground test":
                        cur0 = np.r_[np.zeros(20), rng.normal(30, 2, 180), np.zeros(30)]
                    else:
                        cur0 = charge_profile(rng, packs[1], temp0, x_now, ocv_slip, ocv_shape)
                    n = len(cur0)
                    logged = rng.random() > (0.04 if kind == "charging" else 0.03)
                    tm = np.arange(n) * DT / 60.0
                    rows = {"time_min": tm, "time_s": tm * 60.0}
                    for k, p in packs.items():
                        if (label, k) in real_ev and t >= real_ev[(label, k)][0] and p.jump == 0:
                            p.jump = real_ev[(label, k)][1]
                        x = p.x(t)
                        share = 1 + (0.002 if k == 1 else -0.002)
                        cur = cur0 * share + rng.normal(0, 0.05, n)
                        temp = temp0 + (k - 1) * 0.4 + np.cumsum(cur ** 2) * DT * 2.0e-6
                        vm, vn, vx, q, p1, p2, _ = respond(p, cur, temp, x, ocv_slip, ocv_shape)
                        soc0 = float(np.clip(100 * (1 - p.q_out / (C_NOM * p.soh_bms / 100)), 0, 100))
                        soc = np.clip(soc0 - 100 * np.cumsum(cur) * DT / 3600 / (C_NOM * p.soh_bms / 100), 0, 100)
                        if kind == "charging" and soc0 < 85:
                            soc = np.minimum(soc + np.where(vx > 4.15, 1.5, 0), 100)
                        soh_s = np.full(n, round(p.soh_bms))
                        old = p.soh_bms
                        bms_after_session(rng, p, kind, soc0, t, bms_ev)
                        if p.soh_bms != old and kind == "charging":
                            hit = np.flatnonzero(soc >= rng.uniform(69, 89))
                            soh_s[hit[0] if len(hit) else n // 2:] = round(p.soh_bms)
                        else:
                            soh_s[:] = round(p.soh_bms)
                        i_meas = cur + p.bias
                        rows.update({f"bat_{k}_current": np.round(i_meas, 1), f"bat_{k}_voltage": np.round(vm * N_SERIES, 1),
                                     f"bat_{k}_min_cell_volt": np.round((vn - 0.003) * 1000), f"bat_{k}_max_cell_volt": np.round((vx + 0.003) * 1000),
                                     f"bat_{k}_soc": np.round(soc), f"bat_{k}_soh": soh_s.astype(float),
                                     f"bat_{k}_max_cell_temp": np.round(temp + 1), f"bat_{k}_min_cell_temp": np.round(temp - 1),
                                     f"bat_{k}_avg_cell_temp": np.round(temp)})
                        if logged:
                            truth.append(dict(flight_id=fid + 1, aircraft_label=label, battery=k, t=t, x_true=x, cap_true_ah=p.c0 * x,
                                              weak_true=x * (1 + p.delta[0] * (1 + 2 * (1 - x))) / (1 + p.delta[0]),
                                              r25_true_mohm=p.resist(x) * 1000, soh_bms=p.soh_bms))
                            s_min = p.soh_bms + rng.normal(0.3, 0.2)
                            bms_rows.append(dict(flight_id=fid + 1, aircraft_label=label, battery=k, soh_cell_cells_last_min=min(s_min, 100),
                                                 soh_cell_cells_last_mean=min(p.soh_bms + 5.6, 100), soh_cell_cells_last_max=min(p.soh_bms + 10.5, 100),
                                                 cell_est_bat_cap_last=p.soh_bms * C_NOM / 100, uniform_cells=p.soh_bms == 100.0))
                        p.q_out = float(q[-1])
                        p.p = np.array([p1[-1], p2[-1]])
                        if kind != "charging":
                            p.ah += float(np.sum(np.clip(cur, 0, None)) * DT / 3600)
                    dur = float(tm[-1])
                    if logged:
                        fid += 1
                        session_n += 1
                        d_ = pd.DataFrame(rows)
                        d_.insert(0, "flight_id", fid)
                        d_["aircraft_label"] = label
                        d_["quality_flag"] = np.zeros(n, np.int32)
                        d_["init_unsettled_groups"] = np.zeros(n, np.int32)
                        d_["segment_type"] = "rest"
                        d_["avionics_valid"] = kind != "charging"
                        d_["oat"] = (temp0 - 6.0) if kind != "charging" else 0.0
                        scale_bad = label == "C-GAUW" and 12 <= session_n <= 15
                        if scale_bad:
                            for k in (1, 2):
                                d_[f"bat_{k}_current"] *= 12.0
                        signals = [c for c in d_.columns if c.startswith("bat_")]
                        d_.loc[0, signals] = 0.0               # all-zero first row
                        d_.loc[0, "quality_flag"] = 1
                        hot = int(rng.integers(0, 15))
                        tcols = [c for c in signals if "temp" in c]
                        d_.loc[:hot, tcols] = 0.0              # temperatures come alive later
                        if rng.random() < 0.15 and n > 900:     # a logging gap
                            g0 = int(rng.integers(100, n - 400))
                            d_ = d_.drop(index=range(g0, g0 + int(rng.integers(3, 60)))).reset_index(drop=True)
                        tb = pa.Table.from_pandas(d_, preserve_index=False)
                        if writer is None:
                            schema = tb.schema
                            writer = pq.ParquetWriter(proc / "analysis_ready_timeseries.parquet", schema)
                        writer.write_table(tb.cast(schema))
                        flights.append(dict(flight_id=fid, aircraft_label=label, fleet=fleet, flight_type=kind, start_datetime=t.floor("min"),
                                            flight_date=t.normalize(), duration_min=dur, n_rows=len(d_), scale_anomaly=scale_bad,
                                            is_duplicate_content=False, cross_aircraft_duplicate=False))
                    t = t + pd.Timedelta(minutes=dur) + pd.Timedelta(minutes=float(rng.uniform(25, 150)))
            t = t.normalize() + pd.Timedelta(days=float(1 + rng.exponential(1.3 / scale)), hours=float(rng.uniform(13, 18)))
        log(f"[sim] {label}: {session_n} sessions")
    writer.close()
    pd.DataFrame(flights).to_csv(proc / "flights_with_qc.csv", index=False)
    pd.DataFrame(bms_rows).to_csv(proc / "bms_session.csv", index=False)
    tr = pd.DataFrame(truth)
    tr = tr.drop_duplicates(["flight_id", "battery"])
    tr.to_csv(proc / "sim_truth.csv", index=False)
    log(f"[sim] {len(flights)} sessions written to {proc}")
    return proc
