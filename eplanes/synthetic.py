"""SYNTHETIC data generator used ONLY to test the pipeline end to end.

Mimics the layout of the real portal export (inspected Oct 2026): a main CSV (time(ms), time(min), pack and
cell voltages, temperatures, avionics block, drivetrain) and per-battery BMS files `_1`/`_2` with the BMS
estimator state (Kalman/coulomb SOC, 96 per-cell SOH / BATT CAP / FSK values, capacity estimate, reset flag).
Planted behaviours: all-zero first row in the main log; temperatures-only zero first row in the BMS files;
avionics block all-zero when the avionics are off (charging); occasional 65535 sentinels; occasional BMS
"reset" sessions where per-cell values become uniform defaults and reported SOH jumps to 100.
It contains NO real measurements; results computed from it mean nothing scientifically.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DT_S = 0.5          # synthetic main-log step (real: 0.1 s); BMS files at 2x the step
Q_AH = 36.0
N_CELLS = 96


def _ocv(soc):
    return 3350 + 8.5 * np.clip(soc, 0, 100)  # mV per cell, crude


def _sim(rng, kind, soc0, soh, temp0, oat, dur_min):
    n = int(dur_min * 60 / DT_S) + 1
    t = np.arange(n) * DT_S / 60.0
    I = np.zeros(n)
    if kind == "flight test":
        tk, cr, ds = t < 3, (t >= 3) & (t < dur_min - 6), t >= dur_min - 6
        I[tk] = rng.normal(165, 10, tk.sum())
        I[cr] = rng.normal(70, 6, cr.sum())
        I[ds] = rng.normal(15, 4, ds.sum())
    elif kind == "charging":
        x = t / dur_min
        I = -(26 * np.sin(np.pi * np.clip(x, 0, 1) ** 0.7) + 2) + rng.normal(0, 0.4, n)
    else:
        m = (t > 1) & (t < dur_min - 1)
        I[m] = rng.normal(35, 5, m.sum())
    out = {}
    for b, gain in ((1, 1.0), (2, rng.normal(1.0, 0.02))):
        Ib = np.round(I * gain + rng.normal(0, 0.5, n), 1)
        soc = np.clip(soc0 + (b - 1) * rng.normal(0, 1) - np.cumsum(Ib) * DT_S / 3600 / (Q_AH * soh / 100) * 100, 0, 100)
        vcell = _ocv(soc) - 1.8 * Ib / 12
        T = temp0 + np.cumsum(np.abs(Ib)) * DT_S / 3600 * 0.6
        out[b] = dict(I=Ib, soc=soc, vcell=vcell, T=T)
    airborne = (kind == "flight test") & (t > 2) & (t < dur_min - 2)
    alt = 290 + np.where(airborne, np.minimum((t - 2) * 150, 600), 0) + rng.normal(0, 0.5, n)
    return t, out, alt, airborne


def make_synthetic(out_dir: str | Path, flights_per_aircraft: int = 40, seed: int = 7,
                   aircraft: dict | None = None) -> Path:
    rng = np.random.default_rng(seed)
    aircraft = aircraft or {"C-GAUW": ("fleet_A", 166), "C-GMUW": ("fleet_A", 192),
                            "VSW1280102": ("fleet_B", 191), "VSW1280107": ("fleet_B", 196)}
    raw = Path(out_dir) / "raw"
    rows, fid = [], 5000
    start = pd.Timestamp("2023-06-03 09:00")
    for ai, (label, (acct, aid)) in enumerate(aircraft.items()):
        cell_soh = {b: 100 - np.abs(rng.normal(0, 0.5, N_CELLS)) for b in (1, 2)}
        soc, day = 60.0, 0
        for k in range(flights_per_aircraft):
            fid += 1
            day += int(rng.integers(1, 12)) if rng.random() > 0.1 else int(rng.integers(60, 200))
            when = start + pd.Timedelta(days=day + ai * 3, hours=float(rng.integers(0, 8)))
            kind = rng.choice(["flight test", "charging", "ground test"], p=[0.4, 0.5, 0.1])
            if label == "VSW1280107" and kind == "charging":
                kind = "flight test"
            oat = 8 + 14 * (1 + np.sin((when.dayofyear - 110) / 365 * 2 * np.pi)) / 2 + rng.normal(0, 1.5)
            dur = {"flight test": rng.uniform(25, 50), "charging": rng.uniform(40, 90), "ground test": rng.uniform(8, 20)}[kind]
            soc = float(rng.uniform(15, 40)) if kind == "charging" else float(rng.uniform(80, 98)) if kind == "flight test" else soc
            reset = rng.random() < 0.06
            for b in (1, 2):
                cell_soh[b] = np.clip(cell_soh[b] - np.abs(rng.normal(0.3, 0.1, N_CELLS)), 50, 100)
            soh_rep = {b: (100.0 if reset else float(np.round(np.mean(cell_soh[b])))) for b in (1, 2)}
            t, sim, alt, air = _sim(rng, kind, soc, np.mean(cell_soh[1]), oat + 2, oat, dur)
            n = len(t)
            avion = kind != "charging"
            main = {"time(ms)": np.round(t * 60000 + 2300 + rng.normal(0, 3, n), 3), "time(min)": t}
            for b in (1, 2):
                s_ = sim[b]
                main[f"bat {b} current"] = s_["I"]
                main[f"bat {b} voltage"] = np.round(s_["vcell"] * N_CELLS / 1000, 1)
            for b in (1, 2):
                main[f"bat {b} soc"] = np.round(sim[b]["soc"])
            for b in (1, 2):
                main[f"bat {b} soh"] = soh_rep[b]
            for b in (1, 2):
                main[f"bat {b} min cell temp"] = np.round(sim[b]["T"] - 1)
                main[f"bat {b} max cell temp"] = np.round(sim[b]["T"])
                main[f"bat {b} avg cell temp"] = np.round(sim[b]["T"] - 0.5)
            for b in (1, 2):
                main[f"bat {b} min cell volt"] = np.round(sim[b]["vcell"] - 6, 1)
                main[f"bat {b} max cell volt"] = np.round(sim[b]["vcell"] + 6, 1)
            P = (sim[1]["I"] * main["bat 1 voltage"] + sim[2]["I"] * main["bat 2 voltage"]) / 1000
            main["motor power"] = np.round(np.clip(P, 0, None))
            main["motor rpm"] = np.where(rng.random(n) < 0.001, 65535, np.round(np.clip(P, 0, None) * 50))
            main["IAS"] = np.where(air, 45 + rng.normal(0, 2, n), 0) if avion else 0.0
            main["PRESSURE_ALT"] = alt if avion else 0.0
            main["OAT"] = np.round(oat) if avion else 0.0
            main["ACC_NORM"] = (1.0 + rng.normal(0, 0.003, n)) if avion else 0.0
            sec = (when.hour + 4) * 3600 + when.minute * 60 + np.floor(t * 60)  # UTC = local + 4 h (EDT)
            main["TIMESTAMP"] = ((sec // 3600) % 24 * 10000 + (sec % 3600) // 60 * 100 + sec % 60).astype(int) if avion else 0
            df = pd.DataFrame(main)
            df.iloc[0, 2:] = 0.0  # all-zero first row (time columns aside)
            if rng.random() < 0.03:
                df = df.drop(index=range(300, 360)).reset_index(drop=True)
            soc = float(df["bat 1 soc"].iloc[-1])
            subs = {}
            for b in (1, 2):
                idx = np.arange(0, n, 2)
                sb = {"time(ms)": df["time(ms)"].reindex(idx).ffill().to_numpy() + 150,
                      f"bat {b} current": sim[b]["I"][idx], f"bat {b} voltage": main[f"bat {b} voltage"][idx],
                      f"bat {b} soc": np.round(sim[b]["soc"][idx]), f"bat {b} soh": soh_rep[b],
                      f"bat {b} min cell temp": np.round(sim[b]["T"][idx] - 1), f"bat {b} max cell temp": np.round(sim[b]["T"][idx]),
                      f"bat {b} kalman soc": np.round(sim[b]["soc"][idx] + rng.normal(0, 0.8), 2),
                      f"bat {b} coulomb soc out": np.round(sim[b]["soc"][idx] + rng.normal(0, 1.2), 2),
                      f"bat {b} cell est bat cap": 9611 if reset else int(np.mean(cell_soh[b]) * 96),
                      f"bat {b} cell flg rst coulomb": 1 if reset else int(rng.random() < 0.05)}
                for c in range(N_CELLS):
                    sb[f"B{b} SOH CELL {c}"] = 100.0 if reset else round(float(cell_soh[b][c]), 2)
                for c in range(N_CELLS):
                    sb[f"B{b} BATT CAP {c}"] = 96.11 if reset else round(float(cell_soh[b][c]) * 0.97, 2)
                sdf = pd.DataFrame(sb)
                for c in (f"bat {b} min cell temp", f"bat {b} max cell temp"):
                    sdf.loc[0, c] = 0.0  # temperatures only start at zero in the BMS files
                for c in [c for c in sdf.columns if c.startswith(f"B{b} ")]:
                    sdf.loc[0, c] = 0.0
                subs[b] = sdf
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr(f"csv-{fid}.csv", df.to_csv(index=False))
                z.writestr(f"csv-{fid}_1.csv", subs[1].to_csv(index=False))
                z.writestr(f"csv-{fid}_2.csv", subs[2].to_csv(index=False))
                z.writestr(f"csv-{fid}_warns.csv", "time(min),SOC<30%,STALL WARNING\n0,0,0\n")
            data = buf.getvalue()
            zp = raw / "zips" / label / f"flight_{fid}.zip"
            zp.parent.mkdir(parents=True, exist_ok=True)
            zp.write_bytes(data)
            mp = raw / "meta" / label / f"flight_{fid}.json"
            mp.parent.mkdir(parents=True, exist_ok=True)
            info = {"Date": when.strftime("%B %d, %Y, %I:%M %p").replace(" 0", " "),
                    "Flight type": kind.capitalize(), "Registration": label if label.startswith("C-") else "",
                    "Serial number": "" if label.startswith("C-") else label}
            mp.write_text(json.dumps({"flight_id": fid, "account": acct, "aircraft_id": aid, "aircraft_label": label,
                                      "page_info": info, "download_url": "synthetic"}))
            rows.append(dict(flight_id=fid, account=acct, aircraft_id=aid, aircraft_label=label, status="ok",
                             http_note="synthetic", zip_path=str(zp), meta_path=str(mp),
                             sha256=hashlib.sha256(data).hexdigest(), zip_bytes=len(data), n_members=4,
                             downloaded_utc="synthetic", synthetic_reset=reset))
    pd.DataFrame(rows).to_csv(raw / "manifest.csv", index=False)
    return raw
