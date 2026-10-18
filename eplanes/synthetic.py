"""SYNTHETIC data generator used ONLY to test the pipeline end to end.

It mimics the file layout and schema seen in the project notebooks (14-column
CSV at 0.02 min, all-zero first row, bin-averaged temperature start-up, integer
SOC, negative charging current, BMS SOH with jumps) but contains NO real
measurements. Results computed from it mean nothing scientifically.
Output goes to a separate directory and is never mixed with real data.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAW_COLS = ["time(min)", "bat 1 min cell volt", "bat 1 max cell volt", "bat 2 min cell volt",
            "bat 2 max cell volt", "bat 1 soc", "bat 2 soc", "bat 1 soh", "bat 2 soh",
            "bat 1 current", "bat 2 current", "bat 1 max cell temp", "bat 2 max cell temp",
            "bat 1 avg cell temp"]
DT = 0.02
Q_AH = 22.0


def _ocv(soc):
    return 3350 + 8.5 * np.clip(soc, 0, 100)  # mV, crude


def _sim_flight(rng, kind, soc0, soh, temp0, dur_min):
    n = int(dur_min / DT) + 1
    t = np.arange(n) * DT
    I = np.zeros(n)
    if kind == "flight test":
        tk = t < 3
        cr = (t >= 3) & (t < dur_min - 6)
        ds = t >= dur_min - 6
        I[tk] = rng.normal(165, 10, tk.sum())
        I[cr] = rng.normal(70, 6, cr.sum())
        I[ds] = rng.normal(15, 4, ds.sum())
    elif kind == "charging":
        # smooth ramped charger (no clean CC/CV plateau)
        x = t / dur_min
        I = -(26 * np.sin(np.pi * np.clip(x, 0, 1) ** 0.7) + 2) + rng.normal(0, 0.4, n)
    else:  # ground test
        I[(t > 1) & (t < dur_min - 1)] = rng.normal(35, 5, int(((t > 1) & (t < dur_min - 1)).sum()))
    I2 = I * rng.normal(1.0, 0.02) + rng.normal(0, 0.5, n)
    I1 = I + rng.normal(0, 0.5, n)
    soc1 = soc0 - np.cumsum(I1) * DT / 60 / (Q_AH * soh / 100) * 100
    soc2 = soc0 + rng.normal(0, 1) - np.cumsum(I2) * DT / 60 / (Q_AH * soh / 100) * 100
    soc1, soc2 = np.clip(np.round(soc1), 0, 100), np.clip(np.round(soc2), 0, 100)
    base = lambda s, i, r: _ocv(s) - r * i  # noqa: E731
    v1 = base(soc1, I1, 1.8)
    v2 = base(soc2, I2, 1.9)
    T1 = temp0 + np.cumsum(np.abs(I1)) * DT / 60 * 0.012
    T2 = T1 + rng.normal(0.4, 0.1)
    df = pd.DataFrame({
        "time(min)": t,
        "bat 1 min cell volt": v1 - 5, "bat 1 max cell volt": v1 + 6,
        "bat 2 min cell volt": v2 - 5, "bat 2 max cell volt": v2 + 6,
        "bat 1 soc": soc1, "bat 2 soc": soc2,
        "bat 1 soh": np.nan, "bat 2 soh": np.nan,
        "bat 1 current": I1, "bat 2 current": I2,
        "bat 1 max cell temp": np.round(T1 * 3) / 3, "bat 2 max cell temp": np.round(T2 * 3) / 3,
        "bat 1 avg cell temp": np.round((T1 - 1) * 3) / 3,
    })
    return df


def _apply_init_artifacts(df, rng):
    """Row 0 all zero; first bin-averaged temps contaminated by a zero (x2/3)."""
    df = df.copy()
    for c in df.columns[1:]:
        df.loc[0, c] = 0.0
    for c in ("bat 1 max cell temp", "bat 2 max cell temp", "bat 1 avg cell temp"):
        df.loc[1, c] = df.loc[2, c] * 2 / 3
    if rng.random() < 0.3:  # some sessions: voltage settles slowly (physical-like)
        k = np.arange(1, 40)
        for c in ("bat 1 min cell volt", "bat 2 min cell volt"):
            df.loc[k, c] = df.loc[k, c] - 25 * np.exp(-k / 8)
    return df


def make_synthetic(out_dir: str | Path, flights_per_aircraft: int = 40, seed: int = 7,
                   aircraft: dict | None = None) -> Path:
    """Write data/raw/{zips,meta,manifest.csv} under out_dir."""
    rng = np.random.default_rng(seed)
    aircraft = aircraft or {"C-GAUW": ("fleet_A", 166), "C-GMUW": ("fleet_A", 192),
                            "VSW1280102": ("fleet_B", 191), "VSW1280107": ("fleet_B", 196)}
    raw = Path(out_dir) / "raw"
    rows, fid = [], 5000
    start = pd.Timestamp("2023-06-03 09:00")
    for ai, (label, (acct, aid)) in enumerate(aircraft.items()):
        soh1, soh2 = 97.0, 97.0
        soc = 60.0
        day = 0
        for k in range(flights_per_aircraft):
            fid += 1
            day += int(rng.integers(1, 12)) if rng.random() > 0.1 else int(rng.integers(60, 200))
            when = start + pd.Timedelta(days=day + ai * 3, hours=float(rng.integers(0, 8)))
            kind = rng.choice(["flight test", "charging", "ground test"], p=[0.4, 0.5, 0.1])
            if label == "VSW1280107" and kind == "charging":
                kind = "flight test"  # mimics 'no usable charging' noted in project memory
            temp0 = 8 + 14 * (1 + np.sin((when.dayofyear - 110) / 365 * 2 * np.pi)) / 2 + rng.normal(0, 1.5)
            dur = {"flight test": rng.uniform(25, 50), "charging": rng.uniform(40, 90),
                   "ground test": rng.uniform(8, 20)}[kind]
            if kind == "charging":
                soc = float(rng.uniform(15, 40))
            elif kind == "flight test":
                soc = float(rng.uniform(80, 98))
            df = _sim_flight(rng, kind, soc, (soh1 + soh2) / 2, temp0, dur)
            soh1 = max(40, soh1 - abs(rng.normal(0.15, 0.4)) + (6 if rng.random() < 0.02 else 0))
            soh2 = max(40, soh2 - abs(rng.normal(0.12, 0.3)))
            df["bat 1 soh"] = round(soh1)
            df["bat 2 soh"] = round(soh2)
            df = _apply_init_artifacts(df, rng)
            if rng.random() < 0.04:  # sensor dropout
                a = int(rng.integers(100, max(101, len(df) - 200)))
                df.loc[a:a + 30, "bat 1 current"] = np.nan
            if rng.random() < 0.03:  # logging gap
                df = df.drop(index=range(300, 360)).reset_index(drop=True)
            soc = float(df["bat 1 soc"].iloc[-1])
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr(f"csv-{fid}.csv", df.to_csv(index=False))
                z.writestr(f"csv-{fid}_warns.csv", "time(min),warning\n")
            data = buf.getvalue()
            zp = raw / "zips" / label / f"flight_{fid}.zip"
            zp.parent.mkdir(parents=True, exist_ok=True)
            zp.write_bytes(data)
            mp = raw / "meta" / label / f"flight_{fid}.json"
            mp.parent.mkdir(parents=True, exist_ok=True)
            info = {"Date": when.strftime("%B %d, %Y, %I:%M %p").replace(" 0", " "),
                    "Flight type": kind.capitalize(), "Registration": label if label.startswith("C-") else "",
                    "Serial number": "" if label.startswith("C-") else label}
            mp.write_text(json.dumps({"flight_id": fid, "account": acct, "aircraft_id": aid,
                                      "aircraft_label": label, "page_info": info,
                                      "download_url": "synthetic"}))
            rows.append(dict(flight_id=fid, account=acct, aircraft_id=aid, aircraft_label=label,
                             status="ok", http_note="synthetic", zip_path=str(zp), meta_path=str(mp),
                             sha256=hashlib.sha256(data).hexdigest(), zip_bytes=len(data), n_members=2,
                             downloaded_utc="synthetic"))
    pd.DataFrame(rows).to_csv(raw / "manifest.csv", index=False)
    return raw
