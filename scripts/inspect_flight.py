#!/usr/bin/env python
"""Summarise every column of every CSV inside one (or more) downloaded session ZIPs.

Usage:  python scripts/inspect_flight.py 4574 [4575 ...]
Writes data/raw/debug/inspect_<flight>_<member-kind>.csv and prints a short overview.
Output contains per-column statistics only (no raw rows). Used to pin units, sampling rates
and signal availability before analysis.
"""
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eplanes.build.raw import member_kind  # noqa: E402


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        num = pd.to_numeric(s, errors="coerce")
        frac_num = float(num.notna().sum() / max(s.notna().sum(), 1))
        r = dict(column=c, non_null_frac=round(float(s.notna().mean()), 4), numeric_frac=round(frac_num, 4),
                 n_unique=int(s.nunique(dropna=True)))
        if frac_num > 0.99:
            v = num.dropna()
            nz = v[v != 0]
            r.update(first=v.iloc[0] if len(v) else np.nan, min=v.min(), p05=v.quantile(.05), median=v.median(),
                     p95=v.quantile(.95), max=v.max(), frac_zero=round(float((v == 0).mean()), 4),
                     median_nonzero=nz.median() if len(nz) else np.nan,
                     median_abs_step=float(np.median(np.abs(np.diff(v.values)))) if len(v) > 2 else np.nan)
        else:
            vals = s.dropna().astype(str)
            r.update(first=vals.iloc[0] if len(vals) else "", example_last=vals.iloc[-1] if len(vals) else "")
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    man = pd.read_csv(ROOT / "data" / "raw" / "manifest.csv")
    out = ROOT / "data" / "raw" / "debug"
    out.mkdir(parents=True, exist_ok=True)
    for fid in map(int, sys.argv[1:]):
        row = man[man.flight_id == fid]
        if row.empty or row.status.iloc[0] != "ok":
            print(f"flight {fid}: not downloaded")
            continue
        with zipfile.ZipFile(row.zip_path.iloc[0]) as z:
            for name in z.namelist():
                kind = member_kind(name)
                with z.open(name) as f:
                    df = pd.read_csv(f, low_memory=False)
                df.columns = [c.strip() for c in df.columns]
                sm = summarise(df)
                p = out / f"inspect_{fid}_{kind}.csv"
                sm.to_csv(p, index=False)
                tcol = "time(ms)" if "time(ms)" in df else ("time(min)" if "time(min)" in df else None)
                dt = ""
                if tcol and len(df) > 2:
                    d = np.diff(pd.to_numeric(df[tcol], errors="coerce").dropna().values)
                    dt = f"  {tcol}: median step {np.median(d):g}, p99 {np.percentile(d, 99):g}, max {d.max():g}, span {np.nansum(d):g}"
                print(f"flight {fid} {kind:6s} {len(df):>7,} rows x {df.shape[1]:>4} cols -> {p.name}{dt}")


if __name__ == "__main__":
    main()
