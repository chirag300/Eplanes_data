"""Per-session summaries of the BMS internal state (the `_1` / `_2` files inside each ZIP).

Read directly from the original ZIPs (only the needed columns), cached per session in
data/processed/bms_cache/flight_<id>.json, combined into data/processed/bms_session.csv.

Per battery and session:
  * reported SOC vs the BMS's own Kalman SOC and coulomb-counter SOC (start, end, median and max abs differences)
  * reported SOH and the 96 per-cell SOH / BATT CAP / FSK values (min, mean, max, std across cells)
  * `uniform_cells`: every cell reports the identical SOH and BATT CAP (std == 0) - the signature seen in
    session 4576 (all cells SOH 100, BATT CAP 96.11) that looks like an estimator reset to defaults
  * capacity estimates (cell est bat cap, cap est), reset/new-estimate flags, worst cell, available energy
  * which channels are exactly zero in the first sample (start-up evidence; main log vs BMS file)
Column meanings come from the header names only; the BMS internals are not documented publicly, so values are
reported as logged and interpreted cautiously.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Settings
from .raw import member_kind

SCALARS = ["soc", "soh", "kalman soc", "coulomb soc out", "cell est bat cap", "cell tmp bat cap", "cap est",
           "cell flg rst coulomb", "cell flg new est batt cap", "worst cell", "box avaliable energy",
           "cell r2", "r series model soc", "min cell temp", "max cell temp", "voltage", "current"]
PERCELL = ["SOH CELL", "BATT CAP", "FSK"]


def _stats(prefix: str, a: np.ndarray) -> dict:
    a = a[np.isfinite(a)]
    if not len(a):
        return {f"{prefix}_{k}": np.nan for k in ("min", "mean", "max", "std")}
    return {f"{prefix}_min": float(a.min()), f"{prefix}_mean": float(a.mean()), f"{prefix}_max": float(a.max()),
            f"{prefix}_std": float(a.std())}


def summarise_member(raw_bytes: bytes, b: int) -> dict:
    hdr = pd.read_csv(io.BytesIO(raw_bytes), nrows=0)
    hdr.columns = [c.strip() for c in hdr.columns]
    cols_raw = list(pd.read_csv(io.BytesIO(raw_bytes), nrows=0).columns)
    want = {}
    for c_raw, c in zip(cols_raw, hdr.columns):
        if c == "time(ms)":
            want[c_raw] = "time_ms"
        for s in SCALARS:
            if c == f"bat {b} {s}":
                want[c_raw] = s
        m = re.match(rf"B{b} (SOH CELL|BATT CAP|FSK) (\d+)$", c)
        if m:
            want[c_raw] = f"{m.group(1)}|{m.group(2)}"
    try:
        d = pd.read_csv(io.BytesIO(raw_bytes), usecols=list(want), engine="pyarrow")
    except Exception:  # noqa: BLE001
        d = pd.read_csv(io.BytesIO(raw_bytes), usecols=list(want), low_memory=False)
    d = d.rename(columns=want).apply(pd.to_numeric, errors="coerce")
    out = {"n_rows": len(d)}
    if not len(d):
        return out
    # start-up evidence: which channel groups are exactly zero in the first sample
    first = d.iloc[0]
    for grp, keys in (("temp", ["min cell temp", "max cell temp"]), ("voltage", ["voltage"]), ("soc", ["soc"]),
                      ("soh", ["soh"]), ("percell", [c for c in d.columns if "|" in c])):
        ks = [k for k in keys if k in d]
        out[f"first_zero_{grp}"] = bool(ks and all(first[k] == 0 for k in ks))
    cell_cols = {p: [c for c in d.columns if c.startswith(p + "|")] for p in PERCELL}
    valid = np.ones(len(d), bool)
    valid[: min(5, len(d))] = False
    if cell_cols["SOH CELL"]:
        valid &= (d[cell_cols["SOH CELL"]].to_numpy() > 0).all(axis=1)
    v = d[valid] if valid.any() else d.iloc[0:0]
    out["n_valid_rows"] = int(valid.sum())
    for s in ("soc", "soh", "kalman soc", "coulomb soc out", "box avaliable energy", "cell est bat cap", "cap est",
              "cell tmp bat cap", "cell r2", "r series model soc"):
        if s in v and len(v):
            key = s.replace(" ", "_")
            out[f"{key}_first"] = float(v[s].iloc[0])
            out[f"{key}_last"] = float(v[s].iloc[-1])
            out[f"{key}_median"] = float(v[s].median())
    for a, bname in (("kalman soc", "kalman"), ("coulomb soc out", "coulomb")):
        if a in v and "soc" in v and len(v):
            diff = (v["soc"] - v[a]).to_numpy(float)
            out[f"soc_minus_{bname}_median"] = float(np.nanmedian(diff))
            out[f"soc_minus_{bname}_maxabs"] = float(np.nanmax(np.abs(diff)))
            out[f"soc_minus_{bname}_end"] = float(diff[-1])
    if "kalman soc" in v and "coulomb soc out" in v and len(v):
        dd = (v["kalman soc"] - v["coulomb soc out"]).to_numpy(float)
        out["kalman_minus_coulomb_median"] = float(np.nanmedian(dd))
        out["kalman_minus_coulomb_maxabs"] = float(np.nanmax(np.abs(dd)))
    for flag in ("cell flg rst coulomb", "cell flg new est batt cap"):
        if flag in v and len(v):
            f = v[flag].fillna(0).to_numpy() > 0
            key = flag.replace(" ", "_")
            out[f"{key}_frac"] = float(f.mean())
            out[f"{key}_rises"] = int(((~f[:-1]) & f[1:]).sum()) if len(f) > 1 else 0
    if "worst cell" in v and len(v):
        out["worst_cell_mode"] = float(v["worst cell"].mode().iloc[0])
    for p in PERCELL:
        cc = cell_cols[p]
        if cc and len(v):
            last = v[cc].iloc[-1].to_numpy(float)
            key = p.lower().replace(" ", "_")
            out.update(_stats(f"{key}_cells_last", last))
            out[f"{key}_n_cells"] = len(cc)
            out[f"{key}_changed_in_session"] = bool((v[cc].nunique() > 1).any())
    if cell_cols["SOH CELL"] and cell_cols["BATT CAP"] and len(v):
        out["uniform_cells"] = bool(out.get("soh_cell_cells_last_std", 1) == 0 and out.get("batt_cap_cells_last_std", 1) == 0)
    return out


def extract_bms_all(settings: Settings, log=print) -> pd.DataFrame:
    man = pd.read_csv(settings.data_dir / "raw" / "manifest.csv")
    man = man[man.status == "ok"].sort_values("flight_id")
    cache = settings.data_dir / "processed" / "bms_cache"
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, r in enumerate(man.itertuples(), 1):
        cp = cache / f"flight_{r.flight_id}.json"
        if cp.exists() and cp.stat().st_mtime >= Path(r.zip_path).stat().st_mtime:
            rec = json.loads(cp.read_text())
        else:
            rec = {"flight_id": int(r.flight_id), "aircraft_label": r.aircraft_label, "batteries": {}}
            with zipfile.ZipFile(r.zip_path) as z:
                for name in z.namelist():
                    kind = member_kind(name)
                    if kind in ("sub_1", "sub_2"):
                        b = int(kind[-1])
                        try:
                            rec["batteries"][str(b)] = summarise_member(z.read(name), b)
                        except Exception as e:  # noqa: BLE001
                            rec["batteries"][str(b)] = {"error": f"{type(e).__name__}: {e}"[:200]}
                    elif kind == "main":
                        with z.open(name) as f:
                            first = pd.read_csv(f, nrows=2)
                        first.columns = [c.strip() for c in first.columns]
                        sig = [c for c in first.columns if c not in ("time(ms)", "time(min)")]
                        rec["main_first_row_all_zero"] = bool((pd.to_numeric(first[sig].iloc[0], errors="coerce").fillna(0) == 0).all())
            cp.write_text(json.dumps(rec, default=float))
        for b, s in rec["batteries"].items():
            rows.append(dict(flight_id=rec["flight_id"], aircraft_label=rec["aircraft_label"], battery=int(b),
                             main_first_row_all_zero=rec.get("main_first_row_all_zero"), **s))
        if k % 50 == 0 or k == len(man):
            log(f"  bms {k}/{len(man)}")
    df = pd.DataFrame(rows)
    df.to_csv(settings.data_dir / "processed" / "bms_session.csv", index=False)
    return df
