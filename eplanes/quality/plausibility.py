"""Session-level physical plausibility (data-driven; no hard-coded session IDs).

Motivation (real data, Oct 2026): 19 C-GAUW sessions of Jul-Aug 2023 log |current| ~320 A and max cell voltage
up to ~10 V while the reported SOC changes normally, i.e. charge integrated from current is 5-16x what the SOC
change implies. Such sessions must not enter current-, voltage- or energy-based analyses.

Per session and pack:
  ah_per_soc_point   integrated |current| (Ah) per SOC point traversed (sum |dSOC|); physically ~ capacity/100
  max_abs_current_a, frac_cellv_over_4300, frac_packv_over_420
A pack is implausible if  max |I| > screening limit, OR > 0.1 % of cell-voltage samples exceed 4.3 V,
OR ah_per_soc_point > 3x (or < 1/3 of) the fleet median (SOC traversed >= 10 points).
Also returns out-of-range sample counts per signal family (excluding each signal's leading zero run).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..build.clean import iter_flights
from ..config import Settings

COLS = ["time_min", "aircraft_label"] + [f"bat_{b}_{s}" for b in (1, 2) for s in
                                          ("current", "soc", "soh", "min_cell_volt", "max_cell_volt", "voltage",
                                           "max_cell_temp", "min_cell_temp")]


def _lead_masked(x: np.ndarray) -> np.ndarray:
    x = x.astype(float).copy()
    nz = np.flatnonzero(np.isfinite(x) & (x != 0))
    if len(nz) and nz[0] > 0:
        x[: nz[0]] = np.nan
    return x


def session_plausibility(settings: Settings, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = settings.raw
    b = cfg["screening_bounds"]
    gap_min = cfg["quality"].get("gap_min_s", 2.0) / 60.0
    path = settings.data_dir / "processed" / "analysis_ready_timeseries.parquet"
    rows, oor = [], []
    for g in iter_flights(path, COLS):
        fid = int(g.flight_id.iloc[0])
        ac = g.aircraft_label.iloc[0] if "aircraft_label" in g else ""
        t = g.time_min.to_numpy(float)
        dt = np.diff(t)
        pos = dt[np.isfinite(dt) & (dt > 0)]
        max_dt = max(cfg["quality"]["gap_factor"] * float(np.median(pos)), gap_min) if len(pos) else np.inf
        rec = dict(flight_id=fid, aircraft_label=ac)
        o = dict(flight_id=fid, aircraft_label=ac)
        for k in (1, 2):
            i = _lead_masked(g[f"bat_{k}_current"].to_numpy()) if f"bat_{k}_current" in g else np.full(len(g), np.nan)
            s = _lead_masked(g[f"bat_{k}_soc"].to_numpy()) if f"bat_{k}_soc" in g else np.full(len(g), np.nan)
            vmax = _lead_masked(g[f"bat_{k}_max_cell_volt"].to_numpy()) if f"bat_{k}_max_cell_volt" in g else np.full(len(g), np.nan)
            vp = _lead_masked(g[f"bat_{k}_voltage"].to_numpy()) if f"bat_{k}_voltage" in g else np.full(len(g), np.nan)
            ia = np.abs(i)
            im = 0.5 * (ia[1:] + ia[:-1])
            okk = np.isfinite(im) & (dt > 0) & (dt <= max_dt)
            ah = float(np.sum(im[okk] * dt[okk]) / 60.0)
            ds = np.abs(np.diff(s))
            soc_trav = float(np.nansum(ds[ds < 20]))  # ignore implausible single-step SOC jumps
            rec[f"b{k}_ah"] = ah
            rec[f"b{k}_soc_traversed"] = soc_trav
            rec[f"b{k}_ah_per_soc_point"] = ah / soc_trav if soc_trav >= 10 else np.nan
            rec[f"b{k}_max_abs_current_a"] = float(np.nanmax(ia)) if np.isfinite(ia).any() else np.nan
            fv = vmax[np.isfinite(vmax)]
            rec[f"b{k}_frac_cellv_over_4300"] = float((fv > 4300).mean()) if len(fv) else np.nan
            fp = vp[np.isfinite(vp)]
            rec[f"b{k}_frac_packv_over_420"] = float((fp > 420).mean()) if len(fp) else np.nan
            for fam, col, lo, hi in (("current", "current", -b["abs_current_a"][1], b["abs_current_a"][1]),
                                     ("soc", "soc", *b["soc"]), ("soh", "soh", *b["soh"]),
                                     ("min_cell_volt", "min_cell_volt", *b["cell_volt_mv"]),
                                     ("max_cell_volt", "max_cell_volt", *b["cell_volt_mv"]),
                                     ("pack_voltage", "voltage", *b.get("pack_voltage_v", [250, 420])),
                                     ("max_cell_temp", "max_cell_temp", *b["temp_c"]),
                                     ("min_cell_temp", "min_cell_temp", *b["temp_c"])):
                c = f"bat_{k}_{col}"
                if c in g:
                    x = _lead_masked(g[c].to_numpy())
                    o[f"b{k}_{fam}"] = int((np.isfinite(x) & ((x < lo) | (x > hi))).sum())
        rows.append(rec)
        oor.append(o)
    p = pd.DataFrame(rows)
    med = np.nanmedian(pd.concat([p.b1_ah_per_soc_point, p.b2_ah_per_soc_point]))
    lim = cfg["screening_bounds"]["abs_current_a"][1]
    for k in (1, 2):
        r = p[f"b{k}_ah_per_soc_point"] / med
        p[f"b{k}_implausible"] = ((p[f"b{k}_max_abs_current_a"] > lim) | (p[f"b{k}_frac_cellv_over_4300"] > 0.001) |
                                  (r > 3) | (r < 1 / 3)).fillna(False)
    p["fleet_median_ah_per_soc_point"] = med
    p["scale_anomaly"] = p.b1_implausible | p.b2_implausible
    return p, pd.DataFrame(oor)
