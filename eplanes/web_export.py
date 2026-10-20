"""Website bundle and dataset release.

`python -m eplanes export-web`  -> outputs/web_bundle/ + outputs/eplanes_web_bundle_<ts>.zip
    A compact, self-describing copy of the dataset that powers the Eplanes website and the paper
    figures: one row per session, one row per (session, pack), downsampled traces (default 2 s),
    sign-corrected GPS tracks with matched airfields, within-session SOH steps, quality summaries,
    the data dictionary and the release manifest.

`python -m eplanes release`     -> outputs/release/<version>/
    The downloadable dataset: original portal ZIPs per aircraft (raw), the analysis-ready 10 Hz
    data per aircraft (clean), session tables, the 2 s traces, documentation and SHA-256 checksums.

Nothing here changes a raw value. Masking follows the analysis cleaning rule (default `leadzero`),
and avionics channels are set to missing when the avionics block is off (they read exactly 0 then).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import geo
from .analysis.features import DEFAULT_RULE, apply_rule
from .build.clean import iter_flights
from .config import SIGNAL_DESCRIPTIONS, Settings

BUNDLE_FORMAT = 1

# canonical column -> (bundle column, aggregation within a time bin)
SERIES = {
    **{f"bat_{b}_current": (f"b{b}_current_a", "mean") for b in (1, 2)},
    **{f"bat_{b}_voltage": (f"b{b}_voltage_v", "mean") for b in (1, 2)},
    **{f"bat_{b}_soc": (f"b{b}_soc_pct", "median") for b in (1, 2)},
    **{f"bat_{b}_soh": (f"b{b}_soh_pct", "median") for b in (1, 2)},
    **{f"bat_{b}_min_cell_volt": (f"b{b}_min_cell_mv", "min") for b in (1, 2)},
    **{f"bat_{b}_max_cell_volt": (f"b{b}_max_cell_mv", "max") for b in (1, 2)},
    **{f"bat_{b}_max_cell_temp": (f"b{b}_max_cell_temp_c", "max") for b in (1, 2)},
    **{f"bat_{b}_avg_cell_temp": (f"b{b}_avg_cell_temp_c", "mean") for b in (1, 2)},
    "motor_power": ("motor_power_kw", "mean"),
}
AVIONICS_SERIES = {"pressure_alt": ("pressure_alt_m", "mean"), "ias": ("ias", "mean"), "oat": ("oat_c", "mean"),
                   "ground_speed": ("ground_speed", "mean"), "heading": ("heading_deg", "median")}
GPS_COLS = ("lat", "lng")

SERIES_DICTIONARY = {
    "flight_id": ("Portal session identifier", ""),
    "t_s": ("Seconds since the start of the session recording (start of the time bin)", "s"),
    "utc": ("Absolute time = portal start time (UTC, minute resolution) + t_s", "UTC"),
    "bX_current_a": ("Pack current, mean over the bin; negative = charging", "A"),
    "bX_voltage_v": ("Pack voltage (96 cells in series), mean over the bin", "V"),
    "bX_soc_pct": ("BMS-reported state of charge, median over the bin (integer-quantised)", "%"),
    "bX_soh_pct": ("BMS-reported state of health, median over the bin (a BMS estimate, not a measured capacity)", "%"),
    "bX_min_cell_mv": ("Lowest cell voltage in the pack, minimum over the bin", "mV"),
    "bX_max_cell_mv": ("Highest cell voltage in the pack, maximum over the bin", "mV"),
    "bX_max_cell_temp_c": ("Highest cell temperature in the pack, maximum over the bin (integer sensor)", "degC"),
    "bX_avg_cell_temp_c": ("Mean cell temperature reported by the pack, mean over the bin", "degC"),
    "motor_power_kw": ("Motor power channel as logged (equals ~0.98 x pack 1 + pack 2 power)", "kW"),
    "pressure_alt_m": ("Pressure altitude (unit assumed metres); missing when the avionics are off", "m"),
    "ias": ("Indicated airspeed as logged (unit undocumented); missing when the avionics are off", "as logged"),
    "oat_c": ("Outside air temperature; missing when the avionics are off", "degC"),
    "ground_speed": ("GPS ground speed as logged (unit undocumented); missing when the avionics are off", "as logged"),
    "heading_deg": ("GPS heading", "deg"),
    "lat": ("Latitude, sign recovered from airfield matching (logger stores magnitudes only); missing if unresolved", "deg"),
    "lon": ("Longitude, sign recovered from airfield matching (logger stores magnitudes only); missing if unresolved", "deg"),
    "n_samples": ("Raw 10 Hz samples aggregated into this bin", ""),
}


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _bin_flight(d: pd.DataFrame, res_s: float) -> pd.DataFrame:
    t = d["time_s"].to_numpy(float) if "time_s" in d else d["time_min"].to_numpy(float) * 60
    ok = np.isfinite(t)
    d = d.loc[ok].copy()
    d["_bin"] = np.floor(t[ok] / res_s).astype(np.int64)
    aggs = {}
    for src, (dst, how) in {**SERIES, **AVIONICS_SERIES}.items():
        if src in d:
            aggs[dst] = (src, how)
    for c in GPS_COLS:
        if c in d:
            aggs[c] = (c, "median")
    aggs["n_samples"] = ("time_min", "size")
    out = d.groupby("_bin").agg(**aggs).reset_index()
    out["t_s"] = out.pop("_bin").astype(float) * res_s
    return out


def _soh_steps(x: pd.DataFrame, fid: int) -> list[dict]:
    """Every change of reported SOH inside a session (full 10 Hz resolution, after masking)."""
    rows = []
    t = x["time_min"].to_numpy(float)
    for b in (1, 2):
        c = f"bat_{b}_soh"
        if c not in x:
            continue
        s = x[c].to_numpy(float)
        ok = np.flatnonzero(np.isfinite(s) & (s > 0))
        if len(ok) < 2:
            continue
        sv = s[ok]
        ch = np.flatnonzero(np.diff(sv) != 0)
        soc = x.get(f"bat_{b}_soc", pd.Series(np.nan, index=x.index)).to_numpy(float)
        temp = x.get(f"bat_{b}_max_cell_temp", pd.Series(np.nan, index=x.index)).to_numpy(float)
        cur = x.get(f"bat_{b}_current", pd.Series(np.nan, index=x.index)).to_numpy(float)
        for k in ch:
            i0, i1 = ok[k], ok[k + 1]
            rows.append(dict(flight_id=fid, battery=b, t_min=float(t[i1]), soh_before=float(sv[k]), soh_after=float(sv[k + 1]),
                             delta=float(sv[k + 1] - sv[k]), soc_at_step=float(soc[i1]), max_cell_temp_c=float(temp[i1]),
                             current_a=float(cur[i1]), soc_at_start=float(np.nanmedian(soc[ok[:5]])) if len(ok) else np.nan,
                             temp_at_start_c=float(np.nanmedian(temp[ok[:50]])) if len(ok) else np.nan))
    return rows


def _session_frame(settings: Settings, rule: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(sessions: one row per session, packs: one row per session and pack)."""
    from .analysis.context import load_context
    proc = settings.data_dir / "processed"
    ctx = load_context(settings, rule)
    fl = pd.read_csv(proc / "flights_with_qc.csv", parse_dates=["start_datetime", "flight_date"])
    use_p = settings.outputs_dir / "quality" / "usability_matrix_by_flight.csv"
    use = pd.read_csv(use_p) if use_p.exists() else pd.DataFrame({"flight_id": fl.flight_id})
    keep = [c for c in ["flight_id", "account", "aircraft_id", "aircraft_label", "fleet", "download_status", "flight_type", "flight_type_raw",
                        "page_registration", "page_serial_number", "start_datetime", "n_rows", "duration_min", "median_dt_min", "n_gaps",
                        "max_gap_min", "n_duplicate_time", "n_nonmonotonic", "n_out_of_range", "n_sentinel", "frac_avionics_valid",
                        "start_state", "frac_charge", "frac_discharge", "n_segments", "max_abs_current", "is_duplicate_content", "cross_aircraft_duplicate",
                        "has_gap", "activity_from_data", "label_content_mismatch", "scale_anomaly", "has_timeseries", "offset_h_rounded"]
            if c in fl]
    s = fl[keep].copy()
    s = s.rename(columns={"start_datetime": "start_utc", "page_registration": "registration", "page_serial_number": "serial_number",
                          "flight_type": "session_type", "flight_type_raw": "session_type_portal"})
    ucols = [c for c in use.columns if c not in ("aircraft_label", "flight_type") and c != "flight_id"]
    s = s.merge(use[["flight_id"] + ucols].rename(columns={c: f"usable_{c}" if not c.startswith("usable") else c for c in ucols}),
                on="flight_id", how="left")
    s["end_utc"] = s["start_utc"] + pd.to_timedelta(s["duration_min"], unit="min")
    packs = ctx.copy()
    drop = [c for c in packs.columns if c in ("aircraft_label", "fleet", "flight_type", "start_datetime", "flight_date", "start_state",
                                               "n_gaps", "max_gap_min", "is_duplicate_content", "activity_from_data",
                                               "label_content_mismatch") or c.endswith(("_A_overview", "_B_profiles"))]
    packs = packs.drop(columns=[c for c in drop if c in packs])
    bms_p = proc / "bms_session.csv"
    if bms_p.exists():
        bms = pd.read_csv(bms_p)
        bkeep = [c for c in ["flight_id", "battery", "kalman_soc_median", "coulomb_soc_out_median", "soc_minus_kalman_median",
                             "soc_minus_coulomb_median", "cell_est_bat_cap_last", "soh_cell_cells_last_min", "soh_cell_cells_last_mean",
                             "soh_cell_cells_last_max", "soh_cell_cells_last_std", "batt_cap_cells_last_mean", "soh_cell_changed_in_session",
                             "uniform_cells", "cell_flg_new_est_batt_cap_rises", "cell_flg_rst_coulomb_rises"] if c in bms]
        packs = packs.merge(bms[bkeep], on=["flight_id", "battery"], how="left")
    # headline per-pack values onto the session row (b1_*, b2_*)
    head = ["soc_start", "soc_end", "delta_soc", "soh_start", "soh_end", "soh_median", "ah_discharged", "ah_charged", "energy_discharged_kwh",
            "energy_charged_kwh", "i_peak_discharge_a", "i_peak_charge_a", "i_mean_abs_a", "power_peak_kw", "temp_start_c", "temp_max_c",
            "temp_delta_c", "charge_time_min", "implied_capacity_ah", "r0_proxy_mohm_cell", "soh_cell_cells_last_min", "soh_cell_cells_last_mean",
            "uniform_cells"]
    for b in (1, 2):
        pb = packs[packs.battery == b][["flight_id"] + [c for c in head if c in packs]]
        s = s.merge(pb.rename(columns={c: f"b{b}_{c}" for c in pb.columns if c != "flight_id"}), on="flight_id", how="left")
    for c in ("oat_c_median", "airborne_min", "climb_min", "cruise_min", "descent_min", "alt_gain_m", "has_flight_state",
              "pack_total_power_peak_kw", "motor_power_peak_kw", "motor_over_pack_power_median"):
        if c in packs:
            s = s.merge(packs[packs.battery == 1][["flight_id", c]], on="flight_id", how="left")
    return s, packs


def export_web(settings: Settings, rule: str = DEFAULT_RULE, resolution_s: float = 2.0, out_dir: Path | None = None,
               make_zip: bool = True, log=print) -> Path:
    out = Path(out_dir or settings.outputs_dir / "web_bundle")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    sessions, packs = _session_frame(settings, rule)
    log(f"[web] {len(sessions)} sessions, {len(packs)} pack rows")

    clean = settings.data_dir / "processed" / "analysis_ready_timeseries.parquet"
    want = list(SERIES) + list(AVIONICS_SERIES) + list(GPS_COLS) + ["aircraft_label", "time_min", "time_s", "avionics_valid", "quality_flag",
                                                                     "init_unsettled_groups"]
    bins, steps, endpoints = [], [], []
    for g in iter_flights(clean, want):
        fid = int(g["flight_id"].iloc[0])
        x = apply_rule(g, rule)
        av = x["avionics_valid"].to_numpy(bool) if "avionics_valid" in x else np.zeros(len(x), bool)
        for c in list(AVIONICS_SERIES) + list(GPS_COLS) + ["motor_power"]:
            if c in x:
                if c != "motor_power":
                    x.loc[~av, c] = np.nan
                # avionics channels also open with exact zeros before they report (e.g. OAT 0 for the first samples)
                v = x[c].to_numpy(float)
                nz = np.flatnonzero(np.isfinite(v) & (v != 0))
                if c != "motor_power" and len(nz) and nz[0] > 0:
                    x.loc[x.index[: nz[0]], c] = np.nan
        if all(c in x for c in GPS_COLS):
            okf = geo.valid_fix_mask(x["lat"], x["lng"])
            x.loc[~okf, list(GPS_COLS)] = np.nan
            idx = np.flatnonzero(okf)
            if len(idx):
                for pos, i in (("start", idx[0]), ("end", idx[-1])):
                    endpoints.append(dict(flight_id=fid, aircraft_label=g["aircraft_label"].iloc[0], pos=pos,
                                          lat=float(x["lat"].iat[i]), lng=float(x["lng"].iat[i])))
        steps.extend(_soh_steps(x, fid))
        b = _bin_flight(x, resolution_s)
        b.insert(0, "flight_id", fid)
        bins.append(b)
    series = pd.concat(bins, ignore_index=True) if bins else pd.DataFrame(columns=["flight_id", "t_s"])
    ep = pd.DataFrame(endpoints, columns=["flight_id", "aircraft_label", "pos", "lat", "lng"])

    # --- GPS: signs per aircraft, then airports per session -------------------------------------------------
    signs = geo.resolve_signs(ep) if len(ep) else pd.DataFrame(columns=["aircraft_label", "lat_sign", "lng_sign", "resolved"])
    signs.to_csv(out / "gps_sign_resolution.csv", index=False)
    log("[web] GPS sign resolution:\n" + (signs.to_string(index=False) if len(signs) else "  no GPS fixes"))
    ac_of = sessions.set_index("flight_id")["aircraft_label"]
    sg = signs.set_index("aircraft_label") if len(signs) else pd.DataFrame()
    if "lat" in series:
        lab = series["flight_id"].map(ac_of)
        ls = lab.map(sg["lat_sign"]) if len(sg) else np.nan
        gs_ = lab.map(sg["lng_sign"]) if len(sg) else np.nan
        series["lat"] = series["lat"].abs() * ls
        series["lon"] = series.pop("lng").abs() * gs_
    airports_used, ap_rows = {}, []
    if len(ep) and len(sg):
        ep["lat_s"] = ep.lat.abs() * ep.aircraft_label.map(sg["lat_sign"])
        ep["lon_s"] = ep.lng.abs() * ep.aircraft_label.map(sg["lng_sign"])
        for r in ep.dropna(subset=["lat_s", "lon_s"]).itertuples():
            code, name, dkm = geo.nearest_airport(r.lat_s, r.lon_s, 3.0)
            ap_rows.append(dict(flight_id=r.flight_id, pos=r.pos, code=code, dist_km=dkm))
            if code:
                airports_used[code] = name
    ap = pd.DataFrame(ap_rows, columns=["flight_id", "pos", "code", "dist_km"])
    if len(ap):
        piv = ap.pivot_table(index="flight_id", columns="pos", values="code", aggfunc="first")
        sessions = sessions.merge(piv.rename(columns={"start": "dep_airport", "end": "arr_airport"}).reset_index(), on="flight_id", how="left")
    for c in ("dep_airport", "arr_airport"):
        if c not in sessions:
            sessions[c] = None
    # track summaries
    tsum = []
    if "lat" in series:
        for fid, g in series.dropna(subset=["lat", "lon"]).groupby("flight_id"):
            ts = geo.track_summary(g.lat, g.lon, g.get("pressure_alt_m"), g.t_s)
            tsum.append(dict(flight_id=fid, gps_points=len(g), **ts))
    tsum = pd.DataFrame(tsum, columns=["flight_id", "gps_points", "track_km", "max_range_km", "alt_range_m"])
    sessions = sessions.merge(tsum, on="flight_id", how="left")
    sessions["gps_points"] = sessions["gps_points"].fillna(0).astype(int)
    sessions["has_gps_track"] = sessions["gps_points"] >= 10

    # absolute time per bin
    st = series["flight_id"].map(sessions.set_index("flight_id")["start_utc"])
    series.insert(2, "utc", st + pd.to_timedelta(series["t_s"], unit="s"))
    series = series.sort_values(["flight_id", "t_s"]).reset_index(drop=True)
    num = series.select_dtypes("float64").columns.difference(["t_s", "lat", "lon"])
    series[num] = series[num].astype("float32")

    airports = geo.load_airports()
    airports = airports[airports.code.isin(airports_used)][["code", "name", "latitude_deg", "longitude_deg", "elevation_ft",
                                                             "iso_country", "iso_region", "municipality", "iata_code", "type"]]
    home = (sessions.dropna(subset=["dep_airport"]).groupby("aircraft_label").dep_airport
            .agg(lambda v: v.value_counts().index[0]) if sessions.dep_airport.notna().any() else pd.Series(dtype=object))
    aircraft = []
    for acct, a in settings["accounts"].items():
        for ac in a["aircraft"]:
            g = sessions[sessions.aircraft_label == ac["label"]]
            reg = g.registration.dropna().mode() if "registration" in g else pd.Series(dtype=object)
            ser = g.serial_number.dropna().mode() if "serial_number" in g else pd.Series(dtype=object)
            aircraft.append(dict(label=ac["label"], portal_aircraft_id=ac["aircraft_id"], fleet=acct,
                                 registration=str(reg.iat[0]) if len(reg) else None, serial_number=str(ser.iat[0]) if len(ser) else None,
                                 type="Pipistrel Velis Electro (Virus SW 128)", packs=2, cells_in_series=settings.raw.get("n_series_cells"),
                                 home_airfield=home.get(ac["label"]) if len(home) else None, sessions=int(len(g)),
                                 first_utc=str(g.start_utc.min()) if len(g) else None, last_utc=str(g.start_utc.max()) if len(g) else None))

    # --- write -------------------------------------------------------------------------------------------------
    sessions.to_parquet(out / "sessions.parquet", index=False)
    packs.to_parquet(out / "packs.parquet", index=False)
    pq.write_table(pa.Table.from_pandas(series, preserve_index=False), out / "series.parquet", row_group_size=100_000,
                   compression="zstd")
    pd.DataFrame(steps).to_parquet(out / "soh_steps.parquet", index=False)
    airports.to_csv(out / "airports.csv", index=False)
    (out / "aircraft.json").write_text(json.dumps(aircraft, indent=1, default=str))
    _copy_quality(settings, out)
    _dictionary(settings, sessions, packs).to_csv(out / "data_dictionary.csv", index=False)
    rel = settings.outputs_dir / "release"
    mans = sorted(rel.glob("*/release_manifest.json")) if rel.exists() else []
    if mans:
        shutil.copy(mans[-1], out / "release_manifest.json")
    st_valid = sessions.start_utc.dropna()
    manifest = dict(bundle_format=BUNDLE_FORMAT, created_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                    pipeline_commit=_git_commit(settings.root), cleaning_rule=rule, resolution_s=resolution_s,
                    n_sessions=int(len(sessions)), n_aircraft=int(sessions.aircraft_label.nunique()), n_series_rows=int(len(series)),
                    first_session_utc=str(st_valid.min()) if len(st_valid) else None,
                    last_session_utc=str(st_valid.max()) if len(st_valid) else None,
                    source="Pipistrel cloud portal exports (main log 10 Hz, BMS files 5 Hz)",
                    synthetic=(settings.data_dir / "SYNTHETIC").exists())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log(f"[web] bundle written to {out} ({sum(f.stat().st_size for f in out.iterdir()) / 1e6:.1f} MB)")
    if make_zip:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M")
        zp = settings.outputs_dir / f"eplanes_web_bundle_{ts}.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(out.iterdir()):
                z.write(f, f"web_bundle/{f.name}")
        log(f"[web] zipped -> {zp} ({zp.stat().st_size / 1e6:.1f} MB)  <- upload this file")
    return out


def _copy_quality(settings: Settings, out: Path):
    q = settings.outputs_dir / "quality"
    gate = q / "QUALITY_COMPLETE.json"
    payload = json.loads(gate.read_text()) if gate.exists() else {}
    for name in ("usability_by_aircraft_and_analysis.csv", "qc_download_status.csv", "qc_startup_classes.csv", "qc_duplicates.csv",
                 "qc_timestamps.csv", "qc_session_plausibility.csv", "usability_by_battery_signal.csv"):
        if (q / name).exists():
            shutil.copy(q / name, out / f"quality__{name}")
    (out / "quality.json").write_text(json.dumps(payload, indent=1, default=str))


def _dictionary(settings: Settings, sessions: pd.DataFrame, packs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, (desc, unit) in SERIES_DICTIONARY.items():
        rows.append(dict(table="series (2 s traces)", column=k, unit=unit, description=desc))
    sdesc = {
        "flight_id": "Portal session identifier (unique)", "aircraft_label": "Aircraft (registration or serial as used by the operator)",
        "fleet": "Portal account the aircraft belongs to", "session_type": "Session type as labelled on the portal",
        "start_utc": "Session start (portal, minute resolution; on the avionics GPS clock, i.e. UTC)", "end_utc": "start_utc + duration",
        "duration_min": "Recording duration", "n_rows": "Rows in the 10 Hz main log", "activity_from_data": "Dominant activity from measured current",
        "label_content_mismatch": "Portal label contradicts the measured current direction", "scale_anomaly": "Current/voltage scaling fault (exclude from electrical analyses)",
        "is_duplicate_content": "Copy of another session with identical content; excluded from analyses",
        "cross_aircraft_duplicate": "The same log is listed under more than one aircraft; only the copy matching that aircraft's SOH history is kept", "has_gap": "Contains a logging gap > max(5 dt, 2 s)",
        "dep_airport": "Airfield at the first valid GPS fix (within 3 km)", "arr_airport": "Airfield at the last valid GPS fix (within 3 km)",
        "track_km": "GPS track length (2 s points, jumps > 600 km/h removed)", "has_gps_track": "At least 10 valid GPS points",
        "bX_soc_start/bX_soc_end": "Reported SOC after the start-up window / at the end, pack X", "bX_soh_median": "Median reported SOH in the session, pack X",
        "bX_ah_discharged / bX_ah_charged": "Charge out / in, integrated from current (gaps not bridged)", "bX_energy_discharged_kwh": "Energy out, integrated V x I",
        "bX_temp_max_c": "Maximum of the pack's maximum-cell-temperature channel", "usable_*": "Analysis-block usability flags from the quality stage (A-G)",
    }
    for k, v in sdesc.items():
        rows.append(dict(table="sessions", column=k, unit="", description=v))
    for k, (desc, unit) in SIGNAL_DESCRIPTIONS.items():
        rows.append(dict(table="analysis-ready 10 Hz", column=k if k == "time_min" else f"bat_<1|2>_{k}", unit=unit, description=desc))
    rows += [dict(table="analysis-ready 10 Hz", column="quality_flag", unit="bitmask", description="Row-level quality flags (see TRANSFORMATIONS.md)"),
             dict(table="analysis-ready 10 Hz", column="segment_type", unit="", description="charge / discharge / rest from the 30 s rolling median of current"),
             dict(table="analysis-ready 10 Hz", column="avionics_valid", unit="bool", description="Avionics block alive (ACC_NORM > 0.5 g); when False its channels read 0 = not logged")]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- release
def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_release(settings: Settings, version: str = "1.0.0", rule: str = DEFAULT_RULE, include_raw: bool = True, log=print) -> Path:
    """Write the downloadable dataset release with a manifest the website reads."""
    rel = settings.outputs_dir / "release" / f"v{version}"
    rel.mkdir(parents=True, exist_ok=True)
    files = []

    def add(path: Path, kind: str, subset: str, description: str, fmt: str):
        files.append(dict(file=path.name, kind=kind, subset=subset, format=fmt, description=description,
                          bytes=path.stat().st_size, sha256=_sha256(path)))
        log(f"[release] {path.name}  {path.stat().st_size / 1e6:.1f} MB")

    bundle = settings.outputs_dir / "web_bundle"
    if not (bundle / "sessions.parquet").exists():
        export_web(settings, rule, make_zip=False, log=log)
    sessions = pd.read_parquet(bundle / "sessions.parquet")
    packs = pd.read_parquet(bundle / "packs.parquet")
    p = rel / "eplanes_sessions.csv"
    sessions.to_csv(p, index=False)
    add(p, "tables", "all", "One row per session: aircraft, type, UTC times, quality flags, airfields, headline per-pack values", "csv")
    p = rel / "eplanes_pack_sessions.csv"
    packs.to_csv(p, index=False)
    add(p, "tables", "all", "One row per session and pack: SOC/SOH, charge, energy, current, power, temperature, BMS internals", "csv")
    p = rel / "eplanes_traces_2s.parquet"
    shutil.copy(bundle / "series.parquet", p)
    add(p, "clean", "all", "All sessions resampled to 2 s (cleaned, start-up zeros masked, GPS sign-corrected)", "parquet")
    p = rel / "eplanes_soh_steps.csv"
    pd.read_parquet(bundle / "soh_steps.parquet").to_csv(p, index=False)
    add(p, "tables", "all", "Every change of reported SOH inside a session with time, SOC and temperature at the step", "csv")

    clean = settings.data_dir / "processed" / "analysis_ready_timeseries.parquet"
    pf = pq.ParquetFile(clean)
    labels = sorted(sessions.aircraft_label.dropna().unique())
    writers = {}
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i)
        lab = t.column("aircraft_label")[0].as_py()
        if lab not in writers:
            writers[lab] = pq.ParquetWriter(rel / f"eplanes_clean_10hz_{lab}.parquet", t.schema, compression="zstd")
        writers[lab].write_table(t.cast(writers[lab].schema))
    for lab, w in writers.items():
        w.close()
        add(rel / f"eplanes_clean_10hz_{lab}.parquet", "clean", lab,
            f"Analysis-ready 10 Hz data for {lab}: every raw row and value kept, plus quality flags and derived columns", "parquet")

    if include_raw:
        man = pd.read_csv(settings.data_dir / "raw" / "manifest.csv")
        zdir = settings.data_dir / "raw" / "zips"
        for lab in labels:
            p = rel / f"eplanes_raw_portal_exports_{lab}.zip"
            with zipfile.ZipFile(p, "w", zipfile.ZIP_STORED) as z:
                for r in man[(man.aircraft_label == lab) & (man.status == "ok")].itertuples():
                    zp = Path(str(getattr(r, "zip_path", "") or ""))
                    if not zp.is_file():  # manifest paths are absolute; fall back to the standard layout if the data moved
                        zp = zdir / lab / f"flight_{r.flight_id}.zip"
                    if zp.is_file():
                        z.write(zp, f"{lab}/{zp.name}")
            add(p, "raw", lab, f"Original portal ZIP exports for {lab}, byte-for-byte (main log, two BMS files, warnings per session)", "zip of zip")

    proc = settings.data_dir / "processed"
    for name, desc in (("TRANSFORMATIONS.md", "Every transformation from raw to analysis-ready"),):
        if (proc / name).exists():
            shutil.copy(proc / name, rel / name)
            add(rel / name, "docs", "all", desc, "markdown")
    shutil.copy(bundle / "data_dictionary.csv", rel / "data_dictionary.csv")
    add(rel / "data_dictionary.csv", "docs", "all", "Columns, units and descriptions for every table", "csv")
    readme = rel / "README.md"
    readme.write_text(_release_readme(version, sessions))
    add(readme, "docs", "all", "What is in the release and how to load it", "markdown")
    manifest = dict(dataset="Eplanes open field dataset of electric-aircraft battery operations", version=version,
                    created_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                    first_session_utc=str(sessions.start_utc.min()), last_session_utc=str(sessions.start_utc.max()),
                    n_sessions=int(len(sessions)), aircraft=labels, licence=settings.raw.get("release", {}).get("licence", "TO BE CONFIRMED"),
                    doi=settings.raw.get("release", {}).get("doi"), files=files)
    (rel / "release_manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    (rel / "checksums.sha256").write_text("".join(f"{f['sha256']}  {f['file']}\n" for f in files))
    log(f"[release] {len(files)} files in {rel}")
    return rel


def _release_readme(version: str, sessions: pd.DataFrame) -> str:
    return f"""# Eplanes open field dataset, version {version}

Battery and flight telemetry from Pipistrel Velis Electro aircraft, exported from the Pipistrel cloud portal.
Sessions: {len(sessions):,}; first {sessions.start_utc.min()} UTC, last {sessions.start_utc.max()} UTC.

## Files
- `eplanes_sessions.csv` - one row per session (start here).
- `eplanes_pack_sessions.csv` - one row per session and battery pack.
- `eplanes_soh_steps.csv` - every change of the BMS-reported SOH inside a session.
- `eplanes_traces_2s.parquet` - all sessions at 2 s resolution (cleaned).
- `eplanes_clean_10hz_<aircraft>.parquet` - full-resolution analysis-ready data (no rows removed; flags added).
- `eplanes_raw_portal_exports_<aircraft>.zip` - original portal exports, byte-for-byte.
- `data_dictionary.csv`, `TRANSFORMATIONS.md`, `checksums.sha256`, `release_manifest.json`.

## Load
```python
import pandas as pd
s = pd.read_csv("eplanes_sessions.csv", parse_dates=["start_utc", "end_utc"])
tr = pd.read_parquet("eplanes_traces_2s.parquet", filters=[("flight_id", "==", int(s.flight_id.iloc[0]))])
```

## Read before analysing
- Current is negative while charging. Cell voltages are in mV, pack voltage in V; 96 cells in series.
- The first samples of most sessions are zeros written before the channels report; they are masked (NaN) in the 2 s traces
  and flagged (`quality_flag`) in the 10 Hz data.
- Avionics channels (OAT, altitude, GPS) read exactly 0 when the avionics are off; they are NaN in the 2 s traces.
- Reported SOH is the BMS estimate for the weakest cell, not a measured capacity, and it is re-estimated mostly during charging.
- Sessions with `scale_anomaly = True` have a current/voltage scaling fault; exclude them from electrical analyses.
"""
