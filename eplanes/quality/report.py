"""Data-quality validation, usability matrix, report and the quality GATE.

The analysis stages refuse to run unless outputs/quality/QUALITY_COMPLETE.json exists
(see require_quality_gate), implementing "no scientific analysis before QC is documented".
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..build.clean import SIGNALS
from ..build.flights import build_flights_table
from ..config import Settings
from ..viz.style import (AIRCRAFT_STYLE, BLUE, GRID, INK2, ORANGE, SEQ, FigureSink,
                         aircraft_color)

GATE = "QUALITY_COMPLETE.json"


def require_quality_gate(settings: Settings):
    p = settings.outputs_dir / "quality" / GATE
    if not p.exists():
        raise SystemExit("Quality gate not passed: run `eplanes quality` first "
                         f"(missing {p}). Analysis must not start before data-quality checks are documented.")
    return json.loads(p.read_text())


def _md_table(df: pd.DataFrame, floatfmt=".3g") -> str:
    if df.empty:
        return "_(none)_\n"
    return df.to_markdown(index=False, floatfmt=floatfmt) + "\n" if hasattr(df, "to_markdown") else df.to_string(index=False) + "\n"


def run_quality(settings: Settings, external_flights: str | None = None, log=print) -> dict:
    cfg = settings.raw
    proc = settings.data_dir / "processed"
    qdir = settings.outputs_dir / "quality"
    qdir.mkdir(parents=True, exist_ok=True)
    figs = FigureSink(qdir / "figures", "qc_")

    flights = build_flights_table(settings)
    flights.to_csv(proc / "flights_table.csv", index=False)
    qc = pd.read_csv(proc / "flight_qc.csv")
    init = pd.read_csv(proc / "init_records.csv")
    man = pd.read_csv(settings.data_dir / "raw" / "manifest.csv")
    members = pd.read_csv(settings.data_dir / "raw_combined" / "zip_members.csv")
    coerce = pd.read_csv(settings.data_dir / "raw_combined" / "coercion_log.csv")
    f = flights.merge(qc, on="flight_id", how="left")
    findings: list[str] = []
    limits: list[str] = []

    # 0 download completeness ---------------------------------------------------
    dl = man.groupby(["aircraft_label", "status"]).size().unstack(fill_value=0).reset_index()
    dl.to_csv(qdir / "qc_download_status.csv", index=False)
    n_bad = int((man.status != "ok").sum())
    findings.append(f"Download: {len(man)} flights listed, {int((man.status == 'ok').sum())} downloaded OK, {n_bad} not usable "
                    f"({', '.join(f'{k}={v}' for k, v in man.status.value_counts().items() if k != 'ok') or 'none'}).")

    # 1 duplicates ----------------------------------------------------------------
    dup_rows = []
    for name, key in (("identical ZIP bytes", "zip_sha256"), ("identical numeric content", "content_hash")):
        g = f[f[key].notna() & (f[key] != "")].groupby(key)["flight_id"].apply(list)
        for k, ids in g[g.map(len) > 1].items():
            dup_rows.append(dict(kind=name, flight_ids=ids, n=len(ids)))
    if "start_datetime" in f:
        g = f.dropna(subset=["start_datetime"]).groupby(["aircraft_label", "start_datetime"])["flight_id"].apply(list)
        for k, ids in g[g.map(len) > 1].items():
            dup_rows.append(dict(kind=f"same aircraft+start time {k[0]} {k[1]}", flight_ids=ids, n=len(ids)))
    dup = pd.DataFrame(dup_rows, columns=["kind", "flight_ids", "n"])
    dup.to_csv(qdir / "qc_duplicates.csv", index=False)
    findings.append(f"Duplicates: {len(dup)} duplicate groups (identical ZIP/content or same aircraft+start time); "
                    f"duplicated flight_id in manifest: {int(man.flight_id.duplicated().sum())}.")
    f["is_duplicate_content"] = f.duplicated("content_hash", keep="first") & f["content_hash"].notna()

    # 2 timestamps / sampling ------------------------------------------------------
    ts = f.groupby("aircraft_label").agg(
        flights=("flight_id", "size"), median_dt_min=("median_dt_min", "median"),
        flights_with_gaps=("n_gaps", lambda s: int((s > 0).sum())), total_gaps=("n_gaps", "sum"),
        max_gap_min=("max_gap_min", "max"), nonmono_samples=("n_nonmonotonic", "sum"),
        duplicate_time_samples=("n_duplicate_time", "sum"),
        missing_start_datetime=("start_datetime", lambda s: int(s.isna().sum()))).reset_index()
    ts.to_csv(qdir / "qc_timestamps.csv", index=False)
    nd = f["median_dt_min"].round(4).value_counts()
    findings.append(f"Sampling: median dt by flight = {', '.join(f'{k} min ({v} flights)' for k, v in nd.head(3).items())}; "
                    f"{int((f.n_gaps > 0).sum())} flights contain logging gaps > {cfg['quality']['gap_factor']}x median dt; "
                    f"{int(f.n_nonmonotonic.sum())} non-monotonic and {int(f.n_duplicate_time.sum())} duplicate time samples.")
    ordered = f.dropna(subset=["start_datetime"]).sort_values(["aircraft_label", "start_datetime"])
    # GPS time-of-day clock vs portal start time -> time-zone offset and clock agreement
    if "gps_tod_start_s" in f and f.gps_tod_start_s.notna().any():
        g = f.dropna(subset=["gps_tod_start_s", "start_datetime"]).copy()
        portal_tod = g.start_datetime.dt.hour * 3600 + g.start_datetime.dt.minute * 60
        off = ((g.gps_tod_start_s - portal_tod + 43200) % 86400 - 43200) / 3600.0
        g["offset_h"] = off
        g["offset_h_rounded"] = (off * 2).round() / 2
        g["month"] = g.start_datetime.dt.month
        g[["flight_id", "aircraft_label", "start_datetime", "gps_tod_start_s", "offset_h", "offset_h_rounded",
           "gps_clock_slope", "gps_n_ticks"]].to_csv(qdir / "qc_gps_clock_vs_portal.csv", index=False)
        vc = g.groupby("aircraft_label").offset_h_rounded.agg(lambda x: ", ".join(f"{k:+g} h: {v}" for k, v in x.value_counts().head(3).items()))
        drift_ppm = (g.gps_clock_slope - 1) * 1e6
        findings.append(f"Clocks: {len(g)} sessions carry the avionics time-of-day clock (TIMESTAMP, HHMMSS). Offset of that clock from the "
                        "portal start time (rounded to 0.5 h) by aircraft -> " + "; ".join(f"{k}: {v}" for k, v in vc.items()) +
                        f". Logger vs GPS clock rate differs by median {drift_ppm.median():.0f} ppm (IQR {drift_ppm.quantile(.25):.0f} to "
                        f"{drift_ppm.quantile(.75):.0f}). A constant offset that switches by 1 h between summer and winter would indicate "
                        "UTC avionics time and local portal time; see qc_gps_clock_vs_portal.csv.")
        f = f.merge(g[["flight_id", "offset_h_rounded"]], on="flight_id", how="left")
        limits.append("Portal start times are minute-resolution local times of undocumented zone; absolute times are cross-checked against "
                      "the avionics clock only for sessions where the avionics are alive (not charging).")
    else:
        limits.append("Portal timestamps give the session START (date and time of day) only; the time zone is not documented and "
                      "within-session time is `time_min` from the start.")

    # 3 feature availability --------------------------------------------------------
    av = []
    for ac, g in f.groupby("aircraft_label"):
        for s in SIGNALS:
            av.append(dict(aircraft=ac, signal=s, flights=len(g),
                           mean_missing_frac=g[f"missing_frac__{s}"].mean(),
                           frac_flights_all_missing=(g[f"missing_frac__{s}"] >= 0.999).mean(),
                           frac_flights_all_zero=g[f"all_zero__{s}"].mean(),
                           frac_flights_constant=g[f"constant__{s}"].mean()))
    av = pd.DataFrame(av)
    av.to_csv(qdir / "qc_feature_availability.csv", index=False)
    absent = av[av.frac_flights_all_missing > 0.99][["aircraft", "signal"]]
    if len(absent):
        findings.append("Signals absent for an aircraft: " + "; ".join(f"{a}:{s}" for a, s in absent.values) + ".")

    # 4 ranges / units / zeros ----------------------------------------------------
    rng = f.groupby("aircraft_label")[["n_out_of_range", "n_zero_sample_after_init", "n_row0_zero", "n_rows"]].sum().reset_index()
    rng.to_csv(qdir / "qc_range_violations.csv", index=False)
    findings.append(f"Screening: {int(f.n_out_of_range.sum()):,} samples outside screening bounds (flagged, not removed); "
                    f"{int(f.n_zero_sample_after_init.sum()):,} exact-zero voltage samples after row 0; "
                    f"{int(f.n_row0_zero.sum())}/{len(f)} flights have an all-zero first row.")
    # unit magnitudes (read a light sample per aircraft)
    path = settings.data_dir / "processed" / "analysis_ready_timeseries.parquet"
    samp = pd.read_parquet(path, columns=["aircraft_label", "bat_1_min_cell_volt", "bat_1_soc",
                                          "bat_1_max_cell_temp", "bat_1_current"])
    units = samp.replace(0, np.nan).groupby("aircraft_label").median().reset_index()
    units.to_csv(qdir / "qc_unit_magnitudes.csv", index=False)
    vol_ratio = units.bat_1_min_cell_volt.max() / max(units.bat_1_min_cell_volt.min(), 1e-9)
    findings.append("Units: median non-zero cell voltage magnitudes by aircraft: " +
                    ", ".join(f"{r.aircraft_label}={r.bat_1_min_cell_volt:.0f}" for r in units.itertuples()) +
                    f" -> {'consistent (all mV-scale)' if vol_ratio < 1.5 else 'INCONSISTENT magnitudes: check units'}.")
    del samp
    limits.append("Units are inferred from magnitudes (cell voltage in mV, current in A with negative = charging, temperature in degC); "
                  "the portal does not document them.")

    # 5 label vs content (session segmentation) ----------------------------------
    def act(r):
        if r.frac_charge > 0.2 and r.frac_discharge > 0.2:
            return "mixed"
        if r.frac_charge > 0.2:
            return "charge"
        if r.frac_discharge > 0.2:
            return "discharge"
        return "rest/low-current"
    f["activity_from_data"] = f.apply(act, axis=1)
    xt = pd.crosstab(f["flight_type"].fillna("unknown"), f["activity_from_data"])
    xt.reset_index().to_csv(qdir / "qc_label_vs_content.csv", index=False)
    mism = f[(f.flight_type == "charging") & (f.activity_from_data.isin(["discharge"])) |
             (f.flight_type.isin(["flight test"])) & (f.activity_from_data == "charge")]
    f["label_content_mismatch"] = f.flight_id.isin(mism.flight_id)
    findings.append(f"Session labelling: {len(mism)} flights where the portal label contradicts the dominant current direction "
                    f"(charging-labelled flights that discharge, or flight-test-labelled sessions that charge); "
                    f"{int((f.n_segments > 3).sum())} flights contain >3 activity segments (possible multi-activity sessions).")

    # 6 cross-source ----------------------------------------------------------------
    if external_flights:
        ext = pd.read_csv(external_flights)
        ext["flight_id"] = ext["flight_id"].astype(int)
        m = f.merge(ext, on="flight_id", how="outer", suffixes=("", "_ext"), indicator=True)
        cs = dict(only_in_scraped=int((m._merge == "left_only").sum()), only_in_external=int((m._merge == "right_only").sum()))
        if "flight_date_ext" in m:
            both = m[m._merge == "both"]
            cs["date_mismatch"] = int((pd.to_datetime(both.flight_date_ext).dt.normalize() != both.flight_date).sum())
        if "flight_type_ext" in m:
            cs["type_mismatch"] = int((both.flight_type_ext.str.lower() != both.flight_type).sum())
        pd.DataFrame([cs]).to_csv(qdir / "qc_cross_source.csv", index=False)
        findings.append("Cross-source (portal vs external flights table): " + ", ".join(f"{k}={v}" for k, v in cs.items()) + ".")
    else:
        limits.append("Cross-source reconciliation against the local SQL flights table was NOT run (pass --external-flights "
                      "with a CSV exported from the `flights` table: flight_id, flight_date, flight_type).")

    # 7 start-up artefacts ----------------------------------------------------------
    init = init.merge(f[["flight_id", "aircraft_label", "flight_type"]], on="flight_id", how="left")
    init["signal_short"] = init.signal.str.replace(r"^bat_\d_", "", regex=True)
    init["battery"] = init.signal.str.extract(r"bat_(\d)")
    tab = init.groupby(["aircraft_label", "signal", "cls"]).size().unstack(fill_value=0).reset_index()
    tab.to_csv(qdir / "qc_startup_classes.csv", index=False)
    st = init.groupby(["start_state", "signal_short", "cls"]).size().unstack(fill_value=0).reset_index()
    st.to_csv(qdir / "qc_startup_by_start_state.csv", index=False)
    sm = init[init.settle_idx > 0].groupby(["aircraft_label", "signal"]).settle_min.describe().reset_index()
    sm.to_csv(qdir / "qc_startup_settle_time.csv", index=False)
    z = init[(init.cls.isin(["zero_fill", "bin_avg_zero"]))]
    findings.append(f"Start-up readings: of {len(init)} flight-signal records, {int((init.cls=='zero_fill').sum())} are zero_fill, "
                    f"{int((init.cls=='bin_avg_zero').sum())} bin_avg_zero, {int((init.cls=='monotone_settling').sum())} monotone_settling, "
                    f"{int((init.cls=='load_onset').sum())} load_onset (treated as physical, NOT flagged), "
                    f"{int(init.extended_settling.sum()) if 'extended_settling' in init else 'n/a'} show settling beyond the first 2 samples. "
                    f"Median settle time of flagged records = {init[init.settle_idx>0].settle_min.median():.2f} min.")
    # 7b start-up evidence from the BMS files (independent stream at 5 Hz) ------------------
    from ..build.bms import extract_bms_all
    bms = extract_bms_all(settings, log=lambda *a: None)
    if len(bms):
        ev = pd.DataFrame([dict(evidence=k, frac_sessions=float(bms[k].mean()), n=int(bms[k].notna().sum()))
                           for k in ("main_first_row_all_zero", "first_zero_temp", "first_zero_voltage", "first_zero_soc",
                                     "first_zero_soh", "first_zero_percell") if k in bms])
        ev.to_csv(qdir / "qc_startup_bms_evidence.csv", index=False)
        e = dict(zip(ev.evidence, ev.frac_sessions))
        findings.append("Start-up evidence across streams: main-log first row all zero in "
                        f"{e.get('main_first_row_all_zero', float('nan')):.0%} of sessions; in the BMS files the first sample has zero "
                        f"temperatures in {e.get('first_zero_temp', float('nan')):.0%}, zero pack voltage in "
                        f"{e.get('first_zero_voltage', float('nan')):.0%} and zero SOC in {e.get('first_zero_soc', float('nan')):.0%}. "
                        "A zero confined to the temperature channels while voltage/SOC are valid is consistent with temperature "
                        "data not yet received at logger start, rather than with a sensor or BMS warm-up transient.")
        limits.append("The physical cause of zero first samples is inferred from cross-stream patterns, not documented by Pipistrel.")
    else:
        limits.append("Cause of start-up artefacts cannot be proven from logged data (no BMS files available): patterns are "
                      "reported as 'consistent with' logger zero-fill, resampling, or warm-up.")

    # 7c avionics block, invalid-value markers, pack topology --------------------------------
    av2 = f.groupby("aircraft_label").agg(sessions=("flight_id", "size"),
                                          sessions_with_avionics=("frac_avionics_valid", lambda x: int((x > 0.5).sum())),
                                          sentinel_samples=("n_sentinel", "sum"),
                                          series_ratio_b1_median=("series_ratio_b1", "median"),
                                          series_ratio_b2_median=("series_ratio_b2", "median")).reset_index()
    av2.to_csv(qdir / "qc_avionics_sentinels_topology.csv", index=False)
    by_type = f.groupby("flight_type").frac_avionics_valid.apply(lambda x: float((x > 0.5).mean())).to_dict()
    findings.append("Avionics block (OAT, pressure altitude, IAS, GPS, attitude) is alive (ACC_NORM > "
                    f"{cfg.get('avionics_valid_acc_norm', 0.5)} g) in: " + ", ".join(f"{k} {v:.0%}" for k, v in by_type.items()) +
                    " of sessions; when it is off all its channels read exactly 0 and are treated as not logged.")
    sr = pd.concat([f.series_ratio_b1, f.series_ratio_b2]).dropna()
    if len(sr):
        findings.append(f"Pack topology check: pack voltage / mean cell voltage has median {sr.median():.2f} "
                        f"(IQR {sr.quantile(.25):.2f}-{sr.quantile(.75):.2f}) across sessions, consistent with "
                        f"{settings.raw.get('n_series_cells')} cells in series.")
    findings.append(f"Invalid-value markers (65535): {int(f.n_sentinel.sum()):,} samples flagged (e.g. motor rpm); not removed.")

    # 8 usability matrix ------------------------------------------------------------
    minr = cfg["quality"]["min_flight_rows"]
    ok_rows = (f.n_rows >= minr) & (f.is_duplicate_content == False)  # noqa: E712
    def has(sig, b): return (f[f"missing_frac__bat_{b}_{sig}"] < 0.05) & (~f[f"all_zero__bat_{b}_{sig}"].fillna(True))
    U = pd.DataFrame({"flight_id": f.flight_id, "aircraft_label": f.aircraft_label, "flight_type": f.flight_type})
    U["A_overview"] = f.start_datetime.notna() & (f.download_status == "ok")
    U["B_profiles"] = ok_rows & has("current", 1) & has("current", 2) & (f.frac_discharge > 0.05) & (f.max_gap_min < 1.0)
    U["C_soc_dod"] = ok_rows & has("soc", 1) & has("soc", 2)
    U["D_charging"] = ok_rows & has("current", 1) & (f.frac_charge > 0.2)
    U["E_thermal"] = ok_rows & has("max_cell_temp", 1) & has("max_cell_temp", 2)
    U["F_soh"] = ok_rows & has("soh", 1) & has("soh", 2)
    U["G_capacity"] = ok_rows & has("soc", 1) & has("current", 1) & ((f.frac_charge > 0.2) | (f.frac_discharge > 0.2))
    U["exclusion_reason"] = np.select(
        [~ok_rows & (f.is_duplicate_content == True), f.n_rows < minr, f.download_status != "ok"],  # noqa: E712
        ["duplicate_content", "too_few_rows", "download_failed"], "")
    U.to_csv(qdir / "usability_matrix_by_flight.csv", index=False)
    blocks = [c for c in U.columns if c[0] in "ABCDEFG" and c[1] == "_"]
    ub = U.groupby("aircraft_label")[blocks].sum().astype(int)
    ub.insert(0, "flights", U.groupby("aircraft_label").size())
    ub.reset_index().to_csv(qdir / "usability_by_aircraft_and_analysis.csv", index=False)
    bat_rows = []
    for ac, g in f.groupby("aircraft_label"):
        for b in (1, 2):
            for s in ("soc", "soh", "current", "min_cell_volt", "max_cell_volt", "max_cell_temp", "avg_cell_temp"):
                col = f"missing_frac__bat_{b}_{s}"
                usable = ((g[col] < 0.05) & (~g[f"all_zero__bat_{b}_{s}"])).mean()
                bat_rows.append(dict(aircraft=ac, battery=b, signal=s, frac_flights_usable=usable))
    bs = pd.DataFrame(bat_rows)
    bs.to_csv(qdir / "usability_by_battery_signal.csv", index=False)
    f.to_csv(proc / "flights_with_qc.csv", index=False)

    # figures -------------------------------------------------------------------------
    _figures(f, av, init, figs)
    init_trace_figure(settings, f, figs)

    # report ---------------------------------------------------------------------------
    nflights = len(f)
    md = [f"# Data-quality report\n\nGenerated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC. "
          f"{nflights} flights, {int(f.n_rows.sum()):,} samples, {f.aircraft_label.nunique()} aircraft.\n",
          "## Key findings\n", *[f"- {x}" for x in findings], "\n## Download status\n", _md_table(dl),
          "## Usability by aircraft and analysis block (flights usable)\n", _md_table(ub.reset_index()),
          "Blocks: A overview, B flight profiles, C SOC/DoD, D charging, E thermal, F SOH, G degradation-indicator extraction. "
          "Rules are in eplanes/quality/report.py; per-flight reasons in usability_matrix_by_flight.csv.\n",
          "## Timestamps and sampling\n", _md_table(ts),
          "## Label vs content (portal flight type vs current-derived activity)\n", _md_table(xt.reset_index()),
          "## Start-up readings\n", _md_table(tab),
          "![settle](figures/qc_startup_classes.png)\n",
          "## Limitations and unresolved questions\n", *[f"- {x}" for x in limits],
          "\n## Decisions\n- No sample is deleted. Questionable samples are flagged (`quality_flag`) and each analysis states its cleaning rule.\n"
          "- Default rule `settle` masks, per signal group, the data-driven start-up window; alternatives are compared in the sensitivity analysis.\n"
          "- Flights with identical numeric content to an earlier flight are excluded from analyses as duplicates (kept in the data).\n"]
    (qdir / "data_quality_report.md").write_text("\n".join(md))
    gate = dict(passed_utc=datetime.now(timezone.utc).isoformat(), n_flights=nflights, n_samples=int(f.n_rows.sum()),
                findings=findings, limitations=limits)
    (qdir / GATE).write_text(json.dumps(gate, indent=1, default=str))
    log("\n".join(findings))
    return gate


def _figures(f, av, init, figs: FigureSink):
    # signal missingness heatmap (aircraft x signal)
    piv = av.pivot(index="aircraft", columns="signal", values="mean_missing_frac")
    fig, ax = plt.subplots(figsize=(8.2, 0.55 * len(piv) + 1.6))
    im = ax.imshow(piv.values, cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list("s", SEQ), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), [c.replace("_", " ") for c in piv.columns], rotation=45, ha="right")
    ax.set_yticks(range(piv.shape[0]), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.55 else "#0b0b0b")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="mean fraction missing per flight")
    figs.save(fig, "signal_missingness", "Mean fraction of missing samples per flight for each signal and aircraft. "
              "A value of 1 means the signal is never logged for that aircraft (e.g. battery-2 average cell temperature).")

    # coverage heatmap: flights per month x aircraft
    g = f.dropna(subset=["start_datetime"]).copy()
    g["month"] = g.start_datetime.dt.to_period("M").dt.to_timestamp()
    pv = g.pivot_table(index="aircraft_label", columns="month", values="flight_id", aggfunc="count", fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(pv) + 1.6))
    im = ax.imshow(pv.values, cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list("s", SEQ), aspect="auto")
    ax.set_yticks(range(len(pv)), pv.index)
    step = max(1, len(pv.columns) // 12)
    ax.set_xticks(range(0, len(pv.columns), step), [d.strftime("%Y-%m") for d in pv.columns[::step]], rotation=45, ha="right")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="sessions per month")
    figs.save(fig, "coverage_heatmap", "Number of logged sessions (all types) per calendar month and aircraft. Blank (lightest) months have no data.")

    # start-up classes by signal
    ct = init.groupby(["signal_short", "cls"]).size().unstack(fill_value=0)
    ct = ct.div(ct.sum(axis=1), axis=0)
    cols = {"zero_fill": "#e34948", "bin_avg_zero": "#eb6834", "monotone_settling": "#eda100", "irregular_start": "#6250d6",
            "load_onset": "#2a78d6", "stable": "#1baf7a", "not_assessable": "#9b9a94"}
    fig, ax = plt.subplots(figsize=(8, 3.6))
    left = np.zeros(len(ct))
    for c in [c for c in cols if c in ct.columns]:
        ax.barh(ct.index, ct[c], left=left, color=cols[c], label=c.replace("_", " "), edgecolor="#fcfcfb", linewidth=1.2)
        left += ct[c].values
    ax.set_xlabel("share of flights")
    ax.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    figs.save(fig, "startup_classes", "Classification of start-up behaviour per signal (share of flights). zero_fill: first sample exactly 0; "
              "bin_avg_zero: early sample is (k-1)/k of the settled value; load_onset: session starts under load so a physical transient "
              "cannot be excluded and nothing is flagged. Labels state consistency with a pattern, not a proven cause.")

    # settle-time distribution
    s = init[(init.settle_idx > 0)]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for sig, c in (("soc", BLUE), ("min_cell_volt", ORANGE), ("max_cell_temp", "#1baf7a")):
        v = s[s.signal_short == sig].settle_min.dropna()
        if len(v):
            ax.hist(v.clip(upper=6), bins=40, histtype="step", color=c, label=f"{sig.replace('_', ' ')} (n={len(v)})")
    ax.set_xlabel("time until signal settles [min] (clipped at 6)")
    ax.set_ylabel("flights")
    ax.legend()
    figs.save(fig, "settle_time_distribution", "Data-driven settle time (first time after which the signal stays within a noise-scaled "
              "band around its settled value) for flights with a detected start-up window.")

    return None


def init_trace_figure(settings, f, figs: FigureSink, n_examples: int = 3, window_min: float = 4.0):
    """First minutes of representative flights: raw values with flagged rows shaded."""
    from ..analysis.features import read_flight
    cand = f[(f.n_init_unsettled > 0) & (f.n_rows > 200)].sort_values("flight_id")
    if cand.empty:
        cand = f[f.n_rows > 200]
    ids = cand.flight_id.iloc[np.linspace(0, len(cand) - 1, min(n_examples, len(cand))).astype(int)].tolist()
    sigs = [("bat_1_soc", "SOC [%]"), ("bat_1_min_cell_volt", "min cell V [mV]"),
            ("bat_1_max_cell_temp", "max cell T [degC]"), ("bat_1_current", "current [A]")]
    fig, axes = plt.subplots(len(sigs), len(ids), figsize=(3.2 * len(ids), 1.9 * len(sigs)), squeeze=False, sharex="col")
    for j, fid in enumerate(ids):
        d = read_flight(settings, fid)
        d = d[d.time_min <= window_min]
        flagged = (d.init_unsettled_groups > 0).to_numpy()
        for i, (c, lab) in enumerate(sigs):
            ax = axes[i, j]
            ax.plot(d.time_min, d[c], ".-", ms=2.5, lw=0.8, color=BLUE)
            if flagged.any():
                ax.axvspan(d.time_min[flagged].min(), d.time_min[flagged].max(), color=ORANGE, alpha=0.18, lw=0)
            ax.set_ylabel(lab if j == 0 else "")
            if i == 0:
                ax.set_title(f"flight {fid}")
        axes[-1, j].set_xlabel("time [min]")
    figs.save(fig, "startup_example_traces", "First minutes of representative flights (raw values, nothing removed). Orange shading marks the "
              "data-driven start-up window flagged by the quality procedure; row 0 is typically all-zero.")
