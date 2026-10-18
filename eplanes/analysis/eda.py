"""Exploratory analysis sections A-G (paper sections 6-8). Every figure/table is produced from code.

Rules used throughout: the cleaning rule is `settle` unless stated; analyses use only flights marked usable
for that block in the quality stage; the unit of analysis is the flight (x battery), never the sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt

from ..build.clean import iter_flights
from ..config import Settings
from ..quality.report import require_quality_gate
from ..viz.style import (AIRCRAFT_STYLE, AQUA, BATTERY_STYLE, BLUE, GRID, INK2, ORANGE, SEQ, VIOLET,
                         FigureSink, aircraft_color)
from .context import boot_ci, load_context
from .features import apply_rule, read_flight

TYPE_COLOR = {"flight test": BLUE, "charging": AQUA, "ground test": ORANGE}


def _ac_order(df):
    return sorted(df.aircraft_label.dropna().unique())


def _legend_handles(labels):
    from matplotlib.lines import Line2D
    return [Line2D([0], [0], marker=aircraft_color(l, i)[1], color="w", markerfacecolor=aircraft_color(l, i)[0],
                   markersize=7, label=l) for i, l in enumerate(labels)]


class Out:
    def __init__(self, settings: Settings):
        self.t = settings.outputs_dir / "tables"
        self.t.mkdir(parents=True, exist_ok=True)
        self.figs = FigureSink(settings.outputs_dir / "figures")
        self.notes: list[str] = []

    def table(self, df: pd.DataFrame, name: str):
        df.to_csv(self.t / f"{name}.csv", index=False)
        return df

    def note(self, s: str):
        self.notes.append(s)


# ───────────────────────── A. overview ────────────────────────────────────────

def section_A(settings, ctx, out: Out):
    fl = pd.read_csv(settings.data_dir / "processed" / "flights_with_qc.csv", parse_dates=["start_datetime"])
    rows = []
    for ac, g in fl.groupby("aircraft_label"):
        g = g.sort_values("start_datetime")
        gaps = g.start_datetime.diff().dt.total_seconds() / 86400
        rows.append(dict(aircraft=ac, fleet=g.fleet.iloc[0], sessions=len(g),
                         **{f"n_{k.replace(' ', '_')}": int((g.flight_type == k).sum()) for k in
                            ["flight test", "charging", "ground test"]},
                         other_types=int((~g.flight_type.isin(["flight test", "charging", "ground test"])).sum()),
                         first=g.start_datetime.min(), last=g.start_datetime.max(),
                         active_days=int(g.start_datetime.dt.normalize().nunique()),
                         span_days=(g.start_datetime.max() - g.start_datetime.min()).days,
                         hours_logged=g.duration_min.sum() / 60, samples=int(g.n_rows.sum()),
                         median_gap_days=gaps.median(), max_idle_days=gaps.max()))
    ov = out.table(pd.DataFrame(rows), "A1_fleet_overview")
    out.note(f"A: {int(ov.sessions.sum())} sessions, {int(ov.samples.sum()):,} samples over {len(ov)} aircraft; "
             f"{ov['first'].min():%Y-%m-%d} to {ov['last'].max():%Y-%m-%d}.")

    # timeline
    acs = _ac_order(fl)
    fig, ax = plt.subplots(figsize=(8.5, 0.7 * len(acs) + 1.6))
    for i, ac in enumerate(acs):
        g = fl[fl.aircraft_label == ac]
        for k, c in TYPE_COLOR.items():
            gg = g[g.flight_type == k]
            ax.scatter(gg.start_datetime, np.full(len(gg), i), s=14, color=c, alpha=0.8, label=k if i == 0 else None, linewidths=0)
    ax.set_yticks(range(len(acs)), acs)
    ax.set_ylim(-0.6, len(acs) - 0.4)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    ax.set_xlabel("session start (portal date)")
    out.figs.save(fig, "A1_fleet_timeline", "Timeline of logged sessions per aircraft, coloured by portal session type.")

    # monthly counts small multiples
    fl["month"] = fl.start_datetime.dt.to_period("M").dt.to_timestamp()
    fig, axes = plt.subplots(len(acs), 1, figsize=(8.5, 1.5 * len(acs) + 0.8), sharex=True, squeeze=False)
    for ax, ac in zip(axes[:, 0], acs):
        g = fl[fl.aircraft_label == ac].groupby(["month", "flight_type"]).size().unstack(fill_value=0)
        bottom = np.zeros(len(g))
        for k, c in TYPE_COLOR.items():
            if k in g:
                ax.bar(g.index, g[k], width=24, bottom=bottom, color=c, label=k, edgecolor="#fcfcfb", linewidth=0.8)
                bottom += g[k].values
        ax.set_ylabel(ac, rotation=0, ha="right", va="center")
    axes[0, 0].legend(ncol=3, loc="lower right", bbox_to_anchor=(1, 1))
    axes[-1, 0].set_xlabel("month")
    out.figs.save(fig, "A2_sessions_per_month", "Sessions per month by aircraft and session type (stacked).")

    # idle gaps
    fig, ax = plt.subplots(figsize=(6, 3.3))
    for i, ac in enumerate(acs):
        g = fl[fl.aircraft_label == ac].sort_values("start_datetime")
        gaps = (g.start_datetime.diff().dt.total_seconds() / 86400).dropna().sort_values()
        if len(gaps):
            c, m = aircraft_color(ac, i)
            ax.step(gaps, np.arange(1, len(gaps) + 1) / len(gaps), where="post", color=c, label=ac)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("idle time between consecutive sessions [days]")
    ax.set_ylabel("ECDF")
    ax.legend()
    out.figs.save(fig, "A3_idle_gap_ecdf", "Empirical CDF of idle time between consecutive logged sessions for each aircraft (symmetric-log x axis).")

    av = pd.read_csv(settings.outputs_dir / "quality" / "qc_feature_availability.csv")
    out.table(av, "A2_feature_availability")


# ───────────────────────── B. flight profiles ─────────────────────────────────

def phase_proxy(t, cur_dis, med_dt):
    """Current-defined phase proxies (no altitude/speed in the data):
    high-power onset (takeoff/climb proxy) -> plateau (cruise proxy) -> low-power tail (descent/landing proxy)."""
    sm = pd.Series(cur_dis).rolling(max(5, int(0.5 / med_dt)), center=True, min_periods=3).median().to_numpy()
    p95 = np.nanpercentile(sm, 95)
    out = dict(has_pattern=False, t_climb_min=np.nan, t_cruise_min=np.nan, t_tail_min=np.nan)
    if not np.isfinite(p95) or p95 < 10:
        return out
    active = np.where(sm > 0.1 * p95)[0]
    if len(active) < 50:
        return out
    a, b = active[0], active[-1]
    hi = sm > 0.75 * p95
    k = a
    while k < b and hi[k]:
        k += 1
    climb_end = k
    tail = b
    while tail > climb_end and sm[tail] < 0.4 * np.nanmedian(sm[climb_end:b + 1]):
        tail -= 1
    out["t_climb_min"] = (climb_end - a) * med_dt
    out["t_cruise_min"] = max(0, (tail - climb_end)) * med_dt
    out["t_tail_min"] = (b - tail) * med_dt
    out["has_pattern"] = bool(out["t_climb_min"] >= 0.5 and out["t_cruise_min"] >= 2 and 0.1 < sm[climb_end:tail].mean() / p95 < 0.9) if tail > climb_end else False
    return out


def section_B(settings, ctx, out: Out):
    cfg = settings.raw
    ft = ctx[(ctx.flight_type == "flight test") & (ctx.B_profiles == True)].copy()  # noqa: E712
    cols = ["duration_min", "i_peak_discharge_a", "i_mean_abs_a", "ah_discharged", "power_index_peak_w_per_cell",
            "delta_soc", "temp_delta_c"]
    if cfg.get("n_series_cells"):
        cols += ["power_peak_kw", "energy_discharged_kwh"]
    t1 = ft.groupby(["aircraft_label", "battery"])[cols].agg(["count", "median", lambda s: s.quantile(.25), lambda s: s.quantile(.75)])
    t1.columns = [f"{a}__{b if not str(b).startswith('<') else ('q25' if 'q25' not in str(b) else 'q75')}" for a, b in t1.columns]
    t1 = t1.reset_index()
    # readable quartiles
    rows = []
    for (ac, b), g in ft.groupby(["aircraft_label", "battery"]):
        r = dict(aircraft=ac, battery=b, n_flights=len(g))
        for c in cols:
            r[f"{c}_median"] = g[c].median()
            r[f"{c}_q25"] = g[c].quantile(.25)
            r[f"{c}_q75"] = g[c].quantile(.75)
        rows.append(r)
    out.table(pd.DataFrame(rows), "B1_flight_profile_summary")
    out.note(f"B: {ft.flight_id.nunique()} usable flight-test sessions for profile analysis; "
             f"pack power in kW is {'computed' if cfg.get('n_series_cells') else 'NOT computed (n_series_cells unset): current and per-cell power index only'}.")

    # peak current / duration distributions
    acs = _ac_order(ft)
    if acs:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
        for ax, (c, lab) in zip(axes, [("i_peak_discharge_a", "peak discharge current [A]"), ("duration_min", "session duration [min]"),
                                       ("ah_discharged", "discharged charge [Ah]")]):
            data = [ft[(ft.aircraft_label == a) & (ft.battery == 1)][c].dropna() for a in acs]
            bp = ax.boxplot(data, tick_labels=acs, showfliers=False, patch_artist=True, widths=0.55)
            for i, (box, a) in enumerate(zip(bp["boxes"], acs)):
                box.set(facecolor=aircraft_color(a, i)[0], alpha=0.35, edgecolor=INK2)
                ax.scatter(np.random.default_rng(i).normal(i + 1, 0.06, len(data[i])), data[i], s=5, color=aircraft_color(a, i)[0], alpha=0.5, linewidths=0)
            ax.set_ylabel(lab)
            ax.tick_params(axis="x", rotation=30)
        out.figs.save(fig, "B1_flight_distributions", "Flight-test sessions, battery 1: peak discharge current, duration and discharged charge by aircraft "
                      "(box = IQR, line = median, points = individual flights).")

    # battery 1 vs 2 agreement
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
    w = ft.pivot_table(index=["flight_id", "aircraft_label"], columns="battery", values=["i_mean_abs_a", "i_peak_discharge_a"]).dropna().reset_index()
    for ax, c, lab in zip(axes, ["i_mean_abs_a", "i_peak_discharge_a"], ["mean |current| [A]", "peak discharge current [A]"]):
        for i, ac in enumerate(acs):
            g = w[w.aircraft_label == ac]
            col, mk = aircraft_color(ac, i)
            ax.scatter(g[(c, 1)], g[(c, 2)], s=12, color=col, marker=mk, alpha=0.7, label=ac, linewidths=0)
        lim = [w[(c, 1)].min(), w[(c, 1)].max()] if len(w) else [0, 1]
        ax.plot(lim, lim, color=INK2, lw=0.8, ls=":")
        ax.set_xlabel(f"battery 1 {lab}")
        ax.set_ylabel(f"battery 2 {lab}")
    axes[0].legend(fontsize=7)
    out.figs.save(fig, "B2_battery1_vs_battery2", "Per-flight agreement between the two battery packs (dotted line: equality).")

    # representative traces + phase proxies
    ph_rows = []
    path_cache = {}
    ids = ft.drop_duplicates("flight_id")
    ex_ids = []
    for ac in acs:
        g = ids[ids.aircraft_label == ac]
        if len(g):
            ex_ids.append(int(g.iloc[len(g) // 2].flight_id))
    for fid in ids.flight_id.tolist():
        d = apply_rule(read_flight(settings, fid).sort_values("time_min"), "settle")
        t = d.time_min.to_numpy()
        dt = np.diff(t)
        med = float(np.median(dt[dt > 0])) if (dt > 0).any() else 0.02
        cur = -cfg["charge_sign"] * d.bat_1_current.to_numpy()
        ph = phase_proxy(t, cur, med)
        ph.update(flight_id=fid)
        ph_rows.append(ph)
        if fid in ex_ids:
            path_cache[fid] = d
    ph = pd.DataFrame(ph_rows)
    if len(ph):
        ph = ph.merge(ids[["flight_id", "aircraft_label"]], on="flight_id")
        out.table(ph, "B2_phase_proxies")
        frac = ph.has_pattern.mean()
        out.note(f"B: a climb -> plateau -> tail current pattern is identifiable in {frac:.0%} of usable flights "
                 f"(n={len(ph)}); phases are CURRENT-DEFINED PROXIES because altitude/airspeed are not logged.")
    if path_cache:
        n = len(path_cache)
        fig, axes = plt.subplots(4, n, figsize=(3.3 * n, 6.5), sharex="col", squeeze=False)
        for j, (fid, d) in enumerate(path_cache.items()):
            ac = ids[ids.flight_id == fid].aircraft_label.iloc[0]
            for i, (c, lab) in enumerate([("current", "current [A]"), ("cell_volt", "cell V [mV]"), ("soc", "SOC [%]"), ("max_cell_temp", "max cell T [degC]")]):
                ax = axes[i, j]
                for b in (1, 2):
                    col, ls = BATTERY_STYLE[b]
                    if c == "cell_volt":
                        ax.plot(d.time_min, d[f"bat_{b}_min_cell_volt"], color=col, ls=ls, lw=1, label=f"bat {b} min")
                        ax.plot(d.time_min, d[f"bat_{b}_max_cell_volt"], color=col, ls=ls, lw=0.6, alpha=0.6)
                    else:
                        ax.plot(d.time_min, d[f"bat_{b}_{c}"], color=col, ls=ls, lw=1, label=f"bat {b}")
                if j == 0:
                    ax.set_ylabel(lab)
                if i == 0:
                    ax.set_title(f"{ac}, flight {fid}")
            axes[-1, j].set_xlabel("time [min]")
        axes[0, 0].legend(fontsize=7)
        out.figs.save(fig, "B3_representative_flights", "Representative flight-test sessions (one per aircraft, median position in time). Solid blue = battery 1, "
                      "dashed orange = battery 2. Start-up samples flagged by the quality stage are masked (gaps in lines).")


# ───────────────────────── C. SOC / depth of discharge ────────────────────────

def section_C(settings, ctx, out: Out):
    cfg = settings.raw["analysis"]
    ft = ctx[(ctx.flight_type == "flight test") & (ctx.C_soc_dod == True) & (ctx.frac_discharge > 0.05)].copy()  # noqa: E712
    ft["dod"] = ft.soc_start - ft.soc_end
    ft["dod_class"] = pd.cut(ft.dod, [-1e9, cfg["partial_dod"], cfg["near_full_dod"], 1e9],
                             labels=["shallow (<%d)" % cfg["partial_dod"], "intermediate", "deep (>=%d)" % cfg["near_full_dod"]])
    rows = []
    for (ac, b), g in ft.groupby(["aircraft_label", "battery"]):
        lo, hi = boot_ci(g.dod, np.median, 1000)
        rows.append(dict(aircraft=ac, battery=b, n=len(g), soc_start_median=g.soc_start.median(), soc_end_median=g.soc_end.median(),
                         dod_median=g.dod.median(), dod_ci_low=lo, dod_ci_high=hi, dod_q25=g.dod.quantile(.25), dod_q75=g.dod.quantile(.75),
                         frac_deep=(g.dod_class.astype(str).str.startswith("deep")).mean(),
                         frac_shallow=(g.dod_class.astype(str).str.startswith("shallow")).mean(),
                         frac_start_ge95=(g.soc_start >= 95).mean(), frac_end_le20=(g.soc_end <= 20).mean()))
    out.table(pd.DataFrame(rows), "C1_soc_dod_summary")
    out.table(ft[["flight_id", "aircraft_label", "battery", "soc_start", "soc_end", "dod", "dod_class"]], "C1_flight_level_dod")
    out.note(f"C: {ft.flight_id.nunique()} usable discharging flight-test sessions; median DoD (bat 1) = {ft[ft.battery == 1].dod.median():.1f} SOC points.")
    acs = _ac_order(ft)
    if not acs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    for i, ac in enumerate(acs):
        g = ft[(ft.aircraft_label == ac) & (ft.battery == 1)]
        col, _ = aircraft_color(ac, i)
        for ax, c in zip(axes[:2], ["soc_start", "soc_end"]):
            v = np.sort(g[c].dropna())
            ax.step(v, np.arange(1, len(v) + 1) / max(len(v), 1), where="post", color=col, label=ac)
        v = np.sort(g.dod.dropna())
        axes[2].step(v, np.arange(1, len(v) + 1) / max(len(v), 1), where="post", color=col, label=ac)
    for ax, lab in zip(axes, ["SOC at start [%]", "SOC at end [%]", "depth of discharge [SOC points]"]):
        ax.set_xlabel(lab)
        ax.set_ylabel("ECDF")
    axes[0].legend(fontsize=7)
    out.figs.save(fig, "C1_soc_ecdfs", "Empirical CDFs of start SOC, end SOC and depth of discharge for flight-test sessions (battery 1), by aircraft. "
                  "SOC is the BMS-reported, integer-quantised value taken after the start-up window.")

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    cc = ft[ft.battery == 1].groupby(["aircraft_label", "dod_class"], observed=False).size().unstack(fill_value=0)
    cc = cc.div(cc.sum(axis=1), axis=0)
    left = np.zeros(len(cc))
    for c, color in zip(cc.columns, [SEQ[1], SEQ[3], SEQ[5]]):
        ax.barh(cc.index, cc[c], left=left, color=color, edgecolor="#fcfcfb", linewidth=1.2, label=c)
        left += cc[c].values
    ax.set_xlabel("share of flights")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    out.figs.save(fig, "C2_dod_classes", "Share of flight-test sessions in shallow, intermediate and deep depth-of-discharge classes "
                  "(thresholds in config.yaml), battery 1.")

    # battery agreement in delta SOC
    w = ft.pivot_table(index=["flight_id", "aircraft_label"], columns="battery", values="dod").dropna().reset_index()
    if len(w):
        fig, ax = plt.subplots(figsize=(4.2, 4))
        for i, ac in enumerate(acs):
            g = w[w.aircraft_label == ac]
            col, mk = aircraft_color(ac, i)
            ax.scatter(g[1], g[2], s=12, color=col, marker=mk, alpha=0.7, label=ac, linewidths=0)
        lim = [0, max(w[1].max(), w[2].max())]
        ax.plot(lim, lim, ":", color=INK2, lw=0.8)
        ax.set_xlabel("battery 1 DoD [SOC points]")
        ax.set_ylabel("battery 2 DoD [SOC points]")
        ax.legend(fontsize=7)
        out.figs.save(fig, "C3_dod_pack_agreement", "Depth of discharge of battery 2 versus battery 1 per flight (dotted line: equality).")


# ───────────────────────── D. charging ───────────────────────────────────────

def section_D(settings, ctx, out: Out):
    cfg = settings.raw
    ch = ctx[(ctx.D_charging == True) & (ctx.charge_time_min.notna()) & (ctx.charge_time_min > 3)].copy()  # noqa: E712
    ch["delta_soc_chg"] = ch.charge_soc_end - ch.charge_soc_start
    ch["charge_class"] = np.where(ch.charge_soc_end >= 95, "to full (end>=95)", "partial")
    rows = []
    for (ac, b), g in ch.groupby(["aircraft_label", "battery"]):
        rows.append(dict(aircraft=ac, battery=b, n_sessions=len(g), duration_min_median=g.charge_time_min.median(),
                         soc_start_median=g.charge_soc_start.median(), soc_end_median=g.charge_soc_end.median(),
                         mean_current_a_median=g.charge_i_mean_a.median(), peak_current_a_median=g.charge_i_peak_a.median(),
                         ah_charged_median=g.ah_charged.median(), frac_partial=(g.charge_class == "partial").mean(),
                         frac_cc_plateau=g.chg_has_cc.mean(), frac_taper=g.chg_has_taper.mean(), frac_ramp_like=g.chg_ramp_like.mean()))
    d1 = out.table(pd.DataFrame(rows), "D1_charging_summary")
    out.note(f"D: {ch.flight_id.nunique()} usable charging sessions; CC plateau (within 1% of p90 current for >= max(5 min, 15% of the segment)) in "
             f"{ch[ch.battery == 1].chg_has_cc.mean():.0%} and CV-like taper in {ch[ch.battery == 1].chg_has_taper.mean():.0%} of battery-1 sessions. "
             "Phase identification is therefore " + ("supported" if ch[ch.battery == 1].chg_has_taper.mean() > 0.5 else "NOT supported for most sessions (negative result)") + ".")
    acs = _ac_order(ch)
    if not acs:
        return

    # profiles: current vs SOC, max cell V vs SOC for sampled sessions
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    rng = np.random.default_rng(0)
    for i, ac in enumerate(acs):
        col, _ = aircraft_color(ac, i)
        ids = ch[(ch.aircraft_label == ac) & (ch.battery == 1)].flight_id.tolist()
        ids = list(rng.choice(ids, min(8, len(ids)), replace=False)) if ids else []
        for fid in ids:
            d = apply_rule(read_flight(settings, int(fid)).sort_values("time_min"), "settle")
            m = d.segment_type == "charge"
            if m.sum() < 50:
                continue
            axes[0].plot(d.time_min[m] - d.time_min[m].iloc[0], d.bat_1_current[m].abs(), color=col, lw=0.7, alpha=0.6)
            axes[1].plot(d.bat_1_soc[m], d.bat_1_current[m].abs(), color=col, lw=0.7, alpha=0.6)
            axes[2].plot(d.bat_1_soc[m], d.bat_1_max_cell_volt[m], color=col, lw=0.7, alpha=0.6)
        axes[0].plot([], [], color=col, label=ac)
    axes[0].set_xlabel("time in charge segment [min]"); axes[0].set_ylabel("|current| [A]")
    axes[1].set_xlabel("SOC [%]"); axes[1].set_ylabel("|current| [A]")
    axes[2].set_xlabel("SOC [%]"); axes[2].set_ylabel("max cell voltage [mV]")
    axes[0].legend(fontsize=7)
    fig.subplots_adjust(wspace=0.4)
    out.figs.save(fig, "D1_charging_profiles", "Charging profiles of up to 8 randomly chosen charging sessions per aircraft (battery 1): current versus time, "
                  "current versus SOC and maximum cell voltage versus SOC. A clean constant-current plateau followed by a decaying taper would be visible as "
                  "a flat then falling current trace; its presence or absence is quantified in table D1.")

    # duration vs delta SOC & start/end SOC
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.3))
    for i, ac in enumerate(acs):
        g = ch[(ch.aircraft_label == ac) & (ch.battery == 1)]
        col, mk = aircraft_color(ac, i)
        axes[0].scatter(g.delta_soc_chg, g.charge_time_min, s=12, color=col, marker=mk, alpha=0.7, label=ac, linewidths=0)
        axes[1].scatter(g.charge_soc_start, g.charge_soc_end, s=12, color=col, marker=mk, alpha=0.7, linewidths=0)
    axes[0].set_xlabel("SOC gained in charge segment [points]"); axes[0].set_ylabel("charging duration [min]")
    axes[1].set_xlabel("SOC at start of charging [%]"); axes[1].set_ylabel("SOC at end of charging [%]")
    axes[0].legend(fontsize=7)
    out.figs.save(fig, "D2_charging_duration_soc", "Charging duration versus SOC gained, and start versus end SOC of charging segments (battery 1).")

    # rest periods charge -> next session
    fl = pd.read_csv(settings.data_dir / "processed" / "flights_with_qc.csv", parse_dates=["start_datetime"]).sort_values(["aircraft_label", "start_datetime"])
    fl["end_dt"] = fl.start_datetime + pd.to_timedelta(fl.duration_min.fillna(0), unit="m")
    fl["next_type"] = fl.groupby("aircraft_label").flight_type.shift(-1)
    fl["next_start"] = fl.groupby("aircraft_label").start_datetime.shift(-1)
    fl["rest_h"] = (fl.next_start - fl.end_dt).dt.total_seconds() / 3600
    rest = fl[(fl.flight_type == "charging") & fl.next_type.notna() & (fl.rest_h >= 0)][["flight_id", "aircraft_label", "next_type", "rest_h"]]
    out.table(rest, "D2_rest_after_charging")
    c2f = rest[rest.next_type == "flight test"]
    if len(c2f):
        fig, ax = plt.subplots(figsize=(5.5, 3.3))
        for i, ac in enumerate(acs):
            v = np.sort(c2f[c2f.aircraft_label == ac].rest_h)
            if len(v):
                ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=aircraft_color(ac, i)[0], label=ac)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel("rest between end of charging and next flight [h]")
        ax.set_ylabel("ECDF")
        ax.legend(fontsize=7)
        out.figs.save(fig, "D3_rest_charge_to_flight", "Rest time between the end of a charging session and the start of the next flight-test session. "
                      "Session start times are logged to the minute at best and end times are derived from the recorded duration, so short rests are approximate.")


# ───────────────────────── E. thermal ────────────────────────────────────────

def section_E(settings, ctx, out: Out):
    th = ctx[ctx.E_thermal == True].copy()  # noqa: E712
    rows = []
    for (ac, b, ty), g in th.groupby(["aircraft_label", "battery", "flight_type"]):
        rows.append(dict(aircraft=ac, battery=b, flight_type=ty, n=len(g), temp_start_median=g.temp_start_c.median(),
                         temp_max_median=g.temp_max_c.median(), temp_max_p95=g.temp_max_c.quantile(.95), temp_max_max=g.temp_max_c.max(),
                         temp_rise_median=g.temp_delta_c.median()))
    out.table(pd.DataFrame(rows), "E1_thermal_summary")
    out.note("E: ambient temperature is NOT logged. Only cell temperatures are available: max cell temperature (both packs) and average "
             "cell temperature (battery 1 only). The first valid cell temperature of a session is used as a start-of-session proxy, NOT as ambient.")
    acs = _ac_order(th)
    if not acs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for i, ac in enumerate(acs):
        col, _ = aircraft_color(ac, i)
        g = th[(th.aircraft_label == ac) & (th.battery == 1)]
        for ax, c in zip(axes[:2], ["temp_max_c", "temp_delta_c"]):
            v = np.sort(g[c].dropna())
            ax.step(v, np.arange(1, len(v) + 1) / max(len(v), 1), where="post", color=col, label=ac)
    ft = th[(th.battery == 1) & (th.flight_type == "flight test")]
    for i, ac in enumerate(acs):
        g = ft[ft.aircraft_label == ac]
        col, mk = aircraft_color(ac, i)
        axes[2].scatter(g.i_mean_abs_a, g.temp_delta_c, s=12, color=col, marker=mk, alpha=0.7, linewidths=0)
    axes[0].set_xlabel("max cell temperature [degC]"); axes[1].set_xlabel("temperature rise within session [degC]")
    axes[2].set_xlabel("mean |current| [A] (flight tests)"); axes[2].set_ylabel("temperature rise [degC]")
    axes[0].set_ylabel("ECDF"); axes[1].set_ylabel("ECDF")
    axes[0].legend(fontsize=7)
    out.figs.save(fig, "E1_temperature_distributions", "Max-cell-temperature and within-session temperature-rise distributions by aircraft (battery 1, all session types) "
                  "and temperature rise versus mean current for flight tests.")

    seasons = ["winter", "spring", "summer", "autumn"]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    w = 0.8 / len(acs)
    for i, ac in enumerate(acs):
        g = th[(th.aircraft_label == ac) & (th.battery == 1)]
        data = [g[g.season == s].temp_max_c.dropna() for s in seasons]
        pos = np.arange(len(seasons)) + i * w - 0.4 + w / 2
        bp = ax.boxplot([d if len(d) else [np.nan] for d in data], positions=pos, widths=w * 0.85, showfliers=False, patch_artist=True)
        for b in bp["boxes"]:
            b.set(facecolor=aircraft_color(ac, i)[0], alpha=0.45, edgecolor=INK2)
        ax.plot([], [], color=aircraft_color(ac, i)[0], lw=6, alpha=0.6, label=ac)
    ax.set_xticks(range(len(seasons)), seasons)
    ax.set_ylabel("max cell temperature [degC]")
    ax.legend(ncol=len(acs), fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    out.figs.save(fig, "E2_seasonal_temperature", "Maximum cell temperature per session by meteorological season (Dec-Feb winter) and aircraft, battery 1. "
                  "Seasons are not balanced across aircraft; empty boxes mean no data.")

    # pack comparison
    w2 = th.pivot_table(index=["flight_id", "aircraft_label"], columns="battery", values="temp_max_c").dropna().reset_index()
    if len(w2):
        fig, ax = plt.subplots(figsize=(4.2, 4))
        for i, ac in enumerate(acs):
            g = w2[w2.aircraft_label == ac]
            col, mk = aircraft_color(ac, i)
            ax.scatter(g[1], g[2], s=12, color=col, marker=mk, alpha=0.7, label=ac, linewidths=0)
        lim = [w2[[1, 2]].min().min(), w2[[1, 2]].max().max()]
        ax.plot(lim, lim, ":", color=INK2, lw=0.8)
        ax.set_xlabel("battery 1 max cell temp [degC]"); ax.set_ylabel("battery 2 max cell temp [degC]")
        ax.legend(fontsize=7)
        out.figs.save(fig, "E3_pack_temperature", "Per-session maximum cell temperature of battery 2 versus battery 1 (dotted line: equality).")


# ───────────────────────── F. BMS SOH ────────────────────────────────────────

def soh_series(ctx):
    d = ctx[(ctx.F_soh == True) & ctx.soh_median.notna() & ctx.start_datetime.notna()].copy()  # noqa: E712
    d = d.sort_values(["aircraft_label", "battery", "start_datetime", "flight_id"])
    d["cum_ah"] = d.groupby(["aircraft_label", "battery"]).ah_throughput.cumsum()
    return d


def section_F(settings, ctx, out: Out):
    d = soh_series(ctx)
    rows, jumps = [], []
    for (ac, b), g in d.groupby(["aircraft_label", "battery"]):
        s = g.soh_median.to_numpy()
        if len(s) < 3:
            continue
        ds = np.diff(s)
        days = g.start_datetime.diff().dt.total_seconds().to_numpy()[1:] / 86400
        mad = 1.4826 * np.median(np.abs(ds - np.median(ds))) or 0.5
        big = (np.abs(ds) >= max(2.0, 4 * mad))
        for k in np.where(big)[0]:
            jumps.append(dict(aircraft=ac, battery=b, flight_id=int(g.flight_id.iloc[k + 1]), date=g.start_datetime.iloc[k + 1],
                              delta_soh=float(ds[k]), days_since_previous=float(days[k])))
        run = best = 1
        for k in range(1, len(s)):
            run = run + 1 if s[k] == s[k - 1] else 1
            best = max(best, run)
        tau, p = stats.kendalltau(g.start_datetime.map(pd.Timestamp.toordinal), s) if len(s) > 5 else (np.nan, np.nan)
        rows.append(dict(aircraft=ac, battery=b, n_sessions=len(s), first_soh=s[0], last_soh=s[-1], total_change=s[-1] - s[0],
                         min_soh=s.min(), max_soh=s.max(), n_increases=int((ds > 0).sum()), n_decreases=int((ds < 0).sum()),
                         frac_steps_increasing=float((ds > 0).mean()), n_large_jumps=int(big.sum()), max_abs_step=float(np.abs(ds).max()),
                         longest_constant_run_sessions=best, kendall_tau_vs_date=tau, kendall_p=p,
                         monotone_nonincreasing=bool((ds <= 0).all()),
                         span_days=(g.start_datetime.max() - g.start_datetime.min()).days,
                         within_session_soh_range_median=float((g.soh_end - g.soh_start).abs().median())))
    sm = out.table(pd.DataFrame(rows), "F1_soh_trajectory_summary")
    out.table(pd.DataFrame(jumps), "F2_soh_large_jumps")
    if len(sm):
        out.note(f"F: BMS-reported SOH is non-monotone in {int((~sm.monotone_nonincreasing).sum())}/{len(sm)} battery series; "
                 f"{int(sm.n_large_jumps.sum())} large session-to-session jumps (>=2 points and >=4 robust sigma). "
                 "BMS SOH is a reported number, not a measured capacity; changes are NOT interpreted as degradation here.")
    acs = _ac_order(d)
    if not acs:
        return
    fig, axes = plt.subplots(len(acs), 1, figsize=(9, 2.3 * len(acs) + 0.6), sharex=True, squeeze=False)
    for ax, ac in zip(axes[:, 0], acs):
        for b in (1, 2):
            g = d[(d.aircraft_label == ac) & (d.battery == b)]
            col, ls = BATTERY_STYLE[b]
            ax.plot(g.start_datetime, g.soh_median, ls, color=col, lw=0.8, marker="o", ms=2.5, label=f"battery {b}")
        ax.set_ylabel(f"{ac}\nBMS SOH [%]")
    axes[0, 0].legend(ncol=2, loc="lower right", bbox_to_anchor=(1, 1))
    axes[-1, 0].set_xlabel("session date")
    out.figs.save(fig, "F1_soh_trajectories", "BMS-reported SOH (session median after the start-up window) versus session date, per aircraft and battery pack. "
                  "Straight segments across long data gaps are interpolation between sessions only.")

    fig, axes = plt.subplots(1, len(acs), figsize=(3.1 * len(acs), 3.2), sharey=True, squeeze=False)
    for ax, ac in zip(axes[0], acs):
        for b in (1, 2):
            g = d[(d.aircraft_label == ac) & (d.battery == b)]
            col, ls = BATTERY_STYLE[b]
            ax.plot(g.cum_ah, g.soh_median, ls, color=col, lw=0.8, marker="o", ms=2.2, label=f"battery {b}")
        ax.set_title(ac)
        ax.set_xlabel("cumulative |Ah| logged")
    axes[0, 0].set_ylabel("BMS SOH [%]")
    axes[0, 0].legend(fontsize=7)
    out.figs.save(fig, "F2_soh_vs_throughput", "BMS-reported SOH versus cumulative charge throughput integrated from logged current (only logged sessions are counted, "
                  "so throughput is a lower bound).")

    steps = []
    for (ac, b), g in d.groupby(["aircraft_label", "battery"]):
        steps.append(pd.DataFrame({"aircraft": ac, "battery": b, "step": np.diff(g.soh_median.to_numpy())}))
    if steps:
        st = pd.concat(steps)
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.hist(st.step.clip(-15, 15), bins=np.arange(-15.5, 16.5, 1), color=BLUE, edgecolor="#fcfcfb")
        ax.set_yscale("symlog")
        ax.set_xlabel("session-to-session change in reported SOH [points] (clipped at +/-15)")
        ax.set_ylabel("count")
        out.figs.save(fig, "F3_soh_step_sizes", "Distribution of session-to-session changes in BMS-reported SOH (all batteries). Positive steps are physically "
                      "implausible as degradation and indicate re-calibration, estimator resets or reporting artefacts.")

    g2 = d.pivot_table(index=["aircraft_label", "flight_id", "start_datetime"], columns="battery", values="soh_median").dropna().reset_index()
    if len(g2):
        fig, ax = plt.subplots(figsize=(7, 3.2))
        for i, ac in enumerate(acs):
            g = g2[g2.aircraft_label == ac]
            col, mk = aircraft_color(ac, i)
            ax.scatter(g.start_datetime, g[1] - g[2], s=10, color=col, marker=mk, alpha=0.7, label=ac, linewidths=0)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_ylabel("SOH(battery 1) - SOH(battery 2) [points]")
        ax.legend(fontsize=7, ncol=len(acs))
        out.figs.save(fig, "F4_soh_pack_gap", "Difference between the two packs' reported SOH over time.")


# ───────────────────────── G. observability of degradation indicators ────────

def grade(cov, cv, r2=None):
    if cov < 0.05:
        return "not recoverable"
    score = 0
    score += cov >= 0.3
    score += (cv is not None and np.isfinite(cv) and cv < 0.15)
    if r2 is not None:
        score += (np.isfinite(r2) and r2 > 0.8)
    return {0: "unreliable", 1: "weak", 2: "moderate", 3: "good"}[min(int(score), 3)]


def section_G(settings, ctx, out: Out):
    cfg = settings.raw
    rows = []
    usable = ctx[ctx.G_capacity == True]  # noqa: E712
    n_sess = ctx[ctx.G_capacity == True].flight_id.nunique()

    # 1 coulomb counting consistency: delta SOC vs net Ah
    cc_rows = []
    for (ac, b), g in usable.groupby(["aircraft_label", "battery"]):
        g = g[(g.delta_soc.abs() >= 10) & (g.ah_net_discharge.abs() > 0.5)]
        if len(g) >= 8:
            sl = stats.linregress(-g.ah_net_discharge, g.delta_soc)  # soc gain per Ah charged
            cap = 100.0 / sl.slope if sl.slope > 0 else np.nan
            q = (-g.ah_net_discharge / (g.delta_soc / 100.0)).replace([np.inf, -np.inf], np.nan)
            q = q[(q > 0)]
            cc_rows.append(dict(aircraft=ac, battery=b, n=len(g), slope_soc_per_ah=sl.slope, r2=sl.rvalue ** 2,
                                implied_capacity_ah_from_slope=cap, per_session_implied_median=q.median(),
                                per_session_cv=q.std() / q.mean() if q.mean() else np.nan))
    cc = out.table(pd.DataFrame(cc_rows), "G1_soc_vs_integrated_current")
    # 2 matched-window capacity
    cap = usable[usable.capacity_reason == "ok"]
    cap_rows = []
    for (ac, b), g in usable.groupby(["aircraft_label", "battery"]):
        ok = g[g.capacity_reason == "ok"]
        tau = stats.spearmanr(ok.start_datetime.map(pd.Timestamp.toordinal), ok.implied_capacity_ah)[0] if len(ok) > 5 else np.nan
        cap_rows.append(dict(aircraft=ac, battery=b, sessions_tried=len(g), n_ok=len(ok), coverage=len(ok) / max(len(g), 1),
                             median_ah=ok.implied_capacity_ah.median(), iqr_ah=ok.implied_capacity_ah.quantile(.75) - ok.implied_capacity_ah.quantile(.25),
                             cv=ok.implied_capacity_ah.std() / ok.implied_capacity_ah.mean() if len(ok) > 2 else np.nan, spearman_vs_date=tau,
                             window=str(cfg["analysis"]["soc_window"])))
    cp = out.table(pd.DataFrame(cap_rows), "G2_window_capacity")
    reasons = usable.capacity_reason.str.replace(r"\d+pct", "NNpct", regex=True).value_counts().rename_axis("reason").reset_index(name="count")
    out.table(reasons, "G2b_capacity_rejection_reasons")
    # 4 R0 proxy
    r0 = usable[usable.r0_n_windows > 0]
    r0_rows = []
    for (ac, b), g in usable.groupby(["aircraft_label", "battery"]):
        ok = g[g.r0_n_windows > 0]
        rho = stats.spearmanr(ok.temp_start_c, ok.r0_proxy_mohm_cell, nan_policy="omit")[0] if len(ok) > 5 else np.nan
        r0_rows.append(dict(aircraft=ac, battery=b, sessions=len(g), n_ok=len(ok), coverage=len(ok) / max(len(g), 1), median_mohm_per_cell=ok.r0_proxy_mohm_cell.median(),
                            cv=ok.r0_proxy_mohm_cell.std() / ok.r0_proxy_mohm_cell.mean() if len(ok) > 2 else np.nan, spearman_vs_start_temp=rho))
    rr = out.table(pd.DataFrame(r0_rows), "G3_resistance_proxy")
    # 5/6 charging taper & dQ/dV
    chg = ctx[(ctx.D_charging == True) & (ctx.charge_time_min.fillna(0) > 3)]
    taper_cov = chg.chg_has_taper.mean() if len(chg) else 0.0
    cc_cov = chg.chg_has_cc.mean() if len(chg) else 0.0
    dq_cov = chg.dqdv_usable.fillna(False).astype(bool).mean() if len(chg) else 0.0

    def agg(df, col, f=np.nanmedian):
        return float(f(df[col])) if len(df) and col in df else np.nan

    thr_cov = float(cap.flight_id.nunique() / max(n_sess, 1))
    ind = [
        dict(indicator="Coulomb-counted charge (Ah)", signals="current, time", method="trapezoid integration, gaps and NaNs skipped",
             assumptions="current sensor unbiased; no logging outside sessions", coverage=float(usable.ah_throughput.gt(0).mean()),
             spread_cv=np.nan, consistency=agg(cc, "r2"), grade=grade(float(usable.ah_throughput.gt(0).mean()), 0.0, agg(cc, "r2")),
             limitation="Only logged sessions; offset/drift of current sensor unknown; integration across gaps is excluded."),
        dict(indicator="SOC vs integrated-current consistency (implied capacity from slope)", signals="SOC, current", method="regression of dSOC on net Ah (|dSOC|>=10)",
             assumptions="BMS SOC is coulomb-based and capacity is constant over the sample", coverage=float(len(cc) / max(ctx.groupby(['aircraft_label', 'battery']).ngroups, 1)),
             spread_cv=agg(cc, "per_session_cv"), consistency=agg(cc, "r2"), grade=grade(1.0 if len(cc) else 0.0, agg(cc, "per_session_cv"), agg(cc, "r2")),
             limitation="SOC is integer-quantised and may be re-calibrated by the BMS; implied capacity mixes true capacity with SOC-estimator behaviour."),
        dict(indicator="Capacity over matched SOC window", signals="SOC, current, time", method=f"single monotone pass within SOC {cfg['analysis']['soc_window']}, scaled to full range",
             assumptions="one monotone pass; SOC span covers >=%.0f%% of window" % (100 * cfg["analysis"]["min_window_coverage"]),
             coverage=thr_cov, spread_cv=agg(cp, "cv"), consistency=np.nan, grade=grade(thr_cov, agg(cp, "cv")),
             limitation="Few sessions span the window; multi-pass sessions rejected to avoid double counting; depends on SOC accuracy."),
        dict(indicator="Energy throughput", signals="current (+ pack voltage for Wh)", method="Ah always; kWh only if series cell count known",
             assumptions="pack voltage reconstructable from cell voltage x series count", coverage=1.0 if usable.ah_throughput.notna().any() else 0.0, spread_cv=np.nan,
             consistency=np.nan, grade="good (Ah) / not available (kWh)" if not cfg.get("n_series_cells") else "good",
             limitation="Pack voltage is not logged; set n_series_cells in config.yaml after verifying pack topology."),
        dict(indicator="DC internal resistance proxy", signals="cell voltage, current, SOC", method="windowed dV/dI (60 s, std(I)>=2 A, dSOC<=1, R2>=0.5)",
             assumptions="load steps/variation exist; temperature known; OCV drift small in window", coverage=float(rr.coverage.mean()) if len(rr) else 0.0,
             spread_cv=agg(rr, "cv"), consistency=np.nan, grade=grade(float(rr.coverage.mean()) if len(rr) else 0.0, agg(rr, "cv")),
             limitation="Only min/max cell voltage logged (no pack voltage); per-cell value is a proxy; strongly temperature dependent; smooth charger ramps give weak current variation."),
        dict(indicator="Charging taper / CC-CV phases", signals="current, max cell voltage", method="plateau (within 1% of p90 for >= max(5 min, 15% of segment)) then monotone decay below 50% with cell voltage near its maximum",
             assumptions="charger uses CC-CV", coverage=float(taper_cov), spread_cv=np.nan, consistency=float(cc_cov), grade=grade(float(taper_cov), None) if taper_cov >= 0.05 else "not recoverable",
             limitation=f"CC plateau found in {cc_cov:.0%} and taper in {taper_cov:.0%} of charging sessions; observed profiles are ramp-like."),
        dict(indicator="dQ/dV (incremental capacity)", signals="current, cell voltage, SOC", method="Q from integrated current vs cell voltage during near-constant current charging",
             assumptions="low constant rate, >=30 SOC points, >=20 voltage bins", coverage=float(dq_cov), spread_cv=np.nan, consistency=np.nan,
             grade="not recoverable" if dq_cov < 0.05 else grade(float(dq_cov), None),
             limitation="Requires near-constant low-rate charge; only min/max cell voltage (not pack OCV); integer SOC."),
        dict(indicator="Calendar-time and utilisation indicators", signals="session dates, current", method="elapsed days, session counts, cumulative |Ah|, flights per month",
             assumptions="logged sessions are the complete history", coverage=1.0, spread_cv=np.nan, consistency=np.nan, grade="good (but confounded with each other)",
             limitation="Calendar time and throughput are collinear; unlogged use is invisible."),
    ]
    obs = out.table(pd.DataFrame(ind), "G0_observability_table")
    out.note("G: observability grades -> " + "; ".join(f"{r.indicator}: {r.grade}" for r in obs.itertuples()))

    # consistency figure
    if len(usable):
        acs = _ac_order(usable)
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
        for ax, b in zip(axes, (1, 2)):
            g = usable[(usable.battery == b) & (usable.delta_soc.abs() >= 5)]
            for i, ac in enumerate(acs):
                gg = g[g.aircraft_label == ac]
                col, mk = aircraft_color(ac, i)
                ax.scatter(-gg.ah_net_discharge, gg.delta_soc, s=12, color=col, marker=mk, alpha=0.7, label=ac, linewidths=0)
            ax.axhline(0, color=INK2, lw=0.6); ax.axvline(0, color=INK2, lw=0.6)
            ax.set_xlabel("net charge in [Ah] (integrated current)")
            ax.set_title(f"battery {b}")
        axes[0].set_ylabel("change in reported SOC [points]")
        axes[0].legend(fontsize=7)
        out.figs.save(fig, "G1_soc_vs_integrated_current", "Change in BMS-reported SOC versus net charge integrated from logged current, per session and pack. "
                      "If SOC were a pure coulomb counter with fixed capacity Q, points would fall on a line of slope 100/Q through the origin; scatter shows how far "
                      "reported SOC departs from current integration.")

        fig, ax = plt.subplots(figsize=(7, 3.3))
        for i, ac in enumerate(acs):
            g = cap[(cap.aircraft_label == ac) & (cap.battery == 1)]
            col, mk = aircraft_color(ac, i)
            ax.scatter(g.start_datetime, g.implied_capacity_ah, s=14, color=col, marker=mk, alpha=0.8, label=ac, linewidths=0)
        ax.set_ylabel("matched-window capacity [Ah]")
        ax.set_xlabel("session date")
        ax.legend(fontsize=7)
        out.figs.save(fig, "G2_window_capacity_vs_time", "Capacity estimated from single monotone passes inside the SOC window (battery 1). "
                      "Estimates are shown without smoothing; sparse coverage limits trend inference.")


def run_eda(settings: Settings, rule="settle", log=print):
    require_quality_gate(settings)
    ctx = load_context(settings, rule, recompute=True)
    out = Out(settings)
    for name, fn in (("A", section_A), ("B", section_B), ("C", section_C), ("D", section_D),
                     ("E", section_E), ("F", section_F), ("G", section_G)):
        log(f"[eda] section {name}")
        fn(settings, ctx, out)
    (settings.outputs_dir / "analysis_summary.md").write_text("# Analysis summary (auto-generated from results)\n\n" + "\n\n".join(f"- {n}" for n in out.notes) + "\n")
    return out
