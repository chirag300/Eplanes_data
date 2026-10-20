"""Figures and tables for the dataset paper, generated from the released data.

usage:
    python paper/make_figures.py --bundle outputs/web_bundle --results <unzipped results archive> --out paper

Inputs
    bundle   : output of `python -m eplanes export-web` (sessions, packs, 2 s traces, SOH steps, quality)
    results  : output of `python -m eplanes all` (tables/ and quality/ from the analysis run), optional
Outputs
    <out>/figures/*.pdf (+ .png previews) and <out>/tables/*.tex (booktabs bodies) and <out>/numbers.json
    (every number quoted in the paper text, so the manuscript can be checked against the data).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the eplanes package, when run as a script from the repo

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

INK, INK2, INK3, GRID = "#16212b", "#4a5763", "#7c8792", "#e3e6e9"
AC = ["C-GAUW", "C-GMUW", "VSW1280102", "VSW1280107"]
AC_COL = {"C-GAUW": "#2a78d6", "C-GMUW": "#eb6834", "VSW1280102": "#1baf7a", "VSW1280107": "#7d8996"}
AC_MK = {"C-GAUW": "o", "C-GMUW": "s", "VSW1280102": "^", "VSW1280107": "D"}
PACK = {1: ("#2a78d6", "-"), 2: ("#eb6834", "--")}
TYPE_COL = {"flight": "#2a78d6", "charge": "#1baf7a", "other": "#7d8996"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.labelsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 7, "lines.linewidth": 1.2, "savefig.dpi": 300, "figure.dpi": 120,
    "axes.titlelocation": "left", "pdf.fonttype": 42, "ps.fonttype": 42,
})
W1, W2 = 3.45, 7.1  # single / double column width (inches)


class Out:
    def __init__(self, root: Path):
        self.fig = root / "figures"
        self.tab = root / "tables"
        self.fig.mkdir(parents=True, exist_ok=True)
        self.tab.mkdir(parents=True, exist_ok=True)
        self.numbers = {}

    def save(self, fig, name):
        fig.savefig(self.fig / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(self.fig / f"{name}.png", bbox_inches="tight", dpi=170)
        plt.close(fig)

    def table(self, name, header, rows, align):
        lines = [r"\begin{tabular}{" + align + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
        for r in rows:
            lines.append(r"\midrule" if r == "MID" else " & ".join(str(c) for c in r) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (self.tab / f"{name}.tex").write_text("\n".join(lines) + "\n")

    def num(self, key, value):
        self.numbers[key] = value if not isinstance(value, (np.floating, np.integer)) else value.item()
        return value


def f(x, d=1, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    s = f"{x:+,.{d}f}" if sign else f"{x:,.{d}f}"
    return s.replace("-", "$-$")


def is_flight(t):
    return t.astype(str).str.startswith("flight test")


def kind(t):
    return np.where(t.astype(str).str.startswith("flight"), "flight", np.where(t.astype(str) == "charging", "charge", "other"))


def load(bundle: Path):
    s = pd.read_parquet(bundle / "sessions.parquet")
    s["start_utc"] = pd.to_datetime(s["start_utc"], errors="coerce")
    p = pd.read_parquet(bundle / "packs.parquet")
    p = p.merge(s[["flight_id", "aircraft_label", "session_type", "start_utc", "scale_anomaly", "is_duplicate_content"]], on="flight_id", how="left")
    st = pd.read_parquet(bundle / "soh_steps.parquet") if (bundle / "soh_steps.parquet").exists() else pd.DataFrame()
    man = json.loads((bundle / "manifest.json").read_text())
    ac = json.loads((bundle / "aircraft.json").read_text())
    ap = pd.read_csv(bundle / "airports.csv") if (bundle / "airports.csv").exists() else pd.DataFrame()
    q = json.loads((bundle / "quality.json").read_text()) if (bundle / "quality.json").exists() else {}
    s, p, st = fix_cross_aircraft(s, p, st)
    return s, p, st, man, ac, ap, q


CROSS = {}


def fix_cross_aircraft(s, p, st):
    """Re-attribute logs listed under two aircraft in bundles exported before eplanes.quality.crossdup existed.

    Those bundles kept the copy with the lowest flight_id. Identical sessions are recognised by an exact signature
    (duration and pack-1 SOC and SOH values to 1e-5) and attributed by eplanes.quality.crossdup.resolve; the kept copy
    inherits the usability flags of the copy that was kept before (same content, same usability)."""
    if "cross_aircraft_duplicate" in s:
        return s, p, st
    from eplanes.quality.crossdup import resolve
    sig_cols = ["duration_min", "b1_soc_start", "b1_soc_end", "b1_soh_median", "b2_soh_median"]
    d = s.dropna(subset=sig_cols).copy()
    d["sig"] = ["|".join(f"{v:.5f}" for v in row) for row in d[sig_cols].to_numpy(float)]
    d["soh"] = d[["b1_soh_median", "b2_soh_median"]].mean(axis=1)
    res = resolve(d.rename(columns={"start_utc": "start_datetime"}), key="sig", soh="soh")
    s = s.copy()
    s["cross_aircraft_duplicate"] = s.flight_id.isin(res.flight_id)
    ucols = [c for c in s.columns if c.startswith("usable_") and c != "usable_exclusion_reason"]
    pcols = [c for c in ("A_overview", "B_profiles", "C_soc_dod", "D_charging", "E_thermal", "F_soh", "G_capacity") if c in p]
    p = p.copy()
    for _, g in res.groupby("group"):
        ids = list(g.flight_id)
        was = s[s.flight_id.isin(ids) & ~s.is_duplicate_content.fillna(False)].flight_id.tolist()
        keep = g[g.keep].flight_id.tolist()
        src = was[0] if was else None
        # snapshot the flags of the previously kept copy before any copy is changed
        s_src = s.loc[s.flight_id == src, ucols].to_numpy() if src is not None else None
        p_src = {b: p.loc[(p.flight_id == src) & (p.battery == b), pcols].to_numpy() for b in (1, 2)} if src is not None else None
        for fid in ids:
            m, mp = s.flight_id == fid, p.flight_id == fid
            if fid in keep and src is not None and src != fid:
                s.loc[m, ucols] = s_src
                for b in (1, 2):
                    p.loc[mp & (p.battery == b), pcols] = p_src[b]
            if fid not in keep:
                s.loc[m, ucols] = False
                p.loc[mp, pcols] = False
            s.loc[m, "is_duplicate_content"] = fid not in keep
            p.loc[mp, "is_duplicate_content"] = fid not in keep
        CROSS[int(g.group.iloc[0])] = {"ids": [f"{fid} ({ac})" for fid, ac in zip(g.flight_id, g.aircraft_label)],
                                       "kept": keep, "previously_kept": was}
    return s, p, st


def series_for(bundle: Path, fids):
    return pd.read_parquet(bundle / "series.parquet", filters=[("flight_id", "in", list(map(int, fids)))])


# ----------------------------------------------------------------------------------------------- figures
def fig_coverage(o, s):
    acs = [a for a in AC if (s.aircraft_label == a).any()]
    fig, ax = plt.subplots(figsize=(W2, 1.9))
    s = s.dropna(subset=["start_utc"]).copy()
    s["k"] = kind(s.session_type)
    for i, a in enumerate(acs):
        g = s[s.aircraft_label == a]
        for j, (k, lab) in enumerate((("flight", "flight test"), ("charge", "charging"), ("other", "ground test / other"))):
            gg = g[g.k == k]
            ax.scatter(gg.start_utc, np.full(len(gg), i) + (j - 1) * 0.22, s=5, marker="|", color=TYPE_COL[k], lw=0.8,
                       label=lab if i == 0 else None)
    ax.set_yticks(range(len(acs)), acs)
    ax.set_ylim(-0.6, len(acs) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, markerscale=2.5, handletextpad=0.3)
    o.save(fig, "fig_coverage")


def fig_maps(o, s, bundle, ap):
    tr = pd.read_parquet(bundle / "series.parquet", columns=["flight_id", "lat", "lon"]).dropna()
    if tr.empty:
        return
    tr = tr.merge(s.loc[~s.is_duplicate_content.fillna(False).astype(bool), ["flight_id", "aircraft_label", "dep_airport"]], on="flight_id")
    homes = s.dep_airport.value_counts()
    homes = [h for h in homes.index if homes[h] >= 5][:2]
    fig, axes = plt.subplots(1, max(len(homes), 1), figsize=(W2, 3.1))
    axes = np.atleast_1d(axes)
    for ax, home in zip(axes, homes):
        a = ap[ap.code == home].iloc[0]
        g = tr[tr.dep_airport == home]
        k = np.cos(np.radians(a.latitude_deg))
        for (fid, lab), t in g.groupby(["flight_id", "aircraft_label"]):
            t = t.iloc[::3]
            ax.plot((t.lon - a.longitude_deg) * 111.32 * k, (t.lat - a.latitude_deg) * 110.57, color=AC_COL.get(lab, INK3), lw=0.4, alpha=0.45)
        for _, r in ap.iterrows():
            x, y = (r.longitude_deg - a.longitude_deg) * 111.32 * k, (r.latitude_deg - a.latitude_deg) * 110.57
            if abs(x) < 60 and abs(y) < 60:
                ax.scatter(x, y, s=22, facecolor="white", edgecolor=INK, lw=0.8, zorder=5)
                ax.annotate(r.code, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7, zorder=6)
        ax.set_aspect("equal")
        ax.set_xlabel("km east of " + home)
        ax.set_ylabel("km north of " + home)
        labs = sorted(g.aircraft_label.unique(), key=AC.index)
        ax.set_title(f"{'ab'[list(homes).index(home)]}  {home}: {g.flight_id.nunique()} tracks")
        for lab in labs:
            ax.plot([], [], color=AC_COL[lab], lw=1.5, label=lab)
        ax.legend(loc="lower left")
    o.save(fig, "fig_maps")


def pick_representative(s, typ, has_gps=False):
    g = s[(s.session_type == typ) & (~s.scale_anomaly.fillna(False)) & (~s.is_duplicate_content.fillna(False)) & s.b1_soc_start.notna()]
    if has_gps:
        g = g[g.has_gps_track]
    if g.empty:
        return None
    med = g.duration_min.median()
    return int(g.iloc[(g.duration_min - med).abs().argsort()].flight_id.iloc[0])


def first_bin_masked(v: pd.Series) -> pd.Series:
    """Blank the first 2 s bin of each run of avionics data.

    Bundles exported before the 10 Hz leading-zero mask for avionics channels can average the last zeros before
    switch-on into that bin (e.g. 14.3 = 2/3 of 21.5 degC); newer bundles never contain such a bin."""
    v = v.copy()
    start = v.notna() & ~v.shift(1).notna()
    v[start] = np.nan
    return v


def fig_representative(o, s, bundle):
    fl, ch = pick_representative(s, "flight test", True), pick_representative(s, "charging")
    ids = [x for x in (fl, ch) if x is not None]
    if not ids:
        return
    tr = series_for(bundle, ids)
    rows = [("b{}_current_a", "Current (A)"), ("b{}_soc_pct", "SOC (%)"), ("b{}_max_cell_temp_c", "Max cell temp. (°C)"), ("pressure_alt_m", "Pressure alt. (m)")]
    fig, axes = plt.subplots(len(rows), len(ids), figsize=(W2, 5.0), sharex="col", squeeze=False)
    for j, fid in enumerate(ids):
        t = tr[tr.flight_id == fid].sort_values("t_s")
        x = t.t_s / 60
        meta = s[s.flight_id == fid].iloc[0]
        for i, (col, lab) in enumerate(rows):
            ax = axes[i, j]
            if "{}" in col:
                for b in (1, 2):
                    c, ls = PACK[b]
                    ax.plot(x, t[col.format(b)], color=c, ls=ls, lw=1.0, label=f"pack {b}")
                if col.startswith("b{}_max_cell_temp") and t.oat_c.notna().any():
                    oat = first_bin_masked(t.oat_c)
                    ax.plot(x, oat, color=INK3, lw=1.0, ls=":")
                    k = oat.last_valid_index()
                    ax.annotate("outside\nair", (x[k], oat[k]), xytext=(4, 0), textcoords="offset points", ha="left", va="center", fontsize=6.5, color=INK3)
            elif t[col].notna().any():
                ax.plot(x, first_bin_masked(t[col]), color=INK2, lw=1.0)
            else:
                ax.text(0.5, 0.5, "not logged (avionics off)", transform=ax.transAxes, ha="center", va="center", color=INK3, fontsize=7)
            if j == 0:
                ax.set_ylabel(lab)
            if i == 0:
                ax.set_title(f"{'ab'[j]}  {'Flight test' if j == 0 and fl else 'Charging'}, {meta.aircraft_label}, session {fid}")
                if j == 0:
                    ax.legend(loc="upper right", ncol=2)
        axes[-1, j].set_xlabel("time since start of recording (min)")
    o.save(fig, "fig_representative")
    o.num("representative_flight_id", fl)
    o.num("representative_charge_id", ch)


def fig_operations(o, s):
    ok = s[~s.scale_anomaly.fillna(False) & ~s.is_duplicate_content.fillna(False)]
    # same selections as the operations table: flight-profile and charging usability flags
    fl = ok[is_flight(ok.session_type) & ok.b1_soc_start.notna() & ok.b1_soc_end.notna() & ok.usable_B_profiles.fillna(False).astype(bool)]
    ch = ok[(ok.session_type == "charging") & ok.b1_soc_start.notna() & ok.b1_soc_end.notna() & ok.usable_D_charging.fillna(False).astype(bool)]
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.2))
    for a in AC:
        g = fl[fl.aircraft_label == a]
        if len(g) < 5:
            continue
        d = np.sort((g.b1_soc_start - g.b1_soc_end).to_numpy())
        ax[0].step(d, np.arange(1, len(d) + 1) / len(d), where="post", color=AC_COL[a], label=f"{a} (n={len(d)})")
        e = (g.b1_energy_discharged_kwh + g.b2_energy_discharged_kwh).dropna()
        ax[1].scatter(g.loc[e.index, "duration_min"], e, s=4, color=AC_COL[a], marker=AC_MK[a], alpha=0.6, lw=0)
    ax[0].set_xlabel("depth of discharge, pack 1 (SOC points)")
    ax[0].set_ylabel("cumulative share of flights")
    ax[0].legend(loc="center left", fontsize=6.5)
    ax[0].set_title("a  Depth of discharge")
    ax[1].set_xlabel("session duration (min)")
    ax[1].set_ylabel("energy out, both packs (kWh)")
    ax[1].set_title("b  Energy per flight")
    for a in AC:
        g = ch[ch.aircraft_label == a]
        if len(g) < 5:
            continue
        ax[2].scatter(g.b1_soc_start, g.b1_soc_end, s=4, color=AC_COL[a], marker=AC_MK[a], alpha=0.6, lw=0, label=a)
    ax[2].set_xlabel("SOC at start of charge (%)")
    ax[2].set_ylabel("SOC at end of charge (%)")
    ax[2].set_title("c  Charging sessions")
    fig.tight_layout(w_pad=1.2)
    o.save(fig, "fig_operations")
    for a in AC:
        g = fl[fl.aircraft_label == a]
        if len(g):
            o.num(f"dod_median_{a}", float(np.median(g.b1_soc_start - g.b1_soc_end)))
            o.num(f"flights_end_le20_{a}", int((g.b1_soc_end <= 20).sum()))


def fig_thermal(o, s):
    g = s[s.oat_c_median.notna() & s.b1_temp_max_c.notna() & ~s.is_duplicate_content.fillna(False)]
    fl = s[is_flight(s.session_type) & s.b1_temp_delta_c.notna() & s.usable_B_profiles.fillna(False).astype(bool)]
    if g.empty:
        return
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.4))
    lo, hi = np.floor(min(g.oat_c_median.min(), g.b1_temp_max_c.min()) / 5) * 5, np.ceil(max(g.oat_c_median.max(), g.b1_temp_max_c.max()) / 5) * 5
    for a in AC:
        x = g[g.aircraft_label == a]
        ax[0].scatter(x.oat_c_median, x.b1_temp_max_c, s=5, color=AC_COL[a], marker=AC_MK[a], alpha=0.6, lw=0, label=a)
        y = fl[fl.aircraft_label == a]
        ax[1].scatter(y.b1_i_mean_abs_a if "b1_i_mean_abs_a" in y else y.b1_ah_discharged, y.b1_temp_delta_c, s=5, color=AC_COL[a], marker=AC_MK[a], alpha=0.6, lw=0)
    ax[0].plot([lo, hi], [lo, hi], color=INK3, ls=":", lw=0.8)
    ax[0].set_xlabel("outside air temperature, session median (°C)")
    ax[0].set_ylabel("maximum cell temperature (°C)")
    ax[0].set_title("a  Cells versus outside air")
    ax[0].legend(loc="lower right", markerscale=2)
    ax[1].set_xlabel("mean |current|, pack 1 (A)" if "b1_i_mean_abs_a" in fl else "charge out, pack 1 (Ah)")
    ax[1].set_ylabel("temperature rise in flight (°C)")
    ax[1].set_title("b  Temperature rise versus load")
    fig.tight_layout(w_pad=1.5)
    o.save(fig, "fig_thermal")
    o.num("n_sessions_with_oat", int(len(g)))
    gap = g.b1_temp_max_c - g.oat_c_median
    o.num("cell_over_ambient_median", float(gap.median()))
    cold, warm = g[g.oat_c_median < 5], g[g.oat_c_median > 20]
    o.num("cell_over_ambient_below5C", float((cold.b1_temp_max_c - cold.oat_c_median).median()) if len(cold) else None)
    o.num("cell_over_ambient_above20C", float((warm.b1_temp_max_c - warm.oat_c_median).median()) if len(warm) else None)


def fig_soh(o, s, p):
    # an aircraft needs reported SOH on several distinct days for a trajectory
    acs = [a for a in AC if s[(s.aircraft_label == a) & s.b1_soh_median.notna()].start_utc.dt.date.nunique() >= 5]
    o.num("soh_trajectory_aircraft", acs)
    fig, axes = plt.subplots(len(acs), 1, figsize=(W2, 1.55 * len(acs) + 0.3), sharex=True, squeeze=False)
    for ax, a in zip(axes[:, 0], acs):
        g = s[(s.aircraft_label == a) & s.b1_soh_median.notna() & ~s.is_duplicate_content.fillna(False)].sort_values("start_utc")
        for b in (1, 2):
            c, ls = PACK[b]
            ax.step(g.start_utc, g[f"b{b}_soh_median"], where="post", color=c, ls=ls, lw=1.0, label=f"reported SOH, pack {b}")
        pp = p[(p.aircraft_label == a) & (p.battery == 1) & p.soh_cell_cells_last_mean.notna() & (p.soh_cell_cells_last_max <= 100.5)]
        ax.scatter(pp.start_utc, pp.soh_cell_cells_last_mean, s=3, color=PACK[1][0], alpha=0.35, lw=0, label="mean of 96 cells, pack 1 (BMS)")
        u = g[g.b1_uniform_cells.fillna(False).astype(bool)]
        ax.scatter(u.start_utc, u.b1_soh_median, s=26, facecolor="none", edgecolor=INK, lw=0.8, label="uniform-cell state (reset)", zorder=5)
        # shade idle periods > 60 days
        d = g.start_utc.diff().dt.days
        for i in np.flatnonzero(d.to_numpy() > 60):
            ax.axvspan(g.start_utc.iloc[i - 1], g.start_utc.iloc[i], color=GRID, alpha=0.6, lw=0)
        ax.set_ylabel(f"{a}\nSOH (%)")
        ax.set_ylim(max(0, np.floor(g[["b1_soh_median", "b2_soh_median"]].min().min() / 10) * 10 - 5), 102)
    axes[0, 0].legend(loc="lower left", ncol=2, fontsize=6.5)
    o.save(fig, "fig_soh_trajectories")


def fig_soh_vs_cells(o, p):
    g = p[p.soh_median.notna() & p.soh_cell_cells_last_min.notna() & (p.soh_cell_cells_last_max <= 100.5) & ~p.uniform_cells.fillna(False).astype(bool)
          & ~p.is_duplicate_content.fillna(False).astype(bool)]
    if g.empty:
        return
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6), sharey=True)
    for a in AC:
        x = g[g.aircraft_label == a]
        ax[0].scatter(x.soh_cell_cells_last_min, x.soh_median, s=4, color=AC_COL[a], marker=AC_MK[a], alpha=0.5, lw=0, label=a)
        ax[1].scatter(x.soh_cell_cells_last_mean, x.soh_median, s=4, color=AC_COL[a], marker=AC_MK[a], alpha=0.5, lw=0)
    for k, (col, t) in enumerate((("soh_cell_cells_last_min", "a  Weakest cell"), ("soh_cell_cells_last_mean", "b  Mean of 96 cells"))):
        lo = np.floor(min(g[col].min(), g.soh_median.min()) / 10) * 10
        ax[k].plot([lo, 100], [lo, 100], color=INK3, ls=":", lw=0.8)
        ax[k].set_xlabel(f"{'minimum' if k == 0 else 'mean'} per-cell SOH, BMS file (%)")
        ax[k].set_title(t)
        rho = stats.spearmanr(g[col], g.soh_median)[0]
        within = float(((g.soh_median - g[col]).abs() <= 1).mean())
        ax[k].text(0.97, 0.05, f"Spearman {rho:.3f}\nwithin 1 pt: {100 * within:.0f}%", transform=ax[k].transAxes, ha="right", va="bottom", fontsize=7)
        o.num(f"soh_vs_{'min' if k == 0 else 'mean'}_rho", float(rho))
        o.num(f"soh_vs_{'min' if k == 0 else 'mean'}_within1", within)
    ax[0].set_ylabel("reported pack SOH (%)")
    ax[0].legend(loc="upper left", markerscale=2)
    fig.tight_layout(w_pad=1)
    o.save(fig, "fig_soh_vs_cells")
    o.num("n_pack_sessions_soh_cells", int(len(g)))
    o.num("soh_minus_min_median", float((g.soh_median - g.soh_cell_cells_last_min).median()))
    o.num("mean_minus_soh_mean", float((g.soh_cell_cells_last_mean - g.soh_median).mean()))
    o.num("max_minus_soh_mean", float((g.soh_cell_cells_last_max - g.soh_median).mean()))
    o.num("cell_spread_median", {f"{a}|{b}": float((x.soh_cell_cells_last_max - x.soh_cell_cells_last_min).median())
                                 for (a, b), x in g.groupby(["aircraft_label", "battery"]) if len(x) >= 10})


def fig_soh_mechanics(o, s, st):
    """Where and when the BMS revises reported SOH inside a session."""
    ok = s[~s.is_duplicate_content.fillna(False) & s.b1_soh_median.notna()].copy()
    ok["k"] = kind(ok.session_type)
    steps = st[st.battery == 1] if len(st) else pd.DataFrame(columns=["flight_id", "delta", "t_min", "temp_at_start_c", "soc_at_step"])
    first = steps.sort_values("t_min").groupby("flight_id").first()
    ok["net"] = ok.flight_id.map(steps.groupby("flight_id").delta.sum()).fillna(0)
    ok["changed"] = ok.flight_id.isin(steps.flight_id)
    ch = ok[ok.k == "charge"].copy()
    ch["band"] = pd.cut(ch.b1_soc_start, [-1, 30, 45, 60, 101], labels=["<30", "30–45", "45–60", "≥60"])
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.3), gridspec_kw={"width_ratios": [1, 1.25, 1.25]})
    by = ok.groupby("k").changed.agg(["mean", "size"]).reindex(["flight", "charge", "other"]).dropna()
    ax[0].bar(range(len(by)), 100 * by["mean"], color=[TYPE_COL[k] for k in by.index], width=0.6)
    ax[0].set_xticks(range(len(by)), [f"{k}\n(n={int(n)})" for k, n in zip(by.index, by["size"])])
    ax[0].set_ylabel("sessions with an SOH step, pack 1 (%)")
    ax[0].set_title("a  Session type")
    cnt = ch.groupby("band", observed=False).apply(lambda g: pd.Series({"up": (g.net > 0).mean(), "none": (g.net == 0).mean(), "down": (g.net < 0).mean(), "n": len(g)}))
    bottom = np.zeros(len(cnt))
    for lab, col in (("down", "#eb6834"), ("none", "#c9cfd4"), ("up", "#2a78d6")):
        ax[1].bar(range(len(cnt)), 100 * cnt[lab], bottom=100 * bottom, color=col, width=0.6, label={"up": "SOH up", "none": "no change", "down": "SOH down"}[lab], edgecolor="white", lw=0.5)
        bottom += cnt[lab].to_numpy()
    ax[1].set_xticks(range(len(cnt)), [f"{b}\n(n={int(n)})" for b, n in zip(cnt.index, cnt.n)])
    ax[1].set_xlabel("SOC at start of charge (%)")
    ax[1].set_ylabel("share of charging sessions (%)")
    ax[1].set_title("b  Charging start SOC")
    ax[1].set_ylim(0, 122)
    ax[1].set_yticks(range(0, 101, 20))
    h_, l_ = ax[1].get_legend_handles_labels()
    ax[1].legend(h_[::-1], l_[::-1], loc="upper center", ncol=3, fontsize=6.3, handlelength=1, columnspacing=0.8, borderaxespad=0.1)
    neg = steps[steps.delta < 0].sort_values("t_min").groupby("flight_id").first()
    neg = neg[neg.index.isin(ch.flight_id)].dropna(subset=["temp_at_start_c"])
    if len(neg) >= 5:
        ac_of_ = ch.set_index("flight_id").aircraft_label
        lab_ = neg.index.map(ac_of_)
        for a in AC:
            m_ = np.asarray(lab_ == a)
            if m_.any():
                ax[2].scatter(neg.temp_at_start_c[m_], neg.t_min[m_], s=8, marker=AC_MK[a], facecolor="none",
                              edgecolor=INK2, lw=0.6, alpha=0.8, label=f"{a} (n={int(m_.sum())})")
        ax[2].legend(loc="center right", fontsize=6.3, handletextpad=0.2, borderaxespad=0.1)
        r = stats.spearmanr(neg.temp_at_start_c, neg.t_min)
        ax[2].text(0.03, 0.97, f"pooled Spearman {r[0]:.2f}", transform=ax[2].transAxes, ha="left", va="top", fontsize=6.5)
        o.num("neg_step_timing_rho", float(r[0]))
        o.num("neg_step_timing_p", float(r[1]))
        o.num("neg_step_timing_n", int(len(neg)))
        cold, hot = neg[neg.temp_at_start_c < 15], neg[neg.temp_at_start_c >= 25]
        o.num("neg_step_minutes_cold_median", float(cold.t_min.median()) if len(cold) else None)
        o.num("neg_step_minutes_hot_median", float(hot.t_min.median()) if len(hot) else None)
        o.num("neg_step_soc_median", float(neg.soc_at_step.median()))
        ac_of = ch.set_index("flight_id").aircraft_label
        for ac, g in neg.groupby(neg.index.map(ac_of)):
            if len(g) < 10:
                continue
            ra = stats.spearmanr(g.temp_at_start_c, g.t_min)
            c, w = g[g.temp_at_start_c < 15], g[g.temp_at_start_c >= 25]
            o.num(f"neg_step_{ac}", {"rho": float(ra[0]), "p": float(ra[1]), "n": int(len(g)),
                                      "cold_min": float(c.t_min.median()) if len(c) else None, "n_cold": int(len(c)),
                                      "hot_min": float(w.t_min.median()) if len(w) else None, "n_hot": int(len(w))})
    ax[2].set_xlabel("max cell temperature at start of charge (°C)")
    ax[2].set_ylabel("minutes to first downward step")
    ax[2].set_title("c  Timing of downward steps")
    fig.tight_layout(w_pad=1.0)
    o.save(fig, "fig_soh_mechanics")
    for k in by.index:
        o.num(f"soh_step_share_{k}", float(by.loc[k, "mean"]))
    for b in cnt.index:
        o.num(f"charge_band_{b}", {k: float(cnt.loc[b, k]) for k in ("up", "none", "down", "n")})
    up = steps[steps.delta > 0]
    # is the direction of an in-charge revision related to the start temperature? (charges with a net change only)
    chg = ch[ch.net != 0]
    for a, g in chg.groupby("aircraft_label"):
        if len(g) >= 30 and "b1_temp_start_c" in g:
            r = stats.spearmanr(g.b1_temp_start_c, (g.net > 0).astype(int), nan_policy="omit")
            o.num(f"direction_vs_temp_{a}", {"rho": float(r[0]), "p": float(r[1]), "n": int(g.b1_temp_start_c.notna().sum())})
    o.num("steps_soc_iqr", [float(steps.soc_at_step.quantile(.25)), float(steps.soc_at_step.quantile(.75))] if len(steps) else None)
    o.num("n_steps_pack1", int(len(steps)))
    o.num("n_up_steps_pack1", int(len(up)))


def fig_observability(o, s, p):
    g = p[~p.scale_anomaly.fillna(False) & ~p.is_duplicate_content.fillna(False) & p.delta_soc.notna() & p.ah_net_discharge.notna()]
    g = g[(g.delta_soc.abs() >= 10)]
    cap = (p[(p.capacity_reason == "ok") & (p.battery == 1) & p.implied_capacity_ah.notna() & ~p.scale_anomaly.fillna(False)
             & ~p.is_duplicate_content.fillna(False)] if "capacity_reason" in p else pd.DataFrame())
    fig = plt.figure(figsize=(W2, 3.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.5], hspace=0.12)
    a0 = fig.add_subplot(gs[:, 0])
    for a in AC:
        x = g[(g.aircraft_label == a) & (g.battery == 1)]
        a0.scatter(-x.ah_net_discharge, x.delta_soc, s=4, color=AC_COL[a], marker=AC_MK[a], alpha=0.5, lw=0, label=a)
    a0.axhline(0, color=INK3, lw=0.6)
    a0.axvline(0, color=INK3, lw=0.6)
    a0.set_xlabel("net charge in, integrated current (Ah)")
    a0.set_ylabel("change in reported SOC (points)")
    a0.set_title("a  SOC follows integrated current")
    a0.legend(loc="upper left", markerscale=2, fontsize=6.5)
    a1 = fig.add_subplot(gs[0, 1])
    a2 = fig.add_subplot(gs[1, 1], sharex=a1)
    for a in AC[:3]:
        c = cap[cap.aircraft_label == a].dropna(subset=["start_utc"])
        if len(c) < 5:
            continue
        m = c.set_index("start_utc").implied_capacity_ah.resample("MS").median().dropna()
        a1.scatter(c.start_utc, c.implied_capacity_ah, s=3, color=AC_COL[a], alpha=0.25, lw=0)
        a1.plot(m.index + pd.Timedelta(days=15), m, color=AC_COL[a], marker=AC_MK[a], ms=3, lw=1.0, label=a)
        r = s[(s.aircraft_label == a) & s.b1_soh_median.notna() & ~s.is_duplicate_content.fillna(False).astype(bool)].set_index("start_utc").b1_soh_median.resample("MS").median().dropna()
        a2.plot(r.index + pd.Timedelta(days=15), r, color=AC_COL[a], marker=AC_MK[a], ms=3, lw=1.0)
    a1.set_ylabel("window capacity (Ah)")
    a1.set_title("b  Monthly medians, pack 1")
    a1.legend(loc="lower left", ncol=3, fontsize=6.5)
    plt.setp(a1.get_xticklabels(), visible=False)
    a2.set_ylabel("reported SOH (%)")
    import matplotlib.dates as mdates
    loc = mdates.AutoDateLocator(minticks=4, maxticks=8)
    a2.xaxis.set_major_locator(loc)
    a2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    o.save(fig, "fig_observability")


# ----------------------------------------------------------------------------------------------- tables
def tab_dataset(o, s, ac, ap, man):
    for a_ in AC:
        g_ = s[s.aircraft_label == a_]
        o.num(f"usable_{a_}", {c.removeprefix("usable_"): int(g_[c].fillna(False).astype(bool).sum())
                               for c in s.columns if c.startswith("usable_") and c != "usable_exclusion_reason"} | {"downloaded": int(g_.has_timeseries.fillna(False).sum()) if "has_timeseries" in g_ else None})
    rows = []
    apn = dict(zip(ap.code, ap.name)) if len(ap) else {}
    for a in AC:
        g = s[s.aircraft_label == a]
        if g.empty:
            continue
        meta = next((x for x in ac if x["label"] == a), {})
        k = kind(g.session_type)
        st = g.start_utc.dropna()
        rows.append([a, meta.get("serial_number") or "--", meta.get("fleet", "").replace("fleet_", ""), meta.get("home_airfield") or "--",
                     len(g), int((k == "flight").sum()), int((k == "charge").sum()), int((k == "other").sum()),
                     f(g.duration_min.sum() / 60, 0), f"{st.min():%Y-%m-%d}", f"{st.max():%Y-%m-%d}", int(g.has_gps_track.sum())])
        o.num(f"sessions_{a}", int(len(g)))
    k = kind(s.session_type)
    st = s.start_utc.dropna()
    rows.append("MID")
    rows.append([r"\textbf{All}", "", "", "", len(s), int((k == "flight").sum()), int((k == "charge").sum()), int((k == "other").sum()),
                 f(s.duration_min.sum() / 60, 0), f"{st.min():%Y-%m-%d}", f"{st.max():%Y-%m-%d}", int(s.has_gps_track.sum())])
    o.table("tab_dataset", ["Aircraft", "Serial", "Acct.", "Base", "Sessions", "Flights", "Charges", "Other", "Hours", "First", "Last", "GPS"],
            rows, "llllrrrrrllr")
    o.num("n_sessions", int(len(s)))
    o.num("first_session", f"{st.min():%Y-%m-%d}")
    o.num("last_session", f"{st.max():%Y-%m-%d}")
    o.num("hours", float(s.duration_min.sum() / 60))
    o.num("n_gps_tracks", int(s.has_gps_track.sum()))
    o.num("bases", {a.get("label"): a.get("home_airfield") for a in ac})
    o.num("base_names", {c: apn.get(c) for c in set(a.get("home_airfield") for a in ac) if c})


def _theil(t, y):
    ok = np.isfinite(t) & np.isfinite(y)
    if ok.sum() < 4 or np.ptp(t[ok]) <= 0:
        return (np.nan, np.nan, np.nan)
    r = stats.theilslopes(y[ok], t[ok])
    return (r.slope, r.low_slope, r.high_slope)


def tab_trends_bundle(o, p, min_months=6):
    """Fade per year, pack 1: Theil-Sen over sessions and over calendar-month medians (each month one vote).

    Same method as eplanes.analysis.eda.monthly_trends, with explicit filters: duplicates (including logs listed under
    two aircraft) are excluded from every series, and per-cell estimates above 100.5 % (scale-fault sessions) are dropped."""
    b = p[(p.battery == 1) & p.start_utc.notna() & ~p.is_duplicate_content.fillna(False).astype(bool)]
    series = [
        ("reported SOH", "pts", b[b.F_soh.fillna(False).astype(bool)], "soh_median"),
        ("min per-cell SOH", "pts", b[b.soh_cell_cells_last_max <= 100.5], "soh_cell_cells_last_min"),
        ("mean per-cell SOH", "pts", b[b.soh_cell_cells_last_max <= 100.5], "soh_cell_cells_last_mean"),
        ("window capacity", "Ah", b[b.G_capacity.fillna(False).astype(bool) & (b.capacity_reason == "ok")], "implied_capacity_ah"),
    ]
    rows, out = [], {}
    for a in AC:
        first_row = True
        for name, unit, d, col in series:
            g = d[(d.aircraft_label == a)].dropna(subset=[col])
            if len(g) < 8:
                continue
            t = ((g.start_utc - pd.Timestamp("1970-01-01")).dt.total_seconds() / 86400 / 365.25).to_numpy(float)
            s_sl = _theil(t, g[col].to_numpy(float))[0]
            m = g.assign(month=g.start_utc.dt.to_period("M")).groupby("month")[col].median()
            tm = np.array([((pp.start_time + pd.Timedelta(days=15)) - pd.Timestamp("1970-01-01")).total_seconds() / 86400 / 365.25 for pp in m.index])
            m_sl, lo, hi = _theil(tm, m.to_numpy(float)) if len(m) >= min_months else (np.nan, np.nan, np.nan)
            pct = 100 * m_sl / m.iloc[0] if unit == "Ah" and np.isfinite(m_sl) else np.nan
            out[f"{a}|{name}"] = dict(months=len(m), sessions=len(g), session=s_sl, monthly=m_sl, lo=lo, hi=hi, pct=pct)
            if len(m) < 3:
                continue
            ci = f"[{f(lo, 1, True)}, {f(hi, 1, True)}]" if np.isfinite(lo) else "--"
            rows.append([a if first_row else "", name, unit, len(m), f(s_sl, 1), f(m_sl, 1), ci, f(pct, 1) if np.isfinite(pct) else "--"])
            first_row = False
        if not first_row:
            rows.append("MID")
    if rows:
        o.table("tab_trends", ["Aircraft", "Indicator", "Unit", "Months", "Per session", "Monthly", "95\\% CI", r"\%/yr"], rows[:-1], "lllrrrlr")
    o.num("trends", out)


def soh_jumps(o, p):
    """Session-to-session changes of reported SOH (method of eplanes.analysis.eda.section_F): a jump is a change of at
    least 2 points and 4 robust standard deviations of that pack's changes."""
    d = p[p.F_soh.fillna(False).astype(bool) & p.soh_median.notna() & p.start_utc.notna() &
          ~p.is_duplicate_content.fillna(False).astype(bool)].sort_values(["aircraft_label", "battery", "start_utc", "flight_id"])
    jumps, share = [], {}
    for (a, b), g in d.groupby(["aircraft_label", "battery"]):
        x = g.soh_median.to_numpy(float)
        if len(x) < 3:
            continue
        ds = np.diff(x)
        days = g.start_utc.diff().dt.total_seconds().to_numpy()[1:] / 86400
        mad = 1.4826 * np.median(np.abs(ds - np.median(ds))) or 0.5
        big = np.abs(ds) >= max(2.0, 4 * mad)
        share[f"{a}|{b}"] = float((ds > 0).mean())
        jumps += [dict(aircraft=a, battery=b, flight_id=int(g.flight_id.iloc[k + 1]), delta=float(ds[k]), idle_days=float(days[k]))
                  for k in np.flatnonzero(big)]
    j = pd.DataFrame(jumps)
    o.num("n_large_jumps", int(len(j)))
    o.num("increase_share", share)
    idle = j[(j.battery == 1) & (j.idle_days > 30)] if len(j) else j
    tau = {}
    for (a, b), g in d.groupby(["aircraft_label", "battery"]):
        if len(g) > 5 and g.start_utc.dt.date.nunique() > 5:
            tau[f"{a}|{b}"] = float(stats.kendalltau(g.start_utc.astype("int64"), g.soh_median)[0])
    o.num("kendall_tau_date", tau)
    w = d[d.aircraft_label == "C-GAUW"].pivot_table(index="flight_id", columns="battery", values="soh_median").dropna()
    if len(w) >= 10:
        diff = (w[1] - w[2]).to_numpy()
        nz = diff[diff != 0]
        rk = stats.rankdata(np.abs(nz))
        rbc = float((rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum()) if len(nz) else np.nan
        o.num("pack_gap_C-GAUW", {"n": int(len(w)), "median": float(np.median(diff)), "rank_biserial": rbc,
                                  "p": float(stats.wilcoxon(nz)[1]) if len(nz) > 10 else None})
    o.num("idle_jumps_pack1", idle.to_dict("records"))


def reset_test(o, p):
    """Do reported-SOH increases (>= 2 points between consecutive sessions with BMS files) coincide with the BMS
    entering a uniform-cell state? Method of eplanes.analysis.stats (H9), duplicates excluded."""
    b = p[p.start_utc.notna() & p.soh_cell_cells_last_max.notna() & (p.soh_cell_cells_last_max <= 100.5) &
          ~p.is_duplicate_content.fillna(False).astype(bool)].sort_values(["aircraft_label", "battery", "start_utc", "flight_id"]).copy()
    b["uniform_cells"] = b.uniform_cells.fillna(False).astype(bool)
    grp = b.groupby(["aircraft_label", "battery"])
    b["d_soh"] = grp.soh_median.diff()
    b["enter"] = b.uniform_cells & ~grp.uniform_cells.shift(1).fillna(False).astype(bool)
    ev = b.dropna(subset=["d_soh"])
    inc = ev.d_soh >= 2
    tab = np.array([[int((inc & ev.enter).sum()), int((inc & ~ev.enter).sum())], [int((~inc & ev.enter).sum()), int((~inc & ~ev.enter).sum())]])
    odds, pv = stats.fisher_exact(tab, alternative="greater") if tab[0].sum() and tab[:, 0].sum() else (np.nan, np.nan)
    o.num("reset_test", {"table": tab.tolist(), "odds_ratio": float(odds), "p": float(pv), "n_pairs": int(tab.sum()),
                         "n_increases": int(tab[0].sum()), "n_increase_with_entry": int(tab[0, 0]), "n_uniform": int(b.uniform_cells.sum())})


def tab_operations(o, s):
    ok = s[~s.scale_anomaly.fillna(False) & ~s.is_duplicate_content.fillna(False)]
    rows = []
    for a in AC:
        g = ok[(ok.aircraft_label == a)]
        fl = g[is_flight(g.session_type) & g.b1_soc_start.notna() & g.usable_B_profiles.fillna(False).astype(bool)]
        ch = g[(g.session_type == "charging") & g.b1_soc_start.notna() & g.usable_D_charging.fillna(False).astype(bool)]
        if len(fl) < 3:
            continue
        e = fl.b1_energy_discharged_kwh + fl.b2_energy_discharged_kwh
        pk = fl.b1_power_peak_kw + fl.b2_power_peak_kw
        rows.append([a, len(fl), f(fl.duration_min.median(), 0), f(fl.airborne_min.median(), 0) if "airborne_min" in fl else "--",
                     f((fl.b1_soc_start - fl.b1_soc_end).median(), 0), f(e.median(), 1), f(pk.median(), 0), f(fl.b1_temp_delta_c.median(), 0),
                     len(ch), f(ch.duration_min.median(), 0) if len(ch) else "--", f(ch.b1_soc_start.median(), 0) if len(ch) else "--"])
    o.table("tab_operations", ["Aircraft", "Flights", "Dur.", "Airborne", "DoD", "Energy", "Peak", r"$\Delta T$", "Charges", "Dur.", "Start SOC"], rows, "lrrrrrrrrrr")


def tab_results(o, res: Path):
    t = res / "tables"
    if (t / "G4_trend_monthly_vs_session.csv").exists():
        g4 = pd.read_csv(t / "G4_trend_monthly_vs_session.csv")
        g4 = g4[(g4.battery == 1) & g4.aircraft.isin(AC[:3])]
        rows = []
        for a in AC[:3]:
            for q in ("reported SOH", "min per-cell SOH (BMS)", "mean per-cell SOH (BMS)", "window capacity"):
                r = g4[(g4.aircraft == a) & (g4.quantity == q)]
                if r.empty:
                    continue
                r = r.iloc[0]
                rows.append([a if q == "reported SOH" else "", q.replace(" (BMS)", ""), "pts" if r.unit == "pts" else "Ah", int(r.n_months),
                             f(r.session_slope_per_year, 1, True), f(r.monthly_slope_per_year, 1, True),
                             f"[{f(r.monthly_ci_low, 1, True)}, {f(r.monthly_ci_high, 1, True)}]",
                             f(r.monthly_pct_per_year, 1, True) if r.unit == "Ah" else "--"])
            rows.append("MID")
        o.table("tab_trends", ["Aircraft", "Indicator", "Unit", "Months", "Per session", "Monthly", "95\\% CI", r"\%/yr"], rows[:-1], "lllrrrlr")
    if (t / "G0_observability_table.csv").exists():
        g0 = pd.read_csv(t / "G0_observability_table.csv")
        o.table("tab_observability", ["Indicator", "Grade", r"Coverage"],
                [[r.indicator.replace("&", r"\&").replace("%", r"\%"), r.grade, f"{100 * r.coverage:.0f}\\%" if pd.notna(r.coverage) else "--"] for r in g0.itertuples()],
                r">{\raggedright\arraybackslash}p{7.2cm}>{\raggedright\arraybackslash}p{4.6cm}r")


def write_macros(n: dict, path: Path):
    """LaTeX macros for every data-derived number quoted in the manuscript text."""
    def pct(x):
        return "\\pending" if x is None or not np.isfinite(x) else f"{100 * x:.0f}\\%"

    def num(x, d=0):
        return "\\pending" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:,.{d}f}"
    bands = {k.removeprefix("charge_band_"): v for k, v in n.items() if k.startswith("charge_band_")}
    hi = bands.get("≥60", {}) or {}
    low = {k: v for k, v in bands.items() if k in ("<30", "30–45")}
    low_n = sum(v["n"] for v in low.values()) or 1
    low_change = sum((v["up"] + v["down"]) * v["n"] for v in low.values()) / low_n if low else None
    base_names = n.get("base_names", {}) or {}
    bases = n.get("bases", {}) or {}
    m = {
        "NSessionsBundle": num(n.get("n_sessions")), "NGpsTracks": num(n.get("n_gps_tracks")),
        "BaseA": bases.get("C-GAUW") or "\\pending", "BaseB": bases.get("VSW1280102") or "\\pending",
        "BaseAName": base_names.get(bases.get("C-GAUW"), "\\pending") or "\\pending",
        "BaseBName": base_names.get(bases.get("VSW1280102"), "\\pending") or "\\pending",
        "SohStepShareCharge": pct(n.get("soh_step_share_charge")), "SohStepShareFlight": pct(n.get("soh_step_share_flight")),
        "SohStepShareOther": pct(n.get("soh_step_share_other")),
        "TopOffNoChange": pct(hi.get("none")), "TopOffN": num(hi.get("n")), "LowStartChange": pct(low_change),
        "NegStepRho": num(n.get("neg_step_timing_rho"), 2), "NegStepN": num(n.get("neg_step_timing_n")),
        "NegStepColdMin": num(n.get("neg_step_minutes_cold_median")), "NegStepHotMin": num(n.get("neg_step_minutes_hot_median")),
        "NegStepSoc": num(n.get("neg_step_soc_median")),
        # LaTeX macro names cannot contain digits: C-GAUW -> Gauw, VSW1280102 -> Vsw
        **{f"NegStep{key}{tag}": num((n.get(f"neg_step_{ac}") or {}).get(f), d)
           for ac, tag in (("C-GAUW", "Gauw"), ("VSW1280102", "Vsw"))
           for key, f, d in (("Rho", "rho", 2), ("N", "n", 0), ("Cold", "cold_min", 0), ("Hot", "hot_min", 0), ("NCold", "n_cold", 0), ("NHot", "n_hot", 0))},
        "StepSocLo": num((n.get("steps_soc_iqr") or [None, None])[0]), "StepSocHi": num((n.get("steps_soc_iqr") or [None, None])[1]),
        "CellOverAmbient": num(n.get("cell_over_ambient_median"), 1), "CellOverAmbientCold": num(n.get("cell_over_ambient_below5C"), 1),
        "CellOverAmbientWarm": num(n.get("cell_over_ambient_above20C"), 1), "NOat": num(n.get("n_sessions_with_oat")),
        "RepFlight": str(n.get("representative_flight_id") or "\\pending"), "RepCharge": str(n.get("representative_charge_id") or "\\pending"),
    }
    # fade trends (magnitudes of declining slopes; CI bounds swap when negated)
    tr = n.get("trends", {}) or {}
    main2 = ("C-GAUW", "VSW1280102")

    def tv(a, q, k):
        v = (tr.get(f"{a}|{q}") or {}).get(k)
        return None if v is None or not np.isfinite(v) else float(v)

    def rng(q, k, d=1, neg=True):
        v = [tv(a, q, k) for a in main2]
        v = [(-x if neg else x) for x in v if x is not None]
        return "\\pending" if not v else (num(min(v), d) if f"{min(v):.{d}f}" == f"{max(v):.{d}f}" else f"{min(v):.{d}f}--{max(v):.{d}f}")
    ratios = [tv(a, q, "session") / tv(a, q, "monthly") for a in main2 for q in ("reported SOH", "min per-cell SOH", "mean per-cell SOH", "window capacity")
              if tv(a, q, "session") is not None and tv(a, q, "monthly")]
    for a, tag in (("C-GAUW", "Gauw"), ("VSW1280102", "Vsw")):
        m[f"TrSoh{tag}"] = num(-tv(a, "reported SOH", "monthly") if tv(a, "reported SOH", "monthly") is not None else None, 1)
        m[f"TrSoh{tag}Lo"] = num(-tv(a, "reported SOH", "hi") if tv(a, "reported SOH", "hi") is not None else None, 1)
        m[f"TrSoh{tag}Hi"] = num(-tv(a, "reported SOH", "lo") if tv(a, "reported SOH", "lo") is not None else None, 1)
    m.update({"TrMinCell": rng("min per-cell SOH", "monthly"), "TrMeanCell": rng("mean per-cell SOH", "monthly"),
              "TrCap": rng("window capacity", "monthly"), "TrCapPct": rng("window capacity", "pct"),
              "TrRatioLo": num(min(ratios), 1) if ratios else "\\pending", "TrRatioHi": num(max(ratios), 1) if ratios else "\\pending",
              "TrMonthsGmuw": num((tr.get("C-GMUW|reported SOH") or {}).get("months"))})
    idle = n.get("idle_jumps_pack1") or []
    m.update({"NLargeJumps": num(n.get("n_large_jumps")), "IdleJumpsN": num(len(idle)) if idle else "\\pending",
              "IdleJumpsDown": num(sum(1 for r in idle if r["delta"] < 0)) if idle else "\\pending"})
    sp = n.get("cell_spread_median") or {}
    def spread(a):
        v = [x for k, x in sp.items() if k.startswith(a + "|")]
        return "\\pending" if not v else (f"{min(v):.0f}" if round(min(v)) == round(max(v)) else f"{min(v):.0f}--{max(v):.0f}")
    taus = [v for k, v in (n.get("kendall_tau_date") or {}).items() if k.split("|")[0] in ("C-GAUW", "C-GMUW", "VSW1280102")]
    pg = n.get("pack_gap_C-GAUW") or {}
    ug = n.get("usable_C-GAUW") or {}
    dv = {a: n.get(f"direction_vs_temp_{a}") or {} for a in main2}
    m.update({"SohMinRho": num(n.get("soh_vs_min_rho"), 3), "SohMinWithin": pct(n.get("soh_vs_min_within1")), "SohCellsN": num(n.get("n_pack_sessions_soh_cells")),
              "SohMinOffset": num(-(n.get("soh_minus_min_median") or 0) if n.get("soh_minus_min_median") is not None else None, 1),
              "SohBelowMean": num(n.get("mean_minus_soh_mean"), 1), "SohBelowMax": num(n.get("max_minus_soh_mean"), 1),
              "SpreadGauw": spread("C-GAUW"), "SpreadVsw": spread("VSW1280102"), "SpreadGmuw": spread("C-GMUW"),
              "TauLo": num(min(taus), 2) if taus else "\\pending", "TauHi": num(max(taus), 2) if taus else "\\pending",
              "PackGapN": num(pg.get("n")), "PackGapMedian": num(abs(pg["median"]) if pg.get("median") is not None else None, 0),
              "PackGapR": num(pg.get("rank_biserial"), 2),
              "UseGauwDl": num(ug.get("downloaded")), "UseGauwB": num(ug.get("B_profiles")), "UseGauwF": num(ug.get("F_soh")), "UseGauwG": num(ug.get("G_capacity")),
              "DirTempGauw": num(dv["C-GAUW"].get("rho"), 2), "DirTempVsw": num(dv["VSW1280102"].get("rho"), 2),
              "DirTempGauwP": num(dv["C-GAUW"].get("p"), 2), "DirTempVswP": num(dv["VSW1280102"].get("p"), 2)})
    rt = n.get("reset_test") or {}
    m.update({"ResetN": num(rt.get("n_uniform")), "ResetInc": num(rt.get("n_increases")), "ResetIncEnter": num(rt.get("n_increase_with_entry")),
              "ResetOR": num(rt.get("odds_ratio"), 1), "ResetP": num(rt.get("p"), 2), "ResetPairs": num(rt.get("n_pairs")),
              "ResetIncPct": pct(rt["n_increase_with_entry"] / rt["n_increases"]) if rt.get("n_increases") else "\\pending"})
    inc = n.get("increase_share") or {}
    incv = [v for k, v in inc.items() if k.split("|")[0] in main2]
    m["IncShare"] = f"{100 * min(incv):.0f}--{100 * max(incv):.0f}\\%" if incv else "\\pending"
    lines = ["% generated by paper/make_figures.py from the web bundle; do not edit by hand",
             f"% bundle created {n.get('bundle_created_utc')}; synthetic={n.get('synthetic')}"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    path.write_text("\n".join(lines) + "\n")


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--bundle", required=True)
    ap_.add_argument("--results", default=None)
    ap_.add_argument("--out", default="paper")
    a = ap_.parse_args()
    bundle, out = Path(a.bundle), Out(Path(a.out))
    s, p, st, man, ac, ap, q = load(bundle)
    out.num("bundle_created_utc", man.get("created_utc"))
    out.num("synthetic", man.get("synthetic"))
    fig_coverage(out, s)
    fig_maps(out, s, bundle, ap)
    fig_representative(out, s, bundle)
    fig_operations(out, s)
    fig_thermal(out, s)
    fig_soh(out, s, p)
    fig_soh_vs_cells(out, p)
    fig_soh_mechanics(out, s, st)
    fig_observability(out, s, p)
    tab_dataset(out, s, ac, ap, man)
    tab_operations(out, s)
    if a.results:
        tab_results(out, Path(a.results))
    tab_trends_bundle(out, p)  # supersedes the results-archive trend table: same method, corrected attribution
    soh_jumps(out, p)
    reset_test(out, p)
    out.num("cross_aircraft_groups", CROSS)
    (Path(a.out) / "numbers.json").write_text(json.dumps(out.numbers, indent=1, default=str))
    write_macros(out.numbers, Path(a.out) / "numbers.tex")
    print(f"figures -> {out.fig}, tables -> {out.tab}, numbers -> {Path(a.out) / 'numbers.json'}")


if __name__ == "__main__":
    main()
