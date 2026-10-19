"""Hypothesis tests (paper: Operational Patterns / Quality / SOH behaviour).

Principles
  * unit of analysis = flight (battery-pair differences for paired tests); samples within a flight are never pooled
  * non-parametric tests (distributions are skewed, n is modest, SOC is integer-quantised)
  * effect sizes + bootstrap CIs resampling FLIGHTS; Holm correction within each hypothesis family and
    Benjamini-Hochberg across all tests
  * robustness column: the same test after collapsing to one value per aircraft-day (removes same-day pseudo-replication)
  * results are labelled descriptive/exploratory when n is small or assumptions are doubtful
Nothing is tuned to reach significance; every test that is run is reported.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from ..config import Settings
from ..quality.report import require_quality_gate
from .context import bh, boot_ci, cliffs_delta, holm, load_context

MIN_N = 8
warnings.filterwarnings("ignore", message="An input array is constant")


def _kw(groups: dict[str, np.ndarray]):
    g = {k: np.asarray(v, float)[np.isfinite(v)] for k, v in groups.items()}
    g = {k: v for k, v in g.items() if len(v) >= 3}
    if len(g) < 2:
        return None
    H, p = stats.kruskal(*g.values())
    n = sum(len(v) for v in g.values())
    eps2 = (H - len(g) + 1) / (n - len(g)) if n > len(g) else np.nan  # epsilon-squared
    return dict(stat=float(H), p=float(p), effect_name="epsilon^2", effect=float(eps2), n=int(n),
                groups=";".join(f"{k}(n={len(v)})" for k, v in g.items())), g


def _pairwise(g: dict, reps=1000):
    rows = []
    for a, b in itertools.combinations(g, 2):
        u, p = stats.mannwhitneyu(g[a], g[b], alternative="two-sided")
        d = cliffs_delta(g[a], g[b])
        rng = np.random.default_rng(1)
        bs = [cliffs_delta(rng.choice(g[a], len(g[a])), rng.choice(g[b], len(g[b]))) for _ in range(reps // 4)]
        rows.append(dict(a=a, b=b, p=float(p), cliffs_delta=d, ci_low=float(np.percentile(bs, 2.5)), ci_high=float(np.percentile(bs, 97.5)),
                         n_a=len(g[a]), n_b=len(g[b]), median_diff=float(np.median(g[a]) - np.median(g[b]))))
    pw = pd.DataFrame(rows)
    pw["p_holm"] = holm(pw.p.values) if len(pw) else []
    return pw


def _day_collapse(df, value, by):
    d = df.copy()
    d["day"] = d.start_datetime.dt.normalize()
    return d.groupby(by + ["aircraft_label", "day"], dropna=False)[value].median().reset_index()


def _result(rows, hid, family, h0, h1, test, unit, res, extra=None):
    if res is None:
        rows.append(dict(id=hid, family=family, H0=h0, H1=h1, test=test, unit=unit, n=0, note="insufficient data"))
        return
    r = dict(id=hid, family=family, H0=h0, H1=h1, test=test, unit=unit)
    r.update(res)
    if extra:
        r.update(extra)
    rows.append(r)


def run_stats(settings: Settings, rule="leadzero", log=print) -> pd.DataFrame:
    require_quality_gate(settings)
    out_dir = settings.outputs_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = load_context(settings, rule)
    rows, pairs = [], []
    reps = settings["analysis"]["bootstrap_reps"]

    ft = ctx[(ctx.flight_type == "flight test") & (ctx.C_soc_dod == True) & (ctx.frac_discharge > 0.05) & (ctx.battery == 1)].copy()  # noqa: E712
    ft["dod"] = ft.soc_start - ft.soc_end
    prof = ctx[(ctx.flight_type == "flight test") & (ctx.B_profiles == True) & (ctx.battery == 1)]

    # H1 flight-level SOC change differs between aircraft
    for var, lab in (("dod", "depth of discharge"), ("soc_start", "start SOC")):
        res = _kw({a: g[var].values for a, g in ft.groupby("aircraft_label")})
        if res:
            r, g = res
            col = _day_collapse(ft, var, [])
            r2 = _kw({a: gg[var].values for a, gg in col.groupby("aircraft_label")})
            _result(rows, f"H1_{var}", "H1", f"{lab} has the same distribution in all aircraft", "at least one aircraft differs",
                    "Kruskal-Wallis", "flight (battery 1)", r, dict(p_day_collapsed=r2[0]["p"] if r2 else np.nan))
            pw = _pairwise(g)
            pw.insert(0, "id", f"H1_{var}")
            pairs.append(pw)
        else:
            _result(rows, f"H1_{var}", "H1", "", "", "Kruskal-Wallis", "flight", None)

    # H2 energy / power demand across aircraft and seasons
    for var, lab in (("i_peak_discharge_a", "peak discharge current"), ("ah_discharged", "discharged charge")):
        res = _kw({a: g[var].values for a, g in prof.groupby("aircraft_label")})
        if res:
            r, g = res
            col = _day_collapse(prof, var, [])
            r2 = _kw({a: gg[var].values for a, gg in col.groupby("aircraft_label")})
            _result(rows, f"H2_{var}_aircraft", "H2", f"{lab} same across aircraft", "differs", "Kruskal-Wallis", "flight (battery 1)", r,
                    dict(p_day_collapsed=r2[0]["p"] if r2 else np.nan))
            pw = _pairwise(g)
            pw.insert(0, "id", f"H2_{var}_aircraft")
            pairs.append(pw)
        for ac, gg in prof.groupby("aircraft_label"):
            res = _kw({s: x[var].values for s, x in gg.groupby("season")})
            if res:
                _result(rows, f"H2_{var}_season_{ac}", "H2", f"{lab} same across seasons within {ac}", "differs", "Kruskal-Wallis", "flight (battery 1)", res[0])

    # H3 charging patterns differ between aircraft
    ch = ctx[(ctx.D_charging == True) & (ctx.charge_time_min.fillna(0) > 3) & (ctx.battery == 1)].copy()  # noqa: E712
    for var in ("charge_time_min", "charge_soc_start", "charge_i_mean_a"):
        res = _kw({a: g[var].values for a, g in ch.groupby("aircraft_label")})
        if res:
            r, g = res
            _result(rows, f"H3_{var}", "H3", f"{var} same across aircraft", "differs", "Kruskal-Wallis", "charging session (battery 1)", r)
            pw = _pairwise(g)
            pw.insert(0, "id", f"H3_{var}")
            pairs.append(pw)

    # H4 temperature vs season (within aircraft) and vs current (Spearman, within aircraft)
    th = ctx[(ctx.E_thermal == True) & (ctx.battery == 1)]
    for ac, g in th.groupby("aircraft_label"):
        res = _kw({s: x.temp_max_c.values for s, x in g.groupby("season")})
        if res:
            _result(rows, f"H4_tempmax_season_{ac}", "H4", f"max cell temperature same across seasons ({ac})", "differs", "Kruskal-Wallis", "session (battery 1)", res[0])
    for ac, g in th[th.flight_type == "flight test"].groupby("aircraft_label"):
        x = g[["i_mean_abs_a", "temp_delta_c"]].dropna()
        if len(x) >= MIN_N and x.temp_delta_c.nunique() >= 3 and x.i_mean_abs_a.nunique() >= 3:
            rho, p = stats.spearmanr(x.i_mean_abs_a, x.temp_delta_c)
            lo, hi = boot_ci(np.arange(len(x)), lambda idx: stats.spearmanr(x.i_mean_abs_a.values[idx.astype(int)], x.temp_delta_c.values[idx.astype(int)])[0], 400)
            _result(rows, f"H4_dT_vs_current_{ac}", "H4", f"no monotone association between mean current and temperature rise ({ac})", "association",
                    "Spearman", "flight test (battery 1)", dict(stat=float(rho), p=float(p), effect_name="rho", effect=float(rho), n=len(x), ci_low=lo, ci_high=hi))

    # H5 packs differ systematically (paired within flight)
    for var, lab, sub in (("delta_soc", "SOC change", ctx[(ctx.C_soc_dod == True)]), ("i_mean_abs_a", "mean |current|", ctx[(ctx.B_profiles == True)]),
                          ("temp_max_c", "max cell temperature", ctx[(ctx.E_thermal == True)]), ("soh_median", "reported SOH", ctx[(ctx.F_soh == True)])):
        w = sub.pivot_table(index=["flight_id", "aircraft_label"], columns="battery", values=var).dropna().reset_index()
        for ac, g in list(w.groupby("aircraft_label")) + [("ALL", w)]:
            d = (g[1] - g[2]).values
            if len(d) >= MIN_N and np.any(d != 0):
                wst = stats.wilcoxon(d)
                lo, hi = boot_ci(d, np.median, reps)
                # matched-pairs rank-biserial correlation
                rk = stats.rankdata(np.abs(d[d != 0]))
                rbc = (rk[d[d != 0] > 0].sum() - rk[d[d != 0] < 0].sum()) / rk.sum()
                _result(rows, f"H5_{var}_{ac}", "H5", f"median(bat1 - bat2) {lab} = 0 ({ac})", "!= 0", "Wilcoxon signed-rank", "flight (paired packs)",
                        dict(stat=float(wst.statistic), p=float(wst.pvalue), effect_name="rank-biserial r", effect=float(rbc), n=len(d),
                             median_diff=float(np.median(d)), ci_low=lo, ci_high=hi))

    # H6 SOH reporting vs utilisation / calendar (first differences to avoid trend-induced correlation)
    d = ctx[(ctx.F_soh == True) & ctx.soh_median.notna() & ctx.start_datetime.notna()].sort_values(["aircraft_label", "battery", "start_datetime"]).copy()
    d["cum_ah"] = d.groupby(["aircraft_label", "battery"]).ah_throughput.cumsum()
    for (ac, b), g in d.groupby(["aircraft_label", "battery"]):
        if len(g) < MIN_N + 1:
            continue
        dsoh = g.soh_median.diff().values[1:]
        dah = g.cum_ah.diff().values[1:]
        ddays = g.start_datetime.diff().dt.total_seconds().values[1:] / 86400
        for nm, x in (("d_throughput", dah), ("d_days", ddays)):
            ok = np.isfinite(dsoh) & np.isfinite(x)
            if ok.sum() >= MIN_N:
                rho, p = stats.spearmanr(x[ok], dsoh[ok])
                _result(rows, f"H6_dSOH_vs_{nm}_{ac}_b{b}", "H6", f"no monotone link between session-to-session change in reported SOH and {nm} ({ac} bat {b})",
                        "link", "Spearman on first differences", "consecutive session pair",
                        dict(stat=float(rho), p=float(p), effect_name="rho", effect=float(rho), n=int(ok.sum())),
                        dict(note="first differences; consecutive sessions are serially dependent -> exploratory"))

    # H7 initial readings differ from settled values
    ini = pd.read_csv(settings.data_dir / "processed" / "init_records.csv")
    ini["signal_short"] = ini.signal.str.replace(r"^bat_\d_", "", regex=True)
    for sig, g in ini.groupby("signal_short"):
        g = g[np.isfinite(g.ref) & (g.ref.abs() > 5 * g.tol) & np.isfinite(g.x0)]
        if len(g) >= MIN_N and np.any((g.x0 - g.ref) != 0):
            d = ((g.x0 - g.ref) / g.ref.abs()).values
            w = stats.wilcoxon(d)
            lo, hi = boot_ci(d, np.median, reps)
            _result(rows, f"H7_row0_vs_settled_{sig}", "H7", f"first sample equals settled value ({sig})", "differs", "Wilcoxon signed-rank on relative deviation",
                    "flight", dict(stat=float(w.statistic), p=float(w.pvalue), effect_name="median relative deviation", effect=float(np.median(d)), n=len(d), ci_low=lo, ci_high=hi))
    anyart = ini.assign(art=ini.cls.isin(["zero_fill", "bin_avg_zero", "monotone_settling", "irregular_start"]))
    for sig, g in anyart.groupby("signal_short"):
        a = g[g.start_state == "rest_start"].art
        b = g[g.start_state == "load_start"].art
        if len(a) >= MIN_N and len(b) >= MIN_N and (a.nunique() > 1 or b.nunique() > 1):
            tab = np.array([[a.sum(), (~a).sum()], [b.sum(), (~b).sum()]])
            odds, p = stats.fisher_exact(tab)
            _result(rows, f"H7_artifact_by_start_state_{sig}", "H7", f"artefact rate equal for rest and load starts ({sig})", "differs", "Fisher exact", "flight",
                    dict(stat=float(odds), p=float(p), effect_name="odds ratio", effect=float(odds), n=int(tab.sum())))

    # H4b cell temperature rise over ambient vs mean current (sessions with OAT)
    if "oat_c_median" in ctx:
        g4 = ctx[(ctx.E_thermal == True) & (ctx.battery == 1) & ctx.oat_c_median.notna() & (ctx.flight_type == "flight test")].copy()  # noqa: E712
        g4["over_ambient"] = g4.temp_max_c - g4.oat_c_median
        for ac, g in list(g4.groupby("aircraft_label")) + [("ALL", g4)]:
            x = g[["i_mean_abs_a", "over_ambient"]].dropna()
            if len(x) >= MIN_N and x.over_ambient.nunique() >= 3:
                rho, p = stats.spearmanr(x.i_mean_abs_a, x.over_ambient)
                _result(rows, f"H4b_over_ambient_vs_current_{ac}", "H4", f"no monotone association between mean current and cell-over-ambient temperature ({ac})",
                        "association", "Spearman", "flight test (battery 1, OAT logged)",
                        dict(stat=float(rho), p=float(p), effect_name="rho", effect=float(rho), n=len(x)))

    # H9 reported-SOH increases coincide with the BMS entering a uniform-cell (reset-like) state
    try:
        from .eda import bms_context
        bb = bms_context(settings, ctx)
    except Exception:  # noqa: BLE001
        bb = pd.DataFrame()
    if len(bb) and "enter_uniform" in bb and "d_soh" in bb:
        ev = bb.dropna(subset=["d_soh"])
        inc = ev.d_soh >= 2
        tab = np.array([[int((inc & ev.enter_uniform).sum()), int((inc & ~ev.enter_uniform).sum())],
                        [int((~inc & ev.enter_uniform).sum()), int((~inc & ~ev.enter_uniform).sum())]])
        if tab.sum() >= MIN_N and tab[0].sum() > 0 and tab[:, 0].sum() > 0:
            odds, p = stats.fisher_exact(tab, alternative="greater")
            _result(rows, "H9_soh_increase_vs_uniform_entry", "H9",
                    "reported-SOH increases (>=2 points between consecutive sessions) are no more likely when the BMS enters a uniform-cell state",
                    "more likely (one-sided)", "Fisher exact (one-sided)", "consecutive battery-session pair",
                    dict(stat=float(odds), p=float(p), effect_name="odds ratio", effect=float(odds), n=int(tab.sum())),
                    dict(note=f"table [[inc&entry, inc&no-entry],[no-inc&entry, no-inc&no-entry]] = {tab.tolist()}"))
        else:
            _result(rows, "H9_soh_increase_vs_uniform_entry", "H9", "", "", "Fisher exact", "battery-session pair", None,
                    dict(note=f"not testable: table {tab.tolist()}"))

    # H8 data-quality problems concentrated by aircraft / type
    fl = pd.read_csv(settings.data_dir / "processed" / "flights_with_qc.csv")
    fl["has_gap"] = fl["has_gap"].astype(bool) if "has_gap" in fl else fl.n_gaps > 0
    fl["any_oor"] = fl.n_out_of_range > 0
    fl["label_mismatch"] = fl.label_content_mismatch.fillna(False).astype(bool)
    for var in ("has_gap", "any_oor", "label_mismatch"):
        for by in ("aircraft_label", "flight_type"):
            tab = pd.crosstab(fl[by], fl[var])
            if tab.shape[0] >= 2 and tab.shape[1] == 2 and tab.values.sum() >= MIN_N:
                chi2, p, dof, exp = stats.chi2_contingency(tab)
                small = (exp < 5).any()
                if small and tab.shape == (2, 2):
                    p = stats.fisher_exact(tab.values)[1]
                v = np.sqrt(chi2 / tab.values.sum() / (min(tab.shape) - 1))
                _result(rows, f"H8_{var}_by_{by}", "H8", f"{var} independent of {by}", "dependent",
                        "chi-square" + (" (Fisher for 2x2 with small expected counts)" if small and tab.shape == (2, 2) else ""), "flight",
                        dict(stat=float(chi2), p=float(p), effect_name="Cramer V", effect=float(v), n=int(tab.values.sum())),
                        dict(note="expected counts < 5 in some cells" if small else ""))

    res = pd.DataFrame(rows)
    if "p" in res:
        res["p_holm_family"] = np.nan
        for fam, idx in res.groupby("family").groups.items():
            res.loc[idx, "p_holm_family"] = holm(res.loc[idx, "p"].values)
        res["q_bh_global"] = bh(res["p"].values)
        res["exploratory_flag"] = (res.n.fillna(0) < 30) | res.note.fillna("").str.contains("exploratory|expected counts")
        res["interpretation"] = np.where(res.p_holm_family < 0.05, "evidence against H0 after Holm correction within family",
                                         np.where(res.p < 0.05, "nominal p<0.05 only; does not survive family correction -> not claimed",
                                                  "no evidence against H0 (absence of evidence, not proof of equality)"))
    res.to_csv(out_dir / "H0_hypothesis_tests.csv", index=False)
    if pairs:
        pd.concat(pairs, ignore_index=True).to_csv(out_dir / "H1_pairwise_posthoc.csv", index=False)
    log(f"[stats] {len(res)} tests written")
    return res
