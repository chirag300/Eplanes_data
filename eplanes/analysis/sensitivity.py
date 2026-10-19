"""Sensitivity of headline results to cleaning, session-segmentation and timestamp-alignment choices.

One-at-a-time design around the baseline (rule=leadzero, active current 2 A, logged timestamps):
  cleaning rule      : none | row0 | leadzero | settle | fixed:0.5 | fixed:1.0 | fixed:3.0
  segmentation       : active-current threshold 1 | 2 | 5 A
  timestamp alignment: logged | uniform (index x median dt) | gapbridge (integrate across gaps)
For each scenario the headline metrics are recomputed and compared to the baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt

from ..config import Settings
from ..quality.report import require_quality_gate
from ..viz.style import BLUE, FigureSink
from .context import attach_meta
from .features import extract_all

BASE = dict(rule="leadzero", seg=2.0, time_rule="logged")
BASELINE = "rule=leadzero"
SCENARIOS = ([dict(name=f"rule={r}", rule=r, seg=2.0, time_rule="logged") for r in ["none", "row0", "leadzero", "settle", "fixed:0.5", "fixed:1.0", "fixed:3.0"]] +
             [dict(name=f"seg={x:g}A", rule="leadzero", seg=x, time_rule="logged") for x in (1.0, 5.0)] +
             [dict(name=f"time={t}", rule="leadzero", seg=2.0, time_rule=t) for t in ("uniform", "gapbridge")])


def headline_metrics(ctx: pd.DataFrame) -> dict:
    m = {}
    ft = ctx[(ctx.flight_type == "flight test") & (ctx.C_soc_dod == True) & (ctx.frac_discharge > 0.05) & (ctx.battery == 1)].copy()  # noqa: E712
    ft["dod"] = ft.soc_start - ft.soc_end
    for ac, g in ft.groupby("aircraft_label"):
        m[f"median_soc_start[{ac}]"] = g.soc_start.median()
        m[f"median_dod[{ac}]"] = g.dod.median()
    groups = [g.dod.dropna().values for _, g in ft.groupby("aircraft_label") if g.dod.notna().sum() >= 3]
    if len(groups) >= 2:
        H, p = stats.kruskal(*groups)
        m["H1_KW_p(dod across aircraft)"] = p
    w = ctx[ctx.C_soc_dod == True].pivot_table(index="flight_id", columns="battery", values="delta_soc").dropna()  # noqa: E712
    d = (w[1] - w[2]).values if len(w) else np.array([])
    if len(d) >= 8 and np.any(d != 0):
        m["H5_paired_median_diff_dSOC"] = float(np.median(d))
        m["H5_wilcoxon_p"] = float(stats.wilcoxon(d).pvalue)
    prof = ctx[(ctx.flight_type == "flight test") & (ctx.B_profiles == True) & (ctx.battery == 1)]
    for ac, g in prof.groupby("aircraft_label"):
        m[f"median_Ah_discharged[{ac}]"] = g.ah_discharged.median()
    cap = ctx[(ctx.G_capacity == True) & (ctx.capacity_reason == "ok")]
    m["window_capacity_median_Ah"] = cap.implied_capacity_ah.median() if len(cap) else np.nan
    m["window_capacity_n_ok"] = float(len(cap))
    chg = ctx[(ctx.battery == 1) & (ctx.frac_charge > 0.2)]
    m["n_charging_sessions_detected"] = float(len(chg))
    m["median_start_temp_C"] = ctx[ctx.battery == 1].temp_start_c.median()
    return m


def run_sensitivity(settings: Settings, max_flights: int | None = None, log=print) -> pd.DataFrame:
    require_quality_gate(settings)
    ids = None
    per_ac = settings["analysis"].get("sensitivity_max_per_aircraft")
    fl = pd.read_csv(settings.data_dir / "processed" / "flights_with_qc.csv")
    fl = fl[fl.n_rows.fillna(0) > 0]
    if max_flights:
        ids = set(fl.sort_values("flight_id").flight_id.iloc[:max_flights])
    elif per_ac:
        # fixed-seed stratified sample per aircraft (each scenario re-extracts all features, so the full set is slow)
        ids = set(fl.groupby("aircraft_label", group_keys=False)
                    .apply(lambda g: g.sample(min(len(g), int(per_ac)), random_state=settings["analysis"]["random_seed"])).flight_id)
        log(f"[sensitivity] using a fixed-seed sample of up to {per_ac} sessions per aircraft ({len(ids)} sessions); "
            "set analysis.sensitivity_max_per_aircraft: null in config.yaml for all sessions")
    rows = []
    for sc in SCENARIOS:
        log(f"[sensitivity] {sc['name']}")
        feat = extract_all(settings, sc["rule"], sc["seg"], ids, sc["time_rule"])
        ctx = attach_meta(settings, feat)
        # usability flags come from the quality stage (computed once, independent of the scenario)
        for k, v in headline_metrics(ctx).items():
            rows.append(dict(scenario=sc["name"], metric=k, value=v))
    long = pd.DataFrame(rows)
    base = long[long.scenario == BASELINE].set_index("metric").value
    long["baseline"] = long.metric.map(base)
    isp = long.metric.str.contains("_p")
    long["rel_dev_pct"] = np.where(isp, np.nan, 100 * (long.value - long.baseline) / long.baseline.abs().replace(0, np.nan))
    long["abs_dev"] = long.value - long.baseline
    long["same_side_of_0.05"] = np.where(isp, (long.value < 0.05) == (long.baseline < 0.05), np.nan)
    out = settings.outputs_dir / "tables"
    out.mkdir(parents=True, exist_ok=True)
    long.to_csv(out / "S1_sensitivity_long.csv", index=False)
    wide = long.pivot(index="metric", columns="scenario", values="value")
    wide.reset_index().to_csv(out / "S2_sensitivity_wide.csv", index=False)
    verdict = []
    for mtr, g in long.groupby("metric"):
        if "_p" in mtr:
            stable = bool(g["same_side_of_0.05"].dropna().all())
            verdict.append(dict(metric=mtr, kind="p-value", worst_dev="; ".join(f"{r.scenario}:{r.value:.3g}" for r in g.itertuples()), stable=stable))
        else:
            w = g.rel_dev_pct.abs().max()
            w2 = g[g.scenario != "rule=none"].rel_dev_pct.abs().max()
            verdict.append(dict(metric=mtr, kind="estimate", worst_dev=f"{w:.1f}% max |relative deviation| ({w2:.1f}% excluding the no-cleaning rule)",
                                stable_excl_no_cleaning=bool(w2 < 10) if np.isfinite(w2) else np.nan,
                                stable=bool(w < 10) if np.isfinite(w) else np.nan))
    pd.DataFrame(verdict).to_csv(out / "S3_sensitivity_verdict.csv", index=False)

    # figure: relative deviation heat map
    est = long[~isp].pivot(index="metric", columns="scenario", values="rel_dev_pct")
    if len(est):
        est = est[[s["name"] for s in SCENARIOS]]
        fig, ax = plt.subplots(figsize=(9, 0.32 * len(est) + 1.8))
        lim = max(10, np.nanpercentile(np.abs(est.values), 95)) if np.isfinite(est.values).any() else 10
        im = ax.imshow(est.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(est.shape[1]), est.columns, rotation=45, ha="right")
        ax.set_yticks(range(est.shape[0]), est.index, fontsize=7)
        for i in range(est.shape[0]):
            for j in range(est.shape[1]):
                v = est.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6)
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="deviation from baseline [%]")
        FigureSink(settings.outputs_dir / "figures").save(fig, "S1_sensitivity_heatmap",
            "Relative deviation (%) of headline estimates from the baseline (rule=leadzero, 2 A activity threshold, logged timestamps) under alternative "
            "cleaning, segmentation and timestamp-alignment rules. Diverging scale centred at zero (grey = no change). p-values are listed in table S2.")
    return long
