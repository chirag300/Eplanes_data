"""Figures, a one-file summary and a ZIP of the results. The ZIP holds tables, figures and logs; no raw telemetry."""
from __future__ import annotations

import datetime as dt
import json
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import Ctx, Log

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
PHYS, BMS, WIN = "#2a78d6", "#eb6834", "#1baf7a"      # categorical slots 1, 2, 3 of the validated palette, in fixed order
BAND = "#2a78d6"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False, "figure.dpi": 150})


def fig_label(ctx: Ctx, lab: pd.DataFrame, obs: pd.DataFrame):
    keys = [k for k, g in lab.groupby(["aircraft_label", "battery"]) if g.t_days.max() >= 30]
    if not keys:
        return
    fig, axes = plt.subplots(len(keys), 1, figsize=(8.5, 1.9 * len(keys) + 0.6), sharex=True, squeeze=False)
    for ax, k in zip(axes[:, 0], keys):
        g = lab[(lab.aircraft_label == k[0]) & (lab.battery == k[1])].sort_values("start_datetime")
        ob = obs[(obs.aircraft_label == k[0]) & (obs.battery == k[1]) & obs.modality.isin(["chg", "r2r"])]
        ax.scatter(ob.start_datetime, 100 * ob.y_c, s=2, color=MUTED, alpha=0.25, linewidths=0, label="capacity readings")
        ax.step(g.start_datetime, g.soh_bms_median, where="post", color=BMS, lw=1.5, label="reported by BMS")
        ax.fill_between(g.start_datetime, g.soh_phys - 2 * g.soh_sd, g.soh_phys + 2 * g.soh_sd, color=BAND, alpha=0.18, linewidth=0)
        ax.plot(g.start_datetime, g.soh_phys, color=PHYS, lw=1.5, label="physics label, 2 sd band")
        ax.set_ylabel(f"{k[0]} pack {k[1]}\nSOH (%)")
        ax.set_ylim(min(30, np.nanmin(g.soh_bms_median) - 3), 112)
    axes[0, 0].legend(loc="lower left", ncols=3, fontsize=8, markerscale=5)
    fig.tight_layout()
    fig.savefig(ctx.figs / "F1_physics_label_vs_bms.png", dpi=170)
    plt.close(fig)


def fig_steps(ctx: Ctx):
    p = ctx.tables / "V1_bms_step_events.csv"
    if not p.exists() or p.stat().st_size < 5:
        return
    t = pd.read_csv(p)
    if t.empty:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    lim = float(np.nanmax(np.abs(t.d_bms))) * 1.08
    ax.plot([-lim, lim], [-lim, lim], ":", color=MUTED, lw=1)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.scatter(t.d_bms, t.d_soc_window, s=22, color=WIN, alpha=0.75, marker="s", linewidths=0, label="SOC-window capacity")
    ax.scatter(t.d_bms, t.d_phys, s=24, color=PHYS, alpha=0.85, linewidths=0, label="voltage-anchored capacity")
    ax.set_xlabel("change in reported SOH across a BMS jump (points)")
    ax.set_ylabel("change in capacity estimate (points)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(ctx.figs / "F2_bms_step_test.png", dpi=170)
    plt.close(fig)


def fig_forecast(ctx: Ctx):
    p = ctx.tables / "M2_cv_forecast_scores.csv"
    if not p.exists():
        return
    s = pd.read_csv(p)
    if s.empty:
        return
    hs = sorted(s.h.unique())
    fig, axes = plt.subplots(1, len(hs), figsize=(3.4 * len(hs), 3.6), sharex=False, squeeze=False)
    for ax, h in zip(axes[0], hs):
        g = s[s.h == h].sort_values("rmse", ascending=False)
        fam = {"ude": PHYS, "pinn": PHYS, "gru": PHYS, "transformer": PHYS, "residual_gbm": PHYS, "residual_gp": PHYS, "physics_ode": WIN, "physics_bayes": WIN}
        ax.barh(g.model, g.rmse, color=[fam.get(m, "#b9b8b0") for m in g.model], height=0.68)
        ax.set_title(f"{h} days ahead", fontsize=9, color=INK)
        ax.set_xlabel("RMSE (SOH points)")
        ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=PHYS, label="hybrid"), Patch(color=WIN, label="physics only"), Patch(color="#b9b8b0", label="baseline or no physics")], loc="lower center", ncols=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(ctx.figs / "F3_forecast_rmse.png", dpi=170)
    plt.close(fig)


def _table(ctx, name, n=None):
    p = ctx.tables / name
    if not p.exists() or p.stat().st_size < 5:
        return "(not available)\n"
    d = pd.read_csv(p)
    if n:
        d = d.head(n)
    return d.round(3).to_markdown(index=False) + "\n"


def write_summary(ctx: Ctx):
    v = json.loads((ctx.out / "validation_summary.json").read_text()) if (ctx.out / "validation_summary.json").exists() else {}
    syn = (ctx.settings.data_dir / "SYNTHETIC").exists()
    lock = ctx.out / "FINAL_TEST_LOCK.json"
    txt = ["# SOH pipeline results", ""]
    if syn:
        txt += ["**SIMULATED DATA. These numbers show the method works in principle and say nothing about the real fleet.**", ""]
    txt += ["## Coverage", _table(ctx, "E1_extraction_coverage.csv"), "## Label", _table(ctx, "L3_label_summary.csv"), "### Steps the label accepted", _table(ctx, "L2_detected_steps.csv"),
            "## Validation", "```json", json.dumps(v, indent=1, default=str), "```", "### Fade by indicator", _table(ctx, "V3_fade_by_indicator.csv"),
            "## Forecast, training fleet (rolling origin)", _table(ctx, "M2_cv_forecast_scores.csv"), "## Nowcast by ablation run, training fleet", _table(ctx, "M4_cv_nowcast_ablation.csv"),
            "### Raw-curve CNN", _table(ctx, "M5_cv_curve_cnn.csv"), "### Conformal intervals", _table(ctx, "M3_cv_conformal.csv")]
    if lock.exists():
        st = json.loads(lock.read_text())
        txt += [f"## Held-out fleet (scored {st['runs']} time(s), first {st['first_run_utc']})", _table(ctx, "T2_final_forecast_scores.csv"), _table(ctx, "T3_final_nowcast_ablation.csv")]
    else:
        txt += ["## Held-out fleet", "Not scored yet."]
    (ctx.out / "SOH_RESULTS.md").write_text("\n".join(txt))


def run_report(ctx: Ctx, log=None):
    log = log or Log(ctx)
    lab = pd.read_parquet(ctx.cache / "labels.parquet")
    obs = pd.read_parquet(ctx.cache / "observations_calibrated.parquet")
    fig_label(ctx, lab, obs)
    fig_steps(ctx)
    fig_forecast(ctx)
    write_summary(ctx)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M")
    zp = ctx.settings.outputs_dir / f"soh_results_{ts}.zip"
    hyper = ctx.cache / "label_hyper.json"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(ctx.out.rglob("*")):
            if f.is_file():
                z.write(f, f"soh/{f.relative_to(ctx.out)}")
        if hyper.exists():
            z.write(hyper, "soh/label_hyper.json")
    log(f"[report] {ctx.out / 'SOH_RESULTS.md'}")
    log(f"[report] upload this file: {zp} ({zp.stat().st_size / 1e6:.1f} MB)")
    return zp
