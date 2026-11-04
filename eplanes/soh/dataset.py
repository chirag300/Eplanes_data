"""The modelling table and the leakage rules.

Feature groups (config: runs)
  phys     what one session shows without any capacity information: resistance at 25 degC, temperature, load, cell spread
  curve    capacity-bearing session measurements: the session's own capacity readings, voltage reached for the Ah moved
  history  usage so far: age, cumulative Ah, thermal exposure. These are time proxies
  bms      reported SOH and SOC-window capacity. Excluded from every BMS-free run

Leakage rules
  * Nowcast models that take features from a session are trained and scored on the leave-one-out label (that session's own
    measurements removed).
  * Training labels are recomputed from data before each cutoff. A smoothed label looks into the future, so using it across
    a cutoff would leak.
  * The current pack's history reaches a forecaster only as a causal (filtered) state.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .common import Ctx
from .label import Hyper, build_grid, kalman, robust_pass

PHYS = ["r_step25_mohm", "r_step3025_mohm", "r_ramp25_mohm", "r_step_rel", "r_ramp_rel", "temp_mean_c", "temp_max_c", "temp_rise_c", "temp_start_c", "i_rms_a", "i_peak_dis_a",
        "spread_mv_median", "spread_mv_max", "spread_mv_end", "ah_out", "is_charge"]
CURVE = ["y_chg", "y_dis", "y_r2r", "v_load_min_cell", "rest_start_v", "rest_end_v", "v_mean_dis", "charge_min", "ah_in", "v_max_cell", "v_min_cell"]
HISTORY = ["t_days", "sqrt_years", "ah_cum", "efc_cum", "hot_ah_cum", "cold_ah_cum", "temp_hist_mean", "depth_hist_mean", "rest_v_hist_mean", "gap_days", "n_prior"]
BMS = ["soh_bms_median", "soh_bms_first", "soh_bms_last", "q_soc_window_ah", "soh_cell_cells_last_min", "soh_cell_cells_last_mean", "soh_cell_cells_last_max", "cell_est_bat_cap_last"]
GROUPS = dict(phys=PHYS, curve=CURVE, history=HISTORY, bms=BMS)
# per-session drivers for the sequence models: all of them can be planned ahead. Voltages must never be here: a future session's
# voltage already carries that session's capacity, so feeding it to a forecaster would leak the answer.
STRESS = ["gap_days", "ah_out", "temp_mean_c", "i_rms_a", "sqrt_years", "is_flight"]


def build_table(ctx: Ctx) -> pd.DataFrame:
    lab = pd.read_parquet(ctx.cache / "labels.parquet")
    obs = pd.read_parquet(ctx.cache / "session_observables_r.parquet")
    keep = [c for c in obs.columns if c not in lab.columns or c in ("flight_id", "battery")]
    d = lab.merge(obs[keep], on=["flight_id", "battery"], how="left")
    long = pd.read_parquet(ctx.cache / "observations_calibrated.parquet")
    for m in ("chg", "dis", "r2r"):
        q = long[long.modality == m].drop_duplicates(["flight_id", "battery"])[["flight_id", "battery", "y_c"]].rename(columns={"y_c": f"y_{m}"})
        d = d.merge(q, on=["flight_id", "battery"], how="left")
    bp = ctx.proc / "bms_session.csv"
    if bp.exists():
        b = pd.read_csv(bp)
        want = [c for c in BMS if c in b and c not in d]
        d = d.merge(b[["flight_id", "battery"] + want], on=["flight_id", "battery"], how="left")
    d = d.sort_values(["aircraft_label", "battery", "start_datetime", "flight_id"]).reset_index(drop=True)
    # resistance relative to the pack's own early life: absolute values differ between packs, growth carries over between aircraft
    lc = ctx.cfg["label"]
    early = d[d.t_days.between(lc["settle_days"], lc["settle_days"] + lc["baseline_days"])]
    for col, rel in (("r_step25_mohm", "r_step_rel"), ("r_ramp25_mohm", "r_ramp_rel")):
        if col in d:
            base = early.groupby(["aircraft_label", "battery"])[col].quantile(lc["r_baseline_q"]).rename("_b")
            d[rel] = d[col] / d.join(base, on=["aircraft_label", "battery"])._b
    d["is_charge"] = (d.charge_min.fillna(0) >= 5).astype(float)
    d["is_flight"] = (d.discharge_min.fillna(0) >= 3).astype(float)
    d["v_mean_dis"] = 1000.0 * d.kwh_out / d.ah_out.where(d.ah_out >= 3) / 96.0
    d["sqrt_years"] = np.sqrt((d.t_days + ctx.cfg["label"]["tau0_days"]) / 365.25)
    d["efc_cum"] = d.ah_cum / 26.0
    g = d.groupby(["aircraft_label", "battery"], sort=False)
    d["gap_days"] = g.t_days.diff().fillna(0.0)
    d["n_prior"] = g.cumcount()
    ahout = d.ah_out.fillna(0.0)
    d["hot_ah_cum"] = ahout.where(d.temp_mean_c > 30, 0.0).groupby([d.aircraft_label, d.battery]).cumsum().groupby([d.aircraft_label, d.battery]).shift(1).fillna(0.0)
    d["cold_ah_cum"] = ahout.where(d.temp_mean_c < 10, 0.0).groupby([d.aircraft_label, d.battery]).cumsum().groupby([d.aircraft_label, d.battery]).shift(1).fillna(0.0)
    d["temp_hist_mean"] = g.temp_mean_c.transform(lambda s: s.expanding().mean().shift(1))
    d["depth_hist_mean"] = d.ah_out.where(d.ah_out >= 3).groupby([d.aircraft_label, d.battery]).transform(lambda s: s.expanding().mean().shift(1))
    d["rest_v_hist_mean"] = g.rest_start_v.transform(lambda s: s.expanding().mean().shift(1))
    return d


def feature_list(cfg: dict, run: str, df: pd.DataFrame) -> list[str]:
    cols = [c for gr in cfg["runs"][run] for c in GROUPS[gr]]
    return [c for c in dict.fromkeys(cols) if c in df and df[c].notna().any()]


class LabelEngine:
    """Recomputes labels from data before a cutoff, using hyperparameters fitted on the fit fleet."""

    def __init__(self, ctx: Ctx):
        self.lc = ctx.cfg["label"]
        self.hyper = json.loads((ctx.cache / "label_hyper.json").read_text())
        self.h = Hyper(self.hyper["a"], self.hyper["b"], self.hyper["q"])
        self.params = self.hyper["params"]
        self.mods = self.hyper["mods"]
        obs = pd.read_parquet(ctx.cache / "session_observables_r.parquet")
        self.grids = build_grid(obs)
        d = pd.read_parquet(ctx.cache / "observations_calibrated.parquet")
        self.d = {k: g for k, g in d.groupby(["aircraft_label", "battery"], sort=False)}
        p = ctx.tables / "L2_detected_steps.csv"
        st = pd.read_csv(p, parse_dates=["when"]) if p.exists() and p.stat().st_size > 2 else pd.DataFrame()
        self.steps = {k: g for k, g in st.groupby(["aircraft_label", "battery"], sort=False)} if len(st) else {}
        self._cache = {}

    def upto(self, key: tuple, cutoff: pd.Timestamp) -> pd.DataFrame:
        ck = (key, cutoff)
        if ck in self._cache:
            return self._cache[ck]
        grid = self.grids[key]
        g = grid[grid.start_datetime <= cutoff].reset_index(drop=True)
        if not len(g) or key not in self.d:
            return pd.DataFrame()
        dp = self.d[key]
        dp = dp[dp.flight_id.isin(g.flight_id)]
        tau0 = self.lc["tau0_days"]
        extra = np.zeros(len(g))
        if key in self.steps:
            pos = {f: i for i, f in enumerate(g.flight_id)}
            for r in self.steps[key].itertuples():
                if r.flight_id in pos and r.when <= cutoff - pd.Timedelta(days=30):   # a step is only knowable once sessions have followed it
                    extra[pos[r.flight_id]] = (2 * r.delta) ** 2
        s, _ = robust_pass(g, dp, self.params, self.mods, self.h, tau0, self.lc["t_nu"], 2, extra)
        xf, pf, xs, ps, _ = kalman(s, self.h, tau0)
        prec = 1.0 / ps
        lp = np.maximum(prec - s.lam, 1e-3)
        out = pd.DataFrame(dict(flight_id=g.flight_id, t_days=g.t_days, start_datetime=g.start_datetime, ah_cum=g.ah_cum, x_s=xs, sd_s=np.sqrt(ps), x_f=xf, sd_f=np.sqrt(pf),
                                x_loo=(xs * prec - s.eta) / lp, sd_loo=np.sqrt(1 / lp), has_obs=s.lam > 0))
        self._cache[ck] = out
        return out


def keys_of(df: pd.DataFrame, fleet, min_sessions: int) -> list[tuple]:
    g = df[df.fleet.astype(str) == str(fleet)].groupby(["aircraft_label", "battery"]).lam.apply(lambda s: int((s > 0).sum()))
    return [k for k, n in g.items() if n >= min_sessions]


def impute(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]):
    """Median fill plus missing flags, for models that cannot take NaN."""
    med = train[cols].median()
    a, b = train[cols].fillna(med).fillna(0.0).to_numpy(float), test[cols].fillna(med).fillna(0.0).to_numpy(float)
    return a, b
