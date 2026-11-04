"""Evaluation. Stage 1 runs on the training fleet only and is where models are compared. Stage 2 scores the held-out fleet once.

Forecast: at an origin t0 a model sees the pack's causal state and history up to t0, plus the usage that really followed
          (a conditional forecast: "if the aircraft is used like this"). Target: the full-data label at t0 + horizon.
Nowcast:  features of one session -> leave-one-out label. Ablation runs A to D.
Metrics:  points of SOH. R2 alone misleads on a slow trend, so skill against persistence is reported beside RMSE.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .common import Ctx, Log, write_json
from .dataset import LabelEngine, STRESS, build_table, feature_list, keys_of
from .models.physics import FleetSlope, LinearPack, Persistence, PhysicsBayes, PhysicsODE
from .models.tabular import GBMNow, GPNow, PhysicsReadingNow, READINGS, ResidualGBM, ResidualGBMNow, ResidualGP, RidgeNow

BASE_FORECASTERS = ["persistence", "fleet_slope", "linear_pack", "physics_ode", "physics_bayes"]
HYBRID_FORECASTERS = ["residual_gbm", "residual_gp", "ude", "pinn", "gru", "transformer", "gru_free"]
NOW_MODELS = {"ridge": RidgeNow, "gbm_monotone": GBMNow, "gp": GPNow, "physics_reading": PhysicsReadingNow, "residual_gbm": ResidualGBMNow}
NEEDS_READING = {"physics_reading", "residual_gbm"}


def make_forecasters(names, tau0, sessions, cfg, horizons, seeds=None):
    out = []
    for n in names:
        if n == "persistence":
            out.append(Persistence())
        elif n == "fleet_slope":
            out.append(FleetSlope())
        elif n == "linear_pack":
            out.append(LinearPack())
        elif n == "physics_ode":
            out.append(PhysicsODE(tau0))
        elif n == "physics_bayes":
            out.append(PhysicsBayes(tau0))
        elif n == "residual_gbm":
            out.append(ResidualGBM(tau0, sessions, horizons))
        elif n == "residual_gp":
            out.append(ResidualGP(tau0, sessions, horizons))
        else:
            from .models.neural import PINN, DecrementForecaster
            cols = [c for c in STRESS if any(c in s and s[c].notna().any() for s in sessions.values())]
            out.append(PINN(tau0, sessions, cols, cfg, seeds) if n == "pinn" else DecrementForecaster(n, tau0, sessions, cols, cfg, seeds))
    return out


def forecast_scores(models, engine, targets, sessions, full, origins_for, cut, cfg_d) -> pd.DataFrame:
    rows = []
    for k in targets:
        lab = full[k]
        for t0 in origins_for(k):
            hist = engine.upto(k, t0)
            if len(hist) < 10 or not hist.has_obs.any():
                continue
            hist = hist.assign(aircraft_label=k[0], battery=k[1])
            t0d = hist.t_days.iloc[-1]
            tt = lab.t_days.to_numpy()
            rec = []
            for h in cfg_d["horizons_days"]:
                j = int(np.searchsorted(tt, t0d + h))
                if j < len(tt) and tt[j] - (t0d + h) <= 15:
                    rec.append((h, j))
            if not rec:
                continue
            fut = lab.iloc[[j for _, j in rec]]
            truth = fut.soh_phys.to_numpy(float)
            sd = fut.soh_sd.to_numpy(float)
            for m in models:
                try:
                    p, s = m.predict(hist, fut)
                except Exception as e:  # noqa: BLE001
                    p, s = np.full(len(fut), np.nan), np.full(len(fut), np.nan)
                    if not getattr(m, "_warned", False):
                        print(f"  {m.name} failed on {k}: {type(e).__name__}: {e}")
                        m._warned = True
                for (h, _), y, yhat, ss, lsd in zip(rec, truth, 100 * p, 100 * s, sd):
                    rows.append(dict(model=m.name, aircraft=k[0], pack=k[1], origin=t0, h=h, y=y, pred=yhat, sd=ss, label_sd=lsd, t0_days=t0d))
    return pd.DataFrame(rows)


def summarise_forecast(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    base = df[df.model == "persistence"].set_index(["aircraft", "pack", "origin", "h"]).pred.rename("base")
    d = df.join(base, on=["aircraft", "pack", "origin", "h"])
    d["e"], d["e0"] = d.pred - d.y, d.base - d.y
    rows = []
    for (m, h), g in d.groupby(["model", "h"]):
        g = g.dropna(subset=["e"])
        if not len(g):
            continue
        rmse, rmse0 = float(np.sqrt((g.e ** 2).mean())), float(np.sqrt((g.e0 ** 2).mean()))
        rows.append(dict(model=m, h=h, n=len(g), rmse=rmse, mae=float(g.e.abs().mean()), bias=float(g.e.mean()), skill_vs_persistence=1 - rmse ** 2 / rmse0 ** 2 if rmse0 > 0 else np.nan,
                         cover_1p64sd=float((g.e.abs() <= 1.645 * np.hypot(g.sd, g.label_sd)).mean())))
    return pd.DataFrame(rows).sort_values(["h", "rmse"]).reset_index(drop=True)


def conformal(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Split conformal on the early origins, tested on the later ones. Coverage under drift is reported as it comes out."""
    rows = []
    for (m, h), g in df.dropna(subset=["pred"]).groupby(["model", "h"]):
        g = g.sort_values("origin")
        n = len(g)
        if n < 20:
            continue
        a, b = g.iloc[: n // 2], g.iloc[n // 2:]
        q = np.quantile(np.abs(a.pred - a.y), min(1.0, (1 - alpha) * (1 + 1 / len(a))))
        rows.append(dict(model=m, h=h, n_cal=len(a), n_test=len(b), half_width=float(q), coverage=float((np.abs(b.pred - b.y) <= q).mean()), target=1 - alpha))
    return pd.DataFrame(rows)


def nowcast_cv(table, keys, cuts, cfg, engine, log) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = table[[(a, b) in set(keys) for a, b in zip(table.aircraft_label, table.battery)]]
    res, preds = [], []
    for run in cfg["runs"]:
        cols = feature_list(cfg, run, t)
        for name, cls in NOW_MODELS.items():
            if name in NEEDS_READING and not set(READINGS) & set(cols):
                continue
            ps = []
            for a, b in cuts:
                trs = []
                for k in keys:
                    h = engine.upto(k, a)
                    if len(h):
                        trs.append(h[["flight_id", "x_loo", "sd_loo"]].assign(battery=k[1]))
                tr = pd.concat(trs).merge(t, on=["flight_id", "battery"], how="inner", suffixes=("_cut", ""))
                te = t[(t.start_datetime > a) & (t.start_datetime <= b)]
                if len(tr) < 30 or len(te) < 5:
                    continue
                w = 1.0 / np.maximum(tr.sd_loo_cut.to_numpy(float), 2e-3) ** 2
                m = cls().fit(tr[cols].to_numpy(float), 100 * tr.x_loo_cut.to_numpy(float), w, cols)
                ps.append(te[["aircraft_label", "battery", "flight_id", "soh_loo", "soh_phys"]].assign(pred=m.predict(te[cols].to_numpy(float))))
            if ps:
                p = pd.concat(ps).assign(model=name, run=run)
                preds.append(p)
                res.append(score_now(p, run, name, len(cols)))
    # zero-skill baseline
    ps = []
    for a, b in cuts:
        tr = t[t.start_datetime <= a]
        te = t[(t.start_datetime > a) & (t.start_datetime <= b)]
        if len(tr) >= 30 and len(te) >= 5:
            ps.append(te[["aircraft_label", "battery", "flight_id", "soh_loo", "soh_phys"]].assign(pred=tr.soh_loo.mean()))
    if ps:
        p = pd.concat(ps).assign(model="train_mean", run="-")
        preds.append(p)
        res.append(score_now(p, "-", "train_mean", 0))
    return pd.DataFrame(res), (pd.concat(preds) if preds else pd.DataFrame())


def score_now(p: pd.DataFrame, run: str, name: str, n_feat: int) -> dict:
    p = p.dropna(subset=["soh_loo", "pred"])
    e = p.pred - p.soh_loo
    ss = float(np.sum((p.soh_loo - p.soh_loo.mean()) ** 2))
    sp = []
    for _, g in p.groupby(["aircraft_label", "battery"]):
        if len(g) >= 8 and g.pred.nunique() > 1:
            sp.append(spearmanr(g.pred, g.soh_loo).statistic)
    return dict(run=run, model=name, n=len(p), n_features=n_feat, rmse=float(np.sqrt((e ** 2).mean())), mae=float(e.abs().mean()), bias=float(e.mean()), r2=1 - float(np.sum(e ** 2)) / ss if ss > 0 else np.nan, spearman_within_pack=float(np.nanmean(sp)) if sp else np.nan)


def time_cuts(table, keys, n):
    t = table[[(a, b) in set(keys) for a, b in zip(table.aircraft_label, table.battery)]].start_datetime
    lo, hi = t.min(), t.max()
    span = hi - lo
    q = [lo + span * (0.45 + 0.55 * i / n) for i in range(n + 1)]
    return list(zip(q[:-1], q[1:]))


def prepare(ctx):
    table = build_table(ctx)
    table.to_parquet(ctx.cache / "model_table.parquet", index=False)
    engine = LabelEngine(ctx)
    sessions = {k: g.reset_index(drop=True) for k, g in table.groupby(["aircraft_label", "battery"], sort=False)}
    return table, engine, sessions


def origins_factory(sessions, cfg_d, lo=None, hi=None):
    def f(k):
        s = sessions[k]
        t0 = s.start_datetime.iloc[0] + pd.Timedelta(days=cfg_d["first_origin_days"])
        end = s.start_datetime.iloc[-1] - pd.Timedelta(days=min(cfg_d["horizons_days"]))
        a = max(t0, lo) if lo is not None else t0
        b = min(end, hi) if hi is not None else end
        out = []
        while a <= b:
            out.append(a)
            a = a + pd.Timedelta(days=cfg_d["origin_step_days"])
        return out
    return f


def run_train(ctx: Ctx, log=None, models=None, seeds=None) -> dict:
    """Stage 1: everything inside the training fleet."""
    log = log or Log(ctx)
    cfg, D = ctx.cfg, ctx.cfg["data"]
    table, engine, sessions = prepare(ctx)
    tau0 = cfg["label"]["tau0_days"]
    keys = keys_of(table, D["train_fleet"], D["min_pack_sessions"])
    log(f"[train] training packs: {keys}")
    if not keys:
        raise SystemExit("no training pack has enough measured sessions; see outputs/soh/tables/L3_label_summary.csv")
    cuts = time_cuts(table, keys, D["cv_folds"])
    names = models or (BASE_FORECASTERS + HYBRID_FORECASTERS)
    all_f = []
    t_start = time.time()
    for i, (a, b) in enumerate(cuts):
        train = {k: engine.upto(k, a) for k in keys}
        train = {k: v for k, v in train.items() if len(v) >= 20}
        if not train:
            continue
        ms = []
        for m in make_forecasters(names, tau0, sessions, cfg["models"], D["horizons_days"], seeds):
            t1 = time.time()
            try:
                m.fit(train)
                ms.append(m)
            except Exception as e:  # noqa: BLE001
                log(f"  {m.name} failed to train: {type(e).__name__}: {e}")
            log(f"  fold {i + 1}/{len(cuts)} {m.name}: {time.time() - t1:.1f} s")
        full = {k: table[(table.aircraft_label == k[0]) & (table.battery == k[1])].reset_index(drop=True) for k in keys}
        all_f.append(forecast_scores(ms, engine, keys, sessions, full, origins_factory(sessions, D, a, b), a, D).assign(fold=i))
    f = pd.concat(all_f, ignore_index=True) if all_f else pd.DataFrame()
    f.to_csv(ctx.tables / "M1_cv_forecast_predictions.csv", index=False)
    s = summarise_forecast(f)
    s.to_csv(ctx.tables / "M2_cv_forecast_scores.csv", index=False)
    c = conformal(f, cfg["models"]["conformal_alpha"]) if len(f) else pd.DataFrame()
    c.to_csv(ctx.tables / "M3_cv_conformal.csv", index=False)
    if len(s):
        log("[train] rolling-origin forecast, training fleet (SOH points)\n" + s.round(3).to_string(index=False))
    n, p = nowcast_cv(table, keys, cuts, cfg, engine, log)
    n.to_csv(ctx.tables / "M4_cv_nowcast_ablation.csv", index=False)
    if len(n):
        log("[train] rolling-origin nowcast by ablation run (SOH points)\n" + n.round(3).to_string(index=False))
    from .curves import curve_cv
    cv = curve_cv(ctx, table, keys, cuts, engine, log, seeds)
    if len(cv):
        cv.to_csv(ctx.tables / "M5_cv_curve_cnn.csv", index=False)
    write_json(ctx.out / "train_summary.json", dict(train_packs=[list(k) for k in keys], cuts=[(str(a), str(b)) for a, b in cuts], minutes=(time.time() - t_start) / 60))
    log(f"[train] done in {(time.time() - t_start) / 60:.1f} min")
    return dict(forecast=s, nowcast=n)


def run_final(ctx: Ctx, log=None, force: bool = False, models=None, seeds=None) -> dict:
    """Stage 2: the held-out fleet is scored once. A lock file records the run; a repeat needs --force-rescore and is logged."""
    import json
    import datetime as dt
    import hashlib
    log = log or Log(ctx)
    cfg, D = ctx.cfg, ctx.cfg["data"]
    lock = ctx.out / "FINAL_TEST_LOCK.json"
    state = json.loads(lock.read_text()) if lock.exists() else None
    if state and not force:
        raise SystemExit(f"The held-out fleet was already scored on {state['first_run_utc']}. Scoring it again after seeing results turns it into a "
                         "development set. If you really mean to, pass --force-rescore; the repeat is recorded in the lock file and in the report.")
    table, engine, sessions = prepare(ctx)
    tau0 = cfg["label"]["tau0_days"]
    tr_keys = keys_of(table, D["train_fleet"], D["min_pack_sessions"])
    te_keys = keys_of(table, D["test_fleet"], D["min_pack_sessions"])
    log(f"[final] train {tr_keys} -> test {te_keys}")
    never = table.start_datetime.max() + pd.Timedelta(days=1)
    train = {k: engine.upto(k, never) for k in tr_keys}
    names = models or (BASE_FORECASTERS + HYBRID_FORECASTERS)
    ms = []
    for m in make_forecasters(names, tau0, sessions, cfg["models"], D["horizons_days"], seeds):
        m.fit(train)
        ms.append(m)
    full = {k: table[(table.aircraft_label == k[0]) & (table.battery == k[1])].reset_index(drop=True) for k in te_keys}
    f = forecast_scores(ms, engine, te_keys, sessions, full, origins_factory(sessions, D), never, D)
    f.to_csv(ctx.tables / "T1_final_forecast_predictions.csv", index=False)
    s = summarise_forecast(f)
    s.to_csv(ctx.tables / "T2_final_forecast_scores.csv", index=False)
    if len(s):
        log("[final] cross-fleet forecast (SOH points)\n" + s.round(3).to_string(index=False))
    # nowcast, every ablation run
    tr = table[[(a, b) in set(tr_keys) for a, b in zip(table.aircraft_label, table.battery)]]
    te = table[[(a, b) in set(te_keys) for a, b in zip(table.aircraft_label, table.battery)]]
    res = []
    for run in cfg["runs"]:
        cols = feature_list(cfg, run, tr)
        for name, cls in NOW_MODELS.items():
            if name in NEEDS_READING and not set(READINGS) & set(cols):
                continue
            w = 1.0 / np.maximum(tr.soh_loo_sd.to_numpy(float) / 100.0, 2e-3) ** 2
            m = cls().fit(tr[cols].to_numpy(float), tr.soh_loo.to_numpy(float), w, cols)
            p = te[["aircraft_label", "battery", "flight_id", "soh_loo", "soh_phys"]].assign(pred=m.predict(te[cols].to_numpy(float)))
            res.append(score_now(p, run, name, len(cols)))
    n = pd.DataFrame(res)
    n.to_csv(ctx.tables / "T3_final_nowcast_ablation.csv", index=False)
    if len(n):
        log("[final] cross-fleet nowcast by ablation run (SOH points)\n" + n.round(3).to_string(index=False))
    # leave-one-aircraft-out, secondary: every main aircraft takes a turn as the unseen one, with forecasters that train quickly
    loao = []
    every = keys_of(table, D["train_fleet"], D["min_pack_sessions"]) + te_keys
    quick = [x for x in names if x in BASE_FORECASTERS + ["residual_gbm", "ude"]]
    for ac in sorted({k[0] for k in every}):
        tr2 = {k: engine.upto(k, never) for k in every if k[0] != ac}
        tk = [k for k in every if k[0] == ac]
        if not tr2 or not tk:
            continue
        mm = [m.fit(tr2) for m in make_forecasters(quick, tau0, sessions, cfg["models"], D["horizons_days"], seeds)]
        fl = {k: table[(table.aircraft_label == k[0]) & (table.battery == k[1])].reset_index(drop=True) for k in tk}
        r = summarise_forecast(forecast_scores(mm, engine, tk, sessions, fl, origins_factory(sessions, D), never, D))
        if len(r):
            loao.append(r.assign(unseen_aircraft=ac))
    if loao:
        pd.concat(loao).to_csv(ctx.tables / "T4_final_leave_one_aircraft_out.csv", index=False)
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]
    new = dict(first_run_utc=state["first_run_utc"] if state else now, runs=(state["runs"] if state else 0) + 1, config_hash=cfg_hash, test_packs=[list(k) for k in te_keys],
               rescored_utc=(state.get("rescored_utc", []) + [now]) if state else [])
    write_json(lock, new)
    log(f"[final] lock written: {lock} (run {new['runs']})")
    return dict(forecast=s, nowcast=n)
