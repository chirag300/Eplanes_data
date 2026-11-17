"""Hybrid models on tabular features: physics forecast + learned residual, and session-level nowcasters."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .physics import PhysicsODE, phi

ORIGIN_FEATURES = ["h_days", "x0", "slope_90d", "d_sqrt", "d_ah", "temp_hist", "temp_fut", "depth_hist", "depth_fut", "r25_recent", "phys_delta"]


def origin_features(hist, fut, sess, phys: PhysicsODE) -> pd.DataFrame:
    """Features known at a forecast origin, plus the planned usage over the horizon."""
    t0 = hist.t_days.iloc[-1]
    rec = hist[hist.t_days >= t0 - 90]
    slope = float(np.polyfit(rec.t_days, rec.x_s, 1)[0]) * 365.25 if len(rec) >= 8 and np.ptp(rec.t_days) >= 20 else np.nan
    p0 = phys.rate_path(hist.iloc[[-1]])[0]
    r0, ah0 = phi(hist.iloc[[-1]], phys.tau0)
    r, ah = phi(fut, phys.tau0)
    past = sess[sess.t_days <= t0]
    cum = sess[sess.t_days > t0]
    out = pd.DataFrame(dict(h_days=fut.t_days.to_numpy(float) - t0, x0=hist.x_f.iloc[-1], slope_90d=slope, d_sqrt=r - r0[0], d_ah=ah - ah0[0], phys_delta=phys.rate_path(fut) - p0))
    out["temp_hist"] = past.temp_mean_c.tail(60).mean()
    out["depth_hist"] = past.ah_out.where(past.ah_out >= 3).tail(60).mean()
    out["r25_recent"] = past.r_step25_mohm.tail(30).median() if "r_step25_mohm" in past else np.nan
    tf, df_ = [], []
    tv, av = cum.t_days.to_numpy(), cum.ah_out.to_numpy(float)
    te = cum.temp_mean_c.to_numpy(float)
    for t in fut.t_days.to_numpy(float):
        m = tv <= t
        tf.append(np.nanmean(te[m]) if m.any() else np.nan)
        a = av[m]
        a = a[a >= 3]
        df_.append(a.mean() if len(a) else np.nan)
    out["temp_fut"], out["depth_fut"] = tf, df_
    return out[ORIGIN_FEATURES]


class ResidualBase:
    """Physics forecast + a model of its error, learned on past forecasts made on the training packs."""
    step_days = 15

    def __init__(self, tau0, sessions: dict, horizons):
        self.tau0, self.sessions, self.horizons = tau0, sessions, horizons

    def training_set(self, train):
        X, y = [], []
        for k, h in train.items():
            sess = self.sessions[k]
            t = h.t_days.to_numpy(float)
            for t0 in np.arange(max(60.0, t[0] + 60), t[-1], self.step_days):
                i = int(np.searchsorted(t, t0, side="right"))
                if i < 10:
                    continue
                hh = h.iloc[:i]
                targets = []
                for hz in self.horizons:
                    j = int(np.searchsorted(t, t0 + hz))
                    if j < len(t) and t[j] - (t0 + hz) <= 15:
                        targets.append(j)
                if not targets:
                    continue
                fut = h.iloc[targets]
                f = origin_features(hh, fut, sess, self.phys)
                base = hh.x_s.iloc[-1] - (self.phys.rate_path(fut) - self.phys.rate_path(hh.iloc[[-1]])[0])
                X.append(f)
                y.append(fut.x_s.to_numpy(float) - base)
        if not X:
            return pd.DataFrame(columns=ORIGIN_FEATURES), np.array([])
        return pd.concat(X, ignore_index=True), np.concatenate(y)

    def fit(self, train):
        self.phys = PhysicsODE(self.tau0).fit(train)
        X, y = self.training_set(train)
        self.n_train = len(y)
        self._fit(X, y)
        return self

    def predict(self, hist, fut):
        base, sd = self.phys.predict(hist, fut)
        k = (hist.aircraft_label.iloc[0], int(hist.battery.iloc[0]))
        f = origin_features(hist, fut, self.sessions[k], self.phys)
        r, rs = self._pred(f)
        return base + r, np.hypot(sd, rs)


class ResidualGBM(ResidualBase):
    name = "residual_gbm"

    def _fit(self, X, y):
        self.m = None
        if len(y) >= 40:
            mono = [0] * len(ORIGIN_FEATURES)
            self.m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.03, max_iter=150, min_samples_leaf=15, l2_regularization=2.0, monotonic_cst=mono, random_state=0).fit(X, y)
            self.s = float(np.std(y - self.m.predict(X)))

    def _pred(self, f):
        if self.m is None:
            return np.zeros(len(f)), np.zeros(len(f))
        return self.m.predict(f), np.full(len(f), self.s)


class ResidualGP(ResidualBase):
    name = "residual_gp"
    use = ["h_days", "d_sqrt", "d_ah", "slope_90d", "temp_hist", "depth_hist"]

    def _fit(self, X, y):
        self.m = None
        if len(y) >= 40:
            rng = np.random.default_rng(0)
            if len(y) > 500:
                i = rng.choice(len(y), 500, replace=False)
                X, y = X.iloc[i], y[i]
            self.med = X[self.use].median()
            A = X[self.use].fillna(self.med).fillna(0).to_numpy()
            kern = ConstantKernel(1e-4, (1e-7, 1e-2)) * RBF(np.ones(len(self.use)), (0.5, 50)) + WhiteKernel(1e-5, (1e-7, 1e-3))
            self.m = make_pipeline(StandardScaler(), GaussianProcessRegressor(kern, normalize_y=False, n_restarts_optimizer=1, random_state=0))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.m.fit(A, y)

    def _pred(self, f):
        if self.m is None:
            return np.zeros(len(f)), np.zeros(len(f))
        A = f[self.use].fillna(self.med).fillna(0).to_numpy()
        sc, gp = self.m[0], self.m[1]
        return gp.predict(sc.transform(A), return_std=True)


# --------------------------------------------------------------------------------- nowcasters (session -> capacity now)
class MeanNow:
    name = "mean"

    def fit(self, X, y, w, cols):
        self.v = float(np.average(y, weights=w))
        return self

    def predict(self, X):
        return np.full(len(X), self.v)


class RidgeNow:
    name = "ridge"

    def fit(self, X, y, w, cols):
        self.med = np.nanmedian(X, axis=0)
        self.med = np.where(np.isfinite(self.med), self.med, 0.0)
        A = np.where(np.isfinite(X), X, self.med)
        self.m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))
        self.m.fit(A, y, ridgecv__sample_weight=w)
        return self

    def predict(self, X):
        return self.m.predict(np.where(np.isfinite(X), X, self.med))


class GBMNow:
    """Monotone where physics gives a sign: more resistance or more usage can only mean less capacity."""
    name = "gbm_monotone"
    decreasing = {"r_step25_mohm", "r_step3025_mohm", "r_ramp25_mohm", "t_days", "sqrt_years", "ah_cum", "efc_cum", "hot_ah_cum", "cold_ah_cum", "n_prior"}

    def fit(self, X, y, w, cols):
        cst = [-1 if c in self.decreasing else 0 for c in cols]
        self.m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=20, l2_regularization=2.0, monotonic_cst=cst, random_state=0)
        self.m.fit(X, y, sample_weight=w)
        return self

    def predict(self, X):
        return self.m.predict(X)


READINGS = ["y_chg", "y_r2r"]   # the flight-load reading is biased by resistance growth on real data; it stays a feature, not a reading


def reading_of(X, cols):
    """The session's own physics capacity reading in points (median of the readings it has), NaN if it has none."""
    idx = [cols.index(c) for c in READINGS if c in cols]
    if not idx:
        return np.full(len(X), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return 100 * np.nanmedian(X[:, idx], axis=1)


class PhysicsReadingNow:
    """No learning: the session's own capacity reading. What physics alone gives for one session."""
    name = "physics_reading"

    def fit(self, X, y, w, cols):
        self.cols, self.fallback = list(cols), float(np.average(y, weights=w))
        return self

    def predict(self, X):
        r = reading_of(X, self.cols)
        return np.where(np.isfinite(r), r, self.fallback)


class ResidualGBMNow:
    """Hybrid nowcast: physics reading + boosted correction. Trees cannot extrapolate below the capacities they have seen, a fatal
    flaw for a fading battery, but their correction to a physics reading has no such limit."""
    name = "residual_gbm"

    def fit(self, X, y, w, cols):
        self.cols = list(cols)
        r = reading_of(X, self.cols)
        ok = np.isfinite(r)
        self.fallback = float(np.average(y, weights=w))
        self.m = None
        if ok.sum() >= 40:
            self.m = HistGradientBoostingRegressor(max_depth=2, learning_rate=0.03, max_iter=150, min_samples_leaf=25, l2_regularization=5.0, random_state=0)
            self.m.fit(X[ok], (y - r)[ok], sample_weight=w[ok])
        return self

    def predict(self, X):
        r = reading_of(X, self.cols)
        c = self.m.predict(X) if self.m is not None else np.zeros(len(X))
        return np.where(np.isfinite(r), r + c, self.fallback)


class GPNow:
    name = "gp"

    def fit(self, X, y, w, cols):
        rng = np.random.default_rng(0)
        i = rng.choice(len(y), min(len(y), 400), replace=False)
        self.med = np.nanmedian(X, axis=0)
        self.med = np.where(np.isfinite(self.med), self.med, 0.0)
        A = np.where(np.isfinite(X), X, self.med)[i]
        keep = np.std(A, axis=0) > 0
        self.keep = keep
        self.mean = float(np.average(y, weights=w))
        kern = ConstantKernel(1e-3, (1e-6, 1e-1)) * RBF(np.full(keep.sum(), 5.0), (0.5, 200)) + WhiteKernel(1e-4, (1e-7, 1e-1))
        self.m = make_pipeline(StandardScaler(), GaussianProcessRegressor(kern, random_state=0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.m.fit(A[:, keep], y[i] - self.mean)
        return self

    def predict(self, X):
        return self.mean + self.m.predict(np.where(np.isfinite(X), X, self.med)[:, self.keep])
