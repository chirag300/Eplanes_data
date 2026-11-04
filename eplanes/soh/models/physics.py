"""Physics and statistical forecasters. Units: relative capacity x (1.0 = baseline), time in days, charge in Ah."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls


def phi(df: pd.DataFrame, tau0: float) -> tuple[np.ndarray, np.ndarray]:
    return np.sqrt((df.t_days.to_numpy(float) + tau0) / 365.25), df.ah_cum.to_numpy(float) / 1000.0


class Persistence:
    name = "persistence"

    def fit(self, train):
        return self

    def predict(self, hist, fut):
        return np.full(len(fut), hist.x_f.iloc[-1]), np.full(len(fut), hist.sd_f.iloc[-1])


class FleetSlope:
    """Anchor at the pack's causal state, then the mean fade rate of the training packs."""
    name = "fleet_slope"

    def fit(self, train):
        rates = []
        for h in train.values():
            if h.t_days.iloc[-1] - h.t_days.iloc[0] >= 60:
                rates.append(np.polyfit(h.t_days, h.x_s, 1)[0])
        self.rate = float(np.median(rates)) if rates else 0.0
        return self

    def predict(self, hist, fut):
        dt = fut.t_days.to_numpy(float) - hist.t_days.iloc[-1]
        return hist.x_f.iloc[-1] + self.rate * dt, np.full(len(fut), hist.sd_f.iloc[-1])


class LinearPack:
    """Straight line through the pack's own recent history."""
    name = "linear_pack"

    def __init__(self, window_days=270):
        self.window = window_days

    def fit(self, train):
        self.fallback = FleetSlope().fit(train).rate
        return self

    def predict(self, hist, fut):
        h = hist[hist.t_days >= hist.t_days.iloc[-1] - self.window]
        rate = self.fallback
        if len(h) >= 15 and np.ptp(h.t_days) >= 60:
            rate = float(np.polyfit(h.t_days, h.x_s, 1)[0])
        dt = fut.t_days.to_numpy(float) - hist.t_days.iloc[-1]
        return hist.x_f.iloc[-1] + rate * dt, np.full(len(fut), hist.sd_f.iloc[-1])


class PhysicsODE:
    """x(t) = x(t0) - a [sqrt(tau) - sqrt(tau0)] - b [Ah - Ah0], with a and b fitted on the training packs."""
    name = "physics_ode"

    def __init__(self, tau0):
        self.tau0 = tau0

    def fit(self, train):
        """Least squares with one offset per pack. a and b stay non-negative: in one aircraft the two usage terms are nearly collinear,
        so an unconstrained fit can return a negative rate that has no physical meaning."""
        rows = []
        for k, h in train.items():
            r, ah = phi(h, self.tau0)
            rows.append((r - r[0], ah - ah[0], h.x_s.to_numpy(float)))
        off = [y.mean() for _, _, y in rows]
        coef = np.zeros(2)
        for _ in range(8):
            A = np.vstack([np.c_[r, ah] for r, ah, _ in rows])
            target = np.concatenate([o - y for o, (_, _, y) in zip(off, rows)])
            coef, _ = nnls(A, target)
            off = [float(np.mean(y + coef[0] * r + coef[1] * ah)) for r, ah, y in rows]
        self.a, self.b = float(coef[0]), float(coef[1])
        return self

    def rate_path(self, df):
        r, ah = phi(df, self.tau0)
        return self.a * r + self.b * ah

    def predict(self, hist, fut):
        p0 = self.rate_path(hist.iloc[[-1]])[0]
        return hist.x_f.iloc[-1] - (self.rate_path(fut) - p0), np.full(len(fut), hist.sd_f.iloc[-1])


class PhysicsBayes(PhysicsODE):
    """Physics ODE with a pack-specific rate multiplier m ~ N(1, tau^2), updated from the pack's own history.

    A new pack starts at the fleet law and earns its own rate as evidence arrives, which is the sensible behaviour for one
    aircraft's worth of training data."""
    name = "physics_bayes"

    def __init__(self, tau0, prior_sd=0.5, noise_sd=0.006):
        super().__init__(tau0)
        self.prior_sd, self.noise_sd = prior_sd, noise_sd

    def posterior(self, hist):
        h = hist.iloc[::max(1, len(hist) // 40)]
        p = self.rate_path(h)
        x = p - p[0]
        y = h.x_s.to_numpy(float)
        n = len(h)
        A = np.c_[np.ones(n), -x]
        P0 = np.diag([1e4, 1.0 / self.prior_sd ** 2])
        m0 = np.array([y[0], 1.0])
        S = P0 + A.T @ A / self.noise_sd ** 2
        cov = np.linalg.inv(S)
        mean = cov @ (P0 @ m0 + A.T @ y / self.noise_sd ** 2)
        return float(mean[1]), float(np.sqrt(cov[1, 1]))

    def predict(self, hist, fut):
        m, sd = self.posterior(hist)
        p0 = self.rate_path(hist.iloc[[-1]])[0]
        dp = self.rate_path(fut) - p0
        return hist.x_f.iloc[-1] - m * dp, np.hypot(hist.sd_f.iloc[-1], sd * dp)
