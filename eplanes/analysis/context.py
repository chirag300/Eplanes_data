"""Shared loading/aggregation helpers for the analysis stages."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Settings
from .features import DEFAULT_RULE, extract_all


def season_of(month: int, seasons: dict) -> str:
    for k, ms in seasons.items():
        if month in ms:
            return k
    return "unknown"


def attach_meta(settings: Settings, feat: pd.DataFrame) -> pd.DataFrame:
    """Join a battery-session feature table with flight metadata and the quality-stage usability flags."""
    proc = settings.data_dir / "processed"
    fl = pd.read_csv(proc / "flights_with_qc.csv", parse_dates=["start_datetime", "flight_date"])
    use = pd.read_csv(settings.outputs_dir / "quality" / "usability_matrix_by_flight.csv")
    keep = ["flight_id", "aircraft_label", "fleet", "flight_type", "start_datetime", "flight_date",
            "start_state", "n_gaps", "max_gap_min", "is_duplicate_content", "activity_from_data", "label_content_mismatch"]
    df = feat.merge(fl[keep], on="flight_id", how="left").merge(
        use.drop(columns=["aircraft_label", "flight_type"]), on="flight_id", how="left")
    df["month"] = df.start_datetime.dt.month
    df["season"] = df.month.map(lambda m: season_of(m, settings["analysis"]["seasons"]) if pd.notna(m) else np.nan)
    df["battery"] = df["battery"].astype(int)
    return df.sort_values(["aircraft_label", "battery", "start_datetime", "flight_id"]).reset_index(drop=True)


def load_context(settings: Settings, rule: str = DEFAULT_RULE, recompute: bool = False) -> pd.DataFrame:
    """Battery-session feature table joined with flight metadata and usability flags."""
    proc = settings.data_dir / "processed"
    fp = proc / f"features_battery_session__{rule.replace(':', '_')}.csv"
    if fp.exists() and not recompute:
        feat = pd.read_csv(fp)
    else:
        feat = extract_all(settings, rule)
        feat.to_csv(fp, index=False)
    return attach_meta(settings, feat)


def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def boot_ci(x, stat=np.median, reps=2000, seed=0, alpha=0.05):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = [stat(rng.choice(x, len(x), replace=True)) for _ in range(reps)]
    return float(np.percentile(bs, 100 * alpha / 2)), float(np.percentile(bs, 100 * (1 - alpha / 2)))


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i]))
        out[i] = run
    return out


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    if len(ok) == 0:
        return out
    order = ok[np.argsort(p[ok])]
    m = len(order)
    q = p[order] * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out[order] = np.minimum(q, 1.0)
    return out
