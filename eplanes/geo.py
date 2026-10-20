"""GPS handling: hemisphere recovery, airport matching and track summaries.

The Velis Electro avionics log LAT / LNG as unsigned magnitudes: a session on the apron of
Region of Waterloo International Airport (43.46 N, 80.38 W) is logged as LAT 43.46, LNG 80.38.
The sign of each axis is therefore recovered from the data, never assumed: for every aircraft the
four sign combinations are scored by how many session start/end fixes fall within `match_km`
of a known airfield (OurAirports, public domain), and the best combination is kept only when it
clearly wins. Aircraft whose sign cannot be established get no map coordinates.

Airport identifiers are only reported when a fix lies within `match_km` of the airfield's
reference point; everything else is "unmatched", not guessed.
"""
from __future__ import annotations

import gzip
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

AIRPORTS_FILE = Path(__file__).parent / "data" / "ourairports_airports.csv.gz"
EARTH_KM = 6371.0088
SIGN_COMBOS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


@lru_cache(maxsize=1)
def load_airports() -> pd.DataFrame:
    """Small, medium and large airports from OurAirports (https://ourairports.com/data/, public domain)."""
    with gzip.open(AIRPORTS_FILE, "rt", encoding="utf-8") as fh:
        a = pd.read_csv(fh, keep_default_na=False, na_values=[""])
    a = a.dropna(subset=["latitude_deg", "longitude_deg"]).reset_index(drop=True)
    a["code"] = a["icao_code"].fillna(a["gps_code"]).fillna(a["ident"])
    return a


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, float)) for x in (lat1, lon1, lat2, lon2))
    d = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(d, 0, 1)))


def nearest_airport(lat: float, lon: float, max_km: float = 3.0, airports: pd.DataFrame | None = None):
    """(code, name, distance_km) of the closest airfield within max_km, else (None, None, nan)."""
    if lat is None or lon is None or not np.isfinite(lat) or not np.isfinite(lon):
        return None, None, np.nan
    a = load_airports() if airports is None else airports
    box = a[(a.latitude_deg.sub(lat).abs() < 0.5) & (a.longitude_deg.sub(lon).abs() < 0.8)]
    if box.empty:
        return None, None, np.nan
    d = haversine_km(lat, lon, box.latitude_deg.to_numpy(), box.longitude_deg.to_numpy())
    i = int(np.argmin(d))
    if d[i] > max_km:
        return None, None, float(d[i])
    r = box.iloc[i]
    return str(r["code"]), str(r["name"]), float(d[i])


def valid_fix_mask(lat, lng, avionics_valid=None):
    lat = np.asarray(lat, float)
    lng = np.asarray(lng, float)
    ok = np.isfinite(lat) & np.isfinite(lng) & (np.abs(lat) > 0.01) & (np.abs(lng) > 0.01) & (np.abs(lat) <= 90) & (np.abs(lng) <= 180)
    if avionics_valid is not None:
        ok &= np.asarray(avionics_valid, bool)
    return ok


def resolve_signs(endpoints: pd.DataFrame, match_km: float = 3.0, min_matches: int = 3) -> pd.DataFrame:
    """Per aircraft, choose the LAT/LNG sign combination under which session endpoints land on airfields.

    `endpoints`: aircraft_label, lat, lng (raw logged values; one row per session start or end fix).
    Returns aircraft_label, lat_sign, lng_sign, n_fixes, n_matched, runner_up_matched, resolved, method.
    """
    rows = []
    a = load_airports()
    for ac, g in endpoints.groupby("aircraft_label"):
        g = g.dropna(subset=["lat", "lng"])
        if len(g) > 400:
            g = g.sample(400, random_state=0)
        scores = []
        for sl, sg in SIGN_COMBOS:
            hits = 0
            for la, lo in zip(g.lat.to_numpy(), g.lng.to_numpy()):
                code, _, _ = nearest_airport(sl * abs(la), sg * abs(lo), match_km, a)
                hits += code is not None
            scores.append(((sl, sg), hits))
        scores.sort(key=lambda x: -x[1])
        (best, n_best), (_, n_next) = scores[0], scores[1]
        resolved = n_best >= min_matches and n_best >= 2 * max(n_next, 1)
        rows.append(dict(aircraft_label=ac, lat_sign=best[0] if resolved else np.nan, lng_sign=best[1] if resolved else np.nan,
                         n_fixes=len(g), n_matched=n_best, runner_up_matched=n_next, resolved=bool(resolved),
                         method=f"session endpoints matched to OurAirports airfields within {match_km:g} km"))
    return pd.DataFrame(rows)


def track_summary(lat, lon, alt=None, t_s=None) -> dict:
    """Length, extent and endpoints of an already sign-corrected track (NaN-free arrays)."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    if len(lat) < 2:
        return dict(track_km=0.0, max_range_km=0.0)
    step = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
    if t_s is not None:
        dt = np.diff(np.asarray(t_s, float))
        speed = step / np.maximum(dt, 1e-3) * 3600
        step = np.where(speed > 600, 0.0, step)  # drop GPS jumps faster than 600 km/h (Velis cruise ~ 170 km/h)
    rng = haversine_km(lat[0], lon[0], lat, lon)
    out = dict(track_km=float(step.sum()), max_range_km=float(np.nanmax(rng)))
    if alt is not None:
        alt = np.asarray(alt, float)
        if np.isfinite(alt).any():
            out["alt_range_m"] = float(np.nanmax(alt) - np.nanmin(alt))
    return out
