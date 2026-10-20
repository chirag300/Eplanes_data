"""Sessions listed under more than one aircraft.

The portal can list the same log under two aircraft, even in different accounts. The generic duplicate rule keeps
the copy with the lowest flight_id, which can attribute the log to the wrong aircraft and put a spurious excursion
into that aircraft's SOH history.

For every group of identical sessions that spans aircraft, :func:`resolve` keeps the copy whose reported SOH agrees
best with the same aircraft's other sessions within +-``window_days``; reported SOH changes slowly within an aircraft
but differs by several points between aircraft. A copy that cannot be scored (no SOH, or no neighbouring sessions)
scores +inf. If no copy can be scored, or the best two are within one point, no copy is kept: the session is
excluded rather than attributed by guesswork.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def resolve(df: pd.DataFrame, key: str = "content_hash", soh: str = "soh", time: str = "start_datetime",
            window_days: float = 60.0) -> pd.DataFrame:
    """Return one row per member of a cross-aircraft group: flight_id, group, aircraft_label, score, keep."""
    need = df[df[key].notna()]
    groups = need.groupby(key).filter(lambda g: g.aircraft_label.nunique() > 1)
    if groups.empty:
        return pd.DataFrame(columns=["flight_id", "group", "aircraft_label", "score", "keep"])
    members = set(groups.flight_id)
    pool = df[~df.flight_id.isin(members) & df[soh].notna() & df[time].notna()]
    win = pd.Timedelta(days=window_days)
    rows = []
    for gi, (_, g) in enumerate(groups.groupby(key, sort=False)):
        scored = []
        # records, not itertuples: itertuples renames columns that start with an underscore
        for r in g.to_dict("records"):
            t, v = r[time], r[soh]
            nb = pool[(pool.aircraft_label == r["aircraft_label"]) & ((pool[time] - t).abs() <= win)] if pd.notna(t) else pool.iloc[:0]
            score = float(abs(nb[soh].median() - v)) if len(nb) and pd.notna(v) else np.inf
            scored.append((r["flight_id"], r["aircraft_label"], score))
        ranked = sorted(scored, key=lambda x: x[2])
        best = ranked[0]
        decisive = np.isfinite(best[2]) and (len(ranked) == 1 or ranked[1][2] - best[2] >= 1.0)
        for fid, ac, sc in scored:
            rows.append(dict(flight_id=fid, group=gi, aircraft_label=ac, score=sc, keep=bool(decisive and fid == best[0])))
    return pd.DataFrame(rows)


def session_soh(parquet_path, cols=("bat_1_soh", "bat_2_soh")) -> pd.Series:
    """Median non-zero reported SOH per session (mean of the packs), read row group by row group."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(parquet_path)
    have = [c for c in cols if c in pf.schema_arrow.names]
    if "flight_id" not in pf.schema_arrow.names or not have:
        return pd.Series(dtype=float)
    parts = []
    for i in range(pf.num_row_groups):
        d = pf.read_row_group(i, columns=["flight_id", *have]).to_pandas()
        d[have] = d[have].where(d[have] > 0)
        parts.append(d.groupby("flight_id")[have].median())
    m = pd.concat(parts).groupby(level=0).median()
    return m.mean(axis=1)
