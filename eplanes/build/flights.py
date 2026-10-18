"""Flight-level metadata table (one row per flight) from the saved portal pages."""
from __future__ import annotations

import json

import pandas as pd

from ..acquisition.portal import parse_date
from ..config import Settings


def normalise_type(raw: str) -> str:
    t = (raw or "").lower()
    if "ground test" in t and "charging" in t:
        return "ground test and charging"
    if "flight" in t and "charging" in t:
        return "flight test and charging"
    if "charging" in t:
        return "charging"
    if "ground test" in t:
        return "ground test"
    if "flight" in t:
        return "flight test"
    return t or "unknown"


def build_flights_table(settings: Settings) -> pd.DataFrame:
    man = pd.read_csv(settings.data_dir / "raw" / "manifest.csv")
    rows = []
    for r in man.itertuples():
        rec = dict(flight_id=int(r.flight_id), account=r.account, aircraft_id=r.aircraft_id,
                   aircraft_label=r.aircraft_label, fleet=r.account.replace("fleet_", ""),
                   download_status=r.status, zip_sha256=r.sha256)
        if isinstance(r.meta_path, str) and r.meta_path:
            try:
                info = json.load(open(r.meta_path))["page_info"]
            except Exception:  # noqa: BLE001
                info = {}
            dt = parse_date(info.get("Date", ""))
            rec.update(date_raw=info.get("Date", ""), start_datetime=dt,
                       flight_type_raw=info.get("Flight type", ""),
                       flight_type=normalise_type(info.get("Flight type", "")))
            for k, v in info.items():
                rec[f"page_{k.strip().lower().replace(' ', '_')}"] = v
        rows.append(rec)
    df = pd.DataFrame(rows)
    df["start_datetime"] = pd.to_datetime(df.get("start_datetime"), errors="coerce")
    df["flight_date"] = df["start_datetime"].dt.normalize()
    return df.sort_values(["aircraft_label", "start_datetime", "flight_id"]).reset_index(drop=True)
