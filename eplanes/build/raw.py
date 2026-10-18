"""Combined RAW dataset: every flight, every column, values untouched.

Outputs (data/raw_combined/):
  combined_raw_timeseries.parquet   main CSV of every flight, + flight_id/aircraft_label/account/row_in_file
  combined_raw_aux_<kind>.parquet   '_warns', '_1', '_2' (and any other) CSV members, all values as strings
  zip_members.csv                   inventory of every ZIP member (name, kind, bytes, rows, columns)
  column_map.csv                    original header -> canonical name
  coercion_log.csv                  values that were not numeric in numeric columns (kept as NaN, counted here)
Nothing is dropped, rounded or unit-converted in this stage.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..config import RAW_TO_CANON, Settings

ID_COLS = ["flight_id", "aircraft_label", "account", "row_in_file"]


def slugify(col: str) -> str:
    c = col.strip().lower()
    if c in RAW_TO_CANON:
        return RAW_TO_CANON[c]
    return re.sub(r"[^a-z0-9]+", "_", c).strip("_") or "unnamed"


def member_kind(name: str) -> str:
    stem = Path(name).stem
    if not name.lower().endswith(".csv"):
        return "other"
    if stem.endswith("_warns"):
        return "warns"
    if re.search(r"_(\d+)$", stem):
        return "sub_" + re.search(r"_(\d+)$", stem).group(1)
    return "main"


def _read_member(z: zipfile.ZipFile, name: str, as_str=False) -> pd.DataFrame:
    with z.open(name) as f:
        df = pd.read_csv(f, low_memory=False, dtype=str if as_str else None)
    df.columns = [c.strip() for c in df.columns]
    return df


def build_raw(settings: Settings, log=print) -> Path:
    raw_dir = settings.data_dir / "raw"
    out_dir = settings.data_dir / "raw_combined"
    tmp = out_dir / "_per_flight"
    tmp.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(raw_dir / "manifest.csv")
    man = man[man.status == "ok"].sort_values("flight_id")

    inventory, colmap, coerce = [], {}, []
    aux_frames: dict[str, list[pd.DataFrame]] = {}
    schemas = []
    for r in man.itertuples():
        pf = tmp / f"flight_{r.flight_id}.parquet"
        with zipfile.ZipFile(r.zip_path) as z:
            main_name = None
            for n in z.namelist():
                kind = member_kind(n)
                info = z.getinfo(n)
                rows = cols = None
                if kind != "other":
                    d = _read_member(z, n, as_str=(kind != "main"))
                    rows, cols = len(d), "|".join(d.columns)
                    if kind == "main" and main_name is None:
                        main_name = n
                        df = d
                    elif kind != "main":
                        d.insert(0, "flight_id", r.flight_id)
                        d.insert(1, "aircraft_label", r.aircraft_label)
                        d.insert(2, "source_member", n)
                        aux_frames.setdefault(kind, []).append(d)
                inventory.append(dict(flight_id=r.flight_id, aircraft_label=r.aircraft_label,
                                      member=n, kind=kind, bytes=info.file_size, rows=rows, columns=cols))
        if main_name is None:
            inventory.append(dict(flight_id=r.flight_id, aircraft_label=r.aircraft_label,
                                  member="<no main csv>", kind="MISSING", bytes=0, rows=0, columns=""))
            continue
        orig = list(df.columns)
        canon = [slugify(c) for c in orig]
        for o, c in zip(orig, canon):
            colmap[o] = c
        df.columns = canon
        for c in canon:
            if df[c].dtype == object:
                num = pd.to_numeric(df[c], errors="coerce")
                bad = int((num.isna() & df[c].notna()).sum())
                if bad:
                    coerce.append(dict(flight_id=r.flight_id, column=c, n_non_numeric=bad,
                                       examples="|".join(map(str, df.loc[num.isna() & df[c].notna(), c].unique()[:5]))))
                df[c] = num
        df.insert(0, "row_in_file", range(len(df)))
        df.insert(0, "account", r.account)
        df.insert(0, "aircraft_label", r.aircraft_label)
        df.insert(0, "flight_id", int(r.flight_id))
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, pf)
        schemas.append(table.schema)

    unified = pa.unify_schemas(schemas, promote_options="permissive")
    out = out_dir / "combined_raw_timeseries.parquet"
    with pq.ParquetWriter(out, unified) as w:
        for r in man.itertuples():
            pf = tmp / f"flight_{r.flight_id}.parquet"
            if pf.exists():
                t = pq.read_table(pf)
                for f in unified:
                    if f.name not in t.schema.names:
                        t = t.append_column(f, pa.nulls(t.num_rows, f.type))
                w.write_table(t.select(unified.names).cast(unified))
    for kind, frames in aux_frames.items():
        pd.concat(frames, ignore_index=True).to_parquet(out_dir / f"combined_raw_aux_{kind}.parquet", index=False)
    pd.DataFrame(inventory).to_csv(out_dir / "zip_members.csv", index=False)
    pd.DataFrame([dict(original_header=k, canonical=v) for k, v in colmap.items()]).to_csv(out_dir / "column_map.csv", index=False)
    pd.DataFrame(coerce, columns=["flight_id", "column", "n_non_numeric", "examples"]).to_csv(out_dir / "coercion_log.csv", index=False)
    for f in tmp.glob("*.parquet"):
        f.unlink()
    tmp.rmdir()
    log(f"raw combined: {len(man)} flights -> {out}")
    return out
