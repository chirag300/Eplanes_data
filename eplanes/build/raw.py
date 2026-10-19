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


def _read_member(z: zipfile.ZipFile, name: str, as_str=False, nrows=None) -> pd.DataFrame:
    with z.open(name) as f:
        df = pd.read_csv(f, low_memory=False, dtype=str if as_str else None, nrows=nrows)
    df.columns = [c.strip() for c in df.columns]
    return df


def _numeric_or_keep(df: pd.DataFrame, flight_id: int, coerce: list) -> pd.DataFrame:
    """Convert object columns to numbers only when EVERY non-empty value is numeric.
    Columns with any text are kept as text (nothing is turned into NaN silently); partial cases are logged."""
    for c in df.columns:
        if df[c].dtype == object:
            num = pd.to_numeric(df[c], errors="coerce")
            bad = num.isna() & df[c].notna() & (df[c].astype(str).str.strip() != "")
            if not bad.any():
                df[c] = num
            else:
                df[c] = df[c].astype("string")
                coerce.append(dict(flight_id=flight_id, column=c, n_non_numeric=int(bad.sum()),
                                   examples="|".join(map(str, df.loc[bad, c].unique()[:5])), kept_as="text"))
    return df


def _unify(schemas: list[pa.Schema]) -> pa.Schema:
    """Union of columns; a column that is text in any flight becomes text everywhere, numbers become float64."""
    types: dict[str, set] = {}
    order: list[str] = []
    for sc in schemas:
        for f in sc:
            if f.name not in types:
                types[f.name] = set()
                order.append(f.name)
            types[f.name].add("text" if pa.types.is_string(f.type) or pa.types.is_large_string(f.type) else
                              "int" if pa.types.is_integer(f.type) and f.name in ID_COLS else
                              "text" if f.name in ("aircraft_label", "account") else "num")
    fields = []
    for n in order:
        t = types[n]
        typ = pa.string() if "text" in t else (pa.int64() if t == {"int"} else pa.float64())
        fields.append(pa.field(n, typ))
    return pa.schema(fields)


def build_raw(settings: Settings, with_aux: bool = False, log=print) -> Path:
    """Main CSVs -> one combined parquet (streamed). Per-flight parquet files are cached in
    raw_combined/_per_flight and reused on re-runs. BMS sub-files (_1/_2) and _warns are inventoried;
    with_aux=True also converts them to raw_combined/aux_<kind>/flight_<id>.parquet (large: ~2x37 MB CSV per session).
    The original ZIPs in data/raw/zips are never modified and remain the authoritative raw copy."""
    raw_dir = settings.data_dir / "raw"
    out_dir = settings.data_dir / "raw_combined"
    tmp = out_dir / "_per_flight"
    tmp.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(raw_dir / "manifest.csv")
    man = man[man.status == "ok"].sort_values("flight_id")

    inventory, colmap, coerce = [], {}, []
    n = len(man)
    for k, r in enumerate(man.itertuples(), 1):
        pf = tmp / f"flight_{r.flight_id}.parquet"
        cached = pf.exists() and pf.stat().st_mtime >= Path(r.zip_path).stat().st_mtime
        with zipfile.ZipFile(r.zip_path) as z:
            main_name = None
            for name in z.namelist():
                kind = member_kind(name)
                info = z.getinfo(name)
                cols = rows = None
                if kind == "main" and main_name is None:
                    main_name = name
                    if not cached:
                        df = _read_member(z, name)
                        rows, cols = len(df), "|".join(df.columns)
                    else:
                        hdr = _read_member(z, name, nrows=0)
                        cols, rows = "|".join(hdr.columns), pq.ParquetFile(pf).metadata.num_rows
                elif kind not in ("other", "main"):
                    hdr = _read_member(z, name, nrows=0)
                    cols = "|".join(hdr.columns)
                    if with_aux:
                        ap = out_dir / f"aux_{kind}" / f"flight_{r.flight_id}.parquet"
                        if not ap.exists():
                            ap.parent.mkdir(parents=True, exist_ok=True)
                            d = _read_member(z, name)
                            d.columns = [slugify(c) for c in d.columns]
                            d = _numeric_or_keep(d, r.flight_id, coerce)
                            d.insert(0, "row_in_file", range(len(d)))
                            d.insert(0, "aircraft_label", r.aircraft_label)
                            d.insert(0, "flight_id", int(r.flight_id))
                            d.to_parquet(ap, index=False)
                            rows = len(d)
                            del d
                inventory.append(dict(flight_id=r.flight_id, aircraft_label=r.aircraft_label, member=name, kind=kind,
                                      bytes=info.file_size, compressed_bytes=info.compress_size, rows=rows, columns=cols))
        if main_name is None:
            inventory.append(dict(flight_id=r.flight_id, aircraft_label=r.aircraft_label,
                                  member="<no main csv>", kind="MISSING", bytes=0, rows=0, columns=""))
            continue
        for o in cols.split("|"):
            colmap[o] = slugify(o)
        if not cached:
            df.columns = [slugify(c) for c in df.columns]
            df = df.loc[:, ~pd.Index(df.columns).duplicated()]
            df = _numeric_or_keep(df, r.flight_id, coerce)
            df.insert(0, "row_in_file", range(len(df)))
            df.insert(0, "account", r.account)
            df.insert(0, "aircraft_label", r.aircraft_label)
            df.insert(0, "flight_id", int(r.flight_id))
            pq.write_table(pa.Table.from_pandas(df, preserve_index=False), pf)
            del df
        if k % 25 == 0 or k == n:
            log(f"  build-raw {k}/{n}")

    files = [tmp / f"flight_{fid}.parquet" for fid in man.flight_id if (tmp / f"flight_{fid}.parquet").exists()]
    unified = _unify([pq.read_schema(f) for f in files])
    out = out_dir / "combined_raw_timeseries.parquet"
    with pq.ParquetWriter(out, unified, compression="zstd") as w:
        for f in files:
            t = pq.read_table(f)
            cols = []
            for fld in unified:
                if fld.name in t.schema.names:
                    c = t[fld.name]
                    if pa.types.is_string(fld.type) and not pa.types.is_string(c.type):
                        c = pa.array(c.to_pandas().astype("string"), type=pa.string())
                    cols.append(c.cast(fld.type))
                else:
                    cols.append(pa.nulls(t.num_rows, fld.type))
            w.write_table(pa.Table.from_arrays(cols, schema=unified))
    pd.DataFrame(inventory).to_csv(out_dir / "zip_members.csv", index=False)
    pd.DataFrame([dict(original_header=k, canonical=v) for k, v in colmap.items()]).to_csv(out_dir / "column_map.csv", index=False)
    pd.DataFrame(coerce, columns=["flight_id", "column", "n_non_numeric", "examples", "kept_as"]).to_csv(out_dir / "coercion_log.csv", index=False)
    log(f"raw combined: {len(files)} flights -> {out}  (per-flight cache kept in {tmp})")
    return out
