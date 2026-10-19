"""Analysis-ready dataset: raw rows are KEPT; flags and derived columns are ADDED.

Guarantees (asserted at the end of build_clean):
  * row count of the analysis-ready table == row count of the combined raw table
  * every original canonical column is carried through unmodified
Added columns
  time_s, dt_min                    derived from time_min
  *_cell_volt_v                     mV -> V (unit inferred from magnitude)
  quality_flag (bitmask)            see FLAG_BITS
  init_unsettled_groups (bitmask)   signal groups still inside their start-up window (init_artifacts.GROUP_BITS)
  segment_id, segment_type          activity segmentation (rest / charge / discharge) from smoothed pack current
Quality-control by-products
  flight_qc.csv                     one row per flight
  init_records.csv                  one row per flight x signal (start-up behaviour)
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..config import RAW_TO_CANON, Settings
from ..quality.init_artifacts import GROUP_BITS, group_of, init_records

# canonical battery columns; any that a given export lacks are added as all-missing so absence is documented
CANON = list(dict.fromkeys(RAW_TO_CANON.values()))
SIGNALS = [c for c in CANON if c != "time_min"]
FLAG_BITS = {"row0_zero": 1, "init_unsettled": 2, "dropout_nan": 4, "out_of_range": 8,
             "duplicate_time": 16, "nonmonotonic_time": 32, "after_gap": 64, "zero_sample_after_init": 128,
             "sentinel_value": 256}


def iter_flights(path: Path, columns: list[str] | None = None):
    """Yield one DataFrame per flight from a combined parquet (sorted by flight_id).
    `columns` limits what is read (missing names are ignored)."""
    pf = pq.ParquetFile(path)
    if columns is not None:
        names = set(pf.schema_arrow.names)
        columns = list(dict.fromkeys(["flight_id"] + [c for c in columns if c in names]))
    carry = None
    for i in range(pf.num_row_groups):
        d = pf.read_row_group(i, columns=columns).to_pandas()
        if carry is not None:
            d = pd.concat([carry, d], ignore_index=True)
            carry = None
        ids = d["flight_id"].to_numpy()
        last = ids[-1]
        cut = np.searchsorted(ids, last, side="left") if (ids == last).any() else len(d)
        for fid, g in d.iloc[:cut].groupby("flight_id", sort=False):
            yield g.reset_index(drop=True)
        carry = d.iloc[cut:]
    if carry is not None and len(carry):
        yield carry.reset_index(drop=True)


def _runs(label: np.ndarray):
    change = np.r_[True, label[1:] != label[:-1]]
    starts = np.where(change)[0]
    ends = np.r_[starts[1:], len(label)]
    return list(zip(starts, ends))


def segment_activity(t, current, cfg) -> tuple[np.ndarray, np.ndarray]:
    thr = cfg["segmentation"]["active_current_a"]
    sign = cfg["charge_sign"]
    med_dt = np.nanmedian(np.diff(t)) if len(t) > 2 else np.nan
    win = max(5, int(cfg["segmentation"].get("smooth_s", 30) / 60.0 / med_dt)) if np.isfinite(med_dt) and med_dt > 0 else 25
    sm = pd.Series(current).rolling(win, center=True, min_periods=3).median().to_numpy()
    lab = np.where(sm * sign > thr, 1, np.where(-sm * sign > thr, 2, 0))  # 1 charge, 2 discharge
    min_n = max(1, int(cfg["segmentation"]["min_segment_min"] / max(np.nanmedian(np.diff(t)), 1e-6)))
    for _ in range(2):  # merge too-short runs into the previous run
        for a, b in _runs(lab):
            if b - a < min_n and a > 0:
                lab[a:b] = lab[a - 1]
    seg_id = np.zeros(len(lab), dtype=np.int32)
    for k, (a, b) in enumerate(_runs(lab)):
        seg_id[a:b] = k
    names = np.array(["rest", "charge", "discharge"])
    return seg_id, names[lab]


def process_flight(g: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict, list[dict]]:
    cq = cfg["quality"]
    b = cfg["screening_bounds"]
    d = g.copy()
    for c in CANON:
        if c not in d:
            d[c] = np.nan
    fid = int(d["flight_id"].iloc[0])
    t = d["time_min"].to_numpy(float)
    n = len(d)
    dt = np.diff(t, prepend=np.nan)
    pos = dt[np.isfinite(dt) & (dt > 0)]
    med_dt = float(np.median(pos)) if len(pos) else np.nan

    present = [c for c in SIGNALS if d[c].notna().any()]
    flags = np.zeros(n, dtype=np.int32)
    vals = d[present].to_numpy(float) if present else np.empty((n, 0))
    if n and len(present) and np.all((vals[0] == 0) | ~np.isfinite(vals[0])) and np.any(vals[0] == 0):
        flags[0] |= FLAG_BITS["row0_zero"]
    dropout = np.zeros(n, bool)
    for c in present:
        dropout |= d[c].isna().to_numpy()
    flags[dropout] |= FLAG_BITS["dropout_nan"]
    oor = np.zeros(n, bool)
    for c in present:
        x = d[c].to_numpy(float)
        if c.endswith("_soc"):
            lo, hi = b["soc"]
        elif c.endswith("_soh"):
            lo, hi = b["soh"]
        elif "cell_volt" in c:
            lo, hi = b["cell_volt_mv"]
        elif "temp" in c:
            lo, hi = b["temp_c"]
        elif c.endswith("_voltage"):
            lo, hi = b.get("pack_voltage_v", [250, 420])
        elif c.endswith("current"):
            lo, hi = -b["abs_current_a"][1], b["abs_current_a"][1]
        else:
            continue
        bad = np.isfinite(x) & ((x < lo) | (x > hi))
        oor |= bad
    oor &= (flags & FLAG_BITS["row0_zero"]) == 0  # the all-zero start row is reported as row0_zero, not as out-of-range
    sent = np.zeros(n, bool)
    for c in d.columns:
        if c in ("flight_id", "row_in_file") or not pd.api.types.is_numeric_dtype(d[c]):
            continue
        for v in cfg.get("sentinel_values", []):
            sent |= (d[c].to_numpy() == v)
    flags[sent] |= FLAG_BITS["sentinel_value"]
    flags[oor] |= FLAG_BITS["out_of_range"]
    flags[np.isfinite(dt) & (dt == 0)] |= FLAG_BITS["duplicate_time"]
    flags[np.isfinite(dt) & (dt < 0)] |= FLAG_BITS["nonmonotonic_time"]
    if np.isfinite(med_dt):
        flags[np.isfinite(dt) & (dt > cq["gap_factor"] * med_dt)] |= FLAG_BITS["after_gap"]

    # start-up analysis (on time-sorted valid rows only)
    cols = {c: d[c].to_numpy(float) for c in present}
    recs = init_records(fid, t, cols, cfg) if n >= cq["min_flight_rows"] else []
    grp_mask = np.zeros(n, dtype=np.int32)
    for r in recs:
        gname = group_of(r["signal"])
        si = r["settle_idx"]
        if gname and np.isfinite(si) and si > 0 and r["cls"] not in ("load_onset", "stable", "not_assessable"):
            grp_mask[: int(si)] |= GROUP_BITS[gname]
    flags[grp_mask > 0] |= FLAG_BITS["init_unsettled"]
    for c in present:
        if "volt" in c:
            z = (d[c].to_numpy(float) == 0)
            z[0] = False
            flags[z] |= FLAG_BITS["zero_sample_after_init"]

    cur = np.nanmean(np.vstack([d["bat_1_current"].to_numpy(float), d["bat_2_current"].to_numpy(float)]), axis=0) \
        if n else np.array([])
    seg_id, seg_type = segment_activity(t, cur, cfg) if n >= 5 else (np.zeros(n, np.int32), np.array(["rest"] * n))

    d["time_s"] = t * 60.0
    thr = cfg.get("avionics_valid_acc_norm", 0.5)
    d["avionics_valid"] = (pd.to_numeric(d["acc_norm"], errors="coerce") > thr).to_numpy() if "acc_norm" in d else False
    with np.errstate(divide="ignore", invalid="ignore"):
        pv = [d[f"bat_{b}_voltage"].to_numpy(float) / (0.5 * (d[f"bat_{b}_min_cell_volt"].to_numpy(float) +
              d[f"bat_{b}_max_cell_volt"].to_numpy(float)) / 1000.0) for b in (1, 2)]
    d["dt_min"] = dt
    for c in ("bat_1_min_cell_volt", "bat_1_max_cell_volt", "bat_2_min_cell_volt", "bat_2_max_cell_volt"):
        d[c.replace("volt", "volt_v")] = d[c] / 1000.0
    d["quality_flag"] = flags
    d["init_unsettled_groups"] = grp_mask
    d["segment_id"] = seg_id
    d["segment_type"] = seg_type

    qc = dict(flight_id=fid, n_rows=n, duration_min=float(np.nanmax(t)) if n else np.nan,
              median_dt_min=med_dt, p99_dt_min=float(np.nanpercentile(pos, 99)) if len(pos) else np.nan,
              n_gaps=int(((flags & FLAG_BITS["after_gap"]) > 0).sum()),
              max_gap_min=float(np.nanmax(dt)) if n > 1 else np.nan,
              n_duplicate_time=int(((flags & FLAG_BITS["duplicate_time"]) > 0).sum()),
              n_nonmonotonic=int(((flags & FLAG_BITS["nonmonotonic_time"]) > 0).sum()),
              n_row0_zero=int(flags[0] & 1) if n else 0,
              n_init_unsettled=int(((flags & FLAG_BITS["init_unsettled"]) > 0).sum()),
              n_out_of_range=int(((flags & FLAG_BITS["out_of_range"]) > 0).sum()),
              n_zero_sample_after_init=int(((flags & FLAG_BITS["zero_sample_after_init"]) > 0).sum()),
              n_sentinel=int(((flags & FLAG_BITS["sentinel_value"]) > 0).sum()),
              frac_avionics_valid=float(np.mean(d["avionics_valid"])) if n else np.nan,
              series_ratio_b1=float(np.nanmedian(pv[0][(pv[0] > 50) & (pv[0] < 200)])) if n and ((pv[0] > 50) & (pv[0] < 200)).any() else np.nan,
              series_ratio_b2=float(np.nanmedian(pv[1][(pv[1] > 50) & (pv[1] < 200)])) if n and ((pv[1] > 50) & (pv[1] < 200)).any() else np.nan,
              start_state=recs[0]["start_state"] if recs else "unknown",
              frac_charge=float((seg_type == "charge").mean()) if n else np.nan,
              frac_discharge=float((seg_type == "discharge").mean()) if n else np.nan,
              n_segments=int(seg_id.max() + 1) if n else 0,
              max_abs_current=float(np.nanmax(np.abs(cur))) if n and np.isfinite(cur).any() else np.nan,
              content_hash=hashlib.sha256(np.round(d[CANON].to_numpy(float), 6).tobytes()).hexdigest() if n else "")
    for c in SIGNALS:
        x = d[c].to_numpy(float)
        qc[f"missing_frac__{c}"] = float(np.isnan(x).mean()) if n else np.nan
        fin = x[np.isfinite(x)]
        qc[f"all_zero__{c}"] = bool(len(fin) > 0 and np.all(fin == 0))
        qc[f"constant__{c}"] = bool(len(fin) > 1 and np.nanmax(fin) == np.nanmin(fin))
    return d, qc, recs


def build_clean(settings: Settings, log=print) -> Path:
    cfg = settings.raw
    raw_path = settings.data_dir / "raw_combined" / "combined_raw_timeseries.parquet"
    out_dir = settings.data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "analysis_ready_timeseries.parquet"
    writer, qcs, recs_all, n_out = None, [], [], 0
    for g in iter_flights(raw_path):
        d, qc, recs = process_flight(g, cfg)
        qcs.append(qc)
        recs_all.extend(recs)
        t = pa.Table.from_pandas(d, preserve_index=False)
        if writer is None:
            schema = t.schema
            writer = pq.ParquetWriter(out, schema)
        writer.write_table(t.cast(schema))
        n_out += len(d)
    writer.close()
    n_raw = pq.ParquetFile(raw_path).metadata.num_rows
    assert n_raw == n_out, f"row count changed: raw={n_raw} clean={n_out}"
    pd.DataFrame(qcs).to_csv(out_dir / "flight_qc.csv", index=False)
    pd.DataFrame(recs_all).to_csv(out_dir / "init_records.csv", index=False)
    (out_dir / "TRANSFORMATIONS.md").write_text(_transformations_doc(cfg, n_raw))
    log(f"analysis-ready: {n_out:,} rows (== raw), {len(qcs)} flights -> {out}")
    return out


def _transformations_doc(cfg: dict, n_rows: int) -> str:
    flag_lines = "\n".join(f"| {v} | {k} |" for k, v in FLAG_BITS.items())
    grp_lines = "\n".join(f"| {v} | {k} |" for k, v in GROUP_BITS.items())
    return f"""# Transformations applied to build the analysis-ready dataset

Rows: {n_rows:,} (identical to the combined raw dataset; NO rows deleted, NO raw values changed).

1. Column names: portal headers -> snake_case (see raw_combined/column_map.csv).
2. Added `time_s`, `dt_min`; added `*_cell_volt_v` = mV / 1000 (unit inferred from magnitude).
3. Row-level `quality_flag` bitmask (flags only; nothing removed):

| bit | meaning |
|---|---|
{flag_lines}

4. `init_unsettled_groups` bitmask - signal groups still inside their start-up window, only for
   signals whose start-up behaviour was classified as zero_fill / bin_avg_zero / monotone_settling /
   irregular_start (load_onset is treated as physical and is never flagged):

| bit | group |
|---|---|
{grp_lines}

5. `segment_id` / `segment_type`: activity segmentation from the 30 s rolling median of the mean pack current
   (|I| > {cfg['segmentation']['active_current_a']} A is active; charging sign = {cfg['charge_sign']}; runs shorter than
   {cfg['segmentation']['min_segment_min']} min merged into the previous run).
6. Screening bounds used for `out_of_range` are in config/config.yaml (`screening_bounds`) and are screening limits only.

Analyses choose which flags to honour through an explicit *cleaning rule*
(none | row0 | settle | fixed:<minutes>); see eplanes/analysis/features.py and the sensitivity analysis.
"""
