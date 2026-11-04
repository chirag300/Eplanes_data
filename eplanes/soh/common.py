"""Shared paths, settings and small helpers for the SOH pipeline."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..config import ROOT, Settings, load_settings

SOH_CONFIG = ROOT / "config" / "soh.yaml"


@dataclass
class Ctx:
    """Everything a stage needs: the base settings, the SOH config and the output folders."""
    settings: Settings
    cfg: dict
    proc: Path        # data/processed
    cache: Path       # data/processed/soh_cache
    out: Path         # outputs/soh
    tables: Path
    figs: Path

    def log_path(self) -> Path:
        return self.out / "run_log.txt"


def load_ctx(data_dir=None, outputs_dir=None, config=None, soh_config=None) -> Ctx:
    s = load_settings(config)
    if data_dir or outputs_dir:
        s = s.with_dirs(Path(data_dir or s.data_dir).resolve(), Path(outputs_dir or s.outputs_dir).resolve())
    cfg = yaml.safe_load(open(soh_config or SOH_CONFIG))
    proc = s.data_dir / "processed"
    out = s.outputs_dir / "soh"
    for p in (proc / "soh_cache" / "sessions", out / "tables", out / "figures"):
        p.mkdir(parents=True, exist_ok=True)
    return Ctx(s, cfg, proc, proc / "soh_cache", out, out / "tables", out / "figures")


class Log:
    """Print and keep a copy in outputs/soh/run_log.txt, so the log travels with the results ZIP."""

    def __init__(self, ctx: Ctx):
        self.path = ctx.log_path()

    def __call__(self, *a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        with open(self.path, "a") as f:
            f.write(msg + "\n")


def load_sessions(ctx: Ctx) -> pd.DataFrame:
    """Session table with the exclusions the dataset paper defines; no row is dropped, usable_electrical says what may be used."""
    p = ctx.proc / "flights_with_qc.csv"
    if not p.exists():
        raise SystemExit(f"{p} not found. Run `python -m eplanes quality` first.")
    f = pd.read_csv(p, parse_dates=["start_datetime"], low_memory=False)
    for c in ("scale_anomaly", "is_duplicate_content", "cross_aircraft_duplicate"):
        f[c] = f[c].astype("boolean").fillna(False).astype(bool) if c in f else False
    if "duration_min" not in f:
        f["duration_min"] = np.nan
    f["usable_electrical"] = ~f.scale_anomaly & ~f.is_duplicate_content & f.start_datetime.notna()
    return f.sort_values(["aircraft_label", "start_datetime", "flight_id"]).reset_index(drop=True)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    raise TypeError(type(o))


def wmedian(x: np.ndarray, w: np.ndarray | None = None) -> float:
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if not ok.any():
        return np.nan
    if w is None:
        return float(np.median(x[ok]))
    x, w = x[ok], np.asarray(w, float)[ok]
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, 0.5 * c[-1])])


def mad_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else np.nan
