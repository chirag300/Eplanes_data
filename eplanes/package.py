"""Results ZIP: organised outputs + reproducibility info, with a credential/sensitive-content scan."""
from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import ROOT, Settings

SECRET_PATTERNS = [re.compile(r"(?i)password\s*[:=]\s*['\"]?[^\s'\"]{4,}"),
                   re.compile(r"(?i)(api|secret|token)[_-]?key\s*[:=]\s*\S{8,}"),
                   re.compile(r"AccountKey=[A-Za-z0-9+/=]{20,}")]


def _env_secrets() -> list[str]:
    import os
    vals = [v for k, v in os.environ.items() if k.startswith("PIPISTREL_") and v and k.endswith(("PASSWORD", "USER"))]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                v = line.split("=", 1)[1].strip()
                if v:
                    vals.append(v)
    return [v for v in set(vals) if len(v) >= 4]


def scan_for_secrets(paths: list[Path]) -> list[str]:
    secrets = _env_secrets()
    hits = []
    for p in paths:
        if p.suffix.lower() not in {".md", ".csv", ".json", ".txt", ".yaml", ".yml", ".py", ".tex"}:
            continue
        try:
            txt = p.read_text(errors="ignore")
        except Exception:  # noqa: BLE001
            continue
        for s in secrets:
            if s in txt:
                hits.append(f"{p}: contains a configured credential value")
        for pat in SECRET_PATTERNS:
            if pat.search(txt):
                hits.append(f"{p}: matches secret pattern {pat.pattern[:30]}")
    return hits


def repro_info(settings: Settings) -> dict:
    def sh(cmd):
        try:
            return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:  # noqa: BLE001
            return ""
    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    except Exception:  # noqa: BLE001
        freeze = ""
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), python=sys.version, platform=platform.platform(),
                git_commit=sh(["git", "rev-parse", "HEAD"]), git_dirty=bool(sh(["git", "status", "--porcelain"])),
                config=settings.raw, pip_freeze=freeze.splitlines())


def build_data_dictionary(settings: Settings, out: Path):
    from .config import SIGNAL_DESCRIPTIONS
    from .build.clean import FLAG_BITS
    rows = []
    for k, (desc, unit) in SIGNAL_DESCRIPTIONS.items():
        rows.append(dict(column=k if k == "time_min" else f"bat_<1|2>_{k}", description=desc, unit=unit, dataset="raw + analysis-ready",
                         notes="bat_2_avg_cell_temp is not logged" if k == "avg_cell_temp" else ""))
    rows += [dict(column="flight_id", description="Portal session id", unit="-", dataset="all", notes=""),
             dict(column="aircraft_label", description="Registration or serial", unit="-", dataset="all", notes=""),
             dict(column="row_in_file", description="Row number in the original CSV", unit="-", dataset="raw", notes=""),
             dict(column="time_s", description="time_min*60", unit="s", dataset="analysis-ready", notes="derived"),
             dict(column="dt_min", description="difference to previous time stamp", unit="min", dataset="analysis-ready", notes="derived"),
             dict(column="*_cell_volt_v", description="cell voltage in volts", unit="V", dataset="analysis-ready", notes="mV/1000"),
             dict(column="quality_flag", description="bitmask: " + ", ".join(f"{v}={k}" for k, v in FLAG_BITS.items()), unit="-", dataset="analysis-ready", notes="flags only"),
             dict(column="init_unsettled_groups", description="bitmask of signal groups inside start-up window (volt=1,soc=2,soh=4,current=8,temp=16)", unit="-", dataset="analysis-ready", notes=""),
             dict(column="segment_id / segment_type", description="activity segment (rest/charge/discharge)", unit="-", dataset="analysis-ready", notes="current threshold in config")]
    pd.DataFrame(rows).to_csv(out, index=False)


def build_zip(settings: Settings, include_clean_data: bool = False, include_raw: bool = False, log=print) -> Path:
    od = settings.outputs_dir
    stage = od / "_zip_stage"
    if stage.exists():
        import shutil
        shutil.rmtree(stage)
    (stage / "docs").mkdir(parents=True)
    build_data_dictionary(settings, stage / "docs" / "data_dictionary.csv")
    for src in ("TRANSFORMATIONS.md",):
        p = settings.data_dir / "processed" / src
        if p.exists():
            (stage / "docs" / src).write_text(p.read_text())
    (stage / "docs" / "reproducibility.json").write_text(json.dumps(repro_info(settings), indent=1, default=str))
    for name in ("column_map.csv", "zip_members.csv", "coercion_log.csv"):
        p = settings.data_dir / "raw_combined" / name
        if p.exists():
            (stage / "docs" / name).write_text(p.read_text())

    files: list[tuple[Path, str]] = []
    for sub in ("quality", "tables", "figures"):
        d = od / sub
        if d.exists():
            files += [(p, f"{sub}/{p.relative_to(d)}") for p in d.rglob("*") if p.is_file()]
    for name in ("analysis_summary.md", "KEY_FINDINGS.md"):
        if (od / name).exists():
            files.append((od / name, name))
    files += [(p, f"docs/{p.name}") for p in (stage / "docs").glob("*")]
    proc = settings.data_dir / "processed"
    if include_clean_data:
        for n in ("flights_with_qc.csv", "flight_qc.csv", "init_records.csv", "features_battery_session__settle.csv"):
            if (proc / n).exists():
                files.append((proc / n, f"data/{n}"))
        ts = proc / "analysis_ready_timeseries.parquet"
        if ts.exists():
            files.append((ts, "data/analysis_ready_timeseries.parquet"))
    if include_raw:
        rc = settings.data_dir / "raw_combined" / "combined_raw_timeseries.parquet"
        if rc.exists():
            files.append((rc, "data/combined_raw_timeseries.parquet"))
    hits = scan_for_secrets([p for p, _ in files])
    if hits:
        raise SystemExit("Refusing to build ZIP, possible secrets found:\n" + "\n".join(hits))
    # captions file
    capf = od / "figures" / "figure_captions.json"
    if capf.exists():
        caps = json.loads(capf.read_text())
        (stage / "docs" / "figure_captions.md").write_text("\n\n".join(f"**{k}**: {v}" for k, v in sorted(caps.items())))
        files.append((stage / "docs" / "figure_captions.md", "docs/figure_captions.md"))
    qcaps = od / "quality" / "figures" / "figure_captions.json"
    if qcaps.exists():
        caps = json.loads(qcaps.read_text())
        (stage / "docs" / "quality_figure_captions.md").write_text("\n\n".join(f"**{k}**: {v}" for k, v in sorted(caps.items())))
        files.append((stage / "docs" / "quality_figure_captions.md", "docs/quality_figure_captions.md"))
    zpath = od / f"eplanes_results_{datetime.now():%Y%m%d_%H%M}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p, arc in files:
            z.write(p, arc)
    log(f"wrote {zpath} ({zpath.stat().st_size/1e6:.1f} MB, {len(files)} files)")
    return zpath
