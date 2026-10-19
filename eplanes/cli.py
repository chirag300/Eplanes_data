"""Command-line entry point:  python -m eplanes <command>   (or: eplanes <command>)

  discover      verify aircraft IDs and flight counts on the portal (needs .env)
  download      resumable download of every accessible flight (needs .env)
  build-raw     combined RAW dataset
  build-clean   analysis-ready dataset (flags + derived columns, no rows deleted)
  quality       data-quality validation + report + quality gate
  analyze       exploratory analysis A-G, figures and tables
  stats         hypothesis tests
  sensitivity   cleaning / segmentation / timestamp sensitivity
  package       results ZIP (+ KEY_FINDINGS.md)
  all           build-raw -> build-clean -> quality -> analyze -> stats -> sensitivity -> package
  selftest      full pipeline on SYNTHETIC data in a temp directory (no network, no credentials)
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from .config import load_settings


def main(argv=None):
    ap = argparse.ArgumentParser(prog="eplanes", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["discover", "download", "build-raw", "build-clean", "quality", "analyze", "stats",
                                        "sensitivity", "package", "all", "selftest"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--outputs-dir", default=None)
    ap.add_argument("--accounts", nargs="*", default=None, help="subset of accounts, e.g. fleet_A")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--with-aux", action="store_true", help="build-raw: also convert BMS _1/_2 and _warns files to parquet (large)")
    ap.add_argument("--login-mode", choices=["auto", "requests", "selenium"], default=None,
                    help="override download.login_mode (selenium = real Chrome login, as in the notebooks)")
    ap.add_argument("--limit", type=int, default=None, help="download at most N new flights (testing)")
    ap.add_argument("--external-flights", default=None, help="CSV exported from the local SQL flights table for cross-source checks")
    ap.add_argument("--include-clean-data", action="store_true", help="put analysis-ready data in the ZIP (check redistribution rights first)")
    ap.add_argument("--include-raw", action="store_true", help="put combined raw data in the ZIP (check redistribution rights first)")
    ap.add_argument("--max-flights", type=int, default=None, help="sensitivity: use only the first N flights")
    ap.add_argument("--rule", default="settle", help="cleaning rule for analyses: none|row0|settle|fixed:<min>")
    a = ap.parse_args(argv)

    s = load_settings(a.config)
    if a.data_dir or a.outputs_dir:
        s = s.with_dirs(Path(a.data_dir or s.data_dir).resolve(), Path(a.outputs_dir or s.outputs_dir).resolve())
    if a.login_mode:
        s.raw["download"]["login_mode"] = a.login_mode
    cmd = a.command

    if cmd == "selftest":
        return selftest()
    if cmd == "discover":
        from .acquisition.downloader import run_discover
        run_discover(s, a.accounts)
    elif cmd == "download":
        from .acquisition.downloader import run_download
        run_download(s, a.accounts, a.retry_failed, a.limit)
    steps = {"build-raw": lambda: __import__("eplanes.build.raw", fromlist=["x"]).build_raw(s, a.with_aux),
             "build-clean": lambda: __import__("eplanes.build.clean", fromlist=["x"]).build_clean(s),
             "quality": lambda: __import__("eplanes.quality.report", fromlist=["x"]).run_quality(s, a.external_flights),
             "analyze": lambda: __import__("eplanes.analysis.eda", fromlist=["x"]).run_eda(s, a.rule),
             "stats": lambda: __import__("eplanes.analysis.stats", fromlist=["x"]).run_stats(s, a.rule),
             "sensitivity": lambda: __import__("eplanes.analysis.sensitivity", fromlist=["x"]).run_sensitivity(s, a.max_flights)}
    order = ["build-raw", "build-clean", "quality", "analyze", "stats", "sensitivity"]
    if cmd in steps:
        steps[cmd]()
    elif cmd == "all":
        for c in order:
            print(f"\n=== {c} ===")
            steps[c]()
        _package(s, a)
    elif cmd == "package":
        _package(s, a)


def _package(s, a):
    from .findings import write_key_findings
    from .package import build_zip
    write_key_findings(s)
    build_zip(s, a.include_clean_data, a.include_raw)


def selftest(flights_per_aircraft: int = 25):
    """Run everything on synthetic data. Exits non-zero on failure."""
    from .synthetic import make_synthetic
    tmp = Path(tempfile.mkdtemp(prefix="eplanes_selftest_"))
    s = load_settings().with_dirs(tmp / "data_synthetic", tmp / "outputs_synthetic")
    make_synthetic(s.data_dir, flights_per_aircraft)
    ns = argparse.Namespace(include_clean_data=True, include_raw=False)
    from .analysis.eda import run_eda
    from .analysis.sensitivity import run_sensitivity
    from .analysis.stats import run_stats
    from .build.clean import build_clean
    from .build.raw import build_raw
    from .quality.report import run_quality
    build_raw(s)
    build_clean(s)
    run_quality(s)
    run_eda(s)
    run_stats(s)
    run_sensitivity(s)
    _package(s, ns)
    print(f"\nSELFTEST OK (synthetic data only; outputs in {tmp})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
