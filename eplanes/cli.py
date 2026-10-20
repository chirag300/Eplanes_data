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
  export-web    compact website/paper bundle (sessions, 2 s traces, GPS tracks, SOH steps) + ZIP to upload
  release       downloadable dataset release (raw ZIPs, clean 10 Hz, tables, docs, checksums, manifest)
  web           run the Eplanes website locally (http://127.0.0.1:8000) on the web bundle
  web-static    build a static snapshot of the website (no server needed, e.g. GitHub Pages)
  selftest      full pipeline on SYNTHETIC data in a temp directory (no network, no credentials)
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from pathlib import Path

from .config import load_settings

warnings.filterwarnings("ignore", message="An input array is constant")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="(invalid value|divide by zero|Mean of empty|All-NaN)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="eplanes", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["discover", "download", "build-raw", "build-clean", "quality", "analyze", "stats",
                                        "sensitivity", "package", "all", "selftest",
                                        "export-web", "release", "web", "web-static"])
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
    ap.add_argument("--from", dest="from_step", default=None,
                    choices=["build-raw", "build-clean", "quality", "analyze", "stats", "sensitivity", "package"],
                    help="with `all`: start at this step (earlier outputs are reused)")
    ap.add_argument("--rule", default="leadzero", help="cleaning rule for analyses: leadzero|none|row0|settle|fixed:<min>")
    ap.add_argument("--resolution", type=float, default=2.0, help="export-web: trace resolution in seconds")
    ap.add_argument("--version", dest="release_version", default="1.0.0", help="release: dataset version")
    ap.add_argument("--no-raw", action="store_true", help="release: skip the original portal ZIPs")
    ap.add_argument("--bundle", default=None, help="web / web-static: web bundle directory (default outputs/web_bundle)")
    ap.add_argument("--release-dir", default=None, help="web: directory with the dataset release (downloads)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", default=None, help="web-static: output directory")
    ap.add_argument("--downloads-url", default=None,
                    help="web-static: base URL where the release files are hosted (e.g. https://zenodo.org/records/<id>/files)")
    ap.add_argument("--copy-release", action="store_true", help="web-static: copy the release files into the site (small releases only)")
    a = ap.parse_args(argv)

    s = load_settings(a.config)
    if a.data_dir or a.outputs_dir:
        s = s.with_dirs(Path(a.data_dir or s.data_dir).resolve(), Path(a.outputs_dir or s.outputs_dir).resolve())
    if a.login_mode:
        s.raw["download"]["login_mode"] = a.login_mode
    cmd = a.command

    if cmd == "selftest":
        return selftest()
    if cmd == "export-web":
        from .web_export import export_web
        export_web(s, a.rule, a.resolution)
        return 0
    if cmd == "release":
        from .web_export import build_release
        build_release(s, a.release_version, a.rule, include_raw=not a.no_raw)
        return 0
    if cmd in ("web", "web-static"):
        from .web.server import serve, build_static
        bundle = Path(a.bundle or s.outputs_dir / "web_bundle")
        rel = Path(a.release_dir) if a.release_dir else None
        if cmd == "web":
            serve(bundle, rel, a.host, a.port, settings=s)
        else:
            build_static(bundle, Path(a.out or s.outputs_dir / "web_static"), rel, copy_release=a.copy_release, downloads_url=a.downloads_url)
        return 0
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
        if a.from_step == "package":
            order = []
        elif a.from_step:
            order = order[order.index(a.from_step):]
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
