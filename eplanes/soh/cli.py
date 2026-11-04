"""python -m eplanes.soh <command>

  check        confirm the inputs exist and report what is missing
  extract      per-session observables from the 10 Hz data (resumable, runs once)
  label        physics SOH label: capacity from voltage, state-space smoother
  validate     tests of the label that need no ground truth
  train        hybrid models, training fleet only (rolling-origin CV, ablation runs)
  test-final   score the held-out fleet ONCE (refuses to repeat)
  report       figures, SOH_RESULTS.md and a ZIP to upload
  all          extract -> label -> validate -> train -> report   (does not touch the held-out fleet)
  simulate     write a truth-known simulated dataset to --data-dir
  selftest     all of the above on simulated data in a temporary folder
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from pathlib import Path

from .common import Log, load_ctx

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def check(ctx, log):
    need = [ctx.proc / "analysis_ready_timeseries.parquet", ctx.proc / "flights_with_qc.csv"]
    opt = [ctx.proc / "bms_session.csv"]
    ok = True
    for p in need:
        log(("found   " if p.exists() else "MISSING ") + str(p))
        ok &= p.exists()
    for p in opt:
        log(("found   " if p.exists() else "absent (optional) ") + str(p))
    try:
        import torch
        log(f"torch {torch.__version__}")
    except ImportError:
        log("torch is not installed: neural models will fail. pip install -r requirements-soh.txt")
        ok = False
    try:
        import sklearn
        log(f"scikit-learn {sklearn.__version__}")
    except ImportError:
        log("scikit-learn is not installed. pip install -r requirements-soh.txt")
        ok = False
    if not ok:
        raise SystemExit("Fix the items above first. The two data files come from `python -m eplanes all` (build-clean and quality).")
    log("ready")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m eplanes.soh", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "extract", "label", "validate", "train", "test-final", "report", "all", "simulate", "selftest"])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--outputs-dir", default=None)
    ap.add_argument("--models", nargs="*", default=None, help="subset of forecasters, e.g. --models persistence physics_bayes ude")
    ap.add_argument("--quick", action="store_true", help="few epochs and one seed: for checking that things run, not for results")
    ap.add_argument("--force-rescore", action="store_true", help="test-final: score the held-out fleet again (recorded in the lock file)")
    ap.add_argument("--scale", type=float, default=0.3, help="simulate: amount of data")
    ap.add_argument("--ocv-slip", type=float, default=0.0, help="simulate: stress test, ageing shifts the voltage curve")
    ap.add_argument("--ocv-shape", type=float, default=0.0, help="simulate: stress test, ageing stretches the voltage curve (V per unit of fade)")
    a = ap.parse_args(argv)
    if a.command == "selftest":
        return selftest(a)
    if a.command == "simulate":
        if not a.data_dir:
            raise SystemExit("simulate needs --data-dir, so simulated files can never land on top of the real data")
        from .simulate import make_sim
        make_sim(a.data_dir, a.scale, ocv_slip=a.ocv_slip, ocv_shape=a.ocv_shape)
        return 0
    ctx = load_ctx(a.data_dir, a.outputs_dir)
    if a.quick:
        quick(ctx)
    run(ctx, a.command, a.models, a.force_rescore)
    return 0


def quick(ctx):
    ctx.cfg["models"].update(epochs=60, patience=20, seeds=1, cnn_epochs=10)
    ctx.cfg["data"]["cv_folds"] = 2


def run(ctx, cmd, models=None, force=False):
    log = Log(ctx)
    from .evaluate import run_final, run_train
    from .label import run_label
    from .observables import run_extract
    from .report import run_report
    from .validate import run_validate
    if cmd == "check":
        check(ctx, log)
    elif cmd == "extract":
        run_extract(ctx, log)
    elif cmd == "label":
        run_label(ctx, log)
    elif cmd == "validate":
        run_validate(ctx, log)
    elif cmd == "train":
        run_train(ctx, log, models)
    elif cmd == "test-final":
        run_final(ctx, log, force, models)
    elif cmd == "report":
        run_report(ctx, log)
    elif cmd == "all":
        check(ctx, log)
        run_extract(ctx, log)
        run_label(ctx, log)
        run_validate(ctx, log)
        run_train(ctx, log, models)
        run_report(ctx, log)


def selftest(a):
    from .simulate import make_sim
    tmp = Path(tempfile.mkdtemp(prefix="eplanes_soh_selftest_"))
    make_sim(tmp / "data", 0.12, log=lambda *x: None)
    ctx = load_ctx(tmp / "data", tmp / "out")
    quick(ctx)
    run(ctx, "all", a.models)
    run(ctx, "test-final", a.models)
    run(ctx, "report")
    print(f"\nSELFTEST OK (simulated data; outputs in {tmp})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
