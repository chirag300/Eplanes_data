"""End-to-end smoke test on SYNTHETIC data (no network, no credentials)."""
import pandas as pd
import pyarrow.parquet as pq
import pytest

from eplanes.analysis.eda import run_eda
from eplanes.analysis.stats import run_stats
from eplanes.build.clean import build_clean
from eplanes.build.raw import build_raw
from eplanes.config import load_settings
from eplanes.quality.report import GATE, run_quality
from eplanes.synthetic import make_synthetic


@pytest.fixture(scope="module")
def s(tmp_path_factory):
    d = tmp_path_factory.mktemp("syn")
    st = load_settings().with_dirs(d / "data", d / "out")
    make_synthetic(st.data_dir, flights_per_aircraft=10)
    build_raw(st, log=lambda *a: None)
    build_clean(st, log=lambda *a: None)
    return st


def test_gate_blocks_analysis_before_quality(s):
    with pytest.raises(SystemExit):
        run_eda(s)


def test_clean_keeps_every_row_and_raw_values(s):
    raw = pq.read_table(s.data_dir / "raw_combined" / "combined_raw_timeseries.parquet").to_pandas()
    cl = pq.read_table(s.data_dir / "processed" / "analysis_ready_timeseries.parquet").to_pandas()
    assert len(raw) == len(cl)
    for c in ("bat_1_soc", "bat_1_current", "time_min", "bat_2_min_cell_volt"):
        assert raw[c].fillna(-999).reset_index(drop=True).equals(cl[c].fillna(-999).reset_index(drop=True))
    assert (cl.loc[cl.row_in_file == 0, "quality_flag"] & 1).all()  # row 0 flagged, not deleted


def test_quality_then_analysis(s):
    g = run_quality(s, log=lambda *a: None)
    assert (s.outputs_dir / "quality" / GATE).exists() and g["n_flights"] == 44
    o = run_eda(s, log=lambda *a: None)
    assert (s.outputs_dir / "tables" / "G0_observability_table.csv").exists()
    r = run_stats(s, log=lambda *a: None)
    assert {"p", "p_holm_family", "q_bh_global"} <= set(r.columns)
    caps = (s.outputs_dir / "figures" / "figure_captions.json").read_text()
    assert "A1_fleet_timeline" in caps


def test_text_columns_kept_not_nulled(s):
    raw = pq.read_table(s.data_dir / "raw_combined" / "combined_raw_timeseries.parquet", columns=["timestamp"]).to_pandas()
    assert raw["timestamp"].notna().all()


def test_bms_reset_detection_matches_planted(s):
    import pandas as pd
    from eplanes.build.bms import extract_bms_all
    b = extract_bms_all(s, log=lambda *a: None)
    m = pd.read_csv(s.data_dir / "raw" / "manifest.csv")[["flight_id", "synthetic_reset"]]
    x = b.merge(m, on="flight_id")
    assert (x.uniform_cells.astype(bool) == x.synthetic_reset.astype(bool)).all()
    assert x.first_zero_temp.all() and not x.first_zero_voltage.any()


def test_scale_anomaly_detected(s):
    import pandas as pd
    from eplanes.quality.plausibility import session_plausibility
    p, oor = session_plausibility(s, log=lambda *a: None)
    m = pd.read_csv(s.data_dir / "raw" / "manifest.csv")[["flight_id", "synthetic_scale_anomaly"]]
    x = p.merge(m, on="flight_id")
    assert (x.scale_anomaly == x.synthetic_scale_anomaly.astype(bool)).all()


def test_leadzero_rule_masks_only_leading_zeros():
    import numpy as np
    import pandas as pd
    from eplanes.analysis.features import apply_rule
    d = pd.DataFrame({"time_min": np.arange(6) * 0.1, "bat_1_max_cell_temp": [0, 0, 0, 21, 0, 22.0],
                      "bat_1_current": [0, 0, -5, -20, -20, 0.0], "quality_flag": 0, "init_unsettled_groups": 0})
    x = apply_rule(d, "leadzero")
    assert x.bat_1_max_cell_temp.isna().tolist() == [True, True, True, False, False, False]
    assert x.bat_1_current.isna().sum() == 2 and x.bat_1_current.iloc[-1] == 0
