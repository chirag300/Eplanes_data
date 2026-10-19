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
    assert (s.outputs_dir / "quality" / GATE).exists() and g["n_flights"] == 40
    o = run_eda(s, log=lambda *a: None)
    assert (s.outputs_dir / "tables" / "G0_observability_table.csv").exists()
    r = run_stats(s, log=lambda *a: None)
    assert {"p", "p_holm_family", "q_bh_global"} <= set(r.columns)
    caps = (s.outputs_dir / "figures" / "figure_captions.json").read_text()
    assert "A1_fleet_timeline" in caps


def test_text_columns_kept_not_nulled(s):
    raw = pq.read_table(s.data_dir / "raw_combined" / "combined_raw_timeseries.parquet", columns=["timestamp"]).to_pandas()
    assert raw["timestamp"].notna().all()
