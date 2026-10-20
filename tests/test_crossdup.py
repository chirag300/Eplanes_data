"""The same log listed under two aircraft is attributed by SOH continuity, not by flight_id order."""
import numpy as np
import pandas as pd

from eplanes.quality.crossdup import resolve


def _frame():
    t0 = pd.Timestamp("2025-07-01")
    rows = []
    # aircraft A sits near SOH 55, aircraft B near SOH 63
    for i in range(10):
        rows.append(dict(flight_id=100 + i, aircraft_label="A", start_datetime=t0 + pd.Timedelta(days=3 * i), soh=55.0, content_hash=f"a{i}"))
        rows.append(dict(flight_id=200 + i, aircraft_label="B", start_datetime=t0 + pd.Timedelta(days=3 * i), soh=63.0, content_hash=f"b{i}"))
    # one B log also listed under A, with the lower flight_id
    rows.append(dict(flight_id=150, aircraft_label="A", start_datetime=t0 + pd.Timedelta(days=12), soh=62.5, content_hash="x"))
    rows.append(dict(flight_id=250, aircraft_label="B", start_datetime=t0 + pd.Timedelta(days=12), soh=62.5, content_hash="x"))
    return pd.DataFrame(rows)


def test_copy_is_kept_under_the_aircraft_whose_soh_it_matches():
    r = resolve(_frame()).set_index("flight_id")
    assert r.loc[250, "keep"] and not r.loc[150, "keep"]


def test_undecidable_group_keeps_no_copy():
    f = _frame()
    f.loc[f.flight_id.isin([150, 250]), "soh"] = np.nan
    r = resolve(f)
    assert len(r) == 2 and not r.keep.any()


def test_groups_within_one_aircraft_are_left_to_the_generic_rule():
    f = _frame()
    f.loc[f.flight_id == 101, "content_hash"] = "a0"  # duplicate inside aircraft A only
    assert set(resolve(f).flight_id) == {150, 250}


def test_soh_column_may_start_with_underscore():
    # the quality stage passes a temporary column named "_soh"
    r = resolve(_frame().rename(columns={"soh": "_soh"}), soh="_soh").set_index("flight_id")
    assert r.loc[250, "keep"] and not r.loc[150, "keep"]
