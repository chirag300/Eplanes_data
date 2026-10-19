# Where each paper section gets its evidence

The 13-section outline (Introduction ... Code Availability) comes from the project brief; the supplied LaTeX outline was not
available when this repository was written, so map by section title and adjust figure numbers to your LaTeX.

| Paper section | Evidence produced (outputs/) | Code |
|---|---|---|
| 3 Platform and Data Acquisition | `data/raw/manifest.csv`, `aircraft_inventory.csv`, `docs/column_map.csv`, `zip_members.csv` | `eplanes/acquisition/` |
| 4 Dataset Description | `docs/data_dictionary.csv`, `TRANSFORMATIONS.md`, `A1_fleet_overview`, `A2_feature_availability`, figs A1-A3 | `eplanes/build/`, `analysis/eda.py` (A) |
| 5 Data Quality and Coverage | `quality/data_quality_report.md`, `qc_*.csv` (incl. `qc_startup_bms_evidence`, `qc_avionics_sentinels_topology`), `usability_*.csv`, figs `qc_*` (missingness, coverage, start-up classes/traces) | `eplanes/quality/` |
| 6 Operational Patterns | B1/B2 flight profiles, B3/B4 altitude-based flight phases, E2/E4 cell vs ambient (OAT), C1 SOC/DoD, D1/D2 charging, E1 thermal; figs B1-B3, C1-C3, D1-D3, E1-E3; `H0_hypothesis_tests.csv` | `analysis/eda.py` (B-E), `analysis/stats.py` |
| 7 Behaviour of BMS-reported SOC and SOH | F1/F2 tables, figs F1-F4; BMS internals F5-F7 (reported vs per-cell SOH, uniform-cell/reset state, reported vs Kalman/coulomb SOC); G1 SOC-vs-current; tests H5, H6, H9 | `analysis/eda.py` (F, G) |
| 8 Observability of Degradation Indicators | `G0_observability_table.csv`, G1-G3 tables, `G4_trend_monthly_vs_session.csv` (use for fade rates), figs G1-G2 | `analysis/eda.py` (G), `analysis/features.py` |
| 9-10 Discussion / Limitations | `KEY_FINDINGS.md` (auto), `S1-S3` sensitivity tables + fig | `analysis/sensitivity.py`, `findings.py` |
| 12-13 Data / Code Availability | this repository; see README "Data governance" | - |
| 1-2 Introduction / Related Work | not produced by code; references in the slides must be verified before citing | - |
