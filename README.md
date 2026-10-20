# Eplanes: electric-aircraft battery dataset and exploratory-analysis pipeline

Reproducible pipeline for the paper *Interpreting State-of-Health Patterns in Electric Aircraft Operations: An Open Field Dataset and Analysis*
(Pipistrel Velis Electro, four aircraft in two fleets, BMS telemetry from the Pipistrel cloud portal).

**Scope:** data acquisition, dataset construction, data-quality validation, exploratory analysis, hypothesis tests, sensitivity analysis.
No predictive modelling. Reported SOH is treated as a BMS-reported number, never as measured capacity.

> **Status honesty.** The code was developed and tested on **synthetic data** shaped like your real files (the build sandbox could not reach
> `cloud.pipistrel.si`). The portal downloader is written from your notebooks' working logic and covered by mocked-HTML tests only,
> so run `discover` first and read its output before a full `download`. No result in this repository comes from real flights until you run it.

## Quick start (your computer)

```bash
git clone https://github.com/chirag300/eplanes_data && cd eplanes_data
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .

# 1. credentials -> local .env (git-ignored). Either copy .env.example and fill it in, or pull them from your notebooks:
python scripts/extract_credentials.py path/to/EDA_Data.ipynb --fleet-a <portal user for C-GAUW/C-GMUW> --fleet-b <portal user for VSW1280102/107>
#    then delete the passwords from the notebooks (and rotate them if the notebooks were ever shared)

# 2. prove the install works without network/credentials
python -m eplanes selftest

# 3. real run
python -m eplanes discover          # verifies aircraft IDs 166/192/191/196, counts flights per aircraft -> data/raw/aircraft_inventory.csv
python -m eplanes download          # resumable; re-run after any interruption; --retry-failed for errors; --limit 5 to test
```

Run order after the download (each step is also available on its own):

```bash
python scripts/inspect_flight.py 4574   # optional: per-column summary of one session (units, rates)
python -m eplanes build-raw        # data/raw_combined/  combined RAW dataset (nothing changed); --with-aux also converts BMS files
python -m eplanes build-clean      # data/processed/     analysis-ready dataset (flags + derived columns; no rows deleted)
python -m eplanes quality          # outputs/quality/    QC report + usability matrix + QUALITY GATE
python -m eplanes analyze          # outputs/figures, outputs/tables   (refuses to run before the gate exists)
python -m eplanes stats            # outputs/tables/H0_hypothesis_tests.csv
python -m eplanes sensitivity      # outputs/tables/S1-S3 + figure
python -m eplanes package          # outputs/eplanes_results_<timestamp>.zip  (+ KEY_FINDINGS.md)
# or everything after download:   python -m eplanes all
# resume later steps only:         python -m eplanes all --from quality   (reuses the built datasets)
```

Optional: `--external-flights flights.csv` (quality) cross-checks the portal against your local SQL `flights` table;
`--include-clean-data` / `--include-raw` (package) add data to the ZIP *only if you are permitted to share it*.

## Repository layout

```
eplanes_data/
├── README.md  requirements.txt  pyproject.toml  Makefile  .env.example  .gitignore
├── config/config.yaml            public settings: aircraft IDs, thresholds, bounds (NO credentials)
├── scripts/extract_credentials.py  builds local .env from your notebooks
├── eplanes/
│   ├── cli.py  config.py  synthetic.py  package.py  findings.py
│   ├── acquisition/  portal.py (login, pagination, parsing, retry)  downloader.py (manifest, resume)
│   ├── build/        raw.py (combined raw)  flights.py (flight table)  clean.py (analysis-ready)
│   ├── quality/      init_artifacts.py (start-up classifier)  report.py (checks, usability, gate)
│   ├── analysis/     features.py  context.py  eda.py (A-G)  stats.py  sensitivity.py
│   └── viz/style.py  figure style + caption registry
├── tests/            parsing, start-up classifier, estimators (incl. positive/negative controls), synthetic end-to-end, repo hygiene
└── docs/PAPER_MAP.md  paper section -> evidence map

# created at run time, git-ignored
data/raw/{zips,meta,manifest.csv}   original ZIPs byte-for-byte + parsed flight pages
data/raw_combined/                  combined raw parquet, ZIP-member inventory, column map, coercion log
data/processed/                     analysis-ready parquet, flight_qc.csv, init_records.csv, features, TRANSFORMATIONS.md
outputs/{quality,tables,figures}    reports, tables (csv), figures (png + pdf) + figure_captions.json
outputs/eplanes_results_*.zip       the archive to upload back for paper writing
```

## Method decisions (short)

* **Raw vs clean are separate and the clean table has the same number of rows.** Questionable samples get flag bits (`quality_flag`), they are never deleted.
* **Start-up readings.** The first row of a session is typically all-zero and early temperatures can equal (k-1)/k of the settled value (consistent with a resampling bin
  that averaged in a zero). Each flight x signal is classified (`zero_fill`, `bin_avg_zero`, `monotone_settling`, `load_onset`, `stable`) against a *data-driven* settled value
  (linear reference over minutes 1-3, noise-scaled tolerance, only the contiguous start run counts). Sessions that start under load are never flagged as artefacts
  (a physical transient cannot be excluded). Labels mean "consistent with"; the cause is listed as unresolved. Default cleaning rule `leadzero` masks each signal's leading run of exact zeros; `none | row0 | settle | fixed:<min>` are compared in the sensitivity analysis.
* **Physical plausibility:** sessions whose integrated current per SOC point departs >3x from the fleet median, or whose current / cell voltage exceed physical limits (seen in 19 C-GAUW sessions, Jul-Aug 2023), are excluded from current-, voltage- and energy-based analyses (`quality/plausibility.py`).
* **Quality gate.** `analyze`, `stats` and `sensitivity` stop unless `outputs/quality/QUALITY_COMPLETE.json` exists.
* **Observability is tested, including negative results.** CC/CV phases and dQ/dV use strict criteria with a positive and a negative control in the unit tests.
* **Statistics.** Flight is the unit of analysis (never samples); non-parametric tests; Holm within hypothesis family, BH globally; effect sizes and bootstrap CIs;
  per-aircraft-day collapsed robustness p-values; small-n (<30) or degenerate tests (|rho|=1, zero-width CI) are flagged and never counted as claimable findings (`claimable` column).
* **Trends** are reported on calendar-month medians (G4) next to per-session Theil-Sen slopes, because busy months contribute many sessions.
* **What the portal export contains** (checked on real sessions, Oct 2026): a main log at 10 Hz (~90 columns: pack voltage and current,
  SOC/SOH, min/max/avg and 16 individual cell temperatures per pack, drivetrain, and an avionics block with OAT, pressure altitude,
  IAS, GPS, attitude) plus one BMS file per pack at 5 Hz (Kalman and coulomb SOC, 96 per-cell SOH / capacity / FSK values,
  capacity estimate, reset flags, model terms) and a warnings file.
* **Units** are inferred from magnitudes and re-checked in the QC report: pack voltage V, cell voltage mV, current A
  (negative = charging), temperatures degC (integer), OAT degC, pressure altitude assumed m. Pack topology 96 cells in series
  (pack/cell voltage ratio and the BMS cell count); kW and kWh come from pack voltage x current.
* **Avionics block** channels are exactly 0 when the avionics are off (e.g. charging); `avionics_valid` (ACC_NORM > 0.5 g) separates
  "not logged" from a true 0 degC OAT. Flight phases (climb/cruise/descent) come from pressure altitude when it is alive;
  current-defined proxies are kept as a complement.
* **Invalid-value markers** (65535, e.g. motor rpm) are flagged (`quality_flag` bit 256). Some BMS model terms are signed 16-bit values
  stored unsigned (65488 = -48); they are reported as logged.
* **BMS internals** (`eplanes/build/bms.py`, run in the quality stage, cached per session) are summarised per session and pack,
  including a *uniform-cell* signature (all 96 cells report identical SOH and capacity) consistent with an estimator reset;
  hypothesis H9 tests whether reported-SOH increases coincide with entering that state.

## Website

`python -m eplanes export-web` builds the website bundle, `release` builds the downloadable dataset, `web` serves the site
on your machine and `web-static` writes a static snapshot. Hosting options, the map (OpenFreeMap, no API key), logos and the
pre-publication checklist are in [docs/WEBSITE.md](docs/WEBSITE.md). Nothing is deployed; the site stays private until the
paper is published.

## Data governance

* `.env`, `data/`, `outputs/` and all ZIP/parquet files are git-ignored. `tests/test_repo_hygiene.py` fails if a password literal is committed. `package` scans the ZIP for credentials before writing it.
* The export contains **GPS positions** (LAT/LNG) that reveal home airfields and routes; drop or coarsen them before any public release.
* The BMS files expose the manufacturer's **internal estimator state**; ask Pipistrel whether it may be redistributed.
* **Raw and cleaned flight data are not in this repository.** Confirm with the account owners/Pipistrel that redistribution is permitted before publishing any data
  (and decide whether registrations/serials such as C-GAUW should be anonymised in the release).
* Passwords currently sit in plain text in your notebooks; treat them as exposed if those notebooks were ever shared.

## Tests

```bash
python -m pytest -q          # 28 tests, ~2 min, no network
```
