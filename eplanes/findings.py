"""KEY_FINDINGS.md assembled from the result tables (no numbers typed by hand)."""
from __future__ import annotations

import json

import pandas as pd

from .config import Settings


def write_key_findings(settings: Settings) -> str:
    od = settings.outputs_dir
    parts = ["# Key findings, limitations and unresolved questions\n",
             "_Auto-generated from the result tables; read together with the data-quality report. Reported SOH is a BMS number, "
             "not a measured capacity._\n"]
    gate = od / "quality" / "QUALITY_COMPLETE.json"
    if gate.exists():
        g = json.loads(gate.read_text())
        parts += ["## Data quality", *[f"- {x}" for x in g["findings"]], ""]
    s = od / "analysis_summary.md"
    if s.exists():
        parts += ["## Exploratory analysis", s.read_text().split("\n", 2)[2], ""]
    t = od / "tables"
    if (t / "H0_hypothesis_tests.csv").exists():
        h = pd.read_csv(t / "H0_hypothesis_tests.csv")
        sig = h[h.get("p_holm_family", pd.Series(dtype=float)) < 0.05]
        parts += ["## Hypothesis tests", f"- {len(h)} tests run; {len(sig)} survive Holm correction within their family "
                  f"({', '.join(sig.id.head(12)) or 'none'}).",
                  f"- {int(h.exploratory_flag.sum()) if 'exploratory_flag' in h else 'n/a'} tests are flagged exploratory (small n or dependence).", ""]
    if (t / "S3_sensitivity_verdict.csv").exists():
        v = pd.read_csv(t / "S3_sensitivity_verdict.csv")
        unstable = v[(v.get("stable_excl_no_cleaning", v.stable) == False) | ((v.kind == "p-value") & (v.stable == False))]  # noqa: E712
        parts += ["## Sensitivity", f"- {len(v)} headline metrics checked across cleaning/segmentation/timestamp rules; "
                  f"{len(unstable)} are sensitive: {', '.join(unstable.metric.head(10)) or 'none'}.",
                  "- Using raw data with no start-up handling changes start-of-session values (the first row is all-zero), see S1/S2.", ""]
    if (t / "G0_observability_table.csv").exists():
        o = pd.read_csv(t / "G0_observability_table.csv")
        parts += ["## Observability of degradation indicators", *[f"- {r.indicator}: **{r.grade}**. {r.limitation}" for r in o.itertuples()], ""]
    if gate.exists():
        parts += ["## Limitations", *[f"- {x}" for x in g["limitations"]],
                  "- Aircraft/battery identity, operating season and fleet are partly confounded; between-aircraft differences are not causal.",
                  "- Sessions are serially dependent over time; p-values treat flights as independent (see per-aircraft-day robustness column).",
                  "- Only logged sessions are counted: throughput and utilisation are lower bounds.", "",
                  "## Unresolved questions",
                  "- Physical cause of start-up artefacts (logger zero-fill vs resampling vs sensor/BMS warm-up).",
                  "- Whether SOH jumps coincide with maintenance, BMS recalibration or firmware events (needs the maintenance log).",
                  "- Pack topology (series cell count) for kW/kWh estimates; time zone of portal timestamps.",
                  "- Permission from the dataset owners to redistribute raw/cleaned data."]
    md = "\n".join(parts)
    (od / "KEY_FINDINGS.md").write_text(md)
    return md
