import * as api from "../api.js";
import { acKey, AIRCRAFT_ORDER, chip, copyButton, css, dataTable, fmt, h, icon, isFlight, sum, typeLabel } from "../ui.js";

const CODE = {
  python: `import pandas as pd

# one row per session (start here)
s = pd.read_csv("eplanes_sessions.csv", parse_dates=["start_utc", "end_utc"])
flights = s[s.session_type.str.startswith("flight test") & ~s.scale_anomaly & ~s.is_duplicate_content]

# 2 s traces for those flights (pyarrow reads only the matching row groups)
tr = pd.read_parquet("eplanes_traces_2s.parquet",
                     filters=[("flight_id", "in", flights.flight_id.head(20).tolist())])

# reported SOH per session, pack 1, one aircraft
soh = s[s.aircraft_label == "C-GAUW"].set_index("start_utc").b1_soh_median.sort_index()`,
  duckdb: `-- DuckDB reads the parquet files in place, no import step
SELECT aircraft_label,
       count(*)                          AS sessions,
       median(b1_soc_start - b1_soc_end) AS median_dod
FROM read_csv_auto('eplanes_sessions.csv')
WHERE session_type LIKE 'flight test%' AND NOT scale_anomaly
GROUP BY 1 ORDER BY 1;

SELECT flight_id, t_s, b1_current_a, b1_soc_pct, b1_max_cell_temp_c
FROM 'eplanes_traces_2s.parquet' WHERE flight_id = 4620 ORDER BY t_s;`,
  r: `library(arrow); library(dplyr)

s  <- read.csv("eplanes_sessions.csv")
tr <- open_dataset("eplanes_traces_2s.parquet") |>
      filter(flight_id == 4620) |>
      collect()`,
  clean: `import pyarrow.dataset as ds

# full-resolution (10 Hz) analysis-ready data for one aircraft
d = ds.dataset("eplanes_clean_10hz_C-GAUW.parquet")
one = d.to_table(filter=ds.field("flight_id") == 4620).to_pandas()

# honour the quality flags yourself: e.g. drop rows inside the start-up window
one = one[one.init_unsettled_groups == 0]`,
};

export async function renderDataset(app, anchor) {
  const [sessions, dict, dl, quality, aircraft] = await Promise.all([api.sessions(), api.get("dictionary"), api.get("downloads"), api.get("quality"), api.get("aircraft")]);
  app.innerHTML = "";
  const wrap = h("div", { class: "wrap" });
  app.append(wrap);
  const valid = sessions.filter((s) => s.start);
  const first = valid.length ? new Date(Math.min(...valid.map((s) => s.start))) : null, last = valid.length ? new Date(Math.max(...valid.map((s) => s.start))) : null;

  wrap.append(h("div", { class: "section", style: { paddingBottom: "6px" } },
    h("h1", {}, "The Eplanes dataset"),
    h("p", { class: "muted", style: { fontSize: "18px" } },
      `${fmt.int(sessions.length)} sessions from ${new Set(sessions.map((s) => s.aircraft_label)).size} Pipistrel Velis Electro aircraft, ${fmt.date(first)} to ${fmt.date(last)}: `,
      "battery pack voltage, current, state of charge, reported state of health and cell temperatures for both packs, the BMS's per-cell estimates, and flight data (outside air temperature, altitude, GPS) where the avionics were on."),
    h("div", { class: "chips", style: { marginTop: "14px" } },
      ["files", "raw-vs-clean", "dictionary", "quality", "load", "cite"].map((id) => h("a", { class: "chip", href: `#/dataset/${id}` },
        { files: "Download files", "raw-vs-clean": "Raw and cleaned", dictionary: "Data dictionary", quality: "Quality and preprocessing", load: "Load the data", cite: "Licence and citation" }[id])))));

  // ---- coverage summary
  const cov = h("table", { class: "data" }, h("thead", {}, h("tr", {}, h("th", {}, "Aircraft"), h("th", { class: "num" }, "Sessions"), h("th", { class: "num" }, "Flights"), h("th", { class: "num" }, "Charges"),
    h("th", { class: "num" }, "Other"), h("th", { class: "num" }, "Hours"), h("th", {}, "First session"), h("th", {}, "Last session"))));
  const tb = h("tbody");
  for (const a of AIRCRAFT_ORDER) {
    const g = sessions.filter((s) => s.aircraft_label === a);
    if (!g.length) continue;
    const st = g.filter((s) => s.start).map((s) => s.start);
    tb.append(h("tr", { style: { cursor: "default" } }, h("td", {}, acKey(a)), h("td", { class: "num" }, fmt.int(g.length)), h("td", { class: "num" }, fmt.int(g.filter((s) => isFlight(s.session_type)).length)),
      h("td", { class: "num" }, fmt.int(g.filter((s) => s.session_type === "charging").length)), h("td", { class: "num" }, fmt.int(g.filter((s) => !isFlight(s.session_type) && s.session_type !== "charging").length)),
      h("td", { class: "num" }, fmt.int(sum(g.map((s) => s.duration_min)) / 60)), h("td", {}, fmt.date(Math.min(...st))), h("td", {}, fmt.date(Math.max(...st)))));
  }
  cov.append(tb);
  wrap.append(h("section", { class: "section" }, h("h2", { style: { marginBottom: "12px" } }, "Coverage"), h("div", { class: "table-wrap" }, cov),
    h("p", { class: "note" }, "Dates are session start times from the portal, which match the aircraft's GPS clock (UTC). Collection is ongoing; a release is a frozen cut-off of the sessions processed at that time.")));

  // ---- files
  const files = h("section", { class: "section", id: "files" }, h("h2", { style: { marginBottom: "8px" } }, "Download files"));
  wrap.append(files);
  if (!dl.available) files.append(h("div", { class: "pending" }, dl.message || "No release has been built yet."));
  else {
    files.append(h("p", { class: "muted" }, `Release ${dl.version}, built ${fmt.dateTime(dl.created_utc)}. Every file has a SHA-256 checksum; large files are parquet so a few sessions can be read without loading everything.`));
    let kind = "";
    const kinds = [...new Set(dl.files.map((f) => f.kind))];
    const tableBox = h("div");
    const kchips = h("div", { class: "chips", style: { margin: "6px 0 12px" } });
    const draw = () => {
      tableBox.innerHTML = "";
      const rows = dl.files.filter((f) => !kind || f.kind === kind);
      tableBox.append(dataTable({ columns: [
        { key: "file", label: "File", render: (f) => h("span", { class: "mono" }, f.file) },
        { key: "kind", label: "Kind" }, { key: "subset", label: "Subset" },
        { key: "description", label: "Contents", cls: "wrap", sortable: false },
        { key: "bytes", label: "Size", num: true, render: (f) => fmt.bytes(f.bytes) },
        { key: "sha256", label: "SHA-256", sortable: false, render: (f) => h("span", { title: f.sha256, class: "mono faint" }, f.sha256.slice(0, 10) + "…") },
        { key: "get", label: "", sortable: false, render: (f) => (f.available && f.url ? h("a", { class: "btn small", href: f.url, download: f.file }, icon("download"), "Download") : h("span", { class: "faint", title: "Listed in the release manifest but not served by this site" }, "not hosted here")) },
      ], rows, pageSize: 30 }));
    };
    kchips.append(chip("All", true, () => {}), ...kinds.map((k) => chip(k, false, () => {})));
    kchips.querySelectorAll(".chip").forEach((c, i) => c.addEventListener("click", () => {
      kchips.querySelectorAll(".chip").forEach((x) => x.setAttribute("aria-pressed", "false"));
      c.setAttribute("aria-pressed", "true"); kind = i === 0 ? "" : kinds[i - 1]; draw();
    }));
    files.append(kchips, tableBox, h("p", { class: "note" }, "Subsets: per-aircraft files are split by aircraft; the flight list can also be filtered and exported as CSV from ", h("a", { href: "#/flights" }, "Flights"), ", and any single session's 2 s trace downloaded from its page."));
    draw();
  }

  // ---- raw vs clean
  wrap.append(h("section", { class: "section", id: "raw-vs-clean" }, h("h2", { style: { marginBottom: "12px" } }, "Raw and cleaned versions"),
    h("div", { class: "compare" },
      h("div", { class: "panel prose" }, h("h3", {}, "Raw"),
        h("p", {}, "The portal's own exports, byte for byte: per session a ZIP with the 10 Hz main log, one 5 Hz file per battery management system (BMS) and a warnings file."),
        h("ul", {}, h("li", {}, "Use it to reproduce the processing or to apply your own rules."),
          h("li", {}, "Values include the zeros written before channels report, avionics zeros when the avionics are off, and the 19 sessions with a scaling fault."),
          h("li", {}, "Longitude and latitude are unsigned magnitudes, exactly as logged."))),
      h("div", { class: "panel prose" }, h("h3", {}, "Cleaned (analysis-ready)"),
        h("p", {}, "The same rows and values with columns renamed to a consistent schema and information added, never removed:"),
        h("ul", {}, h("li", {}, "A per-row quality bitmask (out-of-range, gaps, duplicate time stamps, start-up window, invalid markers)."),
          h("li", {}, "Activity segments (charge, discharge, rest) and an avionics-valid flag."),
          h("li", {}, "Session and pack tables with derived quantities (depth of discharge, charge and energy throughput, peak power, temperatures) and quality flags."),
          h("li", {}, "2 s traces with start-up zeros and avionics-off channels set to missing and GPS signs recovered."))))));

  // ---- dictionary
  const dictBox = h("div");
  let q = "";
  const dDraw = () => {
    dictBox.innerHTML = "";
    const rows = dict.filter((r) => !q || `${r.column} ${r.description} ${r.table}`.toLowerCase().includes(q.toLowerCase()));
    dictBox.append(dataTable({ columns: [{ key: "table", label: "Table" }, { key: "column", label: "Column", render: (r) => h("span", { class: "mono" }, r.column) }, { key: "unit", label: "Unit" }, { key: "description", label: "Description", cls: "wrap", sortable: false }], rows, pageSize: 25 }));
  };
  wrap.append(h("section", { class: "section", id: "dictionary" }, h("div", { class: "section-head" }, h("h2", {}, "Data dictionary"),
    h("label", { class: "field" }, "Search columns", h("input", { type: "search", placeholder: "e.g. soc, temperature", oninput: (e) => { q = e.target.value; dDraw(); } }))), dictBox,
    h("p", { class: "note" }, "Units the portal does not document were inferred from magnitudes and cross-checked: cell voltage in mV, pack voltage in V, current in A with charging negative, temperatures in °C. 96 cells are in series in each pack.")));
  dDraw();

  // ---- quality
  const g = quality.gate || {};
  const qs = h("section", { class: "section", id: "quality" }, h("h2", { style: { marginBottom: "10px" } }, "Quality checks and preprocessing"));
  qs.append(h("div", { class: "grid two" },
    checkedPanel(g.findings || []),
    h("div", { class: "panel prose" }, h("h3", {}, "Unreliable first readings"),
      h("p", {}, "Most sessions open with rows of exact zeros written before the channels report; temperatures can stay at zero for up to a minute. Every session and signal was classified from the data (zero-fill, settling, load onset, stable)."),
      h("p", {}, "The cleaned traces mask each signal's leading run of zeros (rule ", h("span", { class: "mono" }, "leadzero"), "). Sessions that start under load are never masked, because a real transient cannot be excluded. A sensitivity analysis shows the published statistics change by at most a few percent across reasonable rules, but are badly wrong without any rule."),
      h("h3", {}, "Limitations"), h("ul", {}, (g.limitations || []).map((x) => h("li", {}, x))))));
  wrap.append(qs);

  // ---- load
  const tabs = h("div", { class: "tabs", role: "tablist" });
  const codeBox = h("div");
  const show = (k) => {
    tabs.querySelectorAll("button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.k === k)));
    codeBox.innerHTML = "";
    const pre = h("pre", { class: "code" }, CODE[k]);
    pre.append(copyButton(() => CODE[k]));
    codeBox.append(pre);
  };
  for (const [k, l] of [["python", "Python"], ["duckdb", "DuckDB SQL"], ["r", "R"], ["clean", "10 Hz data"]]) tabs.append(h("button", { role: "tab", "data-k": k, type: "button", onclick: () => show(k) }, l));
  wrap.append(h("section", { class: "section", id: "load" }, h("h2", { style: { marginBottom: "10px" } }, "Load the data"),
    h("div", { class: "grid side" }, h("div", { class: "panel" }, tabs, codeBox),
      h("div", { class: "panel prose" }, h("h3", {}, "Choosing a subset"), h("ul", {},
        h("li", {}, "Battery health over time: the session table, one row per session; filter out ", h("span", { class: "mono" }, "scale_anomaly"), " and duplicates."),
        h("li", {}, "Flight power and energy: flight tests with ", h("span", { class: "mono" }, "usable_B_profiles"), "."),
        h("li", {}, "Charging behaviour and capacity indicators: charging sessions with ", h("span", { class: "mono" }, "usable_D_charging"), " or ", h("span", { class: "mono" }, "usable_G_capacity"), "."),
        h("li", {}, "Thermal studies: sessions with outside air temperature (avionics on), or cell temperature alone otherwise."),
        h("li", {}, "Model benchmarks: split by aircraft or by time, never randomly by sample, because consecutive samples and sessions are dependent."))))));
  show("python");

  // ---- cite
  const lic = dl.available ? dl.licence : null;
  wrap.append(h("section", { class: "section", id: "cite" }, h("h2", { style: { marginBottom: "10px" } }, "Licence and citation"),
    h("div", { class: "grid two" },
      h("div", { class: "panel prose" }, h("h3", {}, "Licence"),
        !lic || /confirm/i.test(lic) ? h("div", { class: "pending" }, "The licence will be stated here once the data owners have confirmed the terms of release. Until then the files are for review only.") : h("p", {}, lic),
        h("h3", {}, "Version"), h("p", {}, dl.available ? `Release ${dl.version}, ${fmt.int(dl.n_sessions)} sessions, last session ${fmt.date(dl.last_session_utc)}.` : "No release built yet.")),
      h("div", { class: "panel prose" }, h("h3", {}, "How to cite"),
        dl.available && dl.doi ? h("p", {}, `DOI: ${dl.doi}`) : h("div", { class: "pending" }, "A DOI and the paper reference will appear here at publication. Please cite the dataset paper once it is available."),
        h("p", { class: "faint", style: { marginTop: "10px" } }, "Interpreting State-of-Health Patterns in Electric Aircraft Operations: An Open Field Dataset and Analysis (manuscript in preparation).")))));

  if (anchor) setTimeout(() => { const el = document.getElementById(anchor); if (el) el.scrollIntoView({ block: "start" }); }, 30);
}

function checkedPanel(findings) {
  const list = h("ul", {});
  const more = h("button", { class: "btn small", type: "button" }, `Show all ${findings.length} checks`);
  const draw = (all) => { list.innerHTML = ""; (all ? findings : findings.slice(0, 6)).forEach((x) => list.append(h("li", {}, x))); more.hidden = all || findings.length <= 6; };
  more.addEventListener("click", () => draw(true));
  draw(false);
  return h("div", { class: "panel prose" }, h("h3", {}, "What was checked"),
    h("p", { class: "muted" }, "Automatic findings of the quality stage, in the words of the pipeline report."), list, more);
}
