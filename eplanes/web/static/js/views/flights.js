import * as api from "../api.js";
import { axis, legend, make, tooltip, emptyChart, tokens } from "../charts.js";
import { addAirports, addTracks, canvasFallback, createMap, fit, highlight } from "../map.js";
import { acColor, acKey, acSymbol, AIRCRAFT_ORDER, callout, chip, css, dataTable, errorBox, fmt, h, icon, isFlight, typeLabel } from "../ui.js";

const TYPES = ["flight test", "charging", "ground test", "flight test and charging", "ground test and charging"];

function readFilters(params, all) {
  const acs = AIRCRAFT_ORDER.filter((a) => all.some((s) => s.aircraft_label === a));
  return {
    aircraft: params.get("aircraft") ? params.get("aircraft").split(",") : acs,
    type: params.get("type") || "",
    from: params.get("from") || "",
    to: params.get("to") || "",
    dmin: params.get("dmin") || "",
    dmax: params.get("dmax") || "",
    gps: params.get("gps") === "1",
    clean: params.get("clean") === "1",
    q: params.get("q") || "",
  };
}

function writeFilters(f, all) {
  const acs = AIRCRAFT_ORDER.filter((a) => all.some((s) => s.aircraft_label === a));
  const p = new URLSearchParams();
  if (f.aircraft.length !== acs.length) p.set("aircraft", f.aircraft.join(","));
  for (const k of ["type", "from", "to", "dmin", "dmax", "q"]) if (f[k]) p.set(k, f[k]);
  if (f.gps) p.set("gps", "1");
  if (f.clean) p.set("clean", "1");
  const qs = p.toString();
  history.replaceState(null, "", `#/flights${qs ? "?" + qs : ""}`);
}

export function applyFilters(all, f) {
  const from = f.from ? Date.parse(f.from + "T00:00:00Z") : null;
  const to = f.to ? Date.parse(f.to + "T23:59:59Z") : null;
  const q = f.q.trim();
  return all.filter((s) => f.aircraft.includes(s.aircraft_label)
    && (!f.type || (f.type === "flight" ? isFlight(s.session_type) : s.session_type === f.type))
    && (!from || (s.start && s.start >= from)) && (!to || (s.start && s.start <= to))
    && (!f.dmin || (s.duration_min ?? 0) >= +f.dmin) && (!f.dmax || (s.duration_min ?? 0) <= +f.dmax)
    && (!f.gps || s.has_gps_track) && (!f.clean || (!s.scale_anomaly && !s.is_duplicate_content && !s.label_content_mismatch))
    && (!q || String(s.flight_id).includes(q) || (s.dep_airport || "").includes(q.toUpperCase()) || (s.arr_airport || "").includes(q.toUpperCase())));
}

function flags(s) {
  const out = [];
  if (s.scale_anomaly) out.push(h("span", { class: "tag serious", title: "Current/voltage scaling fault" }, "scale fault"));
  if (s.is_duplicate_content) out.push(h("span", { class: "tag caution", title: "Duplicate of an earlier session" }, "duplicate"));
  if (s.label_content_mismatch) out.push(h("span", { class: "tag caution", title: "Portal label contradicts the measured current" }, "label?"));
  if (s.has_gap) out.push(h("span", { class: "tag", title: `${s.n_gaps} logging gap(s)` }, "gap"));
  if (s.b1_uniform_cells || s.b2_uniform_cells) out.push(h("span", { class: "tag caution", title: "BMS estimator reset signature" }, "BMS reset"));
  return out.length ? h("span", { style: { display: "inline-flex", gap: "4px" } }, out) : "";
}

export async function renderFlights(app, params) {
  const all = await api.sessions();
  const f = readFilters(params, all);
  const selected = new Set();
  app.innerHTML = "";
  const wrap = h("div", { class: "wrap" });
  app.append(wrap);
  wrap.append(h("div", { class: "section", style: { paddingBottom: "14px" } },
    h("h1", {}, "Flights and sessions"),
    h("p", { class: "muted" }, "Every recorded session: flights, charges and ground runs. Filter, sort, pick a session to see its telemetry and route, or select up to four to compare.")));

  // ---- filters
  const acs = AIRCRAFT_ORDER.filter((a) => all.some((s) => s.aircraft_label === a));
  const summary = h("p", { class: "active-filters", "aria-live": "polite" });
  const update = () => { writeFilters(f, all); redraw(); };
  const acChips = h("div", { class: "chips", role: "group", "aria-label": "Aircraft" },
    acs.map((a) => chip(a, f.aircraft.includes(a), (on) => { f.aircraft = on ? [...new Set([...f.aircraft, a])] : f.aircraft.filter((x) => x !== a); update(); }, acColor(a))));
  const typeSel = h("select", { "aria-label": "Session type", onchange: (e) => { f.type = e.target.value; update(); } },
    h("option", { value: "" }, "All types"), h("option", { value: "flight" }, "All flights"),
    TYPES.filter((t) => all.some((s) => s.session_type === t)).map((t) => h("option", { value: t }, typeLabel(t))));
  typeSel.value = f.type;
  const dates = all.filter((s) => s.start).map((s) => s.start);
  const minD = dates.length ? fmt.dateIso(Math.min(...dates)) : "", maxD = dates.length ? fmt.dateIso(Math.max(...dates)) : "";
  const fromIn = h("input", { type: "date", min: minD, max: maxD, value: f.from, onchange: (e) => { f.from = e.target.value; update(); } });
  const toIn = h("input", { type: "date", min: minD, max: maxD, value: f.to, onchange: (e) => { f.to = e.target.value; update(); } });
  const dmin = h("input", { type: "number", min: 0, step: 5, value: f.dmin, placeholder: "0", onchange: (e) => { f.dmin = e.target.value; update(); } });
  const dmax = h("input", { type: "number", min: 0, step: 5, value: f.dmax, placeholder: "any", onchange: (e) => { f.dmax = e.target.value; update(); } });
  const q = h("input", { type: "search", value: f.q, placeholder: "ID or airfield", oninput: (e) => { f.q = e.target.value; update(); } });
  const gps = h("input", { type: "checkbox", checked: f.gps ? true : null, onchange: (e) => { f.gps = e.target.checked; update(); } });
  const clean = h("input", { type: "checkbox", checked: f.clean ? true : null, onchange: (e) => { f.clean = e.target.checked; update(); } });
  const presets = h("div", { class: "chips" },
    [["Last 90 days", 90], ["Last year", 365], ["All", 0]].map(([l, d]) => h("button", { class: "chip", type: "button", onclick: () => {
      f.to = ""; f.from = d ? fmt.dateIso(Math.max(...dates) - d * 864e5) : ""; fromIn.value = f.from; toIn.value = ""; update(); } }, l)));
  wrap.append(h("div", { class: "filters", role: "search" },
    h("div", { class: "field" }, "Aircraft", acChips),
    h("label", { class: "field" }, "Type", typeSel),
    h("div", { class: "field" }, "Date range (UTC)", h("div", { style: { display: "flex", gap: "6px", alignItems: "center" } }, fromIn, h("span", { class: "faint" }, "to"), toIn)),
    h("div", { class: "field" }, "Quick range", presets),
    h("div", { class: "field" }, "Duration (min)", h("div", { style: { display: "flex", gap: "6px", alignItems: "center" } }, dmin, h("span", { class: "faint" }, "to"), dmax)),
    h("label", { class: "field" }, "Search", q),
    h("div", { class: "field" }, "Only", h("div", { style: { display: "flex", gap: "12px" } },
      h("label", { class: "check" }, gps, "With GPS track"), h("label", { class: "check", title: "Excludes scale faults, duplicates and mislabelled sessions" }, clean, "Without data faults")))),
    summary);

  // ---- timeline
  const chartEl = h("div", { class: "chart" });
  wrap.append(h("div", { class: "panel", style: { marginTop: "16px" } },
    h("h3", {}, "Session timeline"), h("p", { class: "sub" }, "Duration against date. Drag across the chart to narrow the date range; click a point to open the session."), chartEl));

  // ---- table
  const actions = h("div", { style: { display: "flex", gap: "8px", flexWrap: "wrap" } });
  const compareBtn = h("button", { class: "btn", type: "button", disabled: true, onclick: () => { location.hash = `#/flights/compare/${[...selected].join(",")}`; } }, "Compare selected");
  const dlBtn = h("button", { class: "btn", type: "button", onclick: () => {
    const rows = applyFilters(all, f);
    api.downloadCsv(`eplanes_sessions_${rows.length}.csv`, rows, Object.keys(rows[0] || {}).filter((k) => k !== "start"));
  } }, icon("download"), "Download these rows (CSV)");
  actions.append(compareBtn, dlBtn);
  const cols = [
    { key: "select", label: "", sortable: false, render: (s) => {
      const cb = h("input", { type: "checkbox", "aria-label": `Select session ${s.flight_id}`, checked: selected.has(s.flight_id) ? true : null });
      cb.addEventListener("change", () => { if (cb.checked) { if (selected.size >= 4) { cb.checked = false; return; } selected.add(s.flight_id); } else selected.delete(s.flight_id);
        compareBtn.disabled = selected.size < 2; compareBtn.textContent = selected.size >= 2 ? `Compare ${selected.size} sessions` : "Compare selected"; });
      return cb; } },
    { key: "flight_id", label: "Session", render: (s) => h("a", { href: `#/flights/${s.flight_id}` }, s.flight_id) },
    { key: "aircraft_label", label: "Aircraft", render: (s) => acKey(s.aircraft_label) },
    { key: "session_type", label: "Type", render: (s) => typeLabel(s.session_type) },
    { key: "start", label: "Start (UTC)", render: (s) => fmt.dateTime(s.start_utc), sortValue: (s) => s.start },
    { key: "duration_min", label: "Duration", num: true, render: (s) => fmt.dur(s.duration_min) },
    { key: "soc", label: "SOC pack 1", num: true, render: (s) => (s.b1_soc_start !== null ? `${fmt.int(s.b1_soc_start)} → ${fmt.int(s.b1_soc_end)}%` : "–"), sortValue: (s) => s.b1_soc_start },
    { key: "energy", label: "Energy out", num: true, render: (s) => (s.b1_energy_discharged_kwh !== null && !s.scale_anomaly ? `${fmt.num((s.b1_energy_discharged_kwh || 0) + (s.b2_energy_discharged_kwh || 0), 1)} kWh` : "–"),
      sortValue: (s) => (s.scale_anomaly ? null : (s.b1_energy_discharged_kwh || 0) + (s.b2_energy_discharged_kwh || 0)) },
    { key: "b1_temp_max_c", label: "Max cell", num: true, render: (s) => (s.b1_temp_max_c !== null ? `${fmt.int(s.b1_temp_max_c)} °C` : "–") },
    { key: "b1_soh_median", label: "SOH 1 / 2", num: true, render: (s) => (s.b1_soh_median !== null ? `${fmt.int(s.b1_soh_median)} / ${fmt.int(s.b2_soh_median)}%` : "–") },
    { key: "route", label: "Airfields", render: (s) => (s.dep_airport || s.arr_airport ? `${s.dep_airport || "?"} → ${s.arr_airport || "?"}` : h("span", { class: "faint" }, s.has_gps_track ? "unmatched" : "no GPS")), sortValue: (s) => s.dep_airport },
    { key: "flags", label: "Flags", sortable: false, render: flags },
  ];
  const table = dataTable({ columns: cols, rows: [], pageSize: 50, initialSort: { key: "start", dir: "desc" }, onRow: (s) => { location.hash = `#/flights/${s.flight_id}`; }, caption: "Sessions" });
  wrap.append(h("div", { class: "section-head", style: { marginTop: "28px", marginBottom: "10px" } }, h("h2", {}, "Session list"), actions), table);

  let chart = null;
  function redraw() {
    const rows = applyFilters(all, f);
    const parts = [];
    parts.push(h("b", {}, fmt.int(rows.length)), ` of ${fmt.int(all.length)} sessions`);
    parts.push(` · ${f.aircraft.length === acs.length ? "all aircraft" : f.aircraft.join(", ") || "no aircraft"}`);
    parts.push(` · ${f.type ? (f.type === "flight" ? "all flights" : typeLabel(f.type).toLowerCase()) : "all types"}`);
    parts.push(` · ${f.from || minD} to ${f.to || maxD}`);
    if (f.dmin || f.dmax) parts.push(` · ${f.dmin || 0}–${f.dmax || "any"} min`);
    if (f.gps) parts.push(" · with GPS");
    if (f.clean) parts.push(" · without data faults");
    summary.innerHTML = "";
    summary.append(...parts);
    table.update(rows);
    drawTimeline(rows);
  }
  function drawTimeline(rows) {
    const byAc = AIRCRAFT_ORDER.filter((a) => rows.some((s) => s.aircraft_label === a));
    if (!rows.length) { if (chart) { chart.dispose(); chart = null; } emptyChart(chartEl, "No sessions match these filters."); return; }
    chartEl.style.display = ""; chartEl.innerHTML = "";
    if (chart) chart.dispose();
    chart = make(chartEl, (t) => ({
      grid: { left: 52, right: 16, top: 44, bottom: 36 },
      legend: legend(t, { data: byAc, left: "center" }),
      tooltip: tooltip(t, { trigger: "item", formatter: (p) => `<b>Session ${p.value[2]}</b> · ${p.seriesName}<br>${typeLabel(p.value[3])}<br>${fmt.dateTime(p.value[0])}<br>${fmt.dur(p.value[1])}` }),
      brush: { toolbox: ["lineX", "clear"], xAxisIndex: 0, brushStyle: { color: "rgba(70,200,232,0.12)", borderColor: t.accent } },
      toolbox: { show: false },
      xAxis: { type: "time", ...axis(t, { splitLine: { show: false } }) },
      yAxis: { type: "value", name: "minutes", ...axis(t) },
      series: byAc.map((a) => ({ name: a, type: "scatter", symbol: acSymbol(a), symbolSize: 7, large: rows.length > 3000,
        itemStyle: { color: acColor(a), opacity: 0.85, borderColor: t.panel, borderWidth: 0.5 },
        data: rows.filter((s) => s.aircraft_label === a && s.start).map((s) => [s.start, s.duration_min, s.flight_id, s.session_type]) })),
    }));
    if (!chart) return;
    chart.dispatchAction({ type: "takeGlobalCursor", key: "brush", brushOption: { brushType: "lineX", brushMode: "single" } });
    chart.on("brushEnd", (e) => {
      const area = e.areas && e.areas[0];
      if (!area || !area.coordRange) return;
      const [a, b] = area.coordRange;
      f.from = fmt.dateIso(a); f.to = fmt.dateIso(b); fromIn.value = f.from; toIn.value = f.to;
      update();
    });
    chart.on("click", (p) => { if (p.value) location.hash = `#/flights/${p.value[2]}`; });
  }
  redraw();
}

// ------------------------------------------------------------------------------------------------ detail
const PANELS = [
  { title: "Current", unit: "A", cols: [["b1_current_a", "Pack 1"], ["b2_current_a", "Pack 2"]], note: "positive = discharge" },
  { title: "State of charge", unit: "%", cols: [["b1_soc_pct", "Pack 1"], ["b2_soc_pct", "Pack 2"]] },
  { title: "Cell voltage, min and max", unit: "mV", cols: [["b1_min_cell_mv", "Pack 1 min"], ["b1_max_cell_mv", "Pack 1 max"], ["b2_min_cell_mv", "Pack 2 min"], ["b2_max_cell_mv", "Pack 2 max"]] },
  { title: "Maximum cell temperature", unit: "°C", cols: [["b1_max_cell_temp_c", "Pack 1"], ["b2_max_cell_temp_c", "Pack 2"], ["oat_c", "Outside air"]] },
  { title: "Pressure altitude", unit: "m", cols: [["pressure_alt_m", "Altitude"]] },
  { title: "Reported state of health", unit: "%", cols: [["b1_soh_pct", "Pack 1"], ["b2_soh_pct", "Pack 2"]] },
];

function lineColor(t, col) {
  if (col.startsWith("b1")) return t.pack1;
  if (col.startsWith("b2")) return t.pack2;
  if (col === "oat_c") return t.ink3;
  return t.accent;
}

export function telemetry(el, series, { title = true, height = 150 } = {}) {
  const t = tokens();
  const present = PANELS.map((p) => ({ ...p, cols: p.cols.filter(([c]) => series.columns[c]) })).filter((p) => p.cols.length);
  const missing = PANELS.flatMap((p) => p.cols.filter(([c]) => !series.columns[c]).map(([, l]) => `${p.title.toLowerCase()} (${l.toLowerCase()})`));
  el.style.height = `${present.length * height + 60}px`;
  const x = series.t_s.map((s) => s / 60);
  const grids = present.map((p, i) => ({ left: 62, right: 18, top: 28 + i * height, height: height - 46 }));
  make(el, () => ({
    axisPointer: { link: [{ xAxisIndex: "all" }], lineStyle: { color: t.accent } },
    tooltip: tooltip(t, { trigger: "axis", axisPointer: { type: "line" }, formatter: (ps) => `<b>${fmt.num(ps[0].axisValue, 1)} min</b><br>` + ps.filter((p) => p.value[1] !== null).map((p) => `${p.marker}${p.seriesName}: ${fmt.num(p.value[1], 1)}`).join("<br>") }),
    dataZoom: [{ type: "inside", xAxisIndex: present.map((_, i) => i) }, { type: "slider", xAxisIndex: present.map((_, i) => i), bottom: 4, height: 18, borderColor: t.line, textStyle: { color: t.ink3 }, fillerColor: "rgba(70,200,232,0.12)", handleStyle: { color: t.accent } }],
    grid: grids,
    title: present.map((p, i) => ({ text: `${p.title} (${p.unit})`, left: 62, top: 6 + i * height, textStyle: { color: t.ink2, fontSize: 13, fontWeight: 600, fontFamily: t.display } })),
    legend: present.map((p, i) => ({ top: 6 + i * height, right: 18, itemWidth: 14, itemHeight: 6, textStyle: { color: t.ink3, fontSize: 11 }, data: p.cols.map(([c, l]) => `${l}·${i}`), formatter: (n) => n.split("·")[0] })),
    xAxis: present.map((_, i) => ({ type: "value", gridIndex: i, min: 0, max: Math.ceil(x[x.length - 1]), ...axis(t, { splitLine: { show: false }, axisLabel: { color: t.ink3, show: i === present.length - 1, showMaxLabel: false, formatter: (v) => `${Math.round(v)} min` } }) })),
    yAxis: present.map((p, i) => (p.unit === "%" ? { type: "value", gridIndex: i, max: 100, min: p.title.startsWith("State of charge") ? 0 : (v) => Math.max(0, Math.floor((v.min - 5) / 10) * 10), ...axis(t, { splitNumber: 3 }) }
      : { type: "value", gridIndex: i, scale: true, ...axis(t, { splitNumber: 3 }) })),
    series: present.flatMap((p, i) => p.cols.map(([c, l]) => ({
      name: `${l}·${i}`, type: "line", xAxisIndex: i, yAxisIndex: i, showSymbol: false, sampling: "lttb", connectNulls: false,
      step: c.includes("soc") || c.includes("soh") ? "end" : false,
      lineStyle: { width: 1.6, color: lineColor(t, c), type: c.startsWith("b2") || c.includes("max_cell_mv") && c.startsWith("b1") ? (c.startsWith("b2") ? [5, 3] : "solid") : "solid", opacity: c.includes("min_cell") ? 0.7 : 1 },
      itemStyle: { color: lineColor(t, c) },
      data: x.map((xi, k) => [xi, series.columns[c][k]]),
    }))),
  }));
  return missing;
}

export async function renderFlightDetail(app, fid) {
  let d;
  try { d = await api.get(`sessions/${fid}`); }
  catch (e) {
    app.innerHTML = "";
    app.append(h("div", { class: "wrap section" }, e.status === 404 ? h("div", { class: "empty" }, h("p", {}, `Session ${fid} is not in the processed dataset. It may be new on the portal and not yet processed.`), h("a", { href: "#/flights" }, "Back to all sessions")) : errorBox(e)));
    return;
  }
  const s = d.session;
  app.innerHTML = "";
  const wrap = h("div", { class: "wrap" });
  app.append(wrap);
  wrap.append(h("div", { class: "crumbs" }, h("a", { href: "#/flights" }, "Flights"), h("span", {}, "/"), h("a", { href: `#/aircraft/${s.aircraft_label}` }, s.aircraft_label), h("span", {}, "/"), h("span", {}, `Session ${fid}`)));
  wrap.append(h("div", { class: "detail-head" },
    h("div", {}, h("h1", {}, `${typeLabel(s.session_type)}, ${fmt.date(s.start_utc)}`),
      h("div", { class: "meta" }, acKey(s.aircraft_label), `  ·  Session ${fid}  ·  ${fmt.dateTime(s.start_utc)}  ·  ${fmt.dur(s.duration_min)}`,
        s.dep_airport || s.arr_airport ? `  ·  ${s.dep_airport || "?"} → ${s.arr_airport || "?"}` : "")),
    h("div", { style: { display: "flex", gap: "8px", flexWrap: "wrap" } },
      d.neighbours.previous ? h("a", { class: "btn small", href: `#/flights/${d.neighbours.previous}` }, "Previous session") : null,
      d.neighbours.next ? h("a", { class: "btn small", href: `#/flights/${d.neighbours.next}` }, "Next session") : null,
      d.has_series ? h("button", { class: "btn small", type: "button", onclick: () => downloadTrace(fid) }, icon("download"), "Trace (CSV)") : null)));
  for (const n of d.notes) wrap.append(callout(n.level === "serious" ? "serious" : n.level === "warning" ? "warning" : "info", n.text));

  const grid = h("div", { class: "grid side", style: { marginTop: "16px" } });
  wrap.append(grid);
  const mapBox = h("div", { class: "mini-map" });
  grid.append(h("div", { class: "panel" }, h("h3", {}, "Route"), h("p", { class: "sub" }, s.has_gps_track ? `${fmt.num(s.track_km, 1)} km of GPS track${s.alt_gain_m ? `, ${fmt.int(s.alt_gain_m)} m above the field at most` : ""}.` : "No GPS track: the avionics block was off or logged no position."), mapBox));
  grid.append(h("div", { class: "panel" }, h("h3", {}, "Per pack"), h("p", { class: "sub" }, "Values after start-up masking. Energy from pack voltage × current."), packTable(d.packs, s)));

  const telEl = h("div", { class: "chart" });
  const telPanel = h("div", { class: "panel", style: { marginTop: "20px" } }, h("h3", {}, "Telemetry"), h("p", { class: "sub" }, `Resampled to ${d.session ? "2 s" : ""} bins; scroll or drag the bar below to zoom. Gaps are periods with no logged samples.`), telEl);
  wrap.append(telPanel);
  if (d.soh_steps && d.soh_steps.length) {
    wrap.append(h("div", { class: "panel", style: { marginTop: "20px" } }, h("h3", {}, "Reported SOH changes inside this session"),
      h("p", { class: "sub" }, "The BMS revised its state-of-health estimate during the session at these moments."),
      dataTable({ columns: [
        { key: "battery", label: "Pack" }, { key: "t_min", label: "Time", num: true, render: (r) => `${fmt.num(r.t_min, 1)} min` },
        { key: "delta", label: "Change", num: true, render: (r) => `${r.soh_before} → ${r.soh_after}%` },
        { key: "soc_at_step", label: "SOC at step", num: true, render: (r) => `${fmt.int(r.soc_at_step)}%` },
        { key: "max_cell_temp_c", label: "Max cell temp", num: true, render: (r) => `${fmt.int(r.max_cell_temp_c)} °C` },
        { key: "current_a", label: "Current", num: true, render: (r) => `${fmt.num(r.current_a, 1)} A` }], rows: d.soh_steps, pageSize: 20 })));
  }

  const [ser, tracks, airports] = await Promise.all([d.has_series ? api.series(fid, 3000) : null, api.get("tracks"), api.get("airports")]);
  if (ser) {
    const missing = telemetry(telEl, ser);
    if (missing.length) telPanel.append(h("p", { class: "missing-list" }, `Not logged in this session: ${missing.join(", ")}.`));
  } else emptyChart(telEl, "No telemetry stored for this session.");
  const tr = tracks.find((t) => t.flight_id === fid);
  if (!tr) { mapBox.append(h("div", { class: "map-empty" }, "No position data for this session.")); return; }
  const near = airports.filter((a) => a.code === s.dep_airport || a.code === s.arr_airport);
  const map = createMap(mapBox);
  if (!map) { canvasFallback(mapBox, [tr], near, { highlightId: fid }); return; }
  map.on("load", () => { addTracks(map, [tr], { colorBy: "route", width: 2.4, opacity: 1 }); addAirports(map, near); fit(map, [tr.coords], 30); highlight(map, null); });
}

function packTable(packs, s) {
  const p = Object.fromEntries((packs || []).map((r) => [r.battery, r]));
  const rows = [
    ["SOC start → end", (r) => (r.soc_start !== null ? `${fmt.int(r.soc_start)} → ${fmt.int(r.soc_end)}%` : "–")],
    ["Charge out / in", (r) => `${fmt.num(r.ah_discharged, 1)} / ${fmt.num(r.ah_charged, 1)} Ah`],
    ["Energy out / in", (r) => `${fmt.num(r.energy_discharged_kwh, 2)} / ${fmt.num(r.energy_charged_kwh, 2)} kWh`],
    ["Peak discharge current", (r) => `${fmt.num(r.i_peak_discharge_a, 0)} A`],
    ["Peak power", (r) => `${fmt.num(r.power_peak_kw, 1)} kW`],
    ["Cell temperature start → max", (r) => (r.temp_start_c !== null ? `${fmt.int(r.temp_start_c)} → ${fmt.int(r.temp_max_c)} °C` : "–")],
    ["Reported SOH (median)", (r) => (r.soh_median !== null ? `${fmt.int(r.soh_median)}%` : "–")],
    ["Weakest / mean cell SOH (BMS)", (r) => (r.soh_cell_cells_last_min !== null && r.soh_cell_cells_last_min !== undefined ? `${fmt.num(r.soh_cell_cells_last_min, 1)} / ${fmt.num(r.soh_cell_cells_last_mean, 1)}%` : "–")],
  ];
  const kv = h("div", { class: "kv" }, h("div", { class: "h" }, ""), h("div", { class: "h r" }, "Pack 1"), h("div", { class: "h r" }, "Pack 2"));
  for (const [label, f] of rows) kv.append(h("div", {}, label), h("div", { class: "r" }, p[1] ? f(p[1]) : "–"), h("div", { class: "r" }, p[2] ? f(p[2]) : "–"));
  const box = h("div", {}, kv);
  if (s.scale_anomaly) box.append(h("p", { class: "note" }, "Electrical values in this session are mis-scaled and not physical."));
  return box;
}

async function downloadTrace(fid) {
  const ser = await api.series(fid, 20000);
  const cols = Object.keys(ser.columns);
  const rows = ser.t_s.map((t, i) => Object.fromEntries([["flight_id", fid], ["t_s", t], ...cols.map((c) => [c, ser.columns[c][i]])]));
  api.downloadCsv(`eplanes_session_${fid}_2s.csv`, rows, ["flight_id", "t_s", ...cols]);
}

// ------------------------------------------------------------------------------------------------ compare
export async function renderCompare(app, ids) {
  ids = ids.slice(0, 4);
  const details = await Promise.all(ids.map((i) => api.get(`sessions/${i}`).catch(() => null)));
  const sers = await Promise.all(ids.map((i, k) => (details[k] && details[k].has_series ? api.series(i, 1500) : null)));
  app.innerHTML = "";
  const wrap = h("div", { class: "wrap" });
  app.append(wrap);
  wrap.append(h("div", { class: "crumbs" }, h("a", { href: "#/flights" }, "Flights"), h("span", {}, "/"), h("span", {}, "Compare")));
  wrap.append(h("h1", { style: { marginBottom: "8px" } }, `Comparing ${ids.length} sessions`));
  wrap.append(h("p", { class: "muted" }, "Pack 1 of each session against elapsed time. Colours here mark the session, not the aircraft."));
  const palette = ["--s1", "--s2", "--s3", "--route"].map(css);
  const legendRow = h("div", { class: "chips", style: { margin: "10px 0 16px" } }, ids.map((id, k) => details[k] ? h("a", { class: "chip", href: `#/flights/${id}`, "aria-pressed": "true" },
    h("span", { class: "key", style: { background: palette[k] } }), `${id} · ${details[k].session.aircraft_label} · ${typeLabel(details[k].session.session_type)} · ${fmt.date(details[k].session.start_utc)}`) : h("span", { class: "chip" }, `${id}: not found`)));
  wrap.append(legendRow);
  const grid = h("div", { class: "grid two" });
  wrap.append(grid);
  for (const [col, title, unit] of [["b1_current_a", "Current, pack 1", "A"], ["b1_soc_pct", "State of charge, pack 1", "%"], ["b1_max_cell_temp_c", "Maximum cell temperature, pack 1", "°C"], ["pressure_alt_m", "Pressure altitude", "m"]]) {
    const el = h("div", { class: "chart" });
    grid.append(h("div", { class: "panel" }, h("h3", {}, `${title} (${unit})`), el));
    const have = sers.some((s) => s && s.columns[col]);
    if (!have) { emptyChart(el, "Not logged in these sessions."); continue; }
    make(el, (t) => ({
      grid: { left: 48, right: 12, top: 16, bottom: 34 },
      tooltip: tooltip(t, { trigger: "axis", formatter: (ps) => `<b>${fmt.num(ps[0].value[0], 1)} min</b><br>` + ps.map((p) => `${p.marker}${p.seriesName}: ${fmt.num(p.value[1], 1)}`).join("<br>") }),
      dataZoom: [{ type: "inside" }],
      xAxis: { type: "value", name: "min", ...axis(t, { splitLine: { show: false } }) },
      yAxis: { type: "value", scale: true, ...axis(t) },
      series: sers.map((s, k) => (s && s.columns[col] ? { name: String(ids[k]), type: "line", showSymbol: false, lineStyle: { width: 1.6, color: palette[k] }, itemStyle: { color: palette[k] },
        data: s.t_s.map((ts, i) => [ts / 60, s.columns[col][i]]) } : null)).filter(Boolean),
    }));
  }
  const rows = details.filter(Boolean).map((d) => d.session);
  wrap.append(h("div", { class: "panel", style: { marginTop: "20px" } }, h("h3", {}, "Summary"), dataTable({ columns: [
    { key: "flight_id", label: "Session" }, { key: "aircraft_label", label: "Aircraft" }, { key: "session_type", label: "Type", render: (s) => typeLabel(s.session_type) },
    { key: "duration_min", label: "Duration", num: true, render: (s) => fmt.dur(s.duration_min) },
    { key: "soc", label: "SOC pack 1", num: true, render: (s) => `${fmt.int(s.b1_soc_start)} → ${fmt.int(s.b1_soc_end)}%` },
    { key: "b1_energy_discharged_kwh", label: "Energy out pack 1", num: true, render: (s) => `${fmt.num(s.b1_energy_discharged_kwh, 2)} kWh` },
    { key: "b1_i_peak_discharge_a", label: "Peak current", num: true, render: (s) => `${fmt.num(s.b1_i_peak_discharge_a, 0)} A` },
    { key: "b1_temp_max_c", label: "Max cell", num: true, render: (s) => `${fmt.int(s.b1_temp_max_c)} °C` },
    { key: "oat_c_median", label: "Outside air", num: true, render: (s) => (s.oat_c_median !== null ? `${fmt.num(s.oat_c_median, 0)} °C` : "–") }], rows, pageSize: 10 })));
}
