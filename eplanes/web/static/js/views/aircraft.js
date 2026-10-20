import * as api from "../api.js";
import { axis, legend, make, tooltip, emptyChart } from "../charts.js";
import { acColor, acKey, callout, css, dataTable, fmt, h, isFlight, median, sum, typeLabel, AIRCRAFT_ORDER } from "../ui.js";
import { state } from "../app.js";

function photoFor(label) {
  const site = state.site || {};
  const ph = (site.photos && (site.photos[label] || site.photos.default)) || null;
  return ph;
}

function photoEl(label, cls = "photo") {
  const ph = photoFor(label);
  const box = h("div", { class: cls });
  const silhouette = () => {
    box.innerHTML = "";
    box.append(h("div", { class: "silhouette", "aria-hidden": "true", html: '<svg viewBox="0 0 200 90" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M100 8v70M18 40l82-8 82 8M78 78l22-3 22 3"/><circle cx="100" cy="12" r="4"/></svg>' }));
  };
  if (!ph) { silhouette(); return box; }
  const img = h("img", { src: ph.local || ph.src, alt: ph.alt || "", loading: "lazy" });
  img.addEventListener("error", () => { if (ph.local && img.src.indexOf(ph.local) >= 0 && ph.src) img.src = ph.src; else silhouette(); });
  box.append(img, h("span", { class: "photo-credit" }, ph.credit || ""));
  return box;
}

export async function renderAircraftIndex(app) {
  const aircraft = await api.get("aircraft");
  app.innerHTML = "";
  const wrap = h("div", { class: "wrap" });
  app.append(wrap);
  wrap.append(h("div", { class: "section", style: { paddingBottom: "18px" } }, h("h1", {}, "Aircraft"),
    h("p", { class: "muted" }, "Pipistrel Velis Electro (type Virus SW 128): a two-seat electric trainer with two battery packs of 96 lithium-ion cells in series each. Registrations and serial numbers come from the portal; home airfields from GPS.")));
  const fleet = h("div", { class: "fleet" });
  wrap.append(fleet);
  for (const a of [...aircraft].sort((x, y) => AIRCRAFT_ORDER.indexOf(x.label) - AIRCRAFT_ORDER.indexOf(y.label))) {
    fleet.append(h("a", { class: "aircraft-card", href: `#/aircraft/${encodeURIComponent(a.label)}` }, photoEl(a.label),
      h("div", { class: "body" },
        h("div", { class: "reg" }, acKey(a.label)),
        h("div", { class: "muted", style: { fontSize: "14px" } }, [a.serial_number ? `Serial ${a.serial_number}` : null, a.home_airfield ? `based at ${a.home_airfield}` : null, a.fleet ? `account ${a.fleet.replace("fleet_", "")}` : null].filter(Boolean).join(", ")),
        h("dl", {},
          h("div", {}, h("dt", {}, "Sessions"), h("dd", {}, fmt.int(a.n_sessions))),
          h("div", {}, h("dt", {}, "Flight tests"), h("dd", {}, fmt.int(a.n_flight_tests))),
          h("div", {}, h("dt", {}, "Recorded"), h("dd", {}, `${fmt.date(a.first_utc)} – ${fmt.date(a.last_utc)}`)),
          h("div", {}, h("dt", {}, "Latest reported SOH"), h("dd", {}, a.latest_soh_b1 !== null ? `${fmt.int(a.latest_soh_b1)}% / ${fmt.int(a.latest_soh_b2)}%` : "–"))))));
  }
}

export async function renderAircraft(app, label) {
  const [aircraft, all] = await Promise.all([api.get("aircraft"), api.sessions()]);
  const a = aircraft.find((x) => x.label === label);
  app.innerHTML = "";
  const wrap = h("div", { class: "wrap" });
  app.append(wrap);
  if (!a) { wrap.append(h("div", { class: "empty section" }, `No aircraft called ${label} in this dataset. `, h("a", { href: "#/aircraft" }, "See all aircraft"))); return; }
  const mine = all.filter((s) => s.aircraft_label === label && s.start).sort((x, y) => x.start - y.start);
  const dates = mine.map((s) => s.start);
  const range = { from: dates.length ? fmt.dateIso(dates[0]) : "", to: dates.length ? fmt.dateIso(dates[dates.length - 1]) : "" };

  wrap.append(h("div", { class: "crumbs" }, h("a", { href: "#/aircraft" }, "Aircraft"), h("span", {}, "/"), h("span", {}, label)));
  const head = h("div", { class: "grid side", style: { alignItems: "stretch" } },
    h("div", {}, h("h1", {}, acKey(label)),
      h("p", { class: "muted", style: { marginTop: "8px" } }, `Pipistrel Velis Electro${a.serial_number ? `, serial ${a.serial_number}` : ""}${a.registration && a.registration !== label ? `, registration ${a.registration}` : ""}. `,
        `${fmt.int(a.n_sessions)} sessions from ${fmt.date(a.first_utc)} to ${fmt.date(a.last_utc)}${a.home_airfield ? `, most often departing ${a.home_airfield}` : ""}.`),
      h("div", { class: "instruments", style: { position: "relative", marginTop: "18px", border: "1px solid var(--line)", borderRadius: "8px", gridTemplateColumns: "repeat(3, minmax(0,1fr))" } },
        rd(fmt.int(a.n_flight_tests), "Flight tests"), rd(fmt.int(a.n_charging), "Charging sessions"), rd(`${fmt.int(a.hours_logged)} h`, "Telemetry recorded"))),
    h("div", { class: "panel", style: { padding: 0, overflow: "hidden" } }, photoEl(label, "photo")));
  head.querySelector(".photo").style.height = "100%";
  head.querySelector(".photo").style.minHeight = "220px";
  head.querySelector(".photo").style.position = "relative";
  wrap.append(head);

  const warn = [];
  const sa = mine.filter((s) => s.scale_anomaly).length, dup = mine.filter((s) => s.is_duplicate_content).length, mis = mine.filter((s) => s.label_content_mismatch).length;
  if (sa) warn.push(callout("serious", `${sa} sessions have a current and voltage scaling fault; they are excluded from electrical charts below.`));
  if (dup) warn.push(callout("warning", `${dup} sessions are byte-identical duplicates and are excluded.`));
  if (mis) warn.push(callout("warning", `${mis} sessions carry a portal label that contradicts the measured current.`));
  if (warn.length) wrap.append(h("div", { style: { marginTop: "16px" } }, warn));

  // date range shared by all charts
  const fromIn = h("input", { type: "date", min: range.from, max: range.to, value: range.from });
  const toIn = h("input", { type: "date", min: range.from, max: range.to, value: range.to });
  const charts = h("div");
  wrap.append(h("div", { class: "filters", style: { marginTop: "22px" } },
    h("div", { class: "field" }, "Date range for the charts (UTC)", h("div", { style: { display: "flex", gap: "6px", alignItems: "center" } }, fromIn, h("span", { class: "faint" }, "to"), toIn)),
    h("button", { class: "btn small", type: "button", onclick: () => { fromIn.value = range.from; toIn.value = range.to; draw(); } }, "Full period")), charts);
  fromIn.addEventListener("change", draw);
  toIn.addEventListener("change", draw);

  function draw() {
    const f = Date.parse(fromIn.value + "T00:00:00Z"), t = Date.parse(toIn.value + "T23:59:59Z");
    const sel = mine.filter((s) => s.start >= f && s.start <= t && !s.is_duplicate_content);
    charts.innerHTML = "";
    const g1 = h("div", { class: "grid two", style: { marginTop: "20px" } });
    const g2 = h("div", { class: "grid two", style: { marginTop: "20px" } });
    const g3 = h("div", { class: "grid two", style: { marginTop: "20px" } });
    charts.append(g1, g2, g3);
    const p = (title, sub, cls = "chart") => { const el = h("div", { class: cls }); return [h("div", { class: "panel" }, h("h3", {}, title), h("p", { class: "sub" }, sub), el), el]; };

    const [pa, ea] = p("Reported state of health", "Median per session. Dots: mean of the BMS's 96 per-cell SOH values. The reported figure follows the weakest cell.");
    const [pb, eb] = p("Sessions per month", "Stacked by portal session type.");
    const [pc, ec] = p("Depth of discharge per flight", "Pack 1 SOC at start minus SOC at end, flight tests only.");
    const [pd, ed] = p("Charging sessions", "Duration against SOC gained, pack 1.");
    const [pe, ee] = p("Cell temperature against outside air", "Maximum cell temperature per session, sessions with live avionics. Dotted line: equal temperatures.");
    const [pf, ef] = p("Energy per flight", "Pack 1 + pack 2 energy out per flight test (scale-fault sessions excluded).");
    g1.append(pa, pb); g2.append(pc, pd); g3.append(pe, pf);
    soh(ea, sel); monthly(eb, sel); dod(ec, sel); charging(ed, sel); thermal(ee, sel); energy(ef, sel);

    const cov = signalCoverage(sel);
    charts.append(h("div", { class: "panel", style: { marginTop: "20px" } }, h("h3", {}, "Signal availability"),
      h("p", { class: "sub" }, "Share of the selected sessions in which each quantity could be computed. Missing values are never treated as zero."), cov));
  }
  draw();
}

function rd(v, k) { return h("div", { class: "readout" }, h("div", { class: "v" }, v), h("div", { class: "k" }, k)); }

function soh(el, sel) {
  const g = sel.filter((s) => s.b1_soh_median !== null);
  if (g.length < 2) { emptyChart(el, "Too few sessions with reported SOH in this range."); return; }
  make(el, (t) => ({
    grid: { left: 44, right: 12, top: 30, bottom: 50 },
    legend: legend(t, { data: ["Pack 1", "Pack 2", "Mean cell, pack 1", "Mean cell, pack 2"] }),
    tooltip: tooltip(t, { trigger: "axis", formatter: (ps) => `${fmt.date(ps[0].value[0])}<br>` + ps.map((p) => `${p.marker}${p.seriesName}: ${fmt.num(p.value[1], 1)}%`).join("<br>") }),
    dataZoom: [{ type: "inside" }, { type: "slider", height: 16, bottom: 6, borderColor: t.line, textStyle: { color: t.ink3 }, fillerColor: "rgba(70,200,232,0.12)" }],
    xAxis: { type: "time", ...axis(t, { splitLine: { show: false } }) },
    yAxis: { type: "value", max: 100, min: (v) => Math.max(0, Math.floor(v.min / 10) * 10), ...axis(t, { axisLabel: { color: t.ink3, formatter: "{value}%" } }) },
    series: [
      { name: "Pack 1", type: "line", step: "end", showSymbol: false, data: g.map((s) => [s.start, s.b1_soh_median]), lineStyle: { color: t.pack1, width: 2 }, itemStyle: { color: t.pack1 } },
      { name: "Pack 2", type: "line", step: "end", showSymbol: false, data: g.filter((s) => s.b2_soh_median !== null).map((s) => [s.start, s.b2_soh_median]), lineStyle: { color: t.pack2, width: 2, type: [5, 3] }, itemStyle: { color: t.pack2 } },
      { name: "Mean cell, pack 1", type: "scatter", symbolSize: 4, data: g.filter((s) => s.b1_soh_cell_cells_last_mean !== null && s.b1_soh_cell_cells_last_mean !== undefined).map((s) => [s.start, s.b1_soh_cell_cells_last_mean]), itemStyle: { color: t.pack1, opacity: 0.5 } },
      { name: "Mean cell, pack 2", type: "scatter", symbol: "rect", symbolSize: 4, data: g.filter((s) => s.b2_soh_cell_cells_last_mean !== null && s.b2_soh_cell_cells_last_mean !== undefined).map((s) => [s.start, s.b2_soh_cell_cells_last_mean]), itemStyle: { color: t.pack2, opacity: 0.5 } },
    ],
  }));
}

function monthly(el, sel) {
  if (!sel.length) { emptyChart(el, "No sessions in this range."); return; }
  const key = (d) => `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, "0")}`;
  const first = sel[0].start, last = sel[sel.length - 1].start;
  const months = [];
  for (let y = first.getUTCFullYear(), m = first.getUTCMonth(); y < last.getUTCFullYear() || (y === last.getUTCFullYear() && m <= last.getUTCMonth()); m === 11 ? (y++, m = 0) : m++) months.push(`${y}-${String(m + 1).padStart(2, "0")}`);
  const types = [["flight", "Flights", (s) => isFlight(s.session_type)], ["charging", "Charges", (s) => s.session_type === "charging"], ["other", "Other", (s) => !isFlight(s.session_type) && s.session_type !== "charging"]];
  make(el, (t) => ({
    grid: { left: 36, right: 10, top: 30, bottom: 30 },
    legend: legend(t),
    tooltip: tooltip(t, { trigger: "axis", axisPointer: { type: "shadow" } }),
    xAxis: { type: "category", data: months, ...axis(t, { splitLine: { show: false }, axisLabel: { color: t.ink3, interval: Math.max(0, Math.floor(months.length / 8) - 1) } }) },
    yAxis: { type: "value", minInterval: 1, ...axis(t) },
    series: types.map(([k, name, pred], i) => ({ name, type: "bar", stack: "a", barCategoryGap: "30%", itemStyle: { color: [t.s[0], t.s[2], t.s[3]][i], borderColor: t.panel, borderWidth: 1 },
      data: months.map((m) => sel.filter((s) => key(s.start) === m && pred(s)).length) })),
  }));
}

function dod(el, sel) {
  const g = sel.filter((s) => isFlight(s.session_type) && s.b1_soc_start !== null && s.b1_soc_end !== null);
  if (!g.length) { emptyChart(el, "No flight tests with SOC in this range."); return; }
  const med = median(g.map((s) => s.b1_soc_start - s.b1_soc_end));
  make(el, (t) => ({
    grid: { left: 48, right: 16, top: 34, bottom: 30 },
    tooltip: tooltip(t, { trigger: "item", formatter: (p) => `Session ${p.value[2]}<br>${fmt.date(p.value[0])}<br>DoD ${fmt.int(p.value[1])} points` }),
    xAxis: { type: "time", ...axis(t, { splitLine: { show: false } }) },
    yAxis: { type: "value", name: "SOC points", min: 0, max: 100, ...axis(t) },
    series: [{ type: "scatter", symbolSize: 6, data: g.map((s) => [s.start, s.b1_soc_start - s.b1_soc_end, s.flight_id]), itemStyle: { color: t.pack1, opacity: 0.75 },
      markLine: { silent: true, symbol: "none", lineStyle: { color: t.ink2, type: "dashed" }, label: { color: t.ink2, position: "insideEndTop", formatter: `median ${fmt.int(med)}` }, data: [{ yAxis: med }] } }],
  }));
}

function charging(el, sel) {
  const g = sel.filter((s) => s.session_type === "charging" && s.b1_soc_start !== null && s.b1_soc_end !== null && !s.scale_anomaly);
  if (!g.length) { emptyChart(el, "No charging sessions in this range."); return; }
  make(el, (t) => ({
    grid: { left: 52, right: 16, top: 34, bottom: 44 },
    tooltip: tooltip(t, { trigger: "item", formatter: (p) => `Session ${p.value[2]}<br>${fmt.dur(p.value[0])}, +${fmt.int(p.value[1])} SOC points` }),
    xAxis: { type: "value", name: "minutes", nameLocation: "middle", nameGap: 26, ...axis(t, { splitLine: { show: false } }) },
    yAxis: { type: "value", name: "SOC gained", ...axis(t) },
    series: [{ type: "scatter", symbolSize: 6, data: g.map((s) => [s.duration_min, s.b1_soc_end - s.b1_soc_start, s.flight_id]), itemStyle: { color: t.s[2], opacity: 0.8 } }],
  }));
}

function thermal(el, sel) {
  const g = sel.filter((s) => s.oat_c_median !== null && s.b1_temp_max_c !== null);
  if (!g.length) { emptyChart(el, "No sessions with both outside-air and cell temperature in this range."); return; }
  const lo = Math.floor(Math.min(...g.map((s) => Math.min(s.oat_c_median, s.b1_temp_max_c))) / 5) * 5, hi = Math.ceil(Math.max(...g.map((s) => Math.max(s.oat_c_median, s.b1_temp_max_c))) / 5) * 5;
  make(el, (t) => ({
    grid: { left: 52, right: 16, top: 34, bottom: 44 },
    tooltip: tooltip(t, { trigger: "item", formatter: (p) => (p.seriesType === "scatter" ? `Session ${p.value[2]}<br>Outside ${fmt.num(p.value[0], 0)} °C, cells ${fmt.int(p.value[1])} °C` : "") }),
    xAxis: { type: "value", name: "outside air °C", nameLocation: "middle", nameGap: 26, min: lo, max: hi, ...axis(t, { splitLine: { show: false } }) },
    yAxis: { type: "value", name: "max cell °C", min: lo, max: hi, ...axis(t) },
    series: [{ type: "line", data: [[lo, lo], [hi, hi]], showSymbol: false, lineStyle: { color: t.ink3, type: "dotted", width: 1 }, silent: true },
      { type: "scatter", symbolSize: 6, data: g.map((s) => [s.oat_c_median, s.b1_temp_max_c, s.flight_id]), itemStyle: { color: t.pack1, opacity: 0.75 } }],
  }));
}

function energy(el, sel) {
  const g = sel.filter((s) => isFlight(s.session_type) && !s.scale_anomaly && s.b1_energy_discharged_kwh !== null);
  if (!g.length) { emptyChart(el, "No flight tests with energy in this range."); return; }
  make(el, (t) => ({
    grid: { left: 48, right: 16, top: 34, bottom: 30 },
    tooltip: tooltip(t, { trigger: "item", formatter: (p) => `Session ${p.value[2]}<br>${fmt.date(p.value[0])}<br>${fmt.num(p.value[1], 2)} kWh` }),
    xAxis: { type: "time", ...axis(t, { splitLine: { show: false } }) },
    yAxis: { type: "value", name: "kWh", ...axis(t) },
    series: [{ type: "scatter", symbolSize: 6, data: g.map((s) => [s.start, (s.b1_energy_discharged_kwh || 0) + (s.b2_energy_discharged_kwh || 0), s.flight_id]), itemStyle: { color: t.s[0], opacity: 0.75 } }],
  }));
}

function signalCoverage(sel) {
  const items = [
    ["Reported SOC, pack 1 / 2", (s) => s.b1_soc_start !== null, (s) => s.b2_soc_start !== null],
    ["Reported SOH, pack 1 / 2", (s) => s.b1_soh_median !== null, (s) => s.b2_soh_median !== null],
    ["Charge integrated from current", (s) => (s.b1_ah_discharged || 0) + (s.b1_ah_charged || 0) > 0 && !s.scale_anomaly, (s) => (s.b2_ah_discharged || 0) + (s.b2_ah_charged || 0) > 0 && !s.scale_anomaly],
    ["Maximum cell temperature", (s) => s.b1_temp_max_c !== null, (s) => s.b2_temp_max_c !== null],
    ["Per-cell SOH from the BMS files", (s) => s.b1_soh_cell_cells_last_mean !== null && s.b1_soh_cell_cells_last_mean !== undefined, (s) => s.b2_soh_cell_cells_last_mean !== null && s.b2_soh_cell_cells_last_mean !== undefined],
    ["Outside air temperature (avionics)", (s) => s.oat_c_median !== null, null],
    ["GPS track (avionics)", (s) => !!s.has_gps_track, null],
  ];
  const n = sel.length || 1;
  const rows = items.map(([label, f1, f2]) => ({ label, p1: sel.filter(f1).length / n, p2: f2 ? sel.filter(f2).length / n : null }));
  const bar = (v) => (v === null ? h("span", { class: "faint" }, "n/a") : h("span", { style: { display: "inline-flex", alignItems: "center", gap: "8px" } },
    h("span", { style: { width: "90px", height: "6px", background: css("--panel-3"), borderRadius: "3px", overflow: "hidden", display: "inline-block" } },
      h("span", { style: { display: "block", height: "100%", width: `${Math.round(v * 100)}%`, background: css("--good") } })), fmt.pct(v)));
  return dataTable({ columns: [{ key: "label", label: "Quantity", sortable: false }, { key: "p1", label: "Pack 1 (or aircraft)", render: (r) => bar(r.p1), sortable: false }, { key: "p2", label: "Pack 2", render: (r) => bar(r.p2), sortable: false }], rows, pageSize: 20 });
}
