import * as api from "../api.js";
import { axis, legend, make, tooltip, emptyChart } from "../charts.js";
import { addAirports, addTracks, canvasFallback, createMap, fit, hoverTracks } from "../map.js";
import { acColor, acKey, acSymbol, AIRCRAFT_ORDER, css, fmt, h, icon, isFlight, sum, typeLabel } from "../ui.js";

export async function renderOverview(app) {
  const [sessions, tracks, airports, aircraft, quality] = await Promise.all([
    api.sessions(), api.get("tracks"), api.get("airports"), api.get("aircraft"), api.get("quality"),
  ]);
  app.innerHTML = "";
  const valid = sessions.filter((s) => s.start);
  const nAircraft = new Set(sessions.map((s) => s.aircraft_label)).size;
  const first = valid.length ? new Date(Math.min(...valid.map((s) => s.start))) : null;
  const last = valid.length ? new Date(Math.max(...valid.map((s) => s.start))) : null;
  const latest = valid.reduce((a, s) => (!a || s.start > a.start ? s : a), null);
  const flights = sessions.filter((s) => isFlight(s.session_type));
  const charges = sessions.filter((s) => s.session_type === "charging");
  const hours = sum(sessions.map((s) => s.duration_min)) / 60;
  const homes = [...new Set(aircraft.map((a) => a.home_airfield).filter(Boolean))];
  const homeNames = homes.map((c) => (airports.find((a) => a.code === c) || {}).name || c);

  // ---- hero: moving map + instrument strip
  const mapEl = h("div", { class: "map", role: "img", "aria-label": `Map of ${tracks.length} recorded GPS tracks` });
  const hero = h("section", { class: "hero" },
    mapEl,
    h("div", { class: "hero-copy" },
      h("h1", {}, `Every logged session of ${nAircraft} electric training aircraft`),
      h("p", {}, `${fmt.int(sessions.length)} flights, charges and ground runs of Pipistrel Velis Electro aircraft between ${fmt.date(first)} and ${fmt.date(last)}, `,
        `with battery telemetry for both packs${homeNames.length ? `, flown from ${homeNames.join(" and ")}` : ""}.`),
      h("div", { class: "actions" },
        h("a", { class: "btn primary", href: "#/flights" }, "Explore flights"),
        h("a", { class: "btn", href: "#/dataset" }, icon("download"), "Get the dataset"))),
    h("div", { class: "map-legend", "aria-hidden": "true" },
      h("div", { class: "row" }, h("span", { class: "sw" }), `${fmt.int(tracks.length)} GPS tracks`),
      h("div", { class: "row" }, h("span", { class: "sw ring" }), "Airfield matched from GPS")),
    h("div", { class: "instruments" },
      readout(fmt.int(sessions.length), "Sessions recorded"),
      readout(fmt.int(flights.length), "Flight tests"),
      readout(fmt.int(charges.length), "Charging sessions"),
      readout(fmt.int(hours), "Hours of telemetry", "h"),
      readout(first ? `${first.getUTCFullYear()}–${last.getUTCFullYear()}` : "–", `Collection, ${fmt.int((last - first) / 864e5)} days`),
      readout(latest ? fmt.date(latest.start) : "–", latest ? `Latest: ${typeLabel(latest.session_type).toLowerCase()}, ${latest.aircraft_label}` : "Latest session")));
  app.append(hero);
  drawHeroMap(mapEl, tracks, airports);

  const wrap = h("div", { class: "wrap" });
  app.append(wrap);

  // ---- activity
  const actEl = h("div", { class: "chart short" });
  wrap.append(h("section", { class: "section" },
    h("div", { class: "section-head" }, h("div", {}, h("h2", {}, "When the aircraft were used"),
      h("p", {}, "Sessions per week and aircraft. Gaps are periods with no logged sessions, such as winter storage or maintenance."))),
    h("div", { class: "panel" }, actEl)));
  activity(actEl, valid);

  // ---- SOH small multiples
  const sohGrid = h("div", { class: "grid three" });
  wrap.append(h("section", { class: "section" },
    h("div", { class: "section-head" }, h("div", {}, h("h2", {}, "Battery health as the BMS reports it"),
      h("p", {}, "Median reported state of health per session, both packs. The figure is the battery management system's estimate for its weakest cell: it steps up and down, mostly when the aircraft charges."))),
    sohGrid));
  for (const ac of AIRCRAFT_ORDER) {
    const g = valid.filter((s) => s.aircraft_label === ac && s.b1_soh_median !== null).sort((a, b) => a.start - b.start);
    if (g.length < 3) continue;
    const el = h("div", { class: "chart short" });
    sohGrid.append(h("div", { class: "panel" }, h("h3", {}, h("a", { href: `#/aircraft/${ac}`, style: { color: "inherit" } }, ac)),
      h("p", { class: "sub" }, `${fmt.int(g.length)} sessions, ${fmt.date(g[0].start)} – ${fmt.date(g[g.length - 1].start)}`), el));
    sohSmall(el, g);
  }

  // ---- data completeness + latest sessions
  const grid = h("div", { class: "grid side" });
  wrap.append(h("section", { class: "section" },
    h("div", { class: "section-head" }, h("div", {}, h("h2", {}, "What the data can support"),
      h("p", {}, "Share of each aircraft's sessions that pass the quality checks for each kind of analysis."))),
    grid));
  grid.append(h("div", { class: "panel" }, usabilityTable(quality), h("p", { class: "note" },
    "A session can fail one block and pass another: a charge has no flight profile, and sessions with a logging fault are kept for SOC and SOH but excluded from electrical analyses. ",
    h("a", { href: "#/dataset/quality" }, "How the checks work"))));
  const latestList = h("div", { class: "panel" }, h("h3", {}, "Latest recorded sessions"),
    h("p", { class: "sub" }, "Most recent sessions in the processed dataset, newest first."));
  for (const s of [...valid].sort((a, b) => b.start - a.start).slice(0, 7)) {
    latestList.append(h("a", { href: `#/flights/${s.flight_id}`, style: { display: "grid", gridTemplateColumns: "auto 1fr auto", gap: "10px", alignItems: "center", padding: "9px 0", borderTop: `1px solid ${css("--line")}`, color: "inherit", textDecoration: "none" } },
      icon(isFlight(s.session_type) ? "plane" : s.session_type === "charging" ? "bolt" : "ground"),
      h("span", {}, h("b", {}, typeLabel(s.session_type)), " ", h("span", { class: "muted" }, `· ${s.aircraft_label} · ${fmt.dur(s.duration_min)}`)),
      h("span", { class: "faint num" }, fmt.date(s.start))));
  }
  grid.append(latestList);
}

function readout(v, k, unit) {
  return h("div", { class: "readout" }, h("div", { class: "v" }, v, unit ? h("small", {}, unit) : null), h("div", { class: "k" }, k));
}

function drawHeroMap(el, tracks, airports) {
  if (!tracks.length) { el.append(h("div", { class: "map-empty" }, "No GPS tracks in this dataset: the avionics block was off or positions could not be placed.")); return; }
  const map = createMap(el, { interactive: true });
  if (!map) { canvasFallback(el, tracks, airports, { colorBy: "route" }); return; }
  map.scrollZoom.disable();
  map.on("load", () => {
    addTracks(map, tracks, { colorBy: "route", width: 1.3, opacity: 0.55 });
    addAirports(map, airports);
    const home = airports.slice().sort((a, b) => b.fixes - a.fixes)[0];
    const nearHome = home ? tracks.filter((t) => Math.abs(t.coords[0][1] - home.latitude_deg) < 1 && Math.abs(t.coords[0][0] - home.longitude_deg) < 1.5) : tracks;
    fit(map, (nearHome.length ? nearHome : tracks).map((t) => t.coords), { top: 40, bottom: 120, left: window.innerWidth > 820 ? 620 : 30, right: 40 });
    const pop = new window.maplibregl.Popup({ closeButton: false, closeOnClick: false, offset: 10 });
    hoverTracks(map, "tracks", (p) => { if (!p) pop.remove(); });
    map.on("mousemove", "tracks", (e) => {
      const p = e.features[0].properties;
      pop.setLngLat(e.lngLat).setHTML(`<b>${p.aircraft}</b>, ${fmt.date(p.start)}<br>${p.dep || "?"} to ${p.arr || "?"}, ${fmt.num(p.km, 0)} km<br><span style="opacity:.7">Click to open the session</span>`).addTo(map);
    });
    map.on("click", "tracks", (e) => { location.hash = `#/flights/${e.features[0].properties.flight_id}`; });
  });
}

function activity(el, sessions) {
  if (!sessions.length) { emptyChart(el, "No dated sessions."); return; }
  const wk = (d) => { const x = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate())); x.setUTCDate(x.getUTCDate() - ((x.getUTCDay() + 6) % 7)); return x.getTime(); };
  const t0 = wk(new Date(Math.min(...sessions.map((s) => s.start)))), t1 = wk(new Date(Math.max(...sessions.map((s) => s.start))));
  const weeks = [];
  for (let t = t0; t <= t1; t += 7 * 864e5) weeks.push(t);
  const acs = AIRCRAFT_ORDER.filter((a) => sessions.some((s) => s.aircraft_label === a));
  const counts = new Map();
  for (const s of sessions) { const k = `${wk(s.start)}|${s.aircraft_label}`; counts.set(k, (counts.get(k) || 0) + 1); }
  const data = [];
  let max = 1;
  weeks.forEach((w, i) => acs.forEach((a, j) => { const c = counts.get(`${w}|${a}`) || 0; if (c) { data.push([i, j, c]); max = Math.max(max, c); } }));
  make(el, (t) => ({
    grid: { left: 96, right: 16, top: 8, bottom: 52 },
    tooltip: tooltip(t, { formatter: (p) => `<b>${acs[p.value[1]]}</b><br>Week of ${fmt.date(weeks[p.value[0]])}<br>${p.value[2]} session${p.value[2] > 1 ? "s" : ""}` }),
    xAxis: { type: "category", data: weeks.map((w) => fmt.dateIso(w)), ...axis(t, { splitLine: { show: false }, axisLabel: { color: t.ink3, formatter: (v) => v.slice(0, 7), interval: Math.max(1, Math.floor(weeks.length / 9)) } }) },
    yAxis: { type: "category", data: acs, ...axis(t, { splitLine: { show: false }, axisLabel: { color: t.ink2 } }) },
    visualMap: { min: 1, max, calculable: false, orient: "horizontal", left: 96, bottom: 0, itemHeight: 140, itemWidth: 10,
                 text: [`${max} per week`, "1"], textStyle: { color: t.ink3, fontSize: 12 }, inRange: { color: [t.seq[1], t.seq[3], t.seq[5]] } },
    series: [{ type: "heatmap", data, itemStyle: { borderColor: t.panel, borderWidth: 1, borderRadius: 2 }, emphasis: { itemStyle: { borderColor: t.accent, borderWidth: 1 } } }],
  }));
}

function sohSmall(el, g) {
  make(el, (t) => ({
    grid: { left: 40, right: 10, top: 26, bottom: 26 },
    legend: legend(t, { data: ["Pack 1", "Pack 2"] }),
    tooltip: tooltip(t, { trigger: "axis", formatter: (ps) => `${fmt.date(ps[0].value[0])}<br>` + ps.map((p) => `${p.marker}${p.seriesName}: ${fmt.num(p.value[1], 0)}%`).join("<br>") }),
    xAxis: { type: "time", ...axis(t, { splitLine: { show: false } }) },
    yAxis: { type: "value", min: (v) => Math.max(0, Math.floor(v.min / 10) * 10), max: 100, ...axis(t, { axisLabel: { color: t.ink3, formatter: "{value}%" } }) },
    series: [
      { name: "Pack 1", type: "line", step: "end", showSymbol: false, data: g.map((s) => [s.start, s.b1_soh_median]), lineStyle: { width: 2, color: t.pack1 }, itemStyle: { color: t.pack1 } },
      { name: "Pack 2", type: "line", step: "end", showSymbol: false, data: g.filter((s) => s.b2_soh_median !== null).map((s) => [s.start, s.b2_soh_median]),
        lineStyle: { width: 2, color: t.pack2, type: [5, 3] }, itemStyle: { color: t.pack2 } },
    ],
  }));
}

function usabilityTable(quality) {
  const rows = (quality.tables && quality.tables.usability_by_aircraft_and_analysis) || [];
  if (!rows.length) return h("div", { class: "empty" }, "No quality summary in this export.");
  const blocks = [["B_profiles", "Profile"], ["C_soc_dod", "SOC use"], ["D_charging", "Charging"], ["E_thermal", "Thermal"], ["F_soh", "SOH"], ["G_capacity", "Capacity"]];
  const seq = [css("--seq-1"), css("--seq-2"), css("--seq-3"), css("--seq-4")];
  const table = h("table", { class: "data" }, h("thead", {}, h("tr", {}, h("th", {}, "Aircraft"), h("th", { class: "num" }, "Sessions"), ...blocks.map(([, l]) => h("th", { class: "num" }, l)))));
  const tb = h("tbody");
  for (const r of rows) {
    tb.append(h("tr", { style: { cursor: "default" } }, h("td", {}, acKey(r.aircraft_label)), h("td", { class: "num" }, fmt.int(r.flights)),
      ...blocks.map(([k]) => {
        const n = r[k] || 0, f = r.flights ? n / r.flights : 0;
        return h("td", { class: "num", title: `${n} of ${r.flights} sessions` }, h("span", { style: { display: "inline-block", width: "8px", height: "8px", borderRadius: "2px", marginRight: "6px", background: seq[Math.min(3, Math.floor(f * 4))] } }), fmt.int(n));
      })));
  }
  table.append(tb);
  const t2 = h("div", { class: "table-wrap", style: { border: "0" } }, table);
  t2.append(h("p", { class: "note" }, "Counts of sessions usable for each analysis; the square shades with the share of the aircraft's sessions."));
  return t2;
}
